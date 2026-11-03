"""Reference implementation of GCA token-level advantages (arXiv 2610.23148, Sec. 4).

Pure numpy, CPU only. This is the "ground truth" that the verl integration will be
tested against later. Equation numbers refer to the paper.

Pipeline for one group of n rollouts:
  1. classify each rollout by (H, C, timeout, Retr) into A / B1 / B2 / C / D / D0   (Eq. 1, Table 1)
  2. every trainable token belongs to a segment: VSEARCH / RSEARCH / ANS              (Sec. 4.1)
  3. routing table gives signed coefficient P*lambda for (type, segment)             (Table 1 / 9)
  4. salience c in [0, c_max] from old-policy margin + entropy                        (Eq. 5-6, App. E)
  5. A_token = A_hat + gate * eta * |A_hat| * P*lambda * c                            (Eq. 4)

Bound (Prop. 2): |A_token / A_hat - 1| <= kappa = eta * lambda_max * c_max = 0.2 * 1 * 0.5 = 0.1
"""

from __future__ import annotations

import numpy as np

# ---- segments (per token) ----
NONE, VSEARCH, RSEARCH, ANS = -1, 0, 1, 2  # NONE = tool-response / prompt tokens (not trained)

# ---- rollout types ----
A, B1, B2, C, D, D0 = "A", "B1", "B2", "C", "D", "D0"  # D0 = D with Retr=0 (corpus lacks the answer)

# Table 1: signed residual coefficient P*lambda for (type, segment).
# None  -> BASE (no residual, keep A_hat);  MASK -> rollout removed from the policy loss.
MASK = "MASK"
ROUTING = {
    A:  {VSEARCH: +0.5, RSEARCH: -0.1, ANS: +1.0},
    B1: {VSEARCH: +0.5, RSEARCH: -0.2, ANS: -1.0},
    B2: {VSEARCH: +0.5, RSEARCH: -0.1, ANS: 0.0},
    C:  {VSEARCH: None, RSEARCH: None, ANS: None},   # BASE: no conflict to correct
    D:  {VSEARCH: None, RSEARCH: -0.2, ANS: -0.1},   # VSEARCH absent when H=0
    D0: MASK,
}

ETA, C_MAX, LAMBDA_MAX = 0.2, 0.5, 1.0
KAPPA = ETA * LAMBDA_MAX * C_MAX  # 0.1
ALPHA, BETA, GAMMA, TAU = 0.6, 0.3, 0.1, 0.5


def classify(hit: bool, correct: bool, timeout: bool = False, retr: bool = True) -> str:
    """(H, C) -> rollout type. timeout = turn budget exhausted without an answer."""
    if hit:
        if correct:
            return A
        return B2 if timeout else B1
    if not correct:
        return C
    return D if retr else D0


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def margin(logp_sampled: np.ndarray, logp_competitors: np.ndarray) -> np.ndarray:
    """Eq. 5/8-9. delta_t = l_t - log( mean_{v in C_t} pi(v) ), C_t = K=3 top tokens other than y_t.

    logp_sampled: (T,)   logp_competitors: (T, K)
    """
    comp = np.log(np.mean(np.exp(logp_competitors), axis=-1))
    return logp_sampled - comp


def norm_entropy(logp_topm: np.ndarray) -> np.ndarray:
    """Eq. 10. Entropy of the renormalised top-M distribution, divided by log M. (T, M) -> (T,) in [0,1]."""
    p = np.exp(logp_topm)
    q = p / p.sum(axis=-1, keepdims=True)
    M = logp_topm.shape[-1]
    return -(q * np.log(np.clip(q, 1e-30, None))).sum(axis=-1) / np.log(M)


def salience(delta: np.ndarray, H_norm: np.ndarray, seg: np.ndarray, polarity: np.ndarray) -> np.ndarray:
    """Eq. 6 / 11-15 for ONE rollout.

    delta, H_norm, seg, polarity: (T,). polarity is +1 / -1 / 0 per token (sign of P*lambda).
    Margin is centred by the median within each segment of this rollout.
    Returns c in [0, C_MAX].
    """
    c = np.zeros_like(delta, dtype=float)
    for s in (VSEARCH, RSEARCH, ANS):
        idx = seg == s
        if not idx.any():
            continue
        dbar = delta[idx] - np.median(delta[idx])
        d_plus, d_minus = _sigmoid(-dbar / TAU), _sigmoid(dbar / TAU)
        h, conf = H_norm[idx], 1.0 - H_norm[idx]
        c_plus = ALPHA * d_plus + BETA * h + GAMMA * d_plus * h        # uncertain-but-useful
        c_minus = ALPHA * d_minus + BETA * conf + GAMMA * d_minus * conf  # confident-but-harmful
        pol = polarity[idx]
        c[idx] = np.where(pol > 0, c_plus, np.where(pol < 0, c_minus, 0.0))
    return np.clip(c, 0.0, C_MAX)


def coeff_matrix(types: list[str], seg: np.ndarray) -> np.ndarray:
    """(n,T) signed P*lambda per token; 0 where BASE / not trainable / masked."""
    out = np.zeros(seg.shape, dtype=float)
    for i, t in enumerate(types):
        if ROUTING[t] == MASK:
            continue
        for s, v in ROUTING[t].items():
            if v is not None:
                out[i][seg[i] == s] = v
    return out


def gca_token_advantage(
    a_hat: np.ndarray,          # (n,)  GRPO group-relative advantage per rollout
    types: list[str],           # (n,)  from classify()
    seg: np.ndarray,            # (n,T) segment id per token (NONE for non-trainable)
    c: np.ndarray,              # (n,T) salience in [0, C_MAX]
    is_struct_tag: np.ndarray | None = None,  # (n,T) bool, e.g. <tool_call>, </answer>
    eta: float = ETA,
) -> tuple[np.ndarray, np.ndarray]:
    """Eq. 4. Returns (token advantages (n,T), loss mask (n,T))."""
    n, T = seg.shape
    a_hat = np.asarray(a_hat, dtype=float)
    base = np.repeat(a_hat[:, None], T, axis=1)

    trainable = seg != NONE
    rollout_mask = np.array([ROUTING[t] != MASK for t in types])
    loss_mask = trainable & rollout_mask[:, None]

    if np.allclose(a_hat, 0.0):  # zero reward variance in the group -> gate closed everywhere
        return base, loss_mask

    pl = coeff_matrix(types, seg)
    gate = (pl != 0) & trainable & np.array([t != C for t in types])[:, None]
    if is_struct_tag is not None:
        gate &= ~is_struct_tag

    resid = eta * np.abs(a_hat)[:, None] * pl * np.clip(c, 0.0, C_MAX)
    return base + gate * resid, loss_mask
