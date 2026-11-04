"""Register the "gca" advantage estimator in verl.

Imported by the patched verl (patches/verl_gca.patch + verl_gca_v1.patch) when algorithm.adv_estimator=gca.
Needs gca/reward.py as the custom reward. The V1 trainer (default, same code path as the GRPO baseline) keeps the
reward extra info per sample in the TransferQueue field "extra_fields"["reward_extra_info"]; the V1 patch reads it
with gca_fields_from_extra() so non_tensor_batch carries gca_first_hit / gca_answered / gca_type.

The math lives in numpy (gca.advantage, gca.segments) and is unit-tested on CPU;
this file only converts tensors and computes the GRPO baseline the same way verl does.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from gca.advantage import C_MAX, ETA, gca_token_advantage
from gca.segments import batch_segment_ids


GCA_KEYS = ("gca_type", "gca_first_hit", "gca_answered")


def gca_fields_from_extra(extra_fields) -> dict[str, np.ndarray]:
    """V1 trainer: per-sample extra_fields dicts (from the TransferQueue) -> non_tensor_batch arrays of GCA labels.

    extra_fields may be a NonTensorStack, a numpy object array or a list; each item is a dict holding
    "reward_extra_info" (what gca/reward.py returned besides "score"). Fails loudly if a label is missing,
    because silently defaulting would train plain GRPO under the GCA name.
    """
    items = extra_fields.tolist() if hasattr(extra_fields, "tolist") else list(extra_fields)
    out = {k: np.empty(len(items), dtype=object) for k in GCA_KEYS}
    for i, ef in enumerate(items):
        info = ef.get("reward_extra_info", {}) if isinstance(ef, dict) else {}
        missing = [k for k in GCA_KEYS if k not in info]
        if missing:
            raise KeyError(f"sample {i}: reward_extra_info lacks {missing} (is gca/reward.py the custom reward?); "
                           f"has {sorted(info)}")
        for k in GCA_KEYS:
            v = info[k]
            if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 1:
                v = v[0]
            out[k][i] = v
    return out


def grpo_scalar_advantage(scores: np.ndarray, index, eps: float = 1e-6, norm_by_std: bool = True) -> np.ndarray:
    """Same as verl compute_grpo_outcome_advantage, but one scalar per rollout.
    Group of size 1 -> mean 0, std 1. std is unbiased (torch.std default)."""
    groups = defaultdict(list)
    for i, u in enumerate(index):
        groups[u].append(i)
    adv = np.zeros(len(scores), dtype=float)
    for idxs in groups.values():
        s = scores[idxs]
        if len(idxs) == 1:
            mean, std = 0.0, 1.0
        else:
            mean, std = s.mean(), s.std(ddof=1)
        adv[idxs] = (s - mean) / (std + eps) if norm_by_std else (s - mean)
    return adv


def gca_numpy(scores, response_mask, index, first_hit, answered, types, c=None, eta=ETA, norm_by_std=True):
    """Everything except tensor conversion. Returns (token advantages (n,T), loss mask (n,T))."""
    response_mask = np.asarray(response_mask)
    a_hat = grpo_scalar_advantage(np.asarray(scores, dtype=float), index, norm_by_std=norm_by_std)
    fh = [None if int(k) < 0 else int(k) for k in first_hit]
    seg = batch_segment_ids(response_mask, fh, [bool(a) for a in answered])
    if c is None:
        # TODO: margin-entropy salience needs top-k log-probs from the multi-turn rollout (not yet exposed).
        # Until then use a constant salience = C_MAX, i.e. the upper bound of the residual (kappa = 0.1).
        c = np.full(seg.shape, C_MAX)
    adv, loss_mask = gca_token_advantage(a_hat, list(types), seg, c, eta=eta)
    return adv * response_mask, loss_mask


def _register():
    import torch
    from verl.trainer.ppo.core_algos import register_adv_est

    @register_adv_est("gca")
    def compute_gca_advantage(token_level_rewards, response_mask, index=None, config=None,
                              non_tensor_batch=None, batch=None, **kwargs):
        assert non_tensor_batch is not None and "gca_type" in non_tensor_batch, (
            "GCA needs gca/reward.py as custom reward and both verl patches (bash scripts/apply_verl_patch.sh)"
        )
        def _cfg(key, default):
            try:
                v = config.get(key, default) if config is not None else default
            except Exception:
                v = getattr(config, key, default)
            return default if v is None else v

        eta = float(_cfg("gca_eta", ETA))
        norm = bool(_cfg("norm_adv_by_std_in_grpo", True))
        scores = token_level_rewards.sum(dim=-1).float().cpu().numpy()
        mask = response_mask.cpu().numpy()
        adv, loss_mask = gca_numpy(
            scores, mask, index,
            non_tensor_batch["gca_first_hit"], non_tensor_batch["gca_answered"], non_tensor_batch["gca_type"],
            eta=eta, norm_by_std=norm,
        )
        # D0 rollouts (Retr=0) are masked out of the loss: zero their advantage (they still counted in group stats)
        adv = adv * loss_mask
        out = torch.tensor(adv, dtype=token_level_rewards.dtype, device=token_level_rewards.device)
        return out, out

    return compute_gca_advantage


try:
    compute_gca_advantage = _register()
except ImportError:  # no torch/verl (CPU unit tests): numpy functions above still usable
    compute_gca_advantage = None
