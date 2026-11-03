"""Exit checks for GCA (README phase 1): eta=0 == GRPO, hand-computed A/B/C/D cases, sign bound.

All numbers in the hand-computed tests are worked out in the comments so they can be checked on paper.
"""

import numpy as np
import pytest

from gca.advantage import (
    A, ANS, B1, B2, C, C_MAX, D, D0, KAPPA, NONE, RSEARCH, VSEARCH,
    classify, gca_token_advantage, margin, norm_entropy, salience,
)

# One toy layout shared by every rollout: [VS, VS, tool, RS, ANS, ANS]
LAYOUT = [VSEARCH, VSEARCH, NONE, RSEARCH, ANS, ANS]


def seg_for(types):
    """H=0 rollouts have no VSEARCH: their searches count as RSEARCH."""
    rows = []
    for t in types:
        row = [RSEARCH if (s == VSEARCH and t in (C, D, D0)) else s for s in LAYOUT]
        rows.append(row)
    return np.array(rows)


# ---------------- classification ----------------

@pytest.mark.parametrize("h,c,timeout,retr,expected", [
    (1, 1, 0, 1, A), (1, 0, 0, 1, B1), (1, 0, 1, 1, B2),
    (0, 0, 0, 1, C), (0, 1, 0, 1, D), (0, 1, 0, 0, D0),
])
def test_classify(h, c, timeout, retr, expected):
    assert classify(bool(h), bool(c), bool(timeout), bool(retr)) == expected


# ---------------- exit check 1: eta = 0 reproduces GRPO ----------------

def test_eta_zero_equals_grpo():
    rng = np.random.default_rng(0)
    types = [A, B1, B2, C, D]
    a_hat = rng.normal(size=5)
    seg = seg_for(types)
    c = rng.uniform(0, C_MAX, size=seg.shape)
    adv, _ = gca_token_advantage(a_hat, types, seg, c, eta=0.0)
    np.testing.assert_allclose(adv, np.repeat(a_hat[:, None], seg.shape[1], axis=1))


# ---------------- exit check 2: hand-computed cases ----------------

def test_hand_computed_A_B1_C_D():
    # rewards [1,0,0,1] -> a_hat = [+1,-1,-1,+1] (given directly), salience c = 0.5 everywhere
    # residual = eta*|a|*Plambda*c = 0.2*1*Plambda*0.5 = 0.1*Plambda
    types = [A, B1, C, D]
    a_hat = np.array([1.0, -1.0, -1.0, 1.0])
    seg = seg_for(types)
    c = np.full(seg.shape, 0.5)
    adv, mask = gca_token_advantage(a_hat, types, seg, c)

    #                 VS     VS    tool   RS     ANS    ANS
    expected = np.array([
        [1.05,  1.05,  1.0,  0.99,  1.10,  1.10],   # A : +0.5, -0.1, +1.0
        [-0.95, -0.95, -1.0, -1.02, -1.10, -1.10],  # B1: +0.5, -0.2, -1.0
        [-1.0,  -1.0,  -1.0, -1.0,  -1.0,  -1.0],   # C : BASE
        [0.98,  0.98,  1.0,  0.98,  0.99,  0.99],   # D : searches are RS (-0.2), ANS -0.1
    ])
    np.testing.assert_allclose(adv, expected)
    assert not mask[:, 2].any()  # tool-response tokens never trained


def test_residual_scales_with_abs_advantage():
    # |a|=2 doubles the residual: A·ANS = 2 + 0.2*2*1.0*0.5 = 2.2 ; B1·ANS = -2 + 0.2*2*(-1.0)*0.5 = -2.2
    types = [A, B1]
    adv, _ = gca_token_advantage(np.array([2.0, -2.0]), types, seg_for(types), np.full((2, 6), 0.5))
    assert adv[0, 4] == pytest.approx(2.2) and adv[1, 4] == pytest.approx(-2.2)


def test_B2_answer_gets_no_residual():
    types = [B2, A]
    adv, _ = gca_token_advantage(np.array([-1.0, 1.0]), types, seg_for(types), np.full((2, 6), 0.5))
    assert adv[0, 4] == pytest.approx(-1.0)   # lambda = 0 on B2·ANS
    assert adv[0, 0] == pytest.approx(-0.95)  # but its valid search is still protected


def test_D_without_retrievable_answer_is_masked():
    types = [D0, C]
    adv, mask = gca_token_advantage(np.array([1.0, -1.0]), types, seg_for(types), np.full((2, 6), 0.5))
    assert not mask[0].any()                       # out of the policy loss
    np.testing.assert_allclose(adv[0], 1.0)       # (still counted in group stats upstream)


def test_zero_variance_group_gets_nothing():
    types = [A, A, A]
    adv, _ = gca_token_advantage(np.zeros(3), types, seg_for(types), np.full((3, 6), 0.5))
    assert np.all(adv == 0.0)


def test_structural_tags_keep_base():
    types = [A]
    tags = np.zeros((1, 6), dtype=bool)
    tags[0, 4] = True  # e.g. "<answer>"
    adv, _ = gca_token_advantage(np.array([1.0]), types, seg_for(types), np.full((1, 6), 0.5), tags)
    assert adv[0, 4] == 1.0 and adv[0, 5] == pytest.approx(1.1)


# ---------------- Proposition 2: sign preserved, bounded by kappa ----------------

def test_sign_preservation_random():
    rng = np.random.default_rng(1)
    for _ in range(200):
        types = list(rng.choice([A, B1, B2, C, D], size=5))
        a_hat = rng.normal(size=5)
        seg = seg_for(types)
        c = rng.uniform(0, 1.0, size=seg.shape)  # even above the cap: function must clamp
        adv, _ = gca_token_advantage(a_hat, types, seg, c)
        ratio = adv / a_hat[:, None]
        assert np.all(ratio > 0)
        assert np.all(np.abs(ratio - 1) <= KAPPA + 1e-12)


# ---------------- salience ----------------

def test_margin_hand_computed():
    # sampled p=0.5, competitors 0.2, 0.2, 0.1 -> mean 0.1667 -> delta = ln(0.5/0.16667) = ln 3
    d = margin(np.log([0.5]), np.log([[0.2, 0.2, 0.1]]))
    assert d[0] == pytest.approx(np.log(3))


def test_entropy_bounds():
    uniform = np.log(np.full((1, 32), 1 / 32))
    peaked = np.log(np.array([[0.99] + [0.01 / 31] * 31]))
    assert norm_entropy(uniform)[0] == pytest.approx(1.0)
    assert norm_entropy(peaked)[0] < 0.1


def test_salience_picks_the_right_tokens():
    # two tokens in one ANS segment: token0 = uncertain (low margin, high entropy), token1 = confident
    delta = np.array([-1.0, 2.0])
    H = np.array([0.9, 0.1])
    seg = np.array([ANS, ANS])
    c_pos = salience(delta, H, seg, np.array([1, 1]))
    c_neg = salience(delta, H, seg, np.array([-1, -1]))
    assert c_pos[0] > c_pos[1]   # positive cells reward uncertain-but-useful tokens
    assert c_neg[1] > c_neg[0]   # negative cells hit confident-but-harmful tokens
    assert np.all(c_pos <= C_MAX) and np.all(c_neg <= C_MAX)
