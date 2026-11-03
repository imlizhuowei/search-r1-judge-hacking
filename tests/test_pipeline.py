"""End-to-end on CPU: decoded rollout text -> reward dict -> segments -> GRPO baseline -> GCA token advantages."""

import numpy as np
import pytest

from gca.reward import compute_score, observations_from_text
from gca.verl_plugin import gca_numpy, grpo_scalar_advantage

GT = {"target": ["Thomas Jefferson"]}
HIT = "<tool_response>\nDoc 1 (Title: Declaration)\nDrafted by Thomas Jefferson.\n</tool_response>"
MISS = "<tool_response>\nDoc 1 (Title: Paris)\nCapital of France.\n</tool_response>"
CALL = '<tool_call>{"name": "search", "arguments": {"query_list": ["q"]}}</tool_call>'


def text(*parts):
    return "\n".join(parts)


ROLLOUTS = {
    "A":  text(CALL, HIT, "<answer>Thomas Jefferson</answer>"),
    "B1": text(CALL, HIT, "<answer>John Adams</answer>"),
    "C":  text(CALL, MISS, "<answer>John Adams</answer>"),
    "D":  text(CALL, MISS, "<answer>Thomas Jefferson</answer>"),
}


def test_observations_parsed_in_order():
    assert len(observations_from_text(text(CALL, MISS, CALL, HIT, "<answer>x</answer>"))) == 2


@pytest.mark.parametrize("name", list(ROLLOUTS))
def test_reward_labels(name):
    r = compute_score("searchR1_nq", ROLLOUTS[name], GT)
    assert r["gca_type"] == name
    assert r["score"] == (1.0 if name in ("A", "D") else 0.0)
    assert r["gca_first_hit"] == (1 if name in ("A", "B1") else -1)


def test_score_matches_verl_em_rule():
    spam = "<answer>Thomas Jefferson</answer>" * 11
    assert compute_score("searchR1_nq", spam, GT)["score"] == 0.25


def test_grpo_baseline_matches_hand_calc():
    # group [1,0,0,1]: mean .5, unbiased std = sqrt(1/3) = .57735 -> adv = +-0.8660
    adv = grpo_scalar_advantage(np.array([1.0, 0, 0, 1]), ["u"] * 4)
    np.testing.assert_allclose(adv, [0.866025, -0.866025, -0.866025, 0.866025], atol=1e-5)
    assert grpo_scalar_advantage(np.array([1.0]), ["solo"])[0] == pytest.approx(1.0, abs=1e-5)


def _batch():
    labels = [compute_score("searchR1_nq", ROLLOUTS[k], GT) for k in ["A", "B1", "C", "D"]]
    # token layout per rollout: [search x2][tool x2][answer x2][pad x2]
    mask = np.tile([1, 1, 0, 0, 1, 1, 0, 0], (4, 1))
    return (
        np.array([l["score"] for l in labels]), mask, ["u"] * 4,
        [l["gca_first_hit"] for l in labels], [l["gca_answered"] for l in labels], [l["gca_type"] for l in labels],
    )


def test_eta_zero_is_grpo():
    scores, mask, idx, fh, ans, types = _batch()
    adv, _ = gca_numpy(scores, mask, idx, fh, ans, types, eta=0.0)
    a = grpo_scalar_advantage(scores, idx)
    np.testing.assert_allclose(adv, a[:, None] * mask)


def test_full_pipeline_hand_calc():
    scores, mask, idx, fh, ans, types = _batch()
    adv, _ = gca_numpy(scores, mask, idx, fh, ans, types)  # constant salience c = 0.5
    a = 0.8660254
    r = 0.2 * a * 0.5  # eta*|A|*c
    expected = np.array([
        [a + r * 0.5, a + r * 0.5, 0, 0, a + r * 1.0, a + r * 1.0, 0, 0],      # A : VS +0.5, ANS +1.0
        [-a + r * 0.5, -a + r * 0.5, 0, 0, -a - r * 1.0, -a - r * 1.0, 0, 0],  # B1: VS +0.5, ANS -1.0
        [-a, -a, 0, 0, -a, -a, 0, 0],                                          # C : base
        [a - r * 0.2, a - r * 0.2, 0, 0, a - r * 0.1, a - r * 0.1, 0, 0],      # D : RS -0.2, ANS -0.1
    ])
    np.testing.assert_allclose(adv, expected, atol=1e-5)  # hand calc ignores GRPO's eps=1e-6 in the std
