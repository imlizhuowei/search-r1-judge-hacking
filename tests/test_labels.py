import pytest

from gca.labels import (
    em, extract_answer, grounding_gap, grounding_metrics, label_rollout, normalize_answer, sub_em, tool_hit,
)

GOLD = ["Thomas Jefferson", "Jefferson"]
DOC_HIT = "Doc 1 (Title: Declaration)\nIt was drafted by Thomas Jefferson in 1776."
DOC_MISS = "Doc 1 (Title: Paris)\nParis is the capital of France."


def test_normalize_matches_verl():
    assert normalize_answer("  The Thomas-Jefferson! ") == "thomasjefferson"
    assert normalize_answer("An apple") == "apple"


def test_extract_last_answer():
    assert extract_answer("<answer>a</answer> ... <answer> b </answer>") == "b"
    assert extract_answer("no tags") is None


def test_em_vs_sub_em():
    assert em("the Jefferson", GOLD) == 1
    assert em("President Thomas Jefferson", GOLD) == 0
    assert sub_em("President Thomas Jefferson", GOLD) == 1


def test_tool_hit_first_index():
    assert tool_hit([DOC_MISS, DOC_HIT, DOC_HIT], GOLD) == (1, 2)
    assert tool_hit([DOC_MISS], GOLD) == (0, None)
    assert tool_hit(["thomas JEFFERSON wrote"], GOLD)[0] == 1  # case-insensitive


@pytest.mark.parametrize("answer,obs,retr,expected", [
    ("<answer>Thomas Jefferson</answer>", [DOC_HIT], True, "A"),
    ("<answer>John Adams</answer>", [DOC_HIT], True, "B1"),
    ("<tool_call>...</tool_call>", [DOC_HIT], True, "B2"),     # never answered
    ("<answer>John Adams</answer>", [DOC_MISS], True, "C"),
    ("<answer>Jefferson</answer>", [DOC_MISS], True, "D"),      # right, but not from evidence
    ("<answer>Jefferson</answer>", [DOC_MISS], False, "D0"),    # corpus lacks the answer
    ("<answer>Jefferson</answer>", [], True, "D"),              # never searched at all
])
def test_rollout_types(answer, obs, retr, expected):
    assert label_rollout(answer, obs, GOLD, retr).rtype == expected


def test_metrics_hand_computed():
    # 4 rollouts: A, B1, C, D
    labels = [
        label_rollout("<answer>Thomas Jefferson</answer>", [DOC_HIT], GOLD),
        label_rollout("<answer>John Adams</answer>", [DOC_HIT], GOLD),
        label_rollout("<answer>John Adams</answer>", [DOC_MISS], GOLD),
        label_rollout("<answer>Jefferson</answer>", [DOC_MISS], GOLD),
    ]
    m = grounding_metrics(labels)
    assert m["EM"] == 0.5 and m["Acc"] == 0.5
    assert m["ToolHit"] == 0.5                 # 2 of 4 saw the answer
    assert m["HitAcc"] == 0.5                  # of those, 1 answered correctly
    assert m["SupCor"] == 0.25                 # 0.5 * 0.5
    assert m["GroundedShare"] == 0.5           # half of correct answers were backed by evidence
    # Delta = Pr[C|H=1] - Pr[C|H=0] = 1/2 - 1/2 = 0  -> GRPO gives searching no net credit here
    assert grounding_gap(labels) == 0.0


def test_gap_undefined_when_all_hit():
    labels = [label_rollout("<answer>Jefferson</answer>", [DOC_HIT], GOLD)] * 3
    assert grounding_gap(labels) is None
