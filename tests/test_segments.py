import numpy as np

from gca.advantage import ANS, NONE, RSEARCH, VSEARCH
from gca.segments import runs_of_ones, segment_ids

N, V, R, S = NONE, VSEARCH, RSEARCH, ANS


def test_runs():
    assert runs_of_ones(np.array([1, 1, 0, 1, 0, 0, 1])) == [(0, 2), (3, 4), (6, 7)]
    assert runs_of_ones(np.array([0, 0])) == []


def test_two_searches_hit_on_first():
    # [search1 x2][tool x2][search2 x2][tool x1][answer x2][pad x2]
    mask = np.array([1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0])
    seg = segment_ids(mask, first_hit=1)
    assert seg.tolist() == [V, V, N, N, R, R, N, S, S, N, N]   # 2nd search is redundant


def test_hit_on_second_search():
    mask = np.array([1, 0, 1, 0, 1])
    assert segment_ids(mask, first_hit=2).tolist() == [V, N, V, N, S]


def test_no_hit_all_searches_rsearch():
    mask = np.array([1, 0, 1, 0, 1])
    assert segment_ids(mask, first_hit=None).tolist() == [R, N, R, N, S]


def test_answer_without_search():
    assert segment_ids(np.array([1, 1, 1, 0]), first_hit=None).tolist() == [S, S, S, N]


def test_unanswered_last_turn_is_search():
    # B2: budget ran out after a tool response; final assistant turn was another search attempt
    mask = np.array([1, 0, 1])
    assert segment_ids(mask, first_hit=1, answered=False).tolist() == [V, N, R]
