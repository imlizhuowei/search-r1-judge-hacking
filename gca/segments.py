"""Map a multi-turn response to per-token GCA segments using only verl's response_mask.

In verl's tool agent loop the response is:
    [assistant turn 1][tool response 1][assistant turn 2][tool response 2] ... [assistant final turn]
response_mask = 1 on assistant tokens (trained), 0 on tool-response tokens (not trained).

So: each maximal run of 1s is one assistant turn. A turn followed by a run of 0s made a tool call
(search turn k = 1, 2, ...). The last assistant turn, if not followed by a tool response, is the
answer turn (ANS). Search turns k <= k* are VSEARCH, later ones RSEARCH (GCA Sec. 4.1).
Rollouts with H=0 (k* is None) have all searches in RSEARCH.

Padding: verl right-pads responses with mask 0; trailing 0s after the last 1 are padding, not a tool response.
"""

from __future__ import annotations

import numpy as np

from gca.advantage import ANS, NONE, RSEARCH, VSEARCH


def runs_of_ones(mask: np.ndarray) -> list[tuple[int, int]]:
    """[(start, end_exclusive), ...] for each maximal run of 1s."""
    m = np.concatenate([[0], np.asarray(mask, dtype=int), [0]])
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def segment_ids(mask: np.ndarray, first_hit: int | None, answered: bool = True) -> np.ndarray:
    """Per-token segment ids for one rollout. mask: (T,) 0/1. first_hit: k* (1-based) or None.

    answered=False (e.g. turn budget exhausted, B2): the last turn is a search turn cut off, not an answer.
    """
    seg = np.full(len(mask), NONE)
    turns = runs_of_ones(mask)
    if not turns:
        return seg
    n_search = len(turns) - 1 if answered else len(turns)
    for j, (s, e) in enumerate(turns, start=1):
        if j > n_search:
            seg[s:e] = ANS
        elif first_hit is not None and j <= first_hit:
            seg[s:e] = VSEARCH
        else:
            seg[s:e] = RSEARCH
    return seg


def batch_segment_ids(response_mask: np.ndarray, first_hits: list, answered: list) -> np.ndarray:
    """(n, T) version."""
    return np.stack([segment_ids(m, k, a) for m, k, a in zip(response_mask, first_hits, answered)])
