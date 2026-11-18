"""Per-test-question strata for the eval tables (from the 11-03 data audit):

  q_in_train  the exact (normalised) question also appears in the training set  -> 60 such test questions
  dup         how many times (dataset, question) occurs in the test set          -> 1,377 extra copies
  ans_seen    one of its gold answers is also a gold answer of some training question (NQ 49.95% ...)

Usage: python scripts/make_strata.py --data <data dir>   -> <data>/test_strata.json
Key: "<dataset>\\t<normalised question>".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from data_audit import iter_rows  # noqa: E402
from gca.labels import normalize_answer  # noqa: E402


def key(ds: str, question: str) -> str:
    return f"{ds}\t{normalize_answer(question)}"


def build(data: str) -> dict:
    train_q, train_a = set(), set()
    for _, q, golds in iter_rows(os.path.join(data, "train.parquet")):
        train_q.add(normalize_answer(q))
        train_a.update(normalize_answer(g) for g in golds)
    count, out = Counter(), {}
    for ds, q, golds in iter_rows(os.path.join(data, "test.parquet")):
        k = key(ds, q)
        count[k] += 1
        out[k] = {"q_in_train": int(normalize_answer(q) in train_q),
                  "ans_seen": int(any(normalize_answer(g) in train_a for g in golds))}
    for k, c in count.items():
        out[k]["dup"] = c
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/autodl-tmp/data/searchR1_processed_direct")
    a = ap.parse_args()
    s = build(a.data)
    path = os.path.join(a.data, "test_strata.json")
    json.dump(s, open(path, "w"))
    print(f"wrote {path}: {len(s)} distinct test questions, "
          f"{sum(v['q_in_train'] for v in s.values())} also in train")
