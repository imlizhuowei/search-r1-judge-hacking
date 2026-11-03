"""Stratified validation subset: N questions per test dataset, excluding test questions that also appear in train.

Full test set is 51,713 questions (too slow to evaluate every few steps). Same seed -> same subset for every run.

Usage: python scripts/make_val_subset.py --data /root/autodl-tmp/data/searchR1_processed_direct --per-ds 100
Writes <data>/val_subset_<per_ds>.parquet
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gca.labels import normalize_answer  # noqa: E402


def train_questions(path: str) -> set[str]:
    qs = set()
    pf = pq.ParquetFile(path)
    for b in pf.iter_batches(batch_size=20000, columns=["extra_info"]):
        for r in b.to_pylist():
            qs.add(normalize_answer((r.get("extra_info") or {}).get("question", "")))
    return qs


def build(data: str, per_ds: int, seed: int = 0) -> str:
    leaked = train_questions(os.path.join(data, "train.parquet"))
    test = pq.read_table(os.path.join(data, "test.parquet"))
    rows = test.to_pylist()
    by_ds, seen = defaultdict(list), set()
    for i, r in enumerate(rows):
        q = normalize_answer((r.get("extra_info") or {}).get("question", ""))
        if q in leaked or (r["data_source"], q) in seen:   # drop leaked and duplicate questions
            continue
        seen.add((r["data_source"], q))
        by_ds[r["data_source"]].append(i)
    rng = random.Random(seed)
    keep = []
    for ds in sorted(by_ds):
        idx = by_ds[ds]
        keep += sorted(rng.sample(idx, min(per_ds, len(idx))))
    out = os.path.join(data, f"val_subset_{per_ds}.parquet")
    pq.write_table(test.take(pa.array(keep)), out)
    print(f"wrote {out}: {len(keep)} rows from {len(by_ds)} datasets (leaked/duplicate questions removed)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/autodl-tmp/data/searchR1_processed_direct")
    ap.add_argument("--per-ds", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.data, a.per_ds, a.seed)
