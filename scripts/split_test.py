"""Split test.parquet (7 datasets, 51,713 questions) into one parquet per dataset, for scripts/eval.sh.

One eval run per dataset means progress shows up dataset by dataset, a crash loses at most one dataset,
and rerunning eval.sh skips datasets that are already done.

Usage: python scripts/split_test.py --data /root/autodl-tmp/data/searchR1_processed_direct
Writes <data>/test_by_ds/<name>.parquet (name = data_source without the "searchR1_" prefix) and prints counts.
"""

from __future__ import annotations

import argparse
import os
import random

import pyarrow.compute as pc
import pyarrow.parquet as pq


def ds_name(data_source: str) -> str:
    return data_source.replace("searchR1_", "")


def split(data: str, sample: int = 0, seed: int = 0) -> dict[str, int]:
    """sample > 0: keep a fixed random subset of at most `sample` questions per dataset (same seed -> same
    questions, so base and trained models are compared on identical questions)."""
    test = pq.read_table(os.path.join(data, "test.parquet"))
    out_dir = os.path.join(data, f"test_by_ds_s{sample}" if sample else "test_by_ds")
    os.makedirs(out_dir, exist_ok=True)
    counts = {}
    for src in sorted(set(test.column("data_source").to_pylist())):
        part = test.filter(pc.equal(test.column("data_source"), src))
        if sample and part.num_rows > sample:
            idx = sorted(random.Random(f"{seed}-{src}").sample(range(part.num_rows), sample))
            part = part.take(idx)
        pq.write_table(part, os.path.join(out_dir, f"{ds_name(src)}.parquet"))
        counts[ds_name(src)] = part.num_rows
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/autodl-tmp/data/searchR1_processed_direct")
    ap.add_argument("--sample", type=int, default=0, help="max questions per dataset (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for k, v in split(a.data, a.sample, a.seed).items():
        print(f"{k}\t{v}")
