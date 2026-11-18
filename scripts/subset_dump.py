"""Reuse a finished full-dataset eval dump for a sampled eval: keep only the rows whose question is in the
sampled parquet (counting duplicates), so the dataset does not have to be run again.

  python scripts/subset_dump.py <full_dump_dir> <sample.parquet> <out_dir>
Exit 1 (and write nothing) if the full dump does not cover every sampled question.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter

import pyarrow.parquet as pq


def subset(full_dir: str, sample_parquet: str, out_dir: str) -> int:
    files = sorted(glob.glob(os.path.join(full_dir, "*.jsonl")))
    if not files:
        return 0
    need = Counter((r or {}).get("question", "") for r in pq.read_table(sample_parquet).column("extra_info").to_pylist())
    keep = []
    for line in open(files[-1]):
        q = json.loads(line).get("question", "")
        if need[q] > 0:
            need[q] -= 1
            keep.append(line)
    if sum(need.values()):
        return 0
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, os.path.basename(files[-1])), "w") as f:
        f.writelines(keep)
    return len(keep)


if __name__ == "__main__":
    n = subset(*sys.argv[1:4])
    print(n)
    sys.exit(0 if n else 1)
