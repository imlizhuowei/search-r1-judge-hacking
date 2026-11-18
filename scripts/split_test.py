"""Split test.parquet (7 datasets, 51,713 questions) into one parquet per dataset, for scripts/eval.sh.

One eval run per dataset means progress shows up dataset by dataset, a crash loses at most one dataset,
and rerunning eval.sh skips datasets that are already done.

Usage: python scripts/split_test.py --data /root/autodl-tmp/data/searchR1_processed_direct
Writes <data>/test_by_ds/<name>.parquet (name = data_source without the "searchR1_" prefix) and prints counts.
"""

from __future__ import annotations

import argparse
import os

import pyarrow.compute as pc
import pyarrow.parquet as pq


def ds_name(data_source: str) -> str:
    return data_source.replace("searchR1_", "")


def split(data: str) -> dict[str, int]:
    test = pq.read_table(os.path.join(data, "test.parquet"))
    out_dir = os.path.join(data, "test_by_ds")
    os.makedirs(out_dir, exist_ok=True)
    counts = {}
    for src in sorted(set(test.column("data_source").to_pylist())):
        part = test.filter(pc.equal(test.column("data_source"), src))
        pq.write_table(part, os.path.join(out_dir, f"{ds_name(src)}.parquet"))
        counts[ds_name(src)] = part.num_rows
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/autodl-tmp/data/searchR1_processed_direct")
    a = ap.parse_args()
    for k, v in split(a.data).items():
        print(f"{k}\t{v}")
