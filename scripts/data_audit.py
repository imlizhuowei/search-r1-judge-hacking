"""Data audit: per-dataset counts, duplicates, train/test leakage, single- vs multi-hop strata.

Low memory (runs in AutoDL no-GPU mode, 2 GB RAM): reads only the needed columns in batches.

Usage:
  python scripts/data_audit.py --data /root/autodl-tmp/data/searchR1_processed_direct --out results/data_audit
Writes <out>.json and <out>.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gca.labels import normalize_answer  # noqa: E402

MULTI_HOP = {"hotpotqa", "2wikimultihopqa", "musique", "bamboogle"}


def ds_name(data_source: str) -> str:
    return data_source.replace("searchR1_", "")


def iter_rows(path: str, batch_size: int = 20000):
    pf = pq.ParquetFile(path)
    cols = [c for c in ("data_source", "extra_info", "reward_model") if c in pf.schema_arrow.names]
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        for r in batch.to_pylist():
            q = (r.get("extra_info") or {}).get("question", "")
            gt = ((r.get("reward_model") or {}).get("ground_truth") or {})
            golds = gt.get("target", gt) if isinstance(gt, dict) else gt
            golds = [str(g) for g in (golds or [])]
            yield ds_name(r.get("data_source", "")), q, golds


def audit(data_dir: str) -> dict:
    splits = {}
    for split in ("train", "test"):
        path = os.path.join(data_dir, f"{split}.parquet")
        per_ds = Counter()
        qcount = Counter()
        answers = defaultdict(set)  # dataset -> normalised gold answers
        qset = set()
        empty_q = 0
        for ds, q, golds in iter_rows(path):
            per_ds[ds] += 1
            nq = normalize_answer(q)
            empty_q += not nq
            qcount[(ds, nq)] += 1
            qset.add(nq)
            for g in golds:
                answers[ds].add(normalize_answer(g))
        dup = sum(c - 1 for c in qcount.values() if c > 1)
        splits[split] = {"per_ds": per_ds, "qset": qset, "qcount": qcount, "answers": answers,
                         "dup_within": dup, "empty_q": empty_q}

    train_q = splits["train"]["qset"]
    train_ans = set().union(*splits["train"]["answers"].values()) if splits["train"]["answers"] else set()

    leak = Counter()
    test_total = Counter()
    for (ds, nq), c in splits["test"]["qcount"].items():
        test_total[ds] += c
        if nq in train_q:
            leak[ds] += c

    ans_overlap = {}
    for ds, ans in splits["test"]["answers"].items():
        ans_overlap[ds] = round(len(ans & train_ans) / max(len(ans), 1), 4)

    report = {
        "counts": {s: dict(splits[s]["per_ds"]) for s in splits},
        "duplicate_questions_within_split": {s: splits[s]["dup_within"] for s in splits},
        "empty_questions": {s: splits[s]["empty_q"] for s in splits},
        "test_question_in_train": {ds: {"n": leak[ds], "frac": round(leak[ds] / test_total[ds], 4)}
                                   for ds in test_total},
        "test_answer_seen_as_train_answer": ans_overlap,
        "hop": {ds: ("multi" if ds in MULTI_HOP else "single") for ds in test_total},
    }
    return report


def to_markdown(r: dict) -> str:
    lines = ["# Data audit", "", "| dataset | hop | train | test | test Q in train | test answers seen in train |",
             "|---|---|---|---|---|---|"]
    for ds in sorted(set(r["counts"].get("train", {})) | set(r["counts"].get("test", {}))):
        lk = r["test_question_in_train"].get(ds, {"n": 0, "frac": 0})
        lines.append(f"| {ds} | {r['hop'].get(ds, '-')} | {r['counts']['train'].get(ds, 0)} | "
                     f"{r['counts']['test'].get(ds, 0)} | {lk['n']} ({lk['frac']:.2%}) | "
                     f"{r['test_answer_seen_as_train_answer'].get(ds, 0):.2%} |")
    lines += ["", f"Duplicate questions within split: {r['duplicate_questions_within_split']}",
              f"Empty questions: {r['empty_questions']}", "",
              "Note: 'answers seen in train' is expected to be high for common answers (years, countries); "
              "it flags memorisable questions, not leakage by itself."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/autodl-tmp/data/searchR1_processed_direct")
    ap.add_argument("--out", default="results/data_audit")
    a = ap.parse_args()
    rep = audit(a.data)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".json", "w") as f:
        json.dump(rep, f, indent=2)
    md = to_markdown(rep)
    with open(a.out + ".md", "w") as f:
        f.write(md)
    print(md)
