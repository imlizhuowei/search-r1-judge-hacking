"""Grounding analysis of saved training rollouts (verl trainer.rollout_data_dir, one <step>.jsonl per step).

Per step: EM (reward), ToolHit, HitAcc, SupCor, GroundedShare, rollout types A/B1/B2/C/D,
mean grounding gap Delta over groups (GCA Prop. 1), searches per rollout, response length,
and how many outputs still contain <tool_response> tags (sanity check for the labels).

Usage: python scripts/analyze_rollouts.py /root/autodl-fs/runs/grpo_em_s1 [--every 10]
Writes <run>/analysis/grounding_by_step.csv and prints a compact table.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gca.labels import grounding_gap, grounding_metrics, label_rollout  # noqa: E402
from gca.reward import observations_from_text  # noqa: E402


def golds_of(gts) -> list[str]:
    if isinstance(gts, dict):
        gts = gts.get("target", [])
    return [str(g) for g in (gts or [])]


def analyze_step(path: str) -> dict:
    groups = defaultdict(list)
    labels, n_search, lens, tagged = [], [], [], 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out = r["output"]
            obs = observations_from_text(out)
            lab = label_rollout(out, obs, golds_of(r.get("gts")))
            labels.append(lab)
            groups[r["input"]].append(lab)
            n_search.append(out.count("<tool_call>"))
            lens.append(len(out))
            tagged += "<tool_response>" in out
    m = grounding_metrics(labels)
    gaps = [g for g in (grounding_gap(v) for v in groups.values()) if g is not None]
    n = len(labels)
    row = {
        "step": int(os.path.basename(path).split(".")[0]),
        "n": n,
        **{k: round(v, 4) for k, v in m.items() if not isinstance(v, dict)},
        **{f"type_{k}": round(v, 4) for k, v in m["types"].items()},
        "delta_mean": round(sum(gaps) / len(gaps), 4) if gaps else None,
        "delta_groups": len(gaps),
        "searches_mean": round(sum(n_search) / n, 3),
        "resp_chars_mean": round(sum(lens) / n, 1),
        "tool_response_tag_rate": round(tagged / n, 4),
    }
    return row


def main(run_dir: str, every: int = 1):
    files = sorted(glob.glob(os.path.join(run_dir, "rollouts", "*.jsonl")),
                   key=lambda p: int(os.path.basename(p).split(".")[0]))
    if not files:
        sys.exit(f"no rollout files under {run_dir}/rollouts")
    rows = [analyze_step(p) for i, p in enumerate(files) if i % every == 0 or p == files[-1]]
    out_dir = os.path.join(run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "grounding_by_step.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    cols = ["step", "EM", "ToolHit", "HitAcc", "SupCor", "type_A", "type_B1", "type_D",
            "delta_mean", "searches_mean", "tool_response_tag_rate"]
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(str(r.get(c)) for c in cols))
    print(f"\nwrote {out}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--every", type=int, default=10)
    a = ap.parse_args()
    main(a.run_dir, a.every)
