"""Turn the eval dumps of scripts/eval.sh into the reproduction table.

  python scripts/summarize_eval.py /root/autodl-fs/runs/eval_grpo_em_s1 --tag grpo_em_s1
      -> results/eval_grpo_em_s1.json  (per dataset: n, EM, lenient Acc, answered, searches, ToolHit;
                                        and EM on strata: clean / question-in-train / answer-seen / not)
  python scripts/summarize_eval.py --table          -> results/eval_table.md: every results/eval_*.json
                                                       next to the Search-R1 paper's 3B numbers
  python scripts/summarize_eval.py <out> --one nq   -> prints that dataset's EM (for the progress push)

EM here = mean of the reward, which is exactly verl's Search-R1 EM (gca/reward.py; the >10 <answer> tags
rule can give 0.25). "clean" drops the 60 test questions that also occur in train and keeps one copy of each
duplicated test question (strata from scripts/make_strata.py, if present).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gca.labels import normalize_answer  # noqa: E402

DATASETS = ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]
SHORT = {"2wikimultihopqa": "2wiki"}
STRATA_PATH = "/root/autodl-tmp/data/searchR1_processed_direct/test_strata.json"

# Search-R1 (Jin et al. 2025, arXiv 2503.09516), Qwen2.5-3B, EM. Table 2 (PPO) and Table 3 (GRPO).
# Their training: batch 512 x 500 steps, rollout n=5, KL 0.001, max 4 searches, 500-token responses.
PAPER = {
    "paper: Search-R1-instruct GRPO": [0.397, 0.565, 0.391, 0.331, 0.310, 0.124, 0.232],
    "paper: Search-R1-instruct PPO": [0.341, 0.545, 0.378, 0.324, 0.319, 0.103, 0.264],
    "paper: RAG (no RL)": [0.348, 0.544, 0.387, 0.255, 0.226, 0.047, 0.080],
    "paper: Direct inference": [0.106, 0.288, 0.108, 0.149, 0.244, 0.020, 0.024],
}


def load_dumps(out: str) -> dict[str, list[dict]]:
    rows = {}
    for ds in DATASETS:
        files = sorted(glob.glob(os.path.join(out, ds, "*.jsonl")))
        if files:
            rows[ds] = [json.loads(line) for line in open(files[-1])]
    return rows


def mean(xs):
    xs = list(xs)
    return round(sum(xs) / len(xs), 4) if xs else None


def stratum_of(r: dict, ds: str, strata: dict) -> dict | None:
    q = r.get("question") or ""
    return strata.get(f"{ds}\t{normalize_answer(q)}") if q else None


def summarize(out: str, strata: dict | None = None) -> dict:
    rows = load_dumps(out)
    rep = {"run_dir": out, "per_dataset": {}}
    ck = os.path.join(out, "checkpoint.txt")
    if os.path.exists(ck):
        rep["checkpoint"] = open(ck).read().strip()
    for ds, rs in rows.items():
        d = {
            "n": len(rs),
            "EM": mean(r["score"] for r in rs),
            "Acc_lenient": mean(r.get("acc_lenient", 0) for r in rs),
            "answered": mean(r.get("has_answer", 0) for r in rs),
            "tool_calls_mean": mean(r.get("n_tool_calls", 0) for r in rs),
            "ToolHit": mean(r.get("gca_hit", 0) for r in rs),
        }
        if strata:
            seen, groups = set(), {"clean": [], "q_in_train": [], "ans_seen": [], "ans_unseen": []}
            for r in rs:
                st = stratum_of(r, ds, strata)
                if st is None:
                    continue
                k = normalize_answer(r.get("question", ""))
                if st["q_in_train"]:
                    groups["q_in_train"].append(r["score"])
                elif k not in seen:
                    groups["clean"].append(r["score"])
                seen.add(k)
                groups["ans_seen" if st["ans_seen"] else "ans_unseen"].append(r["score"])
            d["strata"] = {g: {"n": len(v), "EM": mean(v)} for g, v in groups.items()}
        rep["per_dataset"][ds] = d
    done = [d for d in rep["per_dataset"].values()]
    rep["avg_EM_over_datasets"] = mean(d["EM"] for d in done) if len(done) == len(DATASETS) else None
    rep["datasets_done"] = list(rep["per_dataset"])
    return rep


def table(results_dir: str = "results") -> str:
    head = "| model | " + " | ".join(SHORT.get(d, d) for d in DATASETS) + " | avg |"
    lines = [head, "|" + "---|" * (len(DATASETS) + 2)]
    for f in sorted(glob.glob(os.path.join(results_dir, "eval_*.json"))):
        r = json.load(open(f))
        tag = os.path.basename(f)[5:-5]
        cells = [r["per_dataset"].get(d, {}).get("EM") for d in DATASETS]
        avg = r.get("avg_EM_over_datasets")
        lines.append(f"| ours: {tag} ({r.get('checkpoint', '')}) | "
                     + " | ".join("—" if c is None else f"{c:.3f}" for c in cells)
                     + f" | {'—' if avg is None else f'{avg:.3f}'} |")
    for name, vals in PAPER.items():
        lines.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in vals) + f" | {sum(vals) / len(vals):.3f} |")
    lines += ["", "Ours: batch 64 x 300 steps (~13x fewer training prompts than the paper's 512 x 500), "
              "max 5 assistant turns, 2048 response tokens, hermes <tool_call> format; same retriever, top-3, EM."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("--tag")
    ap.add_argument("--one")
    ap.add_argument("--line", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--strata", default=STRATA_PATH)
    ap.add_argument("--results", default="results")
    a = ap.parse_args(argv)
    if a.table:
        os.makedirs(a.results, exist_ok=True)
        md = table(a.results)
        open(os.path.join(a.results, "eval_table.md"), "w").write(md)
        print(md)
        return
    strata = json.load(open(a.strata)) if os.path.exists(a.strata) else None
    rep = summarize(a.out, strata)
    if a.one:
        print(rep["per_dataset"].get(a.one, {}).get("EM"))
        return
    if a.line:
        cells = " ".join(f"{SHORT.get(d, d)} {v['EM']:.3f}" for d, v in rep["per_dataset"].items())
        print(f"avg {rep['avg_EM_over_datasets']} | {cells}")
        return
    os.makedirs(a.results, exist_ok=True)
    path = os.path.join(a.results, f"eval_{a.tag}.json")
    json.dump(rep, open(path, "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
