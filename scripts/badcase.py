"""Badcase attribution for an eval run: where do the wrong answers come from?

Every question gets exactly one category (checked in this order):

  correct_grounded    EM = 1 and a gold answer appeared in what the search returned
  correct_unsupported EM = 1 but no retrieved passage contained a gold answer (memory, or lucky)
  no_answer           no <answer> tag: ran out of turns / tokens, or rambled
  fabricated          more <tool_response> blocks than <tool_call>s: the model wrote "search results" itself
  no_search           answered wrong without ever calling the search tool
  retrieval_miss      searched, but no returned passage contained a gold answer
  near_miss           gold answer is inside the prediction but EM fails (extra words, e.g. "Paris, France")
  evidence_ignored    a returned passage contained a gold answer, yet the final answer is wrong

"contained" = case-insensitive substring of a gold alias (same naive test as ToolHit; '1' matches '1999'),
so retrieval_miss is a lower bound on true misses and evidence_ignored an upper bound.

Usage: python scripts/badcase.py /root/autodl-fs/runs/eval_grpo_em_s1 --tag grpo_em_s1
Writes results/badcase_<tag>.json (counts per dataset) and results/badcase_<tag>.md (shares + 3 examples
per category, to read by hand).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from gca.labels import extract_answer, label_rollout  # noqa: E402
from gca.reward import observations_from_text  # noqa: E402
from summarize_eval import DATASETS, load_dumps  # noqa: E402

CATEGORIES = ["correct_grounded", "correct_unsupported", "no_answer", "fabricated", "no_search",
              "retrieval_miss", "near_miss", "evidence_ignored"]
WRONG = CATEGORIES[2:]


def golds_of(gts) -> list[str]:
    if isinstance(gts, dict):
        gts = gts.get("target", [])
    return [str(g) for g in (gts or [])]


def categorize(output: str, golds: list[str]) -> str:
    obs = observations_from_text(output)
    lab = label_rollout(output, obs, golds)
    n_call, n_resp = output.count("<tool_call>"), output.count("<tool_response>")
    if lab.correct:
        return "correct_grounded" if lab.hit else "correct_unsupported"
    if not lab.answered:
        return "no_answer"
    if n_resp > n_call:
        return "fabricated"
    if n_call == 0:
        return "no_search"
    if not lab.hit:
        return "retrieval_miss"
    if lab.lenient:
        return "near_miss"
    return "evidence_ignored"


def analyze(out: str, n_examples: int = 3, seed: int = 0) -> tuple[dict, dict]:
    rows = load_dumps(out)
    counts = {ds: Counter() for ds in rows}
    examples = defaultdict(list)
    for ds, rs in rows.items():
        for r in rs:
            golds = golds_of(r.get("gts"))
            c = categorize(r["output"], golds)
            counts[ds][c] += 1
            examples[c].append({"dataset": ds, "question": r.get("question", ""), "gold": golds,
                                "pred": extract_answer(r["output"]), "output": r["output"][-1500:]})
    total = Counter()
    for c in counts.values():
        total.update(c)
    all_rows = [r for rs in rows.values() for r in rs]
    called = [r for r in all_rows if "<tool_call>" in r["output"]]
    rng = random.Random(seed)
    picked = {c: rng.sample(v, min(n_examples, len(v))) for c, v in examples.items()}
    rep = {"run_dir": out, "total": dict(total), "n": sum(total.values()),
           "per_dataset": {ds: dict(c) for ds, c in counts.items()},
           # sanity check (NEXT.md): if the decoded outputs lost the <tool_response> tags, every searched
           # question would look like a retrieval miss. Should be ~1.0.
           "tool_response_tag_rate_when_searched":
               round(sum("<tool_response>" in r["output"] for r in called) / len(called), 4) if called else None}
    return rep, picked


def to_markdown(rep: dict, picked: dict, tag: str) -> str:
    n = rep["n"] or 1
    wrong = sum(rep["total"].get(c, 0) for c in WRONG) or 1
    tag_rate = rep.get("tool_response_tag_rate_when_searched")
    warn = []
    if tag_rate is not None and tag_rate < 0.9:
        warn = [f"**WARNING: only {tag_rate:.0%} of outputs that call the tool contain <tool_response> tags; "
                "retrieved text is probably missing from the decoded output, so retrieval_miss is inflated. "
                "Fix this before trusting the table.**", ""]
    lines = [f"# Badcase attribution: {tag}", "", *warn, f"{rep['n']} questions.", "",
             "| category | count | share of all | share of wrong |", "|---|---|---|---|"]
    for c in CATEGORIES:
        k = rep["total"].get(c, 0)
        sw = f"{k / wrong:.1%}" if c in WRONG else ""
        lines.append(f"| {c} | {k} | {k / n:.1%} | {sw} |")
    lines += ["", "## Per dataset (share of all questions)", "",
              "| dataset | " + " | ".join(CATEGORIES) + " |", "|" + "---|" * (len(CATEGORIES) + 1)]
    for ds in DATASETS:
        c = rep["per_dataset"].get(ds)
        if not c:
            continue
        m = sum(c.values()) or 1
        lines.append(f"| {ds} | " + " | ".join(f"{c.get(k, 0) / m:.1%}" for k in CATEGORIES) + " |")
    lines += ["", "## Examples (last 1500 characters of each output)"]
    for c in CATEGORIES:
        for e in picked.get(c, []):
            lines += ["", f"### {c} — {e['dataset']}", f"Q: {e['question']}", f"gold: {e['gold']} | pred: {e['pred']}",
                      "", "```", e["output"], "```"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--results", default="results")
    a = ap.parse_args(argv)
    rep, picked = analyze(a.out)
    os.makedirs(a.results, exist_ok=True)
    json.dump(rep, open(os.path.join(a.results, f"badcase_{a.tag}.json"), "w"), indent=1)
    md = to_markdown(rep, picked, a.tag)
    open(os.path.join(a.results, f"badcase_{a.tag}.md"), "w").write(md)
    print(md[:3000])


if __name__ == "__main__":
    main()
