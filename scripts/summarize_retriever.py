"""Summarise the retriever's per-batch log (retriever/batcher.py, RETRIEVER_STATS) over a time window.

Answers, for the timing measurement (LOG 11-18):
  * how busy the retriever is (busy seconds / wall seconds), queries per second, mean batch size;
  * where its time goes: query encoding (E5 on CPU), faiss search over 21M vectors, loading passages;
  * how long rollouts wait for it: queue wait and wait + batch time per request (p50 / p90 / p99);
  * how often the same query repeats (the ceiling for a query cache);
  * with --train-log: retriever busy time as a share of verl's timing_s/gen (rollout time), summed over steps.

Usage: python scripts/summarize_retriever.py <stats.jsonl> [--since EPOCH] [--until EPOCH] [--train-log LOG]
                                             [--out results/retriever_probe.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter


def pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round(p / 100 * (len(xs) - 1)))))
    return round(xs[i], 4)


def load(path: str, since: float | None = None, until: float | None = None) -> list[dict]:
    recs = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if since is not None and r["t"] < since:
                continue
            if until is not None and r["t"] > until:
                continue
            recs.append(r)
    return recs


def summarize(recs: list[dict], since: float | None = None, until: float | None = None) -> dict:
    if not recs:
        return {"n_batches": 0}
    t_first, t_last = recs[0]["t"], recs[-1]["t"] + recs[-1]["total_s"]
    wall = (until if until is not None else t_last) - (since if since is not None else t_first)
    busy = sum(r["total_s"] for r in recs)
    n_q = sum(r["n_q"] for r in recs)
    split = {k: round(sum(r.get(k, 0.0) for r in recs), 2) for k in ("encode_s", "search_s", "docs_s")}
    waits = [w for r in recs for w in r.get("wait_s", [])]
    latency = [w + r["total_s"] for r in recs for w in r.get("wait_s", [])]
    keys = [k for r in recs for k in r.get("keys", [])]
    counts = Counter(keys)
    return {
        "n_batches": len(recs),
        "n_requests": sum(r["n_req"] for r in recs),
        "n_queries": n_q,
        "wall_s": round(wall, 1),
        "busy_s": round(busy, 1),
        "utilization": round(busy / wall, 4) if wall > 0 else None,
        "queries_per_s": round(n_q / wall, 2) if wall > 0 else None,
        "mean_batch_queries": round(n_q / len(recs), 1),
        "time_split_s": split,
        "time_split_share": {k: round(v / busy, 4) if busy else None for k, v in split.items()},
        "ms_per_query_in_batch": round(1000 * busy / n_q, 2) if n_q else None,
        "queue_wait_s": {"p50": pct(waits, 50), "p90": pct(waits, 90), "p99": pct(waits, 99),
                         "max": round(max(waits), 4) if waits else None},
        "request_latency_s": {"p50": pct(latency, 50), "p90": pct(latency, 90), "p99": pct(latency, 99)},
        # exact repeats (lower-cased, whitespace-collapsed): the most a query cache could skip
        "repeat_rate": round(1 - len(counts) / len(keys), 4) if keys else None,
        "unique_queries": len(counts),
        "top_repeat_counts": [c for _, c in counts.most_common(5)],
        "errors": sum(not r.get("ok", True) for r in recs),
    }


def with_train_log(rep: dict, train_log: str) -> dict:
    sys.path.insert(0, os.path.dirname(__file__))
    from summarize_timing import parse_steps

    steps = parse_steps(train_log)
    if not steps:
        return rep
    gen = sum(s.get("timing_s/gen", 0.0) for s in steps)
    step = sum(s.get("timing_s/step", 0.0) for s in steps)
    rep["train"] = {
        "steps": len(steps),
        "gen_s_total": round(gen, 1),
        "step_s_total": round(step, 1),
        "gen_share_of_step": round(gen / step, 4) if step else None,
        # retriever busy time vs rollout time. Searches overlap with other rollouts' generation, so this is how
        # much of the rollout phase the retriever was working, not how much it delayed the step.
        "retriever_busy_over_gen": round(rep["busy_s"] / gen, 4) if gen else None,
        "mean_per_step": {k: round(sum(s.get(k, 0.0) for s in steps) / len(steps), 2)
                          for k in sorted({k for s in steps for k in s if k.startswith("timing_s/")})},
    }
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stats")
    ap.add_argument("--since", type=float)
    ap.add_argument("--until", type=float)
    ap.add_argument("--train-log")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    rep = summarize(load(a.stats, a.since, a.until), a.since, a.until)
    if a.train_log and rep.get("n_batches"):
        rep = with_train_log(rep, a.train_log)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
