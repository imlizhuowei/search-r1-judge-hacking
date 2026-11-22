"""Why did searches fail under load? Reads the retriever's per-batch log (retriever/batcher.py, RETRIEVER_STATS)
and tells apart the two failure modes (LOG 11-22):

  * queueing: the retriever is alive the whole time but slower than requests arrive, so requests wait in the queue
    longer than the client's timeout (configs/search_tool.yaml: 30 s, 3 tries). Signature: batches keep coming,
    queue waits grow past 30 s, and the same query shows up again ~30 s later (client retries; the server still
    processes the abandoned original, so retries add load: a retry storm).
  * crash / restart: the server dies and launch.sh reloads the 64 GB index (~4 min), during which every request
    fails. Signature: gaps of minutes with no batch at all.

  python scripts/diagnose_retriever.py <stats.jsonl> [--since EPOCH] [--until EPOCH] [--timeout 30] [--out x.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from summarize_retriever import load, pct  # noqa: E402


def diagnose(recs: list[dict], timeout: float = 30.0, gap_s: float = 60.0, retry_window: float = 120.0) -> dict:
    if not recs:
        return {"n_batches": 0, "verdict": "no data"}
    recs = sorted(recs, key=lambda r: r["t"])
    waits = [w for r in recs for w in r.get("wait_s", [])]
    n_req = len(waits)
    timed_out = sum(w > timeout for w in waits)          # client had already given up when the batch started

    # gaps: nothing processed for > gap_s between the end of one batch and the start of the next
    gaps = []
    for a, b in zip(recs, recs[1:]):
        g = b["t"] - (a["t"] + a["total_s"])
        if g > gap_s:
            gaps.append({"after_t": round(a["t"] + a["total_s"], 1), "gap_s": round(g, 1)})

    # retries: the same query processed again within retry_window seconds
    last_seen: dict[str, float] = {}
    repeats_soon = 0
    n_q = 0
    for r in recs:
        for k in r.get("keys", []):
            n_q += 1
            if k in last_seen and r["t"] - last_seen[k] <= retry_window:
                repeats_soon += 1
            last_seen[k] = r["t"]

    # per-minute timeline: is the queue building up?
    per_min: dict[int, list] = defaultdict(list)
    for r in recs:
        per_min[int(r["t"] // 60)].append(r)
    timeline = []
    for m in sorted(per_min):
        rs = per_min[m]
        ws = [w for r in rs for w in r.get("wait_s", [])]
        timeline.append({"minute": m * 60, "batches": len(rs), "queries": sum(r["n_q"] for r in rs),
                         "busy_s": round(sum(r["total_s"] for r in rs), 1), "wait_p50": pct(ws, 50),
                         "wait_max": round(max(ws), 1) if ws else None})
    busy_minutes = sum(t["busy_s"] >= 55 for t in timeline)

    gap_total = sum(g["gap_s"] for g in gaps)
    span = recs[-1]["t"] + recs[-1]["total_s"] - recs[0]["t"]
    if timed_out / max(n_req, 1) > 0.2 and len(gaps) == 0:
        verdict = "queueing"
    elif gaps and timed_out / max(n_req, 1) <= 0.2:
        verdict = "crash_restart"
    elif gaps:
        verdict = "both"
    else:
        verdict = "healthy"
    return {
        "verdict": verdict,
        "n_batches": len(recs),
        "n_requests": n_req,
        "span_s": round(span, 1),
        "requests_waited_over_timeout": timed_out,
        "share_waited_over_timeout": round(timed_out / n_req, 4) if n_req else None,
        "queue_wait_s": {"p50": pct(waits, 50), "p90": pct(waits, 90), "max": round(max(waits), 1) if waits else None},
        "batch_s": {"p50": pct([r["total_s"] for r in recs], 50), "max": round(max(r["total_s"] for r in recs), 1)},
        "gaps": gaps[:20],
        "n_gaps": len(gaps),
        "gap_share_of_span": round(gap_total / span, 4) if span else None,
        "queries_repeated_within_window": repeats_soon,
        "repeat_share": round(repeats_soon / n_q, 4) if n_q else None,
        "minutes_fully_busy": busy_minutes,
        "minutes": len(timeline),
        "errors": sum(not r.get("ok", True) for r in recs),
        "timeline": timeline,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stats")
    ap.add_argument("--since", type=float)
    ap.add_argument("--until", type=float)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    rep = diagnose(load(a.stats, a.since, a.until), timeout=a.timeout)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(rep, open(a.out, "w"), indent=1)
    short = {k: v for k, v in rep.items() if k != "timeline"}
    print(json.dumps(short, indent=1))
    print("\nper minute (first 15 busy minutes): minute | batches | queries | busy_s | wait_p50 | wait_max")
    for t in [t for t in rep.get("timeline", []) if t["queries"]][:15]:
        print(f"  {t['minute']} | {t['batches']} | {t['queries']} | {t['busy_s']} | {t['wait_p50']} | {t['wait_max']}")


if __name__ == "__main__":
    main()
