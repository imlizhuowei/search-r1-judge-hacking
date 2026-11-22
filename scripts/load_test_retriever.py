"""Load test for the retrieval server: does it keep up with training-level concurrency? (LOG 11-22)

N clients send real test questions at the same time, each like a rollout's tool call (1-4 queries per request, one
request at a time, next one right after the answer). Reports error rate, latency percentiles and throughput.
Pass = no errors and p99 latency under the client timeout (configs/search_tool.yaml, 120 s).

Training: 64 prompts x 5 rollouts = 320 rollouts searching at once, the tool caps in-flight requests (max_concurrency).
  python3 scripts/load_test_retriever.py --clients 64 --seconds 300
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time

import requests


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))], 3)


def load_questions(path: str, n: int = 5000, seed: int = 0) -> list[str]:
    import pyarrow.parquet as pq

    qs = [(r or {}).get("question", "") for r in pq.read_table(path, columns=["extra_info"]).column(0).to_pylist()]
    qs = [q for q in qs if q]
    random.Random(seed).shuffle(qs)
    return qs[:n]


def run(post, questions: list[str], clients: int, seconds: float, timeout: float, max_q: int = 4, seed: int = 0):
    lat, errs, nq = [], [], [0]
    lock = threading.Lock()
    stop = time.time() + seconds

    def client(i):
        rng = random.Random(seed + i)
        while time.time() < stop:
            qs = [rng.choice(questions) for _ in range(rng.randint(1, max_q))]
            t0 = time.time()
            try:
                r = post(qs, timeout)
                ok = len(r.get("result", [])) == len(qs)
                err = None if ok else "wrong number of results"
            except Exception as e:
                err = type(e).__name__
            with lock:
                if err:
                    errs.append(err)
                else:
                    lat.append(time.time() - t0)
                    nq[0] += len(qs)

    t0 = time.time()
    ths = [threading.Thread(target=client, args=(i,)) for i in range(clients)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.time() - t0
    n = len(lat) + len(errs)
    rep = {"clients": clients, "seconds": round(wall, 1), "requests": n, "errors": len(errs),
           "error_rate": round(len(errs) / n, 4) if n else None, "error_kinds": sorted(set(errs)),
           "queries_per_s": round(nq[0] / wall, 2),
           "latency_s": {"p50": pct(lat, 50), "p90": pct(lat, 90), "p99": pct(lat, 99), "max": pct(lat, 100)},
           "timeout_s": timeout}
    rep["pass"] = n > 0 and not errs and (rep["latency_s"]["p99"] or 0) < timeout
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/retrieve")
    ap.add_argument("--questions", default="/root/autodl-tmp/data/searchR1_processed_direct/test.parquet")
    ap.add_argument("--clients", type=int, default=64)
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--out", default="results/retriever_load_test.json")
    a = ap.parse_args(argv)

    def post(qs, timeout):
        r = requests.post(a.url, json={"queries": qs, "topk": 3, "return_scores": True}, timeout=timeout)
        r.raise_for_status()
        return r.json()

    rep = run(post, load_questions(a.questions), a.clients, a.seconds, a.timeout)
    rep["url"] = a.url
    print(json.dumps(rep, indent=1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
