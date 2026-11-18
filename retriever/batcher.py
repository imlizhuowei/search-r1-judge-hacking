"""Single-threaded request batcher for the retrieval server (no torch/faiss imports here)."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
from concurrent.futures import Future


def query_key(q: str) -> str:
    """Short hash of a lower-cased, whitespace-collapsed query: enough to count repeats without logging text."""
    return hashlib.md5(" ".join(q.lower().split()).encode()).hexdigest()[:10]


class SearchBatcher:
    """Single worker thread owns torch + faiss.

    Concurrent HTTP requests used to call encode()/index.search() from many threads at once; torch and faiss
    each bring their own OpenMP runtime and that combination segfaulted the server mid-training (11-04).
    Now requests are queued, the worker merges whatever arrived within `wait_ms` (up to `max_queries`) into
    one batched search, and hands results back. Batched flat search is also much faster than many small ones.
    """

    def __init__(self, retriever, max_queries: int = 256, wait_ms: float = 5.0, stats_path: str | None = None):
        self.retriever = retriever
        self.max_queries = max_queries
        self.wait_s = wait_ms / 1000
        # One JSON line per batch (time split, queue wait, query hashes) for the timing measurement (LOG 11-18).
        # Off unless a path is given (env RETRIEVER_STATS in launch.sh).
        self.stats_path = stats_path
        self.q: "queue.Queue[tuple[list[str], int, Future, float]]" = queue.Queue()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, queries: list[str], topk: int) -> Future:
        fut: Future = Future()
        self.q.put((queries, topk, fut, time.time()))
        return fut

    def _loop(self):
        while True:
            first = self.q.get()
            items = [first]
            n = len(first[0])
            deadline = time.time() + self.wait_s
            while n < self.max_queries:
                try:
                    item = self.q.get(timeout=max(0.0, deadline - time.time()))
                except queue.Empty:
                    break
                items.append(item)
                n += len(item[0])
            # group by topk (normally all the same)
            by_k: dict[int, list] = {}
            for it in items:
                by_k.setdefault(it[1], []).append(it)
            for k, group in by_k.items():
                flat = [q for qs, _, _, _ in group for q in qs]
                t0 = time.time()
                try:
                    res, scores = self.retriever.batch_search(query_list=flat, num=k, return_score=True)
                    i = 0
                    for qs, _, fut, _ in group:
                        fut.set_result((res[i : i + len(qs)], scores[i : i + len(qs)]))
                        i += len(qs)
                    ok = True
                except Exception as e:  # never let the worker thread die
                    ok = False
                    for _, _, fut, _ in group:
                        if not fut.done():
                            fut.set_exception(e)
                if self.stats_path:
                    self._log(group, flat, t0, time.time(), ok)

    def _log(self, group, flat, t0, t1, ok):
        try:
            split = getattr(self.retriever, "last_stats", None) or {}
            rec = {
                "t": round(t0, 3),
                "n_req": len(group),
                "n_q": len(flat),
                "total_s": round(t1 - t0, 4),
                **{k: round(v, 4) for k, v in split.items()},
                # how long each request sat in the queue before its batch started (= time a rollout waited)
                "wait_s": [round(t0 - it[3], 4) for it in group],
                "keys": [query_key(q) for q in flat],
                "ok": ok,
            }
            os.makedirs(os.path.dirname(os.path.abspath(self.stats_path)), exist_ok=True)
            with open(self.stats_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass  # stats must never break retrieval
