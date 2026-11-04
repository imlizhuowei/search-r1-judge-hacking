"""Single-threaded request batcher for the retrieval server (no torch/faiss imports here)."""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future


class SearchBatcher:
    """Single worker thread owns torch + faiss.

    Concurrent HTTP requests used to call encode()/index.search() from many threads at once; torch and faiss
    each bring their own OpenMP runtime and that combination segfaulted the server mid-training (11-04).
    Now requests are queued, the worker merges whatever arrived within `wait_ms` (up to `max_queries`) into
    one batched search, and hands results back. Batched flat search is also much faster than many small ones.
    """

    def __init__(self, retriever, max_queries: int = 256, wait_ms: float = 5.0):
        self.retriever = retriever
        self.max_queries = max_queries
        self.wait_s = wait_ms / 1000
        self.q: "queue.Queue[tuple[list[str], int, Future]]" = queue.Queue()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, queries: list[str], topk: int) -> Future:
        fut: Future = Future()
        self.q.put((queries, topk, fut))
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
                flat = [q for qs, _, _ in group for q in qs]
                try:
                    res, scores = self.retriever.batch_search(query_list=flat, num=k, return_score=True)
                    i = 0
                    for qs, _, fut in group:
                        fut.set_result((res[i : i + len(qs)], scores[i : i + len(qs)]))
                        i += len(qs)
                except Exception as e:  # never let the worker thread die
                    for _, _, fut in group:
                        if not fut.done():
                            fut.set_exception(e)
