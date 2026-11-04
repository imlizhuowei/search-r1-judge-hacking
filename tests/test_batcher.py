import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "retriever"))
from batcher import SearchBatcher  # noqa: E402


class FakeRetriever:
    """Records which thread calls it and how big each batch is; fails loudly if called concurrently."""

    def __init__(self):
        self.threads, self.batches, self.active = set(), [], 0
        self.lock = threading.Lock()

    def batch_search(self, query_list, num, return_score):
        with self.lock:
            self.active += 1
            assert self.active == 1, "called concurrently"
        self.threads.add(threading.get_ident())
        self.batches.append(len(query_list))
        res = [[{"document": {"contents": f"T\n{q}"}}] * num for q in query_list]
        sc = [[0.5] * num for _ in query_list]
        with self.lock:
            self.active -= 1
        return res, sc


def test_concurrent_requests_are_serialized_and_routed_correctly():
    fr = FakeRetriever()
    b = SearchBatcher(fr, max_queries=64, wait_ms=20)
    out = {}

    def client(i):
        res, sc = b.submit([f"q{i}a", f"q{i}b"], 3).result(timeout=10)
        out[i] = res

    ts = [threading.Thread(target=client, args=(i,)) for i in range(40)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(fr.threads) == 1                       # only the worker thread touched the retriever
    assert sum(fr.batches) == 80 and len(fr.batches) < 40   # requests were merged into fewer batches
    for i in range(40):                               # every client got its own results back
        assert [r[0]["document"]["contents"] for r in out[i]] == [f"T\nq{i}a", f"T\nq{i}b"]


def test_error_is_returned_not_fatal():
    class Boom(FakeRetriever):
        def batch_search(self, *a, **k):
            raise RuntimeError("faiss exploded")

    b = SearchBatcher(Boom(), wait_ms=1)
    try:
        b.submit(["x"], 3).result(timeout=5)
        assert False
    except RuntimeError as e:
        assert "exploded" in str(e)
    assert b.thread.is_alive()                        # worker survives
