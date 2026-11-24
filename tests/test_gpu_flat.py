"""GPU brute-force search (retriever/gpu_flat.py) on CPU with small data: same answers as exact search."""

import json
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "retriever"))
from gpu_flat import GpuFlatIndex  # noqa: E402


def _data(n=5000, d=32, nq=50, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    q = rng.normal(size=(nq, d)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return x, q


def test_matches_exact_search_across_chunks():
    x, q = _data()
    idx = GpuFlatIndex(vectors=x, device="cpu", chunk_rows=700)        # several chunks, a ragged last one
    s, i = idx.search(q, 3)
    exact = np.argsort(-(q @ x.T), axis=1)[:, :3]
    assert (i == exact).mean() > 0.97                                  # fp16 storage may swap near-ties
    assert np.allclose(s, np.take_along_axis(q @ x.T, i, 1), atol=2e-3)
    assert s.shape == (50, 3) and (np.diff(s, axis=1) <= 1e-6).all()   # sorted best first


def test_on_chunk_sees_every_row_once():
    x, _ = _data(n=2345)
    seen = []
    GpuFlatIndex(vectors=x, device="cpu", chunk_rows=1000, on_chunk=lambda b, c: seen.append((b, len(c))))
    assert seen == [(0, 1000), (1000, 1000), (2000, 345)]


def test_bench_end_to_end(tmp_path):
    import bench_gpu_search as b
    x, q = _data(n=3000, nq=40)
    np.save(tmp_path / "x.npy", x)
    np.save(tmp_path / "q.npy", q)
    out = tmp_path / "r.json"
    b.main(["--vectors-npy", str(tmp_path / "x.npy"), "--queries-npy", str(tmp_path / "q.npy"), "--device", "cpu",
            "--chunk-rows", "1000", "--batches", "1,16", "--repeat", "1", "--out", str(out)])
    r = json.load(open(out))
    assert r["ntotal"] == 3000 and all(row["topk_overlap"] > 0.95 for row in r["rows"])
