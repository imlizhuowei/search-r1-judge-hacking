"""fp16 index conversion: header parsing, exact top-k, agreement check, refuses to write on failure."""

import json
import os
import sys

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "retriever"))
from convert_index_fp16 import flat_layout, main, merge_topk  # noqa: E402


def make(tmp_path, n=20000, d=64, metric="ip"):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, d)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    idx = faiss.IndexFlatIP(d) if metric == "ip" else faiss.IndexFlatL2(d)
    idx.add(x)
    src = str(tmp_path / "flat.index")
    faiss.write_index(idx, src)
    q = x[rng.choice(n, 200, replace=False)] + 0.3 * rng.standard_normal((200, d)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    np.save(tmp_path / "q.npy", q)
    return src, x, q


def test_layout_matches_faiss(tmp_path):
    src, x, _ = make(tmp_path)
    d, n, metric, off = flat_layout(src)
    mm = np.memmap(src, dtype=np.float32, mode="r", offset=off, shape=(n, d))
    assert (d, n, metric) == (64, 20000, faiss.METRIC_INNER_PRODUCT)
    assert np.array_equal(np.asarray(mm), x)


def test_merge_topk_equals_global():
    rng = np.random.default_rng(1)
    s = rng.standard_normal((5, 1000)).astype(np.float32)
    bs, bi = np.full((5, 0), -np.inf, np.float32), np.zeros((5, 0), np.int64)
    for b in range(0, 1000, 137):
        bs, bi = merge_topk(bs, bi, s[:, b:b + 137], b, 3)
    assert np.array_equal(bi, np.argsort(-s, axis=1)[:, :3])


@pytest.mark.parametrize("metric", ["ip", "l2"])
def test_convert_passes_and_writes(tmp_path, metric):
    src, x, q = make(tmp_path, metric=metric)
    dst, rep = str(tmp_path / "fp16.index"), str(tmp_path / "rep.json")
    main(["--src", src, "--dst", dst, "--queries-npy", str(tmp_path / "q.npy"), "--chunk", "3000", "--report", rep])
    r = json.load(open(rep))
    assert r["pass"] and r["topk_set_agree"] >= 0.99 and r["top1_agree"] >= 0.99
    out = faiss.read_index(dst)
    assert out.ntotal == 20000 and os.path.getsize(dst) < os.path.getsize(src) * 0.6
    # exact top-3 by brute force agrees with what the converter reported
    _, ref = faiss.read_index(src).search(q, 3)
    _, got = out.search(q, 3)
    assert np.mean([set(a) == set(b) for a, b in zip(ref, got)]) == r["topk_set_agree"]


def test_fail_does_not_write(tmp_path):
    src, _, _ = make(tmp_path)
    dst = str(tmp_path / "fp16.index")
    with pytest.raises(SystemExit, match="FAIL"):
        main(["--src", src, "--dst", dst, "--queries-npy", str(tmp_path / "q.npy"), "--min-agree", "1.01",
              "--report", str(tmp_path / "r.json")])
    assert not os.path.exists(dst)


def test_rejects_non_flat(tmp_path):
    idx = faiss.IndexHNSWFlat(8, 4)
    idx.add(np.random.rand(10, 8).astype(np.float32))
    p = str(tmp_path / "h.index")
    faiss.write_index(idx, p)
    with pytest.raises(ValueError, match="Flat only"):
        flat_layout(p)
