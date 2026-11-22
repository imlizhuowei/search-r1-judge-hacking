"""IVF index builder on a small synthetic Flat index (no model, no GPU)."""

import json
import os
import sys

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "retriever"))
from build_ivf_index import main, recall_at_k  # noqa: E402


def test_recall_at_k():
    exact = np.array([[1, 2, 3], [4, 5, 6]])
    assert recall_at_k(np.array([[1, 2, 3], [6, 9, 9]]), exact) == (pytest.approx(4 / 6), 0.5)


def test_build_ivf_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    # clustered unit vectors, like real embeddings (IVF works because neighbours share a cluster)
    centers = rng.normal(size=(50, 32)).astype(np.float32)
    x = centers[rng.integers(0, 50, 20000)] + 0.3 * rng.normal(size=(20000, 32)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    flat = faiss.IndexFlatIP(32)
    flat.add(x)
    src = str(tmp_path / "flat.index")
    faiss.write_index(flat, src)
    q = x[rng.choice(20000, 200, replace=False)] + 0.05 * rng.normal(size=(200, 32)).astype(np.float32)
    np.save(tmp_path / "q.npy", q)
    dst, rep = str(tmp_path / "ivf.index"), str(tmp_path / "rep.json")
    main(["--src", src, "--dst", dst, "--queries-npy", str(tmp_path / "q.npy"), "--nlist", "64",
          "--nprobes", "1,4,16,64", "--chunk", "3000", "--report", rep])
    r = json.load(open(rep))
    assert r["pass"] and os.path.exists(dst)
    assert r["sweep"][-1]["topk_overlap"] == 1.0          # probing every list = exact search
    assert r["sweep"][0]["topk_overlap"] <= r["sweep"][-1]["topk_overlap"]
    ivf = faiss.read_index(dst)
    assert ivf.ntotal == 20000 and ivf.nprobe == r["chosen_nprobe"]
    # exact top-k from the chunked pass equals the Flat index's own answer
    _, fi = flat.search(q, 3)
    ivf.nprobe = 64
    assert recall_at_k(ivf.search(q, 3)[1], fi)[0] == 1.0


def test_build_ivf_refuses_when_recall_too_low(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(5000, 16)).astype(np.float32)        # no cluster structure: IVF recall is poor
    flat = faiss.IndexFlatIP(16)
    flat.add(x)
    src = str(tmp_path / "flat.index")
    faiss.write_index(flat, src)
    np.save(tmp_path / "q.npy", rng.normal(size=(100, 16)).astype(np.float32))
    with pytest.raises(SystemExit):
        main(["--src", src, "--dst", str(tmp_path / "ivf.index"), "--queries-npy", str(tmp_path / "q.npy"),
              "--nlist", "64", "--nprobes", "1", "--min-recall", "0.99", "--report", str(tmp_path / "r.json")])
    # the index is kept (a 1-hour build is not thrown away); a wider sweep from the saved files passes
    assert (tmp_path / "ivf.index").exists()
    main(["--dst", str(tmp_path / "ivf.index"), "--sweep-only", "--nprobes", "1,64", "--min-recall", "0.99",
          "--report", str(tmp_path / "r.json")])
    r = json.load(open(tmp_path / "r.json"))
    assert r["pass"] and r["chosen_nprobe"] == 64 and r["sweep"][0]["topk_overlap"] < 0.99
