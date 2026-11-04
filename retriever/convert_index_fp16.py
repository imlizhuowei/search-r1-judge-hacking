"""Convert the E5 Flat (float32) faiss index to float16 storage (IndexScalarQuantizer QT_fp16), and check that
retrieval does not change.

Why: the Flat index is 21M x 768 x 4 B ~= 60 GiB of RAM; with the trainer (~50 GB) the 120 GB container sits at its
limit (11-04: OOM). fp16 halves it (~30 GiB). Search is still exact brute force; only the stored vectors are rounded.

How (one pass, low memory):
  * the Flat file is read through numpy.memmap (never loaded whole): after the header, IndexFlat stores
    ntotal*d float32 contiguously; the offset is derived and checked against the file size;
  * each chunk is (a) added to the fp16 index and (b) scored exactly (float32) against N real questions, keeping the
    exact top-k; afterwards the fp16 index is searched with the same questions and the two top-k lists are compared.

Report (results/retriever_fp16_check.json): top-1 agreement, top-k set agreement, mean overlap, max score diff.
PASS = top-k set agreement >= --min-agree (default 0.99).

  /root/autodl-tmp/envs/retriever/bin/python retriever/convert_index_fp16.py            # defaults below
Stop the retriever and training first: this needs ~32 GB for the new index plus a few GB of working memory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
import time

import numpy as np

DATA = "/root/autodl-fs/searchr1"


def flat_layout(path: str) -> tuple[int, int, int, int]:
    """Parse a faiss IndexFlat file header -> (d, ntotal, metric_type, data_offset_bytes)."""
    with open(path, "rb") as f:
        head = f.read(64)
    fourcc = head[:4]
    if fourcc not in (b"IxFI", b"IxF2"):
        raise ValueError(f"not a plain IndexFlat file (fourcc={fourcc!r}); this converter handles Flat only")
    d, ntotal, _, _ = struct.unpack_from("<iqqq", head, 4)          # d, ntotal, dummy, dummy
    off = 4 + 4 + 8 + 8 + 8
    off += 1                                                          # is_trained (bool)
    (metric,) = struct.unpack_from("<i", head, off)
    off += 4
    if metric > 1:
        off += 4                                                      # metric_arg (float)
    (n,) = struct.unpack_from("<Q", head, off)                        # vector length prefix
    off += 8
    # newer faiss: uint8 codes (n = ntotal*d*4); older: float xb (n = ntotal*d)
    if n not in (ntotal * d * 4, ntotal * d):
        raise ValueError(f"unexpected data length {n} for ntotal={ntotal} d={d}")
    size = os.path.getsize(path)
    if off + ntotal * d * 4 != size:
        raise ValueError(f"layout check failed: offset {off} + data {ntotal * d * 4} != file size {size}")
    return d, ntotal, metric, off


def merge_topk(best_s, best_i, s, base, k):
    """Running exact top-k (largest scores) over chunks. s: (q, m) scores for rows base..base+m."""
    m = s.shape[1]
    kk = min(k, m)
    part = np.argpartition(-s, kk - 1, axis=1)[:, :kk]
    cs = np.take_along_axis(s, part, axis=1)
    ci = part + base
    alls = np.concatenate([best_s, cs], axis=1)
    alli = np.concatenate([best_i, ci], axis=1)
    order = np.argsort(-alls, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(alls, order, axis=1), np.take_along_axis(alli, order, axis=1)


def encode_questions(model_path: str, parquet: str, n: int, seed: int = 0) -> np.ndarray:
    import pandas as pd

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from retrieval_server import Encoder  # same encoder as the server

    df = pd.read_parquet(parquet)
    qs = [(r or {}).get("question", "") for r in df["extra_info"]]
    qs = [q for q in qs if q]
    rng = np.random.default_rng(seed)
    qs = [qs[i] for i in rng.choice(len(qs), size=min(n, len(qs)), replace=False)]
    enc = Encoder("e5", model_path, "mean", 256, True)
    return np.concatenate([enc.encode(qs[i:i + 128]) for i in range(0, len(qs), 128)]).astype(np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=f"{DATA}/e5_Flat.index")
    ap.add_argument("--dst", default=f"{DATA}/e5_SQfp16.index")
    ap.add_argument("--model", default="/root/autodl-tmp/models/e5-base-v2")
    ap.add_argument("--questions", default="/root/autodl-tmp/data/searchR1_processed_direct/test.parquet")
    ap.add_argument("--queries-npy", default=None, help="precomputed query embeddings (tests)")
    ap.add_argument("--n-queries", type=int, default=1000)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=500_000)
    ap.add_argument("--min-agree", type=float, default=0.99)
    ap.add_argument("--report", default="results/retriever_fp16_check.json")
    a = ap.parse_args(argv)
    import faiss

    t0 = time.time()
    d, ntotal, metric, off = flat_layout(a.src)
    need = ntotal * d * 2 * 1.05
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(a.dst))).free
    if free < need:
        sys.exit(f"not enough disk for {a.dst}: need {need / 2**30:.1f} GiB, free {free / 2**30:.1f} GiB")
    xb = np.memmap(a.src, dtype=np.float32, mode="r", offset=off, shape=(ntotal, d))
    norms = np.linalg.norm(np.asarray(xb[:1000]), axis=1)
    print(f"[layout] d={d} ntotal={ntotal} metric={metric} offset={off}; first-1000 norm mean={norms.mean():.4f} "
          f"(e5 vectors are unit-norm, expect ~1)", flush=True)

    q = np.load(a.queries_npy).astype(np.float32) if a.queries_npy else \
        encode_questions(a.model, a.questions, a.n_queries)
    print(f"[queries] {q.shape} ready after {time.time() - t0:.0f}s", flush=True)
    sign = 1.0 if metric == faiss.METRIC_INNER_PRODUCT else -1.0   # L2: rank by -distance

    out = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_fp16, metric)
    best_s = np.full((len(q), 0), -np.inf, dtype=np.float32)
    best_i = np.zeros((len(q), 0), dtype=np.int64)
    t_gemm = 0.0
    for base in range(0, ntotal, a.chunk):
        x = np.ascontiguousarray(xb[base:base + a.chunk])
        out.add(x)
        tg = time.time()
        if metric == faiss.METRIC_INNER_PRODUCT:
            s = q @ x.T
        else:
            s = -(np.sum(q * q, 1)[:, None] - 2 * q @ x.T + np.sum(x * x, 1)[None, :])
        best_s, best_i = merge_topk(best_s, best_i, s, base, a.k)
        t_gemm += time.time() - tg
        if (base // a.chunk) % 10 == 0:
            print(f"[convert] {base + len(x):,}/{ntotal:,}  {time.time() - t0:.0f}s", flush=True)
    del xb

    ts = time.time()
    fs, fi = out.search(q, a.k)
    t_sq = time.time() - ts
    fs = fs * sign
    top1 = float(np.mean(fi[:, 0] == best_i[:, 0]))
    same_set = float(np.mean([set(fi[r]) == set(best_i[r]) for r in range(len(q))]))
    overlap = float(np.mean([len(set(fi[r]) & set(best_i[r])) / a.k for r in range(len(q))]))
    max_diff = float(np.max(np.abs(fs[:, 0] - best_s[:, 0])))
    rep = {"src": a.src, "dst": a.dst, "ntotal": int(ntotal), "d": int(d), "k": a.k, "n_queries": int(len(q)),
           "top1_agree": top1, "topk_set_agree": same_set, "topk_mean_overlap": overlap,
           "max_top1_score_diff": max_diff,
           # search cost per 256 queries (the batcher's max batch): float32 BLAS scan (= what the Flat index does)
           # vs the fp16 index. fp16 search has no BLAS path, so it is slower; this says by how much.
           "flat_s_per_256q": round(t_gemm * 256 / len(q), 2), "fp16_s_per_256q": round(t_sq * 256 / len(q), 2), "min_agree": a.min_agree, "pass": same_set >= a.min_agree,
           "seconds": round(time.time() - t0)}
    print(json.dumps(rep, indent=1), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    json.dump(rep, open(a.report, "w"), indent=1)
    if not rep["pass"]:
        sys.exit("FAIL: fp16 index changes retrieval too much; NOT writing it (keep using the Flat index)")
    tmp = a.dst + ".tmp"
    faiss.write_index(out, tmp)
    os.replace(tmp, a.dst)                      # never leave a half-written index under the real name
    print(f"PASS -> wrote {a.dst} ({os.path.getsize(a.dst) / 2**30:.1f} GiB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
