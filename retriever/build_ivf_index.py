"""Build an IVF (approximate) index from the E5 Flat index, and measure what it costs in retrieval quality.

Why (LOG 11-22): under training / eval load the retriever served ~3.7 queries/s; 90% of its time was the Flat index's
brute-force scan of all 21M vectors on CPU. IVF clusters the vectors into `nlist` groups (k-means); a query is only
compared with the vectors in its `nprobe` nearest groups, so it scans ~nprobe/nlist of the corpus instead of all of it.
The price: a true top-3 passage that sits in a group we did not probe is missed. This script measures that price.

How (one pass, reads the Flat file through numpy.memmap like convert_index_fp16.py):
  1. train k-means centroids on a random sample of the vectors;
  2. add every vector in chunks; on each chunk also score N real test questions exactly (float32 inner product),
     keeping the exact top-k = what the Flat index returns;
  3. for each nprobe in --nprobes: search the questions, report recall (overlap with the exact top-k) and time.
Report: results/retriever_ivf_check.json. The index is written only if some nprobe reaches --min-recall; the chosen
nprobe (smallest that reaches it) is in the report and must be passed to the server (launch.sh: NPROBE=...).

  /root/autodl-tmp/envs/retriever/bin/python retriever/build_ivf_index.py
Needs the GPU-mode instance's RAM (~64 GB for the new index + working memory; the no-GPU mode has 2 GB).
Stop the retriever and training first.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_index_fp16 import encode_questions, flat_layout, merge_topk  # noqa: E402

DATA = "/root/autodl-fs/searchr1"


def recall_at_k(found: np.ndarray, exact: np.ndarray) -> tuple[float, float]:
    """(mean overlap of the top-k sets, share of queries whose top-1 matches)."""
    k = exact.shape[1]
    overlap = float(np.mean([len(set(found[r]) & set(exact[r])) / k for r in range(len(exact))]))
    top1 = float(np.mean(found[:, 0] == exact[:, 0]))
    return overlap, top1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=f"{DATA}/e5_Flat.index")
    ap.add_argument("--dst", default=f"{DATA}/e5_IVF.index")
    ap.add_argument("--model", default="/root/autodl-tmp/models/e5-base-v2")
    ap.add_argument("--questions", default="/root/autodl-tmp/data/searchR1_processed_direct/test.parquet")
    ap.add_argument("--queries-npy", default=None, help="precomputed query embeddings (tests)")
    ap.add_argument("--n-queries", type=int, default=1000)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--nlist", type=int, default=16384)
    ap.add_argument("--train-size", type=int, default=0, help="k-means sample; 0 = 50 * nlist (faiss wants >= 39)")
    ap.add_argument("--nprobes", default="8,16,32,64,128,256")
    ap.add_argument("--min-recall", type=float, default=0.95, help="top-k overlap the chosen nprobe must reach")
    ap.add_argument("--chunk", type=int, default=500_000)
    ap.add_argument("--report", default="results/retriever_ivf_check.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    import faiss

    t0 = time.time()
    print(f"[threads] faiss uses {faiss.omp_get_max_threads()} threads, os sees {os.cpu_count()} CPUs", flush=True)
    d, ntotal, metric, off = flat_layout(a.src)
    need = ntotal * d * 4 * 1.05 + ntotal * 8
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(a.dst))).free
    if free < need:
        sys.exit(f"not enough disk for {a.dst}: need {need / 2**30:.1f} GiB, free {free / 2**30:.1f} GiB")
    xb = np.memmap(a.src, dtype=np.float32, mode="r", offset=off, shape=(ntotal, d))

    q = np.load(a.queries_npy).astype(np.float32) if a.queries_npy else \
        encode_questions(a.model, a.questions, a.n_queries)
    print(f"[queries] {q.shape} after {time.time() - t0:.0f}s", flush=True)

    # 1. k-means on a random sample
    n_train = min(ntotal, a.train_size or 50 * a.nlist)
    rng = np.random.default_rng(a.seed)
    sample = np.sort(rng.choice(ntotal, size=n_train, replace=False))
    xt = np.ascontiguousarray(xb[sample])
    quantizer = faiss.IndexFlat(d, metric)
    index = faiss.IndexIVFFlat(quantizer, d, a.nlist, metric)
    tt = time.time()
    index.train(xt)
    del xt
    t_train = time.time() - tt
    print(f"[train] {a.nlist} centroids on {n_train:,} vectors in {t_train:.0f}s", flush=True)

    # 2. add everything; keep the exact top-k of the questions on the way (what the Flat index returns)
    best_s = np.full((len(q), 0), -np.inf, dtype=np.float32)
    best_i = np.zeros((len(q), 0), dtype=np.int64)
    ta = time.time()
    for base in range(0, ntotal, a.chunk):
        x = np.ascontiguousarray(xb[base:base + a.chunk])
        index.add(x)
        s = q @ x.T if metric == faiss.METRIC_INNER_PRODUCT else \
            -(np.sum(q * q, 1)[:, None] - 2 * q @ x.T + np.sum(x * x, 1)[None, :])
        best_s, best_i = merge_topk(best_s, best_i, s, base, a.k)
        if (base // a.chunk) % 10 == 0:
            print(f"[add] {base + len(x):,}/{ntotal:,}  {time.time() - t0:.0f}s", flush=True)
    del xb
    t_add = time.time() - ta
    sizes = np.array([index.invlists.list_size(i) for i in range(a.nlist)])

    # 3. recall / speed per nprobe (time is per 256 queries = the batcher's largest batch)
    sweep = []
    for nprobe in [int(x) for x in a.nprobes.split(",")]:
        index.nprobe = nprobe
        ts = time.time()
        _, fi = index.search(q, a.k)
        dt = time.time() - ts
        overlap, top1 = recall_at_k(fi, best_i)
        sweep.append({"nprobe": nprobe, "topk_overlap": round(overlap, 4), "top1_agree": round(top1, 4),
                      "s_per_256q": round(dt * 256 / len(q), 3), "ms_per_query": round(1000 * dt / len(q), 2)})
        print(f"[nprobe {nprobe}] overlap={overlap:.4f} top1={top1:.4f} {1000 * dt / len(q):.2f} ms/query", flush=True)
    ok = [r for r in sweep if r["topk_overlap"] >= a.min_recall]
    chosen = min(ok, key=lambda r: r["nprobe"]) if ok else None

    rep = {"src": a.src, "dst": a.dst, "ntotal": int(ntotal), "d": int(d), "k": a.k, "n_queries": int(len(q)),
           "nlist": a.nlist, "train_size": int(n_train), "threads": faiss.omp_get_max_threads(),
           "list_size": {"mean": round(float(sizes.mean()), 1), "max": int(sizes.max()),
                         "empty": int((sizes == 0).sum())},
           "train_s": round(t_train), "add_s": round(t_add), "sweep": sweep, "min_recall": a.min_recall,
           "chosen_nprobe": chosen["nprobe"] if chosen else None, "pass": chosen is not None,
           # for the write-up: the Flat index on the same machine served ~250 ms/query in batches (LOG 11-22)
           "seconds": round(time.time() - t0)}
    print(json.dumps(rep, indent=1), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    json.dump(rep, open(a.report, "w"), indent=1)
    if not chosen:
        sys.exit(f"FAIL: no nprobe in {a.nprobes} reaches top-{a.k} overlap {a.min_recall}; NOT writing the index")
    index.nprobe = chosen["nprobe"]
    tmp = a.dst + ".tmp"
    faiss.write_index(index, tmp)
    os.replace(tmp, a.dst)
    print(f"PASS -> wrote {a.dst} ({os.path.getsize(a.dst) / 2**30:.1f} GiB); start the server with "
          f"INDEX={a.dst} NPROBE={chosen['nprobe']}")


if __name__ == "__main__":
    main()
