"""How fast and how exact is GPU brute-force search (retriever/gpu_flat.py), and how much GPU memory does it take?
(LOG 11-24: CPU faiss Flat maxes out at ~7.5 queries/s; option B = the same exact search on the A800.)

While the vectors are copied to the GPU, N real test questions are scored exactly in fp32 against every chunk:
that is what the CPU Flat index returns. Then, per batch size: ms per query and top-k agreement with it.

  /root/autodl-tmp/envs/retriever/bin/python retriever/bench_gpu_search.py
Stop the retriever and anything else on the GPU first. ~10 min (reading 64 GB from file storage dominates).
Report: results/bench_gpu_search.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_flat import GpuFlatIndex  # noqa: E402


def overlap(found: np.ndarray, exact: np.ndarray) -> tuple[float, float]:
    k = exact.shape[1]
    return (float(np.mean([len(set(found[r]) & set(exact[r])) / k for r in range(len(exact))])),
            float(np.mean(found[:, 0] == exact[:, 0])))


def gpu_mem_used_mib() -> str | None:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="/root/autodl-fs/searchr1/e5_Flat.index")
    ap.add_argument("--model", default="/root/autodl-tmp/models/e5-base-v2")
    ap.add_argument("--questions", default="/root/autodl-tmp/data/searchR1_processed_direct/test.parquet")
    ap.add_argument("--queries-npy", default=None, help="precomputed query embeddings (tests)")
    ap.add_argument("--n-queries", type=int, default=1024)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--batches", default="1,16,64,256,1024")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk-rows", type=int, default=1_000_000)
    ap.add_argument("--vectors-npy", default=None, help="tests: vectors instead of --index")
    ap.add_argument("--out", default="results/bench_gpu_search.json")
    a = ap.parse_args(argv)

    if a.queries_npy:
        q = np.load(a.queries_npy).astype(np.float32)
    else:
        from convert_index_fp16 import encode_questions
        q = encode_questions(a.model, a.questions, a.n_queries)
    print(f"[queries] {q.shape}", flush=True)
    dev = torch.device(a.device)
    qg = torch.as_tensor(q, device=dev)
    k = a.k
    state = {"s": None, "i": None}

    def exact_chunk(base, x32):                       # exact fp32 top-k, chunk by chunk = the CPU Flat answer
        s = qg @ x32.T
        cs, ci = torch.topk(s, min(k, s.shape[1]), dim=1)
        ci = ci + base
        if state["s"] is None:
            state["s"], state["i"] = cs, ci
        else:
            alls, alli = torch.cat([state["s"], cs], 1), torch.cat([state["i"], ci], 1)
            top = torch.topk(alls, k, dim=1).indices
            state["s"], state["i"] = alls.gather(1, top), alli.gather(1, top)

    t0 = time.time()
    vectors = np.load(a.vectors_npy) if a.vectors_npy else None
    index = GpuFlatIndex(a.index if vectors is None else None, device=a.device, chunk_rows=a.chunk_rows,
                         vectors=vectors, on_chunk=exact_chunk)
    load_s = time.time() - t0
    exact = state["i"].cpu().numpy()
    info = {"ntotal": index.ntotal, "d": index.d, "load_s": round(load_s), "device": a.device,
            "torch": torch.__version__,
            "index_gib": round(index.xb.numel() * 2 / 2**30, 1),
            "torch_allocated_gib": round(torch.cuda.memory_allocated(dev) / 2**30, 1) if dev.type == "cuda" else None,
            "nvidia_smi_used_total": gpu_mem_used_mib()}
    print(json.dumps(info), flush=True)

    rows = []
    for b in [int(x) for x in a.batches.split(",")]:
        b = min(b, len(q))
        for r in range(a.repeat):
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            ts = time.time()
            found = np.concatenate([index.search(q[i:i + b], k)[1] for i in range(0, len(q), b)])
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            dt = time.time() - ts
            ov, top1 = overlap(found, exact)
            row = {"batch": b, "rep": r, "ms_per_query": round(1000 * dt / len(q), 3),
                   "queries_per_s": round(len(q) / dt, 1), "topk_overlap": round(ov, 4), "top1_agree": round(top1, 4)}
            rows.append(row)
            print(row, flush=True)
    info["rows"] = rows
    if dev.type == "cuda":
        info["torch_peak_gib"] = round(torch.cuda.max_memory_allocated(dev) / 2**30, 1)
    best = max(rows, key=lambda r: r["queries_per_s"])
    info["best"] = best
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(info, open(a.out, "w"), indent=1)
    print(f"best: batch {best['batch']}, {best['queries_per_s']} queries/s, top-{k} overlap {best['topk_overlap']}; "
          f"index {info['index_gib']} GiB on GPU, peak {info.get('torch_peak_gib')} GiB")


if __name__ == "__main__":
    main()
