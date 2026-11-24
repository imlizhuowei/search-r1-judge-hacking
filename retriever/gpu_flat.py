"""Exact (brute-force) inner-product search on the GPU with plain PyTorch, as a drop-in for the faiss Flat index.

Why (LOG 11-24): faiss Flat on the container's 14 CPUs tops out at ~7.5 queries/s (14 threads, batches of 256),
far below what training needs (~1000-3000 queries per step). The paper runs the retriever on GPUs. The installed
faiss is CPU-only, so instead of installing faiss-gpu this does the same arithmetic with torch:

  * the 21M x 768 vectors are kept on the GPU in fp16 (~32 GB instead of 64 GB fp32);
  * a batch of queries is scored against every vector (a matrix multiply per chunk of 1M rows), top-`cand`
    candidates per chunk are kept, and the survivors are re-scored in fp32 so the final order does not depend on
    fp16 rounding of the scores. What remains is fp16 rounding of the stored vectors (relative ~1e-3);
    retriever/bench_gpu_search.py measures how often that changes the top-k vs the exact fp32 answer.

Same call as faiss: scores, ids = index.search(queries_float32_numpy, k).
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_index_fp16 import flat_layout  # noqa: E402


class GpuFlatIndex:
    def __init__(self, path: str | None = None, device: str = "cuda", chunk_rows: int = 1_000_000,
                 vectors: np.ndarray | None = None, on_chunk=None, cand: int = 32):
        """Load the Flat index file (or `vectors`) onto the GPU in fp16, chunk by chunk.
        on_chunk(base, fp32_chunk_on_gpu) is called for every chunk while it is still fp32 (used for exact checks)."""
        self.device = torch.device(device)
        self.cand = cand
        if vectors is None:
            self.d, self.ntotal, metric, off = flat_layout(path)
            if metric != 0:          # faiss METRIC_INNER_PRODUCT == 0
                raise ValueError(f"only inner-product Flat indexes are supported (metric={metric})")
            src = np.memmap(path, dtype=np.float32, mode="r", offset=off, shape=(self.ntotal, self.d))
        else:
            src = vectors
            self.ntotal, self.d = vectors.shape
        self.xb = torch.empty((self.ntotal, self.d), dtype=torch.float16, device=self.device)
        self.chunk_rows = chunk_rows
        t0 = time.time()
        for base in range(0, self.ntotal, chunk_rows):
            x = torch.from_numpy(np.ascontiguousarray(src[base:base + chunk_rows])).to(self.device)
            if on_chunk is not None:
                on_chunk(base, x)
            self.xb[base:base + len(x)] = x.half()
            del x
            if (base // chunk_rows) % 5 == 0:
                print(f"[gpu-flat] {min(base + chunk_rows, self.ntotal):,}/{self.ntotal:,} on {device} "
                      f"{time.time() - t0:.0f}s", flush=True)
        torch.cuda.synchronize(self.device) if self.device.type == "cuda" else None

    @torch.no_grad()
    def search(self, q: np.ndarray, k: int):
        q32 = torch.as_tensor(np.ascontiguousarray(q, dtype=np.float32), device=self.device)
        q16 = q32.half()
        cand = max(self.cand, k)
        best_s, best_i = None, None
        for base in range(0, self.ntotal, self.chunk_rows):
            xc = self.xb[base:base + self.chunk_rows]
            s = q16 @ xc.T                                        # (nq, rows) fp16 scores
            c = min(cand, s.shape[1])
            cs, ci = torch.topk(s, c, dim=1)
            ci = ci + base
            if best_s is None:
                best_s, best_i = cs, ci
            else:
                alls, alli = torch.cat([best_s, cs], 1), torch.cat([best_i, ci], 1)
                top = torch.topk(alls, min(cand, alls.shape[1]), dim=1).indices
                best_s, best_i = alls.gather(1, top), alli.gather(1, top)
        # re-score the candidates in fp32 so the final order is not decided by fp16 score rounding
        v = self.xb[best_i].float()                               # (nq, cand, d)
        exact = torch.einsum("qd,qcd->qc", q32, v)
        top = torch.topk(exact, k, dim=1)
        ids = best_i.gather(1, top.indices)
        return top.values.cpu().numpy(), ids.cpu().numpy()
