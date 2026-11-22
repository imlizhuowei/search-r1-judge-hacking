"""Time the Flat index directly, without the server / HTTP / batcher: is faiss itself slow, and with which threads?
(LOG 11-22: load test with 14 threads served 0.07 queries/s, one 101-query batch took 308 s; on 11-19 with 112
threads batches took ~15 s. Something other than thread count, or thread count in another library, is going on.)

For each thread count x batch size: wall time, CPU time, and how long the container was throttled by its CPU quota
(cgroup cpu.stat) during the search. Also times fetching passages (the corpus is read from file storage by offset).

  OMP_NUM_THREADS=14 /root/autodl-tmp/envs/retriever/bin/python retriever/bench_faiss.py
Needs the GPU-mode instance (64 GB RAM). Stop the retriever first. ~10 min to load the index, a few min to run.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cgroup(name: str) -> str | None:
    for p in (f"/sys/fs/cgroup/{name}", f"/sys/fs/cgroup/cpu/{name}", f"/sys/fs/cgroup/memory/{name}",
              f"/sys/fs/cgroup/cpu,cpuacct/{name}"):
        if os.path.exists(p):
            return open(p).read().strip()
    return None


def throttled_s() -> float | None:
    st = cgroup("cpu.stat")
    if not st:
        return None
    kv = dict(line.split() for line in st.splitlines() if len(line.split()) == 2)
    if "throttled_usec" in kv:
        return int(kv["throttled_usec"]) / 1e6
    if "throttled_time" in kv:
        return int(kv["throttled_time"]) / 1e9
    return None


def cpu_s() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="/root/autodl-fs/searchr1/e5_Flat.index")
    ap.add_argument("--threads", default="1,4,14,28,112")
    ap.add_argument("--batches", default="1,16,64,256")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--out", default="results/bench_faiss.json")
    a = ap.parse_args(argv)
    import faiss

    info = {"cpu_quota": cgroup("cpu.cfs_quota_us") or cgroup("cpu.max"), "cpu_period": cgroup("cpu.cfs_period_us"),
            "mem_limit": cgroup("memory.limit_in_bytes") or cgroup("memory.max"), "os_cpus": os.cpu_count(),
            "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "faiss": faiss.__version__}
    try:
        info["blas"] = faiss.get_compile_options()
    except Exception:
        pass
    print(json.dumps(info), flush=True)

    t = time.time()
    index = faiss.read_index(a.index)
    info["load_s"] = round(time.time() - t)
    print(f"[load] {index.ntotal:,} vectors in {info['load_s']} s", flush=True)
    rng = np.random.default_rng(0)
    q = rng.normal(size=(max(int(b) for b in a.batches.split(",")), index.d)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)

    rows = []
    for th in [int(x) for x in a.threads.split(",")]:
        faiss.omp_set_num_threads(th)
        for b in [int(x) for x in a.batches.split(",")]:
            for r in range(a.repeat):
                t0, c0, z0 = time.time(), cpu_s(), throttled_s()
                index.search(q[:b], 3)
                wall, cpu = time.time() - t0, cpu_s() - c0
                z1 = throttled_s()
                row = {"threads": th, "batch": b, "rep": r, "wall_s": round(wall, 3),
                       "ms_per_query": round(1000 * wall / b, 1), "cpu_s": round(cpu, 2),
                       "cores_used": round(cpu / wall, 1) if wall else None,
                       "throttled_s": round(z1 - z0, 2) if z0 is not None and z1 is not None else None}
                rows.append(row)
                print(row, flush=True)
    info["rows"] = rows
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(info, open(a.out, "w"), indent=1)
    best = min((r for r in rows if r["batch"] == max(x["batch"] for x in rows)), key=lambda r: r["wall_s"])
    print(f"fastest at batch {best['batch']}: {best['threads']} threads, {best['ms_per_query']} ms/query "
          f"= {1000 / best['ms_per_query']:.1f} queries/s")


if __name__ == "__main__":
    main()
