"""Summarise a verl console log + on_boot stage times into results/timing_<job>.json and a short text line.

verl's console logger prints one line per step like
  step:1 - timing_s/gen:123.4 - timing_s/update_actor:56.7 - response_length/mean:812.3 - ...
We keep every timing_s/* key, plus response length / turns / reward, averaged over steps (step 1 is
reported separately because it includes warm-up), and the stage timestamps written by on_boot.sh.

Usage: python scripts/summarize_timing.py <verl_log> <stages_file> <job> [extra key=value ...]
"""

from __future__ import annotations

import json
import os
import re
import sys
from statistics import mean

NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
# values may be printed bare or wrapped, e.g. "num_turns/mean:np.float64(5.125)"
KV = re.compile(rf"([A-Za-z0-9_./@-]+):(?:np\.\w+\()?({NUM})\)?(?=\s|$)")
KEEP = ("timing_s/", "timing_per_token_ms/", "response_length/", "prompt_length/", "num_turns",
        "critic/score/mean", "critic/rewards/mean", "perf/throughput", "perf/max_memory")


def parse_steps(path: str) -> list[dict]:
    steps = []
    with open(path, errors="ignore") as f:
        for line in f:
            if "step:" not in line or "timing_s/" not in line:
                continue
            kv = {k: float(v) for k, v in KV.findall(line)}
            if "step" in kv:
                steps.append({k: v for k, v in kv.items() if k == "step" or k.startswith(KEEP)})
    return steps


def parse_stages(path: str) -> dict:
    out = {}
    if os.path.exists(path):
        for line in open(path):
            name, _, ts = line.strip().partition(" ")
            if ts:
                out[name] = int(ts)
    return out


def summarize(log, stages_file, job, extra):
    steps = parse_steps(log) if os.path.exists(log) else []
    stages = parse_stages(stages_file)
    t0 = stages.get("boot")
    rep = {"job": job, **extra,
           "stage_minutes_since_boot": {k: round((v - t0) / 60, 1) for k, v in stages.items()} if t0 else {},
           "n_steps_logged": len(steps)}
    if steps:
        rep["first_step"] = steps[0]
        later = steps[1:] or steps
        keys = sorted({k for s in later for k in s if k != "step"})
        rep["mean_after_first"] = {k: round(mean(s[k] for s in later if k in s), 3) for k in keys}
    os.makedirs("results", exist_ok=True)
    out = f"results/timing_{job}.json"
    with open(out, "w") as f:
        json.dump(rep, f, indent=2)
    m = rep.get("mean_after_first", {})
    step_s = m.get("timing_s/step") or (rep.get("first_step") or {}).get("timing_s/step")
    line = (f"{job}: {len(steps)} steps, step≈{step_s:.0f}s" if step_s else f"{job}: no step timing found") + \
        (f", gen≈{m['timing_s/gen']:.0f}s" if "timing_s/gen" in m else "") + \
        (f", resp_len≈{m['response_length/mean']:.0f}" if "response_length/mean" in m else "")
    print(line)
    return rep, line


if __name__ == "__main__":
    extra = dict(a.split("=", 1) for a in sys.argv[4:])
    summarize(sys.argv[1], sys.argv[2], sys.argv[3], extra)
