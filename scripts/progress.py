"""One-line progress summary from a verl console log (for phone pushes).

  python scripts/progress.py <log> <total_steps>
  -> "step 37/300 | reward 0.21 (first 0.05) | val 0.18 | 4.3 min/step | ETA 18.8 h"
"""

from __future__ import annotations

import re
import sys

NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
# values may be printed bare or wrapped, e.g. "num_turns/mean:np.float64(5.125)"
KV = re.compile(rf"([A-Za-z0-9_./@-]+):(?:np\.\w+\()?({NUM})\)?(?=\s|$)")


def summarize(path: str, total: int) -> str:
    steps, val = [], None
    with open(path, errors="ignore") as f:
        for line in f:
            if "step:" in line and "timing_s/step" in line:
                kv = {k: float(v) for k, v in KV.findall(line)}
                steps.append(kv)
            m = re.findall(rf"val-core/[^:]+/(?:reward|score|acc)[^:]*:(?:np\.\w+\()?({NUM})", line)
            if m:
                vals = [float(x) for x in m]
                val = sum(vals) / len(vals)
    if not steps:
        return "no training step logged yet" + (f" | val {val:.3f}" if val is not None else "")
    last = steps[-1]
    recent = steps[-5:]
    sec = sum(s.get("timing_s/step", 0) for s in recent) / len(recent)
    n = int(last.get("step", len(steps)))
    eta_h = max(total - n, 0) * sec / 3600
    r_first = steps[0].get("critic/score/mean")
    r_last = sum(s.get("critic/score/mean", 0) for s in recent) / len(recent)
    out = f"step {n}/{total} | reward {r_last:.3f}"
    if r_first is not None:
        out += f" (first {r_first:.3f})"
    if val is not None:
        out += f" | val {val:.3f}"
    out += f" | {sec / 60:.1f} min/step | ETA {eta_h:.1f} h"
    if "response_length/mean" in last:
        out += f" | len {last['response_length/mean']:.0f}"
    return out


if __name__ == "__main__":
    print(summarize(sys.argv[1], int(sys.argv[2])))
