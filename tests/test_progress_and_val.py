import importlib.util
import os

import pytest


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), "..", rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


progress = _load("progress", "scripts/progress.py")

LOG = """(TaskRunnerV1 pid=1) step:0 - val-core/searchR1_nq/reward/mean@1:0.10 - val-core/searchR1_hotpotqa/reward/mean@1:0.20
(TaskRunnerV1 pid=1) step:1 - critic/score/mean:0.05 - timing_s/step:300.0 - response_length/mean:400.0
(TaskRunnerV1 pid=1) step:2 - critic/score/mean:0.15 - timing_s/step:240.0 - response_length/mean:420.0
"""


def test_progress_line(tmp_path):
    p = tmp_path / "train.log"
    p.write_text(LOG)
    line = progress.summarize(str(p), 300)
    assert line.startswith("step 2/300 | reward 0.100 (first 0.050) | val 0.150")
    assert "4.5 min/step" in line and "ETA 22.4 h" in line  # 298 steps x 270 s = 22.35 h


def test_progress_before_first_step(tmp_path):
    p = tmp_path / "train.log"
    p.write_text("loading...\n")
    assert progress.summarize(str(p), 300) == "no training step logged yet"


def test_val_subset(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    vs = _load("make_val_subset", "scripts/make_val_subset.py")

    def row(ds, q):
        return {"data_source": ds, "extra_info": {"question": q}, "reward_model": {"ground_truth": {"target": ["x"]}}}

    pq.write_table(pa.Table.from_pylist([row("searchR1_nq", "leaked q")]), tmp_path / "train.parquet")
    test_rows = [row("searchR1_nq", "Leaked Q?")] + [row("searchR1_nq", f"q{i}") for i in range(5)] + \
                [row("searchR1_nq", "q0")] + [row("searchR1_popqa", f"p{i}") for i in range(2)]
    pq.write_table(pa.Table.from_pylist(test_rows), tmp_path / "test.parquet")
    out = vs.build(str(tmp_path), per_ds=3)
    t = pq.read_table(out).to_pylist()
    qs = [r["extra_info"]["question"] for r in t]
    assert "Leaked Q?" not in qs                       # leaked removed
    assert sum(r["data_source"] == "searchR1_nq" for r in t) == 3
    assert sum(r["data_source"] == "searchR1_popqa" for r in t) == 2   # fewer than per_ds available
    assert len(set(qs)) == len(qs)                    # duplicates removed
    assert vs.build(str(tmp_path), per_ds=3) == out and pq.read_table(out).to_pylist() == t  # deterministic
