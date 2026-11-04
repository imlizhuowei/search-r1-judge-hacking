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


def test_progress_reads_numpy_wrapped_val(tmp_path):
    p = tmp_path / "train.log"
    p.write_text("step:0 - val-core/searchR1_nq/reward/mean@1:np.float64(0.3) - "
                 "val-core/searchR1_hotpotqa/reward/mean@1:np.float64(0.1)\n"
                 "step:1 - critic/score/mean:0.05 - timing_s/step:60.0 - num_turns/mean:np.float64(5.1)\n")
    line = progress.summarize(str(p), 10)
    assert "val 0.200" in line


def test_analyze_rollouts(tmp_path):
    import json
    ar = _load("analyze_rollouts", "scripts/analyze_rollouts.py")
    run = tmp_path / "run"
    (run / "rollouts").mkdir(parents=True)
    hit = "<tool_call>q</tool_call><tool_response>Doc 1 (Title: X) Thomas Jefferson</tool_response>"
    miss = "<tool_call>q</tool_call><tool_response>Doc 1 (Title: Y) Paris</tool_response>"
    gt = {"target": ["Thomas Jefferson"]}
    recs = [  # one group of 4: A, B1, C, D
        (hit + "<answer>Thomas Jefferson</answer>", 1.0),
        (hit + "<answer>John Adams</answer>", 0.0),
        (miss + "<answer>John Adams</answer>", 0.0),
        (miss + "<answer>Thomas Jefferson</answer>", 1.0),
    ]
    with open(run / "rollouts" / "5.jsonl", "w") as f:
        for out, sc in recs:
            f.write(json.dumps({"input": "Q1", "output": out, "gts": gt, "score": sc, "step": 5}) + "\n")
    rows = ar.main(str(run), every=1)
    r = rows[0]
    assert r["step"] == 5 and r["EM"] == 0.5 and r["ToolHit"] == 0.5 and r["SupCor"] == 0.25
    assert r["type_A"] == r["type_B1"] == r["type_C"] == r["type_D"] == 0.25
    assert r["delta_mean"] == 0.0 and r["tool_response_tag_rate"] == 1.0
    assert (run / "analysis" / "grounding_by_step.csv").exists()
