import importlib.util
import json
import os

spec = importlib.util.spec_from_file_location(
    "summarize_timing", os.path.join(os.path.dirname(__file__), "..", "scripts", "summarize_timing.py"))
st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st)

LOG = """(TaskRunner pid=1) step:1 - timing_s/gen:200.5 - timing_s/update_actor:80.0 - timing_s/step:300.0 - response_length/mean:900.0 - actor/lr:1e-06
noise line
(TaskRunner pid=1) step:2 - timing_s/gen:100.0 - timing_s/update_actor:40.0 - timing_s/step:150.0 - response_length/mean:800.0
(TaskRunner pid=1) step:3 - timing_s/gen:110.0 - timing_s/update_actor:44.0 - timing_s/step:170.0 - response_length/mean:820.0
"""


def test_summarize(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "smoke.log").write_text(LOG)
    (tmp_path / "stages").write_text("boot 1000\nretriever_ready 1600\nsmoke_start 1700\nsmoke_end 2300\n")
    rep, line = st.summarize("smoke.log", "stages", "smoke", {"gpu": "A800"})
    assert rep["n_steps_logged"] == 3
    assert rep["first_step"]["timing_s/step"] == 300.0                  # warm-up step kept separately
    assert rep["mean_after_first"]["timing_s/step"] == 160.0            # mean of steps 2-3
    assert "actor/lr" not in rep["mean_after_first"]
    assert rep["stage_minutes_since_boot"]["retriever_ready"] == 10.0
    assert rep["gpu"] == "A800"
    assert "step≈160s" in line
    assert json.load(open("results/timing_smoke.json"))["job"] == "smoke"
