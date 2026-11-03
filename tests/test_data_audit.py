import importlib.util
import os

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

spec = importlib.util.spec_from_file_location(
    "data_audit", os.path.join(os.path.dirname(__file__), "..", "scripts", "data_audit.py"))
da = importlib.util.module_from_spec(spec)
spec.loader.exec_module(da)


def _write(path, rows):
    pq.write_table(pa.Table.from_pylist([
        {"data_source": f"searchR1_{ds}", "extra_info": {"question": q},
         "reward_model": {"ground_truth": {"target": golds}}} for ds, q, golds in rows
    ]), path)


def test_audit_counts_dups_and_leakage(tmp_path):
    _write(tmp_path / "train.parquet", [
        ("nq", "Who wrote X?", ["Alice"]),
        ("nq", "who wrote x", ["Alice"]),            # duplicate after normalisation
        ("hotpotqa", "Where was Y born?", ["Paris"]),
    ])
    _write(tmp_path / "test.parquet", [
        ("nq", "Who wrote X?", ["Alice"]),           # leaked question
        ("nq", "Who painted Z?", ["Bob"]),
        ("musique", "Capital of W?", ["Paris"]),     # answer seen in train, question not
    ])
    r = da.audit(str(tmp_path))
    assert r["counts"]["train"] == {"nq": 2, "hotpotqa": 1}
    assert r["duplicate_questions_within_split"]["train"] == 1
    assert r["test_question_in_train"]["nq"] == {"n": 1, "frac": 0.5}
    assert r["test_question_in_train"]["musique"]["n"] == 0
    assert r["test_answer_seen_as_train_answer"]["musique"] == 1.0
    assert r["hop"]["musique"] == "multi" and r["hop"]["nq"] == "single"
    assert "| nq | single | 2 | 2 | 1 (50.00%)" in da.to_markdown(r)
