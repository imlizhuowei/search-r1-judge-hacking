"""Eval / badcase / timing-measurement tools (LOG 11-18), on fake dumps. No GPU, no verl."""

import json
import os
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "retriever"))

import badcase  # noqa: E402
import make_strata  # noqa: E402
import split_test  # noqa: E402
import summarize_eval  # noqa: E402
import summarize_retriever  # noqa: E402
from batcher import SearchBatcher, query_key  # noqa: E402
from gca.reward import compute_score  # noqa: E402

CALL = '<tool_call>{"name": "search", "arguments": {"query_list": ["q"]}}</tool_call>'
HIT = "<tool_response>\nDoc 1\nDrafted by Thomas Jefferson.\n</tool_response>"
MISS = "<tool_response>\nDoc 1\nCapital of France.\n</tool_response>"
GOLD = {"target": ["Thomas Jefferson"]}

CASES = {
    "correct_grounded": f"{CALL}\n{HIT}\n<answer>Thomas Jefferson</answer>",
    "correct_unsupported": f"{CALL}\n{MISS}\n<answer>Thomas Jefferson</answer>",
    "no_answer": f"{CALL}\n{HIT}\n{CALL}\n{MISS}",
    "fabricated": f"<think>x</think>{HIT}\n<answer>John Adams</answer>",
    "no_search": "<answer>John Adams</answer>",
    "retrieval_miss": f"{CALL}\n{MISS}\n<answer>John Adams</answer>",
    "near_miss": f"{CALL}\n{HIT}\n<answer>Thomas Jefferson of Virginia</answer>",
    "evidence_ignored": f"{CALL}\n{HIT}\n<answer>John Adams</answer>",
}


def test_badcase_categories():
    for want, out in CASES.items():
        assert badcase.categorize(out, ["Thomas Jefferson"]) == want, want


def _dump(out_dir, ds, outputs, question="who wrote the declaration?"):
    os.makedirs(os.path.join(out_dir, ds), exist_ok=True)
    with open(os.path.join(out_dir, ds, "300.jsonl"), "w") as f:
        for o in outputs:
            r = compute_score(f"searchR1_{ds}", o, GOLD, {"question": question})
            f.write(json.dumps({"input": "p", "output": o, "gts": GOLD, "score": r["score"], "step": 300, **r}) + "\n")


def test_summarize_eval_and_strata(tmp_path):
    out = str(tmp_path / "eval_x")
    _dump(out, "nq", [CASES["correct_grounded"], CASES["retrieval_miss"]] * 2)          # EM 0.5
    _dump(out, "bamboogle", [CASES["correct_grounded"]], question="unique bamboogle q")  # EM 1.0
    strata = {"nq\twho wrote declaration": {"q_in_train": 1, "ans_seen": 1, "dup": 4},
              "bamboogle\tunique bamboogle q": {"q_in_train": 0, "ans_seen": 0, "dup": 1}}
    rep = summarize_eval.summarize(out, strata)
    nq = rep["per_dataset"]["nq"]
    assert nq["n"] == 4 and nq["EM"] == 0.5 and nq["ToolHit"] == 0.5 and nq["tool_calls_mean"] == 1
    assert nq["strata"]["q_in_train"]["n"] == 4 and nq["strata"]["clean"]["n"] == 0   # leaked -> not clean
    assert rep["per_dataset"]["bamboogle"]["strata"]["clean"] == {"n": 1, "EM": 1.0}
    assert rep["avg_EM_over_datasets"] is None          # only 2 of 7 datasets done

    res = str(tmp_path / "results")
    summarize_eval.main([out, "--tag", "x", "--strata", "/nonexistent", "--results", res])
    md = summarize_eval.table(res)
    assert "ours: x" in md and "paper: Search-R1-instruct GRPO" in md and "0.336" in md


def test_badcase_report(tmp_path):
    out = str(tmp_path / "eval_y")
    _dump(out, "nq", list(CASES.values()))
    rep, picked = badcase.analyze(out)
    assert rep["n"] == 8 and all(rep["total"][c] == 1 for c in badcase.CATEGORIES)
    md = badcase.to_markdown(rep, picked, "y")
    assert "| evidence_ignored | 1 | 12.5% | 16.7% |" in md and "WARNING" not in md


def test_badcase_warns_when_tool_response_tags_missing(tmp_path):
    out = str(tmp_path / "eval_z")
    _dump(out, "nq", [f"{CALL}\nDrafted by Thomas Jefferson.\n<answer>John Adams</answer>"] * 3)
    rep, picked = badcase.analyze(out)
    assert rep["tool_response_tag_rate_when_searched"] == 0
    assert "WARNING" in badcase.to_markdown(rep, picked, "z")


def test_split_and_strata(tmp_path):
    def rows(items):
        return pa.Table.from_pylist([
            {"data_source": f"searchR1_{ds}", "extra_info": {"question": q},
             "reward_model": {"ground_truth": {"target": [a]}}} for ds, q, a in items])
    pq.write_table(rows([("nq", "Q one", "A"), ("hotpotqa", "Q two", "B")]), tmp_path / "train.parquet")
    pq.write_table(rows([("nq", "Q one", "A"), ("nq", "Q three", "Z"), ("nq", "Q three", "Z"),
                         ("bamboogle", "Q four", "B")]), tmp_path / "test.parquet")
    assert split_test.split(str(tmp_path)) == {"bamboogle": 1, "nq": 3}
    assert pq.read_table(tmp_path / "test_by_ds" / "nq.parquet").num_rows == 3
    s = make_strata.build(str(tmp_path))
    assert s["nq\tq one"] == {"q_in_train": 1, "ans_seen": 1, "dup": 1}
    assert s["nq\tq three"] == {"q_in_train": 0, "ans_seen": 0, "dup": 2}
    assert s["bamboogle\tq four"]["ans_seen"] == 1


class _FakeRetriever:
    def batch_search(self, query_list, num, return_score):
        self.last_stats = {"encode_s": 0.001, "search_s": 0.002, "docs_s": 0.0005}
        return [[{"document": {"contents": q}}] * num for q in query_list], [[0.5] * num for _ in query_list]


def test_batcher_writes_stats_and_summary(tmp_path):
    path = str(tmp_path / "stats" / "s.jsonl")
    b = SearchBatcher(_FakeRetriever(), wait_ms=1, stats_path=path)
    for qs in (["Who wrote it?", "who  wrote it?"], ["capital of France"]):
        b.submit(qs, 3).result(timeout=5)
    import time
    for _ in range(50):
        if os.path.exists(path) and len(open(path).readlines()) == 2:
            break
        time.sleep(0.02)
    recs = summarize_retriever.load(path)
    assert len(recs) == 2 and recs[0]["keys"][0] == recs[0]["keys"][1] == query_key("who wrote it?")
    rep = summarize_retriever.summarize(recs)
    assert rep["n_queries"] == 3 and rep["unique_queries"] == 2 and rep["repeat_rate"] == round(1 - 2 / 3, 4)
    assert rep["time_split_s"]["search_s"] == 0.0 or rep["time_split_s"]["search_s"] >= 0
    assert rep["queue_wait_s"]["p50"] is not None

    log = tmp_path / "train.log"
    log.write_text("step:301 - timing_s/gen:100.0 - timing_s/update_actor:40.0 - timing_s/step:150.0\n")
    rep2 = summarize_retriever.with_train_log(dict(rep, busy_s=30.0), str(log))
    assert rep2["train"]["retriever_busy_over_gen"] == 0.3 and rep2["train"]["gen_share_of_step"] == round(100 / 150, 4)


def test_shell_scripts_parse():
    for s in ("scripts/eval.sh", "scripts/timing_probe.sh", "scripts/train.sh", "scripts/on_boot.sh",
              "retriever/launch.sh"):
        subprocess.run(["bash", "-n", os.path.join(ROOT, s)], check=True)
