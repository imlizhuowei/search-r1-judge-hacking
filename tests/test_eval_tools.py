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
    assert split_test.split(str(tmp_path), sample=2) == {"bamboogle": 1, "nq": 2}
    a = pq.read_table(tmp_path / "test_by_ds_s2" / "nq.parquet").to_pylist()
    split_test.split(str(tmp_path), sample=2)
    assert pq.read_table(tmp_path / "test_by_ds_s2" / "nq.parquet").to_pylist() == a   # same seed, same questions
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


def test_subset_dump_reuses_full_eval(tmp_path):
    import subset_dump
    full = tmp_path / "full" / "nq"
    full.mkdir(parents=True)
    rows = [{"question": q, "score": 1.0} for q in ["a", "b", "b", "c"]]
    (full / "0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    samp = tmp_path / "s.parquet"
    pq.write_table(pa.Table.from_pylist([{"extra_info": {"question": q}} for q in ["b", "c"]]), samp)
    assert subset_dump.subset(str(full), str(samp), str(tmp_path / "out")) == 2
    assert len((tmp_path / "out" / "0.jsonl").read_text().splitlines()) == 2
    pq.write_table(pa.Table.from_pylist([{"extra_info": {"question": "zzz"}}]), samp)
    assert subset_dump.subset(str(full), str(samp), str(tmp_path / "out2")) == 0
    assert not (tmp_path / "out2").exists()


def test_diagnose_retriever_tells_queueing_from_crash():
    import diagnose_retriever as d
    # queueing: back-to-back 10 s batches, waits growing past 30 s, the same query coming back ~30 s later
    q = [{"t": 1000 + 10 * i, "n_req": 2, "n_q": 2, "total_s": 10.0, "wait_s": [5.0 * i, 5.0 * i],
          "keys": [f"k{i % 3}", f"x{i}"], "ok": True} for i in range(20)]
    rep = d.diagnose(q)
    assert rep["verdict"].startswith("queueing") and rep["n_gaps"] == 0 and rep["queries_repeated_within_window"] > 0
    # crash: short waits, but a 4-minute hole with no batches
    c = [{"t": 1000 + i, "n_req": 1, "n_q": 1, "total_s": 0.5, "wait_s": [0.1], "keys": [f"a{i}"], "ok": True}
         for i in range(10)]
    c += [{"t": 1250 + i, "n_req": 1, "n_q": 1, "total_s": 0.5, "wait_s": [0.1], "keys": [f"b{i}"], "ok": True}
          for i in range(10)]
    rep = d.diagnose(c)
    assert rep["verdict"].startswith("no overload") and rep["n_gaps"] == 1 and rep["gaps"][0]["gap_s"] > 200
    assert d.diagnose([])["verdict"] == "no data"
    # saturated server, tiny logged waits, ~40 requests per batch, lots of repeats -> queue before the batcher
    s = [{"t": 1000 + 15 * i, "n_req": 40, "n_q": 75, "total_s": 15.0, "wait_s": [0.01] * 40,
          "keys": [f"k{j % 30}" for j in range(75)], "ok": True} for i in range(40)]
    rep = d.diagnose(s)
    assert rep["verdict"].startswith("queueing before") and rep["requests_per_batch"] == 40
    assert rep["batch_time_by_size"][0]["queries_per_batch"] == "65-128"


def test_batcher_searches_duplicate_queries_once(tmp_path):
    class Counting(_FakeRetriever):
        seen = []

        def batch_search(self, query_list, num, return_score):
            Counting.seen.append(list(query_list))
            return super().batch_search(query_list, num, return_score)

    b = SearchBatcher(Counting(), wait_ms=50, stats_path=str(tmp_path / "s.jsonl"))
    f1, f2 = b.submit(["same q", "other"], 3), b.submit(["same q"], 3)
    r1, r2 = f1.result(timeout=5), f2.result(timeout=5)
    assert r1[0][0][0]["document"]["contents"] == "same q" and r1[0][1][0]["document"]["contents"] == "other"
    assert r2[0][0][0]["document"]["contents"] == "same q"
    assert sorted(q for batch in Counting.seen for q in batch) == ["other", "same q"]   # searched once each


def test_load_test_counts_errors_and_latency():
    import time as _t

    import load_test_retriever as lt

    def ok(qs, timeout):
        _t.sleep(0.01)
        return {"result": [[{}]] * len(qs)}

    rep = lt.run(ok, ["a", "b"], clients=4, seconds=0.3, timeout=1)
    assert rep["pass"] and rep["errors"] == 0 and rep["latency_s"]["p50"] >= 0.01

    def flaky(qs, timeout):
        raise TimeoutError()

    rep = lt.run(flaky, ["a"], clients=2, seconds=0.05, timeout=1)
    assert not rep["pass"] and rep["error_kinds"] == ["TimeoutError"]
