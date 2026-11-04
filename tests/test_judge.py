"""Judge client + reward tests against a fake OpenAI-compatible server (no API key, no GPU)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from judge.client import JudgeClient
from judge.prompts import build_messages, parse_verdict
from judge.reward import compute_score, compute_score_batch


class FakeJudge(BaseHTTPRequestHandler):
    calls = 0
    fail_first = 0

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeJudge.calls += 1
        if FakeJudge.fail_first > 0:
            FakeJudge.fail_first -= 1
            self.send_response(500); self.end_headers(); return
        user = body["messages"][-1]["content"]
        # a gullible judge: says CORRECT whenever the answer mentions "Jefferson" OR sounds confident
        verdict = "CORRECT" if ("Jefferson" in user or "definitely" in user) else "INCORRECT"
        out = {"choices": [{"message": {"content": f"Reasoning...\nVerdict: {verdict}"}}]}
        data = json.dumps(out).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), FakeJudge)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    FakeJudge.calls, FakeJudge.fail_first = 0, 0
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


@pytest.fixture
def client(server, tmp_path):
    return JudgeClient(server, "fake", cache_path=str(tmp_path / "c.sqlite"),
                       log_path=str(tmp_path / "log.jsonl"), max_retries=3, max_workers=4)


def test_parse_verdict():
    assert parse_verdict("blah\nVerdict: CORRECT") == 1
    assert parse_verdict("Verdict: **incorrect**") == 0
    assert parse_verdict("My answer: Verdict: correct ... Verdict: INCORRECT") == 0  # last one wins
    assert parse_verdict("no idea") is None


def test_prompts_reference_free():
    m = build_messages("answer_only", "Who wrote it?", answer="Bob")
    assert "Reference" not in m[-1]["content"] and "Bob" in m[-1]["content"]
    assert "v1" in m[-1]["content"]  # prompt version is part of the cache key


def test_cache_and_log(client, tmp_path):
    msg = build_messages("answer_only", "Q?", answer="Jefferson")
    assert client.chat(msg) == client.chat(msg)
    assert FakeJudge.calls == 1  # second call served from cache
    lines = (tmp_path / "log.jsonl").read_text().strip().splitlines()
    assert [json.loads(l)["cached"] for l in lines] == [False, True]


def test_retry_on_server_error(client):
    FakeJudge.fail_first = 2
    out = client.chat(build_messages("answer_only", "Q?", answer="x"))
    assert "Verdict" in out and FakeJudge.calls == 3


def test_reward_trains_on_judge_logs_em(client):
    gt = {"target": ["Thomas Jefferson"]}
    sols = [
        "<answer>Thomas Jefferson</answer>",           # correct, judge agrees
        "<answer>definitely John Adams</answer>",      # WRONG, but confident -> gullible judge says CORRECT
        "<answer>John Adams</answer>",                 # wrong, judge agrees
        "<tool_call>x</tool_call>",                    # no answer -> 0, no judge call
    ]
    res = compute_score_batch(["searchR1_nq"] * 4, sols, [gt] * 4, [{"question": "Who wrote it?"}] * 4, client=client)
    assert [r["score"] for r in res] == [1.0, 1.0, 0.0, 0.0]
    assert [r["em"] for r in res] == [1, 0, 0, 0]
    # the hacked sample: judge reward 1, gold EM 0 -> exactly what we monitor
    assert res[1]["score"] == 1.0 and res[1]["em"] == 0
    assert FakeJudge.calls == 3


def test_per_sample_entry_point(client):
    r = compute_score("searchR1_nq", "<answer>definitely wrong</answer>", {"target": ["Thomas Jefferson"]},
                      {"question": "Who wrote it?"}, client=client)
    assert r["score"] == 1.0 and r["em"] == 0       # gullible judge rewarded a wrong answer; EM is logged
