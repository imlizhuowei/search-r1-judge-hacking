"""CPU-only tests for agent/search_tool.py. No GPU, no retriever needed.

The HTTP call is replaced by a fake (monkeypatch), so we test our own logic:
argument checking, output format, multi-query joining, error handling, cleanup.

Run on the server (real verl):   cd <repo> && /root/autodl-tmp/verl/.venv/bin/python -m pytest -q tests
Run anywhere without verl:       python -m pytest -q tests   (uses a minimal stand-in for BaseTool)
"""

import asyncio
import sys
import types

import pytest

# ---- use real verl if importable, otherwise a tiny stand-in with the same interface ----
try:
    import verl.tools.base_tool  # noqa: F401
except Exception:
    from pydantic import BaseModel

    class ToolResponse(BaseModel):
        text: str | None = None

    class OpenAIFunctionToolSchema(BaseModel):
        type: str = "function"
        function: dict = {}

    class BaseTool:
        def __init__(self, config, tool_schema):
            self.config, self.tool_schema = config, tool_schema

    for name in ["verl", "verl.tools"]:
        sys.modules.setdefault(name, types.ModuleType(name))
    bt = types.ModuleType("verl.tools.base_tool")
    bt.BaseTool = BaseTool
    sc = types.ModuleType("verl.tools.schemas")
    sc.ToolResponse, sc.OpenAIFunctionToolSchema = ToolResponse, OpenAIFunctionToolSchema
    sys.modules["verl.tools.base_tool"], sys.modules["verl.tools.schemas"] = bt, sc

from agent import search_tool as st  # noqa: E402

SCHEMA = st.OpenAIFunctionToolSchema.model_validate(
    {"type": "function", "function": {"name": "search", "description": "", "parameters": {"type": "object", "properties": {}, "required": []}}}
)


def fake_result(n_queries, topk=3):
    """Shape returned by retriever/retrieval_server.py with return_scores=True."""
    return {
        "result": [
            [{"document": {"contents": f"Title{q}-{k}\nbody {q}-{k}"}, "score": 0.9} for k in range(topk)]
            for q in range(n_queries)
        ]
    }


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def tool(monkeypatch):
    t = st.SearchTool({"max_retries": 2}, SCHEMA)
    calls = []

    def fake_post(queries):
        calls.append(list(queries))
        return fake_result(len(queries))

    monkeypatch.setattr(t, "_post", fake_post)
    t.calls = calls
    return t


def test_format_matches_search_r1():
    text = st.passages_to_string(fake_result(1)["result"][0])
    assert text.startswith("Doc 1 (Title: Title0-0)\nbody 0-0")
    assert "Doc 3 (Title: Title0-2)" in text


def test_single_query(tool):
    iid, _ = run(tool.create())
    resp, reward, m = run(tool.execute(iid, {"query_list": ["who wrote it"]}))
    assert m["status"] == "ok" and reward == 0.0
    assert resp.text.count("Doc ") == 3 and "---" not in resp.text


def test_multi_query_joined(tool):
    iid, _ = run(tool.create())
    resp, _, m = run(tool.execute(iid, {"query_list": ["a", "b"]}))
    assert m["n_queries"] == 2 and resp.text.count("---") == 1


def test_string_instead_of_list_is_accepted(tool):
    iid, _ = run(tool.create())
    _, _, m = run(tool.execute(iid, {"query_list": "just a string"}))
    assert m["status"] == "ok" and tool.calls[-1] == ["just a string"]


@pytest.mark.parametrize("params", [{}, {"query_list": []}, {"query_list": [1, 2]}, {"query": "x"}])
def test_bad_args_do_not_call_retriever(tool, params):
    iid, _ = run(tool.create())
    resp, _, m = run(tool.execute(iid, params))
    assert m["status"] == "bad_args" and "Error" in resp.text and tool.calls == []


def test_too_many_queries_truncated(tool):
    iid, _ = run(tool.create())
    run(tool.execute(iid, {"query_list": list("abcdefg")}))
    assert len(tool.calls[-1]) == tool.max_queries


def test_retriever_failure_returns_error_text(monkeypatch):
    t = st.SearchTool({}, SCHEMA)

    def boom(queries):
        raise RuntimeError("server down")

    monkeypatch.setattr(t, "_post", boom)
    iid, _ = run(t.create())
    resp, _, m = run(t.execute(iid, {"query_list": ["x"]}))
    assert m["status"] == "error" and "server down" in resp.text  # rollout continues instead of crashing


def test_release_cleans_up(tool):
    iid, _ = run(tool.create())
    run(tool.execute(iid, {"query_list": ["x"]}))
    run(tool.release(iid))
    assert iid not in tool._instances
