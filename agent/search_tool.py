"""Search tool for verl's tool agent loop (current verl, BaseTool API).

Ported from the old verl `verl/tools/search_tool.py`, simplified:
- no Ray rate-limit actor; concurrency is capped by an asyncio semaphore per process
- the blocking HTTP call runs in a thread so the agent loop's event loop is not blocked
- output format matches Search-R1: "Doc i (Title: t)\\ntext"

The model calls it like:
<tool_call>{"name": "search", "arguments": {"query_list": ["..."]}}</tool_call>
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Optional
from uuid import uuid4

import requests

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def passages_to_string(results: list[dict]) -> str:
    """Format one query's results the same way Search-R1 does."""
    out = []
    for i, item in enumerate(results):
        contents = item["document"]["contents"]
        title, _, text = contents.partition("\n")
        out.append(f"Doc {i + 1} (Title: {title})\n{text}")
    return "\n\n".join(out)


class SearchTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.url = config.get("retrieval_service_url", "http://127.0.0.1:8000/retrieve")
        self.topk = config.get("topk", 3)
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)
        self.max_queries = config.get("max_queries_per_call", 4)
        self._sem = asyncio.Semaphore(config.get("max_concurrency", 64))
        self._instances: dict[str, dict] = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        instance_id = instance_id or str(uuid4())
        # create_kwargs from the dataset (ground_truth, question, data_source) arrive here.
        # We keep them only for logging; the tool itself never looks at the gold answer.
        self._instances[instance_id] = {"n_calls": 0, "queries": []}
        return instance_id, ToolResponse()

    def _post(self, queries: list[str]) -> dict:
        payload = {"queries": queries, "topk": self.topk, "return_scores": True}
        last_err = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(self.url, json=payload, timeout=self.timeout)
                r.raise_for_status()
                return r.json()
            except Exception as e:  # network errors, 5xx, bad JSON
                last_err = e
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"retrieval failed after {self.max_retries} tries: {last_err}")

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        queries = parameters.get("query_list")
        if isinstance(queries, str):
            queries = [queries]
        if not isinstance(queries, list) or not queries or not all(isinstance(q, str) for q in queries):
            msg = "Error: 'query_list' must be a non-empty list of strings."
            return ToolResponse(text=msg), 0.0, {"status": "bad_args"}
        queries = queries[: self.max_queries]

        state = self._instances.setdefault(instance_id, {"n_calls": 0, "queries": []})
        state["n_calls"] += 1
        state["queries"].extend(queries)

        t0 = time.time()
        try:
            async with self._sem:
                data = await asyncio.to_thread(self._post, queries)
            blocks = [passages_to_string(res) for res in data["result"]]
            text = "\n\n---\n\n".join(blocks) if len(blocks) > 1 else blocks[0]
            status = "ok"
        except Exception as e:
            logger.warning("search error: %s", e)
            text, status = f"Search error: {e}", "error"

        metrics = {"status": status, "n_queries": len(queries), "latency_s": round(time.time() - t0, 3)}
        return ToolResponse(text=text), 0.0, metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0  # reward comes from the outcome reward function, not the tool

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)


if __name__ == "__main__":
    # Quick standalone check against a running retriever:
    #   python -m agent.search_tool
    schema = OpenAIFunctionToolSchema.model_validate(
        {"type": "function", "function": {"name": "search", "description": "", "parameters": {"type": "object", "properties": {}, "required": []}}}
    )
    tool = SearchTool({}, schema)

    async def main():
        iid, _ = await tool.create()
        resp, _, m = await tool.execute(iid, {"query_list": ["Who wrote the Declaration of Independence?"]})
        print(m)
        print(resp.text[:800])

    asyncio.run(main())
