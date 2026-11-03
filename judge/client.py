"""Cached, logged, rate-limited client for any OpenAI-compatible chat endpoint.

Same code for both judges:
  - training judge: local vLLM server   (base_url=http://127.0.0.1:8001/v1, model=<local path or name>)
  - audit judge:    Kimi API            (base_url=https://api.moonshot.cn/v1, api key from env)

Reproducibility rules (README "Protocol"):
  - temperature 0, fixed model id, fixed prompt version
  - every request/response cached in SQLite (key = hash of everything that affects the output)
  - every call (incl. cache hits) appended to a JSONL log
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests


class JudgeClient:
    def __init__(self, base_url: str, model: str, api_key_env: str | None = None,
                 cache_path: str = "judge_cache/cache.sqlite", log_path: str | None = "judge_cache/calls.jsonl",
                 temperature: float = 0.0, max_tokens: int = 256, timeout: float = 60.0,
                 max_retries: int = 5, max_workers: int = 32, min_interval_s: float = 0.0):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.key = os.environ.get(api_key_env, "") if api_key_env else ""
        self.params = {"temperature": temperature, "max_tokens": max_tokens}
        self.timeout, self.max_retries, self.max_workers = timeout, max_retries, max_workers
        self.min_interval_s = min_interval_s  # crude global rate limit for APIs
        self._last = 0.0
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        self.db = sqlite3.connect(cache_path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
        self.log_path = log_path

    # ---- cache ----
    def _key(self, messages) -> str:
        blob = json.dumps({"m": self.model, "p": self.params, "msg": messages}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _get(self, k):
        with self._lock:
            row = self.db.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
        return row[0] if row else None

    def _put(self, k, v):
        with self._lock:
            self.db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (k, v))
            self.db.commit()

    def _log(self, rec):
        if self.log_path:
            with self._lock, open(self.log_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- calls ----
    def _post(self, messages) -> str:
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        body = {"model": self.model, "messages": messages, **self.params}
        err = None
        for attempt in range(self.max_retries):
            if self.min_interval_s:
                with self._lock:
                    wait = self._last + self.min_interval_s - time.time()
                    self._last = max(time.time(), self._last + self.min_interval_s)
                if wait > 0:
                    time.sleep(wait)
            try:
                r = requests.post(self.url, json=body, headers=headers, timeout=self.timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"judge call failed after {self.max_retries} tries: {err}")

    def chat(self, messages, tag: str = "") -> str:
        k = self._key(messages)
        t0 = time.time()
        out = self._get(k)
        cached = out is not None
        if not cached:
            out = self._post(messages)
            self._put(k, out)
        self._log({"t": time.time(), "tag": tag, "model": self.model, "key": k, "cached": cached,
                   "latency_s": round(time.time() - t0, 3), "messages": messages, "output": out})
        return out

    def chat_many(self, batch: list, tags: list | None = None) -> list[str]:
        tags = tags or [""] * len(batch)
        with ThreadPoolExecutor(self.max_workers) as ex:
            return list(ex.map(lambda mt: self.chat(*mt), zip(batch, tags)))
