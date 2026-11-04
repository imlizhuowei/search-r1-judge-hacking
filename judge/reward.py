"""verl reward using an LLM judge.

  reward.custom_reward_function.path=<repo>/judge/reward.py  reward.custom_reward_function.name=compute_score
  (judge prompt variant from env JUDGE_VARIANT, default answer_only)

verl's reward loop calls compute_score once per sample (sync functions run in a thread pool, so judge
calls overlap). compute_score_batch is the same logic for a list (used by calibration scripts).

The judge's verdict is the training reward. Gold-based EM, ToolHit and GCA labels are computed too but
ONLY logged as extra info (train without gold, monitor with gold): this gives the judge-vs-EM gap per step,
which is the first signal of judge hacking.

Judge endpoint from env: JUDGE_BASE_URL (default local vLLM on :8001), JUDGE_MODEL, JUDGE_API_KEY_ENV.
"""

from __future__ import annotations

import os

from gca.labels import extract_answer, label_rollout
from gca.reward import observations_from_text
from judge.client import JudgeClient
from judge.prompts import build_messages, parse_verdict

_CLIENT = None


def get_client() -> JudgeClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = JudgeClient(
            base_url=os.environ.get("JUDGE_BASE_URL", "http://127.0.0.1:8001/v1"),
            model=os.environ.get("JUDGE_MODEL", "judge"),
            api_key_env=os.environ.get("JUDGE_API_KEY_ENV") or None,
            cache_path=os.environ.get("JUDGE_CACHE", "/root/autodl-tmp/judge_cache/train.sqlite"),
            log_path=os.environ.get("JUDGE_LOG", "/root/autodl-tmp/judge_cache/train_calls.jsonl"),
        )
    return _CLIENT


def _golds(gt):
    g = gt.get("target", gt) if isinstance(gt, dict) else gt
    return [str(x) for x in (g or [])]


def compute_score(data_source, solution_str, ground_truth, extra_info=None, client: JudgeClient | None = None,
                  **kwargs) -> dict:
    """Per-sample entry point for verl's reward loop."""
    variant = os.environ.get("JUDGE_VARIANT", "answer_only")
    return compute_score_batch([data_source], [solution_str], [ground_truth], [extra_info or {}],
                               variant=variant, client=client)[0]


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos, variant: str = "answer_only",
                        client: JudgeClient | None = None, **kwargs) -> list[dict]:
    client = client or get_client()
    msgs, idx, results = [], [], []
    for i, (sol, gt, ex) in enumerate(zip(solution_strs, ground_truths, extra_infos)):
        golds = _golds(gt)
        lab = label_rollout(sol, observations_from_text(sol), golds)
        ans = extract_answer(sol)
        results.append({
            "score": 0.0, "judge_verdict": -1, "judge_parsed": 0,
            "em": lab.correct, "acc_lenient": lab.lenient, "gca_hit": lab.hit,
            "gca_first_hit": lab.first_hit if lab.first_hit is not None else -1,
            "gca_answered": int(lab.answered), "gca_type": lab.rtype,
            "resp_len_chars": len(sol),
        })
        if ans is None:
            continue  # no answer -> reward 0 without calling the judge
        question = (ex or {}).get("question", "")
        m = build_messages(variant, question, answer=ans, trajectory=sol,
                           golds=golds if variant == "reference" else None)
        msgs.append(m)
        idx.append(i)

    outs = client.chat_many(msgs, tags=[variant] * len(msgs)) if msgs else []
    for i, out in zip(idx, outs):
        v = parse_verdict(out)
        results[i]["judge_parsed"] = int(v is not None)
        results[i]["judge_verdict"] = -1 if v is None else v
        results[i]["score"] = float(v or 0)
    return results
