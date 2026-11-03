"""Custom reward for verl: Search-R1 EM reward + GCA labels as extra info.

Plug in with:
  custom_reward_function.path=<repo>/gca/reward.py  custom_reward_function.name=compute_score

Returns a dict; verl's naive reward manager uses dict["score"] as the reward and stores every key
in reward_extra_info, which the (legacy) trainer merges into batch.non_tensor_batch — that is how
the GCA advantage estimator sees hit / first_hit / rtype per rollout.

The score itself is identical to verl's search_r1_like_qa_em.compute_score (EM, format_score=0),
so an EM+GRPO run with this reward is the same baseline as without it.
"""

from __future__ import annotations

import re

from gca.labels import extract_answer, label_rollout

TOOL_RESP = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)


def observations_from_text(solution_str: str) -> list[str]:
    """Tool responses in order, from the decoded response (Qwen2.5 hermes template wraps them in tags)."""
    return [m.strip() for m in TOOL_RESP.findall(solution_str)]


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> dict:
    golds = ground_truth["target"] if isinstance(ground_truth, dict) else ground_truth
    golds = [str(g) for g in (golds if isinstance(golds, (list, tuple)) else list(golds))]
    obs = observations_from_text(solution_str)
    lab = label_rollout(solution_str, obs, golds, retr=True)  # Retr(x) refined later (needs oracle queries)

    # verl's EM reward also quarters the score if the model spams <answer> tags
    score = float(lab.correct)
    if score and (solution_str.count("<answer>") > 10 or solution_str.count("</answer>") > 10):
        score /= 4

    return {
        "score": score,
        "gca_hit": lab.hit,
        "gca_first_hit": lab.first_hit if lab.first_hit is not None else -1,  # -1 = no hit (np arrays dislike None)
        "gca_answered": int(lab.answered),
        "gca_type": lab.rtype,
        "gca_n_obs": len(obs),
        "acc_lenient": lab.lenient,
        "has_answer": int(extract_answer(solution_str) is not None),
    }
