"""Verifiable sub-outcomes (GCA Eq. 1) and grounding metrics (GCA Sec. 5.1).

Used two ways:
  - training (EM-reward runs): label each rollout -> type A/B1/B2/C/D/D0 for GCA routing
  - evaluation (all runs, incl. judge-reward runs): EM, Acc, ToolHit, HitAcc, SupCor, grounding gap

normalize_answer / EM / sub-EM are copied from verl/utils/reward_score/search_r1_like_qa_em.py
so our labels agree exactly with the reward verl computes.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass

from gca.advantage import classify


# ---------- answer normalisation (identical to verl's Search-R1 reward) ----------

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def extract_answer(text: str) -> str | None:
    """Last <answer>...</answer> in the rollout, or None."""
    m = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    return m[-1].group(1).strip() if m else None


def em(pred: str | None, golds: list[str]) -> int:
    """C_i in Eq. 1: exact match after normalisation."""
    if pred is None:
        return 0
    p = normalize_answer(pred)
    return int(any(normalize_answer(g) == p for g in golds))


def sub_em(pred: str | None, golds: list[str]) -> int:
    """Lenient Acc (C~_i): normalised gold contained in normalised prediction."""
    if pred is None:
        return 0
    p = normalize_answer(pred)
    return int(any(normalize_answer(g) in p for g in golds))


# ---------- ToolHit (Eq. 1) ----------

def contains(alias: str, obs: str) -> bool:
    """m(a, o) = 1[lower(strip(a)) ⊆ lower(o)]  — plain case-insensitive substring, as in the paper.
    Note: deliberately naive (e.g. '1' matches '1999'); we keep it to stay comparable to GCA."""
    a = alias.strip().lower()
    return bool(a) and a in obs.lower()


def tool_hit(observations: list[str], golds: list[str]) -> tuple[int, int | None]:
    """Returns (H_i, k*_i). k* is the 1-based index of the first search whose result contains a gold alias."""
    for k, obs in enumerate(observations, start=1):
        if any(contains(a, obs) for a in golds):
            return 1, k
    return 0, None


# ---------- one rollout ----------

@dataclass
class RolloutLabel:
    hit: int
    first_hit: int | None   # k*
    correct: int            # EM
    lenient: int            # sub-EM (Acc)
    answered: bool
    rtype: str


def label_rollout(response_text: str, observations: list[str], golds: list[str], retr: bool = True) -> RolloutLabel:
    """observations: tool responses in order. retr: Retr(x), whether the corpus can return the answer.

    B2 ("turn budget exhausted") is approximated as "hit but no <answer> produced".
    """
    pred = extract_answer(response_text)
    h, k = tool_hit(observations, golds)
    c = em(pred, golds)
    return RolloutLabel(h, k, c, sub_em(pred, golds), pred is not None,
                        classify(bool(h), bool(c), timeout=pred is None, retr=retr))


# ---------- metrics over many rollouts ----------

def grounding_metrics(labels: list[RolloutLabel]) -> dict:
    n = len(labels)
    if n == 0:
        return {}
    hits = [l for l in labels if l.hit]
    tool_hit_rate = len(hits) / n
    hit_acc = sum(l.lenient for l in hits) / len(hits) if hits else 0.0
    acc = sum(l.lenient for l in labels) / n
    sup_cor = tool_hit_rate * hit_acc
    return {
        "EM": sum(l.correct for l in labels) / n,
        "Acc": acc,
        "ToolHit": tool_hit_rate,
        "HitAcc": hit_acc,
        "SupCor": sup_cor,
        "GroundedShare": sup_cor / acc if acc else 0.0,
        "types": {t: sum(l.rtype == t for l in labels) / n for t in ["A", "B1", "B2", "C", "D", "D0"]},
    }


def grounding_gap(labels: list[RolloutLabel]) -> float | None:
    """Delta = Pr[C | H=1] - Pr[C | H=0] within one group (Prop. 1). None if all hit or none hit."""
    h1 = [l.correct for l in labels if l.hit]
    h0 = [l.correct for l in labels if not l.hit]
    if not h1 or not h0:
        return None
    return sum(h1) / len(h1) - sum(h0) / len(h0)
