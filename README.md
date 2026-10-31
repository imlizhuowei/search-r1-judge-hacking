# Judge Rewards in Agentic Search RL: Hacking, Grounding, and Credit Assignment

> Status: 🚧 In progress (plan finalized 2026-10-31; no results yet)

## Motivation

Search agents trained with outcome-reward RL (Search-R1-style GRPO) are usually scored by exact match (EM) against a gold answer. Recent work shows that even with EM, accuracy can rise **without grounding**: the agent increasingly answers from parametric memory instead of retrieved evidence (GCA, arXiv 2610.23148). GCA traces this to GRPO's credit assignment and fixes it, but all of its signals depend on gold-answer string matching.

Many real tasks have no gold answer, so training relies on an **LLM judge**. Judges can be fooled (e.g., *More Convincing, Not More Correct*, arXiv 2607.05904), and concurrent work trains grounding-aware rewards for search agents (SOAR, CSEG).

**Questions**
1. When a search agent is trained with a reference-free LLM-judge reward, in which ways does the judge get exploited, and how does this interact with ungrounded answering?
2. Do credit-assignment fixes like GCA still work when their sub-outcome signals come from a judge that can itself be fooled?

## Protocol

- **Train without gold, evaluate with gold.** Rewards come from a judge that does not see the reference answer. Evaluation sets (NQ, HotpotQA, etc.) do have gold answers, so EM serves as an objective anchor; judge scores are reported as process metrics, plus a human-checked subset.
- Small control: a reference-based judge on key configurations only.
- Judge via API with a pinned model version, temperature 0, and every request/response cached and logged.
- Main claims rest on MuSiQue / 2Wiki / HotpotQA; Bamboogle (125 questions) is supporting evidence only. Paired bootstrap CIs are reported.

## Plan

| Phase | Work | Exit checks |
|---|---|---|
| **1. Synchronous base** | Port SearchTool to current verl → Qwen2.5-3B + EM + GRPO baseline → API judge reward → collect top-k token probabilities, entropy and margin, aligned per token → synchronous GCA | (1) η=0 reproduces GRPO numerically (2) hand-built A/B/C/D trajectories match hand-computed advantages (3) the baseline actually learns: clear EM gain, sane curves, gap to the GCA paper's baseline recorded (4) top-k statistics verified by recomputing a fixed batch with the same weights |
| **Research (synchronous)** | Judge-reward training → hacking analysis → GCA transferred to judge-derived sub-outcomes (core) → lightweight judge fix (evidence-conditioned judging) → data control using gold answers (upper bound, not part of the main method) | Complete evidence chain: phenomenon → pattern → cause → fix → alternative explanations ruled out (e.g., length, overall capability, question mix) |
| **Seeds** | Exploration runs use one seed; *judge + GRPO* and *judge + GCA* use matched seeds, 3 each | Gap exceeds seed variance; add seeds if not |
| **2. Asynchronous (≤ 1 week)** | One-step off-policy; track behavior-policy version and per-token statistics across versions; asynchronous judge requests (cache, rate limits, retries, timeouts); every trajectory traceable to its generating model, judge call, and training version | Sync vs. async comparison: test whether asynchrony changes the main conclusions; compare throughput, waiting time, and consistency of behavior-policy statistics |

## Data and analysis deliverables

- Train/test de-duplication and leakage check; question stratification (single-hop vs. multi-hop, memorizable, hard to retrieve)
- Judge quality check set: answers sampled across training stages, judged by a stronger independent judge, with a human-checked subset; agreement with EM and human labels
- Trajectory table: searches, retrieved evidence, judge score, EM, evidence support, answer pattern, before/after fixes
- Hacking pattern taxonomy with frequencies, and how each fix changes them
- Grounding gap Δ (GCA, Prop. 1) tracked during training under EM vs. judge rewards

## Setup

- Framework: [verl](https://github.com/verl-project/verl) (current main), Search-R1-like multi-turn tool calling
- Retriever: E5 over 2018 Wikipedia, top-3 passages, local retrieval server
- Training data: NQ + HotpotQA; evaluation: 7 QA benchmarks (subsampled for validation)
- Policy: Qwen2.5-3B-Instruct (1.5B only for pipeline debugging)

## Results

_To be added._

## Experiment Log

See [`LOG.md`](LOG.md).

## References

- Jin et al., *Search-R1*, COLM 2025. [arXiv:2503.09516](https://arxiv.org/abs/2503.09516)
- Liu, Chen et al., *Grounded Credit Assignment in RL for Multi-Turn Search Agents*. [arXiv:2610.23148](https://arxiv.org/abs/2610.23148)
- *SOAR: Source-Aware RL with Precise Credit Assignment for Search Agents*. [arXiv:2610.13157](https://arxiv.org/abs/2610.13157)
- *Benchmarking and Training Reward Models for Cross-Source Evidence Grounding in Search Agents (CSEG)*. [arXiv:2610.17827](https://arxiv.org/abs/2610.17827)
- *Retrieval, Reward, and Training Protocols: What Matters in Training Search Agents?* [arXiv:2605.27881](https://arxiv.org/abs/2605.27881)
- *More Convincing, Not More Correct: Self-Play Reward Hacking of Reference-Free LLM Judges*. [arXiv:2607.05904](https://arxiv.org/abs/2607.05904)
- *Chaining the Evidence: Citation-Aware Rubric Rewards for Deep Search Agents (CaRR)*. [arXiv:2601.06021](https://arxiv.org/abs/2601.06021)
- *Proof-of-Use: Mitigating Tool-Call Hacking in Deep Research Agents*, EMNLP 2026.
- *One Token to Fool LLM-as-a-Judge*. [arXiv:2507.08794](https://arxiv.org/abs/2507.08794)
- Feng et al., *GiGPO*. [arXiv:2505.10978](https://arxiv.org/abs/2505.10978); *Tree Search for LLM Agent RL (Tree-GRPO)*. [arXiv:2509.21240](https://arxiv.org/abs/2509.21240)
