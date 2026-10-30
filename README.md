# Reward Hacking in Agentic RL: Search-R1 Reproduction & Robust Rewards

> Status: 🚧 In progress

## Motivation

Search-R1 trains an LLM with reinforcement learning to decide **when and what to search** while reasoning, using only an exact-match (EM) outcome reward. The authors deliberately avoid neural reward models, noting that LLMs are sensitive to the specific form of rewards in large-scale RL.

In practice, many open-ended tasks cannot be scored by exact match, so LLM-as-a-Judge rewards are increasingly common. Recent work (e.g., *One Token to Fool LLM-as-a-Judge*) shows such judges can be fooled by superficial patterns.

**Core question: when a tool-using agent is trained with an LLM-judge reward, does its "improvement" reflect real capability, or does it learn to exploit the reward?**

## Plan

1. **Reproduce the baseline** — Train a small model (Qwen2.5, 0.5B–1.5B) with Search-R1-style multi-turn RL on verl, using the original EM reward. Compare training dynamics (reward, response length, number of valid searches) with the paper.
2. **Swap in an LLM-judge reward** — Replace EM with an LLM judge and check whether the policy exploits it: training reward rises while strict EM accuracy does not.
3. **Fix and verify** — Try more robust reward designs (e.g., rule + judge hybrid, adversarially hardened judges) and verify gains across multiple random seeds.
4. **Capability-boundary check** — Compare pass@1 vs. pass@k before and after training to see whether RL expands what the model can solve or only sharpens existing behavior.

## Setup

- Framework: [verl](https://github.com/volcengine/verl) (Search-R1-style multi-turn search example)
- Retriever: E5 + 2018 Wikipedia, local retrieval server
- Training data: NQ + HotpotQA; Evaluation: 7 QA benchmarks (EM)

## Results

_To be added._

## Experiment Log

See [`LOG.md`](LOG.md).

## References

- Jin et al., *Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning*, COLM 2025. [arXiv:2503.09516](https://arxiv.org/abs/2503.09516)
- *One Token to Fool LLM-as-a-Judge*, NeurIPS 2025.
- Yue et al., *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?*, NeurIPS 2025.
