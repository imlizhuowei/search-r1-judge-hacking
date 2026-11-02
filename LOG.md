# Experiment Log

Record every run here: date, config, what you expected, what happened, and what you concluded.

| Date | Run | Model | Config / change | Result | Notes |
|---|---|---|---|---|---|
| | | | | | |

## Decisions

| Date | Decision | Why |
|---|---|---|
| 2026-10-31 | Focus on judge-reward hacking + GCA-style credit assignment, not a plain Search-R1 reproduction | Plain Search-R1 reproductions are common; GCA's own limitation (needs gold answers) leaves the judge setting open |
| 2026-10-31 | Reference-free judge as main line; small reference-based control | Matches the no-gold setting; the control shows how much a reference helps |
| 2026-10-31 | Qwen2.5-3B (not 1.5B) | Tree-GRPO reports chain-based GRPO struggles to elicit multi-turn tool use at 1.5B; 3B allows comparison with GCA's numbers |
| 2026-10-31 | Current verl instead of verl-agent | Matches GCA's setup (concatenated trajectories in verl); has built-in async for phase 2 |
| 2026-10-31 | Sync first; async capped at one week | Async adds a confound (policy lag); core claims come from sync runs |
| 2026-10-31 | Gold-based data filtering only as an upper-bound control | Using gold answers in the main method would contradict the no-gold premise |

## Notes & Issues

- 2026-11-02: Environments ready. Retriever venv (`/root/autodl-tmp/envs/retriever`, Python 3.12, torch 2.6.0, faiss-cpu 1.15.1, transformers 5.22.1; e5-base-v2 at `/root/autodl-tmp/models/e5-base-v2`). verl pinned at 8e48d72 via shallow fetch; `uv sync --frozen --all-packages --extra vllm --extra fsdp` → torch 2.13.0+cu130, vllm 0.32.0 (`UV_CACHE_DIR=/root/autodl-tmp/uv_cache` to spare the 30GB system disk). Pitfalls: Tsinghua conda mirror 403 (used venv instead); torch 2.4 disabled by transformers ≥5 (needs ≥2.6); stale uv lock after killing tmux (`pkill -9 -x uv`); bracketed-paste garbage in tmux. Retriever server not yet tested (needs GPU-mode RAM).
- 2026-11-01: Retrieval data ready on AutoDL file storage (`/root/autodl-fs/searchr1`, A800 zone). `e5_Flat.index` = 64,559,075,373 bytes (= part_aa 42,949,672,960 + part_ab 21,609,402,413, both matching HF sizes); `wiki-18.jsonl` = 14,393,579,520 bytes. Downloaded via hf-mirror (`hf download` hit a brotli/httpx `TypeError`; `wget -c` worked). A800 instances can't expand the data disk, hence file storage.
- 
