# Status and next steps (keep this file current; it is the hand-off note between sessions)

Updated 2026-11-05.

## Running (queue on the A800 instance, /root/autodl-tmp/queue.txt)
1. gca_em_s1  — GCA, seed 1 (started 11-04 ~18:30 CST, ~10 h)
2. grpo_em_s2 — GRPO baseline, seed 2
3. gca_em_s2  — GCA, seed 2
Then idle 5 h -> auto shutdown. Results: /root/autodl-fs/runs/<EXP>/ (train.log, rollouts/, val/, ckpt).

## Facts to remember
- Retriever now uses the fp16 index (e5_SQfp16.index): top-3 agreement 99.5% vs fp32 Flat, top-1 score diff <= 5.4e-7
  (results/retriever_fp16_check.json). grpo_em_s1 (val 0.237) used the fp32 Flat index; all later runs use fp16.
- GCA runs on the V1 trainer (same code path as the baseline) via patches/verl_gca_v1.patch.
- Server has had no outbound internet since 11-04 afternoon: code updates go in as git bundles
  (git fetch <file>.bundle main && git merge --ff-only FETCH_HEAD).
- mem.log "container_used" includes page cache; real usage = anon in /sys/fs/cgroup/memory.stat.

## Next (in order)
1. When runs finish: scripts/analyze_rollouts.py on each run; compare val curves (grpo s1/s2 vs gca s1/s2).
2. Judge arm: vLLM judge server script (Qwen2.5-7B-Instruct), REWARD=judge switch in train.sh, judge-vs-EM
   calibration on baseline rollouts. Needs: judge model downloaded (server internet), ideally a 2-GPU instance
   (GPU0 policy training unchanged, GPU1 judge server).
3. mem monitor: log anon/file split.
4. Look into response-length growth in the baseline (279 -> 474).
