# Status and next steps (keep this file current; it is the hand-off note between sessions)

Updated 2026-11-05.

## Where the rest lives
- Research question, protocol, plan, deliverables: README.md (Motivation / Protocol / Plan).
- Every decision since 10-31 with its reason (3B model, verl not verl-agent, sync first, reference-free judge,
  local training judge + Kimi audit judge, gold filtering only as control, V1 trainer, fp16 index): LOG.md "Decisions".
- GCA hyper-parameters (eta 0.2, c_max 0.5, kappa 0.1, routing table) and KL 0.005: gca/advantage.py, scripts/train.sh.
- Scope agreed with the user: 1 seed per arm first, then more seeds (ordering, not cutting scope).

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

## Caveats for interpreting the current GCA runs (gca_em_s1/s2)
- They are **GCA with constant salience c = c_max** (top-k log-probs not wired yet, LOG 11-03 design note), plus
  Retr(x)=1, B2 approximated as "no <answer>", D0 masked as zero advantage. Report them as "GCA (const-salience)",
  not as a full reproduction of the paper's method.
- Phase-1 exit checks still open (README Plan): (3) record the gap to the GCA paper's baseline (our val 0.237 on a
  700-q subset vs the paper's numbers, ~20x more training there); (4) verify top-k statistics on a fixed batch.

## Seeds: what actually varies
- verl 8e48d72: `data.seed` only sets the train-dataloader shuffle; vLLM sampling (`actor_rollout_ref.rollout.seed`)
  and model init seeds default to 42. The 4 queued/finished runs (s1, s2) were launched before this was noticed:
  their seeds vary the DATA ORDER only (sampling seed fixed at 42). From commit after aa5e9ca, train.sh also sets
  rollout.seed=$SEED. Do NOT bundle that commit to the server until gca_em_s2 has started, so grpo_em_s2 and
  gca_em_s2 stay matched. The 3-seed judge runs use the full seeding.

## Check FIRST (affects the running GCA runs)
- LOG 11-03 open item (a): the decoded response the reward sees must keep the `<tool_response>` tags, otherwise
  ToolHit = 0 everywhere and every GCA label degenerates. Check on the gca_em_s1 rollouts: tag rate and the
  gca_type mix (should contain A/B/C/D, not all one type).

## Next (in order)
1. When runs finish: scripts/analyze_rollouts.py on each run; compare val curves (grpo s1/s2 vs gca s1/s2).
   Stratify eval by "answer seen in train" (LOG 11-03 data audit) and look at the 1,377 duplicate test questions.
2. Judge arm: vLLM judge server script (Qwen2.5-7B-Instruct), REWARD=judge switch in train.sh, judge-vs-EM
   calibration on baseline rollouts. Needs: judge model downloaded (server internet), ideally a 2-GPU instance
   (GPU0 policy training unchanged, GPU1 judge server).
   Audit: Kimi on small samples + EM/substring as the free first line + a human-checked subset.
3. Full GCA salience from top-k log-probs (LOG 11-03: option B, top-32 from the actor's forward) before the
   judge+GCA runs; then exit check (4).
4. Seeds target (README): judge+GRPO and judge+GCA, 3 matched seeds each.
5. mem monitor: log anon/file split.
6. tools/grab_gpu.py does not work with the sub-account token yet (capture the sub-account console requests).
7. Look into response-length growth in the baseline (279 -> 474).
