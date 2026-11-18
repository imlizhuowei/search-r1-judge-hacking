#!/usr/bin/env bash
# Timing measurement (LOG 11-18): where does one training step's time go?
#
# Resumes the GRPO baseline from its last checkpoint and trains 10 more steps (no checkpoint saved, no
# validation), so the timing is for the late-training regime (long responses, several searches), not step 1.
# verl logs per step: timing_s/gen (rollout incl. waiting for search), timing_s/update_actor, timing_s/ref, ...
# The retriever logs per batch: encode / faiss search / loading passages, queue wait, query hashes.
# Output: results/timing_probe_train.json (verl side), results/timing_run_<run>.json (earlier full runs) and results/retriever_probe.json (retriever side + repeat rate).
#
#   bash scripts/timing_probe.sh            # SRC=grpo_em_s1 PROBE_STEPS=10 by default
set -uo pipefail
REPO=${REPO:-/root/autodl-tmp/search-r1-judge-hacking}
PY=/root/autodl-tmp/verl/.venv/bin/python
SRC=${SRC:-grpo_em_s1}
PROBE_STEPS=${PROBE_STEPS:-10}
EXP=timing_probe
LOG=/root/autodl-fs/runs/$EXP/train.log
cd "$REPO"
mkdir -p "$(dirname "$LOG")" results

CKPT=$(ls -d /root/autodl-fs/runs/$SRC/ckpt/global_step_* 2>/dev/null | sort -V | tail -1)
[ -d "${CKPT:-}" ] || { echo "no checkpoint for $SRC"; exit 1; }
START=$(basename "$CKPT" | sed 's/global_step_//')
STATS=$(ls -t /root/autodl-fs/runs/_retriever_stats/*.jsonl 2>/dev/null | head -1)
[ -n "$STATS" ] || echo "WARNING: no retriever stats file (retriever started without RETRIEVER_STATS?)"

t0=$(date +%s)
EXP=$EXP ADV=grpo STEPS=$(( START + PROBE_STEPS )) SAVE_FREQ=-1 TEST_FREQ=-1 VAL_BEFORE_TRAIN=False \
    RESUME_MODE=resume_path RESUME_FROM="$CKPT" bash scripts/train.sh >> "$LOG" 2>&1
rc=$?
t1=$(date +%s)

python3 scripts/summarize_timing.py "$LOG" /dev/null probe_train src="$SRC" from_step="$START" > /dev/null 2>&1 || true
# the whole earlier runs, for comparison: grpo_em_s1 (Flat index) and gca_em_s1 (fp16 index, ~4x slower)
for run in grpo_em_s1 gca_em_s1; do
    [ -f /root/autodl-fs/runs/$run/train.log ] && \
        python3 scripts/summarize_timing.py /root/autodl-fs/runs/$run/train.log /dev/null "run_$run" > /dev/null 2>&1 || true
done
STATS=$(ls -t /root/autodl-fs/runs/_retriever_stats/*.jsonl 2>/dev/null | head -1)
[ -n "$STATS" ] && $PY scripts/summarize_retriever.py "$STATS" --since "$t0" --until "$t1" --train-log "$LOG" \
    --out results/retriever_probe.json
exit $rc
