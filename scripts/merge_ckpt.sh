#!/usr/bin/env bash
# Turn a verl FSDP checkpoint into a plain HuggingFace model folder (weights only, no optimizer state):
#   bash scripts/merge_ckpt.sh grpo_em_s1            # latest global_step_* of that run
# Prints the output folder. Skips the work if it already exists.
#
# Why (LOG 11-19): evaluating straight from the checkpoint (trainer.resume_mode=resume_path) also loads the
# Adam state onto the GPU (~24 GB for 3B params), and vLLM then runs out of memory when it wakes up. A merged
# model is loaded exactly like the base model: same eval path, no optimizer, no OOM.
set -euo pipefail
RUN=${1:?run name, e.g. grpo_em_s1}
CKPT=${CKPT:-$(ls -d /root/autodl-fs/runs/$RUN/ckpt/global_step_* 2>/dev/null | sort -V | tail -1)}
[ -d "${CKPT:-}/actor" ] || { echo "no checkpoint for $RUN" >&2; exit 1; }
OUT=/root/autodl-fs/models/${RUN}_$(basename "$CKPT")
if [ ! -f "$OUT/config.json" ]; then
    mkdir -p "$(dirname "$OUT")"
    (cd /root/autodl-tmp/verl && .venv/bin/python -m verl.model_merger merge --backend fsdp \
        --local_dir "$CKPT/actor" --target_dir "$OUT.tmp" >&2) && mv "$OUT.tmp" "$OUT"
fi
echo "$OUT"
