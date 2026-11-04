#!/usr/bin/env bash
# Start the local E5 + 2018-Wikipedia retrieval server (CPU faiss; index stays in RAM).
# Usage: bash retriever/launch.sh            (run inside the retriever venv)
set -uo pipefail   # no -e: the restart loop must survive a crash
DATA=${DATA:-/root/autodl-fs/searchr1}
MODEL=${MODEL:-/root/autodl-tmp/models/e5-base-v2}
# Keep the datasets/Arrow cache off the 30GB system disk.
# (corpus is read line-by-line via cached byte offsets; no datasets/Arrow cache needed)
export HF_HOME=${HF_HOME:-/root/autodl-tmp/hf_cache}
# Only ONE retriever ever. A second launch.sh (e.g. started while the first was reloading the index and not
# answering yet) would load another ~60 GB copy, fail to bind the port, and loop forever re-loading it:
# that is what pushed the 120 GB container into OOM on 11-04 (two retrievers: 60.6 + 52.3 GB).
LOCK=${LOCK:-/root/autodl-tmp/retriever.lock}
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "[$(date +%H:%M:%S)] another retriever/launch.sh holds $LOCK; not starting a second one" >&2
    exit 0
fi
# Restart on crash (it segfaulted once mid-training on 11-04). Each restart reloads the 64 GB index (~4 min);
# tool calls during that window return "Search error" text instead of crashing the rollout.
# Index: the fp16 copy if it exists (retriever/convert_index_fp16.py; ~30 GB instead of ~60 GB RAM, retrieval
# agreement checked in results/retriever_fp16_check.json), else the original Flat. INDEX=... overrides.
if [ -z "${INDEX:-}" ]; then
    INDEX="$DATA/e5_Flat.index"
    [ -f "$DATA/e5_SQfp16.index" ] && INDEX="$DATA/e5_SQfp16.index"
fi
echo "[$(date +%H:%M:%S)] retriever index: $INDEX" >&2
while true; do
python "$(dirname "$0")/retrieval_server.py" \
  --index_path "$INDEX" \
  --corpus_path "$DATA/wiki-18.jsonl" \
  --topk 3 \
  --retriever_name e5 \
  --retriever_model "$MODEL" \
  --port "${PORT:-8000}"
  rc=$?
  echo "[$(date +%H:%M:%S)] retrieval_server exited with code $rc; restarting in 5 s" >&2
  sleep 5
done
