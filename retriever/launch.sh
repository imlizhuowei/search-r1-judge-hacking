#!/usr/bin/env bash
# Start the local E5 + 2018-Wikipedia retrieval server (CPU faiss; index stays in RAM).
# Usage: bash retriever/launch.sh            (run inside the retriever venv)
set -euo pipefail
DATA=${DATA:-/root/autodl-fs/searchr1}
MODEL=${MODEL:-/root/autodl-tmp/models/e5-base-v2}
# Keep the datasets/Arrow cache off the 30GB system disk.
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/root/autodl-tmp/hf_cache/datasets}
export HF_HOME=${HF_HOME:-/root/autodl-tmp/hf_cache}
python "$(dirname "$0")/retrieval_server.py" \
  --index_path "$DATA/e5_Flat.index" \
  --corpus_path "$DATA/wiki-18.jsonl" \
  --topk 3 \
  --retriever_name e5 \
  --retriever_model "$MODEL" \
  --port "${PORT:-8000}"
