#!/usr/bin/env bash
# Rebuild wiki-18.jsonl from a fresh copy of the .gz and verify it byte-for-byte.
# Why: our first decompression left a corrupted last line + 5,903 NUL bytes at the end.
# Usage: nohup bash scripts/repair_corpus.sh > /root/autodl-tmp/repair.log 2>&1 &
set -euo pipefail
D=${D:-/root/autodl-fs/searchr1}
URL=https://hf-mirror.com/datasets/PeterJinGo/wiki-18-corpus/resolve/main/wiki-18.jsonl.gz
cd "$D"
echo "[repair] $(date) downloading gz (resumable)"
wget -c -q "$URL" -O wiki-18.jsonl.gz
ls -l wiki-18.jsonl.gz
echo "[repair] $(date) gzip integrity test"
gzip -t wiki-18.jsonl.gz && echo "[repair] gzip OK"
echo "[repair] $(date) where does the current file first differ from the gz? (cmp)"
gunzip -c wiki-18.jsonl.gz | cmp - wiki-18.jsonl || true
echo "[repair] $(date) decompressing to wiki-18.jsonl.new"
gunzip -c wiki-18.jsonl.gz > wiki-18.jsonl.new
sync
echo "[repair] $(date) verifying the new file against the gz"
gunzip -c wiki-18.jsonl.gz | cmp - wiki-18.jsonl.new && echo "[repair] VERIFIED identical"
ls -l wiki-18.jsonl wiki-18.jsonl.new
mv wiki-18.jsonl.new wiki-18.jsonl
rm -f wiki-18.offsets.npy
echo "[repair] $(date) done. Restart the retriever; it will rescan line offsets once."
