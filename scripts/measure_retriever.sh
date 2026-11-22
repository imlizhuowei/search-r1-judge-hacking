#!/usr/bin/env bash
# Restart the retriever (threads now from the CPU quota, LOG 11-22) and load-test it, unattended:
#   bash scripts/measure_retriever.sh                 # Flat index (the paper's), 64 clients x 5 min
#   USE_IVF_INDEX=1 NPROBE=512 bash scripts/measure_retriever.sh
#   SHUTDOWN=1 ...                                     # power off at the end
# Output: results/retriever_load_<tag>.json (load test) and results/retriever_diag_<tag>.json (per-batch log),
# summary at the end of /root/autodl-tmp/measure_retriever.log.
set -uo pipefail
REPO=${REPO:-/root/autodl-tmp/search-r1-judge-hacking}
RETRIEVER_PY=${RETRIEVER_PY:-/root/autodl-tmp/envs/retriever/bin/python}
VERL_PY=${VERL_PY:-/root/autodl-tmp/verl/.venv/bin/python}
CLIENTS=${CLIENTS:-64}
SECONDS_RUN=${SECONDS_RUN:-300}
TAG=${TAG:-$([ "${USE_IVF_INDEX:-0}" = 1 ] && echo "ivf_np${NPROBE:-0}" || echo flat)_${CLIENTS}c}
cd "$REPO"
mkdir -p results

pkill -f retrieval_server; pkill -f retriever/launch.sh; sleep 3
rm -f /root/autodl-tmp/retriever.lock
STATS=/root/autodl-fs/runs/_retriever_stats/measure_${TAG}_$(date +%m%d_%H%M).jsonl
echo "[$(date +%H:%M:%S)] starting retriever ($TAG), stats -> $STATS"
RETRIEVER_STATS=$STATS PATH="$(dirname "$RETRIEVER_PY"):$PATH" \
    nohup bash retriever/launch.sh > /root/autodl-tmp/retriever.log 2>&1 &

ok=0
for i in $(seq 1 120); do          # up to 30 min to load the index
    if curl -s -m 10 -X POST localhost:8000/retrieve -H 'Content-Type: application/json' \
            -d '{"queries":["test"],"topk":1}' | grep -q result; then ok=1; break; fi
    sleep 15
done
if [ $ok = 0 ]; then
    echo "retriever not ready after 30 min"; tail -20 /root/autodl-tmp/retriever.log
else
    echo "[$(date +%H:%M:%S)] ready after $((i * 15 / 60)) min; $(grep -m1 'retriever threads' /root/autodl-tmp/retriever.log)"
    $VERL_PY scripts/load_test_retriever.py --clients "$CLIENTS" --seconds "$SECONDS_RUN" \
        --out "results/retriever_load_${TAG}.json"
    $VERL_PY scripts/diagnose_retriever.py "$STATS" --out "results/retriever_diag_${TAG}.json" | head -60
fi
[ "${SHUTDOWN:-0}" = 1 ] && /usr/bin/shutdown
