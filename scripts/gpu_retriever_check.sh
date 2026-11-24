#!/usr/bin/env bash
# Option B check (LOG 11-24), unattended, three parts; each writes its own result, a failed part does not stop the next:
#   1. bench:  GPU brute-force search alone: speed, top-3 agreement with exact fp32, GPU memory
#              -> results/bench_gpu_search.json
#   2. load:   retriever server with the index on the GPU + 64-client load test
#              -> results/retriever_load_gpuflat_64c.json
#   3. train:  a few GRPO steps from the base model WITH that retriever on the same GPU: does it all fit, how long is
#              a step now that searches work? (vLLM share and FSDP offload set below)
#              -> /root/autodl-tmp/gpu_trial.log (console), results/timing_gpu_trial.json, nvidia-smi peak
#   SHUTDOWN=1 setsid nohup bash scripts/gpu_retriever_check.sh > /root/autodl-tmp/gpu_check.log 2>&1 < /dev/null &
set -uo pipefail
REPO=${REPO:-/root/autodl-tmp/search-r1-judge-hacking}
RPY=/root/autodl-tmp/envs/retriever/bin/python
VPY=/root/autodl-tmp/verl/.venv/bin/python
cd "$REPO"
mkdir -p results
say() { echo "[$(date +%H:%M:%S)] $*"; }

pkill -f retrieval_server; pkill -f retriever/launch.sh; pkill -f verl.trainer.main_ppo; sleep 5
say "GPU before: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"

if [ "${SKIP_BENCH:-0}" = 1 ]; then say "== 1. bench skipped"; else
    say "== 1. bench"
    $RPY retriever/bench_gpu_search.py || say "bench FAILED (exit $?)"
fi

say "== 2. retriever on GPU + load test"
SHUTDOWN=0 USE_GPU_FLAT=1 bash scripts/measure_retriever.sh || say "load test FAILED (exit $?)"
say "GPU with retriever: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"

if curl -s -m 30 -X POST localhost:8000/retrieve -H 'Content-Type: application/json' \
        -d '{"queries":["test"],"topk":1}' | grep -q result; then
    say "== 3. training trial: ${TRIAL_STEPS:-3} steps, VLLM_MEM=${VLLM_MEM:-0.3} OPTIM_OFFLOAD=${OPTIM_OFFLOAD:-True} PARAM_OFFLOAD=${PARAM_OFFLOAD:-False} MICRO_BS=${MICRO_BS:-4} REF_OFFLOAD=${REF_OFFLOAD:-False}"
    bash -c 'while sleep 10; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; done' \
        > /root/autodl-tmp/gpu_trial_mem.txt 2>/dev/null &
    MON=$!
    rm -rf /root/autodl-fs/runs/gpu_trial
    EXP=gpu_trial ADV=grpo STEPS=${TRIAL_STEPS:-3} SAVE_FREQ=-1 TEST_FREQ=-1 VAL_BEFORE_TRAIN=False RESUME_MODE=disable \
        VLLM_MEM=${VLLM_MEM:-0.3} OPTIM_OFFLOAD=${OPTIM_OFFLOAD:-True} PARAM_OFFLOAD=${PARAM_OFFLOAD:-False} \
        bash scripts/train.sh > /root/autodl-tmp/gpu_trial.log 2>&1
    rc=$?
    kill "$MON" 2>/dev/null || true
    say "training trial exit $rc; GPU peak $(sort -n /root/autodl-tmp/gpu_trial_mem.txt | tail -1) MiB"
    grep -E "OutOfMemory|out of memory|CUDA error|Error|error:" /root/autodl-tmp/gpu_trial.log | grep -v "Search error" | tail -8
    grep -oE "step:[0-9]+ .*timing_s/step:[0-9.]+" /root/autodl-tmp/gpu_trial.log 2>/dev/null \
        | grep -oE "step:[0-9]+|timing_s/(gen|update_actor|step):[0-9.]+" | paste -sd' ' | sed 's/ step:/\nstep:/g'
    python3 scripts/summarize_timing.py /root/autodl-tmp/gpu_trial.log /dev/null gpu_trial > /dev/null 2>&1 || true
    grep -c "Search error" /root/autodl-fs/runs/gpu_trial/rollouts/*.jsonl 2>/dev/null | head -5
else
    say "== 3. skipped: retriever not answering"
fi
say "done"
[ "${SHUTDOWN:-0}" = 1 ] && /usr/bin/shutdown
