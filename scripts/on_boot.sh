#!/usr/bin/env bash
# Runs ON the server right after a GPU boot (started by tools/grab_gpu.py over SSH, or by hand).
#   bash scripts/on_boot.sh smoke        # retriever + tool check + 3-step GRPO smoke run
#   bash scripts/on_boot.sh smoke_gca    # same, but ADV=gca
# Progress goes to /root/autodl-tmp/on_boot.log; each stage pushes to your phone if BARK_KEY is set.
set -uo pipefail

JOB=${1:-smoke}
REPO=/root/autodl-tmp/search-r1-judge-hacking
LOCK=/root/autodl-tmp/on_boot.lock
cd "$REPO"

push() {  # push "title" "body"
    echo "[$(date +%H:%M:%S)] $1 $2"
    [ -n "${BARK_KEY:-}" ] && curl -s -m 10 -G "https://api.day.app/${BARK_KEY}/$(python3 -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1]))' "$1")" \
        --data-urlencode "body=$2" > /dev/null || true
}

if [ -e "$LOCK" ] && kill -0 "$(cat $LOCK)" 2>/dev/null; then
    echo "on_boot already running (pid $(cat $LOCK))"; exit 0
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { push "没有 GPU" "nvidia-smi 失败"; exit 1; }
git pull -q || true
bash scripts/apply_verl_patch.sh > /dev/null 2>&1 || true

# 1. retriever (CPU faiss, ~64 GB RAM; first start rescans corpus offsets)
if ! curl -s -m 5 -X POST localhost:8000/retrieve -H 'Content-Type: application/json' \
        -d '{"queries":["test"],"topk":1}' > /dev/null; then
    nohup bash retriever/launch.sh > /root/autodl-tmp/retriever.log 2>&1 &
    push "检索服务启动中" "首次要扫描偏移量，可能要几十分钟"
fi
for i in $(seq 1 240); do   # wait up to 2 h
    curl -s -m 5 -X POST localhost:8000/retrieve -H 'Content-Type: application/json' \
        -d '{"queries":["test"],"topk":1}' > /dev/null && break
    sleep 30
done
curl -s -m 5 -X POST localhost:8000/retrieve -H 'Content-Type: application/json' -d '{"queries":["test"],"topk":1}' > /dev/null \
    || { push "检索服务没起来" "看 retriever.log"; exit 1; }
push "检索服务就绪" ""

# 2. tool check against the real retriever
/root/autodl-tmp/verl/.venv/bin/python -m agent.search_tool > /root/autodl-tmp/tool_check.log 2>&1 \
    && grep -q "'status': 'ok'" /root/autodl-tmp/tool_check.log \
    || { push "搜索工具检查失败" "看 tool_check.log"; exit 1; }
push "搜索工具正常" ""

# 3. job
case "$JOB" in
    smoke)     ADV=grpo bash scripts/smoke_train.sh > /root/autodl-tmp/smoke.log 2>&1 ;;
    smoke_gca) ADV=gca  bash scripts/smoke_train.sh > /root/autodl-tmp/smoke_gca.log 2>&1 ;;
    none)      push "环境就绪" "未启动训练"; exit 0 ;;
    *)         push "未知任务" "$JOB"; exit 1 ;;
esac
rc=$?
if [ $rc -eq 0 ]; then push "冒烟训练完成 ($JOB)" "看日志"; else push "冒烟训练报错 ($JOB)" "exit $rc，看日志末尾"; fi
