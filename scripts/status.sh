#!/usr/bin/env bash
# One-screen status of whatever is running (for when the phone push is not available):
#   bash scripts/status.sh
# Read-only: changes nothing.
cd /root/autodl-tmp/search-r1-judge-hacking 2>/dev/null || cd "$(dirname "$0")/.."
PY=/root/autodl-tmp/verl/.venv/bin/python
echo "== 现在 $(date '+%m-%d %H:%M')"

if curl -s -m 10 -X POST localhost:8000/retrieve -H 'Content-Type: application/json' \
        -d '{"queries":["test"],"topk":1}' | grep -q document; then
    echo "== 检索服务：正常"
else
    echo "== 检索服务：没有响应（刚启动的话要等约 5 分钟加载索引）"
    tail -2 /root/autodl-tmp/retriever.log 2>/dev/null | cut -c1-160
fi

echo "== GPU：$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)"
echo "== 队列最近 4 条："
tail -4 /root/autodl-tmp/queue.log 2>/dev/null | cut -c1-160
echo "== 队列里还没跑的：$(grep -vcE '^\s*(#|$)' /root/autodl-tmp/queue.txt 2>/dev/null) 条"
grep -vE '^\s*(#|$)' /root/autodl-tmp/queue.txt 2>/dev/null | sed 's/^/     /'

for d in $(ls -dt /root/autodl-fs/runs/eval_* 2>/dev/null | head -3); do
    echo "== $(basename "$d")"
    for ds in bamboogle musique nq hotpotqa 2wikimultihopqa triviaqa popqa; do
        f=$(ls "$d/$ds"/*.jsonl 2>/dev/null | tail -1)
        if [ -n "$f" ]; then
            printf "     %-16s 完成  %5s 题  EM %s\n" "$ds" "$(wc -l < "$f")" \
                "$($PY scripts/summarize_eval.py "$d" --one "$ds" 2>/dev/null | tail -1)"
        fi
    done
done
if pgrep -f "verl.trainer.main_ppo" > /dev/null; then
    echo "== 正在运行：$(pgrep -af 'verl.trainer.main_ppo' | grep -o 'val_files=[^ ]*' | head -1 | xargs -r basename)"
else
    echo "== 没有评测/训练在跑"
fi
