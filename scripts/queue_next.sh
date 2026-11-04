#!/usr/bin/env bash
# Keep the GPU busy without shutting down between jobs.
#
# 1) waits for the training run that is going now (if any) to finish,
# 2) runs the jobs in $QUEUE one by one,
# 3) then keeps reading new jobs from $QUEUE_FILE (one per line) — append to it at any time:
#        echo "EXP=judge_grpo_s1 ADV=grpo train" >> /root/autodl-tmp/queue.txt
#        echo "shutdown"                         >> /root/autodl-tmp/queue.txt
#    A line is: [VAR=value ...] <job>, where <job> is an on_boot.sh job (train, smoke, smoke_gca, none)
#    or the word "shutdown". When the file is empty the GPU stays on (unless AFTER=shutdown),
#    but only for IDLE_HOURS (default 5): idle longer than that -> shut down.
#
#   QUEUE="smoke_gca train" EXP=gca_em_s1 ADV=gca BARK_KEY=... MAX_RUN_HOURS=40 \
#     setsid nohup bash scripts/queue_next.sh > /root/autodl-tmp/queue.log 2>&1 < /dev/null &
#
# A failed job stops the queue and keeps the GPU on for debugging.
set -uo pipefail
REPO=${REPO:-/root/autodl-tmp/search-r1-judge-hacking}
QUEUE_FILE=${QUEUE_FILE:-/root/autodl-tmp/queue.txt}
ON_BOOT=${ON_BOOT:-bash scripts/on_boot.sh}
POLL=${POLL:-60}
IDLE_HOURS=${IDLE_HOURS:-5}
cd "$REPO"
touch "$QUEUE_FILE"

push() {
    local q='import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1] or " ", safe=""))'
    [ -n "${BARK_KEY:-}" ] && curl -s -m 10 \
        "https://api.day.app/${BARK_KEY}/$(python3 -c "$q" "$1")/$(python3 -c "$q" "${2:- }")" > /dev/null || true
    echo "[$(date +%H:%M:%S)] $1 $2"
}

pop_line() {  # print and remove the first non-empty, non-comment line of the queue file
    local line
    line=$(grep -m1 -vE '^\s*(#|$)' "$QUEUE_FILE" || true)
    [ -z "$line" ] && return 1
    python3 - "$QUEUE_FILE" "$line" <<'PY'
import sys
path, target = sys.argv[1], sys.argv[2]
lines = open(path).read().splitlines(True)
for i, l in enumerate(lines):
    if l.strip() == target.strip():
        del lines[i]
        break
open(path, "w").writelines(lines)
PY
    echo "$line"
}

run_line() {  # run "[VAR=value ...] job"
    local -a parts
    read -ra parts <<< "$1"
    local job=${parts[-1]}
    if [ "$job" = shutdown ]; then
        push "队列：关机" "1 分钟后关机"
        sleep 60
        /usr/bin/shutdown
        exit 0
    fi
    env "${parts[@]:0:${#parts[@]}-1}" AFTER=keep $ON_BOOT "$job"
}

push "排队中" "等当前训练结束；之后: ${QUEUE:-<空>}，再读 $QUEUE_FILE"
while pgrep -f "verl.trainer.main_ppo" > /dev/null; do sleep "$POLL"; done
if [ "${SKIP_WAIT_CLEANUP:-0}" != 1 ]; then
    sleep 30
    /root/autodl-tmp/verl/.venv/bin/ray stop --force > /dev/null 2>&1 || true   # free GPU memory from leftover workers
    sleep 10
    PREV=$(ls -t /root/autodl-fs/runs/*/train.log 2>/dev/null | head -1)
    push "上一个训练已结束" "$( [ -n "$PREV" ] && python3 scripts/progress.py "$PREV" 300 2>&1 | tail -1 )"
fi

for job in ${QUEUE:-}; do
    AFTER=keep $ON_BOOT "$job"
    rc=$?
    if [ $rc -ne 0 ]; then push "队列停止" "$job 失败 (exit $rc)，GPU 保持开着，等你排查"; exit $rc; fi
done

idle_pushed=0
idle_since=0
while true; do
    if line=$(pop_line); then
        idle_pushed=0
        idle_since=0
        push "队列：开始" "$line"
        run_line "$line"
        rc=$?
        if [ $rc -ne 0 ]; then push "队列停止" "$line 失败 (exit $rc)，GPU 保持开着"; exit $rc; fi
    else
        if [ "${AFTER:-keep}" = shutdown ]; then run_line "shutdown"; fi
        if [ $idle_pushed = 0 ]; then
            idle_since=$(date +%s)
            push "队列空了，GPU 空闲中" "${IDLE_HOURS} 小时内没有新任务就自动关机。追加: echo '... train' >> $QUEUE_FILE"
            idle_pushed=1
        fi
        if [ $(( $(date +%s) - idle_since )) -ge $(awk "BEGIN{print int(${IDLE_HOURS}*3600)}") ]; then
            push "空闲超过 ${IDLE_HOURS} 小时" "自动关机"
            run_line "shutdown"
        fi
        [ "${EXIT_WHEN_EMPTY:-0}" = 1 ] && exit 0
        sleep "$POLL"
    fi
done
