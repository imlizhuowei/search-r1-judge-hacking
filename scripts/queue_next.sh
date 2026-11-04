#!/usr/bin/env bash
# Keep the GPU: wait for the training run that is going now to finish, then start the next job(s)
# on the same machine, without shutting down in between.
#
#   QUEUE="smoke_gca train" EXP=gca_em_s1 ADV=gca BARK_KEY=... AFTER=keep MAX_RUN_HOURS=40 \
#     setsid nohup bash scripts/queue_next.sh > /root/autodl-tmp/queue.log 2>&1 < /dev/null &
#
# Each job in QUEUE runs through scripts/on_boot.sh (retriever check, tool check, job, timing summary).
# A job that fails stops the queue (no point launching a long GCA run if the GCA smoke test failed).
set -uo pipefail
cd /root/autodl-tmp/search-r1-judge-hacking
QUEUE=${QUEUE:-train}

push() {
    local q='import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1] or " ", safe=""))'
    [ -n "${BARK_KEY:-}" ] && curl -s -m 10 \
        "https://api.day.app/${BARK_KEY}/$(python3 -c "$q" "$1")/$(python3 -c "$q" "${2:- }")" > /dev/null || true
    echo "[$(date +%H:%M:%S)] $1 $2"
}

push "排队中" "等当前训练结束后运行: $QUEUE (EXP=${EXP:-?})"
while pgrep -f "verl.trainer.main_ppo" > /dev/null; do sleep 60; done
sleep 30
/root/autodl-tmp/verl/.venv/bin/ray stop --force > /dev/null 2>&1 || true   # free GPU memory held by leftover workers
sleep 10
PREV=$(ls -t /root/autodl-fs/runs/*/train.log 2>/dev/null | head -1)
push "上一个训练已结束" "$( [ -n "$PREV" ] && python3 scripts/progress.py "$PREV" 300 2>&1 | tail -1 )"

for job in $QUEUE; do
    # last job honours AFTER; earlier ones always keep the GPU for the next job
    if [ "$job" = "${QUEUE##* }" ]; then A=${AFTER:-keep}; else A=keep; fi
    AFTER=$A bash scripts/on_boot.sh "$job"
    rc=$?
    if [ $rc -ne 0 ]; then
        push "队列停止" "$job 失败 (exit $rc)，GPU 保持开着，等你排查"
        exit $rc
    fi
done
push "队列全部完成" "$QUEUE"
