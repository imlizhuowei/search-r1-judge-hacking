#!/usr/bin/env bash
# Evaluate one model on the 7 Search-R1 test sets (51,713 questions), the way the paper reports it:
# exact match, greedy decoding, top-3 passages from the same E5 + Wikipedia-2018 retriever.
#
#   EVAL_TAG=base       bash scripts/eval.sh     # Qwen2.5-3B-Instruct, no RL
#   EVAL_TAG=grpo_em_s1 bash scripts/eval.sh     # latest checkpoint of that run (CKPT=... to pick one)
#
# How: scripts/train.sh with trainer.val_only=True, so evaluation runs the exact rollout code (tool loop,
# prompt, max 5 assistant turns, 2048 response tokens) that training used. One run per dataset, smallest
# first; a dataset whose dump already exists is skipped, so rerunning the script resumes.
# Scoring uses gca/reward.py: the same EM as verl's Search-R1 reward (incl. the >10 <answer> tags rule), plus
# the per-sample fields the badcase analysis needs (data source, hit, answered, ...).
#
# Output: /root/autodl-fs/runs/eval_<tag>/<dataset>/<step>.jsonl (one line per question), eval.log,
#         then results/eval_<tag>.json + .md (scripts/summarize_eval.py) and badcase (scripts/badcase.py).
set -uo pipefail
REPO=${REPO:-/root/autodl-tmp/search-r1-judge-hacking}
DATA=${DATA:-/root/autodl-tmp/data/searchR1_processed_direct}
PY=/root/autodl-tmp/verl/.venv/bin/python
TAG=${EVAL_TAG:?set EVAL_TAG=base or a run name like grpo_em_s1}
# SAMPLE=1000: a fixed random 1000 questions per dataset (bamboogle has 125, kept whole) instead of all 51,713.
# Same seed for every model, so models are compared on identical questions. Results are tagged <tag>_s1000.
SAMPLE=${SAMPLE:-0}
if [ "$SAMPLE" -gt 0 ]; then
    TEST_DIR="$DATA/test_by_ds_s$SAMPLE"; TAG_OUT="${TAG}_s$SAMPLE"
else
    TEST_DIR="$DATA/test_by_ds"; TAG_OUT="$TAG"
fi
OUT=/root/autodl-fs/runs/eval_$TAG_OUT
DATASETS=${DATASETS:-"bamboogle musique nq hotpotqa 2wikimultihopqa triviaqa popqa"}   # small -> large
cd "$REPO"
mkdir -p "$OUT"

push() {
    echo "[$(date +%H:%M:%S)] $1 $2"
    local q='import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1] or " ", safe=""))'
    [ -n "${BARK_KEY:-}" ] && curl -s -m 10 \
        "https://api.day.app/${BARK_KEY}/$(python3 -c "$q" "$1")/$(python3 -c "$q" "${2:- }")" > /dev/null || true
}

if [ "$TAG" = base ]; then
    RESUME=(RESUME_MODE=disable)
    EXTRA=()
    STEP_DESC="no RL (step 0)"
else
    CKPT=${CKPT:-$(ls -d /root/autodl-fs/runs/$TAG/ckpt/global_step_* 2>/dev/null | sort -V | tail -1)}
    [ -d "${CKPT:-}" ] || { push "评测失败" "找不到 $TAG 的检查点"; exit 1; }
    RESUME=(RESUME_MODE=resume_path)
    EXTRA=(trainer.resume_from_path="$CKPT")
    STEP_DESC="$(basename "$CKPT")"
fi
echo "$STEP_DESC" > "$OUT/checkpoint.txt"

[ -d "$TEST_DIR" ] || $PY scripts/split_test.py --data "$DATA" --sample "$SAMPLE" || { push "评测失败" "拆分测试集出错"; exit 1; }
[ -f "$DATA/test_strata.json" ] || $PY scripts/make_strata.py --data "$DATA" || true   # leak / answer-seen strata

for ds in $DATASETS; do
    f="$TEST_DIR/$ds.parquet"
    [ -f "$f" ] || { push "评测：跳过 $ds" "没有 $f"; continue; }
    if ls "$OUT/$ds"/*.jsonl > /dev/null 2>&1; then
        echo "[$(date +%H:%M:%S)] $ds already done, skipping"; continue
    fi
    # sampled eval: if the full eval of this dataset already finished, just take the sampled rows from it
    if [ "$SAMPLE" -gt 0 ] && n=$($PY scripts/subset_dump.py "/root/autodl-fs/runs/eval_$TAG/$ds" "$f" "$OUT/$ds" 2>/dev/null); then
        push "评测 $TAG_OUT：$ds 复用完整评测" "$n 题，EM $($PY scripts/summarize_eval.py "$OUT" --one "$ds" 2>/dev/null | tail -1)"
        continue
    fi
    t0=$(date +%s)
    env "${RESUME[@]}" EXP="eval_${TAG_OUT}_${ds}" ADV=grpo STEPS=1 SAVE_FREQ=-1 TEST_FREQ=-1 VAL_BEFORE_TRAIN=True \
        VAL_FILE="$f" VAL_DIR="$OUT/$ds" VAL_BATCH=${VAL_BATCH:-256} \
        bash scripts/train.sh \
            trainer.val_only=True \
            reward.custom_reward_function.path="$REPO/gca/reward.py" \
            reward.custom_reward_function.name=compute_score \
            "${EXTRA[@]}" >> "$OUT/eval.log" 2>&1
    rc=$?
    n=$(cat "$OUT/$ds"/*.jsonl 2>/dev/null | wc -l)
    if [ $rc -ne 0 ] || [ "$n" = 0 ]; then
        push "评测出错 $TAG_OUT/$ds" "exit $rc，看 $OUT/eval.log 末尾"; exit 1
    fi
    em=$($PY scripts/summarize_eval.py "$OUT" --one "$ds" 2>/dev/null | tail -1)
    push "评测 $TAG_OUT：$ds 完成" "$n 题，EM $em，用时 $(( ($(date +%s) - t0) / 60 )) 分钟"
done

$PY scripts/summarize_eval.py "$OUT" --tag "$TAG_OUT" > /dev/null
$PY scripts/summarize_eval.py --table > /dev/null
$PY scripts/badcase.py "$OUT" --tag "$TAG_OUT" > /dev/null
push "评测 $TAG_OUT 全部完成" "$($PY scripts/summarize_eval.py "$OUT" --tag "$TAG_OUT" --line 2>&1 | tail -1)"
