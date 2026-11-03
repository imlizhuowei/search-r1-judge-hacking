#!/usr/bin/env bash
# Real training run: Qwen2.5-3B-Instruct + search tool, 1 GPU.
#   EXP=grpo_em_s1 ADV=grpo bash scripts/train.sh
#   EXP=gca_em_s1  ADV=gca  bash scripts/train.sh
# Sized from the smoke run (11-04): ~27k tokens/step at 8x4 took ~39 s; 64x5 is ~10x the tokens.
# Expect ~4-5 min/step on one A800 -> 300 steps ~ 20-25 h (+ validation).
#
# Memory: the retriever keeps ~64 GB in CPU RAM, and the smoke run peaked at ~125 GB of 120 GB.
# So: 2 dataloader workers (default 8, each copies the dataset), ref model on GPU (no CPU offload),
# Ray object store capped at 10 GB.
#
# Checkpoints (model + optimizer, ~45 GB each) go to file storage (/root/autodl-fs), not the 50 GB data disk.
# Rerunning the same EXP resumes from the latest checkpoint (trainer.resume_mode=auto).

set -xeuo pipefail

REPO=${REPO:-/root/autodl-tmp/search-r1-judge-hacking}
VERL=${VERL:-/root/autodl-tmp/verl}
MODEL=${MODEL:-/root/autodl-tmp/models/Qwen2.5-3B-Instruct}
DATA=${DATA:-/root/autodl-tmp/data/searchR1_processed_direct}
EXP=${EXP:-grpo_em_s1}
ADV=${ADV:-grpo}
SEED=${SEED:-1}
STEPS=${STEPS:-300}
BATCH=${BATCH:-64}
N=${N:-5}
RUN_DIR=/root/autodl-fs/runs/$EXP
mkdir -p "$RUN_DIR"

VAL="$DATA/val_subset_100.parquet"
[ -f "$VAL" ] || /root/autodl-tmp/verl/.venv/bin/python "$REPO/scripts/make_val_subset.py" --data "$DATA" --per-ds 100

KL=0.001
ADV_ARGS=(algorithm.adv_estimator=grpo)
if [ "$ADV" = gca ]; then
    KL=0.005   # GCA paper App. D.3 (FM4)
    ADV_ARGS=(
        algorithm.adv_estimator=gca
        trainer.use_v1=False
        custom_reward_function.path="$REPO/gca/reward.py"
        custom_reward_function.name=compute_score
    )
fi

export UV_CACHE_DIR=/root/autodl-tmp/uv_cache
export HF_HOME=/root/autodl-tmp/hf_cache
export PYTHONPATH="${REPO}${PYTHONPATH:+:$PYTHONPATH}"
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost

cd "$VERL"
uv run --frozen --all-packages --extra vllm --extra fsdp python3 -m verl.trainer.main_ppo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$DATA/train.parquet" \
    data.val_files="$VAL" \
    data.train_batch_size=$BATCH \
    data.val_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    data.dataloader_num_workers=2 \
    data.seed=$SEED \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=$N \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=512 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$REPO/configs/search_tool.yaml" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.logger='["console"]' \
    trainer.project_name=search_r1_judge \
    trainer.experiment_name="$EXP" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.test_freq=25 \
    trainer.total_training_steps="$STEPS" \
    trainer.default_local_dir="$RUN_DIR/ckpt" \
    trainer.rollout_data_dir="$RUN_DIR/rollouts" \
    trainer.validation_data_dir="$RUN_DIR/val" \
    trainer.resume_mode=auto \
    +ray_kwargs.ray_init.object_store_memory=10737418240 \
    ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages --extra vllm --extra fsdp" \
    "${ADV_ARGS[@]}" \
    "$@"
