#!/usr/bin/env bash
# Smoke run: Qwen2.5-3B-Instruct + search tool + EM reward + GRPO, 1 GPU, 3 steps.
# Goal is "the pipeline runs end to end", not "the model learns".
# Prereq: retriever is up on :8000 (bash retriever/launch.sh) and test_query.sh works.
#
# Usage (from anywhere):  bash scripts/smoke_train.sh  [extra hydra overrides]

set -xeuo pipefail

REPO=${REPO:-/root/autodl-tmp/search-r1-judge-hacking}
VERL=${VERL:-/root/autodl-tmp/verl}
MODEL=${MODEL:-/root/autodl-tmp/models/Qwen2.5-3B-Instruct}
DATA=${DATA:-/root/autodl-tmp/data/searchR1_processed_direct}
STEPS=${STEPS:-3}
ADV=${ADV:-grpo}   # grpo | gca   (gca needs: bash scripts/apply_verl_patch.sh, once)

KL=0.001
if [ "$ADV" = gca ]; then
    KL=0.005   # GCA paper: KL must scale with the residual (App. D.3, FM4)
    ADV_ARGS=(
        algorithm.adv_estimator=gca
        trainer.use_v1=False   # legacy trainer merges reward extra info into the batch
        # NOTE: must be reward.custom_reward_function.* — the top-level custom_reward_function is silently ignored
        #       by the reward loop in this verl version (only fully_async migrates it)
        reward.custom_reward_function.path="$REPO/gca/reward.py"
        reward.custom_reward_function.name=compute_score
    )
else
    ADV_ARGS=(algorithm.adv_estimator=grpo)
fi

export UV_CACHE_DIR=/root/autodl-tmp/uv_cache
export HF_HOME=/root/autodl-tmp/hf_cache
# Our tool lives in this repo; Ray workers inherit this env var from the driver.
export PYTHONPATH="${REPO}${PYTHONPATH:+:$PYTHONPATH}"
# Don't send localhost retrieval traffic through any proxy.
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost

cd "$VERL"   # uv run --frozen needs verl's pyproject/uv.lock

uv run --frozen --all-packages --extra vllm --extra fsdp python3 -m verl.trainer.main_ppo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$DATA/train.parquet" \
    data.val_files="$DATA/test.parquet" \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    data.dataloader_num_workers=2 \
    reward.num_workers=2 \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
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
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=512 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$REPO/configs/search_tool.yaml" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console"]' \
    trainer.project_name=search_r1_judge \
    trainer.experiment_name=smoke_3b \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_training_steps="$STEPS" \
    trainer.default_local_dir=/root/autodl-tmp/ckpt/smoke \
    ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages --extra vllm --extra fsdp" \
    "${ADV_ARGS[@]}" \
    "$@"
