#!/bin/bash
# Full supervised fine-tuning of Qwen2.5-VL-7B -> MMedAgent-Plus.
# Matches the configuration reported in the paper (Methods, "Model Training"):
#   4x A100-80GB, per-device batch 1, grad accumulation 64 (effective batch 256),
#   lr 1e-5, cosine schedule, 10% warmup, weight decay 0.01, 1 epoch, bf16,
#   208,928 training samples.
#
# Usage (from the repo root):
#   export MMEDAGENT_DATA_ROOT=/path/to/MMedAgent-Plus-data
#   bash scripts/sft_7b.sh

set -euo pipefail

# ----------------------------
# Distributed training config
# ----------------------------
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}

# ----------------------------
# Paths (override via environment variables)
# ----------------------------
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-VL-7B-Instruct"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/mmedagent-plus-7b"}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"./scripts/deepspeed/zero3_offload.json"}
: "${MMEDAGENT_DATA_ROOT:?Set MMEDAGENT_DATA_ROOT to the downloaded dataset directory}"

# ----------------------------
# Training hyperparameters
# ----------------------------
lr=1e-5
batch_size=1
grad_accum_steps=64
epochs=1.0

entry_file=./qwenvl/train/train_qwen.py
datasets=single_tool_multi_round,multi_tool_multi_round,multi_tool_single_round
run_name=${RUN_NAME:-"mmedagent-plus-7b-sft"}

# Save 4 checkpoints over the run.
total_samples=208928  # size of the training corpus reported in the paper
effective_bs=$(( NPROC_PER_NODE * batch_size * grad_accum_steps ))
total_steps=$(( total_samples / effective_bs ))
save_steps=$(( total_steps / 4 > 0 ? total_steps / 4 : 1 ))

args="
    --deepspeed ${DEEPSPEED_CONFIG} \
    --model_name_or_path ${MODEL_PATH} \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${OUTPUT_DIR} \
    --logging_dir ${OUTPUT_DIR}/tensorboard_logs \
    --report_to tensorboard \
    --num_train_epochs ${epochs} \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 12288 \
    --min_pixels 256 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps ${save_steps} \
    --save_total_limit 3 \
    --learning_rate ${lr} \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --remove_unused_columns False \
    --run_name ${run_name} \
    --seed 42 \
    --data_seed 42 \
    "

torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
