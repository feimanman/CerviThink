#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QWEN_FINETUNE_ROOT="${QWEN_FINETUNE_ROOT:-$ROOT/../Ground-R1/qwen-vl-finetune}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$QWEN_FINETUNE_ROOT/scripts/zero2.json}"

: "${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH to a Qwen2.5-VL checkpoint.}"
: "${DATASET_JSONL:?Set DATASET_JSONL to a CerviCoT JSONL file.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for checkpoints.}"

SFT_JSON="$QWEN_FINETUNE_ROOT/qwenvl/data/Ground_SFT.json"
DEEPSPEED_ARGS=()
if [[ "${USE_DEEPSPEED:-1}" == "1" ]]; then
  DEEPSPEED_ARGS=(--deepspeed "$DEEPSPEED_CONFIG")
fi

python3 "$ROOT/scripts/prepare_qwen_sft.py" \
  --input "$DATASET_JSONL" \
  --output "$SFT_JSON" \
  --split "${SFT_SPLIT:-train}"

cd "$QWEN_FINETUNE_ROOT"

torchrun --nproc_per_node="${NPROC_PER_NODE:-1}" \
  qwenvl/train/train_qwen.py \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --dataset_use data_33K_2stage \
  --data_flatten True \
  --tune_mm_vision False \
  --tune_mm_mlp True \
  --tune_mm_llm True \
  --output_dir "$OUTPUT_DIR" \
  --bf16 true \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-3}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-2}" \
  --learning_rate "${LEARNING_RATE:-1e-6}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}" \
  "${DEEPSPEED_ARGS[@]}" \
  --max_pixels "${MAX_PIXELS:-401408}" \
  --min_pixels "${MIN_PIXELS:-3136}" \
  --eval_strategy "no" \
  --save_steps "${SAVE_STEPS:-500}" \
  --logging_steps 1 \
  --report_to "${REPORT_TO:-none}" \
  "${@}"
