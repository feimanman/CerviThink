#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QWEN_FINETUNE_ROOT="${QWEN_FINETUNE_ROOT:-$ROOT/../Ground-R1/qwen-vl-finetune}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$QWEN_FINETUNE_ROOT/scripts/zero2.json}"

: "${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH to a Qwen2.5-VL checkpoint.}"
: "${DATASET_JSONL:?Set DATASET_JSONL to a CerviCoT JSONL file.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for checkpoints.}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
absolute_path() { "$PYTHON_BIN" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$1"; }
QWEN_FINETUNE_ROOT="$(absolute_path "$QWEN_FINETUNE_ROOT")"
DATASET_JSONL="$(absolute_path "$DATASET_JSONL")"
OUTPUT_DIR="$(absolute_path "$OUTPUT_DIR")"
DEEPSPEED_CONFIG="$(absolute_path "$DEEPSPEED_CONFIG")"
if [[ -e "$MODEL_NAME_OR_PATH" ]]; then MODEL_NAME_OR_PATH="$(absolute_path "$MODEL_NAME_OR_PATH")"; fi
SFT_JSON="$OUTPUT_DIR/prepared_sft.json"
DEEPSPEED_ARGS=()
if [[ "${USE_DEEPSPEED:-1}" == "1" ]]; then
  DEEPSPEED_ARGS=(--deepspeed "$DEEPSPEED_CONFIG")
fi

"$PYTHON_BIN" "$ROOT/scripts/prepare_qwen_sft.py" \
  --input "$DATASET_JSONL" \
  --output "$SFT_JSON" \
  --split "${SFT_SPLIT:-train}"

torchrun --nproc_per_node="${NPROC_PER_NODE:-1}" \
  --master_port="${MASTER_PORT:-12345}" \
  "$ROOT/scripts/train_cervithink_sft.py" \
  --qwen-finetune-root "$QWEN_FINETUNE_ROOT" \
  --sft-json "$SFT_JSON" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --dataset_use cervithink_local \
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
  --model_max_length "${SFT_MAX_LENGTH:-4096}" \
  --eval_strategy "no" \
  --save_steps "${SAVE_STEPS:-500}" \
  --logging_steps 1 \
  --report_to "${REPORT_TO:-none}" \
  --seed "${SEED:-42}" \
  "${@}"
