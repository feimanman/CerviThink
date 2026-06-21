#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMOKE_ROOT="${SMOKE_ROOT:-outputs/smoke}"

: "${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH to a small Qwen2.5-VL-compatible checkpoint.}"
: "${DATASET_JSONL:?Set DATASET_JSONL to a CerviCoT/Ground-R1 JSONL file.}"

mkdir -p "$SMOKE_ROOT"

python3 "$ROOT/scripts/check_training_environment.py" \
  --dataset-jsonl "$DATASET_JSONL" \
  --model "$MODEL_NAME_OR_PATH" \
  --output "$SMOKE_ROOT/environment.json"

if [[ "${RUN_SFT_SMOKE:-1}" == "1" ]]; then
  OUTPUT_DIR="$SMOKE_ROOT/sft" \
  NPROC_PER_NODE="${NPROC_PER_NODE:-1}" \
  NUM_TRAIN_EPOCHS=1 \
  PER_DEVICE_TRAIN_BATCH_SIZE=1 \
  GRADIENT_ACCUMULATION_STEPS=1 \
  SAVE_STEPS=100000 \
  SFT_SPLIT="${SFT_SPLIT:-train}" \
  bash "$ROOT/scripts/train_sft.sh" \
    --max_steps "${SFT_MAX_STEPS:-1}"
fi

if [[ "${RUN_GRPO_SMOKE:-1}" == "1" ]]; then
  OUTPUT_DIR="$SMOKE_ROOT/grpo" \
  NPROC_PER_NODE="${NPROC_PER_NODE:-1}" \
  DVHR_AUX_BATCH_SIZE="${DVHR_AUX_BATCH_SIZE:-1}" \
  bash "$ROOT/scripts/train_grpo.sh" \
    --max_steps "${GRPO_MAX_STEPS:-1}" \
    --save_steps 100000
fi
