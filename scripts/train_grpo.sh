#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GROUND_R1_OPEN_R1="${GROUND_R1_OPEN_R1:-$ROOT/../Ground-R1/r1-v/src/open_r1}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$ROOT/../Ground-R1/r1-v/local_scripts/zero2.json}"

: "${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH to a Qwen2.5-VL checkpoint.}"
: "${DATASET_JSONL:?Set DATASET_JSONL to a CerviCoT/Ground-R1 JSONL file.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for checkpoints.}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
absolute_path() { "$PYTHON_BIN" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$1"; }
GROUND_R1_OPEN_R1="$(absolute_path "$GROUND_R1_OPEN_R1")"
DATASET_JSONL="$(absolute_path "$DATASET_JSONL")"
OUTPUT_DIR="$(absolute_path "$OUTPUT_DIR")"
DEEPSPEED_CONFIG="$(absolute_path "$DEEPSPEED_CONFIG")"
if [[ -e "$MODEL_NAME_OR_PATH" ]]; then MODEL_NAME_OR_PATH="$(absolute_path "$MODEL_NAME_OR_PATH")"; fi
MASTER_PORT="${MASTER_PORT:-12345}"
DEEPSPEED_ARGS=()
if [[ "${USE_DEEPSPEED:-1}" == "1" ]]; then
  DEEPSPEED_ARGS=(--deepspeed "$DEEPSPEED_CONFIG")
fi

torchrun --nproc_per_node="$NPROC_PER_NODE" \
  --master_port="$MASTER_PORT" \
  "$ROOT/scripts/train_cervithink_grpo.py" \
  --ground-r1-open-r1 "$GROUND_R1_OPEN_R1" \
  --dataset_name "$DATASET_JSONL" \
  --output_dir "$OUTPUT_DIR" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --max_context_tokens "${MAX_CONTEXT_TOKENS:-8192}" \
  --max_completion_length 256 \
  --num_generations 8 \
  --num_generations_stage1 4 \
  --answer_rollouts 2 \
  --enable_dvhr_auxiliary true \
  --dvhr_aux_batch_size "${DVHR_AUX_BATCH_SIZE:-4}" \
  --beta 0.0 \
  --clip_epsilon 0.2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --learning_rate "${LEARNING_RATE:-5e-7}" \
  "${DEEPSPEED_ARGS[@]}" \
  --logging_steps 1 \
  --bf16 \
  --gradient_checkpointing false \
  --attn_implementation flash_attention_2 \
  --max_pixels 401408 \
  --num_train_epochs 2 \
  --save_steps 100 \
  --save_only_model true \
  --eval_strategy no \
  --report_to none \
  --seed "${SEED:-42}" \
  "${@}"
