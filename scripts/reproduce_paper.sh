#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/reproduce}"

: "${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH to the Qwen2.5-VL-7B-Instruct checkpoint.}"
: "${PUBMED_KNOWLEDGE_JSONL:?Set PUBMED_KNOWLEDGE_JSONL to local PubMed retrieval snippets.}"

mkdir -p "$OUTPUT_ROOT"

prepare_dataset() {
  local name="$1"
  local annotations_var="$2"
  local image_root_var="$3"
  local test_ratio="$4"
  local resize_full_image="${5:-0}"
  local annotations="${!annotations_var:-}"
  local image_root="${!image_root_var:-}"

  if [[ -z "$annotations" ]]; then
    echo "Skip $name: $annotations_var is not set."
    return
  fi

  local cropped_jsonl="$OUTPUT_ROOT/${name}_cropped.jsonl"
  local rationale_jsonl="$OUTPUT_ROOT/${name}_rationales.jsonl"
  local cervicot_jsonl="$OUTPUT_ROOT/${name}_cervicot.jsonl"
  local crop_dir="$OUTPUT_ROOT/${name}_256"
  local resize_args=()
  if [[ "$resize_full_image" == "1" ]]; then
    resize_args=(--resize-full-image)
  fi

  python3 "$ROOT/scripts/prepare_cell_crops.py" \
    --input "$annotations" \
    --output "$cropped_jsonl" \
    --output-image-dir "$crop_dir" \
    --image-root "$image_root" \
    --dataset "$name" \
    --split-if-missing \
    --test-ratio "$test_ratio" \
    --seed "${SEED:-42}" \
    "${resize_args[@]}"

  if [[ -n "${CERVICOT_GENERATOR_CMD:-}" || "${CERVICOT_USE_FALLBACK_TEMPLATE:-0}" == "1" ]]; then
    local rationale_args=()
    if [[ -n "${CERVICOT_GENERATOR_CMD:-}" ]]; then
      rationale_args=(--generator-cmd "$CERVICOT_GENERATOR_CMD")
    else
      rationale_args=(--fallback-template)
    fi

    python3 "$ROOT/scripts/generate_cervicot_rationales.py" \
      --annotations "$cropped_jsonl" \
      --knowledge "$PUBMED_KNOWLEDGE_JSONL" \
      --output "$rationale_jsonl" \
      --top-k "${RAG_TOP_K:-3}" \
      "${rationale_args[@]}"

    python3 "$ROOT/scripts/build_cervicot_rag.py" \
      --annotations "$rationale_jsonl" \
      --knowledge "$PUBMED_KNOWLEDGE_JSONL" \
      --output "$cervicot_jsonl" \
      --top-k "${RAG_TOP_K:-3}"
  else
    python3 "$ROOT/scripts/build_cervicot_rag.py" \
      --annotations "$cropped_jsonl" \
      --knowledge "$PUBMED_KNOWLEDGE_JSONL" \
      --output "$cervicot_jsonl" \
      --top-k "${RAG_TOP_K:-3}"
  fi
}

prepare_dataset "DST" "DST_ANNOTATIONS" "DST_IMAGE_ROOT" "${DST_TEST_RATIO:-0.2}"
prepare_dataset "ComparisonDetector" "COMPARISON_ANNOTATIONS" "COMPARISON_IMAGE_ROOT" "${FEWSHOT_TEST_RATIO:-0.9}"
prepare_dataset "HiCervix" "HICERVIX_ANNOTATIONS" "HICERVIX_IMAGE_ROOT" "${FEWSHOT_TEST_RATIO:-0.9}" "${HICERVIX_RESIZE_FULL_IMAGE:-0}"

if [[ "${RUN_TRAINING:-1}" == "1" ]]; then
  : "${TRAIN_DATASET_JSONL:?Set TRAIN_DATASET_JSONL, for example $OUTPUT_ROOT/DST_cervicot.jsonl.}"
  : "${SFT_OUTPUT_DIR:?Set SFT_OUTPUT_DIR.}"
  : "${GRPO_OUTPUT_DIR:?Set GRPO_OUTPUT_DIR.}"

  DATASET_JSONL="$TRAIN_DATASET_JSONL" OUTPUT_DIR="$SFT_OUTPUT_DIR" bash "$ROOT/scripts/train_sft.sh"
  MODEL_NAME_OR_PATH="$SFT_OUTPUT_DIR" DATASET_JSONL="$TRAIN_DATASET_JSONL" OUTPUT_DIR="$GRPO_OUTPUT_DIR" bash "$ROOT/scripts/train_grpo.sh"
fi

if [[ "${RUN_EVAL:-1}" == "1" ]]; then
  : "${EVAL_MODEL:?Set EVAL_MODEL to the final checkpoint, for example GRPO_OUTPUT_DIR.}"
  for dataset_jsonl in "$OUTPUT_ROOT"/*_cervicot.jsonl; do
    [[ -e "$dataset_jsonl" ]] || continue
    name="$(basename "$dataset_jsonl" _cervicot.jsonl)"
    python3 "$ROOT/scripts/evaluate_dataset.py" \
      --model "$EVAL_MODEL" \
      --input "$dataset_jsonl" \
      --output "$OUTPUT_ROOT/${name}_predictions.jsonl" \
      --report-output "$OUTPUT_ROOT/${name}_metrics.json"
  done
fi

if [[ "${RUN_BASELINES:-0}" == "1" ]]; then
  : "${BASELINE_CONFIG:?Set BASELINE_CONFIG to a baseline JSON config.}"
  for dataset_jsonl in "$OUTPUT_ROOT"/*_cervicot.jsonl; do
    [[ -e "$dataset_jsonl" ]] || continue
    name="$(basename "$dataset_jsonl" _cervicot.jsonl)"
    DATASET_JSONL="$dataset_jsonl" \
      OUTPUT_DIR="$OUTPUT_ROOT/baselines/$name" \
      BASELINE_CONFIG="$BASELINE_CONFIG" \
      bash "$ROOT/scripts/run_hf_baselines.sh"
  done
fi
