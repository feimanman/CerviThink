#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${BASELINE_CONFIG:?Set BASELINE_CONFIG to a JSON file like configs/baselines.json.example.}"
: "${DATASET_JSONL:?Set DATASET_JSONL to a CerviCoT/Ground-R1 JSONL file.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for baseline predictions.}"

mkdir -p "$OUTPUT_DIR"

python3 - <<'PY' "$BASELINE_CONFIG" "$OUTPUT_DIR/.baseline_commands.sh" "$ROOT" "$DATASET_JSONL" "$OUTPUT_DIR"
import json
import shlex
import sys

config_path, command_path, root, dataset_jsonl, output_dir = sys.argv[1:]
with open(config_path, "r", encoding="utf-8") as handle:
    baselines = json.load(handle)

commands = []
for item in baselines:
    name = item["name"]
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    family = item.get("family", "auto")
    model = item["model"]
    command = [
        "python3",
        f"{root}/scripts/evaluate_hf_baseline.py",
        "--model",
        model,
        "--family",
        family,
        "--input",
        dataset_jsonl,
        "--output",
        f"{output_dir}/{safe_name}_predictions.jsonl",
        "--report-output",
        f"{output_dir}/{safe_name}_metrics.json",
    ]
    commands.append(" ".join(shlex.quote(part) for part in command))

with open(command_path, "w", encoding="utf-8") as handle:
    handle.write("\n".join(commands) + "\n")
PY

bash "$OUTPUT_DIR/.baseline_commands.sh"
