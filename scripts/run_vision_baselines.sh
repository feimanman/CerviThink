#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${DATASET_JSONL:?Set DATASET_JSONL to a CerviCoT/Ground-R1 JSONL file.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for baseline outputs.}"

BASELINE_CONFIG="${BASELINE_CONFIG:-$ROOT/configs/vision_baselines.json.example}"
mkdir -p "$OUTPUT_DIR"

python3 - <<'PY' "$BASELINE_CONFIG" "$OUTPUT_DIR/.vision_baseline_commands.sh" "$ROOT" "$DATASET_JSONL" "$OUTPUT_DIR"
import json
import shlex
import sys
from pathlib import Path

config_path, command_path, root, dataset_jsonl, output_dir = sys.argv[1:]
baselines = json.loads(Path(config_path).read_text(encoding="utf-8"))
lines = ["set -euo pipefail"]
for item in baselines:
    name = item["name"]
    out_dir = str(Path(output_dir) / name)
    command = [
        "python3",
        f"{root}/scripts/train_vision_baseline.py",
        "--input",
        dataset_jsonl,
        "--output-dir",
        out_dir,
        "--arch",
        item["arch"],
        "--weights",
        item.get("weights", "imagenet"),
        "--epochs",
        str(item.get("epochs", 30)),
        "--batch-size",
        str(item.get("batch_size", 32)),
        "--learning-rate",
        str(item.get("learning_rate", 1e-4)),
    ]
    if "image_root" in item:
        command.extend(["--image-root", item["image_root"]])
    lines.append(" ".join(shlex.quote(part) for part in command))
Path(command_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

bash "$OUTPUT_DIR/.vision_baseline_commands.sh"
