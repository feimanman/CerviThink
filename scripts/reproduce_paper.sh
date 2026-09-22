#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${EXPERIMENT_CONFIG:?Set EXPERIMENT_CONFIG to an explicit experiment JSON; see configs/method_experiments.json.example.}"
exec "${PYTHON_BIN:-python3}" "$ROOT/scripts/run_experiments.py" --config "$EXPERIMENT_CONFIG" "$@"
