#!/usr/bin/env python3
"""Evaluate cervical classification predictions from JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cervithink.data import read_jsonl
from cervithink.metrics import classification_report
from cervithink.parsing import extract_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSONL with label and prediction fields.")
    parser.add_argument("--label-key", default="label")
    parser.add_argument("--prediction-key", default="prediction")
    parser.add_argument("--output", default=None, help="Optional JSON report output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    y_true = [row[args.label_key] for row in rows]
    y_pred = [extract_answer(row[args.prediction_key]) for row in rows]
    report = classification_report(y_true, y_pred)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
