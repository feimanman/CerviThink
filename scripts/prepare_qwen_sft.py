#!/usr/bin/env python3
"""Convert CerviCoT/Ground-R1 JSONL to qwen-vl-finetune SFT JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cervithink.data import read_jsonl
from cervithink.prompts import sft_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CerviCoT/Ground-R1 JSONL.")
    parser.add_argument("--output", required=True, help="Qwen SFT JSON output.")
    parser.add_argument("--split", default="train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    out = []
    for record in records:
        if record.get("split", args.split) != args.split:
            continue
        width = int(record.get("input_width", record.get("width", 1024)))
        height = int(record.get("input_height", record.get("height", 1024)))
        prompt = "<image>\n" + sft_prompt(record["problem"], width, height)
        out.append(
            {
                "id": str(record.get("problem_id", len(out) + 1)),
                "image": record["image"],
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": record["solution"]},
                ],
            }
        )

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(out, handle, ensure_ascii=False, indent=2)
    print(f"Wrote {len(out)} SFT records to {args.output}")


if __name__ == "__main__":
    main()
