#!/usr/bin/env python3
"""Convert raw cervical annotations to CerviCoT/Ground-R1 JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cervithink.data import convert_records, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Raw JSONL annotations.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--image-root", default=None, help="Prefix for relative image paths.")
    parser.add_argument("--split-if-missing", action="store_true", help="Create stratified train/test splits.")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    converted = convert_records(
        records,
        image_root=args.image_root,
        split_if_missing=args.split_if_missing,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    write_jsonl(converted, args.output)
    print(f"Wrote {len(converted)} records to {args.output}")


if __name__ == "__main__":
    main()
