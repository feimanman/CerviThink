#!/usr/bin/env python3
"""Validate patient/WSI partitions and select few-shot samples only from train."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cervithink.data import read_jsonl, write_jsonl
from cervithink.experiment_data import prepare_experiment_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--group-key", required=True)
    parser.add_argument("--image-root")
    parser.add_argument("--label-mapping")
    parser.add_argument("--create-split", action="store_true")
    parser.add_argument("--test-ratio", type=float, default=.2)
    parser.add_argument("--train-fraction", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    mapping = json.loads(Path(args.label_mapping).read_text(encoding="utf-8")) if args.label_mapping else None
    rows, manifest = prepare_experiment_rows(
        read_jsonl(args.input), group_key=args.group_key,
        image_root=args.image_root or str(Path(args.input).resolve().parent),
        create_split=args.create_split, test_ratio=args.test_ratio,
        train_fraction=args.train_fraction, seed=args.seed, label_mapping=mapping,
    )
    write_jsonl(rows, args.output)
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Validated {len(rows)} samples; test membership preserved")


if __name__ == "__main__":
    main()
