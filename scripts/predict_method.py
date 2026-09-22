#!/usr/bin/env python3
"""Seeded inference for method verification. Writes predictions; computes NO metrics."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cervithink.data import read_jsonl
from cervithink.hf_qwen import HFQwenVLModel
from cervithink.pipeline import CerviThinkConfig, CerviThinkPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    from transformers import set_seed
    set_seed(args.seed)
    rows = read_jsonl(args.input)
    if any("split" not in row for row in rows):
        raise ValueError("Input must have explicit frozen splits")
    rows = [r for r in rows if r["split"] == args.split]
    if not rows:
        raise ValueError("Selected inference partition is empty")
    output = Path(args.output)
    if output.exists():
        raise ValueError("Choose a new output path to preserve previous predictions")
    output.parent.mkdir(parents=True, exist_ok=True)
    config = CerviThinkConfig()
    pipeline = CerviThinkPipeline(HFQwenVLModel(args.model), config)
    with output.open("x", encoding="utf-8") as handle:
        for row in rows:
            result = pipeline.run(row["image"], question=row.get("problem"), true_label=None)
            handle.write(json.dumps({
                "problem_id": row.get("problem_id"), "image": row["image"],
                "label": row["label"], "prediction": result.prediction,
                "candidates": [asdict(c) for c in result.candidates],
            }, ensure_ascii=False) + "\n")
            handle.flush()
    output.with_suffix(output.suffix+".manifest.json").write_text(json.dumps({
        "status": "predictions-only-evaluator-deferred", "seed": args.seed,
        "model": args.model, "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        "sample_count": len(rows), "config": asdict(config),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
