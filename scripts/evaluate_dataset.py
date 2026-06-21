#!/usr/bin/env python3
"""Run CerviThink inference on a JSONL split and write predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cervithink.data import read_jsonl
from cervithink.hf_qwen import HFQwenVLModel
from cervithink.metrics import classification_report
from cervithink.pipeline import CerviThinkConfig, CerviThinkPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Qwen2.5-VL checkpoint.")
    parser.add_argument("--input", required=True, help="CerviCoT/Ground-R1 JSONL.")
    parser.add_argument("--output", required=True, help="Prediction JSONL.")
    parser.add_argument("--report-output", default=None, help="Optional metrics JSON.")
    parser.add_argument("--image-root", default=None, help="Prefix for relative image paths.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--grounding-rollouts", type=int, default=4)
    parser.add_argument("--answer-rollouts", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def resolve_image(path: str, image_root: str | None) -> str:
    image_path = Path(path)
    if image_root and not image_path.is_absolute():
        image_path = Path(image_root) / image_path
    return str(image_path)


def serialize_result(record: dict, result) -> dict[str, object]:
    return {
        "problem_id": record.get("problem_id"),
        "image": record.get("image"),
        "label": record.get("label"),
        "prediction": result.prediction,
        "question": result.question,
        "candidates": [
            {
                "grounding": candidate.grounding,
                "bbox": candidate.bbox,
                "reward": candidate.reward.as_dict() if candidate.reward else None,
                "selection_score": candidate.selection_score,
                "answers": [
                    {
                        "final_answer": answer.final_answer,
                        "crop_answer": answer.crop_answer,
                        "background_answer": answer.background_answer,
                        "reward": answer.reward.as_dict() if answer.reward else None,
                        "selection_score": answer.selection_score,
                    }
                    for answer in candidate.answers
                ],
            }
            for candidate in result.candidates
        ],
    }


def main() -> None:
    args = parse_args()
    rows = [row for row in read_jsonl(args.input) if row.get("split", args.split) == args.split]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    model = HFQwenVLModel(args.model)
    pipeline = CerviThinkPipeline(
        model,
        config=CerviThinkConfig(
            grounding_rollouts=args.grounding_rollouts,
            answer_rollouts=args.answer_rollouts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        ),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions: list[str] = []
    labels: list[str] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(rows, start=1):
            image = resolve_image(str(record["image"]), args.image_root)
            label = str(record.get("label", ""))
            result = pipeline.run(image, question=record.get("problem"), true_label=None)
            payload = serialize_result(record, result)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            labels.append(label)
            predictions.append(result.prediction)
            print(f"[{index}/{len(rows)}] {record.get('problem_id', index)} -> {result.prediction}")

    report = classification_report(labels, predictions)
    report_text = json.dumps(report, ensure_ascii=False, indent=2)
    print(report_text)
    if args.report_output:
        report_path = Path(args.report_output)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
