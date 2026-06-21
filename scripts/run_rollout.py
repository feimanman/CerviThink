#!/usr/bin/env python3
"""Run one CerviThink rollout with a Hugging Face Qwen2.5-VL model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cervithink.hf_qwen import HFQwenVLModel
from cervithink.pipeline import CerviThinkConfig, CerviThinkPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Qwen2.5-VL model path or HF id.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", default=None)
    parser.add_argument(
        "--label",
        default=None,
        help="Optional ground-truth label for debugging DVHR scoring; do not use for blind inference.",
    )
    parser.add_argument("--grounding-rollouts", type=int, default=4)
    parser.add_argument("--answer-rollouts", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = HFQwenVLModel(args.model)
    config = CerviThinkConfig(
        grounding_rollouts=args.grounding_rollouts,
        answer_rollouts=args.answer_rollouts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    result = CerviThinkPipeline(model, config=config).run(
        args.image,
        question=args.question,
        true_label=args.label,
    )
    payload = {
        "image": result.image,
        "question": result.question,
        "prediction": result.prediction,
        "candidates": [
            {
                "grounding": candidate.grounding,
                "bbox": candidate.bbox,
                "final_answer": candidate.final_answer,
                "crop_answer": candidate.crop_answer,
                "background_answer": candidate.background_answer,
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
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
