#!/usr/bin/env python3
"""Evaluate a Hugging Face vision-language baseline on CerviThink JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cervithink.constants import CERVICAL_LABELS
from cervithink.data import read_jsonl
from cervithink.metrics import classification_report
from cervithink.parsing import extract_answer, normalize_label


PROMPT_TEMPLATE = """Classify this cervical cytology image as one of: {labels}.
Return only the final label in this format: <answer>label</answer>."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model path or id.")
    parser.add_argument("--input", required=True, help="CerviCoT/Ground-R1 JSONL.")
    parser.add_argument("--output", required=True, help="Prediction JSONL.")
    parser.add_argument("--report-output", default=None)
    parser.add_argument("--family", choices=("qwen2_5_vl", "llava", "auto"), default="auto")
    parser.add_argument("--split", default="test")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    return parser.parse_args()


def resolve_image(path: str, image_root: str | None) -> str:
    image_path = Path(path)
    if image_root and not image_path.is_absolute():
        image_path = Path(image_root) / image_path
    return str(image_path)


def load_model(args):
    try:
        import torch
        from transformers import AutoProcessor
    except Exception as exc:
        raise SystemExit(f"Missing baseline dependencies: {exc}")

    dtype = "auto" if args.torch_dtype == "auto" else getattr(torch, args.torch_dtype)
    model_kwargs = {"device_map": args.device_map, "torch_dtype": dtype}
    family = args.family
    if family == "auto":
        lowered = args.model.lower()
        family = "qwen2_5_vl" if "qwen" in lowered else "llava"

    processor = AutoProcessor.from_pretrained(args.model)
    if family == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, **model_kwargs)
    else:
        try:
            from transformers import LlavaForConditionalGeneration

            model = LlavaForConditionalGeneration.from_pretrained(args.model, **model_kwargs)
        except Exception:
            from transformers import AutoModelForVision2Seq

            model = AutoModelForVision2Seq.from_pretrained(args.model, **model_kwargs)
    return family, processor, model


def qwen_messages(prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def generate_prediction(family: str, processor, model, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
    if family == "qwen2_5_vl":
        text = processor.apply_chat_template(qwen_messages(prompt), tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)
    else:
        text = f"USER: <image>\n{prompt}\nASSISTANT:"
        inputs = processor(text=text, images=image, padding=True, return_tensors="pt").to(model.device)

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_length = inputs.input_ids.shape[1]
    trimmed = output_ids[:, input_length:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def main() -> None:
    args = parse_args()
    rows = [row for row in read_jsonl(args.input) if row.get("split", args.split) == args.split]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    family, processor, model = load_model(args)
    prompt = PROMPT_TEMPLATE.format(labels=", ".join(CERVICAL_LABELS))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels: list[str] = []
    predictions: list[str] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            image = Image.open(resolve_image(str(row["image"]), args.image_root)).convert("RGB")
            raw = generate_prediction(family, processor, model, image, prompt, args.max_new_tokens)
            prediction = normalize_label(extract_answer(raw))
            payload = {
                "problem_id": row.get("problem_id"),
                "image": row.get("image"),
                "label": row.get("label"),
                "prediction": prediction,
                "raw_prediction": raw,
                "model": args.model,
                "family": family,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            labels.append(str(row.get("label", "")))
            predictions.append(prediction)
            print(f"[{index}/{len(rows)}] {row.get('problem_id', index)} -> {prediction}")

    report = classification_report(labels, predictions)
    report_text = json.dumps(report, ensure_ascii=False, indent=2)
    print(report_text)
    if args.report_output:
        report_path = Path(args.report_output)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
