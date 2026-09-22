#!/usr/bin/env python3
"""Crop annotated cervical cells and resize them to the paper input size."""

from __future__ import annotations

import argparse
import random
import re
import sys
from dataclasses import replace
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cervithink.data import (
    CerviSample,
    convert_records,
    group_representative_label,
    make_train_test_split,
    normalize_record,
    read_jsonl,
    split_group_key,
    write_jsonl,
)
from cervithink.visual_ops import clamp_bbox, focus_operation, scale_bbox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Raw annotation JSONL.")
    parser.add_argument("--output", required=True, help="Output cropped annotation JSONL.")
    parser.add_argument("--output-image-dir", required=True, help="Directory for resized cell images.")
    parser.add_argument("--image-root", default=None, help="Prefix for relative source image paths.")
    parser.add_argument("--size", type=int, default=256, help="Output image side length.")
    parser.add_argument("--context-scale", type=float, default=1.0, help="Scale bbox before cropping.")
    parser.add_argument("--resize-full-image", action="store_true", help="Resize the full image instead of cropping bbox.")
    parser.add_argument("--split-if-missing", action="store_true", help="Create a stratified train/test split.")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default=None, help="Dataset name written to each row.")
    parser.add_argument("--to-cervicot", action="store_true", help="Write final CerviCoT/Ground-R1 records.")
    parser.add_argument("--keep-source-id", action="store_true", help="Keep source sample IDs in the output JSONL.")
    parser.add_argument("--rationale-mode", choices=("none", "provided", "template"), default="provided")
    return parser.parse_args()


def safe_stem(value: str, fallback: str) -> str:
    text = value.strip() or fallback
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text[:120] or fallback


def resized_bbox_for_full_image(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    size: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return clamp_bbox(
        (
            x1 * size / width,
            y1 * size / height,
            x2 * size / width,
            y2 * size / height,
        ),
        size,
        size,
        min_size=1,
    )


def split_pairs(
    pairs: list[tuple[CerviSample, bool]],
    test_ratio: float,
    seed: int,
) -> list[tuple[CerviSample, bool]]:
    train, test = make_train_test_split([p[0] for p in pairs], test_ratio, seed)
    keep_by_identity = {id(sample): keep for sample, keep in pairs}
    return [(replace(s, split=split), keep_by_identity[id(s)])
            for split, subset in (("train", train), ("test", test)) for s in subset]


def make_resized_sample(
    sample: CerviSample,
    out_dir: Path,
    index: int,
    size: int,
    context_scale: float,
    resize_full_image: bool,
    dataset: str | None,
    keep_rationale: bool,
    keep_source_id: bool,
) -> dict[str, object]:
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(sample.image).convert("RGB")

    if resize_full_image:
        patch = image.resize((size, size), Image.Resampling.BICUBIC)
        new_bbox = resized_bbox_for_full_image(sample.bbox, image.width, image.height, size)
    else:
        bbox = sample.bbox
        if context_scale != 1.0:
            bbox = scale_bbox(bbox, image.width, image.height, context_scale)
        patch = focus_operation(image, bbox).resize((size, size), Image.Resampling.BICUBIC)
        # Keep the original target inside a padded crop, rather than labeling
        # the entire padded view as the target.
        crop_box = clamp_bbox(bbox, image.width, image.height, min_size=28)
        x1, y1, x2, y2 = sample.bbox
        cx1, cy1, cx2, cy2 = crop_box
        new_bbox = clamp_bbox((
            (x1-cx1)*size/(cx2-cx1), (y1-cy1)*size/(cy2-cy1),
            (x2-cx1)*size/(cx2-cx1), (y2-cy1)*size/(cy2-cy1),
        ), size, size)

    stem = safe_stem(f"sample_{index:06d}", f"sample_{index:06d}")
    out_path = out_dir / f"{index:06d}_{stem}.png"
    patch.save(out_path)

    prepared = replace(
        sample,
        image=str(out_path),
        width=size,
        height=size,
        bbox=new_bbox,
        dataset=dataset or sample.dataset,
    )
    return {
        "image": prepared.image,
        "width": prepared.width,
        "height": prepared.height,
        "label": prepared.label,
        "bbox": list(prepared.bbox),
        "question": prepared.question,
        "rationale": prepared.rationale if keep_rationale else "",
        "id": prepared.sample_id if keep_source_id and prepared.sample_id else str(index),
        "split": prepared.split,
        "dataset": prepared.dataset,
    }


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    pairs = [
        (
            normalize_record(row, image_root=args.image_root),
            any(row.get(key) for key in ("rationale", "cot", "thought")),
        )
        for row in rows
    ]
    if args.split_if_missing and all(sample.split == "train" for sample, _ in pairs):
        pairs = split_pairs(pairs, test_ratio=args.test_ratio, seed=args.seed)

    prepared_rows = [
        make_resized_sample(
            sample=sample,
            out_dir=Path(args.output_image_dir),
            index=index,
            size=args.size,
            context_scale=args.context_scale,
            resize_full_image=args.resize_full_image,
            dataset=args.dataset,
            keep_rationale=keep_rationale,
            keep_source_id=args.keep_source_id,
        )
        for index, (sample, keep_rationale) in enumerate(pairs, start=1)
    ]
    if args.to_cervicot:
        output_rows = convert_records(
            prepared_rows,
            split_if_missing=args.split_if_missing,
            test_ratio=args.test_ratio,
            seed=args.seed,
            rationale_mode=args.rationale_mode,
        )
    else:
        output_rows = prepared_rows
    write_jsonl(output_rows, args.output)
    print(f"Wrote {len(output_rows)} records to {args.output}")


if __name__ == "__main__":
    main()
