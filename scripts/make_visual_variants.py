#!/usr/bin/env python3
"""Save focus/transform/ignore images for one sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cervithink.visual_ops import make_visual_variants, save_variants


def parse_bbox(text: str) -> tuple[float, float, float, float]:
    values = json.loads(text)
    if len(values) != 4:
        raise ValueError("--bbox must contain four numbers")
    return tuple(float(value) for value in values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--bbox", required=True, help='JSON list, e.g. "[10,20,100,120]"')
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="sample")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = make_visual_variants(args.image, parse_bbox(args.bbox))
    paths = save_variants(variants, args.output_dir, prefix=args.prefix)
    print(json.dumps({"bbox": variants.bbox, "paths": paths}, indent=2))


if __name__ == "__main__":
    main()
