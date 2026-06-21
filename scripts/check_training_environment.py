#!/usr/bin/env python3
"""Check whether the local machine can run CerviThink SFT/GRPO jobs."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MI2026 = ROOT.parent
sys.path.insert(0, str(ROOT))

from cervithink.data import read_jsonl
from cervithink.constants import CERVICAL_LABELS


REQUIRED_COLUMNS = {
    "image",
    "width",
    "height",
    "input_width",
    "input_height",
    "bboxs",
    "problem",
    "solution",
    "label",
    "split",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", default=os.getenv("DATASET_JSONL"))
    parser.add_argument("--model", default=os.getenv("MODEL_NAME_OR_PATH"))
    parser.add_argument("--ground-r1-root", default=str(MI2026 / "Ground-R1"))
    parser.add_argument("--qwen-finetune-root", default=os.getenv("QWEN_FINETUNE_ROOT"))
    parser.add_argument("--ground-r1-open-r1", default=os.getenv("GROUND_R1_OPEN_R1"))
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if optional checks fail.")
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    return parser.parse_args()


def check_import(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    version = getattr(module, "__version__", "")
    return {"ok": True, "version": version}


def check_torch() -> dict[str, Any]:
    result = check_import("torch")
    if not result["ok"]:
        return result
    import torch

    result.update(
        {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        }
    )
    return result


def check_path(path: str | None, kind: str = "exists") -> dict[str, Any]:
    if not path:
        return {"ok": False, "error": "not set"}
    p = Path(path)
    if kind == "file":
        ok = p.is_file()
    elif kind == "dir":
        ok = p.is_dir()
    else:
        ok = p.exists()
    return {"ok": ok, "path": str(p), "error": "" if ok else "missing"}


def check_dataset(path: str | None, max_rows: int = 20) -> dict[str, Any]:
    base = check_path(path, kind="file")
    if not base["ok"]:
        return base
    rows = read_jsonl(path)
    if not rows:
        return {"ok": False, "path": path, "error": "empty JSONL"}

    missing = sorted(REQUIRED_COLUMNS - set(rows[0]))
    labels = {str(row.get("label", "")) for row in rows[:max_rows]}
    bad_labels = sorted(label for label in labels if label not in CERVICAL_LABELS)
    image_missing = []
    for row in rows[:max_rows]:
        image = Path(str(row.get("image", "")))
        if not image.exists():
            image_missing.append(str(image))

    ok = not missing and not bad_labels and not image_missing
    return {
        "ok": ok,
        "path": path,
        "rows": len(rows),
        "missing_columns": missing,
        "bad_labels": bad_labels,
        "missing_images_checked": image_missing[:5],
    }


def main() -> int:
    args = parse_args()
    ground_root = Path(args.ground_r1_root)
    qwen_finetune = args.qwen_finetune_root or str(ground_root / "qwen-vl-finetune")
    open_r1 = args.ground_r1_open_r1 or str(ground_root / "r1-v" / "src" / "open_r1")

    report = {
        "python": sys.version.split()[0],
        "packages": {
            "torch": check_torch(),
            "transformers": check_import("transformers"),
            "trl": check_import("trl"),
            "datasets": check_import("datasets"),
            "accelerate": check_import("accelerate"),
            "deepspeed": check_import("deepspeed"),
            "cv2": check_import("cv2"),
            "PIL": check_import("PIL"),
        },
        "paths": {
            "model": check_path(args.model, kind="exists"),
            "ground_r1_root": check_path(str(ground_root), kind="dir"),
            "ground_r1_open_r1": check_path(open_r1, kind="dir"),
            "qwen_finetune_root": check_path(qwen_finetune, kind="dir"),
        },
        "dataset": check_dataset(args.dataset_jsonl),
    }

    sys.path.insert(0, open_r1)
    try:
        from trainer import Qwen2VLGRPOTrainer  # noqa: F401

        report["ground_r1_import"] = {"ok": True}
    except Exception as exc:
        report["ground_r1_import"] = {"ok": False, "error": str(exc)}

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")

    required_ok = [
        report["packages"]["torch"]["ok"],
        report["packages"]["transformers"]["ok"],
        report["packages"]["trl"]["ok"],
        report["packages"]["datasets"]["ok"],
        report["paths"]["ground_r1_open_r1"]["ok"],
        report["ground_r1_import"]["ok"],
    ]
    optional_ok = [
        report["paths"]["model"]["ok"],
        report["paths"]["qwen_finetune_root"]["ok"],
        report["dataset"]["ok"],
    ]
    if not all(required_ok):
        return 1
    if args.strict and not all(optional_ok):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
