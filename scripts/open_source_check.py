#!/usr/bin/env python3
"""Check repository hygiene before public release."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_DIR_NAMES = {"examples", "__pycache__", ".pytest_cache", "build", "dist"}
FORBIDDEN_DIR_SUFFIXES = {".egg-info"}
FORBIDDEN_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
}
PLACEHOLDER_PATTERNS = [
    re.compile(r"CerviThink Authors"),
    re.compile(r"email@anonymized\.com"),
    re.compile(r"Anonymized"),
]


def iter_files():
    for path in ROOT.rglob("*"):
        if path.is_file():
            yield path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="Allow anonymous paper placeholders before camera-ready release.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    issues = []

    for path in ROOT.rglob("*"):
        if path.is_dir() and path.name in FORBIDDEN_DIR_NAMES:
            issues.append(f"forbidden directory: {path.relative_to(ROOT)}")
        if path.is_dir() and any(path.name.endswith(suffix) for suffix in FORBIDDEN_DIR_SUFFIXES):
            issues.append(f"forbidden metadata directory: {path.relative_to(ROOT)}")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            issues.append(f"forbidden binary asset: {path.relative_to(ROOT)}")

    if not args.allow_placeholders:
        for path in iter_files():
            if path.name == "open_source_check.py":
                continue
            if path.suffix.lower() not in {".md", ".cff", ".cfg", ".txt", ".py", ".toml"} and path.name not in {
                "LICENSE",
                "NOTICE",
            }:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for pattern in PLACEHOLDER_PATTERNS:
                if pattern.search(text):
                    issues.append(f"placeholder metadata: {path.relative_to(ROOT)}")
                    break

    if issues:
        print("open_source_check: issues found")
        for issue in issues:
            print(issue)
        return 1
    print("open_source_check: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
