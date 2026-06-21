#!/usr/bin/env python3
"""Run basic checks before publishing the repository."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "outputs",
    "wandb",
    "runs",
    "logs",
    "checkpoints",
    "build",
    "dist",
}

TEXT_SUFFIXES = {
    ".cff",
    ".cfg",
    ".env",
    ".example",
    ".in",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

def block_patterns() -> list[re.Pattern[str]]:
    patterns = [
        re.compile(r"\b" + "OPEN" + "AI_API_KEY" + r"\b"),
        re.compile(r"\b" + "OPEN" + "ROUTER_API_KEY" + r"\b"),
        re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
        re.compile(r"hf_[A-Za-z0-9]{20,}"),
        re.compile(r"AKIA[0-9A-Z]{16}"),
    ]
    for value in {str(Path.home()), os.getenv("PWD", "")}:
        if value and value != "/" and len(value) > 5:
            patterns.append(re.compile(re.escape(value)))
    return patterns


BLOCK_PATTERNS = block_patterns()


def iter_text_files(root: Path):
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            yield path


def scan_tree() -> list[str]:
    hits = []
    for path in iter_text_files(ROOT):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pattern in BLOCK_PATTERNS:
                if pattern.search(line):
                    rel = path.relative_to(ROOT)
                    hits.append(f"{rel}:{lineno}: {line.strip()}")
    return hits


def run(cmd: list[str]) -> int:
    print("+", " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


def clean_pycache() -> None:
    for path in ROOT.rglob("__pycache__"):
        if path.is_dir():
            for child in path.iterdir():
                child.unlink()
            path.rmdir()
    for path in ROOT.rglob("*.pyc"):
        path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    hits = scan_tree()
    if hits:
        print("Sensitive or local-only strings found:")
        for hit in hits:
            print(hit)
        return 1

    if not args.skip_tests:
        for script in sorted((ROOT / "scripts").glob("*.sh")):
            rc = run(["bash", "-n", str(script)])
            if rc != 0:
                return rc
        rc = run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
        if rc != 0:
            return rc
        rc = run([sys.executable, "-m", "compileall", "cervithink", "scripts", "tests"])
        if rc != 0:
            return rc
        clean_pycache()
        rc = run([sys.executable, "scripts/open_source_check.py"])
        if rc != 0:
            return rc

    print("release_check: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
