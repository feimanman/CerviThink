#!/usr/bin/env python3
"""Print a compact table from metric JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", help="Metric JSON files from evaluate_dataset.py.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for report in args.reports:
        path = Path(report)
        data = json.loads(path.read_text(encoding="utf-8"))
        weighted = data["weighted"]
        rows.append(
            (
                path.stem.replace("_metrics", ""),
                weighted["precision"] * 100,
                weighted["recall"] * 100,
                weighted["f1"] * 100,
            )
        )

    print("| Dataset | Precision | Recall | F1-score |")
    print("|---|---:|---:|---:|")
    for name, precision, recall, f1 in rows:
        print(f"| {name} | {precision:.2f} | {recall:.2f} | {f1:.2f} |")


if __name__ == "__main__":
    main()
