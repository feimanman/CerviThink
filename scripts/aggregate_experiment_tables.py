#!/usr/bin/env python3
"""Aggregate metric JSON files into paper-style tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Experiment manifest JSON.")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_weighted(path: str) -> dict[str, float]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    weighted = data["weighted"]
    return {
        "precision": float(weighted["precision"]) * 100.0,
        "recall": float(weighted["recall"]) * 100.0,
        "f1": float(weighted["f1"]) * 100.0,
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["table", "dataset", "method", "precision", "recall", "f1"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(table: str, rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        f"# {table}",
        "",
        "| Dataset | Method | Precision | Recall | F1-score |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {method} | {precision:.2f} | {recall:.2f} | {f1:.2f} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))

    tables: dict[str, list[dict[str, object]]] = defaultdict(list)
    missing = []
    for item in manifest:
        metrics_path = Path(item["metrics"])
        if not metrics_path.exists():
            missing.append(str(metrics_path))
            continue
        weighted = load_weighted(str(metrics_path))
        row = {
            "table": item.get("table", "results"),
            "dataset": item.get("dataset", ""),
            "method": item.get("method", ""),
            **weighted,
        }
        tables[str(row["table"])].append(row)

    all_rows = [row for rows in tables.values() for row in rows]
    write_csv(all_rows, out_dir / "all_results.csv")
    for table, rows in tables.items():
        write_markdown(table, rows, out_dir / f"{table}.md")

    if missing:
        print("Missing metric files:")
        for path in missing:
            print(path)
    print(f"Wrote {len(all_rows)} rows to {out_dir}")


if __name__ == "__main__":
    main()
