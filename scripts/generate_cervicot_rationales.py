#!/usr/bin/env python3
"""Generate CerviCoT rationales with an external local text generator."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cervithink.constants import LABEL_PRIORS
from cervithink.data import read_jsonl, write_jsonl
from cervithink.parsing import normalize_label


PROMPT_TEMPLATE = """You are preparing a cervical cytology reasoning annotation.
Label: {label}
Bounding box: {bbox}
Retrieved literature:
{knowledge}

Write a concise diagnostic rationale that checks cell boundary, cytoplasm,
nuclear size, nuclear-to-cytoplasmic ratio, chromatin texture, and diagnostic
category. Do not include patient information. Return plain text only."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, help="Annotation JSONL.")
    parser.add_argument("--knowledge", required=True, help="PubMed knowledge JSONL.")
    parser.add_argument("--output", required=True, help="Output enriched annotation JSONL.")
    parser.add_argument(
        "--generator-cmd",
        default=os.getenv("CERVICOT_GENERATOR_CMD"),
        help="Command that reads prompt from stdin and writes rationale to stdout.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--fallback-template", action="store_true", help="Use deterministic template if generator is unset.")
    return parser.parse_args()


def knowledge_text(row: dict) -> str:
    parts = [str(row.get("title", "")), str(row.get("abstract", "")), str(row.get("text", ""))]
    return " ".join(part for part in parts if part).strip()


def select_knowledge(label: str, knowledge: list[dict], top_k: int) -> list[dict]:
    exact = [row for row in knowledge if normalize_label(str(row.get("label", ""))) == label]
    if exact:
        return exact[:top_k]
    return knowledge[:top_k]


def template_rationale(record: dict, selected: list[dict]) -> str:
    label = normalize_label(str(record.get("label", "")))
    prior = LABEL_PRIORS.get(label, LABEL_PRIORS["Normal"])
    bbox = record.get("bbox") or record.get("box") or record.get("cell_bbox") or [0, 0, 256, 256]
    evidence = "; ".join((row.get("title") or row.get("pmid") or "") for row in selected if row)
    evidence_text = f" Retrieved evidence considered: {evidence}." if evidence else ""
    return (
        f"The candidate cell region is localized around {bbox}. "
        f"The reasoning checks the boundary and cytoplasm, then evaluates "
        f"{prior['nucleus']} and {prior['chromatin']}.{evidence_text} "
        f"These findings support {prior['decision']}."
    )


def build_prompt(record: dict, selected: list[dict]) -> str:
    label = normalize_label(str(record.get("label", "")))
    bbox = record.get("bbox") or record.get("box") or record.get("cell_bbox") or [0, 0, 256, 256]
    snippets = []
    for row in selected:
        text = knowledge_text(row)
        if text:
            snippets.append(f"- {text[:1200]}")
    return PROMPT_TEMPLATE.format(
        label=label,
        bbox=bbox,
        knowledge="\n".join(snippets) or "- No retrieved snippet.",
    )


def run_generator(command: str, prompt: str) -> str:
    completed = subprocess.run(
        shlex.split(command),
        input=prompt,
        text=True,
        check=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.annotations)
    knowledge = read_jsonl(args.knowledge)
    if not args.generator_cmd and not args.fallback_template:
        raise SystemExit("Set --generator-cmd or use --fallback-template.")

    enriched = []
    for row in rows:
        label = normalize_label(str(row.get("label", "")))
        selected = select_knowledge(label, knowledge, args.top_k)
        new_row = dict(row)
        if args.generator_cmd:
            prompt = build_prompt(new_row, selected)
            new_row["rationale"] = run_generator(args.generator_cmd, prompt)
        else:
            new_row["rationale"] = template_rationale(new_row, selected)
        new_row["retrieved_knowledge"] = [
            str(item.get("title") or item.get("pmid") or item.get("id") or "")
            for item in selected
        ]
        enriched.append(new_row)

    write_jsonl(enriched, args.output)
    print(f"Wrote {len(enriched)} enriched rows to {args.output}")


if __name__ == "__main__":
    main()
