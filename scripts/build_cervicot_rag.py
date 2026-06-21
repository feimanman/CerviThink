#!/usr/bin/env python3
"""Build CerviCoT records with retrieval-augmented diagnostic rationales."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cervithink.constants import LABEL_PRIORS
from cervithink.data import convert_records, read_jsonl, write_jsonl
from cervithink.parsing import normalize_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, help="Raw or cropped annotation JSONL.")
    parser.add_argument("--knowledge", required=True, help="JSONL with PubMed title/text/abstract fields.")
    parser.add_argument("--output", required=True, help="Output CerviCoT/Ground-R1 JSONL.")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--split-if-missing", action="store_true")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z-]{2,}")


def tokens(text: str) -> Counter[str]:
    return Counter(token.lower() for token in TOKEN_PATTERN.findall(text or ""))


def knowledge_text(row: dict[str, Any]) -> str:
    parts = [
        str(row.get("title", "")),
        str(row.get("abstract", "")),
        str(row.get("text", "")),
        str(row.get("keywords", "")),
    ]
    return " ".join(part for part in parts if part)


def rank_knowledge(record: dict[str, Any], knowledge: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    label = normalize_label(str(record.get("label", record.get("class", ""))))
    prior = LABEL_PRIORS.get(label, {})
    query = " ".join(
        [
            str(record.get("question", "")),
            label,
            str(prior.get("nucleus", "")),
            str(prior.get("chromatin", "")),
            str(prior.get("decision", "")),
        ]
    )
    query_tokens = tokens(query)
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in knowledge:
        body = knowledge_text(row)
        body_tokens = tokens(body)
        overlap = sum(min(count, body_tokens.get(term, 0)) for term, count in query_tokens.items())
        label_boost = 2.0 if normalize_label(str(row.get("label", ""))) == label else 0.0
        score = overlap + label_boost
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in scored[:top_k]]


def compose_rationale(record: dict[str, Any], retrieved: list[dict[str, Any]]) -> str:
    label = normalize_label(str(record.get("label", record.get("class", ""))))
    prior = LABEL_PRIORS.get(label, LABEL_PRIORS["Normal"])
    bbox = record.get("bbox") or record.get("box") or record.get("cell_bbox") or [0, 0, 256, 256]
    x1, y1, x2, y2 = [int(float(value)) for value in bbox[:4]]

    evidence_bits = []
    for row in retrieved:
        title = str(row.get("title", "")).strip()
        body = knowledge_text(row)
        sentence = re.split(r"(?<=[.!?])\s+", body.strip())[0] if body.strip() else ""
        if title:
            evidence_bits.append(title)
        elif sentence:
            evidence_bits.append(sentence[:160])

    evidence = "; ".join(evidence_bits[:3])
    if evidence:
        evidence = f" Retrieved medical evidence emphasizes {evidence}."

    return (
        f"The candidate cell region is localized around [{x1},{y1},{x2},{y2}]. "
        f"The reasoning first checks the cell boundary and cytoplasm, then evaluates "
        f"{prior['nucleus']} and {prior['chromatin']}.{evidence} "
        f"These observations support {prior['decision']}."
    )


def main() -> None:
    args = parse_args()
    annotations = read_jsonl(args.annotations)
    knowledge = read_jsonl(args.knowledge)
    enriched = []
    for row in annotations:
        new_row = dict(row)
        if not new_row.get("rationale") and not new_row.get("cot") and not new_row.get("thought"):
            retrieved = rank_knowledge(new_row, knowledge, args.top_k)
            new_row["rationale"] = compose_rationale(new_row, retrieved)
            new_row["retrieved_knowledge"] = [
                str(item.get("title") or item.get("pmid") or item.get("id") or "")
                for item in retrieved
            ]
        enriched.append(new_row)

    converted = convert_records(
        enriched,
        image_root=args.image_root,
        split_if_missing=args.split_if_missing,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    retrieval_by_sample = {}
    for row in enriched:
        key = (
            str(row.get("image", "")),
            normalize_label(str(row.get("label", row.get("class", "")))),
        )
        retrieval_by_sample[key] = row.get("retrieved_knowledge")
    for output_row in converted:
        key = (str(output_row.get("image", "")), normalize_label(str(output_row.get("label", ""))))
        if retrieval_by_sample.get(key):
            output_row["retrieved_knowledge"] = retrieval_by_sample[key]
    write_jsonl(converted, args.output)
    print(f"Wrote {len(converted)} records to {args.output}")


if __name__ == "__main__":
    main()
