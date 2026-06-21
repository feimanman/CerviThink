"""Text parsing."""

from __future__ import annotations

import re
from typing import Iterable, Optional

from .constants import CERVICAL_LABELS, LABEL_ALIASES

BOX_PATTERN = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)
NEGATIVE_NORMALITY_PATTERN = re.compile(r"(?<![a-z0-9])(abnormal|non-normal|not normal)(?![a-z0-9])")


def contains_label_term(text: str, term: str) -> bool:
    pattern = re.escape(term.lower()).replace(r"\ ", r"\s+")
    return re.search(fr"(?<![a-z0-9]){pattern}(?![a-z0-9])", text) is not None


def extract_box(text: str) -> Optional[tuple[float, float, float, float]]:
    """Extract the first [x1,y1,x2,y2] box from text."""

    match = BOX_PATTERN.search(text or "")
    if not match:
        return None
    return tuple(float(value) for value in match.groups())


def extract_all_boxes(text: str) -> list[tuple[float, float, float, float]]:
    """Extract all boxes from text."""

    return [tuple(float(value) for value in match) for match in BOX_PATTERN.findall(text or "")]


def extract_tag(text: str, tag: str) -> Optional[str]:
    """Extract text between XML-like tags, returning None when absent."""

    pattern = re.compile(fr"<{tag}>\s*(.*?)\s*</{tag}>", re.IGNORECASE | re.DOTALL)
    match = pattern.search(text or "")
    if not match:
        return None
    return match.group(1).strip()


def extract_answer(text: str) -> str:
    """Extract the answer text, falling back to the whole response."""

    return extract_tag(text, "answer") or (text or "").strip()


def normalize_label(label: str, labels: Iterable[str] = CERVICAL_LABELS) -> str:
    """Normalize model or dataset labels to one of the target cervical labels."""

    if label is None:
        return ""
    raw = str(label).strip()
    if raw in labels:
        return raw

    compact = re.sub(r"\s+", " ", raw.lower()).strip(" .,:;()[]{}")
    compact = compact.replace("_", "-")
    if NEGATIVE_NORMALITY_PATTERN.search(compact):
        return raw
    if compact in LABEL_ALIASES:
        return LABEL_ALIASES[compact]

    for candidate in labels:
        if contains_label_term(compact, candidate):
            return candidate

    for alias, candidate in LABEL_ALIASES.items():
        if contains_label_term(compact, alias):
            return candidate
    return raw


def answer_matches(prediction: str, target: str) -> bool:
    """Compare predicted and target labels after normalization."""

    return normalize_label(extract_answer(prediction)) == normalize_label(target)


def has_single_tag_pair(text: str, tag: str) -> bool:
    """Return True when exactly one opening and closing tag pair is present."""

    lowered = (text or "").lower()
    return lowered.count(f"<{tag}>") == 1 and lowered.count(f"</{tag}>") == 1


def has_cervithink_answer_format(text: str) -> bool:
    """Check the answering format."""

    return has_single_tag_pair(text, "think") and has_single_tag_pair(text, "answer")


def has_cervithink_grounding_format(text: str) -> bool:
    """Check the grounding format with one think tag and one box tag."""

    return (
        has_single_tag_pair(text, "think")
        and has_single_tag_pair(text, "box")
        and extract_box(extract_tag(text, "box") or text) is not None
    )


def has_full_cervithink_format(text: str) -> bool:
    """Check a combined response containing think, box, rethink, and answer tags."""

    required = ("think", "box", "rethink", "answer")
    if not all(has_single_tag_pair(text, tag) for tag in required):
        return False
    return extract_box(extract_tag(text, "box") or text) is not None
