"""Shared action/prompt protocol; metric parsing remains untouched."""
from __future__ import annotations

import math
import re

from .parsing import BOX_PATTERN
from .prompts import grounding_prompt, answer_prompt

GROUND = re.compile(r"\s*<think>(?P<thought>.*?)</think>\s*<box>(?P<box>.*?)</box>\s*", re.S)
ANSWER = re.compile(r"\s*<rethink>(?P<thought>.*?)</rethink>\s*<answer>(?P<label>.*?)</answer>\s*", re.S)


def user_message(text: str, image_count: int) -> list[dict]:
    return [{"role": "user", "content": [
        *[{"type": "image"} for _ in range(image_count)], {"type": "text", "text": text},
    ]}]


def grounding_messages(question: str, width: int, height: int) -> list[dict]:
    return user_message(grounding_prompt(question, width, height), 1)


def answer_messages(question: str, width: int, height: int, grounding: str) -> list[dict]:
    """Five image entries total: original turn, then original/focus/transform/ignore."""
    return [
        *grounding_messages(question, width, height),
        {"role": "assistant", "content": [{"type": "text", "text": grounding}]},
        *user_message(answer_prompt(question), 4),
    ]


def grounding_format(text: str) -> bool:
    match = GROUND.fullmatch(text or "")
    return bool(match and match["thought"].strip() and "<" not in match["thought"]
                and BOX_PATTERN.fullmatch(match["box"].strip()))


def answer_format(text: str) -> bool:
    match = ANSWER.fullmatch(text or "")
    return bool(match and match["thought"].strip() and match["label"].strip()
                and "<" not in match["thought"] and "<" not in match["label"])


def trajectory_format(grounding: str, answer: str) -> bool:
    return grounding_format(grounding) and answer_format(answer)


def policy_box(text: str, width: int, height: int):
    """Original-image coordinates only. Reject invalid actions; never rescale by processor grid."""
    if not grounding_format(text):
        return None
    match = GROUND.fullmatch(text or "")
    if not match:
        return None
    box = BOX_PATTERN.fullmatch(match["box"].strip())
    if not box:
        return None
    values = tuple(map(float, box.groups()))
    if not all(math.isfinite(v) for v in values):
        return None
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        return None
    return values
