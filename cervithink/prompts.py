"""Prompt templates used by CerviThink."""

from __future__ import annotations

from .constants import CERVICAL_LABELS


def label_options() -> str:
    return ", ".join(CERVICAL_LABELS)


GROUNDING_PROMPT_TEMPLATE = """Question: {question}
You are analyzing a cervical cytology image. First identify the most diagnostically useful cell region for deciding among: {labels}.
Reason about nuclear size, nuclear-to-cytoplasmic ratio, chromatin texture, membrane irregularity, overlapping cells, and artifacts.
Return exactly one region selected from the original image coordinates.
The image size is Width:{width}, Height:{height}. The box must stay inside the image.
Use this exact format:
<think>brief visual reasoning</think> <box>[x1,y1,x2,y2]</box>"""


ANSWER_PROMPT_TEMPLATE = """Question: {question}
You are given the original cervical cytology image plus one or more manipulated evidence images.
Classify the cell as one of: {labels}.
Use the focused crop for morphology, the transformed crop for fine texture, and the ignored/inpainted image to verify whether the remaining background is normal.
Use this exact format:
<think>step-by-step diagnostic reasoning</think> <answer>one label from the list</answer>"""


SFT_PROMPT_TEMPLATE = """Question: {question}
You are analyzing a cervical cytology image for one of these labels: {labels}.
Identify the diagnostically useful cell region, reason through the morphology, reconsider the focused evidence, and then give the final class.
The image size is Width:{width}, Height:{height}. The box must stay inside the image.
Use this exact format:
<think>step-by-step diagnostic reasoning</think> <box>[x1,y1,x2,y2]</box> <rethink>focused evidence check</rethink> <answer>one label from the list</answer>"""


BACKGROUND_PROMPT_TEMPLATE = """The abnormal candidate cell has been masked or ignored.
Classify the remaining visible background as Normal or Abnormal.
Use this exact format:
<think>brief background verification</think> <answer>Normal or Abnormal</answer>"""


CROP_CONSISTENCY_PROMPT_TEMPLATE = """Question: {question}
You are given a focused cervical cytology crop selected as diagnostic evidence.
Classify the crop as one of: {labels}.
Use this exact format:
<think>brief diagnostic verification</think> <answer>one label from the list</answer>"""


SFT_RESPONSE_TEMPLATE = (
    "<think>{rationale}</think> "
    "<box>[{x1},{y1},{x2},{y2}]</box> "
    "<rethink>{rethink}</rethink> "
    "<answer>{label}</answer>"
)


def grounding_prompt(question: str, width: int, height: int) -> str:
    return GROUNDING_PROMPT_TEMPLATE.format(
        question=question,
        width=width,
        height=height,
        labels=label_options(),
    )


def answer_prompt(question: str) -> str:
    return ANSWER_PROMPT_TEMPLATE.format(question=question, labels=label_options())


def crop_consistency_prompt(question: str) -> str:
    return CROP_CONSISTENCY_PROMPT_TEMPLATE.format(
        question=question,
        labels=label_options(),
    )


def sft_prompt(question: str, width: int, height: int) -> str:
    return SFT_PROMPT_TEMPLATE.format(
        question=question,
        width=width,
        height=height,
        labels=label_options(),
    )


def background_prompt() -> str:
    return BACKGROUND_PROMPT_TEMPLATE


def default_question() -> str:
    return "Classify this cervical cell image."
