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
You are given four images in order: original, focused crop, transformed crop, and ignored/inpainted original.
Classify the cell as one of: {labels}.
Use the focused crop for morphology and the transformed crop for texture. Treat the inpainted background as an auxiliary view, not independent proof of normality.
Use this exact format:
<rethink>step-by-step diagnostic reasoning</rethink> <answer>one label from the list</answer>"""


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


def sft_prompt(question: str, width: int, height: int, rationale_mode: str = "provided") -> str:
    if rationale_mode == "none":
        return (
            f"Question: {question}\nClassify the cervical cell as one of: {label_options()}.\n"
            f"The image size is Width:{width}, Height:{height}.\n"
            "Return only the region and label, without a diagnostic rationale:\n"
            "<box>[x1,y1,x2,y2]</box> <answer>one label from the list</answer>"
        )
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
