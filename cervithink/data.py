"""CerviCoT-style dataset preparation."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from PIL import Image

from .constants import CERVICAL_LABELS, LABEL_PRIORS
from .parsing import normalize_label
from .prompts import SFT_RESPONSE_TEMPLATE, default_question
from .visual_ops import clamp_bbox


@dataclass(frozen=True)
class CerviSample:
    image: str
    label: str
    bbox: tuple[int, int, int, int]
    width: int
    height: int
    question: str = default_question()
    rationale: str = ""
    sample_id: str = ""
    split_group: str = ""
    split: str = "train"
    dataset: str = "cervicot"


SPLIT_GROUP_KEYS = (
    "patient_id",
    "patient",
    "case_id",
    "slide_id",
    "subject_id",
    "study_id",
    "group_id",
)
LABEL_PRIORITY = {label: index for index, label in enumerate(CERVICAL_LABELS)}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _first_present(record: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return default


def _read_image_size(image_path: str) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def _extract_bbox(record: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
    bbox = _first_present(record, ("bbox", "box", "cell_bbox"), None)
    if bbox is None:
        boxes = _first_present(record, ("bboxs", "boxes", "annotations"), None)
        if boxes:
            bbox = boxes[0]
    if bbox is None:
        bbox = [0, 0, width, height]
    return clamp_bbox(tuple(float(v) for v in bbox[:4]), width, height, min_size=28)


def build_rule_based_rationale(label: str, bbox: tuple[int, int, int, int]) -> str:
    """Create a deterministic CoT rationale when RAG output is unavailable."""

    normalized = normalize_label(label)
    prior = LABEL_PRIORS.get(normalized, LABEL_PRIORS["Normal"])
    x1, y1, x2, y2 = bbox
    return (
        f"The candidate cell region is localized around [{x1},{y1},{x2},{y2}]. "
        f"The diagnostic check first inspects cell boundary and cytoplasm, then assesses "
        f"{prior['nucleus']}, followed by {prior['chromatin']}. "
        f"These findings support the category {prior['decision']}."
    )


def normalize_record(record: dict[str, Any], image_root: Optional[str | Path] = None) -> CerviSample:
    image = str(_first_present(record, ("image", "image_path", "path", "file_name")))
    if image_root and not Path(image).is_absolute():
        image = str(Path(image_root) / image)

    width = int(_first_present(record, ("width", "w"), 0) or 0)
    height = int(_first_present(record, ("height", "h"), 0) or 0)
    if width <= 0 or height <= 0:
        width, height = _read_image_size(image)

    label = normalize_label(str(_first_present(record, ("label", "class", "answer", "solution"))))
    if label not in CERVICAL_LABELS:
        raise ValueError(f"Unknown cervical label: {label!r}")

    bbox = _extract_bbox(record, width, height)
    rationale = str(_first_present(record, ("rationale", "cot", "thought"), ""))
    if not rationale:
        rationale = build_rule_based_rationale(label, bbox)

    question = str(_first_present(record, ("question", "problem"), default_question()))
    sample_id = str(_first_present(record, ("id", "sample_id", "problem_id"), ""))
    split_group = str(_first_present(record, SPLIT_GROUP_KEYS, ""))
    split = str(_first_present(record, ("split",), "train"))
    dataset = str(_first_present(record, ("dataset",), "cervicot"))

    return CerviSample(
        image=image,
        label=label,
        bbox=bbox,
        width=width,
        height=height,
        question=question,
        rationale=rationale,
        sample_id=sample_id,
        split_group=split_group,
        split=split,
        dataset=dataset,
    )


def sample_to_groundr1_record(sample: CerviSample, problem_id: int) -> dict[str, Any]:
    x1, y1, x2, y2 = sample.bbox
    rethink = (
        "After focusing on the candidate cell, compare abnormal morphology with the "
        "surrounding background and keep the final label clinically specific."
    )
    solution = SFT_RESPONSE_TEMPLATE.format(
        rationale=sample.rationale,
        rethink=rethink,
        label=sample.label,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )
    return {
        "image": sample.image,
        "width": sample.width,
        "height": sample.height,
        "input_width": sample.width,
        "input_height": sample.height,
        "dataset": sample.dataset,
        "split": sample.split,
        "bboxs": [list(sample.bbox)],
        "problem": sample.question,
        "solution": solution,
        "label": sample.label,
        "rationale": sample.rationale,
        "problem_id": sample.sample_id or problem_id,
    }


def make_train_test_split(
    samples: list[CerviSample],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[CerviSample], list[CerviSample]]:
    rng = random.Random(seed)
    sample_groups: dict[str, list[CerviSample]] = {}
    for index, sample in enumerate(samples):
        sample_groups.setdefault(split_group_key(sample, str(index)), []).append(sample)

    grouped: dict[str, list[list[CerviSample]]] = {}
    for group_samples in sample_groups.values():
        grouped.setdefault(group_representative_label(group_samples), []).append(group_samples)

    train: list[CerviSample] = []
    test: list[CerviSample] = []
    for label_groups in grouped.values():
        label_groups = label_groups[:]
        rng.shuffle(label_groups)
        n_test = max(1, int(round(len(label_groups) * test_ratio))) if len(label_groups) > 1 else 0
        for group_samples in label_groups[:n_test]:
            test.extend(group_samples)
        for group_samples in label_groups[n_test:]:
            train.extend(group_samples)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def split_group_key(sample: CerviSample, fallback: str) -> str:
    return sample.split_group or f"sample:{fallback}"


def group_representative_label(samples: Iterable[CerviSample]) -> str:
    return min((sample.label for sample in samples), key=lambda label: LABEL_PRIORITY.get(label, len(LABEL_PRIORITY)))


def convert_records(
    records: Iterable[dict[str, Any]],
    image_root: Optional[str | Path] = None,
    split_if_missing: bool = False,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> list[dict[str, Any]]:
    samples = [normalize_record(record, image_root=image_root) for record in records]
    if split_if_missing and all(sample.split == "train" for sample in samples):
        train, test = make_train_test_split(samples, test_ratio=test_ratio, seed=seed)
        samples = [
            CerviSample(**{**sample.__dict__, "split": "train"}) for sample in train
        ] + [
            CerviSample(**{**sample.__dict__, "split": "test"}) for sample in test
        ]
    return [sample_to_groundr1_record(sample, idx + 1) for idx, sample in enumerate(samples)]
