"""Strict experiment splits; keep the test set frozen before few-shot selection.

These helpers do not infer missing dataset labels or patient identifiers.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

from .constants import CERVICAL_LABELS
from .data import make_train_test_split, normalize_record


def sample_key(row: dict) -> str:
    identity = [row.get("image"), row.get("bbox"), row.get("label")]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def membership_hash(rows: list[dict]) -> str:
    return hashlib.sha256("\n".join(sorted(sample_key(r) for r in rows)).encode()).hexdigest()


def validate_splits(rows: list[dict], group_key: str) -> None:
    groups, images = {}, {}
    for row in rows:
        if not row.get(group_key):
            raise ValueError(f"Every row must have a nonempty {group_key}; no sample-level fallback")
        split = row.get("split")
        if split not in {"train", "test", "val", "unused_train"}:
            raise ValueError(f"Missing or unsupported split: {split!r}")
        # Unused few-shot samples still belong to the original train partition.
        partition = "train" if split == "unused_train" else split
        for mapping, key in ((groups, str(row[group_key])), (images, str(row["image"]))):
            if key in mapping and mapping[key] != partition:
                raise ValueError("Patient/WSI or source image overlaps across partitions")
            mapping[key] = partition
    if not any(r["split"] == "train" for r in rows) or not any(r["split"] == "test" for r in rows):
        raise ValueError("Both nonempty train and test partitions are required")


def select_fewshot(rows: list[dict], fraction: float, seed: int) -> list[dict]:
    """Stratified sample-level selection within train; never reassign test/val.

    Select round-half-up(N_train*fraction) total samples. Per-class quotas are
    proportional, with one sample per present class where the budget allows.
    Remainder rows become unused_train, not test.
    """
    if not 0 < fraction <= 1:
        raise ValueError("train_fraction must be in (0,1]")
    rows = [dict(r) for r in rows]
    buckets = defaultdict(list)
    for i, row in enumerate(rows):
        if row["split"] == "train":
            buckets[row["label"]].append(i)
    n = sum(map(len, buckets.values()))
    target = math.floor(n * fraction + .5)
    if target < len(buckets):
        raise ValueError("Few-shot budget is smaller than the number of training classes")
    quotas = {label: max(1, math.floor(len(indices) * fraction))
              for label, indices in buckets.items()}
    while sum(quotas.values()) < target:
        label = max((k for k in quotas if quotas[k] < len(buckets[k])),
                    key=lambda k: (len(buckets[k])*fraction - quotas[k], k))
        quotas[label] += 1
    while sum(quotas.values()) > target:
        label = max((k for k in quotas if quotas[k] > 1),
                    key=lambda k: (quotas[k] - len(buckets[k])*fraction, k))
        quotas[label] -= 1
    rng = random.Random(seed)
    selected = set()
    for label in sorted(buckets):
        indices = sorted(buckets[label], key=lambda i: sample_key(rows[i]))
        rng.shuffle(indices)
        selected.update(indices[:quotas[label]])
    for i, row in enumerate(rows):
        if row["split"] == "train" and i not in selected:
            row["split"] = "unused_train"
    return rows


def prepare_experiment_rows(
    rows: list[dict], *, group_key: str, image_root: str | None = None,
    create_split: bool = False, test_ratio: float = .2, train_fraction: float = 1.,
    seed: int = 42, label_mapping: dict | None = None,
) -> tuple[list[dict], dict]:
    normalized = []
    excluded = Counter()
    for row in rows:
        row = dict(row)
        if label_mapping is not None:
            raw = str(row.get("label", ""))
            if raw not in label_mapping:
                raise ValueError(f"Missing explicit mapping for source label {raw!r}")
            label = label_mapping[raw]
            if label is None:
                excluded[raw] += 1
                continue
            if label not in CERVICAL_LABELS:
                raise ValueError(f"Invalid target label {label!r}")
            row["label"] = label
        sample = normalize_record(row, image_root)
        row.update(image=str(Path(sample.image).expanduser().resolve()),
                   label=sample.label, bbox=list(sample.bbox),
                   width=sample.width, height=sample.height)
        if not row.get(group_key):
            raise ValueError(f"Missing {group_key}; provide a private grouping manifest")
        normalized.append(row)
    if len({sample_key(r) for r in normalized}) != len(normalized):
        raise ValueError("Duplicate image/bbox/label records detected")
    present = [bool(r.get("split")) for r in normalized]
    if any(present) and not all(present):
        raise ValueError("Partially specified splits; complete the split manifest")
    if not any(present):
        if not create_split:
            raise ValueError("Supply original train/test splits; no implicit 10/90 repartition")
        samples = []
        for row in normalized:
            private = {**row, "patient_id": str(row[group_key])}
            samples.append(normalize_record(private))
        train, test = make_train_test_split(samples, test_ratio, seed)
        test_groups = {s.split_group for s in test}
        for row in normalized:
            row["split"] = "test" if str(row[group_key]) in test_groups else "train"
    validate_splits(normalized, group_key)
    if any(r["split"] == "unused_train" for r in normalized):
        raise ValueError("Input is already subsampled; use the original split manifest")
    test_before = membership_hash([r for r in normalized if r["split"] == "test"])
    original_train_count = sum(r["split"] == "train" for r in normalized)
    result = select_fewshot(normalized, train_fraction, seed)
    validate_splits(result, group_key)
    assert membership_hash([r for r in result if r["split"] == "test"]) == test_before
    manifest = {
        "protocol": "frozen-test-stratified-train-samples-v1",
        "seed": seed, "grouping_column": group_key, "sampling_unit": "cell_sample",
        "requested_train_fraction": train_fraction, "original_train_count": original_train_count,
        "actual_train_fraction": sum(r["split"] == "train" for r in result) / original_train_count,
        "excluded_source_labels": dict(excluded),
        "partitions": {
            split: {"count": len(subset), "label_counts": dict(Counter(r["label"] for r in subset)),
                    "membership_sha256": membership_hash(subset),
                    "sample_keys": sorted(sample_key(r) for r in subset)}
            for split in ("train", "test", "val", "unused_train")
            for subset in [[r for r in result if r["split"] == split]]
        },
    }
    return result, manifest
