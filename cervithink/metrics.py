"""Classification metrics without requiring scikit-learn."""

from __future__ import annotations

from dataclasses import dataclass

from .constants import CERVICAL_LABELS
from .parsing import normalize_label


@dataclass(frozen=True)
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int


def confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: tuple[str, ...] = CERVICAL_LABELS,
) -> list[list[int]]:
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for truth, pred in zip(y_true, y_pred):
        t = normalize_label(truth)
        p = normalize_label(pred)
        if t in index and p in index:
            matrix[index[t]][index[p]] += 1
    return matrix


def classification_report(
    y_true: list[str],
    y_pred: list[str],
    labels: tuple[str, ...] = CERVICAL_LABELS,
) -> dict[str, object]:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    per_class: dict[str, ClassMetrics] = {}
    total_support = 0
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0

    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fp = sum(matrix[row][i] for row in range(len(labels))) - tp
        fn = sum(matrix[i][col] for col in range(len(labels))) - tp
        support = sum(matrix[i])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = ClassMetrics(precision, recall, f1, support)
        total_support += support
        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support

    if total_support:
        weighted_precision /= total_support
        weighted_recall /= total_support
        weighted_f1 /= total_support

    return {
        "labels": labels,
        "confusion_matrix": matrix,
        "per_class": {label: metrics.__dict__ for label, metrics in per_class.items()},
        "weighted": {
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1": weighted_f1,
            "support": total_support,
        },
    }
