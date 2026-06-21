"""DVHR reward terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .constants import NORMAL_LABEL
from .parsing import (
    answer_matches,
    extract_answer,
    has_cervithink_answer_format,
    has_cervithink_grounding_format,
    has_full_cervithink_format,
    normalize_label,
)


@dataclass(frozen=True)
class RewardWeights:
    format: float = 1.0
    accuracy: float = 1.0
    consistency: float = 1.0
    background: float = 1.0


@dataclass(frozen=True)
class RewardBreakdown:
    format: float
    accuracy: float
    consistency: float
    background: float
    total: float

    def as_dict(self) -> dict[str, float]:
        return {
            "format": self.format,
            "accuracy": self.accuracy,
            "consistency": self.consistency,
            "background": self.background,
            "total": self.total,
        }


def format_reward(text: str, mode: str = "answer") -> float:
    """Reward exact structural compliance."""

    if mode == "grounding":
        return 1.0 if has_cervithink_grounding_format(text) else 0.0
    if mode == "full":
        return 1.0 if has_full_cervithink_format(text) else 0.0
    return 1.0 if has_cervithink_answer_format(text) else 0.0


def accuracy_reward(completion: str, true_label: str) -> float:
    return 1.0 if answer_matches(completion, true_label) else 0.0


def consistency_reward(crop_completion: Optional[str], true_label: str) -> float:
    if not crop_completion:
        return 0.0
    return accuracy_reward(crop_completion, true_label)


def background_normality_reward(background_completion: Optional[str]) -> float:
    if not background_completion:
        return 0.0
    return 1.0 if normalize_label(extract_answer(background_completion)) == NORMAL_LABEL else 0.0


def compute_dvhr(
    final_completion: str,
    true_label: str,
    crop_completion: Optional[str] = None,
    background_completion: Optional[str] = None,
    weights: RewardWeights = RewardWeights(),
    format_mode: str = "answer",
) -> RewardBreakdown:
    """Return the reward breakdown for one rollout."""

    rf = format_reward(final_completion, mode=format_mode)
    ra = accuracy_reward(final_completion, true_label)
    rc = consistency_reward(crop_completion, true_label)
    rb = background_normality_reward(background_completion)
    total = (
        weights.format * rf
        + weights.accuracy * ra
        + weights.consistency * rc
        + weights.background * rb
    )
    return RewardBreakdown(format=rf, accuracy=ra, consistency=rc, background=rb, total=total)


def batch_dvhr(
    final_completions: list[str],
    true_labels: list[str],
    crop_completions: Optional[list[Optional[str]]] = None,
    background_completions: Optional[list[Optional[str]]] = None,
    weights: RewardWeights = RewardWeights(),
) -> list[RewardBreakdown]:
    crop_completions = crop_completions or [None] * len(final_completions)
    background_completions = background_completions or [None] * len(final_completions)
    return [
        compute_dvhr(final, label, crop, background, weights=weights)
        for final, label, crop, background in zip(
            final_completions,
            true_labels,
            crop_completions,
            background_completions,
        )
    ]


def grpo_advantages(rewards: list[float], eps: float = 1e-6) -> list[float]:
    """Normalize group rewards as used by GRPO."""

    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    std = variance ** 0.5
    if std < eps:
        return [0.0 for _ in rewards]
    return [(reward - mean) / std for reward in rewards]
