"""Grounding and answering loop."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .constants import CERVICAL_LABELS, NORMAL_LABEL
from .parsing import (
    extract_answer,
    extract_box,
    has_cervithink_answer_format,
    has_cervithink_grounding_format,
    normalize_label,
)
from .prompts import background_prompt, crop_consistency_prompt, default_question, grounding_prompt
from .protocol import answer_messages, answer_format, policy_box, trajectory_format
from .rewards import RewardBreakdown, compute_dvhr
from .visual_ops import VisualVariants, load_image, make_visual_variants


@dataclass(frozen=True)
class CerviThinkConfig:
    grounding_rollouts: int = 4
    answer_rollouts: int = 2
    transform_zoom: float | None = None
    transform_contrast: float | None = None
    focus_context_scale: float = 1.15
    max_new_tokens: int = 256
    temperature: float = 1.0


@dataclass
class CerviThinkAnswer:
    final_answer: str = ""
    crop_answer: str = ""
    background_answer: str = ""
    reward: RewardBreakdown | None = None
    selection_score: float = 0.0


@dataclass
class CerviThinkCandidate:
    grounding: str
    bbox: tuple[int, int, int, int] | None
    answers: list[CerviThinkAnswer] = field(default_factory=list)
    reward: RewardBreakdown | None = None
    selection_score: float = 0.0
    operation_parameters: dict = field(default_factory=dict)

    @property
    def best_answer(self) -> CerviThinkAnswer | None:
        if not self.answers:
            return None
        scored = [answer for answer in self.answers if answer.reward is not None]
        if scored:
            return max(scored, key=lambda answer: answer.reward.total if answer.reward else -1.0)
        return max(self.answers, key=lambda answer: answer.selection_score)

    @property
    def final_answer(self) -> str:
        best = self.best_answer
        return best.final_answer if best else ""

    @property
    def crop_answer(self) -> str:
        best = self.best_answer
        return best.crop_answer if best else ""

    @property
    def background_answer(self) -> str:
        best = self.best_answer
        return best.background_answer if best else ""


@dataclass
class CerviThinkResult:
    image: str
    question: str
    candidates: list[CerviThinkCandidate] = field(default_factory=list)

    @property
    def best(self) -> CerviThinkCandidate | None:
        if not self.candidates:
            return None
        scored = [candidate for candidate in self.candidates if candidate.reward is not None]
        if scored:
            return max(scored, key=lambda candidate: candidate.reward.total if candidate.reward else -1.0)
        return max(self.candidates, key=lambda candidate: candidate.selection_score)

    @property
    def prediction(self) -> str:
        best = self.best
        if not best:
            return ""
        answer = extract_answer(best.final_answer)
        label = normalize_label(answer)
        return label if label in CERVICAL_LABELS else answer


class CerviThinkPipeline:
    def __init__(self, model, config: CerviThinkConfig | None = None):
        self.model = model
        self.config = config or CerviThinkConfig()

    def _user_message(self, text: str, image_count: int) -> list[dict[str, object]]:
        content: list[dict[str, object]] = [{"type": "image"} for _ in range(image_count)]
        content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}]

    def _ground(self, image: Image.Image, question: str) -> str:
        prompt = grounding_prompt(question, width=image.width, height=image.height)
        return self.model.generate(
            self._user_message(prompt, image_count=1),
            [image],
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )

    def _answer(self, images: list[Image.Image], question: str, grounding: str) -> str:
        return self.model.generate(
            answer_messages(question, images[0].width, images[0].height, grounding),
            [images[0], *images],
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )

    def _crop_answer(self, image: Image.Image, question: str) -> str:
        return self.model.generate(
            self._user_message(crop_consistency_prompt(question), 1), [image],
            max_new_tokens=self.config.max_new_tokens, temperature=0.0,
        )

    def _background_answer(self, image: Image.Image) -> str:
        return self.model.generate(
            self._user_message(background_prompt(), image_count=1),
            [image],
            max_new_tokens=self.config.max_new_tokens,
            temperature=0.0,
        )

    def _variants(self, image: Image.Image, bbox: tuple[float, float, float, float]) -> VisualVariants:
        return make_visual_variants(
            image,
            bbox,
            focus_context_scale=self.config.focus_context_scale,
            transform_zoom=self.config.transform_zoom,
            transform_contrast=self.config.transform_contrast,
        )

    def _label_from_completion(self, completion: str) -> str:
        label = normalize_label(extract_answer(completion))
        return label if label in CERVICAL_LABELS else ""

    def _label_votes(self, result: CerviThinkResult) -> Counter[str]:
        votes: Counter[str] = Counter()
        for candidate in result.candidates:
            for answer in candidate.answers:
                label = self._label_from_completion(answer.final_answer)
                if label:
                    votes[label] += 1
        return votes

    def _answer_selection_score(self, answer: CerviThinkAnswer, votes: Counter[str]) -> float:
        final_label = self._label_from_completion(answer.final_answer)
        crop_label = self._label_from_completion(answer.crop_answer)
        background_label = self._label_from_completion(answer.background_answer)

        score = 0.0
        if answer_format(answer.final_answer):
            score += 1.0
        if final_label:
            score += 2.0
            score += 0.5 * votes.get(final_label, 0)
        if has_cervithink_answer_format(answer.crop_answer):
            score += 0.25
        if crop_label and final_label and crop_label == final_label:
            score += 1.5
        if has_cervithink_answer_format(answer.background_answer):
            score += 0.25
        if background_label == NORMAL_LABEL:
            score += 1.0
        return score

    def _candidate_selection_score(self, candidate: CerviThinkCandidate) -> float:
        score = 0.0
        if candidate.bbox is not None:
            score += 0.5
        if has_cervithink_grounding_format(candidate.grounding):
            score += 1.0
        best = candidate.best_answer
        if best is not None:
            score += best.selection_score
        return score

    def _score_label_free(self, result: CerviThinkResult) -> None:
        votes = self._label_votes(result)
        for candidate in result.candidates:
            for answer in candidate.answers:
                answer.selection_score = self._answer_selection_score(answer, votes)
            candidate.selection_score = self._candidate_selection_score(candidate)

    def run(
        self,
        image: str | Path | Image.Image,
        question: str | None = None,
        true_label: str | None = None,
    ) -> CerviThinkResult:
        img = load_image(image)
        question = question or default_question()
        result = CerviThinkResult(image=str(image), question=question)

        for _ in range(self.config.grounding_rollouts):
            grounding = self._ground(img, question)
            raw_box = policy_box(grounding, img.width, img.height)
            candidate = CerviThinkCandidate(grounding=grounding, bbox=None)
            if raw_box is None:
                result.candidates.append(candidate)
                continue

            variants = self._variants(img, raw_box)
            candidate.bbox = variants.bbox
            candidate.operation_parameters = {"zoom": variants.zoom, "contrast": variants.contrast}
            evidence_images = [img, variants.focus, variants.transform, variants.ignore]

            for _ in range(self.config.answer_rollouts):
                answer = CerviThinkAnswer(
                    final_answer=self._answer(evidence_images, question, grounding),
                    crop_answer=self._crop_answer(variants.focus, question),
                    background_answer=self._background_answer(variants.ignore),
                )
                if true_label is not None:
                    answer.reward = compute_dvhr(
                        grounding + " " + answer.final_answer,
                        true_label,
                        crop_completion=answer.crop_answer,
                        background_completion=answer.background_answer,
                        format_mode="full",
                    )
                candidate.answers.append(answer)

            best_answer = candidate.best_answer
            candidate.reward = best_answer.reward if best_answer else None
            result.candidates.append(candidate)

        if true_label is None:
            self._score_label_free(result)
        return result
