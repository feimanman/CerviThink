from pathlib import Path
import random
import sys
import tempfile
import unittest

from PIL import Image

from cervithink.data import CerviSample, make_train_test_split
from cervithink.metrics import classification_report
from cervithink.parsing import extract_answer, extract_box, normalize_label
from cervithink.pipeline import CerviThinkConfig, CerviThinkPipeline
from cervithink.prompts import sft_prompt
from cervithink.rewards import compute_dvhr, grpo_advantages
from cervithink.visual_ops import (
    DEFAULT_INPAINT_RADIUS,
    OPENCV_INPAINT_METHOD,
    make_visual_variants,
    sample_transform_params,
)
from scripts.train_cervithink_grpo import (
    cervithink_background_reward,
    cervithink_consistency_reward,
)
from scripts.generate_cervicot_rationales import run_generator


class CoreTest(unittest.TestCase):
    def test_parse_box_and_label(self):
        self.assertEqual(extract_box("<box>[1, 2, 30, 40]</box>"), (1.0, 2.0, 30.0, 40.0))
        self.assertEqual(extract_answer("<answer>HSIL</answer>"), "HSIL")
        self.assertEqual(normalize_label("ascus"), "ASC-US")
        self.assertEqual(normalize_label("Abnormal"), "Abnormal")
        self.assertEqual(normalize_label("not normal"), "not normal")

    def test_dvhr(self):
        reward = compute_dvhr(
            "<think>x</think> <answer>HSIL</answer>",
            "HSIL",
            crop_completion="<think>x</think> <answer>HSIL</answer>",
            background_completion="<think>x</think> <answer>Normal</answer>",
        )
        self.assertEqual(
            reward.as_dict(),
            {
                "format": 1.0,
                "accuracy": 1.0,
                "consistency": 1.0,
                "background": 1.0,
                "total": 4.0,
            },
        )

    def test_visual_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            _ = Path(tmp)
            image = Image.new("RGB", (64, 64), (220, 220, 220))
            variants = make_visual_variants(image, (10, 10, 30, 30))
            self.assertGreater(variants.focus.size[0], 0)
            self.assertGreater(variants.transform.size[0], 0)
            self.assertEqual(variants.ignore.size, image.size)
        self.assertEqual(DEFAULT_INPAINT_RADIUS, 5)
        self.assertEqual(OPENCV_INPAINT_METHOD, "navier-stokes")
        for _ in range(5):
            scale, contrast = sample_transform_params(random.Random(7))
            self.assertGreaterEqual(scale, 1.0)
            self.assertLessEqual(scale, 3.0)
            self.assertGreaterEqual(contrast, 0.8)
            self.assertLessEqual(contrast, 1.5)

    def test_metrics_and_advantages(self):
        report = classification_report(["HSIL", "Normal"], ["HSIL", "ASC-US"])
        self.assertEqual(report["weighted"]["support"], 2)
        advantages = grpo_advantages([1.0, 2.0, 3.0])
        self.assertEqual(round(sum(advantages), 6), 0.0)

    def test_sft_prompt_matches_solution_tags(self):
        prompt = sft_prompt("Classify this cervical cell image.", 256, 256)
        for tag in ("<think>", "<box>", "<rethink>", "<answer>"):
            self.assertIn(tag, prompt)

    def test_train_test_split_keeps_groups_separate(self):
        samples = [
            CerviSample("a", "HSIL", (0, 0, 32, 32), 64, 64, split_group="patient_1"),
            CerviSample("b", "Normal", (0, 0, 32, 32), 64, 64, split_group="patient_1"),
            CerviSample("c", "HSIL", (0, 0, 32, 32), 64, 64, split_group="patient_2"),
            CerviSample("d", "Normal", (0, 0, 32, 32), 64, 64, split_group="patient_2"),
        ]
        train, test = make_train_test_split(samples, test_ratio=0.5, seed=0)
        train_groups = {sample.split_group for sample in train}
        test_groups = {sample.split_group for sample in test}
        self.assertTrue(train_groups.isdisjoint(test_groups))

    def test_training_rewards_preserve_generation_count(self):
        completions = [
            "<think>a</think> <answer>HSIL</answer>",
            "<think>b</think> <answer>Normal</answer>",
        ]
        consistency = cervithink_consistency_reward(
            completions,
            label=["HSIL"],
            crop_answer=["<answer>HSIL</answer>"],
        )
        background = cervithink_background_reward(
            completions,
            background_answer=["<answer>Normal</answer>"],
        )
        abnormal_background = cervithink_background_reward(
            completions[:1],
            background_answer=["<answer>Abnormal</answer>"],
        )
        self.assertEqual(consistency, [1.0, 0.0])
        self.assertEqual(background, [1.0, 0.0])
        self.assertEqual(abnormal_background, [0.0])

    def test_pipeline_scores_all_rollouts(self):
        class PipelineStub:
            def generate(self, messages, images, max_new_tokens=256, temperature=0.7):
                text = messages[-1]["content"][-1]["text"]
                if "Return exactly one region" in text:
                    return "<think>cell region</think> <box>[8,8,40,40]</box>"
                if "remaining visible background" in text:
                    return "<think>background is clear</think> <answer>Normal</answer>"
                return "<rethink>features support HSIL</rethink> <answer>HSIL</answer>"

        image = Image.new("RGB", (64, 64), (220, 220, 220))
        config = CerviThinkConfig(grounding_rollouts=2, answer_rollouts=2)
        result = CerviThinkPipeline(PipelineStub(), config=config).run(image, true_label="HSIL")
        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(sum(len(candidate.answers) for candidate in result.candidates), 4)
        self.assertEqual(result.prediction, "HSIL")
        self.assertEqual(result.best.reward.total, 4.0)

    def test_pipeline_uses_label_free_selection_without_label(self):
        class AlternatingModel:
            def __init__(self):
                self.answer_calls = 0

            def generate(self, messages, images, max_new_tokens=256, temperature=0.7):
                text = messages[-1]["content"][-1]["text"]
                if "Return exactly one region" in text:
                    return "<think>cell region</think> <box>[8,8,40,40]</box>"
                if "remaining visible background" in text:
                    return "<think>background is clear</think> <answer>Normal</answer>"
                self.answer_calls += 1
                if self.answer_calls == 1:
                    return "<rethink>ambiguous</rethink> <answer>Normal</answer>"
                return "<rethink>features support HSIL</rethink> <answer>HSIL</answer>"

        image = Image.new("RGB", (64, 64), (220, 220, 220))
        config = CerviThinkConfig(grounding_rollouts=1, answer_rollouts=2)
        result = CerviThinkPipeline(AlternatingModel(), config=config).run(image)
        self.assertEqual(result.prediction, "HSIL")
        self.assertGreater(result.best.selection_score, 0.0)

    def test_pipeline_prefers_self_consistent_vote_without_label(self):
        class VotingModel:
            def __init__(self):
                self.final_calls = 0
                self.last_label = "LSIL"

            def generate(self, messages, images, max_new_tokens=256, temperature=0.7):
                text = messages[-1]["content"][-1]["text"]
                if "Return exactly one region" in text:
                    return "<think>cell region</think> <box>[8,8,40,40]</box>"
                if "remaining visible background" in text:
                    return "<think>background is clear</think> <answer>Normal</answer>"
                if "focused cervical cytology crop" in text:
                    return f"<think>crop agrees</think> <answer>{self.last_label}</answer>"
                self.final_calls += 1
                self.last_label = "ASC-US" if self.final_calls == 2 else "LSIL"
                return f"<rethink>features support {self.last_label}</rethink> <answer>{self.last_label}</answer>"

        image = Image.new("RGB", (64, 64), (220, 220, 220))
        config = CerviThinkConfig(grounding_rollouts=1, answer_rollouts=3)
        result = CerviThinkPipeline(VotingModel(), config=config).run(image)
        self.assertEqual(result.prediction, "LSIL")

    def test_pipeline_prediction_normalizes_aliases(self):
        class VerboseModel:
            def generate(self, messages, images, max_new_tokens=256, temperature=0.7):
                text = messages[-1]["content"][-1]["text"]
                if "Return exactly one region" in text:
                    return "<think>cell region</think> <box>[8,8,40,40]</box>"
                if "remaining visible background" in text:
                    return "<think>background is clear</think> <answer>Normal</answer>"
                return (
                    "<rethink>features support high-grade disease</rethink> "
                    "<answer>high-grade squamous intraepithelial lesion</answer>"
                )

        image = Image.new("RGB", (64, 64), (220, 220, 220))
        config = CerviThinkConfig(grounding_rollouts=1, answer_rollouts=1)
        result = CerviThinkPipeline(VerboseModel(), config=config).run(image)
        self.assertEqual(result.prediction, "HSIL")

    def test_rationale_generator_uses_argument_vector(self):
        command = [sys.executable, "-c", "import sys; print(sys.stdin.read().upper())"]
        self.assertEqual(run_generator(command, "abc").strip(), "ABC")


if __name__ == "__main__":
    unittest.main()
