"""Regression tests for methods only. The legacy evaluator is not repaired here."""
import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from cervithink.data import CerviSample, convert_records, read_jsonl, write_jsonl
from cervithink.experiment_data import prepare_experiment_rows, membership_hash
from cervithink.protocol import policy_box, answer_messages, trajectory_format
from cervithink.prompts import sft_prompt
from cervithink.visual_ops import make_visual_variants, transform_operation
from scripts.prepare_cell_crops import make_resized_sample
from scripts.run_experiments import build_plan
from scripts.train_cervithink_grpo import cervithink_format_reward

ROOT = Path(__file__).resolve().parents[1]


def records(n=100):
    return [
        {"image": f"image_{i}.png", "width": 64, "height": 64,
         "bbox": [4, 4, 40, 40], "label": "HSIL" if i % 2 else "Normal",
         "patient_id": f"patient_{i}", "split": "train" if i < n else "test"}
        for i in range(n+20)
    ]


class MethodDataTests(unittest.TestCase):
    def test_none_mode_never_injects_reasoning(self):
        rows = convert_records(records(4), rationale_mode="none")
        for row in rows:
            self.assertEqual(row["rationale"], "")
            self.assertNotIn("<think>", row["solution"])
            self.assertNotIn("<rethink>", row["solution"])
        prompt = sft_prompt("Classify", 64, 64, rationale_mode="none")
        self.assertNotIn("<think>", prompt)
        self.assertIn("<box>", prompt)

    def test_provided_rationale_required_for_train_only(self):
        with self.assertRaisesRegex(ValueError, "Missing training rationale"):
            convert_records(records(4))
        test = [{**records(4)[-1], "split": "test"}]
        self.assertEqual(convert_records(test)[0]["rationale"], "")

    def test_template_requires_explicit_mode(self):
        result = convert_records(records(4), rationale_mode="template")
        self.assertTrue(result[0]["rationale"])
        self.assertEqual(result[-1]["rationale"], "")

    def test_fewshot_preserves_test_and_exact_budget(self):
        source = records()
        result, manifest = prepare_experiment_rows(source, group_key="patient_id", train_fraction=.1)
        self.assertEqual(sum(r["split"] == "train" for r in result), 10)
        self.assertEqual(sum(r["split"] == "unused_train" for r in result), 90)
        self.assertEqual(sum(r["split"] == "test" for r in result), 20)
        full, _ = prepare_experiment_rows(source, group_key="patient_id")
        self.assertEqual(membership_hash([r for r in full if r["split"] == "test"]),
                         membership_hash([r for r in result if r["split"] == "test"]))
        self.assertEqual(manifest["partitions"]["train"]["label_counts"], {"Normal": 5, "HSIL": 5})

    def test_fewshot_selection_stable_after_input_reordering(self):
        source = records()
        shuffled = list(reversed(source))
        _, a = prepare_experiment_rows(source, group_key="patient_id", train_fraction=.1, seed=123)
        _, b = prepare_experiment_rows(shuffled, group_key="patient_id", train_fraction=.1, seed=123)
        self.assertEqual(a["partitions"], b["partitions"])

    def test_group_missing_overlap_and_partial_split_fail(self):
        for mutation in ("missing", "overlap", "partial"):
            rows = records(4)
            if mutation == "missing":
                rows[0].pop("patient_id")
            elif mutation == "overlap":
                rows[-1]["patient_id"] = rows[0]["patient_id"]
            else:
                rows[0].pop("split")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                prepare_experiment_rows(rows, group_key="patient_id")

    def test_explicit_initial_group_split(self):
        rows = records(20)
        for row in rows:
            row.pop("split")
        with self.assertRaises(ValueError):
            prepare_experiment_rows(rows, group_key="patient_id")
        result, _ = prepare_experiment_rows(rows, group_key="patient_id", create_split=True)
        train = {r["patient_id"] for r in result if r["split"] == "train"}
        test = {r["patient_id"] for r in result if r["split"] == "test"}
        self.assertTrue(train and test and train.isdisjoint(test))

    def test_mapping_is_explicit_never_synthesizes_normal(self):
        rows = records()
        with self.assertRaisesRegex(ValueError, "Missing explicit mapping"):
            prepare_experiment_rows(rows, group_key="patient_id", label_mapping={"HSIL": "HSIL"})

    def test_padded_crop_keeps_inner_target_and_absolute_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"image.png"
            Image.new("RGB", (128, 128)).save(path)
            sample = CerviSample(str(path), "HSIL", (40, 40, 80, 80), 128, 128)
            row = make_resized_sample(sample, Path(tmp)/"out", 1, 256, 2., False, "DST", False, False)
            self.assertEqual(row["bbox"], [64, 64, 192, 192])
            self.assertTrue(Path(row["image"]).is_absolute())


class MethodProtocolTests(unittest.TestCase):
    def test_box_uses_declared_original_coordinates(self):
        self.assertEqual(policy_box("<think>region</think><box>[32,32,128,128]</box>", 256, 256),
                         (32., 32., 128., 128.))
        self.assertIsNone(policy_box("<think>region</think><box>[32,32,400,128]</box>", 256, 256))
        self.assertEqual(policy_box("<think>ignore [1,2,3,4]</think><box>[32,32,128,128]</box>", 256, 256),
                         (32., 32., 128., 128.))

    def test_format_requires_both_stages(self):
        grounding = "<think>x</think><box>[1,2,30,40]</box>"
        answer = "<rethink>y</rethink><answer>HSIL</answer>"
        self.assertEqual(cervithink_format_reward([[grounding, answer], [grounding], [answer]]), [1., 0., 0.])
        self.assertFalse(trajectory_format(grounding, "</rethink>y<rethink><answer>HSIL</answer>"))

    def test_main_prompt_preserves_history_and_image_slots(self):
        messages = answer_messages("Classify", 256, 256, "GROUNDING TEXT")
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "user"])
        self.assertEqual(messages[1]["content"][0]["text"], "GROUNDING TEXT")
        self.assertEqual(sum(c["type"] == "image" for m in messages for c in m["content"]), 5)
        self.assertIn("<rethink>", messages[-1]["content"][-1]["text"])

    def test_transform_magnifies_fixed_focus_not_expanded_box(self):
        image = Image.new("RGB", (128, 128), (128, 128, 128))
        v = make_visual_variants(image, (32, 32, 64, 64), transform_zoom=2, transform_contrast=1)
        self.assertEqual(v.transform.size, (v.focus.width*2, v.focus.height*2))
        self.assertEqual(v.transform.tobytes(), transform_operation(v.focus, zoom=2, contrast=1).tobytes())

    def test_paper_operations_fail_if_opencv_missing(self):
        with patch("cervithink.visual_ops._opencv_inpaint", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "requires OpenCV"):
                make_visual_variants(Image.new("RGB", (64, 64)), (4, 4, 40, 40))


class PlanTests(unittest.TestCase):
    def config(self):
        return {
            "output_root": "outputs", "base_model": "base_model", "ground_r1_root": "upstream",
            "datasets": [{"name": "DST", "annotations": "dst.jsonl", "group_key": "patient_id"},
                         {"name": "HiCervix", "annotations": "hi.jsonl", "group_key": "slide_id",
                          "label_mapping": "mapping.json"}],
            "experiments": [
                {"name": "dst_full", "dataset": "DST", "rationale_mode": "provided"},
                {"name": "dst_base", "dataset": "DST", "rationale_mode": "provided",
                 "rewards": ["accuracy", "format"]},
                {"name": "hi_full", "dataset": "HiCervix", "rationale_mode": "provided"},
            ],
        }

    def test_external_adaptation_and_reward_ablation_share_sft(self):
        plan = build_plan(self.config(), ROOT)
        sft = [s for s in plan if s["name"].endswith("/sft")]
        grpo = [s for s in plan if s["name"].endswith("/grpo")]
        self.assertEqual(len(sft), 2)
        self.assertEqual(len(grpo), 3)
        self.assertEqual(grpo[0]["env"]["MODEL_NAME_OR_PATH"], grpo[1]["env"]["MODEL_NAME_OR_PATH"])
        self.assertNotEqual(grpo[0]["env"]["DATASET_JSONL"], grpo[2]["env"]["DATASET_JSONL"])
        self.assertTrue(all(s["command"][-2:] == ["--enable_dvhr_auxiliary", "true"] for s in grpo))
        self.assertFalse(any("evaluate" in str(s["command"]) for s in plan))

    def test_external_split_and_generator_not_guessed(self):
        config = self.config()
        config["datasets"][1]["create_split"] = True
        with self.assertRaises(ValueError):
            build_plan(config, ROOT)
        config = self.config()
        config["experiments"][0]["rationale_mode"] = "rag"
        with self.assertRaises(ValueError):
            build_plan(config, ROOT)

    def test_prepare_only_cli_end_to_end_without_training_or_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rows = []
            for i in range(30):
                image = tmp/f"image_{i}.png"
                Image.new("RGB", (64, 64), (i, i, i)).save(image)
                rows.append(dict(image=str(image), label="HSIL", bbox=[8, 8, 48, 48],
                                 patient_id=str(i), split="train" if i < 20 else "test"))
            write_jsonl(rows, tmp/"annotations.jsonl")
            cfg = {
                "output_root": str(tmp/"output"), "base_model": str(tmp/"unused_model"),
                "ground_r1_root": str(tmp/"unused_upstream"),
                "datasets": [{"name": "DST", "annotations": str(tmp/"annotations.jsonl"),
                              "group_key": "patient_id", "train_fraction": .1}],
                "experiments": [{"name": "without", "dataset": "DST", "rationale_mode": "none"}],
            }
            path = tmp/"config.json"
            path.write_text(json.dumps(cfg))
            result = subprocess.run([sys.executable, "-B", str(ROOT/"scripts/run_experiments.py"),
                                     "--config", str(path), "--execute", "--prepare-only"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            data = read_jsonl(tmp/"output/activation/DST/none/cervicot.jsonl")
            self.assertEqual(sum(r["split"] == "train" for r in data), 2)
            self.assertEqual(sum(r["split"] == "test" for r in data), 10)
            self.assertTrue(all("<think>" not in r["solution"] for r in data))
            manifest = json.loads((tmp/"output/execution_manifest.json").read_text())
            self.assertEqual(manifest["status"], "prepared")
            self.assertTrue(all(s.startswith("prepare/") for s in manifest["completed"]))


if __name__ == "__main__":
    unittest.main()
