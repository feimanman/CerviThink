import json
from pathlib import Path
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image
from cervithink.data import write_jsonl, read_jsonl
from cervithink.training_io import load_training_model, write_training_manifest

ROOT = Path(__file__).resolve().parents[1]


class TrainingInterfaceTests(unittest.TestCase):
    def test_training_manifest_supports_source_zip_without_git_history(self):
        @dataclass
        class ScriptArgs:
            dataset_name: str = "nonexistent_test_dataset.jsonl"
        @dataclass
        class ModelArgs:
            model_name_or_path: str = "test_model"
        with tempfile.TemporaryDirectory() as tmp:
            missing_git = subprocess.CompletedProcess([], 128, stdout="", stderr="not a git repository")
            with patch("cervithink.training_io.subprocess.run", return_value=missing_git):
                write_training_manifest(
                    tmp, ScriptArgs(), SimpleNamespace(to_dict=lambda: {"seed": 42}),
                    ModelArgs(), tmp,
                )
            manifest = json.loads(Path(tmp, "method_manifest.json").read_text(encoding="utf-8"))
            self.assertIsNone(manifest["source_commit"])
            self.assertIsNone(manifest["source_dirty"])
            self.assertEqual(manifest["source_package"]["version"], "0.2.0-alpha.1")

    def test_model_loader_uses_config_not_checkpoint_directory_name(self):
        config = SimpleNamespace(model_type="qwen2_5_vl")
        processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=2))
        module = SimpleNamespace(
            AutoConfig=SimpleNamespace(from_pretrained=MagicMock(return_value=config)),
            AutoProcessor=SimpleNamespace(from_pretrained=MagicMock(return_value=processor)),
            Qwen2_5_VLForConditionalGeneration=SimpleNamespace(from_pretrained=MagicMock(return_value=object())),
        )
        with patch.dict(sys.modules, {"transformers": module}):
            model, actual_processor = load_training_model("/renamed/checkpoint")
        module.Qwen2_5_VLForConditionalGeneration.from_pretrained.assert_called_once()
        self.assertEqual(module.AutoConfig.from_pretrained.call_args.args[0], "/renamed/checkpoint")
        self.assertIs(actual_processor, processor)
        self.assertEqual(actual_processor.pad_token_id, 0)

    def test_generator_receives_images_and_only_runs_on_training_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            image = tmp/"image.png"
            Image.new("RGB", (64, 64)).save(image)
            rows = [dict(image=str(image), bbox=[0, 0, 64, 64], label="HSIL", split=split)
                    for split in ("train", "test", "unused_train")]
            write_jsonl(rows, tmp/"in.jsonl")
            write_jsonl([dict(title="Synthetic knowledge", label="HSIL")], tmp/"knowledge.jsonl")
            generator = tmp/"generator.py"
            generator.write_text(
                "import json,sys\nfrom pathlib import Path\n"
                "record=json.load(sys.stdin)\nassert record['schema']=='cervithink-rationale-v1'\n"
                "assert Path(record['image']).is_file()\n"
                "print('Synthetic rationale from image '+Path(record['image']).name)\n"
            )
            command = [
                sys.executable, "-B", str(ROOT/"scripts/generate_cervicot_rationales.py"),
                "--annotations", str(tmp/"in.jsonl"), "--knowledge", str(tmp/"knowledge.jsonl"),
                "--output", str(tmp/"out.jsonl"), "--generator-id", "synthetic-test-v1",
                "--generator-argv-json", json.dumps([sys.executable, str(generator)]),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            out = read_jsonl(tmp/"out.jsonl")
            self.assertIn("image.png", out[0]["rationale"])
            self.assertFalse(out[1].get("rationale"))
            self.assertFalse(out[2].get("rationale"))

    def test_sft_json_contains_absolute_paths_and_correct_none_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            image = tmp/"image.png"
            Image.new("RGB", (64, 64)).save(image)
            row = dict(image=str(image), problem="Classify", width=64, height=64, split="train",
                       rationale_mode="none", solution="<box>[0,0,64,64]</box><answer>HSIL</answer>")
            write_jsonl([row], tmp/"input.jsonl")
            result = subprocess.run([
                sys.executable, "-B", str(ROOT/"scripts/prepare_qwen_sft.py"),
                "--input", str(tmp/"input.jsonl"), "--output", str(tmp/"nested/output.json"),
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            out = json.loads((tmp/"nested/output.json").read_text())
            self.assertTrue(Path(out[0]["image"]).is_absolute())
            self.assertNotIn("<think>", out[0]["conversations"][0]["value"])
            self.assertNotIn("<think>", out[0]["conversations"][1]["value"])


if __name__ == "__main__":
    unittest.main()
