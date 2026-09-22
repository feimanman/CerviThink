"""Model/config-based loading and private experiment provenance."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess


GROUND_R1_COMMIT = "e10eae6890fd2c2be1aec1745c6b49b43ebc753e"


def check_upstream_revision(path):
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual != GROUND_R1_COMMIT:
        raise ValueError(f"Upstream version mismatch: expected {GROUND_R1_COMMIT}, found {actual}")
    dirty = subprocess.check_output(["git", "-C", str(path), "diff", "--name-only", "HEAD"], text=True)
    if dirty.strip():
        raise ValueError("Upstream has tracked modifications; validate and record them before training")


def load_training_model(path, attn_implementation="flash_attention_2", min_pixels=3136, max_pixels=401408):
    import torch
    from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration
    config = AutoConfig.from_pretrained(path)
    if config.model_type != "qwen2_5_vl":
        raise ValueError(f"Expected Qwen2.5-VL config, found {config.model_type}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path, config=config, torch_dtype=torch.bfloat16, attn_implementation=attn_implementation,
    )
    processor = AutoProcessor.from_pretrained(path, min_pixels=min_pixels, max_pixels=max_pixels)
    processor.pad_token_id = processor.tokenizer.pad_token_id
    processor.eos_token_id = processor.tokenizer.eos_token_id
    return model, processor


def write_training_manifest(output, script_args, training_args, model_args, upstream_path):
    def revision(path):
        result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                                text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else None
    root = Path(__file__).resolve().parents[1]
    packages = {}
    for name in ("torch", "transformers", "trl", "accelerate", "deepspeed"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    payload = {
        "status": "method-repair-not-verified-paper-reproduction",
        "source_commit": revision(root),
        "source_dirty": (
            bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True))
            if revision(root) else None
        ),
        "source_package": (
            json.loads((root/"RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
            if (root/"RELEASE_MANIFEST.json").is_file() else None
        ),
        "source_file_hashes": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for folder in ("cervithink", "scripts") for p in (root/folder).rglob("*.py")},
        "upstream_commit": revision(upstream_path), "expected_upstream_commit": GROUND_R1_COMMIT,
        "packages": packages, "script_args": asdict(script_args),
        "training_args": training_args.to_dict(), "model_args": asdict(model_args),
        "trajectory": "G1 grounding; G2 answers per grounding; fixed focus/transform/ignore",
        "old_policy": "snapshot per batch, one on-policy update per sampled batch (mu=1)",
        "clipping_note": "At mu=1 old/current normally coincide before update; no multi-epoch PPO claim.",
    }
    dataset = Path(script_args.dataset_name)
    if dataset.is_file():
        payload["dataset_sha256"] = hashlib.sha256(dataset.read_bytes()).hexdigest()
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output)/"method_manifest.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
