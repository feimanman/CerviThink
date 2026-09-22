#!/usr/bin/env python3
"""Plan/execute explicit method experiments. Metrics are intentionally deferred."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def build_plan(config: dict, config_dir: Path) -> list[dict]:
    def path(value):
        p = Path(value).expanduser()
        return str((p if p.is_absolute() else config_dir/p).resolve())
    output = Path(path(config["output_root"]))
    model = path(config["base_model"])
    upstream = path(config["ground_r1_root"])
    seed = int(config.get("seed", 42))
    nproc = int(config.get("nproc_per_node", 4))
    steps = []
    datasets = {}

    def add(name, command, env=None):
        steps.append({"name": name, "command": command, "env": env or {}})

    for dataset in config["datasets"]:
        name = dataset["name"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in datasets:
            raise ValueError("Dataset names must be unique simple path components")
        external = name in {"ComparisonDetector", "HiCervix"}
        if external and dataset.get("create_split", False):
            raise ValueError("External few-shot experiments must preserve supplied original partitions")
        if external and not dataset.get("label_mapping"):
            raise ValueError("External datasets require an explicit label-mapping file; Normal is not inferred")
        base = output/"data"/name
        split = str(base/"split.jsonl")
        crop = str(base/"cropped.jsonl")
        args = [sys.executable, str(ROOT/"scripts/prepare_experiment_split.py"),
                "--input", path(dataset["annotations"]), "--output", split,
                "--manifest", str(base/"split_manifest.json"), "--group-key", dataset["group_key"],
                "--seed", str(seed), "--train-fraction", str(dataset.get("train_fraction", .1 if external else 1.))]
        if dataset.get("image_root"):
            args += ["--image-root", path(dataset["image_root"])]
        if dataset.get("label_mapping"):
            args += ["--label-mapping", path(dataset["label_mapping"])]
        if dataset.get("create_split", False):
            args += ["--create-split", "--test-ratio", str(dataset.get("test_ratio", .2))]
        add(f"prepare/{name}/split", args)
        args = [sys.executable, str(ROOT/"scripts/prepare_cell_crops.py"),
                "--input", split, "--output", crop, "--output-image-dir", str(base/"images"),
                "--dataset", name, "--context-scale", str(dataset.get("context_scale", 1.))]
        if dataset.get("resize_full_image", name == "HiCervix"):
            args += ["--resize-full-image"]
        add(f"prepare/{name}/crop", args)
        datasets[name] = crop

    names = set()
    activations = set()
    for experiment in config["experiments"]:
        name, dataset = experiment["name"], experiment["dataset"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in names:
            raise ValueError("Experiment names must be unique simple path components")
        names.add(name)
        crop = datasets[dataset]
        out = output/"runs"/name
        mode = experiment["rationale_mode"]
        if mode not in {"none", "provided", "rag", "template"}:
            raise ValueError("Explicit rationale mode required")
        activation = output/"activation"/dataset/mode
        activation_key = (dataset, mode)
        first_activation = activation_key not in activations
        cervicot = str(activation/"cervicot.jsonl")
        enriched = crop
        if mode == "rag" and first_activation:
            generator = config.get("rationale_generator", {})
            if not generator.get("argv") or not generator.get("model_id"):
                raise ValueError("RAG requires a real generator argv and model/version; no silent template fallback")
            enriched = str(activation/"generated_rationales.jsonl")
            add(f"prepare/{name}/rag", [
                sys.executable, str(ROOT/"scripts/generate_cervicot_rationales.py"),
                "--annotations", crop, "--knowledge", path(config["knowledge"]),
                "--output", enriched, "--generator-argv-json", json.dumps(generator["argv"]),
                "--generator-id", generator["model_id"], "--generator-input-format", "json",
            ])
        if first_activation:
            add(f"prepare/{name}/convert", [
                sys.executable, str(ROOT/"scripts/prepare_cervicot.py"), "--input", enriched,
                "--output", cervicot, "--rationale-mode", "provided" if mode == "rag" else mode,
            ])
        rewards = experiment.get("rewards", ["accuracy", "format", "consistency", "background"])
        valid_rewards = {"accuracy", "format", "consistency", "background"}
        if not rewards or len(set(rewards)) != len(rewards) or not set(rewards) <= valid_rewards:
            raise ValueError("Use unique explicit reward components, not overlapping DVHR aliases")
        env = {
            "MODEL_NAME_OR_PATH": model, "DATASET_JSONL": cervicot,
            "QWEN_FINETUNE_ROOT": str(Path(upstream)/"qwen-vl-finetune"),
            "GROUND_R1_OPEN_R1": str(Path(upstream)/"r1-v/src/open_r1"),
            "NPROC_PER_NODE": str(nproc), "SEED": str(seed), "PYTHON_BIN": sys.executable,
            "REPORT_TO": "none", "USE_DEEPSPEED": str(config.get("use_deepspeed", 1)),
        }
        sft_env = {**env, "OUTPUT_DIR": str(activation/"sft"),
                   "LEARNING_RATE": str(config.get("sft_learning_rate", 1e-6)),
                   "PER_DEVICE_TRAIN_BATCH_SIZE": "1", "GRADIENT_ACCUMULATION_STEPS": "2"}
        if first_activation:
            add(f"train/{name}/sft", ["bash", str(ROOT/"scripts/train_sft.sh")], sft_env)
            activations.add(activation_key)
        grpo_env = {**env, "MODEL_NAME_OR_PATH": str(activation/"sft"), "OUTPUT_DIR": str(out/"grpo"),
                    "LEARNING_RATE": str(config.get("grpo_learning_rate", 5e-7)),
                    "GRADIENT_ACCUMULATION_STEPS": "1"}
        add(f"train/{name}/grpo", [
            "bash", str(ROOT/"scripts/train_grpo.sh"), "--reward_funcs", *rewards,
            # Keep auxiliary calls even when their reward weights are zero.
            "--enable_dvhr_auxiliary", "true",
        ], grpo_env)
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute", action="store_true", help="Without this flag, only print the plan")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    steps = build_plan(config, config_path.parent)
    if args.prepare_only:
        steps = [s for s in steps if s["name"].startswith("prepare/")]
    if not args.execute:
        print(json.dumps({"status": "PLAN ONLY; evaluator remains deferred", "steps": steps}, indent=2))
        return
    output = Path(config["output_root"])
    output = (output if output.is_absolute() else config_path.parent/output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output_root; existing experiment artifacts are preserved")
    output.mkdir(parents=True, exist_ok=True)
    plan_file = output/"execution_manifest.json"
    manifest = {
        "status": "running", "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "config": config, "steps": steps, "completed": [], "metrics": "DEFERRED, NOT VERIFIED",
    }
    def save():
        plan_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    save()
    try:
        for step in steps:
            env = {**os.environ, **step["env"]}
            subprocess.run(step["command"], env=env, cwd=ROOT, check=True)
            manifest["completed"].append(step["name"])
            save()
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        save()
        raise
    manifest["status"] = "prepared" if args.prepare_only else "training-complete-pending-validation"
    save()


if __name__ == "__main__":
    main()
