#!/usr/bin/env python3
"""GRPO entry based on the local Ground-R1 trainer."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
MI2026 = ROOT.parent
sys.path.insert(0, str(ROOT))

from cervithink.parsing import (
    answer_matches,
    extract_answer,
    has_cervithink_answer_format,
    has_cervithink_grounding_format,
    has_full_cervithink_format,
    normalize_label,
)
from cervithink.prompts import GROUNDING_PROMPT_TEMPLATE, grounding_prompt, label_options
from cervithink.groundr1_dvhr import install_cervithink_dvhr


def _ground_r1_template() -> str:
    return (
        GROUNDING_PROMPT_TEMPLATE.replace("{question}", "{Question}")
        .replace("{width}", "{input_width}")
        .replace("{height}", "{input_height}")
        .replace("{labels}", label_options())
    )


def _final_text(completion) -> str:
    stages = _completion_stages(completion)
    if stages:
        return stages[-1]
    return str(completion)


def _completion_stages(completion) -> list[str]:
    if isinstance(completion, (list, tuple)):
        return [str(item) for item in completion if str(item).strip()]
    return [str(completion)] if str(completion).strip() else []


def _targets(solution=None, label=None) -> list[str]:
    labels = label or solution or []
    if isinstance(labels, str):
        labels = [labels]
    return [extract_answer(str(item)) for item in labels]


def _fit_length(values: list[str], length: int) -> list[str]:
    values = values[:length]
    if len(values) < length:
        values.extend([""] * (length - len(values)))
    return values


def _optional_column(name: str, kwargs, length: int) -> list[str]:
    values = kwargs.get(name)
    if values is None:
        return [""] * length
    if isinstance(values, str):
        values = [values]
    return _fit_length([str(value or "") for value in values], length)


def _rollout_format_ok(completion) -> bool:
    stages = _completion_stages(completion)
    if not stages:
        return False
    if len(stages) == 1:
        text = stages[0]
        return (
            has_full_cervithink_format(text)
            or has_cervithink_answer_format(text)
            or has_cervithink_grounding_format(text)
        )
    return all(has_cervithink_grounding_format(text) for text in stages[:-1]) and (
        has_full_cervithink_format(stages[-1]) or has_cervithink_answer_format(stages[-1])
    )


def cervithink_accuracy_reward(completions, solution=None, label=None, **kwargs):
    labels = _fit_length(_targets(solution=solution, label=label), len(completions))
    rewards = []
    for completion, target in zip(completions, labels):
        rewards.append(1.0 if answer_matches(_final_text(completion), target) else 0.0)
    return rewards


def cervithink_format_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        rewards.append(1.0 if _rollout_format_ok(completion) else 0.0)
    return rewards


def cervithink_consistency_reward(completions, solution=None, label=None, **kwargs):
    labels = _fit_length(_targets(solution=solution, label=label), len(completions))
    crop_answers = _optional_column("crop_answer", kwargs, len(completions))
    if not any(crop_answers):
        crop_answers = _optional_column("crop_completion", kwargs, len(completions))

    rewards = []
    for completion, target, crop_answer in zip(completions, labels, crop_answers):
        if crop_answer:
            rewards.append(1.0 if answer_matches(crop_answer, target) else 0.0)
            continue
        intermediate_answers = [
            stage for stage in _completion_stages(completion)[:-1] if "<answer>" in stage.lower()
        ]
        rewards.append(1.0 if any(answer_matches(stage, target) for stage in intermediate_answers) else 0.0)
    return rewards


def cervithink_background_reward(completions, **kwargs):
    background_answers = _optional_column("background_answer", kwargs, len(completions))
    if not any(background_answers):
        background_answers = _optional_column("background_completion", kwargs, len(completions))
    rewards = []
    for background_answer in background_answers:
        rewards.append(1.0 if background_answer and normalize_label(extract_answer(background_answer)) == "Normal" else 0.0)
    return rewards


def cervithink_dvhr_reward(completions, solution=None, label=None, **kwargs):
    accuracy = cervithink_accuracy_reward(completions, solution=solution, label=label, **kwargs)
    fmt = cervithink_format_reward(completions, **kwargs)
    consistency = cervithink_consistency_reward(completions, solution=solution, label=label, **kwargs)
    background = cervithink_background_reward(completions, **kwargs)
    return [sum(parts) for parts in zip(fmt, accuracy, consistency, background)]


@dataclass
class CerviThinkScriptArguments:
    dataset_name: str = field(metadata={"help": "Training JSONL path or HF dataset name."})
    dataset_train_split: str = "train"
    dataset_test_split: str = "test"
    dataset_config: Optional[str] = None
    reward_funcs: list[str] = field(default_factory=lambda: ["accuracy", "format", "consistency", "background"])
    score_funcs: list[str] = field(default_factory=list)
    max_pixels: Optional[int] = 401408
    min_pixels: Optional[int] = 3136
    num_generations_stage1: Optional[int] = 4
    answer_rollouts: Optional[int] = 2
    enable_dvhr_auxiliary: bool = True
    dvhr_aux_batch_size: Optional[int] = 4
    dvhr_aux_do_sample: bool = False
    ground_r1_open_r1: str = str(MI2026 / "Ground-R1" / "r1-v" / "src" / "open_r1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-r1-open-r1", default=str(MI2026 / "Ground-R1" / "r1-v" / "src" / "open_r1"))
    bootstrap_args, remaining_args = parser.parse_known_args()

    sys.path.insert(0, bootstrap_args.ground_r1_open_r1)
    try:
        from datasets import Dataset, DatasetDict, load_dataset
        from trainer import Qwen2VLGRPOTrainer
        import trainer.grpo_trainer as ground_trainer
        from trl import GRPOConfig, ModelConfig, TrlParser, get_peft_config
    except Exception as exc:
        raise SystemExit(f"Ground-R1 training environment error: {exc}")

    ground_trainer.STAGE_ONE_TEMPLATE = _ground_r1_template()
    reward_registry = {
        "accuracy": cervithink_accuracy_reward,
        "format": cervithink_format_reward,
        "consistency": cervithink_consistency_reward,
        "background": cervithink_background_reward,
        "dvhr": cervithink_dvhr_reward,
    }

    trl_parser = TrlParser((CerviThinkScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = trl_parser.parse_args_and_config(args=remaining_args)
    needs_auxiliary = any(
        name in {"consistency", "background", "dvhr"} for name in script_args.reward_funcs
    )
    if script_args.enable_dvhr_auxiliary and needs_auxiliary:
        install_cervithink_dvhr(ground_trainer)
    if training_args.num_generations % int(script_args.num_generations_stage1) != 0:
        raise SystemExit(
            "num_generations must be divisible by num_generations_stage1. "
            "For the paper setting use num_generations=8, num_generations_stage1=4, answer_rollouts=2."
        )
    expected = int(script_args.num_generations_stage1) * int(script_args.answer_rollouts)
    if training_args.num_generations != expected:
        raise SystemExit(
            f"Expected num_generations={expected} from G1={script_args.num_generations_stage1} "
            f"and G2={script_args.answer_rollouts}, got {training_args.num_generations}."
        )
    reward_funcs = [reward_registry[name] for name in script_args.reward_funcs]

    if script_args.dataset_name.endswith(".jsonl"):
        ds = Dataset.from_json(script_args.dataset_name)
        if "split" in ds.column_names:
            train_ds = ds.filter(lambda row: row.get("split", "train") == script_args.dataset_train_split)
            test_ds = ds.filter(lambda row: row.get("split", "train") == script_args.dataset_test_split)
            dataset = DatasetDict({"train": train_ds, "test": test_ds})
        else:
            dataset = DatasetDict({"train": ds})
    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": grounding_prompt(
                                example["problem"],
                                int(example.get("input_width", example.get("width", 1024))),
                                int(example.get("input_height", example.get("height", 1024))),
                            ),
                        },
                    ],
                }
            ]
        }

    dataset = dataset.map(make_conversation_image)
    eval_dataset = None
    if training_args.eval_strategy != "no" and script_args.dataset_test_split in dataset:
        eval_dataset = dataset[script_args.dataset_test_split]

    trainer = Qwen2VLGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        score_funcs=[],
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        num_generations_stage1=script_args.num_generations_stage1,
    )
    trainer.cervithink_aux_batch_size = int(script_args.dvhr_aux_batch_size or 4)
    trainer.cervithink_aux_do_sample = bool(script_args.dvhr_aux_do_sample)
    trainer.train()
    trainer.save_state(training_args.output_dir)
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
