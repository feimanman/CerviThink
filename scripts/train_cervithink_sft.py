#!/usr/bin/env python3
# Adapted from Ground-R1 / FastChat / Stanford Alpaca training sequence.
# Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
# Licensed under the Apache License, Version 2.0.
# http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# See the License for the specific language governing permissions and limitations.
"""SFT entry with per-run data registration and config-based Qwen2.5-VL loading.

Uses Ground-R1's qwenvl data/attention/trainability utilities. Training sequence
adapted from Ground-R1 qwen-vl-finetune/qwenvl/train/train_qwen.py.
Modified 2026-09-21: no shared upstream JSON overwrite, no cwd-based paths,
no checkpoint-name dispatch, complete processor saving.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def main():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--qwen-finetune-root", required=True)
    bootstrap.add_argument("--sft-json", required=True)
    args, remaining = bootstrap.parse_known_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from cervithink.training_io import check_upstream_revision
    check_upstream_revision(args.qwen_finetune_root)
    sys.path.insert(0, str(Path(args.qwen_finetune_root).resolve()))
    sys.path.insert(0, str(Path(args.qwen_finetune_root).resolve()/"qwenvl/train"))
    import torch
    import transformers
    from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwenvl import data as registry
    from qwenvl.train import train_qwen as upstream

    registry.data_dict["cervithink_local"] = {
        "annotation_path": str(Path(args.sft_json).resolve()), "data_path": "",
    }
    parser = transformers.HfArgumentParser((
        upstream.ModelArguments, upstream.DataArguments, upstream.TrainingArguments,
    ))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=remaining)
    if str(training_args.eval_strategy) not in {"no", "IntervalStrategy.NO"}:
        raise ValueError("This SFT entry uses train-only data; set eval_strategy=no")
    transformers.set_seed(training_args.seed)
    cfg = AutoConfig.from_pretrained(model_args.model_name_or_path)
    if cfg.model_type != "qwen2_5_vl":
        raise ValueError(f"Expected Qwen2.5-VL; found {cfg.model_type}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, config=cfg, cache_dir=training_args.cache_dir,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if training_args.bf16 else None,
    )
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    data_args.image_processor = processor.image_processor
    data_args.model_type = "qwen2.5vl"
    if data_args.data_flatten:
        upstream.replace_qwen2_vl_attention_class()
    model.config.use_cache = False
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length, padding_side="right", use_fast=False,
    )
    upstream.set_model(model_args, model)
    trainer = transformers.Trainer(
        model=model, processing_class=tokenizer, args=training_args,
        **upstream.make_supervised_data_module(tokenizer=tokenizer, data_args=data_args),
    )
    # Resume is explicit; a stale checkpoint in a reused directory is not selected silently.
    trainer.train()
    trainer.save_state()
    model.config.use_cache = True
    upstream.safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    if trainer.is_world_process_zero():
        processor.tokenizer = tokenizer
        processor.save_pretrained(training_args.output_dir)
        import hashlib
        Path(training_args.output_dir, "sft_manifest.json").write_text(json.dumps({
            "status": "method-repair-not-verified-paper-reproduction",
            "sft_json_sha256": hashlib.sha256(Path(args.sft_json).read_bytes()).hexdigest(),
            "training_args": training_args.to_dict(), "model_type": cfg.model_type,
        }, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    main()
