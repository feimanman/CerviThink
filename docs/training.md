# Training

## SFT

The SFT script writes `qwenvl/data/Ground_SFT.json` inside the Qwen-VL
finetuning directory and launches `qwenvl/train/train_qwen.py`.

```bash
export QWEN_FINETUNE_ROOT=/path/to/Ground-R1/qwen-vl-finetune
export MODEL_NAME_OR_PATH=/path/to/Qwen2.5-VL-7B-Instruct
export DATASET_JSONL=outputs/cervicot.jsonl
export OUTPUT_DIR=outputs/sft

bash scripts/train_sft.sh
```

## GRPO

The GRPO script loads the Ground-R1 trainer from `GROUND_R1_OPEN_R1`.

```bash
export GROUND_R1_OPEN_R1=/path/to/Ground-R1/r1-v/src/open_r1
export MODEL_NAME_OR_PATH=outputs/sft/checkpoint
export DATASET_JSONL=outputs/cervicot.jsonl
export OUTPUT_DIR=outputs/grpo

bash scripts/train_grpo.sh
```

The GRPO script follows the paper setting `G1=4`, `G2=2`, so it launches
`num_generations=8` and `num_generations_stage1=4`. It also sets `beta=0` to
match the paper objective without a KL penalty.

The reward registry contains `accuracy`, `format`, `consistency`, `background`,
and `dvhr`. By default, `scripts/train_cervithink_grpo.py` patches the
Ground-R1 trainer at runtime so grounding rollouts are followed by a separate
answering prompt over the original, focused, transformed, and ignored images.
Each rollout also receives:

```text
crop_answer: auxiliary classification on the focused crop
background_answer: auxiliary classification on the ignored/inpainted image
```

These auxiliary forward passes are consumed by the consistency and background
normality rewards. Set `--enable_dvhr_auxiliary false` only for ablations that
intentionally remove this behavior.

For the complete end-to-end experiment flow, see `docs/reproduction.md`.
