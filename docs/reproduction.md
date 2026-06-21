# Reproducing The Paper Experiments

This repository contains the experiment pipeline used by CerviThink:

1. Crop annotated cells and resize them to `256 x 256`.
2. Build CerviCoT JSONL records with retrieval-augmented generated rationales.
3. Run Qwen2.5-VL SFT with the Ground-R1 finetuning stack.
4. Run GRPO with `G1=4`, `G2=2`, and `beta=0`.
5. Evaluate weighted Precision, Recall, and F1 on DST, ComparisonDetector, and HiCervix.

Private hospital images, downloaded public datasets, and PubMed snippets are not
included in this repository. Prepare local JSONL files with these fields:

```json
{"image": "IMAGE_PATH_001", "label": "HSIL", "bbox": [120, 95, 220, 210]}
```

For PubMed knowledge, use JSONL rows such as:

```json
{"title": "Cervical cytology review", "abstract": "Text used for retrieval.", "label": "HSIL"}
```

You can create this file with NCBI E-utilities:

```bash
python3 scripts/fetch_pubmed_knowledge.py \
  --output outputs/pubmed_cervical_knowledge.jsonl \
  --retmax 20 \
  --email YOUR_EMAIL
```

## One-Command Pipeline

Create a private environment file from `configs/reproduce_paper.env.example`,
then run:

```bash
set -a
source /path/to/private_reproduce_paper.env
set +a
bash scripts/reproduce_paper.sh
```

The script writes cropped datasets, CerviCoT JSONL files, checkpoints,
predictions, and metric reports under `OUTPUT_ROOT`. Set
`CERVICOT_GENERATOR_CMD` to use a local rationale generator for the paper-style
RAG CerviCoT construction. If it is unset, the script uses deterministic
retrieval-augmented rationale composition; set `CERVICOT_USE_FALLBACK_TEMPLATE=1`
to run the generator wrapper without an external model.

## Manual Steps

Crop and resize DST or ComparisonDetector cells:

```bash
python3 scripts/prepare_cell_crops.py \
  --input YOUR_ANNOTATIONS.jsonl \
  --image-root YOUR_IMAGE_ROOT \
  --output outputs/DST_cropped.jsonl \
  --output-image-dir outputs/DST_256 \
  --dataset DST \
  --split-if-missing \
  --test-ratio 0.2
```

Build CerviCoT with deterministic retrieval-augmented rationale composition:

```bash
python3 scripts/build_cervicot_rag.py \
  --annotations outputs/DST_cropped.jsonl \
  --knowledge YOUR_PUBMED_KNOWLEDGE.jsonl \
  --output outputs/DST_cervicot.jsonl
```

For the paper-style generation-based CerviCoT variant, first create
rationale-enriched rows with a local generator command:

```bash
python3 scripts/generate_cervicot_rationales.py \
  --annotations outputs/DST_cropped.jsonl \
  --knowledge outputs/pubmed_cervical_knowledge.jsonl \
  --output outputs/DST_rationales.jsonl \
  --generator-cmd "python3 /path/to/local_generator.py"
```

The command in `--generator-cmd` reads the prompt from stdin and writes the
rationale to stdout. Use `--fallback-template` for a deterministic dry run.
`scripts/reproduce_paper.sh` takes this path automatically when
`CERVICOT_GENERATOR_CMD` is set.

Run SFT and GRPO:

```bash
export MODEL_NAME_OR_PATH=/path/to/Qwen2.5-VL-7B-Instruct
export DATASET_JSONL=outputs/DST_cervicot.jsonl
export OUTPUT_DIR=outputs/sft
bash scripts/train_sft.sh

export MODEL_NAME_OR_PATH=outputs/sft
export OUTPUT_DIR=outputs/grpo
bash scripts/train_grpo.sh
```

Evaluate:

```bash
python3 scripts/evaluate_dataset.py \
  --model outputs/grpo \
  --input outputs/DST_cervicot.jsonl \
  --output outputs/DST_predictions.jsonl \
  --report-output outputs/DST_metrics.json
```

Summarize reports:

```bash
python3 scripts/summarize_metrics.py outputs/*_metrics.json
```

Check the training environment before launching long jobs:

```bash
python3 scripts/check_training_environment.py \
  --dataset-jsonl outputs/DST_cervicot.jsonl \
  --model /path/to/Qwen2.5-VL-7B-Instruct
```

Run a one-step SFT/GRPO smoke test:

```bash
export MODEL_NAME_OR_PATH=/path/to/Qwen2.5-VL-7B-Instruct
export DATASET_JSONL=outputs/DST_cervicot.jsonl
bash scripts/smoke_training.sh
```

Run Hugging Face baselines:

```bash
export BASELINE_CONFIG=/path/to/baselines.json
export DATASET_JSONL=outputs/DST_cervicot.jsonl
export OUTPUT_DIR=outputs/baselines/DST
bash scripts/run_hf_baselines.sh
```

Run vision-only baselines:

```bash
export BASELINE_CONFIG=configs/vision_baselines.json.example
export DATASET_JSONL=outputs/DST_cervicot.jsonl
export OUTPUT_DIR=outputs/vision_baselines/DST
bash scripts/run_vision_baselines.sh
```

For baselines that require external services or separate upstream code, such as
Gemini-2.5 or the original ComparisonDetector model, export their predictions as
JSONL with `label` and `prediction`, then run `scripts/evaluate_predictions.py`
and add the metric JSON to the experiment manifest.

Aggregate paper tables:

```bash
python3 scripts/aggregate_experiment_tables.py \
  --manifest /path/to/experiment_manifest.json \
  --output-dir outputs/tables
```

## Paper Settings

The default scripts use the paper settings:

```text
Backbone: Qwen2.5-VL-7B
Image size: 256 x 256
DST train/test split: 4:1
ComparisonDetector/HiCervix training split: 10% few-shot
SFT learning rate: 1e-6
GRPO learning rate: 5e-7
SFT effective batch size: 8
GRPO effective batch size: 4
Training hardware: 4 x NVIDIA A100
DeepSpeed: ZeRO-2
Grounding rollouts: G1=4
Answer rollouts: G2=2
GRPO generations: 8
KL beta: 0
DVHR auxiliary forward: enabled
Transform scale: uniform [1, 3]
Transform contrast: uniform [0.8, 1.5]
Ignore inpainting: OpenCV Navier-Stokes, radius 5
Metrics: weighted Precision, weighted Recall, weighted F1
Labels: HSIL, ASC-H, LSIL, ASC-US, Normal
```

## Ablations

Reward ablations can be launched by overriding `reward_funcs`:

```bash
bash scripts/train_grpo.sh --reward_funcs accuracy format --enable_dvhr_auxiliary false
bash scripts/train_grpo.sh --reward_funcs accuracy format consistency
bash scripts/train_grpo.sh --reward_funcs accuracy format background
bash scripts/train_grpo.sh --reward_funcs accuracy format consistency background
```

CerviCoT ablations use the same training commands with different SFT JSONL
files:

```text
Without CerviCoT: omit rationale/cot/thought fields before conversion.
Partial CerviCoT: provide manual rationale fields.
CerviThink: generate rationales with scripts/generate_cervicot_rationales.py
and convert them with scripts/build_cervicot_rag.py.
```
