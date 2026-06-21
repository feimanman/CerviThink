# CerviThink

Code for cervical cytology classification with grounded visual reasoning.

The project contains data conversion, visual operations, reward calculation,
rollout inference, and training wrappers for Qwen2.5-VL. The SFT and GRPO
scripts reuse the training stack from Ground-R1.

> This repository is for research use only and is not intended for clinical
> diagnosis or treatment decisions.

## Install

For data preparation and local tests:

```bash
cd CerviThink
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

If `python3 -m venv` is unavailable, install the system `python3-venv` package
or use a managed environment such as Conda. For a non-editable user install,
run `python3 -m pip install --user .`.

For SFT/GRPO training, install the Ground-R1 training environment before
running the training scripts:

```bash
cd /path/to/Ground-R1/r1-v
python -m pip install -e .

cd /path/to/CerviThink
python -m pip install -e .
```

## Data Format

Input annotations are JSONL files with one sample per line:

```json
{"image": "IMAGE_PATH_001", "width": 256, "height": 256, "label": "HSIL", "bbox": [120, 95, 220, 210]}
```

Labels:

```text
HSIL, ASC-H, LSIL, ASC-US, Normal
```

## Data Preparation

For the paper experiments, first crop annotated cell regions and resize them to
`256 x 256`:

```bash
python3 scripts/prepare_cell_crops.py \
  --input YOUR_ANNOTATIONS.jsonl \
  --image-root YOUR_IMAGE_ROOT \
  --output outputs/DST_cropped.jsonl \
  --output-image-dir outputs/DST_256 \
  --split-if-missing
```

Then build CerviCoT. If you already have rationales in the annotation JSONL,
use the direct converter:

```bash
python3 scripts/prepare_cervicot.py \
  --input YOUR_ANNOTATIONS.jsonl \
  --output outputs/cervicot.jsonl \
  --split-if-missing \
  --test-ratio 0.2
```

The converted file follows the Ground-R1 JSONL layout:

```text
image, width, height, input_width, input_height, bboxs, problem, solution, label, split
```

If you have local PubMed retrieval snippets, use the RAG builder:

```bash
python3 scripts/build_cervicot_rag.py \
  --annotations outputs/DST_cropped.jsonl \
  --knowledge YOUR_PUBMED_KNOWLEDGE.jsonl \
  --output outputs/cervicot.jsonl
```

To fetch PubMed snippets into the expected JSONL format:

```bash
python3 scripts/fetch_pubmed_knowledge.py \
  --output outputs/pubmed_cervical_knowledge.jsonl \
  --email YOUR_EMAIL
```

To use a local generator for CerviCoT rationales:

```bash
python3 scripts/generate_cervicot_rationales.py \
  --annotations outputs/DST_cropped.jsonl \
  --knowledge outputs/pubmed_cervical_knowledge.jsonl \
  --output outputs/DST_rationales.jsonl \
  --generator-cmd "python3 /path/to/local_generator.py"
```

The full reproduction script uses this generation path when
`CERVICOT_GENERATOR_CMD` is set in the private environment file.

## Visual Operations

```bash
python3 scripts/make_visual_variants.py \
  --image IMAGE_PATH \
  --bbox "[120,95,220,210]" \
  --output-dir outputs/visual_debug
```

The command writes a focused crop, an enhanced crop, and an ignored image to
the output directory.

## SFT

```bash
export MODEL_NAME_OR_PATH=/path/to/Qwen2.5-VL-7B-Instruct
export DATASET_JSONL=outputs/cervicot.jsonl
export OUTPUT_DIR=outputs/sft
export NPROC_PER_NODE=4

bash scripts/train_sft.sh
```

## GRPO

```bash
export MODEL_NAME_OR_PATH=outputs/sft
export DATASET_JSONL=outputs/cervicot.jsonl
export OUTPUT_DIR=outputs/grpo
export NPROC_PER_NODE=4

bash scripts/train_grpo.sh
```

The GRPO entry uses the paper defaults `G1=4`, `G2=2`, `num_generations=8`,
and `beta=0`. It patches the Ground-R1 trainer at runtime to run auxiliary
crop and background forward passes for the full DVHR reward.

## Inference

```bash
python3 scripts/run_rollout.py \
  --model /path/to/Qwen2.5-VL-7B-Instruct \
  --image IMAGE_PATH
```

Inference does not require labels. When multiple grounding/answering rollouts
are generated, CerviThink selects the final prediction with a label-free score
based on output format, valid label extraction, crop/final consistency,
background normality confirmation, and self-consistency votes across rollouts.
Use `--label` only for debugging on annotated samples; it enables DVHR scoring
and should not be used for blind test inference.

## Evaluation

Prediction JSONL files should contain `label` and `prediction`:

```bash
python3 scripts/evaluate_predictions.py --input YOUR_PREDICTIONS.jsonl
```

For end-to-end reproduction across DST, ComparisonDetector, and HiCervix, see
`docs/reproduction.md` and `configs/reproduce_paper.env.example`.

Baseline evaluation and paper-table aggregation are available through:

```bash
bash scripts/run_hf_baselines.sh
bash scripts/run_vision_baselines.sh
python3 scripts/aggregate_experiment_tables.py --manifest YOUR_MANIFEST.json --output-dir outputs/tables
```

Use `scripts/smoke_training.sh` for a one-step training check before launching
full SFT/GRPO jobs.

## Tests

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
python3 scripts/release_check.py
```

## Citation

If this code is useful for your work, please cite the accompanying paper:

```bibtex
@inproceedings{cervithink2026,
  title={CerviThink: A Reinforced Visual Reasoning Framework for Cervical Cancer Cell Classification},
  author={CerviThink Contributors},
  booktitle={MICCAI},
  year={2026}
}
```
