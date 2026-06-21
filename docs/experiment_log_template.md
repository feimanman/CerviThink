# Experiment Log

Use this template for each public result table.

## Run Metadata

```text
Run name:
Date:
Commit:
Host:
CUDA devices:
Python:
PyTorch:
Transformers:
TRL:
Ground-R1 path or commit:
```

## Data

```text
Dataset:
Annotation file:
Image root:
Number of rows:
Train rows:
Test rows:
Label mapping:
Crop command:
RAG command:
```

## Training

```text
Base model:
SFT checkpoint:
GRPO checkpoint:
SFT command:
GRPO command:
G1:
G2:
GRPO generations:
KL beta:
DVHR auxiliary forward:
Random seed:
```

## Evaluation

```text
Evaluation command:
Prediction JSONL:
Metric JSON:
Weighted Precision:
Weighted Recall:
Weighted F1:
```

## Notes

Record deviations from the paper setup, failed runs, manual label mappings, and
hardware or dependency changes.
