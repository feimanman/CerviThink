# Third-Party Components

This repository contains CerviThink source code only. It does not redistribute
third-party model weights, clinical images, private annotations, or benchmark
datasets.

## Training Stack

- Ground-R1: used for Qwen2.5-VL SFT/GRPO training integration.
- TRL / Transformers / Accelerate / DeepSpeed: used by the Ground-R1 training
  stack.
- PyTorch: model training and inference backend.

Install and use these packages according to their own licenses.

## Models

The scripts can be configured with:

- Qwen2.5-VL
- LLaVA-family checkpoints
- pathology-specific LLaVA variants
- torchvision ImageNet weights for ResNet-50 and ViT-B/16

Model weights are not included. Users must download weights from the original
providers and follow the corresponding license and acceptable-use terms.

## Datasets

The repository does not include:

- DST private hospital data
- ComparisonDetector images or annotations
- HiCervix images or annotations
- PubMed abstracts or downloaded snippets

Dataset users are responsible for obtaining the data through the proper access
channels and complying with all data-use agreements.

## PubMed

`scripts/fetch_pubmed_knowledge.py` uses NCBI E-utilities to fetch literature
metadata and abstracts into a local JSONL file. Follow NCBI usage policies,
provide a contact email when possible, and avoid redistributing downloaded
abstracts unless your use case permits it.

## Clinical Disclaimer

CerviThink is a research artifact for cervical cytology image classification.
It is not intended for clinical diagnosis, screening, treatment planning, or
medical decision-making.

## Release Metadata

Before public release, update:

```text
CITATION.cff
README.md citation block
setup.cfg author field
LICENSE copyright notice
NOTICE copyright notice
```

Run strict metadata checks:

```bash
python3 scripts/open_source_check.py
```
