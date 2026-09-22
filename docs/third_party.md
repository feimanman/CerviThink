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

## Source provenance

The runtime training integration targets:

```text
https://github.com/zzzhhzzz/Ground-R1
commit e10eae6890fd2c2be1aec1745c6b49b43ebc753e
```

Relevant upstream files include:

- `r1-v/src/open_r1/trainer/grpo_trainer.py`, with the HuggingFace 2025
  Apache-2.0 notice.
- `qwen-vl-finetune/qwenvl/train/train_qwen.py`, whose source attributes
  FastChat and Stanford Alpaca and carries the listed 2023 copyright notice.

The original public adapter incorporated the upstream trainer structure.
This release replaces that adapter with an explicit CerviThink trajectory and
uses the upstream class as the training base. The SFT wrapper adapts the upstream
training sequence. Source headers, NOTICE and CHANGELOG record these adaptations.
The full upstream source tree is not redistributed in this package.

`LICENSE` is the complete standard text from:

```text
https://www.apache.org/licenses/LICENSE-2.0.txt
SHA256 cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30
```

Copyright ownership and any institutional release approvals remain matters for
the maintainers to confirm. This technical source/license inventory is not an
institutional authorization or a legal opinion.

## Release Metadata

`CITATION.cff` includes manuscript author/title information and the software alpha
version. No unverified DOI, proceedings pages or publication date is asserted.
The software contributor label is retained rather than inventing individual
copyright ownership.

Run strict metadata checks:

```bash
python3 scripts/open_source_check.py
```
