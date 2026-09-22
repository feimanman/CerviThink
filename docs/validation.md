# Validation record for v0.2.0-alpha.1

## Checks performed on the release candidate

- 43 local regression tests passed, with zero failures, errors or skips.
- Python source parsing and Bash script syntax checks passed.
- The source-only release inventory and common secret/local-path scans passed.
- LICENSE matches the complete standard Apache-2.0 text by SHA-256.
- Six legacy evaluation-related files match baseline `134a937` fingerprints.
- The method check runs from a source directory with no `.git` history.

Local environment: Windows, Python 3.13.2, PyTorch 2.6.0+cu124,
Transformers 4.51.3. The model tests used CPU tensors. No datasets, TRL,
DeepSpeed or flash-attention training stack was installed as part of this
validation.

Tests include data/split preparation, group-overlap protection, few-shot budgets,
explicit no-CoT conversion, generator input handling, prompt/coordinate
consistency, clipped gradients, generated-token masks, synthetic adapter
forward/backward and a tiny randomly initialized Qwen2.5-VL multimodal
forward/backward. Generated responses in the tiny-model test are scripted;
this is not a trained-model inference or performance test.

## Not performed

- Actual 7B SFT or GRPO training, including distributed execution.
- Real training checkpoint save/reload validation.
- Clinical dataset evaluation or paper-metric reproduction.
- Proof that all execution paths, model configurations or input edge cases work.
- Institutional/legal authorization or privacy validation for data not bundled.

## Re-run checks

With the test dependencies installed:

```bash
python -B scripts/release_check.py
```

For a non-training inventory scan only:

```bash
python -B scripts/release_check.py --skip-tests
```

The full check writes a local report under `outputs/release_checks/`, which is
excluded from the release source selection. Passing checks does not promote the
untested capabilities above to verified status.
