# v0.2.0-alpha.1 — Method implementation and experiment workflow

This alpha prerelease updates CerviThink's method implementation and experiment
workflow from baseline commit `134a937`.

## Changes

- Preserve existing test partitions and select few-shot samples only within train.
- Validate patient/WSI grouping and require explicit external-dataset label maps.
- Keep reward-ablation trajectories, prompts and image operations consistent,
  and share the same SFT checkpoint within each activation group.
- Make no-CoT, provided rationale, RAG and template modes explicit.
- Align grounding coordinates, answer history and auxiliary prompts across
  training and inference.
- Add explicit old-policy likelihoods and a token-masked clipped objective,
  with a single on-policy update per sampled batch (`mu=1`).
- Repair model loading, path handling, final-state saving and processor export.
- Include a seeded prediction-only entrypoint that does not compute metrics.
- Restore the complete standard Apache-2.0 license and record upstream notices.

## Validation status

Local regression tests cover synthetic data preparation, method logic, gradients,
and an actual image/text forward-backward pass through a tiny randomly initialized
Qwen2.5-VL model. No pretrained model weights were downloaded for these tests.
See `docs/validation.md` for the recorded test run.

Full 7B SFT/GRPO execution, save/reload validation on the real training stack, and
reproduction of the paper's numerical results have not been performed for this
release. The legacy evaluator is intentionally unchanged and has a known
invalid-prediction filtering issue. Its metrics must not be treated as validated
paper results from this release.

This is a method-code prerelease, not a clinical tool or a claim of verified
paper reproduction. See `docs/known_limitations.md`.
