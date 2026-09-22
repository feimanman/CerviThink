# Changelog

## v0.2.0-alpha.1 — prerelease candidate

### Method and workflow changes relative to `134a937`

- `cervithink/data.py`, `experiment_data.py`, and preparation scripts:
  explicit rationale modes, validated group partitions, train-only few-shot
  sampling, correct inner target coordinates for padded input crops.
- `protocol.py`, `pipeline.py`, `prompts.py`, `hf_qwen.py`, `visual_ops.py`:
  shared two-stage interaction/coordinate protocol; fixed-focus transforms;
  explicit OpenCV requirement; matched auxiliary prompts.
- `groundr1_dvhr.py` and `grpo_objective.py`: replace the original rollout adapter
  with explicit two-stage sampling, old-policy log-probabilities, masks and the
  single-update clipped objective. These semantics require experimental
  revalidation; they are not asserted to recover the original private trainer.
- `train_cervithink_sft.py`, `train_cervithink_grpo.py`, shell wrappers,
  `training_io.py`: configuration-based model loading, per-run SFT registration,
  correct state saving, processor export and provenance.
- `run_experiments.py` and example configuration: explicit per-dataset experiment
  plans; shared activation checkpoint for reward comparisons; no automatic metric
  calculation; no silent overwrite of previous outputs.
- `predict_method.py`: seeded prediction-only output.
- Tests: data, protocol, objective, adapter, generator and model-loading checks.

### Distribution changes

- Standard Apache-2.0 license text replaces the shortened nonstandard text.
- Upstream notices and manuscript citation metadata are recorded.
- Alpha version, release notes and known limitations are explicit.
- Local workspace handoff documents and execution outputs are not distributed.
- The method checker works from a source ZIP using baseline fingerprints,
  without requiring the original Git history.
- Training metadata can identify a source package when a checkout has no `.git`.
- Release checks do not recursively delete local caches or execute training jobs.

### Intentionally deferred

- Legacy metric and label-normalization repair.
- Validation of original benchmark predictions and reported numerical results.
- Full real-model training, distributed execution and save/reload testing.
- Missing original dataset mappings, generator specification and experiment records.

This changelog describes code changes, not independently verified changes in
model quality.
