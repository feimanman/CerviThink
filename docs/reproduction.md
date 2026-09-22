# Explicit method experiments (repair branch)

The former single-DST-training / 10:90-external-repartition script has been
replaced. No metrics are automatically calculated in this workflow.

## Configuration

Copy `configs/method_experiments.json.example` to a private location. Relative
paths in that JSON resolve against the configuration's directory. Generator
argv entries should use absolute script/model paths. Keep the configuration and
outputs private: they can contain clinical paths and original group identifiers.

Each dataset needs:

- Raw rows with image, label, bbox, and one explicitly chosen grouping column.
- Original `train`/`test` assignments for external datasets. `val` is supported.
- A complete original-label-to-target-label mapping for external datasets. Null
  means exclude. Unknown source labels are errors. Normal is never fabricated.
- `train_fraction=0.1` for external few-shot runs; default DST fraction is 1.
- HiCervix already-cropped images should use `resize_full_image=true`.

Only DST may explicitly request a new patient-level split via
`create_split=true`, `test_ratio=0.2`. An existing split is always validated.

The few-shot sampling unit is a cell sample within the already isolated train
partition. Stratified quotas total round-half-up(N_train * fraction), with one
per present class if the budget allows; otherwise preparation stops. Unselected
train samples are labeled `unused_train`, never moved to test. The manifest
records realized counts, class supports, seed, and test membership hashes.

## Plan and execute

```bash
python scripts/run_experiments.py --config /path/to/private_config.json
python scripts/run_experiments.py --config /path/to/private_config.json --execute --prepare-only
```

The first command only prints commands. The second runs data preparation on a
fresh output root. To run the complete preparation/training plan, use a separate
fresh output root and run:

```bash
python scripts/run_experiments.py --config /path/to/private_training_config.json --execute
```

Existing nonempty output roots are preserved and rejected, not deleted or
silently reused. This implementation does not resume partial experiment plans.

The bash compatibility entry accepts the same explicit configuration:

```bash
EXPERIMENT_CONFIG=/path/to/private_config.json bash scripts/reproduce_paper.sh
```

Add `--execute` explicitly to launch work. Old `FEWSHOT_TEST_RATIO=0.9`, global
`EVAL_MODEL`, and single `TRAIN_DATASET_JSONL` reproduction semantics are retired.

## Experiment matrix

Every `(dataset, rationale_mode)` gets its own prepared CerviCoT and SFT model.
All reward variants within that activation reuse the exact same SFT checkpoint.
External datasets get their own SFT+GRPO on their selected train samples; no
shared DST model is silently substituted for an external adaptation run.

The current explicit default is independent per-dataset adaptation from the
configured base Qwen model. Whether the original paper instead initialized
external adaptation from a DST-trained model remains an author-confirmation
item; the new default is not evidence of the original protocol.

Supported rationale modes:

- `none`: no reasoning text in the target or SFT prompt; box and answer retained.
- `provided`: requires nonempty supplied training rationale. Use for audited
  manual rationales or actual precomputed rationales, and identify their origin.
- `rag`: requires real local generator argv, model/version, and frozen literature.
- `template`: explicitly named deterministic baseline, not a silent RAG substitute.

RAG only generates on selected training rows. The generator receives JSON stdin
containing `schema`, an absolute `image` path, label, bbox, prompt, and retrieved
literature; it writes plain rationale text to stdout. The generator must actually
open/use the image if the experiment claims visually conditioned explanations.
Providing an image path alone does not prove visual grounding. The repository
does not invent a generator model, clinical rationales, or original retrieval
snapshot. Empty output and timeout are failures.

## Recording and verification

The plan writes `execution_manifest.json`. Training writes per-run manifests;
split membership and source/configuration hashes are recorded. Full test and
unused-train rows never enter SFT/GRPO training. No test-based model selection is
implemented. Use `predict_method.py` to collect seeded predictions without metrics.

`--prepare-only` and CPU unit tests do not validate real-model training. Run
`smoke_training.sh` in the pinned Linux/GPU environment before long jobs. Then
verify a saved checkpoint can be loaded for prediction. Reconcile against the
original experiment records before claiming paper reproduction.
