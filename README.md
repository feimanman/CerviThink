# CerviThink - v0.2.0-alpha.1

This alpha prerelease contains repairs to the method implementation and experiment workflow
from public commit `134a937`. It is **not a verified reproduction of the paper's
reported scores**. The original evaluator is intentionally unchanged and still
has the invalid-prediction filtering defect documented in the audit.

Do not use the legacy evaluator's numbers as validated results from this branch.
This release does not change or revalidate paper/poster results. See
`RELEASE_NOTES.md` and `docs/known_limitations.md` for the validation boundary.

## Start here

- Release scope and limitations: `RELEASE_NOTES.md`
- Explicit experiment workflow: `docs/reproduction.md`
- Actual training objective and stage protocol: `docs/training.md`
- Dataset/group/label requirements: `docs/datasets.md`

```bash
python -m pip install -e .
python -B scripts/check_method_repair.py
python scripts/run_experiments.py --config configs/method_experiments.json.example
```

The last command prints a plan; it does not launch training. Replace all private
paths and provide actual data, grouping, label maps, and generator versions in a
private copy of the configuration before using `--execute`.

## Scope

- Original test membership is preserved; few-shot selection is inside train.
- Group/WSI overlaps and missing grouping data are rejected by the experiment path.
- Reward ablations share one SFT activation checkpoint, prompts, images, and
  rollout sampler; by default they retain auxiliary forward passes too.
- CoT modes are explicit: `none`, `provided`, `rag`, and `template`.
- Grounding and answering receive a joint clipped policy-gradient loss with
  explicit old-policy likelihoods. The current implementation uses one on-policy
  update per sampled batch (mu=1); it does not implement multi-epoch PPO reuse.
- The input-image coordinate system, main-answer history, and crop auxiliary
  prompt are shared between training and inference.
- A prediction-only entry is available without calling the legacy evaluator:

```bash
python scripts/predict_method.py --model /path/to/checkpoint \
  --input /path/to/cervicot.jsonl --output /path/to/new_predictions.jsonl --seed 42
```

## Outstanding validation

Real Linux/GPU SFT and GRPO smoke tests, clinical data checks, original experiment
configuration reconciliation and evaluation repair remain outstanding for a
verified paper-reproduction release. LICENSE contains the complete standard
Apache-2.0 text; upstream attribution is recorded in NOTICE. Synthetic tests
do not establish paper performance, complete runtime correctness, or clinical validity.

This project is a research artifact, not a clinical diagnostic tool.
