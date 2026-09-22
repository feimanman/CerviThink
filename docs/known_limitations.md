# Known limitations

## Legacy evaluation is not validated

`cervithink/metrics.py` discards predictions outside the five-label set before
forming its confusion matrix. For example, a two-sample result with one correct
answer and one empty answer can be reported with support=1 and F1=1.0. This is a
known defect, deliberately deferred from the method-only update.

The existing label normalizer can also select one label from an ambiguous
multi-label response. The corresponding files remain unchanged; baseline
fingerprints are in `configs/evaluator_baseline_sha256.json`.

Use `scripts/predict_method.py` to collect predictions, not legacy metric reports
as evidence of reproduction. Do not omit failed/invalid predictions in downstream
evaluation. The new experiment runner does not call the evaluator.

## Real-model execution remains untested

Local tests do not establish that full 7B training fits the intended GPUs, that
distributed DeepSpeed execution succeeds, or that saved production checkpoints
reload correctly in the full environment. Those checks have been deferred.

## Method semantics and original experiment records

This release implements a fixed grounding-answering process, not a verified
uncertainty-triggered tool controller. The clipped objective has explicit old
likelihoods but uses one on-policy update per sampled batch (`mu=1`); its clipping
is normally inactive at the pre-update point.

Background Normal is a model-derived proxy reward, not independent proof that
all abnormal cells have been removed. Focus/Ignore padding, transformation
semantics and candidate-selection weights are documented but not independently
validated against the original private experiment implementation.

External class mappings, the provenance of Normal examples in ComparisonDetector,
original patient/WSI splits, external adaptation initialization and the exact
rationale generator require the authors' experiment records. The release does
not invent those inputs. Image paths passed to a generator do not by themselves
prove that the generator used visual evidence.

## Data and model artifacts

No clinical images, patient identifiers, original annotations, trained weights
or downloaded PubMed snippets are bundled. Private configurations, generated
rationales and predictions may contain sensitive paths or text and need their own
review before sharing. Software licensing is separate from data/model access and
institutional permissions.

These limitations are compatible with publishing an honestly scoped alpha code
release. They are not evidence that the paper's original experiments were wrong,
nor grounds to claim that this release has reproduced them.
