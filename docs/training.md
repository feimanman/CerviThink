# Actual method/training contract - repaired implementation

## Upstream boundary

The entrypoints require Ground-R1 commit
`e10eae6890fd2c2be1aec1745c6b49b43ebc753e` with no tracked changes.
`requirements-train.txt` now pins Transformers 4.51.3 alongside TRL 0.14.0.
This is an interface target, not a claim that the entire GPU environment has
already been smoke-tested. Install the upstream stack and its flash-attention /
vision dependencies in a separate Linux environment; do not replace a running
experiment environment in place.

SFT uses the upstream dataset, attention and trainability helpers through
`train_cervithink_sft.py`. Data is registered per process; no shared upstream
`Ground_SFT.json` is overwritten. All image, dataset and output paths are resolved
before training. Model selection uses `config.model_type`, not checkpoint names.
A full processor/tokenizer is saved with each final model. Resuming old output
checkpoints is not automatic.

SFT text/image context is explicitly configured with `SFT_MAX_LENGTH=4096` by
default; verify actual rationale lengths before long runs. This is a repaired
configuration, not an inferred original paper hyperparameter.

## Fixed trajectory

For each source input image, sample G1=4 grounding texts:

```text
<think>...</think><box>[x1,y1,x2,y2]</box>
```

Coordinates refer to the original input PIL image, not the processor's rounded
patch grid. The box must be finite, inside the input, and nondegenerate. The same
parser is used by training and inference. GT boxes are used in input preparation
and SFT targets, not as an RL IoU reward or inference input.

Each valid grounding produces one set of Focus/Transform/Ignore images, shared
by its G2=2 answers. Stage 2 retains the initial user turn and assistant grounding,
then supplies original/focus/transform/ignore in a new user turn. Its output is:

```text
<rethink>...</rethink><answer>label</answer>
```

The full trajectory therefore contains think/box/rethink/answer. Auxiliary crop
and background prompts are single-view prompts, greedily decoded, used only for
reward calculation. They have no separate direct token loss.

Invalid grounding gets zero reward and no answer-token loss. Training retains
dummy original-image forward calls for these slots to keep distributed generation
schedules equal. Inference reports an invalid candidate instead. No hidden oracle
box or true-label answer-selection step is introduced.

## Objective

Grounding and answering are scored as two explicit stages with separate attention
and generated-token masks. User/padding tokens are excluded. G1 grounding scores
are indexed/reused for the corresponding G2 answer trajectories. Old-policy
likelihoods are recorded under no_grad at rollout time, then current likelihoods
are recomputed for the joint token-normalized loss:

```text
ratio = exp(current_logp - old_logp)
surrogate = min(ratio*A, clip(ratio, 1-epsilon, 1+epsilon)*A)
loss = mean_over_G1xG2(-sum(mask*surrogate) / sum(mask))
```

Defaults: epsilon=.2, population-standard-deviation group normalization,
delta=1e-4, beta=0. Nonzero beta optionally adds reference KL. At beta=0 there is
no reference forward in compute_loss; the legacy base trainer may still allocate
a reference while constructing itself. Memory consumption needs real GPU checks.

The implemented update schedule is mu=1: one on-policy update per sampled batch.
Old/current normally coincide before this update, so clipping is usually locally
inactive. This is not multi-epoch clipped PPO and must not be described as such.
Tests separately exercise ratios outside the clip interval and their gradients.
Sampling temperature is 1, top_p=1, top_k=0, matching raw policy likelihoods.

A max_context_tokens guard stops oversized multimodal inputs; image-token sequences
are not silently truncated. Maximum new tokens per stage defaults to 256.

## Reward ablations

Always use the CerviThink sampler, even for accuracy+format only. The experiment
planner retains both auxiliary forward calls in all reward comparisons. Change
only the selected reward components:

```bash
bash scripts/train_grpo.sh --reward_funcs accuracy format
bash scripts/train_grpo.sh --reward_funcs accuracy format consistency
bash scripts/train_grpo.sh --reward_funcs accuracy format background
bash scripts/train_grpo.sh --reward_funcs accuracy format consistency background
```

The compact `dvhr` alias exists for direct use but must not be combined with its
component rewards. Explicitly disabling auxiliaries is allowed only with no
visual reward selected, and is a compute-budget variant, not the default fair
reward ablation. Batch size per device is currently one; accumulation and GPUs
provide the effective batch.

## Visual operations and limits

Focus retains the existing 1.15 context padding. Transform now zooms that SAME
focus crop (scale [1,3], contrast [.8,1.5]); it does not enlarge the source field
again. Extra sharpening was removed. Ignore inpaints the original input using
the 1.08-dilated predicted box, OpenCV NS radius=5. A missing OpenCV backend is an
error for method runs, not a silent blur substitution.

These are explicit repaired semantics, not a proven recovery of the original
private implementation. The workflow is fixed two-stage; there is no learned
uncertainty-triggered tool controller or validated iterative clinical correction.
Background Normal remains a proxy reward; no new gating/weights were invented.

## Inference

Use `scripts/predict_method.py` for seeded prediction-only verification. It uses
the same dialogue construction, coordinate protocol, views and greedy crop/bg
prompts. Existing label-free candidate selection weights are retained pending
original-protocol confirmation. The legacy metric and label-normalization code
is intentionally unchanged; output scores from it remain unvalidated.
