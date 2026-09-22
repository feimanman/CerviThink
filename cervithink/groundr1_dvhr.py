"""Fixed two-stage CerviThink rollout adapter for the pinned Ground-R1 trainer.

Modified 2026-09-21: independent stage masks, original-image coordinates,
shared visuals across G2 answers, and explicit old-policy log probabilities.
Integration target: Ground-R1 e10eae6890fd2c2be1aec1745c6b49b43ebc753e.
The underlying trainer originates from Ground-R1 / HuggingFace (Apache-2.0);
this replacement implements the CerviThink-specific trajectory explicitly.
"""
from __future__ import annotations

import copy

from .grpo_objective import joint_clipped_loss
from .prompts import background_prompt, crop_consistency_prompt
from .protocol import answer_messages, grounding_messages, policy_box, user_message
from .visual_ops import load_image, make_visual_variants


def install_cervithink_dvhr(ground_trainer) -> None:
    """Install the SAME trajectory sampler for every reward ablation."""
    trainer_cls = ground_trainer.Qwen2VLGRPOTrainer
    if getattr(trainer_cls, "_cervithink_dvhr_installed", False):
        return
    torch = ground_trainer.torch

    def encode(self, messages, images):
        texts = [self.processing_class.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True, add_vision_id=True,
        ) for message in messages]
        inputs = self.processing_class(
            text=texts, images=[image for group in images for image in group],
            return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False,
        )
        context_limit = getattr(self, "cervithink_max_context_tokens", 8192)
        if inputs["input_ids"].shape[1] + self.generation_config.max_new_tokens > context_limit:
            raise ValueError("Multimodal context exceeds max_context_tokens; reduce image/token budget explicitly")
        # Never truncate multimodal input IDs independently of image features.
        return super(trainer_cls, self)._prepare_inputs(inputs)

    def sample(self, model, messages, images, *, auxiliary=False):
        encoded = encode(self, messages, images)
        config = copy.deepcopy(self.generation_config)
        config.num_return_sequences = 1
        config.do_sample = not auxiliary
        config.temperature = 1.0
        config.top_k = 0
        config.top_p = 1.0
        generated = model.generate(**encoded, generation_config=config)
        prompt_len = encoded["input_ids"].shape[1]
        completion = generated[:, prompt_len:]
        eos = self.processing_class.tokenizer.eos_token_id
        eos_mask = completion == eos
        # Include the first EOS; exclude every padding token after it.
        after_eos = eos_mask.cumsum(dim=1) - eos_mask.to(torch.long)
        mask = (after_eos == 0).to(encoded["attention_mask"].dtype)
        batch = dict(encoded)
        batch["input_ids"] = generated
        batch["attention_mask"] = torch.cat([encoded["attention_mask"], mask], dim=1)
        texts = self.processing_class.batch_decode(completion, skip_special_tokens=True)
        return batch, mask, texts

    def logps(self, model, batch, mask):
        # Keep the attention mask used at sampling, especially left padding.
        logits = model(**batch).logits[:, :-1, :]
        ids = batch["input_ids"][:, 1:]
        # Row-wise FP32 log-softmax limits the extra peak vocabulary allocation.
        # Slice away prompt positions before log-softmax: only generated tokens
        # enter this stage's objective.
        logits = logits[:, -mask.shape[1]:, :]
        ids = ids[:, -mask.shape[1]:]
        values = [row.float().log_softmax(-1).gather(1, tokens[:, None]).squeeze(1)
                  for row, tokens in zip(logits, ids)]
        return torch.stack(values)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("This rollout loss is training-only; use the inference entry for predictions")
        if len(inputs) != 1:
            raise ValueError("CerviThink currently requires per_device_train_batch_size=1")
        original = load_image(inputs[0]["image"])
        question = inputs[0]["problem"]
        g1 = int(self.num_generations_stage1)
        if g1 <= 0:
            raise ValueError("G1 must be positive")
        g2 = self.num_generations // g1
        if g2 <= 0 or g1 * g2 != self.num_generations:
            raise ValueError("Generation count must be a positive G1*G2")
        include_aux = bool(getattr(self, "cervithink_enable_auxiliary", True))
        was_training = model.training
        model.eval()
        try:
            with ground_trainer.unwrap_model_for_generation(model, self.accelerator) as unwrapped:
                with torch.no_grad():
                    grounding_batch, grounding_mask, groundings = sample(
                        self, unwrapped,
                        [grounding_messages(question, original.width, original.height) for _ in range(g1)],
                        [[original] for _ in range(g1)],
                    )
                    variants = []
                    for text in groundings:
                        box = policy_box(text, original.width, original.height)
                        variants.append(make_visual_variants(original, box) if box else None)
                    parent_indices = [i for i in range(g1) for _ in range(g2)]
                    messages, views, focus_views, background_views = [], [], [], []
                    for i in parent_indices:
                        variant = variants[i]
                        # Invalid actions get zero reward and zero answer loss mask.
                        # Dummy calls keep distributed generation schedules aligned.
                        focus = variant.focus if variant else original
                        transformed = variant.transform if variant else original
                        ignored = variant.ignore if variant else original
                        messages.append(answer_messages(question, original.width, original.height, groundings[i]))
                        views.append([original, original, focus, transformed, ignored])
                        focus_views.append([focus])
                        background_views.append([ignored])
                    answer_batch, answer_mask, answers = sample(self, unwrapped, messages, views)
                    valid = torch.tensor([variants[i] is not None for i in parent_indices],
                                         device=answer_mask.device)
                    answer_mask = answer_mask * valid[:, None]
                    if include_aux:
                        _, _, crops = sample(self, unwrapped,
                            [user_message(crop_consistency_prompt(question), 1) for _ in parent_indices],
                            focus_views, auxiliary=True)
                        _, _, backgrounds = sample(self, unwrapped,
                            [user_message(background_prompt(), 1) for _ in parent_indices],
                            background_views, auxiliary=True)
                    else:
                        crops = backgrounds = [""] * self.num_generations
                    # Rollout-policy likelihood snapshots, reused below.
                    old_grounding = logps(self, unwrapped, grounding_batch, grounding_mask).detach()
                    old_answer = logps(self, unwrapped, answer_batch, answer_mask).detach()
                    ref_grounding = ref_answer = None
                    if self.beta:
                        if self.ref_model is not None:
                            ref_grounding = logps(self, self.ref_model, grounding_batch, grounding_mask)
                            ref_answer = logps(self, self.ref_model, answer_batch, answer_mask)
                        else:
                            with unwrapped.disable_adapter():
                                ref_grounding = logps(self, unwrapped, grounding_batch, grounding_mask)
                                ref_answer = logps(self, unwrapped, answer_batch, answer_mask)

            valid_list = valid.tolist()
            completions = [[groundings[i], answer] if ok else [""]
                           for i, answer, ok in zip(parent_indices, answers, valid_list)]
            columns = {key: [value] * self.num_generations for key, value in inputs[0].items()
                       if key not in {"prompt", "completion"}}
            columns.update(
                crop_answer=[text if ok else "" for text, ok in zip(crops, valid_list)],
                background_answer=[text if ok else "" for text, ok in zip(backgrounds, valid_list)],
            )
            reward_columns = []
            for reward in self.reward_funcs:
                if isinstance(reward, ground_trainer.PreTrainedModel):
                    raise ValueError("This adapter requires callable CerviThink rewards")
                values = reward(prompts=[inputs[0]["prompt"]] * self.num_generations,
                                completions=completions, **columns)
                if len(values) != self.num_generations:
                    raise ValueError("Reward function returned the wrong rollout count")
                reward_columns.append(torch.tensor(values, device=answer_mask.device, dtype=torch.float32) * valid)
            rewards_per_func = torch.stack(reward_columns, dim=1)
            rewards = rewards_per_func.sum(1)
            std = rewards.std(unbiased=False)
            advantages = (rewards - rewards.mean()) / (std + 1e-4)
            current_grounding = logps(self, model, grounding_batch, grounding_mask)
            current_answer = logps(self, model, answer_batch, answer_mask)
            loss = joint_clipped_loss(
                current_grounding, old_grounding, grounding_mask,
                current_answer, old_answer, answer_mask,
                parent_indices, advantages, epsilon=self.cervithink_clip_epsilon,
                beta=self.beta, ref_grounding=ref_grounding, ref_answer=ref_answer,
            )
            for i, reward in enumerate(self.reward_funcs):
                self._metrics[f"rewards/{reward.__name__}"].append(rewards_per_func[:, i].mean().item())
            self._metrics["reward"].append(rewards.mean().item())
            self._metrics["reward_std"].append(std.item())
            self._metrics["cervithink/valid_grounding"].append(valid.float().mean().item())
            self._metrics["cervithink/auxiliary_enabled"].append(float(include_aux))
            return loss
        finally:
            model.train(was_training)

    trainer_cls.compute_loss = compute_loss
    trainer_cls._cervithink_dvhr_installed = True
