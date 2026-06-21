"""CerviThink DVHR adapter for the Ground-R1 GRPO trainer."""

from __future__ import annotations

import copy
import gc
import re
from typing import Any

from PIL import Image

from .parsing import extract_box
from .prompts import answer_prompt, background_prompt, crop_consistency_prompt
from .visual_ops import VisualVariants, make_visual_variants


def install_cervithink_dvhr(ground_trainer) -> None:
    """Patch Ground-R1's trainer to provide crop/background answers to rewards."""

    trainer_cls = ground_trainer.Qwen2VLGRPOTrainer
    if getattr(trainer_cls, "_cervithink_dvhr_installed", False):
        return

    torch = ground_trainer.torch
    unwrap_model_for_generation = ground_trainer.unwrap_model_for_generation
    is_conversational = ground_trainer.is_conversational
    apply_chat_template = ground_trainer.apply_chat_template
    process_vision_info = ground_trainer.process_vision_info
    PreTrainedModel = ground_trainer.PreTrainedModel

    def _prepare_with_trainer(self, inputs):
        return super(trainer_cls, self)._prepare_inputs(inputs)

    def _cervithink_aux_config(self):
        config = copy.deepcopy(self.generation_config)
        config.num_return_sequences = 1
        config.do_sample = bool(getattr(self, "cervithink_aux_do_sample", False))
        return config

    def _cervithink_make_visual_variants(
        self,
        output_text: str,
        origin_image: Image.Image,
        input_width: int,
        input_height: int,
        width: int,
        height: int,
    ) -> VisualVariants | None:
        bbox = extract_box(output_text)
        if bbox is None:
            return None
        mapped = ground_trainer.bbox_adjust(
            list(bbox),
            input_width,
            input_height,
            width,
            height,
            min_size=28,
        )
        if not mapped:
            return None
        return make_visual_variants(origin_image, tuple(mapped))

    def _cervithink_prepare_answer_stage(
        self,
        origin_prompt,
        origin_problem: str,
        combined_images: list[Image.Image],
        grounding_text: str,
        variants: VisualVariants,
    ):
        answer_entry = {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": answer_prompt(origin_problem)},
            ],
        }
        origin_prompt.extend(
            [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": str(grounding_text)}],
                },
                answer_entry,
            ]
        )
        combined_images.extend([combined_images[0], variants.focus, variants.transform, variants.ignore])
        return origin_prompt, combined_images

    def _cervithink_generate_auxiliary_answers(
        self,
        model,
        images: list[Image.Image | None],
        prompt_text: str,
    ) -> list[str]:
        answers = ["" for _ in images]
        tasks = [(idx, image) for idx, image in enumerate(images) if image is not None]
        if not tasks:
            return answers

        batch_size = int(getattr(self, "cervithink_aux_batch_size", 4))
        generation_config = self._cervithink_aux_config()
        for start in range(0, len(tasks), batch_size):
            chunk = tasks[start : start + batch_size]
            messages = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]
                for _idx, _image in chunk
            ]
            texts = [
                self.processing_class.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True,
                    add_vision_id=True,
                )
                for message in messages
            ]
            batch_images = [image for _idx, image in chunk]
            aux_inputs = self.processing_class(
                text=texts,
                images=batch_images,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            aux_inputs = _prepare_with_trainer(self, aux_inputs)
            outputs = model.generate(**aux_inputs, generation_config=generation_config)
            generated = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(aux_inputs.input_ids, outputs)
            ]
            decoded = self.processing_class.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for (idx, _image), text in zip(chunk, decoded):
                answers[idx] = text
        return answers

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        origin_problem = inputs[0]["problem"]
        prompts = [x["prompt"] for x in inputs]
        prompts_text = self.processing_class.apply_chat_template(
            inputs[0]["prompt"],
            tokenize=False,
            add_generation_prompt=True,
            add_vision_id=True,
        )
        if "image" in inputs[0]:
            images = []
            for cur_idx, cur_input in enumerate(inputs):
                copy_input = copy.deepcopy(cur_input)
                copy_input["prompt"][0]["content"][0]["image"] = inputs[cur_idx]["image"]
                images.append(process_vision_info(copy_input["prompt"])[0])
            images = images[0]

        prompt_inputs = self.processing_class(
            text=prompts_text,
            images=images if "image" in inputs[0] else None,
            videos=None,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = _prepare_with_trainer(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        input_height = prompt_inputs["image_grid_thw"][0][1] * 14
        input_width = prompt_inputs["image_grid_thw"][0][2] * 14
        width, height = images[0].size

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            num_generations = self.generation_config.num_return_sequences
            temp_generation_config = copy.deepcopy(self.generation_config)
            temp_generation_config.num_return_sequences = self.num_generations_stage1

            all_completions = [None] * num_generations
            crop_bbox_to_cal_iou_stage1: list[Any] = []
            prompt_for_generation = [copy.deepcopy(inputs[0]["prompt"]) for _ in range(num_generations)]
            all_images_final_list = [copy.deepcopy(images) for _ in range(num_generations)]
            aux_focus_images: list[Image.Image | None] = [None] * num_generations
            aux_background_images: list[Image.Image | None] = [None] * num_generations

            with torch.no_grad():
                completion_stage1 = unwrapped_model.generate(
                    **prompt_inputs,
                    generation_config=temp_generation_config,
                )

                input_length = prompt_inputs["input_ids"].shape[1]
                generated_ids_list = [out_ids[input_length:] for out_ids in completion_stage1]
                output_text_stage1_list = self.processing_class.batch_decode(
                    generated_ids_list,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                del generated_ids_list
                gc.collect()
                torch.cuda.empty_cache()

                original_list = output_text_stage1_list
                repeat_factor = num_generations // len(original_list)
                output_text_stage1_list = (original_list * repeat_factor)[:num_generations]
                completion_stage1 = list(completion_stage1)
                completion_stage1 = (completion_stage1 * repeat_factor)[:num_generations]
                iteration = 0

                while self.accelerator.reduce(
                    torch.tensor(
                        any(item is None for item in all_completions),
                        device=self.accelerator.device,
                    ).int()
                ).item() > 0:
                    messages_stage2_list = []
                    all_images_stage2_list = []
                    for idx in range(len(output_text_stage1_list)):
                        if all_completions[idx] is not None:
                            continue
                        output_text = output_text_stage1_list[idx]
                        if "<answer>" in output_text or iteration == 4:
                            all_completions[idx] = completion_stage1[idx].unsqueeze(0)
                            continue

                        variants = self._cervithink_make_visual_variants(
                            output_text,
                            images[0],
                            input_width,
                            input_height,
                            width,
                            height,
                        )
                        if variants is None:
                            all_completions[idx] = completion_stage1[idx].unsqueeze(0)
                            continue
                        aux_focus_images[idx] = variants.focus
                        aux_background_images[idx] = variants.ignore

                        prompt_for_generation[idx], all_images_final_list[idx] = (
                            self._cervithink_prepare_answer_stage(
                                prompt_for_generation[idx],
                                origin_problem,
                                all_images_final_list[idx],
                                output_text,
                                variants,
                            )
                        )
                        messages_stage2_list.append(prompt_for_generation[idx])
                        all_images_stage2_list.extend(all_images_final_list[idx])

                    if messages_stage2_list:
                        prompt_stage2_inputs = self._generate_for_stage2_batch(
                            prompt_for_generation,
                            all_images_final_list,
                        )
                        temp_stage2_config = copy.deepcopy(temp_generation_config)
                        temp_stage2_config.num_return_sequences = 1
                        completion_stage1 = unwrapped_model.generate(
                            **prompt_stage2_inputs,
                            generation_config=temp_stage2_config,
                        )

                        generated_ids_trimmed = [
                            out_ids[len(in_ids) :]
                            for in_ids, out_ids in zip(
                                prompt_stage2_inputs.input_ids,
                                completion_stage1,
                            )
                        ]
                        output_text_stage1_list = self.processing_class.batch_decode(
                            generated_ids_trimmed,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )
                    else:
                        warmup_config = copy.deepcopy(temp_generation_config)
                        warmup_config.num_return_sequences = 1
                        warmup_input = {
                            "input_ids": torch.tensor(
                                [[self.tokenizer.pad_token_id]],
                                device=self.accelerator.device,
                            ),
                            "attention_mask": torch.tensor([[1]], device=self.accelerator.device),
                        }
                        _ = unwrapped_model.generate(**warmup_input, generation_config=warmup_config)
                    iteration += 1

                prompt_stage2_inputs = self._generate_for_stage2_batch(
                    prompt_for_generation,
                    all_images_final_list,
                )

                crop_answers = self._cervithink_generate_auxiliary_answers(
                    unwrapped_model,
                    aux_focus_images,
                    crop_consistency_prompt(origin_problem),
                )
                background_answers = self._cervithink_generate_auxiliary_answers(
                    unwrapped_model,
                    aux_background_images,
                    background_prompt(),
                )

                del output_text_stage1_list, completion_stage1, prompt_for_generation, all_images_final_list
                gc.collect()
                torch.cuda.empty_cache()

            max_length = max(completion.size(1) for completion in all_completions)
            padded_completions = []
            indices_list = []

            for completion in all_completions:
                seq = completion[0]
                start_pattern = torch.tensor([151644, 77091, 198], device=seq.device)
                end_token = torch.tensor(151645, device=seq.device)
                sample_indices = []
                start_indices = (
                    (seq.unfold(0, len(start_pattern), 1) == start_pattern)
                    .all(dim=1)
                    .nonzero(as_tuple=True)[0]
                )
                for start_index in start_indices:
                    start_index = start_index.item() + len(start_pattern)
                    end_indices = (seq[start_index:] == end_token).nonzero(as_tuple=True)[0]
                    end_index = start_index + (
                        end_indices[0].item() + 1 if end_indices.numel() > 0 else len(seq) - start_index
                    )
                    sample_indices.append(list(range(start_index, end_index)))
                indices_list.append(sample_indices)

                if completion.size(1) < max_length:
                    padding = torch.full(
                        (completion.size(0), max_length - completion.size(1)),
                        self.processing_class.tokenizer.pad_token_id,
                        dtype=completion.dtype,
                        device=completion.device,
                    )
                    padded_completion = torch.cat([completion, padding], dim=1)
                else:
                    padded_completion = completion
                padded_completions.append(padded_completion)

            prompt_completion_ids = torch.cat(padded_completions, dim=0)

        completion_ids = prompt_completion_ids[:]
        device = self.accelerator.device
        batch_size, seq_length = completion_ids.size()
        completion_mask = torch.zeros((batch_size, seq_length), dtype=torch.int, device=device)
        for i, indices in enumerate(indices_list):
            for index_range in indices:
                completion_mask[i, index_range] = 1

        prompt_stage2_inputs.pop("input_ids")
        prompt_stage2_inputs.pop("attention_mask")
        per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_stage2_inputs)
        completion_mask = completion_mask[:, 1:]

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model,
                    prompt_completion_ids,
                    **prompt_stage2_inputs,
                )
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        model,
                        prompt_completion_ids,
                        **prompt_stage2_inputs,
                    )

        x_clamped = torch.clamp(ref_per_token_logps - per_token_logps, min=-10, max=10)
        per_token_kl = torch.exp(x_clamped) - x_clamped - 1

        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [
                re.findall(r"(?<=\nassistant\n)(.*?)(?=\nuser\n|\Z)", completion, re.S)
                for completion in completions
            ]

        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    padding_side="right",
                    add_special_tokens=False,
                )
                reward_inputs = _prepare_with_trainer(self, reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                reward_kwargs["crop_answer"] = crop_answers
                reward_kwargs["background_answer"] = background_answers
                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    **reward_kwargs,
                )
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        scores_per_func = torch.zeros(len(prompts), len(self.score_funcs), device=device)
        for i, (score_func, _score_processing_class) in enumerate(
            zip(self.score_funcs, self.score_processing_classes)
        ):
            score_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
            for key in score_kwargs:
                for example in inputs:
                    score_kwargs[key].extend([example[key]] * self.num_generations)
            score_kwargs["crop_answer"] = crop_answers
            score_kwargs["background_answer"] = background_answers
            output_score_func = score_func(prompts=prompts, completions=completions, **score_kwargs)
            scores_per_func[:, i] = torch.tensor(output_score_func, dtype=torch.float32, device=device)

        rewards = rewards_per_func.sum(dim=1)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        token_counts = completion_mask.sum(dim=1).clamp_min(1)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / token_counts).mean()

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        self._metrics["cervithink/crop_aux_coverage"].append(
            sum(bool(answer) for answer in crop_answers) / max(1, len(crop_answers))
        )
        self._metrics["cervithink/background_aux_coverage"].append(
            sum(bool(answer) for answer in background_answers) / max(1, len(background_answers))
        )

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        score_per_func = self.accelerator.gather_for_metrics(scores_per_func).mean(0)
        for i, score_func in enumerate(self.score_funcs):
            score_func_name = score_func.__name__
            self._metrics[f"scores/{score_func_name}"].append(score_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / token_counts).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        return loss

    trainer_cls._cervithink_aux_config = _cervithink_aux_config
    trainer_cls._cervithink_make_visual_variants = _cervithink_make_visual_variants
    trainer_cls._cervithink_prepare_answer_stage = _cervithink_prepare_answer_stage
    trainer_cls._cervithink_generate_auxiliary_answers = _cervithink_generate_auxiliary_answers
    trainer_cls.compute_loss = compute_loss
    trainer_cls._cervithink_dvhr_installed = True
