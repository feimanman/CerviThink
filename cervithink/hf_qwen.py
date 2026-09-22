"""Qwen2.5-VL inference wrapper."""

from __future__ import annotations

from typing import Any

from PIL import Image


class HFQwenVLModel:
    def __init__(
        self,
        model_name_or_path: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        attn_implementation: str | None = None,
    ):
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except Exception as exc:
            raise ImportError(
                "HFQwenVLModel requires torch and transformers."
            ) from exc

        kwargs: dict[str, Any] = {"device_map": device_map}
        if torch_dtype != "auto":
            kwargs["torch_dtype"] = getattr(torch, torch_dtype)
        else:
            kwargs["torch_dtype"] = "auto"
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation

        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path, min_pixels=3136, max_pixels=401408,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name_or_path, **kwargs)
        self.model.eval()

    def generate(
        self,
        messages: list[dict[str, Any]],
        images: list[Image.Image],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=True,
        )
        inputs = self.processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_k=0,
            top_p=1.0,
        )
        trimmed = output_ids[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
