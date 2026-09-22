"""CPU gradient and full-adapter tests with a tiny synthetic model (no weights)."""
from collections import defaultdict
from contextlib import nullcontext
import copy
import importlib.util
import json
import math
from types import SimpleNamespace
from types import MethodType
import unittest
from unittest.mock import patch

from PIL import Image

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
from cervithink.grpo_objective import joint_clipped_loss
from cervithink.groundr1_dvhr import install_cervithink_dvhr
from scripts.train_cervithink_grpo import (
    cervithink_accuracy_reward, cervithink_format_reward,
    cervithink_consistency_reward, cervithink_background_reward,
)


@unittest.skipUnless(TORCH_AVAILABLE, "CPU torch required for gradient checks")
class ObjectiveTests(unittest.TestCase):
    def test_grounding_and_answer_receive_gradient(self):
        ground = torch.tensor([[-1., -2.]], requires_grad=True)
        answer = torch.tensor([[-1., -3.]], requires_grad=True)
        loss = joint_clipped_loss(ground, ground.detach().clone(), torch.ones_like(ground),
                                 answer, answer.detach().clone(), torch.ones_like(answer),
                                 [0], torch.tensor([1.]))
        loss.backward()
        self.assertGreater(ground.grad.abs().sum().item(), 0)
        self.assertGreater(answer.grad.abs().sum().item(), 0)

    def test_clip_limits_gradient_for_positive_and_negative_advantage(self):
        for advantage, ratio in [(1., 2.), (-1., .5)]:
            with self.subTest(advantage=advantage):
                ground = torch.tensor([[math.log(ratio)]], requires_grad=True)
                answer = torch.tensor([[math.log(ratio)]], requires_grad=True)
                old = torch.zeros_like(ground, requires_grad=True)
                loss = joint_clipped_loss(ground, old, torch.ones_like(ground),
                                         answer, old, torch.ones_like(answer),
                                         [0], torch.tensor([advantage]), epsilon=.2)
                loss.backward()
                self.assertAlmostEqual(ground.grad.item(), 0.)
                self.assertAlmostEqual(answer.grad.item(), 0.)
                self.assertIsNone(old.grad)

    def test_masked_tokens_have_no_gradient(self):
        ground = torch.tensor([[-1., -1.]], requires_grad=True)
        answer = torch.tensor([[-1., -1.]], requires_grad=True)
        loss = joint_clipped_loss(ground, ground.detach(), torch.tensor([[1., 0.]]),
                                 answer, answer.detach(), torch.tensor([[1., 0.]]),
                                 [0], torch.tensor([1.]))
        loss.backward()
        self.assertEqual(ground.grad[0, 1].item(), 0.)
        self.assertEqual(answer.grad[0, 1].item(), 0.)

    def test_no_kl_reference_required_at_zero_beta(self):
        value = torch.zeros((1, 1), requires_grad=True)
        loss = joint_clipped_loss(value, value.detach(), torch.ones_like(value),
                                 value, value.detach(), torch.ones_like(value),
                                 [0], torch.ones(1), beta=0)
        self.assertTrue(torch.isfinite(loss))
        with self.assertRaisesRegex(ValueError, "reference"):
            joint_clipped_loss(value, value.detach(), torch.ones_like(value),
                              value, value.detach(), torch.ones_like(value),
                              [0], torch.ones(1), beta=.01)


@unittest.skipUnless(TORCH_AVAILABLE, "CPU torch required for synthetic adapter test")
class AdapterTests(unittest.TestCase):
    def make_trainer(self, rewards=None, invalid=False, auxiliary=True):
        class Processor:
            tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0)
            def __init__(self):
                self.encodings = []
            def apply_chat_template(self, message, **kwargs):
                return json.dumps(message)
            def __call__(self, text, images, **kwargs):
                self.encodings.append((text, images))
                last = json.loads(text[0])[-1]["content"][-1]["text"]
                code = (10 if "Return exactly one region" in last else
                        12 if "focused cervical cytology crop" in last else
                        13 if "remaining visible background" in last else 11)
                # Left-pad one row, to make loss/padding equality observable.
                ids = torch.tensor([[0, code, 4] if i == 0 else [5, code, 4] for i in range(len(text))])
                return {"input_ids": ids, "attention_mask": (ids != 0).long()}
            def batch_decode(self, ids, **kwargs):
                mapping = {
                    21: "<think>region one</think><box>[8,8,40,40]</box>",
                    22: "<think>region two</think><box>[16,16,48,48]</box>",
                    23: "invalid grounding",
                    31: "<rethink>lesion</rethink><answer>HSIL</answer>",
                    32: "<rethink>normal</rethink><answer>Normal</answer>",
                    41: "<think>crop</think><answer>HSIL</answer>",
                    42: "<think>crop</think><answer>Normal</answer>",
                    51: "<think>background</think><answer>Normal</answer>",
                }
                return [mapping[int(row[0])] for row in ids]

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.values = torch.nn.Parameter(torch.zeros(64))
                self.generations = []
                self.forwards = []
            def generate(self, input_ids, attention_mask, generation_config):
                self.generations.append((input_ids.clone(), copy.deepcopy(generation_config)))
                code = int(input_ids[0, 1])
                n = len(input_ids)
                if code == 10:
                    tokens = [23 if invalid else 21+i%2 for i in range(n)]
                elif code == 11:
                    tokens = [31 if i < n//2 else 32 for i in range(n)]
                elif code == 12:
                    tokens = [41 if i < n//2 else 42 for i in range(n)]
                else:
                    tokens = [51 for i in range(n)]
                tail = torch.tensor([[t, 2, 0] for t in tokens])
                return torch.cat([input_ids, tail], 1)
            def forward(self, input_ids, attention_mask):
                self.forwards.append((torch.is_grad_enabled(), attention_mask.clone()))
                return SimpleNamespace(logits=self.values.expand(*input_ids.shape, 64))

        class Base:
            def _prepare_inputs(self, inputs):
                return inputs
        class Trainer(Base):
            pass
        gt = SimpleNamespace(
            Qwen2VLGRPOTrainer=Trainer, torch=torch,
            unwrap_model_for_generation=lambda model, acc: nullcontext(model),
            PreTrainedModel=type("RewardModel", (), {}),
        )
        install_cervithink_dvhr(gt)
        trainer = Trainer()
        trainer.processing_class = Processor()
        trainer.generation_config = SimpleNamespace(num_return_sequences=4, max_new_tokens=3)
        trainer.num_generations = 4
        trainer.num_generations_stage1 = 2
        trainer.cervithink_enable_auxiliary = auxiliary
        trainer.cervithink_clip_epsilon = .2
        trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
        trainer._metrics = defaultdict(list)
        trainer.beta = 0
        # Poison reference would error if the no-KL path tried to call it.
        trainer.ref_model = object()
        trainer.reward_funcs = rewards or [
            cervithink_accuracy_reward, cervithink_format_reward,
            cervithink_consistency_reward, cervithink_background_reward,
        ]
        data = [{"image": Image.new("RGB", (64, 64)), "problem": "Classify", "label": "HSIL",
                 "prompt": [{"role": "user", "content": "Classify"}]}]
        return trainer, Model(), data

    def test_complete_adapter_loss_backward_and_shared_views(self):
        trainer, model, data = self.make_trainer()
        loss = trainer.compute_loss(model, data)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(model.values.grad.abs().sum().item(), 0.)
        self.assertNotEqual(model.values.grad[21].item(), 0.)
        self.assertEqual([len(ids) for ids, _ in model.generations], [2, 4, 4, 4])
        self.assertEqual([enabled for enabled, _ in model.forwards], [False, False, True, True])
        for old, current in zip(model.forwards[:2], model.forwards[2:]):
            self.assertTrue(torch.equal(old[1], current[1]))
            self.assertTrue(torch.all(current[1][:, -1] == 0))
        # Stage 2 has 5 image entries per answer. Siblings share the same variants.
        images = trainer.processing_class.encodings[1][1]
        self.assertEqual(len(images), 20)
        self.assertIs(images[2], images[7])
        self.assertIs(images[3], images[8])
        self.assertTrue(model.training)

    def test_reward_ablation_keeps_generation_protocol(self):
        full, full_model, data = self.make_trainer()
        base, base_model, _ = self.make_trainer([cervithink_accuracy_reward, cervithink_format_reward])
        full.compute_loss(full_model, data)
        base.compute_loss(base_model, data)
        for a, b in zip(full_model.generations, base_model.generations):
            self.assertTrue(torch.equal(a[0], b[0]))
            self.assertEqual(a[1].__dict__, b[1].__dict__)
        self.assertEqual(len(full_model.generations), len(base_model.generations))

    def test_all_invalid_grounding_stays_finite(self):
        trainer, model, data = self.make_trainer(invalid=True)
        loss = trainer.compute_loss(model, data)
        loss.backward()
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(trainer._metrics["cervithink/valid_grounding"], [0.])

    def test_disabling_auxiliary_keeps_main_sampler(self):
        trainer, model, data = self.make_trainer(
            [cervithink_accuracy_reward, cervithink_format_reward], auxiliary=False)
        loss = trainer.compute_loss(model, data)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(len(model.generations), 2)

    def test_batch_size_guard(self):
        trainer, model, data = self.make_trainer()
        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            trainer.compute_loss(model, data+data)

    @unittest.skipUnless(importlib.util.find_spec("transformers") is not None, "Transformers required")
    def test_tiny_real_qwen_multimodal_forward_backward(self):
        """Actual Qwen image+text forward; generated text is scripted, no downloads."""
        from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
        trainer, fake_model, data = self.make_trainer()
        processor_type = type(trainer.processing_class)

        class TinyProcessor(processor_type):
            def __call__(self, text, images, **kwargs):
                base = super().__call__(text, images, **kwargs)
                image_counts = [
                    sum(c["type"] == "image" for m in json.loads(t) for c in m["content"])
                    for t in text
                ]
                self_outer = []
                for row, count in zip(base["input_ids"], image_counts):
                    self_outer.append(torch.cat([row, torch.tensor([61, 60, 62]*count)]))
                ids = torch.stack(self_outer)
                self.assert_image_count = sum(image_counts)
                assert self.assert_image_count == len(images)
                return {
                    "input_ids": ids, "attention_mask": (ids != 0).long(),
                    "pixel_values": torch.zeros(len(images)*4, 3*2*14*14),
                    "image_grid_thw": torch.tensor([[1, 2, 2]]*len(images)),
                }

        trainer.processing_class = TinyProcessor()
        config = Qwen2_5_VLConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=256, eos_token_id=2, pad_token_id=0,
            image_token_id=60, vision_start_token_id=61, vision_end_token_id=62, video_token_id=63,
            rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
            vision_config=dict(
                depth=1, hidden_size=32, intermediate_size=64, num_heads=4, out_hidden_size=32,
                patch_size=14, spatial_merge_size=2, temporal_patch_size=2,
                window_size=28, fullatt_block_indexes=[0],
            ),
        )
        config._attn_implementation = "eager"
        model = Qwen2_5_VLForConditionalGeneration(config)
        model.generations = []
        scripted_generate = type(fake_model).generate
        # Preserve the actual Qwen forward, while using known output tokens.
        def generate(self, input_ids, attention_mask, generation_config, **kwargs):
            return scripted_generate(self, input_ids, attention_mask, generation_config)
        model.generate = MethodType(generate, model)
        loss = trainer.compute_loss(model, data)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(model.lm_head.weight.grad.abs().sum().item(), 0.)


if __name__ == "__main__":
    unittest.main()
