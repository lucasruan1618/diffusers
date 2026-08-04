# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import PIL.Image
import torch
from transformers import Qwen2Tokenizer, Qwen3VLConfig, Qwen3VLModel, Qwen3VLTextConfig, Qwen3VLVisionConfig

from diffusers import (
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
    Ideogram4AutoBlocks,
    Ideogram4ModularPipeline,
    Ideogram4Pipeline,
    Ideogram4Transformer2DModel,
)
from diffusers.models.transformers.transformer_ideogram4 import LLM_TOKEN_INDICATOR

from ...testing_utils import enable_full_determinism
from ..test_modular_pipelines_common import ModularPipelineTesterMixin


enable_full_determinism()


IDEOGRAM4_WORKFLOWS = {
    "text2image": [
        ("prompt_upsample", "Ideogram4PromptUpsampleStep"),
        ("text_encoder", "Ideogram4TextEncoderStep"),
        ("denoise.input", "Ideogram4TextInputsStep"),
        ("denoise.prepare_latents", "Ideogram4PrepareLatentsStep"),
        ("denoise.set_timesteps", "Ideogram4SetTimestepsStep"),
        ("denoise.prepare_additional_inputs", "Ideogram4PrepareAdditionalInputsStep"),
        ("denoise.denoise", "Ideogram4DenoiseStep"),
        ("denoise.after_denoise", "Ideogram4AfterDenoiseStep"),
        ("decode", "Ideogram4DecodeStep"),
    ],
    "image2image": [
        ("vae_encoder.preprocess", "Ideogram4ProcessImageInputStep"),
        ("vae_encoder.encode", "Ideogram4VaeEncoderStep"),
        ("prompt_upsample", "Ideogram4PromptUpsampleStep"),
        ("text_encoder", "Ideogram4TextEncoderStep"),
        ("denoise.text_inputs", "Ideogram4TextInputsStep"),
        ("denoise.image_inputs", "Ideogram4ImageInputsStep"),
        ("denoise.prepare_latents", "Ideogram4PrepareLatentsStep"),
        ("denoise.set_timesteps", "Ideogram4SetTimestepsStep"),
        ("denoise.apply_strength", "Ideogram4ApplyStrengthStep"),
        ("denoise.prepare_image_latents", "Ideogram4PrepareLatentsWithStrengthStep"),
        ("denoise.prepare_additional_inputs", "Ideogram4PrepareAdditionalInputsStep"),
        ("denoise.denoise", "Ideogram4DenoiseStep"),
        ("denoise.after_denoise", "Ideogram4AfterDenoiseStep"),
        ("decode", "Ideogram4DecodeStep"),
    ],
    "inpainting": [
        ("vae_encoder.preprocess", "Ideogram4InpaintProcessImagesInputStep"),
        ("vae_encoder.encode", "Ideogram4VaeEncoderStep"),
        ("prompt_upsample", "Ideogram4PromptUpsampleStep"),
        ("text_encoder", "Ideogram4TextEncoderStep"),
        ("denoise.text_inputs", "Ideogram4TextInputsStep"),
        ("denoise.image_inputs", "Ideogram4ImageInputsStep"),
        ("denoise.mask_inputs", "Ideogram4MaskInputsStep"),
        ("denoise.prepare_latents", "Ideogram4PrepareLatentsStep"),
        ("denoise.set_timesteps", "Ideogram4SetTimestepsStep"),
        ("denoise.apply_strength", "Ideogram4ApplyStrengthStep"),
        ("denoise.prepare_image_latents", "Ideogram4PrepareLatentsWithStrengthStep"),
        ("denoise.prepare_mask_latents", "Ideogram4PrepareMaskLatentsStep"),
        ("denoise.prepare_additional_inputs", "Ideogram4PrepareAdditionalInputsStep"),
        ("denoise.denoise", "Ideogram4InpaintDenoiseStep"),
        ("denoise.after_denoise", "Ideogram4AfterDenoiseStep"),
        ("decode", "Ideogram4InpaintDecodeStep"),
    ],
}


def get_dummy_components():
    torch.manual_seed(0)
    transformer = Ideogram4Transformer2DModel(
        in_channels=16,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        intermediate_size=32,
        adaln_dim=8,
        llm_features_dim=52,
        rope_theta=10_000,
        mrope_section=(2, 1, 1),
    ).eval()

    torch.manual_seed(0)
    unconditional_transformer = Ideogram4Transformer2DModel(
        in_channels=16,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        intermediate_size=32,
        adaln_dim=8,
        llm_features_dim=52,
        rope_theta=10_000,
        mrope_section=(2, 1, 1),
    ).eval()

    torch.manual_seed(0)
    vae = AutoencoderKLFlux2(
        block_out_channels=(8, 8, 8, 8),
        decoder_block_out_channels=(8, 8, 8, 8),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=4,
        sample_size=32,
        mid_block_add_attention=False,
        patch_size=(2, 2),
    ).eval()

    text_config = Qwen3VLTextConfig(
        vocab_size=152064,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=36,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=128,
        use_cache=False,
    )
    vision_config = Qwen3VLVisionConfig(
        depth=1,
        hidden_size=4,
        intermediate_size=8,
        num_heads=1,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=4,
        deepstack_visual_indexes=(),
    )
    torch.manual_seed(0)
    text_encoder = Qwen3VLModel(Qwen3VLConfig(text_config=text_config, vision_config=vision_config)).eval()
    tokenizer = Qwen2Tokenizer.from_pretrained("hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration")
    scheduler = FlowMatchEulerDiscreteScheduler()

    return {
        "transformer": transformer,
        "unconditional_transformer": unconditional_transformer,
        "vae": vae,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "scheduler": scheduler,
    }


def get_dummy_image(seed=0):
    image = np.random.default_rng(seed).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    return PIL.Image.fromarray(image)


class TestIdeogram4ModularPipelineFast(ModularPipelineTesterMixin):
    pipeline_class = Ideogram4ModularPipeline
    pipeline_blocks_class = Ideogram4AutoBlocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-ideogram4-modular-pipe"

    params = frozenset(["prompt", "height", "width", "image", "mask_image"])
    batch_params = frozenset(["prompt", "image", "mask_image"])
    expected_workflow_blocks = IDEOGRAM4_WORKFLOWS

    def get_pipeline(self, components_manager=None, dtype=torch.float32):
        pipe = self.pipeline_blocks_class().init_pipeline(components_manager=components_manager)
        pipe.update_components(**get_dummy_components())
        pipe.to(dtype=dtype)
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def get_dummy_inputs(self, seed=0):
        return {
            "prompt": "cat wizard",
            "generator": self.get_generator(seed),
            "num_inference_steps": 2,
            "guidance_schedule": [1.0, 1.0],
            "height": 32,
            "width": 32,
            "max_sequence_length": 32,
            "output_type": "pt",
        }

    def test_img2img_and_inpaint(self):
        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        inputs.update({"image": get_dummy_image(), "strength": 1.0})

        image = pipe(**inputs, output="images")
        assert image.shape == (1, 3, 32, 32)
        assert not torch.isnan(image).any()

        inputs["generator"] = self.get_generator(0)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 255
        inputs["mask_image"] = PIL.Image.fromarray(mask)
        inputs["padding_mask_crop"] = 2
        inputs["output_type"] = "pil"
        image = pipe(**inputs, output="images")
        assert len(image) == 1
        assert image[0].size == (32, 32)
        assert np.isfinite(np.asarray(image[0])).all()

    def test_prompt_padding_is_compacted(self):
        prompts = ["cat", "cat wizard"]
        max_sequence_length = 32

        standard_pipe = Ideogram4Pipeline(**get_dummy_components())
        prompt_embeds, position_ids, segment_ids, indicator = standard_pipe.encode_prompt(
            prompt=prompts,
            grid_h=2,
            grid_w=2,
            max_sequence_length=max_sequence_length,
            device=torch.device("cpu"),
        )
        text_lengths = (indicator == LLM_TOKEN_INDICATOR).sum(dim=1)
        assert prompt_embeds.shape[1] == text_lengths.max().item() + 4
        assert prompt_embeds.shape[1] < max_sequence_length + 4

        compact_inputs = standard_pipe.encode_prompt(
            prompt=prompts[0],
            grid_h=2,
            grid_w=2,
            max_sequence_length=max_sequence_length,
            device=torch.device("cpu"),
        )
        image_latents = torch.randn(1, 4, standard_pipe.transformer.config.in_channels)
        image_outputs = []
        for conditioning in (compact_inputs, (prompt_embeds[:1], position_ids[:1], segment_ids[:1], indicator[:1])):
            embeddings, positions, segments, indicators = conditioning
            num_text_tokens = embeddings.shape[1] - 4
            hidden_states = torch.cat([torch.zeros(1, num_text_tokens, image_latents.shape[-1]), image_latents], dim=1)
            output = standard_pipe.transformer(
                hidden_states=hidden_states,
                timestep=torch.zeros(1),
                encoder_hidden_states=embeddings,
                position_ids=positions,
                segment_ids=segments,
                indicator=indicators,
                return_dict=False,
            )[0]
            image_outputs.append(output[:, -4:])
        torch.testing.assert_close(*image_outputs)

        pipe = self.get_pipeline()
        inputs = self.get_dummy_inputs()
        inputs.update({"prompt": prompts, "max_sequence_length": max_sequence_length})
        outputs = pipe(**inputs, output=["text_features", "text_lengths"])
        assert outputs["text_features"].shape[1] == max(outputs["text_lengths"])
        assert outputs["text_features"].shape[1] < max_sequence_length
