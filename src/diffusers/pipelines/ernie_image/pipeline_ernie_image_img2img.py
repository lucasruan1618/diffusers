# Copyright 2025 Baidu ERNIE-Image Team and The HuggingFace Team. All rights reserved.
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

from typing import Callable, List, Optional, Union

import torch

from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...utils.torch_utils import randn_tensor
from .pipeline_ernie_image import ErnieImagePipeline
from .pipeline_output import ErnieImagePipelineOutput


def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class ErnieImageImg2ImgPipeline(ErnieImagePipeline):
    """
    Pipeline for image-to-image generation with ERNIE-Image.
    """

    def _get_image_processor(self) -> VaeImageProcessor:
        if not hasattr(self, "image_processor"):
            self.image_processor = VaeImageProcessor(
                vae_scale_factor=self.vae_scale_factor,
                vae_latent_channels=self.vae.config.latent_channels,
                do_convert_rgb=True,
            )
        return self.image_processor

    def _encode_vae_image(
        self, image: torch.Tensor, generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None
    ) -> torch.Tensor:
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i]) for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        image_latents = self._patchify_latents(image_latents)

        bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(device=image_latents.device, dtype=image_latents.dtype)
        bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + 1e-5).to(
            device=image_latents.device, dtype=image_latents.dtype
        )
        image_latents = (image_latents - bn_mean) / bn_std

        return image_latents

    def get_timesteps(self, num_inference_steps: int, strength: float, device: torch.device):
        init_timestep = min(num_inference_steps * strength, num_inference_steps)
        t_start = int(max(num_inference_steps - init_timestep, 0))
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :].to(device)

        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start

    def prepare_latents(
        self,
        image: torch.Tensor,
        timestep: torch.Tensor,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch "
                f"size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        image = image.to(device=device, dtype=dtype)

        if image.shape[1] == self.transformer.config.in_channels:
            image_latents = image
        elif image.shape[1] == self.vae.config.latent_channels:
            image_latents = self._patchify_latents(image)
            bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(
                device=image_latents.device, dtype=image_latents.dtype
            )
            bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + 1e-5).to(
                device=image_latents.device, dtype=image_latents.dtype
            )
            image_latents = (image_latents - bn_mean) / bn_std
        else:
            image_latents = self._encode_vae_image(image=image, generator=generator)

        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            image_latents = torch.cat([image_latents] * (batch_size // image_latents.shape[0]), dim=0)
        elif batch_size > image_latents.shape[0]:
            raise ValueError(f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} prompts.")

        noise = randn_tensor(image_latents.shape, generator=generator, device=device, dtype=dtype)
        latents = self.scheduler.scale_noise(image_latents, timestep, noise)

        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        negative_prompt: Optional[Union[str, List[str]]] = "",
        strength: float = 0.6,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[list[torch.FloatTensor]] = None,
        negative_prompt_embeds: Optional[list[torch.FloatTensor]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        use_pe: bool = True,
    ):
        """
        Generate images from a prompt and an init image.
        """
        device = self._execution_device
        dtype = self.transformer.dtype

        self._guidance_scale = guidance_scale

        if image is None:
            raise ValueError("`image` must be provided for img2img generation.")
        if strength < 0 or strength > 1:
            raise ValueError(f"`strength` must be in [0.0, 1.0], but is {strength}.")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Must provide either `prompt` or `prompt_embeds`.")
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot provide both `prompt` and `prompt_embeds` at the same time.")
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` must be a subset of {self._callback_tensor_inputs}, but got "
                f"{callback_on_step_end_tensor_inputs}."
            )
        callback_inputs = callback_on_step_end_tensor_inputs or []

        latent_input = isinstance(image, torch.Tensor) and image.ndim == 4 and image.shape[1] in {
            self.vae.config.latent_channels,
            self.transformer.config.in_channels,
        }

        if latent_input:
            init_image = image
            if image.shape[1] == self.transformer.config.in_channels:
                height = height or image.shape[-2] * self.vae_scale_factor
                width = width or image.shape[-1] * self.vae_scale_factor
            else:
                latent_scale_factor = max(self.vae_scale_factor // 2, 1)
                height = height or image.shape[-2] * latent_scale_factor
                width = width or image.shape[-1] * latent_scale_factor
        else:
            image_processor = self._get_image_processor()
            init_image = image_processor.preprocess(image, height=height, width=width)
            height, width = init_image.shape[-2:]
            init_image = init_image.to(dtype=torch.float32)

        if prompt is not None and isinstance(prompt, str):
            prompt = [prompt]

        revised_prompts: Optional[List[str]] = None
        if prompt is not None and use_pe and self.pe is not None and self.pe_tokenizer is not None:
            prompt = [self._enhance_prompt_with_pe(p, device, width=width, height=height) for p in prompt]
            revised_prompts = list(prompt)

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = len(prompt_embeds)
        total_batch_size = batch_size * num_images_per_prompt

        if negative_prompt is None:
            negative_prompt = ""
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * batch_size
        if len(negative_prompt) != batch_size:
            raise ValueError(f"`negative_prompt` must have the same batch size as the prompt batch ({batch_size}).")

        if prompt_embeds is not None:
            text_hiddens = prompt_embeds
        else:
            text_hiddens = self.encode_prompt(prompt, device, num_images_per_prompt)

        if self.do_classifier_free_guidance:
            if negative_prompt_embeds is not None:
                uncond_text_hiddens = negative_prompt_embeds
            else:
                uncond_text_hiddens = self.encode_prompt(negative_prompt, device, num_images_per_prompt)

        sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1, dtype=torch.float32)
        self.scheduler.set_timesteps(sigmas=sigmas[:-1], device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        if num_inference_steps < 1:
            raise ValueError(
                f"After adjusting `num_inference_steps` by `strength={strength}`, the resulting number of steps is "
                f"{num_inference_steps}, which is not valid for img2img generation."
            )

        latent_timestep = timesteps[:1].repeat(total_batch_size)
        latents = self.prepare_latents(
            init_image,
            latent_timestep,
            total_batch_size,
            dtype,
            device,
            generator,
            latents=latents,
        )

        if self.do_classifier_free_guidance:
            cfg_text_hiddens = list(uncond_text_hiddens) + list(text_hiddens)
        else:
            cfg_text_hiddens = text_hiddens
        text_bth, text_lens = self._pad_text(
            text_hiddens=cfg_text_hiddens, device=device, dtype=dtype, text_in_dim=self.transformer.config.text_in_dim
        )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents, latents], dim=0)
                    t_batch = torch.full((total_batch_size * 2,), t.item(), device=device, dtype=dtype)
                else:
                    latent_model_input = latents
                    t_batch = torch.full((total_batch_size,), t.item(), device=device, dtype=dtype)

                pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=t_batch,
                    text_bth=text_bth,
                    text_lens=text_lens,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    pred_uncond, pred_cond = pred.chunk(2, dim=0)
                    pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

                latents = self.scheduler.step(pred, t, latents).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    if callback_outputs is not None:
                        latents = callback_outputs.pop("latents", latents)

                progress_bar.update()

        if output_type == "latent":
            return latents

        bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(device=device, dtype=latents.dtype)
        bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + 1e-5).to(device=device, dtype=latents.dtype)
        latents = latents * bn_std + bn_mean
        latents = self._unpatchify_latents(latents)

        images = self.vae.decode(latents, return_dict=False)[0]
        images = (images.clamp(-1, 1) + 1) / 2
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            images = self._get_image_processor().numpy_to_pil(images)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (images,)

        return ErnieImagePipelineOutput(images=images, revised_prompts=revised_prompts)
