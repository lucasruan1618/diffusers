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

"""
Ernie-Image Inpaint Pipeline for HuggingFace Diffusers.

Architecture notes
──────────────────
* VAE        : AutoencoderKLFlux2  – uses internal BatchNorm stats (running_mean /
               running_var) to normalise/denormalise latents.
* Patchify   : 2 × 2 spatial patchification  [B, C, H, W] ↔ [B, 4C, H/2, W/2]
* Text enc   : single Qwen-style encoder; embeddings are a *list* of variable-length
               tensors that are later padded in `_pad_text`.
* Scheduler  : FlowMatchEulerDiscreteScheduler with explicit sigma linspace.
* Inpaint    : repaint strategy – at every denoising step the unmasked region is
               replaced with the noise-scaled image latent for that timestep.
"""

import json
from typing import Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKLFlux2
from ...models.transformers import ErnieImageTransformer2DModel
from ...pipelines.pipeline_utils import DiffusionPipeline
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils.torch_utils import randn_tensor
from .pipeline_output import ErnieImagePipelineOutput


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _retrieve_latents(
    encoder_output: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
) -> torch.Tensor:
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class ErnieImageInpaintPipeline(DiffusionPipeline):
    """
    Pipeline for text-guided inpainting using ErnieImageTransformer2DModel.

    The pipeline accepts an *image* and a *mask_image* (white = repaint,
    black = keep) and fills the masked region according to ``prompt``.

    At every denoising step the unmasked latents are replaced with the
    noise-scaled original image latents, so the unmasked area is always
    anchored to the input image.

    Args:
        transformer (:class:`ErnieImageTransformer2DModel`):
            DiT backbone.
        vae (:class:`AutoencoderKLFlux2`):
            VAE that uses BN running statistics for latent normalisation.
        text_encoder (:class:`~transformers.AutoModel`):
            Qwen-style language model used for text conditioning.
        tokenizer (:class:`~transformers.AutoTokenizer`):
            Tokenizer paired with ``text_encoder``.
        scheduler (:class:`FlowMatchEulerDiscreteScheduler`):
            Flow-matching scheduler.
        pe (:class:`~transformers.AutoModelForCausalLM`, *optional*):
            Prompt-enhancement (rewrite) model.
        pe_tokenizer (:class:`~transformers.AutoTokenizer`, *optional*):
            Tokenizer for the PE model.
    """

    model_cpu_offload_seq = "pe->text_encoder->transformer->vae"
    _optional_components = ["pe", "pe_tokenizer"]
    _callback_tensor_inputs = ["latents"]

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        transformer: ErnieImageTransformer2DModel,
        vae: AutoencoderKLFlux2,
        text_encoder: AutoModel,
        tokenizer: AutoTokenizer,
        scheduler: FlowMatchEulerDiscreteScheduler,
        pe: Optional[AutoModelForCausalLM] = None,
        pe_tokenizer: Optional[AutoTokenizer] = None,
    ):
        super().__init__()
        self.register_modules(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            pe=pe,
            pe_tokenizer=pe_tokenizer,
        )

        # VAE spatial downscale factor (e.g. 16 for Flux2 VAE with 4 block levels)
        self.vae_scale_factor = (
            2 ** len(self.vae.config.block_out_channels) if getattr(self, "vae", None) else 16
        )

        # Image processor – pixels are normalised to [-1, 1] by default
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        # Mask processor – grayscale, binarised, NOT normalised (stays in [0,1])
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def guidance_scale(self) -> float:
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self._guidance_scale > 1.0

    # ------------------------------------------------------------------
    # prompt enhancement (PE)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _enhance_prompt_with_pe(
        self,
        prompt: str,
        device: torch.device,
        width: int = 1024,
        height: int = 1024,
        system_prompt: Optional[str] = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
    ) -> str:
        """Rewrite a short prompt via the PE chat model."""
        user_content = json.dumps(
            {"prompt": prompt, "width": width, "height": height},
            ensure_ascii=False,
        )
        messages: List[dict] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        input_text = self.pe_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        inputs = self.pe_tokenizer(input_text, return_tensors="pt").to(device)
        output_ids = self.pe.generate(
            **inputs,
            max_new_tokens=self.pe_tokenizer.model_max_length,
            do_sample=temperature != 1.0 or top_p != 1.0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.pe_tokenizer.pad_token_id,
            eos_token_id=self.pe_tokenizer.eos_token_id,
        )
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.pe_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # text encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        num_images_per_prompt: int = 1,
    ) -> List[torch.Tensor]:
        """Encode text prompts into a list of hidden-state tensors."""
        if isinstance(prompt, str):
            prompt = [prompt]

        text_hiddens: List[torch.Tensor] = []
        for p in prompt:
            ids = self.tokenizer(
                p,
                add_special_tokens=True,
                truncation=True,
                padding=False,
            )["input_ids"]

            if len(ids) == 0:
                ids = (
                    [self.tokenizer.bos_token_id]
                    if self.tokenizer.bos_token_id is not None
                    else [0]
                )

            input_ids = torch.tensor([ids], device=device)
            outputs = self.text_encoder(
                input_ids=input_ids,
                output_hidden_states=True,
            )
            # second-to-last hidden state, shape [T, H]
            hidden = outputs.hidden_states[-2][0]

            for _ in range(num_images_per_prompt):
                text_hiddens.append(hidden)

        return text_hiddens

    # ------------------------------------------------------------------
    # patchify / unpatchify
    # ------------------------------------------------------------------

    @staticmethod
    def _patchify_latents(latents: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] → [B, 4C, H/2, W/2]"""
        b, c, h, w = latents.shape
        latents = latents.view(b, c, h // 2, 2, w // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        return latents.reshape(b, c * 4, h // 2, w // 2)

    @staticmethod
    def _unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
        """[B, 4C, H/2, W/2] → [B, C, H, W]"""
        b, c, h, w = latents.shape
        latents = latents.reshape(b, c // 4, 2, 2, h, w)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        return latents.reshape(b, c // 4, h * 2, w * 2)

    @staticmethod
    def _to_pil_list(
        image: Optional[Union[Image.Image, np.ndarray, torch.Tensor, List[Image.Image]]],
        convert_mode: Optional[str] = None,
    ) -> List[Image.Image]:
        if image is None:
            return []
        if not isinstance(image, list):
            image = [image]

        pil_images: List[Image.Image] = []
        for item in image:
            if isinstance(item, Image.Image):
                pil = item
            elif isinstance(item, np.ndarray):
                array = item
                if array.ndim == 4:
                    if array.shape[0] != 1:
                        raise ValueError("Only single-image numpy batches can be composited back to PIL.")
                    array = array[0]
                if array.ndim == 3 and array.shape[0] in (1, 3):
                    array = np.transpose(array, (1, 2, 0))
                if array.dtype != np.uint8:
                    array = array.astype(np.float32)
                    if array.min() < 0:
                        array = (array + 1.0) / 2.0
                    array = np.clip(array, 0.0, 1.0)
                    array = (array * 255.0).round().astype(np.uint8)
                if array.ndim == 2 or (array.ndim == 3 and array.shape[-1] == 1):
                    pil = Image.fromarray(array.squeeze(-1) if array.ndim == 3 else array, mode="L")
                else:
                    pil = Image.fromarray(array)
            elif isinstance(item, torch.Tensor):
                tensor = item.detach().cpu()
                if tensor.ndim == 4:
                    if tensor.shape[0] != 1:
                        raise ValueError("Only single-image tensor batches can be composited back to PIL.")
                    tensor = tensor[0]
                if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
                    tensor = tensor.permute(1, 2, 0)
                array = tensor.float().numpy()
                if array.min() < 0:
                    array = (array + 1.0) / 2.0
                array = np.clip(array, 0.0, 1.0)
                array = (array * 255.0).round().astype(np.uint8)
                if array.ndim == 2 or (array.ndim == 3 and array.shape[-1] == 1):
                    pil = Image.fromarray(array.squeeze(-1) if array.ndim == 3 else array, mode="L")
                else:
                    pil = Image.fromarray(array)
            else:
                raise TypeError(f"Unsupported image type for PIL conversion: {type(item)!r}")

            if convert_mode is not None:
                pil = pil.convert(convert_mode)

            pil_images.append(pil)

        return pil_images

    @staticmethod
    def _match_batch_size(images: List[Image.Image], batch_size: int, name: str) -> List[Image.Image]:
        if len(images) == batch_size:
            return images
        if len(images) == 1:
            return images * batch_size
        raise ValueError(f"`{name}` batch size ({len(images)}) does not match generated image batch ({batch_size}).")

    @classmethod
    def _prepare_overlay_inputs(
        cls,
        original_image: Union[Image.Image, np.ndarray, torch.Tensor, List[Image.Image]],
        mask_image: Union[Image.Image, np.ndarray, torch.Tensor, List[Image.Image]],
        batch_size: int,
    ) -> tuple[List[Image.Image], List[Image.Image]]:
        original_images = cls._match_batch_size(
            cls._to_pil_list(original_image, convert_mode="RGB"),
            batch_size,
            "image",
        )
        mask_images = cls._match_batch_size(
            cls._to_pil_list(mask_image, convert_mode="L"),
            batch_size,
            "mask_image",
        )

        overlay_masks: List[Image.Image] = []
        for original, mask in zip(original_images, mask_images):
            if mask.size != original.size:
                mask = mask.resize(original.size, resample=Image.NEAREST)
            mask = mask.point(lambda value: 255 if value >= 128 else 0, mode="L")
            overlay_masks.append(mask)

        return original_images, overlay_masks

    @staticmethod
    def _dilate_mask(mask: torch.Tensor, mask_dilate_pixels: int) -> torch.Tensor:
        if mask_dilate_pixels <= 0:
            return mask

        kernel_size = 2 * mask_dilate_pixels + 1
        dilated = F.max_pool2d(mask.float(), kernel_size=kernel_size, stride=1, padding=mask_dilate_pixels)
        return dilated.to(dtype=mask.dtype)

    # ------------------------------------------------------------------
    # text padding utility
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_text(
        text_hiddens: List[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        text_in_dim: int,
    ):
        B = len(text_hiddens)
        if B == 0:
            return (
                torch.zeros((0, 0, text_in_dim), device=device, dtype=dtype),
                torch.zeros((0,), device=device, dtype=torch.long),
            )
        normalised = [
            th.squeeze(1).to(device, dtype) if th.dim() == 3 else th.to(device, dtype)
            for th in text_hiddens
        ]
        lens = torch.tensor([t.shape[0] for t in normalised], device=device, dtype=torch.long)
        Tmax = int(lens.max().item())
        text_bth = torch.zeros((B, Tmax, text_in_dim), device=device, dtype=dtype)
        for i, t in enumerate(normalised):
            text_bth[i, : t.shape[0]] = t
        return text_bth, lens

    # ------------------------------------------------------------------
    # VAE encode / decode helpers
    # ------------------------------------------------------------------

    def _vae_encode(
        self,
        image: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Encode pixel-space image to raw latents [B, C_vae, H/sf, W/sf]."""
        if isinstance(generator, list):
            latents = torch.cat(
                [
                    _retrieve_latents(self.vae.encode(image[i: i + 1]), generator=generator[i])
                    for i in range(image.shape[0])
                ],
                dim=0,
            )
        else:
            latents = _retrieve_latents(self.vae.encode(image), generator=generator)
        return latents

    def _bn_normalize(self, latents: torch.Tensor) -> torch.Tensor:
        """Normalize patchified latents [B, 4·C, H/2, W/2] with VAE BN running stats."""
        bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1).to(latents.device, latents.dtype) + 1e-5
        )
        return (latents - bn_mean) / bn_std

    def _vae_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Denormalise patchified latents and decode back to pixel space [-1, 1]."""
        bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1).to(latents.device, latents.dtype) + 1e-5
        )
        latents = latents * bn_std + bn_mean
        latents = self._unpatchify_latents(latents)
        return self.vae.decode(latents, return_dict=False)[0]

    # ------------------------------------------------------------------
    # latent preparation
    # ------------------------------------------------------------------

    def _prepare_latents(
        self,
        image: torch.Tensor,
        timestep: torch.Tensor,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator],
        latents: Optional[torch.Tensor] = None,
    ):
        """
        Build the initial noisy latents for inpainting.

        vae_scale_factor = vae_spatial_factor × patch_size (e.g. 8 × 2 = 16).
        All operations before patchification use the pre-patchify spatial size
        (height // (vae_scale_factor // 2)).

        Returns
        -------
        latents          : noisy latents fed to the denoising loop  [B, 4C, H/sf, W/sf]
        noise            : pure Gaussian noise in patchified space  [B, 4C, H/sf, W/sf]
        image_latents    : BN-normalised patchified clean image     [B, 4C, H/sf, W/sf]
        """
        # Pre-patchify spatial dimensions (VAE output resolution)
        pre_h = height * 2 // self.vae_scale_factor
        pre_w = width * 2 // self.vae_scale_factor

        # Encode image → raw latents [B, C, pre_h, pre_w]
        image = image.to(device=device, dtype=dtype)
        image_latents = self._vae_encode(image, generator=generator)

        # Expand to batch size if needed
        if image_latents.shape[0] < batch_size:
            if batch_size % image_latents.shape[0] != 0:
                raise ValueError(
                    f"Cannot tile image of batch size {image_latents.shape[0]} "
                    f"to requested batch size {batch_size}."
                )
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)

        # Noise lives in pre-patchify space so it matches image_latents
        noise_shape = (batch_size, num_channels_latents, pre_h, pre_w)
        noise = randn_tensor(noise_shape, generator=generator, device=device, dtype=dtype)

        if latents is None:
            # Mix image latents with noise according to the flow-matching schedule
            latents = self.scheduler.scale_noise(image_latents, timestep, noise)
        else:
            latents = latents.to(device)

        # Patchify → [B, 4C, H/sf, W/sf], then BN-normalize image-derived tensors.
        # BN running stats have 4C channels (post-patchify); noise is already Gaussian.
        noise = self._patchify_latents(noise)
        image_latents = self._bn_normalize(self._patchify_latents(image_latents))
        latents = self._bn_normalize(self._patchify_latents(latents))

        return latents, noise, image_latents

    def _prepare_mask_latents(
        self,
        mask: torch.Tensor,
        masked_image: torch.Tensor,
        batch_size: int,
        num_images_per_prompt: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator],
    ):
        """
        Resize the binary mask to latent resolution and encode the masked image.

        Returns
        -------
        mask                  : [B, 1, pre_h, pre_w]  (0 = keep, 1 = repaint)
                                at pre-patchify resolution so that expand+patchify
                                in __call__ yields [B, 4C, H/sf, W/sf].
        masked_image_latents  : [B, C_vae, pre_h, pre_w]  raw VAE latents of the
                                masked image (masked area zeroed out).
        """
        total_batch = batch_size * num_images_per_prompt

        # Pre-patchify spatial dimensions (same as VAE output)
        pre_h = height * 2 // self.vae_scale_factor
        pre_w = width * 2 // self.vae_scale_factor

        # Resize mask to pre-patchify resolution
        mask = F.interpolate(mask.float(), size=(pre_h, pre_w), mode="nearest")
        mask = mask.to(device=device, dtype=dtype)

        # Encode the image with masked region zeroed out
        masked_image = masked_image.to(device=device, dtype=dtype)
        masked_image_latents = self._vae_encode(masked_image, generator=generator)

        # Tile to total batch size
        if mask.shape[0] < total_batch:
            if total_batch % mask.shape[0] != 0:
                raise ValueError(
                    f"Cannot tile mask of batch size {mask.shape[0]} "
                    f"to total batch size {total_batch}."
                )
            mask = mask.repeat(total_batch // mask.shape[0], 1, 1, 1)

        if masked_image_latents.shape[0] < total_batch:
            if total_batch % masked_image_latents.shape[0] != 0:
                raise ValueError(
                    f"Cannot tile masked image latents of batch size "
                    f"{masked_image_latents.shape[0]} to total batch size {total_batch}."
                )
            masked_image_latents = masked_image_latents.repeat(
                total_batch // masked_image_latents.shape[0], 1, 1, 1
            )

        return mask, masked_image_latents

    # ------------------------------------------------------------------
    # input validation
    # ------------------------------------------------------------------

    def _check_inputs(
        self,
        prompt,
        prompt_embeds,
        negative_prompt,
        negative_prompt_embeds,
        image,
        mask_image,
        height,
        width,
        strength,
        callback_on_step_end_tensor_inputs,
    ):
        if prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot provide both `prompt` and `prompt_embeds`.")
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError("Cannot provide both `negative_prompt` and `negative_prompt_embeds`.")
        if image is None:
            raise ValueError("`image` must be provided for inpainting.")
        if mask_image is None:
            raise ValueError("`mask_image` must be provided for inpainting.")
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"`strength` must be in [0, 1] but got {strength}.")
        if height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0:
            raise ValueError(
                f"`height` and `width` must be divisible by {self.vae_scale_factor} "
                f"but got {height} and {width}."
            )
        if callback_on_step_end_tensor_inputs is not None:
            bad = [
                k for k in callback_on_step_end_tensor_inputs
                if k not in self._callback_tensor_inputs
            ]
            if bad:
                raise ValueError(
                    f"`callback_on_step_end_tensor_inputs` contains invalid keys {bad}. "
                    f"Valid keys: {self._callback_tensor_inputs}."
                )

    # ------------------------------------------------------------------
    # timestep helpers
    # ------------------------------------------------------------------

    def _get_timesteps(self, num_inference_steps: int, strength: float, device):
        """Skip the first (1-strength) fraction of timesteps."""
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order:]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)
        return timesteps, num_inference_steps - t_start

    # ------------------------------------------------------------------
    # main call
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = "",
        image: Optional[Union[Image.Image, torch.Tensor]] = None,
        mask_image: Optional[Union[Image.Image, torch.Tensor]] = None,
        masked_image_latents: Optional[torch.Tensor] = None,
        height: int = 1024,
        width: int = 1024,
        padding_mask_crop: Optional[int] = None,
        strength: float = 0.99,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        negative_prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        use_pe: bool = True,
        mask_dilate_pixels: int = 0,
    ):
        """
        Generate an inpainted image.

        Args:
            prompt: Text prompt(s) describing the desired fill content.
            negative_prompt: Negative prompt(s) for CFG. Default ``""``.
            image: Source image to inpaint.  Accepts ``PIL.Image``, a
                ``torch.Tensor`` in ``[0, 1]`` of shape ``(B, C, H, W)``, or a
                ``numpy.ndarray``.
            mask_image: Binary mask where **white (1) = repaint** and
                **black (0) = keep**.  Same format options as ``image``.
            masked_image_latents: Pre-computed masked image latents.  When
                supplied the masked-image encoding step is skipped.
            height: Output height in pixels (must be divisible by
                ``vae_scale_factor``). Default 1024.
            width: Output width in pixels. Default 1024.
            padding_mask_crop: If set, the image and mask are cropped to a
                tight bounding-box around the masked region (padded by this
                many pixels) before processing, then composited back.
            strength: How strongly the model should deviate from ``image``.
                ``1.0`` = full repaint; ``0.0`` = return ``image`` unchanged.
                Default 0.99.
            num_inference_steps: Denoising steps. Default 50.
            guidance_scale: Classifier-free guidance scale. Default 4.0.
            num_images_per_prompt: Images to generate per prompt entry.
            generator: ``torch.Generator`` for reproducibility.
            latents: Pre-generated initial latent noise (optional).
            prompt_embeds: Pre-computed positive text embeddings (skips
                ``encode_prompt`` when provided).
            negative_prompt_embeds: Pre-computed negative text embeddings.
            output_type: ``"pil"`` (default) or ``"latent"``.
            return_dict: Return :class:`ErnieImagePipelineOutput` when ``True``.
            callback_on_step_end: Optional callable invoked after every
                denoising step with signature
                ``(pipeline, step_index, timestep, callback_kwargs) -> dict``.
            callback_on_step_end_tensor_inputs: Tensor names forwarded to the
                callback (must be a subset of ``_callback_tensor_inputs``).
            use_pe: Enhance prompts with the PE model before generation.
            mask_dilate_pixels: Expand the repaint mask by this many pixels
                before latent blending and final compositing. Useful when a
                tight mask leaves a small halo at the border.

        Returns:
            :class:`ErnieImagePipelineOutput` with ``images`` and
            ``revised_prompts``, or a plain ``tuple`` when
            ``return_dict=False``.
        """
        device = self._execution_device
        dtype = self.transformer.dtype

        self._guidance_scale = guidance_scale

        if mask_dilate_pixels < 0:
            raise ValueError(f"`mask_dilate_pixels` must be >= 0 but got {mask_dilate_pixels}.")

        # ── 0. Input validation ──────────────────────────────────────────────
        if isinstance(prompt, str):
            prompt = [prompt]

        self._check_inputs(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt=negative_prompt,
            negative_prompt_embeds=negative_prompt_embeds,
            image=image,
            mask_image=mask_image,
            height=height,
            width=width,
            strength=strength,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        # ── 1. Pre-process image and mask ────────────────────────────────────
        if padding_mask_crop is not None:
            crops_coords = self.mask_processor.get_crop_region(
                mask_image, width, height, pad=padding_mask_crop
            )
            resize_mode = "fill"
        else:
            crops_coords = None
            resize_mode = "default"

        original_image = image

        init_image = self.image_processor.preprocess(
            image, height=height, width=width,
            crops_coords=crops_coords, resize_mode=resize_mode,
        ).to(dtype=torch.float32)  # [B, 3, H, W]  in [-1, 1]

        mask_condition = self.mask_processor.preprocess(
            mask_image, height=height, width=width,
            crops_coords=crops_coords, resize_mode=resize_mode,
        )  # [B, 1, H, W]  in {0, 1}
        mask_condition = self._dilate_mask(mask_condition, mask_dilate_pixels)

        # Masked image: zero-out the region to repaint so the VAE encoder
        # doesn't "see" the original content in the masked area.
        if masked_image_latents is None:
            masked_image = init_image * (mask_condition < 0.5)
        else:
            masked_image = masked_image_latents

        # ── 2. Batch-size bookkeeping ────────────────────────────────────────
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = len(prompt_embeds)

        total_batch_size = batch_size * num_images_per_prompt

        # ── 3. Prompt enhancement (PE) ───────────────────────────────────────
        revised_prompts: Optional[List[str]] = None
        if (
            prompt is not None
            and use_pe
            and self.pe is not None
            and self.pe_tokenizer is not None
        ):
            prompt = [
                self._enhance_prompt_with_pe(p, device, width=width, height=height)
                for p in prompt
            ]
            revised_prompts = list(prompt)

        # ── 4. Negative prompt normalisation ─────────────────────────────────
        if negative_prompt is None:
            negative_prompt = ""
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * batch_size
        if len(negative_prompt) != batch_size:
            raise ValueError(
                f"`negative_prompt` length ({len(negative_prompt)}) must match "
                f"`prompt` length ({batch_size})."
            )

        # ── 5. Text encoding ─────────────────────────────────────────────────
        if prompt_embeds is not None:
            text_hiddens = prompt_embeds
        else:
            text_hiddens = self.encode_prompt(prompt, device, num_images_per_prompt)

        if self.do_classifier_free_guidance:
            if negative_prompt_embeds is not None:
                uncond_text_hiddens = negative_prompt_embeds
            else:
                uncond_text_hiddens = self.encode_prompt(
                    negative_prompt, device, num_images_per_prompt
                )

        # ── 6. Timestep schedule ─────────────────────────────────────────────
        sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1)
        self.scheduler.set_timesteps(sigmas=sigmas[:-1], device=device)

        timesteps, num_inference_steps = self._get_timesteps(
            num_inference_steps, strength, device
        )

        if num_inference_steps < 1:
            raise ValueError(
                f"After applying strength={strength} the effective number of "
                f"inference steps is {num_inference_steps}, which is < 1."
            )

        latent_timestep = timesteps[:1].repeat(total_batch_size)

        # ── 7. Latent preparation ────────────────────────────────────────────
        # Number of channels *before* patchification
        # transformer.config.in_channels is after patchify (×4), so divide back
        num_channels_latents = self.transformer.config.in_channels // 4

        latents, noise, image_latents = self._prepare_latents(
            image=init_image,
            timestep=latent_timestep,
            batch_size=total_batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        mask, _masked_image_latents_enc = self._prepare_mask_latents(
            mask=mask_condition,
            masked_image=masked_image,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
        )

        # Patchify mask and masked-image latents so they match latents shape
        # mask: [B, 1, lH, lW] → repeat over channels then patchify
        mask_patchified = self._patchify_latents(
            mask.expand(-1, num_channels_latents, -1, -1)
        )
        _masked_image_latents_enc = self._patchify_latents(_masked_image_latents_enc)

        # ── 8. Build combined text tensors for denoising loop ─────────────────
        if self.do_classifier_free_guidance:
            cfg_text_hiddens = list(uncond_text_hiddens) + list(text_hiddens)
        else:
            cfg_text_hiddens = text_hiddens

        text_bth, text_lens = self._pad_text(
            text_hiddens=cfg_text_hiddens,
            device=device,
            dtype=dtype,
            text_in_dim=self.transformer.config.text_in_dim,
        )

        # ── 9. Denoising loop ────────────────────────────────────────────────
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Repaint: replace unmasked latents with noise-scaled
                # image latents for this timestep.
                if i < len(timesteps) - 1:
                    # Next-step timestep used to re-noise the clean image latents
                    next_t = timesteps[i + 1]
                    noisy_image_latents = self.scheduler.scale_noise(
                        self._unpatchify_latents(image_latents),
                        torch.tensor([next_t], device=device),
                        self._unpatchify_latents(noise),
                    )
                    noisy_image_latents = self._patchify_latents(noisy_image_latents)
                else:
                    noisy_image_latents = image_latents

                # ── CFG duplication ──────────────────────────────────────────
                if self.do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents, latents], dim=0)
                    t_batch = torch.full(
                        (total_batch_size * 2,), t.item(), device=device, dtype=dtype
                    )
                else:
                    latent_model_input = latents
                    t_batch = torch.full(
                        (total_batch_size,), t.item(), device=device, dtype=dtype
                    )

                # ── Transformer forward ──────────────────────────────────────
                pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=t_batch,
                    text_bth=text_bth,
                    text_lens=text_lens,
                    return_dict=False,
                )[0]

                # ── CFG merge ───────────────────────────────────────────────
                if self.do_classifier_free_guidance:
                    pred_uncond, pred_cond = pred.chunk(2, dim=0)
                    pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

                # ── Scheduler step ───────────────────────────────────────────
                latents = self.scheduler.step(pred, t, latents).prev_sample

                # ── Repaint: paste back unmasked region ──────────────────────
                # mask_patchified: 1 = repaint, 0 = keep original
                latents = (
                    (1.0 - mask_patchified) * noisy_image_latents
                    + mask_patchified * latents
                )

                # ── Callback ─────────────────────────────────────────────────
                if callback_on_step_end is not None:
                    cb_kwargs = {
                        k: locals()[k] for k in callback_on_step_end_tensor_inputs
                    }
                    cb_outputs = callback_on_step_end(self, i, t, cb_kwargs)
                    latents = cb_outputs.pop("latents", latents)

                progress_bar.update()

        # ── 10. Decode ───────────────────────────────────────────────────────
        if output_type == "latent":
            return latents

        images = self._vae_decode(latents)

        # Normalise pixels to [0, 1] then convert to uint8
        images = (images.clamp(-1, 1) + 1) / 2
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            pil_images = [Image.fromarray((img * 255).astype("uint8")) for img in images]
            original_pil_images, overlay_masks = self._prepare_overlay_inputs(
                original_image=original_image,
                mask_image=mask_image,
                batch_size=len(pil_images),
            )

            # Always composite untouched pixels back in image space. This avoids
            # VAE decode seam softening on the keep side of the mask boundary.
            pil_images = [
                self.image_processor.apply_overlay(overlay_mask, original_pil, img, crops_coords)
                for img, original_pil, overlay_mask in zip(pil_images, original_pil_images, overlay_masks)
            ]

            images = pil_images

        self.maybe_free_model_hooks()

        if not return_dict:
            return (images,)

        return ErnieImagePipelineOutput(images=images, revised_prompts=revised_prompts)
