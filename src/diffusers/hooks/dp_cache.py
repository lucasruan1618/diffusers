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

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from ..utils.torch_utils import unwrap_module
from ._common import _ALL_TRANSFORMER_BLOCK_IDENTIFIERS
from ._helpers import TransformerBlockRegistry
from .hooks import BaseState, HookRegistry, ModelHook, StateManager


_DP_CACHE_HOOK = "dp_cache"
_DP_CACHE_BLOCK_HOOK = "dp_cache_block"


def _update_factors(features, previous_factors, distance, max_order):
    factors = [features.detach().float().clone()]
    for previous in previous_factors[:max_order]:
        factors.append((factors[-1] - previous) / distance)
    return factors


def _predict(factors, distance):
    output = factors[0].clone()
    for order, factor in enumerate(factors[1:], 1):
        output.add_(factor, alpha=distance**order / math.factorial(order))
    return output


@torch.no_grad()
def build_dp_cache_cost_tensor(features: list[torch.Tensor], max_order: int = 2) -> torch.Tensor:
    """Build DPCache's path-aware costs from one full trajectory of final-stack features.

    Indices increase in execution order. Entry `[i, j, k]` sums mean absolute prediction errors at steps `j + 1`
    through `k`, including the endpoint as in equation 6. Index `T` is a sentinel extrapolated from the final
    consecutive features, not an additional model evaluation. Invalid transitions have infinite cost.

    For second-order calibration, the derivative at anchor `i` uses feature `i - 1`, as in the reference
    implementation. This is a proxy for the earlier selected history, which a three-index cost tensor cannot encode.
    Features and cost construction are kept on CPU; the returned tensor has shape `(T + 1, T + 1, T + 1)`.
    """
    if type(max_order) is not int or max_order not in (0, 1, 2):
        raise ValueError("`max_order` must be 0, 1, or 2.")
    total_steps = len(features)
    if total_steps < max(2, max_order + 1):
        raise ValueError("Calibration needs at least two steps and `max_order + 1` features.")
    if any(feature.shape != features[0].shape for feature in features):
        raise ValueError("Calibration features must have the same shape throughout a trajectory.")
    clip_fp16 = features[0].dtype == torch.float16
    features = [feature.detach().to(device="cpu", dtype=torch.float32) for feature in features]
    if any(not torch.isfinite(feature).all() for feature in features):
        raise ValueError("Calibration features must be finite.")

    tail_factors = []
    for feature in features[-(max_order + 1) :]:
        tail_factors = _update_factors(feature, tail_factors, 1, max_order)
    sentinel = _predict(tail_factors, 1)
    features.append(sentinel.clamp(-65504, 65504) if clip_fp16 else sentinel)
    costs = torch.full((total_steps + 1,) * 3, torch.inf, dtype=torch.float64)
    for previous in range(total_steps - 1):
        previous_factors = []
        for feature in features[max(0, previous - max_order + 1) : previous + 1]:
            previous_factors = _update_factors(feature, previous_factors, 1, max_order)
        for current in range(previous + 1, total_steps):
            factors = _update_factors(features[current], previous_factors, current - previous, max_order)
            error = 0.0
            for following in range(current + 1, total_steps + 1):
                prediction = _predict(factors, following - current)
                if clip_fp16:
                    prediction = prediction.clamp(-65504, 65504)
                error += (prediction - features[following]).abs().mean().item()
                costs[previous, current, following] = error
    return costs


def compute_dp_cache_schedule(cost_tensor: torch.Tensor, num_compute_steps: int, warmup_steps: int = 3) -> list[int]:
    """Select exactly `num_compute_steps` zero-based steps, with a fixed consecutive warmup.

    The terminal sentinel contributes cost but consumes no compute budget and is excluded from the result. Unlike the
    paper's one-predecessor recurrence, this solver retains both preceding steps in its state to find the exact minimum
    for the supplied path-dependent costs. Time is O(K T^3); backtracking storage is O(K T^2).
    """
    if cost_tensor.ndim != 3 or len(set(cost_tensor.shape)) != 1:
        raise ValueError("`cost_tensor` must have shape (T + 1, T + 1, T + 1).")
    total_steps = cost_tensor.shape[0] - 1
    if type(warmup_steps) is not int or not 2 <= warmup_steps <= total_steps:
        raise ValueError("`warmup_steps` must be an integer between 2 and T.")
    if type(num_compute_steps) is not int or not warmup_steps <= num_compute_steps <= total_steps:
        raise ValueError("`num_compute_steps` must be an integer between `warmup_steps` and T.")
    costs = cost_tensor.detach().to(device="cpu", dtype=torch.float64).numpy()
    if np.isnan(costs).any() or (costs < 0).any():
        raise ValueError("Costs must be non-negative or positive infinity.")
    size = total_steps + 1
    scores = np.full((size, size), np.inf)
    scores[warmup_steps - 2, warmup_steps - 1] = 0
    parents = np.full((num_compute_steps + 2, size, size), -1, dtype=np.int64)
    for count in range(warmup_steps + 1, num_compute_steps + 2):
        next_scores = np.full_like(scores, np.inf)
        for current in range(warmup_steps - 1, total_steps):
            following_steps = [total_steps] if count == num_compute_steps + 1 else range(current + 1, total_steps)
            for following in following_steps:
                candidates = scores[:current, current] + costs[:current, current, following]
                previous = int(np.argmin(candidates))
                next_scores[current, following] = candidates[previous]
                parents[count, current, following] = previous
        scores = next_scores
    current = int(np.argmin(scores[:, total_steps]))
    if not np.isfinite(scores[current, total_steps]):
        raise ValueError("No finite schedule satisfies the requested compute budget and warmup.")
    following = total_steps
    selected = [current]
    for count in range(num_compute_steps + 1, warmup_steps, -1):
        previous = int(parents[count, current, following])
        selected.append(previous)
        current, following = previous, current
    return list(range(warmup_steps - 2)) + selected[::-1]


@dataclass
class DPCacheConfig:
    r"""Configuration for [DPCache](https://huggingface.co/papers/2602.22654).

    Args:
        num_inference_steps (`int`):
            Number of denoiser calls per cache context in a complete sampling trajectory. Use the same scheduler, step
            count, model, resolution, and conditioning setup for calibration and inference.
        compute_steps (`list[int]` or `tuple[int, ...]`, *optional*):
            Sorted, unique, zero-based steps that run the transformer stack. Must include the first `max_order + 1`
            steps. Required for inference; obtain these with `compute_schedule` after calibration.
        max_order (`int`, defaults to `2`):
            Taylor prediction order: 0, 1, or 2. Factors are computed and stored in float32. Outputs use the model
            dtype, with float16 predictions clamped to its finite range.
        calibrate (`bool`, defaults to `False`):
            Run every step and collect final-stack features on CPU. Each completed cache context contributes one sample
            to `cost_tensor`, the mean path-aware cost tensor. `num_calibration_samples` counts these samples.

    Wan and FLUX use their existing registered transformer blocks. Conditional and unconditional contexts maintain
    separate histories. Caching is bypassed when autograd is enabled. Compile with `compile_repeated_blocks` to keep
    cache management outside compiled regions.
    """

    num_inference_steps: int
    compute_steps: list[int] | tuple[int, ...] | None = None
    max_order: int = 2
    calibrate: bool = False
    cost_tensor: torch.Tensor | None = field(default=None, init=False, repr=False)
    num_calibration_samples: int = field(default=0, init=False)

    def __post_init__(self):
        if type(self.max_order) is not int or self.max_order not in (0, 1, 2):
            raise ValueError("`max_order` must be 0, 1, or 2.")
        if type(self.num_inference_steps) is not int or self.num_inference_steps < max(2, self.max_order + 1):
            raise ValueError("`num_inference_steps` must be an integer at least max(2, max_order + 1).")
        if self.calibrate:
            if self.compute_steps is not None:
                raise ValueError("Do not provide `compute_steps` when calibrating.")
        else:
            if self.compute_steps is None:
                raise ValueError("Provide calibrated `compute_steps`, or set `calibrate=True`.")
            self.compute_steps = tuple(self.compute_steps)
            if (
                any(type(step) is not int or not 0 <= step < self.num_inference_steps for step in self.compute_steps)
                or tuple(sorted(set(self.compute_steps))) != self.compute_steps
                or self.compute_steps[: self.max_order + 1] != tuple(range(self.max_order + 1))
            ):
                raise ValueError(
                    "`compute_steps` must be sorted, unique indices in [0, num_inference_steps), "
                    "starting with steps 0 through max_order."
                )

    def compute_schedule(self, num_compute_steps: int, warmup_steps: int = 3) -> list[int]:
        """Select a schedule from the accumulated calibration costs without changing this configuration."""
        if self.cost_tensor is None:
            raise ValueError("Run a complete calibration trajectory before selecting a schedule.")
        if warmup_steps < self.max_order + 1:
            raise ValueError("`warmup_steps` must be at least `max_order + 1`.")
        return compute_dp_cache_schedule(self.cost_tensor, num_compute_steps, warmup_steps)


class DPCacheState(BaseState):
    def __init__(self):
        self.reset()

    def reset(self):
        self.step = -1
        self.last_compute_step = -1
        self.factors = []
        self.features = []
        self.output_dtype = None
        self.should_compute = True


class DPCacheHook(ModelHook):
    _is_stateful = True

    def __init__(self, config: DPCacheConfig):
        super().__init__()
        self.config = config
        self.state_manager = StateManager(DPCacheState)

    def reset_state(self, module):
        self.state_manager.reset()

    def deinitalize_hook(self, module):
        self.state_manager.reset()
        return module

    @torch.compiler.disable(recursive=False)
    def pre_forward(self, module, *args, **kwargs):
        if torch.is_grad_enabled():
            return args, kwargs
        context = self.state_manager.context
        state = self.state_manager.get_state()
        config = self.config
        if context.num_inference_steps is not None and context.num_inference_steps != config.num_inference_steps:
            raise ValueError("DPCache was calibrated for a different number of inference steps.")
        step = context.step_index if context.step_index is not None else state.step + 1
        if step == 0:
            state.reset()
        if step != state.step + 1 or not 0 <= step < config.num_inference_steps:
            raise ValueError("DPCache requires consecutive steps starting at 0; reset the cache before a new run.")
        state.step = step
        state.should_compute = config.calibrate or step in config.compute_steps
        return args, kwargs


class DPCacheBlockHook(ModelHook):
    def __init__(self, config, state_manager, metadata, is_tail):
        super().__init__()
        self.config = config
        self.state_manager = state_manager
        self.metadata = metadata
        self.is_tail = is_tail

    def initialize_hook(self, module):
        self.compiled_forward = None
        self.instance_compile = module.__dict__.get("compile")
        module.compile = self.compile
        return module

    def compile(self, *args, **kwargs):
        """Compile the original block body while leaving cache dispatch outside the graph."""
        self.compiled_forward = torch.compile(self.fn_ref.original_forward, *args, **kwargs)

    def deinitalize_hook(self, module):
        if self.instance_compile is None:
            delattr(module, "compile")
        else:
            module.compile = self.instance_compile
        return module

    @torch.compiler.disable(recursive=False)
    def new_forward(self, module, *args, **kwargs):
        forward = self.compiled_forward if self.compiled_forward is not None else self.fn_ref.original_forward
        if torch.is_grad_enabled():
            return forward(*args, **kwargs)
        state = self.state_manager.get_state()
        metadata = self.metadata
        if state.should_compute:
            output = forward(*args, **kwargs)
            if not self.is_tail:
                return output
            features = output if isinstance(output, torch.Tensor) else output[metadata.return_hidden_states_index]
            if self.config.calibrate:
                state.features.append(features.detach().to(device="cpu", copy=True))
                if state.step == self.config.num_inference_steps - 1:
                    costs = build_dp_cache_cost_tensor(state.features, self.config.max_order)
                    if self.config.cost_tensor is None:
                        self.config.cost_tensor = costs
                    else:
                        finite = torch.isfinite(costs)
                        self.config.cost_tensor[finite] += (costs[finite] - self.config.cost_tensor[finite]) / (
                            self.config.num_calibration_samples + 1
                        )
                    self.config.num_calibration_samples += 1
                    state.features.clear()
            else:
                state.factors = _update_factors(
                    features, state.factors, state.step - state.last_compute_step, self.config.max_order
                )
                state.last_compute_step = state.step
                state.output_dtype = features.dtype
            return output

        hidden_states = metadata._get_parameter_from_args_kwargs(metadata.hidden_states_argument_name, args, kwargs)
        if self.is_tail:
            hidden_states = _predict(state.factors, state.step - state.last_compute_step)
            if state.output_dtype == torch.float16:
                hidden_states = hidden_states.clamp(-65504, 65504)
            hidden_states = hidden_states.to(state.output_dtype)
        if metadata.return_encoder_hidden_states_index is None:
            return hidden_states
        encoder_hidden_states = metadata._get_parameter_from_args_kwargs(
            metadata.encoder_hidden_states_argument_name, args, kwargs
        )
        output = [None, None]
        output[metadata.return_hidden_states_index] = hidden_states
        output[metadata.return_encoder_hidden_states_index] = encoder_hidden_states
        return tuple(output)


def apply_dp_cache(module: torch.nn.Module, config: DPCacheConfig):
    """Attach DPCache to existing block lists using `TransformerBlockRegistry` metadata.

    Enable caching before calling `compile_repeated_blocks`. Each hooked block compiles its original forward body; the
    cache dispatch remains eager. Disabling the cache restores ordinary block compilation methods.
    """
    module = unwrap_module(module)
    blocks = []
    for name, children in module.named_children():
        if name not in _ALL_TRANSFORMER_BLOCK_IDENTIFIERS or not isinstance(children, torch.nn.ModuleList):
            continue
        for block in children:
            block = unwrap_module(block)
            if block._compiled_call_impl is not None:
                raise ValueError("Enable DPCache before compiling the repeated blocks.")
            metadata = TransformerBlockRegistry.get(type(block))
            if metadata.return_encoder_hidden_states_index is not None and (
                metadata.return_hidden_states_index,
                metadata.return_encoder_hidden_states_index,
            ) not in ((0, 1), (1, 0)):
                raise ValueError("DPCache requires tensor outputs or a pair of hidden and encoder hidden states.")
            blocks.append((block, metadata))
    if not blocks:
        raise ValueError("DPCache found no registered transformer block lists on the model.")
    root_hook = DPCacheHook(config)
    registry = HookRegistry.check_if_exists_or_initialize(module)
    registry.register_hook(root_hook, _DP_CACHE_HOOK)
    for index, (block, metadata) in enumerate(blocks):
        hook = DPCacheBlockHook(config, root_hook.state_manager, metadata, index == len(blocks) - 1)
        HookRegistry.check_if_exists_or_initialize(block).register_hook(hook, _DP_CACHE_BLOCK_HOOK)
