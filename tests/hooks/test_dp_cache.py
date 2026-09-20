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

import itertools

import pytest
import torch

from diffusers import (
    AutoencoderKL,
    AutoencoderKLWan,
    DPCacheConfig,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
    WanPipeline,
    WanTransformer3DModel,
)
from diffusers.hooks.dp_cache import (
    _DP_CACHE_HOOK,
    _predict,
    _update_factors,
    build_dp_cache_cost_tensor,
    compute_dp_cache_schedule,
)


@pytest.fixture(params=["flux", "wan"])
def model_and_inputs(request):
    torch.manual_seed(0)
    if request.param == "flux":
        model = FluxTransformer2DModel(
            patch_size=1,
            in_channels=4,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=16,
            pooled_projection_dim=16,
            axes_dims_rope=(4, 4, 8),
        )
        inputs = {
            "hidden_states": torch.randn(1, 4, 4),
            "encoder_hidden_states": torch.randn(1, 3, 16),
            "pooled_projections": torch.randn(1, 16),
            "img_ids": torch.zeros(4, 3),
            "txt_ids": torch.zeros(3, 3),
            "timestep": torch.tensor([0.5]),
        }
    else:
        model = WanTransformer3DModel(
            patch_size=(1, 2, 2),
            in_channels=4,
            out_channels=4,
            num_layers=2,
            attention_head_dim=12,
            num_attention_heads=2,
            text_dim=16,
            freq_dim=16,
            ffn_dim=32,
            rope_max_seq_len=16,
        )
        inputs = {
            "hidden_states": torch.randn(1, 4, 1, 4, 4),
            "encoder_hidden_states": torch.randn(1, 3, 16),
            "timestep": torch.tensor([500.0]),
        }
    return model.eval(), inputs


@pytest.mark.parametrize("order", [0, 1, 2])
def test_nonuniform_prediction(order):
    factors = []
    previous = -1
    for step, value in [(0, 1.0), (1, 3.0), (4, 13.0)]:
        factors = _update_factors(torch.tensor([value]), factors, step - previous, order)
        previous = step
    expected = 13.0
    if order >= 1:
        expected += 2 * (10 / 3)
    if order == 2:
        expected += 2 * ((10 / 3 - 2) / 3)
    torch.testing.assert_close(_predict(factors, 2), torch.tensor([expected]))
    assert factors[0].item() == 13.0


def test_cost_tensor_endpoints_and_sentinel():
    features = [torch.tensor([value]) for value in [0.0, 1.0, 4.0, 9.0]]
    costs = build_dp_cache_cost_tensor(features, max_order=1)
    assert costs.shape == (5, 5, 5)
    assert costs[0, 1, 2] == 2
    assert costs[0, 1, 3] == 8
    assert costs[0, 1, 4] == 18
    assert costs[1, 3, 4] == 1
    assert torch.isinf(costs[2, 1, 3])


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("warmup", [2, 3])
def test_schedule_matches_exhaustive_search(seed, warmup):
    total, budget = 8, 5
    generator = torch.Generator().manual_seed(seed)
    costs = torch.rand((total + 1,) * 3, generator=generator, dtype=torch.float64)

    def path_cost(path):
        path = [*path, total]
        return sum(costs[path[i - 1], path[i], path[i + 1]].item() for i in range(warmup - 1, budget))

    candidates = [[*range(warmup), *tail] for tail in itertools.combinations(range(warmup, total), budget - warmup)]
    expected = min(candidates, key=path_cost)
    actual = compute_dp_cache_schedule(costs, budget, warmup)
    assert actual == expected


@pytest.mark.parametrize("budget", [3, 6])
def test_schedule_boundary_budgets(budget):
    costs = build_dp_cache_cost_tensor([torch.tensor([float(i)]) for i in range(6)])
    schedule = compute_dp_cache_schedule(costs, budget)
    assert len(schedule) == budget
    assert schedule[:3] == [0, 1, 2]
    assert len(set(schedule)) == budget
    assert max(schedule) < 6


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"compute_steps": []},
        {"compute_steps": [0, 1, 2, 2]},
        {"compute_steps": [0, 1, 3]},
        {"compute_steps": [0, 2, 1]},
        {"compute_steps": [0, 1, 2, 6]},
        {"compute_steps": [False, 1, 2]},
        {"compute_steps": [0, 1, 2], "calibrate": True},
        {"calibrate": True, "max_order": 3},
        {"calibrate": True, "max_order": 1.0},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        DPCacheConfig(num_inference_steps=6, **kwargs)


def test_invalid_calibration_and_schedule():
    with pytest.raises(ValueError, match="complete calibration"):
        DPCacheConfig(6, calibrate=True).compute_schedule(3)
    with pytest.raises(ValueError, match="No finite schedule"):
        compute_dp_cache_schedule(torch.full((7, 7, 7), torch.inf), 3)
    with pytest.raises(ValueError, match="finite"):
        build_dp_cache_cost_tensor([torch.tensor([float("nan")])] * 3)
    with pytest.raises(ValueError, match="shape"):
        compute_dp_cache_schedule(torch.zeros(6, 6), 3)


@torch.no_grad()
def test_all_compute_matches_uncached_and_disable_restores(model_and_inputs):
    model, inputs = model_and_inputs
    blocks = (
        list(model.transformer_blocks) + list(model.single_transformer_blocks)
        if hasattr(model, "transformer_blocks")
        else list(model.blocks)
    )
    original = [block.forward for block in blocks]
    expected = model(**inputs).sample
    model.enable_cache(DPCacheConfig(6, compute_steps=list(range(6))))
    for step in range(6):
        with model.cache_context("cond", step_index=step, num_inference_steps=6):
            torch.testing.assert_close(model(**inputs).sample, expected, rtol=0, atol=0)
    model.disable_cache()
    assert not model.is_cache_enabled
    assert [block.forward for block in blocks] == original
    assert all("compile" not in block.__dict__ for block in blocks)
    torch.testing.assert_close(model(**inputs).sample, expected, rtol=0, atol=0)


@torch.no_grad()
def test_cache_skips_stack_and_separates_contexts(model_and_inputs):
    model, inputs = model_and_inputs
    block = model.transformer_blocks[0] if hasattr(model, "transformer_blocks") else model.blocks[0]
    calls = []
    attention = block.attn if hasattr(block, "attn") else block.attn1
    handle = attention.register_forward_hook(lambda *args: calls.append(1))
    model.enable_cache(DPCacheConfig(6, compute_steps=[0, 1, 2, 4]))
    outputs = {}
    for step in range(6):
        for name, offset in [("cond", 0.0), ("uncond", 2.0)]:
            current_inputs = {**inputs, "hidden_states": inputs["hidden_states"] + offset}
            with model.cache_context(name, step_index=step):
                outputs[name] = model(**current_inputs).sample
            assert torch.isfinite(outputs[name]).all()
    handle.remove()
    assert len(calls) == 8
    assert not torch.allclose(outputs["cond"], outputs["uncond"])
    states = model._diffusers_hook.get_hook(_DP_CACHE_HOOK).state_manager._state_cache
    assert states["cond"].last_compute_step == states["uncond"].last_compute_step == 4
    assert states["cond"].factors[0].data_ptr() != states["uncond"].factors[0].data_ptr()


@torch.no_grad()
def test_reset_and_reconfigure(model_and_inputs):
    model, inputs = model_and_inputs
    model.enable_cache(DPCacheConfig(6, compute_steps=[0, 1, 2]))
    first_run = []
    for _ in range(2):
        outputs = []
        for step in range(6):
            with model.cache_context("cond"):
                outputs.append(model(**inputs).sample)
        if first_run:
            torch.testing.assert_close(outputs, first_run, rtol=0, atol=0)
        first_run = outputs
        model._reset_stateful_cache()
    model.disable_cache()
    model.enable_cache(DPCacheConfig(6, compute_steps=list(range(6))))
    with model.cache_context("cond", step_index=0):
        torch.testing.assert_close(model(**inputs).sample, first_run[0], rtol=0, atol=0)


def test_autograd_bypasses_cache(model_and_inputs):
    model, inputs = model_and_inputs
    model.enable_cache(DPCacheConfig(6, compute_steps=[0, 1, 2]))
    model.enable_gradient_checkpointing()
    model(**inputs).sample.sum().backward()
    assert model.proj_out.weight.grad is not None
    assert not model._diffusers_hook.get_hook(_DP_CACHE_HOOK).state_manager._state_cache


@torch.no_grad()
def test_compile_disable_preserves_model_structure(model_and_inputs):
    model, inputs = model_and_inputs
    keys = set(model.state_dict())
    modules = dict(model.named_modules())
    expected = model(**inputs).sample
    model.enable_cache(DPCacheConfig(6, compute_steps=[0, 1, 2]))
    model.compile_repeated_blocks(backend="eager", fullgraph=True)
    with model.cache_context("cond", step_index=0):
        torch.testing.assert_close(model(**inputs).sample, expected)
    model.disable_cache()
    assert set(model.state_dict()) == keys
    assert dict(model.named_modules()) == modules
    assert all("compile" not in module.__dict__ for module in model.modules())
    torch.testing.assert_close(model(**inputs).sample, expected)
    model.compile_repeated_blocks(backend="eager", fullgraph=True)
    torch.testing.assert_close(model(**inputs).sample, expected)
    torch._dynamo.reset()


def test_enable_after_compile_rejected_without_partial_hooks(model_and_inputs):
    model, _ = model_and_inputs
    model.compile_repeated_blocks(backend="eager", fullgraph=True)
    with pytest.raises(ValueError, match="before compiling"):
        model.enable_cache(DPCacheConfig(6, compute_steps=[0, 1, 2]))
    assert not model.is_cache_enabled
    assert all(not hasattr(module, "_diffusers_hook") for module in model.modules())
    torch._dynamo.reset()


@torch.no_grad()
def test_calibration_preserves_outputs_and_aggregates(model_and_inputs):
    model, inputs = model_and_inputs
    baseline = model(**inputs).sample
    config = DPCacheConfig(6, calibrate=True)
    model.enable_cache(config)
    for run in range(2):
        for step in range(6):
            with model.cache_context("cond", step_index=step):
                torch.testing.assert_close(model(**inputs).sample, baseline, rtol=0, atol=0)
        assert config.num_calibration_samples == run + 1
    assert config.cost_tensor.shape == (7, 7, 7)
    assert config.cost_tensor.device.type == "cpu"
    schedule = config.compute_schedule(4)
    assert len(schedule) == 4
    state = model._diffusers_hook.get_hook(_DP_CACHE_HOOK).state_manager._state_cache["cond"]
    assert not state.features
    model.disable_cache()
    model.enable_cache(DPCacheConfig(6, compute_steps=schedule))
    for step in range(6):
        with model.cache_context("cond", step_index=step):
            assert torch.isfinite(model(**inputs).sample).all()


@torch.no_grad()
def test_invalid_runtime_context(model_and_inputs):
    model, inputs = model_and_inputs
    model.enable_cache(DPCacheConfig(6, compute_steps=[0, 1, 2]))
    with pytest.raises(ValueError, match="No cache context"):
        model(**inputs)
    with model.cache_context("cond", step_index=0, num_inference_steps=7):
        with pytest.raises(ValueError, match="different number"):
            model(**inputs)
    with model.cache_context("cond", step_index=3):
        with pytest.raises(ValueError, match="consecutive steps"):
            model(**inputs)


@torch.no_grad()
def test_regional_compile_fullgraph_without_recompile(model_and_inputs):
    model, inputs = model_and_inputs
    model.enable_cache(DPCacheConfig(6, compute_steps=[0, 1, 2, 4]))
    graphs = []

    def backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    torch._dynamo.reset()
    model.compile_repeated_blocks(backend=backend, fullgraph=True)
    with model.cache_context("cond", step_index=0):
        first = model(**inputs).sample
    assert graphs
    num_graphs = len(graphs)
    with torch._dynamo.config.patch(error_on_recompile=True):
        for step in range(1, 6):
            with model.cache_context("cond", step_index=step):
                output = model(**{**inputs, "hidden_states": inputs["hidden_states"] + step * 0.01}).sample
            assert output.shape == first.shape
            assert torch.isfinite(output).all()
    assert len(graphs) == num_graphs
    torch._dynamo.reset()


@torch.no_grad()
def test_pipeline_calibration_and_repeated_generation(model_and_inputs):
    model, inputs = model_and_inputs
    scheduler = FlowMatchEulerDiscreteScheduler()
    kwargs = {
        "prompt_embeds": inputs["encoder_hidden_states"],
        "num_inference_steps": 6,
        "height": 16,
        "width": 16,
        "output_type": "latent",
    }
    if isinstance(model, FluxTransformer2DModel):
        vae = AutoencoderKL(
            block_out_channels=(8,),
            down_block_types=("DownEncoderBlock2D",),
            up_block_types=("UpDecoderBlock2D",),
            latent_channels=1,
            norm_num_groups=4,
        )
        pipe = FluxPipeline(scheduler, vae, None, None, None, None, model)
        kwargs.update(pooled_prompt_embeds=inputs["pooled_projections"], guidance_scale=1.0)
    else:
        vae = AutoencoderKLWan(
            base_dim=4,
            z_dim=4,
            dim_mult=[1, 1, 1, 1],
            num_res_blocks=1,
            latents_mean=[0.0] * 4,
            latents_std=[1.0] * 4,
        )
        pipe = WanPipeline(None, None, vae, scheduler, model)
        kwargs.update(num_frames=1, negative_prompt_embeds=-inputs["encoder_hidden_states"], guidance_scale=3.0)
    pipe.set_progress_bar_config(disable=True)

    def generate():
        return pipe(**kwargs, generator=torch.Generator().manual_seed(0))[0]

    baseline = generate()
    calibration = DPCacheConfig(6, calibrate=True)
    model.enable_cache(calibration)
    torch.testing.assert_close(generate(), baseline, rtol=0, atol=0)
    assert calibration.num_calibration_samples == (1 if isinstance(model, FluxTransformer2DModel) else 2)
    schedule = calibration.compute_schedule(4)
    model.disable_cache()
    model.enable_cache(DPCacheConfig(6, compute_steps=schedule))
    first, second = generate(), generate()
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.isfinite(first).all()
    assert first.shape == baseline.shape
    state_manager = model._diffusers_hook.get_hook(_DP_CACHE_HOOK).state_manager
    assert not state_manager._state_cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.compile
@torch.no_grad()
def test_cuda_inductor_matches_eager_cache(model_and_inputs):
    model, inputs = model_and_inputs
    model.to(device="cuda", dtype=torch.bfloat16)
    inputs = {key: value.to(device="cuda", dtype=torch.bfloat16) for key, value in inputs.items()}
    model.enable_cache(DPCacheConfig(6, compute_steps=[0, 1, 2, 4]))

    def run():
        outputs = []
        for step in range(6):
            with model.cache_context("cond", step_index=step):
                outputs.append(model(**inputs).sample.clone())
        return outputs

    expected = run()
    torch._dynamo.reset()
    model.compile_repeated_blocks(fullgraph=True)
    with model.cache_context("cond", step_index=0):
        model(**inputs)
    with torch._dynamo.config.patch(error_on_recompile=True):
        actual = run()
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)
    torch._dynamo.reset()
