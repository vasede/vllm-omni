# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for Wan2.2 TeaCache support.

Covers:
- Polynomial coefficients correctness (WanTransformer3DModel)
- _teacache_init_loop_state / _teacache_should_compute logic
- 3D (T2V/I2V) vs 4D (TI2V) timestep handling in extract_wan2_2_context
- FSDP unshard/reshard path (mocked)
- Edge cases: rel_l1_thresh exceeded mid-sequence, first-step always-compute
- TeaCache on vs off consistency (CPU, mock denoising loop)
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.cache.teacache.backend import (
    _teacache_init_loop_state,
    _teacache_should_compute,
    enable_wan2_2_teacache,
)
from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig, _MODEL_COEFFICIENTS
from vllm_omni.diffusion.cache.teacache.extractors import extract_wan2_2_context
from vllm_omni.diffusion.data import DiffusionCacheConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline(rel_l1_thresh: float = 0.2) -> MagicMock:
    pipeline = MagicMock()
    pipeline._tea_cache_config = TeaCacheConfig(
        transformer_type="WanTransformer3DModel",
        rel_l1_thresh=rel_l1_thresh,
    )
    return pipeline


# ---------------------------------------------------------------------------
# 1. Polynomial coefficients
# ---------------------------------------------------------------------------

class TestWan22Coefficients:
    """Verify WanTransformer3DModel polynomial coefficients are registered and valid."""

    def test_coefficients_registered(self):
        assert "WanTransformer3DModel" in _MODEL_COEFFICIENTS

    def test_coefficients_length(self):
        coeffs = _MODEL_COEFFICIENTS["WanTransformer3DModel"]
        assert len(coeffs) == 5

    def test_coefficients_are_finite(self):
        coeffs = _MODEL_COEFFICIENTS["WanTransformer3DModel"]
        for c in coeffs:
            assert np.isfinite(c), f"Non-finite coefficient: {c}"

    def test_polynomial_monotone_on_typical_range(self):
        """Polynomial should be roughly monotone on [0.0, 0.5] (at most 3 inflections)."""
        coeffs = _MODEL_COEFFICIENTS["WanTransformer3DModel"]
        poly = np.poly1d(coeffs)
        xs = np.linspace(0.0, 0.5, 50)
        ys = poly(xs)
        sign_changes = int(np.sum(np.diff(np.sign(np.diff(ys))) != 0))
        assert sign_changes <= 3, (
            f"Polynomial has {sign_changes} inflection points on [0, 0.5]"
        )

    def test_teacache_config_uses_wan_coefficients(self):
        cfg = TeaCacheConfig(transformer_type="WanTransformer3DModel")
        assert cfg.coefficients == _MODEL_COEFFICIENTS["WanTransformer3DModel"]

    def test_teacache_config_custom_coefficients_override(self):
        custom = [1.0, 2.0, 3.0, 4.0, 5.0]
        cfg = TeaCacheConfig(transformer_type="WanTransformer3DModel", coefficients=custom)
        assert cfg.coefficients == custom

    def test_teacache_config_invalid_rel_l1_thresh(self):
        with pytest.raises(ValueError, match="rel_l1_thresh must be positive"):
            TeaCacheConfig(transformer_type="WanTransformer3DModel", rel_l1_thresh=0.0)

    def test_teacache_config_wrong_coeff_length(self):
        with pytest.raises(ValueError, match="coefficients must contain exactly 5"):
            TeaCacheConfig(
                transformer_type="WanTransformer3DModel",
                coefficients=[1.0, 2.0, 3.0],
            )


# ---------------------------------------------------------------------------
# 2. Loop state init and should_compute logic
# ---------------------------------------------------------------------------

class TestTeaCacheLoopState:
    """Unit tests for _teacache_init_loop_state and _teacache_should_compute."""

    def test_init_returns_none_when_no_config(self):
        pipeline = MagicMock(spec=[])
        state = _teacache_init_loop_state(pipeline)
        assert state is None

    def test_init_returns_state_dict(self):
        state = _teacache_init_loop_state(_make_pipeline())
        assert state is not None
        for key in ("config", "rescale", "acc_dist", "prev_modulated_input",
                    "prev_noise_pred", "cnt"):
            assert key in state

    def test_init_acc_dist_zero(self):
        assert _teacache_init_loop_state(_make_pipeline())["acc_dist"] == 0.0

    def test_init_cnt_zero(self):
        assert _teacache_init_loop_state(_make_pipeline())["cnt"] == 0

    def test_should_compute_none_state(self):
        assert _teacache_should_compute(None, torch.randn(1, 16, 32)) is True

    def test_should_compute_none_modulated_input(self):
        state = _teacache_init_loop_state(_make_pipeline())
        assert _teacache_should_compute(state, None) is True

    def test_first_step_always_computes(self):
        """cnt=0 must always return True regardless of input similarity."""
        state = _teacache_init_loop_state(_make_pipeline())
        result = _teacache_should_compute(state, torch.zeros(1, 16, 32))
        assert result is True

    def test_prev_modulated_input_stored_after_first_step(self):
        state = _teacache_init_loop_state(_make_pipeline())
        mod_input = torch.randn(1, 16, 32)
        _teacache_should_compute(state, mod_input)
        assert state["prev_modulated_input"] is not None
        assert state["prev_modulated_input"].shape == mod_input.shape

    def test_identical_inputs_may_skip_compute(self):
        """Identical consecutive inputs should eventually trigger cache reuse."""
        state = _teacache_init_loop_state(_make_pipeline(rel_l1_thresh=0.5))
        mod_input = torch.ones(1, 16, 32)
        _teacache_should_compute(state, mod_input)
        state["cnt"] += 1

        skipped = False
        for _ in range(20):
            result = _teacache_should_compute(state, mod_input.clone())
            if not result:
                skipped = True
                break
            state["cnt"] += 1
        assert skipped, "Identical inputs should trigger cache reuse within 20 steps"

    def test_large_change_resets_acc_dist(self):
        """When threshold exceeded, acc_dist resets to 0."""
        state = _teacache_init_loop_state(_make_pipeline(rel_l1_thresh=0.01))
        prev = torch.ones(1, 16, 32)
        _teacache_should_compute(state, prev)
        state["cnt"] += 1

        very_different = torch.ones(1, 16, 32) * 1000.0
        result = _teacache_should_compute(state, very_different)
        assert result is True
        assert state["acc_dist"] == 0.0

    def test_very_high_threshold_uses_cache(self):
        """With very high threshold, step 1 should use cache (acc_dist < thresh)."""
        state = _teacache_init_loop_state(_make_pipeline(rel_l1_thresh=1e6))
        mod_input = torch.ones(1, 4, 8) * 0.5
        _teacache_should_compute(state, mod_input)
        state["cnt"] += 1
        result = _teacache_should_compute(state, mod_input.clone())
        assert result is False, "Very high threshold should use cache on step 1"


# ---------------------------------------------------------------------------
# 3. enable_wan2_2_teacache
# ---------------------------------------------------------------------------

class TestEnableWan22TeaCache:

    def test_attaches_tea_cache_config(self):
        pipeline = MagicMock()
        enable_wan2_2_teacache(pipeline, DiffusionCacheConfig(rel_l1_thresh=0.3))
        assert isinstance(pipeline._tea_cache_config, TeaCacheConfig)

    def test_config_rel_l1_thresh_propagated(self):
        pipeline = MagicMock()
        enable_wan2_2_teacache(pipeline, DiffusionCacheConfig(rel_l1_thresh=0.42))
        assert pipeline._tea_cache_config.rel_l1_thresh == pytest.approx(0.42)

    def test_config_transformer_type(self):
        pipeline = MagicMock()
        enable_wan2_2_teacache(pipeline, DiffusionCacheConfig())
        assert pipeline._tea_cache_config.transformer_type == "WanTransformer3DModel"

    def test_custom_coefficients_propagated(self):
        pipeline = MagicMock()
        custom = [1.0, 2.0, 3.0, 4.0, 5.0]
        enable_wan2_2_teacache(pipeline, DiffusionCacheConfig(coefficients=custom))
        assert pipeline._tea_cache_config.coefficients == custom

    def test_wan22_pipeline_in_custom_enablers(self):
        from vllm_omni.diffusion.cache.teacache.backend import CUSTOM_TEACACHE_ENABLERS
        assert "Wan22Pipeline" in CUSTOM_TEACACHE_ENABLERS
        assert "Wan22I2VPipeline" in CUSTOM_TEACACHE_ENABLERS
        assert CUSTOM_TEACACHE_ENABLERS["Wan22Pipeline"] is enable_wan2_2_teacache
        assert CUSTOM_TEACACHE_ENABLERS["Wan22I2VPipeline"] is enable_wan2_2_teacache


# ---------------------------------------------------------------------------
# 4. extract_wan2_2_context — timestep shape and FSDP
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Minimal CPU-only mock for WanTransformer3DModel
# ---------------------------------------------------------------------------

class _MockAdaLayerNorm(nn.Module):
    """CPU-safe AdaLayerNorm substitute: ignores scale/shift, returns input."""

    def forward(self, hidden_states: torch.Tensor, scale=None, shift=None) -> torch.Tensor:
        return hidden_states


class _MockWanBlock(nn.Module):
    """Minimal WanTransformerBlock for CPU unit tests."""

    def __init__(self, inner_dim: int):
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.zeros(1, 6, inner_dim))
        self.norm1 = _MockAdaLayerNorm()


class _MockPatchEmbedding(nn.Module):
    """Returns a fixed-shape 5D tensor regardless of input."""

    def __init__(self, inner_dim: int, spatial_out: int = 2):
        super().__init__()
        self._inner_dim = inner_dim
        self._spatial_out = spatial_out

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B = hidden_states.shape[0]
        return torch.randn(B, self._inner_dim, 1, self._spatial_out, self._spatial_out)


class _MockConditionEmbedder(nn.Module):
    """Returns (temb, timestep_proj, None, None) with correct shapes."""

    def __init__(self, inner_dim: int):
        super().__init__()
        self._inner_dim = inner_dim

    def forward(self, timestep_flat, encoder_hidden_states, encoder_hidden_states_image,
                timestep_seq_len=None):
        B = encoder_hidden_states.shape[0]
        D = self._inner_dim
        if timestep_seq_len is not None:
            temb = torch.randn(B, timestep_seq_len, D)
            ts_proj = torch.randn(B, timestep_seq_len, 6 * D)
        else:
            temb = torch.randn(B, D)
            ts_proj = torch.randn(B, 6 * D)
        return temb, ts_proj, None, None


class _MockTimestepProjPrepare(nn.Module):
    """Reshapes timestep_proj for block modulation."""

    def __init__(self, inner_dim: int):
        super().__init__()
        self._inner_dim = inner_dim

    def forward(self, ts_proj: torch.Tensor, ts_seq_len=None) -> torch.Tensor:
        D = self._inner_dim
        B = ts_proj.shape[0]
        if ts_seq_len is not None:
            return ts_proj.view(B, ts_seq_len, 6, D)
        return ts_proj.view(B, 6, D)


class _MockWanModule(nn.Module):
    """
    CPU-safe mock of WanTransformer3DModel.

    Provides the interface used by extract_wan2_2_context without any NPU ops.
    """

    def __init__(self, inner_dim: int = 16, num_blocks: int = 2):
        super().__init__()
        self.patch_embedding = _MockPatchEmbedding(inner_dim)
        self.condition_embedder = _MockConditionEmbedder(inner_dim)
        self.timestep_proj_prepare = _MockTimestepProjPrepare(inner_dim)
        self.blocks = nn.ModuleList([_MockWanBlock(inner_dim) for _ in range(num_blocks)])
        self._inner_dim = inner_dim


def _make_mock_wan_module(inner_dim: int = 16, image_dim: int | None = None) -> _MockWanModule:
    """Return a CPU-safe mock WanTransformer3DModel."""
    return _MockWanModule(inner_dim=inner_dim)


class TestExtractWan22Context:
    """
    Unit tests for extract_wan2_2_context.

    Uses a mock WanTransformer3DModel that runs on CPU without NPU ops.
    Tests 3D timestep (T2V/I2V) and 4D timestep (TI2V) paths.
    """

    INNER_DIM = 16

    @pytest.fixture
    def tiny_wan_module(self):
        return _make_mock_wan_module(inner_dim=self.INNER_DIM)

    @pytest.fixture
    def tiny_wan_i2v_module(self):
        return _make_mock_wan_module(inner_dim=self.INNER_DIM, image_dim=8)

    def _t2v_inputs(self):
        return {
            "hidden_states": torch.randn(1, 4, 1, 4, 4),
            "timestep": torch.tensor([500]),
            "encoder_hidden_states": torch.randn(1, 4, 32),
            "encoder_hidden_states_image": None,
        }

    def _ti2v_inputs(self):
        return {
            "hidden_states": torch.randn(1, 4, 1, 4, 4),
            "timestep": torch.randint(0, 1000, (1, 4)),
            "encoder_hidden_states": torch.randn(1, 4, 32),
            "encoder_hidden_states_image": torch.randn(1, 4, 16),
        }

    def test_invalid_module_raises(self):
        bad_module = MagicMock()
        bad_module.blocks = []
        with pytest.raises(ValueError, match="Module must have blocks"):
            extract_wan2_2_context(
                bad_module,
                hidden_states=torch.randn(1, 4, 1, 4, 4),
                timestep=torch.tensor([500]),
                encoder_hidden_states=torch.randn(1, 4, 32),
            )

    def test_3d_timestep_modulated_input_shape(self, tiny_wan_module):
        """3D timestep (T2V): modulated_input is [B, S, inner_dim]."""
        ctx = extract_wan2_2_context(tiny_wan_module, **self._t2v_inputs())
        assert ctx.modulated_input.ndim == 3
        assert ctx.modulated_input.shape[0] == 1
        assert ctx.modulated_input.shape[2] == self.INNER_DIM

    def test_4d_timestep_modulated_input_shape(self, tiny_wan_i2v_module):
        """4D timestep (TI2V): modulated_input is [B, S, inner_dim]."""
        ctx = extract_wan2_2_context(tiny_wan_i2v_module, **self._ti2v_inputs())
        assert ctx.modulated_input.ndim == 3
        assert ctx.modulated_input.shape[0] == 1
        assert ctx.modulated_input.shape[2] == self.INNER_DIM

    def test_modulated_input_no_grad(self, tiny_wan_module):
        """modulated_input must not require grad (no memory leak across steps)."""
        ctx = extract_wan2_2_context(tiny_wan_module, **self._t2v_inputs())
        assert not ctx.modulated_input.requires_grad

    def test_run_transformer_blocks_is_none(self, tiny_wan_module):
        """Pipeline-level TeaCache: run_transformer_blocks is None."""
        ctx = extract_wan2_2_context(tiny_wan_module, **self._t2v_inputs())
        assert ctx.run_transformer_blocks is None

    def test_postprocess_is_none(self, tiny_wan_module):
        """Pipeline-level TeaCache: postprocess is None."""
        ctx = extract_wan2_2_context(tiny_wan_module, **self._t2v_inputs())
        assert ctx.postprocess is None

    def test_temb_is_not_none(self, tiny_wan_module):
        ctx = extract_wan2_2_context(tiny_wan_module, **self._t2v_inputs())
        assert ctx.temb is not None

    def test_no_image_encoder_hidden_states_t2v(self, tiny_wan_module):
        """T2V: encoder_hidden_states_image=None is handled without error."""
        ctx = extract_wan2_2_context(tiny_wan_module, **self._t2v_inputs())
        assert ctx is not None

    def test_fsdp_unshard_called_when_fsdp_module(self, tiny_wan_module):
        """FSDP path: unshard() is called on root module when it is an FSDPModule."""
        try:
            from torch.distributed.fsdp._fully_shard._fully_shard import FSDPModule
        except ImportError:
            pytest.skip("FSDPModule not available in this PyTorch version")

        # Wrap the real mock module in a MagicMock that looks like FSDPModule
        mock_module = MagicMock(spec=FSDPModule)
        mock_module.blocks = tiny_wan_module.blocks
        mock_module.patch_embedding = tiny_wan_module.patch_embedding
        mock_module.condition_embedder = tiny_wan_module.condition_embedder
        mock_module.timestep_proj_prepare = tiny_wan_module.timestep_proj_prepare

        try:
            extract_wan2_2_context(mock_module, **self._t2v_inputs())
        except Exception:
            pass  # May fail due to mock internals; we only verify unshard was called
        mock_module.unshard.assert_called()


# ---------------------------------------------------------------------------
# 5. TeaCache on vs off consistency (mock denoising loop)
# ---------------------------------------------------------------------------

class TestTeaCacheConsistency:
    """
    Verify TeaCache produces consistent outputs with TeaCache disabled.

    Uses a mock denoising loop that mirrors the pipeline-level caching logic
    from pipeline_wan2_2.py / pipeline_wan2_2_i2v.py without requiring NPU.
    """

    def _run_loop(self, noise_fn, num_steps, rel_l1_thresh):
        pipeline = MagicMock()
        if rel_l1_thresh is not None:
            pipeline._tea_cache_config = TeaCacheConfig(
                transformer_type="WanTransformer3DModel",
                rel_l1_thresh=rel_l1_thresh,
            )
        else:
            type(pipeline)._tea_cache_config = property(lambda self: None)

        _tc_state = _teacache_init_loop_state(pipeline)
        results = []
        for step in range(num_steps):
            mod_input = torch.ones(1, 16, 32) * (1.0 + step * 0.001)
            if _teacache_should_compute(_tc_state, mod_input):
                noise_pred = noise_fn(step)
                if _tc_state is not None:
                    _tc_state["prev_noise_pred"] = noise_pred
            else:
                noise_pred = _tc_state["prev_noise_pred"]
            results.append(noise_pred.clone())
            if _tc_state is not None:
                _tc_state["cnt"] += 1
        return results

    def test_disabled_all_steps_computed(self):
        """With TeaCache disabled, every step calls noise_fn."""
        calls = [0]

        def noise_fn(step):
            calls[0] += 1
            return torch.randn(1, 4, 8)

        self._run_loop(noise_fn, 10, None)
        assert calls[0] == 10

    def test_enabled_skips_some_steps(self):
        """With slowly-changing inputs and moderate threshold, some steps are skipped."""
        calls = [0]

        def noise_fn(step):
            calls[0] += 1
            return torch.ones(1, 4, 8) * float(step)

        self._run_loop(noise_fn, 20, 0.3)
        assert calls[0] < 20, "TeaCache should skip some steps"

    def test_very_low_threshold_matches_no_cache(self):
        """With very low threshold, acc_dist always exceeds it so cache is never used.

        _teacache_should_compute returns False (use cache) when acc_dist < rel_l1_thresh.
        With rel_l1_thresh=1e-10, abs(poly(rel_l1)) >> 1e-10 for any non-trivial input,
        so every step recomputes and outputs match the no-cache run exactly.
        """
        # Use a fixed sequence so both runs produce identical noise_pred values
        noise_seq = [torch.ones(1, 4, 8) * float(i) for i in range(10)]

        def noise_fn(step):
            return noise_seq[step].clone()

        no_cache = self._run_loop(noise_fn, 10, None)
        with_cache = self._run_loop(noise_fn, 10, 1e-10)

        for i, (a, b) in enumerate(zip(no_cache, with_cache)):
            assert torch.allclose(a, b, atol=1e-5), f"Step {i} mismatch"

    def test_first_step_always_matches(self):
        """Step 0 output must always match no-cache output."""
        def noise_fn(step):
            return torch.ones(1, 4, 8) * float(step)

        no_cache = self._run_loop(noise_fn, 5, None)
        with_cache = self._run_loop(noise_fn, 5, 0.01)
        assert torch.allclose(no_cache[0], with_cache[0]), "Step 0 must always be computed"
