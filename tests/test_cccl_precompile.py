"""Tests for CCCL pre-compilation warmup infrastructure (ADR-0034)."""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from vibespatial.cccl_precompile import (
    PRECOMPILE_ENV_VAR,
    SPEC_REGISTRY,
    CCCLPrecompiler,
    precompile_enabled,
    precompile_status,
    request_warmup,
)
from vibespatial.nvrtc_precompile import NVRTCPrecompiler


@pytest.fixture(autouse=True)
def _reset_precompilers():
    """Reset both singletons between tests."""
    CCCLPrecompiler._reset()
    NVRTCPrecompiler._reset()
    yield
    CCCLPrecompiler._reset()
    NVRTCPrecompiler._reset()


# ---------------------------------------------------------------------------
# Toggle
# ---------------------------------------------------------------------------

class TestPrecompileEnabled:
    def test_enabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop(PRECOMPILE_ENV_VAR, None)
            assert precompile_enabled() is True

    def test_empty_string_is_enabled(self):
        with patch.dict(os.environ, {PRECOMPILE_ENV_VAR: ""}):
            assert precompile_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "off", "no", "False", "OFF"])
    def test_disabled_values(self, value: str):
        with patch.dict(os.environ, {PRECOMPILE_ENV_VAR: value}):
            assert precompile_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "on", "yes", "anything"])
    def test_enabled_values(self, value: str):
        with patch.dict(os.environ, {PRECOMPILE_ENV_VAR: value}):
            assert precompile_enabled() is True


# ---------------------------------------------------------------------------
# Spec registry
# ---------------------------------------------------------------------------

class TestSpecRegistry:
    def test_registry_has_20_specs(self):
        assert len(SPEC_REGISTRY) == 20

    def test_all_expected_specs_present(self):
        expected = {
            "exclusive_scan_i32", "exclusive_scan_i64",
            "select_i32", "select_i64",
            "reduce_sum_f64", "reduce_sum_i32",
            "segmented_reduce_sum_f64",
            "segmented_reduce_min_f64",
            "segmented_reduce_max_f64",
            "lower_bound_i32", "lower_bound_u64",
            "upper_bound_i32", "upper_bound_u64",
            "radix_sort_i32_i32", "radix_sort_u64_i32",
            "merge_sort_u64_i32",
            "unique_by_key_i32_i32", "unique_by_key_u64_i32",
            "segmented_sort_asc_f64", "segmented_sort_asc_i32",
        }
        assert set(SPEC_REGISTRY.keys()) == expected

    def test_spec_names_match_keys(self):
        for key, spec in SPEC_REGISTRY.items():
            assert spec.name == key


# ---------------------------------------------------------------------------
# Precompiler logic (no GPU needed)
# ---------------------------------------------------------------------------

class TestCCCLPrecompilerNoGPU:
    def test_request_warmup_noop_without_gpu(self):
        with patch("vibespatial.runtime.has_gpu_runtime", return_value=False):
            request_warmup(["exclusive_scan_i32"])
        assert CCCLPrecompiler._instance is None

    def test_request_warmup_noop_when_disabled(self):
        with patch.dict(os.environ, {PRECOMPILE_ENV_VAR: "0"}):
            request_warmup(["exclusive_scan_i32"])
        assert CCCLPrecompiler._instance is None

    def test_precompile_status_empty_when_no_warmup(self):
        assert precompile_status() == {}

    def test_deduplication(self):
        precompiler = CCCLPrecompiler(max_workers=1)
        with patch.object(precompiler, "_compile_one", return_value=None):
            precompiler.request(["exclusive_scan_i32", "exclusive_scan_i64"])
            precompiler.request(["exclusive_scan_i32", "select_i32"])

        assert len(precompiler._submitted) == 3
        assert "exclusive_scan_i32" in precompiler._submitted
        assert "exclusive_scan_i64" in precompiler._submitted
        assert "select_i32" in precompiler._submitted

    def test_unknown_spec_skipped(self):
        precompiler = CCCLPrecompiler(max_workers=1)
        precompiler.request(["nonexistent_spec"])
        assert "nonexistent_spec" not in precompiler._submitted

    def test_get_compiled_returns_none_for_unknown(self):
        precompiler = CCCLPrecompiler(max_workers=1)
        assert precompiler.get_compiled("nonexistent_spec") is None

    def test_status_shape(self):
        precompiler = CCCLPrecompiler(max_workers=1)
        status = precompiler.status()
        assert "submitted" in status
        assert "compiled" in status
        assert "pending" in status
        assert "failed" in status
        assert "wall_ms" in status
        assert "per_primitive" in status
        assert isinstance(status["per_primitive"], list)
