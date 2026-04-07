"""Tests for CCCL pre-compilation warmup infrastructure (ADR-0034)."""
from __future__ import annotations

import os
import warnings
from unittest.mock import patch

import pytest

from vibespatial.cuda.cccl_precompile import (
    PRECOMPILE_ENV_VAR,
    SPEC_REGISTRY,
    CCCLPrecompiler,
    _warm_many_vs_one_overlay_remainder_route,
    ensure_pipelines_warm,
    precompile_enabled,
    precompile_status,
    request_warmup,
)
from vibespatial.cuda.nvrtc_precompile import NVRTCPrecompiler
from vibespatial.runtime import has_gpu_runtime


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
    def test_registry_has_23_specs(self):
        assert len(SPEC_REGISTRY) == 23

    def test_all_expected_specs_present(self):
        expected = {
            "exclusive_scan_i32", "exclusive_scan_i64",
            "select_i32", "select_i64",
            "reduce_sum_f64", "reduce_sum_i32",
            "segmented_reduce_sum_f64", "segmented_reduce_sum_i32",
            "segmented_reduce_min_f64",
            "segmented_reduce_max_f64",
            "lower_bound_i32", "lower_bound_i64", "lower_bound_u64",
            "upper_bound_i32", "upper_bound_u64",
            "radix_sort_i32_i32", "radix_sort_i64_i32", "radix_sort_u64_i32",
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

    def test_ensure_pipelines_warm_calls_exact_polygon_probe(self):
        with patch(
            "vibespatial.cuda.cccl_precompile._warm_exact_polygon_intersection_route",
        ) as probe:
            ensure_pipelines_warm()
        probe.assert_called_once_with()

    def test_ensure_pipelines_warm_calls_many_vs_one_overlay_probe(self):
        with patch(
            "vibespatial.cuda.cccl_precompile._warm_many_vs_one_overlay_remainder_route",
        ) as probe:
            ensure_pipelines_warm()
        probe.assert_called_once_with()


@pytest.mark.gpu
def test_warm_many_vs_one_overlay_probe_stays_off_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    import vibespatial.cuda.cccl_precompile as precompile_module

    monkeypatch.setattr(precompile_module, "_many_vs_one_overlay_warm_done", False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warm_many_vs_one_overlay_remainder_route()

    assert not any("many-vs-one remainder" in str(entry.message) for entry in caught)

class TestCCCLPrecompilerCore:
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
        assert "deferred" in status
        assert "pending" in status
        assert "failed" in status
        assert "wall_ms" in status
        assert "per_primitive" in status
        assert isinstance(status["per_primitive"], list)


# ---------------------------------------------------------------------------
# Deferred disk loading
# ---------------------------------------------------------------------------

class TestCCCLDeferredDiskLoading:
    def test_request_defers_cached_specs(self):
        """Specs with disk cache entries are deferred, not submitted to thread pool."""
        precompiler = CCCLPrecompiler(max_workers=1)
        with patch(
            "vibespatial.cuda.cccl_cubin_cache._cached_spec_name_set",
            return_value=frozenset({"exclusive_scan_i32"}),
        ):
            precompiler.request(["exclusive_scan_i32", "select_i32"])
        assert "exclusive_scan_i32" in precompiler._deferred_disk
        assert "exclusive_scan_i32" in precompiler._submitted
        assert "select_i32" not in precompiler._deferred_disk
        assert "select_i32" in precompiler._submitted
        assert "select_i32" in precompiler._futures
        assert "exclusive_scan_i32" not in precompiler._futures

    def test_no_executor_when_all_deferred(self):
        """Thread pool is not created when all specs are deferred."""
        precompiler = CCCLPrecompiler(max_workers=1)
        with patch(
            "vibespatial.cuda.cccl_cubin_cache._cached_spec_name_set",
            return_value=frozenset({"exclusive_scan_i32", "select_i32"}),
        ):
            precompiler.request(["exclusive_scan_i32", "select_i32"])
        assert precompiler._executor is None
        assert len(precompiler._deferred_disk) == 2

    def test_executor_created_on_cache_miss(self):
        """Thread pool IS created when at least one spec is a cache miss."""
        precompiler = CCCLPrecompiler(max_workers=1)
        with patch(
            "vibespatial.cuda.cccl_cubin_cache._cached_spec_name_set",
            return_value=frozenset({"exclusive_scan_i32"}),
        ), patch.object(precompiler, "_compile_one", return_value=None):
            precompiler.request(["exclusive_scan_i32", "select_i32"])
        assert precompiler._executor is not None
        assert "exclusive_scan_i32" in precompiler._deferred_disk
        assert "select_i32" in precompiler._futures

    def test_get_compiled_lazy_loads_deferred(self):
        """get_compiled() lazy-loads a deferred spec via _lazy_load_deferred."""
        from vibespatial.cuda.cccl_precompile import PrecompiledPrimitive

        precompiler = CCCLPrecompiler(max_workers=1)
        precompiler._deferred_disk.add("exclusive_scan_i32")
        precompiler._submitted.add("exclusive_scan_i32")
        mock_result = PrecompiledPrimitive(
            name="exclusive_scan_i32", make_callable=None,
            temp_storage=None, temp_storage_bytes=0,
            high_water_n=128, warmup_ms=1.0,
        )
        with patch.object(precompiler, "_lazy_load_deferred", return_value=mock_result):
            result = precompiler.get_compiled("exclusive_scan_i32")
        assert result is mock_result

    def test_get_compiled_returns_cache_first(self):
        """get_compiled() returns from _cache without touching _deferred_disk."""
        from vibespatial.cuda.cccl_precompile import PrecompiledPrimitive

        precompiler = CCCLPrecompiler(max_workers=1)
        mock_result = PrecompiledPrimitive(
            name="exclusive_scan_i32", make_callable=None,
            temp_storage=None, temp_storage_bytes=0,
            high_water_n=128, warmup_ms=1.0,
        )
        precompiler._cache["exclusive_scan_i32"] = mock_result
        precompiler._deferred_disk.add("exclusive_scan_i32")
        result = precompiler.get_compiled("exclusive_scan_i32")
        assert result is mock_result

    def test_status_reports_deferred_count(self):
        precompiler = CCCLPrecompiler(max_workers=1)
        precompiler._deferred_disk.add("exclusive_scan_i32")
        precompiler._deferred_disk.add("select_i32")
        status = precompiler.status()
        assert status["deferred"] == 2

    def test_ensure_warm_loads_deferred_specs(self):
        """ensure_warm() loads all deferred specs."""
        from vibespatial.cuda.cccl_precompile import PrecompiledPrimitive

        precompiler = CCCLPrecompiler(max_workers=1)
        precompiler._deferred_disk.add("exclusive_scan_i32")
        precompiler._submitted.add("exclusive_scan_i32")
        mock_result = PrecompiledPrimitive(
            name="exclusive_scan_i32", make_callable=None,
            temp_storage=None, temp_storage_bytes=0,
            high_water_n=128, warmup_ms=1.0,
        )
        with patch.object(precompiler, "_lazy_load_deferred", return_value=mock_result):
            cold = precompiler.ensure_warm()
        assert cold == []

    def test_shutdown_noop_when_no_executor(self):
        """shutdown() doesn't crash when executor was never created."""
        precompiler = CCCLPrecompiler(max_workers=1)
        assert precompiler._executor is None
        precompiler.shutdown()  # should not raise
