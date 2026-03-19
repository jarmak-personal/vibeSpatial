"""Tests for NVRTC pre-compilation warmup infrastructure (ADR-0034)."""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from vibespatial.cccl_precompile import PRECOMPILE_ENV_VAR
from vibespatial.nvrtc_precompile import (
    NVRTCPrecompiler,
    request_nvrtc_warmup,
)


@pytest.fixture(autouse=True)
def _reset_precompiler():
    """Reset the singleton between tests."""
    NVRTCPrecompiler._reset()
    yield
    NVRTCPrecompiler._reset()


class TestNVRTCPrecompilerNoGPU:
    def test_request_noop_without_gpu(self):
        with patch("vibespatial.runtime.has_gpu_runtime", return_value=False):
            request_nvrtc_warmup([("test", "source", ("k1",))])
        assert NVRTCPrecompiler._instance is None

    def test_request_noop_when_disabled(self):
        with patch.dict(os.environ, {PRECOMPILE_ENV_VAR: "0"}):
            request_nvrtc_warmup([("test", "source", ("k1",))])
        assert NVRTCPrecompiler._instance is None

    def test_deduplication(self):
        precompiler = NVRTCPrecompiler(max_workers=1)
        with patch.object(precompiler, "_compile_one", return_value=None):
            with patch(
                "vibespatial.cuda_runtime.make_kernel_cache_key",
                side_effect=lambda prefix, source: f"{prefix}-hash",
            ):
                precompiler.request([
                    ("a", "source_a", ("k1",)),
                    ("b", "source_b", ("k2",)),
                ])
                precompiler.request([
                    ("a", "source_a", ("k1",)),  # duplicate
                    ("c", "source_c", ("k3",)),
                ])
        assert len(precompiler._submitted) == 3

    def test_status_shape(self):
        precompiler = NVRTCPrecompiler(max_workers=1)
        status = precompiler.status()
        assert "submitted" in status
        assert "compiled" in status
        assert "deferred" in status
        assert "pending" in status
        assert "failed" in status
        assert "wall_ms" in status
        assert "per_unit" in status
        assert isinstance(status["per_unit"], list)


# ---------------------------------------------------------------------------
# Deferred disk loading
# ---------------------------------------------------------------------------

class TestNVRTCDeferredDiskLoading:
    def test_request_defers_cached_units(self):
        """Units with disk cache entries are deferred, not submitted to thread pool."""
        precompiler = NVRTCPrecompiler(max_workers=1)
        mock_runtime = type("MockRuntime", (), {
            "compute_capability": (8, 9),
            "_module_cache": {},
        })()
        with patch(
            "vibespatial.cuda_runtime.make_kernel_cache_key",
            side_effect=lambda prefix, source: f"{prefix}-hash",
        ), patch(
            "vibespatial.cuda_runtime._nvrtc_cached_key_set",
            return_value=frozenset({"v2-sm89-nvrtc0.0-a-hash"}),
        ), patch(
            "vibespatial.cuda_runtime._disk_cache_key",
            side_effect=lambda ck, cc, opts, ver: f"v2-sm89-nvrtc0.0-{ck}",
        ), patch(
            "vibespatial.cuda_runtime._nvrtc_version",
            return_value=(0, 0),
        ), patch(
            "vibespatial.cuda_runtime.get_cuda_runtime",
            return_value=mock_runtime,
        ):
            precompiler.request([("a", "source_a", ("k1",))])
        assert "a-hash" in precompiler._deferred_disk
        assert "a-hash" in precompiler._submitted
        assert precompiler._executor is None

    def test_no_executor_when_all_deferred(self):
        """Thread pool is not created when all units are disk-cached."""
        precompiler = NVRTCPrecompiler(max_workers=1)
        mock_runtime = type("MockRuntime", (), {
            "compute_capability": (8, 9),
            "_module_cache": {},
        })()
        with patch(
            "vibespatial.cuda_runtime.make_kernel_cache_key",
            side_effect=lambda prefix, source: f"{prefix}-hash",
        ), patch(
            "vibespatial.cuda_runtime._nvrtc_cached_key_set",
            return_value=frozenset({"v2-sm89-nvrtc0.0-a-hash", "v2-sm89-nvrtc0.0-b-hash"}),
        ), patch(
            "vibespatial.cuda_runtime._disk_cache_key",
            side_effect=lambda ck, cc, opts, ver: f"v2-sm89-nvrtc0.0-{ck}",
        ), patch(
            "vibespatial.cuda_runtime._nvrtc_version",
            return_value=(0, 0),
        ), patch(
            "vibespatial.cuda_runtime.get_cuda_runtime",
            return_value=mock_runtime,
        ):
            precompiler.request([
                ("a", "src_a", ("k1",)),
                ("b", "src_b", ("k2",)),
            ])
        assert precompiler._executor is None
        assert len(precompiler._deferred_disk) == 2

    def test_status_reports_deferred_count(self):
        precompiler = NVRTCPrecompiler(max_workers=1)
        precompiler._deferred_disk.add("test-key")
        status = precompiler.status()
        assert status["deferred"] == 1

    def test_shutdown_noop_when_no_executor(self):
        """shutdown() doesn't crash when executor was never created."""
        precompiler = NVRTCPrecompiler(max_workers=1)
        assert precompiler._executor is None
        precompiler.shutdown()  # should not raise
