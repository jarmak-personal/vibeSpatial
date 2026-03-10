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
        assert "pending" in status
        assert "failed" in status
        assert "wall_ms" in status
        assert "per_unit" in status
        assert isinstance(status["per_unit"], list)
