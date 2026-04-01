from __future__ import annotations

import pytest

from vibespatial import ExecutionMode
from vibespatial import runtime as runtime_module
from vibespatial.runtime import _runtime as _runtime_impl
from vibespatial.testing import (
    cuda_runtime_available,
    normalize_dispatch_modes,
    resolve_dispatch_modes,
)


def test_cuda_runtime_available_returns_bool() -> None:
    assert isinstance(cuda_runtime_available(), bool)


def test_normalize_dispatch_modes_preserves_order_and_deduplicates() -> None:
    modes = normalize_dispatch_modes(["cpu", ExecutionMode.GPU, "cpu", "auto"])
    assert modes == (ExecutionMode.CPU, ExecutionMode.GPU, ExecutionMode.AUTO)


def test_resolve_dispatch_modes_defaults_to_cpu_without_gpu_request() -> None:
    modes = resolve_dispatch_modes(None, cuda_available=False, run_gpu=False)
    assert modes == (ExecutionMode.CPU,)


def test_resolve_dispatch_modes_adds_gpu_when_requested_and_available() -> None:
    modes = resolve_dispatch_modes(None, cuda_available=True, run_gpu=True)
    assert modes == (ExecutionMode.CPU, ExecutionMode.GPU)


def test_resolve_dispatch_modes_allows_explicit_auto_mode() -> None:
    modes = resolve_dispatch_modes(["auto"], cuda_available=False, run_gpu=False)
    assert modes == (ExecutionMode.AUTO,)


def test_resolve_dispatch_modes_rejects_unavailable_explicit_gpu() -> None:
    with pytest.raises(ValueError, match="no CUDA runtime is available"):
        resolve_dispatch_modes(["gpu"], cuda_available=False, run_gpu=False)


def test_dispatch_selection_matches_requested_mode(dispatch_mode, dispatch_selection) -> None:
    assert dispatch_selection.requested is dispatch_mode
    if dispatch_mode is ExecutionMode.CPU:
        assert dispatch_selection.selected is ExecutionMode.CPU


@pytest.mark.cpu_fallback
def test_auto_runtime_selection_reports_explicit_fallback_without_cuda(monkeypatch) -> None:
    monkeypatch.setattr(_runtime_impl, "has_gpu_runtime", lambda: False)
    selection = runtime_module.select_runtime(ExecutionMode.AUTO)
    assert selection.selected is ExecutionMode.CPU
    assert "explicit CPU fallback" in selection.reason


@pytest.mark.gpu
def test_explicit_gpu_selection_uses_gpu_when_available() -> None:
    selection = runtime_module.select_runtime(ExecutionMode.GPU)
    assert selection.selected is ExecutionMode.GPU
