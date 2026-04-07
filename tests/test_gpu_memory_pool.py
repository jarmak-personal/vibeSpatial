from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# OOM callback unit tests (pure Python, no GPU required)
# ---------------------------------------------------------------------------


def test_oom_callback_retries_up_to_max() -> None:
    """Callback returns True for max_retries attempts, then False."""
    from vibespatial.cuda._runtime import _make_oom_callback

    callback = _make_oom_callback(max_retries=3)

    # First 3 calls should return True (retry)
    assert callback(1024) is True
    assert callback(1024) is True
    assert callback(1024) is True

    # 4th call should return False (give up) and reset counter
    assert callback(1024) is False


def test_oom_callback_resets_after_exhaustion() -> None:
    """After exhausting retries the counter resets, allowing a new retry cycle."""
    from vibespatial.cuda._runtime import _make_oom_callback

    callback = _make_oom_callback(max_retries=2)

    # Exhaust first cycle
    assert callback(256) is True
    assert callback(256) is True
    assert callback(256) is False

    # New cycle should work again
    assert callback(256) is True
    assert callback(256) is True
    assert callback(256) is False


def test_oom_callback_time_reset(monkeypatch) -> None:
    """Counter resets when >1 s passes between callback calls (new OOM event)."""
    import time as _time

    from vibespatial.cuda._runtime import _make_oom_callback

    fake_time = [0.0]
    monkeypatch.setattr(_time, "monotonic", lambda: fake_time[0])

    callback = _make_oom_callback(max_retries=3)

    # First OOM: use 1 retry, then "allocation succeeds" (callback not called)
    assert callback(1024) is True

    # Simulate >1 s gap (successful allocation happened in between)
    fake_time[0] = 2.0

    # Second OOM should get full retry budget (counter reset by time gap)
    assert callback(1024) is True
    assert callback(1024) is True
    assert callback(1024) is True
    assert callback(1024) is False


# ---------------------------------------------------------------------------
# CuPy fallback when RMM is not available
# ---------------------------------------------------------------------------


def test_cupy_fallback_when_rmm_unavailable(monkeypatch) -> None:
    """When rmm is None, runtime falls back to CuPy memory pool."""
    import vibespatial.cuda._runtime as rt_mod

    # Monkeypatch rmm to None at the module level
    monkeypatch.setattr(rt_mod, "rmm", None)
    monkeypatch.setattr(rt_mod, "rmm_cupy_allocator", None)

    runtime = rt_mod.CudaDriverRuntime()
    # Should use CuPy pool (or "none" if CuPy is not available)
    assert runtime._memory_backend in ("cupy", "none")
    assert runtime._rmm_mr is None


# ---------------------------------------------------------------------------
# memory_pool_stats returns expected shape
# ---------------------------------------------------------------------------


def test_memory_pool_stats_cupy_backend_returns_expected_keys(monkeypatch) -> None:
    """CuPy backend stats include used_bytes, total_bytes, free_bytes."""
    import vibespatial.cuda._runtime as rt_mod

    monkeypatch.setattr(rt_mod, "rmm", None)
    monkeypatch.setattr(rt_mod, "rmm_cupy_allocator", None)

    runtime = rt_mod.CudaDriverRuntime()
    stats = runtime.memory_pool_stats()

    if runtime._memory_backend == "cupy":
        assert "used_bytes" in stats
        assert "total_bytes" in stats
        assert "free_bytes" in stats
    else:
        # No CuPy available — stats should be empty dict
        assert stats == {}


def test_memory_pool_stats_no_backend_returns_empty() -> None:
    """When backend is 'none', stats returns empty dict."""
    import vibespatial.cuda._runtime as rt_mod

    runtime = rt_mod.CudaDriverRuntime.__new__(rt_mod.CudaDriverRuntime)
    runtime._memory_backend = "none"
    runtime._memory_pool = None
    runtime._rmm_mr = None
    stats = runtime.memory_pool_stats()
    assert stats == {}


def test_maybe_trim_pool_memory_skips_by_default(monkeypatch) -> None:
    """Hot paths should not eagerly flush the pool unless explicitly enabled."""
    import vibespatial.cuda._runtime as rt_mod

    calls: list[str] = []
    runtime = SimpleNamespace(free_pool_memory=lambda: calls.append("trim"))

    monkeypatch.delenv("VIBESPATIAL_EAGER_GPU_POOL_TRIM", raising=False)
    rt_mod.maybe_trim_pool_memory(runtime)

    assert calls == []


def test_maybe_trim_pool_memory_respects_env_opt_in(monkeypatch) -> None:
    """The eager trim escape hatch should still call through when requested."""
    import vibespatial.cuda._runtime as rt_mod

    calls: list[str] = []
    runtime = SimpleNamespace(free_pool_memory=lambda: calls.append("trim"))

    monkeypatch.setenv("VIBESPATIAL_EAGER_GPU_POOL_TRIM", "1")
    rt_mod.maybe_trim_pool_memory(runtime)

    assert calls == ["trim"]


# ---------------------------------------------------------------------------
# free_pool_memory does not raise
# ---------------------------------------------------------------------------


def test_free_pool_memory_cupy_backend_does_not_raise(monkeypatch) -> None:
    """free_pool_memory should not raise with CuPy backend."""
    import vibespatial.cuda._runtime as rt_mod

    monkeypatch.setattr(rt_mod, "rmm", None)
    monkeypatch.setattr(rt_mod, "rmm_cupy_allocator", None)

    runtime = rt_mod.CudaDriverRuntime()
    # Should not raise regardless of backend
    runtime.free_pool_memory()


def test_free_pool_memory_no_backend_does_not_raise() -> None:
    """free_pool_memory should not raise when backend is 'none'."""
    import vibespatial.cuda._runtime as rt_mod

    runtime = rt_mod.CudaDriverRuntime.__new__(rt_mod.CudaDriverRuntime)
    runtime._memory_backend = "none"
    runtime._memory_pool = None
    runtime._rmm_mr = None
    # Should not raise
    runtime.free_pool_memory()


# ---------------------------------------------------------------------------
# GPU-requiring tests: RMM allocation round-trip
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_rmm_allocation_round_trip() -> None:
    """Allocate via runtime, write data, read back, verify correctness."""
    from vibespatial.cuda._runtime import get_cuda_runtime

    runtime = get_cuda_runtime()
    # Ensure context is active (triggers deferred RMM setup if applicable)
    runtime._ensure_context()

    # Allocate a device array
    shape = (1024,)
    dtype = np.dtype(np.float64)
    dev_arr = runtime.allocate(shape, dtype, zero=True)

    # Write known pattern from host
    host_data = np.arange(1024, dtype=np.float64)
    runtime.copy_host_to_device(host_data, dev_arr)

    # Read back and verify
    result = runtime.copy_device_to_host(dev_arr)
    np.testing.assert_array_equal(result, host_data)


@pytest.mark.gpu
def test_memory_pool_stats_returns_dict_on_gpu() -> None:
    """On GPU, memory_pool_stats returns a dict (possibly with keys)."""
    from vibespatial.cuda._runtime import get_cuda_runtime

    runtime = get_cuda_runtime()
    runtime._ensure_context()

    stats = runtime.memory_pool_stats()
    assert isinstance(stats, dict)


@pytest.mark.gpu
def test_free_pool_memory_does_not_raise_on_gpu() -> None:
    """free_pool_memory must not raise on a live GPU runtime."""
    from vibespatial.cuda._runtime import get_cuda_runtime

    runtime = get_cuda_runtime()
    runtime._ensure_context()

    # Should not raise
    runtime.free_pool_memory()


@pytest.mark.gpu
def test_memory_backend_is_set_on_gpu() -> None:
    """After context init, memory_backend should be set to a known value."""
    from vibespatial.cuda._runtime import get_cuda_runtime

    runtime = get_cuda_runtime()
    runtime._ensure_context()

    assert runtime._memory_backend in ("cupy", "rmm-pool", "rmm-safe", "rmm-managed")
