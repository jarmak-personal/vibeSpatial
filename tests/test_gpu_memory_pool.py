from __future__ import annotations

import gc
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest


def test_ptds_stream_identity_retains_lifetime_unique_thread_tokens() -> None:
    import vibespatial.cuda._runtime as rt_mod

    stream = SimpleNamespace(ptr=2)

    def _identity():
        return rt_mod.cuda_stream_identity(stream)

    identities = []
    for _ in range(2):
        thread = threading.Thread(target=lambda: identities.append(_identity()))
        thread.start()
        thread.join()

    assert {identity.handle for identity in identities} == {2}
    assert identities[0] != identities[1]
    assert identities[0].owner_thread is not identities[1].owner_thread


def test_ptds_retirements_record_and_claim_in_the_submitting_thread(monkeypatch) -> None:
    import vibespatial.cuda._runtime as rt_mod

    recorded_threads: list[int] = []

    class _Event:
        done = False

        def __init__(self, **_kwargs) -> None:
            pass

        def record(self, _stream) -> None:
            recorded_threads.append(threading.get_ident())

    retainer = rt_mod.CudaCompletionRetainer()
    monkeypatch.setattr(rt_mod.cp.cuda, "Event", _Event)
    monkeypatch.setattr(
        rt_mod.CudaDriverRuntime,
        "_cupy_stream",
        staticmethod(lambda stream: stream),
    )
    monkeypatch.setattr(retainer, "_ensure_worker_locked", lambda: None)
    barrier = threading.Barrier(2)

    def _defer_and_claim(label: str):
        stream = SimpleNamespace(ptr=2)
        thread_id = threading.get_ident()
        retainer.defer(stream, label, lambda _payload: None)
        barrier.wait()
        return thread_id, retainer.claim_stream_retirements(stream)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = [
            future.result()
            for future in (
                executor.submit(_defer_and_claim, "left"),
                executor.submit(_defer_and_claim, "right"),
            )
        ]

    assert sorted(recorded_threads) == sorted(thread_id for thread_id, _ in results)
    assert sorted(claimed[0][0] for _thread_id, claimed in results) == ["left", "right"]
    assert all(len(claimed) == 1 for _thread_id, claimed in results)


def test_driver_launch_inherits_cupy_current_stream(monkeypatch) -> None:
    """Implicit driver launches must share CuPy's producer/consumer stream."""
    import vibespatial.cuda._runtime as rt_mod

    current_stream = object()
    normalized_streams: list[object] = []
    launch_streams: list[object] = []
    driver_stream = object()
    runtime = rt_mod.CudaDriverRuntime.__new__(rt_mod.CudaDriverRuntime)

    monkeypatch.setattr(
        rt_mod.cp.cuda,
        "get_current_stream",
        lambda: current_stream,
    )
    monkeypatch.setattr(
        rt_mod,
        "_normalize_stream_handle",
        lambda stream: normalized_streams.append(stream) or driver_stream,
    )
    monkeypatch.setattr(runtime, "activate", nullcontext)
    monkeypatch.setattr(rt_mod, "_check_driver", lambda result: result)

    def _launch(*args):
        launch_streams.append(args[8])
        return None

    monkeypatch.setattr(rt_mod.cu, "cuLaunchKernel", _launch)

    runtime.launch(
        rt_mod.CompiledKernel(name="test", function=object()),
        grid=(1, 1, 1),
        block=(1, 1, 1),
        params=((), ()),
    )

    assert normalized_streams == [current_stream]
    assert launch_streams == [driver_stream]


def test_explicit_free_waits_for_active_stream_completion(monkeypatch) -> None:
    """Manual frees must not bypass work already queued on the CuPy stream."""
    import vibespatial.cuda._runtime as rt_mod

    free_calls: list[str] = []
    memory = SimpleNamespace(free=lambda: free_calls.append("free"))
    device_array = SimpleNamespace(data=SimpleNamespace(mem=memory))
    current_stream = object()
    runtime = rt_mod.CudaDriverRuntime.__new__(rt_mod.CudaDriverRuntime)

    class _Retainer:
        def __init__(self) -> None:
            self.retirements = []

        def defer(self, stream, payload, release) -> None:
            self.retirements.append((stream, payload, release))

        def claim_stream_retirements(self, stream):
            selected = [item[1:] for item in self.retirements if item[0] is stream]
            self.retirements = [item for item in self.retirements if item[0] is not stream]
            return selected

        @staticmethod
        def release_claimed_retirements(retirements) -> None:
            for payload, release in retirements:
                release(payload)

    retainer = _Retainer()

    monkeypatch.setattr(runtime, "activate", nullcontext)
    monkeypatch.setattr(rt_mod.cp.cuda, "get_current_stream", lambda: current_stream)
    monkeypatch.setattr(rt_mod, "get_cuda_completion_retainer", lambda: retainer)

    runtime.free(device_array)

    assert free_calls == []
    assert retainer.retirements == [(current_stream, device_array, runtime._free_now)]

    claimed = retainer.claim_stream_retirements(current_stream)
    retainer.release_claimed_retirements(claimed)

    assert free_calls == ["free"]
    assert retainer.retirements == []


@pytest.mark.gpu
def test_explicit_free_retires_after_cupy_stream_completion() -> None:
    """A final partial retirement window must not require another runtime call."""
    import cupy as cp

    import vibespatial
    from vibespatial.cuda._runtime import get_cuda_runtime

    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    runtime = get_cuda_runtime()
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        allocation = cp.empty(2_000_000, dtype=cp.float64)
        allocation.fill(3.0)
        allocation_ref = weakref.ref(allocation)
        runtime.free(allocation)
    del allocation

    assert allocation_ref() is not None
    stream.synchronize()
    deadline = time.monotonic() + 2.0
    while allocation_ref() is not None and time.monotonic() < deadline:
        gc.collect()
        time.sleep(0.01)

    assert allocation_ref() is None


@pytest.mark.gpu
def test_pylibcudf_stream_wrapper_follows_transient_stream_lifetime() -> None:
    import cupy as cp

    import vibespatial
    from vibespatial.cuda._runtime import pylibcudf_current_stream

    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    stream = cp.cuda.Stream(non_blocking=True)
    stream_ref = weakref.ref(stream)
    with stream:
        wrapper = pylibcudf_current_stream()
        assert pylibcudf_current_stream() is wrapper
    stream.synchronize()
    del wrapper
    del stream

    deadline = time.monotonic() + 2.0
    while stream_ref() is not None and time.monotonic() < deadline:
        gc.collect()
        time.sleep(0.01)

    assert stream_ref() is None


@pytest.mark.parametrize("failure_stage", ["record", "query"])
def test_completion_retainer_event_failures_synchronize_and_release(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    import vibespatial.cuda._runtime as rt_mod

    released: list[object] = []
    synchronized: list[int] = []
    stream = SimpleNamespace(ptr=41)
    stream_key = rt_mod.cuda_stream_identity(stream)
    retainer = rt_mod.CudaCompletionRetainer()

    class _SyncStream:
        def synchronize(self) -> None:
            synchronized.append(41)

    monkeypatch.setattr(
        rt_mod.CudaDriverRuntime,
        "_cupy_stream",
        staticmethod(lambda value: _SyncStream()),
    )

    if failure_stage == "record":
        monkeypatch.setattr(
            rt_mod.cp.cuda,
            "Event",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("record failed")),
        )
        retainer._open[stream_key] = (stream, 0.0, [("payload", released.append)])
        with retainer._condition:
            retainer._record_due_batches_locked(1.0)
        batch = retainer._pending[0]
        assert batch.event is None
    else:

        class _BrokenEvent:
            @property
            def done(self):
                raise RuntimeError("query failed")

        batch = rt_mod._CompletionRetirementBatch(
            event=_BrokenEvent(),
            stream=stream,
            stream_key=stream_key,
            retirements=(("payload", released.append),),
        )

    assert retainer._batch_completed(batch) is True
    retainer._apply_retirements(batch.retirements)
    assert synchronized == [41]
    assert released == ["payload"]


def test_completion_retainer_keeps_payload_until_a_sync_boundary_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vibespatial.cuda._runtime as rt_mod

    released: list[object] = []
    stream_attempts: list[int] = []
    device_attempts: list[int] = []
    device_sync_succeeds = False
    stream = SimpleNamespace(ptr=43)
    stream_key = rt_mod.cuda_stream_identity(stream)
    retainer = rt_mod.CudaCompletionRetainer()

    class _BrokenStream:
        def synchronize(self) -> None:
            stream_attempts.append(43)
            raise RuntimeError("stream synchronization failed")

    def _device_synchronize() -> None:
        device_attempts.append(43)
        if not device_sync_succeeds:
            raise RuntimeError("context synchronization failed")

    monkeypatch.setattr(
        rt_mod.CudaDriverRuntime,
        "_cupy_stream",
        staticmethod(lambda value: _BrokenStream()),
    )
    monkeypatch.setattr(rt_mod.cp.cuda.runtime, "deviceSynchronize", _device_synchronize)
    batch = rt_mod._CompletionRetirementBatch(
        event=None,
        stream=stream,
        stream_key=stream_key,
        retirements=(("payload", released.append),),
    )

    assert retainer._batch_completed(batch) is False
    assert batch.failure_count == 1
    assert batch.retry_after > time.monotonic()
    assert released == []

    batch.retry_after = 0.0
    device_sync_succeeds = True
    assert retainer._batch_completed(batch) is True
    retainer._apply_retirements(batch.retirements)

    assert stream_attempts == [43, 43]
    assert device_attempts == [43, 43]
    assert released == ["payload"]


def test_failed_ptds_event_recovery_never_synchronizes_from_a_nonowner_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vibespatial.cuda._runtime as rt_mod

    owner = threading.Thread()
    device_syncs: list[str] = []
    batch = rt_mod._CompletionRetirementBatch(
        event=None,
        stream=SimpleNamespace(ptr=2),
        stream_key=rt_mod.CudaStreamIdentity(handle=2, owner_thread=owner),
        retirements=(("payload", lambda _payload: None),),
    )
    monkeypatch.setattr(
        rt_mod.CudaDriverRuntime,
        "_cupy_stream",
        staticmethod(
            lambda _stream: (_ for _ in ()).throw(
                AssertionError("a non-owner thread cannot synchronize another thread's PTDS")
            )
        ),
    )
    monkeypatch.setattr(
        rt_mod.cp.cuda.runtime,
        "deviceSynchronize",
        lambda: device_syncs.append("context"),
    )

    assert rt_mod.CudaCompletionRetainer._batch_completed(batch) is True
    assert device_syncs == ["context"]


def test_stream_sync_claim_excludes_retirements_deferred_after_boundary() -> None:
    import vibespatial.cuda._runtime as rt_mod

    released: list[str] = []
    stream = SimpleNamespace(ptr=47)
    stream_key = rt_mod.cuda_stream_identity(stream)
    retainer = rt_mod.CudaCompletionRetainer()
    retainer._open[stream_key] = (
        stream,
        time.monotonic() + 60.0,
        [("before", released.append)],
    )

    boundary_claimed = threading.Event()

    def _defer_after_boundary() -> None:
        boundary_claimed.wait()
        with retainer._condition:
            retainer._open[stream_key] = (
                stream,
                time.monotonic() + 60.0,
                [("after", released.append)],
            )

    later_submitter = threading.Thread(target=_defer_after_boundary)
    later_submitter.start()

    claimed = retainer.claim_stream_retirements(stream)
    boundary_claimed.set()
    later_submitter.join()
    retainer.release_claimed_retirements(claimed)

    assert released == ["before"]
    assert [payload for payload, _release in retainer._open[stream_key][2]] == ["after"]

    later = retainer.claim_stream_retirements(stream)
    retainer.release_claimed_retirements(later)
    assert released == ["before", "after"]


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


def test_memory_pool_configuration_waits_for_active_context(monkeypatch) -> None:
    """Importable CuPy must not install an allocator before context creation."""
    import vibespatial.cuda._runtime as rt_mod

    # Monkeypatch rmm to None at the module level
    monkeypatch.setattr(rt_mod, "rmm", None)
    monkeypatch.setattr(rt_mod, "rmm_cupy_allocator", None)

    runtime = rt_mod.CudaDriverRuntime()
    assert runtime._memory_backend == "none"
    assert runtime._memory_pool is None
    assert runtime._rmm_mr is None
    assert runtime._memory_pool_configured is False


# ---------------------------------------------------------------------------
# memory_pool_stats returns expected shape
# ---------------------------------------------------------------------------


def test_memory_pool_stats_cupy_backend_returns_expected_keys() -> None:
    """CuPy backend stats include used_bytes, total_bytes, free_bytes."""
    import vibespatial.cuda._runtime as rt_mod

    pool = SimpleNamespace(
        used_bytes=lambda: 11,
        total_bytes=lambda: 29,
        free_bytes=lambda: 18,
    )
    runtime = rt_mod.CudaDriverRuntime.__new__(rt_mod.CudaDriverRuntime)
    runtime._memory_backend = "cupy"
    runtime._memory_pool = pool
    runtime._rmm_mr = None
    stats = runtime.memory_pool_stats()

    assert stats == {"used_bytes": 11, "total_bytes": 29, "free_bytes": 18}


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


def test_free_pool_memory_cupy_backend_does_not_raise() -> None:
    """free_pool_memory should not raise with CuPy backend."""
    import vibespatial.cuda._runtime as rt_mod

    calls: list[str] = []
    runtime = rt_mod.CudaDriverRuntime.__new__(rt_mod.CudaDriverRuntime)
    runtime._memory_backend = "cupy"
    runtime._memory_pool = SimpleNamespace(
        free_all_blocks=lambda: calls.append("free"),
    )
    runtime._rmm_mr = None
    runtime.free_pool_memory()

    assert calls == ["free"]


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
