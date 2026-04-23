from __future__ import annotations

import warnings

import numpy as np
import pytest

from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.execution_trace import (
    TRACE_WARNINGS_ENV_VAR,
    TraceStep,
    TraceTransfer,
    VibeTraceWarning,
    execution_trace,
    get_active_trace,
    notify_dispatch,
    notify_transfer,
)


def test_trace_context_manager_sets_and_clears_active_trace() -> None:
    assert get_active_trace() is None
    with execution_trace("test_pipeline") as ctx:
        assert get_active_trace() is ctx
        assert ctx.pipeline == "test_pipeline"
    assert get_active_trace() is None


def test_nested_traces_restore_parent() -> None:
    with execution_trace("outer") as outer:
        assert get_active_trace() is outer
        with execution_trace("inner") as inner:
            assert get_active_trace() is inner
        assert get_active_trace() is outer
    assert get_active_trace() is None


def test_dispatch_steps_are_recorded() -> None:
    with execution_trace("test") as ctx:
        notify_dispatch(
            surface="kernel_a",
            operation="compute",
            selected=ExecutionMode.GPU,
            implementation="gpu_kernel_a",
        )
        notify_dispatch(
            surface="kernel_b",
            operation="compute",
            selected=ExecutionMode.GPU,
            implementation="gpu_kernel_b",
        )
    assert len(ctx.steps) == 2
    assert ctx.steps[0].selected is ExecutionMode.GPU
    assert ctx.steps[1].selected is ExecutionMode.GPU


def test_no_trace_does_not_raise() -> None:
    # notify_dispatch outside any trace should be a no-op
    notify_dispatch(
        surface="orphan",
        operation="op",
        selected=ExecutionMode.CPU,
        implementation="cpu",
    )
    notify_transfer(direction="d2h", trigger="test", reason="test")


def test_gpu_offramp_emits_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with execution_trace("test_offramp") as ctx:
            notify_dispatch(
                surface="kernel_a",
                operation="step1",
                selected=ExecutionMode.GPU,
                implementation="gpu_impl",
            )
            notify_dispatch(
                surface="kernel_b",
                operation="step2",
                selected=ExecutionMode.CPU,
                implementation="cpu_impl",
            )
    offramp_warnings = [w for w in caught if "GPU offramp" in str(w.message)]
    assert len(offramp_warnings) == 1
    assert offramp_warnings[0].category is VibeTraceWarning
    msg = str(offramp_warnings[0].message)
    assert "test_offramp" in msg
    assert "kernel_a" in msg
    assert "kernel_b" in msg
    summary = ctx.summary()
    assert summary["offramps"] == 1
    assert summary["gpu_steps"] == 1
    assert summary["cpu_steps"] == 1


def test_no_offramp_when_all_gpu() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with execution_trace("all_gpu"):
            notify_dispatch(
                surface="a", operation="op",
                selected=ExecutionMode.GPU, implementation="gpu",
            )
            notify_dispatch(
                surface="b", operation="op",
                selected=ExecutionMode.GPU, implementation="gpu",
            )
    offramp_warnings = [w for w in caught if "GPU offramp" in str(w.message)]
    assert len(offramp_warnings) == 0


def test_d2h_transfer_emits_warning_when_gpu_active() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with execution_trace("test_d2h") as ctx:
            notify_dispatch(
                surface="gpu_kernel",
                operation="compute",
                selected=ExecutionMode.GPU,
                implementation="gpu",
            )
            notify_transfer(
                direction="d2h",
                trigger="user-materialization",
                reason="to_shapely called mid-pipeline",
            )
    d2h_warnings = [w for w in caught if "D->H transfer" in str(w.message)]
    assert len(d2h_warnings) == 1
    assert "test_d2h" in str(d2h_warnings[0].message)
    assert ctx.summary()["d2h_transfers"] == 1


def test_ping_pong_emits_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with execution_trace("test_pingpong") as ctx:
            notify_dispatch(
                surface="gpu_kernel",
                operation="compute",
                selected=ExecutionMode.GPU,
                implementation="gpu",
            )
            notify_transfer(
                direction="d2h",
                trigger="unsupported-gpu-path",
                reason="materialize for CPU step",
            )
            notify_transfer(
                direction="h2d",
                trigger="explicit-runtime-request",
                reason="re-upload for next GPU kernel",
            )
    pingpong_warnings = [w for w in caught if "ping-pong" in str(w.message)]
    assert len(pingpong_warnings) == 1
    assert "test_pingpong" in str(pingpong_warnings[0].message)
    summary = ctx.summary()
    assert summary["d2h_transfers"] == 1
    assert summary["h2d_transfers"] == 1


def test_summary_counts() -> None:
    with execution_trace("summary_test") as ctx:
        ctx.record_step(TraceStep("a", "op", ExecutionMode.GPU, "gpu"))
        ctx.record_step(TraceStep("b", "op", ExecutionMode.GPU, "gpu"))
        ctx.record_step(TraceStep("c", "op", ExecutionMode.CPU, "cpu"))
        ctx.record_transfer(TraceTransfer("d2h", "test", "reason1"))
        ctx.record_transfer(TraceTransfer("h2d", "test", "reason2"))
    summary = ctx.summary()
    assert summary == {
        "pipeline": "summary_test",
        "total_steps": 3,
        "gpu_steps": 2,
        "cpu_steps": 1,
        "d2h_transfers": 1,
        "h2d_transfers": 1,
        "d2h_transfer_bytes": 0,
        "h2d_transfer_bytes": 0,
        "d2h_transfer_seconds": 0.0,
        "h2d_transfer_seconds": 0.0,
        "runtime_d2h_transfers": 0,
        "runtime_h2d_transfers": 0,
        "runtime_d2h_transfer_bytes": 0,
        "runtime_h2d_transfer_bytes": 0,
        "runtime_d2h_transfer_seconds": 0.0,
        "runtime_h2d_transfer_seconds": 0.0,
        "offramps": 1,
    }


@pytest.mark.gpu
def test_cuda_runtime_d2h_copy_records_trace_event() -> None:
    from vibespatial.runtime import has_gpu_runtime

    if not has_gpu_runtime():
        pytest.skip("GPU required")

    pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        get_cuda_runtime,
        get_d2h_transfer_profile,
        reset_d2h_transfer_count,
    )

    runtime = get_cuda_runtime()
    device = runtime.from_host(np.arange(4, dtype=np.int32))
    reset_d2h_transfer_count()

    with execution_trace("runtime_copy") as ctx:
        host = runtime.copy_device_to_host(device)

    assert host.tolist() == [0, 1, 2, 3]
    assert len(ctx.transfers) == 1
    transfer = ctx.transfers[0]
    assert transfer.direction == "d2h"
    assert transfer.source == "cuda_runtime"
    assert transfer.item_count == 4
    assert transfer.bytes_transferred == 16
    assert transfer.elapsed_seconds >= 0.0
    summary = ctx.summary()
    assert summary["d2h_transfers"] == 1
    assert summary["runtime_d2h_transfers"] == 1
    assert summary["runtime_d2h_transfer_bytes"] == 16
    assert summary["runtime_d2h_transfer_seconds"] >= 0.0
    count, bytes_transferred, seconds = get_d2h_transfer_profile()
    assert count == 1
    assert bytes_transferred == 16
    assert seconds >= 0.0


def test_env_var_suppresses_trace_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TRACE_WARNINGS_ENV_VAR, "0")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with execution_trace("suppressed"):
            notify_dispatch(
                surface="a", operation="op",
                selected=ExecutionMode.GPU, implementation="gpu",
            )
            notify_dispatch(
                surface="b", operation="op",
                selected=ExecutionMode.CPU, implementation="cpu",
            )
            notify_transfer(direction="d2h", trigger="test", reason="test")
    trace_warnings = [w for w in caught if issubclass(w.category, VibeTraceWarning)]
    assert len(trace_warnings) == 0


def test_filterwarnings_suppresses_by_category() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=VibeTraceWarning)
        with execution_trace("filtered"):
            notify_dispatch(
                surface="a", operation="op",
                selected=ExecutionMode.GPU, implementation="gpu",
            )
            notify_dispatch(
                surface="b", operation="op",
                selected=ExecutionMode.CPU, implementation="cpu",
            )
    trace_warnings = [w for w in caught if issubclass(w.category, VibeTraceWarning)]
    assert len(trace_warnings) == 0
