from __future__ import annotations

import os
import threading
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from ._runtime import ExecutionMode

TRACE_WARNINGS_ENV_VAR = "VIBESPATIAL_TRACE_WARNINGS"


class VibeTraceWarning(UserWarning):
    pass


def _trace_warnings_enabled() -> bool:
    value = os.environ.get(TRACE_WARNINGS_ENV_VAR, "1")
    return value.lower() not in {"0", "false", "no", "off"}


def _trace_warn(message: str, stacklevel: int = 2) -> None:
    if _trace_warnings_enabled():
        warnings.warn(message, VibeTraceWarning, stacklevel=stacklevel + 1)


@dataclass
class TraceTransfer:
    direction: str  # "d2h" or "h2d"
    trigger: str
    reason: str


@dataclass
class TraceStep:
    surface: str
    operation: str
    selected: ExecutionMode
    implementation: str


@dataclass
class ExecutionTraceContext:
    pipeline: str
    steps: list[TraceStep] = field(default_factory=list)
    transfers: list[TraceTransfer] = field(default_factory=list)
    _last_gpu_step: TraceStep | None = field(default=None, repr=False)

    def record_step(self, step: TraceStep) -> None:
        prev = self._last_gpu_step
        self.steps.append(step)

        if step.selected is ExecutionMode.GPU:
            self._last_gpu_step = step
            return

        # CPU step after a prior GPU step => offramp
        if prev is not None and step.selected is ExecutionMode.CPU:
            _trace_warn(
                f"[vibeSpatial] GPU offramp in '{self.pipeline}': "
                f"{prev.surface}({prev.operation}) ran on GPU, "
                f"then {step.surface}({step.operation}) dropped to CPU "
                f"[impl={step.implementation}]",
                stacklevel=4,
            )

    def record_transfer(self, transfer: TraceTransfer) -> None:
        self.transfers.append(transfer)
        prev = self.transfers[-2] if len(self.transfers) >= 2 else None

        if transfer.direction == "d2h" and self._last_gpu_step is not None:
            _trace_warn(
                f"[vibeSpatial] D->H transfer in '{self.pipeline}': "
                f"{transfer.reason} "
                f"(trigger={transfer.trigger})",
                stacklevel=4,
            )

        # Ping-pong: D->H followed by H->D (or vice versa)
        if prev is not None:
            if prev.direction == "d2h" and transfer.direction == "h2d":
                _trace_warn(
                    f"[vibeSpatial] H/D ping-pong in '{self.pipeline}': "
                    f"D->H ({prev.reason}) then H->D ({transfer.reason})",
                    stacklevel=4,
                )
            elif prev.direction == "h2d" and transfer.direction == "d2h":
                _trace_warn(
                    f"[vibeSpatial] D/H ping-pong in '{self.pipeline}': "
                    f"H->D ({prev.reason}) then D->H ({transfer.reason})",
                    stacklevel=4,
                )

    def summary(self) -> dict[str, Any]:
        gpu_steps = [s for s in self.steps if s.selected is ExecutionMode.GPU]
        cpu_steps = [s for s in self.steps if s.selected is ExecutionMode.CPU]
        d2h = [t for t in self.transfers if t.direction == "d2h"]
        h2d = [t for t in self.transfers if t.direction == "h2d"]
        offramps = 0
        prev_gpu = False
        for s in self.steps:
            if s.selected is ExecutionMode.CPU and prev_gpu:
                offramps += 1
            prev_gpu = s.selected is ExecutionMode.GPU
        return {
            "pipeline": self.pipeline,
            "total_steps": len(self.steps),
            "gpu_steps": len(gpu_steps),
            "cpu_steps": len(cpu_steps),
            "d2h_transfers": len(d2h),
            "h2d_transfers": len(h2d),
            "offramps": offramps,
        }


_thread_local = threading.local()


def get_active_trace() -> ExecutionTraceContext | None:
    return getattr(_thread_local, "trace", None)


@contextmanager
def execution_trace(pipeline: str):
    parent = get_active_trace()
    ctx = ExecutionTraceContext(pipeline=pipeline)
    _thread_local.trace = ctx
    try:
        yield ctx
    finally:
        _thread_local.trace = parent


def notify_dispatch(
    *,
    surface: str,
    operation: str,
    selected: ExecutionMode | str,
    implementation: str,
) -> None:
    ctx = get_active_trace()
    if ctx is None:
        return
    mode = selected if isinstance(selected, ExecutionMode) else ExecutionMode(selected)
    ctx.record_step(TraceStep(
        surface=surface,
        operation=operation,
        selected=mode,
        implementation=implementation,
    ))


def notify_transfer(
    *,
    direction: str,
    trigger: str,
    reason: str,
) -> None:
    ctx = get_active_trace()
    if ctx is None:
        return
    ctx.record_transfer(TraceTransfer(
        direction=direction,
        trigger=trigger,
        reason=reason,
    ))
