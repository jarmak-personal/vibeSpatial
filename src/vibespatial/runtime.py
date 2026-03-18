from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from importlib.util import find_spec

from vibespatial.cuda_runtime import has_cuda_device

EXECUTION_MODE_ENV_VAR = "VIBESPATIAL_EXECUTION_MODE"


class ExecutionMode(StrEnum):
    AUTO = "auto"
    GPU = "gpu"
    CPU = "cpu"


@dataclass(frozen=True)
class RuntimeSelection:
    requested: ExecutionMode
    selected: ExecutionMode
    reason: str


def _has_module(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def has_gpu_runtime() -> bool:
    if not (_has_module("cuda.bindings.driver") or _has_module("cuda")):
        return False
    if not _has_module("cupy"):
        return False
    return has_cuda_device()


def select_runtime(requested: ExecutionMode | str = ExecutionMode.AUTO) -> RuntimeSelection:
    mode = requested if isinstance(requested, ExecutionMode) else ExecutionMode(requested)

    if mode is ExecutionMode.CPU:
        return RuntimeSelection(requested=mode, selected=ExecutionMode.CPU, reason="CPU requested")

    if mode is ExecutionMode.GPU and has_gpu_runtime():
        return RuntimeSelection(
            requested=mode,
            selected=ExecutionMode.GPU,
            reason="CUDA Python runtime detected",
        )

    if mode is ExecutionMode.GPU:
        raise RuntimeError("GPU execution was requested, but no CUDA Python runtime is installed")

    return RuntimeSelection(
        requested=mode,
        selected=ExecutionMode.CPU,
        reason=(
            "auto mode stays on CPU until a kernel-specific planner selects a GPU variant"
            if has_gpu_runtime()
            else "CUDA Python runtime not available; using explicit CPU fallback"
        ),
    )


# ---------------------------------------------------------------------------
# Session-scoped execution mode override (mirrors determinism.py pattern)
# ---------------------------------------------------------------------------

_override_mode: ExecutionMode | None = None


def set_execution_mode(mode: ExecutionMode | str | None) -> None:
    """Override the session execution mode. Pass None to clear.

    Also invalidates the adaptive runtime snapshot cache so the planner
    re-evaluates on the next dispatch.
    """
    global _override_mode
    if mode is None:
        _override_mode = None
    else:
        _override_mode = mode if isinstance(mode, ExecutionMode) else ExecutionMode(mode)
    # Avoid circular import: adaptive_runtime imports from runtime, so
    # we import lazily here.
    try:
        from vibespatial.adaptive_runtime import invalidate_snapshot_cache

        invalidate_snapshot_cache()
    except ImportError:
        pass


def get_requested_mode() -> ExecutionMode:
    """Return the session-wide requested execution mode.

    Priority: explicit set_execution_mode() > env var > AUTO.
    """
    if _override_mode is not None:
        return _override_mode
    raw = os.environ.get(EXECUTION_MODE_ENV_VAR)
    if raw is not None:
        return ExecutionMode(raw.lower())
    return ExecutionMode.AUTO
