from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from importlib.util import find_spec

from vibespatial.cuda_runtime import has_cuda_device


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
