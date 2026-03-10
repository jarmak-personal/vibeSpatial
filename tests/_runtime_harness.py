from __future__ import annotations

from collections.abc import Iterable

from vibespatial import ExecutionMode, has_gpu_runtime


def cuda_runtime_available() -> bool:
    return has_gpu_runtime()


def normalize_dispatch_modes(values: Iterable[ExecutionMode | str]) -> tuple[ExecutionMode, ...]:
    normalized: list[ExecutionMode] = []
    seen: set[ExecutionMode] = set()

    for value in values:
        mode = value if isinstance(value, ExecutionMode) else ExecutionMode(value)
        if mode in seen:
            continue
        seen.add(mode)
        normalized.append(mode)

    return tuple(normalized)


def resolve_dispatch_modes(
    requested_modes: Iterable[ExecutionMode | str] | None,
    *,
    cuda_available: bool,
    run_gpu: bool,
) -> tuple[ExecutionMode, ...]:
    if requested_modes:
        modes = normalize_dispatch_modes(requested_modes)
        if ExecutionMode.GPU in modes and not cuda_available:
            raise ValueError("GPU dispatch mode was requested, but no CUDA runtime is available")
        return modes

    modes = [ExecutionMode.CPU]
    if run_gpu and cuda_available:
        modes.append(ExecutionMode.GPU)
    return tuple(modes)
