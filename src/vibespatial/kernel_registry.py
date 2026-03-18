from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from vibespatial.precision import KernelClass, PrecisionMode
from vibespatial.residency import Residency
from vibespatial.runtime import ExecutionMode

KernelCallable = Callable[..., Any]
KERNEL_VARIANTS: dict[str, list[KernelVariantSpec]] = defaultdict(list)


@dataclass(frozen=True)
class KernelVariantSpec:
    kernel_name: str
    variant: str
    qualified_name: str
    kernel_class: KernelClass | None
    execution_modes: tuple[ExecutionMode, ...]
    geometry_families: tuple[str, ...]
    supports_mixed: bool
    preferred_residency: Residency | None
    precision_modes: tuple[PrecisionMode, ...]
    min_rows: int | None
    max_rows: int | None
    tags: tuple[str, ...]


def _normalize_execution_modes(
    variant: str,
    execution_modes: tuple[ExecutionMode | str, ...] | None,
) -> tuple[ExecutionMode, ...]:
    if execution_modes:
        return tuple(
            mode if isinstance(mode, ExecutionMode) else ExecutionMode(mode)
            for mode in execution_modes
        )
    if variant == ExecutionMode.CPU.value:
        return (ExecutionMode.CPU,)
    if variant == ExecutionMode.GPU.value:
        return (ExecutionMode.GPU,)
    return (ExecutionMode.GPU,)


def get_kernel_variants(kernel_name: str) -> tuple[KernelVariantSpec, ...]:
    return tuple(KERNEL_VARIANTS.get(kernel_name, ()))


def register_kernel_variant(
    kernel_name: str,
    variant: str,
    *,
    kernel_class: KernelClass | str | None = None,
    execution_modes: tuple[ExecutionMode | str, ...] | None = None,
    geometry_families: tuple[str, ...] = (),
    supports_mixed: bool = True,
    preferred_residency: Residency | str | None = None,
    precision_modes: tuple[PrecisionMode | str, ...] = (
        PrecisionMode.AUTO,
        PrecisionMode.FP32,
        PrecisionMode.FP64,
    ),
    min_rows: int | None = None,
    max_rows: int | None = None,
    tags: tuple[str, ...] = (),
) -> Callable[[KernelCallable], KernelCallable]:
    """Track declared variants and their dispatch metadata for adaptive planning."""

    def decorator(func: KernelCallable) -> KernelCallable:
        spec = KernelVariantSpec(
            kernel_name=kernel_name,
            variant=variant,
            qualified_name=f"{func.__module__}.{func.__qualname__}",
            kernel_class=kernel_class if isinstance(kernel_class, KernelClass) or kernel_class is None else KernelClass(kernel_class),
            execution_modes=_normalize_execution_modes(variant, execution_modes),
            geometry_families=tuple(geometry_families),
            supports_mixed=supports_mixed,
            preferred_residency=(
                preferred_residency
                if isinstance(preferred_residency, Residency) or preferred_residency is None
                else Residency(preferred_residency)
            ),
            precision_modes=tuple(
                mode if isinstance(mode, PrecisionMode) else PrecisionMode(mode)
                for mode in precision_modes
            ),
            min_rows=min_rows,
            max_rows=max_rows,
            tags=tuple(tags),
        )
        if spec not in KERNEL_VARIANTS[kernel_name]:
            KERNEL_VARIANTS[kernel_name].append(spec)
        return func

    return decorator
