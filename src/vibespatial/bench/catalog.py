"""Benchmark operation registry for vsbench CLI.

Operations self-register via the ``@benchmark_operation`` decorator.
The registry is populated when operation modules under
``vibespatial.bench.operations`` are imported.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Protocol

from .schema import BenchmarkResult

# ---------------------------------------------------------------------------
# Callable protocol
# ---------------------------------------------------------------------------

class BenchmarkCallable(Protocol):
    def __call__(
        self,
        *,
        scale: int,
        repeat: int,
        compare: str | None,
        precision: str,
        nvtx: bool,
        gpu_sparkline: bool,
        trace: bool,
        **kwargs: Any,
    ) -> BenchmarkResult: ...


# ---------------------------------------------------------------------------
# Operation spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OperationSpec:
    """Metadata for a registered benchmark operation."""

    name: str
    description: str
    category: str  # "constructive", "predicate", "spatial", "overlay", "io", "misc"
    geometry_types: tuple[str, ...]
    default_scale: int
    tier: int  # performance tier 1-5
    callable: BenchmarkCallable
    legacy_script: str | None = None
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "geometry_types": list(self.geometry_types),
            "default_scale": self.default_scale,
            "tier": self.tier,
            "tags": list(self.tags),
        }
        if self.legacy_script:
            d["legacy_script"] = self.legacy_script
        return d


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_OPERATION_REGISTRY: OrderedDict[str, OperationSpec] = OrderedDict()


def benchmark_operation(
    name: str,
    *,
    description: str,
    category: str,
    geometry_types: tuple[str, ...] = ("polygon",),
    default_scale: int = 100_000,
    tier: int = 3,
    legacy_script: str | None = None,
    tags: tuple[str, ...] = (),
):
    """Decorator that registers a benchmark operation for CLI discovery."""

    def decorator(func: BenchmarkCallable) -> BenchmarkCallable:
        spec = OperationSpec(
            name=name,
            description=description,
            category=category,
            geometry_types=geometry_types,
            default_scale=default_scale,
            tier=tier,
            callable=func,
            legacy_script=legacy_script,
            tags=tags,
        )
        _OPERATION_REGISTRY[name] = spec
        return func

    return decorator


def get_operation(name: str) -> OperationSpec:
    """Look up a registered operation by name.

    Raises ``KeyError`` if the name is not registered.
    """
    if name not in _OPERATION_REGISTRY:
        available = ", ".join(_OPERATION_REGISTRY) or "(none registered)"
        raise KeyError(
            f"Unknown benchmark operation: {name!r}. "
            f"Available: {available}"
        )
    return _OPERATION_REGISTRY[name]


def list_operations(*, category: str | None = None) -> tuple[OperationSpec, ...]:
    """Return all registered operations, optionally filtered by category."""
    ops = tuple(_OPERATION_REGISTRY.values())
    if category is not None:
        ops = tuple(op for op in ops if op.category == category)
    return ops


def list_categories() -> tuple[str, ...]:
    """Return the distinct categories across all registered operations."""
    seen: dict[str, None] = {}
    for op in _OPERATION_REGISTRY.values():
        seen.setdefault(op.category, None)
    return tuple(seen)


def ensure_operations_loaded() -> None:
    """Import the operations subpackage so all decorators fire.

    This is called lazily by the CLI to avoid slow startup for
    ``vsbench --help``.
    """
    if _OPERATION_REGISTRY:
        return
    from vibespatial.bench import operations as _  # noqa: F401
