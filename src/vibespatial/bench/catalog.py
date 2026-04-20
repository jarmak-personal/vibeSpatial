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
        input_format: str,
        nvtx: bool,
        gpu_sparkline: bool,
        trace: bool,
        **kwargs: Any,
    ) -> BenchmarkResult: ...


# ---------------------------------------------------------------------------
# Operation parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OperationParameterSpec:
    """Typed operation-specific argument metadata for ``vsbench run``."""

    name: str
    value_type: str  # "str", "int", "float", "bool", "choice", "float_list"
    description: str
    default: Any | None = None
    choices: tuple[str, ...] = ()
    arity: int | None = None

    def parse(self, raw: str) -> Any:
        """Parse a CLI ``key=value`` string into the declared runtime type."""
        if self.value_type == "str":
            return raw
        if self.value_type == "int":
            return int(raw)
        if self.value_type == "float":
            return float(raw)
        if self.value_type == "bool":
            lowered = raw.strip().lower()
            truthy = {"1", "true", "yes", "on"}
            falsy = {"0", "false", "no", "off"}
            if lowered in truthy:
                return True
            if lowered in falsy:
                return False
            raise ValueError(
                f"invalid boolean value {raw!r} for {self.name}; expected one of "
                f"{sorted(truthy | falsy)}"
            )
        if self.value_type == "choice":
            if raw not in self.choices:
                raise ValueError(
                    f"invalid value {raw!r} for {self.name}; expected one of {list(self.choices)}"
                )
            return raw
        if self.value_type == "float_list":
            values = [float(part.strip()) for part in raw.split(",") if part.strip()]
            if self.arity is not None and len(values) != self.arity:
                raise ValueError(
                    f"invalid value {raw!r} for {self.name}; expected {self.arity} comma-separated floats"
                )
            return tuple(values)
        raise ValueError(f"unsupported operation parameter type: {self.value_type}")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "type": self.value_type,
            "description": self.description,
        }
        if self.default is not None:
            d["default"] = self.default
        if self.choices:
            d["choices"] = list(self.choices)
        if self.arity is not None:
            d["arity"] = self.arity
        return d


def resolve_operation_args(
    spec: OperationSpec,
    raw_args: list[str] | tuple[str, ...] | None,
) -> dict[str, Any]:
    """Validate and coerce operation-specific CLI args against the operation schema."""
    params = {param.name: param for param in spec.parameters}
    resolved = {
        param.name: param.default
        for param in spec.parameters
        if param.default is not None
    }
    if not raw_args:
        return resolved

    for item in raw_args:
        if "=" not in item:
            raise ValueError(
                f"invalid operation arg {item!r}; expected key=value for operation {spec.name!r}"
            )
        key, raw_value = item.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        param = params.get(key)
        if param is None:
            available = ", ".join(sorted(params)) or "(none)"
            raise ValueError(
                f"unknown operation arg {key!r} for operation {spec.name!r}; available: {available}"
            )
        try:
            resolved[key] = param.parse(raw_value)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
    return resolved


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
    parameters: tuple[OperationParameterSpec, ...] = ()
    tags: tuple[str, ...] = ()
    max_scale: int | None = None
    public_api: bool = True

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
        if self.parameters:
            d["parameters"] = [param.to_dict() for param in self.parameters]
        if self.max_scale is not None:
            d["max_scale"] = self.max_scale
        d["public_api"] = self.public_api
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
    parameters: tuple[OperationParameterSpec, ...] = (),
    tags: tuple[str, ...] = (),
    max_scale: int | None = None,
    public_api: bool = True,
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
            parameters=parameters,
            tags=tags,
            max_scale=max_scale,
            public_api=public_api,
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


def list_operations(
    *,
    category: str | None = None,
    include_internal: bool = False,
) -> tuple[OperationSpec, ...]:
    """Return all registered operations, optionally filtered by category."""
    ops = tuple(_OPERATION_REGISTRY.values())
    if category is not None:
        ops = tuple(op for op in ops if op.category == category)
    if not include_internal:
        ops = tuple(op for op in ops if op.public_api)
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
