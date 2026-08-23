"""Private ownership and admission contracts for point-partition reducers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from vibespatial.api._native_state import NativeStreamReadiness
from vibespatial.cuda._runtime import (
    cuda_stream_identity,
    get_cuda_completion_retainer,
    get_cuda_runtime,
)
from vibespatial.runtime import ExecutionMode, RuntimeSelection
from vibespatial.runtime.precision import (
    KernelClass,
    PrecisionMode,
    PrecisionPlan,
    select_precision_plan,
)

_I32_MAX = (1 << 31) - 1
_I64_MAX = (1 << 63) - 1


@dataclass(frozen=True, slots=True)
class _PointPartitionPlanSeal:
    owner_identity: int
    variant: PointPartitionVariant
    prepared_identity: int
    query_bounds_identity: int
    query_counts_identity: int
    partitions: tuple[tuple[int, int, int], ...]
    pair_budget: int
    relation_admission_identity: int


class PointPartitionVariant(StrEnum):
    GRID = "grid"
    MORTON = "morton"


@dataclass(frozen=True, slots=True)
class PointPartitionCacheKey:
    """Identity of one prepared derivative of a NativeSpatialIndex."""

    variant: PointPartitionVariant
    native_index_identity: int
    source_token: str | None
    row_count: int
    order_identity: int
    morton_identity: int
    bounds_identity: int
    device_id: int
    context_identity: int
    producer_event_identity: int
    parameters: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class PointPartitionQueryPlan:
    """Device query counts and bounded host schedule for one prepared index."""

    owner: Any
    variant: PointPartitionVariant
    prepared: Any
    cache_key: PointPartitionCacheKey
    query_bounds: Any
    query_counts: Any
    partitions: tuple[tuple[int, int, int], ...]
    pair_budget: int
    selection_reason: str
    relation_admission: PointPartitionPreflight | None
    readiness: NativeStreamReadiness
    _seal: object

    def slices(self) -> Iterator[PointPartitionQuerySlice]:
        for ordinal in range(len(self.partitions)):
            yield PointPartitionQuerySlice(plan=self, ordinal=ordinal)

    def validate(self, owner: Any, variant: PointPartitionVariant, prepared: Any) -> None:
        seal = self._seal
        if not isinstance(seal, _PointPartitionPlanSeal):
            raise ValueError("point-partition query plan has no provenance seal")
        if (
            self.owner is not owner
            or self.variant is not variant
            or self.prepared is not prepared
            or self.cache_key != prepared.cache_key
            or int(self.query_bounds.shape[0]) != int(self.query_counts.shape[0])
            or seal.owner_identity != id(owner)
            or seal.variant is not variant
            or seal.prepared_identity != id(prepared)
            or seal.query_bounds_identity != id(self.query_bounds)
            or seal.query_counts_identity != id(self.query_counts)
            or seal.partitions != self.partitions
            or seal.pair_budget != int(self.pair_budget)
            or seal.relation_admission_identity
            != id(self.relation_admission)
        ):
            raise ValueError("point-partition query-plan provenance was modified")


@dataclass(frozen=True, slots=True)
class PointPartitionQuerySlice:
    """Provenance-bound capacity token for one candidate scatter."""

    plan: PointPartitionQueryPlan
    ordinal: int

    def __post_init__(self) -> None:
        if not 0 <= int(self.ordinal) < len(self.plan.partitions):
            raise ValueError("point-partition query-slice ordinal is out of range")

    @property
    def query_start(self) -> int:
        return int(self.plan.partitions[self.ordinal][0])

    @property
    def query_stop(self) -> int:
        return int(self.plan.partitions[self.ordinal][1])

    @property
    def capacity(self) -> int:
        return int(self.plan.partitions[self.ordinal][2])

    @property
    def oversized(self) -> bool:
        return self.capacity > int(self.plan.pair_budget)

    @property
    def query_bounds(self):
        return self.plan.query_bounds[self.query_start : self.query_stop]

    @property
    def query_counts(self):
        return self.plan.query_counts[self.query_start : self.query_stop]

    def validate(self, owner: Any, variant: PointPartitionVariant, prepared: Any) -> None:
        self.plan.validate(owner, variant, prepared)


@dataclass(frozen=True, slots=True)
class PointPartitionDecline:
    variant: PointPartitionVariant
    reason: str
    memory_decline: bool = False


@dataclass(frozen=True, slots=True)
class PointPartitionPreflight:
    """Pure provider shape and memory estimate before provider submission."""

    variant: PointPartitionVariant
    required_bytes: int
    reason: str
    owner_identity: int
    query_count: int
    pair_budget: int
    cache_key: PointPartitionCacheKey
    admitted: bool = False

    def validate_admission(
        self,
        owner: Any,
        variant: PointPartitionVariant,
        *,
        query_count: int,
        pair_budget: int,
        cache_key: PointPartitionCacheKey,
        required_bytes: int,
    ) -> None:
        if (
            not self.admitted
            or self.owner_identity != id(owner)
            or self.variant is not variant
            or self.query_count != int(query_count)
            or self.pair_budget != int(pair_budget)
            or self.cache_key != cache_key
            or int(required_bytes) > self.required_bytes
        ):
            raise ValueError("point-partition admission token provenance mismatch")


_TEST_VARIANT: ContextVar[PointPartitionVariant | None] = ContextVar(
    "vibespatial_point_partition_test_variant",
    default=None,
)


@contextmanager
def force_point_partition_variant_for_testing(
    variant: PointPartitionVariant | str | None,
):
    """Privately force a provider in tests and profiling, never public dispatch."""
    normalized = None if variant is None else PointPartitionVariant(variant)
    token = _TEST_VARIANT.set(normalized)
    try:
        yield
    finally:
        _TEST_VARIANT.reset(token)


def forced_point_partition_variant_for_testing() -> PointPartitionVariant | None:
    return _TEST_VARIANT.get(None)


def checked_i32(value: int, *, name: str) -> int:
    result = int(value)
    if not 0 <= result <= _I32_MAX:
        raise OverflowError(f"{name} does not fit a nonnegative int32")
    return result


def checked_i64(value: int, *, name: str) -> int:
    result = int(value)
    if not 0 <= result <= _I64_MAX:
        raise OverflowError(f"{name} does not fit a nonnegative int64")
    return result


def checked_product(left: int, right: int, *, name: str) -> int:
    left = checked_i64(left, name=f"{name} left operand")
    right = checked_i64(right, name=f"{name} right operand")
    if left and right > _I64_MAX // left:
        raise OverflowError(f"{name} exceeds int64 capacity")
    return left * right


def checked_sum(*values: int, name: str) -> int:
    total = 0
    for ordinal, value in enumerate(values):
        value = checked_i64(value, name=f"{name} operand {ordinal}")
        if value > _I64_MAX - total:
            raise OverflowError(f"{name} exceeds int64 capacity")
        total += value
    return total


def point_partition_fp64_coarse_plan() -> PrecisionPlan:
    """Return the explicit fp64 COARSE plan required by conservative bounds."""
    plan = select_precision_plan(
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            reason="point-partition conservative bounds require fp64",
        ),
        kernel_class=KernelClass.COARSE,
        requested=PrecisionMode.FP64,
    )
    if plan.compute_precision is not PrecisionMode.FP64:
        raise RuntimeError("point-partition COARSE precision must resolve to fp64")
    return plan


def point_partition_all_bounds_finite(native_index: Any) -> bool:
    """Return and cache the one-scalar finite-row admission proof."""
    with native_index.point_partition_lock:
        metadata = native_index.metadata
        bounds = None if metadata is None else metadata.bounds
        if bounds is None:
            return False
        cache_key = f"point_partition_all_finite:{id(bounds)}"
        cached = native_index.index_parameters.get(cache_key)
        if cached is not None:
            return bool(cached)
        if hasattr(bounds, "__cuda_array_interface__"):
            import cupy as cp

            # Selection is itself a consumer of carrier-owned device bounds.
            # Establish the producer -> consumer dependency before the first
            # preflight read, not only later when a provider is prepared.
            wait_for_point_partition(native_index.readiness)
            proof = cp.isfinite(cp.asarray(bounds, dtype=cp.float64)).all().reshape(1)
            value = bool(
                get_cuda_runtime().copy_device_to_host(
                    proof,
                    reason="point-partition finite-bounds preflight planning packet",
                )[0]
            )
        else:
            import numpy as np

            value = bool(np.isfinite(np.asarray(bounds, dtype=np.float64)).all())
        native_index.index_parameters[cache_key] = value
        return value


def point_partition_cache_key(
    native_index: Any,
    variant: PointPartitionVariant,
    *,
    parameters: dict[str, int],
) -> PointPartitionCacheKey:
    import cupy as cp

    checked_i32(native_index.row_count, name="point-partition row count")
    metadata = native_index.metadata
    bounds = None if metadata is None else metadata.bounds
    # Device ordinals are not sufficient cache identity after a context reset:
    # raw pointers from the prior context must never be reused.
    cp.empty(0, dtype=cp.uint8)
    context_identity = int(cp.cuda.driver.ctxGetCurrent())
    producer_event = getattr(native_index.readiness, "event", None)
    return PointPartitionCacheKey(
        variant=variant,
        native_index_identity=id(native_index),
        source_token=native_index.source_token,
        row_count=int(native_index.row_count),
        order_identity=id(native_index.order),
        morton_identity=id(native_index.morton_keys),
        bounds_identity=id(bounds),
        device_id=int(cp.cuda.runtime.getDevice()),
        context_identity=context_identity,
        producer_event_identity=id(producer_event),
        parameters=tuple(sorted((str(key), int(value)) for key, value in parameters.items())),
    )


def cached_point_partition(native_index: Any, key: PointPartitionCacheKey):
    return native_index.point_partition_cache.get(key)


def publish_point_partition(
    native_index: Any,
    key: PointPartitionCacheKey,
    prepared: Any,
) -> Any:
    """Atomically publish only a fully formed preparation."""
    existing = native_index.point_partition_cache.setdefault(key, prepared)
    if existing is not prepared:
        wait_for_point_partition(existing.readiness)
    return existing


def record_point_partition_readiness() -> NativeStreamReadiness:
    import cupy as cp

    stream = cp.cuda.get_current_stream()
    event = cp.cuda.Event(disable_timing=True)
    event.record(stream)
    return NativeStreamReadiness(stream=stream, event=event, ready=False)


def wait_for_point_partition(readiness: NativeStreamReadiness) -> None:
    if readiness.ready or readiness.event is None:
        return
    import cupy as cp

    consumer = cp.cuda.get_current_stream()
    if readiness.stream is not None and (
        cuda_stream_identity(readiness.stream) == cuda_stream_identity(consumer)
    ):
        return
    consumer.wait_event(readiness.event)


def retain_point_partition_completion(*owners: Any) -> None:
    """Keep raw-pointer producers alive through the active stream's work."""
    import cupy as cp

    get_cuda_completion_retainer().defer(
        cp.cuda.get_current_stream(),
        owners,
        lambda _owners: None,
    )


def query_plan(
    *,
    owner: Any,
    variant: PointPartitionVariant,
    prepared: Any,
    query_bounds: Any,
    query_counts: Any,
    partitions: tuple[tuple[int, int, int], ...],
    pair_budget: int,
    selection_reason: str = "",
    relation_admission: PointPartitionPreflight | None = None,
) -> PointPartitionQueryPlan:
    checked_i64(pair_budget, name="point-partition pair budget")
    cursor = 0
    for start, stop, capacity in partitions:
        if not 0 <= int(start) <= int(stop) <= int(query_bounds.shape[0]):
            raise ValueError("point-partition query partition is out of bounds")
        if int(start) != cursor:
            raise ValueError("point-partition query partitions must be contiguous")
        cursor = int(stop)
        checked_i64(capacity, name="point-partition query-slice capacity")
        if int(capacity) > int(pair_budget) and int(stop) != int(start) + 1:
            raise ValueError("oversized point-partition slices must contain one row")
    if cursor != int(query_bounds.shape[0]):
        raise ValueError("point-partition query partitions must cover every query row")
    readiness = record_point_partition_readiness()
    seal = _PointPartitionPlanSeal(
        owner_identity=id(owner),
        variant=variant,
        prepared_identity=id(prepared),
        query_bounds_identity=id(query_bounds),
        query_counts_identity=id(query_counts),
        partitions=partitions,
        pair_budget=int(pair_budget),
        relation_admission_identity=id(relation_admission),
    )
    return PointPartitionQueryPlan(
        owner=owner,
        variant=variant,
        prepared=prepared,
        cache_key=prepared.cache_key,
        query_bounds=query_bounds,
        query_counts=query_counts,
        partitions=partitions,
        pair_budget=int(pair_budget),
        selection_reason=str(selection_reason),
        relation_admission=relation_admission,
        readiness=readiness,
        _seal=seal,
    )


__all__ = [
    "PointPartitionCacheKey",
    "PointPartitionDecline",
    "PointPartitionPreflight",
    "PointPartitionQueryPlan",
    "PointPartitionQuerySlice",
    "PointPartitionVariant",
]
