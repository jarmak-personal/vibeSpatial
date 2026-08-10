from __future__ import annotations

import threading
from contextlib import nullcontext
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from ._runtime import (
    CudaStreamIdentity,
    DeviceArray,
    cuda_stream_identity,
    get_cuda_runtime,
)

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    cp = None

try:
    from cuda.compute import algorithms
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    algorithms = None


class PairSortStrategy(StrEnum):
    AUTO = "auto"
    CUPY = "cupy"
    RADIX = "radix"
    MERGE = "merge"


class CompactionStrategy(StrEnum):
    AUTO = "auto"
    CUPY = "cupy"
    CCCL_SELECT = "cccl-select"


class ScanStrategy(StrEnum):
    AUTO = "auto"
    CUPY = "cupy"
    CCCL_EXCLUSIVE_SCAN = "cccl-exclusive-scan"


class ReductionStrategy(StrEnum):
    AUTO = "auto"
    CUPY = "cupy"
    CCCL_REDUCE = "cccl-reduce"


class SegmentedSortStrategy(StrEnum):
    AUTO = "auto"
    CCCL_SEGMENTED_SORT = "cccl-segmented-sort"


class PartitionStrategy(StrEnum):
    AUTO = "auto"
    CCCL_THREE_WAY = "cccl-three-way"


@dataclass(frozen=True, slots=True)
class CompactionResult:
    values: DeviceArray
    count: int


@dataclass(frozen=True, slots=True)
class PairSortResult:
    keys: DeviceArray
    values: DeviceArray | None
    strategy: PairSortStrategy


@dataclass(frozen=True, slots=True)
class UniqueByKeyResult:
    keys: DeviceArray
    values: DeviceArray
    count: int


@dataclass(frozen=True, slots=True)
class SegmentedReduceResult:
    values: DeviceArray
    segment_count: int


@dataclass(frozen=True, slots=True)
class SegmentedSortResult:
    keys: DeviceArray
    values: DeviceArray | None
    segment_count: int


@dataclass(frozen=True, slots=True)
class ThreeWayPartitionResult:
    values: DeviceArray
    first_count: int
    second_count: int


def has_cccl_primitives() -> bool:
    return cp is not None and algorithms is not None


def _require_cccl_primitives() -> tuple[object, object]:
    if cp is None or algorithms is None:
        raise RuntimeError("CCCL Python primitives are not installed")
    return cp, algorithms


def _validate_vector(name: str, values: DeviceArray) -> None:
    if getattr(values, "ndim", None) != 1:
        raise ValueError(f"{name} must be a 1D device array")


def _device_int_scalar(value, *, reason: str) -> int:
    host = get_cuda_runtime().copy_device_to_host(value, reason=reason)
    return int(np.asarray(host).reshape(-1)[0])


def _sum_op(left, right):  # pragma: no cover - exercised through CCCL JIT
    return left + right


def _less_than(left, right):  # pragma: no cover - exercised through CCCL JIT
    return left < right


def _greater_than(left, right):  # pragma: no cover - exercised through CCCL JIT
    return left > right


def _equal_to(left, right):  # pragma: no cover - exercised through CCCL JIT
    return left == right


def _min_op(left, right):  # pragma: no cover - exercised through CCCL JIT
    return left if left < right else right


def _max_op(left, right):  # pragma: no cover - exercised through CCCL JIT
    return left if left > right else right


# ---------------------------------------------------------------------------
# make_* fast path helpers (ADR-0034, Phase 3)
# ---------------------------------------------------------------------------

_DTYPE_SUFFIX = {
    np.dtype(np.int32): "i32",
    np.dtype(np.int64): "i64",
    np.dtype(np.uint64): "u64",
    np.dtype(np.float64): "f64",
}


def _dtype_suffix(dtype: np.dtype) -> str:
    return _DTYPE_SUFFIX.get(np.dtype(dtype), "")


def _get_precompiled(name: str):
    """Get one typed primitive, recovering warm state after test/runtime resets."""
    from .cccl_precompile import get_or_request_compiled

    return get_or_request_compiled(name)


def _cccl_is_warm(spec_name: str) -> bool:
    """Check if a CCCL spec is compiled or deferred on disk (no blocking wait)."""
    from .cccl_precompile import CCCLPrecompiler

    if CCCLPrecompiler._instance is None:
        return False
    inst = CCCLPrecompiler._instance
    return spec_name in inst._cache or spec_name in inst._deferred_disk


def _effective_stream(cp_module, stream):
    if stream is not None:
        return stream
    return cp_module.cuda.get_current_stream()


def _stream_context(cp_module, stream):
    if stream is None:
        return nullcontext()
    if isinstance(stream, cp_module.cuda.Stream):
        return stream
    if callable(getattr(stream, "__cuda_stream__", None)):
        return cp_module.cuda.Stream.from_external(stream)
    from cuda.compute._utils.protocols import validate_and_get_stream

    return cp_module.cuda.ExternalStream(validate_and_get_stream(stream))


def _stream_synchronize(cp_module, stream) -> None:
    effective = _effective_stream(cp_module, stream)
    synchronize = getattr(effective, "synchronize", None)
    if callable(synchronize):
        synchronize()
        return
    from cuda.compute._utils.protocols import validate_and_get_stream

    cp_module.cuda.ExternalStream(validate_and_get_stream(effective)).synchronize()


def _release_cccl_pointer_owners(make_callable) -> None:
    """Clear array owners retained by a callable and its CCCL adapters."""
    pending = [make_callable]
    visited: set[int] = set()
    while pending:
        callable_part = pending.pop()
        identity = id(callable_part)
        if identity in visited:
            continue
        visited.add(identity)
        inner = getattr(callable_part, "_inner", None)
        if inner is not None:
            pending.append(inner)
        for name in dir(callable_part):
            if not name.endswith("_cccl"):
                continue
            iterator = getattr(callable_part, name, None)
            is_pointer = getattr(iterator, "is_kind_pointer", None)
            if callable(is_pointer) and bool(is_pointer()):
                iterator.state = 0


def _invoke_precompiled(make_callable, *args, stream=None):
    effective_stream = _effective_stream(cp, stream)
    try:
        try:
            return make_callable(*args, stream=effective_stream)
        except TypeError as exc:
            if "unexpected keyword argument 'stream'" not in str(exc):
                raise
            return make_callable(*args)
    finally:
        # Kernel arguments are copied at launch, so the completion retainer is
        # the sole owner needed after this call returns.
        _release_cccl_pointer_owners(make_callable)


def _query_precompiled_scratch_bytes(make_callable, *args) -> int:
    """Query scratch capacity without leaving pointer iterators bound."""
    try:
        needed = make_callable(None, *args)
        return max(int(needed) if needed else 1, 1)
    finally:
        _release_cccl_pointer_owners(make_callable)


@dataclass(slots=True)
class _PrecompiledScratchSlot:
    temp: Any
    temp_storage_bytes: int
    high_water_n: int
    stream: Any | None = None
    stream_key: CudaStreamIdentity | None = None
    token: object | None = None


def _precompiled_lock(precompiled) -> threading.RLock:
    lock = getattr(precompiled, "invocation_lock", None)
    if lock is None:
        lock = threading.RLock()
        precompiled.invocation_lock = lock
    return lock


def _completion_retainer():
    from ._runtime import get_cuda_completion_retainer

    return get_cuda_completion_retainer()


def _precompiled_slots(precompiled) -> list[_PrecompiledScratchSlot]:
    slots = getattr(precompiled, "scratch_slots", None)
    if slots is None:
        slots = []
        precompiled.scratch_slots = slots
    if not slots:
        slots.append(
            _PrecompiledScratchSlot(
                temp=precompiled.temp_storage,
                temp_storage_bytes=int(precompiled.temp_storage_bytes),
                high_water_n=int(precompiled.high_water_n),
            )
        )
    return slots


def _precompiled_slots_for_execution(
    precompiled,
) -> tuple[list[_PrecompiledScratchSlot], bool]:
    """Return scratch slots without mutating an empty cache before allocation."""
    slots = getattr(precompiled, "scratch_slots", None)
    if slots:
        return slots, False
    return [
        _PrecompiledScratchSlot(
            temp=precompiled.temp_storage,
            temp_storage_bytes=int(precompiled.temp_storage_bytes),
            high_water_n=int(precompiled.high_water_n),
        )
    ], True


def _retire_precompiled_launch(
    payload: tuple[Any, _PrecompiledScratchSlot, object, tuple[Any, ...]],
) -> None:
    """Release one scratch lease after its stream completion event."""
    precompiled, slot, token, _references = payload
    with _precompiled_lock(precompiled):
        if slot.token is not token:
            return
        slot.stream = None
        slot.stream_key = None
        slot.token = None
        slots = _precompiled_slots(precompiled)
        if slot is not slots[0]:
            slots.remove(slot)


def _execute_precompiled(
    precompiled,
    cp_module,
    *,
    num_items: int,
    args: tuple[Any, ...],
    stream=None,
    synchronize: bool,
    references: tuple[Any, ...] = (),
) -> None:
    """Launch a cached CCCL callable with a concurrency-safe scratch lease."""
    effective_stream = _effective_stream(cp_module, stream)
    stream_key = cuda_stream_identity(effective_stream)
    lock = _precompiled_lock(precompiled)
    retainer = _completion_retainer()
    retirement = None
    try:
        with lock:
            slots, slots_need_commit = _precompiled_slots_for_execution(precompiled)
            slot = next(
                (
                    candidate
                    for candidate in slots
                    if candidate.token is None or candidate.stream_key == stream_key
                ),
                None,
            )
            if slot is None:
                needed = _query_precompiled_scratch_bytes(
                    precompiled.make_callable,
                    *args,
                )
                slot = _PrecompiledScratchSlot(
                    temp=cp_module.empty(needed, dtype=cp_module.uint8),
                    temp_storage_bytes=needed,
                    high_water_n=num_items,
                )
                slots.append(slot)
            elif num_items > slot.high_water_n:
                needed = _query_precompiled_scratch_bytes(
                    precompiled.make_callable,
                    *args,
                )
                if needed > slot.temp_storage_bytes:
                    slot.temp = cp_module.empty(needed, dtype=cp_module.uint8)
                    slot.temp_storage_bytes = needed
                slot.high_water_n = num_items
                if slot is slots[0]:
                    precompiled.temp_storage = slot.temp
                    precompiled.temp_storage_bytes = slot.temp_storage_bytes
                    precompiled.high_water_n = slot.high_water_n

            if slots_need_commit:
                cached_slots = getattr(precompiled, "scratch_slots", None)
                if cached_slots is None:
                    precompiled.scratch_slots = slots
                else:
                    cached_slots.extend(slots)

            token = object()
            slot.stream = effective_stream
            slot.stream_key = stream_key
            slot.token = token
            retirement = (precompiled, slot, token, (slot.temp, *references))
            _invoke_precompiled(
                precompiled.make_callable,
                slot.temp,
                *args,
                stream=effective_stream,
            )
    except BaseException:
        if retirement is not None:
            retainer.defer(
                effective_stream,
                retirement,
                _retire_precompiled_launch,
            )
        raise

    if synchronize:
        claimed = retainer.claim_stream_retirements(effective_stream)
        try:
            _stream_synchronize(cp_module, effective_stream)
        except BaseException:
            retainer.restore_stream_retirements(effective_stream, claimed)
            retainer.defer(
                effective_stream,
                retirement,
                _retire_precompiled_launch,
            )
            raise
        _retire_precompiled_launch(retirement)
        retainer.release_claimed_retirements(claimed)
        return

    retainer.defer(
        effective_stream,
        retirement,
        _retire_precompiled_launch,
    )


def _release_one_shot_references(_references: tuple[Any, ...]) -> None:
    return


def _execute_one_shot(
    invoke,
    cp_module,
    *,
    args: tuple[Any, ...],
    stream=None,
    synchronize: bool,
    references: tuple[Any, ...] = (),
) -> Any:
    """Invoke an uncached CCCL launch with completion-scoped ownership."""
    effective_stream = _effective_stream(cp_module, stream)
    retainer = _completion_retainer()
    retirement = (*args, *references)
    try:
        result = invoke(*args, stream=effective_stream)
    except BaseException:
        retainer.defer(
            effective_stream,
            retirement,
            _release_one_shot_references,
        )
        raise
    if not synchronize:
        retainer.defer(
            effective_stream,
            retirement,
            _release_one_shot_references,
        )
        return result
    claimed = retainer.claim_stream_retirements(effective_stream)
    try:
        _stream_synchronize(cp_module, effective_stream)
    except BaseException:
        retainer.restore_stream_retirements(effective_stream, claimed)
        retainer.defer(
            effective_stream,
            retirement,
            _release_one_shot_references,
        )
        raise
    retainer.release_claimed_retirements(claimed)
    return result


def _active_precompiled_launch_count(precompiled) -> int:
    """Return active scratch leases for regression tests and diagnostics."""
    with _precompiled_lock(precompiled):
        return sum(slot.token is not None for slot in _precompiled_slots(precompiled))


# ---------------------------------------------------------------------------
# Compaction (bool mask → indices)
# ---------------------------------------------------------------------------
# ADR-0033 benchmark (2026-03-12): CCCL select beats CuPy flatnonzero at all
# scales once JIT is warm.  make_* is 1.4-3.1x faster than CuPy.
# Cold-call penalty is ~950ms (one-time per process).


def _compact_indices_cccl(mask: DeviceArray, *, stream=None) -> CompactionResult:
    cp_module, cccl_algorithms = _require_cccl_primitives()
    stream = _effective_stream(cp_module, stream)
    _validate_vector("mask", mask)
    item_count = int(mask.size)
    out = cp_module.empty((item_count,), dtype=cp_module.int32)
    if item_count == 0:
        return CompactionResult(values=out, count=0)

    count = cp_module.empty((1,), dtype=cp_module.int32)
    indices = cp_module.arange(item_count, dtype=cp_module.int32)

    def _selected(index):  # pragma: no cover - exercised through CCCL JIT
        return mask[index] != 0

    _execute_one_shot(
        cccl_algorithms.select,
        cp_module,
        args=(indices, out, count, _selected, item_count),
        stream=stream,
        synchronize=True,
        references=(mask,),
    )
    # Returning a sliced output requires an explicit host count materialization.
    selected_count = _device_int_scalar(
        count,
        reason="cccl select selected-count scalar fence",
    )
    return CompactionResult(values=out[:selected_count], count=selected_count)


def compact_indices(
    mask: DeviceArray,
    *,
    strategy: CompactionStrategy | str = CompactionStrategy.AUTO,
    stream=None,
) -> CompactionResult:
    cp_module, _ = _require_cccl_primitives()
    _validate_vector("mask", mask)
    resolved = (
        strategy if isinstance(strategy, CompactionStrategy) else CompactionStrategy(strategy)
    )
    if resolved is CompactionStrategy.AUTO:
        # Unlike exclusive_sum (where make_* works because _sum_op is a
        # stateless module-level function), make_select bakes the
        # predicate closure's device pointers into the compiled kernel,
        # so the precompiled callable cannot be reused with different
        # mask arrays.  The one-shot select() API re-JITs per array
        # size class (~5-6s each).  CuPy flatnonzero is 0.2ms with no
        # JIT and comparable warm throughput.
        resolved = CompactionStrategy.CUPY
    if resolved is CompactionStrategy.CUPY:
        with _stream_context(cp_module, stream):
            out = cp_module.flatnonzero(mask).astype(cp_module.int32, copy=False)
        return CompactionResult(values=out, count=int(out.size))
    return _compact_indices_cccl(mask, stream=stream)


# ---------------------------------------------------------------------------
# Exclusive prefix sum
# ---------------------------------------------------------------------------
# ADR-0033 benchmark (2026-03-12): CCCL exclusive_scan beats CuPy cumsum at
# >=1M elements even without make_*.  make_* is 1.8-3.7x faster than CuPy
# across all scales.  Cold-call penalty is ~1460ms (one-time per process).


def _exclusive_sum_cccl(
    values: DeviceArray,
    *,
    synchronize: bool = False,
    stream=None,
) -> DeviceArray:
    cp_module, cccl_algorithms = _require_cccl_primitives()
    stream = _effective_stream(cp_module, stream)
    _validate_vector("values", values)
    out = cp_module.empty_like(values)
    if values.size == 0:
        return out
    n = int(values.size)
    init = np.asarray(0, dtype=values.dtype)
    # make_* fast path (ADR-0034)
    precompiled = _get_precompiled(f"exclusive_scan_{_dtype_suffix(values.dtype)}")
    if precompiled is not None:
        _execute_precompiled(
            precompiled,
            cp_module,
            num_items=n,
            args=(values, out, _sum_op, n, init),
            stream=stream,
            synchronize=synchronize,
            references=(values, out),
        )
        return out
    # Fallback: one-shot API
    _execute_one_shot(
        cccl_algorithms.exclusive_scan,
        cp_module,
        args=(values, out, _sum_op, init, n),
        stream=stream,
        synchronize=synchronize,
    )
    return out


def exclusive_sum(
    values: DeviceArray,
    *,
    strategy: ScanStrategy | str = ScanStrategy.AUTO,
    synchronize: bool = False,
    stream=None,
) -> DeviceArray:
    cp_module, _ = _require_cccl_primitives()
    _validate_vector("values", values)
    resolved = strategy if isinstance(strategy, ScanStrategy) else ScanStrategy(strategy)
    if resolved is ScanStrategy.AUTO:
        # CCCL exclusive_scan beats CuPy at 10M+ scale once JIT is warm.
        # At <=1M the difference is marginal (<0.01ms).  Use CuPy when
        # CCCL is cold to avoid 5-11s JIT overhead (ADR-0034 finding).
        spec = f"exclusive_scan_{_dtype_suffix(values.dtype)}"
        resolved = ScanStrategy.CCCL_EXCLUSIVE_SCAN if _cccl_is_warm(spec) else ScanStrategy.CUPY
    if resolved is ScanStrategy.CUPY:
        with _stream_context(cp_module, stream):
            out = cp_module.cumsum(values, dtype=values.dtype)
            out -= values
        if synchronize:
            _stream_synchronize(cp_module, stream)
        return out
    return _exclusive_sum_cccl(values, synchronize=synchronize, stream=stream)


def inclusive_max(
    values: DeviceArray,
    *,
    synchronize: bool = False,
    stream=None,
) -> DeviceArray:
    """Return an inclusive prefix maximum without a host cardinality fence."""
    cp_module, cccl_algorithms = _require_cccl_primitives()
    stream = _effective_stream(cp_module, stream)
    _validate_vector("values", values)
    out = cp_module.empty_like(values)
    if values.size == 0:
        return out
    dtype = np.dtype(values.dtype)
    if np.issubdtype(dtype, np.floating):
        init_value = np.asarray(-np.inf, dtype=dtype)
    elif np.issubdtype(dtype, np.integer):
        init_value = np.asarray(np.iinfo(dtype).min, dtype=dtype)
    else:
        raise TypeError("inclusive_max supports floating-point and integer device vectors")
    _execute_one_shot(
        cccl_algorithms.inclusive_scan,
        cp_module,
        args=(values, out, _max_op, init_value, int(values.size)),
        stream=stream,
        synchronize=synchronize,
    )
    return out


# ---------------------------------------------------------------------------
# Reduction
# ---------------------------------------------------------------------------
# ADR-0033 benchmark (2026-03-12): CuPy cp.sum is faster than CCCL
# reduce_into at small scales (<1M), but make_* closes the gap.
# Keeping CuPy as default for now; CCCL reduce_into is available for
# fused pipelines via TransformIterator.


def reduce_sum(
    values: DeviceArray,
    *,
    strategy: ReductionStrategy | str = ReductionStrategy.AUTO,
    synchronize: bool = False,
    stream=None,
) -> DeviceArray:
    """Reduce values to a single sum.  Returns a 1-element device array."""
    cp_module, cccl_algorithms = _require_cccl_primitives()
    stream = _effective_stream(cp_module, stream)
    _validate_vector("values", values)
    resolved = strategy if isinstance(strategy, ReductionStrategy) else ReductionStrategy(strategy)
    if resolved is ReductionStrategy.AUTO:
        # CuPy sum is marginally faster at small scales; CCCL make_* is
        # competitive but the win is small enough to keep CuPy as default.
        resolved = ReductionStrategy.CUPY
    if resolved is ReductionStrategy.CUPY:
        with _stream_context(cp_module, stream):
            out = cp_module.empty(1, dtype=values.dtype)
            out[0] = cp_module.sum(values)
        if synchronize:
            _stream_synchronize(cp_module, stream)
        return out
    out = cp_module.empty(1, dtype=values.dtype)
    h_init = np.asarray(0, dtype=values.dtype)
    n = int(values.size)
    # make_* fast path (ADR-0034)
    precompiled = _get_precompiled(f"reduce_sum_{_dtype_suffix(values.dtype)}")
    if precompiled is not None:
        _execute_precompiled(
            precompiled,
            cp_module,
            num_items=n,
            args=(values, out, _sum_op, n, h_init),
            stream=stream,
            synchronize=synchronize,
            references=(values, out),
        )
        return out
    # Fallback: one-shot API
    _execute_one_shot(
        cccl_algorithms.reduce_into,
        cp_module,
        args=(values, out, _sum_op, n, h_init),
        stream=stream,
        synchronize=synchronize,
    )
    return out


# ---------------------------------------------------------------------------
# Segmented reduce
# ---------------------------------------------------------------------------
# ADR-0033 benchmark (2026-03-12): CCCL segmented_reduce is 1.4-3.3x faster
# than the CuPy cumsum + fancy-index workaround at polygon-relevant scales
# (100-10K segments).  No CuPy equivalent exists.


def segmented_reduce_sum(
    values: DeviceArray,
    starts: DeviceArray,
    ends: DeviceArray,
    *,
    num_segments: int | None = None,
    synchronize: bool = False,
    stream=None,
) -> SegmentedReduceResult:
    """Sum values within offset-delimited segments.

    Parameters
    ----------
    values : 1-D device array of float64 or int32
    starts : 1-D device array of int32 segment start offsets
    ends : 1-D device array of int32 segment end offsets (exclusive)
    num_segments : optional, inferred from starts.size if omitted
    synchronize : if False, skip the trailing stream synchronize

    Returns
    -------
    SegmentedReduceResult with per-segment sums
    """
    cp_module, cccl_algorithms = _require_cccl_primitives()
    stream = _effective_stream(cp_module, stream)
    _validate_vector("values", values)
    _validate_vector("starts", starts)
    _validate_vector("ends", ends)
    n_segs = num_segments if num_segments is not None else int(starts.size)
    out = cp_module.empty(n_segs, dtype=values.dtype)
    if n_segs == 0:
        return SegmentedReduceResult(values=out, segment_count=0)
    h_init = np.asarray(0, dtype=values.dtype)
    # make_* fast path (ADR-0034)
    precompiled = _get_precompiled(f"segmented_reduce_sum_{_dtype_suffix(values.dtype)}")
    if precompiled is not None:
        n = int(values.size)
        _execute_precompiled(
            precompiled,
            cp_module,
            num_items=n,
            args=(values, out, starts, ends, _sum_op, n_segs, h_init),
            stream=stream,
            synchronize=synchronize,
            references=(values, out, starts, ends),
        )
        return SegmentedReduceResult(values=out, segment_count=n_segs)
    # Fallback: one-shot API
    _execute_one_shot(
        cccl_algorithms.segmented_reduce,
        cp_module,
        args=(values, out, starts, ends, _sum_op, h_init, n_segs),
        stream=stream,
        synchronize=synchronize,
    )
    return SegmentedReduceResult(values=out, segment_count=n_segs)


def segmented_reduce_min(
    values: DeviceArray,
    starts: DeviceArray,
    ends: DeviceArray,
    *,
    num_segments: int | None = None,
    synchronize: bool = False,
    stream=None,
) -> SegmentedReduceResult:
    """Min-reduce values within offset-delimited segments."""
    cp_module, cccl_algorithms = _require_cccl_primitives()
    stream = _effective_stream(cp_module, stream)
    _validate_vector("values", values)
    n_segs = num_segments if num_segments is not None else int(starts.size)
    out = cp_module.empty(n_segs, dtype=values.dtype)
    if n_segs == 0:
        return SegmentedReduceResult(values=out, segment_count=0)
    dtype = np.dtype(values.dtype)
    if dtype.kind == "f":
        h_init = np.asarray(np.inf, dtype=dtype)
    else:
        h_init = np.asarray(np.iinfo(dtype).max, dtype=dtype)
    # make_* fast path (ADR-0034)
    precompiled = _get_precompiled(f"segmented_reduce_min_{_dtype_suffix(values.dtype)}")
    if precompiled is not None:
        n = int(values.size)
        _execute_precompiled(
            precompiled,
            cp_module,
            num_items=n,
            args=(values, out, starts, ends, _min_op, n_segs, h_init),
            stream=stream,
            synchronize=synchronize,
            references=(values, out, starts, ends),
        )
        return SegmentedReduceResult(values=out, segment_count=n_segs)
    # Fallback: one-shot API
    _execute_one_shot(
        cccl_algorithms.segmented_reduce,
        cp_module,
        args=(values, out, starts, ends, _min_op, h_init, n_segs),
        stream=stream,
        synchronize=synchronize,
    )
    return SegmentedReduceResult(values=out, segment_count=n_segs)


def segmented_reduce_max(
    values: DeviceArray,
    starts: DeviceArray,
    ends: DeviceArray,
    *,
    num_segments: int | None = None,
    synchronize: bool = False,
    stream=None,
) -> SegmentedReduceResult:
    """Max-reduce values within offset-delimited segments."""
    cp_module, cccl_algorithms = _require_cccl_primitives()
    stream = _effective_stream(cp_module, stream)
    _validate_vector("values", values)
    n_segs = num_segments if num_segments is not None else int(starts.size)
    out = cp_module.empty(n_segs, dtype=values.dtype)
    if n_segs == 0:
        return SegmentedReduceResult(values=out, segment_count=0)
    dtype = np.dtype(values.dtype)
    if dtype.kind == "f":
        h_init = np.asarray(-np.inf, dtype=dtype)
    else:
        h_init = np.asarray(np.iinfo(dtype).min, dtype=dtype)
    # make_* fast path (ADR-0034)
    precompiled = _get_precompiled(f"segmented_reduce_max_{_dtype_suffix(values.dtype)}")
    if precompiled is not None:
        n = int(values.size)
        _execute_precompiled(
            precompiled,
            cp_module,
            num_items=n,
            args=(values, out, starts, ends, _max_op, n_segs, h_init),
            stream=stream,
            synchronize=synchronize,
            references=(values, out, starts, ends),
        )
        return SegmentedReduceResult(values=out, segment_count=n_segs)
    # Fallback: one-shot API
    _execute_one_shot(
        cccl_algorithms.segmented_reduce,
        cp_module,
        args=(values, out, starts, ends, _max_op, h_init, n_segs),
        stream=stream,
        synchronize=synchronize,
    )
    return SegmentedReduceResult(values=out, segment_count=n_segs)


# ---------------------------------------------------------------------------
# Binary search (lower_bound / upper_bound)
# ---------------------------------------------------------------------------


def lower_bound(
    sorted_data: DeviceArray,
    query_values: DeviceArray,
    *,
    synchronize: bool = False,
    stream=None,
) -> DeviceArray:
    """Find the first insertion point for each query value in a sorted array.

    Returns a 1-D uint32 device array of indices (CCCL requires unsigned).
    """
    cp_module, cccl_algorithms = _require_cccl_primitives()
    stream = _effective_stream(cp_module, stream)
    _validate_vector("sorted_data", sorted_data)
    _validate_vector("query_values", query_values)
    out = cp_module.empty(int(query_values.size), dtype=np.uintp)
    if int(query_values.size) == 0:
        return out
    n_sorted = int(sorted_data.size)
    n_query = int(query_values.size)
    # make_* fast path (ADR-0034)
    precompiled = _get_precompiled(f"lower_bound_{_dtype_suffix(sorted_data.dtype)}")
    if precompiled is not None:
        _execute_precompiled(
            precompiled,
            cp_module,
            num_items=n_sorted + n_query,
            args=(sorted_data, query_values, out, n_sorted, n_query),
            stream=stream,
            synchronize=synchronize,
            references=(sorted_data, query_values, out),
        )
        return out
    # Fallback: one-shot API
    _execute_one_shot(
        cccl_algorithms.lower_bound,
        cp_module,
        args=(sorted_data, query_values, out, n_sorted, n_query),
        stream=stream,
        synchronize=synchronize,
    )
    return out


def lower_bound_counting(
    sorted_data: DeviceArray,
    start: int,
    count: int,
    *,
    dtype=np.int32,
    synchronize: bool = False,
    stream=None,
) -> DeviceArray:
    """Find insertion points for a lazy ``[start, start + count)`` sequence.

    This avoids materializing a query array purely to drive binary search.
    The current precompiled lower_bound fast path is array-specialized, so the
    counting-iterator path uses the generic CCCL algorithm entry point.
    """
    cp_module, cccl_algorithms = _require_cccl_primitives()
    stream = _effective_stream(cp_module, stream)
    _validate_vector("sorted_data", sorted_data)
    out = cp_module.empty(int(count), dtype=np.uintp)
    if int(count) == 0:
        return out
    query_values = counting_iterator(start=start, dtype=dtype)
    n_sorted = int(sorted_data.size)
    n_query = int(count)
    _execute_one_shot(
        cccl_algorithms.lower_bound,
        cp_module,
        args=(sorted_data, query_values, out, n_sorted, n_query),
        stream=stream,
        synchronize=synchronize,
    )
    return out


def upper_bound(
    sorted_data: DeviceArray,
    query_values: DeviceArray,
    *,
    synchronize: bool = False,
    stream=None,
) -> DeviceArray:
    """Find the last insertion point for each query value in a sorted array.

    Returns a 1-D uint32 device array of indices (CCCL requires unsigned).
    """
    cp_module, cccl_algorithms = _require_cccl_primitives()
    stream = _effective_stream(cp_module, stream)
    _validate_vector("sorted_data", sorted_data)
    _validate_vector("query_values", query_values)
    out = cp_module.empty(int(query_values.size), dtype=np.uintp)
    if int(query_values.size) == 0:
        return out
    n_sorted = int(sorted_data.size)
    n_query = int(query_values.size)
    # make_* fast path (ADR-0034)
    precompiled = _get_precompiled(f"upper_bound_{_dtype_suffix(sorted_data.dtype)}")
    if precompiled is not None:
        _execute_precompiled(
            precompiled,
            cp_module,
            num_items=n_sorted + n_query,
            args=(sorted_data, query_values, out, n_sorted, n_query),
            stream=stream,
            synchronize=synchronize,
            references=(sorted_data, query_values, out),
        )
        return out
    # Fallback: one-shot API
    _execute_one_shot(
        cccl_algorithms.upper_bound,
        cp_module,
        args=(sorted_data, query_values, out, n_sorted, n_query),
        stream=stream,
        synchronize=synchronize,
    )
    return out


def upper_bound_counting(
    sorted_data: DeviceArray,
    start: int,
    count: int,
    *,
    dtype=np.int32,
    synchronize: bool = False,
    stream=None,
) -> DeviceArray:
    """Find upper-bound insertion points for a lazy ``[start, start + count)`` sequence."""
    cp_module, cccl_algorithms = _require_cccl_primitives()
    stream = _effective_stream(cp_module, stream)
    _validate_vector("sorted_data", sorted_data)
    out = cp_module.empty(int(count), dtype=np.uintp)
    if int(count) == 0:
        return out
    query_values = counting_iterator(start=start, dtype=dtype)
    n_sorted = int(sorted_data.size)
    n_query = int(count)
    _execute_one_shot(
        cccl_algorithms.upper_bound,
        cp_module,
        args=(sorted_data, query_values, out, n_sorted, n_query),
        stream=stream,
        synchronize=synchronize,
    )
    return out


# ---------------------------------------------------------------------------
# Key-value sort
# ---------------------------------------------------------------------------


def select_pair_sort_strategy(
    key_dtype: np.dtype[np.generic] | str,
    value_dtype: np.dtype[np.generic] | str | None = None,
    *,
    strategy: PairSortStrategy | str = PairSortStrategy.AUTO,
) -> PairSortStrategy:
    requested = strategy if isinstance(strategy, PairSortStrategy) else PairSortStrategy(strategy)
    if requested is not PairSortStrategy.AUTO:
        return requested
    dtype = np.dtype(key_dtype)
    # Check if the CCCL radix_sort spec for this key dtype is already warm.
    # If cold, use CuPy argsort to avoid 5-11s JIT overhead.
    key_suffix = _dtype_suffix(dtype)
    val_suffix = _dtype_suffix(np.dtype(value_dtype)) if value_dtype is not None else "i32"
    if dtype.kind in {"b", "i", "u", "f"}:
        spec = f"radix_sort_{key_suffix}_{val_suffix}"
        if _cccl_is_warm(spec):
            return PairSortStrategy.RADIX
        return PairSortStrategy.CUPY
    return PairSortStrategy.MERGE


def sort_pairs(
    keys: DeviceArray,
    values: DeviceArray | None = None,
    *,
    descending: bool = False,
    strategy: PairSortStrategy | str = PairSortStrategy.AUTO,
    synchronize: bool = False,
    stream=None,
) -> PairSortResult:
    cp_module, cccl_algorithms = _require_cccl_primitives()
    stream = _effective_stream(cp_module, stream)
    _validate_vector("keys", keys)
    if values is not None:
        _validate_vector("values", values)
        if int(values.size) != int(keys.size):
            raise ValueError("values must match keys size")

    val_dtype = values.dtype if values is not None else np.dtype(np.int32)
    resolved = select_pair_sort_strategy(keys.dtype, val_dtype, strategy=strategy)
    out_keys = cp_module.empty_like(keys)
    out_values = None if values is None else cp_module.empty_like(values)
    item_count = int(keys.size)
    if item_count == 0:
        return PairSortResult(keys=out_keys, values=out_values, strategy=resolved)

    if resolved is PairSortStrategy.CUPY:
        # CuPy argsort fallback — avoids CCCL JIT when specs are cold.
        with _stream_context(cp_module, stream):
            if descending:
                idx = cp_module.argsort(-keys)
            else:
                idx = cp_module.argsort(keys)
            out_keys = keys[idx]
            out_values = values[idx] if values is not None else None
        if synchronize:
            _stream_synchronize(cp_module, stream)
        return PairSortResult(keys=out_keys, values=out_values, strategy=resolved)
    elif resolved is PairSortStrategy.RADIX:
        order = (
            cccl_algorithms.SortOrder.DESCENDING
            if descending
            else cccl_algorithms.SortOrder.ASCENDING
        )
        precompiled = None
        if not descending and values is not None:
            precompiled = _get_precompiled(
                f"radix_sort_{_dtype_suffix(keys.dtype)}_{_dtype_suffix(values.dtype)}",
            )
        if precompiled is not None:
            _execute_precompiled(
                precompiled,
                cp_module,
                num_items=item_count,
                args=(
                    keys,
                    out_keys,
                    values,
                    out_values,
                    item_count,
                ),
                stream=stream,
                synchronize=synchronize,
                references=(keys, out_keys, values, out_values),
            )
            return PairSortResult(keys=out_keys, values=out_values, strategy=resolved)
        else:
            _execute_one_shot(
                cccl_algorithms.radix_sort,
                cp_module,
                args=(keys, out_keys, values, out_values, order, item_count),
                stream=stream,
                synchronize=synchronize,
            )
            return PairSortResult(keys=out_keys, values=out_values, strategy=resolved)
    else:
        comparison = _greater_than if descending else _less_than
        precompiled = None
        if not descending and values is not None:
            precompiled = _get_precompiled(
                f"merge_sort_{_dtype_suffix(keys.dtype)}_{_dtype_suffix(values.dtype)}",
            )
        if precompiled is not None:
            _execute_precompiled(
                precompiled,
                cp_module,
                num_items=item_count,
                args=(
                    keys,
                    values,
                    out_keys,
                    out_values,
                    comparison,
                    item_count,
                ),
                stream=stream,
                synchronize=synchronize,
                references=(keys, values, out_keys, out_values),
            )
            return PairSortResult(keys=out_keys, values=out_values, strategy=resolved)
        else:
            _execute_one_shot(
                cccl_algorithms.merge_sort,
                cp_module,
                args=(keys, values, out_keys, out_values, comparison, item_count),
                stream=stream,
                synchronize=synchronize,
            )
            return PairSortResult(keys=out_keys, values=out_values, strategy=resolved)


# ---------------------------------------------------------------------------
# Unique-by-key
# ---------------------------------------------------------------------------


def unique_sorted_pairs(
    keys: DeviceArray,
    values: DeviceArray,
    *,
    synchronize: bool = False,
    stream=None,
) -> UniqueByKeyResult:
    cp_module, cccl_algorithms = _require_cccl_primitives()
    stream = _effective_stream(cp_module, stream)
    _validate_vector("keys", keys)
    _validate_vector("values", values)
    if int(values.size) != int(keys.size):
        raise ValueError("values must match keys size")

    item_count = int(keys.size)
    out_keys = cp_module.empty_like(keys)
    out_values = cp_module.empty_like(values)
    if item_count == 0:
        return UniqueByKeyResult(keys=out_keys, values=out_values, count=0)

    out_count = cp_module.empty((1,), dtype=cp_module.int32)
    # make_* fast path (ADR-0034)
    key_suffix = _dtype_suffix(keys.dtype)
    val_suffix = _dtype_suffix(values.dtype)
    precompiled = _get_precompiled(f"unique_by_key_{key_suffix}_{val_suffix}")
    if precompiled is not None:
        _execute_precompiled(
            precompiled,
            cp_module,
            num_items=item_count,
            args=(
                keys,
                values,
                out_keys,
                out_values,
                out_count,
                _equal_to,
                item_count,
            ),
            stream=stream,
            synchronize=True,
            references=(keys, values, out_keys, out_values, out_count),
        )
        # Returning a trimmed output still requires a host-materialized count.
        selected_count = _device_int_scalar(
            out_count,
            reason="cccl unique-by-key selected-count scalar fence",
        )
        return UniqueByKeyResult(
            keys=out_keys[:selected_count],
            values=out_values[:selected_count],
            count=selected_count,
        )
    # Fallback: one-shot API
    _execute_one_shot(
        cccl_algorithms.unique_by_key,
        cp_module,
        args=(keys, values, out_keys, out_values, out_count, _equal_to, item_count),
        stream=stream,
        synchronize=True,
    )
    # Returning a trimmed output still requires a host-materialized count.
    selected_count = _device_int_scalar(
        out_count,
        reason="cccl unique-by-key selected-count scalar fence",
    )
    return UniqueByKeyResult(
        keys=out_keys[:selected_count],
        values=out_values[:selected_count],
        count=selected_count,
    )


# ---------------------------------------------------------------------------
# Segmented sort
# ---------------------------------------------------------------------------
# ADR-0033 Tier 3a: High-value primitive for polygon work — half-edge angle
# sort (overlay), ring vertex sort (clip), split event sort (make_valid).


def _segmented_sort_cccl(
    keys: DeviceArray,
    values: DeviceArray | None,
    starts: DeviceArray,
    ends: DeviceArray,
    *,
    descending: bool = False,
    num_segments: int | None = None,
    synchronize: bool = False,
    stream=None,
) -> SegmentedSortResult:
    cp_module, cccl_algorithms = _require_cccl_primitives()
    stream = _effective_stream(cp_module, stream)
    n_segs = num_segments if num_segments is not None else int(starts.size)
    out_keys = cp_module.empty_like(keys)
    out_values = None if values is None else cp_module.empty_like(values)
    item_count = int(keys.size)
    if item_count == 0:
        return SegmentedSortResult(keys=out_keys, values=out_values, segment_count=n_segs)

    order = (
        cccl_algorithms.SortOrder.DESCENDING if descending else cccl_algorithms.SortOrder.ASCENDING
    )
    # make_* fast path (ADR-0034)
    suffix = f"{'desc' if descending else 'asc'}_{_dtype_suffix(keys.dtype)}"
    precompiled = _get_precompiled(f"segmented_sort_{suffix}")
    if precompiled is not None:
        _execute_precompiled(
            precompiled,
            cp_module,
            num_items=item_count,
            args=(
                keys,
                out_keys,
                values,
                out_values,
                item_count,
                n_segs,
                starts,
                ends,
                order,
            ),
            stream=stream,
            synchronize=synchronize,
            references=(keys, out_keys, values, out_values, starts, ends),
        )
        return SegmentedSortResult(keys=out_keys, values=out_values, segment_count=n_segs)
    # Fallback: one-shot API
    _execute_one_shot(
        cccl_algorithms.segmented_sort,
        cp_module,
        args=(keys, out_keys, values, out_values, item_count, n_segs, starts, ends, order),
        stream=stream,
        synchronize=synchronize,
    )
    return SegmentedSortResult(keys=out_keys, values=out_values, segment_count=n_segs)


def segmented_sort(
    keys: DeviceArray,
    values: DeviceArray | None = None,
    *,
    starts: DeviceArray,
    ends: DeviceArray,
    descending: bool = False,
    num_segments: int | None = None,
    strategy: SegmentedSortStrategy | str = SegmentedSortStrategy.AUTO,
    synchronize: bool = False,
    stream=None,
) -> SegmentedSortResult:
    """Sort key-value pairs within offset-delimited segments.

    Parameters
    ----------
    keys : 1-D device array of sortable type
    values : optional 1-D device array to permute alongside keys
    starts : 1-D int32 device array of segment start offsets
    ends : 1-D int32 device array of segment end offsets (exclusive)
    descending : sort in descending order within each segment
    num_segments : optional, inferred from starts.size if omitted
    strategy : dispatch strategy (AUTO always routes to CCCL)
    synchronize : if False, skip the trailing null-stream synchronize

    Returns
    -------
    SegmentedSortResult with sorted keys, permuted values, and segment count
    """
    _require_cccl_primitives()
    _validate_vector("keys", keys)
    _validate_vector("starts", starts)
    _validate_vector("ends", ends)
    if values is not None:
        _validate_vector("values", values)
        if int(values.size) != int(keys.size):
            raise ValueError("values must match keys size")
    return _segmented_sort_cccl(
        keys,
        values,
        starts,
        ends,
        descending=descending,
        num_segments=num_segments,
        synchronize=synchronize,
        stream=stream,
    )


# ---------------------------------------------------------------------------
# Three-way partition
# ---------------------------------------------------------------------------
# ADR-0033 Tier 3a: Family-tag partitioning — split mixed-geometry columns
# into point/line/polygon groups in one pass instead of three flatnonzero calls.


def _three_way_partition_cccl(
    values: DeviceArray,
    first_pred,
    second_pred,
) -> ThreeWayPartitionResult:
    cp_module, cccl_algorithms = _require_cccl_primitives()
    item_count = int(values.size)
    if item_count == 0:
        out = cp_module.empty_like(values)
        return ThreeWayPartitionResult(values=out, first_count=0, second_count=0)

    d_first = cp_module.empty(item_count, dtype=values.dtype)
    d_second = cp_module.empty(item_count, dtype=values.dtype)
    d_unselected = cp_module.empty(item_count, dtype=values.dtype)
    d_counts = cp_module.empty(2, dtype=cp_module.int32)
    _execute_one_shot(
        cccl_algorithms.three_way_partition,
        cp_module,
        args=(
            values,
            d_first,
            d_second,
            d_unselected,
            d_counts,
            first_pred,
            second_pred,
            item_count,
        ),
        synchronize=False,
    )
    h_counts = get_cuda_runtime().copy_device_to_host(
        d_counts,
        reason="cccl three-way partition count export",
    )
    first_count = int(h_counts[0])
    second_count = int(h_counts[1])
    # Concatenate the three partitions into a single output array
    out = cp_module.empty(item_count, dtype=values.dtype)
    out[:first_count] = d_first[:first_count]
    out[first_count : first_count + second_count] = d_second[:second_count]
    out[first_count + second_count :] = d_unselected[: item_count - first_count - second_count]
    return ThreeWayPartitionResult(
        values=out,
        first_count=first_count,
        second_count=second_count,
    )


def three_way_partition(
    values: DeviceArray,
    first_pred,
    second_pred,
    *,
    strategy: PartitionStrategy | str = PartitionStrategy.AUTO,
) -> ThreeWayPartitionResult:
    """Partition values into three groups based on two predicates.

    Elements satisfying first_pred come first, then those satisfying
    second_pred (but not first_pred), then the remainder.

    Parameters
    ----------
    values : 1-D device array
    first_pred : callable(element) -> bool for first partition
    second_pred : callable(element) -> bool for second partition
    strategy : dispatch strategy (AUTO always routes to CCCL)

    Returns
    -------
    ThreeWayPartitionResult with partitioned values and partition sizes
    """
    _require_cccl_primitives()
    _validate_vector("values", values)
    return _three_way_partition_cccl(values, first_pred, second_pred)


# ---------------------------------------------------------------------------
# CCCL Iterators (Tier 3c) — zero-allocation lazy evaluation
# ---------------------------------------------------------------------------


def counting_iterator(start: int = 0, dtype=np.int32):
    """Create a CCCL CountingIterator starting from the given value.

    Replaces ``cp.arange`` allocations with a lazy integer sequence
    that doesn't materialize a device array.

    Parameters
    ----------
    start : starting value for the iterator
    dtype : numpy dtype for the iterator values (default: int32)

    Returns
    -------
    A CCCL CountingIterator object usable in CCCL algorithm calls.
    """
    _require_cccl_primitives()
    from cuda.compute.iterators import CountingIterator

    return CountingIterator(np.dtype(dtype).type(start))


def transform_iterator(input_iter, transform_op):
    """Create a CCCL TransformIterator that lazily applies a transform.

    Fuses coordinate transforms with algorithms, avoiding intermediate
    buffer allocations.

    Parameters
    ----------
    input_iter : device array or another iterator
    transform_op : callable applied element-wise during iteration

    Returns
    -------
    A CCCL TransformIterator object usable in CCCL algorithm calls.
    """
    _require_cccl_primitives()
    from cuda.compute.iterators import TransformIterator

    return TransformIterator(input_iter, transform_op)
