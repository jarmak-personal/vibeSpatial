"""Demand-driven background pre-compilation of CCCL make_* callables.

ADR-0034: Each consumer module declares the CCCL primitives it needs via
request_warmup() at module scope.  The singleton CCCLPrecompiler accumulates
specs incrementally and compiles them on background threads.  The make_*
call triggers CCCL's JIT compilation (which releases the GIL via Cython
``with nogil:``), so threads achieve true CPU parallelism.

Toggle with VIBESPATIAL_PRECOMPILE env var (default: enabled).
"""
from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

PRECOMPILE_ENV_VAR = "VIBESPATIAL_PRECOMPILE"


def precompile_enabled() -> bool:
    """Return True unless the user explicitly disables precompilation."""
    value = os.environ.get(PRECOMPILE_ENV_VAR, "")
    if not value:
        return True
    return value.lower() not in {"0", "false", "off", "no"}


# ---------------------------------------------------------------------------
# Algorithm families
# ---------------------------------------------------------------------------

class AlgorithmFamily(StrEnum):
    EXCLUSIVE_SCAN = "exclusive_scan"
    SELECT = "select"
    REDUCE_INTO = "reduce_into"
    SEGMENTED_REDUCE = "segmented_reduce"
    LOWER_BOUND = "lower_bound"
    UPPER_BOUND = "upper_bound"
    RADIX_SORT = "radix_sort"
    MERGE_SORT = "merge_sort"
    UNIQUE_BY_KEY = "unique_by_key"
    SEGMENTED_SORT = "segmented_sort"


# ---------------------------------------------------------------------------
# Spec and result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CCCLWarmupSpec:
    """Specification for a CCCL make_* pre-compilation target."""
    name: str
    family: AlgorithmFamily
    key_dtype: np.dtype[Any]
    value_dtype: np.dtype[Any] | None  # for sort / unique_by_key
    op_name: str  # key into _OPS table


@dataclass(slots=True)
class PrecompiledPrimitive:
    """A compiled make_* callable with pre-allocated temp storage."""
    name: str
    make_callable: Any
    temp_storage: Any  # CuPy device array
    temp_storage_bytes: int
    high_water_n: int
    warmup_ms: float


@dataclass(frozen=True, slots=True)
class WarmupDiagnostic:
    name: str
    elapsed_ms: float
    success: bool
    error: str = ""


# ---------------------------------------------------------------------------
# Spec registry (all 18 specs from ADR-0034)
# ---------------------------------------------------------------------------

def _build_spec_registry() -> dict[str, CCCLWarmupSpec]:
    S = CCCLWarmupSpec
    F = AlgorithmFamily
    i32 = np.dtype(np.int32)
    i64 = np.dtype(np.int64)
    u64 = np.dtype(np.uint64)
    f64 = np.dtype(np.float64)
    return {
        # Exclusive scan
        "exclusive_scan_i32": S("exclusive_scan_i32", F.EXCLUSIVE_SCAN, i32, None, "sum"),
        "exclusive_scan_i64": S("exclusive_scan_i64", F.EXCLUSIVE_SCAN, i64, None, "sum"),
        # Select (compaction)
        "select_i32": S("select_i32", F.SELECT, i32, None, "select_predicate"),
        "select_i64": S("select_i64", F.SELECT, i64, None, "select_predicate"),
        # Reduce
        "reduce_sum_f64": S("reduce_sum_f64", F.REDUCE_INTO, f64, None, "sum"),
        "reduce_sum_i32": S("reduce_sum_i32", F.REDUCE_INTO, i32, None, "sum"),
        # Segmented reduce
        "segmented_reduce_sum_f64": S("segmented_reduce_sum_f64", F.SEGMENTED_REDUCE, f64, None, "sum"),
        "segmented_reduce_min_f64": S("segmented_reduce_min_f64", F.SEGMENTED_REDUCE, f64, None, "min"),
        "segmented_reduce_max_f64": S("segmented_reduce_max_f64", F.SEGMENTED_REDUCE, f64, None, "max"),
        # Binary search
        "lower_bound_i32": S("lower_bound_i32", F.LOWER_BOUND, i32, None, "none"),
        "lower_bound_u64": S("lower_bound_u64", F.LOWER_BOUND, u64, None, "none"),
        "upper_bound_i32": S("upper_bound_i32", F.UPPER_BOUND, i32, None, "none"),
        "upper_bound_u64": S("upper_bound_u64", F.UPPER_BOUND, u64, None, "none"),
        # Sort
        "radix_sort_i32_i32": S("radix_sort_i32_i32", F.RADIX_SORT, i32, i32, "ascending"),
        "radix_sort_i64_i32": S("radix_sort_i64_i32", F.RADIX_SORT, np.dtype(np.int64), i32, "ascending"),
        "radix_sort_u64_i32": S("radix_sort_u64_i32", F.RADIX_SORT, u64, i32, "ascending"),
        "merge_sort_u64_i32": S("merge_sort_u64_i32", F.MERGE_SORT, u64, i32, "less_than"),
        # Unique by key
        "unique_by_key_i32_i32": S("unique_by_key_i32_i32", F.UNIQUE_BY_KEY, i32, i32, "equal_to"),
        "unique_by_key_u64_i32": S("unique_by_key_u64_i32", F.UNIQUE_BY_KEY, u64, i32, "equal_to"),
        # Segmented sort (Tier 3a: half-edge angle sort, ring vertex sort)
        "segmented_sort_asc_f64": S("segmented_sort_asc_f64", F.SEGMENTED_SORT, f64, i32, "less_than"),
        "segmented_sort_asc_i32": S("segmented_sort_asc_i32", F.SEGMENTED_SORT, i32, i32, "less_than"),
    }


SPEC_REGISTRY: dict[str, CCCLWarmupSpec] = _build_spec_registry()


# ---------------------------------------------------------------------------
# Operator table (lazy — imports from cccl_primitives only when compiling)
# ---------------------------------------------------------------------------

def _get_op(op_name: str, *, algorithms: Any = None, cp_module: Any = None, dtype: Any = None):
    """Return the operator callable for a given op_name."""
    from .cccl_primitives import (
        _equal_to,
        _less_than,
        _max_op,
        _min_op,
        _sum_op,
    )
    table = {
        "sum": _sum_op,
        "min": _min_op,
        "max": _max_op,
        "less_than": _less_than,
        "equal_to": _equal_to,
    }
    if op_name in table:
        return table[op_name]
    if op_name == "ascending" and algorithms is not None:
        return algorithms.SortOrder.ASCENDING
    return None


def _get_h_init(op_name: str, dtype: np.dtype[Any]) -> np.ndarray | None:
    """Return the host initial value for the given op and dtype."""
    if op_name == "sum":
        return np.asarray(0, dtype=dtype)
    if op_name == "min":
        if dtype.kind == "f":
            return np.asarray(np.inf, dtype=dtype)
        return np.asarray(np.iinfo(dtype).max, dtype=dtype)
    if op_name == "max":
        if dtype.kind == "f":
            return np.asarray(-np.inf, dtype=dtype)
        return np.asarray(np.iinfo(dtype).min, dtype=dtype)
    return None


# ---------------------------------------------------------------------------
# Compilation helpers (one per algorithm family)
# ---------------------------------------------------------------------------

_N = 128  # representative array size for warmup


def _compile_exclusive_scan(
    spec: CCCLWarmupSpec, cp_module: Any, algorithms: Any,
) -> PrecompiledPrimitive:
    make_fn = algorithms.make_exclusive_scan
    d_in = cp_module.empty(_N, dtype=spec.key_dtype)
    d_out = cp_module.empty(_N, dtype=spec.key_dtype)
    op = _get_op(spec.op_name)
    h_init = _get_h_init(spec.op_name, spec.key_dtype)
    t0 = perf_counter()
    callable_obj = make_fn(d_in, d_out, op, h_init)
    temp_bytes = callable_obj(None, d_in, d_out, op, _N, h_init)
    temp_bytes = max(int(temp_bytes) if temp_bytes else 1, 1)
    d_temp = cp_module.empty(temp_bytes, dtype=cp_module.uint8)
    elapsed = (perf_counter() - t0) * 1000.0
    return PrecompiledPrimitive(
        name=spec.name, make_callable=callable_obj,
        temp_storage=d_temp, temp_storage_bytes=temp_bytes,
        high_water_n=_N, warmup_ms=elapsed,
    )


def _compile_select(
    spec: CCCLWarmupSpec, cp_module: Any, algorithms: Any,
) -> PrecompiledPrimitive:
    make_fn = algorithms.make_select
    d_indices = cp_module.arange(_N, dtype=spec.key_dtype)
    d_out = cp_module.empty(_N, dtype=spec.key_dtype)
    d_count = cp_module.empty(1, dtype=spec.key_dtype)
    d_mask = cp_module.ones(_N, dtype=spec.key_dtype)

    def _warmup_predicate(index):  # pragma: no cover - exercised through CCCL JIT
        return d_mask[index] != 0

    t0 = perf_counter()
    callable_obj = make_fn(d_indices, d_out, d_count, _warmup_predicate)
    temp_bytes = callable_obj(None, d_indices, d_out, d_count, _warmup_predicate, _N)
    temp_bytes = max(int(temp_bytes) if temp_bytes else 1, 1)
    d_temp = cp_module.empty(temp_bytes, dtype=cp_module.uint8)
    elapsed = (perf_counter() - t0) * 1000.0
    return PrecompiledPrimitive(
        name=spec.name, make_callable=callable_obj,
        temp_storage=d_temp, temp_storage_bytes=temp_bytes,
        high_water_n=_N, warmup_ms=elapsed,
    )


def _compile_reduce_into(
    spec: CCCLWarmupSpec, cp_module: Any, algorithms: Any,
) -> PrecompiledPrimitive:
    make_fn = algorithms.make_reduce_into
    d_in = cp_module.empty(_N, dtype=spec.key_dtype)
    d_out = cp_module.empty(1, dtype=spec.key_dtype)
    op = _get_op(spec.op_name)
    h_init = _get_h_init(spec.op_name, spec.key_dtype)
    t0 = perf_counter()
    callable_obj = make_fn(d_in, d_out, op, h_init)
    temp_bytes = callable_obj(None, d_in, d_out, op, _N, h_init)
    temp_bytes = max(int(temp_bytes) if temp_bytes else 1, 1)
    d_temp = cp_module.empty(temp_bytes, dtype=cp_module.uint8)
    elapsed = (perf_counter() - t0) * 1000.0
    return PrecompiledPrimitive(
        name=spec.name, make_callable=callable_obj,
        temp_storage=d_temp, temp_storage_bytes=temp_bytes,
        high_water_n=_N, warmup_ms=elapsed,
    )


def _compile_segmented_reduce(
    spec: CCCLWarmupSpec, cp_module: Any, algorithms: Any,
) -> PrecompiledPrimitive:
    make_fn = algorithms.make_segmented_reduce
    n_segs = 4
    seg_size = _N // n_segs
    d_values = cp_module.empty(_N, dtype=spec.key_dtype)
    d_out = cp_module.empty(n_segs, dtype=spec.key_dtype)
    d_starts = cp_module.arange(0, _N, seg_size, dtype=cp_module.int32)
    d_ends = d_starts + seg_size
    op = _get_op(spec.op_name)
    h_init = _get_h_init(spec.op_name, spec.key_dtype)
    t0 = perf_counter()
    callable_obj = make_fn(d_values, d_out, d_starts, d_ends, op, h_init)
    temp_bytes = callable_obj(
        None, d_values, d_out, d_starts, d_ends, op, n_segs, h_init,
    )
    temp_bytes = max(int(temp_bytes) if temp_bytes else 1, 1)
    d_temp = cp_module.empty(temp_bytes, dtype=cp_module.uint8)
    elapsed = (perf_counter() - t0) * 1000.0
    return PrecompiledPrimitive(
        name=spec.name, make_callable=callable_obj,
        temp_storage=d_temp, temp_storage_bytes=temp_bytes,
        high_water_n=_N, warmup_ms=elapsed,
    )


def _compile_lower_bound(
    spec: CCCLWarmupSpec, cp_module: Any, algorithms: Any,
) -> PrecompiledPrimitive:
    make_fn = algorithms.make_lower_bound
    d_sorted = cp_module.arange(_N, dtype=spec.key_dtype)
    d_query = cp_module.empty(_N, dtype=spec.key_dtype)
    d_out = cp_module.empty(_N, dtype=np.uintp)
    t0 = perf_counter()
    callable_obj = make_fn(d_sorted, d_query, d_out)
    temp_bytes = callable_obj(None, d_sorted, d_query, d_out, _N, _N)
    temp_bytes = max(int(temp_bytes) if temp_bytes else 1, 1)
    d_temp = cp_module.empty(temp_bytes, dtype=cp_module.uint8)
    elapsed = (perf_counter() - t0) * 1000.0
    return PrecompiledPrimitive(
        name=spec.name, make_callable=callable_obj,
        temp_storage=d_temp, temp_storage_bytes=temp_bytes,
        high_water_n=_N, warmup_ms=elapsed,
    )


def _compile_upper_bound(
    spec: CCCLWarmupSpec, cp_module: Any, algorithms: Any,
) -> PrecompiledPrimitive:
    make_fn = algorithms.make_upper_bound
    d_sorted = cp_module.arange(_N, dtype=spec.key_dtype)
    d_query = cp_module.empty(_N, dtype=spec.key_dtype)
    d_out = cp_module.empty(_N, dtype=np.uintp)
    t0 = perf_counter()
    callable_obj = make_fn(d_sorted, d_query, d_out)
    temp_bytes = callable_obj(None, d_sorted, d_query, d_out, _N, _N)
    temp_bytes = max(int(temp_bytes) if temp_bytes else 1, 1)
    d_temp = cp_module.empty(temp_bytes, dtype=cp_module.uint8)
    elapsed = (perf_counter() - t0) * 1000.0
    return PrecompiledPrimitive(
        name=spec.name, make_callable=callable_obj,
        temp_storage=d_temp, temp_storage_bytes=temp_bytes,
        high_water_n=_N, warmup_ms=elapsed,
    )


def _compile_radix_sort(
    spec: CCCLWarmupSpec, cp_module: Any, algorithms: Any,
) -> PrecompiledPrimitive:
    make_fn = algorithms.make_radix_sort
    d_keys = cp_module.empty(_N, dtype=spec.key_dtype)
    d_out_keys = cp_module.empty(_N, dtype=spec.key_dtype)
    d_values = cp_module.empty(_N, dtype=spec.value_dtype)
    d_out_values = cp_module.empty(_N, dtype=spec.value_dtype)
    order = algorithms.SortOrder.ASCENDING
    t0 = perf_counter()
    callable_obj = make_fn(d_keys, d_out_keys, d_values, d_out_values, order)
    temp_bytes = callable_obj(
        None, d_keys, d_out_keys, d_values, d_out_values, _N,
    )
    temp_bytes = max(int(temp_bytes) if temp_bytes else 1, 1)
    d_temp = cp_module.empty(temp_bytes, dtype=cp_module.uint8)
    elapsed = (perf_counter() - t0) * 1000.0
    return PrecompiledPrimitive(
        name=spec.name, make_callable=callable_obj,
        temp_storage=d_temp, temp_storage_bytes=temp_bytes,
        high_water_n=_N, warmup_ms=elapsed,
    )


def _compile_merge_sort(
    spec: CCCLWarmupSpec, cp_module: Any, algorithms: Any,
) -> PrecompiledPrimitive:
    make_fn = algorithms.make_merge_sort
    d_keys = cp_module.empty(_N, dtype=spec.key_dtype)
    d_values = cp_module.empty(_N, dtype=spec.value_dtype)
    d_out_keys = cp_module.empty(_N, dtype=spec.key_dtype)
    d_out_values = cp_module.empty(_N, dtype=spec.value_dtype)
    op = _get_op(spec.op_name)
    t0 = perf_counter()
    callable_obj = make_fn(d_keys, d_values, d_out_keys, d_out_values, op)
    temp_bytes = callable_obj(
        None, d_keys, d_values, d_out_keys, d_out_values, op, _N,
    )
    temp_bytes = max(int(temp_bytes) if temp_bytes else 1, 1)
    d_temp = cp_module.empty(temp_bytes, dtype=cp_module.uint8)
    elapsed = (perf_counter() - t0) * 1000.0
    return PrecompiledPrimitive(
        name=spec.name, make_callable=callable_obj,
        temp_storage=d_temp, temp_storage_bytes=temp_bytes,
        high_water_n=_N, warmup_ms=elapsed,
    )


def _compile_unique_by_key(
    spec: CCCLWarmupSpec, cp_module: Any, algorithms: Any,
) -> PrecompiledPrimitive:
    make_fn = algorithms.make_unique_by_key
    d_keys = cp_module.arange(_N, dtype=spec.key_dtype)
    d_values = cp_module.empty(_N, dtype=spec.value_dtype)
    d_out_keys = cp_module.empty(_N, dtype=spec.key_dtype)
    d_out_values = cp_module.empty(_N, dtype=spec.value_dtype)
    d_out_count = cp_module.empty(1, dtype=cp_module.int32)
    op = _get_op(spec.op_name)
    t0 = perf_counter()
    callable_obj = make_fn(d_keys, d_values, d_out_keys, d_out_values, d_out_count, op)
    temp_bytes = callable_obj(
        None, d_keys, d_values, d_out_keys, d_out_values, d_out_count, op, _N,
    )
    temp_bytes = max(int(temp_bytes) if temp_bytes else 1, 1)
    d_temp = cp_module.empty(temp_bytes, dtype=cp_module.uint8)
    elapsed = (perf_counter() - t0) * 1000.0
    return PrecompiledPrimitive(
        name=spec.name, make_callable=callable_obj,
        temp_storage=d_temp, temp_storage_bytes=temp_bytes,
        high_water_n=_N, warmup_ms=elapsed,
    )


def _compile_segmented_sort(
    spec: CCCLWarmupSpec, cp_module: Any, algorithms: Any,
) -> PrecompiledPrimitive:
    make_fn = algorithms.make_segmented_sort
    n_segs = 4
    seg_size = _N // n_segs
    d_keys = cp_module.empty(_N, dtype=spec.key_dtype)
    d_out_keys = cp_module.empty(_N, dtype=spec.key_dtype)
    d_values = cp_module.empty(_N, dtype=spec.value_dtype) if spec.value_dtype is not None else None
    d_out_values = cp_module.empty(_N, dtype=spec.value_dtype) if spec.value_dtype is not None else None
    d_starts = cp_module.arange(0, _N, seg_size, dtype=cp_module.int32)
    d_ends = d_starts + seg_size
    order = algorithms.SortOrder.ASCENDING
    t0 = perf_counter()
    callable_obj = make_fn(
        d_keys, d_out_keys, d_values, d_out_values,
        d_starts, d_ends, order,
    )
    temp_bytes = callable_obj(
        None, d_keys, d_out_keys, d_values, d_out_values,
        _N, n_segs, d_starts, d_ends, order,
    )
    temp_bytes = max(int(temp_bytes) if temp_bytes else 1, 1)
    d_temp = cp_module.empty(temp_bytes, dtype=cp_module.uint8)
    elapsed = (perf_counter() - t0) * 1000.0
    return PrecompiledPrimitive(
        name=spec.name, make_callable=callable_obj,
        temp_storage=d_temp, temp_storage_bytes=temp_bytes,
        high_water_n=_N, warmup_ms=elapsed,
    )


_FAMILY_COMPILERS = {
    AlgorithmFamily.EXCLUSIVE_SCAN: _compile_exclusive_scan,
    AlgorithmFamily.SELECT: _compile_select,
    AlgorithmFamily.REDUCE_INTO: _compile_reduce_into,
    AlgorithmFamily.SEGMENTED_REDUCE: _compile_segmented_reduce,
    AlgorithmFamily.LOWER_BOUND: _compile_lower_bound,
    AlgorithmFamily.UPPER_BOUND: _compile_upper_bound,
    AlgorithmFamily.RADIX_SORT: _compile_radix_sort,
    AlgorithmFamily.MERGE_SORT: _compile_merge_sort,
    AlgorithmFamily.UNIQUE_BY_KEY: _compile_unique_by_key,
    AlgorithmFamily.SEGMENTED_SORT: _compile_segmented_sort,
}

# Map family enum to the make_* function name for hasattr checks.
_FAMILY_MAKE_FN = {
    AlgorithmFamily.EXCLUSIVE_SCAN: "make_exclusive_scan",
    AlgorithmFamily.SELECT: "make_select",
    AlgorithmFamily.REDUCE_INTO: "make_reduce_into",
    AlgorithmFamily.SEGMENTED_REDUCE: "make_segmented_reduce",
    AlgorithmFamily.LOWER_BOUND: "make_lower_bound",
    AlgorithmFamily.UPPER_BOUND: "make_upper_bound",
    AlgorithmFamily.RADIX_SORT: "make_radix_sort",
    AlgorithmFamily.MERGE_SORT: "make_merge_sort",
    AlgorithmFamily.UNIQUE_BY_KEY: "make_unique_by_key",
    AlgorithmFamily.SEGMENTED_SORT: "make_segmented_sort",
}


# ---------------------------------------------------------------------------
# Cached-build helpers — construct _Cached* from disk cache entries
# ---------------------------------------------------------------------------


def _build_cached_callable(
    entry: Any, spec: CCCLWarmupSpec, cp_module: Any, algorithms: Any,
) -> Any:
    """Construct a _Cached* callable from a CacheEntry + spec."""
    from cuda.compute._bindings import Op as CcclOp
    from cuda.compute._bindings import OpKind
    from cuda.compute._cccl_interop import (
        to_cccl_input_iter,
        to_cccl_output_iter,
        to_cccl_value,
    )
    from cuda.compute.op import make_op_adapter

    from .cccl_cubin_cache import (
        _CachedBinarySearch,
        _CachedMergeSort,
        _CachedRadixSort,
        _CachedScanReduce,
        _CachedSegmentedReduce,
        _CachedUniqueByKey,
        reconstruct_build,
    )

    build = reconstruct_build(entry)

    family = spec.family
    if family in (AlgorithmFamily.EXCLUSIVE_SCAN, AlgorithmFamily.REDUCE_INTO):
        d_in = cp_module.empty(_N, dtype=spec.key_dtype)
        d_out = cp_module.empty(
            1 if family == AlgorithmFamily.REDUCE_INTO else _N,
            dtype=spec.key_dtype,
        )
        op = _get_op(spec.op_name)
        h_init = _get_h_init(spec.op_name, spec.key_dtype)
        d_in_cccl = to_cccl_input_iter(d_in)
        d_out_cccl = to_cccl_output_iter(d_out)
        op_adapter = make_op_adapter(op)
        from cuda.compute._cccl_interop import get_value_type
        vt = get_value_type(h_init)
        op_cccl = op_adapter.compile((vt, vt), vt)
        init_cccl = to_cccl_value(h_init)

        c_func = (
            "cccl_device_exclusive_scan"
            if family == AlgorithmFamily.EXCLUSIVE_SCAN
            else "cccl_device_reduce"
        )
        return _CachedScanReduce(
            build, spec.name, c_func,
            d_in_cccl, d_out_cccl, op_cccl, init_cccl, op_adapter,
        )

    if family == AlgorithmFamily.SEGMENTED_REDUCE:
        n_segs = 4
        seg_size = _N // n_segs
        d_values = cp_module.empty(_N, dtype=spec.key_dtype)
        d_out = cp_module.empty(n_segs, dtype=spec.key_dtype)
        d_starts = cp_module.arange(0, _N, seg_size, dtype=cp_module.int32)
        d_ends = d_starts + seg_size
        op = _get_op(spec.op_name)
        h_init = _get_h_init(spec.op_name, spec.key_dtype)
        d_in_cccl = to_cccl_input_iter(d_values)
        d_out_cccl = to_cccl_output_iter(d_out)
        d_starts_cccl = to_cccl_input_iter(d_starts)
        d_ends_cccl = to_cccl_input_iter(d_ends)
        op_adapter = make_op_adapter(op)
        from cuda.compute._cccl_interop import get_value_type
        vt = get_value_type(h_init)
        op_cccl = op_adapter.compile((vt, vt), vt)
        init_cccl = to_cccl_value(h_init)
        return _CachedSegmentedReduce(
            build, spec.name,
            d_in_cccl, d_out_cccl, d_starts_cccl, d_ends_cccl,
            op_cccl, init_cccl, op_adapter,
        )

    if family in (AlgorithmFamily.LOWER_BOUND, AlgorithmFamily.UPPER_BOUND):
        d_sorted = cp_module.arange(_N, dtype=spec.key_dtype)
        d_query = cp_module.empty(_N, dtype=spec.key_dtype)
        d_out = cp_module.empty(_N, dtype=np.uintp)
        d_data_cccl = to_cccl_input_iter(d_sorted)
        d_values_cccl = to_cccl_input_iter(d_query)
        d_out_cccl = to_cccl_output_iter(d_out)
        # Binary search uses a compiled lambda comparator (not OpKind.LESS,
        # because well-known ops don't carry type info for JIT)
        import numba
        def _default_less(a, b):
            return a < b
        from cuda.compute._cccl_interop import get_value_type
        comp_adapter = make_op_adapter(_default_less)
        vt = get_value_type(d_sorted)
        op_cccl = comp_adapter.compile((vt, vt), numba.types.uint8)
        return _CachedBinarySearch(
            build, spec.name,
            d_data_cccl, d_values_cccl, d_out_cccl, op_cccl,
        )

    if family == AlgorithmFamily.RADIX_SORT:
        d_keys = cp_module.empty(_N, dtype=spec.key_dtype)
        d_out_keys = cp_module.empty(_N, dtype=spec.key_dtype)
        d_values = cp_module.empty(_N, dtype=spec.value_dtype)
        d_out_values = cp_module.empty(_N, dtype=spec.value_dtype)
        d_keys_in_cccl = to_cccl_input_iter(d_keys)
        d_keys_out_cccl = to_cccl_output_iter(d_out_keys)
        d_vals_in_cccl = to_cccl_input_iter(d_values)
        d_vals_out_cccl = to_cccl_output_iter(d_out_values)
        # Radix sort uses a dummy (empty) decomposer op
        from cuda.compute._bindings import Op as CcclOp
        decomposer_cccl = CcclOp(
            name="", operator_type=OpKind.STATELESS,
            ltoir=b"", state_alignment=1, state=None,
        )
        return _CachedRadixSort(
            build, spec.name,
            d_keys_in_cccl, d_keys_out_cccl,
            d_vals_in_cccl, d_vals_out_cccl,
            decomposer_cccl,
        )

    if family == AlgorithmFamily.MERGE_SORT:
        d_keys = cp_module.empty(_N, dtype=spec.key_dtype)
        d_values = cp_module.empty(_N, dtype=spec.value_dtype)
        d_out_keys = cp_module.empty(_N, dtype=spec.key_dtype)
        d_out_values = cp_module.empty(_N, dtype=spec.value_dtype)
        op = _get_op(spec.op_name)
        d_keys_in_cccl = to_cccl_input_iter(d_keys)
        d_vals_in_cccl = to_cccl_input_iter(d_values)
        d_keys_out_cccl = to_cccl_output_iter(d_out_keys)
        d_vals_out_cccl = to_cccl_output_iter(d_out_values)
        op_adapter = make_op_adapter(op)
        from cuda.compute._cccl_interop import get_value_type
        vt = get_value_type(d_keys)
        op_cccl = op_adapter.compile((vt, vt), vt)
        return _CachedMergeSort(
            build, spec.name,
            d_keys_in_cccl, d_vals_in_cccl,
            d_keys_out_cccl, d_vals_out_cccl,
            op_cccl, op_adapter,
        )

    if family == AlgorithmFamily.UNIQUE_BY_KEY:
        d_keys = cp_module.arange(_N, dtype=spec.key_dtype)
        d_values = cp_module.empty(_N, dtype=spec.value_dtype)
        d_out_keys = cp_module.empty(_N, dtype=spec.key_dtype)
        d_out_values = cp_module.empty(_N, dtype=spec.value_dtype)
        d_count = cp_module.empty(1, dtype=cp_module.int32)
        op = _get_op(spec.op_name)
        d_keys_in_cccl = to_cccl_input_iter(d_keys)
        d_vals_in_cccl = to_cccl_input_iter(d_values)
        d_keys_out_cccl = to_cccl_output_iter(d_out_keys)
        d_vals_out_cccl = to_cccl_output_iter(d_out_values)
        d_count_cccl = to_cccl_output_iter(d_count)
        op_adapter = make_op_adapter(op)
        from cuda.compute._cccl_interop import get_value_type
        vt = get_value_type(d_keys)
        op_cccl = op_adapter.compile((vt, vt), vt)
        return _CachedUniqueByKey(
            build, spec.name,
            d_keys_in_cccl, d_vals_in_cccl,
            d_keys_out_cccl, d_vals_out_cccl, d_count_cccl,
            op_cccl, op_adapter,
        )

    # Unsupported family (select, segmented_sort) — fall through to build
    return None


def _query_cached_temp(
    cached_callable: Any, spec: CCCLWarmupSpec, cp_module: Any,
) -> int:
    """Query temp storage size from a cached callable."""
    family = spec.family

    if family in (AlgorithmFamily.EXCLUSIVE_SCAN, AlgorithmFamily.REDUCE_INTO):
        d_in = cp_module.empty(_N, dtype=spec.key_dtype)
        d_out = cp_module.empty(
            1 if family == AlgorithmFamily.REDUCE_INTO else _N,
            dtype=spec.key_dtype,
        )
        op = _get_op(spec.op_name)
        h_init = _get_h_init(spec.op_name, spec.key_dtype)
        return cached_callable(None, d_in, d_out, op, _N, h_init)

    if family == AlgorithmFamily.SEGMENTED_REDUCE:
        n_segs = 4
        seg_size = _N // n_segs
        d_values = cp_module.empty(_N, dtype=spec.key_dtype)
        d_out = cp_module.empty(n_segs, dtype=spec.key_dtype)
        d_starts = cp_module.arange(0, _N, seg_size, dtype=cp_module.int32)
        d_ends = d_starts + seg_size
        op = _get_op(spec.op_name)
        h_init = _get_h_init(spec.op_name, spec.key_dtype)
        return cached_callable(
            None, d_values, d_out, d_starts, d_ends, op, n_segs, h_init,
        )

    if family in (AlgorithmFamily.LOWER_BOUND, AlgorithmFamily.UPPER_BOUND):
        d_sorted = cp_module.arange(_N, dtype=spec.key_dtype)
        d_query = cp_module.empty(_N, dtype=spec.key_dtype)
        d_out = cp_module.empty(_N, dtype=np.uintp)
        return cached_callable(None, d_sorted, d_query, d_out, _N, _N)

    if family == AlgorithmFamily.RADIX_SORT:
        d_keys = cp_module.empty(_N, dtype=spec.key_dtype)
        d_out_keys = cp_module.empty(_N, dtype=spec.key_dtype)
        d_values = cp_module.empty(_N, dtype=spec.value_dtype)
        d_out_values = cp_module.empty(_N, dtype=spec.value_dtype)
        return cached_callable(
            None, d_keys, d_out_keys, d_values, d_out_values, _N,
        )

    if family == AlgorithmFamily.MERGE_SORT:
        d_keys = cp_module.empty(_N, dtype=spec.key_dtype)
        d_values = cp_module.empty(_N, dtype=spec.value_dtype)
        d_out_keys = cp_module.empty(_N, dtype=spec.key_dtype)
        d_out_values = cp_module.empty(_N, dtype=spec.value_dtype)
        op = _get_op(spec.op_name)
        return cached_callable(
            None, d_keys, d_values, d_out_keys, d_out_values, op, _N,
        )

    if family == AlgorithmFamily.UNIQUE_BY_KEY:
        d_keys = cp_module.arange(_N, dtype=spec.key_dtype)
        d_values = cp_module.empty(_N, dtype=spec.value_dtype)
        d_out_keys = cp_module.empty(_N, dtype=spec.key_dtype)
        d_out_values = cp_module.empty(_N, dtype=spec.value_dtype)
        d_count = cp_module.empty(1, dtype=cp_module.int32)
        op = _get_op(spec.op_name)
        return cached_callable(
            None, d_keys, d_values, d_out_keys, d_out_values, d_count, op, _N,
        )

    return 1  # fallback


# ---------------------------------------------------------------------------
# Precompiler singleton
# ---------------------------------------------------------------------------

class CCCLPrecompiler:
    """Demand-driven background pre-compilation of CCCL make_* callables.

    Unlike a blast-all-at-import precompiler, this accumulates specs
    incrementally via request() calls from consumer modules.
    Each consumer declares only the specs it needs.
    """

    _instance: CCCLPrecompiler | None = None

    def __init__(self, max_workers: int = 8) -> None:
        self._cache: dict[str, PrecompiledPrimitive] = {}
        self._futures: dict[str, Future[PrecompiledPrimitive | None]] = {}
        self._submitted: set[str] = set()
        self._deferred_disk: set[str] = set()
        self._lock = threading.Lock()
        self._deferred_lock = threading.Lock()
        self._max_workers = max_workers
        self._executor: ThreadPoolExecutor | None = None
        self._start_time: float | None = None
        self._diagnostics: list[WarmupDiagnostic] = []

    @classmethod
    def get(cls) -> CCCLPrecompiler:
        """Lazy singleton.  Created on first request_warmup() call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def _reset(cls) -> None:
        """Reset singleton for testing."""
        if cls._instance is not None:
            cls._instance.shutdown()
            cls._instance = None

    def _ensure_executor(self) -> ThreadPoolExecutor:
        """Create the thread pool lazily, only when there are actual cache misses."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix="cccl-warmup",
            )
        return self._executor

    def request(self, spec_names: list[str]) -> None:
        """Submit specs for background compilation.  Idempotent, never blocks.

        Specs that are already cached on disk are deferred — they will be
        loaded lazily on the first get_compiled() call instead of eagerly
        in a background thread.  If all specs are cached, no thread pool
        is created.
        """
        with self._lock:
            new_specs = [n for n in spec_names if n not in self._submitted]
            if not new_specs:
                return
            if self._start_time is None:
                self._start_time = perf_counter()

            # Batch probe: which of the new specs are already on disk?
            from .cccl_cubin_cache import _cached_spec_name_set
            cached_on_disk = _cached_spec_name_set()

            for name in new_specs:
                if name not in SPEC_REGISTRY:
                    logger.warning("Unknown CCCL warmup spec: %s", name)
                    continue
                self._submitted.add(name)
                if name in cached_on_disk:
                    self._deferred_disk.add(name)
                    logger.debug("CCCL warmup: %s deferred (disk cached)", name)
                else:
                    spec = SPEC_REGISTRY[name]
                    future = self._ensure_executor().submit(self._compile_one, spec)
                    self._futures[name] = future

    def _compile_one(self, spec: CCCLWarmupSpec) -> PrecompiledPrimitive | None:
        """JIT-compile one make_* callable with dummy arrays.  Runs on a worker thread.

        Checks the on-disk CUBIN cache first.  On miss, builds via CCCL
        make_* and caches the result for next time.
        """
        try:
            import cupy as cp_module
            from cuda.compute import algorithms
        except ImportError:
            diag = WarmupDiagnostic(spec.name, 0.0, False, "import failed")
            self._diagnostics.append(diag)
            return None

        make_fn_name = _FAMILY_MAKE_FN.get(spec.family)
        if make_fn_name and not hasattr(algorithms, make_fn_name):
            diag = WarmupDiagnostic(
                spec.name, 0.0, False,
                f"{make_fn_name} not available in this CCCL version",
            )
            self._diagnostics.append(diag)
            return None

        compiler = _FAMILY_COMPILERS.get(spec.family)
        if compiler is None:
            diag = WarmupDiagnostic(spec.name, 0.0, False, f"unknown family {spec.family}")
            self._diagnostics.append(diag)
            return None

        # --- CCCL CUBIN disk cache: try cache-hit path first ---
        cached_result = self._try_cached_compile(spec, cp_module, algorithms)
        if cached_result is not None:
            return cached_result

        # --- Standard CCCL build (cache miss or cache disabled) ---
        t0 = perf_counter()
        try:
            result = compiler(spec, cp_module, algorithms)
            self._cache[spec.name] = result
            elapsed = (perf_counter() - t0) * 1000.0
            result.warmup_ms = elapsed
            self._diagnostics.append(WarmupDiagnostic(spec.name, elapsed, True))
            logger.debug("CCCL warmup: %s compiled in %.1fms", spec.name, elapsed)

            # Cache for next time
            self._save_to_disk_cache(spec, result)

            return result
        except Exception as exc:
            elapsed = (perf_counter() - t0) * 1000.0
            self._diagnostics.append(WarmupDiagnostic(spec.name, elapsed, False, str(exc)))
            logger.debug("CCCL warmup: %s failed: %s", spec.name, exc)
            return None

    def _try_cached_compile(
        self,
        spec: CCCLWarmupSpec,
        cp_module: Any,
        algorithms: Any,
    ) -> PrecompiledPrimitive | None:
        """Try to load a cached CUBIN and construct a _Cached* callable."""
        try:
            from vibespatial import cccl_cubin_cache

            entry = cccl_cubin_cache.try_load_cached(spec.name, spec.family)
            if entry is None:
                return None

            t0 = perf_counter()
            cached_callable = _build_cached_callable(
                entry, spec, cp_module, algorithms,
            )
            if cached_callable is None:
                return None

            # Query temp storage
            temp_bytes = _query_cached_temp(cached_callable, spec, cp_module)
            temp_bytes = max(int(temp_bytes) if temp_bytes else 1, 1)
            d_temp = cp_module.empty(temp_bytes, dtype=cp_module.uint8)

            elapsed = (perf_counter() - t0) * 1000.0
            result = PrecompiledPrimitive(
                name=spec.name,
                make_callable=cached_callable,
                temp_storage=d_temp,
                temp_storage_bytes=temp_bytes,
                high_water_n=_N,
                warmup_ms=elapsed,
            )
            self._cache[spec.name] = result
            self._diagnostics.append(
                WarmupDiagnostic(spec.name, elapsed, True),
            )
            logger.debug(
                "CCCL warmup: %s loaded from disk cache in %.1fms",
                spec.name, elapsed,
            )
            return result
        except Exception:
            logger.debug(
                "CCCL cache: hit but load failed for %s, falling back",
                spec.name, exc_info=True,
            )
            return None

    def _save_to_disk_cache(
        self, spec: CCCLWarmupSpec, result: PrecompiledPrimitive,
    ) -> None:
        """Save a freshly built result to disk cache."""
        try:
            from vibespatial import cccl_cubin_cache
            cccl_cubin_cache.save_after_build(
                spec.name, spec.family, result.make_callable,
            )
        except Exception:
            logger.debug(
                "CCCL cache: save failed for %s", spec.name, exc_info=True,
            )

    def get_compiled(
        self, name: str, timeout: float = 5.0,
    ) -> PrecompiledPrimitive | None:
        """Get a pre-compiled primitive.  Blocks if compilation in progress.

        Returns None if the spec was never requested or compilation failed,
        in which case the caller should use the one-shot fallback.
        """
        if name in self._cache:
            return self._cache[name]
        if name in self._deferred_disk:
            return self._lazy_load_deferred(name)
        if name in self._futures:
            try:
                return self._futures[name].result(timeout=timeout)
            except Exception:
                return None
        return None

    def _lazy_load_deferred(self, name: str) -> PrecompiledPrimitive | None:
        """Synchronously load a deferred disk-cached spec into memory.

        Thread-safe: concurrent callers for the same spec serialize via
        _deferred_lock; the second caller gets the cached result.
        """
        if name in self._cache:
            return self._cache[name]

        with self._deferred_lock:
            if name in self._cache:
                return self._cache[name]

            spec = SPEC_REGISTRY.get(name)
            if spec is None:
                self._deferred_disk.discard(name)
                return None

            try:
                import cupy as cp_module
                from cuda.compute import algorithms
            except ImportError:
                self._deferred_disk.discard(name)
                return None

            t0 = perf_counter()
            result = self._try_cached_compile(spec, cp_module, algorithms)
            if result is not None:
                self._deferred_disk.discard(name)
                return result

            # Cache file disappeared or is corrupt — fall back to full compile
            logger.debug(
                "CCCL lazy load: %s disk cache miss, compiling synchronously",
                name,
            )
            self._deferred_disk.discard(name)
            compiler = _FAMILY_COMPILERS.get(spec.family)
            if compiler is None:
                return None
            try:
                result = compiler(spec, cp_module, algorithms)
                self._cache[name] = result
                elapsed = (perf_counter() - t0) * 1000.0
                result.warmup_ms = elapsed
                self._diagnostics.append(WarmupDiagnostic(name, elapsed, True))
                self._save_to_disk_cache(spec, result)
                return result
            except Exception as exc:
                elapsed = (perf_counter() - t0) * 1000.0
                self._diagnostics.append(
                    WarmupDiagnostic(name, elapsed, False, str(exc)),
                )
                return None

    def status(self) -> dict[str, Any]:
        """Diagnostic snapshot for observability."""
        return {
            "submitted": len(self._submitted),
            "compiled": len(self._cache),
            "deferred": len(self._deferred_disk),
            "pending": sum(1 for f in self._futures.values() if not f.done()),
            "failed": sum(
                1 for f in self._futures.values()
                if f.done() and f.exception() is not None
            ),
            "wall_ms": (perf_counter() - self._start_time) * 1000
            if self._start_time
            else 0,
            "per_primitive": [
                {"name": d.name, "ms": round(d.elapsed_ms, 1), "ok": d.success}
                for d in self._diagnostics
            ],
        }

    def ensure_warm(self, timeout: float = 30.0) -> list[str]:
        """Block until all submitted specs are compiled.

        Returns a list of spec names that timed out or failed.
        Used by Level 3 pipeline-aware warmup (ADR-0034) to front-load
        compilation before the first pipeline stage executes.
        """
        cold: list[str] = []
        # Load all deferred disk-cached specs first
        for name in list(self._deferred_disk):
            result = self._lazy_load_deferred(name)
            if result is None:
                cold.append(name)
        # Wait for background-compiled specs
        for name, future in list(self._futures.items()):
            if name in self._cache:
                continue
            try:
                future.result(timeout=timeout)
            except Exception:
                cold.append(name)
        return cold

    def shutdown(self) -> None:
        """Shut down the thread pool.  For testing cleanup."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def request_warmup(spec_names: list[str]) -> None:
    """Non-blocking request to pre-compile CCCL specs.

    Safe to call at module scope.  No-op if GPU is not available
    or if precompilation is disabled via VIBESPATIAL_PRECOMPILE=0.
    """
    if not precompile_enabled():
        return
    from vibespatial.runtime import has_gpu_runtime

    if not has_gpu_runtime():
        return
    CCCLPrecompiler.get().request(spec_names)


def precompile_status() -> dict[str, Any]:
    """Return combined warmup status, or empty dict if no warmup was started."""
    from .nvrtc_precompile import NVRTCPrecompiler

    result: dict[str, Any] = {}
    if CCCLPrecompiler._instance is not None:
        result["cccl"] = CCCLPrecompiler.get().status()
    if NVRTCPrecompiler._instance is not None:
        result["nvrtc"] = NVRTCPrecompiler.get().status()
    return result


def ensure_pipelines_warm(timeout: float = 60.0) -> dict[str, Any]:
    """Level 3 pipeline-aware warmup (ADR-0034): block until all submitted
    CCCL and NVRTC compilations have finished.

    Call this once before executing a pipeline to front-load all
    compilation into a single predictable wait instead of paying
    JIT costs scattered across pipeline stages.

    Returns a dict with 'cccl_cold' and 'nvrtc_cold' listing any
    specs/units that timed out.
    """
    from .nvrtc_precompile import NVRTCPrecompiler

    result: dict[str, Any] = {"cccl_cold": [], "nvrtc_cold": []}
    if CCCLPrecompiler._instance is not None:
        result["cccl_cold"] = CCCLPrecompiler.get().ensure_warm(timeout=timeout)
    if NVRTCPrecompiler._instance is not None:
        result["nvrtc_cold"] = NVRTCPrecompiler.get().ensure_warm(timeout=timeout)
    return result


# Modules whose import triggers request_nvrtc_warmup() at module scope.
_NVRTC_CONSUMER_MODULES: tuple[str, ...] = (
    "vibespatial.spatial.indexing",
    "vibespatial.io.geojson_gpu",
    "vibespatial.io.wkb",
    "vibespatial.kernels.core.geometry_analysis",
    "vibespatial.kernels.core.spatial_query_kernels",
    "vibespatial.kernels.core.wkb_decode",
    "vibespatial.kernels.predicates.point_in_polygon",
    "vibespatial.overlay.gpu",
    "vibespatial.constructive.make_valid_gpu",
    "vibespatial.constructive.make_valid_pipeline",
    "vibespatial.predicates.point_relations",
    "vibespatial.constructive.point",
    "vibespatial.spatial.point_distance",
    "vibespatial.constructive.polygon",
    "vibespatial.predicates.polygon",
    "vibespatial.constructive.linestring",
    "vibespatial.spatial.segment_distance",
    "vibespatial.spatial.segment_primitives",
    "vibespatial.constructive.measurement",
    "vibespatial.constructive.centroid",
    "vibespatial.constructive.clip_rect",
    "vibespatial.constructive.validity",
    "vibespatial.io.gpu_parse.structural",
    "vibespatial.io.gpu_parse.numeric",
    "vibespatial.io.gpu_parse.pattern",
)


def precompile_all(timeout: float = 120.0) -> dict[str, Any]:
    """Pre-compile every CCCL algorithm spec and NVRTC kernel in the library.

    This populates both the CCCL CUBIN disk cache and the NVRTC disk cache
    so that subsequent process starts pay near-zero JIT cost.  Intended for
    one-time use after installation or in a CI warm-up step::

        uv run python -c "from .cccl_precompile import precompile_all; print(precompile_all())"

    Blocks until all compilations finish (or *timeout* seconds elapse).

    Returns a dict with ``cccl`` and ``nvrtc`` status sub-dicts plus
    ``cccl_cold`` / ``nvrtc_cold`` lists of any specs that failed or
    timed out.
    """
    import importlib

    from vibespatial.runtime import has_gpu_runtime

    if not has_gpu_runtime():
        return {"error": "no GPU runtime available"}

    # ---- CCCL: request every spec in the registry ----
    request_warmup(list(SPEC_REGISTRY.keys()))

    # ---- NVRTC: import consumer modules to trigger their warmup requests ----
    for mod_name in _NVRTC_CONSUMER_MODULES:
        try:
            importlib.import_module(mod_name)
        except Exception:
            logger.debug("precompile_all: failed to import %s", mod_name)

    # ---- Block until done ----
    cold = ensure_pipelines_warm(timeout=timeout)

    # ---- Return combined status ----
    result = precompile_status()
    result["cccl_cold"] = cold.get("cccl_cold", [])
    result["nvrtc_cold"] = cold.get("nvrtc_cold", [])
    return result
