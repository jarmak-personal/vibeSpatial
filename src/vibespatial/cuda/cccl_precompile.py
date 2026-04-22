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
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from vibespatial.cccl_host import get_host_init
from vibespatial.runtime.cccl_warmup_specs import (
    AlgorithmFamily,
    CCCLWarmupSpec,
    WarmupDiagnostic,
    build_spec_registry,
)

logger = logging.getLogger(__name__)

PRECOMPILE_ENV_VAR = "VIBESPATIAL_PRECOMPILE"
_exact_polygon_intersection_warm_lock = threading.Lock()
_exact_polygon_intersection_warm_done = False
_many_vs_one_overlay_warm_lock = threading.Lock()
_many_vs_one_overlay_warm_done = False
_overlay_difference_warm_lock = threading.Lock()
_overlay_difference_warm_done = False
_device_linestring_buffer_warm_lock = threading.Lock()
_device_linestring_buffer_warm_done = False
_device_centroid_buffer_warm_lock = threading.Lock()
_device_centroid_buffer_warm_done = False


def precompile_enabled() -> bool:
    """Return True unless the user explicitly disables precompilation."""
    value = os.environ.get(PRECOMPILE_ENV_VAR, "")
    if not value:
        return True
    return value.lower() not in {"0", "false", "off", "no"}


# ---------------------------------------------------------------------------
# Spec and result types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PrecompiledPrimitive:
    """A compiled make_* callable with pre-allocated temp storage."""
    name: str
    make_callable: Any
    temp_storage: Any  # CuPy device array
    temp_storage_bytes: int
    high_water_n: int
    warmup_ms: float


SPEC_REGISTRY: dict[str, CCCLWarmupSpec] = build_spec_registry()


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

# ---------------------------------------------------------------------------
# CCCL warning suppression
# ---------------------------------------------------------------------------


_SUPPRESS_CCCL_STDERR = os.environ.get(
    "VIBESPATIAL_SUPPRESS_CCCL_WARNINGS", "1",
).lower() not in {"0", "false", "off", "no"}
"""Suppress CCCL deprecation warnings on stderr during JIT compilation.

CCCL emits #1444-D (double4 deprecated) warnings to C-level stderr that
cannot be caught by Python.  We redirect fd 2 to /dev/null during
compilation.  Set VIBESPATIAL_SUPPRESS_CCCL_WARNINGS=0 to disable.
"""

_stderr_lock = threading.Lock()
_stderr_refcount = 0
_stderr_saved_fd: int | None = None
_stderr_saved_file: Any = None  # Python file wrapping the real stderr fd


def get_real_stderr():
    """Return a file object that writes to the real stderr, even when
    fd 2 is temporarily redirected to /dev/null for CCCL warning
    suppression.  Falls back to sys.stderr when no redirect is active."""
    with _stderr_lock:
        f = _stderr_saved_file
        if f is not None and not f.closed:
            return f
    return sys.stderr


def _suppress_cccl_warnings(
    compiler: Any, spec: CCCLWarmupSpec, cp_module: Any, algorithms: Any,
) -> PrecompiledPrimitive:
    """Run a CCCL make_* compiler with C-level stderr suppressed.

    Uses reference counting so multiple background threads can compile in
    parallel (CCCL releases the GIL) while fd 2 stays redirected to
    /dev/null for the entire duration.  The first thread to enter saves
    fd 2 and redirects; the last thread to leave restores it.

    Other code that needs to write to the real stderr during suppression
    (e.g. progress output) should use ``get_real_stderr()``.
    """
    global _stderr_refcount, _stderr_saved_fd, _stderr_saved_file

    if not _SUPPRESS_CCCL_STDERR:
        return compiler(spec, cp_module, algorithms)

    # --- acquire: redirect fd 2 on first entrant ---
    with _stderr_lock:
        _stderr_refcount += 1
        if _stderr_refcount == 1:
            try:
                _stderr_saved_fd = os.dup(2)
                _stderr_saved_file = os.fdopen(
                    os.dup(_stderr_saved_fd), "w", closefd=True,
                )
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, 2)
                os.close(devnull)
            except OSError:
                _stderr_saved_fd = None
                _stderr_saved_file = None
                _stderr_refcount -= 1
                return compiler(spec, cp_module, algorithms)
    try:
        return compiler(spec, cp_module, algorithms)
    finally:
        # --- release: restore fd 2 when last thread exits ---
        with _stderr_lock:
            _stderr_refcount -= 1
            if _stderr_refcount == 0 and _stderr_saved_fd is not None:
                os.dup2(_stderr_saved_fd, 2)
                os.close(_stderr_saved_fd)
                _stderr_saved_fd = None
                if _stderr_saved_file is not None:
                    _stderr_saved_file.close()
                    _stderr_saved_file = None


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
    h_init = get_host_init(spec.op_name, spec.key_dtype)
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
    h_init = get_host_init(spec.op_name, spec.key_dtype)
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
    h_init = get_host_init(spec.op_name, spec.key_dtype)
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


class _BinarySearchAdapter:
    """Wrap a CCCL _BinarySearch callable to match the temp-storage calling
    convention used by the precompiled primitive call-sites.

    CCCL ``make_lower_bound`` / ``make_upper_bound`` return callables with
    signature ``(d_data, d_values, d_out, comp, num_items, num_values)``.
    The precompiled call-sites in ``cccl_primitives.py`` expect the
    ``(temp_storage, d_data, d_values, d_out, num_items, num_values)``
    pattern shared by scan/reduce/sort families.  This adapter bridges the
    gap so both the fresh-compile and disk-cache paths expose the same
    interface.

    Binary search has no temp storage requirement, so the ``temp_storage``
    argument is accepted but ignored (returning ``1`` when ``None`` for the
    query convention).
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    @property
    def build_result(self) -> Any:
        """Proxy ``build_result`` for disk-cache serialisation."""
        return self._inner.build_result

    def __call__(
        self,
        temp_storage: Any,
        d_data: Any,
        d_values: Any,
        d_out: Any,
        num_items: int,
        num_values: int,
        stream: Any = None,
    ) -> int:
        if temp_storage is None:
            return 1  # Binary search needs no temp storage.
        self._inner(d_data, d_values, d_out, None, num_items, num_values, stream)
        return 0


def _compile_lower_bound(
    spec: CCCLWarmupSpec, cp_module: Any, algorithms: Any,
) -> PrecompiledPrimitive:
    make_fn = algorithms.make_lower_bound
    d_sorted = cp_module.arange(_N, dtype=spec.key_dtype)
    d_query = cp_module.empty(_N, dtype=spec.key_dtype)
    d_out = cp_module.empty(_N, dtype=cp_module.uintp)
    t0 = perf_counter()
    raw_callable = make_fn(d_sorted, d_query, d_out)
    # Validate the compiled callable by executing it once.
    raw_callable(d_sorted, d_query, d_out, None, _N, _N)
    elapsed = (perf_counter() - t0) * 1000.0
    # Wrap so the call-site can use the temp-storage calling convention.
    adapted = _BinarySearchAdapter(raw_callable)
    d_temp = cp_module.empty(1, dtype=cp_module.uint8)
    return PrecompiledPrimitive(
        name=spec.name, make_callable=adapted,
        temp_storage=d_temp, temp_storage_bytes=1,
        high_water_n=_N, warmup_ms=elapsed,
    )


def _compile_upper_bound(
    spec: CCCLWarmupSpec, cp_module: Any, algorithms: Any,
) -> PrecompiledPrimitive:
    make_fn = algorithms.make_upper_bound
    d_sorted = cp_module.arange(_N, dtype=spec.key_dtype)
    d_query = cp_module.empty(_N, dtype=spec.key_dtype)
    d_out = cp_module.empty(_N, dtype=cp_module.uintp)
    t0 = perf_counter()
    raw_callable = make_fn(d_sorted, d_query, d_out)
    # Validate the compiled callable by executing it once.
    raw_callable(d_sorted, d_query, d_out, None, _N, _N)
    elapsed = (perf_counter() - t0) * 1000.0
    # Wrap so the call-site can use the temp-storage calling convention.
    adapted = _BinarySearchAdapter(raw_callable)
    d_temp = cp_module.empty(1, dtype=cp_module.uint8)
    return PrecompiledPrimitive(
        name=spec.name, make_callable=adapted,
        temp_storage=d_temp, temp_storage_bytes=1,
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
        h_init = get_host_init(spec.op_name, spec.key_dtype)
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
        h_init = get_host_init(spec.op_name, spec.key_dtype)
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
        d_out = cp_module.empty(_N, dtype=cp_module.uintp)
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
        h_init = get_host_init(spec.op_name, spec.key_dtype)
        return cached_callable(None, d_in, d_out, op, _N, h_init)

    if family == AlgorithmFamily.SEGMENTED_REDUCE:
        n_segs = 4
        seg_size = _N // n_segs
        d_values = cp_module.empty(_N, dtype=spec.key_dtype)
        d_out = cp_module.empty(n_segs, dtype=spec.key_dtype)
        d_starts = cp_module.arange(0, _N, seg_size, dtype=cp_module.int32)
        d_ends = d_starts + seg_size
        op = _get_op(spec.op_name)
        h_init = get_host_init(spec.op_name, spec.key_dtype)
        return cached_callable(
            None, d_values, d_out, d_starts, d_ends, op, n_segs, h_init,
        )

    if family in (AlgorithmFamily.LOWER_BOUND, AlgorithmFamily.UPPER_BOUND):
        d_sorted = cp_module.arange(_N, dtype=spec.key_dtype)
        d_query = cp_module.empty(_N, dtype=spec.key_dtype)
        d_out = cp_module.empty(_N, dtype=cp_module.uintp)
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
            result = _suppress_cccl_warnings(compiler, spec, cp_module, algorithms)
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
            from vibespatial.cuda import cccl_cubin_cache

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
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            logger.debug(
                "CCCL cache: hit but load failed for %s, falling back: %s",
                spec.name, exc,
            )
            return None

    def _save_to_disk_cache(
        self, spec: CCCLWarmupSpec, result: PrecompiledPrimitive,
    ) -> None:
        """Save a freshly built result to disk cache."""
        try:
            from vibespatial.cuda import cccl_cubin_cache
            cccl_cubin_cache.save_after_build(
                spec.name, spec.family, result.make_callable,
            )
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            logger.debug(
                "CCCL cache: save failed for %s: %s", spec.name, exc,
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
                result = _suppress_cccl_warnings(compiler, spec, cp_module, algorithms)
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
    if not SPEC_REGISTRY:
        return
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
    result["cccl_cold"].extend(_drain_requested_cccl_specs(timeout))
    if NVRTCPrecompiler._instance is not None:
        result["nvrtc_cold"].extend(NVRTCPrecompiler.get().ensure_warm(timeout=timeout))
    _warm_exact_polygon_intersection_route(timeout=timeout)
    _warm_many_vs_one_overlay_remainder_route(timeout=timeout)
    _warm_overlay_difference_segmented_union_route(timeout=timeout)
    _warm_device_linestring_buffer_route(timeout=timeout)
    _warm_device_centroid_buffer_route(timeout=timeout)
    result["cccl_cold"].extend(_drain_requested_cccl_specs(timeout))
    if NVRTCPrecompiler._instance is not None:
        result["nvrtc_cold"].extend(NVRTCPrecompiler.get().ensure_warm(timeout=timeout))
    result["cccl_cold"] = list(dict.fromkeys(result["cccl_cold"]))
    result["nvrtc_cold"] = list(dict.fromkeys(result["nvrtc_cold"]))
    return result


def _drain_requested_cccl_specs(timeout: float) -> list[str]:
    if CCCLPrecompiler._instance is None:
        return []
    return CCCLPrecompiler.get().ensure_warm(timeout=timeout)


def _drain_requested_pipeline_compilation(timeout: float) -> None:
    from .nvrtc_precompile import NVRTCPrecompiler

    _drain_requested_cccl_specs(timeout)
    if NVRTCPrecompiler._instance is not None:
        NVRTCPrecompiler.get().ensure_warm(timeout=timeout)


def _warm_exact_polygon_intersection_route(timeout: float = 60.0) -> None:
    """Warm the exact rowwise polygon intersection route used by public clip().

    The default NVRTC/CCCL warm registry does not naturally touch the exact
    rowwise polygon intersection dispatch used for concave polygon masks in
    strict-native clip workloads. Front-load that first-use cost here so
    shootout and benchmark children do not pay a 6-8s cold tax on their first
    concave polygon clip.
    """
    global _exact_polygon_intersection_warm_done

    with _exact_polygon_intersection_warm_lock:
        if _exact_polygon_intersection_warm_done:
            return

        try:
            from vibespatial.api.geoseries import GeoSeries
            from vibespatial.constructive.binary_constructive import (
                binary_constructive_owned,
            )
            from vibespatial.runtime import ExecutionMode, has_gpu_runtime

            if not has_gpu_runtime():
                return

            left = GeoSeries.from_wkt(
                [
                    "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
                    "POLYGON ((1 1, 5 1, 5 5, 1 5, 1 1))",
                ],
            ).values.to_owned()
            right = GeoSeries.from_wkt(
                [
                    "POLYGON ((0 0, 5 0, 5 1, 2 1, 2 4, 0 4, 0 0))",
                ],
            ).values.to_owned()
            _drain_requested_pipeline_compilation(timeout)
            binary_constructive_owned(
                "intersection",
                left,
                right,
                dispatch_mode=ExecutionMode.GPU,
                _prefer_exact_polygon_intersection=True,
            )
        except Exception:
            logger.debug(
                "exact polygon intersection warm probe failed",
                exc_info=True,
            )
            return

        _exact_polygon_intersection_warm_done = True


def _warm_many_vs_one_overlay_remainder_route(timeout: float = 60.0) -> None:
    """Warm the N-vs-1 polygon overlay remainder route used by public overlay().

    The public vegetation corridor workflow intersects many vegetation polygons
    against a single dissolved corridor polygon. When the corridor is concave or
    otherwise not eligible for SH clipping, ``_many_vs_one_intersection_owned``
    falls into the exact overlay microcell route. That path uses a wider CCCL
    surface area than the exact rowwise clip warm probe above, so prime it once
    here to avoid paying first-use compilation inside benchmark timings.
    """
    global _many_vs_one_overlay_warm_done

    with _many_vs_one_overlay_warm_lock:
        if _many_vs_one_overlay_warm_done:
            return

        try:
            from vibespatial.api.geoseries import GeoSeries
            from vibespatial.api.tools.overlay import _many_vs_one_intersection_owned
            from vibespatial.runtime import has_gpu_runtime
            from vibespatial.runtime.residency import Residency, TransferTrigger

            if not has_gpu_runtime():
                return

            left = GeoSeries.from_wkt(
                [
                    "POLYGON ((-1 1, 2 1, 2 4, -1 4, -1 1))",
                    "POLYGON ((1 -1, 4 -1, 4 2, 1 2, 1 -1))",
                    "POLYGON ((3 3, 6 3, 6 6, 3 6, 3 3))",
                ],
            ).values.to_owned()
            right = GeoSeries.from_wkt(
                [
                    "POLYGON ((0 0, 6 0, 6 2, 2 2, 2 6, 0 6, 0 0))",
                ],
            ).values.to_owned()
            left.move_to(
                Residency.DEVICE,
                trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                reason="warm many-vs-one overlay remainder route on device",
            )
            right.move_to(
                Residency.DEVICE,
                trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                reason="warm many-vs-one overlay remainder route on device",
            )
            _drain_requested_pipeline_compilation(timeout)
            _many_vs_one_intersection_owned(left, right, 0)
        except Exception:
            logger.debug(
                "many-vs-one polygon overlay warm probe failed",
                exc_info=True,
            )
            return

        _many_vs_one_overlay_warm_done = True


def _warm_overlay_difference_segmented_union_route(timeout: float = 60.0) -> None:
    """Warm the overlay-difference segmented-union route used by redevelopment flows.

    The strict-native redevelopment and site suitability shootouts hit
    ``_batched_overlay_difference_owned``. That path unions grouped right-side
    polygons via ``segmented_union_all(...)``, which in turn exercises the
    overlay split/candidate pipeline and CCCL ``upper_bound``. If that binary
    search callable is still cold, its first-use build cost lands inside the
    timed shootout child and can dominate or even exhaust the per-script
    timeout budget.
    """
    global _overlay_difference_warm_done

    with _overlay_difference_warm_lock:
        if _overlay_difference_warm_done:
            return

        try:
            from vibespatial.api import overlay
            from vibespatial.api.geodataframe import GeoDataFrame
            from vibespatial.api.geoseries import GeoSeries
            from vibespatial.runtime import has_gpu_runtime
            from vibespatial.spatial import segment_primitives as _segment_primitives  # noqa: F401

            if not has_gpu_runtime():
                return

            left = GeoDataFrame(
                {
                    "geometry": GeoSeries.from_wkt(
                        [
                            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
                            "POLYGON ((5 0, 9 0, 9 4, 5 4, 5 0))",
                        ],
                        crs="EPSG:4326",
                    )
                },
                geometry="geometry",
                crs="EPSG:4326",
            )
            right = GeoDataFrame(
                {
                    "geometry": GeoSeries.from_wkt(
                        [
                            "POLYGON ((1 1, 2 1, 2 3, 1 3, 1 1))",
                            "POLYGON ((2 1, 3 1, 3 3, 2 3, 2 1))",
                            "POLYGON ((6 1, 8 1, 8 3, 6 3, 6 1))",
                        ],
                        crs="EPSG:4326",
                    )
                },
                geometry="geometry",
                crs="EPSG:4326",
            )
            _drain_requested_pipeline_compilation(timeout)
            overlay(left, right, how="difference", keep_geom_type=True, make_valid=False)
        except Exception:
            logger.debug(
                "overlay difference segmented-union warm probe failed",
                exc_info=True,
            )
            return

        _overlay_difference_warm_done = True


def _warm_device_linestring_buffer_route(timeout: float = 60.0) -> None:
    """Warm the device-resident public linestring-buffer route.

    Small device-resident corridor-style line buffers now stay on the GPU to
    preserve ADR-0042's device-native boundary. Prime that exact public route
    here so shootout subprocesses do not pay the first-use NVRTC/CCCL tax
    inside their timed stage.
    """
    global _device_linestring_buffer_warm_done

    with _device_linestring_buffer_warm_lock:
        if _device_linestring_buffer_warm_done:
            return

        try:
            from vibespatial.api.geoseries import GeoSeries
            from vibespatial.geometry.device_array import DeviceGeometryArray
            from vibespatial.runtime import has_gpu_runtime
            from vibespatial.runtime.residency import Residency, TransferTrigger

            if not has_gpu_runtime():
                return

            owned = GeoSeries.from_wkt(
                [
                    "LINESTRING (0 0, 2 1, 4 0)",
                    "LINESTRING (5 0, 7 1, 9 0)",
                ],
                crs="EPSG:3857",
            ).values.to_owned()
            owned.move_to(
                Residency.DEVICE,
                trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                reason="warm device linestring-buffer public route on device",
            )
            _drain_requested_pipeline_compilation(timeout)
            DeviceGeometryArray._from_owned(owned, crs="EPSG:3857").buffer(1.0)
        except Exception:
            logger.debug(
                "device linestring-buffer warm probe failed",
                exc_info=True,
            )
            return

        _device_linestring_buffer_warm_done = True


def _warm_device_centroid_buffer_route(timeout: float = 60.0) -> None:
    """Warm the device polygon-centroid -> point-buffer public route.

    Flood exposure and related public flows produce a device-backed polygon
    column, then call ``centroid.buffer(...)`` on a filtered subset. The
    generic NVRTC/CCCL warm registry compiles the constituent units, but the
    first real public invocation in a fresh child process still pays route
    setup and module-load cost. Prime that exact route here so short-lived
    shootout subprocesses do not take the hit in their timed stage.
    """
    global _device_centroid_buffer_warm_done

    with _device_centroid_buffer_warm_lock:
        if _device_centroid_buffer_warm_done:
            return

        try:
            from vibespatial.api.geoseries import GeoSeries
            from vibespatial.geometry.device_array import DeviceGeometryArray
            from vibespatial.runtime import has_gpu_runtime
            from vibespatial.runtime.residency import Residency, TransferTrigger

            if not has_gpu_runtime():
                return

            owned = GeoSeries.from_wkt(
                [
                    "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))",
                    "POLYGON ((5 0, 9 0, 9 4, 5 4, 5 0))",
                ],
                crs="EPSG:3857",
            ).values.to_owned()
            owned.move_to(
                Residency.DEVICE,
                trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                reason="warm device centroid->buffer public route on device",
            )
            _drain_requested_pipeline_compilation(timeout)
            DeviceGeometryArray._from_owned(owned, crs="EPSG:3857").centroid.buffer(50.0)
        except Exception:
            logger.debug(
                "device centroid->buffer warm probe failed",
                exc_info=True,
            )
            return

        _device_centroid_buffer_warm_done = True

# Modules whose import triggers request_nvrtc_warmup() at module scope.
_NVRTC_CONSUMER_MODULES: tuple[str, ...] = (
    "vibespatial.spatial.indexing",
    "vibespatial.io.geojson_gpu",
    "vibespatial.io.wkb",
    "vibespatial.kernels.owned_take",
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
    "vibespatial.kernels.constructive.polygon_rect_intersection",
    "vibespatial.constructive.validity",
    "vibespatial.io.gpu_parse.structural",
    "vibespatial.io.gpu_parse.numeric",
    "vibespatial.io.gpu_parse.pattern",
    "vibespatial.io.wkt_gpu",
    "vibespatial.io.csv_gpu",
    "vibespatial.io.kml_gpu",
    "vibespatial.io.dbf_gpu",
    "vibespatial.io.shp_gpu",
    "vibespatial.io.gpu_parse.indexing",
    "vibespatial.io.osm_gpu",
    "vibespatial.io.fgb_gpu",
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
