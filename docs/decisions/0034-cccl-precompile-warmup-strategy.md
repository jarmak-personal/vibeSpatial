---
id: ADR-0034
status: accepted
date: 2026-03-14
deciders:
  - claude-opus
  - vibeSpatial maintainers
tags:
  - gpu-primitives
  - performance
  - cccl
  - startup
  - precompilation
  - make-reusable
---

# ADR-0034: CCCL make_* Pre-Compilation and Warmup Strategy

## Context

CCCL Python's `cuda.compute` algorithms are JIT-compiled on first
invocation. The cold-call penalty is ~950--1460 ms per unique
(algorithm, dtype, op, compute-capability) combination (measured in
ADR-0033 benchmarks). After that first call, CCCL caches the compiled
GPU code in process memory and subsequent calls are fast.

The `make_*` object-based API goes further: it returns a reusable
callable with explicit temp-storage management, eliminating both the
JIT-compilation check _and_ the per-call temp-storage query/allocation
overhead. Our benchmarks show `make_*` variants are 1.4--3.7x faster
than CuPy at all scales for scan and compaction.

**Current state:** All 9 CCCL algorithm call-sites in
`cccl_primitives.py` use the one-shot API. The `make_*` pattern is
only exercised in the benchmark script. NVRTC kernels are cached
in-memory by `CudaDriverRuntime._module_cache` but compilation
happens lazily on first kernel launch.

**Problem:** The first spatial operation a user runs pays the full
cold-JIT cost of every CCCL primitive in its execution path. For a
multi-stage pipeline (e.g., `sjoin` touching scan + select + sort +
segmented reduce), this can add 4--6 seconds of latency to the first
call --- entirely host-side compilation, not useful GPU work.

**Opportunity:** vibeSpatial knows at import time the full inventory
of CCCL primitives it uses. At GeoDataFrame construction time, it
additionally knows the geometry families present, which narrows the
relevant kernel set. At pipeline construction time, it knows the exact
operation sequence. We can exploit all three levels of knowledge to
pre-compile kernels on background CPU threads well before the user's
first operation lands.

### Thread-Safety and Parallelism Analysis

A key question: does CCCL's JIT release the GIL, and can we throw
many CPU threads at compilation in parallel? **Yes to both.**

#### GIL release during compilation

The expensive work runs outside the GIL at every layer:

| Layer | Call | GIL released? | Evidence |
|---|---|---|---|
| CCCL Cython bindings | `cccl_device_*_build()` | **Yes** | `with nogil:` in `_bindings_impl.pyx` for reduce, scan, select, sort, unique_by_key, segmented_reduce, binary_search |
| cuda-python NVRTC | `nvrtcCompileProgram()` | **Yes** | `with nogil:` in `nvrtc.pyx` |
| cuda-python driver | `cuModuleLoadData()` | **Yes** | `with nogil:` in `driver.pyx` |
| Numba op compilation | `numba.cuda.compile()` | **No** | Holds GIL during LLVM pipeline (~50--100ms per unique op, `lru_cache`'d after first call) |

The `cccl_device_*_build()` step is the dominant cost (~1--1.5s),
and it runs fully parallel across threads.

#### NVRTC is thread-safe

NVRTC operates on independent `nvrtcProgram` handles with no shared
mutable state between programs. Different threads can concurrently
compile different programs. NVRTC does not require a CUDA context
for compilation --- context is only needed for `cuModuleLoadData` to
load the resulting CUBIN (compiled to native SASS via `-arch=sm_XX`).

#### Serialization bottlenecks

| Bottleneck | Severity | Detail |
|---|---|---|
| CCCL cache (plain dict, no lock) | Low | Check-then-set is not atomic, but different cache keys = zero contention. Same key = redundant compilation (not corruption). |
| Numba `lru_cache` | Low | GIL protects dict ops; first compile of each unique op holds GIL ~50--100ms. Only ~6 unique ops across 18 specs. |
| `_wrapper_name_lock` in CCCL | Negligible | `threading.Lock()` protecting a global counter for symbol names. Acquired for ~ns. |
| CUDA driver `cuModuleLoadData` | Low | Supports concurrent loads on the same context from multiple threads. |
| Our `CudaDriverRuntime._module_cache` | **Medium** | No lock. Two threads compiling the same kernel race. **Must add a lock before Level 2 NVRTC warmup.** |

#### Thread scaling

Each NVRTC compilation inside `cccl_device_*_build()` is
single-threaded C++ compilation (does not parallelize internally).
With N cores, N concurrent compilations achieve ~N× throughput
until CPU-bound:

| Threads | Wall time (18 specs) | Speedup vs. serial | Notes |
|---|---|---|---|
| 1 | ~18--22s | 1× | Current lazy behavior |
| 4 | ~4--5s | ~4× | Conservative, safe on 4-core laptops |
| 8 | ~2--3s | ~7× | Sweet spot for most machines |
| 16 | ~1.5s | ~12× | Works; slight diminishing returns from Numba GIL phase |
| 32 | ~1.2s | ~15× | Past diminishing returns; memory waste (compiler temps ~100--200 MB per concurrent compile) |

The Numba GIL bottleneck is small: ~6 unique operators × ~100ms =
~600ms serialized, after which every thread hits the Numba cache and
skips straight to the `nogil` CCCL build.

#### Recommended `max_workers`

- **Level 1 (CCCL warmup):** `max_workers=8`. Only 2--12 specs
  submitted at a time (demand-driven). 8 threads finish any single
  module's specs in <1s wall.
- **Level 2 (NVRTC warmup):** `max_workers=min(os.cpu_count(), 16)`.
  Larger kernel set, benefits from more cores, but needs the
  `_module_cache` lock fix first.

## Design

### Demand-Driven Warmup Architecture

**Core principle:** Don't pre-compile everything at `import` time.
Pre-compile only the CCCL primitives that the user's code path
actually needs, triggered by module-level dependency declarations.

The dependency graph from consumer modules to CCCL primitives is
static and known at code-writing time:

```
sjoin / spatial_query.py
  ├── compact_indices    → select
  ├── exclusive_sum      → exclusive_scan
  ├── sort_pairs         → radix_sort / merge_sort
  ├── lower_bound        → lower_bound
  ├── upper_bound        → upper_bound
  └── segmented_reduce_* → segmented_reduce

overlay / overlay_gpu.py (kernel source in overlay/gpu_kernels.py)
  ├── exclusive_sum      → exclusive_scan
  ├── sort_pairs         → radix_sort / merge_sort
  └── unique_sorted_pairs → unique_by_key

dissolve / dissolve_pipeline.py
  └── sort_pairs         → radix_sort / merge_sort

spatial index / indexing.py
  └── sort_pairs         → radix_sort / merge_sort

GeoArrow IO / io_arrow.py
  └── exclusive_sum      → exclusive_scan

point_in_polygon / predicates
  └── compact_indices    → select

segment_primitives.py, point_constructive.py
  └── compact_indices    → select
```

A user who does `gs.within(other)` only needs `select` (2 specs:
int32 and int64). Pre-compiling `segmented_reduce`, `unique_by_key`,
and sort for them wastes ~14s of background CPU.

```
Level 1: Demand-driven module warmup
  - Triggered: when a consumer module first imports from cccl_primitives
  - Scope: only the CCCL specs that module declares as dependencies
  - Parallelism: ThreadPoolExecutor across the module's spec set
  - Blocking: NO --- background threads; first real call blocks
    only if its own spec isn't compiled yet

Level 2: GeoDataFrame-aware NVRTC warmup (deferred)
  - Triggered: when GeoDataFrame is created with device-resident data
  - Scope: NVRTC kernels for the geometry families present

Level 3: Pipeline-aware pre-compilation (deferred)
  - Triggered: when a pipeline plan is constructed (sjoin, overlay)
  - Scope: ensures all deps are warm before first stage executes
```

### Level 1: Demand-Driven Module Warmup

Each consumer module declares the CCCL primitives it needs via a
module-level constant. When that module is first imported, the
precompiler picks up only those specs and submits them to background
threads.

#### Primitive Spec Registry

```python
@dataclass(frozen=True, slots=True)
class CCCLWarmupSpec:
    """Specification for a CCCL make_* pre-compilation target."""
    name: str                    # e.g. "exclusive_scan_i32"
    make_fn: str                 # e.g. "make_exclusive_scan"
    input_dtype: np.dtype        # e.g. np.int32
    output_dtype: np.dtype       # e.g. np.int32
    op: Callable | OpKind        # the binary/unary operator
    h_init: np.ndarray | None    # host initial value
    extra_arrays: int            # additional typed arrays needed
    representative_n: int        # dummy array size (e.g. 128)
```

The full inventory (derived from current `cccl_primitives.py`):

| Spec Name | make_* function | Input dtype | Op | Used by |
|---|---|---|---|---|
| `exclusive_scan_i32` | `make_exclusive_scan` | int32 | sum | `exclusive_sum()` |
| `exclusive_scan_i64` | `make_exclusive_scan` | int64 | sum | `exclusive_sum()` |
| `select_i32` | `make_select` | int32 | predicate | `compact_indices()` |
| `select_i64` | `make_select` | int64 | predicate | `compact_indices()` |
| `reduce_sum_f64` | `make_reduce_into` | float64 | sum | `reduce_sum()` |
| `reduce_sum_i32` | `make_reduce_into` | int32 | sum | `reduce_sum()` |
| `segmented_reduce_sum_f64` | `make_segmented_reduce` | float64 | sum | `segmented_reduce_sum()` |
| `segmented_reduce_min_f64` | `make_segmented_reduce` | float64 | min | `segmented_reduce_min()` |
| `segmented_reduce_max_f64` | `make_segmented_reduce` | float64 | max | `segmented_reduce_max()` |
| `lower_bound_i32` | `make_lower_bound` | int32 | None | `lower_bound()` |
| `lower_bound_u64` | `make_lower_bound` | uint64 | None | `lower_bound()` |
| `upper_bound_i32` | `make_upper_bound` | int32 | None | `upper_bound()` |
| `upper_bound_u64` | `make_upper_bound` | uint64 | None | `upper_bound()` |
| `radix_sort_i32_i32` | `make_radix_sort` | int32/int32 | ascending | `sort_pairs()` |
| `radix_sort_u64_i32` | `make_radix_sort` | uint64/int32 | ascending | `sort_pairs()` |
| `merge_sort_u64_i32` | `make_merge_sort` | uint64/int32 | less | `sort_pairs()` |
| `unique_by_key_i32_i32` | `make_unique_by_key` | int32/int32 | eq | `unique_sorted_pairs()` |
| `unique_by_key_u64_i32` | `make_unique_by_key` | uint64/int32 | eq | `unique_sorted_pairs()` |

18 specs total, but no single operation uses all 18.

#### Module Dependency Declarations

Each consumer module declares its CCCL needs at the top of the file.
This is a static declaration, not a runtime probe:

```python
# spatial_query.py
from vibespatial.cuda.cccl_primitives import (
    compact_indices,
    exclusive_sum,
    lower_bound,
    segmented_reduce_min,
    sort_pairs,
    upper_bound,
)
from vibespatial.cuda.cccl_precompile import request_warmup

# Declare the 12 specs this module actually uses.
# request_warmup is a no-op if GPU is unavailable or specs
# are already submitted. It never blocks.
request_warmup([
    "exclusive_scan_i32", "exclusive_scan_i64",
    "select_i32", "select_i64",
    "radix_sort_i32_i32", "radix_sort_u64_i32",
    "merge_sort_u64_i32",
    "lower_bound_i32", "lower_bound_u64",
    "upper_bound_i32", "upper_bound_u64",
    "segmented_reduce_min_f64",
])
```

```python
# point_in_polygon.py --- only needs select
from vibespatial.cuda.cccl_precompile import request_warmup
request_warmup(["select_i32", "select_i64"])
```

```python
# io_arrow.py --- only needs exclusive_scan
from vibespatial.cuda.cccl_precompile import request_warmup
request_warmup(["exclusive_scan_i32", "exclusive_scan_i64"])
```

Because many vibespatial consumer modules are lazily imported from
the vendored GeoPandas layer (e.g., `spatial_query` and `indexing`
are imported inside method bodies in `array.py`), the warmup
naturally fires only when the user first touches code that needs
those primitives --- not at bare `import geopandas`.

#### What About Eagerly-Imported Modules?

Some consumer modules _are_ imported eagerly at `import geopandas`
time (e.g., `dissolve_pipeline` via `geodataframe.py`). For those,
their `request_warmup(["radix_sort_i32_i32", ...])` fires at import
time. This is fine --- it's only the 3 sort specs that `dissolve`
needs, not all 18. The cost is ~1s background CPU, not ~3s.

#### Deduplication

`request_warmup` is idempotent. If `spatial_query.py` requests
`select_i32` and `point_in_polygon.py` also requests `select_i32`,
the spec is compiled once. The precompiler tracks submitted spec
names in a `set` and skips duplicates.

#### Cost by Operation

| First operation used | Specs warmed | Background CPU cost |
|---|---|---|
| `gs.within(other)` | 2 (select) | ~1s on 2 threads |
| `gs.to_wkb()` | 2 (exclusive_scan) | ~1s on 2 threads |
| `gpd.sjoin(a, b)` | 12 (scan+select+sort+bound+seg_reduce) | ~2s on 8 threads |
| `gpd.overlay(a, b)` | 6 (scan+sort+unique) | ~1.5s on 6 threads |
| `gs.dissolve(by=...)` | 3 (sort) | ~1s on 3 threads |

Compare to the blast-everything-at-import approach: ~3s background
CPU for _every_ user, even those who never touch sjoin.

#### Compilation Flow

```python
class CCCLPrecompiler:
    """Demand-driven background pre-compilation of CCCL make_* callables.

    Unlike a blast-all-at-import precompiler, this accumulates specs
    incrementally via request_warmup() calls from consumer modules.
    Each consumer declares only the specs it needs.
    """

    _instance: CCCLPrecompiler | None = None

    def __init__(self, max_workers: int = 8):
        self._spec_registry: dict[str, CCCLWarmupSpec] = {}
        self._cache: dict[str, PrecompiledPrimitive] = {}
        self._futures: dict[str, Future] = {}
        self._submitted: set[str] = set()  # dedup guard
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="cccl-warmup",
        )
        self._start_time: float | None = None
        self._diagnostics: list[WarmupDiagnostic] = []

    @classmethod
    def get(cls) -> CCCLPrecompiler:
        """Lazy singleton. Created on first request_warmup() call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def request(self, spec_names: list[str]) -> None:
        """Submit specs for background compilation. Idempotent.

        Called by consumer modules at import time. Never blocks.
        Skips specs that are already submitted or compiled.
        """
        with self._lock:
            new_specs = [
                n for n in spec_names
                if n not in self._submitted
            ]
            if not new_specs:
                return
            if self._start_time is None:
                self._start_time = perf_counter()
            for name in new_specs:
                self._submitted.add(name)
                spec = SPEC_REGISTRY[name]
                future = self._executor.submit(self._compile_one, spec)
                self._futures[name] = future

    def _compile_one(self, spec: CCCLWarmupSpec) -> PrecompiledPrimitive:
        """JIT-compile one make_* callable with dummy arrays."""
        import cupy as cp
        from cuda.compute import algorithms

        t0 = perf_counter()

        # Create small dummy arrays with the right dtypes
        d_in = cp.empty(spec.representative_n, dtype=spec.input_dtype)
        d_out = cp.empty(spec.representative_n, dtype=spec.output_dtype)

        # Call the make_* function (triggers JIT compilation)
        make_fn = getattr(algorithms, spec.make_fn)
        # ... build args from spec ...
        callable_obj = make_fn(*args)

        # Query temp storage size for a representative workload
        temp_bytes = callable_obj(None, *query_args)
        d_temp = cp.empty(max(int(temp_bytes), 1), dtype=cp.uint8)

        elapsed_ms = (perf_counter() - t0) * 1000.0
        result = PrecompiledPrimitive(
            name=spec.name,
            make_callable=callable_obj,
            temp_storage=d_temp,
            temp_storage_bytes=int(temp_bytes),
            warmup_ms=elapsed_ms,
        )
        self._cache[spec.name] = result
        return result

    def get_compiled(self, name: str, timeout: float = 5.0) -> PrecompiledPrimitive | None:
        """Get a pre-compiled primitive. Blocks if compilation in progress.

        If the spec was never requested (consumer didn't declare it),
        returns None immediately --- caller uses one-shot fallback.
        """
        if name in self._cache:
            return self._cache[name]
        if name in self._futures:
            try:
                return self._futures[name].result(timeout=timeout)
            except (TimeoutError, Exception):
                return None  # Fall back to one-shot
        return None

    def status(self) -> dict:
        """Diagnostic snapshot for observability."""
        return {
            "submitted": len(self._submitted),
            "compiled": len(self._cache),
            "pending": sum(1 for f in self._futures.values() if not f.done()),
            "failed": sum(1 for f in self._futures.values()
                          if f.done() and f.exception()),
            "wall_ms": (perf_counter() - self._start_time) * 1000
                       if self._start_time else 0,
            "per_primitive": [
                {"name": d.name, "ms": d.elapsed_ms, "ok": d.success}
                for d in self._diagnostics
            ],
        }


# Module-level convenience function used by consumer modules.
def request_warmup(spec_names: list[str]) -> None:
    """Non-blocking request to pre-compile CCCL specs.

    Safe to call at module scope. No-op if GPU is not available.
    """
    if not has_gpu_runtime():
        return
    CCCLPrecompiler.get().request(spec_names)
```

#### Integration with cccl_primitives.py

Each function in `cccl_primitives.py` gains a fast path that checks
the precompiler cache before falling back to one-shot:

```python
def exclusive_sum(values, out=None, *, strategy=PrimitiveStrategy.AUTO):
    """Exclusive prefix sum."""
    # ... existing strategy selection ...
    if use_cccl:
        dtype_key = f"exclusive_scan_{_dtype_suffix(values.dtype)}"
        precompiled = CCCLPrecompiler.get().get_compiled(dtype_key, timeout=2.0)
        if precompiled is not None:
            # Ensure temp storage is large enough
            temp = _ensure_temp(precompiled, int(values.size))
            precompiled.make_callable(
                temp, values, out, _sum_op, int(values.size), _h_init
            )
            return out
        # Fallback: one-shot API (cold or cache miss)
        algorithms.exclusive_scan(values, out, _sum_op, _h_init, int(values.size))
        return out
```

#### Temp Storage Sizing

The `make_*` callable's temp storage requirement depends on
`num_items`, not on the data content. For most CUB algorithms, temp
storage scales as O(n) or less. Strategy:

1. Pre-allocate temp storage at `representative_n = 128` during warmup
   (just to trigger JIT; actual buffer is tiny).
2. At first real call, re-query temp storage for actual `num_items`
   and allocate. Cache this as the "high-water mark" buffer.
3. On subsequent calls, reuse if `num_items <= high_water_n`, else
   re-query and grow. Never shrink --- spatial workloads are typically
   consistent in scale within a session.

```python
def _ensure_temp(precompiled: PrecompiledPrimitive, num_items: int):
    """Grow temp storage if the current buffer is too small."""
    if num_items <= precompiled.high_water_n:
        return precompiled.temp_storage
    # Re-query for the new size
    needed = precompiled.make_callable(None, *query_args_for(num_items))
    if needed > precompiled.temp_storage_bytes:
        precompiled.temp_storage = cp.empty(needed, dtype=cp.uint8)
        precompiled.temp_storage_bytes = needed
    precompiled.high_water_n = num_items
    return precompiled.temp_storage
```

### Level 2: Demand-Driven NVRTC Warmup

The same demand-driven pattern applies to our 14 custom NVRTC
compilation units. Unlike CCCL (where each spec is a unique
algorithm × dtype × op combo), NVRTC kernels are simpler: each
source string compiles exactly once per process — no dtype variance,
no compile-option variance, deterministic cache keys.

#### NVRTC Kernel Inventory

| # | Source Constant | Module | CUDA Lines | Kernels | Est. Compile Cost |
|---|---|---|---|---|---|
| 1 | `_SPATIAL_QUERY_KERNEL_SOURCE` | `spatial_query_kernels.py` | 948 | 19 | ~200--400ms |
| 2 | `_POLYGON_PREDICATES_KERNEL_SOURCE` | `polygon_kernels.py` | 750 | 10 | ~150--300ms |
| 3 | `_SEGMENT_DISTANCE_KERNEL_SOURCE` | `segment_distance_kernels.py` | 493 | 10 | ~100--250ms |
| 4 | `_POINT_IN_POLYGON_KERNEL_SOURCE` | `point_in_polygon_source.py` | 434 | 8 | ~80--200ms |
| 5 | `_POINT_DISTANCE_KERNEL_SOURCE` | `point_distance_kernels.py` | 399 | 4 | ~80--180ms |
| 6 | `_POINT_BINARY_RELATIONS_KERNEL_SOURCE` | `point_relations_kernels.py` | 345 | 5 | ~70--150ms |
| 7 | `_MULTIPOINT_BINARY_RELATIONS_KERNEL_SOURCE` | `point_relations_kernels.py` | 329 | 6 | ~60--140ms |
| 8 | `_OVERLAY_SPLIT_KERNEL_SOURCE` | `overlay/gpu_kernels.py` | 253 | 4 | ~50--120ms |
| 9 | `_BOUNDS_KERNEL_SOURCE` | `geometry_analysis_source.py` | 247 | 6 | ~50--100ms |
| 10 | `_WKB_ENCODE_KERNEL_SOURCE` | `wkb_kernels.py` | 213 | 6 | ~40--90ms |
| 11 | `_MORTON_RANGE_KERNEL_SOURCE` | `spatial_query_kernels.py` | 133 | 3 | ~30--60ms |
| 12 | `_SEGMENT_INTERSECTION_KERNEL_SOURCE` | `segment_primitives_kernels.py` | 117 | 1 | ~25--50ms |
| 13 | `_POINT_CONSTRUCTIVE_KERNEL_SOURCE` | `point_kernels.py` | 83 | 3 | ~20--40ms |
| 14 | `_INDEXING_KERNEL_SOURCE` | `indexing_kernels.py` | 41 | 1 | ~10--20ms |

**Totals:** ~4,785 lines of CUDA C, 85 `__global__` entry points,
14 compilation units, ~1.0--2.5s serial compilation.

#### Why It's Simpler Than CCCL Warmup

- **No type specialization.** All kernels use FP64 exclusively. One
  source string → one cache key → one compilation. No combinatorial
  explosion of (algorithm, dtype, op) tuples.
- **No operator JIT.** No Numba compilation of Python callables. The
  CUDA C source is a module-level Python string constant, fully
  determined at import time.
- **Deterministic cache keys.** `make_kernel_cache_key(prefix, source)`
  hashes the source string. Since sources are constants, cache keys
  are process-invariant.
- **Same GIL release.** `nvrtcCompileProgram` and `cuModuleLoadData`
  both run with `nogil` in Cython. 14 compilations on 8 threads =
  **~200--400ms wall time** vs. ~1--2.5s serial.

#### Trigger Pattern

Like CCCL Level 1, use the same demand-driven module-import trigger.
Each module that defines NVRTC kernel sources declares its compile
units at module scope:

```python
# spatial_query.py — imports source from spatial_query_kernels.py
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.kernels.core.spatial_query_kernels import (
    _SPATIAL_QUERY_KERNEL_SOURCE, _MORTON_RANGE_KERNEL_SOURCE,
)

request_nvrtc_warmup([
    ("spatial-query", _SPATIAL_QUERY_KERNEL_SOURCE, _SQ_KERNEL_NAMES),
    ("morton-range", _MORTON_RANGE_KERNEL_SOURCE, _MR_KERNEL_NAMES),
])
```

```python
# point_in_polygon.py — imports source from point_in_polygon_source.py
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.kernels.predicates.point_in_polygon_source import (
    _POINT_IN_POLYGON_KERNEL_SOURCE,
)

request_nvrtc_warmup([
    ("pip", _POINT_IN_POLYGON_KERNEL_SOURCE, _PIP_KERNEL_NAMES),
])
```

The singleton precompiler deduplicates by cache key and submits
new compilations to its `ThreadPoolExecutor`. The `_xxx_kernels()`
lazy-init functions gain a fast path checking the cache first:

```python
def _spatial_query_kernels():
    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key("spatial-query",
                                       _SPATIAL_QUERY_KERNEL_SOURCE)
    # Fast path: already warm from background thread
    if cache_key in runtime._module_cache:
        return runtime._module_cache[cache_key]
    # Slow path: compile synchronously (first call before warmup finishes)
    return runtime.compile_kernels(
        cache_key=cache_key,
        source=_SPATIAL_QUERY_KERNEL_SOURCE,
        kernel_names=_SQ_KERNEL_NAMES,
    )
```

#### NVRTCPrecompiler Sketch

```python
class NVRTCPrecompiler:
    """Demand-driven background NVRTC compilation.

    Follows the same singleton + request() + dedup pattern
    as CCCLPrecompiler, but simpler: no type specs, just
    (prefix, source, kernel_names) tuples.
    """
    _instance: NVRTCPrecompiler | None = None

    def __init__(self, max_workers: int | None = None):
        self._submitted: set[str] = set()  # cache keys
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers or min(os.cpu_count() or 4, 16),
            thread_name_prefix="nvrtc-warmup",
        )

    @classmethod
    def get(cls) -> NVRTCPrecompiler:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def request(self, units: list[tuple[str, str, tuple[str, ...]]]) -> None:
        """Submit (prefix, source, kernel_names) for background compile."""
        runtime = get_cuda_runtime()
        with self._lock:
            for prefix, source, kernel_names in units:
                cache_key = make_kernel_cache_key(prefix, source)
                if cache_key in self._submitted:
                    continue
                self._submitted.add(cache_key)
                self._executor.submit(
                    runtime.compile_kernels,
                    cache_key=cache_key,
                    source=source,
                    kernel_names=kernel_names,
                )
```

#### Cost by Operation (NVRTC only)

| First operation used | Compilation units | Background wall time |
|---|---|---|
| `gs.within(other)` (PIP) | 1 unit (434 lines) | ~80--200ms |
| `gs.distance(other)` (points) | 1 unit (399 lines) | ~80--180ms |
| `gpd.sjoin(a, b)` | 2--3 units (spatial_query + bounds + pip) | ~200--400ms |
| `gpd.overlay(a, b)` | 2 units (overlay + seg_intersection) | ~75--170ms |
| All 14 units | 14 units (~4,785 lines) | ~200--400ms on 8+ threads |

#### Prerequisite: `_module_cache` Thread Safety

`CudaDriverRuntime.compile_kernels()` currently has no lock on
`_module_cache`. Before enabling NVRTC warmup, add a
`threading.Lock` to guard the check-then-compile-then-store sequence:

```python
class CudaDriverRuntime:
    def __init__(self, ...):
        ...
        self._module_cache: dict[str, dict[str, Any]] = {}
        self._module_cache_lock = threading.Lock()

    def compile_kernels(self, *, cache_key, source, kernel_names, options=()):
        # Fast path: no lock needed for reads (dict reads are GIL-atomic)
        if cache_key in self._module_cache:
            return self._module_cache[cache_key]
        with self._module_cache_lock:
            # Double-check after acquiring lock
            if cache_key in self._module_cache:
                return self._module_cache[cache_key]
            # ... compile ...
            self._module_cache[cache_key] = kernels
        return kernels
```

### Level 3: Pipeline-Aware Pre-Compilation

When a high-level operation like `sjoin`, `overlay`, or `dissolve` is
invoked, it constructs an execution plan. Before executing the first
stage, the plan can declare its CCCL and NVRTC requirements:

```python
class PipelinePlan:
    """Execution plan for a multi-stage spatial operation."""
    stages: list[PipelineStage]
    cccl_requirements: list[str]  # spec names needed
    nvrtc_requirements: list[str]  # cache keys needed

    def ensure_warm(self, timeout: float = 3.0) -> list[str]:
        """Block until all required primitives are compiled.
        Returns list of any specs that timed out (will use one-shot)."""
        cold = []
        for spec_name in self.cccl_requirements:
            result = _precompiler.get(spec_name, timeout=timeout)
            if result is None:
                cold.append(spec_name)
        return cold
```

This level has the highest precision but requires pipeline code to
declare its requirements. It naturally emerges as pipelines are built.

### Key Design Properties

#### 1. Never blocks import

Level 1 warmup runs on daemon threads. `import vibespatial` returns
immediately. If a user operation arrives before warmup completes,
`get()` blocks for up to `timeout` seconds on just the needed
primitive, then falls back to one-shot. Worst case = same as today.

#### 2. Never fails loudly

If CCCL or GPU is not available, the precompiler is never
instantiated. If individual `make_*` calls fail (unsupported API
version, etc.), the fallback is the existing one-shot path. All
failures are logged to diagnostics, never raised.

#### 3. Observable

Warmup status is visible via `vibespatial.precompile_status()` and
dispatch events log whether a pre-compiled callable was used:

```python
>>> import geopandas as gpd
>>> # At this point, dissolve_pipeline imported eagerly.
>>> # Only 3 specs (sort variants) are warming in background.
>>> from vibespatial.cuda.cccl_precompile import CCCLPrecompiler
>>> CCCLPrecompiler.get().status()
{"submitted": 3, "compiled": 3, "pending": 0, "wall_ms": 1012.4, ...}

>>> # User calls sjoin --- spatial_query.py is lazily imported.
>>> # Its request_warmup() fires, adding 12 specs (some deduplicated
>>> # against the 3 sort specs already compiled by dissolve).
>>> gpd.sjoin(a, b)
>>> CCCLPrecompiler.get().status()
{"submitted": 15, "compiled": 15, "pending": 0, "wall_ms": 2341.7, ...}
```

#### 4. Thread-safe --- and truly parallel

- Every `cccl_device_*_build()` call releases the GIL via Cython
  `with nogil:`. Threads run on separate CPU cores during the
  ~1--1.5s compilation phase --- this is real parallelism, not just
  concurrency.
- CCCL's internal JIT cache is thread-safe (keyed by compute
  capability, dtype, op bytecode). Different specs use different
  cache keys → zero contention.
- `make_*` returns an independent callable; no shared mutable state.
- Pre-compiled objects are stored in a `dict` protected by our
  `_lock` and the GIL (safe for concurrent reads after write).
- CuPy allocations on background threads use CuPy's per-thread
  default memory pool.
- See "Thread-Safety and Parallelism Analysis" in Context for full
  bottleneck inventory and scaling data.

#### 5. Compatible with CCCL's own caching

CCCL already caches compiled GPU code internally. Our `make_*`
pre-compilation triggers that cache population as a side effect. Even
if the user later calls the one-shot API (bypassing our `make_*`
object), the one-shot call also hits CCCL's warm cache. The
pre-compilation is strictly additive.

### Cost Model

**CCCL (Level 1):**

| What | Cost | When Paid |
|---|---|---|
| Per-spec `make_*` JIT | ~1--1.5s CPU per spec | Background, on first import of consumer module |
| Dummy arrays per spec (128 elements) | ~2 KB device memory | Freed after warmup |
| Temp-storage buffers per spec | ~few hundred bytes | Grown on first real call |
| `ThreadPoolExecutor` (8 threads) | ~1 MB RSS | Lazy singleton, lives for process |

Worst case (user touches every operation): all 18 specs warm across
multiple background bursts, total ~2--3s wall on 8 threads. Typical
case (single operation family): 2--6 specs, ~1s wall.

**NVRTC (Level 2):**

| What | Cost | When Paid |
|---|---|---|
| Per-unit NVRTC compile | ~20--400ms CPU per unit | Background, on first import of kernel module |
| 14 compilation units total | ~1.0--2.5s serial | ~200--400ms wall on 8+ threads |
| `ThreadPoolExecutor` (up to 16 threads) | ~1 MB RSS | Lazy singleton, lives for process |

NVRTC warmup is cheaper per unit and lacks the Numba GIL bottleneck.
Thread scaling is near-linear because `nvrtcCompileProgram` and
`cuModuleLoadData` both release the GIL.

**Combined worst case:** user imports every operation module and
triggers all 18 CCCL specs + 14 NVRTC units. Background CPU time:
~22--30s total across two thread pools, but wall time is ~2--3s
(CCCL-dominated). CPU-only users pay exactly zero.

### What About On-Disk Caching?

CCCL does not currently expose an API to serialize compiled GPU code
to disk. NVRTC CUBIN output is already disk-cached by
`CudaDriverRuntime` (see `_read_cached_cubin` / `_write_cached_cubin`
in `cuda_runtime.py`), keyed by compute capability, NVRTC version,
source hash, and compile options. This eliminates NVRTC recompilation
across process restarts. Defer CCCL on-disk caching until CCCL ships
its own persistent cache or benchmarks show restart latency is
critical.

### What About cuda.coop?

`cuda.coop` provides block- and warp-level primitives for use inside
Numba CUDA kernels. These have a different compilation model (they
produce `.files` that link into `@cuda.jit` kernels). Pre-compilation
would mean calling `make_*` eagerly at import time, which pre-compiles
the LTO-IR. This is compatible with the same background-thread
pattern but only becomes relevant if we adopt Numba CUDA for kernel
authoring. Currently all our kernels are NVRTC-based.

**Action:** Revisit when we evaluate Numba CUDA for any kernel family.

## Decision

1. **Implement Level 1** (demand-driven CCCL warmup) as the first
   deliverable. Each consumer module declares its specs via
   `request_warmup()` at module scope; the singleton precompiler
   accumulates and deduplicates across modules.

2. **Implement Level 2** (demand-driven NVRTC warmup) alongside
   Level 1. Same pattern: each kernel module declares its
   `(prefix, source, kernel_names)` tuples via
   `request_nvrtc_warmup()` at module scope. Simpler than Level 1
   (no dtype/op variance --- just 14 fixed compilation units).

3. **Add `_module_cache_lock`** to `CudaDriverRuntime` with
   double-checked locking in `compile_kernels()`. Required
   prerequisite for Level 2 thread safety.

4. **Refactor `cccl_primitives.py`** to use `make_*` callables at
   runtime, with the precompiler cache as the primary path and
   one-shot as fallback.

5. **Add `CCCLWarmupSpec` declarations** in a central `SPEC_REGISTRY`
   dict inside `cccl_precompile.py`, keyed by spec name.

6. **Add `request_warmup()` / `request_nvrtc_warmup()` calls** to
   each consumer module listing only their own deps.

7. **Expose `precompile_status()`** for observability across both
   CCCL and NVRTC precompilers.

8. **Defer Level 3** (pipeline-aware pre-compilation) until
   pipeline plan objects are mature enough to declare requirements.

9. **Do not implement on-disk caching** until CCCL provides a stable
   serialization API or benchmarks show restart latency is critical.

## Consequences

### Positive

- GPU operations see zero JIT latency when their consumer module
  was imported early enough for background threads to finish.
- Both CCCL and NVRTC compilation run outside the GIL, enabling
  true multi-core parallelism during background warmup.
- Users who only touch one operation family (e.g., point-in-polygon)
  pay only for their CCCL specs + 1 NVRTC unit, not all 18 + 14.
- `make_*` reusable callables with pre-allocated temp storage are
  1.4--3.7x faster than CuPy for scan/compaction at every call,
  not just the first.
- NVRTC warmup eliminates the ~50--400ms first-launch penalty per
  kernel module, which compounds to ~1--2.5s in multi-stage pipelines.
- Observable warmup status prevents silent cold-start regressions.
- Strictly additive: no risk of breaking existing one-shot or
  lazy-compile paths.
- CPU-only users pay exactly zero (both `request_warmup` and
  `request_nvrtc_warmup` are no-ops without GPU).

### Negative

- Background threads consume CPU when consumer modules are imported
  (mitigated: only the specs/units that module needs, bounded pools).
- Two thread pools (CCCL + NVRTC) may compete for CPU with user
  code in the first ~2--3s (mitigated: daemon threads at normal
  priority, total background CPU is bounded).
- Small device memory allocation during CCCL warmup (mitigated:
  dummy arrays are 128 elements, freed after warmup).
- Each consumer module needs a `request_warmup()` / 
  `request_nvrtc_warmup()` call at module scope (mitigated: simple
  one-liner, enforced by test).
- Adding `_module_cache_lock` to `CudaDriverRuntime` adds a small
  overhead to the cache-hit path (mitigated: double-checked locking
  means the lock is only acquired on cache miss).

### Neutral

- CCCL version upgrades may change `make_*` signatures. Guarded
  by `hasattr` checks, same as current benchmark code.
- Does not change the GPU-first / CPU-fallback contract.

## Alternatives Considered

### A: Lazy caching only (no background warmup)

Let CCCL's internal cache handle warm-up. First call pays JIT, all
subsequent calls are fast.

**Rejected:** This is the status quo. The first-operation latency
spike is a real UX problem, especially in notebook workflows where
users expect sub-second response after `import`.

### B: Blast all 18 specs at `import vibespatial`

Pre-compile every CCCL primitive signature at import time regardless
of which operations the user will actually call.

**Rejected:** Wastes ~3s of background CPU for users who only need 2
specs. Also forces CUDA context initialization for CPU-only users
unless guarded. Demand-driven warmup is strictly better: same
background-thread mechanism, but scoped to actual usage.

### C: Explicit `vibespatial.warmup()` API only

Require the user to opt in to pre-compilation.

**Rejected:** Users who don't know about warmup get the worst
experience. The background-thread approach gives the benefit
automatically without requiring user action.

### D: Fork / multiprocessing for warmup isolation

Use a separate process for warmup to avoid GIL contention.

**Rejected:** CCCL JIT caches are per-process. Compiling in a child
process and transferring compiled GPU code back is not supported.
Thread-based warmup is sufficient because every `cccl_device_*_build()`
call releases the GIL via Cython `with nogil:` --- threads achieve
true CPU parallelism, not just concurrency. Measured scaling is
near-linear up to 8 threads (see Thread-Safety Analysis in Context).

### E: `max_workers=32` to minimize wall time

Throw maximum CPU at warmup to finish compilation as fast as possible.

**Rejected for Level 1:** We only have 2--12 specs per demand burst.
8 threads already finish in <1s wall. Beyond 8 threads, diminishing
returns from Numba GIL serialization (~600ms) and compiler memory
overhead (~100--200 MB per concurrent NVRTC compile). Revisit for
Level 2 if the NVRTC kernel set grows large.
