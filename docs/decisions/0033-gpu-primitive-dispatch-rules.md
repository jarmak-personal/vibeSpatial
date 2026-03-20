---
id: ADR-0033
status: accepted
date: 2026-03-12
deciders:
  - claude-opus
  - vibeSpatial maintainers
tags:
  - gpu-primitives
  - performance
  - cupy
  - cccl
  - kernel-strategy
---

# ADR-0033: GPU Primitive Dispatch Rules

## Context

Phase 9 proved that CCCL Python's `make_*` reusable callables beat CuPy
at all scales for overlapping primitives (compaction, scan, reduction)
once JIT is warm. CCCL also wins on algorithmic primitives CuPy lacks
(radix sort, unique-by-key, segmented reduce, binary search). As we
expand GPU coverage to polygon-heavy and multi-stage pipelines, the team
needs clear rules for when to reach for each tool — without
re-discovering the tradeoffs from scratch on every new kernel.

Benchmark results (2026-03-12, `scripts/benchmark_cccl_vs_cupy.py`):

- **Compaction (select):** CCCL warm beats CuPy at all scales (10K–10M).
  `make_*` is 1.4–3.1× faster than CuPy.
- **Exclusive scan:** CCCL warm beats CuPy at ≥1M. `make_*` is
  1.8–3.7× faster than CuPy at all scales.
- **Reduction:** CuPy is marginally faster at small scales; CCCL `make_*`
  is competitive. Difference is small enough to keep CuPy as default.
- **Segmented reduce:** No CuPy equivalent. CCCL only.

Cold-call JIT penalty is ~950–1460ms (one-time per process), but
amortised across all subsequent calls. The `make_*` pattern eliminates
this entirely for reusable callables.

This ADR captures the rules, explains *why* each rule holds, identifies
where the rules shift for polygon-family work, and defines the decision
procedure for borderline cases.

### CCCL Python surface (unstable, as of 2026-03-12)

Reference: https://nvidia.github.io/cccl/unstable/python/index.html

The `cuda.compute.algorithms` module exposes significantly more than we
currently use. Full inventory of device-wide algorithms:

| Algorithm | What it does | We use it? |
|---|---|---|
| `reduce_into` | Device-wide reduction | Available, CuPy default |
| `inclusive_scan` | Inclusive prefix scan | No (CuPy `cumsum`) |
| `exclusive_scan` | Exclusive prefix scan | **Yes — CCCL default** |
| `select` | Stream compaction by predicate | **Yes — CCCL default** |
| `radix_sort` | Key-value radix sort | **Yes** |
| `merge_sort` | Key-value merge sort (custom comparator) | **Yes** |
| `unique_by_key` | Deduplicate sorted keys, carry values | **Yes** |
| `segmented_sort` | Sort within offset-delimited segments | **No — high value for polygon work** |
| `segmented_reduce` | Reduce within offset-delimited segments | **Yes** (sum, min, max) |
| `lower_bound` | Binary search (first insertion point) in sorted array | **Yes** |
| `upper_bound` | Binary search (last insertion point) in sorted array | **Yes** |
| `three_way_partition` | Partition into 3 groups by two predicates | **No — useful for family-tag partitioning** |
| `histogram_even` | Evenly-spaced histogram bins | No (grid index potential) |
| `unary_transform` | Element-wise unary transform | No (CuPy) |
| `binary_transform` | Element-wise binary transform | No (CuPy) |

Every algorithm also has a `make_*` variant that returns a reusable
callable with pre-allocated temporary storage. This eliminates the
cold-call JIT penalty on second and subsequent invocations.

Iterators (lazy, zero-allocation):

| Iterator | What it does | Impact |
|---|---|---|
| `CountingIterator` | Integer sequence without materializing array | Replaces `cp.arange` allocations |
| `TransformIterator` | Lazy element-wise transform on input | Fuses transforms with algorithms |
| `TransformOutputIterator` | Lazy transform on write | Fuses output transforms |
| `ZipIterator` | Combine arrays into tuple stream | Avoids struct-of-arrays shuffling |

Block/warp-level (`cuda.coop`): block-level scan, reduce, sort,
exchange, load/store — usable inside Numba CUDA kernels. Not directly
applicable to our NVRTC kernel pattern but relevant if we adopt Numba
for any kernel authoring.

## Decision

### Tier 1: Custom NVRTC kernels (geometry-specific compute)

**Use for:** Any operation whose inner loop is geometry-specific —
point-in-polygon winding, segment intersection classification, split
event emission, half-edge traversal, coordinate extraction from WKB
payloads, etc.

**Why:** These operations have no CuPy or CCCL equivalent. They must be
hand-written CUDA. The NVRTC compile cost is paid once per
`make_kernel_cache_key` and cached in `CudaDriverRuntime._module_cache`
for the process lifetime.

**Pattern:** Write the kernel as a `_KERNEL_SOURCE` string constant,
compile via `runtime.compile_kernels()`, launch via `runtime.launch()`.
This is the established pattern in `point_in_polygon.py`,
`segment_primitives.py`, `overlay_gpu.py`, `point_binary_relations.py`,
`point_constructive.py`, `spatial_query.py`, and `indexing.py`.

**Polygon-family note:** This tier gets *more* important for polygons,
not less. Ring traversal, part offset indirection, winding number
accumulation across holes — these are all geometry-specific loops that
only custom kernels can express.

### Tier 2: CuPy built-ins (element-wise and allocation-friendly ops)

**Use for:** Element-wise transforms, boolean masking, reductions
(where the margin is small), gather/scatter via fancy indexing,
concatenation, and any operation where CuPy's zero-JIT path is the
simplest option.

**Specific mappings (CuPy remains default):**

| Operation | CuPy call | Notes |
|---|---|---|
| Boolean mask count | `int(mask.sum())` | Simpler than CCCL reduce |
| Element-wise transform | `cp.where(cond, a, b)` | No CCCL advantage |
| Reduction | `cp.sum(values)` / `cp.min(values)` | Marginal CuPy edge at small scale |
| Gather/scatter | fancy indexing `a[indices]` | No CCCL equivalent |
| Concatenation | `cp.concatenate(...)` | No CCCL equivalent |

**Operations moved to CCCL default (benchmarked 2026-03-12):**

| Operation | CCCL call | Why switched |
|---|---|---|
| Compaction (bool mask → indices) | `cp.flatnonzero` (CuPy) | CCCL `make_select` bakes predicate closure device pointers, preventing reuse; one-shot `select()` re-JITs per array size class (~5-6s each). CuPy is 0.2ms with no JIT. CCCL available via explicit `CompactionStrategy.CCCL_SELECT`. |
| Exclusive prefix sum | `cccl_algorithms.exclusive_scan` | `make_*` is 1.8–3.7× faster than CuPy at all scales |

CuPy paths remain available via strategy overrides for any pipeline
that needs them.

**Why CuPy still wins for element-wise ops:**

1. **No per-call JIT.** CuPy dispatches to pre-compiled CUDA kernels.
   For element-wise ops where CCCL offers no algorithmic advantage,
   the simpler CuPy path is preferred.

2. **No Python-callable boundary.** CuPy's built-ins have no tracing
   step, making them ideal for simple transforms and reductions.

**When to reconsider:** If a pipeline chains multiple CuPy element-wise
ops that could be fused with a CCCL `TransformIterator`, the fused
path may win on memory bandwidth. Benchmark before switching.

### Tier 3a: CCCL algorithmic primitives (operations CuPy lacks)

**Use for:** Radix sort, merge sort with custom comparators,
unique-by-key, segmented sort, segmented reduce, binary search, and
multi-way partitioning.

**Currently used:**

| Operation | CCCL call | Why not CuPy |
|---|---|---|
| Key-value radix sort | `radix_sort` | CuPy `sort` is comparison-based, no key-value variant |
| Key-value merge sort | `merge_sort` | CuPy has no stable sort with custom comparator on device |
| Unique-by-key | `unique_by_key` | CuPy `unique` doesn't carry associated values |

**High-value additions for polygon expansion:**

| Operation | CCCL call | Use case |
|---|---|---|
| Segmented sort | `segmented_sort` | Sort ring coordinates within per-polygon offset spans; order segments within parts; sort half-edges by angle within per-node adjacency lists |
| Segmented reduce | `segmented_reduce` | Per-polygon area, per-ring winding number, per-group bounds in dissolve, per-row vertex count |
| Binary search (lower) | `lower_bound` | Spatial index: find first candidate in sorted Morton key array; offset lookup: map flat coordinate index → row index via offset array binary search |
| Binary search (upper) | `upper_bound` | Same family as lower_bound; find range ends in sorted arrays |
| Three-way partition | `three_way_partition` | Family-tag partitioning: split mixed-geometry columns into point/line/polygon groups in one pass instead of three separate `flatnonzero` calls |

**Why CCCL wins here:** These are genuinely different algorithms, not
just alternative implementations of the same primitive. CuPy doesn't
expose segmented sort, segmented reduce, or key-value radix sort at
all. Doing `segmented_reduce` via CuPy requires `cumsum` + fancy
indexing + `reduceat` simulation — multiple kernels and intermediate
arrays vs one fused CCCL call.

**Polygon-family note:** This is where the biggest shift happens.
Point-centric pipelines rarely need segmented operations because each
row is a single coordinate pair. Polygon pipelines operate on
variable-length structures (rings within parts within rows) constantly.
Every per-ring, per-part, or per-row reduction or sort over
offset-delimited groups is a natural fit for `segmented_reduce` or
`segmented_sort`.

### Tier 3b: CCCL `make_*` reusable callables (amortized JIT)

**Use for:** Any CCCL algorithm called repeatedly with the same types
and operators within a pipeline or across pipeline invocations.

Every `cuda.compute.algorithms` function has a `make_*` variant:

```python
# One-shot (pays JIT on every cold call):
segmented_reduce(d_in, d_out, starts, ends, op, h_init, n_segments)

# Reusable (pays JIT once, reuse across calls):
reducer = make_segmented_reduce(d_in, d_out, starts, ends, op, h_init)
reducer(num_segments=n1)  # first call: JIT + execute
reducer(num_segments=n2)  # subsequent: execute only
```

**This partially addresses the cold-call JIT problem.** For algorithms
where CuPy currently wins solely due to JIT overhead (scan, select,
reduce), creating `make_*` objects at module import time or pipeline
construction time could close the gap.

**Action:** When wrapping a new CCCL primitive in `cccl_primitives.py`,
also expose a `make_*` factory that returns a reusable callable.
Benchmark the reusable variant against CuPy before changing AUTO
defaults.

### Tier 3c: CCCL iterators (zero-allocation lazy evaluation)

**Use for:** Eliminating intermediate array allocations in algorithm
pipelines.

| Iterator | Replaces | Memory saved |
|---|---|---|
| `CountingIterator(start)` | `cp.arange(n, dtype=...)` | One full array allocation |
| `TransformIterator(array, func)` | `func(array)` materialized | One full array allocation |
| `ZipIterator(x, y)` | Interleaving or struct packing | Avoids copy |
| `TransformOutputIterator(array, func)` | `array[:] = func(result)` | Avoids temporary |

**Impact for polygon pipelines:** Polygon overlay and dissolve produce
large intermediate arrays (segment coordinates, split event buffers,
half-edge adjacency). Feeding iterators instead of materialized arrays
into CCCL algorithms can reduce peak device memory usage significantly.

**Example — current vs iterator-based:**

```python
# Current: materializes arange, then sorts key-value pairs
event_indices = cp.arange(int(all_keys.size), dtype=cp.int32)
sorted_pairs = sort_pairs(all_keys, event_indices)

# Iterator: no arange allocation
from cuda.compute import CountingIterator
counting = CountingIterator(np.int32(0))
# pass counting as values to radix_sort — no materialized array
```

**Action:** When adding new CCCL algorithm calls, check if any input
can be replaced with an iterator. Prioritize `CountingIterator` (most
common case — we use `cp.arange` extensively).

### Tier 4: Remaining CuPy-default primitives (flip when benchmarks justify)

**Current state:** Compaction and exclusive scan have been promoted to
CCCL default based on benchmarks (2026-03-12). The remaining overlapping
primitives still default to CuPy:

| Operation | CuPy default | Why not yet flipped |
|---|---|---|
| `reduce_into` | `cp.sum` | CuPy is marginally faster at small scales; difference too small to justify the switch |
| `inclusive_scan` | `cp.cumsum` | Not yet benchmarked with `make_*`; likely same outcome as exclusive scan |
| `unary_transform` | `cp.where` / element-wise | No algorithmic advantage from CCCL |
| `binary_transform` | element-wise ops | No algorithmic advantage from CCCL |

**When to promote:**
- When `make_*` benchmarks show a clear CCCL win (as happened for
  scan and compaction)
- When combining with CCCL iterators (Tier 3c) enables fusion that
  CuPy can't express (e.g., `reduce_into` with `TransformIterator`
  fuses transform + reduce into one kernel pass)
- When CCCL Python ships kernel caching that persists across process
  restarts

The strategy enum pattern in `cccl_primitives.py` makes it trivial to
flip the AUTO default per-primitive when benchmarks justify it.

## Decision procedure for new kernels

When implementing a new GPU operation, follow this decision tree:

```
Is the inner loop geometry-specific?
  → Yes: Tier 1 (custom NVRTC kernel)
  → No: Is it a segmented operation (per-row, per-ring, per-group)?
    → Yes: Tier 3a (CCCL segmented_sort / segmented_reduce)
    → No: Is it a sort, unique, search, partition, compaction, or scan?
      → Yes: Tier 3a (CCCL — default for all of these)
      → No: Is it element-wise, gather/scatter, or concat?
        → Yes: Tier 2 (CuPy built-in)
        → No: Tier 1 (custom NVRTC kernel)

Then, for any CCCL call:
  - Can an input be replaced with CountingIterator or TransformIterator?
    → Yes: Use iterator (Tier 3c) to avoid allocation
  - Will this algorithm be called repeatedly with the same types?
    → Yes: Use make_* variant (Tier 3b) to amortize JIT
```

## Concrete polygon-expansion playbook

These are the operations we expect to implement as we expand GPU
coverage to polygon-family geometry types. Each maps to a tier:

### Polygon-polygon binary predicates (DE-9IM)

| Stage | Tier | Primitive |
|---|---|---|
| Segment extraction from rings | Tier 1 | Custom NVRTC kernel (offset indirection) |
| Pairwise segment MBR filter | Tier 1 | Custom NVRTC kernel (2D overlap test) |
| Segment intersection classification | Tier 1 | Existing `classify_segment_pairs` kernel |
| Candidate compaction | Tier 3a | CCCL `select` (default) |
| DE-9IM matrix assembly per geometry pair | Tier 3a | `segmented_reduce` over intersection results grouped by geometry pair |
| Predicate resolution from DE-9IM bits | Tier 2 | CuPy element-wise bitwise ops |

### Polygon overlay (intersection, union, difference)

| Stage | Tier | Primitive |
|---|---|---|
| Split event sort by segment + t | Tier 3a | `radix_sort` (already used) |
| Split event deduplication | Tier 3a | `unique_by_key` (already used) |
| Half-edge angle sort within node | Tier 3a | **`segmented_sort`** (new — replaces current host sort) |
| Face area computation per face | Tier 3a | **`segmented_reduce`** (new — sum of cross products within face ring) |
| Face containment test (point-in-polygon) | Tier 1 | Existing GPU point-in-polygon kernel |
| Output polygon assembly | Tier 1 + 2 | Custom kernel for ring gathering + CuPy for offset computation |

### Dissolve (groupby + union)

| Stage | Tier | Primitive |
|---|---|---|
| Group key encoding | Tier 2 | CuPy fancy indexing |
| Group key sort | Tier 3a | `radix_sort` (already used) |
| Group boundary detection | Tier 3a | `unique_by_key` or CuPy diff |
| Per-group bounds reduction | Tier 3a | **`segmented_reduce`** (new — min/max bounds per group) |
| Per-group polygon union | Tier 1 | Custom kernel or Shapely fallback (topologically complex) |

### Spatial index lookup

| Stage | Tier | Primitive |
|---|---|---|
| Morton key computation | Tier 1 | Existing NVRTC kernel |
| Morton key sort | Tier 3a | `radix_sort` (already used) |
| Query point → candidate range | Tier 3a | **`lower_bound` + `upper_bound`** (new — binary search in sorted keys) |
| Candidate refinement | Tier 1 | Custom NVRTC kernel |

### Mixed-geometry column partitioning

| Stage | Tier | Primitive |
|---|---|---|
| Family-tag extraction | Tier 2 | CuPy element-wise |
| Three-way split (point/line/polygon) | Tier 3a | **`three_way_partition`** (new — one pass instead of three `flatnonzero`) |
| Per-family dispatch | Application logic | Route to family-specific kernels |

## What NOT to change

Custom NVRTC kernels remain the right choice for geometry-specific
inner loops regardless of what CCCL adds. The geometry domain has too
much structural specificity (offset indirection, ring orientation,
winding rules) for generic parallel primitives to express efficiently.

CuPy element-wise ops (fancy indexing, `cp.where`, `cp.concatenate`)
should stay as-is — CCCL offers no algorithmic advantage here. Only
flip remaining CuPy defaults (reduction, inclusive scan) when `make_*`
benchmarks show a clear win.

## Risks

- **CCCL Python API stability.** The `cuda.compute.algorithms` API is
  marked unstable. The strategy enum pattern and `cccl_primitives.py`
  wrapper layer insulate the codebase from API changes.
- **CuPy version coupling.** CuPy's kernel pool and fusion behavior
  can change between versions. Pin CuPy in CI and test against the
  specific version before upgrading.
- **Segmented primitive cold-call cost.** `segmented_sort` and
  `segmented_reduce` will have JIT overhead on first invocation. Use
  `make_*` variants for any segmented primitive called in a pipeline
  that runs more than once.
- **Iterator compatibility.** Not all CCCL algorithms may accept all
  iterator types. Test iterator inputs against each algorithm before
  committing to an iterator-based pattern.

## Verification

The rules are embedded in `cccl_primitives.py` via the AUTO strategy
defaults. To verify:

```bash
uv run pytest tests/test_cccl_primitives.py
uv run pytest tests/test_gpu_bounds.py tests/test_gpu_point_in_polygon.py
uv run pytest tests/test_gpu_constructive.py tests/test_gpu_overlay_intersection.py
```

To benchmark CuPy vs CCCL on a specific primitive at a specific scale:

```python
from vibespatial.cuda.cccl_primitives import compact_indices, CompactionStrategy
# Force CCCL path:
result = compact_indices(mask, strategy=CompactionStrategy.CCCL_SELECT)
# Force CuPy path:
result = compact_indices(mask, strategy=CompactionStrategy.CUPY)
```
