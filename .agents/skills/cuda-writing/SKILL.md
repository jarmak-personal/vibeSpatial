---
name: cuda-writing
description: "PROACTIVELY USE THIS SKILL when writing, modifying, or reviewing GPU kernels, CUDA/NVRTC kernel source, CCCL primitive usage, device memory management, stream-based pipelining, or any GPU dispatch logic in src/vibespatial/. Use it for ADR-0046 physical workload shape contracts, ADR-0044 native carriers (NativeFrameState, NativeRowSet, NativeRelation, NativeGrouped, NativeSpatialIndex, NativeGeometryMetadata, NativeExpression), shape-level work estimates, kernel lifecycle, ADR-0033 tier selection, count/scan/scatter pipelines, precompilation (ADR-0034), precision dispatch (ADR-0002), and GPU saturation."
---

# GPU Kernel Development Guide — vibeSpatial

You are writing GPU code for vibeSpatial. Follow these rules strictly.
Physical workload shape is governed by ADR-0046. Native carriers are governed
by ADR-0044. GPU primitive dispatch decisions are governed by ADR-0033.
Precision dispatch is governed by ADR-0002. Precompilation is governed by
ADR-0034.

## 0. ADR-0046 Physical Shape Gate

Before writing GPU code, declare the physical workload shape. Do this before
choosing CuPy, CCCL, NVRTC, launch thresholds, or public dispatch wiring.

Record:

- public semantics and admissibility boundary
- physical shape family: aligned pairwise, broadcast, matrix, candidate-refine,
  rowset take, relation consume, segmented grouped reduction, dynamic-output
  assembly, terminal native export, or another named reusable shape
- work units: rows, coordinates, vertices, segments, rings, tiles, candidate
  pairs, relation pairs, groups, output rows, output bytes, temporary bytes
- native input and output carriers: `NativeFrameState`, `NativeRowSet`,
  `NativeRelation`, `NativeGrouped`, `NativeSpatialIndex`,
  `NativeGeometryMetadata`, `NativeExpression`, or owned geometry/scalar arrays
- temporary execution layout: CSR/COO pairs, dense tiles, sorted partitions,
  grouped offsets, primitive work queues, transposed coordinate views, centered
  fp32 buffers, or scratch arenas
- saturation plan for large-single, many-small, sparse, dense, and skewed cases
- precision plan, transfer/sync/export boundary, and shape canary or benchmark

If the public result is row-aligned, do not assume row-shaped execution. The
right GPU shape may be vertex, segment, ring, tile, candidate-pair, group,
relation, or output-byte shaped before reducing back to public rows.

If no reusable physical shape can be stated, stop and draft the shape contract
or ADR amendment first. Do not hide a weak shape behind a polished kernel.

## 1. ADR-0033 Tier Decision Tree

After the ADR-0046 shape gate, classify the implementation primitive for that
shape:

```
Is the inner loop geometry-specific (ring traversal, winding, segment intersection)?
  -> Yes: Tier 1 (custom NVRTC kernel)
  -> No: Is it segmented (per-row/ring/group reduction, sort, scan)?
    -> Yes: Tier 3a (CCCL segmented_*)
    -> No: Is it sort/unique/search/partition/compaction/scan?
      -> Yes: Tier 3a (CCCL)
      -> No: Is it element-wise/gather/scatter/concat?
        -> Yes: Tier 2 (CuPy)
        -> No: Tier 1 (custom NVRTC kernel)
```

Then for CCCL (Tier 3a): Can input use `CountingIterator`/`TransformIterator`?
Use Tier 3c iterator. Called repeatedly with same types/ops? Use Tier 3b `make_*`.

### Tier 1 — Custom NVRTC Kernels

Geometry-specific inner loops: point-in-polygon winding, segment
intersection, split events, half-edge traversal, WKB extraction.
Compiled via `runtime.compile_kernels()`, launched via `runtime.launch()`.
Cached in `CudaDriverRuntime._module_cache` by SHA1 of source.
Polygon work uses MORE Tier 1, not less (ring traversal, part offset
indirection, winding across holes).

### Tier 2 — CuPy Built-Ins (Default for Element-Wise)

Element-wise transforms, boolean masking, gather/scatter, concat.
Specific operations: `cp.where`, `cp.sum`, fancy indexing, `cp.concatenate`.
Why: zero JIT path; simpler than CCCL for no algorithmic advantage.

**CuPy remains the AUTO default for** (ADR-0033 Tier 4):
- `reduce_into` (`cp.sum`) — marginal difference; promote when `make_*` benchmarks justify
- `inclusive_scan` (`cp.cumsum`) — same rationale

### Tier 3a — CCCL Algorithmic Primitives

**Promoted to CCCL default** (benchmarked 2026-03-12):
- `exclusive_scan` — 1.8-3.7x faster than CuPy via `make_*`

**Reverted to CuPy default** (2026-03-17):
- `select` (compaction by bool mask) — CCCL `make_select` bakes predicate
  closure device pointers, preventing reuse across calls. One-shot `select()`
  re-JITs per array size class (~5-6s each). CuPy `flatnonzero` is 0.2ms
  with no JIT. CCCL available via explicit `CompactionStrategy.CCCL_SELECT`.

**Already CCCL-only** (no CuPy equivalent):
- `radix_sort`, `merge_sort` with custom comparators
- `unique_by_key`, `segmented_reduce`, `lower_bound`, `upper_bound`
- `segmented_sort`, `three_way_partition`

**High-value for polygon expansion:**
- `segmented_sort` — sort within offset spans (angle-sort half-edges)
- `segmented_reduce` — per-polygon area, winding, bounds, vertex count
- `three_way_partition` — split mixed point/line/polygon in one pass

### Tier 3b — CCCL `make_*` Reusable Callables

For algorithms called repeatedly with the same types and operators.
`make_*` objects pre-allocate temp storage and eliminate both the
cold-call JIT check (~950-1460ms first call) and per-call temp-storage
overhead.

```python
# One-shot API (cold-call pays full JIT):
cccl_algorithms.exclusive_scan(values, out, _sum_op, init, n)

# make_* API (pre-compiled, reusable):
scanner = cccl_algorithms.make_exclusive_scan(None, values, out, _sum_op, n, init)
temp = cp.empty(scanner_temp_bytes, dtype=cp.uint8)
scanner(temp, values, out, _sum_op, n, init)  # no JIT check
```

**Rule:** Wrap new CCCL primitives in `make_*` factory. Benchmark
before changing the AUTO default from CuPy.

### Tier 3c — CCCL Iterators (Zero-Allocation)

- `CountingIterator(start)` — replaces `cp.arange` allocations
- `TransformIterator(input, op)` — fuses element-wise transforms with algorithms
- `ZipIterator(iter_a, iter_b)` — avoids struct-of-arrays shuffling

Impact: reduces peak device memory for large intermediates.

```python
from vibespatial.cuda.cccl_primitives import counting_iterator, transform_iterator

# Instead of: indices = cp.arange(n, dtype=cp.int32)
indices = counting_iterator(0, dtype=np.int32)

# Fuse a transform with a reduction (no intermediate buffer):
squared = transform_iterator(d_values, lambda x: x * x)
```

### Strategy Enum Pattern (API Stability)

The `cuda.compute.algorithms` API is marked unstable. Each CCCL
primitive in `cccl_primitives.py` is wrapped behind a strategy enum
(e.g., `CompactionStrategy`, `ScanStrategy`, `PairSortStrategy`) that
makes it trivial to flip the AUTO default per-primitive when benchmarks
justify it, insulating the codebase from upstream API changes.

```python
# Callers use strategy enums, never raw CCCL calls:
result = compact_indices(mask, strategy=CompactionStrategy.AUTO)
offsets = exclusive_sum(counts, strategy=ScanStrategy.AUTO)
```

---

## 2. Kernel Launch Lifecycle (Tier 1)

Every NVRTC kernel follows this exact sequence:

```python
from vibespatial.cuda._runtime import (
    get_cuda_runtime, make_kernel_cache_key,
    KERNEL_PARAM_PTR, KERNEL_PARAM_I32, KERNEL_PARAM_F64,
)

# 1. Get runtime singleton
runtime = get_cuda_runtime()

# 2. Compile (cached via SHA1 of source)
cache_key = make_kernel_cache_key("my-kernel", _KERNEL_SOURCE)
kernels = runtime.compile_kernels(
    cache_key=cache_key, source=_KERNEL_SOURCE,
    kernel_names=("my_kernel",),
)

# 3. Parameters as (values_tuple, types_tuple)
ptr = runtime.pointer
params = (
    (ptr(d_input), ptr(d_output), width, height, some_float),
    (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32, KERNEL_PARAM_I32, KERNEL_PARAM_F64),
)

# 4. Occupancy-based launch config (NEVER hardcode block=(256,1,1))
grid, block = runtime.launch_config(kernels["my_kernel"], item_count)

# 5. Launch
runtime.launch(kernels["my_kernel"], grid=grid, block=block, params=params)
```

### Parameter Type Constants
```python
KERNEL_PARAM_PTR = ctypes.c_void_p    # Device memory pointers
KERNEL_PARAM_I32 = ctypes.c_int       # 32-bit integers
KERNEL_PARAM_F64 = ctypes.c_double    # 64-bit floats
```

**CRITICAL:** Always use the named constants from `vibespatial.cuda._runtime`.

### Occupancy-Based Block Sizing

**NEVER** hardcode `block=(256, 1, 1)`. Always use:

```python
grid, block = runtime.launch_config(kernel, item_count)
# Or: block_size = runtime.optimal_block_size(kernel, shared_mem_bytes=0)
```

Uses `cuOccupancyMaxPotentialBlockSize` for optimal threads-per-block
based on register pressure and shared memory. Falls back to 256 if
the API is unavailable. Results cached per kernel function.

**Note**: The occupancy API maximizes *occupancy*, not necessarily
*performance*. For kernels with high register usage, lower occupancy
with more registers per thread can outperform higher occupancy due to
increased ILP (instruction-level parallelism). Profile before overriding.

### `__launch_bounds__` Directive

Use `__launch_bounds__` in NVRTC kernel source to guide register
allocation and ensure minimum blocks per SM:

```c
extern "C" __global__ void __launch_bounds__(256, 4)
my_kernel(const double* __restrict__ x, ...) {{
    // 256 max threads/block, at least 4 blocks per SM
}}
```

Without `__launch_bounds__`, the compiler assumes the maximum possible
threads per block, which can over-estimate shared memory needs and
reduce occupancy. Always use when the block size is known at compile time.

---

## 3. Device Memory Management

### Allocation

```python
# WITHOUT zero-fill (default — use when kernel writes every element)
d_output = runtime.allocate((n,), np.float64)

# WITH zero-fill (use for count arrays, sparse-update targets)
d_counts = runtime.allocate((n,), np.int32, zero=True)
```

### Transfers

```python
d_data = runtime.from_host(host_array)              # H2D
host_result = runtime.copy_device_to_host(d_output)  # D2H
ptr = runtime.pointer(d_array)                       # device pointer (int), 0 for None
```

### Memory Pool

CuPy MemoryPool is configured at runtime init. Freed allocations are
cached for reuse. `VIBESPATIAL_GPU_POOL_LIMIT` env var caps pool size.

```python
stats = runtime.memory_pool_stats()  # {"used_bytes", "total_bytes", "free_bytes"}
runtime.free_pool_memory()           # release cached memory
```

---

## 4. CUDA Streams — When and How

### When to Use Streams

Streams let independent GPU operations run concurrently. Use them when:

1. **Independent H2D uploads** — uploading query and tree geometry
   simultaneously (different source arrays, no dependency).
2. **Independent kernel launches** — polygon bounds and multipolygon
   bounds kernels that operate on different family buffers.
3. **Overlapping D2H transfer with compute** — start transferring
   results from pass 1 while pass 2 runs on a different output buffer.
4. **Count-scatter total computation** — batch two last-element reads
   into one async transfer instead of two sequential `.get()` syncs.

### When NOT to Use Streams

- **Sequential data dependencies** — kernel B reads what kernel A wrote.
  Same-stream ordering handles this automatically.
- **Tiny transfers** — overhead of stream creation (~1-2us) exceeds
  the transfer time.
- **Single-kernel operations** — no independent work to overlap.

### Stream API

```python
# Manual create/destroy
stream = runtime.create_stream()
runtime.launch(kernel, grid=g, block=b, params=p, stream=stream)
stream.synchronize()
runtime.destroy_stream(stream)

# Context manager (auto-sync + cleanup)
with runtime.stream_context() as stream:
    runtime.launch(kernel, grid=g, block=b, params=p, stream=stream)
```

### Async Transfers

```python
# Pinned host memory enables true async DMA
h_buf = runtime.allocate_pinned((n,), np.int32)

# Async D2H (enqueue only — sync stream before reading)
runtime.copy_device_to_host_async(d_array, stream, h_buf)

# Async H2D
runtime.copy_host_to_device_async(h_array, d_array, stream)
```

### Pattern: Count-Scatter Total

```python
from vibespatial.cuda._runtime import count_scatter_total, count_scatter_total_with_transfer

# Simple: single-sync async pinned transfer (replaces 2x .get())
total = count_scatter_total(runtime, device_counts, device_offsets)

# Advanced: also starts full counts D2H on background stream
total, xfer_stream, pinned_counts = count_scatter_total_with_transfer(
    runtime, device_counts, device_offsets,
)
# ... launch scatter kernel on null stream ...
runtime.synchronize()
xfer_stream.synchronize()  # counts already transferred
runtime.destroy_stream(xfer_stream)
host_counts = pinned_counts
```

### Pattern: Independent Uploads

```python
stream_q = runtime.create_stream()
stream_t = runtime.create_stream()
d_query = runtime.allocate(query_bounds.shape, query_bounds.dtype)
d_tree = runtime.allocate(tree_bounds.shape, tree_bounds.dtype)
runtime.copy_host_to_device_async(query_bounds, d_query, stream_q)
runtime.copy_host_to_device_async(tree_bounds, d_tree, stream_t)
stream_q.synchronize()
stream_t.synchronize()
runtime.destroy_stream(stream_q)
runtime.destroy_stream(stream_t)
```

### Pattern: Independent Family Kernels

```python
with runtime.stream_context() as s_poly:
    runtime.launch(bounds_polygon_kernel, ..., stream=s_poly)
with runtime.stream_context() as s_mpoly:
    runtime.launch(bounds_multipolygon_kernel, ..., stream=s_mpoly)
```

### Synchronization Rules

- `runtime.synchronize()` syncs the CUDA context (all streams). Use
  sparingly — only before host reads of device data.
- `stream.synchronize()` syncs one stream only. Prefer this.
- CUDA guarantees execution order within a single stream. Do NOT sync
  between consecutive launches on the same stream.
- Remove sync calls between: kernel -> CCCL, kernel -> kernel, CCCL ->
  CCCL (all on same null stream).
- Keep sync calls before: `copy_device_to_host`, `.get()`, `cp.asnumpy`.

---

## 5. CCCL Primitives

### Available in cccl_primitives.py

```python
from vibespatial.cuda.cccl_primitives import (
    # Tier 3a — CCCL default (beats CuPy)
    exclusive_sum,            # Prefix sum (1.8-3.7x faster)
    compact_indices,          # Bool mask -> indices (CuPy default; CCCL via explicit strategy)
    sort_pairs,               # Radix or merge sort with values
    unique_sorted_pairs,      # Unique-by-key on sorted input
    segmented_reduce_sum,     # Per-segment sum
    segmented_reduce_min,     # Per-segment min
    segmented_reduce_max,     # Per-segment max
    segmented_sort,           # Sort within offset-delimited segments
    lower_bound,              # Binary search (first insertion point)
    upper_bound,              # Binary search (last insertion point)
    three_way_partition,      # Split into 3 groups by predicates

    # Tier 3c — Zero-allocation iterators
    counting_iterator,        # Lazy [0,1,2,...] (replaces cp.arange)
    transform_iterator,       # Fused element-wise transform

    # Tier 4 — Still CuPy-default (CCCL available but marginal win)
    reduce_sum,               # Scalar reduction (CuPy cp.sum faster at small scale)
)
```

### Synchronize Parameter

Wrappers that return device arrays accept `synchronize=False` to skip
the internal stream sync when the result feeds into the next GPU op:

```python
offsets = exclusive_sum(counts, synchronize=False)
sorted_result = sort_pairs(keys, values, synchronize=False)
lb = lower_bound(sorted_data, queries, synchronize=False)
```

Do NOT pass `synchronize=False` to `compact_indices` or
`unique_sorted_pairs` — they internally call `.get()` (requires sync).

---

## 6. Precompilation and Warmup (ADR-0034)

### The Problem

Cold-call JIT penalty: ~950ms for `select`, ~1460ms for `exclusive_scan`
(one-time per process per unique spec). NVRTC kernels: ~20-400ms per
compilation unit.

### Three-Level Demand-Driven Strategy

**Level 1 — CCCL Module Warmup:** Each consumer module declares specs at
module scope via `request_warmup()`. A singleton `CCCLPrecompiler` with
a background `ThreadPoolExecutor` (8 threads) compiles specs in parallel.
Never blocks import; first real call may block up to 5s waiting for its
needed spec.

```python
from vibespatial.cuda.cccl_precompile import request_warmup
request_warmup(["exclusive_scan_i32", "exclusive_scan_i64", "select_i32"])
```

**Level 2 — NVRTC Module Warmup:** Same pattern for NVRTC kernels.
14 kernel compilation units, ~4,785 lines CUDA C, 85 entry points.
Thread-safe via `_module_cache_lock` on `CudaDriverRuntime`.

```python
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
request_nvrtc_warmup([
    ("my-kernel", _KERNEL_SOURCE, _KERNEL_NAMES),
])
```

**Level 3 — Pipeline-Aware (deferred):** When pipeline plans are
constructed (sjoin, overlay), declare all requirements and block until
compiled.

### Key Properties

- `request_warmup()` is idempotent — duplicate specs are skipped.
- All compilation releases the GIL (Cython `with nogil:` in CCCL
  bindings, NVRTC handles are independent).
- Numba operator JIT holds GIL (~50-100ms per unique op, ~6 ops total,
  `lru_cache`'d after first call).

### Thread Scaling (18 CCCL specs)

| Threads | Wall Time | Speedup |
|---------|-----------|---------|
| 1       | ~18-22s   | 1x      |
| 4       | ~4-5s     | 4x      |
| 8       | ~2-3s     | 7x (recommended for CCCL) |
| 16      | ~1.5s     | 12x (recommended for NVRTC) |

### Cost by Operation

| Operation           | CCCL Specs | NVRTC Units | Est. Cold Cost |
|---------------------|------------|-------------|----------------|
| `gs.within(other)`  | 2          | 1           | ~1s            |
| `gs.to_wkb()`       | 2          | 0           | ~1s            |
| `gpd.sjoin(a, b)`   | 12         | 2-3         | ~2s            |
| `gpd.overlay(a, b)` | 6          | 2           | ~1.5s          |
| `gs.dissolve(by=)`  | 3          | 0           | ~1s            |

### Observability

```python
import vibespatial
status = vibespatial.precompile_status()  # dict of spec -> compiled/pending/failed
```

---

## 7. Warp-Level Intrinsics

Use in NVRTC kernel source strings for intra-warp coordination.

### Warp Ballot — Early Exit

Skip expensive work when no thread in the warp needs it:

```c
const bool valid = row < row_count;
const unsigned char is_candidate = valid ? candidate_mask[row] : 0;
if (__ballot_sync(0xFFFFFFFF, is_candidate) == 0) {{
    return;  // entire warp skips all global memory reads
}}
if (!valid || !is_candidate) return;
```

After bounds filtering at ~5% hit rate, most warps skip entirely.

### `__syncwarp()` — Warp Reconvergence (Volta+)

On Volta+ (CC 7.0+) with independent thread scheduling, warps can
remain diverged after conditional blocks. Use `__syncwarp()` before
warp-level primitives to guarantee all lanes are converged:

```c
if (condition) {{
    // ... divergent work ...
}}
__syncwarp(0xFFFFFFFF);  // reconverge before shuffle/ballot
```

### Warp Shuffle — Intra-Warp Reduction

Reduce across 32 lanes without shared memory:

```c
const unsigned int FULL_MASK = 0xFFFFFFFF;
for (int offset = 16; offset > 0; offset >>= 1) {{
    my_crossings ^= __shfl_xor_sync(FULL_MASK, my_crossings, offset);
    my_boundary  |= __shfl_xor_sync(FULL_MASK, my_boundary, offset);
}}
```

### Block-Level Reduction (via shared memory)

```c
__shared__ int warp_results[8];  // up to 256 threads = 8 warps
const int warp_id = threadIdx.x / 32;
const int lane_id = threadIdx.x % 32;

// 1. Warp-level shuffle reduction
for (int offset = 16; offset > 0; offset >>= 1)
    my_value ^= __shfl_xor_sync(0xFFFFFFFF, my_value, offset);

// 2. Lane 0 writes to shared memory
if (lane_id == 0) warp_results[warp_id] = my_value;
__syncthreads();

// 3. Thread 0 reduces across warps
if (threadIdx.x == 0) {{
    int total = 0;
    for (int w = 0; w < num_warps; ++w) total ^= warp_results[w];
    output[blockIdx.x] = total;
}}
```

---

## 8. Two-Pass Count-Scatter Pattern

Variable-output kernels use a two-pass approach:

```
Pass 0 (count):   Each thread computes output size -> counts[tid]
Prefix sum:        exclusive_sum(counts) -> offsets[tid]
Get total:         count_scatter_total(runtime, counts, offsets)
Allocate output:   runtime.allocate((total,), dtype)
Pass 1 (scatter):  Each thread writes output at offsets[tid]
```

### Best Practices

- Use `count_scatter_total()` instead of sequential `.get()` calls.
- Build full offsets on device when possible (avoid D2H -> modify -> H2D).
- Do NOT sync between count kernel -> exclusive_sum -> scatter kernel
  (same stream ordering handles it).
- Sync only before reading total on host.

---

## 9. Shared Memory

### When to Use

- **Cooperative edge iteration** — all threads process edges of the
  same ring, accumulate in shared memory.
- **Stencil/neighborhood** — tile data with halo cells.
- **Block-level reduction** — inter-warp reduction via shared arrays.

### Cooperative Intra-Ring PIP

Instead of one-thread-per-ring (load imbalanced), all threads split
edges within each ring:

```c
for (int ring = ring_start; ring < ring_end; ++ring) {{
    int edge_count = ring_offsets[ring + 1] - ring_offsets[ring] - 1;
    for (int e = threadIdx.x; e < edge_count; e += blockDim.x) {{
        // ... even-odd test on edge e ...
    }}
    // Warp shuffle + shared memory reduction for this ring
}}
```

### Bank Conflict Avoidance

Shared memory has 32 banks, each 4 bytes wide. If multiple threads in
a warp access different addresses in the same bank, accesses serialize.

**The +1 padding trick** eliminates bank conflicts on column access:

```c
// BAD: 32-way bank conflict when reading columns
__shared__ float tile[32][32];

// GOOD: +1 padding shifts bank mapping — no conflicts
__shared__ float tile[32][33];
```

For `double` (8 bytes = 2 banks), stride-1 access has inherent 2-way
conflict. Mitigate with `+1` padding per row or swizzled indexing.

NVIDIA benchmark: padding improved matrix transpose bandwidth from
51.3 GB/s to 99.5 GB/s (~2x).

### Declaring Shared Memory

```python
runtime.launch(kernel, grid=g, block=b, params=p, shared_mem_bytes=1024)
```

```c
__shared__ int scratch[256];                  // static
extern __shared__ float dynamic_smem[];       // dynamic (via shared_mem_bytes)
```

**Max shared memory per block** (default is 48 KB; requesting more
requires `cudaFuncSetAttribute` with `cudaFuncAttributeMaxDynamicSharedMemorySize`):
- A100: 163 KB
- H100: 227 KB
- RTX 3090/4090: 99 KB

### Async Copy to Shared Memory (CUDA 11.0+)

For data tiling, `__pipeline_memcpy_async()` copies directly from global
to shared memory, bypassing the register file. This reduces register
pressure and can increase occupancy. Best with 8 or 16-byte elements:

```c
__pipeline_memcpy_async(&shared_data[tid], &global_data[idx], sizeof(double));
__pipeline_commit();
__pipeline_wait_prior(0);  // wait for all committed copies
__syncthreads();
```

---

## 10. Precision Dispatch (ADR-0002)

### Policy by Kernel Class

| Kernel Class | Consumer GPU (fp64:fp32 < 0.25) | Datacenter GPU (fp64:fp32 >= 0.25) |
|-------------|-------------------------------|-----------------------------------|
| **COARSE** (bounds, index, filter) | Staged fp32 with coordinate centering | Native fp64 |
| **METRIC** (distance, area, length) | Staged fp32 with Kahan compensation | Native fp64 |
| **PREDICATE** (PIP, binary preds) | Staged fp32 coarse pass + selective fp64 refinement for ambiguous rows | Native fp64 |
| **CONSTRUCTIVE** (clip, overlay, buffer) | Native fp64 (until robustness work proves cheaper path) | Native fp64 |

### How It Works

1. `select_precision_plan()` reads the device's fp64:fp32 throughput
   ratio (via `CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO`)
   and returns a `PrecisionPlan` with `compute_precision`, `compensation`,
   `center_coordinates`, and `refinement` fields.

2. `auto` mode chooses based on device profile AND kernel class — consumer
   GPUs get fp32 for coarse/metric/predicate; datacenter GPUs get fp64.

3. The plan is passed to the kernel launcher, which formats the CUDA
   source template accordingly.

### Implementation Checklist

- Template kernel source: `typedef {compute_type} compute_t;`
- Storage reads stay `double`: `compute_t lx = (compute_t)(x[i] - center_x);`
- Cache key includes precision: `f"my-kernel-{compute_type}"`
- Coordinate centering when `plan.center_coordinates is True`
- Kahan compensation for metric accumulations when `plan.compensation is KAHAN`
- CONSTRUCTIVE kernels: wire plan for observability but stay fp64

See `precision-compliance` skill for the full step-by-step procedure.

---

## 11. Performance Rules

### Memory Access

- **Coalesced reads**: Adjacent threads read adjacent addresses. SoA
  layout (separate x[], y[]) is already coalesced.
- **Avoid AoS**: Never interleave x,y in a single array. NVIDIA GTC 2024
  benchmark: AoS was **5.9x slower** than SoA due to strided access.
- **Minimize global writes**: Use shared memory or registers for
  intermediates; write to global once.
- **`const __restrict__`**: Always annotate read-only pointer parameters.
  On CC 3.5+, the compiler automatically routes through the read-only
  data cache (`__ldg` path), increasing effective cache capacity:
  ```c
  extern "C" __global__ void my_kernel(
      const double* __restrict__ x,   // read-only -> __ldg cache
      const double* __restrict__ y,
      double* __restrict__ output,    // write-only
      ...
  ```
- **Vectorized loads** for bandwidth-bound bulk I/O. Use 128-bit wide
  loads to reduce instruction count and increase bandwidth (1.3-1.5x
  speedup per NVIDIA benchmarks):
  ```c
  // Instead of scalar loads:
  double val = input[idx];
  // Use 128-bit vectorized loads (requires aligned pointer):
  double2 vals = reinterpret_cast<const double2*>(input)[idx];
  // For fp32: float4 vals = reinterpret_cast<const float4*>(input)[idx];
  ```
  `cudaMalloc` guarantees 256-byte alignment, so device pointers are valid.
  Handle remainder elements with a scalar tail.

### Kernel Launch

- Never launch with <32 threads of real work (wastes a warp).
- Avoid many tiny kernels — each launch has ~3-5us overhead.
- Remove unnecessary syncs between same-stream operations.
- **Wave quantization**: On modern GPUs (128 SMs on RTX 4090, 132 on
  H100), launching grid_size = SM_count + 1 wastes ~50% of the final
  wave. Size grids to exactly fill the GPU and use grid-stride loops.

### Grid-Stride Loops with ILP

Prefer grid-stride loops over one-thread-per-element. Process multiple
elements per thread for instruction-level parallelism:

```c
// Grid-stride loop with ILP (4 elements/thread)
const int stride = blockDim.x * gridDim.x;
for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
     idx < n;
     idx += stride * 4) {{
    #pragma unroll
    for (int j = 0; j < 4; j++) {{
        int elem = idx + j * stride;
        if (elem < n) output[elem] = compute(input[elem]);
    }}
}}
```

NVIDIA GTC 2024 data: 4 elements/thread at 50% occupancy outperformed
1 element/thread at 100% occupancy (82% vs 51% bandwidth utilization).

**Grid sizing for grid-stride loops**:
```python
grid_size = min(
    (n + block_size - 1) // block_size,
    sm_count * max_blocks_per_sm  # fill GPU exactly — avoid wave waste
)
```

### Divergence and Load Balancing

- Avoid warp-divergent branches in inner loops.
- Use predicated writes: `output[idx] = valid ? result : nodata;`
- For variable-work-per-thread, use work-size binning:

```python
_WORK_BINS = [64, 1024]  # simple < 64 verts, medium 64-1024, complex > 1024

def _should_bin_dispatch(work_estimates):
    if len(work_estimates) < 1024: return False
    return work_estimates.std() / work_estimates.mean() > 2.0
```

For the "complex" bin, use a block-per-item kernel where all threads
cooperatively process one work item (see cooperative edge iteration).

### Arithmetic Micro-Optimizations

- **Float constant suffix**: In fp32 kernels, use `1.0f` not `1.0`.
  Without the `f` suffix, constants are double-precision and force a
  conversion instruction. When using `compute_t`, cast explicitly:
  `(compute_t)0.0` or `(compute_t)1e-7`.
- **Integer division/modulo**: Compiles to up to 20 instructions. When
  the divisor is a power of 2, use bitwise ops:
  `threadIdx.x / 32` -> `threadIdx.x >> 5`
  `threadIdx.x % 32` -> `threadIdx.x & 31`
- **`sinpi`/`cospi`**: If trig functions are needed (e.g., great circle
  distance), use `sinpi(x)` instead of `sin(x * M_PI)` — faster and
  more accurate.

### L2 Cache-Aware Patterns

For multi-kernel pipelines (count -> scatter, bounds -> filter -> refine),
consider these L2-friendly patterns:

- **Reverse block traversal**: Alternate block indexing direction between
  consecutive kernels so kernel B starts from data kernel A last touched:
  ```c
  // Kernel A: forward
  int blockId = blockIdx.x;
  // Kernel B: reverse (hits A's still-cached tail data)
  int blockId = gridDim.x - blockIdx.x - 1;
  ```
- **Cache tiling**: If intermediate data fits in L2 (40 MB A100, 72 MB
  RTX 4090, 50 MB H100), process data in L2-sized chunks end-to-end
  rather than running each stage over the full dataset. L2 cache hits
  can yield up to 10x bandwidth improvement over HBM misses.

---

## 12. Dispatcher Pattern

```python
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.precision import KernelClass
from vibespatial.runtime._runtime._runtime import ExecutionMode

selection = plan_dispatch_selection(
    kernel_name="my_kernel",
    kernel_class=KernelClass.COARSE,
    row_count=n,
    # Prefer shape-level estimates when the dispatch API supports them:
    # vertex_count=vertex_count,
    # segment_count=segment_count,
    # candidate_pair_count=candidate_pair_count,
    # output_byte_estimate=output_byte_estimate,
    requested_mode=dispatch_mode,
)
if selection.selected is ExecutionMode.GPU:
    return _my_kernel_gpu(...)
else:
    return _my_kernel_cpu(...)
```

---

## 13. Testing Pattern

```python
import pytest

requires_gpu = pytest.mark.skipif(not _has_gpu(), reason="GPU not available")

@requires_gpu
def test_my_kernel_gpu():
    """GPU produces same results as Shapely oracle."""
    gpu_result = my_op_gpu(data)
    cpu_result = shapely_reference(data)
    np.testing.assert_allclose(gpu_result, cpu_result, rtol=1e-10)

def test_my_op_auto_dispatch():
    """Auto-dispatch falls back to CPU gracefully."""
    result = my_op(data)
    assert result is not None
```
