---
name: gpu-code-review
description: >
  PROACTIVELY USE THIS SKILL when reviewing GPU kernel code, CUDA/NVRTC source,
  CuPy operations, CCCL primitive usage, device memory management, stream-based
  pipelining, or any GPU dispatch logic. This skill contains quantitative
  thresholds, anti-pattern detection rules, and architecture-specific guidance
  for A100, H100, RTX 3090, and RTX 4090 GPUs. Use it to catch performance
  regressions, memory management issues, synchronization bugs, and precision
  problems before they land.
user-invocable: true
argument-hint: <file-path or PR diff to review>
---

# GPU Code Review Reference — vibeSpatial

You are reviewing GPU code for vibeSpatial. Use this reference to identify
performance issues, anti-patterns, and correctness risks. Every finding should
cite a specific rule from this document.

This library uses cuda-python, CuPy, CCCL (CUDA C++ Core Libraries with Python
bindings), and NVRTC for runtime compilation. Target hardware: A100, H100
(datacenter) and RTX 3090, RTX 4090 (consumer).

---

## 0. Target Hardware Reference Card

Use these numbers to evaluate whether code is appropriately tuned.

| Spec | A100 (CC 8.0) | H100 (CC 9.0) | RTX 3090 (CC 8.6) | RTX 4090 (CC 8.9) |
|------|---------------|---------------|-------------------|-------------------|
| SMs | 108 | 132 | 82 | 128 |
| Registers per SM | 65,536 (32-bit) | 65,536 (32-bit) | 65,536 (32-bit) | 65,536 (32-bit) |
| Max registers/thread | 255 | 255 | 255 | 255 |
| Max threads per SM | 2,048 | 2,048 | 1,536 | 1,536 |
| Max blocks per SM | 32 | 32 | 16 | 16 |
| Shared mem per SM | 164 KB | 228 KB | 100 KB | 100 KB |
| Max shared mem/block | 163 KB | 227 KB | 99 KB | 99 KB |
| L1/Texture cache | 192 KB | 256 KB | 128 KB | 128 KB |
| L2 cache | 40 MB | 50 MB | 6 MB | 72 MB |
| Memory bandwidth | 1,555 GB/s (HBM2) | 3,350 GB/s (HBM3) | 936 GB/s (GDDR6X) | 1,008 GB/s (GDDR6X) |
| FP64 TFLOPS | 19.5 | 33.5 (SXM) | ~0.6 | ~1.3 |
| FP32 TFLOPS | 19.5 | 67 (SXM) | 35.6 | 82.6 |
| FP64:FP32 ratio | 1:1 (full rate) | 1:2 | 1:64 | 1:64 |
| Warp schedulers/SM | 4 | 4 | 4 | 4 |
| Warp size | 32 | 32 | 32 | 32 |
| VRAM | 40/80 GB | 80 GB | 24 GB | 24 GB |

**Critical implication**: On consumer GPUs (RTX 3090/4090), fp64 runs at
1/64th the throughput of fp32. A kernel hardcoded to `double` on an RTX 4090
runs at ~1.6% of its fp32 potential. On datacenter GPUs (A100), fp64 runs at
full rate (1:1). H100 is 1:2. This is why ADR-0002 precision dispatch exists.

---

## 1. Memory Management with RMM

### 1.1 Pool Allocation vs Direct CUDA Allocation

**Rule**: Always use a memory pool. Never call raw `cudaMalloc`/`cudaFree` in
hot paths.

- RMM pool suballocations cost <1 microsecond each (alloc + free)
- Raw `cudaMalloc` costs ~100-1000 microseconds depending on size
- RMM pools are ~1,000x faster than `cudaMalloc`/`cudaFree`
- Real-world impact: 4.6-10x speedup from pool allocation in RAPIDS benchmarks

**vibeSpatial pattern**: CuPy MemoryPool is configured at runtime init.
`VIBESPATIAL_GPU_POOL_LIMIT` env var caps pool size. Check via
`runtime.memory_pool_stats()`.

### 1.2 Pool Sizing

**Rule**: Reserve 200-500 MB for the CUDA context. Do not allocate 100% of VRAM.

```python
# Good: leave headroom
pool = rmm.mr.PoolMemoryResource(
    initial_pool_size=int(0.75 * total_vram),  # 75% of VRAM
    maximum_pool_size=int(0.90 * total_vram),  # 90% ceiling
)

# Bad: allocate everything
pool = rmm.mr.PoolMemoryResource(initial_pool_size=total_vram)
```

### 1.3 Pool vs Managed Memory

**Rule**: Never combine pooling with managed (unified) memory.

- Pool allocation prefers keeping data in VRAM for speed
- Managed memory transparently migrates between host and device
- Mixing them causes unpredictable performance and fragmentation
- Pick one strategy per allocation class

### 1.4 Async Memory Pools (cudaMallocAsync)

Use stream-ordered allocation when:
- Memory lifetimes are tied to specific stream operations
- Library code needs independent memory management without affecting the
  application's default pool
- Frequent alloc/free cycles within a loop (set release threshold to
  `UINT64_MAX` to prevent reallocation overhead)

Configuration:
```c
// Prevent pool from releasing memory between iterations
uint64_t threshold = UINT64_MAX;
cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
```

### 1.5 Sub-Allocator Patterns for Geometry Buffers

**Binning allocator**: Use RMM `binning_memory_resource` for mixed-size
geometry allocations:
- Small allocations (< 64 MB): `fixed_size_memory_resource` per size class
  (eliminates fragmentation)
- Large allocations (>= 64 MB): `pool_memory_resource` with coalescing

**Arena allocator**: `arena_memory_resource` carves per-thread pools for small
allocations, shares a pool for large ones. Good for multithreaded geometry
processing.

### 1.6 Fragmentation Avoidance

**Review checklist**:
- [ ] Allocations freed in LIFO order where possible (pool coalescing works best)
- [ ] No interleaving of long-lived and short-lived allocations on same pool
- [ ] Count-scatter pattern uses a single output allocation sized by the prefix
      sum total, not incremental resizing
- [ ] Temporary buffers freed before output allocation when possible

---

## 2. CUDA Stream Best Practices

### 2.1 When Streams Actually Help

**Use streams when**:
1. Independent H2D uploads (different source arrays, no dependency)
2. Independent kernel launches on separate data
3. Overlapping D2H transfer with compute on different buffers
4. Count-scatter total: batch async reads instead of sequential `.get()` syncs

**Do NOT use streams when**:
1. Sequential data dependencies (kernel B reads A's output) — same-stream
   ordering handles this with zero overhead
2. Tiny transfers — stream creation costs ~1-2 microseconds; if transfer time
   is less, streams add overhead
3. Single-kernel operations — nothing to overlap with
4. More than 8 concurrent streams — most GPUs see no benefit beyond 8, and
   scheduling overhead increases (20-30% execution time increase from
   queue mismanagement)

### 2.2 Stream-Ordered Memory Allocation

**Pattern**: Allocate and free memory as part of stream operations:
```c
cudaMallocAsync(&ptr, size, stream);
kernel<<<grid, block, 0, stream>>>(ptr);
cudaFreeAsync(ptr, stream);
```

**Benefits**:
- Memory freed on one stream can be immediately reused on the same stream
  without synchronization
- Cross-stream reuse works when dependency established via
  `cudaEventRecord`/`cudaStreamWaitEvent`
- Pool persists memory across iterations when release threshold is set

### 2.3 Synchronization Pitfalls

**Over-synchronization** (most common issue in vibeSpatial code reviews):

```python
# BAD: unnecessary sync between same-stream operations
runtime.launch(kernel_a, ...)
runtime.synchronize()           # UNNECESSARY — same stream guarantees order
runtime.launch(kernel_b, ...)   # reads kernel_a's output
runtime.synchronize()           # UNNECESSARY
offsets = exclusive_sum(counts)  # CCCL on same null stream

# GOOD: sync only before host-side reads
runtime.launch(kernel_a, ...)
runtime.launch(kernel_b, ...)
offsets = exclusive_sum(counts, synchronize=False)
runtime.synchronize()  # only when host needs the result
host_data = runtime.copy_device_to_host(d_output)
```

**Under-synchronization** (dangerous, causes data races):

```python
# BAD: reading device memory on host without sync
runtime.launch(kernel, ...)
result = d_output.get()  # .get() syncs implicitly — OK but hidden

# DANGEROUS: async copy without sync before use
runtime.copy_device_to_host_async(d_array, stream, h_buf)
print(h_buf[0])  # RACE: h_buf may not be populated yet
```

**Default stream serialization**:
Any operation on the default (null) stream blocks until ALL operations on
ALL other streams complete, and no subsequent operation on any stream begins
until it finishes. This makes the null stream a serialization point.

### 2.4 Operations That Cause Implicit Synchronization

Flag these in code review — they serialize the GPU pipeline:

| Operation | Sync Type |
|-----------|-----------|
| `cudaMalloc()` | Device-wide |
| `cudaFree()` | Device-wide |
| `cudaMemset()` | Device-wide |
| `cudaMemcpy()` (non-async) | Blocking + sync |
| Page-locked host memory allocation | Device-wide |
| Any operation on the null/default stream | Serializes all streams |
| L1/shared memory configuration change | Device-wide |
| Device memory copy to same device | Device-wide |
| CuPy `ndarray.get()` | Implicit sync + D2H copy |
| CuPy `.item()` | Implicit sync + D2H copy |
| CuPy `cp.asnumpy()` | Implicit sync + D2H copy |
| `print()` of a CuPy array | Implicit sync + D2H copy |
| Python `int(cupy_scalar)` / `float(cupy_scalar)` | Implicit sync |

### 2.5 Stream Priorities

```c
int leastPriority, greatestPriority;
cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatestPriority);
```

- Lower numbers = higher priority
- Priorities are **hints**, not guarantees
- Useful for prioritizing latency-sensitive operations (e.g., spatial query
  responses) over background work (e.g., buffer compaction)
- Not respected for memory transfers on most architectures

---

## 3. Kernel Launch Optimization

### 3.1 Occupancy-Based Block Sizing

**Rule**: NEVER hardcode `block=(256, 1, 1)`. Always use the occupancy API.

vibeSpatial pattern:
```python
grid, block = runtime.launch_config(kernel, item_count)
# Uses cuOccupancyMaxPotentialBlockSize internally
```

The occupancy API considers register pressure, shared memory usage, and SM
limits to select the block size that maximizes multiprocessor occupancy.

**Occupancy formula**:
```
occupancy = (active_warps_per_SM) / (max_warps_per_SM)
           = (active_blocks * threads_per_block / 32) / (max_threads_per_SM / 32)
```

| GPU | Max warps/SM | For 50% occupancy need |
|-----|-------------|----------------------|
| A100 | 64 | 32 warps = 1,024 threads |
| H100 | 64 | 32 warps = 1,024 threads |
| RTX 3090 | 48 | 24 warps = 768 threads |
| RTX 4090 | 48 | 24 warps = 768 threads |

**When manual tuning beats the API**: The occupancy API maximizes occupancy,
not performance. For kernels with high register usage or heavy shared memory
use, lower occupancy with more registers per thread can outperform higher
occupancy. Profile before overriding.

### 3.2 Grid-Stride Loops vs One-Thread-Per-Element

**Grid-stride loop** (preferred for most vibeSpatial kernels):
```c
for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
     idx < n;
     idx += blockDim.x * gridDim.x) {
    // process element idx
}
```

**Advantages**:
- Grid size independent of problem size (set based on SM count and occupancy)
- Naturally handles large datasets without oversized grids
- Amortizes launch overhead across multiple elements per thread
- Better instruction cache utilization

**One-thread-per-element**: Use only when work per element is large enough
to justify the grid size calculation overhead, or when cooperative groups
need exactly one block per work item.

**Grid sizing for grid-stride loops**:
```python
# Aim for enough blocks to fill all SMs with target occupancy
grid_size = min(
    (n + block_size - 1) // block_size,
    sm_count * max_blocks_per_sm
)
```

### 3.3 Cooperative Groups

Use `cooperative_groups` when you need **inter-block synchronization**
(grid-wide barrier). This requires `cudaLaunchCooperativeKernel`.

**When needed**:
- Global reduction to a single value in one kernel launch
- Iterative algorithms that need grid-wide convergence checks
- Multi-pass algorithms (count-scatter) fused into a single launch

**Constraints**:
- Grid size limited to `maxActiveBlocksPerMultiprocessor * SM_count`
  (cannot exceed occupancy — every block must be resident simultaneously)
- Exceeding limit causes `CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE`
- Higher launch overhead than standard kernel launch

**For vibeSpatial**: Prefer the two-pass count-scatter pattern over cooperative
groups. The explicit prefix sum between passes is simpler and has no grid size
constraint.

### 3.4 Kernel Fusion Opportunities

**Fuse when**:
- Output of kernel A is input to kernel B (producer-consumer, eliminates
  global memory round-trip)
- Both kernels are memory-bound (fusion increases arithmetic intensity)
- Intermediate data fits in registers or shared memory

**Do NOT fuse when**:
- Kernels have different thread-to-data mappings (would need partial syncs)
- Fusion would exceed register limits, causing spills
- Kernels have different optimal block sizes
- Cross-block communication would be needed

**Quantitative fusion guidance**:

| Pattern | Typical Speedup | Example |
|---------|----------------|---------|
| GEMM + epilogue | 25-33% | Not applicable to vibeSpatial |
| Pointwise chain (2-3 ops) | 2-3x | bounds compute + filter |
| Multi-stage scan | 2-4x | count + prefix sum (already handled by CCCL) |
| Softmax + matmul | 20-50% | Not applicable |

**Register pressure from fusion**: Monitor with `--maxrregcount` in NVRTC
or check via `cuFuncGetAttribute(CU_FUNC_ATTRIBUTE_NUM_REGS)`. If fused
kernel exceeds ~40 registers/thread on consumer GPUs (CC 8.6/8.9 with 48
max warps/SM), occupancy drops below 50%.

### 3.5 Launch Overhead Amortization

**Kernel launch overhead**: 2.5-5 microseconds on modern hardware (PCIe 4/5).
Can reach ~25 microseconds on older systems or WDDM (Windows).

**Minimum work per launch thresholds**:

| Scenario | Minimum to justify launch |
|----------|--------------------------|
| Trivial kernel (few ALU ops) | ~10,000 threads (320 warps) |
| Moderate kernel (50-100 ALU ops) | ~1,000 threads (32 warps) |
| Complex kernel (shared mem, loops) | ~256 threads (8 warps, 1 block per SM min) |
| Kernel execution < 25 us | Consider batching or fusing |

**Rule**: If kernel execution time < 10x launch overhead (~25-50 us), batch
the work into fewer, larger launches.

**NVIDIA guidance**: Ideal kernel execution times range from ~1 ms to ~200 ms.
Below 1 ms, launch overhead becomes significant. Above 200 ms, latency
hiding diminishes.

---

## 4. Memory Access Patterns

### 4.1 Coalesced Access and SoA vs AoS

**Rule**: Use Structure of Arrays (SoA) for geometry coordinate data. NEVER
use Array of Structures (AoS).

```c
// GOOD: SoA — coalesced access, threads read consecutive addresses
// x[0], x[1], x[2], ... x[31]  -> single 128-byte transaction
double* x;  // all x-coordinates contiguous
double* y;  // all y-coordinates contiguous

// BAD: AoS — strided access, 2-way bank conflict or 2x transactions
struct Point { double x, y; };
Point* points;  // points[0].x, points[0].y, points[1].x, points[1].y, ...
// Thread 0 reads points[0].x, Thread 1 reads points[1].x -> stride-2 access
```

**Coalescing rules** (compute capability 6.0+):
- Fundamental transaction size: 32 bytes
- A warp's memory access is coalesced into 128-byte transactions when threads
  access consecutive 4-byte or 8-byte elements
- Stride-2 access = 50% bandwidth utilization
- Stride-N access = 1/N bandwidth utilization (up to 1/32 = 3.1% for worst case)
- Coalesced access can improve throughput by up to 80% vs uncoalesced

**vibeSpatial convention**: Coordinate arrays are always separate `x[]` and
`y[]` arrays (SoA). This is enforced by the `OwnedGeometryArray` contract.

### 4.2 Shared Memory Usage

**Bank structure**: 32 banks, each 32 bits (4 bytes) wide, 1 bank per clock cycle.

**Bank conflict rules**:
- 32 consecutive 4-byte words map to 32 banks (word `i` -> bank `i % 32`)
- If 2+ threads in a warp access the same bank (different addresses), accesses
  serialize -> N-way conflict degrades bandwidth by factor N
- If all threads access the SAME address in a bank, it broadcasts (no conflict)
- Worst case: 32-way conflict = 32x slower

**Padding to avoid conflicts**:
```c
// BAD: column access on 32-wide array -> all threads hit same bank
__shared__ double matrix[32][32];  // column 0: banks 0,2,4,... (stride-2)

// GOOD: pad by 1 to shift bank mapping
__shared__ double matrix[32][33];  // column 0: banks 0,2,5,7,... (no conflict)
```

**For fp64 data**: Each `double` occupies 2 banks (8 bytes / 4 bytes per bank).
Stride-1 access of `double` arrays has inherent 2-way bank conflict. Mitigate
with padding (+1 element per row) or swizzling.

**Shared memory sizing for vibeSpatial kernels**:
- A100: up to 163 KB per block (configure via `cudaFuncSetAttribute`)
- H100: up to 227 KB per block
- RTX 3090/4090: up to 99 KB per block
- Default carveout: 48 KB if not explicitly configured

### 4.3 L2 Cache Residency Hints (CC 8.0+)

For data accessed repeatedly across kernel launches (e.g., spatial index
nodes, ring offset arrays):

```c
cudaAccessPolicyWindow window;
window.base_ptr = (void*)d_index_nodes;
window.num_bytes = index_size_bytes;
window.hitRatio = 1.0f;       // try to keep 100% in L2
window.hitProp = cudaAccessPropertyPersisting;
window.missProp = cudaAccessPropertyStreaming;

cudaStreamSetAccessPolicyWindow(stream, &window);
```

**Sizing rules**:
- Set aside up to 75% of L2 for persistent data:
  `size = min(l2CacheSize * 0.75, persistingL2CacheMaxSize)`
- If persistent data > set-aside, use `hitRatio < 1.0` to avoid thrashing:
  `hitRatio = set_aside_bytes / data_bytes`
- Exceeding L2 set-aside without hitRatio adjustment causes ~10% performance
  degradation from cache thrashing
- Proper L2 residency yields ~50% performance improvement for repeatedly
  accessed data

**L2 capacity for target GPUs**:
- A100: 40 MB set-aside available
- H100: 50 MB
- RTX 4090: 72 MB (largest L2 of all target GPUs)
- RTX 3090: 6 MB (severely limited — L2 hints less effective)

### 4.4 Read-Only Data Cache (__ldg)

For read-only coordinate data accessed through pointers:

```c
// In NVRTC kernel source — use __ldg for read-only global memory
double xi = __ldg(&x[i]);  // goes through texture/L1 read-only path
double yi = __ldg(&y[i]);
```

**When to use**:
- Read-only arrays accessed with spatial locality (coordinate arrays in
  ring traversal)
- Data not written by any thread in the kernel
- Alternative to texture objects (simpler, no binding required)

**Benefit**: Uses the read-only data cache path (separate from L1), increasing
effective cache capacity. Most beneficial when data access has 2D spatial
locality (neighboring coordinates).

**Caveat**: On CC 3.5+, the compiler automatically uses `__ldg` for
`const __restrict__` pointers. Prefer annotating pointers in NVRTC source:
```c
__global__ void kernel(const double* __restrict__ x,
                       const double* __restrict__ y, ...) {
```

---

## 5. Warp-Level Programming

### 5.1 Ballot/Vote for Predicate Evaluation

**Warp-wide early exit** (heavily used in vibeSpatial spatial filtering):

```c
const bool valid = row < row_count;
const unsigned char is_candidate = valid ? candidate_mask[row] : 0;
if (__ballot_sync(0xFFFFFFFF, is_candidate) == 0) {
    return;  // entire warp skips — no threads have work
}
```

At ~5% spatial filter hit rate, ~95% of warps exit immediately, avoiding all
global memory reads for non-candidate geometry.

**Primitives reference**:

| Primitive | Returns | Use Case |
|-----------|---------|----------|
| `__ballot_sync(mask, pred)` | 32-bit mask of threads where `pred` is true | Early exit, population count |
| `__any_sync(mask, pred)` | 1 if any thread has `pred` true | Skip work if no thread needs it |
| `__all_sync(mask, pred)` | 1 if all threads have `pred` true | Verify uniform condition |
| `__popc(__ballot_sync(...))` | Count of true predicates | Count matching elements |

### 5.2 Shuffle for Intra-Warp Communication

**Warp reduction** (register-only, no shared memory):

```c
#define FULL_MASK 0xFFFFFFFF
// XOR reduction for crossing parity (point-in-polygon)
for (int offset = 16; offset > 0; offset >>= 1) {
    my_crossings ^= __shfl_xor_sync(FULL_MASK, my_crossings, offset);
}

// Sum reduction
for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(FULL_MASK, val, offset);
}
// After reduction, lane 0 holds the result
```

**Shuffle primitives**:

| Primitive | Data Flow | Use |
|-----------|----------|-----|
| `__shfl_sync(mask, val, src_lane)` | Read from specific lane | Broadcast |
| `__shfl_down_sync(mask, val, delta)` | Read from lane + delta | Tree reduction |
| `__shfl_up_sync(mask, val, delta)` | Read from lane - delta | Prefix scan |
| `__shfl_xor_sync(mask, val, lane_mask)` | Read from lane XOR mask | Butterfly reduction |

**Performance**: Shuffle is register-to-register communication. No shared
memory load/store/address overhead. Always prefer shuffle for intra-warp
operations.

### 5.3 Warp Divergence Minimization

**Branch divergence penalty**:
- On Kepler: 32 cycles per divergent branch
- On Maxwell: 26 cycles per divergent branch
- On Volta+: Independent thread scheduling reduces penalty but does NOT
  eliminate it — divergent paths still execute serially within the warp

**Loop divergence**: When threads in a warp have different loop iteration
counts, ALL threads execute for `max(iterations)` cycles. Threads that finish
early are masked off but still consume SM resources.

**Strategies for mixed geometry types**:

1. **Sort by geometry type before dispatch** — group points, lines, polygons
   into contiguous ranges so warps process uniform types:
   ```python
   # Pre-sort by type (CCCL radix_sort by family code)
   sorted_indices = sort_pairs(family_codes, indices)
   # Launch separate kernels per family, or use sorted order
   ```

2. **Predicated execution** instead of branching:
   ```c
   // BAD: branch divergence
   if (geom_type == POLYGON) { result = polygon_op(); }
   else { result = line_op(); }

   // GOOD: compute both, select result (only if both are cheap)
   double poly_result = polygon_op();  // all threads compute
   double line_result = line_op();
   result = (geom_type == POLYGON) ? poly_result : line_result;
   ```

3. **Work-size binning** for variable complexity:
   ```python
   _WORK_BINS = [64, 1024]  # simple < 64 verts, medium 64-1024, complex > 1024
   # Bin geometries by vertex count, launch per-bin kernels
   ```

### 5.4 Block-Level Reductions

**Pattern**: warp shuffle -> shared memory -> final reduction:

```c
__shared__ int warp_results[8];  // up to 256 threads = 8 warps
const int warp_id = threadIdx.x / 32;
const int lane_id = threadIdx.x % 32;

// Step 1: warp-level shuffle reduction
for (int offset = 16; offset > 0; offset >>= 1)
    my_value += __shfl_down_sync(0xFFFFFFFF, my_value, offset);

// Step 2: lane 0 of each warp writes to shared memory
if (lane_id == 0) warp_results[warp_id] = my_value;
__syncthreads();

// Step 3: first warp reduces across all warp results
if (threadIdx.x < (blockDim.x + 31) / 32) {
    my_value = warp_results[threadIdx.x];
    for (int offset = 16; offset > 0; offset >>= 1)
        my_value += __shfl_down_sync(0xFFFFFFFF, my_value, offset);
    if (threadIdx.x == 0) output[blockIdx.x] = my_value;
}
```

**Rule**: For block-level reductions, prefer this shuffle+shared pattern over
pure shared memory reduction trees. It uses fewer `__syncthreads()` barriers
and fewer shared memory transactions.

**Alternative**: Use CUB/CCCL `BlockReduce` when available — it selects the
optimal algorithm automatically.

---

## 6. Consumer GPU Considerations (fp64 Performance)

### 6.1 FP64:FP32 Throughput Ratios

| GPU | FP64 TFLOPS | FP32 TFLOPS | Ratio | Implication |
|-----|------------|------------|-------|-------------|
| A100 SXM | 19.5 | 19.5 | 1:1 | fp64 is free — use it everywhere |
| H100 SXM | 33.5 | 67 | 1:2 | fp64 at half speed — still fast |
| RTX 3090 | ~0.6 | 35.6 | 1:64 | fp64 is catastrophically slow |
| RTX 4090 | ~1.3 | 82.6 | 1:64 | fp64 is catastrophically slow |

**Detection**: Use `CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO`
(returns the integer ratio). Do NOT use compute capability or device name
heuristics (per project feedback rule `no-cc-heuristics`).

### 6.2 Mixed Precision Strategy (ADR-0002)

| Kernel Class | Consumer GPU Strategy | Datacenter GPU Strategy |
|-------------|----------------------|------------------------|
| COARSE (bounds, index) | fp32 with coordinate centering | Native fp64 |
| METRIC (distance, area) | fp32 with Kahan compensation | Native fp64 |
| PREDICATE (PIP, predicates) | fp32 coarse + selective fp64 refinement | Native fp64 |
| CONSTRUCTIVE (clip, overlay) | Native fp64 (correctness required) | Native fp64 |

### 6.3 When fp32 Is Sufficient

- **Bounding box computation**: min/max operations, ~7 decimal digits is
  adequate for spatial filtering
- **Coarse spatial filters**: candidate generation (false positives OK,
  false negatives are not — use conservative fp32 rounding for bounds:
  round min DOWN, max UP)
- **Distance comparisons** (not absolute values): relative ordering preserved
- **Integer-like operations**: counting, indexing, flag setting

### 6.4 When fp64 Is Required

- **Constructive geometry**: intersection points, clipping coordinates —
  fp32 error compounds and produces topologically invalid results
- **Area/length when absolute accuracy matters**: fp32 Kahan summation
  achieves ~8e-4 max error at 1e6-magnitude coords, which may be insufficient
  for some applications
- **Coordinate differences at small scales**: when `|x1 - x2| / |x1| < 1e-6`,
  fp32 catastrophic cancellation occurs

### 6.5 Kahan Summation for fp32 Accumulation

```c
// Kahan-compensated summation — effectively doubles fp32 precision for sums
compute_t sum = (compute_t)0.0;
compute_t kahan_c = (compute_t)0.0;  // compensation variable

for (int i = 0; i < n; i++) {
    compute_t y = value[i] - kahan_c;
    compute_t t = sum + y;
    kahan_c = (t - sum) - y;  // recovers the lost low-order bits
    sum = t;
}
```

- Error bound: O(epsilon) independent of n, vs O(n * epsilon) for naive sum
- Overhead: 4 extra FLOPs per accumulation step (negligible vs memory access)
- Used in vibeSpatial for: shoelace area, polygon centroid, distance accumulation

### 6.6 Review Checklist for Precision

- [ ] Kernel source uses `compute_t` typedef, not hardcoded `double`
- [ ] Storage reads stay `double`, cast to `compute_t` after subtraction
- [ ] Cache key includes precision variant: `f"kernel-{compute_type}"`
- [ ] Coordinate centering applied when `plan.center_coordinates is True`
- [ ] Kahan compensation for METRIC accumulations when `plan.compensation is KAHAN`
- [ ] CONSTRUCTIVE kernels wired through PrecisionPlan but stay fp64
- [ ] Boundary tolerances widened for fp32 (e.g., 1e-7 instead of 1e-12)

---

## 7. NVRTC Best Practices

### 7.1 Template Specialization for Precision Dispatch

```python
_KERNEL_SOURCE_TEMPLATE = """
typedef {compute_type} compute_t;

extern "C" __global__ void my_kernel(
    const double* __restrict__ x,   // storage always fp64
    const double* __restrict__ y,
    double* __restrict__ output,
    const double center_x,
    const double center_y,
    const int n
) {{
    // Cast to compute_t after centering
    compute_t lx = (compute_t)(x[idx] - center_x);
    compute_t ly = (compute_t)(y[idx] - center_y);
    // ... computation in compute_t ...
    output[idx] = (double)result;   // write back as fp64
}}
"""

# Format for specific precision
compute_type = "float" if plan.compute_precision is PrecisionMode.FP32 else "double"
source = _KERNEL_SOURCE_TEMPLATE.format(compute_type=compute_type)
cache_key = make_kernel_cache_key(f"my-kernel-{compute_type}", source)
```

### 7.2 Compilation Caching

vibeSpatial uses SHA1-based caching in `CudaDriverRuntime._module_cache`.

**NVRTC built-in cache** (CUDA 12.8+): Precompiled headers (PCH):
- `--pch` flag enables automatic PCH creation and reuse
- PCH heap default: 256 MB (configurable via `NVRTC_PCH_HEAP_SIZE`)
- PCH files require matching: compiler options, preprocessor defines,
  compiler version, heap base address

**Application-level caching** (vibeSpatial pattern):
```python
cache_key = make_kernel_cache_key("kernel-name", source_string)
kernels = runtime.compile_kernels(
    cache_key=cache_key,
    source=source_string,
    kernel_names=("entry_point_1", "entry_point_2"),
)
# Subsequent calls with same cache_key skip compilation
```

**Compile to CUBIN when possible**: Target real architectures (`sm_80`,
`sm_89`, `sm_90`) instead of virtual (`compute_80`) to avoid JIT compilation
at load time. Use `nvrtcGetCUBIN()` instead of `nvrtcGetPTX()`.

### 7.3 Precompilation/Warmup (ADR-0034)

Cold-call JIT costs:
- CCCL primitives: ~950-1,460 ms per unique spec
- NVRTC kernels: ~20-400 ms per compilation unit

vibeSpatial uses three-level demand-driven warmup:
1. **CCCL Module Warmup**: `request_warmup()` at module scope, background
   ThreadPoolExecutor (8 threads)
2. **NVRTC Module Warmup**: `request_nvrtc_warmup()` for kernel source units
3. **Pipeline-Aware**: declare all requirements at plan construction

**Thread scaling** (18 CCCL specs): 1 thread = ~20s, 8 threads = ~2-3s
(7x speedup)

### 7.4 Useful NVRTC Compiler Flags

| Flag | Effect | When to Use |
|------|--------|-------------|
| `--use_fast_math` | Enables ftz, fast div/sqrt, fmad | Non-precision-critical kernels only |
| `--maxrregcount=N` | Cap registers per thread | When occupancy is limited by register pressure |
| `--std=c++17` | C++17 features | Default for NVRTC (vibeSpatial convention) |
| `--gpu-architecture=sm_XX` | Target specific SM | Compile to CUBIN for target GPU |
| `--split-compile=0` | Parallel optimization passes | Large kernels, use all CPU threads |
| `--extra-device-vectorization` | Aggressive vectorization | Memory-bound kernels with regular access |
| `--ftz=true` | Flush denormals to zero | fp32 kernels where denormals are noise |
| `--Ofast-compile=mid` | Trade compile speed for runtime perf | Development iteration |

### 7.5 JIT Cost Amortization

**Review checklist**:
- [ ] Kernel source uses `make_kernel_cache_key()` for SHA1 caching
- [ ] Cache key includes ALL parameterizations (precision, block size, etc.)
- [ ] Module declares `request_nvrtc_warmup()` at module scope
- [ ] CCCL specs declare `request_warmup()` at module scope
- [ ] No compilation in hot loops (compile once, launch many times)

---

## 8. Anti-Pattern Detection Checklist

### 8.1 Implicit Synchronization

**Severity: HIGH** — silent performance killers.

| Pattern | Why It's Bad | Fix |
|---------|-------------|-----|
| `print(cupy_array)` in debug code | Triggers D2H copy + sync | Remove or gate behind `if DEBUG:` |
| `len(cupy_array)` that requires `.get()` | Hidden sync | Use `.shape[0]` (metadata, no sync) |
| `int(cupy_scalar)` / `float(cupy_scalar)` | Scalar D2H sync | Keep as device scalar until needed |
| `cp.asnumpy(arr)` in middle of pipeline | Sync + full D2H | Defer to pipeline end |
| `ndarray.get()` between kernel launches | Sync + D2H | Use `count_scatter_total()` pattern |
| `runtime.synchronize()` between same-stream ops | Unnecessary pipeline stall | Remove — stream ordering suffices |
| `cudaMalloc` in kernel loop | Device-wide implicit sync | Pre-allocate or use pool |

### 8.2 Host-Device Ping-Pong

**Severity: CRITICAL** — the most impactful anti-pattern.

```python
# BAD: D->H->D round-trip in a loop
for i in range(n_geometries):
    count = d_counts[i].get()        # D2H sync
    if count > 0:
        d_output = runtime.allocate((count,), np.float64)  # allocation on host
        runtime.launch(scatter_kernel, ...)                  # H2D of params

# GOOD: keep everything on device
offsets = exclusive_sum(d_counts, synchronize=False)
total = count_scatter_total(runtime, d_counts, offsets)  # single sync
d_output = runtime.allocate((total,), np.float64)
runtime.launch(scatter_kernel, ...)  # one launch for all geometries
```

**Detection rules**:
- Any `.get()`, `cp.asnumpy()`, `int()`, or `float()` of device data followed
  by a decision that feeds back into a kernel launch = ping-pong
- Any Python `for` loop over device array elements = ping-pong
- Any conditional allocation based on device-side values without
  `count_scatter_total()` = ping-pong

### 8.3 Unnecessary Copies Between CuPy and cuda-python

**Zero-copy interop mechanisms**:
- `__cuda_array_interface__`: CuPy arrays expose device pointers directly
- `DLPack`: `cupy.from_dlpack()` / `cupy.ndarray.__dlpack__()` for zero-copy
- `UnownedMemory`: Wrap external device pointers without copy
- `runtime.pointer(cupy_array)`: Get raw device pointer for cuda-python launch

```python
# BAD: unnecessary copy
host_data = cupy_array.get()            # D2H
d_ptr = runtime.from_host(host_data)    # H2D (same data, round-tripped!)

# GOOD: direct pointer extraction
d_ptr = runtime.pointer(cupy_array)     # zero-copy, same device memory
```

**Stream interop**: Use `cupy.cuda.Stream.from_external()` when mixing CuPy
and cuda-python stream contexts.

### 8.4 Small Kernel Launches

**Severity: MEDIUM** — accumulates in pipelines with many stages.

**Threshold**: If kernel execution time < 25 microseconds and launch overhead
is ~3-5 microseconds, you're spending >10% on launch overhead.

**Detection rules**:
- Kernel launched with < 32 threads of real work (wastes a warp)
- Kernel launched per-geometry in a loop instead of batched
- Grid size = 1 (single block — GPU is >99% idle)
- Multiple kernels that could be fused (same input, similar work)

**Minimum grid sizes to saturate GPU**:

| GPU | SMs | Min blocks for 50% | Min threads (256/block) |
|-----|-----|--------------------|-----------------------|
| A100 | 108 | 54 | 13,824 |
| H100 | 132 | 66 | 16,896 |
| RTX 3090 | 82 | 41 | 10,496 |
| RTX 4090 | 128 | 64 | 16,384 |

### 8.5 Branch Divergence in Inner Loops

**Severity: HIGH** in geometry processing kernels.

```c
// BAD: divergent branch in edge traversal inner loop
for (int e = 0; e < edge_count; e++) {
    if (geom_type[row] == POLYGON) {
        // polygon edge processing (half the warp)
    } else {
        // linestring edge processing (other half)
    }
}
// Each branch executes serially — 2x slowdown

// GOOD: sort by type and launch separate kernels, or use predication
```

**Loop divergence** is especially damaging: if one thread in a warp processes
a polygon with 10,000 edges while others process polygons with 10 edges,
ALL threads execute for 10,000 iterations (9,990 wasted per thread).

**Mitigation**: Work-size binning (see Section 5.3).

### 8.6 Register Pressure

**Severity: MEDIUM** — causes occupancy drops and spilling.

**Max registers per thread**: 255 (all target architectures).

**Occupancy impact** (A100 with 65,536 registers/SM, 2,048 max threads/SM):
- 32 regs/thread -> 2,048 threads/SM -> 100% occupancy
- 64 regs/thread -> 1,024 threads/SM -> 50% occupancy
- 128 regs/thread -> 512 threads/SM -> 25% occupancy
- 255 regs/thread -> 256 threads/SM -> 12.5% occupancy

**On CC 8.6/8.9** (RTX 3090/4090 with 1,536 max threads/SM):
- 32 regs/thread -> 1,536 threads/SM -> 100% occupancy
- 42 regs/thread -> 1,536 threads/SM -> 100% (register limit is not yet binding)
- 64 regs/thread -> 1,024 threads/SM -> 66% occupancy
- 128 regs/thread -> 512 threads/SM -> 33% occupancy

**Spilling**: When a kernel exceeds register limits, the compiler spills to
local memory (actually global memory with L1 caching). Spills in hot loops
cause dramatic slowdowns.

**Detection**: Check with `cuFuncGetAttribute(CU_FUNC_ATTRIBUTE_NUM_REGS, func)`.
Flag if > 64 registers per thread for compute-bound kernels, or > 40 for
memory-bound kernels where occupancy matters more.

**Reducing register pressure**:
- Break complex kernels into smaller device functions
- Use `--maxrregcount=N` to cap registers (compiler will spill)
- Replace `double` with `float` where precision allows (halves register usage
  for each variable)
- Move infrequently used variables to shared memory

### 8.7 Global Memory Atomics

**Severity: MEDIUM** — serializes access to contended addresses.

```c
// BAD: global atomic in inner loop (all threads contend)
for (int e = 0; e < edge_count; e++) {
    atomicAdd(&global_count[bin], 1);  // serialized
}

// GOOD: local accumulation + single atomic at end
int local_count = 0;
for (int e = 0; e < edge_count; e++) {
    local_count++;  // register, no contention
}
atomicAdd(&global_count[bin], local_count);  // one atomic per thread

// BETTER: warp-level reduction + single atomic per warp
// ... shuffle reduction to lane 0 ...
if (lane_id == 0) atomicAdd(&global_count[bin], warp_sum);
```

**Rule**: Replace per-iteration atomics with:
1. Register accumulation -> single atomic
2. Warp shuffle reduction -> single atomic per warp
3. Block-level shared memory reduction -> single atomic per block
4. CCCL `reduce` / `segmented_reduce` for known reduction patterns

---

## 9. Review Procedure

When reviewing GPU code, check in this order (highest impact first):

### Pass 1: Host-Device Boundary (CRITICAL)
- [ ] No D->H->D ping-pong patterns (Section 8.2)
- [ ] No `.get()` / `cp.asnumpy()` in middle of GPU pipeline
- [ ] No Python loops over device array elements
- [ ] All D2H transfers deferred to pipeline end
- [ ] `count_scatter_total()` used instead of sequential `.get()` calls

### Pass 2: Synchronization (HIGH)
- [ ] No `runtime.synchronize()` between same-stream operations (Section 2.3)
- [ ] No implicit sync from debug prints, scalar conversions (Section 8.1)
- [ ] Stream sync only before host reads of device data
- [ ] No `cudaMalloc`/`cudaFree` in hot paths (use pool)

### Pass 3: Kernel Efficiency (HIGH)
- [ ] Block size via occupancy API, not hardcoded (Section 3.1)
- [ ] Grid size sufficient to saturate GPU (Section 8.4)
- [ ] No branch divergence in inner loops (Section 8.5)
- [ ] Work-size binning for variable-complexity geometry (Section 5.3)
- [ ] SoA layout for coordinate data (Section 4.1)

### Pass 4: Precision Compliance (MEDIUM-HIGH)
- [ ] `compute_t` typedef, not hardcoded `double` (Section 6.6)
- [ ] PrecisionPlan wired through dispatch (Section 6.2)
- [ ] Cache key includes precision variant (Section 7.1)
- [ ] Kahan compensation for METRIC fp32 accumulations (Section 6.5)

### Pass 5: Memory Management (MEDIUM)
- [ ] Pool allocation, not raw `cudaMalloc` (Section 1.1)
- [ ] LIFO deallocation order where possible (Section 1.6)
- [ ] Pre-sized output buffers via count-scatter (Section 1.6)
- [ ] Shared memory bank conflicts avoided (Section 4.2)

### Pass 6: NVRTC/Compilation (LOW-MEDIUM)
- [ ] SHA1 cache key covers all parameterizations (Section 7.2)
- [ ] Module-scope warmup declared (Section 7.3)
- [ ] No compilation in hot paths
- [ ] `const __restrict__` on read-only pointer params (Section 4.4)

---

## 10. Quick Reference: Quantitative Thresholds

| Metric | Threshold | Action |
|--------|-----------|--------|
| Kernel launch overhead | 3-5 us (modern PCIe 4/5) | If kernel < 25 us execution, consider batching |
| Minimum threads per launch | 32 (1 warp) absolute minimum | Prefer 10,000+ for saturation |
| Occupancy target | >= 50% for memory-bound kernels | Check with occupancy API |
| Registers per thread (warning) | > 64 (datacenter), > 40 (consumer) | Profile; may need `--maxrregcount` |
| Shared memory bank conflict | Any N-way conflict in inner loop | Pad or swizzle |
| fp64:fp32 penalty (consumer) | 64x slower | Use ADR-0002 precision dispatch |
| RMM pool vs cudaMalloc | 1,000x faster | Always use pool |
| L2 set-aside thrashing | > 10% perf drop if oversized | Tune hitRatio = set_aside / data_size |
| Stream count (diminishing returns) | > 8 concurrent streams | Consolidate work |
| Kahan summation overhead | 4 extra FLOPs per step | Negligible vs memory access |
| Work-size binning threshold | CV > 2.0 (std/mean of work sizes) | Bin by vertex count |
| CCCL cold-call JIT | 950-1,460 ms | Mandate `request_warmup()` |
| NVRTC cold compilation | 20-400 ms per unit | Mandate `request_nvrtc_warmup()` |
| Grid-stride loop grid size | SM_count * max_blocks_per_SM | Maximize SM utilization |

---

## Sources

This reference was compiled from the following sources:

- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)
- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- [NVIDIA Ada Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html)
- [CUDA Programming Guide: Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)
- [CUDA Programming Guide: L2 Cache Control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html)
- [CUDA Programming Guide: Stream-Ordered Memory Allocator](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html)
- [RMM: Fast, Flexible Allocation for CUDA (NVIDIA Blog)](https://developer.nvidia.com/blog/fast-flexible-allocation-for-cuda-with-rapids-memory-manager/)
- [Using CUDA Warp-Level Primitives (NVIDIA Blog)](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)
- [CUDA Pro Tip: Occupancy API (NVIDIA Blog)](https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/)
- [CUDA Stream-Ordered Memory Allocator (NVIDIA Blog)](https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/)
- [Using Shared Memory in CUDA (NVIDIA Blog)](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
- [Faster Parallel Reductions on Kepler (NVIDIA Blog)](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/)
- [CUDA Kernel Fusion Strategies (Emergent Mind)](https://www.emergentmind.com/topics/cuda-kernel-fusion)
- [NVRTC Documentation](https://docs.nvidia.com/cuda/nvrtc/index.html)
- [CuPy Interoperability Guide](https://docs.cupy.dev/en/stable/user_guide/interoperability.html)
- [CuPy Implicit Synchronization Detection (GitHub Issue #2808)](https://github.com/cupy/cupy/issues/2808)
- [CUDA Implicit Synchronization (NVIDIA Forums)](https://forums.developer.nvidia.com/t/cuda-implicit-synchronization-behavior-and-conditions-in-detail/251729)
- [Kernel Launch Overhead Discussion (NVIDIA Forums)](https://forums.developer.nvidia.com/t/launch-of-many-small-kernels-10x-slower-compared-to-one-kernel/350194)
- [RMM GitHub Repository](https://github.com/rapidsai/rmm)
- [NVIDIA Jitify (NVRTC simplification)](https://github.com/NVIDIA/jitify)
- [Benchmarking Thread Divergence in CUDA (arXiv)](https://arxiv.org/pdf/1504.01650)
- [Kahan Summation Algorithm (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm)
