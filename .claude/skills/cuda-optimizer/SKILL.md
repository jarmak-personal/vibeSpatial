---
name: cuda-optimizer
description: "Use this skill to optimize existing CUDA/NVRTC kernel code, CuPy operations, CCCL primitive usage, or GPU dispatch logic in src/vibespatial/. Unlike gpu-code-review (which flags issues) and cuda-writing (which guides new code), this skill reads existing code and produces concrete rewrites with measured justification. Invoke on pre-existing kernel files to bring them up to NVIDIA best-practice performance standards."
user-invocable: true
argument-hint: <file-path or module-name to optimize>
---

# CUDA Optimizer — vibeSpatial

You are optimizing GPU code in vibeSpatial. Your job is to read the target
file, identify every concrete optimization opportunity, and produce
ready-to-apply rewrites ranked by expected impact.

**Target:** `$ARGUMENTS`

---

## Procedure

### Step 0: Read the Code

Read the target file completely. Also read the companion `*_kernels.py`
or `*_source.py` sibling file where kernel source now lives (kernel source
is no longer inline in dispatch files). Also read any other files the
target imports from `vibespatial.*` that contain GPU dispatch logic.
Identify:

- All NVRTC kernel source strings (look for `_KERNEL_SOURCE` or multi-line
  strings containing `__global__` in `*_kernels.py` / `*_source.py` files)
- All CuPy operations (`cp.` calls)
- All CCCL primitive usage (`cccl_primitives.*`, `cccl_algorithms.*`)
- All `runtime.launch()` calls and their parameters
- All host-device transfers (`.get()`, `cp.asnumpy()`, `from_host()`,
  `copy_device_to_host()`)
- All synchronization points (`runtime.synchronize()`, `stream.synchronize()`)

### Step 1: Host-Device Boundary (CRITICAL — highest impact)

Scan for these patterns and produce rewrites:

**1a. D->H->D ping-pong in loops**
```python
# BEFORE: ping-pong per geometry
for i in range(n):
    count = d_counts[i].get()        # D2H sync per iteration
    if count > 0:
        d_out = runtime.allocate((count,), ...)

# AFTER: single bulk operation
offsets = exclusive_sum(d_counts, synchronize=False)
total = count_scatter_total(runtime, d_counts, offsets)
d_out = runtime.allocate((total,), ...)
```

**1b. Python loops over device arrays**
```python
# BEFORE: element-wise Python loop
results = []
for geom in device_geometries:
    results.append(process(geom.get()))  # N syncs

# AFTER: bulk kernel or CuPy vectorized op
results = bulk_process_gpu(device_geometries)  # 1 launch
```

**1c. Mid-pipeline `.get()` / `cp.asnumpy()`**

Any `.get()` or `cp.asnumpy()` that is NOT at the final return of a
pipeline is suspect. Check if the host value is used to make a decision
that feeds back into GPU work. If yes, rewrite to keep the decision on
device (e.g., use a device-side mask instead of host-side conditional).

**1d. Sequential `.get()` for count-scatter totals**
```python
# BEFORE: two syncs
count_total = d_counts[-1].get() + d_offsets[-1].get()

# AFTER: single async pinned transfer
total = count_scatter_total(runtime, d_counts, d_offsets)
```

### Step 2: Synchronization Elimination (HIGH impact)

**2a. Unnecessary `runtime.synchronize()` between same-stream ops**

CUDA guarantees execution order within a single stream. Remove syncs
between consecutive kernel launches, CCCL calls, or kernel->CCCL
sequences on the same (null) stream.

```python
# BEFORE
runtime.launch(kernel_a, ...)
runtime.synchronize()           # REMOVE
runtime.launch(kernel_b, ...)   # reads kernel_a output — stream order suffices
runtime.synchronize()           # REMOVE
offsets = exclusive_sum(counts)  # same null stream

# AFTER
runtime.launch(kernel_a, ...)
runtime.launch(kernel_b, ...)
offsets = exclusive_sum(counts, synchronize=False)
runtime.synchronize()  # single sync before host reads
```

**2b. Implicit syncs from scalar reads**

Flag: `int(cupy_scalar)`, `float(cupy_scalar)`, `print(cupy_array)`,
`len()` that triggers `.get()`. Replace with `.shape[0]` for length,
keep scalars on device until pipeline end.

**2c. `cudaMalloc`/`cudaFree` in hot paths**

Each is a device-wide implicit sync. Pre-allocate or use pool.

### Step 3: Kernel Source Optimization (HIGH impact)

Read every NVRTC kernel source string and check:

**3a. `const __restrict__` on read-only pointers**

Every pointer parameter that the kernel only reads from should have
`const ... __restrict__`:
```c
// BEFORE
extern "C" __global__ void kernel(double* x, double* y, double* out, int n)

// AFTER
extern "C" __global__ void kernel(
    const double* __restrict__ x,
    const double* __restrict__ y,
    double* __restrict__ out,
    const int n)
```
This enables the compiler to use the `__ldg` read-only cache path
automatically (CC 3.5+), increasing effective cache capacity.

**3b. `__launch_bounds__` directive**

If the kernel is launched with a known max block size, add the directive:
```c
// BEFORE
extern "C" __global__ void kernel(...)

// AFTER
extern "C" __global__ void __launch_bounds__(256, 4) kernel(...)
```
Without it, the compiler assumes max threads per block, which can
over-allocate registers and reduce occupancy.

**3c. Grid-stride loop with ILP**

If the kernel uses a simple grid-stride loop processing one element per
iteration, rewrite with multi-element ILP:
```c
// BEFORE: 1 element/thread
for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
     idx < n; idx += blockDim.x * gridDim.x) {{
    output[idx] = compute(input[idx]);
}}

// AFTER: 4 elements/thread with ILP
const int stride = blockDim.x * gridDim.x;
for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
     idx < n; idx += stride * 4) {{
    #pragma unroll
    for (int j = 0; j < 4; j++) {{
        int elem = idx + j * stride;
        if (elem < n) output[elem] = compute(input[elem]);
    }}
}}
```
NVIDIA data: 4 elements/thread at 50% occupancy beat 1 element/thread
at 100% occupancy (82% vs 51% bandwidth utilization).

**3d. Vectorized loads/stores for bulk data movement**

If the kernel copies or transforms contiguous arrays element by element,
use 128-bit vector loads:
```c
// BEFORE: scalar 8-byte loads
double val = input[idx];

// AFTER: 128-bit vectorized (2x doubles at once)
double2 vals = reinterpret_cast<const double2*>(input)[idx];
// Process vals.x and vals.y
```
1.3-1.5x bandwidth improvement for memory-bound kernels.

**3e. Warp-divergent branches in inner loops**

Look for `if/else` on geometry type or variable-length iteration inside
the innermost loop. Propose:
- Sort by type before dispatch (CCCL `radix_sort` by family code)
- Predicated writes instead of branches (when both paths are cheap)
- Work-size binning for variable iteration counts (CV > 2.0)

**3f. Shared memory bank conflicts**

If shared memory arrays are 32-wide and accessed in columns, add +1
padding:
```c
// BEFORE: 32-way bank conflict on column access
__shared__ float tile[32][32];

// AFTER: no conflicts
__shared__ float tile[32][33];
```

**3g. `__syncwarp()` after divergent branches (Volta+)**

On CC 7.0+, warps can remain diverged after conditionals. If warp
shuffle or ballot is used after a conditional block, ensure
`__syncwarp(0xFFFFFFFF)` is called first.

**3h. Atomic contention reduction**

If `atomicAdd` or similar appears inside an inner loop:
```c
// BEFORE: per-iteration global atomic
for (int e = 0; e < edge_count; e++)
    atomicAdd(&count[bin], 1);

// AFTER: register accumulation + warp reduction + single atomic
int local = 0;
for (int e = 0; e < edge_count; e++) local++;
// Warp shuffle reduction to lane 0...
if (lane_id == 0) atomicAdd(&count[bin], warp_sum);
```

**3i. Float constant precision**

In fp32 kernels (`compute_t = float`), check for unqualified float
constants (`0.0`, `1.0`, `1e-7`) which compile as doubles and force
conversion. Use explicit casts: `(compute_t)0.0` or `0.0f`.

**3j. Integer division/modulo optimization**

`threadIdx.x / 32` -> `threadIdx.x >> 5`
`threadIdx.x % 32` -> `threadIdx.x & 31`

Integer div/mod compiles to up to 20 instructions. Use bitwise ops
when the divisor is a power of 2.

### Step 4: Launch Configuration (MEDIUM impact)

**4a. Hardcoded block sizes**

Any `block=(256, 1, 1)` or similar literal should use the occupancy API:
```python
grid, block = runtime.launch_config(kernel, item_count)
```

**4b. Wave quantization**

Check if `grid_size` is just barely over `SM_count * max_blocks_per_SM`.
On RTX 4090 (128 SMs), grid=129 runs a second wave with 1 block = ~50%
GPU waste. Use grid-stride loops and cap grid size:
```python
grid_size = min(
    (n + block_size - 1) // block_size,
    sm_count * max_blocks_per_sm
)
```

**4c. Tiny kernel launches**

Flag any kernel launched with < 32 threads of real work, or launched
per-geometry in a Python loop. Propose batched alternatives.

### Step 5: Memory Access Patterns (MEDIUM impact)

**5a. AoS layout**

Any struct-like interleaving of coordinates (`[x0,y0,x1,y1,...]`) should
be converted to SoA (`x[0..n], y[0..n]`). AoS is 5.9x slower than SoA
in NVIDIA benchmarks due to strided access.

**5b. L2 cache-aware patterns for multi-kernel pipelines**

If two kernels run back-to-back on the same data, consider reverse
block traversal on the second kernel:
```c
// Kernel A: forward
int blockId = blockIdx.x;
// Kernel B: reverse — hits A's still-cached tail data in L2
int blockId = gridDim.x - blockIdx.x - 1;
```

**5c. Async copy to shared memory**

For kernels that tile global data into shared memory via register loads,
consider `__pipeline_memcpy_async()` (CUDA 11.0+) to bypass the register
file and reduce register pressure:
```c
__pipeline_memcpy_async(&smem[tid], &gmem[idx], sizeof(double));
__pipeline_commit();
__pipeline_wait_prior(0);
__syncthreads();
```

### Step 6: CCCL/CuPy Tier Optimization (MEDIUM impact)

Check every CuPy operation against the ADR-0033 tier system:

| Operation | Current | Better | Why |
|-----------|---------|--------|-----|
| `cp.cumsum()` | Tier 4 (CuPy) | OK | Marginal CCCL advantage |
| `cp.sum()` | Tier 4 (CuPy) | OK | Marginal CCCL advantage |
| `cp.arange(n)` | Tier 2 (CuPy) | `counting_iterator` (Tier 3c) | Zero allocation |
| Element-wise + reduce | 2 kernels | `transform_iterator` + reduce (Tier 3c) | Kernel fusion |
| `exclusive_sum` | Must be CCCL | Check `make_*` reuse | Pre-compiled, no JIT per call |
| Custom sort with comparator | CuPy `argsort` | CCCL `radix_sort`/`merge_sort` | Correct and faster |

Also check:
- Is `exclusive_sum` using the `make_*` API when called repeatedly?
- Is `compact_indices` using CuPy default (not CCCL, per 2026-03-17 revert)?
- Are CCCL specs declared in `request_warmup()` at module scope?
- Are NVRTC kernels declared in `request_nvrtc_warmup()` at module scope?

### Step 7: Precision Dispatch (MEDIUM impact)

Check if the kernel is ADR-0002 compliant:

- Does the kernel source use `compute_t` typedef or hardcoded `double`?
- Is `select_precision_plan()` called at the dispatch layer?
- Does the cache key include precision variant?
- For METRIC kernels: is Kahan compensation wired for fp32?
- For COARSE kernels: are bounds staying fp64 (correct) or using
  unguarded fp32 (will cause false negatives)?

If non-compliant, refer to the `precision-compliance` skill for the
full implementation procedure.

### Step 8: Precompilation (LOW impact, cold-start only)

- Does the module call `request_warmup()` for its CCCL specs?
- Does the module call `request_nvrtc_warmup()` for its kernel sources?
- Missing warmup = 950-1,460 ms cold-call JIT penalty per CCCL spec,
  20-400 ms per NVRTC compilation unit.

---

## Output Format

For each finding, produce:

```
### [PRIORITY] Finding Title

**File:** `path/to/file.py:LINE`
**Impact:** Brief explanation of why this matters and expected improvement
**Category:** (host-device | sync | kernel-source | launch-config | memory-access | tier | precision | warmup)

**Before:**
\```python (or c)
<exact current code>
\```

**After:**
\```python (or c)
<concrete rewrite>
\```
```

Sort findings by priority: CRITICAL > HIGH > MEDIUM > LOW.

At the end, produce a summary table:

```
| # | Priority | Category | File:Line | Finding | Est. Impact |
|---|----------|----------|-----------|---------|-------------|
```

---

## Rules

- **Read before recommending.** Never suggest changes to code you
  haven't read. Always quote the exact current code in "Before."
- **Concrete rewrites only.** Every finding must have a "Before" and
  "After" block. No vague advice like "consider optimizing."
- **Verify tier compliance.** Check ADR-0033 before suggesting a CuPy
  -> CCCL migration — some operations were intentionally reverted.
- **Don't break correctness.** Precision changes must go through
  `PrecisionPlan`. Don't change `double` to `float` directly.
- **Measure claims.** When citing speedup numbers, use the NVIDIA
  benchmarks from the cuda-writing and gpu-code-review skills, not
  made-up estimates.
- **One file at a time.** If the target imports GPU code from other
  modules, note cross-file findings but don't rewrite imported modules
  without being asked.
- **Respect existing patterns.** Use `runtime.launch()`, `runtime.pointer()`,
  `make_kernel_cache_key()`, and other vibeSpatial conventions. Don't
  introduce raw cuda-python calls.
