---
name: new-kernel-checklist
description: "PROACTIVELY USE THIS SKILL when adding a new GPU kernel, NVRTC kernel, CCCL primitive wrapper, or any new GPU-dispatched operation to src/vibespatial/. This checklist ensures every registration, warmup, caching, precision, dispatch, test, benchmark, and documentation step is completed. The cuda-engineer agent and gpu-code-review skill should reference this checklist. Trigger on: \"new kernel\", \"add kernel\", \"implement kernel\", \"write kernel\", \"create kernel\", \"new GPU operation\", \"add GPU\", \"scaffold kernel\"."
user-invocable: true
argument-hint: <kernel-name or description of the new operation>
---

# New Kernel Checklist — vibeSpatial

You are adding a new GPU kernel or GPU-dispatched operation. This checklist
ensures nothing is missed. Work through each section in order — every item
marked **[REQUIRED]** must be completed before the kernel can land.

Target kernel: **$ARGUMENTS**

---

## Phase 1: Classification and Design

### 1.1 Tier Classification (ADR-0033) [REQUIRED]

Run the decision tree to determine the correct tier:

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

Then for CCCL (Tier 3a):
- Can input use `CountingIterator`/`TransformIterator`? -> Tier 3c iterator
- Called repeatedly with same types/ops? -> Tier 3b `make_*`

Record: `Tier: ___  Rationale: ___`

### 1.2 Kernel Class (ADR-0002) [REQUIRED]

Classify precision behavior using `KernelClass` from
`src/vibespatial/runtime/precision.py`:

| Class | When | Precision Policy (Consumer GPU) |
|-------|------|---------------------------------|
| `COARSE` | Bounds, filters, sort keys, spatial index | Staged fp32 + centering (but bounds stay fp64) |
| `METRIC` | Area, length, distance, centroid | Staged fp32 + Kahan compensation + centering |
| `PREDICATE` | Point-in-polygon, binary predicates | Staged fp32 coarse pass + selective fp64 refinement |
| `CONSTRUCTIVE` | Clip, overlay, buffer, union | Native fp64 (wire plan for observability only) |

Record: `KernelClass: ___`

### 1.3 Geometry Families [REQUIRED]

List which geometry types this kernel handles:
`point`, `multipoint`, `linestring`, `multilinestring`, `polygon`, `multipolygon`

Record: `geometry_families: (___)`

---

## Phase 2: Kernel Source (Tier 1 only — skip for Tier 2/3/4)

### 2.1 Write Kernel Source String [REQUIRED for Tier 1]

Define as a module-level constant. Key rules:

- [ ] Source is a raw string constant: `_KERNEL_SOURCE = r""" ... """`
- [ ] Use `typedef {compute_type} compute_t;` for precision templating
- [ ] All compute paths use `compute_t`, not hardcoded `double`
- [ ] Storage reads stay `double`: `compute_t lx = (compute_t)(x[i] - center_x);`
- [ ] All read-only pointers use `const ... * __restrict__`
- [ ] Use `__launch_bounds__(threads, min_blocks)` on kernel functions
- [ ] Use SoA layout (separate `x[]`, `y[]`), never AoS
- [ ] Float constants use type-correct suffix: `(compute_t)0.0`, `(compute_t)1e-7`
- [ ] Integer division by power-of-2 uses bitwise: `>> 5` not `/ 32`

### 2.2 Define Kernel Names Tuple [REQUIRED for Tier 1]

```python
_KERNEL_NAMES = ("my_kernel_main", "my_kernel_cooperative")
```

Every `extern "C" __global__` function in the source must appear here.

### 2.3 Precision Variants [REQUIRED for Tier 1]

Create fp64 and fp32 variants of the source:

```python
_KERNEL_FP64 = _KERNEL_SOURCE_TEMPLATE.format(compute_type="double")
_KERNEL_FP32 = _KERNEL_SOURCE_TEMPLATE.format(compute_type="float")
```

For METRIC kernels, also include Kahan summation macros and centering
macros (`CX(val)`, `CY(val)`) in the fp32 variant.

---

## Phase 3: Compilation and Caching

### 3.1 Kernel Compilation (Tier 1) [REQUIRED for Tier 1]

Use the standard compilation pattern:

```python
from vibespatial.cuda._runtime import get_cuda_runtime, make_kernel_cache_key

def _compile_my_kernel(compute_type: str = "double"):
    source = _KERNEL_FP64 if compute_type == "double" else _KERNEL_FP32
    suffix = "fp64" if compute_type == "double" else "fp32"
    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key(f"my-kernel-{suffix}", source)
    return runtime.compile_kernels(
        cache_key=cache_key,
        source=source,
        kernel_names=_KERNEL_NAMES,
    )
```

- [ ] Cache key includes precision suffix (`-fp64` / `-fp32`)
- [ ] Uses `make_kernel_cache_key()` (SHA1-based)
- [ ] Uses `runtime.compile_kernels()` (memory + disk cached)

### 3.2 CCCL Primitive Wrapper (Tier 3) [REQUIRED for Tier 3]

If adding a new CCCL primitive to `cccl_primitives.py`:

- [ ] Wrap behind a strategy enum (e.g., `MyStrategy.AUTO/CCCL/CUPY`)
- [ ] Implement `make_*` fast path checking `CCCLPrecompiler.get().get_compiled()`
- [ ] Implement one-shot fallback via `cccl_algorithms.*()` API
- [ ] Add `synchronize` parameter (default `True`)
- [ ] Benchmark `make_*` vs CuPy before setting AUTO default

---

## Phase 4: Warmup and Precompilation Registration (ADR-0034)

### 4.1 NVRTC Warmup (Tier 1) [REQUIRED for Tier 1]

At module scope (after defining source constants), register for
background precompilation:

```python
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup

request_nvrtc_warmup([
    ("my-kernel-fp64", _KERNEL_FP64, _KERNEL_NAMES),
    ("my-kernel-fp32", _KERNEL_FP32, _KERNEL_NAMES),
])
```

- [ ] Called at module scope (not inside a function)
- [ ] Registers BOTH fp64 and fp32 variants
- [ ] Prefix matches the cache key prefix used in compilation
- [ ] Kernel names tuple matches exactly

### 4.2 Add to NVRTC Consumer Modules List [REQUIRED for Tier 1]

Add your module to `_NVRTC_CONSUMER_MODULES` in
`src/vibespatial/cuda/cccl_precompile.py` (line ~1066):

```python
_NVRTC_CONSUMER_MODULES: tuple[str, ...] = (
    ...
    "vibespatial.my_module.my_kernel",  # <-- ADD HERE
    ...
)
```

This ensures `precompile_all()` (used in CI and post-install) discovers
and compiles your kernel.

### 4.3 CCCL Warmup Specs (Tier 3b/3c) [REQUIRED if using make_*]

If your kernel uses CCCL `make_*` callables:

**Step A — Add specs to `SPEC_REGISTRY`** in
`src/vibespatial/cuda/cccl_precompile.py`, inside `_build_spec_registry()`:

```python
CCCLWarmupSpec(
    name="my_primitive_i32",
    family=AlgorithmFamily.MY_PRIMITIVE,
    key_dtype=np.int32,
    value_dtype=np.int32,
    op_name="sum",
)
```

- [ ] One spec per unique (algorithm, dtype, op) combination
- [ ] Name follows convention: `{algorithm}_{dtype}` (e.g., `exclusive_scan_i32`)

**Step B — Request warmup at module scope** in your consumer module:

```python
from vibespatial.cuda.cccl_precompile import request_warmup

request_warmup(["my_primitive_i32", "my_primitive_i64"])
```

- [ ] Called at module scope
- [ ] Lists only the specs THIS module actually uses
- [ ] Idempotent (safe if multiple modules request the same spec)

### 4.4 CCCL CUBIN Disk Cache [CONDITIONAL — if using make_*]

If your `make_*` family is cacheable (no embedded LTOIR / Numba closures):

- [ ] Add a `_Cached{Family}` wrapper class in `cccl_cubin_cache.py`
- [ ] Add ctypes struct definition matching the C `build_result` layout
- [ ] Register the family in `extract_cache_entry()` and `reconstruct_build()`
- [ ] Test with `VIBESPATIAL_CCCL_CACHE=1` and verify cache hits

Note: `select` and `segmented_sort` are NOT cacheable due to embedded
LTOIR pointers. If your algorithm bakes closures into the build result,
it falls into this category.

---

## Phase 5: Dispatch and Precision Wiring

### 5.1 Kernel Variant Registration [REQUIRED]

Register both GPU and CPU variants using `@register_kernel_variant()`:

```python
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode
from vibespatial.runtime._runtime import ExecutionMode
from vibespatial.runtime.residency import Residency

@register_kernel_variant(
    "my_kernel",
    "gpu-cuda-python",
    kernel_class=KernelClass.THE_CLASS,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("point", "polygon"),
    supports_mixed=True,
    precision_modes=(PrecisionMode.AUTO, PrecisionMode.FP32, PrecisionMode.FP64),
    preferred_residency=Residency.DEVICE,
    tags=("cuda-python",),
)
def _my_kernel_gpu(owned, *, precision_plan=None):
    ...

@register_kernel_variant(
    "my_kernel",
    "cpu",
    kernel_class=KernelClass.THE_CLASS,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=("point", "polygon"),
    supports_mixed=True,
    tags=("shapely",),
)
def _my_kernel_cpu(owned):
    ...
```

- [ ] GPU variant has `preferred_residency=Residency.DEVICE`
- [ ] GPU variant lists all three `precision_modes`
- [ ] CPU variant exists as Shapely fallback
- [ ] `kernel_class` matches your classification from Phase 1
- [ ] `geometry_families` matches your list from Phase 1

### 5.2 Precision Plan at Public API [REQUIRED]

At the public entry point, compute and pass through the precision plan:

```python
from vibespatial.runtime.precision import select_precision_plan, KernelClass
from vibespatial.runtime.adaptive import plan_dispatch_selection

selection = plan_dispatch_selection(
    kernel_name="my_kernel",
    kernel_class=KernelClass.THE_CLASS,
    row_count=n,
    requested_mode=dispatch_mode,
)
precision_plan = select_precision_plan(
    runtime_selection=selection,
    kernel_class=KernelClass.THE_CLASS,
    requested=precision,
    coordinate_stats=coord_stats,  # if available
)
```

- [ ] `select_precision_plan()` called before kernel launch
- [ ] Plan passed to GPU kernel implementation
- [ ] Plan included in result metadata (if applicable)
- [ ] Public API accepts `precision: PrecisionMode | str = PrecisionMode.AUTO`

### 5.3 Kernel Launch (Tier 1) [REQUIRED for Tier 1]

```python
runtime = get_cuda_runtime()
grid, block = runtime.launch_config(kernels["my_kernel"], item_count)
runtime.launch(kernels["my_kernel"], grid=grid, block=block, params=params)
```

- [ ] Uses `runtime.launch_config()` for occupancy-based sizing (NEVER hardcode block)
- [ ] Uses correct parameter type constants (`KERNEL_PARAM_PTR`, `KERNEL_PARAM_I32`, `KERNEL_PARAM_F64`)
- [ ] No unnecessary `runtime.synchronize()` between same-stream operations
- [ ] Sync only before host reads of device data

---

## Phase 6: Tests

### 6.1 Unit Tests [REQUIRED]

Create `tests/test_{kernel_name}.py`:

- [ ] Uses `@compare_with_shapely(reference=ref_func)` decorator
- [ ] Parametrized with `dispatch_mode` fixture (tests both CPU and GPU)
- [ ] Tests null/empty/mixed geometry inputs
- [ ] Uses `oracle_runner` fixture for Shapely comparison
- [ ] Handles `NotImplementedError` for scaffold stubs via `pytest.xfail`
- [ ] Marked with `@pytest.mark.gpu` for GPU-specific tests

### 6.2 Precision Tests [REQUIRED]

- [ ] Test with explicit `PrecisionMode.FP32` — verify correct results
- [ ] Test with explicit `PrecisionMode.FP64` — verify correct results
- [ ] fp32 results within acceptable tolerance of fp64 ground truth
- [ ] If PREDICATE: test that ambiguous-case refinement produces correct results

### 6.3 Edge Cases [REQUIRED]

- [ ] Empty geometry collections
- [ ] Single-element inputs
- [ ] Very large inputs (>1M rows if applicable)
- [ ] Degenerate geometries (zero-area polygons, coincident points, etc.)

---

## Phase 7: Benchmarks

### 7.1 Operation Benchmark [RECOMMENDED]

Add an entry in `src/vibespatial/bench/operations/{category}_ops.py`:

```python
from vibespatial.bench.catalog import benchmark_operation

@benchmark_operation(
    name="my_kernel",
    description="Description of what is benchmarked",
    category="predicate",  # or constructive, spatial, overlay, io, misc
    geometry_types=("point", "polygon"),
    default_scale="100K",
    tier=1,
)
def bench_my_kernel(scale, precision, **kwargs):
    ...
```

- [ ] Registered via `@benchmark_operation()` decorator
- [ ] Discoverable via `vsbench list operations`

### 7.2 Kernel Microbenchmark (NVBench) [RECOMMENDED for Tier 1]

Create `src/vibespatial/bench/kernels/bench_{kernel_name}.py`:

- [ ] Auto-discovered by `nvbench_runner.py` via `bench_*.py` naming
- [ ] Supports `--scale`, `--precision`, `--output-json` CLI args
- [ ] Reports bandwidth metrics if memory-bound

---

## Phase 8: Manifest and Documentation

### 8.1 Variant Manifest [REQUIRED]

Update `src/vibespatial/kernels/variant_manifest.json`:

```json
{
  "kernel": "my_kernel",
  "module": "vibespatial.kernels.predicates",
  "tier": 1,
  "geom_types": ["point", "polygon"],
  "variants": ["cpu", "gpu-cuda-python"]
}
```

Or run the scaffold generator which does this automatically:
```bash
uv run python scripts/generate_kernel_scaffold.py my_kernel --tier 1 --geom-types point,polygon
```

### 8.2 Kernel Inventory Doc [REQUIRED]

Ensure `docs/testing/kernel-inventory.md` has a row for the new kernel.
The scaffold generator updates this automatically.

### 8.3 Package Exports [REQUIRED]

If the kernel module is in a sub-package under `src/vibespatial/kernels/`,
ensure the `__init__.py` exports it:

```python
# src/vibespatial/kernels/{subpackage}/__init__.py
from .my_kernel import my_kernel
```

---

## Phase 9: Quality Gates

### 9.1 Run /cuda-optimizer [REQUIRED]

Invoke the `cuda-optimizer` skill on your kernel file to verify:
- No unnecessary host round-trips
- No redundant synchronizations
- Efficient memory access patterns
- Correct use of streams

### 9.2 Run /precision-compliance [REQUIRED]

Invoke the `precision-compliance` skill to verify:
- Kernel source is templated on `compute_t`
- Cache key includes precision
- Centering and compensation wired correctly
- Update the compliance ledger in that skill

### 9.3 Run /gpu-code-review [REQUIRED]

Invoke the `gpu-code-review` skill to catch:
- Anti-patterns (hardcoded block sizes, missing `__restrict__`, AoS layout)
- Memory management issues
- Synchronization bugs
- Performance regressions

### 9.4 Run /pre-land-review [REQUIRED before commit]

Final gate. Must pass before creating a git commit.

---

## Quick Reference: Files to Touch

| What | File(s) |
|------|---------|
| Kernel source | `src/vibespatial/{module}/{kernel}.py` |
| NVRTC warmup call | Same file as kernel source (module scope) |
| Consumer modules list | `src/vibespatial/cuda/cccl_precompile.py` (`_NVRTC_CONSUMER_MODULES`) |
| CCCL warmup specs | `src/vibespatial/cuda/cccl_precompile.py` (`_build_spec_registry`) |
| CCCL warmup call | Consumer module (module scope) |
| CCCL disk cache | `src/vibespatial/cuda/cccl_cubin_cache.py` (if new family) |
| CCCL primitive wrapper | `src/vibespatial/cuda/cccl_primitives.py` (if new primitive) |
| Variant registration | Same file as kernel source (decorator) |
| Dispatch wiring | Public API module |
| Tests | `tests/test_{kernel}.py` |
| Precision tests | `tests/test_{kernel}.py` or `tests/test_precision_policy.py` |
| Operation benchmark | `src/vibespatial/bench/operations/{category}_ops.py` |
| Kernel benchmark | `src/vibespatial/bench/kernels/bench_{kernel}.py` |
| Variant manifest | `src/vibespatial/kernels/variant_manifest.json` |
| Kernel inventory | `docs/testing/kernel-inventory.md` |
| Package `__init__.py` | `src/vibespatial/kernels/{subpackage}/__init__.py` |

---

## Conditional Checklist Summary

Use this table to determine which phases apply based on your tier:

| Phase | Tier 1 (NVRTC) | Tier 2 (CuPy) | Tier 3 (CCCL) | Tier 4 (CuPy default) |
|-------|:-:|:-:|:-:|:-:|
| 1. Classification | YES | YES | YES | YES |
| 2. Kernel Source | YES | skip | skip | skip |
| 3.1 NVRTC Compilation | YES | skip | skip | skip |
| 3.2 CCCL Wrapper | skip | skip | YES | skip |
| 4.1 NVRTC Warmup | YES | skip | skip | skip |
| 4.2 Consumer Modules | YES | skip | skip | skip |
| 4.3 CCCL Warmup Specs | skip | skip | if make_* | skip |
| 4.4 CCCL Disk Cache | skip | skip | if cacheable | skip |
| 5. Dispatch & Precision | YES | YES | YES | YES |
| 6. Tests | YES | YES | YES | YES |
| 7. Benchmarks | YES | optional | optional | optional |
| 8. Manifest & Docs | YES | YES | YES | YES |
| 9. Quality Gates | YES | YES | YES | YES |
