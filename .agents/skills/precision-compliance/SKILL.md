---
name: precision-compliance
description: "PROACTIVELY USE THIS SKILL whenever you are writing, modifying, reviewing, or adding a GPU kernel, CUDA kernel, NVRTC kernel source, or any computation in src/vibespatial/ that touches coordinate data on the GPU. This includes work on bounds, distance, point-in-polygon, binary predicates, segment intersection, spatial indexing, overlay, clip, buffer, constructive ops, dissolve, or any new kernel. ADR-0002 requires dual-precision dispatch (fp32/fp64) via PrecisionPlan. This skill contains the full compliance procedure, the precision infrastructure API, kernel-class-specific implementation patterns, and the current compliance status of every kernel file."
---

# Precision Compliance: Wire ADR-0002 Dual-Precision Dispatch Into a Kernel

You are bringing a GPU kernel into compliance with ADR-0002
(dual-precision dispatch). The target kernel is: **$ARGUMENTS**

## The Problem

The precision planning layer (`src/vibespatial/runtime/precision.py`) correctly
computes a `PrecisionPlan` that selects fp32 or fp64 compute precision
based on device profile and kernel class. But many CUDA kernel source
strings are hardcoded to `double` and ignore the plan entirely. On
consumer GPUs with 1:64 fp64:fp32 throughput (CC 8.6/8.9), this means
kernels run at ~1.6% of potential throughput for no correctness benefit.

## Step 1: Read the Target Kernel and Classify It

Read the target file. Determine:

1. **Kernel class** (from `src/vibespatial/runtime/precision.py:KernelClass`):
   - `COARSE`: bounds, filters, sort keys, spatial index — cheap geometry-local work
   - `METRIC`: area, length, distance, centroid — accumulation-heavy reductions
   - `PREDICATE`: point-in-polygon, binary predicates, orientation — boolean classification
   - `CONSTRUCTIVE`: clip, intersection, union, buffer — geometry-producing ops

2. **Current state**: Does it import/use `PrecisionPlan`? Does the CUDA
   source use `double` exclusively? Is there any fp32 path?

3. **ADR-0002 policy for this kernel class on consumer GPUs**:
   - COARSE: staged fp32 with coordinate centering when large absolute magnitudes
   - METRIC: staged fp32 with Kahan-compensated accumulation
   - PREDICATE: staged fp32 for coarse pass, selective fp64 refinement for ambiguous results
   - CONSTRUCTIVE: stay on native fp64 (no change needed — but still wire the plan through for observability)

## Step 2: Read the Precision Infrastructure

Read these files to understand the API you must use:

- `src/vibespatial/runtime/precision.py` — `PrecisionPlan`, `select_precision_plan()`, `KernelClass`, `PrecisionMode`, `CompensationMode`, `CoordinateStats`
- `src/vibespatial/runtime/kernel_registry.py` — `register_kernel_variant()` decorator and `KernelVariantSpec`
- `src/vibespatial/runtime/robustness.py` — `select_robustness_plan()` (takes a PrecisionPlan)

## Step 3: Implement the Precision-Aware Kernel

Follow the pattern appropriate to the kernel class:

### Pattern A: COARSE Kernels (bounds, indexing, filters)

These are pure min/max or comparison operations. fp32 is trivially safe
**EXCEPT for bounds kernels**: bounds outputs are used as spatial filter
inputs, and fp32 rounding can shrink bounds, causing false negatives in
candidate generation. Bounds kernels should stay fp64 or use conservative
rounding (round min DOWN, max UP in fp32). Prefer fp64 for bounds since
they are memory-bound, not compute-bound — fp32 provides no throughput
advantage.

1. **Template the CUDA source** on a precision typedef:
   ```c
   // At top of kernel source string, parameterize:
   typedef {compute_type} compute_t;
   // Replace all `double` in compute paths with `compute_t`
   // Keep buffer reads as `double` (storage is always fp64)
   ```

2. **Add coordinate centering** when `plan.center_coordinates is True`:
   - Compute centroid of the coordinate extent on host
   - Pass `center_x`, `center_y` as kernel parameters
   - In kernel: `compute_t lx = (compute_t)(x[i] - center_x);`

3. **Format the source** based on the plan:
   ```python
   compute_type = "float" if plan.compute_precision is PrecisionMode.FP32 else "double"
   source = _KERNEL_SOURCE_TEMPLATE.format(compute_type=compute_type)
   ```

4. **Cache key must include precision** so fp32 and fp64 variants compile separately:
   ```python
   cache_key = make_kernel_cache_key(f"my-kernel-{plan.compute_precision.value}", source)
   ```

### Pattern B: METRIC Kernels (distance, area, length)

These accumulate values where fp32 cancellation is a risk.

1. **Template the CUDA source** same as Pattern A.

2. **Add Kahan summation** when `plan.compensation is CompensationMode.KAHAN`:
   ```c
   // Instead of: sum += value;
   // Use:
   compute_t kahan_c = (compute_t)0.0;
   // ...
   compute_t y = value - kahan_c;
   compute_t t = sum + y;
   kahan_c = (t - sum) - y;
   sum = t;
   ```

3. **Add coordinate centering** same as Pattern A when `plan.center_coordinates`.

4. **Final result stays fp64**: write accumulated result back as `double` in the output buffer.

### Pattern C: PREDICATE Kernels (point-in-polygon, binary predicates)

These need a two-pass architecture:

1. **Coarse pass in fp32**: Run the winding/crossing test in fp32.
   Mark results as DEFINITE_TRUE, DEFINITE_FALSE, or AMBIGUOUS based on
   margin of confidence (e.g., cross product magnitude relative to scale).

2. **Selective fp64 refinement**: When `plan.refinement is RefinementMode.SELECTIVE_FP64`,
   re-run only AMBIGUOUS rows in fp64. This is the key optimization — most
   rows are geometrically obvious and don't need fp64.

3. **Implementation approach**:
   - Add a `margin` or `confidence` output to the coarse kernel
   - Use a threshold to classify ambiguous cases
   - Compact ambiguous indices (CCCL `select`)
   - Launch fp64 refinement kernel only on ambiguous subset
   - Scatter refined results back

### Pattern D: CONSTRUCTIVE Kernels (clip, overlay, buffer)

Per ADR-0002, constructive kernels stay fp64 on all devices until
robustness work proves a cheaper path. But still:

1. **Wire the plan through** for observability (logging, profiling, result metadata).
2. **Accept `precision: PrecisionMode` parameter** at the public API.
3. **Call `select_precision_plan()`** and attach to result.
4. The plan will always return fp64 for constructive class, so no kernel change needed yet.

## Step 4: Wire the Plan Through the Dispatch Layer

Follow the existing pattern from `binary_predicates.py` / `segment_primitives.py`:

```python
from vibespatial.runtime.precision import (
    KernelClass, PrecisionMode, PrecisionPlan, select_precision_plan,
)
from vibespatial.runtime.robustness import RobustnessPlan, select_robustness_plan

# At the public API entry point:
precision_plan = select_precision_plan(
    runtime_selection=runtime_selection,
    kernel_class=KernelClass.THE_CLASS,
    requested=precision,
    coordinate_stats=coord_stats,  # if available
)
robustness_plan = select_robustness_plan(
    kernel_class=KernelClass.THE_CLASS,
    precision_plan=precision_plan,
)

# Pass plan to kernel launcher:
_launch_kernel(..., precision_plan=precision_plan)

# Include plan in result metadata:
return SomeResult(..., precision_plan=precision_plan, robustness_plan=robustness_plan)
```

## Step 5: Register the Kernel Variant

Use `@register_kernel_variant` with correct metadata:

```python
@register_kernel_variant(
    "kernel_name",
    "gpu-cuda-python",
    kernel_class=KernelClass.THE_CLASS,
    execution_modes=(ExecutionMode.GPU,),
    precision_modes=(PrecisionMode.AUTO, PrecisionMode.FP32, PrecisionMode.FP64),
    preferred_residency=Residency.DEVICE,
)
def _kernel_gpu_impl(...):
    ...
```

## Step 6: Add Tests

Add precision-specific tests following `tests/test_precision_policy.py` pattern:

```python
def test_kernel_respects_fp32_plan():
    """Verify kernel produces correct results with fp32 compute."""
    # Force fp32 precision
    plan = select_precision_plan(
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            reason="test",
        ),
        kernel_class=KernelClass.THE_CLASS,
        requested=PrecisionMode.FP32,
    )
    assert plan.compute_precision is PrecisionMode.FP32
    # Run kernel with fp32 plan, compare against Shapely oracle
    ...

def test_kernel_respects_fp64_plan():
    """Verify kernel still works with explicit fp64."""
    ...
```

Also add a Shapely oracle comparison test that verifies fp32 results
are within acceptable tolerance of the fp64 ground truth.

## Step 7: Verify

```bash
# Run precision policy tests
uv run pytest tests/test_precision_policy.py -v

# Run the kernel's own tests
uv run pytest tests/test_<kernel_name>.py -v

# If the kernel is used in pipelines, run pipeline benchmarks
uv run pytest tests/test_pipeline_benchmarks.py -q
```

## Step 8: Update Compliance Status

After completing the work, you MUST update the compliance ledger in this
skill file so the next session has an accurate starting point.

Edit the "Current Compliance Status" section below:
- Move the file you just fixed from a non-compliant category to the
  "Fully compliant" category.
- If you partially wired precision (e.g., plan computed but no fp32
  kernel variant yet), note that in the entry.
- Update the "as of" date.
- If you added a new kernel file that doesn't exist in the ledger yet,
  add it to the appropriate category.

The compliance ledger file is at:
`.agents/skills/precision-compliance/SKILL.md` (this file — the section below).

## Current Compliance Status (as of 2026-03-16, updated during session)

### Fully compliant (plan computed AND kernel respects it)

- `src/vibespatial/kernels/core/geometry_analysis.py` (COARSE) — precision plan wired through dispatch; GPU kernel is templated on compute_t but bounds intentionally stay fp64 because fp32 rounding shrinks bounds and causes false negatives in spatial filtering. The precision infrastructure is in place for future use if conservative rounding is implemented.
- `src/vibespatial/spatial/point_distance.py` (METRIC) — kernel templated on compute_t with coordinate centering via center_x/center_y parameters. Subtraction happens in fp64 before cast to compute_t. On consumer GPUs, uses staged fp32; on datacenter GPUs, native fp64. The point-in-polygon helper embedded in distance also uses centered fp32.
- `src/vibespatial/kernels/predicates/point_in_polygon.py` (PREDICATE) — all 9 kernel entry points templated on compute_t with centering. Device helpers (point_on_segment, ring_contains_even_odd, polygon/multipolygon_contains_point) use centered fp32 arithmetic. Boundary tolerance widened from 1e-12 to 1e-7 for fp32 noise floor. Dispatch selects compute_type from cached device snapshot. All launcher functions accept compute_type/center params.

- `src/vibespatial/constructive/make_valid_pipeline.py` (PREDICATE) — check_ring_validity kernel templated on compute_t with precision-dependent closure tolerance (1e-24 fp64, 1e-10 fp32). reduce_ring_to_polygon_validity is integer-only. PrecisionPlan wired through dispatch. Both fp32 and fp64 variants precompiled via NVRTC warmup.

### Plan wired at dispatch layer; kernel uses fp64 by design (CONSTRUCTIVE per ADR-0002)

- `src/vibespatial/predicates/binary.py` (PREDICATE) — dispatch uses plan_kernel_dispatch; downstream PIP kernels now use staged fp32
- `src/vibespatial/spatial/segment_primitives.py` (CONSTRUCTIVE) — dispatch uses plan_dispatch_selection; kernel stays fp64
- `src/vibespatial/constructive/clip_rect.py` (CONSTRUCTIVE) — dispatch uses plan_dispatch_selection; kernel stays fp64
- `src/vibespatial/predicates/support.py` (PREDICATE) — dispatch uses plan_kernel_dispatch; downstream PIP kernels now use staged fp32
- `src/vibespatial/overlay/gpu.py` (CONSTRUCTIVE) — fp64 by design per ADR-0002; kernel source in `overlay/gpu_kernels.py`
- `src/vibespatial/constructive/point.py` (CONSTRUCTIVE) — fp64 by design per ADR-0002; dispatch via plan_dispatch_selection
- `src/vibespatial/constructive/linestring.py` (CONSTRUCTIVE) — fp64 by design per ADR-0002; dispatch via plan_dispatch_selection
- `src/vibespatial/constructive/polygon.py` (mixed CONSTRUCTIVE/METRIC) — buffer kernels fp64 by design per ADR-0002; **polygon_centroid kernel (METRIC) fully compliant**: templated on compute_t with Kahan-compensated shoelace accumulation, coordinate centering via center_x/center_y, precision-keyed NVRTC cache (fp32/fp64 variants precompiled). fp32+center+Kahan achieves 8e-4 max error vs Shapely at 1e6-magnitude coords.
- `src/vibespatial/spatial/query.py` (COARSE) — dispatch via plan_kernel_dispatch; no owned CUDA kernel to template
- `src/vibespatial/overlay/dissolve.py` (mixed) — delegates to lower-level GPU ops
- `src/vibespatial/spatial/indexing.py` (COARSE) — dispatch via plan_dispatch_selection; no owned CUDA kernel to template
- `src/vibespatial/constructive/make_valid_gpu.py` (CONSTRUCTIVE) — fp64 by design per ADR-0002; GPU repair kernels (close_rings, flag_duplicate_vertices, reverse_ring_coords, split event kernels) all use fp64 storage and compute

### Priority order (highest performance impact first)

1. COARSE kernels (bounds, indexing) — trivial fp32 safety, pure min/max
2. METRIC kernels (distance) — fp32 + Kahan is well-understood
3. PREDICATE kernels (point-in-polygon, binary predicates) — needs two-pass architecture
4. CONSTRUCTIVE kernels — wire plan through for observability only

## Key Rules

- Storage is ALWAYS fp64. Never change buffer storage precision.
- Compute precision is selected at dispatch time via `PrecisionPlan`.
- Kernel cache keys MUST include precision so fp32/fp64 compile separately.
- Coordinate centering is an execution-local artifact, not a buffer change.
- On datacenter GPUs (fp64:fp32 >= 0.25), the plan will select fp64 for everything — the fp32 path only activates on consumer hardware.
- Never add ad-hoc precision booleans. Always go through `PrecisionPlan`.
- Per ADR-0033, custom NVRTC kernels (Tier 1) are the right tool for geometry-specific compute. The precision parameterization goes into these kernel source strings.
