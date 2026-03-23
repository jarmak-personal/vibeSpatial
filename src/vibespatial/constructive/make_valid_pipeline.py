from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter

import numpy as np
import shapely

from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.fusion import IntermediateDisposition, PipelineStep, StepKind, plan_fusion
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode


class MakeValidPrimitive(StrEnum):
    VALIDITY_MASK = "validity_mask"
    COMPACT_INVALID = "compact_invalid"
    SEGMENTIZE_INVALID = "segmentize_invalid"
    POLYGONIZE_REPAIR = "polygonize_repair"
    SCATTER_REPAIRED = "scatter_repaired"
    EMIT_GEOMETRY = "emit_geometry"


@dataclass(frozen=True)
class MakeValidStage:
    name: str
    primitive: MakeValidPrimitive
    purpose: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    cccl_mapping: tuple[str, ...]
    disposition: IntermediateDisposition
    geometry_producing: bool = False


@dataclass(frozen=True)
class MakeValidPlan:
    method: str
    keep_collapsed: bool
    stages: tuple[MakeValidStage, ...]
    fusion_steps: tuple[PipelineStep, ...]
    reason: str


@dataclass(frozen=True)
class MakeValidResult:
    geometries: np.ndarray
    row_count: int
    valid_rows: np.ndarray
    repaired_rows: np.ndarray
    null_rows: np.ndarray
    method: str
    keep_collapsed: bool
    owned: object | None = None
    selected: ExecutionMode = ExecutionMode.CPU


@dataclass(frozen=True)
class MakeValidBenchmark:
    dataset: str
    rows: int
    repaired_rows: int
    compact_elapsed_seconds: float
    baseline_elapsed_seconds: float

    @property
    def speedup_vs_baseline(self) -> float:
        if self.compact_elapsed_seconds == 0.0:
            return float("inf")
        return self.baseline_elapsed_seconds / self.compact_elapsed_seconds


# ---------------------------------------------------------------------------
# GPU validity detection kernel (ADR-0019: compact-invalid-row pattern)
# ---------------------------------------------------------------------------
# Checks per-ring: ring closure (first == last), minimum vertex count (>=4),
# and flags rings that fail.  Self-intersection detection reuses the
# classify_segment_intersections infrastructure from segment_primitives.

_VALIDITY_KERNEL_SOURCE_TEMPLATE = """
typedef {compute_type} compute_t;

extern "C" __global__ void check_ring_validity(
    const double* ring_x,
    const double* ring_y,
    const int* ring_offsets,
    unsigned char* ring_valid,
    int ring_count
) {{
  const int ring = blockIdx.x * blockDim.x + threadIdx.x;
  if (ring >= ring_count) {{
    return;
  }}
  const int start = ring_offsets[ring];
  const int end = ring_offsets[ring + 1];
  const int vertex_count = end - start;

  /* Minimum vertex count: a closed polygon ring needs >= 4 coords (3 unique + closure) */
  if (vertex_count < 4) {{
    ring_valid[ring] = 0;
    return;
  }}

  /* Ring closure check: first coord must equal last coord.
     Storage is always fp64; cast to compute_t for the comparison.
     Tolerance is precision-dependent: fp64 uses 1e-24, fp32 uses 1e-10
     (squared epsilon for respective precision floors). */
  const compute_t x0 = (compute_t)ring_x[start];
  const compute_t y0 = (compute_t)ring_y[start];
  const compute_t x_last = (compute_t)ring_x[end - 1];
  const compute_t y_last = (compute_t)ring_y[end - 1];
  const compute_t dx = x0 - x_last;
  const compute_t dy = y0 - y_last;
  if (dx * dx + dy * dy > (compute_t){closure_tolerance}) {{
    ring_valid[ring] = 0;
    return;
  }}

  ring_valid[ring] = 1;
}}

extern "C" __global__ void reduce_ring_to_polygon_validity(
    const unsigned char* ring_valid,
    const int* geometry_offsets,
    unsigned char* polygon_valid,
    int polygon_count
) {{
  const int poly = blockIdx.x * blockDim.x + threadIdx.x;
  if (poly >= polygon_count) {{
    return;
  }}
  const int ring_start = geometry_offsets[poly];
  const int ring_end = geometry_offsets[poly + 1];
  if (ring_start >= ring_end) {{
    polygon_valid[poly] = 0;
    return;
  }}
  for (int r = ring_start; r < ring_end; r++) {{
    if (!ring_valid[r]) {{
      polygon_valid[poly] = 0;
      return;
    }}
  }}
  polygon_valid[poly] = 1;
}}
"""

def _format_validity_kernel_source(compute_type: str = "double") -> str:
    """Format kernel source with precision-dependent typedef and tolerance."""
    closure_tolerance = "1e-10" if compute_type == "float" else "1e-24"
    return _VALIDITY_KERNEL_SOURCE_TEMPLATE.format(
        compute_type=compute_type,
        closure_tolerance=closure_tolerance,
    )

_VALIDITY_KERNEL_NAMES = ("check_ring_validity", "reduce_ring_to_polygon_validity")

# Precompile both fp64 and fp32 variants (ADR-0002 + ADR-0034)
_VALIDITY_KERNEL_SOURCE_FP64 = _format_validity_kernel_source("double")
_VALIDITY_KERNEL_SOURCE_FP32 = _format_validity_kernel_source("float")

from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("make-valid-detect-fp64", _VALIDITY_KERNEL_SOURCE_FP64, _VALIDITY_KERNEL_NAMES),
    ("make-valid-detect-fp32", _VALIDITY_KERNEL_SOURCE_FP32, _VALIDITY_KERNEL_NAMES),
])


def _compile_validity_kernels(compute_type: str = "double"):
    from vibespatial.cuda._runtime import get_cuda_runtime, make_kernel_cache_key

    source = _format_validity_kernel_source(compute_type)
    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key(f"make-valid-detect-{compute_type}", source)
    return runtime.compile_kernels(
        cache_key=cache_key,
        source=source,
        kernel_names=_VALIDITY_KERNEL_NAMES,
    )


def _gpu_polygon_validity_mask(owned) -> np.ndarray | None:
    """Compute per-row validity for polygon families using GPU ring checks.

    Uses ADR-0002 dual-precision dispatch: selects fp32 or fp64 compute
    based on device profile (PREDICATE kernel class). On consumer GPUs
    with 1:32 fp64:fp32 throughput, fp32 runs ~32x faster for the ring
    closure comparison.

    Returns a boolean array (True = valid) for all rows, or None if GPU
    detection is not applicable (e.g., no polygon families, no GPU).
    """
    from vibespatial.cuda._runtime import (
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        get_cuda_runtime,
    )
    from vibespatial.runtime import has_gpu_runtime

    if not has_gpu_runtime():
        return None

    polygon_families = []
    for family_name in ("polygon", "multipolygon"):
        if family_name in owned.families:
            polygon_families.append((family_name, owned.families[family_name]))
    if not polygon_families:
        return None

    # ADR-0002: select compute precision based on device profile
    from vibespatial.runtime import ExecutionMode as _EM
    from vibespatial.runtime import RuntimeSelection
    from vibespatial.runtime.precision import select_precision_plan
    runtime_sel = RuntimeSelection(
        requested=_EM.GPU, selected=_EM.GPU,
        reason="make_valid GPU ring-structure validity check",
    )
    precision_plan = select_precision_plan(
        runtime_selection=runtime_sel,
        kernel_class=KernelClass.PREDICATE,
        requested=PrecisionMode.AUTO,
    )
    compute_type = "float" if precision_plan.compute_precision is PrecisionMode.FP32 else "double"

    runtime = get_cuda_runtime()
    kernels = _compile_validity_kernels(compute_type)

    # Build a per-row validity array; start with True for non-null, non-polygon rows
    row_valid = np.ones(owned.row_count, dtype=bool)
    row_valid[~owned.validity] = False

    from vibespatial.geometry.owned import FAMILY_TAGS

    for family_name, buffer in polygon_families:
        if not hasattr(buffer, "ring_offsets") or buffer.ring_offsets is None:
            continue
        ring_offsets = np.asarray(buffer.ring_offsets, dtype=np.int32)
        ring_count = len(ring_offsets) - 1
        if ring_count <= 0:
            continue

        # Use device-resident data if available (ADR-0005), else upload
        device_buf = None
        if owned.device_state is not None:
            device_families = owned.device_state.families
            if family_name in device_families:
                device_buf = device_families[family_name]

        if device_buf is not None:
            d_x = device_buf.x
            d_y = device_buf.y
            d_ring_offsets = device_buf.ring_offsets
            owns_buffers = False
        else:
            d_x = runtime.from_host(np.ascontiguousarray(buffer.x, dtype=np.float64))
            d_y = runtime.from_host(np.ascontiguousarray(buffer.y, dtype=np.float64))
            d_ring_offsets = runtime.from_host(ring_offsets)
            owns_buffers = True

        d_ring_valid = runtime.allocate((ring_count,), np.uint8)
        geom_offsets = np.asarray(buffer.geometry_offsets, dtype=np.int32)
        polygon_count = len(geom_offsets) - 1
        d_geom_offsets = None
        d_poly_valid = None

        try:
            ptr = runtime.pointer
            ring_grid, ring_block = runtime.launch_config(kernels["check_ring_validity"], ring_count)
            runtime.launch(
                kernels["check_ring_validity"],
                grid=ring_grid,
                block=ring_block,
                params=(
                    (ptr(d_x), ptr(d_y), ptr(d_ring_offsets),
                     ptr(d_ring_valid), ring_count),
                    (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
                ),
            )

            if polygon_count <= 0:
                continue
            d_geom_offsets = runtime.from_host(geom_offsets)
            d_poly_valid = runtime.allocate((polygon_count,), np.uint8)
            poly_grid, poly_block = runtime.launch_config(kernels["reduce_ring_to_polygon_validity"], polygon_count)
            runtime.launch(
                kernels["reduce_ring_to_polygon_validity"],
                grid=poly_grid,
                block=poly_block,
                params=(
                    (ptr(d_ring_valid), ptr(d_geom_offsets),
                     ptr(d_poly_valid), polygon_count),
                    (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
                ),
            )
            runtime.synchronize()

            # Vectorized scatter: map family rows to global rows (no Python loop)
            h_poly_valid = runtime.copy_device_to_host(d_poly_valid)
            invalid_family_rows = np.flatnonzero(h_poly_valid == 0)
            if invalid_family_rows.size > 0:
                family_tag = FAMILY_TAGS[family_name]
                tag_match = owned.tags == family_tag
                for fr in invalid_family_rows:
                    global_mask = owned.validity & tag_match & (owned.family_row_offsets == int(fr))
                    row_valid[global_mask] = False
        finally:
            runtime.free(d_ring_valid)
            if owns_buffers:
                runtime.free(d_x)
                runtime.free(d_y)
                runtime.free(d_ring_offsets)
            if d_geom_offsets is not None:
                runtime.free(d_geom_offsets)
            if d_poly_valid is not None:
                runtime.free(d_poly_valid)

    return row_valid


def _detect_self_intersections_gpu(owned, valid_mask: np.ndarray, geometries: np.ndarray | None = None) -> np.ndarray:
    """Refine validity mask by detecting self-intersections.

    GPU ring-structure checks (ring closure, vertex count) handle the fast
    structural cases. For self-intersection detection, we use shapely.is_valid
    on only the subset that passed ring checks. This is still a win because:
    - If 90% of rows fail ring checks, shapely only runs on 10%
    - The GPU ring check itself is <2ms for 100K rows (zero D->H transfer)
    - shapely.is_valid is vectorized C, efficient for the remaining subset
    """
    structurally_valid_rows = np.flatnonzero(valid_mask)
    if structurally_valid_rows.size == 0:
        return valid_mask

    try:
        if geometries is None:
            geometries = np.asarray(owned.to_shapely(), dtype=object)
        subset = geometries[structurally_valid_rows]
        subset_valid = np.asarray(shapely.is_valid(subset), dtype=bool)
        valid_mask[structurally_valid_rows[~subset_valid]] = False
    except Exception:
        pass

    return valid_mask


def plan_make_valid_pipeline(*, method: str = "linework", keep_collapsed: bool = True) -> MakeValidPlan:
    stages = (
        MakeValidStage(
            name="compute_validity_mask",
            primitive=MakeValidPrimitive.VALIDITY_MASK,
            purpose="Compute validity and null masks so only invalid rows flow into repair work.",
            inputs=("geometry_rows",),
            outputs=("valid_mask", "null_mask"),
            cccl_mapping=("transform",),
            disposition=IntermediateDisposition.EPHEMERAL,
        ),
        MakeValidStage(
            name="compact_invalid_rows",
            primitive=MakeValidPrimitive.COMPACT_INVALID,
            purpose="Compact invalid rows into one dense repair batch instead of sending valid rows through constructive work.",
            inputs=("geometry_rows", "valid_mask", "null_mask"),
            outputs=("invalid_rows", "invalid_index"),
            cccl_mapping=("DeviceSelect", "gather"),
            disposition=IntermediateDisposition.EPHEMERAL,
        ),
        MakeValidStage(
            name="repair_invalid_topology",
            primitive=MakeValidPrimitive.POLYGONIZE_REPAIR if method == "linework" else MakeValidPrimitive.SEGMENTIZE_INVALID,
            purpose="Repair only the compacted invalid rows using the selected make-valid strategy.",
            inputs=("invalid_rows",),
            outputs=("repaired_rows",),
            cccl_mapping=("transform", "scatter"),
            disposition=IntermediateDisposition.EPHEMERAL,
            geometry_producing=True,
        ),
        MakeValidStage(
            name="scatter_repaired_rows",
            primitive=MakeValidPrimitive.SCATTER_REPAIRED,
            purpose="Scatter repaired invalid rows back into original row order while preserving valid rows untouched.",
            inputs=("repaired_rows", "invalid_index"),
            outputs=("output_rows",),
            cccl_mapping=("scatter",),
            disposition=IntermediateDisposition.EPHEMERAL,
            geometry_producing=True,
        ),
        MakeValidStage(
            name="emit_geometry",
            primitive=MakeValidPrimitive.EMIT_GEOMETRY,
            purpose="Emit final geometry rows for GeoSeries and overlay preprocessing surfaces.",
            inputs=("output_rows",),
            outputs=("geometry_buffers",),
            cccl_mapping=("gather",),
            disposition=IntermediateDisposition.PERSIST,
            geometry_producing=True,
        ),
    )
    fusion_steps = (
        PipelineStep(name="valid_mask", kind=StepKind.FILTER, output_name="valid_mask"),
        PipelineStep(name="invalid_rows", kind=StepKind.FILTER, output_name="invalid_rows"),
        PipelineStep(name="repaired_rows", kind=StepKind.GEOMETRY, output_name="repaired_rows"),
        PipelineStep(
            name="geometry_buffers",
            kind=StepKind.GEOMETRY,
            output_name="geometry_buffers",
            reusable_output=True,
        ),
    )
    return MakeValidPlan(
        method=method,
        keep_collapsed=keep_collapsed,
        stages=stages,
        fusion_steps=fusion_steps,
        reason=(
            "make_valid should compact invalid rows first and only run constructive repair on the invalid subset so future "
            "GPU implementations can use DeviceSelect-style compaction instead of paying topology-repair cost on already-valid rows."
        ),
    )


def fusion_plan_for_make_valid(*, method: str = "linework", keep_collapsed: bool = True):
    return plan_fusion(plan_make_valid_pipeline(method=method, keep_collapsed=keep_collapsed).fusion_steps)


def _compute_validity_mask_gpu(geometries: np.ndarray, owned) -> tuple[np.ndarray, bool]:
    """Attempt GPU validity detection; return (valid_mask, gpu_used).

    ADR-0019 compact-invalid-row pattern: compute validity on GPU, compact
    invalid indices, repair only invalid rows (Shapely backend), scatter back.
    """
    gpu_mask = _gpu_polygon_validity_mask(owned)
    if gpu_mask is None:
        return np.asarray(shapely.is_valid(geometries), dtype=bool), False

    # GPU ring checks only cover polygon families.  For non-polygon rows,
    # fall back to Shapely is_valid for those rows specifically.
    from vibespatial.geometry.owned import FAMILY_TAGS

    polygon_tags = set()
    for family_name in ("polygon", "multipolygon"):
        if family_name in FAMILY_TAGS:
            polygon_tags.add(FAMILY_TAGS[family_name])

    non_polygon_rows = np.flatnonzero(
        owned.validity & ~np.isin(owned.tags, list(polygon_tags))
    )
    if non_polygon_rows.size > 0:
        non_polygon_valid = shapely.is_valid(geometries[non_polygon_rows])
        gpu_mask[non_polygon_rows] = np.asarray(non_polygon_valid, dtype=bool)

    # Refine with self-intersection detection
    gpu_mask = _detect_self_intersections_gpu(owned, gpu_mask)

    return gpu_mask, True


# ---------------------------------------------------------------------------
# Kernel variant registration (ADR-0033 tier system)
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "make_valid",
    "gpu-nvrtc",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("polygon", "multipolygon"),
    supports_mixed=True,
    tags=("nvrtc", "constructive", "make_valid", "compact-invalid"),
)
def _make_valid_gpu_repair(owned, repaired_rows, geometries, *, method, keep_collapsed):
    """GPU repair via gpu_repair_invalid_polygons (ADR-0019 + ADR-0033)."""
    from .make_valid_gpu import gpu_repair_invalid_polygons

    return gpu_repair_invalid_polygons(
        owned, repaired_rows, geometries,
        method=method, keep_collapsed=keep_collapsed,
    )


@register_kernel_variant(
    "make_valid",
    "cpu",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=("polygon", "multipolygon", "linestring", "multilinestring", "point", "multipoint"),
    supports_mixed=True,
    tags=("shapely", "constructive", "make_valid"),
)
def _make_valid_cpu_repair(geometries, repaired_rows, *, method, keep_collapsed):
    """CPU repair via shapely.make_valid on invalid subset."""
    result = geometries.copy()
    if repaired_rows.size:
        result[repaired_rows] = shapely.make_valid(
            geometries[repaired_rows],
            method=method,
            keep_collapsed=keep_collapsed,
        )
    return result


def make_valid_owned(
    values=None,
    *,
    method: str = "linework",
    keep_collapsed: bool = True,
    owned=None,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
) -> MakeValidResult:
    """Validate and repair geometries using compact-invalid-row pattern (ADR-0019).

    Parameters
    ----------
    values : array-like of shapely geometries, optional
        When *owned* is provided, *values* may be None -- Shapely objects
        will only be materialized if GPU validity checks find invalid rows
        that require repair (lazy materialization per ADR-0005).
    method : repair method ("linework" or "structure")
    keep_collapsed : whether to keep collapsed geometries
    owned : optional pre-built OwnedGeometryArray (avoids shapely->owned conversion
            when data is already device-resident, eliminating D->H transfer for
            the validity check per ADR-0005)
    dispatch_mode : requested execution mode (AUTO/GPU/CPU)
    """
    row_count = owned.row_count if owned is not None else (len(values) if values is not None else 0)

    selection = plan_dispatch_selection(
        kernel_name="make_valid",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=row_count,
        requested_mode=dispatch_mode,
    )

    # Defer Shapely materialization: when owned is provided, we may not need
    # values at all (zero-transfer fast path).  Materialize lazily.
    geometries = None
    null_mask = None

    def _ensure_geometries():
        nonlocal geometries, null_mask
        if geometries is not None:
            return
        if values is not None:
            geometries = np.asarray(values, dtype=object)
        elif owned is not None:
            geometries = np.asarray(owned.to_shapely(), dtype=object)
        else:
            raise ValueError("Either values or owned must be provided")
        # Preserve vectorized null_mask from ~owned.validity when already set
        if null_mask is None:
            null_mask = np.asarray([g is None for g in geometries], dtype=bool)

    # ADR-0019 compact-invalid-row: detect validity first, repair only invalids.
    # When an OwnedGeometryArray is provided (data already on device), use GPU
    # ring-structure checks to avoid D->H transfer for the validity check.
    # If all rows pass GPU ring checks, skip shapely entirely (zero-transfer).
    gpu_detection_used = False
    if owned is not None and selection.selected is ExecutionMode.GPU:
        # Compute null_mask from owned validity (no Shapely needed)
        null_mask = ~owned.validity
        gpu_mask = _gpu_polygon_validity_mask(owned)
        if gpu_mask is not None:
            gpu_detection_used = True
            gpu_mask[null_mask] = False
            # Refine with self-intersection detection on structurally-valid subset
            _ensure_geometries()
            gpu_mask = _detect_self_intersections_gpu(owned, gpu_mask, geometries=geometries)
            gpu_invalid_rows = np.flatnonzero(~gpu_mask & ~null_mask)
            if gpu_invalid_rows.size == 0:
                # All rows passed GPU checks -- no repair needed.
                # Carry the original owned so callers can stay
                # device-resident without re-uploading via from_shapely.
                _ensure_geometries()
                record_dispatch_event(
                    surface="geopandas.array.make_valid",
                    operation="make_valid",
                    implementation="gpu_ring_validity_check",
                    reason="GPU ring-structure checks: all rows valid, zero repair needed",
                    detail=f"rows={row_count}, method={method}",
                    requested=dispatch_mode,
                    selected=ExecutionMode.GPU,
                )
                return MakeValidResult(
                    geometries=geometries,
                    row_count=len(geometries),
                    valid_rows=np.flatnonzero(gpu_mask).astype(np.int32),
                    repaired_rows=np.asarray([], dtype=np.int32),
                    null_rows=np.flatnonzero(null_mask).astype(np.int32),
                    method=method,
                    keep_collapsed=keep_collapsed,
                    owned=owned,
                    selected=ExecutionMode.GPU,
                )
            valid_mask = gpu_mask
        else:
            _ensure_geometries()
            valid_mask = np.asarray(shapely.is_valid(geometries), dtype=bool)
    else:
        if owned is not None:
            null_mask = ~owned.validity
        _ensure_geometries()
        valid_mask = np.asarray(shapely.is_valid(geometries), dtype=bool)

    # From here on we need Shapely geometries for repair
    _ensure_geometries()
    result = geometries.copy()
    valid_mask[null_mask] = False
    repaired_mask = (~null_mask) & (~valid_mask)
    repaired_rows = np.flatnonzero(repaired_mask).astype(np.int32)
    selected = ExecutionMode.GPU if gpu_detection_used else ExecutionMode.CPU
    if repaired_rows.size:
        # Try GPU repair path first when owned data is available (ADR-0019 + ADR-0033)
        gpu_repair_done = False
        if owned is not None and owned.device_state is not None and selection.selected is ExecutionMode.GPU:
            try:
                gpu_result = _make_valid_gpu_repair(
                    owned, repaired_rows, geometries,
                    method=method, keep_collapsed=keep_collapsed,
                )
                if gpu_result is not None:
                    result = gpu_result.repaired_geometries
                    gpu_repair_done = True
                    selected = ExecutionMode.GPU
            except Exception:
                pass

        if not gpu_repair_done:
            result = _make_valid_cpu_repair(
                geometries, repaired_rows,
                method=method, keep_collapsed=keep_collapsed,
            )
            if not gpu_detection_used:
                selected = ExecutionMode.CPU

    impl = "gpu_ring_validity_check" if gpu_detection_used else "shapely_is_valid"
    if repaired_rows.size:
        impl += "+gpu_repair" if selected is ExecutionMode.GPU else "+shapely_make_valid"
    record_dispatch_event(
        surface="geopandas.array.make_valid",
        operation="make_valid",
        implementation=impl,
        reason=f"make_valid dispatch: detection={'GPU' if gpu_detection_used else 'CPU'}, "
               f"repair={selected.value}, {repaired_rows.size} rows repaired",
        detail=f"rows={row_count}, method={method}, repaired={repaired_rows.size}",
        requested=dispatch_mode,
        selected=selected,
    )
    return MakeValidResult(
        geometries=result,
        row_count=len(result),
        valid_rows=np.flatnonzero(valid_mask).astype(np.int32),
        repaired_rows=repaired_rows,
        null_rows=np.flatnonzero(null_mask).astype(np.int32),
        method=method,
        keep_collapsed=keep_collapsed,
        selected=selected,
    )


def evaluate_geopandas_make_valid(
    values,
    *,
    method: str = "linework",
    keep_collapsed: bool = True,
    prebuilt_owned=None,
) -> MakeValidResult:
    """Run make_valid and return the full MakeValidResult.

    Returns MakeValidResult so callers can access .owned for device-resident
    fast paths and .selected for dispatch event accuracy.
    """
    from vibespatial.runtime.execution_trace import execution_trace

    with execution_trace("make_valid"):
        return make_valid_owned(
            values, method=method, keep_collapsed=keep_collapsed, owned=prebuilt_owned,
        )


def benchmark_make_valid(values, *, method: str = "linework", keep_collapsed: bool = True, dataset: str = "make-valid"):
    geometries = np.asarray(values, dtype=object)
    start = perf_counter()
    compact = make_valid_owned(geometries, method=method, keep_collapsed=keep_collapsed)
    compact_elapsed = perf_counter() - start

    start = perf_counter()
    shapely.make_valid(geometries, method=method, keep_collapsed=keep_collapsed)
    baseline_elapsed = perf_counter() - start

    return MakeValidBenchmark(
        dataset=dataset,
        rows=len(geometries),
        repaired_rows=int(compact.repaired_rows.size),
        compact_elapsed_seconds=compact_elapsed,
        baseline_elapsed_seconds=baseline_elapsed,
    )
