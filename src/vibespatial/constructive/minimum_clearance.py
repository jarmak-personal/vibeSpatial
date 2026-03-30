"""GPU-accelerated minimum_clearance and minimum_clearance_line computation.

Minimum clearance is the smallest distance between any two non-adjacent
vertices/edges of a geometry.  Returns a scalar float per geometry.

Minimum clearance line returns a 2-point LineString connecting the two
closest non-adjacent points of a geometry.

Architecture (ADR-0033 tier classification):
- Point/MultiPoint: infinity / empty LineString (no segments, no clearance)
- LineString: O(n^2) pairwise non-adjacent segment distance (Tier 1 NVRTC)
- Polygon: O(n^2) all-segments including cross-ring pairs (Tier 1 NVRTC)
- MultiLineString: O(n^2) all segments across all parts (Tier 1 NVRTC)
- MultiPolygon: O(n^2) all segments across all parts and rings (Tier 1 NVRTC)

Precision (ADR-0002):
- minimum_clearance: METRIC class -- fp64 required for distance computation.
- minimum_clearance_line: CONSTRUCTIVE class -- fp64 required (output is geometry).
Both are templated on compute_t for observability but stay fp64.

Two segments are "adjacent" if they share a vertex index in the flattened
coordinate array.  Adjacent segments always have distance 0 at the shared
vertex and are excluded from the minimum.

The kernel computes full segment-to-segment distance (not just
vertex-to-vertex) using the parametric closest-approach method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass

if TYPE_CHECKING:
    from vibespatial.runtime.precision import PrecisionMode, PrecisionPlan

from vibespatial.constructive.minimum_clearance_kernels import (
    _LINESTRING_CLEARANCE_FP64,
    _LINESTRING_CLEARANCE_LINE_FP64,
    _LINESTRING_CLEARANCE_LINE_NAMES,
    _LINESTRING_CLEARANCE_NAMES,
    _MULTILINESTRING_CLEARANCE_FP64,
    _MULTILINESTRING_CLEARANCE_LINE_FP64,
    _MULTILINESTRING_CLEARANCE_LINE_NAMES,
    _MULTILINESTRING_CLEARANCE_NAMES,
    _MULTIPOLYGON_CLEARANCE_FP64,
    _MULTIPOLYGON_CLEARANCE_LINE_FP64,
    _MULTIPOLYGON_CLEARANCE_LINE_NAMES,
    _MULTIPOLYGON_CLEARANCE_NAMES,
    _POLYGON_CLEARANCE_FP64,
    _POLYGON_CLEARANCE_LINE_FP64,
    _POLYGON_CLEARANCE_LINE_NAMES,
    _POLYGON_CLEARANCE_NAMES,
)

# Background precompilation (ADR-0034)
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup

request_nvrtc_warmup([
    ("linestring-clearance-fp64", _LINESTRING_CLEARANCE_FP64, _LINESTRING_CLEARANCE_NAMES),
    ("polygon-clearance-fp64", _POLYGON_CLEARANCE_FP64, _POLYGON_CLEARANCE_NAMES),
    ("multipolygon-clearance-fp64", _MULTIPOLYGON_CLEARANCE_FP64, _MULTIPOLYGON_CLEARANCE_NAMES),
    ("multilinestring-clearance-fp64", _MULTILINESTRING_CLEARANCE_FP64, _MULTILINESTRING_CLEARANCE_NAMES),
    ("linestring-clearance-line-fp64", _LINESTRING_CLEARANCE_LINE_FP64, _LINESTRING_CLEARANCE_LINE_NAMES),
    ("polygon-clearance-line-fp64", _POLYGON_CLEARANCE_LINE_FP64, _POLYGON_CLEARANCE_LINE_NAMES),
    ("multipolygon-clearance-line-fp64", _MULTIPOLYGON_CLEARANCE_LINE_FP64, _MULTIPOLYGON_CLEARANCE_LINE_NAMES),
    ("multilinestring-clearance-line-fp64", _MULTILINESTRING_CLEARANCE_LINE_FP64, _MULTILINESTRING_CLEARANCE_LINE_NAMES),
])


# ---------------------------------------------------------------------------
# GPU implementation
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "minimum_clearance",
    "gpu-cuda-python",
    kernel_class=KernelClass.METRIC,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("linestring", "polygon", "multipolygon", "multilinestring"),
    supports_mixed=True,
    tags=("cuda-python", "metric", "clearance", "segment-distance"),
)
def _minimum_clearance_gpu(
    owned: OwnedGeometryArray,
    precision_plan: PrecisionPlan | None = None,
) -> np.ndarray:
    """GPU-accelerated minimum clearance.  Returns float64 array of shape (row_count,)."""
    runtime = get_cuda_runtime()
    row_count = owned.row_count
    result = np.full(row_count, np.inf, dtype=np.float64)

    tags = owned.tags
    family_row_offsets = owned.family_row_offsets
    device_state = owned.device_state

    # --- LineString family ---
    _launch_linestring_clearance(
        owned, result, tags, family_row_offsets, device_state, runtime,
    )

    # --- MultiLineString family ---
    _launch_multilinestring_clearance(
        owned, result, tags, family_row_offsets, device_state, runtime,
    )

    # --- Polygon family ---
    _launch_polygon_clearance(
        owned, result, tags, family_row_offsets, device_state, runtime,
    )

    # --- MultiPolygon family ---
    _launch_multipolygon_clearance(
        owned, result, tags, family_row_offsets, device_state, runtime,
    )

    # Point, MultiPoint: clearance = infinity (already initialized)
    return result


def _launch_linestring_clearance(
    owned, result, tags, family_row_offsets, device_state, runtime,
):
    """Launch linestring minimum clearance kernel."""
    tag = FAMILY_TAGS[GeometryFamily.LINESTRING]
    mask = tags == tag
    if not np.any(mask) or GeometryFamily.LINESTRING not in owned.families:
        return
    buf = owned.families[GeometryFamily.LINESTRING]
    if buf.row_count == 0 or len(buf.geometry_offsets) < 2:
        return

    kernels = compile_kernel_group(
        "linestring-clearance-fp64",
        _LINESTRING_CLEARANCE_FP64,
        _LINESTRING_CLEARANCE_NAMES,
    )
    kernel = kernels["linestring_minimum_clearance"]
    global_rows = np.flatnonzero(mask)
    family_rows = family_row_offsets[global_rows]
    n = buf.row_count

    needs_free = (
        device_state is None
        or GeometryFamily.LINESTRING not in (device_state.families if device_state else {})
    )
    allocated = []
    if not needs_free:
        ds = device_state.families[GeometryFamily.LINESTRING]
        d_x, d_y = ds.x, ds.y
        d_geom = ds.geometry_offsets
    else:
        d_x = runtime.from_host(buf.x)
        d_y = runtime.from_host(buf.y)
        d_geom = runtime.from_host(buf.geometry_offsets.astype(np.int32))
        allocated.extend([d_x, d_y, d_geom])

    d_out = runtime.allocate((n,), np.float64)
    try:
        ptr = runtime.pointer
        params = (
            (ptr(d_x), ptr(d_y), ptr(d_geom), ptr(d_out), n),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        )
        grid, block = runtime.launch_config(kernel, n)
        runtime.launch(kernel, grid=grid, block=block, params=params)
        family_result = runtime.copy_device_to_host(d_out)
        result[global_rows] = family_result[family_rows]
    finally:
        runtime.free(d_out)
        for d in allocated:
            runtime.free(d)


def _launch_multilinestring_clearance(
    owned, result, tags, family_row_offsets, device_state, runtime,
):
    """Launch multilinestring minimum clearance kernel."""
    tag = FAMILY_TAGS[GeometryFamily.MULTILINESTRING]
    mask = tags == tag
    if not np.any(mask) or GeometryFamily.MULTILINESTRING not in owned.families:
        return
    buf = owned.families[GeometryFamily.MULTILINESTRING]
    if buf.row_count == 0 or buf.part_offsets is None or len(buf.geometry_offsets) < 2:
        return

    kernels = compile_kernel_group(
        "multilinestring-clearance-fp64",
        _MULTILINESTRING_CLEARANCE_FP64,
        _MULTILINESTRING_CLEARANCE_NAMES,
    )
    kernel = kernels["multilinestring_minimum_clearance"]
    global_rows = np.flatnonzero(mask)
    family_rows = family_row_offsets[global_rows]
    n = buf.row_count

    needs_free = (
        device_state is None
        or GeometryFamily.MULTILINESTRING not in (device_state.families if device_state else {})
    )
    allocated = []
    if not needs_free:
        ds = device_state.families[GeometryFamily.MULTILINESTRING]
        d_x, d_y = ds.x, ds.y
        d_part = ds.part_offsets
        d_geom = ds.geometry_offsets
    else:
        d_x = runtime.from_host(buf.x)
        d_y = runtime.from_host(buf.y)
        d_part = runtime.from_host(buf.part_offsets.astype(np.int32))
        d_geom = runtime.from_host(buf.geometry_offsets.astype(np.int32))
        allocated.extend([d_x, d_y, d_part, d_geom])

    d_out = runtime.allocate((n,), np.float64)
    try:
        ptr = runtime.pointer
        params = (
            (ptr(d_x), ptr(d_y), ptr(d_part), ptr(d_geom), ptr(d_out), n),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        )
        grid, block = runtime.launch_config(kernel, n)
        runtime.launch(kernel, grid=grid, block=block, params=params)
        family_result = runtime.copy_device_to_host(d_out)
        result[global_rows] = family_result[family_rows]
    finally:
        runtime.free(d_out)
        for d in allocated:
            runtime.free(d)


def _launch_polygon_clearance(
    owned, result, tags, family_row_offsets, device_state, runtime,
):
    """Launch polygon minimum clearance kernel."""
    tag = FAMILY_TAGS[GeometryFamily.POLYGON]
    mask = tags == tag
    if not np.any(mask) or GeometryFamily.POLYGON not in owned.families:
        return
    buf = owned.families[GeometryFamily.POLYGON]
    if buf.row_count == 0 or buf.ring_offsets is None or len(buf.geometry_offsets) < 2:
        return

    kernels = compile_kernel_group(
        "polygon-clearance-fp64",
        _POLYGON_CLEARANCE_FP64,
        _POLYGON_CLEARANCE_NAMES,
    )
    kernel = kernels["polygon_minimum_clearance"]
    global_rows = np.flatnonzero(mask)
    family_rows = family_row_offsets[global_rows]
    n = buf.row_count

    needs_free = (
        device_state is None
        or GeometryFamily.POLYGON not in (device_state.families if device_state else {})
    )
    allocated = []
    if not needs_free:
        ds = device_state.families[GeometryFamily.POLYGON]
        d_x, d_y = ds.x, ds.y
        d_ring = ds.ring_offsets
        d_geom = ds.geometry_offsets
    else:
        d_x = runtime.from_host(buf.x)
        d_y = runtime.from_host(buf.y)
        d_ring = runtime.from_host(buf.ring_offsets.astype(np.int32))
        d_geom = runtime.from_host(buf.geometry_offsets.astype(np.int32))
        allocated.extend([d_x, d_y, d_ring, d_geom])

    d_out = runtime.allocate((n,), np.float64)
    try:
        ptr = runtime.pointer
        params = (
            (ptr(d_x), ptr(d_y), ptr(d_ring), ptr(d_geom), ptr(d_out), n),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        )
        grid, block = runtime.launch_config(kernel, n)
        runtime.launch(kernel, grid=grid, block=block, params=params)
        family_result = runtime.copy_device_to_host(d_out)
        result[global_rows] = family_result[family_rows]
    finally:
        runtime.free(d_out)
        for d in allocated:
            runtime.free(d)


def _launch_multipolygon_clearance(
    owned, result, tags, family_row_offsets, device_state, runtime,
):
    """Launch multipolygon minimum clearance kernel."""
    tag = FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]
    mask = tags == tag
    if not np.any(mask) or GeometryFamily.MULTIPOLYGON not in owned.families:
        return
    buf = owned.families[GeometryFamily.MULTIPOLYGON]
    if (
        buf.row_count == 0
        or buf.ring_offsets is None
        or buf.part_offsets is None
        or len(buf.geometry_offsets) < 2
    ):
        return

    kernels = compile_kernel_group(
        "multipolygon-clearance-fp64",
        _MULTIPOLYGON_CLEARANCE_FP64,
        _MULTIPOLYGON_CLEARANCE_NAMES,
    )
    kernel = kernels["multipolygon_minimum_clearance"]
    global_rows = np.flatnonzero(mask)
    family_rows = family_row_offsets[global_rows]
    n = buf.row_count

    needs_free = (
        device_state is None
        or GeometryFamily.MULTIPOLYGON not in (device_state.families if device_state else {})
    )
    allocated = []
    if not needs_free:
        ds = device_state.families[GeometryFamily.MULTIPOLYGON]
        d_x, d_y = ds.x, ds.y
        d_ring = ds.ring_offsets
        d_part = ds.part_offsets
        d_geom = ds.geometry_offsets
    else:
        d_x = runtime.from_host(buf.x)
        d_y = runtime.from_host(buf.y)
        d_ring = runtime.from_host(buf.ring_offsets.astype(np.int32))
        d_part = runtime.from_host(buf.part_offsets.astype(np.int32))
        d_geom = runtime.from_host(buf.geometry_offsets.astype(np.int32))
        allocated.extend([d_x, d_y, d_ring, d_part, d_geom])

    d_out = runtime.allocate((n,), np.float64)
    try:
        ptr = runtime.pointer
        params = (
            (ptr(d_x), ptr(d_y), ptr(d_ring), ptr(d_part), ptr(d_geom),
             ptr(d_out), n),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_I32),
        )
        grid, block = runtime.launch_config(kernel, n)
        runtime.launch(kernel, grid=grid, block=block, params=params)
        family_result = runtime.copy_device_to_host(d_out)
        result[global_rows] = family_result[family_rows]
    finally:
        runtime.free(d_out)
        for d in allocated:
            runtime.free(d)


# ---------------------------------------------------------------------------
# CPU fallback
# ---------------------------------------------------------------------------

def _minimum_clearance_cpu(owned: OwnedGeometryArray) -> np.ndarray:
    """CPU fallback via Shapely."""
    import shapely as _shapely

    geoms = np.asarray(owned.to_shapely(), dtype=object)
    return _shapely.minimum_clearance(geoms)


# ---------------------------------------------------------------------------
# Public dispatch entry point
# ---------------------------------------------------------------------------

def minimum_clearance_owned(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = "auto",
) -> np.ndarray:
    """Compute minimum clearance from OwnedGeometryArray coordinate buffers.

    GPU path uses ADR-0002 METRIC-class precision dispatch (fp64 required
    for distance computation).  Returns float64 array of shape (row_count,).

    Zero host/device transfers mid-process.  When owned.device_state
    is populated (vibeFrame path), GPU kernels read directly from
    device pointers with no copy.
    """
    from vibespatial.runtime import RuntimeSelection
    from vibespatial.runtime.precision import select_precision_plan

    row_count = owned.row_count
    if row_count == 0:
        return np.empty(0, dtype=np.float64)

    selection = plan_dispatch_selection(
        kernel_name="minimum_clearance",
        kernel_class=KernelClass.METRIC,
        row_count=row_count,
        requested_mode=dispatch_mode,
    )

    if selection.selected is ExecutionMode.GPU:
        precision_plan = select_precision_plan(
            runtime_selection=RuntimeSelection(
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.GPU,
                reason="minimum_clearance GPU dispatch",
            ),
            kernel_class=KernelClass.METRIC,
            requested=precision,
        )
        try:
            result = _minimum_clearance_gpu(owned, precision_plan=precision_plan)
            result[~owned.validity] = np.nan
        except Exception:
            pass  # fall through to CPU
        else:
            record_dispatch_event(
                surface="geopandas.array.minimum_clearance",
                operation="minimum_clearance",
                implementation="gpu_nvrtc_segment_distance",
                reason="GPU NVRTC minimum clearance kernel",
                detail=f"rows={row_count}",
                selected=ExecutionMode.GPU,
            )
            return result

    record_dispatch_event(
        surface="geopandas.array.minimum_clearance",
        operation="minimum_clearance",
        implementation="shapely",
        reason="CPU fallback",
        detail=f"rows={row_count}",
        selected=ExecutionMode.CPU,
    )
    result = _minimum_clearance_cpu(owned)
    result[~owned.validity] = np.nan
    return result


# ===========================================================================
# minimum_clearance_line GPU implementation
# ===========================================================================


@register_kernel_variant(
    "minimum_clearance_line",
    "gpu-cuda-python",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("linestring", "polygon", "multipolygon", "multilinestring"),
    supports_mixed=True,
    tags=("cuda-python", "constructive", "clearance-line", "segment-distance"),
)
def _minimum_clearance_line_gpu(
    owned: OwnedGeometryArray,
    precision_plan: PrecisionPlan | None = None,
) -> OwnedGeometryArray:
    """GPU-accelerated minimum clearance line.

    Returns an OwnedGeometryArray of 2-point LineStrings.  Each LineString
    connects the two closest non-adjacent points of the input geometry.
    Degenerate cases (points, too few segments) yield empty LineStrings.
    """
    runtime = get_cuda_runtime()
    row_count = owned.row_count

    # Per-row output: 4 doubles (ax, ay, bx, by). NaN = empty.
    out_ax = np.full(row_count, np.nan, dtype=np.float64)
    out_ay = np.full(row_count, np.nan, dtype=np.float64)
    out_bx = np.full(row_count, np.nan, dtype=np.float64)
    out_by = np.full(row_count, np.nan, dtype=np.float64)

    tags = owned.tags
    family_row_offsets = owned.family_row_offsets
    device_state = owned.device_state

    _launch_linestring_clearance_line(
        owned, out_ax, out_ay, out_bx, out_by,
        tags, family_row_offsets, device_state, runtime,
    )
    _launch_multilinestring_clearance_line(
        owned, out_ax, out_ay, out_bx, out_by,
        tags, family_row_offsets, device_state, runtime,
    )
    _launch_polygon_clearance_line(
        owned, out_ax, out_ay, out_bx, out_by,
        tags, family_row_offsets, device_state, runtime,
    )
    _launch_multipolygon_clearance_line(
        owned, out_ax, out_ay, out_bx, out_by,
        tags, family_row_offsets, device_state, runtime,
    )

    # Mark invalid rows as NaN (empty LineString)
    validity = owned.validity
    out_ax[~validity] = np.nan
    out_ay[~validity] = np.nan
    out_bx[~validity] = np.nan
    out_by[~validity] = np.nan

    return _build_clearance_line_oga(row_count, out_ax, out_ay, out_bx, out_by, validity)


# ---------------------------------------------------------------------------
# Per-family launch helpers for clearance line
# ---------------------------------------------------------------------------


def _launch_linestring_clearance_line(
    owned, out_ax, out_ay, out_bx, out_by,
    tags, family_row_offsets, device_state, runtime,
):
    """Launch linestring clearance line kernel."""
    tag = FAMILY_TAGS[GeometryFamily.LINESTRING]
    mask = tags == tag
    if not np.any(mask) or GeometryFamily.LINESTRING not in owned.families:
        return
    buf = owned.families[GeometryFamily.LINESTRING]
    if buf.row_count == 0 or len(buf.geometry_offsets) < 2:
        return

    kernels = compile_kernel_group(
        "linestring-clearance-line-fp64",
        _LINESTRING_CLEARANCE_LINE_FP64,
        _LINESTRING_CLEARANCE_LINE_NAMES,
    )
    kernel = kernels["linestring_clearance_line"]
    global_rows = np.flatnonzero(mask)
    family_rows = family_row_offsets[global_rows]
    n = buf.row_count

    needs_free = (
        device_state is None
        or GeometryFamily.LINESTRING not in (device_state.families if device_state else {})
    )
    allocated = []
    if not needs_free:
        ds = device_state.families[GeometryFamily.LINESTRING]
        d_x, d_y = ds.x, ds.y
        d_geom = ds.geometry_offsets
    else:
        d_x = runtime.from_host(buf.x)
        d_y = runtime.from_host(buf.y)
        d_geom = runtime.from_host(buf.geometry_offsets.astype(np.int32))
        allocated.extend([d_x, d_y, d_geom])

    d_oax = runtime.allocate((n,), np.float64)
    d_oay = runtime.allocate((n,), np.float64)
    d_obx = runtime.allocate((n,), np.float64)
    d_oby = runtime.allocate((n,), np.float64)
    try:
        ptr = runtime.pointer
        params = (
            (ptr(d_x), ptr(d_y), ptr(d_geom),
             ptr(d_oax), ptr(d_oay), ptr(d_obx), ptr(d_oby), n),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        )
        grid, block = runtime.launch_config(kernel, n)
        runtime.launch(kernel, grid=grid, block=block, params=params)
        h_ax = runtime.copy_device_to_host(d_oax)
        h_ay = runtime.copy_device_to_host(d_oay)
        h_bx = runtime.copy_device_to_host(d_obx)
        h_by = runtime.copy_device_to_host(d_oby)
        out_ax[global_rows] = h_ax[family_rows]
        out_ay[global_rows] = h_ay[family_rows]
        out_bx[global_rows] = h_bx[family_rows]
        out_by[global_rows] = h_by[family_rows]
    finally:
        for d in (d_oax, d_oay, d_obx, d_oby):
            runtime.free(d)
        for d in allocated:
            runtime.free(d)


def _launch_multilinestring_clearance_line(
    owned, out_ax, out_ay, out_bx, out_by,
    tags, family_row_offsets, device_state, runtime,
):
    """Launch multilinestring clearance line kernel."""
    tag = FAMILY_TAGS[GeometryFamily.MULTILINESTRING]
    mask = tags == tag
    if not np.any(mask) or GeometryFamily.MULTILINESTRING not in owned.families:
        return
    buf = owned.families[GeometryFamily.MULTILINESTRING]
    if buf.row_count == 0 or buf.part_offsets is None or len(buf.geometry_offsets) < 2:
        return

    kernels = compile_kernel_group(
        "multilinestring-clearance-line-fp64",
        _MULTILINESTRING_CLEARANCE_LINE_FP64,
        _MULTILINESTRING_CLEARANCE_LINE_NAMES,
    )
    kernel = kernels["multilinestring_clearance_line"]
    global_rows = np.flatnonzero(mask)
    family_rows = family_row_offsets[global_rows]
    n = buf.row_count

    needs_free = (
        device_state is None
        or GeometryFamily.MULTILINESTRING not in (device_state.families if device_state else {})
    )
    allocated = []
    if not needs_free:
        ds = device_state.families[GeometryFamily.MULTILINESTRING]
        d_x, d_y = ds.x, ds.y
        d_part = ds.part_offsets
        d_geom = ds.geometry_offsets
    else:
        d_x = runtime.from_host(buf.x)
        d_y = runtime.from_host(buf.y)
        d_part = runtime.from_host(buf.part_offsets.astype(np.int32))
        d_geom = runtime.from_host(buf.geometry_offsets.astype(np.int32))
        allocated.extend([d_x, d_y, d_part, d_geom])

    d_oax = runtime.allocate((n,), np.float64)
    d_oay = runtime.allocate((n,), np.float64)
    d_obx = runtime.allocate((n,), np.float64)
    d_oby = runtime.allocate((n,), np.float64)
    try:
        ptr = runtime.pointer
        params = (
            (ptr(d_x), ptr(d_y), ptr(d_part), ptr(d_geom),
             ptr(d_oax), ptr(d_oay), ptr(d_obx), ptr(d_oby), n),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        )
        grid, block = runtime.launch_config(kernel, n)
        runtime.launch(kernel, grid=grid, block=block, params=params)
        h_ax = runtime.copy_device_to_host(d_oax)
        h_ay = runtime.copy_device_to_host(d_oay)
        h_bx = runtime.copy_device_to_host(d_obx)
        h_by = runtime.copy_device_to_host(d_oby)
        out_ax[global_rows] = h_ax[family_rows]
        out_ay[global_rows] = h_ay[family_rows]
        out_bx[global_rows] = h_bx[family_rows]
        out_by[global_rows] = h_by[family_rows]
    finally:
        for d in (d_oax, d_oay, d_obx, d_oby):
            runtime.free(d)
        for d in allocated:
            runtime.free(d)


def _launch_polygon_clearance_line(
    owned, out_ax, out_ay, out_bx, out_by,
    tags, family_row_offsets, device_state, runtime,
):
    """Launch polygon clearance line kernel."""
    tag = FAMILY_TAGS[GeometryFamily.POLYGON]
    mask = tags == tag
    if not np.any(mask) or GeometryFamily.POLYGON not in owned.families:
        return
    buf = owned.families[GeometryFamily.POLYGON]
    if buf.row_count == 0 or buf.ring_offsets is None or len(buf.geometry_offsets) < 2:
        return

    kernels = compile_kernel_group(
        "polygon-clearance-line-fp64",
        _POLYGON_CLEARANCE_LINE_FP64,
        _POLYGON_CLEARANCE_LINE_NAMES,
    )
    kernel = kernels["polygon_clearance_line"]
    global_rows = np.flatnonzero(mask)
    family_rows = family_row_offsets[global_rows]
    n = buf.row_count

    needs_free = (
        device_state is None
        or GeometryFamily.POLYGON not in (device_state.families if device_state else {})
    )
    allocated = []
    if not needs_free:
        ds = device_state.families[GeometryFamily.POLYGON]
        d_x, d_y = ds.x, ds.y
        d_ring = ds.ring_offsets
        d_geom = ds.geometry_offsets
    else:
        d_x = runtime.from_host(buf.x)
        d_y = runtime.from_host(buf.y)
        d_ring = runtime.from_host(buf.ring_offsets.astype(np.int32))
        d_geom = runtime.from_host(buf.geometry_offsets.astype(np.int32))
        allocated.extend([d_x, d_y, d_ring, d_geom])

    d_oax = runtime.allocate((n,), np.float64)
    d_oay = runtime.allocate((n,), np.float64)
    d_obx = runtime.allocate((n,), np.float64)
    d_oby = runtime.allocate((n,), np.float64)
    try:
        ptr = runtime.pointer
        params = (
            (ptr(d_x), ptr(d_y), ptr(d_ring), ptr(d_geom),
             ptr(d_oax), ptr(d_oay), ptr(d_obx), ptr(d_oby), n),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        )
        grid, block = runtime.launch_config(kernel, n)
        runtime.launch(kernel, grid=grid, block=block, params=params)
        h_ax = runtime.copy_device_to_host(d_oax)
        h_ay = runtime.copy_device_to_host(d_oay)
        h_bx = runtime.copy_device_to_host(d_obx)
        h_by = runtime.copy_device_to_host(d_oby)
        out_ax[global_rows] = h_ax[family_rows]
        out_ay[global_rows] = h_ay[family_rows]
        out_bx[global_rows] = h_bx[family_rows]
        out_by[global_rows] = h_by[family_rows]
    finally:
        for d in (d_oax, d_oay, d_obx, d_oby):
            runtime.free(d)
        for d in allocated:
            runtime.free(d)


def _launch_multipolygon_clearance_line(
    owned, out_ax, out_ay, out_bx, out_by,
    tags, family_row_offsets, device_state, runtime,
):
    """Launch multipolygon clearance line kernel."""
    tag = FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]
    mask = tags == tag
    if not np.any(mask) or GeometryFamily.MULTIPOLYGON not in owned.families:
        return
    buf = owned.families[GeometryFamily.MULTIPOLYGON]
    if (
        buf.row_count == 0
        or buf.ring_offsets is None
        or buf.part_offsets is None
        or len(buf.geometry_offsets) < 2
    ):
        return

    kernels = compile_kernel_group(
        "multipolygon-clearance-line-fp64",
        _MULTIPOLYGON_CLEARANCE_LINE_FP64,
        _MULTIPOLYGON_CLEARANCE_LINE_NAMES,
    )
    kernel = kernels["multipolygon_clearance_line"]
    global_rows = np.flatnonzero(mask)
    family_rows = family_row_offsets[global_rows]
    n = buf.row_count

    needs_free = (
        device_state is None
        or GeometryFamily.MULTIPOLYGON not in (device_state.families if device_state else {})
    )
    allocated = []
    if not needs_free:
        ds = device_state.families[GeometryFamily.MULTIPOLYGON]
        d_x, d_y = ds.x, ds.y
        d_ring = ds.ring_offsets
        d_part = ds.part_offsets
        d_geom = ds.geometry_offsets
    else:
        d_x = runtime.from_host(buf.x)
        d_y = runtime.from_host(buf.y)
        d_ring = runtime.from_host(buf.ring_offsets.astype(np.int32))
        d_part = runtime.from_host(buf.part_offsets.astype(np.int32))
        d_geom = runtime.from_host(buf.geometry_offsets.astype(np.int32))
        allocated.extend([d_x, d_y, d_ring, d_part, d_geom])

    d_oax = runtime.allocate((n,), np.float64)
    d_oay = runtime.allocate((n,), np.float64)
    d_obx = runtime.allocate((n,), np.float64)
    d_oby = runtime.allocate((n,), np.float64)
    try:
        ptr = runtime.pointer
        params = (
            (ptr(d_x), ptr(d_y), ptr(d_ring), ptr(d_part), ptr(d_geom),
             ptr(d_oax), ptr(d_oay), ptr(d_obx), ptr(d_oby), n),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_I32),
        )
        grid, block = runtime.launch_config(kernel, n)
        runtime.launch(kernel, grid=grid, block=block, params=params)
        h_ax = runtime.copy_device_to_host(d_oax)
        h_ay = runtime.copy_device_to_host(d_oay)
        h_bx = runtime.copy_device_to_host(d_obx)
        h_by = runtime.copy_device_to_host(d_oby)
        out_ax[global_rows] = h_ax[family_rows]
        out_ay[global_rows] = h_ay[family_rows]
        out_bx[global_rows] = h_bx[family_rows]
        out_by[global_rows] = h_by[family_rows]
    finally:
        for d in (d_oax, d_oay, d_obx, d_oby):
            runtime.free(d)
        for d in allocated:
            runtime.free(d)


# ---------------------------------------------------------------------------
# Build OwnedGeometryArray of 2-point LineStrings from clearance point pairs
# ---------------------------------------------------------------------------


def _build_clearance_line_oga(
    row_count: int,
    out_ax: np.ndarray,
    out_ay: np.ndarray,
    out_bx: np.ndarray,
    out_by: np.ndarray,
    validity: np.ndarray,
) -> OwnedGeometryArray:
    """Assemble a 2-point LineString OGA from closest-point pair arrays.

    Each row where (ax,ay) and (bx,by) are finite produces a 2-point
    LineString.  Rows with NaN produce empty LineStrings.
    """
    # Determine which rows have valid clearance lines
    has_line = np.isfinite(out_ax) & validity

    # Build coordinate arrays: 2 points per valid row, 0 per empty
    coords_per_row = np.where(has_line, 2, 0).astype(np.int32)
    # geometry_offsets: cumulative sum [0, c0, c0+c1, ...]
    geometry_offsets = np.empty(row_count + 1, dtype=np.int32)
    geometry_offsets[0] = 0
    np.cumsum(coords_per_row, out=geometry_offsets[1:])
    total_coords = int(geometry_offsets[-1])

    x = np.empty(total_coords, dtype=np.float64)
    y = np.empty(total_coords, dtype=np.float64)

    # Scatter coordinates for valid rows
    valid_indices = np.flatnonzero(has_line)
    if len(valid_indices) > 0:
        starts = geometry_offsets[valid_indices]
        x[starts] = out_ax[valid_indices]
        y[starts] = out_ay[valid_indices]
        x[starts + 1] = out_bx[valid_indices]
        y[starts + 1] = out_by[valid_indices]

    empty_mask = ~has_line
    out_tags = np.full(row_count, FAMILY_TAGS[GeometryFamily.LINESTRING], dtype=np.int8)
    out_validity = validity.copy()
    family_row_offsets = np.arange(row_count, dtype=np.int32)

    ls_buffer = FamilyGeometryBuffer(
        family=GeometryFamily.LINESTRING,
        schema=get_geometry_buffer_schema(GeometryFamily.LINESTRING),
        row_count=row_count,
        x=x,
        y=y,
        geometry_offsets=geometry_offsets,
        empty_mask=empty_mask,
    )

    return OwnedGeometryArray(
        validity=out_validity,
        tags=out_tags,
        family_row_offsets=family_row_offsets,
        families={GeometryFamily.LINESTRING: ls_buffer},
    )


# ---------------------------------------------------------------------------
# CPU fallback for clearance line
# ---------------------------------------------------------------------------


def _minimum_clearance_line_cpu(owned: OwnedGeometryArray) -> OwnedGeometryArray:
    """CPU fallback via Shapely."""
    import shapely as _shapely

    from vibespatial.geometry.owned import from_shapely_geometries

    geoms = np.asarray(owned.to_shapely(), dtype=object)
    result_geoms = _shapely.minimum_clearance_line(geoms)
    return from_shapely_geometries(result_geoms)


# ---------------------------------------------------------------------------
# Public dispatch entry point for clearance line
# ---------------------------------------------------------------------------


def minimum_clearance_line_owned(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = "auto",
) -> OwnedGeometryArray:
    """Compute minimum clearance line from OwnedGeometryArray coordinate buffers.

    GPU path uses ADR-0002 CONSTRUCTIVE-class precision dispatch (fp64 required,
    output is geometry).  Returns an OwnedGeometryArray of 2-point LineStrings.

    Each output LineString connects the two closest non-adjacent points of the
    corresponding input geometry.  Degenerate cases (points, multipoints,
    geometries with fewer than 3 non-adjacent segments) yield empty LineStrings.

    Zero host/device transfers mid-process.  When owned.device_state
    is populated (vibeFrame path), GPU kernels read directly from
    device pointers with no copy.
    """
    from vibespatial.runtime import RuntimeSelection
    from vibespatial.runtime.precision import select_precision_plan

    row_count = owned.row_count
    if row_count == 0:
        return _build_clearance_line_oga(
            0,
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=bool),
        )

    selection = plan_dispatch_selection(
        kernel_name="minimum_clearance_line",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=row_count,
        requested_mode=dispatch_mode,
    )

    if selection.selected is ExecutionMode.GPU:
        precision_plan = select_precision_plan(
            runtime_selection=RuntimeSelection(
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.GPU,
                reason="minimum_clearance_line GPU dispatch",
            ),
            kernel_class=KernelClass.CONSTRUCTIVE,
            requested=precision,
        )
        try:
            result = _minimum_clearance_line_gpu(owned, precision_plan=precision_plan)
        except Exception:
            pass  # fall through to CPU
        else:
            record_dispatch_event(
                surface="geopandas.array.minimum_clearance_line",
                operation="minimum_clearance_line",
                implementation="gpu_nvrtc_segment_closest_points",
                reason="GPU NVRTC minimum clearance line kernel",
                detail=f"rows={row_count}",
                selected=ExecutionMode.GPU,
            )
            return result

    record_dispatch_event(
        surface="geopandas.array.minimum_clearance_line",
        operation="minimum_clearance_line",
        implementation="shapely",
        reason="CPU fallback",
        detail=f"rows={row_count}",
        selected=ExecutionMode.CPU,
    )
    return _minimum_clearance_line_cpu(owned)
