from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vibespatial.constructive.measurement import (
    _coord_stats_from_owned,
    _fp32_center_coords,
)
from vibespatial.constructive.polygon_buffer_cpu import build_polygon_buffers_cpu
from vibespatial.cuda._runtime import (
    KERNEL_PARAM_F64,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    count_scatter_total_with_transfer,
    get_cuda_runtime,
)
from vibespatial.cuda.cccl_primitives import exclusive_sum
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    OwnedGeometryArray,
    build_device_resident_owned,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass
from vibespatial.runtime.residency import Residency, TransferTrigger

if TYPE_CHECKING:
    from vibespatial.runtime.precision import PrecisionMode, PrecisionPlan


from vibespatial.constructive.polygon_kernels import (
    _POLYGON_BUFFER_KERNEL_NAMES,
    _POLYGON_BUFFER_KERNEL_SOURCE,
    _POLYGON_CENTROID_FP32_SOURCE,
    _POLYGON_CENTROID_FP64_SOURCE,
    _POLYGON_CENTROID_KERNEL_NAMES,
    _RING_WINDING_KERNEL_NAMES,
    _RING_WINDING_KERNEL_SOURCE,
)
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup

request_nvrtc_warmup([
    ("polygon-buffer", _POLYGON_BUFFER_KERNEL_SOURCE, _POLYGON_BUFFER_KERNEL_NAMES),
    ("ring-winding", _RING_WINDING_KERNEL_SOURCE, _RING_WINDING_KERNEL_NAMES),
    ("polygon-centroid-fp64", _POLYGON_CENTROID_FP64_SOURCE, _POLYGON_CENTROID_KERNEL_NAMES),
    ("polygon-centroid-fp32", _POLYGON_CENTROID_FP32_SOURCE, _POLYGON_CENTROID_KERNEL_NAMES),
])

from vibespatial.cuda.cccl_precompile import request_warmup  # noqa: E402

request_warmup(["exclusive_scan_i32"])

_CAP_STYLE_MAP = {"round": 0, "flat": 1, "square": 2}
_JOIN_STYLE_MAP = {"round": 0, "mitre": 1, "bevel": 2}


def _ring_winding_kernels():
    return compile_kernel_group("ring-winding", _RING_WINDING_KERNEL_SOURCE, _RING_WINDING_KERNEL_NAMES)


def _polygon_centroid_kernels(compute_type: str = "double"):
    source = _POLYGON_CENTROID_FP64_SOURCE if compute_type == "double" else _POLYGON_CENTROID_FP32_SOURCE
    prefix = f"polygon-centroid-{compute_type[:2]}"  # fp64 / fl (float)
    return compile_kernel_group(prefix, source, _POLYGON_CENTROID_KERNEL_NAMES)


@register_kernel_variant(
    "polygon_centroid",
    "gpu-cuda-python",
    kernel_class=KernelClass.METRIC,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("polygon", "multipolygon"),
    supports_mixed=True,
    tags=("cuda-python", "metric", "centroid", "kahan", "centered"),
)
def _polygon_centroids_gpu(
    owned: OwnedGeometryArray,
    precision_plan: PrecisionPlan | None = None,
    return_owned: bool = False,
) -> tuple[np.ndarray, np.ndarray] | OwnedGeometryArray | None:
    """GPU-accelerated polygon centroid computation via NVRTC shoelace kernel.

    When *return_owned* is False (default), returns (cx, cy) numpy arrays of
    shape (row_count,) or None if GPU is unavailable or no polygon families
    are present.

    When *return_owned* is True, builds and returns a device-resident point
    ``OwnedGeometryArray`` directly from the GPU centroid buffers -- no D->H
    transfer.  The centroid device arrays become owned by the returned object.

    Respects ADR-0002 precision dispatch: fp32 with Kahan summation +
    coordinate centering on consumer GPUs, native fp64 on datacenter GPUs.
    """
    import cupy as cp

    from vibespatial.constructive.point import _build_device_backed_point_output
    from vibespatial.runtime.precision import PrecisionMode

    compute_type = "double"
    center_x, center_y = 0.0, 0.0
    if precision_plan is not None and precision_plan.compute_precision is PrecisionMode.FP32:
        compute_type = "float"
        if precision_plan.center_coordinates:
            center_x, center_y = _fp32_center_coords(owned)

    runtime = get_cuda_runtime()
    kernels = _polygon_centroid_kernels(compute_type)
    kernel = kernels["polygon_centroid"]

    row_count = owned.row_count

    poly_tag = FAMILY_TAGS[GeometryFamily.POLYGON]
    mpoly_tag = FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]
    tags = owned.tags
    family_row_offsets = owned.family_row_offsets
    device_state = owned.device_state

    # Collect per-family kernel results for the return_owned path.  When
    # return_owned is False we scatter into host numpy arrays as before.
    if not return_owned:
        cx = np.full(row_count, np.nan, dtype=np.float64)
        cy = np.full(row_count, np.nan, dtype=np.float64)

    # Track family results for return_owned device-side scatter.
    family_results: list[tuple[np.ndarray, np.ndarray, object, object]] = []

    for tag, family_key in ((poly_tag, GeometryFamily.POLYGON), (mpoly_tag, GeometryFamily.MULTIPOLYGON)):
        row_mask = tags == tag
        if not np.any(row_mask):
            continue
        if family_key not in owned.families:
            continue
        buf = owned.families[family_key]
        device_buffer = None
        if device_state is not None and family_key in device_state.families:
            device_buffer = device_state.families[family_key]
        if buf.row_count == 0:
            continue
        if device_buffer is not None:
            if int(device_buffer.geometry_offsets.size) < 2:
                continue
        elif buf.geometry_offsets is None or len(buf.geometry_offsets) < 2:
            continue

        family_rows_count = buf.row_count
        global_rows = np.flatnonzero(row_mask)
        family_rows = family_row_offsets[global_rows]

        # Build a compact mapping: for each family row that appears in this
        # batch, we launch the kernel once with family_rows_count items.
        needs_free = device_buffer is None
        if not needs_free:
            d_x = device_buffer.x
            d_y = device_buffer.y
            d_ring_offsets = device_buffer.ring_offsets
            d_geom_offsets = device_buffer.geometry_offsets
            d_part_offsets = None if family_key is GeometryFamily.POLYGON else device_buffer.part_offsets
        else:
            d_x = runtime.from_host(buf.x)
            d_y = runtime.from_host(buf.y)
            d_ring_offsets = runtime.from_host(buf.ring_offsets.astype(np.int32))
            d_geom_offsets = runtime.from_host(buf.geometry_offsets.astype(np.int32))
            d_part_offsets = (
                None
                if family_key is GeometryFamily.POLYGON
                else runtime.from_host(buf.part_offsets.astype(np.int32))
            )
        d_cx = runtime.allocate((family_rows_count,), np.float64)
        d_cy = runtime.allocate((family_rows_count,), np.float64)

        cx_cy_owned = False  # tracks whether d_cx/d_cy ownership transferred
        try:
            ptr = runtime.pointer
            if family_key is GeometryFamily.POLYGON:
                kernel = kernels["polygon_centroid"]
                params = (
                    (ptr(d_x), ptr(d_y), ptr(d_ring_offsets), ptr(d_geom_offsets),
                     ptr(d_cx), ptr(d_cy), center_x, center_y, family_rows_count),
                    (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_F64, KERNEL_PARAM_F64,
                     KERNEL_PARAM_I32),
                )
            else:
                kernel = kernels["multipolygon_centroid"]
                params = (
                    (ptr(d_x), ptr(d_y), ptr(d_ring_offsets), ptr(d_part_offsets), ptr(d_geom_offsets),
                     ptr(d_cx), ptr(d_cy), center_x, center_y, family_rows_count),
                    (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_F64, KERNEL_PARAM_F64,
                     KERNEL_PARAM_I32),
                )
            grid, block = runtime.launch_config(kernel, family_rows_count)
            runtime.launch(kernel, grid=grid, block=block, params=params)

            if return_owned:
                # Keep d_cx/d_cy on device; record for scatter below.
                family_results.append((global_rows, family_rows, d_cx, d_cy))
                cx_cy_owned = True
            else:
                family_cx = runtime.copy_device_to_host(d_cx)
                family_cy = runtime.copy_device_to_host(d_cy)

                # Scatter family results back to global row positions
                cx[global_rows] = family_cx[family_rows]
                cy[global_rows] = family_cy[family_rows]
        finally:
            if needs_free:
                runtime.free(d_x)
                runtime.free(d_y)
                runtime.free(d_ring_offsets)
                if d_part_offsets is not None:
                    runtime.free(d_part_offsets)
                runtime.free(d_geom_offsets)
            if not cx_cy_owned:
                runtime.free(d_cx)
                runtime.free(d_cy)

    if not return_owned:
        return cx, cy

    # ------------------------------------------------------------------
    # return_owned path: build a device-resident point OwnedGeometryArray
    # directly from device centroid buffers (zero D->H transfer).
    # ------------------------------------------------------------------
    if not family_results:
        from vibespatial.constructive.point import _empty_point_output
        return _empty_point_output()

    if len(family_results) == 1:
        # Single-family fast path: only one of polygon / multipolygon is
        # present.  Scatter family-indexed results to global positions on
        # device via CuPy fancy indexing.
        global_rows, family_rows, d_cx_family, d_cy_family = family_results[0]
        if global_rows.shape[0] == row_count:
            # All rows belong to this family -- the family d_cx/d_cy arrays
            # are already correctly sized (family_rows_count == row_count)
            # but indexed by family row.  Gather into global order.
            d_family_rows = cp.asarray(family_rows, dtype=np.int32)
            d_cx_global = d_cx_family[d_family_rows]
            d_cy_global = d_cy_family[d_family_rows]
            runtime.free(d_cx_family)
            runtime.free(d_cy_family)
        else:
            # Partial family: allocate global arrays, scatter.
            d_cx_global = cp.full(row_count, np.nan, dtype=np.float64)
            d_cy_global = cp.full(row_count, np.nan, dtype=np.float64)
            d_global_rows = cp.asarray(global_rows, dtype=np.int32)
            d_family_rows = cp.asarray(family_rows, dtype=np.int32)
            d_cx_global[d_global_rows] = d_cx_family[d_family_rows]
            d_cy_global[d_global_rows] = d_cy_family[d_family_rows]
            runtime.free(d_cx_family)
            runtime.free(d_cy_family)
    else:
        # Multi-family path: both polygon AND multipolygon families present.
        # Allocate global device arrays and scatter each family's results.
        d_cx_global = cp.full(row_count, np.nan, dtype=np.float64)
        d_cy_global = cp.full(row_count, np.nan, dtype=np.float64)
        for global_rows, family_rows, d_cx_family, d_cy_family in family_results:
            d_global_rows = cp.asarray(global_rows, dtype=np.int32)
            d_family_rows = cp.asarray(family_rows, dtype=np.int32)
            d_cx_global[d_global_rows] = d_cx_family[d_family_rows]
            d_cy_global[d_global_rows] = d_cy_family[d_family_rows]
            runtime.free(d_cx_family)
            runtime.free(d_cy_family)

    return _build_device_backed_point_output(
        d_cx_global, d_cy_global, row_count=row_count,
    )


def _polygon_buffer_kernels():
    return compile_kernel_group("polygon-buffer", _POLYGON_BUFFER_KERNEL_SOURCE, _POLYGON_BUFFER_KERNEL_NAMES)


def _build_device_backed_polygon_output_variable(
    device_x,
    device_y,
    *,
    row_count: int,
    geometry_offsets,
    ring_offsets,
    bounds: np.ndarray | None = None,
) -> OwnedGeometryArray:
    import cupy as cp

    return build_device_resident_owned(
        device_families={
            GeometryFamily.POLYGON: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.POLYGON,
                x=device_x,
                y=device_y,
                geometry_offsets=get_cuda_runtime().from_host(geometry_offsets),
                empty_mask=cp.zeros(row_count, dtype=cp.bool_),
                ring_offsets=get_cuda_runtime().from_host(ring_offsets),
                bounds=None if bounds is None else get_cuda_runtime().from_host(bounds),
            )
        },
        row_count=row_count,
        tags=cp.full(row_count, FAMILY_TAGS[GeometryFamily.POLYGON], dtype=cp.int8),
        validity=cp.ones(row_count, dtype=cp.bool_),
        family_row_offsets=cp.arange(row_count, dtype=cp.int32),
        execution_mode="gpu",
    )


def polygon_buffer_owned_array(
    polygons: OwnedGeometryArray,
    distance: float | np.ndarray,
    *,
    quad_segs: int = 8,
    join_style: str = "round",
    mitre_limit: float = 5.0,
    dispatch_mode: ExecutionMode = ExecutionMode.AUTO,
) -> OwnedGeometryArray:
    if GeometryFamily.POLYGON not in polygons.families or len(polygons.families) != 1:
        raise ValueError("polygon_buffer_owned_array requires a polygon-only OwnedGeometryArray")
    if not np.all(polygons.validity):
        raise ValueError("polygon_buffer_owned_array requires non-null rows only")
    if np.any(polygons.tags != FAMILY_TAGS[GeometryFamily.POLYGON]):
        raise ValueError("polygon_buffer_owned_array requires polygon-only rows")

    poly_buffer = polygons.families[GeometryFamily.POLYGON]
    if np.any(poly_buffer.empty_mask):
        raise ValueError("polygon_buffer_owned_array requires non-empty rows only")

    radii = (
        np.full(polygons.row_count, float(distance), dtype=np.float64)
        if np.isscalar(distance)
        else np.asarray(distance, dtype=np.float64)
    )
    if radii.shape != (polygons.row_count,):
        raise ValueError("distance must be a scalar or length-matched vector")

    join_int = _JOIN_STYLE_MAP.get(join_style, 0)

    selected_mode = plan_dispatch_selection(
        kernel_name="polygon_buffer",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=polygons.row_count,
        requested_mode=dispatch_mode,
    ).selected

    if selected_mode is not ExecutionMode.GPU:
        return build_polygon_buffers_cpu(
            polygons, radii, quad_segs=quad_segs,
            join_style=join_style, mitre_limit=mitre_limit,
        )

    return _build_polygon_buffers_gpu(
        polygons, radii, quad_segs=quad_segs,
        join_style=join_int, mitre_limit=mitre_limit,
    )
def _build_polygon_buffers_gpu(
    polygons: OwnedGeometryArray,
    radii: np.ndarray,
    *,
    quad_segs: int,
    join_style: int = 0,
    mitre_limit: float = 5.0,
) -> OwnedGeometryArray:
    polygons.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="polygon_buffer_owned_array selected GPU execution",
    )
    runtime = get_cuda_runtime()
    state = polygons._ensure_device_state()
    poly_buf = state.families[GeometryFamily.POLYGON]

    # Build ring-level mapping arrays on host
    host_poly_buf = polygons.families[GeometryFamily.POLYGON]
    input_geo_offsets = host_poly_buf.geometry_offsets
    ring_counts_per_poly = np.diff(input_geo_offsets).astype(np.int32)
    total_rings = int(input_geo_offsets[-1])

    if total_rings == 0:
        device_x = runtime.allocate((0,), np.float64)
        device_y = runtime.allocate((0,), np.float64)
        return _build_device_backed_polygon_output_variable(
            device_x, device_y,
            row_count=polygons.row_count,
            geometry_offsets=input_geo_offsets.copy(),
            ring_offsets=np.zeros(1, dtype=np.int32),
        )

    ring_to_row = np.repeat(
        np.arange(polygons.row_count, dtype=np.int32), ring_counts_per_poly
    )
    ring_is_hole = np.ones(total_rings, dtype=np.int32)
    ring_is_hole[input_geo_offsets[:-1]] = 0

    # Compute actual winding direction per ring via signed area (shoelace)
    # on GPU — one thread per ring, no Python loop.
    winding_kernels = _ring_winding_kernels()
    device_ring_winding = runtime.allocate((total_rings,), np.float64)
    winding_grid, winding_block = runtime.launch_config(
        winding_kernels["compute_ring_winding"], total_rings,
    )
    winding_params = (
        (
            runtime.pointer(poly_buf.x),
            runtime.pointer(poly_buf.y),
            runtime.pointer(poly_buf.ring_offsets),
            runtime.pointer(device_ring_winding),
            total_rings,
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
        ),
    )
    runtime.launch(
        winding_kernels["compute_ring_winding"],
        grid=winding_grid, block=winding_block, params=winding_params,
    )

    device_radii = runtime.from_host(radii)
    device_ring_to_row = runtime.from_host(ring_to_row)
    device_ring_is_hole = runtime.from_host(ring_is_hole)
    device_ring_counts = runtime.allocate((total_rings,), np.int32)
    device_ring_offsets = None
    device_x = None
    device_y = None
    success = False

    try:
        kernels = _polygon_buffer_kernels()
        ptr = runtime.pointer

        # Pass 1: count output vertices per ring
        count_params = (
            (
                ptr(device_ring_to_row),
                ptr(device_ring_is_hole),
                ptr(device_ring_winding),
                ptr(poly_buf.ring_offsets),
                ptr(poly_buf.x),
                ptr(poly_buf.y),
                ptr(device_radii),
                quad_segs,
                join_style,
                mitre_limit,
                ptr(device_ring_counts),
                total_rings,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        count_grid, count_block = runtime.launch_config(kernels["polygon_buffer_ring_count"], total_rings)
        runtime.launch(kernels["polygon_buffer_ring_count"],
                       grid=count_grid, block=count_block, params=count_params)

        # Compute exclusive prefix sum for scatter offsets
        device_ring_offsets = exclusive_sum(device_ring_counts)

        # Get total and start async full-counts D2H transfer.  The
        # transfer runs on a dedicated stream, overlapping with the
        # scatter kernel on the null stream.
        total_verts, xfer_stream, pinned_counts = count_scatter_total_with_transfer(
            runtime, device_ring_counts, device_ring_offsets,
        )

        if total_verts == 0:
            xfer_stream.synchronize()
            runtime.destroy_stream(xfer_stream)
            device_x = runtime.allocate((0,), np.float64)
            device_y = runtime.allocate((0,), np.float64)
            out_ring_offsets = np.zeros(total_rings + 1, dtype=np.int32)
            success = True
            return _build_device_backed_polygon_output_variable(
                device_x, device_y,
                row_count=polygons.row_count,
                geometry_offsets=input_geo_offsets.copy(),
                ring_offsets=out_ring_offsets,
            )

        # Allocate output coordinate arrays
        device_x = runtime.allocate((total_verts,), np.float64)
        device_y = runtime.allocate((total_verts,), np.float64)

        # Pass 2: scatter vertices
        scatter_params = (
            (
                ptr(device_ring_to_row),
                ptr(device_ring_is_hole),
                ptr(device_ring_winding),
                ptr(poly_buf.ring_offsets),
                ptr(poly_buf.x),
                ptr(poly_buf.y),
                ptr(device_radii),
                quad_segs,
                join_style,
                mitre_limit,
                ptr(device_ring_offsets),
                ptr(device_x),
                ptr(device_y),
                total_rings,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        scatter_grid, scatter_block = runtime.launch_config(kernels["polygon_buffer_ring_scatter"], total_rings)
        runtime.launch(kernels["polygon_buffer_ring_scatter"],
                       grid=scatter_grid, block=scatter_block, params=scatter_params)
        runtime.synchronize()

        # Counts D2H transfer was started before the scatter kernel;
        # wait for it now (should already be done).
        xfer_stream.synchronize()
        runtime.destroy_stream(xfer_stream)
        host_ring_counts = pinned_counts
        out_ring_offsets = np.empty(total_rings + 1, dtype=np.int32)
        out_ring_offsets[0] = 0
        np.cumsum(host_ring_counts, out=out_ring_offsets[1:])

        # geometry_offsets mirrors input (same ring structure per polygon)
        out_geometry_offsets = input_geo_offsets.copy()

        success = True
        return _build_device_backed_polygon_output_variable(
            device_x, device_y,
            row_count=polygons.row_count,
            geometry_offsets=out_geometry_offsets,
            ring_offsets=out_ring_offsets,
        )
    finally:
        runtime.free(device_radii)
        runtime.free(device_ring_to_row)
        runtime.free(device_ring_is_hole)
        runtime.free(device_ring_winding)
        runtime.free(device_ring_counts)
        if device_ring_offsets is not None:
            runtime.free(device_ring_offsets)
        if not success:
            if device_x is not None:
                runtime.free(device_x)
            if device_y is not None:
                runtime.free(device_y)


def polygon_centroids_owned(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = "auto",
    return_owned: bool = False,
) -> tuple[np.ndarray, np.ndarray] | OwnedGeometryArray:
    """Compute polygon centroids directly from OwnedGeometryArray coordinate buffers.

    Uses the shoelace formula on ring coordinates (exterior ring only):
      cx = sum((x_i + x_{i+1}) * cross_i) / (6 * area)
      cy = sum((y_i + y_{i+1}) * cross_i) / (6 * area)
      where cross_i = x_i * y_{i+1} - x_{i+1} * y_i

    When *return_owned* is False (default), returns (cx, cy) numpy arrays of
    shape (row_count,).

    When *return_owned* is True, returns a device-resident point
    ``OwnedGeometryArray`` built directly from GPU centroid buffers (zero
    D->H transfer on the GPU path).  On CPU fallback the centroids are
    computed on host and then uploaded via ``point_owned_from_xy_device``.

    GPU path uses ADR-0002 precision dispatch: fp32 with Kahan summation +
    coordinate centering on consumer GPUs, native fp64 on datacenter GPUs.
    """

    row_count = owned.row_count
    if row_count == 0:
        if return_owned:
            from vibespatial.constructive.point import _empty_point_output
            return _empty_point_output()
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    # Dispatch decision
    selection = plan_dispatch_selection(
        kernel_name="polygon_centroid",
        kernel_class=KernelClass.METRIC,
        row_count=row_count,
        requested_mode=dispatch_mode,
    )
    if selection.selected is ExecutionMode.GPU:
        from vibespatial.runtime.precision import CoordinateStats
        max_abs, coord_min, coord_max = _coord_stats_from_owned(owned)
        span = coord_max - coord_min if np.isfinite(coord_min) else 0.0
        # The centroid shoelace formula involves products of coordinates
        # (xi*yi1 - xi1*yi) which require constructive-level precision.
        # fp32 introduces unacceptable absolute errors even with Kahan
        # summation and coordinate centering (observed >1 unit error for
        # coordinate ranges [0, 1000]).  Request CONSTRUCTIVE precision
        # planning to guarantee fp64 compute on consumer GPUs.
        selection = plan_dispatch_selection(
            kernel_name="polygon_centroid",
            kernel_class=KernelClass.METRIC,
            row_count=row_count,
            requested_mode=dispatch_mode,
            requested_precision=precision,
            precision_kernel_class=KernelClass.CONSTRUCTIVE,
            coordinate_stats=CoordinateStats(max_abs_coord=max_abs, span=span),
        )
        precision_plan = selection.precision_plan
        result = _polygon_centroids_gpu(
            owned, precision_plan=precision_plan, return_owned=return_owned,
        )
        if result is not None:
            return result

    # CPU fallback
    cx_cpu, cy_cpu = _polygon_centroids_cpu(owned)
    if return_owned:
        from vibespatial.constructive.point import point_owned_from_xy_device
        return point_owned_from_xy_device(cx_cpu, cy_cpu)
    return cx_cpu, cy_cpu


@register_kernel_variant(
    "polygon_centroid",
    "cpu",
    kernel_class=KernelClass.METRIC,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=("polygon", "multipolygon"),
    supports_mixed=True,
    tags=("numpy", "metric", "centroid"),
)
def _polygon_centroids_cpu(
    owned: OwnedGeometryArray,
) -> tuple[np.ndarray, np.ndarray]:
    """CPU fallback: area-weighted centroid across polygon rings and parts."""
    row_count = owned.row_count
    cx = np.empty(row_count, dtype=np.float64)
    cy = np.empty(row_count, dtype=np.float64)

    poly_tag = FAMILY_TAGS[GeometryFamily.POLYGON]
    mpoly_tag = FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]

    tags = owned.tags
    family_row_offsets = owned.family_row_offsets

    def _ring_centroid_terms(
        xs: np.ndarray,
        ys: np.ndarray,
    ) -> tuple[float, float, float, float, float, int]:
        if len(xs) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0
        if len(xs) >= 2 and xs[0] == xs[-1] and ys[0] == ys[-1]:
            xs = xs[:-1]
            ys = ys[:-1]
        if len(xs) < 3:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0

        xs_next = np.roll(xs, -1)
        ys_next = np.roll(ys, -1)
        cross = xs * ys_next - xs_next * ys
        signed_area = float(cross.sum() * 0.5)
        if abs(signed_area) < 1e-30:
            return 0.0, 0.0, 0.0, float(xs.sum()), float(ys.sum()), len(xs)

        ring_cx = float(((xs + xs_next) * cross).sum() / (6.0 * signed_area))
        ring_cy = float(((ys + ys_next) * cross).sum() / (6.0 * signed_area))
        return abs(signed_area), ring_cx, ring_cy, 0.0, 0.0, 0

    for tag, family_key in ((poly_tag, GeometryFamily.POLYGON), (mpoly_tag, GeometryFamily.MULTIPOLYGON)):
        row_mask = tags == tag
        if not np.any(row_mask):
            continue
        if family_key not in owned.families:
            continue
        buf = owned.families[family_key]
        if buf.row_count == 0 or buf.geometry_offsets is None or len(buf.geometry_offsets) < 2:
            continue
        x = buf.x
        y = buf.y
        ring_offsets = buf.ring_offsets
        geom_offsets = buf.geometry_offsets
        part_offsets = buf.part_offsets

        global_rows = np.flatnonzero(row_mask)
        family_rows = family_row_offsets[global_rows]

        for gr, fr in zip(global_rows, family_rows, strict=True):
            total_area = 0.0
            total_cx = 0.0
            total_cy = 0.0
            fallback_x = 0.0
            fallback_y = 0.0
            fallback_n = 0

            if family_key is GeometryFamily.POLYGON:
                ring_start = int(geom_offsets[fr])
                ring_stop = int(geom_offsets[fr + 1])
                polygon_ranges = [(ring_start, ring_stop)]
            else:
                poly_start = int(geom_offsets[fr])
                poly_stop = int(geom_offsets[fr + 1])
                polygon_ranges = [
                    (int(part_offsets[poly_idx]), int(part_offsets[poly_idx + 1]))
                    for poly_idx in range(poly_start, poly_stop)
                ]

            for ring_start, ring_stop in polygon_ranges:
                for ring_idx in range(ring_start, ring_stop):
                    coord_start = int(ring_offsets[ring_idx])
                    coord_end = int(ring_offsets[ring_idx + 1])
                    xs = x[coord_start:coord_end]
                    ys = y[coord_start:coord_end]
                    ring_area, ring_cx, ring_cy, mean_x, mean_y, mean_n = _ring_centroid_terms(xs, ys)
                    if mean_n > 0:
                        fallback_x += mean_x
                        fallback_y += mean_y
                        fallback_n += mean_n
                        continue
                    if ring_area <= 0.0:
                        continue
                    weight = ring_area if ring_idx == ring_start else -ring_area
                    total_area += weight
                    total_cx += ring_cx * weight
                    total_cy += ring_cy * weight

            if abs(total_area) >= 1e-30:
                cx[gr] = total_cx / total_area
                cy[gr] = total_cy / total_area
            elif fallback_n > 0:
                cx[gr] = fallback_x / fallback_n
                cy[gr] = fallback_y / fallback_n
            else:
                cx[gr] = np.nan
                cy[gr] = np.nan

    # Handle any point/line rows that somehow got here (shouldn't happen, but safe)
    remaining = ~np.isin(tags, [poly_tag, mpoly_tag])
    if np.any(remaining):
        cx[remaining] = np.nan
        cy[remaining] = np.nan

    return cx, cy
