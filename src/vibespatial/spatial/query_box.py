from __future__ import annotations

import numpy as np
from shapely.geometry.base import BaseGeometry

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - CPU-only installs
    cp = None

from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.cccl_primitives import compact_indices, exclusive_sum
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection

request_warmup(["select_i32", "select_i64", "exclusive_scan_i32", "exclusive_scan_i64"])
from vibespatial.cuda._runtime import (  # noqa: E402
    KERNEL_PARAM_F64,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    count_scatter_total,
    get_cuda_runtime,
)
from vibespatial.geometry.buffers import GeometryFamily  # noqa: E402
from vibespatial.geometry.owned import OwnedGeometryArray  # noqa: E402
from vibespatial.kernels.core.geometry_analysis import (  # noqa: E402
    compute_geometry_bounds,
    compute_geometry_bounds_device,
)
from vibespatial.kernels.core.spatial_query_kernels import _spatial_query_kernels  # noqa: E402
from vibespatial.runtime import has_gpu_runtime  # noqa: E402
from vibespatial.runtime.precision import KernelClass  # noqa: E402
from vibespatial.runtime.residency import Residency, TransferTrigger  # noqa: E402

from .query_types import _DeviceCandidates  # noqa: E402
from .query_utils import _gpu_bounds_dispatch_mode  # noqa: E402


def _device_scalar_bool(value, *, reason: str) -> bool:
    runtime = get_cuda_runtime()
    host = runtime.copy_device_to_host(
        cp.asarray(value, dtype=cp.bool_).reshape(1),
        reason=reason,
    )
    return bool(np.asarray(host).reshape(-1)[0])


def _query_regular_grid_point_index(
    flat_index,
    query_owned: OwnedGeometryArray,
    *,
    predicate: str | None,
) -> _DeviceCandidates | tuple[np.ndarray, np.ndarray] | None:
    metadata = getattr(flat_index, "regular_grid", None)
    if metadata is None or predicate not in (None, "intersects"):
        return None
    if GeometryFamily.POINT not in query_owned.families or len(query_owned.families) != 1:
        return None
    selection = plan_dispatch_selection(
        kernel_name="point_regular_grid_candidates",
        kernel_class=KernelClass.COARSE,
        row_count=query_owned.row_count,
        gpu_available=has_gpu_runtime(),
        current_residency=query_owned.residency,
    )
    if selection.selected is not ExecutionMode.GPU:
        return None

    query_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="query_spatial_index selected regular-grid point GPU execution",
    )
    runtime = get_cuda_runtime()
    state = query_owned._ensure_device_state()
    point_buffer = state.families[GeometryFamily.POINT]
    device_right = runtime.allocate((query_owned.row_count * 4,), np.int32)
    device_counts = runtime.allocate((query_owned.row_count,), np.uint8)
    device_counts_i32 = None
    device_offsets = None
    device_left_out = None
    device_right_out = None
    try:
        kernel = _spatial_query_kernels()["point_regular_grid_candidates"]
        ptr = runtime.pointer
        params = (
            (
                ptr(state.family_row_offsets),
                ptr(point_buffer.geometry_offsets),
                ptr(point_buffer.empty_mask),
                ptr(point_buffer.x),
                ptr(point_buffer.y),
                metadata.origin_x,
                metadata.origin_y,
                metadata.cell_width,
                metadata.cell_height,
                metadata.cols,
                metadata.rows,
                metadata.size,
                ptr(device_right),
                ptr(device_counts),
                query_owned.row_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        grid, block = runtime.launch_config(kernel, query_owned.row_count)
        runtime.launch(kernel, grid=grid, block=block, params=params)
        device_counts_i32 = device_counts.astype(np.int32)
        device_offsets = exclusive_sum(device_counts_i32)
        total_pairs = (
            count_scatter_total(
                runtime,
                device_counts_i32,
                device_offsets,
                reason="spatial query regular-grid point-pair allocation fence",
            )
            if query_owned.row_count
            else 0
        )
        if total_pairs == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

        device_left_out = runtime.allocate((total_pairs,), np.int32)
        device_right_out = runtime.allocate((total_pairs,), np.int32)
        scatter_kernel = _spatial_query_kernels()["point_regular_grid_scatter_pairs"]
        scatter_params = (
            (
                ptr(device_right),
                ptr(device_offsets),
                ptr(device_counts_i32),
                ptr(device_left_out),
                ptr(device_right_out),
                query_owned.row_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        scatter_grid, scatter_block = runtime.launch_config(scatter_kernel, query_owned.row_count)
        runtime.launch(
            scatter_kernel,
            grid=scatter_grid,
            block=scatter_block,
            params=scatter_params,
        )
        # Return device-resident candidates (ADR-0005: no mid-pipeline D→H).
        result = _DeviceCandidates(
            d_left=device_left_out,
            d_right=device_right_out,
            total_pairs=total_pairs,
        )
        # Prevent finally block from freeing the returned device arrays.
        device_left_out = None
        device_right_out = None
        return result
    finally:
        runtime.free(device_right)
        runtime.free(device_counts)
        runtime.free(device_counts_i32)
        runtime.free(device_offsets)
        runtime.free(device_left_out)
        runtime.free(device_right_out)


def _query_regular_grid_rect_box_index(
    flat_index,
    query_bounds: np.ndarray | None,
    *,
    predicate: str | None,
) -> _DeviceCandidates | tuple[np.ndarray, np.ndarray] | None:
    metadata = getattr(flat_index, "regular_grid", None)
    if metadata is None or predicate not in (None, "intersects"):
        return None
    if query_bounds is None:
        return None
    query_count = int(query_bounds.shape[0])
    if query_count == 0:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty

    selection = plan_dispatch_selection(
        kernel_name="bbox_overlap_candidates",
        kernel_class=KernelClass.COARSE,
        row_count=query_count * flat_index.size,
        gpu_available=has_gpu_runtime(),
        current_residency=flat_index.geometry_array.residency,
    )
    if selection.selected is not ExecutionMode.GPU:
        return None

    runtime = get_cuda_runtime()
    device_query_bounds = runtime.from_host(np.ascontiguousarray(query_bounds, dtype=np.float64).ravel())
    device_counts = runtime.allocate((query_count,), np.int32)
    device_offsets = None
    device_left = None
    device_right = None
    try:
        kernels = _spatial_query_kernels()
        ptr = runtime.pointer

        count_params = (
            (
                ptr(device_query_bounds),
                metadata.origin_x,
                metadata.origin_y,
                metadata.cell_width,
                metadata.cell_height,
                metadata.cols,
                metadata.rows,
                metadata.size,
                ptr(device_counts),
                query_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        count_grid, count_block = runtime.launch_config(kernels["regular_grid_box_overlap_count"], query_count)
        runtime.launch(
            kernels["regular_grid_box_overlap_count"],
            grid=count_grid,
            block=count_block,
            params=count_params,
        )

        device_offsets = exclusive_sum(device_counts)
        total_pairs = (
            count_scatter_total(
                runtime,
                device_counts,
                device_offsets,
                reason="spatial query regular-grid box-pair allocation fence",
            )
            if query_count > 0
            else 0
        )
        if total_pairs == 0:
            empty = np.empty(0, dtype=np.int32)
            return empty, empty

        device_left = runtime.allocate((total_pairs,), np.int32)
        device_right = runtime.allocate((total_pairs,), np.int32)
        scatter_params = (
            (
                ptr(device_query_bounds),
                metadata.origin_x,
                metadata.origin_y,
                metadata.cell_width,
                metadata.cell_height,
                metadata.cols,
                metadata.rows,
                metadata.size,
                ptr(device_offsets),
                ptr(device_left),
                ptr(device_right),
                query_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        scatter_grid, scatter_block = runtime.launch_config(kernels["regular_grid_box_overlap_scatter"], query_count)
        runtime.launch(
            kernels["regular_grid_box_overlap_scatter"],
            grid=scatter_grid,
            block=scatter_block,
            params=scatter_params,
        )

        # Return device-resident candidates (ADR-0005: no mid-pipeline D→H).
        result = _DeviceCandidates(
            d_left=device_left,
            d_right=device_right,
            total_pairs=total_pairs,
        )
        # Prevent finally block from freeing the returned device arrays.
        device_left = None
        device_right = None
        return result
    finally:
        runtime.free(device_query_bounds)
        runtime.free(device_counts)
        runtime.free(device_offsets)
        runtime.free(device_left)
        runtime.free(device_right)


def _is_axis_aligned_box(geometry: BaseGeometry) -> bool:
    if geometry is None or geometry.is_empty or geometry.geom_type != "Polygon":
        return False
    if len(getattr(geometry, "interiors", ())) != 0:
        return False
    coords = list(geometry.exterior.coords)
    if len(coords) != 5:
        return False
    minx, miny, maxx, maxy = geometry.bounds
    tol = 1e-9 * max(abs(maxx - minx), abs(maxy - miny), 1.0)
    x_matches = [abs(coord[0] - minx) <= tol or abs(coord[0] - maxx) <= tol for coord in coords]
    y_matches = [abs(coord[1] - miny) <= tol or abs(coord[1] - maxy) <= tol for coord in coords]
    if not all(x_matches) or not all(y_matches):
        return False
    edge_same_x = [abs(coords[index + 1][0] - coords[index][0]) <= tol for index in range(4)]
    edge_same_y = [abs(coords[index + 1][1] - coords[index][1]) <= tol for index in range(4)]
    return all(same_x ^ same_y for same_x, same_y in zip(edge_same_x, edge_same_y, strict=True))


def _coords_form_axis_aligned_box(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float, float] | None:
    if xs.size != 5 or ys.size != 5:
        return None
    minx = float(xs.min())
    miny = float(ys.min())
    maxx = float(xs.max())
    maxy = float(ys.max())
    tol = 1e-9 * max(abs(maxx - minx), abs(maxy - miny), 1.0)
    if abs(xs[0] - xs[-1]) > tol or abs(ys[0] - ys[-1]) > tol:
        return None
    x_matches = np.isclose(xs, minx, atol=tol) | np.isclose(xs, maxx, atol=tol)
    y_matches = np.isclose(ys, miny, atol=tol) | np.isclose(ys, maxy, atol=tol)
    if not bool(np.all(x_matches) and np.all(y_matches)):
        return None
    edge_same_x = np.isclose(xs[1:], xs[:-1], atol=tol)
    edge_same_y = np.isclose(ys[1:], ys[:-1], atol=tol)
    if not bool(np.all(np.logical_xor(edge_same_x, edge_same_y))):
        return None
    return minx, miny, maxx, maxy


def _extract_box_query_bounds(
    predicate: str,
    query_values: np.ndarray,
) -> np.ndarray | None:
    bounds = np.full((len(query_values), 4), np.nan, dtype=np.float64)
    for row_index, geometry in enumerate(query_values):
        if geometry is None or geometry.is_empty:
            continue
        if not isinstance(geometry, BaseGeometry) or not _is_axis_aligned_box(geometry):
            return None
        bounds[row_index] = np.asarray(geometry.bounds, dtype=np.float64)
    return bounds


def _extract_box_query_bounds_shapely(query_values: np.ndarray) -> np.ndarray | None:
    """Vectorized box detection from Shapely arrays — no OwnedGeometryArray needed.

    Mirrors ``_extract_owned_polygon_box_bounds`` logic but operates directly
    on Shapely geometry arrays using vectorized shapely functions.  Returns
    (N, 4) bounds array if all non-empty geometries are axis-aligned boxes,
    or None otherwise.
    """
    import shapely as _shapely

    n = len(query_values)
    if n == 0:
        return np.full((0, 4), np.nan, dtype=np.float64)

    missing = _shapely.is_missing(query_values)
    empty = np.zeros(n, dtype=bool)
    non_missing = ~missing
    empty[non_missing] = _shapely.is_empty(query_values[non_missing])
    valid = non_missing & ~empty

    if not np.any(valid):
        return np.full((n, 4), np.nan, dtype=np.float64)

    valid_geoms = query_values[valid]

    # All valid geometries must be Polygons (type_id == 3).
    type_ids = _shapely.get_type_id(valid_geoms)
    if not np.all(type_ids == 3):
        return None

    # No interior rings.
    interior_counts = _shapely.get_num_interior_rings(valid_geoms)
    if not np.all(interior_counts == 0):
        return None

    valid_bounds = _shapely.bounds(valid_geoms)
    minx = valid_bounds[:, 0]
    miny = valid_bounds[:, 1]
    maxx = valid_bounds[:, 2]
    maxy = valid_bounds[:, 3]
    bbox_area = (maxx - minx) * (maxy - miny)
    if not np.all(bbox_area > 0.0):
        return None

    # For a valid polygon with no holes, area equal to its envelope area means
    # it covers the whole axis-aligned envelope. This accepts harmless
    # collinear-vertex boxes while rejecting L-shapes and rotated polygons.
    area = _shapely.area(valid_geoms)
    tol = 1e-9 * np.maximum(np.abs(bbox_area), 1.0)
    if not np.all(np.abs(area - bbox_area) <= tol):
        return None

    bounds = np.full((n, 4), np.nan, dtype=np.float64)
    bounds[valid, 0] = minx
    bounds[valid, 1] = miny
    bounds[valid, 2] = maxx
    bounds[valid, 3] = maxy
    return bounds


def _extract_owned_polygon_box_bounds(query_owned: OwnedGeometryArray) -> np.ndarray | None:
    if GeometryFamily.POLYGON not in query_owned.families or len(query_owned.families) != 1:
        return None
    polygon_buffer = query_owned.families[GeometryFamily.POLYGON]
    if polygon_buffer.ring_offsets is None:
        return None

    n = query_owned.row_count
    if n == 0:
        return np.full((0, 4), np.nan, dtype=np.float64)

    # Vectorized check: all polygons must have exactly 1 ring.
    geo_offsets = polygon_buffer.geometry_offsets
    ring_counts = geo_offsets[1:] - geo_offsets[:-1]
    if not np.all(ring_counts == 1):
        return None

    # Vectorized check: all rings must have exactly 5 coords.
    ring_offsets = polygon_buffer.ring_offsets
    coord_counts = ring_offsets[1:] - ring_offsets[:-1]
    if coord_counts.size != n or not np.all(coord_counts == 5):
        return None

    if (
        cp is not None
        and query_owned.residency is Residency.DEVICE
    ):
        state = query_owned._ensure_device_state()
        polygon_state = state.families[GeometryFamily.POLYGON]
        if polygon_state.ring_offsets is None:
            return None

        d_geom_starts = cp.asarray(polygon_state.geometry_offsets[:-1]).astype(cp.int32, copy=False)
        d_geom_ends = cp.asarray(polygon_state.geometry_offsets[1:]).astype(cp.int32, copy=False)
        if not _device_scalar_bool(
            cp.all((d_geom_ends - d_geom_starts) == 1),
            reason="spatial query polygon-box single-ring scalar fence",
        ):
            return None

        d_ring_offsets = cp.asarray(polygon_state.ring_offsets).astype(cp.int32, copy=False)
        d_coord_starts = d_ring_offsets[d_geom_starts]
        d_coord_ends = d_ring_offsets[d_geom_ends]
        if not _device_scalar_bool(
            cp.all((d_coord_ends - d_coord_starts) == 5),
            reason="spatial query polygon-box coordinate-count scalar fence",
        ):
            return None

        compute_geometry_bounds_device(query_owned)
        d_bounds = cp.asarray(query_owned.device_state.row_bounds).reshape(n, 4)
        d_offsets = d_coord_starts[:, None] + cp.arange(5, dtype=cp.int32)[None, :]
        d_x = cp.asarray(polygon_state.x)[d_offsets]
        d_y = cp.asarray(polygon_state.y)[d_offsets]

        d_minx = d_bounds[:, 0][:, None]
        d_miny = d_bounds[:, 1][:, None]
        d_maxx = d_bounds[:, 2][:, None]
        d_maxy = d_bounds[:, 3][:, None]
        d_tol = 1e-9 * cp.maximum(
            cp.maximum(cp.abs(d_bounds[:, 2] - d_bounds[:, 0]), cp.abs(d_bounds[:, 3] - d_bounds[:, 1])),
            1.0,
        )
        d_tol_2d = d_tol[:, None]

        closed = (
            (cp.abs(d_x[:, 0] - d_x[:, -1]) <= d_tol)
            & (cp.abs(d_y[:, 0] - d_y[:, -1]) <= d_tol)
        )
        x_at_min_or_max = (cp.abs(d_x - d_minx) <= d_tol_2d) | (cp.abs(d_x - d_maxx) <= d_tol_2d)
        y_at_min_or_max = (cp.abs(d_y - d_miny) <= d_tol_2d) | (cp.abs(d_y - d_maxy) <= d_tol_2d)
        edge_same_x = cp.abs(d_x[:, 1:] - d_x[:, :-1]) <= d_tol_2d
        edge_same_y = cp.abs(d_y[:, 1:] - d_y[:, :-1]) <= d_tol_2d
        if not _device_scalar_bool(
            cp.all(
                closed
                & cp.all(x_at_min_or_max, axis=1)
                & cp.all(y_at_min_or_max, axis=1)
                & cp.all(cp.logical_xor(edge_same_x, edge_same_y), axis=1)
            ),
            reason="spatial query polygon-box axis-aligned scalar fence",
        ):
            return None

        d_row_offsets = cp.asarray(state.family_row_offsets).astype(cp.int32, copy=False)
        d_valid = d_row_offsets >= 0
        d_polygon_rows = cp.clip(d_row_offsets, 0, None)
        d_empty_mask = cp.asarray(polygon_state.empty_mask).astype(cp.bool_, copy=False)
        d_valid = d_valid & ~d_empty_mask[d_polygon_rows]
        d_bounds = cp.where(d_valid[:, None], d_bounds, cp.nan)
        return get_cuda_runtime().copy_device_to_host(
            d_bounds,
            reason="spatial query polygon-box bounds host export",
        )

    # Structural checks above use only offsets (available on host even for
    # device-resident OGAs).  Coordinate verification needs x/y buffers;
    # lazily materialise them -- _ensure_host_state skips already-populated
    # offsets and only transfers coordinates.
    query_owned._ensure_host_state()
    polygon_buffer = query_owned.families[GeometryFamily.POLYGON]
    # Reshape coords to (N, 5) for vectorized box validation.
    all_x = polygon_buffer.x.reshape(n, 5)
    all_y = polygon_buffer.y.reshape(n, 5)

    minx = all_x.min(axis=1)
    maxx = all_x.max(axis=1)
    miny = all_y.min(axis=1)
    maxy = all_y.max(axis=1)

    tol = 1e-9 * np.maximum(np.maximum(np.abs(maxx - minx), np.abs(maxy - miny)), 1.0)

    # Closed ring check.
    if not np.all(np.abs(all_x[:, 0] - all_x[:, -1]) <= tol):
        return None
    if not np.all(np.abs(all_y[:, 0] - all_y[:, -1]) <= tol):
        return None

    # All coords must be at min or max x/y.
    tol_2d = tol[:, None]
    x_at_min_or_max = (np.abs(all_x - minx[:, None]) <= tol_2d) | (np.abs(all_x - maxx[:, None]) <= tol_2d)
    y_at_min_or_max = (np.abs(all_y - miny[:, None]) <= tol_2d) | (np.abs(all_y - maxy[:, None]) <= tol_2d)
    if not np.all(x_at_min_or_max):
        return None
    if not np.all(y_at_min_or_max):
        return None

    # Each edge must be axis-aligned (same x or same y, not both).
    edge_same_x = np.abs(all_x[:, 1:] - all_x[:, :-1]) <= tol_2d
    edge_same_y = np.abs(all_y[:, 1:] - all_y[:, :-1]) <= tol_2d
    if not np.all(np.logical_xor(edge_same_x, edge_same_y)):
        return None

    # Build result bounds, marking empty/invalid rows as NaN.
    bounds = np.full((n, 4), np.nan, dtype=np.float64)
    valid = query_owned.family_row_offsets >= 0
    polygon_rows = query_owned.family_row_offsets.clip(0)
    valid = valid & ~polygon_buffer.empty_mask[polygon_rows]
    bounds[valid, 0] = minx[valid]
    bounds[valid, 1] = miny[valid]
    bounds[valid, 2] = maxx[valid]
    bounds[valid, 3] = maxy[valid]
    return bounds


def _extract_box_query_bounds_from_owned(
    predicate: str | None,
    query_owned: OwnedGeometryArray,
) -> np.ndarray | None:
    if predicate is None:
        return compute_geometry_bounds(query_owned, dispatch_mode=_gpu_bounds_dispatch_mode(query_owned))
    return _extract_owned_polygon_box_bounds(query_owned)


def _point_box_predicate_mode(predicate: str | None) -> int | None:
    if predicate in (None, "intersects", "covers"):
        return 0
    if predicate in ("contains", "contains_properly"):
        return 1
    if predicate == "touches":
        return 2
    return None


def _query_point_tree_box_index(
    tree_owned: OwnedGeometryArray,
    *,
    predicate: str | None,
    query_row_count: int,
    box_bounds: np.ndarray | None,
    force_gpu: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    predicate_mode = _point_box_predicate_mode(predicate)
    if predicate_mode is None:
        return None
    if GeometryFamily.POINT not in tree_owned.families or len(tree_owned.families) != 1:
        return None
    if query_row_count == 0:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty
    if box_bounds is None:
        return None

    try:
        selection = plan_dispatch_selection(
            kernel_name="point_box_query",
            kernel_class=KernelClass.COARSE,
            row_count=tree_owned.row_count * query_row_count,
            requested_mode=ExecutionMode.GPU if force_gpu else ExecutionMode.AUTO,
            gpu_available=has_gpu_runtime(),
            current_residency=tree_owned.residency,
        )
    except RuntimeError:
        return None
    if selection.selected is not ExecutionMode.GPU:
        return None

    tree_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="query_spatial_index selected GPU point-tree box execution",
    )
    runtime = get_cuda_runtime()
    state = tree_owned._ensure_device_state()
    point_buffer = state.families[GeometryFamily.POINT]
    kernel = _spatial_query_kernels()["point_box_query_mask"]
    ptr = runtime.pointer
    grid, block = runtime.launch_config(kernel, tree_owned.row_count)

    left_out: list[np.ndarray] = []
    right_out: list[np.ndarray] = []
    device_mask = runtime.allocate((tree_owned.row_count,), np.uint8)
    try:
        for query_index, bounds in enumerate(box_bounds):
            if np.isnan(bounds).any():
                continue
            params = (
                (
                    ptr(state.family_row_offsets),
                    ptr(point_buffer.geometry_offsets),
                    ptr(point_buffer.empty_mask),
                    ptr(point_buffer.x),
                    ptr(point_buffer.y),
                    float(bounds[0]),
                    float(bounds[1]),
                    float(bounds[2]),
                    float(bounds[3]),
                    predicate_mode,
                    ptr(device_mask),
                    tree_owned.row_count,
                ),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                ),
            )
            runtime.launch(kernel, grid=grid, block=block, params=params)
            matched = compact_indices(device_mask).values
            matched_host = runtime.copy_device_to_host(
                matched,
                reason="spatial query point-box matched-row host export",
            ).astype(np.int32, copy=False)
            if matched_host.size == 0:
                continue
            left_out.append(np.full(matched_host.size, query_index, dtype=np.int32))
            right_out.append(matched_host)
    finally:
        runtime.free(device_mask)

    if not left_out:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty
    return np.concatenate(left_out), np.concatenate(right_out)


def _query_point_tree_box_row_positions_device(
    tree_owned: OwnedGeometryArray,
    *,
    predicate: str | None,
    box_bounds: np.ndarray,
    force_gpu: bool = False,
):
    """Return device row positions for one point-tree vs box predicate query."""
    predicate_mode = _point_box_predicate_mode(predicate)
    if predicate_mode is None:
        return None
    if GeometryFamily.POINT not in tree_owned.families or len(tree_owned.families) != 1:
        return None
    bounds = np.asarray(box_bounds, dtype=np.float64)
    if bounds.shape == (1, 4):
        bounds = bounds[0]
    if bounds.shape != (4,) or np.isnan(bounds).any():
        return None

    try:
        selection = plan_dispatch_selection(
            kernel_name="point_box_query",
            kernel_class=KernelClass.COARSE,
            row_count=tree_owned.row_count,
            requested_mode=ExecutionMode.GPU if force_gpu else ExecutionMode.AUTO,
            gpu_available=has_gpu_runtime(),
            current_residency=tree_owned.residency,
        )
    except RuntimeError:
        return None
    if selection.selected is not ExecutionMode.GPU:
        return None

    tree_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="query_spatial_index selected GPU point-tree box execution",
    )
    runtime = get_cuda_runtime()
    state = tree_owned._ensure_device_state()
    point_buffer = state.families[GeometryFamily.POINT]
    kernel = _spatial_query_kernels()["point_box_query_mask"]
    ptr = runtime.pointer
    grid, block = runtime.launch_config(kernel, tree_owned.row_count)
    device_mask = runtime.allocate((tree_owned.row_count,), np.uint8)
    try:
        params = (
            (
                ptr(state.family_row_offsets),
                ptr(point_buffer.geometry_offsets),
                ptr(point_buffer.empty_mask),
                ptr(point_buffer.x),
                ptr(point_buffer.y),
                float(bounds[0]),
                float(bounds[1]),
                float(bounds[2]),
                float(bounds[3]),
                predicate_mode,
                ptr(device_mask),
                tree_owned.row_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        runtime.launch(kernel, grid=grid, block=block, params=params)
        positions = compact_indices(device_mask).values
        # Keep the temporary mask alive until compaction has consumed it.
        runtime.synchronize()
        return positions
    finally:
        runtime.free(device_mask)
