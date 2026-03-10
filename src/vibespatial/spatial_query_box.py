from __future__ import annotations

import numpy as np
from shapely.geometry.base import BaseGeometry

from vibespatial.adaptive_runtime import plan_kernel_dispatch
from vibespatial.cccl_primitives import compact_indices, exclusive_sum
from vibespatial.cccl_precompile import request_warmup
from vibespatial.crossover import DispatchDecision

request_warmup(["select_i32", "select_i64", "exclusive_scan_i32", "exclusive_scan_i64"])
from vibespatial.cuda_runtime import (  # noqa: E402
    KERNEL_PARAM_F64,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    count_scatter_total,
    get_cuda_runtime,
)
from vibespatial.geometry_buffers import GeometryFamily  # noqa: E402
from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds  # noqa: E402
from vibespatial.kernels.core.spatial_query_kernels import _spatial_query_kernels  # noqa: E402
from vibespatial.owned_geometry import OwnedGeometryArray  # noqa: E402
from vibespatial.precision import KernelClass  # noqa: E402
from vibespatial.residency import Residency, TransferTrigger  # noqa: E402
from vibespatial.runtime import has_gpu_runtime  # noqa: E402
from vibespatial.spatial_query_types import _DeviceCandidates  # noqa: E402
from vibespatial.spatial_query_utils import _gpu_bounds_dispatch_mode  # noqa: E402


def _query_regular_grid_point_index(
    flat_index,
    query_owned: OwnedGeometryArray,
    *,
    predicate: str | None,
) -> _DeviceCandidates | tuple[np.ndarray, np.ndarray] | None:
    metadata = getattr(flat_index, "regular_grid", None)
    if metadata is None or predicate not in (None, "intersects") or not has_gpu_runtime():
        return None
    if GeometryFamily.POINT not in query_owned.families or len(query_owned.families) != 1:
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
        total_pairs = count_scatter_total(runtime, device_counts_i32, device_offsets) if query_owned.row_count else 0
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
    if metadata is None or predicate not in (None, "intersects") or not has_gpu_runtime():
        return None
    if query_bounds is None:
        return None
    query_count = int(query_bounds.shape[0])
    if query_count == 0:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty

    plan = plan_kernel_dispatch(
        kernel_name="bbox_overlap_candidates",
        kernel_class=KernelClass.COARSE,
        row_count=query_count * flat_index.size,
        gpu_available=has_gpu_runtime(),
    )
    dispatch_decision = plan.dispatch_decision
    if dispatch_decision is not DispatchDecision.GPU:
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
        total_pairs = count_scatter_total(runtime, device_counts, device_offsets) if query_count > 0 else 0
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
        return compute_geometry_bounds(query_owned, dispatch_mode=_gpu_bounds_dispatch_mode())
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
    if predicate_mode is None or not has_gpu_runtime():
        return None
    if GeometryFamily.POINT not in tree_owned.families or len(tree_owned.families) != 1:
        return None
    if query_row_count == 0:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty
    if box_bounds is None:
        return None

    if not force_gpu:
        plan = plan_kernel_dispatch(
            kernel_name="point_box_query",
            kernel_class=KernelClass.COARSE,
            row_count=tree_owned.row_count * query_row_count,
            gpu_available=has_gpu_runtime(),
        )
        dispatch_decision = plan.dispatch_decision
        if dispatch_decision is not DispatchDecision.GPU:
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
            matched_host = runtime.copy_device_to_host(matched).astype(np.int32, copy=False)
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
