from __future__ import annotations

import numpy as np

from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    OwnedGeometryArray,
    build_device_resident_owned,
    seed_homogeneous_host_metadata,
)
from vibespatial.kernels.constructive.segmented_union import segmented_union_all
from vibespatial.overlay.contract import OverlayMicrocellComponents
from vibespatial.overlay.host_fallback import (
    _face_sample_point,
    _point_in_ring,
    _signed_area_and_centroid,
)
from vibespatial.overlay.microcells import OverlayMicrocellLabels
from vibespatial.runtime import ExecutionMode

from ._host_boundary import overlay_device_to_host

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    cp = None


def _coalesce_selected_microcells(
    labels: OverlayMicrocellLabels,
    selected_ids,
    *,
    tolerance: float = 1e-12,
    component_ids=None,
):
    if cp is None:
        raise RuntimeError("CuPy is required for contraction reconstruction")

    selected_ids = cp.asarray(selected_ids, dtype=cp.int64)
    count = int(selected_ids.size)
    if count == 0:
        empty_i64 = cp.empty(0, dtype=cp.int64)
        empty_f64 = cp.empty(0, dtype=cp.float64)
        return {
            "row_indices": empty_i64,
            "x_left": empty_f64,
            "x_right": empty_f64,
            "y_lower_left": empty_f64,
            "y_lower_right": empty_f64,
            "y_upper_left": empty_f64,
            "y_upper_right": empty_f64,
        }

    bands = labels.bands
    row_indices = cp.asarray(bands.row_indices, dtype=cp.int64)[selected_ids]
    interval_indices = cp.asarray(bands.interval_indices, dtype=cp.int64)[selected_ids]
    lower_segment_ids = cp.asarray(bands.lower_segment_ids, dtype=cp.int64)[selected_ids]
    upper_segment_ids = cp.asarray(bands.upper_segment_ids, dtype=cp.int64)[selected_ids]
    x_left = cp.asarray(bands.x_left, dtype=cp.float64)[selected_ids]
    x_right = cp.asarray(bands.x_right, dtype=cp.float64)[selected_ids]
    y_lower_left = cp.asarray(bands.y_lower_left, dtype=cp.float64)[selected_ids]
    y_lower_right = cp.asarray(bands.y_lower_right, dtype=cp.float64)[selected_ids]
    y_upper_left = cp.asarray(bands.y_upper_left, dtype=cp.float64)[selected_ids]
    y_upper_right = cp.asarray(bands.y_upper_right, dtype=cp.float64)[selected_ids]
    component_ids = None if component_ids is None else cp.asarray(component_ids, dtype=cp.int64)

    run_breaks = cp.ones(count, dtype=cp.bool_)
    if count > 1:
        same_row = row_indices[1:] == row_indices[:-1]
        next_interval = interval_indices[1:] == (interval_indices[:-1] + 1)
        same_lower = lower_segment_ids[1:] == lower_segment_ids[:-1]
        same_upper = upper_segment_ids[1:] == upper_segment_ids[:-1]
        connected_x = cp.abs(x_left[1:] - x_right[:-1]) <= tolerance
        connected_lower = cp.abs(y_lower_left[1:] - y_lower_right[:-1]) <= tolerance
        connected_upper = cp.abs(y_upper_left[1:] - y_upper_right[:-1]) <= tolerance
        run_breaks[1:] = ~(
            same_row
            & next_interval
            & same_lower
            & same_upper
            & connected_x
            & connected_lower
            & connected_upper
        )
        if component_ids is not None:
            run_breaks[1:] = run_breaks[1:] | (component_ids[1:] != component_ids[:-1])

    run_starts = cp.flatnonzero(run_breaks).astype(cp.int64, copy=False)
    run_ends = cp.concatenate(
        (run_starts[1:] - 1, cp.asarray([count - 1], dtype=cp.int64))
    )
    result = {
        "row_indices": row_indices[run_starts],
        "x_left": x_left[run_starts],
        "x_right": x_right[run_ends],
        "y_lower_left": y_lower_left[run_starts],
        "y_lower_right": y_lower_right[run_ends],
        "y_upper_left": y_upper_left[run_starts],
        "y_upper_right": y_upper_right[run_ends],
    }
    if component_ids is not None:
        result["component_ids"] = component_ids[run_starts]
    return result


def _select_microcell_mask(labels: OverlayMicrocellLabels, operation: str):
    left_inside = labels.left_inside
    right_inside = labels.right_inside
    match operation:
        case "intersection":
            return left_inside & right_inside
        case "union":
            return left_inside | right_inside
        case "difference":
            return left_inside & ~right_inside
        case "symmetric_difference":
            return left_inside ^ right_inside
        case "identity":
            return left_inside
        case _:
            raise ValueError(f"unsupported contraction reconstruction operation: {operation}")


def _build_microcell_polygon_rows(coalesced) -> OwnedGeometryArray:
    if cp is None:
        raise RuntimeError("CuPy is required for contraction reconstruction")

    x_left = cp.asarray(coalesced["x_left"], dtype=cp.float64)
    count = int(x_left.size)
    if count == 0:
        return build_device_resident_owned(
            device_families={},
            row_count=0,
            tags=cp.empty(0, dtype=cp.int8),
            validity=cp.empty(0, dtype=cp.bool_),
            family_row_offsets=cp.empty(0, dtype=cp.int32),
            execution_mode="gpu",
        )

    x_right = cp.asarray(coalesced["x_right"], dtype=cp.float64)
    y_lower_left = cp.asarray(coalesced["y_lower_left"], dtype=cp.float64)
    y_lower_right = cp.asarray(coalesced["y_lower_right"], dtype=cp.float64)
    y_upper_left = cp.asarray(coalesced["y_upper_left"], dtype=cp.float64)
    y_upper_right = cp.asarray(coalesced["y_upper_right"], dtype=cp.float64)

    x = cp.empty(count * 5, dtype=cp.float64)
    y = cp.empty(count * 5, dtype=cp.float64)
    x[0::5] = x_left
    x[1::5] = x_right
    x[2::5] = x_right
    x[3::5] = x_left
    x[4::5] = x_left
    y[0::5] = y_lower_left
    y[1::5] = y_lower_right
    y[2::5] = y_upper_right
    y[3::5] = y_upper_left
    y[4::5] = y_lower_left

    row_count = count
    d_geom_offsets = cp.arange(row_count + 1, dtype=cp.int32)
    d_ring_offsets = cp.arange(0, (row_count + 1) * 5, 5, dtype=cp.int32)
    d_empty = cp.zeros(row_count, dtype=cp.bool_)
    d_tags = cp.full(row_count, FAMILY_TAGS[GeometryFamily.POLYGON], dtype=cp.int8)
    d_validity = cp.ones(row_count, dtype=cp.bool_)
    d_family_row_offsets = cp.arange(row_count, dtype=cp.int32)
    d_bounds = cp.column_stack(
        (
            cp.minimum(x_left, x_right),
            cp.minimum(cp.minimum(y_lower_left, y_lower_right), cp.minimum(y_upper_left, y_upper_right)),
            cp.maximum(x_left, x_right),
            cp.maximum(cp.maximum(y_lower_left, y_lower_right), cp.maximum(y_upper_left, y_upper_right)),
        )
    )
    result = build_device_resident_owned(
        device_families={
            GeometryFamily.POLYGON: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.POLYGON,
                x=x,
                y=y,
                geometry_offsets=d_geom_offsets,
                empty_mask=d_empty,
                ring_offsets=d_ring_offsets,
                bounds=d_bounds,
            )
        },
        row_count=row_count,
        tags=d_tags,
        validity=d_validity,
        family_row_offsets=d_family_row_offsets,
        execution_mode="gpu",
    )
    seed_homogeneous_host_metadata(result, GeometryFamily.POLYGON)
    return result


def _microcell_group_offsets_host(
    selected_row_indices,
    *,
    row_count: int,
    total_cells: int,
) -> np.ndarray:
    """Build CSR group offsets for sorted selected microcell rows."""
    if row_count <= 0:
        return np.zeros(1, dtype=np.int64)
    if row_count == 1:
        return np.asarray([0, int(total_cells)], dtype=np.int64)
    selected_rows = overlay_device_to_host(
        selected_row_indices,
        reason="overlay contraction selected-row group-offset metadata",
        dtype=np.int64,
    )
    group_counts = np.bincount(selected_rows, minlength=row_count).astype(
        np.int64,
        copy=False,
    )
    return np.concatenate(([0], np.cumsum(group_counts, dtype=np.int64)))


def _walk_boundary_rings(coalesced) -> dict[int, list[list[np.ndarray]]]:
    if cp is None:
        raise RuntimeError("CuPy is required for contraction reconstruction")

    row_indices = cp.asarray(coalesced["row_indices"], dtype=cp.int64)
    x_left = cp.asarray(coalesced["x_left"], dtype=cp.float64)
    x_right = cp.asarray(coalesced["x_right"], dtype=cp.float64)
    y_lower_left = cp.asarray(coalesced["y_lower_left"], dtype=cp.float64)
    y_lower_right = cp.asarray(coalesced["y_lower_right"], dtype=cp.float64)
    y_upper_left = cp.asarray(coalesced["y_upper_left"], dtype=cp.float64)
    y_upper_right = cp.asarray(coalesced["y_upper_right"], dtype=cp.float64)

    cell_count = int(row_indices.size)
    if cell_count == 0:
        return {}

    edge_row = cp.repeat(row_indices, 4)
    sx = cp.concatenate((x_left, x_right, x_right, x_left))
    sy = cp.concatenate((y_lower_left, y_lower_right, y_upper_right, y_upper_left))
    tx = cp.concatenate((x_right, x_right, x_left, x_left))
    ty = cp.concatenate((y_lower_right, y_upper_right, y_upper_left, y_lower_left))

    sx_bits = sx.view(cp.uint64)
    sy_bits = sy.view(cp.uint64)
    tx_bits = tx.view(cp.uint64)
    ty_bits = ty.view(cp.uint64)

    swap = (sx_bits > tx_bits) | ((sx_bits == tx_bits) & (sy_bits > ty_bits))
    ax_bits = cp.where(swap, tx_bits, sx_bits)
    ay_bits = cp.where(swap, ty_bits, sy_bits)
    bx_bits = cp.where(swap, sx_bits, tx_bits)
    by_bits = cp.where(swap, sy_bits, ty_bits)

    order = cp.lexsort(cp.stack((by_bits, bx_bits, ay_bits, ax_bits, edge_row)))
    sorted_row = edge_row[order]
    sorted_sx = sx[order]
    sorted_sy = sy[order]
    sorted_tx = tx[order]
    sorted_ty = ty[order]
    sorted_sx_bits = sx_bits[order]
    sorted_sy_bits = sy_bits[order]
    sorted_tx_bits = tx_bits[order]
    sorted_ty_bits = ty_bits[order]
    sorted_ax_bits = ax_bits[order]
    sorted_ay_bits = ay_bits[order]
    sorted_bx_bits = bx_bits[order]
    sorted_by_bits = by_bits[order]

    run_breaks = cp.ones(int(order.size), dtype=cp.bool_)
    if int(order.size) > 1:
        run_breaks[1:] = (
            (sorted_row[1:] != sorted_row[:-1])
            | (sorted_ax_bits[1:] != sorted_ax_bits[:-1])
            | (sorted_ay_bits[1:] != sorted_ay_bits[:-1])
            | (sorted_bx_bits[1:] != sorted_bx_bits[:-1])
            | (sorted_by_bits[1:] != sorted_by_bits[:-1])
        )
    run_starts = cp.flatnonzero(run_breaks).astype(cp.int64, copy=False)
    run_ends = cp.concatenate(
        (run_starts[1:], cp.asarray([int(order.size)], dtype=cp.int64))
    )
    run_counts = run_ends - run_starts
    keep_runs = (run_counts & 1) == 1
    keep_indices = run_starts[keep_runs]
    if int(keep_indices.size) == 0:
        return {}

    row_h = overlay_device_to_host(
        sorted_row[keep_indices],
        reason="overlay contraction boundary-walk row metadata",
        dtype=np.int64,
    )
    sx_h = overlay_device_to_host(
        sorted_sx[keep_indices],
        reason="overlay contraction boundary-walk source-x metadata",
        dtype=np.float64,
    )
    sy_h = overlay_device_to_host(
        sorted_sy[keep_indices],
        reason="overlay contraction boundary-walk source-y metadata",
        dtype=np.float64,
    )
    tx_h = overlay_device_to_host(
        sorted_tx[keep_indices],
        reason="overlay contraction boundary-walk target-x metadata",
        dtype=np.float64,
    )
    ty_h = overlay_device_to_host(
        sorted_ty[keep_indices],
        reason="overlay contraction boundary-walk target-y metadata",
        dtype=np.float64,
    )
    sx_bits_h = overlay_device_to_host(
        sorted_sx_bits[keep_indices],
        reason="overlay contraction boundary-walk source-x bits metadata",
        dtype=np.uint64,
    )
    sy_bits_h = overlay_device_to_host(
        sorted_sy_bits[keep_indices],
        reason="overlay contraction boundary-walk source-y bits metadata",
        dtype=np.uint64,
    )
    tx_bits_h = overlay_device_to_host(
        sorted_tx_bits[keep_indices],
        reason="overlay contraction boundary-walk target-x bits metadata",
        dtype=np.uint64,
    )
    ty_bits_h = overlay_device_to_host(
        sorted_ty_bits[keep_indices],
        reason="overlay contraction boundary-walk target-y bits metadata",
        dtype=np.uint64,
    )

    start_map: dict[tuple[int, int, int], list[int]] = {}
    for edge_index, (row, sxb, syb) in enumerate(zip(row_h, sx_bits_h, sy_bits_h, strict=False)):
        start_map.setdefault((int(row), int(sxb), int(syb)), []).append(edge_index)

    visited = np.zeros(sx_h.size, dtype=bool)
    cycle_rings: dict[int, np.ndarray] = {}
    cycle_samples: dict[int, tuple[float, float]] = {}
    cycle_areas: dict[int, float] = {}
    cycle_rows: dict[int, int] = {}
    cycle_id = 0

    for edge_index in range(sx_h.size):
        if visited[edge_index]:
            continue
        row = int(row_h[edge_index])
        ring_points: list[tuple[float, float]] = [
            (float(sx_h[edge_index]), float(sy_h[edge_index]))
        ]
        current = edge_index
        while not visited[current]:
            visited[current] = True
            ring_points.append((float(tx_h[current]), float(ty_h[current])))
            next_key = (row, int(tx_bits_h[current]), int(ty_bits_h[current]))
            next_edge = None
            for candidate in start_map.get(next_key, []):
                if not visited[candidate]:
                    next_edge = candidate
                    break
            if next_edge is None:
                break
            current = next_edge

        ring = np.asarray(ring_points, dtype=np.float64)
        if ring.shape[0] < 4:
            continue
        if not np.allclose(ring[0], ring[-1]):
            continue
        area, _, _ = _signed_area_and_centroid(ring[:-1])
        if abs(area) <= 1e-12:
            continue
        cycle_rings[cycle_id] = ring
        cycle_samples[cycle_id] = _face_sample_point(ring[:-1])
        cycle_areas[cycle_id] = abs(float(area))
        cycle_rows[cycle_id] = row
        cycle_id += 1

    final_rows: dict[int, list[list[np.ndarray]]] = {}
    for row in sorted(set(cycle_rows.values())):
        row_cycles = [ci for ci, r in cycle_rows.items() if r == row]
        containment_depth = {ci: 0 for ci in row_cycles}
        for cycle_index in row_cycles:
            ca = cycle_areas[cycle_index]
            sample_x, sample_y = cycle_samples[cycle_index]
            for container_index in row_cycles:
                if container_index == cycle_index or cycle_areas[container_index] <= ca:
                    continue
                if _point_in_ring(sample_x, sample_y, cycle_rings[container_index]):
                    containment_depth[cycle_index] += 1
        exterior_indices = [
            ci for ci in row_cycles if containment_depth[ci] % 2 == 0
        ]
        exterior_indices.sort(key=lambda ci: (cycle_areas[ci], ci))
        hole_map: dict[int, list[int]] = {ci: [] for ci in exterior_indices}
        candidate_holes: dict[int, int] = {}
        exterior_set = set(exterior_indices)
        for cycle_index in row_cycles:
            if cycle_index in exterior_set:
                continue
            ca = cycle_areas[cycle_index]
            sample_x, sample_y = cycle_samples[cycle_index]
            for exterior_index in exterior_indices:
                if cycle_areas[exterior_index] <= ca:
                    continue
                if _point_in_ring(sample_x, sample_y, cycle_rings[exterior_index]):
                    candidate_holes[cycle_index] = exterior_index
                    break
        holes_by_container: dict[int, list[int]] = {}
        for cycle_index, container in candidate_holes.items():
            holes_by_container.setdefault(container, []).append(cycle_index)
        for cycle_index, container in candidate_holes.items():
            sample_x, sample_y = cycle_samples[cycle_index]
            local_depth = 0
            for other_index in holes_by_container.get(container, []):
                if other_index == cycle_index or cycle_areas[other_index] <= cycle_areas[cycle_index]:
                    continue
                if _point_in_ring(sample_x, sample_y, cycle_rings[other_index]):
                    local_depth += 1
            if local_depth % 2 == 0:
                hole_map[container].append(cycle_index)

        polygons: list[list[np.ndarray]] = []
        for exterior_index in exterior_indices:
            exterior_ring = cycle_rings[exterior_index]
            exterior_area, _, _ = _signed_area_and_centroid(exterior_ring[:-1])
            if exterior_area < 0.0:
                exterior_ring = exterior_ring[::-1].copy()
            rings = [exterior_ring]
            for hole_index in sorted(hole_map[exterior_index]):
                hole_ring = cycle_rings[hole_index]
                hole_area, _, _ = _signed_area_and_centroid(hole_ring[:-1])
                if hole_area > 0.0:
                    hole_ring = hole_ring[::-1].copy()
                rings.append(hole_ring)
            polygons.append(rings)
        final_rows[row] = polygons
    return final_rows


def reconstruct_overlay_from_microcells(
    labels: OverlayMicrocellLabels,
    operation: str,
    *,
    components: OverlayMicrocellComponents | None = None,
    row_count: int | None = None,
    dispatch_mode: ExecutionMode | str = ExecutionMode.GPU,
) -> OwnedGeometryArray:
    if cp is None:
        raise RuntimeError("CuPy is required for contraction reconstruction")

    band_row_count = labels.bands.row_count if row_count is None else int(row_count)
    mask = _select_microcell_mask(labels, operation)
    selected_ids = cp.flatnonzero(mask.astype(cp.bool_, copy=False)).astype(cp.int64, copy=False)
    comp_ids = None
    if int(selected_ids.size) > 0:
        row_ids = cp.asarray(labels.bands.row_indices, dtype=cp.int64)[selected_ids]
        interval_ids = cp.asarray(labels.bands.interval_indices, dtype=cp.int64)[selected_ids]
        if components is not None:
            comp_ids = cp.asarray(components.component_ids, dtype=cp.int64)[selected_ids]
            order = cp.lexsort(cp.stack((interval_ids, comp_ids, row_ids)))
            comp_ids = comp_ids[order]
        else:
            order = cp.lexsort(cp.stack((interval_ids, row_ids)))
        selected_ids = selected_ids[order]
    coalesced = _coalesce_selected_microcells(labels, selected_ids, component_ids=comp_ids)
    cell_rows = _build_microcell_polygon_rows(coalesced)
    if band_row_count == 0:
        return cell_rows

    selected_row_indices = coalesced["row_indices"]
    if int(selected_row_indices.size) == 0:
        group_offsets = np.zeros(band_row_count + 1, dtype=np.int64)
        return segmented_union_all(
            cell_rows,
            group_offsets,
            dispatch_mode=dispatch_mode,
        )

    group_offsets = _microcell_group_offsets_host(
        selected_row_indices,
        row_count=band_row_count,
        total_cells=int(cell_rows.row_count),
    )
    return segmented_union_all(
        cell_rows,
        group_offsets,
        dispatch_mode=dispatch_mode,
    )
