from __future__ import annotations

import numpy as np

from vibespatial.constructive.nonpolygon_binary_output import (
    build_device_backed_multipoint_output,
)
from vibespatial.cuda._runtime import get_cuda_runtime
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import OwnedGeometryArray, from_shapely_geometries
from vibespatial.runtime.residency import Residency, TransferTrigger


def _empty_multipoint_output(row_count: int) -> OwnedGeometryArray:
    import cupy as cp

    return build_device_backed_multipoint_output(
        cp.empty(0, dtype=cp.float64),
        cp.empty(0, dtype=cp.float64),
        row_count=row_count,
        validity=cp.zeros(row_count, dtype=cp.bool_),
        geometry_offsets=cp.zeros(row_count + 1, dtype=cp.int32),
    )


def multipoint_polygon_intersection(
    multipoints: OwnedGeometryArray,
    polygons: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """MultiPoint-Polygon intersection: keep points inside polygon."""
    import cupy as cp
    from shapely.geometry import MultiPoint as ShapelyMultiPoint
    from shapely.geometry import Point as ShapelyPoint

    n = multipoints.row_count
    multipoints.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="multipoint_polygon_intersection GPU",
    )
    polygons.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="multipoint_polygon_intersection GPU",
    )

    mp_state = multipoints.device_state
    poly_state = polygons.device_state

    mp_buf = mp_state.families[GeometryFamily.MULTIPOINT]
    if GeometryFamily.POLYGON in poly_state.families:
        poly_buf = poly_state.families[GeometryFamily.POLYGON]
    elif GeometryFamily.MULTIPOLYGON in poly_state.families:
        poly_buf = poly_state.families[GeometryFamily.MULTIPOLYGON]
    else:
        return _empty_multipoint_output(n)

    d_mp_valid = mp_state.validity.astype(cp.bool_) & ~mp_buf.empty_mask.astype(cp.bool_)
    d_poly_valid = poly_state.validity.astype(cp.bool_) & ~poly_buf.empty_mask.astype(cp.bool_)
    d_both_valid = d_mp_valid & d_poly_valid

    runtime = get_cuda_runtime()
    d_mp_offsets = cp.asarray(mp_buf.geometry_offsets)
    total_points = int(d_mp_offsets[-1])

    if total_points == 0:
        return _empty_multipoint_output(n)

    h_mp_offsets = runtime.copy_device_to_host(d_mp_offsets)
    h_both_valid = runtime.copy_device_to_host(d_both_valid)

    h_mp_x = mp_buf.x if mp_buf.host_materialized else runtime.copy_device_to_host(mp_buf.x)
    h_mp_y = mp_buf.y if mp_buf.host_materialized else runtime.copy_device_to_host(mp_buf.y)
    poly_shapely = polygons.to_shapely()

    point_geoms = []
    poly_geoms = []
    for i in range(n):
        start = h_mp_offsets[i]
        end = h_mp_offsets[i + 1]
        for j in range(start, end):
            point_geoms.append(ShapelyPoint(float(h_mp_x[j]), float(h_mp_y[j])))
            poly_geoms.append(poly_shapely[i])

    if len(point_geoms) == 0:
        return _empty_multipoint_output(n)

    from vibespatial.kernels.predicates.point_in_polygon import point_in_polygon

    pt_oga = from_shapely_geometries(point_geoms)
    poly_oga = from_shapely_geometries(poly_geoms)
    pip_mask = point_in_polygon(pt_oga, poly_oga, _return_device=True)
    if hasattr(pip_mask, "__cuda_array_interface__"):
        h_pip = runtime.copy_device_to_host(pip_mask)
    else:
        h_pip = np.asarray(pip_mask, dtype=bool)

    result_geoms = []
    for i in range(n):
        if not h_both_valid[i]:
            result_geoms.append(None)
            continue
        start = h_mp_offsets[i]
        end = h_mp_offsets[i + 1]
        kept = [point_geoms[j] for j in range(start, end) if h_pip[j]]
        if len(kept) == 0:
            result_geoms.append(None)
        elif len(kept) == 1:
            result_geoms.append(kept[0])
        else:
            result_geoms.append(ShapelyMultiPoint(kept))

    return from_shapely_geometries(result_geoms)


def multipoint_polygon_difference(
    multipoints: OwnedGeometryArray,
    polygons: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """MultiPoint-Polygon difference: keep points outside polygon."""
    import cupy as cp
    from shapely.geometry import MultiPoint as ShapelyMultiPoint
    from shapely.geometry import Point as ShapelyPoint

    n = multipoints.row_count
    multipoints.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="multipoint_polygon_difference GPU",
    )
    polygons.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="multipoint_polygon_difference GPU",
    )

    mp_state = multipoints.device_state
    poly_state = polygons.device_state
    mp_buf = mp_state.families[GeometryFamily.MULTIPOINT]

    d_mp_valid = mp_state.validity.astype(cp.bool_) & ~mp_buf.empty_mask.astype(cp.bool_)
    d_poly_valid = poly_state.validity.astype(cp.bool_)
    d_both_valid = d_mp_valid & d_poly_valid

    runtime = get_cuda_runtime()
    d_mp_offsets = cp.asarray(mp_buf.geometry_offsets)
    total_points = int(d_mp_offsets[-1])

    if total_points == 0:
        return _empty_multipoint_output(n)

    h_mp_offsets = runtime.copy_device_to_host(d_mp_offsets)
    h_mp_valid = runtime.copy_device_to_host(d_mp_valid)
    h_both_valid = runtime.copy_device_to_host(d_both_valid)

    h_mp_x = mp_buf.x if mp_buf.host_materialized else runtime.copy_device_to_host(mp_buf.x)
    h_mp_y = mp_buf.y if mp_buf.host_materialized else runtime.copy_device_to_host(mp_buf.y)
    poly_shapely = polygons.to_shapely()

    point_geoms = []
    poly_geoms = []
    for i in range(n):
        start = h_mp_offsets[i]
        end = h_mp_offsets[i + 1]
        for j in range(start, end):
            point_geoms.append(ShapelyPoint(float(h_mp_x[j]), float(h_mp_y[j])))
            poly_geoms.append(poly_shapely[i])

    if len(point_geoms) == 0:
        return _empty_multipoint_output(n)

    from vibespatial.kernels.predicates.point_in_polygon import point_in_polygon

    pt_oga = from_shapely_geometries(point_geoms)
    poly_oga = from_shapely_geometries(poly_geoms)
    pip_mask = point_in_polygon(pt_oga, poly_oga, _return_device=True)
    if hasattr(pip_mask, "__cuda_array_interface__"):
        h_pip = runtime.copy_device_to_host(pip_mask)
    else:
        h_pip = np.asarray(pip_mask, dtype=bool)

    result_geoms = []
    for i in range(n):
        if not h_mp_valid[i]:
            result_geoms.append(None)
            continue
        start = h_mp_offsets[i]
        end = h_mp_offsets[i + 1]
        if h_both_valid[i]:
            kept = [point_geoms[j] for j in range(start, end) if not h_pip[j]]
        else:
            kept = [point_geoms[j] for j in range(start, end)]
        if len(kept) == 0:
            result_geoms.append(None)
        elif len(kept) == 1:
            result_geoms.append(kept[0])
        else:
            result_geoms.append(ShapelyMultiPoint(kept))

    return from_shapely_geometries(result_geoms)
