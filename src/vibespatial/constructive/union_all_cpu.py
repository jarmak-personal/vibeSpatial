"""CPU-only helpers for union-all style constructive reductions."""

from __future__ import annotations

import numpy as np
import shapely

from vibespatial.geometry.owned import OwnedGeometryArray, from_shapely_geometries

_EMPTY_POLYGON_SENTINEL = None
_EMPTY_OWNED_SENTINEL = None


def empty_polygon():
    """Lazily create a Shapely empty polygon."""
    global _EMPTY_POLYGON_SENTINEL
    if _EMPTY_POLYGON_SENTINEL is None:
        _EMPTY_POLYGON_SENTINEL = shapely.Polygon()
    return _EMPTY_POLYGON_SENTINEL


def empty_owned() -> OwnedGeometryArray:
    """Lazily create a 1-row OGA containing an empty polygon."""
    global _EMPTY_OWNED_SENTINEL
    if _EMPTY_OWNED_SENTINEL is None:
        _EMPTY_OWNED_SENTINEL = from_shapely_geometries([empty_polygon()])
    return _EMPTY_OWNED_SENTINEL


def merge_pair_cpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    op: str,
) -> OwnedGeometryArray:
    """CPU fallback for a single pairwise union/intersection tree-reduce step."""
    try:
        left_geom = left.to_shapely()[0]
    except Exception:
        left_geom = empty_polygon()
    try:
        right_geom = right.to_shapely()[0]
    except Exception:
        right_geom = empty_polygon()

    try:
        if op == "union":
            merged = shapely.union(left_geom, right_geom)
        else:
            merged = shapely.intersection(left_geom, right_geom)
        if merged is not None and not shapely.is_valid(merged):
            merged = shapely.make_valid(merged)
    except Exception:
        merged = empty_polygon()

    return from_shapely_geometries([merged if merged is not None else empty_polygon()])


def reduce_all_cpu(
    owned: OwnedGeometryArray,
    *,
    op: str,
    grid_size: float | None = None,
) -> OwnedGeometryArray:
    """CPU fallback for global union/intersection-style reductions."""
    geoms = owned.to_shapely()
    arr = np.empty(len(geoms), dtype=object)
    arr[:] = geoms

    if op == "union_all":
        result = shapely.union_all(arr, grid_size=grid_size)
    elif op == "coverage_union_all":
        result = shapely.coverage_union_all(arr)
    elif op == "intersection_all":
        result = shapely.intersection_all(arr)
    else:
        raise ValueError(f"Unsupported reduction op: {op}")

    if result is None:
        result = empty_polygon()
    return from_shapely_geometries([result])
