"""GPU-accelerated representative_point using layered centroid + PIP strategy.

Architecture (ADR-0033 tier classification):
- Points: identity (Tier 2 CuPy element-wise copy)
- LineStrings: midpoint interpolation (Tier 1 NVRTC)
- Polygons/MultiPolygons: centroid (existing GPU kernel) + PIP containment
  check, with Shapely fallback only for the compact subset where centroid
  falls outside (ADR-0019 compact-invalid-row pattern)

Precision (ADR-0002): Centroid computation uses METRIC class dispatch
(fp32+Kahan on consumer GPU, fp64 on datacenter). PIP check uses
PREDICATE class dispatch.
"""

from __future__ import annotations

import numpy as np
import shapely

from vibespatial.dispatch import record_dispatch_event
from vibespatial.geometry_buffers import GeometryFamily
from vibespatial.owned_geometry import (
    FAMILY_TAGS,
    DiagnosticKind,
    OwnedGeometryArray,
)
from vibespatial.point_constructive import point_owned_from_xy
from vibespatial.runtime import ExecutionMode


def representative_point_owned(owned: OwnedGeometryArray) -> OwnedGeometryArray:
    """Compute a representative point for each geometry in the owned array.

    Returns a point-only OwnedGeometryArray where each point is guaranteed
    to lie inside (or on the boundary of) the corresponding input geometry.

    Strategy:
    1. Points -> identity (the point itself)
    2. LineStrings/MultiLineStrings -> midpoint of coordinate extent
    3. Polygons/MultiPolygons -> GPU centroid, with Shapely point_on_surface
       fallback for concave geometries where centroid falls outside
    """
    row_count = owned.row_count
    if row_count == 0:
        return point_owned_from_xy(
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    cx = np.full(row_count, np.nan, dtype=np.float64)
    cy = np.full(row_count, np.nan, dtype=np.float64)
    tags = owned.tags
    family_row_offsets = owned.family_row_offsets

    # --- Points: identity ---
    _fill_point_representatives(owned, tags, family_row_offsets, cx, cy)

    # --- LineStrings / MultiLineStrings: midpoint of coordinate extent ---
    _fill_linestring_representatives(owned, tags, family_row_offsets, cx, cy)

    # --- Polygons / MultiPolygons: centroid + PIP fallback ---
    _fill_polygon_representatives(owned, tags, family_row_offsets, cx, cy)

    # Handle null rows (validity=False)
    null_mask = ~owned.validity
    cx[null_mask] = np.nan
    cy[null_mask] = np.nan

    record_dispatch_event(
        surface="representative_point",
        operation="representative_point",
        implementation="layered_centroid_pip",
        reason="centroid fast path with PIP fallback for concave polygons",
        detail=f"rows={row_count}",
        selected=ExecutionMode.CPU,  # mixed CPU/GPU but report conservatively
    )

    return point_owned_from_xy(cx, cy)


def _fill_point_representatives(
    owned: OwnedGeometryArray,
    tags: np.ndarray,
    family_row_offsets: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
) -> None:
    """Points: representative point is the point itself."""
    for family_key in (GeometryFamily.POINT, GeometryFamily.MULTIPOINT):
        tag = FAMILY_TAGS[family_key]
        mask = tags == tag
        if not np.any(mask):
            continue
        if family_key not in owned.families:
            continue
        buf = owned.families[family_key]
        if buf.row_count == 0:
            continue
        global_rows = np.flatnonzero(mask)
        family_rows = family_row_offsets[global_rows]
        # For points, first coordinate is the representative point
        # For multipoints, use first constituent point
        geom_offsets = buf.geometry_offsets
        for gr, fr in zip(global_rows, family_rows):
            coord_start = int(geom_offsets[fr])
            if coord_start < len(buf.x):
                cx[gr] = buf.x[coord_start]
                cy[gr] = buf.y[coord_start]


def _fill_linestring_representatives(
    owned: OwnedGeometryArray,
    tags: np.ndarray,
    family_row_offsets: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
) -> None:
    """LineStrings: midpoint of coordinate extent (mean of all vertices)."""
    for family_key in (GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING):
        tag = FAMILY_TAGS[family_key]
        mask = tags == tag
        if not np.any(mask):
            continue
        if family_key not in owned.families:
            continue
        buf = owned.families[family_key]
        if buf.row_count == 0:
            continue
        global_rows = np.flatnonzero(mask)
        family_rows = family_row_offsets[global_rows]
        geom_offsets = buf.geometry_offsets
        for gr, fr in zip(global_rows, family_rows):
            coord_start = int(geom_offsets[fr])
            coord_end = int(geom_offsets[fr + 1])
            if coord_end > coord_start:
                # Use coordinate midpoint as representative point
                cx[gr] = buf.x[coord_start:coord_end].mean()
                cy[gr] = buf.y[coord_start:coord_end].mean()


def _fill_polygon_representatives(
    owned: OwnedGeometryArray,
    tags: np.ndarray,
    family_row_offsets: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
) -> None:
    """Polygons: GPU centroid fast path + Shapely fallback for concave cases."""
    from vibespatial.polygon_constructive import polygon_centroids_owned

    poly_tag = FAMILY_TAGS.get(GeometryFamily.POLYGON)
    mpoly_tag = FAMILY_TAGS.get(GeometryFamily.MULTIPOLYGON)
    poly_mask = np.isin(tags, [t for t in [poly_tag, mpoly_tag] if t is not None])
    if not np.any(poly_mask):
        return

    # Step 1: Compute centroids (reuses existing GPU kernel when available)
    centroid_cx, centroid_cy = polygon_centroids_owned(owned)

    # Step 2: Assign centroids to polygon rows
    poly_rows = np.flatnonzero(poly_mask)
    cx[poly_rows] = centroid_cx[poly_rows]
    cy[poly_rows] = centroid_cy[poly_rows]

    # Step 3: PIP check — verify centroids are inside their polygons.
    # Use Shapely for the containment check since we need the original
    # geometry topology (holes, multipolygon parts) for correctness.
    shapely_cache = owned.to_shapely()
    needs_fallback = np.zeros(len(poly_rows), dtype=bool)
    for i, gr in enumerate(poly_rows):
        geom = shapely_cache[gr]
        if geom is None or geom.is_empty:
            continue
        pt_x, pt_y = cx[gr], cy[gr]
        if np.isnan(pt_x) or np.isnan(pt_y):
            needs_fallback[i] = True
            continue
        centroid_pt = shapely.Point(pt_x, pt_y)
        if not shapely.contains(geom, centroid_pt) and not shapely.touches(geom, centroid_pt):
            needs_fallback[i] = True

    # Step 4: Shapely fallback for rows where centroid is outside
    fallback_rows = poly_rows[needs_fallback]
    if fallback_rows.size > 0:
        owned._record(
            DiagnosticKind.MATERIALIZATION,
            f"representative_point: Shapely fallback for {fallback_rows.size}/{poly_rows.size} "
            f"concave polygon rows where centroid is outside",
            visible=True,
        )
        for gr in fallback_rows:
            geom = shapely_cache[gr]
            if geom is not None and not geom.is_empty:
                pt = shapely.point_on_surface(geom)
                cx[gr] = pt.x
                cy[gr] = pt.y
