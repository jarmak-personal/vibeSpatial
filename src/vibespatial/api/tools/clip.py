"""Module to clip vector data using GeoPandas."""

import warnings

import numpy as np
import pandas as pd
import pandas.api.types
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box

from vibespatial.api import GeoDataFrame, GeoSeries
from vibespatial.api._compat import PANDAS_GE_30
from vibespatial.api.geometry_array import (
    LINE_GEOM_TYPES,
    POINT_GEOM_TYPES,
    POLYGON_GEOM_TYPES,
    _check_crs,
    _crs_mismatch_warn,
)
from vibespatial.api.geometry_array import (
    from_shapely as _geometryarray_from_shapely,
)


def _mask_is_list_like_rectangle(mask):
    """
    Check if the input mask is list-like and not an instance of
    specific geometric types.

    Parameters
    ----------
    mask : GeoDataFrame, GeoSeries, (Multi)Polygon, list-like
        Polygon vector layer used to clip ``gdf``.

    Returns
    -------
    bool
        True if `mask` is list-like and not an instance of `GeoDataFrame`,
        `GeoSeries`, `Polygon`, or `MultiPolygon`, otherwise False.
    """
    return pandas.api.types.is_list_like(mask) and not isinstance(
        mask, GeoDataFrame | GeoSeries | Polygon | MultiPolygon
    )


def _rectangle_bounds_from_mask(mask):
    """Return rectangle bounds for axis-aligned rectangle masks, else None."""
    if _mask_is_list_like_rectangle(mask):
        return tuple(float(v) for v in mask)
    if isinstance(mask, MultiPolygon):
        return None
    if not isinstance(mask, Polygon) or mask.is_empty or len(mask.interiors) != 0:
        return None
    coords = np.asarray(mask.exterior.coords)
    if len(coords) != 5:
        return None
    xs = np.unique(coords[:4, 0])
    ys = np.unique(coords[:4, 1])
    if len(xs) != 2 or len(ys) != 2:
        return None
    return (float(xs[0]), float(ys[0]), float(xs[1]), float(ys[1]))


def _clip_polygon_partition_with_rectangle_mask(
    partition,
    rectangle_bounds: tuple[float, float, float, float],
):
    """Clip polygon rows to a rectangle mask while preserving sliver leftovers.

    The owned rectangle clip path returns polygonal area correctly, but
    `OwnedGeometryArray` still cannot represent per-row GeometryCollections.
    For rectangle masks we recover lower-dimensional leftovers at the
    GeoPandas/Shapely boundary, then assemble any mixed polygon+line result
    there as well.
    """
    xmin, ymin, xmax, ymax = rectangle_bounds
    rectangle_mask = box(xmin, ymin, xmax, ymax)
    source_values = partition.geometry.values if isinstance(partition, GeoDataFrame) else partition.values
    source_geoms = np.asarray(source_values, dtype=object)
    area_values = source_values.clip_by_rect(xmin, ymin, xmax, ymax)
    area_geoms = np.asarray(area_values, dtype=object)

    assembled = np.empty(len(partition), dtype=object)
    for row_index in range(len(partition)):
        source_geom = source_geoms[row_index]
        if source_geom is None or source_geom.is_empty:
            assembled[row_index] = None
            continue
        area_geom = area_geoms[row_index]
        if area_geom is not None and area_geom.is_empty:
            area_geom = None

        edge_geom = source_geom.boundary.intersection(rectangle_mask)
        if area_geom is not None and edge_geom is not None:
            edge_geom = edge_geom.difference(area_geom.boundary)
        if edge_geom is not None and edge_geom.is_empty:
            edge_geom = None

        if area_geom is None and edge_geom is None:
            assembled[row_index] = None
        elif area_geom is None:
            assembled[row_index] = edge_geom
        elif edge_geom is None:
            assembled[row_index] = area_geom
        else:
            assembled[row_index] = GeometryCollection([area_geom, edge_geom])

    clipped_partition = partition.copy(deep=not PANDAS_GE_30)
    if isinstance(clipped_partition, GeoDataFrame):
        clipped_partition[clipped_partition._geometry_column_name] = assembled
    else:
        clipped_partition[:] = assembled
    return clipped_partition


def _clip_gdf_with_mask(gdf, mask, sort=False):
    """
    Clip geometry to the polygon/rectangle extent.

    Clip an input GeoDataFrame to the polygon extent of the polygon
    parameter.

    Parameters
    ----------
    gdf : GeoDataFrame, GeoSeries
        Dataframe to clip.

    mask : (Multi)Polygon, list-like
        Reference polygon/rectangle for clipping.

    sort : boolean, default False
        If True, the results will be sorted in ascending order using the
        geometries' indexes as the primary key.

    Returns
    -------
    GeoDataFrame
        The returned GeoDataFrame is a clipped subset of gdf
        that intersects with polygon/rectangle.
    """
    clipping_by_rectangle = _mask_is_list_like_rectangle(mask)
    rectangle_bounds = _rectangle_bounds_from_mask(mask)
    if clipping_by_rectangle:
        intersection_polygon = box(*mask)
    else:
        intersection_polygon = mask

    candidate_rows = np.asarray(
        gdf.sindex.query(intersection_polygon, predicate="intersects", sort=sort),
        dtype=np.int32,
    )
    gdf_sub = gdf.iloc[candidate_rows]

    # For performance reasons Points don't need to be intersected with the
    # mask at all. For the remaining rows, keep line-like and polygon-like
    # subsets homogeneous so public clip(mask) does not depend on mixed-family
    # constructive dispatch for simple viewport-style workloads.
    point_mask = gdf_sub.geom_type == "Point"
    non_point_mask = ~point_mask
    line_mask = gdf_sub.geom_type.isin(LINE_GEOM_TYPES)
    multiline_mask = gdf_sub.geom_type == "MultiLineString"
    simple_line_mask = line_mask & ~multiline_mask
    polygon_mask = gdf_sub.geom_type.isin(POLYGON_GEOM_TYPES)
    generic_mask = non_point_mask & ~(simple_line_mask | multiline_mask | polygon_mask)

    if not non_point_mask.any():
        # only points, directly return
        return gdf_sub

    def _clip_partition(partition, *, use_rect_fast_path=False):
        if (
            not clipping_by_rectangle
            and rectangle_bounds is not None
            and partition.geom_type.isin(POLYGON_GEOM_TYPES).all()
        ):
            return _clip_polygon_partition_with_rectangle_mask(partition, rectangle_bounds)

        if isinstance(partition, GeoDataFrame):
            clipped_partition = partition.copy(deep=not PANDAS_GE_30)
            geom_name = clipped_partition._geometry_column_name
            geom_values = (
                partition.geometry.values.clip_by_rect(*rectangle_bounds)
                if use_rect_fast_path
                else partition.geometry.values.intersection(mask)
            )
            if not clipping_by_rectangle and partition.geom_type.isin(POLYGON_GEOM_TYPES).all():
                geom_values = geom_values.remove_repeated_points(0.0).normalize()
            clipped_partition[geom_name] = geom_values
            return clipped_partition

        clipped_partition = partition.copy(deep=not PANDAS_GE_30)
        geom_values = (
            partition.values.clip_by_rect(*rectangle_bounds)
            if use_rect_fast_path
            else partition.values.intersection(mask)
        )
        if not clipping_by_rectangle and partition.geom_type.isin(POLYGON_GEOM_TYPES).all():
            geom_values = geom_values.remove_repeated_points(0.0).normalize()
        clipped_partition[:] = geom_values
        return clipped_partition

    parts = []
    if point_mask.any():
        parts.append(gdf_sub[point_mask].copy(deep=not PANDAS_GE_30))
    if simple_line_mask.any():
        parts.append(
            _clip_partition(
                gdf_sub[simple_line_mask],
                use_rect_fast_path=clipping_by_rectangle,
            )
        )
    if multiline_mask.any():
        parts.append(
            _clip_partition(
                gdf_sub[multiline_mask],
                use_rect_fast_path=(
                    clipping_by_rectangle
                    or rectangle_bounds is not None
                ),
            )
        )
    if polygon_mask.any():
        parts.append(
            _clip_partition(
                gdf_sub[polygon_mask],
                use_rect_fast_path=clipping_by_rectangle,
            )
        )
    if generic_mask.any():
        parts.append(
            _clip_partition(
                gdf_sub[generic_mask],
                use_rect_fast_path=clipping_by_rectangle,
            )
        )

    clipped = pd.concat(parts).loc[gdf_sub.index]
    if isinstance(clipped, GeoDataFrame):
        geom_name = clipped._geometry_column_name
        clipped = clipped.copy(deep=not PANDAS_GE_30)
        clipped[geom_name] = GeoSeries(
            _geometryarray_from_shapely(np.asarray(clipped[geom_name], dtype=object), crs=gdf.crs),
            index=clipped.index,
            crs=gdf.crs,
        )
    else:
        clipped = GeoSeries(
            _geometryarray_from_shapely(np.asarray(clipped, dtype=object), crs=gdf.crs),
            index=clipped.index,
            crs=gdf.crs,
        )

    if clipping_by_rectangle:
        # clip_by_rect might return empty geometry collections in edge cases
        clipped = clipped[~clipped.is_empty]
    else:
        # GPU intersection may produce degenerate zero-area polygon slivers at
        # exact clip boundaries where GEOS/Shapely returns lower-dimensional
        # results (LineStrings/Points).  Remove empty geometries and zero-area
        # polygon slivers so clip output matches stock geopandas behaviour.
        keep = ~clipped.is_empty
        if non_point_mask.any():
            poly_rows = clipped.geom_type.isin(POLYGON_GEOM_TYPES)
            if poly_rows.any():
                keep = keep & ~(poly_rows & (clipped.geometry.area <= 0))
        clipped = clipped[keep]
    return clipped


def clip(gdf, mask, keep_geom_type=False, sort=False):
    """Clip points, lines, or polygon geometries to the mask extent.

    Both layers must be in the same Coordinate Reference System (CRS).
    The ``gdf`` will be clipped to the full extent of the clip object.

    If there are multiple polygons in mask, data from ``gdf`` will be
    clipped to the total boundary of all polygons in mask.

    If the ``mask`` is list-like with four elements ``(minx, miny, maxx, maxy)``, a
    faster rectangle clipping algorithm will be used. Note that this can lead to
    slightly different results in edge cases, e.g. if a line would be reduced to a
    point, this point might not be returned.
    The geometry is clipped in a fast but possibly dirty way. The output is not
    guaranteed to be valid. No exceptions will be raised for topological errors.

    Parameters
    ----------
    gdf : GeoDataFrame or GeoSeries
        Vector layer (point, line, polygon) to be clipped to mask.
    mask : GeoDataFrame, GeoSeries, (Multi)Polygon, list-like
        Polygon vector layer used to clip ``gdf``.
        The mask's geometry is dissolved into one geometric feature
        and intersected with ``gdf``.
        If the mask is list-like with four elements ``(minx, miny, maxx, maxy)``,
        ``clip`` will use a faster rectangle clipping (:meth:`~GeoSeries.clip_by_rect`),
        possibly leading to slightly different results.
    keep_geom_type : boolean, default False
        If True, return only geometries of original type in case of intersection
        resulting in multiple geometry types or GeometryCollections.
        If False, return all resulting geometries (potentially mixed-types).
    sort : boolean, default False
        If True, the results will be sorted in ascending order using the
        geometries' indexes as the primary key.

    Returns
    -------
    GeoDataFrame or GeoSeries
         Vector data (points, lines, polygons) from ``gdf`` clipped to
         polygon boundary from mask.

    See Also
    --------
    GeoDataFrame.clip : equivalent GeoDataFrame method
    GeoSeries.clip : equivalent GeoSeries method

    Examples
    --------
    Clip points (grocery stores) with polygons (the Near West Side community):

    >>> import geodatasets
    >>> chicago = geopandas.read_file(
    ...     geodatasets.get_path("geoda.chicago_health")
    ... )
    >>> near_west_side = chicago[chicago["community"] == "NEAR WEST SIDE"]
    >>> groceries = geopandas.read_file(
    ...     geodatasets.get_path("geoda.groceries")
    ... ).to_crs(chicago.crs)
    >>> groceries.shape
    (148, 8)

    >>> nws_groceries = geopandas.clip(groceries, near_west_side)
    >>> nws_groceries.shape
    (7, 8)
    """
    if not isinstance(gdf, GeoDataFrame | GeoSeries):
        raise TypeError(f"'gdf' should be GeoDataFrame or GeoSeries, got {type(gdf)}")

    clipping_by_rectangle = _mask_is_list_like_rectangle(mask)
    if (
        not isinstance(mask, GeoDataFrame | GeoSeries | Polygon | MultiPolygon)
        and not clipping_by_rectangle
    ):
        raise TypeError(
            "'mask' should be GeoDataFrame, GeoSeries,"
            f"(Multi)Polygon or list-like, got {type(mask)}"
        )

    if clipping_by_rectangle and len(mask) != 4:
        raise TypeError(
            "If 'mask' is list-like, it must have four values (minx, miny, maxx, maxy)"
        )

    if isinstance(mask, GeoDataFrame | GeoSeries):
        if not _check_crs(gdf, mask):
            _crs_mismatch_warn(gdf, mask, stacklevel=3)

    if isinstance(mask, GeoDataFrame | GeoSeries):
        box_mask = mask.total_bounds
    elif clipping_by_rectangle:
        box_mask = mask
    else:
        # Avoid empty tuple returned by .bounds when geometry is empty. A tuple of
        # all nan values is consistent with the behavior of
        # {GeoSeries, GeoDataFrame}.total_bounds for empty geometries.
        # TODO(shapely) can simpely use mask.bounds once relying on Shapely 2.0
        box_mask = mask.bounds if not mask.is_empty else (np.nan,) * 4
    box_gdf = gdf.total_bounds
    if not (
        ((box_mask[0] <= box_gdf[2]) and (box_gdf[0] <= box_mask[2]))
        and ((box_mask[1] <= box_gdf[3]) and (box_gdf[1] <= box_mask[3]))
    ):
        return gdf.iloc[:0]

    if isinstance(mask, GeoDataFrame | GeoSeries):
        combined_mask = mask.geometry.union_all()
    else:
        combined_mask = mask

    clipped = _clip_gdf_with_mask(gdf, combined_mask, sort=sort)

    if keep_geom_type:
        geomcoll_concat = (clipped.geom_type == "GeometryCollection").any()
        geomcoll_orig = (gdf.geom_type == "GeometryCollection").any()

        new_collection = geomcoll_concat and not geomcoll_orig

        if geomcoll_orig:
            warnings.warn(
                "keep_geom_type can not be called on a "
                "GeoDataFrame with GeometryCollection.",
                stacklevel=2,
            )
        else:
            # Check that the gdf for multiple geom types (points, lines and/or polys)
            orig_types_total = sum(
                [
                    gdf.geom_type.isin(POLYGON_GEOM_TYPES).any(),
                    gdf.geom_type.isin(LINE_GEOM_TYPES).any(),
                    gdf.geom_type.isin(POINT_GEOM_TYPES).any(),
                ]
            )

            # Check how many geometry types are in the clipped GeoDataFrame
            clip_types_total = sum(
                [
                    clipped.geom_type.isin(POLYGON_GEOM_TYPES).any(),
                    clipped.geom_type.isin(LINE_GEOM_TYPES).any(),
                    clipped.geom_type.isin(POINT_GEOM_TYPES).any(),
                ]
            )

            # Check there aren't any new geom types in the clipped GeoDataFrame
            more_types = orig_types_total < clip_types_total

            if orig_types_total > 1:
                warnings.warn(
                    "keep_geom_type can not be called on a mixed type GeoDataFrame.",
                    stacklevel=2,
                )
            elif new_collection or more_types:
                orig_type = gdf.geom_type.iloc[0]
                if new_collection:
                    clipped = clipped.explode(index_parts=False)
                if orig_type in POLYGON_GEOM_TYPES:
                    clipped = clipped.loc[clipped.geom_type.isin(POLYGON_GEOM_TYPES)]
                elif orig_type in LINE_GEOM_TYPES:
                    clipped = clipped.loc[clipped.geom_type.isin(LINE_GEOM_TYPES)]

    return clipped
