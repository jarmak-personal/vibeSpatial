import warnings

import numpy as np
import pandas as pd
import shapely

from vibespatial.api import GeoDataFrame, GeoSeries
from vibespatial.api._compat import PANDAS_GE_30
from vibespatial.api.geometry_array import (
    LINE_GEOM_TYPES,
    POINT_GEOM_TYPES,
    POLYGON_GEOM_TYPES,
    GeometryArray,
    _check_crs,
    _crs_mismatch_warn,
)
from vibespatial.runtime._runtime import ExecutionMode
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.fallbacks import strict_native_mode_enabled


def _extract_owned_pair(df1, df2):
    """Return (left_owned, right_owned) if both DataFrames have owned backing, else (None, None)."""
    ga1 = df1.geometry.values
    ga2 = df2.geometry.values
    left_owned = getattr(ga1, '_owned', None)
    right_owned = getattr(ga2, '_owned', None)
    if left_owned is not None and right_owned is not None:
        return left_owned, right_owned
    return None, None


def _ensure_geometry_column(df):
    """Ensure that the geometry column is called 'geometry'.

    If another column with that name exists, it will be dropped.
    """
    if not df._geometry_column_name == "geometry":
        if PANDAS_GE_30:
            if "geometry" in df.columns:
                df = df.drop("geometry", axis=1)
            df = df.rename_geometry("geometry")
        else:
            if "geometry" in df.columns:
                df.drop("geometry", axis=1, inplace=True)
            df.rename_geometry("geometry", inplace=True)
    return df


def _intersecting_index_pairs(df1, df2):
    # ADR-0036 boundary: produces spatial index arrays only.
    # sindex.query has its own owned-dispatch path (sindex.py lines 334-378)
    # that routes through query_spatial_index when both sides support owned.
    if not strict_native_mode_enabled():
        return df2.sindex.query(df1.geometry, predicate="intersects", sort=True)

    idx1, idx2 = df2.sindex._tree.query(
        np.asarray(df1.geometry.array, dtype=object),
        predicate="intersects",
    )
    idx1 = np.asarray(idx1, dtype=np.int32)
    idx2 = np.asarray(idx2, dtype=np.int32)
    # Sort by idx1 to match the non-strict path (sort=True) contract.
    # _overlay_difference relies on sorted idx1 for np.split grouping.
    order = np.argsort(idx1, kind="stable")
    return idx1[order], idx2[order]


def _assemble_intersection_attributes(idx1, idx2, df1, df2):
    """ADR-0036 boundary: attribute assembly from index arrays.

    Receives integer index arrays and attribute-only DataFrames (geometry
    columns already dropped).  Returns a merged DataFrame with attributes
    from both sides joined via the spatial index pairs.
    """
    pairs = pd.DataFrame({"__idx1": idx1, "__idx2": idx2})
    result = pairs.merge(
        df1,
        left_on="__idx1",
        right_index=True,
    )
    result = result.merge(
        df2,
        left_on="__idx2",
        right_index=True,
        suffixes=("_1", "_2"),
    )
    return result


def _overlay_intersection(df1, df2, left_owned=None, right_owned=None):
    """Overlay Intersection operation used in overlay function.

    Returns
    -------
    tuple[GeoDataFrame, bool]
        Result GeoDataFrame and whether the owned dispatch path was used.
    """
    # ADR-0036 boundary: spatial index produces index arrays only.
    idx1, idx2 = _intersecting_index_pairs(df1, df2)
    used_owned = False
    # Create pairs of geometries in both dataframes to be intersected
    if idx1.size > 0 and idx2.size > 0:
        # Owned-path dispatch: OwnedGeometryArray.take() operates at buffer
        # level (no Shapely materialization), then binary_constructive_owned
        # routes to GPU when available.  GeoSeries.take() breaks the DGA chain
        # by materializing to Shapely, so we bypass it when owned is present.
        # Note: GeometryArray.copy() preserves _owned, and __setitem__
        # invalidates it on mutation, so owned survives _make_valid when
        # all geometries are already valid.
        intersections = None
        if left_owned is not None and right_owned is not None:
            from vibespatial.constructive.binary_constructive import (
                binary_constructive_owned,
            )

            left_sub = left_owned.take(np.asarray(idx1))
            right_sub = right_owned.take(np.asarray(idx2))
            try:
                result_owned = binary_constructive_owned(
                    "intersection", left_sub, right_sub,
                )
                intersections = GeoSeries(
                    GeometryArray.from_owned(result_owned, crs=df1.crs),
                )
                used_owned = True
            except NotImplementedError:
                # binary_constructive_owned can't handle result types like
                # GeometryCollections — fall back using already-gathered subsets
                # to avoid re-materializing the full arrays.
                left_shapely = np.asarray(left_sub.to_shapely(), dtype=object)
                right_shapely = np.asarray(right_sub.to_shapely(), dtype=object)
                intersections = GeoSeries(
                    shapely.intersection(left_shapely, right_shapely),
                    crs=df1.crs,
                )

        if intersections is None:
            # ADR-0036 boundary: geometry operations on geometry arrays.
            left = df1.geometry.take(idx1)
            left.reset_index(drop=True, inplace=True)
            right = df2.geometry.take(idx2)
            right.reset_index(drop=True, inplace=True)
            intersections = left.intersection(right)

        poly_ix = intersections.geom_type.isin(POLYGON_GEOM_TYPES)
        intersections.loc[poly_ix] = intersections[poly_ix].make_valid()

        geom_intersect = intersections

        # ADR-0036 boundary: attribute assembly from index arrays.
        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)
        dfinter = _assemble_intersection_attributes(
            idx1, idx2,
            df1.drop(df1._geometry_column_name, axis=1),
            df2.drop(df2._geometry_column_name, axis=1),
        )

        return GeoDataFrame(dfinter, geometry=geom_intersect, crs=df1.crs), used_owned
    else:
        result = df1.iloc[:0].merge(
            df2.iloc[:0].drop(df2.geometry.name, axis=1),
            left_index=True,
            right_index=True,
            suffixes=("_1", "_2"),
        )
        result["__idx1"] = np.nan
        result["__idx2"] = np.nan
        return result[
            result.columns.drop(df1.geometry.name).tolist() + [df1.geometry.name]
        ], used_owned


def _overlay_difference(df1, df2, left_owned=None, right_owned=None):
    """Overlay Difference operation used in overlay function.

    Returns
    -------
    tuple[GeoDataFrame, bool]
        Result GeoDataFrame and whether the owned dispatch path was used.
    """
    # ADR-0036 boundary: spatial index produces index arrays only.
    idx1, idx2 = _intersecting_index_pairs(df1, df2)

    n_left = len(df1)
    used_owned = False
    result_geoms = None

    # Owned-path dispatch: GPU segmented union + GPU difference when both
    # DataFrames have owned backing.  Avoids Shapely materialization for
    # the union step when the segmented_union_all GPU kernel is available.
    if idx1.size > 0 and left_owned is not None and right_owned is not None:
        try:
            from vibespatial.constructive.binary_constructive import (
                binary_constructive_owned,
            )
            from vibespatial.kernels.constructive.segmented_union import (
                segmented_union_all,
            )

            # Gather right geometries at all neighbor indices (buffer-level).
            right_gathered = right_owned.take(idx2)

            # Build group offsets from the sorted idx1 split points.
            idx1_unique, idx1_split_at = np.unique(idx1, return_index=True)
            group_offsets = np.concatenate([idx1_split_at, [len(idx2)]])

            # GPU segmented union: one union per left geometry's neighbors.
            right_unions_owned = segmented_union_all(
                right_gathered, group_offsets,
            )

            # GPU difference: left[unique] - union(right_neighbors[unique]).
            left_sub = left_owned.take(idx1_unique)
            diff_owned = binary_constructive_owned(
                "difference", left_sub, right_unions_owned,
            )

            # Assemble full result: unchanged left rows from GeoDataFrame,
            # differenced rows materialized from owned result.
            diff_shapely = np.asarray(diff_owned.to_shapely(), dtype=object)
            result_geoms = np.asarray(df1.geometry, dtype=object).copy()
            result_geoms[idx1_unique] = diff_shapely

            used_owned = True
        except (ImportError, NotImplementedError):
            pass

    if result_geoms is None:
        # Vectorized grouped-union approach: for each left geometry, compute
        # left_i - union(overlapping right geometries).  Replaces per-geometry
        # Python loop with grouped shapely.union_all + vectorized shapely.difference.
        left_geoms = np.asarray(df1.geometry, dtype=object)
        result_geoms = left_geoms.copy()

        if idx1.size > 0:
            right_geoms = np.asarray(df2.geometry, dtype=object)
            right_unions = np.empty(n_left, dtype=object)
            right_unions.fill(None)

            # O(N log N) grouping via np.split — avoids O(K*N) per-group mask scan.
            idx1_unique, idx1_split_at = np.unique(idx1, return_index=True)
            idx2_groups = np.split(idx2, idx1_split_at[1:])
            for left_pos, neighbors_idx in zip(idx1_unique, idx2_groups):
                neighbors = right_geoms[neighbors_idx]
                if len(neighbors) == 1:
                    right_unions[left_pos] = neighbors[0]
                else:
                    right_unions[left_pos] = shapely.union_all(neighbors)

            has_neighbors = np.zeros(n_left, dtype=bool)
            has_neighbors[idx1_unique] = True
            result_geoms[has_neighbors] = shapely.difference(
                left_geoms[has_neighbors], right_unions[has_neighbors],
            )

    differences = GeoSeries(result_geoms, index=df1.index, crs=df1.crs)
    poly_ix = differences.geom_type.isin(POLYGON_GEOM_TYPES)
    differences.loc[poly_ix] = differences[poly_ix].make_valid()
    geom_diff = differences[~differences.is_empty].copy()
    dfdiff = df1[~differences.is_empty].copy()
    dfdiff[dfdiff._geometry_column_name] = geom_diff
    return dfdiff, used_owned


def _overlay_identity(df1, df2, left_owned=None, right_owned=None):
    """Overlay Identity operation used in overlay function.

    Returns (GeoDataFrame, bool) -- result and whether any sub-op used owned dispatch.
    """
    dfintersection, used_inter = _overlay_intersection(df1, df2, left_owned, right_owned)
    dfdifference, used_diff = _overlay_difference(df1, df2, left_owned, right_owned)
    dfdifference = _ensure_geometry_column(dfdifference)

    # Columns that were suffixed in dfintersection need to be suffixed in dfdifference
    # as well so they can be matched properly in concat.
    new_columns = [
        col if col in dfintersection.columns else f"{col}_1"
        for col in dfdifference.columns
    ]
    dfdifference.columns = new_columns

    # Now we can concatenate the two dataframes
    result = pd.concat([dfintersection, dfdifference], ignore_index=True, sort=False)

    # keep geometry column last
    columns = list(dfintersection.columns)
    columns.remove("geometry")
    columns.append("geometry")
    result = result.reindex(columns=columns)
    if not isinstance(result, GeoDataFrame):
        result = GeoDataFrame(result)
    if result.crs is None and df1.crs is not None:
        result = result.set_crs(df1.crs)
    return result, used_inter or used_diff


def _overlay_symmetric_diff(df1, df2, left_owned=None, right_owned=None):
    """Overlay Symmetric Difference operation used in overlay function.

    Returns (GeoDataFrame, bool) -- result and whether any sub-op used owned dispatch.
    """
    dfdiff1, used1 = _overlay_difference(df1, df2, left_owned, right_owned)
    dfdiff2, used2 = _overlay_difference(df2, df1, right_owned, left_owned)
    dfdiff1["__idx1"] = range(len(dfdiff1))
    dfdiff2["__idx2"] = range(len(dfdiff2))
    dfdiff1["__idx2"] = np.nan
    dfdiff2["__idx1"] = np.nan
    # ensure geometry name (otherwise merge goes wrong)
    dfdiff1 = _ensure_geometry_column(dfdiff1)
    dfdiff2 = _ensure_geometry_column(dfdiff2)
    # combine both 'difference' dataframes
    dfsym = dfdiff1.merge(
        dfdiff2, on=["__idx1", "__idx2"], how="outer", suffixes=("_1", "_2")
    )
    geometry = dfsym.geometry_1.copy()
    geometry.name = "geometry"
    # https://github.com/pandas-dev/pandas/issues/26468 use loc for now
    geometry.loc[dfsym.geometry_1.isnull()] = dfsym.loc[
        dfsym.geometry_1.isnull(), "geometry_2"
    ]
    dfsym.drop(["geometry_1", "geometry_2"], axis=1, inplace=True)
    dfsym.reset_index(drop=True, inplace=True)
    dfsym = GeoDataFrame(dfsym, geometry=geometry, crs=df1.crs)
    return dfsym, used1 or used2


def _overlay_union(df1, df2, left_owned=None, right_owned=None):
    """Overlay Union operation used in overlay function.

    Returns (GeoDataFrame, bool) -- result and whether any sub-op used owned dispatch.
    """
    dfinter, used_inter = _overlay_intersection(df1, df2, left_owned, right_owned)
    dfsym, used_sym = _overlay_symmetric_diff(df1, df2, left_owned, right_owned)
    dfunion = pd.concat([dfinter, dfsym], ignore_index=True, sort=False)
    # keep geometry column last
    columns = list(dfunion.columns)
    columns.remove("geometry")
    columns.append("geometry")
    result = dfunion.reindex(columns=columns)
    if not isinstance(result, GeoDataFrame):
        result = GeoDataFrame(result)
    if result.crs is None and df1.crs is not None:
        result = result.set_crs(df1.crs)
    return result, used_inter or used_sym


def overlay(df1, df2, how="intersection", keep_geom_type=None, make_valid=True):
    """Perform spatial overlay between two GeoDataFrames.

    Currently only supports data GeoDataFrames with uniform geometry types,
    i.e. containing only (Multi)Polygons, or only (Multi)Points, or a
    combination of (Multi)LineString and LinearRing shapes.
    Implements several methods that are all effectively subsets of the union.

    See the User Guide page :doc:`../../user_guide/set_operations` for details.

    Parameters
    ----------
    df1 : GeoDataFrame
    df2 : GeoDataFrame
    how : string
        Method of spatial overlay: 'intersection', 'union',
        'identity', 'symmetric_difference' or 'difference'.
    keep_geom_type : bool
        If True, return only geometries of the same geometry type as df1 has,
        if False, return all resulting geometries. Default is None,
        which will set keep_geom_type to True but warn upon dropping
        geometries.
    make_valid : bool, default True
        If True, any invalid input geometries are corrected with a call to make_valid(),
        if False, a `ValueError` is raised if any input geometries are invalid.

    Returns
    -------
    df : GeoDataFrame
        GeoDataFrame with new set of polygons and attributes
        resulting from the overlay

    Examples
    --------
    >>> from shapely.geometry import Polygon
    >>> polys1 = geopandas.GeoSeries([Polygon([(0,0), (2,0), (2,2), (0,2)]),
    ...                               Polygon([(2,2), (4,2), (4,4), (2,4)])])
    >>> polys2 = geopandas.GeoSeries([Polygon([(1,1), (3,1), (3,3), (1,3)]),
    ...                               Polygon([(3,3), (5,3), (5,5), (3,5)])])
    >>> df1 = geopandas.GeoDataFrame({'geometry': polys1, 'df1_data':[1,2]})
    >>> df2 = geopandas.GeoDataFrame({'geometry': polys2, 'df2_data':[1,2]})

    >>> geopandas.overlay(df1, df2, how='union')
        df1_data  df2_data                                           geometry
    0       1.0       1.0                POLYGON ((2 2, 2 1, 1 1, 1 2, 2 2))
    1       2.0       1.0                POLYGON ((2 2, 2 3, 3 3, 3 2, 2 2))
    2       2.0       2.0                POLYGON ((4 4, 4 3, 3 3, 3 4, 4 4))
    3       1.0       NaN      POLYGON ((2 0, 0 0, 0 2, 1 2, 1 1, 2 1, 2 0))
    4       2.0       NaN  MULTIPOLYGON (((3 4, 3 3, 2 3, 2 4, 3 4)), ((4...
    5       NaN       1.0  MULTIPOLYGON (((2 3, 2 2, 1 2, 1 3, 2 3)), ((3...
    6       NaN       2.0      POLYGON ((3 5, 5 5, 5 3, 4 3, 4 4, 3 4, 3 5))

    >>> geopandas.overlay(df1, df2, how='intersection')
       df1_data  df2_data                             geometry
    0         1         1  POLYGON ((2 2, 2 1, 1 1, 1 2, 2 2))
    1         2         1  POLYGON ((2 2, 2 3, 3 3, 3 2, 2 2))
    2         2         2  POLYGON ((4 4, 4 3, 3 3, 3 4, 4 4))

    >>> geopandas.overlay(df1, df2, how='symmetric_difference')
        df1_data  df2_data                                           geometry
    0       1.0       NaN      POLYGON ((2 0, 0 0, 0 2, 1 2, 1 1, 2 1, 2 0))
    1       2.0       NaN  MULTIPOLYGON (((3 4, 3 3, 2 3, 2 4, 3 4)), ((4...
    2       NaN       1.0  MULTIPOLYGON (((2 3, 2 2, 1 2, 1 3, 2 3)), ((3...
    3       NaN       2.0      POLYGON ((3 5, 5 5, 5 3, 4 3, 4 4, 3 4, 3 5))

    >>> geopandas.overlay(df1, df2, how='difference')
                                                geometry  df1_data
    0      POLYGON ((2 0, 0 0, 0 2, 1 2, 1 1, 2 1, 2 0))         1
    1  MULTIPOLYGON (((3 4, 3 3, 2 3, 2 4, 3 4)), ((4...         2

    >>> geopandas.overlay(df1, df2, how='identity')
       df1_data  df2_data                                           geometry
    0         1       1.0                POLYGON ((2 2, 2 1, 1 1, 1 2, 2 2))
    1         2       1.0                POLYGON ((2 2, 2 3, 3 3, 3 2, 2 2))
    2         2       2.0                POLYGON ((4 4, 4 3, 3 3, 3 4, 4 4))
    3         1       NaN      POLYGON ((2 0, 0 0, 0 2, 1 2, 1 1, 2 1, 2 0))
    4         2       NaN  MULTIPOLYGON (((3 4, 3 3, 2 3, 2 4, 3 4)), ((4...

    See Also
    --------
    sjoin : spatial join
    GeoDataFrame.overlay : equivalent method

    Notes
    -----
    Every operation in GeoPandas is planar, i.e. the potential third
    dimension is not taken into account.
    """
    # Allowed operations
    allowed_hows = [
        "intersection",
        "union",
        "identity",
        "symmetric_difference",
        "difference",  # aka erase
    ]
    # Error Messages
    if how not in allowed_hows:
        raise ValueError(f"`how` was '{how}' but is expected to be in {allowed_hows}")

    if isinstance(df1, GeoSeries) or isinstance(df2, GeoSeries):
        raise NotImplementedError(
            "overlay currently only implemented for GeoDataFrames"
        )

    if not _check_crs(df1, df2):
        _crs_mismatch_warn(df1, df2, stacklevel=3)

    if keep_geom_type is None:
        keep_geom_type = True
        keep_geom_type_warning = True
    else:
        keep_geom_type_warning = False

    for i, df in enumerate([df1, df2]):
        poly_check = df.geom_type.isin(POLYGON_GEOM_TYPES).any()
        lines_check = df.geom_type.isin(LINE_GEOM_TYPES).any()
        points_check = df.geom_type.isin(POINT_GEOM_TYPES).any()
        if sum([poly_check, lines_check, points_check]) > 1:
            raise NotImplementedError(f"df{i + 1} contains mixed geometry types.")

    if how == "intersection":
        box_gdf1 = df1.total_bounds
        box_gdf2 = df2.total_bounds

        if not (
            ((box_gdf1[0] <= box_gdf2[2]) and (box_gdf2[0] <= box_gdf1[2]))
            and ((box_gdf1[1] <= box_gdf2[3]) and (box_gdf2[1] <= box_gdf1[3]))
        ):
            result = df1.iloc[:0].merge(
                df2.iloc[:0].drop(df2.geometry.name, axis=1),
                left_index=True,
                right_index=True,
                suffixes=("_1", "_2"),
            )
            return result[
                result.columns.drop(df1.geometry.name).tolist() + [df1.geometry.name]
            ]

    # Computations
    def _make_valid(df):
        df = df.copy()
        if df.geom_type.isin(POLYGON_GEOM_TYPES).all():
            mask = ~df.geometry.is_valid
            col = df._geometry_column_name
            if make_valid:
                df.loc[mask, col] = df.loc[mask, col].make_valid()
                # Extract only the input geometry type, as make_valid may change it
                if mask.any():
                    df = _collection_extract(
                        df, geom_type="Polygon", keep_geom_type_warning=False
                    )

            elif mask.any():
                raise ValueError(
                    "You have passed make_valid=False along with "
                    f"{mask.sum()} invalid input geometries. "
                    "Use make_valid=True or make sure that all geometries "
                    "are valid before using overlay."
                )
        return df

    # Determine the geometry type before make_valid, as make_valid may change it
    if keep_geom_type:
        geom_type = df1.geom_type.iloc[0]

    df1 = _make_valid(df1)
    df2 = _make_valid(df2)

    # Extract owned arrays AFTER _make_valid.  GeometryArray.copy() now
    # preserves _owned backing, and __setitem__ invalidates it only for
    # mutated rows.  If _make_valid mutated all rows or dropped rows via
    # _collection_extract, _owned will already be None here.
    left_owned, right_owned = _extract_owned_pair(df1, df2)

    _used_owned = False
    with warnings.catch_warnings():  # CRS checked above, suppress array-level warning
        warnings.filterwarnings("ignore", message="CRS mismatch between the CRS")
        if how == "difference":
            result, _used_owned = _overlay_difference(
                df1, df2, left_owned, right_owned,
            )
        elif how == "intersection":
            result, _used_owned = _overlay_intersection(
                df1, df2, left_owned, right_owned,
            )
        elif how == "symmetric_difference":
            result, _used_owned = _overlay_symmetric_diff(
                df1, df2, left_owned, right_owned,
            )
        elif how == "union":
            result, _used_owned = _overlay_union(df1, df2, left_owned, right_owned)
        elif how == "identity":
            result, _used_owned = _overlay_identity(df1, df2, left_owned, right_owned)

        if how in ["intersection", "symmetric_difference", "union", "identity"]:
            result.drop(["__idx1", "__idx2"], axis=1, inplace=True)

    record_dispatch_event(
        surface="geopandas.overlay",
        operation=f"overlay_{how}",
        implementation="owned_dispatch" if _used_owned else "shapely_host",
        reason=(
            f"{how} via owned-path dispatch"
            if _used_owned
            else "no owned backing or host fallback"
        ),
        detail=(
            f"left_rows={len(df1)}, right_rows={len(df2)}, "
            f"how={how}, owned={left_owned is not None}"
        ),
        requested=ExecutionMode.AUTO,
        selected=ExecutionMode.GPU if _used_owned else ExecutionMode.CPU,
    )

    if keep_geom_type:
        result = _collection_extract(result, geom_type, keep_geom_type_warning)

    result.reset_index(drop=True, inplace=True)
    return result


def _collection_extract(df, geom_type, keep_geom_type_warning):
    # Check input
    if geom_type in POLYGON_GEOM_TYPES:
        geom_types = POLYGON_GEOM_TYPES
    elif geom_type in LINE_GEOM_TYPES:
        geom_types = LINE_GEOM_TYPES
    elif geom_type in POINT_GEOM_TYPES:
        geom_types = POINT_GEOM_TYPES
    else:
        raise TypeError(f"`geom_type` does not support {geom_type}.")

    result = df.copy()

    # First we filter the geometry types inside GeometryCollections objects
    # (e.g. GeometryCollection([polygon, point]) -> polygon)
    # we do this separately on only the relevant rows, as this is an expensive
    # operation (an expensive no-op for geometry types other than collections)
    is_collection = result.geom_type == "GeometryCollection"
    if is_collection.any():
        geom_col = result._geometry_column_name
        collections = result.loc[is_collection, [geom_col]]

        exploded = collections.reset_index(drop=True).explode(index_parts=True)
        exploded = exploded.reset_index(level=0)

        orig_num_geoms_exploded = exploded.shape[0]
        exploded.loc[~exploded.geom_type.isin(geom_types), geom_col] = None
        num_dropped_collection = (
            orig_num_geoms_exploded - exploded.geometry.isna().sum()
        )

        # level_0 created with above reset_index operation
        # and represents the original geometry collections
        # TODO avoiding dissolve to call union_all in this case could further
        # improve performance (we only need to collect geometries in their
        # respective Multi version)
        dissolved = exploded.dissolve(by="level_0")
        result.loc[is_collection, geom_col] = dissolved[geom_col].values
    else:
        num_dropped_collection = 0

    # Now we filter all geometries (in theory we don't need to do this
    # again for the rows handled above for GeometryCollections, but filtering
    # them out is probably more expensive as simply including them when this
    # is typically about only a few rows)
    orig_num_geoms = result.shape[0]
    result = result.loc[result.geom_type.isin(geom_types)]
    num_dropped = orig_num_geoms - result.shape[0]

    if (num_dropped > 0 or num_dropped_collection > 0) and keep_geom_type_warning:
        warnings.warn(
            "`keep_geom_type=True` in overlay resulted in "
            f"{num_dropped + num_dropped_collection} dropped geometries of different "
            "geometry types than df1 has. Set `keep_geom_type=False` to retain all "
            "geometries",
            UserWarning,
            stacklevel=2,
        )

    return result
