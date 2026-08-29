from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import shapely
from shapely.geometry.base import BaseGeometry

from vibespatial.api import geometry_array as array
from vibespatial.api import geoseries
from vibespatial.geometry.api_registry import register_device_spatial_index_factory
from vibespatial.geometry.owned import OwnedGeometryArray
from vibespatial.runtime import ExecutionMode, get_requested_mode
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.fallbacks import record_fallback_event
from vibespatial.runtime.materialization import (
    NativeExportBoundary,
    record_native_export_boundary,
)
from vibespatial.runtime.residency import Residency
from vibespatial.spatial.indexing import compact_indexed_spatial_input
from vibespatial.spatial.query import (
    build_owned_spatial_index,
    nearest_spatial_index,
    query_spatial_index,
    supports_owned_spatial_input,
)

from . import _compat as compat

PREDICATES = {p.name for p in shapely.strtree.BinaryPredicate} | {None}
OWNED_QUERY_PREDICATES = PREDICATES

if compat.GEOS_GE_310:
    PREDICATES.update(["dwithin"])


class SpatialIndex:
    """A simple wrapper around Shapely's STRTree.

    Parameters
    ----------
    geometry : np.array of Shapely geometries
        Geometries from which to build the spatial index.
    """

    def __init__(self, geometry, geometry_array=None):
        # set empty geometries to None to avoid segfault on GEOS <= 3.6
        # see:
        # https://github.com/pygeos/pygeos/issues/146
        # https://github.com/pygeos/pygeos/issues/147
        non_empty = geometry.copy()
        non_empty[shapely.is_empty(non_empty)] = None
        # set empty geometries to None to maintain indexing
        self._tree = shapely.STRtree(non_empty)
        # store geometries, including empty geometries for user access
        self.geometries = geometry.copy()
        self._geometry_array = geometry_array
        self._native_spatial_index = None
        self._native_spatial_index_source_token = None
        self._native_spatial_index_flat_index_id = None

    @classmethod
    def _from_device_geometry_array(cls, device_geometry_array):
        """Construct a SpatialIndex backed by a DeviceGeometryArray.

        Defers STRtree construction until a non-owned query path requires it.
        All owned-dispatch queries work without Shapely materialization.
        """
        obj = object.__new__(cls)
        obj._tree = None          # lazy — built on first STRtree-fallback query
        obj.geometries = None     # lazy — populated alongside _tree
        obj._geometry_array = device_geometry_array
        obj._native_spatial_index = None
        obj._native_spatial_index_source_token = None
        obj._native_spatial_index_flat_index_id = None
        return obj

    def _ensure_strtree(self):
        """Lazily build the STRtree when a fallback path needs it."""
        if self._tree is not None:
            return
        if (
            self._geometry_array is not None
            and getattr(self._geometry_array, "_owned", None) is not None
            and getattr(self._geometry_array, "_shapely_data", None) is None
        ):
            record_fallback_event(
                surface="geopandas.array.sindex",
                requested=get_requested_mode(),
                selected=ExecutionMode.CPU,
                reason="owned-backed GeometryArray.sindex materializes a host STRtree",
                detail=f"rows={len(self._geometry_array)}",
                pipeline="spatial_index",
                d2h_transfer=True,
            )
        geometry = np.asarray(self._geometry_array._data, dtype=object)
        non_empty = geometry.copy()
        non_empty[shapely.is_empty(non_empty)] = None
        self._tree = shapely.STRtree(non_empty)
        self.geometries = geometry.copy()

    @property
    def valid_query_predicates(self):
        """Returns valid predicates for the spatial index.

        Returns
        -------
        set
            Set of valid predicates for this spatial index.

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> s = geopandas.GeoSeries([Point(0, 0), Point(1, 1)])
        >>> s.sindex.valid_query_predicates  # doctest: +SKIP
        {None, "contains", "contains_properly", "covered_by", "covers", \
"crosses", "dwithin", "intersects", "overlaps", "touches", "within"}
        """
        return PREDICATES

    def _query_predicate_kwargs(self, predicate, distance) -> dict[str, object]:
        """Validate shared public query arguments and return STRtree kwargs."""
        if predicate not in self.valid_query_predicates:
            if predicate == "dwithin":
                raise ValueError("predicate = 'dwithin' requires GEOS >= 3.10.0")
            raise ValueError(
                f"Got predicate='{predicate}'; "
                f"`predicate` must be one of {self.valid_query_predicates}"
            )
        if predicate == "dwithin":
            if distance is None:
                raise ValueError(
                    "'distance' parameter is required for 'dwithin' predicate"
                )
            return {"distance": distance}
        if distance is not None:
            raise ValueError(
                "'distance' parameter is only supported in combination with "
                "'dwithin' predicate"
            )
        return {}

    def query(
        self,
        geometry,
        predicate=None,
        sort=False,
        distance=None,
        output_format="indices",
        return_device=False,
    ):
        """
        Return all combinations of each input geometry
        and tree geometries where the bounding box of each input geometry
        intersects the bounding box of a tree geometry.

        The result can be returned as an array of 'indices' or a boolean 'sparse' or
        'dense' array. This can be controlled using the ``output_format`` keyword.
        Options are as follows.

        ``'indices'``
            If the input geometry is a scalar, this returns an array of shape (n, ) with
            the indices of the matching tree geometries.  If the input geometry is an
            array_like, this returns an array with shape (2,n) where the subarrays
            correspond to the indices of the input geometries and indices of the
            tree geometries associated with each.  To generate an array of pairs of
            input geometry index and tree geometry index, simply transpose the
            result.
        ``'sparse'``
            If the input geometry is a scalar, this returns a boolean scipy.sparse COO
            array of shape (len(tree), ) with boolean values marking whether the
            bounding box of a geometry in the tree intersects a bounding box of a given
            scalar. If the input geometry is an array_like, this returns a boolean
            scipy.sparse COO array with shape (len(tree), n) with boolean values marking
            whether the bounding box of a geometry in the tree intersects a bounding box
            of a given scalar.
        ``'dense'``
            If the input geometry is a scalar, this returns a boolean numpy
            array of shape (len(tree), ) with boolean values marking whether the
            bounding box of a geometry in the tree intersects a bounding box of a given
            scalar. If the input geometry is an array_like, this returns a boolean
            numpy array with shape (len(tree), n) with boolean values marking
            whether the bounding box of a geometry in the tree intersects a bounding box
            of a given scalar.

        If a predicate is provided, the tree geometries are first queried based
        on the bounding box of the input geometry and then are further filtered
        to those that meet the predicate when comparing the input geometry to
        the tree geometry: ``predicate(geometry, tree_geometry)``.

        The 'dwithin' predicate requires GEOS >= 3.10.

        Bounding boxes are limited to two dimensions and are axis-aligned
        (equivalent to the ``bounds`` property of a geometry); any Z values
        present in input geometries are ignored when querying the tree.

        Any input geometry that is None or empty will never match geometries in
        the tree.

        See the User Guide page :doc:`../../user_guide/spatial_indexing` for more.

        Parameters
        ----------
        geometry : shapely.Geometry or array-like of geometries \
(numpy.ndarray, GeoSeries, GeometryArray)
            A single shapely geometry or array of geometries to query against
            the spatial index. For array-like, accepts both GeoPandas geometry
            iterables (GeoSeries, GeometryArray) or a numpy array of Shapely
            geometries.
        predicate : {None, "contains", "contains_properly", "covered_by", "covers", \
"crosses", "intersects", "overlaps", "touches", "within", "dwithin"}, optional
            If predicate is provided, the input geometries are tested
            using the predicate function against each item in the tree
            whose extent intersects the envelope of the input geometry:
            ``predicate(input_geometry, tree_geometry)``.
            If possible, prepared geometries are used to help speed up the
            predicate operation.
        sort : bool, default False
            If True, the results will be sorted in ascending order. In case
            of 2D array, the result is sorted lexicographically using the
            geometries' indexes as the primary key and the sindex's indexes
            as the secondary key.
            If False, no additional sorting is applied (results are often
            sorted but there is no guarantee).
            Applicable only if output_format="indices".
        distance : number or array_like, optional
            Distances around each input geometry within which to query the tree for
            the 'dwithin' predicate. If array_like, shape must be broadcastable to shape
            of geometry. Required if ``predicate='dwithin'``.
        output_format : {"indices", "sparse", "dense"}, default "indices"
            Type of the output format representing the result of the query.
            Private ``return_device=True`` callers must use ``"indices"``;
            native consumers that need pair flow should use ``query_relation()``
            because dense and sparse outputs are public compatibility exports.

        Returns
        -------
        `If geometry is a scalar:`

        ndarray with shape (n,)
            Integer indices for matching geometries from the spatial index
            tree geometries.  If ``output_format="indices"``.

        OR

        scipy.sparse COO array with shape (len(tree), )
            Boolean array aligned with array of geometries in the tree.
            If ``output_format="sparse"``.

        OR

        ndarray with shape (len(tree), )
            Boolean array aligned with array of geometries in the tree.
            If ``output_format="dense"``.


        `If geometry is an array_like:`

        ndarray with shape (2, n)
            The first subarray contains input geometry integer indices.
            The second subarray contains tree geometry integer indices.
            If ``output_format="indices"``.

        OR

        scipy.sparse COO array with shape (len(tree), n)
            Boolean array aligned with array of geometries in the tree along axis 0 and
            with ``geometry`` along axis 1.
            If ``output_format="sparse"``.

        OR

        ndarray with shape (len(tree), n)
            Boolean array aligned with array of geometries in the tree along axis 0 and
            with ``geometry`` along axis 1.
            If ``output_format="dense"``.


        Examples
        --------
        >>> from shapely.geometry import Point, box
        >>> s = geopandas.GeoSeries(geopandas.points_from_xy(range(10), range(10)))
        >>> s
        0    POINT (0 0)
        1    POINT (1 1)
        2    POINT (2 2)
        3    POINT (3 3)
        4    POINT (4 4)
        5    POINT (5 5)
        6    POINT (6 6)
        7    POINT (7 7)
        8    POINT (8 8)
        9    POINT (9 9)
        dtype: geometry

        Querying the tree with a scalar geometry:

        >>> s.sindex.query(box(1, 1, 3, 3))
        array([1, 2, 3])

        >>> s.sindex.query(box(1, 1, 3, 3), predicate="contains")
        array([2])

        Querying the tree with an array of geometries:

        >>> s2 = geopandas.GeoSeries([box(2, 2, 4, 4), box(5, 5, 6, 6)])
        >>> s2
        0    POLYGON ((4 2, 4 4, 2 4, 2 2, 4 2))
        1    POLYGON ((6 5, 6 6, 5 6, 5 5, 6 5))
        dtype: geometry

        >>> s.sindex.query(s2)
        array([[0, 0, 0, 1, 1],
               [2, 3, 4, 5, 6]])

        >>> s.sindex.query(s2, predicate="contains")
        array([[0],
               [3]])

        >>> s.sindex.query(box(1, 1, 3, 3), predicate="dwithin", distance=0)
        array([1, 2, 3])

        >>> s.sindex.query(box(1, 1, 3, 3), predicate="dwithin", distance=2)
        array([0, 1, 2, 3, 4])

        Returning boolean arrays:

        >>> s.sindex.query(box(1, 1, 3, 3), output_format="sparse")
        <COOrdinate sparse array of dtype 'bool'
            with 3 stored elements and shape (10,)>

        >>> s.sindex.query(box(1, 1, 3, 3), output_format="dense")
        array([False,  True,  True,  True, False, False, False, False, False,
               False])

        >>> s.sindex.query(s2, output_format="sparse")
        <COOrdinate sparse array of dtype 'bool'
            with 5 stored elements and shape (10, 2)>

        >>> s.sindex.query(s2, output_format="dense")
        array([[False, False],
               [False, False],
               [ True, False],
               [ True, False],
               [ True, False],
               [False,  True],
               [False,  True],
               [False, False],
               [False, False],
               [False, False]])

        Notes
        -----
        In the context of a spatial join, input geometries are the "left"
        geometries that determine the order of the results, and tree geometries
        are "right" geometries that are joined against the left geometries. This
        effectively performs an inner join, where only those combinations of
        geometries that can be joined based on overlapping bounding boxes or
        optional predicate are returned.
        """
        if return_device and output_format != "indices":
            raise ValueError(
                "return_device=True is only supported with output_format='indices'; "
                "native consumers should use query_relation() instead of dense "
                "or sparse public exports"
            )

        kwargs = self._query_predicate_kwargs(predicate, distance)

        raw_geometry = geometry
        precomputed_query_bounds = None
        raw_box_array_fast_path = False
        if (
            get_requested_mode() is not ExecutionMode.CPU
            and predicate in (None, "intersects")
            and isinstance(raw_geometry, np.ndarray)
            and raw_geometry.ndim >= 1
            and self._supports_owned_tree_input()
        ):
            tree_owned, flat_index = self._owned_flat_sindex()
            if getattr(flat_index, "regular_grid", None) is not None:
                from vibespatial.spatial.query_box import _extract_box_query_bounds_shapely

                precomputed_query_bounds = _extract_box_query_bounds_shapely(raw_geometry)
                raw_box_array_fast_path = precomputed_query_bounds is not None
        if get_requested_mode() is not ExecutionMode.CPU and (
            raw_box_array_fast_path
            or (
                predicate in OWNED_QUERY_PREDICATES
                and self._supports_owned_query_input(raw_geometry)
            )
        ):
            native_query = self._query_native_relation_for_public_output(
                raw_geometry,
                predicate=predicate,
                sort=sort,
                distance=distance,
                output_format=output_format,
                return_device=return_device,
                raw_box_array_fast_path=raw_box_array_fast_path,
                precomputed_query_bounds=precomputed_query_bounds,
            )
            if native_query is not None:
                output, execution = native_query
                record_dispatch_event(
                    surface="geopandas.sindex.query",
                    operation="query",
                    implementation="native_spatial_index",
                    reason=(
                        "NativeSpatialIndex produced relation pairs for "
                        "public sindex.query export"
                    ),
                    detail=f"predicate={predicate!r}, output_format={output_format!r}",
                    requested=execution.requested,
                    selected=execution.selected,
                )
                self._record_public_spatial_export(
                    surface="geopandas.sindex.query",
                    operation="sindex_query",
                    target=f"sindex-{output_format}",
                    reason=(
                        "NativeSpatialIndex relation pairs exported to public "
                        "sindex.query result"
                    ),
                    output=output,
                    detail=f"predicate={predicate!r}, output_format={output_format!r}",
                )
                return output
            output = self._query_owned_public(
                raw_geometry,
                predicate=predicate,
                sort=sort,
                distance=distance,
                output_format=output_format,
                return_device=return_device,
                raw_box_array_fast_path=raw_box_array_fast_path,
                precomputed_query_bounds=precomputed_query_bounds,
            )
            self._record_public_spatial_export(
                surface="geopandas.sindex.query",
                operation="sindex_query",
                target=f"sindex-{output_format}",
                reason="owned spatial query exported to public sindex.query result",
                output=output,
                detail=f"predicate={predicate!r}, output_format={output_format!r}",
            )
            return output

        self._ensure_strtree()
        geometry = self._as_geometry_array(raw_geometry)

        record_dispatch_event(
            surface="geopandas.sindex.query",
            operation="query",
            implementation="strtree_host",
            reason="Shapely STRtree query is the current first-class host implementation",
            detail=f"predicate={predicate!r}, output_format={output_format!r}",
            selected=ExecutionMode.CPU,
        )

        indices = self._tree.query(geometry, predicate=predicate, **kwargs)

        if output_format == "indices" and sort:
            if indices.ndim == 1:
                indices = np.sort(indices)
            else:
                # sort by first array (geometry) and then second (tree)
                geo_idx, tree_idx = indices
                sort_indexer = np.lexsort((tree_idx, geo_idx))
                indices = np.vstack((geo_idx[sort_indexer], tree_idx[sort_indexer]))

        if output_format == "sparse":
            scipy = compat.import_optional_dependency("scipy")

            if indices.ndim == 1:
                return scipy.sparse.coo_array(
                    (np.ones(len(indices), dtype=np.bool_), indices.reshape(1, -1)),
                    shape=(len(self.geometries),),
                    dtype=np.bool_,
                )
            return scipy.sparse.coo_array(
                (np.ones(len(indices[0]), dtype=np.bool_), indices[::-1]),
                shape=(len(self.geometries), len(geometry)),
                dtype=np.bool_,
            )

        if output_format == "dense":
            if indices.ndim == 1:
                dense = np.zeros(len(self.geometries), dtype=bool)
                dense[indices] = True
            else:
                dense = np.zeros((len(self.geometries), len(geometry)), dtype=bool)
                tree, other = indices[::-1]
                dense[tree, other] = True
            return dense

        if output_format == "indices":
            return indices

        raise ValueError(
            f"Invalid output_format: '{output_format}'. "
            "Use one of 'indices', 'sparse', 'dense'."
        )

    def query_aggregate(
        self,
        geometry,
        aggregations: Mapping[str, str | tuple[object, str]],
        *,
        predicate=None,
        distance=None,
    ) -> pd.DataFrame:
        """Aggregate spatial-index matches without exporting relation pairs.

        Parameters
        ----------
        geometry : shapely.Geometry or array-like of geometries
            Input geometries queried against the indexed tree geometries.
            Output rows preserve this input order and, for a GeoSeries, index.
        aggregations : mapping
            Output-column names mapped either to ``"size"`` for the number of
            matches or to ``(values, "sum")``. ``values`` must be a numeric
            one-dimensional public Series or array aligned with the indexed
            tree geometries.
        predicate : str, optional
            Exact spatial predicate with the same meaning as :meth:`query`.
        distance : number or array-like, optional
            Distance for the ``"dwithin"`` predicate.

        Returns
        -------
        pandas.DataFrame
            One eager, pandas-compatible row per input geometry. Empty groups
            have size and sum equal to zero.

        Notes
        -----
        This vibeSpatial extension is a public eager reduction API, not a lazy
        relation or proxy dataframe. Device-backed inputs lower to a
        ``NativeRelation`` grouped reduction. The computed input-sized columns
        remain device-backed through ordinary public Series arithmetic and
        transfer only when a caller explicitly requests host values.
        Unsupported inputs use the exact observable CPU path.
        """
        normalized = self._normalize_query_aggregations(aggregations)
        self._validate_query_aggregate_values(normalized)
        query_row_count, scalar = self._query_cardinality(geometry)
        output_index = self._query_aggregate_output_index(
            geometry,
            query_row_count=query_row_count,
            scalar=scalar,
        )
        native, reusable_relation = self._query_aggregate_native(
            geometry,
            normalized,
            predicate=predicate,
            distance=distance,
            query_row_count=query_row_count,
            output_index=output_index,
        )
        if native is not None:
            return native

        record_fallback_event(
            surface="vibespatial.api.SpatialIndex.query_aggregate",
            requested=get_requested_mode(),
            selected=ExecutionMode.CPU,
            reason=(
                "query aggregate requires owned geometry and device-backed "
                "numeric source expressions"
            ),
            detail=(
                f"predicate={predicate!r}, query_rows={query_row_count}, "
                f"tree_rows={self.size}"
            ),
            pipeline="spatial_query_aggregate",
            d2h_transfer=any(
                self._query_aggregate_value_is_device(value)
                for value, reducer in normalized.values()
                if reducer != "size"
            ),
        )
        return self._query_aggregate_host(
            geometry,
            normalized,
            predicate=predicate,
            distance=distance,
            query_row_count=query_row_count,
            output_index=output_index,
            scalar=scalar,
            relation=reusable_relation,
        )

    def query_any(
        self,
        geometry,
        *,
        predicate=None,
        distance=None,
    ) -> pd.Series:
        """Return whether each input geometry has any spatial-index match.

        This is an eager, input-sized public reduction. The result preserves
        the input order and GeoSeries index without exposing or materializing
        the underlying relation pairs. Device-backed inputs retain a native
        boolean column until an explicit public host export.

        Parameters
        ----------
        geometry : shapely.Geometry or array-like of geometries
            Input geometries queried against the indexed tree geometries.
        predicate : str, optional
            Exact spatial predicate with the same meaning as :meth:`query`.
        distance : number or array-like, optional
            Distance for the ``"dwithin"`` predicate.

        Returns
        -------
        pandas.Series
            One eager boolean value per input geometry.
        """
        self._query_predicate_kwargs(predicate, distance)
        query_row_count, scalar = self._query_cardinality(geometry)
        output_index = self._query_aggregate_output_index(
            geometry,
            query_row_count=query_row_count,
            scalar=scalar,
        )
        native = self._query_any_native(
            geometry,
            predicate=predicate,
            distance=distance,
            query_row_count=query_row_count,
            output_index=output_index,
        )
        if native is not None:
            return native

        record_fallback_event(
            surface="vibespatial.api.SpatialIndex.query_any",
            requested=get_requested_mode(),
            selected=ExecutionMode.CPU,
            reason="query-any requires owned geometry for native row reduction",
            detail=(
                f"predicate={predicate!r}, query_rows={query_row_count}, "
                f"tree_rows={self.size}"
            ),
            pipeline="spatial_query_any",
            d2h_transfer=(
                self._query_any_input_is_device_backed(self._geometry_array)
                or self._query_any_input_is_device_backed(geometry)
            ),
        )
        pairs = self.query(
            geometry,
            predicate=predicate,
            distance=distance,
            sort=False,
        )
        mask = np.zeros(query_row_count, dtype=bool)
        if scalar:
            mask[0] = bool(np.asarray(pairs).size)
        elif np.asarray(pairs).ndim == 2 and np.asarray(pairs).shape[1]:
            mask[np.asarray(pairs[0], dtype=np.int64)] = True
        return pd.Series(mask, index=output_index, name="has_match")

    @staticmethod
    def _query_any_public_series(
        mask,
        *,
        output_index,
        rowset=None,
        selection=None,
    ) -> pd.Series:
        from vibespatial.api._native_public_arrays import NativeBooleanMaskArray

        return pd.Series(
            NativeBooleanMaskArray(
                row_count=len(output_index),
                rowset=rowset,
                selection=selection,
                mask_values=mask,
                export_surface="vibespatial.api.SpatialIndex.query_any",
                export_operation="query_any_mask_to_public_array",
            ),
            index=output_index,
            name="has_match",
        )

    @staticmethod
    def _query_any_input_is_device_backed(geometry) -> bool:
        values = geometry.values if isinstance(geometry, geoseries.GeoSeries) else geometry
        owned = (
            values
            if isinstance(values, OwnedGeometryArray)
            else getattr(values, "_owned", None) or getattr(values, "owned", None)
        )
        return getattr(owned, "residency", None) is Residency.DEVICE

    def _query_any_buffer_point_native(
        self,
        geometry,
        *,
        predicate,
        distance,
        query_row_count: int,
        output_index,
        query_token: str | None,
    ):
        """Lower intersects against point buffers to bounded exact existence.

        Physical shape: one nearest-point metric bound per query row certifies
        definite true/false rows. Only the thin threshold-ambiguous row mask is
        refined by the buffer index's bounded intersects semijoin. No duplicate
        match pairs are materialized.
        """
        if predicate != "intersects" or distance is not None:
            return None
        values = self._geometry_array
        tag = getattr(values, "_provenance", None)
        if tag is None:
            return None

        from vibespatial.runtime.provenance import _r1_preconditions_met

        if tag.operation != "buffer" or not _r1_preconditions_met(tag):
            return None
        source_values = tag.source_array
        if not hasattr(source_values, "owned_flat_sindex"):
            return None
        source_owned, _source_index = source_values.owned_flat_sindex()
        query_owned = self._owned_query_input(geometry)
        if not (
            isinstance(source_owned, OwnedGeometryArray)
            and isinstance(query_owned, OwnedGeometryArray)
        ):
            return None

        import cupy as cp

        from vibespatial.constructive.centroid import centroid_owned
        from vibespatial.geometry.owned import device_mask_owned_capacity
        from vibespatial.kernels.core.geometry_analysis import (
            compute_geometry_bounds_device,
        )

        radius = float(tag.get_param("distance"))
        quad_segs = int(tag.get_param("quad_segs"))
        buffer_inradius = radius * float(np.cos(np.pi / (4 * quad_segs)))
        query_centers = centroid_owned(
            query_owned,
            dispatch_mode=ExecutionMode.GPU,
            precision="fp64",
        )
        nearest_result, implementation = nearest_spatial_index(
            None,
            None,
            tree_query_nearest=lambda *args, **kwargs: None,
            return_all=False,
            return_distance=True,
            tree_owned=source_owned,
            query_owned=query_centers,
            return_device=True,
        )
        if nearest_result is None or "gpu" not in implementation:
            return None
        (d_query_rows, _d_tree_rows), d_center_distance = nearest_result
        if not all(
            hasattr(values, "__cuda_array_interface__")
            for values in (d_query_rows, d_center_distance)
        ):
            return None

        d_bounds = cp.asarray(
            compute_geometry_bounds_device(
                query_owned,
                preserve_indexed_view=True,
            ),
            dtype=cp.float64,
        ).reshape(query_row_count, 4)
        d_diagonal = cp.hypot(
            d_bounds[:, 2] - d_bounds[:, 0],
            d_bounds[:, 3] - d_bounds[:, 1],
        )
        d_nearest = cp.full(query_row_count, cp.inf, dtype=cp.float64)
        d_nearest[cp.asarray(d_query_rows, dtype=cp.int64)] = cp.asarray(
            d_center_distance,
            dtype=cp.float64,
        )
        d_finite = cp.isfinite(d_nearest) & cp.isfinite(d_diagonal)
        d_definite_match = d_finite & (
            d_nearest <= buffer_inradius - d_diagonal
        )
        d_ambiguous = (
            d_finite
            & (d_nearest > buffer_inradius - d_diagonal)
            & (d_nearest <= radius + d_diagonal)
        )

        ambiguous_queries = device_mask_owned_capacity(
            query_owned,
            d_ambiguous,
            preserve_row_bounds=False,
        )
        d_ambiguous_bounds = cp.where(
            d_ambiguous[:, None],
            d_bounds,
            cp.nan,
        )
        exact_matches, exact_execution = self.query_left_semijoin(
            ambiguous_queries,
            predicate="intersects",
            query_row_count=query_row_count,
            query_token=query_token,
            precomputed_query_bounds=d_ambiguous_bounds,
        )
        if (
            exact_matches is None
            or exact_execution is None
            or exact_execution.selected is not ExecutionMode.GPU
        ):
            return None
        if not hasattr(exact_matches, "source_mask"):
            return None
        d_exact_match = exact_matches.source_mask()
        d_result = d_definite_match | (d_ambiguous & d_exact_match)

        from vibespatial.runtime.provenance import record_rewrite_event

        record_rewrite_event(
            rule_name="R2_query_any_buffer_intersects_to_bounded_refine",
            surface="vibespatial.api.SpatialIndex.query_any",
            original_operation="query_any(buffer, intersects)",
            rewritten_operation="nearest bound + ambiguous buffered intersects",
            reason=(
                "point-buffer existence is certified by centroid distance bounds "
                "and exact bounded buffer refinement"
            ),
            detail=(
                f"buffer_distance={radius}, buffer_inradius={buffer_inradius}, "
                f"quad_segs={quad_segs}, query_rows={query_row_count}"
            ),
        )
        record_dispatch_event(
            surface="vibespatial.api.SpatialIndex.query_any",
            operation="query_any",
            implementation="owned_gpu_buffer_point_existential",
            reason=(
                "input-sized nearest bounds plus exact threshold-ambiguous "
                "buffer semijoin refinement"
            ),
            detail=(
                f"predicate={predicate!r}, query_rows={query_row_count}, "
                f"tree_rows={source_owned.row_count}"
            ),
            requested=get_requested_mode(),
            selected=ExecutionMode.GPU,
        )
        from vibespatial.api._native_rowset import NativeDeviceSelection

        selection = NativeDeviceSelection.from_mask(
            d_result,
            source_token=query_token,
            source_row_count=query_row_count,
        )
        # Driver kernels and CuPy expressions above consume raw pointers
        # asynchronously.  Keep every local producer alive until the active
        # stream passes the selection scatter; otherwise an immediate public
        # consumer may recycle an input allocation before its last kernel use.
        from vibespatial.cuda._runtime import get_cuda_completion_retainer

        get_cuda_completion_retainer().defer(
            cp.cuda.get_current_stream(),
            (
                self,
                values,
                self._native_spatial_index,
                getattr(self._native_spatial_index, "geometry", None),
                source_owned,
                _source_index,
                query_owned,
                query_centers,
                d_query_rows,
                d_center_distance,
                d_bounds,
                d_diagonal,
                d_nearest,
                d_finite,
                d_definite_match,
                d_ambiguous,
                ambiguous_queries,
                d_ambiguous_bounds,
                exact_matches,
                d_exact_match,
            ),
            lambda _owners: None,
        )
        return self._query_any_public_series(
            d_result,
            output_index=output_index,
            selection=selection,
        )

    def _query_any_native(
        self,
        geometry,
        *,
        predicate,
        distance,
        query_row_count: int,
        output_index,
    ):
        if (
            get_requested_mode() is ExecutionMode.CPU
            or not self._supports_owned_query_input(geometry)
        ):
            return None
        from vibespatial.api.geo_base import _native_state_for_owner

        query_state = _native_state_for_owner(geometry)
        query_token = None if query_state is None else query_state.lineage_token
        rewritten = self._query_any_buffer_point_native(
            geometry,
            predicate=predicate,
            distance=distance,
            query_row_count=query_row_count,
            output_index=output_index,
            query_token=query_token,
        )
        if rewritten is not None:
            return rewritten

        query_input = self._owned_query_input(geometry)
        matched, execution = self.query_left_semijoin(
            query_input,
            predicate=predicate,
            distance=distance,
            query_token=query_token,
            query_row_count=query_row_count,
        )
        if execution is None or execution.selected is not ExecutionMode.GPU:
            return None
        from vibespatial.api._native_rowset import NativeDeviceSelection, NativeRowSet

        selection = matched if isinstance(matched, NativeDeviceSelection) else None
        rowset = matched if isinstance(matched, NativeRowSet) else None
        if hasattr(matched, "source_mask"):
            mask = matched.source_mask()
        else:
            import cupy as cp

            mask = cp.zeros(query_row_count, dtype=cp.bool_)
            mask[cp.asarray(matched.positions, dtype=cp.int64)] = True
        record_dispatch_event(
            surface="vibespatial.api.SpatialIndex.query_any",
            operation="query_any",
            implementation=execution.implementation,
            reason="native spatial relation reduced directly to an input-sized mask",
            detail=(
                f"predicate={predicate!r}, query_rows={query_row_count}, "
                f"tree_rows={self.size}"
            ),
            requested=execution.requested,
            selected=execution.selected,
        )
        return self._query_any_public_series(
            mask,
            output_index=output_index,
            rowset=rowset,
            selection=selection,
        )

    def query_pair_aggregate(
        self,
        other,
        geometry,
        *,
        predicate=None,
        distance=None,
    ) -> pd.DataFrame:
        """Aggregate common query matches for two aligned spatial indexes.

        ``self`` and ``other`` must index the same number of position-aligned
        geometries. For every indexed row, the result contains the number of
        matches from each index and the number of query geometries matched by
        both indexed geometries at that row position. Query-row duplicates and
        overlapping query geometries retain their ordinary spatial-join
        multiplicity.

        The eager pandas result has a ``RangeIndex`` aligned to indexed row
        positions and columns ``left_count``, ``right_count``, and
        ``shared_count``. Device-backed inputs consume both native relations
        without exporting pair arrays; computed columns remain device-backed
        until an explicit public scalar or array export.
        """
        if not isinstance(other, SpatialIndex):
            raise TypeError("other must be a SpatialIndex")
        if other.size != self.size:
            raise ValueError("query pair aggregate indexes must have equal size")

        query_row_count, scalar = self._query_cardinality(geometry)
        output_index = pd.RangeIndex(self.size)
        native, fallback_state = self._query_pair_aggregate_native(
            other,
            geometry,
            predicate=predicate,
            distance=distance,
            query_row_count=query_row_count,
            output_index=output_index,
        )
        if native is not None:
            return native

        record_fallback_event(
            surface="vibespatial.api.SpatialIndex.query_pair_aggregate",
            requested=get_requested_mode(),
            selected=ExecutionMode.CPU,
            reason=(
                "query pair aggregate requires aligned owned geometry and "
                "device relation consumption"
            ),
            detail=(
                f"predicate={predicate!r}, query_rows={query_row_count}, "
                f"tree_rows={self.size}"
            ),
            pipeline="spatial_query_pair_aggregate",
            d2h_transfer=fallback_state is not None,
        )
        return self._query_pair_aggregate_host(
            other,
            geometry,
            predicate=predicate,
            distance=distance,
            query_row_count=query_row_count,
            scalar=scalar,
            output_index=output_index,
            fallback_state=fallback_state,
        )

    def _query_pair_aggregate_native(
        self,
        other,
        geometry,
        *,
        predicate,
        distance,
        query_row_count: int,
        output_index,
    ):
        if get_requested_mode() is ExecutionMode.CPU:
            return None, None
        if not (
            self._supports_owned_query_input(geometry)
            and other._supports_owned_query_input(geometry)
        ):
            return None, None
        if predicate is None or predicate == "dwithin" or distance is not None:
            return None, None

        from vibespatial.api._native_expression import NativeExpression
        from vibespatial.api._native_public_arrays import NativeNumericExpressionArray
        query_input = self._owned_query_input(geometry)
        if not isinstance(query_input, OwnedGeometryArray):
            return None, None
        other_tree_owned, _other_flat_index = other._owned_flat_sindex()

        # Let the native reducer compute only the bounds required by its
        # selected physical shape.  Parent-aware multipart lowering works on
        # Polygon-part bounds and would otherwise pay for unused parent bounds.
        query_bounds = None
        left_native_index = self._native_spatial_index_for_query()
        right_native_index = other._native_spatial_index_for_query()
        pair_expressions, left_execution = (
            left_native_index.query_right_pair_match_count_expressions(
                query_input,
                right_native_index,
                predicate=predicate,
                query_row_count=query_row_count,
                return_metadata=True,
                precomputed_query_bounds=query_bounds,
            )
        )
        if (
            pair_expressions is None
            or left_execution is None
            or left_execution.selected is not ExecutionMode.GPU
        ):
            return None, None
        if len(pair_expressions) == 3:
            left_expression, right_expression, shared_expression = pair_expressions
            right_execution = left_execution
        else:
            left_expression, shared_expression = pair_expressions
            if query_bounds is None:
                state = query_input._ensure_device_state(
                    preserve_indexed_view=True,
                )
                query_bounds = state.row_bounds
                if query_bounds is None:
                    from vibespatial.kernels.core.geometry_analysis import (
                        compute_geometry_bounds_device,
                    )

                    query_bounds = compute_geometry_bounds_device(
                        query_input,
                        preserve_indexed_view=True,
                    )
            right_expression, right_execution = (
                right_native_index.query_right_match_count_expression(
                    query_input,
                    predicate=predicate,
                    distance=None,
                    query_row_count=query_row_count,
                    return_metadata=True,
                    precomputed_query_bounds=query_bounds,
                )
            )
            if (
                right_expression is None
                or right_execution is None
                or right_execution.selected is not ExecutionMode.GPU
            ):
                return None, None

        values_by_name = {
            "left_count": left_expression.values,
            "right_count": right_expression.values,
            "shared_count": shared_expression.values,
        }
        columns = {}
        for name, values in values_by_name.items():
            expression = NativeExpression(
                operation=f"spatial_query_pair_aggregate.{name}",
                values=values,
                source_token=None,
                source_row_count=self.size,
                dtype=str(getattr(values, "dtype", "")) or None,
                precision="relation-co-membership",
                null_policy="nan-false",
            )
            columns[name] = pd.Series(
                NativeNumericExpressionArray(
                    expression,
                    export_surface=(
                        "vibespatial.api.SpatialIndex.query_pair_aggregate"
                    ),
                    export_operation="query_pair_aggregate_column_to_public_array",
                ),
                index=output_index,
                name=name,
            )
        result = pd.DataFrame(columns, index=output_index)
        record_dispatch_event(
            surface="vibespatial.api.SpatialIndex.query_pair_aggregate",
            operation="query_pair_aggregate",
            implementation=(
                f"{left_execution.implementation}+{right_execution.implementation}"
            ),
            reason=(
                f"left/shared: {left_execution.reason}; "
                f"right: {right_execution.reason}"
            ),
            detail=(
                f"predicate={predicate!r}, query_rows={query_row_count}, "
                f"tree_rows={self.size}"
            ),
            requested=left_execution.requested,
            selected=ExecutionMode.GPU,
        )
        return result, None

    @staticmethod
    def _query_pair_host_pairs(index, geometry, *, predicate, distance, scalar, relation):
        pairs = (
            index.query(
                geometry,
                predicate=predicate,
                distance=distance,
                sort=False,
            )
            if relation is None
            else index._public_relation_indices_to_host(relation, scalar=scalar)
        )
        if pairs.ndim == 1:
            return (
                np.zeros(len(pairs), dtype=np.int64),
                np.asarray(pairs, dtype=np.int64),
            )
        return (
            np.asarray(pairs[0], dtype=np.int64),
            np.asarray(pairs[1], dtype=np.int64),
        )

    @staticmethod
    def _query_pair_host_structured_keys(query_rows, tree_rows):
        keys = np.empty(
            len(query_rows),
            dtype=[("tree", np.int64), ("query", np.int64)],
        )
        keys["tree"] = tree_rows
        keys["query"] = query_rows
        return keys

    def _query_pair_aggregate_host(
        self,
        other,
        geometry,
        *,
        predicate,
        distance,
        query_row_count: int,
        scalar: bool,
        output_index,
        fallback_state,
    ) -> pd.DataFrame:
        state = fallback_state or {}
        left_relation = state.get("left_relation")
        left_counts = state.get("left_counts")
        shared_counts = state.get("shared_counts")
        if left_counts is not None and shared_counts is not None:
            from vibespatial.cuda._runtime import get_cuda_runtime

            runtime = get_cuda_runtime()
            left_counts = np.asarray(
                runtime.copy_device_to_host(
                    left_counts,
                    reason="query pair aggregate CPU fallback left counts",
                ),
                dtype=np.int64,
            )
            shared_counts = np.asarray(
                runtime.copy_device_to_host(
                    shared_counts,
                    reason="query pair aggregate CPU fallback shared counts",
                ),
                dtype=np.int64,
            )
            left_query_rows = None
            left_tree_rows = None
        else:
            left_query_rows, left_tree_rows = self._query_pair_host_pairs(
                self,
                geometry,
                predicate=predicate,
                distance=distance,
                scalar=scalar,
                relation=left_relation,
            )
            left_counts = np.bincount(
                left_tree_rows,
                minlength=self.size,
            ).astype(np.int64, copy=False)

        right_query_rows, right_tree_rows = self._query_pair_host_pairs(
            other,
            geometry,
            predicate=predicate,
            distance=distance,
            scalar=scalar,
            relation=state.get("right_relation"),
        )
        right_counts = np.bincount(
            right_tree_rows,
            minlength=self.size,
        ).astype(np.int64, copy=False)
        if shared_counts is None:
            left_structured = self._query_pair_host_structured_keys(
                left_query_rows,
                left_tree_rows,
            )
            right_structured = self._query_pair_host_structured_keys(
                right_query_rows,
                right_tree_rows,
            )
            shared = np.intersect1d(left_structured, right_structured)
            shared_counts = np.bincount(
                shared["tree"],
                minlength=self.size,
            ).astype(np.int64, copy=False)
        return pd.DataFrame(
            {
                "left_count": left_counts,
                "right_count": right_counts,
                "shared_count": shared_counts,
            },
            index=output_index,
        )

    @staticmethod
    def _normalize_query_aggregations(aggregations):
        if not isinstance(aggregations, Mapping) or not aggregations:
            raise ValueError("aggregations must be a non-empty mapping")
        normalized = {}
        for name, specification in aggregations.items():
            if not isinstance(name, str):
                raise TypeError("query aggregate output column names must be strings")
            if specification == "size":
                normalized[name] = (None, "size")
                continue
            if not (
                isinstance(specification, tuple)
                and len(specification) == 2
                and specification[1] == "sum"
            ):
                raise ValueError(
                    "query aggregate specifications must be 'size' or "
                    "(values, 'sum')"
                )
            values = specification[0]
            shape = getattr(values, "shape", None)
            if shape is not None and len(shape) != 1:
                raise ValueError("query aggregate values must be one-dimensional")
            normalized[name] = (values, "sum")
        return normalized

    @staticmethod
    def _query_aggregate_output_index(geometry, *, query_row_count: int, scalar: bool):
        if not scalar:
            index = getattr(geometry, "index", None)
            if index is not None and len(index) == query_row_count:
                return index.copy()
        return pd.RangeIndex(query_row_count)

    def _validate_query_aggregate_values(self, aggregations) -> None:
        for values, reducer in aggregations.values():
            if reducer == "size":
                continue
            try:
                value_count = len(values)
            except TypeError as exc:
                raise ValueError(
                    "query aggregate values must be one-dimensional and aligned "
                    "with indexed tree geometries"
                ) from exc
            if value_count != self.size:
                raise ValueError(
                    "query aggregate values must align with indexed tree geometries"
                )
            dtype = getattr(values, "dtype", None)
            if dtype is None:
                dtype = np.asarray(values).dtype
            if not pd.api.types.is_numeric_dtype(dtype):
                raise TypeError("query aggregate sum values must be numeric")

    @staticmethod
    def _query_aggregate_value_is_device(value) -> bool:
        from vibespatial.api.geo_base import _native_expression_from_public_series

        expression = _native_expression_from_public_series(value)
        if expression is not None:
            return expression.is_device
        return hasattr(value, "__cuda_array_interface__")

    def _query_aggregate_native(
        self,
        geometry,
        aggregations,
        *,
        predicate,
        distance,
        query_row_count: int,
        output_index,
    ):
        if not self._supports_owned_query_input(geometry):
            return None, None

        from vibespatial.api._native_expression import NativeExpression
        from vibespatial.api._native_grouped import (
            NativeGroupedAttributeReduction,
            NativeGroupedReduction,
        )
        from vibespatial.api._native_public_arrays import (
            NativeNumericExpressionArray,
        )
        from vibespatial.api.geo_base import (
            _native_expression_from_public_series,
            _native_state_for_owner,
        )

        expressions: dict[str, NativeExpression] = {}
        source_tokens: set[str] = set()
        for name, (values, reducer) in aggregations.items():
            if reducer == "size":
                continue
            expression = _native_expression_from_public_series(values)
            if (
                expression is None
                or not expression.is_device
                or len(expression) != self.size
            ):
                return None, None
            expressions[name] = expression
            if expression.source_token is not None:
                source_tokens.add(expression.source_token)
        if len(source_tokens) > 1:
            return None, None
        source_token = next(iter(source_tokens), None)

        query_state = _native_state_for_owner(geometry)
        query_token = None if query_state is None else query_state.lineage_token
        query_input = self._owned_query_input(geometry)
        relation = None
        count_expression = None
        count_reduction_is_direct = False
        if all(reducer == "size" for _values, reducer in aggregations.values()):
            native_index = self._native_spatial_index_for_query(
                source_token=source_token,
            )
            count_expression, execution = (
                native_index.query_left_match_count_expression(
                    query_input,
                    predicate=predicate,
                    distance=distance,
                    query_token=query_token,
                    query_row_count=query_row_count,
                    return_metadata=True,
                )
            )
            if (
                count_expression is None
                or execution is None
                or execution.selected is not ExecutionMode.GPU
            ):
                return None, None
            count_reduction_is_direct = (
                execution.implementation == "owned_gpu_spatial_match_count"
            )
        else:
            relation, execution = self.query_relation(
                query_input,
                predicate=predicate,
                distance=distance,
                source_token=source_token,
                query_token=query_token,
                query_row_count=query_row_count,
                return_device=True,
            )
            if relation is None:
                return None, None
            if execution.selected is not ExecutionMode.GPU:
                return None, relation

        reductions = {}
        for name, (_values, reducer) in aggregations.items():
            if reducer == "size":
                expression = (
                    count_expression
                    if count_expression is not None
                    else relation.left_match_count_expression(
                        source_row_count=query_row_count,
                    )
                )
                reductions[name] = NativeGroupedReduction(
                    values=expression.values,
                    reducer="count",
                    group_count=query_row_count,
                )
            else:
                reductions[name] = relation.left_reduce_right_numeric(
                    expressions[name].values,
                    reducer,
                    left_row_count=query_row_count,
                )

        reduced = NativeGroupedAttributeReduction(
            columns=reductions,
            group_count=query_row_count,
        )
        result_columns = {}
        for name, reduction in reduced.columns.items():
            values = reduction.values
            expression = NativeExpression(
                operation=f"spatial_query_aggregate.{reduction.reducer}",
                values=values,
                source_token=query_token,
                source_row_count=query_row_count,
                dtype=str(getattr(values, "dtype", "")) or None,
                precision="grouped-reduction",
                null_policy="nan-false",
            )
            result_columns[name] = pd.Series(
                NativeNumericExpressionArray(
                    expression,
                    export_surface=(
                        "vibespatial.api.SpatialIndex.query_aggregate"
                    ),
                    export_operation="query_aggregate_column_to_public_array",
                ),
                index=output_index,
                name=name,
            )
        result = pd.DataFrame(result_columns, index=output_index)
        record_dispatch_event(
            surface="vibespatial.api.SpatialIndex.query_aggregate",
            operation="query_aggregate",
            implementation=execution.implementation,
            reason=(
                "NativeSpatialIndex reduced matches directly into input-sized "
                "count columns before public result export"
                if count_reduction_is_direct
                else "NativeRelation pairs were consumed by selection-aware "
                "grouped reductions before public result export"
            ),
            detail=(
                f"predicate={predicate!r}, query_rows={query_row_count}, "
                f"outputs={len(reductions)}"
            ),
            requested=execution.requested,
            selected=execution.selected,
        )
        return result, None

    def _query_aggregate_host(
        self,
        geometry,
        aggregations,
        *,
        predicate,
        distance,
        query_row_count: int,
        output_index,
        scalar: bool,
        relation=None,
    ) -> pd.DataFrame:
        if relation is None:
            pairs = self.query(
                geometry,
                predicate=predicate,
                distance=distance,
                sort=False,
            )
        else:
            pairs = self._public_relation_indices_to_host(
                relation,
                scalar=scalar,
            )
        if pairs.ndim == 1:
            query_rows = np.zeros(len(pairs), dtype=np.int64)
            tree_rows = np.asarray(pairs, dtype=np.int64)
        else:
            query_rows = np.asarray(pairs[0], dtype=np.int64)
            tree_rows = np.asarray(pairs[1], dtype=np.int64)

        columns = {}
        for name, (values, reducer) in aggregations.items():
            if reducer == "size":
                columns[name] = np.bincount(
                    query_rows,
                    minlength=query_row_count,
                ).astype(np.int64, copy=False)
                continue
            source = np.asarray(values)
            if source.ndim != 1 or len(source) != self.size:
                raise ValueError(
                    "query aggregate values must align with indexed tree geometries"
                )
            if not (
                np.issubdtype(source.dtype, np.number)
                or np.issubdtype(source.dtype, np.bool_)
            ):
                raise TypeError("query aggregate sum values must be numeric")
            output_dtype = (
                np.dtype(np.int64) if source.dtype.kind == "b" else source.dtype
            )
            reduced = np.zeros(query_row_count, dtype=output_dtype)
            np.add.at(
                reduced,
                query_rows,
                source[tree_rows].astype(output_dtype, copy=False),
            )
            columns[name] = reduced
        return pd.DataFrame(columns, index=output_index)

    def _owned_flat_sindex(self):
        if self._geometry_array is not None and hasattr(self._geometry_array, "owned_flat_sindex"):
            return self._geometry_array.owned_flat_sindex()
        return build_owned_spatial_index(np.asarray(self.geometries, dtype=object))

    def _native_spatial_index_for_query(self, *, source_token: str | None = None):
        """Return cached ``NativeSpatialIndex`` state for this public sindex.

        Physical shape: reusable spatial-index execution state.  Native input
        carrier is the cached owned-backed ``FlatSpatialIndex``; native output
        carrier is ``NativeSpatialIndex`` with source-lineage validation by
        token.  Public callers still export through ``query()``.
        """
        _tree_owned, flat_index = self._owned_flat_sindex()
        flat_index_id = id(flat_index)
        if (
            self._native_spatial_index is not None
            and self._native_spatial_index_source_token == source_token
            and self._native_spatial_index_flat_index_id == flat_index_id
        ):
            return self._native_spatial_index
        native_index = flat_index.to_native_spatial_index(source_token=source_token)
        self._native_spatial_index = native_index
        self._native_spatial_index_source_token = source_token
        self._native_spatial_index_flat_index_id = flat_index_id
        return native_index

    def query_relation(
        self,
        geometry,
        *,
        predicate=None,
        sort=False,
        distance=None,
        source_token: str | None = None,
        query_token: str | None = None,
        query_row_count: int | None = None,
        return_device: bool = True,
        tree_shapely: np.ndarray | None = None,
        query_shapely: np.ndarray | None = None,
        precomputed_query_bounds: np.ndarray | None = None,
    ):
        """Query this spatial index as private native relation row flow.

        Physical shape: candidate/predicate pair generation over cached
        ``NativeSpatialIndex`` state.  Native input carriers are
        ``NativeSpatialIndex`` plus owned/query geometry or ``NativeFrameState``;
        the native output carrier is ``NativeRelation``.
        """
        native_index = self._native_spatial_index_for_query(
            source_token=source_token,
        )
        if query_row_count is None:
            query_row_count, _scalar = self._query_cardinality(geometry)
        return native_index.query_relation(
            geometry,
            predicate=predicate,
            sort=sort,
            distance=distance,
            query_token=query_token,
            query_row_count=query_row_count,
            return_device=return_device,
            return_metadata=True,
            tree_shapely=tree_shapely,
            query_shapely=query_shapely,
            precomputed_query_bounds=precomputed_query_bounds,
        )

    def query_left_semijoin(
        self,
        geometry,
        *,
        predicate=None,
        distance=None,
        source_token: str | None = None,
        query_token: str | None = None,
        query_row_count: int | None = None,
        precomputed_query_bounds=None,
    ):
        """Query this index directly into private matched-left row flow."""
        native_index = self._native_spatial_index_for_query(
            source_token=source_token,
        )
        if query_row_count is None:
            query_row_count, _scalar = self._query_cardinality(geometry)
        return native_index.query_left_semijoin(
            geometry,
            predicate=predicate,
            distance=distance,
            query_token=query_token,
            query_row_count=query_row_count,
            return_metadata=True,
            precomputed_query_bounds=precomputed_query_bounds,
        )

    def query_left_antijoin(
        self,
        geometry,
        *,
        predicate=None,
        distance=None,
        source_token: str | None = None,
        query_token: str | None = None,
        query_row_count: int | None = None,
        precomputed_query_bounds=None,
    ):
        """Query this index directly into private unmatched-left row flow."""
        native_index = self._native_spatial_index_for_query(
            source_token=source_token,
        )
        if query_row_count is None:
            query_row_count, _scalar = self._query_cardinality(geometry)
        return native_index.query_left_antijoin(
            geometry,
            predicate=predicate,
            distance=distance,
            query_token=query_token,
            query_row_count=query_row_count,
            return_metadata=True,
            precomputed_query_bounds=precomputed_query_bounds,
        )

    def query_left_match_count_expression(
        self,
        geometry,
        *,
        predicate=None,
        distance=None,
        source_token: str | None = None,
        query_token: str | None = None,
        query_row_count: int | None = None,
        precomputed_query_bounds=None,
    ):
        """Query this index directly into private per-left match counts."""
        native_index = self._native_spatial_index_for_query(
            source_token=source_token,
        )
        if query_row_count is None:
            query_row_count, _scalar = self._query_cardinality(geometry)
        return native_index.query_left_match_count_expression(
            geometry,
            predicate=predicate,
            distance=distance,
            query_token=query_token,
            query_row_count=query_row_count,
            return_metadata=True,
            precomputed_query_bounds=precomputed_query_bounds,
        )

    def query_right_semijoin(
        self,
        geometry,
        *,
        predicate=None,
        distance=None,
        source_token: str | None = None,
        query_row_count: int | None = None,
        precomputed_query_bounds=None,
    ):
        """Query this index directly into private matched-index row flow."""
        native_index = self._native_spatial_index_for_query(
            source_token=source_token,
        )
        if query_row_count is None:
            query_row_count, _scalar = self._query_cardinality(geometry)
        return native_index.query_right_semijoin(
            geometry,
            predicate=predicate,
            distance=distance,
            query_row_count=query_row_count,
            return_metadata=True,
            precomputed_query_bounds=precomputed_query_bounds,
        )

    def query_right_match_count_expression(
        self,
        geometry,
        *,
        predicate=None,
        distance=None,
        source_token: str | None = None,
        query_row_count: int | None = None,
        precomputed_query_bounds=None,
        allow_relation_fallback: bool = True,
    ):
        """Query directly into private per-indexed-row match counts.

        Planning callers can set ``allow_relation_fallback=False`` to decline
        when the direct device reduction is unavailable.  This prevents an
        ostensibly cheap cardinality probe from allocating the full relation
        it is meant to admit or reject.
        """
        native_index = self._native_spatial_index_for_query(
            source_token=source_token,
        )
        if query_row_count is None:
            query_row_count, _scalar = self._query_cardinality(geometry)
        return native_index.query_right_match_count_expression(
            geometry,
            predicate=predicate,
            distance=distance,
            query_row_count=query_row_count,
            return_metadata=True,
            precomputed_query_bounds=precomputed_query_bounds,
            allow_relation_fallback=allow_relation_fallback,
        )

    def query_morton_span_upper_packet(
        self,
        geometry,
        *,
        source_token: str | None = None,
        query_row_count: int | None = None,
        precomputed_query_bounds=None,
    ):
        """Query directly into a fixed structural Morton-span packet."""
        native_index = self._native_spatial_index_for_query(
            source_token=source_token,
        )
        if query_row_count is None:
            query_row_count, _scalar = self._query_cardinality(geometry)
        return native_index.query_morton_span_upper_packet(
            geometry,
            query_row_count=query_row_count,
            return_metadata=True,
            precomputed_query_bounds=precomputed_query_bounds,
        )

    def _query_native_relation_for_public_output(
        self,
        raw_geometry,
        *,
        predicate,
        sort,
        distance,
        output_format,
        return_device: bool,
        raw_box_array_fast_path: bool,
        precomputed_query_bounds,
    ):
        if raw_geometry is None:
            return None
        if output_format not in {"indices", "sparse", "dense"}:
            return None
        if return_device:
            # Internal device callers should use query_relation() directly.
            # The public query() result shape is a NumPy compatibility export.
            return None
        query_input = (
            raw_geometry
            if raw_box_array_fast_path
            else self._owned_query_input(raw_geometry)
        )
        query_row_count, scalar = self._query_cardinality(raw_geometry)
        tree_shapely_arr, query_shapely_arr = self._cached_shapely_inputs(raw_geometry)
        relation, execution = self.query_relation(
            query_input,
            predicate=predicate,
            sort=sort,
            distance=distance,
            query_row_count=query_row_count,
            return_device=True,
            tree_shapely=tree_shapely_arr,
            query_shapely=query_shapely_arr,
            precomputed_query_bounds=precomputed_query_bounds,
        )
        indices = self._public_relation_indices_to_host(relation, scalar=scalar)
        relation_metadata = getattr(relation, "relation", relation)
        return (
            self._format_public_relation_output(
                indices,
                output_format=output_format,
                scalar=scalar,
                query_row_count=query_row_count,
                tree_row_count=relation_metadata.right_row_count,
            ),
            execution,
        )

    @staticmethod
    def _public_relation_indices_to_host(relation, *, scalar: bool):
        """Export native relation pairs at the public ``sindex.query`` boundary."""
        from vibespatial.api._native_relation import NativeRelationSelection
        from vibespatial.api._native_result_core import _host_array

        if isinstance(relation, NativeRelationSelection):
            pair_rows = relation.selection.compact_rowset(
                surface=(
                    "vibespatial.api.sindex.SpatialIndex."
                    "_public_relation_indices_to_host"
                ),
                strict_disallowed=False,
            )
            relation = relation.relation.filter_pairs(pair_rows)

        right = _host_array(
            relation.right_indices,
            dtype=np.intp,
            strict_disallowed=False,
            surface="vibespatial.api.sindex.SpatialIndex._public_relation_indices_to_host",
            operation="sindex_query_relation_indices_to_host",
            reason="public sindex.query output needs NumPy index arrays",
            detail="side=right",
        )
        if scalar:
            return right
        left = _host_array(
            relation.left_indices,
            dtype=np.intp,
            strict_disallowed=False,
            surface="vibespatial.api.sindex.SpatialIndex._public_relation_indices_to_host",
            operation="sindex_query_relation_indices_to_host",
            reason="public sindex.query output needs NumPy index arrays",
            detail="side=left",
        )
        return np.vstack((left, right))

    @staticmethod
    def _format_public_relation_output(
        indices,
        *,
        output_format: str,
        scalar: bool,
        query_row_count: int,
        tree_row_count: int,
    ):
        if output_format == "indices":
            return indices
        if output_format == "sparse":
            scipy = compat.import_optional_dependency("scipy")
            if scalar:
                return scipy.sparse.coo_array(
                    (np.ones(len(indices), dtype=np.bool_), indices.reshape(1, -1)),
                    shape=(tree_row_count,),
                    dtype=np.bool_,
                )
            return scipy.sparse.coo_array(
                (np.ones(len(indices[0]), dtype=np.bool_), indices[::-1]),
                shape=(tree_row_count, query_row_count),
                dtype=np.bool_,
            )
        if output_format == "dense":
            if scalar:
                dense = np.zeros(tree_row_count, dtype=bool)
                dense[indices] = True
                return dense
            dense = np.zeros((tree_row_count, query_row_count), dtype=bool)
            tree, other = indices[::-1]
            dense[tree, other] = True
            return dense
        return None

    def _query_owned_public(
        self,
        raw_geometry,
        *,
        predicate,
        sort,
        distance,
        output_format,
        return_device,
        raw_box_array_fast_path,
        precomputed_query_bounds,
    ):
        tree_owned, flat_index = self._owned_flat_sindex()
        query_input = raw_geometry if raw_box_array_fast_path else self._owned_query_input(raw_geometry)
        # Pass already-materialized Shapely arrays to avoid redundant
        # to_shapely() in predicate refinement.  Only use arrays that
        # are ALREADY cached — never trigger eager materialization here.
        tree_shapely_arr, query_shapely_arr = self._cached_shapely_inputs(raw_geometry)
        indices, execution = query_spatial_index(
            tree_owned,
            flat_index,
            query_input,
            predicate=predicate,
            sort=sort,
            distance=distance,
            output_format=output_format,
            return_metadata=True,
            return_device=return_device,
            tree_shapely=tree_shapely_arr,
            query_shapely=query_shapely_arr,
            precomputed_query_bounds=precomputed_query_bounds,
        )
        record_dispatch_event(
            surface="geopandas.sindex.query",
            operation="query",
            implementation=execution.implementation,
            reason=execution.reason,
            detail=f"predicate={predicate!r}, output_format={output_format!r}",
            requested=execution.requested,
            selected=execution.selected,
        )
        return indices

    def _cached_shapely_inputs(self, raw_geometry):
        tree_shapely_arr = None
        if self.geometries is not None:
            tree_shapely_arr = np.asarray(self.geometries, dtype=object)
        elif (
            self._geometry_array is not None
            and hasattr(self._geometry_array, "_shapely_cache")
            and self._geometry_array._shapely_cache is not None
        ):
            tree_shapely_arr = self._geometry_array._shapely_cache
        query_shapely_arr = None
        if isinstance(raw_geometry, geoseries.GeoSeries):
            ga = raw_geometry.values
            if hasattr(ga, "_shapely_cache") and ga._shapely_cache is not None:
                query_shapely_arr = ga._shapely_cache
        elif isinstance(raw_geometry, array.GeometryArray):
            if (
                hasattr(raw_geometry, "_shapely_cache")
                and raw_geometry._shapely_cache is not None
            ):
                query_shapely_arr = raw_geometry._shapely_cache
        return tree_shapely_arr, query_shapely_arr

    @staticmethod
    def _query_cardinality(geometry) -> tuple[int, bool]:
        if isinstance(geometry, BaseGeometry):
            return 1, True
        if isinstance(geometry, geoseries.GeoSeries):
            return len(geometry), False
        if isinstance(geometry, array.GeometryArray):
            return len(geometry), False
        if isinstance(geometry, OwnedGeometryArray):
            return int(geometry.row_count), False
        row_count = getattr(geometry, "row_count", None)
        if row_count is not None:
            return int(row_count), False
        owned = getattr(geometry, "owned", None)
        if owned is not None:
            return int(owned.row_count), False
        if "DeviceGeometryArray" in type(geometry).__name__ and hasattr(geometry, "to_owned"):
            return int(geometry.to_owned().row_count), False
        try:
            return len(geometry), False
        except TypeError:
            return 1, True

    def _supports_owned_tree_input(self) -> bool:
        if self._geometry_array is not None and hasattr(self._geometry_array, "supports_owned_spatial_input"):
            return self._geometry_array.supports_owned_spatial_input()
        return supports_owned_spatial_input(self.geometries)

    def _supports_owned_query_input(self, geometry) -> bool:
        if not self._supports_owned_tree_input():
            return False
        # Already-owned input is always supported — no conversion needed.
        if isinstance(geometry, OwnedGeometryArray):
            return True
        if isinstance(geometry, geoseries.GeoSeries):
            return geometry.values.supports_owned_spatial_input()
        if isinstance(geometry, array.GeometryArray):
            return geometry.supports_owned_spatial_input()
        return supports_owned_spatial_input(geometry)

    @staticmethod
    def _owned_query_input(geometry):
        # Already-owned — pass through without any H->D conversion.
        if isinstance(geometry, OwnedGeometryArray):
            return compact_indexed_spatial_input(geometry)
        if isinstance(geometry, geoseries.GeoSeries):
            values = geometry.values
            owned = values.to_owned() if hasattr(values, "to_owned") else (
                values._owned if values._owned is not None else values._data
            )
            if isinstance(owned, OwnedGeometryArray):
                compacted = compact_indexed_spatial_input(owned)
                if compacted is not owned and hasattr(values, "_owned"):
                    values._owned = compacted
                    if hasattr(values, "_owned_flat_sindex"):
                        values._owned_flat_sindex = None
                    if hasattr(values, "_owned_flat_sindex_cache"):
                        values._owned_flat_sindex_cache = None
                return compacted
            return owned
        if isinstance(geometry, array.GeometryArray):
            owned = geometry.to_owned()
            compacted = compact_indexed_spatial_input(owned)
            if compacted is not owned:
                geometry._owned = compacted
                geometry._owned_flat_sindex = None
            return compacted
        if isinstance(geometry, np.ndarray) and geometry.ndim >= 1:
            # Keep Shapely arrays as Shapely here. query_spatial_index() has
            # bounds-only regular-grid and point-tree fast paths that avoid
            # full owned conversion unless exact refinement needs it.
            return geometry
        # Scalar BaseGeometry or other types — keep as-is so
        # query_spatial_index() can detect scalar input correctly.
        return geometry

    @staticmethod
    def _as_geometry_array(geometry):
        """Convert geometry into a numpy array of Shapely geometries.

        Parameters
        ----------
        geometry
            An array-like of Shapely geometries, a GeoPandas GeoSeries/GeometryArray,
            shapely.geometry or list of shapely geometries.

        Returns
        -------
        np.ndarray
            A numpy array of Shapely geometries.
        """
        if isinstance(geometry, np.ndarray):
            return array.from_shapely(geometry)._data
        elif isinstance(geometry, geoseries.GeoSeries):
            return geometry.values._data
        elif isinstance(geometry, array.GeometryArray):
            return geometry._data
        elif isinstance(geometry, BaseGeometry):
            return geometry
        elif geometry is None:
            return None
        else:
            return np.asarray(geometry)

    def nearest(
        self,
        geometry,
        return_all=True,
        max_distance=None,
        return_distance=False,
        exclusive=False,
        k=1,
        _return_execution_mode=False,
    ):
        """
        Return the nearest geometry in the tree for each input geometry in
        ``geometry``.

        If multiple tree geometries have the same distance from an input geometry,
        multiple results will be returned for that input geometry by default.
        Specify ``return_all=False`` to only get a single nearest geometry
        (non-deterministic which nearest is returned).

        In the context of a spatial join, input geometries are the "left"
        geometries that determine the order of the results, and tree geometries
        are "right" geometries that are joined against the left geometries.
        If ``max_distance`` is not set, this will effectively be a left join
        because every geometry in ``geometry`` will have a nearest geometry in
        the tree. However, if ``max_distance`` is used, this becomes an
        inner join, since some geometries in ``geometry`` may not have a match
        in the tree.

        For performance reasons, it is highly recommended that you set
        the ``max_distance`` parameter.

        Parameters
        ----------
        geometry : {shapely.geometry, GeoSeries, GeometryArray, numpy.array of Shapely \
geometries}
            A single shapely geometry, one of the GeoPandas geometry iterables
            (GeoSeries, GeometryArray), or a numpy array of Shapely geometries to query
            against the spatial index.
        return_all : bool, default True
            If there are multiple equidistant or intersecting nearest
            geometries, return all those geometries instead of a single
            nearest geometry.
        max_distance : float, optional
            Maximum distance within which to query for nearest items in tree.
            Must be greater than 0. By default None, indicating no distance limit.
        return_distance : bool, optional
            If True, will return distances in addition to indexes. By default False
        exclusive : bool, optional
            if True, the nearest geometries that are equal to the input geometry
            will not be returned. By default False.  Requires Shapely >= 2.0.
        k : int, default 1
            Number of nearest tree geometries to return per input geometry.
            Values greater than one require the native device k-NN path and
            ``return_all=False``; exactly ``k`` rows are returned when enough
            tree geometries exist.

        Returns
        -------
        Indices or tuple of (indices, distances)
            Indices is an ndarray of shape (2,n) and distances (if present) an
            ndarray of shape (n).
            The first subarray of indices contains input geometry indices.
            The second subarray of indices contains tree geometry indices.

        Examples
        --------
        >>> from shapely.geometry import Point, box
        >>> s = geopandas.GeoSeries(geopandas.points_from_xy(range(10), range(10)))
        >>> s.head()
        0    POINT (0 0)
        1    POINT (1 1)
        2    POINT (2 2)
        3    POINT (3 3)
        4    POINT (4 4)
        dtype: geometry

        >>> s.sindex.nearest(Point(1, 1))
        array([[0],
               [1]])

        >>> s.sindex.nearest([box(4.9, 4.9, 5.1, 5.1)])
        array([[0],
               [5]])

        >>> s2 = geopandas.GeoSeries(geopandas.points_from_xy([7.6, 10], [7.6, 10]))
        >>> s2
        0    POINT (7.6 7.6)
        1    POINT (10 10)
        dtype: geometry

        >>> s.sindex.nearest(s2)
        array([[0, 1],
               [8, 9]])
        """
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError("k must be a positive integer")
        if k > 1 and return_all:
            raise ValueError("k > 1 requires return_all=False")
        raw_geometry = geometry

        # Route through the owned nearest engine when inputs support it.
        if self._supports_owned_query_input(raw_geometry):
            def _existing_owned(values):
                if values is None:
                    return None
                owned = getattr(values, "_owned", None)
                if owned is not None:
                    return owned
                if "DeviceGeometryArray" in type(values).__name__ and hasattr(values, "to_owned"):
                    return values.to_owned()
                return None

            query_values_obj = None
            if isinstance(raw_geometry, geoseries.GeoSeries):
                query_values_obj = raw_geometry.values
            elif isinstance(raw_geometry, array.GeometryArray | OwnedGeometryArray):
                query_values_obj = raw_geometry
            elif "DeviceGeometryArray" in type(raw_geometry).__name__:
                query_values_obj = raw_geometry
            tree_owned = _existing_owned(self._geometry_array)
            query_owned = _existing_owned(query_values_obj)

            # Defer STRtree construction: only build it lazily if the
            # nearest engine falls back to the STRtree path.  For GPU-
            # dispatched queries this avoids Shapely materialization.
            if tree_owned is not None and query_owned is not None:
                tree_geoms = None
            elif self.geometries is not None:
                tree_geoms = np.asarray(self.geometries, dtype=object)
            elif self._geometry_array is not None and hasattr(self._geometry_array, "_data"):
                tree_geoms = np.asarray(self._geometry_array._data, dtype=object)
            else:
                self._ensure_strtree()
                tree_geoms = np.asarray(self.geometries, dtype=object)

            def _lazy_tree_query_nearest(*args, **kwargs):
                self._ensure_strtree()
                return self._tree.query_nearest(*args, **kwargs)

            if query_owned is not None:
                query_input = None
            else:
                query_input = self._as_geometry_array(raw_geometry)
                if isinstance(query_input, BaseGeometry) or query_input is None:
                    query_input = [query_input] if query_input is not None else query_input

            if tree_owned is not None and query_owned is not None:
                relation, selected_mode = self.nearest_relation(
                    raw_geometry,
                    return_all=return_all,
                    max_distance=max_distance,
                    exclusive=exclusive,
                    k=k,
                    query_row_count=query_owned.row_count,
                )
                if relation is not None:
                    result = self._format_public_nearest_relation_output(
                        relation,
                        return_distance=return_distance,
                    )
                    record_dispatch_event(
                        surface="geopandas.sindex.nearest",
                        operation="nearest",
                        implementation="native_relation_export",
                        reason=(
                            "NativeRelation nearest pairs formatted as the "
                            "public sindex.nearest export"
                        ),
                        detail=(
                            f"max_distance={max_distance!r}, return_all={return_all}, "
                            f"exclusive={exclusive}, k={k}"
                        ),
                        selected=selected_mode,
                    )
                    self._record_public_spatial_export(
                        surface="geopandas.sindex.nearest",
                        operation="sindex_nearest",
                        target="sindex-nearest",
                        reason=(
                            "NativeRelation nearest pairs exported to public "
                            "sindex.nearest result"
                        ),
                        output=result,
                        detail=f"return_distance={return_distance}",
                    )
                    if _return_execution_mode:
                        return result, selected_mode
                    return result

            if k > 1:
                raise NotImplementedError(
                    "k > 1 nearest queries require owned device geometry inputs"
                )

            result, impl = nearest_spatial_index(
                tree_geoms,
                query_input,
                tree_query_nearest=_lazy_tree_query_nearest,
                return_all=return_all,
                max_distance=max_distance,
                return_distance=return_distance,
                exclusive=exclusive,
                tree_owned=tree_owned if query_owned is not None else None,
                query_owned=query_owned,
            )
            selected_mode = ExecutionMode.GPU if "gpu" in impl else ExecutionMode.CPU
            record_dispatch_event(
                surface="geopandas.sindex.nearest",
                operation="nearest",
                implementation=impl,
                reason=(
                    f"repo-owned nearest engine ({impl})"
                    if impl != "strtree_host"
                    else "Shapely STRtree nearest for unbounded nearest query"
                ),
                detail=f"max_distance={max_distance!r}, return_all={return_all}, exclusive={exclusive}",
                selected=selected_mode,
            )
            self._record_public_spatial_export(
                surface="geopandas.sindex.nearest",
                operation="sindex_nearest",
                target="sindex-nearest",
                reason="owned nearest query exported to public sindex.nearest result",
                output=result,
                detail=f"return_distance={return_distance}",
            )
            if _return_execution_mode:
                return result, selected_mode
            return result

        # Fallback: direct STRtree nearest
        self._ensure_strtree()
        geometry = self._as_geometry_array(raw_geometry)
        if isinstance(geometry, BaseGeometry) or geometry is None:
            geometry = [geometry]

        record_dispatch_event(
            surface="geopandas.sindex.nearest",
            operation="nearest",
            implementation="strtree_host",
            reason="Shapely STRtree nearest is the current first-class host implementation",
            detail=f"max_distance={max_distance!r}, return_all={return_all}, exclusive={exclusive}",
            selected=ExecutionMode.CPU,
        )

        result = self._tree.query_nearest(
            geometry,
            max_distance=max_distance,
            return_distance=return_distance,
            all_matches=return_all,
            exclusive=exclusive,
        )
        if _return_execution_mode:
            if return_distance:
                indices, distances = result
                return (indices, distances), ExecutionMode.CPU
            return result, ExecutionMode.CPU
        if return_distance:
            indices, distances = result
            return indices, distances
        return result

    def _has_native_geometry_backing(self) -> bool:
        values = self._geometry_array
        if values is None:
            return False
        if getattr(values, "_owned", None) is not None:
            return True
        if getattr(values, "owned", None) is not None:
            return True
        return "DeviceGeometryArray" in type(values).__name__

    @staticmethod
    def _public_spatial_export_shape(output) -> tuple[int | None, int | None]:
        if isinstance(output, tuple):
            row_count = None
            byte_count = 0
            for item in output:
                item_rows, item_bytes = SpatialIndex._public_spatial_export_shape(item)
                if row_count is None and item_rows is not None:
                    row_count = item_rows
                if item_bytes is not None:
                    byte_count += item_bytes
            return row_count, byte_count
        nbytes = getattr(output, "nbytes", None)
        if nbytes is not None:
            byte_count = int(nbytes)
        else:
            byte_count = None
        nnz = getattr(output, "nnz", None)
        if nnz is not None:
            return int(nnz), byte_count
        shape = getattr(output, "shape", None)
        if shape is not None:
            if len(shape) == 2 and shape[0] == 2:
                return int(shape[1]), byte_count
            size = getattr(output, "size", None)
            if size is not None:
                return int(size), byte_count
        try:
            return len(output), byte_count
        except TypeError:
            return None, byte_count

    def _record_public_spatial_export(
        self,
        *,
        surface: str,
        operation: str,
        target: str,
        reason: str,
        output,
        detail: str,
    ) -> None:
        if not self._has_native_geometry_backing():
            return
        row_count, byte_count = self._public_spatial_export_shape(output)
        record_native_export_boundary(NativeExportBoundary(
            surface=surface,
            operation=operation,
            target=target,
            reason=reason,
            row_count=row_count,
            byte_count=byte_count,
            detail=detail,
            d2h_transfer=True,
        ))

    @staticmethod
    def _format_public_nearest_relation_output(relation, *, return_distance: bool):
        from vibespatial.api._native_result_core import _host_array

        left = _host_array(
            relation.left_indices,
            dtype=np.intp,
            strict_disallowed=False,
            surface="vibespatial.api.sindex.SpatialIndex._format_public_nearest_relation_output",
            operation="nearest_relation_indices_to_host",
            reason="public nearest query output needs NumPy index arrays",
            detail="side=left",
        )
        right = _host_array(
            relation.right_indices,
            dtype=np.intp,
            strict_disallowed=False,
            surface="vibespatial.api.sindex.SpatialIndex._format_public_nearest_relation_output",
            operation="nearest_relation_indices_to_host",
            reason="public nearest query output needs NumPy index arrays",
            detail="side=right",
        )
        indices = np.vstack((left, right))
        if not return_distance:
            return indices
        distances = _host_array(
            relation.distances,
            dtype=np.float64,
            strict_disallowed=False,
            surface="vibespatial.api.sindex.SpatialIndex._format_public_nearest_relation_output",
            operation="nearest_relation_distances_to_host",
            reason="public nearest query output requested NumPy distances",
        )
        return indices, distances

    def nearest_relation(
        self,
        geometry,
        *,
        return_all=True,
        max_distance=None,
        exclusive=False,
        k=1,
        source_token: str | None = None,
        query_token: str | None = None,
        query_row_count: int | None = None,
    ):
        """Return nearest query output as private ``NativeRelation`` state.

        Physical shape: nearest candidate/refine relation production.  Native
        input carriers are owned query/tree geometry buffers; the native output
        carrier is ``NativeRelation`` with device pair arrays and fp64 device
        distances.  Public nearest callers still use ``nearest()`` and export
        NumPy arrays at the compatibility boundary.
        """
        if not self._supports_owned_query_input(geometry):
            return None, ExecutionMode.CPU

        def _existing_or_owned(values):
            if values is None:
                return None
            owned = getattr(values, "_owned", None)
            if owned is not None:
                return owned
            if isinstance(values, OwnedGeometryArray):
                return values
            if hasattr(values, "to_owned"):
                return values.to_owned()
            return None

        query_values_obj = None
        if isinstance(geometry, geoseries.GeoSeries):
            query_values_obj = geometry.values
        elif isinstance(geometry, array.GeometryArray | OwnedGeometryArray):
            query_values_obj = geometry
        elif "DeviceGeometryArray" in type(geometry).__name__:
            query_values_obj = geometry

        tree_owned = _existing_or_owned(self._geometry_array)
        query_owned = _existing_or_owned(query_values_obj)
        if tree_owned is None or query_owned is None:
            return None, ExecutionMode.CPU

        result, impl = nearest_spatial_index(
            None,
            None,
            tree_query_nearest=lambda *args, **kwargs: None,
            return_all=return_all,
            max_distance=max_distance,
            return_distance=True,
            exclusive=exclusive,
            k=k,
            tree_owned=tree_owned,
            query_owned=query_owned,
            return_device=True,
        )
        selected_mode = ExecutionMode.GPU if "gpu" in impl else ExecutionMode.CPU
        if result is None or selected_mode is not ExecutionMode.GPU:
            return None, selected_mode

        (left_indices, right_indices), distances = result
        if not (
            hasattr(left_indices, "__cuda_array_interface__")
            and hasattr(right_indices, "__cuda_array_interface__")
            and hasattr(distances, "__cuda_array_interface__")
        ):
            return None, selected_mode

        if query_row_count is None:
            query_row_count, _scalar = self._query_cardinality(geometry)

        record_dispatch_event(
            surface="geopandas.sindex.nearest",
            operation="nearest_relation",
            implementation=impl,
            reason=(
                "nearest query produced device NativeRelation pairs and "
                "distance expression input"
            ),
            detail=(
                f"max_distance={max_distance!r}, return_all={return_all}, "
                f"exclusive={exclusive}, k={k}"
            ),
            selected=selected_mode,
        )

        from vibespatial.api._native_relation import NativeRelation

        return (
            NativeRelation(
                left_indices=left_indices,
                right_indices=right_indices,
                left_token=query_token,
                right_token=source_token,
                predicate="nearest",
                distances=distances,
                left_row_count=query_row_count,
                right_row_count=tree_owned.row_count,
                sorted_by_left=True,
            ),
            selected_mode,
        )

    def intersection(self, coordinates):
        """Compatibility wrapper for rtree.index.Index.intersection,
        use ``query`` instead.

        Parameters
        ----------
        coordinates : sequence or array
            Sequence of the form (min_x, min_y, max_x, max_y)
            to query a rectangle or (x, y) to query a point.

        Examples
        --------
        >>> from shapely.geometry import Point, box
        >>> s = geopandas.GeoSeries(geopandas.points_from_xy(range(10), range(10)))
        >>> s
        0    POINT (0 0)
        1    POINT (1 1)
        2    POINT (2 2)
        3    POINT (3 3)
        4    POINT (4 4)
        5    POINT (5 5)
        6    POINT (6 6)
        7    POINT (7 7)
        8    POINT (8 8)
        9    POINT (9 9)
        dtype: geometry

        >>> s.sindex.intersection(box(1, 1, 3, 3).bounds)
        array([1, 2, 3])

        Alternatively, you can use ``query``:

        >>> s.sindex.query(box(1, 1, 3, 3))
        array([1, 2, 3])

        """
        # TODO: we should deprecate this
        # convert bounds to geometry
        # the old API uses tuples of bound, but Shapely uses geometries
        try:
            iter(coordinates)
        except TypeError as err:
            # likely not an iterable
            # this is a check that rtree does, we mimic it
            # to ensure a useful failure message
            raise TypeError(
                "Invalid coordinates, must be iterable in format "
                "(minx, miny, maxx, maxy) (for bounds) or (x, y) (for points). "
                f"Got `coordinates` = {coordinates}."
            ) from err

        # need to convert tuple of bounds to a geometry object
        if len(coordinates) == 4:
            self._ensure_strtree()
            indexes = self._tree.query(shapely.box(*coordinates))
        elif len(coordinates) == 2:
            self._ensure_strtree()
            indexes = self._tree.query(shapely.points(*coordinates))
        else:
            raise TypeError(
                "Invalid coordinates, must be iterable in format "
                "(minx, miny, maxx, maxy) (for bounds) or (x, y) (for points). "
                f"Got `coordinates` = {coordinates}."
            )

        return indexes

    @property
    def size(self):
        """Size of the spatial index.

        Number of leaves (input geometries) in the index.

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> s = geopandas.GeoSeries(geopandas.points_from_xy(range(10), range(10)))
        >>> s
        0    POINT (0 0)
        1    POINT (1 1)
        2    POINT (2 2)
        3    POINT (3 3)
        4    POINT (4 4)
        5    POINT (5 5)
        6    POINT (6 6)
        7    POINT (7 7)
        8    POINT (8 8)
        9    POINT (9 9)
        dtype: geometry

        >>> s.sindex.size
        10
        """
        if self._tree is None:
            return len(self._geometry_array)
        return len(self._tree)

    @property
    def is_empty(self):
        """Check if the spatial index is empty.

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> s = geopandas.GeoSeries(geopandas.points_from_xy(range(10), range(10)))
        >>> s
        0    POINT (0 0)
        1    POINT (1 1)
        2    POINT (2 2)
        3    POINT (3 3)
        4    POINT (4 4)
        5    POINT (5 5)
        6    POINT (6 6)
        7    POINT (7 7)
        8    POINT (8 8)
        9    POINT (9 9)
        dtype: geometry

        >>> s.sindex.is_empty
        False

        >>> s2 = geopandas.GeoSeries()
        >>> s2.sindex.is_empty
        True
        """
        if self._tree is None:
            return len(self._geometry_array) == 0
        return len(self._tree) == 0

    def __len__(self):
        if self._tree is None:
            return len(self._geometry_array)
        return len(self._tree)


register_device_spatial_index_factory(SpatialIndex._from_device_geometry_array)
