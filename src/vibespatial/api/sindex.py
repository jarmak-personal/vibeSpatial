from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry

from vibespatial.api import geometry_array as array
from vibespatial.api import geoseries
from vibespatial.geometry.api_registry import register_device_spatial_index_factory
from vibespatial.geometry.owned import OwnedGeometryArray
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.dispatch import record_dispatch_event
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
        return obj

    def _ensure_strtree(self):
        """Lazily build the STRtree when a fallback path needs it."""
        if self._tree is not None:
            return
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
        if predicate not in self.valid_query_predicates:
            if predicate == "dwithin":
                raise ValueError("predicate = 'dwithin' requires GEOS >= 3.10.0")

            raise ValueError(
                f"Got predicate='{predicate}'; "
                f"`predicate` must be one of {self.valid_query_predicates}"
            )

        # distance argument requirement of predicate `dwithin`
        # and only valid for predicate `dwithin`
        kwargs = {}
        if predicate == "dwithin":
            if distance is None:
                # the distance parameter is needed
                raise ValueError(
                    "'distance' parameter is required for 'dwithin' predicate"
                )
            # add distance to kwargs
            kwargs["distance"] = distance

        elif distance is not None:
            # distance parameter is invalid
            raise ValueError(
                "'distance' parameter is only supported in combination with "
                "'dwithin' predicate"
            )

        raw_geometry = geometry
        precomputed_query_bounds = None
        raw_box_array_fast_path = False
        if (
            predicate in (None, "intersects")
            and isinstance(raw_geometry, np.ndarray)
            and raw_geometry.ndim >= 1
            and self._supports_owned_tree_input()
        ):
            tree_owned, flat_index = self._owned_flat_sindex()
            if getattr(flat_index, "regular_grid", None) is not None:
                from vibespatial.spatial.query_box import _extract_box_query_bounds_shapely

                precomputed_query_bounds = _extract_box_query_bounds_shapely(raw_geometry)
                raw_box_array_fast_path = precomputed_query_bounds is not None
        if (
            raw_box_array_fast_path
            or (
                predicate in OWNED_QUERY_PREDICATES
                and self._supports_owned_query_input(raw_geometry)
            )
        ):
            tree_owned, flat_index = self._owned_flat_sindex()
            query_input = raw_geometry if raw_box_array_fast_path else self._owned_query_input(raw_geometry)
            # Pass already-materialized Shapely arrays to avoid redundant
            # to_shapely() in predicate refinement.  Only use arrays that
            # are ALREADY cached — never trigger eager materialization here.
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
                if hasattr(raw_geometry, "_shapely_cache") and raw_geometry._shapely_cache is not None:
                    query_shapely_arr = raw_geometry._shapely_cache
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

    def _owned_flat_sindex(self):
        if self._geometry_array is not None and hasattr(self._geometry_array, "owned_flat_sindex"):
            return self._geometry_array.owned_flat_sindex()
        return build_owned_spatial_index(np.asarray(self.geometries, dtype=object))

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
            return geometry
        if isinstance(geometry, geoseries.GeoSeries):
            values = geometry.values
            return values.to_owned() if hasattr(values, "to_owned") else (
                values._owned if values._owned is not None else values._data
            )
        if isinstance(geometry, array.GeometryArray):
            return geometry.to_owned()
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
