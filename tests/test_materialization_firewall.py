"""Tests for the materialization firewall.

Verifies that common GeoDataFrame/GeoSeries operations do NOT trigger Shapely
materialization when the geometry column is a DeviceGeometryArray, and that
operations requiring Shapely emit proper diagnostic events.
"""

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, MultiPoint, Point, Polygon

import vibespatial.api as geopandas
from vibespatial.api import GeoDataFrame, GeoSeries
from vibespatial.geometry.device_array import (
    DeviceGeometryArray,
    DeviceGeometryDtype,
)
from vibespatial.geometry.owned import (
    DiagnosticKind,
    OwnedGeometryArray,
    from_shapely_geometries,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def points_owned() -> OwnedGeometryArray:
    geoms = [Point(0, 0), Point(1, 2), Point(3, 4), None, Point(5, 6)]
    return from_shapely_geometries(geoms)


@pytest.fixture
def mixed_owned() -> OwnedGeometryArray:
    geoms = [
        Point(1, 2),
        LineString([(0, 0), (1, 1), (2, 0)]),
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        None,
        MultiPoint([(0, 0), (1, 1)]),
    ]
    return from_shapely_geometries(geoms)


@pytest.fixture
def dga_points(points_owned) -> DeviceGeometryArray:
    return DeviceGeometryArray._from_owned(points_owned)


@pytest.fixture
def dga_mixed(mixed_owned) -> DeviceGeometryArray:
    return DeviceGeometryArray._from_owned(mixed_owned)


@pytest.fixture
def gs_points(dga_points) -> GeoSeries:
    return GeoSeries(dga_points)


@pytest.fixture
def gs_mixed(dga_mixed) -> GeoSeries:
    return GeoSeries(dga_mixed)


@pytest.fixture
def gdf_points(dga_points) -> GeoDataFrame:
    return GeoDataFrame({"a": [10, 20, 30, 40, 50]}, geometry=dga_points)


@pytest.fixture
def gdf_mixed(dga_mixed) -> GeoDataFrame:
    return GeoDataFrame({"a": [10, 20, 30, 40, 50]}, geometry=dga_mixed)


def _materialization_count(dga: DeviceGeometryArray) -> int:
    """Count MATERIALIZATION diagnostic events on the underlying owned array."""
    return sum(
        1 for e in dga.diagnostics if e.kind == DiagnosticKind.MATERIALIZATION
    )


def _has_no_materialization(dga: DeviceGeometryArray) -> bool:
    """Return True if no Shapely materialization has occurred."""
    return dga._shapely_cache is None and _materialization_count(dga) == 0


# ---------------------------------------------------------------------------
# Category A: operations that MUST NOT trigger Shapely materialization
# ---------------------------------------------------------------------------


class TestNoMaterialization:
    """Operations satisfied directly from owned buffers."""

    def test_len(self, dga_points):
        assert len(dga_points) == 5
        assert _has_no_materialization(dga_points)

    def test_dtype(self, dga_points):
        assert dga_points.dtype == DeviceGeometryDtype()
        assert dga_points.dtype.name == "device_geometry"
        assert _has_no_materialization(dga_points)

    def test_nbytes(self, dga_points):
        nb = dga_points.nbytes
        assert nb > 0
        assert _has_no_materialization(dga_points)

    def test_isna(self, dga_points):
        na = dga_points.isna()
        assert na.dtype == bool
        assert na[3] is np.True_
        assert not na[0]
        assert _has_no_materialization(dga_points)

    def test_geom_type_points(self, dga_points):
        gt = dga_points.geom_type
        assert gt[0] == "Point"
        assert gt[1] == "Point"
        assert gt[3] is None  # null row
        assert _has_no_materialization(dga_points)

    def test_geom_type_mixed(self, dga_mixed):
        gt = dga_mixed.geom_type
        assert gt[0] == "Point"
        assert gt[1] == "LineString"
        assert gt[2] == "Polygon"
        assert gt[3] is None
        assert gt[4] == "MultiPoint"
        assert _has_no_materialization(dga_mixed)

    def test_is_empty_points(self, dga_points):
        ie = dga_points.is_empty
        assert ie.dtype == bool
        assert not ie[0]
        assert _has_no_materialization(dga_points)

    def test_bounds_points(self, dga_points):
        b = dga_points.bounds
        assert b.shape == (5, 4)
        # Point(0,0)
        np.testing.assert_array_equal(b[0], [0.0, 0.0, 0.0, 0.0])
        # Point(1,2)
        np.testing.assert_array_equal(b[1], [1.0, 2.0, 1.0, 2.0])
        # null row → NaN
        assert np.all(np.isnan(b[3]))
        assert _has_no_materialization(dga_points)

    def test_bounds_mixed(self, dga_mixed):
        b = dga_mixed.bounds
        assert b.shape == (5, 4)
        # Point(1,2)
        np.testing.assert_array_equal(b[0], [1.0, 2.0, 1.0, 2.0])
        # LineString([(0,0),(1,1),(2,0)])
        np.testing.assert_array_equal(b[1], [0.0, 0.0, 2.0, 1.0])
        # Polygon([(0,0),(1,0),(1,1),(0,1)])
        np.testing.assert_array_equal(b[2], [0.0, 0.0, 1.0, 1.0])
        # null → NaN
        assert np.all(np.isnan(b[3]))
        # MultiPoint([(0,0),(1,1)])
        np.testing.assert_array_equal(b[4], [0.0, 0.0, 1.0, 1.0])
        assert _has_no_materialization(dga_mixed)

    def test_total_bounds_points(self, dga_points):
        tb = dga_points.total_bounds
        np.testing.assert_array_equal(tb, [0.0, 0.0, 5.0, 6.0])
        assert _has_no_materialization(dga_points)

    def test_total_bounds_mixed(self, dga_mixed):
        tb = dga_mixed.total_bounds
        np.testing.assert_array_equal(tb, [0.0, 0.0, 2.0, 2.0])
        assert _has_no_materialization(dga_mixed)

    def test_total_bounds_empty(self):
        owned = from_shapely_geometries([])
        dga = DeviceGeometryArray._from_owned(owned)
        tb = dga.total_bounds
        assert np.all(np.isnan(tb))
        assert _has_no_materialization(dga)

    def test_take_no_materialization(self, dga_points):
        result = dga_points.take(np.array([0, 2, 4]))
        assert len(result) == 3
        assert _has_no_materialization(dga_points)
        assert _has_no_materialization(result)

    def test_copy_no_materialization(self, dga_points):
        result = dga_points.copy()
        assert len(result) == 5
        assert _has_no_materialization(dga_points)

    def test_concat_no_materialization(self, dga_points, dga_mixed):
        result = DeviceGeometryArray._concat_same_type([dga_points, dga_mixed])
        assert len(result) == 10
        assert _has_no_materialization(dga_points)
        assert _has_no_materialization(dga_mixed)

    def test_getitem_slice_no_full_materialization(self, dga_points):
        result = dga_points[1:3]
        assert len(result) == 2
        # Slicing should NOT trigger full cache materialization
        assert dga_points._shapely_cache is None

    def test_to_owned_no_materialization(self, dga_points):
        owned = dga_points.to_owned()
        assert owned is dga_points._owned
        assert _has_no_materialization(dga_points)


# ---------------------------------------------------------------------------
# GeoDataFrame / GeoSeries integration: no materialization
# ---------------------------------------------------------------------------


class TestGeoDataFrameNoMaterialization:
    """GeoDataFrame operations that should NOT trigger materialization."""

    def test_gdf_len(self, gdf_points):
        assert len(gdf_points) == 5
        dga = gdf_points.geometry.values
        assert isinstance(dga, DeviceGeometryArray)
        assert _has_no_materialization(dga)

    def test_gdf_shape(self, gdf_points):
        assert gdf_points.shape == (5, 2)
        dga = gdf_points.geometry.values
        assert _has_no_materialization(dga)

    def test_gdf_dtypes(self, gdf_points):
        dtypes = gdf_points.dtypes
        assert dtypes["geometry"].name == "device_geometry"
        dga = gdf_points.geometry.values
        assert _has_no_materialization(dga)

    def test_gdf_isna(self, gdf_points):
        na = gdf_points.geometry.isna()
        assert na[3]
        assert not na[0]
        dga = gdf_points.geometry.values
        assert _has_no_materialization(dga)

    def test_gdf_bounds(self, gdf_points):
        bounds_df = gdf_points.bounds
        assert list(bounds_df.columns) == ["minx", "miny", "maxx", "maxy"]
        assert bounds_df.shape == (5, 4)
        np.testing.assert_array_equal(bounds_df.iloc[0].values, [0.0, 0.0, 0.0, 0.0])
        dga = gdf_points.geometry.values
        assert _has_no_materialization(dga)

    def test_gdf_total_bounds(self, gdf_points):
        tb = gdf_points.total_bounds
        np.testing.assert_array_equal(tb, [0.0, 0.0, 5.0, 6.0])
        dga = gdf_points.geometry.values
        assert _has_no_materialization(dga)

    def test_gdf_geom_type(self, gdf_mixed):
        gt = gdf_mixed.geom_type
        assert gt.iloc[0] == "Point"
        assert gt.iloc[1] == "LineString"
        assert gt.iloc[2] == "Polygon"
        dga = gdf_mixed.geometry.values
        assert _has_no_materialization(dga)

    def test_gdf_column_access(self, gdf_points):
        col = gdf_points["a"]
        assert len(col) == 5
        dga = gdf_points.geometry.values
        assert _has_no_materialization(dga)

    def test_gdf_non_geom_operations(self, gdf_points):
        """Non-geometry operations should not touch the geometry column."""
        _ = gdf_points["a"].sum()
        _ = gdf_points["a"].mean()
        dga = gdf_points.geometry.values
        assert _has_no_materialization(dga)

    def test_concat_geodataframes(self, gdf_points):
        result = pd.concat([gdf_points, gdf_points])
        assert len(result) == 10
        assert isinstance(result.geometry.values, DeviceGeometryArray)

    def test_gdf_set_geometry_column(self):
        """Assigning a DeviceGeometryArray as geometry column should not materialize."""
        geoms = [Point(0, 0), Point(1, 1), Point(2, 2)]
        owned = from_shapely_geometries(geoms)
        dga = DeviceGeometryArray._from_owned(owned)
        gdf = GeoDataFrame({"a": [1, 2, 3]}, geometry=dga)
        assert isinstance(gdf.geometry.values, DeviceGeometryArray)
        assert _has_no_materialization(dga)

    def test_geoseries_construction(self, dga_points):
        gs = GeoSeries(dga_points)
        assert isinstance(gs.values, DeviceGeometryArray)
        assert _has_no_materialization(dga_points)


# ---------------------------------------------------------------------------
# Category B: operations that MUST trigger materialization with diagnostics
# ---------------------------------------------------------------------------


class TestMaterializationWithDiagnostics:
    """Operations that require Shapely should emit MATERIALIZATION diagnostics."""

    def test_area_no_materialization(self, dga_points):
        """Area is GPU-accelerated from owned buffers — no Shapely materialization."""
        _ = dga_points.area
        mat_events = [
            e for e in dga_points.diagnostics
            if e.kind == DiagnosticKind.MATERIALIZATION
        ]
        assert len(mat_events) == 0

    def test_length_no_materialization(self, dga_mixed):
        """Length is GPU-accelerated from owned buffers — no Shapely materialization."""
        _ = dga_mixed.length
        mat_events = [
            e for e in dga_mixed.diagnostics
            if e.kind == DiagnosticKind.MATERIALIZATION
        ]
        assert len(mat_events) == 0

    def test_is_valid_emits_diagnostic(self, dga_points):
        _ = dga_points.is_valid
        mat_events = [
            e for e in dga_points.diagnostics
            if e.kind == DiagnosticKind.MATERIALIZATION
        ]
        assert len(mat_events) >= 1

    def test_is_simple_emits_diagnostic(self, dga_points):
        _ = dga_points.is_simple
        mat_events = [
            e for e in dga_points.diagnostics
            if e.kind == DiagnosticKind.MATERIALIZATION
        ]
        assert len(mat_events) >= 1

    def test_centroid_no_materialization(self, dga_points):
        """Centroid is GPU-accelerated from owned buffers — no Shapely materialization."""
        _ = dga_points.centroid
        mat_events = [
            e for e in dga_points.diagnostics
            if e.kind == DiagnosticKind.MATERIALIZATION
        ]
        assert len(mat_events) == 0

    def test_buffer_emits_diagnostic(self, dga_points):
        _ = dga_points.buffer(1.0)
        mat_events = [
            e for e in dga_points.diagnostics
            if e.kind == DiagnosticKind.MATERIALIZATION
        ]
        assert len(mat_events) >= 1

    def test_intersects_emits_diagnostic(self, dga_points):
        _ = dga_points.intersects(Point(0, 0))
        mat_events = [
            e for e in dga_points.diagnostics
            if e.kind == DiagnosticKind.MATERIALIZATION
        ]
        assert len(mat_events) >= 1

    def test_distance_emits_diagnostic(self, dga_points):
        _ = dga_points.distance(Point(0, 0))
        mat_events = [
            e for e in dga_points.diagnostics
            if e.kind == DiagnosticKind.MATERIALIZATION
        ]
        assert len(mat_events) >= 1

    def test_data_property_emits_diagnostic(self, dga_points):
        """Accessing _data should emit a materialization diagnostic."""
        _ = dga_points._data
        mat_events = [
            e for e in dga_points.diagnostics
            if e.kind == DiagnosticKind.MATERIALIZATION
        ]
        assert len(mat_events) >= 1

    def test_area_returns_correct_values(self, dga_mixed):
        area = dga_mixed.area
        assert area[0] == 0.0  # Point
        assert area[1] == 0.0  # LineString
        assert area[2] == pytest.approx(1.0)  # unit square Polygon

    def test_length_returns_correct_values(self, dga_mixed):
        length = dga_mixed.length
        assert length[0] == 0.0  # Point
        assert length[1] > 0  # LineString

    def test_is_valid_returns_correct(self, dga_points):
        valid = dga_points.is_valid
        assert valid[0]  # Point(0,0) is valid
        assert valid[1]  # Point(1,2) is valid

    def test_to_wkb_no_materialization(self, dga_points):
        _ = dga_points.to_wkb()
        assert _has_no_materialization(dga_points)

    def test_to_wkt_no_materialization(self, dga_points):
        _ = dga_points.to_wkt()
        assert _has_no_materialization(dga_points)


# ---------------------------------------------------------------------------
# GeoDataFrame integration: materialization required
# ---------------------------------------------------------------------------


class TestGeoDataFrameMaterialization:
    """GeoDataFrame operations that require materialization."""

    def test_gdf_area_no_materialization(self, gdf_mixed):
        """Area is computed from owned buffers — no Shapely materialization."""
        area = gdf_mixed.area
        assert area.iloc[2] == pytest.approx(1.0)
        dga = gdf_mixed.geometry.values
        assert dga._shapely_cache is None

    def test_gdf_length_no_materialization(self, gdf_mixed):
        """Length is computed from owned buffers — no Shapely materialization."""
        length = gdf_mixed.length
        assert length.iloc[0] == 0.0
        dga = gdf_mixed.geometry.values
        assert dga._shapely_cache is None


# ---------------------------------------------------------------------------
# Arrow I/O: _get_geometry_types without materialization
# ---------------------------------------------------------------------------


class TestArrowNoMaterialization:
    """Arrow I/O should use tags directly for DeviceGeometryArray."""

    def test_get_geometry_types_points(self, gs_points):
        from vibespatial.api.io.arrow import _get_geometry_types

        types = _get_geometry_types(gs_points)
        assert "Point" in types
        dga = gs_points.values
        assert _has_no_materialization(dga)

    def test_get_geometry_types_mixed(self, gs_mixed):
        from vibespatial.api.io.arrow import _get_geometry_types

        types = _get_geometry_types(gs_mixed)
        assert "Point" in types
        assert "LineString" in types
        assert "Polygon" in types
        assert "MultiPoint" in types
        dga = gs_mixed.values
        assert _has_no_materialization(dga)


# ---------------------------------------------------------------------------
# Bounds correctness for all geometry families
# ---------------------------------------------------------------------------


class TestBoundsCorrectness:
    """Verify bounds computation matches Shapely for all geometry families."""

    @pytest.mark.parametrize(
        "geoms",
        [
            pytest.param([Point(1, 2), Point(3, 4)], id="points"),
            pytest.param(
                [LineString([(0, 0), (5, 3), (2, 7)])], id="linestring"
            ),
            pytest.param(
                [Polygon([(0, 0), (10, 0), (10, 5), (0, 5)])], id="polygon"
            ),
            pytest.param(
                [MultiPoint([(1, 1), (5, 5), (3, 7)])], id="multipoint"
            ),
        ],
    )
    def test_bounds_match_shapely(self, geoms):
        import shapely

        owned = from_shapely_geometries(geoms)
        dga = DeviceGeometryArray._from_owned(owned)

        device_bounds = dga.bounds
        shapely_bounds = shapely.bounds(np.array(geoms, dtype=object))

        np.testing.assert_array_almost_equal(device_bounds, shapely_bounds)
        assert _has_no_materialization(dga)

    def test_bounds_null_handling(self):
        owned = from_shapely_geometries([None, Point(1, 2), None])
        dga = DeviceGeometryArray._from_owned(owned)
        b = dga.bounds
        assert np.all(np.isnan(b[0]))
        np.testing.assert_array_equal(b[1], [1.0, 2.0, 1.0, 2.0])
        assert np.all(np.isnan(b[2]))
        assert _has_no_materialization(dga)


# ---------------------------------------------------------------------------
# Repr: only materializes displayed rows
# ---------------------------------------------------------------------------


class TestReprNoFullMaterialization:
    """GeoSeries repr should not trigger full materialization."""

    def test_geoseries_getitem_single_no_full_cache(self, dga_points):
        """Getting a single item should not populate the full cache."""
        _ = dga_points[0]
        assert dga_points._shapely_cache is None

    def test_geoseries_head_no_full_cache(self, dga_points):
        """Slicing head should not populate the full cache."""
        result = dga_points[:2]
        assert dga_points._shapely_cache is None
        assert isinstance(result, DeviceGeometryArray)


# ---------------------------------------------------------------------------
# Category E: sindex/sjoin shortcut
# ---------------------------------------------------------------------------


class TestSindexSjoinShortcut:
    """sindex and sjoin operations that detect DeviceGeometryArray and
    shortcut through the owned query engine without Shapely materialization."""

    def test_supports_owned_spatial_input(self, dga_points):
        assert dga_points.supports_owned_spatial_input() is True
        assert _has_no_materialization(dga_points)

    def test_owned_flat_sindex_no_materialization(self, dga_points):
        owned, flat_index = dga_points.owned_flat_sindex()
        assert owned is dga_points._owned
        assert flat_index is not None
        assert _has_no_materialization(dga_points)

    def test_owned_flat_sindex_is_cached(self, dga_points):
        owned1, flat1 = dga_points.owned_flat_sindex()
        owned2, flat2 = dga_points.owned_flat_sindex()
        assert owned1 is owned2
        assert flat1 is flat2
        assert _has_no_materialization(dga_points)

    def test_sindex_property_no_materialization(self, dga_points):
        sindex = dga_points.sindex
        assert sindex is not None
        assert _has_no_materialization(dga_points)

    def test_sindex_is_cached(self, dga_points):
        sindex1 = dga_points.sindex
        sindex2 = dga_points.sindex
        assert sindex1 is sindex2

    def test_has_sindex_false_initially(self, dga_points):
        assert dga_points.has_sindex is False

    def test_has_sindex_true_after_access(self, dga_points):
        _ = dga_points.sindex
        assert dga_points.has_sindex is True

    def test_sindex_len(self, dga_points):
        sindex = dga_points.sindex
        assert len(sindex) == 5

    def test_sindex_size(self, dga_points):
        sindex = dga_points.sindex
        assert sindex.size == 5

    def test_sindex_is_empty(self, dga_points):
        sindex = dga_points.sindex
        assert sindex.is_empty is False

    @pytest.mark.gpu
    def test_sindex_query_no_materialization(self, dga_points):
        """sindex.query() routes through owned dispatch without materializing."""
        from shapely.geometry import box as shapely_box

        sindex = dga_points.sindex
        result = sindex.query(shapely_box(-0.5, -0.5, 0.5, 0.5), predicate="intersects")
        assert 0 in result
        assert _has_no_materialization(dga_points)

    @pytest.mark.gpu
    def test_sindex_query_with_geoseries_no_materialization(self):
        """sindex.query() with a GeoSeries query avoids materialization on both sides."""
        from shapely.geometry import box as shapely_box

        tree_owned = from_shapely_geometries([Point(0, 0), Point(1, 2), Point(3, 4)])
        tree_dga = DeviceGeometryArray._from_owned(tree_owned)
        query_owned = from_shapely_geometries([shapely_box(-0.5, -0.5, 0.5, 0.5)])
        query_dga = DeviceGeometryArray._from_owned(query_owned)

        query_gs = GeoSeries(query_dga)
        sindex = tree_dga.sindex

        result = sindex.query(query_gs, predicate="intersects")
        assert result.shape[0] == 2  # (query_idx, tree_idx) pairs
        assert _has_no_materialization(tree_dga)
        assert _has_no_materialization(query_dga)

    @pytest.mark.gpu
    def test_sjoin_dga_no_materialization(self):
        """sjoin on DGA-backed GeoDataFrames uses owned dispatch without materialization."""
        tree_owned = from_shapely_geometries([Point(0, 0), Point(10, 10)])
        query_owned = from_shapely_geometries(
            [Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])]
        )
        tree_dga = DeviceGeometryArray._from_owned(tree_owned)
        query_dga = DeviceGeometryArray._from_owned(query_owned)
        left = GeoDataFrame({"a": [1]}, geometry=query_dga)
        right = GeoDataFrame({"b": [10, 20]}, geometry=tree_dga)

        result = geopandas.sjoin(left, right, predicate="intersects")
        assert len(result) == 1
        assert _has_no_materialization(tree_dga)
        assert _has_no_materialization(query_dga)

    @pytest.mark.gpu
    def test_sjoin_dga_records_owned_dispatch(self):
        """sjoin on DGA-backed GeoDataFrames records owned_spatial_query dispatch."""
        tree_owned = from_shapely_geometries([Point(0, 0), Point(10, 10)])
        query_owned = from_shapely_geometries(
            [Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])]
        )
        tree_dga = DeviceGeometryArray._from_owned(tree_owned)
        query_dga = DeviceGeometryArray._from_owned(query_owned)
        left = GeoDataFrame({"a": [1]}, geometry=query_dga)
        right = GeoDataFrame({"b": [10, 20]}, geometry=tree_dga)

        geopandas.clear_dispatch_events()
        geopandas.sjoin(left, right, predicate="intersects")
        dispatch_events = geopandas.get_dispatch_events(clear=True)

        sjoin_events = [e for e in dispatch_events if e.surface == "geopandas.tools.sjoin"]
        assert sjoin_events
        assert sjoin_events[-1].implementation == "owned_spatial_query"

    @pytest.mark.gpu
    def test_sjoin_outer_dga_no_materialization(self):
        """Outer sjoin on DGA-backed GeoDataFrames avoids materialization."""
        tree_owned = from_shapely_geometries([Point(0, 0), Point(10, 10)])
        query_owned = from_shapely_geometries(
            [Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])]
        )
        tree_dga = DeviceGeometryArray._from_owned(tree_owned)
        query_dga = DeviceGeometryArray._from_owned(query_owned)
        left = GeoDataFrame({"a": [1]}, geometry=query_dga)
        right = GeoDataFrame({"b": [10, 20]}, geometry=tree_dga)

        result = geopandas.sjoin(left, right, how="outer", predicate="intersects")
        # Outer join: matched + unmatched from both sides
        assert len(result) >= 1
        assert _has_no_materialization(query_dga)

    @pytest.mark.gpu
    def test_sindex_query_dga_tree_shapely_query(self, dga_points):
        """Scalar Shapely query against DGA-backed sindex avoids tree materialization."""
        sindex = dga_points.sindex
        result = sindex.query(Point(0, 0), predicate="intersects")
        assert 0 in result
        assert _has_no_materialization(dga_points)

    def test_mixed_geometry_sindex_no_materialization(self, dga_mixed):
        """Mixed-geometry DGA sindex works without materialization."""
        assert dga_mixed.supports_owned_spatial_input() is True
        owned, flat_index = dga_mixed.owned_flat_sindex()
        assert owned is dga_mixed._owned
        assert _has_no_materialization(dga_mixed)
