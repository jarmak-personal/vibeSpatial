"""Tests for DeviceGeometryArray."""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    box,
)

import vibespatial.api as geopandas
from vibespatial.api.geometry_array import GeometryArray, GeometryDtype
from vibespatial.geometry.device_array import DeviceGeometryArray, DeviceGeometryDtype
from vibespatial.geometry.owned import (
    DiagnosticKind,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
    concat_owned_scatter,
    from_shapely_geometries,
)
from vibespatial.runtime import has_gpu_runtime
from vibespatial.runtime.fallbacks import StrictNativeFallbackError
from vibespatial.runtime.residency import Residency

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def points():
    return [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3), Point(4, 4)]


@pytest.fixture
def points_with_null():
    return [Point(0, 0), None, Point(2, 2), None, Point(4, 4)]


@pytest.fixture
def mixed_geometries():
    return [
        Point(0, 0),
        LineString([(0, 0), (1, 1)]),
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
        MultiPoint([(0, 0), (1, 1)]),
        None,
    ]


@pytest.fixture
def dga_points(points):
    owned = from_shapely_geometries(points)
    return DeviceGeometryArray._from_owned(owned)


@pytest.fixture
def dga_with_nulls(points_with_null):
    owned = from_shapely_geometries(points_with_null)
    return DeviceGeometryArray._from_owned(owned)


@pytest.fixture
def dga_mixed(mixed_geometries):
    owned = from_shapely_geometries(mixed_geometries)
    return DeviceGeometryArray._from_owned(owned)


def _make_device_only_dga(geometries):
    owned = from_shapely_geometries(geometries, residency=Residency.DEVICE)
    owned.families = {
        family: FamilyGeometryBuffer(
            family=buffer.family,
            schema=buffer.schema,
            row_count=buffer.row_count,
            x=np.empty(0, dtype=np.float64),
            y=np.empty(0, dtype=np.float64),
            geometry_offsets=np.empty(0, dtype=np.int32),
            empty_mask=np.empty(0, dtype=np.bool_),
            part_offsets=None,
            ring_offsets=None,
            bounds=None,
            host_materialized=False,
        )
        for family, buffer in owned.families.items()
    }
    return DeviceGeometryArray._from_owned(owned)


# ---------------------------------------------------------------------------
# Dtype
# ---------------------------------------------------------------------------

class TestDeviceGeometryDtype:
    def test_name(self):
        assert DeviceGeometryDtype.name == "device_geometry"

    def test_construct_from_string(self):
        dtype = DeviceGeometryDtype.construct_from_string("device_geometry")
        assert isinstance(dtype, DeviceGeometryDtype)

    def test_construct_from_string_invalid(self):
        with pytest.raises(TypeError):
            DeviceGeometryDtype.construct_from_string("geometry")

    def test_construct_array_type(self):
        assert DeviceGeometryDtype.construct_array_type() is DeviceGeometryArray

    def test_na_value(self):
        assert DeviceGeometryDtype.na_value is None

    def test_geometry_dtype_equality_is_symmetric(self):
        assert DeviceGeometryDtype() == GeometryDtype()
        assert GeometryDtype() == DeviceGeometryDtype()
        assert pd.api.types.is_dtype_equal(GeometryDtype(), DeviceGeometryDtype())
        assert DeviceGeometryDtype() == "geometry"
        assert not (DeviceGeometryDtype() == "device_geometry")
        assert DeviceGeometryDtype.construct_from_string("device_geometry") == DeviceGeometryDtype()
        assert GeometryDtype() == "geometry"
        assert not (GeometryDtype() == "device_geometry")
        assert hash(DeviceGeometryDtype()) == hash(GeometryDtype()) == hash("geometry")
        dtype_lookup = {DeviceGeometryDtype(): "device"}
        assert dtype_lookup[GeometryDtype()] == "device"
        assert dtype_lookup["geometry"] == "device"


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_from_owned(self, points):
        owned = from_shapely_geometries(points)
        dga = DeviceGeometryArray._from_owned(owned)
        assert len(dga) == 5
        assert dga.dtype == DeviceGeometryDtype()

    def test_from_sequence(self, points):
        dga = DeviceGeometryArray._from_sequence(points)
        assert len(dga) == 5

    def test_from_sequence_with_nulls(self, points_with_null):
        dga = DeviceGeometryArray._from_sequence(points_with_null)
        assert len(dga) == 5
        assert dga.isna().sum() == 2

    def test_from_sequence_empty(self):
        dga = DeviceGeometryArray._from_sequence([])
        assert len(dga) == 0

    def test_owned_property(self, dga_points):
        assert isinstance(dga_points.owned, OwnedGeometryArray)

    def test_crs_passthrough(self, points):
        owned = from_shapely_geometries(points)
        dga = DeviceGeometryArray._from_owned(owned, crs="EPSG:4326")
        assert dga.crs == "EPSG:4326"


# ---------------------------------------------------------------------------
# ExtensionArray protocol basics
# ---------------------------------------------------------------------------

class TestProtocol:
    def test_len(self, dga_points):
        assert len(dga_points) == 5

    def test_dtype(self, dga_points):
        assert dga_points.dtype == DeviceGeometryDtype()

    def test_nbytes_positive(self, dga_points):
        assert dga_points.nbytes > 0

    def test_isna_no_nulls(self, dga_points):
        result = dga_points.isna()
        assert result.dtype == bool
        assert not result.any()

    def test_isna_with_nulls(self, dga_with_nulls):
        result = dga_with_nulls.isna()
        np.testing.assert_array_equal(result, [False, True, False, True, False])


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

class TestIndexing:
    def test_getitem_scalar(self, dga_points):
        geom = dga_points[0]
        assert isinstance(geom, Point)
        assert geom.x == 0.0 and geom.y == 0.0

    def test_getitem_scalar_negative(self, dga_points):
        geom = dga_points[-1]
        assert isinstance(geom, Point)
        assert geom.x == 4.0

    def test_getitem_null_returns_none(self, dga_with_nulls):
        assert dga_with_nulls[1] is None

    def test_getitem_slice(self, dga_points):
        result = dga_points[1:3]
        assert isinstance(result, DeviceGeometryArray)
        assert len(result) == 2
        assert result[0].x == 1.0

    def test_getitem_bool_mask(self, dga_points):
        mask = np.array([True, False, True, False, True])
        result = dga_points[mask]
        assert isinstance(result, DeviceGeometryArray)
        assert len(result) == 3

    def test_getitem_fancy_index(self, dga_points):
        idx = np.array([4, 2, 0])
        result = dga_points[idx]
        assert len(result) == 3
        assert result[0].x == 4.0
        assert result[2].x == 0.0

    def test_setitem_scalar(self, dga_points):
        dga_points[0] = Point(99, 99)
        assert dga_points[0].x == 99.0

    def test_setitem_none(self, dga_points):
        dga_points[0] = None
        assert dga_points[0] is None
        assert dga_points.isna()[0]


# ---------------------------------------------------------------------------
# take / copy / concat
# ---------------------------------------------------------------------------

class TestTakeCopyConcatNoRoundtrip:
    """Verify take/copy/concat operate on owned buffers without Shapely round-trip."""

    def test_take_basic(self, dga_points):
        result = dga_points.take(np.array([0, 2, 4]))
        assert len(result) == 3
        assert result[0].x == 0.0
        assert result[1].x == 2.0
        assert result[2].x == 4.0

    def test_take_no_shapely_materialization(self, points):
        """take should NOT trigger Shapely materialization on the source."""
        owned = from_shapely_geometries(points)
        dga = DeviceGeometryArray._from_owned(owned)
        # Clear any construction diagnostics
        dga._owned.diagnostics.clear()

        _result = dga.take(np.array([0, 2]))

        # No MATERIALIZATION event should have been recorded on the source
        mat_events = [
            e for e in dga._owned.diagnostics if e.kind == DiagnosticKind.MATERIALIZATION
        ]
        assert len(mat_events) == 0, f"Unexpected materialization: {mat_events}"

    def test_take_with_nulls(self, dga_with_nulls):
        result = dga_with_nulls.take(np.array([0, 1, 2]))
        assert result[0].x == 0.0
        assert result[1] is None
        assert result[2].x == 2.0

    def test_take_allow_fill(self, dga_points):
        result = dga_points.take(np.array([0, -1, 2]), allow_fill=True)
        assert result[0].x == 0.0
        assert result[1] is None  # filled
        assert result[2].x == 2.0

    def test_copy_independent(self, dga_points):
        copied = dga_points.copy()
        assert len(copied) == len(dga_points)
        # Verify independence
        assert copied._owned is not dga_points._owned
        assert not np.shares_memory(copied._owned.validity, dga_points._owned.validity)

    def test_copy_no_shapely_materialization(self, points):
        owned = from_shapely_geometries(points)
        dga = DeviceGeometryArray._from_owned(owned)
        dga._owned.diagnostics.clear()

        _copied = dga.copy()

        mat_events = [
            e for e in dga._owned.diagnostics if e.kind == DiagnosticKind.MATERIALIZATION
        ]
        assert len(mat_events) == 0

    @pytest.mark.skipif(not has_gpu_runtime(), reason="CUDA runtime not available")
    def test_copy_device_resident_does_not_require_host_state(self, monkeypatch):
        dga = _make_device_only_dga(
            [Point(0, 0), None, Point(), LineString([(0, 0), (2, 4)]), Point(5, 6)]
        )
        dga._owned.diagnostics.clear()

        def _fail_host_state():
            raise AssertionError("copy should not call _ensure_host_state")

        monkeypatch.setattr(dga._owned, "_ensure_host_state", _fail_host_state)

        copied = dga.copy()

        assert copied._owned.residency is Residency.DEVICE
        assert copied._owned.device_state is not None
        assert dga._owned.device_state is not None
        assert copied._owned.device_state.families[next(iter(copied._owned.device_state.families))].x.data.ptr != dga._owned.device_state.families[next(iter(dga._owned.device_state.families))].x.data.ptr
        assert all(not buffer.host_materialized for buffer in copied._owned.families.values())
        assert not any(event.kind == DiagnosticKind.TRANSFER for event in dga._owned.diagnostics)
        assert not any(event.kind == DiagnosticKind.MATERIALIZATION for event in dga._owned.diagnostics)

    def test_concat_same_type(self, points):
        owned1 = from_shapely_geometries(points[:3])
        owned2 = from_shapely_geometries(points[3:])
        dga1 = DeviceGeometryArray._from_owned(owned1)
        dga2 = DeviceGeometryArray._from_owned(owned2)

        result = DeviceGeometryArray._concat_same_type([dga1, dga2])
        assert len(result) == 5
        assert result[0].x == 0.0
        assert result[3].x == 3.0
        assert result[4].x == 4.0

    def test_owned_concat_preserves_complete_validity_cache(self, points):
        owned1 = from_shapely_geometries(points[:3])
        owned2 = from_shapely_geometries(points[3:])
        owned1._cached_is_valid_mask = np.asarray([True, False, True], dtype=bool)
        owned2._cached_is_valid_mask = np.asarray([True, True], dtype=bool)

        result = OwnedGeometryArray.concat([owned1, owned2])

        assert np.array_equal(
            result._cached_is_valid_mask,
            np.asarray([True, False, True, True, True], dtype=bool),
        )

    def test_concat_empty(self):
        result = DeviceGeometryArray._concat_same_type([])
        assert len(result) == 0

    def test_concat_with_nulls(self):
        owned1 = from_shapely_geometries([Point(0, 0), None])
        owned2 = from_shapely_geometries([None, Point(1, 1)])
        dga1 = DeviceGeometryArray._from_owned(owned1)
        dga2 = DeviceGeometryArray._from_owned(owned2)

        result = DeviceGeometryArray._concat_same_type([dga1, dga2])
        assert len(result) == 4
        assert result[0].x == 0.0
        assert result[1] is None
        assert result[2] is None
        assert result[3].x == 1.0

    def test_concat_with_host_geometry_array_promotes_to_owned(self, points):
        owned = from_shapely_geometries(points[:3])
        dga = DeviceGeometryArray._from_owned(owned)
        host = GeometryArray(np.asarray(points[3:], dtype=object))

        result = DeviceGeometryArray._concat_same_type([dga, host])

        assert isinstance(result, DeviceGeometryArray)
        assert len(result) == 5
        assert result[0].x == 0.0
        assert result[4].x == 4.0
        assert getattr(result, "_owned", None) is not None

    def test_concat_no_shapely_materialization(self, points):
        owned1 = from_shapely_geometries(points[:3])
        owned2 = from_shapely_geometries(points[3:])
        dga1 = DeviceGeometryArray._from_owned(owned1)
        dga2 = DeviceGeometryArray._from_owned(owned2)
        dga1._owned.diagnostics.clear()
        dga2._owned.diagnostics.clear()

        _result = DeviceGeometryArray._concat_same_type([dga1, dga2])

        for dga in [dga1, dga2]:
            mat_events = [
                e for e in dga._owned.diagnostics if e.kind == DiagnosticKind.MATERIALIZATION
            ]
            assert len(mat_events) == 0

    @pytest.mark.skipif(not has_gpu_runtime(), reason="CUDA runtime not available")
    def test_concat_owned_scatter_keeps_device_residency(self, points):
        base = from_shapely_geometries(points, residency=Residency.DEVICE)
        replacement = from_shapely_geometries(
            [Point(99, 99), None],
            residency=Residency.DEVICE,
        )

        result = concat_owned_scatter(
            base,
            replacement,
            np.array([1, 3], dtype=np.int64),
        )

        assert result.residency is Residency.DEVICE
        assert result._validity is None
        assert result._tags is None
        assert result._family_row_offsets is None

        shapely_result = result.to_shapely()
        assert shapely_result[0].equals(points[0])
        assert shapely_result[1].equals(Point(99, 99))
        assert shapely_result[2].equals(points[2])
        assert shapely_result[3] is None
        assert shapely_result[4].equals(points[4])


# ---------------------------------------------------------------------------
# Mixed geometry families
# ---------------------------------------------------------------------------

class TestMixedFamilies:
    def test_mixed_roundtrip(self, mixed_geometries, dga_mixed):
        for i, expected in enumerate(mixed_geometries):
            result = dga_mixed[i]
            if expected is None:
                assert result is None
            else:
                assert result.geom_type == expected.geom_type

    def test_take_mixed(self, dga_mixed):
        result = dga_mixed.take(np.array([0, 2, 4]))
        assert result[0].geom_type == "Point"
        assert result[1].geom_type == "Polygon"
        assert result[2] is None

    def test_concat_mixed(self, mixed_geometries):
        owned1 = from_shapely_geometries(mixed_geometries[:2])
        owned2 = from_shapely_geometries(mixed_geometries[2:])
        dga1 = DeviceGeometryArray._from_owned(owned1)
        dga2 = DeviceGeometryArray._from_owned(owned2)

        result = DeviceGeometryArray._concat_same_type([dga1, dga2])
        assert len(result) == 5
        assert result[0].geom_type == "Point"
        assert result[2].geom_type == "Polygon"


# ---------------------------------------------------------------------------
# Device-resident bounds
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not has_gpu_runtime(), reason="CUDA runtime not available")
class TestDeviceBounds:
    def test_bounds_do_not_require_host_state(self, monkeypatch):
        dga = _make_device_only_dga(
            [Point(0, 0), None, Point(), LineString([(0, 0), (2, 4)]), Point(5, 6)]
        )
        dga._owned.diagnostics.clear()

        def _fail_host_state():
            raise AssertionError("bounds should not call _ensure_host_state")

        monkeypatch.setattr(dga._owned, "_ensure_host_state", _fail_host_state)

        bounds = dga.bounds

        np.testing.assert_array_equal(bounds[0], [0.0, 0.0, 0.0, 0.0])
        assert np.all(np.isnan(bounds[1]))
        assert np.all(np.isnan(bounds[2]))
        np.testing.assert_array_equal(bounds[3], [0.0, 0.0, 2.0, 4.0])
        np.testing.assert_array_equal(bounds[4], [5.0, 6.0, 5.0, 6.0])
        assert all(not buffer.host_materialized for buffer in dga._owned.families.values())
        assert not any(
            event.kind == DiagnosticKind.MATERIALIZATION for event in dga._owned.diagnostics
        )

    def test_total_bounds_do_not_require_host_state(self, monkeypatch):
        dga = _make_device_only_dga(
            [Point(0, 0), None, Point(), LineString([(0, 0), (2, 4)]), Point(5, 6)]
        )
        dga._owned.diagnostics.clear()

        def _fail_host_state():
            raise AssertionError("total_bounds should not call _ensure_host_state")

        monkeypatch.setattr(dga._owned, "_ensure_host_state", _fail_host_state)

        total_bounds = dga.total_bounds

        np.testing.assert_array_equal(total_bounds, [0.0, 0.0, 5.0, 6.0])
        assert all(not buffer.host_materialized for buffer in dga._owned.families.values())
        assert not any(
            event.kind == DiagnosticKind.MATERIALIZATION for event in dga._owned.diagnostics
        )


# ---------------------------------------------------------------------------
# Complex geometry families (polygon, multi*)
# ---------------------------------------------------------------------------

class TestComplexFamilies:
    def test_polygon_with_hole(self):
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        hole = [(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]
        poly = Polygon(exterior, [hole])
        dga = DeviceGeometryArray._from_sequence([poly])
        result = dga[0]
        assert result.geom_type == "Polygon"
        assert len(list(result.interiors)) == 1

    def test_multilinestring(self):
        mls = MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]])
        dga = DeviceGeometryArray._from_sequence([mls])
        result = dga[0]
        assert result.geom_type == "MultiLineString"
        assert len(result.geoms) == 2

    def test_multipolygon(self):
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])
        p2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 2)])
        mp = MultiPolygon([p1, p2])
        dga = DeviceGeometryArray._from_sequence([mp])
        result = dga[0]
        assert result.geom_type == "MultiPolygon"
        assert len(result.geoms) == 2

    def test_concat_polygons(self):
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        p2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)])
        dga1 = DeviceGeometryArray._from_sequence([p1])
        dga2 = DeviceGeometryArray._from_sequence([p2])
        result = DeviceGeometryArray._concat_same_type([dga1, dga2])
        assert len(result) == 2
        assert result[0].equals(p1)
        assert result[1].equals(p2)

    def test_concat_multipolygons(self):
        mp1 = MultiPolygon([
            Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
        ])
        mp2 = MultiPolygon([
            Polygon([(2, 2), (3, 2), (3, 3), (2, 2)]),
        ])
        dga1 = DeviceGeometryArray._from_sequence([mp1])
        dga2 = DeviceGeometryArray._from_sequence([mp2])
        result = DeviceGeometryArray._concat_same_type([dga1, dga2])
        assert len(result) == 2
        assert result[0].geom_type == "MultiPolygon"
        assert result[1].geom_type == "MultiPolygon"


# ---------------------------------------------------------------------------
# Empty geometries
# ---------------------------------------------------------------------------

class TestEmptyGeometries:
    def test_empty_point(self):
        dga = DeviceGeometryArray._from_sequence([Point()])
        result = dga[0]
        assert result.is_empty

    def test_empty_linestring(self):
        dga = DeviceGeometryArray._from_sequence([LineString()])
        result = dga[0]
        assert result.is_empty

    def test_empty_polygon(self):
        dga = DeviceGeometryArray._from_sequence([Polygon()])
        result = dga[0]
        assert result.is_empty


# ---------------------------------------------------------------------------
# Diagnostic events
# ---------------------------------------------------------------------------

class TestDiagnostics:
    def test_geometry_array_from_owned_materializes_without_wkb_bridge(
        self, points, monkeypatch
    ):
        from vibespatial.io import wkb as wkb_module

        def _fail_encode(*args, **kwargs):
            raise AssertionError("GeometryArray host materialization should not use WKB bridge")

        monkeypatch.setattr(wkb_module, "encode_wkb_owned", _fail_encode)

        owned = from_shapely_geometries(points)
        ga = GeometryArray.from_owned(owned)

        result = ga._data

        assert result.shape == (len(points),)
        assert result[0].equals(Point(0, 0))
        assert result[-1].equals(Point(4, 4))

    def test_materialization_event_on_shapely_access(self, points):
        owned = from_shapely_geometries(points)
        dga = DeviceGeometryArray._from_owned(owned)
        dga._owned.diagnostics.clear()

        # Force full materialization
        _ = dga._ensure_shapely_cache()

        mat_events = [
            e for e in dga._owned.diagnostics if e.kind == DiagnosticKind.MATERIALIZATION
        ]
        assert len(mat_events) >= 1
        assert mat_events[0].visible_to_user is True

    def test_no_materialization_on_isna(self, points):
        owned = from_shapely_geometries(points)
        dga = DeviceGeometryArray._from_owned(owned)
        dga._owned.diagnostics.clear()

        _ = dga.isna()

        mat_events = [
            e for e in dga._owned.diagnostics if e.kind == DiagnosticKind.MATERIALIZATION
        ]
        assert len(mat_events) == 0

    def test_no_materialization_on_len(self, dga_points):
        dga_points._owned.diagnostics.clear()
        _ = len(dga_points)
        mat_events = [
            e for e in dga_points._owned.diagnostics if e.kind == DiagnosticKind.MATERIALIZATION
        ]
        assert len(mat_events) == 0

    def test_no_materialization_on_nbytes(self, dga_points):
        dga_points._owned.diagnostics.clear()
        _ = dga_points.nbytes
        mat_events = [
            e for e in dga_points._owned.diagnostics if e.kind == DiagnosticKind.MATERIALIZATION
        ]
        assert len(mat_events) == 0


# ---------------------------------------------------------------------------
# GeoDataFrame integration
# ---------------------------------------------------------------------------

class TestGeoDataFrameIntegration:
    def test_as_column_in_dataframe(self, dga_points):
        df = pd.DataFrame({"a": range(5)})
        df["geom"] = dga_points
        assert len(df) == 5
        assert df["geom"].dtype == DeviceGeometryDtype()

    def test_series_with_dga(self, dga_points):
        s = pd.Series(dga_points)
        assert len(s) == 5
        assert s.dtype == DeviceGeometryDtype()

    def test_dataframe_slice(self, dga_points):
        df = pd.DataFrame({"a": range(5), "geom": dga_points})
        sliced = df.iloc[1:3]
        assert len(sliced) == 2
        geom_col = sliced["geom"].values
        assert isinstance(geom_col, DeviceGeometryArray)

    def test_dataframe_concat(self, points):
        owned1 = from_shapely_geometries(points[:3])
        owned2 = from_shapely_geometries(points[3:])
        df1 = pd.DataFrame({"a": range(3), "geom": DeviceGeometryArray._from_owned(owned1)})
        df2 = pd.DataFrame({"a": range(3, 5), "geom": DeviceGeometryArray._from_owned(owned2)})
        result = pd.concat([df1, df2], ignore_index=True)
        assert len(result) == 5
        assert result["geom"].dtype == DeviceGeometryDtype()

    def test_dataframe_concat_host_first_mixed_geometry_promotes_to_device_array(self, points):
        host = pd.DataFrame(
            {"a": range(3), "geom": GeometryArray(np.asarray(points[:3], dtype=object))}
        )
        device = pd.DataFrame(
            {
                "a": range(3, 5),
                "geom": DeviceGeometryArray._from_owned(
                    from_shapely_geometries(points[3:])
                ),
            }
        )

        result = pd.concat([host, device], ignore_index=True)

        assert len(result) == 5
        assert result["geom"].dtype == DeviceGeometryDtype()
        assert isinstance(result["geom"].values, DeviceGeometryArray)
        assert getattr(result["geom"].values, "_owned", None) is not None

    def test_dataframe_take(self, dga_points):
        df = pd.DataFrame({"a": range(5), "geom": dga_points})
        taken = df.iloc[[0, 4]]
        assert len(taken) == 2

    @pytest.mark.skipif(not has_gpu_runtime(), reason="CUDA runtime not available")
    def test_active_geometry_setitem_preserves_device_backing(self):
        gdf = geopandas.GeoDataFrame(
            {"value": [1, 2]},
            geometry=DeviceGeometryArray._from_owned(
                from_shapely_geometries(
                    [Point(0, 0), Point(1, 1)],
                    residency=Residency.DEVICE,
                )
            ),
            crs="EPSG:4326",
        )

        replacement = pd.Series(
            np.asarray([Point(2, 2), Point(3, 3)], dtype=object),
            index=gdf.index,
            name="geometry",
        )

        gdf["geometry"] = replacement

        assert gdf.geometry.dtype == DeviceGeometryDtype()
        assert isinstance(gdf.geometry.values, DeviceGeometryArray)
        assert gdf.geometry.values._owned.residency is Residency.DEVICE

    def test_shapely_make_valid_dispatches_device_geometry_array_directly(self):
        bowtie = Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)])
        dga = DeviceGeometryArray._from_sequence([bowtie])

        result = shapely.make_valid(dga)

        assert isinstance(result, DeviceGeometryArray)
        assert shapely.is_valid(result[0])

    @pytest.mark.skipif(not has_gpu_runtime(), reason="CUDA runtime not available")
    def test_set_geometry_preserves_device_backing_for_array_like_replacement(self):
        gdf = geopandas.GeoDataFrame(
            {"value": [1, 2]},
            geometry=DeviceGeometryArray._from_owned(
                from_shapely_geometries(
                    [Point(0, 0), Point(1, 1)],
                    residency=Residency.DEVICE,
                )
            ),
            crs="EPSG:4326",
        )

        result = gdf.set_geometry(np.asarray([Point(2, 2), Point(3, 3)], dtype=object))

        assert result.geometry.dtype == DeviceGeometryDtype()
        assert isinstance(result.geometry.values, DeviceGeometryArray)
        assert result.geometry.values._owned.residency is Residency.DEVICE

    @pytest.mark.skipif(not has_gpu_runtime(), reason="CUDA runtime not available")
    def test_set_geometry_records_observable_fallback_when_device_rebuild_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        import vibespatial.api.geodataframe as geodataframe_module

        geopandas.clear_fallback_events()
        gdf = geopandas.GeoDataFrame(
            {"value": [1, 2]},
            geometry=DeviceGeometryArray._from_owned(
                from_shapely_geometries(
                    [Point(0, 0), Point(1, 1)],
                    residency=Residency.DEVICE,
                )
            ),
            crs="EPSG:4326",
        )
        monkeypatch.setattr(
            geodataframe_module,
            "_device_geometry_from_shapely",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("device rebuild failed")),
        )

        result = gdf.set_geometry(np.asarray([Point(2, 2), Point(3, 3)], dtype=object))

        assert not isinstance(result.geometry.values, DeviceGeometryArray)
        events = geopandas.get_fallback_events(clear=True)
        assert events[-1].surface == "GeoDataFrame.set_geometry"
        assert "device rebuild failed" in events[-1].detail

    @pytest.mark.skipif(not has_gpu_runtime(), reason="CUDA runtime not available")
    def test_set_geometry_strict_native_blocks_host_fallback_when_device_rebuild_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        import vibespatial.api.geodataframe as geodataframe_module

        geopandas.clear_fallback_events()
        monkeypatch.setenv("VIBESPATIAL_STRICT_NATIVE", "1")
        gdf = geopandas.GeoDataFrame(
            {"value": [1, 2]},
            geometry=DeviceGeometryArray._from_owned(
                from_shapely_geometries(
                    [Point(0, 0), Point(1, 1)],
                    residency=Residency.DEVICE,
                )
            ),
            crs="EPSG:4326",
        )
        monkeypatch.setattr(
            geodataframe_module,
            "_device_geometry_from_shapely",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("device rebuild failed")),
        )

        with pytest.raises(StrictNativeFallbackError):
            gdf.set_geometry(np.asarray([Point(2, 2), Point(3, 3)], dtype=object))

        events = geopandas.get_fallback_events(clear=True)
        assert events[-1].surface == "GeoDataFrame.set_geometry"

    def test_centroid_warning_mentions_operation_name(self):
        dga = DeviceGeometryArray._from_sequence([Point(0, 0)])
        dga.crs = "EPSG:4326"

        with pytest.warns(UserWarning, match=r"Results from 'centroid'"):
            _ = dga.centroid


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_pickle_roundtrip(self, dga_points):
        data = pickle.dumps(dga_points)
        restored = pickle.loads(data)
        assert len(restored) == len(dga_points)
        for i in range(len(dga_points)):
            assert restored[i].equals(dga_points[i])

    def test_pickle_with_nulls(self, dga_with_nulls):
        data = pickle.dumps(dga_with_nulls)
        restored = pickle.loads(data)
        assert len(restored) == len(dga_with_nulls)
        for i in range(len(dga_with_nulls)):
            if dga_with_nulls[i] is None:
                assert restored[i] is None
            else:
                assert restored[i].equals(dga_with_nulls[i])

    @pytest.mark.skipif(not has_gpu_runtime(), reason="CUDA runtime not available")
    def test_pickle_device_resident_avoids_host_state_and_shapely(self, monkeypatch):
        from vibespatial import has_pylibcudf_support

        if not has_pylibcudf_support():
            pytest.skip("pylibcudf device WKB encoder not available")

        dga = _make_device_only_dga(
            [Point(0, 0), None, Point(), LineString([(0, 0), (2, 4)]), Point(5, 6)]
        )
        dga._owned.diagnostics.clear()

        def _fail_host_state():
            raise AssertionError("pickle should not call _ensure_host_state")

        def _fail_shapely_cache():
            raise AssertionError("pickle should not call _ensure_shapely_cache")

        monkeypatch.setattr(dga._owned, "_ensure_host_state", _fail_host_state)
        monkeypatch.setattr(dga, "_ensure_shapely_cache", _fail_shapely_cache)

        data = pickle.dumps(dga)
        restored = pickle.loads(data)

        assert restored._owned.residency is Residency.DEVICE
        assert restored._owned.device_state is not None
        assert len(restored) == len(dga)
        assert restored[0].equals(Point(0, 0))
        assert restored[1] is None
        assert restored[3].equals(LineString([(0, 0), (2, 4)]))
        assert not any(event.kind == DiagnosticKind.MATERIALIZATION for event in dga._owned.diagnostics)


# ---------------------------------------------------------------------------
# WKB / WKT serialization from owned buffers
# ---------------------------------------------------------------------------


class TestWkbWktSerialization:
    def test_to_wkb_uses_owned_buffers_without_materialization(self):
        dga = DeviceGeometryArray._from_sequence([Point(0, 0), None, LineString([(0, 0), (2, 4)])])
        dga._owned.diagnostics.clear()

        values = dga.to_wkb()

        assert isinstance(values, np.ndarray)
        assert values.dtype == object
        assert isinstance(values[0], bytes)
        assert values[1] is None
        assert isinstance(values[2], bytes)
        assert not any(event.kind == DiagnosticKind.MATERIALIZATION for event in dga._owned.diagnostics)

    def test_to_wkt_matches_expected_without_materialization(self):
        dga = DeviceGeometryArray._from_sequence([
            Point(0, 0),
            None,
            Point(),
            LineString([(0, 0), (2, 4)]),
            Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
        ])
        dga._owned.diagnostics.clear()

        values = dga.to_wkt(rounding_precision=-1)

        np.testing.assert_array_equal(
            values,
            np.asarray(
                [
                    "POINT (0 0)",
                    None,
                    "POINT EMPTY",
                    "LINESTRING (0 0, 2 4)",
                    "POLYGON ((0 0, 1 0, 1 1, 0 0))",
                ],
                dtype=object,
            ),
        )
        assert not any(event.kind == DiagnosticKind.MATERIALIZATION for event in dga._owned.diagnostics)

    @pytest.mark.skipif(not has_gpu_runtime(), reason="CUDA runtime not available")
    def test_to_wkt_device_resident_avoids_host_state_and_shapely(self, monkeypatch):
        dga = _make_device_only_dga(
            [Point(0, 0), None, Point(), LineString([(0, 0), (2, 4)]), Point(5, 6)]
        )
        dga._owned.diagnostics.clear()

        def _fail_host_state():
            raise AssertionError("to_wkt should not call _ensure_host_state")

        def _fail_shapely_cache():
            raise AssertionError("to_wkt should not call _ensure_shapely_cache")

        monkeypatch.setattr(dga._owned, "_ensure_host_state", _fail_host_state)
        monkeypatch.setattr(dga, "_ensure_shapely_cache", _fail_shapely_cache)

        values = dga.to_wkt(rounding_precision=-1)

        np.testing.assert_array_equal(
            values,
            np.asarray(
                [
                    "POINT (0 0)",
                    None,
                    "POINT EMPTY",
                    "LINESTRING (0 0, 2 4)",
                    "POINT (5 6)",
                ],
                dtype=object,
            ),
        )
        assert not any(event.kind == DiagnosticKind.MATERIALIZATION for event in dga._owned.diagnostics)

    @pytest.mark.skipif(not has_gpu_runtime(), reason="CUDA runtime not available")
    def test_to_wkb_device_resident_avoids_shapely(self, monkeypatch):
        from vibespatial import has_pylibcudf_support

        if not has_pylibcudf_support():
            pytest.skip("pylibcudf device WKB encoder not available")

        dga = _make_device_only_dga(
            [Point(0, 0), None, Point(), LineString([(0, 0), (2, 4)]), Point(5, 6)]
        )
        dga._owned.diagnostics.clear()

        def _fail_shapely_cache():
            raise AssertionError("to_wkb should not call _ensure_shapely_cache")

        monkeypatch.setattr(dga, "_ensure_shapely_cache", _fail_shapely_cache)

        values = dga.to_wkb()

        assert isinstance(values[0], bytes)
        assert values[1] is None
        assert isinstance(values[2], bytes)
        assert isinstance(values[3], bytes)
        assert not any(event.kind == DiagnosticKind.MATERIALIZATION for event in dga._owned.diagnostics)


# ---------------------------------------------------------------------------
# Equality
# ---------------------------------------------------------------------------

class TestEquality:
    def test_equal_arrays(self, points):
        dga1 = DeviceGeometryArray._from_sequence(points)
        dga2 = DeviceGeometryArray._from_sequence(points)
        result = dga1 == dga2
        assert result.all()

    def test_unequal_arrays(self):
        dga1 = DeviceGeometryArray._from_sequence([Point(0, 0)])
        dga2 = DeviceGeometryArray._from_sequence([Point(1, 1)])
        result = dga1 == dga2
        assert not result.any()


class TestPointCoordinateProperties:
    def test_x_y_use_owned_point_buffers_without_materialization(self):
        dga = _make_device_only_dga([Point(1, 2), Point(), None, Point(3, 4)])

        np.testing.assert_allclose(
            dga.x,
            np.asarray([1.0, np.nan, np.nan, 3.0], dtype=np.float64),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            dga.y,
            np.asarray([2.0, np.nan, np.nan, 4.0], dtype=np.float64),
            equal_nan=True,
        )
        assert dga._shapely_cache is None


# ---------------------------------------------------------------------------
# Predicates and constructive ops consume DGA
# ---------------------------------------------------------------------------

class TestBinaryPredicatesOwned:
    """Binary predicates on DeviceGeometryArray use owned engine without Shapely round-trip."""

    @pytest.fixture
    def left_dga(self):
        geoms = [box(0, 0, 2, 2), box(10, 10, 12, 12), box(5, 5, 7, 7)]
        return DeviceGeometryArray._from_sequence(geoms)

    @pytest.fixture
    def right_dga(self):
        return DeviceGeometryArray._from_sequence([Point(1, 1), Point(20, 20), Point(6, 6)])

    def test_intersects_no_dga_shapely_cache(self, left_dga, right_dga):
        """DGA predicate should NOT populate its own Shapely cache."""
        result = left_dga.intersects(right_dga)
        assert left_dga._shapely_cache is None
        assert list(result) == [True, False, True]

    def test_contains_no_dga_shapely_cache(self, left_dga, right_dga):
        result = left_dga.contains(right_dga)
        assert left_dga._shapely_cache is None
        assert list(result) == [True, False, True]

    def test_disjoint_no_dga_shapely_cache(self, left_dga, right_dga):
        result = left_dga.disjoint(right_dga)
        assert left_dga._shapely_cache is None
        assert list(result) == [False, True, False]

    def test_within_dga_vs_dga(self):
        inner = DeviceGeometryArray._from_sequence([Point(1, 1), Point(20, 20)])
        outer = DeviceGeometryArray._from_sequence([box(0, 0, 2, 2), box(0, 0, 2, 2)])
        result = inner.within(outer)
        assert list(result) == [True, False]

    def test_all_supported_predicates_produce_bool(self, left_dga, right_dga):
        predicates = [
            "intersects", "contains", "within", "touches", "crosses",
            "overlaps", "covers", "covered_by", "disjoint", "contains_properly",
        ]
        for pred in predicates:
            result = getattr(left_dga, pred)(right_dga)
            assert result.dtype == bool, f"{pred} did not return bool array"
            assert len(result) == len(left_dga), f"{pred} length mismatch"

    def test_predicate_dispatch_event_recorded(self, left_dga, right_dga):
        from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events
        clear_dispatch_events()
        _ = left_dga.intersects(right_dga)
        events = get_dispatch_events(clear=True)
        dga_events = [e for e in events if e.surface == "DeviceGeometryArray.intersects"]
        assert len(dga_events) == 1
        assert dga_events[0].implementation in {"owned_gpu_predicate", "owned_cpu_predicate"}

    def test_predicate_with_scalar_geometry(self, left_dga):
        result = left_dga.contains(Point(1, 1))
        assert result[0] is np.True_
        assert result[1] is np.False_

    def test_predicate_dga_vs_shapely_array(self, left_dga):
        right = np.array([Point(1, 1), Point(20, 20), Point(6, 6)], dtype=object)
        result = left_dga.intersects(right)
        assert list(result) == [True, False, True]

    def test_equals_uses_owned_path_for_device_peers(self, left_dga):
        geopandas.clear_fallback_events()
        result = left_dga.equals(left_dga)

        assert result.all()
        assert not geopandas.get_fallback_events(clear=True)

    def test_geom_equals_strict_native_accepts_host_geometry_array(self, monkeypatch):
        geopandas.clear_fallback_events()
        monkeypatch.setenv("VIBESPATIAL_STRICT_NATIVE", "1")
        left = DeviceGeometryArray._from_sequence([Point(0, 0), Point(1, 1)])
        right = GeometryArray(np.asarray([Point(0, 0), Point(1, 1)], dtype=object))

        result = left.geom_equals(right)

        assert result.all()
        assert not geopandas.get_fallback_events(clear=True)


class TestFallbackObservability:
    @pytest.mark.parametrize(
        ("method_name", "other_factory", "expected_surface"),
        [
            (
                "equals",
                lambda: GeometryCollection([Point(0, 0)]),
                "DeviceGeometryArray.equals",
            ),
            (
                "distance",
                lambda: GeometryCollection([Point(0, 0)]),
                "DeviceGeometryArray.distance",
            ),
            (
                "union",
                lambda: GeometryCollection([Point(0, 0)]),
                "DeviceGeometryArray.union",
            ),
        ],
    )
    def test_fallback_events_precede_shapely_materialization(
        self,
        monkeypatch: pytest.MonkeyPatch,
        method_name: str,
        other_factory,
        expected_surface: str,
    ) -> None:
        import vibespatial.geometry.device_array as device_array_module

        geopandas.clear_fallback_events()
        left = DeviceGeometryArray._from_sequence([Point(0, 0), Point(1, 1)])
        other = other_factory()

        original = device_array_module.owned_to_shapely

        def spy(owned, *args, **kwargs):
            events = geopandas.get_fallback_events()
            assert events, "expected a fallback event before Shapely materialization"
            assert events[-1].surface == expected_surface
            return original(owned, *args, **kwargs)

        monkeypatch.setattr(device_array_module, "owned_to_shapely", spy)

        result = getattr(left, method_name)(other)

        assert result is not None
        events = geopandas.get_fallback_events(clear=True)
        assert events[-1].surface == expected_surface

    def test_distance_strict_native_blocks_host_materialization(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import vibespatial.geometry.device_array as device_array_module

        geopandas.clear_fallback_events()
        monkeypatch.setenv("VIBESPATIAL_STRICT_NATIVE", "1")
        left = DeviceGeometryArray._from_sequence([Point(0, 0), Point(1, 1)])
        other = GeometryCollection([Point(0, 0)])

        monkeypatch.setattr(
            device_array_module,
            "owned_to_shapely",
            lambda *args, **kwargs: pytest.fail(
                "distance fallback should not materialize in strict-native mode"
            ),
        )

        with pytest.raises(StrictNativeFallbackError):
            left.distance(other)

        events = geopandas.get_fallback_events(clear=True)
        assert events[-1].surface == "DeviceGeometryArray.distance"


class TestClipByRectOwned:
    """clip_by_rect on DGA uses owned path and returns DGA."""

    def test_clip_returns_dga(self):
        dga = DeviceGeometryArray._from_sequence([Point(1, 1), Point(5, 5), Point(10, 10)])
        result = dga.clip_by_rect(0, 0, 6, 6)
        assert isinstance(result, DeviceGeometryArray)

    def test_clip_correct_results(self):
        import shapely as shp
        geoms = [Point(1, 1), Point(5, 5), Point(10, 10)]
        dga = DeviceGeometryArray._from_sequence(geoms)
        result = dga.clip_by_rect(0, 0, 6, 6)
        expected = shp.clip_by_rect(np.array(geoms, dtype=object), 0, 0, 6, 6)
        for i in range(len(geoms)):
            r = result[i]
            e = expected[i]
            r_empty = r is None or (hasattr(r, "is_empty") and r.is_empty)
            e_empty = e is None or (hasattr(e, "is_empty") and e.is_empty)
            if r_empty or e_empty:
                assert r_empty and e_empty, f"row {i}: {r} vs {e}"
            else:
                assert shp.equals(r, e)

    def test_clip_dispatch_event_recorded(self):
        from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events
        clear_dispatch_events()
        dga = DeviceGeometryArray._from_sequence([Point(1, 1), Point(5, 5)])
        _ = dga.clip_by_rect(0, 0, 3, 3)
        events = get_dispatch_events(clear=True)
        clip_events = [e for e in events if e.surface == "DeviceGeometryArray.clip_by_rect"]
        assert len(clip_events) == 1
        assert clip_events[0].implementation == "owned_clip_by_rect"

    def test_clip_unsupported_gpu_family_routes_to_owned_cpu_path(self):
        import shapely as shp
        from shapely.geometry import MultiPoint

        from vibespatial.runtime import ExecutionMode, set_requested_mode
        from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

        geoms = [MultiPoint([(1, 1), (5, 5)]), MultiPoint([(10, 10), (11, 11)])]
        dga = DeviceGeometryArray._from_sequence(geoms)

        clear_dispatch_events()
        with set_requested_mode(ExecutionMode.AUTO):
            result = dga.clip_by_rect(0, 0, 6, 6)

        expected = shp.clip_by_rect(np.array(geoms, dtype=object), 0, 0, 6, 6)
        for i in range(len(geoms)):
            r = result[i]
            e = expected[i]
            r_empty = r is None or (hasattr(r, "is_empty") and r.is_empty)
            e_empty = e is None or (hasattr(e, "is_empty") and e.is_empty)
            if r_empty or e_empty:
                assert r_empty and e_empty, f"row {i}: {r} vs {e}"
            else:
                assert shp.equals(r, e)

        events = get_dispatch_events(clear=True)
        clip_events = [e for e in events if e.surface == "DeviceGeometryArray.clip_by_rect"]
        assert len(clip_events) == 1
        assert clip_events[0].selected is ExecutionMode.CPU


class TestConstructiveOpsReturnDGA:
    """Constructive / unary ops return DGA to maintain device residency."""

    @pytest.fixture
    def poly_dga(self):
        return DeviceGeometryArray._from_sequence([
            Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
        ])

    def test_boundary_returns_dga(self, poly_dga):
        result = poly_dga.boundary
        assert isinstance(result, DeviceGeometryArray)
        assert len(result) == 2
        # boundary of a polygon round-trips as LineString through owned pipeline
        assert result[0].geom_type in ("LinearRing", "LineString")

    def test_centroid_returns_dga(self, poly_dga):
        result = poly_dga.centroid
        assert isinstance(result, DeviceGeometryArray)
        assert len(result) == 2
        assert result[0].geom_type == "Point"

    def test_convex_hull_returns_dga(self, poly_dga):
        result = poly_dga.convex_hull
        assert isinstance(result, DeviceGeometryArray)
        assert len(result) == 2

    def test_buffer_returns_dga(self, dga_points):
        result = dga_points.buffer(1.0)
        assert isinstance(result, DeviceGeometryArray)
        assert len(result) == 5
        assert result[0].geom_type == "Polygon"
        assert result._provenance is not None
        assert result._provenance.operation == "buffer"

    def test_linestring_buffer_preserves_provenance_through_geodataframe_assignment(self):
        import vibespatial.api as geopandas

        dga = DeviceGeometryArray._from_sequence(
            [
                LineString([(0, 0), (10, 0)]),
                LineString([(0, 5), (10, 5)]),
            ]
        )

        frame = geopandas.GeoDataFrame(
            geometry=geopandas.GeoSeries(dga, crs="EPSG:3857"),
            crs="EPSG:3857",
        )
        frame["geometry"] = frame.geometry.buffer(1.0)

        tag = getattr(frame.geometry.values, "_provenance", None)
        assert tag is not None
        assert tag.operation == "buffer"
        assert tag.source_array is not None

    def test_small_linestring_buffer_honors_auto_crossover(self, monkeypatch):
        import vibespatial.constructive.linestring as linestring_module

        dga = DeviceGeometryArray._from_sequence(
            [
                LineString([(0, 0), (5, 5), (10, 0)]),
                LineString([(10, 0), (15, 5), (20, 0)]),
            ]
        )

        def _fail(*_args, **_kwargs):
            raise AssertionError("small DGA linestring buffer should not force the GPU kernel")

        monkeypatch.setattr(linestring_module, "linestring_buffer_owned_array", _fail)

        result = dga.buffer(1.0)

        assert isinstance(result, DeviceGeometryArray)
        assert len(result) == 2
        materialized = np.asarray(result, dtype=object)
        assert bool(shapely.is_valid(materialized).all())

    @pytest.mark.skipif(not has_gpu_runtime(), reason="No GPU runtime available")
    def test_device_resident_small_linestring_buffer_stays_on_gpu(self, monkeypatch):
        import vibespatial.constructive.linestring as linestring_module
        import vibespatial.runtime.adaptive as adaptive_module
        from vibespatial.runtime import ExecutionMode
        from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

        dga = DeviceGeometryArray._from_owned(
            from_shapely_geometries(
                [
                    LineString([(0, 0), (5, 5), (10, 0)]),
                    LineString([(10, 0), (15, 5), (20, 0)]),
                ],
                residency=Residency.DEVICE,
            )
        )

        called: dict[str, object] = {}

        def _fake_linestring_buffer_owned_array(owned, *_args, **_kwargs):
            called["row_count"] = owned.row_count
            called["residency"] = owned.residency
            return from_shapely_geometries(
                [
                    Polygon([(0, 0), (10, 0), (10, 1), (0, 1), (0, 0)]),
                    Polygon([(10, 0), (20, 0), (20, 1), (10, 1), (10, 0)]),
                ],
                residency=Residency.DEVICE,
            )

        monkeypatch.setattr(
            adaptive_module,
            "plan_dispatch_selection",
            lambda *_args, **_kwargs: SimpleNamespace(
                selected=ExecutionMode.CPU,
                reason="forced cpu for test",
            ),
        )
        monkeypatch.setattr(
            linestring_module,
            "linestring_buffer_owned_array",
            _fake_linestring_buffer_owned_array,
        )

        clear_dispatch_events()
        result = dga.buffer(1.0)
        dispatch_events = get_dispatch_events(clear=True)

        assert isinstance(result, DeviceGeometryArray)
        assert getattr(result._owned, "residency", None) is Residency.DEVICE
        assert called["row_count"] == 2
        assert called["residency"] is Residency.DEVICE
        buffer_events = [event for event in dispatch_events if event.surface == "DeviceGeometryArray.buffer"]
        assert buffer_events
        assert buffer_events[-1].implementation == "linestring_buffer_owned_array"
        assert buffer_events[-1].selected is ExecutionMode.GPU

    def test_large_two_point_linestring_buffer_uses_specialized_gpu_route(self, monkeypatch):
        import vibespatial.constructive.linestring as linestring_module
        import vibespatial.runtime.adaptive as adaptive_module
        from vibespatial.runtime import ExecutionMode
        from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

        dga = DeviceGeometryArray._from_sequence(
            [
                LineString([(0.0, float(i)), (10.0, float(i))])
                for i in range(600)
            ]
        )

        called: dict[str, object] = {}

        def _fake_linestring_buffer_owned_array(owned, *_args, **_kwargs):
            called["row_count"] = owned.row_count
            return from_shapely_geometries(
                [
                    Polygon(
                        [
                            (0.0, float(i)),
                            (10.0, float(i)),
                            (10.0, float(i) + 1.0),
                            (0.0, float(i) + 1.0),
                            (0.0, float(i)),
                        ]
                    )
                    for i in range(owned.row_count)
                ]
            )

        monkeypatch.setattr(
            adaptive_module,
            "plan_dispatch_selection",
            lambda *_args, **_kwargs: SimpleNamespace(
                selected=ExecutionMode.CPU,
                reason="forced cpu for test",
            ),
        )
        monkeypatch.setattr(
            linestring_module,
            "linestring_buffer_owned_array",
            _fake_linestring_buffer_owned_array,
        )

        clear_dispatch_events()
        result = dga.buffer(1.0)
        dispatch_events = get_dispatch_events(clear=True)

        assert isinstance(result, DeviceGeometryArray)
        assert len(result) == 600
        assert called["row_count"] == 600
        buffer_events = [event for event in dispatch_events if event.surface == "DeviceGeometryArray.buffer"]
        assert buffer_events
        assert buffer_events[-1].implementation == "linestring_buffer_owned_array"
        assert buffer_events[-1].selected.value == "gpu"

    def test_two_point_linestring_buffer_honors_auto_crossover(self, monkeypatch):
        import vibespatial.constructive.linestring as linestring_module
        import vibespatial.runtime.adaptive as adaptive_module
        from vibespatial.runtime import ExecutionMode
        from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

        dga = DeviceGeometryArray._from_sequence(
            [
                LineString([(0, 0), (10, 0)]),
                LineString([(0, 5), (10, 5)]),
            ]
        )

        called: dict[str, object] = {}

        def _fake_linestring_buffer_owned_array(owned, *_args, **_kwargs):
            called["row_count"] = owned.row_count
            return from_shapely_geometries(
                [
                    Polygon([(0, 0), (10, 0), (10, 1), (0, 1), (0, 0)]),
                    Polygon([(0, 5), (10, 5), (10, 6), (0, 6), (0, 5)]),
                ]
            )

        monkeypatch.setattr(
            adaptive_module,
            "plan_dispatch_selection",
            lambda *_args, **_kwargs: SimpleNamespace(
                selected=ExecutionMode.CPU,
                reason="forced cpu for test",
            ),
        )
        monkeypatch.setattr(
            linestring_module,
            "linestring_buffer_owned_array",
            _fake_linestring_buffer_owned_array,
        )

        clear_dispatch_events()
        result = dga.buffer(1.0)
        dispatch_events = get_dispatch_events(clear=True)

        assert isinstance(result, DeviceGeometryArray)
        assert len(result) == 2
        assert dispatch_events
        assert "row_count" not in called
        buffer_events = [event for event in dispatch_events if event.surface == "DeviceGeometryArray.buffer"]
        assert buffer_events
        assert all(event.selected.value == "cpu" for event in buffer_events)

    def test_intersection_returns_dga(self, poly_dga):
        other = DeviceGeometryArray._from_sequence([
            Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
            Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
        ])
        result = poly_dga.intersection(other)
        assert isinstance(result, DeviceGeometryArray)
        assert len(result) == 2

    def test_intersection_with_device_backed_geoseries_uses_owned_path(
        self,
        monkeypatch,
        poly_dga,
    ):
        import vibespatial.geometry.device_array as device_array_module

        right_owned = from_shapely_geometries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
            ],
            residency=Residency.DEVICE if has_gpu_runtime() else Residency.HOST,
        )
        right = geopandas.GeoSeries(
            DeviceGeometryArray._from_owned(right_owned, crs="EPSG:3857"),
            crs="EPSG:3857",
        )

        def _unexpected_fallback(**_kwargs):
            raise AssertionError("device-backed GeoSeries should not fall back to Shapely")

        monkeypatch.setattr(
            device_array_module,
            "_record_shapely_fallback_event",
            _unexpected_fallback,
        )

        result = poly_dga.intersection(right)

        assert isinstance(result, DeviceGeometryArray)
        assert len(result) == 2

    def test_intersection_with_public_geometryarray_geoseries_uses_to_owned_path(
        self,
        monkeypatch,
        poly_dga,
    ):
        import vibespatial.geometry.device_array as device_array_module

        right = geopandas.GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
            ],
            crs="EPSG:3857",
        )
        assert getattr(right.values, "_owned", None) is None
        assert hasattr(right.values, "to_owned")

        def _unexpected_fallback(**_kwargs):
            raise AssertionError(
                "public GeometryArray-backed GeoSeries should coerce via to_owned()"
            )

        monkeypatch.setattr(
            device_array_module,
            "_record_shapely_fallback_event",
            _unexpected_fallback,
        )

        result = poly_dga.intersection(right)

        assert isinstance(result, DeviceGeometryArray)
        assert len(result) == 2

    def test_intersection_falls_back_to_host_geometry_array_for_geometrycollection_output(self):
        left_owned = from_shapely_geometries(
            [Polygon([(0, 0), (3, 0), (3, 3), (0, 0)])],
            residency=Residency.DEVICE if has_gpu_runtime() else Residency.HOST,
        )
        left = geopandas.GeoSeries(
            DeviceGeometryArray._from_owned(left_owned, crs="EPSG:3857"),
            crs="EPSG:3857",
        )
        right = geopandas.GeoSeries(
            GeometryArray(
                np.asarray(
                    [
                        GeometryCollection(
                            [
                                Polygon([(1, 0), (3, 0), (3, 2), (1, 0)]),
                                LineString([(0, 1), (3, 1)]),
                            ]
                        )
                    ],
                    dtype=object,
                ),
                crs="EPSG:3857",
            ),
            crs="EPSG:3857",
        )

        result = left.intersection(right)

        assert isinstance(result.values, GeometryArray)
        assert result.iloc[0].geom_type == "GeometryCollection"

    def test_union_returns_dga(self, poly_dga):
        other = DeviceGeometryArray._from_sequence([
            Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
            Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
        ])
        result = poly_dga.union(other)
        assert isinstance(result, DeviceGeometryArray)
        assert len(result) == 2

    def test_difference_returns_dga(self, poly_dga):
        other = DeviceGeometryArray._from_sequence([
            Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
            Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
        ])
        result = poly_dga.difference(other)
        assert isinstance(result, DeviceGeometryArray)
        assert len(result) == 2

    def test_symmetric_difference_returns_dga(self, poly_dga):
        other = DeviceGeometryArray._from_sequence([
            Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
            Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
        ])
        result = poly_dga.symmetric_difference(other)
        assert isinstance(result, DeviceGeometryArray)
        assert len(result) == 2


class TestChainedOperations:
    """Chained operations maintain device residency (DGA → DGA)."""

    def test_predicate_after_clip(self):
        dga = DeviceGeometryArray._from_sequence([Point(1, 1), Point(5, 5), Point(10, 10)])
        clipped = dga.clip_by_rect(0, 0, 6, 6)
        assert isinstance(clipped, DeviceGeometryArray)
        # Use predicate on clipped result
        other = DeviceGeometryArray._from_sequence([Point(1, 1), Point(5, 5), Point(10, 10)])
        result = clipped.intersects(other)
        assert result.dtype == bool

    def test_buffer_then_centroid(self):
        dga = DeviceGeometryArray._from_sequence([Point(0, 0), Point(5, 5)])
        buffered = dga.buffer(1.0)
        assert isinstance(buffered, DeviceGeometryArray)
        centroids = buffered.centroid
        assert isinstance(centroids, DeviceGeometryArray)
        assert centroids[0].geom_type == "Point"

    def test_convex_hull_then_intersects(self):
        dga = DeviceGeometryArray._from_sequence([
            MultiPoint([(0, 0), (2, 0), (1, 2)]),
            MultiPoint([(10, 10), (12, 10), (11, 12)]),
        ])
        hulls = dga.convex_hull
        assert isinstance(hulls, DeviceGeometryArray)
        result = hulls.contains(DeviceGeometryArray._from_sequence([Point(1, 1), Point(20, 20)]))
        assert result.dtype == bool
        assert list(result) == [True, False]

    def test_remove_repeated_points_returns_dga(self):
        polygon = Polygon(
            [
                (0.0, 0.0),
                (3.0, 0.0),
                (3.0, 0.0),
                (3.0, 2.0),
                (0.0, 2.0),
                (0.0, 0.0),
            ]
        )
        dga = DeviceGeometryArray._from_sequence([polygon])

        result = dga.remove_repeated_points(0.0)

        assert isinstance(result, DeviceGeometryArray)
        expected = shapely.remove_repeated_points(
            np.asarray([polygon], dtype=object),
            tolerance=0.0,
        )
        assert result[0].equals(expected[0])
