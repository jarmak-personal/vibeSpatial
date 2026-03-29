"""Tests for GPU-accelerated PostGIS read/write via ADBC.

Unit tests (no database needed) verify:
  - SQL rewriting logic
  - Connection URI extraction
  - Arrow table → GeoDataFrame assembly (WKB binary column → GPU decode)
  - Bare table name detection

Integration tests (marked @pytest.mark.postgis, skipped unless
VIBESPATIAL_TEST_POSTGIS_URI env var is set) verify:
  - Full round-trip: write GeoDataFrame → read back → geometry equality
  - CRS auto-detection from PostGIS metadata
  - if_exists modes: fail, replace, append
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
import pytest
from shapely import to_wkb
from shapely.geometry import LineString, Point, Polygon

from vibespatial.io.postgis_gpu import (
    _arrow_table_to_geodataframe,
    _get_connection_uri,
    _is_simple_table_name,
    _wrap_sql_for_wkb,
    read_postgis_gpu,
    to_postgis_gpu,
)
from vibespatial.runtime._runtime import has_gpu_runtime

needs_gpu = pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime not available")

_POSTGIS_URI = os.environ.get("VIBESPATIAL_TEST_POSTGIS_URI")
needs_postgis = pytest.mark.skipif(
    _POSTGIS_URI is None,
    reason="VIBESPATIAL_TEST_POSTGIS_URI not set",
)


# ---------------------------------------------------------------------------
# SQL rewriting
# ---------------------------------------------------------------------------


class TestSqlRewriting:
    """Verify SQL wrapping produces correct sub-query."""

    def test_simple_select(self) -> None:
        sql = "SELECT id, geom FROM my_table"
        wrapped = _wrap_sql_for_wkb(sql, "geom")
        assert "__vibes_wkb" in wrapped
        assert 'ST_AsBinary("geom")' in wrapped
        assert "__vibes_sub" in wrapped

    def test_bare_table(self) -> None:
        sql = "my_table"
        wrapped = _wrap_sql_for_wkb(sql, "geom")
        assert "__vibes_wkb" in wrapped
        assert "FROM (my_table) AS __vibes_sub" in wrapped

    def test_custom_geom_col(self) -> None:
        sql = "SELECT * FROM parcels"
        wrapped = _wrap_sql_for_wkb(sql, "the_geom")
        assert 'ST_AsBinary("the_geom")' in wrapped

    def test_complex_sql(self) -> None:
        sql = (
            "SELECT a.id, a.geom, b.name "
            "FROM table_a a JOIN table_b b ON a.id = b.fk"
        )
        wrapped = _wrap_sql_for_wkb(sql, "geom")
        assert "__vibes_sub" in wrapped
        assert 'ST_AsBinary("geom")' in wrapped


# ---------------------------------------------------------------------------
# Bare table name detection
# ---------------------------------------------------------------------------


class TestSimpleTableDetection:
    """Verify bare table name regex."""

    def test_bare_table(self) -> None:
        assert _is_simple_table_name("my_table") is True

    def test_schema_table(self) -> None:
        assert _is_simple_table_name("public.my_table") is True

    def test_select_query(self) -> None:
        assert _is_simple_table_name("SELECT * FROM t") is False

    def test_with_whitespace(self) -> None:
        assert _is_simple_table_name("  my_table  ") is True

    def test_empty(self) -> None:
        assert _is_simple_table_name("") is False


# ---------------------------------------------------------------------------
# Connection URI extraction
# ---------------------------------------------------------------------------


class TestConnectionUriExtraction:
    """Verify URI extraction from various connection objects."""

    def test_string_passthrough(self) -> None:
        uri = "postgresql://user:pass@host/db"
        assert _get_connection_uri(uri) == uri

    def test_sqlalchemy_engine_mock(self) -> None:
        engine = MagicMock()
        engine.url = "postgresql://user:pass@host/db"
        assert _get_connection_uri(engine) == "postgresql://user:pass@host/db"

    def test_sqlalchemy_connection_mock(self) -> None:
        conn = MagicMock(spec=[])  # no .url attribute
        conn.engine = MagicMock()
        conn.engine.url = "postgresql://user:pass@host/db"
        assert _get_connection_uri(conn) == "postgresql://user:pass@host/db"

    def test_unsupported_object(self) -> None:
        assert _get_connection_uri(42) is None

    def test_none(self) -> None:
        assert _get_connection_uri(None) is None


# ---------------------------------------------------------------------------
# Arrow table → GeoDataFrame assembly (GPU WKB decode)
# ---------------------------------------------------------------------------


class TestArrowWkbToGeoDataFrame:
    """Verify that a PyArrow table with a WKB binary column is correctly
    decoded via GPU and merged into a GeoDataFrame."""

    @needs_gpu
    def test_points(self) -> None:
        """Arrow table with Point WKB → GeoDataFrame with correct geometry."""
        wkb_data = [to_wkb(Point(i, i * 2)) for i in range(5)]
        table = pa.table({
            "id": [1, 2, 3, 4, 5],
            "name": ["a", "b", "c", "d", "e"],
            "geom": wkb_data,
        })

        gdf = _arrow_table_to_geodataframe(table, geom_col="geom", crs="EPSG:4326")

        assert len(gdf) == 5
        assert "id" in gdf.columns
        assert "name" in gdf.columns
        assert gdf.crs is not None
        assert gdf.crs.to_epsg() == 4326

        # Verify geometry values
        for i in range(5):
            pt = gdf.geometry.iloc[i]
            assert pt.equals(Point(i, i * 2)), f"Row {i}: expected {Point(i, i*2)}, got {pt}"

    @needs_gpu
    def test_polygons(self) -> None:
        """Arrow table with Polygon WKB → GeoDataFrame."""
        polys = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
            Polygon([(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)]),
        ]
        wkb_data = [to_wkb(p) for p in polys]
        table = pa.table({
            "area_id": [10, 20],
            "geom": wkb_data,
        })

        gdf = _arrow_table_to_geodataframe(table, geom_col="geom", crs=None)

        assert len(gdf) == 2
        assert "area_id" in gdf.columns
        for i, expected in enumerate(polys):
            assert gdf.geometry.iloc[i].equals_exact(expected, 1e-9)

    @needs_gpu
    def test_linestrings(self) -> None:
        """Arrow table with LineString WKB → GeoDataFrame."""
        lines = [
            LineString([(0, 0), (1, 1), (2, 0)]),
            LineString([(10, 10), (20, 20)]),
        ]
        wkb_data = [to_wkb(ls) for ls in lines]
        table = pa.table({
            "route_id": [100, 200],
            "geom": wkb_data,
        })

        gdf = _arrow_table_to_geodataframe(table, geom_col="geom", crs="EPSG:32632")

        assert len(gdf) == 2
        assert gdf.crs.to_epsg() == 32632
        for i, expected in enumerate(lines):
            assert gdf.geometry.iloc[i].equals_exact(expected, 1e-9)

    @needs_gpu
    def test_null_rows(self) -> None:
        """Arrow table with null geometry rows produces correct GeoDataFrame."""
        wkb_data = [to_wkb(Point(1, 2)), None, to_wkb(Point(3, 4))]
        table = pa.table({
            "id": [1, 2, 3],
            "geom": pa.array(wkb_data, type=pa.binary()),
        })

        gdf = _arrow_table_to_geodataframe(table, geom_col="geom", crs=None)

        assert len(gdf) == 3
        assert gdf.geometry.iloc[0].equals(Point(1, 2))
        assert gdf.geometry.iloc[1] is None or gdf.geometry.isna().iloc[1]
        assert gdf.geometry.iloc[2].equals(Point(3, 4))

    @needs_gpu
    def test_hex_string_wkb(self) -> None:
        """Arrow table with hex-encoded WKB strings is handled correctly."""
        wkb_hex = [to_wkb(Point(i, i)).hex() for i in range(3)]
        table = pa.table({
            "id": [1, 2, 3],
            "geom": pa.array(wkb_hex, type=pa.string()),
        })

        gdf = _arrow_table_to_geodataframe(table, geom_col="geom", crs=None)

        assert len(gdf) == 3
        for i in range(3):
            assert gdf.geometry.iloc[i].equals(Point(i, i))

    @needs_gpu
    def test_missing_geom_col_raises(self) -> None:
        """Requesting a non-existent geometry column raises ValueError."""
        table = pa.table({"id": [1, 2], "data": [10, 20]})
        with pytest.raises(ValueError, match="not found"):
            _arrow_table_to_geodataframe(table, geom_col="geom", crs=None)

    @needs_gpu
    def test_no_crs(self) -> None:
        """When crs=None and no detection is possible, GeoDataFrame has no CRS."""
        wkb_data = [to_wkb(Point(0, 0))]
        table = pa.table({"geom": wkb_data})

        gdf = _arrow_table_to_geodataframe(table, geom_col="geom", crs=None)

        assert gdf.crs is None

    @needs_gpu
    def test_many_columns_preserved(self) -> None:
        """All non-geometry columns are preserved in the output."""
        wkb_data = [to_wkb(Point(0, 0))] * 3
        table = pa.table({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
            "c": [1.1, 2.2, 3.3],
            "geom": wkb_data,
        })

        gdf = _arrow_table_to_geodataframe(table, geom_col="geom", crs=None)

        assert list(gdf.columns) == ["a", "b", "c", "geometry"]
        np.testing.assert_array_equal(gdf["a"].values, [1, 2, 3])
        np.testing.assert_array_equal(gdf["b"].values, ["x", "y", "z"])


# ---------------------------------------------------------------------------
# Graceful fallback when ADBC is missing
# ---------------------------------------------------------------------------


class TestGracefulFallback:
    """GPU functions return None/False when ADBC is unavailable."""

    def test_read_returns_none_for_bad_connection(self) -> None:
        """read_postgis_gpu returns None when connection object is unsupported."""
        result = read_postgis_gpu("SELECT 1", 42, geom_col="geom")
        assert result is None

    def test_write_returns_false_for_bad_connection(self) -> None:
        """to_postgis_gpu returns False when connection object is unsupported."""
        import vibespatial.api as geopandas

        gdf = geopandas.GeoDataFrame(
            {"id": [1]},
            geometry=geopandas.GeoSeries.from_wkt(["POINT (0 0)"]),
        )
        result = to_postgis_gpu(gdf, "test_table", 42)
        assert result is False

    def test_read_returns_none_for_chunked(self) -> None:
        """Chunked reading is not supported; should return None."""
        result = read_postgis_gpu(
            "my_table",
            "postgresql://user:pass@host/db",
            chunksize=100,
        )
        assert result is None


# ---------------------------------------------------------------------------
# Integration tests (require PostGIS)
# ---------------------------------------------------------------------------


@pytest.mark.postgis
class TestPostgisIntegration:
    """Full round-trip tests against a live PostGIS database.

    Requires VIBESPATIAL_TEST_POSTGIS_URI to be set, e.g.:
      export VIBESPATIAL_TEST_POSTGIS_URI="postgresql://user:pass@localhost:5432/testdb"
    """

    @needs_gpu
    @needs_postgis
    def test_round_trip_points(self) -> None:
        """Write Points to PostGIS → read back → geometry equality."""
        import vibespatial.api as geopandas

        gdf = geopandas.GeoDataFrame(
            {"id": [1, 2, 3], "val": [10.0, 20.0, 30.0]},
            geometry=geopandas.GeoSeries(
                [Point(1, 2), Point(3, 4), Point(5, 6)],
                crs="EPSG:4326",
            ),
        )

        table_name = "_vibes_test_round_trip_points"
        to_postgis_gpu(gdf, table_name, _POSTGIS_URI, if_exists="replace")

        result = read_postgis_gpu(
            f"SELECT * FROM {table_name}",
            _POSTGIS_URI,
            geom_col="geometry",
        )
        assert result is not None
        assert len(result) == 3
        for i in range(3):
            assert result.geometry.iloc[i].equals_exact(gdf.geometry.iloc[i], 1e-9)

    @needs_gpu
    @needs_postgis
    def test_round_trip_polygons(self) -> None:
        """Write Polygons → read back → geometry equality."""
        import vibespatial.api as geopandas

        polys = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
            Polygon([(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)]),
        ]
        gdf = geopandas.GeoDataFrame(
            {"zone": ["a", "b"]},
            geometry=geopandas.GeoSeries(polys, crs="EPSG:4326"),
        )

        table_name = "_vibes_test_round_trip_polygons"
        to_postgis_gpu(gdf, table_name, _POSTGIS_URI, if_exists="replace")

        result = read_postgis_gpu(
            f"SELECT * FROM {table_name}",
            _POSTGIS_URI,
            geom_col="geometry",
        )
        assert result is not None
        assert len(result) == 2
        for i in range(2):
            assert result.geometry.iloc[i].equals_exact(polys[i], 1e-9)

    @needs_gpu
    @needs_postgis
    def test_if_exists_fail(self) -> None:
        """if_exists='fail' raises when table already exists."""
        import vibespatial.api as geopandas

        gdf = geopandas.GeoDataFrame(
            {"id": [1]},
            geometry=geopandas.GeoSeries([Point(0, 0)], crs="EPSG:4326"),
        )

        table_name = "_vibes_test_if_exists_fail"
        to_postgis_gpu(gdf, table_name, _POSTGIS_URI, if_exists="replace")

        # Second write with if_exists='fail' should return False
        result = to_postgis_gpu(gdf, table_name, _POSTGIS_URI, if_exists="fail")
        assert result is False

    @needs_gpu
    @needs_postgis
    def test_if_exists_append(self) -> None:
        """if_exists='append' adds rows to existing table."""
        import vibespatial.api as geopandas

        gdf = geopandas.GeoDataFrame(
            {"id": [1]},
            geometry=geopandas.GeoSeries([Point(0, 0)], crs="EPSG:4326"),
        )

        table_name = "_vibes_test_if_exists_append"
        to_postgis_gpu(gdf, table_name, _POSTGIS_URI, if_exists="replace")
        to_postgis_gpu(gdf, table_name, _POSTGIS_URI, if_exists="append")

        result = read_postgis_gpu(
            f"SELECT * FROM {table_name}",
            _POSTGIS_URI,
            geom_col="geometry",
        )
        assert result is not None
        assert len(result) == 2

    @needs_gpu
    @needs_postgis
    def test_crs_auto_detection(self) -> None:
        """CRS is auto-detected from PostGIS metadata for bare table reads."""
        import vibespatial.api as geopandas

        gdf = geopandas.GeoDataFrame(
            {"id": [1]},
            geometry=geopandas.GeoSeries([Point(0, 0)], crs="EPSG:4326"),
        )

        table_name = "_vibes_test_crs_auto"
        to_postgis_gpu(gdf, table_name, _POSTGIS_URI, if_exists="replace")

        result = read_postgis_gpu(
            table_name,
            _POSTGIS_URI,
            geom_col="geometry",
        )
        assert result is not None
        assert result.crs is not None
        assert result.crs.to_epsg() == 4326
