"""ADR-0042 result-boundary contract tests.

These tests cover the current compatibility seams where low-level query kernels
produce native relation results and GeoPandas assembly happens only at the
explicit export boundary.
"""
from __future__ import annotations

import importlib

import numpy as np
import pyarrow as pa
import pytest
from shapely.geometry import Point, box

import vibespatial.api as geopandas
import vibespatial.api._native_results as native_results_module
from vibespatial import write_geoparquet
from vibespatial.api import GeoDataFrame, GeoSeries
from vibespatial.api._native_results import (
    NativeAttributeTable,
    NativeTabularResult,
    RelationIndexResult,
    RelationJoinExportResult,
    RelationJoinResult,
    to_native_tabular_result,
)
from vibespatial.api.testing import assert_geodataframe_equal
from vibespatial.api.tools.clip import evaluate_geopandas_clip_native
from vibespatial.api.tools.sjoin import (
    _frame_join,
    _frame_join_from_relation_result,
    _nearest_query,
    _sjoin_export_result,
    _sjoin_nearest_export_result,
    _sjoin_nearest_relation_result,
    _sjoin_relation_result,
)
from vibespatial.io.file import write_vector_file
from vibespatial.overlay.dissolve import (
    evaluate_geopandas_dissolve,
    evaluate_geopandas_dissolve_native,
)
from vibespatial.runtime import has_gpu_runtime
from vibespatial.runtime.fallbacks import StrictNativeFallbackError
from vibespatial.spatial.query import build_owned_spatial_index, query_spatial_index
from vibespatial.testing import strict_native_environment

clip_module = importlib.import_module("vibespatial.api.tools.clip")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def left_gdf():
    """Left GeoDataFrame with string, int, and float attribute columns."""
    return GeoDataFrame(
        {
            "name": ["a", "b", "c"],
            "value_int": [10, 20, 30],
            "value_float": [1.1, 2.2, 3.3],
            "geometry": GeoSeries([box(0, 0, 2, 2), box(1, 1, 3, 3), box(4, 4, 6, 6)]),
        }
    )


@pytest.fixture
def right_gdf():
    """Right GeoDataFrame with string, int, and float attribute columns."""
    return GeoDataFrame(
        {
            "label": ["x", "y"],
            "count": [100, 200],
            "score": [9.9, 8.8],
            "geometry": GeoSeries([box(1, 1, 3, 3), box(5, 5, 7, 7)]),
        }
    )


# ---------------------------------------------------------------------------
# Phase 1: Spatial query returns integer numpy arrays
# ---------------------------------------------------------------------------

class TestSpatialQueryReturnsIntegerArrays:
    PREDICATES = ["intersects", "contains", "within", "touches", "covers", "covered_by"]

    @pytest.mark.parametrize("predicate", PREDICATES)
    def test_spatial_query_returns_integer_numpy_arrays(self, predicate):
        tree = np.asarray([box(0, 0, 2, 2), box(1, 1, 3, 3), box(10, 10, 11, 11)], dtype=object)
        query = np.asarray([box(0.5, 0.5, 1.5, 1.5)], dtype=object)
        owned, flat = build_owned_spatial_index(tree)

        result = query_spatial_index(
            owned, flat, query, predicate=predicate, sort=True
        )
        assert isinstance(result, np.ndarray), f"Expected ndarray, got {type(result)}"
        assert np.issubdtype(result.dtype, np.integer), (
            f"Expected integer dtype for predicate={predicate}, got {result.dtype}"
        )


class TestSpatialNearestReturnsIndexArrays:
    def test_spatial_nearest_returns_index_arrays_and_distances(self):
        """sjoin_nearest produces results with correct index types."""
        left = GeoDataFrame(
            {"a": [1, 2]},
            geometry=GeoSeries([Point(0, 0), Point(5, 0)]),
        )
        right = GeoDataFrame(
            {"b": [10, 20, 30]},
            geometry=GeoSeries([Point(1, 0), Point(6, 0), Point(20, 0)]),
        )
        result = geopandas.sjoin_nearest(left, right, how="inner", distance_col="dist")
        assert "dist" in result.columns
        assert result["dist"].dtype == np.float64
        # index_right should be integer-typed
        assert "index_right" in result.columns
        assert np.issubdtype(result["index_right"].dtype, np.integer)

    def test_relation_result_wrapper_matches_direct_frame_join_for_nearest(self):
        left = GeoDataFrame(
            {"a": [1, 2]},
            geometry=GeoSeries([Point(0, 0), Point(1, 1)]),
        )
        right = GeoDataFrame(
            {"b": [10]},
            geometry=GeoSeries([Point(1, 1)]),
        )

        indices, distances, _selected = _nearest_query(
            left,
            right,
            max_distance=None,
            how="inner",
            return_distance=False,
            exclusive=False,
        )

        direct, _ = _frame_join(
            left,
            right,
            indices,
            distances,
            "inner",
            "left",
            "right",
            None,
        )
        wrapped, _ = _frame_join_from_relation_result(
            left,
            right,
            RelationIndexResult(*indices),
            distances,
            "inner",
            "left",
            "right",
            None,
        )

        assert_geodataframe_equal(direct, wrapped)

    def test_sjoin_native_result_defers_frame_materialization(self, monkeypatch):
        left = GeoDataFrame(
            {"a": [1, 2]},
            geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
            crs="EPSG:4326",
        )
        right = GeoDataFrame(
            {"b": [10, 20]},
            geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
            crs="EPSG:4326",
        )

        real_materialize = RelationJoinResult.materialize
        materialize_calls = 0

        def _counting_materialize(self, *args, **kwargs):
            nonlocal materialize_calls
            materialize_calls += 1
            return real_materialize(self, *args, **kwargs)

        monkeypatch.setattr(
            RelationJoinResult,
            "materialize",
            _counting_materialize,
        )

        native_result, _implementation, _execution = _sjoin_relation_result(
            left,
            right,
            "intersects",
            None,
        )

        assert isinstance(native_result, RelationJoinResult)
        assert materialize_calls == 0

        def _fail(*_args, **_kwargs):
            raise AssertionError(
                "native sjoin GeoDataFrame export should not require relation materialization"
            )

        monkeypatch.setattr(RelationJoinResult, "materialize", _fail)
        monkeypatch.setattr(
            native_results_module,
            "_materialize_relation_join_parts",
            _fail,
        )

        materialized = native_result.to_geodataframe(
            left,
            right,
            how="inner",
            lsuffix="left",
            rsuffix="right",
        )
        wrapped = geopandas.sjoin(left, right)

        assert_geodataframe_equal(materialized, wrapped)

    def test_sjoin_export_result_defers_frame_materialization(self, monkeypatch):
        left = GeoDataFrame(
            {"a": [1, 2]},
            geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
            crs="EPSG:4326",
        )
        right = GeoDataFrame(
            {"b": [10, 20]},
            geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
            crs="EPSG:4326",
        )

        real_materialize = RelationJoinResult.materialize
        materialize_calls = 0

        def _counting_materialize(self, *args, **kwargs):
            nonlocal materialize_calls
            materialize_calls += 1
            return real_materialize(self, *args, **kwargs)

        monkeypatch.setattr(
            RelationJoinResult,
            "materialize",
            _counting_materialize,
        )

        export_result, _implementation, _execution = _sjoin_export_result(
            left,
            right,
            "inner",
            "intersects",
            None,
            "left",
            "right",
        )

        assert isinstance(export_result, RelationJoinExportResult)
        assert materialize_calls == 0

        def _fail(*_args, **_kwargs):
            raise AssertionError(
                "native sjoin GeoDataFrame export should not require relation materialization"
            )

        monkeypatch.setattr(RelationJoinResult, "materialize", _fail)
        monkeypatch.setattr(
            native_results_module,
            "_materialize_relation_join_parts",
            _fail,
        )

        materialized = export_result.to_geodataframe()
        wrapped = geopandas.sjoin(left, right)

        assert_geodataframe_equal(materialized, wrapped)

    def test_sjoin_export_result_writes_without_frame_materialization(
        self,
        monkeypatch,
        tmp_path,
    ) -> None:
        left = GeoDataFrame(
            {"a": [1, 2]},
            geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
            crs="EPSG:4326",
        )
        right = GeoDataFrame(
            {"b": [10, 20]},
            geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
            crs="EPSG:4326",
        )
        export_result, _implementation, _execution = _sjoin_export_result(
            left,
            right,
            "inner",
            "intersects",
            None,
            "left",
            "right",
        )
        expected = geopandas.sjoin(left, right)

        def _fail(*_args, **_kwargs):
            raise AssertionError(
                "native sjoin GeoParquet write should not require GeoDataFrame export"
            )

        monkeypatch.setattr(
            RelationJoinExportResult,
            "to_geodataframe",
            _fail,
        )

        path = tmp_path / "sjoin-native.parquet"
        write_geoparquet(export_result, path, geometry_encoding="geoarrow")

        result = geopandas.read_parquet(path)
        assert_geodataframe_equal(result, expected)

    def test_sjoin_export_result_writes_without_relation_materialization(
        self,
        monkeypatch,
        tmp_path,
    ) -> None:
        left = GeoDataFrame(
            {"a": [1, 2]},
            geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
            crs="EPSG:4326",
        )
        right = GeoDataFrame(
            {"b": [10, 20]},
            geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
            crs="EPSG:4326",
        )
        export_result, _implementation, _execution = _sjoin_export_result(
            left,
            right,
            "inner",
            "intersects",
            None,
            "left",
            "right",
        )
        expected = geopandas.sjoin(left, right)

        def _fail(*_args, **_kwargs):
            raise AssertionError(
                "native sjoin GeoParquet write should not require relation materialization"
            )

        monkeypatch.setattr(RelationJoinResult, "materialize", _fail)
        monkeypatch.setattr(NativeTabularResult, "to_geodataframe", _fail)

        path = tmp_path / "sjoin-native-no-materialize.parquet"
        write_geoparquet(export_result, path, geometry_encoding="geoarrow")
        monkeypatch.undo()

        result = geopandas.read_parquet(path)
        assert_geodataframe_equal(result, expected)

    def test_sjoin_export_result_builds_native_tabular_result(self) -> None:
        left = GeoDataFrame(
            {"a": [1, 2]},
            geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
        )
        right = GeoDataFrame(
            {"b": [10, 20]},
            geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
        )

        export_result, _implementation, _execution = _sjoin_export_result(
            left,
            right,
            "inner",
            "intersects",
            None,
            "left",
            "right",
        )

        native_result = to_native_tabular_result(export_result)

        assert isinstance(native_result, NativeTabularResult)
        assert native_result.geometry_name == "geometry"
        assert_geodataframe_equal(
            native_result.to_geodataframe(),
            geopandas.sjoin(left, right),
        )

    def test_sjoin_export_result_native_tabular_skips_join_frame_materializer(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        left = GeoDataFrame(
            {"a": [1, 2]},
            geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
        )
        right = GeoDataFrame(
            {"b": [10, 20]},
            geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
        )

        export_result, _implementation, _execution = _sjoin_export_result(
            left,
            right,
            "inner",
            "intersects",
            None,
            "left",
            "right",
        )
        expected = pa.table(geopandas.sjoin(left, right).to_arrow(geometry_encoding="WKB"))

        real_native_parts = native_results_module._native_relation_join_parts
        native_calls = 0

        def _counting_native_parts(*args, **kwargs):
            nonlocal native_calls
            native_calls += 1
            return real_native_parts(*args, **kwargs)

        def _fail(*_args, **_kwargs):
            raise AssertionError(
                "native sjoin tabular export should not require joined-frame materialization"
            )

        monkeypatch.setattr(
            native_results_module,
            "_native_relation_join_parts",
            _counting_native_parts,
        )
        monkeypatch.setattr(
            native_results_module,
            "_materialize_relation_join_parts",
            _fail,
        )

        native_result = to_native_tabular_result(export_result)

        assert native_calls == 1
        assert isinstance(native_result, NativeTabularResult)
        assert isinstance(native_result.attributes, NativeAttributeTable)
        assert native_result.attributes.arrow_table is not None
        result = pa.table(native_result.to_arrow(geometry_encoding="WKB"))
        assert result.column_names == expected.column_names
        assert result.to_pydict() == expected.to_pydict()

    def test_sjoin_export_result_builds_arrow_without_frame_materialization(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        left = GeoDataFrame(
            {"a": [1, 2]},
            geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
        )
        right = GeoDataFrame(
            {"b": [10, 20]},
            geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
        )

        export_result, _implementation, _execution = _sjoin_export_result(
            left,
            right,
            "inner",
            "intersects",
            None,
            "left",
            "right",
        )
        native_result = to_native_tabular_result(export_result)
        expected = pa.table(geopandas.sjoin(left, right).to_arrow(geometry_encoding="WKB"))

        assert isinstance(native_result, NativeTabularResult)

        def _fail(*_args, **_kwargs):
            raise AssertionError("native Arrow export should not require GeoDataFrame export")

        monkeypatch.setattr(NativeTabularResult, "to_geodataframe", _fail)

        result = pa.table(native_result.to_arrow(geometry_encoding="WKB"))

        assert result.column_names == expected.column_names
        assert result.to_pydict() == expected.to_pydict()

    def test_sjoin_export_result_native_tabular_writes_feather_without_frame_materialization(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        left = GeoDataFrame(
            {"a": [1, 2]},
            geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
        )
        right = GeoDataFrame(
            {"b": [10, 20]},
            geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
        )

        export_result, _implementation, _execution = _sjoin_export_result(
            left,
            right,
            "inner",
            "intersects",
            None,
            "left",
            "right",
        )
        native_result = to_native_tabular_result(export_result)
        expected = geopandas.sjoin(left, right)

        assert isinstance(native_result, NativeTabularResult)

        def _fail(*_args, **_kwargs):
            raise AssertionError("native Feather write should not require GeoDataFrame export")

        monkeypatch.setattr(NativeTabularResult, "to_geodataframe", _fail)

        path = tmp_path / "sjoin-native-no-materialize.feather"
        native_result.to_feather(path)

        result = geopandas.read_feather(path)
        assert_geodataframe_equal(result, expected)

    def test_sjoin_export_result_writes_file_without_frame_materialization(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        left = GeoDataFrame(
            {"a": [1, 2]},
            geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
            crs="EPSG:4326",
        )
        right = GeoDataFrame(
            {"b": [10, 20]},
            geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
            crs="EPSG:4326",
        )
        export_result, _implementation, _execution = _sjoin_export_result(
            left,
            right,
            "inner",
            "intersects",
            None,
            "left",
            "right",
        )
        expected = geopandas.sjoin(left, right)

        def _fail(*_args, **_kwargs):
            raise AssertionError("native file export should not require GeoDataFrame export")

        monkeypatch.setattr(NativeTabularResult, "to_geodataframe", _fail)

        path = tmp_path / "sjoin-native-no-materialize.geojson"
        write_vector_file(export_result, path, driver="GeoJSON", engine="pyogrio")

        result = geopandas.read_file(path, engine="pyogrio")
        assert_geodataframe_equal(
            result,
            expected,
            check_like=True,
            check_dtype=False,
        )

    def test_sjoin_nearest_native_result_defers_frame_materialization(self, monkeypatch):
        left = GeoDataFrame(
            {"a": [1, 2]},
            geometry=GeoSeries([Point(0, 0), Point(5, 0)]),
        )
        right = GeoDataFrame(
            {"b": [10, 20, 30]},
            geometry=GeoSeries([Point(1, 0), Point(6, 0), Point(20, 0)]),
        )

        real_materialize = RelationJoinResult.materialize
        materialize_calls = 0

        def _counting_materialize(self, *args, **kwargs):
            nonlocal materialize_calls
            materialize_calls += 1
            return real_materialize(self, *args, **kwargs)

        monkeypatch.setattr(
            RelationJoinResult,
            "materialize",
            _counting_materialize,
        )

        native_result, _selected = _sjoin_nearest_relation_result(
            left,
            right,
            max_distance=None,
            how="inner",
            return_distance=True,
            exclusive=False,
        )

        assert isinstance(native_result, RelationJoinResult)
        assert materialize_calls == 0

        def _fail(*_args, **_kwargs):
            raise AssertionError(
                "native nearest GeoDataFrame export should not require relation materialization"
            )

        monkeypatch.setattr(RelationJoinResult, "materialize", _fail)
        monkeypatch.setattr(
            native_results_module,
            "_materialize_relation_join_parts",
            _fail,
        )

        materialized = native_result.to_geodataframe(
            left,
            right,
            how="inner",
            lsuffix="left",
            rsuffix="right",
            distance_col="dist",
        )
        wrapped = geopandas.sjoin_nearest(left, right, how="inner", distance_col="dist")

        assert_geodataframe_equal(
            materialized.drop(columns=["dist"]),
            wrapped.drop(columns=["dist"]),
        )
        np.testing.assert_allclose(materialized["dist"], wrapped["dist"])

    def test_sjoin_nearest_export_result_defers_frame_materialization(self, monkeypatch):
        left = GeoDataFrame(
            {"a": [1, 2]},
            geometry=GeoSeries([Point(0, 0), Point(5, 0)]),
        )
        right = GeoDataFrame(
            {"b": [10, 20, 30]},
            geometry=GeoSeries([Point(1, 0), Point(6, 0), Point(20, 0)]),
        )

        real_materialize = RelationJoinResult.materialize
        materialize_calls = 0

        def _counting_materialize(self, *args, **kwargs):
            nonlocal materialize_calls
            materialize_calls += 1
            return real_materialize(self, *args, **kwargs)

        monkeypatch.setattr(
            RelationJoinResult,
            "materialize",
            _counting_materialize,
        )

        export_result, _selected = _sjoin_nearest_export_result(
            left,
            right,
            "inner",
            None,
            "left",
            "right",
            "dist",
            False,
        )

        assert isinstance(export_result, RelationJoinExportResult)
        assert materialize_calls == 0

        def _fail(*_args, **_kwargs):
            raise AssertionError(
                "native nearest GeoDataFrame export should not require relation materialization"
            )

        monkeypatch.setattr(RelationJoinResult, "materialize", _fail)
        monkeypatch.setattr(
            native_results_module,
            "_materialize_relation_join_parts",
            _fail,
        )

        materialized = export_result.to_geodataframe()
        wrapped = geopandas.sjoin_nearest(left, right, how="inner", distance_col="dist")

        assert_geodataframe_equal(
            materialized.drop(columns=["dist"]),
            wrapped.drop(columns=["dist"]),
        )
        np.testing.assert_allclose(materialized["dist"], wrapped["dist"])

    def test_relation_join_pandas_export_uses_fast_inner_path(self, monkeypatch):
        left = GeoDataFrame(
            {"a": [1, 2]},
            geometry=GeoSeries([Point(0, 0), Point(5, 0)]),
        )
        right = GeoDataFrame(
            {"b": [10, 20]},
            geometry=GeoSeries([Point(0, 0), Point(10, 0)]),
        )
        export_result, _impl, _execution = _sjoin_export_result(
            left,
            right,
            "inner",
            "intersects",
            None,
            "left",
            "right",
        )
        expected = export_result.materialize()[0]

        def _fail(*_args, **_kwargs):
            raise AssertionError(
                "pandas GeoDataFrame export should use the inner-join fast path"
            )

        monkeypatch.setattr(
            native_results_module,
            "_relation_join_output_layout",
            _fail,
        )
        monkeypatch.setattr(
            native_results_module,
            "_projected_frames_to_native_tabular_parts",
            _fail,
        )

        result = export_result.to_geodataframe()
        assert_geodataframe_equal(result, expected)

    def test_sjoin_nearest_export_result_native_tabular_skips_join_frame_materializer(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        left = GeoDataFrame(
            {"a": [1, 2]},
            geometry=GeoSeries([Point(0, 0), Point(5, 0)]),
        )
        right = GeoDataFrame(
            {"b": [10, 20, 30]},
            geometry=GeoSeries([Point(1, 0), Point(6, 0), Point(20, 0)]),
        )

        export_result, _selected = _sjoin_nearest_export_result(
            left,
            right,
            "inner",
            None,
            "left",
            "right",
            "dist",
            False,
        )
        expected = geopandas.sjoin_nearest(left, right, how="inner", distance_col="dist")

        real_native_parts = native_results_module._native_relation_join_parts
        native_calls = 0

        def _counting_native_parts(*args, **kwargs):
            nonlocal native_calls
            native_calls += 1
            return real_native_parts(*args, **kwargs)

        def _fail(*_args, **_kwargs):
            raise AssertionError(
                "native nearest tabular export should not require joined-frame materialization"
            )

        monkeypatch.setattr(
            native_results_module,
            "_native_relation_join_parts",
            _counting_native_parts,
        )
        monkeypatch.setattr(
            native_results_module,
            "_materialize_relation_join_parts",
            _fail,
        )

        native_result = to_native_tabular_result(export_result)

        assert native_calls == 1
        assert isinstance(native_result, NativeTabularResult)
        assert isinstance(native_result.attributes, NativeAttributeTable)
        assert native_result.attributes.arrow_table is not None
        result = native_result.to_geodataframe()
        assert_geodataframe_equal(
            result.drop(columns=["dist"]),
            expected.drop(columns=["dist"]),
        )
        np.testing.assert_allclose(result["dist"], expected["dist"])

    def test_sjoin_nearest_export_result_writes_without_relation_materialization(
        self,
        monkeypatch,
        tmp_path,
    ) -> None:
        left = GeoDataFrame(
            {"a": [1, 2]},
            geometry=GeoSeries([Point(0, 0), Point(5, 0)]),
        )
        right = GeoDataFrame(
            {"b": [10, 20, 30]},
            geometry=GeoSeries([Point(1, 0), Point(6, 0), Point(20, 0)]),
        )
        export_result, _selected = _sjoin_nearest_export_result(
            left,
            right,
            "inner",
            None,
            "left",
            "right",
            "dist",
            False,
        )
        expected = geopandas.sjoin_nearest(left, right, how="inner", distance_col="dist")

        def _fail(*_args, **_kwargs):
            raise AssertionError(
                "native nearest GeoParquet write should not require relation materialization"
            )

        monkeypatch.setattr(RelationJoinResult, "materialize", _fail)
        monkeypatch.setattr(NativeTabularResult, "to_geodataframe", _fail)

        path = tmp_path / "sjoin-nearest-native-no-materialize.parquet"
        write_geoparquet(export_result, path, geometry_encoding="geoarrow")
        monkeypatch.undo()

        result = geopandas.read_parquet(path)
        assert_geodataframe_equal(
            result.drop(columns=["dist"]),
            expected.drop(columns=["dist"]),
        )
        np.testing.assert_allclose(result["dist"], expected["dist"])

    @pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime required for exact nearest coverage regression")
    def test_sjoin_nearest_right_unbounded_preserves_all_right_rows(self) -> None:
        left = GeoDataFrame(
            {"geometry": GeoSeries([Point(0, 0), Point(1, 1)])},
        )
        right = GeoDataFrame(
            {"geometry": GeoSeries([Point(x, y) for x, y in zip(range(10), range(10))])},
        )

        joined = geopandas.sjoin_nearest(
            left,
            right,
            how="right",
            distance_col="dist",
        )

        assert len(joined) == len(right) == 10
        assert joined["index_left"].tolist() == [0, 1] + [1] * 8
        np.testing.assert_allclose(
            joined["dist"].to_numpy(),
            np.asarray([0.0, 0.0] + [np.sqrt(2.0 * i * i) for i in range(1, 9)], dtype=np.float64),
        )

    def test_dissolve_native_result_defers_frame_materialization(self, monkeypatch) -> None:
        frame = GeoDataFrame(
            {
                "group": ["a", "a", "b"],
                "value": [1, 2, 3],
            },
            geometry=GeoSeries(
                [
                    box(0, 0, 1, 1),
                    box(1, 0, 2, 1),
                    box(10, 10, 11, 11),
                ]
            ),
        )

        real_export = NativeTabularResult.to_geodataframe
        export_calls = 0

        def _counting_export(self, *args, **kwargs):
            nonlocal export_calls
            export_calls += 1
            return real_export(self, *args, **kwargs)

        monkeypatch.setattr(
            NativeTabularResult,
            "to_geodataframe",
            _counting_export,
        )

        native_result = evaluate_geopandas_dissolve_native(
            frame,
            by="group",
            aggfunc="first",
            as_index=True,
            level=None,
            sort=False,
            observed=False,
            dropna=True,
            method="coverage",
            grid_size=None,
            agg_kwargs={},
        )

        assert isinstance(native_result, NativeTabularResult)
        assert export_calls == 0

        materialized = native_result.to_geodataframe()
        wrapped = evaluate_geopandas_dissolve(
            frame,
            by="group",
            aggfunc="first",
            as_index=True,
            level=None,
            sort=False,
            observed=False,
            dropna=True,
            method="coverage",
            grid_size=None,
            agg_kwargs={},
        )

        assert export_calls == 2
        assert_geodataframe_equal(materialized, wrapped)

    def test_dissolve_native_result_writes_without_frame_materialization(
        self,
        monkeypatch,
        tmp_path,
    ) -> None:
        frame = GeoDataFrame(
            {
                "group": ["a", "a", "b"],
                "value": [1, 2, 3],
            },
            geometry=GeoSeries(
                [
                    box(0, 0, 1, 1),
                    box(1, 0, 2, 1),
                    box(10, 10, 11, 11),
                ]
            ),
        )

        native_result = evaluate_geopandas_dissolve_native(
            frame,
            by="group",
            aggfunc="first",
            as_index=True,
            level=None,
            sort=False,
            observed=False,
            dropna=True,
            method="coverage",
            grid_size=None,
            agg_kwargs={},
        )
        expected = evaluate_geopandas_dissolve(
            frame,
            by="group",
            aggfunc="first",
            as_index=True,
            level=None,
            sort=False,
            observed=False,
            dropna=True,
            method="coverage",
            grid_size=None,
            agg_kwargs={},
        )

        def _fail(*_args, **_kwargs):
            raise AssertionError(
                "native grouped GeoParquet write should not require GeoDataFrame export"
            )

        real_export = NativeTabularResult.to_geodataframe
        monkeypatch.setattr(
            NativeTabularResult,
            "to_geodataframe",
            _fail,
        )

        path = tmp_path / "dissolve-native.parquet"
        write_geoparquet(native_result, path, geometry_encoding="geoarrow")
        monkeypatch.setattr(
            NativeTabularResult,
            "to_geodataframe",
            real_export,
        )

        result = geopandas.read_parquet(path)
        assert_geodataframe_equal(result, expected)

    def test_clip_native_result_defers_frame_materialization(self, monkeypatch) -> None:
        frame = GeoDataFrame(
            {
                "value": [1, 2, 3],
                "geometry": GeoSeries(
                    [
                        box(0, 0, 2, 2),
                        box(1, 1, 3, 3),
                        box(10, 10, 12, 12),
                    ]
                ),
            }
        )
        mask = box(0.5, 0.5, 2.5, 2.5)

        real_export = clip_module._clip_native_tabular_to_spatial
        export_calls = 0

        def _counting_export(*args, **kwargs):
            nonlocal export_calls
            export_calls += 1
            return real_export(*args, **kwargs)

        monkeypatch.setattr(
            clip_module,
            "_clip_native_tabular_to_spatial",
            _counting_export,
        )

        native_result = evaluate_geopandas_clip_native(
            frame,
            mask,
            keep_geom_type=False,
            sort=False,
        )

        assert isinstance(native_result, NativeTabularResult)
        assert export_calls == 0

        materialized = clip_module._clip_native_tabular_to_spatial(
            native_result,
            source=frame,
        )
        wrapped = geopandas.clip(frame, mask)

        assert export_calls == 2
        assert_geodataframe_equal(materialized, wrapped)

    def test_clip_native_result_writes_without_frame_materialization(
        self,
        monkeypatch,
        tmp_path,
    ) -> None:
        frame = GeoDataFrame(
            {
                "value": [1, 2, 3],
                "geometry": GeoSeries(
                    [
                        box(0, 0, 2, 2),
                        box(1, 1, 3, 3),
                        box(10, 10, 12, 12),
                    ]
                ),
            }
        )
        mask = box(0.5, 0.5, 2.5, 2.5)
        native_result = evaluate_geopandas_clip_native(
            frame,
            mask,
            keep_geom_type=False,
            sort=False,
        )
        expected = geopandas.clip(frame, mask)

        def _fail(*_args, **_kwargs):
            raise AssertionError(
                "native clip GeoParquet write should not require GeoDataFrame export"
            )

        monkeypatch.setattr(
            clip_module,
            "_clip_native_tabular_to_spatial",
            _fail,
        )

        path = tmp_path / "clip-native.parquet"
        monkeypatch.setattr(NativeTabularResult, "to_geodataframe", _fail)
        write_geoparquet(native_result, path, geometry_encoding="geoarrow")
        monkeypatch.undo()

        result = geopandas.read_parquet(path)
        assert_geodataframe_equal(result, expected)

    def test_clip_native_result_builds_native_tabular_result(self) -> None:
        frame = GeoDataFrame(
            {
                "value": [1, 2, 3],
                "geometry": GeoSeries(
                    [
                        box(0, 0, 2, 2),
                        box(1, 1, 3, 3),
                        box(10, 10, 12, 12),
                    ]
                ),
            }
        )
        mask = box(0.5, 0.5, 2.5, 2.5)

        native_result = evaluate_geopandas_clip_native(
            frame,
            mask,
            keep_geom_type=False,
            sort=False,
        )

        tabular = to_native_tabular_result(native_result)

        assert isinstance(tabular, NativeTabularResult)
        assert_geodataframe_equal(
            tabular.to_geodataframe(),
            geopandas.clip(frame, mask),
        )

    def test_clip_native_tabular_records_fallback_before_shapely_cleanup(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from vibespatial.geometry.owned import OwnedGeometryArray, from_shapely_geometries
        from vibespatial.runtime.residency import Residency

        source = GeoDataFrame(
            {
                "value": [1],
                "geometry": GeoSeries([box(0, 0, 2, 2)]),
            },
            crs="EPSG:4326",
        )
        owned = from_shapely_geometries([box(0.5, 0.5, 1.5, 1.5)], residency=Residency.HOST)
        native_result = clip_module.ClipNativeResult(
            source=source,
            parts=(
                clip_module._clip_native_part(
                    source,
                    np.asarray([0], dtype=np.intp),
                    clip_module.GeometryArray.from_owned(owned, crs=source.crs),
                ),
            ),
            ordered_index=source.index,
            ordered_row_positions=np.asarray([0], dtype=np.intp),
            clipping_by_rectangle=False,
            has_non_point_candidates=True,
            keep_geom_type=False,
        )

        monkeypatch.setattr(
            native_results_module,
            "_clip_owned_geometry_native_result",
            lambda *args, **kwargs: (None, None),
        )

        real_to_shapely = OwnedGeometryArray.to_shapely

        def _guard_to_shapely(self, *args, **kwargs):
            raise AssertionError(
                "fallback should be recorded before clip tabular host cleanup"
            )

        monkeypatch.setattr(OwnedGeometryArray, "to_shapely", _guard_to_shapely)

        with pytest.raises(StrictNativeFallbackError):
            with strict_native_environment():
                to_native_tabular_result(native_result)

        monkeypatch.setattr(OwnedGeometryArray, "to_shapely", real_to_shapely)

    def test_point_only_clip_native_tabular_skips_spatial_materialization(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        frame = GeoDataFrame(
            {
                "value": [1, 2, 3],
                "geometry": GeoSeries([Point(0, 0), Point(2, 2), Point(10, 10)]),
            }
        )
        mask = box(-1, -1, 3, 3)
        native_result = evaluate_geopandas_clip_native(
            frame,
            mask,
            keep_geom_type=False,
            sort=False,
        )
        expected = geopandas.clip(frame, mask)

        def _fail(*_args, **_kwargs):
            raise AssertionError(
                "point-only native clip tabular export should not require clip source-type export"
            )

        monkeypatch.setattr(clip_module, "_clip_native_tabular_to_spatial", _fail)

        tabular = to_native_tabular_result(native_result)

        assert isinstance(tabular, NativeTabularResult)
        assert isinstance(tabular.attributes, NativeAttributeTable)
        assert tabular.attributes.arrow_table is not None
        assert_geodataframe_equal(tabular.to_geodataframe(), expected)

    def test_polygon_clip_native_tabular_skips_spatial_materialization(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from vibespatial.geometry.owned import OwnedGeometryArray

        frame = GeoDataFrame(
            {
                "value": [1, 2, 3],
                "geometry": GeoSeries(
                    [
                        box(0, 0, 2, 2),
                        box(1, 1, 3, 3),
                        box(10, 10, 12, 12),
                    ]
                ),
            }
        )
        mask = box(0.5, 0.5, 2.5, 2.5)
        native_result = evaluate_geopandas_clip_native(
            frame,
            mask,
            keep_geom_type=False,
            sort=False,
        )
        expected = geopandas.clip(frame, mask)

        def _fail(*_args, **_kwargs):
            raise AssertionError(
                "non-point native clip tabular export should not require clip source-type export"
            )

        monkeypatch.setattr(clip_module, "_clip_native_tabular_to_spatial", _fail)
        to_shapely_calls = 0

        real_to_shapely = OwnedGeometryArray.to_shapely

        def _counting_to_shapely(self, *args, **kwargs):
            nonlocal to_shapely_calls
            to_shapely_calls += 1
            return real_to_shapely(self, *args, **kwargs)

        monkeypatch.setattr(OwnedGeometryArray, "to_shapely", _counting_to_shapely)

        tabular = to_native_tabular_result(native_result)

        assert isinstance(tabular, NativeTabularResult)
        assert isinstance(tabular.attributes, NativeAttributeTable)
        assert tabular.attributes.arrow_table is not None
        assert to_shapely_calls == 0
        monkeypatch.undo()
        assert_geodataframe_equal(tabular.to_geodataframe(), expected)


# ---------------------------------------------------------------------------
# Phase 2: sjoin preserves attributes across all join types
# ---------------------------------------------------------------------------

class TestSjoinPreservesAttributes:
    @pytest.mark.parametrize("how", ["inner", "left", "right"])
    def test_sjoin_preserves_attributes_all_join_types(self, left_gdf, right_gdf, how):
        result = geopandas.sjoin(left_gdf, right_gdf, how=how)
        # All attribute columns from the appropriate side(s) must be present.
        if how in ("inner", "left"):
            for col in ["name", "value_int", "value_float"]:
                assert col in result.columns, f"Missing left column {col} in {how} join"
            for col in ["label", "count", "score"]:
                assert f"index_{col}" in result.columns or col in result.columns
        elif how == "right":
            for col in ["label", "count", "score"]:
                assert col in result.columns, f"Missing right column {col} in {how} join"

    def test_sjoin_no_geometry_in_attribute_reindex(self, left_gdf, right_gdf):
        """The joined result should have exactly one active geometry-like column."""
        result = geopandas.sjoin(left_gdf, right_gdf, how="inner")
        geom_cols = [
            c
            for c in result.columns
            if result[c].dtype.name in {"geometry", "device_geometry"}
        ]
        assert len(geom_cols) == 1


# ---------------------------------------------------------------------------
# Phase 3: Overlay preserves attributes
# ---------------------------------------------------------------------------

class TestOverlayPreservesAttributes:
    def test_overlay_intersection_preserves_attributes(self, left_gdf, right_gdf):
        result = geopandas.overlay(left_gdf, right_gdf, how="intersection")
        if len(result) > 0:
            # Both sides' attributes should be present (possibly suffixed).
            left_attr_found = any(
                c.startswith("name") or c == "name_1" for c in result.columns
            )
            right_attr_found = any(
                c.startswith("label") or c == "label_2" for c in result.columns
            )
            assert left_attr_found, "Left attributes missing from intersection"
            assert right_attr_found, "Right attributes missing from intersection"

    def test_overlay_union_nan_fills_non_overlapping(self, left_gdf, right_gdf):
        result = geopandas.overlay(left_gdf, right_gdf, how="union")
        # Non-overlapping portions should have NaN in the other side's columns.
        # Find a column from right side.
        right_cols = [c for c in result.columns if "label" in c or "count" in c or "score" in c]
        assert len(right_cols) > 0, "No right-side attribute columns found"
        # At least some rows should have NaN (the non-overlapping portions).
        has_nan = result[right_cols[0]].isna().any()
        # Only assert if there are non-overlapping geometries.
        if len(result) > len(left_gdf):
            assert has_nan, "Expected NaN fills for non-overlapping portions"


# ---------------------------------------------------------------------------
# Phase 4: Dissolve separates attribute aggregation from geometry
# ---------------------------------------------------------------------------

class TestDissolveBoundary:
    def test_dissolve_separates_attribute_agg_from_geometry(self):
        gdf = GeoDataFrame(
            {
                "group": ["a", "a", "b"],
                "value": [10, 20, 30],
                "geometry": GeoSeries([box(0, 0, 1, 1), box(0.5, 0, 1.5, 1), box(5, 5, 6, 6)]),
            }
        )
        result = gdf.dissolve(by="group", aggfunc="sum")
        assert result.loc["a", "value"] == 30
        assert result.loc["b", "value"] == 30
        # Geometry should be a union.
        assert result.loc["a", "geometry"].is_valid
        assert result.loc["b", "geometry"].is_valid


# ---------------------------------------------------------------------------
# Phase 5: Clip preserves all attribute columns
# ---------------------------------------------------------------------------

class TestClipPreservesAttributes:
    def test_clip_preserves_all_attribute_columns(self, left_gdf):
        clip_box = box(0, 0, 5, 5)
        result = geopandas.clip(left_gdf, clip_box)
        for col in ["name", "value_int", "value_float"]:
            assert col in result.columns, f"Clip dropped attribute column {col}"
        # Values should be preserved for features within the clip region.
        if len(result) > 0:
            assert result["value_int"].notna().all()


# ---------------------------------------------------------------------------
# GPU-conditional tests
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestGPUBoundary:
    @pytest.mark.skipif(not has_gpu_runtime(), reason="No GPU runtime available")
    def test_gpu_query_returns_host_arrays(self):
        tree = np.asarray([box(0, 0, 2, 2), box(3, 3, 5, 5)], dtype=object)
        query = np.asarray([box(1, 1, 4, 4)], dtype=object)
        owned, flat = build_owned_spatial_index(tree)

        result = query_spatial_index(
            owned, flat, query, predicate="intersects", sort=True
        )
        assert isinstance(result, np.ndarray)
        assert np.issubdtype(result.dtype, np.integer)
        # Should be on host (numpy), not device.
        assert type(result).__module__ == "numpy"

    @pytest.mark.skipif(not has_gpu_runtime(), reason="No GPU runtime available")
    def test_gpu_overlay_preserves_attributes(self):
        left = GeoDataFrame(
            {"a": [1, 2], "geometry": GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)])}
        )
        right = GeoDataFrame(
            {"b": [10, 20], "geometry": GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)])}
        )
        result = geopandas.overlay(left, right, how="intersection")
        if len(result) > 0:
            a_cols = [c for c in result.columns if c.startswith("a")]
            b_cols = [c for c in result.columns if c.startswith("b")]
            assert len(a_cols) > 0
            assert len(b_cols) > 0

    @pytest.mark.skipif(not has_gpu_runtime(), reason="No GPU runtime available")
    def test_gpu_overlay_preserves_device_geometry_backing(self):
        from vibespatial.geometry.device_array import DeviceGeometryArray
        from vibespatial.geometry.owned import from_shapely_geometries
        from vibespatial.runtime.residency import Residency

        left = GeoDataFrame(
            {"a": [1, 2]},
            geometry=GeoSeries(
                DeviceGeometryArray._from_owned(
                    from_shapely_geometries(
                        [box(0, 0, 2, 2), box(3, 3, 5, 5)],
                        residency=Residency.DEVICE,
                    ),
                    crs="EPSG:3857",
                ),
                crs="EPSG:3857",
            ),
            crs="EPSG:3857",
        )
        right = GeoDataFrame(
            {"b": [10, 20]},
            geometry=GeoSeries(
                DeviceGeometryArray._from_owned(
                    from_shapely_geometries(
                        [box(1, 1, 4, 4), box(6, 6, 8, 8)],
                        residency=Residency.DEVICE,
                    ),
                    crs="EPSG:3857",
                ),
                crs="EPSG:3857",
            ),
            crs="EPSG:3857",
        )

        result = geopandas.overlay(left, right, how="intersection")

        assert isinstance(result.geometry.values, DeviceGeometryArray)
        assert getattr(result.geometry.values, "_owned", None) is not None
