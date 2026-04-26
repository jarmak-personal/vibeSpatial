from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import GeometryCollection, LineString, Point, box

import vibespatial.api as geopandas
from vibespatial import (
    DissolveUnionMethod,
    benchmark_dissolve_pipeline,
    fusion_plan_for_dissolve,
    has_gpu_runtime,
    plan_dissolve_pipeline,
)
from vibespatial.api._native_grouped import NativeGrouped, NativeGroupedAttributeReduction
from vibespatial.api._native_results import GeometryNativeResult, NativeTabularResult
from vibespatial.api.geometry_array import GeometryArray
from vibespatial.api.testing import assert_geodataframe_equal
from vibespatial.bench.pipeline import _dissolve_join_heavy_groups, _regular_polygons_frame
from vibespatial.geometry.device_array import DeviceGeometryArray
from vibespatial.geometry.owned import from_shapely_geometries, seed_all_validity_cache
from vibespatial.kernels.constructive.segmented_union import segmented_union_all
from vibespatial.overlay.dissolve import (
    evaluate_geopandas_dissolve,
    evaluate_geopandas_dissolve_native,
    execute_grouped_union,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.fallbacks import StrictNativeFallbackError
from vibespatial.runtime.fusion import IntermediateDisposition
from vibespatial.runtime.provenance import clear_rewrite_events, get_rewrite_events
from vibespatial.runtime.residency import Residency
from vibespatial.testing import strict_native_environment

dissolve_module = importlib.import_module("vibespatial.overlay.dissolve")


def test_dissolve_pipeline_plan_uses_group_encoding_and_grouped_union() -> None:
    plan = plan_dissolve_pipeline("unary")

    assert [stage.name for stage in plan.stages] == [
        "encode_groups",
        "stable_sort_rows",
        "segment_groups",
        "aggregate_attributes",
        "union_group_geometries",
        "assemble_result_frame",
    ]
    assert plan.stages[-1].disposition is IntermediateDisposition.PERSIST
    assert plan.stages[-1].geometry_producing is True


def test_dissolve_fusion_plan_persists_final_frame_only() -> None:
    fusion = fusion_plan_for_dissolve(DissolveUnionMethod.COVERAGE)

    assert len(fusion.stages) >= 2
    assert fusion.stages[-1].disposition is IntermediateDisposition.PERSIST
    assert fusion.stages[-1].steps[-1].output_name == "dissolved_frame"


def test_execute_grouped_union_emits_empty_geometry_for_unobserved_group() -> None:
    geometries = [Point(0, 0), Point(1, 1)]
    grouped = execute_grouped_union(geometries, [pd.Index([0]).to_numpy(), pd.Index([], dtype="int64").to_numpy()])

    assert grouped.group_count == 2
    assert grouped.empty_groups == 1
    assert grouped.geometries[1].geom_type == "GeometryCollection"
    assert grouped.geometries[1].is_empty


@pytest.mark.gpu
def test_execute_grouped_union_owned_coverage_avoids_geometry_materialization() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    class ExplodingGeometries:
        def __array__(self, dtype=None):
            raise AssertionError("owned coverage dissolve materialized geometry objects")

    owned = from_shapely_geometries([box(0, 0, 1, 1), box(2, 0, 3, 1)])
    grouped = execute_grouped_union(
        ExplodingGeometries(),
        [np.asarray([0], dtype=np.int32), np.asarray([1], dtype=np.int32)],
        method=DissolveUnionMethod.COVERAGE,
        owned=owned,
    )

    assert grouped.geometries is None
    assert grouped.owned is not None
    assert grouped.owned.row_count == 2


@pytest.mark.gpu
def test_execute_grouped_union_codes_owned_coverage_avoids_geometry_materialization(
    monkeypatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    class ExplodingGeometries:
        def __array__(self, dtype=None):
            raise AssertionError("owned coverage dissolve codes path materialized geometry objects")

    owned = from_shapely_geometries(
        [box(0, 0, 1, 1), box(2, 0, 3, 1)],
        residency=Residency.DEVICE,
    )
    owned._validity = None
    owned._tags = None
    owned._family_row_offsets = None

    def _fail_host_metadata(*_args, **_kwargs):
        raise AssertionError("device-owned coverage path should not materialize host metadata")

    monkeypatch.setattr(type(owned), "_ensure_host_metadata", _fail_host_metadata)

    grouped = dissolve_module.execute_grouped_union_codes(
        ExplodingGeometries(),
        np.asarray([0, 1], dtype=np.int32),
        group_count=2,
        method=DissolveUnionMethod.COVERAGE,
        owned=owned,
    )

    assert grouped is not None
    assert grouped.geometries is None
    assert grouped.owned is not None
    assert grouped.owned.row_count == 2


@pytest.mark.gpu
def test_execute_grouped_union_codes_device_codes_reports_host_fallback_materialization() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    owned = from_shapely_geometries(
        [
            LineString([(0.0, 0.0), (1.0, 0.0)]),
            LineString([(1.0, 0.0), (2.0, 0.0)]),
        ],
        residency=Residency.DEVICE,
    )
    row_group_codes = cp.asarray([0, 0], dtype=cp.int32)
    clear_materialization_events()

    grouped = dissolve_module.execute_grouped_union_codes(
        (),
        row_group_codes,
        group_count=1,
        method=DissolveUnionMethod.COVERAGE,
        owned=owned,
    )

    assert grouped is None
    events = get_materialization_events(clear=True)
    assert len(events) == 1
    assert events[0].operation == "device_group_codes_to_host"
    assert events[0].d2h_transfer is True
    assert events[0].detail == "rows=2, bytes=8"


@pytest.mark.gpu
def test_execute_grouped_union_codes_device_codes_bulk_disjoint_coverage_stays_native() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    geoms = [
        box(0.0, 0.0, 1.0, 1.0),
        box(10.0, 0.0, 11.0, 1.0),
        box(0.0, 10.0, 1.0, 11.0),
        box(10.0, 10.0, 11.0, 11.0),
    ]
    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    seed_all_validity_cache(owned)
    row_group_codes = cp.asarray([0, 1, 0, 1], dtype=cp.int32)
    clear_materialization_events()
    reset_d2h_transfer_count()

    with assert_zero_d2h_transfers():
        grouped = dissolve_module.execute_grouped_union_codes(
            (),
            row_group_codes,
            group_count=2,
            method=DissolveUnionMethod.COVERAGE,
            owned=owned,
        )

    assert grouped is not None
    assert grouped.geometries is None
    assert grouped.owned is not None
    assert grouped.owned.row_count == 2
    assert get_materialization_events(clear=True) == []
    actual = grouped.owned.to_shapely()
    expected = [
        shapely.coverage_union_all(np.asarray([geoms[0], geoms[2]], dtype=object)),
        shapely.coverage_union_all(np.asarray([geoms[1], geoms[3]], dtype=object)),
    ]
    assert shapely.equals(actual[0], expected[0])
    assert shapely.equals(actual[1], expected[1])


@pytest.mark.gpu
@pytest.mark.parametrize(
    "geoms",
    [
        [box(0.0, 0.0, 1.0, 1.0), box(1.0, 0.0, 2.0, 1.0)],
        [box(0.0, 0.0, 2.0, 2.0), box(1.0, 1.0, 3.0, 3.0)],
    ],
)
def test_grouped_disjoint_subset_union_declines_touching_or_overlapping_groups(
    geoms,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")

    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    seed_all_validity_cache(owned)
    grouped = dissolve_module.execute_grouped_disjoint_subset_union_gpu_owned_codes(
        cp.asarray([0, 0], dtype=cp.int32),
        group_count=1,
        method=DissolveUnionMethod.COVERAGE,
        owned=owned,
    )

    assert grouped is None


def test_evaluate_geopandas_dissolve_matches_current_categorical_semantics() -> None:
    frame = geopandas.GeoDataFrame(
        {
            "cat": pd.Categorical(["a", "a", "b", "b"]),
            "noncat": [1, 1, 1, 2],
            "to_agg": [1, 2, 3, 4],
            "geometry": geopandas.array.from_wkt(
                ["POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)"]
            ),
        }
    )

    result = evaluate_geopandas_dissolve(
        frame,
        by=["cat", "noncat"],
        aggfunc="first",
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )
    expected = frame.copy().dissolve(["cat", "noncat"])

    assert_geodataframe_equal(result, expected)


def test_evaluate_geopandas_dissolve_can_use_dense_group_codes_without_group_positions(
    monkeypatch,
) -> None:
    frame = geopandas.GeoDataFrame(
        {
            "group": pd.Categorical(["b", "a", "b", "a"], categories=["a", "b", "c"]),
            "value": [1, 2, 3, 4],
            "geometry": geopandas.array.from_wkt(
                [
                    "POLYGON ((0 0, 1 0, 0 1, 0 0))",
                    "POLYGON ((1 0, 1 1, 0 1, 1 0))",
                    "POLYGON ((10 10, 11 10, 10 11, 10 10))",
                    "POLYGON ((11 10, 11 11, 10 11, 11 10))",
                ]
            ),
        }
    )

    expected = frame.dissolve(by="group", aggfunc="first", sort=False, method="coverage")

    calls = 0
    real_codes = dissolve_module.execute_grouped_union_codes

    def _count_codes(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_codes(*args, **kwargs)

    def _fail_positions(*args, **kwargs):
        raise AssertionError("dense group code path should avoid _normalize_group_positions")

    monkeypatch.setattr(dissolve_module, "execute_grouped_union_codes", _count_codes)
    monkeypatch.setattr(dissolve_module, "_normalize_group_positions", _fail_positions)

    result = evaluate_geopandas_dissolve(
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

    assert calls == 1
    assert_geodataframe_equal(result, expected)


def test_evaluate_geopandas_dissolve_uses_native_grouped_numeric_reducers(
    monkeypatch,
) -> None:
    frame = geopandas.GeoDataFrame(
        {
            "group": [2, 1, 2, 1],
            "value": [1, 2, 3, 4],
            "flag": [True, False, True, True],
            "geometry": geopandas.array.from_wkt(
                [
                    "POINT (0 0)",
                    "POINT (1 1)",
                    "POINT (2 2)",
                    "POINT (3 3)",
                ]
            ),
        }
    )
    expected = frame.dissolve(
        by="group",
        aggfunc={"value": "sum", "flag": "sum"},
        sort=True,
        method="unary",
    )

    real_reduce = NativeGrouped.reduce_numeric_columns
    reducer_calls = []

    def _record_reduce(self, columns, reducers):
        reducer_calls.append((tuple(columns), dict(reducers)))
        return real_reduce(self, columns, reducers)

    def _fail_groupby(*_args, **_kwargs):
        raise AssertionError("admitted native numeric dissolve should not call pandas groupby")

    monkeypatch.setattr(NativeGrouped, "reduce_numeric_columns", _record_reduce)
    monkeypatch.setattr(pd.DataFrame, "groupby", _fail_groupby)

    geopandas.clear_dispatch_events()
    result = evaluate_geopandas_dissolve(
        frame,
        by="group",
        aggfunc={"value": "sum", "flag": "sum"},
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )
    events = geopandas.get_dispatch_events(clear=True)

    assert reducer_calls == [
        (("value", "flag"), {"value": "sum", "flag": "sum"})
    ]
    assert any(
        event.implementation == "native_grouped_numeric_reducers"
        for event in events
    )
    assert_geodataframe_equal(result, expected)


def test_evaluate_geopandas_dissolve_uses_native_categorical_numeric_reducers(
    monkeypatch,
) -> None:
    frame = geopandas.GeoDataFrame(
        {
            "group": pd.Categorical(["b", "a", "b", "a"], categories=["a", "b", "c"]),
            "value": [1, 2, 3, 4],
            "weight": [1.5, 2.0, 2.5, 4.0],
            "geometry": geopandas.array.from_wkt(
                [
                    "POINT (0 0)",
                    "POINT (1 1)",
                    "POINT (2 2)",
                    "POINT (3 3)",
                ]
            ),
        }
    )
    expected = frame.dissolve(
        by="group",
        aggfunc={"value": "sum", "weight": "mean"},
        sort=False,
        observed=False,
        dropna=True,
        method="unary",
    )

    real_reduce = NativeGrouped.reduce_numeric_columns
    reducer_calls = []

    def _record_reduce(self, columns, reducers):
        reducer_calls.append(
            (
                tuple(columns),
                dict(reducers),
                self.output_index_plan.index.copy(),
                self.group_codes.copy(),
            )
        )
        return real_reduce(self, columns, reducers)

    def _fail_groupby(*_args, **_kwargs):
        raise AssertionError("admitted categorical dissolve should not call pandas groupby")

    monkeypatch.setattr(NativeGrouped, "reduce_numeric_columns", _record_reduce)
    monkeypatch.setattr(pd.DataFrame, "groupby", _fail_groupby)

    result = evaluate_geopandas_dissolve(
        frame,
        by="group",
        aggfunc={"value": "sum", "weight": "mean"},
        as_index=True,
        level=None,
        sort=False,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )

    assert len(reducer_calls) == 1
    reduced_columns, reducers, output_index, group_codes = reducer_calls[0]
    assert reduced_columns == ("value", "weight")
    assert reducers == {"value": "sum", "weight": "mean"}
    assert output_index.tolist() == ["b", "a", "c"]
    assert group_codes.tolist() == [0, 1, 0, 1]
    assert_geodataframe_equal(result, expected)


def test_evaluate_geopandas_dissolve_uses_native_grouped_min_max_reducers(
    monkeypatch,
) -> None:
    frame = geopandas.GeoDataFrame(
        {
            "group": pd.Categorical(["b", "a", "b", "a"], categories=["a", "b", "c"]),
            "value": [1, 2, 3, 4],
            "flag": [True, False, True, True],
            "geometry": geopandas.array.from_wkt(
                [
                    "POINT (0 0)",
                    "POINT (1 1)",
                    "POINT (2 2)",
                    "POINT (3 3)",
                ]
            ),
        }
    )
    expected = frame.dissolve(
        by="group",
        aggfunc={"value": "min", "flag": "max"},
        sort=False,
        observed=False,
        dropna=True,
        method="unary",
    )

    real_reduce = NativeGrouped.reduce_numeric_columns
    reducer_calls = []

    def _record_reduce(self, columns, reducers):
        reducer_calls.append((tuple(columns), dict(reducers), self.output_index_plan.index.copy()))
        return real_reduce(self, columns, reducers)

    def _fail_groupby(*_args, **_kwargs):
        raise AssertionError("admitted categorical min/max dissolve should not call pandas groupby")

    monkeypatch.setattr(NativeGrouped, "reduce_numeric_columns", _record_reduce)
    monkeypatch.setattr(pd.DataFrame, "groupby", _fail_groupby)

    result = evaluate_geopandas_dissolve(
        frame,
        by="group",
        aggfunc={"value": "min", "flag": "max"},
        as_index=True,
        level=None,
        sort=False,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )

    assert len(reducer_calls) == 1
    reduced_columns, reducers, output_index = reducer_calls[0]
    assert reduced_columns == ("value", "flag")
    assert reducers == {"value": "min", "flag": "max"}
    assert output_index.tolist() == ["b", "a", "c"]
    assert result.index.tolist() == ["b", "a", "c"]
    assert result["value"].tolist()[:2] == [1.0, 2.0]
    assert result["flag"].tolist()[:2] == [1.0, 1.0]
    assert np.isnan(result["value"].iloc[2])
    assert np.isnan(result["flag"].iloc[2])
    assert_geodataframe_equal(result, expected)


def test_evaluate_geopandas_dissolve_default_first_uses_native_grouped_reducer(
    monkeypatch,
) -> None:
    frame = geopandas.GeoDataFrame(
        {
            "group": pd.Categorical(["b", "a", "b", "a"], categories=["a", "b", "c"]),
            "value": [1, 2, 3, 4],
            "flag": [True, False, True, True],
            "geometry": geopandas.array.from_wkt(
                [
                    "POINT (0 0)",
                    "POINT (1 1)",
                    "POINT (2 2)",
                    "POINT (3 3)",
                ]
            ),
        }
    )
    expected = frame.dissolve(
        by="group",
        aggfunc="first",
        sort=False,
        observed=False,
        dropna=True,
        method="unary",
    )

    real_reduce = NativeGrouped.reduce_numeric_columns
    reducer_calls = []

    def _record_reduce(self, columns, reducers):
        reducer_calls.append((tuple(columns), dict(reducers), self.output_index_plan.index.copy()))
        return real_reduce(self, columns, reducers)

    def _fail_groupby(*_args, **_kwargs):
        raise AssertionError("admitted default-first dissolve should not call pandas groupby")

    monkeypatch.setattr(NativeGrouped, "reduce_numeric_columns", _record_reduce)
    monkeypatch.setattr(pd.DataFrame, "groupby", _fail_groupby)

    result = evaluate_geopandas_dissolve(
        frame,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=False,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )

    assert len(reducer_calls) == 1
    reduced_columns, reducers, output_index = reducer_calls[0]
    assert reduced_columns == ("value", "flag")
    assert reducers == {"value": "first", "flag": "first"}
    assert output_index.tolist() == ["b", "a", "c"]
    assert result["value"].tolist()[:2] == [1.0, 2.0]
    assert result["flag"].tolist()[:2] == [1.0, 0.0]
    assert np.isnan(result["value"].iloc[2])
    assert np.isnan(result["flag"].iloc[2])
    assert_geodataframe_equal(result, expected)


def test_evaluate_geopandas_dissolve_default_first_uses_native_take_reducers(
    monkeypatch,
) -> None:
    frame = geopandas.GeoDataFrame(
        {
            "group": pd.Categorical(["b", "a", "b", "a"], categories=["a", "b", "c"]),
            "value": [1, 2, 3, 4],
            "label": pd.Series(
                [None, "right", "last-left", "last-right"],
                dtype=object,
            ),
            "category": pd.Categorical(
                [None, "y", "z", "w"],
                categories=["w", "x", "y", "z"],
            ),
            "text": pd.array([pd.NA, "bb", "cc", "dd"], dtype="string"),
            "nullable": pd.array([pd.NA, 2, 3, pd.NA], dtype="Int64"),
            "geometry": geopandas.array.from_wkt(
                [
                    "POINT (0 0)",
                    "POINT (1 1)",
                    "POINT (2 2)",
                    "POINT (3 3)",
                ]
            ),
        }
    )
    expected = frame.dissolve(
        by="group",
        aggfunc="first",
        sort=False,
        observed=False,
        dropna=True,
        method="unary",
    )

    real_take = NativeGrouped.reduce_take
    take_calls = []

    def _record_take(self, values, reducer):
        take_calls.append((getattr(values, "name", None), reducer))
        return real_take(self, values, reducer)

    def _fail_groupby(*_args, **_kwargs):
        raise AssertionError("admitted label-column dissolve should not call pandas groupby")

    monkeypatch.setattr(NativeGrouped, "reduce_take", _record_take)
    monkeypatch.setattr(pd.DataFrame, "groupby", _fail_groupby)

    result = evaluate_geopandas_dissolve(
        frame,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=False,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )

    assert take_calls == [
        ("label", "first"),
        ("category", "first"),
        ("text", "first"),
        ("nullable", "first"),
    ]
    assert result["label"].tolist() == ["last-left", "right", None]
    assert result["category"].tolist() == ["z", "y", np.nan]
    assert isinstance(result["category"].dtype, pd.CategoricalDtype)
    assert result["text"].tolist()[:2] == ["cc", "bb"]
    assert pd.isna(result["text"].iloc[2])
    assert result["nullable"].tolist()[:2] == [3, 2]
    assert pd.isna(result["nullable"].iloc[2])
    assert_geodataframe_equal(result, expected)


def test_evaluate_geopandas_dissolve_categorical_observed_true_uses_observed_codes(
    monkeypatch,
) -> None:
    frame = geopandas.GeoDataFrame(
        {
            "group": pd.Categorical(["b", None, "a", "b"], categories=["a", "b", "c"]),
            "value": [1, 2, 3, 4],
            "geometry": geopandas.array.from_wkt(
                [
                    "POINT (0 0)",
                    "POINT (1 1)",
                    "POINT (2 2)",
                    "POINT (3 3)",
                ]
            ),
        }
    )
    expected = frame.dissolve(
        by="group",
        aggfunc={"value": "count"},
        sort=True,
        observed=True,
        dropna=True,
        method="unary",
    )

    real_reduce = NativeGrouped.reduce_numeric_columns
    reducer_calls = []

    def _record_reduce(self, columns, reducers):
        reducer_calls.append((self.output_index_plan.index.copy(), self.group_codes.copy()))
        return real_reduce(self, columns, reducers)

    def _fail_groupby(*_args, **_kwargs):
        raise AssertionError("observed categorical dissolve should not call pandas groupby")

    monkeypatch.setattr(NativeGrouped, "reduce_numeric_columns", _record_reduce)
    monkeypatch.setattr(pd.DataFrame, "groupby", _fail_groupby)

    result = evaluate_geopandas_dissolve(
        frame,
        by="group",
        aggfunc={"value": "count"},
        as_index=True,
        level=None,
        sort=True,
        observed=True,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )

    assert len(reducer_calls) == 1
    output_index, group_codes = reducer_calls[0]
    assert output_index.tolist() == ["a", "b"]
    assert group_codes.tolist() == [1, -1, 0, 1]
    assert_geodataframe_equal(result, expected)


@pytest.mark.parametrize(
    ("observed", "sort", "expected_index", "expected_values", "expected_codes"),
    [
        (False, False, ["b", np.nan, "a", "c"], [5, 2, 3, 0], [0, 1, 2, 0]),
        (False, True, ["a", "b", "c", np.nan], [3, 5, 0, 2], [1, 3, 0, 1]),
        (True, False, ["b", np.nan, "a"], [5, 2, 3], [0, 1, 2, 0]),
        (True, True, ["a", "b", np.nan], [3, 5, 2], [1, 2, 0, 1]),
    ],
)
def test_evaluate_geopandas_dissolve_categorical_dropna_false_uses_native_null_group(
    monkeypatch,
    observed: bool,
    sort: bool,
    expected_index: list[object],
    expected_values: list[int],
    expected_codes: list[int],
) -> None:
    frame = geopandas.GeoDataFrame(
        {
            "group": pd.Categorical(["b", None, "a", "b"], categories=["a", "b", "c"]),
            "value": [1, 2, 3, 4],
            "geometry": geopandas.array.from_wkt(
                [
                    "POINT (0 0)",
                    "POINT (1 1)",
                    "POINT (2 2)",
                    "POINT (3 3)",
                ]
            ),
        }
    )

    real_reduce = NativeGrouped.reduce_numeric_columns
    reducer_calls = []

    def _record_reduce(self, columns, reducers):
        reducer_calls.append((self.output_index_plan.index.copy(), self.group_codes.copy()))
        return real_reduce(self, columns, reducers)

    def _fail_groupby(*_args, **_kwargs):
        raise AssertionError("categorical null-group dissolve should not call pandas groupby")

    monkeypatch.setattr(NativeGrouped, "reduce_numeric_columns", _record_reduce)
    monkeypatch.setattr(pd.DataFrame, "groupby", _fail_groupby)

    result = evaluate_geopandas_dissolve(
        frame,
        by="group",
        aggfunc={"value": "sum"},
        as_index=True,
        level=None,
        sort=sort,
        observed=observed,
        dropna=False,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )

    assert len(reducer_calls) == 1
    output_index, group_codes = reducer_calls[0]
    assert output_index.tolist() == expected_index
    assert group_codes.tolist() == expected_codes
    assert result.index.tolist() == expected_index
    assert result["value"].tolist() == expected_values


def test_evaluate_geopandas_dissolve_native_as_index_false_defers_attribute_export(
    monkeypatch,
) -> None:
    frame = geopandas.GeoDataFrame(
        {
            "group": ["b", "a", "b", "a"],
            "value": [1, 2, 3, 4],
            "geometry": geopandas.array.from_wkt(
                [
                    "POINT (0 0)",
                    "POINT (1 1)",
                    "POINT (2 2)",
                    "POINT (3 3)",
                ]
            ),
        }
    )

    def _fail_groupby(*_args, **_kwargs):
        raise AssertionError("admitted as_index=False dissolve should not call pandas groupby")

    def _fail_to_pandas(*_args, **_kwargs):
        raise AssertionError("grouped reductions should not export before terminal materialization")

    monkeypatch.setattr(pd.DataFrame, "groupby", _fail_groupby)
    monkeypatch.setattr(NativeGroupedAttributeReduction, "to_pandas", _fail_to_pandas)

    native_result = evaluate_geopandas_dissolve_native(
        frame,
        by="group",
        aggfunc={"value": "sum"},
        as_index=False,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )

    assert native_result.attributes.loader is not None
    assert tuple(native_result.attributes.columns) == ("group", "value")
    assert native_result.column_order == ("group", "geometry", "value")


@pytest.mark.gpu
def test_unary_dissolve_keeps_small_device_coverage_on_exact_gpu_path() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    geoms = [
        box(float(i), float(group) * 10.0, float(i) + 1.0, float(group) * 10.0 + 1.0)
        for group in range(3)
        for i in range(20)
    ]
    groups = np.repeat(np.arange(3, dtype=np.int32), 20)
    frame = geopandas.GeoDataFrame(
        {"group": groups, "value": np.arange(len(geoms), dtype=np.int32)},
        geometry=geopandas.GeoSeries(
            DeviceGeometryArray._from_owned(
                from_shapely_geometries(geoms, residency=Residency.DEVICE),
                crs="EPSG:3857",
            ),
            crs="EPSG:3857",
        ),
        crs="EPSG:3857",
    )

    clear_rewrite_events()
    actual = evaluate_geopandas_dissolve(
        frame,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )
    rewrite_events = get_rewrite_events(clear=True)

    expected = evaluate_geopandas_dissolve(
        frame,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="coverage",
        grid_size=None,
        agg_kwargs={},
    )
    assert not any(
        event.rule_name == "R11_dissolve_unary_polygon_coverage_to_coverage"
        for event in rewrite_events
    )
    assert actual.geometry.dtype.name == "device_geometry"
    assert type(actual.geometry.values).__name__ == "DeviceGeometryArray"
    assert actual["value"].tolist() == expected["value"].tolist()
    actual_geoms = np.asarray(actual.geometry.array, dtype=object)
    expected_geoms = np.asarray(expected.geometry.array, dtype=object)
    assert all(
        bool(shapely.equals(actual_geom, expected_geom))
        for actual_geom, expected_geom in zip(actual_geoms, expected_geoms, strict=True)
    )


@pytest.mark.gpu
def test_unary_dissolve_does_not_rewrite_overlapping_polygon_groups() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    geoms = [
        box(float(i) * 0.5, 0.0, float(i) * 0.5 + 1.0, 1.0)
        for i in range(50)
    ]
    frame = geopandas.GeoDataFrame(
        {"group": np.zeros(len(geoms), dtype=np.int32), "value": np.arange(len(geoms), dtype=np.int32)},
        geometry=geopandas.GeoSeries(
            DeviceGeometryArray._from_owned(
                from_shapely_geometries(geoms, residency=Residency.DEVICE),
                crs="EPSG:3857",
            ),
            crs="EPSG:3857",
        ),
        crs="EPSG:3857",
    )

    clear_rewrite_events()
    result = evaluate_geopandas_dissolve(
        frame,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )
    rewrite_events = get_rewrite_events(clear=True)

    assert not any(
        event.rule_name == "R11_dissolve_unary_polygon_coverage_to_coverage"
        for event in rewrite_events
    )
    actual_geom = np.asarray(result.geometry.array, dtype=object)[0]
    expected_geom = shapely.union_all(np.asarray(frame.geometry.array, dtype=object))
    assert shapely.area(shapely.symmetric_difference(actual_geom, expected_geom)) == 0.0


def test_evaluate_geopandas_dissolve_preserves_owned_backing_through_reset_index() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    frame = geopandas.GeoDataFrame(
        {
            "group": np.zeros(16, dtype=np.int32),
            "value": np.arange(16, dtype=np.int32),
        },
        geometry=geopandas.GeoSeries(
            DeviceGeometryArray._from_owned(
                from_shapely_geometries(
                    [Point(float(i), 0.0) for i in range(16)],
                    residency=Residency.DEVICE,
                ),
                crs="EPSG:3857",
            ),
            crs="EPSG:3857",
        ),
        crs="EPSG:3857",
    )
    buffered = frame.copy()
    buffered["geometry"] = buffered.geometry.buffer(2.0)
    assert getattr(buffered.geometry.values, "_owned", None) is not None

    result = evaluate_geopandas_dissolve(
        buffered,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )

    owned = getattr(result.geometry.values, "_owned", None)
    assert owned is not None
    assert owned.residency is Residency.DEVICE

    reset = result.reset_index(drop=True)
    reset_owned = getattr(reset.geometry.values, "_owned", None)
    assert reset_owned is not None
    assert reset_owned.residency is Residency.DEVICE


def test_benchmark_dissolve_pipeline_reports_group_count() -> None:
    frame = geopandas.GeoDataFrame(
        {
            "group": [0, 0, 1, 1],
            "value": [1, 2, 3, 4],
            "geometry": [Point(0, 0), Point(1, 1), Point(10, 10), Point(11, 11)],
        }
    )

    benchmark = benchmark_dissolve_pipeline(
        frame, by="group", dataset="points", iterations=1, warmup=0,
    )

    assert benchmark.rows == 4
    assert benchmark.groups == 2
    assert benchmark.iterations == 1
    assert benchmark.pipeline_elapsed_seconds >= 0.0
    assert benchmark.baseline_elapsed_seconds >= 0.0


def test_execute_grouped_union_codes_avoids_geometry_array_materialization_for_owned_unary(
    monkeypatch,
) -> None:
    geometry_array = GeometryArray.from_owned(
        from_shapely_geometries(
            [
                box(0, 0, 1, 1),
                box(1, 1, 2, 2),
            ]
        )
    )
    owned = getattr(geometry_array, "_owned", None)
    assert owned is not None

    def _fail(*_args, **_kwargs):
        raise AssertionError("owned unary grouped union should not materialize the full GeometryArray")

    monkeypatch.setattr(GeometryArray, "__array__", _fail, raising=False)

    grouped = dissolve_module.execute_grouped_union_codes(
        geometry_array,
        pd.Index([0, 0], dtype="int32").to_numpy(),
        group_count=1,
        method="unary",
        owned=owned,
    )

    assert grouped is not None
    assert grouped.group_count == 1
    assert grouped.non_empty_groups == 1
    assert grouped.owned is not None
    assert grouped.geometries is None


def test_execute_native_grouped_union_consumes_native_grouped_offsets_for_owned_unary(
    monkeypatch,
) -> None:
    geometry_array = GeometryArray.from_owned(
        from_shapely_geometries(
            [
                box(0, 0, 1, 1),
                box(1, 1, 2, 2),
                box(5, 0, 6, 1),
            ]
        )
    )
    owned = getattr(geometry_array, "_owned", None)
    assert owned is not None

    def _fail(*_args, **_kwargs):
        raise AssertionError("native grouped union should not materialize geometries")

    monkeypatch.setattr(GeometryArray, "__array__", _fail, raising=False)
    grouped = NativeGrouped.from_dense_codes(
        pd.Index([0, 0, 1], dtype="int32").to_numpy(),
        group_count=2,
    )

    result = dissolve_module.execute_native_grouped_union(
        grouped,
        _geometries=geometry_array,
        method="unary",
        owned=owned,
    )

    assert result is not None
    assert result.group_count == 2
    assert result.non_empty_groups == 2
    assert result.empty_groups == 0
    assert result.owned is not None
    assert result.geometries is None


def test_execute_native_grouped_union_consumes_device_codes_for_owned_coverage(
    monkeypatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for NativeGrouped coverage box reducer")
    cp = pytest.importorskip("cupy")

    owned = from_shapely_geometries(
        [
            box(0, 0, 1, 1),
            box(1, 0, 2, 1),
            box(10, 10, 11, 11),
            box(11, 10, 12, 11),
        ],
        residency=Residency.DEVICE,
    )
    grouped = NativeGrouped.from_dense_codes(
        cp.asarray([0, 0, 1, 1], dtype=cp.int32),
        group_count=2,
    )

    def _fail(*_args, **_kwargs):
        raise AssertionError("device NativeGrouped coverage union should not host-normalize group codes")

    monkeypatch.setattr(dissolve_module, "_group_non_empty_counts", _fail)

    result = dissolve_module.execute_native_grouped_union(
        grouped,
        _geometries=(),
        method="coverage",
        owned=owned,
    )

    assert result is not None
    assert result.group_count == 2
    assert result.non_empty_groups == 2
    assert result.empty_groups == 0
    assert result.owned is not None
    assert result.geometries is None
    assert result.owned.residency is Residency.DEVICE
    actual = np.asarray(result.owned.to_shapely(), dtype=object)
    expected = np.asarray([box(0, 0, 2, 1), box(10, 10, 12, 11)], dtype=object)
    assert shapely.equals(actual, expected).tolist() == [
        True,
        True,
    ]


@pytest.mark.gpu
def test_execute_grouped_union_codes_batches_multi_group_unary_on_gpu(
    monkeypatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import vibespatial.constructive.binary_constructive as binary_constructive_module
    import vibespatial.constructive.union_all as union_all_module

    def _fail_serial_union_all(*_args, **_kwargs):
        raise AssertionError("multi-group unary dissolve should not run serial union_all per group")

    monkeypatch.setattr(union_all_module, "union_all_gpu_owned", _fail_serial_union_all)
    real_binary_constructive_owned = binary_constructive_module.binary_constructive_owned
    skip_contraction_flags = []

    def _record_binary_constructive_owned(*args, **kwargs):
        skip_contraction_flags.append(kwargs.get("_skip_polygon_contraction"))
        return real_binary_constructive_owned(*args, **kwargs)

    monkeypatch.setattr(
        binary_constructive_module,
        "binary_constructive_owned",
        _record_binary_constructive_owned,
    )

    values = [
        *[box(float(i) * 0.5, 0, float(i) * 0.5 + 1.0, 1.0) for i in range(12)],
        *[box(20.0 + float(i) * 0.5, 0, 21.0 + float(i) * 0.5, 1.0) for i in range(12)],
    ]
    owned = from_shapely_geometries(values, residency=Residency.DEVICE)

    class ExplodingGeometries:
        def __array__(self, dtype=None):
            raise AssertionError("owned unary grouped union should not materialize geometry objects")

    grouped = dissolve_module.execute_grouped_union_codes(
        ExplodingGeometries(),
        np.repeat(np.asarray([0, 1], dtype=np.int32), 12),
        group_count=2,
        method="unary",
        owned=owned,
    )

    assert grouped is not None
    assert grouped.geometries is None
    assert grouped.owned is not None
    assert grouped.owned.row_count == 2
    assert all(flag is True for flag in skip_contraction_flags)

    actual = np.asarray(grouped.owned.to_shapely(), dtype=object)
    expected = [
        shapely.union_all(np.asarray(values[:12], dtype=object)),
        shapely.union_all(np.asarray(values[12:], dtype=object)),
    ]
    assert bool(shapely.equals(actual[0], expected[0]))
    assert bool(shapely.equals(actual[1], expected[1]))


@pytest.mark.gpu
def test_small_grouped_constructive_reduce_batches_many_tiny_groups_on_gpu(
    monkeypatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import vibespatial.kernels.constructive.segmented_union as segmented_union_module

    group_sizes = np.asarray([2, 3, 4, 5, 6, 7, 8, 2], dtype=np.int32)
    group_offsets = np.concatenate(
        [np.asarray([0], dtype=np.int32), np.cumsum(group_sizes, dtype=np.int32)]
    )
    values = []
    for group_index, group_size in enumerate(group_sizes):
        x0 = float(group_index) * 20.0
        for row in range(int(group_size)):
            left = x0 + float(row) * 0.45
            values.append(box(left, 0.0, left + 1.0, 1.0))

    owned = from_shapely_geometries(values, residency=Residency.DEVICE)

    def _fail_serial(*_args, **_kwargs):
        raise AssertionError("many small grouped reductions should batch, not dispatch per group")

    monkeypatch.setattr(segmented_union_module, "_segmented_union_serial_gpu", _fail_serial)

    geopandas.clear_dispatch_events()
    result = segmented_union_all(
        owned,
        group_offsets,
        dispatch_mode=ExecutionMode.GPU,
    )
    events = geopandas.get_dispatch_events(clear=True)

    assert result.residency is Residency.DEVICE
    assert result.row_count == len(group_sizes)
    assert any(
        event.surface == "segmented_union_all"
        and event.operation == "segmented_union_strategy"
        and event.implementation == "gpu_grouped_overlay_many_small_groups"
        for event in events
    )

    actual = np.asarray(result.to_shapely(), dtype=object)
    expected = [
        shapely.union_all(
            np.asarray(values[int(start) : int(end)], dtype=object)
        )
        for start, end in zip(group_offsets[:-1], group_offsets[1:], strict=True)
    ]
    for got, want in zip(actual, expected, strict=True):
        got_norm = shapely.normalize(got)
        want_norm = shapely.normalize(want)
        assert bool(got_norm.equals_exact(want_norm, tolerance=1.0e-9))
        assert shapely.area(shapely.symmetric_difference(got, want)) == pytest.approx(
            0.0,
            abs=1.0e-9,
        )


@pytest.mark.gpu
def test_public_dissolve_then_convex_hull_uses_grouped_hull_rewrite() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    values = [
        *[box(float(i) * 0.55, 0.0, float(i) * 0.55 + 1.0, 1.0) for i in range(12)],
        *[box(20.0 + float(i) * 0.55, 0.0, 21.0 + float(i) * 0.55, 1.0) for i in range(12)],
    ]
    groups = np.repeat(np.asarray([0, 1], dtype=np.int32), 12)
    frame = geopandas.GeoDataFrame(
        {
            "group": groups,
            "value": np.arange(len(values), dtype=np.int32),
        },
        geometry=geopandas.GeoSeries(
            DeviceGeometryArray._from_owned(
                from_shapely_geometries(values, residency=Residency.DEVICE),
                crs="EPSG:3857",
            ),
            crs="EPSG:3857",
        ),
        crs="EPSG:3857",
    )

    geopandas.clear_dispatch_events()
    clear_rewrite_events()
    dissolved = frame.dissolve(
        by="group",
        aggfunc={"value": "sum"},
        method="unary",
    ).reset_index()
    actual = np.asarray(dissolved.geometry.convex_hull.array, dtype=object)
    dispatch_events = geopandas.get_dispatch_events(clear=True)
    rewrite_events = get_rewrite_events(clear=True)

    expected = [
        shapely.convex_hull(shapely.union_all(np.asarray(values[:12], dtype=object))),
        shapely.convex_hull(shapely.union_all(np.asarray(values[12:], dtype=object))),
    ]

    assert any(
        event.implementation == "grouped_dissolve_convex_hull_gpu"
        for event in dispatch_events
    )
    assert any(
        event.rule_name == "R12_dissolve_grouped_union_to_grouped_convex_hull"
        for event in rewrite_events
    )
    assert shapely.area(shapely.symmetric_difference(actual[0], expected[0])) == 0.0
    assert shapely.area(shapely.symmetric_difference(actual[1], expected[1])) == 0.0


def test_public_dissolve_matches_public_union_all_component_order_for_nybb_fixture() -> None:
    path = Path("tests/upstream/geopandas/tests/data/nybb_16a.zip")
    frame = geopandas.read_file(path)
    frame = frame[["geometry", "BoroName", "BoroCode"]]
    frame = frame.rename(columns={"geometry": "myshapes"})
    frame = frame.set_geometry("myshapes")
    frame["manhattan_bronx"] = 5
    frame.loc[3:4, "manhattan_bronx"] = 6

    dissolved = frame.dissolve("manhattan_bronx")
    expected = frame.loc[0:2].geometry.union_all()

    # The dissolve exact path should preserve the same MultiPolygon component
    # ordering as the standalone public union_all() path for the grouped input.
    assert shapely.to_wkb(dissolved.loc[5, "myshapes"]) == shapely.to_wkb(expected)


def test_evaluate_geopandas_dissolve_routes_small_buffered_line_unary_to_exact_cpu_rescue() -> None:
    frame = geopandas.GeoDataFrame(
        {
            "group": np.zeros(64, dtype=np.int32),
            "value": np.arange(64, dtype=np.int32),
            "geometry": [
                LineString([(float(i), 0.0), (float(i), 10.0)])
                for i in range(64)
            ],
        },
        crs="EPSG:3857",
    )
    buffered = frame.copy()
    buffered["geometry"] = buffered.geometry.buffer(0.5)

    clear_rewrite_events()
    geopandas.clear_fallback_events()
    actual = evaluate_geopandas_dissolve(
        buffered,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )
    rewrite_events = get_rewrite_events(clear=True)
    fallback_events = geopandas.get_fallback_events(clear=True)

    expected = evaluate_geopandas_dissolve(
        buffered,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="disjoint_subset",
        grid_size=None,
        agg_kwargs={},
    )

    assert not any(
        event.rule_name == "R8_dissolve_buffered_lines_to_disjoint_subset"
        for event in rewrite_events
    )
    assert any(
        event.surface == "geopandas.geodataframe.dissolve"
        and "small buffered-line dissolve" in event.reason
        for event in fallback_events
    )
    actual_geom = np.asarray(actual.geometry.array, dtype=object)[0]
    expected_geom = np.asarray(expected.geometry.array, dtype=object)[0]
    assert shapely.area(shapely.symmetric_difference(actual_geom, expected_geom)) == 0.0
    assert actual.iloc[0]["value"] == expected.iloc[0]["value"]


def test_duplicate_two_point_buffered_line_dissolve_prefers_small_cpu_rescue() -> None:
    lines = [
        LineString([(0.0, 0.0), (10.0, 0.0)]),
        LineString([(10.0, 0.0), (0.0, 0.0)]),
        LineString([(0.0, 5.0), (10.0, 5.0)]),
        LineString([(10.0, 5.0), (0.0, 5.0)]),
    ] * 32
    frame = geopandas.GeoDataFrame(
        {
            "group": np.zeros(len(lines), dtype=np.int32),
            "value": np.arange(len(lines), dtype=np.int32),
            "geometry": lines,
        },
        crs="EPSG:3857",
    )
    buffered = frame.copy()
    buffered["geometry"] = buffered.geometry.buffer(0.5)

    clear_rewrite_events()
    geopandas.clear_fallback_events()
    actual = evaluate_geopandas_dissolve(
        buffered,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )
    rewrite_events = get_rewrite_events(clear=True)
    fallback_events = geopandas.get_fallback_events(clear=True)

    assert not any(
        event.rule_name == "R9_dissolve_buffered_two_point_lines_exact_union"
        for event in rewrite_events
    )
    assert any(
        event.surface == "geopandas.geodataframe.dissolve"
        and "small buffered-line dissolve" in event.reason
        for event in fallback_events
    )
    actual_geom = np.asarray(actual.geometry.array, dtype=object)[0]
    expected_geom = shapely.union_all(np.asarray(buffered.geometry.array, dtype=object))
    assert shapely.area(shapely.symmetric_difference(actual_geom, expected_geom)) == 0.0


def test_evaluate_geopandas_dissolve_rewrites_duplicate_two_point_buffered_lines_to_exact_gpu_union(
    monkeypatch,
) -> None:
    if not has_gpu_runtime():
        return

    lines = [
        LineString([(0.0, 0.0), (10.0, 0.0)]),
        LineString([(10.0, 0.0), (0.0, 0.0)]),
        LineString([(0.0, 5.0), (10.0, 5.0)]),
        LineString([(10.0, 5.0), (0.0, 5.0)]),
    ] * 32
    frame = geopandas.GeoDataFrame(
        {
            "group": np.zeros(len(lines), dtype=np.int32),
            "value": np.arange(len(lines), dtype=np.int32),
        },
        geometry=geopandas.GeoSeries(
            DeviceGeometryArray._from_owned(
                from_shapely_geometries(lines, residency=Residency.DEVICE),
                crs="EPSG:3857",
            ),
            crs="EPSG:3857",
        ),
        crs="EPSG:3857",
    )
    buffered = frame.copy()
    buffered["geometry"] = buffered.geometry.buffer(0.5)

    def _fail(*_args, **_kwargs):
        raise AssertionError("duplicate two-point buffered-line dissolve should bypass the generic grouped union path")

    monkeypatch.setattr(dissolve_module, "execute_grouped_union_codes", _fail)

    clear_rewrite_events()
    geopandas.clear_fallback_events()
    actual = evaluate_geopandas_dissolve(
        buffered,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )
    rewrite_events = get_rewrite_events(clear=True)
    fallback_events = geopandas.get_fallback_events(clear=True)

    expected = geopandas.GeoDataFrame(
        {
            "geometry": [shapely.union_all(np.asarray(buffered.geometry.array, dtype=object))],
            "value": [0],
        },
        geometry="geometry",
        index=pd.Index([0], name="group"),
        crs=buffered.crs,
    )

    assert any(
        event.rule_name == "R9_dissolve_buffered_two_point_lines_exact_union"
        for event in rewrite_events
    )
    assert not any(
        event.surface == "geopandas.geodataframe.dissolve"
        and "small buffered-line dissolve" in event.reason
        for event in fallback_events
    )
    actual_geom = np.asarray(actual.geometry.array, dtype=object)[0]
    expected_geom = np.asarray(expected.geometry.array, dtype=object)[0]
    assert shapely.area(shapely.symmetric_difference(actual_geom, expected_geom)) == 0.0
    assert actual.iloc[0]["value"] == expected.iloc[0]["value"]


def test_dedupe_two_point_linestring_rows_prefers_host_metadata_when_available(
    monkeypatch,
) -> None:
    lines = [
        LineString([(0.0, 0.0), (10.0, 0.0)]),
        LineString([(10.0, 0.0), (0.0, 0.0)]),
        LineString([(0.0, 5.0), (10.0, 5.0)]),
        LineString([(10.0, 5.0), (0.0, 5.0)]),
        LineString([(2.0, 2.0), (2.0, 8.0)]),
    ] * 8
    owned = from_shapely_geometries(lines)
    owned._ensure_host_state()

    if dissolve_module.cp is not None:
        def _fail(*_args, **_kwargs):
            raise AssertionError("host-materialized dedupe should not call device-side lexsort")

        monkeypatch.setattr(dissolve_module.cp, "lexsort", _fail)

    unique_rows = dissolve_module._dedupe_two_point_linestring_rows_gpu(owned)

    assert unique_rows is not None
    deduped = np.asarray(owned.take(unique_rows.astype(np.int64, copy=False)).to_shapely(), dtype=object)
    assert deduped.shape == (3,)

    def _normalized_endpoints(geom) -> tuple[tuple[float, float], tuple[float, float]]:
        start = (float(geom.coords[0][0]), float(geom.coords[0][1]))
        end = (float(geom.coords[-1][0]), float(geom.coords[-1][1]))
        return (start, end) if start <= end else (end, start)

    assert {_normalized_endpoints(geom) for geom in deduped} == {
        ((0.0, 0.0), (10.0, 0.0)),
        ((0.0, 5.0), (10.0, 5.0)),
        ((2.0, 2.0), (2.0, 8.0)),
    }


def test_buffered_two_point_line_exact_union_rewrite_accepts_large_deduped_sets(
    monkeypatch,
) -> None:
    if not has_gpu_runtime():
        return

    unique_lines = [
        LineString([(0.0, float(i) * 4.0), (20.0, float(i) * 4.0)])
        for i in range(66)
    ]
    lines = unique_lines * 16
    frame = geopandas.GeoDataFrame(
        {
            "group": np.zeros(len(lines), dtype=np.int32),
            "value": np.arange(len(lines), dtype=np.int32),
        },
        geometry=geopandas.GeoSeries(
            DeviceGeometryArray._from_owned(
                from_shapely_geometries(lines, residency=Residency.DEVICE),
                crs="EPSG:3857",
            ),
            crs="EPSG:3857",
        ),
        crs="EPSG:3857",
    )
    buffered = frame.copy()
    buffered["geometry"] = buffered.geometry.buffer(0.5)

    def _fail(*_args, **_kwargs):
        raise AssertionError("large deduped buffered-line dissolve should bypass the generic grouped union path")

    monkeypatch.setattr(dissolve_module, "_BUFFERED_LINE_EXACT_CPU_MAX_ROWS", 1)
    monkeypatch.setattr(dissolve_module, "execute_grouped_union_codes", _fail)

    clear_rewrite_events()
    actual = evaluate_geopandas_dissolve(
        buffered,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )
    rewrite_events = get_rewrite_events(clear=True)

    assert any(
        event.rule_name == "R9_dissolve_buffered_two_point_lines_exact_union"
        for event in rewrite_events
    )
    assert getattr(actual.geometry.array, "_owned", None) is not None
    actual_geom = np.asarray(actual.geometry.array, dtype=object)[0]
    expected_geom = shapely.union_all(np.asarray(buffered.geometry.array, dtype=object))
    assert shapely.area(shapely.symmetric_difference(actual_geom, expected_geom)) == 0.0


def test_union_small_partial_rows_gpu_matches_union_all_for_two_partials() -> None:
    if not has_gpu_runtime():
        return

    partials = [
        from_shapely_geometries([box(0.0, 0.0, 2.0, 2.0)], residency=Residency.DEVICE),
        from_shapely_geometries([box(1.0, 0.0, 3.0, 2.0)], residency=Residency.DEVICE),
    ]

    actual = dissolve_module._union_small_partial_rows_gpu(partials)

    assert actual is not None
    assert actual.row_count == 1
    actual_geom = np.asarray(actual.to_shapely(), dtype=object)[0]
    expected_geom = shapely.union_all(
        np.asarray(
            [partial.to_shapely()[0] for partial in partials],
            dtype=object,
        )
    )
    assert shapely.area(shapely.symmetric_difference(actual_geom, expected_geom)) == 0.0


def test_union_small_partial_rows_gpu_defers_four_partials_to_union_all() -> None:
    if not has_gpu_runtime():
        return

    partials = [
        from_shapely_geometries([box(0.0, 0.0, 2.0, 2.0)], residency=Residency.DEVICE),
        from_shapely_geometries([box(1.0, 0.0, 3.0, 2.0)], residency=Residency.DEVICE),
        from_shapely_geometries([box(2.0, 0.0, 4.0, 2.0)], residency=Residency.DEVICE),
        from_shapely_geometries([box(3.0, 0.0, 5.0, 2.0)], residency=Residency.DEVICE),
    ]

    assert dissolve_module._union_small_partial_rows_gpu(partials) is None


def test_reorder_small_partial_union_groups_by_overlap_pairs_low_overlap_groups() -> None:
    expanded_bounds = np.asarray(
        [
            [0.0, 0.0, 4.0, 10.0],
            [2.0, 3.0, 12.0, 5.0],
            [10.0, 0.0, 14.0, 10.0],
            [2.0, 6.0, 12.0, 8.0],
        ],
        dtype=np.float64,
    )
    color_rows = [
        np.asarray([0], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
        np.asarray([2], dtype=np.int64),
        np.asarray([3], dtype=np.int64),
    ]

    reordered = dissolve_module._reorder_small_partial_union_groups_by_overlap(
        expanded_bounds,
        color_rows,
    )

    assert [rows.tolist() for rows in reordered] == [[0], [2], [1], [3]]


def test_reduce_partial_rows_gpu_uses_direct_tree_reduce_for_four_partials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        return

    union_all_module = importlib.import_module("vibespatial.constructive.union_all")

    partials = [
        from_shapely_geometries([box(0.0, 0.0, 2.0, 2.0)], residency=Residency.DEVICE),
        from_shapely_geometries([box(1.0, 0.0, 3.0, 2.0)], residency=Residency.DEVICE),
        from_shapely_geometries([box(2.0, 0.0, 4.0, 2.0)], residency=Residency.DEVICE),
        from_shapely_geometries([box(3.0, 0.0, 5.0, 2.0)], residency=Residency.DEVICE),
    ]

    monkeypatch.setattr(
        union_all_module,
        "union_all_gpu_owned",
        lambda *_args, **_kwargs: pytest.fail(
            "four partial rows should use direct tree reduction before generic union_all"
        ),
    )

    actual = dissolve_module._reduce_partial_rows_gpu(partials)

    assert actual is not None
    assert actual.row_count == 1
    actual_geom = np.asarray(actual.to_shapely(), dtype=object)[0]
    expected_geom = shapely.union_all(
        np.asarray(
            [partial.to_shapely()[0] for partial in partials],
            dtype=object,
        )
    )
    assert shapely.area(shapely.symmetric_difference(actual_geom, expected_geom)) == 0.0


def test_evaluate_geopandas_dissolve_does_not_rewrite_polygon_buffer_unary() -> None:
    frame = geopandas.GeoDataFrame(
        {
            "group": np.zeros(64, dtype=np.int32),
            "value": np.arange(64, dtype=np.int32),
            "geometry": [
                box(float(i), 0.0, float(i) + 0.5, 0.5)
                for i in range(64)
            ],
        },
        crs="EPSG:3857",
    )
    buffered = frame.copy()
    buffered["geometry"] = buffered.geometry.buffer(0.25)

    clear_rewrite_events()
    evaluate_geopandas_dissolve(
        buffered,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )
    rewrite_events = get_rewrite_events(clear=True)

    assert not any(
        event.rule_name == "R8_dissolve_buffered_lines_to_disjoint_subset"
        for event in rewrite_events
    )


@pytest.mark.gpu
def test_device_backed_buffered_line_dissolve_preserves_owned_backing() -> None:
    lines = [
        LineString([(float(i), 0.0), (float(i), 10.0)])
        for i in range(64)
    ]
    frame = geopandas.GeoDataFrame(
        {
            "group": np.zeros(64, dtype=np.int32),
            "value": np.arange(64, dtype=np.int32),
        },
        geometry=geopandas.GeoSeries(
            DeviceGeometryArray._from_owned(
                from_shapely_geometries(lines, residency=Residency.DEVICE),
                crs="EPSG:3857",
            ),
            crs="EPSG:3857",
        ),
        crs="EPSG:3857",
    )

    buffered = frame.copy()
    buffered["geometry"] = buffered.geometry.buffer(0.5)
    assert getattr(buffered.geometry.values, "_provenance", None) is not None

    clear_rewrite_events()
    geopandas.clear_fallback_events()
    result = evaluate_geopandas_dissolve(
        buffered,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )
    rewrite_events = get_rewrite_events(clear=True)
    fallback_events = geopandas.get_fallback_events(clear=True)

    assert len(result) == 1
    assert isinstance(result.geometry.values, DeviceGeometryArray)
    assert getattr(result.geometry.values, "_owned", None) is not None
    assert any(
        event.rule_name == "R9_dissolve_buffered_two_point_lines_exact_union"
        for event in rewrite_events
    )
    assert not any(
        event.surface == "geopandas.geodataframe.dissolve"
        and "small buffered-line dissolve" in event.reason
        for event in fallback_events
    )
    actual_geom = np.asarray(result.geometry.array, dtype=object)[0]
    expected_geom = shapely.union_all(np.asarray(buffered.geometry.array, dtype=object))
    assert shapely.area(shapely.symmetric_difference(actual_geom, expected_geom)) == 0.0


@pytest.mark.gpu
def test_device_backed_buffered_line_dissolve_strict_native_uses_exact_gpu_rewrite() -> None:
    lines = [
        LineString([(float(i), 0.0), (float(i), 10.0)])
        for i in range(64)
    ]
    frame = geopandas.GeoDataFrame(
        {
            "group": np.zeros(64, dtype=np.int32),
            "value": np.arange(64, dtype=np.int32),
        },
        geometry=geopandas.GeoSeries(
            DeviceGeometryArray._from_owned(
                from_shapely_geometries(lines, residency=Residency.DEVICE),
                crs="EPSG:3857",
            ),
            crs="EPSG:3857",
        ),
        crs="EPSG:3857",
    )

    buffered = frame.copy()
    buffered["geometry"] = buffered.geometry.buffer(0.5)
    assert getattr(buffered.geometry.values, "_provenance", None) is not None

    clear_rewrite_events()
    geopandas.clear_fallback_events()
    with strict_native_environment():
        result = evaluate_geopandas_dissolve(
            buffered,
            by="group",
            aggfunc="first",
            as_index=True,
            level=None,
            sort=True,
            observed=False,
            dropna=True,
            method="unary",
            grid_size=None,
            agg_kwargs={},
        )
    rewrite_events = get_rewrite_events(clear=True)
    fallback_events = geopandas.get_fallback_events(clear=True)

    assert isinstance(result.geometry.values, DeviceGeometryArray)
    assert any(
        event.rule_name == "R9_dissolve_buffered_two_point_lines_exact_union"
        for event in rewrite_events
    )
    assert not any(
        event.surface == "geopandas.geodataframe.dissolve"
        and "small buffered-line dissolve" in event.reason
        for event in fallback_events
    )


@pytest.mark.gpu
def test_device_backed_multivertex_buffered_line_dissolve_uses_exact_cpu_rescue() -> None:
    lines = [
        LineString(
            [
                (0.0, float(i)),
                (5.0, float(i) + 0.25),
                (10.0, float(i)),
                (15.0, float(i) + 0.25),
            ]
        )
        for i in range(64)
    ]
    frame = geopandas.GeoDataFrame(
        {
            "group": np.zeros(64, dtype=np.int32),
            "value": np.arange(64, dtype=np.int32),
        },
        geometry=geopandas.GeoSeries(
            DeviceGeometryArray._from_owned(
                from_shapely_geometries(lines, residency=Residency.DEVICE),
                crs="EPSG:3857",
            ),
            crs="EPSG:3857",
        ),
        crs="EPSG:3857",
    )

    buffered = frame.copy()
    buffered["geometry"] = buffered.geometry.buffer(0.5)

    clear_rewrite_events()
    geopandas.clear_fallback_events()
    result = evaluate_geopandas_dissolve(
        buffered,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )
    rewrite_events = get_rewrite_events(clear=True)
    fallback_events = geopandas.get_fallback_events(clear=True)

    assert not any(
        event.rule_name == "R8_dissolve_buffered_lines_to_disjoint_subset"
        for event in rewrite_events
    )
    assert any(
        event.surface == "geopandas.geodataframe.dissolve"
        and "small buffered-line dissolve" in event.reason
        and event.d2h_transfer
        for event in fallback_events
    )
    actual_geom = np.asarray(result.geometry.array, dtype=object)[0]
    expected_geom = shapely.union_all(np.asarray(buffered.geometry.array, dtype=object))
    assert shapely.area(shapely.symmetric_difference(actual_geom, expected_geom)) == 0.0


def test_grouped_union_geometry_result_prefers_owned_make_valid_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validity_module = importlib.import_module("vibespatial.constructive.validity")
    make_valid_module = importlib.import_module("vibespatial.constructive.make_valid_pipeline")

    original_owned = from_shapely_geometries([box(0.0, 0.0, 1.0, 1.0)])
    repaired_owned = from_shapely_geometries([box(0.0, 0.0, 2.0, 2.0)])

    def _fake_is_valid_owned(arg, *args, **kwargs):
        return np.asarray([arg is repaired_owned], dtype=bool)

    def _fake_make_valid_owned(*, owned, dispatch_mode, **kwargs):
        assert owned is original_owned
        return SimpleNamespace(
            owned=repaired_owned,
            geometries=np.asarray([box(0.0, 0.0, 2.0, 2.0)], dtype=object),
            repaired_rows=np.asarray([0], dtype=np.int32),
            selected=ExecutionMode.GPU,
        )

    monkeypatch.setattr(validity_module, "is_valid_owned", _fake_is_valid_owned)
    monkeypatch.setattr(make_valid_module, "make_valid_owned", _fake_make_valid_owned)

    grouped = dissolve_module.GroupedUnionResult(
        geometries=None,
        group_count=1,
        non_empty_groups=1,
        empty_groups=0,
        method=DissolveUnionMethod.UNARY,
        owned=original_owned,
    )

    result = dissolve_module._grouped_union_geometry_result(
        grouped,
        geometry_name="geometry",
        crs="EPSG:3857",
    )

    assert result.owned is repaired_owned
    assert result.series is None


def test_grouped_union_geometry_result_strict_native_raises_on_host_repair_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validity_module = importlib.import_module("vibespatial.constructive.validity")
    make_valid_module = importlib.import_module("vibespatial.constructive.make_valid_pipeline")

    owned = from_shapely_geometries([box(0.0, 0.0, 1.0, 1.0)])

    def _fake_is_valid_owned(*args, **kwargs):
        return np.asarray([False], dtype=bool)

    def _fake_make_valid_owned(*, owned, dispatch_mode, **kwargs):
        return SimpleNamespace(
            owned=None,
            geometries=np.asarray([box(0.0, 0.0, 1.0, 1.0)], dtype=object),
            repaired_rows=np.asarray([0], dtype=np.int32),
            selected=ExecutionMode.CPU,
        )

    monkeypatch.setattr(validity_module, "is_valid_owned", _fake_is_valid_owned)
    monkeypatch.setattr(make_valid_module, "make_valid_owned", _fake_make_valid_owned)

    grouped = dissolve_module.GroupedUnionResult(
        geometries=None,
        group_count=1,
        non_empty_groups=1,
        empty_groups=0,
        method=DissolveUnionMethod.UNARY,
        owned=owned,
    )

    with strict_native_environment(), pytest.raises(StrictNativeFallbackError):
        dissolve_module._grouped_union_geometry_result(
            grouped,
            geometry_name="geometry",
            crs="EPSG:3857",
        )


def test_grouped_union_geometry_result_make_valid_fallback_extracts_polygonal_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validity_module = importlib.import_module("vibespatial.constructive.validity")
    make_valid_module = importlib.import_module("vibespatial.constructive.make_valid_pipeline")

    owned = from_shapely_geometries([box(0.0, 0.0, 1.0, 1.0)])
    repaired = GeometryCollection(
        [
            box(0.0, 0.0, 2.0, 2.0),
            LineString([(0.0, 0.0), (2.0, 2.0)]),
        ]
    )

    monkeypatch.setattr(
        validity_module,
        "is_valid_owned",
        lambda *_args, **_kwargs: np.asarray([False], dtype=bool),
    )
    monkeypatch.setattr(
        make_valid_module,
        "make_valid_owned",
        lambda **_kwargs: SimpleNamespace(
            owned=None,
            geometries=np.asarray([repaired], dtype=object),
            repaired_rows=np.asarray([0], dtype=np.int32),
            selected=ExecutionMode.CPU,
        ),
    )

    grouped = dissolve_module.GroupedUnionResult(
        geometries=None,
        group_count=1,
        non_empty_groups=1,
        empty_groups=0,
        method=DissolveUnionMethod.UNARY,
        owned=owned,
    )

    result = dissolve_module._grouped_union_geometry_result(
        grouped,
        geometry_name="geometry",
        crs="EPSG:3857",
    )

    assert result.series is not None
    assert result.series.iloc[0].geom_type == "Polygon"
    assert result.series.iloc[0].equals(box(0.0, 0.0, 2.0, 2.0))


def test_execute_grouped_union_codes_recomputes_invalid_gpu_rows_from_original_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segmented_union_module = importlib.import_module("vibespatial.kernels.constructive.segmented_union")

    geoms = [box(0.0, 0.0, 1.0, 1.0), box(1.0, 0.0, 2.0, 1.0)]
    owned = from_shapely_geometries(geoms)
    invalid_union = shapely.Polygon([(0.0, 0.0), (2.0, 0.0), (0.0, 1.0), (2.0, 1.0), (0.0, 0.0)])

    monkeypatch.setattr(
        segmented_union_module,
        "segmented_union_all",
        lambda *_args, **_kwargs: from_shapely_geometries([invalid_union]),
    )

    grouped = dissolve_module.execute_grouped_union_codes(
        np.asarray(geoms, dtype=object),
        np.asarray([0, 0], dtype=np.int64),
        group_count=1,
        method=DissolveUnionMethod.UNARY,
        owned=owned,
    )

    assert grouped is not None
    assert grouped.owned is not None
    actual = grouped.owned.to_shapely()[0]
    expected = shapely.union_all(np.asarray(geoms, dtype=object))
    assert actual.geom_type == expected.geom_type
    assert shapely.equals(actual, expected)


def test_execute_grouped_box_union_gpu_owned_codes_builds_device_backed_coverage_rectangles() -> None:
    if not has_gpu_runtime():
        return

    geometry_array = GeometryArray.from_owned(
        from_shapely_geometries(
            [
                box(0, 0, 1, 1),
                box(1, 0, 2, 1),
                box(10, 10, 11, 11),
                box(11, 10, 12, 11),
            ]
        )
    )
    owned = getattr(geometry_array, "_owned", None)
    assert owned is not None

    grouped = dissolve_module.execute_grouped_box_union_gpu_owned_codes(
        pd.Index([0, 0, 1, 1], dtype="int32").to_numpy(),
        group_count=2,
        owned=owned,
    )

    assert grouped is not None
    assert grouped.group_count == 2
    assert grouped.non_empty_groups == 2
    assert grouped.owned is not None
    assert grouped.geometries is None
    actual = np.asarray(grouped.owned.to_shapely(), dtype=object)
    expected = np.asarray([box(0, 0, 2, 1), box(10, 10, 12, 11)], dtype=object)
    assert all(actual_geom.equals(expected_geom) for actual_geom, expected_geom in zip(actual, expected, strict=True))


def test_execute_grouped_box_union_gpu_owned_codes_accepts_fractional_rectangles() -> None:
    if not has_gpu_runtime():
        return

    geometry_array = GeometryArray.from_owned(
        from_shapely_geometries(
            [
                box(0.0, 0.0, 0.5, 1.0),
                box(0.5, 0.0, 1.0, 1.0),
                box(2.0, 0.0, 2.5, 1.0),
                box(2.5, 0.0, 3.0, 1.0),
            ]
        )
    )
    owned = getattr(geometry_array, "_owned", None)
    assert owned is not None

    grouped = dissolve_module.execute_grouped_box_union_gpu_owned_codes(
        np.asarray([0, 0, 1, 1], dtype=np.int32),
        group_count=2,
        owned=owned,
    )

    assert grouped is not None
    assert grouped.owned is not None
    assert grouped.geometries is None


def test_execute_grouped_box_union_gpu_owned_codes_accepts_unsorted_codes(
    monkeypatch,
) -> None:
    if not has_gpu_runtime():
        return

    cp = pytest.importorskip("cupy")
    owned = from_shapely_geometries(
        [
            box(0, 0, 1, 1),
            box(10, 10, 11, 11),
            box(1, 0, 2, 1),
            box(11, 10, 12, 11),
        ],
        residency=Residency.DEVICE,
    )
    owned._validity = None
    owned._tags = None
    owned._family_row_offsets = None

    def _fail_host_metadata(*_args, **_kwargs):
        raise AssertionError("unsorted owned coverage reducer should stay device-native")

    monkeypatch.setattr(type(owned), "_ensure_host_metadata", _fail_host_metadata)

    grouped = dissolve_module.execute_grouped_box_union_gpu_owned_codes(
        cp.asarray([0, 1, 0, 1], dtype=cp.int32),
        group_count=2,
        owned=owned,
    )

    assert grouped is not None
    assert grouped.owned is not None
    assert grouped.geometries is None
    assert grouped.owned._validity is None
    assert grouped.owned._tags is None
    assert grouped.owned._family_row_offsets is None
    monkeypatch.undo()
    actual = np.asarray(grouped.owned.to_shapely(), dtype=object)
    expected = np.asarray([box(0, 0, 2, 1), box(10, 10, 12, 11)], dtype=object)
    assert shapely.equals(actual, expected).tolist() == [True, True]


def test_execute_grouped_box_union_gpu_owned_codes_rejects_gapped_groups() -> None:
    if not has_gpu_runtime():
        return

    geometry_array = GeometryArray.from_owned(
        from_shapely_geometries(
            [
                box(0.0, 0.0, 1.0, 1.0),
                box(0.0, 2.0, 1.0, 3.0),
            ]
        )
    )
    owned = getattr(geometry_array, "_owned", None)
    assert owned is not None

    grouped = dissolve_module.execute_grouped_box_union_gpu_owned_codes(
        np.asarray([0, 0], dtype=np.int32),
        group_count=1,
        owned=owned,
    )

    assert grouped is None


def test_execute_grouped_union_codes_linestrings_skip_owned_segmented_union_fast_path(
    monkeypatch,
) -> None:
    geometry_array = GeometryArray.from_owned(
        from_shapely_geometries(
            [
                LineString([(0, 0), (1, 1)]),
                LineString([(1, 1), (2, 2)]),
            ]
        )
    )
    owned = getattr(geometry_array, "_owned", None)
    assert owned is not None

    import vibespatial.kernels.constructive.segmented_union as segmented_union_module

    def _fail(*_args, **_kwargs):
        raise AssertionError("linestring dissolve should not route through segmented_union_all")

    monkeypatch.setattr(segmented_union_module, "segmented_union_all", _fail)

    grouped = dissolve_module.execute_grouped_union_codes(
        geometry_array,
        pd.Index([0, 0], dtype="int32").to_numpy(),
        group_count=1,
        method="unary",
        owned=owned,
    )

    assert grouped is None


def test_join_heavy_synthetic_groups_match_between_coverage_and_unary() -> None:
    frame = _regular_polygons_frame(256)
    frame["group"] = pd.Categorical(np.arange(len(frame), dtype=np.int32) % 128)

    coverage = evaluate_geopandas_dissolve(
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
    unary = evaluate_geopandas_dissolve(
        frame,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=False,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )

    assert_geodataframe_equal(unary, coverage)


def test_join_heavy_direct_grouped_dissolve_matches_public_coverage_dissolve(
    monkeypatch,
) -> None:
    frame = _regular_polygons_frame(256)
    unique_right_index = np.arange(len(frame), dtype=np.int64)
    calls: list[DissolveUnionMethod] = []
    real_codes = dissolve_module.execute_grouped_union_codes

    def _count_codes(*args, **kwargs):
        calls.append(DissolveUnionMethod(kwargs["method"]))
        return real_codes(*args, **kwargs)

    def _fail(*_args, **_kwargs):
        raise AssertionError("join-heavy benchmark helper should not fall back to public dissolve here")

    monkeypatch.setattr("vibespatial.bench.pipeline.execute_grouped_union_codes", _count_codes)
    monkeypatch.setattr("vibespatial.bench.pipeline.evaluate_geopandas_dissolve", _fail)

    dissolved, used_direct = _dissolve_join_heavy_groups(
        frame.geometry,
        unique_right_index,
        scale=len(frame),
    )
    expected = evaluate_geopandas_dissolve(
        geopandas.GeoDataFrame(
            {
                "group": pd.Categorical(unique_right_index % 128),
                "geometry": frame.geometry,
            },
            geometry="geometry",
            crs=frame.crs,
        ),
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

    assert used_direct is True
    assert calls == [DissolveUnionMethod.COVERAGE]
    assert isinstance(dissolved, NativeTabularResult)
    assert_geodataframe_equal(dissolved.to_geodataframe(), expected)


@pytest.mark.gpu
def test_join_heavy_device_grouped_dissolve_matches_public_coverage_dissolve() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")

    frame = _regular_polygons_frame(256)
    owned = from_shapely_geometries(list(frame.geometry), residency=Residency.DEVICE)
    seed_all_validity_cache(owned)
    unique_right_index = cp.arange(len(frame), dtype=cp.int64)

    dissolved, used_direct = _dissolve_join_heavy_groups(
        GeometryNativeResult.from_owned(owned, crs=frame.crs),
        unique_right_index,
        scale=len(frame),
    )
    expected = evaluate_geopandas_dissolve(
        geopandas.GeoDataFrame(
            {
                "group": pd.Categorical(np.arange(len(frame), dtype=np.int64) % 128),
                "geometry": frame.geometry,
            },
            geometry="geometry",
            crs=frame.crs,
        ),
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

    assert used_direct is True
    assert isinstance(dissolved, NativeTabularResult)
    assert_geodataframe_equal(dissolved.to_geodataframe(), expected)
