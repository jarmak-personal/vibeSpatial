from __future__ import annotations

import importlib

import pandas as pd
from shapely.geometry import LineString, Point, box

import vibespatial.api as geopandas
from vibespatial import (
    DissolveUnionMethod,
    benchmark_dissolve_pipeline,
    fusion_plan_for_dissolve,
    plan_dissolve_pipeline,
)
from vibespatial.api.geometry_array import GeometryArray
from vibespatial.api.testing import assert_geodataframe_equal
from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.overlay.dissolve import evaluate_geopandas_dissolve, execute_grouped_union
from vibespatial.runtime.fusion import IntermediateDisposition

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
