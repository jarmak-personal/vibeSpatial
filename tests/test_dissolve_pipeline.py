from __future__ import annotations

import pandas as pd
from shapely.geometry import Point

import vibespatial.api as geopandas
from vibespatial.api.testing import assert_geodataframe_equal

from vibespatial import (
    DissolveUnionMethod,
    benchmark_dissolve_pipeline,
    fusion_plan_for_dissolve,
    plan_dissolve_pipeline,
)
from vibespatial.dissolve_pipeline import evaluate_geopandas_dissolve, execute_grouped_union
from vibespatial.fusion import IntermediateDisposition


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


def test_benchmark_dissolve_pipeline_reports_group_count() -> None:
    frame = geopandas.GeoDataFrame(
        {
            "group": [0, 0, 1, 1],
            "value": [1, 2, 3, 4],
            "geometry": [Point(0, 0), Point(1, 1), Point(10, 10), Point(11, 11)],
        }
    )

    benchmark = benchmark_dissolve_pipeline(frame, by="group", dataset="points")

    assert benchmark.rows == 4
    assert benchmark.groups == 2
    assert benchmark.pipeline_elapsed_seconds >= 0.0
    assert benchmark.baseline_elapsed_seconds >= 0.0
