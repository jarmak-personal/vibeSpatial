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
from vibespatial.api._native_results import NativeTabularResult
from vibespatial.api.geometry_array import GeometryArray
from vibespatial.api.testing import assert_geodataframe_equal
from vibespatial.bench.pipeline import _dissolve_join_heavy_groups, _regular_polygons_frame
from vibespatial.geometry.device_array import DeviceGeometryArray
from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.overlay.dissolve import evaluate_geopandas_dissolve, execute_grouped_union
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


def test_evaluate_geopandas_dissolve_rewrites_buffered_line_unary_to_disjoint_subset() -> None:
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

    assert any(
        event.rule_name == "R8_dissolve_buffered_lines_to_disjoint_subset"
        for event in rewrite_events
    )
    actual_geom = np.asarray(actual.geometry.array, dtype=object)[0]
    expected_geom = np.asarray(expected.geometry.array, dtype=object)[0]
    assert shapely.area(shapely.symmetric_difference(actual_geom, expected_geom)) == 0.0
    assert actual.iloc[0]["value"] == expected.iloc[0]["value"]


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

    assert len(result) == 1
    assert isinstance(result.geometry.values, DeviceGeometryArray)
    assert getattr(result.geometry.values, "_owned", None) is not None
    assert not any(
        event.rule_name == "R8_dissolve_buffered_lines_to_disjoint_subset"
        for event in rewrite_events
    )


def test_device_backed_buffered_line_dissolve_strict_native_skips_host_rewrite() -> None:
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

    assert len(result) == 1
    assert isinstance(result.geometry.values, DeviceGeometryArray)
    assert getattr(result.geometry.values, "_owned", None) is not None
    assert not any(
        event.rule_name == "R8_dissolve_buffered_lines_to_disjoint_subset"
        for event in rewrite_events
    )


def test_device_backed_multivertex_buffered_line_dissolve_rewrites_to_disjoint_subset() -> None:
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

    assert any(
        event.rule_name == "R8_dissolve_buffered_lines_to_disjoint_subset"
        for event in rewrite_events
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

    frame = _regular_polygons_frame(32)
    geometry_array = GeometryArray.from_owned(
        from_shapely_geometries(list(frame.geometry))
    )
    owned = getattr(geometry_array, "_owned", None)
    assert owned is not None

    grouped = dissolve_module.execute_grouped_box_union_gpu_owned_codes(
        np.arange(len(frame), dtype=np.int32) % 16,
        group_count=16,
        owned=owned,
    )

    assert grouped is not None
    assert grouped.owned is not None
    assert grouped.geometries is None


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
