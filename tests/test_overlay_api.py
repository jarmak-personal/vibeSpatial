from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import shapely
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point, Polygon, box

import vibespatial
from vibespatial.api import GeoDataFrame, GeoSeries, read_file
from vibespatial.api.geometry_array import GeometryArray
from vibespatial.api.tools.overlay import overlay
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    TAG_FAMILIES,
    DiagnosticKind,
    from_shapely_geometries,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.residency import Residency
from vibespatial.testing import strict_native_environment

overlay_module = importlib.import_module("vibespatial.api.tools.overlay")
overlay_gpu_module = importlib.import_module("vibespatial.overlay.gpu")
_SHOOTOUT_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "shootout"
if str(_SHOOTOUT_DIR) not in sys.path:
    sys.path.insert(0, str(_SHOOTOUT_DIR))


def _assert_owned_row_mapping_valid(series: GeoSeries) -> None:
    owned = getattr(series.values, "_owned", None)
    assert owned is not None
    state = owned._ensure_device_state()

    def _host_array(values):
        return values.get() if hasattr(values, "get") else np.asarray(values)

    def _collect_bad_rows(
        validity: np.ndarray,
        tags: np.ndarray,
        family_row_offsets: np.ndarray,
        family_counts: dict[GeometryFamily, int],
    ) -> list[tuple[int, str, int, int]]:
        bad_rows: list[tuple[int, str, int, int]] = []
        for row_index in np.flatnonzero(validity):
            family = TAG_FAMILIES.get(int(tags[row_index]))
            assert family is not None, f"valid row {row_index} has null tag"
            family_count = family_counts[family]
            family_row = int(family_row_offsets[row_index])
            if family_row < 0 or family_row >= family_count:
                bad_rows.append((int(row_index), family.value, family_row, family_count))
                if len(bad_rows) >= 8:
                    break
        return bad_rows

    host_bad_rows = _collect_bad_rows(
        np.asarray(owned.validity, dtype=bool),
        np.asarray(owned.tags),
        np.asarray(owned.family_row_offsets),
        {family: owned.families[family].row_count for family in owned.families},
    )
    device_bad_rows = _collect_bad_rows(
        _host_array(state.validity),
        _host_array(state.tags),
        _host_array(state.family_row_offsets),
        {
            family: int(buffer.geometry_offsets.size) - 1
            for family, buffer in state.families.items()
        },
    )

    assert not host_bad_rows, f"owned host row mapping invalid: {host_bad_rows}"
    assert not device_bad_rows, f"owned device row mapping invalid: {device_bad_rows}"


def test_geometry_array_owned_supports_spatial_input_without_materialization() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    owned = from_shapely_geometries(
        [Polygon([(0, 0), (2, 0), (2, 2), (0, 0)])],
        residency=Residency.DEVICE,
    )
    owned.diagnostics.clear()

    array = GeometryArray.from_owned(owned)

    assert array.supports_owned_spatial_input() is True
    assert [
        event for event in owned.diagnostics
        if event.kind == DiagnosticKind.MATERIALIZATION
    ] == []


def test_overlay_union_keep_geom_type_false_drops_empty_difference_rows() -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1, 2, 3]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )

    result = overlay(left, right, how="union", keep_geom_type=False)

    assert len(result) == 6
    assert result.geometry.notna().all()


def test_overlay_union_reuses_intersecting_pair_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )

    real_query = overlay_module._intersecting_index_pairs
    calls = 0

    def _counting_query(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_query(*args, **kwargs)

    monkeypatch.setattr(overlay_module, "_intersecting_index_pairs", _counting_query)
    result = overlay(left, right, how="union")

    assert len(result) == 7
    assert calls == 1


def test_overlay_intersection_keep_geom_type_preserves_geometry_collection_boundary() -> None:
    left = GeoDataFrame(
        {
            "left": [0, 1],
            "geometry": [
                box(0, 0, 1, 1),
                box(1, 1, 3, 3).union(box(1, 3, 5, 5)),
            ],
        }
    )
    right = GeoDataFrame(
        {
            "right": [0, 1],
            "geometry": [
                box(0, 0, 1, 1),
                box(3, 1, 4, 2).union(box(4, 1, 5, 4)),
            ],
        }
    )

    kept = overlay(left, right, keep_geom_type=True)
    assert kept.geometry.geom_type.tolist() == ["Polygon", "Polygon"]

    all_geoms = overlay(left, right, keep_geom_type=False)
    assert all_geoms.geometry.geom_type.tolist() == [
        "Polygon",
        "Point",
        "GeometryCollection",
    ]
    assert all_geoms.geometry.iloc[2].equals(
        GeometryCollection([box(4, 3, 5, 4), LineString([(3, 1), (3, 2)])])
    )


def test_overlay_intersecting_index_pairs_can_bypass_public_sindex_query_for_small_owned_polygons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(10, 10), (12, 10), (12, 12), (10, 12)]),
            ]
        ),
    )
    left_owned = left.geometry.values.to_owned()
    right_owned = right.geometry.values.to_owned()

    class _Pairs:
        left_indices = np.asarray([0], dtype=np.int32)
        right_indices = np.asarray([0], dtype=np.int32)

    monkeypatch.setattr(
        overlay_module,
        "generate_bounds_pairs",
        lambda *args, **kwargs: _Pairs(),
    )
    monkeypatch.setattr(
        right.sindex,
        "query",
        lambda *args, **kwargs: pytest.fail("public sindex.query should not run on the bbox fast path"),
    )

    idx1, idx2 = overlay_module._intersecting_index_pairs(
        left,
        right,
        left_owned=left_owned,
        right_owned=right_owned,
    )

    assert idx1.tolist() == [0]
    assert idx2.tolist() == [0]


def test_overlay_symmetric_difference_reuses_intersecting_pair_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )

    real_query = overlay_module._intersecting_index_pairs
    calls = 0

    def _counting_query(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_query(*args, **kwargs)

    monkeypatch.setattr(overlay_module, "_intersecting_index_pairs", _counting_query)
    result = overlay(left, right, how="symmetric_difference")

    assert len(result) == 4
    assert calls == 1


def test_overlay_identity_reuses_intersecting_pair_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )

    real_query = overlay_module._intersecting_index_pairs
    calls = 0

    def _counting_query(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_query(*args, **kwargs)

    monkeypatch.setattr(overlay_module, "_intersecting_index_pairs", _counting_query)
    result = overlay(left, right, how="identity")

    assert len(result) == 5
    assert calls == 1


def test_overlay_intersection_uses_public_sindex_query_in_strict_mode() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )

    with strict_native_environment():
        vibespatial.clear_dispatch_events()
        result = overlay(left, right, how="intersection")
        events = vibespatial.get_dispatch_events(clear=True)

    assert len(result) == 3
    assert any(
        event.surface in {"geopandas.sindex.query", "geopandas.overlay.sindex"}
        for event in events
    )


def test_overlay_intersection_drops_empty_rows_after_bbox_false_positive_in_strict_mode() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1]},
        geometry=GeoSeries(
            [
                Polygon([(0, 0), (2, 0), (0, 2), (0, 0)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1]},
        geometry=GeoSeries(
            [
                Polygon([(2, 2), (4, 2), (2, 4), (2, 2)]),
            ]
        ),
    )

    real_query = overlay_module._intersecting_index_pairs
    try:
        overlay_module._intersecting_index_pairs = lambda *args, **kwargs: (
            np.asarray([0], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
        )
        with strict_native_environment():
            result = overlay(left, right, how="intersection")
    finally:
        overlay_module._intersecting_index_pairs = real_query

    assert len(result) == 0


def test_overlay_union_promotes_small_pairwise_intersection_in_strict_mode() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )

    with strict_native_environment():
        vibespatial.clear_dispatch_events()
        result = overlay(left, right, how="union")
        events = vibespatial.get_dispatch_events(clear=True)

    assert len(result) == 7
    assert any(
        event.surface in {"geopandas.sindex.query", "geopandas.overlay.sindex"}
        for event in events
    )
    assert any(
        event.surface == "geopandas.array.intersection"
        and event.selected is ExecutionMode.GPU
        for event in events
    )
    assert any(
        event.surface == "geopandas.array.difference"
        and event.selected is ExecutionMode.GPU
        for event in events
    )


def test_overlay_difference_keeps_split_polygon_result_in_single_row_under_strict_mode() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1]},
        geometry=GeoSeries(
            [
                Polygon([(0, 0), (10, 0), (10, 4), (0, 4), (0, 0)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(2, 0), (4, 0), (4, 4), (2, 4), (2, 0)]),
                Polygon([(6, 0), (8, 0), (8, 4), (6, 4), (6, 0)]),
            ]
        ),
    )

    with strict_native_environment():
        vibespatial.clear_dispatch_events()
        result = overlay(left, right, how="difference")
        events = vibespatial.get_dispatch_events(clear=True)

    assert len(result) == 1
    assert getattr(result.geometry.values, "_owned", None) is not None
    assert result.geometry.iloc[0].geom_type in {"Polygon", "MultiPolygon"}
    assert result.geometry.iloc[0].area > 0
    assert any(
        event.surface == "geopandas.array.difference"
        and event.selected is ExecutionMode.GPU
        for event in events
    )


def test_group_source_rows_from_offsets_expands_group_ids() -> None:
    got = overlay_module._group_source_rows_from_offsets(
        np.asarray([0, 2, 2, 5], dtype=np.int64)
    )
    assert got.tolist() == [0, 0, 2, 2, 2]


def test_grouped_overlay_difference_owned_builds_one_grouped_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = from_shapely_geometries(
        [
            box(0, 0, 10, 10),
            box(20, 0, 30, 10),
        ]
    )
    right = from_shapely_geometries(
        [
            box(1, 1, 2, 2),
            box(21, 1, 22, 2),
            box(23, 1, 24, 2),
        ]
    )

    build_calls: list[dict[str, object]] = []
    materialize_calls: list[dict[str, object]] = []
    sentinel = from_shapely_geometries(
        [
            box(0, 0, 9, 9),
            box(20, 0, 29, 9),
        ]
    )

    def _fake_build(left_batch, right_batch, **kwargs):
        build_calls.append(
            {
                "left_rows": left_batch.row_count,
                "right_rows": right_batch.row_count,
                **kwargs,
            }
        )
        return object()

    def _fake_materialize(plan, **kwargs):
        materialize_calls.append(kwargs)
        return sentinel, ExecutionMode.GPU

    monkeypatch.setattr(overlay_gpu_module, "_build_overlay_execution_plan", _fake_build)
    monkeypatch.setattr(overlay_gpu_module, "_materialize_overlay_execution_plan", _fake_materialize)

    result = overlay_module._grouped_overlay_difference_owned(
        left,
        right,
        np.asarray([0, 1, 3], dtype=np.int64),
        dispatch_mode=ExecutionMode.AUTO,
    )

    assert result is sentinel
    assert len(build_calls) == 1
    assert build_calls[0]["left_rows"] == 2
    assert build_calls[0]["right_rows"] == 3
    assert build_calls[0]["_row_isolated"] is True
    assert np.array_equal(
        np.asarray(build_calls[0]["_right_geometry_source_rows"]),
        np.asarray([0, 1, 1], dtype=np.int32),
    )
    assert len(materialize_calls) == 1
    assert materialize_calls[0]["operation"] == "difference"
    assert materialize_calls[0]["preserve_row_count"] == 2


def test_grouped_overlay_difference_owned_falls_back_to_sequential_exact_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = from_shapely_geometries([box(0, 0, 10, 10)])
    right = from_shapely_geometries([box(1, 1, 2, 2), box(3, 3, 4, 4)])
    sentinel = from_shapely_geometries([box(0, 0, 8, 8)])
    fallback_calls: list[dict[str, object]] = []

    def _raising_build(*args, **kwargs):
        raise RuntimeError("boom")

    def _fake_sequential(left_batch, right_batch, group_offsets, *, dispatch_mode):
        fallback_calls.append(
            {
                "left_rows": left_batch.row_count,
                "right_rows": right_batch.row_count,
                "group_offsets": np.asarray(group_offsets).tolist(),
                "dispatch_mode": dispatch_mode,
            }
        )
        return sentinel

    monkeypatch.setattr(overlay_gpu_module, "_build_overlay_execution_plan", _raising_build)
    monkeypatch.setattr(overlay_module, "_sequential_grouped_difference_owned", _fake_sequential)

    result = overlay_module._grouped_overlay_difference_owned(
        left,
        right,
        np.asarray([0, 2], dtype=np.int64),
        dispatch_mode=ExecutionMode.GPU,
    )

    assert result is sentinel
    assert fallback_calls == [
        {
            "left_rows": 1,
            "right_rows": 2,
            "group_offsets": [0, 2],
            "dispatch_mode": ExecutionMode.GPU,
        }
    ]


def test_overlay_difference_uses_grouped_overlay_plan_for_grouped_neighbors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(0, 0), (10, 0), (10, 6), (0, 6), (0, 0)]),
                Polygon([(12, 0), (22, 0), (22, 6), (12, 6), (12, 0)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1, 2, 3, 4]},
        geometry=GeoSeries(
            [
                Polygon([(2, 0), (4, 0), (4, 6), (2, 6), (2, 0)]),
                Polygon([(6, 0), (8, 0), (8, 6), (6, 6), (6, 0)]),
                Polygon([(14, 0), (16, 0), (16, 6), (14, 6), (14, 0)]),
                Polygon([(18, 0), (20, 0), (20, 6), (18, 6), (18, 0)]),
            ]
        ),
    )

    from vibespatial.constructive import binary_constructive as constructive_module

    build_calls = 0
    materialize_calls = 0
    difference_calls = 0
    original_build = overlay_gpu_module._build_overlay_execution_plan
    original_materialize = overlay_gpu_module._materialize_overlay_execution_plan
    original = constructive_module.binary_constructive_owned

    def _wrapped_build(*args, **kwargs):
        nonlocal build_calls
        build_calls += 1
        return original_build(*args, **kwargs)

    def _wrapped_materialize(*args, **kwargs):
        nonlocal materialize_calls
        if kwargs.get("operation") == "difference":
            materialize_calls += 1
        return original_materialize(*args, **kwargs)

    def _wrapped_binary_constructive_owned(*args, **kwargs):
        nonlocal difference_calls
        if args and args[0] == "difference":
            difference_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        overlay_gpu_module,
        "_build_overlay_execution_plan",
        _wrapped_build,
    )
    monkeypatch.setattr(
        overlay_gpu_module,
        "_materialize_overlay_execution_plan",
        _wrapped_materialize,
    )
    monkeypatch.setattr(
        constructive_module,
        "binary_constructive_owned",
        _wrapped_binary_constructive_owned,
    )

    with strict_native_environment():
        result = overlay(left, right, how="difference")

    assert len(result) == 2
    assert build_calls == 1
    assert materialize_calls == 1
    assert difference_calls == 0


def test_overlay_difference_matches_union_for_overlapping_gt2_neighbors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1]},
        geometry=GeoSeries(
            [Polygon([(0, 0), (12, 0), (12, 8), (0, 8), (0, 0)])]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1, 2, 3]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (6, 1), (6, 7), (1, 7), (1, 1)]),
                Polygon([(4, 1), (9, 1), (9, 7), (4, 7), (4, 1)]),
                Polygon([(7, 1), (11, 1), (11, 7), (7, 7), (7, 1)]),
            ]
        ),
    )

    from vibespatial.constructive import binary_constructive as constructive_module
    from vibespatial.kernels.constructive import segmented_union as segmented_union_module

    original = segmented_union_module.segmented_union_all
    original_build = overlay_gpu_module._build_overlay_execution_plan
    original_materialize = overlay_gpu_module._materialize_overlay_execution_plan
    original_binary_constructive_owned = constructive_module.binary_constructive_owned
    grouped_union_calls = 0
    grouped_plan_calls = 0
    grouped_materialize_calls = 0
    difference_calls = 0

    def _counted_segmented_union_all(*args, **kwargs):
        nonlocal grouped_union_calls
        grouped_union_calls += 1
        return original(*args, **kwargs)

    def _counted_build(*args, **kwargs):
        nonlocal grouped_plan_calls
        grouped_plan_calls += 1
        return original_build(*args, **kwargs)

    def _counted_materialize(*args, **kwargs):
        nonlocal grouped_materialize_calls
        if kwargs.get("operation") == "difference":
            grouped_materialize_calls += 1
        return original_materialize(*args, **kwargs)

    def _guarded_binary_constructive_owned(*args, **kwargs):
        nonlocal difference_calls
        if args and args[0] == "difference":
            difference_calls += 1
        return original_binary_constructive_owned(*args, **kwargs)

    monkeypatch.setattr(
        segmented_union_module,
        "segmented_union_all",
        _counted_segmented_union_all,
    )
    monkeypatch.setattr(
        overlay_gpu_module,
        "_build_overlay_execution_plan",
        _counted_build,
    )
    monkeypatch.setattr(
        overlay_gpu_module,
        "_materialize_overlay_execution_plan",
        _counted_materialize,
    )
    monkeypatch.setattr(
        constructive_module,
        "binary_constructive_owned",
        _guarded_binary_constructive_owned,
    )

    with strict_native_environment():
        result = overlay(left, right, how="difference")

    expected = shapely.difference(
        left.geometry.iloc[0],
        shapely.union_all(np.asarray(right.geometry, dtype=object)),
    )
    assert len(result) == 1
    assert grouped_union_calls == 0
    assert grouped_plan_calls == 1
    assert grouped_materialize_calls == 1
    assert difference_calls == 0
    assert shapely.symmetric_difference(result.geometry.iloc[0], expected).area < 1e-8


def test_overlay_difference_redevelopment_like_followup_overlay_stays_strict_native(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from _data import setup_fixtures

    monkeypatch.setenv("VSBENCH_SCALE", "10k")
    fixtures = setup_fixtures(tmp_path)

    parcels = vibespatial.read_parquet(fixtures["parcels"])
    zones = vibespatial.read_parquet(fixtures["zones"])
    exclusions = vibespatial.read_parquet(fixtures["exclusion_zones"])
    transit = read_file(fixtures["transit"])

    bounds = parcels.total_bounds
    dx = (bounds[2] - bounds[0]) * 0.15
    dy = (bounds[3] - bounds[1]) * 0.15
    clip_box = box(bounds[0] + dx, bounds[1] + dy, bounds[2] - dx, bounds[3] - dy)

    with strict_native_environment():
        study_parcels = vibespatial.clip(parcels, clip_box)
        study_parcels = study_parcels[
            study_parcels.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        ].copy()
        developable = overlay(study_parcels, exclusions, how="difference")
        _assert_owned_row_mapping_valid(developable.geometry)

        transit_buffers = transit.copy()
        transit_buffers["geometry"] = transit_buffers.geometry.buffer(150.0)
        near_transit = vibespatial.sjoin(
            developable,
            transit_buffers[["station_id", "geometry"]],
            predicate="intersects",
        )
        candidate_rows = near_transit.index.unique()
        candidates = (
            developable.loc[candidate_rows].copy()
            if len(candidate_rows) > 0
            else developable.iloc[:0].copy()
        )
        _assert_owned_row_mapping_valid(candidates.geometry)
        zoned = overlay(candidates, zones[["zone_type", "geometry"]], how="intersection")

    assert zoned is not None


def test_binary_constructive_intersection_stays_strict_native_for_multipolygon_polygon_batch() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.constructive.binary_constructive import binary_constructive_owned

    left = from_shapely_geometries(
        [
            shapely.MultiPolygon(
                [
                    box(0, 0, 3, 3),
                    box(5, 0, 8, 3),
                ]
            ),
            shapely.MultiPolygon(
                [
                    box(10, 0, 14, 4),
                    box(12, 6, 16, 10),
                ]
            ),
        ]
    )
    right = from_shapely_geometries(
        [
            box(1, 1, 6, 2.5),
            box(11, 1, 15, 8),
        ]
    )

    with strict_native_environment():
        result = binary_constructive_owned(
            "intersection",
            left,
            right,
            dispatch_mode=ExecutionMode.GPU,
        )

    got = result.to_shapely()
    expected = shapely.intersection(
        np.asarray(left.to_shapely(), dtype=object),
        np.asarray(right.to_shapely(), dtype=object),
    ).tolist()
    assert len(got) == len(expected) == 2
    for actual, oracle in zip(got, expected, strict=True):
        if shapely.is_empty(oracle):
            assert actual is None or shapely.is_empty(actual)
            continue
        assert actual is not None
        assert shapely.normalize(actual).equals_exact(
            shapely.normalize(oracle),
            tolerance=1e-9,
        )


def test_binary_constructive_difference_skips_mixed_dispatch_for_polygonal_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from shapely.geometry import MultiPolygon

    from vibespatial.constructive import binary_constructive as constructive_module
    from vibespatial.geometry.owned import from_shapely_geometries

    left = from_shapely_geometries(
        [
            Polygon([(0, 0), (6, 0), (6, 4), (0, 4), (0, 0)]),
            MultiPolygon(
                [
                    Polygon([(10, 0), (13, 0), (13, 4), (10, 4), (10, 0)]),
                    Polygon([(14, 0), (17, 0), (17, 4), (14, 4), (14, 0)]),
                ]
            ),
        ]
    )
    right = from_shapely_geometries(
        [
            Polygon([(2, 0), (4, 0), (4, 4), (2, 4), (2, 0)]),
            Polygon([(11, 1), (16, 1), (16, 3), (11, 3), (11, 1)]),
        ]
    )

    mixed_dispatch_called = False
    original_mixed_dispatch = constructive_module._dispatch_mixed_binary_constructive_gpu

    def _wrapped_mixed_dispatch(*args, **kwargs):
        nonlocal mixed_dispatch_called
        mixed_dispatch_called = True
        return original_mixed_dispatch(*args, **kwargs)

    monkeypatch.setattr(
        constructive_module,
        "_dispatch_mixed_binary_constructive_gpu",
        _wrapped_mixed_dispatch,
    )

    with strict_native_environment():
        result = constructive_module.binary_constructive_owned(
            "difference",
            left,
            right,
            dispatch_mode=ExecutionMode.GPU,
        )

    assert result.row_count == 2
    assert not mixed_dispatch_called


def test_overlay_difference_polygon_line_keeps_noded_polygon_boundary_under_strict_mode() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col3": [1, 2]},
        geometry=GeoSeries(
            [
                LineString([(2, 0), (2, 4), (6, 4)]),
                LineString([(0, 3), (6, 3)]),
            ]
        ),
    )

    with strict_native_environment():
        result = overlay(left, right, how="difference", keep_geom_type=True)

    expected = GeoSeries(
        [
            Polygon([(1, 1), (2, 1), (3, 1), (3, 3), (2, 3), (1, 3), (1, 1)]),
            Polygon([(3, 3), (5, 3), (5, 4), (5, 5), (3, 5), (3, 4), (3, 3)]),
        ],
        crs=left.crs,
    )

    assert len(result) == 2
    for got, exp in zip(result.geometry, expected, strict=True):
        assert got.normalize().equals_exact(exp.normalize(), tolerance=1e-9)


def test_overlay_identity_polygon_point_keeps_difference_rows_separate_under_strict_mode() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col4": [1, 2]},
        geometry=GeoSeries([Point((2, 2)), Point((3, 3))]),
    )

    with strict_native_environment():
        result = overlay(left, right, how="identity", keep_geom_type=True)

    expected = GeoSeries(
        [
            Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
            Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
        ],
        crs=left.crs,
    )

    assert len(result) == 2
    for got, exp in zip(result.geometry, expected, strict=True):
        assert got.normalize().equals_exact(exp.normalize(), tolerance=1e-9)


def test_overlay_difference_line_polygon_splits_outside_segments_under_strict_mode() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col3": [1, 2]},
        geometry=GeoSeries(
            [
                LineString([(2, 0), (2, 4), (6, 4)]),
                LineString([(0, 3), (6, 3)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )

    with strict_native_environment():
        result = overlay(left, right, how="difference", keep_geom_type=True)

    expected = GeoSeries(
        [
            MultiLineString(
                [
                    [(2, 0), (2, 1)],
                    [(2, 3), (2, 4), (3, 4)],
                    [(5, 4), (6, 4)],
                ]
            ),
            MultiLineString(
                [
                    [(0, 3), (1, 3)],
                    [(5, 3), (6, 3)],
                ]
            ),
        ],
        crs=left.crs,
    )

    assert len(result) == 2
    for got, exp in zip(result.geometry, expected, strict=True):
        assert got.normalize().equals_exact(exp.normalize(), tolerance=1e-9)


def test_overlay_union_collapses_split_polygon_fragments_under_strict_mode() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1]},
        geometry=GeoSeries(
            [
                Polygon([(0, 0), (10, 0), (10, 4), (0, 4), (0, 0)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(2, 0), (4, 0), (4, 4), (2, 4), (2, 0)]),
                Polygon([(6, 0), (8, 0), (8, 4), (6, 4), (6, 0)]),
            ]
        ),
    )

    with strict_native_environment():
        result = overlay(left, right, how="union")

    assert len(result) == 3


def test_overlay_union_survives_strict_native_mode_for_small_pairwise_polygons() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
                Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )

    with strict_native_environment():
        result = overlay(left, right, how="union")

    assert len(result) == 7
    assert result.geometry.notna().all()


def test_overlay_union_warns_but_succeeds_on_crs_mismatch_in_strict_mode() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1]},
        geometry=GeoSeries(
            [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            ]
        ),
        crs=4326,
    )
    right = GeoDataFrame(
        {"col2": [1]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
            ]
        ),
        crs=3857,
    )

    with strict_native_environment():
        with pytest.warns(UserWarning, match="CRS mismatch between the CRS"):
            result = overlay(left, right, how="union")

    assert result.crs == left.crs
    assert len(result) == 3
    assert result.geometry.notna().all()


def test_overlay_intersection_warns_on_dropped_lower_dim_results_in_strict_mode() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )
    right = GeoDataFrame(
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )

    with strict_native_environment():
        with pytest.warns(UserWarning, match="`keep_geom_type=True` in overlay"):
            result = overlay(left, right, keep_geom_type=None)

    assert list(result.geom_type) == ["Polygon", "Polygon"]
    assert len(result) == 2


def test_overlay_intersection_keeps_touch_line_when_keep_geom_type_false_in_strict_mode() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )
    right = GeoDataFrame(
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )

    with strict_native_environment():
        result = overlay(left, right, keep_geom_type=False)

    geom_types = result.geom_type.tolist()
    assert geom_types.count("Polygon") == 2
    assert any(geom_type in {"LineString", "MultiLineString"} for geom_type in geom_types)


def test_overlay_intersection_keep_geom_type_true_skips_geometry_collection_cpu_fallback() -> None:
    left = GeoDataFrame(
        {"left": [0, 1]},
        geometry=GeoSeries(
            [
                box(0, 0, 1, 1),
                box(1, 1, 3, 3).union(box(1, 3, 5, 5)),
            ]
        ),
    )
    right = GeoDataFrame(
        {"right": [0, 1]},
        geometry=GeoSeries(
            [
                box(0, 0, 1, 1),
                box(3, 1, 4, 2).union(box(4, 1, 5, 4)),
            ]
        ),
    )

    result = overlay(left, right, how="intersection", keep_geom_type=True)

    assert set(result.geometry.geom_type.unique()) <= {"Polygon", "MultiPolygon"}


def test_overlay_intersection_keep_geom_type_true_skips_full_lower_dim_assembly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )
    right = GeoDataFrame(
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )

    monkeypatch.setattr(
        overlay_module,
        "_assemble_polygon_intersection_rows_with_lower_dim",
        lambda *args, **kwargs: pytest.fail("full lower-dim assembly should be skipped"),
    )

    with strict_native_environment():
        result = overlay(left, right, keep_geom_type=True)

    assert list(result.geom_type) == ["Polygon", "Polygon"]


def test_overlay_intersection_keep_geom_type_none_warns_for_geometry_collection_rows() -> None:
    left = GeoDataFrame(
        {"left": [0, 1]},
        geometry=GeoSeries(
            [
                box(0, 0, 1, 1),
                box(1, 1, 3, 3).union(box(1, 3, 5, 5)),
            ]
        ),
    )
    right = GeoDataFrame(
        {"right": [0, 1]},
        geometry=GeoSeries(
            [
                box(0, 0, 1, 1),
                box(3, 1, 4, 2).union(box(4, 1, 5, 4)),
            ]
        ),
    )

    with pytest.warns(UserWarning, match="`keep_geom_type=True` in overlay"):
        result = overlay(left, right, how="intersection", keep_geom_type=None)

    assert set(result.geometry.geom_type.unique()) <= {"Polygon", "MultiPolygon"}


def test_overlay_intersection_keep_geom_type_none_strict_warning_matches_host_count() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"left": [0, 1]},
        geometry=GeoSeries(
            [
                box(0, 0, 1, 1),
                box(1, 1, 3, 3).union(box(1, 3, 5, 5)),
            ]
        ),
    )
    right = GeoDataFrame(
        {"right": [0, 1]},
        geometry=GeoSeries(
            [
                box(0, 0, 1, 1),
                box(3, 1, 4, 2).union(box(4, 1, 5, 4)),
            ]
        ),
    )

    with pytest.warns(UserWarning, match="`keep_geom_type=True` in overlay") as host_warning:
        overlay(left, right, how="intersection", keep_geom_type=None)

    with strict_native_environment():
        with pytest.warns(UserWarning, match="`keep_geom_type=True` in overlay") as strict_warning:
            overlay(left, right, how="intersection", keep_geom_type=None)

    assert str(strict_warning[0].message) == str(host_warning[0].message)


def test_overlay_intersection_keep_geom_type_none_skips_full_lower_dim_assembly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )
    right = GeoDataFrame(
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )

    monkeypatch.setattr(
        overlay_module,
        "_assemble_polygon_intersection_rows_with_lower_dim",
        lambda *args, **kwargs: pytest.fail("full lower-dim assembly should be skipped"),
    )

    with strict_native_environment():
        with pytest.warns(UserWarning, match="`keep_geom_type=True` in overlay"):
            result = overlay(left, right, keep_geom_type=None)

    assert list(result.geom_type) == ["Polygon", "Polygon"]


def test_keep_geom_type_filter_uses_geometry_array_values_not_geoseries_wrappers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left_pairs = GeoSeries(
        [
            box(0, 0, 1, 1),
            box(0, 0, 1, 1),
            box(0, 0, 1, 1),
        ]
    )
    right_pairs = GeoSeries(
        [
            box(0, 0, 1, 1),
            box(1, 0, 2, 1),
            box(2, 2, 3, 3),
        ]
    )
    area_pairs = GeoSeries(
        [
            GeometryCollection([box(0, 0, 1, 1), LineString([(0, 0), (1, 0)])]),
            LineString([(1, 0), (1, 1)]),
            None,
        ]
    )

    def _fail(*_args, **_kwargs):
        pytest.fail("GeoSeries wrapper path should stay cold")

    monkeypatch.setattr(GeoSeries, "__array__", _fail, raising=False)
    monkeypatch.setattr(GeoSeries, "isna", _fail, raising=False)
    monkeypatch.setattr(GeoSeries, "is_empty", property(lambda self: _fail()))
    monkeypatch.setattr(GeoSeries, "geom_type", property(lambda self: _fail()))

    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            left_pairs,
            right_pairs,
            area_pairs,
            keep_geom_type_warning=True,
        )
    )

    assert keep_mask.tolist() == [True, False, False]
    assert dropped == 1
    assert len(filtered) == 1
    filtered_values = np.asarray(filtered.array, dtype=object)
    assert shapely.get_type_id(filtered_values).tolist() == [3]


def test_keep_geom_type_filter_preserves_owned_results_without_array_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left_pairs = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 1, 1),
                    box(0, 0, 1, 1),
                    box(0, 0, 1, 1),
                ]
            )
        )
    )
    right_pairs = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 1, 1),
                    box(1, 0, 2, 1),
                    box(2, 2, 3, 3),
                ]
            )
        )
    )
    area_pairs = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 1, 1),
                    LineString([(1, 0), (1, 1)]),
                    None,
                ]
            )
        )
    )

    def _fail(*_args, **_kwargs):
        pytest.fail("owned keep-geom-type filter should not materialize full geometry arrays")

    monkeypatch.setattr(GeometryArray, "__array__", _fail, raising=False)

    filtered, dropped, keep_mask = overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
        left_pairs,
        right_pairs,
        area_pairs,
        keep_geom_type_warning=False,
    )

    assert keep_mask.tolist() == [True, False, False]
    assert dropped == 0
    assert len(filtered) == 1
    assert getattr(filtered.values, "_owned", None) is not None


def test_keep_geom_type_filter_falls_back_when_owned_family_empty_mask_is_inconsistent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left_pairs = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 1, 1),
                    box(0, 0, 1, 1),
                    box(0, 0, 1, 1),
                ]
            )
        )
    )
    right_pairs = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 1, 1),
                    box(1, 0, 2, 1),
                    box(2, 2, 3, 3),
                ]
            )
        )
    )
    area_pairs = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 1, 1),
                    LineString([(1, 0), (1, 1)]),
                    None,
                ]
            )
        )
    )

    left_values = np.asarray(left_pairs.array, dtype=object)
    right_values = np.asarray(right_pairs.array, dtype=object)
    area_values = np.asarray(area_pairs.array, dtype=object)
    area_owned = getattr(area_pairs.values, "_owned", None)
    assert area_owned is not None
    object.__setattr__(
        area_owned.families[GeometryFamily.POLYGON],
        "empty_mask",
        np.empty(0, dtype=bool),
    )

    def _object_values(series: GeoSeries) -> np.ndarray:
        if series is left_pairs:
            return left_values
        if series is right_pairs:
            return right_values
        if series is area_pairs:
            return area_values
        raise AssertionError("unexpected GeoSeries passed to object-value helper")

    def _take_object_values(series: GeoSeries, rows: np.ndarray) -> np.ndarray:
        return _object_values(series)[rows]

    monkeypatch.setattr(overlay_module, "_geoseries_object_values", _object_values)
    monkeypatch.setattr(overlay_module, "_take_geoseries_object_values", _take_object_values)

    filtered, dropped, keep_mask = overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
        left_pairs,
        right_pairs,
        area_pairs,
        keep_geom_type_warning=True,
    )

    assert keep_mask.tolist() == [True, False, False]
    assert dropped == 1
    assert len(filtered) == 1


def test_overlay_intersection_many_vs_one_remainder_prefers_rowwise_overlay_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2, 3]},
        geometry=GeoSeries(
            [
                Polygon([(-1, 1), (2, 1), (2, 4), (-1, 4), (-1, 1)]),
                Polygon([(1, -1), (4, -1), (4, 2), (1, 2), (1, -1)]),
                Polygon([(3, 3), (6, 3), (6, 6), (3, 6), (3, 3)]),
            ]
        ),
    )
    right = GeoDataFrame(
        geometry=GeoSeries(
            [
                Polygon([(0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6), (0, 0)]),
            ]
        ),
    )

    from vibespatial.constructive import binary_constructive as constructive_module

    called = False
    original = constructive_module._dispatch_polygon_intersection_overlay_rowwise_gpu

    def _wrapped(*args, **kwargs):
        nonlocal called
        called = True
        return original(*args, **kwargs)

    monkeypatch.setattr(
        constructive_module,
        "_dispatch_polygon_intersection_overlay_rowwise_gpu",
        _wrapped,
    )

    with strict_native_environment():
        result = overlay(left, right, how="intersection", keep_geom_type=True)

    assert called
    assert list(result["col1"]) == [1, 2]


def test_overlay_intersection_many_vs_one_remainder_avoids_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2, 3]},
        geometry=GeoSeries(
            [
                Polygon([(-1, 1), (2, 1), (2, 4), (-1, 4), (-1, 1)]),
                Polygon([(1, -1), (4, -1), (4, 2), (1, 2), (1, -1)]),
                Polygon([(3, 3), (6, 3), (6, 6), (3, 6), (3, 3)]),
            ]
        ),
    )
    right = GeoDataFrame(
        geometry=GeoSeries(
            [
                Polygon([(0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6), (0, 0)]),
            ]
        ),
    )

    original = overlay_module.record_fallback_event

    def _wrapped_record_fallback_event(*args, **kwargs):
        reason = kwargs.get("reason", "")
        if "many-vs-one remainder" in str(reason):
            pytest.fail("many-vs-one polygon remainder should stay on GPU")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        overlay_module,
        "record_fallback_event",
        _wrapped_record_fallback_event,
    )

    result = overlay(left, right, how="intersection", keep_geom_type=True)

    assert list(result["col1"]) == [1, 2]


def test_overlay_intersection_many_vs_one_fast_path_retries_after_first_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2, 3]},
        geometry=GeoSeries(
            [
                Polygon([(-1, 1), (2, 1), (2, 4), (-1, 4), (-1, 1)]),
                Polygon([(1, -1), (4, -1), (4, 2), (1, 2), (1, -1)]),
                Polygon([(3, 3), (6, 3), (6, 6), (3, 6), (3, 3)]),
            ]
        ),
    )
    right = GeoDataFrame(
        geometry=GeoSeries(
            [
                Polygon([(0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6), (0, 0)]),
            ]
        ),
    )

    original = overlay_module._many_vs_one_intersection_owned
    calls = 0

    def _wrapped_many_vs_one(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise MemoryError("synthetic many-vs-one fast-path failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        overlay_module,
        "_many_vs_one_intersection_owned",
        _wrapped_many_vs_one,
    )

    with strict_native_environment():
        result = overlay(left, right, how="intersection", keep_geom_type=True)

    assert calls == 2
    assert list(result["col1"]) == [1, 2]


def test_overlay_difference_survives_strict_native_mode_for_small_overlap_polygons() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [Point(0, 0).buffer(1, quad_segs=2), Point(1.5, 0).buffer(1, quad_segs=2)]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1, 2]},
        geometry=GeoSeries(
            [Point(1, 1).buffer(1, quad_segs=2), Point(2, 2).buffer(1, quad_segs=2)]
        ),
    )

    with strict_native_environment():
        result = overlay(left, right, how="difference")

    assert len(result) == 2
    assert result.geometry.notna().all()


def test_overlay_strict_nybb_single_pair_intersection_matches_host() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    data = os.path.join(
        os.path.dirname(__file__),
        "upstream",
        "geopandas",
        "tests",
        "data",
    )
    overlay_data = os.path.join(data, "overlay", "nybb_qgis")
    left = read_file(f"zip://{os.path.join(data, 'nybb_16a.zip')}").iloc[[4]].copy()
    right = read_file(os.path.join(overlay_data, "polydf2.shp")).iloc[[8]].copy()

    with strict_native_environment():
        result = overlay(left, right, how="intersection")

    expected = left.geometry.iloc[0].intersection(right.geometry.iloc[0])
    assert len(result) == 1
    assert result.geometry.iloc[0].geom_type == expected.geom_type
    assert result.geometry.iloc[0].normalize().equals_exact(expected.normalize(), tolerance=1e-6)
