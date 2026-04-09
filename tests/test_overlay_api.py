from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import shapely
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point, Polygon, box

import vibespatial
import vibespatial.api as geopandas
import vibespatial.api._native_results as native_results_module
from vibespatial import write_geoparquet
from vibespatial.api import GeoDataFrame, GeoSeries, read_file
from vibespatial.api._native_results import (
    ConcatConstructiveResult,
    LeftConstructiveFragment,
    LeftConstructiveResult,
    NativeAttributeTable,
    NativeTabularResult,
    PairwiseConstructiveFragment,
    PairwiseConstructiveResult,
    SymmetricDifferenceConstructiveResult,
    to_native_tabular_result,
)
from vibespatial.api.geometry_array import GeometryArray
from vibespatial.api.testing import assert_geodataframe_equal
from vibespatial.api.tools.overlay import overlay
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.device_array import DeviceGeometryArray
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
overlay_split_module = importlib.import_module("vibespatial.overlay.split")
segment_primitives_module = importlib.import_module("vibespatial.spatial.segment_primitives")
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


def test_overlay_difference_grouped_plan_avoids_brittle_same_row_fast_path(
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

    def _should_not_run(*args, **kwargs):
        raise AssertionError("grouped overlay plan should not depend on same-row warp candidates")

    monkeypatch.setattr(
        segment_primitives_module,
        "_generate_candidates_gpu_same_row_warp",
        _should_not_run,
    )

    expected = [
        shapely.difference(
            left.geometry.iloc[0],
            shapely.union_all(np.asarray(right.geometry.iloc[:2], dtype=object)),
        ),
        shapely.difference(
            left.geometry.iloc[1],
            shapely.union_all(np.asarray(right.geometry.iloc[2:], dtype=object)),
        ),
    ]

    with strict_native_environment():
        vibespatial.clear_dispatch_events()
        result = overlay(left, right, how="difference")
        events = vibespatial.get_dispatch_events(clear=True)

    assert len(result) == 2
    assert any(
        event.implementation == "grouped_overlay_difference_gpu"
        and event.selected is ExecutionMode.GPU
        for event in events
    )
    assert not any(
        event.implementation == "grouped_overlay_difference_plan_build_failed_gpu"
        for event in events
    )
    for got, want in zip(np.asarray(result.geometry, dtype=object), expected, strict=True):
        assert shapely.symmetric_difference(got, want).area < 1e-8


def test_grouped_overlay_difference_forces_gpu_segment_classification(
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

    original = overlay_split_module.classify_segment_intersections
    seen_dispatch_modes: list[ExecutionMode] = []

    def _wrapped_classify(*args, **kwargs):
        if kwargs.get("_require_same_row"):
            seen_dispatch_modes.append(kwargs["dispatch_mode"])
        return original(*args, **kwargs)

    monkeypatch.setattr(
        overlay_split_module,
        "classify_segment_intersections",
        _wrapped_classify,
    )

    with strict_native_environment():
        result = overlay(left, right, how="difference")

    assert len(result) == 2
    assert seen_dispatch_modes
    assert all(mode is ExecutionMode.GPU for mode in seen_dispatch_modes)


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


def test_overlay_union_keeps_boundary_touching_difference_rows_separate_in_strict_native_mode() -> None:
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
        {"col2": [1, 2, 3]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )

    with strict_native_environment():
        result = overlay(left, right, how="union", keep_geom_type=True)

    result = result.sort_values(["col1", "col2"], na_position="first").reset_index(drop=True)
    assert len(result) == 3
    assert result.geometry.iloc[0].equals(
        Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)])
    )


def test_overlay_symmetric_difference_boundary_touches_preserve_polygon_in_strict_native_mode() -> None:
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
        {"col2": [1, 2, 3]},
        geometry=GeoSeries(
            [
                Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
                Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)]),
                Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
            ]
        ),
    )

    with strict_native_environment():
        result = overlay(left, right, how="symmetric_difference", keep_geom_type=False)

    assert len(result) == 1
    assert result.geometry.iloc[0].equals(
        Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)])
    )


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


def test_overlay_intersection_device_backed_auto_stays_on_gpu_boundary() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"left": [0, 1]},
        geometry=DeviceGeometryArray._from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 2, 2),
                    box(2, 2, 4, 4),
                ],
                residency=Residency.DEVICE,
            )
        ),
    )
    right = GeoDataFrame(
        {"right": [0, 1]},
        geometry=DeviceGeometryArray._from_owned(
            from_shapely_geometries(
                [
                    box(1, 1, 3, 3),
                    box(3, 3, 5, 5),
                ],
                residency=Residency.DEVICE,
            )
        ),
    )

    vibespatial.clear_dispatch_events()
    result = overlay(left, right, how="intersection")
    events = vibespatial.get_dispatch_events(clear=True)

    assert len(result) == 3
    assert any(
        event.surface == "geopandas.array.intersection"
        and event.selected is ExecutionMode.GPU
        for event in events
    )
    assert any(
        event.surface == "geopandas.overlay"
        and event.selected is ExecutionMode.GPU
        for event in events
    )
    assert not any(
        event.surface == "geopandas.overlay"
        and event.implementation == "shapely_host"
        for event in events
    )
    assert not any(
        event.surface == "geopandas.array.make_valid"
        and event.selected is ExecutionMode.CPU
        for event in events
    )


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


def test_overlay_intersection_many_vs_one_remainder_prefers_direct_row_isolated_overlay(
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

    vibespatial.clear_dispatch_events()
    with strict_native_environment():
        result = overlay(left, right, how="intersection", keep_geom_type=True)
    events = vibespatial.get_dispatch_events(clear=True)

    assert list(result["col1"]) == [1, 2]
    assert any(
        event.surface == "geopandas.overlay"
        and event.operation == "overlay_intersection"
        and event.implementation == "owned_dispatch"
        and getattr(getattr(event, "selected", None), "value", None) == "gpu"
        for event in events
    )
    assert not any(
        event.surface == "geopandas.overlay"
        and event.implementation == "shapely_host"
        for event in events
    )


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


def test_overlay_intersection_many_vs_one_auto_keeps_public_path_on_gpu() -> None:
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

    vibespatial.clear_dispatch_events()
    result = overlay(left, right, how="intersection", keep_geom_type=True)
    events = vibespatial.get_dispatch_events(clear=True)

    assert list(result["col1"]) == [1, 2]
    assert not any(
        event.surface == "vibespatial.predicates.binary"
        and event.operation == "covered_by"
        for event in events
    )
    assert any(
        event.surface == "geopandas.overlay"
        and event.operation == "overlay_intersection"
        and event.implementation == "owned_dispatch"
        and getattr(getattr(event, "selected", None), "value", None) == "gpu"
        for event in events
    )
    assert not any(
        event.surface == "geopandas.overlay"
        and event.implementation == "shapely_host"
        for event in events
    )


def test_overlay_intersection_many_vs_one_small_remainder_uses_broadcast_right_exact_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left_owned = from_shapely_geometries(
        [
            Polygon([(-1, 1), (2, 1), (2, 4), (-1, 4), (-1, 1)]),
            Polygon([(1, -1), (4, -1), (4, 2), (1, 2), (1, -1)]),
            Polygon([(3, 3), (6, 3), (6, 6), (3, 6), (3, 3)]),
        ],
        residency=Residency.DEVICE,
    )
    right_owned = from_shapely_geometries(
        [Polygon([(0, 0), (6, 0), (6, 2), (2, 2), (2, 6), (0, 6), (0, 0)])],
        residency=Residency.DEVICE,
    )

    binary_module = importlib.import_module("vibespatial.constructive.binary_constructive")
    original = binary_module._dispatch_polygon_intersection_overlay_broadcast_right_gpu
    exact_calls: list[tuple[int, int]] = []

    def _wrapped_broadcast_exact(
        left_arg,
        right_arg,
        *,
        dispatch_mode=ExecutionMode.GPU,
        _cached_right_segments=None,
    ):
        exact_calls.append((left_arg.row_count, right_arg.row_count))
        return original(
            left_arg,
            right_arg,
            dispatch_mode=dispatch_mode,
            _cached_right_segments=_cached_right_segments,
        )

    monkeypatch.setattr(
        binary_module,
        "_dispatch_polygon_intersection_overlay_broadcast_right_gpu",
        _wrapped_broadcast_exact,
    )
    monkeypatch.setattr(
        binary_module,
        "_dispatch_polygon_intersection_overlay_rowwise_gpu",
        lambda *args, **kwargs: pytest.fail(
            "many-vs-one exact remainder should not materialize a tiled right operand "
            "and fall back to the legacy rowwise helper"
        ),
    )
    monkeypatch.setattr(
        overlay_module,
        "record_dispatch_event",
        lambda *args, **kwargs: None,
    )

    result = overlay_module._many_vs_one_intersection_owned(
        left_owned,
        right_owned,
        0,
    )

    assert exact_calls == [(3, 1)]
    assert result.row_count == 3


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


def test_overlay_intersection_few_right_fast_path_batches_exact_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.constructive import binary_constructive as constructive_module

    left = GeoDataFrame(
        {"col1": np.arange(24, dtype=np.int32)},
        geometry=GeoSeries([box(i, 0, i + 1, 1) for i in range(24)]),
    )
    right = GeoDataFrame(
        {"zone_type": ["A", "B", "C"]},
        geometry=GeoSeries(
            [
                Polygon(
                    [(0, 0), (16, 0), (16, 2), (0, 2)],
                    holes=[[(4, 0.5), (6, 0.5), (6, 1.5), (4, 1.5)]],
                ),
                Polygon(
                    [(8, 0), (24, 0), (24, 2), (8, 2)],
                    holes=[[(12, 0.5), (14, 0.5), (14, 1.5), (12, 1.5)]],
                ),
                Polygon(
                    [(16, 0), (32, 0), (32, 2), (16, 2)],
                    holes=[[(20, 0.5), (22, 0.5), (22, 1.5), (20, 1.5)]],
                ),
            ]
        ),
    )
    left_owned = left.geometry.values.to_owned()
    right_owned = right.geometry.values.to_owned()
    idx1 = np.arange(24, dtype=np.int32)
    idx2 = np.repeat(np.arange(3, dtype=np.int32), 8)

    exact_calls: list[int] = []

    def _fake_rowwise_exact(left_arg, right_arg, *, dispatch_mode=ExecutionMode.GPU):
        exact_calls.append(left_arg.row_count)
        geoms = [
            box(float(5000 + i), 0.0, float(5000 + i + 0.5), 0.5)
            for i in range(left_arg.row_count)
        ]
        return from_shapely_geometries(geoms, residency=Residency.DEVICE)

    monkeypatch.setattr(
        overlay_module,
        "_prepare_many_vs_one_intersection_chunks",
        lambda *args, **kwargs: pytest.fail(
            "few-right intersection should not decompose into many-vs-one preparations"
        ),
    )
    monkeypatch.setattr(
        constructive_module,
        "_dispatch_polygon_intersection_overlay_rowwise_gpu",
        _fake_rowwise_exact,
    )
    monkeypatch.setattr(
        constructive_module,
        "binary_constructive_owned",
        lambda *args, **kwargs: pytest.fail(
            "few-right public intersection should fuse exact leftovers "
            "through the rowwise helper"
        ),
    )

    result, used_owned = overlay_module._overlay_intersection(
        left,
        right,
        left_owned=left_owned,
        right_owned=right_owned,
        _prefer_exact_polygon_gpu=True,
        _index_result=(idx1, idx2),
    )

    assert used_owned is True
    assert exact_calls == [24]
    assert result["col1"].tolist() == idx1.tolist()
    assert result["zone_type"].tolist() == ["A"] * 8 + ["B"] * 8 + ["C"] * 8
    expected = [
        box(float(5000 + i), 0.0, float(5000 + i + 0.5), 0.5)
        for i in range(24)
    ]
    for got_geom, expected_geom in zip(result.geometry, expected, strict=True):
        assert got_geom.normalize().equals(expected_geom.normalize())


def test_overlay_intersection_native_defers_attribute_assembly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
    )

    real_assembler = overlay_module._assemble_intersection_attributes
    assemble_calls = 0

    def _counting_assembler(*args, **kwargs):
        nonlocal assemble_calls
        assemble_calls += 1
        return real_assembler(*args, **kwargs)

    monkeypatch.setattr(
        overlay_module,
        "_assemble_intersection_attributes",
        _counting_assembler,
    )

    native_result, used_owned = overlay_module._overlay_intersection_native(left, right)

    assert isinstance(native_result, PairwiseConstructiveResult)
    assert assemble_calls == 0

    materialized = native_result.to_geodataframe(
        left.reset_index(drop=True),
        right.reset_index(drop=True),
        attribute_assembler=overlay_module._assemble_intersection_attributes,
    )
    wrapped, wrapped_used = overlay_module._overlay_intersection(left, right)

    assert assemble_calls == 2
    assert used_owned is wrapped_used
    assert materialized["col1"].tolist() == wrapped["col1"].tolist()
    assert materialized["col2"].tolist() == wrapped["col2"].tolist()
    assert len(materialized) == len(wrapped) == 2
    assert all(
        got.normalize().equals(expected.normalize())
        for got, expected in zip(materialized.geometry, wrapped.geometry, strict=True)
    )


def test_overlay_difference_native_defers_left_frame_materialization() -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10]},
        geometry=GeoSeries([box(1, 1, 4, 4)]),
    )

    native_result, used_owned = overlay_module._overlay_difference_native(left, right)

    assert isinstance(native_result, LeftConstructiveResult)

    materialized = native_result.to_geodataframe(left)
    wrapped, wrapped_used = overlay_module._overlay_difference(left, right)

    assert used_owned is wrapped_used
    assert materialized["col1"].tolist() == wrapped["col1"].tolist()
    assert len(materialized) == len(wrapped)
    assert all(
        got.normalize().equals(expected.normalize())
        for got, expected in zip(materialized.geometry, wrapped.geometry, strict=True)
    )


def test_overlay_intersection_export_result_defers_fragment_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
    )

    real_pairwise = PairwiseConstructiveResult.to_geodataframe
    pairwise_calls = 0

    def _count_pairwise(self, *args, **kwargs):
        nonlocal pairwise_calls
        pairwise_calls += 1
        return real_pairwise(self, *args, **kwargs)

    monkeypatch.setattr(PairwiseConstructiveResult, "to_geodataframe", _count_pairwise)

    export_result, used_owned = overlay_module._overlay_intersection_export_result(left, right)

    assert isinstance(export_result, PairwiseConstructiveFragment)
    assert pairwise_calls == 0

    materialized = export_result.to_geodataframe()
    wrapped, wrapped_used = overlay_module._overlay_intersection(left, right)

    assert pairwise_calls == 2
    assert used_owned is wrapped_used
    assert_geodataframe_equal(materialized, wrapped, normalize=True, check_column_type=False)


def test_overlay_difference_export_result_defers_fragment_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10]},
        geometry=GeoSeries([box(1, 1, 4, 4)]),
    )

    real_left = LeftConstructiveResult.to_geodataframe
    left_calls = 0

    def _count_left(self, *args, **kwargs):
        nonlocal left_calls
        left_calls += 1
        return real_left(self, *args, **kwargs)

    monkeypatch.setattr(LeftConstructiveResult, "to_geodataframe", _count_left)

    export_result, used_owned = overlay_module._overlay_difference_export_result(left, right)

    assert isinstance(export_result, LeftConstructiveFragment)
    assert left_calls == 0

    materialized = export_result.to_geodataframe()
    wrapped, wrapped_used = overlay_module._overlay_difference(left, right)

    assert left_calls == 2
    assert used_owned is wrapped_used
    assert_geodataframe_equal(materialized, wrapped, normalize=True, check_column_type=False)


def test_overlay_identity_native_defers_fragment_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
    )

    real_pairwise = PairwiseConstructiveResult.to_geodataframe
    real_left = LeftConstructiveResult.to_geodataframe
    pairwise_calls = 0
    left_calls = 0

    def _count_pairwise(self, *args, **kwargs):
        nonlocal pairwise_calls
        pairwise_calls += 1
        return real_pairwise(self, *args, **kwargs)

    def _count_left(self, *args, **kwargs):
        nonlocal left_calls
        left_calls += 1
        return real_left(self, *args, **kwargs)

    monkeypatch.setattr(PairwiseConstructiveResult, "to_geodataframe", _count_pairwise)
    monkeypatch.setattr(LeftConstructiveResult, "to_geodataframe", _count_left)

    native_result, used_owned = overlay_module._overlay_identity_native(left, right)

    assert isinstance(native_result, ConcatConstructiveResult)
    assert pairwise_calls == 0
    assert left_calls == 0

    materialized = native_result.to_geodataframe()
    wrapped, wrapped_used = overlay_module._overlay_identity(left, right)

    assert pairwise_calls == 2
    assert left_calls == 2
    assert used_owned is wrapped_used
    assert_geodataframe_equal(materialized, wrapped, normalize=True, check_column_type=False)


def test_overlay_symmetric_difference_native_defers_fragment_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
    )

    real_left = LeftConstructiveResult.to_geodataframe
    left_calls = 0

    def _count_left(self, *args, **kwargs):
        nonlocal left_calls
        left_calls += 1
        return real_left(self, *args, **kwargs)

    monkeypatch.setattr(LeftConstructiveResult, "to_geodataframe", _count_left)

    native_result, used_owned = overlay_module._overlay_symmetric_diff_native(left, right)

    assert isinstance(native_result, SymmetricDifferenceConstructiveResult)
    assert left_calls == 0

    materialized = native_result.to_geodataframe()
    wrapped, wrapped_used = overlay_module._overlay_symmetric_diff(left, right)

    assert left_calls == 4
    assert used_owned is wrapped_used
    assert_geodataframe_equal(materialized, wrapped, normalize=True, check_column_type=False)


def test_overlay_union_native_defers_fragment_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
    )

    real_pairwise = PairwiseConstructiveResult.to_geodataframe
    real_left = LeftConstructiveResult.to_geodataframe
    pairwise_calls = 0
    left_calls = 0

    def _count_pairwise(self, *args, **kwargs):
        nonlocal pairwise_calls
        pairwise_calls += 1
        return real_pairwise(self, *args, **kwargs)

    def _count_left(self, *args, **kwargs):
        nonlocal left_calls
        left_calls += 1
        return real_left(self, *args, **kwargs)

    monkeypatch.setattr(PairwiseConstructiveResult, "to_geodataframe", _count_pairwise)
    monkeypatch.setattr(LeftConstructiveResult, "to_geodataframe", _count_left)

    native_result, used_owned = overlay_module._overlay_union_native(left, right)

    assert isinstance(native_result, ConcatConstructiveResult)
    assert pairwise_calls == 0
    assert left_calls == 0

    materialized = native_result.to_geodataframe()
    wrapped, wrapped_used = overlay_module._overlay_union(left, right)

    assert pairwise_calls == 2
    assert left_calls == 4
    assert used_owned is wrapped_used
    assert_geodataframe_equal(materialized, wrapped, normalize=True, check_column_type=False)


def test_overlay_union_native_writes_without_fragment_materialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
    )
    expected = overlay(left, right, how="union")

    def _fail(*_args, **_kwargs):
        raise AssertionError(
            "native union GeoParquet write should not require GeoDataFrame export"
        )

    monkeypatch.setattr(ConcatConstructiveResult, "to_geodataframe", _fail)
    monkeypatch.setattr(SymmetricDifferenceConstructiveResult, "to_geodataframe", _fail)
    monkeypatch.setattr(PairwiseConstructiveResult, "to_geodataframe", _fail)
    monkeypatch.setattr(LeftConstructiveResult, "to_geodataframe", _fail)
    monkeypatch.setattr(NativeTabularResult, "to_geodataframe", _fail)

    native_result, _used_owned = overlay_module._overlay_union_native(left, right)

    path = tmp_path / "overlay-union-native.parquet"
    write_geoparquet(native_result, path, geometry_encoding="geoarrow")

    result = geopandas.read_parquet(path)
    assert_geodataframe_equal(result, expected, normalize=True, check_column_type=False)


def test_overlay_union_native_builds_native_tabular_result() -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
    )

    native_result, _used_owned = overlay_module._overlay_union_native(left, right)
    tabular = to_native_tabular_result(native_result)

    assert isinstance(tabular, NativeTabularResult)
    assert_geodataframe_equal(
        tabular.to_geodataframe(),
        overlay(left, right, how="union"),
        normalize=True,
        check_column_type=False,
    )


def test_overlay_intersection_export_native_tabular_skips_pandas_attribute_assembler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
    )
    expected = overlay(left, right, how="intersection")

    real_native_pairwise = native_results_module._native_pairwise_attribute_table
    native_calls = 0

    def _counting_native_pairwise(*args, **kwargs):
        nonlocal native_calls
        native_calls += 1
        return real_native_pairwise(*args, **kwargs)

    def _fail(*_args, **_kwargs):
        raise AssertionError(
            "native overlay intersection tabular export should not require pandas attribute assembly"
        )

    monkeypatch.setattr(
        native_results_module,
        "_native_pairwise_attribute_table",
        _counting_native_pairwise,
    )
    monkeypatch.setattr(overlay_module, "_assemble_intersection_attributes", _fail)

    export_result, _used_owned = overlay_module._overlay_intersection_export_result(left, right)
    tabular = to_native_tabular_result(export_result)

    assert native_calls == 1
    assert isinstance(tabular, NativeTabularResult)
    assert isinstance(tabular.attributes, NativeAttributeTable)
    assert tabular.attributes.arrow_table is not None
    assert_geodataframe_equal(
        tabular.to_geodataframe(),
        expected,
        normalize=True,
        check_column_type=False,
    )


def test_overlay_difference_export_native_tabular_uses_arrow_backed_attributes() -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10]},
        geometry=GeoSeries([box(1, 1, 4, 4)]),
    )
    expected, _ = overlay_module._overlay_difference(left, right)

    export_result, _used_owned = overlay_module._overlay_difference_export_result(left, right)
    tabular = to_native_tabular_result(export_result)

    assert isinstance(tabular, NativeTabularResult)
    assert isinstance(tabular.attributes, NativeAttributeTable)
    assert tabular.attributes.arrow_table is not None
    assert_geodataframe_equal(
        tabular.to_geodataframe(),
        expected,
        normalize=True,
        check_column_type=False,
    )


def test_overlay_union_native_tabular_skips_pandas_attribute_assembler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
    )
    expected = overlay(left, right, how="union")

    real_native_pairwise = native_results_module._native_pairwise_attribute_table
    native_calls = 0

    def _counting_native_pairwise(*args, **kwargs):
        nonlocal native_calls
        native_calls += 1
        return real_native_pairwise(*args, **kwargs)

    def _fail(*_args, **_kwargs):
        raise AssertionError(
            "native overlay union tabular export should not require pandas attribute assembly"
        )

    monkeypatch.setattr(
        native_results_module,
        "_native_pairwise_attribute_table",
        _counting_native_pairwise,
    )
    monkeypatch.setattr(overlay_module, "_assemble_intersection_attributes", _fail)

    native_result, _used_owned = overlay_module._overlay_union_native(left, right)
    tabular = to_native_tabular_result(native_result)

    assert native_calls == 1
    assert isinstance(tabular, NativeTabularResult)
    assert isinstance(tabular.attributes, NativeAttributeTable)
    assert tabular.attributes.arrow_table is not None
    assert_geodataframe_equal(
        tabular.to_geodataframe(),
        expected,
        normalize=True,
        check_column_type=False,
    )


def test_overlay_union_native_tabular_builds_arrow_without_frame_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
    )

    native_result, _used_owned = overlay_module._overlay_union_native(left, right)
    tabular = to_native_tabular_result(native_result)

    assert isinstance(tabular, NativeTabularResult)

    def _fail(*_args, **_kwargs):
        raise AssertionError("native Arrow export should not require GeoDataFrame export")

    monkeypatch.setattr(NativeTabularResult, "to_geodataframe", _fail)

    expected = overlay(left, right, how="union")
    result = pa.table(tabular.to_arrow(geometry_encoding="WKB"))

    assert result.column_names == ["col1", "col2", "geometry"]
    assert_geodataframe_equal(
        GeoDataFrame.from_arrow(result),
        expected,
        normalize=True,
        check_column_type=False,
    )


def test_overlay_union_native_tabular_writes_feather_without_frame_materialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
    )

    native_result, _used_owned = overlay_module._overlay_union_native(left, right)
    tabular = to_native_tabular_result(native_result)

    assert isinstance(tabular, NativeTabularResult)

    def _fail(*_args, **_kwargs):
        raise AssertionError("native Feather write should not require GeoDataFrame export")

    monkeypatch.setattr(NativeTabularResult, "to_geodataframe", _fail)

    path = tmp_path / "overlay-union-native.feather"
    tabular.to_feather(path)

    result = geopandas.read_feather(path)
    assert_geodataframe_equal(
        result,
        overlay(left, right, how="union"),
        normalize=True,
        check_column_type=False,
    )


def test_overlay_intersection_export_result_writes_without_fragment_materialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
    )
    expected = overlay(left, right, how="intersection")

    def _fail(*_args, **_kwargs):
        raise AssertionError(
            "native intersection GeoParquet write should not require GeoDataFrame export"
        )

    monkeypatch.setattr(PairwiseConstructiveFragment, "to_geodataframe", _fail)
    monkeypatch.setattr(PairwiseConstructiveResult, "to_geodataframe", _fail)
    monkeypatch.setattr(NativeTabularResult, "to_geodataframe", _fail)

    export_result, _used_owned = overlay_module._overlay_intersection_export_result(left, right)

    path = tmp_path / "overlay-intersection-export.parquet"
    write_geoparquet(export_result, path, geometry_encoding="geoarrow")

    result = geopandas.read_parquet(path)
    assert_geodataframe_equal(result, expected, normalize=True, check_column_type=False)


def test_overlay_difference_export_result_writes_without_fragment_materialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10]},
        geometry=GeoSeries([box(1, 1, 4, 4)]),
    )
    expected, _ = overlay_module._overlay_difference(left, right)

    def _fail(*_args, **_kwargs):
        raise AssertionError(
            "native difference GeoParquet write should not require GeoDataFrame export"
        )

    monkeypatch.setattr(LeftConstructiveFragment, "to_geodataframe", _fail)
    monkeypatch.setattr(LeftConstructiveResult, "to_geodataframe", _fail)
    monkeypatch.setattr(NativeTabularResult, "to_geodataframe", _fail)

    export_result, _used_owned = overlay_module._overlay_difference_export_result(left, right)

    path = tmp_path / "overlay-difference-export.parquet"
    write_geoparquet(export_result, path, geometry_encoding="geoarrow")

    result = geopandas.read_parquet(path)
    assert_geodataframe_equal(result, expected, normalize=True, check_column_type=False)


def test_overlay_symmetric_difference_native_writes_without_fragment_materialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
    )
    expected = overlay(left, right, how="symmetric_difference")

    def _fail(*_args, **_kwargs):
        raise AssertionError(
            "native symmetric-difference GeoParquet write should not require GeoDataFrame export"
        )

    monkeypatch.setattr(SymmetricDifferenceConstructiveResult, "to_geodataframe", _fail)
    monkeypatch.setattr(LeftConstructiveResult, "to_geodataframe", _fail)
    monkeypatch.setattr(NativeTabularResult, "to_geodataframe", _fail)

    native_result, _used_owned = overlay_module._overlay_symmetric_diff_native(left, right)

    path = tmp_path / "overlay-symdiff-native.parquet"
    write_geoparquet(native_result, path, geometry_encoding="geoarrow")

    result = geopandas.read_parquet(path)
    assert_geodataframe_equal(result, expected, normalize=True, check_column_type=False)


def test_overlay_symmetric_difference_native_builds_native_tabular_result() -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(3, 3, 5, 5)]),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
    )

    native_result, _used_owned = overlay_module._overlay_symmetric_diff_native(left, right)
    tabular = to_native_tabular_result(native_result)

    assert isinstance(tabular, NativeTabularResult)
    assert_geodataframe_equal(
        tabular.to_geodataframe(),
        overlay(left, right, how="symmetric_difference"),
        normalize=True,
        check_column_type=False,
    )


def test_overlay_intersection_few_right_skips_non_polygon_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.constructive import binary_constructive as constructive_module

    left = GeoDataFrame(
        {"col1": np.arange(24, dtype=np.int32)},
        geometry=GeoSeries([Point(float(i), 0.0) for i in range(24)]),
    )
    right = GeoDataFrame(
        {"zone_type": ["A", "B", "C"]},
        geometry=GeoSeries(
            [
                box(-1, -1, 8, 1),
                box(7, -1, 16, 1),
                box(15, -1, 24, 1),
            ]
        ),
    )
    left_owned = left.geometry.values.to_owned()
    right_owned = right.geometry.values.to_owned()
    idx1 = np.arange(24, dtype=np.int32)
    idx2 = np.repeat(np.arange(3, dtype=np.int32), 8)

    binary_calls: list[int] = []

    monkeypatch.setattr(
        overlay_module,
        "_few_right_intersection_owned",
        lambda *args, **kwargs: pytest.fail(
            "few-right polygon shortcut must not run for non-polygon inputs"
        ),
    )

    def _fake_binary(op, left_arg, right_arg, **kwargs):
        assert op == "intersection"
        binary_calls.append(left_arg.row_count)
        return left_arg

    monkeypatch.setattr(
        constructive_module,
        "binary_constructive_owned",
        _fake_binary,
    )

    result, used_owned = overlay_module._overlay_intersection(
        left,
        right,
        left_owned=left_owned,
        right_owned=right_owned,
        _prefer_exact_polygon_gpu=True,
        _index_result=(idx1, idx2),
    )

    assert used_owned is True
    assert binary_calls == [24]
    assert result["col1"].tolist() == idx1.tolist()
    assert result["zone_type"].tolist() == ["A"] * 8 + ["B"] * 8 + ["C"] * 8


def test_overlay_intersection_few_right_prefers_exact_constructive_for_rectangle_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.constructive import binary_constructive as constructive_module

    left = GeoDataFrame(
        {"col1": np.arange(24, dtype=np.int32)},
        geometry=GeoSeries([box(i, 0, i + 1, 1) for i in range(24)]),
    )
    right = GeoDataFrame(
        {"zone_type": ["A", "B", "C"]},
        geometry=GeoSeries(
            [
                Point(8.0, 1.0).buffer(9.0),
                Point(16.0, 1.0).buffer(9.0),
                Point(24.0, 1.0).buffer(9.0),
            ]
        ),
    )
    left_owned = left.geometry.values.to_owned()
    right_owned = right.geometry.values.to_owned()
    idx1 = np.arange(24, dtype=np.int32)
    idx2 = np.repeat(np.arange(3, dtype=np.int32), 8)

    binary_calls: list[tuple[int, bool]] = []

    monkeypatch.setattr(
        constructive_module,
        "_dispatch_polygon_intersection_overlay_rowwise_gpu",
        lambda *args, **kwargs: pytest.fail(
            "rectangle-capable few-right intersection should bypass rowwise overlay"
        ),
    )

    def _fake_binary(op, left_arg, right_arg, **kwargs):
        assert op == "intersection"
        binary_calls.append(
            (
                left_arg.row_count,
                bool(kwargs.get("_prefer_exact_polygon_intersection")),
            )
        )
        geoms = [
            box(float(7000 + i), 0.0, float(7000 + i + 0.5), 0.5)
            for i in range(left_arg.row_count)
        ]
        return from_shapely_geometries(geoms, residency=Residency.DEVICE)

    monkeypatch.setattr(
        constructive_module,
        "binary_constructive_owned",
        _fake_binary,
    )

    result, used_owned = overlay_module._overlay_intersection(
        left,
        right,
        left_owned=left_owned,
        right_owned=right_owned,
        _prefer_exact_polygon_gpu=True,
        _index_result=(idx1, idx2),
    )

    assert used_owned is True
    assert binary_calls == [(24, True)]
    assert result["col1"].tolist() == idx1.tolist()
    assert result["zone_type"].tolist() == ["A"] * 8 + ["B"] * 8 + ["C"] * 8


def test_overlay_intersection_few_right_fallback_preserves_exact_polygon_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.constructive import binary_constructive as constructive_module

    left = GeoDataFrame(
        {"col1": np.arange(24, dtype=np.int32)},
        geometry=GeoSeries([box(i, 0, i + 1, 1) for i in range(24)]),
    )
    right = GeoDataFrame(
        {"zone_type": ["A", "B", "C"]},
        geometry=GeoSeries(
            [
                Polygon(
                    [(0, 0), (16, 0), (16, 2), (0, 2)],
                    holes=[[(4, 0.5), (6, 0.5), (6, 1.5), (4, 1.5)]],
                ),
                Polygon(
                    [(8, 0), (24, 0), (24, 2), (8, 2)],
                    holes=[[(12, 0.5), (14, 0.5), (14, 1.5), (12, 1.5)]],
                ),
                Polygon(
                    [(16, 0), (32, 0), (32, 2), (16, 2)],
                    holes=[[(20, 0.5), (22, 0.5), (22, 1.5), (20, 1.5)]],
                ),
            ]
        ),
    )
    left_owned = left.geometry.values.to_owned()
    right_owned = right.geometry.values.to_owned()
    idx1 = np.arange(24, dtype=np.int32)
    idx2 = np.repeat(np.arange(3, dtype=np.int32), 8)

    fallback_calls: list[tuple[int, bool]] = []

    monkeypatch.setattr(
        overlay_module,
        "_prepare_many_vs_one_intersection_chunks",
        lambda *args, **kwargs: pytest.fail(
            "few-right intersection fallback should preserve one-batch semantics"
        ),
    )
    monkeypatch.setattr(
        constructive_module,
        "_dispatch_polygon_intersection_overlay_rowwise_gpu",
        lambda *args, **kwargs: None,
    )

    def _fake_binary(op, left_arg, right_arg, **kwargs):
        assert op == "intersection"
        fallback_calls.append(
            (
                left_arg.row_count,
                bool(kwargs.get("_prefer_exact_polygon_intersection")),
            )
        )
        geoms = [
            box(float(5000 + i), 0.0, float(5000 + i + 0.5), 0.5)
            for i in range(left_arg.row_count)
        ]
        return from_shapely_geometries(geoms, residency=Residency.DEVICE)

    monkeypatch.setattr(
        constructive_module,
        "binary_constructive_owned",
        _fake_binary,
    )

    result, used_owned = overlay_module._overlay_intersection(
        left,
        right,
        left_owned=left_owned,
        right_owned=right_owned,
        _prefer_exact_polygon_gpu=True,
        _index_result=(idx1, idx2),
    )

    assert used_owned is True
    assert fallback_calls == [(24, True)]
    assert result["col1"].tolist() == idx1.tolist()
    assert result["zone_type"].tolist() == ["A"] * 8 + ["B"] * 8 + ["C"] * 8


def test_overlay_intersection_exact_mode_prefers_rectangle_kernel_for_rectangles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.constructive import binary_constructive as constructive_module

    polygon_rect_intersection_module = importlib.import_module(
        "vibespatial.kernels.constructive.polygon_rect_intersection"
    )
    polygon_intersection_module = importlib.import_module(
        "vibespatial.kernels.constructive.polygon_intersection"
    )

    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                box(0, 0, 2, 2),
                box(2, 2, 4, 4),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries(
            [
                box(1, 1, 3, 3),
                box(3, 3, 5, 5),
            ]
        ),
    )
    left_owned = left.geometry.values.to_owned()
    right_owned = right.geometry.values.to_owned()
    idx = np.arange(2, dtype=np.int32)

    kernel_calls: list[int] = []

    def _fake_polygon_rect_intersection(left_arg, right_arg, *, dispatch_mode=ExecutionMode.GPU):
        kernel_calls.append(left_arg.row_count)
        return from_shapely_geometries(
            [
                box(100.0, 0.0, 101.0, 1.0),
                box(200.0, 0.0, 201.0, 1.0),
            ],
            residency=Residency.DEVICE,
        )

    monkeypatch.setattr(
        polygon_rect_intersection_module,
        "polygon_rect_intersection",
        _fake_polygon_rect_intersection,
    )
    monkeypatch.setattr(
        polygon_intersection_module,
        "polygon_intersection",
        lambda *args, **kwargs: pytest.fail(
            "rectangle-capable intersection should use polygon_rect_intersection before polygon_intersection"
        ),
    )
    monkeypatch.setattr(
        constructive_module,
        "_dispatch_polygon_intersection_overlay_rowwise_gpu",
        lambda *args, **kwargs: pytest.fail(
            "exact rectangle-capable intersection should use polygon_rect_intersection before rowwise overlay"
        ),
    )

    result, used_owned = overlay_module._overlay_intersection(
        left,
        right,
        left_owned=left_owned,
        right_owned=right_owned,
        _prefer_exact_polygon_gpu=True,
        _index_result=(idx, idx),
    )

    assert used_owned is True
    assert kernel_calls == [2]
    assert result["col1"].tolist() == [1, 2]
    assert result["col2"].tolist() == [10, 20]
    assert [geom.bounds for geom in result.geometry] == [
        (100.0, 0.0, 101.0, 1.0),
        (200.0, 0.0, 201.0, 1.0),
    ]


def test_overlay_intersection_default_keep_geom_type_drops_touch_only_rectangle_rows() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"col1": [1]},
        geometry=GeoSeries([box(0.0, 0.0, 2.0, 2.0)]),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries([
            box(1.0, 0.0, 3.0, 2.0),
            box(2.0, 0.0, 4.0, 2.0),
        ]),
    )

    with strict_native_environment():
        with pytest.warns(UserWarning, match="`keep_geom_type=True` in overlay"):
            result = overlay(left, right, how="intersection", keep_geom_type=None)

    assert len(result) == 1
    assert result["col1"].tolist() == [1]
    assert result["col2"].tolist() == [10]
    assert result.geometry.iloc[0].equals(box(1.0, 0.0, 2.0, 2.0))


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


def test_overlay_difference_preserves_left_geometry_name() -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries([box(0, 0, 2, 2), box(2, 2, 4, 4)]),
    ).rename(columns={"geometry": "polygons"}).set_geometry("polygons")
    right = GeoDataFrame(
        {"col2": [10]},
        geometry=GeoSeries([box(1, 1, 3, 3)]),
    )

    result = overlay(left, right, how="difference")

    assert result.geometry.name == "polygons"
    assert result._geometry_column_name == "polygons"


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
