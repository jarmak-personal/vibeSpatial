from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

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
    OwnedGeometryArray,
    from_shapely_geometries,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.fallbacks import StrictNativeFallbackError
from vibespatial.runtime.residency import Residency, TransferTrigger
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


def _assert_all_geometry_coordinates_finite(series: GeoSeries) -> None:
    for row_index, geom in enumerate(np.asarray(series.array, dtype=object)):
        if geom is None:
            continue
        coords = shapely.get_coordinates(geom)
        if coords.size == 0:
            continue
        assert np.isfinite(coords).all(), f"row {row_index} contained non-finite coordinates"
        assert float(np.abs(coords).max()) < 1.0e7, (
            f"row {row_index} contained implausible coordinate magnitude"
        )


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


def test_overlay_intersection_reuses_cached_sjoin_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=DeviceGeometryArray._from_sequence(
            [
                box(0, 0, 2, 2),
                box(3, 0, 5, 2),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1, 2]},
        geometry=DeviceGeometryArray._from_sequence(
            [
                box(1, 1, 4, 3),
                box(10, 10, 12, 12),
            ]
        ),
    )

    geopandas.sjoin(left, right, predicate="intersects")
    monkeypatch.setattr(
        overlay_module,
        "_intersecting_index_pairs",
        lambda *args, **kwargs: pytest.fail("overlay should reuse cached sjoin pairs"),
    )

    result = overlay(left, right, how="intersection")

    assert len(result) == 2


def test_overlay_intersection_reuses_cached_sjoin_pairs_for_polygon_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2, 3]},
        geometry=DeviceGeometryArray._from_sequence(
            [
                box(0, 0, 2, 2),
                Point(100, 100),
                box(3, 0, 5, 2),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1, 2]},
        geometry=DeviceGeometryArray._from_sequence(
            [
                box(1, 1, 4, 3),
                box(10, 10, 12, 12),
            ]
        ),
    )

    geopandas.sjoin(left, right, predicate="intersects")
    poly_mask = left.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    left_poly = left[poly_mask]
    monkeypatch.setattr(
        overlay_module,
        "_intersecting_index_pairs",
        lambda *args, **kwargs: pytest.fail("overlay should reuse cached subset-compatible sjoin pairs"),
    )

    result = overlay(left_poly, right, how="intersection")

    assert len(result) == 2


def test_overlay_intersection_reuses_cached_pairs_when_only_nonparticipating_rows_are_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2, 3]},
        geometry=DeviceGeometryArray._from_sequence(
            [
                box(0, 0, 2, 2),
                Polygon([(20, 20), (22, 22), (20, 22), (22, 20), (20, 20)]),
                box(3, 0, 5, 2),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [1]},
        geometry=DeviceGeometryArray._from_sequence(
            [box(1, 1, 4, 3)]
        ),
    )

    geopandas.sjoin(left, right, predicate="intersects")
    monkeypatch.setattr(
        overlay_module,
        "_intersecting_index_pairs",
        lambda *args, **kwargs: pytest.fail("overlay should reuse cached pairs for valid participating rows"),
    )

    result = overlay(left, right, how="intersection")

    assert len(result) == 2


def test_overlay_intersection_single_mask_does_not_rewrite_to_clip_on_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"name": ["west", "east"]},
        geometry=GeoSeries(
            [
                box(0, 0, 2, 2),
                box(3, 0, 5, 2),
            ]
        ),
    )
    right = GeoDataFrame(
        geometry=GeoSeries([box(1, -1, 4, 3)]),
    )

    clip_module = importlib.import_module("vibespatial.api.tools.clip")
    monkeypatch.setattr(
        clip_module,
        "clip",
        lambda *args, **kwargs: pytest.fail(
            "single-mask overlay intersection should stay on the native overlay path"
        ),
    )

    result = overlay(left, right, how="intersection")

    expected = GeoDataFrame(
        {"name": ["west", "east"]},
        geometry=GeoSeries(
            [
                box(1, 0, 2, 2),
                box(3, 0, 4, 2),
            ]
        ),
    )
    assert_geodataframe_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_like=True,
    )


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


def test_overlay_intersection_single_geometry_only_mask_rewrites_to_clip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = GeoDataFrame(
        {"col1": [1, 2]},
        geometry=GeoSeries(
            [
                box(0, 0, 2, 2),
                box(2, 0, 4, 2),
            ]
        ),
    )
    right = GeoDataFrame(
        geometry=GeoSeries([box(1, -1, 3, 1)]),
    )

    expected = geopandas.clip(
        left,
        right.geometry.iloc[0],
        keep_geom_type=True,
        sort=False,
    )

    def _fail_overlay_intersection(*_args, **_kwargs):
        raise AssertionError("single-mask geometry-only intersection should rewrite to clip")

    monkeypatch.setattr(
        overlay_module,
        "_overlay_intersection",
        _fail_overlay_intersection,
    )

    result = overlay(left, right, how="intersection")

    assert_geodataframe_equal(result, expected)


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
    assert grouped_union_calls == 1
    assert grouped_plan_calls >= 1
    assert grouped_materialize_calls == 1
    assert difference_calls == 1
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
        event.implementation in {"grouped_overlay_difference_gpu", "grouped_union_difference_gpu"}
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
    assert zoned.geometry.is_valid.all()

    zoned["zone_group"] = zoned["zone_type"].astype(str)
    dissolved = zoned.dissolve(by="zone_group").reset_index()
    assert dissolved.geometry.is_valid.all()


def test_overlay_intersection_accessibility_redevelopment_fixture_matches_pairwise_oracle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from _data import setup_fixtures

    polygonal_types = ["Polygon", "MultiPolygon"]
    max_nearest_distance_m = 1_800.0
    transit_buffer_m = 900.0

    monkeypatch.setenv("VSBENCH_SCALE", "1000")
    fixtures = setup_fixtures(tmp_path)

    buildings = read_file(fixtures["access_buildings"])
    parcels = vibespatial.read_parquet(fixtures["access_parcels"])
    transit = read_file(fixtures["access_transit"])
    exclusions = vibespatial.read_parquet(fixtures["access_exclusions"])
    admin = read_file(fixtures["access_admin_boundary"])

    utm_crs = buildings.geometry.estimate_utm_crs()
    buildings = buildings.to_crs(utm_crs)
    parcels = parcels.to_crs(utm_crs)
    transit = transit.to_crs(utm_crs)
    exclusions = exclusions.to_crs(utm_crs)
    admin = admin.to_crs(utm_crs)

    buildings = geopandas.clip(buildings, admin)
    parcels = geopandas.clip(parcels, admin)
    buildings = buildings[buildings.geometry.geom_type.isin(polygonal_types)].copy()
    parcels = parcels[parcels.geometry.geom_type.isin(polygonal_types)].copy()

    building_points = buildings[["building_id", "geometry"]].copy()
    building_points["geometry"] = buildings.geometry.centroid

    nearest = building_points.sjoin_nearest(
        transit[["station_id", "geometry"]],
        how="inner",
        max_distance=max_nearest_distance_m,
        distance_col="station_distance_m",
    )
    nearest = nearest.sort_values(
        ["building_id", "station_distance_m", "station_id"]
    ).drop_duplicates("building_id")

    nearby_building_ids = nearest.loc[
        nearest["station_distance_m"] <= max_nearest_distance_m,
        "building_id",
    ].drop_duplicates()
    nearby_buildings = buildings[
        buildings["building_id"].isin(nearby_building_ids)
    ].copy()

    transit_buffers = transit.copy()
    transit_buffers["geometry"] = transit_buffers.geometry.buffer(transit_buffer_m)

    developable = overlay(parcels, exclusions, how="difference")
    developable = developable[
        developable.geometry.geom_type.isin(polygonal_types)
    ].copy()

    served = geopandas.sjoin(
        developable,
        transit_buffers[["station_id", "geometry"]],
        predicate="intersects",
    )
    served_rows = served.index.unique()
    served_parcels = (
        developable.loc[served_rows].copy()
        if len(served_rows) > 0
        else developable.iloc[:0].copy()
    )

    left = served_parcels[["parcel_id", "geometry"]]
    right = nearby_buildings[["building_id", "geometry"]]

    actual = overlay(left, right, how="intersection")
    _assert_all_geometry_coordinates_finite(actual.geometry)

    left_owned, right_owned = overlay_module._extract_owned_pair(left, right)
    index_result = overlay_module._intersecting_index_pairs(
        left,
        right,
        left_owned=left_owned,
        right_owned=right_owned,
    )
    if isinstance(index_result, np.ndarray) and index_result.ndim == 2:
        idx1, idx2 = index_result
    else:
        idx1, idx2 = index_result
    idx1 = np.asarray(idx1, dtype=np.intp)
    idx2 = np.asarray(idx2, dtype=np.intp)

    pair_left = left.iloc[idx1].reset_index(drop=True)
    pair_right = right.iloc[idx2].reset_index(drop=True)
    exact_values = np.asarray(
        shapely.intersection(
            np.asarray(pair_left.geometry.array, dtype=object),
            np.asarray(pair_right.geometry.array, dtype=object),
        ),
        dtype=object,
    )
    keep_mask = np.array(
        [
            geom is not None
            and not shapely.is_empty(geom)
            and geom.geom_type in polygonal_types
            for geom in exact_values
        ],
        dtype=bool,
    )

    expected = GeoDataFrame(
        {
            "parcel_id": pair_left.loc[keep_mask, "parcel_id"].to_numpy(),
            "building_id": pair_right.loc[keep_mask, "building_id"].to_numpy(),
            "geometry": exact_values[keep_mask],
        },
        geometry="geometry",
        crs=left.crs,
    ).sort_values(["parcel_id", "building_id"]).reset_index(drop=True)
    actual = actual.sort_values(["parcel_id", "building_id"]).reset_index(drop=True)

    assert len(actual) == 4
    assert len(actual) == len(expected)
    assert actual[["parcel_id", "building_id"]].equals(expected[["parcel_id", "building_id"]])
    for actual_geom, expected_geom in zip(actual.geometry, expected.geometry, strict=True):
        assert shapely.normalize(actual_geom).equals_exact(
            shapely.normalize(expected_geom),
            tolerance=1e-6,
        )


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


def test_binary_constructive_intersection_tiny_mixed_polygonal_batch_prefers_rowwise_overlay(
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

    rowwise_calls: list[tuple[int, int]] = []

    monkeypatch.setattr(
        constructive_module,
        "_dispatch_polygon_intersection_overlay_rowwise_gpu",
        lambda left_arg, right_arg, **kwargs: (
            rowwise_calls.append((left_arg.row_count, right_arg.row_count)),
            left_arg,
        )[1],
    )
    monkeypatch.setattr(
        constructive_module,
        "_dispatch_mixed_binary_constructive_gpu",
        lambda *args, **kwargs: pytest.fail(
            "tiny mixed polygonal intersections should use the unified rowwise "
            "overlay shortcut before the mixed tag-pair dispatcher"
        ),
    )

    result = constructive_module._binary_constructive_gpu(
        "intersection",
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )

    assert rowwise_calls == [(2, 2)]
    assert result is left


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


def test_overlay_intersection_many_vs_one_remainder_records_fallback_before_host_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left_rem = from_shapely_geometries([box(0, 0, 2, 2)], residency=Residency.HOST)
    right_one = from_shapely_geometries([box(1, 1, 3, 3)], residency=Residency.HOST)

    monkeypatch.setattr(overlay_module, "has_gpu_runtime", lambda: False)
    monkeypatch.setattr(
        overlay_module,
        "_prepare_many_vs_one_intersection_chunks",
        lambda *args, **kwargs: (
            [],
            left_rem,
            np.asarray([0], dtype=np.intp),
            right_one,
            ExecutionMode.AUTO,
        ),
    )

    original_to_shapely = OwnedGeometryArray.to_shapely

    def _wrapped_to_shapely(self, *args, **kwargs):
        events = vibespatial.get_fallback_events()
        assert len(events) == 1
        assert events[0].surface == "geopandas.overlay.intersection"
        assert "many-vs-one remainder" in events[0].reason
        return original_to_shapely(self, *args, **kwargs)

    monkeypatch.setattr(OwnedGeometryArray, "to_shapely", _wrapped_to_shapely)

    vibespatial.clear_fallback_events()
    result = overlay_module._many_vs_one_intersection_owned(left_rem, right_one, 0)

    assert result.row_count == 1
    assert result.to_shapely()[0].equals(box(0, 0, 2, 2).intersection(box(1, 1, 3, 3)))
    assert vibespatial.get_fallback_events(clear=True)[0].reason.startswith(
        "many-vs-one remainder: vectorized Shapely intersection"
    )


def test_keep_geom_type_filter_device_sources_stay_off_host_semantic_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(1, 1, 3, 3).union(box(1, 3, 5, 5)),
                ],
                residency=Residency.DEVICE,
            )
        )
    )
    right_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(3, 1, 4, 2).union(box(4, 1, 5, 4)),
                ],
                residency=Residency.DEVICE,
            )
        )
    )
    area_pairs = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(4, 3, 5, 4),
                ]
            )
        )
    )

    def _fail_take(series: GeoSeries, rows: np.ndarray) -> np.ndarray:
        raise AssertionError("device-backed keep_geom_type classification should not host-probe")

    monkeypatch.setattr(overlay_module, "_take_geoseries_object_values", _fail_take)

    vibespatial.clear_fallback_events()
    with strict_native_environment():
        filtered, dropped, keep_mask = (
            overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
                left_pairs=None,
                right_pairs=None,
                area_pairs=area_pairs,
                keep_geom_type_warning=True,
                left_source=left_source,
                right_source=right_source,
                left_rows=np.asarray([0], dtype=np.intp),
                right_rows=np.asarray([0], dtype=np.intp),
            )
        )

    assert keep_mask.tolist() == [True]
    assert dropped == 1
    assert len(filtered) == 1
    assert getattr(filtered.values, "_owned", None) is not None
    assert filtered.values._owned.residency is Residency.DEVICE
    assert vibespatial.get_fallback_events(clear=True) == []


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


def test_overlay_intersection_host_backed_polygons_with_owned_pair_stay_on_gpu_boundary() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left = GeoDataFrame(
        {"left": [0, 1]},
        geometry=GeoSeries(
            [
                box(0, 0, 2, 2),
                box(2, 2, 4, 4),
            ]
        ),
    )
    right = GeoDataFrame(
        {"right": [0, 1]},
        geometry=GeoSeries(
            [
                box(1, 1, 3, 3),
                box(3, 3, 5, 5),
            ]
        ),
    )

    left_owned, right_owned = overlay_module._extract_owned_pair(left, right)
    assert left_owned is not None
    assert right_owned is not None

    vibespatial.clear_dispatch_events()
    result = overlay(left, right, how="intersection")
    events = vibespatial.get_dispatch_events(clear=True)

    assert len(result) == 3
    assert any(
        event.surface == "geopandas.array.intersection"
        and event.selected is ExecutionMode.GPU
        for event in events
    )


def test_extract_owned_pair_promotes_host_peer_when_other_side_is_device_backed() -> None:
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
        geometry=GeoSeries(
            [
                box(1, 1, 3, 3),
                box(3, 3, 5, 5),
            ]
        ),
    )

    vibespatial.clear_dispatch_events()
    left_owned, right_owned = overlay_module._extract_owned_pair(left, right)
    events = vibespatial.get_dispatch_events(clear=True)

    assert left_owned is not None
    assert right_owned is not None
    assert left_owned.residency is Residency.DEVICE
    assert right_owned.residency is Residency.DEVICE
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


def test_keep_geom_type_filter_skips_device_warning_refinement_when_exact_values_cover_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left_pairs = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
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
                ]
            )
        )
    )
    area_pairs = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 1, 1),
                    None,
                ]
            )
        )
    )
    area_pairs.values._exact_intersection_values = np.asarray(
        [
            GeometryCollection(
                [
                    box(0, 0, 1, 1),
                    LineString([(0, 0), (1, 1)]),
                ]
            ),
            LineString([(1, 0), (1, 1)]),
        ],
        dtype=object,
    )
    area_pairs.values._exact_intersection_value_mask = np.ones(2, dtype=bool)

    monkeypatch.setattr(
        overlay_module,
        "_clear_device_exact_keep_geom_type_warnings",
        lambda *args, **kwargs: pytest.fail(
            "cached exact intersection values should bypass device warning refinement"
        ),
    )

    filtered, dropped, keep_mask = overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
        left_pairs,
        right_pairs,
        area_pairs,
        keep_geom_type_warning=True,
    )

    assert keep_mask.tolist() == [True, False]
    assert dropped == 2
    assert len(filtered) == 1
    assert filtered.iloc[0].equals(box(0, 0, 1, 1))


def test_overlay_intersection_small_exact_boundary_reuses_existing_owned_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = GeoDataFrame({"geometry": [box(0, 0, 2, 2)]}, geometry="geometry")
    right = GeoDataFrame({"geometry": [box(1, 1, 3, 3)]}, geometry="geometry")
    left_owned = left.geometry.values.to_owned()
    right_owned = right.geometry.values.to_owned()

    def _fail(self):
        pytest.fail("small exact boundary path should reuse caller-provided owned pairs")

    monkeypatch.setattr(GeometryArray, "to_owned", _fail)

    result, used_owned = overlay_module._overlay_intersection(
        left,
        right,
        left_owned,
        right_owned,
        _warn_on_dropped_lower_dim_polygon_results=True,
    )

    assert used_owned is True
    assert_geodataframe_equal(
        result.reset_index(drop=True),
        GeoDataFrame({"geometry": [box(1, 1, 2, 2)]}, geometry="geometry"),
    )


def test_keep_geom_type_filter_warning_can_use_source_rows_without_pair_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left_source = GeoSeries(
        [
            box(0, 0, 1, 1),
            box(0, 0, 1, 1),
            box(0, 0, 1, 1),
        ]
    )
    right_source = GeoSeries(
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

    real_take = overlay_module._take_geoseries_object_values

    def _take_only_sources(series: GeoSeries, rows: np.ndarray) -> np.ndarray:
        if series is left_source or series is right_source or series is area_pairs:
            return real_take(series, rows)
        raise AssertionError("pair-series materialization should stay cold")

    monkeypatch.setattr(overlay_module, "_take_geoseries_object_values", _take_only_sources)

    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            None,
            None,
            area_pairs,
            keep_geom_type_warning=True,
            left_source=left_source,
            right_source=right_source,
            left_rows=np.arange(3, dtype=np.intp),
            right_rows=np.arange(3, dtype=np.intp),
        )
    )

    assert keep_mask.tolist() == [True, False, False]
    assert dropped == 1
    assert len(filtered) == 1


def test_keep_geom_type_filter_skips_all_row_take_when_every_polygon_row_survives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left_source = GeoSeries(
        [
            box(0, 0, 1, 1),
            box(2, 0, 3, 1),
        ]
    )
    right_source = GeoSeries(
        [
            box(0, 0, 1, 1),
            box(2, 0, 3, 1),
        ]
    )
    area_pairs = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 1, 1),
                    box(2, 0, 3, 1),
                ],
                residency=Residency.DEVICE,
            )
        )
    )

    monkeypatch.setattr(
        GeoSeries,
        "take",
        lambda *args, **kwargs: pytest.fail("all-surviving keep-geom-type filter should not take every row"),
    )

    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            None,
            None,
            area_pairs,
            keep_geom_type_warning=True,
            left_source=left_source,
            right_source=right_source,
            left_rows=np.arange(2, dtype=np.intp),
            right_rows=np.arange(2, dtype=np.intp),
        )
    )

    assert keep_mask.tolist() == [True, True]
    assert dropped == 0
    assert len(filtered) == 2


def test_keep_geom_type_filter_skips_warning_count_when_kept_rows_have_no_boundary_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left_source = GeoSeries(
        [
            box(0, 0, 5, 5),
            box(10, 10, 16, 16),
        ]
    )
    right_source = GeoSeries(
        [
            box(1, 1, 2, 2),
            box(12, 12, 13, 13),
        ]
    )
    area_pairs = GeoSeries(
        [
            box(1, 1, 2, 2),
            box(12, 12, 13, 13),
        ]
    )

    monkeypatch.setattr(
        overlay_module,
        "_count_dropped_polygon_intersection_parts",
        lambda *_args, **_kwargs: pytest.fail(
            "nested polygon rows with no boundary overlap should not trigger dropped-count exact intersection"
        ),
    )

    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            None,
            None,
            area_pairs,
            keep_geom_type_warning=True,
            left_source=left_source,
            right_source=right_source,
            left_rows=np.arange(2, dtype=np.intp),
            right_rows=np.arange(2, dtype=np.intp),
        )
    )

    assert keep_mask.tolist() == [True, True]
    assert dropped == 0
    assert len(filtered) == 2


def test_keep_geom_type_filter_uses_rect_kernel_overlap_flag_to_skip_warning_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    area_pairs = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 1, 1),
                    box(2, 2, 3, 3),
                ],
                residency=Residency.DEVICE,
            )
        )
    )
    area_pairs.values._owned._polygon_rect_boundary_overlap = np.zeros(2, dtype=bool)

    monkeypatch.setattr(
        overlay_module,
        "_count_dropped_polygon_intersection_parts",
        lambda *_args, **_kwargs: pytest.fail(
            "rectangle overlap flag should let keep-geom-type warning skip host dropped-count rebuild"
        ),
    )
    monkeypatch.setattr(
        overlay_module,
        "_warning_candidate_mask_for_polygon_keep_geom_type",
        lambda *_args, **_kwargs: pytest.fail(
            "rectangle overlap flag should bypass generic warning candidate classification"
        ),
    )

    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            None,
            None,
            area_pairs,
            keep_geom_type_warning=True,
            left_source=GeoSeries([box(-1, -1, 2, 2), box(1, 1, 4, 4)]),
            right_source=GeoSeries([box(0, 0, 1, 1), box(2, 2, 3, 3)]),
            left_rows=np.arange(2, dtype=np.intp),
            right_rows=np.arange(2, dtype=np.intp),
        )
    )

    assert keep_mask.tolist() == [True, True]
    assert dropped == 0
    assert len(filtered) == 2


def test_keep_geom_type_filter_rect_overlap_mask_only_materializes_warning_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    owned = from_shapely_geometries(
        [
            box(0, 0, 1, 1),
            box(2, 2, 3, 3),
            box(4, 4, 5, 5),
        ],
        residency=Residency.DEVICE,
    )
    owned._polygon_rect_boundary_overlap = np.asarray([False, True, False], dtype=bool)
    area_pairs = GeoSeries(GeometryArray.from_owned(owned))
    left_source = GeoSeries(
        [
            box(0, 0, 4, 4),
            box(2, 2, 6, 6),
            box(4, 4, 8, 8),
        ]
    )
    right_source = GeoSeries(
        [
            box(0, 0, 1, 1),
            box(2, 2, 3, 3),
            box(4, 4, 5, 5),
        ]
    )

    observed_rows: list[tuple[int, ...]] = []

    def _take_only_warning_row(series: GeoSeries, rows: np.ndarray) -> np.ndarray:
        observed_rows.append(tuple(int(v) for v in rows))
        return np.asarray(series, dtype=object)[rows]

    monkeypatch.setattr(overlay_module, "_take_geoseries_object_values", _take_only_warning_row)
    monkeypatch.setattr(
        overlay_module,
        "_count_dropped_polygon_intersection_parts",
        lambda left_values, right_values, row_count, **_kwargs: row_count,
    )

    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            None,
            None,
            area_pairs,
            keep_geom_type_warning=True,
            left_source=left_source,
            right_source=right_source,
            left_rows=np.arange(3, dtype=np.intp),
            right_rows=np.arange(3, dtype=np.intp),
        )
    )

    assert keep_mask.tolist() == [True, True, True]
    assert dropped == 1
    assert len(filtered) == 3
    assert observed_rows == [(1,), (1,)]
    assert np.asarray(
        getattr(filtered.values._owned, "_polygon_rect_boundary_overlap", None),
        dtype=bool,
    ).tolist() == [False, True, False]


def test_keep_geom_type_filter_rect_overlap_device_sources_stay_off_host_probe() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 4, 4),
                    box(10, 0, 14, 4),
                ],
                residency=Residency.DEVICE,
            )
        )
    )
    right_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(2, 0, 6, 4),
                    box(12, 0, 16, 4),
                ],
                residency=Residency.DEVICE,
            )
        )
    )
    area_owned = from_shapely_geometries(
        [
            box(2, 0, 4, 4),
            box(12, 0, 14, 4),
        ],
        residency=Residency.DEVICE,
    )
    area_owned._polygon_rect_boundary_overlap = np.asarray([True, True], dtype=bool)
    area_pairs = GeoSeries(GeometryArray.from_owned(area_owned))

    vibespatial.clear_fallback_events()
    filtered, dropped, keep_mask = overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
        None,
        None,
        area_pairs,
        keep_geom_type_warning=True,
        left_source=left_source,
        right_source=right_source,
        left_rows=np.arange(2, dtype=np.intp),
        right_rows=np.arange(2, dtype=np.intp),
    )

    assert keep_mask.tolist() == [True, True]
    assert dropped == 0
    assert len(filtered) == 2
    assert vibespatial.get_fallback_events(clear=True) == []


def test_keep_geom_type_filter_rect_overlap_missing_polygon_empty_mask_stays_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 10, 4),
                    box(0, 0, 10, 4),
                ],
                residency=Residency.DEVICE,
            )
        )
    )
    right_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(2, 0, 4, 4),
                    box(6, 0, 8, 4),
                ],
                residency=Residency.DEVICE,
            )
        )
    )
    area_owned = from_shapely_geometries(
        [
            box(2, 0, 4, 4),
            box(6, 0, 8, 4),
        ],
        residency=Residency.DEVICE,
    )
    area_owned._polygon_rect_boundary_overlap = np.asarray([True, True], dtype=bool)
    object.__setattr__(
        area_owned.families[GeometryFamily.POLYGON],
        "empty_mask",
        np.empty(0, dtype=bool),
    )
    area_pairs = GeoSeries(GeometryArray.from_owned(area_owned))

    monkeypatch.setattr(
        overlay_module,
        "_take_geoseries_object_values",
        lambda *_args, **_kwargs: pytest.fail(
            "rect-overlap keep-geom-type should not materialize host values when polygon empty_mask metadata is absent"
        ),
    )

    vibespatial.clear_fallback_events()
    filtered, dropped, keep_mask = overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
        None,
        None,
        area_pairs,
        keep_geom_type_warning=True,
        left_source=left_source,
        right_source=right_source,
        left_rows=np.arange(2, dtype=np.intp),
        right_rows=np.arange(2, dtype=np.intp),
    )

    assert keep_mask.tolist() == [True, True]
    assert dropped == 0
    assert len(filtered) == 2
    assert vibespatial.get_fallback_events(clear=True) == []


def test_keep_geom_type_filter_rect_overlap_device_sources_fall_back_to_conservative_native_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 4, 4),
                    box(10, 0, 14, 4),
                ],
                residency=Residency.DEVICE,
            )
        )
    )
    right_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(2, 0, 6, 4),
                    box(12, 0, 16, 4),
                ],
                residency=Residency.DEVICE,
            )
        )
    )
    area_owned = from_shapely_geometries(
        [
            box(2, 0, 4, 4),
            box(12, 0, 14, 4),
        ],
        residency=Residency.DEVICE,
    )
    area_owned._polygon_rect_boundary_overlap = np.asarray([True, True], dtype=bool)
    area_pairs = GeoSeries(GeometryArray.from_owned(area_owned))

    monkeypatch.setattr(
        overlay_module,
        "_device_count_dropped_polygon_intersection_warning_rows",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        overlay_module,
        "_take_geoseries_object_values",
        lambda *_args, **_kwargs: pytest.fail(
            "device-backed rect-overlap warning counting should stay native when refinement fails"
        ),
    )

    vibespatial.clear_fallback_events()
    filtered, dropped, keep_mask = overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
        None,
        None,
        area_pairs,
        keep_geom_type_warning=True,
        left_source=left_source,
        right_source=right_source,
        left_rows=np.arange(2, dtype=np.intp),
        right_rows=np.arange(2, dtype=np.intp),
    )

    assert keep_mask.tolist() == [True, True]
    assert dropped == 2
    assert len(filtered) == 2
    assert vibespatial.get_fallback_events(clear=True) == []


def test_keep_geom_type_filter_rect_overlap_host_sources_use_device_pairs_before_host_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left_geoms = [
        box(0, 0, 10, 4),
        box(0, 0, 10, 4),
    ]
    right_geoms = [
        box(2, 0, 4, 4),
        box(6, 0, 8, 4),
    ]
    left_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(left_geoms, residency=Residency.HOST)
        )
    )
    right_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(right_geoms, residency=Residency.HOST)
        )
    )
    left_pairs = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(left_geoms, residency=Residency.DEVICE)
        )
    )
    right_pairs = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(right_geoms, residency=Residency.DEVICE)
        )
    )
    area_owned = from_shapely_geometries(
        [
            box(2, 0, 4, 4),
            box(6, 0, 8, 4),
        ],
        residency=Residency.DEVICE,
    )
    area_owned._polygon_rect_boundary_overlap = np.asarray([True, True], dtype=bool)
    area_pairs = GeoSeries(GeometryArray.from_owned(area_owned))

    monkeypatch.setattr(
        overlay_module,
        "_take_geoseries_object_values",
        lambda *_args, **_kwargs: pytest.fail(
            "host-backed sources should not materialize when device-backed pair rows are available"
        ),
    )

    vibespatial.clear_fallback_events()
    filtered, dropped, keep_mask = overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
        left_pairs,
        right_pairs,
        area_pairs,
        keep_geom_type_warning=True,
        left_source=left_source,
        right_source=right_source,
        left_rows=np.arange(2, dtype=np.intp),
        right_rows=np.arange(2, dtype=np.intp),
    )

    assert keep_mask.tolist() == [True, True]
    assert dropped == 0
    assert len(filtered) == 2
    assert vibespatial.get_fallback_events(clear=True) == []


def test_keep_geom_type_filter_kept_rows_with_shared_boundary_on_area_boundary_stays_native() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    left_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 10, 4),
                    box(0, 0, 10, 4),
                ],
                residency=Residency.DEVICE,
            )
        )
    )
    right_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(2, 0, 4, 4),
                    box(6, 0, 8, 4),
                ],
                residency=Residency.DEVICE,
            )
        )
    )
    area_pairs = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(2, 0, 4, 4),
                    box(6, 0, 8, 4),
                ],
                residency=Residency.DEVICE,
            )
        )
    )

    vibespatial.clear_fallback_events()
    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            None,
            None,
            area_pairs,
            keep_geom_type_warning=True,
            left_source=left_source,
            right_source=right_source,
            left_rows=np.arange(2, dtype=np.intp),
            right_rows=np.arange(2, dtype=np.intp),
        )
    )

    assert keep_mask.tolist() == [True, True]
    assert dropped == 0
    assert len(filtered) == 2
    assert vibespatial.get_fallback_events(clear=True) == []


def test_device_count_dropped_polygon_intersection_warning_rows_handles_large_distinct_pair_batches() -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    dropped_pairs = 40
    kept_pairs = 40

    left_geoms: list[object] = []
    right_geoms: list[object] = []
    area_geoms: list[object] = []
    keep_mask = np.zeros(dropped_pairs + kept_pairs, dtype=bool)

    for row in range(dropped_pairs):
        x0 = float(row * 10)
        left_geoms.append(box(x0, 0.0, x0 + 1.0, 1.0))
        right_geoms.append(box(x0 + 1.0, 0.0, x0 + 2.0, 1.0))
        area_geoms.append(None)

    for local_row in range(kept_pairs):
        row = dropped_pairs + local_row
        x0 = float(row * 10)
        left_geoms.append(box(x0, 0.0, x0 + 2.0, 2.0))
        right_geoms.append(box(x0 + 1.0, 1.0, x0 + 3.0, 3.0))
        area_geoms.append(box(x0 + 1.0, 1.0, x0 + 2.0, 2.0))
        keep_mask[row] = True

    left_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(left_geoms, residency=Residency.DEVICE)
        )
    )
    right_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(right_geoms, residency=Residency.DEVICE)
        )
    )
    area_owned = from_shapely_geometries(area_geoms, residency=Residency.DEVICE)

    dropped = overlay_module._device_count_dropped_polygon_intersection_warning_rows(
        area_owned,
        keep_mask,
        np.arange(dropped_pairs + kept_pairs, dtype=np.intp),
        left_source=left_source,
        right_source=right_source,
        left_rows=np.arange(dropped_pairs + kept_pairs, dtype=np.intp),
        right_rows=np.arange(dropped_pairs + kept_pairs, dtype=np.intp),
    )

    assert dropped == dropped_pairs


def test_device_count_dropped_polygon_intersection_warning_rows_skips_large_warning_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    constructive_module = importlib.import_module("vibespatial.constructive.binary_constructive")

    rows = 129
    left_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [box(float(i), 0.0, float(i + 1), 1.0) for i in range(rows)],
                residency=Residency.DEVICE,
            )
        )
    )
    right_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [box(float(i + 1), 0.0, float(i + 2), 1.0) for i in range(rows)],
                residency=Residency.DEVICE,
            )
        )
    )
    area_owned = from_shapely_geometries([None] * rows, residency=Residency.DEVICE)

    monkeypatch.setattr(
        constructive_module,
        "binary_constructive_owned",
        lambda *_args, **_kwargs: pytest.fail(
            "large advisory warning batches should bypass exact boundary constructive refinement"
        ),
    )

    dropped = overlay_module._device_count_dropped_polygon_intersection_warning_rows(
        area_owned,
        np.zeros(rows, dtype=bool),
        np.arange(rows, dtype=np.intp),
        left_source=left_source,
        right_source=right_source,
        left_rows=np.arange(rows, dtype=np.intp),
        right_rows=np.arange(rows, dtype=np.intp),
    )

    assert dropped is None


def test_clear_device_exact_keep_geom_type_warnings_only_checks_kept_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_rows: list[np.ndarray] = []

    def _fake_cover_mask(
        left_source,
        right_source,
        left_rows,
        right_rows,
        warning_rows: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        observed_rows.append(np.asarray(warning_rows, dtype=np.intp))
        return np.asarray([True, False], dtype=bool)

    monkeypatch.setattr(
        overlay_module,
        "_device_polygon_keep_geom_type_cover_mask",
        _fake_cover_mask,
    )

    warning_mask = np.asarray([True, True, True], dtype=bool)
    keep_mask = np.asarray([False, True, True], dtype=bool)

    updated_mask, warning_rows = overlay_module._clear_device_exact_keep_geom_type_warnings(
        warning_mask,
        keep_mask,
        left_source=None,
        right_source=None,
        left_rows=np.arange(3, dtype=np.intp),
        right_rows=np.zeros(3, dtype=np.intp),
    )

    assert [rows.tolist() for rows in observed_rows] == [[1, 2]]
    assert updated_mask.tolist() == [True, False, True]
    assert warning_rows.tolist() == [0, 2]


def test_device_polygon_keep_geom_type_cover_mask_uses_broadcast_right(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    predicate_module = importlib.import_module("vibespatial.predicates.binary")
    observed_calls: list[tuple[str, int, int]] = []

    def _fake_evaluate_binary_predicate(predicate, left, right, **kwargs):
        observed_calls.append((predicate, left.row_count, right.row_count))
        assert kwargs["dispatch_mode"] is ExecutionMode.GPU
        return SimpleNamespace(values=np.ones(left.row_count, dtype=bool))

    monkeypatch.setattr(
        predicate_module,
        "evaluate_binary_predicate",
        _fake_evaluate_binary_predicate,
    )

    left_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [
                    box(0, 0, 1, 1),
                    box(2, 0, 3, 1),
                    box(4, 0, 5, 1),
                ],
                residency=Residency.DEVICE,
            )
        )
    )
    right_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [box(-1, -1, 10, 10)],
                residency=Residency.DEVICE,
            )
        )
    )

    cover_mask = overlay_module._device_polygon_keep_geom_type_cover_mask(
        left_source,
        right_source,
        np.arange(3, dtype=np.intp),
        np.zeros(3, dtype=np.intp),
        np.arange(3, dtype=np.intp),
    )

    assert np.asarray(cover_mask, dtype=bool).tolist() == [True, True, True]
    assert observed_calls == [
        ("covered_by", 3, 1),
    ]


def test_keep_geom_type_filter_many_vs_one_metadata_skips_covered_by_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    monkeypatch.setattr(
        overlay_module,
        "_device_polygon_keep_geom_type_cover_mask",
        lambda *_args, **_kwargs: pytest.fail(
            "many-vs-one containment metadata should skip the advisory cover probe"
        ),
    )
    monkeypatch.setattr(
        overlay_module,
        "_device_count_dropped_polygon_intersection_warning_rows",
        lambda *_args, **_kwargs: 0,
    )

    left_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [box(0, 0, 4, 4)],
                residency=Residency.DEVICE,
            )
        )
    )
    right_source = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [box(2, 0, 8, 4)],
                residency=Residency.DEVICE,
            )
        )
    )
    area_owned = from_shapely_geometries(
        [box(2, 0, 4, 4)],
        residency=Residency.DEVICE,
    )
    area_owned._polygon_rect_boundary_overlap = np.asarray([True], dtype=bool)
    area_owned._many_vs_one_left_containment_bypass_applied = True
    area_pairs = GeoSeries(GeometryArray.from_owned(area_owned))

    filtered, dropped, keep_mask = overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
        None,
        None,
        area_pairs,
        keep_geom_type_warning=True,
        left_source=left_source,
        right_source=right_source,
        left_rows=np.asarray([0], dtype=np.intp),
        right_rows=np.asarray([0], dtype=np.intp),
    )

    assert keep_mask.tolist() == [True]
    assert dropped == 0
    assert len(filtered) == 1


def test_assemble_indexed_owned_chunks_preserves_exact_intersection_cache() -> None:
    first = from_shapely_geometries(
        [box(2, 0, 3, 1)],
        residency=Residency.HOST,
    )
    first._exact_intersection_values = np.asarray(
        [GeometryCollection([box(2, 0, 3, 1), LineString([(2, 0), (3, 0)])])],
        dtype=object,
    )
    first._exact_intersection_value_mask = np.asarray([True], dtype=bool)

    second = from_shapely_geometries(
        [box(0, 0, 1, 1)],
        residency=Residency.HOST,
    )

    assembled = overlay_module._assemble_indexed_owned_chunks(
        [
            (np.asarray([1], dtype=np.intp), first),
            (np.asarray([0], dtype=np.intp), second),
        ],
        2,
    )

    assert np.asarray(
        getattr(assembled, "_exact_intersection_value_mask", None),
        dtype=bool,
    ).tolist() == [False, True]
    exact_values = getattr(assembled, "_exact_intersection_values", None)
    assert exact_values is not None
    assert exact_values[1].geom_type == "GeometryCollection"


def test_keep_geom_type_filter_reuses_cached_exact_intersection_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned = from_shapely_geometries(
        [
            box(0, 0, 1, 1),
            box(2, 0, 3, 1),
        ],
        residency=Residency.HOST,
    )
    owned._exact_intersection_values = np.asarray(
        [
            GeometryCollection([box(0, 0, 1, 1), LineString([(0, 0), (1, 0)])]),
            GeometryCollection([box(2, 0, 3, 1), Point(2, 0)]),
        ],
        dtype=object,
    )
    owned._exact_intersection_value_mask = np.asarray([True, True], dtype=bool)
    area_pairs = GeoSeries(GeometryArray.from_owned(owned))

    monkeypatch.setattr(
        overlay_module,
        "_warning_candidate_mask_for_polygon_keep_geom_type",
        lambda *_args, **_kwargs: np.asarray([True, True], dtype=bool),
    )
    monkeypatch.setattr(
        shapely,
        "intersection",
        lambda *_args, **_kwargs: pytest.fail(
            "cached exact intersection values should avoid a second shapely.intersection rebuild"
        ),
    )

    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            GeoSeries([box(-1, -1, 2, 2), box(1, -1, 4, 2)]),
            GeoSeries([box(0, 0, 1, 1), box(2, 0, 3, 1)]),
            area_pairs,
            keep_geom_type_warning=True,
        )
    )

    assert keep_mask.tolist() == [True, True]
    assert dropped == 2
    assert len(filtered) == 2


def test_keep_geom_type_filter_uses_cached_exact_values_for_warning_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned = from_shapely_geometries(
        [
            box(0, 0, 1, 1),
            box(2, 0, 3, 1),
        ],
        residency=Residency.HOST,
    )
    owned._exact_intersection_values = np.asarray(
        [
            GeometryCollection([box(0, 0, 1, 1), LineString([(0, 0), (1, 0)])]),
            box(2, 0, 3, 1),
        ],
        dtype=object,
    )
    owned._exact_intersection_value_mask = np.asarray([True, True], dtype=bool)
    area_pairs = GeoSeries(GeometryArray.from_owned(owned))

    monkeypatch.setattr(
        overlay_module,
        "_take_geoseries_object_values",
        lambda *_args, **_kwargs: pytest.fail(
            "fully cached exact intersections should avoid left/right object materialization"
        ),
    )
    monkeypatch.setattr(
        overlay_module,
        "_warning_candidate_mask_for_polygon_keep_geom_type",
        lambda *_args, **_kwargs: pytest.fail(
            "fully cached exact intersections should avoid boundary-overlap warning probing"
        ),
    )

    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            None,
            None,
            area_pairs,
            keep_geom_type_warning=True,
        )
    )

    assert keep_mask.tolist() == [True, True]
    assert dropped == 1
    assert len(filtered) == 2


def test_keep_geom_type_filter_rect_overlap_cached_exact_values_skip_source_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned = from_shapely_geometries(
        [
            box(0, 0, 1, 1),
            box(2, 0, 3, 1),
        ],
        residency=Residency.HOST,
    )
    owned._polygon_rect_boundary_overlap = np.asarray([True, True], dtype=bool)
    owned._exact_intersection_values = np.asarray(
        [
            GeometryCollection([box(0, 0, 1, 1), LineString([(0, 0), (1, 0)])]),
            box(2, 0, 3, 1),
        ],
        dtype=object,
    )
    owned._exact_intersection_value_mask = np.asarray([True, True], dtype=bool)
    area_pairs = GeoSeries(GeometryArray.from_owned(owned))
    left_source = GeoSeries([box(-1, -1, 2, 2), box(1, -1, 4, 2)])
    right_source = GeoSeries([box(0, 0, 1, 1), box(2, 0, 3, 1)])

    monkeypatch.setattr(
        overlay_module,
        "_take_geoseries_object_values",
        lambda *_args, **_kwargs: pytest.fail(
            "fully cached exact warning rows should avoid source object materialization"
        ),
    )

    filtered, dropped, keep_mask = (
        overlay_module._filter_polygon_intersection_rows_for_keep_geom_type(
            None,
            None,
            area_pairs,
            keep_geom_type_warning=True,
            left_source=left_source,
            right_source=right_source,
            left_rows=np.arange(2, dtype=np.intp),
            right_rows=np.arange(2, dtype=np.intp),
        )
    )

    assert keep_mask.tolist() == [True, True]
    assert dropped == 1
    assert len(filtered) == 2


def test_repair_invalid_polygon_output_rows_repairs_rect_boundary_spikes() -> None:
    invalid = Polygon(
        [
            (680.0, 390.0),
            (680.0, 400.0),
            (680.0, 400.0),
            (680.0, 391.929775259655),
            (679.7726140184767, 390.0),
            (680.0, 390.0),
        ]
    )
    valid = box(0.0, 0.0, 1.0, 1.0)
    owned = from_shapely_geometries(
        [invalid, valid],
        residency=Residency.DEVICE if vibespatial.has_gpu_runtime() else Residency.HOST,
    )
    owned._polygon_rect_boundary_overlap = np.asarray([True, False], dtype=bool)
    geometries = GeoSeries(GeometryArray.from_owned(owned))

    repaired = overlay_module._repair_invalid_polygon_output_rows(geometries)

    assert bool(shapely.is_valid(repaired.iloc[0]))
    assert shapely.equals(repaired.iloc[0], shapely.make_valid(invalid))
    assert shapely.equals(repaired.iloc[1], valid)
    assert np.asarray(
        getattr(repaired.values, "_polygon_rect_boundary_overlap", None),
        dtype=bool,
    ).tolist() == [True, False]


def test_repair_invalid_polygon_output_rows_small_batch_falls_back_without_overlap_mask() -> None:
    invalid = Polygon(
        [
            (680.0, 390.0),
            (680.0, 400.0),
            (680.0, 400.0),
            (680.0, 391.929775259655),
            (679.7726140184767, 390.0),
            (680.0, 390.0),
        ]
    )
    geometries = GeoSeries([invalid])

    repaired = overlay_module._repair_invalid_polygon_output_rows(geometries)

    assert bool(shapely.is_valid(repaired.iloc[0]))
    assert shapely.equals(repaired.iloc[0], shapely.make_valid(invalid))


def test_repair_invalid_polygon_output_rows_small_batch_falls_back_when_overlap_mask_is_empty() -> None:
    invalid = Polygon(
        [
            (680.0, 390.0),
            (680.0, 400.0),
            (680.0, 400.0),
            (680.0, 391.929775259655),
            (679.7726140184767, 390.0),
            (680.0, 390.0),
        ]
    )
    owned = from_shapely_geometries(
        [invalid],
        residency=Residency.DEVICE if vibespatial.has_gpu_runtime() else Residency.HOST,
    )
    owned._polygon_rect_boundary_overlap = np.zeros(1, dtype=bool)
    geometries = GeoSeries(GeometryArray.from_owned(owned))

    repaired = overlay_module._repair_invalid_polygon_output_rows(geometries)

    assert bool(shapely.is_valid(repaired.iloc[0]))
    assert shapely.equals(repaired.iloc[0], shapely.make_valid(invalid))


def test_repair_invalid_polygon_output_rows_owned_valid_batch_skips_host_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned = from_shapely_geometries(
        [box(0.0, 0.0, 1.0, 1.0), box(2.0, 2.0, 3.0, 3.0)],
        residency=Residency.DEVICE if vibespatial.has_gpu_runtime() else Residency.HOST,
    )
    geometries = GeoSeries(GeometryArray.from_owned(owned))

    import vibespatial.geometry.host_bridge as host_bridge_module

    def _fail(*_args, **_kwargs):
        raise AssertionError("valid owned rows should not materialize through host_bridge")

    monkeypatch.setattr(host_bridge_module, "owned_to_shapely", _fail)

    repaired = overlay_module._repair_invalid_polygon_output_rows(geometries)

    assert repaired is geometries


def test_overlay_make_valid_owned_rewrap_failure_records_fallback_before_host_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vibespatial.constructive import make_valid_pipeline as make_valid_pipeline_module

    left = GeoDataFrame(
        {"left": [0]},
        geometry=GeoSeries(
            GeometryArray.from_owned(
                from_shapely_geometries(
                    [Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])],
                    residency=Residency.HOST,
                )
            )
        ),
    )
    right = GeoDataFrame(
        {"right": [0]},
        geometry=GeoSeries(
            GeometryArray.from_owned(
                from_shapely_geometries(
                    [box(-1, -1, 3, 3)],
                    residency=Residency.HOST,
                )
            )
        ),
    )

    fallback_owned = from_shapely_geometries(
        [box(0, 0, 1, 1)],
        residency=Residency.HOST,
    )
    fallback_result = make_valid_pipeline_module.MakeValidResult(
        row_count=1,
        valid_rows=np.asarray([False], dtype=bool),
        repaired_rows=np.asarray([0], dtype=np.intp),
        null_rows=np.asarray([False], dtype=bool),
        method="test",
        keep_collapsed=True,
        owned=fallback_owned,
        selected=ExecutionMode.CPU,
    )

    original_make_valid_owned = make_valid_pipeline_module.make_valid_owned
    make_valid_calls = 0

    def _wrapped_make_valid_owned(*args, **kwargs):
        nonlocal make_valid_calls
        make_valid_calls += 1
        if make_valid_calls == 1:
            return fallback_result
        return original_make_valid_owned(*args, **kwargs)

    monkeypatch.setattr(
        make_valid_pipeline_module,
        "make_valid_owned",
        _wrapped_make_valid_owned,
    )

    original_from_owned = GeometryArray.from_owned
    from_owned_calls = 0

    def _wrapped_from_owned(owned, *args, **kwargs):
        nonlocal from_owned_calls
        if owned is fallback_owned:
            from_owned_calls += 1
            raise NotImplementedError("test rewrap failure")
        return original_from_owned(owned, *args, **kwargs)

    monkeypatch.setattr(GeometryArray, "from_owned", _wrapped_from_owned)

    original_to_shapely = fallback_owned.to_shapely
    materialized = False

    def _wrapped_to_shapely(*args, **kwargs):
        nonlocal materialized
        materialized = True
        events = vibespatial.get_fallback_events()
        assert len(events) == 1
        assert events[0].surface == "geopandas.array.make_valid"
        assert "host materialization required" in events[0].reason
        return original_to_shapely(*args, **kwargs)

    monkeypatch.setattr(fallback_owned, "to_shapely", _wrapped_to_shapely)

    vibespatial.clear_fallback_events()
    with pytest.raises(StrictNativeFallbackError):
        with strict_native_environment():
            overlay(left, right, how="intersection")

    assert from_owned_calls == 1
    assert materialized is False
    events = vibespatial.get_fallback_events(clear=True)
    assert len(events) == 1
    assert events[0].surface == "geopandas.array.make_valid"
    assert events[0].selected is ExecutionMode.CPU


def test_make_valid_geoseries_uses_seeded_validity_cache_without_recompute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validity_module = importlib.import_module("vibespatial.constructive.validity")

    owned = from_shapely_geometries(
        [box(0.0, 0.0, 1.0, 1.0), box(2.0, 2.0, 3.0, 3.0)],
        residency=Residency.DEVICE if vibespatial.has_gpu_runtime() else Residency.HOST,
    )
    owned._cached_is_valid_mask = np.ones(owned.row_count, dtype=bool)
    geometries = GeoSeries(GeometryArray.from_owned(owned))

    monkeypatch.setattr(
        validity_module,
        "plan_dispatch_selection",
        lambda *args, **kwargs: pytest.fail(
            "seeded validity cache should bypass a fresh is_valid dispatch"
        ),
    )

    repaired = overlay_module._make_valid_geoseries(
        geometries,
        dispatch_mode=ExecutionMode.GPU if vibespatial.has_gpu_runtime() else ExecutionMode.AUTO,
    )

    assert repaired is geometries


def test_overlay_intersection_seeds_polygon_validity_cache_on_owned_result() -> None:
    residency = Residency.DEVICE if vibespatial.has_gpu_runtime() else Residency.HOST
    left = GeoDataFrame(
        {"left": [1]},
        geometry=GeoSeries(
            GeometryArray.from_owned(
                from_shapely_geometries([box(0.0, 0.0, 4.0, 4.0)], residency=residency)
            )
        ),
    )
    right = GeoDataFrame(
        {"right": [2]},
        geometry=GeoSeries(
            GeometryArray.from_owned(
                from_shapely_geometries([box(1.0, 1.0, 3.0, 3.0)], residency=residency)
            )
        ),
    )

    result = overlay(left, right, how="intersection")

    owned = getattr(result.geometry.values, "_owned", None)
    assert owned is not None
    assert owned._cached_is_valid_mask is not None
    np.testing.assert_array_equal(
        owned._cached_is_valid_mask,
        np.ones(len(result), dtype=bool),
    )
    assert result.geometry.iloc[0].equals(box(1.0, 1.0, 3.0, 3.0))


def test_strip_non_polygon_collection_parts_removes_repair_line_artifacts() -> None:
    invalid = Polygon(
        [
            (680.0, 390.0),
            (680.0, 400.0),
            (680.0, 400.0),
            (680.0, 391.929775259655),
            (679.7726140184767, 390.0),
            (680.0, 390.0),
        ]
    )

    repaired = shapely.make_valid(invalid)
    stripped = overlay_module._strip_non_polygon_collection_parts(
        np.asarray([repaired], dtype=object)
    )[0]

    assert stripped.geom_type in {"Polygon", "MultiPolygon"}
    assert shapely.equals(
        stripped,
        Polygon(
            [
                (680.0, 390.0),
                (679.7726140184767, 390.0),
                (680.0, 391.929775259655),
                (680.0, 390.0),
            ]
        ),
    )


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


def test_overlay_intersection_keep_geom_type_warning_uses_source_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    real_filter = overlay_module._filter_polygon_intersection_rows_for_keep_geom_type
    observed: dict[str, object] = {}

    def _wrapped_filter(left_pairs, right_pairs, area_pairs, **kwargs):
        observed["left_pairs"] = left_pairs
        observed["right_pairs"] = right_pairs
        observed["left_source"] = kwargs.get("left_source")
        observed["right_source"] = kwargs.get("right_source")
        observed["left_rows"] = kwargs.get("left_rows")
        observed["right_rows"] = kwargs.get("right_rows")
        return real_filter(left_pairs, right_pairs, area_pairs, **kwargs)

    monkeypatch.setattr(
        overlay_module,
        "_filter_polygon_intersection_rows_for_keep_geom_type",
        _wrapped_filter,
    )

    with strict_native_environment():
        with pytest.warns(UserWarning, match="`keep_geom_type=True` in overlay"):
            overlay(left, right, how="intersection", keep_geom_type=None)

    assert observed["left_pairs"] is None
    assert observed["right_pairs"] is None
    assert observed["left_source"].equals(left.geometry)
    assert observed["right_source"].equals(right.geometry)
    assert np.asarray(observed["left_rows"]).dtype == np.intp
    assert np.asarray(observed["right_rows"]).dtype == np.intp


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
        {"zone": [1]},
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
        {"zone": [1]},
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
        {"zone": [1]},
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


def test_overlay_intersection_many_vs_one_small_remainder_prefers_broadcast_right_exact_helper(
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
    overlay_gpu_module = importlib.import_module("vibespatial.overlay.gpu")
    broadcast_calls: list[tuple[int, int]] = []

    monkeypatch.setattr(
        overlay_gpu_module,
        "_overlay_owned",
        lambda *args, **kwargs: pytest.fail(
            "small many-vs-one exact remainders should try the broadcast-right helper "
            "before the batched row-isolated overlay graph"
        ),
    )
    monkeypatch.setattr(
        binary_module,
        "_dispatch_polygon_intersection_overlay_broadcast_right_gpu",
        lambda left_arg, right_arg, **kwargs: (
            broadcast_calls.append((left_arg.row_count, right_arg.row_count)),
            left_arg,
        )[1],
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

    assert broadcast_calls == [(3, 1)]
    assert result.row_count == 3


def test_overlay_intersection_many_vs_one_large_remainder_uses_cached_right_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    binary_module = importlib.import_module("vibespatial.constructive.binary_constructive")
    complex_left = from_shapely_geometries(
        [box(float(i) - 1.0, 1.0, float(i) + 2.0, 4.0) for i in range(24)],
        residency=Residency.DEVICE,
    )
    right_one = from_shapely_geometries(
        [Polygon([(0, 0), (32, 0), (32, 2), (8, 2), (8, 6), (0, 6), (0, 0)])],
        residency=Residency.DEVICE,
    )

    helper_calls: list[tuple[int, int]] = []
    overlay_cached_flags: list[bool] = []
    real_broadcast = binary_module._broadcast_right_cached_segments
    real_overlay_owned = overlay_gpu_module._overlay_owned

    monkeypatch.setattr(
        overlay_module,
        "_prepare_many_vs_one_intersection_chunks",
        lambda *args, **kwargs: (
            [],
            complex_left,
            np.arange(complex_left.row_count, dtype=np.intp),
            right_one,
            ExecutionMode.GPU,
        ),
    )
    monkeypatch.setattr(
        overlay_module,
        "_OVERLAY_ROWWISE_REMAINDER_MAX",
        16,
    )

    def _wrapped_broadcast(right_arg, row_count, *, _cached_right_segments=None):
        helper_calls.append((right_arg.row_count, row_count))
        return real_broadcast(
            right_arg,
            row_count,
            _cached_right_segments=_cached_right_segments,
        )

    def _wrapped_overlay(*args, **kwargs):
        overlay_cached_flags.append(kwargs.get("_cached_right_segments") is not None)
        return real_overlay_owned(*args, **kwargs)

    monkeypatch.setattr(
        binary_module,
        "_broadcast_right_cached_segments",
        _wrapped_broadcast,
    )
    monkeypatch.setattr(overlay_gpu_module, "_overlay_owned", _wrapped_overlay)

    result = overlay_module._many_vs_one_intersection_owned(
        complex_left,
        right_one,
        0,
    )

    assert helper_calls == [(1, 24)]
    assert overlay_cached_flags == [True]
    assert result.row_count == complex_left.row_count


def test_overlay_intersection_many_vs_one_large_host_remainder_promotes_to_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    overlay_gpu_module = importlib.import_module("vibespatial.overlay.gpu")

    complex_left = from_shapely_geometries(
        [box(float(i) - 1.0, 1.0, float(i) + 2.0, 4.0) for i in range(24)],
        residency=Residency.HOST,
    )
    right_one = from_shapely_geometries(
        [Polygon([(0, 0), (32, 0), (32, 2), (8, 2), (8, 6), (0, 6), (0, 0)])],
        residency=Residency.HOST,
    )

    overlay_calls: list[tuple[str, str, int]] = []
    real_overlay_owned = overlay_gpu_module._overlay_owned

    monkeypatch.setattr(
        overlay_module,
        "_prepare_many_vs_one_intersection_chunks",
        lambda *args, **kwargs: (
            [],
            complex_left,
            np.arange(complex_left.row_count, dtype=np.intp),
            right_one,
            ExecutionMode.GPU,
        ),
    )
    monkeypatch.setattr(
        overlay_module,
        "_OVERLAY_ROWWISE_REMAINDER_MAX",
        16,
    )
    monkeypatch.setattr(
        overlay_module,
        "OVERLAY_GPU_REMAINDER_THRESHOLD",
        8,
    )

    def _wrapped_overlay(*args, **kwargs):
        overlay_calls.append(
            (
                str(args[0].residency),
                str(args[1].residency),
                args[0].row_count,
            )
        )
        return real_overlay_owned(*args, **kwargs)

    monkeypatch.setattr(overlay_gpu_module, "_overlay_owned", _wrapped_overlay)
    monkeypatch.setattr(
        overlay_module,
        "record_fallback_event",
        lambda *args, **kwargs: pytest.fail(
            "large many-vs-one host remainder should promote to device before CPU fallback"
        )
        if "many-vs-one remainder" in str(kwargs.get("reason", ""))
        else None,
    )

    result = overlay_module._many_vs_one_intersection_owned(
        complex_left,
        right_one,
        0,
    )

    assert overlay_calls == [("device", "host", 24)]
    assert result.row_count == complex_left.row_count


def test_prepare_many_vs_one_intersection_chunks_low_yield_uses_full_batch_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    bypass_module = importlib.import_module("vibespatial.overlay.bypass")

    left_owned = from_shapely_geometries(
        [
            box(0.0, 0.0, 1.0, 1.0),
            box(2.0, 0.0, 3.0, 1.0),
            box(4.0, 0.0, 5.0, 1.0),
        ],
        residency=Residency.DEVICE,
    )
    right_owned = from_shapely_geometries(
        [box(-1.0, -1.0, 10.0, 10.0)],
        residency=Residency.DEVICE,
    )
    contained = left_owned.take(np.asarray([0], dtype=np.int64))

    monkeypatch.setattr(
        overlay_module,
        "_MANY_VS_ONE_DIRECT_FULL_BATCH_MIN_ROWS",
        1,
    )
    monkeypatch.setattr(
        overlay_module,
        "_MANY_VS_ONE_DIRECT_FULL_BATCH_MAX_CONTAINED_FRACTION",
        0.5,
    )
    monkeypatch.setattr(
        bypass_module,
        "_containment_bypass_gpu",
        lambda *args, **kwargs: (
            contained,
            overlay_module.cp.asarray(
                [False, True, True],
                dtype=overlay_module.cp.bool_,
            ),
        ),
    )
    monkeypatch.setattr(
        bypass_module,
        "_is_clip_polygon_sh_eligible",
        lambda *args, **kwargs: (False, 0),
    )

    prepared = overlay_module._prepare_many_vs_one_intersection_chunks(
        left_owned,
        right_owned,
        0,
        global_positions=np.arange(left_owned.row_count, dtype=np.intp),
    )

    index_oga_pairs, complex_left, complex_positions, right_one, _pairwise_mode, use_full_batch = (
        prepared
    )
    assert use_full_batch is True
    assert index_oga_pairs == []
    assert complex_left is left_owned
    assert np.array_equal(
        complex_positions,
        np.arange(left_owned.row_count, dtype=np.intp),
    )
    assert right_one.row_count == 1


def test_prepare_many_vs_one_intersection_chunks_large_low_yield_stays_chunked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    bypass_module = importlib.import_module("vibespatial.overlay.bypass")

    left_owned = from_shapely_geometries(
        [box(float(i), 0.0, float(i) + 1.0, 1.0) for i in range(64)],
        residency=Residency.DEVICE,
    )
    right_owned = from_shapely_geometries(
        [box(-1.0, -1.0, 128.0, 8.0)],
        residency=Residency.DEVICE,
    )
    contained = left_owned.take(np.asarray([0], dtype=np.int64))

    monkeypatch.setattr(
        overlay_module,
        "_MANY_VS_ONE_DIRECT_FULL_BATCH_MIN_ROWS",
        1,
    )
    monkeypatch.setattr(
        overlay_module,
        "_MANY_VS_ONE_DIRECT_FULL_BATCH_MAX_CONTAINED_FRACTION",
        0.5,
    )
    monkeypatch.setattr(
        overlay_module,
        "_OVERLAY_ROWWISE_REMAINDER_MAX",
        32,
    )
    monkeypatch.setattr(
        bypass_module,
        "_containment_bypass_gpu",
        lambda *args, **kwargs: (
            contained,
            overlay_module.cp.asarray(
                [False] + [True] * 63,
                dtype=overlay_module.cp.bool_,
            ),
        ),
    )
    monkeypatch.setattr(
        bypass_module,
        "_is_clip_polygon_sh_eligible",
        lambda *args, **kwargs: (False, 0),
    )

    prepared = overlay_module._prepare_many_vs_one_intersection_chunks(
        left_owned,
        right_owned,
        0,
        global_positions=np.arange(left_owned.row_count, dtype=np.intp),
    )

    index_oga_pairs, complex_left, complex_positions, right_one, _pairwise_mode, use_full_batch = (
        prepared
    )
    assert use_full_batch is False
    assert complex_left is not None
    assert complex_left.row_count == left_owned.row_count
    assert np.array_equal(complex_positions, np.arange(left_owned.row_count, dtype=np.intp))
    assert right_one.row_count == 1


def test_overlay_intersection_many_vs_one_low_yield_prefers_full_batch_exact_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    complex_left = from_shapely_geometries(
        [box(float(i), 0.0, float(i) + 1.0, 1.0) for i in range(24)],
        residency=Residency.DEVICE,
    )
    right_one = from_shapely_geometries(
        [box(-1.0, -1.0, 32.0, 4.0)],
        residency=Residency.DEVICE,
    )

    binary_module = importlib.import_module("vibespatial.constructive.binary_constructive")
    broadcast_calls: list[tuple[int, int]] = []

    monkeypatch.setattr(
        overlay_module,
        "_prepare_many_vs_one_intersection_chunks",
        lambda *args, **kwargs: (
            [],
            complex_left,
            np.arange(complex_left.row_count, dtype=np.intp),
            right_one,
            ExecutionMode.GPU,
            True,
        ),
    )
    monkeypatch.setattr(
        binary_module,
        "_dispatch_polygon_intersection_overlay_broadcast_right_gpu",
        lambda left_arg, right_arg, **kwargs: (
            broadcast_calls.append((left_arg.row_count, right_arg.row_count)),
            left_arg,
        )[1],
    )
    monkeypatch.setattr(
        binary_module,
        "_dispatch_polygon_intersection_overlay_rowwise_gpu",
        lambda *args, **kwargs: pytest.fail(
            "low-yield full-batch exact overlay should stay on the scalar-right "
            "broadcast helper before the legacy rowwise path"
        ),
    )

    result = overlay_module._many_vs_one_intersection_owned(
        complex_left,
        right_one,
        0,
    )

    assert broadcast_calls == [(24, 1)]
    assert result.row_count == complex_left.row_count


def test_overlay_intersection_many_vs_one_marks_only_complex_rows_as_keep_geom_type_suspects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left_sub = from_shapely_geometries(
        [
            box(0.0, 0.0, 1.0, 1.0),
            box(2.0, 0.0, 3.0, 1.0),
            box(4.0, 0.0, 5.0, 1.0),
        ],
        residency=Residency.HOST,
    )
    right_one = from_shapely_geometries(
        [box(0.0, 0.0, 10.0, 10.0)],
        residency=Residency.HOST,
    )
    contained = from_shapely_geometries(
        [
            box(0.0, 0.0, 1.0, 1.0),
            box(4.0, 0.0, 5.0, 1.0),
        ],
        residency=Residency.HOST,
    )
    complex_left = from_shapely_geometries(
        [box(2.0, 0.0, 3.0, 1.0)],
        residency=Residency.HOST,
    )

    monkeypatch.setattr(
        overlay_module,
        "_prepare_many_vs_one_intersection_chunks",
        lambda *args, **kwargs: (
            [(np.asarray([0, 2], dtype=np.intp), contained)],
            complex_left,
            np.asarray([1], dtype=np.intp),
            right_one,
            ExecutionMode.CPU,
        ),
    )
    monkeypatch.setattr(overlay_module, "has_gpu_runtime", lambda: False)

    result = overlay_module._many_vs_one_intersection_owned(
        left_sub,
        right_one,
        0,
    )

    assert np.asarray(
        getattr(result, "_polygon_rect_boundary_overlap", None),
        dtype=bool,
    ).tolist() == [False, True, False]


def test_overlay_intersection_many_vs_one_cpu_selected_exact_host_batch_skips_prepare(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left_sub = from_shapely_geometries(
        [box(float(i), 0.0, float(i) + 2.0, 2.0) for i in range(64)],
        residency=Residency.DEVICE if vibespatial.has_gpu_runtime() else Residency.HOST,
    )
    right_one = from_shapely_geometries(
        [box(0.5, 0.5, 63.5, 2.5)],
        residency=Residency.DEVICE if vibespatial.has_gpu_runtime() else Residency.HOST,
    )

    monkeypatch.setattr(
        overlay_module,
        "plan_dispatch_selection",
        lambda *args, **kwargs: SimpleNamespace(
            selected=ExecutionMode.CPU,
            requested=ExecutionMode.AUTO,
            reason="synthetic cpu selection",
        ),
    )
    monkeypatch.setattr(
        overlay_module,
        "_prepare_many_vs_one_intersection_chunks",
        lambda *args, **kwargs: pytest.fail(
            "CPU-selected many-vs-one exact host batch should bypass containment prep"
        ),
    )

    result = overlay_module._many_vs_one_intersection_owned(
        left_sub,
        right_one,
        0,
    )

    exact_mask = np.asarray(getattr(result, "_exact_intersection_value_mask", None), dtype=bool)
    overlap_mask = np.asarray(getattr(result, "_polygon_rect_boundary_overlap", None), dtype=bool)

    assert result.row_count == left_sub.row_count
    assert int(np.count_nonzero(exact_mask)) == left_sub.row_count
    assert not bool(overlap_mask.any())


def test_host_exact_polygon_intersection_series_batch_preserves_exact_cache() -> None:
    left = GeoSeries(
        [
            box(0.0, 0.0, 2.0, 2.0),
            box(4.0, 0.0, 6.0, 2.0),
        ]
    )
    right = GeoSeries(
        [
            box(1.0, 0.0, 3.0, 2.0),
            box(5.0, 0.0, 7.0, 2.0),
        ]
    )

    result = overlay_module._host_exact_polygon_intersection_series_batch(
        left,
        right,
        np.asarray([0, 1], dtype=np.intp),
        np.asarray([0, 1], dtype=np.intp),
        crs=left.crs,
        requested=ExecutionMode.AUTO,
        reason="test host exact pair batch",
    )

    exact_mask = np.asarray(
        getattr(getattr(result.values, "_owned", None), "_exact_intersection_value_mask", None),
        dtype=bool,
    )
    assert exact_mask.tolist() == [True, True]
    assert result.geom_type.tolist() == ["Polygon", "Polygon"]


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
        {"zone": [1]},
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

    def _fake_rowwise_exact(
        left_arg,
        right_arg,
        *,
        dispatch_mode=ExecutionMode.GPU,
        _cached_right_segments=None,
    ):
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

    assert assemble_calls == 1
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

    real_projected_pairwise = native_results_module._project_pairwise_attribute_frame
    native_calls = 0

    def _count_projected_pairwise(*args, **kwargs):
        nonlocal native_calls
        native_calls += 1
        return real_projected_pairwise(*args, **kwargs)

    monkeypatch.setattr(
        native_results_module,
        "_project_pairwise_attribute_frame",
        _count_projected_pairwise,
    )
    monkeypatch.setattr(
        PairwiseConstructiveResult,
        "to_geodataframe",
        lambda *_args, **_kwargs: pytest.fail(
            "overlay fragment export should stay on native pairwise projection"
        ),
    )

    export_result, used_owned = overlay_module._overlay_intersection_export_result(left, right)

    assert isinstance(export_result, PairwiseConstructiveFragment)
    assert native_calls == 0

    materialized = export_result.to_geodataframe()
    wrapped, wrapped_used = overlay_module._overlay_intersection(left, right)

    assert native_calls == 2
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

    real_projected_pairwise = native_results_module._project_pairwise_attribute_frame
    real_left = LeftConstructiveResult.to_geodataframe
    pairwise_native_calls = 0
    left_calls = 0

    def _count_projected_pairwise(*args, **kwargs):
        nonlocal pairwise_native_calls
        pairwise_native_calls += 1
        return real_projected_pairwise(*args, **kwargs)

    def _count_left(self, *args, **kwargs):
        nonlocal left_calls
        left_calls += 1
        return real_left(self, *args, **kwargs)

    monkeypatch.setattr(
        native_results_module,
        "_project_pairwise_attribute_frame",
        _count_projected_pairwise,
    )
    monkeypatch.setattr(
        PairwiseConstructiveResult,
        "to_geodataframe",
        lambda *_args, **_kwargs: pytest.fail(
            "identity pairwise fragment export should stay on native pairwise projection"
        ),
    )
    monkeypatch.setattr(LeftConstructiveResult, "to_geodataframe", _count_left)

    native_result, used_owned = overlay_module._overlay_identity_native(left, right)

    assert isinstance(native_result, ConcatConstructiveResult)
    assert pairwise_native_calls == 0
    assert left_calls == 0

    materialized = native_result.to_geodataframe()
    wrapped, wrapped_used = overlay_module._overlay_identity(left, right)

    assert pairwise_native_calls == 2
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

    real_projected_pairwise = native_results_module._project_pairwise_attribute_frame
    real_left = LeftConstructiveResult.to_geodataframe
    pairwise_native_calls = 0
    left_calls = 0

    def _count_projected_pairwise(*args, **kwargs):
        nonlocal pairwise_native_calls
        pairwise_native_calls += 1
        return real_projected_pairwise(*args, **kwargs)

    def _count_left(self, *args, **kwargs):
        nonlocal left_calls
        left_calls += 1
        return real_left(self, *args, **kwargs)

    monkeypatch.setattr(
        native_results_module,
        "_project_pairwise_attribute_frame",
        _count_projected_pairwise,
    )
    monkeypatch.setattr(
        PairwiseConstructiveResult,
        "to_geodataframe",
        lambda *_args, **_kwargs: pytest.fail(
            "union pairwise fragment export should stay on native pairwise projection"
        ),
    )
    monkeypatch.setattr(LeftConstructiveResult, "to_geodataframe", _count_left)

    native_result, used_owned = overlay_module._overlay_union_native(left, right)

    assert isinstance(native_result, ConcatConstructiveResult)
    assert pairwise_native_calls == 0
    assert left_calls == 0

    materialized = native_result.to_geodataframe()
    wrapped, wrapped_used = overlay_module._overlay_union(left, right)

    assert pairwise_native_calls == 2
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


def test_overlay_intersection_few_right_large_batches_reuse_cached_right_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.constructive import binary_constructive as constructive_module

    left = GeoDataFrame(
        {"col1": np.arange(320, dtype=np.int32)},
        geometry=GeoSeries([box(float(i), 0.0, float(i) + 1.0, 1.0) for i in range(320)]),
    )
    right = GeoDataFrame(
        {"zone_type": ["A", "B", "C", "D"]},
        geometry=GeoSeries(
            [
                Polygon(
                    [(-20, -2), (100, -2), (100, 3), (-20, 3)],
                    holes=[[(10, -1), (20, -1), (20, 2), (10, 2)]],
                ),
                Polygon(
                    [(60, -2), (180, -2), (180, 3), (60, 3)],
                    holes=[[(90, -1), (100, -1), (100, 2), (90, 2)]],
                ),
                Polygon(
                    [(140, -2), (260, -2), (260, 3), (140, 3)],
                    holes=[[(170, -1), (180, -1), (180, 2), (170, 2)]],
                ),
                Polygon(
                    [(220, -2), (340, -2), (340, 3), (220, 3)],
                    holes=[[(250, -1), (260, -1), (260, 2), (250, 2)]],
                ),
            ]
        ),
    )
    left_owned = left.geometry.values.to_owned()
    right_owned = right.geometry.values.to_owned()
    idx1 = np.arange(320, dtype=np.int32)
    idx2 = np.repeat(np.arange(4, dtype=np.int32), 80)

    rowwise_calls: list[tuple[int, bool]] = []

    monkeypatch.setattr(
        constructive_module,
        "binary_constructive_owned",
        lambda *args, **kwargs: pytest.fail(
            "large few-right batches should stay on one exact rowwise pass"
        ),
    )

    def _fake_rowwise(left_arg, right_arg, *, dispatch_mode=ExecutionMode.GPU, _cached_right_segments=None):
        rowwise_calls.append((left_arg.row_count, _cached_right_segments is not None))
        geoms = [
            box(float(9000 + i), 0.0, float(9000 + i + 0.5), 0.5)
            for i in range(left_arg.row_count)
        ]
        return from_shapely_geometries(geoms, residency=Residency.DEVICE)

    monkeypatch.setattr(
        constructive_module,
        "_dispatch_polygon_intersection_overlay_rowwise_gpu",
        _fake_rowwise,
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
    assert rowwise_calls == [(320, True)]
    assert result["col1"].tolist() == idx1.tolist()
    assert result["zone_type"].tolist() == ["A"] * 80 + ["B"] * 80 + ["C"] * 80 + ["D"] * 80


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


def test_overlay_intersection_exact_mode_resolves_indexed_rectangle_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    import cupy as cp

    from vibespatial.constructive import binary_constructive as constructive_module
    from vibespatial.spatial.query_types import DeviceSpatialJoinResult

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
                Point(0.0, 0.0).buffer(5.0),
                Point(20.0, 0.0).buffer(5.0),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries(
            [
                box(-2.0, -2.0, 2.0, 2.0),
                box(18.0, -3.0, 24.0, 3.0),
            ]
        ),
    )
    left_owned = left.geometry.values.to_owned()
    right_owned = right.geometry.values.to_owned()
    left_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test indexed rectangle overlay exact mode",
    )
    right_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test indexed rectangle overlay exact mode",
    )

    d_idx = cp.asarray([0, 0, 1, 1], dtype=cp.int32)
    index_result = DeviceSpatialJoinResult(d_idx, d_idx)

    kernel_calls: list[int] = []

    def _fake_polygon_rect_intersection(left_arg, right_arg, *, dispatch_mode=ExecutionMode.GPU):
        kernel_calls.append(left_arg.row_count)
        return from_shapely_geometries(
            [
                box(100.0, 0.0, 101.0, 1.0),
                box(110.0, 0.0, 111.0, 1.0),
                box(200.0, 0.0, 201.0, 1.0),
                box(210.0, 0.0, 211.0, 1.0),
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
            "indexed rectangle-capable batches should use polygon_rect_intersection before polygon_intersection"
        ),
    )
    monkeypatch.setattr(
        constructive_module,
        "_dispatch_polygon_intersection_overlay_rowwise_gpu",
        lambda *args, **kwargs: pytest.fail(
            "indexed rectangle-capable batches should use polygon_rect_intersection before rowwise overlay"
        ),
    )

    result, used_owned = overlay_module._overlay_intersection(
        left,
        right,
        left_owned=left_owned,
        right_owned=right_owned,
        _prefer_exact_polygon_gpu=True,
        _index_result=index_result,
    )

    assert used_owned is True
    assert kernel_calls == [4]
    assert result["col1"].tolist() == [1, 1, 2, 2]
    assert result["col2"].tolist() == [10, 10, 20, 20]
    assert [geom.bounds for geom in result.geometry] == [
        (100.0, 0.0, 101.0, 1.0),
        (110.0, 0.0, 111.0, 1.0),
        (200.0, 0.0, 201.0, 1.0),
        (210.0, 0.0, 211.0, 1.0),
    ]


def test_overlay_intersection_host_polygon_boundary_prefers_pair_owned_gpu_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.constructive import binary_constructive as constructive_module

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
    idx = np.arange(2, dtype=np.int32)

    kernel_calls: list[int] = []

    def _fake_binary(*args, **kwargs):
        kernel_calls.append(args[1].row_count)
        return from_shapely_geometries(
            [
                box(100.0, 0.0, 101.0, 1.0),
                box(200.0, 0.0, 201.0, 1.0),
            ],
            residency=Residency.DEVICE,
        )

    monkeypatch.setattr(
        constructive_module,
        "binary_constructive_owned",
        _fake_binary,
    )
    monkeypatch.setattr(
        overlay_module,
        "_take_geoseries_object_values",
        lambda *_args, **_kwargs: pytest.fail(
            "small polygon pair batches should prefer pair-owned GPU exact path before host exact intersection"
        ),
    )

    result, used_owned = overlay_module._overlay_intersection(
        left,
        right,
        _index_result=(idx, idx),
    )

    assert used_owned is True
    assert kernel_calls == [2]
    assert result["col1"].tolist() == [1, 2]
    assert result["col2"].tolist() == [10, 20]


def test_overlay_intersection_warning_path_prefers_pair_owned_gpu_boundary_for_small_device_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.constructive import binary_constructive as constructive_module

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
    left_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test overlay warning path prefers pair-owned exact boundary",
    )
    right_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test overlay warning path prefers pair-owned exact boundary",
    )
    kernel_calls: list[int] = []

    def _fake_binary(*args, **kwargs):
        kernel_calls.append(args[1].row_count)
        result = from_shapely_geometries(
            [
                box(100.0, 0.0, 101.0, 1.0),
                box(200.0, 0.0, 201.0, 1.0),
            ],
            residency=Residency.DEVICE,
        )
        result._polygon_rect_boundary_overlap = np.zeros(2, dtype=bool)
        return result

    monkeypatch.setattr(
        constructive_module,
        "binary_constructive_owned",
        _fake_binary,
    )
    monkeypatch.setattr(
        constructive_module,
        "_dispatch_polygon_intersection_overlay_rowwise_gpu",
        lambda *args, **kwargs: pytest.fail(
            "small device-backed warning batches should stay on the pair-owned boundary path"
        ),
    )

    result, used_owned = overlay_module._overlay_intersection(
        left,
        right,
        left_owned=left_owned,
        right_owned=right_owned,
        _prefer_exact_polygon_gpu=True,
        _warn_on_dropped_lower_dim_polygon_results=True,
        _index_result=(np.arange(2, dtype=np.int32), np.arange(2, dtype=np.int32)),
    )

    assert used_owned is True
    assert kernel_calls == [2]
    assert result["col1"].tolist() == [1, 2]
    assert result["col2"].tolist() == [10, 20]


def test_overlay_intersection_rect_overlap_metadata_skips_generic_make_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not vibespatial.has_gpu_runtime():
        pytest.skip("GPU runtime not available")

    from vibespatial.constructive import binary_constructive as constructive_module

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
    left_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test overlay rect overlap metadata skips generic make_valid",
    )
    right_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test overlay rect overlap metadata skips generic make_valid",
    )

    def _fake_binary(*args, **kwargs):
        result = from_shapely_geometries(
            [
                box(100.0, 0.0, 101.0, 1.0),
                box(200.0, 0.0, 201.0, 1.0),
            ],
            residency=Residency.DEVICE,
        )
        result._polygon_rect_boundary_overlap = np.zeros(2, dtype=bool)
        return result

    monkeypatch.setattr(
        constructive_module,
        "binary_constructive_owned",
        _fake_binary,
    )
    monkeypatch.setattr(
        overlay_module,
        "_make_valid_geoseries",
        lambda *_args, **_kwargs: pytest.fail(
            "rectangle-overlap-tagged intersection batches should defer generic make_valid"
        ),
    )

    result, used_owned = overlay_module._overlay_intersection(
        left,
        right,
        left_owned=left_owned,
        right_owned=right_owned,
        _prefer_exact_polygon_gpu=True,
        _warn_on_dropped_lower_dim_polygon_results=True,
        _index_result=(np.arange(2, dtype=np.int32), np.arange(2, dtype=np.int32)),
    )

    assert used_owned is True
    assert result["col1"].tolist() == [1, 2]
    assert result["col2"].tolist() == [10, 20]


def test_overlay_intersection_exact_mode_splits_mixed_rectangle_batches(
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
                Point(0.0, 0.0).buffer(5.0),
                Polygon(
                    [
                        (18.0, -5.0),
                        (26.0, -5.0),
                        (26.0, 5.0),
                        (18.0, 5.0),
                        (18.0, -5.0),
                    ],
                    holes=[[(20.0, -1.0), (24.0, -1.0), (24.0, 1.0), (20.0, 1.0), (20.0, -1.0)]],
                ),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries(
            [
                box(-2.0, -2.0, 2.0, 2.0),
                box(18.0, -3.0, 24.0, 3.0),
            ]
        ),
    )
    left_owned = left.geometry.values.to_owned()
    right_owned = right.geometry.values.to_owned()

    rect_calls: list[int] = []
    rowwise_calls: list[int] = []

    def _fake_polygon_rect_intersection(left_arg, right_arg, *, dispatch_mode=ExecutionMode.GPU):
        rect_calls.append(left_arg.row_count)
        return from_shapely_geometries(
            [box(100.0, 0.0, 101.0, 1.0)],
            residency=Residency.DEVICE,
        )

    def _fake_rowwise(left_arg, right_arg, *, dispatch_mode=ExecutionMode.GPU, _cached_right_segments=None):
        rowwise_calls.append(left_arg.row_count)
        return from_shapely_geometries(
            [box(200.0, 0.0, 201.0, 1.0)],
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
            "mixed rectangle batches should bypass polygon_intersection and split into rectangle + remainder paths"
        ),
    )
    monkeypatch.setattr(
        constructive_module,
        "_dispatch_polygon_intersection_overlay_rowwise_gpu",
        _fake_rowwise,
    )

    result, used_owned = overlay_module._overlay_intersection(
        left,
        right,
        left_owned=left_owned,
        right_owned=right_owned,
        _prefer_exact_polygon_gpu=True,
        _index_result=(np.arange(2, dtype=np.int32), np.arange(2, dtype=np.int32)),
    )

    assert used_owned is True
    assert rect_calls == [1]
    assert rowwise_calls == [1]
    assert result["col1"].tolist() == [1, 2]
    assert result["col2"].tolist() == [10, 20]
    assert [geom.bounds for geom in result.geometry] == [
        (100.0, 0.0, 101.0, 1.0),
        (200.0, 0.0, 201.0, 1.0),
    ]


def test_overlay_intersection_exact_mode_recovers_all_handled_rect_batches_when_batch_probe_misses(
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
                Point(0.0, 0.0).buffer(5.0),
                Point(20.0, 0.0).buffer(5.0),
            ]
        ),
    )
    right = GeoDataFrame(
        {"col2": [10, 20]},
        geometry=GeoSeries(
            [
                box(-2.0, -2.0, 2.0, 2.0),
                box(18.0, -3.0, 24.0, 3.0),
            ]
        ),
    )
    left_owned = left.geometry.values.to_owned()
    right_owned = right.geometry.values.to_owned()

    rect_calls: list[int] = []

    def _fake_polygon_rect_intersection(left_arg, right_arg, *, dispatch_mode=ExecutionMode.GPU):
        rect_calls.append(left_arg.row_count)
        return from_shapely_geometries(
            [
                box(100.0, 0.0, 101.0, 1.0),
                box(200.0, 0.0, 201.0, 1.0),
            ],
            residency=Residency.DEVICE,
        )

    monkeypatch.setattr(
        polygon_rect_intersection_module,
        "polygon_rect_intersection_can_handle",
        lambda *_args, **_kwargs: False,
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
            "all-rectangle-capable batches should still recover through the rectangle subset path"
        ),
    )
    monkeypatch.setattr(
        constructive_module,
        "_dispatch_polygon_intersection_overlay_rowwise_gpu",
        lambda *args, **kwargs: pytest.fail(
            "all-rectangle-capable batches should not fall through to rowwise overlay when the batch probe misses"
        ),
    )

    result, used_owned = overlay_module._overlay_intersection(
        left,
        right,
        left_owned=left_owned,
        right_owned=right_owned,
        _prefer_exact_polygon_gpu=True,
        _index_result=(np.arange(2, dtype=np.int32), np.arange(2, dtype=np.int32)),
    )

    assert used_owned is True
    assert rect_calls == [2]
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
