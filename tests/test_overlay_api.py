from __future__ import annotations

import importlib
import os

import numpy as np
import pytest
import shapely
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point, Polygon, box

import vibespatial
from vibespatial.api import GeoDataFrame, GeoSeries, read_file
from vibespatial.api.geometry_array import GeometryArray
from vibespatial.api.tools.overlay import overlay
from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.runtime import ExecutionMode
from vibespatial.testing import strict_native_environment

overlay_module = importlib.import_module("vibespatial.api.tools.overlay")


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
    assert result.geometry.iloc[0].geom_type in {"Polygon", "MultiPolygon"}
    assert result.geometry.iloc[0].area > 0
    assert any(
        event.surface == "geopandas.array.difference"
        and event.selected is ExecutionMode.GPU
        for event in events
    )


def test_overlay_difference_prefers_rowwise_gpu_path_after_grouped_unions(
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

    recorded: list[bool] = []
    original = constructive_module.binary_constructive_owned

    def _wrapped_binary_constructive_owned(*args, **kwargs):
        if args and args[0] == "difference":
            recorded.append(
                bool(kwargs.get("_prefer_rowwise_polygon_difference_overlay", False))
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(
        constructive_module,
        "binary_constructive_owned",
        _wrapped_binary_constructive_owned,
    )

    with strict_native_environment():
        result = overlay(left, right, how="difference")

    assert len(result) == 2
    assert all(recorded)


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
    assert dropped == 2
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
