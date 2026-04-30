"""Tests for broadcast-right support in distance and metric operations.

Validates that scalar (1-row) right operands produce correct results for
distance, dwithin, hausdorff_distance, and frechet_distance.  Oracle:
Shapely with tiled-pairwise comparison.

Covers nsf.4: distance and metric broadcast-right support.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, Point, box

import vibespatial
from vibespatial import has_gpu_runtime
from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.fallbacks import STRICT_NATIVE_ENV_VAR, StrictNativeFallbackError
from vibespatial.runtime.precision import PrecisionMode
from vibespatial.runtime.residency import Residency
from vibespatial.spatial.distance_metrics import frechet_distance_owned, hausdorff_distance_owned
from vibespatial.spatial.distance_owned import distance_owned, dwithin_owned

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_distance_metrics_d2h_exports_are_runtime_accounted() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "src" / "vibespatial" / "spatial" / "distance_metrics.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    raw_exports: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr == "asnumpy":
            raw_exports.append(f"{path.relative_to(repo_root)}:{node.lineno}")
        if func.attr == "get":
            raw_exports.append(f"{path.relative_to(repo_root)}:{node.lineno}")
        if func.attr == "copy_device_to_host" and not any(
            keyword.arg == "reason" for keyword in node.keywords
        ):
            raw_exports.append(f"{path.relative_to(repo_root)}:{node.lineno}")
    assert raw_exports == []


def _shapely_distance_oracle(
    left_geoms: list[object | None],
    right_geom: object | None,
) -> np.ndarray:
    """Compute expected distances by tiling right_geom and using Shapely."""
    left_arr = np.asarray(left_geoms, dtype=object)
    right_arr = np.asarray([right_geom] * len(left_geoms), dtype=object)
    return np.asarray(shapely.distance(left_arr, right_arr), dtype=np.float64)


# ---------------------------------------------------------------------------
# 1. distance broadcast: distance(N points, 1 polygon) == tiled-pairwise
# ---------------------------------------------------------------------------

def test_distance_broadcast_point_polygon() -> None:
    """distance(N points, 1 polygon) matches tiled Shapely oracle."""
    left = [
        Point(1, 1),
        Point(0, 0),
        Point(3, 3),
        Point(0.5, 0.5),
        Point(10, 10),
    ]
    right_geom = box(0, 0, 2, 2)

    left_owned = from_shapely_geometries(left)
    right_owned = from_shapely_geometries([right_geom])

    result = distance_owned(left_owned, right_owned)
    expected = _shapely_distance_oracle(left, right_geom)

    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_distance_broadcast_point_point() -> None:
    """distance(N points, 1 point) matches tiled Shapely oracle."""
    left = [Point(0, 0), Point(1, 1), Point(3, 4)]
    right_geom = Point(1, 0)

    left_owned = from_shapely_geometries(left)
    right_owned = from_shapely_geometries([right_geom])

    result = distance_owned(left_owned, right_owned)
    expected = _shapely_distance_oracle(left, right_geom)

    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_distance_broadcast_polygon_polygon() -> None:
    """distance(N polygons, 1 polygon) matches tiled Shapely oracle."""
    left = [box(0, 0, 1, 1), box(2, 2, 3, 3), box(5, 5, 6, 6)]
    right_geom = box(1.5, 1.5, 2.5, 2.5)

    left_owned = from_shapely_geometries(left)
    right_owned = from_shapely_geometries([right_geom])

    result = distance_owned(left_owned, right_owned)
    expected = _shapely_distance_oracle(left, right_geom)

    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_distance_broadcast_linestring_polygon() -> None:
    """distance(N linestrings, 1 polygon) matches tiled Shapely oracle."""
    left = [
        LineString([(0, 0), (1, 0)]),
        LineString([(3, 3), (4, 4)]),
        LineString([(0, 0), (5, 5)]),
    ]
    right_geom = box(1.5, 1.5, 2.5, 2.5)

    left_owned = from_shapely_geometries(left)
    right_owned = from_shapely_geometries([right_geom])

    result = distance_owned(left_owned, right_owned)
    expected = _shapely_distance_oracle(left, right_geom)

    np.testing.assert_allclose(result, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# 2. dwithin broadcast: dwithin(N points, 1 polygon, threshold) == tiled
# ---------------------------------------------------------------------------

def test_dwithin_broadcast_point_polygon() -> None:
    """dwithin(N points, 1 polygon, threshold) matches tiled result."""
    left = [
        Point(1, 1),      # inside -> dist=0 -> True
        Point(0, 0),      # on corner -> dist=0 -> True
        Point(3, 3),      # outside -> dist=sqrt(2) ~ 1.41 -> False (threshold=1.0)
        Point(2.5, 1),    # outside -> dist=0.5 -> True
        Point(10, 10),    # far outside -> False
    ]
    right_geom = box(0, 0, 2, 2)
    threshold = 1.0

    left_owned = from_shapely_geometries(left)
    right_owned = from_shapely_geometries([right_geom])

    result = dwithin_owned(left_owned, right_owned, threshold)

    # Oracle: compute distances and compare
    distances = _shapely_distance_oracle(left, right_geom)
    expected = np.where(np.isnan(distances), False, distances <= threshold)

    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# 3. Null broadcast: single null geometry -> all-NaN for distance
# ---------------------------------------------------------------------------

def test_distance_broadcast_null_right() -> None:
    """Broadcast of null right geometry produces all-NaN distances."""
    left = [Point(1, 1), Point(2, 2), Point(3, 3)]
    left_owned = from_shapely_geometries(left)
    right_owned = from_shapely_geometries([None])

    result = distance_owned(left_owned, right_owned)

    assert len(result) == 3
    assert all(np.isnan(result)), f"Expected all NaN, got {result}"


def test_dwithin_broadcast_null_right() -> None:
    """Broadcast of null right geometry produces all-False for dwithin."""
    left = [Point(1, 1), Point(2, 2)]
    left_owned = from_shapely_geometries(left)
    right_owned = from_shapely_geometries([None])

    result = dwithin_owned(left_owned, right_owned, 10.0)

    assert len(result) == 2
    assert not any(result), f"Expected all False, got {result}"


# ---------------------------------------------------------------------------
# 4. Index preservation: result length == left length
# ---------------------------------------------------------------------------

def test_distance_broadcast_result_length() -> None:
    """Result length matches left array length, not right."""
    n = 50
    left = [Point(i, i) for i in range(n)]
    right_geom = Point(0, 0)

    left_owned = from_shapely_geometries(left)
    right_owned = from_shapely_geometries([right_geom])

    result = distance_owned(left_owned, right_owned)
    assert len(result) == n, f"Expected {n}, got {len(result)}"


def test_dwithin_broadcast_result_length() -> None:
    """dwithin result length matches left array length."""
    n = 30
    left = [Point(i, i) for i in range(n)]
    right_geom = Point(0, 0)

    left_owned = from_shapely_geometries(left)
    right_owned = from_shapely_geometries([right_geom])

    result = dwithin_owned(left_owned, right_owned, 100.0)
    assert len(result) == n, f"Expected {n}, got {len(result)}"


# ---------------------------------------------------------------------------
# 5. Tiling equivalence regression: broadcast == old N-copy
# ---------------------------------------------------------------------------

def test_distance_tiling_equivalence() -> None:
    """Broadcast distance matches old N-copy tiling approach."""
    left = [Point(0, 0), Point(1, 1), Point(3, 4)]
    right_geom = Point(1, 0)

    left_owned = from_shapely_geometries(left)

    # New broadcast path
    right_broadcast = from_shapely_geometries([right_geom])
    result_broadcast = distance_owned(left_owned, right_broadcast)

    # Old tiling path
    right_tiled = from_shapely_geometries([right_geom] * len(left))
    result_tiled = distance_owned(left_owned, right_tiled)

    np.testing.assert_allclose(result_broadcast, result_tiled, rtol=1e-12)


# ---------------------------------------------------------------------------
# 6. hausdorff_distance broadcast
# ---------------------------------------------------------------------------

def test_hausdorff_broadcast() -> None:
    """hausdorff_distance(N linestrings, 1 linestring) matches tiled oracle."""
    left = [
        LineString([(0, 0), (1, 0)]),
        LineString([(0, 0), (2, 0)]),
        LineString([(0, 0), (0.5, 0)]),
    ]
    right_geom = LineString([(0, 1), (1, 1)])

    left_owned = from_shapely_geometries(left)
    right_owned = from_shapely_geometries([right_geom])

    result = hausdorff_distance_owned(left_owned, right_owned)

    # Oracle: tiled Shapely hausdorff_distance
    expected = np.array([
        shapely.hausdorff_distance(geom, right_geom)
        for geom in left
    ], dtype=np.float64)

    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_hausdorff_broadcast_tiling_equivalence() -> None:
    """Broadcast hausdorff_distance matches old N-copy tiling."""
    left = [
        LineString([(0, 0), (1, 0), (2, 1)]),
        LineString([(0, 0), (3, 0)]),
    ]
    right_geom = LineString([(0, 1), (1, 1), (2, 1)])

    left_owned = from_shapely_geometries(left)

    result_broadcast = hausdorff_distance_owned(
        left_owned, from_shapely_geometries([right_geom]),
    )
    result_tiled = hausdorff_distance_owned(
        left_owned, from_shapely_geometries([right_geom] * len(left)),
    )

    np.testing.assert_allclose(result_broadcast, result_tiled, rtol=1e-10)


def test_hausdorff_owned_densify_matches_shapely() -> None:
    """Owned Hausdorff honors densify instead of silently using vertices only."""
    left_geom = LineString([(0, 0), (10, 0)])
    right_geom = LineString([(0, 0), (5, 5), (10, 0)])

    result = hausdorff_distance_owned(
        from_shapely_geometries([left_geom]),
        from_shapely_geometries([right_geom]),
        densify=0.5,
    )
    expected = np.asarray(
        [shapely.hausdorff_distance(left_geom, right_geom, densify=0.5)],
        dtype=np.float64,
    )

    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_hausdorff_gpu_densify_matches_shapely() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for Hausdorff densify kernel")

    left_geom = LineString([(0, 0), (10, 0)])
    right_geom = LineString([(0, 0), (5, 5), (10, 0)])
    left_owned = from_shapely_geometries([left_geom] * 128, residency=Residency.DEVICE)
    right_owned = from_shapely_geometries([right_geom] * 128, residency=Residency.DEVICE)

    result = hausdorff_distance_owned(left_owned, right_owned, densify=0.5)
    expected = np.full(
        128,
        shapely.hausdorff_distance(left_geom, right_geom, densify=0.5),
        dtype=np.float64,
    )

    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_hausdorff_dispatch_forwards_precision_plan(monkeypatch) -> None:
    import vibespatial.spatial.distance_metrics as distance_metrics

    left_owned = from_shapely_geometries([
        LineString([(1_000_000.0, 1_000_000.0), (1_000_001.0, 1_000_000.0)]),
        LineString([(1_000_002.0, 1_000_000.0), (1_000_003.0, 1_000_000.0)]),
    ])
    right_owned = from_shapely_geometries([
        LineString([(1_000_000.0, 1_000_001.0), (1_000_001.0, 1_000_001.0)])
    ])

    precision_plan = SimpleNamespace(
        compute_precision=PrecisionMode.FP32,
        center_coordinates=True,
    )
    selection = SimpleNamespace(
        selected=ExecutionMode.GPU,
        precision_plan=precision_plan,
        reason="test gpu",
        requested=ExecutionMode.GPU,
    )
    captured = {}

    def fake_plan_dispatch_selection(**kwargs):
        captured["selection_kwargs"] = kwargs
        return selection

    def fake_hausdorff_gpu(owned_a, owned_b, *, precision_plan=None, densify_steps=1):
        captured["precision_plan"] = precision_plan
        captured["densify_steps"] = densify_steps
        return np.full(owned_a.row_count, 4.0, dtype=np.float64)

    monkeypatch.setattr(
        distance_metrics, "plan_dispatch_selection", fake_plan_dispatch_selection,
    )
    monkeypatch.setattr(distance_metrics, "_hausdorff_gpu", fake_hausdorff_gpu)

    result = hausdorff_distance_owned(
        left_owned, right_owned, precision=PrecisionMode.FP32,
    )

    assert captured["selection_kwargs"]["requested_precision"] is PrecisionMode.FP32
    assert captured["selection_kwargs"]["coordinate_stats"].max_abs_coord >= 1_000_003.0
    assert captured["precision_plan"] is precision_plan
    assert captured["densify_steps"] == 1
    np.testing.assert_allclose(result, [4.0, 4.0])


def test_hausdorff_strict_native_gpu_failure_is_not_swallowed(monkeypatch) -> None:
    import vibespatial.spatial.distance_metrics as distance_metrics

    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")
    vibespatial.clear_fallback_events()
    line = LineString([(0, 0), (1, 1)])
    left_owned = from_shapely_geometries([line], residency=Residency.DEVICE)
    right_owned = from_shapely_geometries([line], residency=Residency.DEVICE)
    precision_plan = SimpleNamespace(compute_precision=PrecisionMode.FP64)
    selection = SimpleNamespace(
        selected=ExecutionMode.GPU,
        precision_plan=precision_plan,
        reason="test gpu",
        requested=ExecutionMode.GPU,
    )

    monkeypatch.setattr(
        distance_metrics,
        "plan_dispatch_selection",
        lambda **kwargs: selection,
    )

    def fake_hausdorff_gpu(*args, **kwargs):
        raise RuntimeError("forced hausdorff failure")

    monkeypatch.setattr(distance_metrics, "_hausdorff_gpu", fake_hausdorff_gpu)

    with pytest.raises(StrictNativeFallbackError, match="hausdorff_distance"):
        hausdorff_distance_owned(left_owned, right_owned)

    events = vibespatial.get_fallback_events(clear=True)
    assert events
    assert events[-1].surface == "geopandas.array.hausdorff_distance"
    assert events[-1].reason == "GPU hausdorff_distance failed"
    assert events[-1].d2h_transfer is True


# ---------------------------------------------------------------------------
# 7. frechet_distance broadcast
# ---------------------------------------------------------------------------

def test_frechet_broadcast() -> None:
    """frechet_distance(N linestrings, 1 linestring) matches tiled oracle."""
    left = [
        LineString([(0, 0), (1, 0)]),
        LineString([(0, 0), (2, 0)]),
        LineString([(0, 0), (0.5, 0)]),
    ]
    right_geom = LineString([(0, 1), (1, 1)])

    left_owned = from_shapely_geometries(left)
    right_owned = from_shapely_geometries([right_geom])

    result = frechet_distance_owned(left_owned, right_owned)

    # Oracle: tiled Shapely frechet_distance
    expected = np.array([
        shapely.frechet_distance(geom, right_geom)
        for geom in left
    ], dtype=np.float64)

    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_frechet_broadcast_tiling_equivalence() -> None:
    """Broadcast frechet_distance matches old N-copy tiling."""
    left = [
        LineString([(0, 0), (1, 0), (2, 1)]),
        LineString([(0, 0), (3, 0)]),
    ]
    right_geom = LineString([(0, 1), (1, 1), (2, 1)])

    left_owned = from_shapely_geometries(left)

    result_broadcast = frechet_distance_owned(
        left_owned, from_shapely_geometries([right_geom]),
    )
    result_tiled = frechet_distance_owned(
        left_owned, from_shapely_geometries([right_geom] * len(left)),
    )

    np.testing.assert_allclose(result_broadcast, result_tiled, rtol=1e-10)


def test_frechet_owned_densify_matches_shapely() -> None:
    left_geom = LineString([(100, 0), (0, 0), (0, 100)])
    right_geom = LineString([(5, 5), (5, 100), (100, 5)])

    result = frechet_distance_owned(
        from_shapely_geometries([left_geom]),
        from_shapely_geometries([right_geom]),
        densify=0.25,
    )
    expected = np.asarray(
        [shapely.frechet_distance(left_geom, right_geom, densify=0.25)],
        dtype=np.float64,
    )

    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_frechet_gpu_densify_matches_shapely() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for Frechet densify kernel")

    left_geom = LineString([(100, 0), (0, 0), (0, 100)])
    right_geom = LineString([(5, 5), (5, 100), (100, 5)])
    left_owned = from_shapely_geometries([left_geom] * 128, residency=Residency.DEVICE)
    right_owned = from_shapely_geometries([right_geom] * 128, residency=Residency.DEVICE)

    result = frechet_distance_owned(left_owned, right_owned, densify=0.25)
    expected = np.full(
        128,
        shapely.frechet_distance(left_geom, right_geom, densify=0.25),
        dtype=np.float64,
    )

    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_frechet_strict_native_gpu_failure_is_not_swallowed(monkeypatch) -> None:
    import vibespatial.spatial.distance_metrics as distance_metrics

    monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")
    vibespatial.clear_fallback_events()
    line = LineString([(0, 0), (1, 1)])
    left_owned = from_shapely_geometries([line], residency=Residency.DEVICE)
    right_owned = from_shapely_geometries([line], residency=Residency.DEVICE)
    selection = SimpleNamespace(
        selected=ExecutionMode.GPU,
        reason="test gpu",
        requested=ExecutionMode.GPU,
    )

    monkeypatch.setattr(
        distance_metrics,
        "plan_dispatch_selection",
        lambda **kwargs: selection,
    )

    def fake_frechet_gpu(*args, **kwargs):
        raise RuntimeError("forced frechet failure")

    monkeypatch.setattr(distance_metrics, "_frechet_gpu", fake_frechet_gpu)

    with pytest.raises(StrictNativeFallbackError, match="frechet_distance"):
        frechet_distance_owned(left_owned, right_owned)

    events = vibespatial.get_fallback_events(clear=True)
    assert events
    assert events[-1].surface == "geopandas.array.frechet_distance"
    assert events[-1].reason == "GPU frechet_distance failed"
    assert events[-1].d2h_transfer is True
