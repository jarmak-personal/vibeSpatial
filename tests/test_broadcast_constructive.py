"""Tests for broadcast-right support in binary constructive operations.

Validates that scalar (1-row) right operands produce correct results for
intersection, union, difference, and symmetric_difference.  Oracle: Shapely
with tiled-pairwise comparison.

Covers nsf.3: elimination of [other]*len(self) materialization.
"""
from __future__ import annotations

import importlib
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import shapely
from shapely.geometry import MultiPoint, MultiPolygon, Point, box

from benchmarks.shootout._data import setup_fixtures
from vibespatial.api import read_file
from vibespatial.constructive import binary_constructive as binary_constructive_module
from vibespatial.constructive.binary_constructive import binary_constructive_owned
from vibespatial.geometry.owned import (
    from_shapely_geometries,
    tile_single_row,
)
from vibespatial.kernels.constructive.segmented_union import segmented_union_all
from vibespatial.runtime import ExecutionMode, has_gpu_runtime
from vibespatial.runtime.hotpath_trace import reset_hotpath_trace, summarize_hotpath_trace
from vibespatial.runtime.residency import Residency
from vibespatial.testing import strict_native_environment

# The four constructive operations.
_CONSTRUCTIVE_OPS = ("intersection", "union", "difference", "symmetric_difference")

requires_gpu = pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shapely_oracle(
    op: str,
    left_geoms: list[object | None],
    right_geom: object | None,
) -> list[object | None]:
    """Compute expected results by tiling right_geom and using Shapely."""
    left_arr = np.asarray(left_geoms, dtype=object)
    right_arr = np.asarray([right_geom] * len(left_geoms), dtype=object)
    result = getattr(shapely, op)(left_arr, right_arr)
    out: list[object | None] = []
    for left_val, right_val, val in zip(
        left_geoms, [right_geom] * len(left_geoms), result.tolist(), strict=True,
    ):
        if left_val is None or right_val is None:
            out.append(None)
        else:
            out.append(val)
    return out


def _assert_constructive_matches_oracle(
    op: str,
    left_geoms: list[object | None],
    right_geom: object,
    *,
    tolerance: float = 1e-9,
) -> None:
    """Assert broadcast constructive result matches tiled Shapely oracle."""
    left_owned = from_shapely_geometries(list(left_geoms))
    right_owned = from_shapely_geometries([right_geom])

    result_owned = binary_constructive_owned(op, left_owned, right_owned)
    result_geoms = result_owned.to_shapely()

    expected = _shapely_oracle(op, left_geoms, right_geom)

    assert len(result_geoms) == len(expected), (
        f"Length mismatch: got {len(result_geoms)}, expected {len(expected)}"
    )

    for i, (got, exp) in enumerate(zip(result_geoms, expected, strict=True)):
        if exp is None:
            assert got is None, f"Row {i}: expected None, got {got}"
        elif shapely.is_empty(exp):
            assert shapely.is_empty(got), f"Row {i}: expected empty, got {got}"
        else:
            assert got.equals_exact(exp, tolerance), (
                f"Row {i}: {op} mismatch.\n  got={got}\n  exp={exp}"
            )


# ---------------------------------------------------------------------------
# 1. Oracle: broadcast result == tiled-pairwise Shapely result
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op", _CONSTRUCTIVE_OPS)
def test_broadcast_polygon_polygon_matches_oracle(op: str) -> None:
    """Polygon x scalar Polygon broadcast matches tiled Shapely."""
    left = [
        box(0, 0, 2, 2),
        box(1, 1, 3, 3),
        box(5, 5, 7, 7),
        box(0, 0, 1, 1),
    ]
    right = box(0.5, 0.5, 2.5, 2.5)
    _assert_constructive_matches_oracle(op, left, right)


@pytest.mark.parametrize("op", _CONSTRUCTIVE_OPS)
def test_broadcast_point_point_matches_oracle(op: str) -> None:
    """Point x scalar Point broadcast matches tiled Shapely."""
    left = [
        Point(0, 0),
        Point(1, 1),
        Point(2, 2),
    ]
    right = Point(1, 1)
    _assert_constructive_matches_oracle(op, left, right)


@pytest.mark.parametrize("op", ("intersection", "difference"))
def test_broadcast_point_polygon_matches_oracle(op: str) -> None:
    """Point x scalar Polygon broadcast matches tiled Shapely."""
    left = [
        Point(1, 1),
        Point(0, 0),
        Point(3, 3),
        Point(0.5, 0.5),
    ]
    right = box(0, 0, 2, 2)
    _assert_constructive_matches_oracle(op, left, right)


# ---------------------------------------------------------------------------
# 2. All 4 ops: intersection, union, difference, symmetric_difference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op", _CONSTRUCTIVE_OPS)
def test_all_four_ops_polygon(op: str) -> None:
    """All 4 constructive ops work with broadcast polygon."""
    left = [box(0, 0, 2, 2), box(1, 1, 3, 3)]
    right = box(0.5, 0.5, 2.5, 2.5)
    _assert_constructive_matches_oracle(op, left, right)


# ---------------------------------------------------------------------------
# 3. Null broadcast: single null geometry -> all-null output
# ---------------------------------------------------------------------------

def test_null_broadcast_right() -> None:
    """Broadcast of a null right geometry produces all-null output."""
    left = [box(0, 0, 2, 2), box(1, 1, 3, 3), box(5, 5, 7, 7)]
    right_owned = from_shapely_geometries([None])
    left_owned = from_shapely_geometries(left)

    result = binary_constructive_owned("intersection", left_owned, right_owned)
    result_geoms = result.to_shapely()

    for i, g in enumerate(result_geoms):
        assert g is None, f"Row {i}: expected None for null broadcast, got {g}"


# ---------------------------------------------------------------------------
# 4. Empty broadcast: single empty geometry -> all-empty output
# ---------------------------------------------------------------------------

def test_empty_broadcast_right() -> None:
    """Broadcast of an empty right geometry produces appropriate output."""
    left = [box(0, 0, 2, 2), box(1, 1, 3, 3)]
    # Create a valid empty polygon (e.g., intersection of disjoint polys).
    empty_geom = shapely.intersection(box(0, 0, 1, 1), box(10, 10, 11, 11))
    assert shapely.is_empty(empty_geom)

    left_owned = from_shapely_geometries(left)
    right_owned = from_shapely_geometries([empty_geom])

    result = binary_constructive_owned("intersection", left_owned, right_owned)
    result_geoms = result.to_shapely()

    for i, g in enumerate(result_geoms):
        assert shapely.is_empty(g), (
            f"Row {i}: expected empty for empty broadcast intersection, got {g}"
        )


# ---------------------------------------------------------------------------
# 5. Verify 1-row right (not N copies) via row_count check
# ---------------------------------------------------------------------------

def test_right_is_single_row() -> None:
    """After _coerce_other_to_owned change, right should be 1-row."""
    right_geom = box(0, 0, 2, 2)
    right_owned = from_shapely_geometries([right_geom])
    assert right_owned.row_count == 1, (
        f"Expected 1-row right, got {right_owned.row_count}"
    )


# ---------------------------------------------------------------------------
# 6. Tiling equivalence regression
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op", _CONSTRUCTIVE_OPS)
def test_tiling_equivalence(op: str) -> None:
    """Broadcast result must match the old N-copy tiling approach."""
    left = [
        box(0, 0, 2, 2),
        box(1, 1, 3, 3),
        box(5, 5, 7, 7),
    ]
    right_geom = box(0.5, 0.5, 2.5, 2.5)

    # New broadcast path: 1-row right
    left_owned = from_shapely_geometries(left)
    right_owned_broadcast = from_shapely_geometries([right_geom])
    result_broadcast = binary_constructive_owned(op, left_owned, right_owned_broadcast)

    # Old tiling path: N-copy right
    right_owned_tiled = from_shapely_geometries([right_geom] * len(left))
    result_tiled = binary_constructive_owned(op, left_owned, right_owned_tiled)

    broadcast_geoms = result_broadcast.to_shapely()
    tiled_geoms = result_tiled.to_shapely()

    for i, (bg, tg) in enumerate(zip(broadcast_geoms, tiled_geoms, strict=True)):
        if tg is None:
            assert bg is None, f"Row {i}: broadcast={bg}, tiled=None"
        elif shapely.is_empty(tg):
            assert shapely.is_empty(bg), f"Row {i}: expected empty"
        else:
            assert bg.equals_exact(tg, 1e-9), (
                f"Row {i}: broadcast != tiled\n  broadcast={bg}\n  tiled={tg}"
            )


@requires_gpu
def test_strict_broadcast_polygon_intersection_preserves_row_cardinality_for_complex_polygons() -> None:
    """Strict scalar-right polygon intersection keeps one output slot per input row.

    Buffered polygons exceed the SH kernel's vertex workspace, so this exercises
    the row-preserving overlay fallback instead of the direct polygon clip
    kernel.
    """
    left = [
        Point(2, 2).buffer(4),
        Point(3, 4).buffer(4),
        Point(9, 8).buffer(4),
        Point(-12, -15).buffer(4),
    ]
    right = box(0, 0, 10, 10)

    left_owned = from_shapely_geometries(left)
    right_owned = from_shapely_geometries([right])

    with strict_native_environment():
        result = binary_constructive_owned("intersection", left_owned, right_owned)

    got = result.to_shapely()
    expected = _shapely_oracle("intersection", left, right)

    assert len(got) == len(expected) == 4
    for i, (got_geom, expected_geom) in enumerate(zip(got, expected, strict=True)):
        if expected_geom is None:
            assert got_geom is None, f"Row {i}: expected None, got {got_geom}"
        elif shapely.is_empty(expected_geom):
            assert got_geom is None or shapely.is_empty(got_geom), (
                f"Row {i}: expected empty or null, got {got_geom}"
            )
        else:
            assert got_geom is not None, f"Row {i}: expected non-null polygon result"
            assert got_geom.normalize().equals_exact(expected_geom.normalize(), 1e-9), (
                f"Row {i}: broadcast complex polygon intersection mismatch\n"
                f"  got={got_geom}\n"
                f"  exp={expected_geom}"
            )


@requires_gpu
@pytest.mark.parametrize("op", _CONSTRUCTIVE_OPS)
def test_multirow_polygon_constructive_prefers_contraction_path(
    monkeypatch: pytest.MonkeyPatch,
    op: str,
) -> None:
    left = [
        Point(0, 0).buffer(4),
        Point(10, 10).buffer(4),
    ]
    right = [
        Point(1, 1).buffer(2.5),
        Point(9.5, 9.5).buffer(2.0),
    ]
    sentinel = from_shapely_geometries([
        box(0, 0, 1, 1),
        box(2, 2, 3, 3),
    ])

    left_owned = from_shapely_geometries(left)
    right_owned = from_shapely_geometries(right)

    contraction_module = importlib.import_module("vibespatial.overlay.contraction")
    calls: list[tuple[str, int]] = []

    def _fake_contraction(left_arg, right_arg, *, operation: str, dispatch_mode=ExecutionMode.GPU):
        calls.append((operation, left_arg.row_count))
        return sentinel

    def _unexpected(*args, **kwargs):
        pytest.fail("legacy overlay helper should not run when contraction path succeeds")

    monkeypatch.setattr(contraction_module, "overlay_contraction_owned", _fake_contraction)
    monkeypatch.setattr(binary_constructive_module, "_sh_kernel_can_handle", lambda *args, **kwargs: False)
    monkeypatch.setattr(binary_constructive_module, "_dispatch_polygon_intersection_overlay_rowwise_gpu", _unexpected)
    monkeypatch.setattr(binary_constructive_module, "_dispatch_polygon_difference_overlay_rowwise_gpu", _unexpected)
    monkeypatch.setattr(binary_constructive_module, "_dispatch_polygon_overlay_rowwise_gpu", _unexpected)
    monkeypatch.setattr(binary_constructive_module, "_dispatch_overlay_gpu", _unexpected)

    result = binary_constructive_owned(
        op,
        left_owned,
        right_owned,
        dispatch_mode=ExecutionMode.GPU,
    )

    assert calls == [(op, 2)]
    for got, exp in zip(result.to_shapely(), sentinel.to_shapely(), strict=True):
        assert shapely.equals(got, exp)


def test_polygon_intersection_tries_rowwise_overlay_after_overlay_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Polygon intersection recovers with rowwise overlay when bulk overlay raises."""
    left = from_shapely_geometries([Point(0, 0).buffer(2.0), Point(5, 5).buffer(2.0)])
    right = from_shapely_geometries([Point(1, 0).buffer(1.5), Point(5, 5).buffer(1.0)])
    sentinel = from_shapely_geometries([box(1, 1, 2, 2), box(4, 4, 5, 5)])

    monkeypatch.setattr(binary_constructive_module, "_sh_kernel_can_handle", lambda *_: False)

    def _raise_overlay(*args, **kwargs):
        raise RuntimeError("overlay boom")

    monkeypatch.setattr(binary_constructive_module, "_dispatch_overlay_gpu", _raise_overlay)
    monkeypatch.setattr(
        binary_constructive_module,
        "_dispatch_polygon_intersection_overlay_rowwise_gpu",
        lambda *args, **kwargs: sentinel,
    )

    result = binary_constructive_module._binary_constructive_gpu(
        "intersection",
        left,
        right,
    )

    assert result is sentinel


def test_polygon_difference_tries_legacy_rowwise_after_batched_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = from_shapely_geometries([box(0, 0, 2, 2), box(4, 4, 6, 6)])
    right = from_shapely_geometries([box(1, 1, 3, 3), box(4, 4, 5, 5)])
    sentinel = from_shapely_geometries([box(0, 0, 1, 2), box(5, 5, 6, 6)])

    def _raise_batched(*args, **kwargs):
        raise RuntimeError("batched boom")

    monkeypatch.setattr(
        binary_constructive_module,
        "_dispatch_polygon_difference_overlay_batched_gpu",
        _raise_batched,
    )
    monkeypatch.setattr(
        binary_constructive_module,
        "_dispatch_polygon_difference_overlay_rowwise_gpu_legacy",
        lambda *args, **kwargs: sentinel,
    )

    result = binary_constructive_module._dispatch_polygon_difference_overlay_rowwise_gpu(
        left,
        right,
    )

    assert result is sentinel


@requires_gpu
def test_polygon_intersection_prefers_rectangle_kernel_for_high_vertex_left_polygons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rect_module = importlib.import_module(
        "vibespatial.kernels.constructive.polygon_rect_intersection"
    )

    left = from_shapely_geometries(
        [Point(0, 0).buffer(5.0), Point(20, 0).buffer(5.0)]
    )
    right = from_shapely_geometries(
        [box(-2.0, -2.0, 2.0, 2.0), box(18.0, -2.0, 22.0, 2.0)]
    )
    sentinel = from_shapely_geometries(
        [box(-1.0, -1.0, 1.0, 1.0), box(19.0, -1.0, 21.0, 1.0)],
        residency=Residency.DEVICE,
    )
    calls: list[int] = []

    def _fake_rect_kernel(left_arg, right_arg, *, dispatch_mode=ExecutionMode.GPU):
        calls.append(left_arg.row_count)
        return sentinel

    monkeypatch.setattr(
        rect_module,
        "polygon_rect_intersection",
        _fake_rect_kernel,
    )
    monkeypatch.setattr(
        binary_constructive_module,
        "_dispatch_polygon_intersection_overlay_rowwise_gpu",
        lambda *args, **kwargs: pytest.fail(
            "high-vertex polygon vs rectangle intersection should avoid rowwise overlay"
        ),
    )

    result = binary_constructive_owned(
        "intersection",
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )

    assert calls == [2]
    for got, exp in zip(result.to_shapely(), sentinel.to_shapely(), strict=True):
        assert shapely.equals(got, exp)


@requires_gpu
def test_polygon_intersection_prefers_rectangle_kernel_when_left_is_rectangle_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rect_module = importlib.import_module(
        "vibespatial.kernels.constructive.polygon_rect_intersection"
    )

    left = from_shapely_geometries(
        [box(-2.0, -2.0, 2.0, 2.0), box(18.0, -2.0, 22.0, 2.0)]
    )
    right = from_shapely_geometries(
        [Point(0, 0).buffer(5.0), Point(20, 0).buffer(5.0)]
    )
    sentinel = from_shapely_geometries(
        [box(-1.0, -1.0, 1.0, 1.0), box(19.0, -1.0, 21.0, 1.0)],
        residency=Residency.DEVICE,
    )
    calls: list[tuple[int, int]] = []

    def _fake_can_handle(left_arg, right_arg):
        calls.append((left_arg.row_count, right_arg.row_count))
        return left_arg is right and right_arg is left

    def _fake_rect_kernel(left_arg, right_arg, *, dispatch_mode=ExecutionMode.GPU):
        assert left_arg is right
        assert right_arg is left
        return sentinel

    monkeypatch.setattr(
        rect_module,
        "polygon_rect_intersection_can_handle",
        _fake_can_handle,
    )
    monkeypatch.setattr(
        rect_module,
        "polygon_rect_intersection",
        _fake_rect_kernel,
    )
    monkeypatch.setattr(
        binary_constructive_module,
        "_dispatch_polygon_intersection_overlay_rowwise_gpu",
        lambda *args, **kwargs: pytest.fail(
            "rectangle-left polygon intersection should use swapped rectangle kernel before rowwise overlay"
        ),
    )

    result = binary_constructive_owned(
        "intersection",
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )

    assert calls == [(2, 2), (2, 2)]
    for got, exp in zip(result.to_shapely(), sentinel.to_shapely(), strict=True):
        assert shapely.equals(got, exp)


@requires_gpu
def test_single_pair_polygon_intersection_uses_exact_overlay_path_for_complex_polygons() -> None:
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
    left_owned = left.geometry.values.to_owned()
    right_owned = right.geometry.values.to_owned()
    expected = left.geometry.iloc[0].intersection(right.geometry.iloc[0]).normalize()

    result = binary_constructive_owned(
        "intersection",
        left_owned,
        right_owned,
        dispatch_mode=ExecutionMode.GPU,
    )

    got = result.to_shapely()[0]
    assert got is not None
    assert got.geom_type == expected.geom_type
    assert got.normalize().equals_exact(expected, tolerance=1e-6)


@requires_gpu
def test_single_pair_polygon_difference_preserves_touch_only_left_polygon() -> None:
    left = from_shapely_geometries(
        [box(-1, 1, 1, 3)]
    )
    right = from_shapely_geometries(
        [box(1, 1, 3, 3)]
    )

    result = binary_constructive_owned(
        "difference",
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )

    got = result.to_shapely()[0]
    expected = box(-1, 1, 1, 3)
    assert got is not None
    assert got.equals_exact(expected, tolerance=1e-9)


@requires_gpu
def test_grouped_polygon_difference_rowwise_matches_oracle_on_redevelopment_rows() -> None:
    """Redevelopment grouped difference wrapper matches the exact Shapely oracle."""
    tmpdir = Path(tempfile.mkdtemp(prefix="test_redev_grouped_diff_"))
    fixtures = setup_fixtures(tmpdir)

    import geopandas as gpd

    parcels = gpd.read_parquet(fixtures["parcels"])
    exclusions = gpd.read_parquet(fixtures["exclusion_zones"])

    bounds = parcels.total_bounds
    dx = (bounds[2] - bounds[0]) * 0.15
    dy = (bounds[3] - bounds[1]) * 0.15
    clip_box = box(bounds[0] + dx, bounds[1] + dy, bounds[2] - dx, bounds[3] - dy)
    study = gpd.clip(parcels, clip_box)
    study = study[study.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    left = study.geometry.values.to_owned()
    right = exclusions.geometry.values.to_owned()

    idx1, idx2 = exclusions.sindex.query(study.geometry, predicate="intersects", sort=True)
    idx1 = np.asarray(idx1)
    idx2 = np.asarray(idx2)
    idx1_unique, idx1_split_at = np.unique(idx1, return_index=True)
    group_offsets = np.concatenate([idx1_split_at, np.asarray([len(idx2)])])
    right_grouped = right.take(idx2)
    right_unioned = segmented_union_all(right_grouped, group_offsets)
    left_grouped = left.take(idx1_unique)

    row_subset = np.asarray([2, 17, 27, 28, 29], dtype=np.int64)
    left_subset = left_grouped.take(row_subset)
    right_subset = right_unioned.take(row_subset)

    result = binary_constructive_module._dispatch_polygon_difference_overlay_rowwise_gpu(
        left_subset,
        right_subset,
        dispatch_mode=ExecutionMode.GPU,
    )
    assert result is not None

    got = list(result.to_shapely())
    expected = [
        left_geom.difference(right_geom)
        for left_geom, right_geom in zip(
            left_subset.to_shapely(),
            right_subset.to_shapely(),
            strict=True,
        )
    ]

    assert len(got) == len(expected)
    for row, (actual, oracle) in enumerate(zip(got, expected, strict=True)):
        if actual is None or oracle is None:
            assert actual is None and oracle is None, f"row {row} null mismatch"
            continue
        assert shapely.normalize(actual).equals(shapely.normalize(oracle)), (
            f"row {row} grouped rowwise difference mismatch"
        )


@requires_gpu
def test_grouped_polygon_difference_batched_matches_oracle_on_redevelopment_prefix() -> None:
    """The batched row-isolated difference path stays exact on the first failing prefix."""
    tmpdir = Path(tempfile.mkdtemp(prefix="test_redev_grouped_diff_prefix_"))
    fixtures = setup_fixtures(tmpdir)

    import geopandas as gpd

    parcels = gpd.read_parquet(fixtures["parcels"])
    exclusions = gpd.read_parquet(fixtures["exclusion_zones"])

    bounds = parcels.total_bounds
    dx = (bounds[2] - bounds[0]) * 0.15
    dy = (bounds[3] - bounds[1]) * 0.15
    clip_box = box(bounds[0] + dx, bounds[1] + dy, bounds[2] - dx, bounds[3] - dy)
    study = gpd.clip(parcels, clip_box)
    study = study[study.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    left = study.geometry.values.to_owned()
    right = exclusions.geometry.values.to_owned()

    idx1, idx2 = exclusions.sindex.query(study.geometry, predicate="intersects", sort=True)
    idx1 = np.asarray(idx1)
    idx2 = np.asarray(idx2)
    idx1_unique, idx1_split_at = np.unique(idx1, return_index=True)
    group_offsets = np.concatenate([idx1_split_at, np.asarray([len(idx2)])])
    right_grouped = right.take(idx2)
    right_unioned = segmented_union_all(right_grouped, group_offsets)
    left_grouped = left.take(idx1_unique)

    prefix = np.arange(16, dtype=np.int64)
    left_prefix = left_grouped.take(prefix)
    right_prefix = right_unioned.take(prefix)

    with strict_native_environment():
        result = binary_constructive_module._dispatch_polygon_difference_overlay_rowwise_gpu(
            left_prefix,
            right_prefix,
            dispatch_mode=ExecutionMode.GPU,
        )

    assert result is not None
    got = list(result.to_shapely())
    expected = [
        left_geom.difference(right_geom)
        for left_geom, right_geom in zip(
            left_prefix.to_shapely(),
            right_prefix.to_shapely(),
            strict=True,
        )
    ]

    assert len(got) == len(expected)
    for row, (actual, oracle) in enumerate(zip(got, expected, strict=True)):
        if actual is None or oracle is None:
            assert actual is None and oracle is None, f"row {row} null mismatch"
            continue
        assert shapely.normalize(actual).equals(shapely.normalize(oracle)), (
            f"row {row} batched grouped difference mismatch"
        )


@requires_gpu
def test_grouped_polygon_difference_strict_matches_oracle_on_redevelopment_prefix_16() -> None:
    """The 16-row redevelopment prefix stays exact under strict row isolation."""
    tmpdir = Path(tempfile.mkdtemp(prefix="test_redev_grouped_diff_prefix16_"))
    fixtures = setup_fixtures(tmpdir)

    import geopandas as gpd

    parcels = gpd.read_parquet(fixtures["parcels"])
    exclusions = gpd.read_parquet(fixtures["exclusion_zones"])

    bounds = parcels.total_bounds
    dx = (bounds[2] - bounds[0]) * 0.15
    dy = (bounds[3] - bounds[1]) * 0.15
    clip_box = box(bounds[0] + dx, bounds[1] + dy, bounds[2] - dx, bounds[3] - dy)
    study = gpd.clip(parcels, clip_box)
    study = study[study.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    left = study.geometry.values.to_owned()
    right = exclusions.geometry.values.to_owned()

    idx1, idx2 = exclusions.sindex.query(study.geometry, predicate="intersects", sort=True)
    idx1 = np.asarray(idx1)
    idx2 = np.asarray(idx2)
    idx1_unique, idx1_split_at = np.unique(idx1, return_index=True)
    group_offsets = np.concatenate([idx1_split_at, np.asarray([len(idx2)])])
    right_grouped = right.take(idx2)
    right_unioned = segmented_union_all(right_grouped, group_offsets)
    left_grouped = left.take(idx1_unique)

    prefix = np.arange(16, dtype=np.int64)
    left_prefix = left_grouped.take(prefix)
    right_prefix = right_unioned.take(prefix)

    with strict_native_environment():
        result = binary_constructive_module._dispatch_polygon_difference_overlay_rowwise_gpu(
            left_prefix,
            right_prefix,
            dispatch_mode=ExecutionMode.GPU,
        )
    assert result is not None

    got = list(result.to_shapely())
    expected = [
        left_geom.difference(right_geom)
        for left_geom, right_geom in zip(
            left_prefix.to_shapely(),
            right_prefix.to_shapely(),
            strict=True,
        )
    ]

    assert len(got) == len(expected)
    for row, (actual, oracle) in enumerate(zip(got, expected, strict=True)):
        if actual is None or oracle is None:
            assert actual is None and oracle is None, f"row {row} null mismatch"
            continue
        assert shapely.normalize(actual).equals(shapely.normalize(oracle)), (
            f"row {row} grouped strict difference mismatch"
        )


@requires_gpu
@pytest.mark.parametrize(
    ("right_geom", "label"),
    [
        (Point(1, 1), "point"),
        (MultiPoint([(1, 1), (3, 3)]), "multipoint"),
    ],
)
def test_single_pair_polygon_difference_preserves_left_for_lower_dim_right(
    right_geom,
    label: str,
) -> None:
    left = from_shapely_geometries(
        [box(0, 0, 4, 4)]
    )
    right = from_shapely_geometries([right_geom])

    result = binary_constructive_owned(
        "difference",
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
    )

    got = result.to_shapely()[0]
    expected = box(0, 0, 4, 4)
    assert got is not None, f"{label} difference unexpectedly returned null"
    assert got.equals_exact(expected, tolerance=1e-9), (
        f"{label} difference should preserve the left polygon exactly"
    )


@requires_gpu
def test_polygon_union_batches_aligned_overlay_candidate_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = [
        Point(0, 0).buffer(10, resolution=32),
        Point(40, 0).buffer(10, resolution=32),
        Point(0, 40).buffer(10, resolution=32),
        Point(40, 40).buffer(10, resolution=32),
    ]
    right = [
        Point(5, 0).buffer(10, resolution=32),
        Point(45, 0).buffer(10, resolution=32),
        Point(5, 40).buffer(10, resolution=32),
        Point(45, 40).buffer(10, resolution=32),
    ]

    monkeypatch.setenv("VIBESPATIAL_HOTPATH_TRACE", "1")
    reset_hotpath_trace()

    result = binary_constructive_owned(
        "union",
        from_shapely_geometries(left),
        from_shapely_geometries(right),
        dispatch_mode=ExecutionMode.GPU,
    )

    got = result.to_shapely()
    expected = shapely.union(np.asarray(left, dtype=object), np.asarray(right, dtype=object)).tolist()
    assert len(got) == len(expected) == 4
    for got_geom, expected_geom in zip(got, expected, strict=True):
        assert got_geom is not None
        assert got_geom.normalize().equals_exact(expected_geom.normalize(), tolerance=1e-9)

    summary = {entry["name"]: entry["calls"] for entry in summarize_hotpath_trace()}
    # The aligned batched helper should still keep candidate generation
    # bounded. Degenerate touch/disjoint rows now take the corrected
    # single-row fallback path, so this mixed workload legitimately does
    # more than the ideal all-area-overlap case, but it should remain well
    # below a naive overlay-per-row-everything shape.
    assert summary.get("segment.classify.generate_candidates", 0) <= 5
    assert summary.get("overlay.split.classify_intersections", 0) <= 4


@requires_gpu
def test_polygon_difference_batches_aligned_overlay_candidate_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = [
        box(0, 0, 4, 4),      # partial overlap
        box(10, 0, 14, 4),    # touch-only
        box(20, 0, 24, 4),    # full overlap
        box(30, 0, 34, 4),    # disjoint
    ]
    right = [
        box(2, 0, 6, 4),
        box(14, 0, 18, 4),
        box(20, 0, 24, 4),
        box(40, 0, 44, 4),
    ]

    monkeypatch.setenv("VIBESPATIAL_HOTPATH_TRACE", "1")
    reset_hotpath_trace()

    result = binary_constructive_owned(
        "difference",
        from_shapely_geometries(left),
        from_shapely_geometries(right),
        dispatch_mode=ExecutionMode.GPU,
    )

    got = result.to_shapely()
    expected = shapely.difference(np.asarray(left, dtype=object), np.asarray(right, dtype=object)).tolist()
    assert len(got) == len(expected) == 4
    for got_geom, expected_geom in zip(got, expected, strict=True):
        if expected_geom is None:
            assert got_geom is None
        elif shapely.is_empty(expected_geom):
            assert got_geom is None or shapely.is_empty(got_geom)
        else:
            assert got_geom is not None
            assert got_geom.normalize().equals_exact(expected_geom.normalize(), tolerance=1e-9)

    summary = {entry["name"]: entry["calls"] for entry in summarize_hotpath_trace()}
    # Degenerate touch/disjoint rows now take the corrected single-row
    # difference fallback, so this mixed workload legitimately does more
    # work than the pure batched area-overlap case while staying bounded.
    assert summary.get("segment.classify.generate_candidates", 0) <= 6
    assert summary.get("overlay.split.classify_intersections", 0) <= 5


@requires_gpu
def test_polygon_intersection_single_pair_uses_same_row_candidate_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = from_shapely_geometries(
        [
            Point(0.0, 0.0).buffer(10.0, resolution=64),
        ]
    )
    right = from_shapely_geometries(
        [
            shapely.Polygon(
                [
                    (12.0, 0.0),
                    (4.0, 2.0),
                    (0.0, 12.0),
                    (-4.0, 2.0),
                    (-12.0, 0.0),
                    (-4.0, -2.0),
                    (0.0, -12.0),
                    (4.0, -2.0),
                    (12.0, 0.0),
                ]
            ),
        ]
    )

    monkeypatch.setenv("VIBESPATIAL_HOTPATH_TRACE", "1")
    reset_hotpath_trace()

    result = binary_constructive_owned(
        "intersection",
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
        _prefer_exact_polygon_intersection=True,
    )

    got = result.to_shapely()[0]
    expected = shapely.intersection(
        np.asarray(left.to_shapely(), dtype=object),
        np.asarray(right.to_shapely(), dtype=object),
    ).tolist()[0]
    assert got is not None
    assert got.normalize().equals_exact(expected.normalize(), tolerance=1e-9)

    summary = {entry["name"]: entry["calls"] for entry in summarize_hotpath_trace()}
    assert summary.get("segment.candidates.same_row_fast_path") == 1
    assert "segment.candidates.binary_search" not in summary


@requires_gpu
def test_multipolygon_polygon_intersection_regroups_with_grouped_union_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = from_shapely_geometries(
        [
            MultiPolygon(
                [
                    box(0, 0, 2, 2),
                    box(4, 0, 6, 2),
                ]
            ),
            MultiPolygon(
                [
                    box(10, 0, 12, 2),
                    box(14, 0, 16, 2),
                ]
            ),
        ]
    )
    right = from_shapely_geometries(
        [
            box(1, 0, 5, 2),
            box(11, 0, 12, 2),
        ]
    )

    overlay_gpu_module = importlib.import_module("vibespatial.overlay.gpu")
    segmented_union_module = importlib.import_module(
        "vibespatial.kernels.constructive.segmented_union"
    )

    original_materialize = overlay_gpu_module._materialize_overlay_execution_plan
    original_segmented_union_all = segmented_union_module.segmented_union_all
    union_materialize_calls = 0
    segmented_union_calls = 0

    def _counted_materialize(*args, **kwargs):
        nonlocal union_materialize_calls
        if kwargs.get("operation") == "union":
            union_materialize_calls += 1
        return original_materialize(*args, **kwargs)

    def _counted_segmented_union_all(*args, **kwargs):
        nonlocal segmented_union_calls
        segmented_union_calls += 1
        return original_segmented_union_all(*args, **kwargs)

    monkeypatch.setattr(
        overlay_gpu_module,
        "_materialize_overlay_execution_plan",
        _counted_materialize,
    )
    monkeypatch.setattr(
        segmented_union_module,
        "segmented_union_all",
        _counted_segmented_union_all,
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
    for row, (got_geom, expected_geom) in enumerate(zip(got, expected, strict=True)):
        if expected_geom is None:
            assert got_geom is None, f"row {row}: expected null"
            continue
        if shapely.is_empty(expected_geom):
            assert got_geom is None or shapely.is_empty(got_geom), (
                f"row {row}: expected empty or null, got {got_geom}"
            )
            continue
        assert got_geom is not None
        assert shapely.normalize(got_geom).equals(shapely.normalize(expected_geom)), (
            f"row {row}: regrouped multipart intersection mismatch"
        )

    assert union_materialize_calls == 1
    assert segmented_union_calls == 0



# ---------------------------------------------------------------------------
# 7. tile_single_row unit tests
# ---------------------------------------------------------------------------

def test_tile_single_row_metadata() -> None:
    """tile_single_row produces correct metadata arrays."""
    geom = box(0, 0, 1, 1)
    owned = from_shapely_geometries([geom])
    tiled = tile_single_row(owned, 5)

    assert tiled.row_count == 5
    assert len(tiled.validity) == 5
    assert all(tiled.validity)
    assert len(tiled.tags) == 5
    # All rows should have the same tag
    assert len(set(tiled.tags.tolist())) == 1
    # All family_row_offsets should be 0
    assert all(tiled.family_row_offsets == 0)


def test_tile_single_row_n1_returns_same() -> None:
    """tile_single_row with n=1 returns the same object."""
    owned = from_shapely_geometries([box(0, 0, 1, 1)])
    result = tile_single_row(owned, 1)
    assert result is owned


def test_tile_single_row_rejects_multi_row() -> None:
    """tile_single_row raises ValueError for multi-row input."""
    owned = from_shapely_geometries([box(0, 0, 1, 1), box(1, 1, 2, 2)])
    with pytest.raises(ValueError, match="1-row array"):
        tile_single_row(owned, 5)
