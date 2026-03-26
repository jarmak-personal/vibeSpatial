"""Tests for broadcast-right support in binary constructive operations.

Validates that scalar (1-row) right operands produce correct results for
intersection, union, difference, and symmetric_difference.  Oracle: Shapely
with tiled-pairwise comparison.

Covers nsf.3: elimination of [other]*len(self) materialization.
"""
from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import Point, box

from vibespatial.constructive.binary_constructive import binary_constructive_owned
from vibespatial.geometry.owned import (
    from_shapely_geometries,
    tile_single_row,
)

# The four constructive operations.
_CONSTRUCTIVE_OPS = ("intersection", "union", "difference", "symmetric_difference")


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
