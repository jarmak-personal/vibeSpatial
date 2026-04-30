"""Tests for GPU shared_paths kernel.

Verifies the NVRTC shared_paths kernel produces results matching
Shapely's shared_paths for all supported linear geometry family
combinations (LineString x LineString, MultiLineString x LineString,
MultiLineString x MultiLineString).
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import shapely
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
)

from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.runtime import has_gpu_runtime

requires_gpu = pytest.mark.skipif(
    not has_gpu_runtime(), reason="GPU not available",
)


def test_constructive_helper_d2h_exports_are_runtime_accounted() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    files = (
        "normalize.py",
        "polygon.py",
        "shared_paths.py",
        "multipoint_polygon_constructive.py",
        "linear_ref.py",
        "snap.py",
    )
    paths = tuple(repo_root / "src" / "vibespatial" / "constructive" / name for name in files)
    offenders: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr == "asnumpy":
                offenders.append(f"{path.relative_to(repo_root)}:{node.lineno}")
            if func.attr == "copy_device_to_host" and not any(
                keyword.arg == "reason" for keyword in node.keywords
            ):
                offenders.append(f"{path.relative_to(repo_root)}:{node.lineno}")
    assert offenders == []


def _assert_shared_paths_match(gpu_results, left_geoms, right_geoms, atol=1e-10):
    """Assert GPU shared_paths matches Shapely for each geometry pair."""
    for i, (lg, rg) in enumerate(zip(left_geoms, right_geoms)):
        if lg is None or rg is None:
            continue

        expected = shapely.shared_paths(lg, rg)
        if expected is None:
            continue

        actual = gpu_results[i]
        assert actual is not None, f"Row {i}: GPU returned None"

        # Compare forward paths
        expected_fwd = expected.geoms[0]  # MultiLineString
        actual_fwd = actual.geoms[0]

        # Compare backward paths
        expected_bwd = expected.geoms[1]  # MultiLineString
        actual_bwd = actual.geoms[1]

        _compare_multilinestrings(actual_fwd, expected_fwd, i, "forward", atol)
        _compare_multilinestrings(actual_bwd, expected_bwd, i, "backward", atol)


def _compare_multilinestrings(actual, expected, row, direction, atol):
    """Compare two MultiLineStrings, allowing for segment reordering."""
    if expected.is_empty:
        assert actual.is_empty, (
            f"Row {row} {direction}: expected empty but got {actual.wkt}"
        )
        return

    if actual.is_empty:
        pytest.fail(
            f"Row {row} {direction}: expected {expected.wkt} but got empty"
        )

    # Extract segments from both and compare as sets (order may differ)
    expected_segs = _extract_segments_sorted(expected)
    actual_segs = _extract_segments_sorted(actual)

    assert len(actual_segs) == len(expected_segs), (
        f"Row {row} {direction}: "
        f"expected {len(expected_segs)} segments but got {len(actual_segs)}. "
        f"Expected: {expected.wkt}, Got: {actual.wkt}"
    )

    for j, (a_seg, e_seg) in enumerate(zip(actual_segs, expected_segs)):
        np.testing.assert_allclose(
            a_seg, e_seg, atol=atol,
            err_msg=(
                f"Row {row} {direction} segment {j}: "
                f"GPU={a_seg}, Shapely={e_seg}"
            ),
        )


def _extract_segments_sorted(mls):
    """Extract all segment coordinate arrays from a MultiLineString, sorted."""
    segs = []
    for line in mls.geoms:
        coords = np.array(line.coords)
        segs.append(coords)
    # Sort by first coordinate then second
    segs.sort(key=lambda s: (s[0, 0], s[0, 1], s[-1, 0], s[-1, 1]))
    return segs


# ---------------------------------------------------------------------------
# LineString x LineString
# ---------------------------------------------------------------------------

@requires_gpu
def test_shared_paths_ls_ls_forward():
    """GPU shared_paths: LS x LS with forward shared segment."""
    left_geoms = [LineString([(0, 0), (1, 1), (2, 0)])]
    right_geoms = [LineString([(0, 0), (1, 1), (0, 2)])]

    left = from_shapely_geometries(left_geoms)
    right = from_shapely_geometries(right_geoms)

    from vibespatial.constructive.shared_paths import shared_paths_owned
    result = shared_paths_owned(left, right, dispatch_mode="gpu")

    assert len(result) == 1
    _assert_shared_paths_match(result, left_geoms, right_geoms)


@requires_gpu
def test_shared_paths_ls_ls_backward():
    """GPU shared_paths: LS x LS with backward (opposite direction) shared segment."""
    left_geoms = [LineString([(0, 0), (1, 0)])]
    right_geoms = [LineString([(1, 0), (0, 0)])]

    left = from_shapely_geometries(left_geoms)
    right = from_shapely_geometries(right_geoms)

    from vibespatial.constructive.shared_paths import shared_paths_owned
    result = shared_paths_owned(left, right, dispatch_mode="gpu")

    assert len(result) == 1
    _assert_shared_paths_match(result, left_geoms, right_geoms)


@requires_gpu
def test_shared_paths_ls_ls_no_shared():
    """GPU shared_paths: LS x LS with no shared segments."""
    left_geoms = [LineString([(0, 0), (1, 1)])]
    right_geoms = [LineString([(5, 5), (6, 6)])]

    left = from_shapely_geometries(left_geoms)
    right = from_shapely_geometries(right_geoms)

    from vibespatial.constructive.shared_paths import shared_paths_owned
    result = shared_paths_owned(left, right, dispatch_mode="gpu")

    assert len(result) == 1
    gc = result[0]
    assert gc.geoms[0].is_empty  # no forward
    assert gc.geoms[1].is_empty  # no backward


@requires_gpu
def test_shared_paths_ls_ls_partial_overlap():
    """GPU shared_paths: LS x LS with partial overlap along a segment."""
    left_geoms = [LineString([(0, 0), (2, 0)])]
    right_geoms = [LineString([(1, 0), (3, 0)])]

    left = from_shapely_geometries(left_geoms)
    right = from_shapely_geometries(right_geoms)

    from vibespatial.constructive.shared_paths import shared_paths_owned
    result = shared_paths_owned(left, right, dispatch_mode="gpu")

    assert len(result) == 1
    _assert_shared_paths_match(result, left_geoms, right_geoms)


@requires_gpu
def test_shared_paths_ls_ls_multiple_segments():
    """GPU shared_paths: LS x LS with multiple shared segments."""
    left_geoms = [LineString([(0, 0), (1, 0), (2, 0), (3, 0)])]
    right_geoms = [LineString([(0, 0), (1, 0), (1.5, 1), (2, 0), (3, 0)])]

    left = from_shapely_geometries(left_geoms)
    right = from_shapely_geometries(right_geoms)

    from vibespatial.constructive.shared_paths import shared_paths_owned
    result = shared_paths_owned(left, right, dispatch_mode="gpu")

    assert len(result) == 1
    _assert_shared_paths_match(result, left_geoms, right_geoms)


@requires_gpu
def test_shared_paths_ls_ls_multiple_rows():
    """GPU shared_paths: multiple rows of LS x LS."""
    left_geoms = [
        LineString([(0, 0), (1, 1), (2, 0)]),
        LineString([(0, 0), (1, 0)]),
        LineString([(5, 5), (6, 6)]),
    ]
    right_geoms = [
        LineString([(0, 0), (1, 1), (0, 2)]),
        LineString([(1, 0), (0, 0)]),
        LineString([(10, 10), (11, 11)]),
    ]

    left = from_shapely_geometries(left_geoms)
    right = from_shapely_geometries(right_geoms)

    from vibespatial.constructive.shared_paths import shared_paths_owned
    result = shared_paths_owned(left, right, dispatch_mode="gpu")

    assert len(result) == 3
    _assert_shared_paths_match(result, left_geoms, right_geoms)


# ---------------------------------------------------------------------------
# MultiLineString x LineString
# ---------------------------------------------------------------------------

@requires_gpu
def test_shared_paths_mls_ls():
    """GPU shared_paths: MultiLineString x LineString."""
    left_geoms = [
        MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]]),
    ]
    right_geoms = [
        LineString([(0, 0), (1, 1), (2, 2), (3, 3)]),
    ]

    left = from_shapely_geometries(left_geoms)
    right = from_shapely_geometries(right_geoms)

    from vibespatial.constructive.shared_paths import shared_paths_owned
    result = shared_paths_owned(left, right, dispatch_mode="gpu")

    assert len(result) == 1
    _assert_shared_paths_match(result, left_geoms, right_geoms)


@requires_gpu
def test_shared_paths_ls_mls():
    """GPU shared_paths: LineString x MultiLineString (reversed)."""
    left_geoms = [
        LineString([(0, 0), (1, 1), (2, 2), (3, 3)]),
    ]
    right_geoms = [
        MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]]),
    ]

    left = from_shapely_geometries(left_geoms)
    right = from_shapely_geometries(right_geoms)

    from vibespatial.constructive.shared_paths import shared_paths_owned
    result = shared_paths_owned(left, right, dispatch_mode="gpu")

    assert len(result) == 1
    _assert_shared_paths_match(result, left_geoms, right_geoms)


# ---------------------------------------------------------------------------
# MultiLineString x MultiLineString
# ---------------------------------------------------------------------------

@requires_gpu
def test_shared_paths_mls_mls():
    """GPU shared_paths: MultiLineString x MultiLineString."""
    left_geoms = [
        MultiLineString([[(0, 0), (1, 0)], [(2, 0), (3, 0)]]),
    ]
    right_geoms = [
        MultiLineString([[(0, 0), (1, 0)], [(2, 0), (3, 0)]]),
    ]

    left = from_shapely_geometries(left_geoms)
    right = from_shapely_geometries(right_geoms)

    from vibespatial.constructive.shared_paths import shared_paths_owned
    result = shared_paths_owned(left, right, dispatch_mode="gpu")

    assert len(result) == 1
    _assert_shared_paths_match(result, left_geoms, right_geoms)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@requires_gpu
def test_shared_paths_null_geometry():
    """GPU shared_paths: null geometry produces empty result."""
    left_geoms = [LineString([(0, 0), (1, 1)]), None]
    right_geoms = [LineString([(0, 0), (1, 1)]), LineString([(0, 0), (1, 0)])]

    left = from_shapely_geometries(left_geoms)
    right = from_shapely_geometries(right_geoms)

    from vibespatial.constructive.shared_paths import shared_paths_owned
    result = shared_paths_owned(left, right, dispatch_mode="gpu")

    assert len(result) == 2
    # Row 0: should have forward shared segment
    gc0 = result[0]
    assert not gc0.geoms[0].is_empty  # forward
    # Row 1: null left -> empty result
    gc1 = result[1]
    assert gc1.geoms[0].is_empty
    assert gc1.geoms[1].is_empty


@requires_gpu
def test_shared_paths_identical_lines():
    """GPU shared_paths: identical lines produce forward shared segment."""
    left_geoms = [LineString([(0, 0), (1, 1)])]
    right_geoms = [LineString([(0, 0), (1, 1)])]

    left = from_shapely_geometries(left_geoms)
    right = from_shapely_geometries(right_geoms)

    from vibespatial.constructive.shared_paths import shared_paths_owned
    result = shared_paths_owned(left, right, dispatch_mode="gpu")

    assert len(result) == 1
    _assert_shared_paths_match(result, left_geoms, right_geoms)


@requires_gpu
def test_shared_paths_broadcast_right():
    """GPU shared_paths: N vs 1 broadcast mode."""
    left_geoms = [
        LineString([(0, 0), (1, 1)]),
        LineString([(1, 1), (0, 0)]),
        LineString([(5, 5), (6, 6)]),
    ]
    right_geoms = [LineString([(0, 0), (1, 1)])]

    left = from_shapely_geometries(left_geoms)
    right = from_shapely_geometries(right_geoms)

    from vibespatial.constructive.shared_paths import shared_paths_owned
    result = shared_paths_owned(left, right, dispatch_mode="gpu")

    assert len(result) == 3
    # Manually verify: row 0 forward, row 1 backward, row 2 empty
    # Use Shapely as oracle after broadcasting
    from shapely.geometry import LineString as SLineString
    right_bc = SLineString([(0, 0), (1, 1)])
    for i, lg in enumerate(left_geoms):
        expected = shapely.shared_paths(lg, right_bc)
        actual = result[i]
        _compare_multilinestrings(actual.geoms[0], expected.geoms[0], i, "forward", 1e-10)
        _compare_multilinestrings(actual.geoms[1], expected.geoms[1], i, "backward", 1e-10)


@requires_gpu
def test_shared_paths_mixed_forward_backward():
    """GPU shared_paths: geometry with both forward and backward segments."""
    # A has segments (0,0)->(1,0) and (1,0)->(2,0)
    # B has (1,0)->(0,0) then goes to (0,1) then (2,0)->(1,0)
    # Both segments of A overlap with B but in reverse direction
    left_geoms = [LineString([(0, 0), (1, 0), (2, 0)])]
    right_geoms = [LineString([(1, 0), (0, 0), (0, 1), (2, 0), (1, 0)])]

    left = from_shapely_geometries(left_geoms)
    right = from_shapely_geometries(right_geoms)

    from vibespatial.constructive.shared_paths import shared_paths_owned
    result = shared_paths_owned(left, right, dispatch_mode="gpu")

    assert len(result) == 1
    _assert_shared_paths_match(result, left_geoms, right_geoms)


@requires_gpu
def test_shared_paths_perpendicular_no_overlap():
    """GPU shared_paths: perpendicular lines share no segments."""
    left_geoms = [LineString([(0, 0), (1, 0)])]
    right_geoms = [LineString([(0.5, -1), (0.5, 1)])]

    left = from_shapely_geometries(left_geoms)
    right = from_shapely_geometries(right_geoms)

    from vibespatial.constructive.shared_paths import shared_paths_owned
    result = shared_paths_owned(left, right, dispatch_mode="gpu")

    assert len(result) == 1
    gc = result[0]
    assert gc.geoms[0].is_empty
    assert gc.geoms[1].is_empty


# ---------------------------------------------------------------------------
# CPU fallback
# ---------------------------------------------------------------------------

def test_shared_paths_cpu_fallback():
    """CPU fallback produces correct results."""
    from vibespatial.constructive.shared_paths import shared_paths_owned

    left_geoms = [LineString([(0, 0), (1, 1)])]
    right_geoms = [LineString([(0, 0), (1, 1)])]

    left = from_shapely_geometries(left_geoms)
    right = from_shapely_geometries(right_geoms)

    result = shared_paths_owned(left, right, dispatch_mode="cpu")
    assert len(result) == 1
    gc = result[0]
    assert not gc.geoms[0].is_empty  # forward
    assert gc.geoms[1].is_empty  # no backward


# ---------------------------------------------------------------------------
# Integration with GeometryArray
# ---------------------------------------------------------------------------

@requires_gpu
def test_shared_paths_geometry_array_api():
    """Test shared_paths through the GeometryArray API."""
    from shapely.geometry import LineString

    import geopandas as gpd

    s1 = gpd.GeoSeries([
        LineString([(0, 0), (1, 1)]),
        LineString([(0, 0), (1, 0)]),
    ])
    s2 = gpd.GeoSeries([
        LineString([(0, 0), (1, 1)]),
        LineString([(1, 0), (0, 0)]),
    ])

    result = s1.shared_paths(s2)
    assert len(result) == 2

    # Row 0: forward
    gc0 = result.iloc[0]
    assert isinstance(gc0, GeometryCollection)

    # Row 1: backward
    gc1 = result.iloc[1]
    assert isinstance(gc1, GeometryCollection)
