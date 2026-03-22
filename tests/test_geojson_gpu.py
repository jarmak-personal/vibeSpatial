"""Tests for GPU byte-classification GeoJSON parser."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.io.geojson import (
    plan_geojson_ingest,
    read_geojson_owned,
)
from vibespatial.runtime.residency import Residency

try:
    import cupy as cp

    HAS_GPU = True
except (ImportError, ModuleNotFoundError):
    HAS_GPU = False

needs_gpu = pytest.mark.skipif(not HAS_GPU, reason="GPU not available")


def _make_polygon_feature(coords: list[list[list[float]]], properties: dict | None = None) -> dict:
    return {
        "type": "Feature",
        "properties": properties or {},
        "geometry": {
            "type": "Polygon",
            "coordinates": coords,
        },
    }


def _make_feature_collection(features: list[dict]) -> dict:
    return {
        "type": "FeatureCollection",
        "features": features,
    }


def _write_geojson(path: Path, fc: dict) -> None:
    path.write_text(json.dumps(fc), encoding="utf-8")


def _simple_square(x0: float, y0: float, size: float = 0.01) -> list[list[list[float]]]:
    return [
        [
            [x0, y0],
            [x0 + size, y0],
            [x0 + size, y0 + size],
            [x0, y0 + size],
            [x0, y0],
        ]
    ]


# ---------------------------------------------------------------------------
# Test 1: Correctness — compare GPU vs fast-json baseline
# ---------------------------------------------------------------------------
@needs_gpu
def test_correctness_against_fast_json(tmp_path):
    features = []
    for i in range(100):
        x0 = -80.0 + i * 0.01
        y0 = 27.0 + i * 0.01
        props = {"id": i, "name": f"feature_{i}"}
        features.append(_make_polygon_feature(_simple_square(x0, y0), props))
    fc = _make_feature_collection(features)
    path = tmp_path / "test_correctness.geojson"
    _write_geojson(path, fc)

    # GPU path
    gpu_batch = read_geojson_owned(path, prefer="gpu-byte-classify")
    # CPU baseline
    cpu_batch = read_geojson_owned(path, prefer="fast-json")

    # Compare coordinate arrays
    gpu_owned = gpu_batch.geometry
    cpu_owned = cpu_batch.geometry

    assert GeometryFamily.POLYGON in gpu_owned.families
    assert GeometryFamily.POLYGON in cpu_owned.families

    gpu_buf = gpu_owned.families[GeometryFamily.POLYGON]
    cpu_buf = cpu_owned.families[GeometryFamily.POLYGON]

    # Row counts should match
    assert gpu_buf.row_count == cpu_buf.row_count == 100

    # Compare coordinates (GPU may need materialization)
    if gpu_owned.device_state is not None:
        dev_buf = gpu_owned.device_state.families[GeometryFamily.POLYGON]
        gpu_x = cp.asnumpy(dev_buf.x)
        gpu_y = cp.asnumpy(dev_buf.y)
    else:
        gpu_x = gpu_buf.x
        gpu_y = gpu_buf.y

    np.testing.assert_allclose(gpu_x, cpu_buf.x, atol=1e-10)
    np.testing.assert_allclose(gpu_y, cpu_buf.y, atol=1e-10)


# ---------------------------------------------------------------------------
# Test 2: Quote-state — property containing "coordinates": as text
# ---------------------------------------------------------------------------
@needs_gpu
def test_quote_state_no_false_match(tmp_path):
    features = [
        _make_polygon_feature(
            _simple_square(0.0, 0.0),
            {"description": 'The key "coordinates": is in this string'},
        ),
        _make_polygon_feature(_simple_square(1.0, 1.0)),
    ]
    fc = _make_feature_collection(features)
    path = tmp_path / "test_quotes.geojson"
    _write_geojson(path, fc)

    batch = read_geojson_owned(path, prefer="gpu-byte-classify")
    owned = batch.geometry
    assert GeometryFamily.POLYGON in owned.families
    assert owned.families[GeometryFamily.POLYGON].row_count == 2


# ---------------------------------------------------------------------------
# Test 3: Escape handling — \" in property strings
# ---------------------------------------------------------------------------
@needs_gpu
def test_escape_handling(tmp_path):
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"label": 'He said \\"hello\\"'},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
            }
        ],
    }
    path = tmp_path / "test_escape.geojson"
    path.write_text(json.dumps(fc), encoding="utf-8")

    batch = read_geojson_owned(path, prefer="gpu-byte-classify")
    assert batch.geometry.families[GeometryFamily.POLYGON].row_count == 1


# ---------------------------------------------------------------------------
# Test 4: Empty geometry — "coordinates": []
# ---------------------------------------------------------------------------
@needs_gpu
def test_empty_geometry(tmp_path):
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": 1},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [],
                },
            },
            _make_polygon_feature(_simple_square(0, 0), {"id": 2}),
        ],
    }
    path = tmp_path / "test_empty.geojson"
    _write_geojson(path, fc)

    batch = read_geojson_owned(path, prefer="gpu-byte-classify")
    owned = batch.geometry
    # Should have 2 features total
    assert len(owned.validity) == 2
    # First feature should be marked as empty/invalid
    assert not owned.validity[0]
    assert owned.validity[1]


# ---------------------------------------------------------------------------
# Test 5: Float precision — negatives, scientific notation, many decimals
# ---------------------------------------------------------------------------
@needs_gpu
def test_float_precision(tmp_path):
    coords = [
        [-80.92302345678, 27.95740812345],
        [1.5e2, -2.5e1],
        [0.123456789012, -0.987654321098],
        [-80.92302345678, 27.95740812345],
    ]
    fc = _make_feature_collection([
        _make_polygon_feature([coords]),
    ])
    path = tmp_path / "test_precision.geojson"
    _write_geojson(path, fc)

    batch = read_geojson_owned(path, prefer="gpu-byte-classify")
    owned = batch.geometry
    dev_buf = owned.device_state.families[GeometryFamily.POLYGON]
    gpu_x = cp.asnumpy(dev_buf.x)
    gpu_y = cp.asnumpy(dev_buf.y)

    expected_x = np.array([c[0] for c in coords], dtype=np.float64)
    expected_y = np.array([c[1] for c in coords], dtype=np.float64)

    np.testing.assert_allclose(gpu_x, expected_x, atol=1e-6)
    np.testing.assert_allclose(gpu_y, expected_y, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 6: Hybrid properties — lazy-loaded properties match baseline
# ---------------------------------------------------------------------------
@needs_gpu
def test_hybrid_properties(tmp_path):
    features = [
        _make_polygon_feature(_simple_square(0, 0), {"id": 1, "name": "alpha"}),
        _make_polygon_feature(_simple_square(1, 1), {"id": 2, "name": "beta"}),
        _make_polygon_feature(_simple_square(2, 2), {"id": 3, "name": "gamma"}),
    ]
    fc = _make_feature_collection(features)
    path = tmp_path / "test_props.geojson"
    _write_geojson(path, fc)

    batch = read_geojson_owned(path, prefer="gpu-byte-classify")
    props = batch.properties

    assert len(props) == 3
    assert props[0]["id"] == 1
    assert props[0]["name"] == "alpha"
    assert props[1]["id"] == 2
    assert props[2]["name"] == "gamma"


# ---------------------------------------------------------------------------
# Test 7: Device residency
# ---------------------------------------------------------------------------
@needs_gpu
def test_device_residency(tmp_path):
    fc = _make_feature_collection([
        _make_polygon_feature(_simple_square(0, 0)),
    ])
    path = tmp_path / "test_residency.geojson"
    _write_geojson(path, fc)

    batch = read_geojson_owned(path, prefer="gpu-byte-classify")
    owned = batch.geometry

    assert owned.residency == Residency.DEVICE
    assert owned.device_state is not None


# ---------------------------------------------------------------------------
# Test 8: Fallback — GPU unavailable
# ---------------------------------------------------------------------------
def test_fallback_without_gpu(tmp_path, monkeypatch):
    fc = _make_feature_collection([
        _make_polygon_feature(_simple_square(0, 0), {"id": 1}),
    ])
    path = tmp_path / "test_fallback.geojson"
    _write_geojson(path, fc)

    # fast-json strategy should always work regardless of GPU
    batch = read_geojson_owned(path, prefer="fast-json")
    assert batch.geometry.families[GeometryFamily.POLYGON].row_count == 1


# ---------------------------------------------------------------------------
# Test 9: Strategy plan
# ---------------------------------------------------------------------------
def test_plan_gpu_byte_classify():
    plan = plan_geojson_ingest(prefer="gpu-byte-classify")
    assert plan.selected_strategy == "gpu-byte-classify"
    assert plan.implementation == "geojson_gpu_byte_classify"
    assert "byte-classification" in plan.reason


# ---------------------------------------------------------------------------
# Test 10: Multiple rings (polygon with hole)
# ---------------------------------------------------------------------------
@needs_gpu
def test_polygon_with_hole(tmp_path):
    outer = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
    hole = [[2, 2], [8, 2], [8, 8], [2, 8], [2, 2]]
    fc = _make_feature_collection([
        _make_polygon_feature([outer, hole]),
    ])
    path = tmp_path / "test_hole.geojson"
    _write_geojson(path, fc)

    batch = read_geojson_owned(path, prefer="gpu-byte-classify")
    owned = batch.geometry
    assert GeometryFamily.POLYGON in owned.families

    dev_buf = owned.device_state.families[GeometryFamily.POLYGON]
    # 5 coords in outer ring + 5 in hole = 10 total
    assert len(cp.asnumpy(dev_buf.x)) == 10

    # geometry_offsets should show 2 rings for the single feature
    geom_off = cp.asnumpy(dev_buf.geometry_offsets)
    assert geom_off[0] == 0
    assert geom_off[1] == 2  # 2 rings

    # ring_offsets should show the split
    ring_off = cp.asnumpy(dev_buf.ring_offsets)
    assert ring_off[0] == 0
    assert ring_off[1] == 5
    assert ring_off[2] == 10


# ---------------------------------------------------------------------------
# Point geometry helpers
# ---------------------------------------------------------------------------
def _make_point_feature(coords: list[float], properties: dict | None = None) -> dict:
    return {
        "type": "Feature",
        "properties": properties or {},
        "geometry": {
            "type": "Point",
            "coordinates": coords,
        },
    }


def _make_linestring_feature(coords: list[list[float]], properties: dict | None = None) -> dict:
    return {
        "type": "Feature",
        "properties": properties or {},
        "geometry": {
            "type": "LineString",
            "coordinates": coords,
        },
    }


# ---------------------------------------------------------------------------
# Test 11: Homogeneous Points
# ---------------------------------------------------------------------------
@needs_gpu
def test_homogeneous_points(tmp_path):
    features = [_make_point_feature([i * 0.1, i * 0.2], {"id": i}) for i in range(100)]
    fc = _make_feature_collection(features)
    path = tmp_path / "points.geojson"
    _write_geojson(path, fc)

    gpu_batch = read_geojson_owned(path, prefer="gpu-byte-classify")
    cpu_batch = read_geojson_owned(path, prefer="fast-json")

    gpu_owned = gpu_batch.geometry
    cpu_owned = cpu_batch.geometry

    assert GeometryFamily.POINT in gpu_owned.families
    gpu_buf = gpu_owned.families[GeometryFamily.POINT]
    cpu_buf = cpu_owned.families[GeometryFamily.POINT]
    assert gpu_buf.row_count == cpu_buf.row_count == 100

    if gpu_owned.device_state is not None:
        dev_buf = gpu_owned.device_state.families[GeometryFamily.POINT]
        gpu_x = cp.asnumpy(dev_buf.x)
        gpu_y = cp.asnumpy(dev_buf.y)
    else:
        gpu_x = gpu_buf.x
        gpu_y = gpu_buf.y

    np.testing.assert_allclose(gpu_x, cpu_buf.x, atol=1e-10)
    np.testing.assert_allclose(gpu_y, cpu_buf.y, atol=1e-10)


# ---------------------------------------------------------------------------
# Test 12: Homogeneous LineStrings
# ---------------------------------------------------------------------------
@needs_gpu
def test_homogeneous_linestrings(tmp_path):
    features = []
    for i in range(50):
        coords = [[j * 0.1 + i, j * 0.2 + i] for j in range(3 + i % 5)]
        features.append(_make_linestring_feature(coords, {"id": i}))
    fc = _make_feature_collection(features)
    path = tmp_path / "lines.geojson"
    _write_geojson(path, fc)

    gpu_batch = read_geojson_owned(path, prefer="gpu-byte-classify")
    cpu_batch = read_geojson_owned(path, prefer="fast-json")

    gpu_owned = gpu_batch.geometry
    cpu_owned = cpu_batch.geometry

    assert GeometryFamily.LINESTRING in gpu_owned.families
    gpu_buf = gpu_owned.families[GeometryFamily.LINESTRING]
    cpu_buf = cpu_owned.families[GeometryFamily.LINESTRING]
    assert gpu_buf.row_count == cpu_buf.row_count == 50

    if gpu_owned.device_state is not None:
        dev_buf = gpu_owned.device_state.families[GeometryFamily.LINESTRING]
        gpu_x = cp.asnumpy(dev_buf.x)
        gpu_y = cp.asnumpy(dev_buf.y)
    else:
        gpu_x = gpu_buf.x
        gpu_y = gpu_buf.y

    np.testing.assert_allclose(gpu_x, cpu_buf.x, atol=1e-10)
    np.testing.assert_allclose(gpu_y, cpu_buf.y, atol=1e-10)


# ---------------------------------------------------------------------------
# Test 13: Mixed Point + LineString
# ---------------------------------------------------------------------------
@needs_gpu
def test_mixed_point_linestring(tmp_path):
    features = [
        _make_point_feature([1.0, 2.0]),
        _make_linestring_feature([[3.0, 4.0], [5.0, 6.0]]),
        _make_point_feature([7.0, 8.0]),
        _make_linestring_feature([[9.0, 10.0], [11.0, 12.0], [13.0, 14.0]]),
    ]
    fc = _make_feature_collection(features)
    path = tmp_path / "mixed_pt_ls.geojson"
    _write_geojson(path, fc)

    batch = read_geojson_owned(path, prefer="gpu-byte-classify")
    owned = batch.geometry

    assert GeometryFamily.POINT in owned.families
    assert GeometryFamily.LINESTRING in owned.families
    assert owned.families[GeometryFamily.POINT].row_count == 2
    assert owned.families[GeometryFamily.LINESTRING].row_count == 2

    # Check Point coordinates
    dev_pt = owned.device_state.families[GeometryFamily.POINT]
    pt_x = cp.asnumpy(dev_pt.x)
    pt_y = cp.asnumpy(dev_pt.y)
    np.testing.assert_allclose(pt_x, [1.0, 7.0], atol=1e-10)
    np.testing.assert_allclose(pt_y, [2.0, 8.0], atol=1e-10)

    # Check LineString coordinates
    dev_ls = owned.device_state.families[GeometryFamily.LINESTRING]
    ls_x = cp.asnumpy(dev_ls.x)
    ls_y = cp.asnumpy(dev_ls.y)
    np.testing.assert_allclose(ls_x, [3.0, 5.0, 9.0, 11.0, 13.0], atol=1e-10)
    np.testing.assert_allclose(ls_y, [4.0, 6.0, 10.0, 12.0, 14.0], atol=1e-10)


# ---------------------------------------------------------------------------
# Test 14: Mixed Point + Polygon
# ---------------------------------------------------------------------------
@needs_gpu
def test_mixed_point_polygon(tmp_path):
    features = [
        _make_point_feature([0.0, 0.0]),
        _make_polygon_feature(_simple_square(1.0, 1.0)),
        _make_point_feature([2.0, 2.0]),
    ]
    fc = _make_feature_collection(features)
    path = tmp_path / "mixed_pt_pg.geojson"
    _write_geojson(path, fc)

    batch = read_geojson_owned(path, prefer="gpu-byte-classify")
    owned = batch.geometry

    assert GeometryFamily.POINT in owned.families
    assert GeometryFamily.POLYGON in owned.families
    assert owned.families[GeometryFamily.POINT].row_count == 2
    assert owned.families[GeometryFamily.POLYGON].row_count == 1


# ---------------------------------------------------------------------------
# Test 15: Mixed all three types
# ---------------------------------------------------------------------------
@needs_gpu
def test_mixed_all_three(tmp_path):
    features = [
        _make_point_feature([1.0, 2.0]),
        _make_linestring_feature([[3.0, 4.0], [5.0, 6.0]]),
        _make_polygon_feature(_simple_square(0.0, 0.0)),
    ]
    fc = _make_feature_collection(features)
    path = tmp_path / "mixed_all.geojson"
    _write_geojson(path, fc)

    batch = read_geojson_owned(path, prefer="gpu-byte-classify")
    owned = batch.geometry

    assert GeometryFamily.POINT in owned.families
    assert GeometryFamily.LINESTRING in owned.families
    assert GeometryFamily.POLYGON in owned.families
    assert owned.families[GeometryFamily.POINT].row_count == 1
    assert owned.families[GeometryFamily.LINESTRING].row_count == 1
    assert owned.families[GeometryFamily.POLYGON].row_count == 1


# ---------------------------------------------------------------------------
# Test 16: Mixed with properties
# ---------------------------------------------------------------------------
@needs_gpu
def test_mixed_properties(tmp_path):
    features = [
        _make_point_feature([1.0, 2.0], {"name": "pt"}),
        _make_linestring_feature([[3.0, 4.0], [5.0, 6.0]], {"name": "ls"}),
    ]
    fc = _make_feature_collection(features)
    path = tmp_path / "mixed_props.geojson"
    _write_geojson(path, fc)

    batch = read_geojson_owned(path, prefer="gpu-byte-classify")
    props = batch.properties
    assert len(props) == 2
    assert props[0]["name"] == "pt"
    assert props[1]["name"] == "ls"


# ---------------------------------------------------------------------------
# Test 17: Single point
# ---------------------------------------------------------------------------
@needs_gpu
def test_single_point(tmp_path):
    fc = _make_feature_collection([_make_point_feature([42.0, -73.5])])
    path = tmp_path / "single_pt.geojson"
    _write_geojson(path, fc)

    batch = read_geojson_owned(path, prefer="gpu-byte-classify")
    owned = batch.geometry
    assert GeometryFamily.POINT in owned.families
    assert owned.families[GeometryFamily.POINT].row_count == 1

    dev_buf = owned.device_state.families[GeometryFamily.POINT]
    np.testing.assert_allclose(cp.asnumpy(dev_buf.x), [42.0], atol=1e-10)
    np.testing.assert_allclose(cp.asnumpy(dev_buf.y), [-73.5], atol=1e-10)


# ---------------------------------------------------------------------------
# Test 18: LineString with 2 points (minimum valid)
# ---------------------------------------------------------------------------
@needs_gpu
def test_linestring_two_points(tmp_path):
    fc = _make_feature_collection([
        _make_linestring_feature([[0.0, 0.0], [1.0, 1.0]]),
    ])
    path = tmp_path / "ls_min.geojson"
    _write_geojson(path, fc)

    batch = read_geojson_owned(path, prefer="gpu-byte-classify")
    owned = batch.geometry
    assert GeometryFamily.LINESTRING in owned.families
    dev_buf = owned.device_state.families[GeometryFamily.LINESTRING]
    assert len(cp.asnumpy(dev_buf.x)) == 2


# ---------------------------------------------------------------------------
# Test 19: Type key not confused with Feature/FeatureCollection type
# ---------------------------------------------------------------------------
@needs_gpu
def test_type_key_not_confused(tmp_path):
    fc = _make_feature_collection([
        _make_point_feature([1.0, 2.0]),
        _make_polygon_feature(_simple_square(0.0, 0.0)),
    ])
    path = tmp_path / "type_filter.geojson"
    _write_geojson(path, fc)

    batch = read_geojson_owned(path, prefer="gpu-byte-classify")
    owned = batch.geometry
    # Should have exactly 2 features, not confused by "type":"Feature" etc.
    assert len(owned.validity) == 2


# ---------------------------------------------------------------------------
# Test 20: "type" inside property string
# ---------------------------------------------------------------------------
@needs_gpu
def test_type_in_property_string(tmp_path):
    features = [
        _make_point_feature([1.0, 2.0], {"desc": 'The "type": is in this string'}),
        _make_point_feature([3.0, 4.0]),
    ]
    fc = _make_feature_collection(features)
    path = tmp_path / "type_in_string.geojson"
    _write_geojson(path, fc)

    batch = read_geojson_owned(path, prefer="gpu-byte-classify")
    assert GeometryFamily.POINT in batch.geometry.families
    assert batch.geometry.families[GeometryFamily.POINT].row_count == 2


# ---------------------------------------------------------------------------
# Test 21: Unsupported MultiPoint raises error
# ---------------------------------------------------------------------------
@needs_gpu
def test_unsupported_multipoint(tmp_path):
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "MultiPoint",
                    "coordinates": [[0, 0], [1, 1]],
                },
            }
        ],
    }
    path = tmp_path / "multipoint.geojson"
    _write_geojson(path, fc)

    with pytest.raises(NotImplementedError, match="unsupported geometry types"):
        read_geojson_owned(path, prefer="gpu-byte-classify")
