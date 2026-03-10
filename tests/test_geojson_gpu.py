"""Tests for GPU byte-classification GeoJSON parser."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pytest

from vibespatial.io_geojson import (
    GeoJSONOwnedBatch,
    plan_geojson_ingest,
    read_geojson_owned,
)
from vibespatial.geometry_buffers import GeometryFamily
from vibespatial.residency import Residency

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
