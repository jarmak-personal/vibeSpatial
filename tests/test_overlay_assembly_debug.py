from __future__ import annotations

import os

import numpy as np
import pytest
import shapely

from vibespatial.api import read_file
from vibespatial.constructive.binary_constructive import _dispatch_overlay_gpu
from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.overlay.assemble import _build_polygon_output_from_faces_gpu
from vibespatial.overlay.faces import _select_overlay_face_indices_gpu, build_gpu_overlay_faces
from vibespatial.overlay.gpu import (
    _build_overlay_execution_plan,
    _materialize_overlay_execution_plan,
)
from vibespatial.overlay.graph import build_gpu_half_edge_graph
from vibespatial.overlay.host_fallback import _build_polygon_output_from_faces
from vibespatial.overlay.split import build_gpu_atomic_edges, build_gpu_split_events
from vibespatial.runtime import ExecutionMode


@pytest.mark.gpu
def test_nybb_pair_gpu_face_assembly_matches_host() -> None:
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
    split_events = build_gpu_split_events(left_owned, right_owned)
    atomic_edges = build_gpu_atomic_edges(split_events)
    half_edge_graph = build_gpu_half_edge_graph(atomic_edges)
    faces = build_gpu_overlay_faces(left_owned, right_owned, half_edge_graph=half_edge_graph)
    selected = _select_overlay_face_indices_gpu(faces, operation="intersection")

    gpu_result = _build_polygon_output_from_faces_gpu(half_edge_graph, faces, selected)
    host_result = _build_polygon_output_from_faces(half_edge_graph, faces, selected.get())

    host_geom = host_result.to_shapely()[0]
    gpu_geom = gpu_result.to_shapely()[0]
    assert host_geom.is_valid
    assert gpu_geom.is_valid
    assert gpu_geom.geom_type == host_geom.geom_type
    assert gpu_geom.normalize().equals_exact(host_geom.normalize(), tolerance=1e-6)


@pytest.mark.gpu
def test_overlay_execution_plan_materializes_multiple_ops_from_one_topology() -> None:
    left = from_shapely_geometries([
        shapely.box(0.0, 0.0, 4.0, 4.0),
    ])
    right = from_shapely_geometries([
        shapely.box(2.0, 1.0, 5.0, 3.0),
    ])

    plan = _build_overlay_execution_plan(left, right)
    intersection, intersection_mode = _materialize_overlay_execution_plan(
        plan,
        operation="intersection",
        requested=ExecutionMode.GPU,
    )
    difference, difference_mode = _materialize_overlay_execution_plan(
        plan,
        operation="difference",
        requested=ExecutionMode.GPU,
    )

    assert intersection_mode is ExecutionMode.GPU
    assert difference_mode is ExecutionMode.GPU

    expected_intersection = shapely.intersection(left.to_shapely()[0], right.to_shapely()[0])
    expected_difference = shapely.difference(left.to_shapely()[0], right.to_shapely()[0])

    actual_intersection = intersection.to_shapely()[0]
    actual_difference = difference.to_shapely()[0]

    assert actual_intersection.normalize().equals_exact(
        expected_intersection.normalize(),
        tolerance=1e-6,
    )
    assert actual_difference.normalize().equals_exact(
        expected_difference.normalize(),
        tolerance=1e-6,
    )


@pytest.mark.gpu
def test_row_isolated_difference_preserves_sparse_multipolygon_rows() -> None:
    left = from_shapely_geometries([
        shapely.box(0.0, 0.0, 10.0, 10.0),
        shapely.box(20.0, 20.0, 30.0, 30.0),
        shapely.box(40.0, 40.0, 50.0, 50.0),
    ])
    right = from_shapely_geometries([
        shapely.box(0.0, 0.0, 10.0, 10.0),
        shapely.box(20.0, 20.0, 30.0, 30.0),
        shapely.box(44.0, 39.0, 46.0, 51.0),
    ])

    result = _dispatch_overlay_gpu(
        "difference",
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
        _row_isolated=True,
    )

    got = result.to_shapely()
    expected = shapely.difference(
        np.asarray(left.to_shapely(), dtype=object),
        np.asarray(right.to_shapely(), dtype=object),
    ).tolist()

    assert result.row_count == 3
    assert got[0] is None or shapely.is_empty(got[0])
    assert got[1] is None or shapely.is_empty(got[1])
    assert got[2] is not None
    assert got[2].geom_type == "MultiPolygon"
    assert got[2].normalize().equals_exact(expected[2].normalize(), tolerance=1e-6)
