from __future__ import annotations

import os

import cupy as cp
import pytest

from vibespatial.api import read_file
from vibespatial.overlay.assemble import _build_polygon_output_from_faces_gpu
from vibespatial.overlay.faces import _select_overlay_face_indices_gpu, build_gpu_overlay_faces
from vibespatial.overlay.graph import build_gpu_half_edge_graph
from vibespatial.overlay.host_fallback import _build_polygon_output_from_faces
from vibespatial.overlay.split import build_gpu_atomic_edges, build_gpu_split_events


@pytest.mark.gpu
def test_debug_overlay_single_pair_gpu_vs_host_assembly() -> None:
    data = os.path.join(
        os.path.dirname(__file__),
        "upstream",
        "geopandas",
        "tests",
        "data",
    )
    overlay_data = os.path.join(data, "overlay", "nybb_qgis")
    left = read_file(f"zip://{os.path.join(data, 'nybb_16a.zip')}").iloc[[4]].geometry.values.to_owned()
    right = read_file(os.path.join(overlay_data, "polydf2.shp")).iloc[[8]].geometry.values.to_owned()

    split_events = build_gpu_split_events(left, right)
    atomic_edges = build_gpu_atomic_edges(split_events)
    half_edge_graph = build_gpu_half_edge_graph(atomic_edges)
    faces = build_gpu_overlay_faces(left, right, half_edge_graph=half_edge_graph, atomic_edges=atomic_edges, split_events=split_events)
    d_selected = _select_overlay_face_indices_gpu(faces, operation="intersection")

    gpu_result = _build_polygon_output_from_faces_gpu(half_edge_graph, faces, d_selected)
    host_result = _build_polygon_output_from_faces(half_edge_graph, faces, cp.asnumpy(d_selected))

    gpu_geom = gpu_result.to_shapely()[0]
    host_geom = host_result.to_shapely()[0]
    gpu_norm = gpu_geom.normalize()
    host_norm = host_geom.normalize()
    assert gpu_geom.is_valid == host_geom.is_valid
    assert gpu_norm.equals_exact(host_norm, 1e-9), (
        f"gpu_valid={gpu_geom.is_valid} gpu_area={gpu_geom.area} "
        f"host_valid={host_geom.is_valid} host_area={host_geom.area}"
    )
