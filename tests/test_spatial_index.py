from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Point, Polygon, box

from vibespatial import (
    ExecutionMode,
    RuntimeSelection,
    build_flat_spatial_index,
    from_shapely_geometries,
    has_gpu_runtime,
)


def test_build_flat_spatial_index_sorts_by_morton_key() -> None:
    owned = from_shapely_geometries([Point(2, 2), Point(0, 0), Point(1, 1)])

    index = build_flat_spatial_index(
        owned,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.CPU,
            reason="cpu baseline",
        ),
    )

    ordered_keys = index.morton_keys[index.order]
    assert ordered_keys.tolist() == sorted(index.morton_keys.tolist())
    assert any("cpu baseline" in reason for reason in owned.diagnostics_report()["runtime_history"])


@pytest.mark.gpu
def test_query_bounds_returns_matching_rows() -> None:
    owned = from_shapely_geometries([Point(0, 0), Point(10, 10), box(5, 5, 7, 7)])
    index = build_flat_spatial_index(owned)

    matches = index.query_bounds((4.0, 4.0, 6.0, 6.0))

    assert set(matches.tolist()) == {2}


@pytest.mark.gpu
def test_query_returns_candidate_pairs_for_sindex_style_workload() -> None:
    left = from_shapely_geometries([Point(0, 0), Point(8, 8), Point(50, 50)])
    right = from_shapely_geometries(
        [
            Polygon([(0, 0), (3, 0), (3, 3), (0, 0)]),
            Polygon([(7, 7), (9, 7), (9, 9), (7, 7)]),
            Polygon([(100, 100), (110, 100), (110, 110), (100, 100)]),
        ]
    )
    index = build_flat_spatial_index(right)

    pairs = index.query(left, tile_size=2)

    assert set(zip(pairs.left_indices.tolist(), pairs.right_indices.tolist(), strict=True)) == {
        (0, 0),
        (1, 1),
    }


def test_build_flat_spatial_index_gpu_matches_cpu_order() -> None:
    if not has_gpu_runtime():
        return

    owned = from_shapely_geometries([Point(2, 2), Point(0, 0), Point(1, 1), Point(), None])
    cpu = build_flat_spatial_index(
        owned,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.CPU,
            reason="cpu baseline",
        ),
    )
    gpu = build_flat_spatial_index(
        owned,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.GPU,
            reason="gpu morton sort",
        ),
    )

    assert gpu.morton_keys.tolist() == cpu.morton_keys.tolist()
    assert gpu.order.tolist() == cpu.order.tolist()


@pytest.mark.gpu
def test_build_flat_spatial_index_default_prefers_gpu_when_available() -> None:
    owned = from_shapely_geometries([Point(2, 2), Point(0, 0), Point(1, 1)])

    build_flat_spatial_index(owned)

    expected = "gpu" if has_gpu_runtime() else "cpu"
    assert any(selection.selected.value == expected for selection in owned.runtime_history)


@pytest.mark.gpu
def test_build_flat_spatial_index_detects_regular_grid_from_device_polygon_stubs() -> None:
    pytest.importorskip("cupy")

    from vibespatial.cuda._runtime import get_cuda_runtime
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import (
        FAMILY_TAGS,
        DeviceFamilyGeometryBuffer,
        build_device_resident_owned,
    )

    runtime = get_cuda_runtime()
    polygon_family = GeometryFamily.POLYGON
    x = np.asarray(
        [
            0.0, 1.0, 1.0, 0.0, 0.0,
            1.0, 2.0, 2.0, 1.0, 1.0,
            0.0, 1.0, 1.0, 0.0, 0.0,
            1.0, 2.0, 2.0, 1.0, 1.0,
        ],
        dtype=np.float64,
    )
    y = np.asarray(
        [
            0.0, 0.0, 1.0, 1.0, 0.0,
            0.0, 0.0, 1.0, 1.0, 0.0,
            1.0, 1.0, 2.0, 2.0, 1.0,
            1.0, 1.0, 2.0, 2.0, 1.0,
        ],
        dtype=np.float64,
    )
    geometry_offsets = np.asarray([0, 1, 2, 3, 4], dtype=np.int32)
    ring_offsets = np.asarray([0, 5, 10, 15, 20], dtype=np.int32)
    empty_mask = np.zeros(4, dtype=np.bool_)

    owned = build_device_resident_owned(
        device_families={
            polygon_family: DeviceFamilyGeometryBuffer(
                family=polygon_family,
                x=runtime.from_host(x),
                y=runtime.from_host(y),
                geometry_offsets=runtime.from_host(geometry_offsets),
                empty_mask=runtime.from_host(empty_mask),
                ring_offsets=runtime.from_host(ring_offsets),
            ),
        },
        row_count=4,
        tags=np.full(4, FAMILY_TAGS[polygon_family], dtype=np.int8),
        validity=np.ones(4, dtype=np.bool_),
        family_row_offsets=np.arange(4, dtype=np.int32),
    )

    stub = owned.families[polygon_family]
    assert not stub.host_materialized
    assert stub.geometry_offsets.size == 0
    assert stub.ring_offsets is None

    index = build_flat_spatial_index(
        owned,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.CPU,
            reason="cpu baseline",
        ),
    )

    hydrated = owned.families[polygon_family]
    assert index.regular_grid is not None
    assert index.regular_grid.cols == 2
    assert index.regular_grid.rows == 2
    assert hydrated.host_materialized is False
    assert hydrated.x.size == 0
    assert hydrated.y.size == 0
    np.testing.assert_array_equal(hydrated.geometry_offsets, geometry_offsets)
    np.testing.assert_array_equal(hydrated.ring_offsets, ring_offsets)
