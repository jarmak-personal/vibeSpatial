from __future__ import annotations

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
    assert owned.runtime_history[-1].selected.value == expected
