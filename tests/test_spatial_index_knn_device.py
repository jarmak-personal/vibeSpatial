from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Point, box

from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device
from vibespatial.runtime import has_gpu_runtime
from vibespatial.runtime.residency import Residency
from vibespatial.spatial.indexing import build_flat_spatial_index
from vibespatial.spatial.query_types import CandidateRelationCapacityError
from vibespatial.spatial.spatial_index_knn_device import (
    _KnnWorkspacePlan,
    _plan_knn_workspace,
    _topk_per_query,
    spatial_index_knn_device,
)

requires_gpu = pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required")


def _device_knn(tree_geometries, query_geometries, *, k, max_distance=None):
    tree_owned = from_shapely_geometries(
        tree_geometries,
        residency=Residency.DEVICE,
    )
    query_owned = from_shapely_geometries(
        query_geometries,
        residency=Residency.DEVICE,
    )
    flat_index = build_flat_spatial_index(tree_owned)
    native_index = flat_index.to_native_spatial_index()
    result = spatial_index_knn_device(
        query_owned,
        tree_owned,
        compute_geometry_bounds_device(query_owned),
        compute_geometry_bounds_device(tree_owned),
        native_spatial_index=native_index,
        k=k,
        max_distance=max_distance,
        return_all=False,
    )
    assert result is not None
    return result, flat_index


def _oracle(tree_geometries, query_geometries, *, k, max_distance=None):
    expected = []
    for query_row, query_geometry in enumerate(query_geometries):
        if query_geometry is None or query_geometry.is_empty:
            continue
        ranked = []
        for target_row, target_geometry in enumerate(tree_geometries):
            if target_geometry is None or target_geometry.is_empty:
                continue
            distance = float(query_geometry.distance(target_geometry))
            if max_distance is None or distance <= max_distance:
                ranked.append((distance, target_row))
        for distance, target_row in sorted(ranked)[:k]:
            expected.append((query_row, target_row, distance))
    return expected


@requires_gpu
def test_topk_orders_equal_distances_by_target_row() -> None:
    cp = pytest.importorskip("cupy")

    query, target, distance, count = _topk_per_query(
        cp.asarray([0, 0, 0, 1, 1], dtype=cp.int32),
        cp.asarray([7, 2, 4, 8, 3], dtype=cp.int32),
        cp.asarray([1.0, 1.0, 1.0, 2.0, 2.0], dtype=cp.float64),
        2,
        2,
    )

    assert count == 4
    assert cp.asnumpy(query).tolist() == [0, 0, 1, 1]
    assert cp.asnumpy(target).tolist() == [2, 4, 3, 8]
    assert cp.asnumpy(distance).tolist() == [1.0, 1.0, 2.0, 2.0]


@requires_gpu
@pytest.mark.parametrize("max_distance", [None, 1.25])
def test_bounded_fixed_k_matches_bruteforce_oracle(max_distance) -> None:
    tree = [
        box(0.0, 0.0, 0.2, 0.2),
        box(1.0, 0.0, 1.2, 0.2),
        box(2.0, 0.0, 2.2, 0.2),
        box(3.0, 0.0, 3.2, 0.2),
        box(4.0, 0.0, 4.2, 0.2),
    ]
    query = [Point(0.1, 1.0), Point(2.63, 0.8), Point(10.0, 10.0)]
    result, _flat_index = _device_knn(
        tree,
        query,
        k=3,
        max_distance=max_distance,
    )
    left, right, distances = result.to_host()

    actual = list(zip(left.tolist(), right.tolist(), distances.tolist(), strict=True))
    expected = _oracle(tree, query, k=3, max_distance=max_distance)
    assert [(q, t) for q, t, _ in actual] == [(q, t) for q, t, _ in expected]
    np.testing.assert_allclose(
        [distance for _, _, distance in actual],
        [distance for _, _, distance in expected],
        rtol=1.0e-6,
        atol=1.0e-9,
    )


@requires_gpu
def test_max_distance_refines_fp32_false_negative_before_filtering() -> None:
    """A true distance just inside the threshold survives coarse rounding."""
    true_distance = 1.00000006
    max_distance = 1.00000008
    result, _flat_index = _device_knn(
        [Point(true_distance, 0.0)],
        [Point(0.0, 0.0)],
        k=1,
        max_distance=max_distance,
    )

    left, right, distances = result.to_host()
    assert left.tolist() == [0]
    assert right.tolist() == [0]
    np.testing.assert_allclose(distances, [true_distance], rtol=0.0, atol=1.0e-12)


@requires_gpu
def test_nonfinite_fp32_distances_refine_before_exact_filtering() -> None:
    """Finite fp64 distances that overflow fp32 must not disappear."""
    result, _flat_index = _device_knn(
        [Point(1.0e39, 0.0), Point(2.0e39, 0.0)],
        [Point(0.0, 0.0)],
        k=2,
        max_distance=3.0e39,
    )

    left, right, distances = result.to_host()
    assert left.tolist() == [0, 0]
    assert right.tolist() == [0, 1]
    np.testing.assert_allclose(distances, [1.0e39, 2.0e39], rtol=1.0e-15)


@requires_gpu
def test_fixed_k_handles_missing_empty_and_k_larger_than_targets() -> None:
    tree = [Point(0.0, 0.0), None, Point(), Point(2.0, 0.0)]
    query = [None, Point(), Point(1.0, 0.0)]
    result, _flat_index = _device_knn(tree, query, k=5)
    left, right, distances = result.to_host()

    assert left.tolist() == [2, 2]
    assert right.tolist() == [0, 3]
    assert distances.tolist() == [1.0, 1.0]


@requires_gpu
def test_fixed_k_streams_query_and_target_tiles(monkeypatch) -> None:
    import vibespatial.spatial.spatial_index_knn_device as knn_module
    from vibespatial.cuda._runtime import get_d2h_transfer_events

    monkeypatch.setattr(
        knn_module,
        "_plan_knn_workspace",
        lambda n_queries, n_tree, k: _KnnWorkspacePlan(
            query_tile_rows=1,
            target_tile_rows=2,
            pair_capacity=2,
            final_output_bytes=n_queries * k * 32,
            tile_fixed_bytes=512,
            admitted_workspace_bytes=4096,
        ),
    )
    monkeypatch.setattr(
        knn_module,
        "_initial_search_radius",
        lambda *args, **kwargs: 0.25,
    )
    candidate_pointers = []
    original_query = knn_module.spatial_index_device_query

    def record_candidate_workspace(*args, **kwargs):
        candidates, execution = original_query(*args, **kwargs)
        if candidates is not None and candidates.total_pairs:
            candidate_pointers.append(int(candidates.d_left.data.ptr))
        return candidates, execution

    monkeypatch.setattr(
        knn_module,
        "spatial_index_device_query",
        record_candidate_workspace,
    )
    tree = [Point(float(value), 0.0) for value in range(7)]
    query = [Point(0.5, 0.0), Point(5.5, 0.0)]
    get_d2h_transfer_events(clear=True)
    result, _flat_index = _device_knn(tree, query, k=3)
    engine_transfers = get_d2h_transfer_events(clear=True)
    left, right, distances = result.to_host()

    expected = _oracle(tree, query, k=3)
    assert list(zip(left.tolist(), right.tolist(), strict=True)) == [
        (q, t) for q, t, _ in expected
    ]
    np.testing.assert_allclose(
        distances,
        [distance for _, _, distance in expected],
    )
    assert result.telemetry is not None
    assert result.telemetry.query_tiles == 2
    assert result.telemetry.target_stream_tiles > 2
    assert result.telemetry.radius_iterations > 2
    assert result.telemetry.max_candidate_pairs <= 2
    assert result.telemetry.allocation_count < 5_000
    assert result.telemetry.d2h_count < 256
    assert result.telemetry.d2h_bytes < 2_048
    assert result.telemetry.materialization_count == 0
    assert len(candidate_pointers) > 2
    assert len(set(candidate_pointers)) == 1
    assert engine_transfers
    assert max(event.bytes_transferred for event in engine_transfers) <= 40


@requires_gpu
def test_fixed_k_peak_telemetry_is_operation_local() -> None:
    """A prior process high-water allocation must not become this run's peak."""
    cp = pytest.importorskip("cupy")
    historical_allocation_bytes = 256 * 1024 * 1024
    historical = cp.empty(historical_allocation_bytes, dtype=cp.uint8)
    del historical

    result, _flat_index = _device_knn(
        [Point(0.0, 0.0), Point(2.0, 0.0), Point(4.0, 0.0)],
        [Point(1.0, 0.0)],
        k=2,
    )

    assert result.telemetry is not None
    assert result.telemetry.peak_device_bytes is not None
    assert result.telemetry.allocation_count is not None
    assert 0 < result.telemetry.peak_device_bytes < historical_allocation_bytes
    assert result.telemetry.allocation_count > 0


@requires_gpu
def test_all_invalid_unbounded_telemetry_includes_extent_fence() -> None:
    from vibespatial.cuda._runtime import get_d2h_transfer_stats

    start_count, start_bytes = get_d2h_transfer_stats()
    result, _flat_index = _device_knn([None], [None], k=1)
    end_count, end_bytes = get_d2h_transfer_stats()

    assert result.total_pairs == 0
    assert result.telemetry is not None
    assert result.telemetry.d2h_count == end_count - start_count == 1
    assert result.telemetry.d2h_bytes == end_bytes - start_bytes == 40


@requires_gpu
def test_all_invalid_bounded_telemetry_counts_terminating_fence() -> None:
    result, _flat_index = _device_knn(
        [Point(0.0, 0.0)],
        [None],
        k=1,
        max_distance=1.0,
    )

    assert result.total_pairs == 0
    assert result.telemetry is not None
    assert result.telemetry.radius_iterations == 0
    assert result.telemetry.scalar_fences == 1
    assert result.telemetry.d2h_count == 1


@requires_gpu
def test_fixed_k_reuses_supplied_native_index(monkeypatch) -> None:
    import vibespatial.spatial.spatial_index_knn_device as knn_module

    seen = []
    original = knn_module.spatial_index_device_query

    def wrapped(flat_index, *args, **kwargs):
        seen.append(flat_index)
        return original(flat_index, *args, **kwargs)

    monkeypatch.setattr(knn_module, "spatial_index_device_query", wrapped)
    result, flat_index = _device_knn(
        [Point(0.0, 0.0), Point(2.0, 0.0), Point(4.0, 0.0)],
        [Point(1.0, 0.0)],
        k=2,
    )

    assert result.total_pairs == 2
    assert seen
    assert all(candidate_index is flat_index for candidate_index in seen)


@requires_gpu
def test_public_fixed_k_reuses_cached_native_index() -> None:
    from vibespatial.api import GeoSeries
    from vibespatial.api.geometry_array import GeometryArray

    tree = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [Point(0.0, 0.0), Point(2.0, 0.0), Point(4.0, 0.0)],
                residency=Residency.DEVICE,
            )
        )
    )
    query = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries(
                [Point(1.0, 0.0), Point(3.0, 0.0)],
                residency=Residency.DEVICE,
            )
        )
    )
    sindex = tree.sindex

    first = sindex.nearest(query, return_all=False, return_distance=True, k=2)
    cached = sindex._native_spatial_index
    second = sindex.nearest(query, return_all=False, return_distance=True, k=2)

    assert cached is not None
    assert sindex._native_spatial_index is cached
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_allclose(first[1], second[1])


@requires_gpu
def test_fixed_k_refines_large_offset_near_ties() -> None:
    base = 1.0e9
    tree = [
        Point(base + 0.1, base),
        Point(base - 0.1, base),
        Point(base + 0.1000002, base),
        Point(base + 2.0, base),
    ]
    query = [Point(base, base)]
    result, _flat_index = _device_knn(tree, query, k=3)
    left, right, distances = result.to_host()

    expected = _oracle(tree, query, k=3)
    assert left.tolist() == [0, 0, 0]
    assert right.tolist() == [target for _, target, _ in expected]
    np.testing.assert_allclose(
        distances,
        [distance for _, _, distance in expected],
        rtol=0.0,
        atol=1.0e-12,
    )


@requires_gpu
def test_workspace_plan_streams_single_query_when_tree_exceeds_capacity(
    monkeypatch,
) -> None:
    from vibespatial.cuda._runtime import get_cuda_runtime

    runtime = get_cuda_runtime()
    monkeypatch.setattr(
        type(runtime),
        "query_memory_remaining_bytes",
        lambda self: 1_000_000,
    )
    plan = _plan_knn_workspace(1_000, 100_000, 5)

    assert plan.query_tile_rows == 1
    assert plan.target_tile_rows < 100_000
    assert plan.pair_capacity == plan.target_tile_rows
    assert plan.admitted_workspace_bytes <= 1_000_000


@requires_gpu
def test_workspace_plan_applies_int32_pair_limit_before_tile_shape(
    monkeypatch,
) -> None:
    from vibespatial.cuda._runtime import get_cuda_runtime

    runtime = get_cuda_runtime()
    monkeypatch.setattr(
        type(runtime),
        "query_memory_remaining_bytes",
        lambda self: 1_000_000_000_000,
    )
    tree_rows = np.iinfo(np.int32).max
    plan = _plan_knn_workspace(2, tree_rows, 1)

    assert plan.query_tile_rows == 1
    assert plan.target_tile_rows == tree_rows
    assert plan.pair_capacity == tree_rows


@requires_gpu
def test_workspace_plan_rejects_minimum_before_allocation(monkeypatch) -> None:
    from vibespatial.cuda._runtime import get_cuda_runtime

    runtime = get_cuda_runtime()
    monkeypatch.setattr(
        type(runtime),
        "query_memory_remaining_bytes",
        lambda self: 255,
    )

    with pytest.raises(
        CandidateRelationCapacityError,
        match="cannot admit its minimum device workspace",
    ):
        _plan_knn_workspace(1, 1, 1)


def test_public_fixed_k_rejects_return_all_semantics() -> None:
    from vibespatial.api import GeoSeries

    tree = GeoSeries([Point(0.0, 0.0), Point(1.0, 0.0)])

    with pytest.raises(ValueError, match="k > 1 requires return_all=False"):
        tree.sindex.nearest([Point(0.5, 0.0)], k=2, return_all=True)


@pytest.mark.parametrize("max_distance", [0.0, -1.0])
def test_public_fixed_k_rejects_nonpositive_max_distance(max_distance) -> None:
    from vibespatial.api import GeoSeries

    tree = GeoSeries([Point(0.0, 0.0), Point(1.0, 0.0)])

    with pytest.raises(ValueError, match="max_distance must be greater than 0"):
        tree.sindex.nearest(
            [Point(0.5, 0.0)],
            k=2,
            return_all=False,
            max_distance=max_distance,
        )


def test_public_fixed_k_exclusive_declines_observably() -> None:
    from vibespatial.api import GeoSeries
    from vibespatial.api.geometry_array import GeometryArray
    from vibespatial.runtime.fallbacks import clear_fallback_events, get_fallback_events

    tree = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries([Point(0.0, 0.0), Point(1.0, 0.0)])
        )
    )
    query = GeoSeries(
        GeometryArray.from_owned(from_shapely_geometries([Point(0.0, 0.0)]))
    )
    clear_fallback_events()

    with pytest.raises(NotImplementedError, match="k > 1"):
        tree.sindex.nearest(
            query,
            return_all=False,
            exclusive=True,
            k=2,
        )

    events = get_fallback_events(clear=True)
    assert len(events) == 1
    assert "geometric exclusion" in events[0].reason


def test_public_fixed_k_exclusive_fails_in_strict_native_mode() -> None:
    from vibespatial.api import GeoSeries
    from vibespatial.api.geometry_array import GeometryArray
    from vibespatial.runtime.fallbacks import StrictNativeFallbackError
    from vibespatial.testing import strict_native_environment

    tree = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries([Point(0.0, 0.0), Point(1.0, 0.0)])
        )
    )
    query = GeoSeries(
        GeometryArray.from_owned(from_shapely_geometries([Point(0.0, 0.0)]))
    )

    with strict_native_environment(), pytest.raises(StrictNativeFallbackError):
        tree.sindex.nearest(
            query,
            return_all=False,
            exclusive=True,
            k=2,
        )
