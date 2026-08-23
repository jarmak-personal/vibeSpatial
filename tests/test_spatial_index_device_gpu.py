"""Tests for spatial_index_device_query — GPU BVH-style traversal.

Validates that the unified device spatial index query function:
  - Produces the same candidate pairs as CPU STRtree (no false negatives)
  - Correctly selects brute-force vs Morton range strategy
  - Handles dwithin distance expansion
  - Integrates with sjoin end-to-end
  - Reports correct execution metadata
"""

from __future__ import annotations

import inspect
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon, box

import vibespatial.spatial.query_types as query_types_module
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.kernels.core.geometry_analysis import (
    compute_geometry_bounds,
    compute_geometry_bounds_device,
)
from vibespatial.runtime import ExecutionMode, RuntimeSelection, has_gpu_runtime
from vibespatial.runtime.precision import PrecisionMode
from vibespatial.runtime.residency import Residency
from vibespatial.spatial.indexing import build_flat_spatial_index
from vibespatial.spatial.query import (
    build_owned_spatial_index,
    query_spatial_index,
    spatial_index_device_query,
)
from vibespatial.spatial.query_types import (
    CandidateRelationCapacityError,
    available_device_memory_bytes,
    require_device_candidate_pair_capacity,
)
from vibespatial.spatial.spatial_index_device import (
    _MORTON_SPAN_BUCKET_UPPER_BOUNDS,
    _classify_homogeneous_reduction_tile,
    _family_group_launch_capacities,
    _morton_range_query,
    _morton_reduction_span_schedule,
    _spatial_index_device_relation_reduction,
    _spatial_reduction_tile_lane_capacity,
)

requires_gpu = pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required")


def test_point_reduction_tile_propagates_explicit_precision_plan(monkeypatch) -> None:
    pytest.importorskip("cupy")
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.predicates import point_relations

    precision_plan = point_relations._plan_indexed_point_precision(
        PrecisionMode.AUTO,
    )
    sentinel = object()
    observed = []

    def classify(*args, precision_plan, **kwargs):
        observed.append(precision_plan)
        return sentinel

    monkeypatch.setattr(
        point_relations,
        "classify_homogeneous_point_predicates_indexed_device",
        classify,
    )

    with pytest.raises(TypeError, match="explicit PrecisionPlan"):
        _classify_homogeneous_reduction_tile(
            "intersects",
            object(),
            object(),
            np.asarray([0], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
            query_family=GeometryFamily.POINT,
            tree_family=GeometryFamily.POINT,
        )

    result = _classify_homogeneous_reduction_tile(
        "intersects",
        object(),
        object(),
        np.asarray([0], dtype=np.int32),
        np.asarray([0], dtype=np.int32),
        query_family=GeometryFamily.POINT,
        tree_family=GeometryFamily.POINT,
        precision_plan=precision_plan,
    )

    assert result is sentinel
    assert observed == [precision_plan]


def test_candidate_relation_capacity_reports_inherent_device_limit() -> None:
    device_capacity = 24 * 1024**3
    pair_count = 6_241_973_507

    with pytest.raises(CandidateRelationCapacityError, match=r"46\.51 GiB"):
        require_device_candidate_pair_capacity(
            pair_count,
            relation_name="test Morton relation",
            device_capacity_bytes=device_capacity,
        )

    assert require_device_candidate_pair_capacity(
        1_000_000,
        relation_name="test relation",
        device_capacity_bytes=device_capacity,
    ) == 8_000_000

    with pytest.raises(CandidateRelationCapacityError, match="reserved allocation budget"):
        require_device_candidate_pair_capacity(
            1_000_000,
            relation_name="live-memory relation",
            device_capacity_bytes=12_000_000,
        )

    with pytest.raises(CandidateRelationCapacityError, match="scratch"):
        require_device_candidate_pair_capacity(
            500_000,
            relation_name="refinement relation",
            device_capacity_bytes=16_000_000,
            temporary_bytes=6_000_000,
        )


def test_spatial_reduction_tile_capacity_tracks_live_memory(monkeypatch) -> None:
    monkeypatch.setattr(
        "vibespatial.spatial.spatial_index_device.available_device_memory_bytes",
        lambda: 64 * 400 * 4,
    )

    assert _spatial_reduction_tile_lane_capacity(
        object(),
        object(),
        predicate="intersects",
        family_admission=(True, False, False),
    ) == 400


@pytest.mark.parametrize(
    ("profile", "available_bytes", "expected_lanes"),
    [
        ("h100_80gb", 80 * 1024**3, 16 * 1024 * 1024),
        ("a100_40gb", 40 * 1024**3, 16 * 1024 * 1024),
        ("rtx_3090_24gb", 24 * 1024**3, 16 * 1024 * 1024),
        ("constrained", 64 * 400 * 4, 400),
        ("unknown", None, 16 * 1024 * 1024),
    ],
)
def test_spatial_reduction_tile_capacity_safety_profiles(
    monkeypatch,
    profile,
    available_bytes,
    expected_lanes,
) -> None:
    """Named safety profiles affect capacity only through current free bytes."""
    monkeypatch.setattr(
        "vibespatial.spatial.spatial_index_device.available_device_memory_bytes",
        lambda: available_bytes,
    )

    assert _spatial_reduction_tile_lane_capacity(
        object(),
        object(),
        predicate="intersects",
        family_admission=(True, False, False),
    ) == expected_lanes, profile


def test_spatial_reduction_tile_capacity_tracks_segment_pair_shape(monkeypatch) -> None:
    monkeypatch.setattr(
        "vibespatial.spatial.spatial_index_device.available_device_memory_bytes",
        lambda: 8 * 1024**3,
    )
    spans = iter((100, 200))
    monkeypatch.setattr(
        "vibespatial.geometry.owned.ensure_device_geometry_size_bounds",
        lambda *args, **kwargs: next(spans),
    )

    assert _spatial_reduction_tile_lane_capacity(
        object(),
        object(),
        predicate="intersects",
        family_admission=(False, False, True),
    ) == (8 * 1024 * 1024) // (100 * 200)


def test_spatial_reduction_uses_structural_tiles_without_active_row_rounds() -> None:
    source = inspect.getsource(_spatial_index_device_relation_reduction)

    assert "while " not in source
    assert "_filter_predicate_pairs_owned" not in source
    assert "range(0, tree_count" not in source
    assert "_morton_reduction_span_schedule" in source
    assert "for bucket_index" in source
    assert "for query_order_start in range" in source
    assert "for position_start in range" in source
    # Broadcasting the public distance input to query-row shape is bounded;
    # candidate-pair-shaped broadcasting remains forbidden.
    assert source.count("cp.broadcast_to(") == 1
    assert "cp.broadcast_to(raw_distance_thresholds, (query_count,))" in source
    assert 'kernels["morton_range_tile_count"]' in source
    assert 'kernels["morton_range_tile_scatter"]' in source
    assert "logical_count=d_candidate_count_i64" in source
    assert "logical_count=d_candidate_count_i32" in source
    assert "candidate_selection.active_capacity_mask()" in source
    assert "NativeDeviceSelection.from_mask(d_reduced != 0)" in source
    assert source.count("family_partition_type.from_pair_capacity(") == 1
    # The backing grouped relation and shared output/scratch arrays retain the
    # full tile capacity; only each family kernel's admitted launch is smaller.
    assert "pair_capacity=pair_capacity" in source
    assert "launch_capacity=family_launch_capacity" in source
    assert "pair_capacity=family_partition.capacity" not in source
    assert "launch_capacity=partition.capacity" not in source


@requires_gpu
def test_relation_family_partition_groups_mixed_capacity_correctly() -> None:
    cp = pytest.importorskip("cupy")
    from vibespatial.api._native_relation import NativeRelationFamilyPartition
    from vibespatial.geometry.buffers import GeometryFamily

    partition = NativeRelationFamilyPartition.from_pair_capacity(
        cp.asarray([0, 1, 2, 1, 0, 2], dtype=cp.int32),
        cp.asarray([2, 1, 0, 0, 2, 1], dtype=cp.int32),
        cp.asarray([True, True, True, False, True, False]),
        cp.asarray([0, 1, 2], dtype=cp.int8),
        cp.asarray([0, 1, 2], dtype=cp.int8),
        family_count=3,
    )

    point_polygon = partition.family_pair(
        left_family=GeometryFamily.POINT,
        right_family=GeometryFamily.POLYGON,
        left_family_tag=0,
        right_family_tag=2,
        launch_capacity=2,
    )
    logical_count = int(cp.asnumpy(point_polygon.logical_count)[0])
    source_offset = int(cp.asnumpy(point_polygon.source_offset)[0])

    assert partition.capacity == 6
    assert cp.asnumpy(partition.group_counts).tolist() == [0, 0, 2, 0, 1, 0, 1, 0, 0]
    assert logical_count == 2
    assert point_polygon.left_indices is partition.left_indices
    assert point_polygon.right_indices is partition.right_indices
    assert cp.asnumpy(
        point_polygon.left_indices[source_offset : source_offset + logical_count]
    ).tolist() == [0, 0]
    assert cp.asnumpy(
        point_polygon.right_indices[source_offset : source_offset + logical_count]
    ).tolist() == [2, 2]


def test_relation_family_pair_is_metadata_only_and_launches_are_tile_bounded() -> None:
    from vibespatial.api._native_relation import NativeRelationFamilyPartition

    source = inspect.getsource(NativeRelationFamilyPartition.family_pair)
    capacities = _family_group_launch_capacities(1_000_000, 36)

    assert "cp.arange(self.capacity)" not in source
    assert "cp.where" not in source
    assert "active_capacity_mask" not in source
    assert len(capacities) == 36
    assert all(capacity > 0 for capacity in capacities)
    assert sum(capacities) <= 1_000_000 + 36
    assert sum(capacities) < 36 * 1_000_000


@requires_gpu
def test_relation_family_partition_uses_one_pass_at_1m_capacity(monkeypatch) -> None:
    cp = pytest.importorskip("cupy")
    from vibespatial.api._native_relation import NativeRelationFamilyPartition
    from vibespatial.cuda import cccl_primitives
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    sort_calls = 0
    sort_pairs = cccl_primitives.sort_pairs

    def counted_sort_pairs(*args, **kwargs):
        nonlocal sort_calls
        sort_calls += 1
        return sort_pairs(*args, **kwargs)

    monkeypatch.setattr(cccl_primitives, "sort_pairs", counted_sort_pairs)
    capacity = 1_000_000
    d_lanes = cp.arange(capacity, dtype=cp.int32)
    d_left = d_lanes % cp.int32(6)
    d_right = (d_lanes // cp.int32(6)) % cp.int32(6)
    d_active = d_lanes < cp.int32(capacity - 17)
    d_tags = cp.arange(6, dtype=cp.int8)

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    partition = NativeRelationFamilyPartition.from_pair_capacity(
        d_left,
        d_right,
        d_active,
        d_tags,
        d_tags,
        family_count=6,
    )
    events = get_d2h_transfer_events(clear=True)

    assert partition.capacity == capacity
    assert partition.group_offsets.shape == (37,)
    assert partition.group_counts.shape == (36,)
    assert sort_calls == 1
    assert events == []
    assert int(cp.asnumpy(cp.sum(partition.group_counts))) == capacity - 17


@requires_gpu
def test_morton_reduction_span_schedule_bounds_total_capacity() -> None:
    cp = pytest.importorskip("cupy")
    spans = cp.asarray([0, 1, 2, 3, 4, 5, 31, 32, 33, 1_000], dtype=cp.uint64)
    starts = cp.zeros_like(spans)

    order, bucket_counts = _morton_reduction_span_schedule(starts, spans)

    scheduled_lanes = sum(
        int(count) * int(upper)
        for count, upper in zip(
            bucket_counts,
            _MORTON_SPAN_BUCKET_UPPER_BOUNDS,
            strict=True,
        )
    )
    actual_lanes = int(cp.asnumpy(cp.sum(spans)))
    assert scheduled_lanes < 2 * actual_lanes
    ordered_spans = cp.asnumpy(spans[order]).tolist()
    assert ordered_spans == sorted(ordered_spans)


def test_device_memory_probe_propagates_runtime_failure(monkeypatch) -> None:
    pytest.importorskip("cupy")

    def fail_runtime():
        raise RuntimeError("driver memory query failed")

    monkeypatch.setattr(query_types_module, "get_cuda_runtime", fail_runtime)
    with pytest.raises(RuntimeError, match="driver memory query failed"):
        available_device_memory_bytes()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grid_boxes(n_cols: int, n_rows: int) -> np.ndarray:
    """Create a regular grid of unit boxes."""
    geoms = []
    for r in range(n_rows):
        for c in range(n_cols):
            geoms.append(box(c, r, c + 1, r + 1))
    return np.asarray(geoms, dtype=object)


def _make_random_points(n: int, *, seed: int = 42, extent: float = 100.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, extent, size=(n, 2))
    return np.asarray([Point(x, y) for x, y in coords], dtype=object)


def _make_random_boxes(
    n: int, *, seed: int = 42, extent: float = 100.0, size: float = 5.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mins = rng.uniform(0, extent - size, size=(n, 2))
    geoms = []
    for x, y in mins:
        geoms.append(box(x, y, x + size, y + size))
    return np.asarray(geoms, dtype=object)


def _cpu_bbox_pairs(query_bounds: np.ndarray, tree_bounds: np.ndarray):
    """CPU reference: brute-force bbox overlap detection."""
    left = []
    right = []
    for q in range(query_bounds.shape[0]):
        qb = query_bounds[q]
        if np.isnan(qb).any():
            continue
        for t in range(tree_bounds.shape[0]):
            tb = tree_bounds[t]
            if np.isnan(tb).any():
                continue
            if qb[0] <= tb[2] and qb[2] >= tb[0] and qb[1] <= tb[3] and qb[3] >= tb[1]:
                left.append(q)
                right.append(t)
    return np.array(left, dtype=np.int32), np.array(right, dtype=np.int32)


# ---------------------------------------------------------------------------
# Tests: basic correctness
# ---------------------------------------------------------------------------


@requires_gpu
def test_device_query_matches_cpu_brute_force_small():
    """Device query produces identical pairs to CPU brute-force for small input."""
    tree_geoms = _make_grid_boxes(5, 5)  # 25 boxes
    query_geoms = np.asarray([
        box(0.5, 0.5, 2.5, 2.5),  # overlaps several
        box(10, 10, 10.5, 10.5),   # no overlap
        box(3.5, 3.5, 4.5, 4.5),  # corner overlap
    ], dtype=object)

    tree_owned, flat_index = build_owned_spatial_index(tree_geoms)
    query_owned = from_shapely_geometries(query_geoms)
    query_bounds = compute_geometry_bounds(query_owned)
    tree_bounds = flat_index.bounds

    cands, execution = spatial_index_device_query(flat_index, query_bounds)
    assert execution.selected is ExecutionMode.GPU
    assert cands is not None

    gpu_left, gpu_right = cands.to_host()
    cpu_left, cpu_right = _cpu_bbox_pairs(query_bounds, tree_bounds)

    # Sort both for deterministic comparison.
    gpu_pairs = set(zip(gpu_left.tolist(), gpu_right.tolist()))
    cpu_pairs = set(zip(cpu_left.tolist(), cpu_right.tolist()))
    assert gpu_pairs == cpu_pairs, (
        f"GPU pairs != CPU pairs. GPU-only: {gpu_pairs - cpu_pairs}, "
        f"CPU-only: {cpu_pairs - gpu_pairs}"
    )


@requires_gpu
def test_point_grid_superset_is_exactly_refined_before_public_export(monkeypatch):
    """Cell false positives stay device-resident and exact GIS semantics win."""
    import shapely

    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )
    from vibespatial.predicates import point_location_index
    from vibespatial.spatial import point_grid_index

    tree_values = np.asarray(
        [
            Point(0.5, 0.5),
            Point(2.0, 2.0),  # inside the first polygon's hole
            Point(0.0, 2.0),  # on the first polygon's boundary
            Point(5.5, 0.5),
            Point(7.5, 0.5),
            Point(20.0, 20.0),
        ],
        dtype=object,
    )
    query_values = np.asarray(
        [
            Polygon(
                [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)],
                holes=[[(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)]],
            ),
            MultiPolygon([box(5, 0, 6, 1), box(7, 0, 8, 1)]),
            box(-20, -20, -10, -10),
        ],
        dtype=object,
    )
    tree_owned = from_shapely_geometries(tree_values, residency=Residency.DEVICE)
    flat_index = build_flat_spatial_index(
        tree_owned,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            reason="test Native-owned point-grid exact-refinement carrier",
        ),
    )
    query_owned = from_shapely_geometries(
        query_values,
        residency=Residency.DEVICE,
    )
    monkeypatch.setattr(point_grid_index, "_MIN_POINT_GRID_ROWS", 0)
    monkeypatch.setattr(point_location_index, "_MIN_PREPARED_COORDINATES", 0)
    assert flat_index.device_bounds is not None

    reset_d2h_transfer_count()
    result = query_spatial_index(
        tree_owned,
        flat_index,
        query_owned,
        predicate="contains",
    )
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    expected = {
        (query_row, tree_row)
        for query_row, polygon in enumerate(query_values)
        for tree_row, point in enumerate(tree_values)
        if point is not None and shapely.contains(polygon, point)
    }
    actual = set(zip(result[0].tolist(), result[1].tolist(), strict=True))
    assert actual == expected
    # Relation-producing public APIs use the Native-owned grid only when the
    # full pair-shaped output passes memory admission. Exact refinement above
    # remains the source of GIS truth.
    assert flat_index.point_grid is None
    native_index = flat_index.to_native_spatial_index()
    assert {
        key.variant for key in native_index.point_partition_cache
    } == {point_grid_index.PointPartitionVariant.GRID}
    assert set(query_owned.device_state.point_location_indexes) == {
        GeometryFamily.POLYGON,
        GeometryFamily.MULTIPOLYGON,
    }
    assert "point-grid relation candidate allocation fence" in reasons
    assert not any("candidate" in reason and "host export" in reason for reason in reasons)
    assert "runtime.synchronize" not in inspect.getsource(
        point_grid_index.point_grid_superset_query
    )


@requires_gpu
def test_point_grid_superset_is_not_used_for_bbox_only_queries(monkeypatch):
    """Public predicate=None queries return exact bbox hits, never cell supersets."""
    from vibespatial.spatial import point_grid_index

    tree_values = np.asarray(
        [Point(0.10, 0.10), Point(0.90, 0.90), Point(10.0, 10.0)],
        dtype=object,
    )
    tree_owned = from_shapely_geometries(tree_values, residency=Residency.DEVICE)
    flat_index = build_flat_spatial_index(
        tree_owned,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.CPU,
            selected=ExecutionMode.CPU,
            reason="test bbox-only point-grid exclusion",
        ),
    )
    query_owned = from_shapely_geometries(
        np.asarray([box(0.0, 0.0, 0.2, 0.2)], dtype=object),
        residency=Residency.DEVICE,
    )
    monkeypatch.setattr(point_grid_index, "_MIN_POINT_GRID_ROWS", 0)

    result = query_spatial_index(
        tree_owned,
        flat_index,
        query_owned,
        predicate=None,
    )

    assert result.tolist() == [[0], [0]]
    assert flat_index.point_grid is None


@requires_gpu
def test_native_relation_grid_reuses_original_carrier_and_propagates_scatter_fault(
    monkeypatch,
) -> None:
    import cupy as cp

    from vibespatial.cuda._runtime import get_cuda_runtime
    from vibespatial.spatial import point_grid_index, spatial_index_device

    tree_owned = from_shapely_geometries(
        np.asarray([Point(0.25, 0.25), Point(0.75, 0.75)], dtype=object),
        residency=Residency.DEVICE,
    )
    flat_index = build_flat_spatial_index(
        tree_owned,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            reason="native relation carrier ownership test",
        ),
    )
    native_index = flat_index.to_native_spatial_index(source_token="tree-lineage")
    query_owned = from_shapely_geometries(
        np.asarray(
            [Polygon([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)])],
            dtype=object,
        ),
        residency=Residency.DEVICE,
    )
    monkeypatch.setattr(point_grid_index, "_MIN_POINT_GRID_ROWS", 0)
    runtime = get_cuda_runtime()
    original_admit = runtime.admit_device_memory
    relation_admissions = []

    def _record_relation_admission(*, stage, required_bytes, requested_units=0):
        if stage == "spatial.point_grid_relation_complete":
            relation_admissions.append((required_bytes, requested_units))
        return original_admit(
            stage=stage,
            required_bytes=required_bytes,
            requested_units=requested_units,
        )

    monkeypatch.setattr(runtime, "admit_device_memory", _record_relation_admission)
    original_superset = point_grid_index.point_grid_superset_query

    def _faulted_superset(*args, **kwargs):
        candidates = original_superset(*args, **kwargs)
        assert candidates is not None
        return replace(candidates, error_flag=cp.ones(1, dtype=cp.uint32))

    monkeypatch.setattr(point_grid_index, "point_grid_superset_query", _faulted_superset)

    def _no_morton_retry(*_args, **_kwargs):
        raise AssertionError("scatter faults must not retry Morton")

    monkeypatch.setattr(
        spatial_index_device,
        "_prepare_morton_range_query",
        _no_morton_retry,
    )
    with pytest.raises(RuntimeError, match="sealed capacity"):
        native_index.query_relation(query_owned, predicate="contains")

    assert flat_index._native_spatial_index is native_index
    assert native_index.source_token == "tree-lineage"
    assert len(native_index.point_partition_cache) == 1
    assert len(relation_admissions) == 1
    relation_bytes, relation_pairs = relation_admissions[0]
    assert relation_pairs == 2
    assert relation_bytes >= relation_pairs * 73


@requires_gpu
def test_native_relation_grid_complete_admission_decline_is_not_retried(
    monkeypatch,
) -> None:
    from vibespatial.cuda._runtime import DeviceMemoryAdmission, get_cuda_runtime
    from vibespatial.spatial import point_grid_index, spatial_index_device
    from vibespatial.spatial.query_types import CandidateRelationCapacityError

    tree_owned = from_shapely_geometries(
        np.asarray([Point(0.25, 0.25), Point(0.75, 0.75)], dtype=object),
        residency=Residency.DEVICE,
    )
    native_index = build_flat_spatial_index(
        tree_owned,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            reason="relation complete-admission no-retry test",
        ),
    ).to_native_spatial_index()
    query_owned = from_shapely_geometries(
        np.asarray(
            [Polygon([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)])],
            dtype=object,
        ),
        residency=Residency.DEVICE,
    )
    monkeypatch.setattr(point_grid_index, "_MIN_POINT_GRID_ROWS", 0)
    runtime = get_cuda_runtime()
    original_admit = runtime.admit_device_memory

    def _decline_complete_relation(*, stage, required_bytes, requested_units=0):
        if stage == "spatial.point_grid_relation_complete":
            return DeviceMemoryAdmission(
                stage=stage,
                required_bytes=required_bytes,
                remaining_bytes=0,
                budget_bytes=0,
                admitted=False,
                requested_units=requested_units,
                admitted_units=0,
                bytes_per_unit=required_bytes,
            )
        return original_admit(
            stage=stage,
            required_bytes=required_bytes,
            requested_units=requested_units,
        )

    monkeypatch.setattr(runtime, "admit_device_memory", _decline_complete_relation)

    def _no_morton_retry(*_args, **_kwargs):
        raise AssertionError("complete relation admission must not retry Morton")

    monkeypatch.setattr(
        spatial_index_device,
        "_prepare_morton_range_query",
        _no_morton_retry,
    )
    with pytest.raises(
        CandidateRelationCapacityError,
        match="refusing a post-submission Morton retry",
    ):
        native_index.query_relation(query_owned, predicate="contains")


def test_point_grid_memory_estimate_covers_bounds_and_scales_with_cells() -> None:
    from vibespatial.spatial.point_grid_index import (
        _point_grid_required_bytes,
        prepare_point_grid_index,
    )

    with_bounds = _point_grid_required_bytes(
        100_000,
        16_384,
        needs_device_bounds=True,
    )
    without_bounds = _point_grid_required_bytes(
        100_000,
        16_384,
        needs_device_bounds=False,
    )

    assert with_bounds > 100_000 * 65 + 16_384 * 32
    assert with_bounds > without_bounds
    assert _point_grid_required_bytes(
        100_000,
        65_536,
        needs_device_bounds=True,
    ) > with_bounds
    assert _point_grid_required_bytes(
        100_000,
        16_384,
        needs_device_bounds=False,
        query_count=8_192,
        pair_budget=1_000_000,
    ) > without_bounds
    assert "bincount" not in inspect.getsource(prepare_point_grid_index)


@requires_gpu
def test_point_grid_preflight_declines_finite_overflowing_tree_extents(
    monkeypatch,
) -> None:
    import cupy as cp

    from vibespatial.spatial import point_grid_index
    from vibespatial.spatial.indexing import FlatSpatialIndex

    owned = from_shapely_geometries(
        np.asarray(
            [Point(-1.0e308, -1.0e308), Point(1.0e308, 1.0e308)],
            dtype=object,
        ),
        residency=Residency.DEVICE,
    )
    flat_index = FlatSpatialIndex(
        geometry_array=owned,
        _host_order=None,
        _host_morton_keys=None,
        _host_bounds=None,
        total_bounds=(-1.0e308, -1.0e308, 1.0e308, 1.0e308),
        device_order=cp.arange(2, dtype=cp.int32),
        device_morton_keys=cp.arange(2, dtype=cp.uint64),
        device_bounds=cp.asarray(
            [
                [-1.0e308, -1.0e308, -1.0e308, -1.0e308],
                [1.0e308, 1.0e308, 1.0e308, 1.0e308],
            ],
            dtype=cp.float64,
        ),
    )
    native_index = flat_index.to_native_spatial_index()
    monkeypatch.setattr(point_grid_index, "_MIN_POINT_GRID_ROWS", 0)

    preflight, decline = point_grid_index.point_grid_preflight(
        native_index,
        query_count=1,
        pair_budget=2,
        force_eligible=True,
    )

    assert preflight is None
    assert decline is not None
    assert "finite positive x and y extents" in decline.reason
    assert native_index.point_partition_cache == {}


@requires_gpu
def test_point_grid_clamps_extreme_finite_query_before_normalization(
    monkeypatch,
) -> None:
    import cupy as cp

    from vibespatial.spatial import point_grid_index
    from vibespatial.spatial.indexing import FlatSpatialIndex

    owned = from_shapely_geometries(
        np.asarray(
            [Point(-1.0e308, -1.0e308), Point(-9.0e307, -9.0e307)],
            dtype=object,
        ),
        residency=Residency.DEVICE,
    )
    flat_index = FlatSpatialIndex(
        geometry_array=owned,
        _host_order=None,
        _host_morton_keys=None,
        _host_bounds=None,
        total_bounds=(-1.0e308, -1.0e308, -9.0e307, -9.0e307),
        device_order=cp.arange(2, dtype=cp.int32),
        device_morton_keys=cp.arange(2, dtype=cp.uint64),
        device_bounds=cp.asarray(
            [
                [-1.0e308, -1.0e308, -1.0e308, -1.0e308],
                [-9.0e307, -9.0e307, -9.0e307, -9.0e307],
            ],
            dtype=cp.float64,
        ),
    )
    native_index = flat_index.to_native_spatial_index()
    monkeypatch.setattr(point_grid_index, "_MIN_POINT_GRID_ROWS", 0)
    query_bounds = cp.asarray(
        [[-1.0e308, -1.0e308, 1.0e308, 1.0e308]],
        dtype=cp.float64,
    )

    plan, decline = point_grid_index.point_grid_query_row_partitions(
        native_index,
        query_bounds,
        pair_budget=2,
        force_eligible=True,
    )

    assert decline is None
    assert plan is not None
    assert cp.asnumpy(plan.query_counts).tolist() == [2]
    candidates = point_grid_index.point_grid_superset_query(
        native_index,
        next(plan.slices()),
    )
    assert candidates is not None
    candidates.validate_error_flag()
    assert sorted(cp.asnumpy(candidates.d_right).tolist()) == [0, 1]


@requires_gpu
def test_automatic_point_partition_keeps_fully_admitted_grid(monkeypatch) -> None:
    import cupy as cp

    from vibespatial.spatial import point_grid_index
    from vibespatial.spatial.point_partition import PointPartitionVariant
    from vibespatial.spatial.spatial_index_device import (
        _point_partition_reduction_plan,
    )

    tree_owned = from_shapely_geometries(
        np.asarray([Point(0.0, 0.0), Point(1.0, 1.0)], dtype=object),
        residency=Residency.DEVICE,
    )
    flat_index = build_flat_spatial_index(
        tree_owned,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            reason="admitted point-grid selector test",
        ),
    )
    native_index = flat_index.to_native_spatial_index()
    monkeypatch.setattr(point_grid_index, "_MIN_POINT_GRID_ROWS", 0)

    plan, decline = _point_partition_reduction_plan(
        native_index,
        cp.asarray([[-1.0, -1.0, 2.0, 2.0]], dtype=cp.float64),
        predicate="contains",
        query_family=GeometryFamily.POLYGON,
        tree_family=GeometryFamily.POINT,
        reduction="right_pair_count",
        pair_budget=16,
    )

    assert decline is None
    assert plan is not None
    assert plan.variant is PointPartitionVariant.GRID
    assert {
        key.variant for key in native_index.point_partition_cache
    } == {PointPartitionVariant.GRID}


@requires_gpu
def test_point_partition_admission_rejects_cache_eviction_growth(
    monkeypatch,
) -> None:
    from vibespatial.spatial import point_grid_index
    from vibespatial.spatial.point_partition import PointPartitionVariant
    from vibespatial.spatial.spatial_index_device import (
        _point_partition_preflight_selection,
    )

    tree_owned = from_shapely_geometries(
        np.asarray([Point(0.0, 0.0), Point(1.0, 1.0)], dtype=object),
        residency=Residency.DEVICE,
    )
    native_index = build_flat_spatial_index(
        tree_owned,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            reason="cache-eviction admission growth test",
        ),
    ).to_native_spatial_index()
    from vibespatial.spatial.point_grid_index import prepare_point_grid_index

    selected = PointPartitionVariant.GRID
    monkeypatch.setattr(point_grid_index, "_MIN_POINT_GRID_ROWS", 0)

    def prepare(admission=None):
        return prepare_point_grid_index(
            native_index,
            query_count=1,
            pair_budget=16,
            force_eligible=True,
            admission=admission,
        )

    prepared, decline = prepare()
    assert decline is None and prepared is not None
    selection, decline = _point_partition_preflight_selection(
        native_index,
        query_count=1,
        pair_budget=16,
        forced=selected,
    )
    assert decline is None and selection is not None
    native_index.point_partition_cache.clear()

    with pytest.raises(ValueError, match="admission token provenance mismatch"):
        prepare(selection)


@requires_gpu
def test_point_partition_large_query_count_uses_bounded_block_packet() -> None:
    import cupy as cp

    from vibespatial.spatial.point_grid_index import (
        point_grid_query_row_partitions,
    )

    tree_owned = from_shapely_geometries(
        np.asarray([Point(0.0, 0.0), Point(1.0, 1.0)], dtype=object),
        residency=Residency.DEVICE,
    )
    native_index = build_flat_spatial_index(
        tree_owned,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            reason="large point-partition query-count test",
        ),
    ).to_native_spatial_index()
    query_bounds = cp.tile(
        cp.asarray([[2.0, 2.0, 3.0, 3.0]], dtype=cp.float64),
        (9_000, 1),
    )

    plan, decline = point_grid_query_row_partitions(
        native_index,
        query_bounds,
        pair_budget=16,
        force_eligible=True,
    )

    assert decline is None
    assert plan is not None
    assert plan.query_counts.shape == (9_000,)
    assert sum(stop - start for start, stop, _capacity in plan.partitions) == 9_000


@pytest.mark.parametrize(
    ("predicate", "query_family", "tree_family", "reduction"),
    [
        (None, GeometryFamily.POLYGON, GeometryFamily.POINT, "right_count"),
        ("disjoint", GeometryFamily.POLYGON, GeometryFamily.POINT, "right_count"),
        ("dwithin", GeometryFamily.POLYGON, GeometryFamily.POINT, "right_count"),
        ("within", GeometryFamily.POLYGON, GeometryFamily.POINT, "right_count"),
        ("contains", GeometryFamily.LINESTRING, GeometryFamily.POINT, "right_count"),
        ("contains", GeometryFamily.POLYGON, GeometryFamily.MULTIPOINT, "right_count"),
        ("contains", GeometryFamily.POLYGON, GeometryFamily.POINT, "right_exists"),
    ],
)
def test_point_partition_selector_declines_every_out_of_scope_shape(
    predicate,
    query_family,
    tree_family,
    reduction,
) -> None:
    from vibespatial.spatial.spatial_index_device import (
        _point_partition_reduction_plan,
    )

    plan, decline = _point_partition_reduction_plan(
        object(),
        np.empty((0, 4), dtype=np.float64),
        predicate=predicate,
        query_family=query_family,
        tree_family=tree_family,
        reduction=reduction,
        pair_budget=16,
    )

    assert plan is None
    assert decline is None


@requires_gpu
def test_point_partition_query_slice_seal_rejects_modified_provenance() -> None:
    import cupy as cp

    from vibespatial.spatial.point_partition import (
        PointPartitionVariant,
        query_plan,
    )

    owner = object()
    prepared = SimpleNamespace(cache_key=object())
    plan = query_plan(
        owner=owner,
        variant=PointPartitionVariant.GRID,
        prepared=prepared,
        query_bounds=cp.zeros((1, 4), dtype=cp.float64),
        query_counts=cp.ones(1, dtype=cp.int64),
        partitions=((0, 1, 1),),
        pair_budget=1,
    )
    plan.slices().__next__().validate(owner, PointPartitionVariant.GRID, prepared)

    modified = replace(plan, partitions=((0, 1, 0),))
    with pytest.raises(ValueError, match="provenance was modified"):
        modified.slices().__next__().validate(
            owner,
            PointPartitionVariant.GRID,
            prepared,
        )


@requires_gpu
def test_point_partition_exclusion_rejects_modified_plan(
    monkeypatch,
) -> None:
    import cupy as cp

    from vibespatial.spatial.point_partition import PointPartitionVariant, query_plan

    selected = PointPartitionVariant.GRID
    tree_owned = from_shapely_geometries(
        np.asarray([Point(0.0, 0.0), Point(1.0, 1.0)], dtype=object),
        residency=Residency.DEVICE,
    )
    native_index = build_flat_spatial_index(
        tree_owned,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            reason="point-partition exclusion seal test",
        ),
    ).to_native_spatial_index()
    from vibespatial.spatial import point_grid_index as provider_module
    from vibespatial.spatial.point_grid_index import (
        point_grid_candidate_not_in_other_superset as exclude,
    )
    from vibespatial.spatial.point_grid_index import prepare_point_grid_index

    prepared, decline = prepare_point_grid_index(
        native_index,
        query_count=1,
        pair_budget=2,
        force_eligible=True,
    )
    assert decline is None
    plan = query_plan(
        owner=native_index,
        variant=selected,
        prepared=prepared,
        query_bounds=cp.asarray([[-1.0, -1.0, 2.0, 2.0]], dtype=cp.float64),
        query_counts=cp.asarray([2], dtype=cp.int64),
        partitions=((0, 1, 2),),
        pair_budget=2,
    )
    def _unexpected_reprepare(*_args, **_kwargs):
        raise AssertionError("sealed exclusion plans must reuse prepared state")

    monkeypatch.setattr(
        provider_module,
        "prepare_point_grid_index",
        _unexpected_reprepare,
    )
    valid = exclude(
        native_index,
        plan,
        cp.asarray([0], dtype=cp.int32),
        cp.asarray([0], dtype=cp.int32),
    )
    assert valid is not None
    modified = replace(plan, query_counts=plan.query_counts.copy())
    with pytest.raises(ValueError, match="provenance was modified"):
        exclude(
            native_index,
            modified,
            cp.asarray([0], dtype=cp.int32),
            cp.asarray([0], dtype=cp.int32),
        )


@requires_gpu
def test_paired_provider_decline_happens_before_either_provider_submits(
    monkeypatch,
) -> None:
    import cupy as cp

    from vibespatial.cuda._runtime import get_cuda_runtime
    from vibespatial.spatial import point_grid_index
    from vibespatial.spatial.point_partition import (
        PointPartitionDecline,
        PointPartitionVariant,
    )
    from vibespatial.spatial.spatial_index_device import (
        _paired_point_partition_preflight_selections,
    )

    def _native(points, reason):
        owned = from_shapely_geometries(
            np.asarray(points, dtype=object),
            residency=Residency.DEVICE,
        )
        return build_flat_spatial_index(
            owned,
            runtime_selection=RuntimeSelection(
                requested=ExecutionMode.GPU,
                selected=ExecutionMode.GPU,
                reason=reason,
            ),
        ).to_native_spatial_index()

    left = _native([Point(0.0, 0.0), Point(1.0, 1.0)], "paired-left")
    right = _native([Point(0.0, 1.0), Point(1.0, 0.0)], "paired-right")
    monkeypatch.setattr(point_grid_index, "_MIN_POINT_GRID_ROWS", 0)
    original_preflight = point_grid_index.point_grid_preflight

    def _decline_aligned(native_index, **kwargs):
        if native_index is right:
            return None, PointPartitionDecline(
                PointPartitionVariant.GRID,
                "injected aligned structural decline",
            )
        return original_preflight(native_index, **kwargs)

    monkeypatch.setattr(point_grid_index, "point_grid_preflight", _decline_aligned)
    runtime = get_cuda_runtime()
    original_launch = runtime.launch
    provider_launches = 0

    def _count_provider_launches(kernel, **kwargs):
        nonlocal provider_launches
        if kernel.name.startswith("point_grid"):
            provider_launches += 1
        return original_launch(kernel, **kwargs)

    monkeypatch.setattr(runtime, "launch", _count_provider_launches)
    left_selection, right_selection, decline = (
        _paired_point_partition_preflight_selections(
            left,
            right,
            query_count=1,
            pair_budget=16,
            forced=None,
        )
    )
    assert left_selection is None and right_selection is None
    assert decline is not None
    assert provider_launches == 0
    cp.cuda.get_current_stream().synchronize()


def test_same_owner_grid_cache_miss_serializes_concurrent_builders(monkeypatch) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import RLock
    from time import sleep

    from vibespatial.spatial import point_grid_index

    native_index = SimpleNamespace(point_partition_lock=RLock())
    active_builders = 0
    maximum_active_builders = 0

    def _record_locked_build(*_args, **_kwargs):
        nonlocal active_builders, maximum_active_builders
        active_builders += 1
        maximum_active_builders = max(maximum_active_builders, active_builders)
        sleep(0.01)
        active_builders -= 1
        return object(), None

    monkeypatch.setattr(
        point_grid_index,
        "_prepare_point_grid_index_locked",
        _record_locked_build,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(
            executor.map(
                lambda _ordinal: point_grid_index.prepare_point_grid_index(
                    native_index,
                    query_count=1,
                    pair_budget=1,
                    force_eligible=True,
                ),
                range(2),
            )
        )
    assert len(results) == 2
    assert maximum_active_builders == 1


@requires_gpu
def test_point_partition_scatter_capacity_guard_sets_device_fault(
    monkeypatch,
) -> None:
    import cupy as cp

    from vibespatial.spatial.point_partition import (
        PointPartitionVariant,
        query_plan,
    )

    tree_owned = from_shapely_geometries(
        np.asarray([Point(0.0, 0.0), Point(1.0, 1.0)], dtype=object),
        residency=Residency.DEVICE,
    )
    flat_index = build_flat_spatial_index(
        tree_owned,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            reason="point-partition capacity-guard test",
        ),
    )
    native_index = flat_index.to_native_spatial_index()
    bounds = cp.asarray([[-1.0, -1.0, 2.0, 2.0]], dtype=cp.float64)
    counts = cp.asarray([2], dtype=cp.int64)
    selected = PointPartitionVariant.GRID
    from vibespatial.spatial import point_grid_index as provider_module
    from vibespatial.spatial.point_grid_index import (
        point_grid_superset_query as superset_query,
    )
    from vibespatial.spatial.point_grid_index import prepare_point_grid_index

    prepared, decline = prepare_point_grid_index(
        native_index,
        query_count=1,
        pair_budget=1,
        force_eligible=True,
    )
    assert decline is None
    plan = query_plan(
        owner=native_index,
        variant=selected,
        prepared=prepared,
        query_bounds=bounds,
        query_counts=counts,
        partitions=((0, 1, 1),),
        pair_budget=1,
    )

    def _unexpected_reprepare(*_args, **_kwargs):
        raise AssertionError("a sealed query slice must reuse its prepared owner")

    monkeypatch.setattr(
        provider_module,
        "prepare_point_grid_index",
        _unexpected_reprepare,
    )

    candidates = superset_query(native_index, next(plan.slices()))

    assert candidates is not None
    assert candidates.total_pairs == 1
    assert int(cp.asnumpy(candidates.error_flag)[0]) == 1


@requires_gpu
def test_point_partition_post_submission_fault_is_not_retried(monkeypatch) -> None:
    import cupy as cp

    from vibespatial.cuda._runtime import get_cuda_runtime
    from vibespatial.spatial.point_grid_index import (
        point_grid_query_row_partitions,
        point_grid_superset_query,
    )

    tree_owned = from_shapely_geometries(
        np.asarray([Point(0.0, 0.0), Point(1.0, 1.0)], dtype=object),
        residency=Residency.DEVICE,
    )
    flat_index = build_flat_spatial_index(
        tree_owned,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            reason="point-partition post-submission fault test",
        ),
    )
    native_index = flat_index.to_native_spatial_index()
    plan, decline = point_grid_query_row_partitions(
        native_index,
        cp.asarray([[-1.0, -1.0, 2.0, 2.0]], dtype=cp.float64),
        pair_budget=16,
        force_eligible=True,
    )
    assert decline is None
    assert plan is not None
    runtime = get_cuda_runtime()
    original_launch = runtime.launch
    scatter_submissions = 0

    def fail_after_scatter_submission(kernel, **kwargs):
        nonlocal scatter_submissions
        original_launch(kernel, **kwargs)
        if kernel.name == "point_grid_query_scatter":
            scatter_submissions += 1
            raise RuntimeError("injected post-submission point-grid fault")

    monkeypatch.setattr(runtime, "launch", fail_after_scatter_submission)

    with pytest.raises(RuntimeError, match="post-submission point-grid fault"):
        point_grid_superset_query(native_index, next(plan.slices()))
    assert scatter_submissions == 1


@requires_gpu
def test_device_query_matches_cpu_brute_force_medium():
    """Device query matches CPU for 100x100 (10K N*M) input."""
    tree_geoms = _make_random_boxes(100, seed=1, extent=50.0, size=3.0)
    query_geoms = _make_random_boxes(100, seed=2, extent=50.0, size=3.0)

    tree_owned, flat_index = build_owned_spatial_index(tree_geoms)
    query_owned = from_shapely_geometries(query_geoms)
    query_bounds = compute_geometry_bounds(query_owned)
    tree_bounds = flat_index.bounds

    cands, execution = spatial_index_device_query(flat_index, query_bounds)
    assert execution.selected is ExecutionMode.GPU
    assert cands is not None

    gpu_left, gpu_right = cands.to_host()
    cpu_left, cpu_right = _cpu_bbox_pairs(query_bounds, tree_bounds)

    gpu_pairs = set(zip(gpu_left.tolist(), gpu_right.tolist()))
    cpu_pairs = set(zip(cpu_left.tolist(), cpu_right.tolist()))
    assert gpu_pairs == cpu_pairs


@requires_gpu
def test_device_query_no_candidates_returns_empty_device_candidates():
    """When GPU runs and no bbox overlaps exist, result stays device-resident."""
    tree_geoms = np.asarray([box(0, 0, 1, 1)], dtype=object)
    query_geoms = np.asarray([box(100, 100, 101, 101)], dtype=object)

    tree_owned, flat_index = build_owned_spatial_index(tree_geoms)
    query_owned = from_shapely_geometries(query_geoms)
    query_bounds = compute_geometry_bounds(query_owned)

    cands, execution = spatial_index_device_query(flat_index, query_bounds)
    assert execution.selected is ExecutionMode.GPU
    assert cands is not None
    assert cands.total_pairs == 0
    gpu_left, gpu_right = cands.to_host()
    assert gpu_left.size == 0
    assert gpu_right.size == 0


@requires_gpu
def test_device_query_empty_inputs():
    """Empty query inputs preserve the selected GPU path with empty candidates."""
    tree_geoms = np.asarray([box(0, 0, 1, 1)], dtype=object)
    tree_owned, flat_index = build_owned_spatial_index(tree_geoms)

    empty_bounds = np.empty((0, 4), dtype=np.float64)
    cands, execution = spatial_index_device_query(flat_index, empty_bounds)
    assert execution.selected is ExecutionMode.GPU
    assert cands is not None
    assert cands.total_pairs == 0


@requires_gpu
def test_device_query_scalar_single_query():
    """Single query row uses scalar fast path."""
    tree_geoms = _make_grid_boxes(10, 10)  # 100 boxes
    query_geoms = np.asarray([box(2.5, 2.5, 7.5, 7.5)], dtype=object)

    tree_owned, flat_index = build_owned_spatial_index(tree_geoms)
    query_owned = from_shapely_geometries(query_geoms)
    query_bounds = compute_geometry_bounds(query_owned)
    tree_bounds = flat_index.bounds

    cands, execution = spatial_index_device_query(flat_index, query_bounds)
    assert execution.selected is ExecutionMode.GPU
    assert cands is not None

    gpu_left, gpu_right = cands.to_host()
    cpu_left, cpu_right = _cpu_bbox_pairs(query_bounds, tree_bounds)

    gpu_pairs = set(zip(gpu_left.tolist(), gpu_right.tolist()))
    cpu_pairs = set(zip(cpu_left.tolist(), cpu_right.tolist()))
    assert gpu_pairs == cpu_pairs


# ---------------------------------------------------------------------------
# Tests: dwithin distance expansion
# ---------------------------------------------------------------------------


@requires_gpu
def test_device_query_with_distance_expansion():
    """Distance parameter expands query bounds for dwithin candidates."""
    tree_geoms = np.asarray([box(0, 0, 1, 1), box(5, 5, 6, 6)], dtype=object)
    query_geoms = np.asarray([box(2, 2, 3, 3)], dtype=object)

    tree_owned, flat_index = build_owned_spatial_index(tree_geoms)
    query_owned = from_shapely_geometries(query_geoms)
    query_bounds = compute_geometry_bounds(query_owned)

    # Without distance: only close box might overlap.
    cands_no_dist, _ = spatial_index_device_query(flat_index, query_bounds)

    # With large distance: both boxes should be candidates.
    distances = np.array([3.0], dtype=np.float64)
    cands_dist, exec_dist = spatial_index_device_query(
        flat_index, query_bounds, distance=distances,
    )
    assert exec_dist.selected is ExecutionMode.GPU
    assert cands_dist is not None
    gpu_left, gpu_right = cands_dist.to_host()
    # With distance=3.0, query box [2,2,3,3] expands to [-1,-1,6,6]
    # which should overlap both tree boxes.
    assert gpu_right.size == 2, f"Expected 2 candidates, got {gpu_right.size}"


@requires_gpu
def test_device_query_accepts_device_bounds_without_d2h(strict_device_guard):
    """Device-resident query bounds stay on device through candidate generation."""
    tree_geoms = np.asarray([box(0, 0, 1, 1), box(5, 5, 6, 6)], dtype=object)
    query_geoms = [box(2, 2, 3, 3)]

    tree_owned, flat_index = build_owned_spatial_index(tree_geoms)
    query_owned = from_shapely_geometries(query_geoms, residency=Residency.DEVICE)

    query_bounds = compute_geometry_bounds_device(query_owned)
    cands, execution = spatial_index_device_query(
        flat_index,
        query_bounds,
        distance=np.asarray([3.0], dtype=np.float64),
    )

    assert execution.selected is ExecutionMode.GPU
    assert cands is not None
    assert hasattr(query_bounds, "__cuda_array_interface__")
    gpu_left, gpu_right = cands.to_host()
    assert gpu_left.tolist() == [0, 0]
    assert gpu_right.tolist() == [0, 1]


# ---------------------------------------------------------------------------
# Tests: Morton range strategy selection
# ---------------------------------------------------------------------------


@requires_gpu
def test_device_query_uses_morton_range_for_large_input():
    """For large N*M, Morton range strategy is selected (detectable via execution reason)."""
    # Create enough geometries to exceed the Morton range crossover (1M).
    # 1000 x 1000 = 1M.
    tree_geoms = _make_random_boxes(1000, seed=10, extent=200.0, size=2.0)
    query_geoms = _make_random_boxes(1000, seed=11, extent=200.0, size=2.0)

    tree_owned, flat_index = build_owned_spatial_index(tree_geoms)
    query_owned = from_shapely_geometries(query_geoms)
    query_bounds = compute_geometry_bounds(query_owned)
    tree_bounds = flat_index.bounds

    cands, execution = spatial_index_device_query(flat_index, query_bounds)
    assert execution.selected is ExecutionMode.GPU
    assert cands is not None

    # Verify correctness against CPU reference.
    gpu_left, gpu_right = cands.to_host()
    cpu_left, cpu_right = _cpu_bbox_pairs(query_bounds, tree_bounds)

    gpu_pairs = set(zip(gpu_left.tolist(), gpu_right.tolist()))
    cpu_pairs = set(zip(cpu_left.tolist(), cpu_right.tolist()))
    # Morton range may produce a superset (false positives are acceptable
    # since they get refined by predicate evaluation), but must not have
    # false negatives.
    assert cpu_pairs.issubset(gpu_pairs), (
        f"GPU Morton range has false negatives: {cpu_pairs - gpu_pairs}"
    )
    # Verify execution reason mentions Morton range.
    assert "Morton" in execution.reason or "brute" in execution.reason


@requires_gpu
def test_device_query_hydrates_host_index_for_morton_range():
    """A reusable host-built index keeps indexed query shape on the GPU."""
    tree_geoms = _make_random_boxes(1000, seed=110, extent=200.0, size=2.0)
    query_geoms = _make_random_boxes(1000, seed=111, extent=200.0, size=2.0)
    tree_owned = from_shapely_geometries(tree_geoms)
    flat_index = build_flat_spatial_index(
        tree_owned,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.CPU,
            selected=ExecutionMode.CPU,
            reason="test host-built reusable spatial index",
        ),
    )
    query_owned = from_shapely_geometries(query_geoms, residency=Residency.DEVICE)
    query_bounds = compute_geometry_bounds_device(query_owned)

    assert flat_index.device_morton_keys is None
    candidates, execution = spatial_index_device_query(flat_index, query_bounds)

    assert candidates is not None
    assert "Morton range" in execution.reason
    assert flat_index.device_bounds is not None
    assert flat_index.device_morton_keys is not None
    assert flat_index.device_order is not None
    gpu_left, gpu_right = candidates.to_host()
    cpu_left, cpu_right = _cpu_bbox_pairs(query_bounds.get(), flat_index.bounds)
    assert set(zip(cpu_left.tolist(), cpu_right.tolist())).issubset(
        set(zip(gpu_left.tolist(), gpu_right.tolist()))
    )


@requires_gpu
def test_device_query_does_not_treat_regular_grid_identity_as_morton_keys():
    """Regular-grid identity placeholders are not valid Morton codes."""
    tree_geoms = np.asarray(
        [
            box(float(col), float(row), float(col + 1), float(row + 1))
            for row in range(100)
            for col in range(100)
        ],
        dtype=object,
    )
    tree_owned = from_shapely_geometries(tree_geoms)
    flat_index = build_flat_spatial_index(
        tree_owned,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.CPU,
            selected=ExecutionMode.CPU,
            reason="test host-built regular-grid spatial index",
        ),
    )
    query_bounds = np.repeat(
        np.asarray([[40.25, 40.25, 47.75, 47.75]], dtype=np.float64),
        101,
        axis=0,
    )

    assert flat_index.regular_grid is not None
    candidates, execution = spatial_index_device_query(flat_index, query_bounds)

    assert candidates is not None
    assert "brute-force" in execution.reason
    gpu_left, gpu_right = candidates.to_host()
    cpu_left, cpu_right = _cpu_bbox_pairs(query_bounds, flat_index.bounds)
    assert set(zip(gpu_left.tolist(), gpu_right.tolist())) == set(
        zip(cpu_left.tolist(), cpu_right.tolist())
    )


@requires_gpu
def test_device_morton_dwithin_refines_with_expanded_bounds():
    """Morton dwithin retains near rows whose original bboxes do not overlap."""
    tree_geoms = np.asarray(
        [box(float(i), 0.0, float(i) + 0.1, 0.1) for i in range(1000)],
        dtype=object,
    )
    query_geoms = np.asarray(
        [box(float(i) + 0.15, 0.0, float(i) + 0.2, 0.1) for i in range(1000)],
        dtype=object,
    )
    tree_owned = from_shapely_geometries(tree_geoms)
    flat_index = build_flat_spatial_index(
        tree_owned,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.CPU,
            selected=ExecutionMode.CPU,
            reason="test host-built Morton dwithin index",
        ),
    )
    query_owned = from_shapely_geometries(query_geoms, residency=Residency.DEVICE)
    query_bounds = compute_geometry_bounds_device(query_owned)

    candidates, execution = spatial_index_device_query(
        flat_index,
        query_bounds,
        distance=0.1,
    )

    assert candidates is not None
    assert "Morton range" in execution.reason
    left, right = candidates.to_host()
    pairs = set(zip(left.tolist(), right.tolist()))
    assert {(index, index) for index in range(1000)} <= pairs


@requires_gpu
def test_empty_morton_query_cleanup_does_not_context_synchronize(monkeypatch):
    """Morton temporaries retire on-stream when no candidate pairs are emitted."""
    from unittest.mock import Mock

    from vibespatial.cuda._runtime import get_cuda_runtime

    tree_geoms = np.asarray(
        [
            box(0.0, 0.0, 1.0, 1.0),
            box(2.0, 2.0, 3.0, 3.0),
            box(4.0, 4.0, 5.0, 5.0),
        ],
        dtype=object,
    )
    tree_owned = from_shapely_geometries(tree_geoms, residency=Residency.DEVICE)
    flat_index = build_flat_spatial_index(
        tree_owned,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            reason="test Morton cleanup deferred-free ownership",
        ),
    )
    query_bounds = np.asarray([[100.0, 100.0, 101.0, 101.0]], dtype=np.float64)

    runtime = get_cuda_runtime()
    guarded_runtime = Mock(wraps=runtime)
    guarded_runtime.synchronize.side_effect = AssertionError(
        "Morton cleanup must use stream-ordered deferred frees"
    )

    with monkeypatch.context() as sync_guard:
        sync_guard.setattr(
            "vibespatial.spatial.spatial_index_device.get_cuda_runtime",
            lambda: guarded_runtime,
        )
        candidates = _morton_range_query(flat_index, query_bounds, query_bounds)

    assert candidates is not None
    assert candidates.total_pairs == 0
    guarded_runtime.synchronize.assert_not_called()


@requires_gpu
def test_native_spatial_index_reductions_do_not_context_synchronize(monkeypatch):
    """Semijoin, antijoin, and counts keep exact semantics without a context fence."""
    from unittest.mock import Mock

    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import get_cuda_runtime
    from vibespatial.spatial import spatial_index_device as spatial_index_device_module

    tree_owned = from_shapely_geometries(
        [
            box(0.0, 0.0, 2.0, 2.0),
            box(0.0, 0.0, 1.0, 1.0),
            box(10.0, 10.0, 11.0, 11.0),
        ],
        residency=Residency.DEVICE,
    )
    flat_index = build_flat_spatial_index(
        tree_owned,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            reason="test native reduction deferred-free ownership",
        ),
    )
    query_owned = from_shapely_geometries(
        [
            Point(0.5, 0.5),
            Point(1.5, 1.5),
            Point(5.0, 5.0),
            Point(10.5, 10.5),
        ],
        residency=Residency.DEVICE,
    )
    query_bounds = compute_geometry_bounds_device(query_owned)
    native_index = flat_index.to_native_spatial_index(source_token="tree")
    runtime = get_cuda_runtime()
    guarded_runtime = Mock(wraps=runtime)
    guarded_runtime.synchronize.side_effect = AssertionError(
        "native spatial reductions must not context-synchronize"
    )
    observed_precision_plans = []
    classify_tile = spatial_index_device_module._classify_homogeneous_reduction_tile

    def classify_tile_with_precision(*args, **kwargs):
        observed_precision_plans.append(kwargs["precision_plan"])
        return classify_tile(*args, **kwargs)

    query_kwargs = {
        "predicate": "intersects",
        "query_token": "query",
        "precomputed_query_bounds": query_bounds,
        "return_metadata": True,
    }
    with monkeypatch.context() as sync_guard:
        sync_guard.setattr(
            "vibespatial.spatial.spatial_index_device.get_cuda_runtime",
            lambda: guarded_runtime,
        )
        sync_guard.setattr(
            spatial_index_device_module,
            "_classify_homogeneous_reduction_tile",
            classify_tile_with_precision,
        )
        semijoin, semijoin_execution = native_index.query_left_semijoin(
            query_owned,
            **query_kwargs,
        )
        antijoin, antijoin_execution = native_index.query_left_antijoin(
            query_owned,
            **query_kwargs,
        )
        counts, count_execution = native_index.query_left_match_count_expression(
            query_owned,
            **query_kwargs,
        )
        right_semijoin, right_execution = native_index.query_right_semijoin(
            query_owned,
            predicate="intersects",
            precomputed_query_bounds=query_bounds,
            return_metadata=True,
        )

    assert semijoin_execution.implementation == "owned_gpu_spatial_semijoin"
    assert antijoin_execution.implementation == "owned_gpu_spatial_semijoin"
    assert count_execution.implementation == "owned_gpu_spatial_match_count"
    assert right_execution.implementation == "owned_gpu_spatial_right_semijoin"
    semijoin_count = int(cp.asnumpy(semijoin.logical_count)[0])
    antijoin_count = int(cp.asnumpy(antijoin.logical_count)[0])
    right_count = int(cp.asnumpy(right_semijoin.logical_count)[0])
    assert cp.asnumpy(semijoin.positions[:semijoin_count]).tolist() == [0, 1, 3]
    assert cp.asnumpy(antijoin.positions[:antijoin_count]).tolist() == [2]
    assert cp.asnumpy(counts.values).tolist() == [2, 1, 0, 1]
    assert cp.asnumpy(right_semijoin.positions[:right_count]).tolist() == [0, 1, 2]
    assert observed_precision_plans
    assert all(
        plan.compute_precision is PrecisionMode.FP64
        for plan in observed_precision_plans
    )
    guarded_runtime.synchronize.assert_not_called()


@requires_gpu
def test_mixed_polygon_reduction_checks_every_multipolygon_component():
    cp = pytest.importorskip("cupy")

    components = [box(10.0 * i, 0.0, 10.0 * i + 1.0, 1.0) for i in range(34)]
    tree_owned = from_shapely_geometries(
        [MultiPolygon(components), Point(-100.0, -100.0)],
        residency=Residency.DEVICE,
    )
    flat_index = build_flat_spatial_index(
        tree_owned,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            reason="test unbounded multipolygon component traversal",
        ),
    )
    query_owned = from_shapely_geometries(
        [box(330.25, 0.25, 330.75, 0.75), box(500.0, 0.0, 501.0, 1.0)],
        residency=Residency.DEVICE,
    )
    query_bounds = compute_geometry_bounds_device(query_owned)
    native_index = flat_index.to_native_spatial_index(source_token="tree")
    query_kwargs = {
        "predicate": "intersects",
        "query_token": "query",
        "precomputed_query_bounds": query_bounds,
        "return_metadata": True,
    }

    semijoin, semijoin_execution = native_index.query_left_semijoin(
        query_owned,
        **query_kwargs,
    )
    antijoin, antijoin_execution = native_index.query_left_antijoin(
        query_owned,
        **query_kwargs,
    )
    counts, count_execution = native_index.query_left_match_count_expression(
        query_owned,
        **query_kwargs,
    )

    assert semijoin_execution.implementation == "owned_gpu_spatial_semijoin"
    assert antijoin_execution.implementation == "owned_gpu_spatial_semijoin"
    assert count_execution.implementation == "owned_gpu_spatial_match_count"
    semijoin_count = int(cp.asnumpy(semijoin.logical_count)[0])
    antijoin_count = int(cp.asnumpy(antijoin.logical_count)[0])
    assert cp.asnumpy(semijoin.positions[:semijoin_count]).tolist() == [0]
    assert cp.asnumpy(antijoin.positions[:antijoin_count]).tolist() == [1]
    assert cp.asnumpy(counts.values).tolist() == [1, 0]


@requires_gpu
def test_device_query_brute_force_for_small_input():
    """For small N*M, brute-force strategy is used."""
    tree_geoms = _make_grid_boxes(3, 3)  # 9 boxes
    query_geoms = np.asarray([box(0.5, 0.5, 2.5, 2.5)], dtype=object)

    tree_owned, flat_index = build_owned_spatial_index(tree_geoms)
    query_owned = from_shapely_geometries(query_geoms)
    query_bounds = compute_geometry_bounds(query_owned)

    cands, execution = spatial_index_device_query(flat_index, query_bounds)
    assert execution.selected is ExecutionMode.GPU
    assert "brute" in execution.reason


# ---------------------------------------------------------------------------
# Tests: end-to-end sjoin integration
# ---------------------------------------------------------------------------


@requires_gpu
def test_sjoin_uses_device_query_end_to_end():
    """sjoin produces correct results using the device spatial index query."""
    import vibespatial as gpd

    # Left: random points.
    left_points = _make_random_points(50, seed=100, extent=10.0)
    left_gdf = gpd.GeoDataFrame(
        {"id": range(len(left_points))},
        geometry=list(left_points),
    )

    # Right: grid of boxes.
    right_boxes = _make_grid_boxes(10, 10)
    right_gdf = gpd.GeoDataFrame(
        {"zone": range(len(right_boxes))},
        geometry=list(right_boxes),
    )

    # sjoin should work correctly.
    result = gpd.sjoin(left_gdf, right_gdf, predicate="intersects")
    assert len(result) > 0

    # Verify a few known overlaps manually: every point in [0,10] x [0,10]
    # should match at least one grid cell.
    for idx in range(min(5, len(left_gdf))):
        pt = left_gdf.geometry.iloc[idx]
        # Find expected zone(s).
        col = int(pt.x)
        row = int(pt.y)
        if col >= 10:
            col = 9
        if row >= 10:
            row = 9
        # Check that the point appears in the result with this zone.
        point_results = result[result.index == idx]
        assert len(point_results) > 0, f"Point {idx} at ({pt.x:.2f}, {pt.y:.2f}) not in sjoin result"


@requires_gpu
def test_sjoin_with_polygons():
    """sjoin correctly handles polygon-polygon spatial joins."""
    import vibespatial as gpd

    left_polys = np.asarray([
        box(0, 0, 5, 5),
        box(10, 10, 15, 15),
        box(20, 20, 25, 25),
    ], dtype=object)
    right_polys = np.asarray([
        box(3, 3, 8, 8),    # overlaps first left
        box(12, 12, 18, 18), # overlaps second left
        box(50, 50, 55, 55), # no overlap
    ], dtype=object)

    left_gdf = gpd.GeoDataFrame(
        {"left_id": [0, 1, 2]}, geometry=list(left_polys),
    )
    right_gdf = gpd.GeoDataFrame(
        {"right_id": [0, 1, 2]}, geometry=list(right_polys),
    )

    result = gpd.sjoin(left_gdf, right_gdf, predicate="intersects")
    assert len(result) == 2  # two overlapping pairs

    # Verify the correct pairs.
    result_pairs = set(zip(result["left_id"].tolist(), result["right_id"].tolist()))
    assert (0, 0) in result_pairs
    assert (1, 1) in result_pairs


# ---------------------------------------------------------------------------
# Tests: execution metadata
# ---------------------------------------------------------------------------


@requires_gpu
def test_device_query_returns_execution_metadata():
    """Execution metadata has correct structure."""
    tree_geoms = _make_grid_boxes(5, 5)
    query_geoms = np.asarray([box(0.5, 0.5, 2.5, 2.5)], dtype=object)

    tree_owned, flat_index = build_owned_spatial_index(tree_geoms)
    query_owned = from_shapely_geometries(query_geoms)
    query_bounds = compute_geometry_bounds(query_owned)

    cands, execution = spatial_index_device_query(flat_index, query_bounds)
    assert execution.selected is ExecutionMode.GPU
    assert execution.implementation == "owned_gpu_spatial_query"
    assert len(execution.reason) > 0


def test_device_query_cpu_fallback_when_no_gpu():
    """When GPU is unavailable, function returns None with CPU execution."""
    # This test runs even without GPU.
    import unittest.mock

    tree_geoms = _make_grid_boxes(3, 3)

    with unittest.mock.patch(
        "vibespatial.spatial.spatial_index_device.has_gpu_runtime",
        return_value=False,
    ):
        tree_owned, flat_index = build_owned_spatial_index(tree_geoms)
        query_bounds = np.array([[0.5, 0.5, 2.5, 2.5]], dtype=np.float64)

        cands, execution = spatial_index_device_query(flat_index, query_bounds)
        assert cands is None
        assert execution.selected is ExecutionMode.CPU

        empty_bounds = np.empty((0, 4), dtype=np.float64)
        cands, execution = spatial_index_device_query(flat_index, empty_bounds)
        assert cands is None
        assert execution.selected is ExecutionMode.CPU


# ---------------------------------------------------------------------------
# Tests: NaN handling
# ---------------------------------------------------------------------------


@requires_gpu
def test_device_query_handles_nan_bounds():
    """NaN bounds in query or tree are handled gracefully."""
    tree_geoms = np.asarray([box(0, 0, 1, 1), None, box(2, 2, 3, 3)], dtype=object)
    tree_owned, flat_index = build_owned_spatial_index(tree_geoms)

    # Query with a mix of valid and NaN bounds.
    query_bounds = np.array([
        [0.5, 0.5, 1.5, 1.5],   # overlaps first tree box
        [np.nan, np.nan, np.nan, np.nan],  # invalid
    ], dtype=np.float64)

    cands, execution = spatial_index_device_query(flat_index, query_bounds)
    assert execution.selected is ExecutionMode.GPU
    if cands is not None:
        gpu_left, gpu_right = cands.to_host()
        # Only the first query row should produce candidates.
        assert np.all(gpu_left == 0) or gpu_left.size == 0


# ---------------------------------------------------------------------------
# Tests: correctness via query_spatial_index integration
# ---------------------------------------------------------------------------


@requires_gpu
def test_query_spatial_index_uses_device_query():
    """query_spatial_index routes through spatial_index_device_query."""
    tree_geoms = _make_grid_boxes(10, 10)
    query_geoms = _make_random_points(50, seed=200, extent=10.0)

    tree_owned, flat_index = build_owned_spatial_index(tree_geoms)

    result = query_spatial_index(
        tree_owned, flat_index, query_geoms,
        predicate="intersects",
        return_metadata=True,
    )
    indices, execution = result
    # Should use GPU path.
    assert execution.selected is ExecutionMode.GPU
    # Indices should be 2D (left_idx, right_idx).
    assert indices.ndim == 2
    assert indices.shape[0] == 2
    assert indices.shape[1] > 0  # at least some intersecting pairs


@requires_gpu
def test_device_query_large_correctness():
    """Correctness check with 500 query x 500 tree (250K N*M)."""
    tree_geoms = _make_random_boxes(500, seed=30, extent=100.0, size=5.0)
    query_geoms = _make_random_boxes(500, seed=31, extent=100.0, size=5.0)

    tree_owned, flat_index = build_owned_spatial_index(tree_geoms)
    query_owned = from_shapely_geometries(query_geoms)
    query_bounds = compute_geometry_bounds(query_owned)
    tree_bounds = flat_index.bounds

    cands, execution = spatial_index_device_query(flat_index, query_bounds)
    assert execution.selected is ExecutionMode.GPU
    assert cands is not None

    gpu_left, gpu_right = cands.to_host()
    cpu_left, cpu_right = _cpu_bbox_pairs(query_bounds, tree_bounds)

    gpu_pairs = set(zip(gpu_left.tolist(), gpu_right.tolist()))
    cpu_pairs = set(zip(cpu_left.tolist(), cpu_right.tolist()))
    # Must have no false negatives.
    assert cpu_pairs.issubset(gpu_pairs), (
        f"False negatives: {cpu_pairs - gpu_pairs}"
    )
    # For brute-force, should be exact match.
    if "brute" in execution.reason:
        assert gpu_pairs == cpu_pairs


@requires_gpu
def test_bounded_many_by_few_device_query_avoids_pair_count_fence():
    """Bounded N*M brute force compacts a pair mask without scalarizing output size."""
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    tree_geoms = _make_random_boxes(7, seed=71, extent=20.0, size=3.0)
    query_geoms = _make_random_boxes(128, seed=72, extent=20.0, size=3.0)

    tree_owned, flat_index = build_owned_spatial_index(tree_geoms)
    query_owned = from_shapely_geometries(query_geoms, residency=Residency.DEVICE)
    query_bounds = compute_geometry_bounds_device(query_owned)

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    cands, execution = spatial_index_device_query(flat_index, query_bounds)
    events = get_d2h_transfer_events(clear=True)

    assert execution.selected is ExecutionMode.GPU
    assert cands is not None
    assert "brute-force" in execution.reason
    assert not any(
        event.reason == "device spatial-index candidate-pair allocation fence"
        for event in events
    )


@requires_gpu
def test_device_bounds_only_index_feeds_bounded_query_without_metadata_fences():
    """Small device joins can consume bounds without building Morton metadata."""
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    tree_owned = from_shapely_geometries(
        _make_random_boxes(7, seed=81, extent=20.0, size=3.0),
        residency=Residency.DEVICE,
    )
    query_owned = from_shapely_geometries(
        _make_random_boxes(128, seed=82, extent=20.0, size=3.0),
        residency=Residency.DEVICE,
    )

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    flat_index = build_flat_spatial_index(
        tree_owned,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            reason="test device-bounds-only spatial index",
        ),
        device_bounds_only=True,
    )
    query_bounds = compute_geometry_bounds_device(query_owned)
    cands, execution = spatial_index_device_query(flat_index, query_bounds)
    events = get_d2h_transfer_events(clear=True)

    assert execution.selected is ExecutionMode.GPU
    assert cands is not None
    assert flat_index.device_bounds is not None
    assert flat_index.device_order is None
    assert flat_index.device_morton_keys is None
    assert not any(
        event.reason
        in {
            "spatial index regular-grid summary scalar fence",
            "flat spatial index device total-bounds scalar fence",
            "device spatial-index candidate-pair allocation fence",
        }
        for event in events
    )
