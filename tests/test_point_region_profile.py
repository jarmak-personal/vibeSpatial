from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon, box

from vibespatial.api import GeoDataFrame, GeoSeries, points_from_xy, read_parquet
from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_I64,
    KERNEL_PARAM_PTR,
    DeviceMemoryAdmission,
    get_cuda_runtime,
)
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.predicates.point_location_index import (
    _point_location_preparation_metrics,
    point_location_bin_policy,
    point_location_part_y_index_profile_kernels,
    prepare_polygon_part_y_index,
)
from vibespatial.predicates.point_location_index_kernels import (
    _POINT_LOCATION_PART_Y_INDEX_PROFILE_SOURCE,
    _POINT_LOCATION_PART_Y_INDEX_SOURCE,
    coverage_grid_width_for_bin_count,
)
from vibespatial.predicates.point_region_profile import profile_point_region
from vibespatial.predicates.point_relations import (
    _point_region_level0_packet,
    _sync_hotpath,
    classify_point_region_gpu,
)
from vibespatial.runtime import has_gpu_runtime
from vibespatial.runtime.fallbacks import clear_fallback_events, get_fallback_events
from vibespatial.runtime.residency import Residency
from vibespatial.spatial.point_grid_index import (
    _point_grid_preparation_metrics,
)


def test_level0_preparation_packets_use_host_carrier_metadata() -> None:
    grid = SimpleNamespace(
        cache_key=SimpleNamespace(row_count=600_000_000),
        grid_size=2048,
        device_bytes=9_000_000_000,
    )
    grid_sums, grid_maxima, grid_unavailable = _point_grid_preparation_metrics(
        grid,
        built=False,
        cache_hit=True,
        declined=False,
        query_count=224_676,
        pair_budget=16_000_000,
    )

    assert grid_sums["cache_hits"] == 1
    assert grid_sums["build_count"] == 0
    assert grid_maxima == {
        "source_rows": 600_000_000,
        "grid_cells": 2048 * 2048,
        "persistent_bytes": 9_000_000_000,
        "pair_budget_slots": 16_000_000,
    }
    assert "avoidable_rebuild_seconds" in grid_unavailable

    part_y = SimpleNamespace(
        geometry_count=224_676,
        part_count=288_955,
        bin_count=8,
        target_bin_count=128,
        nominal_vram_class_gib=100,
        edge_membership_count=78_341_756,
        device_bytes=345_729_984,
        peak_build_bytes=702_000_000,
        coverage_grid_width=8,
    )
    part_y_sums, part_y_maxima, part_y_unavailable = (
        _point_location_preparation_metrics(
            part_y,
            built=True,
            cache_hit=False,
            cache_miss=True,
            declined=False,
        )
    )

    assert part_y_sums["build_count"] == 1
    assert part_y_sums["cache_misses"] == 1
    assert part_y_maxima == {
        "source_geometries": 224_676,
        "source_parts": 288_955,
        "part_y_bin_slots": 2_311_640,
        "target_bin_count": 128,
        "admitted_bin_count": 8,
        "nominal_vram_class_gib": 100,
        "edge_memberships": 78_341_756,
        "persistent_bytes": 345_729_984,
        "peak_build_bytes": 702_000_000,
        "coverage_grid_width": 8,
        "coverage_cells": 288_955 * 64,
    }
    assert "avoidable_rebuild_seconds" in part_y_unavailable

    _, declined_grid_maxima, declined_grid_unavailable = (
        _point_grid_preparation_metrics(
            None,
            built=False,
            cache_hit=False,
            declined=True,
            query_count=10,
            pair_budget=1_000,
        )
    )
    assert declined_grid_maxima == {"pair_budget_slots": 1_000}
    assert "source_rows" in declined_grid_unavailable
    assert "cache_hits" in declined_grid_unavailable

    _, declined_part_y_maxima, declined_part_y_unavailable = (
        _point_location_preparation_metrics(
            None,
            built=False,
            cache_hit=False,
            cache_miss=True,
            declined=True,
        )
    )
    assert declined_part_y_maxima == {}
    assert "source_parts" in declined_part_y_unavailable


@pytest.mark.parametrize(
    ("reported_gib", "expected_nominal_gib", "expected_bins"),
    [
        (7.7, 8, 8),
        (8.1, 8, 16),
        (15.3, 16, 16),
        (17.0, 17, 32),
        (22.0, 22, 32),
        (22.9, 24, 64),
        (24.0, 24, 64),
        (45.7, 48, 128),
        (48.0, 48, 128),
        (95.1, 100, 256),
        (100.0, 100, 256),
    ],
)
def test_point_location_bin_policy_uses_tolerant_nominal_vram_classes(
    reported_gib,
    expected_nominal_gib,
    expected_bins,
) -> None:
    nominal_gib, bins = point_location_bin_policy(int(reported_gib * (1 << 30)))

    assert nominal_gib == expected_nominal_gib
    assert bins == expected_bins


def test_point_region_profiler_is_absent_from_production_kernel_source() -> None:
    """Disabled profiling must not add counters or atomics to production code."""
    assert "VS_PROFILE_COUNTER_COUNT" not in _POINT_LOCATION_PART_Y_INDEX_SOURCE
    assert "_profiled" not in _POINT_LOCATION_PART_Y_INDEX_SOURCE
    assert "VS_PROFILE_COUNTER_COUNT" in _POINT_LOCATION_PART_Y_INDEX_PROFILE_SOURCE
    assert "point_in_multipolygon_prepared_part_y_index_profiled" in (
        _POINT_LOCATION_PART_Y_INDEX_PROFILE_SOURCE
    )


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime required")
@pytest.mark.parametrize("bin_count", [8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize(
    ("family", "as_region"),
    [
        (GeometryFamily.POLYGON, lambda polygon: polygon),
        (GeometryFamily.MULTIPOLYGON, lambda polygon: MultiPolygon([polygon])),
    ],
)
def test_uniform_part_y_width_variants_preserve_exact_point_location(
    monkeypatch,
    bin_count,
    family,
    as_region,
) -> None:
    polygon = Polygon(
        [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)],
        holes=[[(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)]],
    )
    regions = from_shapely_geometries(
        np.asarray([as_region(polygon)] * 4, dtype=object),
        residency=Residency.DEVICE,
    )
    points = from_shapely_geometries(
        np.asarray(
            [Point(0.5, 0.5), Point(2, 2), Point(0, 2), Point(10, 10)],
            dtype=object,
        ),
        residency=Residency.DEVICE,
    )
    monkeypatch.setattr(
        "vibespatial.predicates.point_location_index._MIN_PREPARED_COORDINATES",
        0,
    )

    prepared = prepare_polygon_part_y_index(
        regions,
        family,
        _target_bin_count=bin_count,
    )
    relation = classify_point_region_gpu(
        np.arange(4, dtype=np.int32),
        points,
        regions,
        region_family=family,
    )

    assert prepared is not None
    assert prepared.bin_count == bin_count
    assert prepared.coverage_grid_width == coverage_grid_width_for_bin_count(bin_count)
    assert relation.tolist() == [2, 0, 1, 0]


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime required")
@pytest.mark.parametrize(
    ("family", "as_region"),
    [
        (GeometryFamily.POLYGON, lambda polygon: polygon),
        (GeometryFamily.MULTIPOLYGON, lambda polygon: MultiPolygon([polygon])),
    ],
)
def test_conservative_coverage_grid_matches_dense_shapely_point_locations(
    monkeypatch,
    family,
    as_region,
) -> None:
    polygon = Polygon(
        [(0, 0), (5, 0), (5, 1), (2, 1), (2, 5), (0, 5), (0, 0)],
        holes=[[(0.5, 2), (1.5, 2), (1.5, 4), (0.5, 4), (0.5, 2)]],
    )
    axis = np.linspace(-0.5, 5.5, 49)
    point_values = [Point(x, y) for y in axis for x in axis]
    point_values.extend(
        [
            Point(0, 0),
            Point(5, 0.5),
            Point(2, 3),
            Point(1, 2),
            Point(0.5, 3),
            Point(1, 3),
        ]
    )
    expected = np.asarray(
        [
            1 if polygon.touches(point) else 2 if polygon.contains(point) else 0
            for point in point_values
        ],
        dtype=np.uint8,
    )
    regions = from_shapely_geometries(
        np.asarray([as_region(polygon)] * len(point_values), dtype=object),
        residency=Residency.DEVICE,
    )
    points = from_shapely_geometries(
        np.asarray(point_values, dtype=object),
        residency=Residency.DEVICE,
    )
    monkeypatch.setattr(
        "vibespatial.predicates.point_location_index._MIN_PREPARED_COORDINATES",
        0,
    )

    prepared = prepare_polygon_part_y_index(
        regions,
        family,
        _target_bin_count=64,
    )
    actual = classify_point_region_gpu(
        np.arange(len(point_values), dtype=np.int32),
        points,
        regions,
        region_family=family,
    )

    assert prepared is not None
    assert prepared.coverage_grid_width == 8
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime required")
def test_part_y_capacity_declines_to_the_next_compiled_width(monkeypatch) -> None:
    regions = from_shapely_geometries(
        np.asarray([box(0, 0, 4, 4)], dtype=object),
        residency=Residency.DEVICE,
    )
    monkeypatch.setattr(
        "vibespatial.predicates.point_location_index._MIN_PREPARED_COORDINATES",
        0,
    )
    runtime = get_cuda_runtime()
    original_admit = runtime.admit_device_memory

    def _decline_128(*, stage, required_bytes, requested_units=0):
        if stage.endswith(".b128.structural"):
            return DeviceMemoryAdmission(
                stage=stage,
                required_bytes=required_bytes,
                remaining_bytes=required_bytes - 1,
                budget_bytes=required_bytes - 1,
                admitted=False,
                requested_units=requested_units,
                admitted_units=0,
                bytes_per_unit=1,
            )
        return original_admit(
            stage=stage,
            required_bytes=required_bytes,
            requested_units=requested_units,
        )

    monkeypatch.setattr(runtime, "admit_device_memory", _decline_128)
    prepared = prepare_polygon_part_y_index(
        regions,
        GeometryFamily.POLYGON,
        _target_bin_count=128,
    )

    assert prepared is not None
    assert prepared.target_bin_count == 128
    assert prepared.bin_count == 64
    assert prepared.decline_reason is not None
    assert "b128: structural peak" in prepared.decline_reason
    decision = regions.device_state.point_location_index_decisions[
        GeometryFamily.POLYGON
    ]
    assert decision.admitted_bin_count == 64
    assert decision.cache_hit is False

    assert (
        prepare_polygon_part_y_index(
            regions,
            GeometryFamily.POLYGON,
            _target_bin_count=128,
        )
        is prepared
    )
    assert regions.device_state.point_location_index_decisions[
        GeometryFamily.POLYGON
    ].cache_hit is True


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime required")
def test_coverage_capacity_decline_retains_exact_part_y_index(monkeypatch) -> None:
    regions = from_shapely_geometries(
        np.asarray([box(0, 0, 4, 4)] * 4, dtype=object),
        residency=Residency.DEVICE,
    )
    points = from_shapely_geometries(
        np.asarray(
            [Point(1, 1), Point(0, 2), Point(5, 5), Point(3, 3)],
            dtype=object,
        ),
        residency=Residency.DEVICE,
    )
    monkeypatch.setattr(
        "vibespatial.predicates.point_location_index._MIN_PREPARED_COORDINATES",
        0,
    )
    runtime = get_cuda_runtime()
    original_admit = runtime.admit_device_memory

    def _decline_coverage(*, stage, required_bytes, requested_units=0):
        if stage.endswith(".coverage_grid"):
            return DeviceMemoryAdmission(
                stage=stage,
                required_bytes=required_bytes,
                remaining_bytes=required_bytes - 1,
                budget_bytes=required_bytes - 1,
                admitted=False,
                requested_units=requested_units,
                admitted_units=0,
                bytes_per_unit=1,
            )
        return original_admit(
            stage=stage,
            required_bytes=required_bytes,
            requested_units=requested_units,
        )

    monkeypatch.setattr(runtime, "admit_device_memory", _decline_coverage)
    prepared = prepare_polygon_part_y_index(
        regions,
        GeometryFamily.POLYGON,
        _target_bin_count=64,
    )
    actual = classify_point_region_gpu(
        np.arange(4, dtype=np.int32),
        points,
        regions,
        region_family=GeometryFamily.POLYGON,
    )

    assert prepared is not None
    assert prepared.bin_count == 64
    assert prepared.coverage_grid_width == 0
    assert prepared.coverage is None
    assert prepared.coverage_decline_reason is not None
    assert actual.tolist() == [2, 1, 0, 2]


def test_part_y_builder_has_no_width_sized_per_thread_cursor_array() -> None:
    assert "cursors[VS_PART_Y_BIN_COUNT]" not in _POINT_LOCATION_PART_Y_INDEX_SOURCE
    assert "atomicAdd(cursors + key, 1u)" in _POINT_LOCATION_PART_Y_INDEX_SOURCE
    assert "count_polygon_edge_y_bin_memberships" in _POINT_LOCATION_PART_Y_INDEX_SOURCE
    assert "initialize_polygon_part_coverage_cells" in (
        _POINT_LOCATION_PART_Y_INDEX_SOURCE
    )
    assert "mark_polygon_edge_coverage_cells" in _POINT_LOCATION_PART_Y_INDEX_SOURCE


def test_point_region_level0_packet_uses_only_host_known_structure() -> None:
    class Prepared:
        geometry_count = 3
        part_count = 7
        edge_membership_count = 42
        device_bytes = 512

    packet = _point_region_level0_packet(
        kernel_name="point_in_multipolygon_prepared_part_y_index",
        region_family=GeometryFamily.MULTIPOLYGON,
        candidate_count=100,
        launch_items=128,
        logical_count=object(),
        prepared=Prepared(),
    )

    assert packet["sum"] == {
        "candidate_lanes": 100,
        "launch_items": 128,
        "prepared_consumer_count": 1,
    }
    assert packet["max"] == {
        "launch_capacity": 128,
        "prepared_geometry_count": 3,
        "prepared_part_count": 7,
        "prepared_edge_membership_count": 42,
        "prepared_device_bytes": 512,
    }
    assert packet["semantic_contract"]["logical_count_state"] == "device_resident"
    assert "survivors" in packet["unavailable"]


def test_point_region_hotpath_sync_is_full_timing_only(monkeypatch) -> None:
    calls: list[str] = []
    runtime = SimpleNamespace(synchronize=lambda: calls.append("sync"))

    for mode in ("off", "counter"):
        monkeypatch.setenv("VIBESPATIAL_HOTPATH_TRACE", mode)
        _sync_hotpath(runtime)
    assert calls == []

    monkeypatch.setenv("VIBESPATIAL_HOTPATH_TRACE", "full")
    _sync_hotpath(runtime)
    assert calls == ["sync"]


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime required")
@pytest.mark.parametrize(
    ("count", "sample_count", "expected"),
    [
        (10, 4, [2, 4, 7, 9]),
        (3, 3, [0, 1, 2]),
        (10, 0, []),
    ],
)
def test_point_region_profile_samples_across_the_full_logical_launch(
    count,
    sample_count,
    expected,
) -> None:
    import cupy as cp

    runtime = get_cuda_runtime()
    kernel = point_location_part_y_index_profile_kernels()[
        "point_region_profile_sample_mask"
    ]
    mask = cp.empty(count, dtype=cp.uint8)
    grid, block = runtime.launch_config(kernel, count)
    runtime.launch(
        kernel,
        grid=grid,
        block=block,
        params=(
            (count, sample_count, runtime.pointer(mask)),
            (KERNEL_PARAM_I32, KERNEL_PARAM_I64, KERNEL_PARAM_PTR),
        ),
    )

    actual = np.flatnonzero(cp.asnumpy(mask)).tolist()
    assert actual == expected


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime required")
def test_point_region_profile_observes_public_pair_aggregate_boundedly(
    tmp_path,
) -> None:
    from vibespatial.spatial.point_partition import (
        PointPartitionVariant,
        force_point_partition_variant_for_testing,
    )

    source = GeoDataFrame(
        {
            "dropoff": points_from_xy([2.5, 0.5, -1, 10], [2.5, 0.5, -1, 10]),
        },
        geometry=points_from_xy([0.5, 2.5, 10, -1], [0.5, 2.5, 10, -1]),
        crs="EPSG:4326",
    ).rename_geometry("pickup")
    source["dropoff"] = source["dropoff"].set_crs(source.crs)
    path = tmp_path / "aligned-points.parquet"
    source.to_parquet(path, geometry_encoding="geoarrow", index=False)
    source = read_parquet(path)
    pickup = source.set_geometry("pickup").geometry
    dropoff = source.set_geometry("dropoff").geometry
    zones = GeoSeries(
        [MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)])]
    )
    clear_fallback_events()

    with force_point_partition_variant_for_testing(PointPartitionVariant.GRID):
        with profile_point_region(
            label="public-paired-count",
            sample_limit=2,
            force_prepared_index=True,
        ) as profile:
            result = pickup.sindex.query_pair_aggregate(
                dropoff.sindex,
                zones,
                predicate="contains",
            )
            snapshot = profile.snapshot()

    assert result.to_dict("list") == {
        "left_count": [1, 1, 0, 0],
        "right_count": [1, 1, 0, 0],
        "shared_count": [1, 1, 0, 0],
    }
    assert get_fallback_events(clear=True) == []
    assert snapshot["forced_prepared_index"] is True
    assert snapshot["sample_limit"] == 2
    assert len(snapshot["groups"]) == 1
    group = snapshot["groups"][0]
    assert group["family"] == "multipolygon"
    assert group["coverage_grid_width"] == 8
    assert group["launches"] == 3
    assert group["counters"]["candidates"] == 4
    assert group["counters"]["parts_considered"] == 6
    assert group["counters"]["active_parts"] == 4
    assert group["counters"]["edges_visited"] == 0
    assert group["counters"]["sample_reservations"] == 2
    assert group["counters"]["sampled_candidates"] == 2
    assert group["parts_considered_percentiles"] == {
        "p50": 1,
        "p95": 2,
        "p99": 2,
    }
    assert snapshot["index_preparation"][0]["build_count"] == 1
    assert snapshot["index_preparation"][0]["cache_hits"] == 2
    assert snapshot["index_preparation"][0]["build_wall_seconds"] > 0.0


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime required")
def test_public_pair_aggregate_retains_prepared_point_grid(
    monkeypatch,
    tmp_path,
) -> None:
    from vibespatial.spatial import point_grid_index
    from vibespatial.spatial.point_partition import (
        PointPartitionVariant,
        force_point_partition_variant_for_testing,
    )

    monkeypatch.setattr(point_grid_index, "_MIN_POINT_GRID_ROWS", 0)
    source = GeoDataFrame(
        {
            "dropoff": points_from_xy([2.5, 0.5, -1, 10], [2.5, 0.5, -1, 10]),
        },
        geometry=points_from_xy([0.5, 2.5, 10, -1], [0.5, 2.5, 10, -1]),
        crs="EPSG:4326",
    ).rename_geometry("pickup")
    source["dropoff"] = source["dropoff"].set_crs(source.crs)
    path = tmp_path / "retained-point-grid.parquet"
    source.to_parquet(path, geometry_encoding="geoarrow", index=False)
    source = read_parquet(path)
    pickup_index = source.set_geometry("pickup").geometry.sindex
    dropoff_index = source.set_geometry("dropoff").geometry.sindex
    zones = GeoSeries([MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)])])
    clear_fallback_events()

    with force_point_partition_variant_for_testing(PointPartitionVariant.GRID):
        first = pickup_index.query_pair_aggregate(
            dropoff_index,
            zones,
            predicate="contains",
        )
        native_index = pickup_index._native_spatial_index
        assert native_index is not None
        prepared_grid = next(iter(native_index.point_partition_cache.values()))

        second = pickup_index.query_pair_aggregate(
            dropoff_index,
            zones,
            predicate="contains",
        )

    assert second.to_dict("list") == first.to_dict("list")
    assert next(iter(native_index.point_partition_cache.values())) is prepared_grid
    assert native_index.to_flat_index().point_grid is None
    assert get_fallback_events(clear=True) == []


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime required")
def test_public_pair_aggregate_propagates_guarded_scatter_fault(monkeypatch, tmp_path) -> None:
    from dataclasses import replace

    import cupy as cp

    from vibespatial.spatial import point_grid_index, spatial_index_device
    from vibespatial.spatial.point_partition import (
        PointPartitionVariant,
        force_point_partition_variant_for_testing,
    )

    points = points_from_xy([0.25, 0.75], [0.25, 0.75], crs="EPSG:3857")
    source = GeoDataFrame({"aligned": points}, geometry=points, crs=points.crs)
    path = tmp_path / "public-scatter-fault.parquet"
    source.to_parquet(path, geometry_encoding="geoarrow", index=False)
    source = read_parquet(path)
    left_index = source.geometry.sindex
    right_index = source.set_geometry("aligned").geometry.sindex
    regions = GeoSeries([box(0.0, 0.0, 1.0, 1.0)], crs=source.crs)
    original_superset = point_grid_index.point_grid_superset_query

    def _faulted_superset(*args, **kwargs):
        candidates = original_superset(*args, **kwargs)
        assert candidates is not None
        return replace(candidates, error_flag=cp.ones(1, dtype=cp.uint32))

    def _no_morton_retry(*_args, **_kwargs):
        raise AssertionError("guarded scatter faults must not retry Morton")

    monkeypatch.setattr(point_grid_index, "point_grid_superset_query", _faulted_superset)
    monkeypatch.setattr(
        spatial_index_device,
        "_prepare_morton_range_query",
        _no_morton_retry,
    )
    with force_point_partition_variant_for_testing(PointPartitionVariant.GRID):
        with pytest.raises(RuntimeError, match="sealed capacity"):
            left_index.query_pair_aggregate(
                right_index,
                regions,
                predicate="contains",
            )


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime required")
@pytest.mark.parametrize("variant", ["grid"])
def test_point_partition_mixed_nonfinite_rows_use_morton_oracle(
    tmp_path,
    variant,
) -> None:
    from vibespatial.spatial.point_partition import (
        PointPartitionVariant,
        force_point_partition_variant_for_testing,
    )

    points = GeoSeries(
        [
            Point(0.0, 0.0),
            Point(0.0, 0.0),
            None,
            Point(),
            Point(float("inf"), 1.0),
            Point(1.0e12, -1.0e12),
        ],
        crs="EPSG:3857",
    )
    source = GeoDataFrame({"aligned": points}, geometry=points, crs=points.crs)
    path = tmp_path / f"mixed-nonfinite-{variant}.parquet"
    source.to_parquet(path, geometry_encoding="geoarrow", index=False)
    source = read_parquet(path)
    left_index = source.geometry.sindex
    right_index = source.set_geometry("aligned").geometry.sindex
    regions = GeoSeries(
        [
            box(-1.0, -1.0, 1.0, 1.0),
            box(-1.0, -1.0, 1.0, 1.0),
            None,
            Polygon(),
        ],
        crs=source.crs,
    )
    with force_point_partition_variant_for_testing(PointPartitionVariant.MORTON):
        expected = left_index.query_pair_aggregate(
            right_index,
            regions,
            predicate="contains",
        ).to_dict("list")
    clear_fallback_events()
    with force_point_partition_variant_for_testing(PointPartitionVariant(variant)):
        actual = left_index.query_pair_aggregate(
            right_index,
            regions,
            predicate="contains",
        ).to_dict("list")
    assert actual == expected
    assert get_fallback_events(clear=True) == []
    assert left_index._native_spatial_index.point_partition_cache == {}
    assert right_index._native_spatial_index.point_partition_cache == {}
    assert get_fallback_events(clear=True) == []


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime required")
@pytest.mark.parametrize("variant", ["grid", "morton"])
@pytest.mark.parametrize(
    ("predicate", "expected"),
    [
        ("intersects", [1, 1, 0]),
        ("contains", [1, 0, 0]),
        ("covers", [1, 1, 0]),
        ("contains_properly", [1, 0, 0]),
        ("touches", [0, 1, 0]),
    ],
)
def test_public_pair_aggregate_forced_variants_match_boundary_oracle(
    tmp_path,
    variant,
    predicate,
    expected,
) -> None:
    from vibespatial.runtime.dispatch import (
        clear_dispatch_events,
        get_dispatch_events,
    )
    from vibespatial.spatial.point_partition import (
        PointPartitionVariant,
        force_point_partition_variant_for_testing,
    )

    points = points_from_xy([0.5, 0.0, 2.0], [0.5, 0.5, 2.0], crs="EPSG:3857")
    source = GeoDataFrame({"aligned": points}, geometry=points, crs=points.crs)
    path = tmp_path / f"forced-{variant}-{predicate}.parquet"
    source.to_parquet(path, geometry_encoding="geoarrow", index=False)
    source = read_parquet(path)
    left_index = source.geometry.sindex
    right_index = source.set_geometry("aligned").geometry.sindex
    regions = GeoSeries([box(0.0, 0.0, 1.0, 1.0)], crs=source.crs)
    clear_fallback_events()
    clear_dispatch_events()

    with force_point_partition_variant_for_testing(PointPartitionVariant(variant)):
        result = left_index.query_pair_aggregate(
            right_index,
            regions,
            predicate=predicate,
        )

    assert result.to_dict("list") == {
        "left_count": expected,
        "right_count": expected,
        "shared_count": expected,
    }
    assert get_fallback_events(clear=True) == []
    event = next(
        event
        for event in get_dispatch_events(clear=True)
        if event.surface == "vibespatial.api.SpatialIndex.query_pair_aggregate"
    )
    if variant == "morton":
        assert "_grid" not in event.implementation
    else:
        assert f"_{variant}" in event.implementation
        assert f"point-{variant}" in event.reason


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime required")
@pytest.mark.parametrize("variant", ["grid"])
def test_point_partition_cached_public_reduction_is_strict_native_and_bounded(
    tmp_path,
    variant,
) -> None:
    from vibespatial.api._native_public_arrays import NativeNumericExpressionArray
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )
    from vibespatial.spatial.point_partition import (
        PointPartitionVariant,
        force_point_partition_variant_for_testing,
    )
    from vibespatial.testing import strict_native_environment

    points = points_from_xy([0.25, 0.75, 2.0], [0.25, 0.75, 2.0], crs="EPSG:3857")
    source = GeoDataFrame({"aligned": points}, geometry=points, crs=points.crs)
    path = tmp_path / f"strict-native-{variant}.parquet"
    source.to_parquet(path, geometry_encoding="geoarrow", index=False)
    source = read_parquet(path)
    left_index = source.geometry.sindex
    right_index = source.set_geometry("aligned").geometry.sindex
    regions = GeoSeries([box(0.0, 0.0, 1.0, 1.0)], crs=source.crs)
    selected = PointPartitionVariant(variant)
    with force_point_partition_variant_for_testing(selected):
        left_index.query_pair_aggregate(right_index, regions, predicate="contains")
        reset_d2h_transfer_count()
        clear_materialization_events()
        clear_fallback_events()
        with strict_native_environment():
            result = left_index.query_pair_aggregate(
                right_index,
                regions,
                predicate="contains",
            )

    assert all(
        isinstance(result[column].array, NativeNumericExpressionArray)
        for column in result.columns
    )
    assert get_fallback_events(clear=True) == []
    assert get_materialization_events(clear=True) == []
    transfer_events = get_d2h_transfer_events(clear=True)
    assert transfer_events
    assert all("planning packet" in event.reason for event in transfer_events)
