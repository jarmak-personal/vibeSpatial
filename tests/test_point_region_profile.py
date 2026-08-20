from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import MultiPolygon, box

from vibespatial.api import GeoDataFrame, GeoSeries, points_from_xy, read_parquet
from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_I64,
    KERNEL_PARAM_PTR,
    get_cuda_runtime,
)
from vibespatial.predicates.point_location_index import (
    point_location_part_y_index_profile_kernels,
)
from vibespatial.predicates.point_location_index_kernels import (
    _POINT_LOCATION_PART_Y_INDEX_PROFILE_SOURCE,
    _POINT_LOCATION_PART_Y_INDEX_SOURCE,
)
from vibespatial.predicates.point_region_profile import profile_point_region
from vibespatial.runtime import has_gpu_runtime
from vibespatial.runtime.fallbacks import clear_fallback_events, get_fallback_events


def test_point_region_profiler_is_absent_from_production_kernel_source() -> None:
    """Disabled profiling must not add counters or atomics to production code."""
    assert "VS_PROFILE_COUNTER_COUNT" not in _POINT_LOCATION_PART_Y_INDEX_SOURCE
    assert "_profiled" not in _POINT_LOCATION_PART_Y_INDEX_SOURCE
    assert "VS_PROFILE_COUNTER_COUNT" in _POINT_LOCATION_PART_Y_INDEX_PROFILE_SOURCE
    assert "point_in_multipolygon_prepared_part_y_index_profiled" in (
        _POINT_LOCATION_PART_Y_INDEX_PROFILE_SOURCE
    )


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
    assert group["launches"] == 3
    assert group["counters"]["candidates"] == 6
    assert group["counters"]["parts_considered"] == 9
    assert group["counters"]["active_parts"] == 6
    assert group["counters"]["edges_visited"] == 12
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
