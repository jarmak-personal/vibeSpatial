from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon

from vibespatial import Residency, TransferTrigger, from_shapely_geometries, has_gpu_runtime
from vibespatial.cuda._runtime import get_cuda_runtime
from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    FamilyGeometryBuffer,
    build_device_resident_owned,
    tile_single_row,
)
from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds
from vibespatial.runtime import ExecutionMode


def _sample_geometries():
    return [
        Point(1, 2),
        Point(),
        LineString([(0, 0), (2, 4)]),
        Polygon([(0, 0), (3, 0), (3, 3), (0, 0)]),
        None,
    ]


def test_owned_geometry_can_allocate_device_buffers() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    owned = from_shapely_geometries(_sample_geometries())
    owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="device allocation test",
    )

    assert owned.residency is Residency.DEVICE
    assert owned.device_state is not None
    assert owned.diagnostics_report()["device_buffers_allocated"] is True


def test_owned_geometry_round_trips_through_device_residency() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    owned = from_shapely_geometries(_sample_geometries())
    baseline = owned.to_shapely()

    owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="device round-trip test",
    )
    restored = owned.to_shapely()

    for left, right in zip(restored, baseline, strict=True):
        if left is None or right is None:
            assert left is right
            continue
        assert left.equals(right)


def test_device_subset_to_shapely_materializes_lazy_host_stubs() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    owned = from_shapely_geometries(
        [
            Polygon([(0, 0), (3, 0), (3, 3), (0, 0)]),
            Polygon([(5, 5), (8, 5), (8, 8), (5, 5)]),
            None,
        ],
        residency=Residency.DEVICE,
    )

    subset = owned.take(np.asarray([1, 0], dtype=np.int32))
    restored = subset.to_shapely()

    assert restored[0] is not None and restored[0].equals(owned.to_shapely()[1])
    assert restored[1] is not None and restored[1].equals(owned.to_shapely()[0])


def test_device_bounds_buffers_preserve_cached_values() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    owned = from_shapely_geometries(_sample_geometries(), residency=Residency.DEVICE)
    point_family = next(family for family in owned.families if family.value == "point")
    cached = np.asarray([[1.0, 2.0, 1.0, 2.0], [np.nan, np.nan, np.nan, np.nan]], dtype=np.float64)
    owned.cache_bounds(
        np.asarray(
            [
                [1.0, 2.0, 1.0, 2.0],
                [np.nan, np.nan, np.nan, np.nan],
                [0.0, 0.0, 2.0, 4.0],
                [0.0, 0.0, 3.0, 3.0],
                [np.nan, np.nan, np.nan, np.nan],
            ],
            dtype=np.float64,
        )
    )

    state = owned._ensure_device_state()
    assert state.families[point_family].bounds is not None
    assert np.allclose(
        get_cuda_runtime().copy_device_to_host(state.families[point_family].bounds),
        cached,
        equal_nan=True,
    )


def test_broadcast_device_bounds_cache_uses_physical_family_rows() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    base = from_shapely_geometries(
        [Polygon([(0, 0), (3, 0), (3, 3), (0, 0)])],
        residency=Residency.DEVICE,
    )
    broadcast = tile_single_row(base, 8)

    bounds = compute_geometry_bounds(broadcast, dispatch_mode=ExecutionMode.GPU)
    state = broadcast._ensure_device_state()
    polygon_family = next(family for family in broadcast.families if family.value == "polygon")
    family_bounds = get_cuda_runtime().copy_device_to_host(state.families[polygon_family].bounds)

    assert bounds.shape == (8, 4)
    assert family_bounds.shape == (1, 4)
    np.testing.assert_allclose(
        bounds,
        np.repeat(family_bounds, 8, axis=0),
        equal_nan=True,
    )


def test_cache_bounds_ignores_invalid_rows_with_stale_family_tags() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    owned = from_shapely_geometries(
        [
            Polygon([(0, 0), (2, 0), (2, 2), (0, 0)]),
            Polygon([(3, 0), (5, 0), (5, 2), (3, 0)]),
        ],
        residency=Residency.DEVICE,
    )
    polygon_family = next(family for family in owned.families if family.value == "polygon")
    polygon_tag = FAMILY_TAGS[polygon_family]

    owned._validity = np.asarray([True, False], dtype=np.bool_)
    owned._tags = np.asarray([polygon_tag, polygon_tag], dtype=np.int8)
    owned._family_row_offsets = np.asarray([0, 2], dtype=np.int32)

    owned.cache_bounds(
        np.asarray(
            [
                [0.0, 0.0, 2.0, 2.0],
                [np.nan, np.nan, np.nan, np.nan],
            ],
            dtype=np.float64,
        )
    )

    state = owned._ensure_device_state()
    family_bounds = get_cuda_runtime().copy_device_to_host(state.families[polygon_family].bounds)

    assert family_bounds.shape == (2, 4)
    np.testing.assert_allclose(family_bounds[0], [0.0, 0.0, 2.0, 2.0], equal_nan=True)
    assert np.isnan(family_bounds[1]).all()


def test_cache_bounds_prefers_device_family_row_count_when_host_stub_lags() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    owned = from_shapely_geometries(
        [
            Polygon([(0, 0), (2, 0), (2, 2), (0, 0)]),
            Polygon([(3, 0), (5, 0), (5, 2), (3, 0)]),
        ],
        residency=Residency.DEVICE,
    )
    polygon_family = next(family for family in owned.families if family.value == "polygon")
    host_buffer = owned.families[polygon_family]
    owned.families[polygon_family] = FamilyGeometryBuffer(
        family=host_buffer.family,
        schema=host_buffer.schema,
        row_count=1,
        x=host_buffer.x,
        y=host_buffer.y,
        geometry_offsets=host_buffer.geometry_offsets,
        empty_mask=host_buffer.empty_mask,
        part_offsets=host_buffer.part_offsets,
        ring_offsets=host_buffer.ring_offsets,
        bounds=host_buffer.bounds,
        host_materialized=host_buffer.host_materialized,
    )

    owned.cache_bounds(
        np.asarray(
            [
                [0.0, 0.0, 2.0, 2.0],
                [3.0, 0.0, 5.0, 2.0],
            ],
            dtype=np.float64,
        )
    )

    family_bounds = get_cuda_runtime().copy_device_to_host(
        owned._ensure_device_state().families[polygon_family].bounds
    )

    assert owned.families[polygon_family].row_count == 2
    np.testing.assert_allclose(
        family_bounds,
        np.asarray(
            [
                [0.0, 0.0, 2.0, 2.0],
                [3.0, 0.0, 5.0, 2.0],
            ],
            dtype=np.float64,
        ),
        equal_nan=True,
    )


def test_cache_bounds_prefers_device_family_offsets_when_host_metadata_lags() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    owned = from_shapely_geometries(
        [
            Polygon([(0, 0), (2, 0), (2, 2), (0, 0)]),
            Polygon([(3, 0), (5, 0), (5, 2), (3, 0)]),
        ],
        residency=Residency.DEVICE,
    )
    polygon_family = next(family for family in owned.families if family.value == "polygon")

    owned._validity = np.asarray([True, True], dtype=np.bool_)
    owned._tags = np.asarray(
        [FAMILY_TAGS[polygon_family], FAMILY_TAGS[polygon_family]],
        dtype=np.int8,
    )
    owned._family_row_offsets = np.asarray([0, 2], dtype=np.int32)

    owned.cache_bounds(
        np.asarray(
            [
                [0.0, 0.0, 2.0, 2.0],
                [3.0, 0.0, 5.0, 2.0],
            ],
            dtype=np.float64,
        )
    )

    family_bounds = get_cuda_runtime().copy_device_to_host(
        owned._ensure_device_state().families[polygon_family].bounds
    )

    np.testing.assert_allclose(
        family_bounds,
        np.asarray(
            [
                [0.0, 0.0, 2.0, 2.0],
                [3.0, 0.0, 5.0, 2.0],
            ],
            dtype=np.float64,
        ),
        equal_nan=True,
    )


def test_ensure_host_family_structure_hydrates_offsets_without_coordinates() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    runtime = get_cuda_runtime()
    polygon_family = GeometryFamily.POLYGON
    tags = np.asarray([FAMILY_TAGS[polygon_family]], dtype=np.int8)
    validity = np.asarray([True], dtype=np.bool_)
    family_row_offsets = np.asarray([0], dtype=np.int32)

    owned = build_device_resident_owned(
        device_families={
            polygon_family: DeviceFamilyGeometryBuffer(
                family=polygon_family,
                x=runtime.from_host(np.asarray([0.0, 2.0, 2.0, 0.0, 0.0], dtype=np.float64)),
                y=runtime.from_host(np.asarray([0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float64)),
                geometry_offsets=runtime.from_host(np.asarray([0, 1], dtype=np.int32)),
                empty_mask=runtime.from_host(np.asarray([False], dtype=np.bool_)),
                ring_offsets=runtime.from_host(np.asarray([0, 5], dtype=np.int32)),
            )
        },
        row_count=1,
        tags=tags,
        validity=validity,
        family_row_offsets=family_row_offsets,
    )

    stub = owned.families[polygon_family]
    assert not stub.host_materialized
    assert stub.x.size == 0
    assert stub.geometry_offsets.size == 0
    assert stub.ring_offsets is None

    owned._ensure_host_family_structure(polygon_family)

    hydrated = owned.families[polygon_family]
    assert hydrated.schema is get_geometry_buffer_schema(polygon_family)
    assert not hydrated.host_materialized
    assert hydrated.x.size == 0
    np.testing.assert_array_equal(hydrated.geometry_offsets, np.asarray([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(hydrated.empty_mask, np.asarray([False], dtype=np.bool_))
    np.testing.assert_array_equal(hydrated.ring_offsets, np.asarray([0, 5], dtype=np.int32))


def test_runtime_from_host_passes_through_device_arrays(strict_device_guard) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import cupy as cp

    runtime = get_cuda_runtime()
    d_input = cp.asarray(np.asarray([1, 2, 3], dtype=np.int32))

    d_output = runtime.from_host(d_input)

    assert isinstance(d_output, cp.ndarray)
    assert int(d_output.data.ptr) == int(d_input.data.ptr)


def test_build_device_resident_owned_accepts_device_metadata_arrays(strict_device_guard) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import cupy as cp

    runtime = get_cuda_runtime()
    polygon_family = GeometryFamily.POLYGON

    owned = build_device_resident_owned(
        device_families={
            polygon_family: DeviceFamilyGeometryBuffer(
                family=polygon_family,
                x=runtime.from_host(np.asarray([0.0, 2.0, 2.0, 0.0, 0.0], dtype=np.float64)),
                y=runtime.from_host(np.asarray([0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float64)),
                geometry_offsets=runtime.from_host(np.asarray([0, 1], dtype=np.int32)),
                empty_mask=runtime.from_host(np.asarray([False], dtype=np.bool_)),
                ring_offsets=runtime.from_host(np.asarray([0, 5], dtype=np.int32)),
            )
        },
        row_count=1,
        tags=cp.asarray(np.asarray([FAMILY_TAGS[polygon_family]], dtype=np.int8)),
        validity=cp.asarray(np.asarray([True], dtype=np.bool_)),
        family_row_offsets=cp.asarray(np.asarray([0], dtype=np.int32)),
    )

    assert owned.row_count == 1
    assert owned.residency is Residency.DEVICE
    assert owned.device_state is not None
    assert owned._validity is None
    assert owned._tags is None
    assert owned._family_row_offsets is None
    assert owned.families[polygon_family].row_count == 1


def test_build_device_resident_owned_rejects_host_metadata_in_gpu_mode() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import cupy as cp

    runtime = get_cuda_runtime()
    polygon_family = GeometryFamily.POLYGON

    with pytest.raises(AssertionError, match="host metadata"):
        build_device_resident_owned(
            device_families={
                polygon_family: DeviceFamilyGeometryBuffer(
                    family=polygon_family,
                    x=cp.asarray(np.asarray([0.0, 2.0, 2.0, 0.0, 0.0], dtype=np.float64)),
                    y=cp.asarray(np.asarray([0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float64)),
                    geometry_offsets=runtime.from_host(np.asarray([0, 1], dtype=np.int32)),
                    empty_mask=runtime.from_host(np.asarray([False], dtype=np.bool_)),
                    ring_offsets=runtime.from_host(np.asarray([0, 5], dtype=np.int32)),
                )
            },
            row_count=1,
            tags=np.asarray([FAMILY_TAGS[polygon_family]], dtype=np.int8),
            validity=np.asarray([True], dtype=np.bool_),
            family_row_offsets=np.asarray([0], dtype=np.int32),
            execution_mode="gpu",
        )
