from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import Point

import vibespatial.api as geopandas
from vibespatial import has_gpu_runtime
from vibespatial.api._native_grouped import NativeGrouped
from vibespatial.api.testing import assert_geodataframe_equal
from vibespatial.constructive.grouped_point_union import (
    grouped_point_union_owned,
    supports_grouped_point_union,
)
from vibespatial.constructive.point import point_owned_from_xy_device
from vibespatial.cuda._runtime import assert_zero_d2h_transfers, reset_d2h_transfer_count
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import FamilyGeometryBuffer
from vibespatial.runtime import ExecutionMode, set_requested_mode
from vibespatial.runtime.crossover import default_crossover_policy
from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events
from vibespatial.runtime.execution_trace import execution_trace
from vibespatial.runtime.precision import KernelClass


def test_points_from_xy_builds_owned_point_storage_without_shapely_objects() -> None:
    geometry = geopandas.points_from_xy([1.0, 2.0], [3.0, 4.0], crs="EPSG:3857")

    assert geometry._owned is not None
    assert geometry._shapely_data is None
    assert geometry.crs.to_epsg() == 3857
    assert shapely.to_wkt(geometry._data).tolist() == ["POINT (1 3)", "POINT (2 4)"]


def test_points_from_xy_z_keeps_upstream_three_dimensional_contract() -> None:
    geometry = geopandas.points_from_xy([1.0], [2.0], [3.0])

    assert bool(shapely.has_z(geometry[0]))
    assert shapely.get_z(geometry[0]) == 3.0


@pytest.mark.parametrize(
    ("x", "y"),
    [
        (np.asarray([[1.0, 2.0]]), np.asarray([[3.0, 4.0]])),
        (np.asarray([1.0, 2.0]), np.asarray([[3.0, 4.0]])),
    ],
)
def test_points_from_xy_rejects_non_1d_inputs(x, y) -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        geopandas.points_from_xy(x, y)


def test_grouped_point_union_declines_empty_points() -> None:
    owned = geopandas.GeoSeries([Point(), Point(1.0, 1.0)]).array.to_owned()
    grouped = NativeGrouped.from_dense_codes(np.asarray([0, 0], dtype=np.int32), group_count=1)

    # The support probe is intentionally structural so small AUTO workloads do
    # not pay for semantic scans.  Explicit GPU selection reaches the exact
    # semantic gate and declines empty Point rows.
    assert supports_grouped_point_union(owned)
    with set_requested_mode(ExecutionMode.GPU):
        assert grouped_point_union_owned(grouped, owned) is None


def test_grouped_point_union_uses_measured_10k_auto_crossover() -> None:
    policy = default_crossover_policy("segmented_point_union", KernelClass.CONSTRUCTIVE)

    assert policy.auto_min_rows == 10_000


def test_small_auto_decline_precedes_grouped_point_semantic_scan(monkeypatch) -> None:
    import vibespatial.constructive.grouped_point_union as grouped_point_union_module

    owned = geopandas.points_from_xy([1.0, 2.0], [3.0, 4.0]).to_owned()
    grouped = NativeGrouped.from_dense_codes(np.asarray([0, 0], dtype=np.int32), group_count=1)

    def fail_semantic_scan(_owned):
        raise AssertionError("small AUTO workload reached semantic coordinate scans")

    monkeypatch.setattr(
        grouped_point_union_module,
        "_grouped_point_union_semantically_admissible",
        fail_semantic_scan,
    )
    with set_requested_mode(ExecutionMode.AUTO):
        result = grouped_point_union_owned(grouped, owned)

    assert result is None


@pytest.mark.gpu
def test_grouped_point_union_matches_shapely_and_deduplicates() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    x = np.asarray([2.0, 1.0, 2.0, -0.0, 0.0, 9.0], dtype=np.float64)
    y = np.asarray([2.0, 1.0, 2.0, 0.0, -0.0, 9.0], dtype=np.float64)
    codes = np.asarray([0, 0, 0, 1, 1, 2], dtype=np.int32)
    owned = point_owned_from_xy_device(x, y)
    grouped = NativeGrouped.from_dense_codes(codes, group_count=3)

    clear_dispatch_events()
    with set_requested_mode(ExecutionMode.GPU):
        result = grouped_point_union_owned(grouped, owned)

    assert result is not None
    actual = np.asarray(result.to_shapely(), dtype=object)
    source = shapely.points(x, y)
    expected = np.asarray(
        [shapely.union_all(source[codes == group]) for group in range(3)],
        dtype=object,
    )
    assert [geometry.geom_type for geometry in actual] == [
        geometry.geom_type for geometry in expected
    ]
    assert all(
        bool(shapely.equals_exact(left, right, tolerance=0.0))
        for left, right in zip(actual, expected, strict=True)
    )
    event = next(
        event
        for event in reversed(get_dispatch_events(clear=True))
        if event.operation == "grouped_point_union"
    )
    assert f"coordinate_capacity={x.size}" in event.detail
    assert "unique_coordinates=" not in event.detail


@pytest.mark.gpu
def test_grouped_point_union_accepts_device_only_point_carrier_without_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import cupy as cp

    x = np.asarray([2.0, 1.0, 2.0, 9.0], dtype=np.float64)
    y = np.asarray([2.0, 1.0, 2.0, 9.0], dtype=np.float64)
    owned = point_owned_from_xy_device(x, y)
    host_point = owned.families[GeometryFamily.POINT]
    owned.families[GeometryFamily.POINT] = FamilyGeometryBuffer(
        family=host_point.family,
        schema=host_point.schema,
        row_count=host_point.row_count,
        x=np.empty(0, dtype=np.float64),
        y=np.empty(0, dtype=np.float64),
        geometry_offsets=np.empty(0, dtype=np.int32),
        empty_mask=np.empty(0, dtype=np.bool_),
        bounds=None,
        host_materialized=False,
    )
    owned._validity = None
    owned._tags = None
    owned._family_row_offsets = None
    grouped = NativeGrouped.from_dense_codes(
        cp.asarray([0, 0, 0, 1], dtype=cp.int32),
        group_count=2,
    )

    assert supports_grouped_point_union(owned)
    reset_d2h_transfer_count()
    with (
        execution_trace("grouped-point-device-native") as trace,
        assert_zero_d2h_transfers(),
        set_requested_mode(ExecutionMode.GPU),
    ):
        result = grouped_point_union_owned(grouped, owned)

    assert result is not None
    assert result.residency.value == "device"
    assert result._validity is None
    assert result._tags is None
    assert result._family_row_offsets is None
    assert result.device_state is not None
    point_buffer = result.device_state.families[GeometryFamily.POINT]
    multipoint_buffer = result.device_state.families[GeometryFamily.MULTIPOINT]
    # Dynamic deduplication retains source-row coordinate capacity.  Logical
    # output size lives in the device offsets, so success cannot depend on a
    # host scalar allocation packet or CuPy boolean compaction.
    assert int(point_buffer.x.size) == x.size
    assert int(point_buffer.geometry_offsets.size) == x.size + 1
    assert int(multipoint_buffer.x.size) == x.size
    assert cp.asnumpy(multipoint_buffer.geometry_offsets).tolist() == [0, 2, 3]
    assert trace.transfers == []
    reset_d2h_transfer_count()


@pytest.mark.gpu
def test_grouped_point_union_observes_single_host_group_code_ingress() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    owned = point_owned_from_xy_device(
        np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
    )
    grouped = NativeGrouped.from_dense_codes(
        np.asarray([0, 0, 1, 1], dtype=np.int32),
        group_count=2,
    )

    with (
        execution_trace("grouped-point-host-code-ingress") as trace,
        set_requested_mode(ExecutionMode.GPU),
    ):
        result = grouped_point_union_owned(grouped, owned)

    assert result is not None
    assert trace.summary()["d2h_transfers"] == 0
    assert trace.summary()["h2d_transfers"] == 1
    transfer = trace.transfers[0]
    assert transfer.direction == "h2d"
    assert transfer.reason == "grouped Point union dense group-code ingress"
    assert transfer.bytes_transferred == 4 * np.dtype(np.int32).itemsize


@pytest.mark.gpu
@pytest.mark.parametrize("non_finite", [np.nan, np.inf, -np.inf])
def test_grouped_point_union_declines_non_finite_device_carrier_without_d2h(
    non_finite,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    owned = point_owned_from_xy_device(
        np.asarray([non_finite, 1.0], dtype=np.float64),
        np.asarray([0.0, 1.0], dtype=np.float64),
    )
    grouped = NativeGrouped.from_dense_codes(
        np.asarray([0, 0], dtype=np.int32),
        group_count=1,
    )

    assert supports_grouped_point_union(owned)
    assert owned.device_state is not None
    assert owned.device_state.trusted_all_finite_coordinates is False
    reset_d2h_transfer_count()
    with assert_zero_d2h_transfers(), set_requested_mode(ExecutionMode.GPU):
        result = grouped_point_union_owned(grouped, owned)

    assert result is None
    reset_d2h_transfer_count()


@pytest.mark.gpu
def test_finite_proof_false_becomes_unknown_after_device_selection() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import cupy as cp

    owned = point_owned_from_xy_device(
        np.asarray([np.nan, 1.0], dtype=np.float64),
        np.asarray([0.0, 1.0], dtype=np.float64),
    )
    assert owned.device_state is not None
    assert owned.device_state.trusted_all_finite_coordinates is False

    reset_d2h_transfer_count()
    with assert_zero_d2h_transfers():
        selected = owned.device_take(cp.asarray([1], dtype=cp.int64))
        selected_state = selected._ensure_device_state(preserve_indexed_view=True)

    assert selected_state.trusted_all_finite_coordinates is None
    reset_d2h_transfer_count()


@pytest.mark.gpu
def test_public_point_dissolve_uses_exact_native_grouped_shape() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    frame = geopandas.GeoDataFrame(
        {
            "group": [0, 0, 0, 1, 1],
            "value": [1, 2, 3, 4, 5],
        },
        geometry=geopandas.points_from_xy(
            [2.0, 1.0, 2.0, 10.0, 11.0],
            [2.0, 1.0, 2.0, 10.0, 11.0],
        ),
    )
    with set_requested_mode(ExecutionMode.CPU):
        expected = frame.copy().dissolve(by="group", aggfunc="first")

    with set_requested_mode(ExecutionMode.GPU):
        actual = frame.dissolve(by="group", aggfunc="first")

    assert_geodataframe_equal(actual, expected)
    assert getattr(
        actual.geometry.array._owned,
        "_native_grouped_union_implementation",
        None,
    ) == "native_segmented_point_set_union"
