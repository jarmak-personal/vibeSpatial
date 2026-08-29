from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Point, Polygon, box

import vibespatial.api as geopandas
from vibespatial.api.geometry_array import GeometryArray
from vibespatial.geometry.owned import DiagnosticKind
from vibespatial.runtime._runtime import has_gpu_runtime
from vibespatial.runtime.residency import Residency

needs_gpu = pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime not available")


_SLICE_CASES = (
    slice(1, 7, 2),
    slice(None, None, -1),
    slice(100, 200, 1),
    slice(-100, 100, 3),
    slice(7, -100, -2),
)


def _point_geometry_array(row_count: int = 8) -> GeometryArray:
    data = np.empty(row_count, dtype=object)
    data[:] = [Point(float(row), float(row + 10)) for row in range(row_count)]
    return GeometryArray(data)


def _device_point_geometry_array(row_count: int = 8) -> GeometryArray:
    import cupy as cp

    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import (
        FAMILY_TAGS,
        DeviceFamilyGeometryBuffer,
        build_device_resident_owned,
    )

    family = GeometryFamily.POINT
    owned = build_device_resident_owned(
        device_families={
            family: DeviceFamilyGeometryBuffer(
                family=family,
                x=cp.arange(row_count, dtype=cp.float64),
                y=cp.arange(row_count, dtype=cp.float64) + 10.0,
                geometry_offsets=cp.arange(row_count + 1, dtype=cp.int32),
                empty_mask=cp.zeros(row_count, dtype=cp.bool_),
            )
        },
        row_count=row_count,
        tags=cp.full(row_count, FAMILY_TAGS[family], dtype=cp.int8),
        validity=cp.ones(row_count, dtype=cp.bool_),
        family_row_offsets=cp.arange(row_count, dtype=cp.int32),
        execution_mode="gpu",
    )
    return GeometryArray.from_owned(owned)


@pytest.mark.parametrize("row_slice", _SLICE_CASES)
def test_geometry_array_host_slice_keeps_python_semantics(row_slice: slice) -> None:
    array = _point_geometry_array()
    expected_rows = np.arange(len(array), dtype=np.int64)[row_slice]

    result = array[row_slice]

    assert isinstance(result, GeometryArray)
    assert [int(point.x) for point in result] == expected_rows.tolist()


def test_geometry_array_host_zero_step_slice_raises() -> None:
    with pytest.raises(ValueError, match="slice step cannot be zero"):
        _point_geometry_array()[::0]


@needs_gpu
def test_geometry_array_full_slice_preserves_device_owned_index_caches() -> None:
    from vibespatial.cuda._runtime import assert_zero_d2h_transfers

    array = _device_point_geometry_array()
    spatial_index = object()
    flat_spatial_index = object()
    array._sindex = spatial_index
    array._owned_flat_sindex = flat_spatial_index
    array._owned_spatial_input_supported = True

    with assert_zero_d2h_transfers():
        result = array[:]

    assert result._owned is array._owned
    assert result._shapely_data is None
    assert result._sindex is spatial_index
    assert result._owned_flat_sindex is flat_spatial_index
    assert result._owned_spatial_input_supported is True


@needs_gpu
@pytest.mark.parametrize("row_slice", _SLICE_CASES)
def test_geometry_array_device_slice_is_row_indirection_without_host_transfer(
    row_slice: slice,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cupy as cp

    from vibespatial.cuda._runtime import assert_zero_d2h_transfers

    array = _device_point_geometry_array()
    source_owned = array._owned
    expected_rows = np.arange(len(array), dtype=np.int64)[row_slice]
    host_index_uploads: list[type] = []
    original_asarray = cp.asarray

    def _track_host_index_upload(value, *args, **kwargs):
        if isinstance(value, (np.ndarray, list, tuple)):
            host_index_uploads.append(type(value))
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(cp, "asarray", _track_host_index_upload)
    with assert_zero_d2h_transfers():
        result = array[row_slice]

    assert array._shapely_data is None
    assert result._shapely_data is None
    assert result._owned is not None
    assert result._owned.residency is Residency.DEVICE
    assert result._owned.is_indexed_view
    assert result._owned._base is source_owned
    assert host_index_uploads == []
    assert np.array_equal(cp.asnumpy(result._owned._index_map), expected_rows)
    assert not any(
        event.kind is DiagnosticKind.MATERIALIZATION
        for event in (*source_owned.diagnostics, *result._owned.diagnostics)
    )


@needs_gpu
def test_geometry_array_device_boolean_subset_retains_exact_base_row_indirection() -> None:
    import cupy as cp

    from vibespatial.cuda._runtime import assert_zero_d2h_transfers

    array = _device_point_geometry_array()
    source_owned = array._owned

    with assert_zero_d2h_transfers():
        result = array[np.asarray([True, False, True, False, False, True, False, True])]

    assert result._owned is not None
    assert result._owned.residency is Residency.DEVICE
    assert result._owned.is_indexed_view
    assert result._owned._base is source_owned
    assert result._owned._index_map_unique
    assert cp.asnumpy(result._owned._index_map).tolist() == [0, 2, 5, 7]


@needs_gpu
def test_public_iloc_slice_stays_device_resident_and_gpu_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cupy as cp

    from vibespatial.cuda._runtime import assert_zero_d2h_transfers

    geometry = _device_point_geometry_array()
    frame = geopandas.GeoDataFrame(
        {
            "row_id": np.arange(len(geometry), dtype=np.int64),
            "geometry": geometry,
        }
    )
    source_array = frame.geometry.values
    source_owned = source_array._owned
    host_index_uploads: list[type] = []
    original_asarray = cp.asarray

    def _track_host_index_upload(value, *args, **kwargs):
        if isinstance(value, (np.ndarray, list, tuple)):
            host_index_uploads.append(type(value))
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(cp, "asarray", _track_host_index_upload)
    with assert_zero_d2h_transfers():
        selected = frame.iloc[::-2]

    selected_array = selected.geometry.values
    assert selected.row_id.tolist() == [7, 5, 3, 1]
    assert selected.index.tolist() == [7, 5, 3, 1]
    assert source_array._shapely_data is None
    assert selected_array._shapely_data is None
    assert selected_array._owned is not None
    assert selected_array._owned.residency is Residency.DEVICE
    assert selected_array._owned.is_indexed_view
    assert selected_array._owned._base is source_owned
    assert host_index_uploads == []
    assert cp.asnumpy(selected_array._owned._index_map).tolist() == [7, 5, 3, 1]
    assert not any(
        event.kind is DiagnosticKind.MATERIALIZATION
        for event in (*source_owned.diagnostics, *selected_array._owned.diagnostics)
    )

    # A native constructive consumer may physicalize the rowset on device,
    # but it must not require a host geometry bridge to do so.
    with assert_zero_d2h_transfers():
        reversed_geometry = selected.geometry.reverse()

    assert reversed_geometry.values._owned is not None
    assert reversed_geometry.values._owned.residency is Residency.DEVICE
    assert reversed_geometry.values._shapely_data is None


@needs_gpu
def test_owned_flat_sindex_auto_uses_device_state_and_caches_without_d2h() -> None:
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )
    from vibespatial.runtime import ExecutionMode, set_requested_mode

    geometry = _device_point_geometry_array(4_096)
    source_owned = geometry._owned
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)

    with set_requested_mode(ExecutionMode.AUTO):
        indexed_owned, flat_index = geometry.owned_flat_sindex()

    build_events = get_d2h_transfer_events(clear=True)
    assert indexed_owned is source_owned
    assert source_owned.runtime_history[-1].selected is ExecutionMode.GPU
    assert source_owned.residency is Residency.DEVICE
    assert flat_index.device_bounds is not None
    assert flat_index.device_order is not None
    assert flat_index.device_morton_keys is not None
    assert flat_index._host_bounds is None
    assert flat_index._host_order is None
    assert flat_index._host_morton_keys is None
    assert geometry._shapely_data is None
    assert [event.reason for event in build_events] == [
        "flat spatial index device total-bounds scalar fence"
    ]
    assert build_events[0].bytes_transferred == 5 * np.dtype(np.float64).itemsize
    assert not any(
        event.kind is DiagnosticKind.MATERIALIZATION for event in source_owned.diagnostics
    )

    with assert_zero_d2h_transfers(), set_requested_mode(ExecutionMode.AUTO):
        cached_owned, cached_index = geometry.owned_flat_sindex()

    assert cached_owned is source_owned
    assert cached_index is flat_index


@needs_gpu
def test_owned_flat_sindex_compacts_variable_width_device_row_view() -> None:
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import from_shapely_geometries
    from vibespatial.runtime import ExecutionMode, set_requested_mode

    polygons = [
        (
            box(float(row), 0.0, float(row + 1), 1.0)
            if row % 2 == 0
            else Polygon(
                [
                    (float(row), 0.0),
                    (float(row + 1), 0.0),
                    (float(row), 1.0),
                    (float(row), 0.0),
                ]
            )
        )
        for row in range(8)
    ]
    source_owned = from_shapely_geometries(polygons, residency=Residency.DEVICE)
    source = GeometryArray.from_owned(source_owned)
    selected = source[np.asarray([False, True, False, True] + [False] * 4)]

    assert selected._owned.is_indexed_view
    assert selected._owned._base is source_owned

    with set_requested_mode(ExecutionMode.GPU):
        indexed_owned, flat_index = selected.owned_flat_sindex()

    polygon = indexed_owned.device_state.families[GeometryFamily.POLYGON]
    assert indexed_owned is selected._owned
    assert indexed_owned.row_count == 2
    assert not indexed_owned.is_indexed_view
    assert indexed_owned._base is None
    assert int(polygon.x.size) == 8
    assert int(flat_index.device_bounds.shape[0]) == 2


@needs_gpu
def test_owned_query_input_compacts_variable_width_geoseries_row_view() -> None:
    from vibespatial.api.sindex import SpatialIndex
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import from_shapely_geometries

    polygons = [
        (
            box(float(row), 0.0, float(row + 1), 1.0)
            if row % 2 == 0
            else Polygon(
                [
                    (float(row), 0.0),
                    (float(row + 1), 0.0),
                    (float(row), 1.0),
                    (float(row), 0.0),
                ]
            )
        )
        for row in range(8)
    ]
    source_owned = from_shapely_geometries(polygons, residency=Residency.DEVICE)
    selected = GeometryArray.from_owned(source_owned)[
        np.asarray([False, True, False, True] + [False] * 4)
    ]
    series = geopandas.GeoSeries(selected)

    assert series.values._owned.is_indexed_view
    compacted = SpatialIndex._owned_query_input(series)

    polygon = compacted.device_state.families[GeometryFamily.POLYGON]
    assert compacted is series.values._owned
    assert compacted.row_count == 2
    assert not compacted.is_indexed_view
    assert compacted._base is None
    assert int(polygon.x.size) == 8


@needs_gpu
def test_spatial_compaction_exactly_sizes_device_only_variable_width_selection(
    monkeypatch,
) -> None:
    import cupy as cp

    from vibespatial.cuda._runtime import get_cuda_runtime
    from vibespatial.geometry import owned as owned_module
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import (
        FAMILY_TAGS,
        DeviceFamilyGeometryBuffer,
        build_device_resident_owned,
    )
    from vibespatial.spatial.indexing import compact_indexed_spatial_input

    family = GeometryFamily.POLYGON
    coordinate_capacity = 1_005
    source = build_device_resident_owned(
        device_families={
            family: DeviceFamilyGeometryBuffer(
                family=family,
                x=cp.arange(coordinate_capacity, dtype=cp.float64),
                y=cp.arange(coordinate_capacity, dtype=cp.float64),
                geometry_offsets=cp.asarray([0, 1, 2], dtype=cp.int32),
                ring_offsets=cp.asarray([0, 5, coordinate_capacity], dtype=cp.int32),
                empty_mask=cp.zeros(2, dtype=cp.bool_),
            )
        },
        row_count=2,
        tags=cp.full(2, FAMILY_TAGS[family], dtype=cp.int8),
        validity=cp.ones(2, dtype=cp.bool_),
        family_row_offsets=cp.arange(2, dtype=cp.int32),
        execution_mode="gpu",
    )
    selected = source._device_indexed_take(
        cp.asarray([0], dtype=cp.int64),
        assume_unique_indices=True,
    )
    runtime = get_cuda_runtime()
    runtime.memory_admission_events(clear=True)
    original_take = owned_module._device_take_family_buffer

    def _assert_admitted_before_family_allocation(*args, **kwargs):
        admissions = runtime.memory_admission_events()
        assert len(admissions) == 2
        assert admissions[0].stage == "geometry.exact_row_physicalization.plan"
        assert admissions[0].admitted
        assert admissions[-1].stage == "geometry.exact_row_physicalization"
        assert admissions[-1].admitted
        return original_take(*args, **kwargs)

    monkeypatch.setattr(
        owned_module,
        "_device_take_family_buffer",
        _assert_admitted_before_family_allocation,
    )

    compacted = compact_indexed_spatial_input(selected)

    polygon = compacted.device_state.families[family]
    assert not compacted.is_indexed_view
    assert compacted.row_count == 1
    assert int(polygon.x.size) == 5
    assert int(polygon.y.size) == 5


@needs_gpu
def test_spatial_compaction_declines_before_exact_row_planning_allocations(
    monkeypatch,
) -> None:
    import cupy as cp

    from vibespatial.cuda._runtime import DeviceMemoryAdmission, get_cuda_runtime
    from vibespatial.geometry import owned as owned_module
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import (
        FAMILY_TAGS,
        DeviceFamilyGeometryBuffer,
        build_device_resident_owned,
        device_physicalize_owned_row_selections_exact,
    )

    family = GeometryFamily.POLYGON
    source = build_device_resident_owned(
        device_families={
            family: DeviceFamilyGeometryBuffer(
                family=family,
                x=cp.asarray([0.0, 1.0, 1.0, 0.0, 0.0]),
                y=cp.asarray([0.0, 0.0, 1.0, 1.0, 0.0]),
                geometry_offsets=cp.asarray([0, 1], dtype=cp.int32),
                ring_offsets=cp.asarray([0, 5], dtype=cp.int32),
                empty_mask=cp.zeros(1, dtype=cp.bool_),
            )
        },
        row_count=1,
        tags=cp.full(1, FAMILY_TAGS[family], dtype=cp.int8),
        validity=cp.ones(1, dtype=cp.bool_),
        family_row_offsets=cp.zeros(1, dtype=cp.int32),
        execution_mode="gpu",
    )
    runtime = get_cuda_runtime()

    def _decline_plan(*, stage, required_bytes, requested_units=0):
        assert stage == "geometry.exact_row_physicalization.plan"
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

    def _unexpected_flatten(*args, **kwargs):
        raise AssertionError("planning allocation ran before admission")

    monkeypatch.setattr(runtime, "admit_device_memory", _decline_plan)
    monkeypatch.setattr(
        owned_module,
        "_flatten_exact_device_row_selection",
        _unexpected_flatten,
    )

    with pytest.raises(MemoryError, match="planning requires"):
        device_physicalize_owned_row_selections_exact(
            [(source, None)],
            reason="declined exact planning test",
        )


@needs_gpu
def test_spatial_compaction_all_null_output_is_fully_admitted(monkeypatch) -> None:
    import cupy as cp

    from vibespatial.cuda._runtime import get_cuda_runtime
    from vibespatial.geometry.owned import build_null_owned_array
    from vibespatial.spatial.indexing import compact_indexed_spatial_input

    row_count = 7
    source = build_null_owned_array(row_count + 1, residency=Residency.DEVICE)
    selected = source._device_indexed_take(cp.arange(row_count, dtype=cp.int64))
    monkeypatch.setattr(selected, "_device_take_prefers_row_indirection", lambda: True)
    runtime = get_cuda_runtime()
    runtime.memory_admission_events(clear=True)

    compacted = compact_indexed_spatial_input(selected)

    admissions = runtime.memory_admission_events(clear=True)
    assert [event.stage for event in admissions] == [
        "geometry.exact_row_physicalization.plan",
        "geometry.exact_row_physicalization",
    ]
    assert admissions[-1].required_bytes >= row_count * 6
    assert compacted.row_count == row_count
    assert not compacted.is_indexed_view
    assert not bool(cp.any(compacted.device_state.validity))


def test_owned_flat_sindex_explicit_cpu_keeps_host_execution() -> None:
    from vibespatial.runtime import ExecutionMode, set_requested_mode

    geometry = _point_geometry_array().to_owned()
    array = GeometryArray.from_owned(geometry)

    with set_requested_mode(ExecutionMode.CPU):
        indexed_owned, flat_index = array.owned_flat_sindex()

    assert indexed_owned is geometry
    assert geometry.runtime_history[0].requested is ExecutionMode.CPU
    assert geometry.runtime_history[0].selected is ExecutionMode.CPU
    assert flat_index.device_order is None
    assert flat_index.device_morton_keys is None
    assert flat_index._host_bounds is not None
    assert flat_index._host_order is not None
    assert flat_index._host_morton_keys is not None


@pytest.mark.parametrize("row_slice", _SLICE_CASES)
def test_public_iloc_host_slice_keeps_attribute_and_geometry_order(
    row_slice: slice,
) -> None:
    geometry = _point_geometry_array()
    frame = geopandas.GeoDataFrame(
        {
            "row_id": np.arange(len(geometry), dtype=np.int64),
            "geometry": geometry,
        }
    )
    expected_rows = np.arange(len(frame), dtype=np.int64)[row_slice]

    result = frame.iloc[row_slice]

    assert result.row_id.tolist() == expected_rows.tolist()
    assert result.index.tolist() == expected_rows.tolist()
    assert [int(point.x) for point in result.geometry] == expected_rows.tolist()
