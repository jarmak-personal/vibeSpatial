from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from shapely.geometry import LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon

if TYPE_CHECKING:
    from vibespatial.geometry.owned import OwnedGeometryArray

from vibespatial import (
    BufferSharingMode,
    DiagnosticKind,
    ExecutionMode,
    Residency,
    RuntimeSelection,
    TransferTrigger,
    compute_geometry_bounds,
    compute_morton_keys,
    compute_offset_spans,
    compute_total_bounds,
    from_geoarrow,
    from_shapely_geometries,
    from_wkb,
    has_gpu_runtime,
)
from vibespatial.geometry.buffers import GeometryFamily


@pytest.mark.gpu
def test_device_geometry_composition_requires_explicit_owned_physicalization() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    pytest.importorskip("cupy")
    from shapely.geometry import box

    from vibespatial.api._native_result_core import (
        GeometryNativeResult,
        NativeGeometryComposition,
    )
    from vibespatial.cuda._runtime import assert_zero_d2h_transfers
    from vibespatial.geometry.device_array import DeviceGeometryArray

    parts = [
        GeometryNativeResult.from_owned(
            from_shapely_geometries([geometry], residency=Residency.DEVICE),
            crs=None,
        )
        for geometry in (box(0.0, 0.0, 1.0, 1.0), box(2.0, 0.0, 3.0, 1.0))
    ]
    composition = NativeGeometryComposition.concat(parts, crs=None)
    array = DeviceGeometryArray._from_composition(composition, crs=None)

    assert array.cached_owned() is None
    assert array.native_composition is composition
    assert composition._singular_owned_cache is None
    with pytest.raises(AttributeError, match="physicalize_owned"):
        _ = array._owned

    with assert_zero_d2h_transfers():
        physical = array.physicalize_owned()

    assert physical.row_count == 2
    assert array.cached_owned() is physical
    assert array.native_composition is None


def test_geometry_host_bridge_d2h_exports_are_runtime_accounted() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paths = (
        repo_root / "src" / "vibespatial" / "geometry" / "device_array.py",
        repo_root / "src" / "vibespatial" / "geometry" / "equality.py",
    )
    offenders: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr == "asnumpy":
                offenders.append(f"{path.relative_to(repo_root)}:{node.lineno}")
            if func.attr == "copy_device_to_host" and not any(
                keyword.arg == "reason" for keyword in node.keywords
            ):
                offenders.append(f"{path.relative_to(repo_root)}:{node.lineno}")
    assert offenders == []


def test_device_take_nested_sizing_uses_structure_or_capacity() -> None:
    source = (
        Path(__file__)
        .resolve()
        .parents[1]
        .joinpath("src/vibespatial/geometry/owned.py")
        .read_text()
    )
    planner_start = source.index("def _device_take_family_size_plan_from_device(")
    planner_end = source.index("\ndef ", planner_start + 1)
    planner_source = source[planner_start:planner_end]
    take_start = source.index("def _device_take_family_buffer(")
    take_end = source.index("\ndef ", take_start + 1)
    take_source = source[take_start:take_end]

    assert "count_scatter_total" not in planner_source
    assert "device-take nested slice-size allocation fence" not in planner_source
    assert "_device_take_capacity_multiplier" in take_source
    assert "if allow_capacity_allocation" not in take_source
    assert "owned geometry device-take slice-size allocation fence" not in source


@pytest.mark.parametrize(
    ("family", "geometries", "expected"),
    (
        (
            GeometryFamily.MULTIPOINT,
            (MultiPoint([(0, 0)]), MultiPoint([(0, 0), (1, 1), (2, 2)])),
            (None, None, None, None, None, 3),
        ),
        (
            GeometryFamily.LINESTRING,
            (LineString([(0, 0), (1, 1)]), LineString([(0, 0), (1, 1), (2, 1)])),
            (None, None, None, None, None, 3),
        ),
        (
            GeometryFamily.POLYGON,
            (
                Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]),
                Polygon(
                    [(0, 0), (6, 0), (6, 6), (0, 6), (0, 0)],
                    [[(2, 2), (4, 2), (4, 4), (2, 4), (2, 2)]],
                ),
            ),
            (None, None, None, 2, None, 10),
        ),
        (
            GeometryFamily.MULTILINESTRING,
            (
                MultiLineString([[(0, 0), (1, 1)]]),
                MultiLineString([[(0, 0), (1, 0)], [(2, 0), (3, 0), (4, 0)]]),
            ),
            (None, None, None, 2, None, 5),
        ),
        (
            GeometryFamily.MULTIPOLYGON,
            (
                MultiPolygon([Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])]),
                MultiPolygon(
                    [
                        Polygon(
                            [(0, 0), (6, 0), (6, 6), (0, 6), (0, 0)],
                            [[(2, 2), (4, 2), (4, 4), (2, 4), (2, 2)]],
                        ),
                        Polygon([(8, 0), (10, 0), (10, 2), (8, 2), (8, 0)]),
                    ]
                ),
            ),
            (None, None, None, 2, 3, 15),
        ),
    ),
)
def test_host_geometry_metadata_preserves_variable_width_capacity_bounds(
    family: GeometryFamily,
    geometries,
    expected: tuple[int | None, int | None, int | None, int | None, int | None, int | None],
) -> None:
    from vibespatial.geometry.owned import _host_fixed_geometry_size_metadata

    owned = from_shapely_geometries(geometries, residency=Residency.HOST)
    metadata = _host_fixed_geometry_size_metadata(family, owned.families[family])

    assert metadata is not None
    assert (
        metadata.first_level_count_per_row,
        metadata.second_level_count_per_row,
        metadata.coord_count_per_row,
        metadata.max_first_level_count_per_row,
        metadata.max_second_level_count_per_row,
        metadata.max_coord_count_per_row,
    ) == expected


def test_device_take_plan_keeps_variable_width_bounds_as_capacities() -> None:
    from vibespatial.geometry.owned import (
        DeviceFamilyGeometryBuffer,
        DeviceFixedGeometrySizeMetadata,
        _device_take_family_size_plan_from_device,
    )

    buffer = DeviceFamilyGeometryBuffer(
        family=GeometryFamily.POLYGON,
        x=np.empty(20, dtype=np.float64),
        y=np.empty(20, dtype=np.float64),
        geometry_offsets=np.empty(3, dtype=np.int32),
        empty_mask=np.empty(2, dtype=np.bool_),
        ring_offsets=np.empty(5, dtype=np.int32),
        fixed_size=DeviceFixedGeometrySizeMetadata(
            max_first_level_count_per_row=2,
            max_coord_count_per_row=10,
        ),
    )

    plan = _device_take_family_size_plan_from_device(
        GeometryFamily.POLYGON,
        buffer,
        np.empty(2, dtype=np.int64),
    )

    assert plan.first_level_count is None
    assert plan.coord_count is None
    assert plan.first_level_capacity == 4
    assert plan.coord_capacity == 20


@pytest.mark.gpu
def test_indexed_variable_width_multipolygon_physicalizes_from_capacity_metadata() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    geometries = [
        MultiPolygon([Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])]),
        MultiPolygon(
            [
                Polygon(
                    [(10, 0), (16, 0), (16, 6), (10, 6), (10, 0)],
                    [[(12, 2), (14, 2), (14, 4), (12, 4), (12, 2)]],
                ),
                Polygon([(18, 0), (20, 0), (20, 2), (18, 2), (18, 0)]),
            ]
        ),
    ]
    owned = from_shapely_geometries(geometries, residency=Residency.DEVICE)
    indexed = owned._device_indexed_take(cp.asarray([1, 0, 1], dtype=cp.int64))
    metadata = indexed.device_state.families[GeometryFamily.MULTIPOLYGON].fixed_size

    assert metadata is not None
    assert metadata.first_level_count_per_row is None
    assert metadata.second_level_count_per_row is None
    assert metadata.coord_count_per_row is None
    assert metadata.max_first_level_count_per_row == 2
    assert metadata.max_second_level_count_per_row == 3
    assert metadata.max_coord_count_per_row == 15

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    with assert_zero_d2h_transfers():
        physical = indexed.physicalize_device_rows(allow_capacity_allocation=True)

    assert physical.row_count == 3
    assert not physical.is_indexed_view
    actual = physical.to_shapely()
    expected = [geometries[1], geometries[0], geometries[1]]
    assert all(
        got.equals_exact(want, tolerance=1.0e-9) for got, want in zip(actual, expected, strict=True)
    )


@pytest.mark.gpu
def test_expanded_device_view_detaches_without_copying_family_buffers() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from shapely.geometry import box

    from vibespatial.cuda._runtime import assert_zero_d2h_transfers

    geometries = [
        box(0.0, 0.0, 1.0, 1.0),
        box(2.0, 0.0, 3.0, 1.0),
    ]
    owned = from_shapely_geometries(geometries, residency=Residency.DEVICE)
    source_buffer = owned.device_state.families[GeometryFamily.POLYGON]
    view = owned._device_indexed_take(cp.asarray([1, 0, 1], dtype=cp.int32))

    with assert_zero_d2h_transfers():
        detached = view.detach_expanded_device_view()

    assert detached is view
    assert not detached.is_indexed_view
    assert detached.device_state.families[GeometryFamily.POLYGON] is source_buffer
    assert cp.asnumpy(detached.device_state.family_row_offsets).tolist() == [1, 0, 1]
    actual = detached.to_shapely()
    assert actual[0].equals(geometries[1])
    assert actual[1].equals(geometries[0])
    assert actual[2].equals(geometries[1])


@pytest.mark.gpu
def test_device_size_bound_packet_recovers_variable_nested_metadata() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from shapely.geometry import box

    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )
    from vibespatial.geometry.owned import (
        device_mask_owned_capacity,
        ensure_device_geometry_size_bounds,
    )

    geometries = [
        Polygon(
            [(0, 0), (8, 0), (8, 8), (0, 8), (0, 0)],
            [[(2, 2), (6, 2), (6, 6), (2, 6), (2, 2)]],
        ),
        MultiPolygon([box(10, 0, 12, 2), box(14, 0, 16, 2)]),
    ]
    owned = from_shapely_geometries(geometries, residency=Residency.DEVICE)
    for buffer in owned.device_state.families.values():
        buffer.fixed_size = None
    owned._active_family_row_segment_capacity_bound = None
    indexed = owned._device_indexed_take(cp.asarray([1, 0, 1], dtype=cp.int64))

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    segment_bound = ensure_device_geometry_size_bounds(
        indexed,
        reason="test variable geometry size-bound packet",
    )
    events = get_d2h_transfer_events(clear=True)

    polygon = owned.device_state.families[GeometryFamily.POLYGON].fixed_size
    multipolygon = owned.device_state.families[GeometryFamily.MULTIPOLYGON].fixed_size
    assert segment_bound == 10
    assert polygon is not None
    assert polygon.max_first_level_count_per_row == 2
    assert polygon.max_coord_count_per_row == 10
    assert multipolygon is not None
    assert multipolygon.max_first_level_count_per_row == 2
    assert multipolygon.max_second_level_count_per_row == 2
    assert multipolygon.max_coord_count_per_row == 10
    assert [event.reason for event in events] == [
        "test variable geometry size-bound packet"
    ]

    masked = device_mask_owned_capacity(
        indexed,
        cp.asarray([True, False, True], dtype=cp.bool_),
    )
    physical = masked.physicalize_device_rows(allow_capacity_allocation=True)
    assert masked._active_family_row_segment_capacity_bound == 10
    assert physical._active_family_row_segment_capacity_bound == 10
    assert int(physical.device_state.families[GeometryFamily.MULTIPOLYGON].x.size) <= 30


@pytest.mark.gpu
def test_nested_indexed_physicalization_preserves_activity_and_capacity() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from shapely.geometry import box

    from vibespatial.cuda._runtime import assert_zero_d2h_transfers

    geometries = [
        box(0.0, 0.0, 1.0, 1.0),
        box(2.0, 0.0, 3.0, 1.0),
        box(4.0, 0.0, 5.0, 1.0),
    ]
    owned = from_shapely_geometries(geometries, residency=Residency.DEVICE)
    first = owned._device_indexed_take(
        cp.asarray([2, 0, 1], dtype=cp.int64),
        assume_unique_indices=True,
    )._apply_row_activity(
        cp.asarray([True, False, True], dtype=cp.bool_),
        assume_active_indices_unique=True,
    )
    nested = first._device_indexed_take(
        cp.asarray([2, 0, 1], dtype=cp.int64),
        assume_unique_indices=True,
    )

    with assert_zero_d2h_transfers():
        physical = nested.physicalize_device_rows(allow_capacity_allocation=True)

    assert not physical.is_indexed_view
    polygon = physical.device_state.families[GeometryFamily.POLYGON]
    source_polygon = owned.device_state.families[GeometryFamily.POLYGON]
    assert int(polygon.x.size) <= int(source_polygon.x.size)
    actual = physical.to_shapely()
    assert actual[0].equals(geometries[1])
    assert actual[1].equals(geometries[2])
    assert actual[2] is None


@pytest.mark.gpu
def test_capacity_physicalization_preserves_resident_rectangle_metadata() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from shapely.geometry import box

    from vibespatial.cuda._runtime import assert_zero_d2h_transfers
    from vibespatial.geometry.owned import (
        device_physicalize_owned_row_selection_capacity,
    )
    from vibespatial.kernels.constructive.polygon_rect_intersection import (
        device_trusted_rectangle_bounds_matrix,
    )

    geometries = [
        box(0.0, 0.0, 1.0, 1.0),
        box(2.0, 0.0, 3.0, 1.0),
        box(4.0, 0.0, 5.0, 1.0),
    ]
    owned = from_shapely_geometries(geometries, residency=Residency.DEVICE)

    with assert_zero_d2h_transfers():
        physical = device_physicalize_owned_row_selection_capacity(
            owned,
            cp.asarray([True, False, True], dtype=cp.bool_),
        )
        bounds = device_trusted_rectangle_bounds_matrix(physical)

    assert bounds is not None
    assert tuple(bounds.shape) == (3, 4)
    actual = physical.to_shapely()
    assert actual[0].equals(geometries[0])
    assert actual[1] is None
    assert actual[2].equals(geometries[2])


@pytest.mark.gpu
def test_device_capacity_mask_preserves_duplicate_indexed_rows() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from shapely.geometry import box

    from vibespatial.geometry.owned import device_mask_owned_capacity

    geometry = box(0.0, 0.0, 1.0, 1.0)
    owned = from_shapely_geometries(
        [geometry],
        residency=Residency.DEVICE,
    )
    indexed = owned._indexed_view(
        owned,
        cp.asarray([0, 0], dtype=cp.int64),
    )

    masked = device_mask_owned_capacity(
        indexed,
        cp.asarray([True, True], dtype=cp.bool_),
    )

    assert masked.is_indexed_view
    assert cp.asnumpy(masked._index_map).tolist() == [0, 0]
    assert all(value.equals(geometry) for value in masked.to_shapely())


@pytest.mark.gpu
def test_device_indexed_concat_interns_shared_physical_root() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    from shapely.geometry import box

    from vibespatial.geometry.owned import OwnedGeometryArray

    geometries = [
        box(0.0, 0.0, 1.0, 1.0),
        box(2.0, 0.0, 3.0, 1.0),
        box(4.0, 0.0, 5.0, 1.0),
    ]
    owned = from_shapely_geometries(geometries, residency=Residency.DEVICE)
    left = owned._device_indexed_take(
        cp.asarray([0, 1], dtype=cp.int64),
        assume_unique_indices=True,
    )
    right = owned._device_indexed_take(
        cp.asarray([1, 2], dtype=cp.int64),
        assume_unique_indices=True,
    )

    combined = OwnedGeometryArray.concat([left, right])

    assert combined.is_indexed_view
    assert combined._base is owned
    assert combined._base.row_count == owned.row_count
    actual = combined.to_shapely()
    expected = [geometries[0], geometries[1], geometries[1], geometries[2]]
    assert all(
        got.equals_exact(want, tolerance=1.0e-9) for got, want in zip(actual, expected, strict=True)
    )


@pytest.mark.gpu
def test_device_empty_polygon_rows_publish_zero_width_metadata() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    from vibespatial.geometry.owned import build_empty_polygon_rows_device

    empty = build_empty_polygon_rows_device(4)
    metadata = empty.device_state.families[GeometryFamily.POLYGON].fixed_size

    assert metadata is not None
    assert metadata.first_level_count_per_row == 0
    assert metadata.coord_count_per_row == 0
    assert metadata.max_first_level_count_per_row == 0
    assert metadata.max_coord_count_per_row == 0


def test_capacity_offset_slice_gather_uses_guarded_row_kernels() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root.joinpath("src/vibespatial/geometry/owned.py").read_text()
    gather_start = source.index("def _device_gather_offset_slices(")
    gather_end = source.index("\n@dataclass", gather_start)
    gather_source = source[gather_start:gather_end]

    capacity_branch = gather_source.split(
        "if allocation_capacity is not None and precomputed_total is None:", 1
    )[1].split("# Build flat gather indices", 1)[0]
    assert "owned_take_gather_values_i32" in capacity_branch
    assert "owned_take_gather_values_f64x2" in capacity_branch
    assert "cp.searchsorted" not in capacity_branch
    assert "owned geometry offset-slice allocation fence" not in gather_source
    assert "unknown-size offset-slice gather requires an explicit" in gather_source
    assert "reason=allocation_reason" in gather_source

    kernel_source = repo_root.joinpath("src/vibespatial/kernels/owned_take.py").read_text()
    assert "for (int j = threadIdx.x; j < length; j += blockDim.x)" in kernel_source
    assert '"owned_take_gather_values_i32"' in kernel_source
    assert '"owned_take_gather_values_f64x2"' in kernel_source

    for relative_path in (
        "src/vibespatial/io/wkt_gpu.py",
        "src/vibespatial/io/kml_gpu.py",
        "src/vibespatial/io/geojson_gpu.py",
    ):
        parser_source = repo_root.joinpath(relative_path).read_text()
        assert parser_source.count("_device_gather_offset_slices(") == (
            parser_source.count("allocation_reason=")
        )


def test_owned_geometry_has_no_raw_cupy_scalar_syncs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "src" / "vibespatial" / "geometry" / "owned.py"
    tree = ast.parse(path.read_text(), filename=str(path))

    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "item":
            offenders.append(f"{path.relative_to(repo_root)}:{node.lineno}: .item()")
            continue
        if (
            isinstance(func, ast.Name)
            and func.id in {"bool", "int", "float"}
            and node.args
            and isinstance(node.args[0], ast.Call)
            and isinstance(node.args[0].func, ast.Attribute)
            and isinstance(node.args[0].func.value, ast.Name)
            and node.args[0].func.value.id == "cp"
        ):
            offenders.append(f"{path.relative_to(repo_root)}:{node.lineno}: {func.id}(cp.*)")

    assert offenders == []


def test_capacity_partition_selection_preserves_row_indirection_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "src/vibespatial/geometry/owned.py").read_text()
    start = source.index("def device_select_owned_capacity_partitions(")
    end = source.index("\ndef ", start + 1)
    function_source = source[start:end]

    assert "OwnedGeometryArray.concat(arrays)" in function_source
    assert "OwnedGeometryArray._indexed_view" in function_source
    assert "cp.where(d_active" in function_source
    assert "cp.flatnonzero" not in function_source
    assert ".device_take(" not in function_source


def test_capacity_selection_scatter_preserves_dynamic_row_indirection() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "src/vibespatial/geometry/owned.py").read_text()
    start = source.index("def device_scatter_owned_capacity_selection(")
    end = source.index("\ndef ", start + 1)
    function_source = source[start:end]

    assert "selection.active_capacity_mask()" in function_source
    assert "selection.safe_capacity_positions()" in function_source
    assert "OwnedGeometryArray.concat([base, replacement])" in function_source
    assert "OwnedGeometryArray._indexed_view" in function_source
    assert "cp.flatnonzero" not in function_source
    assert ".device_take(" not in function_source


def test_fused_capacity_scatter_uses_exact_multi_root_physicalization() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "src/vibespatial/geometry/owned.py").read_text()
    start = source.index("def device_scatter_owned_capacity_selections_many(")
    end = source.index("\ndef ", start + 1)
    function_source = source[start:end]

    assert "device_physicalize_owned_row_selections_exact(" in function_source
    assert function_source.count("OwnedGeometryArray.concat(arrays)") == 1
    assert "selection.safe_capacity_positions()" in function_source
    assert "cp.flatnonzero" not in function_source


def test_family_capacity_take_keeps_shared_device_family_storage() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "src/vibespatial/geometry/owned.py").read_text()
    start = source.index("def device_take_owned_family_capacity_selection(")
    end = source.index("\ndef ", start + 1)
    function_source = source[start:end]

    assert "selection.active_capacity_mask()" in function_source
    assert "selection.partition_capacity_positions()" in function_source
    assert "state.trusted_unique_family_rows is True" in function_source
    assert "device_families={family: state.families[family]}" in function_source
    assert "family_row_offsets=d_family_rows" in function_source
    assert "_device_take_family_buffer(" not in function_source
    assert "physicalize_device_rows(" not in function_source
    assert "cp.flatnonzero" not in function_source
    assert "copy_device_to_host" not in function_source


def test_capacity_mask_shares_physical_buffers_without_compaction() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "src/vibespatial/geometry/owned.py").read_text()
    start = source.index("def device_mask_owned_capacity(")
    end = source.index("\ndef ", start + 1)
    function_source = source[start:end]

    assert "families=dict(state.families)" in function_source
    assert "d_validity = cp.asarray(state.validity" in function_source
    assert "preserve_indexed_view=True" in function_source
    assert "cp.flatnonzero" not in function_source
    assert ".device_take(" not in function_source
    assert "_physical_device_take(" not in function_source


@pytest.mark.gpu
def test_capacity_mask_keeps_polygon_buffers_and_nulls_inactive_rows() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")

    from shapely.geometry import box

    from vibespatial.cuda._runtime import assert_zero_d2h_transfers
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import device_mask_owned_capacity

    owned = from_shapely_geometries(
        [box(0.0, 0.0, 1.0, 1.0), box(2.0, 0.0, 3.0, 1.0)],
        residency=Residency.DEVICE,
    )
    source_state = owned._ensure_device_state()
    with assert_zero_d2h_transfers():
        masked = device_mask_owned_capacity(
            owned,
            cp.asarray([True, False]),
        )

    masked_state = masked._ensure_device_state()
    assert (
        masked_state.families[GeometryFamily.POLYGON].x
        is source_state.families[GeometryFamily.POLYGON].x
    )
    assert cp.asnumpy(masked_state.validity).tolist() == [True, False]
    assert cp.asnumpy(masked_state.tags).tolist()[1] == -1


@pytest.mark.gpu
def test_capacity_mask_preserves_indexed_logical_rows_without_physical_take() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")

    from shapely.geometry import box

    from vibespatial.cuda._runtime import assert_zero_d2h_transfers
    from vibespatial.geometry.owned import device_mask_owned_capacity

    base = from_shapely_geometries(
        [box(0.0, 0.0, 1.0, 1.0), box(2.0, 0.0, 3.0, 1.0)],
        residency=Residency.DEVICE,
    )
    indexed = base.device_take(cp.asarray([1, 0, 1], dtype=cp.int64))
    with assert_zero_d2h_transfers():
        masked = device_mask_owned_capacity(
            indexed,
            cp.asarray([True, False, True]),
        )

    assert not masked.is_indexed_view
    assert masked.row_count == 3
    state = masked._ensure_device_state(preserve_indexed_view=True)
    assert cp.asnumpy(state.validity).tolist() == [True, False, True]
    actual = masked.to_shapely()
    assert actual[0].equals(box(2.0, 0.0, 3.0, 1.0))
    assert actual[1] is None
    assert actual[2].equals(box(2.0, 0.0, 3.0, 1.0))


def _sample_geometries() -> list[object | None]:
    return [
        Point(1, 2),
        None,
        Point(),
        LineString([(0, 0), (2, 4)]),
        Polygon([(0, 0), (3, 0), (3, 3), (0, 0)]),
        MultiPolygon(
            [
                Polygon([(10, 10), (12, 10), (12, 12), (10, 10)]),
                Polygon([(20, 20), (21, 20), (21, 21), (20, 20)]),
            ]
        ),
    ]


@pytest.mark.gpu
def test_device_geom_type_single_family_no_nulls_avoids_tag_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )
    from vibespatial.geometry.device_array import DeviceGeometryArray

    owned = from_shapely_geometries(
        [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 0)]),
        ],
        residency=Residency.DEVICE,
    )
    assert owned.device_state is not None
    owned._validity = None
    owned._tags = None
    owned._family_row_offsets = None
    array = DeviceGeometryArray._from_owned(owned)

    reset_d2h_transfer_count()
    with assert_zero_d2h_transfers():
        geom_types = array.geom_type

    assert geom_types.tolist() == ["Polygon", "Polygon"]
    assert owned._tags is None


@pytest.mark.gpu
def test_area_single_family_no_nulls_avoids_metadata_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.constructive.measurement import area_owned
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [
            Polygon([(0, 0), (2, 0), (2, 2), (0, 0)]),
            Polygon([(4, 0), (6, 0), (6, 2), (4, 0)]),
        ],
        residency=Residency.DEVICE,
    )
    assert owned.device_state is not None
    owned._validity = None
    owned._tags = None
    owned._family_row_offsets = None

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    areas = area_owned(owned, dispatch_mode=ExecutionMode.GPU, precision="fp64")
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert np.allclose(areas, [2.0, 2.0])
    assert not any("owned geometry host metadata" in reason for reason in reasons)
    assert owned._validity is None
    assert owned._tags is None
    assert owned._family_row_offsets is None


@pytest.mark.gpu
def test_device_take_with_host_indices_reuses_existing_host_metadata() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.cuda._runtime import (
        assert_zero_d2h_transfers,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [Point(0, 0), None, Point(2, 2)],
        residency=Residency.DEVICE,
    )
    reset_d2h_transfer_count()

    taken = owned.take(np.asarray([2, 0], dtype=np.int64))

    with assert_zero_d2h_transfers():
        assert taken.validity.tolist() == [True, True]
        assert taken.tags.tolist() == owned.tags[[2, 0]].tolist()
        assert taken.family_row_offsets.tolist() == [0, 1]

    actual = taken.to_shapely()
    assert actual[0].equals(Point(2, 2))
    assert actual[1].equals(Point(0, 0))


@pytest.mark.gpu
def test_take_full_host_index_vector_preserves_device_owned_identity() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.cuda._runtime import assert_zero_d2h_transfers

    owned = from_shapely_geometries(
        [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 0)]),
        ],
        residency=Residency.DEVICE,
    )

    with assert_zero_d2h_transfers():
        taken = owned.take(np.arange(owned.row_count, dtype=np.int64))

    assert taken is owned


@pytest.mark.gpu
def test_to_shapely_materializes_inconsistent_host_stub_from_device_state() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema
    from vibespatial.geometry.owned import FamilyGeometryBuffer

    expected = Polygon([(0, 0), (2, 0), (2, 2), (0, 0)])
    owned = from_shapely_geometries([expected], residency=Residency.DEVICE)
    original = owned.families[GeometryFamily.POLYGON]
    owned.families[GeometryFamily.POLYGON] = FamilyGeometryBuffer(
        family=GeometryFamily.POLYGON,
        schema=get_geometry_buffer_schema(GeometryFamily.POLYGON),
        row_count=original.row_count,
        x=np.empty(0, dtype=np.float64),
        y=np.empty(0, dtype=np.float64),
        geometry_offsets=original.geometry_offsets.copy(),
        empty_mask=original.empty_mask.copy(),
        ring_offsets=original.ring_offsets.copy(),
        host_materialized=True,
    )

    restored = owned.to_shapely()

    assert restored[0].equals(expected)


@pytest.mark.gpu
def test_concatenate_owned_arrays_preserves_nonpoint_device_state() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.geometry.owned import concatenate_owned_arrays

    expected = [
        Polygon([(0, 0), (2, 0), (2, 2), (0, 0)]),
        Polygon([(10, 0), (12, 0), (12, 2), (10, 0)]),
    ]
    base = from_shapely_geometries(expected, residency=Residency.DEVICE)
    left = base.device_take(cp.asarray([0], dtype=cp.int64))
    right = base.device_take(cp.asarray([1], dtype=cp.int64))

    combined = concatenate_owned_arrays([left, right])

    assert combined.residency is Residency.DEVICE
    assert combined.device_state is not None
    assert [geom.equals(exp) for geom, exp in zip(combined.to_shapely(), expected)] == [
        True,
        True,
    ]


def test_unique_tag_pairs_numpy_uses_geometry_tag_pairs() -> None:
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import FAMILY_TAGS, unique_tag_pairs

    left = np.asarray(
        [
            FAMILY_TAGS[GeometryFamily.POLYGON],
            FAMILY_TAGS[GeometryFamily.POLYGON],
            FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
        ],
        dtype=np.int8,
    )
    right = np.asarray(
        [
            FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
            FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
            FAMILY_TAGS[GeometryFamily.POLYGON],
        ],
        dtype=np.int8,
    )

    assert unique_tag_pairs(left, right) == [
        (FAMILY_TAGS[GeometryFamily.POLYGON], FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]),
        (FAMILY_TAGS[GeometryFamily.MULTIPOLYGON], FAMILY_TAGS[GeometryFamily.POLYGON]),
    ]


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU not available")
def test_unique_tag_pairs_cupy_avoids_heavy_unique(monkeypatch: pytest.MonkeyPatch) -> None:
    import cupy as cp

    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import FAMILY_TAGS, unique_tag_pairs

    def _fail_unique(*_args, **_kwargs):
        raise AssertionError("fixed-domain tag pairs should not dispatch cp.unique")

    monkeypatch.setattr(cp, "unique", _fail_unique)
    left = cp.asarray(
        [
            FAMILY_TAGS[GeometryFamily.POLYGON],
            FAMILY_TAGS[GeometryFamily.POLYGON],
            FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
        ],
        dtype=cp.int8,
    )
    right = cp.asarray(
        [
            FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
            FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
            FAMILY_TAGS[GeometryFamily.POLYGON],
        ],
        dtype=cp.int8,
    )

    assert unique_tag_pairs(left, right) == [
        (FAMILY_TAGS[GeometryFamily.POLYGON], FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]),
        (FAMILY_TAGS[GeometryFamily.MULTIPOLYGON], FAMILY_TAGS[GeometryFamily.POLYGON]),
    ]


def test_shapely_round_trip_preserves_null_and_empty() -> None:
    owned = from_shapely_geometries(_sample_geometries())
    restored = owned.to_shapely()

    assert restored[0].equals(Point(1, 2))
    assert restored[1] is None
    assert restored[2].is_empty
    assert restored[3].equals(LineString([(0, 0), (2, 4)]))


def test_owned_to_shapely_does_not_route_through_wkb_bridge(monkeypatch) -> None:
    from vibespatial.io import wkb as wkb_module

    def _fail_encode(*args, **kwargs):
        raise AssertionError("owned->Shapely materialization should not use WKB bridge")

    monkeypatch.setattr(wkb_module, "encode_wkb_owned", _fail_encode)

    owned = from_shapely_geometries(_sample_geometries())
    restored = owned.to_shapely()

    assert restored[0].equals(Point(1, 2))
    assert restored[4].equals(Polygon([(0, 0), (3, 0), (3, 3), (0, 0)]))


def test_owned_to_shapely_ignores_polygon_coordinate_capacity_tail() -> None:
    from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema
    from vibespatial.geometry.owned import FAMILY_TAGS, FamilyGeometryBuffer, OwnedGeometryArray

    buffer = FamilyGeometryBuffer(
        family=GeometryFamily.POLYGON,
        schema=get_geometry_buffer_schema(GeometryFamily.POLYGON),
        row_count=1,
        x=np.asarray([0.0, 2.0, 2.0, 0.0, 0.0, 999.0, 999.0], dtype=np.float64),
        y=np.asarray([0.0, 0.0, 2.0, 2.0, 0.0, 999.0, 999.0], dtype=np.float64),
        geometry_offsets=np.asarray([0, 1], dtype=np.int32),
        empty_mask=np.asarray([False], dtype=bool),
        ring_offsets=np.asarray([0, 5], dtype=np.int32),
    )
    owned = OwnedGeometryArray(
        validity=np.asarray([True], dtype=bool),
        tags=np.asarray([FAMILY_TAGS[GeometryFamily.POLYGON]], dtype=np.int8),
        family_row_offsets=np.asarray([0], dtype=np.int32),
        families={GeometryFamily.POLYGON: buffer},
        residency=Residency.HOST,
    )

    assert owned.to_shapely()[0].equals(Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]))


def test_owned_to_shapely_ignores_multipolygon_coordinate_capacity_tail() -> None:
    from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema
    from vibespatial.geometry.owned import FAMILY_TAGS, FamilyGeometryBuffer, OwnedGeometryArray

    buffer = FamilyGeometryBuffer(
        family=GeometryFamily.MULTIPOLYGON,
        schema=get_geometry_buffer_schema(GeometryFamily.MULTIPOLYGON),
        row_count=1,
        x=np.asarray([0.0, 2.0, 2.0, 0.0, 0.0, 999.0], dtype=np.float64),
        y=np.asarray([0.0, 0.0, 2.0, 2.0, 0.0, 999.0], dtype=np.float64),
        geometry_offsets=np.asarray([0, 1], dtype=np.int32),
        empty_mask=np.asarray([False], dtype=bool),
        part_offsets=np.asarray([0, 1], dtype=np.int32),
        ring_offsets=np.asarray([0, 5], dtype=np.int32),
    )
    owned = OwnedGeometryArray(
        validity=np.asarray([True], dtype=bool),
        tags=np.asarray([FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]], dtype=np.int8),
        family_row_offsets=np.asarray([0], dtype=np.int32),
        families={GeometryFamily.MULTIPOLYGON: buffer},
        residency=Residency.HOST,
    )

    expected = MultiPolygon([Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])])
    assert owned.to_shapely()[0].equals(expected)


def _overallocated_polygon_owned(x_offset: float):
    from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema
    from vibespatial.geometry.owned import FAMILY_TAGS, FamilyGeometryBuffer, OwnedGeometryArray

    buffer = FamilyGeometryBuffer(
        family=GeometryFamily.POLYGON,
        schema=get_geometry_buffer_schema(GeometryFamily.POLYGON),
        row_count=1,
        x=np.asarray(
            [x_offset, x_offset + 1, x_offset + 1, x_offset, x_offset, 999.0],
            dtype=np.float64,
        ),
        y=np.asarray([0.0, 0.0, 1.0, 1.0, 0.0, 999.0], dtype=np.float64),
        geometry_offsets=np.asarray([0, 1], dtype=np.int32),
        empty_mask=np.asarray([False], dtype=bool),
        ring_offsets=np.asarray([0, 5], dtype=np.int32),
    )
    return OwnedGeometryArray(
        validity=np.asarray([True], dtype=bool),
        tags=np.asarray([FAMILY_TAGS[GeometryFamily.POLYGON]], dtype=np.int8),
        family_row_offsets=np.asarray([0], dtype=np.int32),
        families={GeometryFamily.POLYGON: buffer},
        residency=Residency.HOST,
    )


def _overallocated_polygon_ring_capacity_owned(x_offset: float):
    from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema
    from vibespatial.geometry.owned import FAMILY_TAGS, FamilyGeometryBuffer, OwnedGeometryArray

    buffer = FamilyGeometryBuffer(
        family=GeometryFamily.POLYGON,
        schema=get_geometry_buffer_schema(GeometryFamily.POLYGON),
        row_count=1,
        x=np.asarray(
            [x_offset, x_offset + 1, x_offset + 1, x_offset, x_offset, 999.0, 999.0, 999.0],
            dtype=np.float64,
        ),
        y=np.asarray([0.0, 0.0, 1.0, 1.0, 0.0, 999.0, 999.0, 999.0], dtype=np.float64),
        geometry_offsets=np.asarray([0, 1], dtype=np.int32),
        empty_mask=np.asarray([False], dtype=bool),
        ring_offsets=np.asarray([0, 5, 5], dtype=np.int32),
    )
    return OwnedGeometryArray(
        validity=np.asarray([True], dtype=bool),
        tags=np.asarray([FAMILY_TAGS[GeometryFamily.POLYGON]], dtype=np.int8),
        family_row_offsets=np.asarray([0], dtype=np.int32),
        families={GeometryFamily.POLYGON: buffer},
        residency=Residency.HOST,
    )


def test_owned_concat_ignores_polygon_coordinate_capacity_tail() -> None:
    from vibespatial.geometry.owned import OwnedGeometryArray

    owned = OwnedGeometryArray.concat(
        [
            _overallocated_polygon_owned(0.0),
            _overallocated_polygon_owned(10.0),
        ]
    )

    restored = owned.to_shapely()
    assert restored[0].equals(Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]))
    assert restored[1].equals(Polygon([(10, 0), (11, 0), (11, 1), (10, 1), (10, 0)]))


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU not available")
def test_device_host_materialization_copies_only_logical_nested_prefixes() -> None:
    from vibespatial.geometry.owned import FamilyGeometryBuffer

    owned = _overallocated_polygon_ring_capacity_owned(0.0).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test logical-prefix device host materialization",
    )
    host_buffer = owned.families[GeometryFamily.POLYGON]
    owned._validity = None
    owned._tags = None
    owned._family_row_offsets = None
    owned.families[GeometryFamily.POLYGON] = FamilyGeometryBuffer(
        family=host_buffer.family,
        schema=host_buffer.schema,
        row_count=host_buffer.row_count,
        x=np.empty(0, dtype=np.float64),
        y=np.empty(0, dtype=np.float64),
        geometry_offsets=np.empty(0, dtype=np.int32),
        empty_mask=np.empty(0, dtype=bool),
        host_materialized=False,
    )

    restored = owned.to_shapely()

    assert restored[0].equals(Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]))
    host_buffer = owned.families[GeometryFamily.POLYGON]
    assert host_buffer.x.size == 5
    assert host_buffer.y.size == 5
    assert host_buffer.ring_offsets.tolist() == [0, 5]


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU not available")
def test_device_owned_concat_ignores_polygon_coordinate_capacity_tail() -> None:
    from vibespatial.cuda._runtime import assert_zero_d2h_transfers
    from vibespatial.geometry.owned import OwnedGeometryArray

    left = _overallocated_polygon_owned(0.0).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test device concat capacity tail",
    )
    right = _overallocated_polygon_owned(10.0).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test device concat capacity tail",
    )

    with assert_zero_d2h_transfers():
        owned = OwnedGeometryArray.concat([left, right])

    restored = owned.to_shapely()
    assert restored[0].equals(Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]))
    assert restored[1].equals(Polygon([(10, 0), (11, 0), (11, 1), (10, 1), (10, 0)]))


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU not available")
def test_device_owned_concat_initializes_inactive_ring_offset_capacity() -> None:
    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import assert_zero_d2h_transfers
    from vibespatial.geometry.owned import OwnedGeometryArray

    left = _overallocated_polygon_ring_capacity_owned(0.0).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test device concat ring capacity tail",
    )
    right = _overallocated_polygon_ring_capacity_owned(10.0).move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="test device concat ring capacity tail",
    )

    with assert_zero_d2h_transfers():
        owned = OwnedGeometryArray.concat([left, right])

    d_polygon = owned.device_state.families[GeometryFamily.POLYGON]
    np.testing.assert_array_equal(
        cp.asnumpy(d_polygon.ring_offsets),
        np.asarray([0, 5, 10, 10, 10], dtype=np.int32),
    )
    restored = owned.to_shapely()
    assert restored[0].equals(Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]))
    assert restored[1].equals(Polygon([(10, 0), (11, 0), (11, 1), (10, 1), (10, 0)]))


def test_wkb_round_trip_matches_shapely_path() -> None:
    baseline = from_shapely_geometries(_sample_geometries())
    wkb = baseline.to_wkb()
    restored = from_wkb(wkb)

    restored_shapes = restored.to_shapely()
    baseline_shapes = baseline.to_shapely()
    for left, right in zip(restored_shapes, baseline_shapes, strict=True):
        if left is None or right is None:
            assert left is right
            continue
        assert left.equals(right)


def test_geoarrow_style_round_trip_preserves_family_buffers() -> None:
    baseline = from_shapely_geometries(_sample_geometries())
    restored = from_geoarrow(baseline.to_geoarrow())

    assert restored.row_count == baseline.row_count
    assert np.array_equal(restored.validity, baseline.validity)
    assert np.array_equal(restored.tags, baseline.tags)
    assert np.array_equal(restored.family_row_offsets, baseline.family_row_offsets)


def test_geoarrow_share_mode_reuses_buffers_and_delays_host_geometry_materialization() -> None:
    owned = from_shapely_geometries(_sample_geometries())
    shared_view = owned.to_geoarrow(sharing=BufferSharingMode.SHARE)
    adopted = from_geoarrow(shared_view, sharing=BufferSharingMode.AUTO)

    point_family = next(iter(adopted.families))
    assert np.shares_memory(shared_view.validity, adopted.validity)
    assert np.shares_memory(shared_view.families[point_family].x, adopted.families[point_family].x)
    assert adopted.geoarrow_backed is True
    assert adopted.shares_geoarrow_memory is True
    assert not any(event.kind is DiagnosticKind.MATERIALIZATION for event in adopted.diagnostics)

    adopted.to_shapely()

    assert any(event.kind is DiagnosticKind.MATERIALIZATION for event in adopted.diagnostics)


def test_geoarrow_share_mode_reuses_cached_view_object() -> None:
    owned = from_shapely_geometries(_sample_geometries())

    first = owned.to_geoarrow(sharing=BufferSharingMode.SHARE)
    second = owned.to_geoarrow(sharing=BufferSharingMode.SHARE)

    assert first is second


def test_geoarrow_share_mode_reuses_cached_family_wrappers_on_import() -> None:
    owned = from_shapely_geometries(_sample_geometries())
    shared_view = owned.to_geoarrow(sharing=BufferSharingMode.SHARE)

    first = from_geoarrow(shared_view, sharing=BufferSharingMode.AUTO)
    second = from_geoarrow(shared_view, sharing=BufferSharingMode.AUTO)

    point_family = next(iter(first.families))
    assert first is not second
    assert first.families is not second.families
    assert first.families[point_family] is second.families[point_family]
    assert np.shares_memory(first.validity, second.validity)


def test_bounds_and_total_bounds_ignore_nulls_and_empty() -> None:
    owned = from_shapely_geometries(_sample_geometries())

    bounds = compute_geometry_bounds(owned)
    total = compute_total_bounds(owned)

    assert np.allclose(bounds[0], np.asarray([1.0, 2.0, 1.0, 2.0]))
    assert np.isnan(bounds[1]).all()
    assert np.isnan(bounds[2]).all()
    assert total == (0.0, 0.0, 21.0, 21.0)


def test_cached_bounds_follow_indexed_view_family_rows() -> None:
    owned = from_shapely_geometries(
        [
            Point(1, 2),
            LineString([(10, 10), (12, 14)]),
            Polygon([(20, 20), (23, 20), (23, 24), (20, 20)]),
        ]
    )
    compute_geometry_bounds(owned, dispatch_mode=ExecutionMode.CPU)
    view = type(owned)._indexed_view(
        owned,
        np.asarray([2, 0, 2, 1], dtype=np.int64),
    )

    bounds = compute_geometry_bounds(view, dispatch_mode=ExecutionMode.CPU)

    np.testing.assert_array_equal(
        bounds,
        np.asarray(
            [
                [20.0, 20.0, 23.0, 24.0],
                [1.0, 2.0, 1.0, 2.0],
                [20.0, 20.0, 23.0, 24.0],
                [10.0, 10.0, 12.0, 14.0],
            ]
        ),
    )


def test_offset_spans_expose_payload_hierarchy() -> None:
    owned = from_shapely_geometries(_sample_geometries())

    geometry_spans = compute_offset_spans(owned, level="geometry")
    part_spans = compute_offset_spans(owned, level="part")

    assert geometry_spans
    assert any(span.size > 0 for span in geometry_spans.values())
    assert any(span.size > 0 for span in part_spans.values())


def test_morton_keys_place_null_and_empty_at_end() -> None:
    owned = from_shapely_geometries(_sample_geometries())

    keys = compute_morton_keys(owned)

    assert keys.shape == (owned.row_count,)
    assert keys[1] == np.iinfo(np.uint64).max
    assert keys[2] == np.iinfo(np.uint64).max


@pytest.mark.gpu
def test_diagnostics_capture_residency_and_runtime_changes() -> None:
    owned = from_shapely_geometries(_sample_geometries())
    owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="explicit gpu execution requested",
    )
    owned.record_runtime_selection(
        RuntimeSelection(
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.CPU,
            reason="GPU runtime unavailable; using explicit CPU fallback",
        )
    )
    owned.to_shapely()
    report = owned.diagnostics_report()

    assert report["residency"] == Residency.DEVICE.value
    assert any(event["kind"] == DiagnosticKind.TRANSFER.value for event in report["events"])
    assert any(event["kind"] == DiagnosticKind.MATERIALIZATION.value for event in report["events"])
    assert any("CPU fallback" in reason for reason in report["runtime_history"])


def test_fallback_diagnostic_kind_is_reportable() -> None:
    owned = from_shapely_geometries(_sample_geometries())
    owned._record(
        DiagnosticKind.FALLBACK,
        "GPU operation declined before compatibility fallback",
        visible=True,
    )

    report = owned.diagnostics_report()

    assert any(event["kind"] == "fallback" for event in report["events"])


def test_device_take_preserves_cached_row_bounds() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device row-bounds propagation")
    cp = pytest.importorskip("cupy")

    from vibespatial.cuda._runtime import get_cuda_runtime
    from vibespatial.geometry.owned import device_concat_owned_scatter
    from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device

    geoms = [
        Polygon([(0, 0), (2, 0), (2, 1), (1, 2), (0, 1), (0, 0)]),
        Polygon([(10, 10), (13, 10), (13, 11), (12, 12), (10, 11), (10, 10)]),
        Polygon([(20, 20), (21, 20), (22, 22), (20, 21), (20, 20)]),
    ]
    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    compute_geometry_bounds_device(owned)

    view = owned.device_take(cp.asarray([2, 0, 1], dtype=cp.int64))

    assert view.device_state is not None
    assert view.device_state.row_bounds is not None

    bounds = (
        get_cuda_runtime()
        .copy_device_to_host(
            compute_geometry_bounds_device(view, preserve_indexed_view=True),
            reason="test device-take row-bounds terminal export",
            terminal_export=True,
        )
        .reshape(-1, 4)
    )

    np.testing.assert_array_equal(
        bounds,
        np.asarray(
            [
                [20.0, 20.0, 22.0, 22.0],
                [0.0, 0.0, 2.0, 2.0],
                [10.0, 10.0, 13.0, 12.0],
            ],
            dtype=np.float64,
        ),
    )

    replacement = from_shapely_geometries(
        [Polygon([(100, 100), (101, 100), (101, 101), (100, 100)])],
        residency=Residency.DEVICE,
    )
    compute_geometry_bounds_device(replacement)
    scattered = device_concat_owned_scatter(
        owned,
        replacement,
        cp.asarray([1], dtype=cp.int64),
    )

    assert scattered.device_state is not None
    assert scattered.device_state.row_bounds is not None

    scattered_bounds = (
        get_cuda_runtime()
        .copy_device_to_host(
            compute_geometry_bounds_device(scattered, preserve_indexed_view=True),
            reason="test device-scatter row-bounds terminal export",
            terminal_export=True,
        )
        .reshape(-1, 4)
    )
    np.testing.assert_array_equal(
        scattered_bounds,
        np.asarray(
            [
                [0.0, 0.0, 2.0, 2.0],
                [100.0, 100.0, 101.0, 101.0],
                [20.0, 20.0, 22.0, 22.0],
            ],
            dtype=np.float64,
        ),
    )


def test_move_to_device_allocates_device_mirrors_when_gpu_is_available() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    owned = from_shapely_geometries(_sample_geometries())
    assert owned.device_state is None

    owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="explicit gpu execution requested",
    )

    report = owned.diagnostics_report()
    assert owned.device_state is not None
    assert report["device_buffers_allocated"] is True
    assert report["residency"] == Residency.DEVICE.value


def test_take_by_indices_preserves_geometries() -> None:
    owned = from_shapely_geometries(_sample_geometries())
    full_shapely = owned.to_shapely()

    subset = owned.take(np.array([0, 3, 4, 5]))
    subset_shapely = subset.to_shapely()

    assert subset.row_count == 4
    assert subset_shapely[0].equals(full_shapely[0])  # Point
    assert subset_shapely[1].equals(full_shapely[3])  # LineString
    assert subset_shapely[2].equals(full_shapely[4])  # Polygon
    assert subset_shapely[3].equals(full_shapely[5])  # MultiPolygon


def test_take_by_boolean_mask_preserves_geometries() -> None:
    owned = from_shapely_geometries(_sample_geometries())
    full_shapely = owned.to_shapely()

    mask = np.array([True, False, False, False, True, True])
    subset = owned.take(mask)
    subset_shapely = subset.to_shapely()

    assert subset.row_count == 3
    assert subset_shapely[0].equals(full_shapely[0])  # Point
    assert subset_shapely[1].equals(full_shapely[4])  # Polygon
    assert subset_shapely[2].equals(full_shapely[5])  # MultiPolygon


def test_take_preserves_null_and_empty() -> None:
    owned = from_shapely_geometries(_sample_geometries())

    subset = owned.take(np.array([1, 2]))
    subset_shapely = subset.to_shapely()

    assert subset.row_count == 2
    assert subset_shapely[0] is None  # null
    assert subset_shapely[1].is_empty  # empty Point


def test_take_single_row() -> None:
    owned = from_shapely_geometries(_sample_geometries())
    full_shapely = owned.to_shapely()

    for i in range(len(full_shapely)):
        subset = owned.take(np.array([i]))
        result = subset.to_shapely()
        assert subset.row_count == 1
        if full_shapely[i] is None:
            assert result[0] is None
        else:
            assert result[0].equals(full_shapely[i])


def test_take_polygon_with_holes() -> None:
    poly_with_hole = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
        [[(2, 2), (4, 2), (4, 4), (2, 2)]],
    )
    geoms = [Point(1, 1), poly_with_hole, Point(5, 5)]
    owned = from_shapely_geometries(geoms)
    subset = owned.take(np.array([1]))
    result = subset.to_shapely()

    assert subset.row_count == 1
    assert result[0].equals(poly_with_hole)
    assert len(list(result[0].interiors)) == 1


def test_take_all_families() -> None:
    geoms = [
        Point(1, 2),
        LineString([(0, 0), (1, 1), (2, 0)]),
        Polygon([(0, 0), (3, 0), (3, 3), (0, 0)]),
        MultiPoint([(0, 0), (1, 1)]),
        MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]]),
        MultiPolygon(
            [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
                Polygon([(2, 2), (3, 2), (3, 3), (2, 2)]),
            ]
        ),
    ]
    owned = from_shapely_geometries(geoms)
    for i, geom in enumerate(geoms):
        subset = owned.take(np.array([i]))
        result = subset.to_shapely()
        assert result[0].equals(geom), f"family {geom.geom_type} at index {i} failed round-trip"


def test_take_empty_indices() -> None:
    owned = from_shapely_geometries(_sample_geometries())

    subset = owned.take(np.array([], dtype=np.int64))
    assert subset.row_count == 0
    assert subset.to_shapely() == []


@pytest.mark.gpu
def test_compute_geometry_bounds_gpu_matches_cpu_reference() -> None:
    owned = from_shapely_geometries(_sample_geometries())

    cpu_bounds = compute_geometry_bounds(owned, dispatch_mode=ExecutionMode.CPU)
    gpu_bounds = compute_geometry_bounds(owned, dispatch_mode=ExecutionMode.GPU)

    assert np.allclose(cpu_bounds, gpu_bounds, equal_nan=True)


# ---------------------------------------------------------------------------
# Device-resident concat (lyy.29)
# ---------------------------------------------------------------------------


def _make_device_resident(geoms: list[object | None]) -> OwnedGeometryArray:
    """Create a device-resident OwnedGeometryArray with host stubs cleared."""
    from vibespatial.geometry.owned import FamilyGeometryBuffer

    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    # Clear host family buffers to simulate true device-only arrays
    owned.families = {
        family: FamilyGeometryBuffer(
            family=buffer.family,
            schema=buffer.schema,
            row_count=buffer.row_count,
            x=np.empty(0, dtype=np.float64),
            y=np.empty(0, dtype=np.float64),
            geometry_offsets=np.empty(0, dtype=np.int32),
            empty_mask=np.empty(0, dtype=np.bool_),
            host_materialized=False,
        )
        for family, buffer in owned.families.items()
    }
    return owned


@pytest.mark.skipif(not has_gpu_runtime(), reason="CUDA runtime not available")
class TestDeviceResidentConcat:
    """Verify that OwnedGeometryArray.concat() stays device-resident when all
    inputs are device-resident, with no D->H transfer."""

    def test_concat_points_stays_device_resident(self) -> None:
        """Concatenating device-resident point arrays produces a device-resident result."""
        from vibespatial.geometry.owned import OwnedGeometryArray

        owned1 = _make_device_resident([Point(0, 0), Point(1, 1)])
        owned2 = _make_device_resident([Point(2, 2), Point(3, 3)])

        result = OwnedGeometryArray.concat([owned1, owned2])

        assert result.residency is Residency.DEVICE
        assert result.device_state is not None
        assert result.row_count == 4
        # Verify host metadata is not materialized (stays None)
        assert result._validity is None
        assert result._tags is None
        assert result._family_row_offsets is None

        # Verify round-trip correctness after host materialization
        shapely_result = result.to_shapely()
        assert shapely_result[0].equals(Point(0, 0))
        assert shapely_result[1].equals(Point(1, 1))
        assert shapely_result[2].equals(Point(2, 2))
        assert shapely_result[3].equals(Point(3, 3))

    def test_concat_polygons_stays_device_resident(self) -> None:
        """Concatenating device-resident polygon arrays preserves ring offsets."""
        from vibespatial.geometry.owned import OwnedGeometryArray

        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        p2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)])

        owned1 = _make_device_resident([p1])
        owned2 = _make_device_resident([p2])

        result = OwnedGeometryArray.concat([owned1, owned2])

        assert result.residency is Residency.DEVICE
        assert result.row_count == 2

        shapely_result = result.to_shapely()
        assert shapely_result[0].equals(p1)
        assert shapely_result[1].equals(p2)

    def test_concat_multipolygons_stays_device_resident(self) -> None:
        """Concatenating device-resident multipolygon arrays preserves 3-level offsets."""
        from vibespatial.geometry.owned import OwnedGeometryArray

        mp1 = MultiPolygon(
            [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
                Polygon([(2, 2), (3, 2), (3, 3), (2, 2)]),
            ]
        )
        mp2 = MultiPolygon(
            [
                Polygon([(10, 10), (11, 10), (11, 11), (10, 10)]),
            ]
        )

        owned1 = _make_device_resident([mp1])
        owned2 = _make_device_resident([mp2])

        result = OwnedGeometryArray.concat([owned1, owned2])

        assert result.residency is Residency.DEVICE
        assert result.row_count == 2

        shapely_result = result.to_shapely()
        assert shapely_result[0].equals(mp1)
        assert shapely_result[1].equals(mp2)

    def test_concat_linestrings_stays_device_resident(self) -> None:
        """Concatenating device-resident linestring arrays."""
        from vibespatial.geometry.owned import OwnedGeometryArray

        ls1 = LineString([(0, 0), (1, 1), (2, 0)])
        ls2 = LineString([(5, 5), (6, 6)])

        owned1 = _make_device_resident([ls1])
        owned2 = _make_device_resident([ls2])

        result = OwnedGeometryArray.concat([owned1, owned2])

        assert result.residency is Residency.DEVICE
        assert result.row_count == 2

        shapely_result = result.to_shapely()
        assert shapely_result[0].equals(ls1)
        assert shapely_result[1].equals(ls2)

    def test_concat_multilinestrings_stays_device_resident(self) -> None:
        """Concatenating device-resident multilinestring arrays preserves part offsets."""
        from vibespatial.geometry.owned import OwnedGeometryArray

        mls1 = MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]])
        mls2 = MultiLineString([[(10, 10), (11, 11)]])

        owned1 = _make_device_resident([mls1])
        owned2 = _make_device_resident([mls2])

        result = OwnedGeometryArray.concat([owned1, owned2])

        assert result.residency is Residency.DEVICE
        assert result.row_count == 2

        shapely_result = result.to_shapely()
        assert shapely_result[0].equals(mls1)
        assert shapely_result[1].equals(mls2)

    def test_concat_with_nulls_stays_device_resident(self) -> None:
        """Null rows are correctly handled in device-resident concat."""
        from vibespatial.geometry.owned import OwnedGeometryArray

        owned1 = _make_device_resident([Point(0, 0), None])
        owned2 = _make_device_resident([None, Point(1, 1)])

        result = OwnedGeometryArray.concat([owned1, owned2])

        assert result.residency is Residency.DEVICE
        assert result.row_count == 4
        assert result.device_state.trusted_family_domain == (GeometryFamily.POINT,)

        shapely_result = result.to_shapely()
        assert shapely_result[0].equals(Point(0, 0))
        assert shapely_result[1] is None
        assert shapely_result[2] is None
        assert shapely_result[3].equals(Point(1, 1))

    def test_concat_mixed_families_stays_device_resident(self) -> None:
        """Concatenating arrays with different geometry families."""
        from vibespatial.geometry.owned import OwnedGeometryArray

        owned1 = _make_device_resident(
            [
                Point(0, 0),
                Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
            ]
        )
        owned2 = _make_device_resident(
            [
                LineString([(2, 2), (3, 3)]),
                Point(4, 4),
            ]
        )

        result = OwnedGeometryArray.concat([owned1, owned2])

        assert result.residency is Residency.DEVICE
        assert result.row_count == 4

        shapely_result = result.to_shapely()
        assert shapely_result[0].equals(Point(0, 0))
        assert shapely_result[1].equals(Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]))
        assert shapely_result[2].equals(LineString([(2, 2), (3, 3)]))
        assert shapely_result[3].equals(Point(4, 4))

    def test_concat_no_d2h_transfer(self, monkeypatch) -> None:
        """Device-resident concat must not call _ensure_host_state."""
        from vibespatial.geometry.owned import OwnedGeometryArray

        owned1 = _make_device_resident([Point(0, 0), Point(1, 1)])
        owned2 = _make_device_resident([Point(2, 2)])

        calls = []

        def _spy_host_state(self_inner):
            calls.append("_ensure_host_state")

        monkeypatch.setattr(
            OwnedGeometryArray,
            "_ensure_host_state",
            _spy_host_state,
        )

        result = OwnedGeometryArray.concat([owned1, owned2])

        assert result.residency is Residency.DEVICE
        assert len(calls) == 0, "_ensure_host_state was called during device-resident concat"

    def test_device_geometry_copy_no_metadata_d2h(self) -> None:
        """Copying a device-only geometry column must not materialize host metadata."""
        from vibespatial.cuda._runtime import assert_zero_d2h_transfers
        from vibespatial.geometry.device_array import DeviceGeometryArray

        owned = _make_device_resident(
            [
                Point(0, 0),
                Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
            ]
        )
        owned._validity = None
        owned._tags = None
        owned._family_row_offsets = None

        with assert_zero_d2h_transfers():
            copied = DeviceGeometryArray._from_owned(owned).copy()

        assert copied._owned.residency is Residency.DEVICE
        assert copied._owned.device_state.trusted_family_domain == (
            GeometryFamily.POINT,
            GeometryFamily.POLYGON,
        )
        assert copied._owned.row_count == 2
        assert copied._owned._validity is None
        assert copied._owned._tags is None
        assert copied._owned._family_row_offsets is None

    def test_device_geometry_nbytes_no_metadata_d2h(self) -> None:
        """Byte accounting for device arrays should not materialize host metadata."""
        from vibespatial.cuda._runtime import (
            get_d2h_transfer_events,
            reset_d2h_transfer_count,
        )
        from vibespatial.geometry.device_array import DeviceGeometryArray

        owned = _make_device_resident(
            [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 0)]),
            ]
        )
        owned._validity = None
        owned._tags = None
        owned._family_row_offsets = None

        reset_d2h_transfer_count()
        get_d2h_transfer_events(clear=True)
        byte_count = DeviceGeometryArray._from_owned(owned).nbytes
        reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

        assert byte_count > 0
        assert not any("owned geometry host metadata" in reason for reason in reasons)
        assert owned._validity is None
        assert owned._tags is None
        assert owned._family_row_offsets is None

    def test_empty_deferred_indexed_view_exposes_array_metadata(self) -> None:
        """A zero-row relation view must resolve metadata from its device base."""
        cp = pytest.importorskip("cupy")

        from vibespatial.geometry.device_array import DeviceGeometryArray
        from vibespatial.geometry.owned import OwnedGeometryArray

        base = _make_device_resident([Point(0.0, 0.0)])
        view = OwnedGeometryArray._indexed_view(
            base,
            cp.empty(0, dtype=cp.int64),
            assume_unique_indices=True,
            expand_device_metadata=False,
        )
        array = DeviceGeometryArray._from_owned(view)

        assert view.device_state is None
        assert array.nbytes >= 0
        assert array.is_empty.shape == (0,)
        assert array.isna().shape == (0,)
        assert view.device_state is not None

    def test_provenance_infer_geom_types_uses_device_family_domain_no_d2h(self) -> None:
        """Provenance type summaries should use resident device family metadata."""
        from vibespatial.cuda._runtime import (
            get_d2h_transfer_events,
            reset_d2h_transfer_count,
        )
        from vibespatial.geometry.device_array import DeviceGeometryArray
        from vibespatial.runtime.provenance import infer_geom_types

        owned = _make_device_resident(
            [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 0)]),
            ]
        )
        owned._validity = None
        owned._tags = None
        owned._family_row_offsets = None

        reset_d2h_transfer_count()
        get_d2h_transfer_events(clear=True)
        geom_types = infer_geom_types(DeviceGeometryArray._from_owned(owned))
        reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

        assert geom_types == {"polygon"}
        assert "provenance geometry tag-domain summary scalar fence" not in reasons
        assert not any("owned geometry host metadata" in reason for reason in reasons)
        assert owned._validity is None
        assert owned._tags is None
        assert owned._family_row_offsets is None

    def test_device_geometry_array_isna_exports_only_validity(self) -> None:
        """Public NA checks should not materialize full owned host metadata."""
        from vibespatial.cuda._runtime import (
            get_d2h_transfer_events,
            reset_d2h_transfer_count,
        )
        from vibespatial.geometry.device_array import DeviceGeometryArray

        owned = _make_device_resident([Point(0, 0), None])
        owned._validity = None
        owned._tags = None
        owned._family_row_offsets = None

        reset_d2h_transfer_count()
        get_d2h_transfer_events(clear=True)
        mask = DeviceGeometryArray._from_owned(owned).isna()
        reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

        assert mask.tolist() == [False, True]
        assert "DeviceGeometryArray isna validity terminal export" in reasons
        assert not any("owned geometry host metadata" in reason for reason in reasons)
        assert owned._validity is None
        assert owned._tags is None
        assert owned._family_row_offsets is None

    def test_device_geometry_scalar_getitem_uses_device_row_metadata(self) -> None:
        """Scalar row export should not materialize full host metadata triplets."""
        from vibespatial.cuda._runtime import (
            get_d2h_transfer_events,
            reset_d2h_transfer_count,
        )
        from vibespatial.geometry.device_array import DeviceGeometryArray

        owned = _make_device_resident(
            [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
            ]
        )
        owned._validity = None
        owned._tags = None
        owned._family_row_offsets = None

        reset_d2h_transfer_count()
        get_d2h_transfer_events(clear=True)
        geom = DeviceGeometryArray._from_owned(owned)[0]
        reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

        assert geom is not None
        assert geom.geom_type == "Polygon"
        assert "DeviceGeometryArray scalar row metadata device export" in reasons
        assert not any("owned geometry host metadata" in reason for reason in reasons)
        assert owned._validity is None
        assert owned._tags is None
        assert owned._family_row_offsets is None

    def test_device_geometry_buffer_uses_device_admission_metadata(self) -> None:
        """DeviceGeometryArray.buffer should admit device rows without host metadata."""
        from vibespatial.cuda._runtime import (
            get_d2h_transfer_events,
            reset_d2h_transfer_count,
        )
        from vibespatial.geometry.device_array import DeviceGeometryArray

        owned = _make_device_resident([Point(0, 0), Point(1, 1)])
        owned._validity = None
        owned._tags = None
        owned._family_row_offsets = None

        reset_d2h_transfer_count()
        get_d2h_transfer_events(clear=True)
        buffered = DeviceGeometryArray._from_owned(owned).buffer(1.0, quad_segs=1)
        reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

        assert len(buffered) == 2
        assert "point buffer validity admission scalar fence" not in reasons
        assert "point buffer empty-point admission scalar fence" not in reasons
        assert not any("owned geometry host metadata" in reason for reason in reasons)
        assert owned._validity is None
        assert owned._tags is None
        assert owned._family_row_offsets is None

    def test_mixed_device_area_uses_device_row_mapping_metadata(self) -> None:
        """Mixed-family public area should not export tag/family metadata triplets."""
        from vibespatial.constructive.measurement import area_owned
        from vibespatial.cuda._runtime import (
            get_d2h_transfer_events,
            reset_d2h_transfer_count,
        )

        owned = _make_device_resident(
            [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 0)]),
                MultiPolygon(
                    [
                        Polygon([(4, 0), (6, 0), (6, 2), (4, 0)]),
                    ]
                ),
            ]
        )
        owned._validity = None
        owned._tags = None
        owned._family_row_offsets = None

        reset_d2h_transfer_count()
        get_d2h_transfer_events(clear=True)
        areas = area_owned(owned, dispatch_mode=ExecutionMode.GPU, precision="fp64")
        reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

        assert np.allclose(areas, [2.0, 2.0])
        assert "geometry area device-result host export" in reasons
        assert not any("owned geometry host metadata" in reason for reason in reasons)
        assert owned._validity is None
        assert owned._tags is None
        assert owned._family_row_offsets is None

    def test_owned_to_shapely_exports_requested_device_metadata_rows(self) -> None:
        """Terminal Shapely export should not populate generic host metadata."""
        from vibespatial.cuda._runtime import (
            get_d2h_transfer_events,
            reset_d2h_transfer_count,
        )
        from vibespatial.geometry.host_bridge import owned_to_shapely

        owned = _make_device_resident(
            [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 0)]),
            ]
        )
        owned._validity = None
        owned._tags = None
        owned._family_row_offsets = None

        reset_d2h_transfer_count()
        get_d2h_transfer_events(clear=True)
        values = owned_to_shapely(owned)
        reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

        assert values[0].geom_type == "Polygon"
        assert "owned geometry shapely export validity metadata boundary" in reasons
        assert not any("owned geometry host metadata" in reason for reason in reasons)
        assert owned._validity is None
        assert owned._tags is None
        assert owned._family_row_offsets is None

    def test_owned_to_shapely_indexed_view_uses_base_family_rows(self) -> None:
        """Row-indirected Shapely export must address the shared base buffers."""
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        cp = pytest.importorskip("cupy")

        from vibespatial.cuda._runtime import (
            get_d2h_transfer_events,
            reset_d2h_transfer_count,
        )
        from vibespatial.geometry.host_bridge import owned_to_shapely

        polygons = []
        for i in range(14):
            x0 = float(i * 4)
            shell = [(x0, 0), (x0 + 3, 0), (x0 + 3, 3), (x0, 3), (x0, 0)]
            if i % 2:
                polygons.append(
                    Polygon(
                        shell,
                        [
                            [
                                (x0 + 0.5, 0.5),
                                (x0 + 1.0, 0.5),
                                (x0 + 1.0, 1.0),
                                (x0 + 0.5, 0.5),
                            ]
                        ],
                    )
                )
            else:
                polygons.append(Polygon(shell))
        owned = _make_device_resident(polygons)
        owned._validity = None
        owned._tags = None
        owned._family_row_offsets = None

        view = owned._device_indexed_take(
            cp.asarray([13, 0, 12], dtype=cp.int64),
        )
        view._validity = None
        view._tags = None
        view._family_row_offsets = None

        reset_d2h_transfer_count()
        get_d2h_transfer_events(clear=True)
        values = owned_to_shapely(view)
        reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

        expected = [polygons[13], polygons[0], polygons[12]]
        for value, expected_geometry in zip(values, expected, strict=True):
            assert value.equals(expected_geometry)
        assert "owned geometry shapely export family-row metadata boundary" in reasons
        assert not any("owned geometry host metadata" in reason for reason in reasons)
        assert view.is_indexed_view
        assert view._validity is None
        assert view._tags is None
        assert view._family_row_offsets is None

    def test_owned_to_shapely_composes_nested_device_index_maps(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Terminal export must not physicalize an indexed-view ancestry."""
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        cp = pytest.importorskip("cupy")

        from vibespatial.geometry.host_bridge import owned_to_shapely
        from vibespatial.geometry.owned import OwnedGeometryArray

        polygons = [
            Polygon(
                [(i * 4, 0), (i * 4 + 3, 0), (i * 4 + 3, 3), (i * 4, 0)]
            )
            for i in range(8)
        ]
        owned = _make_device_resident(polygons)
        first = OwnedGeometryArray._indexed_view(
            owned,
            cp.asarray([7, 0, 6, 1, 5, 2], dtype=cp.int64),
            assume_unique_indices=True,
        )
        second = OwnedGeometryArray._indexed_view(
            first,
            cp.asarray([4, 0, 2], dtype=cp.int64),
            assume_unique_indices=True,
        )

        def _fail_physicalize(*_args, **_kwargs):
            raise AssertionError("nested terminal export must preserve row indirection")

        monkeypatch.setattr(
            OwnedGeometryArray,
            "physicalize_device_rows",
            _fail_physicalize,
        )
        values = owned_to_shapely(second)

        expected = [polygons[5], polygons[7], polygons[6]]
        assert len(values) == len(expected)
        for value, expected_geometry in zip(values, expected, strict=True):
            assert value.equals(expected_geometry)

    def test_dense_polygon_device_take_no_d2h(self) -> None:
        """Fixed-width one-ring polygon takes should stay fully device-side."""
        import cupy as cp

        from vibespatial.cuda._runtime import assert_zero_d2h_transfers

        owned = from_shapely_geometries(
            [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 1), (2, 0)]),
                Polygon([(4, 0), (5, 0), (5, 1), (4, 1), (4, 0)]),
            ],
            residency=Residency.DEVICE,
        )

        with assert_zero_d2h_transfers():
            result = owned.device_take(cp.asarray([2, 0], dtype=cp.int64))

        assert result.residency is Residency.DEVICE
        assert result.row_count == 2
        assert result.device_state is not None
        polygon_buffer = result.device_state.families[next(iter(result.device_state.families))]
        assert int(polygon_buffer.x.size) == 10

    def test_variable_polygon_host_index_device_take_uses_host_sizing_no_d2h(self) -> None:
        """Host-known rowsets should not copy device totals just to size polygon takes."""
        from vibespatial.cuda._runtime import assert_zero_d2h_transfers

        polygon_with_hole = Polygon(
            [(0, 0), (6, 0), (6, 6), (0, 6), (0, 0)],
            [[(1, 1), (2, 1), (2, 2), (1, 1)]],
        )
        simple_polygon = Polygon([(10, 0), (13, 0), (13, 3), (10, 0)])
        owned = from_shapely_geometries(
            [polygon_with_hole, simple_polygon],
            residency=Residency.DEVICE,
        )

        with assert_zero_d2h_transfers():
            result = owned.take(np.asarray([1, 0], dtype=np.int64))

        assert result.residency is Residency.DEVICE
        assert result.device_state is not None
        restored = result.to_shapely()
        assert restored[0].equals(simple_polygon)
        assert restored[1].equals(polygon_with_hole)

    def test_variable_polygon_host_bool_take_uses_host_sizing_no_d2h(self) -> None:
        """Host boolean masks should lower to integer sizing mirrors before device take."""
        from vibespatial.cuda._runtime import assert_zero_d2h_transfers

        polygon_with_hole = Polygon(
            [(0, 0), (6, 0), (6, 6), (0, 6), (0, 0)],
            [[(1, 1), (2, 1), (2, 2), (1, 1)]],
        )
        simple_polygon = Polygon([(10, 0), (13, 0), (13, 3), (10, 0)])
        owned = from_shapely_geometries(
            [polygon_with_hole, simple_polygon],
            residency=Residency.DEVICE,
        )

        with assert_zero_d2h_transfers():
            result = owned.take(np.asarray([False, True], dtype=bool))

        assert result.residency is Residency.DEVICE
        restored = result.to_shapely()
        assert len(restored) == 1
        assert restored[0].equals(simple_polygon)

    def test_variable_polygon_device_index_take_preserves_row_indirection_no_d2h(self) -> None:
        """Device-only variable polygon rowsets should not size a physical gather."""
        import cupy as cp

        from vibespatial.cuda._runtime import (
            get_d2h_transfer_events,
            reset_d2h_transfer_count,
        )

        polygon_with_hole = Polygon(
            [(0, 0), (6, 0), (6, 6), (0, 6), (0, 0)],
            [[(1, 1), (2, 1), (2, 2), (1, 1)]],
        )
        simple_polygon = Polygon([(10, 0), (13, 0), (13, 3), (10, 0)])
        owned = from_shapely_geometries(
            [polygon_with_hole, simple_polygon],
            residency=Residency.DEVICE,
        )

        reset_d2h_transfer_count()
        get_d2h_transfer_events(clear=True)
        result = owned.device_take(cp.asarray([1, 0], dtype=cp.int64))
        events = get_d2h_transfer_events(clear=True)

        assert result.residency is Residency.DEVICE
        assert result.is_indexed_view
        assert [event.reason for event in events if "device-take" in event.reason] == []
        restored = result.to_shapely()
        assert restored[0].equals(simple_polygon)
        assert restored[1].equals(polygon_with_hole)

    def test_fixed_width_polygon_device_index_take_uses_structural_size_no_d2h(self) -> None:
        """Fixed-width one-ring polygons can size device takes from metadata."""
        import cupy as cp

        from vibespatial.cuda._runtime import assert_zero_d2h_transfers

        left = Polygon([(0, 0), (2, 0), (0, 2), (0, 0)])
        right = Polygon([(10, 0), (12, 0), (10, 2), (10, 0)])
        owned = from_shapely_geometries([left, right], residency=Residency.DEVICE)

        with assert_zero_d2h_transfers():
            result = owned.device_take(cp.asarray([1, 0], dtype=cp.int64))

        assert result.residency is Residency.DEVICE
        restored = result.to_shapely()
        assert restored[0].equals(right)
        assert restored[1].equals(left)

    def test_variable_polygon_device_take_host_sizing_hint_no_d2h(self) -> None:
        """Device rowsets may carry a host sizing mirror without re-uploading indices."""
        import cupy as cp

        from vibespatial.cuda._runtime import assert_zero_d2h_transfers

        polygon_with_hole = Polygon(
            [(0, 0), (6, 0), (6, 6), (0, 6), (0, 0)],
            [[(1, 1), (2, 1), (2, 2), (1, 1)]],
        )
        simple_polygon = Polygon([(10, 0), (13, 0), (13, 3), (10, 0)])
        owned = from_shapely_geometries(
            [polygon_with_hole, simple_polygon],
            residency=Residency.DEVICE,
        )
        host_rows = np.asarray([1, 0], dtype=np.int64)
        device_rows = cp.asarray(host_rows, dtype=cp.int64)

        with assert_zero_d2h_transfers():
            result = owned.device_take(
                device_rows,
                host_indices_for_sizing=host_rows,
            )

        assert result.residency is Residency.DEVICE
        assert result._validity is not None
        assert result._tags is not None
        assert result._family_row_offsets is not None
        restored = result.to_shapely()
        assert restored[0].equals(simple_polygon)
        assert restored[1].equals(polygon_with_hole)

    def test_geometry_array_take_preserves_host_sizing_hint_no_d2h(self) -> None:
        """Public GeometryArray takes should keep the host row mirror for sizing."""
        from vibespatial.api.geometry_array import GeometryArray
        from vibespatial.cuda._runtime import assert_zero_d2h_transfers

        polygon_with_hole = Polygon(
            [(0, 0), (6, 0), (6, 6), (0, 6), (0, 0)],
            [[(1, 1), (2, 1), (2, 2), (1, 1)]],
        )
        simple_polygon = Polygon([(10, 0), (13, 0), (13, 3), (10, 0)])
        owned = from_shapely_geometries(
            [polygon_with_hole, simple_polygon],
            residency=Residency.DEVICE,
        )
        array = GeometryArray.from_owned(owned)

        with assert_zero_d2h_transfers():
            result = array.take(np.asarray([1, 0], dtype=np.int64))

        result_owned = result.to_owned()
        assert result_owned.residency is Residency.DEVICE
        restored = result_owned.to_shapely()
        assert restored[0].equals(simple_polygon)
        assert restored[1].equals(polygon_with_hole)

    def test_device_scatter_keeps_host_metadata_lazy_no_d2h(self) -> None:
        """Variable-width scatter should retain an injective device row map."""
        from vibespatial.cuda._runtime import assert_zero_d2h_transfers
        from vibespatial.geometry.owned import concat_owned_scatter

        polygon_with_hole = Polygon(
            [(0, 0), (6, 0), (6, 6), (0, 6), (0, 0)],
            [[(1, 1), (2, 1), (2, 2), (1, 1)]],
        )
        simple_polygon = Polygon([(10, 0), (13, 0), (13, 3), (10, 0)])
        replacement_polygon = Polygon([(20, 0), (24, 0), (24, 4), (20, 0)])
        base = from_shapely_geometries(
            [polygon_with_hole, simple_polygon],
            residency=Residency.DEVICE,
        )
        replacement = from_shapely_geometries(
            [replacement_polygon],
            residency=Residency.DEVICE,
        )

        with assert_zero_d2h_transfers():
            result = concat_owned_scatter(
                base,
                replacement,
                np.asarray([1], dtype=np.int64),
            )

        assert result.residency is Residency.DEVICE
        assert result.is_indexed_view
        assert result._index_map_unique
        assert result._validity is None
        assert result._tags is None
        assert result._family_row_offsets is None
        restored = result.to_shapely()
        assert restored[0].equals(polygon_with_hole)
        assert restored[1].equals(replacement_polygon)

    def test_device_scatter_uses_rowset_shape_for_family_presence_no_d2h(self) -> None:
        """Device scatter should not scalar-probe family presence."""
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        cp = pytest.importorskip("cupy")
        from vibespatial.cuda._runtime import assert_zero_d2h_transfers
        from vibespatial.geometry.owned import concat_owned_scatter

        base_polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])
        replacement_polygon = Polygon([(2, 0), (3, 0), (3, 1), (2, 0)])
        base = from_shapely_geometries(
            [base_polygon, LineString([(10, 0), (11, 0)])],
            residency=Residency.DEVICE,
        )
        replacement = from_shapely_geometries(
            [replacement_polygon],
            residency=Residency.DEVICE,
        )

        with assert_zero_d2h_transfers():
            result = concat_owned_scatter(
                base,
                replacement,
                cp.asarray([1], dtype=cp.int64),
            )

        assert result.residency is Residency.DEVICE
        restored = result.to_shapely()
        assert restored[0].equals(base_polygon)
        assert restored[1].equals(replacement_polygon)

    def test_mixed_device_concat_uses_row_aligned_family_offsets_no_compaction(self) -> None:
        """Mixed device concat should assemble row offsets without row compaction."""
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        cp = pytest.importorskip("cupy")
        from vibespatial.cuda._runtime import assert_zero_d2h_transfers
        from vibespatial.geometry.owned import OwnedGeometryArray

        left_geoms = [
            Point(0, 0),
            Polygon([(1, 0), (2, 0), (2, 1), (1, 0)]),
            MultiPolygon(
                [
                    Polygon([(3, 0), (4, 0), (4, 1), (3, 0)]),
                ]
            ),
        ]
        right_geoms = [
            Polygon([(5, 0), (6, 0), (6, 1), (5, 0)]),
            Point(7, 0),
        ]
        left = from_shapely_geometries(left_geoms, residency=Residency.DEVICE)
        right = from_shapely_geometries(right_geoms, residency=Residency.DEVICE)

        original_flatnonzero = cp.flatnonzero
        calls = 0

        def _count_flatnonzero(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original_flatnonzero(*args, **kwargs)

        cp.flatnonzero = _count_flatnonzero
        try:
            with assert_zero_d2h_transfers():
                result = OwnedGeometryArray.concat([left, right])
        finally:
            cp.flatnonzero = original_flatnonzero

        assert result.residency is Residency.DEVICE
        assert calls == 0
        restored = result.to_shapely()
        expected = left_geoms + right_geoms
        assert [geom.geom_type for geom in restored] == [geom.geom_type for geom in expected]
        assert all(
            actual.equals(expected_geom)
            for actual, expected_geom in zip(restored, expected, strict=True)
        )

    def test_device_scatter_device_rowset_returns_indexed_carrier_no_take_fence(self) -> None:
        """Device scatter rowsets should stay row-indirected instead of gathering."""
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        cp = pytest.importorskip("cupy")
        from vibespatial.cuda._runtime import (
            assert_zero_d2h_transfers,
            get_d2h_transfer_events,
            reset_d2h_transfer_count,
        )
        from vibespatial.geometry.owned import concat_owned_scatter

        polygon_with_hole = Polygon(
            [(0, 0), (6, 0), (6, 6), (0, 6), (0, 0)],
            [[(1, 1), (2, 1), (2, 2), (1, 1)]],
        )
        simple_polygon = Polygon([(10, 0), (13, 0), (13, 3), (10, 0)])
        replacement_polygon = Polygon([(20, 0), (24, 0), (24, 4), (20, 0)])
        base = _make_device_resident([polygon_with_hole, simple_polygon])
        replacement = _make_device_resident([replacement_polygon])

        reset_d2h_transfer_count()
        get_d2h_transfer_events(clear=True)
        with assert_zero_d2h_transfers():
            result = concat_owned_scatter(
                base,
                replacement,
                cp.asarray([1], dtype=cp.int64),
            )
        reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

        assert result.residency is Residency.DEVICE
        assert result.is_indexed_view
        assert (
            getattr(result, "_device_scatter_implementation", None)
            == "device_scatter_row_indirection"
        )
        assert not any("device-take" in reason or "offset-slice" in reason for reason in reasons)
        restored = result.to_shapely()
        assert restored[0].equals(polygon_with_hole)
        assert restored[1].equals(replacement_polygon)

    def test_device_scatter_full_replacement_permutation_no_take_fence(self) -> None:
        """Full replacement scatter should be a rowset permutation carrier."""
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        cp = pytest.importorskip("cupy")
        from vibespatial.cuda._runtime import (
            assert_zero_d2h_transfers,
            get_d2h_transfer_events,
            reset_d2h_transfer_count,
        )
        from vibespatial.geometry.owned import concat_owned_scatter

        base = _make_device_resident(
            [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 0)]),
                Polygon([(4, 0), (5, 0), (5, 1), (4, 0)]),
            ]
        )
        replacement_geoms = [
            MultiPolygon(
                [
                    Polygon([(10, 0), (12, 0), (12, 2), (10, 0)]),
                    Polygon([(13, 0), (14, 0), (14, 1), (13, 0)]),
                ]
            ),
            Polygon(
                [(20, 0), (26, 0), (26, 6), (20, 0)],
                [[(21, 1), (22, 1), (22, 2), (21, 1)]],
            ),
            MultiPolygon(
                [
                    Polygon([(30, 0), (32, 0), (32, 2), (30, 0)]),
                ]
            ),
        ]
        replacement = _make_device_resident(replacement_geoms)

        reset_d2h_transfer_count()
        get_d2h_transfer_events(clear=True)
        with assert_zero_d2h_transfers():
            result = concat_owned_scatter(
                base,
                replacement,
                cp.asarray([2, 0, 1], dtype=cp.int64),
            )
        reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

        assert result.residency is Residency.DEVICE
        assert result.is_indexed_view
        assert (
            getattr(result, "_device_scatter_implementation", None)
            == "device_scatter_row_indirection"
        )
        assert not any("device-take" in reason or "offset-slice" in reason for reason in reasons)
        restored = result.to_shapely()
        expected = [replacement_geoms[1], replacement_geoms[2], replacement_geoms[0]]
        for actual, expected_geometry in zip(restored, expected, strict=True):
            assert actual.equals(expected_geometry)

    def test_mixed_family_device_take_uses_rowset_shape_for_presence_no_d2h(self) -> None:
        """Mixed-family device takes should not scalar-probe family presence."""
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        cp = pytest.importorskip("cupy")
        from vibespatial.cuda._runtime import assert_zero_d2h_transfers

        point = Point(0, 0)
        polygon = Polygon([(1, 0), (2, 0), (2, 1), (1, 0)])
        owned = from_shapely_geometries(
            [point, polygon],
            residency=Residency.DEVICE,
        )

        with assert_zero_d2h_transfers():
            result = owned.take(cp.asarray([1, 0], dtype=cp.int64))

        assert result.residency is Residency.DEVICE
        restored = result.to_shapely()
        assert restored[0].equals(polygon)
        assert restored[1].equals(point)

    def test_repeated_host_take_unique_base_preserves_host_sizing_no_d2h(self) -> None:
        """Indexed-view unique-base gathers should keep host sizing mirrors."""
        from vibespatial.cuda._runtime import assert_zero_d2h_transfers

        polygon_with_hole = Polygon(
            [(0, 0), (6, 0), (6, 6), (0, 6), (0, 0)],
            [[(1, 1), (2, 1), (2, 2), (1, 1)]],
        )
        simple_polygon = Polygon([(10, 0), (13, 0), (13, 3), (10, 0)])
        owned = from_shapely_geometries(
            [polygon_with_hole, simple_polygon],
            residency=Residency.DEVICE,
        )
        rows = np.tile(np.asarray([1, 0], dtype=np.int64), 600)

        with assert_zero_d2h_transfers():
            result = owned.take(rows)

        assert result.residency is Residency.DEVICE
        assert result.is_indexed_view
        restored = result.to_shapely()
        assert restored[0].equals(simple_polygon)
        assert restored[1].equals(polygon_with_hole)

    def test_single_family_device_take_host_sizing_hint_no_d2h_without_metadata(self) -> None:
        """Single-family takes can size from host row mirrors without routing metadata."""
        import cupy as cp

        from vibespatial.cuda._runtime import assert_zero_d2h_transfers

        polygon_with_hole = Polygon(
            [(0, 0), (6, 0), (6, 6), (0, 6), (0, 0)],
            [[(1, 1), (2, 1), (2, 2), (1, 1)]],
        )
        simple_polygon = Polygon([(10, 0), (13, 0), (13, 3), (10, 0)])
        owned = from_shapely_geometries(
            [polygon_with_hole, simple_polygon],
            residency=Residency.DEVICE,
        )
        owned._validity = None
        owned._tags = None
        owned._family_row_offsets = None

        with assert_zero_d2h_transfers():
            result = owned.device_take(
                cp.asarray([1, 0], dtype=cp.int64),
                host_indices_for_sizing=np.asarray([1, 0], dtype=np.int64),
            )

        assert result.residency is Residency.DEVICE
        restored = result.to_shapely()
        assert restored[0].equals(simple_polygon)
        assert restored[1].equals(polygon_with_hole)

    def test_single_family_device_take_uses_family_row_offsets(self) -> None:
        """Single-family device takes must respect reordered family buffers."""
        import cupy as cp

        from vibespatial.cuda._runtime import assert_zero_d2h_transfers
        from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema
        from vibespatial.geometry.owned import (
            FAMILY_TAGS,
            FamilyGeometryBuffer,
            OwnedGeometryArray,
            OwnedGeometryDeviceState,
        )

        logical = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1), (2, 0)]),
            Polygon([(4, 0), (5, 0), (5, 1), (4, 1), (4, 0)]),
        ]
        physical = from_shapely_geometries(
            [logical[2], logical[0], logical[1]],
            residency=Residency.DEVICE,
        )
        family = GeometryFamily.POLYGON
        schema = get_geometry_buffer_schema(family)
        owned = OwnedGeometryArray(
            validity=None,
            tags=None,
            family_row_offsets=None,
            families={
                family: FamilyGeometryBuffer(
                    family=family,
                    schema=schema,
                    row_count=3,
                    x=np.empty(0, dtype=np.float64),
                    y=np.empty(0, dtype=np.float64),
                    geometry_offsets=np.empty(0, dtype=np.int32),
                    empty_mask=np.empty(0, dtype=np.bool_),
                    host_materialized=False,
                )
            },
            residency=Residency.DEVICE,
            device_state=OwnedGeometryDeviceState(
                validity=cp.ones(3, dtype=cp.bool_),
                tags=cp.full(3, FAMILY_TAGS[family], dtype=cp.int8),
                family_row_offsets=cp.asarray([1, 2, 0], dtype=cp.int32),
                families=physical.device_state.families,
            ),
            _row_count=3,
        )

        with assert_zero_d2h_transfers():
            result = owned.device_take(cp.asarray([0, 2], dtype=cp.int64))

        restored = result.to_shapely()
        assert restored[0].equals(logical[0])
        assert restored[1].equals(logical[2])

    def test_concat_host_fallback_when_mixed_residency(self) -> None:
        """When some inputs are host-resident, falls back to host concat."""
        from vibespatial.geometry.owned import OwnedGeometryArray

        device_owned = _make_device_resident([Point(0, 0)])
        host_owned = from_shapely_geometries([Point(1, 1)])

        result = OwnedGeometryArray.concat([device_owned, host_owned])

        # Should fall through to host path and produce correct result
        assert result.residency is Residency.HOST
        assert result.row_count == 2

        shapely_result = result.to_shapely()
        assert shapely_result[0].equals(Point(0, 0))
        assert shapely_result[1].equals(Point(1, 1))

    def test_concat_three_arrays_stays_device_resident(self) -> None:
        """Concatenating 3+ device-resident arrays."""
        from vibespatial.geometry.owned import OwnedGeometryArray

        owned1 = _make_device_resident([Point(0, 0)])
        owned2 = _make_device_resident([Point(1, 1)])
        owned3 = _make_device_resident([Point(2, 2)])

        result = OwnedGeometryArray.concat([owned1, owned2, owned3])

        assert result.residency is Residency.DEVICE
        assert result.row_count == 3

        shapely_result = result.to_shapely()
        assert shapely_result[0].equals(Point(0, 0))
        assert shapely_result[1].equals(Point(1, 1))
        assert shapely_result[2].equals(Point(2, 2))

    def test_concat_polygon_with_hole_stays_device_resident(self) -> None:
        """Polygons with holes preserve interior rings through device concat."""
        from vibespatial.geometry.owned import OwnedGeometryArray

        poly_hole = Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
            [[(2, 2), (4, 2), (4, 4), (2, 2)]],
        )
        poly_simple = Polygon([(20, 20), (21, 20), (21, 21), (20, 20)])

        owned1 = _make_device_resident([poly_hole])
        owned2 = _make_device_resident([poly_simple])

        result = OwnedGeometryArray.concat([owned1, owned2])

        assert result.residency is Residency.DEVICE
        shapely_result = result.to_shapely()
        assert shapely_result[0].equals(poly_hole)
        assert len(list(shapely_result[0].interiors)) == 1
        assert shapely_result[1].equals(poly_simple)

    def test_concat_all_families_device_resident(self) -> None:
        """Concatenating arrays covering all 6 geometry families."""
        from vibespatial.geometry.owned import OwnedGeometryArray

        geoms1 = [
            Point(1, 2),
            LineString([(0, 0), (1, 1)]),
            Polygon([(0, 0), (3, 0), (3, 3), (0, 0)]),
        ]
        geoms2 = [
            MultiPoint([(0, 0), (1, 1)]),
            MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]]),
            MultiPolygon(
                [
                    Polygon([(10, 10), (12, 10), (12, 12), (10, 10)]),
                ]
            ),
        ]

        owned1 = _make_device_resident(geoms1)
        owned2 = _make_device_resident(geoms2)

        result = OwnedGeometryArray.concat([owned1, owned2])

        assert result.residency is Residency.DEVICE
        assert result.row_count == 6

        shapely_result = result.to_shapely()
        for i, expected in enumerate(geoms1 + geoms2):
            assert shapely_result[i].equals(expected), (
                f"Mismatch at index {i}: {expected.geom_type}"
            )

    def test_concat_disjoint_families_device_resident(self) -> None:
        """One array has only polygons, the other has only points."""
        from vibespatial.geometry.owned import OwnedGeometryArray

        owned1 = _make_device_resident(
            [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
            ]
        )
        owned2 = _make_device_resident(
            [
                Point(5, 5),
            ]
        )

        result = OwnedGeometryArray.concat([owned1, owned2])

        assert result.residency is Residency.DEVICE
        assert result.row_count == 2

        shapely_result = result.to_shapely()
        assert shapely_result[0].equals(Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]))
        assert shapely_result[1].equals(Point(5, 5))


class TestEnsureDeviceStateSafetyCheck:
    """Verify that _ensure_device_state refuses to upload unmaterialised stubs.

    When an OwnedGeometryArray has host_materialized=False stubs (empty
    x/y arrays) AND no device_state, uploading those stubs creates
    zero-length device buffers while metadata references rows that should
    contain coordinates.  Kernels then read garbage from uninitialized GPU
    memory, producing denormalized-double coordinates (e.g. 8e-309) and
    downstream TopologyException crashes.
    """

    @pytest.mark.skipif(not has_gpu_runtime(), reason="GPU not available")
    def test_ensure_device_state_rejects_unmaterialised_stubs(self) -> None:
        """_ensure_device_state raises RuntimeError for empty stubs without device_state."""
        from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema
        from vibespatial.geometry.owned import FamilyGeometryBuffer, OwnedGeometryArray

        # Build an OGA that simulates the bug pattern:
        # - residency=HOST (or DEVICE)
        # - host families have host_materialized=False stubs
        # - device_state is None (lost during incorrect construction)
        polygon_stub = FamilyGeometryBuffer(
            family=GeometryFamily.POLYGON,
            schema=get_geometry_buffer_schema(GeometryFamily.POLYGON),
            row_count=1,
            x=np.empty(0, dtype=np.float64),
            y=np.empty(0, dtype=np.float64),
            geometry_offsets=np.empty(0, dtype=np.int32),
            empty_mask=np.empty(0, dtype=np.bool_),
            host_materialized=False,
        )
        from vibespatial.geometry.owned import FAMILY_TAGS

        oga = OwnedGeometryArray(
            validity=np.array([True], dtype=bool),
            tags=np.array([FAMILY_TAGS[GeometryFamily.POLYGON]], dtype=np.int8),
            family_row_offsets=np.array([0], dtype=np.int32),
            families={GeometryFamily.POLYGON: polygon_stub},
            residency=Residency.HOST,
            device_state=None,
        )

        with pytest.raises(RuntimeError, match="unmaterialised stubs"):
            oga._ensure_device_state()

    @pytest.mark.skipif(not has_gpu_runtime(), reason="GPU not available")
    def test_ensure_device_state_succeeds_for_materialised_host(self) -> None:
        """_ensure_device_state succeeds when host families are properly materialised."""
        from vibespatial.geometry.buffers import GeometryFamily as GF

        owned = from_shapely_geometries(
            [Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])],
            residency=Residency.HOST,
        )
        # This should succeed -- host families have real data
        d_state = owned._ensure_device_state()
        assert d_state is not None
        assert GF.POLYGON in d_state.families

    @pytest.mark.skipif(not has_gpu_runtime(), reason="GPU not available")
    def test_ensure_device_state_shortcircuits_when_device_state_exists(self) -> None:
        """_ensure_device_state returns existing device_state without re-uploading."""
        owned = _make_device_resident(
            [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
            ]
        )
        assert owned.device_state is not None

        # Families are unmaterialised stubs, but device_state exists
        # so _ensure_device_state should short-circuit and not hit the check
        d_state = owned._ensure_device_state()
        assert d_state is owned.device_state
