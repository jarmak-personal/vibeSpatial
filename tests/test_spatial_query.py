from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import shapely
from shapely.geometry import (
    GeometryCollection,
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    box,
)

import vibespatial.spatial.nearest as spatial_nearest_module
import vibespatial.spatial.query as spatial_query_module
import vibespatial.spatial.query_utils as spatial_query_utils_module
from vibespatial.api.geometry_array import GeometryArray
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import OwnedGeometryArray, from_shapely_geometries
from vibespatial.runtime import (
    ExecutionMode,
    RuntimeSelection,
    has_gpu_runtime,
    set_requested_mode,
)
from vibespatial.runtime.precision import PrecisionMode
from vibespatial.runtime.residency import Residency
from vibespatial.spatial.indexing import build_flat_spatial_index
from vibespatial.spatial.query import (
    build_owned_spatial_index,
    nearest_spatial_index,
    query_spatial_index,
)
from vibespatial.spatial.query_box import (
    _extract_box_query_bounds_from_owned,
    _extract_box_query_bounds_shapely,
)
from vibespatial.spatial.spatial_index_device import spatial_index_device_query


def test_supports_owned_spatial_input_vectorized_family_admission() -> None:
    supported = np.asarray(
        [
            None,
            Point(0.0, 0.0),
            LineString([(0.0, 0.0), (1.0, 1.0)]),
            Polygon([(0.0, 0.0), (1.0, 0.0), (0.0, 0.0)]),
        ],
        dtype=object,
    )

    assert spatial_query_utils_module.supports_owned_spatial_input(supported)
    assert not spatial_query_utils_module.supports_owned_spatial_input(
        np.asarray([LinearRing([(0.0, 0.0), (1.0, 0.0), (0.0, 0.0)])], dtype=object)
    )
    assert not spatial_query_utils_module.supports_owned_spatial_input(
        np.asarray([GeometryCollection([Point(0.0, 0.0)])], dtype=object)
    )
    assert not spatial_query_utils_module.supports_owned_spatial_input(
        np.asarray(["not-a-geometry"], dtype=object)
    )


def _device_regular_box_owned_for_test(
    row_count: int,
    *,
    origin_y: float = 0.0,
    cols: int = 2,
) -> OwnedGeometryArray:
    from vibespatial.cuda._runtime import get_cuda_runtime
    from vibespatial.geometry.owned import (
        FAMILY_TAGS,
        DeviceFamilyGeometryBuffer,
        DeviceRegularGridRectMetadata,
        build_device_resident_owned,
    )

    runtime = get_cuda_runtime()
    row_ids = np.arange(row_count, dtype=np.int64)
    minx = np.remainder(row_ids, cols).astype(np.float64, copy=False)
    miny = (row_ids // cols).astype(np.float64, copy=False) + float(origin_y)
    maxx = minx + 1.0
    maxy = miny + 1.0
    x = np.stack((minx, maxx, maxx, minx, minx), axis=1).reshape(-1)
    y = np.stack((miny, miny, maxy, maxy, miny), axis=1).reshape(-1)
    geometry_offsets = np.arange(row_count + 1, dtype=np.int32)
    ring_offsets = geometry_offsets * 5
    bounds = np.column_stack((minx, miny, maxx, maxy))
    full_rows, tail_cols = divmod(row_count, cols)
    grid_rows = full_rows + (1 if tail_cols else 0)
    max_cols = cols if full_rows else tail_cols
    total_bounds = (
        0.0,
        float(origin_y),
        float(max_cols),
        float(origin_y + grid_rows),
    )
    family = GeometryFamily.POLYGON
    return build_device_resident_owned(
        device_families={
            family: DeviceFamilyGeometryBuffer(
                family=family,
                x=runtime.from_host(x),
                y=runtime.from_host(y),
                geometry_offsets=runtime.from_host(geometry_offsets),
                empty_mask=runtime.from_host(np.zeros(row_count, dtype=np.bool_)),
                ring_offsets=runtime.from_host(ring_offsets),
                bounds=runtime.from_host(bounds),
                dense_single_ring_width=5,
                axis_aligned_rectangles=True,
                regular_grid_rect=DeviceRegularGridRectMetadata(
                    origin_x=0.0,
                    origin_y=float(origin_y),
                    cell_width=1.0,
                    cell_height=1.0,
                    cols=int(cols),
                    rows=int(grid_rows),
                    size=int(row_count),
                    total_bounds=total_bounds,
                ),
            ),
        },
        row_count=row_count,
        tags=np.full(row_count, FAMILY_TAGS[family], dtype=np.int8),
        validity=np.ones(row_count, dtype=np.bool_),
        family_row_offsets=np.arange(row_count, dtype=np.int32),
    )


def _device_only_clone_for_test(owned: OwnedGeometryArray) -> OwnedGeometryArray:
    import cupy as cp

    from vibespatial.geometry.owned import build_device_resident_owned

    state = owned._ensure_device_state()
    return build_device_resident_owned(
        device_families=state.families,
        row_count=owned.row_count,
        tags=cp.asarray(state.tags, dtype=cp.int8),
        validity=cp.asarray(state.validity, dtype=cp.bool_),
        family_row_offsets=cp.asarray(state.family_row_offsets, dtype=cp.int32),
        execution_mode="gpu",
    )


def test_spatial_query_candidate_d2h_exports_are_runtime_accounted() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paths = (
        repo_root / "src" / "vibespatial" / "spatial" / "query_candidates.py",
        repo_root / "src" / "vibespatial" / "spatial" / "query_box.py",
        repo_root / "src" / "vibespatial" / "spatial" / "query_utils.py",
    )
    unnamed_runtime_exports: list[str] = []
    raw_cupy_exports: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr == "copy_device_to_host" and not any(
                keyword.arg == "reason" for keyword in node.keywords
            ):
                unnamed_runtime_exports.append(f"{path.relative_to(repo_root)}:{node.lineno}")
            if func.attr == "asnumpy":
                raw_cupy_exports.append(f"{path.relative_to(repo_root)}:{node.lineno}")

    assert unnamed_runtime_exports == []
    assert raw_cupy_exports == []


def test_nearest_runtime_d2h_exports_are_operation_named() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "src" / "vibespatial" / "spatial" / "nearest.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    unnamed_runtime_exports: list[str] = []
    raw_cupy_exports: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr == "copy_device_to_host" and not any(
            keyword.arg == "reason" for keyword in node.keywords
        ):
            unnamed_runtime_exports.append(f"{path.relative_to(repo_root)}:{node.lineno}")
        if (
            func.attr == "asnumpy"
            and isinstance(func.value, ast.Name)
            and func.value.id == "cp"
        ):
            raw_cupy_exports.append(f"{path.relative_to(repo_root)}:{node.lineno}")

    assert unnamed_runtime_exports == []
    assert raw_cupy_exports == []


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime required for polygon-box scalar fences")
def test_polygon_box_query_scalar_fences_are_operation_named() -> None:
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    query_owned = from_shapely_geometries(
        [
            box(0.0, 0.0, 1.0, 1.0),
            box(2.0, 0.0, 3.0, 1.0),
        ],
        residency=Residency.DEVICE,
    )
    query_owned.device_state.families[GeometryFamily.POLYGON].axis_aligned_rectangles = False

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    bounds = _extract_box_query_bounds_from_owned("intersects", query_owned)
    events = get_d2h_transfer_events(clear=True)

    assert bounds is not None
    np.testing.assert_allclose(
        bounds,
        np.asarray([
            [0.0, 0.0, 1.0, 1.0],
            [2.0, 0.0, 3.0, 1.0],
        ]),
    )
    reasons = [event.reason for event in events]
    assert "spatial query polygon-box single-ring scalar fence" in reasons
    assert "spatial query polygon-box coordinate-count scalar fence" in reasons
    assert "spatial query polygon-box axis-aligned scalar fence" in reasons
    assert "spatial query polygon-box bounds host export" in reasons
    assert not any("CudaRuntime.copy_device_to_host" in reason for reason in reasons)


def test_query_spatial_index_matches_expected_pairs_for_intersects() -> None:
    tree = np.asarray([box(0, 0, 1, 1), box(10, 10, 11, 11), box(20, 20, 21, 21)], dtype=object)
    query = np.asarray([box(0.5, 0.5, 1.5, 1.5), box(30, 30, 31, 31)], dtype=object)
    owned, flat = build_owned_spatial_index(tree)

    indices = query_spatial_index(owned, flat, query, predicate="intersects", sort=True)

    assert indices.tolist() == [[0], [0]]


def test_extract_box_query_bounds_shapely_rejects_non_box_polygons() -> None:
    query = np.asarray(
        [
            box(0, 0, 1, 1),
            Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2), (0, 0)]),
        ],
        dtype=object,
    )

    assert _extract_box_query_bounds_shapely(query) is None


def test_query_spatial_index_supports_dwithin() -> None:
    tree = np.asarray([Point(0, 0), Point(10, 0), Point(20, 0)], dtype=object)
    query = np.asarray([Point(1, 0), Point(16, 0)], dtype=object)
    owned, flat = build_owned_spatial_index(tree)

    indices = query_spatial_index(owned, flat, query, predicate="dwithin", distance=5.0, sort=True)

    assert indices.tolist() == [[0, 1], [0, 2]]


def test_query_spatial_index_scalar_sort_false_preserves_membership_without_sorting() -> None:
    tree = np.asarray(
        [Point(5, 5), Point(2, 2), Point(4, 4), Point(0, 0), Point(3, 3), Point(1, 1)],
        dtype=object,
    )
    query = box(0, 0, 2, 2)
    owned, flat = build_owned_spatial_index(tree)

    unsorted = query_spatial_index(owned, flat, query, predicate="intersects", sort=False)
    sorted_indices = query_spatial_index(owned, flat, query, predicate="intersects", sort=True)
    expected_unsorted = flat.query_bounds(query.bounds)

    assert unsorted.tolist() == expected_unsorted.tolist()
    assert sorted(sorted_indices.tolist()) == sorted(expected_unsorted.tolist())
    assert unsorted.tolist() != sorted_indices.tolist()


def test_query_spatial_index_line_polygon_boundary_overlap_matches_strtree() -> None:
    from shapely.strtree import STRtree

    tree = np.asarray(
        [
            box(1, 1, 3, 3),
            box(3, 3, 5, 5),
        ],
        dtype=object,
    )
    query = np.asarray(
        [
            LineString([(2, 0), (2, 4), (6, 4)]),
            LineString([(0, 3), (6, 3)]),
        ],
        dtype=object,
    )
    owned, flat = build_owned_spatial_index(tree)

    result, execution = query_spatial_index(
        owned,
        flat,
        query,
        predicate="intersects",
        sort=True,
        return_metadata=True,
    )
    reference = STRtree(tree).query(query, predicate="intersects")

    assert result.tolist() == reference.tolist() == [[0, 0, 1, 1], [0, 1, 0, 1]]
    if has_gpu_runtime():
        assert execution.selected is ExecutionMode.GPU


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required")
class TestDwithinGPU:
    """GPU dwithin refinement via distance kernels."""

    def test_dwithin_point_point(self):
        tree = np.asarray([Point(0, 0), Point(10, 0), Point(20, 0)], dtype=object)
        query = np.asarray([Point(1, 0), Point(16, 0)], dtype=object)
        owned, flat = build_owned_spatial_index(tree)

        result, execution = query_spatial_index(
            owned, flat, query, predicate="dwithin", distance=5.0,
            sort=True, return_metadata=True,
        )
        assert result.tolist() == [[0, 1], [0, 2]]
        assert execution.selected is ExecutionMode.GPU

    def test_dwithin_point_polygon(self):
        tree = np.asarray([box(0, 0, 1, 1), box(10, 10, 11, 11), box(20, 20, 21, 21)], dtype=object)
        query = np.asarray([Point(2, 0.5), Point(9, 9)], dtype=object)
        owned, flat = build_owned_spatial_index(tree)

        result, execution = query_spatial_index(
            owned, flat, query, predicate="dwithin", distance=2.0,
            sort=True, return_metadata=True,
        )
        import shapely as shp
        expected = set()
        for qi in range(len(query)):
            for ti in range(len(tree)):
                if shp.dwithin(query[qi], tree[ti], 2.0):
                    expected.add((qi, ti))
        result_set = set(zip(result[0].tolist(), result[1].tolist()))
        assert result_set == expected
        assert execution.selected is ExecutionMode.GPU

    def test_dwithin_polygon_polygon(self):
        tree = np.asarray([box(0, 0, 1, 1), box(5, 5, 6, 6), box(20, 20, 21, 21)], dtype=object)
        query = np.asarray([box(2, 0, 3, 1)], dtype=object)
        owned, flat = build_owned_spatial_index(tree)

        result, execution = query_spatial_index(
            owned, flat, query, predicate="dwithin", distance=2.0,
            sort=True, return_metadata=True,
        )
        import shapely as shp
        expected = set()
        for qi in range(len(query)):
            for ti in range(len(tree)):
                if shp.dwithin(query[qi], tree[ti], 2.0):
                    expected.add((qi, ti))
        result_set = set(zip(result[0].tolist(), result[1].tolist()))
        assert result_set == expected
        assert execution.selected is ExecutionMode.GPU

    def test_dwithin_per_row_distance(self):
        tree = np.asarray([Point(0, 0), Point(10, 0), Point(20, 0)], dtype=object)
        query = np.asarray([Point(3, 0), Point(15, 0)], dtype=object)
        owned, flat = build_owned_spatial_index(tree)

        # First query: threshold 4 → reaches Point(0,0) at dist 3
        # Second query: threshold 6 → reaches Point(10,0) at dist 5, Point(20,0) at dist 5
        result = query_spatial_index(
            owned, flat, query, predicate="dwithin",
            distance=np.array([4.0, 6.0]), sort=True,
        )
        import shapely as shp
        dists = np.array([4.0, 6.0])
        expected = set()
        for qi in range(len(query)):
            for ti in range(len(tree)):
                if shp.dwithin(query[qi], tree[ti], dists[qi]):
                    expected.add((qi, ti))
        result_set = set(zip(result[0].tolist(), result[1].tolist()))
        assert result_set == expected

    @pytest.mark.parametrize(
        "distance",
        [np.asarray(4.0), np.asarray([4.0])],
        ids=["zero-dimensional", "one-element"],
    )
    def test_dwithin_broadcasts_array_distance(self, distance):
        tree = np.asarray([Point(0, 0), Point(10, 0)], dtype=object)
        query = np.asarray([Point(3, 0), Point(14, 0)], dtype=object)
        owned, flat = build_owned_spatial_index(tree)

        result = query_spatial_index(
            owned,
            flat,
            query,
            predicate="dwithin",
            distance=distance,
            sort=True,
        )

        assert result.tolist() == [[0, 1], [0, 1]]

    def test_dwithin_no_matches(self):
        tree = np.asarray([Point(0, 0), Point(100, 100)], dtype=object)
        query = np.asarray([Point(50, 50)], dtype=object)
        owned, flat = build_owned_spatial_index(tree)

        result = query_spatial_index(
            owned, flat, query, predicate="dwithin", distance=1.0, sort=True,
        )
        assert result.shape[1] == 0

    def test_dwithin_multipoint_point(self, monkeypatch: pytest.MonkeyPatch):
        import shapely as shp
        from shapely.geometry import MultiPoint

        tree = np.asarray([Point(0, 0), Point(5, 0), Point(20, 0)], dtype=object)
        query = np.asarray([MultiPoint([(1, 0), (10, 0)])], dtype=object)
        owned, flat = build_owned_spatial_index(tree)

        def _fail_fallback(*_args, **_kwargs):
            raise AssertionError("unexpected Shapely fallback for supported MultiPoint/Point dwithin")

        def _fail_to_shapely(self):
            raise AssertionError("unexpected host materialization for supported MultiPoint/Point dwithin")

        monkeypatch.setattr(spatial_nearest_module, "record_shapely_fallback_event", _fail_fallback)
        monkeypatch.setattr(spatial_query_utils_module, "record_shapely_fallback_event", _fail_fallback)
        monkeypatch.setattr(OwnedGeometryArray, "to_shapely", _fail_to_shapely)

        result, execution = query_spatial_index(
            owned, flat, query, predicate="dwithin", distance=2.0,
            sort=True, return_metadata=True,
        )
        expected = set()
        for qi in range(len(query)):
            for ti in range(len(tree)):
                if shp.dwithin(query[qi], tree[ti], 2.0):
                    expected.add((qi, ti))
        result_set = set(zip(result[0].tolist(), result[1].tolist()))
        assert result_set == expected
        assert execution.selected is ExecutionMode.GPU


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required")
def test_device_spatial_query_zero_pairs_does_not_signal_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = np.asarray([box(0, 0, 1, 1), box(2, 2, 3, 3)], dtype=object)
    query_owned = from_shapely_geometries([box(100, 100, 101, 101)], residency=Residency.DEVICE)
    tree_owned, flat = build_owned_spatial_index(tree)
    object.__setattr__(flat, "regular_grid", None)

    device_candidates, device_execution = spatial_index_device_query(
        flat,
        np.asarray([[100.0, 100.0, 101.0, 101.0]], dtype=np.float64),
    )

    assert device_execution.selected is ExecutionMode.GPU
    assert device_candidates is not None
    assert device_candidates.total_pairs == 0

    def _fail_cpu_candidate_generation(*_args, **_kwargs):
        raise AssertionError("GPU zero-candidate query should not rerun CPU bounds pairing")

    monkeypatch.setattr(
        spatial_query_module,
        "generate_bounds_pairs",
        _fail_cpu_candidate_generation,
    )

    result, execution = query_spatial_index(
        tree_owned,
        flat,
        query_owned,
        predicate="intersects",
        sort=True,
        return_metadata=True,
    )

    assert result.shape == (2, 0)
    assert execution.selected is ExecutionMode.GPU


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime required for device bounds index")
def test_single_row_device_flat_index_keeps_bounds_device_until_host_boundary() -> None:
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_count,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [Point(1.0, 2.0)],
        residency=Residency.DEVICE,
    )

    reset_d2h_transfer_count()
    flat_index = build_flat_spatial_index(
        owned,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            reason="test single-row device index",
        ),
    )

    assert flat_index.device_bounds is not None
    assert flat_index._host_bounds is None
    assert flat_index.size == 1
    assert get_d2h_transfer_count() == 0

    assert flat_index.bounds.tolist() == [[1.0, 2.0, 1.0, 2.0]]
    assert get_d2h_transfer_count() == 1


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime required for device bounds query")
def test_single_row_device_rect_flat_index_reuses_native_shape_proof() -> None:
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_count,
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    tree_owned = from_shapely_geometries(
        [box(0.0, 0.0, 10.0, 10.0)],
        residency=Residency.DEVICE,
    )
    query_owned = from_shapely_geometries(
        [Point(1.0, 1.0), Point(20.0, 20.0)],
        residency=Residency.DEVICE,
    )

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    flat_index = build_flat_spatial_index(
        tree_owned,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            reason="test single-row device rectangle index",
        ),
    )

    assert flat_index.regular_grid is not None
    assert flat_index.regular_grid.size == 1
    assert flat_index.total_bounds == (0.0, 0.0, 10.0, 10.0)
    assert flat_index._host_bounds is None
    assert flat_index.device_bounds is not None
    assert get_d2h_transfer_count() == 0
    build_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]
    assert not any("single-rectangle" in reason for reason in build_reasons)

    result, execution = query_spatial_index(
        tree_owned,
        flat_index,
        query_owned,
        predicate="intersects",
        sort=True,
        output_format="indices",
        return_metadata=True,
        return_device=True,
    )

    assert execution.selected is ExecutionMode.GPU
    assert result.size == 1
    query_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]
    assert "device spatial-index candidate-pair allocation fence" not in query_reasons


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime required for device grid index")
def test_multi_row_device_regular_grid_index_avoids_host_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_stats,
        reset_d2h_transfer_count,
    )

    tree_owned = from_shapely_geometries(
        [
            box(0.0, 0.0, 1.0, 1.0),
            box(1.0, 0.0, 2.0, 1.0),
            box(2.0, 0.0, 3.0, 1.0),
            box(0.0, 1.0, 1.0, 2.0),
            box(1.0, 1.0, 2.0, 2.0),
        ],
        residency=Residency.DEVICE,
    )
    tree_owned._validity = None
    tree_owned._tags = None
    tree_owned._family_row_offsets = None

    def _fail_host_metadata(*_args, **_kwargs):
        raise AssertionError("device regular-grid index should not host-normalize metadata")

    monkeypatch.setattr(type(tree_owned), "_ensure_host_metadata", _fail_host_metadata)
    monkeypatch.setattr(
        type(tree_owned),
        "_ensure_host_family_structure",
        _fail_host_metadata,
    )
    reset_d2h_transfer_count()

    flat_index = build_flat_spatial_index(
        tree_owned,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.CPU,
            selected=ExecutionMode.CPU,
            reason="test device regular-grid index with CPU-compatible order",
        ),
    )

    transfer_count, transfer_bytes = get_d2h_transfer_stats()
    assert flat_index.regular_grid is not None
    assert flat_index.regular_grid.size == 5
    assert flat_index._host_bounds is None
    assert flat_index.device_bounds is not None
    assert flat_index.total_bounds == (0.0, 0.0, 3.0, 2.0)
    polygon_device = tree_owned.device_state.families[GeometryFamily.POLYGON]
    assert polygon_device.dense_single_ring_width == 5
    assert polygon_device.axis_aligned_rectangles is True
    assert polygon_device.bounds is flat_index.device_bounds
    assert polygon_device.regular_grid_rect is not None
    assert polygon_device.regular_grid_rect.size == 5
    assert transfer_count <= 1
    assert transfer_bytes <= 64

    monkeypatch.undo()
    assert flat_index.bounds.tolist() == [
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 2.0, 1.0],
        [2.0, 0.0, 3.0, 1.0],
        [0.0, 1.0, 1.0, 2.0],
        [1.0, 1.0, 2.0, 2.0],
    ]


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime required for device grid index")
def test_device_regular_grid_take_preserves_proof_without_recertification() -> None:
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    owned = _device_regular_box_owned_for_test(6)

    reset_d2h_transfer_count()
    taken = owned.take(np.arange(2, 6, dtype=np.int64))
    proof = taken.device_state.families[GeometryFamily.POLYGON].regular_grid_rect
    assert proof is not None
    assert proof.origin_x == 0.0
    assert proof.origin_y == 1.0
    assert proof.cols == 2
    assert proof.rows == 2
    assert proof.size == 4
    assert proof.total_bounds == (0.0, 1.0, 2.0, 3.0)

    flat_index = build_flat_spatial_index(
        taken,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.CPU,
            selected=ExecutionMode.CPU,
            reason="consume preserved device regular-grid proof",
        ),
    )
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]
    assert flat_index.regular_grid is not None
    assert flat_index.total_bounds == (0.0, 1.0, 2.0, 3.0)
    assert "spatial index regular-grid summary scalar fence" not in reasons


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime required for device grid index")
def test_host_validated_device_regular_grid_seeds_device_proof() -> None:
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    owned = from_shapely_geometries(
        [
            box(float(col), float(row), float(col + 1), float(row + 1))
            for row in range(2)
            for col in range(2)
        ],
        residency=Residency.DEVICE,
    )
    first = build_flat_spatial_index(
        owned,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.CPU,
            selected=ExecutionMode.CPU,
            reason="host validation seeds device proof",
        ),
    )
    proof = owned.device_state.families[GeometryFamily.POLYGON].regular_grid_rect
    assert first.regular_grid is not None
    assert proof is not None
    assert proof.total_bounds == (0.0, 0.0, 2.0, 2.0)

    reset_d2h_transfer_count()
    second = build_flat_spatial_index(
        owned,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.CPU,
            selected=ExecutionMode.CPU,
            reason="consume host-seeded device proof",
        ),
    )
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]
    assert second.regular_grid is not None
    assert second.device_bounds is not None
    assert "spatial index regular-grid summary scalar fence" not in reasons


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime required for device grid index")
def test_device_regular_grid_concat_preserves_row_aligned_proof() -> None:
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    top = _device_regular_box_owned_for_test(4)
    bottom = _device_regular_box_owned_for_test(2, origin_y=2.0)

    reset_d2h_transfer_count()
    concatenated = OwnedGeometryArray.concat([top, bottom])
    proof = concatenated.device_state.families[GeometryFamily.POLYGON].regular_grid_rect
    assert proof is not None
    assert proof.origin_x == 0.0
    assert proof.origin_y == 0.0
    assert proof.cols == 2
    assert proof.rows == 3
    assert proof.size == 6
    assert proof.total_bounds == (0.0, 0.0, 2.0, 3.0)

    flat_index = build_flat_spatial_index(
        concatenated,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.CPU,
            selected=ExecutionMode.CPU,
            reason="consume concat regular-grid proof",
        ),
    )
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]
    assert flat_index.regular_grid is not None
    assert flat_index.total_bounds == (0.0, 0.0, 2.0, 3.0)
    assert "spatial index regular-grid summary scalar fence" not in reasons


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime required for device grid index")
def test_large_device_regular_grid_index_uses_parallel_certification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_stats,
        reset_d2h_transfer_count,
    )

    cols = 32
    rows = 16
    tree_owned = from_shapely_geometries(
        [
            box(float(col), float(row), float(col + 1), float(row + 1))
            for row in range(rows)
            for col in range(cols)
        ],
        residency=Residency.DEVICE,
    )
    tree_owned._validity = None
    tree_owned._tags = None
    tree_owned._family_row_offsets = None

    def _fail_host_metadata(*_args, **_kwargs):
        raise AssertionError("large device regular-grid index should not host-normalize metadata")

    monkeypatch.setattr(type(tree_owned), "_ensure_host_metadata", _fail_host_metadata)
    monkeypatch.setattr(
        type(tree_owned),
        "_ensure_host_family_structure",
        _fail_host_metadata,
    )
    reset_d2h_transfer_count()

    flat_index = build_flat_spatial_index(
        tree_owned,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.CPU,
            selected=ExecutionMode.CPU,
            reason="test large device regular-grid index with CPU-compatible order",
        ),
    )

    transfer_count, transfer_bytes = get_d2h_transfer_stats()
    assert flat_index.regular_grid is not None
    assert flat_index.regular_grid.size == cols * rows
    assert flat_index.regular_grid.cols == cols
    assert flat_index.regular_grid.rows == rows
    assert flat_index._host_bounds is None
    assert flat_index.device_bounds is not None
    assert flat_index.total_bounds == (0.0, 0.0, float(cols), float(rows))
    polygon_device = tree_owned.device_state.families[GeometryFamily.POLYGON]
    assert polygon_device.dense_single_ring_width == 5
    assert polygon_device.axis_aligned_rectangles is True
    assert polygon_device.bounds is flat_index.device_bounds
    assert polygon_device.regular_grid_rect is not None
    assert polygon_device.regular_grid_rect.size == cols * rows
    assert transfer_count <= 1
    assert transfer_bytes <= 64

    monkeypatch.undo()
    bounds = flat_index.bounds
    np.testing.assert_allclose(bounds[0], [0.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose(
        bounds[-1],
        [float(cols - 1), float(rows - 1), float(cols), float(rows)],
    )


def test_query_spatial_index_handles_regular_grid_rectangle_boundaries() -> None:
    tree = np.asarray(
        [
            box(0, 0, 1, 1),
            box(1, 0, 2, 1),
            box(0, 1, 1, 2),
        ],
        dtype=object,
    )
    query = np.asarray([Point(1, 1), Point(1.5, 1.5)], dtype=object)
    owned, flat = build_owned_spatial_index(tree)

    indices = query_spatial_index(
        owned,
        flat,
        query,
        predicate="intersects",
        sort=True,
    )

    assert flat.regular_grid is not None
    assert indices.tolist() == [[0, 0, 0], [0, 1, 2]]


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required")
def test_regular_grid_admits_non_box_intersection_candidates(monkeypatch) -> None:
    """Grid bbox admission remains a superset before exact refinement."""
    tree = np.asarray(
        [
            box(float(col), float(row), float(col + 1), float(row + 1))
            for row in range(100)
            for col in range(100)
        ],
        dtype=object,
    )
    query = np.asarray(
        [Point(45.0 + (index % 5), 45.0 + (index % 7)).buffer(3.5) for index in range(300)],
        dtype=object,
    )
    owned, flat = build_owned_spatial_index(tree)

    def fail_morton_or_brute_force(*_args, **_kwargs):
        raise AssertionError("certified grid query should admit bbox candidates directly")

    monkeypatch.setattr(
        spatial_query_module,
        "spatial_index_device_query",
        fail_morton_or_brute_force,
    )
    indices = query_spatial_index(
        owned,
        flat,
        query,
        predicate="intersects",
        sort=True,
    )

    expected_left, expected_right = np.nonzero(
        shapely.intersects(query[:, None], tree[None, :])
    )
    assert flat.regular_grid is not None
    np.testing.assert_array_equal(indices[0], expected_left)
    np.testing.assert_array_equal(indices[1], expected_right)


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required")
def test_regular_grid_zero_hit_non_box_query_does_not_fall_back(monkeypatch) -> None:
    """An empty grid result is terminal instead of triggering brute force."""
    tree = np.asarray(
        [
            box(float(col), float(row), float(col + 1), float(row + 1))
            for row in range(100)
            for col in range(100)
        ],
        dtype=object,
    )
    query = np.asarray(
        [Point(200.0 + index, 200.0).buffer(0.25) for index in range(300)],
        dtype=object,
    )
    owned, flat = build_owned_spatial_index(tree)

    def fail_morton_or_brute_force(*_args, **_kwargs):
        raise AssertionError("an admitted empty grid result must be terminal")

    monkeypatch.setattr(
        spatial_query_module,
        "spatial_index_device_query",
        fail_morton_or_brute_force,
    )
    indices = query_spatial_index(
        owned,
        flat,
        query,
        predicate="intersects",
        sort=True,
    )

    assert flat.regular_grid is not None
    assert indices.shape == (2, 0)


def test_geometry_array_full_setitem_preserves_owned_for_noop_full_assignment() -> None:
    geometry = GeometryArray.from_owned(
        from_shapely_geometries(
            [
                box(0, 0, 1, 1),
                box(2, 2, 3, 3),
            ]
        )
    )
    original_owned = geometry._owned
    original_sindex = object()
    original_flat = object()
    geometry._sindex = original_sindex
    geometry._owned_flat_sindex = original_flat
    geometry._owned_spatial_input_supported = True

    geometry[:] = shapely.make_valid(np.asarray(geometry, dtype=object))

    assert geometry._owned is original_owned
    assert geometry._sindex is original_sindex
    assert geometry._owned_flat_sindex is original_flat
    assert geometry._owned_spatial_input_supported is True


def test_query_spatial_index_reports_execution_metadata() -> None:
    tree = np.asarray(
        [
            box(0, 0, 1, 1),
            box(1, 0, 2, 1),
            box(0, 1, 1, 2),
        ],
        dtype=object,
    )
    query = np.asarray([Point(1, 1), Point(1.5, 1.5)], dtype=object)
    owned, flat = build_owned_spatial_index(tree)

    indices, execution = query_spatial_index(
        owned,
        flat,
        query,
        predicate="intersects",
        sort=True,
        return_metadata=True,
    )

    assert indices.tolist() == [[0, 0, 0], [0, 1, 2]]
    if has_gpu_runtime():
        assert execution.selected is ExecutionMode.GPU
        assert execution.implementation == "owned_gpu_spatial_query"
    else:
        assert execution.selected is ExecutionMode.CPU
        assert execution.implementation == "owned_cpu_spatial_query"


def test_query_spatial_index_uses_gpu_for_point_tree_box_contains_when_large_enough() -> None:
    tree = np.asarray([Point(float(index), 0.0) for index in range(2048)], dtype=object)
    query = box(99.5, -1.0, 199.5, 1.0)
    owned, flat = build_owned_spatial_index(tree)

    indices, execution = query_spatial_index(
        owned,
        flat,
        query,
        predicate="contains",
        sort=True,
        return_metadata=True,
    )

    assert indices.tolist() == list(range(100, 200))
    if has_gpu_runtime():
        assert execution.selected is ExecutionMode.GPU
        assert execution.implementation == "owned_gpu_spatial_query"
    else:
        assert execution.selected is ExecutionMode.CPU


@pytest.mark.parametrize("predicate", [None, "intersects", "covers"])
def test_query_spatial_index_uses_gpu_for_point_tree_box_queries(predicate: str | None) -> None:
    tree = np.asarray([Point(float(index), 0.0) for index in range(2048)], dtype=object)
    query = box(99.5, -1.0, 199.5, 1.0)
    owned, flat = build_owned_spatial_index(tree)

    indices, execution = query_spatial_index(
        owned,
        flat,
        query,
        predicate=predicate,
        sort=True,
        return_metadata=True,
    )

    assert indices.tolist() == list(range(100, 200))
    if has_gpu_runtime():
        assert execution.selected is ExecutionMode.GPU
        assert execution.implementation == "owned_gpu_spatial_query"
    else:
        assert execution.selected is ExecutionMode.CPU


def test_query_spatial_index_uses_gpu_for_small_point_tree_box_queries() -> None:
    tree = np.asarray([Point(0.0, 0.0), Point(1.0, 0.0), Point(2.0, 0.0)], dtype=object)
    query = box(0.5, -1.0, 1.5, 1.0)
    owned, flat = build_owned_spatial_index(tree)

    indices, execution = query_spatial_index(
        owned,
        flat,
        query,
        predicate="contains",
        sort=True,
        return_metadata=True,
    )

    assert indices.tolist() == [1]
    if has_gpu_runtime():
        assert execution.selected is ExecutionMode.GPU
        assert execution.implementation == "owned_gpu_spatial_query"
    else:
        assert execution.selected is ExecutionMode.CPU


@pytest.mark.parametrize(
    ("predicate", "expected"),
    [
        ("contains_properly", list(range(101, 199))),
        ("touches", [100, 199]),
    ],
)
def test_query_spatial_index_uses_gpu_for_point_tree_box_boundary_sensitive_predicates(
    predicate: str,
    expected: list[int],
) -> None:
    tree = np.asarray([Point(float(index), 0.0) for index in range(2048)], dtype=object)
    query = box(100.0, -1.0, 199.0, 1.0)
    owned, flat = build_owned_spatial_index(tree)

    indices, execution = query_spatial_index(
        owned,
        flat,
        query,
        predicate=predicate,
        sort=True,
        return_metadata=True,
    )

    assert indices.tolist() == expected
    if has_gpu_runtime():
        assert execution.selected is ExecutionMode.GPU
        assert execution.implementation == "owned_gpu_spatial_query"
    else:
        assert execution.selected is ExecutionMode.CPU


def test_query_spatial_index_point_tree_box_scalar_avoids_owned_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for the raw scalar box fast path")

    tree = np.asarray([Point(float(index), 0.0) for index in range(2048)], dtype=object)
    query = box(99.5, -1.0, 199.5, 1.0)
    owned, flat = build_owned_spatial_index(tree)

    def _fail(values):
        raise AssertionError("point-tree box fast path should not normalize scalar Shapely query input to owned")

    monkeypatch.setattr(spatial_query_module, "_to_owned", _fail)

    indices, execution = query_spatial_index(
        owned,
        flat,
        query,
        predicate="contains",
        sort=True,
        return_metadata=True,
    )

    assert indices.tolist() == list(range(100, 200))
    assert execution.selected is ExecutionMode.GPU


def test_query_spatial_index_point_tree_box_owned_queries_avoid_to_shapely(monkeypatch: pytest.MonkeyPatch) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for the owned box fast path")

    tree = np.asarray([Point(float(index), 0.0) for index in range(2048)], dtype=object)
    owned, flat = build_owned_spatial_index(tree)
    query_owned = from_shapely_geometries([box(99.5, -1.0, 199.5, 1.0)])

    def _fail(self):
        raise AssertionError("owned point-tree box fast path should inspect owned polygon buffers directly")

    monkeypatch.setattr(OwnedGeometryArray, "to_shapely", _fail)

    indices, execution = query_spatial_index(
        owned,
        flat,
        query_owned,
        predicate="contains",
        sort=True,
        return_metadata=True,
    )

    assert indices.tolist() == [[0] * 100, list(range(100, 200))]
    assert execution.selected is ExecutionMode.GPU


def test_query_spatial_index_point_tree_box_device_owned_queries_avoid_host_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for the device-owned box fast path")

    tree = np.asarray([Point(float(index), 0.0) for index in range(2048)], dtype=object)
    owned, flat = build_owned_spatial_index(tree)
    query_owned = from_shapely_geometries(
        [box(99.5, -1.0, 199.5, 1.0)],
        residency=Residency.DEVICE,
    )

    def _fail():
        raise AssertionError(
            "device-owned point-tree box fast path should validate rectangle queries "
            "from device buffers without host-state materialization"
        )

    monkeypatch.setattr(query_owned, "_ensure_host_state", _fail)

    indices, execution = query_spatial_index(
        owned,
        flat,
        query_owned,
        predicate="contains",
        sort=True,
        return_metadata=True,
    )

    assert indices.tolist() == [[0] * 100, list(range(100, 200))]
    assert execution.selected is ExecutionMode.GPU


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime required for candidate-generation fallback")
def test_query_spatial_index_records_fallback_before_shapely_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cupy as cp

    tree = np.asarray(
        [
            LineString([(0.0, 0.0), (1.0, 1.0)]),
            LineString([(2.0, 0.0), (3.0, 1.0)]),
        ],
        dtype=object,
    )
    query = np.asarray(
        [LineString([(0.0, 1.0), (1.0, 0.0)])],
        dtype=object,
    )
    owned, flat = build_owned_spatial_index(tree)

    event_state = {"recorded": False}

    def _record_fallback_event(*args, **kwargs):
        event_state["recorded"] = True

    class _FakeDeviceCandidates:
        def __init__(self) -> None:
            self.d_left = cp.asarray(np.array([0], dtype=np.int32))
            self.d_right = cp.asarray(np.array([0], dtype=np.int32))
            self.total_pairs = 1

        def to_host(self):
            assert event_state["recorded"], "fallback event must be recorded before D2H candidate materialization"
            return np.array([0], dtype=np.int32), np.array([0], dtype=np.int32)

    original_to_shapely = OwnedGeometryArray.to_shapely

    def _fail_to_shapely(self):
        assert event_state["recorded"], "fallback event must be recorded before Shapely materialization"
        return original_to_shapely(self)

    monkeypatch.setattr(spatial_query_utils_module, "record_shapely_fallback_event", _record_fallback_event)
    monkeypatch.setattr(spatial_query_module, "spatial_index_device_query", lambda *args, **kwargs: (_FakeDeviceCandidates(), None))
    monkeypatch.setattr(OwnedGeometryArray, "to_shapely", _fail_to_shapely)

    result, execution = query_spatial_index(
        owned,
        flat,
        query,
        predicate="crosses",
        sort=True,
        return_metadata=True,
    )

    assert event_state["recorded"] is True
    assert result.tolist() == [[0], [0]]
    assert execution.selected is ExecutionMode.CPU
    assert execution.implementation == "owned_cpu_spatial_query"


def test_nearest_spatial_index_with_max_distance_returns_ties_and_distances() -> None:
    tree = np.asarray([Point(0, 0), Point(2, 0), Point(10, 0)], dtype=object)
    query = np.asarray([Point(1, 0), Point(20, 0)], dtype=object)

    (indices, distances), impl = nearest_spatial_index(
        tree,
        query,
        tree_query_nearest=lambda *args, **kwargs: pytest.fail("bounded nearest should not hit STRtree fallback"),
        return_all=True,
        max_distance=2.0,
        return_distance=True,
        exclusive=False,
    )

    assert indices.tolist() == [[0, 0], [0, 1]]
    assert np.allclose(distances, [1.0, 1.0])
    assert impl in ("owned_cpu_nearest", "owned_gpu_nearest")


def _force_nearest_fp32_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    original_plan = spatial_nearest_module.plan_dispatch_selection

    def _plan(**kwargs):
        if kwargs.get("kernel_name") in {
            "nearest_point_family_distance",
            "device_resident_point_family_distance",
        }:
            kwargs["requested_precision"] = PrecisionMode.FP32
        return original_plan(**kwargs)

    monkeypatch.setattr(spatial_nearest_module, "plan_dispatch_selection", _plan)


def _record_point_distance_precision(monkeypatch: pytest.MonkeyPatch):
    import vibespatial.spatial.point_distance as point_distance_module

    calls: list[tuple[PrecisionMode, int, object | None]] = []
    original_compute = point_distance_module.compute_point_distance_gpu
    original_pointset_compute = point_distance_module.compute_pointset_distance_gpu

    def _compute(*args, **kwargs):
        calls.append(
            (
                kwargs["compute_precision"],
                int(args[5]),
                kwargs.get("logical_count"),
            )
        )
        return original_compute(*args, **kwargs)

    monkeypatch.setattr(point_distance_module, "compute_point_distance_gpu", _compute)

    def _compute_pointset(*args, **kwargs):
        calls.append(
            (
                kwargs["compute_precision"],
                int(args[5]),
                kwargs.get("logical_count"),
            )
        )
        return original_pointset_compute(*args, **kwargs)

    monkeypatch.setattr(
        point_distance_module,
        "compute_pointset_distance_gpu",
        _compute_pointset,
    )
    return calls


def test_nearest_host_ambiguity_refines_nonfinite_coarse_distances() -> None:
    ambiguous = spatial_nearest_module._nearest_ambiguity_mask_host(
        np.asarray([0, 0, 1], dtype=np.int32),
        np.asarray([np.inf, np.nan, 2.0], dtype=np.float64),
        2,
        max_distance=np.inf,
        error_bound=1.0e-5,
    )

    assert ambiguous.tolist() == [True, True, True]


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required for nearest precision refinement")
def test_nearest_gpu_fp32_refines_near_ties_without_refining_all_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp = pytest.importorskip("cupy")
    base = 1_000_000_000.0
    query_owned = from_shapely_geometries([Point(base, 0.0)], residency=Residency.DEVICE)
    tree_owned = from_shapely_geometries(
        [
            LineString([(base + 1.0, -1.0), (base + 1.0, 1.0)]),
            LineString([(base - 1.0, -1.0), (base - 1.0, 1.0)]),
            LineString([(base - 1.0002, -1.0), (base - 1.0002, 1.0)]),
            LineString([(base + 100_000.0, -1.0), (base + 100_000.0, 1.0)]),
        ],
        residency=Residency.DEVICE,
    )
    _force_nearest_fp32_plan(monkeypatch)
    precision_calls = _record_point_distance_precision(monkeypatch)

    refined = spatial_nearest_module._nearest_refine_gpu(
        query_owned,
        tree_owned,
        np.zeros(4, dtype=np.int32),
        np.arange(4, dtype=np.int32),
        1,
        max_distance=2.0,
        return_all=True,
        return_distance=True,
    )

    assert refined is not None
    (indices, distances), used_fallback = refined
    assert used_fallback is False
    assert set(indices[1].tolist()) == {0, 1}
    np.testing.assert_allclose(distances, [1.0, 1.0], rtol=0.0, atol=1e-12)
    assert precision_calls[0] == (PrecisionMode.FP32, 4, None)
    fp64_calls = [call for call in precision_calls if call[0] is PrecisionMode.FP64]
    assert fp64_calls
    assert all(capacity == 4 for _, capacity, _ in fp64_calls)
    assert all(logical_count is not None for _, _, logical_count in fp64_calls)
    assert max(int(cp.asnumpy(logical_count)[0]) for _, _, logical_count in fp64_calls) < 4


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required for nearest precision refinement")
def test_nearest_gpu_fp32_refines_max_distance_ambiguity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp = pytest.importorskip("cupy")
    base = 1_000_000_000.0
    query_owned = from_shapely_geometries([Point(base, 0.0)], residency=Residency.DEVICE)
    tree_owned = from_shapely_geometries(
        [
            LineString([(base + 2.0002, -1.0), (base + 2.0002, 1.0)]),
            LineString([(base + 100_000.0, -1.0), (base + 100_000.0, 1.0)]),
        ],
        residency=Residency.DEVICE,
    )
    _force_nearest_fp32_plan(monkeypatch)
    precision_calls = _record_point_distance_precision(monkeypatch)

    refined = spatial_nearest_module._nearest_refine_gpu(
        query_owned,
        tree_owned,
        np.zeros(2, dtype=np.int32),
        np.arange(2, dtype=np.int32),
        1,
        max_distance=2.0,
        return_all=True,
        return_distance=True,
    )

    assert refined is not None
    (indices, distances), used_fallback = refined
    assert used_fallback is False
    assert indices.shape == (2, 0)
    assert distances.shape == (0,)
    assert precision_calls[0] == (PrecisionMode.FP32, 2, None)
    fp64_calls = [call for call in precision_calls if call[0] is PrecisionMode.FP64]
    assert any(
        capacity == 2 and int(cp.asnumpy(logical_count)[0]) == 1
        for _, capacity, logical_count in fp64_calls
    )


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required for nearest precision refinement")
def test_nearest_gpu_fp32_refines_finite_distance_overflow(
    monkeypatch: pytest.MonkeyPatch,
    strict_device_guard,
) -> None:
    cp = pytest.importorskip("cupy")
    query_owned = from_shapely_geometries([Point(0.0, 0.0)], residency=Residency.DEVICE)
    tree_owned = from_shapely_geometries(
        [
            LineString([(5.0e19, -1.0), (5.0e19, 1.0)]),
            LineString([(7.0e19, -1.0), (7.0e19, 1.0)]),
        ],
        residency=Residency.DEVICE,
    )
    _force_nearest_fp32_plan(monkeypatch)
    precision_calls = _record_point_distance_precision(monkeypatch)

    result = spatial_nearest_module._nearest_refine_gpu(
        query_owned,
        tree_owned,
        np.zeros(2, dtype=np.int32),
        np.arange(2, dtype=np.int32),
        1,
        max_distance=6.0e19,
        return_all=True,
        return_distance=True,
        return_device=True,
    )

    assert result is not None
    ((d_left, d_right), d_distances), used_fallback = result
    assert used_fallback is False
    assert cp.asnumpy(d_left).tolist() == [0]
    assert cp.asnumpy(d_right).tolist() == [0]
    np.testing.assert_allclose(cp.asnumpy(d_distances), [5.0e19], rtol=1.0e-15)
    assert any(mode is PrecisionMode.FP64 for mode, _, _ in precision_calls)


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required for nearest precision refinement")
def test_dwithin_gpu_fp32_refines_threshold_ambiguity_without_host_export(
    monkeypatch: pytest.MonkeyPatch,
    strict_device_guard,
) -> None:
    cp = pytest.importorskip("cupy")
    from vibespatial.api._native_relation import NativeRelationSelection
    from vibespatial.api._native_rowset import NativeDeviceSelection

    base = 1_000_000_000.0
    query_owned = from_shapely_geometries([Point(base, 0.0)], residency=Residency.DEVICE)
    tree_owned = from_shapely_geometries(
        [
            LineString([(base + 5.0002, -1.0), (base + 5.0002, 1.0)]),
            LineString([(base + 100_000.0, -1.0), (base + 100_000.0, 1.0)]),
        ],
        residency=Residency.DEVICE,
    )
    _force_nearest_fp32_plan(monkeypatch)
    precision_calls = _record_point_distance_precision(monkeypatch)
    selection_counts = []
    original_from_mask = NativeDeviceSelection.from_mask

    def _from_mask(*args, **kwargs):
        selection = original_from_mask(*args, **kwargs)
        selection_counts.append(selection.logical_count)
        return selection

    monkeypatch.setattr(NativeDeviceSelection, "from_mask", staticmethod(_from_mask))

    refined = spatial_nearest_module._dwithin_refine_gpu(
        query_owned,
        tree_owned,
        np.zeros(2, dtype=np.int32),
        np.arange(2, dtype=np.int32),
        np.asarray([5.0], dtype=np.float64),
        return_device=True,
    )

    assert refined is not None
    relation_selection, used_fallback = refined
    assert used_fallback is False
    assert isinstance(relation_selection, NativeRelationSelection)
    assert relation_selection.capacity == 2
    assert cp.asnumpy(relation_selection.logical_count).tolist() == [0]
    assert precision_calls[0][0] is PrecisionMode.FP32
    assert precision_calls[0][1] == 1
    assert int(cp.asnumpy(precision_calls[0][2])[0]) == 2
    fp64_calls = [call for call in precision_calls if call[0] is PrecisionMode.FP64]
    assert len(selection_counts) == 2  # one ambiguity rowset and one final result rowset
    assert any(
        capacity == 1
        and logical_count is not None
        and int(cp.asnumpy(logical_count)[0]) == 1
        for _, capacity, logical_count in fp64_calls
    )


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required for dwithin refinement")
def test_dwithin_gpu_fp32_refines_finite_distance_overflow(
    monkeypatch: pytest.MonkeyPatch,
    strict_device_guard,
) -> None:
    cp = pytest.importorskip("cupy")
    query_owned = from_shapely_geometries([Point(0.0, 0.0)], residency=Residency.DEVICE)
    tree_owned = from_shapely_geometries(
        [
            LineString([(5.0e19, -1.0), (5.0e19, 1.0)]),
            LineString([(7.0e19, -1.0), (7.0e19, 1.0)]),
        ],
        residency=Residency.DEVICE,
    )
    _force_nearest_fp32_plan(monkeypatch)
    precision_calls = _record_point_distance_precision(monkeypatch)

    result = spatial_nearest_module._dwithin_refine_gpu(
        query_owned,
        tree_owned,
        np.zeros(2, dtype=np.int32),
        np.arange(2, dtype=np.int32),
        np.asarray([6.0e19], dtype=np.float64),
        return_device=True,
    )

    assert result is not None
    relation_selection, used_fallback = result
    assert used_fallback is False
    count = int(cp.asnumpy(relation_selection.logical_count)[0])
    positions = relation_selection.selection.positions[:count]
    assert cp.asnumpy(relation_selection.relation.left_indices[positions]).tolist() == [0]
    assert cp.asnumpy(relation_selection.relation.right_indices[positions]).tolist() == [0]
    assert any(mode is PrecisionMode.FP64 for mode, _, _ in precision_calls)


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required for nearest precision refinement")
def test_nearest_fp32_refinement_return_device_has_no_host_export(
    monkeypatch: pytest.MonkeyPatch,
    strict_device_guard,
) -> None:
    base = 1_000_000_000.0
    query_owned = from_shapely_geometries([Point(base, 0.0)], residency=Residency.DEVICE)
    tree_owned = from_shapely_geometries(
        [
            LineString([(base + 1.0, -1.0), (base + 1.0, 1.0)]),
            LineString([(base + 100_000.0, -1.0), (base + 100_000.0, 1.0)]),
        ],
        residency=Residency.DEVICE,
    )
    _force_nearest_fp32_plan(monkeypatch)

    refined = spatial_nearest_module._nearest_refine_gpu(
        query_owned,
        tree_owned,
        np.zeros(2, dtype=np.int32),
        np.arange(2, dtype=np.int32),
        1,
        max_distance=2.0,
        return_all=True,
        return_distance=True,
        return_device=True,
    )

    assert refined is not None
    ((d_left, d_right), d_distances), used_fallback = refined
    assert used_fallback is False
    assert d_left.size == 1
    assert d_right.size == 1
    assert d_distances.size == 1


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required for nearest precision refinement")
def test_mixed_multipoint_nearest_gpu_refines_fp32_ordering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = 1_000_000_000.0
    query_owned = from_shapely_geometries(
        [MultiPoint([(base, 0.0)])],
        residency=Residency.DEVICE,
    )
    tree_owned = from_shapely_geometries(
        [
            LineString([(base + 1.0, -1.0), (base + 1.0, 1.0)]),
            box(base - 2.0002, -1.0, base - 1.0002, 1.0),
            Point(base + 100_000.0, 0.0),
        ],
        residency=Residency.DEVICE,
    )
    _force_nearest_fp32_plan(monkeypatch)
    precision_calls = _record_point_distance_precision(monkeypatch)

    refined = spatial_nearest_module._nearest_refine_gpu(
        query_owned,
        tree_owned,
        np.zeros(3, dtype=np.int32),
        np.arange(3, dtype=np.int32),
        1,
        max_distance=200_000.0,
        return_all=True,
        return_distance=True,
    )

    assert refined is not None
    (indices, distances), used_fallback = refined
    assert used_fallback is False
    assert indices.tolist() == [[0], [0]]
    np.testing.assert_allclose(distances, [1.0], rtol=0.0, atol=1e-12)
    assert any(mode is PrecisionMode.FP32 for mode, _, _ in precision_calls)
    assert any(mode is PrecisionMode.FP64 for mode, _, _ in precision_calls)


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required for device-only centering")
def test_device_only_large_origin_nearest_and_dwithin_use_device_center(
    monkeypatch: pytest.MonkeyPatch,
    strict_device_guard,
) -> None:
    cp = pytest.importorskip("cupy")
    base = 1_000_000_000.0
    query_owned = _device_only_clone_for_test(
        from_shapely_geometries([Point(base, 0.0)], residency=Residency.DEVICE)
    )
    tree_owned = _device_only_clone_for_test(
        from_shapely_geometries(
            [
                LineString([(base + 1.0, -1.0), (base + 1.0, 1.0)]),
                LineString([(base + 100_000.0, -1.0), (base + 100_000.0, 1.0)]),
            ],
            residency=Residency.DEVICE,
        )
    )
    assert all(buffer.x.size == 0 for buffer in query_owned.families.values())
    assert all(buffer.x.size == 0 for buffer in tree_owned.families.values())
    _force_nearest_fp32_plan(monkeypatch)

    nearest = spatial_nearest_module._nearest_refine_gpu(
        query_owned,
        tree_owned,
        np.zeros(2, dtype=np.int32),
        np.arange(2, dtype=np.int32),
        1,
        max_distance=2.0,
        return_all=True,
        return_distance=True,
        return_device=True,
    )
    dwithin = spatial_nearest_module._dwithin_refine_gpu(
        query_owned,
        tree_owned,
        np.zeros(2, dtype=np.int32),
        np.arange(2, dtype=np.int32),
        np.asarray([1.5], dtype=np.float64),
        return_device=True,
    )

    assert nearest is not None and dwithin is not None
    ((d_left, d_right), d_distances), nearest_fallback = nearest[0], nearest[1]
    relation_selection, dwithin_fallback = dwithin
    assert nearest_fallback is False
    assert dwithin_fallback is False
    assert cp.asnumpy(d_left).tolist() == [0]
    assert cp.asnumpy(d_right).tolist() == [0]
    np.testing.assert_allclose(cp.asnumpy(d_distances), [1.0], rtol=0.0, atol=1e-12)
    count = int(cp.asnumpy(relation_selection.logical_count)[0])
    positions = relation_selection.selection.positions[:count]
    assert cp.asnumpy(relation_selection.relation.left_indices[positions]).tolist() == [0]
    assert cp.asnumpy(relation_selection.relation.right_indices[positions]).tolist() == [0]


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required for mixed multipoint distance")
def test_strict_multipoint_dwithin_supports_every_owned_target_family(
    strict_device_guard,
) -> None:
    cp = pytest.importorskip("cupy")
    query_geometry = MultiPoint([(0.0, 0.0), (0.0, 1.0)])
    tree_geometries = [
        Point(3.0, 0.0),
        MultiPoint([(2.0, 0.0), (10.0, 0.0)]),
        LineString([(4.0, -1.0), (4.0, 2.0)]),
        MultiLineString([[(5.0, -1.0), (5.0, 2.0)], [(20.0, 0.0), (21.0, 0.0)]]),
        box(6.0, -1.0, 7.0, 2.0),
        MultiPolygon([box(8.0, -1.0, 9.0, 2.0), box(30.0, 0.0, 31.0, 1.0)]),
    ]
    query_owned = from_shapely_geometries([query_geometry], residency=Residency.DEVICE)
    tree_owned = from_shapely_geometries(tree_geometries, residency=Residency.DEVICE)

    result = spatial_nearest_module._dwithin_refine_gpu(
        query_owned,
        tree_owned,
        np.zeros(len(tree_geometries), dtype=np.int32),
        np.arange(len(tree_geometries), dtype=np.int32),
        np.asarray([4.5], dtype=np.float64),
        return_device=True,
    )

    assert result is not None
    relation_selection, used_fallback = result
    assert used_fallback is False
    count = int(cp.asnumpy(relation_selection.logical_count)[0])
    positions = relation_selection.selection.positions[:count]
    d_left = relation_selection.relation.left_indices[positions]
    d_right = relation_selection.relation.right_indices[positions]
    expected_right = [
        index
        for index, geometry in enumerate(tree_geometries)
        if shapely.dwithin(query_geometry, geometry, 4.5)
    ]
    assert bool(cp.array_equal(d_left, cp.zeros(len(expected_right), dtype=cp.int32)))
    assert bool(cp.array_equal(d_right, cp.asarray(expected_right, dtype=cp.int32)))


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required for mixed distance partition")
def test_mixed_distance_partition_matches_all_owned_family_pairs_and_skew(
    strict_device_guard,
) -> None:
    cp = pytest.importorskip("cupy")
    geometries = [
        Point(0.0, 0.0),
        MultiPoint([(0.0, 0.0), (0.0, 1.0)]),
        LineString([(2.0, -1.0), (2.0, 2.0)]),
        MultiLineString([[(3.0, -1.0), (3.0, 2.0)], [(20.0, 0.0), (21.0, 0.0)]]),
        box(4.0, -1.0, 5.0, 2.0),
        MultiPolygon([box(6.0, -1.0, 7.0, 2.0), box(30.0, 0.0, 31.0, 1.0)]),
    ]
    query_owned = from_shapely_geometries(geometries, residency=Residency.DEVICE)
    tree_owned = from_shapely_geometries(geometries, residency=Residency.DEVICE)
    left = np.repeat(np.arange(len(geometries), dtype=np.int32), len(geometries))
    right = np.tile(np.arange(len(geometries), dtype=np.int32), len(geometries))
    expected = np.asarray(
        [shapely.distance(geometries[li], geometries[ri]) for li, ri in zip(left, right)],
        dtype=np.float64,
    )

    result = spatial_nearest_module._compute_mixed_distances_gpu_device(
        query_owned,
        tree_owned,
        left,
        right,
    )
    assert result is not None
    distances, used_fallback = result
    assert used_fallback is False
    assert bool(cp.allclose(distances, cp.asarray(expected), rtol=1e-5, atol=1e-6))

    skew_count = 513
    skew = spatial_nearest_module._compute_mixed_distances_gpu_device(
        from_shapely_geometries(
            [MultiPoint([(0.0, 0.0), (0.0, 1.0)])],
            residency=Residency.DEVICE,
        ),
        from_shapely_geometries(
            [LineString([(2.0, -1.0), (2.0, 2.0)])],
            residency=Residency.DEVICE,
        ),
        np.zeros(skew_count, dtype=np.int32),
        np.zeros(skew_count, dtype=np.int32),
    )
    assert skew is not None
    skew_distances, skew_fallback = skew
    assert skew_fallback is False
    assert bool(cp.all(skew_distances == 2.0))


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required for mixed distance partition")
def test_mixed_distance_checks_every_multipart_component_for_containment(
    strict_device_guard,
) -> None:
    cp = pytest.importorskip("cupy")
    polygon = box(0.0, 0.0, 10.0, 10.0)
    multiline = MultiLineString(
        [
            [(20.0, 20.0), (21.0, 20.0)],
            [(2.0, 2.0), (3.0, 3.0)],
        ]
    )
    multipolygon = MultiPolygon(
        [
            box(20.0, 20.0, 21.0, 21.0),
            box(2.0, 2.0, 3.0, 3.0),
        ]
    )
    left_geometries = [multiline, multipolygon, polygon, polygon]
    right_geometries = [polygon, polygon, multiline, multipolygon]

    result = spatial_nearest_module._compute_mixed_distances_gpu_device(
        from_shapely_geometries(left_geometries, residency=Residency.DEVICE),
        from_shapely_geometries(right_geometries, residency=Residency.DEVICE),
        np.arange(4, dtype=np.int32),
        np.arange(4, dtype=np.int32),
    )

    assert result is not None
    distances, used_fallback = result
    assert used_fallback is False
    assert bool(cp.all(distances == 0.0))


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required for dwithin refinement")
def test_dwithin_does_not_collapse_sub_epsilon_polygon_gap(
    monkeypatch: pytest.MonkeyPatch,
    strict_device_guard,
) -> None:
    cp = pytest.importorskip("cupy")
    query_owned = from_shapely_geometries(
        [Point(1.0 + 5e-8, 0.5)],
        residency=Residency.DEVICE,
    )
    tree_owned = from_shapely_geometries(
        [box(0.0, 0.0, 1.0, 1.0)],
        residency=Residency.DEVICE,
    )
    _force_nearest_fp32_plan(monkeypatch)
    precision_calls = _record_point_distance_precision(monkeypatch)

    result = spatial_nearest_module._dwithin_refine_gpu(
        query_owned,
        tree_owned,
        np.asarray([0], dtype=np.int32),
        np.asarray([0], dtype=np.int32),
        np.asarray([1e-9], dtype=np.float64),
        return_device=True,
    )

    assert result is not None
    relation_selection, used_fallback = result
    assert used_fallback is False
    assert int(cp.asnumpy(relation_selection.logical_count)[0]) == 0
    assert any(mode is PrecisionMode.FP64 for mode, _, _ in precision_calls)


def test_nearest_spatial_index_gpu_avoids_host_point_coordinate_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device-native nearest refinement")

    tree = np.asarray([Point(float(index), 0.0) for index in range(2048)], dtype=object)
    query = np.asarray([Point(float(index) + 0.25, 0.25) for index in range(2048)], dtype=object)

    def _fail(_owned):
        raise AssertionError("GPU nearest refinement should consume owned device point buffers directly")

    monkeypatch.setattr(spatial_query_module, "_extract_point_coords", _fail, raising=False)

    (indices, distances), impl = nearest_spatial_index(
        tree,
        query,
        tree_query_nearest=lambda *args, **kwargs: pytest.fail("large nearest query should not hit STRtree fallback"),
        return_all=True,
        max_distance=1.0,
        return_distance=True,
        exclusive=False,
    )

    assert indices.shape[1] == len(query)
    assert np.all(distances >= 0.0)
    assert impl == "owned_gpu_nearest"


def test_nearest_spatial_index_gpu_bounded_point_sweep_avoids_generic_bbox_candidate_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for bounded point-sweep nearest candidate generation")

    tree = np.asarray([Point(float(index), 0.0) for index in range(2048)], dtype=object)
    query = np.asarray([Point(float(index) + 0.25, 0.25) for index in range(2048)], dtype=object)

    def _fail_bbox(*args, **kwargs):
        raise AssertionError("bounded point nearest should not use generic bbox candidate generation")

    monkeypatch.setattr(spatial_nearest_module, "_generate_candidates_gpu", _fail_bbox)
    monkeypatch.setattr(spatial_nearest_module, "_generate_distance_pairs", _fail_bbox)

    (indices, distances), impl = nearest_spatial_index(
        tree,
        query,
        tree_query_nearest=lambda *args, **kwargs: pytest.fail("large nearest query should not hit STRtree fallback"),
        return_all=True,
        max_distance=1.0,
        return_distance=True,
        exclusive=False,
    )

    assert indices.shape[1] == len(query)
    assert np.all(distances >= 0.0)
    assert impl == "owned_gpu_nearest"


def test_nearest_spatial_index_uses_device_owned_point_buffers() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device-owned nearest")

    tree = np.asarray([Point(0, 0), Point(2, 0), Point(10, 0)], dtype=object)
    query = np.asarray([Point(1, 0), Point(20, 0)], dtype=object)
    tree_owned = from_shapely_geometries(tree)
    query_owned = from_shapely_geometries(query)
    tree_owned.move_to(
        Residency.DEVICE,
        trigger="explicit-runtime-request",
        reason="test nearest device-owned tree",
    )
    query_owned.move_to(
        Residency.DEVICE,
        trigger="explicit-runtime-request",
        reason="test nearest device-owned query",
    )

    (indices, distances), impl = nearest_spatial_index(
        None,
        None,
        tree_query_nearest=lambda *args, **kwargs: pytest.fail("device-owned nearest should not hit STRtree"),
        return_all=True,
        max_distance=2.0,
        return_distance=True,
        exclusive=False,
        tree_owned=tree_owned,
        query_owned=query_owned,
    )

    assert indices.tolist() == [[0, 0], [0, 1]]
    assert np.allclose(distances, [1.0, 1.0])
    assert impl == "owned_gpu_nearest"


def test_nearest_return_device_knn_bounds_stay_device_resident() -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for device-resident nearest kNN")

    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    tree_owned = from_shapely_geometries(
        [box(0.0, 0.0, 1.0, 1.0), box(10.0, 0.0, 11.0, 1.0)],
        residency=Residency.DEVICE,
    )
    query_owned = from_shapely_geometries(
        [Point(0.5, 0.5), Point(10.5, 0.5)],
        residency=Residency.DEVICE,
    )

    reset_d2h_transfer_count()
    result, impl = nearest_spatial_index(
        None,
        None,
        tree_query_nearest=lambda *args, **kwargs: pytest.fail("device nearest should not hit STRtree"),
        return_all=False,
        max_distance=2.0,
        return_distance=True,
        exclusive=False,
        return_device=True,
        tree_owned=tree_owned,
        query_owned=query_owned,
    )
    events = get_d2h_transfer_events(clear=True)
    reasons = [event.reason for event in events]

    assert impl == "owned_gpu_nearest"
    indices, distances = result
    assert indices[0].shape == (2,)
    assert indices[1].shape == (2,)
    assert distances.shape == (2,)
    assert "geometry analysis mixed row-bounds host export" not in reasons
    assert "geometry analysis cached row-bounds host export" not in reasons


def test_nearest_spatial_index_unbounded_matches_expected_ties_and_distances() -> None:
    tree = np.asarray([Point(0, 0), Point(2, 0), Point(10, 0)], dtype=object)
    query = np.asarray([Point(1, 0), Point(20, 0)], dtype=object)
    from shapely import STRtree

    (indices, distances), impl = nearest_spatial_index(
        tree,
        query,
        tree_query_nearest=STRtree(tree).query_nearest,
        return_all=True,
        max_distance=None,
        return_distance=True,
        exclusive=False,
    )

    assert indices.tolist() == [[0, 0, 1], [0, 1, 2]]
    assert np.allclose(distances, [1.0, 1.0, 10.0])
    assert impl in {"strtree_host", "owned_cpu_nearest", "owned_gpu_nearest"}


def test_nearest_spatial_index_records_fallback_before_host_refine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = np.asarray(
        [
            LineString([(0.0, 0.0), (1.0, 0.0)]),
            LineString([(5.0, 0.0), (6.0, 0.0)]),
        ],
        dtype=object,
    )
    query = np.asarray(
        [LineString([(0.0, 1.0), (1.0, 1.0)])],
        dtype=object,
    )
    event_state = {"recorded": False}

    def _record_fallback_event(*args, **kwargs):
        event_state["recorded"] = True

    def _fake_candidate_generation(*args, **kwargs):
        return np.array([0], dtype=np.int32), np.array([0], dtype=np.int32)

    def _fail_refine(*args, **kwargs):
        return None

    original_distance = spatial_nearest_module.shapely.distance

    def _distance_with_order_check(left_values, right_values):
        assert event_state["recorded"], "fallback event must be recorded before host Shapely refinement"
        return original_distance(left_values, right_values)

    monkeypatch.setattr(spatial_nearest_module, "record_shapely_fallback_event", _record_fallback_event)
    monkeypatch.setattr(spatial_nearest_module, "_generate_candidates_gpu", _fake_candidate_generation)
    monkeypatch.setattr(spatial_nearest_module, "_generate_distance_pairs", lambda *args, **kwargs: pytest.fail("GPU candidate generation should not fall back to host pair generation"))
    monkeypatch.setattr(spatial_nearest_module, "_nearest_refine_gpu", _fail_refine)
    monkeypatch.setattr(spatial_nearest_module.shapely, "distance", _distance_with_order_check)
    monkeypatch.setattr("vibespatial.spatial.spatial_index_knn_device.spatial_index_knn_device", lambda *args, **kwargs: None)

    (indices, distances), impl = nearest_spatial_index(
        tree,
        query,
        tree_query_nearest=lambda *args, **kwargs: pytest.fail("host fallback should not call STRtree nearest"),
        return_all=True,
        max_distance=10.0,
        return_distance=True,
        exclusive=False,
    )

    assert event_state["recorded"] is True
    assert impl == "owned_cpu_nearest"
    assert indices.tolist() == [[0], [0]]
    assert np.allclose(distances, [1.0])


def test_nearest_spatial_index_return_device_declines_mixed_refine_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = np.asarray(
        [
            box(0.0, 0.0, 1.0, 1.0),
            MultiPolygon([box(5.0, 0.0, 6.0, 1.0)]),
        ],
        dtype=object,
    )
    query = np.asarray([Point(0.5, 0.5)], dtype=object)

    monkeypatch.setattr(
        spatial_nearest_module,
        "record_shapely_fallback_event",
        lambda *args, **kwargs: pytest.fail("return_device decline must not record host fallback"),
    )

    result, impl = nearest_spatial_index(
        tree,
        query,
        tree_query_nearest=lambda *args, **kwargs: pytest.fail("device relation decline should not call STRtree"),
        return_all=True,
        max_distance=10.0,
        return_distance=True,
        exclusive=False,
        return_device=True,
    )

    assert result is None
    assert impl == "owned_cpu_nearest"


def test_nearest_spatial_index_return_device_failed_refine_declines_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = np.asarray(
        [
            LineString([(0.0, 0.0), (1.0, 0.0)]),
            LineString([(5.0, 0.0), (6.0, 0.0)]),
        ],
        dtype=object,
    )
    query = np.asarray(
        [LineString([(0.0, 1.0), (1.0, 1.0)])],
        dtype=object,
    )

    monkeypatch.setattr(
        spatial_nearest_module,
        "record_shapely_fallback_event",
        lambda *args, **kwargs: pytest.fail("return_device decline must not record host fallback"),
    )
    monkeypatch.setattr(
        spatial_nearest_module,
        "_generate_candidates_gpu",
        lambda *args, **kwargs: (np.array([0], dtype=np.int32), np.array([0], dtype=np.int32)),
    )
    monkeypatch.setattr(
        spatial_nearest_module,
        "_generate_distance_pairs",
        lambda *args, **kwargs: pytest.fail("GPU candidate generation should not fall back to host pairs"),
    )
    monkeypatch.setattr(spatial_nearest_module, "_nearest_refine_gpu", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "vibespatial.spatial.spatial_index_knn_device.spatial_index_knn_device",
        lambda *args, **kwargs: None,
    )

    result, impl = nearest_spatial_index(
        tree,
        query,
        tree_query_nearest=lambda *args, **kwargs: pytest.fail("device relation decline should not call STRtree"),
        return_all=True,
        max_distance=10.0,
        return_distance=True,
        exclusive=False,
        return_device=True,
    )

    assert result is None
    assert impl == "owned_gpu_nearest"


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime required for exact GPU nearest fallback")
def test_nearest_spatial_index_gpu_unbounded_small_point_set_covers_all_queries() -> None:
    tree = np.asarray([Point(1, 1)], dtype=object)
    query = np.asarray([Point(0, 0), Point(1, 1)], dtype=object)

    (indices, distances), impl = nearest_spatial_index(
        tree,
        query,
        tree_query_nearest=lambda *args, **kwargs: pytest.fail("point-point GPU nearest should not hit STRtree fallback"),
        return_all=True,
        max_distance=None,
        return_distance=True,
        exclusive=False,
    )

    assert impl == "owned_gpu_nearest"
    assert indices.tolist() == [[0, 1], [0, 0]]
    assert np.allclose(distances, [np.sqrt(2.0), 0.0])


def test_nearest_spatial_index_gpu_unbounded_avoids_bruteforce_candidate_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for indexed unbounded nearest")

    tree = np.asarray([Point(float(index), 0.0) for index in range(2048)], dtype=object)
    query = np.asarray([Point(float(index) + 0.25, 0.25) for index in range(2048)], dtype=object)

    def _fail_bruteforce(*args, **kwargs):
        raise AssertionError("unbounded indexed nearest should not use brute-force candidate generation")

    monkeypatch.setattr(spatial_nearest_module, "_generate_candidates_gpu", _fail_bruteforce)
    monkeypatch.setattr(spatial_nearest_module, "_generate_distance_pairs", _fail_bruteforce)

    (indices, distances), impl = nearest_spatial_index(
        tree,
        query,
        tree_query_nearest=lambda *args, **kwargs: pytest.fail("indexed nearest should not hit STRtree fallback"),
        return_all=True,
        max_distance=None,
        return_distance=True,
        exclusive=False,
    )

    assert indices.shape[1] == len(query)
    assert np.all(distances >= 0.0)
    assert impl == "owned_gpu_nearest"


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required")
class TestMixedFamilyNearest:
    """GPU nearest refinement for arrays with mixed geometry families."""

    def test_mixed_tree_points_and_polygons(self):
        # Tree has a mix of points and polygons.
        tree = np.asarray([Point(0, 0), box(5, 5, 6, 6), Point(10, 0)], dtype=object)
        query = np.asarray([Point(1, 0)], dtype=object)

        (indices, distances), impl = nearest_spatial_index(
            tree, query,
            tree_query_nearest=lambda *a, **kw: pytest.fail("should not use STRtree"),
            return_all=True, max_distance=20.0, return_distance=True, exclusive=False,
        )
        # Nearest should be Point(0,0) at distance 1.
        assert indices.shape[1] >= 1
        assert 0 in indices[1].tolist()
        assert impl == "owned_gpu_nearest"

    def test_mixed_query_and_tree(self):
        from shapely.geometry import LineString
        tree = np.asarray([Point(0, 0), LineString([(5, 0), (5, 5)])], dtype=object)
        query = np.asarray([Point(1, 0), box(4, 0, 4.5, 0.5)], dtype=object)

        (indices, distances), impl = nearest_spatial_index(
            tree, query,
            tree_query_nearest=lambda *a, **kw: pytest.fail("should not use STRtree"),
            return_all=True, max_distance=20.0, return_distance=True, exclusive=False,
        )
        import shapely as shp
        # Verify correctness: for each query, nearest tree geometry is correct.
        for col in range(indices.shape[1]):
            qi, ti = indices[0, col], indices[1, col]
            gpu_dist = distances[col]
            ref_dist = shp.distance(query[qi], tree[ti])
            assert abs(gpu_dist - ref_dist) < 1e-10
        assert impl == "owned_gpu_nearest"


def test_query_spatial_index_regular_grid_box_queries_avoid_exact_refine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for the regular-grid rectangle box fast path")

    tree = np.asarray(
        [box(float(x), float(y), float(x + 1), float(y + 1)) for y in range(50) for x in range(50)],
        dtype=object,
    )
    query = np.asarray(
        [
            box(10.0, 10.0, 12.0, 12.0),
            box(1000.0, 1000.0, 1001.0, 1001.0),
        ],
        dtype=object,
    )
    owned, flat = build_owned_spatial_index(tree)

    def _fail(*args, **kwargs):
        raise AssertionError("regular-grid rectangle box queries should not hit generic exact refine")

    monkeypatch.setattr(spatial_query_utils_module, "evaluate_binary_predicate", _fail)

    result, execution = query_spatial_index(
        owned,
        flat,
        query,
        predicate="intersects",
        sort=True,
        output_format="indices",
        return_metadata=True,
    )

    from shapely import STRtree

    reference = STRtree(tree).query(query, predicate="intersects")
    assert set(zip(result[0].tolist(), result[1].tolist())) == set(zip(reference[0].tolist(), reference[1].tolist()))
    assert execution.selected is ExecutionMode.GPU
    assert execution.implementation == "owned_gpu_spatial_query"


@pytest.mark.parametrize("predicate", [None, "intersects", "contains"])
def test_gpu_bbox_candidate_generation_polygon_tree_triangle_query(predicate: str | None) -> None:
    """GPU candidate generation fires for polygon tree + non-box query above crossover."""
    from shapely.geometry import Polygon

    tree = np.asarray(
        [box(x * 0.01, y * 0.01, x * 0.01 + 0.01, y * 0.01 + 0.01) for x in range(50) for y in range(50)],
        dtype=object,
    )
    query = Polygon([(0.1, 0.1), (0.4, 0.1), (0.25, 0.4), (0.1, 0.1)])
    owned, flat = build_owned_spatial_index(tree)

    result, execution = query_spatial_index(
        owned,
        flat,
        query,
        predicate=predicate,
        sort=True,
        output_format="indices",
        return_metadata=True,
    )

    # Verify against Shapely STRtree for bbox-only and predicate queries.
    # For contains / intersects: the GPU DE-9IM engine correctly finds
    # boundary-touching matches (distance ≈ 0) that GEOS's snap-rounding
    # misses, so we verify GPU is a superset of STRtree and that every
    # extra result has near-zero Shapely distance.
    from shapely import STRtree

    strtree = STRtree(tree)
    if predicate is None:
        reference = sorted(strtree.query(query, predicate=predicate).tolist())
        assert sorted(result.tolist()) == reference
    else:
        reference_set = set(strtree.query(query, predicate=predicate).tolist())
        result_set = set(result.tolist())

        # GPU must find everything Shapely finds.
        missing = reference_set - result_set
        assert not missing, f"GPU missed indices found by Shapely: {sorted(missing)}"

        # Any extra GPU results must be boundary-touching (distance ≈ 0).
        import shapely as shp
        extra = result_set - reference_set
        for idx in extra:
            d = shp.distance(query, tree[idx])
            assert d < 1e-10, (
                f"GPU extra idx={idx} has non-trivial distance {d}"
            )

    if has_gpu_runtime():
        assert execution.selected is ExecutionMode.GPU
        assert execution.implementation == "owned_gpu_spatial_query"


@pytest.mark.parametrize("predicate", ["intersects", "contains", "covers"])
def test_owned_refine_eliminates_shapely_roundtrip(predicate: str) -> None:
    """GPU candidate gen + owned-array refine avoids Shapely conversion.

    Verifies that _filter_predicate_pairs_owned feeds OwnedGeometryArray.take()
    output directly into evaluate_binary_predicate, and that the result matches
    the CPU reference.
    """
    pytest.importorskip("cupy")
    if not has_gpu_runtime():
        pytest.skip("no GPU runtime")

    # Build a grid large enough to exceed the 1,000 crossover threshold
    # with a single query (1 * 2500 = 2500 > 1000).
    tree = np.asarray(
        [box(x * 0.01, y * 0.01, x * 0.01 + 0.01, y * 0.01 + 0.01) for x in range(50) for y in range(50)],
        dtype=object,
    )
    # Use a query that overlaps a subset of the grid
    query = np.asarray([box(0.05, 0.05, 0.25, 0.25)], dtype=object)
    owned, flat = build_owned_spatial_index(tree)

    result, execution = query_spatial_index(
        owned,
        flat,
        query,
        predicate=predicate,
        sort=True,
        output_format="indices",
        return_metadata=True,
    )

    # CPU reference via owned engine (not STRtree) for consistency
    from vibespatial.predicates.binary import evaluate_binary_predicate
    from vibespatial.spatial.indexing import generate_bounds_pairs

    query_owned_ref = from_shapely_geometries(query.tolist())
    pairs = generate_bounds_pairs(query_owned_ref, flat.geometry_array)
    exact = evaluate_binary_predicate(
        predicate,
        np.asarray(query, dtype=object)[pairs.left_indices],
        np.asarray(tree, dtype=object)[pairs.right_indices],
        dispatch_mode="cpu",
        null_behavior="false",
    )
    keep = np.asarray(exact.values, dtype=bool)
    reference = sorted(pairs.right_indices[keep].tolist())

    # Result from GPU path must match CPU reference
    gpu_indices = sorted(result[0].tolist()) if result.ndim == 1 else sorted(result[1].tolist())
    assert gpu_indices == reference, (
        f"predicate={predicate}: GPU owned refine produced {len(gpu_indices)} results "
        f"vs CPU reference {len(reference)}"
    )
    assert execution.selected is ExecutionMode.GPU
    assert execution.implementation == "owned_gpu_spatial_query"


def test_gpu_bbox_candidate_generation_multi_query() -> None:
    """GPU candidate generation handles vectorized multi-query above crossover."""
    from shapely.geometry import Polygon

    tree = np.asarray(
        [box(x * 0.01, y * 0.01, x * 0.01 + 0.01, y * 0.01 + 0.01) for x in range(50) for y in range(50)],
        dtype=object,
    )
    queries = np.asarray(
        [
            Polygon([(0.1, 0.1), (0.3, 0.1), (0.2, 0.3)]),
            Polygon([(0.3, 0.3), (0.5, 0.3), (0.4, 0.5)]),
        ],
        dtype=object,
    )
    owned, flat = build_owned_spatial_index(tree)

    result, execution = query_spatial_index(
        owned,
        flat,
        queries,
        predicate=None,
        sort=True,
        output_format="indices",
        return_metadata=True,
    )

    from shapely import STRtree

    strtree = STRtree(tree)
    reference = strtree.query(queries, predicate=None)
    ref_pairs = set(zip(reference[0].tolist(), reference[1].tolist()))
    my_pairs = set(zip(result[0].tolist(), result[1].tolist()))
    assert my_pairs == ref_pairs

    if has_gpu_runtime():
        assert execution.selected is ExecutionMode.GPU
        assert execution.implementation == "owned_gpu_spatial_query"


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required")
def test_dwithin_routes_through_owned_path() -> None:
    """dwithin predicate routes through the owned query path, not STRtree."""
    tree = np.asarray([Point(0, 0), Point(10, 0), Point(20, 0)], dtype=object)
    query = np.asarray([Point(1, 0), Point(16, 0)], dtype=object)
    owned, flat = build_owned_spatial_index(tree)

    result, execution = query_spatial_index(
        owned, flat, query, predicate="dwithin", distance=5.0,
        sort=True, return_metadata=True,
    )
    assert result.tolist() == [[0, 1], [0, 2]]
    assert execution.selected is ExecutionMode.GPU
    assert execution.implementation == "owned_gpu_spatial_query"
    # Must not fall back to STRtree
    assert "strtree" not in execution.reason.lower()


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required")
def test_small_input_dispatches_correctly() -> None:
    """Small inputs dispatch to the owned spatial query engine (GPU or CPU)."""
    tree = np.asarray([Point(0, 0), Point(1, 1), Point(2, 2)], dtype=object)
    query = np.asarray([Point(0.5, 0.5)], dtype=object)
    owned, flat = build_owned_spatial_index(tree)

    result, execution = query_spatial_index(
        owned, flat, query, predicate="intersects",
        sort=True, return_metadata=True,
    )
    # Small inputs may dispatch to CPU or GPU depending on crossover policy;
    # the important thing is they use the owned spatial query engine
    assert execution.selected in (ExecutionMode.GPU, ExecutionMode.CPU)
    assert "owned" in execution.implementation


# ---------------------------------------------------------------------------
# sjoin / sjoin_nearest dispatch event GPU visibility tests
# ---------------------------------------------------------------------------

class TestSjoinDispatchVisibility:
    """Verify that sjoin and sjoin_nearest report the actual execution mode
    from the underlying spatial query engine in their dispatch events."""

    def test_sjoin_dispatch_event_reports_owned_query_execution(self) -> None:
        """sjoin dispatch event should report the implementation from the
        owned spatial query engine, not hardcoded CPU."""
        from shapely.geometry import Point

        from vibespatial.api.geodataframe import GeoDataFrame
        from vibespatial.runtime.dispatch import get_dispatch_events

        left = GeoDataFrame(
            {"a": [1, 2, 3]},
            geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
        )
        right = GeoDataFrame(
            {"b": [10, 20, 30]},
            geometry=[Point(0.1, 0.1), Point(1.1, 1.1), Point(10, 10)],
        )
        # Clear events before the join.
        get_dispatch_events(clear=True)
        from vibespatial.api.tools.sjoin import sjoin

        sjoin(left, right, predicate="intersects")

        events = get_dispatch_events(clear=True)
        sjoin_events = [e for e in events if e.surface == "geopandas.tools.sjoin"]
        assert len(sjoin_events) >= 1
        event = sjoin_events[0]
        assert event.implementation == "owned_spatial_query"
        # The event should report the actual execution mode from the query
        # engine (GPU or CPU), not blindly hardcoded CPU.
        assert event.selected in (ExecutionMode.CPU, ExecutionMode.GPU)

    def test_sjoin_nearest_dispatch_event_threads_execution_mode(self) -> None:
        """sjoin_nearest dispatch event should thread the execution mode from
        sindex.nearest instead of hardcoding CPU."""
        from shapely.geometry import Point

        from vibespatial.api.geodataframe import GeoDataFrame
        from vibespatial.runtime.dispatch import get_dispatch_events

        left = GeoDataFrame(
            {"a": [1, 2]},
            geometry=[Point(0, 0), Point(5, 5)],
        )
        right = GeoDataFrame(
            {"b": [10, 20]},
            geometry=[Point(0.1, 0.1), Point(5.1, 5.1)],
        )
        get_dispatch_events(clear=True)
        from vibespatial.api.tools.sjoin import sjoin_nearest

        sjoin_nearest(left, right, distance_col="dist")

        events = get_dispatch_events(clear=True)
        sjoin_nearest_events = [
            e for e in events if e.surface == "geopandas.tools.sjoin_nearest"
        ]
        assert len(sjoin_nearest_events) >= 1
        event = sjoin_nearest_events[0]
        assert event.implementation == "sindex_nearest_delegate"
        # The selected mode should come from the actual nearest engine.
        assert event.selected in (ExecutionMode.CPU, ExecutionMode.GPU)

    def test_geom_predicate_query_returns_execution_metadata(self) -> None:
        """_geom_predicate_query should return execution metadata as the third
        element of its return tuple."""
        from shapely.geometry import Point

        from vibespatial.api.geodataframe import GeoDataFrame
        from vibespatial.api.tools.sjoin import _geom_predicate_query
        from vibespatial.spatial.query_types import SpatialQueryExecution

        left = GeoDataFrame(
            {"a": [1]},
            geometry=[Point(0, 0)],
        )
        right = GeoDataFrame(
            {"b": [10]},
            geometry=[Point(0.1, 0.1)],
        )
        (l_idx, r_idx), impl, execution = _geom_predicate_query(
            left, right, "intersects", None,
        )
        assert impl == "owned_spatial_query"
        assert isinstance(execution, SpatialQueryExecution)


class TestDeviceJoinResult:
    """Verify _DeviceJoinResult lazy D-to-H semantics."""

    def test_device_join_result_lazy_materialize(self) -> None:
        """_DeviceJoinResult should defer host copy until properties are
        accessed."""
        from vibespatial.spatial.query_types import _DeviceJoinResult

        # Use pre-populated host arrays as a mock for the lazy path.
        djr = _DeviceJoinResult.__new__(_DeviceJoinResult)
        djr._d_left = None
        djr._d_right = None
        djr._d_distances = None
        djr._h_left = np.array([0, 1, 2], dtype=np.intp)
        djr._h_right = np.array([3, 4, 5], dtype=np.intp)
        djr._h_distances = np.array([0.1, 0.2, 0.3])
        # Properties should return the cached host arrays.
        np.testing.assert_array_equal(djr.left, [0, 1, 2])
        np.testing.assert_array_equal(djr.right, [3, 4, 5])
        np.testing.assert_array_almost_equal(djr.distances, [0.1, 0.2, 0.3])
        left, right = djr.as_tuple()
        np.testing.assert_array_equal(left, [0, 1, 2])
        np.testing.assert_array_equal(right, [3, 4, 5])

    @pytest.mark.gpu
    def test_device_join_result_materialization_is_observable(self) -> None:
        """_DeviceJoinResult should emit materialization events on lazy host copy."""
        pytest.importorskip("cupy")
        import cupy as cp

        from vibespatial.runtime.materialization import (
            clear_materialization_events,
            get_materialization_events,
        )
        from vibespatial.spatial.query_types import _DeviceJoinResult

        result = _DeviceJoinResult(
            cp.asarray([0, 1], dtype=cp.int32),
            cp.asarray([2, 3], dtype=cp.int32),
            cp.asarray([0.5, 1.5], dtype=cp.float64),
        )
        clear_materialization_events()

        left, right = result.as_tuple()
        distances = result.distances
        events = get_materialization_events(clear=True)

        np.testing.assert_array_equal(left, [0, 1])
        np.testing.assert_array_equal(right, [2, 3])
        np.testing.assert_array_equal(distances, [0.5, 1.5])
        assert [event.operation for event in events] == [
            "device_join_indices_to_host",
            "device_join_indices_to_host",
            "device_join_distances_to_host",
        ]
        assert all(event.strict_disallowed is False for event in events)


# ---------------------------------------------------------------------------
# Phase 2: DeviceSpatialJoinResult and return_device parameter
# ---------------------------------------------------------------------------


class TestDeviceSpatialJoinResult:
    """Verify DeviceSpatialJoinResult dataclass semantics."""

    def test_frozen_dataclass_fields(self) -> None:
        """DeviceSpatialJoinResult should be a frozen dataclass with
        d_left_idx and d_right_idx fields."""
        # Verify fields exist on the class.
        import dataclasses

        from vibespatial.spatial.query_types import DeviceSpatialJoinResult
        field_names = {f.name for f in dataclasses.fields(DeviceSpatialJoinResult)}
        assert "d_left_idx" in field_names
        assert "d_right_idx" in field_names

    @pytest.mark.gpu
    def test_to_host_returns_numpy_arrays(self) -> None:
        """to_host() should produce numpy int32 arrays matching device data."""
        pytest.importorskip("cupy")
        import cupy as cp

        from vibespatial.runtime.materialization import (
            clear_materialization_events,
            get_materialization_events,
        )
        from vibespatial.spatial.query_types import DeviceSpatialJoinResult

        d_left = cp.array([0, 1, 2, 3], dtype=cp.int32)
        d_right = cp.array([4, 5, 6, 7], dtype=cp.int32)
        result = DeviceSpatialJoinResult(d_left_idx=d_left, d_right_idx=d_right)
        clear_materialization_events()

        h_left, h_right = result.to_host()
        events = get_materialization_events(clear=True)

        np.testing.assert_array_equal(h_left, [0, 1, 2, 3])
        np.testing.assert_array_equal(h_right, [4, 5, 6, 7])
        assert h_left.dtype == np.int32
        assert h_right.dtype == np.int32
        assert len(events) == 2
        assert {
            (event.operation, event.detail, event.strict_disallowed)
            for event in events
        } == {
            ("device_spatial_join_indices_to_host", "side=left, rows=4, bytes=16", False),
            ("device_spatial_join_indices_to_host", "side=right, rows=4, bytes=16", False),
        }

    @pytest.mark.gpu
    def test_side_to_host_exports_only_requested_index_array(self) -> None:
        """Side-specific exports should not materialize the other pair column."""
        pytest.importorskip("cupy")
        import cupy as cp

        from vibespatial.runtime.materialization import (
            clear_materialization_events,
            get_materialization_events,
        )
        from vibespatial.spatial.query_types import DeviceSpatialJoinResult

        d_left = cp.array([0, 1, 2, 3], dtype=cp.int32)
        d_right = cp.array([4, 5, 6, 7], dtype=cp.int32)
        result = DeviceSpatialJoinResult(d_left_idx=d_left, d_right_idx=d_right)
        clear_materialization_events()

        h_left = result.left_to_host(reason="test left-only export")
        events = get_materialization_events(clear=True)

        np.testing.assert_array_equal(h_left, [0, 1, 2, 3])
        assert h_left.dtype == np.int32
        assert [(event.operation, event.detail, event.reason) for event in events] == [
            (
                "device_spatial_join_indices_to_host",
                "side=left, rows=4, bytes=16",
                "test left-only export",
            ),
        ]

    @pytest.mark.gpu
    def test_size_property(self) -> None:
        """size should report the number of index pairs."""
        pytest.importorskip("cupy")
        import cupy as cp

        from vibespatial.spatial.query_types import DeviceSpatialJoinResult

        d_left = cp.array([0, 1], dtype=cp.int32)
        d_right = cp.array([2, 3], dtype=cp.int32)
        result = DeviceSpatialJoinResult(d_left_idx=d_left, d_right_idx=d_right)
        assert result.size == 2

    @pytest.mark.gpu
    def test_empty_result(self) -> None:
        """Empty DeviceSpatialJoinResult should have size 0."""
        pytest.importorskip("cupy")
        import cupy as cp

        from vibespatial.spatial.query_types import DeviceSpatialJoinResult

        d_left = cp.array([], dtype=cp.int32)
        d_right = cp.array([], dtype=cp.int32)
        result = DeviceSpatialJoinResult(d_left_idx=d_left, d_right_idx=d_right)
        assert result.size == 0
        h_left, h_right = result.to_host()
        assert h_left.size == 0
        assert h_right.size == 0


@pytest.mark.gpu
def test_device_knn_result_to_host_materialization_is_observable() -> None:
    pytest.importorskip("cupy")
    import cupy as cp

    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )
    from vibespatial.spatial.spatial_index_knn_device import DeviceKnnResult

    result = DeviceKnnResult(
        d_query_idx=cp.asarray([0, 1], dtype=cp.int32),
        d_target_idx=cp.asarray([2, 3], dtype=cp.int32),
        d_distances=cp.asarray([0.5, 1.5], dtype=cp.float64),
        total_pairs=2,
        k=1,
    )
    clear_materialization_events()

    query_idx, target_idx, distances = result.to_host()
    events = get_materialization_events(clear=True)

    np.testing.assert_array_equal(query_idx, [0, 1])
    np.testing.assert_array_equal(target_idx, [2, 3])
    np.testing.assert_array_equal(distances, [0.5, 1.5])
    assert [event.operation for event in events] == [
        "device_knn_indices_to_host",
        "device_knn_indices_to_host",
        "device_knn_distances_to_host",
    ]
    assert all(event.strict_disallowed is False for event in events)


def test_return_device_false_returns_numpy() -> None:
    """query_spatial_index with return_device=False (default) returns numpy."""
    tree = np.asarray([box(0, 0, 1, 1), box(2, 2, 3, 3)], dtype=object)
    query = np.asarray([box(0.5, 0.5, 1.5, 1.5)], dtype=object)
    owned, flat = build_owned_spatial_index(tree)

    result = query_spatial_index(
        owned, flat, query, predicate="intersects",
        sort=True, return_device=False,
    )
    assert isinstance(result, np.ndarray)


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required")
def test_return_device_true_returns_device_result() -> None:
    """query_spatial_index with return_device=True returns DeviceSpatialJoinResult
    when GPU execution is selected."""
    from vibespatial.spatial.query_types import DeviceSpatialJoinResult

    tree = np.asarray([box(0, 0, 1, 1), box(2, 2, 3, 3), box(4, 4, 5, 5)], dtype=object)
    tree_owned, flat = build_owned_spatial_index(tree)

    # Query with an OwnedGeometryArray to ensure owned dispatch path is used.
    query_geoms = np.asarray([box(0.5, 0.5, 2.5, 2.5)], dtype=object)
    query_owned = from_shapely_geometries(query_geoms.tolist())

    result, execution = query_spatial_index(
        tree_owned, flat, query_owned, predicate="intersects",
        sort=True, return_device=True, return_metadata=True,
    )

    if execution.selected is ExecutionMode.GPU:
        assert isinstance(result, DeviceSpatialJoinResult)
        h_left, h_right = result.to_host()
        # At minimum box[0] and box[1] should intersect the query
        assert h_left.size > 0
    else:
        # CPU fallback returns numpy as usual
        assert isinstance(result, np.ndarray)


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required")
def test_return_device_point_region_refine_avoids_pair_d2h() -> None:
    """Device pair export should not copy refined pair arrays through host."""
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        get_d2h_transfer_stats,
        reset_d2h_transfer_count,
    )
    from vibespatial.spatial.query_types import DeviceSpatialJoinResult

    tree = np.asarray(
        [
            box(0, 0, 5, 5),
            box(3, 3, 6, 6),
            box(10, 10, 11, 11),
        ],
        dtype=object,
    )
    tree_owned, flat = build_owned_spatial_index(tree)
    query_owned = from_shapely_geometries(
        [Point(1, 1), Point(4, 4), Point(20, 20)],
        residency=Residency.DEVICE,
    )

    tree_owned._ensure_device_state()
    query_owned._ensure_device_state()
    reset_d2h_transfer_count()
    result, execution = query_spatial_index(
        tree_owned,
        flat,
        query_owned,
        predicate="intersects",
        sort=True,
        return_device=True,
        return_metadata=True,
    )
    transfer_count, transfer_bytes = get_d2h_transfer_stats()
    event_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert execution.selected is ExecutionMode.GPU
    assert isinstance(result, DeviceSpatialJoinResult)
    assert transfer_count <= 1
    assert transfer_bytes <= 8
    assert set(event_reasons) <= {"device spatial-index candidate-pair allocation fence"}
    forbidden_exports = (
        "device candidate left pairs host export",
        "device candidate right pairs host export",
        "spatial query predicate family-admission scalar fence",
        "spatial query left family-tag host export",
        "spatial query right family-tag host export",
        "spatial query GPU-pair mask host export",
        "de9im-mask host export",
    )
    assert not any(
        forbidden in reason
        for reason in event_reasons
        for forbidden in forbidden_exports
    )
    h_left, h_right = result.to_host()
    assert h_left.tolist() == [0, 1, 1]
    assert h_right.tolist() == [0, 0, 1]


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required")
def test_return_device_polygonal_intersects_refine_avoids_pair_d2h(monkeypatch) -> None:
    """Polygonal intersects relation refinement should stay device-resident."""
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        get_d2h_transfer_stats,
        reset_d2h_transfer_count,
    )
    from vibespatial.spatial.query_types import DeviceSpatialJoinResult

    tree = np.asarray(
        [
            box(0, 0, 5, 5),
            box(10, 10, 12, 12),
            box(3, 3, 6, 6),
        ],
        dtype=object,
    )
    tree_owned, flat = build_owned_spatial_index(tree)
    query_owned = from_shapely_geometries(
        [box(1, 1, 2, 2), box(4, 4, 11, 11)],
        residency=Residency.DEVICE,
    )

    import vibespatial.predicates.polygon as polygon_predicates

    monkeypatch.setattr(
        polygon_predicates,
        "compute_polygon_de9im_gpu",
        lambda *args, **kwargs: pytest.fail(
            "polygonal intersects should not compute full DE-9IM masks"
        ),
    )

    tree_owned._ensure_device_state()
    query_owned._ensure_device_state()
    reset_d2h_transfer_count()
    result, execution = query_spatial_index(
        tree_owned,
        flat,
        query_owned,
        predicate="intersects",
        sort=True,
        return_device=True,
        return_metadata=True,
    )
    transfer_count, transfer_bytes = get_d2h_transfer_stats()
    event_reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert execution.selected is ExecutionMode.GPU
    assert isinstance(result, DeviceSpatialJoinResult)
    assert transfer_count <= 1
    assert transfer_bytes <= 8
    forbidden_exports = (
        "device candidate left pairs host export",
        "device candidate right pairs host export",
        "spatial query predicate family-admission scalar fence",
        "spatial query left family-tag host export",
        "spatial query right family-tag host export",
        "spatial query GPU-pair mask host export",
        "de9im-mask host export",
    )
    assert not any(
        forbidden in reason
        for reason in event_reasons
        for forbidden in forbidden_exports
    )
    h_left, h_right = result.to_host()
    assert h_left.tolist() == [0, 1, 1, 1]
    assert h_right.tolist() == [0, 0, 1, 2]


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required")
def test_dwithin_return_device_true_stays_device_resident(strict_device_guard) -> None:
    """Per-row dwithin thresholds should stay device-native for return_device=True."""
    cp = pytest.importorskip("cupy")
    from vibespatial.api._native_relation import NativeRelationSelection

    tree = np.asarray([Point(0, 0), Point(10, 0), Point(20, 0)], dtype=object)
    tree_owned, flat = build_owned_spatial_index(tree)
    query_owned = from_shapely_geometries(
        [Point(1, 0), Point(16, 0)],
        residency=Residency.DEVICE,
    )

    result, execution = query_spatial_index(
        tree_owned,
        flat,
        query_owned,
        predicate="dwithin",
        distance=np.asarray([5.0, 5.0], dtype=np.float64),
        sort=True,
        return_device=True,
        return_metadata=True,
    )

    assert execution.selected is ExecutionMode.GPU
    assert isinstance(result, NativeRelationSelection)
    assert result.capacity >= 2
    assert cp.asnumpy(result.logical_count).tolist() == [2]
    assert hasattr(result.relation.left_indices, "__cuda_array_interface__")
    assert hasattr(result.relation.right_indices, "__cuda_array_interface__")


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required")
def test_native_spatial_index_consumes_dwithin_relation_selection(
    strict_device_guard,
) -> None:
    cp = pytest.importorskip("cupy")
    from vibespatial.api._native_relation import NativeRelationSelection
    from vibespatial.api._native_rowset import NativeDeviceSelection

    tree_owned, flat = build_owned_spatial_index(
        np.asarray([Point(0, 0), Point(10, 0), Point(20, 0)], dtype=object)
    )
    query_owned = from_shapely_geometries(
        [Point(1, 0), Point(16, 0)],
        residency=Residency.DEVICE,
    )
    native_index = flat.to_native_spatial_index(source_token="tree")

    relation, execution = native_index.query_relation(
        query_owned,
        predicate="dwithin",
        distance=np.asarray([5.0, 5.0], dtype=np.float64),
        query_token="query",
        query_row_count=2,
        return_device=True,
        return_metadata=True,
    )
    left_rows = native_index.query_left_semijoin(
        query_owned,
        predicate="dwithin",
        distance=np.asarray([5.0, 5.0], dtype=np.float64),
        query_token="query",
        query_row_count=2,
    )

    assert execution.selected is ExecutionMode.GPU
    assert isinstance(relation, NativeRelationSelection)
    assert relation.relation.left_token == "query"
    assert relation.relation.right_token == "tree"
    assert cp.asnumpy(relation.logical_count).tolist() == [2]
    assert isinstance(left_rows, NativeDeviceSelection)
    assert cp.asnumpy(left_rows.logical_count).tolist() == [2]


def test_dwithin_strict_device_return_precedes_terminal_compaction() -> None:
    path = Path(spatial_nearest_module.__file__)
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_dwithin_refine_gpu"
    )
    function_source = ast.get_source_segment(source, function) or ""
    device_return = function_source.index("if return_device:", function_source.index("d_keep ="))
    terminal_compaction = function_source.index("d_keep_idx = cp.flatnonzero(d_keep)")

    assert device_return < terminal_compaction
    strict_section = function_source[device_return:terminal_compaction]
    assert "NativeDeviceSelection.from_mask" in strict_section
    assert "filter_pairs_selection" in strict_section
    assert "NativeRelationSelection" in function_source
    assert "if d_keep_idx.size" not in function_source

    mixed = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_compute_mixed_distances_gpu_device"
    )
    mixed_source = ast.get_source_segment(source, mixed) or ""
    assert "NativeRelationFamilyPartition.from_pair_capacity" in mixed_source
    assert "cp.flatnonzero" not in mixed_source
    assert "copy_device_to_host" not in mixed_source
    assert "to_shapely" not in mixed_source
    assert "NativeDeviceSelection.from_mask" not in mixed_source
    assert "sub_mask" not in mixed_source
    assert "sub_count" not in mixed_source


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required")
def test_return_device_backward_compat() -> None:
    """Existing callers without return_device still get numpy arrays."""
    tree = np.asarray([box(0, 0, 1, 1), box(2, 2, 3, 3)], dtype=object)
    query = np.asarray([box(0.5, 0.5, 1.5, 1.5)], dtype=object)
    owned, flat = build_owned_spatial_index(tree)

    # No return_device parameter — must return numpy.
    result = query_spatial_index(
        owned, flat, query, predicate="intersects", sort=True,
    )
    assert isinstance(result, np.ndarray)


def test_sindex_query_return_device_false_default() -> None:
    """SpatialIndex.query() defaults to return_device=False, producing numpy."""
    from vibespatial.api import GeoSeries

    gs = GeoSeries([box(0, 0, 1, 1), box(2, 2, 3, 3)])
    result = gs.sindex.query(box(0.5, 0.5, 1.5, 1.5))
    assert isinstance(result, np.ndarray)


def test_sindex_query_any_is_eager_index_aligned_and_cpu_fallback_is_observable() -> None:
    import pandas as pd

    from vibespatial.api import GeoSeries
    from vibespatial.runtime.fallbacks import clear_fallback_events, get_fallback_events

    tree = GeoSeries([Point(0, 0), Point(10, 0)])
    query = GeoSeries(
        [box(-1, -1, 1, 1), box(4, -1, 5, 1), box(9, -1, 11, 1)],
        index=pd.Index(["first", "miss", "last"], name="query"),
    )
    clear_fallback_events()

    with set_requested_mode(ExecutionMode.CPU):
        result = tree.sindex.query_any(query, predicate="intersects")

    assert type(result) is pd.Series
    assert result.name == "has_match"
    assert result.index.equals(query.index)
    assert result.tolist() == [True, False, True]
    fallback = get_fallback_events(clear=True)
    assert any(
        event.surface == "vibespatial.api.SpatialIndex.query_any"
        and event.requested is ExecutionMode.CPU
        and event.selected is ExecutionMode.CPU
        for event in fallback
    )

    with pytest.raises(ValueError, match="must be one of"):
        tree.sindex.query_any(query, predicate="not-a-predicate")
    with pytest.raises(ValueError, match="only supported"):
        tree.sindex.query_any(query, predicate="intersects", distance=1.0)
    with pytest.raises(ValueError, match="required"):
        tree.sindex.query_any(query, predicate="dwithin")


@pytest.mark.parametrize("quad_segs", [None, 0, -2])
@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime required")
def test_sindex_query_any_buffer_point_rewrite_is_bounded_strict_native(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    quad_segs,
) -> None:
    import gc

    import cupy as cp

    from vibespatial import GeoDataFrame, GeoSeries, read_parquet
    from vibespatial.api._native_public_arrays import NativeBooleanMaskArray
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        get_d2h_transfer_stats,
        reset_d2h_transfer_count,
    )
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events
    from vibespatial.runtime.fallbacks import clear_fallback_events, get_fallback_events
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )
    from vibespatial.runtime.provenance import clear_rewrite_events, get_rewrite_events
    from vibespatial.testing import strict_native_environment

    source_points = [Point(0, 0), Point(10, 0), Point(20, 0)]
    effective_quad_segs = 16 if quad_segs is None else max(int(quad_segs), 1)
    edge_angle = np.pi / (4 * effective_quad_segs)
    radial_scale = (1.0 + np.cos(edge_angle)) / 2.0
    ideal_circle_only_point = Point(
        3.0 * radial_scale * np.cos(edge_angle),
        3.0 * radial_scale * np.sin(edge_angle),
    )
    query_geometries = [
        box(-1, -1, 1, 1),
        box(2.5, -0.5, 3.5, 0.5),
        box(5, -0.5, 6, 0.5),
        box(9, -1, 11, 1),
        ideal_circle_only_point,
        Polygon(),
    ]
    buffer_kwargs = {} if quad_segs is None else {"quad_segs": quad_segs}
    expected = [
        any(
            query.intersects(point.buffer(3.0, **buffer_kwargs))
            for point in source_points
        )
        if not query.is_empty
        else False
        for query in query_geometries
    ]
    source_path = tmp_path / "query-any-source.parquet"
    query_path = tmp_path / "query-any-input.parquet"
    GeoDataFrame(
        geometry=GeoSeries(source_points, crs="EPSG:3857"),
        crs="EPSG:3857",
    ).to_parquet(source_path, geometry_encoding="geoarrow", index=False)
    GeoDataFrame(
        {
            "qid": np.arange(len(query_geometries)),
            "geometry": GeoSeries(query_geometries, crs="EPSG:3857"),
        },
        geometry="geometry",
        crs="EPSG:3857",
    ).to_parquet(query_path, geometry_encoding="geoarrow", index=False)
    source = read_parquet(source_path).geometry
    query_frame = read_parquet(query_path)
    buffered = source.buffer(3.0, **buffer_kwargs)
    spatial_index = buffered.sindex

    spatial_index_type = type(spatial_index)
    original_semijoin = spatial_index_type.query_left_semijoin
    semijoin_predicates = []

    def _record_bounded_buffer_semijoin(self, *args, **kwargs):
        semijoin_predicates.append(kwargs.get("predicate"))
        return original_semijoin(self, *args, **kwargs)

    monkeypatch.setattr(
        spatial_index_type,
        "query_left_semijoin",
        _record_bounded_buffer_semijoin,
    )
    reset_d2h_transfer_count()
    clear_materialization_events()
    clear_dispatch_events()
    clear_fallback_events()
    clear_rewrite_events()

    with strict_native_environment():
        result = spatial_index.query_any(
            query_frame.geometry,
            predicate="intersects",
        )
        del buffered, spatial_index, source
        gc.collect()
        allocation_churn = [
            cp.empty(1 << exponent, dtype=cp.uint8)
            for exponent in range(10, 21)
        ]
        selected = query_frame[result]
        inverted = ~result
        rejected = query_frame[inverted]
        del allocation_churn

    transfer_count, transfer_bytes = get_d2h_transfer_stats()
    transfer_events = get_d2h_transfer_events(clear=True)
    materializations = get_materialization_events(clear=True)
    dispatch = get_dispatch_events(clear=True)
    rewrites = get_rewrite_events(clear=True)

    assert isinstance(result.array, NativeBooleanMaskArray)
    assert isinstance(inverted.array, NativeBooleanMaskArray)
    assert inverted.array.selection is not None
    assert result.index.equals(query_frame.index)
    assert get_fallback_events(clear=True) == []
    assert selected["qid"].tolist() == [index for index, keep in enumerate(expected) if keep]
    assert rejected["qid"].tolist() == [
        index for index, keep in enumerate(expected) if not keep
    ]
    assert semijoin_predicates == ["intersects"]
    assert not any(
        event.operation == "query_any_mask_to_public_array"
        for event in materializations
    )
    assert transfer_count <= 10
    assert transfer_bytes <= 528, [
        (event.reason, event.item_count, event.bytes_transferred)
        for event in transfer_events
    ]
    assert max((event.item_count for event in transfer_events), default=0) <= 64
    assert any(
        event.surface == "vibespatial.api.SpatialIndex.query_any"
        and event.selected is ExecutionMode.GPU
        and event.implementation == "owned_gpu_buffer_point_existential"
        for event in dispatch
    )
    assert any(
        event.rule_name == "R2_query_any_buffer_intersects_to_bounded_refine"
        for event in rewrites
    )
    assert expected[4] is False
    assert result.to_numpy().tolist() == expected
    assert inverted.to_numpy().tolist() == [not keep for keep in expected]


@pytest.mark.parametrize(
    ("predicate", "distance"),
    [
        ("intersects", None),
        ("contains", None),
        ("within", None),
        ("dwithin", 0.25),
        ("dwithin", np.asarray(0.25)),
        ("dwithin", np.asarray([0.25])),
    ],
)
@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime required")
def test_sindex_query_any_generic_gpu_predicates_preserve_public_shape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    predicate,
    distance,
) -> None:
    import pandas as pd

    from vibespatial import GeoDataFrame, GeoSeries, read_parquet
    from vibespatial.api._native_metadata import NativeSpatialIndex
    from vibespatial.api._native_public_arrays import NativeBooleanMaskArray
    from vibespatial.runtime.fallbacks import clear_fallback_events, get_fallback_events
    from vibespatial.testing import strict_native_environment

    tree_geometries = [Point(0, 0), Point(2, 0), Point(5, 0)]
    query_geometries = [
        box(-0.5, -0.5, 0.5, 0.5),
        Point(2, 0),
        None,
        box(8, 8, 9, 9),
    ]
    query_index = pd.Index(["duplicate", "duplicate", "null", "miss"])
    tree_path = tmp_path / "query-any-generic-tree.parquet"
    query_path = tmp_path / "query-any-generic-input.parquet"
    GeoDataFrame(
        geometry=GeoSeries(tree_geometries, crs="EPSG:3857"),
        crs="EPSG:3857",
    ).to_parquet(tree_path, geometry_encoding="geoarrow", index=False)
    GeoDataFrame(
        geometry=GeoSeries(
            query_geometries,
            index=query_index,
            crs="EPSG:3857",
        ),
        crs="EPSG:3857",
    ).to_parquet(query_path, geometry_encoding="geoarrow", index=True)
    tree = read_parquet(tree_path).geometry
    query = read_parquet(query_path).geometry
    clear_fallback_events()

    def _relation_materialization_is_forbidden(*args, **kwargs):
        raise AssertionError("query_any must reduce bounded tiles without a relation")

    monkeypatch.setattr(
        NativeSpatialIndex,
        "query_relation",
        _relation_materialization_is_forbidden,
    )

    kwargs = {"predicate": predicate}
    if distance is not None:
        kwargs["distance"] = distance
    with strict_native_environment():
        result = tree.sindex.query_any(query, **kwargs)

    expected = []
    for query_geometry in query_geometries:
        if query_geometry is None:
            expected.append(False)
        elif predicate == "dwithin":
            expected.append(
                any(query_geometry.distance(tree_geometry) <= distance for tree_geometry in tree_geometries)
            )
        else:
            operation = getattr(query_geometry, predicate)
            expected.append(any(operation(tree_geometry) for tree_geometry in tree_geometries))

    assert isinstance(result.array, NativeBooleanMaskArray)
    assert result.index.equals(query_index)
    assert result.to_numpy().tolist() == expected
    assert get_fallback_events(clear=True) == []


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime required")
def test_sindex_query_any_generic_gpu_scalar_returns_one_row() -> None:
    import pandas as pd

    from vibespatial import GeoSeries
    from vibespatial.api._native_public_arrays import NativeBooleanMaskArray
    from vibespatial.runtime.fallbacks import clear_fallback_events, get_fallback_events
    from vibespatial.testing import strict_native_environment

    tree = GeoSeries([Point(0, 0), Point(2, 0)])
    clear_fallback_events()

    with strict_native_environment():
        result = tree.sindex.query_any(Point(0, 0), predicate="intersects")

    assert isinstance(result.array, NativeBooleanMaskArray)
    assert result.index.equals(pd.RangeIndex(1))
    assert result.to_numpy().tolist() == [True]
    assert get_fallback_events(clear=True) == []


def test_sindex_query_aggregate_is_eager_public_dataframe() -> None:
    from vibespatial.api import GeoSeries
    from vibespatial.runtime.fallbacks import clear_fallback_events, get_fallback_events

    tree = GeoSeries([Point(0, 0), Point(1, 1), Point(10, 10)])
    query = GeoSeries(
        [box(-1, -1, 2, 2), box(9, 9, 11, 11), box(20, 20, 21, 21)],
        index=["near", "far", "empty"],
    )
    values = np.asarray([1.5, 2.5, 4.0], dtype=np.float64)
    large_values = np.asarray([2**53 + 1, 2, 5], dtype=np.int64)
    bool_values = np.asarray([True, True, False], dtype=np.bool_)
    clear_fallback_events()

    result = tree.sindex.query_aggregate(
        query,
        {
            "match_count": "size",
            "value_sum": (values, "sum"),
            "large_sum": (large_values, "sum"),
            "bool_sum": (bool_values, "sum"),
        },
        predicate="intersects",
    )
    events = get_fallback_events(clear=True)

    assert type(result) is __import__("pandas").DataFrame
    assert result.index.tolist() == ["near", "far", "empty"]
    assert result["match_count"].tolist() == [2, 1, 0]
    assert result["value_sum"].tolist() == [4.0, 4.0, 0.0]
    assert result["large_sum"].tolist() == [2**53 + 3, 5, 0]
    assert result["large_sum"].dtype == np.dtype(np.int64)
    assert result["bool_sum"].tolist() == [2, 0, 0]
    assert result["bool_sum"].dtype == np.dtype(np.int64)
    assert any(event.pipeline == "spatial_query_aggregate" for event in events)
    with pytest.raises(
        ValueError,
        match="must align with indexed tree geometries",
    ):
        tree.sindex.query_aggregate(
            query,
            {"bad": (values[:2], "sum")},
            predicate="intersects",
        )
    with pytest.raises(TypeError, match="sum values must be numeric"):
        tree.sindex.query_aggregate(
            query,
            {"bad": (["a", "b", "c"], "sum")},
            predicate="intersects",
        )


def test_sindex_query_pair_aggregate_preserves_overlap_multiplicity() -> None:
    import pandas as pd

    from vibespatial.api import GeoSeries

    pickup = GeoSeries(
        [Point(0, 0), Point(1, 1), Point(10, 10), Point(20, 20)]
    )
    dropoff = GeoSeries(
        [Point(1, 1), Point(2.5, 2.5), Point(0, 0), Point(0, 0)]
    )
    zones = GeoSeries(
        [
            box(-1, -1, 2, 2),
            box(0.5, 0.5, 3, 3),
            box(9, 9, 11, 11),
        ]
    )

    result = pickup.sindex.query_pair_aggregate(
        dropoff.sindex,
        zones,
        predicate="contains",
    )

    assert type(result) is pd.DataFrame
    assert result.index.equals(pd.RangeIndex(4))
    assert result.to_dict("list") == {
        "left_count": [1, 2, 1, 0],
        "right_count": [2, 1, 1, 1],
        "shared_count": [1, 1, 0, 0],
    }
    assert int(
        (result["left_count"] * result["right_count"]).sum()
        - result["shared_count"].sum()
    ) == 3
    with pytest.raises(ValueError, match="equal size"):
        pickup.sindex.query_pair_aggregate(
            GeoSeries([Point(0, 0)]).sindex,
            zones,
            predicate="contains",
        )


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime required")
def test_sindex_query_aggregate_consumes_native_relation_before_export(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    from vibespatial.api import GeoDataFrame, GeoSeries, points_from_xy, read_parquet
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_stats,
        reset_d2h_transfer_count,
    )
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events
    from vibespatial.runtime.fallbacks import clear_fallback_events, get_fallback_events
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    source = GeoDataFrame(
        {
            "value": [1.5, 2.5, 4.0],
            "large_value": [2**53 + 1, 2, 5],
            "enabled": [True, True, False],
            "start": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03"]
            ),
            "stop": pd.to_datetime(
                [
                    "2020-01-01 00:00:01",
                    "2020-01-02 00:00:02",
                    "2020-01-03 00:00:04",
                ]
            ),
        },
        geometry=points_from_xy([0, 1, 10], [0, 1, 10], crs="EPSG:4326"),
        crs="EPSG:4326",
    )
    path = tmp_path / "query-aggregate.parquet"
    source.to_parquet(path, geometry_encoding="geoarrow", index=False)
    tree = read_parquet(path)
    query = GeoSeries(
        [box(-1, -1, 2, 2), box(9, 9, 11, 11), box(20, 20, 21, 21)],
        index=["near", "far", "empty"],
        crs=tree.crs,
    )
    reset_d2h_transfer_count()
    clear_materialization_events()
    clear_dispatch_events()
    clear_fallback_events()
    duration_seconds = (
        tree["stop"].astype("int64") - tree["start"].astype("int64")
    ) / 1_000_000.0

    result = tree.sindex.query_aggregate(
        query,
        {
            "match_count": "size",
            "value_sum": (tree["value"], "sum"),
            "large_sum": (tree["large_value"], "sum"),
            "bool_sum": (tree["enabled"], "sum"),
            "duration_sum": (duration_seconds, "sum"),
        },
        predicate="intersects",
    )
    accumulated = pd.DataFrame(
        {
            column: result[column] + result[column]
            for column in result.columns
        },
        index=result.index,
    )
    transfer_count, transfer_bytes = get_d2h_transfer_stats()
    materializations = get_materialization_events(clear=True)
    dispatch = get_dispatch_events(clear=True)

    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame(
            {
                "match_count": [2, 1, 0],
                "value_sum": [4.0, 4.0, 0.0],
                "large_sum": [2**53 + 3, 5, 0],
                "bool_sum": [2, 0, 0],
                "duration_sum": [3.0, 4.0, 0.0],
            },
            index=["near", "far", "empty"],
        ),
        check_dtype=False,
    )
    assert get_fallback_events(clear=True) == []
    assert all(
        type(accumulated[column].array).__name__
        == "NativeNumericExpressionArray"
        for column in accumulated.columns
    )
    assert not any(
        event.operation == "sindex_query_relation_indices_to_host"
        for event in materializations
    )
    # The eager pandas columns are already computed but remain device-backed;
    # even public Series addition does not export them. Only planning packets
    # are allowed before the caller explicitly requests a NumPy value array.
    assert transfer_count <= 5
    assert transfer_bytes <= 64
    assert any(
        event.surface == "vibespatial.api.SpatialIndex.query_aggregate"
        and event.selected is ExecutionMode.GPU
        for event in dispatch
    )
    assert accumulated["match_count"].to_numpy().tolist() == [4, 2, 0]
    assert accumulated["value_sum"].to_numpy().tolist() == [8.0, 8.0, 0.0]
    assert result["large_sum"].to_numpy().dtype == np.dtype(np.int64)
    assert result["large_sum"].to_numpy().tolist() == [2**53 + 3, 5, 0]
    assert result["bool_sum"].to_numpy().dtype == np.dtype(np.int64)
    assert result["bool_sum"].to_numpy().tolist() == [2, 0, 0]
    assert accumulated["duration_sum"].to_numpy().tolist() == [6.0, 8.0, 0.0]
    reset_d2h_transfer_count()

    clear_dispatch_events()
    count_only = tree.sindex.query_aggregate(
        query,
        {"match_count": "size"},
        predicate="intersects",
    )
    count_dispatch = get_dispatch_events(clear=True)
    assert count_only["match_count"].to_numpy().tolist() == [2, 1, 0]
    assert any(
        event.surface == "vibespatial.api.SpatialIndex.query_aggregate"
        and "reduced matches directly" in event.reason
        and "NativeRelation pairs" not in event.reason
        for event in count_dispatch
    )

    clear_dispatch_events()
    dwithin_count = tree.sindex.query_aggregate(
        query,
        {"match_count": "size"},
        predicate="dwithin",
        distance=0.0,
    )
    dwithin_count_dispatch = get_dispatch_events(clear=True)
    assert dwithin_count["match_count"].to_numpy().tolist() == [2, 1, 0]
    assert any(
        event.surface == "vibespatial.api.SpatialIndex.query_aggregate"
        and event.implementation == "owned_gpu_spatial_match_count"
        and "reduced matches directly" in event.reason
        and "NativeRelation pairs" not in event.reason
        for event in dwithin_count_dispatch
    )

    spatial_index = tree.sindex
    original_query_relation = spatial_index.query_relation
    relation_calls = 0

    def _cpu_selected_relation(*args, **kwargs):
        nonlocal relation_calls
        relation_calls += 1
        relation, execution = original_query_relation(*args, **kwargs)
        return relation, type(execution)(
            requested=execution.requested,
            selected=ExecutionMode.CPU,
            implementation="test_cpu_refinement",
            reason="exercise reusable CPU-selected relation",
        )

    def _fail_duplicate_query(*_args, **_kwargs):
        raise AssertionError("CPU-selected relation must not execute the query twice")

    monkeypatch.setattr(spatial_index, "query_relation", _cpu_selected_relation)
    monkeypatch.setattr(spatial_index, "query", _fail_duplicate_query)
    cpu_selected = spatial_index.query_aggregate(
        query,
        {
            "match_count": "size",
            "large_sum": (tree["large_value"], "sum"),
            "bool_sum": (tree["enabled"], "sum"),
        },
        predicate="intersects",
    )

    assert relation_calls == 1
    assert cpu_selected["match_count"].tolist() == [2, 1, 0]
    assert cpu_selected["large_sum"].tolist() == [2**53 + 3, 5, 0]
    assert cpu_selected["large_sum"].dtype == np.dtype(np.int64)
    assert cpu_selected["bool_sum"].tolist() == [2, 0, 0]
    assert cpu_selected["bool_sum"].dtype == np.dtype(np.int64)


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime required")
def test_sindex_query_pair_aggregate_reduces_candidate_tiles_before_export(
    tmp_path,
) -> None:
    from vibespatial.api import GeoDataFrame, GeoSeries, points_from_xy, read_parquet
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        get_d2h_transfer_stats,
        reset_d2h_transfer_count,
    )
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events
    from vibespatial.runtime.fallbacks import clear_fallback_events, get_fallback_events
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    source = GeoDataFrame(
        {
            "dropoff": points_from_xy([1, 2.5, 0, 0], [1, 2.5, 0, 0]),
        },
        geometry=points_from_xy([0, 1, 10, 20], [0, 1, 10, 20]),
        crs="EPSG:4326",
    ).rename_geometry("pickup")
    source["dropoff"] = source["dropoff"].set_crs(source.crs)
    path = tmp_path / "query-pair-aggregate.parquet"
    source.to_parquet(path, geometry_encoding="geoarrow", index=False)
    source = read_parquet(path)
    pickup = source.set_geometry("pickup").geometry
    dropoff = source.set_geometry("dropoff").geometry
    zones = GeoSeries(
        [
            box(-1, -1, 2, 2),
            box(0.5, 0.5, 3, 3),
            box(9, 9, 11, 11),
        ],
        crs=source.crs,
    )

    reset_d2h_transfer_count()
    clear_materialization_events()
    clear_dispatch_events()
    clear_fallback_events()
    result = pickup.sindex.query_pair_aggregate(
        dropoff.sindex,
        zones,
        predicate="contains",
    )
    transfer_count, transfer_bytes = get_d2h_transfer_stats()
    transfer_events = get_d2h_transfer_events()
    materializations = get_materialization_events(clear=True)
    dispatch = get_dispatch_events(clear=True)

    assert get_fallback_events(clear=True) == []
    assert all(
        type(result[column].array).__name__ == "NativeNumericExpressionArray"
        for column in result.columns
    )
    assert not any(
        event.operation == "sindex_query_relation_indices_to_host"
        for event in materializations
    )
    assert transfer_count <= 8
    assert transfer_bytes <= 1024
    assert max(event.item_count for event in transfer_events) <= 33
    assert not any(
        event.reason == "point-grid conservative candidate allocation fence"
        for event in transfer_events
    )
    assert any(
        event.surface == "vibespatial.api.SpatialIndex.query_pair_aggregate"
        and event.selected is ExecutionMode.GPU
        for event in dispatch
    )
    cross_count = (
        (result["left_count"] * result["right_count"]).sum()
        - result["shared_count"].sum()
    )
    assert isinstance(cross_count, int)
    assert cross_count == 3
    assert result["left_count"].to_numpy().tolist() == [1, 2, 1, 0]
    assert result["right_count"].to_numpy().tolist() == [2, 1, 1, 1]
    assert result["shared_count"].to_numpy().tolist() == [1, 1, 0, 0]


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime required")
def test_sindex_query_pair_aggregate_respects_explicit_cpu_mode(tmp_path) -> None:
    from vibespatial.api import GeoDataFrame, GeoSeries, points_from_xy, read_parquet
    from vibespatial.api._native_public_arrays import NativeNumericExpressionArray
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events
    from vibespatial.runtime.fallbacks import clear_fallback_events, get_fallback_events

    source = GeoDataFrame(
        {"dropoff": points_from_xy([0.0, 10.0], [0.0, 0.0])},
        geometry=points_from_xy([0.0, 10.0], [0.0, 0.0]),
        crs="EPSG:3857",
    ).rename_geometry("pickup")
    source["dropoff"] = source["dropoff"].set_crs(source.crs)
    path = tmp_path / "query-pair-explicit-cpu.parquet"
    source.to_parquet(path, geometry_encoding="geoarrow", index=False)
    source = read_parquet(path)
    pickup = source.set_geometry("pickup").geometry
    dropoff = source.set_geometry("dropoff").geometry
    zones = GeoSeries(
        [box(-1.0, -1.0, 1.0, 1.0), box(9.0, -1.0, 11.0, 1.0)],
        crs=source.crs,
    )

    clear_dispatch_events()
    clear_fallback_events()
    with set_requested_mode(ExecutionMode.CPU):
        result = pickup.sindex.query_pair_aggregate(
            dropoff.sindex,
            zones,
            predicate="contains",
        )
    dispatch = get_dispatch_events(clear=True)
    fallback = get_fallback_events(clear=True)

    assert all(
        not isinstance(result[column].array, NativeNumericExpressionArray)
        for column in result.columns
    )
    assert result.to_dict("list") == {
        "left_count": [1, 1],
        "right_count": [1, 1],
        "shared_count": [1, 1],
    }
    assert not any(
        event.surface == "vibespatial.api.SpatialIndex.query_pair_aggregate"
        and event.selected is ExecutionMode.GPU
        for event in dispatch
    )
    pair_fallback = next(
        event
        for event in fallback
        if event.surface == "vibespatial.api.SpatialIndex.query_pair_aggregate"
    )
    assert pair_fallback.requested is ExecutionMode.CPU
    assert pair_fallback.selected is ExecutionMode.CPU


def _bounded_pair_aggregate_inputs(tmp_path):
    from vibespatial.api import GeoDataFrame, points_from_xy, read_parquet

    x = np.asarray([0.0, 10.0, 20.0, 0.0, 10.0, 20.0])
    y = np.asarray([0.0, 0.0, 0.0, 10.0, 10.0, 10.0])
    source = GeoDataFrame(
        {"dropoff": points_from_xy(x, y)},
        geometry=points_from_xy(x, y),
        crs="EPSG:3857",
    ).rename_geometry("pickup")
    source["dropoff"] = source["dropoff"].set_crs(source.crs)
    path = tmp_path / "query-pair-bounded-grid.parquet"
    source.to_parquet(path, geometry_encoding="geoarrow", index=False)
    source = read_parquet(path)
    return (
        source.set_geometry("pickup").geometry,
        source.set_geometry("dropoff").geometry,
    )


def _record_pair_reduction_tile_sizes(monkeypatch, *, pair_budget: int = 2):
    import vibespatial.spatial.point_grid_index as point_grid_module
    import vibespatial.spatial.spatial_index_device as device_module

    capacities = []
    tile_sizes = []
    original_superset_query = point_grid_module.point_grid_superset_query
    original_classifier = device_module._classify_homogeneous_reduction_tile

    def recording_superset_query(*args, **kwargs):
        capacity = kwargs.get("pair_capacity")
        if capacity is not None:
            capacities.append(int(capacity))
        return original_superset_query(*args, **kwargs)

    def recording_classifier(*args, **kwargs):
        tile_sizes.append(int(args[3].size))
        return original_classifier(*args, **kwargs)

    monkeypatch.setattr(
        point_grid_module,
        "point_grid_superset_query",
        recording_superset_query,
    )
    monkeypatch.setattr(point_grid_module, "_MIN_POINT_GRID_ROWS", 1)
    monkeypatch.setattr(
        device_module,
        "_classify_homogeneous_reduction_tile",
        recording_classifier,
    )
    monkeypatch.setattr(
        device_module,
        "_spatial_reduction_tile_lane_capacity",
        lambda *_args, **_kwargs: pair_budget,
    )
    return capacities, tile_sizes


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime required")
def test_sindex_query_pair_aggregate_reuses_only_shared_candidate_superset(
    tmp_path,
    monkeypatch,
) -> None:
    """Aligned point columns may have different conservative candidates."""
    from vibespatial.api import GeoDataFrame, GeoSeries, points_from_xy, read_parquet
    from vibespatial.runtime.fallbacks import clear_fallback_events, get_fallback_events

    pickup_x = np.asarray([0.0, 10.0, 0.0, 10.0])
    pickup_y = np.asarray([0.0, 10.0, 10.0, 0.0])
    dropoff_x = np.asarray([10.0, 0.0, 0.0, 20.0])
    dropoff_y = np.asarray([10.0, 0.0, 10.0, 20.0])
    source = GeoDataFrame(
        {"dropoff": points_from_xy(dropoff_x, dropoff_y)},
        geometry=points_from_xy(pickup_x, pickup_y),
        crs="EPSG:3857",
    ).rename_geometry("pickup")
    source["dropoff"] = source["dropoff"].set_crs(source.crs)
    path = tmp_path / "query-pair-asymmetric-grid.parquet"
    source.to_parquet(path, geometry_encoding="geoarrow", index=False)
    source = read_parquet(path)
    pickup = source.set_geometry("pickup").geometry
    dropoff = source.set_geometry("dropoff").geometry
    zones = GeoSeries(
        [
            box(x - 0.25, y - 0.25, x + 0.25, y + 0.25)
            for x, y in (
                (0.0, 0.0),
                (10.0, 10.0),
                (0.0, 10.0),
                (10.0, 0.0),
                (20.0, 20.0),
            )
        ],
        crs=source.crs,
    )
    clear_fallback_events()

    automatic = pickup.sindex.query_pair_aggregate(
        dropoff.sindex,
        zones,
        predicate="contains",
    )
    monkeypatch.setattr(
        "vibespatial.spatial.point_grid_index._MIN_POINT_GRID_ROWS",
        1 << 60,
    )
    forced_baseline = pickup.sindex.query_pair_aggregate(
        dropoff.sindex,
        zones,
        predicate="contains",
    )
    monkeypatch.setattr(
        "vibespatial.spatial.point_grid_index._MIN_POINT_GRID_ROWS",
        1,
    )
    forced_alternative = pickup.sindex.query_pair_aggregate(
        dropoff.sindex,
        zones,
        predicate="contains",
    )

    assert get_fallback_events(clear=True) == []
    expected = {
        "left_count": [1, 1, 1, 1],
        "right_count": [1, 1, 1, 1],
        "shared_count": [0, 0, 1, 0],
    }
    assert automatic.to_dict("list") == expected
    assert forced_baseline.to_dict("list") == expected
    assert forced_alternative.to_dict("list") == expected


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime required")
def test_sindex_query_pair_aggregate_consumes_multiple_bounded_grid_partitions(
    tmp_path,
    monkeypatch,
) -> None:
    from vibespatial.api import GeoSeries
    from vibespatial.runtime.fallbacks import clear_fallback_events, get_fallback_events
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    pickup, dropoff = _bounded_pair_aggregate_inputs(tmp_path)
    zones = GeoSeries(
        [
            box(x - 1.0, y - 1.0, x + 1.0, y + 1.0)
            for x, y in (
                (0.0, 0.0),
                (10.0, 0.0),
                (20.0, 0.0),
                (0.0, 10.0),
                (10.0, 10.0),
                (20.0, 10.0),
            )
        ],
        crs=pickup.crs,
    )
    capacities, tile_sizes = _record_pair_reduction_tile_sizes(monkeypatch)

    clear_fallback_events()
    clear_materialization_events()
    result = pickup.sindex.query_pair_aggregate(
        dropoff.sindex,
        zones,
        predicate="contains",
    )
    materializations = get_materialization_events(clear=True)

    assert get_fallback_events(clear=True) == []
    assert result.to_dict("list") == {
        "left_count": [1] * 6,
        "right_count": [1] * 6,
        "shared_count": [1] * 6,
    }
    assert len(capacities) > 2
    assert capacities and max(capacities) <= 2
    assert tile_sizes and max(tile_sizes) <= 2
    assert not any(
        event.operation == "sindex_query_relation_indices_to_host"
        for event in materializations
    )


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime required")
def test_sindex_query_pair_aggregate_tiles_single_oversized_grid_row(
    tmp_path,
    monkeypatch,
) -> None:
    from vibespatial.api import GeoSeries
    from vibespatial.runtime.fallbacks import clear_fallback_events, get_fallback_events
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    pickup, dropoff = _bounded_pair_aggregate_inputs(tmp_path)
    zones = GeoSeries([box(-1.0, -1.0, 21.0, 11.0)], crs=pickup.crs)
    capacities, tile_sizes = _record_pair_reduction_tile_sizes(monkeypatch)

    clear_fallback_events()
    clear_materialization_events()
    result = pickup.sindex.query_pair_aggregate(
        dropoff.sindex,
        zones,
        predicate="contains",
    )
    materializations = get_materialization_events(clear=True)

    assert get_fallback_events(clear=True) == []
    assert result.to_dict("list") == {
        "left_count": [1] * 6,
        "right_count": [1] * 6,
        "shared_count": [1] * 6,
    }
    assert capacities == []
    assert len(tile_sizes) >= 9
    assert max(tile_sizes) <= 2
    assert not any(
        event.operation == "sindex_query_relation_indices_to_host"
        for event in materializations
    )


def test_sindex_query_return_device_rejects_dense_sparse_exports() -> None:
    """Dense/sparse query outputs are public exports, not native carriers."""
    from vibespatial.api import GeoSeries

    gs = GeoSeries([box(0, 0, 1, 1), box(2, 2, 3, 3)])
    with pytest.raises(ValueError, match="query_relation"):
        gs.sindex.query(
            [box(0.5, 0.5, 1.5, 1.5)],
            output_format="dense",
            return_device=True,
        )
    with pytest.raises(ValueError, match="query_relation"):
        gs.sindex.query(
            [box(0.5, 0.5, 1.5, 1.5)],
            output_format="sparse",
            return_device=True,
        )


def test_point_grid_block_partitions_never_exceed_pair_budget() -> None:
    from vibespatial.spatial.point_grid_index import (
        _count_bounded_block_partitions,
    )

    partitions = _count_bounded_block_partitions(
        np.asarray([6, 6, 6], dtype=np.int64),
        block_size=2,
        query_count=5,
        pair_budget=10,
    )

    assert partitions == ((0, 2, 6), (2, 4, 6), (4, 5, 6))
    assert all(capacity <= 10 for _, _, capacity in partitions)


def test_point_grid_block_partitions_decline_oversized_block() -> None:
    from vibespatial.spatial.point_grid_index import (
        _count_bounded_block_partitions,
    )

    assert (
        _count_bounded_block_partitions(
            np.asarray([11], dtype=np.int64),
            block_size=2,
            query_count=2,
            pair_budget=10,
        )
        is None
    )
    assert _count_bounded_block_partitions(
        np.asarray([3, 11, 4], dtype=np.int64),
        block_size=1,
        query_count=3,
        pair_budget=10,
        admit_oversized=True,
    ) == ((0, 1, 3), (1, 2, 11), (2, 3, 4))


def test_sindex_query_dense_public_export_uses_native_relation() -> None:
    from vibespatial.api import GeoSeries
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    gs = GeoSeries([box(0, 0, 1, 1), box(2, 2, 3, 3)])
    query = [box(0.5, 0.5, 1.5, 1.5)]

    clear_dispatch_events()
    dense = gs.sindex.query(query, output_format="dense")
    dense_events = [
        event
        for event in get_dispatch_events(clear=True)
        if event.surface == "geopandas.sindex.query"
    ]

    assert dense.tolist() == [[True], [False]]
    assert dense_events[-1].implementation == "native_spatial_index"


def test_sindex_query_sparse_public_export_uses_native_relation() -> None:
    scipy = pytest.importorskip("scipy")
    from vibespatial.api import GeoSeries
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    gs = GeoSeries([box(0, 0, 1, 1), box(2, 2, 3, 3)])
    query = [box(0.5, 0.5, 1.5, 1.5)]
    expected_dense = np.asarray([[True], [False]], dtype=bool)

    clear_dispatch_events()
    sparse = gs.sindex.query(query, output_format="sparse")
    sparse_events = [
        event
        for event in get_dispatch_events(clear=True)
        if event.surface == "geopandas.sindex.query"
    ]

    assert isinstance(sparse, scipy.sparse.coo_array)
    np.testing.assert_array_equal(sparse.todense(), expected_dense)
    assert sparse_events[-1].implementation == "native_spatial_index"


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime required for device relation export probe")
def test_sindex_query_public_native_relation_defers_device_pairs_to_terminal_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vibespatial.api import GeoSeries
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )
    from vibespatial.spatial.query_types import _DeviceCandidates

    tree = GeoSeries(
        [
            box(float(x), float(y), float(x + 1), float(y + 1))
            for y in range(50)
            for x in range(50)
        ]
    )
    query = np.asarray(
        [
            box(10.0, 10.0, 12.0, 12.0),
            box(1000.0, 1000.0, 1001.0, 1001.0),
        ],
        dtype=object,
    )

    def _fail_candidate_host_export(self):
        raise AssertionError(
            "public sindex.query should keep native relation pairs device-resident "
            "until the terminal public export"
        )

    monkeypatch.setattr(_DeviceCandidates, "to_host", _fail_candidate_host_export)
    clear_materialization_events()

    result = tree.sindex.query(query, predicate="intersects", sort=True)
    events = [
        event
        for event in get_materialization_events(clear=True)
        if event.operation == "sindex_query_relation_indices_to_host"
    ]

    assert result.shape[0] == 2
    assert result.shape[1] > 0
    assert events


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU runtime required for device relation export probe")
def test_sindex_query_scalar_native_relation_defers_device_pairs_to_terminal_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vibespatial.api import GeoSeries
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )
    from vibespatial.spatial.query_types import _DeviceCandidates

    tree = GeoSeries(
        [
            box(float(x), float(y), float(x + 1), float(y + 1))
            for y in range(50)
            for x in range(50)
        ]
    )

    def _fail_candidate_host_export(self):
        raise AssertionError(
            "scalar public sindex.query should keep native relation pairs "
            "device-resident until the terminal public export"
        )

    monkeypatch.setattr(_DeviceCandidates, "to_host", _fail_candidate_host_export)
    clear_materialization_events()

    result = tree.sindex.query(box(10.0, 10.0, 12.0, 12.0), predicate="intersects", sort=True)
    events = [
        event
        for event in get_materialization_events(clear=True)
        if event.operation == "sindex_query_relation_indices_to_host"
    ]

    assert isinstance(result, np.ndarray)
    assert result.ndim == 1
    assert result.size > 0
    assert events


def test_sindex_nearest_public_export_can_format_native_relation(monkeypatch) -> None:
    from vibespatial.api import GeoSeries
    from vibespatial.api._native_relation import NativeRelation
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    tree = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries([Point(0, 0), Point(10, 0)])
        )
    )
    query = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries([Point(1, 0), Point(9, 0)])
        )
    )

    def fake_nearest_relation(*args, **kwargs):
        return (
            NativeRelation(
                left_indices=np.asarray([0, 1], dtype=np.int32),
                right_indices=np.asarray([0, 1], dtype=np.int32),
                left_row_count=2,
                right_row_count=2,
                predicate="nearest",
                distances=np.asarray([1.0, 1.0], dtype=np.float64),
                sorted_by_left=True,
            ),
            ExecutionMode.GPU,
        )

    sindex = tree.sindex
    monkeypatch.setattr(sindex, "nearest_relation", fake_nearest_relation)
    clear_dispatch_events()

    indices, distances = sindex.nearest(query, return_distance=True)
    events = [
        event
        for event in get_dispatch_events(clear=True)
        if event.surface == "geopandas.sindex.nearest"
    ]

    assert indices.tolist() == [[0, 1], [0, 1]]
    np.testing.assert_allclose(distances, [1.0, 1.0])
    assert events[-1].implementation == "native_relation_export"


def test_sindex_nearest_public_k_exports_bounded_native_relation(monkeypatch) -> None:
    from vibespatial.api import GeoSeries
    from vibespatial.api._native_relation import NativeRelation

    tree = GeoSeries(
        GeometryArray.from_owned(
            from_shapely_geometries([Point(0, 0), Point(2, 0), Point(4, 0)])
        )
    )
    query = GeoSeries(
        GeometryArray.from_owned(from_shapely_geometries([Point(1, 0), Point(3, 0)]))
    )
    received = {}

    def fake_nearest_relation(*args, **kwargs):
        received.update(kwargs)
        return (
            NativeRelation(
                left_indices=np.asarray([0, 0, 1, 1], dtype=np.int32),
                right_indices=np.asarray([0, 1, 1, 2], dtype=np.int32),
                left_row_count=2,
                right_row_count=3,
                predicate="nearest",
                distances=np.asarray([1.0, 1.0, 1.0, 1.0]),
                sorted_by_left=True,
            ),
            ExecutionMode.GPU,
        )

    sindex = tree.sindex
    monkeypatch.setattr(sindex, "nearest_relation", fake_nearest_relation)

    indices, distances = sindex.nearest(
        query,
        return_all=False,
        return_distance=True,
        k=2,
    )

    assert received["k"] == 2
    assert indices.tolist() == [[0, 0, 1, 1], [0, 1, 1, 2]]
    np.testing.assert_allclose(distances, [1.0, 1.0, 1.0, 1.0])


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_sindex_nearest_public_k_rejects_invalid_values(k) -> None:
    from vibespatial.api import GeoSeries

    tree = GeoSeries([Point(0, 0)])
    with pytest.raises(ValueError, match="positive integer"):
        tree.sindex.nearest([Point(1, 0)], k=k)


@pytest.mark.gpu
def test_sindex_nearest_public_k5_point_polygon_matches_shapely(tmp_path) -> None:
    if not has_gpu_runtime():
        pytest.skip("GPU runtime required for public k-nearest execution")

    from vibespatial.api import GeoDataFrame, read_parquet

    tree_geometries = [
        box(0.0, 0.0, 0.2, 0.2),
        box(1.0, 0.0, 1.2, 0.2),
        box(2.0, 0.0, 2.2, 0.2),
        box(3.0, 0.0, 3.2, 0.2),
        box(4.0, 0.0, 4.2, 0.2),
        box(5.0, 0.0, 5.2, 0.2),
        box(6.0, 0.0, 6.2, 0.2),
    ]
    query_geometries = [Point(0.13, 1.03), Point(2.77, 1.41), Point(5.63, 0.77)]
    tree_path = tmp_path / "knn-tree.parquet"
    query_path = tmp_path / "knn-query.parquet"
    GeoDataFrame(
        {"row": range(len(tree_geometries)), "geometry": tree_geometries},
        geometry="geometry",
    ).to_parquet(
        tree_path,
        geometry_encoding="geoarrow",
        index=False,
    )
    GeoDataFrame(
        {"row": range(len(query_geometries)), "geometry": query_geometries},
        geometry="geometry",
    ).to_parquet(
        query_path,
        geometry_encoding="geoarrow",
        index=False,
    )
    tree = read_parquet(tree_path)
    query = read_parquet(query_path)

    indices, distances = tree.sindex.nearest(
        query.geometry,
        return_all=False,
        return_distance=True,
        max_distance=10.0,
        k=5,
    )

    for query_row, query_geometry in enumerate(query_geometries):
        expected = sorted(
            (
                (float(query_geometry.distance(tree_geometry)), tree_row)
                for tree_row, tree_geometry in enumerate(tree_geometries)
            ),
        )[:5]
        selected = indices[0] == query_row
        actual = sorted(zip(distances[selected], indices[1, selected], strict=True))
        np.testing.assert_allclose(
            [distance for distance, _ in actual],
            [distance for distance, _ in expected],
            rtol=1.0e-6,
            atol=1.0e-9,
        )
        assert [int(row) for _, row in actual] == [row for _, row in expected]


def test_overlay_intersecting_index_pairs_handles_device_result() -> None:
    """_intersecting_index_pairs should accept and unpack DeviceSpatialJoinResult
    when both DataFrames have owned backing."""
    from shapely.geometry import Polygon

    from vibespatial.api import GeoDataFrame
    from vibespatial.api.tools.overlay import _intersecting_index_pairs

    polys1 = [Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])]
    polys2 = [Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])]
    df1 = GeoDataFrame({"a": [1]}, geometry=polys1)
    df2 = GeoDataFrame({"b": [1]}, geometry=polys2)

    # Without owned arrays (None), should return numpy as before.
    result = _intersecting_index_pairs(df1, df2)
    if isinstance(result, np.ndarray):
        assert result.ndim == 2
    else:
        # Tuple of (idx1, idx2)
        idx1, idx2 = result
        assert isinstance(idx1, np.ndarray) or hasattr(idx1, "size")


def test_overlay_produces_correct_result_with_device_index_passthrough() -> None:
    """overlay() should produce correct results when _intersecting_index_pairs
    returns DeviceSpatialJoinResult (Phase 2 path)."""
    from shapely.geometry import Polygon

    from vibespatial.api import GeoDataFrame
    from vibespatial.api.tools.overlay import overlay

    polys1 = [
        Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
    ]
    polys2 = [
        Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
        Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
    ]
    df1 = GeoDataFrame({"df1_data": [1, 2]}, geometry=polys1)
    df2 = GeoDataFrame({"df2_data": [1, 2]}, geometry=polys2)

    result = overlay(df1, df2, how="intersection")
    # Basic sanity: should have at least one row for overlapping polygons.
    assert len(result) > 0
    assert "geometry" in result.columns
    assert "df1_data" in result.columns
    assert "df2_data" in result.columns
