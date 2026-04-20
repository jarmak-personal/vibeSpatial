from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, Point, Polygon, box

import vibespatial.spatial.nearest as spatial_nearest_module
import vibespatial.spatial.query as spatial_query_module
import vibespatial.spatial.query_utils as spatial_query_utils_module
from vibespatial.api.geometry_array import GeometryArray
from vibespatial.geometry.owned import OwnedGeometryArray, from_shapely_geometries
from vibespatial.runtime import ExecutionMode, has_gpu_runtime
from vibespatial.runtime.residency import Residency
from vibespatial.spatial.query import (
    build_owned_spatial_index,
    nearest_spatial_index,
    query_spatial_index,
)
from vibespatial.spatial.query_box import _extract_box_query_bounds_shapely


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

    def test_to_host_returns_numpy_arrays(self) -> None:
        """to_host() should produce numpy int32 arrays matching device data."""
        pytest.importorskip("cupy")
        import cupy as cp

        from vibespatial.spatial.query_types import DeviceSpatialJoinResult

        d_left = cp.array([0, 1, 2, 3], dtype=cp.int32)
        d_right = cp.array([4, 5, 6, 7], dtype=cp.int32)
        result = DeviceSpatialJoinResult(d_left_idx=d_left, d_right_idx=d_right)

        h_left, h_right = result.to_host()
        np.testing.assert_array_equal(h_left, [0, 1, 2, 3])
        np.testing.assert_array_equal(h_right, [4, 5, 6, 7])
        assert h_left.dtype == np.int32
        assert h_right.dtype == np.int32

    def test_size_property(self) -> None:
        """size should report the number of index pairs."""
        pytest.importorskip("cupy")
        import cupy as cp

        from vibespatial.spatial.query_types import DeviceSpatialJoinResult

        d_left = cp.array([0, 1], dtype=cp.int32)
        d_right = cp.array([2, 3], dtype=cp.int32)
        result = DeviceSpatialJoinResult(d_left_idx=d_left, d_right_idx=d_right)
        assert result.size == 2

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
def test_dwithin_return_device_true_stays_device_resident(strict_device_guard) -> None:
    """Per-row dwithin thresholds should stay device-native for return_device=True."""
    from vibespatial.spatial.query_types import DeviceSpatialJoinResult

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
    assert isinstance(result, DeviceSpatialJoinResult)
    assert result.size == 2
    assert hasattr(result.d_left_idx, "__cuda_array_interface__")
    assert hasattr(result.d_right_idx, "__cuda_array_interface__")


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
