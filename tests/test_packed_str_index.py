"""All-family nearest contracts for packed STR and the public sindex boundary."""

from __future__ import annotations

import numpy as np
import pytest
import shapely

from vibespatial.runtime import has_gpu_runtime

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required")]


def assert_nearest(actual, expected):
    ai, ad = actual
    ei, ed = expected
    ao, eo = np.lexsort(ai[::-1]), np.lexsort(ei[::-1])
    np.testing.assert_array_equal(ai[:, ao], ei[:, eo])
    np.testing.assert_allclose(ad[ao], ed[eo], rtol=1e-10, atol=1e-9)
    np.testing.assert_array_equal(ad[ao] == 0, ed[eo] == 0)


def geometries():
    return np.array(
        [
            None,
            shapely.Point(),
            shapely.Point(0, 0),
            shapely.LineString([(1, -2), (1, 2)]),
            shapely.box(3, 0, 5, 2),
            shapely.MultiPoint([(8, 1), (8, 3)]),
            shapely.MultiLineString([[(11, 0), (11, 2)], [(12, 0), (12, 2)]]),
            shapely.MultiPolygon([shapely.box(15, 0, 17, 2), shapely.box(19, 0, 21, 2)]),
            shapely.Polygon(
                [(23, 0), (33, 0), (33, 10), (23, 10)], holes=[[(27, 4), (29, 4), (29, 6), (27, 6)]]
            ),
        ],
        dtype=object,
    )


def build(tree, precision):
    from vibespatial.kernels.spatial.packed_str_index import packed_str_index
    from vibespatial.runtime._runtime import select_runtime
    from vibespatial.runtime.precision import KernelClass, select_precision_plan
    from vibespatial.testing import build_owned

    plan = select_precision_plan(
        runtime_selection=select_runtime("gpu"),
        kernel_class=KernelClass.COARSE,
        requested=precision,
    )
    return packed_str_index(build_owned(tree), precision_plan=plan)


def export(relation):
    import cupy as cp

    return cp.asnumpy(cp.stack((relation.left_indices, relation.right_indices))), cp.asnumpy(
        relation.distances
    )


@pytest.mark.parametrize("keys_dtype,values_dtype", [("uint32", "int32"), ("int64", "int64")])
def test_str_sort_primitives_use_warmed_radix_without_losing_high_bits(keys_dtype, values_dtype):
    import cupy as cp

    from vibespatial.cuda.cccl_precompile import get_or_request_compiled
    from vibespatial.cuda.cccl_primitives import PairSortStrategy, sort_pairs

    name = "radix_sort_u32_i32" if keys_dtype == "uint32" else "radix_sort_i64_i64"
    assert get_or_request_compiled(name) is not None
    keys = cp.asarray([2**32 - 1, 0, 2**31], dtype=keys_dtype)
    base = 2**40 if values_dtype == "int64" else 0
    values = cp.asarray([base, base + 1, base + 2], dtype=values_dtype)
    result = sort_pairs(keys, values)
    assert result.strategy is PairSortStrategy.RADIX
    np.testing.assert_array_equal(cp.asnumpy(result.keys), [0, 2**31, 2**32 - 1])
    np.testing.assert_array_equal(cp.asnumpy(result.values), [base + 1, base + 2, base])


@pytest.mark.parametrize("precision", ["fp32", "fp64"])
@pytest.mark.parametrize("translation", [0.0, 1e7])
@pytest.mark.parametrize("maximum", [None, 0.5, 1.0])
def test_mixed_nearest(precision, translation, maximum):
    from vibespatial.testing import build_owned

    tree = shapely.transform(geometries(), lambda xy: xy + translation)
    query = shapely.transform(geometries(), lambda xy: xy + translation + 0.75)
    expected = shapely.STRtree(tree).query_nearest(
        query, max_distance=maximum, return_distance=True
    )
    index = build(tree, precision)
    relation = index.query_relation(build_owned(query), max_distance=maximum)
    assert relation.sorted_by_left
    assert_nearest(export(relation), expected)


@pytest.mark.parametrize("return_all", [True, False])
@pytest.mark.parametrize("exclusive", [True, False])
def test_public_nearest_exclusion_and_ties(return_all, exclusive):
    import vibespatial as vs
    from vibespatial.runtime._runtime import set_requested_mode

    line = shapely.LineString([(0, 0), (2, 0)])
    tree = np.array(
        [
            line,
            shapely.reverse(line),
            shapely.LineString([(0, 0), (1, 0), (2, 0)]),
            shapely.LineString([(0, 1), (2, 1)]),
            shapely.LineString([(0, -1), (2, -1)]),
            shapely.box(-1, -1, 3, 1),
        ],
        dtype=object,
    )
    query = np.array([line, shapely.box(-1, -1, 3, 1), None, shapely.Point()], dtype=object)
    expected = shapely.STRtree(tree).query_nearest(query, exclusive=exclusive, return_distance=True)
    with set_requested_mode("gpu"):
        index = vs.GeoSeries(tree).sindex
        result = index.nearest(
            query, exclusive=exclusive, return_all=return_all, return_distance=True
        )
        native = index._native_spatial_index
        assert native is not None
        assert len(native.backend_cache) == 1
        assert len(index.nearest(shapely.Point(50, 0), return_distance=True)[1]) == 1
        assert index._native_spatial_index is native
    if return_all:
        assert_nearest(result, expected)
    else:
        assert len(result[1]) == 2
        assert set(map(tuple, result[0].T)) <= set(map(tuple, expected[0].T))
        np.testing.assert_array_equal(result[1], 0.0)


@pytest.mark.parametrize("precision", ["fp32", "fp64"])
def test_ties_across_waves_and_empty(precision):
    from vibespatial.testing import build_owned

    tree = shapely.box(np.zeros(129), np.zeros(129), np.ones(129), np.ones(129))
    query = shapely.points([0.5, 0.5], [0.5, 0.5])
    index = build(tree, precision)
    actual = export(index.query_relation(build_owned(query)))
    assert_nearest(actual, shapely.STRtree(tree).query_nearest(query, return_distance=True))
    assert export(index.query_relation(build_owned(query), return_all=False))[0].shape == (2, 2)
    assert export(build([], precision).query_relation(build_owned(query)))[0].shape == (2, 0)
    assert export(index.query_relation(build_owned([])))[0].shape == (2, 0)


def test_cached_tree_across_streams_and_mutation():
    import cupy as cp

    import vibespatial as vs
    from vibespatial.runtime._runtime import set_requested_mode

    tree = shapely.box([0.0, 5.0, 10.0], 0.0, [1.0, 6.0, 11.0], 1.0)
    query = shapely.points([2.0, 8.0], [0.5, 0.5])
    with set_requested_mode("gpu"):
        series = vs.GeoSeries(tree)
        with cp.cuda.Stream(non_blocking=True):
            index = series.sindex
            first = index.nearest(query, return_distance=True)
        with cp.cuda.Stream(non_blocking=True):
            second = index.nearest(query, return_distance=True)
        assert_nearest(first, second)
        series.iloc[0] = shapely.box(2.0, 0.0, 3.0, 1.0)
        updated = series.sindex.nearest(query, return_distance=True)
    tree[0] = shapely.box(2.0, 0.0, 3.0, 1.0)
    assert_nearest(updated, shapely.STRtree(tree).query_nearest(query, return_distance=True))


def test_explicit_cpu_nearest_does_not_build_gpu_backend():
    import vibespatial as vs
    from vibespatial.runtime._runtime import set_requested_mode

    with set_requested_mode("cpu"):
        series = vs.GeoSeries(geometries())
        result = series.sindex.nearest([shapely.Point(7, 0)], return_distance=True)
        assert series.sindex._native_spatial_index is None
    assert_nearest(
        result,
        shapely.STRtree(geometries()).query_nearest([shapely.Point(7, 0)], return_distance=True),
    )


@pytest.mark.parametrize(
    "tree_family", ["point", "line", "polygon", "multipoint", "multiline", "multipolygon"]
)
@pytest.mark.parametrize(
    "query_family", ["point", "line", "polygon", "multipoint", "multiline", "multipolygon"]
)
def test_public_all_family_pairs(tree_family, query_family, monkeypatch):
    import vibespatial as vs
    from tests.test_generalized_strtree_experiment import family_geometries
    from vibespatial.geometry.owned import OwnedGeometryArray
    from vibespatial.runtime._runtime import set_requested_mode

    tree = family_geometries(tree_family)
    query = family_geometries(query_family, 0.7)
    expected = shapely.STRtree(tree).query_nearest(query, return_distance=True)
    with set_requested_mode("gpu"):
        values = vs.GeoSeries.from_wkb(shapely.to_wkb(tree))
        queries = vs.GeoSeries.from_wkb(shapely.to_wkb(query))

        def forbid_geometry_export(*args, **kwargs):
            raise AssertionError("native nearest must not materialize Shapely geometries")

        monkeypatch.setattr(OwnedGeometryArray, "to_shapely", forbid_geometry_export)
        result = values.sindex.nearest(queries, return_distance=True)
        assert values.sindex.backend_info["packed-str"]["cached"]
        assert not values.sindex.backend_info["strtree-host"]["cached"]
    assert_nearest(result, expected)


def test_indexed_views_preserve_logical_duplicate_rows():
    import cupy as cp

    from vibespatial.kernels.spatial.packed_str_index import packed_str_index
    from vibespatial.testing import build_owned

    source = geometries()
    rows = np.array([8, 3, 8, 0, 6], dtype=np.int32)
    queries = np.array([6, 4, 6], dtype=np.int32)
    owned = build_owned(source)
    tree = owned.device_take(cp.asarray(rows))
    query = owned.device_take(cp.asarray(queries))
    index = packed_str_index(tree, precision_plan=build([], "fp32").precision_plan)
    assert_nearest(
        export(index.query_relation(query)),
        shapely.STRtree(source[rows]).query_nearest(source[queries], return_distance=True),
    )


def test_single_match_chooses_lowest_tree_id_across_waves():
    from vibespatial.testing import build_owned

    tree = shapely.points(np.tile([-1.0, 1.0], 65), np.zeros(130))
    query = shapely.points([0.0], [0.0])
    result = export(build(tree, "fp32").query_relation(build_owned(query), return_all=False))
    np.testing.assert_array_equal(result[0], [[0], [0]])
    np.testing.assert_array_equal(result[1], [1.0])


def test_exclusive_all_equal_has_no_matches():
    from vibespatial.testing import build_owned

    tree = shapely.linestrings(np.tile([[[0.0, 0.0], [1.0, 0.0]]], (17, 1, 1)))
    query = tree[:1]
    result = export(build(tree, "fp32").query_relation(build_owned(query), exclusive=True))
    assert result[0].shape == (2, 0)


def test_backend_diagnostics_do_not_build_state():
    import vibespatial as vs
    from vibespatial.runtime._runtime import set_requested_mode

    with set_requested_mode("gpu"):
        index = vs.GeoSeries.from_wkb(shapely.to_wkb(geometries())).sindex
        assert not any(row["cached"] for row in index.backend_info.values())
        assert index._native_spatial_index is None


@pytest.mark.parametrize("precision", ["fp32", "fp64"])
def test_nearest_resolves_interior_projection_close_to_shared_endpoint(precision):
    """MA point 806049: world-coordinate reconstruction reversed this ordering."""
    from vibespatial.testing import build_owned

    tree = shapely.linestrings(
        [
            [(215518.2854682623, 934571.2753280042), (215516.12719749726, 934583.2678349018)],
            [(215516.12719749726, 934583.2678349018), (215508.9404225274, 934607.5797440731)],
        ]
    )
    query = shapely.points([[215544.9381540457, 934588.45288826]])
    expected = shapely.STRtree(tree).query_nearest(query, return_distance=True)
    assert expected[0][1].tolist() == [0]
    assert_nearest(export(build(tree, precision).query_relation(build_owned(query))), expected)


@pytest.mark.parametrize("from_wkb", [False, True])
def test_gpu_index_is_lazy_and_counts_only_valid_nonempty_rows(from_wkb):
    import vibespatial as vs
    from vibespatial.runtime._runtime import set_requested_mode

    source = [None, shapely.Point(), shapely.Point(1, 2)]
    with set_requested_mode("gpu"):
        series = vs.GeoSeries.from_wkb(shapely.to_wkb(source)) if from_wkb else vs.GeoSeries(source)
        index = series.sindex
        assert index._tree is None
        assert index.size == len(index) == 1
        assert not index.is_empty
        assert index._tree is None
        assert_nearest(
            index.nearest([shapely.Point(2, 2)], return_distance=True),
            (np.array([[0], [2]]), np.array([1.0])),
        )


def test_index_outlives_public_array_without_a_reference_cycle():
    import gc
    import weakref

    import cupy as cp

    import vibespatial as vs
    from vibespatial.runtime._runtime import set_requested_mode

    with set_requested_mode("gpu"):
        with cp.cuda.Stream(non_blocking=True):
            series = vs.GeoSeries.from_wkb(shapely.to_wkb([shapely.box(0, 0, 1, 1)]))
            index = series.sindex
            array_ref = weakref.ref(series.array)
            first = index.nearest(shapely.Point(2, 0), return_distance=True)
        del series
        assert array_ref() is None
        gc.collect()
        second = index.nearest(shapely.Point(2, 0), return_distance=True)
        assert_nearest(first, second)
        del index
        gc.collect()


@pytest.mark.parametrize("mode", ["cpu", "gpu"])
def test_pair_aggregate_uses_logical_rows_including_nulls_and_empties(mode):
    import vibespatial as vs
    from vibespatial.runtime._runtime import set_requested_mode

    with set_requested_mode(mode):
        left = vs.GeoSeries.from_wkb(shapely.to_wkb([
            shapely.Point(0, 0), None, shapely.Point(), shapely.Point(2, 2),
        ]))
        right = vs.GeoSeries.from_wkb(shapely.to_wkb([
            None, shapely.Point(0, 0), shapely.Point(2, 2), shapely.Point(2, 2),
        ]))
        zones = vs.GeoSeries([shapely.box(-1, -1, 1, 1), shapely.box(1.5, 1.5, 2.5, 2.5)])
        assert left.sindex.size == 2
        assert right.sindex.size == 3
        result = left.sindex.query_pair_aggregate(right.sindex, zones, predicate="contains")
        assert len(result) == 4
        assert result["left_count"].to_numpy().tolist() == [1, 0, 0, 1]
        assert result["right_count"].to_numpy().tolist() == [0, 1, 1, 1]
        assert result["shared_count"].to_numpy().tolist() == [0, 0, 0, 1]


@pytest.mark.parametrize("precision", ["fp32", "fp64"])
@pytest.mark.parametrize("complex_boundary", [False, True])
@pytest.mark.parametrize("tiny", [False, True])
def test_nearest_resolves_nearparallel_and_nonzero_short_segments(precision, complex_boundary, tiny):
    from vibespatial.testing import build_owned

    if tiny:
        query_coords = [(0.0, 0.0), (1e-16, 0.0)]
        tree_coords = [[(2e-16, 0.0), (3e-16, 0.0)], [(-1.5e-16, 0.0), (-1.25e-16, 0.0)]]
    else:
        query_coords = [(0.0, 0.0), (1.0, 0.0)]
        tree_coords = [[(0.0, 1e-8), (1.0, 1e-9)], [(0.0, 5e-9), (1.0, 5e-9)]]
    if complex_boundary:
        tree_coords = [[a] * 64 + [b] for a, b in tree_coords]
    tree = shapely.linestrings(tree_coords)
    query = shapely.linestrings([query_coords])
    expected = shapely.STRtree(tree).query_nearest(query, return_distance=True)
    assert expected[0][1].tolist() == [0]
    index = build(tree, precision)
    actual = export(index.query_relation(build_owned(query)))
    assert (index._segment_refiner is not None) == complex_boundary
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-12, atol=0)


@pytest.mark.parametrize("from_wkb", [False, True])
def test_nearest_admission_never_prepares_flat_backend_in_either_direction(from_wkb, monkeypatch):
    import gc

    import vibespatial as vs
    from vibespatial.cuda._runtime import get_d2h_transfer_events
    from vibespatial.runtime._runtime import set_requested_mode
    from vibespatial.spatial import indexing

    left = shapely.points([0.25, 5.25, 10.25], [0.0, 1.0, 2.0])
    right = shapely.points([0.0, 5.0, 10.0], [0.0, 1.0, 2.0])

    def forbid_flat(*args, **kwargs):
        raise AssertionError("nearest-only input must not construct Morton state or total bounds")

    with set_requested_mode("gpu"):
        lhs, rhs = (
            [vs.GeoSeries.from_wkb(shapely.to_wkb(rows)) for rows in (left, right)]
            if from_wkb else [vs.GeoSeries(rows) for rows in (left, right)]
        )
        monkeypatch.setattr(indexing, "build_flat_spatial_index", forbid_flat)
        monkeypatch.setattr(indexing, "_device_total_bounds", forbid_flat)
        for query, tree, query_host, tree_host in ((lhs, rhs, left, right), (rhs, lhs, right, left)):
            get_d2h_transfer_events(clear=True)
            relation, mode = tree.sindex.nearest_relation(query)
            assert mode.value == "gpu"
            events = get_d2h_transfer_events(clear=True)
            assert sum(event.bytes_transferred for event in events) <= 14
            assert all("STR" in event.reason or "str" in event.reason for event in events)
            assert_nearest(export(relation), shapely.STRtree(tree_host).query_nearest(query_host, return_distance=True))
            assert tree.values._owned_flat_sindex is None
            assert tree.sindex._native_spatial_index.kind == "geometry-bounds"
            assert not tree.sindex.backend_info["flat-morton"]["cached"]
            assert tree.sindex.backend_info["packed-str"]["cached"]
        orphan = rhs.sindex
        native = orphan._native_spatial_index
        del rhs, query, tree
        gc.collect()
        assert_nearest(orphan.nearest(lhs, return_distance=True),
                       shapely.STRtree(right).query_nearest(left, return_distance=True))
        assert orphan._native_spatial_index is native
        assert orphan._geometry_array._owned_flat_sindex is None


def test_nearest_flat_promotion_retains_backend_and_preserves_predicate_and_fixed_k():
    import cupy as cp

    import vibespatial as vs
    from vibespatial.runtime._runtime import set_requested_mode

    tree = shapely.points([0.0, 4.0, 9.0], [0.0, 0.0, 0.0])
    query = shapely.points([1.0, 7.0], [0.0, 0.0])
    with set_requested_mode("gpu"):
        series = vs.GeoSeries.from_wkb(shapely.to_wkb(tree))
        queries = vs.GeoSeries.from_wkb(shapely.to_wkb(query))
        index = series.sindex
        with cp.cuda.Stream(non_blocking=True):
            first = index.nearest(queries, return_distance=True)
        before = index._native_spatial_index
        cache, lock = before.backend_cache, before.backend_lock
        packed = next(iter(cache.values()))
        with cp.cuda.Stream(non_blocking=True):
            relation, _execution = index.query_relation(queries, predicate="dwithin", distance=2.5, sort=True)
        np.testing.assert_array_equal(export_query_pairs(relation),
                                      shapely.STRtree(tree).query_nearest(query))
        promoted = index._native_spatial_index
        assert promoted.kind == "flat-morton"
        assert promoted.order is not None and promoted.morton_keys is not None
        assert promoted.backend_cache is cache and promoted.backend_lock is lock
        assert index.backend_info["flat-morton"]["cached"]
        assert index.backend_info["packed-str"]["cached"]
        pairs, distances = index.nearest(queries, k=2, return_all=False, return_distance=True)
        np.testing.assert_array_equal(pairs, [[0, 0, 1, 1], [0, 1, 2, 1]])
        np.testing.assert_array_equal(distances, [1.0, 3.0, 2.0, 3.0])
        assert_nearest(index.nearest(queries, return_distance=True), first)
        assert any(backend is packed for backend in index._native_spatial_index.backend_cache.values())
        retokened = index._native_spatial_index_for_nearest(series.values.to_owned(), source_token="new-lineage")
        assert retokened.source_token == retokened.metadata.source_token == "new-lineage"
        assert retokened.backend_cache is cache


def test_bounds_native_direct_query_promotes_complete_layout_once(monkeypatch):
    import vibespatial as vs
    from vibespatial.api._native_metadata import NativeSpatialIndex
    from vibespatial.runtime._runtime import set_requested_mode
    from vibespatial.spatial import indexing

    tree = shapely.points([0.0, 4.0, 9.0], [0.0, 0.0, 0.0])
    query = shapely.points([1.0, 7.0], [0.0, 0.0])
    builds = []
    original = indexing.build_flat_spatial_index

    def record_build(*args, **kwargs):
        builds.append(1)
        return original(*args, **kwargs)

    with set_requested_mode("gpu"):
        owned = vs.GeoSeries.from_wkb(shapely.to_wkb(tree)).values.to_owned()
        queries = vs.GeoSeries.from_wkb(shapely.to_wkb(query)).values.to_owned()
        native = NativeSpatialIndex.from_owned_device(owned)
        assert native.cached_flat_index is None
        monkeypatch.setattr(indexing, "build_flat_spatial_index", record_build)
        for _ in range(2):
            relation = native.query_relation(queries, predicate="dwithin", distance=2.5, sort=True)
            np.testing.assert_array_equal(export_query_pairs(relation), shapely.STRtree(tree).query_nearest(query))
        assert builds == [1]
        prepared = native.with_flat_backend()
        assert prepared.order is not None and prepared.morton_keys is not None
        assert prepared.cached_flat_index is native.cached_flat_index


def export_query_pairs(relation):
    import cupy as cp

    from vibespatial.api._native_relation import NativeRelationSelection

    if isinstance(relation, NativeRelationSelection):
        rows = relation.selection.compact_rowset().positions
        return cp.asnumpy(cp.stack((relation.relation.left_indices[rows], relation.relation.right_indices[rows])))
    return cp.asnumpy(cp.stack((relation.left_indices, relation.right_indices)))


@pytest.mark.parametrize("consumer", ["query", "fixed-k"])
def test_native_nearest_promotion_waits_on_bounds_producer_before_flat_build(consumer, monkeypatch):
    import cupy as cp

    import vibespatial as vs
    from vibespatial.runtime._runtime import set_requested_mode
    from vibespatial.spatial import indexing

    class DependencyStream(cp.cuda.Stream):
        def wait_event(self, event):
            waits.append(event)
            return super().wait_event(event)

    waits = []
    tree = shapely.points([0.0, 4.0, 9.0], [0.0, 0.0, 0.0])
    query = shapely.points([1.0, 7.0], [0.0, 0.0])
    with set_requested_mode("gpu"):
        series = vs.GeoSeries.from_wkb(shapely.to_wkb(tree))
        queries = vs.GeoSeries.from_wkb(shapely.to_wkb(query))
        index = series.sindex
        with cp.cuda.Stream(non_blocking=True):
            first, _mode = index.nearest_relation(queries)
        native = index._native_spatial_index
        producer_event = native.readiness.event
        build_flat = indexing.build_flat_spatial_index

        def require_producer_dependency(*args, **kwargs):
            assert producer_event in waits
            return build_flat(*args, **kwargs)

        monkeypatch.setattr(indexing, "build_flat_spatial_index", require_producer_dependency)
        with DependencyStream(non_blocking=True):
            if consumer == "query":
                relation, _execution = index.query_relation(queries, predicate="dwithin", distance=2.5)
            else:
                relation, _mode = index.nearest_relation(queries, k=2, return_all=False)
        # No public result export or explicit synchronization precedes promotion.
        assert producer_event in waits
        assert index._native_spatial_index.backend_cache is native.backend_cache
        assert_nearest(export(first), shapely.STRtree(tree).query_nearest(query, return_distance=True))
        if consumer == "query":
            np.testing.assert_array_equal(export_query_pairs(relation), [[0, 1], [0, 2]])
        else:
            pairs, distances = export(relation)
            np.testing.assert_array_equal(pairs, [[0, 0, 1, 1], [0, 1, 2, 1]])
            np.testing.assert_array_equal(distances, [1.0, 3.0, 2.0, 3.0])
