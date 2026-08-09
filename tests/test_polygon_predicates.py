from __future__ import annotations

import math

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box

from vibespatial import has_gpu_runtime
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.testing import build_owned as _make_owned

pytestmark = pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required")


def _compute_de9im(query_owned, tree_owned, left_idx, right_idx, query_family, tree_family):
    from vibespatial.predicates.polygon import compute_polygon_de9im_gpu
    return compute_polygon_de9im_gpu(
        query_owned, tree_owned,
        np.asarray(left_idx, dtype=np.int32),
        np.asarray(right_idx, dtype=np.int32),
        query_family=query_family,
        tree_family=tree_family,
    )


def _compute_intersects(query_owned, tree_owned, left_idx, right_idx, query_family, tree_family):
    from vibespatial.predicates.polygon import compute_polygonal_intersects_gpu

    return compute_polygonal_intersects_gpu(
        query_owned,
        tree_owned,
        np.asarray(left_idx, dtype=np.int32),
        np.asarray(right_idx, dtype=np.int32),
        query_family=query_family,
        tree_family=tree_family,
    )


def _eval_predicate(masks, predicate):
    from vibespatial.predicates.polygon import evaluate_predicate_from_de9im
    return evaluate_predicate_from_de9im(masks, predicate)


def _eval_predicate_device(masks, predicate):
    import cupy as cp

    from vibespatial.cuda._runtime import get_cuda_runtime
    from vibespatial.predicates.polygon import evaluate_predicate_from_de9im_device

    d_masks = cp.asarray(masks, dtype=cp.uint16)
    d_result = evaluate_predicate_from_de9im_device(d_masks, predicate)
    return get_cuda_runtime().copy_device_to_host(
        d_result,
        reason="test device de9im predicate evaluation terminal export",
        terminal_export=True,
    )


def test_binary_predicate_expression_point_region_orientation_avoids_host_metadata() -> None:
    from vibespatial import Residency, from_shapely_geometries
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.predicates.binary import binary_predicate_expression

    points = from_shapely_geometries(
        [Point(0.5, 0.5), Point(2.0, 2.0)],
        residency=Residency.DEVICE,
    )
    regions = from_shapely_geometries(
        [box(0.0, 0.0, 1.0, 1.0), box(0.0, 0.0, 1.0, 1.0)],
        residency=Residency.DEVICE,
    )
    for owned in (points, regions):
        owned._validity = None
        owned._tags = None
        owned._family_row_offsets = None

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    expression = binary_predicate_expression("covered_by", points, regions)
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert expression is not None
    assert "binary predicate point-region tag-domain summary scalar fence" not in reasons
    assert not any("owned geometry host metadata" in reason for reason in reasons)


def test_multi_predicate_expressions_route_point_region_pairs_to_point_classifier() -> None:
    import cupy as cp

    from vibespatial import Residency, from_shapely_geometries
    from vibespatial.predicates.binary import binary_predicate_expressions

    regions = from_shapely_geometries(
        [box(0.0, 0.0, 2.0, 2.0), box(4.0, 4.0, 6.0, 6.0)],
        residency=Residency.DEVICE,
    )
    points = from_shapely_geometries(
        [Point(1.0, 1.0), Point(8.0, 8.0)],
        residency=Residency.DEVICE,
    )

    expressions = binary_predicate_expressions(
        ("contains", "covers", "intersects"),
        regions,
        points,
    )

    assert expressions is not None
    for predicate in ("contains", "covers", "intersects"):
        assert cp.asarray(expressions[predicate].values).tolist() == [True, False]


def test_point_region_expression_preserves_indexed_polygon_carrier(monkeypatch) -> None:
    import cupy as cp

    from vibespatial import Residency, from_shapely_geometries
    from vibespatial.geometry.owned import OwnedGeometryArray
    from vibespatial.predicates.binary import binary_predicate_expression

    points = from_shapely_geometries(
        [Point(0.5, 0.5), Point(2.5, 0.5)],
        residency=Residency.DEVICE,
    )._device_indexed_take(cp.asarray([0, 1, 0, 1], dtype=cp.int64))
    regions = from_shapely_geometries(
        [box(0.0, 0.0, 1.0, 1.0), box(2.0, 0.0, 3.0, 1.0)],
        residency=Residency.DEVICE,
    )._device_indexed_take(cp.asarray([0, 1, 0, 1], dtype=cp.int64))

    def _fail_physicalize(*_args, **_kwargs):
        raise AssertionError("point-region kernels must consume device row indirection")

    monkeypatch.setattr(
        OwnedGeometryArray,
        "physicalize_device_rows",
        _fail_physicalize,
    )
    expression = binary_predicate_expression("covered_by", points, regions)

    assert expression is not None
    assert cp.asarray(expression.values).tolist() == [True, True, True, True]


def test_binary_predicate_expression_polygon_de9im_skips_point_support_probe() -> None:
    from vibespatial import Residency, from_shapely_geometries
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count
    from vibespatial.predicates.binary import binary_predicate_expression

    left = from_shapely_geometries(
        [box(0.0, 0.0, 2.0, 2.0), box(4.0, 4.0, 6.0, 6.0)],
        residency=Residency.DEVICE,
    )
    right = from_shapely_geometries(
        [box(1.0, 1.0, 3.0, 3.0), box(5.0, 5.0, 7.0, 7.0)],
        residency=Residency.DEVICE,
    )
    for owned in (left, right):
        owned._validity = None
        owned._tags = None
        owned._family_row_offsets = None

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    expression = binary_predicate_expression("intersects", left, right)
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert expression is not None
    assert "binary predicate tag-pairs host export" not in reasons
    assert "binary predicate point-candidate support scalar fence" not in reasons
    assert not any("owned geometry host metadata" in reason for reason in reasons)


def test_indexed_view_device_bounds_preserve_logical_row_order() -> None:
    import cupy as cp

    from vibespatial import Residency, from_shapely_geometries
    from vibespatial.cuda._runtime import get_cuda_runtime
    from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device

    geoms = [
        Polygon([(0, 0), (2, 0), (2, 1), (1, 2), (0, 1), (0, 0)]),
        Polygon([(10, 10), (13, 10), (13, 11), (12, 12), (10, 11), (10, 10)]),
        Polygon([(20, 20), (21, 20), (22, 22), (20, 21), (20, 20)]),
    ]
    owned = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    view = owned.device_take(cp.asarray([2, 0, 1], dtype=cp.int64))

    assert view.is_indexed_view

    d_bounds = compute_geometry_bounds_device(view, preserve_indexed_view=True)
    bounds = get_cuda_runtime().copy_device_to_host(
        d_bounds,
        reason="test indexed-view bounds terminal export",
        terminal_export=True,
    ).reshape(-1, 4)

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
    assert view.is_indexed_view


def test_binary_predicate_expression_de9im_indexed_view_uses_logical_bounds() -> None:
    import cupy as cp

    from vibespatial import Residency, from_shapely_geometries
    from vibespatial.cuda._runtime import (
        get_cuda_runtime,
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )
    from vibespatial.predicates.binary import binary_predicate_expression

    left = from_shapely_geometries(
        [
            Polygon([(0, 0), (2, 0), (2, 1), (1, 2), (0, 1), (0, 0)]),
            Polygon([(10, 10), (13, 10), (13, 11), (12, 12), (10, 11), (10, 10)]),
            Polygon([(20, 20), (21, 20), (22, 22), (20, 21), (20, 20)]),
        ],
        residency=Residency.DEVICE,
    )
    left_view = left.device_take(cp.asarray([2, 0, 1], dtype=cp.int64))
    right = from_shapely_geometries(
        [
            box(19, 19, 23, 23),
            box(-1, -1, 3, 3),
            box(50, 50, 51, 51),
        ],
        residency=Residency.DEVICE,
    )

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    expression = binary_predicate_expression("intersects", left_view, right)
    assert expression is not None
    values = get_cuda_runtime().copy_device_to_host(
        expression.values,
        reason="test indexed-view de9im expression terminal export",
        terminal_export=True,
    )
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert values.tolist() == [True, True, False]
    assert left_view.is_indexed_view
    assert not any("all-valid pair scalar fence" in reason for reason in reasons)
    assert not any("owned geometry device-take" in reason for reason in reasons)
    assert not any("point-region tag-domain" in reason for reason in reasons)
    assert not any("point-candidate support" in reason for reason in reasons)


def test_single_mask_covered_by_preserves_indexed_view_carrier() -> None:
    import cupy as cp

    from vibespatial import Residency, from_shapely_geometries
    from vibespatial.cuda._runtime import (
        get_cuda_runtime,
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )
    from vibespatial.predicates.binary import (
        _evaluate_covered_by_single_polygonal_mask_device,
    )

    left = from_shapely_geometries(
        [
            Polygon([(0, 0), (2, 0), (2, 1), (1, 2), (0, 1), (0, 0)]),
            Polygon([(10, 10), (13, 10), (13, 11), (12, 12), (10, 11), (10, 10)]),
            Polygon([(20, 20), (21, 20), (22, 22), (20, 21), (20, 20)]),
        ],
        residency=Residency.DEVICE,
    )
    left_view = left.device_take(cp.asarray([2, 0, 1], dtype=cp.int64))
    mask = from_shapely_geometries(
        [box(-1, -1, 24, 24)],
        residency=Residency.DEVICE,
    )

    assert left_view.is_indexed_view

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    values_device = _evaluate_covered_by_single_polygonal_mask_device(left_view, mask)
    assert values_device is not None
    values = get_cuda_runtime().copy_device_to_host(
        values_device,
        reason="test single-mask indexed-view terminal export",
        terminal_export=True,
    )
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert values.tolist() == [True, True, True]
    assert left_view.is_indexed_view
    assert not any("owned geometry device-take" in reason for reason in reasons)
    assert (
        "binary predicate covered-by single-mask family-domain scalar fence"
        not in reasons
    )


def test_relation_pair_covered_by_uses_pair_row_polygonal_kernel() -> None:
    import cupy as cp

    import vibespatial
    from vibespatial import Residency, from_shapely_geometries
    from vibespatial.cuda._runtime import get_cuda_runtime
    from vibespatial.predicates.binary import (
        _binary_predicate_relation_pair_values_device,
    )

    left_geoms = [
        box(0.1, 0.1, 0.9, 0.9),
        box(2.2, 0.2, 2.8, 0.8),
        box(4.0, 0.0, 5.0, 1.0),
        box(0.2, 3.2, 0.8, 3.8),
    ]
    right_geoms = [
        box(0.0, 0.0, 1.0, 1.0),
        box(2.0, 0.0, 3.0, 1.0),
        box(10.0, 10.0, 11.0, 11.0),
        box(0.0, 3.0, 1.0, 4.0),
    ]
    left = from_shapely_geometries(left_geoms, residency=Residency.DEVICE)
    right = from_shapely_geometries(right_geoms, residency=Residency.DEVICE)
    d_left = cp.asarray([0, 1, 2, 3], dtype=cp.int32)
    d_right = cp.asarray([0, 1, 2, 3], dtype=cp.int32)

    vibespatial.clear_dispatch_events()
    outputs = _binary_predicate_relation_pair_values_device(
        ("covered_by",),
        left,
        right,
        d_left,
        d_right,
        operation_prefix="test.relation_pair.covered_by",
    )
    events = vibespatial.get_dispatch_events(clear=True)

    assert outputs is not None
    values = get_cuda_runtime().copy_device_to_host(
        outputs["covered_by"],
        reason="test relation-pair covered_by terminal export",
        terminal_export=True,
    )
    assert values.tolist() == [True, True, False, True]
    assert any(
        event.implementation == "relation_pair_covered_by_no_holes_gpu"
        and event.operation == "covered_by"
        for event in events
    )


def test_relation_pair_predicate_preserves_inactive_capacity_slots() -> None:
    import cupy as cp

    from vibespatial import Residency, from_shapely_geometries
    from vibespatial.cuda._runtime import get_cuda_runtime
    from vibespatial.predicates.binary import (
        _binary_predicate_relation_pair_values_device,
    )

    left = from_shapely_geometries(
        [
            box(0.1, 0.1, 0.9, 0.9),
            box(2.1, 0.1, 2.9, 0.9),
            box(8.0, 8.0, 9.0, 9.0),
            box(4.1, 0.1, 4.9, 0.9),
        ],
        residency=Residency.DEVICE,
    )
    right = from_shapely_geometries(
        [
            box(0.0, 0.0, 1.0, 1.0),
            box(2.0, 0.0, 3.0, 1.0),
            box(0.0, 0.0, 1.0, 1.0),
            box(4.0, 0.0, 5.0, 1.0),
        ],
        residency=Residency.DEVICE,
    )
    d_left = cp.asarray([3, 0, 2, 1], dtype=cp.int32)
    d_right = cp.asarray([3, 0, 2, 1], dtype=cp.int32)
    d_active = cp.asarray([True, True, False, False], dtype=cp.bool_)

    outputs = _binary_predicate_relation_pair_values_device(
        ("covered_by",),
        left,
        right,
        d_left,
        d_right,
        d_pair_active=d_active,
    )

    assert outputs is not None
    values = get_cuda_runtime().copy_device_to_host(
        outputs["covered_by"],
        reason="test relation-pair capacity terminal export",
        terminal_export=True,
    )
    assert values.tolist() == [True, True, False, False]


def test_single_right_candidate_predicates_scatter_to_candidate_capacity() -> None:
    import cupy as cp

    from vibespatial import Residency, from_shapely_geometries
    from vibespatial.cuda._runtime import (
        get_cuda_runtime,
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )
    from vibespatial.predicates.binary import (
        _polygonal_single_right_candidate_predicates_device,
    )

    source = from_shapely_geometries(
        [
            box(0.2, 0.2, 0.8, 0.8),
            box(5.0, 5.0, 6.0, 6.0),
            box(1.5, 0.2, 2.5, 0.8),
            MultiPolygon([
                box(0.1, 0.1, 0.3, 0.3),
                box(1.0, 1.0, 1.2, 1.2),
            ]),
            box(-1.0, -1.0, -0.5, -0.5),
        ],
        residency=Residency.DEVICE,
    )
    mask = from_shapely_geometries(
        [box(0.0, 0.0, 2.0, 2.0)],
        residency=Residency.DEVICE,
    )
    d_rows = cp.asarray([2, 0, 3, 1, 4], dtype=cp.int64)
    d_active = cp.asarray([True, True, True, False, False], dtype=cp.bool_)

    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)
    outputs = _polygonal_single_right_candidate_predicates_device(
        ("intersects", "covered_by"),
        source,
        mask,
        d_rows,
        d_candidate_active=d_active,
    )
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert outputs is not None
    intersects = get_cuda_runtime().copy_device_to_host(
        outputs["intersects"],
        reason="test candidate-local intersects terminal export",
        terminal_export=True,
    )
    covered_by = get_cuda_runtime().copy_device_to_host(
        outputs["covered_by"],
        reason="test candidate-local covered-by terminal export",
        terminal_export=True,
    )

    assert intersects.tolist() == [True, True, True, False, False]
    assert covered_by.tolist() == [False, True, True, False, False]
    assert not any("owned geometry device-take" in reason for reason in reasons)


class TestDE9IMBitmask:
    def test_device_predicate_eval_matches_host_for_all_masks(self):
        masks = np.arange(1 << 9, dtype=np.uint16)
        predicates = (
            "intersects",
            "touches",
            "covers",
            "covered_by",
            "contains",
            "within",
            "overlaps",
            "disjoint",
            "contains_properly",
            "equals",
        )
        for predicate in predicates:
            np.testing.assert_array_equal(
                _eval_predicate_device(masks, predicate),
                _eval_predicate(masks, predicate),
            )

    def test_device_transpose_matches_host_for_all_masks(self):
        import cupy as cp

        from vibespatial.cuda._runtime import get_cuda_runtime
        from vibespatial.predicates.polygon import _transpose_de9im, _transpose_de9im_device

        masks = np.arange(1 << 9, dtype=np.uint16)
        d_result = _transpose_de9im_device(cp.asarray(masks, dtype=cp.uint16))
        result = get_cuda_runtime().copy_device_to_host(
            d_result,
            reason="test device de9im transpose terminal export",
            terminal_export=True,
        )
        np.testing.assert_array_equal(result, _transpose_de9im(masks))

    def test_disjoint_boxes(self, make_owned):
        q = [box(0, 0, 1, 1)]
        t = [box(3, 0, 4, 1)]
        masks = _compute_de9im(
            make_owned(q), make_owned(t),
            [0], [0], GeometryFamily.POLYGON, GeometryFamily.POLYGON,
        )
        assert _eval_predicate(masks, "disjoint")[0]
        assert not _eval_predicate(masks, "intersects")[0]

    def test_overlapping_boxes(self, make_owned):
        q = [box(0, 0, 2, 2)]
        t = [box(1, 1, 3, 3)]
        masks = _compute_de9im(
            make_owned(q), make_owned(t),
            [0], [0], GeometryFamily.POLYGON, GeometryFamily.POLYGON,
        )
        assert _eval_predicate(masks, "intersects")[0]
        assert _eval_predicate(masks, "overlaps")[0]
        assert not _eval_predicate(masks, "contains")[0]
        assert not _eval_predicate(masks, "within")[0]
        assert not _eval_predicate(masks, "disjoint")[0]

    def test_contained_box(self, make_owned):
        q = [box(0, 0, 10, 10)]
        t = [box(2, 2, 3, 3)]
        masks = _compute_de9im(
            make_owned(q), make_owned(t),
            [0], [0], GeometryFamily.POLYGON, GeometryFamily.POLYGON,
        )
        assert _eval_predicate(masks, "intersects")[0]
        assert _eval_predicate(masks, "contains")[0]
        assert _eval_predicate(masks, "contains_properly")[0]
        assert not _eval_predicate(masks, "within")[0]
        assert _eval_predicate(masks, "covers")[0]

    def test_within(self, make_owned):
        q = [box(2, 2, 3, 3)]
        t = [box(0, 0, 10, 10)]
        masks = _compute_de9im(
            make_owned(q), make_owned(t),
            [0], [0], GeometryFamily.POLYGON, GeometryFamily.POLYGON,
        )
        assert _eval_predicate(masks, "within")[0]
        assert not _eval_predicate(masks, "contains")[0]
        assert _eval_predicate(masks, "covered_by")[0]

    def test_covered_by_concave_mask_rejects_edge_midpoint_outside(self, make_owned):
        mask = Polygon(
            [
                (
                    500.0 + (200.0 if i % 2 == 0 else 80.0) * math.cos(math.pi * i / 12.0),
                    500.0 + (200.0 if i % 2 == 0 else 80.0) * math.sin(math.pi * i / 12.0),
                )
                for i in range(24)
            ]
        )
        query = box(470.0, 420.0, 480.0, 430.0)
        assert not shapely.covered_by(query, mask)
        masks = _compute_de9im(
            make_owned([query]),
            make_owned([mask]),
            [0],
            [0],
            GeometryFamily.POLYGON,
            GeometryFamily.POLYGON,
        )
        assert not _eval_predicate(masks, "covered_by")[0]

    def test_covered_by_rejects_segment_crossing_between_vertex_samples(self, make_owned):
        mask = shapely.from_wkt(
            "POLYGON ((750 600, 750 590, 752.5019540770263 590, "
            "753.8342407065104 590, 760 590, 760 592.5510137807898, "
            "759.6432890039046 592.9446500112713, 760 593.1254694820173, "
            "760 600, 754.9429865686194 600, 754.5836710928672 "
            "598.5280184635408, 753.2497694866499 600, 750 600))"
        )
        query = shapely.from_wkt(
            "POLYGON ((750 600, 750 594.6239121799581, "
            "758.1213861981413 595.8922999050006, 759.9984988242248 "
            "591.7914812028355, 760 591.7934595572564, 760 600, "
            "750 600))"
        )
        assert not shapely.covered_by(query, mask)
        masks = _compute_de9im(
            make_owned([query]),
            make_owned([mask]),
            [0],
            [0],
            GeometryFamily.POLYGON,
            GeometryFamily.POLYGON,
        )
        assert not _eval_predicate(masks, "covered_by")[0]

    def test_touching_boxes(self, make_owned):
        q = [box(0, 0, 1, 1)]
        t = [box(1, 0, 2, 1)]
        masks = _compute_de9im(
            make_owned(q), make_owned(t),
            [0], [0], GeometryFamily.POLYGON, GeometryFamily.POLYGON,
        )
        assert _eval_predicate(masks, "touches")[0]
        assert _eval_predicate(masks, "intersects")[0]
        assert not _eval_predicate(masks, "overlaps")[0]

    def test_identical_boxes(self, make_owned):
        q = [box(0, 0, 1, 1)]
        t = [box(0, 0, 1, 1)]
        masks = _compute_de9im(
            make_owned(q), make_owned(t),
            [0], [0], GeometryFamily.POLYGON, GeometryFamily.POLYGON,
        )
        assert _eval_predicate(masks, "intersects")[0]
        assert _eval_predicate(masks, "contains")[0]
        assert not _eval_predicate(masks, "contains_properly")[0]
        assert _eval_predicate(masks, "within")[0]
        assert _eval_predicate(masks, "covers")[0]
        assert _eval_predicate(masks, "covered_by")[0]
        assert not _eval_predicate(masks, "disjoint")[0]
        assert not _eval_predicate(masks, "touches")[0]

    def test_reversed_irregular_polygon_has_no_exterior_interval(self, make_owned):
        polygon = Polygon(
            [
                (-73.5541107525234, 45.5091983609661),
                (-73.5546126200639, 45.5086813829106),
                (-73.5540185061397, 45.5084409343852),
                (-73.5539986525799, 45.5084323044531),
                (-73.5535801792994, 45.5089539203786),
            ]
        )
        reversed_polygon = shapely.reverse(polygon)
        masks = _compute_de9im(
            make_owned([polygon]),
            make_owned([reversed_polygon]),
            [0],
            [0],
            GeometryFamily.POLYGON,
            GeometryFamily.POLYGON,
        )
        assert _eval_predicate(masks, "covered_by")[0]
        assert _eval_predicate(masks, "covers")[0]
        assert _eval_predicate(masks, "equals")[0]

    def test_reversed_irregular_multipolygon_single_mask_probe(self, make_owned):
        first = Polygon(
            [
                (-73.5541107525234, 45.5091983609661),
                (-73.5546126200639, 45.5086813829106),
                (-73.5540185061397, 45.5084409343852),
                (-73.5539986525799, 45.5084323044531),
                (-73.5535801792994, 45.5089539203786),
            ]
        )
        second = Polygon(
            [
                (-73.5542465586147, 45.5081555487952),
                (-73.5540185061397, 45.5084409343852),
                (-73.5546126200639, 45.5086813829106),
                (-73.5548825850032, 45.5084033554357),
            ]
        )
        query = MultiPolygon([first, second])
        mask = shapely.reverse(query)

        from vibespatial.predicates.polygon import (
            compute_polygonal_covered_by_single_mask_no_holes_gpu,
        )

        result = compute_polygonal_covered_by_single_mask_no_holes_gpu(
            make_owned([query]),
            make_owned([mask]),
            np.asarray([0], dtype=np.int32),
            query_family=GeometryFamily.MULTIPOLYGON,
            mask_family=GeometryFamily.MULTIPOLYGON,
        )
        assert result.tolist() == [True]

    def test_polygon_with_hole(self, make_owned):
        outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
        q = [Polygon(outer, [hole])]
        t = [box(4, 4, 6, 6)]  # inside the hole
        masks = _compute_de9im(
            make_owned(q), make_owned(t),
            [0], [0], GeometryFamily.POLYGON, GeometryFamily.POLYGON,
        )
        assert _eval_predicate(masks, "disjoint")[0]
        assert not _eval_predicate(masks, "intersects")[0]


class TestMultiPolygonDE9IM:
    def test_mpg_mpg_overlapping(self, make_owned):
        q = [MultiPolygon([box(0, 0, 2, 2), box(10, 10, 12, 12)])]
        t = [MultiPolygon([box(1, 1, 3, 3)])]
        masks = _compute_de9im(
            make_owned(q), make_owned(t),
            [0], [0], GeometryFamily.MULTIPOLYGON, GeometryFamily.MULTIPOLYGON,
        )
        assert _eval_predicate(masks, "intersects")[0]

    def test_pg_mpg_contained(self, make_owned):
        q = [box(0, 0, 10, 10)]
        t = [MultiPolygon([box(1, 1, 2, 2), box(3, 3, 4, 4)])]
        masks = _compute_de9im(
            make_owned(q), make_owned(t),
            [0], [0], GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON,
        )
        assert _eval_predicate(masks, "contains")[0]

    def test_mpg_pg_swap(self, make_owned):
        """MPG × PG swap: within should work correctly after transpose."""
        q = [MultiPolygon([box(1, 1, 2, 2)])]
        t = [box(0, 0, 10, 10)]
        masks = _compute_de9im(
            make_owned(q), make_owned(t),
            [0], [0], GeometryFamily.MULTIPOLYGON, GeometryFamily.POLYGON,
        )
        assert _eval_predicate(masks, "within")[0]
        assert not _eval_predicate(masks, "contains")[0]


class TestBatchPredicates:
    """Test multiple candidate pairs in a single kernel launch."""

    def test_batch_intersects(self, make_owned):
        polys = [box(0, 0, 1, 1), box(2, 2, 3, 3), box(0, 0, 5, 5), box(10, 10, 11, 11)]
        owned = make_owned(polys)

        left = np.array([0, 0, 2, 3], dtype=np.int32)
        right = np.array([1, 2, 3, 0], dtype=np.int32)

        masks = _compute_de9im(
            owned, owned, left, right,
            GeometryFamily.POLYGON, GeometryFamily.POLYGON,
        )
        gpu_result = _eval_predicate(masks, "intersects")

        expected = np.array([
            shapely.intersects(polys[li], polys[ri])
            for li, ri in zip(left, right)
        ])
        np.testing.assert_array_equal(gpu_result, expected)

    def test_batch_intersects_boolean_kernel(self, make_owned):
        polys = [
            box(0, 0, 1, 1),
            box(2, 2, 3, 3),
            box(0, 0, 5, 5),
            box(10, 10, 11, 11),
            Polygon(
                [(0, 0), (10, 0), (10, 10), (0, 10)],
                [[(3, 3), (7, 3), (7, 7), (3, 7)]],
            ),
            box(4, 4, 6, 6),
        ]
        owned = make_owned(polys)

        left = np.array([0, 0, 2, 3, 4, 2], dtype=np.int32)
        right = np.array([1, 2, 3, 0, 5, 5], dtype=np.int32)

        gpu_result = _compute_intersects(
            owned,
            owned,
            left,
            right,
            GeometryFamily.POLYGON,
            GeometryFamily.POLYGON,
        )
        expected = np.array(
            [shapely.intersects(polys[li], polys[ri]) for li, ri in zip(left, right)],
            dtype=bool,
        )
        np.testing.assert_array_equal(gpu_result, expected)

    def test_multipolygon_intersects_boolean_kernel(self, make_owned):
        query = [
            MultiPolygon([box(0, 0, 1, 1), box(5, 5, 6, 6)]),
            MultiPolygon([box(10, 10, 11, 11)]),
        ]
        tree = [
            box(0.5, 0.5, 2, 2),
            box(3, 3, 4, 4),
        ]

        gpu_result = _compute_intersects(
            make_owned(query),
            make_owned(tree),
            [0, 0, 1],
            [0, 1, 0],
            GeometryFamily.MULTIPOLYGON,
            GeometryFamily.POLYGON,
        )
        expected = np.array(
            [
                shapely.intersects(query[0], tree[0]),
                shapely.intersects(query[0], tree[1]),
                shapely.intersects(query[1], tree[0]),
            ],
            dtype=bool,
        )
        np.testing.assert_array_equal(gpu_result, expected)


class TestQueryPipelineIntegration:
    """Test that the query pipeline uses GPU DE-9IM for polygon predicates."""

    def test_polygon_query_intersects(self):
        import geopandas

        polys = [box(0, 0, 2, 2), box(1, 1, 3, 3), box(5, 5, 7, 7), box(10, 10, 12, 12)]
        series = geopandas.GeoSeries(polys)

        query_geom = [box(0, 0, 1.5, 1.5)]
        query_series = geopandas.GeoSeries(query_geom)

        result = series.sindex.query(query_series, predicate="intersects")

        # Query box overlaps polys[0] and polys[1] only.
        result_right = set(result[1]) if result.ndim == 2 else set(result)
        expected = {i for i, p in enumerate(polys) if shapely.intersects(query_geom[0], p)}
        assert result_right == expected

    def test_polygon_self_query_intersects(self):
        """Self-join: all polygon pairs — the 100K benchmark workload shape."""
        import geopandas

        polys = [box(i, 0, i + 1.5, 1.5) for i in range(10)]
        series = geopandas.GeoSeries(polys)

        result = series.sindex.query(series, predicate="intersects")

        # Verify all result pairs actually intersect.
        for col in range(result.shape[1]):
            qi, ti = result[0, col], result[1, col]
            assert shapely.intersects(polys[qi], polys[ti]), f"pair ({qi}, {ti}) should intersect"

        # Verify no intersecting pairs are missing (self-join returns upper triangle only).
        result_set = set(zip(result[0], result[1]))
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                if shapely.intersects(polys[i], polys[j]):
                    has_pair = (i, j) in result_set or (j, i) in result_set
                    assert has_pair, f"missing intersecting pair ({i}, {j})"

    def test_polygon_query_contains(self):
        import geopandas

        polys = [box(0, 0, 10, 10), box(1, 1, 2, 2), box(20, 20, 21, 21)]
        series = geopandas.GeoSeries(polys)

        query_geom = [box(0, 0, 10, 10)]
        query_series = geopandas.GeoSeries(query_geom)

        result = series.sindex.query(query_series, predicate="contains")

        result_right = set(result[1]) if result.ndim == 2 else set(result)
        expected = {i for i, p in enumerate(polys) if shapely.contains(query_geom[0], p)}
        assert result_right == expected


class TestLineDE9IM:
    """Test DE-9IM computation for line × polygon and line × line pairs."""

    def test_line_crossing_polygon(self):
        from shapely.geometry import LineString
        q = [LineString([(0, 0.5), (2, 0.5)])]
        t = [box(0.5, 0, 1.5, 1)]
        masks = _compute_de9im(
            _make_owned(q), _make_owned(t),
            [0], [0], GeometryFamily.LINESTRING, GeometryFamily.POLYGON,
        )
        assert _eval_predicate(masks, "intersects")[0]
        assert not _eval_predicate(masks, "within")[0]
        assert not _eval_predicate(masks, "disjoint")[0]

    def test_line_inside_polygon(self):
        from shapely.geometry import LineString
        q = [LineString([(0.6, 0.5), (1.0, 0.5)])]
        t = [box(0.5, 0, 1.5, 1)]
        masks = _compute_de9im(
            _make_owned(q), _make_owned(t),
            [0], [0], GeometryFamily.LINESTRING, GeometryFamily.POLYGON,
        )
        assert _eval_predicate(masks, "intersects")[0]
        assert _eval_predicate(masks, "within")[0]
        assert not _eval_predicate(masks, "disjoint")[0]

    def test_line_inside_polygon_hole_is_not_covered(self):
        polygon = Polygon(
            [(1, -1), (5, -1), (5, 1), (1, 1), (1, -1)],
            [[(2, -0.5), (4, -0.5), (4, 0.5), (2, 0.5), (2, -0.5)]],
        )
        line = LineString([(2, 0), (4, 0)])
        masks = _compute_de9im(
            _make_owned([line]),
            _make_owned([polygon]),
            [0],
            [0],
            GeometryFamily.LINESTRING,
            GeometryFamily.POLYGON,
        )
        assert not _eval_predicate(masks, "covered_by")[0]
        assert _eval_predicate(masks, "intersects")[0]
        assert _eval_predicate(masks, "touches")[0]

    def test_line_disjoint_from_polygon(self):
        from shapely.geometry import LineString
        q = [LineString([(3, 0), (4, 0)])]
        t = [box(0.5, 0, 1.5, 1)]
        masks = _compute_de9im(
            _make_owned(q), _make_owned(t),
            [0], [0], GeometryFamily.LINESTRING, GeometryFamily.POLYGON,
        )
        assert not _eval_predicate(masks, "intersects")[0]
        assert _eval_predicate(masks, "disjoint")[0]

    def test_line_touching_polygon_boundary(self):
        q = [LineString([(0.5, 1), (0.5, 2)])]
        t = [box(0.5, 0, 1.5, 1)]
        masks = _compute_de9im(
            _make_owned(q), _make_owned(t),
            [0], [0], GeometryFamily.LINESTRING, GeometryFamily.POLYGON,
        )
        assert _eval_predicate(masks, "intersects")[0]
        assert _eval_predicate(masks, "touches")[0]
        assert not _eval_predicate(masks, "within")[0]

    def test_line_intersects_tiny_buffer_polygon_at_large_coordinates(self):
        q = [LineString([(970227.216003418, 145641.63360595703), (970273.9365844727, 145641.63360595703)])]
        t = [Point(970264.7347596437, 145641.63360595703).buffer(1e-8)]
        masks = _compute_de9im(
            _make_owned(q), _make_owned(t),
            [0], [0], GeometryFamily.LINESTRING, GeometryFamily.POLYGON,
        )
        assert _eval_predicate(masks, "intersects")[0]
        assert not _eval_predicate(masks, "disjoint")[0]

    def test_crossing_lines(self):
        q = [LineString([(0, 0), (2, 2)])]
        t = [LineString([(0, 2), (2, 0)])]
        masks = _compute_de9im(
            _make_owned(q), _make_owned(t),
            [0], [0], GeometryFamily.LINESTRING, GeometryFamily.LINESTRING,
        )
        assert _eval_predicate(masks, "intersects")[0]
        assert not _eval_predicate(masks, "disjoint")[0]

    def test_parallel_lines(self):
        from shapely.geometry import LineString
        q = [LineString([(0, 0), (1, 0)])]
        t = [LineString([(0, 1), (1, 1)])]
        masks = _compute_de9im(
            _make_owned(q), _make_owned(t),
            [0], [0], GeometryFamily.LINESTRING, GeometryFamily.LINESTRING,
        )
        assert not _eval_predicate(masks, "intersects")[0]
        assert _eval_predicate(masks, "disjoint")[0]

    def test_collinear_overlapping_lines(self):
        from shapely.geometry import LineString
        q = [LineString([(0, 0), (2, 0)])]
        t = [LineString([(1, 0), (3, 0)])]
        masks = _compute_de9im(
            _make_owned(q), _make_owned(t),
            [0], [0], GeometryFamily.LINESTRING, GeometryFamily.LINESTRING,
        )
        assert _eval_predicate(masks, "intersects")[0]
        assert not _eval_predicate(masks, "disjoint")[0]

    def test_line_endpoint_touching_line(self):
        from shapely.geometry import LineString
        q = [LineString([(0, 0), (1, 0)])]
        t = [LineString([(1, 0), (2, 0)])]
        masks = _compute_de9im(
            _make_owned(q), _make_owned(t),
            [0], [0], GeometryFamily.LINESTRING, GeometryFamily.LINESTRING,
        )
        assert _eval_predicate(masks, "intersects")[0]
        assert _eval_predicate(masks, "touches")[0]

    def test_multilinestring_polygon(self):
        from shapely.geometry import MultiLineString
        q = [MultiLineString([[(0.6, 0.5), (1.0, 0.5)], [(3, 3), (4, 4)]])]
        t = [box(0.5, 0, 1.5, 1)]
        masks = _compute_de9im(
            _make_owned(q), _make_owned(t),
            [0], [0], GeometryFamily.MULTILINESTRING, GeometryFamily.POLYGON,
        )
        assert _eval_predicate(masks, "intersects")[0]

    def test_polygon_line_swap(self):
        """PG × LS swap: within should work correctly after transpose."""
        from shapely.geometry import LineString
        q = [box(0, 0, 10, 10)]
        t = [LineString([(1, 1), (2, 2)])]
        masks = _compute_de9im(
            _make_owned(q), _make_owned(t),
            [0], [0], GeometryFamily.POLYGON, GeometryFamily.LINESTRING,
        )
        # Polygon contains the line.
        assert _eval_predicate(masks, "contains")[0]
        assert _eval_predicate(masks, "intersects")[0]
