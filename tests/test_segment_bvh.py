"""Native complex polygon and mixed-family segment acceleration oracles."""

from __future__ import annotations

import numpy as np
import pytest
import shapely

from tests.test_packed_str_index import assert_nearest, build, export
from vibespatial.runtime import has_gpu_runtime

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required")]


def test_segment_capacity_is_rejected_before_extraction(monkeypatch):
    from types import SimpleNamespace

    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.kernels.spatial.segment_bvh import RowSegmentBVH
    from vibespatial.spatial import segment_primitives

    def forbid_extraction(*args, **kwargs):
        raise AssertionError("oversized input reached int32 segment extraction")

    monkeypatch.setattr(segment_primitives, "_extract_segments_gpu", forbid_extraction)
    state = SimpleNamespace(families={
        GeometryFamily.POLYGON: SimpleNamespace(x=SimpleNamespace(size=2**29)),
    })
    geometry = SimpleNamespace(move_to=lambda *args, **kwargs: None,
                               _ensure_device_state=lambda: state)
    with pytest.raises(ValueError, match="non-point coordinates"):
        RowSegmentBVH(geometry, bounds_plan=None)


@pytest.mark.parametrize("padding", [None, "simple"])
def test_outlier_refinement_survives_unrelated_padding(padding):
    from vibespatial.testing import build_owned

    theta = np.linspace(0, 2 * np.pi, 512, endpoint=False)
    polygon = shapely.Polygon(np.stack((np.cos(theta), np.sin(theta)), axis=1))
    unrelated = None if padding is None else shapely.box(100, 100, 101, 101)
    tree = [polygon] + [unrelated] * 256
    query = [shapely.transform(polygon, lambda xy: xy + 3)] + [None] * 256
    index = build(tree, "fp32")
    actual = export(index.query_relation(build_owned(query)))
    assert index._segment_refiner is not None
    assert_nearest(actual, shapely.STRtree(tree).query_nearest(query, return_distance=True))


@pytest.mark.parametrize("precision", ["fp32", "fp64"])
@pytest.mark.parametrize("translation", [0.0, 1e7])
@pytest.mark.parametrize("mixed", [False, True])
def test_complex_holes_and_multipart(precision, translation, mixed):
    from vibespatial.testing import build_owned

    theta = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    shell = np.stack((np.cos(theta), np.sin(theta)), axis=1) * (10 + np.sin(theta * 7))[:, None]
    hole = np.stack((np.cos(theta[::-1]), np.sin(theta[::-1])), axis=1) * 2
    polygon = shapely.Polygon(shell, [hole])
    tree = np.array(
        [
            polygon,
            shapely.transform(polygon, lambda xy: xy + 30),
            shapely.MultiPolygon(
                [shapely.transform(polygon, lambda xy: xy + 60), shapely.box(80, 80, 81, 81)]
            ),
        ],
        dtype=object,
    )
    query = np.array(
        [
            shapely.box(-0.5, -0.5, 0.5, 0.5),
            shapely.box(3, 3, 4, 4),
            shapely.box(20, 20, 21, 21),
            shapely.LineString([(-11, 0), (11, 0)]),
            polygon,
        ],
        dtype=object,
    )
    if mixed:
        tree = np.concatenate(
            (tree, [shapely.Point(100, 100), shapely.MultiPoint([(110, 110), (120, 120)]), None])
        )
        query = np.concatenate(
            (query, [shapely.Point(105, 105), shapely.MultiPoint([(1, 1), (33, 33)]), None])
        )
    tree = shapely.transform(tree, lambda xy: xy + translation)
    query = shapely.transform(query, lambda xy: xy + translation)
    index = build(tree, precision)
    query_owned = build_owned(query)
    actual = export(index.query_relation(query_owned))
    assert index._segment_refiner is not None
    assert_nearest(actual, shapely.STRtree(tree).query_nearest(query, return_distance=True))
    retained = index._segment_refiner.tree
    assert_nearest(export(index.query_relation(query_owned)), actual)
    assert index._segment_refiner.tree is retained
