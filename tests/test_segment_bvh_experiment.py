"""Exact candidate-distance oracles for the complex-geometry experiment."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
import shapely

from vibespatial.runtime import has_gpu_runtime

pytestmark = pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required")


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("experiment_segment_bvh")


@pytest.mark.parametrize("hierarchy", [False, True])
@pytest.mark.parametrize("translation", [0.0, 1e7])
@pytest.mark.parametrize("tile_segments", [0, 32])
def test_segment_bvh_mixed_candidates(module, hierarchy, translation, tile_segments):
    import cupy as cp

    from vibespatial.runtime._runtime import select_runtime
    from vibespatial.runtime.precision import KernelClass, PrecisionMode, select_precision_plan
    from vibespatial.testing import build_owned

    polygon = shapely.Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10)], holes=[[(4, 4), (6, 4), (6, 6), (4, 6)]]
    )
    geoms = np.array(
        [
            None,
            shapely.LineString(),
            shapely.LineString([(-1, 2), (11, 2)]),
            shapely.LineString([(4.5, 5), (5.5, 5)]),
            polygon,
            shapely.box(1, 1, 2, 2),
            shapely.MultiLineString([[(20, 0), (20, 5)], [(2, 2), (3, 3)]]),
            shapely.MultiPolygon([shapely.box(30, 0, 31, 1), shapely.box(40, 0, 42, 2)]),
        ],
        dtype=object,
    )
    geoms = shapely.transform(geoms, lambda xy: xy + translation)
    query = build_owned(geoms)
    tree = build_owned(geoms[::-1])
    left, right = np.indices((len(geoms), len(geoms))).reshape(2, -1).astype(np.int32)
    expected = shapely.distance(geoms[left], geoms[::-1][right])
    plan = select_precision_plan(
        runtime_selection=select_runtime("gpu"),
        kernel_class=KernelClass.METRIC,
        requested=PrecisionMode.FP64,
    )
    refiner = module.SegmentBVHRefiner(tree, plan, hierarchy=hierarchy, tile_segments=tile_segments)
    refine = refiner.for_query(query)
    actual = cp.asnumpy(
        refine(cp.asarray(left), cp.asarray(right), cp.asarray(np.isfinite(expected)))[0]
    )
    assert np.array_equal(actual == 0.0, expected == 0.0)
    assert np.all(np.isinf(actual[~np.isfinite(expected)]))
    np.testing.assert_allclose(
        actual[np.isfinite(expected)], expected[np.isfinite(expected)], rtol=1e-10, atol=1e-10
    )


@pytest.mark.parametrize("hierarchy", [False, True])
@pytest.mark.parametrize("tile_segments", [0, 32])
def test_segment_bvh_complex_nearest(module, hierarchy, tile_segments):
    from types import SimpleNamespace

    from benchmark_osm_nearest import compare
    from experiment_generalized_strtree import GeneralizedSTRtree

    from vibespatial.testing import build_owned

    angle = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    shell = np.stack((np.cos(angle), np.sin(angle)), axis=1) * (10.0 + np.sin(7 * angle))[:, None]
    hole = np.stack((np.cos(angle[::-1]), np.sin(angle[::-1])), axis=1) * 2.0
    polygons = shapely.polygons(
        shapely.linearrings(shell[None, :, :] + np.array([0.0, 30.0, 60.0])[:, None, None]),
        holes=shapely.linearrings(hole[None, :, :] + np.array([0.0, 30.0, 60.0])[:, None, None])[
            :, None
        ],
    )
    query = np.array(
        [shapely.box(-0.5, -0.5, 0.5, 0.5), shapely.box(3, 3, 4, 4), shapely.box(20, 20, 21, 21)],
        dtype=object,
    )

    def wrap(geoms):
        return SimpleNamespace(array=SimpleNamespace(_owned=build_owned(geoms)))

    tree = GeneralizedSTRtree(
        wrap(polygons),
        slots=1,
        refiner_factory=lambda g, p: module.SegmentBVHRefiner(
            g, p, hierarchy=hierarchy, tile_segments=tile_segments
        ),
    )
    result = tree.query_nearest(wrap(query))
    oracle = shapely.STRtree(polygons).query_nearest(query, all_matches=True, return_distance=True)
    assert compare(result, oracle)["passed"]
