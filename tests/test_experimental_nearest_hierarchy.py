"""Oracle canaries for the isolated hierarchy experiment, not public dispatch."""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
import shapely

from vibespatial.runtime import has_gpu_runtime

pytestmark = pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required")


@pytest.fixture
def experiment(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("experiment_nearest_hierarchy")


@pytest.mark.parametrize("distribution", ["uniform", "clustered"])
def test_hierarchy_independent_100k_distributions(experiment, distribution):
    """Independent large fixtures catch Morton pruning and row-ID errors."""
    import vibespatial as vs
    from vibespatial.runtime._runtime import set_requested_mode

    rng = np.random.default_rng(30491)
    n = 100_000
    starts = rng.uniform(-1e4, 1e4, size=(n, 2))
    queries = rng.uniform(-1e4, 1e4, size=(n, 2))
    if distribution == "clustered":
        centres = rng.uniform(-1e4, 1e4, size=(16, 2))
        starts = centres[rng.integers(16, size=n)] + rng.normal(0, 50, size=(n, 2))
        queries = centres[rng.integers(16, size=n)] + rng.normal(0, 50, size=(n, 2))
    lines = shapely.linestrings(np.stack((starts, starts + rng.normal(0, 10, size=(n, 2))), axis=1))
    points = shapely.points(queries)
    oracle = shapely.STRtree(lines).query_nearest(points, all_matches=True, return_distance=True)
    with set_requested_mode("gpu"):
        tree = experiment.SegmentHierarchy(vs.GeoSeries.from_wkb(shapely.to_wkb(lines)))
        result = tree.query_nearest(vs.GeoSeries.from_wkb(shapely.to_wkb(points)))
    assert result[0].dtype == np.dtype("int64")
    assert experiment.compare(result, oracle)["passed"]


@pytest.mark.parametrize("width", [1, 8, 32])
def test_hierarchy_all_ties_and_single_leaf(experiment, width):
    """A large tie relation must survive count/scan/scatter without truncation."""
    import vibespatial as vs
    from vibespatial.runtime._runtime import set_requested_mode

    points = shapely.points(np.zeros((257, 2)))
    for n in (1, 1025):
        lines = shapely.linestrings(np.tile([[[-1., 0.], [1., 0.]]], (n, 1, 1)))
        oracle = shapely.STRtree(lines).query_nearest(points, all_matches=True, return_distance=True)
        with set_requested_mode("gpu"):
            tree = experiment.SegmentHierarchy(vs.GeoSeries.from_wkb(shapely.to_wkb(lines)), leaf_width=width)
            result = tree.query_nearest(vs.GeoSeries.from_wkb(shapely.to_wkb(points)))
        assert result[0].shape == (2, n * len(points))
        assert experiment.compare(result, oracle)["passed"]


def test_hierarchy_rejects_leaf_width_that_overflows_kernel_indices(experiment):
    import vibespatial as vs
    from vibespatial.runtime._runtime import set_requested_mode

    with set_requested_mode("gpu"):
        lines = vs.GeoSeries.from_wkb(shapely.to_wkb(shapely.linestrings([[[0., 0.], [1., 1.]]])))
        for width in (0, -1, 2**31):
            with pytest.raises(ValueError, match="leaf width"):
                experiment.SegmentHierarchy(lines, leaf_width=width)


@pytest.mark.parametrize("fanout", [2, 4, 8])
@pytest.mark.parametrize("packing", ["morton", "str", "str-recursive"])
@pytest.mark.parametrize("coherent_seeded", [False, True])
@pytest.mark.parametrize("bounds_precision", ["fp64", "fp32-outward"])
def test_strategy_whole_road_ties(experiment, fanout, packing, coherent_seeded, bounds_precision):
    """Packing/order/seeds cannot change whole-road tie identities."""
    import vibespatial as vs
    from vibespatial.runtime._runtime import set_requested_mode

    strategy = importlib.import_module("experiment_nearest_strategies")
    rng = np.random.default_rng(812)
    counts = rng.integers(2, 12, size=100)
    coords = rng.normal(size=(counts.sum(), 2)) + 1e6
    lines = shapely.linestrings(coords, indices=np.repeat(np.arange(100), counts))
    lines = np.concatenate((lines, lines[:3], shapely.from_wkt(["LINESTRING (1000000 1000000, 1000000 1000000)"])))
    points = shapely.points(np.concatenate((rng.normal(size=(500, 2)) + 1e6, coords[:2], [[1e7, 1e7]])))
    oracle = shapely.STRtree(lines).query_nearest(points, all_matches=True, return_distance=True)
    with set_requested_mode("gpu"):
        tree = strategy.StrategyHierarchy(vs.GeoSeries.from_wkb(shapely.to_wkb(lines)),
                                         fanout=fanout, packing=packing, parent_roads=True,
                                         query_order="morton" if coherent_seeded else "input",
                                         seed_resolution=32 if coherent_seeded else 0, bounds_precision=bounds_precision)
        result = tree.query_nearest(vs.GeoSeries.from_wkb(shapely.to_wkb(points)))
    assert experiment.compare(result, oracle)["passed"]


@pytest.mark.parametrize("resolution", [1, 8, 32])
def test_direct_grid_ownership_and_search_certificate(experiment, resolution):
    import vibespatial as vs
    from vibespatial.runtime._runtime import set_requested_mode

    grid = importlib.import_module("experiment_nearest_grid")
    rng = np.random.default_rng(324)
    roads = shapely.linestrings(rng.normal(size=(100, 3, 2)))
    roads = np.concatenate((roads, roads[:3], shapely.from_wkt([
        "LINESTRING (-100 -100, 100 100)", "LINESTRING (0 0, 0 0)",
        "LINESTRING (-100 100, 100 -100)",
    ])))
    points = shapely.points(np.concatenate((rng.normal(size=(500, 2)),
                                           [[0, 0], [10, 10], [1000, 1000], [-1000, 3], [4, -1000]])))
    oracle = shapely.STRtree(roads).query_nearest(points, all_matches=True, return_distance=True)
    with set_requested_mode("gpu"):
        tree = grid.GridHierarchy(vs.GeoSeries.from_wkb(shapely.to_wkb(roads)),
                                  parent_roads=True, seed_resolution=resolution, query_order="morton")
        result = tree.query_nearest(vs.GeoSeries.from_wkb(shapely.to_wkb(points)))
    assert experiment.compare(result, oracle)["passed"]


@pytest.mark.parametrize("span", [25., 100.])
@pytest.mark.parametrize("parent_roads", [False, True])
@pytest.mark.parametrize("bounds_precision", ["fp64", "fp32-outward"])
def test_virtual_bounds_preserve_original_segment_metric(experiment, span, parent_roads, bounds_precision):
    """Bound-only subdivision must never replace the original distance input."""
    import vibespatial as vs
    from vibespatial.runtime._runtime import set_requested_mode

    strategy = importlib.import_module("experiment_nearest_strategies")
    rng = np.random.default_rng(159)
    end = rng.uniform(-1000, 1000, size=(129, 2))
    lines = shapely.linestrings(np.stack((-end, end), axis=1) + 1e7)
    lines = np.concatenate((lines, lines[:2]))
    points = shapely.points(np.concatenate((rng.uniform(-900, 900, size=(1000, 2)), [[0, 0]])) + 1e7)
    oracle = shapely.STRtree(lines).query_nearest(points, all_matches=True, return_distance=True)
    with set_requested_mode("gpu"):
        tree = strategy.StrategyHierarchy(vs.GeoSeries.from_wkb(shapely.to_wkb(lines)),
                                         packing="str-recursive", fanout=8, leaf_width=4,
                                         query_order="morton", seed_resolution=32,
                                         max_span=span, parent_roads=parent_roads, bounds_precision=bounds_precision)
        result = tree.query_nearest(vs.GeoSeries.from_wkb(shapely.to_wkb(points)))
    assert tree.n > len(lines)
    parity = experiment.compare(result, oracle)
    assert parity["passed"]
    assert parity["max_distance_error_m"] == 0.
