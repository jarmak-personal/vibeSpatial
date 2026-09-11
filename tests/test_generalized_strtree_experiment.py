"""Same-oracle canaries for the shared bounds / existing-refiner experiment."""
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
    return importlib.import_module("experiment_generalized_strtree")


def family_geometries(family, shift=0.):
    x = np.arange(17) * 5. + shift
    if family == "point":
        return shapely.points(x, x * .13)
    if family == "line":
        return shapely.linestrings(np.stack((np.stack((x, x * .13), axis=-1), np.stack((x + 2., x * .13 + 1.), axis=-1)), axis=1))
    if family == "polygon":
        return shapely.box(x, x * .13, x + 2., x * .13 + 2.)
    single = family_geometries(family.removeprefix("multi"), shift)
    other = family_geometries(family.removeprefix("multi"), shift + 1.)
    constructor = {"multipoint": shapely.multipoints, "multiline": shapely.multilinestrings, "multipolygon": shapely.multipolygons}[family]
    if family == "multipolygon":
        other = family_geometries("polygon", shift + 2.5)
    return constructor(np.stack((single, other), axis=1))


FAMILIES = ("point", "line", "polygon", "multipoint", "multiline", "multipolygon")


@pytest.mark.parametrize("tree_family", FAMILIES)
@pytest.mark.parametrize("query_family", FAMILIES)
def test_generalized_family_matrix(experiment, tree_family, query_family):
    from benchmark_osm_nearest import compare

    import vibespatial as vs
    from vibespatial.runtime._runtime import set_requested_mode

    tree_geoms = family_geometries(tree_family)
    query = family_geometries(query_family, .7)
    oracle = shapely.STRtree(tree_geoms).query_nearest(query, all_matches=True, return_distance=True)
    with set_requested_mode("gpu"):
        tree = experiment.GeneralizedSTRtree(vs.GeoSeries.from_wkb(shapely.to_wkb(tree_geoms)), batch_size=8)
        result = tree.query_nearest(vs.GeoSeries.from_wkb(shapely.to_wkb(query)))
    assert compare(result, oracle)["passed"], compare(result, oracle)


@pytest.mark.parametrize("precision", ["fp64", "fp32-outward"])
@pytest.mark.parametrize("translation", [0., 1e7])
def test_generalized_holes_nulls_mixed_ties(experiment, precision, translation):
    from benchmark_osm_nearest import compare

    import vibespatial as vs
    from vibespatial.runtime._runtime import set_requested_mode

    polygon = shapely.Polygon([(0,0), (10,0), (10,10), (0,10)], holes=[[(4,4), (6,4), (6,6), (4,6)]])
    tree_geoms = np.array([None, shapely.Point(), polygon, polygon, shapely.Point(5,5),
        shapely.LineString([(20,0), (20,10)]), shapely.MultiPoint([(30,0),(30,1)]),
        shapely.MultiPolygon([shapely.box(40,0,41,1), shapely.box(50,0,51,1)])], dtype=object)
    query = np.array([None, shapely.Polygon(), shapely.Point(2,2), shapely.Point(5,5), shapely.Point(0,5),
        shapely.box(1,1,3,3), shapely.LineString([(-1,2),(11,2)]), shapely.Point(20,5), shapely.Point(30,.5),
        shapely.MultiLineString([[(39,0),(42,0)],[(45,0),(49,0)]])], dtype=object)
    if translation:
        tree_geoms = shapely.transform(tree_geoms, lambda xy: xy + translation)
        query = shapely.transform(query, lambda xy: xy + translation)
    oracle = shapely.STRtree(tree_geoms).query_nearest(query, all_matches=True, return_distance=True)
    with set_requested_mode("gpu"):
        tree = experiment.GeneralizedSTRtree(vs.GeoSeries.from_wkb(shapely.to_wkb(tree_geoms)), bounds_precision=precision, batch_size=3, slots=1)
        points = vs.GeoSeries.from_wkb(shapely.to_wkb(query))
        result = tree.query_nearest(points)
        relation = tree.query_relation(points)
        assert hasattr(relation.distances, "__cuda_array_interface__")
    assert compare(result, oracle)["passed"], compare(result, oracle)


@pytest.mark.parametrize("empty_tree", [True, False])
def test_generalized_empty_inputs(experiment, empty_tree):
    import vibespatial as vs
    from vibespatial.runtime._runtime import set_requested_mode

    with set_requested_mode("gpu"):
        empty = vs.GeoSeries.from_wkb(shapely.to_wkb([None, shapely.Point()]))
        full = vs.GeoSeries.from_wkb(shapely.to_wkb([shapely.Point(0,0)]))
        tree = experiment.GeneralizedSTRtree(empty if empty_tree else full)
        result = tree.query_nearest(full if empty_tree else empty)
    assert result[0].shape == (2,0)
    assert result[0].dtype == np.int64


@pytest.mark.parametrize("slots", [1, 8, 16])
def test_generalized_ties_cross_frontier_tiles(experiment, slots):
    from benchmark_osm_nearest import compare

    import vibespatial as vs
    from vibespatial.runtime._runtime import set_requested_mode

    polygons = shapely.box(np.zeros(129), np.zeros(129), np.ones(129), np.ones(129))
    points = shapely.points(np.full(17, .5), np.full(17, .5))
    oracle = shapely.STRtree(polygons).query_nearest(points, all_matches=True, return_distance=True)
    with set_requested_mode("gpu"):
        tree = experiment.GeneralizedSTRtree(vs.GeoSeries.from_wkb(shapely.to_wkb(polygons)), slots=slots, batch_size=7)
        query = vs.GeoSeries.from_wkb(shapely.to_wkb(points))
        result = tree.query_nearest(query)
        one = tree.query_nearest(query, all_matches=False)
    assert compare(result, oracle)["passed"]
    assert result[0].shape == (2,129 * 17)
    assert one[0].shape == (2,17)
    assert np.array_equal(np.sort(one[0][0]), np.arange(17))
    assert np.all(one[1] == 0.)


def test_generalized_zero_length_arrays(experiment):
    import vibespatial as vs
    from vibespatial.runtime._runtime import set_requested_mode

    with set_requested_mode("gpu"):
        empty = vs.GeoSeries.from_wkb(np.empty(0, dtype=object))
        tree = experiment.GeneralizedSTRtree(empty)
        result = tree.query_nearest(empty)
    assert result[0].shape == (2,0)


@pytest.mark.parametrize("translation", [0., 1e7])
def test_generalized_positive_ties_across_families(experiment, translation):
    from benchmark_osm_nearest import compare

    import vibespatial as vs
    from vibespatial.runtime._runtime import set_requested_mode

    geometries = np.array([
        shapely.Point(1,0), shapely.LineString([(-2,-1),(2,-1)]),
        shapely.box(1,-1,2,1), shapely.MultiPoint([(-1,0),(-3,0)]),
        shapely.MultiLineString([[(0,1),(2,1)],[(3,3),(4,4)]]),
        shapely.MultiPolygon([shapely.box(-2,-1,-1,1),shapely.box(5,5,6,6)]),
    ], dtype=object)
    query = np.array([shapely.Point(0,0)], dtype=object)
    geometries = shapely.transform(geometries, lambda xy: xy + translation)
    query = shapely.transform(query, lambda xy: xy + translation)
    oracle = shapely.STRtree(geometries).query_nearest(query, all_matches=True, return_distance=True)
    with set_requested_mode("gpu"):
        tree = experiment.GeneralizedSTRtree(vs.GeoSeries.from_wkb(shapely.to_wkb(geometries)), slots=1)
        result = tree.query_nearest(vs.GeoSeries.from_wkb(shapely.to_wkb(query)))
    assert compare(result, oracle)["passed"]
    assert result[0].shape == (2,6)
    assert np.all(result[1] == 1.)
