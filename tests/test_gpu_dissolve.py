from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import Polygon, box

import vibespatial.api as geopandas
from vibespatial import DissolveUnionMethod, has_gpu_runtime
from vibespatial.api.testing import assert_geodataframe_equal
from vibespatial.overlay.dissolve import (
    evaluate_geopandas_dissolve,
    execute_grouped_box_union_gpu,
    execute_grouped_coverage_edge_union,
    execute_grouped_coverage_union_gpu,
    execute_grouped_disjoint_subset_union_codes,
)

dissolve_module = importlib.import_module("vibespatial.overlay.dissolve")


def _build_group_positions(groups: np.ndarray) -> list[np.ndarray]:
    unique = np.unique(groups)
    return [np.flatnonzero(groups == value).astype(np.int32, copy=False) for value in unique.tolist()]


@pytest.mark.gpu
def test_gpu_grouped_box_union_matches_shapely_coverage_union() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    groups = np.asarray([0, 0, 1, 1], dtype=np.int32)
    geometries = np.asarray(
        [
            box(0, 0, 1, 1),
            box(1, 0, 2, 1),
            box(10, 10, 11, 11),
            box(11, 10, 12, 11),
        ],
        dtype=object,
    )

    grouped = execute_grouped_box_union_gpu(geometries, _build_group_positions(groups))

    assert grouped is not None
    expected = np.asarray(
        [
            shapely.coverage_union_all(geometries[groups == 0]),
            shapely.coverage_union_all(geometries[groups == 1]),
        ],
        dtype=object,
    )
    assert grouped.method is DissolveUnionMethod.COVERAGE
    assert grouped.group_count == 2
    assert all(bool(shapely.equals(left, right)) for left, right in zip(grouped.geometries, expected, strict=True))


@pytest.mark.gpu
def test_gpu_dissolve_matches_geopandas_for_box_coverages() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    frame = geopandas.GeoDataFrame(
        {
            "group": pd.Categorical([0, 0, 1, 1]),
            "value": [1, 2, 3, 4],
            "geometry": [
                box(0, 0, 1, 1),
                box(1, 0, 2, 1),
                box(10, 10, 11, 11),
                box(11, 10, 12, 11),
            ],
        },
        crs="EPSG:3857",
    )

    actual = evaluate_geopandas_dissolve(
        frame,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="coverage",
        grid_size=None,
        agg_kwargs={},
    )
    expected = frame.dissolve(by="group", aggfunc="first", method="coverage")

    assert_geodataframe_equal(actual, expected)


@pytest.mark.gpu
def test_gpu_grouped_coverage_union_matches_shapely_for_non_rectangular_coverages() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    groups = np.asarray([0, 0, 1, 1], dtype=np.int32)
    geometries = np.asarray(
        [
            Polygon([(0, 0), (1, 0), (0, 1)]),
            Polygon([(1, 0), (1, 1), (0, 1)]),
            Polygon([(10, 10), (11, 10), (10, 11)]),
            Polygon([(11, 10), (11, 11), (10, 11)]),
        ],
        dtype=object,
    )
    owned = geopandas.GeoSeries(geometries).values.to_owned()

    grouped = execute_grouped_coverage_union_gpu(
        geometries,
        _build_group_positions(groups),
        owned=owned,
    )

    assert grouped is not None
    expected = np.asarray(
        [
            shapely.coverage_union_all(geometries[groups == 0]),
            shapely.coverage_union_all(geometries[groups == 1]),
        ],
        dtype=object,
    )
    assert grouped.method is DissolveUnionMethod.COVERAGE
    assert all(bool(shapely.equals(left, right)) for left, right in zip(grouped.geometries, expected, strict=True))


@pytest.mark.gpu
def test_grouped_coverage_edge_union_matches_shapely_for_non_rectangular_coverages() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    groups = np.asarray([0, 0, 1, 1], dtype=np.int32)
    geometries = np.asarray(
        [
            Polygon([(0, 0), (1, 0), (0, 1)]),
            Polygon([(1, 0), (1, 1), (0, 1)]),
            Polygon([(10, 10), (11, 10), (10, 11)]),
            Polygon([(11, 10), (11, 11), (10, 11)]),
        ],
        dtype=object,
    )

    grouped = execute_grouped_coverage_edge_union(
        geometries,
        _build_group_positions(groups),
    )

    assert grouped is not None
    expected = np.asarray(
        [
            shapely.coverage_union_all(geometries[groups == 0]),
            shapely.coverage_union_all(geometries[groups == 1]),
        ],
        dtype=object,
    )
    assert grouped.method is DissolveUnionMethod.COVERAGE
    assert all(bool(shapely.equals(left, right)) for left, right in zip(grouped.geometries, expected, strict=True))


@pytest.mark.gpu
def test_grouped_coverage_edge_union_preserves_holes() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    groups = np.asarray([0, 0, 0, 0], dtype=np.int32)
    geometries = np.asarray(
        [box(x, y, x + 1, y + 1) for y in range(3) for x in range(3) if not (x == 1 and y == 1)],
        dtype=object,
    )
    groups = np.zeros(geometries.size, dtype=np.int32)

    grouped = execute_grouped_coverage_edge_union(
        geometries,
        _build_group_positions(groups),
    )

    assert grouped is not None
    expected = np.asarray([shapely.coverage_union_all(geometries)], dtype=object)
    assert grouped.method is DissolveUnionMethod.COVERAGE
    assert bool(shapely.equals(grouped.geometries[0], expected[0]))


@pytest.mark.gpu
def test_gpu_dissolve_can_route_non_rectangular_coverages_to_disjoint_subset_union(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    frame = geopandas.GeoDataFrame(
        {
            "group": pd.Categorical([0, 0, 1, 1]),
            "value": [1, 2, 3, 4],
            "geometry": [
                Polygon([(0, 0), (1, 0), (0, 1)]),
                Polygon([(1, 0), (1, 1), (0, 1)]),
                Polygon([(10, 10), (11, 10), (10, 11)]),
                Polygon([(11, 10), (11, 11), (10, 11)]),
            ],
        },
        crs="EPSG:3857",
    )
    expected = frame.dissolve(by="group", aggfunc="first", method="coverage")

    calls = 0
    real_fn = dissolve_module.execute_grouped_disjoint_subset_union_codes

    def _counting_gpu(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(dissolve_module, "OVERLAY_UNION_ALL_GPU_THRESHOLD", 1)
    monkeypatch.setattr(dissolve_module, "execute_grouped_disjoint_subset_union_codes", _counting_gpu)

    actual = evaluate_geopandas_dissolve(
        frame,
        by="group",
        aggfunc="first",
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="coverage",
        grid_size=None,
        agg_kwargs={},
    )

    assert calls == 1
    assert_geodataframe_equal(actual, expected)


@pytest.mark.gpu
def test_grouped_disjoint_subset_union_codes_preserves_holes() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    geometries = np.asarray(
        [box(x, y, x + 1, y + 1) for y in range(3) for x in range(3) if not (x == 1 and y == 1)],
        dtype=object,
    )
    row_group_codes = np.zeros(geometries.size, dtype=np.int32)

    grouped = execute_grouped_disjoint_subset_union_codes(
        geometries,
        row_group_codes,
        group_count=1,
    )

    assert grouped is not None
    assert grouped.method is DissolveUnionMethod.DISJOINT_SUBSET
    expected = shapely.coverage_union_all(geometries)
    assert bool(shapely.equals(grouped.geometries[0], expected))
