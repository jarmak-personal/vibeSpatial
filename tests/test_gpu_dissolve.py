from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import box

import vibespatial.api as geopandas
from vibespatial import DissolveUnionMethod, has_gpu_runtime
from vibespatial.api.testing import assert_geodataframe_equal
from vibespatial.overlay.dissolve import (
    evaluate_geopandas_dissolve,
    execute_grouped_box_union_gpu,
)


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
