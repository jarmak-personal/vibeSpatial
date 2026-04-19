from __future__ import annotations

import importlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import LineString, Polygon, box

import vibespatial.api as geopandas
from vibespatial import DissolveUnionMethod, has_gpu_runtime
from vibespatial.api.testing import assert_geodataframe_equal
from vibespatial.geometry.device_array import DeviceGeometryArray
from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.overlay.dissolve import (
    evaluate_geopandas_dissolve,
    execute_grouped_box_union_gpu,
    execute_grouped_coverage_edge_union,
    execute_grouped_coverage_union_gpu,
    execute_grouped_disjoint_subset_union_codes,
)
from vibespatial.runtime.event_log import EVENT_LOG_ENV_VAR, read_event_records
from vibespatial.runtime.residency import Residency
from vibespatial.testing import strict_native_environment

dissolve_module = importlib.import_module("vibespatial.overlay.dissolve")


def _river_lines(count: int, *, seed: int, vertices: int = 12) -> list[LineString]:
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.0, 1000.0, vertices)
    amplitude = 1000.0 / 8.0
    geoms: list[LineString] = []
    for offset in rng.uniform(0.0, 1000.0, count):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        coords = [
            (
                float(x),
                float(np.clip(offset + amplitude * np.sin(phase + i / 2.0), 0.0, 1000.0)),
            )
            for i, x in enumerate(xs)
        ]
        geoms.append(LineString(coords))
    return geoms


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
def test_public_dissolve_gpu_coverage_smoke() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    lines = [
        LineString([(0.0, 0.0), (10.0, 0.0)]),
        LineString([(10.0, 0.0), (0.0, 0.0)]),
        LineString([(0.0, 5.0), (10.0, 5.0)]),
        LineString([(10.0, 5.0), (0.0, 5.0)]),
    ] * 32
    frame = geopandas.GeoDataFrame(
        {
            "group": np.zeros(len(lines), dtype=np.int32),
            "value": np.arange(len(lines), dtype=np.int32),
        },
        geometry=geopandas.GeoSeries(
            DeviceGeometryArray._from_owned(
                from_shapely_geometries(lines, residency=Residency.DEVICE),
                crs="EPSG:3857",
            ),
            crs="EPSG:3857",
        ),
        crs="EPSG:3857",
    )
    buffered = frame.copy()
    buffered["geometry"] = buffered.geometry.buffer(0.5)

    actual = buffered.dissolve(by="group")
    expected_geom = shapely.union_all(np.asarray(buffered.geometry.array, dtype=object))
    actual_geom = np.asarray(actual.geometry.array, dtype=object)[0]
    assert shapely.area(shapely.symmetric_difference(actual_geom, expected_geom)) == 0.0
    assert actual.geometry.dtype.name == "device_geometry"
    assert type(actual.geometry.values).__name__ == "DeviceGeometryArray"
    actual_owned = getattr(actual.geometry.values, "_owned", None)
    assert actual_owned is not None
    assert actual_owned.residency is Residency.DEVICE

    event_log = os.environ.get(EVENT_LOG_ENV_VAR)
    if event_log:
        records = read_event_records(Path(event_log))
        assert any(
            record.get("event_type") == "dispatch"
            and record.get("surface") == "geopandas.geodataframe.dissolve"
            and record.get("operation") == "dissolve"
            and record.get("selected") == "gpu"
            for record in records
        )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "kwargs",
    [
        {"by": None},
        {"aggfunc": "mean", "numeric_only": True},
    ],
)
def test_public_dissolve_none_strict_native_keeps_device_geometry(kwargs) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    frame = geopandas.GeoDataFrame(
        {"value": np.asarray([1, 2], dtype=np.int64)},
        geometry=geopandas.GeoSeries(
            DeviceGeometryArray._from_owned(
                from_shapely_geometries(
                    [box(0.0, 0.0, 1.0, 1.0), box(1.0, 0.0, 2.0, 1.0)],
                    residency=Residency.DEVICE,
                ),
                crs="EPSG:3857",
            ),
            crs="EPSG:3857",
        ),
        crs="EPSG:3857",
    )

    with strict_native_environment():
        result = frame.dissolve(**kwargs)

    assert result.geometry.dtype.name == "device_geometry"
    assert type(result.geometry.values).__name__ == "DeviceGeometryArray"
    owned = getattr(result.geometry.values, "_owned", None)
    assert owned is not None
    assert owned.residency is Residency.DEVICE


@pytest.mark.gpu
def test_public_collapse_all_dissolve_keeps_device_backing_for_follow_on_make_valid() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    lines = [
        LineString([(0.0, 0.0), (10.0, 0.0)]),
        LineString([(10.0, 0.0), (0.0, 0.0)]),
        LineString([(0.0, 5.0), (10.0, 5.0)]),
        LineString([(10.0, 5.0), (0.0, 5.0)]),
    ] * 16
    frame = geopandas.GeoDataFrame(
        {"value": np.arange(len(lines), dtype=np.int32)},
        geometry=geopandas.GeoSeries(
            DeviceGeometryArray._from_owned(
                from_shapely_geometries(lines, residency=Residency.DEVICE),
                crs="EPSG:3857",
            ),
            crs="EPSG:3857",
        ),
        crs="EPSG:3857",
    )
    buffered = frame.copy()
    buffered["geometry"] = buffered.geometry.buffer(0.5)

    geopandas.clear_dispatch_events()
    dissolved = buffered.dissolve()
    make_valid = dissolved.geometry.make_valid()
    events = geopandas.get_dispatch_events(clear=True)

    assert dissolved.geometry.dtype.name == "device_geometry"
    assert type(dissolved.geometry.values).__name__ == "DeviceGeometryArray"
    assert make_valid.dtype.name == "device_geometry"
    assert type(make_valid.values).__name__ == "DeviceGeometryArray"
    assert any(
        event.surface == "geopandas.array.make_valid"
        and event.operation == "make_valid"
        and event.selected.value == "gpu"
        for event in events
    )


@pytest.mark.gpu
def test_buffered_line_dissolve_gpu_result_is_valid() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    lines = _river_lines(200, seed=10)
    frame = geopandas.GeoDataFrame(
        {"group": np.zeros(len(lines), dtype=np.int32)},
        geometry=geopandas.GeoSeries(
            DeviceGeometryArray._from_owned(
                from_shapely_geometries(lines, residency=Residency.DEVICE),
                crs="EPSG:4326",
            ),
            crs="EPSG:4326",
        ),
        crs="EPSG:4326",
    )
    buffered = frame.copy()
    buffered["geometry"] = buffered.geometry.buffer(10.0)

    geopandas.clear_dispatch_events()
    dissolved = buffered.dissolve(by="group")
    events = geopandas.get_dispatch_events(clear=True)
    dissolved_geom = np.asarray(dissolved.geometry.array, dtype=object)[0]

    assert dissolved.geometry.dtype.name == "device_geometry"
    assert bool(shapely.is_valid(dissolved_geom))
    assert not any(
        event.surface == "constructive.disjoint_subset_union_all"
        and event.operation == "disjoint_subset_union_all"
        and event.implementation == "disjoint_subset_union_all_gpu"
        and event.selected.value == "gpu"
        for event in events
    )


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
    assert grouped.geometries is None
    assert grouped.owned is not None
    actual = np.asarray(grouped.owned.to_shapely(), dtype=object)
    assert all(bool(shapely.equals(left, right)) for left, right in zip(actual, expected, strict=True))


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
def test_gpu_dissolve_can_route_non_rectangular_coverages_to_edge_union_first(
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
    real_fn = dissolve_module.execute_grouped_coverage_edge_union_codes

    def _counting_gpu(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(dissolve_module, "OVERLAY_GROUPED_COVERAGE_EDGE_THRESHOLD", 1)
    monkeypatch.setattr(dissolve_module, "execute_grouped_coverage_edge_union_codes", _counting_gpu)

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
