from __future__ import annotations

import vibespatial.api as geopandas
import numpy as np
import pytest
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

import vibespatial.io_arrow as io_arrow
from vibespatial import has_gpu_runtime, has_pylibcudf_support, read_geoparquet_owned
from vibespatial.residency import Residency


def _require_gpu_decode_runtime() -> None:
    if not has_gpu_runtime() or not has_pylibcudf_support():
        pytest.skip("GPU decode tests require CUDA runtime and pylibcudf")


def _assert_geometry_lists_equal(left, right) -> None:
    assert len(left) == len(right)
    for got, expected in zip(left, right, strict=True):
        if expected is None:
            assert got is None
            continue
        assert got is not None
        assert got.equals(expected)


@pytest.mark.parametrize(
    ("family", "geometries"),
    [
        ("point", [Point(0, 0), None, Point(2, 2), Point(3, 3)]),
        (
            "linestring",
            [LineString([(0, 0), (1, 1)]), LineString(), None, LineString([(2, 2), (3, 3), (4, 4)])],
        ),
        (
            "polygon",
            [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
                Polygon(),
                None,
                Polygon([(2, 2), (3, 2), (3, 3), (2, 2)]),
            ],
        ),
        ("multipoint", [MultiPoint([(0, 0), (1, 1)]), MultiPoint([]), None, MultiPoint([(2, 2)])]),
        (
            "multilinestring",
            [
                MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]]),
                MultiLineString([]),
                None,
                MultiLineString([[(4, 4), (5, 5)]]),
            ],
        ),
        (
            "multipolygon",
            [
                MultiPolygon([Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])]),
                MultiPolygon([]),
                None,
                MultiPolygon([Polygon([(2, 2), (3, 2), (3, 3), (2, 2)])]),
            ],
        ),
    ],
)
def test_read_geoparquet_owned_gpu_geoarrow_decode_matches_cpu_for_each_family(
    tmp_path,
    family: str,
    geometries,
) -> None:
    _require_gpu_decode_runtime()

    path = tmp_path / f"{family}.parquet"
    frame = geopandas.GeoDataFrame({"geometry": geometries}, geometry="geometry")
    frame.to_parquet(path, geometry_encoding="geoarrow")

    gpu_owned = read_geoparquet_owned(path, backend="gpu")
    cpu_owned = read_geoparquet_owned(path, backend="cpu")

    assert gpu_owned.residency is Residency.DEVICE
    assert gpu_owned.device_state is not None
    decoded_family = next(iter(gpu_owned.families))
    assert gpu_owned.families[decoded_family].host_materialized is False

    _assert_geometry_lists_equal(gpu_owned.to_shapely(), cpu_owned.to_shapely())

    assert gpu_owned.families[decoded_family].host_materialized is True


def test_device_child_selection_mask_matches_host_reference() -> None:
    _require_gpu_decode_runtime()
    cupy = pytest.importorskip("cupy")

    offsets = np.asarray([0, 2, 2, 5, 7, 7, 9], dtype=np.int32)
    parent_mask = np.asarray([True, False, True, True, False, True], dtype=bool)

    expected_offsets, expected_mask = io_arrow._child_selection_mask(offsets, parent_mask)
    actual_offsets, actual_mask = io_arrow._device_child_selection_mask(
        cupy.asarray(offsets),
        cupy.asarray(parent_mask),
    )

    np.testing.assert_array_equal(cupy.asnumpy(actual_offsets), expected_offsets)
    np.testing.assert_array_equal(cupy.asnumpy(actual_mask), expected_mask)


def test_pylibcudf_wkb_header_scan_matches_mixed_family_reference(tmp_path) -> None:
    _require_gpu_decode_runtime()
    cupy = pytest.importorskip("cupy")

    path = tmp_path / "mixed-wkb.parquet"
    frame = geopandas.GeoDataFrame(
        {
            "geometry": [
                Point(0, 0),
                LineString([(0, 0), (1, 1)]),
                Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
                None,
            ]
        },
        geometry="geometry",
    )
    frame.to_parquet(path, geometry_encoding="WKB")

    table = io_arrow._read_geoparquet_table_with_pylibcudf(path, columns=["geometry"])
    scan = io_arrow._scan_pylibcudf_wkb_headers(table.columns()[0])

    np.testing.assert_array_equal(cupy.asnumpy(scan.validity), np.asarray([True, True, True, False]))
    np.testing.assert_array_equal(cupy.asnumpy(scan.type_ids), np.asarray([1, 2, 3, -1], dtype=np.int32))
    np.testing.assert_array_equal(cupy.asnumpy(scan.point_mask), np.asarray([True, False, False, False]))
    np.testing.assert_array_equal(cupy.asnumpy(scan.fallback_mask), np.asarray([False, False, False, False]))
    assert scan.native_count == 3
    assert scan.fallback_count == 0


def test_read_geoparquet_owned_gpu_wkb_point_decode_matches_cpu(tmp_path) -> None:
    _require_gpu_decode_runtime()

    path = tmp_path / "point-wkb.parquet"
    frame = geopandas.GeoDataFrame(
        {"geometry": [Point(0, 0), None, Point(2, 2), Point()]},
        geometry="geometry",
    )
    frame.to_parquet(path, geometry_encoding="WKB")

    gpu_owned = read_geoparquet_owned(path, backend="gpu")
    cpu_owned = read_geoparquet_owned(path, backend="cpu")

    assert gpu_owned.residency is Residency.DEVICE
    assert gpu_owned.device_state is not None
    point_family = next(iter(gpu_owned.families))
    assert gpu_owned.families[point_family].host_materialized is False

    _assert_geometry_lists_equal(gpu_owned.to_shapely(), cpu_owned.to_shapely())


def test_read_geoparquet_owned_gpu_wkb_linestring_decode_matches_cpu(tmp_path) -> None:
    _require_gpu_decode_runtime()

    path = tmp_path / "linestring-wkb.parquet"
    frame = geopandas.GeoDataFrame(
        {
            "geometry": [
                LineString([(0, 0), (1, 1)]),
                None,
                LineString(),
                LineString([(2, 2), (3, 3), (4, 4)]),
            ]
        },
        geometry="geometry",
    )
    frame.to_parquet(path, geometry_encoding="WKB")

    gpu_owned = read_geoparquet_owned(path, backend="gpu")
    cpu_owned = read_geoparquet_owned(path, backend="cpu")

    assert gpu_owned.residency is Residency.DEVICE
    assert gpu_owned.device_state is not None
    linestring_family = next(iter(gpu_owned.families))
    assert gpu_owned.families[linestring_family].host_materialized is False

    _assert_geometry_lists_equal(gpu_owned.to_shapely(), cpu_owned.to_shapely())


def test_read_geoparquet_owned_gpu_wkb_mixed_point_linestring_decode_matches_cpu(tmp_path) -> None:
    _require_gpu_decode_runtime()

    path = tmp_path / "mixed-point-linestring-wkb.parquet"
    frame = geopandas.GeoDataFrame(
        {
            "geometry": [
                Point(0, 0),
                LineString([(0, 0), (1, 1)]),
                None,
                Point(),
                LineString(),
                Point(2, 2),
            ]
        },
        geometry="geometry",
    )
    frame.to_parquet(path, geometry_encoding="WKB")

    gpu_owned = read_geoparquet_owned(path, backend="gpu")
    cpu_owned = read_geoparquet_owned(path, backend="cpu")

    assert gpu_owned.residency is Residency.DEVICE
    assert gpu_owned.device_state is not None
    assert set(gpu_owned.families) == {io_arrow.GeometryFamily.POINT, io_arrow.GeometryFamily.LINESTRING}
    for family in gpu_owned.families.values():
        assert family.host_materialized is False

    _assert_geometry_lists_equal(gpu_owned.to_shapely(), cpu_owned.to_shapely())


# --- WKB multipoint decode tests ---


def test_read_geoparquet_owned_gpu_wkb_multipoint_decode_matches_cpu(tmp_path) -> None:
    _require_gpu_decode_runtime()

    path = tmp_path / "multipoint-wkb.parquet"
    frame = geopandas.GeoDataFrame(
        {
            "geometry": [
                MultiPoint([(0, 0), (1, 1)]),
                MultiPoint([]),
                None,
                MultiPoint([(2, 2)]),
            ]
        },
        geometry="geometry",
    )
    frame.to_parquet(path, geometry_encoding="WKB")

    gpu_owned = read_geoparquet_owned(path, backend="gpu")
    cpu_owned = read_geoparquet_owned(path, backend="cpu")

    assert gpu_owned.residency is Residency.DEVICE
    assert gpu_owned.device_state is not None
    multipoint_family = next(iter(gpu_owned.families))
    assert gpu_owned.families[multipoint_family].host_materialized is False

    _assert_geometry_lists_equal(gpu_owned.to_shapely(), cpu_owned.to_shapely())


def test_read_geoparquet_owned_gpu_wkb_multipoint_varied_part_counts(tmp_path) -> None:
    _require_gpu_decode_runtime()

    path = tmp_path / "multipoint-varied-wkb.parquet"
    frame = geopandas.GeoDataFrame(
        {
            "geometry": [
                MultiPoint([(i, i + 1) for i in range(10)]),
                MultiPoint([(100, 200)]),
                MultiPoint([]),
                None,
                MultiPoint([(3, 4), (5, 6), (7, 8)]),
            ]
        },
        geometry="geometry",
    )
    frame.to_parquet(path, geometry_encoding="WKB")

    gpu_owned = read_geoparquet_owned(path, backend="gpu")
    cpu_owned = read_geoparquet_owned(path, backend="cpu")

    _assert_geometry_lists_equal(gpu_owned.to_shapely(), cpu_owned.to_shapely())


# --- WKB multilinestring decode tests ---


def test_read_geoparquet_owned_gpu_wkb_multilinestring_decode_matches_cpu(tmp_path) -> None:
    _require_gpu_decode_runtime()

    path = tmp_path / "multilinestring-wkb.parquet"
    frame = geopandas.GeoDataFrame(
        {
            "geometry": [
                MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]]),
                MultiLineString([]),
                None,
                MultiLineString([[(4, 4), (5, 5)]]),
            ]
        },
        geometry="geometry",
    )
    frame.to_parquet(path, geometry_encoding="WKB")

    gpu_owned = read_geoparquet_owned(path, backend="gpu")
    cpu_owned = read_geoparquet_owned(path, backend="cpu")

    assert gpu_owned.residency is Residency.DEVICE
    assert gpu_owned.device_state is not None
    mls_family = next(iter(gpu_owned.families))
    assert gpu_owned.families[mls_family].host_materialized is False

    _assert_geometry_lists_equal(gpu_owned.to_shapely(), cpu_owned.to_shapely())


def test_read_geoparquet_owned_gpu_wkb_multilinestring_varied_part_counts(tmp_path) -> None:
    _require_gpu_decode_runtime()

    path = tmp_path / "multilinestring-varied-wkb.parquet"
    frame = geopandas.GeoDataFrame(
        {
            "geometry": [
                MultiLineString([
                    [(0, 0), (1, 1), (2, 2)],
                    [(10, 10), (11, 11)],
                    [(20, 20), (21, 21), (22, 22), (23, 23)],
                ]),
                MultiLineString([[(100, 100), (200, 200)]]),
                MultiLineString([]),
                None,
                MultiLineString([[(3, 4), (5, 6)]]),
            ]
        },
        geometry="geometry",
    )
    frame.to_parquet(path, geometry_encoding="WKB")

    gpu_owned = read_geoparquet_owned(path, backend="gpu")
    cpu_owned = read_geoparquet_owned(path, backend="cpu")

    _assert_geometry_lists_equal(gpu_owned.to_shapely(), cpu_owned.to_shapely())


# --- WKB multipolygon decode tests ---


def test_read_geoparquet_owned_gpu_wkb_multipolygon_decode_matches_cpu(tmp_path) -> None:
    _require_gpu_decode_runtime()

    path = tmp_path / "multipolygon-wkb.parquet"
    frame = geopandas.GeoDataFrame(
        {
            "geometry": [
                MultiPolygon([Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])]),
                MultiPolygon([]),
                None,
                MultiPolygon([Polygon([(2, 2), (3, 2), (3, 3), (2, 2)])]),
            ]
        },
        geometry="geometry",
    )
    frame.to_parquet(path, geometry_encoding="WKB")

    gpu_owned = read_geoparquet_owned(path, backend="gpu")
    cpu_owned = read_geoparquet_owned(path, backend="cpu")

    assert gpu_owned.residency is Residency.DEVICE
    assert gpu_owned.device_state is not None
    mpg_family = next(iter(gpu_owned.families))
    assert gpu_owned.families[mpg_family].host_materialized is False

    _assert_geometry_lists_equal(gpu_owned.to_shapely(), cpu_owned.to_shapely())


def test_read_geoparquet_owned_gpu_wkb_multipolygon_varied_structure(tmp_path) -> None:
    _require_gpu_decode_runtime()

    path = tmp_path / "multipolygon-varied-wkb.parquet"
    frame = geopandas.GeoDataFrame(
        {
            "geometry": [
                # Two polygons, one with a hole
                MultiPolygon([
                    Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
                            [[(2, 2), (8, 2), (8, 8), (2, 8), (2, 2)]]),
                    Polygon([(20, 20), (30, 20), (30, 30), (20, 20)]),
                ]),
                # Single simple polygon
                MultiPolygon([Polygon([(100, 100), (200, 100), (200, 200), (100, 100)])]),
                MultiPolygon([]),
                None,
                # Three simple polygons
                MultiPolygon([
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
                    Polygon([(2, 2), (3, 2), (3, 3), (2, 2)]),
                    Polygon([(4, 4), (5, 4), (5, 5), (4, 4)]),
                ]),
            ]
        },
        geometry="geometry",
    )
    frame.to_parquet(path, geometry_encoding="WKB")

    gpu_owned = read_geoparquet_owned(path, backend="gpu")
    cpu_owned = read_geoparquet_owned(path, backend="cpu")

    _assert_geometry_lists_equal(gpu_owned.to_shapely(), cpu_owned.to_shapely())


# --- Mixed all-family WKB decode test ---


def test_read_geoparquet_owned_gpu_wkb_mixed_all_families_decode_matches_cpu(tmp_path) -> None:
    _require_gpu_decode_runtime()

    path = tmp_path / "mixed-all-families-wkb.parquet"
    frame = geopandas.GeoDataFrame(
        {
            "geometry": [
                Point(0, 0),
                LineString([(0, 0), (1, 1)]),
                Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
                MultiPoint([(2, 2), (3, 3)]),
                MultiLineString([[(4, 4), (5, 5)]]),
                MultiPolygon([Polygon([(6, 6), (7, 6), (7, 7), (6, 6)])]),
                None,
            ]
        },
        geometry="geometry",
    )
    frame.to_parquet(path, geometry_encoding="WKB")

    gpu_owned = read_geoparquet_owned(path, backend="gpu")
    cpu_owned = read_geoparquet_owned(path, backend="cpu")

    assert gpu_owned.residency is Residency.DEVICE
    assert gpu_owned.device_state is not None
    assert len(gpu_owned.families) == 6

    _assert_geometry_lists_equal(gpu_owned.to_shapely(), cpu_owned.to_shapely())
