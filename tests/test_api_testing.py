from __future__ import annotations

from shapely.geometry import GeometryCollection, Polygon

from vibespatial import GeoDataFrame, GeoSeries
from vibespatial.api.testing import (
    assert_geodataframe_equal,
    assert_geoseries_equal,
    geom_almost_equals,
    geom_equals,
)
from vibespatial.api.tools.clip import clip
from vibespatial.testing import strict_native_environment


def test_geom_equals_helper_scalar_broadcast_survives_strict_native() -> None:
    left = GeoSeries(
        [
            Polygon([(0.0, 0.0), (2.2, 0.0), (2.2, 2.2), (0.0, 2.2), (0.0, 0.0)]),
        ]
    )
    right = Polygon([(0.8, 0.8), (2.2, 0.8), (2.2, 2.2), (0.8, 2.2), (0.8, 0.8)])

    with strict_native_environment():
        assert geom_equals(left.intersection(right), right)


def test_geom_almost_equals_helper_scalar_broadcast_survives_strict_native() -> None:
    left = GeoSeries(
        [
            Polygon([(0.8, 0.8), (2.2000001, 0.8), (2.2000001, 2.2000001), (0.8, 2.2000001), (0.8, 0.8)]),
        ]
    )
    right = Polygon([(0.8, 0.8), (2.2, 0.8), (2.2, 2.2), (0.8, 2.2), (0.8, 0.8)])

    with strict_native_environment():
        assert geom_almost_equals(left, right)


def test_assert_geoseries_equal_allows_geometrycollection_under_strict_native() -> None:
    left = GeoSeries([GeometryCollection(), GeometryCollection()])
    right = GeoSeries([GeometryCollection(), GeometryCollection()])

    with strict_native_environment():
        assert_geoseries_equal(left, right)


def test_assert_geodataframe_equal_uses_public_equality_for_clip_polygon_rounding() -> None:
    source = GeoDataFrame(
        [1],
        geometry=[Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])],
        crs="EPSG:3857",
    )
    source["attr2"] = "site-boundary"
    mask = Polygon([(0, 0), (5, 12), (10, 0), (0, 0)])
    expected = GeoDataFrame(
        [1],
        geometry=[mask.intersection(source.geometry.iloc[0])],
        crs="EPSG:3857",
    )
    expected["attr2"] = "site-boundary"

    with strict_native_environment():
        actual = clip(source, mask)
        assert_geodataframe_equal(actual, expected)
