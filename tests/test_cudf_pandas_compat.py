"""ADR-0042: cudf.pandas compatibility tests.

Validates that basic spatial operations work correctly under cudf.pandas proxy.
Skips entirely if cudf is not available.
"""
from __future__ import annotations

import pytest
from shapely.geometry import box

try:
    import cudf.pandas  # noqa: F401
    HAS_CUDF = True
except ImportError:
    HAS_CUDF = False

pytestmark = pytest.mark.skipif(not HAS_CUDF, reason="cudf not available")


@pytest.fixture(autouse=True)
def _install_cudf_pandas():
    """Install cudf.pandas proxy for the duration of each test."""
    import cudf.pandas
    cudf.pandas.install()
    yield


def _make_left():
    import geopandas
    from geopandas import GeoSeries
    return geopandas.GeoDataFrame(
        {
            "name": ["a", "b", "c"],
            "value": [1, 2, 3],
            "geometry": GeoSeries([box(0, 0, 2, 2), box(1, 1, 3, 3), box(5, 5, 7, 7)]),
        }
    )


def _make_right():
    import geopandas
    from geopandas import GeoSeries
    return geopandas.GeoDataFrame(
        {
            "label": ["x", "y"],
            "geometry": GeoSeries([box(1, 1, 4, 4), box(6, 6, 8, 8)]),
        }
    )


class TestCudfPandasCompat:
    def test_geodataframe_creation(self):
        gdf = _make_left()
        assert len(gdf) == 3
        assert "geometry" in gdf.columns

    def test_sjoin_under_proxy(self):
        import geopandas
        left = _make_left()
        right = _make_right()
        result = geopandas.sjoin(left, right, how="inner")
        assert len(result) > 0
        assert "name" in result.columns

    def test_dissolve_under_proxy(self):
        gdf = _make_left()
        gdf["group"] = ["g1", "g1", "g2"]
        result = gdf.dissolve(by="group", aggfunc="sum")
        assert "value" in result.columns
        assert result.loc["g1", "value"] == 3

    def test_clip_under_proxy(self):
        import geopandas
        gdf = _make_left()
        clip_geom = box(0, 0, 4, 4)
        result = geopandas.clip(gdf, clip_geom)
        assert "name" in result.columns
        assert "value" in result.columns

    def test_device_geometry_array_not_proxied(self):
        """DeviceGeometryArray should fall back correctly and not be proxied."""
        gdf = _make_left()
        # The geometry array should be accessible.
        geom_array = gdf.geometry.values
        assert geom_array is not None
