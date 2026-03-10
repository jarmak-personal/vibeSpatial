"""API surface smoke test -- canary for the de-vendoring migration.

Every public name that the ``geopandas`` facade currently exports is imported
and lightly exercised here.  If any phase of the migration breaks the surface,
this test fails immediately.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

# ---------------------------------------------------------------------------
# 1. Top-level names
# ---------------------------------------------------------------------------


def test_core_classes_importable():
    import geopandas

    assert hasattr(geopandas, "GeoDataFrame")
    assert hasattr(geopandas, "GeoSeries")
    assert hasattr(geopandas, "points_from_xy")


def test_runtime_api_importable():
    import geopandas

    assert hasattr(geopandas, "ExecutionMode")
    assert hasattr(geopandas, "RuntimeSelection")
    assert callable(geopandas.get_runtime_selection)
    assert callable(geopandas.get_dispatch_events)
    assert callable(geopandas.get_fallback_events)
    assert callable(geopandas.clear_dispatch_events)
    assert callable(geopandas.clear_fallback_events)


def test_io_functions_importable():
    import geopandas

    assert callable(geopandas.read_file)
    assert callable(geopandas.read_parquet)
    assert callable(geopandas.read_feather)
    assert callable(geopandas.list_layers)


def test_tools_importable():
    import geopandas

    assert callable(geopandas.sjoin)
    assert callable(geopandas.sjoin_nearest)
    assert callable(geopandas.overlay)
    assert callable(geopandas.clip)


def test_options_object():
    import geopandas

    assert hasattr(geopandas, "options")
    assert hasattr(geopandas.options, "display_precision")


# ---------------------------------------------------------------------------
# 2. Submodule aliasing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "submod",
    [
        "geopandas.array",
        "geopandas.base",
        "geopandas.geodataframe",
        "geopandas.geoseries",
        "geopandas.io",
        "geopandas.tools",
        "geopandas.sindex",
        "geopandas.testing",
        "geopandas._compat",
        "geopandas._config",
    ],
)
def test_submodule_aliased(submod):
    """Upstream tests import geopandas.array, geopandas.io etc. directly."""
    import geopandas  # noqa: F401 -- triggers aliasing

    mod = importlib.import_module(submod)
    assert mod is not None


# ---------------------------------------------------------------------------
# 3. Light functional smoke tests
# ---------------------------------------------------------------------------


def test_geodataframe_construction():
    import geopandas

    gdf = geopandas.GeoDataFrame(
        {"val": [1, 2, 3]},
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
    )
    assert len(gdf) == 3
    assert gdf.geometry.name == "geometry"
    assert gdf.crs is None


def test_geoseries_construction():
    import geopandas

    gs = geopandas.GeoSeries([Point(0, 0), Point(1, 1)])
    assert len(gs) == 2


def test_points_from_xy():
    import geopandas

    gs = geopandas.points_from_xy([0, 1], [0, 1])
    assert len(gs) == 2


def test_geodataframe_area():
    import geopandas

    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    gdf = geopandas.GeoDataFrame({"val": [1]}, geometry=[poly])
    areas = gdf.geometry.area
    assert np.isclose(areas.iloc[0], 1.0)


def test_geodataframe_buffer():
    import geopandas

    gdf = geopandas.GeoDataFrame(
        {"val": [1]}, geometry=[Point(0, 0)]
    )
    buffered = gdf.geometry.buffer(1.0)
    assert len(buffered) == 1
    assert buffered.iloc[0].area > 3.0  # roughly pi


def test_runtime_selection_callable():
    import geopandas

    sel = geopandas.get_runtime_selection()
    assert hasattr(sel, "selected")


def test_dispatch_events_roundtrip():
    import geopandas

    geopandas.clear_dispatch_events()
    events = geopandas.get_dispatch_events()
    assert isinstance(events, list)


def test_fallback_events_roundtrip():
    import geopandas

    geopandas.clear_fallback_events()
    events = geopandas.get_fallback_events()
    assert isinstance(events, list)


# ---------------------------------------------------------------------------
# 4. Deep submodule imports used by upstream tests
# ---------------------------------------------------------------------------


def test_array_submodule_exports():
    from geopandas.array import GeometryArray, GeometryDtype, from_shapely

    arr = from_shapely(np.array([Point(0, 0), Point(1, 1)]))
    assert isinstance(arr, GeometryArray)
    assert isinstance(arr.dtype, GeometryDtype)


def test_testing_submodule_exports():
    from geopandas.testing import (
        assert_geodataframe_equal,
        assert_geoseries_equal,
    )

    assert callable(assert_geodataframe_equal)
    assert callable(assert_geoseries_equal)


def test_compat_submodule_exports():
    from geopandas._compat import import_optional_dependency

    assert callable(import_optional_dependency)
