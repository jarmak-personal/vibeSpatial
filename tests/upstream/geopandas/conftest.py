# ruff: noqa: I001
import os
import os.path

# Suppress the deprecation warning from the geopandas compatibility shim
# and ensure upstream tests can import geopandas transparently.
os.environ["_VIBESPATIAL_GEOPANDAS_COMPAT"] = "1"

import geopandas
import pytest
import shapely

from tests.upstream.geopandas.tests.util import (
    _NATURALEARTH_CITIES,
    _NATURALEARTH_LOWRES,
    _NYBB,
)

# Vendored files that fail collection due to pandas/arrow incompatibility
# (IndexError in ArrowExtensionArray.__getitem__ at import time).
collect_ignore_glob = [
    "tests/test_array.py",
    "tools/tests/test_sjoin.py",
]

_GEOS_313_OVERLAY_FIXTURE_NODEIDS = {
    "tests/upstream/geopandas/tests/test_overlay.py::test_overlay_nybb[union]",
    "tests/upstream/geopandas/tests/test_overlay.py::test_overlay_nybb[intersection]",
    "tests/upstream/geopandas/tests/test_overlay.py::test_overlay_nybb[symmetric_difference]",
    "tests/upstream/geopandas/tests/test_overlay.py::test_overlay_nybb[identity]",
}


@pytest.fixture(autouse=True)
def add_geopandas(doctest_namespace):
    doctest_namespace["geopandas"] = geopandas


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if shapely.geos_version < (3, 13, 0):
        return

    marker = pytest.mark.xfail(
        reason=(
            "GEOS 3.13 overlay exact areas diverge from the saved QGIS nybb "
            "fixtures; raw Shapely overlay matches vibeSpatial on this stack."
        )
    )
    for item in items:
        if item.nodeid in _GEOS_313_OVERLAY_FIXTURE_NODEIDS:
            item.add_marker(marker)


# Datasets used in our tests


@pytest.fixture(scope="session")
def naturalearth_lowres() -> str:
    # skip if data missing, unless on github actions
    if os.path.isfile(_NATURALEARTH_LOWRES) or os.getenv("GITHUB_ACTIONS"):
        return _NATURALEARTH_LOWRES
    else:
        pytest.skip("Naturalearth lowres dataset not found")


@pytest.fixture(scope="session")
def naturalearth_cities() -> str:
    # skip if data missing, unless on github actions
    if os.path.isfile(_NATURALEARTH_CITIES) or os.getenv("GITHUB_ACTIONS"):
        return _NATURALEARTH_CITIES
    else:
        pytest.skip("Naturalearth cities dataset not found")


@pytest.fixture(scope="session")
def nybb_filename() -> str:
    # skip if data missing, unless on github actions
    if os.path.isfile(_NYBB[len("zip://") :]) or os.getenv("GITHUB_ACTIONS"):
        return _NYBB
    else:
        pytest.skip("NYBB dataset not found")


@pytest.fixture(scope="class")
def _setup_class_nybb_filename(nybb_filename, request):
    """Attach nybb_filename class attribute for unittest style setup_method"""
    request.cls.nybb_filename = nybb_filename
