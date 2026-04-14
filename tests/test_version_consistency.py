from __future__ import annotations

import importlib.metadata
import warnings

import vibespatial


def test_runtime_version_matches_distribution_metadata() -> None:
    assert vibespatial.__version__ == importlib.metadata.version("vibespatial")


def test_geopandas_compat_version_matches_vibespatial() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import geopandas

    assert geopandas.__version__ == vibespatial.__version__
