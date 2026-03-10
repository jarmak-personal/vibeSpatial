from __future__ import annotations

import geopandas


def test_import_uses_local_package() -> None:
    assert "/vibeSpatial/src/geopandas/__init__.py" in geopandas.__file__


def test_runtime_selection_is_exposed() -> None:
    runtime = geopandas.get_runtime_selection()
    assert runtime.selected in {geopandas.ExecutionMode.CPU, geopandas.ExecutionMode.GPU}


def test_public_submodule_aliases_match_top_level_options() -> None:
    from geopandas import _config

    assert _config.options is geopandas.options
