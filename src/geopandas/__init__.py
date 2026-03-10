from __future__ import annotations

import importlib
import os
import sys
import warnings

# Suppress the deprecation warning when loading within upstream test conftest
# or when explicitly opted-in via environment variable.
if not os.environ.get("_VIBESPATIAL_GEOPANDAS_COMPAT"):
    warnings.warn(
        "Importing from the 'geopandas' compatibility shim is deprecated. "
        "Use 'import vibespatial' instead. This shim will be removed in a "
        "future release.",
        DeprecationWarning,
        stacklevel=2,
    )

from vibespatial import (
    ExecutionMode,
    RuntimeSelection,
    clear_dispatch_events as _clear_dispatch_events,
    clear_fallback_events as _clear_fallback_events,
    get_dispatch_events as _get_dispatch_events,
    get_fallback_events as _get_fallback_events,
    select_runtime,
)


_API_ROOT = "vibespatial.api"
_API_PACKAGE = importlib.import_module(_API_ROOT)

# Mapping from geopandas submodule names to vibespatial.api module names.
# Most map 1:1, but array.py -> geometry_array.py and base.py -> geo_base.py.
_SUBMODULE_MAP = {
    "_compat": "_compat",
    "_config": "_config",
    "_decorator": "_decorator",
    "_version": "_version",
    "accessors": "accessors",
    "array": "geometry_array",
    "base": "geo_base",
    "datasets": "datasets",
    "explore": "explore",
    "geodataframe": "geodataframe",
    "geoseries": "geoseries",
    "io": "io",
    "plotting": "plotting",
    "sindex": "sindex",
    "testing": "testing",
    "tools": "tools",
}


def _alias_submodule(geopandas_name: str, api_name: str) -> None:
    sys.modules[f"geopandas.{geopandas_name}"] = importlib.import_module(
        f"{_API_ROOT}.{api_name}"
    )


for _gp_name, _api_name in _SUBMODULE_MAP.items():
    _alias_submodule(_gp_name, _api_name)

# Make aliased submodules accessible as attributes (e.g. geopandas.array).
for _gp_name, _api_name in _SUBMODULE_MAP.items():
    globals()[_gp_name] = importlib.import_module(f"{_API_ROOT}.{_api_name}")


for attribute_name in dir(_API_PACKAGE):
    if attribute_name.startswith("__") and attribute_name not in {"__version__"}:
        continue
    globals()[attribute_name] = getattr(_API_PACKAGE, attribute_name)


def get_runtime_selection(
    requested: ExecutionMode | str = ExecutionMode.AUTO,
) -> RuntimeSelection:
    return select_runtime(requested)


def get_fallback_events(*, clear: bool = False):
    return _get_fallback_events(clear=clear)


def clear_fallback_events() -> None:
    _clear_fallback_events()


def get_dispatch_events(*, clear: bool = False):
    return _get_dispatch_events(clear=clear)


def clear_dispatch_events() -> None:
    _clear_dispatch_events()


__all__ = sorted(
    {
        *[name for name in globals() if not name.startswith("_")],
        "ExecutionMode",
        "RuntimeSelection",
        "clear_dispatch_events",
        "clear_fallback_events",
        "get_dispatch_events",
        "get_fallback_events",
        "get_runtime_selection",
    }
)
