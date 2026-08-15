from __future__ import annotations

import importlib
import sys


def _alias_compat_submodule(name: str) -> None:
    module = importlib.import_module(f"vibespatial.api.io.{name}")
    sys.modules[f"geopandas.io.{name}"] = module
    globals()[name] = module

    compat_package = sys.modules.get("geopandas.io")
    if compat_package is not None:
        setattr(compat_package, name, module)


for _name in ("arrow", "_geoarrow", "file", "sql", "util"):
    _alias_compat_submodule(_name)

del _alias_compat_submodule
del _name
