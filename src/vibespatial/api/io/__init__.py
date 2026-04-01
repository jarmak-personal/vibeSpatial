from __future__ import annotations

import importlib
import sys


def _alias_compat_submodule(name: str) -> None:
    module = importlib.import_module(f"vibespatial.api.io.{name}")
    sys.modules.setdefault(f"geopandas.io.{name}", module)


for _name in ("arrow", "_geoarrow"):
    _alias_compat_submodule(_name)

del _alias_compat_submodule
del _name
