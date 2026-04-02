from __future__ import annotations

from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.runtime.residency import Residency


def build_owned(geoms, *, residency: Residency = Residency.HOST):
    return from_shapely_geometries(list(geoms), residency=residency)
