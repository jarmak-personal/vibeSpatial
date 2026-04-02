from __future__ import annotations

import numpy as np
import shapely

from vibespatial.geometry.owned import OwnedGeometryArray, from_shapely_geometries


def _minimum_clearance_cpu(owned: OwnedGeometryArray) -> np.ndarray:
    """CPU fallback via Shapely."""
    geoms = np.asarray(owned.to_shapely(), dtype=object)
    return shapely.minimum_clearance(geoms)


def _minimum_clearance_line_cpu(owned: OwnedGeometryArray):
    """CPU minimum-clearance line via Shapely."""
    geoms = np.asarray(owned.to_shapely(), dtype=object)
    result_geoms = shapely.minimum_clearance_line(geoms)
    return from_shapely_geometries(result_geoms)
