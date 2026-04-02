from __future__ import annotations

import numpy as np
import shapely

from vibespatial.geometry.owned import OwnedGeometryArray, from_shapely_geometries


def build_linestring_buffers_cpu(
    lines: OwnedGeometryArray,
    radii: np.ndarray,
    *,
    quad_segs: int,
    cap_style: str = "round",
    join_style: str = "round",
    mitre_limit: float = 5.0,
) -> OwnedGeometryArray:
    lines._ensure_host_state()
    shapely_geoms = lines.to_shapely()
    result_geoms = shapely.buffer(
        np.asarray(shapely_geoms, dtype=object),
        radii,
        quad_segs=quad_segs,
        cap_style=cap_style,
        join_style=join_style,
        mitre_limit=mitre_limit,
    )
    return from_shapely_geometries(list(result_geoms))
