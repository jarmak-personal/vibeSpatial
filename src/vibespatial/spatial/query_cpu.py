"""CPU-only Shapely helpers for spatial query bounds extraction."""

from __future__ import annotations

import numpy as np
import shapely


def shapely_bounds_array(query_values) -> np.ndarray:
    """Vectorized host bounds extraction for Shapely-backed query inputs."""
    return np.asarray(shapely.bounds(query_values), dtype=np.float64)
