from __future__ import annotations

from typing import Any

import numpy as np
import shapely


def iter_geometry_parts(geometry: Any) -> list[Any]:
    """Return multipart Shapely members through an explicit CPU boundary."""
    return shapely.get_parts(np.asarray([geometry], dtype=object)).tolist()
