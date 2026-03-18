from vibespatial.api._config import options

from vibespatial.api.geoseries import GeoSeries
from vibespatial.api.geodataframe import GeoDataFrame
from vibespatial.api.geometry_array import points_from_xy

from vibespatial.io_file import read_vector_file as read_file
from vibespatial.api.io.file import _list_layers as list_layers
from vibespatial.io_arrow import read_geoparquet as read_parquet
from vibespatial.api.io.arrow import _read_feather as read_feather
from vibespatial.api.io.sql import _read_postgis as read_postgis
from vibespatial.api.tools import sjoin, sjoin_nearest
from vibespatial.api.tools import overlay
from vibespatial.api.tools._show_versions import show_versions
from vibespatial.api.tools import clip


import vibespatial.api.datasets

# Backwards-compatible submodule aliases (e.g. geopandas.array -> geometry_array).
from vibespatial.api import geometry_array as array  # noqa: F401
from vibespatial.api import geo_base as base  # noqa: F401

import pandas as pd
import numpy as np

from vibespatial.api._version import __version__

# Re-export vibespatial runtime/dispatch APIs so that code using
# ``import vibespatial.api as geopandas`` can call
# ``geopandas.clear_dispatch_events()`` etc.
from vibespatial.runtime import (
    ExecutionMode,
    RuntimeSelection,
    get_requested_mode,
    select_runtime,
    set_execution_mode,
)
from vibespatial.dispatch import (
    DispatchEvent,
    clear_dispatch_events,
    get_dispatch_events,
    record_dispatch_event,
)
from vibespatial.fallbacks import (
    FallbackEvent,
    clear_fallback_events,
    get_fallback_events,
    record_fallback_event,
)


def get_runtime_selection(
    requested: ExecutionMode | str = ExecutionMode.AUTO,
) -> RuntimeSelection:
    return select_runtime(requested)
