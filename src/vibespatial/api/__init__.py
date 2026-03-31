import numpy as np
import pandas as pd

import vibespatial.api.datasets
from vibespatial.api import geo_base as base  # GeoPandas compat: geopandas.base
from vibespatial.api import geometry_array as array  # GeoPandas compat: geopandas.array
from vibespatial.api._config import options
from vibespatial.api._version import __version__
from vibespatial.api.geodataframe import GeoDataFrame
from vibespatial.api.geometry_array import points_from_xy
from vibespatial.api.geoseries import GeoSeries
from vibespatial.api.io.arrow import _read_feather as read_feather
from vibespatial.api.io.file import _list_layers as list_layers
from vibespatial.api.io.sql import _read_postgis as read_postgis
from vibespatial.api.tools import clip, overlay, sjoin, sjoin_nearest
from vibespatial.api.tools._show_versions import show_versions
from vibespatial.io.arrow import read_geoparquet as read_parquet
from vibespatial.io.file import read_vector_file as read_file

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
from vibespatial.runtime.dispatch import (
    DispatchEvent,
    clear_dispatch_events,
    get_dispatch_events,
    record_dispatch_event,
)
from vibespatial.runtime.fallbacks import (
    FallbackEvent,
    clear_fallback_events,
    get_fallback_events,
    record_fallback_event,
)


def get_runtime_selection(
    requested: ExecutionMode | str = ExecutionMode.AUTO,
) -> RuntimeSelection:
    return select_runtime(requested)
