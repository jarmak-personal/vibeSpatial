from __future__ import annotations

import inspect
import logging
import numbers
import operator
import typing
import warnings
from functools import lru_cache
from time import perf_counter as _perf_counter
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
import shapely
import shapely.affinity
import shapely.geometry
import shapely.ops
import shapely.wkt
from pandas.api.extensions import (
    ExtensionArray,
    ExtensionDtype,
    register_extension_dtype,
)
from shapely.geometry.base import BaseGeometry

from vibespatial.api._compat import (
    GEOS_GE_312,
    HAS_PYPROJ,
    requires_pyproj,
)
from vibespatial.api.sindex import SpatialIndex
from vibespatial.constructive.clip_rect import evaluate_geopandas_clip_by_rect
from vibespatial.constructive.make_valid_pipeline import evaluate_geopandas_make_valid
from vibespatial.constructive.stroke import (
    evaluate_geopandas_buffer,
    evaluate_geopandas_offset_curve,
)
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    NULL_TAG,
    TAG_FAMILIES,
    OwnedGeometryArray,
    from_shapely_geometries,
)
from vibespatial.predicates.binary import (
    evaluate_geopandas_binary_predicate,
    supports_binary_predicate,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.provenance import (
    ProvenanceTag,
    attempt_provenance_rewrite,
    make_buffer_tag,
    provenance_rewrites_enabled,
    record_rewrite_event,
)
from vibespatial.spatial.distance_owned import evaluate_geopandas_dwithin

if typing.TYPE_CHECKING:
    import numpy.typing as npt

    from vibespatial.api.geo_base import GeoPandasBase

    if HAS_PYPROJ:
        from pyproj import CRS
    else:
        CRS = Any

if HAS_PYPROJ:
    from pyproj import Transformer

    TransformerFromCRS = lru_cache(Transformer.from_crs)

from vibeproj import Transformer as _VibeTransformer

logger = logging.getLogger(__name__)


def _make_transform_func(src_crs, dst_crs):
    """Return a coordinate transform callable, preferring vibeProj over pyproj."""
    try:
        t = _VibeTransformer.from_crs(src_crs, dst_crs, always_xy=True)

        def _vibe_transform(x, y, z=None):
            return t.transform(x, y, z=z)

        return _vibe_transform
    except Exception:
        pass
    transformer = TransformerFromCRS(src_crs, dst_crs, always_xy=True)
    return transformer.transform





_names = {
    "MISSING": None,
    "NAG": None,
    "POINT": "Point",
    "LINESTRING": "LineString",
    "LINEARRING": "LinearRing",
    "POLYGON": "Polygon",
    "MULTIPOINT": "MultiPoint",
    "MULTILINESTRING": "MultiLineString",
    "MULTIPOLYGON": "MultiPolygon",
    "GEOMETRYCOLLECTION": "GeometryCollection",
}

POLYGON_GEOM_TYPES = {"Polygon", "MultiPolygon"}
LINE_GEOM_TYPES = {"LineString", "MultiLineString", "LinearRing"}
POINT_GEOM_TYPES = {"Point", "MultiPoint"}

type_mapping = {p.value: _names[p.name] for p in shapely.GeometryType}
geometry_type_ids = list(type_mapping.keys())
geometry_type_values = np.array(list(type_mapping.values()), dtype=object)

# Map OwnedGeometryArray family tags to geometry type name strings.
# Used by geom_type to avoid host-side shapely.get_type_id when _owned
# is already materialised.
_TAG_TO_GEOM_TYPE_NAME: dict[int, str] = {
    tag: {
        "point": "Point",
        "linestring": "LineString",
        "polygon": "Polygon",
        "multipoint": "MultiPoint",
        "multilinestring": "MultiLineString",
        "multipolygon": "MultiPolygon",
    }[family.value]
    for tag, family in TAG_FAMILIES.items()
}


def _geom_type_from_tags(owned: OwnedGeometryArray) -> np.ndarray:
    """Vectorised geom_type lookup from owned tags -- no Shapely call."""
    tags = owned.tags
    result = np.empty(len(tags), dtype=object)
    for tag_value, name in _TAG_TO_GEOM_TYPE_NAME.items():
        result[tags == tag_value] = name
    result[tags == NULL_TAG] = None
    return result


def _to_owned_via_wkb(data: np.ndarray) -> OwnedGeometryArray:
    """Build OwnedGeometryArray from a Shapely geometry array via native WKB.

    Uses ``shapely.to_wkb`` (vectorised C) to serialise the entire array,
    then feeds the WKB bytes through the native staged decoder which parses
    binary payloads directly into coordinate buffers without per-geometry
    Python object introspection.

    Falls back to ``from_shapely_geometries`` if the WKB path fails (e.g.
    GeometryCollection or other unsupported WKB types).
    """
    from vibespatial.io.wkb import _decode_native_wkb

    try:
        # shapely.to_wkb is a vectorised C call -- much faster than
        # self._data.tolist() which creates a Python list of Shapely objects.
        missing = shapely.is_missing(data)
        wkb_values: list[bytes | None] = [None] * len(data)
        if not missing.all():
            present_mask = ~missing
            wkb_array = shapely.to_wkb(data[present_mask])
            present_indices = np.flatnonzero(present_mask)
            for i, idx in enumerate(present_indices):
                wkb_values[idx] = bytes(wkb_array[i])
        owned, _plan = _decode_native_wkb(wkb_values)
        return owned
    except Exception:
        # GeometryCollection or other edge cases -- fall back to the
        # original per-geometry path.
        return from_shapely_geometries(data.tolist())


class GeometryDtype(ExtensionDtype):
    type = BaseGeometry
    name = "geometry"
    na_value = None

    @classmethod
    def construct_from_string(cls, string):
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )
        elif string == cls.name:
            return cls()
        else:
            raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'")

    @classmethod
    def construct_array_type(cls):
        return GeometryArray


register_extension_dtype(GeometryDtype)


def _check_crs(
    left: GeoPandasBase, right: GeoPandasBase, allow_none: bool = False
) -> bool:
    """
    Check if the projection of both arrays is the same.

    If allow_none is True, empty CRS is treated as the same.
    """
    if allow_none:
        if not left.crs or not right.crs:
            return True
    if not left.crs == right.crs:
        return False
    return True


def _crs_mismatch_warn(
    left: GeoPandasBase, right: GeoPandasBase, stacklevel: int = 3
) -> None:
    """Raise a CRS mismatch warning with the information on the assigned CRS."""
    if left.crs:
        left_srs = left.crs.to_string()
        left_srs = left_srs if len(left_srs) <= 50 else " ".join([left_srs[:50], "..."])
    else:
        left_srs = None

    if right.crs:
        right_srs = right.crs.to_string()
        right_srs = (
            right_srs if len(right_srs) <= 50 else " ".join([right_srs[:50], "..."])
        )
    else:
        right_srs = None

    warnings.warn(
        "CRS mismatch between the CRS of left geometries "
        "and the CRS of right geometries.\n"
        "Use `to_crs()` to reproject one of "
        "the input geometries to match the CRS of the other.\n\n"
        f"Left CRS: {left_srs}\n"
        f"Right CRS: {right_srs}\n",
        UserWarning,
        stacklevel=stacklevel,
    )


def isna(value: None | float | pd.NA) -> bool:
    """
    Check if scalar value is NA-like (None, np.nan or pd.NA).

    Custom version that only works for scalars (returning True or False),
    as `pd.isna` also works for array-like input returning a boolean array.
    """
    if value is None:
        return True
    elif isinstance(value, float) and np.isnan(value):
        return True
    elif value is pd.NA:
        return True
    else:
        return False


# -----------------------------------------------------------------------------
# Constructors / converters to other formats
# -----------------------------------------------------------------------------


def _is_scalar_geometry(geom) -> bool:
    return isinstance(geom, BaseGeometry)


_FROM_SHAPELY_OWNED_THRESHOLD = 100
"""Threshold for eagerly building OwnedGeometryArray after a Shapely-
computed result (e.g. buffer fallback) that is likely to feed subsequent
GPU operations.  Used in compute paths only — the ``from_shapely``
constructor defers owned construction to ``to_owned()``."""


def from_shapely(data, crs: CRS | None = None) -> GeometryArray:
    """
    Convert a list or array of shapely objects to a GeometryArray.

    Validates the elements.

    Parameters
    ----------
    data : array-like
        list or array of shapely objects
    crs : value, optional
        Coordinate Reference System of the geometry objects. Can be anything accepted by
        :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:4326") or a WKT string.

    """
    if not isinstance(data, np.ndarray):
        arr = np.empty(len(data), dtype=object)
        arr[:] = data
    elif len(data) == 0 and data.dtype == "float64":
        arr = data.astype(object)
    else:
        arr = data

    if not shapely.is_valid_input(arr).all():
        out = []

        for geom in data:
            if isinstance(geom, BaseGeometry):
                out.append(geom)
            elif hasattr(geom, "__geo_interface__"):
                geom = shapely.geometry.shape(geom)
                out.append(geom)
            elif isna(geom):
                out.append(None)
            else:
                raise TypeError(f"Input must be valid geometry objects: {geom}")
        arr = np.array(out, dtype=object)

    # Owned backing is constructed lazily on first GPU use via to_owned().
    # This avoids spending ~2-8 ms per array on owned construction for
    # write-only paths (e.g. to_parquet, to_file) that never need GPU
    # dispatch.
    return GeometryArray(arr, crs=crs)


def to_shapely(geoms: GeometryArray) -> np.ndarray:
    """Convert GeometryArray to numpy object array of shapely objects."""
    if not isinstance(geoms, GeometryArray):
        raise ValueError("'geoms' must be a GeometryArray")
    return geoms._data


def from_wkb(
    data,
    crs: Any | None = None,
    on_invalid: Literal["raise", "warn", "ignore"] = "raise",
) -> GeometryArray:
    """
    Convert a list or array of WKB objects to a GeometryArray.

    Parameters
    ----------
    data : array-like
        list or array of WKB objects
    crs : value, optional
        Coordinate Reference System of the geometry objects. Can be anything accepted by
        :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:4326") or a WKT string.
    on_invalid: {"raise", "warn", "ignore"}, default "raise"
        - raise: an exception will be raised if a WKB input geometry is invalid.
        - warn: a warning will be raised and invalid WKB geometries will be returned as
          None.
        - ignore: invalid WKB geometries will be returned as None without a warning.
        - fix: an effort is made to fix invalid input geometries (e.g. close
          unclosed rings). If this is not possible, they are returned as ``None``
          without a warning. Requires GEOS >= 3.11 and shapely >= 2.1.

    """
    if isinstance(data, ExtensionArray):
        data = data.to_numpy(na_value=None)
    return GeometryArray(shapely.from_wkb(data, on_invalid=on_invalid), crs=crs)


def to_wkb(geoms: GeometryArray, hex: bool = False, **kwargs):
    """Convert GeometryArray to a numpy object array of WKB objects."""
    if not isinstance(geoms, GeometryArray):
        raise ValueError("'geoms' must be a GeometryArray")
    return shapely.to_wkb(geoms, hex=hex, **kwargs)


def from_wkt(
    data,
    crs: Any | None = None,
    on_invalid: Literal["raise", "warn", "ignore"] = "raise",
) -> GeometryArray:
    """
    Convert a list or array of WKT objects to a GeometryArray.

    Parameters
    ----------
    data : array-like
        list or array of WKT objects
    crs : value, optional
        Coordinate Reference System of the geometry objects. Can be anything accepted by
        :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:4326") or a WKT string.
    on_invalid : {"raise", "warn", "ignore"}, default "raise"
        - raise: an exception will be raised if a WKT input geometry is invalid.
        - warn: a warning will be raised and invalid WKT geometries will be
          returned as ``None``.
        - ignore: invalid WKT geometries will be returned as ``None`` without a warning.
        - fix: an effort is made to fix invalid input geometries (e.g. close
          unclosed rings). If this is not possible, they are returned as ``None``
          without a warning. Requires GEOS >= 3.11 and shapely >= 2.1.

    """
    if isinstance(data, ExtensionArray):
        data = data.to_numpy(na_value=None)
    return GeometryArray(shapely.from_wkt(data, on_invalid=on_invalid), crs=crs)


def to_wkt(geoms: GeometryArray, **kwargs):
    """Convert GeometryArray to a numpy object array of WKT objects."""
    if not isinstance(geoms, GeometryArray):
        raise ValueError("'geoms' must be a GeometryArray")
    return shapely.to_wkt(geoms, **kwargs)


def points_from_xy(
    x: npt.ArrayLike, y: npt.ArrayLike, z: npt.ArrayLike = None, crs: Any | None = None
) -> GeometryArray:
    """
    Generate GeometryArray of shapely Point geometries from x, y(, z) coordinates.

    In case of geographic coordinates, it is assumed that longitude is captured by
    ``x`` coordinates and latitude by ``y``.

    Parameters
    ----------
    x, y, z : iterable
    crs : value, optional
        Coordinate Reference System of the geometry objects. Can be anything accepted by
        :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:4326") or a WKT string.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'x': [0, 1, 2], 'y': [0, 1, 2], 'z': [0, 1, 2]})
    >>> df
       x  y  z
    0  0  0  0
    1  1  1  1
    2  2  2  2
    >>> geometry = geopandas.points_from_xy(x=[1, 0], y=[0, 1])
    >>> geometry = geopandas.points_from_xy(df['x'], df['y'], df['z'])
    >>> gdf = geopandas.GeoDataFrame(
    ...     df, geometry=geopandas.points_from_xy(df['x'], df['y']))

    Having geographic coordinates:

    >>> df = pd.DataFrame({'longitude': [-140, 0, 123], 'latitude': [-65, 1, 48]})
    >>> df
       longitude  latitude
    0       -140       -65
    1          0         1
    2        123        48
    >>> geometry = geopandas.points_from_xy(df.longitude, df.latitude, crs="EPSG:4326")

    Returns
    -------
    output : GeometryArray
    """
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    if z is not None:
        z = np.asarray(z, dtype="float64")

    return GeometryArray(shapely.points(x, y, z), crs=crs)


class GeometryArray(ExtensionArray):
    """Class wrapping a numpy array of Shapely objects.

    It also holds the array-based implementations.
    """

    _dtype = GeometryDtype()

    def __init__(self, data, crs: Any | None = None):
        _source_provenance: ProvenanceTag | None = None
        if isinstance(data, self.__class__):
            if not crs:
                crs = data.crs
            _source_provenance = data._provenance
            data = data._data
        elif hasattr(data, "_data") and hasattr(data, "dtype") and hasattr(data.dtype, "name") and data.dtype.name == "device_geometry":
            # Accept DeviceGeometryArray by extracting its Shapely cache
            if not crs and hasattr(data, "crs"):
                crs = data.crs
            data = data._data
        elif not isinstance(data, np.ndarray):
            raise TypeError(
                "'data' should be array of geometry objects. Use from_shapely, "
                "from_wkb, from_wkt functions to construct a GeometryArray."
            )
        elif not data.ndim == 1:
            raise ValueError(
                "'data' should be a 1-dimensional array of geometry objects."
            )
        self._shapely_data = data

        self._crs = None
        self.crs = crs
        self._sindex = None
        self._owned: OwnedGeometryArray | None = None
        self._owned_flat_sindex = None
        self._owned_spatial_input_supported: bool | None = None
        self._provenance: ProvenanceTag | None = _source_provenance

    @classmethod
    def from_owned(cls, owned: OwnedGeometryArray, crs=None) -> GeometryArray:
        """Create a GeometryArray backed by an OwnedGeometryArray without materializing Shapely."""
        obj = object.__new__(cls)
        obj._shapely_data = None
        obj._crs = None
        obj.crs = crs
        obj._sindex = None
        obj._owned = owned
        obj._owned_flat_sindex = None
        obj._owned_spatial_input_supported = None
        obj._provenance = None
        return obj

    @property
    def _data(self) -> np.ndarray:
        if self._shapely_data is None:
            assert self._owned is not None
            self._shapely_data = np.asarray(self._owned.to_shapely(), dtype=object)
        return self._shapely_data

    @property
    def sindex(self) -> SpatialIndex:
        """Spatial index for the geometries in this array."""
        if self._sindex is None:
            self._sindex = SpatialIndex(self._data, geometry_array=self)
        return self._sindex

    @property
    def has_sindex(self) -> bool:
        """Check the existence of the spatial index without generating it.

        Use the `.sindex` attribute on a GeoDataFrame or GeoSeries
        to generate a spatial index if it does not yet exist,
        which may take considerable time based on the underlying index
        implementation.

        Note that the underlying spatial index may not be fully
        initialized until the first use.

        See Also
        --------
        GeoDataFrame.has_sindex

        Returns
        -------
        bool
            `True` if the spatial index has been generated or
            `False` if not.
        """
        return self._sindex is not None

    @property
    def crs(self) -> CRS:
        """The Coordinate Reference System (CRS) represented as a ``pyproj.CRS`` object.

        Returns a ``pyproj.CRS`` or None. When setting, the value
        Coordinate Reference System of the geometry objects. Can be anything accepted by
        :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:4326") or a WKT string.
        """
        return self._crs

    @crs.setter
    def crs(self, value: Any) -> None:
        """Set the value of the crs."""
        if HAS_PYPROJ:
            from pyproj import CRS

            self._crs = None if not value else CRS.from_user_input(value)
        else:
            if value is not None:
                warnings.warn(
                    "Cannot set the CRS, falling back to None. The CRS support requires"
                    " the 'pyproj' package, but it is not installed or does not import"
                    " correctly. The functions depending on CRS will raise an error or"
                    " may produce unexpected results.",
                    UserWarning,
                    stacklevel=2,
                )
            self._crs = None

    def check_geographic_crs(self, stacklevel: int) -> None:
        """Check CRS and warn if the planar operation is done in a geographic CRS."""
        if self.crs and self.crs.is_geographic:
            warnings.warn(
                "Geometry is in a geographic CRS. Results from "
                f"'{inspect.stack()[1].function}' are likely incorrect. "
                "Use 'GeoSeries.to_crs()' to re-project geometries to a "
                "projected CRS before this operation.\n",
                UserWarning,
                stacklevel=stacklevel,
            )

    @property
    def dtype(self) -> GeometryDtype:
        return self._dtype

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx) -> GeometryArray:
        if isinstance(idx, numbers.Integral):
            return self._data[idx]
        elif (
            isinstance(idx, slice)
            and idx.start is None
            and idx.stop is None
            and idx.step is None
        ):
            # special case of a full slice -> preserve the sindex
            # (to ensure view() preserves it as well)
            if self._owned is not None and self._shapely_data is None:
                # Avoid triggering Shapely materialization for identity
                # slice (e.g. pandas view() during DataFrame copy).
                result = GeometryArray.from_owned(self._owned, crs=self.crs)
            else:
                result = GeometryArray(self._data[idx], crs=self.crs)
                result._owned = self._owned
            result._sindex = self._sindex
            result._owned_flat_sindex = self._owned_flat_sindex
            result._owned_spatial_input_supported = self._owned_spatial_input_supported
            result._provenance = self._provenance
            return result

        # array-like, slice
        # validate and convert IntegerArray/BooleanArray
        # to numpy array, pass-through non-array-like indexers
        idx = pd.api.indexers.check_array_indexer(self, idx)

        # Preserve _owned backing through partial indexing so that
        # downstream GPU dispatches don't silently fall back to Shapely.
        if self._owned is not None and isinstance(idx, np.ndarray):
            int_indices = np.flatnonzero(idx) if idx.dtype == bool else idx
            subset_owned = self._owned.take(int_indices)
            return GeometryArray.from_owned(subset_owned, crs=self.crs)

        return GeometryArray(self._data[idx], crs=self.crs)

    def __setitem__(self, key, value):
        # validate and convert IntegerArray/BooleanArray
        # keys to numpy array, pass-through non-array-like indexers
        key = pd.api.indexers.check_array_indexer(self, key)
        if isinstance(value, pd.Series):
            value = value.values
        if isinstance(value, pd.DataFrame):
            value = value.values.flatten()
        if isinstance(value, list | np.ndarray):
            value = from_shapely(value)
        if isinstance(value, GeometryArray):
            if isinstance(key, numbers.Integral):
                raise ValueError("cannot set a single element with an array")
            self._data[key] = value._data
        elif isinstance(value, BaseGeometry) or isna(value):
            if isna(value):
                # internally only use None as missing value indicator
                # but accept others
                value = None
            elif isinstance(value, BaseGeometry):
                value = from_shapely([value])._data[0]
            else:
                raise TypeError("should be valid geometry")
            if isinstance(key, slice | list | np.ndarray):
                value_array = np.empty(1, dtype=object)
                value_array[:] = [value]
                self._data[key] = value_array
            else:
                self._data[key] = value
        else:
            raise TypeError(
                f"Value should be either a BaseGeometry or None, got {value!s}"
            )

        # invalidate spatial index
        self._sindex = None
        self._owned = None
        self._owned_flat_sindex = None
        self._owned_spatial_input_supported = None

        # TODO: use this once pandas-dev/pandas#33457 is fixed
        # if hasattr(value, "crs"):
        #     if value.crs and (value.crs != self.crs):
        #         raise ValueError(
        #             "CRS mismatch between CRS of the passed geometries "
        #             "and CRS of existing geometries."
        #         )

    def __getstate__(self):
        return (shapely.to_wkb(self._data), self._crs)

    def __setstate__(self, state):
        if not isinstance(state, dict):
            # pickle file saved with pygeos
            geoms = shapely.from_wkb(state[0])
            self._crs = state[1]
            self._sindex = None  # pygeos.STRtree could not be pickled yet
            self._shapely_data = geoms
            self.base = None
            self._owned = None
            self._owned_flat_sindex = None
            self._owned_spatial_input_supported = None
            self._provenance = None
        else:
            if "data" in state:
                state["_shapely_data"] = state.pop("data")
            if "_data" in state:
                state["_shapely_data"] = state.pop("_data")
            state.setdefault("_shapely_data", None)
            if "_crs" not in state:
                state["_crs"] = None
            state.setdefault("_owned", None)
            state.setdefault("_owned_flat_sindex", None)
            state.setdefault("_owned_spatial_input_supported", None)
            state.setdefault("_provenance", None)
            self.__dict__.update(state)

    def to_owned(self) -> OwnedGeometryArray:
        if self._owned is None:
            # Prefer the fast vectorized builder (~2x faster than WKB
            # round-trip) when Shapely data is available and all geometry
            # types are supported.  Falls back to the WKB path for
            # unsupported types (Multi*, GeometryCollection).
            from vibespatial.geometry.owned import _from_shapely_vectorized

            owned = _from_shapely_vectorized(self._data)
            if owned is None:
                owned = _to_owned_via_wkb(self._data)
            self._owned = owned
        return self._owned

    def owned_flat_sindex(self):
        owned = self._owned
        if owned is None:
            owned = self.to_owned()
        if self._owned_flat_sindex is None:
            from vibespatial.runtime import ExecutionMode, RuntimeSelection
            from vibespatial.spatial.indexing import build_flat_spatial_index

            self._owned_flat_sindex = build_flat_spatial_index(
                owned,
                runtime_selection=RuntimeSelection(
                    requested=ExecutionMode.AUTO,
                    selected=ExecutionMode.CPU,
                    reason="cached GeometryArray owned flat spatial index",
                ),
            )
        return owned, self._owned_flat_sindex

    def supports_owned_spatial_input(self) -> bool:
        if self._owned_spatial_input_supported is None:
            from vibespatial.spatial.query import supports_owned_spatial_input

            self._owned_spatial_input_supported = bool(supports_owned_spatial_input(self._data))
        return self._owned_spatial_input_supported

    # -------------------------------------------------------------------------
    # Geometry related methods
    # -------------------------------------------------------------------------

    @property
    def is_valid(self):
        if self._owned is not None:
            from vibespatial.constructive.validity import is_valid_owned

            return is_valid_owned(self._owned)
        return shapely.is_valid(self._data)

    def is_valid_reason(self):
        return shapely.is_valid_reason(self._data)

    def is_valid_coverage(self, gap_width: float = 0.0):
        if not GEOS_GE_312:
            raise ImportError("Method 'is_valid_coverage' requires GEOS>=3.12.")
        return bool(shapely.coverage_is_valid(self._data, gap_width=gap_width))

    def invalid_coverage_edges(self, gap_width: float = 0.0):
        if not GEOS_GE_312:
            raise ImportError(
                "Method 'invalid_coverage_edges' requires and GEOS>=3.12."
            )
        return shapely.coverage_invalid_edges(self._data, gap_width=gap_width)

    @property
    def is_empty(self):
        if self._owned is not None:
            owned = self._owned
            # Ensure host buffers are materialized (device-resident arrays
            # may have empty host stubs with zero-length empty_mask).
            owned._ensure_host_state()
            result = np.zeros(owned.row_count, dtype=bool)
            # Null rows carry NULL_TAG (-1) so they never match any family
            # tag and correctly stay False (matching shapely.is_empty(None)).
            for family, buf in owned.families.items():
                tag = FAMILY_TAGS[family]
                rows = np.flatnonzero(owned.tags == tag)
                if rows.size > 0:
                    result[rows] = buf.empty_mask[owned.family_row_offsets[rows]]
            return result
        return shapely.is_empty(self._data)

    @property
    def is_simple(self):
        if self._owned is not None:
            from vibespatial.constructive.validity import is_simple_owned

            return is_simple_owned(self._owned)
        return shapely.is_simple(self._data)

    @property
    def is_ring(self):
        if self._owned is not None:
            from vibespatial.constructive.properties import is_ring_owned

            return is_ring_owned(self._owned)
        return shapely.is_ring(self._data)

    @property
    def is_closed(self):
        if self._owned is not None:
            from vibespatial.constructive.properties import is_closed_owned

            return is_closed_owned(self._owned)
        return shapely.is_closed(self._data)

    @property
    def is_ccw(self):
        if self._owned is not None:
            from vibespatial.constructive.properties import is_ccw_owned

            return is_ccw_owned(self._owned)
        return shapely.is_ccw(self._data)

    @property
    def has_z(self):
        if self._owned is not None:
            # OwnedGeometryArray stores only 2D (x, y) coordinates.
            return np.zeros(self._owned.row_count, dtype=bool)
        return shapely.has_z(self._data)

    @property
    def has_m(self):
        if self._owned is not None:
            # OwnedGeometryArray stores only 2D (x, y) coordinates.
            return np.zeros(self._owned.row_count, dtype=bool)
        return shapely.has_m(self._data)

    @property
    def geom_type(self):
        if self._owned is not None:
            return _geom_type_from_tags(self._owned)
        res = shapely.get_type_id(self._data)
        return geometry_type_values[np.searchsorted(geometry_type_ids, res)]

    @property
    def area(self):
        """Return the area of the geometries in this array.

        Raises a UserWarning if the CRS is geographic, as the area
        calculation is not accurate in that case.

        Note that the area is calculated in the units of the CRS.

        Returns
        -------
        np.ndarray of float
            Area of the geometries.
        """
        self.check_geographic_crs(stacklevel=5)
        if self._owned is not None:
            from vibespatial.constructive.measurement import area_owned

            return area_owned(self._owned)
        return shapely.area(self._data)

    @property
    def length(self):
        self.check_geographic_crs(stacklevel=5)
        if self._owned is not None:
            from vibespatial.constructive.measurement import length_owned

            return length_owned(self._owned)
        return shapely.length(self._data)

    def count_coordinates(self):
        if self._owned is not None:
            from vibespatial.constructive.properties import num_coordinates_owned

            return num_coordinates_owned(self._owned)
        return shapely.get_num_coordinates(self._data)

    def count_geometries(self):
        if self._owned is not None:
            from vibespatial.constructive.properties import num_geometries_owned

            return num_geometries_owned(self._owned)
        return shapely.get_num_geometries(self._data)

    def count_interior_rings(self):
        if self._owned is not None:
            from vibespatial.constructive.properties import num_interior_rings_owned

            return num_interior_rings_owned(self._owned)
        return shapely.get_num_interior_rings(self._data)

    def get_precision(self):
        return shapely.get_precision(self._data)

    def get_geometry(self, index):
        if self._owned is not None:
            from vibespatial.constructive.properties import get_geometry_owned

            result_owned = get_geometry_owned(self._owned, index)
            return GeometryArray.from_owned(result_owned, crs=self.crs)
        return GeometryArray(shapely.get_geometry(self._data, index=index), crs=self.crs)

    #
    # Unary operations that return new geometries
    #

    @property
    def boundary(self) -> GeometryArray:
        if self._owned is not None:
            from vibespatial.constructive.boundary import boundary_owned

            result_owned = boundary_owned(self._owned)
            return GeometryArray.from_owned(result_owned, crs=self.crs)
        return GeometryArray(shapely.boundary(self._data), crs=self.crs)

    @property
    def centroid(self) -> GeometryArray:
        self.check_geographic_crs(stacklevel=5)
        if self._owned is not None:
            from vibespatial.constructive.centroid import centroid_owned

            result_owned = centroid_owned(self._owned)
            return GeometryArray.from_owned(result_owned, crs=self.crs)
        return GeometryArray(shapely.centroid(self._data), crs=self.crs)

    def concave_hull(self, ratio, allow_holes):
        return shapely.concave_hull(self._data, ratio=ratio, allow_holes=allow_holes)

    def constrained_delaunay_triangles(self) -> GeometryArray:
        return GeometryArray(
            shapely.constrained_delaunay_triangles(self._data), crs=self.crs
        )

    @property
    def convex_hull(self) -> GeometryArray:
        """Return the convex hull of the geometries in this array."""
        if self._owned is not None:
            from vibespatial.constructive.convex_hull import convex_hull_owned

            result_owned = convex_hull_owned(self._owned)
            return GeometryArray.from_owned(result_owned, crs=self.crs)
        return GeometryArray(shapely.convex_hull(self._data), crs=self.crs)

    @property
    def envelope(self) -> GeometryArray:
        """Return the envelope of the geometries in this array."""
        if self._owned is not None:
            from vibespatial.constructive.envelope import envelope_owned

            result_owned = envelope_owned(self._owned)
            return GeometryArray.from_owned(result_owned, crs=self.crs)
        return GeometryArray(shapely.envelope(self._data), crs=self.crs)

    def minimum_rotated_rectangle(self):
        """Return the minimum rotated rectangle of the geometries in this array."""
        if self._owned is not None:
            from vibespatial.constructive.minimum_rotated_rectangle import (
                minimum_rotated_rectangle_owned,
            )

            result_owned = minimum_rotated_rectangle_owned(self._owned)
            return GeometryArray.from_owned(result_owned, crs=self.crs)
        return GeometryArray(shapely.oriented_envelope(self._data), crs=self.crs)

    @property
    def exterior(self) -> GeometryArray:
        if self._owned is not None:
            from vibespatial.constructive.exterior import exterior_owned

            result_owned = exterior_owned(self._owned)
            return GeometryArray.from_owned(result_owned, crs=self.crs)
        return GeometryArray(shapely.get_exterior_ring(self._data), crs=self.crs)

    def extract_unique_points(self) -> GeometryArray:
        if self._owned is not None:
            from vibespatial.constructive.extract_unique_points import (
                extract_unique_points_owned,
            )

            result_owned = extract_unique_points_owned(self._owned)
            return GeometryArray.from_owned(result_owned, crs=self.crs)
        return GeometryArray(shapely.extract_unique_points(self._data), crs=self.crs)

    def offset_curve(self, distance, quad_segs=8, join_style="round", mitre_limit=5.0):
        owned, selected = evaluate_geopandas_offset_curve(
            self._data,
            distance,
            quad_segs=quad_segs,
            join_style=join_style,
            mitre_limit=mitre_limit,
        )
        if owned is not None:
            record_dispatch_event(
                surface="geopandas.array.offset_curve",
                operation="offset_curve",
                implementation="owned_stroke_kernel",
                reason="repo-owned stroke kernel claimed the GeoPandas host surface",
                detail=f"rows={len(self)}, join_style={join_style}",
                selected=selected,
            )
            return GeometryArray(np.asarray(owned, dtype=object), crs=self.crs)
        record_dispatch_event(
            surface="geopandas.array.offset_curve",
            operation="offset_curve",
            implementation="shapely_host",
            reason="repo-owned offset-curve kernel is not host-competitive yet",
            detail=f"rows={len(self)}, join_style={join_style}",
            selected=ExecutionMode.CPU,
        )
        return GeometryArray(
            shapely.offset_curve(
                self._data,
                distance,
                quad_segs=quad_segs,
                join_style=join_style,
                mitre_limit=mitre_limit,
            ),
            crs=self.crs,
        )

    @property
    def interiors(self) -> np.ndarray:
        if self._owned is not None:
            from vibespatial.constructive.interiors import interiors_owned

            result_owned = interiors_owned(self._owned)
            # Convert MultiLineString OGA to the GeoPandas-expected format:
            # np.ndarray of object, each element is a list of LinearRings
            # (or None for non-polygon / null rows).
            result_geoms = result_owned.to_shapely()
            from shapely.geometry import LinearRing

            has_non_poly = False
            inner_rings = []
            for g in result_geoms:
                if g is None:
                    has_non_poly = True
                    inner_rings.append(None)
                elif hasattr(g, "geoms"):
                    # MultiLineString parts -> LinearRings
                    inner_rings.append(
                        [LinearRing(part.coords) for part in g.geoms]
                    )
                else:
                    inner_rings.append([])
            if has_non_poly:
                warnings.warn(
                    "Only Polygon objects have interior rings. For other "
                    "geometry types, None is returned.",
                    stacklevel=2,
                )
            data = np.empty(len(inner_rings), dtype=object)
            data[:] = inner_rings
            return data
        # no GeometryArray as result
        has_non_poly = False
        inner_rings = []
        for geom in self._data:
            interior_ring_seq = getattr(geom, "interiors", None)
            # polygon case
            if interior_ring_seq is not None:
                inner_rings.append(list(interior_ring_seq))
            # non-polygon case
            else:
                has_non_poly = True
                inner_rings.append(None)
        if has_non_poly:
            warnings.warn(
                "Only Polygon objects have interior rings. For other "
                "geometry types, None is returned.",
                stacklevel=2,
            )
        # need to allocate empty first in case of all empty lists in inner_rings
        data = np.empty(len(inner_rings), dtype=object)
        data[:] = inner_rings
        return data

    def remove_repeated_points(self, tolerance=0.0) -> GeometryArray:
        if self._owned is not None:
            from vibespatial.constructive.remove_repeated_points import (
                remove_repeated_points_owned,
            )

            result_owned = remove_repeated_points_owned(
                self._owned, tolerance,
            )
            return GeometryArray.from_owned(result_owned, crs=self.crs)
        return GeometryArray(
            shapely.remove_repeated_points(self._data, tolerance=tolerance),
            crs=self.crs,
        )

    def representative_point(self) -> GeometryArray:
        if self._owned is not None:
            from vibespatial.constructive.representative_point import representative_point_owned

            result_owned = representative_point_owned(self._owned)
            return GeometryArray.from_owned(result_owned, crs=self.crs)
        return GeometryArray(shapely.point_on_surface(self._data), crs=self.crs)

    def minimum_bounding_circle(self) -> GeometryArray:
        if self._owned is not None:
            from vibespatial.constructive.minimum_bounding_circle import (
                minimum_bounding_circle_owned,
            )

            result_owned = minimum_bounding_circle_owned(self._owned)
            return GeometryArray.from_owned(result_owned, crs=self.crs)
        return GeometryArray(shapely.minimum_bounding_circle(self._data), crs=self.crs)

    def maximum_inscribed_circle(self, tolerance) -> GeometryArray:
        return GeometryArray(
            shapely.maximum_inscribed_circle(self._data, tolerance=tolerance),
            crs=self.crs,
        )

    def minimum_bounding_radius(self):
        if self._owned is not None:
            from vibespatial.constructive.minimum_bounding_circle import (
                minimum_bounding_radius_owned,
            )

            return minimum_bounding_radius_owned(self._owned)
        return shapely.minimum_bounding_radius(self._data)

    def minimum_clearance(self):
        if self._owned is not None:
            from vibespatial.constructive.minimum_clearance import (
                minimum_clearance_owned,
            )

            return minimum_clearance_owned(self._owned)
        return shapely.minimum_clearance(self._data)

    def minimum_clearance_line(self) -> GeometryArray:
        if self._owned is not None:
            from vibespatial.constructive.minimum_clearance import (
                minimum_clearance_line_owned,
            )

            result_owned = minimum_clearance_line_owned(self._owned)
            return GeometryArray.from_owned(result_owned, crs=self.crs)
        return GeometryArray(shapely.minimum_clearance_line(self._data), crs=self.crs)

    def normalize(self) -> GeometryArray:
        if self._owned is not None:
            from vibespatial.constructive.normalize import normalize_owned

            result_owned = normalize_owned(self._owned)
            return GeometryArray.from_owned(result_owned, crs=self.crs)
        return GeometryArray(shapely.normalize(self._data), crs=self.crs)

    def orient_polygons(self, exterior_cw: bool = False) -> GeometryArray:
        if self._owned is not None:
            from vibespatial.constructive.orient import orient_owned

            result_owned = orient_owned(self._owned, exterior_cw=exterior_cw)
            return GeometryArray.from_owned(result_owned, crs=self.crs)
        return GeometryArray(
            shapely.orient_polygons(self._data, exterior_cw=exterior_cw), crs=self.crs
        )

    def make_valid(
        self,
        method: Literal["linework", "structure"] = "linework",
        keep_collapsed: bool = True,
    ) -> GeometryArray:
        # Dispatch event is recorded inside make_valid_owned with
        # accurate GPU/CPU selection (dispatch framework gap 7 fix).
        result = evaluate_geopandas_make_valid(
            self._data if self._owned is None else None,
            method=method,
            keep_collapsed=keep_collapsed,
            prebuilt_owned=self._owned,
        )
        if result.owned is not None:
            return GeometryArray.from_owned(result.owned, crs=self.crs)
        return GeometryArray(result.geometries, crs=self.crs)

    def reverse(self) -> GeometryArray:
        if self._owned is not None:
            from vibespatial.constructive.reverse import reverse_owned

            result_owned = reverse_owned(self._owned)
            return GeometryArray.from_owned(result_owned, crs=self.crs)
        return GeometryArray(shapely.reverse(self._data), crs=self.crs)

    def segmentize(self, max_segment_length) -> GeometryArray:
        if self._owned is not None:
            from vibespatial.constructive.segmentize import segmentize_owned

            result_owned = segmentize_owned(self._owned, max_segment_length)
            return GeometryArray.from_owned(result_owned, crs=self.crs)
        return GeometryArray(
            shapely.segmentize(self._data, max_segment_length),
            crs=self.crs,
        )

    def force_2d(self) -> GeometryArray:
        return GeometryArray(shapely.force_2d(self._data), crs=self.crs)

    def force_3d(self, z=0) -> GeometryArray:
        return GeometryArray(shapely.force_3d(self._data, z=z), crs=self.crs)

    def transform(self, transformation, include_z: bool = False) -> GeometryArray:
        return GeometryArray(
            shapely.transform(self._data, transformation, include_z=include_z),
            crs=self.crs,
        )

    def line_merge(self, directed: bool = False) -> GeometryArray:
        if self._owned is not None:
            from vibespatial.constructive.line_merge import line_merge_owned

            result_owned = line_merge_owned(self._owned, directed=directed)
            return GeometryArray.from_owned(result_owned, crs=self.crs)
        return GeometryArray(
            shapely.line_merge(self._data, directed=directed), crs=self.crs
        )

    def set_precision(self, grid_size: float, mode="valid_output"):
        if self._owned is not None:
            from vibespatial.constructive.set_precision import set_precision_owned

            result_owned = set_precision_owned(
                self._owned, grid_size=grid_size, mode=mode,
            )
            return GeometryArray.from_owned(result_owned, crs=self.crs)
        return GeometryArray(
            shapely.set_precision(self._data, grid_size=grid_size, mode=mode),
            crs=self.crs,
        )

    #
    # Binary predicates
    #

    @staticmethod
    def _binary_method(op, left, right, **kwargs):
        # Provenance rewrite check: see if the left operand's provenance
        # enables a cheaper operation (e.g. buffer(r).intersects -> dwithin(r))
        rewrite_result = attempt_provenance_rewrite(op, left, right, **kwargs)
        if rewrite_result is not None:
            return rewrite_result

        if isinstance(right, GeometryArray):
            if len(left) != len(right):
                msg = (
                    "Lengths of inputs do not match. "
                    f"Left: {len(left)}, Right: {len(right)}"
                )
                raise ValueError(msg)
            if not _check_crs(left, right):
                _crs_mismatch_warn(left, right, stacklevel=7)
            right_arg = right._owned if right._owned is not None else right._data
        else:
            right_arg = right

        if supports_binary_predicate(op):
            left_arg = left._owned if left._owned is not None else left._data
            result = evaluate_geopandas_binary_predicate(op, left_arg, right_arg, **kwargs)
            if result is not None:
                return result

        # Shapely fallback needs Shapely arrays
        right_shapely = right._data if isinstance(right, GeometryArray) else right_arg
        return getattr(shapely, op)(left._data, right_shapely, **kwargs)

    def covers(self, other):
        return self._binary_method("covers", self, other)

    def covered_by(self, other):
        return self._binary_method("covered_by", self, other)

    def contains(self, other):
        return self._binary_method("contains", self, other)

    def contains_properly(self, other):
        return self._binary_method("contains_properly", self, other)

    def crosses(self, other):
        return self._binary_method("crosses", self, other)

    def disjoint(self, other):
        return self._binary_method("disjoint", self, other)

    def geom_equals(self, other):
        return self._binary_method("equals", self, other)

    def intersects(self, other):
        return self._binary_method("intersects", self, other)

    def overlaps(self, other):
        return self._binary_method("overlaps", self, other)

    def touches(self, other):
        return self._binary_method("touches", self, other)

    def within(self, other):
        return self._binary_method("within", self, other)

    def dwithin(self, other, distance):
        self.check_geographic_crs(stacklevel=6)
        if isinstance(other, GeometryArray):
            if len(self) != len(other):
                msg = (
                    "Lengths of inputs do not match. "
                    f"Left: {len(self)}, Right: {len(other)}"
                )
                raise ValueError(msg)
            if not _check_crs(self, other):
                _crs_mismatch_warn(self, other, stacklevel=7)
            other_data = other._data
            # Pass OwnedGeometryArrays directly when cached, avoiding
            # a Shapely -> list -> OwnedGeometryArray H->D round-trip.
            left_arg = self._owned if self._owned is not None else self._data
            other_arg = other._owned if other._owned is not None else other_data
        else:
            other_data = other
            left_arg = self._owned if self._owned is not None else self._data
            other_arg = other
        result = evaluate_geopandas_dwithin(left_arg, other_arg, distance)
        if result is not None:
            return result
        return shapely.dwithin(self._data, other_data, distance=distance)

    def geom_equals_exact(self, other, tolerance):
        return self._binary_method("equals_exact", self, other, tolerance=tolerance)

    def geom_equals_identical(self, other):
        return self._binary_method("equals_identical", self, other)

    #
    # Binary operations that return new geometries
    #

    def clip_by_rect(self, xmin, ymin, xmax, ymax) -> GeometryArray:
        owned, selected = evaluate_geopandas_clip_by_rect(
            self._data, xmin, ymin, xmax, ymax, prebuilt_owned=self._owned,
        )
        if owned is None:
            record_dispatch_event(
                surface="geopandas.array.clip_by_rect",
                operation="clip_by_rect",
                implementation="shapely_host",
                reason="owned rectangle-clip path is not host-competitive yet",
                detail=f"rows={len(self)}",
                selected=ExecutionMode.CPU,
            )
        else:
            record_dispatch_event(
                surface="geopandas.array.clip_by_rect",
                operation="clip_by_rect",
                implementation="owned_clip_by_rect",
                reason="repo-owned clip path claimed the GeoPandas host surface",
                detail=f"rows={len(self)}",
                selected=selected,
            )
        return GeometryArray(
            shapely.clip_by_rect(self._data, xmin, ymin, xmax, ymax) if owned is None else owned,
            crs=self.crs,
        )

    def difference(self, other, grid_size=None) -> GeometryArray:
        return self._constructive_or_fallback("difference", other, grid_size=grid_size)

    def intersection(self, other, grid_size=None) -> GeometryArray:
        return self._constructive_or_fallback("intersection", other, grid_size=grid_size)

    def symmetric_difference(self, other, grid_size=None) -> GeometryArray:
        return self._constructive_or_fallback(
            "symmetric_difference", other, grid_size=grid_size,
        )

    def union(self, other, grid_size=None) -> GeometryArray:
        return self._constructive_or_fallback("union", other, grid_size=grid_size)

    def _constructive_or_fallback(self, op, other, **kwargs) -> GeometryArray:
        """Dispatch binary constructive ops through owned path when available."""
        if self._owned is not None:
            from vibespatial.constructive.binary_constructive import binary_constructive_owned

            # Coerce other to OwnedGeometryArray.
            if isinstance(other, BaseGeometry):
                # Broadcast scalar: create a 1-row owned array.  The
                # dispatch layer (binary_constructive_owned) handles
                # broadcast-right semantics so only one row is stored.
                other_owned = from_shapely_geometries([other])
            elif isinstance(other, GeometryArray):
                other_owned = other.to_owned()
            else:
                other_owned = None

            if other_owned is not None:
                try:
                    result_owned = binary_constructive_owned(
                        op, self._owned, other_owned, **kwargs,
                    )
                    # binary_constructive_owned records its own dispatch event
                    # with the accurate selected mode (GPU or CPU).
                    return GeometryArray.from_owned(result_owned, crs=self.crs)
                except NotImplementedError:
                    # GeometryCollection or other unsupported family in result;
                    # fall through to the Shapely host path below.
                    pass

        record_dispatch_event(
            surface=f"geopandas.array.{op}",
            operation=op,
            implementation="shapely_host",
            reason="no owned geometry array available or unsupported other type",
            detail=f"rows={len(self)}",
            selected=ExecutionMode.CPU,
        )
        return GeometryArray(
            self._binary_method(op, self, other, **kwargs), crs=self.crs,
        )

    def shortest_line(self, other) -> GeometryArray:
        if self._owned is not None and isinstance(other, GeometryArray):
            if len(self) != len(other):
                msg = (
                    "Lengths of inputs do not match. "
                    f"Left: {len(self)}, Right: {len(other)}"
                )
                raise ValueError(msg)
            if not _check_crs(self, other):
                _crs_mismatch_warn(self, other, stacklevel=7)
            from vibespatial.constructive.shortest_line import shortest_line_owned

            other_owned = other.to_owned()
            result = shortest_line_owned(self._owned, other_owned)
            if isinstance(result, OwnedGeometryArray):
                return GeometryArray.from_owned(result, crs=self.crs)
            # CPU path returns numpy array of Shapely objects
            return GeometryArray(result, crs=self.crs)
        return GeometryArray(
            self._binary_method("shortest_line", self, other), crs=self.crs
        )

    def snap(self, other, tolerance) -> GeometryArray:
        if self._owned is not None and isinstance(other, GeometryArray):
            if len(self) != len(other):
                msg = (
                    "Lengths of inputs do not match. "
                    f"Left: {len(self)}, Right: {len(other)}"
                )
                raise ValueError(msg)
            if not _check_crs(self, other):
                _crs_mismatch_warn(self, other, stacklevel=7)
            from vibespatial.constructive.snap import snap_owned

            other_owned = other.to_owned()
            result = snap_owned(self._owned, other_owned, tolerance)
            if isinstance(result, OwnedGeometryArray):
                return GeometryArray.from_owned(result, crs=self.crs)
            # CPU path returns numpy array of Shapely objects
            return GeometryArray(result, crs=self.crs)
        return GeometryArray(
            self._binary_method("snap", self, other, tolerance=tolerance), crs=self.crs
        )

    def shared_paths(self, other) -> GeometryArray:
        if self._owned is not None and isinstance(other, GeometryArray):
            if len(self) != len(other):
                msg = (
                    "Lengths of inputs do not match. "
                    f"Left: {len(self)}, Right: {len(other)}"
                )
                raise ValueError(msg)
            if not _check_crs(self, other):
                _crs_mismatch_warn(self, other, stacklevel=7)
            from vibespatial.constructive.shared_paths import shared_paths_owned

            other_owned = other.to_owned()
            result = shared_paths_owned(self._owned, other_owned)
            # shared_paths returns numpy array of Shapely GeometryCollection objects
            return GeometryArray(result, crs=self.crs)
        return GeometryArray(
            self._binary_method("shared_paths", self, other), crs=self.crs
        )

    #
    # Other operations
    #

    def distance(self, other):
        self.check_geographic_crs(stacklevel=6)
        if self._owned is not None and isinstance(other, GeometryArray):
            if len(self) != len(other):
                msg = (
                    "Lengths of inputs do not match. "
                    f"Left: {len(self)}, Right: {len(other)}"
                )
                raise ValueError(msg)
            if not _check_crs(self, other):
                _crs_mismatch_warn(self, other, stacklevel=7)
            from vibespatial.spatial.distance_owned import distance_owned

            other_owned = other.to_owned()
            return distance_owned(self._owned, other_owned)
        return self._binary_method("distance", self, other)

    def hausdorff_distance(self, other, **kwargs):
        self.check_geographic_crs(stacklevel=6)
        if (
            self._owned is not None
            and isinstance(other, GeometryArray)
            and other._owned is not None
        ):
            from vibespatial.spatial.distance_metrics import hausdorff_distance_owned

            return hausdorff_distance_owned(
                self._owned, other._owned, densify=kwargs.get("densify"),
            )
        return self._binary_method("hausdorff_distance", self, other, **kwargs)

    def frechet_distance(self, other, **kwargs):
        self.check_geographic_crs(stacklevel=6)
        if (
            self._owned is not None
            and isinstance(other, GeometryArray)
            and other._owned is not None
        ):
            from vibespatial.spatial.distance_metrics import frechet_distance_owned

            return frechet_distance_owned(
                self._owned, other._owned, densify=kwargs.get("densify"),
            )
        return self._binary_method("frechet_distance", self, other, **kwargs)

    def buffer(self, distance, quad_segs: int | None = None, **kwargs):
        if not (isinstance(distance, int | float) and distance == 0):
            self.check_geographic_crs(stacklevel=5)
        if "resolution" in kwargs:
            if quad_segs is not None:
                msg = (
                    "`buffer` received both `quad_segs` and `resolution` but these are "
                    "aliases for the same parameter. Use `quad_segs` only instead."
                )
                raise ValueError(msg)

            msg = (
                "The `resolution` argument to `buffer` is deprecated, `quad_segs` "
                "should be used instead to align with shapely."
            )
            warnings.warn(
                msg,
                category=DeprecationWarning,
                stacklevel=4,
            )
            quad_segs = kwargs.pop("resolution")
        if quad_segs is None:
            quad_segs = 16  # note shapely default is 8, 16 is historical choice

        cap_style = kwargs.get("cap_style", "round")
        join_style = kwargs.get("join_style", "round")
        mitre_limit = float(kwargs.get("mitre_limit", 5.0))
        single_sided = bool(kwargs.get("single_sided", False))

        _rewrites_on = provenance_rewrites_enabled()

        # R5: buffer(0) is the identity operation
        if isinstance(distance, int | float) and distance == 0 and _rewrites_on:
            _t0 = _perf_counter()
            result = GeometryArray(self._data.copy(), crs=self.crs)
            if self._owned is not None:
                result._owned = self._owned
            _elapsed = _perf_counter() - _t0
            record_rewrite_event(
                rule_name="R5_buffer_zero_identity",
                surface="geopandas.array.buffer",
                original_operation="buffer",
                rewritten_operation="identity",
                reason="buffer(0) is the identity operation",
                detail=f"rows={len(self)}, note=identity_copy_time",
                elapsed_seconds=_elapsed,
            )
            return result

        # R6: buffer(a).buffer(b) -> buffer(a+b) for point geometries
        from vibespatial.runtime.provenance import _r6_preconditions_met

        if (
            _rewrites_on
            and self._provenance is not None
            and self._provenance.operation == "buffer"
            and isinstance(distance, int | float)
            and _r6_preconditions_met(self._provenance, distance, cap_style, join_style)
        ):
            prev_distance = self._provenance.get_param("distance")
            merged_distance = prev_distance + distance
            source = self._provenance.source_array
            _t0 = _perf_counter()
            merged_result = source.buffer(merged_distance, quad_segs=quad_segs, **kwargs)
            _elapsed = _perf_counter() - _t0
            record_rewrite_event(
                rule_name="R6_buffer_chain_merge",
                surface="geopandas.array.buffer",
                original_operation="buffer",
                rewritten_operation="buffer",
                reason="buffer(a).buffer(b) merged to buffer(a+b)",
                detail=f"a={prev_distance}, b={distance}, merged={merged_distance}",
                elapsed_seconds=_elapsed,
            )
            return merged_result

        buffer_result, selected = evaluate_geopandas_buffer(
            self._data,
            distance,
            quad_segs=quad_segs,
            cap_style=cap_style,
            join_style=join_style,
            mitre_limit=mitre_limit,
            single_sided=single_sided,
            prebuilt_owned=self._owned,
        )
        if buffer_result is not None:
            record_dispatch_event(
                surface="geopandas.array.buffer",
                operation="buffer",
                implementation="owned_stroke_kernel",
                reason="repo-owned stroke kernel claimed the GeoPandas host surface",
                detail=f"rows={len(self)}, quad_segs={quad_segs}",
                selected=selected,
            )
            if isinstance(buffer_result, OwnedGeometryArray):
                result = GeometryArray.from_owned(buffer_result, crs=self.crs)
            else:
                result = GeometryArray(np.asarray(buffer_result, dtype=object), crs=self.crs)
            result._provenance = make_buffer_tag(self, distance, cap_style, join_style, single_sided, quad_segs)
            return result
        record_dispatch_event(
            surface="geopandas.array.buffer",
            operation="buffer",
            implementation="shapely_host",
            reason="repo-owned buffer kernel is not host-competitive yet",
            detail=f"rows={len(self)}, quad_segs={quad_segs}",
            selected=ExecutionMode.CPU,
        )

        buf_arr = shapely.buffer(self._data, distance, quad_segs=quad_segs, **kwargs)
        result = GeometryArray(buf_arr, crs=self.crs)
        # Eagerly build owned backing so downstream operations (dissolve,
        # sjoin, clip) can dispatch to GPU instead of falling back to
        # Shapely.  The vectorized builder is fast (~2 ms / 1 k polygons).
        if len(buf_arr) >= _FROM_SHAPELY_OWNED_THRESHOLD:
            from vibespatial.runtime import has_gpu_runtime

            if has_gpu_runtime():
                from vibespatial.geometry.owned import _from_shapely_vectorized

                owned = _from_shapely_vectorized(buf_arr)
                if owned is not None:
                    result._owned = owned
        result._provenance = make_buffer_tag(self, distance, cap_style, join_style, single_sided, quad_segs)
        return result

    def interpolate(self, distance, normalized: bool = False) -> GeometryArray:
        self.check_geographic_crs(stacklevel=5)
        if self._owned is not None:
            from vibespatial.constructive.linear_ref import interpolate_owned

            result_owned = interpolate_owned(self._owned, distance, normalized=normalized)
            return GeometryArray.from_owned(result_owned, crs=self.crs)
        return GeometryArray(
            shapely.line_interpolate_point(self._data, distance, normalized=normalized),
            crs=self.crs,
        )

    def simplify(self, tolerance, preserve_topology: bool = True) -> GeometryArray:
        # R7: simplify(0) is the identity operation
        if isinstance(tolerance, int | float) and tolerance == 0 and provenance_rewrites_enabled():
            _t0 = _perf_counter()
            result = GeometryArray(self._data.copy(), crs=self.crs)
            if self._owned is not None:
                result._owned = self._owned
            _elapsed = _perf_counter() - _t0
            record_rewrite_event(
                rule_name="R7_simplify_zero_identity",
                surface="geopandas.array.simplify",
                original_operation="simplify",
                rewritten_operation="identity",
                reason="simplify(0) is the identity operation",
                detail=f"rows={len(self)}, note=identity_copy_time",
                elapsed_seconds=_elapsed,
            )
            return result
        if self._owned is not None:
            from vibespatial.constructive.simplify import simplify_owned

            result_owned = simplify_owned(
                self._owned, tolerance, preserve_topology=preserve_topology,
            )
            return GeometryArray.from_owned(result_owned, crs=self.crs)
        return GeometryArray(
            shapely.simplify(
                self._data, tolerance, preserve_topology=preserve_topology
            ),
            crs=self.crs,
        )

    def simplify_coverage(
        self, tolerance, simplify_boundary: bool = True
    ) -> GeometryArray:
        if not GEOS_GE_312:
            raise ImportError(
                "'simplify_coverage' requires shapely>=2.1 and GEOS>=3.12."
            )
        return GeometryArray(
            shapely.coverage_simplify(
                self._data, tolerance, simplify_boundary=simplify_boundary
            ),
            crs=self.crs,
        )

    def project(self, other, normalized=False):
        if (
            not normalized
            and self._owned is not None
            and isinstance(other, GeometryArray)
            and other._owned is not None
        ):
            from vibespatial.constructive.linear_ref import project_owned

            return project_owned(self._owned, other._owned)
        if isinstance(other, GeometryArray):
            other = other._data
        return shapely.line_locate_point(self._data, other, normalized=normalized)

    def relate(self, other):
        if self._owned is not None:
            from vibespatial.predicates.relate import relate_de9im

            if isinstance(other, GeometryArray):
                other_owned = other.to_owned()
            else:
                other_owned = from_shapely_geometries(
                    list(other) if isinstance(other, np.ndarray) else [other]
                )

            if other_owned is not None:
                try:
                    return relate_de9im(self._owned, other_owned)
                except Exception:
                    logger.debug(
                        "GPU relate_de9im failed, falling back to Shapely",
                        exc_info=True,
                    )

        if isinstance(other, GeometryArray):
            other = other._data
        return shapely.relate(self._data, other)

    def relate_pattern(self, other, pattern):
        if self._owned is not None:
            from vibespatial.predicates.relate import relate_pattern_match

            if isinstance(other, GeometryArray):
                other_owned = other.to_owned()
            else:
                other_owned = from_shapely_geometries(
                    list(other) if isinstance(other, np.ndarray) else [other]
                )

            if other_owned is not None:
                try:
                    return relate_pattern_match(self._owned, other_owned, pattern)
                except Exception:
                    logger.debug(
                        "GPU relate_pattern_match failed, falling back to Shapely",
                        exc_info=True,
                    )

        if isinstance(other, GeometryArray):
            other = other._data
        return shapely.relate_pattern(self._data, other, pattern)

    #
    # Reduction operations that return a Shapely geometry
    #

    def unary_union(self):
        warnings.warn(
            "The 'unary_union' attribute is deprecated, "
            "use the 'union_all' method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # GPU path: delegate to unary_union_gpu if owned data is available.
        if self._owned is not None:
            result = self._try_gpu_reduction("unary_union")
            if result is not None:
                return result
        return self.union_all()

    def union_all(self, method="unary", grid_size=None):
        if method != "unary" and grid_size is not None:
            raise ValueError(f"grid_size is not supported for method '{method}'.")
        if method == "coverage":
            # GPU path: coverage_union_all_gpu.
            if self._owned is not None:
                result = self._try_gpu_reduction(
                    "coverage_union_all",
                )
                if result is not None:
                    return result
            return shapely.coverage_union_all(self._data)
        elif method == "unary":
            # GPU path: union_all_gpu with optional grid_size.
            if self._owned is not None:
                result = self._try_gpu_reduction(
                    "union_all", grid_size=grid_size,
                )
                if result is not None:
                    return result
            return shapely.union_all(self._data, grid_size=grid_size)
        elif method == "disjoint_subset":
            if self._owned is not None:
                from vibespatial.constructive.union_all import (
                    disjoint_subset_union_all_owned,
                )

                result_owned = disjoint_subset_union_all_owned(self._owned)
                if result_owned is not None:
                    # union_all returns a single Shapely geometry, so
                    # materialise the 1-row OGA to Shapely for API compat.
                    result_geoms = result_owned.to_shapely()
                    if result_geoms:
                        return result_geoms[0]
                # None return means mixed families -- fall through to Shapely.
            if not GEOS_GE_312:
                raise ImportError(
                    "Method 'disjoin_subset' requires shapely>=2.1 and GEOS>=3.12."
                )
            return shapely.disjoint_subset_union_all(self._data)
        else:
            raise ValueError(
                f"Method '{method}' not recognized. Use 'coverage', 'unary' or "
                "'disjoint_subset'."
            )

    def intersection_all(self):
        # GPU path: intersection_all_gpu.
        if self._owned is not None:
            result = self._try_gpu_reduction("intersection_all")
            if result is not None:
                return result
        return shapely.intersection_all(self._data)

    def _try_gpu_reduction(self, op: str, **kwargs):
        """Attempt a GPU global reduction, returning a Shapely geometry or None.

        Imports and calls the appropriate GPU reduction function from
        vibespatial.constructive.union_all.  Returns None if the GPU
        path is unavailable or fails, signalling the caller to fall
        through to the Shapely CPU path.
        """
        try:
            from vibespatial.constructive.union_all import (
                coverage_union_all_gpu_owned,
                intersection_all_gpu_owned,
                unary_union_gpu_owned,
                union_all_gpu_owned,
            )

            dispatch = {
                "union_all": union_all_gpu_owned,
                "coverage_union_all": coverage_union_all_gpu_owned,
                "intersection_all": intersection_all_gpu_owned,
                "unary_union": unary_union_gpu_owned,
            }
            fn = dispatch.get(op)
            if fn is None:
                return None
            result_owned = fn(self._owned, **kwargs)
            if result_owned is not None:
                result_geoms = result_owned.to_shapely()
                if result_geoms:
                    return result_geoms[0]
        except Exception:
            pass
        return None

    #
    # Affinity operations
    #

    @staticmethod
    def _affinity_method(op, left, *args, **kwargs):
        # not all shapely.affinity methods can handle empty geometries:
        # affine_transform itself works (as well as translate), but rotate, scale
        # and skew fail (they try to unpack the bounds).
        # Here: consistently returning empty geom for input empty geom
        out = []
        for geom in left:
            if geom is None or geom.is_empty:
                res = geom
            else:
                res = getattr(shapely.affinity, op)(geom, *args, **kwargs)
            out.append(res)
        data = np.empty(len(left), dtype=object)
        data[:] = out
        return data

    def affine_transform(self, matrix) -> GeometryArray:
        if self._owned is not None:
            from vibespatial.constructive.affine_transform import affine_transform_owned

            result_owned = affine_transform_owned(self._owned, matrix)
            return GeometryArray.from_owned(result_owned, crs=self.crs)
        return GeometryArray(
            self._affinity_method("affine_transform", self._data, matrix),
            crs=self.crs,
        )

    def translate(
        self, xoff: float = 0.0, yoff: float = 0.0, zoff: float = 0.0
    ) -> GeometryArray:
        if self._owned is not None:
            from vibespatial.constructive.affine_transform import translate_owned

            result_owned = translate_owned(self._owned, xoff, yoff, zoff)
            return GeometryArray.from_owned(result_owned, crs=self.crs)
        return GeometryArray(
            self._affinity_method("translate", self._data, xoff, yoff, zoff),
            crs=self.crs,
        )

    def rotate(
        self, angle, origin="center", use_radians: bool = False
    ) -> GeometryArray:
        if self._owned is not None:
            from vibespatial.constructive.affine_transform import rotate_owned

            result_owned = rotate_owned(
                self._owned, angle, origin=origin, use_radians=use_radians,
            )
            return GeometryArray.from_owned(result_owned, crs=self.crs)
        return GeometryArray(
            self._affinity_method(
                "rotate", self._data, angle, origin=origin, use_radians=use_radians
            ),
            crs=self.crs,
        )

    def scale(
        self,
        xfact: float = 1.0,
        yfact: float = 1.0,
        zfact: float = 1.0,
        origin="center",
    ) -> GeometryArray:
        if self._owned is not None:
            from vibespatial.constructive.affine_transform import scale_owned

            result_owned = scale_owned(
                self._owned, xfact, yfact, zfact, origin=origin,
            )
            return GeometryArray.from_owned(result_owned, crs=self.crs)
        return GeometryArray(
            self._affinity_method(
                "scale", self._data, xfact, yfact, zfact, origin=origin
            ),
            crs=self.crs,
        )

    def skew(
        self,
        xs: float = 0.0,
        ys: float = 0.0,
        origin="center",
        use_radians: bool = False,
    ) -> GeometryArray:
        if self._owned is not None:
            from vibespatial.constructive.affine_transform import skew_owned

            result_owned = skew_owned(
                self._owned, xs, ys, origin=origin, use_radians=use_radians,
            )
            return GeometryArray.from_owned(result_owned, crs=self.crs)
        return GeometryArray(
            self._affinity_method(
                "skew", self._data, xs, ys, origin=origin, use_radians=use_radians
            ),
            crs=self.crs,
        )

    @requires_pyproj
    def to_crs(self, crs: Any | None = None, epsg: int | None = None) -> GeometryArray:
        """Transform all geometries to a different coordinate reference system.

        Transform all geometries in a GeometryArray to a different coordinate
        reference system.  The ``crs`` attribute on the current GeometryArray must
        be set.  Either ``crs`` or ``epsg`` may be specified for output.

        This method will transform all points in all objects.  It has no notion
        of projecting entire geometries.  All segments joining points are
        assumed to be lines in the current projection, not geodesics.  Objects
        crossing the dateline (or other projection boundary) will have
        undesirable behavior.

        Parameters
        ----------
        crs : pyproj.CRS, optional if `epsg` is specified
            The value can be anything accepted
            by :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:4326") or a WKT string.
        epsg : int, optional if `crs` is specified
            EPSG code specifying output projection.

        Returns
        -------
        GeometryArray

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> from geopandas.array import from_shapely, to_wkt
        >>> a = from_shapely([Point(1, 1), Point(2, 2), Point(3, 3)], crs=4326)
        >>> to_wkt(a)
        array(['POINT (1 1)', 'POINT (2 2)', 'POINT (3 3)'], dtype=object)
        >>> a.crs  # doctest: +SKIP
        <Geographic 2D CRS: EPSG:4326>
        Name: WGS 84
        Axis Info [ellipsoidal]:
        - Lat[north]: Geodetic latitude (degree)
        - Lon[east]: Geodetic longitude (degree)
        Area of Use:
        - name: World
        - bounds: (-180.0, -90.0, 180.0, 90.0)
        Datum: World Geodetic System 1984
        - Ellipsoid: WGS 84
        - Prime Meridian: Greenwich

        >>> a = a.to_crs(3857)
        >>> to_wkt(a)
        array(['POINT (111319.490793 111325.142866)',
               'POINT (222638.981587 222684.208506)',
               'POINT (333958.47238 334111.171402)'], dtype=object)
        >>> a.crs  # doctest: +SKIP
        <Projected CRS: EPSG:3857>
        Name: WGS 84 / Pseudo-Mercator
        Axis Info [cartesian]:
        - X[east]: Easting (metre)
        - Y[north]: Northing (metre)
        Area of Use:
        - name: World - 85°S to 85°N
        - bounds: (-180.0, -85.06, 180.0, 85.06)
        Coordinate Operation:
        - name: Popular Visualisation Pseudo-Mercator
        - method: Popular Visualisation Pseudo Mercator
        Datum: World Geodetic System 1984
        - Ellipsoid: WGS 84
        - Prime Meridian: Greenwich

        """
        from pyproj import CRS

        if self.crs is None:
            raise ValueError(
                "Cannot transform naive geometries.  "
                "Please set a crs on the object first."
            )
        if crs is not None:
            crs = CRS.from_user_input(crs)
        elif epsg is not None:
            crs = CRS.from_epsg(epsg)
        else:
            raise ValueError("Must pass either crs or epsg.")

        # skip if the input CRS and output CRS are the exact same
        if self.crs.is_exact_same(crs):
            return self

        # Owned-path dispatch: reproject on-device via vibeProj when DGA
        # backing is available, avoiding Shapely materialization entirely.
        if self._owned is not None:
            try:
                from vibespatial.geometry.device_array import _to_crs_owned
                from vibespatial.runtime.dispatch import record_dispatch_event

                dga_result = _to_crs_owned(self._owned, self.crs, crs)
                record_dispatch_event(
                    surface="geopandas.array.to_crs",
                    operation="to_crs",
                    implementation="vibeproj_device",
                    reason="owned backing available, vibeProj on-device transform",
                    detail=f"rows={len(self)}, src={self.crs}, dst={crs}",
                    selected=ExecutionMode.GPU,
                )
                return GeometryArray.from_owned(dga_result._owned, crs=crs)
            except (ImportError, NotImplementedError):
                # vibeProj not installed or CRS not supported -- fall through
                # to Shapely/pyproj path.
                pass

        transform_func = _make_transform_func(self.crs, crs)

        new_data = transform(self._data, transform_func)
        return GeometryArray(new_data, crs=crs)

    @requires_pyproj
    def estimate_utm_crs(self, datum_name: str = "WGS 84") -> CRS:
        """Return the estimated UTM CRS based on the bounds of the dataset.

        .. versionadded:: 0.9

        .. note:: Requires pyproj 3+

        Parameters
        ----------
        datum_name : str, optional
            The name of the datum to use in the query. Default is WGS 84.

        Returns
        -------
        pyproj.CRS

        Examples
        --------
        >>> import geodatasets
        >>> df = geopandas.read_file(
        ...     geodatasets.get_path("geoda.chicago_commpop")
        ... )
        >>> df.geometry.values.estimate_utm_crs()  # doctest: +SKIP
        <Derived Projected CRS: EPSG:32616>
        Name: WGS 84 / UTM zone 16N
        Axis Info [cartesian]:
        - E[east]: Easting (metre)
        - N[north]: Northing (metre)
        Area of Use:
        - name: Between 90°W and 84°W, northern hemisphere between equator and 84°N,...
        - bounds: (-90.0, 0.0, -84.0, 84.0)
        Coordinate Operation:
        - name: UTM zone 16N
        - method: Transverse Mercator
        Datum: World Geodetic System 1984 ensemble
        - Ellipsoid: WGS 84
        - Prime Meridian: Greenwich
        """
        from pyproj import CRS
        from pyproj.aoi import AreaOfInterest
        from pyproj.database import query_utm_crs_info

        if not self.crs:
            raise RuntimeError("crs must be set to estimate UTM CRS.")

        minx, miny, maxx, maxy = self.total_bounds
        if self.crs.is_geographic:
            x_center = np.mean([minx, maxx])
            y_center = np.mean([miny, maxy])
        # ensure using geographic coordinates
        else:
            t = _VibeTransformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
            minx, miny, maxx, maxy = t.transform_bounds(
                minx, miny, maxx, maxy
            )
            y_center = np.mean([miny, maxy])
            # crossed the antimeridian
            if minx > maxx:
                # shift maxx from [-180,180] to [0,360]
                # so both numbers are positive for center calculation
                # Example: -175 to 185
                maxx += 360
                x_center = np.mean([minx, maxx])
                # shift back to [-180,180]
                x_center = ((x_center + 180) % 360) - 180
            else:
                x_center = np.mean([minx, maxx])

        utm_crs_list = query_utm_crs_info(
            datum_name=datum_name,
            area_of_interest=AreaOfInterest(
                west_lon_degree=x_center,
                south_lat_degree=y_center,
                east_lon_degree=x_center,
                north_lat_degree=y_center,
            ),
        )
        try:
            return CRS.from_epsg(utm_crs_list[0].code)
        except IndexError:
            raise RuntimeError("Unable to determine UTM CRS") from None

    #
    # Coordinate related properties
    #

    @property
    def x(self):
        """Return the x location of point geometries in a GeoSeries."""
        if self._owned is not None:
            point_tag = FAMILY_TAGS[GeometryFamily.POINT]
            non_null = ~self.isna()
            if non_null.any() and (self._owned.tags[non_null] == point_tag).all():
                from vibespatial.constructive.properties import get_x_owned

                return get_x_owned(self._owned)
        if (self.geom_type[~self.isna()] == "Point").all():
            empty = self.is_empty
            if empty.any():
                nonempty = ~empty
                coords = np.full_like(nonempty, dtype=float, fill_value=np.nan)
                coords[nonempty] = shapely.get_x(self._data[nonempty])
                return coords
            else:
                return shapely.get_x(self._data)
        else:
            message = "x attribute access only provided for Point geometries"
            raise ValueError(message)

    @property
    def y(self):
        """Return the y location of point geometries in a GeoSeries."""
        if self._owned is not None:
            point_tag = FAMILY_TAGS[GeometryFamily.POINT]
            non_null = ~self.isna()
            if non_null.any() and (self._owned.tags[non_null] == point_tag).all():
                from vibespatial.constructive.properties import get_y_owned

                return get_y_owned(self._owned)
        if (self.geom_type[~self.isna()] == "Point").all():
            empty = self.is_empty
            if empty.any():
                nonempty = ~empty
                coords = np.full_like(nonempty, dtype=float, fill_value=np.nan)
                coords[nonempty] = shapely.get_y(self._data[nonempty])
                return coords
            else:
                return shapely.get_y(self._data)
        else:
            message = "y attribute access only provided for Point geometries"
            raise ValueError(message)

    @property
    def z(self):
        """Return the z location of point geometries in a GeoSeries."""
        if (self.geom_type[~self.isna()] == "Point").all():
            empty = self.is_empty
            if empty.any():
                nonempty = ~empty
                coords = np.full_like(nonempty, dtype=float, fill_value=np.nan)
                coords[nonempty] = shapely.get_z(self._data[nonempty])
                return coords
            else:
                return shapely.get_z(self._data)
        else:
            message = "z attribute access only provided for Point geometries"
            raise ValueError(message)

    @property
    def m(self):
        """Return the m coordinate of point geometries in a GeoSeries."""
        if (self.geom_type[~self.isna()] == "Point").all():
            empty = self.is_empty
            if empty.any():
                nonempty = ~empty
                coords = np.full_like(nonempty, dtype=float, fill_value=np.nan)
                coords[nonempty] = shapely.get_m(self._data[nonempty])
                return coords
            else:
                return shapely.get_m(self._data)
        else:
            message = "m attribute access only provided for Point geometries"
            raise ValueError(message)

    @property
    def bounds(self):
        if self._owned is not None:
            from vibespatial.kernels.core.geometry_analysis import (
                compute_geometry_bounds,
            )

            return compute_geometry_bounds(self._owned)
        return shapely.bounds(self._data)

    @property
    def total_bounds(self):
        if len(self) == 0:
            return np.array([np.nan, np.nan, np.nan, np.nan])
        if self._owned is not None:
            from vibespatial.kernels.core.geometry_analysis import (
                compute_total_bounds,
            )

            return np.array(compute_total_bounds(self._owned))
        b = self.bounds
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", r"All-NaN slice encountered", RuntimeWarning
            )
            return np.array(
                (
                    np.nanmin(b[:, 0]),  # minx
                    np.nanmin(b[:, 1]),  # miny
                    np.nanmax(b[:, 2]),  # maxx
                    np.nanmax(b[:, 3]),  # maxy
                )
            )

    # -------------------------------------------------------------------------
    # general array like compat
    # -------------------------------------------------------------------------

    @property
    def size(self):
        if self._shapely_data is not None:
            return self._shapely_data.size
        return self._owned.row_count

    @property
    def shape(self) -> tuple:
        return (self.size,)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def copy(self, *args, **kwargs) -> GeometryArray:
        # still taking args/kwargs for compat with pandas 0.24
        if self._owned is not None and self._shapely_data is None:
            # Preserve owned backing without triggering Shapely materialization.
            # __setitem__ invalidates _owned on mutation, so sharing is safe
            # for the _make_valid pattern (copy → mutate subset → _owned cleared).
            # Note: the shared OwnedGeometryArray has mutable device_state;
            # callers that trigger move_to(HOST) on one copy affect the other.
            # This is acceptable for the overlay pipeline where both copies
            # operate on the same data at the same residency.
            result = GeometryArray.from_owned(self._owned, crs=self._crs)
        else:
            result = GeometryArray(self._data.copy(), crs=self._crs)
            if self._owned is not None:
                result._owned = self._owned
        result._provenance = self._provenance
        return result

    def take(self, indices, allow_fill: bool = False, fill_value=None) -> GeometryArray:
        from pandas.api.extensions import take

        if allow_fill:
            if fill_value is None or pd.isna(fill_value):
                fill_value = None
            elif not _is_scalar_geometry(fill_value):
                raise TypeError("provide geometry or None as fill value")

        # Owned-path: OwnedGeometryArray.take() operates at buffer level
        # without Shapely materialization.  Preserves DGA chain for downstream
        # dispatch (e.g., overlay intersection via binary_constructive_owned).
        if self._owned is not None:
            idx_arr = np.asarray(indices)
            # When allow_fill is True, pandas uses -1 as the fill sentinel.
            # If no indices are negative, no filling occurs and we can use
            # the owned fast path to preserve GPU dispatch capability.
            needs_fill = allow_fill and int(idx_arr.size) > 0 and bool(np.any(idx_arr < 0))
            if not needs_fill:
                result_owned = self._owned.take(idx_arr)
                return GeometryArray.from_owned(result_owned, crs=self.crs)

        result = take(self._data, indices, allow_fill=allow_fill, fill_value=fill_value)
        if allow_fill and fill_value is None:
            result[~shapely.is_valid_input(result)] = None
        return GeometryArray(result, crs=self.crs)

    # compat for pandas < 3.0
    def _pad_or_backfill(
        self, method, limit=None, limit_area=None, copy=True, **kwargs
    ):
        kwargs["limit_area"] = limit_area
        return super()._pad_or_backfill(method=method, limit=limit, copy=copy, **kwargs)

    def fillna(
        self, value=None, method=None, limit: int | None = None, copy: bool = True
    ) -> GeometryArray:
        """
        Fill NA values with geometry (or geometries) or using the specified method.

        Parameters
        ----------
        value : shapely geometry object or GeometryArray
            If a geometry value is passed it is used to fill all missing values.
            Alternatively, a GeometryArray 'value' can be given. It's expected
            that the GeometryArray has the same length as 'self'.

        method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None
            Method to use for filling holes in reindexed Series
            pad / ffill: propagate last valid observation forward to next valid
            backfill / bfill: use NEXT valid observation to fill gap

        limit : int, default None
            The maximum number of entries where NA values will be filled.

        copy : bool, default True
            Whether to make a copy of the data before filling. If False, then
            the original should be modified and no new memory should be allocated.

        Returns
        -------
        GeometryArray
        """
        if method is not None:
            raise NotImplementedError("fillna with a method is not yet supported")

        mask = self.isna()
        if copy:
            new_values = self.copy()
        else:
            new_values = self

        if not mask.any():
            return new_values

        if limit is not None and limit < len(self):
            modify = mask.cumsum() > limit
            if modify.any():
                mask[modify] = False

        if isna(value):
            value = [None]
        elif _is_scalar_geometry(value):
            value = [value]
        elif isinstance(value, GeometryArray):
            value = value[mask]
        else:
            raise TypeError(
                "'value' parameter must be None, a scalar geometry, or a GeoSeries, "
                f"but you passed a {type(value).__name__!r}"
            )
        value_arr = np.asarray(value, dtype=object)

        new_values._data[mask] = value_arr
        return new_values

    def astype(self, dtype, copy: bool = True) -> np.ndarray:
        """
        Cast to a NumPy array with 'dtype'.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        copy : bool, default True
            Whether to copy the data, even if not necessary. If False,
            a copy is made only if the old dtype does not match the
            new dtype.

        Returns
        -------
        array : ndarray
            NumPy ndarray with 'dtype' for its dtype.
        """
        if isinstance(dtype, GeometryDtype):
            if copy:
                return self.copy()
            else:
                return self
        elif pd.api.types.is_string_dtype(dtype) and not pd.api.types.is_object_dtype(
            dtype
        ):
            string_values = to_wkt(self)
            pd_dtype = pd.api.types.pandas_dtype(dtype)
            if isinstance(pd_dtype, pd.StringDtype):
                # ensure to return a pandas string array instead of numpy array
                return pd.array(string_values, dtype=pd_dtype)
            return string_values.astype(dtype, copy=False)
        else:
            # numpy 2.0 makes copy=False case strict (errors if cannot avoid the copy)
            # -> in that case use `np.asarray` as backwards compatible alternative
            # for `copy=None` (when requiring numpy 2+, this can be cleaned up)
            if not copy:
                return np.asarray(self, dtype=dtype)
            else:
                return np.array(self, dtype=dtype, copy=copy)

    def isna(self) -> np.ndarray:
        """Boolean NumPy array indicating if each value is missing."""
        return shapely.is_missing(self._data)

    def value_counts(
        self,
        dropna: bool = True,
    ) -> pd.Series:
        """
        Compute a histogram of the counts of non-null values.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of NaN

        Returns
        -------
        pd.Series
        """
        # note ExtensionArray usage of value_counts only specifies dropna,
        # so sort, normalize and bins are not arguments
        values = to_wkb(self)
        from pandas import Index, Series

        result = Series(values).value_counts(dropna=dropna)
        # value_counts converts None to nan, need to convert back for from_wkb to work
        # note result.index already has object dtype, not geometry
        # Can't use fillna(None) or Index.putmask, as this gets converted back to nan
        # for object dtypes
        result.index = Index(
            from_wkb(np.where(result.index.isna(), None, result.index), crs=self.crs)
        )
        return result

    def unique(self) -> ExtensionArray:
        """Compute the ExtensionArray of unique values.

        Returns
        -------
        uniques : ExtensionArray
        """
        from pandas import factorize

        _, uniques = factorize(self)
        return uniques

    @property
    def nbytes(self):
        return self._data.nbytes

    def shift(self, periods: int = 1, fill_value: Any | None = None) -> GeometryArray:
        """
        Shift values by desired number.

        Newly introduced missing values are filled with
        ``self.dtype.na_value``.

        Parameters
        ----------
        periods : int, default 1
            The number of periods to shift. Negative values are allowed
            for shifting backwards.

        fill_value : object, optional (default None)
            The scalar value to use for newly introduced missing values.
            The default is ``self.dtype.na_value``.

        Returns
        -------
        GeometryArray
            Shifted.

        Notes
        -----
        If ``self`` is empty or ``periods`` is 0, a copy of ``self`` is
        returned.

        If ``periods > len(self)``, then an array of size
        len(self) is returned, with all values filled with
        ``self.dtype.na_value``.
        """
        shifted = super().shift(periods, fill_value)
        shifted.crs = self.crs
        return shifted

    # -------------------------------------------------------------------------
    # ExtensionArray specific
    # -------------------------------------------------------------------------

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy: bool = False) -> ExtensionArray:
        """
        Construct a new ExtensionArray from a sequence of scalars.

        Parameters
        ----------
        scalars : Sequence
            Each element will be an instance of the scalar type for this
            array, ``cls.dtype.type``.
        dtype : dtype, optional
            Construct for this particular dtype. This should be a Dtype
            compatible with the ExtensionArray.
        copy : boolean, default False
            If True, copy the underlying data.

        Returns
        -------
        ExtensionArray
        """
        # GH 1413
        if isinstance(scalars, BaseGeometry):
            scalars = [scalars]
        return from_shapely(scalars)

    @classmethod
    def _from_sequence_of_strings(cls, strings, *, dtype=None, copy=False):
        """
        Construct a new ExtensionArray from a sequence of strings.

        Parameters
        ----------
        strings : Sequence
            Each element will be an instance of the scalar type for this
            array, ``cls.dtype.type``.
        dtype : dtype, optional
            Construct for this particular dtype. This should be a Dtype
            compatible with the ExtensionArray.
        copy : bool, default False
            If True, copy the underlying data.

        Returns
        -------
        ExtensionArray
        """
        # GH 3099
        return from_wkt(strings)

    def _values_for_factorize(self) -> tuple[np.ndarray, Any]:
        """Return an array and missing value suitable for factorization.

        Returns
        -------
        values : ndarray
            An array suitable for factorization. This should maintain order
            and be a supported dtype (Float64, Int64, UInt64, String, Object).
            By default, the extension array is cast to object dtype.
        na_value : object
            The value in `values` to consider missing. This will be treated
            as NA in the factorization routines, so it will be coded as
            `na_sentinal` and not included in `uniques`. By default,
            ``np.nan`` is used.
        """
        vals = to_wkb(self)
        return vals, None

    @classmethod
    def _from_factorized(cls, values: np.ndarray, original: ExtensionArray):
        """
        Reconstruct an ExtensionArray after factorization.

        Parameters
        ----------
        values : ndarray
            An integer ndarray with the factorized values.
        original : ExtensionArray
            The original ExtensionArray that factorize was called on.

        See Also
        --------
        pandas.factorize
        ExtensionArray.factorize
        """
        return from_wkb(values, crs=original.crs)

    def _values_for_argsort(self) -> np.ndarray:
        """Return values for sorting.

        Returns
        -------
        ndarray
            The transformed values should maintain the ordering between values
            within the array.

        See Also
        --------
        ExtensionArray.argsort
        """
        # Note: this is used in `ExtensionArray.argsort`.
        from vibespatial.api.tools.hilbert_curve import _hilbert_distance

        if self.size == 0:
            # TODO _hilbert_distance fails for empty array
            return np.array([], dtype="uint32")

        mask_empty = self.is_empty
        has_empty = mask_empty.any()
        mask = self.isna() | mask_empty
        if mask.any():
            # if there are missing or empty geometries, we fill those with
            # a dummy geometry so that the _hilbert_distance function can
            # process those. The missing values are handled separately by
            # pandas regardless of the values we return here (to sort
            # first/last depending on 'na_position'), the distances for the
            # empty geometries are replaced below with an appropriate value
            geoms = self.copy()
            indices = np.nonzero(~mask)[0]
            if indices.size:
                geom = self[indices[0]]
            else:
                # for all-empty/NA, just take random geometry
                geom = shapely.geometry.Point(0, 0)

            geoms[mask] = geom
        else:
            geoms = self
        if has_empty:
            # in case we have empty geometries, we need to expand the total
            # bounds with a small percentage, so the empties can be
            # deterministically sorted first
            total_bounds = geoms.total_bounds
            xoff = (total_bounds[2] - total_bounds[0]) * 0.01
            yoff = (total_bounds[3] - total_bounds[1]) * 0.01
            total_bounds += np.array([-xoff, -yoff, xoff, yoff])
        else:
            total_bounds = None
        distances = _hilbert_distance(geoms, total_bounds=total_bounds)
        if has_empty:
            # empty geometries are sorted first ("smallest"), so fill in
            # smallest possible value for uints
            distances[mask_empty] = 0
        return distances

    def argmin(self, skipna: bool = True) -> int:
        raise TypeError("geometries have no minimum or maximum")

    def argmax(self, skipna: bool = True) -> int:
        raise TypeError("geometries have no minimum or maximum")

    def _formatter(self, boxed: bool = False):
        """Return a formatting function for scalar values.

        This is used in the default '__repr__'. The returned formatting
        function receives instances of your scalar type.

        Parameters
        ----------
        boxed: bool, default False
            An indicated for whether or not your array is being printed
            within a Series, DataFrame, or Index (True), or just by
            itself (False). This may be useful if you want scalar values
            to appear differently within a Series versus on its own (e.g.
            quoted or not).

        Returns
        -------
        Callable[[Any], str]
            A callable that gets instances of the scalar type and
            returns a string. By default, :func:`repr` is used
            when ``boxed=False`` and :func:`str` is used when
            ``boxed=True``.
        """
        if boxed:
            import vibespatial.api as geopandas

            precision = geopandas.options.display_precision
            if precision is None:
                if self.crs:
                    if self.crs.is_projected:
                        precision = 3
                    else:
                        precision = 5
                else:
                    # fallback
                    # dummy heuristic based on 10 first geometries that should
                    # work in most cases
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        xmin, ymin, xmax, ymax = self[~self.isna()][:10].total_bounds
                    if (
                        (-180 <= xmin <= 180)
                        and (-180 <= xmax <= 180)
                        and (-90 <= ymin <= 90)
                        and (-90 <= ymax <= 90)
                    ):
                        # geographic coordinates
                        precision = 5
                    else:
                        # typically projected coordinates
                        # (in case of unit meter: mm precision)
                        precision = 3
            return lambda geom: shapely.to_wkt(geom, rounding_precision=precision)
        return repr

    @classmethod
    def _concat_same_type(cls, to_concat):
        """Concatenate multiple array.

        Parameters
        ----------
        to_concat : sequence of this type

        Returns
        -------
        ExtensionArray
        """
        # Owned-path: if ALL arrays have owned backing, concatenate at the
        # buffer level without materializing Shapely objects.  This preserves
        # device-resident geometry through pd.concat on GeoDataFrames.
        if all(ga._owned is not None for ga in to_concat):
            owned_arrays = [ga._owned for ga in to_concat]
            result_owned = OwnedGeometryArray.concat(owned_arrays)
            return GeometryArray.from_owned(result_owned, crs=_get_common_crs(to_concat))

        data = np.concatenate([ga._data for ga in to_concat])
        return GeometryArray(data, crs=_get_common_crs(to_concat))

    def _reduce(self, name: str, skipna: bool = True, keepdims: bool = False, **kwargs):
        # including the base class version here (that raises by default)
        # because this was not yet defined in pandas 0.23
        if name in ("any", "all"):
            return getattr(self._data, name)(keepdims=keepdims)
        raise TypeError(
            f"'{type(self).__name__}' with dtype {self.dtype} "
            f"does not support reduction '{name}'"
        )

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Return the data as a numpy array.

        This is the numpy array interface.

        Returns
        -------
        values : numpy array
        """
        if copy and (dtype is None or dtype == np.dtype("object")):
            return self._data.copy()
        return self._data

    def _binop(self, other, op):
        def convert_values(param):
            if not _is_scalar_geometry(param) and (
                isinstance(param, ExtensionArray) or pd.api.types.is_list_like(param)
            ):
                ovalues = param
            else:  # Assume its an object
                ovalues = [param] * len(self)
            return ovalues

        if isinstance(other, pd.Series | pd.Index | pd.DataFrame):
            # rely on pandas to unbox and dispatch to us
            return NotImplemented

        lvalues = self
        rvalues = convert_values(other)

        if len(lvalues) != len(rvalues):
            raise ValueError("Lengths must match to compare")

        # If the operator is not defined for the underlying objects,
        # a TypeError should be raised
        res = [op(a, b) for (a, b) in zip(lvalues, rvalues)]

        res = np.asarray(res, dtype=bool)
        return res

    def __eq__(self, other):
        return self._binop(other, operator.eq)

    # https://github.com/python/typeshed/issues/2148#issuecomment-520783318
    # Incompatible types in assignment (expression has type "None", base class
    # "object" defined the type as "Callable[[object], int]")
    # (Explicitly mirrored from pandas to declare non hashable)
    __hash__: ClassVar[None]  # type: ignore[assignment]

    def __ne__(self, other):
        return self._binop(other, operator.ne)

    def __contains__(self, item) -> bool:
        """Return for `item in self`."""
        if isna(item):
            if (
                item is self.dtype.na_value
                or isinstance(item, self.dtype.type)
                or item is None
            ):
                return self.isna().any()
            else:
                return False
        return (self == item).any()


def _get_common_crs(arr_seq) -> CRS:
    # mask out all None arrays with no crs (most likely auto generated by pandas
    # from concat with missing column)
    arr_seq = [ga for ga in arr_seq if not (ga.isna().all() and ga.crs is None)]
    # determine unique crs without using a set, because CRS hash can be different
    # for objects with the same CRS
    unique_crs = []
    for arr in arr_seq:
        if arr.crs not in unique_crs:
            unique_crs.append(arr.crs)

    crs_not_none = [crs for crs in unique_crs if crs is not None]
    names = [crs.name for crs in crs_not_none]

    if len(crs_not_none) == 0:
        return None
    if len(crs_not_none) == 1:
        if len(unique_crs) != 1:
            warnings.warn(
                "CRS not set for some of the concatenation inputs. "
                f"Setting output's CRS as {names[0]} "
                "(the single non-null crs provided).",
                stacklevel=2,
            )
        return crs_not_none[0]

    raise ValueError(
        f"Cannot determine common CRS for concatenation inputs, got {names}. "
        "Use `to_crs()` to transform geometries to the same CRS before merging."
    )


def transform(data, func) -> np.ndarray:
    has_z = shapely.has_z(data)

    result = np.empty_like(data)

    coords = shapely.get_coordinates(data[~has_z], include_z=False)
    new_coords_z = func(coords[:, 0], coords[:, 1])
    result[~has_z] = shapely.set_coordinates(
        data[~has_z].copy(), np.array(new_coords_z).T
    )

    coords_z = shapely.get_coordinates(data[has_z], include_z=True)
    new_coords_z = func(coords_z[:, 0], coords_z[:, 1], coords_z[:, 2])
    result[has_z] = shapely.set_coordinates(
        data[has_z].copy(), np.array(new_coords_z).T
    )

    return result
