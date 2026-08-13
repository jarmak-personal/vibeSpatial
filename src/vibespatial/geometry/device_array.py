"""DeviceGeometryArray -- pandas ExtensionArray backed by device-resident OwnedGeometryArray.

This module inverts the storage model: OwnedGeometryArray is the source of truth
(device-resident when GPU is available), and Shapely objects are materialized lazily
on demand.  The goal is to eliminate D->H->Shapely->H->D roundtrips when GPU-decoded
geometry flows through GeoDataFrame into GPU consumers.

See epic vibeSpatial-o17.9.13 and ADR-0005 for design rationale.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray, ExtensionDtype, register_extension_dtype
from shapely.geometry.base import BaseGeometry

from vibespatial.cuda._runtime import get_cuda_runtime
from vibespatial.runtime.config import LINESTRING_TWO_POINT_BUFFER_GPU_THRESHOLD
from vibespatial.runtime.fallbacks import record_fallback_event
from vibespatial.runtime.residency import Residency, TransferTrigger

from .api_registry import (
    apply_host_transform,
    build_device_spatial_index,
    build_host_transform_func,
)
from .buffers import GeometryFamily
from .host_bridge import materialize_family_row, owned_to_shapely
from .owned import (
    FAMILY_TAGS,
    NULL_TAG,
    TAG_FAMILIES,
    DeviceFamilyGeometryBuffer,
    DiagnosticEvent,
    DiagnosticKind,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
    OwnedGeometryDeviceState,
    build_null_owned_array,
    concat_owned_scatter,
    from_shapely_geometries,
    get_geometry_buffer_schema,
)

# Map family tags → shapely geometry type names (matches GeoPandas array.py _names)
_TAG_TO_GEOM_TYPE_NAME: dict[int, str] = {
    FAMILY_TAGS[GeometryFamily.POINT]: "Point",
    FAMILY_TAGS[GeometryFamily.LINESTRING]: "LineString",
    FAMILY_TAGS[GeometryFamily.POLYGON]: "Polygon",
    FAMILY_TAGS[GeometryFamily.MULTIPOINT]: "MultiPoint",
    FAMILY_TAGS[GeometryFamily.MULTILINESTRING]: "MultiLineString",
    FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]: "MultiPolygon",
}


def _owned_requires_host_transfer(owned: OwnedGeometryArray | None) -> bool:
    if owned is None:
        return False
    if owned.residency is Residency.DEVICE:
        return True
    return any(not buffer.host_materialized for buffer in owned.families.values())


def _single_family_geom_type_if_no_nulls(owned: OwnedGeometryArray) -> str | None:
    families = getattr(owned, "families", {})
    if len(families) != 1:
        return None
    family, host_buffer = next(iter(families.items()))
    row_count = getattr(host_buffer, "row_count", None)
    device_state = getattr(owned, "device_state", None)
    if device_state is not None:
        device_buffer = device_state.families.get(family)
        if device_buffer is not None:
            offsets = getattr(device_buffer, "geometry_offsets", None)
            if offsets is not None:
                row_count = int(offsets.size) - 1
    if int(row_count or 0) != int(owned.row_count):
        return None
    return _TAG_TO_GEOM_TYPE_NAME.get(FAMILY_TAGS[family])


def _device_bool_scalar(value: Any, *, reason: str) -> bool:
    """Read one device boolean through the runtime so profiles see the fence."""
    if hasattr(value, "__cuda_array_interface__") or type(value).__module__.startswith("cupy"):
        import cupy as cp

        host = get_cuda_runtime().copy_device_to_host(
            cp.asarray(value).reshape(1),
            reason=reason,
        )
        return bool(np.asarray(host, dtype=bool).reshape(-1)[0])
    return bool(value)


def _device_row_metadata(
    owned: OwnedGeometryArray,
    row_index: int,
    *,
    reason: str,
) -> tuple[bool, int, int] | None:
    """Return validity, family tag, and family row for one device-backed row."""
    if owned.residency is not Residency.DEVICE or owned.device_state is None:
        return None
    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover - guarded by runtime
        return None

    state = owned._ensure_device_state(preserve_indexed_view=True)
    d_metadata = cp.empty(3, dtype=cp.int64)
    d_metadata[0] = cp.asarray(state.validity)[row_index].astype(cp.int64)
    d_metadata[1] = cp.asarray(state.tags)[row_index].astype(cp.int64)
    d_metadata[2] = cp.asarray(state.family_row_offsets)[row_index].astype(cp.int64)
    host_metadata = np.asarray(
        get_cuda_runtime().copy_device_to_host(d_metadata, reason=reason),
        dtype=np.int64,
    )
    return (
        bool(host_metadata[0]),
        int(host_metadata[1]),
        int(host_metadata[2]),
    )


def _buffer_all_valid(
    owned: OwnedGeometryArray,
    families: set[GeometryFamily],
) -> bool:
    cached = getattr(owned, "_validity", None)
    if cached is not None:
        return bool(np.all(cached))
    if owned.residency is Residency.DEVICE and owned.device_state is not None:
        state = owned._ensure_device_state(preserve_indexed_view=True)
        if state.trusted_all_valid is not None:
            return bool(state.trusted_all_valid)
        try:
            import cupy as cp
        except ModuleNotFoundError:  # pragma: no cover - guarded by runtime
            cp = None
        if cp is not None:
            reason = (
                "point buffer validity admission scalar fence"
                if families == {GeometryFamily.POINT}
                else "DeviceGeometryArray.buffer all-valid scalar fence"
            )
            result = _device_bool_scalar(cp.all(cp.asarray(state.validity)), reason=reason)
            if result:
                state.trusted_all_valid = True
            return result
    return bool(np.all(owned.validity))


def _buffer_has_empties(
    owned: OwnedGeometryArray,
    family: GeometryFamily,
) -> bool:
    state = owned.device_state
    if owned.residency is Residency.DEVICE and state is not None and family in state.families:
        if state.trusted_all_non_empty is True:
            return False
        try:
            import cupy as cp
        except ModuleNotFoundError:  # pragma: no cover - guarded by runtime
            cp = None
        if cp is not None:
            reason = (
                "point buffer empty-point admission scalar fence"
                if family is GeometryFamily.POINT
                else f"DeviceGeometryArray.buffer {family.value} empty-mask scalar fence"
            )
            has_empties = _device_bool_scalar(
                cp.any(cp.asarray(state.families[family].empty_mask)),
                reason=reason,
            )
            if not has_empties:
                state.trusted_all_non_empty = True
            return has_empties
    return bool(np.any(owned.families[family].empty_mask))


def _record_shapely_fallback_event(
    *,
    surface: str,
    reason: str,
    owned: OwnedGeometryArray | None,
    detail: str = "",
    requested: Any = "auto",
    pipeline: str = "",
) -> None:
    record_fallback_event(
        surface=surface,
        requested=requested,
        selected="cpu",
        reason=reason,
        detail=detail,
        pipeline=pipeline,
        d2h_transfer=_owned_requires_host_transfer(owned),
    )


def _is_host_geometry_array_like(value: Any) -> bool:
    dtype = getattr(value, "dtype", None)
    return getattr(dtype, "name", None) == "geometry" and hasattr(value, "_data")


class DeviceGeometryDtype(ExtensionDtype):
    """pandas dtype for device-resident geometry arrays."""

    type = BaseGeometry
    name = "device_geometry"
    na_value = None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return other == "geometry"
        return isinstance(other, DeviceGeometryDtype) or getattr(other, "name", None) == "geometry"

    def __hash__(self) -> int:
        return hash("geometry")

    @classmethod
    def construct_from_string(cls, string: str) -> DeviceGeometryDtype:
        if not isinstance(string, str):
            raise TypeError(f"'construct_from_string' expects a string, got {type(string)}")
        if string == cls.name:
            return cls()
        raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'")

    @classmethod
    def construct_array_type(cls) -> type[DeviceGeometryArray]:
        return DeviceGeometryArray


register_extension_dtype(DeviceGeometryDtype)


class DeviceGeometryArray(ExtensionArray):
    """pandas ExtensionArray backed by a device-resident OwnedGeometryArray.

    Source of truth is ``_owned`` (an OwnedGeometryArray).  Shapely objects
    are materialized lazily into ``_shapely_cache`` only when pandas or user
    code requires individual geometry objects.

    Key properties:
    - ``take`` operates on owned buffers without Shapely round-trip.
    - ``copy`` duplicates owned buffers on the current residency side.
    - ``_concat_same_type`` merges owned buffers.
    - Diagnostic events are emitted for every materialization (per ADR-0005).
    """

    _dtype = DeviceGeometryDtype()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        owned: OwnedGeometryArray | None,
        *,
        crs: Any | None = None,
        composition: Any | None = None,
    ) -> None:
        if (owned is None) == (composition is None):
            raise ValueError("exactly one device geometry carrier is required")
        self.__owned = owned
        self._composition = composition
        self._crs = crs
        self._shapely_cache: np.ndarray | None = None
        self._owned_flat_sindex_cache = None
        self._sindex_cache = None
        self._provenance = None

    @classmethod
    def _from_owned(
        cls,
        owned: OwnedGeometryArray,
        *,
        crs: Any | None = None,
        provenance=None,
    ) -> DeviceGeometryArray:
        """Construct directly from an OwnedGeometryArray (zero-copy)."""
        result = cls(owned, crs=crs)
        result._provenance = provenance
        return result

    @classmethod
    def _from_composition(
        cls,
        composition: Any,
        *,
        crs: Any | None = None,
        provenance=None,
    ) -> DeviceGeometryArray:
        """Wrap trusted singular native partitions without physicalizing them."""
        if not bool(getattr(composition, "trusted_singular_rows", False)):
            raise ValueError("partitioned public geometry requires singular-row proof")
        result = cls(None, crs=crs, composition=composition)
        result._provenance = provenance
        return result

    @classmethod
    def _from_sequence(
        cls,
        scalars: Sequence[BaseGeometry | None],
        *,
        dtype: ExtensionDtype | None = None,
        copy: bool = False,
    ) -> DeviceGeometryArray:
        if isinstance(scalars, BaseGeometry):
            scalars = [scalars]
        scalars_list = list(scalars)
        owned = from_shapely_geometries(scalars_list)
        return cls._from_owned(owned)

    @classmethod
    def _from_factorized(
        cls, values: np.ndarray, original: DeviceGeometryArray
    ) -> DeviceGeometryArray:
        return cls._from_sequence(values)

    # ------------------------------------------------------------------
    # ExtensionArray protocol
    # ------------------------------------------------------------------

    @property
    def dtype(self) -> DeviceGeometryDtype:
        return self._dtype

    @property
    def owned(self) -> OwnedGeometryArray:
        """Cross the explicit contiguous-owned boundary when required."""
        return self.to_owned()

    def cached_owned(self) -> OwnedGeometryArray | None:
        """Return a contiguous carrier only when one already exists."""
        return self.__owned

    @property
    def native_composition(self):
        """Return the partitioned native carrier without physicalizing it."""
        return self._composition

    @property
    def native_family_domain(self):
        """Return proven concrete families without probing contiguous storage."""
        if self._composition is None:
            return None
        return self._composition.trusted_family_domain

    @property
    def _owned(self) -> OwnedGeometryArray:
        owned = self.__owned
        if owned is None:
            raise AttributeError(
                "partitioned device geometry has no contiguous owned carrier; "
                "inspect native_composition or call physicalize_owned() explicitly"
            )
        return owned

    @_owned.setter
    def _owned(self, owned: OwnedGeometryArray) -> None:
        self.__owned = owned
        self._composition = None

    def physicalize_owned(self) -> OwnedGeometryArray:
        """Explicitly join a trusted singular composition into owned storage."""
        owned = self.__owned
        if owned is not None:
            return owned
        composition = self._composition
        owned = composition._singular_owned_device()
        if owned is None:
            raise RuntimeError("singular geometry partitions could not be physicalized")
        self.__owned = owned
        self._composition = None
        return owned

    @property
    def crs(self) -> Any | None:
        return self._crs

    @crs.setter
    def crs(self, value: Any | None) -> None:
        if value is not None:
            try:
                from pyproj import CRS

                self._crs = CRS.from_user_input(value)
            except ImportError:
                self._crs = None
        else:
            self._crs = None

    def __len__(self) -> int:
        if self._composition is not None:
            return int(self._composition.row_count)
        return self.__owned.row_count

    @property
    def nbytes(self) -> int:
        if self._composition is not None:
            return sum(
                int(
                    DeviceGeometryArray._from_owned(
                        part.geometry.owned,
                        crs=self._crs,
                    ).nbytes
                )
                for part in self._composition.parts
                if part.geometry.owned is not None
            )
        owned = self.to_owned()
        if owned.residency is Residency.DEVICE and (
            owned._validity is None
            or owned._tags is None
            or owned._family_row_offsets is None
        ):
            state = owned._ensure_device_state(preserve_indexed_view=True)
            total = state.validity.nbytes + state.tags.nbytes
            total += state.family_row_offsets.nbytes
            for buffer in state.families.values():
                total += buffer.x.nbytes + buffer.y.nbytes
                total += buffer.geometry_offsets.nbytes
                total += buffer.empty_mask.nbytes
                if buffer.part_offsets is not None:
                    total += buffer.part_offsets.nbytes
                if buffer.ring_offsets is not None:
                    total += buffer.ring_offsets.nbytes
                if buffer.bounds is not None:
                    total += buffer.bounds.nbytes
            return total

        total = 0
        total += self.to_owned().validity.nbytes
        total += self.to_owned().tags.nbytes
        total += self.to_owned().family_row_offsets.nbytes
        for buffer in self.to_owned().families.values():
            total += buffer.x.nbytes + buffer.y.nbytes
            total += buffer.geometry_offsets.nbytes
            total += buffer.empty_mask.nbytes
            if buffer.part_offsets is not None:
                total += buffer.part_offsets.nbytes
            if buffer.ring_offsets is not None:
                total += buffer.ring_offsets.nbytes
            if buffer.bounds is not None:
                total += buffer.bounds.nbytes
        return total

    def isna(self) -> np.ndarray:
        if self._composition is not None:
            result = np.ones(len(self), dtype=bool)
            for part in self._composition.parts:
                rows = self._composition_part_rows_host(part)
                missing = DeviceGeometryArray._from_owned(
                    part.geometry.owned,
                    crs=self._crs,
                ).isna()
                result[rows[~missing]] = False
            return result
        if getattr(self.to_owned(), "_validity", None) is not None:
            return ~np.asarray(self.to_owned()._validity, dtype=bool)
        state = getattr(self.to_owned(), "device_state", None)
        if state is not None:
            if state.trusted_all_valid is True:
                return np.zeros(int(self.to_owned().row_count), dtype=bool)
            device_state = self.to_owned()._ensure_device_state(preserve_indexed_view=True)
            host_validity = get_cuda_runtime().copy_device_to_host(
                device_state.validity,
                reason="DeviceGeometryArray isna validity terminal export",
                terminal_export=True,
            )
            return ~np.asarray(host_validity, dtype=bool)
        return ~self.to_owned().validity

    @property
    def _data(self) -> np.ndarray:
        """Compatibility shim: GeoPandas internals access `array._data` for
        the numpy array of Shapely objects.  Triggers lazy materialization."""
        return self._ensure_shapely_cache()

    # ------------------------------------------------------------------
    # Device-side properties (NO Shapely materialization)
    # ------------------------------------------------------------------

    @property
    def geom_type(self) -> np.ndarray:
        """Geometry type names from owned tags — no Shapely materialization.

        Uses vectorized numpy indexing instead of per-element Python loop
        (VPAT001 compliance).
        """
        if self._composition is not None:
            result = np.empty(len(self), dtype=object)
            result[:] = None
            for part in self._composition.parts:
                rows = self._composition_part_rows_host(part)
                part_array = DeviceGeometryArray._from_owned(
                    part.geometry.owned,
                    crs=self._crs,
                )
                missing = part_array.isna()
                values = part_array.geom_type
                result[rows[~missing]] = values[~missing]
            return result
        single_type = _single_family_geom_type_if_no_nulls(self.to_owned())
        if single_type is not None:
            return np.full(len(self), single_type, dtype=object)
        if self.to_owned()._tags is None and self.to_owned().device_state is not None:
            self.to_owned()._tags = get_cuda_runtime().copy_device_to_host(
                self.to_owned().device_state.tags,
                reason="device geometry array geom-type tag host export",
            )
        tags = self.to_owned().tags
        result = np.empty(len(tags), dtype=object)
        for tag_value, name in _TAG_TO_GEOM_TYPE_NAME.items():
            result[tags == tag_value] = name
        result[tags == NULL_TAG] = None
        return result

    @property
    def is_empty(self) -> np.ndarray:
        """Per-geometry emptiness from owned empty_mask — no Shapely materialization.

        Uses vectorized numpy indexing per family instead of per-element
        Python loop (VPAT001 compliance).
        """
        if self._composition is not None:
            result = np.zeros(len(self), dtype=bool)
            for part in self._composition.parts:
                rows = self._composition_part_rows_host(part)
                part_array = DeviceGeometryArray._from_owned(
                    part.geometry.owned,
                    crs=self._crs,
                )
                missing = part_array.isna()
                values = part_array.is_empty
                result[rows[~missing]] = values[~missing]
            return result
        tags = self.to_owned().tags
        offsets = self.to_owned().family_row_offsets
        result = np.zeros(len(tags), dtype=bool)
        for tag_value, family in TAG_FAMILIES.items():
            if family not in self.to_owned().families:
                continue
            buf = self.to_owned().families[family]
            mask = tags == tag_value
            if not mask.any():
                continue
            family_rows = offsets[mask]
            valid = (family_rows >= 0) & (family_rows < len(buf.empty_mask))
            # Only index into empty_mask for valid family rows
            family_result = np.zeros(mask.sum(), dtype=bool)
            family_result[valid] = buf.empty_mask[family_rows[valid]]
            result[mask] = family_result
        return result

    @property
    def bounds(self) -> np.ndarray:
        """Per-geometry bounds from owned coordinate buffers — no Shapely materialization.

        Returns (N, 4) float64 array of [minx, miny, maxx, maxy].
        """
        if self._composition is not None:
            result = np.full((len(self), 4), np.nan, dtype=np.float64)
            for part in self._composition.parts:
                rows = self._composition_part_rows_host(part)
                part_array = DeviceGeometryArray._from_owned(
                    part.geometry.owned,
                    crs=self._crs,
                )
                missing = part_array.isna()
                values = _compute_bounds_from_owned(part.geometry.owned)
                result[rows[~missing]] = values[~missing]
            return result
        return _compute_bounds_from_owned(self.to_owned())

    @property
    def total_bounds(self) -> np.ndarray:
        """Aggregate bounds — no Shapely materialization."""
        if len(self) == 0:
            return np.array([np.nan, np.nan, np.nan, np.nan])
        if self._composition is not None:
            part_bounds = np.asarray(
                [
                    _compute_total_bounds_from_owned(part.geometry.owned)
                    for part in self._composition.parts
                    if part.geometry.owned is not None and part.geometry.row_count > 0
                ],
                dtype=np.float64,
            )
            if part_bounds.size == 0:
                return np.array([np.nan, np.nan, np.nan, np.nan])
            return np.array(
                [
                    np.nanmin(part_bounds[:, 0]),
                    np.nanmin(part_bounds[:, 1]),
                    np.nanmax(part_bounds[:, 2]),
                    np.nanmax(part_bounds[:, 3]),
                ]
            )
        return _compute_total_bounds_from_owned(self.to_owned())

    @property
    def x(self) -> np.ndarray:
        """Return point x coordinates without materializing Shapely objects."""
        from vibespatial.constructive.properties import get_x_owned

        non_null = self.to_owned().validity
        point_tag = FAMILY_TAGS[GeometryFamily.POINT]
        if np.all(self.to_owned().tags[non_null] == point_tag):
            return get_x_owned(self.to_owned())
        raise ValueError("x attribute access only provided for Point geometries")

    @property
    def y(self) -> np.ndarray:
        """Return point y coordinates without materializing Shapely objects."""
        from vibespatial.constructive.properties import get_y_owned

        non_null = self.to_owned().validity
        point_tag = FAMILY_TAGS[GeometryFamily.POINT]
        if np.all(self.to_owned().tags[non_null] == point_tag):
            return get_y_owned(self.to_owned())
        raise ValueError("y attribute access only provided for Point geometries")

    # ------------------------------------------------------------------
    # CRS / projection
    # ------------------------------------------------------------------

    def estimate_utm_crs(self, datum_name: str = "WGS 84"):
        """Estimate UTM CRS from bounds -- no Shapely materialization."""
        from pyproj import CRS
        from pyproj.aoi import AreaOfInterest
        from pyproj.database import query_utm_crs_info

        if not self._crs:
            raise RuntimeError("crs must be set to estimate UTM CRS.")

        minx, miny, maxx, maxy = self.total_bounds
        if self._crs.is_geographic:
            x_center = (minx + maxx) / 2
            y_center = (miny + maxy) / 2
        else:
            from vibeproj import Transformer as _VibeTransformer

            t = _VibeTransformer.from_crs(self._crs, "EPSG:4326", always_xy=True)
            minx, miny, maxx, maxy = t.transform_bounds(minx, miny, maxx, maxy)
            y_center = (miny + maxy) / 2
            if minx > maxx:
                maxx += 360
                x_center = ((minx + maxx) / 2 + 180) % 360 - 180
            else:
                x_center = (minx + maxx) / 2

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

    def to_crs(self, crs=None, epsg=None):
        """Reproject via vibeProj transform_buffers -- stays on device."""
        from pyproj import CRS as PyprojCRS

        from vibespatial.runtime import ExecutionMode, get_requested_mode

        if self._crs is None:
            raise ValueError(
                "Cannot transform naive geometries.  Please set a crs on the object first."
            )
        if crs is not None:
            crs = PyprojCRS.from_user_input(crs)
        elif epsg is not None:
            crs = PyprojCRS.from_epsg(epsg)
        else:
            raise ValueError("Must pass either crs or epsg.")

        if self._crs.is_exact_same(crs):
            return self

        if get_requested_mode() is ExecutionMode.CPU:
            # CPU fallback: materialize Shapely, transform on host, rebuild.
            shapely_geoms = self._ensure_shapely_cache()
            transform_func = build_host_transform_func(self._crs, crs)
            new_data = apply_host_transform(shapely_geoms, transform_func)
            new_owned = from_shapely_geometries(new_data)
            return DeviceGeometryArray._from_owned(new_owned, crs=crs)

        return _to_crs_owned(self.to_owned(), self._crs, crs)

    # ------------------------------------------------------------------
    # Shapely-delegated properties (materialization with diagnostics)
    # ------------------------------------------------------------------

    def check_geographic_crs(self, stacklevel: int, *, operation: str | None = None) -> None:
        """Warn if CRS is geographic."""
        if self._crs and self._crs.is_geographic:
            op_name = operation or "this operation"
            warnings.warn(
                "Geometry is in a geographic CRS. Results from "
                f"'{op_name}' are likely incorrect. "
                "Use 'GeoSeries.to_crs()' to re-project geometries to a "
                "projected CRS before this operation.\n",
                UserWarning,
                stacklevel=stacklevel,
            )

    @property
    def area(self) -> np.ndarray:
        """Area — GPU-accelerated from owned coordinate buffers, no Shapely."""
        from vibespatial.constructive.measurement import area_owned

        self.check_geographic_crs(stacklevel=5, operation="area")
        return area_owned(self.to_owned())

    @property
    def length(self) -> np.ndarray:
        """Length — GPU-accelerated from owned coordinate buffers, no Shapely."""
        from vibespatial.constructive.measurement import length_owned

        self.check_geographic_crs(stacklevel=5, operation="length")
        return length_owned(self.to_owned())

    @property
    def is_valid(self) -> np.ndarray:
        """OGC validity — GPU-accelerated from owned coordinate buffers, no Shapely.

        is_valid_owned() covers full OGC validity: ring closure, min coords,
        ring self-intersection, hole-in-shell containment, ring-ring crossing,
        collinear overlap, and interior connectedness (multi-touch detection).
        Zero-copy: reads device buffers directly, returns boolean mask to host.
        """
        from vibespatial.constructive.validity import is_valid_owned

        result = np.asarray(
            is_valid_owned(self.to_owned(), _exact_collinearity=True),
            dtype=bool,
        )
        if not bool(np.all(self.to_owned().validity)):
            # GeoPandas-facing is_valid treats missing rows as False even though
            # the owned structural-validity helper treats null slots as valid.
            result = result.copy()
            result[~self.to_owned().validity] = False
        return result

    @property
    def is_simple(self) -> np.ndarray:
        """Simplicity — GPU-accelerated from owned coordinate buffers, no Shapely."""
        from vibespatial.constructive.validity import is_simple_owned

        return is_simple_owned(self.to_owned())

    @property
    def is_ring(self) -> np.ndarray:
        """Ring test — GPU-accelerated from owned coordinate buffers, no Shapely."""
        from vibespatial.constructive.properties import is_ring_owned

        return is_ring_owned(self.to_owned())

    @property
    def is_closed(self) -> np.ndarray:
        import shapely

        _record_shapely_fallback_event(
            surface="DeviceGeometryArray.is_closed",
            reason="Shapely materialization required",
            owned=self.to_owned(),
            detail=f"rows={len(self)}",
            pipeline="materialization",
        )
        self.to_owned()._record(
            DiagnosticKind.MATERIALIZATION,
            "DeviceGeometryArray.is_closed: Shapely materialization required",
            visible=True,
        )
        return shapely.is_closed(self._ensure_shapely_cache())

    @property
    def has_z(self) -> np.ndarray:
        """has_z from owned metadata — OwnedGeometryArray is 2D (x/y only)."""
        return np.zeros(len(self), dtype=bool)

    @property
    def has_m(self) -> np.ndarray:
        """has_m from owned metadata — OwnedGeometryArray is 2D (x/y only)."""
        return np.zeros(len(self), dtype=bool)

    def is_valid_reason(self) -> np.ndarray:
        import shapely

        _record_shapely_fallback_event(
            surface="DeviceGeometryArray.is_valid_reason",
            reason="Shapely materialization required",
            owned=self.to_owned(),
            detail=f"rows={len(self)}",
            pipeline="materialization",
        )
        self.to_owned()._record(
            DiagnosticKind.MATERIALIZATION,
            "DeviceGeometryArray.is_valid_reason: Shapely materialization required",
            visible=True,
        )
        return shapely.is_valid_reason(self._ensure_shapely_cache())

    # Unary geometry operations that return new GeometryArrays
    @property
    def boundary(self):
        """Boundary — GPU-accelerated via owned path, no Shapely."""
        from vibespatial.constructive.boundary import boundary_owned

        result_owned = boundary_owned(self.to_owned())
        return DeviceGeometryArray._from_owned(result_owned, crs=self._crs)

    @property
    def centroid(self):
        from vibespatial.constructive.centroid import centroid_owned

        self.check_geographic_crs(stacklevel=5, operation="centroid")
        result_owned = centroid_owned(self.to_owned())
        return DeviceGeometryArray._from_owned(result_owned, crs=self._crs)

    @property
    def convex_hull(self):
        """Convex hull — GPU-accelerated via owned path, no Shapely."""
        from vibespatial.constructive.convex_hull import convex_hull_owned

        result_owned = convex_hull_owned(self.to_owned())
        return DeviceGeometryArray._from_owned(result_owned, crs=self._crs)

    @property
    def envelope(self):
        from vibespatial.constructive.envelope import envelope_owned

        result_owned = envelope_owned(self.to_owned())
        return DeviceGeometryArray._from_owned(result_owned, crs=self._crs)

    @property
    def exterior(self):
        from vibespatial.constructive.exterior import exterior_owned

        result_owned = exterior_owned(self.to_owned())
        return DeviceGeometryArray._from_owned(result_owned, crs=self._crs)

    @property
    def unary_union(self):
        import shapely

        _record_shapely_fallback_event(
            surface="DeviceGeometryArray.unary_union",
            reason="Shapely materialization required",
            owned=self.to_owned(),
            detail=f"rows={len(self)}",
            pipeline="constructive/unary_union",
        )
        self.to_owned()._record(
            DiagnosticKind.MATERIALIZATION,
            "DeviceGeometryArray.unary_union: Shapely materialization required",
            visible=True,
        )
        return shapely.union_all(self._ensure_shapely_cache())

    def union_all(self, method="unary", grid_size=None):
        import shapely

        from vibespatial.constructive.union_all import (
            coverage_union_all_gpu_owned,
            disjoint_subset_union_all_owned,
            union_all_gpu_owned,
        )

        if method != "unary" and grid_size is not None:
            raise ValueError(f"grid_size is not supported for method '{method}'.")

        if method == "unary":
            result_owned = union_all_gpu_owned(self.to_owned(), grid_size=grid_size)
            result_geoms = result_owned.to_shapely()
            return result_geoms[0] if len(result_geoms) else shapely.GeometryCollection()

        if method == "coverage":
            result_owned = coverage_union_all_gpu_owned(self.to_owned())
            result_geoms = result_owned.to_shapely()
            return result_geoms[0] if len(result_geoms) else shapely.GeometryCollection()

        if method == "disjoint_subset":
            result_owned = disjoint_subset_union_all_owned(self.to_owned())
            if result_owned is not None:
                result_geoms = result_owned.to_shapely()
                return result_geoms[0] if len(result_geoms) else shapely.GeometryCollection()
            _record_shapely_fallback_event(
                surface="DeviceGeometryArray.union_all",
                reason="mixed families require Shapely materialization",
                owned=self.to_owned(),
                detail=f"rows={len(self)}, method={method}",
                pipeline="constructive/union_all",
            )
            self.to_owned()._record(
                DiagnosticKind.MATERIALIZATION,
                "DeviceGeometryArray.union_all: mixed-family disjoint_subset fallback",
                visible=True,
            )
            return shapely.disjoint_subset_union_all(self._ensure_shapely_cache())

        raise ValueError(
            f"Method '{method}' not recognized. Use 'coverage', 'unary' or 'disjoint_subset'."
        )

    def buffer(self, distance, quad_segs=None, **kwargs):
        """Buffer -- routes through owned dispatch with GPU kernel when possible."""
        from vibespatial.runtime import ExecutionMode, get_requested_mode
        from vibespatial.runtime.dispatch import record_dispatch_event
        from vibespatial.runtime.provenance import make_buffer_tag

        self.check_geographic_crs(stacklevel=5, operation="buffer")
        if isinstance(distance, pd.Series):
            distance = np.asarray(distance)

        # Reconcile quad_segs / resolution (deprecated alias)
        if "resolution" in kwargs:
            if quad_segs is not None:
                raise ValueError(
                    "`buffer` received both `quad_segs` and `resolution` but these are "
                    "aliases for the same parameter. Use `quad_segs` only instead."
                )
            import warnings

            warnings.warn(
                "The `resolution` argument to `buffer` is deprecated, `quad_segs` "
                "should be used instead to align with shapely.",
                category=DeprecationWarning,
                stacklevel=4,
            )
            quad_segs = kwargs.pop("resolution")
        if quad_segs is None:
            quad_segs = 16

        cap_style = kwargs.pop("cap_style", "round")
        join_style = kwargs.pop("join_style", "round")
        mitre_limit = float(kwargs.pop("mitre_limit", 5.0))
        single_sided = bool(kwargs.pop("single_sided", False))
        provenance = make_buffer_tag(
            self,
            distance,
            cap_style,
            join_style,
            single_sided,
            quad_segs,
        )

        # Route via owned metadata -- no Shapely materialization for classification
        owned = self.to_owned()
        if (
            owned.residency is Residency.DEVICE
            and owned.device_state is not None
            and owned.is_indexed_view
        ):
            # Unary stroke kernels own contiguous family-row work. Resolve the
            # logical indexed rows once on device so unused ancestral family
            # buffers cannot masquerade as logical mixed input.
            owned = owned.physicalize_device_rows(allow_capacity_allocation=True)
        families = set(owned.families.keys())
        all_valid = _buffer_all_valid(owned, families)

        if len(families) == 1 and all_valid and not single_sided:
            family = next(iter(families))
            has_empties = _buffer_has_empties(owned, family)
            if (
                not has_empties
                and owned.residency is Residency.DEVICE
                and owned.device_state is not None
            ):
                owned.device_state.trusted_homogeneous_family = family

            if (
                family is GeometryFamily.POINT
                and not has_empties
                and cap_style == "round"
                and join_style == "round"
            ):
                from vibespatial.constructive.point import point_buffer_owned_array

                result = point_buffer_owned_array(
                    owned,
                    distance,
                    quad_segs=quad_segs,
                )
                record_dispatch_event(
                    surface="DeviceGeometryArray.buffer",
                    operation="buffer",
                    implementation="point_buffer_owned_array",
                    reason="DGA direct point buffer dispatch",
                    detail=f"rows={len(self)}, family=point",
                    selected=ExecutionMode.GPU,
                )
                return DeviceGeometryArray._from_owned(
                    result,
                    crs=self._crs,
                    provenance=provenance,
                )

            elif family is GeometryFamily.LINESTRING and not has_empties:
                from vibespatial.constructive.linestring import (
                    linestring_buffer_owned_array,
                    supports_two_point_linestring_buffer_fast_path,
                )
                from vibespatial.runtime.adaptive import plan_dispatch_selection
                from vibespatial.runtime.crossover import (
                    PhysicalWorkEstimate,
                    estimate_physical_work_from_owned,
                )
                from vibespatial.runtime.precision import KernelClass

                # The direct DGA path bypasses GeometryArray.buffer(), so it
                # must honor the normal AUTO crossover itself.
                source_work = estimate_physical_work_from_owned(owned)
                output_coordinate_capacity = source_work.coordinate_count * (4 * int(quad_segs) + 2)
                selection = plan_dispatch_selection(
                    kernel_name="linestring_buffer",
                    kernel_class=KernelClass.CONSTRUCTIVE,
                    row_count=len(self),
                    work_estimate=PhysicalWorkEstimate(
                        row_count=len(self),
                        coordinate_count=source_work.coordinate_count,
                        segment_count=source_work.segment_count,
                        part_count=source_work.part_count,
                        output_row_count=len(self),
                        output_byte_count=output_coordinate_capacity * 16,
                        temporary_byte_count=len(self) * 24,
                        primary_unit_count=max(
                            len(self),
                            source_work.coordinate_count,
                            output_coordinate_capacity,
                        ),
                        primary_unit_name="linestring-buffer-output-coordinate",
                    ),
                    current_residency=owned.residency,
                )
                preserve_device_native = owned.residency is Residency.DEVICE
                force_two_point_gpu = (
                    len(self) >= LINESTRING_TWO_POINT_BUFFER_GPU_THRESHOLD
                    and supports_two_point_linestring_buffer_fast_path(
                        owned,
                        quad_segs=quad_segs,
                        cap_style=cap_style,
                        join_style=join_style,
                        single_sided=single_sided,
                    )
                )
                if (
                    selection.selected is not ExecutionMode.GPU
                    and not force_two_point_gpu
                    and not preserve_device_native
                ):
                    owned._record(
                        DiagnosticKind.RUNTIME,
                        (
                            "DeviceGeometryArray.buffer: linestring buffer honored "
                            f"AUTO crossover ({selection.reason})"
                        ),
                        visible=True,
                    )
                else:
                    result = linestring_buffer_owned_array(
                        owned,
                        distance,
                        quad_segs=quad_segs,
                        cap_style=cap_style,
                        join_style=join_style,
                        mitre_limit=mitre_limit,
                        dispatch_mode=ExecutionMode.GPU,
                    )
                    record_dispatch_event(
                        surface="DeviceGeometryArray.buffer",
                        operation="buffer",
                        implementation="linestring_buffer_owned_array",
                        reason=(
                            "device-resident DGA linestring buffer stayed on GPU"
                            if preserve_device_native
                            and selection.selected is not ExecutionMode.GPU
                            else "DGA direct two-point linestring buffer dispatch"
                            if force_two_point_gpu
                            and selection.selected is not ExecutionMode.GPU
                            else "DGA direct linestring buffer dispatch"
                        ),
                        detail=(
                            f"rows={len(self)}, family=linestring"
                            + (", residency=device" if preserve_device_native else "")
                            + (", shape=simple_two_point" if force_two_point_gpu else "")
                        ),
                        selected=ExecutionMode.GPU,
                    )
                    return DeviceGeometryArray._from_owned(
                        result,
                        crs=self._crs,
                        provenance=provenance,
                    )

            elif family is GeometryFamily.POLYGON and not has_empties:
                from vibespatial.constructive.polygon import polygon_buffer_owned_array

                result = polygon_buffer_owned_array(
                    owned,
                    distance,
                    quad_segs=quad_segs,
                    join_style=join_style,
                    mitre_limit=mitre_limit,
                )
                record_dispatch_event(
                    surface="DeviceGeometryArray.buffer",
                    operation="buffer",
                    implementation="polygon_buffer_owned_array",
                    reason="DGA direct polygon buffer dispatch",
                    detail=f"rows={len(self)}, family=polygon",
                    selected=ExecutionMode.GPU,
                )
                return DeviceGeometryArray._from_owned(
                    result,
                    crs=self._crs,
                    provenance=provenance,
                )

        # Shapely fallback for mixed families, nulls, empties, or unsupported params
        import shapely

        _record_shapely_fallback_event(
            surface="DeviceGeometryArray.buffer",
            reason="mixed families, nulls, empties, or unsupported params",
            owned=self.to_owned(),
            detail=f"rows={len(self)}",
            requested=get_requested_mode(),
            pipeline="constructive/buffer",
        )
        owned._record(
            DiagnosticKind.MATERIALIZATION,
            "DeviceGeometryArray.buffer: Shapely fallback",
            visible=True,
        )
        shapely_geoms = owned_to_shapely(self.to_owned())
        result = shapely.buffer(
            shapely_geoms,
            distance,
            quad_segs=quad_segs,
            cap_style=cap_style,
            join_style=join_style,
            mitre_limit=mitre_limit,
            single_sided=single_sided,
            **kwargs,
        )
        record_dispatch_event(
            surface="DeviceGeometryArray.buffer",
            operation="buffer",
            implementation="shapely_fallback",
            reason="DGA buffer: mixed families, nulls, empties, or unsupported params",
            detail=f"rows={len(self)}",
            selected=ExecutionMode.CPU,
        )
        new_owned = from_shapely_geometries(result.tolist())
        return DeviceGeometryArray._from_owned(
            new_owned,
            crs=self._crs,
            provenance=provenance,
        )

    def simplify(self, tolerance, preserve_topology=True):
        from vibespatial.constructive.simplify import simplify_owned

        result_owned = simplify_owned(self.to_owned(), tolerance, preserve_topology=preserve_topology)
        return DeviceGeometryArray._from_owned(result_owned, crs=self._crs)

    def remove_repeated_points(self, tolerance=0.0):
        from vibespatial.constructive.remove_repeated_points import (
            remove_repeated_points_owned,
        )

        result_owned = remove_repeated_points_owned(self.to_owned(), tolerance)
        return DeviceGeometryArray._from_owned(result_owned, crs=self._crs)

    def normalize(self, precision="auto"):
        from vibespatial.constructive.normalize import normalize_owned

        result = normalize_owned(self.to_owned(), precision=precision)
        return DeviceGeometryArray._from_owned(result, crs=self._crs)

    def offset_curve(self, distance, quad_segs=8, join_style="round", mitre_limit=5.0):
        from vibespatial.runtime import ExecutionMode, get_requested_mode
        from vibespatial.runtime.dispatch import record_dispatch_event

        owned = self.to_owned()

        # Check eligibility from owned metadata: linestring-only + non-round join
        families = set(owned.families.keys())
        is_eligible = (
            join_style != "round" and len(families) == 1 and GeometryFamily.LINESTRING in families
        )

        # Materialize once -- reuse in both eligible and fallback paths
        owned._record(
            DiagnosticKind.MATERIALIZATION,
            "DeviceGeometryArray.offset_curve: Shapely materialization",
            visible=True,
        )
        shapely_geoms = owned_to_shapely(owned)

        if is_eligible:
            from vibespatial.constructive.stroke import offset_curve_owned

            shapely_values = np.empty(len(self), dtype=object)
            shapely_values[:] = shapely_geoms
            result = offset_curve_owned(
                shapely_values,
                distance,
                quad_segs=quad_segs,
                join_style=join_style,
                mitre_limit=mitre_limit,
            )
            if result.owned_result is not None:
                new_owned = result.owned_result
            else:
                result_geometries = np.asarray(result.geometries, dtype=object)
                new_owned = from_shapely_geometries(result_geometries.tolist())
            if result.fallback_rows.size == 0:
                record_dispatch_event(
                    surface="DeviceGeometryArray.offset_curve",
                    operation="offset_curve",
                    implementation="offset_curve_owned",
                    reason="DGA offset_curve owned kernel (linestring, non-round join)",
                    detail=f"rows={len(self)}, join_style={join_style}",
                )
                return DeviceGeometryArray._from_owned(new_owned, crs=self._crs)
            _record_shapely_fallback_event(
                surface="DeviceGeometryArray.offset_curve",
                reason="owned offset-curve kernel used explicit host fallback rows",
                owned=owned,
                detail=(
                    f"rows={len(self)}, join_style={join_style}, "
                    f"fallback_rows={int(result.fallback_rows.size)}"
                ),
                requested=get_requested_mode(),
                pipeline="constructive/offset_curve",
            )
            record_dispatch_event(
                surface="DeviceGeometryArray.offset_curve",
                operation="offset_curve",
                implementation="offset_curve_owned_partial_fallback",
                reason="DGA offset_curve reused owned stroke result with explicit fallback rows",
                detail=f"rows={len(self)}, join_style={join_style}",
                selected=ExecutionMode.CPU,
            )
            return DeviceGeometryArray._from_owned(new_owned, crs=self._crs)

        # Shapely fallback
        import shapely

        _record_shapely_fallback_event(
            surface="DeviceGeometryArray.offset_curve",
            reason="ineligible for owned kernel or kernel had fallback rows",
            owned=owned,
            detail=f"rows={len(self)}, join_style={join_style}",
            requested=get_requested_mode(),
            pipeline="constructive/offset_curve",
        )
        fallback = shapely.offset_curve(
            shapely_geoms,
            distance,
            quad_segs=quad_segs,
            join_style=join_style,
            mitre_limit=mitre_limit,
        )
        record_dispatch_event(
            surface="DeviceGeometryArray.offset_curve",
            operation="offset_curve",
            implementation="shapely_fallback",
            reason="DGA offset_curve: ineligible for owned kernel or kernel had fallback rows",
            detail=f"rows={len(self)}, join_style={join_style}",
            selected=ExecutionMode.CPU,
        )
        new_owned = from_shapely_geometries(np.asarray(fallback, dtype=object).tolist())
        return DeviceGeometryArray._from_owned(new_owned, crs=self._crs)

    def make_valid(self, method="linework", keep_collapsed=True):
        from vibespatial.constructive.make_valid_pipeline import make_valid_owned

        # ADR-0005: pass owned directly — make_valid_owned will use GPU
        # ring-structure checks first and only materialize Shapely objects
        # if invalid rows need repair (lazy materialization).
        result = make_valid_owned(
            owned=self.to_owned(),
            method=method,
            keep_collapsed=keep_collapsed,
        )
        if result.native_geometry is not None:
            from vibespatial.api.geometry_array import GeometryArray

            return GeometryArray(result.geometries, crs=self._crs)
        # When all rows are already valid, result.owned carries the original
        # OwnedGeometryArray — reuse it directly to avoid a Shapely round-trip.
        if result.owned is not None:
            return DeviceGeometryArray._from_owned(result.owned, crs=self._crs)
        geoms = result.geometries
        new_owned = from_shapely_geometries(list(geoms) if not isinstance(geoms, list) else geoms)
        return DeviceGeometryArray._from_owned(new_owned, crs=self._crs)

    def representative_point(self):
        from vibespatial.constructive.representative_point import representative_point_owned

        result = representative_point_owned(self.to_owned())
        return DeviceGeometryArray._from_owned(result, crs=self._crs)

    def affine_transform(self, matrix):
        from vibespatial.constructive.affine_transform import affine_transform_owned

        result_owned = affine_transform_owned(self.to_owned(), matrix)
        return DeviceGeometryArray._from_owned(result_owned, crs=self._crs)

    def translate(self, xoff=0.0, yoff=0.0, zoff=0.0):
        from vibespatial.constructive.affine_transform import translate_owned

        result_owned = translate_owned(self.to_owned(), xoff, yoff, zoff)
        return DeviceGeometryArray._from_owned(result_owned, crs=self._crs)

    def rotate(self, angle, origin="center", use_radians=False):
        from vibespatial.constructive.affine_transform import rotate_owned

        result_owned = rotate_owned(self.to_owned(), angle, origin=origin, use_radians=use_radians)
        return DeviceGeometryArray._from_owned(result_owned, crs=self._crs)

    def scale(self, xfact=1.0, yfact=1.0, zfact=1.0, origin="center"):
        from vibespatial.constructive.affine_transform import scale_owned

        result_owned = scale_owned(self.to_owned(), xfact, yfact, zfact, origin=origin)
        return DeviceGeometryArray._from_owned(result_owned, crs=self._crs)

    def skew(self, xs=0.0, ys=0.0, origin="center", use_radians=False):
        from vibespatial.constructive.affine_transform import skew_owned

        result_owned = skew_owned(self.to_owned(), xs, ys, origin=origin, use_radians=use_radians)
        return DeviceGeometryArray._from_owned(result_owned, crs=self._crs)

    def count_coordinates(self):
        from vibespatial.constructive.properties import num_coordinates_owned

        return num_coordinates_owned(self.to_owned())

    def count_geometries(self):
        from vibespatial.constructive.properties import num_geometries_owned

        return num_geometries_owned(self.to_owned())

    def count_interior_rings(self):
        from vibespatial.constructive.properties import num_interior_rings_owned

        return num_interior_rings_owned(self.to_owned())

    def to_wkb(self, **kwargs):
        hex_output = bool(kwargs.pop("hex", False))
        if kwargs:
            raise TypeError(f"Unsupported to_wkb kwargs: {', '.join(sorted(kwargs))}")
        return np.asarray(
            _encode_owned_wkb_values(self.to_owned(), hex_output=hex_output), dtype=object
        )

    def to_wkt(self, **kwargs):
        return np.asarray(_encode_owned_wkt_values(self.to_owned(), **kwargs), dtype=object)

    def to_owned(self) -> OwnedGeometryArray:
        """Cross the explicit contiguous-owned boundary when required."""
        return self.physicalize_owned()

    # ------------------------------------------------------------------
    # Spatial index support
    # ------------------------------------------------------------------

    def supports_owned_spatial_input(self) -> bool:
        """Device-resident geometries always support the owned query path."""
        return True

    def owned_flat_sindex(self):
        """Return ``(owned, flat_index)`` without Shapely materialization.

        The flat spatial index is built directly from the OwnedGeometryArray
        coordinate buffers and cached for reuse.
        """
        if self._owned_flat_sindex_cache is not None:
            return self.to_owned(), self._owned_flat_sindex_cache

        from vibespatial.runtime.adaptive import plan_dispatch_selection
        from vibespatial.runtime.crossover import estimate_spatial_index_work_from_owned
        from vibespatial.runtime.precision import KernelClass
        from vibespatial.spatial.indexing import build_flat_spatial_index

        selection = plan_dispatch_selection(
            kernel_name="flat_index_build",
            kernel_class=KernelClass.COARSE,
            row_count=self.to_owned().row_count,
            work_estimate=estimate_spatial_index_work_from_owned(self.to_owned()),
            current_residency=self.to_owned().residency,
        )
        self._owned_flat_sindex_cache = build_flat_spatial_index(
            self.to_owned(),
            runtime_selection=selection.runtime_selection,
        )
        return self.to_owned(), self._owned_flat_sindex_cache

    @property
    def sindex(self):
        """Spatial index that routes through the owned query engine.

        Construction does NOT trigger Shapely materialization.  The returned
        ``SpatialIndex`` is backed by a lazy STRtree (built on first
        non-owned query) and eagerly caches the owned flat index.
        """
        if self._sindex_cache is None:
            # Build a lightweight SpatialIndex.  We must supply a numpy
            # Shapely array for the STRtree, but we defer materialization
            # by passing an empty array and relying on the owned dispatch
            # path for all actual queries.  The ``geometry_array=self``
            # lets SpatialIndex call our ``owned_flat_sindex()`` and
            # ``supports_owned_spatial_input()`` directly.
            self._sindex_cache = build_device_spatial_index(self)
        return self._sindex_cache

    @property
    def has_sindex(self) -> bool:
        """Check existence of the spatial index without generating it."""
        return self._sindex_cache is not None

    # ------------------------------------------------------------------
    # Binary predicates (owned-first)
    # ------------------------------------------------------------------

    def _binary_predicate(self, predicate: str, other, **kwargs):
        """Evaluate a binary predicate using owned arrays when possible.

        Routes through evaluate_binary_predicate with the device-resident
        OwnedGeometryArray, eliminating the Shapely round-trip for predicates
        supported by the repo-owned engine.
        """
        from vibespatial.predicates.binary import (
            NullBehavior,
            evaluate_binary_predicate,
            supports_binary_predicate,
        )
        from vibespatial.runtime import ExecutionMode, get_requested_mode
        from vibespatial.runtime.dispatch import record_dispatch_event
        from vibespatial.spatial.query import _extract_box_query_bounds, _query_point_tree_box_index

        if (
            predicate == "intersects"
            and isinstance(other, BaseGeometry)
            and get_requested_mode() is not ExecutionMode.CPU
        ):
            box_bounds = _extract_box_query_bounds(predicate, np.asarray([other], dtype=object))
            if box_bounds is not None:
                point_box_pairs = _query_point_tree_box_index(
                    self.to_owned(),
                    predicate=predicate,
                    query_row_count=1,
                    box_bounds=box_bounds,
                    force_gpu=True,
                )
                if point_box_pairs is not None:
                    mask = np.zeros(len(self), dtype=bool)
                    mask[point_box_pairs[1].astype(np.intp, copy=False)] = True
                    record_dispatch_event(
                        surface=f"DeviceGeometryArray.{predicate}",
                        operation=predicate,
                        implementation="owned_gpu_point_box_predicate",
                        reason="DeviceGeometryArray direct point-box predicate shortcut",
                        detail=f"rows={len(self)}, selected=gpu",
                        requested=ExecutionMode.AUTO,
                        selected=ExecutionMode.GPU,
                    )
                    return mask

        if supports_binary_predicate(predicate):
            left_owned = self.to_owned()
            if isinstance(other, DeviceGeometryArray):
                right_input = other.to_owned()
            elif isinstance(other, np.ndarray):
                right_input = other
            else:
                # Scalar BaseGeometry or other — pass through
                right_input = other

            result = evaluate_binary_predicate(
                predicate,
                left_owned,
                right_input,
                dispatch_mode=get_requested_mode(),
                null_behavior=NullBehavior.FALSE,
                **kwargs,
            )
            implementation = (
                "owned_gpu_predicate"
                if result.runtime_selection.selected is ExecutionMode.GPU
                else "owned_cpu_predicate"
            )
            record_dispatch_event(
                surface=f"DeviceGeometryArray.{predicate}",
                operation=predicate,
                implementation=implementation,
                reason=f"DeviceGeometryArray owned shortcut for {predicate}",
                detail=f"rows={result.row_count}, selected={result.runtime_selection.selected.value}",
                requested=result.runtime_selection.requested,
                selected=result.runtime_selection.selected,
            )
            return np.asarray(result.values, dtype=bool)

        # Unsupported predicate — fall back to Shapely
        import shapely as _shapely

        _record_shapely_fallback_event(
            surface=f"DeviceGeometryArray.{predicate}",
            reason="predicate is unsupported by the owned predicate engine",
            owned=self.to_owned(),
            detail=f"rows={len(self)}, predicate={predicate}",
            requested=get_requested_mode(),
            pipeline="predicate",
        )
        self.to_owned()._record(
            DiagnosticKind.MATERIALIZATION,
            f"DeviceGeometryArray.{predicate}: Shapely materialization required (unsupported predicate)",
            visible=True,
        )
        if isinstance(other, DeviceGeometryArray):
            other = other._ensure_shapely_cache()
        return getattr(_shapely, predicate)(self._ensure_shapely_cache(), other, **kwargs)

    def intersects(self, other, *args, **kwargs):
        return self._binary_predicate("intersects", other, **kwargs)

    def contains(self, other, *args, **kwargs):
        return self._binary_predicate("contains", other, **kwargs)

    def within(self, other, *args, **kwargs):
        return self._binary_predicate("within", other, **kwargs)

    def touches(self, other, *args, **kwargs):
        return self._binary_predicate("touches", other, **kwargs)

    def crosses(self, other, *args, **kwargs):
        return self._binary_predicate("crosses", other, **kwargs)

    def overlaps(self, other, *args, **kwargs):
        return self._binary_predicate("overlaps", other, **kwargs)

    def covers(self, other, *args, **kwargs):
        return self._binary_predicate("covers", other, **kwargs)

    def covered_by(self, other, *args, **kwargs):
        return self._binary_predicate("covered_by", other, **kwargs)

    def disjoint(self, other, *args, **kwargs):
        return self._binary_predicate("disjoint", other, **kwargs)

    def contains_properly(self, other, *args, **kwargs):
        return self._binary_predicate("contains_properly", other, **kwargs)

    def equals(self, other, *args, **kwargs):
        other_owned = self._coerce_other_to_owned(other)
        if other_owned is not None and other_owned.row_count == len(self):
            from .equality import geom_equals_owned

            return geom_equals_owned(self.to_owned(), other_owned)

        import shapely

        from vibespatial.runtime import get_requested_mode

        _record_shapely_fallback_event(
            surface="DeviceGeometryArray.equals",
            reason="Shapely materialization required",
            owned=self.to_owned(),
            detail=f"rows={len(self)}",
            requested=get_requested_mode(),
            pipeline="predicate",
        )
        self.to_owned()._record(
            DiagnosticKind.MATERIALIZATION,
            "DeviceGeometryArray.equals: Shapely materialization required",
            visible=True,
        )
        if isinstance(other, DeviceGeometryArray):
            other = other._ensure_shapely_cache()
        return shapely.equals(self._ensure_shapely_cache(), other, *args, **kwargs)

    def geom_equals(self, other):
        other_owned = self._coerce_other_to_owned(other)
        if other_owned is not None and other_owned.row_count == len(self):
            from .equality import geom_equals_owned

            return geom_equals_owned(self.to_owned(), other_owned)
        return self.equals(other)

    def geom_equals_exact(self, other, tolerance):
        other_owned = self._coerce_other_to_owned(other)
        if other_owned is not None and other_owned.row_count == len(self):
            from .equality import geom_equals_exact_owned

            return geom_equals_exact_owned(self.to_owned(), other_owned, tolerance)
        import shapely

        from vibespatial.runtime import get_requested_mode

        _record_shapely_fallback_event(
            surface="DeviceGeometryArray.geom_equals_exact",
            reason="Shapely fallback for non-DGA other",
            owned=self.to_owned(),
            detail=f"rows={len(self)}, tolerance={tolerance}",
            requested=get_requested_mode(),
            pipeline="predicate",
        )
        self.to_owned()._record(
            DiagnosticKind.MATERIALIZATION,
            "DeviceGeometryArray.geom_equals_exact: Shapely fallback for non-DGA other",
            visible=True,
        )
        return shapely.equals_exact(self._ensure_shapely_cache(), other, tolerance=tolerance)

    def geom_equals_identical(self, other):
        other_owned = self._coerce_other_to_owned(other)
        if other_owned is not None and other_owned.row_count == len(self):
            from .equality import geom_equals_identical_owned

            return geom_equals_identical_owned(self.to_owned(), other_owned)
        import shapely

        from vibespatial.runtime import get_requested_mode

        _record_shapely_fallback_event(
            surface="DeviceGeometryArray.geom_equals_identical",
            reason="Shapely fallback for non-DGA other",
            owned=self.to_owned(),
            detail=f"rows={len(self)}",
            requested=get_requested_mode(),
            pipeline="predicate",
        )
        self.to_owned()._record(
            DiagnosticKind.MATERIALIZATION,
            "DeviceGeometryArray.geom_equals_identical: Shapely fallback for non-DGA other",
            visible=True,
        )
        return shapely.equals_exact(self._ensure_shapely_cache(), other, tolerance=0.0)

    def _coerce_other_to_owned(self, other) -> OwnedGeometryArray | None:
        """Try to convert *other* to an OwnedGeometryArray for owned-path ops.

        Returns None when the type is not supported (caller should fall back
        to Shapely).
        """
        if isinstance(other, DeviceGeometryArray):
            if len(other) != len(self):
                raise ValueError(f"Lengths do not match: {len(self)} vs {len(other)}")
            return other.to_owned()
        other_values = getattr(other, "values", other)
        if isinstance(other_values, DeviceGeometryArray):
            if len(other_values) != len(self):
                raise ValueError(f"Lengths do not match: {len(self)} vs {len(other_values)}")
            return other_values.to_owned()
        other_owned = getattr(other_values, "_owned", None)
        if isinstance(other_owned, OwnedGeometryArray):
            if len(other_values) != len(self):
                raise ValueError(f"Lengths do not match: {len(self)} vs {len(other_values)}")
            return other_owned
        if _is_host_geometry_array_like(other_values):
            if len(other_values) != len(self):
                raise ValueError(f"Lengths do not match: {len(self)} vs {len(other_values)}")
            try:
                return from_shapely_geometries(other_values._data.tolist())
            except NotImplementedError:
                return None
        if hasattr(other_values, "to_owned"):
            if len(other_values) != len(self):
                raise ValueError(f"Lengths do not match: {len(self)} vs {len(other_values)}")
            try:
                coerced_owned = other_values.to_owned()
            except NotImplementedError:
                coerced_owned = None
            if isinstance(coerced_owned, OwnedGeometryArray):
                return coerced_owned
        from shapely.geometry.base import BaseGeometry as _BG

        if isinstance(other, _BG):
            try:
                return from_shapely_geometries([other])
            except NotImplementedError:
                return None
        if isinstance(other, np.ndarray) and other.dtype == object:
            if len(other) != len(self):
                raise ValueError(f"Lengths do not match: {len(self)} vs {len(other)}")
            try:
                return from_shapely_geometries(other.tolist())
            except NotImplementedError:
                return None
        return None

    def distance(self, other, *args, **kwargs):
        self.check_geographic_crs(stacklevel=5, operation="distance")
        other_owned = self._coerce_other_to_owned(other)
        if other_owned is not None:
            from vibespatial.spatial.distance_owned import distance_owned

            return distance_owned(self.to_owned(), other_owned)

        # Shapely fallback for unsupported 'other' types.
        import shapely

        from vibespatial.runtime import get_requested_mode

        _record_shapely_fallback_event(
            surface="DeviceGeometryArray.distance",
            reason="unsupported other type for owned distance path",
            owned=self.to_owned(),
            detail=f"rows={len(self)}",
            requested=get_requested_mode(),
            pipeline="measurement/distance",
        )
        self.to_owned()._record(
            DiagnosticKind.MATERIALIZATION,
            "DeviceGeometryArray.distance: Shapely fallback (unsupported other type)",
            visible=True,
        )
        if isinstance(other, DeviceGeometryArray):
            other = other._ensure_shapely_cache()
        return shapely.distance(self._ensure_shapely_cache(), other, *args, **kwargs)

    def dwithin(self, other, distance):
        self.check_geographic_crs(stacklevel=6, operation="dwithin")
        if isinstance(other, BaseGeometry):
            return _dwithin_scalar(self, other, distance)
        other_owned = self._coerce_other_to_owned(other)
        if other_owned is not None:
            from vibespatial.spatial.distance_owned import dwithin_owned

            return dwithin_owned(self.to_owned(), other_owned, distance)
        # Shapely fallback for unsupported 'other' types.
        from vibespatial.runtime import ExecutionMode
        from vibespatial.runtime.dispatch import record_dispatch_event

        d = self.distance(other)
        record_dispatch_event(
            surface="DeviceGeometryArray.dwithin",
            operation="dwithin",
            implementation="shapely_cpu",
            reason="unsupported other type for owned dwithin path",
            detail=f"rows={len(self)}, other_type={type(other).__name__}",
            selected=ExecutionMode.CPU,
        )
        return np.where(np.isnan(d), False, d <= distance)

    def _dwithin_scalar_with_native_rowset(
        self,
        other,
        distance,
        *,
        source_token=None,
        source_row_count=None,
    ):
        """Return public scalar-dwithin mask plus private device row positions."""
        if not isinstance(other, BaseGeometry):
            return None
        return _dwithin_scalar(
            self,
            other,
            distance,
            return_rowset=True,
            source_token=source_token,
            source_row_count=source_row_count,
        )

    def hausdorff_distance(self, other, densify=None):
        other_owned = self._coerce_other_to_owned(other)
        if other_owned is not None:
            from vibespatial.spatial.distance_metrics import hausdorff_distance_owned

            return hausdorff_distance_owned(self.to_owned(), other_owned, densify=densify)

        import shapely

        from vibespatial.runtime import get_requested_mode

        _record_shapely_fallback_event(
            surface="DeviceGeometryArray.hausdorff_distance",
            reason="unsupported other type for owned distance path",
            owned=self.to_owned(),
            detail=f"rows={len(self)}, densify={densify}",
            requested=get_requested_mode(),
            pipeline="measurement/distance",
        )

        self.to_owned()._record(
            DiagnosticKind.MATERIALIZATION,
            "DeviceGeometryArray.hausdorff_distance: Shapely fallback (unsupported other type)",
            visible=True,
        )
        if isinstance(other, DeviceGeometryArray):
            other = other._ensure_shapely_cache()
        return shapely.hausdorff_distance(self._ensure_shapely_cache(), other, densify=densify)

    def frechet_distance(self, other, densify=None):
        other_owned = self._coerce_other_to_owned(other)
        if other_owned is not None:
            from vibespatial.spatial.distance_metrics import frechet_distance_owned

            return frechet_distance_owned(self.to_owned(), other_owned, densify=densify)

        import shapely

        from vibespatial.runtime import get_requested_mode

        _record_shapely_fallback_event(
            surface="DeviceGeometryArray.frechet_distance",
            reason="unsupported other type for owned distance path",
            owned=self.to_owned(),
            detail=f"rows={len(self)}, densify={densify}",
            requested=get_requested_mode(),
            pipeline="measurement/distance",
        )

        self.to_owned()._record(
            DiagnosticKind.MATERIALIZATION,
            "DeviceGeometryArray.frechet_distance: Shapely fallback (unsupported other type)",
            visible=True,
        )
        if isinstance(other, DeviceGeometryArray):
            other = other._ensure_shapely_cache()
        return shapely.frechet_distance(self._ensure_shapely_cache(), other, densify=densify)

    def clip_by_rect(self, xmin, ymin, xmax, ymax):
        import shapely

        from vibespatial.constructive.clip_rect import clip_by_rect_owned
        from vibespatial.runtime import ExecutionMode, get_requested_mode
        from vibespatial.runtime.dispatch import record_dispatch_event
        from vibespatial.runtime.residency import Residency

        requested_mode = get_requested_mode()
        families = set(self.to_owned().families)
        gpu_family_supported = (
            (GeometryFamily.POINT in families and len(families) == 1)
            or GeometryFamily.LINESTRING in families
            or GeometryFamily.MULTILINESTRING in families
            or GeometryFamily.POLYGON in families
            or GeometryFamily.MULTIPOLYGON in families
        )
        dispatch_mode = (
            ExecutionMode.GPU
            if (
                requested_mode is ExecutionMode.AUTO
                and gpu_family_supported
                and self.to_owned().residency is Residency.DEVICE
            )
            else (
                ExecutionMode.CPU
                if requested_mode is not ExecutionMode.GPU and not gpu_family_supported
                else requested_mode
            )
        )

        result = clip_by_rect_owned(
            self.to_owned(),
            xmin,
            ymin,
            xmax,
            ymax,
            dispatch_mode=dispatch_mode,
        )
        selected = result.runtime_selection.selected
        record_dispatch_event(
            surface="DeviceGeometryArray.clip_by_rect",
            operation="clip_by_rect",
            implementation="owned_clip_by_rect",
            reason="DeviceGeometryArray owned shortcut for clip_by_rect",
            detail=f"rows={len(self)}, selected={selected.value}",
            selected=selected,
        )
        if result.owned_result is not None and result.owned_result_rows is not None:
            owned_result = result.owned_result
            row_map = np.asarray(result.owned_result_rows, dtype=np.int64)
            if (
                owned_result.row_count != self.to_owned().row_count
                or row_map.size != self.to_owned().row_count
                or not np.array_equal(
                    row_map,
                    np.arange(self.to_owned().row_count, dtype=np.int64),
                )
            ):
                base = build_null_owned_array(
                    self.to_owned().row_count,
                    residency=owned_result.residency,
                )
                owned_result = concat_owned_scatter(
                    base,
                    owned_result,
                    row_map,
                )
            return DeviceGeometryArray._from_owned(owned_result, crs=self._crs)
        # result.geometries is np.ndarray of Shapely objects; convert to owned
        # clip_by_rect returns GEOMETRYCOLLECTION EMPTY for clipped-away rows;
        # from_shapely_geometries doesn't support GeometryCollection, so treat
        # empties as None.
        geoms = np.asarray(result.geometries, dtype=object).copy()
        non_null = geoms != None  # noqa: E711 - intentional numpy identity mask
        if np.any(non_null):
            empty = np.zeros(len(geoms), dtype=bool)
            empty[non_null] = np.asarray(shapely.is_empty(geoms[non_null]), dtype=bool)
            geoms[empty] = None
        new_owned = from_shapely_geometries(
            geoms.tolist(),
            residency=self.to_owned().residency,
        )
        return DeviceGeometryArray._from_owned(new_owned, crs=self._crs)

    def _binary_constructive(self, op: str, other, *args, **kwargs):
        """Shared dispatch for binary constructive ops (intersection, union, etc.)."""
        other_owned = self._coerce_other_to_owned(other)
        if other_owned is not None:
            from vibespatial.constructive.binary_constructive import (
                binary_constructive_native,
            )
            from vibespatial.runtime.crossover import WorkloadShape

            grid_size = kwargs.get("grid_size", None)
            workload_shape = None
            from shapely.geometry.base import BaseGeometry as _BG

            if isinstance(other, _BG):
                workload_shape = WorkloadShape.SCALAR_RIGHT
            result_geometry = binary_constructive_native(
                op,
                self.to_owned(),
                other_owned,
                grid_size=grid_size,
                workload_shape=workload_shape,
            )
            result_geometry = result_geometry.with_crs(self._crs)
            if result_geometry.owned is not None:
                return DeviceGeometryArray._from_owned(
                    result_geometry.owned,
                    crs=self._crs,
                )
            return result_geometry.to_geoseries(
                index=pd.RangeIndex(len(self)),
                name="geometry",
            ).values

        # Shapely fallback for unsupported other types
        import shapely

        from vibespatial.runtime import get_requested_mode

        _record_shapely_fallback_event(
            surface=f"DeviceGeometryArray.{op}",
            reason="unsupported other type for owned constructive path",
            owned=self.to_owned(),
            detail=f"rows={len(self)}, op={op}",
            requested=get_requested_mode(),
            pipeline="constructive/binary",
        )
        self.to_owned()._record(
            DiagnosticKind.MATERIALIZATION,
            f"DeviceGeometryArray.{op}: Shapely fallback (unsupported other type)",
            visible=True,
        )
        if isinstance(other, DeviceGeometryArray):
            other = other._ensure_shapely_cache()
        result = getattr(shapely, op)(self._ensure_shapely_cache(), other, *args, **kwargs)
        try:
            new_owned = from_shapely_geometries(result.tolist())
        except NotImplementedError:
            from vibespatial.api.geometry_array import from_shapely

            return from_shapely(result.tolist(), crs=self._crs)
        return DeviceGeometryArray._from_owned(new_owned, crs=self._crs)

    def intersection(self, other, *args, **kwargs):
        return self._binary_constructive("intersection", other, *args, **kwargs)

    def union(self, other, *args, **kwargs):
        return self._binary_constructive("union", other, *args, **kwargs)

    def difference(self, other, *args, **kwargs):
        return self._binary_constructive("difference", other, *args, **kwargs)

    def symmetric_difference(self, other, *args, **kwargs):
        return self._binary_constructive("symmetric_difference", other, *args, **kwargs)

    # ------------------------------------------------------------------
    # Shapely materialization (lazy, demand-driven)
    # ------------------------------------------------------------------

    def _ensure_shapely_cache(self) -> np.ndarray:
        """Populate the full Shapely cache if not already present."""
        if self._shapely_cache is not None:
            return self._shapely_cache
        if self._composition is not None:
            cache = np.empty(len(self), dtype=object)
            cache[:] = None
            for part in self._composition.parts:
                rows = self._composition_part_rows_host(part)
                values = owned_to_shapely(
                    part.geometry.owned,
                    record_event=True,
                )
                present = pd.notna(values)
                cache[rows[present]] = values[present]
            self._shapely_cache = cache
            return cache
        self.to_owned()._record(
            DiagnosticKind.MATERIALIZATION,
            "DeviceGeometryArray: full Shapely cache materialized",
            visible=True,
        )
        cache = owned_to_shapely(self.to_owned(), record_event=False)
        self._shapely_cache = cache
        return cache

    @staticmethod
    def _composition_part_rows_host(part: Any) -> np.ndarray:
        rows = part.output_rows
        if hasattr(rows, "__cuda_array_interface__"):
            rows = get_cuda_runtime().copy_device_to_host(
                rows,
                reason="partitioned geometry row indirection terminal export",
                terminal_export=True,
            )
        return np.asarray(rows, dtype=np.int64)

    def _materialize_row(self, row_index: int) -> BaseGeometry | None:
        """Materialize a single row without populating the full cache."""
        if self._shapely_cache is not None:
            return self._shapely_cache[row_index]
        if self._composition is not None:
            return self._ensure_shapely_cache()[row_index]
        if self.to_owned().is_indexed_view:
            self.to_owned()._ensure_host_state()
            if not bool(self.to_owned().validity[row_index]):
                return None
            family_tag = int(self.to_owned().tags[row_index])
            family_row = int(self.to_owned().family_row_offsets[row_index])
            family = TAG_FAMILIES[family_tag]
            family_buffer = self.to_owned().families[family]
            return materialize_family_row(family_buffer, family_row)

        row_metadata = None
        if (
            getattr(self.to_owned(), "_validity", None) is None
            or getattr(self.to_owned(), "_tags", None) is None
            or getattr(self.to_owned(), "_family_row_offsets", None) is None
        ):
            row_metadata = _device_row_metadata(
                self.to_owned(),
                row_index,
                reason="DeviceGeometryArray scalar row metadata device export",
            )
        if row_metadata is None:
            if not bool(self.to_owned().validity[row_index]):
                return None
            family_tag = int(self.to_owned().tags[row_index])
            family_row = int(self.to_owned().family_row_offsets[row_index])
        else:
            valid, family_tag, family_row = row_metadata
            if not valid:
                return None
        # Ensure host state for the specific family
        self.to_owned()._ensure_host_state()
        family = TAG_FAMILIES[family_tag]
        family_buffer = self.to_owned().families[family]
        return materialize_family_row(family_buffer, family_row)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def view(self, dtype=None) -> DeviceGeometryArray:
        """Return a shallow view sharing the same owned array.

        pandas' ExtensionArray.view() calls ``self[:]`` which triggers a
        full ``take``; override to share the backing OwnedGeometryArray
        directly (semantically identical for immutable-in-practice DGA).
        """
        if self._composition is not None:
            result = DeviceGeometryArray._from_composition(
                self._composition,
                crs=self._crs,
                provenance=getattr(self, "_provenance", None),
            )
        else:
            result = DeviceGeometryArray._from_owned(
                self.__owned,
                crs=self._crs,
                provenance=getattr(self, "_provenance", None),
            )
        if self._shapely_cache is not None:
            result._shapely_cache = self._shapely_cache
        return result

    def __getitem__(self, idx: Any) -> DeviceGeometryArray | BaseGeometry | None:
        if isinstance(idx, (int, np.integer)):
            idx = int(idx)
            if idx < 0:
                idx += len(self)
            return self._materialize_row(idx)

        if isinstance(idx, slice) and idx.start is None and idx.stop is None and idx.step is None:
            return self.view()

        # Slice or array-like
        idx = pd.api.indexers.check_array_indexer(self, idx)
        if isinstance(idx, slice):
            indices = np.arange(len(self))[idx]
        else:
            indices = np.asarray(idx)
            if indices.dtype == bool:
                indices = np.flatnonzero(indices)

        result = self.take(indices)
        # Propagate shapely cache subset if available
        if self._shapely_cache is not None:
            result._shapely_cache = self._shapely_cache[indices]
        return result

    def __setitem__(self, key: Any, value: Any) -> None:
        key = pd.api.indexers.check_array_indexer(self, key)

        # Fast path: DGA-to-DGA assignment at the owned level.
        # Avoids Shapely materialization entirely (zero-copy discipline).
        if isinstance(value, DeviceGeometryArray):
            indices = self._resolve_setitem_indices(key, len(value))
            if indices is not None:
                if len(indices) == len(self) and len(indices) == value.to_owned().row_count:
                    # Full replacement: just swap the owned array.
                    self._owned = value.to_owned()
                    self._shapely_cache = None
                    self._provenance = getattr(value, "_provenance", None)
                    return
                if len(indices) == value.to_owned().row_count:
                    from vibespatial.geometry.owned import concat_owned_scatter

                    self._owned = concat_owned_scatter(
                        self.to_owned(),
                        value.to_owned(),
                        indices,
                    )
                    self._shapely_cache = None
                    self._provenance = None
                    return

        # Slow path: Shapely materialization for scalar or non-DGA values.
        cache = self._ensure_shapely_cache()
        if isinstance(value, DeviceGeometryArray):
            value_cache = value._ensure_shapely_cache()
            cache[key] = value_cache
        elif isinstance(value, BaseGeometry) or value is None:
            cache[key] = value
        else:
            raise TypeError(
                f"Value should be a BaseGeometry, None, or DeviceGeometryArray, got {type(value)}"
            )
        # Rebuild owned from modified shapely cache
        self._owned = from_shapely_geometries(cache.tolist())
        self._shapely_cache = cache
        self._provenance = None

    @staticmethod
    def _resolve_setitem_indices(
        key: Any,
        value_len: int,
    ) -> np.ndarray | None:
        """Convert a __setitem__ key to a flat int64 index array, or None."""
        if isinstance(key, np.ndarray):
            if key.dtype == bool:
                return np.flatnonzero(key)
            return np.asarray(key, dtype=np.int64)
        if isinstance(key, slice):
            # Full slice is the most common case from pandas internals.
            if key == slice(None):
                return np.arange(value_len, dtype=np.int64)
            return None
        if isinstance(key, tuple) and len(key) == 1:
            return DeviceGeometryArray._resolve_setitem_indices(key[0], value_len)
        return None

    def __eq__(self, other: Any) -> np.ndarray:
        if not isinstance(other, DeviceGeometryArray):
            return NotImplemented
        if len(self) != len(other):
            return NotImplemented
        import shapely

        left = self._ensure_shapely_cache()
        right = other._ensure_shapely_cache()
        result = np.zeros(len(self), dtype=bool)
        both_valid = self.to_owned().validity & other.to_owned().validity
        if both_valid.any():
            result[both_valid] = shapely.equals(left[both_valid], right[both_valid])
        return result

    # ------------------------------------------------------------------
    # take / copy / concat  (device-side, no Shapely round-trip)
    # ------------------------------------------------------------------

    def take(
        self,
        indices: np.ndarray | Sequence[int],
        *,
        allow_fill: bool = False,
        fill_value: Any = None,
    ) -> DeviceGeometryArray:
        indices = np.asarray(indices, dtype=np.int64)

        if (
            not allow_fill
            and int(indices.size) == len(self)
            and bool(np.array_equal(indices, np.arange(len(self), dtype=np.int64)))
        ):
            return self.view()

        if allow_fill:
            # pandas convention: negative indices in allow_fill mode mean "fill"
            mask = indices < 0
            if mask.any():
                # Use only valid indices for the take, then patch fill positions
                safe_indices = indices.copy()
                safe_indices[mask] = 0
                new_owned = self.to_owned().take(safe_indices)
                new_owned._apply_row_activity(~mask)
                return DeviceGeometryArray._from_owned(
                    new_owned,
                    crs=self._crs,
                    provenance=getattr(self, "_provenance", None),
                )

        if self._composition is not None:
            result = DeviceGeometryArray._from_composition(
                self._composition.take(indices),
                crs=self._crs,
                provenance=getattr(self, "_provenance", None),
            )
            if self._shapely_cache is not None:
                result._shapely_cache = self._shapely_cache[indices]
            return result

        new_owned = self.to_owned().take(indices)
        result = DeviceGeometryArray._from_owned(
            new_owned,
            crs=self._crs,
            provenance=getattr(self, "_provenance", None),
        )
        # Propagate shapely cache subset if available
        if self._shapely_cache is not None:
            result._shapely_cache = self._shapely_cache[indices]
        return result

    def copy(self) -> DeviceGeometryArray:
        if self._composition is not None:
            result = DeviceGeometryArray._from_composition(
                self._composition,
                crs=self._crs,
                provenance=getattr(self, "_provenance", None),
            )
            if self._shapely_cache is not None:
                result._shapely_cache = self._shapely_cache.copy()
            return result
        new_owned = _copy_owned_array(self.to_owned())
        new_owned._record(DiagnosticKind.CREATED, "DeviceGeometryArray: copy", visible=False)
        result = DeviceGeometryArray._from_owned(
            new_owned,
            crs=self._crs,
            provenance=getattr(self, "_provenance", None),
        )
        if self._shapely_cache is not None:
            result._shapely_cache = self._shapely_cache.copy()
        return result

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[DeviceGeometryArray]) -> DeviceGeometryArray:
        if not to_concat:
            empty_owned = OwnedGeometryArray(
                validity=np.array([], dtype=bool),
                tags=np.array([], dtype=np.int8),
                family_row_offsets=np.array([], dtype=np.int32),
                families={},
            )
            return cls._from_owned(empty_owned)

        target_residency = next(
            (
                native.residency
                for arr in to_concat
                if (
                    native := (
                        getattr(arr, "_owned", None)
                        or getattr(arr, "_composition", None)
                    )
                )
                is not None
            ),
            Residency.HOST,
        )
        normalized = []
        for arr in to_concat:
            if isinstance(arr, cls):
                normalized.append(arr)
                continue
            try:
                normalized.append(
                    cls._from_owned(
                        from_shapely_geometries(
                            np.asarray(arr, dtype=object).tolist(),
                            residency=target_residency,
                        ),
                        crs=getattr(arr, "crs", None),
                    )
                )
            except NotImplementedError:
                from vibespatial.api.geometry_array import GeometryArray

                data = np.concatenate(
                    [np.asarray(geometry, dtype=object) for geometry in to_concat]
                )
                crs = next(
                    (
                        getattr(geometry, "crs", None)
                        for geometry in to_concat
                        if getattr(geometry, "crs", None) is not None
                    ),
                    None,
                )
                return GeometryArray(data, crs=crs)
        to_concat = normalized

        if any(arr._composition is not None for arr in to_concat):
            from vibespatial.api._native_result_core import (
                GeometryNativeResult,
                NativeGeometryComposition,
            )

            geometries = [
                GeometryNativeResult.from_composition(arr._composition, crs=arr._crs)
                if arr._composition is not None
                else GeometryNativeResult.from_owned(arr.__owned, crs=arr._crs)
                for arr in to_concat
            ]
            crs = next((arr._crs for arr in to_concat if arr._crs is not None), None)
            composition = NativeGeometryComposition.concat(geometries, crs=crs)
            provenance = getattr(to_concat[0], "_provenance", None)
            if any(
                getattr(arr, "_provenance", None) != provenance
                for arr in to_concat[1:]
            ):
                provenance = None
            return cls._from_composition(
                composition,
                crs=crs,
                provenance=provenance,
            )

        all_owned = []
        for arr in to_concat:
            owned = getattr(arr, "_owned", None)
            if owned is None:
                try:
                    owned = from_shapely_geometries(
                        np.asarray(arr, dtype=object).tolist(),
                        residency=target_residency,
                    )
                except NotImplementedError:
                    from vibespatial.api.geometry_array import GeometryArray

                    data = np.concatenate([np.asarray(ga, dtype=object) for ga in to_concat])
                    crs = None
                    for ga in to_concat:
                        if getattr(ga, "_crs", None) is not None:
                            crs = ga._crs
                            break
                    return GeometryArray(data, crs=crs)
            elif owned.residency is not target_residency:
                owned = owned.move_to(
                    target_residency,
                    trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                    reason="align mixed geometry arrays for device_geometry concat",
                )
            all_owned.append(owned)
        new_owned = OwnedGeometryArray.concat(all_owned)

        # Use CRS from first array that has one
        crs = None
        for arr in to_concat:
            if arr._crs is not None:
                crs = arr._crs
                break

        provenance = getattr(to_concat[0], "_provenance", None)
        if any(getattr(arr, "_provenance", None) != provenance for arr in to_concat[1:]):
            provenance = None

        return cls._from_owned(new_owned, crs=crs, provenance=provenance)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def __getstate__(self) -> tuple[list[bytes | None], Any, str]:
        wkb = _serialize_owned_wkb(self.to_owned())
        return (wkb, self._crs, self.to_owned().residency.value)

    def __setstate__(
        self, state: tuple[list[bytes | None], Any] | tuple[list[bytes | None], Any, str]
    ) -> None:
        from vibespatial.io.arrow import decode_wkb_owned

        if len(state) == 2:
            wkb_list, crs = state
            residency = Residency.HOST
        else:
            wkb_list, crs, residency_value = state
            residency = Residency(residency_value)
        self._owned = decode_wkb_owned(wkb_list)
        if residency is Residency.DEVICE:
            self.to_owned().move_to(
                Residency.DEVICE,
                trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                reason="restored device-resident DeviceGeometryArray from pickle",
            )
        self._crs = crs
        self._shapely_cache = None
        self._provenance = None

    # ------------------------------------------------------------------
    # pandas interop
    # ------------------------------------------------------------------

    def _values_for_factorize(self) -> tuple[np.ndarray, Any]:
        import shapely

        cache = self._ensure_shapely_cache()
        wkb = np.empty(len(cache), dtype=object)
        for i, geom in enumerate(cache):
            if geom is None:
                wkb[i] = None
            else:
                wkb[i] = shapely.to_wkb(geom)
        return wkb, None

    def _reduce(
        self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs: Any
    ) -> Any:
        if name in ("any", "all"):
            cache = self._ensure_shapely_cache()
            return getattr(cache, name)(keepdims=keepdims)
        raise TypeError(
            f"'{type(self).__name__}' with dtype {self.dtype} does not support reduction '{name}'"
        )

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        from vibespatial.runtime.materialization import (
            NativeExportBoundary,
            record_native_export_boundary,
        )

        record_native_export_boundary(
            NativeExportBoundary(
                surface="vibespatial.geometry.DeviceGeometryArray.__array__",
                operation="device_geometryarray_to_numpy",
                target="numpy",
                reason="device geometry array exported through NumPy array protocol",
                detail=(
                    "residency="
                    f"{getattr(getattr(self._composition, 'residency', None), 'value', 'unknown')}"
                    if self._composition is not None
                    else f"{getattr(getattr(self.__owned, 'residency', None), 'value', 'unknown')}"
                ),
                row_count=len(self),
                d2h_transfer=(
                    self._composition is not None
                    or self.__owned.device_state is not None
                ),
            )
        )
        cache = self._ensure_shapely_cache()
        if dtype is not None:
            return cache.astype(dtype)
        return cache

    @property
    def diagnostics(self) -> list[DiagnosticEvent]:
        if self._composition is not None:
            return [
                event
                for part in self._composition.parts
                for event in part.geometry.owned.diagnostics
            ]
        return self.__owned.diagnostics


def _concat_family_buffers(
    family: GeometryFamily,
    buffers: list[FamilyGeometryBuffer],
) -> FamilyGeometryBuffer:
    """Concatenate multiple FamilyGeometryBuffers for the same family."""
    if len(buffers) == 1:
        return buffers[0]

    schema = get_geometry_buffer_schema(family)
    total_rows = sum(b.row_count for b in buffers)

    all_x = [b.x for b in buffers]
    all_y = [b.y for b in buffers]
    new_x = np.concatenate(all_x) if any(a.size for a in all_x) else np.empty(0, dtype=np.float64)
    new_y = np.concatenate(all_y) if any(a.size for a in all_y) else np.empty(0, dtype=np.float64)
    new_empty_mask = np.concatenate([b.empty_mask for b in buffers])

    # Concatenate bounds if all have them
    if all(b.bounds is not None for b in buffers):
        new_bounds = np.concatenate([b.bounds for b in buffers])
    else:
        new_bounds = None

    # Concatenate geometry_offsets with cumulative shift
    coord_cursor = 0
    geom_offset_parts = []
    for b in buffers:
        shifted = b.geometry_offsets[:-1] + coord_cursor
        geom_offset_parts.append(shifted)
        if family in (
            GeometryFamily.POLYGON,
            GeometryFamily.MULTILINESTRING,
            GeometryFamily.MULTIPOLYGON,
        ):
            # geometry_offsets index into ring/part offsets, not coords
            coord_cursor += int(b.geometry_offsets[-1])
        else:
            coord_cursor += int(b.geometry_offsets[-1])
    geom_offset_parts.append(np.array([coord_cursor], dtype=np.int32))
    new_geometry_offsets = np.concatenate(geom_offset_parts)

    new_part_offsets = None
    new_ring_offsets = None

    if family is GeometryFamily.POLYGON:
        # ring_offsets index into coords
        ring_cursor = 0
        ring_parts = []
        for b in buffers:
            shifted = b.ring_offsets[:-1] + ring_cursor
            ring_parts.append(shifted)
            ring_cursor += int(b.ring_offsets[-1])
        ring_parts.append(np.array([ring_cursor], dtype=np.int32))
        new_ring_offsets = np.concatenate(ring_parts)

    elif family is GeometryFamily.MULTILINESTRING:
        # part_offsets index into coords
        part_cursor = 0
        part_parts = []
        for b in buffers:
            shifted = b.part_offsets[:-1] + part_cursor
            part_parts.append(shifted)
            part_cursor += int(b.part_offsets[-1])
        part_parts.append(np.array([part_cursor], dtype=np.int32))
        new_part_offsets = np.concatenate(part_parts)

    elif family is GeometryFamily.MULTIPOLYGON:
        # part_offsets index into ring_offsets, ring_offsets index into coords
        part_cursor = 0
        part_parts = []
        for b in buffers:
            shifted = b.part_offsets[:-1] + part_cursor
            part_parts.append(shifted)
            part_cursor += int(b.part_offsets[-1])
        part_parts.append(np.array([part_cursor], dtype=np.int32))
        new_part_offsets = np.concatenate(part_parts)

        ring_cursor = 0
        ring_parts = []
        for b in buffers:
            shifted = b.ring_offsets[:-1] + ring_cursor
            ring_parts.append(shifted)
            ring_cursor += int(b.ring_offsets[-1])
        ring_parts.append(np.array([ring_cursor], dtype=np.int32))
        new_ring_offsets = np.concatenate(ring_parts)

    return FamilyGeometryBuffer(
        family=family,
        schema=schema,
        row_count=total_rows,
        x=new_x,
        y=new_y,
        geometry_offsets=new_geometry_offsets,
        empty_mask=new_empty_mask,
        part_offsets=new_part_offsets,
        ring_offsets=new_ring_offsets,
        bounds=new_bounds,
    )


def _copy_family_buffer(buf: FamilyGeometryBuffer) -> FamilyGeometryBuffer:
    return FamilyGeometryBuffer(
        family=buf.family,
        schema=buf.schema,
        row_count=buf.row_count,
        x=buf.x.copy(),
        y=buf.y.copy(),
        geometry_offsets=buf.geometry_offsets.copy(),
        empty_mask=buf.empty_mask.copy(),
        part_offsets=buf.part_offsets.copy() if buf.part_offsets is not None else None,
        ring_offsets=buf.ring_offsets.copy() if buf.ring_offsets is not None else None,
        bounds=buf.bounds.copy() if buf.bounds is not None else None,
        host_materialized=buf.host_materialized,
    )


def _copy_device_family_buffer(
    device_buf: DeviceFamilyGeometryBuffer,
) -> DeviceFamilyGeometryBuffer:
    import cupy as cp

    return DeviceFamilyGeometryBuffer(
        family=device_buf.family,
        x=cp.copy(device_buf.x),
        y=cp.copy(device_buf.y),
        geometry_offsets=cp.copy(device_buf.geometry_offsets),
        empty_mask=cp.copy(device_buf.empty_mask),
        part_offsets=None if device_buf.part_offsets is None else cp.copy(device_buf.part_offsets),
        ring_offsets=None if device_buf.ring_offsets is None else cp.copy(device_buf.ring_offsets),
        bounds=None if device_buf.bounds is None else cp.copy(device_buf.bounds),
        dense_single_ring_width=device_buf.dense_single_ring_width,
        axis_aligned_rectangles=device_buf.axis_aligned_rectangles,
        regular_grid_rect=device_buf.regular_grid_rect,
    )


def _copy_owned_array(owned: OwnedGeometryArray) -> OwnedGeometryArray:
    if getattr(owned, "_is_lazy_grouped_union_owned", False):
        return owned.copy()
    if owned.is_indexed_view and owned._base is not None and owned._index_map is not None:
        index_map = (
            _copy_device_array(owned._index_map)
            if hasattr(owned._index_map, "__cuda_array_interface__")
            else np.asarray(owned._index_map, dtype=np.int64).copy()
        )
        new_owned = OwnedGeometryArray._indexed_view(owned._base, index_map)
        new_owned._validity = None if owned._validity is None else owned._validity.copy()
        new_owned._tags = None if owned._tags is None else owned._tags.copy()
        new_owned._family_row_offsets = (
            None if owned._family_row_offsets is None else owned._family_row_offsets.copy()
        )
        new_owned.residency = owned.residency
        new_owned.geoarrow_backed = owned.geoarrow_backed
        new_owned.shares_geoarrow_memory = owned.shares_geoarrow_memory
        new_owned.device_adopted = owned.device_adopted
        if (
            owned.device_state is not None
            and new_owned.device_state is not None
            and owned.device_state.row_bounds is not None
        ):
            new_owned.device_state.row_bounds = _copy_device_array(owned.device_state.row_bounds)
        if owned.device_state is not None and new_owned.device_state is not None:
            _copy_device_state_semantic_metadata(owned.device_state, new_owned.device_state)
        cached_validity = owned._current_cached_validity_mask()
        if cached_validity is not None:
            new_owned._cached_is_valid_mask = cached_validity.copy()
        if hasattr(owned, "_grouped_convex_hull_source"):
            new_owned._grouped_convex_hull_source = owned._grouped_convex_hull_source
        return new_owned

    new_owned = OwnedGeometryArray(
        validity=None if owned._validity is None else owned._validity.copy(),
        tags=None if owned._tags is None else owned._tags.copy(),
        family_row_offsets=(
            None if owned._family_row_offsets is None else owned._family_row_offsets.copy()
        ),
        families={family: _copy_family_buffer(buf) for family, buf in owned.families.items()},
        residency=owned.residency,
        geoarrow_backed=owned.geoarrow_backed,
        shares_geoarrow_memory=False,
        _row_count=owned.row_count,
    )
    if owned.device_state is not None:
        new_owned.device_state = OwnedGeometryDeviceState(
            validity=_copy_device_array(owned.device_state.validity),
            tags=_copy_device_array(owned.device_state.tags),
            family_row_offsets=_copy_device_array(owned.device_state.family_row_offsets),
            families={
                family: _copy_device_family_buffer(device_buf)
                for family, device_buf in owned.device_state.families.items()
            },
            row_bounds=(
                _copy_device_array(owned.device_state.row_bounds)
                if owned.device_state.row_bounds is not None
                else None
            ),
        )
        _copy_device_state_semantic_metadata(owned.device_state, new_owned.device_state)
    cached_validity = owned._current_cached_validity_mask()
    if cached_validity is not None:
        new_owned._cached_is_valid_mask = cached_validity.copy()
    if hasattr(owned, "_grouped_convex_hull_source"):
        new_owned._grouped_convex_hull_source = owned._grouped_convex_hull_source
    return new_owned


def _copy_device_state_semantic_metadata(source, target) -> None:
    target.trusted_all_valid = source.trusted_all_valid
    target.trusted_homogeneous_family = source.trusted_homogeneous_family
    target.trusted_all_non_empty = source.trusted_all_non_empty
    target.trusted_polygonal_only = source.trusted_polygonal_only
    target.trusted_unique_family_rows = source.trusted_unique_family_rows
    target.trusted_family_domain = source.trusted_family_domain


def _copy_device_array(device_array):
    import cupy as cp

    return cp.copy(device_array)


def _encode_owned_wkb_values(
    owned: OwnedGeometryArray,
    *,
    hex_output: bool = False,
) -> list[bytes | str | None]:
    from vibespatial.io.arrow import encode_wkb_owned

    if all(buf.host_materialized for buf in owned.families.values()):
        return encode_wkb_owned(owned, hex=hex_output)

    try:
        from vibespatial.io.arrow import _encode_owned_wkb_column_device
    except ImportError as exc:  # pragma: no cover - depends on optional GPU IO stack
        raise RuntimeError(
            "Device-resident DeviceGeometryArray WKB serialization requires the native device WKB encoder"
        ) from exc

    import pyarrow as pa

    from vibespatial.cuda._runtime import pylibcudf_to_arrow

    arrow_col = pylibcudf_to_arrow(_encode_owned_wkb_column_device(owned))
    if pa.types.is_string(arrow_col.type) or pa.types.is_large_string(arrow_col.type):
        arrow_col = arrow_col.cast(pa.binary())
    values = arrow_col.to_pylist()
    normalized = [value.encode("latin1") if isinstance(value, str) else value for value in values]
    if hex_output:
        return [None if value is None else value.hex() for value in normalized]
    return normalized


def _serialize_owned_wkb(owned: OwnedGeometryArray) -> list[bytes | None]:
    return [None if value is None else value for value in _encode_owned_wkb_values(owned)]


def _host_view_family_buffer(
    owned: OwnedGeometryArray, family: GeometryFamily
) -> dict[str, np.ndarray | None]:
    buf = owned.families[family]
    if buf.host_materialized:
        return {
            "x": buf.x,
            "y": buf.y,
            "geometry_offsets": buf.geometry_offsets,
            "empty_mask": buf.empty_mask,
            "part_offsets": buf.part_offsets,
            "ring_offsets": buf.ring_offsets,
        }

    runtime = get_cuda_runtime()
    state = owned._ensure_device_state().families[family]
    return {
        "x": runtime.copy_device_to_host(
            state.x,
            reason=f"device geometry array {family.value} x-coordinate host export",
        ),
        "y": runtime.copy_device_to_host(
            state.y,
            reason=f"device geometry array {family.value} y-coordinate host export",
        ),
        "geometry_offsets": (
            buf.geometry_offsets
            if buf.geometry_offsets.size
            else runtime.copy_device_to_host(
                state.geometry_offsets,
                reason=f"device geometry array {family.value} geometry-offset host export",
            )
        ),
        "empty_mask": (
            buf.empty_mask
            if buf.empty_mask.size
            else runtime.copy_device_to_host(
                state.empty_mask,
                reason=f"device geometry array {family.value} empty-mask host export",
            )
        ),
        "part_offsets": (
            buf.part_offsets
            if buf.part_offsets is not None
            else None
            if state.part_offsets is None
            else runtime.copy_device_to_host(
                state.part_offsets,
                reason=f"device geometry array {family.value} part-offset host export",
            )
        ),
        "ring_offsets": (
            buf.ring_offsets
            if buf.ring_offsets is not None
            else None
            if state.ring_offsets is None
            else runtime.copy_device_to_host(
                state.ring_offsets,
                reason=f"device geometry array {family.value} ring-offset host export",
            )
        ),
    }


def _normalize_wkt_kwargs(kwargs: dict[str, Any]) -> tuple[int, bool]:
    rounding_precision = int(kwargs.pop("rounding_precision", 6))
    trim = bool(kwargs.pop("trim", True))
    output_dimension = kwargs.pop("output_dimension", 2)
    old_3d = kwargs.pop("old_3d", False)
    if kwargs:
        raise TypeError(f"Unsupported to_wkt kwargs: {', '.join(sorted(kwargs))}")
    if output_dimension not in (2, None):
        raise NotImplementedError("DeviceGeometryArray.to_wkt currently supports only 2D output")
    if old_3d not in (False, None):
        raise NotImplementedError(
            "DeviceGeometryArray.to_wkt does not support legacy 3D formatting"
        )
    return rounding_precision, trim


def _format_wkt_number(value: float, *, rounding_precision: int, trim: bool) -> str:
    value = float(value)
    if rounding_precision == -1:
        text = repr(value)
    else:
        text = f"{value:.{rounding_precision}f}"
    if trim and "e" not in text.lower():
        text = text.rstrip("0").rstrip(".")
    if text in {"-0", "-0.0", ""}:
        return "0"
    return text


def _format_wkt_coord(x: float, y: float, *, rounding_precision: int, trim: bool) -> str:
    return (
        f"{_format_wkt_number(x, rounding_precision=rounding_precision, trim=trim)} "
        f"{_format_wkt_number(y, rounding_precision=rounding_precision, trim=trim)}"
    )


def _format_wkt_coord_range(
    x: np.ndarray,
    y: np.ndarray,
    start: int,
    stop: int,
    *,
    rounding_precision: int,
    trim: bool,
) -> str:
    return ", ".join(
        _format_wkt_coord(
            float(x[idx]), float(y[idx]), rounding_precision=rounding_precision, trim=trim
        )
        for idx in range(int(start), int(stop))
    )


def _encode_family_row_wkt(
    family: GeometryFamily,
    host_view: dict[str, np.ndarray | None],
    row: int,
    *,
    rounding_precision: int,
    trim: bool,
) -> str:
    x = host_view["x"]
    y = host_view["y"]
    geometry_offsets = host_view["geometry_offsets"]
    empty_mask = host_view["empty_mask"]
    part_offsets = host_view["part_offsets"]
    ring_offsets = host_view["ring_offsets"]

    assert (
        x is not None and y is not None and geometry_offsets is not None and empty_mask is not None
    )
    if bool(empty_mask[row]):
        return f"{family.value.upper()} EMPTY"

    if family is GeometryFamily.POINT:
        coord_index = int(geometry_offsets[row])
        return f"POINT ({_format_wkt_coord(x[coord_index], y[coord_index], rounding_precision=rounding_precision, trim=trim)})"

    if family is GeometryFamily.LINESTRING:
        start = int(geometry_offsets[row])
        stop = int(geometry_offsets[row + 1])
        return f"LINESTRING ({_format_wkt_coord_range(x, y, start, stop, rounding_precision=rounding_precision, trim=trim)})"

    if family is GeometryFamily.MULTIPOINT:
        start = int(geometry_offsets[row])
        stop = int(geometry_offsets[row + 1])
        coords = ", ".join(
            f"({_format_wkt_coord(x[idx], y[idx], rounding_precision=rounding_precision, trim=trim)})"
            for idx in range(start, stop)
        )
        return f"MULTIPOINT ({coords})"

    if family is GeometryFamily.POLYGON:
        assert ring_offsets is not None
        ring_start = int(geometry_offsets[row])
        ring_stop = int(geometry_offsets[row + 1])
        rings = []
        for ring_idx in range(ring_start, ring_stop):
            start = int(ring_offsets[ring_idx])
            stop = int(ring_offsets[ring_idx + 1])
            rings.append(
                f"({_format_wkt_coord_range(x, y, start, stop, rounding_precision=rounding_precision, trim=trim)})"
            )
        return f"POLYGON ({', '.join(rings)})"

    if family is GeometryFamily.MULTILINESTRING:
        assert part_offsets is not None
        part_start = int(geometry_offsets[row])
        part_stop = int(geometry_offsets[row + 1])
        parts = []
        for part_idx in range(part_start, part_stop):
            start = int(part_offsets[part_idx])
            stop = int(part_offsets[part_idx + 1])
            parts.append(
                f"({_format_wkt_coord_range(x, y, start, stop, rounding_precision=rounding_precision, trim=trim)})"
            )
        return f"MULTILINESTRING ({', '.join(parts)})"

    if family is GeometryFamily.MULTIPOLYGON:
        assert part_offsets is not None and ring_offsets is not None
        poly_start = int(geometry_offsets[row])
        poly_stop = int(geometry_offsets[row + 1])
        polygons = []
        for poly_idx in range(poly_start, poly_stop):
            ring_start = int(part_offsets[poly_idx])
            ring_stop = int(part_offsets[poly_idx + 1])
            rings = []
            for ring_idx in range(ring_start, ring_stop):
                start = int(ring_offsets[ring_idx])
                stop = int(ring_offsets[ring_idx + 1])
                rings.append(
                    f"({_format_wkt_coord_range(x, y, start, stop, rounding_precision=rounding_precision, trim=trim)})"
                )
            polygons.append(f"({', '.join(rings)})")
        return f"MULTIPOLYGON ({', '.join(polygons)})"

    raise NotImplementedError(f"Unsupported geometry family for WKT encode: {family.value}")


def _encode_owned_wkt_values(owned: OwnedGeometryArray, **kwargs: Any) -> list[str | None]:
    rounding_precision, trim = _normalize_wkt_kwargs(dict(kwargs))
    if owned.device_state is not None:
        owned._ensure_device_state()
    family_views = {family: _host_view_family_buffer(owned, family) for family in owned.families}
    values: list[str | None] = [None] * owned.row_count
    for row_index in range(owned.row_count):
        if not bool(owned.validity[row_index]):
            continue
        family = TAG_FAMILIES[int(owned.tags[row_index])]
        family_row = int(owned.family_row_offsets[row_index])
        values[row_index] = _encode_family_row_wkt(
            family,
            family_views[family],
            family_row,
            rounding_precision=rounding_precision,
            trim=trim,
        )
    return values


def _compute_family_bounds(
    family: GeometryFamily,
    buf: FamilyGeometryBuffer,
) -> np.ndarray:
    """Compute per-row bounds (N, 4) from a FamilyGeometryBuffer's coordinate arrays."""
    if buf.bounds is not None:
        return buf.bounds

    n = buf.row_count
    result = np.full((n, 4), np.nan, dtype=np.float64)

    if n == 0:
        return result

    # Compute coordinate start/end per family row
    if family is GeometryFamily.POINT:
        # One coordinate per row
        for r in range(n):
            if not buf.empty_mask[r]:
                result[r] = [buf.x[r], buf.y[r], buf.x[r], buf.y[r]]
        return result

    # For all other families, determine the coord range per row
    if family in (GeometryFamily.LINESTRING, GeometryFamily.MULTIPOINT):
        # geometry_offsets index directly into coords
        starts = buf.geometry_offsets[:-1]
        ends = buf.geometry_offsets[1:]
    elif family is GeometryFamily.POLYGON:
        # geometry_offsets → ring_offsets → coords
        ring_starts = buf.geometry_offsets[:-1]
        ring_ends = buf.geometry_offsets[1:]
        starts = buf.ring_offsets[ring_starts]
        ends = buf.ring_offsets[ring_ends]
    elif family is GeometryFamily.MULTILINESTRING:
        # geometry_offsets → part_offsets → coords
        part_idx_starts = buf.geometry_offsets[:-1]
        part_idx_ends = buf.geometry_offsets[1:]
        starts = buf.part_offsets[part_idx_starts]
        ends = buf.part_offsets[part_idx_ends]
    elif family is GeometryFamily.MULTIPOLYGON:
        # geometry_offsets → part_offsets → ring_offsets → coords
        part_idx_starts = buf.geometry_offsets[:-1]
        part_idx_ends = buf.geometry_offsets[1:]
        ring_idx_starts = buf.part_offsets[part_idx_starts]
        ring_idx_ends = buf.part_offsets[part_idx_ends]
        starts = buf.ring_offsets[ring_idx_starts]
        ends = buf.ring_offsets[ring_idx_ends]
    else:
        return result

    for r in range(n):
        s, e = int(starts[r]), int(ends[r])
        if s < e and not buf.empty_mask[r]:
            result[r, 0] = buf.x[s:e].min()
            result[r, 1] = buf.y[s:e].min()
            result[r, 2] = buf.x[s:e].max()
            result[r, 3] = buf.y[s:e].max()

    return result


def _owned_prefers_device_bounds(owned: OwnedGeometryArray) -> bool:
    return owned.residency is Residency.DEVICE or any(
        not buf.host_materialized for buf in owned.families.values()
    )


def _compute_bounds_from_owned_device(owned: OwnedGeometryArray) -> np.ndarray:
    from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds
    from vibespatial.runtime import ExecutionMode

    return compute_geometry_bounds(owned, dispatch_mode=ExecutionMode.GPU)


def _compute_total_bounds_from_owned_device(owned: OwnedGeometryArray) -> np.ndarray:
    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
        return _compute_total_bounds_from_owned_host(owned)

    from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device

    d_bounds = compute_geometry_bounds_device(
        owned,
        preserve_indexed_view=True,
    )
    if int(d_bounds.shape[0]) == 0:
        return np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
    finite = ~cp.isnan(d_bounds[:, 0])
    min_xy = cp.amin(
        cp.where(finite[:, None], d_bounds[:, :2], cp.inf),
        axis=0,
    )
    max_xy = cp.amax(
        cp.where(finite[:, None], d_bounds[:, 2:], -cp.inf),
        axis=0,
    )
    bounds_device = cp.concatenate((min_xy, max_xy)).astype(cp.float64)
    bounds_device = cp.where(cp.any(finite), bounds_device, cp.nan)
    return get_cuda_runtime().copy_device_to_host(
        bounds_device,
        reason="DeviceGeometryArray total-bounds device summary host boundary",
    )


def _compute_total_bounds_from_owned_host(owned: OwnedGeometryArray) -> np.ndarray:
    b = _compute_bounds_from_owned_host(owned)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN slice encountered", RuntimeWarning)
        return np.array(
            (
                np.nanmin(b[:, 0]),
                np.nanmin(b[:, 1]),
                np.nanmax(b[:, 2]),
                np.nanmax(b[:, 3]),
            )
        )


def _compute_total_bounds_from_owned(owned: OwnedGeometryArray) -> np.ndarray:
    if _owned_prefers_device_bounds(owned):
        return _compute_total_bounds_from_owned_device(owned)
    return _compute_total_bounds_from_owned_host(owned)


def _compute_bounds_from_owned_host(owned: OwnedGeometryArray) -> np.ndarray:
    """Compute per-geometry bounds (N, 4) from an OwnedGeometryArray without Shapely."""
    n = owned.row_count
    result = np.full((n, 4), np.nan, dtype=np.float64)

    # Ensure host-side coordinates are available for any family that lacks
    # pre-computed bounds.  Device-backed buffers store x/y on the GPU and
    # keep host arrays empty; _compute_family_bounds needs the host arrays.
    needs_host = any(
        buf.bounds is None and not buf.host_materialized for buf in owned.families.values()
    )
    if needs_host:
        owned._ensure_host_state()

    # Pre-compute per-family bounds
    family_bounds: dict[GeometryFamily, np.ndarray] = {}
    for family, buf in owned.families.items():
        family_bounds[family] = _compute_family_bounds(family, buf)

    # Scatter family bounds into global result
    for i in range(n):
        tag = int(owned.tags[i])
        if tag == NULL_TAG:
            continue
        family = TAG_FAMILIES[tag]
        if family in family_bounds:
            family_row = int(owned.family_row_offsets[i])
            result[i] = family_bounds[family][family_row]

    return result


def _compute_bounds_from_owned(owned: OwnedGeometryArray) -> np.ndarray:
    """Compute per-geometry bounds (N, 4) from an OwnedGeometryArray without Shapely."""
    if _owned_prefers_device_bounds(owned):
        return _compute_bounds_from_owned_device(owned)
    return _compute_bounds_from_owned_host(owned)


def _dwithin_scalar(
    dga: DeviceGeometryArray,
    geom: BaseGeometry,
    dist: float,
    *,
    return_rowset: bool = False,
    source_token=None,
    source_row_count=None,
) -> np.ndarray:
    """dwithin against a single geometry — fully device-native.

    Pipeline (all on device except final D→H of boolean result):
      1. Device per-row bounds (cached after first compute_geometry_bounds GPU call)
      2. CuPy vectorized bbox overlap → device candidate mask (Tier 2)
      3. Tier 1 NVRTC distance kernel on candidate subset via device index arrays
      4. CuPy threshold filter + scatter into device boolean result (Tier 2)
      5. Single cp.asnumpy D→H for the final boolean array
    """

    def _return(mask, rowset=None):
        if return_rowset:
            return mask, rowset
        return mask

    from vibespatial.runtime import ExecutionMode, get_requested_mode
    from vibespatial.runtime.precision import KernelClass, PrecisionMode, select_precision_plan

    try:
        import cupy as cp
    except ImportError:
        cp = None

    from vibespatial.runtime.dispatch import record_dispatch_event

    runtime = get_cuda_runtime()
    if cp is None or not runtime.available() or get_requested_mode() is ExecutionMode.CPU:
        result = _dwithin_scalar_cpu(dga, geom, dist)
        record_dispatch_event(
            surface="DeviceGeometryArray._dwithin_scalar",
            operation="dwithin",
            implementation="shapely_cpu",
            reason="GPU unavailable or CPU mode requested",
            detail=f"rows={len(dga)}",
            selected=ExecutionMode.CPU,
        )
        return _return(result)

    n = len(dga)
    owned = dga._owned

    # Ensure device-resident geometry and per-row bounds on device.
    owned._ensure_device_state()
    d_bounds = _ensure_device_row_bounds(owned)

    # 1. Bbox candidate filter on device (Tier 2 CuPy).
    bx0, by0, bx1, by1 = geom.bounds
    d_cand_mask = (
        (d_bounds[:, 2] >= bx0 - dist)
        & (d_bounds[:, 0] <= bx1 + dist)
        & (d_bounds[:, 3] >= by0 - dist)
        & (d_bounds[:, 1] <= by1 + dist)
    )
    d_candidates = cp.flatnonzero(d_cand_mask)
    cand_count = int(d_candidates.size)

    if cand_count == 0:
        rowset = None
        if return_rowset:
            from vibespatial.api._native_rowset import NativeRowSet

            rowset = NativeRowSet.from_positions(
                cp.empty(0, dtype=cp.int64),
                source_token=source_token,
                source_row_count=source_row_count if source_row_count is not None else n,
                ordered=True,
                unique=True,
            )
        return _return(np.zeros(n, dtype=bool), rowset)

    # 2. Build 1-row query owned and move to device (~30 bytes H→D).
    query_owned = from_shapely_geometries([geom])
    query_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="dwithin_scalar: query geometry for GPU distance",
    )

    # 3. Device index arrays: query always row 0, tree at candidate positions.
    d_left = cp.zeros(cand_count, dtype=cp.int32)
    d_right = d_candidates.astype(cp.int32)
    d_distances = runtime.allocate((cand_count,), np.float64)

    # Resolve precision for the distance kernel (METRIC class per ADR-0002).
    from vibespatial.runtime._runtime import RuntimeSelection

    precision_selection = RuntimeSelection(
        requested=ExecutionMode.GPU,
        selected=ExecutionMode.GPU,
        reason="dwithin_scalar: GPU already confirmed available",
    )
    precision_plan = select_precision_plan(
        runtime_selection=precision_selection,
        kernel_class=KernelClass.METRIC,
        requested=PrecisionMode.AUTO,
    )

    # Dispatch to the appropriate distance kernel based on families.
    ok = _dwithin_dispatch_distance(
        query_owned,
        owned,
        d_left,
        d_right,
        d_distances,
        cand_count,
        compute_precision=precision_plan.compute_precision,
    )

    if not ok:
        # Unsupported family combination — fall back to CPU on candidates only.
        runtime.free(d_distances)
        record_dispatch_event(
            surface="DeviceGeometryArray._dwithin_scalar",
            operation="dwithin",
            implementation="shapely_cpu",
            reason="unsupported family combination for GPU distance kernel",
            detail=f"rows={n}, candidates={cand_count}",
            selected=ExecutionMode.CPU,
        )
        return _return(_dwithin_scalar_cpu(dga, geom, dist))

    # 4. Threshold filter + scatter on device (Tier 2 CuPy).
    d_within = cp.asarray(d_distances) <= dist
    rowset = None
    if return_rowset:
        from vibespatial.api._native_rowset import NativeRowSet

        rowset = NativeRowSet.from_positions(
            d_candidates[d_within].astype(cp.int64, copy=False),
            source_token=source_token,
            source_row_count=source_row_count if source_row_count is not None else n,
            ordered=True,
            unique=True,
        )
    d_result = cp.zeros(n, dtype=cp.bool_)
    d_result[d_candidates[d_within]] = True
    runtime.free(d_distances)

    # 5. Single D→H transfer.
    record_dispatch_event(
        surface="DeviceGeometryArray._dwithin_scalar",
        operation="dwithin",
        implementation="dwithin_scalar_gpu",
        reason="device-native bbox prefilter + distance kernel",
        detail=f"rows={n}, candidates={cand_count}",
        selected=ExecutionMode.GPU,
    )
    result = get_cuda_runtime().copy_device_to_host(
        d_result,
        reason="DeviceGeometryArray dwithin scalar result host boundary",
    )
    return _return(result, rowset)


def _ensure_device_row_bounds(owned: OwnedGeometryArray):
    """Return cached device per-row bounds (N, 4) CuPy array, computing if needed."""
    import cupy as cp

    from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device

    state = owned._ensure_device_state()
    if state.row_bounds is not None:
        return cp.asarray(state.row_bounds)

    # Trigger GPU bounds computation which now caches state.row_bounds.
    compute_geometry_bounds_device(owned)
    if state.row_bounds is not None:
        return cp.asarray(state.row_bounds)

    raise RuntimeError("device row bounds were requested but GPU bounds were not cached")


def _dwithin_dispatch_distance(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
    d_left,
    d_right,
    d_distances,
    pair_count: int,
    *,
    compute_precision=None,
) -> bool:
    """Launch the appropriate GPU distance kernel for dwithin refinement.

    Returns True if a kernel was dispatched, False if unsupported.
    METRIC kernel class per ADR-0002 — precision forwarded to point_distance.
    Segment distance stays fp64 (compliance gap tracked separately).

    *compute_precision*: ``PrecisionMode`` forwarded to point_distance kernels.
    Defaults to ``PrecisionMode.AUTO`` when ``None``.
    """
    from vibespatial.runtime.precision import PrecisionMode
    from vibespatial.spatial.point_distance import compute_point_distance_gpu
    from vibespatial.spatial.segment_distance import compute_segment_distance_gpu

    if compute_precision is None:
        compute_precision = PrecisionMode.AUTO

    query_families = list(query_owned.families.keys())
    if not query_families:
        return False
    query_family = query_families[0]

    tree_families = list(tree_owned.families.keys())
    if not tree_families:
        return False

    PT = GeometryFamily.POINT
    _POINT_DISTANCE_FAMILIES = frozenset(
        {
            GeometryFamily.LINESTRING,
            GeometryFamily.MULTILINESTRING,
            GeometryFamily.POLYGON,
            GeometryFamily.MULTIPOLYGON,
        }
    )
    _SEGMENT_FAMILIES = _POINT_DISTANCE_FAMILIES

    # Point query — most common for dwithin (centroid, POI, etc.)
    if query_family == PT:
        for tree_family in tree_families:
            if tree_family == PT:
                from vibespatial.spatial.nearest import _launch_point_point_distance_kernel

                _launch_point_point_distance_kernel(
                    query_owned,
                    tree_owned,
                    d_left,
                    d_right,
                    d_distances,
                    pair_count,
                )
                return True
            if tree_family in _POINT_DISTANCE_FAMILIES:
                return compute_point_distance_gpu(
                    query_owned,
                    tree_owned,
                    d_left,
                    d_right,
                    d_distances,
                    pair_count,
                    tree_family=tree_family,
                    compute_precision=compute_precision,
                )
        return False

    # Non-point query vs point tree — swap so point is on the query side.
    if PT in tree_families and query_family in _POINT_DISTANCE_FAMILIES:
        return compute_point_distance_gpu(
            tree_owned,
            query_owned,
            d_right,
            d_left,
            d_distances,
            pair_count,
            tree_family=query_family,
            compute_precision=compute_precision,
        )

    # Segment vs segment families.
    if query_family in _SEGMENT_FAMILIES:
        for tree_family in tree_families:
            if tree_family in _SEGMENT_FAMILIES:
                return compute_segment_distance_gpu(
                    query_owned,
                    tree_owned,
                    d_left,
                    d_right,
                    d_distances,
                    pair_count,
                    query_family=query_family,
                    tree_family=tree_family,
                )

    return False


def _dwithin_scalar_cpu(dga: DeviceGeometryArray, geom: BaseGeometry, dist: float) -> np.ndarray:
    """CPU fallback for dwithin — sindex prefilter + Shapely distance on candidates."""
    import shapely

    n = len(dga)
    result = np.zeros(n, dtype=bool)
    bx0, by0, bx1, by1 = geom.bounds
    query_box = (bx0 - dist, by0 - dist, bx1 + dist, by1 + dist)
    candidates = dga.sindex.query(shapely.box(*query_box))
    if len(candidates) == 0:
        return result
    subset = dga.take(candidates)
    candidate_geoms = subset._ensure_shapely_cache()
    dists = shapely.distance(candidate_geoms, geom)
    result[candidates] = dists <= dist
    return result


def _to_crs_owned(owned: OwnedGeometryArray, src_crs, dst_crs) -> DeviceGeometryArray:
    """Reproject an OwnedGeometryArray via vibeProj -- stays on device.

    Iterates over each geometry family's device coordinate buffers and
    calls vibeProj's ``transform_buffers()`` to project x/y in-place on
    the GPU.  No host round-trip, no Shapely materialization.
    """
    import cupy as cp
    from vibeproj import Transformer

    t = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    # Ensure device state exists
    device_state = owned._ensure_device_state()

    new_families: dict = {}
    for family, dbuf in device_state.families.items():
        if dbuf.x.size == 0:
            new_families[family] = DeviceFamilyGeometryBuffer(
                family=family,
                x=dbuf.x,
                y=dbuf.y,
                geometry_offsets=dbuf.geometry_offsets,
                empty_mask=dbuf.empty_mask,
                part_offsets=dbuf.part_offsets,
                ring_offsets=dbuf.ring_offsets,
                bounds=None,  # invalidated by reprojection
            )
            continue

        out_x = cp.empty_like(dbuf.x)
        out_y = cp.empty_like(dbuf.y)
        t.transform_buffers(dbuf.x, dbuf.y, out_x=out_x, out_y=out_y)

        new_families[family] = DeviceFamilyGeometryBuffer(
            family=family,
            x=out_x,
            y=out_y,
            geometry_offsets=dbuf.geometry_offsets,
            empty_mask=dbuf.empty_mask,
            part_offsets=dbuf.part_offsets,
            ring_offsets=dbuf.ring_offsets,
            bounds=None,  # invalidated by reprojection
        )

    new_device_state = OwnedGeometryDeviceState(
        validity=device_state.validity,
        tags=device_state.tags,
        family_row_offsets=device_state.family_row_offsets,
        families=new_families,
    )

    # Build a new OwnedGeometryArray with the projected device state.
    # Host-side family buffers are marked non-materialized so that any
    # access triggers a D->H copy from the *projected* device buffers.
    stale_families: dict = {}
    for family, buf in owned.families.items():
        stale_families[family] = FamilyGeometryBuffer(
            family=family,
            schema=buf.schema,
            row_count=buf.row_count,
            x=np.empty(0, dtype=np.float64),
            y=np.empty(0, dtype=np.float64),
            geometry_offsets=buf.geometry_offsets,
            empty_mask=buf.empty_mask,
            part_offsets=buf.part_offsets,
            ring_offsets=buf.ring_offsets,
            bounds=None,
            host_materialized=False,
        )
    new_owned = OwnedGeometryArray(
        validity=owned.validity,
        tags=owned.tags,
        family_row_offsets=owned.family_row_offsets,
        families=stale_families,
        residency=Residency.DEVICE,
        device_state=new_device_state,
        device_adopted=True,
    )

    # Eagerly compute device per-row bounds so subsequent sindex/dwithin
    # queries don't pay the cold bounds-computation cost.
    _ensure_device_row_bounds(new_owned)

    return DeviceGeometryArray._from_owned(new_owned, crs=dst_crs)
