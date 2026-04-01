from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter
from typing import Any

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

import shapely
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

from vibespatial.cuda._runtime import DeviceArray, get_cuda_runtime
from vibespatial.runtime import RuntimeSelection
from vibespatial.runtime.residency import Residency, TransferTrigger, select_residency_plan

from .buffers import (
    GEOMETRY_BUFFER_SCHEMAS,
    GeometryBufferSchema,
    GeometryFamily,
    get_geometry_buffer_schema,
)

NULL_TAG = -1
FAMILY_TAGS: dict[GeometryFamily, int] = {
    family: index for index, family in enumerate(GeometryFamily)
}
TAG_FAMILIES = {value: key for key, value in FAMILY_TAGS.items()}


def unique_tag_pairs(
    left_tags: np.ndarray,
    right_tags: np.ndarray,
) -> list[tuple[int, int]]:
    """Extract unique (left_tag, right_tag) pairs without Python-level iteration.

    Works with both numpy and CuPy arrays.  Packs two int8 tags into one
    int16 and calls the array library's ``unique``, then unpacks the small
    result (at most 36 pairs for 6 geometry families) on the host.

    This replaces the ``set(zip(left.tolist(), right.tolist()))`` anti-pattern
    which forces a full-array D->H transfer and O(n) Python iteration.

    Precondition: tag values must be non-negative and fit in int8 (0..127).
    Callers must filter null rows (NULL_TAG = -1) before calling.
    """
    xp = cp if cp is not None and not isinstance(left_tags, np.ndarray) else np
    packed = left_tags.astype(xp.int16) * np.int16(256) + right_tags.astype(xp.int16)
    unique_packed = xp.unique(packed)
    if xp is not np:
        unique_packed = unique_packed.get()
    return [(int(p // 256), int(p % 256)) for p in unique_packed]


class DiagnosticKind(StrEnum):
    CREATED = "created"
    TRANSFER = "transfer"
    MATERIALIZATION = "materialization"
    RUNTIME = "runtime"
    CACHE = "cache"


class BufferSharingMode(StrEnum):
    COPY = "copy"
    SHARE = "share"
    AUTO = "auto"


@dataclass(frozen=True)
class DiagnosticEvent:
    kind: DiagnosticKind
    detail: str
    residency: Residency
    visible_to_user: bool = False
    elapsed_seconds: float = 0.0
    bytes_transferred: int = 0


@dataclass(frozen=True)
class FamilyGeometryBuffer:
    family: GeometryFamily
    schema: GeometryBufferSchema
    row_count: int
    x: np.ndarray
    y: np.ndarray
    geometry_offsets: np.ndarray
    empty_mask: np.ndarray
    part_offsets: np.ndarray | None = None
    ring_offsets: np.ndarray | None = None
    bounds: np.ndarray | None = None
    host_materialized: bool = True


@dataclass(frozen=True)
class GeoArrowBufferView:
    family: GeometryFamily
    x: np.ndarray
    y: np.ndarray
    geometry_offsets: np.ndarray
    empty_mask: np.ndarray
    part_offsets: np.ndarray | None = None
    ring_offsets: np.ndarray | None = None
    bounds: np.ndarray | None = None
    shares_memory: bool = False


@dataclass(frozen=True)
class MixedGeoArrowView:
    validity: np.ndarray
    tags: np.ndarray
    family_row_offsets: np.ndarray
    families: dict[GeometryFamily, GeoArrowBufferView]
    shares_memory: bool = False


@dataclass
class DeviceFamilyGeometryBuffer:
    family: GeometryFamily
    x: DeviceArray
    y: DeviceArray
    geometry_offsets: DeviceArray
    empty_mask: DeviceArray
    part_offsets: DeviceArray | None = None
    ring_offsets: DeviceArray | None = None
    bounds: DeviceArray | None = None


@dataclass
class DeviceMetadataState:
    """Device-resident copies of the three routing metadata arrays.

    When ``residency=DEVICE``, these arrays live on GPU and the
    corresponding host numpy arrays in :class:`OwnedGeometryArray` may be
    ``None``.  Accessing the host properties triggers a lazy D->H transfer.
    """

    validity: DeviceArray      # bool
    tags: DeviceArray          # int8
    family_row_offsets: DeviceArray  # int32


@dataclass
class OwnedGeometryDeviceState:
    validity: DeviceArray
    tags: DeviceArray
    family_row_offsets: DeviceArray
    families: dict[GeometryFamily, DeviceFamilyGeometryBuffer]
    _column_refs: list | None = None
    row_bounds: DeviceArray | None = None  # cached per-row (N, 4) fp64 bounds on device


class OwnedGeometryArray:
    """Columnar geometry storage with optional device-resident metadata.

    The three routing metadata arrays -- ``validity``, ``tags``, and
    ``family_row_offsets`` -- are exposed as properties.  When the array
    is device-resident, the host numpy copies may be ``None`` internally;
    accessing any property lazily transfers from GPU to CPU, preserving
    full backward compatibility for host consumers while allowing
    GPU-only pipelines to avoid the D->H transfer entirely.
    """

    def __init__(
        self,
        validity: np.ndarray | None,
        tags: np.ndarray | None,
        family_row_offsets: np.ndarray | None,
        families: dict[GeometryFamily, FamilyGeometryBuffer],
        residency: Residency = Residency.HOST,
        diagnostics: list[DiagnosticEvent] | None = None,
        runtime_history: list[RuntimeSelection] | None = None,
        geoarrow_backed: bool = False,
        shares_geoarrow_memory: bool = False,
        device_adopted: bool = False,
        device_state: OwnedGeometryDeviceState | None = None,
        device_metadata: DeviceMetadataState | None = None,
        _row_count: int | None = None,
    ) -> None:
        self._validity = validity
        self._tags = tags
        self._family_row_offsets = family_row_offsets
        self.families = families
        self.residency = residency
        self.diagnostics: list[DiagnosticEvent] = diagnostics if diagnostics is not None else []
        self.runtime_history: list[RuntimeSelection] = runtime_history if runtime_history is not None else []
        self.geoarrow_backed = geoarrow_backed
        self.shares_geoarrow_memory = shares_geoarrow_memory
        self.device_adopted = device_adopted
        self.device_state = device_state
        self._device_metadata = device_metadata
        # Indexed-view support: when take() detects high index repetition,
        # the result stores a compact base array plus an index map instead
        # of physically copying all coordinate data.  This avoids OOM when
        # sjoin expands 38K unique polygons to 76.8M rows.
        # _base: the unique-row OwnedGeometryArray (physical data)
        # _index_map: int array mapping logical row -> base row.
        #   - numpy int64 when created from take() (host path)
        #   - CuPy int64 when created from device_take() (device path)
        self._base: OwnedGeometryArray | None = None
        self._index_map: np.ndarray | Any | None = None  # numpy or CuPy
        # Cache row_count so we don't need to materialise host arrays just
        # to query the length.  When host validity is present we derive it;
        # otherwise the caller must supply _row_count explicitly.
        if _row_count is not None:
            self._row_count = _row_count
        elif validity is not None:
            self._row_count = int(validity.size)
        elif device_metadata is not None:
            self._row_count = int(device_metadata.validity.size)
        elif device_state is not None:
            self._row_count = int(device_state.validity.size)
        else:
            raise ValueError(
                "Cannot determine row_count: provide validity or "
                "device_metadata or _row_count"
            )

    # ------------------------------------------------------------------
    # Lazy-materialising metadata properties
    # ------------------------------------------------------------------

    def _ensure_host_metadata(self) -> None:
        """Transfer device metadata arrays to host if not already present."""
        if self._validity is not None:
            return  # already materialised
        dm = self._device_metadata
        if dm is None and self.device_state is not None:
            # Fall back to device_state (pre-existing arrays)
            dm = DeviceMetadataState(
                validity=self.device_state.validity,
                tags=self.device_state.tags,
                family_row_offsets=self.device_state.family_row_offsets,
            )
        if dm is None:
            raise RuntimeError(
                "Host metadata is None and no device metadata available "
                "for lazy materialisation"
            )
        runtime = get_cuda_runtime()
        self._validity = runtime.copy_device_to_host(dm.validity)
        self._tags = runtime.copy_device_to_host(dm.tags)
        self._family_row_offsets = runtime.copy_device_to_host(dm.family_row_offsets)

    @property
    def validity(self) -> np.ndarray:
        if self._validity is None:
            self._ensure_host_metadata()
        return self._validity  # type: ignore[return-value]

    @validity.setter
    def validity(self, value: np.ndarray | None) -> None:
        self._validity = value
        if value is not None:
            self._row_count = int(value.size)

    @property
    def tags(self) -> np.ndarray:
        if self._tags is None:
            self._ensure_host_metadata()
        return self._tags  # type: ignore[return-value]

    @tags.setter
    def tags(self, value: np.ndarray | None) -> None:
        self._tags = value

    @property
    def family_row_offsets(self) -> np.ndarray:
        if self._family_row_offsets is None:
            self._ensure_host_metadata()
        return self._family_row_offsets  # type: ignore[return-value]

    @family_row_offsets.setter
    def family_row_offsets(self, value: np.ndarray | None) -> None:
        self._family_row_offsets = value

    @property
    def row_count(self) -> int:
        return self._row_count

    @property
    def is_indexed_view(self) -> bool:
        """True when this array is a virtual indexed view over a compact base."""
        return self._base is not None and self._index_map is not None

    @classmethod
    def _indexed_view(
        cls,
        base: OwnedGeometryArray,
        index_map: np.ndarray,
    ) -> OwnedGeometryArray:
        """Create an indexed view: a virtual expansion of *base* via *index_map*.

        The returned array appears to have ``len(index_map)`` rows, but stores
        only the unique rows in *base*.  Coordinate buffers are shared, not
        copied, saving memory when many output rows map to few unique geometries
        (e.g., sjoin expansion).

        *index_map* may be a numpy array (host path from :meth:`take`) or a
        CuPy array (device path from :meth:`device_take`).  When CuPy AND the
        base is device-resident, metadata expansion happens entirely on device
        — no D2H transfer of the index map or base metadata occurs.  The
        index map is stored as-is (CuPy or numpy) and consumed by the
        appropriate resolve path (:meth:`_device_resolve` or :meth:`_resolve`).

        Physical materialisation is deferred until a consumer needs contiguous
        buffers (kernel dispatch, GeoArrow export, etc.) via :meth:`_resolve`
        or :meth:`_device_resolve`.
        """
        _index_map_on_device = (
            cp is not None and hasattr(index_map, "__cuda_array_interface__")
        )
        index_map_size = int(index_map.size)

        if _index_map_on_device and base.device_state is not None:
            # Device path: expand metadata on device using CuPy fancy indexing.
            # This avoids the D2H transfer of the inverse map and the base
            # metadata arrays that the old code path forced.
            ds = base.device_state
            d_expanded_validity = ds.validity[index_map]
            d_expanded_tags = ds.tags[index_map]
            d_expanded_fro = ds.family_row_offsets[index_map]

            d_meta = DeviceMetadataState(
                validity=d_expanded_validity,
                tags=d_expanded_tags,
                family_row_offsets=d_expanded_fro,
            )

            view = cls(
                validity=None,         # host metadata stays lazy
                tags=None,
                family_row_offsets=None,
                families=base.families,
                residency=Residency.DEVICE,
                device_state=base.device_state,
                device_metadata=d_meta,
                _row_count=index_map_size,
            )
            view._base = base
            # Store CuPy index map directly — no .get()
            view._index_map = index_map.astype(cp.int64, copy=False)
        else:
            # Host path: expand metadata on host via numpy indexing.
            # This is the original behaviour for take() with numpy indices.
            h_index_map = np.asarray(index_map, dtype=np.int64)
            base_validity = base.validity
            base_tags = base.tags
            base_family_row_offsets = base.family_row_offsets

            expanded_validity = base_validity[h_index_map]
            expanded_tags = base_tags[h_index_map]
            expanded_fro = base_family_row_offsets[h_index_map]

            view = cls(
                validity=expanded_validity,
                tags=expanded_tags,
                family_row_offsets=expanded_fro,
                families=base.families,
                residency=base.residency,
                device_state=base.device_state,
                device_metadata=base._device_metadata,
                _row_count=index_map_size,
            )
            view._base = base
            view._index_map = h_index_map

        view._record(
            DiagnosticKind.CREATED,
            f"indexed view: {index_map_size} logical rows over "
            f"{base.row_count} base rows ({base.row_count / max(index_map_size, 1):.1%} unique)",
            visible=False,
        )
        return view

    def _resolve(self) -> OwnedGeometryArray:
        """Materialise an indexed view into a flat (non-virtual) host array.

        If this is already a flat array, returns ``self`` unchanged.
        If this is an indexed view with a CuPy index map and a
        device-resident base, delegates to :meth:`_device_resolve` first,
        then materialises host buffers.  Otherwise performs the host-side
        physical take directly.  Returns ``self`` after mutation.
        """
        if not self.is_indexed_view:
            return self

        _map_on_device = (
            cp is not None
            and hasattr(self._index_map, "__cuda_array_interface__")
        )

        if _map_on_device and self._base is not None and self._base.device_state is not None:
            # Device-resident indexed view: resolve via GPU gather,
            # then let _ensure_host_state handle the D2H if needed.
            self._device_resolve()
            return self

        base = self._base
        index_map = self._index_map
        # Perform the physical take on the compact base.
        # Use _physical_take (not take) to guarantee a flat result --
        # take() might produce another indexed view if the index_map
        # itself has high repetition, which would leave self unresolved.
        resolved = base._physical_take(index_map)
        # Copy resolved state into self, clearing the indexed-view link.
        self._validity = resolved._validity
        self._tags = resolved._tags
        self._family_row_offsets = resolved._family_row_offsets
        self.families = resolved.families
        self.residency = resolved.residency
        self.device_state = resolved.device_state
        self._device_metadata = resolved._device_metadata
        self._base = None
        self._index_map = None
        self._record(
            DiagnosticKind.MATERIALIZATION,
            f"resolved indexed view: materialised {self._row_count} rows",
            visible=False,
        )
        return self

    def _device_resolve(self) -> OwnedGeometryArray:
        """Materialise a device-resident indexed view via GPU gather.

        Requires ``_index_map`` to be a CuPy array and the base to have
        ``device_state``.  Performs ``_physical_device_take`` on the base
        using the device-resident index map, then copies the resolved
        device state into ``self``, clearing the indexed-view fields.

        This is the device-side counterpart of :meth:`_resolve` and
        eliminates the D->H->D round-trip that the old code path forced.
        """
        if not self.is_indexed_view:
            return self
        base = self._base
        d_index_map = self._index_map
        # Physical device take of the base using the CuPy index map.
        resolved = base._physical_device_take(d_index_map)
        # Copy resolved state into self, clearing the indexed-view link.
        self._validity = resolved._validity
        self._tags = resolved._tags
        self._family_row_offsets = resolved._family_row_offsets
        self.families = resolved.families
        self.residency = resolved.residency
        self.device_state = resolved.device_state
        self._device_metadata = resolved._device_metadata
        self._base = None
        self._index_map = None
        self._record(
            DiagnosticKind.MATERIALIZATION,
            f"device-resolved indexed view: materialised {self._row_count} rows on GPU",
            visible=False,
        )
        return self

    def family_has_rows(self, family: GeometryFamily) -> bool:
        """Check whether *family* has at least one geometry row to process.

        Reads from whichever side is authoritative: ``device_state`` when
        populated, host ``FamilyGeometryBuffer`` otherwise.  This avoids the
        bug where host stubs with ``host_materialized=False`` report empty
        offsets even when device buffers have real data.
        """
        if family not in self.families:
            return False

        # Device side is authoritative when populated
        if self.device_state is not None and family in self.device_state.families:
            d_buf = self.device_state.families[family]
            return int(d_buf.geometry_offsets.size) >= 2

        # Host side is authoritative
        buf = self.families[family]
        return buf.row_count > 0 and len(buf.geometry_offsets) >= 2

    def _record(
        self,
        kind: DiagnosticKind,
        detail: str,
        *,
        visible: bool = False,
        elapsed_seconds: float = 0.0,
        bytes_transferred: int = 0,
    ) -> None:
        self.diagnostics.append(
            DiagnosticEvent(
                kind=kind,
                detail=detail,
                residency=self.residency,
                visible_to_user=visible,
                elapsed_seconds=elapsed_seconds,
                bytes_transferred=bytes_transferred,
            )
        )

    def move_to(
        self,
        target: Residency | str,
        *,
        trigger: TransferTrigger | str,
        reason: str | None = None,
    ) -> OwnedGeometryArray:
        target_residency = target if isinstance(target, Residency) else Residency(target)
        self._last_transfer_seconds = 0.0
        self._last_transfer_bytes = 0
        if target_residency is self.residency:
            if target_residency is Residency.DEVICE:
                self._ensure_device_state()
            elif target_residency is Residency.HOST:
                self._ensure_host_state()
            if target_residency is not Residency.HOST or all(
                buffer.host_materialized for buffer in self.families.values()
            ):
                return self
        if target_residency is Residency.DEVICE:
            self._ensure_device_state()
        elif target_residency is Residency.HOST:
            self._ensure_host_state()
        plan = select_residency_plan(current=self.residency, target=target, trigger=trigger)
        old_residency = self.residency
        self.residency = plan.target
        self._record(
            DiagnosticKind.TRANSFER,
            reason or plan.reason,
            visible=plan.visible_to_user,
            elapsed_seconds=self._last_transfer_seconds,
            bytes_transferred=self._last_transfer_bytes,
        )
        if plan.transfer_required:
            from vibespatial.runtime.execution_trace import notify_transfer

            if old_residency is Residency.DEVICE and plan.target is Residency.HOST:
                notify_transfer(
                    direction="d2h",
                    trigger=str(plan.trigger),
                    reason=reason or plan.reason,
                )
            elif old_residency is Residency.HOST and plan.target is Residency.DEVICE:
                notify_transfer(
                    direction="h2d",
                    trigger=str(plan.trigger),
                    reason=reason or plan.reason,
                )
        return self

    def record_runtime_selection(self, selection: RuntimeSelection) -> None:
        self.runtime_history.append(selection)
        self._record(DiagnosticKind.RUNTIME, selection.reason, visible=True)

    def cache_bounds(self, bounds: np.ndarray) -> None:
        self._record(DiagnosticKind.CACHE, "cached per-geometry bounds", visible=False)
        runtime = get_cuda_runtime() if self.device_state is not None else None
        for family, buffer in self.families.items():
            row_indexes = np.flatnonzero(self.tags == FAMILY_TAGS[family])
            if row_indexes.size == 0:
                continue
            family_rows = self.family_row_offsets[row_indexes]
            if buffer.row_count == row_indexes.size and np.array_equal(
                family_rows,
                np.arange(buffer.row_count, dtype=np.int32),
            ):
                family_bounds = np.ascontiguousarray(bounds[row_indexes], dtype=np.float64)
            else:
                # Broadcast and repeated-row views can map many logical rows
                # to the same physical family row. Cache per-family bounds
                # using physical family-row indices so the family buffer shape
                # stays aligned with buffer.row_count.
                family_bounds = np.full((buffer.row_count, 4), np.nan, dtype=np.float64)
                family_bounds[family_rows] = bounds[row_indexes]
            self.families[family] = FamilyGeometryBuffer(
                family=buffer.family,
                schema=buffer.schema,
                row_count=buffer.row_count,
                x=buffer.x,
                y=buffer.y,
                geometry_offsets=buffer.geometry_offsets,
                empty_mask=buffer.empty_mask,
                part_offsets=buffer.part_offsets,
                ring_offsets=buffer.ring_offsets,
                bounds=family_bounds,
                host_materialized=buffer.host_materialized,
            )
            if self.device_state is not None:
                device_buffer = self.device_state.families[family]
                if device_buffer.bounds is None:
                    device_buffer.bounds = runtime.from_host(family_bounds)
                elif tuple(int(dim) for dim in device_buffer.bounds.shape) != family_bounds.shape:
                    runtime.free(device_buffer.bounds)
                    device_buffer.bounds = runtime.from_host(family_bounds)
                else:
                    runtime.copy_host_to_device(family_bounds, device_buffer.bounds)

    def cache_device_bounds(self, family: GeometryFamily, bounds: DeviceArray) -> None:
        state = self._ensure_device_state()
        family_state = state.families[family]
        if family_state.bounds is not None and family_state.bounds is not bounds:
            get_cuda_runtime().free(family_state.bounds)
        family_state.bounds = bounds

    def _ensure_device_state(self) -> OwnedGeometryDeviceState:
        # Indexed views share the base's families dict but have expanded
        # metadata.  Resolve to a flat array before uploading to device
        # so that kernels see contiguous per-row coordinate buffers.
        if self.is_indexed_view:
            _map_on_device = (
                cp is not None
                and hasattr(self._index_map, "__cuda_array_interface__")
            )
            if _map_on_device and self._base is not None and self._base.device_state is not None:
                # Fast path: resolve entirely on device, no D2H round-trip.
                self._device_resolve()
            else:
                self._resolve()
        if self.device_state is not None:
            return self.device_state
        runtime = get_cuda_runtime()
        if not runtime.available():
            raise RuntimeError("GPU execution was requested, but no CUDA device is available")
        # Safety check: detect unmaterialised host stubs that would upload
        # empty (size-0) coordinate buffers to the device.  When
        # host_materialized=False and x.shape=(0,), the family was created
        # as a lazy placeholder for a device-resident array whose
        # device_state has since been lost.  Uploading the empty stubs
        # causes kernels to read garbage from uninitialized GPU memory,
        # producing denormalized-double coordinates (e.g. 8e-309) and
        # downstream TopologyException / CUDA_ERROR_ILLEGAL_ADDRESS.
        for family, buffer in self.families.items():
            if (
                not buffer.host_materialized
                and buffer.row_count > 0
                and buffer.x.shape[0] == 0
            ):
                raise RuntimeError(
                    f"Cannot upload {family.value} family to device: host "
                    f"buffers are unmaterialised stubs (x.shape={buffer.x.shape}, "
                    f"host_materialized=False, row_count={buffer.row_count}). "
                    f"This OwnedGeometryArray was constructed from a "
                    f"device-resident source without propagating its "
                    f"device_state.  Fix the call site to either share the "
                    f"source's device_state or use residency=HOST."
                )
        t0 = perf_counter()
        total_bytes = self.validity.nbytes + self.tags.nbytes + self.family_row_offsets.nbytes
        d_validity = runtime.from_host(self.validity)
        d_tags = runtime.from_host(self.tags)
        d_family_row_offsets = runtime.from_host(self.family_row_offsets)
        self._device_metadata = DeviceMetadataState(
            validity=d_validity,
            tags=d_tags,
            family_row_offsets=d_family_row_offsets,
        )
        self.device_state = OwnedGeometryDeviceState(
            validity=d_validity,
            tags=d_tags,
            family_row_offsets=d_family_row_offsets,
            families={},
        )
        for family, buffer in self.families.items():
            buf_bytes = buffer.x.nbytes + buffer.y.nbytes + buffer.geometry_offsets.nbytes + buffer.empty_mask.nbytes
            if buffer.part_offsets is not None:
                buf_bytes += buffer.part_offsets.nbytes
            if buffer.ring_offsets is not None:
                buf_bytes += buffer.ring_offsets.nbytes
            if buffer.bounds is not None:
                buf_bytes += buffer.bounds.nbytes
            total_bytes += buf_bytes
            self.device_state.families[family] = DeviceFamilyGeometryBuffer(
                family=family,
                x=runtime.from_host(buffer.x),
                y=runtime.from_host(buffer.y),
                geometry_offsets=runtime.from_host(buffer.geometry_offsets),
                empty_mask=runtime.from_host(buffer.empty_mask),
                part_offsets=(
                    None if buffer.part_offsets is None else runtime.from_host(buffer.part_offsets)
                ),
                ring_offsets=(
                    None if buffer.ring_offsets is None else runtime.from_host(buffer.ring_offsets)
                ),
                bounds=None if buffer.bounds is None else runtime.from_host(buffer.bounds),
            )
        elapsed = perf_counter() - t0
        self._last_transfer_seconds = elapsed
        self._last_transfer_bytes = total_bytes
        return self.device_state

    def _ensure_host_state(self) -> None:
        # Indexed views share the base's families dict but have expanded
        # metadata.  Resolve to a flat array before materialising host
        # buffers so that downstream host consumers see correct per-row data.
        if self.is_indexed_view:
            _map_on_device = (
                cp is not None
                and hasattr(self._index_map, "__cuda_array_interface__")
            )
            if _map_on_device and self._base is not None and self._base.device_state is not None:
                # Device-resident indexed view: resolve on GPU first,
                # then fall through to the normal host materialisation below.
                self._device_resolve()
            else:
                self._resolve()
        if self.device_state is None:
            return
        if all(buffer.host_materialized for buffer in self.families.values()):
            return
        runtime = get_cuda_runtime()
        total_bytes = 0
        t0 = perf_counter()
        for family, buffer in tuple(self.families.items()):
            if buffer.host_materialized:
                continue
            device_buffer = self.device_state.families[family]
            x = runtime.copy_device_to_host(device_buffer.x)
            y = runtime.copy_device_to_host(device_buffer.y)
            total_bytes += x.nbytes + y.nbytes
            geometry_offsets = (
                buffer.geometry_offsets
                if buffer.geometry_offsets.size
                else runtime.copy_device_to_host(device_buffer.geometry_offsets)
            )
            empty_mask = (
                buffer.empty_mask
                if buffer.empty_mask.size
                else runtime.copy_device_to_host(device_buffer.empty_mask)
            )
            if not buffer.geometry_offsets.size:
                total_bytes += geometry_offsets.nbytes
            if not buffer.empty_mask.size:
                total_bytes += empty_mask.nbytes
            part_offsets = buffer.part_offsets
            if part_offsets is None and device_buffer.part_offsets is not None:
                part_offsets = runtime.copy_device_to_host(device_buffer.part_offsets)
                total_bytes += part_offsets.nbytes
            ring_offsets = buffer.ring_offsets
            if ring_offsets is None and device_buffer.ring_offsets is not None:
                ring_offsets = runtime.copy_device_to_host(device_buffer.ring_offsets)
                total_bytes += ring_offsets.nbytes
            bounds = buffer.bounds
            if bounds is None and device_buffer.bounds is not None:
                bounds = runtime.copy_device_to_host(device_buffer.bounds)
                total_bytes += bounds.nbytes
            self.families[family] = FamilyGeometryBuffer(
                family=buffer.family,
                schema=buffer.schema,
                row_count=buffer.row_count,
                x=np.ascontiguousarray(x, dtype=np.float64),
                y=np.ascontiguousarray(y, dtype=np.float64),
                geometry_offsets=np.ascontiguousarray(geometry_offsets, dtype=np.int32),
                empty_mask=np.ascontiguousarray(empty_mask, dtype=np.bool_),
                part_offsets=(
                    None if part_offsets is None else np.ascontiguousarray(part_offsets, dtype=np.int32)
                ),
                ring_offsets=(
                    None if ring_offsets is None else np.ascontiguousarray(ring_offsets, dtype=np.int32)
                ),
                bounds=None if bounds is None else np.ascontiguousarray(bounds, dtype=np.float64),
                host_materialized=True,
            )
        elapsed = perf_counter() - t0
        self._last_transfer_seconds = elapsed
        self._last_transfer_bytes = total_bytes

    @classmethod
    def concat(cls, arrays: list[OwnedGeometryArray]) -> OwnedGeometryArray:
        """Concatenate multiple OwnedGeometryArrays at the buffer level.

        When ALL inputs are device-resident (``residency == DEVICE``) and
        have device state populated, concatenation is performed entirely on
        GPU using CuPy -- no D->H transfer occurs.  The result is a
        device-resident OGA with lazy host stubs.

        When ANY input is host-resident (or lacks device state), falls back
        to the existing host-side concatenation path.
        """
        if not arrays:
            return OwnedGeometryArray(
                validity=np.empty(0, dtype=np.bool_),
                tags=np.empty(0, dtype=np.int8),
                family_row_offsets=np.empty(0, dtype=np.int32),
                families={},
                residency=Residency.HOST,
            )
        if len(arrays) == 1:
            return arrays[0]

        # Resolve any indexed views before concatenation — their families
        # dict is shared with a base array and would produce incorrect
        # family_row_offsets when combined with other arrays.
        for arr in arrays:
            if arr.is_indexed_view:
                arr._resolve()

        # --- Device-resident fast path ---
        all_device = (
            cp is not None
            and all(
                a.residency is Residency.DEVICE and a.device_state is not None
                for a in arrays
            )
        )
        if all_device:
            return cls._concat_device(arrays)

        # --- Host fallback path ---
        for arr in arrays:
            arr._ensure_host_state()

        # Concatenate top-level metadata arrays.
        all_validity = np.concatenate([a.validity for a in arrays])
        all_tags = np.concatenate([a.tags for a in arrays])

        # Build concatenated family_row_offsets: each array's family-local
        # offsets must be shifted by the cumulative family row count from
        # preceding arrays.
        total_rows = sum(a.row_count for a in arrays)
        all_family_row_offsets = np.full(total_rows, -1, dtype=np.int32)

        # Collect all families that appear in any array.
        all_family_keys: set[GeometryFamily] = set()
        for arr in arrays:
            all_family_keys.update(arr.families.keys())

        # Per-family cumulative row counts for offset shifting.
        family_cumulative: dict[GeometryFamily, int] = {f: 0 for f in all_family_keys}
        row_cursor = 0
        for arr in arrays:
            n = arr.row_count
            for family in all_family_keys:
                if family not in arr.families:
                    continue
                # Rows in this array belonging to this family.
                family_mask = arr.tags == FAMILY_TAGS[family]
                if not family_mask.any():
                    continue
                # Shift the family-local offsets by the cumulative count.
                shift = family_cumulative[family]
                src_offsets = arr.family_row_offsets[family_mask]
                all_family_row_offsets[row_cursor + np.flatnonzero(family_mask)] = (
                    src_offsets + shift
                )
                family_cumulative[family] += arr.families[family].row_count
            row_cursor += n

        # Concatenate per-family buffers.
        new_families: dict[GeometryFamily, FamilyGeometryBuffer] = {}
        for family in all_family_keys:
            buffers = [a.families[family] for a in arrays if family in a.families]
            if not buffers:
                continue
            new_families[family] = _concat_family_buffers(family, buffers)

        result = OwnedGeometryArray(
            validity=all_validity,
            tags=all_tags,
            family_row_offsets=all_family_row_offsets,
            families=new_families,
            residency=Residency.HOST,
        )
        total = sum(a.row_count for a in arrays)
        result._record(
            DiagnosticKind.CREATED,
            f"concatenated {len(arrays)} arrays totalling {total} rows",
            visible=False,
        )
        return result

    @classmethod
    def _concat_device(cls, arrays: list[OwnedGeometryArray]) -> OwnedGeometryArray:
        """Device-resident concatenation -- all work stays on GPU.

        Precondition: every element of *arrays* has ``residency == DEVICE``
        and a populated ``device_state``.  Called from :meth:`concat` only
        after the precondition has been verified.
        """
        device_states = [a.device_state for a in arrays]

        # 1. Concatenate routing metadata on device.
        d_all_validity = cp.concatenate([ds.validity for ds in device_states])
        d_all_tags = cp.concatenate([ds.tags for ds in device_states])
        total_rows = int(d_all_validity.size)

        # 2. Collect all family keys that appear across inputs.
        all_family_keys: set[GeometryFamily] = set()
        for ds in device_states:
            all_family_keys.update(ds.families.keys())

        # 3. Build concatenated family_row_offsets on device.
        #    Each array's per-family offsets must be shifted by the
        #    cumulative family row count from preceding arrays.
        d_all_family_row_offsets = cp.full(total_rows, -1, dtype=cp.int32)
        family_cumulative: dict[GeometryFamily, int] = {f: 0 for f in all_family_keys}
        row_cursor = 0
        for arr, ds in zip(arrays, device_states, strict=True):
            n = arr.row_count
            for family in all_family_keys:
                if family not in ds.families:
                    continue
                family_tag = FAMILY_TAGS[family]
                # Boolean mask for rows belonging to this family in this chunk
                chunk_tags = ds.tags
                family_mask = chunk_tags == family_tag
                if not bool(cp.any(family_mask)):
                    continue

                shift = family_cumulative[family]
                src_offsets = ds.family_row_offsets[family_mask]

                # Compute global positions for the family rows in this chunk
                chunk_positions = cp.flatnonzero(family_mask).astype(cp.int64)
                d_all_family_row_offsets[row_cursor + chunk_positions] = (
                    src_offsets + shift
                )

                # Count family rows: number of rows in device family buffer
                d_buf = ds.families[family]
                family_row_count = int(d_buf.geometry_offsets.size) - 1 if d_buf.geometry_offsets.size > 0 else 0
                family_cumulative[family] += family_row_count
            row_cursor += n

        # 4. Concatenate per-family device buffers.
        new_device_families: dict[GeometryFamily, DeviceFamilyGeometryBuffer] = {}
        for family in all_family_keys:
            family_bufs = [
                ds.families[family]
                for ds in device_states
                if family in ds.families
            ]
            if not family_bufs:
                continue
            new_device_families[family] = _concat_device_family_buffers(
                family, family_bufs,
            )

        # 5. Build host-side placeholder families (host_materialized=False).
        new_host_families: dict[GeometryFamily, FamilyGeometryBuffer] = {}
        for family, d_buf in new_device_families.items():
            schema = get_geometry_buffer_schema(family)
            fam_row_count = int(d_buf.geometry_offsets.size) - 1 if d_buf.geometry_offsets.size > 0 else 0
            new_host_families[family] = FamilyGeometryBuffer(
                family=family,
                schema=schema,
                row_count=fam_row_count,
                x=np.empty(0, dtype=np.float64),
                y=np.empty(0, dtype=np.float64),
                geometry_offsets=np.empty(0, dtype=np.int32),
                empty_mask=np.empty(0, dtype=np.bool_),
                host_materialized=False,
            )

        # 6. Assemble device-resident OGA -- host metadata arrays are None;
        #    lazy _ensure_host_metadata() will transfer on first access.
        d_meta = DeviceMetadataState(
            validity=d_all_validity,
            tags=d_all_tags,
            family_row_offsets=d_all_family_row_offsets,
        )
        result = OwnedGeometryArray(
            validity=None,
            tags=None,
            family_row_offsets=None,
            families=new_host_families,
            residency=Residency.DEVICE,
            device_state=OwnedGeometryDeviceState(
                validity=d_all_validity,
                tags=d_all_tags,
                family_row_offsets=d_all_family_row_offsets,
                families=new_device_families,
            ),
            device_metadata=d_meta,
            _row_count=total_rows,
        )
        result._record(
            DiagnosticKind.CREATED,
            f"device-resident concatenation of {len(arrays)} arrays "
            f"totalling {total_rows} rows",
            visible=False,
        )
        return result

    def diagnostics_report(self) -> dict[str, Any]:
        return {
            "residency": self.residency.value,
            "geoarrow_backed": self.geoarrow_backed,
            "shares_geoarrow_memory": self.shares_geoarrow_memory,
            "device_buffers_allocated": self.device_state is not None,
            "runtime_history": [selection.reason for selection in self.runtime_history],
            "events": [
                {
                    "kind": event.kind.value,
                    "detail": event.detail,
                    "residency": event.residency.value,
                    "visible_to_user": event.visible_to_user,
                    "elapsed_seconds": event.elapsed_seconds,
                    "bytes_transferred": event.bytes_transferred,
                }
                for event in self.diagnostics
            ],
        }

    # Repetition threshold for indexed-view optimisation.  When
    # unique_count / total_count < threshold AND total_count exceeds
    # _INDEXED_VIEW_MIN_ROWS, take() returns an indexed view instead
    # of physically copying coordinate data.
    _INDEXED_VIEW_RATIO_THRESHOLD: float = 0.5
    _INDEXED_VIEW_MIN_ROWS: int = 1000

    def take(self, indices: np.ndarray) -> OwnedGeometryArray:
        """Return a new OwnedGeometryArray containing only the rows at *indices*.

        Operates entirely at the buffer level -- no Shapely round-trip.
        When the array is DEVICE-resident **or** indices are already on device
        (CuPy / ``__cuda_array_interface__``), dispatches to :meth:`device_take`
        to keep all gathering on GPU.  Otherwise returns a HOST-resident array.

        When the indices have high repetition (many output rows mapping to
        few unique source rows), returns a virtual indexed view that stores
        only the unique rows and an index map, avoiding the physical
        coordinate copy.  This is transparent to consumers: kernel dispatch
        triggers :meth:`_resolve`, and :meth:`to_shapely` expands via cheap
        Python object references.

        Memory pressure is handled by the ADR-0040 tiered allocator:
        Tier B (default) retries with gc.collect on OOM; Tier C (opt-in)
        uses CUDA managed memory for datasets exceeding VRAM.
        """
        # Route to device_take when indices are already on device — avoids
        # a D→H transfer from np.asarray() followed by an H→D re-upload
        # inside device_take.  Phase 3 (vibeSpatial-p23.3).
        _indices_on_device = (
            cp is not None
            and hasattr(indices, "__cuda_array_interface__")
        )
        if (
            cp is not None
            and (
                (self.residency is Residency.DEVICE and self.device_state is not None)
                or _indices_on_device
            )
        ):
            return self.device_take(indices)

        if hasattr(indices, 'dtype') and indices.dtype == bool:
            indices = np.flatnonzero(indices)
        indices = np.asarray(indices, dtype=np.int64)

        # --- Indexed-view fast path ---
        # When repetition is high, avoid physical coordinate copy.
        # Use a cheap sampling heuristic first to avoid the O(n log n) cost
        # of np.unique when repetition is low.
        total = indices.size
        if total >= self._INDEXED_VIEW_MIN_ROWS:
            sample_size = min(1024, total)
            sample = indices[np.linspace(0, total - 1, sample_size, dtype=np.int64)]
            approx_unique = len(set(sample.tolist()))
            if approx_unique / sample_size >= self._INDEXED_VIEW_RATIO_THRESHOLD:
                # Low repetition — skip the full unique computation.
                return self._physical_take(indices)
            unique_indices, inverse = np.unique(indices, return_inverse=True)
            unique_count = unique_indices.size
            if unique_count / max(total, 1) < self._INDEXED_VIEW_RATIO_THRESHOLD:
                # If self is already an indexed view, compose the maps
                # so we don't stack indexed views on top of indexed views.
                if self.is_indexed_view:
                    # Map through: logical -> self._index_map -> base row
                    # unique_indices are logical indices into *self* which
                    # map via _index_map to base rows.
                    composed_map = self._index_map[unique_indices]
                    # Build a physical take of just the unique base rows
                    base_unique, base_inverse = np.unique(composed_map, return_inverse=True)
                    physical_base = self._base.take(base_unique)
                    # The final inverse: for each output row, which base_unique row?
                    final_inverse = base_inverse[inverse]
                    return OwnedGeometryArray._indexed_view(physical_base, final_inverse)

                # Standard case: build physical take of unique rows only.
                physical_base = self._physical_take(unique_indices)
                return OwnedGeometryArray._indexed_view(physical_base, inverse)

        # --- Physical copy path (original behaviour) ---
        return self._physical_take(indices)

    def _physical_take(self, indices: np.ndarray) -> OwnedGeometryArray:
        """Perform a physical (non-virtual) take, copying coordinate data."""
        # If this is an indexed view, resolve first so we have contiguous
        # buffers to gather from.
        if self.is_indexed_view:
            self._resolve()
        self._ensure_host_state()
        indices = np.asarray(indices, dtype=np.int64)
        new_validity = self.validity[indices]
        new_tags = self.tags[indices]
        new_family_row_offsets = np.full(indices.size, -1, dtype=np.int32)
        new_families: dict[GeometryFamily, FamilyGeometryBuffer] = {}

        for family, buffer in self.families.items():
            family_mask = new_tags == FAMILY_TAGS[family]
            if not family_mask.any():
                continue
            # Which rows in the *output* belong to this family, and what
            # were their family-row indices in the *source* buffer?
            source_family_rows = self.family_row_offsets[indices[family_mask]]
            new_family_row_offsets[family_mask] = np.arange(source_family_rows.size, dtype=np.int32)
            new_families[family] = _take_family_buffer(buffer, source_family_rows)

        result = OwnedGeometryArray(
            validity=new_validity,
            tags=new_tags,
            family_row_offsets=new_family_row_offsets,
            families=new_families,
            residency=Residency.HOST,
        )
        result._record(DiagnosticKind.CREATED, f"subset {indices.size} rows via take", visible=False)
        return result

    def device_take(self, indices) -> OwnedGeometryArray:
        """Device-side take — all gathering stays on GPU.

        Accepts numpy or CuPy indices/mask.  Returns a DEVICE-resident
        OwnedGeometryArray with host buffers marked ``host_materialized=False``.
        The host side is lazily populated by :meth:`_ensure_host_state` on demand.

        When indices have high repetition, returns a virtual indexed view
        instead of performing a full device gather.  See :meth:`take` for
        the design rationale.
        """
        if cp is None:
            raise RuntimeError("CuPy is required for device_take")

        # Accept numpy or CuPy indices — skip H→D upload when indices
        # are already on device (Phase 3: vibeSpatial-p23.3).
        if hasattr(indices, "__cuda_array_interface__"):
            d_indices = indices
        else:
            d_indices = cp.asarray(indices)

        # Bool mask → integer indices
        if d_indices.dtype == cp.bool_ or d_indices.dtype == bool:
            d_indices = cp.flatnonzero(d_indices).astype(cp.int64)
        else:
            d_indices = d_indices.astype(cp.int64, copy=False)

        # --- Indexed-view fast path ---
        # Use a cheap sampling heuristic to avoid the O(n log n) cost of
        # cp.unique when repetition is low.
        total = int(d_indices.size)
        if total >= self._INDEXED_VIEW_MIN_ROWS:
            sample_size = min(1024, total)
            d_sample = d_indices[cp.linspace(0, total - 1, sample_size, dtype=cp.int64)]
            approx_unique = int(cp.unique(d_sample).size)
            if approx_unique / sample_size >= self._INDEXED_VIEW_RATIO_THRESHOLD:
                return self._physical_device_take(d_indices)
            d_unique_indices, d_inverse = cp.unique(d_indices, return_inverse=True)
            unique_count = int(d_unique_indices.size)
            if unique_count / max(total, 1) < self._INDEXED_VIEW_RATIO_THRESHOLD:
                # Physical take of only the unique rows (on device)
                physical_base = self._physical_device_take(d_unique_indices)
                # Pass the CuPy inverse map directly -- no D2H transfer.
                # _indexed_view detects the CuPy array and expands metadata
                # on device, keeping the entire path GPU-resident.
                return OwnedGeometryArray._indexed_view(physical_base, d_inverse)

        # --- Physical copy path ---
        return self._physical_device_take(d_indices)

    def _physical_device_take(self, d_indices) -> OwnedGeometryArray:
        """Perform a physical (non-virtual) device-side take."""
        d_state = self._ensure_device_state()

        d_new_validity = d_state.validity[d_indices]
        d_new_tags = d_state.tags[d_indices]
        d_new_family_row_offsets = cp.full(int(d_indices.size), -1, dtype=cp.int32)

        new_device_families: dict[GeometryFamily, DeviceFamilyGeometryBuffer] = {}
        new_host_families: dict[GeometryFamily, FamilyGeometryBuffer] = {}

        for family, device_buffer in d_state.families.items():
            family_mask = d_new_tags == FAMILY_TAGS[family]
            if not bool(cp.any(family_mask)):
                continue
            source_family_rows = d_state.family_row_offsets[d_indices[family_mask]]
            d_new_family_row_offsets[family_mask] = cp.arange(
                int(source_family_rows.size), dtype=cp.int32,
            )
            new_device_families[family] = _device_take_family_buffer(
                device_buffer, family, source_family_rows,
            )
            # Host placeholder — _ensure_host_state will copy from device on demand
            schema = get_geometry_buffer_schema(family)
            new_host_families[family] = FamilyGeometryBuffer(
                family=family,
                schema=schema,
                row_count=int(source_family_rows.size),
                x=np.empty(0, dtype=np.float64),
                y=np.empty(0, dtype=np.float64),
                geometry_offsets=np.empty(0, dtype=np.int32),
                empty_mask=np.empty(0, dtype=np.bool_),
                host_materialized=False,
            )

        # Keep metadata device-only — host arrays stay None.
        # Lazy _ensure_host_metadata() will transfer on first property access.
        d_meta = DeviceMetadataState(
            validity=d_new_validity,
            tags=d_new_tags,
            family_row_offsets=d_new_family_row_offsets,
        )

        result = OwnedGeometryArray(
            validity=None,
            tags=None,
            family_row_offsets=None,
            families=new_host_families,
            residency=Residency.DEVICE,
            device_state=OwnedGeometryDeviceState(
                validity=d_new_validity,
                tags=d_new_tags,
                family_row_offsets=d_new_family_row_offsets,
                families=new_device_families,
            ),
            device_metadata=d_meta,
            _row_count=int(d_indices.size),
        )
        result._record(
            DiagnosticKind.CREATED,
            f"device-side subset {int(d_indices.size)} rows via device_take",
            visible=False,
        )
        return result

    def to_shapely(self) -> list[object | None]:
        # Indexed-view fast path: materialize only the compact base array
        # (e.g. 38K unique polygons) and expand via index map (76.8M Python
        # object references).  This avoids materializing 76.8M coordinate
        # rows -- Shapely objects are refcounted, so the reference expansion
        # is nearly free.
        if self.is_indexed_view:
            base_shapely = self._base.to_shapely()
            # Convert to numpy object array for O(1) index-map expansion
            base_arr = np.empty(len(base_shapely), dtype=object)
            base_arr[:] = base_shapely
            # If the index map is device-resident (CuPy), bring it to host
            # for the numpy object-array expansion.  This is an intentional
            # D2H transfer of the small index map only -- NOT the coordinate
            # buffers -- and is required because Shapely objects are
            # CPU-only Python objects.
            h_index_map = self._index_map
            if cp is not None and hasattr(h_index_map, "__cuda_array_interface__"):
                h_index_map = h_index_map.get()
            expanded = base_arr[h_index_map]
            self._record(
                DiagnosticKind.MATERIALIZATION,
                f"materialized shapely geometries (indexed view: "
                f"{self._base.row_count} base -> {self._row_count} expanded)",
                visible=True,
            )
            return list(expanded)

        if self.residency is Residency.DEVICE:
            self.move_to(
                Residency.HOST,
                trigger=TransferTrigger.USER_MATERIALIZATION,
                reason="materialized geometry objects for CPU validation",
            )
        else:
            self._ensure_host_state()
        self._record(DiagnosticKind.MATERIALIZATION, "materialized shapely geometries", visible=True)
        result: list[object | None] = []
        for row_index in range(self.row_count):
            if not bool(self.validity[row_index]):
                result.append(None)
                continue
            family = TAG_FAMILIES[int(self.tags[row_index])]
            family_buffer = self.families[family]
            family_row = int(self.family_row_offsets[row_index])
            result.append(_materialize_family_row(family_buffer, family_row))
        return result

    def to_wkb(self, *, hex: bool = False) -> list[bytes | str | None]:
        values = self.to_shapely()
        result: list[bytes | str | None] = []
        for value in values:
            if value is None:
                result.append(None)
                continue
            result.append(shapely.to_wkb(value, hex=hex))
        return result

    def to_geoarrow(
        self,
        *,
        sharing: BufferSharingMode | str = BufferSharingMode.COPY,
    ) -> MixedGeoArrowView:
        self._ensure_host_state()
        sharing_mode = normalize_buffer_sharing_mode(sharing)
        share = sharing_mode is not BufferSharingMode.COPY
        views = {
            family: GeoArrowBufferView(
                family=buffer.family,
                x=buffer.x if share else buffer.x.copy(),
                y=buffer.y if share else buffer.y.copy(),
                geometry_offsets=buffer.geometry_offsets if share else buffer.geometry_offsets.copy(),
                empty_mask=buffer.empty_mask if share else buffer.empty_mask.copy(),
                part_offsets=None if buffer.part_offsets is None else (buffer.part_offsets if share else buffer.part_offsets.copy()),
                ring_offsets=None if buffer.ring_offsets is None else (buffer.ring_offsets if share else buffer.ring_offsets.copy()),
                bounds=None if buffer.bounds is None else (buffer.bounds if share else buffer.bounds.copy()),
                shares_memory=share,
            )
            for family, buffer in self.families.items()
        }
        detail = (
            "exposed shared GeoArrow-style buffer view"
            if share
            else "materialized GeoArrow-style buffer view"
        )
        self._record(DiagnosticKind.MATERIALIZATION, detail, visible=True)
        return MixedGeoArrowView(
            validity=self.validity if share else self.validity.copy(),
            tags=self.tags if share else self.tags.copy(),
            family_row_offsets=self.family_row_offsets if share else self.family_row_offsets.copy(),
            families=views,
            shares_memory=share,
        )


def _gather_offset_slices(
    data: np.ndarray,
    offsets: np.ndarray,
    rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Gather coordinate slices for *rows* from an offset-indexed array.

    Returns (gathered_data, new_offsets) where new_offsets is a
    compacted offset array of length ``len(rows) + 1``.
    """
    starts = offsets[rows]
    ends = offsets[rows + 1]
    lengths = ends - starts
    new_offsets = np.empty(rows.size + 1, dtype=np.int32)
    new_offsets[0] = 0
    np.cumsum(lengths, out=new_offsets[1:])
    if data.ndim == 2:
        gathered = np.empty((int(new_offsets[-1]), data.shape[1]), dtype=data.dtype)
    else:
        gathered = np.empty(int(new_offsets[-1]), dtype=data.dtype)
    for i, (s, e) in enumerate(zip(starts, ends, strict=True)):
        gathered[new_offsets[i] : new_offsets[i + 1]] = data[s:e]
    return gathered, new_offsets


def _take_family_buffer(
    buffer: FamilyGeometryBuffer,
    family_rows: np.ndarray,
) -> FamilyGeometryBuffer:
    """Extract *family_rows* from a FamilyGeometryBuffer, compacting offsets."""
    family_rows = np.asarray(family_rows, dtype=np.int64)
    new_empty_mask = buffer.empty_mask[family_rows]
    schema = buffer.schema
    new_bounds = buffer.bounds[family_rows] if buffer.bounds is not None else None

    if buffer.family in (GeometryFamily.POINT, GeometryFamily.LINESTRING, GeometryFamily.MULTIPOINT):
        coords, new_geom_offsets = _gather_offset_slices(
            np.column_stack([buffer.x, buffer.y]),
            buffer.geometry_offsets,
            family_rows,
        )
        new_x = np.ascontiguousarray(coords[:, 0]) if coords.size else np.empty(0, dtype=np.float64)
        new_y = np.ascontiguousarray(coords[:, 1]) if coords.size else np.empty(0, dtype=np.float64)
        return FamilyGeometryBuffer(
            family=buffer.family, schema=schema,
            row_count=family_rows.size,
            x=new_x, y=new_y,
            geometry_offsets=new_geom_offsets,
            empty_mask=new_empty_mask,
            bounds=new_bounds,
        )

    if buffer.family is GeometryFamily.POLYGON:
        # geometry_offsets → ring_offsets → coords
        rings, new_geom_offsets = _gather_offset_slices(
            np.arange(buffer.ring_offsets.size, dtype=np.int32),
            buffer.geometry_offsets,
            family_rows,
        )
        ring_indices = rings.astype(np.int64)
        coords, new_ring_offsets = _gather_offset_slices(
            np.column_stack([buffer.x, buffer.y]),
            buffer.ring_offsets,
            ring_indices,
        )
        new_x = np.ascontiguousarray(coords[:, 0]) if coords.size else np.empty(0, dtype=np.float64)
        new_y = np.ascontiguousarray(coords[:, 1]) if coords.size else np.empty(0, dtype=np.float64)
        return FamilyGeometryBuffer(
            family=buffer.family, schema=schema,
            row_count=family_rows.size,
            x=new_x, y=new_y,
            geometry_offsets=new_geom_offsets,
            empty_mask=new_empty_mask,
            ring_offsets=new_ring_offsets,
            bounds=new_bounds,
        )

    if buffer.family is GeometryFamily.MULTILINESTRING:
        # geometry_offsets → part_offsets → coords
        parts, new_geom_offsets = _gather_offset_slices(
            np.arange(buffer.part_offsets.size, dtype=np.int32),
            buffer.geometry_offsets,
            family_rows,
        )
        part_indices = parts.astype(np.int64)
        coords, new_part_offsets = _gather_offset_slices(
            np.column_stack([buffer.x, buffer.y]),
            buffer.part_offsets,
            part_indices,
        )
        new_x = np.ascontiguousarray(coords[:, 0]) if coords.size else np.empty(0, dtype=np.float64)
        new_y = np.ascontiguousarray(coords[:, 1]) if coords.size else np.empty(0, dtype=np.float64)
        return FamilyGeometryBuffer(
            family=buffer.family, schema=schema,
            row_count=family_rows.size,
            x=new_x, y=new_y,
            geometry_offsets=new_geom_offsets,
            empty_mask=new_empty_mask,
            part_offsets=new_part_offsets,
            bounds=new_bounds,
        )

    if buffer.family is GeometryFamily.MULTIPOLYGON:
        # geometry_offsets → part_offsets → ring_offsets → coords
        parts, new_geom_offsets = _gather_offset_slices(
            np.arange(buffer.part_offsets.size, dtype=np.int32),
            buffer.geometry_offsets,
            family_rows,
        )
        part_indices = parts.astype(np.int64)
        rings, new_part_offsets = _gather_offset_slices(
            np.arange(buffer.ring_offsets.size, dtype=np.int32),
            buffer.part_offsets,
            part_indices,
        )
        ring_indices = rings.astype(np.int64)
        coords, new_ring_offsets = _gather_offset_slices(
            np.column_stack([buffer.x, buffer.y]),
            buffer.ring_offsets,
            ring_indices,
        )
        new_x = np.ascontiguousarray(coords[:, 0]) if coords.size else np.empty(0, dtype=np.float64)
        new_y = np.ascontiguousarray(coords[:, 1]) if coords.size else np.empty(0, dtype=np.float64)
        return FamilyGeometryBuffer(
            family=buffer.family, schema=schema,
            row_count=family_rows.size,
            x=new_x, y=new_y,
            geometry_offsets=new_geom_offsets,
            empty_mask=new_empty_mask,
            part_offsets=new_part_offsets,
            ring_offsets=new_ring_offsets,
            bounds=new_bounds,
        )

    raise NotImplementedError(f"take not implemented for {buffer.family.value}")


def _concat_offset_arrays(
    offset_arrays: list[np.ndarray],
    coord_counts: list[int],
) -> np.ndarray:
    """Concatenate offset arrays, shifting each by the cumulative coordinate count.

    Each offset array has length (row_count + 1).  The result drops the
    leading zero from all arrays after the first and shifts values so they
    form a single contiguous offset array.
    """
    if len(offset_arrays) == 1:
        return offset_arrays[0]
    parts: list[np.ndarray] = [offset_arrays[0]]
    cumulative = coord_counts[0]
    for offsets, count in zip(offset_arrays[1:], coord_counts[1:], strict=True):
        # Drop the leading 0 from subsequent arrays and shift.
        parts.append(offsets[1:] + cumulative)
        cumulative += count
    return np.concatenate(parts).astype(np.int32)


def _concat_device_offset_arrays(
    offset_arrays: list[DeviceArray],
    element_counts: list[int],
) -> DeviceArray:
    """Concatenate offset arrays on device, shifting each by cumulative counts.

    CuPy equivalent of :func:`_concat_offset_arrays`.  Each offset array has
    length ``(row_count + 1)``.  The result drops the leading zero from all
    arrays after the first and shifts values so they form a single contiguous
    offset array.  All work stays on device.
    """
    if len(offset_arrays) == 1:
        return offset_arrays[0]
    parts: list[DeviceArray] = [offset_arrays[0]]
    cumulative = element_counts[0]
    for offsets, count in zip(offset_arrays[1:], element_counts[1:], strict=True):
        # Drop the leading 0 from subsequent arrays and shift.
        parts.append(offsets[1:] + cumulative)
        cumulative += count
    return cp.concatenate(parts).astype(cp.int32)


def _concat_device_family_buffers(
    family: GeometryFamily,
    buffers: list[DeviceFamilyGeometryBuffer],
) -> DeviceFamilyGeometryBuffer:
    """Concatenate multiple DeviceFamilyGeometryBuffers on device.

    CuPy equivalent of :func:`_concat_family_buffers`.  Coordinates are
    concatenated and offset arrays are shifted so the result is a single
    contiguous device buffer.  All work stays on GPU -- no D->H transfer.
    """
    if len(buffers) == 1:
        return buffers[0]

    total_rows = sum(int(b.geometry_offsets.size) - 1 for b in buffers if b.geometry_offsets.size > 0)
    if total_rows == 0:
        return buffers[0]

    new_x = cp.concatenate([b.x for b in buffers]) if any(b.x.size for b in buffers) else cp.empty(0, dtype=cp.float64)
    new_y = cp.concatenate([b.y for b in buffers]) if any(b.y.size for b in buffers) else cp.empty(0, dtype=cp.float64)
    new_empty_mask = cp.concatenate([b.empty_mask for b in buffers])

    # Bounds: concatenate if all have bounds, otherwise drop.
    if all(b.bounds is not None for b in buffers):
        new_bounds = cp.concatenate([b.bounds for b in buffers])
    else:
        new_bounds = None

    if family in (GeometryFamily.POINT, GeometryFamily.LINESTRING, GeometryFamily.MULTIPOINT):
        # Single level of offsets: geometry_offsets -> coords
        coord_counts = [int(b.geometry_offsets[-1]) if b.geometry_offsets.size > 0 else 0 for b in buffers]
        new_geom_offsets = _concat_device_offset_arrays(
            [b.geometry_offsets for b in buffers],
            coord_counts,
        )
        return DeviceFamilyGeometryBuffer(
            family=family,
            x=new_x,
            y=new_y,
            geometry_offsets=new_geom_offsets,
            empty_mask=new_empty_mask,
            bounds=new_bounds,
        )

    if family is GeometryFamily.POLYGON:
        # Two levels: geometry_offsets -> ring_offsets -> coords
        ring_counts = [int(b.ring_offsets[-1]) if b.ring_offsets.size > 0 else 0 for b in buffers]
        geom_ring_counts = [int(b.geometry_offsets[-1]) if b.geometry_offsets.size > 0 else 0 for b in buffers]
        new_ring_offsets = _concat_device_offset_arrays(
            [b.ring_offsets for b in buffers],
            ring_counts,
        )
        new_geom_offsets = _concat_device_offset_arrays(
            [b.geometry_offsets for b in buffers],
            geom_ring_counts,
        )
        return DeviceFamilyGeometryBuffer(
            family=family,
            x=new_x,
            y=new_y,
            geometry_offsets=new_geom_offsets,
            empty_mask=new_empty_mask,
            ring_offsets=new_ring_offsets,
            bounds=new_bounds,
        )

    if family is GeometryFamily.MULTILINESTRING:
        # Two levels: geometry_offsets -> part_offsets -> coords
        coord_counts = [int(b.part_offsets[-1]) if b.part_offsets.size > 0 else 0 for b in buffers]
        geom_part_counts = [int(b.geometry_offsets[-1]) if b.geometry_offsets.size > 0 else 0 for b in buffers]
        new_part_offsets = _concat_device_offset_arrays(
            [b.part_offsets for b in buffers],
            coord_counts,
        )
        new_geom_offsets = _concat_device_offset_arrays(
            [b.geometry_offsets for b in buffers],
            geom_part_counts,
        )
        return DeviceFamilyGeometryBuffer(
            family=family,
            x=new_x,
            y=new_y,
            geometry_offsets=new_geom_offsets,
            empty_mask=new_empty_mask,
            part_offsets=new_part_offsets,
            bounds=new_bounds,
        )

    if family is GeometryFamily.MULTIPOLYGON:
        # Three levels: geometry_offsets -> part_offsets -> ring_offsets -> coords
        ring_coord_counts = [int(b.ring_offsets[-1]) if b.ring_offsets.size > 0 else 0 for b in buffers]
        part_ring_counts = [int(b.part_offsets[-1]) if b.part_offsets.size > 0 else 0 for b in buffers]
        geom_part_counts = [int(b.geometry_offsets[-1]) if b.geometry_offsets.size > 0 else 0 for b in buffers]
        new_ring_offsets = _concat_device_offset_arrays(
            [b.ring_offsets for b in buffers],
            ring_coord_counts,
        )
        new_part_offsets = _concat_device_offset_arrays(
            [b.part_offsets for b in buffers],
            part_ring_counts,
        )
        new_geom_offsets = _concat_device_offset_arrays(
            [b.geometry_offsets for b in buffers],
            geom_part_counts,
        )
        return DeviceFamilyGeometryBuffer(
            family=family,
            x=new_x,
            y=new_y,
            geometry_offsets=new_geom_offsets,
            empty_mask=new_empty_mask,
            part_offsets=new_part_offsets,
            ring_offsets=new_ring_offsets,
            bounds=new_bounds,
        )

    raise NotImplementedError(f"device concat not implemented for {family.value}")


def _concat_family_buffers(
    family: GeometryFamily,
    buffers: list[FamilyGeometryBuffer],
) -> FamilyGeometryBuffer:
    """Concatenate multiple FamilyGeometryBuffers of the same family.

    Coordinates are appended and offset arrays are shifted so that
    the result is a single contiguous buffer.  No Shapely round-trip.
    """
    if len(buffers) == 1:
        return buffers[0]

    schema = buffers[0].schema
    total_rows = sum(b.row_count for b in buffers)

    new_x = np.concatenate([b.x for b in buffers]) if total_rows > 0 else np.empty(0, dtype=np.float64)
    new_y = np.concatenate([b.y for b in buffers]) if total_rows > 0 else np.empty(0, dtype=np.float64)
    new_empty_mask = np.concatenate([b.empty_mask for b in buffers])

    # Bounds: concatenate if all have bounds, otherwise drop.
    if all(b.bounds is not None for b in buffers):
        new_bounds = np.concatenate([b.bounds for b in buffers])
    else:
        new_bounds = None

    if family in (GeometryFamily.POINT, GeometryFamily.LINESTRING, GeometryFamily.MULTIPOINT):
        # Single level of offsets: geometry_offsets -> coords
        coord_counts = [int(b.geometry_offsets[-1]) if b.geometry_offsets.size > 0 else 0 for b in buffers]
        new_geom_offsets = _concat_offset_arrays(
            [b.geometry_offsets for b in buffers],
            coord_counts,
        )
        return FamilyGeometryBuffer(
            family=family,
            schema=schema,
            row_count=total_rows,
            x=new_x,
            y=new_y,
            geometry_offsets=new_geom_offsets,
            empty_mask=new_empty_mask,
            bounds=new_bounds,
        )

    if family is GeometryFamily.POLYGON:
        # Two levels: geometry_offsets -> ring_offsets -> coords
        ring_counts = [int(b.ring_offsets[-1]) if b.ring_offsets.size > 0 else 0 for b in buffers]
        geom_ring_counts = [int(b.geometry_offsets[-1]) if b.geometry_offsets.size > 0 else 0 for b in buffers]
        new_ring_offsets = _concat_offset_arrays(
            [b.ring_offsets for b in buffers],
            ring_counts,
        )
        new_geom_offsets = _concat_offset_arrays(
            [b.geometry_offsets for b in buffers],
            geom_ring_counts,
        )
        return FamilyGeometryBuffer(
            family=family,
            schema=schema,
            row_count=total_rows,
            x=new_x,
            y=new_y,
            geometry_offsets=new_geom_offsets,
            empty_mask=new_empty_mask,
            ring_offsets=new_ring_offsets,
            bounds=new_bounds,
        )

    if family is GeometryFamily.MULTILINESTRING:
        # Two levels: geometry_offsets -> part_offsets -> coords
        coord_counts = [int(b.part_offsets[-1]) if b.part_offsets.size > 0 else 0 for b in buffers]
        geom_part_counts = [int(b.geometry_offsets[-1]) if b.geometry_offsets.size > 0 else 0 for b in buffers]
        new_part_offsets = _concat_offset_arrays(
            [b.part_offsets for b in buffers],
            coord_counts,
        )
        new_geom_offsets = _concat_offset_arrays(
            [b.geometry_offsets for b in buffers],
            geom_part_counts,
        )
        return FamilyGeometryBuffer(
            family=family,
            schema=schema,
            row_count=total_rows,
            x=new_x,
            y=new_y,
            geometry_offsets=new_geom_offsets,
            empty_mask=new_empty_mask,
            part_offsets=new_part_offsets,
            bounds=new_bounds,
        )

    if family is GeometryFamily.MULTIPOLYGON:
        # Three levels: geometry_offsets -> part_offsets -> ring_offsets -> coords
        ring_coord_counts = [int(b.ring_offsets[-1]) if b.ring_offsets.size > 0 else 0 for b in buffers]
        part_ring_counts = [int(b.part_offsets[-1]) if b.part_offsets.size > 0 else 0 for b in buffers]
        geom_part_counts = [int(b.geometry_offsets[-1]) if b.geometry_offsets.size > 0 else 0 for b in buffers]
        new_ring_offsets = _concat_offset_arrays(
            [b.ring_offsets for b in buffers],
            ring_coord_counts,
        )
        new_part_offsets = _concat_offset_arrays(
            [b.part_offsets for b in buffers],
            part_ring_counts,
        )
        new_geom_offsets = _concat_offset_arrays(
            [b.geometry_offsets for b in buffers],
            geom_part_counts,
        )
        return FamilyGeometryBuffer(
            family=family,
            schema=schema,
            row_count=total_rows,
            x=new_x,
            y=new_y,
            geometry_offsets=new_geom_offsets,
            empty_mask=new_empty_mask,
            part_offsets=new_part_offsets,
            ring_offsets=new_ring_offsets,
            bounds=new_bounds,
        )

    raise NotImplementedError(f"concat not implemented for {family.value}")


def _device_gather_offset_slices(
    data: DeviceArray,
    offsets: DeviceArray,
    rows: DeviceArray,
) -> tuple[DeviceArray, DeviceArray]:
    """Device-side gather of offset-indexed slices using CuPy.

    GPU equivalent of :func:`_gather_offset_slices` — replaces the serial
    Python for-loop with vectorized CuPy operations (fancy indexing +
    cumsum expand pattern).  No Python-level iteration over rows.

    Returns ``(gathered_data, new_offsets)`` where *new_offsets* has length
    ``rows.size + 1``.
    """
    starts = offsets[rows]
    ends = offsets[rows + 1]
    lengths = (ends - starts).astype(cp.int32)

    n = int(rows.size)
    new_offsets = cp.empty(n + 1, dtype=cp.int32)
    new_offsets[0] = 0
    if n > 0:
        cp.cumsum(lengths, out=new_offsets[1:])
    total_length = int(new_offsets[-1]) if n > 0 else 0

    if total_length == 0:
        if data.ndim == 2:
            gathered = cp.empty((0, data.shape[1]), dtype=data.dtype)
        else:
            gathered = cp.empty(0, dtype=data.dtype)
        return gathered, new_offsets

    # Build flat gather indices fully on device using searchsorted.
    # segment_ids[j] = i means position j belongs to row i.
    # searchsorted(new_offsets, j, side='right') - 1 gives the segment.
    positions = cp.arange(total_length, dtype=cp.int64)
    segment_ids = (
        cp.searchsorted(new_offsets, positions, side="right").astype(cp.int64) - 1
    )
    flat_indices = (
        positions
        - new_offsets[segment_ids].astype(cp.int64)
        + starts[segment_ids].astype(cp.int64)
    )

    gathered = data[flat_indices]
    return gathered, new_offsets


def _device_take_family_buffer(
    device_buffer: DeviceFamilyGeometryBuffer,
    family: GeometryFamily,
    family_rows: DeviceArray,
) -> DeviceFamilyGeometryBuffer:
    """Device-side extract of *family_rows* from a DeviceFamilyGeometryBuffer.

    GPU equivalent of :func:`_take_family_buffer` — all offset gathering uses
    :func:`_device_gather_offset_slices` instead of the serial host loop.
    """
    new_empty_mask = device_buffer.empty_mask[family_rows]
    new_bounds = device_buffer.bounds[family_rows] if device_buffer.bounds is not None else None

    if family in (GeometryFamily.POINT, GeometryFamily.LINESTRING, GeometryFamily.MULTIPOINT):
        coords = (
            cp.column_stack([device_buffer.x, device_buffer.y])
            if device_buffer.x.size
            else cp.empty((0, 2), dtype=cp.float64)
        )
        gathered, new_geom_offsets = _device_gather_offset_slices(
            coords, device_buffer.geometry_offsets, family_rows,
        )
        new_x = gathered[:, 0].copy() if gathered.size else cp.empty(0, dtype=cp.float64)
        new_y = gathered[:, 1].copy() if gathered.size else cp.empty(0, dtype=cp.float64)
        return DeviceFamilyGeometryBuffer(
            family=family, x=new_x, y=new_y,
            geometry_offsets=new_geom_offsets, empty_mask=new_empty_mask,
            bounds=new_bounds,
        )

    if family is GeometryFamily.POLYGON:
        ring_space = cp.arange(device_buffer.ring_offsets.size, dtype=cp.int32)
        rings, new_geom_offsets = _device_gather_offset_slices(
            ring_space, device_buffer.geometry_offsets, family_rows,
        )
        ring_indices = rings.astype(cp.int64)
        coords = (
            cp.column_stack([device_buffer.x, device_buffer.y])
            if device_buffer.x.size
            else cp.empty((0, 2), dtype=cp.float64)
        )
        gathered, new_ring_offsets = _device_gather_offset_slices(
            coords, device_buffer.ring_offsets, ring_indices,
        )
        new_x = gathered[:, 0].copy() if gathered.size else cp.empty(0, dtype=cp.float64)
        new_y = gathered[:, 1].copy() if gathered.size else cp.empty(0, dtype=cp.float64)
        return DeviceFamilyGeometryBuffer(
            family=family, x=new_x, y=new_y,
            geometry_offsets=new_geom_offsets, empty_mask=new_empty_mask,
            ring_offsets=new_ring_offsets, bounds=new_bounds,
        )

    if family is GeometryFamily.MULTILINESTRING:
        part_space = cp.arange(device_buffer.part_offsets.size, dtype=cp.int32)
        parts, new_geom_offsets = _device_gather_offset_slices(
            part_space, device_buffer.geometry_offsets, family_rows,
        )
        part_indices = parts.astype(cp.int64)
        coords = (
            cp.column_stack([device_buffer.x, device_buffer.y])
            if device_buffer.x.size
            else cp.empty((0, 2), dtype=cp.float64)
        )
        gathered, new_part_offsets = _device_gather_offset_slices(
            coords, device_buffer.part_offsets, part_indices,
        )
        new_x = gathered[:, 0].copy() if gathered.size else cp.empty(0, dtype=cp.float64)
        new_y = gathered[:, 1].copy() if gathered.size else cp.empty(0, dtype=cp.float64)
        return DeviceFamilyGeometryBuffer(
            family=family, x=new_x, y=new_y,
            geometry_offsets=new_geom_offsets, empty_mask=new_empty_mask,
            part_offsets=new_part_offsets, bounds=new_bounds,
        )

    if family is GeometryFamily.MULTIPOLYGON:
        part_space = cp.arange(device_buffer.part_offsets.size, dtype=cp.int32)
        parts, new_geom_offsets = _device_gather_offset_slices(
            part_space, device_buffer.geometry_offsets, family_rows,
        )
        part_indices = parts.astype(cp.int64)
        ring_space = cp.arange(device_buffer.ring_offsets.size, dtype=cp.int32)
        rings, new_part_offsets = _device_gather_offset_slices(
            ring_space, device_buffer.part_offsets, part_indices,
        )
        ring_indices = rings.astype(cp.int64)
        coords = (
            cp.column_stack([device_buffer.x, device_buffer.y])
            if device_buffer.x.size
            else cp.empty((0, 2), dtype=cp.float64)
        )
        gathered, new_ring_offsets = _device_gather_offset_slices(
            coords, device_buffer.ring_offsets, ring_indices,
        )
        new_x = gathered[:, 0].copy() if gathered.size else cp.empty(0, dtype=cp.float64)
        new_y = gathered[:, 1].copy() if gathered.size else cp.empty(0, dtype=cp.float64)
        return DeviceFamilyGeometryBuffer(
            family=family, x=new_x, y=new_y,
            geometry_offsets=new_geom_offsets, empty_mask=new_empty_mask,
            part_offsets=new_part_offsets, ring_offsets=new_ring_offsets,
            bounds=new_bounds,
        )

    raise NotImplementedError(f"device take not implemented for {family.value}")


def normalize_buffer_sharing_mode(mode: BufferSharingMode | str) -> BufferSharingMode:
    return mode if isinstance(mode, BufferSharingMode) else BufferSharingMode(mode)


def _is_shareable_vector(values: np.ndarray, *, dtype: np.dtype[Any]) -> bool:
    return values.dtype == dtype and values.ndim == 1 and bool(values.flags.c_contiguous)


def _is_shareable_bounds(values: np.ndarray) -> bool:
    return values.dtype == np.float64 and values.ndim == 2 and values.shape[1] == 4 and bool(values.flags.c_contiguous)


def _adopt_vector(
    values: np.ndarray,
    *,
    dtype: np.dtype[Any],
    sharing: BufferSharingMode,
) -> tuple[np.ndarray, bool]:
    array = np.asarray(values)
    if sharing is BufferSharingMode.SHARE:
        if not _is_shareable_vector(array, dtype=dtype):
            raise ValueError(f"GeoArrow buffer is not shareable as {dtype}")
        return array, True
    if sharing is BufferSharingMode.AUTO and _is_shareable_vector(array, dtype=dtype):
        return array, True
    if sharing is BufferSharingMode.COPY:
        normalized = np.array(array, dtype=dtype, copy=True, order="C")
    else:
        normalized = np.ascontiguousarray(array, dtype=dtype)
    return normalized, False


def _adopt_bounds(
    values: np.ndarray | None,
    *,
    sharing: BufferSharingMode,
) -> tuple[np.ndarray | None, bool]:
    if values is None:
        return None, True
    array = np.asarray(values)
    if sharing is BufferSharingMode.SHARE:
        if not _is_shareable_bounds(array):
            raise ValueError("GeoArrow bounds buffer is not shareable as float64[4]")
        return array, True
    if sharing is BufferSharingMode.AUTO and _is_shareable_bounds(array):
        return array, True
    if sharing is BufferSharingMode.COPY:
        normalized = np.array(array, dtype=np.float64, copy=True, order="C")
    else:
        normalized = np.ascontiguousarray(array, dtype=np.float64)
    if normalized.ndim != 2 or normalized.shape[1] != 4:
        raise ValueError("GeoArrow bounds buffer must have shape (n, 4)")
    return normalized, False


def _iter_coords(linear: Any) -> list[tuple[float, float]]:
    return [(float(coord[0]), float(coord[1])) for coord in linear.coords]


def _family_for_geometry(geometry: object) -> GeometryFamily:
    geom_type = geometry.geom_type
    mapping = {
        "Point": GeometryFamily.POINT,
        "LineString": GeometryFamily.LINESTRING,
        "Polygon": GeometryFamily.POLYGON,
        "MultiPoint": GeometryFamily.MULTIPOINT,
        "MultiLineString": GeometryFamily.MULTILINESTRING,
        "MultiPolygon": GeometryFamily.MULTIPOLYGON,
    }
    try:
        return mapping[geom_type]
    except KeyError as exc:
        raise NotImplementedError(f"unsupported geometry family: {geom_type}") from exc


def _append_family_geometry(
    family: GeometryFamily,
    geometry: object,
    state: dict[str, Any],
) -> None:
    state["row_count"] += 1
    empty = bool(geometry.is_empty)
    state["empty_mask"].append(empty)

    if family is GeometryFamily.POINT:
        state["geometry_offsets"].append(len(state["geometry_offsets_payload"]))
        if not empty:
            state["geometry_offsets_payload"].append((float(geometry.x), float(geometry.y)))
        return

    if family in {GeometryFamily.LINESTRING, GeometryFamily.MULTIPOINT}:
        state["geometry_offsets"].append(len(state["geometry_offsets_payload"]))
        if family is GeometryFamily.LINESTRING:
            coords = _iter_coords(geometry)
        else:
            coords = [(float(point.x), float(point.y)) for point in geometry.geoms]
        state["geometry_offsets_payload"].extend(coords)
        return

    if family is GeometryFamily.POLYGON:
        state["geometry_offsets"].append(len(state["ring_offsets"]))
        if not empty:
            rings = [geometry.exterior, *geometry.interiors]
            for ring in rings:
                state["ring_offsets"].append(len(state["geometry_offsets_payload"]))
                state["geometry_offsets_payload"].extend(_iter_coords(ring))
        return

    if family is GeometryFamily.MULTILINESTRING:
        state["geometry_offsets"].append(len(state["part_offsets"]))
        if not empty:
            for part in geometry.geoms:
                state["part_offsets"].append(len(state["geometry_offsets_payload"]))
                state["geometry_offsets_payload"].extend(_iter_coords(part))
        return

    if family is GeometryFamily.MULTIPOLYGON:
        state["geometry_offsets"].append(len(state["part_offsets"]))
        if not empty:
            for polygon in geometry.geoms:
                state["part_offsets"].append(len(state["ring_offsets_payload"]))
                rings = [polygon.exterior, *polygon.interiors]
                for ring in rings:
                    state["ring_offsets_payload"].append(len(state["geometry_offsets_payload"]))
                    state["geometry_offsets_payload"].extend(_iter_coords(ring))
        return

    raise NotImplementedError(f"unsupported geometry family: {family.value}")


def _finalize_family_buffer(family: GeometryFamily, state: dict[str, Any]) -> FamilyGeometryBuffer:
    coords = state["geometry_offsets_payload"]
    if coords:
        x = np.asarray([coord[0] for coord in coords], dtype=np.float64)
        y = np.asarray([coord[1] for coord in coords], dtype=np.float64)
    else:
        x = np.asarray([], dtype=np.float64)
        y = np.asarray([], dtype=np.float64)

    geometry_offsets = np.asarray(
        [*state["geometry_offsets"], len(state["geometry_offsets_payload"])],
        dtype=np.int32,
    )
    part_offsets = None
    ring_offsets = None

    if family is GeometryFamily.POLYGON:
        ring_offsets = np.asarray(
            [*state["ring_offsets"], len(state["geometry_offsets_payload"])],
            dtype=np.int32,
        )
        geometry_offsets = np.asarray(
            [*state["geometry_offsets"], len(state["ring_offsets"])],
            dtype=np.int32,
        )
    elif family is GeometryFamily.MULTILINESTRING:
        part_offsets = np.asarray(
            [*state["part_offsets"], len(state["geometry_offsets_payload"])],
            dtype=np.int32,
        )
        geometry_offsets = np.asarray(
            [*state["geometry_offsets"], len(state["part_offsets"])],
            dtype=np.int32,
        )
    elif family is GeometryFamily.MULTIPOLYGON:
        part_offsets = np.asarray(
            [*state["part_offsets"], len(state["ring_offsets_payload"])],
            dtype=np.int32,
        )
        ring_offsets = np.asarray(
            [*state["ring_offsets_payload"], len(state["geometry_offsets_payload"])],
            dtype=np.int32,
        )
        geometry_offsets = np.asarray(
            [*state["geometry_offsets"], len(state["part_offsets"])],
            dtype=np.int32,
        )

    return FamilyGeometryBuffer(
        family=family,
        schema=get_geometry_buffer_schema(family),
        row_count=int(state["row_count"]),
        x=x,
        y=y,
        geometry_offsets=geometry_offsets,
        empty_mask=np.asarray(state["empty_mask"], dtype=bool),
        part_offsets=part_offsets,
        ring_offsets=ring_offsets,
    )


def _from_shapely_vectorized(
    geom_arr: np.ndarray,
    *,
    residency: Residency = Residency.HOST,
) -> OwnedGeometryArray | None:
    """Fast vectorized path for building OwnedGeometryArray from Shapely.

    Uses ``shapely.get_coordinates``, ``shapely.get_rings``, and vectorized
    offset arithmetic instead of per-geometry Python iteration.  Returns
    ``None`` when the input contains geometry types that this fast path does
    not handle, so the caller can fall through to the scalar loop.

    Currently supports homogeneous arrays of:
    - Point (type_id 0)
    - LineString (type_id 1)
    - Polygon (type_id 3)
    - Mixed arrays containing only the above three types
    """
    n = len(geom_arr)
    if n == 0:
        return None

    validity = ~shapely.is_missing(geom_arr)
    valid_mask = validity.copy()
    # Also mark empty as valid-but-empty (is_missing covers None)
    valid_geoms = geom_arr[valid_mask]
    if len(valid_geoms) == 0:
        # All null -- fall through to scalar path for simplicity
        return None

    type_ids = shapely.get_type_id(valid_geoms)
    unique_types = np.unique(type_ids)

    # Only handle Point(0), LineString(1), Polygon(3)
    _SUPPORTED_TYPE_IDS = np.array([0, 1, 3])
    if not np.isin(unique_types, _SUPPORTED_TYPE_IDS).all():
        return None

    empty_flags = shapely.is_empty(valid_geoms)
    tags = np.full(n, NULL_TAG, dtype=np.int8)
    family_row_offsets = np.full(n, -1, dtype=np.int32)
    families: dict[GeometryFamily, FamilyGeometryBuffer] = {}

    # Map type_id -> family
    _type_to_family = {
        0: GeometryFamily.POINT,
        1: GeometryFamily.LINESTRING,
        3: GeometryFamily.POLYGON,
    }

    valid_indices = np.flatnonzero(valid_mask)

    for tid in sorted(unique_types):
        family = _type_to_family[int(tid)]
        fam_local_mask = type_ids == tid
        fam_global_indices = valid_indices[fam_local_mask]
        fam_geoms = valid_geoms[fam_local_mask]
        fam_empty = empty_flags[fam_local_mask]
        fam_count = len(fam_geoms)
        fam_non_empty = fam_geoms[~fam_empty]

        tags[fam_global_indices] = FAMILY_TAGS[family]
        family_row_offsets[fam_global_indices] = np.arange(fam_count, dtype=np.int32)

        if family is GeometryFamily.POINT:
            if len(fam_non_empty) > 0:
                coords = shapely.get_coordinates(fam_non_empty)
                x = coords[:, 0].astype(np.float64, copy=True)
                y = coords[:, 1].astype(np.float64, copy=True)
            else:
                x = np.empty(0, dtype=np.float64)
                y = np.empty(0, dtype=np.float64)
            # Points: geometry_offsets[i] = index into coord array
            # Each non-empty point contributes 1 coord
            go = np.zeros(fam_count + 1, dtype=np.int32)
            go[1:] = np.cumsum(~fam_empty)
            families[family] = FamilyGeometryBuffer(
                family=family,
                schema=get_geometry_buffer_schema(family),
                row_count=fam_count,
                x=x, y=y,
                geometry_offsets=go,
                empty_mask=fam_empty.copy(),
            )

        elif family is GeometryFamily.LINESTRING:
            if len(fam_non_empty) > 0:
                coords = shapely.get_coordinates(fam_non_empty)
                x = coords[:, 0].astype(np.float64, copy=True)
                y = coords[:, 1].astype(np.float64, copy=True)
                num_coords = shapely.get_num_coordinates(fam_non_empty)
            else:
                x = np.empty(0, dtype=np.float64)
                y = np.empty(0, dtype=np.float64)
                num_coords = np.empty(0, dtype=np.int32)
            # geometry_offsets: start index in coord array per geometry
            # Empty geometries contribute 0 coords
            all_num_coords = np.zeros(fam_count, dtype=np.int32)
            all_num_coords[~fam_empty] = num_coords
            go = np.zeros(fam_count + 1, dtype=np.int32)
            np.cumsum(all_num_coords, out=go[1:])
            families[family] = FamilyGeometryBuffer(
                family=family,
                schema=get_geometry_buffer_schema(family),
                row_count=fam_count,
                x=x, y=y,
                geometry_offsets=go,
                empty_mask=fam_empty.copy(),
            )

        elif family is GeometryFamily.POLYGON:
            if len(fam_non_empty) > 0:
                # Get all rings with parent geometry index
                rings, ring_parents = shapely.get_rings(fam_non_empty, return_index=True)
                ring_coords = shapely.get_coordinates(rings)
                x = ring_coords[:, 0].astype(np.float64, copy=True)
                y = ring_coords[:, 1].astype(np.float64, copy=True)
                ring_num_coords = shapely.get_num_coordinates(rings)

                # ring_offsets: start index in coord array per ring
                ro = np.zeros(len(rings) + 1, dtype=np.int32)
                np.cumsum(ring_num_coords, out=ro[1:])

                # Rings per non-empty geometry
                rings_per_ne = np.bincount(ring_parents, minlength=len(fam_non_empty)).astype(np.int32)
            else:
                x = np.empty(0, dtype=np.float64)
                y = np.empty(0, dtype=np.float64)
                ro = np.zeros(1, dtype=np.int32)
                rings_per_ne = np.empty(0, dtype=np.int32)

            # geometry_offsets: index into ring_offsets per geometry
            # Empty polygons contribute 0 rings
            all_rings_per = np.zeros(fam_count, dtype=np.int32)
            all_rings_per[~fam_empty] = rings_per_ne
            go = np.zeros(fam_count + 1, dtype=np.int32)
            np.cumsum(all_rings_per, out=go[1:])
            families[family] = FamilyGeometryBuffer(
                family=family,
                schema=get_geometry_buffer_schema(family),
                row_count=fam_count,
                x=x, y=y,
                geometry_offsets=go,
                empty_mask=fam_empty.copy(),
                ring_offsets=ro,
            )

    array = OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families=families,
        residency=Residency.HOST,
    )
    array._record(
        DiagnosticKind.CREATED,
        "created owned geometry array from shapely input (vectorized fast path)",
        visible=True,
    )
    if residency is Residency.DEVICE:
        array.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="created owned geometry array with device residency requested",
        )
    return array


def from_shapely_geometries(
    geometries: list[object | None] | tuple[object | None, ...],
    *,
    residency: Residency = Residency.HOST,
) -> OwnedGeometryArray:
    # Try vectorized fast path first (50x faster for common types)
    geom_arr = np.empty(len(geometries), dtype=object)
    geom_arr[:] = geometries
    result = _from_shapely_vectorized(geom_arr, residency=residency)
    if result is not None:
        return result

    # Scalar fallback for multi-types and mixed geometry collections
    validity = np.asarray([geometry is not None for geometry in geometries], dtype=bool)
    tags = np.full(len(geometries), NULL_TAG, dtype=np.int8)
    family_row_offsets = np.full(len(geometries), -1, dtype=np.int32)
    states: dict[GeometryFamily, dict[str, Any]] = {
        family: {
            "row_count": 0,
            "empty_mask": [],
            "geometry_offsets": [],
            "geometry_offsets_payload": [],
            "part_offsets": [],
            "ring_offsets": [],
            "ring_offsets_payload": [],
        }
        for family in GEOMETRY_BUFFER_SCHEMAS
    }

    for row_index, geometry in enumerate(geometries):
        if geometry is None:
            continue
        family = _family_for_geometry(geometry)
        family_state = states[family]
        family_row_offsets[row_index] = int(family_state["row_count"])
        tags[row_index] = FAMILY_TAGS[family]
        _append_family_geometry(family, geometry, family_state)

    families = {
        family: _finalize_family_buffer(family, state)
        for family, state in states.items()
        if state["row_count"] > 0
    }
    array = OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families=families,
        residency=Residency.HOST,
    )
    array._record(DiagnosticKind.CREATED, "created owned geometry array from shapely input", visible=True)
    if residency is Residency.DEVICE:
        array.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="created owned geometry array with device residency requested",
        )
    return array


def from_wkb(
    values: list[bytes | str | None] | tuple[bytes | str | None, ...],
    *,
    on_invalid: str = "raise",
    residency: Residency = Residency.HOST,
) -> OwnedGeometryArray:
    geometries: list[object | None] = []
    for value in values:
        if value is None:
            geometries.append(None)
            continue
        try:
            geometries.append(shapely.from_wkb(value, on_invalid=on_invalid))
        except Exception:
            if on_invalid == "ignore":
                geometries.append(None)
                continue
            raise
    array = from_shapely_geometries(geometries, residency=residency)
    array._record(DiagnosticKind.CREATED, "created owned geometry array from WKB input", visible=True)
    return array


def from_geoarrow(
    view: MixedGeoArrowView,
    *,
    residency: Residency = Residency.HOST,
    sharing: BufferSharingMode | str = BufferSharingMode.COPY,
) -> OwnedGeometryArray:
    sharing_mode = normalize_buffer_sharing_mode(sharing)
    families: dict[GeometryFamily, FamilyGeometryBuffer] = {}
    share_results: list[bool] = []
    for family, buffer in view.families.items():
        x, x_shared = _adopt_vector(buffer.x, dtype=np.float64, sharing=sharing_mode)
        y, y_shared = _adopt_vector(buffer.y, dtype=np.float64, sharing=sharing_mode)
        geometry_offsets, geometry_shared = _adopt_vector(
            buffer.geometry_offsets,
            dtype=np.int32,
            sharing=sharing_mode,
        )
        empty_mask, empty_shared = _adopt_vector(buffer.empty_mask, dtype=np.bool_, sharing=sharing_mode)
        if buffer.part_offsets is None:
            part_offsets = None
            part_shared = True
        else:
            part_offsets, part_shared = _adopt_vector(buffer.part_offsets, dtype=np.int32, sharing=sharing_mode)
        if buffer.ring_offsets is None:
            ring_offsets = None
            ring_shared = True
        else:
            ring_offsets, ring_shared = _adopt_vector(buffer.ring_offsets, dtype=np.int32, sharing=sharing_mode)
        bounds, bounds_shared = _adopt_bounds(buffer.bounds, sharing=sharing_mode)
        share_results.extend(
            [
                x_shared,
                y_shared,
                geometry_shared,
                empty_shared,
                part_shared,
                ring_shared,
                bounds_shared,
            ]
        )
        families[family] = FamilyGeometryBuffer(
            family=family,
            schema=get_geometry_buffer_schema(family),
            row_count=int(empty_mask.size),
            x=x,
            y=y,
            geometry_offsets=geometry_offsets,
            empty_mask=empty_mask,
            part_offsets=part_offsets,
            ring_offsets=ring_offsets,
            bounds=bounds,
        )
    validity, validity_shared = _adopt_vector(view.validity, dtype=np.bool_, sharing=sharing_mode)
    tags, tags_shared = _adopt_vector(view.tags, dtype=np.int8, sharing=sharing_mode)
    family_row_offsets, offsets_shared = _adopt_vector(view.family_row_offsets, dtype=np.int32, sharing=sharing_mode)
    shares_memory = all([*share_results, validity_shared, tags_shared, offsets_shared])
    array = OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families=families,
        residency=Residency.HOST,
        geoarrow_backed=True,
        shares_geoarrow_memory=shares_memory,
    )
    detail = (
        "created owned geometry array from shared GeoArrow-style buffers"
        if shares_memory
        else "created owned geometry array from normalized GeoArrow-style buffers"
    )
    array._record(DiagnosticKind.CREATED, detail, visible=True)
    if residency is Residency.DEVICE:
        array.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="created owned geometry array with device residency requested",
        )
    return array


def _materialize_family_row(buffer: FamilyGeometryBuffer, row_index: int) -> object:
    if bool(buffer.empty_mask[row_index]):
        return {
            GeometryFamily.POINT: Point(),
            GeometryFamily.LINESTRING: LineString(),
            GeometryFamily.POLYGON: Polygon(),
            GeometryFamily.MULTIPOINT: MultiPoint([]),
            GeometryFamily.MULTILINESTRING: MultiLineString([]),
            GeometryFamily.MULTIPOLYGON: MultiPolygon([]),
        }[buffer.family]

    if buffer.family in {GeometryFamily.POINT, GeometryFamily.LINESTRING, GeometryFamily.MULTIPOINT}:
        start = int(buffer.geometry_offsets[row_index])
        end = int(buffer.geometry_offsets[row_index + 1])
        coords = list(zip(buffer.x[start:end], buffer.y[start:end], strict=True))
        if buffer.family is GeometryFamily.POINT:
            x, y = coords[0]
            return Point(float(x), float(y))
        if buffer.family is GeometryFamily.LINESTRING:
            return LineString(coords)
        return MultiPoint(coords)

    if buffer.family is GeometryFamily.POLYGON:
        ring_start = int(buffer.geometry_offsets[row_index])
        ring_end = int(buffer.geometry_offsets[row_index + 1])
        rings = []
        for ring_index in range(ring_start, ring_end):
            coord_start = int(buffer.ring_offsets[ring_index])
            coord_end = int(buffer.ring_offsets[ring_index + 1])
            rings.append(list(zip(buffer.x[coord_start:coord_end], buffer.y[coord_start:coord_end], strict=True)))
        valid_holes = [r for r in rings[1:] if len(r) >= 3]
        return Polygon(rings[0], holes=valid_holes)

    if buffer.family is GeometryFamily.MULTILINESTRING:
        part_start = int(buffer.geometry_offsets[row_index])
        part_end = int(buffer.geometry_offsets[row_index + 1])
        parts = []
        for part_index in range(part_start, part_end):
            coord_start = int(buffer.part_offsets[part_index])
            coord_end = int(buffer.part_offsets[part_index + 1])
            parts.append(list(zip(buffer.x[coord_start:coord_end], buffer.y[coord_start:coord_end], strict=True)))
        return MultiLineString(parts)

    if buffer.family is GeometryFamily.MULTIPOLYGON:
        polygon_start = int(buffer.geometry_offsets[row_index])
        polygon_end = int(buffer.geometry_offsets[row_index + 1])
        polygons = []
        for polygon_index in range(polygon_start, polygon_end):
            ring_start = int(buffer.part_offsets[polygon_index])
            ring_end = int(buffer.part_offsets[polygon_index + 1])
            rings = []
            for ring_index in range(ring_start, ring_end):
                coord_start = int(buffer.ring_offsets[ring_index])
                coord_end = int(buffer.ring_offsets[ring_index + 1])
                rings.append(
                    list(zip(buffer.x[coord_start:coord_end], buffer.y[coord_start:coord_end], strict=True))
                )
            valid_holes = [r for r in rings[1:] if len(r) >= 3]
            polygons.append(Polygon(rings[0], holes=valid_holes))
        return MultiPolygon(polygons)

    raise NotImplementedError(f"unsupported geometry family: {buffer.family.value}")


def concat_owned_scatter(
    base: OwnedGeometryArray,
    replacement: OwnedGeometryArray,
    indices: np.ndarray,
) -> OwnedGeometryArray:
    """Scatter *replacement* rows into *base* at *indices*, returning a new array.

    Returns a new OwnedGeometryArray with the same row count as *base* where:
    - rows at *indices* come from *replacement* (in order)
    - all other rows come from *base*

    ``len(indices)`` must equal ``replacement.row_count``.

    Operates entirely at the buffer level — no Shapely materialisation.
    When both inputs are host-resident the result is host-resident; when
    both are device-resident, use :func:`device_concat_owned_scatter`
    instead (future work).
    """
    from vibespatial.geometry.device_array import _concat_family_buffers

    indices = np.asarray(indices, dtype=np.int64)
    n_out = base.row_count
    if indices.size != replacement.row_count:
        raise ValueError(
            f"indices length ({indices.size}) must equal replacement row_count "
            f"({replacement.row_count})"
        )

    base._ensure_host_state()
    replacement._ensure_host_state()

    # 1. Build output metadata by copying base and overwriting at indices
    out_validity = base.validity.copy()
    out_tags = base.tags.copy()
    out_validity[indices] = replacement.validity
    out_tags[indices] = replacement.tags

    # 2. Build a boolean mask identifying which output rows come from replacement
    is_replacement = np.zeros(n_out, dtype=bool)
    is_replacement[indices] = True

    # 3. Per-family: gather rows from base and replacement, concatenate
    out_family_row_offsets = np.full(n_out, -1, dtype=np.int32)
    out_families: dict[GeometryFamily, FamilyGeometryBuffer] = {}

    # Pre-compute the inverse mapping: output position → replacement-local row.
    # inv_map[output_pos] gives the replacement row index when output_pos is a
    # replacement position; -1 otherwise.
    inv_map = np.full(n_out, -1, dtype=np.int64)
    inv_map[indices] = np.arange(replacement.row_count, dtype=np.int64)

    # Collect all families present in the output
    all_family_tags = set()
    for tag_val in np.unique(out_tags):
        if tag_val != NULL_TAG:
            all_family_tags.add(int(tag_val))

    for tag_val in sorted(all_family_tags):
        family = TAG_FAMILIES[tag_val]

        # Which output rows belong to this family?
        family_mask = out_tags == tag_val
        family_global_indices = np.flatnonzero(family_mask)

        # Split into base-sourced and replacement-sourced rows
        from_base_mask = ~is_replacement[family_global_indices]
        from_repl_mask = is_replacement[family_global_indices]

        base_global = family_global_indices[from_base_mask]
        repl_global = family_global_indices[from_repl_mask]

        bufs_to_concat: list[FamilyGeometryBuffer] = []
        base_family_count = 0
        repl_family_count = 0

        # Gather from base's family buffer
        if base_global.size > 0 and family in base.families:
            base_family_rows = base.family_row_offsets[base_global]
            base_taken = _take_family_buffer(base.families[family], base_family_rows)
            bufs_to_concat.append(base_taken)
            base_family_count = base_taken.row_count

        # Gather from replacement's family buffer
        if repl_global.size > 0 and family in replacement.families:
            repl_local = inv_map[repl_global]
            repl_family_rows = replacement.family_row_offsets[repl_local]
            repl_taken = _take_family_buffer(replacement.families[family], repl_family_rows)
            bufs_to_concat.append(repl_taken)
            repl_family_count = repl_taken.row_count

        if bufs_to_concat:
            merged = _concat_family_buffers(family, bufs_to_concat)
            out_families[family] = merged

            # Assign family_row_offsets: base rows get 0..base_count-1,
            # replacement rows get base_count..base_count+repl_count-1
            if base_global.size > 0:
                out_family_row_offsets[base_global] = np.arange(
                    base_family_count, dtype=np.int32,
                )
            if repl_global.size > 0:
                out_family_row_offsets[repl_global] = np.arange(
                    base_family_count,
                    base_family_count + repl_family_count,
                    dtype=np.int32,
                )

    result = OwnedGeometryArray(
        validity=out_validity,
        tags=out_tags,
        family_row_offsets=out_family_row_offsets,
        families=out_families,
        residency=Residency.HOST,
    )
    result._record(
        DiagnosticKind.CREATED,
        f"scatter {replacement.row_count} replacement rows into {base.row_count}-row base",
        visible=False,
    )
    return result


def build_device_resident_owned(
    *,
    device_families: dict[GeometryFamily, DeviceFamilyGeometryBuffer],
    row_count: int,
    tags: np.ndarray,
    validity: np.ndarray,
    family_row_offsets: np.ndarray,
) -> OwnedGeometryArray:
    """Construct an OwnedGeometryArray from device buffers without touching host.

    This is the canonical factory for producing device-resident results from GPU
    kernels.  Host-side FamilyGeometryBuffers are created with empty coordinate
    stubs (``host_materialized=False``); actual data lives only in the
    ``device_state``.  Lazy ``_ensure_host_state`` will copy on demand if the
    caller ever needs Shapely objects.

    Parameters
    ----------
    device_families
        Per-family device buffers produced by a GPU kernel.
    row_count
        Total number of rows (geometries) in the output.
    tags
        int8 array of family tags, length ``row_count``.
    validity
        bool array, length ``row_count``.
    family_row_offsets
        int32 array mapping global row index to family-local row index.
    """
    from vibespatial.cuda._runtime import get_cuda_runtime

    runtime = get_cuda_runtime()

    # Build host-side placeholder families with host_materialized=False
    host_families: dict[GeometryFamily, FamilyGeometryBuffer] = {}
    for family in device_families:
        schema = get_geometry_buffer_schema(family)
        host_families[family] = FamilyGeometryBuffer(
            family=family,
            schema=schema,
            row_count=int(np.sum(tags == FAMILY_TAGS[family])),
            x=np.empty(0, dtype=np.float64),
            y=np.empty(0, dtype=np.float64),
            geometry_offsets=np.empty(0, dtype=np.int32),
            empty_mask=np.empty(0, dtype=np.bool_),
            host_materialized=False,
        )

    d_validity = runtime.from_host(validity)
    d_tags = runtime.from_host(tags)
    d_family_row_offsets = runtime.from_host(family_row_offsets)

    d_meta = DeviceMetadataState(
        validity=d_validity,
        tags=d_tags,
        family_row_offsets=d_family_row_offsets,
    )

    result = OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families=host_families,
        residency=Residency.DEVICE,
        device_state=OwnedGeometryDeviceState(
            validity=d_validity,
            tags=d_tags,
            family_row_offsets=d_family_row_offsets,
            families=device_families,
        ),
        device_metadata=d_meta,
    )
    result._record(
        DiagnosticKind.CREATED,
        f"device-resident owned array, {row_count} rows, "
        f"{len(device_families)} families",
        visible=False,
    )
    return result


# ---------------------------------------------------------------------------
# Broadcast helper: tile a 1-row OwnedGeometryArray to N rows
# ---------------------------------------------------------------------------

def tile_single_row(
    owned: OwnedGeometryArray,
    n: int,
) -> OwnedGeometryArray:
    """Create an N-row owned array from a 1-row owned array.

    The coordinate buffers (x, y) and offset arrays inside each
    :class:`FamilyGeometryBuffer` are **shared** with the original -- only
    the three routing metadata arrays (``validity``, ``tags``,
    ``family_row_offsets``) are replicated.  This makes the operation O(N)
    in tiny int8/int32/bool metadata, not O(N * vertex_count) in fp64
    coordinates, eliminating the host-side materialization bottleneck for
    scalar broadcast (nsf.3/nsf.4).

    Parameters
    ----------
    owned
        Must have ``row_count == 1``.
    n
        Desired number of output rows.

    Returns
    -------
    OwnedGeometryArray
        An *n*-row array where every row references the same geometry as
        the single input row.  The family buffers have ``row_count == 1``
        and every row's ``family_row_offsets`` entry is ``0``.
    """
    if owned.row_count != 1:
        raise ValueError(
            f"tile_single_row expects a 1-row array, got {owned.row_count}"
        )
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if n == 1:
        return owned

    # Tile the three routing metadata arrays.
    validity = np.repeat(owned.validity, n)
    tags = np.repeat(owned.tags, n)
    family_row_offsets = np.repeat(owned.family_row_offsets, n)

    # When the source is device-resident, its host family buffers may be
    # un-materialised stubs (empty x/y arrays with host_materialized=False).
    # We must NOT construct an OwnedGeometryArray that claims DEVICE residency
    # without a device_state -- _ensure_device_state() would re-upload the
    # empty stubs as if they were real data, causing CUDA_ERROR_ILLEGAL_ADDRESS.
    #
    # Instead, when a device_state exists on the source, share its device-side
    # family buffers directly and upload only the small new metadata arrays.
    # This keeps coordinate data on-device (zero-copy for the expensive part)
    # and only crosses the bus once for 6 bytes/row of metadata (H->D).
    if owned.device_state is not None:
        runtime = get_cuda_runtime()
        d_validity = runtime.from_host(validity)
        d_tags = runtime.from_host(tags)
        d_fro = runtime.from_host(family_row_offsets)
        d_meta = DeviceMetadataState(
            validity=d_validity,
            tags=d_tags,
            family_row_offsets=d_fro,
        )
        d_state = OwnedGeometryDeviceState(
            validity=d_validity,
            tags=d_tags,
            family_row_offsets=d_fro,
            families=dict(owned.device_state.families),  # shared reference
        )
        result = OwnedGeometryArray(
            validity=validity,
            tags=tags,
            family_row_offsets=family_row_offsets,
            families=owned.families,  # shared reference (read-only usage)
            residency=Residency.DEVICE,
            device_state=d_state,
            device_metadata=d_meta,
        )
    else:
        result = OwnedGeometryArray(
            validity=validity,
            tags=tags,
            family_row_offsets=family_row_offsets,
            families=owned.families,  # shared reference (read-only usage)
            residency=Residency.HOST,
        )
    return result


def materialize_broadcast(tiled: OwnedGeometryArray) -> OwnedGeometryArray:
    """Physically replicate coordinate buffers in a tiled owned array.

    :func:`tile_single_row` creates an N-row metadata facade that shares
    the 1-row coordinate buffers.  GPU kernels that index into family
    buffers by global row index require ``family_buf.row_count == n``.
    This function converts the metadata-only tile into a fully-
    materialized array where each family buffer has ``n`` physical rows
    with replicated coordinate data.

    The operation is O(N * vertices_per_geometry) in coordinate copies
    but avoids per-element Python loops and the Shapely round-trip that
    would otherwise be required.  It is only called for the GPU path
    of broadcast-right constructive operations.

    Parameters
    ----------
    tiled
        An :class:`OwnedGeometryArray` produced by :func:`tile_single_row`
        (N-row metadata, 1-row family buffers with all
        ``family_row_offsets == 0``).

    Returns
    -------
    OwnedGeometryArray
        Same metadata but with physically replicated family buffers
        where ``family_buf.row_count == tiled.row_count``.
    """
    n = tiled.row_count
    tiled._ensure_host_state()

    new_families: dict[GeometryFamily, FamilyGeometryBuffer] = {}
    for family, buf in tiled.families.items():
        if buf.row_count == 0:
            new_families[family] = buf
            continue

        # How many rows in the tiled array belong to this family?
        family_tag = FAMILY_TAGS[family]
        family_n = int(np.sum(tiled.tags == family_tag))
        if family_n == 0 or buf.row_count >= family_n:
            # Already materialized or empty -- keep as-is.
            new_families[family] = buf
            continue

        # The source has 1 geometry row; replicate it family_n times.
        # Coordinate span for the single source row.
        src_geom_start = int(buf.geometry_offsets[0])
        src_geom_end = int(buf.geometry_offsets[1])
        src_n_rings = src_geom_end - src_geom_start

        if buf.ring_offsets is not None:
            src_coord_start = int(buf.ring_offsets[src_geom_start])
            src_coord_end = int(buf.ring_offsets[src_geom_end])
        else:
            src_coord_start = 0
            src_coord_end = len(buf.x)

        # Replicate coordinates: tile the single row's coords N times.
        src_x = buf.x[src_coord_start:src_coord_end]
        src_y = buf.y[src_coord_start:src_coord_end]
        new_x = np.tile(src_x, family_n)
        new_y = np.tile(src_y, family_n)

        # Build geometry_offsets: each row has src_n_rings rings.
        # [0, R, 2R, ..., family_n*R]
        new_geom_offsets = np.arange(family_n + 1, dtype=np.int32) * src_n_rings

        # Build ring_offsets (if present): each ring has the same length.
        new_ring_offsets = None
        if buf.ring_offsets is not None:
            src_ring_lens = np.diff(
                buf.ring_offsets[src_geom_start:src_geom_end + 1]
            )
            # Tile ring lengths, then cumulative sum with leading 0.
            tiled_ring_lens = np.tile(src_ring_lens, family_n)
            new_ring_offsets = np.empty(
                len(tiled_ring_lens) + 1, dtype=np.int32,
            )
            new_ring_offsets[0] = 0
            np.cumsum(tiled_ring_lens, out=new_ring_offsets[1:])

        # Build part_offsets (if present, for multi-geometries).
        new_part_offsets = None
        if buf.part_offsets is not None:
            src_n_parts = int(buf.part_offsets[1]) - int(buf.part_offsets[0])
            new_part_offsets = np.arange(
                family_n + 1, dtype=np.int32,
            ) * src_n_parts

        # Replicate empty_mask.
        new_empty_mask = np.repeat(buf.empty_mask[:1], family_n)

        new_families[family] = FamilyGeometryBuffer(
            family=family,
            schema=buf.schema,
            row_count=family_n,
            x=new_x,
            y=new_y,
            geometry_offsets=new_geom_offsets,
            empty_mask=new_empty_mask,
            part_offsets=new_part_offsets,
            ring_offsets=new_ring_offsets,
        )

    # Build new family_row_offsets that map to 0..family_n-1 (not all 0).
    # Vectorized per-tag cumulative count (VPAT001: no per-element Python loops).
    new_fro = np.full(n, -1, dtype=np.int32)
    for tag_val in np.unique(tiled.tags):
        if tag_val == NULL_TAG:
            continue
        mask = tiled.tags == tag_val
        new_fro[mask] = np.arange(int(mask.sum()), dtype=np.int32)

    return OwnedGeometryArray(
        validity=tiled.validity.copy(),
        tags=tiled.tags.copy(),
        family_row_offsets=new_fro,
        families=new_families,
        residency=Residency.HOST,
    )
