from __future__ import annotations

from dataclasses import dataclass, field
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
class OwnedGeometryDeviceState:
    validity: DeviceArray
    tags: DeviceArray
    family_row_offsets: DeviceArray
    families: dict[GeometryFamily, DeviceFamilyGeometryBuffer]
    _column_refs: list | None = None
    row_bounds: DeviceArray | None = None  # cached per-row (N, 4) fp64 bounds on device


@dataclass
class OwnedGeometryArray:
    validity: np.ndarray
    tags: np.ndarray
    family_row_offsets: np.ndarray
    families: dict[GeometryFamily, FamilyGeometryBuffer]
    residency: Residency = Residency.HOST
    diagnostics: list[DiagnosticEvent] = field(default_factory=list)
    runtime_history: list[RuntimeSelection] = field(default_factory=list)
    geoarrow_backed: bool = False
    shares_geoarrow_memory: bool = False
    device_adopted: bool = False
    device_state: OwnedGeometryDeviceState | None = None

    @property
    def row_count(self) -> int:
        return int(self.validity.size)

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
            family_bounds = bounds[row_indexes]
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
                else:
                    runtime.copy_host_to_device(family_bounds, device_buffer.bounds)

    def cache_device_bounds(self, family: GeometryFamily, bounds: DeviceArray) -> None:
        state = self._ensure_device_state()
        family_state = state.families[family]
        if family_state.bounds is not None and family_state.bounds is not bounds:
            get_cuda_runtime().free(family_state.bounds)
        family_state.bounds = bounds

    def _ensure_device_state(self) -> OwnedGeometryDeviceState:
        if self.device_state is not None:
            return self.device_state
        runtime = get_cuda_runtime()
        if not runtime.available():
            raise RuntimeError("GPU execution was requested, but no CUDA device is available")
        t0 = perf_counter()
        total_bytes = self.validity.nbytes + self.tags.nbytes + self.family_row_offsets.nbytes
        self.device_state = OwnedGeometryDeviceState(
            validity=runtime.from_host(self.validity),
            tags=runtime.from_host(self.tags),
            family_row_offsets=runtime.from_host(self.family_row_offsets),
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

    def take(self, indices: np.ndarray) -> OwnedGeometryArray:
        """Return a new OwnedGeometryArray containing only the rows at *indices*.

        Operates entirely at the buffer level -- no Shapely round-trip.
        When the array is DEVICE-resident, dispatches to :meth:`device_take`
        to keep all gathering on GPU.  Otherwise returns a HOST-resident array.
        """
        if (
            self.residency is Residency.DEVICE
            and self.device_state is not None
            and cp is not None
        ):
            return self.device_take(indices)
        self._ensure_host_state()
        if indices.dtype == bool:
            indices = np.flatnonzero(indices)
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
        """
        if cp is None:
            raise RuntimeError("CuPy is required for device_take")
        d_state = self._ensure_device_state()

        # Accept numpy or CuPy indices
        if isinstance(indices, np.ndarray):
            d_indices = cp.asarray(indices)
        else:
            d_indices = indices

        # Bool mask → integer indices
        if d_indices.dtype == cp.bool_ or d_indices.dtype == bool:
            d_indices = cp.flatnonzero(d_indices).astype(cp.int64)
        else:
            d_indices = d_indices.astype(cp.int64, copy=False)

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

        # Small D→H for routing metadata
        h_validity = cp.asnumpy(d_new_validity)
        h_tags = cp.asnumpy(d_new_tags)
        h_family_row_offsets = cp.asnumpy(d_new_family_row_offsets)

        result = OwnedGeometryArray(
            validity=h_validity,
            tags=h_tags,
            family_row_offsets=h_family_row_offsets,
            families=new_host_families,
            residency=Residency.DEVICE,
            device_state=OwnedGeometryDeviceState(
                validity=d_new_validity,
                tags=d_new_tags,
                family_row_offsets=d_new_family_row_offsets,
                families=new_device_families,
            ),
        )
        result._record(
            DiagnosticKind.CREATED,
            f"device-side subset {int(d_indices.size)} rows via device_take",
            visible=False,
        )
        return result

    def to_shapely(self) -> list[object | None]:
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
    return [(float(x), float(y)) for x, y in linear.coords]


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


def from_shapely_geometries(
    geometries: list[object | None] | tuple[object | None, ...],
    *,
    residency: Residency = Residency.HOST,
) -> OwnedGeometryArray:
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
