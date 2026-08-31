from __future__ import annotations

import inspect
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

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_I64,
    KERNEL_PARAM_PTR,
    DeviceArray,
    compile_kernel_group,
    count_scatter_total,
    get_cuda_runtime,
)
from vibespatial.kernels.owned_take import (
    _OWNED_TAKE_KERNEL_SOURCE,
    OWNED_TAKE_KERNEL_NAMES,
)
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


def _propagate_row_segment_capacity_bound(result, arrays) -> None:
    """Preserve a proven per-logical-row segment bound across composition."""
    bounds = [getattr(array, "_active_family_row_segment_capacity_bound", None) for array in arrays]
    if bounds and all(bound is not None for bound in bounds):
        result._active_family_row_segment_capacity_bound = max(int(bound) for bound in bounds)


def _owned_take_kernels():
    return compile_kernel_group(
        "owned-take",
        _OWNED_TAKE_KERNEL_SOURCE,
        OWNED_TAKE_KERNEL_NAMES,
    )


def _device_family_row_count(buffer: DeviceFamilyGeometryBuffer) -> int:
    return max(int(buffer.geometry_offsets.size) - 1, 0)


def _owned_device_bool_scalar(value, *, reason: str) -> bool:
    """Read a device scalar through the runtime so profiles see the fence."""
    if cp is not None and (
        hasattr(value, "__cuda_array_interface__") or type(value).__module__.startswith("cupy")
    ):
        d_value = cp.asarray(value).reshape(1)
        host = get_cuda_runtime().copy_device_to_host(d_value, reason=reason)
        return bool(np.asarray(host).reshape(-1)[0])
    return bool(value)


def _device_index_map_is_injective(
    index_map,
    *,
    source_row_count: int,
    active_mask=None,
) -> bool:
    """Prove injectivity for an otherwise-unannotated device row map.

    Native rowset producers should propagate their injectivity proof and never
    enter this allocation boundary.  This check is reserved for arbitrary
    device maps that are explicitly requested as contiguous physical storage.
    """
    d_index_map = cp.asarray(index_map, dtype=cp.int64)
    row_count = int(d_index_map.size)
    if row_count <= 1:
        return True
    if active_mask is None and row_count > int(source_row_count):
        return False

    d_counts = cp.zeros(int(source_row_count), dtype=cp.int32)
    if active_mask is None:
        cp.add.at(d_counts, d_index_map, cp.int32(1))
    else:
        d_active = cp.asarray(active_mask, dtype=cp.bool_)
        if d_active.ndim != 1 or int(d_active.size) != row_count:
            raise ValueError("device row-map activity must match the row-map capacity")
        cp.add.at(d_counts, d_index_map, d_active.astype(cp.int32, copy=False))
    return _owned_device_bool_scalar(
        cp.all(d_counts <= 1),
        reason="owned geometry device row-map injectivity allocation fence",
    )


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
    if cp is not None and not isinstance(left_tags, np.ndarray):
        left = left_tags.astype(cp.int16, copy=False)
        right = right_tags.astype(cp.int16, copy=False)
        if int(left.size) == 0:
            return []

        # Geometry tags occupy a tiny fixed domain (0..5).  Avoid cp.unique()
        # here: on small predicate batches it dispatches a heavyweight CUB
        # segmented-reduce path just to discover at most 36 tag pairs.
        domain = len(FAMILY_TAGS)
        packed = left * np.int16(domain) + right
        present = cp.zeros(domain * domain, dtype=cp.bool_)
        present[packed] = True
        host_present = get_cuda_runtime().copy_device_to_host(
            present,
            reason="owned geometry family-pair domain summary scalar fence",
        )
        return [
            (int(index // domain), int(index % domain)) for index in np.flatnonzero(host_present)
        ]

    packed = left_tags.astype(np.int16) * np.int16(256) + right_tags.astype(np.int16)
    unique_packed = np.unique(packed)
    return [(int(p // 256), int(p % 256)) for p in unique_packed]


def seed_all_validity_cache(owned: OwnedGeometryArray | None) -> None:
    """Seed the per-row validity cache with an all-valid mask.

    Exact overlay/clip results and successful post-repair outputs are already
    normalized geometry buffers. Marking them valid avoids re-running full OGC
    validity scans when those public results feed immediately into another
    polygon operation.
    """
    if owned is None:
        return
    all_valid = np.ones(owned.row_count, dtype=bool)
    owned._cached_is_valid_mask = all_valid
    owned._cached_is_valid_exact_collinearity_mask = all_valid
    if owned.device_state is not None:
        owned.device_state.trusted_all_ogc_valid = True
        if cp is not None:
            owned._device_ogc_validity_proof = cp.ones(
                owned.row_count,
                dtype=cp.bool_,
            )


def seed_homogeneous_host_metadata(
    owned: OwnedGeometryArray | None,
    family: GeometryFamily,
) -> None:
    """Attach lightweight host routing metadata for a homogeneous owned output.

    GPU builders often know that every output row is valid and belongs to one
    family.  In that case validity/tags/family offsets can be synthesized on
    host without materializing any coordinate payload or copying metadata back
    from device state.
    """
    if owned is None:
        return
    row_count = int(owned.row_count)
    owned._validity = np.ones(row_count, dtype=np.bool_)
    owned._tags = np.full(row_count, FAMILY_TAGS[family], dtype=np.int8)
    owned._family_row_offsets = np.arange(row_count, dtype=np.int32)
    if owned.device_state is not None:
        owned.device_state.trusted_all_valid = True
        owned.device_state.trusted_homogeneous_family = family
        owned.device_state.trusted_unique_family_rows = True
        owned.device_state.trusted_family_domain = (family,)
        if family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
            owned.device_state.trusted_polygonal_only = True


def _concat_validity_caches(arrays: list[OwnedGeometryArray]) -> np.ndarray | None:
    """Concatenate per-row validity caches when every input cache is complete."""
    caches: list[np.ndarray] = []
    for array in arrays:
        cache = getattr(array, "_cached_is_valid_mask", None)
        if cache is None or int(cache.size) != array.row_count:
            return None
        caches.append(np.asarray(cache, dtype=bool))
    return np.concatenate(caches) if caches else np.empty(0, dtype=bool)


def _gather_validity_cache(
    base: OwnedGeometryArray,
    index_map: Any,
    *,
    length: int,
) -> np.ndarray | None:
    """Propagate an all-valid indexed-view cache without reading device indices."""
    cached = getattr(base, "_cached_is_valid_mask", None)
    if cached is None or int(cached.size) != base.row_count:
        return None
    cached = np.asarray(cached, dtype=bool)
    if bool(np.all(cached)):
        return np.ones(length, dtype=bool)
    if hasattr(index_map, "__cuda_array_interface__"):
        return None
    return cached[np.asarray(index_map, dtype=np.int64)]


class DiagnosticKind(StrEnum):
    CREATED = "created"
    TRANSFER = "transfer"
    MATERIALIZATION = "materialization"
    RUNTIME = "runtime"
    CACHE = "cache"
    FALLBACK = "fallback"


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
    _cached_shared_family_buffers: (
        tuple[tuple[GeometryFamily, FamilyGeometryBuffer], ...] | None
    ) = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )


@dataclass
class DeviceRegularGridRectMetadata:
    """Trusted regular rectangle-grid proof for device-resident polygon rows."""

    origin_x: float
    origin_y: float
    cell_width: float
    cell_height: float
    cols: int
    rows: int
    size: int
    total_bounds: tuple[float, float, float, float]


@dataclass(frozen=True)
class DeviceFixedGeometrySizeMetadata:
    """Trusted fixed widths and variable-width per-row capacity bounds."""

    first_level_count_per_row: int | None = None
    second_level_count_per_row: int | None = None
    coord_count_per_row: int | None = None
    max_first_level_count_per_row: int | None = None
    max_second_level_count_per_row: int | None = None
    max_coord_count_per_row: int | None = None

    def __post_init__(self) -> None:
        for fixed_name, maximum_name in (
            ("first_level_count_per_row", "max_first_level_count_per_row"),
            ("second_level_count_per_row", "max_second_level_count_per_row"),
            ("coord_count_per_row", "max_coord_count_per_row"),
        ):
            fixed = getattr(self, fixed_name)
            maximum = getattr(self, maximum_name)
            if fixed is not None and maximum is None:
                object.__setattr__(self, maximum_name, int(fixed))
                maximum = fixed
            if fixed is not None and int(fixed) < 0:
                raise ValueError(f"{fixed_name} must be nonnegative")
            if maximum is not None and int(maximum) < 0:
                raise ValueError(f"{maximum_name} must be nonnegative")
            if fixed is not None and maximum is not None and int(fixed) > int(maximum):
                raise ValueError(f"{maximum_name} cannot be smaller than {fixed_name}")


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
    dense_single_ring_width: int | None = None
    axis_aligned_rectangles: bool = False
    regular_grid_rect: DeviceRegularGridRectMetadata | None = None
    fixed_size: DeviceFixedGeometrySizeMetadata | None = None


def device_family_coordinate_counts(
    buffer: DeviceFamilyGeometryBuffer,
    source_rows: DeviceArray | None = None,
) -> DeviceArray:
    """Return device-resident coordinate spans for selected physical rows."""
    if cp is None:  # pragma: no cover - GPU-only helper
        raise RuntimeError("CuPy is required for device coordinate counts")
    rows = (
        cp.arange(_device_family_row_count(buffer), dtype=cp.int32)
        if source_rows is None
        else cp.asarray(source_rows, dtype=cp.int32)
    )
    starts = buffer.geometry_offsets[rows]
    ends = buffer.geometry_offsets[rows + 1]
    if buffer.family in {
        GeometryFamily.POINT,
        GeometryFamily.LINESTRING,
        GeometryFamily.MULTIPOINT,
    }:
        return ends - starts
    if buffer.family in {GeometryFamily.POLYGON, GeometryFamily.MULTILINESTRING}:
        child_offsets = (
            buffer.ring_offsets if buffer.family is GeometryFamily.POLYGON else buffer.part_offsets
        )
        if child_offsets is None:
            raise RuntimeError(f"{buffer.family.value} device buffer is missing child offsets")
        return child_offsets[ends] - child_offsets[starts]
    if buffer.part_offsets is None or buffer.ring_offsets is None:
        raise RuntimeError("multipolygon device buffer is missing nested offsets")
    ring_starts = buffer.part_offsets[starts]
    ring_ends = buffer.part_offsets[ends]
    return buffer.ring_offsets[ring_ends] - buffer.ring_offsets[ring_starts]


def ensure_device_geometry_size_bounds(
    owned: OwnedGeometryArray,
    *,
    reason: str,
) -> int:
    """Attach host-visible per-row size bounds derived from device offsets.

    Variable-width native inputs can arrive without fixed-size metadata even
    though their nested offsets already prove tight allocation bounds. Reduce
    every missing family proof on device and export one small planning packet;
    geometry coordinates and row metadata remain resident.

    Returns the maximum coordinate span of any segment-producing source row.
    A coordinate span is a conservative segment bound for lineal/polygonal
    families and can therefore size row-isolated topology pages safely.
    """
    if cp is None:  # pragma: no cover - GPU-only helper
        raise RuntimeError("CuPy is required for device geometry size bounds")

    root = owned
    while root.is_indexed_view:
        if root._base is None:
            raise RuntimeError("indexed geometry size proof is missing its physical root")
        root = root._base
    state = root._ensure_device_state(preserve_indexed_view=True)

    missing: list[tuple[GeometryFamily, DeviceFamilyGeometryBuffer]] = []
    packets = []

    def _maximum(values):
        return (
            cp.asarray(0, dtype=cp.int64)
            if int(values.size) == 0
            else cp.max(values).astype(cp.int64, copy=False)
        )

    for family, buffer in state.families.items():
        if family is GeometryFamily.POINT:
            # A point row is intrinsically bounded by one coordinate and has
            # no nested structural levels.  Do not spend a planning fence to
            # restate that schema invariant.
            buffer._device_size_bounds_exact = True
            continue
        if getattr(buffer, "_device_size_bounds_exact", False):
            continue
        fixed_size = _device_buffer_fixed_size_metadata(family, buffer)
        exact_width_fields = ["coord_count_per_row"]
        if family in (
            GeometryFamily.POLYGON,
            GeometryFamily.MULTILINESTRING,
            GeometryFamily.MULTIPOLYGON,
        ):
            exact_width_fields.append("first_level_count_per_row")
        if family is GeometryFamily.MULTIPOLYGON:
            exact_width_fields.append("second_level_count_per_row")
        if fixed_size is not None and all(
            getattr(fixed_size, field) is not None for field in exact_width_fields
        ):
            buffer._device_size_bounds_exact = True
            continue

        d_geometry_offsets = cp.asarray(buffer.geometry_offsets, dtype=cp.int64)
        d_first_counts = d_geometry_offsets[1:] - d_geometry_offsets[:-1]
        d_max_first = cp.asarray(0, dtype=cp.int64)
        d_max_second = cp.asarray(0, dtype=cp.int64)
        if family in (
            GeometryFamily.POLYGON,
            GeometryFamily.MULTILINESTRING,
            GeometryFamily.MULTIPOLYGON,
        ):
            d_max_first = _maximum(d_first_counts)
        if family is GeometryFamily.MULTIPOLYGON:
            if buffer.part_offsets is None:
                raise RuntimeError("multipolygon device buffer is missing part offsets")
            d_part_offsets = cp.asarray(buffer.part_offsets, dtype=cp.int64)
            d_ring_starts = d_part_offsets[d_geometry_offsets[:-1]]
            d_ring_ends = d_part_offsets[d_geometry_offsets[1:]]
            d_max_second = _maximum(d_ring_ends - d_ring_starts)

        d_coord_counts = device_family_coordinate_counts(buffer)
        d_max_coord = _maximum(d_coord_counts)
        packets.append(cp.stack((d_max_first, d_max_second, d_max_coord)))
        missing.append((family, buffer))

    if packets:
        d_packet = cp.concatenate(packets)
        h_packet = np.asarray(
            get_cuda_runtime().copy_device_to_host(d_packet, reason=reason),
            dtype=np.int64,
        ).reshape(len(missing), 3)
        for (family, buffer), (max_first, max_second, max_coord) in zip(
            missing,
            h_packet,
            strict=True,
        ):
            existing = _device_buffer_fixed_size_metadata(family, buffer)
            fixed_first = None if existing is None else existing.first_level_count_per_row
            fixed_second = None if existing is None else existing.second_level_count_per_row
            fixed_coord = None if existing is None else existing.coord_count_per_row
            buffer.fixed_size = DeviceFixedGeometrySizeMetadata(
                first_level_count_per_row=fixed_first,
                second_level_count_per_row=fixed_second,
                coord_count_per_row=fixed_coord,
                max_first_level_count_per_row=max(
                    int(max_first),
                    0 if fixed_first is None else int(fixed_first),
                ),
                max_second_level_count_per_row=max(
                    int(max_second),
                    0 if fixed_second is None else int(fixed_second),
                ),
                max_coord_count_per_row=max(
                    int(max_coord),
                    0 if fixed_coord is None else int(fixed_coord),
                ),
            )
            buffer._device_size_bounds_exact = True

    segment_families = {
        GeometryFamily.LINESTRING,
        GeometryFamily.POLYGON,
        GeometryFamily.MULTILINESTRING,
        GeometryFamily.MULTIPOLYGON,
    }
    segment_bound = max(
        (
            int(state.families[family].fixed_size.max_coord_count_per_row)
            for family in state.families
            if family in segment_families
            and state.families[family].fixed_size is not None
            and state.families[family].fixed_size.max_coord_count_per_row is not None
        ),
        default=0,
    )
    carried_bounds = [
        bound
        for bound in (
            root._active_family_row_segment_capacity_bound,
            owned._active_family_row_segment_capacity_bound,
        )
        if bound is not None
    ]
    if carried_bounds:
        segment_bound = min(segment_bound, *(int(bound) for bound in carried_bounds))
    root._active_family_row_segment_capacity_bound = segment_bound
    owned._active_family_row_segment_capacity_bound = segment_bound
    return segment_bound


def build_updated_device_family_buffer(
    family: GeometryFamily,
    device_buf: DeviceFamilyGeometryBuffer,
    d_x_out: DeviceArray,
    d_y_out: DeviceArray,
    d_new_offsets: DeviceArray,
) -> DeviceFamilyGeometryBuffer:
    """Rebuild a device family buffer after a span-preserving coordinate rewrite."""
    if family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
        return DeviceFamilyGeometryBuffer(
            family=family,
            x=d_x_out,
            y=d_y_out,
            geometry_offsets=device_buf.geometry_offsets,
            empty_mask=device_buf.empty_mask,
            part_offsets=device_buf.part_offsets,
            ring_offsets=d_new_offsets,
            bounds=None,
            dense_single_ring_width=device_buf.dense_single_ring_width,
            axis_aligned_rectangles=False,
            regular_grid_rect=None,
            fixed_size=device_buf.fixed_size,
        )
    if family is GeometryFamily.MULTILINESTRING:
        return DeviceFamilyGeometryBuffer(
            family=family,
            x=d_x_out,
            y=d_y_out,
            geometry_offsets=device_buf.geometry_offsets,
            empty_mask=device_buf.empty_mask,
            part_offsets=d_new_offsets,
            ring_offsets=device_buf.ring_offsets,
            bounds=None,
            dense_single_ring_width=device_buf.dense_single_ring_width,
            axis_aligned_rectangles=False,
            regular_grid_rect=None,
            fixed_size=device_buf.fixed_size,
        )
    return DeviceFamilyGeometryBuffer(
        family=family,
        x=d_x_out,
        y=d_y_out,
        geometry_offsets=d_new_offsets,
        empty_mask=device_buf.empty_mask,
        part_offsets=device_buf.part_offsets,
        ring_offsets=device_buf.ring_offsets,
        bounds=None,
        dense_single_ring_width=device_buf.dense_single_ring_width,
        axis_aligned_rectangles=False,
        regular_grid_rect=None,
        fixed_size=device_buf.fixed_size,
    )


def build_updated_host_family_buffer(
    family: GeometryFamily,
    host_buf: FamilyGeometryBuffer,
    x_out: np.ndarray,
    y_out: np.ndarray,
    new_offsets: np.ndarray,
) -> FamilyGeometryBuffer:
    """Rebuild a host family buffer after a span-preserving coordinate rewrite."""
    if family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
        return FamilyGeometryBuffer(
            family=family,
            schema=host_buf.schema,
            row_count=host_buf.row_count,
            x=x_out,
            y=y_out,
            geometry_offsets=host_buf.geometry_offsets,
            empty_mask=host_buf.empty_mask,
            part_offsets=host_buf.part_offsets,
            ring_offsets=new_offsets,
            bounds=None,
        )
    if family is GeometryFamily.MULTILINESTRING:
        return FamilyGeometryBuffer(
            family=family,
            schema=host_buf.schema,
            row_count=host_buf.row_count,
            x=x_out,
            y=y_out,
            geometry_offsets=host_buf.geometry_offsets,
            empty_mask=host_buf.empty_mask,
            part_offsets=new_offsets,
            ring_offsets=host_buf.ring_offsets,
            bounds=None,
        )
    return FamilyGeometryBuffer(
        family=family,
        schema=host_buf.schema,
        row_count=host_buf.row_count,
        x=x_out,
        y=y_out,
        geometry_offsets=new_offsets,
        empty_mask=host_buf.empty_mask,
        part_offsets=host_buf.part_offsets,
        ring_offsets=host_buf.ring_offsets,
        bounds=None,
    )


@dataclass
class OwnedGeometryDeviceState:
    validity: DeviceArray
    tags: DeviceArray
    family_row_offsets: DeviceArray
    families: dict[GeometryFamily, DeviceFamilyGeometryBuffer]
    _column_refs: list | None = None
    row_bounds: DeviceArray | None = None  # cached per-row (N, 4) fp64 bounds on device
    trusted_all_valid: bool | None = None
    trusted_all_ogc_valid: bool | None = None
    trusted_homogeneous_family: GeometryFamily | None = None
    trusted_all_non_empty: bool | None = None
    trusted_all_finite_coordinates: bool | None = None
    trusted_nonempty_polygonal_positive_area: bool | None = None
    trusted_polygonal_only: bool | None = None
    trusted_unique_family_rows: bool | None = None
    trusted_family_domain: tuple[GeometryFamily, ...] | None = None
    point_location_indexes: dict[Any, Any] = field(default_factory=dict)
    point_location_index_decisions: dict[GeometryFamily, Any] = field(
        default_factory=dict
    )
    polygon_certificates: dict[tuple[str, GeometryFamily, int], Any] = field(
        default_factory=dict
    )


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
        _row_count: int | None = None,
    ) -> None:
        self._validity = validity
        self._tags = tags
        self._family_row_offsets = family_row_offsets
        self.families = families
        self.residency = residency
        self.diagnostics: list[DiagnosticEvent] = diagnostics if diagnostics is not None else []
        self.runtime_history: list[RuntimeSelection] = (
            runtime_history if runtime_history is not None else []
        )
        self.geoarrow_backed = geoarrow_backed
        self.shares_geoarrow_memory = shares_geoarrow_memory
        self.device_adopted = device_adopted
        self.device_state = device_state
        self._cached_is_valid_mask: np.ndarray | None = None
        self._device_ogc_validity_proof = None
        self._cached_shared_geoarrow_view: MixedGeoArrowView | None = None
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
        self._index_map_unique = False
        self._active_index_map_unique = False
        self._active_family_row_multiplicity_bound: int | None = None
        self._active_family_row_segment_capacity_bound: int | None = None
        self._row_active_mask: np.ndarray | Any | None = None
        # Cache row_count so we don't need to materialise host arrays just
        # to query the length.  When host validity is present we derive it;
        # otherwise the caller must supply _row_count explicitly.
        if _row_count is not None:
            self._row_count = _row_count
        elif validity is not None:
            self._row_count = int(validity.size)
        elif device_state is not None:
            self._row_count = int(device_state.validity.size)
        else:
            raise ValueError(
                "Cannot determine row_count: provide validity or device_state or _row_count"
            )

    # ------------------------------------------------------------------
    # Lazy-materialising metadata properties
    # ------------------------------------------------------------------

    def _ensure_host_metadata(self) -> None:
        """Transfer device metadata arrays to host if not already present."""
        if self._validity is not None:
            return  # already materialised
        if (
            self.device_state is None
            and self.is_indexed_view
            and self._base is not None
            and self._base.device_state is not None
        ):
            self._ensure_device_state(preserve_indexed_view=True)
        if self.device_state is None:
            raise RuntimeError(
                "Host metadata is None and no device metadata available for lazy materialisation"
            )
        runtime = get_cuda_runtime()
        self._validity = runtime.copy_device_to_host(
            self.device_state.validity,
            reason="owned geometry host metadata validity boundary",
        )
        self._tags = runtime.copy_device_to_host(
            self.device_state.tags,
            reason="owned geometry host metadata family-tag boundary",
        )
        self._family_row_offsets = runtime.copy_device_to_host(
            self.device_state.family_row_offsets,
            reason="owned geometry host metadata family-row-offset boundary",
        )

    def _ensure_host_family_structure(self, family: GeometryFamily) -> None:
        """Materialize host-side structural arrays for one family without x/y."""
        if family not in self.families:
            return
        buffer = self.families[family]
        if buffer.host_materialized:
            return
        if self.device_state is None or family not in self.device_state.families:
            return

        device_buffer = self.device_state.families[family]
        need_geometry_offsets = buffer.geometry_offsets.size == 0
        need_empty_mask = buffer.empty_mask.size == 0
        need_part_offsets = buffer.part_offsets is None and device_buffer.part_offsets is not None
        need_ring_offsets = buffer.ring_offsets is None and device_buffer.ring_offsets is not None
        need_bounds = buffer.bounds is None and device_buffer.bounds is not None
        if not any(
            (
                need_geometry_offsets,
                need_empty_mask,
                need_part_offsets,
                need_ring_offsets,
                need_bounds,
            )
        ):
            return

        runtime = get_cuda_runtime()
        geometry_offsets = (
            runtime.copy_device_to_host(
                device_buffer.geometry_offsets,
                reason=f"owned geometry {family.value} structure geometry-offset boundary",
            )
            if need_geometry_offsets
            else buffer.geometry_offsets
        )
        empty_mask = (
            runtime.copy_device_to_host(
                device_buffer.empty_mask,
                reason=f"owned geometry {family.value} structure empty-mask boundary",
            )
            if need_empty_mask
            else buffer.empty_mask
        )
        part_offsets = (
            runtime.copy_device_to_host(
                device_buffer.part_offsets,
                reason=f"owned geometry {family.value} structure part-offset boundary",
            )
            if need_part_offsets
            else buffer.part_offsets
        )
        ring_offsets = (
            runtime.copy_device_to_host(
                device_buffer.ring_offsets,
                reason=f"owned geometry {family.value} structure ring-offset boundary",
            )
            if need_ring_offsets
            else buffer.ring_offsets
        )
        bounds = (
            runtime.copy_device_to_host(
                device_buffer.bounds,
                reason=f"owned geometry {family.value} structure bounds boundary",
            )
            if need_bounds
            else buffer.bounds
        )
        self.families[family] = FamilyGeometryBuffer(
            family=buffer.family,
            schema=buffer.schema,
            row_count=buffer.row_count,
            x=buffer.x,
            y=buffer.y,
            geometry_offsets=np.ascontiguousarray(geometry_offsets, dtype=np.int32),
            empty_mask=np.ascontiguousarray(empty_mask, dtype=np.bool_),
            part_offsets=(
                None if part_offsets is None else np.ascontiguousarray(part_offsets, dtype=np.int32)
            ),
            ring_offsets=(
                None if ring_offsets is None else np.ascontiguousarray(ring_offsets, dtype=np.int32)
            ),
            bounds=(None if bounds is None else np.ascontiguousarray(bounds, dtype=np.float64)),
            host_materialized=False,
        )

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

    def _current_cached_validity_mask(self) -> np.ndarray | None:
        cached = getattr(self, "_cached_is_valid_mask", None)
        if cached is not None and int(cached.size) == self.row_count:
            return np.asarray(cached, dtype=bool)

        if self.is_indexed_view and self._base is not None and self._index_map is not None:
            base_cached = getattr(self._base, "_cached_is_valid_mask", None)
            if base_cached is not None and int(base_cached.size) == self._base.row_count:
                index_map = self._index_map
                if hasattr(index_map, "get"):
                    if bool(np.all(base_cached)):
                        return np.ones(self.row_count, dtype=bool)
                    return None
                return np.asarray(
                    base_cached[np.asarray(index_map, dtype=np.int64)],
                    dtype=bool,
                )

        return None

    def _apply_row_activity(
        self,
        active_mask,
        *,
        assume_active_indices_unique: bool = False,
        preserve_row_bounds: bool = True,
    ) -> OwnedGeometryArray:
        """Mark inactive capacity lanes null while preserving row layout."""
        if int(active_mask.size) != self.row_count:
            raise ValueError("row activity mask length must match row_count")
        if self._row_active_mask is not None:
            if hasattr(active_mask, "__cuda_array_interface__") or hasattr(
                self._row_active_mask,
                "__cuda_array_interface__",
            ):
                if cp is None:
                    raise RuntimeError("device row activity composition requires CuPy")
                active_mask = cp.asarray(
                    self._row_active_mask,
                    dtype=cp.bool_,
                ) & cp.asarray(active_mask, dtype=cp.bool_)
            else:
                active_mask = np.asarray(
                    self._row_active_mask,
                    dtype=bool,
                ) & np.asarray(active_mask, dtype=bool)
        self._row_active_mask = active_mask
        self._active_index_map_unique |= bool(assume_active_indices_unique)
        mask_on_device = hasattr(active_mask, "__cuda_array_interface__")
        if not mask_on_device:
            h_active = np.asarray(active_mask, dtype=bool)
            if self._validity is not None:
                self._validity = np.asarray(self._validity, dtype=bool) & h_active
            if self._tags is not None:
                self._tags = np.where(h_active, self._tags, NULL_TAG).astype(
                    np.int8,
                    copy=False,
                )
            if self._family_row_offsets is not None:
                self._family_row_offsets = np.where(
                    h_active,
                    self._family_row_offsets,
                    -1,
                ).astype(np.int32, copy=False)
        if self.device_state is not None:
            if cp is None:
                raise RuntimeError("device row activity requires CuPy")
            d_active = cp.asarray(active_mask, dtype=cp.bool_)
            state = self.device_state
            state.validity = cp.asarray(state.validity, dtype=cp.bool_) & d_active
            state.tags = cp.where(
                d_active,
                cp.asarray(state.tags, dtype=cp.int8),
                np.int8(NULL_TAG),
            )
            state.family_row_offsets = cp.where(
                d_active,
                cp.asarray(state.family_row_offsets, dtype=cp.int32),
                cp.int32(-1),
            )
            if state.row_bounds is not None:
                state.row_bounds = (
                    cp.where(
                        d_active[:, None],
                        cp.asarray(state.row_bounds).reshape(self.row_count, 4),
                        cp.asarray(cp.nan, dtype=cp.float64),
                    )
                    if preserve_row_bounds
                    else None
                )
            state.trusted_all_valid = True if self.row_count == 0 else False
            state.trusted_all_non_empty = None
            if (
                assume_active_indices_unique
                and self.is_indexed_view
                and self._base is not None
                and self._base.device_state is not None
                and self._base.device_state.trusted_unique_family_rows is True
            ):
                state.trusted_unique_family_rows = True
        self._cached_is_valid_mask = None
        return self

    def _propagate_cached_validity_mask(
        self,
        result: OwnedGeometryArray,
        indices,
    ) -> OwnedGeometryArray:
        if self._row_active_mask is not None:
            if hasattr(indices, "__cuda_array_interface__"):
                if cp is None:
                    raise RuntimeError("device row activity propagation requires CuPy")
                selected_activity = cp.asarray(self._row_active_mask, dtype=cp.bool_)[
                    cp.asarray(indices, dtype=cp.int64)
                ]
            else:
                selected_activity = np.asarray(
                    self._row_active_mask,
                    dtype=bool,
                )[np.asarray(indices, dtype=np.int64)]
            result._apply_row_activity(selected_activity)

        cached = self._current_cached_validity_mask()
        if cached is None:
            return result

        if hasattr(indices, "get"):
            if bool(np.all(cached)):
                result._cached_is_valid_mask = np.ones(result.row_count, dtype=bool)
            return result
        taken_indices = np.asarray(indices, dtype=np.int64)
        result._cached_is_valid_mask = np.asarray(cached[taken_indices], dtype=bool)
        return result

    @classmethod
    def _indexed_view(
        cls,
        base: OwnedGeometryArray,
        index_map: Any,
        *,
        assume_unique_indices: bool = False,
        expand_device_metadata: bool = True,
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
        _index_map_on_device = cp is not None and hasattr(index_map, "__cuda_array_interface__")
        index_map_size = int(index_map.size)

        if _index_map_on_device:
            if not expand_device_metadata:
                d_index_map = cp.asarray(index_map)
                if d_index_map.dtype not in (cp.int32, cp.int64):
                    d_index_map = d_index_map.astype(cp.int64, copy=False)
                view = cls(
                    validity=None,
                    tags=None,
                    family_row_offsets=None,
                    families=base.families,
                    residency=Residency.DEVICE,
                    device_state=None,
                    _row_count=index_map_size,
                )
                view._base = base
                view._index_map = d_index_map
                view._index_map_unique = bool(assume_unique_indices)
                return view
            # Device path: expand metadata on device using CuPy fancy indexing.
            # This avoids the D2H transfer of the inverse map and the base
            # metadata arrays that the old code path forced.
            ds = base._ensure_device_state(preserve_indexed_view=True)
            d_index_map = cp.asarray(index_map, dtype=cp.int64)
            d_expanded_validity = ds.validity[d_index_map]
            d_expanded_tags = ds.tags[d_index_map]
            d_expanded_fro = ds.family_row_offsets[d_index_map]
            d_state = OwnedGeometryDeviceState(
                validity=d_expanded_validity,
                tags=d_expanded_tags,
                family_row_offsets=d_expanded_fro,
                families=dict(ds.families),
                row_bounds=(
                    None
                    if ds.row_bounds is None
                    else ds.row_bounds[d_index_map].reshape(index_map_size, 4)
                ),
                trusted_all_valid=ds.trusted_all_valid,
                trusted_all_ogc_valid=ds.trusted_all_ogc_valid,
                trusted_homogeneous_family=ds.trusted_homogeneous_family,
                trusted_all_non_empty=ds.trusted_all_non_empty,
                trusted_all_finite_coordinates=(
                    True if ds.trusted_all_finite_coordinates is True else None
                ),
                trusted_nonempty_polygonal_positive_area=(
                    ds.trusted_nonempty_polygonal_positive_area
                ),
                trusted_polygonal_only=(True if ds.trusted_polygonal_only is True else None),
                trusted_unique_family_rows=(
                    True
                    if ds.trusted_unique_family_rows is True and assume_unique_indices
                    else False
                ),
                trusted_family_domain=ds.trusted_family_domain,
            )

            view = cls(
                validity=None,  # host metadata stays lazy
                tags=None,
                family_row_offsets=None,
                families=base.families,
                residency=Residency.DEVICE,
                device_state=d_state,
                _row_count=index_map_size,
            )
            view._base = base
            # Store CuPy index map directly — no .get()
            view._index_map = d_index_map
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
            if cp is not None and base.device_state is not None:
                base_state = base._ensure_device_state(preserve_indexed_view=True)
                d_index_map = cp.asarray(h_index_map, dtype=cp.int64)
                valid_tags = np.unique(expanded_tags[np.asarray(expanded_validity, dtype=bool)])
                trusted_family_domain = tuple(
                    TAG_FAMILIES[int(tag)] for tag in valid_tags if int(tag) in TAG_FAMILIES
                )
                if len(trusted_family_domain) != int(valid_tags.size):
                    trusted_family_domain = None
                trusted_homogeneous_family = (
                    trusted_family_domain[0]
                    if trusted_family_domain is not None and len(trusted_family_domain) == 1
                    else None
                )
                d_state = OwnedGeometryDeviceState(
                    validity=base_state.validity[d_index_map],
                    tags=base_state.tags[d_index_map],
                    family_row_offsets=base_state.family_row_offsets[d_index_map],
                    families=dict(base_state.families),
                    row_bounds=(
                        None
                        if base_state.row_bounds is None
                        else base_state.row_bounds[d_index_map].reshape(index_map_size, 4)
                    ),
                    trusted_all_valid=(True if bool(np.all(expanded_validity)) else None),
                    trusted_all_ogc_valid=base_state.trusted_all_ogc_valid,
                    trusted_homogeneous_family=trusted_homogeneous_family,
                    trusted_all_non_empty=base_state.trusted_all_non_empty,
                    trusted_all_finite_coordinates=(
                        True
                        if base_state.trusted_all_finite_coordinates is True
                        else None
                    ),
                    trusted_nonempty_polygonal_positive_area=(
                        base_state.trusted_nonempty_polygonal_positive_area
                    ),
                    trusted_polygonal_only=(
                        True
                        if trusted_family_domain is not None
                        and set(trusted_family_domain)
                        <= {
                            GeometryFamily.POLYGON,
                            GeometryFamily.MULTIPOLYGON,
                        }
                        else None
                    ),
                    trusted_unique_family_rows=(
                        True
                        if base_state.trusted_unique_family_rows is True and assume_unique_indices
                        else False
                    ),
                    trusted_family_domain=trusted_family_domain,
                )
                view = cls(
                    validity=expanded_validity,
                    tags=expanded_tags,
                    family_row_offsets=expanded_fro,
                    families=base.families,
                    residency=Residency.DEVICE,
                    device_state=d_state,
                    _row_count=index_map_size,
                )
                view._base = base
                view._index_map = d_index_map
            else:
                view = cls(
                    validity=expanded_validity,
                    tags=expanded_tags,
                    family_row_offsets=expanded_fro,
                    families=base.families,
                    residency=base.residency,
                    device_state=None,
                    _row_count=index_map_size,
                )
                view._base = base
                view._index_map = h_index_map

        view._index_map_unique = bool(assume_unique_indices)
        view._active_family_row_segment_capacity_bound = (
            base._active_family_row_segment_capacity_bound
        )

        view._record(
            DiagnosticKind.CREATED,
            f"indexed view: {index_map_size} logical rows over "
            f"{base.row_count} base rows ({base.row_count / max(index_map_size, 1):.1%} unique)",
            visible=False,
        )
        gathered_validity = _gather_validity_cache(
            base,
            index_map,
            length=index_map_size,
        )
        if gathered_validity is not None:
            view._cached_is_valid_mask = gathered_validity
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

        _map_on_device = cp is not None and hasattr(self._index_map, "__cuda_array_interface__")

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
        if self._row_active_mask is not None:
            resolved._apply_row_activity(self._row_active_mask)
        # Copy resolved state into self, clearing the indexed-view link.
        self._validity = resolved._validity
        self._tags = resolved._tags
        self._family_row_offsets = resolved._family_row_offsets
        self.families = resolved.families
        self._cached_shared_geoarrow_view = None
        self.residency = resolved.residency
        self.device_state = resolved.device_state
        self._base = None
        self._index_map = None
        self._index_map_unique = False
        self._active_index_map_unique = False
        self._row_active_mask = None
        self._record(
            DiagnosticKind.MATERIALIZATION,
            f"resolved indexed view: materialised {self._row_count} rows",
            visible=False,
        )
        return self

    def _device_resolve(
        self,
        *,
        allow_capacity_allocation: bool = True,
    ) -> OwnedGeometryArray:
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
        resolved = self.physicalize_device_rows(
            allow_capacity_allocation=allow_capacity_allocation,
        )
        # Copy resolved state into self, clearing the indexed-view link.
        self._validity = resolved._validity
        self._tags = resolved._tags
        self._family_row_offsets = resolved._family_row_offsets
        self.families = resolved.families
        self._cached_shared_geoarrow_view = None
        self.residency = resolved.residency
        self.device_state = resolved.device_state
        self._base = None
        self._index_map = None
        self._index_map_unique = False
        self._active_index_map_unique = False
        self._row_active_mask = None
        self._record(
            DiagnosticKind.MATERIALIZATION,
            f"device-resolved indexed view: materialised {self._row_count} rows on GPU",
            visible=False,
        )
        return self

    def detach_expanded_device_view(self) -> OwnedGeometryArray:
        """Promote an expanded device view to a standalone shared-buffer carrier.

        Device indexed views with an expanded ``device_state`` already own
        their logical routing metadata; only family coordinate buffers are
        shared with the source.  Clearing the source row map is therefore a
        zero-copy ownership transition.  It is useful at bounded chunk
        boundaries where retaining the source scatter root would otherwise
        retain inactive workspace lanes until terminal assembly.

        Deferred-metadata views cannot detach because their row map is still
        required to derive logical routing metadata.
        """
        if not self.is_indexed_view:
            return self
        if self.device_state is None:
            raise RuntimeError(
                "device indexed view requires expanded metadata before detaching"
            )
        if cp is None or not hasattr(self._index_map, "__cuda_array_interface__"):
            raise RuntimeError("device indexed view detachment requires a device row map")

        self._base = None
        self._index_map = None
        self._index_map_unique = False
        self._active_index_map_unique = False
        self._row_active_mask = None
        self._record(
            DiagnosticKind.CREATED,
            "detached expanded device row view with shared family buffers",
            visible=False,
        )
        return self

    def physicalize_device_rows(
        self,
        *,
        allow_capacity_allocation: bool = False,
    ) -> OwnedGeometryArray:
        """Return a contiguous device carrier for an indexed row view.

        This is an explicit physical-layout transition for kernels that cannot
        consume row indirection.  It never exports row metadata and leaves the
        source view intact so callers cannot accidentally erase shared-carrier
        provenance while preparing a constructive input.
        """
        if not self.is_indexed_view:
            return self
        base = self._base
        d_index_map = self._index_map
        if base is None or d_index_map is None:
            raise RuntimeError("indexed device view is missing its base or index map")
        if cp is None or not hasattr(d_index_map, "__cuda_array_interface__"):
            raise RuntimeError("device row physicalization requires a device index map")
        d_index_map = cp.asarray(d_index_map, dtype=cp.int64)
        d_active = cp.ones(self.row_count, dtype=cp.bool_)
        has_activity = self._row_active_mask is not None
        if self._row_active_mask is not None:
            d_active &= cp.asarray(self._row_active_mask, dtype=cp.bool_)
        assume_unique_indices = bool(self._index_map_unique) or bool(
            self._row_active_mask is not None and self._active_index_map_unique
        )
        while base.is_indexed_view:
            parent = base
            parent_base = parent._base
            parent_map = parent._index_map
            if (
                parent_base is None
                or parent_map is None
                or not hasattr(parent_map, "__cuda_array_interface__")
            ):
                raise RuntimeError("nested device row physicalization requires device index maps")
            if parent._row_active_mask is not None:
                d_active &= cp.asarray(parent._row_active_mask, dtype=cp.bool_)[d_index_map]
                has_activity = True
            d_index_map = cp.asarray(parent_map, dtype=cp.int64)[d_index_map]
            parent_indices_unique = bool(parent._index_map_unique) or bool(
                parent._row_active_mask is not None and parent._active_index_map_unique
            )
            assume_unique_indices &= parent_indices_unique
            base = parent_base
        base_state = base._ensure_device_state(preserve_indexed_view=True)
        has_bounded_family_widths = all(
            _device_buffer_has_bounded_row_width(family, buffer)
            for family, buffer in base_state.families.items()
        )
        if not assume_unique_indices and not has_bounded_family_widths:
            assume_unique_indices = _device_index_map_is_injective(
                d_index_map,
                source_row_count=base.row_count,
                active_mask=d_active if has_activity else None,
            )
        host_tags_for_sizing = None
        host_family_rows_for_sizing = None
        if self._tags is not None and self._family_row_offsets is not None:
            host_tags_for_sizing = np.asarray(self._tags, dtype=np.int8)
            host_family_rows_for_sizing = np.asarray(
                self._family_row_offsets,
                dtype=np.int64,
            )
        result = base._physical_device_take(
            d_index_map,
            host_tags_for_sizing=host_tags_for_sizing,
            host_family_rows_for_sizing=host_family_rows_for_sizing,
            allow_capacity_allocation=allow_capacity_allocation,
            assume_unique_indices=assume_unique_indices,
            active_row_mask=d_active if has_activity else None,
        )
        result._active_family_row_segment_capacity_bound = (
            self._active_family_row_segment_capacity_bound
        )
        result._record(
            DiagnosticKind.MATERIALIZATION,
            f"explicit device row physicalization: {self._row_count} rows",
            visible=False,
        )
        return result

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
                self._ensure_device_state(preserve_indexed_view=True)
            elif target_residency is Residency.HOST:
                self._ensure_host_state()
            if target_residency is not Residency.HOST or all(
                buffer.host_materialized for buffer in self.families.values()
            ):
                return self
        if target_residency is Residency.DEVICE:
            self._ensure_device_state(preserve_indexed_view=True)
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
        self._cached_shared_geoarrow_view = None
        runtime = get_cuda_runtime() if self.device_state is not None else None

        host_metadata_valid = (
            self._validity is not None
            and self._tags is not None
            and self._family_row_offsets is not None
            and int(self._validity.size) == self.row_count
            and int(self._tags.size) == self.row_count
            and int(self._family_row_offsets.size) == self.row_count
        )
        if host_metadata_valid:
            for family, buffer in self.families.items():
                tag = np.int8(FAMILY_TAGS[family])
                family_rows = np.asarray(
                    self._family_row_offsets[
                        np.asarray(self._validity, dtype=bool)
                        & (np.asarray(self._tags, dtype=np.int8) == tag)
                    ],
                    dtype=np.int64,
                )
                if family_rows.size == 0:
                    continue
                buffer_row_count = int(buffer.row_count)
                if self.device_state is not None and family in self.device_state.families:
                    buffer_row_count = (
                        int(self.device_state.families[family].geometry_offsets.size) - 1
                    )
                if np.any((family_rows < 0) | (family_rows >= buffer_row_count)):
                    host_metadata_valid = False
                    break

        if host_metadata_valid:
            validity = self._validity
            tags = self._tags
            family_row_offsets = self._family_row_offsets
        elif self.device_state is not None:
            validity = runtime.copy_device_to_host(
                self.device_state.validity,
                reason="owned geometry cache-bounds validity boundary",
            )
            tags = runtime.copy_device_to_host(
                self.device_state.tags,
                reason="owned geometry cache-bounds family-tag boundary",
            )
            family_row_offsets = runtime.copy_device_to_host(
                self.device_state.family_row_offsets,
                reason="owned geometry cache-bounds family-row-offset boundary",
            )
        else:
            validity = self.validity
            tags = self.tags
            family_row_offsets = self.family_row_offsets
        for family, buffer in self.families.items():
            device_buffer = None
            buffer_row_count = buffer.row_count
            if self.device_state is not None:
                device_buffer = self.device_state.families.get(family)
                if device_buffer is not None:
                    buffer_row_count = int(device_buffer.geometry_offsets.size) - 1
            row_indexes = np.flatnonzero(validity & (tags == FAMILY_TAGS[family]))
            if row_indexes.size == 0:
                continue
            family_rows = family_row_offsets[row_indexes]
            if buffer_row_count == row_indexes.size and np.array_equal(
                family_rows,
                np.arange(buffer_row_count, dtype=np.int32),
            ):
                family_bounds = np.ascontiguousarray(bounds[row_indexes], dtype=np.float64)
            else:
                # Broadcast and repeated-row views can map many logical rows
                # to the same physical family row. Cache per-family bounds
                # using physical family-row indices so the family buffer shape
                # stays aligned with buffer.row_count.
                family_bounds = np.full((buffer_row_count, 4), np.nan, dtype=np.float64)
                family_bounds[family_rows] = bounds[row_indexes]
            self.families[family] = FamilyGeometryBuffer(
                family=buffer.family,
                schema=buffer.schema,
                row_count=buffer_row_count,
                x=buffer.x,
                y=buffer.y,
                geometry_offsets=buffer.geometry_offsets,
                empty_mask=buffer.empty_mask,
                part_offsets=buffer.part_offsets,
                ring_offsets=buffer.ring_offsets,
                bounds=family_bounds,
                host_materialized=buffer.host_materialized,
            )
            if device_buffer is not None:
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

    def _ensure_device_state(
        self,
        *,
        preserve_indexed_view: bool = False,
    ) -> OwnedGeometryDeviceState:
        # Indexed views share the base's families dict but have expanded
        # metadata.  Compute kernels still default to contiguous family
        # buffers; terminal/export consumers that explicitly understand the
        # row-indirection carrier pass preserve_indexed_view=True.
        if self.is_indexed_view:
            _map_on_device = cp is not None and hasattr(self._index_map, "__cuda_array_interface__")
            if _map_on_device and self._base is not None and self._base.device_state is not None:
                if preserve_indexed_view:
                    if self.device_state is None:
                        base_state = self._base._ensure_device_state(
                            preserve_indexed_view=True,
                        )
                        d_index_map = self._index_map.astype(cp.int64, copy=False)
                        self.device_state = OwnedGeometryDeviceState(
                            validity=base_state.validity[d_index_map],
                            tags=base_state.tags[d_index_map],
                            family_row_offsets=base_state.family_row_offsets[d_index_map],
                            families=dict(base_state.families),
                            row_bounds=(
                                None
                                if base_state.row_bounds is None
                                else base_state.row_bounds[d_index_map].reshape(
                                    int(d_index_map.size),
                                    4,
                                )
                            ),
                            trusted_all_valid=base_state.trusted_all_valid,
                            trusted_all_ogc_valid=base_state.trusted_all_ogc_valid,
                            trusted_homogeneous_family=base_state.trusted_homogeneous_family,
                            trusted_all_non_empty=base_state.trusted_all_non_empty,
                            trusted_all_finite_coordinates=(
                                True
                                if base_state.trusted_all_finite_coordinates is True
                                else None
                            ),
                            trusted_nonempty_polygonal_positive_area=(
                                base_state.trusted_nonempty_polygonal_positive_area
                            ),
                            trusted_polygonal_only=(
                                True if base_state.trusted_polygonal_only is True else None
                            ),
                            trusted_unique_family_rows=(
                                True
                                if base_state.trusted_unique_family_rows is True
                                and bool(self._index_map_unique)
                                else False
                            ),
                            trusted_family_domain=base_state.trusted_family_domain,
                        )
                    return self.device_state
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
            if not buffer.host_materialized and buffer.row_count > 0 and buffer.x.shape[0] == 0:
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
        trusted_all_valid = (
            True
            if (
                self._validity is not None
                and int(self._validity.size) == int(self.row_count)
                and bool(np.all(self._validity))
            )
            else None
        )
        cached_ogc_validity = self._current_cached_validity_mask()
        trusted_all_ogc_valid = (
            True
            if cached_ogc_validity is not None
            and bool(np.all(np.asarray(cached_ogc_validity, dtype=bool)))
            else None
        )
        trusted_homogeneous_family = None
        trusted_all_non_empty = None
        trusted_all_finite_coordinates = None
        valid_tags = np.unique(
            np.asarray(self.tags, dtype=np.int8)[np.asarray(self.validity, dtype=np.bool_)]
        )
        trusted_family_domain = tuple(
            TAG_FAMILIES[int(tag)] for tag in valid_tags if int(tag) in TAG_FAMILIES
        )
        if len(trusted_family_domain) != int(valid_tags.size):
            trusted_family_domain = None
        trusted_unique_family_rows = True
        for family in self.families:
            family_mask = np.asarray(self.validity, dtype=np.bool_) & (
                np.asarray(self.tags, dtype=np.int8) == np.int8(FAMILY_TAGS[family])
            )
            family_rows = np.asarray(
                self.family_row_offsets,
                dtype=np.int64,
            )[family_mask]
            if bool(np.any(family_rows < 0)) or int(np.unique(family_rows).size) != int(
                family_rows.size
            ):
                trusted_unique_family_rows = False
                break
        if trusted_all_valid is True and len(self.families) == 1:
            family = next(iter(self.families))
            family_tag = np.int8(FAMILY_TAGS[family])
            if (
                self._tags is not None
                and int(self._tags.size) == int(self.row_count)
                and bool(np.all(self._tags == family_tag))
            ):
                trusted_homogeneous_family = family
                host_buffer = self.families.get(family)
                if (
                    host_buffer is not None
                    and int(host_buffer.row_count) == int(self.row_count)
                    and bool(np.all(~np.asarray(host_buffer.empty_mask, dtype=bool)))
                ):
                    trusted_all_non_empty = True
                    if family is GeometryFamily.POINT:
                        trusted_all_finite_coordinates = bool(
                            np.all(np.isfinite(host_buffer.x))
                            and np.all(np.isfinite(host_buffer.y))
                        )
        self.device_state = OwnedGeometryDeviceState(
            validity=d_validity,
            tags=d_tags,
            family_row_offsets=d_family_row_offsets,
            families={},
            trusted_all_valid=trusted_all_valid,
            trusted_all_ogc_valid=trusted_all_ogc_valid,
            trusted_homogeneous_family=trusted_homogeneous_family,
            trusted_all_non_empty=trusted_all_non_empty,
            trusted_all_finite_coordinates=trusted_all_finite_coordinates,
            trusted_polygonal_only=(
                True
                if set(self.families)
                <= {
                    GeometryFamily.POLYGON,
                    GeometryFamily.MULTIPOLYGON,
                }
                else None
            ),
            trusted_unique_family_rows=trusted_unique_family_rows,
            trusted_family_domain=trusted_family_domain,
        )
        for family, buffer in self.families.items():
            buf_bytes = (
                buffer.x.nbytes
                + buffer.y.nbytes
                + buffer.geometry_offsets.nbytes
                + buffer.empty_mask.nbytes
            )
            if buffer.part_offsets is not None:
                buf_bytes += buffer.part_offsets.nbytes
            if buffer.ring_offsets is not None:
                buf_bytes += buffer.ring_offsets.nbytes
            if buffer.bounds is not None:
                buf_bytes += buffer.bounds.nbytes
            total_bytes += buf_bytes
            dense_width = _host_dense_single_ring_width(buffer)
            axis_aligned_rectangles = _host_axis_aligned_rectangle_batch(buffer)
            fixed_size = _host_fixed_geometry_size_metadata(family, buffer)
            regular_grid_rect = None
            if (
                family is GeometryFamily.POLYGON
                and len(self.families) == 1
                and int(buffer.row_count) == int(self.row_count)
                and bool(np.all(self.validity))
                and np.array_equal(
                    self.family_row_offsets,
                    np.arange(self.row_count, dtype=np.int32),
                )
            ):
                regular_grid_rect = _host_regular_grid_rect_metadata(buffer)
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
                dense_single_ring_width=dense_width,
                axis_aligned_rectangles=axis_aligned_rectangles,
                regular_grid_rect=regular_grid_rect,
                fixed_size=fixed_size,
            )
        elapsed = perf_counter() - t0
        self._last_transfer_seconds = elapsed
        self._last_transfer_bytes = total_bytes
        return self.device_state

    def _ensure_host_state(self, *, preserve_indexed_view: bool = False) -> None:
        # Indexed views share the base's families dict but have expanded
        # metadata.  Host consumers default to a compact resolved array unless
        # they explicitly understand the row-indirection carrier.
        if self.is_indexed_view:
            _map_on_device = cp is not None and hasattr(self._index_map, "__cuda_array_interface__")
            if (
                preserve_indexed_view
                and _map_on_device
                and self._base is not None
                and self._base.device_state is not None
            ):
                self._base._ensure_host_state(preserve_indexed_view=True)
                self.families = self._base.families
                return
            if _map_on_device and self._base is not None and self._base.device_state is not None:
                # Device-resident indexed view: resolve on GPU first,
                # then fall through to the normal host materialisation below.
                self._device_resolve()
            else:
                self._resolve()
        if self.device_state is None:
            return

        def _needs_host_materialization(
            family: GeometryFamily,
            buffer: FamilyGeometryBuffer,
        ) -> bool:
            if not buffer.host_materialized:
                return True
            device_buffer = self.device_state.families.get(family)
            if device_buffer is None:
                return False
            if int(device_buffer.x.size) > 0 and (
                int(buffer.x.size) == 0 or int(buffer.y.size) == 0
            ):
                return True
            if (
                int(device_buffer.geometry_offsets.size) > 0
                and int(buffer.geometry_offsets.size) == 0
            ):
                return True
            if int(device_buffer.empty_mask.size) > 0 and int(buffer.empty_mask.size) == 0:
                return True
            if device_buffer.part_offsets is not None and buffer.part_offsets is None:
                return True
            return device_buffer.ring_offsets is not None and buffer.ring_offsets is None

        if not any(
            _needs_host_materialization(family, buffer) for family, buffer in self.families.items()
        ):
            return
        runtime = get_cuda_runtime()
        total_bytes = 0
        t0 = perf_counter()
        for family, buffer in tuple(self.families.items()):
            if not _needs_host_materialization(family, buffer):
                continue
            device_buffer = self.device_state.families[family]
            geometry_offsets = (
                buffer.geometry_offsets
                if buffer.geometry_offsets.size
                else runtime.copy_device_to_host(
                    device_buffer.geometry_offsets,
                    reason=f"owned geometry {family.value} geometry-offset materialization boundary",
                )
            )
            empty_mask = (
                buffer.empty_mask
                if buffer.empty_mask.size
                else runtime.copy_device_to_host(
                    device_buffer.empty_mask,
                    reason=f"owned geometry {family.value} empty-mask materialization boundary",
                )
            )
            if not buffer.geometry_offsets.size:
                total_bytes += geometry_offsets.nbytes
            if not buffer.empty_mask.size:
                total_bytes += empty_mask.nbytes
            first_level_count = 0 if int(geometry_offsets.size) == 0 else int(geometry_offsets[-1])
            part_offsets = buffer.part_offsets
            if part_offsets is None and device_buffer.part_offsets is not None:
                part_offset_count = first_level_count + 1
                part_offsets = runtime.copy_device_to_host(
                    device_buffer.part_offsets[:part_offset_count],
                    reason=f"owned geometry {family.value} part-offset materialization boundary",
                )
                total_bytes += part_offsets.nbytes
            elif part_offsets is not None:
                part_offsets = part_offsets[: first_level_count + 1]

            if family is GeometryFamily.MULTIPOLYGON:
                ring_count = (
                    0
                    if part_offsets is None or int(part_offsets.size) == 0
                    else int(part_offsets[-1])
                )
            elif family is GeometryFamily.POLYGON:
                ring_count = first_level_count
            else:
                ring_count = 0

            ring_offsets = buffer.ring_offsets
            if ring_offsets is None and device_buffer.ring_offsets is not None:
                ring_offsets = runtime.copy_device_to_host(
                    device_buffer.ring_offsets[: ring_count + 1],
                    reason=f"owned geometry {family.value} ring-offset materialization boundary",
                )
                total_bytes += ring_offsets.nbytes
            elif ring_offsets is not None:
                ring_offsets = ring_offsets[: ring_count + 1]

            if family in (
                GeometryFamily.POINT,
                GeometryFamily.LINESTRING,
                GeometryFamily.MULTIPOINT,
            ):
                coordinate_count = first_level_count
            elif family is GeometryFamily.MULTILINESTRING:
                coordinate_count = (
                    0
                    if part_offsets is None or int(part_offsets.size) == 0
                    else int(part_offsets[-1])
                )
            else:
                coordinate_count = (
                    0
                    if ring_offsets is None or int(ring_offsets.size) == 0
                    else int(ring_offsets[-1])
                )

            x = runtime.copy_device_to_host(
                device_buffer.x[:coordinate_count],
                reason=f"owned geometry {family.value} coordinate-x materialization boundary",
            )
            y = runtime.copy_device_to_host(
                device_buffer.y[:coordinate_count],
                reason=f"owned geometry {family.value} coordinate-y materialization boundary",
            )
            total_bytes += x.nbytes + y.nbytes
            bounds = buffer.bounds
            if bounds is None and device_buffer.bounds is not None:
                bounds = runtime.copy_device_to_host(
                    device_buffer.bounds,
                    reason=f"owned geometry {family.value} bounds materialization boundary",
                )
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
                    None
                    if part_offsets is None
                    else np.ascontiguousarray(part_offsets, dtype=np.int32)
                ),
                ring_offsets=(
                    None
                    if ring_offsets is None
                    else np.ascontiguousarray(ring_offsets, dtype=np.int32)
                ),
                bounds=None if bounds is None else np.ascontiguousarray(bounds, dtype=np.float64),
                host_materialized=True,
            )
        elapsed = perf_counter() - t0
        self._last_transfer_seconds = elapsed
        self._last_transfer_bytes = total_bytes

    @classmethod
    def _concat_device_indexed_views(
        cls,
        arrays: list[OwnedGeometryArray],
    ) -> OwnedGeometryArray | None:
        """Concatenate device arrays while preserving row-indirection views."""
        if cp is None or not any(array.is_indexed_view for array in arrays):
            return None

        def _flatten_device_view(
            array: OwnedGeometryArray,
        ) -> tuple[OwnedGeometryArray, object, object, bool, bool, bool] | None:
            if not array.is_indexed_view:
                if array.device_state is None:
                    return None
                d_activity = cp.ones(array.row_count, dtype=cp.bool_)
                has_activity = array._row_active_mask is not None
                if has_activity:
                    d_activity &= cp.asarray(array._row_active_mask, dtype=cp.bool_)
                return (
                    array,
                    cp.arange(array.row_count, dtype=cp.int64),
                    d_activity,
                    True,
                    True,
                    has_activity,
                )

            if (
                array._base is None
                or array._index_map is None
                or not hasattr(array._index_map, "__cuda_array_interface__")
            ):
                return None

            d_index_map = cp.asarray(array._index_map, dtype=cp.int64)
            d_activity = cp.ones(array.row_count, dtype=cp.bool_)
            has_activity = array._row_active_mask is not None
            if has_activity:
                d_activity &= cp.asarray(array._row_active_mask, dtype=cp.bool_)
            raw_unique = bool(array._index_map_unique)
            active_unique = raw_unique or bool(has_activity and array._active_index_map_unique)
            base = array._base
            while base.is_indexed_view:
                if (
                    base._base is None
                    or base._index_map is None
                    or not hasattr(base._index_map, "__cuda_array_interface__")
                ):
                    return None
                parent_has_activity = base._row_active_mask is not None
                if parent_has_activity:
                    d_activity &= cp.asarray(base._row_active_mask, dtype=cp.bool_)[d_index_map]
                    has_activity = True
                parent_raw_unique = bool(base._index_map_unique)
                parent_active_unique = parent_raw_unique or bool(
                    parent_has_activity and base._active_index_map_unique
                )
                raw_unique &= parent_raw_unique
                active_unique &= parent_active_unique
                parent_map = cp.asarray(base._index_map, dtype=cp.int64)
                d_index_map = parent_map[d_index_map]
                base = base._base

            if base.device_state is None:
                return None
            return (
                base,
                d_index_map,
                d_activity,
                raw_unique,
                active_unique,
                has_activity,
            )

        bases: list[OwnedGeometryArray] = []
        maps = []
        activities = []
        raw_uniqueness = []
        active_uniqueness = []
        has_activities = []
        base_row_offset = 0
        base_offsets: dict[int, int] = {}
        for array in arrays:
            flattened = _flatten_device_view(array)
            if flattened is None:
                return None
            (
                base,
                d_index_map,
                d_activity,
                raw_unique,
                active_unique,
                has_activity,
            ) = flattened
            base_key = id(base)
            source_offset = base_offsets.get(base_key)
            if source_offset is None:
                source_offset = base_row_offset
                base_offsets[base_key] = source_offset
                bases.append(base)
                base_row_offset += base.row_count
            maps.append(d_index_map + np.int64(source_offset))
            activities.append(d_activity)
            raw_uniqueness.append(raw_unique)
            active_uniqueness.append(active_unique)
            has_activities.append(has_activity)

        if not bases:
            return None
        physical_base = cls.concat(bases)
        if physical_base.device_state is None:
            return None
        roots_are_disjoint = len(bases) == len(arrays)
        view = cls._indexed_view(
            physical_base,
            cp.concatenate(maps) if len(maps) > 1 else maps[0],
            assume_unique_indices=roots_are_disjoint and all(raw_uniqueness),
        )
        if any(has_activities):
            view._apply_row_activity(
                cp.concatenate(activities) if len(activities) > 1 else activities[0],
                assume_active_indices_unique=(roots_are_disjoint and all(active_uniqueness)),
            )
        if roots_are_disjoint and all(
            array._ensure_device_state(
                preserve_indexed_view=True,
            ).trusted_unique_family_rows
            is True
            for array in arrays
        ):
            view._ensure_device_state(
                preserve_indexed_view=True,
            ).trusted_unique_family_rows = True
        cached_validity = _concat_validity_caches(arrays)
        if cached_validity is not None:
            view._cached_is_valid_mask = cached_validity
        _propagate_row_segment_capacity_bound(view, arrays)
        return view

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

        all_device = cp is not None and all(
            _device_geometry_has_device_root(array) for array in arrays
        )
        if (
            all_device
            and any(array.is_indexed_view for array in arrays)
            and _device_concat_requires_exact_physicalization(arrays)
        ):
            physical = device_physicalize_owned_row_selections_exact(
                [
                    (array, cp.ones(array.row_count, dtype=cp.bool_))
                    for array in arrays
                ],
                reason="indexed geometry concat exact allocation packet",
            )
            compact = [
                (
                    physical_array
                    if physical_array is not None
                    else build_null_owned_array(
                        source_array.row_count,
                        residency=Residency.DEVICE,
                    )
                )
                for source_array, physical_array in zip(
                    arrays,
                    physical,
                    strict=True,
                )
            ]
            return cls.concat(compact)

        indexed_device_concat = cls._concat_device_indexed_views(arrays)
        if indexed_device_concat is not None:
            return indexed_device_concat

        # Resolve any indexed views before concatenation — their families
        # dict is shared with a base array and would produce incorrect
        # family_row_offsets when combined with other arrays.
        for arr in arrays:
            if arr.is_indexed_view:
                arr._resolve()

        # --- Device-resident fast path ---
        all_device = cp is not None and all(a.device_state is not None for a in arrays)
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
        cached_validity = _concat_validity_caches(arrays)
        if cached_validity is not None:
            result._cached_is_valid_mask = cached_validity
        _propagate_row_segment_capacity_bound(result, arrays)
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
        d_all_row_bounds = (
            cp.concatenate(
                [
                    cp.asarray(ds.row_bounds, dtype=cp.float64).reshape(
                        arrays[index].row_count,
                        4,
                    )
                    for index, ds in enumerate(device_states)
                ],
                axis=0,
            )
            if all(ds.row_bounds is not None for ds in device_states)
            else None
        )
        trusted_all_valid = (
            True if all(ds.trusted_all_valid is True for ds in device_states) else None
        )
        trusted_all_ogc_valid = (
            True if all(ds.trusted_all_ogc_valid is True for ds in device_states) else None
        )
        trusted_all_non_empty = (
            True if all(ds.trusted_all_non_empty is True for ds in device_states) else None
        )
        trusted_all_finite_coordinates = (
            True
            if all(ds.trusted_all_finite_coordinates is True for ds in device_states)
            else None
        )
        trusted_nonempty_polygonal_positive_area = (
            True
            if all(ds.trusted_nonempty_polygonal_positive_area is True for ds in device_states)
            else None
        )
        trusted_unique_family_rows = (
            True if all(ds.trusted_unique_family_rows is True for ds in device_states) else None
        )
        trusted_family_domain = None
        if all(ds.trusted_family_domain is not None for ds in device_states):
            trusted_family_domain = tuple(
                dict.fromkeys(family for ds in device_states for family in ds.trusted_family_domain)
            )
        trusted_families = {
            ds.trusted_homogeneous_family
            for ds in device_states
            if ds.trusted_homogeneous_family is not None
        }
        trusted_homogeneous_family = (
            next(iter(trusted_families))
            if len(trusted_families) == 1
            and all(ds.trusted_homogeneous_family is not None for ds in device_states)
            else None
        )

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
            chunk_offsets = cp.asarray(ds.family_row_offsets, dtype=cp.int32)
            shifted_offsets = chunk_offsets
            for family in all_family_keys:
                if family not in ds.families:
                    continue
                d_buf = ds.families[family]
                family_row_count = _device_family_row_count(d_buf)
                if family_row_count == 0:
                    continue

                shift = family_cumulative[family]
                if shift:
                    if shifted_offsets is chunk_offsets:
                        shifted_offsets = chunk_offsets.copy()
                    shifted_offsets = cp.where(
                        ds.tags == FAMILY_TAGS[family],
                        chunk_offsets + np.int32(shift),
                        shifted_offsets,
                    )

                family_cumulative[family] += family_row_count
            d_all_family_row_offsets[row_cursor : row_cursor + n] = shifted_offsets
            row_cursor += n

        # 4. Concatenate per-family device buffers.
        new_device_families: dict[GeometryFamily, DeviceFamilyGeometryBuffer] = {}
        for family in all_family_keys:
            family_bufs = [ds.families[family] for ds in device_states if family in ds.families]
            if not family_bufs:
                continue
            new_device_families[family] = _concat_device_family_buffers(
                family,
                family_bufs,
            )

        # 5. Build host-side placeholder families (host_materialized=False).
        new_host_families: dict[GeometryFamily, FamilyGeometryBuffer] = {}
        for family, d_buf in new_device_families.items():
            schema = get_geometry_buffer_schema(family)
            fam_row_count = (
                int(d_buf.geometry_offsets.size) - 1 if d_buf.geometry_offsets.size > 0 else 0
            )
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
                row_bounds=d_all_row_bounds,
                trusted_all_valid=trusted_all_valid,
                trusted_all_ogc_valid=trusted_all_ogc_valid,
                trusted_homogeneous_family=trusted_homogeneous_family,
                trusted_all_non_empty=trusted_all_non_empty,
                trusted_all_finite_coordinates=trusted_all_finite_coordinates,
                trusted_nonempty_polygonal_positive_area=(trusted_nonempty_polygonal_positive_area),
                trusted_polygonal_only=(
                    True if all(ds.trusted_polygonal_only is True for ds in device_states) else None
                ),
                trusted_unique_family_rows=trusted_unique_family_rows,
                trusted_family_domain=trusted_family_domain,
            ),
            _row_count=total_rows,
        )
        result._record(
            DiagnosticKind.CREATED,
            f"device-resident concatenation of {len(arrays)} arrays totalling {total_rows} rows",
            visible=False,
        )
        cached_validity = _concat_validity_caches(arrays)
        if cached_validity is not None:
            result._cached_is_valid_mask = cached_validity
        _propagate_row_segment_capacity_bound(result, arrays)
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
        _indices_on_device = cp is not None and hasattr(indices, "__cuda_array_interface__")
        host_indices_for_sizing = None
        indices_for_device_take = indices
        if not _indices_on_device:
            host_indices = np.asarray(indices)
            if host_indices.dtype == bool:
                if int(host_indices.size) == self.row_count and bool(np.all(host_indices)):
                    return self
                host_indices_for_sizing = np.flatnonzero(host_indices).astype(
                    np.int64,
                    copy=False,
                )
                indices_for_device_take = host_indices_for_sizing
            else:
                host_indices = host_indices.astype(np.int64, copy=False)
                if int(host_indices.size) == self.row_count and bool(
                    np.array_equal(
                        host_indices,
                        np.arange(self.row_count, dtype=np.int64),
                    )
                ):
                    return self
                host_indices_for_sizing = host_indices
        if cp is not None and (
            (self.residency is Residency.DEVICE and self.device_state is not None)
            or _indices_on_device
        ):
            return self.device_take(
                indices_for_device_take,
                host_indices_for_sizing=host_indices_for_sizing,
            )

        if hasattr(indices, "dtype") and indices.dtype == bool:
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
                return self._propagate_cached_validity_mask(
                    self._physical_take(indices),
                    indices,
                )
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
                    return self._propagate_cached_validity_mask(
                        OwnedGeometryArray._indexed_view(physical_base, final_inverse),
                        indices,
                    )

                # Standard case: build physical take of unique rows only.
                physical_base = self._physical_take(unique_indices)
                return self._propagate_cached_validity_mask(
                    OwnedGeometryArray._indexed_view(physical_base, inverse),
                    indices,
                )

        # --- Physical copy path (original behaviour) ---
        return self._propagate_cached_validity_mask(
            self._physical_take(indices),
            indices,
        )

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
        result._record(
            DiagnosticKind.CREATED, f"subset {indices.size} rows via take", visible=False
        )
        return result

    def _device_take_prefers_row_indirection(self) -> bool:
        """Whether device_take should preserve row order as a native rowset view."""
        state = self._ensure_device_state(preserve_indexed_view=True)
        if not state.families:
            return False
        for family, device_buffer in state.families.items():
            if family is GeometryFamily.POINT:
                continue
            if not _device_buffer_has_exact_row_width(family, device_buffer):
                return True
        return False

    def _device_indexed_take(
        self,
        d_indices,
        *,
        host_indices_for_metadata: np.ndarray | None = None,
        assume_unique_indices: bool = False,
        active_family_row_multiplicity_bound: int | None = None,
        active_family_row_segment_capacity_bound: int | None = None,
        defer_device_metadata: bool = False,
    ) -> OwnedGeometryArray:
        """Return a device-backed row-indirection view for a take rowset."""
        if cp is None:
            raise RuntimeError("CuPy is required for device row-indirection take")

        inherited_active_multiplicity_bound = (
            self._active_family_row_multiplicity_bound
            if (
                active_family_row_multiplicity_bound is None
                and assume_unique_indices
                and self._row_active_mask is not None
            )
            else None
        )
        inherited_active_segment_bound = (
            self._active_family_row_segment_capacity_bound
            if active_family_row_segment_capacity_bound is None
            else None
        )
        if inherited_active_segment_bound is not None:
            active_family_row_segment_capacity_bound = int(inherited_active_segment_bound)
        elif active_family_row_segment_capacity_bound is not None:
            active_family_row_segment_capacity_bound = int(active_family_row_segment_capacity_bound)
            if active_family_row_segment_capacity_bound < 0:
                raise ValueError("active family-row segment capacity bound must be nonnegative")
        if inherited_active_multiplicity_bound is not None:
            active_family_row_multiplicity_bound = int(inherited_active_multiplicity_bound)
        elif active_family_row_multiplicity_bound is not None:
            active_family_row_multiplicity_bound = int(active_family_row_multiplicity_bound)
            if active_family_row_multiplicity_bound <= 0:
                raise ValueError("active family-row multiplicity bound must be positive")
            source_state = self._ensure_device_state(preserve_indexed_view=True)
            if source_state.trusted_unique_family_rows is not True:
                raise ValueError(
                    "active family-row multiplicity requires unique source family rows"
                )

        base = self
        d_index_map = (
            d_indices
            if defer_device_metadata and d_indices.dtype in (cp.int32, cp.int64)
            else d_indices.astype(cp.int64, copy=False)
        )
        index_map_unique = bool(assume_unique_indices)
        selected_activity = None
        active_index_map_unique = False
        if self._row_active_mask is not None:
            selected_activity = cp.asarray(self._row_active_mask, dtype=cp.bool_)[d_index_map]
            active_index_map_unique = bool(assume_unique_indices) and (
                bool(self._index_map_unique) or bool(self._active_index_map_unique)
            )
        if self.is_indexed_view and self._base is not None and self._index_map is not None:
            if hasattr(self._index_map, "__cuda_array_interface__"):
                base = self._base
                d_index_map = self._index_map[d_index_map].astype(cp.int64, copy=False)
                index_map_unique &= bool(getattr(self, "_index_map_unique", False))
            else:
                # Host-indexed views have no resident rowset carrier. Resolve
                # once and build the device view from the flat source.
                self._resolve()
                base = self

        view = OwnedGeometryArray._indexed_view(
            base,
            d_index_map,
            assume_unique_indices=index_map_unique,
            expand_device_metadata=not defer_device_metadata,
        )
        view._active_family_row_multiplicity_bound = active_family_row_multiplicity_bound
        view._active_family_row_segment_capacity_bound = active_family_row_segment_capacity_bound
        if selected_activity is not None:
            view._apply_row_activity(
                selected_activity,
                assume_active_indices_unique=active_index_map_unique,
            )
        if view.device_state is not None and self.device_state is not None:
            source_state = self.device_state
            if source_state.trusted_all_valid is True:
                view.device_state.trusted_all_valid = True
            if source_state.trusted_all_ogc_valid is True:
                view.device_state.trusted_all_ogc_valid = True
            if source_state.trusted_homogeneous_family is not None:
                view.device_state.trusted_homogeneous_family = (
                    source_state.trusted_homogeneous_family
                )
            if source_state.trusted_all_non_empty is True:
                view.device_state.trusted_all_non_empty = True
            if source_state.trusted_all_finite_coordinates is True:
                view.device_state.trusted_all_finite_coordinates = True
            if source_state.trusted_nonempty_polygonal_positive_area is True:
                view.device_state.trusted_nonempty_polygonal_positive_area = True
            if source_state.trusted_polygonal_only is True:
                view.device_state.trusted_polygonal_only = True
        if (
            host_indices_for_metadata is not None
            and self._validity is not None
            and self._tags is not None
            and self._family_row_offsets is not None
        ):
            host_indices = np.asarray(host_indices_for_metadata, dtype=np.int64)
            view._validity = np.ascontiguousarray(
                self._validity[host_indices],
                dtype=np.bool_,
            )
            view._tags = np.ascontiguousarray(
                self._tags[host_indices],
                dtype=np.int8,
            )
            view._family_row_offsets = np.ascontiguousarray(
                self._family_row_offsets[host_indices],
                dtype=np.int32,
            )
            valid_tags = np.unique(view._tags[np.asarray(view._validity, dtype=bool)])
            trusted_family_domain = tuple(
                TAG_FAMILIES[int(tag)] for tag in valid_tags if int(tag) in TAG_FAMILIES
            )
            if len(trusted_family_domain) != int(valid_tags.size):
                trusted_family_domain = None
            if view.device_state is not None:
                view.device_state.trusted_all_valid = True if bool(np.all(view._validity)) else None
                view.device_state.trusted_homogeneous_family = (
                    trusted_family_domain[0]
                    if trusted_family_domain is not None and len(trusted_family_domain) == 1
                    else None
                )
                view.device_state.trusted_polygonal_only = (
                    True
                    if trusted_family_domain is not None
                    and set(trusted_family_domain)
                    <= {
                        GeometryFamily.POLYGON,
                        GeometryFamily.MULTIPOLYGON,
                    }
                    else None
                )
                view.device_state.trusted_family_domain = trusted_family_domain
        view._record(
            DiagnosticKind.CREATED,
            f"device-side row-indirected subset {int(d_index_map.size)} rows via device_take",
            visible=False,
        )
        return view

    def device_take(
        self,
        indices,
        *,
        host_indices_for_sizing: np.ndarray | None = None,
        allow_capacity_allocation: bool = False,
        assume_unique_indices: bool = False,
    ) -> OwnedGeometryArray:
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
        indices_on_device = hasattr(indices, "__cuda_array_interface__")
        host_indices_for_device_sizing = None
        host_indices_for_metadata = None
        if host_indices_for_sizing is not None:
            host_indices = np.asarray(host_indices_for_sizing)
            if host_indices.dtype == bool:
                host_indices = np.flatnonzero(host_indices)
            host_indices_candidate = host_indices.astype(np.int64, copy=False)
            host_indices_for_device_sizing = host_indices_candidate
            index_count = (
                int(indices.size) if hasattr(indices, "size") else int(np.asarray(indices).size)
            )
            if int(host_indices_candidate.size) != index_count:
                raise ValueError("host_indices_for_sizing must match device_take index count")
            if (
                self._validity is not None
                and self._tags is not None
                and self._family_row_offsets is not None
            ):
                host_indices_for_metadata = host_indices_candidate
        if (
            host_indices_for_metadata is None
            and not indices_on_device
            and self._validity is not None
            and self._tags is not None
            and self._family_row_offsets is not None
        ):
            host_indices = np.asarray(indices)
            if host_indices.dtype == bool:
                host_indices = np.flatnonzero(host_indices)
            host_indices_for_device_sizing = host_indices.astype(np.int64, copy=False)
            host_indices_for_metadata = host_indices_for_device_sizing
        host_metadata = None
        if host_indices_for_metadata is not None:
            host_validity = np.ascontiguousarray(
                self._validity[host_indices_for_metadata],
                dtype=np.bool_,
            )
            host_tags = np.ascontiguousarray(
                self._tags[host_indices_for_metadata],
                dtype=np.int8,
            )
            host_family_row_offsets = np.full(host_tags.size, -1, dtype=np.int32)
            for family, tag in FAMILY_TAGS.items():
                if family not in self.families:
                    continue
                family_mask = host_tags == np.int8(tag)
                if family_mask.any():
                    host_family_row_offsets[family_mask] = np.arange(
                        int(family_mask.sum()),
                        dtype=np.int32,
                    )
            host_metadata = (
                host_validity,
                host_tags,
                np.ascontiguousarray(host_family_row_offsets, dtype=np.int32),
            )

        def _attach_host_metadata(result: OwnedGeometryArray) -> OwnedGeometryArray:
            if host_metadata is None or result.is_indexed_view:
                return result
            host_validity, host_tags, host_family_row_offsets = host_metadata
            if result._validity is None:
                result._validity = host_validity
            if result._tags is None:
                result._tags = host_tags
            if result._family_row_offsets is None:
                result._family_row_offsets = host_family_row_offsets
            return result

        if indices_on_device:
            d_indices = indices
        else:
            d_indices = cp.asarray(indices)

        # Bool mask → integer indices
        if d_indices.dtype == cp.bool_ or d_indices.dtype == bool:
            d_indices = cp.flatnonzero(d_indices).astype(cp.int64)
        else:
            d_indices = d_indices.astype(cp.int64, copy=False)

        if self._device_take_prefers_row_indirection():
            return self._propagate_cached_validity_mask(
                self._device_indexed_take(
                    d_indices,
                    host_indices_for_metadata=host_indices_for_device_sizing,
                    assume_unique_indices=assume_unique_indices,
                ),
                d_indices,
            )

        # --- Indexed-view fast path ---
        total = int(d_indices.size)
        if (
            total > 0
            and total < self._INDEXED_VIEW_MIN_ROWS
            and host_indices_for_device_sizing is not None
            and (
                np.unique(host_indices_for_device_sizing).size / total
                < self._INDEXED_VIEW_RATIO_THRESHOLD
            )
        ):
            return self._propagate_cached_validity_mask(
                self._device_indexed_take(
                    d_indices,
                    host_indices_for_metadata=host_indices_for_device_sizing,
                    assume_unique_indices=assume_unique_indices,
                ),
                d_indices,
            )

        # Use a cheap sampling heuristic to avoid the O(n log n) cost of
        # cp.unique when repetition is low.
        if total >= self._INDEXED_VIEW_MIN_ROWS:
            sample_size = min(1024, total)
            d_sample = d_indices[cp.linspace(0, total - 1, sample_size, dtype=cp.int64)]
            approx_unique = int(cp.unique(d_sample).size)
            if approx_unique / sample_size >= self._INDEXED_VIEW_RATIO_THRESHOLD:
                return _attach_host_metadata(
                    self._propagate_cached_validity_mask(
                        self._physical_device_take(
                            d_indices,
                            host_indices_for_sizing=host_indices_for_device_sizing,
                            allow_capacity_allocation=allow_capacity_allocation,
                            assume_unique_indices=assume_unique_indices,
                        ),
                        d_indices,
                    )
                )
            d_unique_indices, d_inverse = cp.unique(d_indices, return_inverse=True)
            unique_count = int(d_unique_indices.size)
            if unique_count / max(total, 1) < self._INDEXED_VIEW_RATIO_THRESHOLD:
                # Physical take of only the unique rows (on device)
                host_unique_indices = (
                    np.unique(host_indices_for_device_sizing)
                    if host_indices_for_device_sizing is not None
                    else None
                )
                physical_base = self._physical_device_take(
                    d_unique_indices,
                    host_indices_for_sizing=host_unique_indices,
                    allow_capacity_allocation=allow_capacity_allocation,
                    assume_unique_indices=True,
                )
                # Pass the CuPy inverse map directly -- no D2H transfer.
                # _indexed_view detects the CuPy array and expands metadata
                # on device, keeping the entire path GPU-resident.
                return _attach_host_metadata(
                    self._propagate_cached_validity_mask(
                        OwnedGeometryArray._indexed_view(physical_base, d_inverse),
                        d_indices,
                    )
                )

        # --- Physical copy path ---
        return _attach_host_metadata(
            self._propagate_cached_validity_mask(
                self._physical_device_take(
                    d_indices,
                    host_indices_for_sizing=host_indices_for_device_sizing,
                    allow_capacity_allocation=allow_capacity_allocation,
                    assume_unique_indices=assume_unique_indices,
                ),
                d_indices,
            )
        )

    def device_take_capacity(self, indices, active_mask) -> OwnedGeometryArray:
        """Take capacity rows and mark inactive device lanes null.

        The logical cardinality remains in the calling native selection
        carrier. Variable-width geometry keeps its indexed row carrier so a
        sparse capacity selection cannot multiply the physical coordinate
        allocation; consumers that require contiguous storage must request an
        explicit physicalization from that carrier.
        """
        if cp is None:
            raise RuntimeError("CuPy is required for device_take_capacity")
        d_indices = cp.asarray(indices, dtype=cp.int64)
        d_active = cp.asarray(active_mask, dtype=cp.bool_)
        if d_indices.ndim != 1 or d_active.ndim != 1:
            raise ValueError("capacity indices and active mask must be one-dimensional")
        if int(d_indices.size) != int(d_active.size):
            raise ValueError("capacity indices and active mask lengths must match")
        capacity = int(d_indices.size)
        if capacity > 0 and self.row_count == 0:
            raise ValueError("nonempty capacity take requires source geometry rows")

        source_state = self._ensure_device_state(preserve_indexed_view=True)
        safe_indices = cp.where(d_active, d_indices, cp.int64(0))
        result = self.device_take(
            safe_indices,
            allow_capacity_allocation=True,
        )
        result._apply_row_activity(d_active)
        state = result._ensure_device_state(preserve_indexed_view=True)
        state.trusted_all_ogc_valid = (
            True if capacity == 0 or source_state.trusted_all_ogc_valid is True else None
        )
        state.trusted_all_non_empty = None
        result._cached_is_valid_mask = None
        result._record(
            DiagnosticKind.CREATED,
            f"device capacity take retained {capacity} lanes with null inactive rows",
            visible=False,
        )
        return result

    def _physical_device_take(
        self,
        d_indices,
        *,
        host_indices_for_sizing: np.ndarray | None = None,
        host_tags_for_sizing: np.ndarray | None = None,
        host_family_rows_for_sizing: np.ndarray | None = None,
        allow_capacity_allocation: bool = False,
        assume_unique_indices: bool = False,
        active_row_mask=None,
    ) -> OwnedGeometryArray:
        """Perform a physical (non-virtual) device-side take."""
        d_state = self._ensure_device_state()

        d_new_validity = d_state.validity[d_indices]
        n_indices = int(d_indices.size)
        d_active_rows = None
        if active_row_mask is not None:
            d_active_rows = cp.asarray(active_row_mask, dtype=cp.bool_)
            if d_active_rows.ndim != 1 or int(d_active_rows.size) != n_indices:
                raise ValueError("device take activity must match the output row capacity")
            d_new_validity &= d_active_rows
        source_all_rows_valid = d_state.trusted_all_valid is True or (
            self._validity is not None
            and int(self._validity.size) == int(self.row_count)
            and bool(np.all(self._validity))
        )

        new_device_families: dict[GeometryFamily, DeviceFamilyGeometryBuffer] = {}
        new_host_families: dict[GeometryFamily, FamilyGeometryBuffer] = {}

        d_new_tags = d_state.tags[d_indices]
        if d_active_rows is not None:
            d_new_tags = cp.where(d_active_rows, d_new_tags, cp.int8(NULL_TAG))

        if len(d_state.families) == 1:
            family, device_buffer = next(iter(d_state.families.items()))
            family_tag = np.int8(FAMILY_TAGS[family])
            host_family = self.families.get(family)
            homogeneous_host_metadata = (
                self._tags is not None
                and int(self._tags.size) == self.row_count
                and self._family_row_offsets is not None
                and int(self._family_row_offsets.size) == self.row_count
                and bool(np.all(self._tags == family_tag))
            )
            trusted_single_family = d_state.trusted_homogeneous_family is family
            single_family_all_rows = (
                (
                    host_family is not None
                    and host_family.host_materialized
                    and int(host_family.row_count) == self.row_count
                )
                or homogeneous_host_metadata
                or (
                    trusted_single_family
                    and d_state.trusted_all_valid is True
                )
            )
            # A device state with one physical family may also contain null
            # rows, but every active tag and family-row offset belongs to that
            # family by construction.  Keep nullable rows masked instead of
            # synchronizing to revalidate the carrier invariant on each take.
            d_family_active = d_new_validity & (d_new_tags == family_tag)
            source_family_rows = d_state.family_row_offsets[d_indices].astype(
                cp.int64,
                copy=False,
            )
            if not single_family_all_rows or d_active_rows is not None:
                source_family_rows = cp.where(
                    d_family_active,
                    source_family_rows,
                    cp.int64(0),
                )
            d_new_family_row_offsets = cp.arange(n_indices, dtype=cp.int32)
            if not single_family_all_rows or d_active_rows is not None:
                d_new_family_row_offsets = cp.where(
                    d_family_active,
                    d_new_family_row_offsets,
                    cp.int32(-1),
                )
            host_family_rows = None
            if single_family_all_rows and host_family_rows_for_sizing is not None:
                host_family_rows = np.asarray(
                    host_family_rows_for_sizing,
                    dtype=np.int64,
                )
            elif single_family_all_rows and host_indices_for_sizing is not None:
                if self._family_row_offsets is not None:
                    host_family_rows = np.asarray(
                        self._family_row_offsets[host_indices_for_sizing],
                        dtype=np.int64,
                    )
                else:
                    host_family_rows = np.asarray(
                        host_indices_for_sizing,
                        dtype=np.int64,
                    )
            new_device_families[family] = _device_take_family_buffer(
                device_buffer,
                family,
                source_family_rows,
                self.families.get(family),
                host_family_rows=host_family_rows,
                allow_capacity_allocation=allow_capacity_allocation,
                assume_unique_indices=assume_unique_indices,
                active_row_mask=(
                    d_active_rows
                    if single_family_all_rows and d_active_rows is not None
                    else (None if single_family_all_rows else d_family_active)
                ),
            )
            schema = get_geometry_buffer_schema(family)
            new_host_families[family] = FamilyGeometryBuffer(
                family=family,
                schema=schema,
                row_count=n_indices,
                x=np.empty(0, dtype=np.float64),
                y=np.empty(0, dtype=np.float64),
                geometry_offsets=np.empty(0, dtype=np.int32),
                empty_mask=np.empty(0, dtype=np.bool_),
                host_materialized=False,
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
                    trusted_unique_family_rows=True,
                    trusted_family_domain=(family,),
                ),
                _row_count=n_indices,
            )
            if result.device_state is not None:
                if d_state.row_bounds is not None:
                    result.device_state.row_bounds = cp.asarray(d_state.row_bounds)[
                        d_indices
                    ].reshape(n_indices, 4)
                if source_all_rows_valid and d_active_rows is None:
                    result.device_state.trusted_all_valid = True
                if single_family_all_rows and d_active_rows is None:
                    result.device_state.trusted_homogeneous_family = family
                if (
                    d_active_rows is None
                    and d_state.trusted_all_non_empty is True
                    and d_state.trusted_homogeneous_family is family
                ):
                    result.device_state.trusted_all_non_empty = True
                if (
                    d_active_rows is None
                    and d_state.trusted_all_finite_coordinates is True
                ):
                    result.device_state.trusted_all_finite_coordinates = True
                if d_state.trusted_polygonal_only is True or family in (
                    GeometryFamily.POLYGON,
                    GeometryFamily.MULTIPOLYGON,
                ):
                    result.device_state.trusted_polygonal_only = True
            result._record(
                DiagnosticKind.CREATED,
                f"device-side subset {n_indices} rows via device_take",
                visible=False,
            )
            return result

        return self._physical_device_take_mixed(
            d_state,
            d_indices,
            d_new_validity,
            d_new_tags,
            n_indices,
            host_indices_for_sizing=host_indices_for_sizing,
            host_tags_for_sizing=host_tags_for_sizing,
            host_family_rows_for_sizing=host_family_rows_for_sizing,
            allow_capacity_allocation=allow_capacity_allocation,
            assume_unique_indices=assume_unique_indices,
        )

    def _physical_device_take_mixed(
        self,
        d_state,
        d_indices,
        d_new_validity,
        d_new_tags,
        n_indices: int,
        *,
        host_indices_for_sizing: np.ndarray | None = None,
        host_tags_for_sizing: np.ndarray | None = None,
        host_family_rows_for_sizing: np.ndarray | None = None,
        allow_capacity_allocation: bool = False,
        assume_unique_indices: bool = False,
    ) -> OwnedGeometryArray:
        """Device-side gather for mixed/null layouts."""
        d_new_family_row_offsets = cp.full(n_indices, -1, dtype=cp.int32)
        new_device_families: dict[GeometryFamily, DeviceFamilyGeometryBuffer] = {}
        new_host_families: dict[GeometryFamily, FamilyGeometryBuffer] = {}
        if host_tags_for_sizing is not None and host_family_rows_for_sizing is not None:
            host_tags_for_sizing = np.asarray(host_tags_for_sizing, dtype=np.int8)
            host_family_rows_for_sizing = np.asarray(
                host_family_rows_for_sizing,
                dtype=np.int64,
            )
        elif (
            host_indices_for_sizing is not None
            and self._tags is not None
            and self._family_row_offsets is not None
        ):
            host_tags_for_sizing = np.asarray(self._tags[host_indices_for_sizing])
            host_family_rows_for_sizing = np.asarray(
                self._family_row_offsets[host_indices_for_sizing],
                dtype=np.int64,
            )

        for family, device_buffer in d_state.families.items():
            family_mask = d_new_tags == FAMILY_TAGS[family]
            if host_tags_for_sizing is not None:
                host_family_mask = host_tags_for_sizing == np.int8(FAMILY_TAGS[family])
                family_has_rows = bool(np.any(host_family_mask))
                family_positions = cp.flatnonzero(family_mask).astype(cp.int64, copy=False)
            else:
                host_family_mask = None
                family_positions = cp.flatnonzero(family_mask).astype(cp.int64, copy=False)
                family_has_rows = int(family_positions.size) > 0
            if not family_has_rows:
                continue
            source_family_rows = d_state.family_row_offsets[d_indices[family_positions]]
            d_new_family_row_offsets[family_positions] = cp.arange(
                int(source_family_rows.size),
                dtype=cp.int32,
            )
            host_family_rows = None
            if host_family_mask is not None and host_family_rows_for_sizing is not None:
                host_family_rows = np.asarray(
                    host_family_rows_for_sizing[host_family_mask],
                    dtype=np.int64,
                )
            new_device_families[family] = _device_take_family_buffer(
                device_buffer,
                family,
                source_family_rows,
                self.families.get(family),
                host_family_rows=host_family_rows,
                allow_capacity_allocation=allow_capacity_allocation,
                assume_unique_indices=assume_unique_indices,
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
                trusted_unique_family_rows=True,
                trusted_family_domain=tuple(new_device_families),
            ),
            _row_count=int(d_indices.size),
        )
        if result.device_state is not None and len(new_device_families) == 1:
            family = next(iter(new_device_families))
            source_all_rows_valid = d_state.trusted_all_valid is True or (
                self._validity is not None
                and int(self._validity.size) == int(self.row_count)
                and bool(np.all(self._validity))
            )
            if source_all_rows_valid:
                result.device_state.trusted_all_valid = True
            if d_state.trusted_all_ogc_valid is True:
                result.device_state.trusted_all_ogc_valid = True
            if d_state.trusted_homogeneous_family is family:
                result.device_state.trusted_homogeneous_family = family
                if d_state.trusted_all_non_empty is True:
                    result.device_state.trusted_all_non_empty = True
                if d_state.trusted_all_finite_coordinates is True:
                    result.device_state.trusted_all_finite_coordinates = True
        if result.device_state is not None:
            polygonal_families = {
                GeometryFamily.POLYGON,
                GeometryFamily.MULTIPOLYGON,
            }
            if (
                d_state.trusted_polygonal_only is True
                or set(new_device_families) <= polygonal_families
            ):
                result.device_state.trusted_polygonal_only = True
        if result.device_state is not None and d_state.row_bounds is not None:
            result.device_state.row_bounds = cp.asarray(d_state.row_bounds)[d_indices].reshape(
                int(d_indices.size), 4
            )
        result._record(
            DiagnosticKind.CREATED,
            f"device-side subset {n_indices} rows via device_take",
            visible=False,
        )
        return result

    def to_shapely(self) -> list[object | None]:
        from vibespatial.geometry.host_bridge import owned_to_shapely
        from vibespatial.runtime.materialization import (
            NativeExportBoundary,
            record_native_export_boundary,
        )

        result = list(owned_to_shapely(self))
        record_native_export_boundary(
            NativeExportBoundary(
                surface="vibespatial.geometry.owned.OwnedGeometryArray.to_shapely",
                operation="owned_geometry_to_shapely",
                target="shapely",
                reason="owned geometry exported to Shapely compatibility objects",
                detail=(
                    f"residency={getattr(getattr(self, 'residency', None), 'value', 'unknown')}"
                ),
                row_count=self.row_count,
                d2h_transfer=self.device_state is not None,
            )
        )
        return result

    def to_wkb(self, *, hex: bool = False) -> list[bytes | str | None]:
        from vibespatial.io.wkb import encode_wkb_owned

        return encode_wkb_owned(self, hex=hex)

    def to_geoarrow(
        self,
        *,
        sharing: BufferSharingMode | str = BufferSharingMode.COPY,
    ) -> MixedGeoArrowView:
        self._ensure_host_state(preserve_indexed_view=True)
        sharing_mode = normalize_buffer_sharing_mode(sharing)
        share = sharing_mode is not BufferSharingMode.COPY
        if share and self._cached_shared_geoarrow_view is not None:
            return self._cached_shared_geoarrow_view
        views = {
            family: GeoArrowBufferView(
                family=buffer.family,
                x=buffer.x if share else buffer.x.copy(),
                y=buffer.y if share else buffer.y.copy(),
                geometry_offsets=buffer.geometry_offsets
                if share
                else buffer.geometry_offsets.copy(),
                empty_mask=buffer.empty_mask if share else buffer.empty_mask.copy(),
                part_offsets=None
                if buffer.part_offsets is None
                else (buffer.part_offsets if share else buffer.part_offsets.copy()),
                ring_offsets=None
                if buffer.ring_offsets is None
                else (buffer.ring_offsets if share else buffer.ring_offsets.copy()),
                bounds=None
                if buffer.bounds is None
                else (buffer.bounds if share else buffer.bounds.copy()),
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
        view = MixedGeoArrowView(
            validity=self.validity if share else self.validity.copy(),
            tags=self.tags if share else self.tags.copy(),
            family_row_offsets=self.family_row_offsets if share else self.family_row_offsets.copy(),
            families=views,
            shares_memory=share,
        )
        if share:
            self._cached_shared_geoarrow_view = view
        return view


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

    if buffer.family in (
        GeometryFamily.POINT,
        GeometryFamily.LINESTRING,
        GeometryFamily.MULTIPOINT,
    ):
        coords, new_geom_offsets = _gather_offset_slices(
            np.column_stack([buffer.x, buffer.y]),
            buffer.geometry_offsets,
            family_rows,
        )
        new_x = np.ascontiguousarray(coords[:, 0]) if coords.size else np.empty(0, dtype=np.float64)
        new_y = np.ascontiguousarray(coords[:, 1]) if coords.size else np.empty(0, dtype=np.float64)
        return FamilyGeometryBuffer(
            family=buffer.family,
            schema=schema,
            row_count=family_rows.size,
            x=new_x,
            y=new_y,
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
            family=buffer.family,
            schema=schema,
            row_count=family_rows.size,
            x=new_x,
            y=new_y,
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
            family=buffer.family,
            schema=schema,
            row_count=family_rows.size,
            x=new_x,
            y=new_y,
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
            family=buffer.family,
            schema=schema,
            row_count=family_rows.size,
            x=new_x,
            y=new_y,
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


def _device_offset_terminal_counts(
    offset_arrays: list[DeviceArray],
) -> DeviceArray:
    return cp.concatenate([cp.asarray(offsets[-1:], dtype=cp.int64) for offsets in offset_arrays])


def _device_offset_values_at_counts(
    offset_arrays: list[DeviceArray],
    interval_counts: DeviceArray,
) -> DeviceArray:
    d_counts = cp.asarray(interval_counts, dtype=cp.int64)
    return cp.concatenate(
        [
            cp.asarray(offsets, dtype=cp.int32)[d_counts[index : index + 1]].astype(
                cp.int64,
                copy=False,
            )
            for index, offsets in enumerate(offset_arrays)
        ]
    )


def _device_count_starts(counts: DeviceArray) -> DeviceArray:
    d_counts = cp.asarray(counts, dtype=cp.int64)
    return cp.cumsum(d_counts, dtype=cp.int64) - d_counts


def _concat_device_offset_arrays(
    offset_arrays: list[DeviceArray],
    value_starts: DeviceArray,
    *,
    active_interval_counts: DeviceArray | None = None,
) -> DeviceArray:
    """Concatenate offsets using device-resident active-element starts."""
    if len(offset_arrays) == 1:
        return offset_arrays[0]
    d_value_starts = cp.asarray(value_starts, dtype=cp.int64)
    if active_interval_counts is not None:
        capacities = [max(int(offsets.size) - 1, 0) for offsets in offset_arrays]
        d_active_counts = cp.asarray(active_interval_counts, dtype=cp.int64)
        d_output_starts = _device_count_starts(d_active_counts)
        out_offsets = cp.empty(sum(capacities) + 1, dtype=cp.int32)
        last_index = len(offset_arrays) - 1
        d_terminal = cp.asarray(offset_arrays[last_index], dtype=cp.int32)[
            d_active_counts[last_index : last_index + 1]
        ] + d_value_starts[last_index : last_index + 1].astype(
            cp.int32,
            copy=False,
        )
        # Capacity consumers may inspect every physical interval.  Make the
        # inactive tail a monotonic sequence of zero-width spans before the
        # compact prefixes are scattered into place.
        out_offsets[...] = d_terminal[0]
        runtime = get_cuda_runtime()
        kernel = _owned_take_kernels()["owned_concat_compact_offsets_i32"]
        ptr = runtime.pointer
        for buffer_index, (offsets, capacity) in enumerate(
            zip(offset_arrays, capacities, strict=True)
        ):
            grid, block = runtime.launch_config(kernel, capacity + 1)
            runtime.launch(
                kernel,
                grid=grid,
                block=block,
                params=(
                    (
                        ptr(offsets),
                        ptr(d_active_counts),
                        ptr(d_output_starts),
                        ptr(d_value_starts),
                        ptr(out_offsets),
                        buffer_index,
                        capacity,
                    ),
                    (
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_I32,
                        KERNEL_PARAM_I64,
                    ),
                ),
            )
        return out_offsets

    parts: list[DeviceArray] = [cp.asarray(offset_arrays[0], dtype=cp.int64) + d_value_starts[0]]
    for index, offsets in enumerate(offset_arrays[1:], start=1):
        parts.append(cp.asarray(offsets[1:], dtype=cp.int64) + d_value_starts[index])
    return cp.concatenate(parts).astype(cp.int32)


def _concat_device_xy_compact(
    buffers: list[DeviceFamilyGeometryBuffer],
    active_coord_counts: DeviceArray,
) -> tuple[DeviceArray, DeviceArray]:
    """Compact active coordinate prefixes into one capacity-backed SoA pair."""
    capacities = [int(buffer.x.size) for buffer in buffers]
    total_capacity = sum(capacities)
    if total_capacity == 0:
        return cp.empty(0, dtype=cp.float64), cp.empty(0, dtype=cp.float64)
    d_active_counts = cp.asarray(active_coord_counts, dtype=cp.int64)
    d_output_starts = _device_count_starts(d_active_counts)
    out_x = cp.empty(total_capacity, dtype=cp.float64)
    out_y = cp.empty(total_capacity, dtype=cp.float64)
    runtime = get_cuda_runtime()
    kernel = _owned_take_kernels()["owned_concat_compact_xy_f64"]
    ptr = runtime.pointer
    for buffer_index, (buffer, capacity) in enumerate(zip(buffers, capacities, strict=True)):
        if capacity == 0:
            continue
        grid, block = runtime.launch_config(kernel, capacity)
        runtime.launch(
            kernel,
            grid=grid,
            block=block,
            params=(
                (
                    ptr(buffer.x),
                    ptr(buffer.y),
                    ptr(d_active_counts),
                    ptr(d_output_starts),
                    ptr(out_x),
                    ptr(out_y),
                    buffer_index,
                    capacity,
                ),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_I64,
                ),
            ),
        )
    return out_x, out_y


def _concat_xy_by_counts(
    buffers: list[FamilyGeometryBuffer],
    coord_counts: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    if not any(coord_counts):
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    return (
        np.concatenate(
            [
                buffer.x[:count]
                for buffer, count in zip(buffers, coord_counts, strict=True)
                if count > 0
            ]
        ),
        np.concatenate(
            [
                buffer.y[:count]
                for buffer, count in zip(buffers, coord_counts, strict=True)
                if count > 0
            ]
        ),
    )


def _regular_grid_total_bounds(
    *,
    origin_x: float,
    origin_y: float,
    cell_width: float,
    cell_height: float,
    cols: int,
    size: int,
) -> tuple[float, float, float, float]:
    """Return total bounds for a row-major regular grid prefix."""
    if size <= 0 or cols <= 0:
        return (float("nan"),) * 4
    full_rows, tail_cols = divmod(int(size), int(cols))
    rows = full_rows + (1 if tail_cols else 0)
    max_cols = int(cols) if full_rows > 0 else int(tail_cols)
    return (
        float(origin_x),
        float(origin_y),
        float(origin_x + max_cols * cell_width),
        float(origin_y + rows * cell_height),
    )


def _regular_grid_rows_for_size(size: int, cols: int) -> int:
    if size <= 0 or cols <= 0:
        return 0
    return (int(size) + int(cols) - 1) // int(cols)


def _regular_grid_scalar_close(left: float, right: float) -> bool:
    scale = max(abs(float(left)), abs(float(right)), 1.0)
    return abs(float(left) - float(right)) <= 1e-12 * scale


def _concat_regular_grid_rect_proofs(
    buffers: list[DeviceFamilyGeometryBuffer],
) -> DeviceRegularGridRectMetadata | None:
    """Preserve regular-grid proofs across row-aligned device concat."""
    if not buffers:
        return None
    proofs = [getattr(buffer, "regular_grid_rect", None) for buffer in buffers]
    if any(proof is None for proof in proofs):
        return None
    first = proofs[0]
    assert first is not None
    cols = int(first.cols)
    cell_width = float(first.cell_width)
    cell_height = float(first.cell_height)
    origin_x = float(first.origin_x)
    origin_y = float(first.origin_y)
    if cols <= 0 or cell_width <= 0.0 or cell_height <= 0.0:
        return None

    total_size = 0
    for index, proof in enumerate(proofs):
        assert proof is not None
        size = int(proof.size)
        if size < 0:
            return None
        expected_y = origin_y + (total_size // cols) * cell_height
        if (
            int(proof.cols) != cols
            or not _regular_grid_scalar_close(float(proof.cell_width), cell_width)
            or not _regular_grid_scalar_close(float(proof.cell_height), cell_height)
            or not _regular_grid_scalar_close(float(proof.origin_x), origin_x)
            or not _regular_grid_scalar_close(float(proof.origin_y), expected_y)
        ):
            return None
        if index < len(proofs) - 1 and size % cols != 0:
            return None
        total_size += size

    total_bounds = _regular_grid_total_bounds(
        origin_x=origin_x,
        origin_y=origin_y,
        cell_width=cell_width,
        cell_height=cell_height,
        cols=cols,
        size=total_size,
    )
    return DeviceRegularGridRectMetadata(
        origin_x=origin_x,
        origin_y=origin_y,
        cell_width=cell_width,
        cell_height=cell_height,
        cols=cols,
        rows=_regular_grid_rows_for_size(total_size, cols),
        size=total_size,
        total_bounds=total_bounds,
    )


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

    total_rows = sum(
        int(b.geometry_offsets.size) - 1 for b in buffers if b.geometry_offsets.size > 0
    )
    if total_rows == 0:
        return buffers[0]

    new_empty_mask = cp.concatenate([b.empty_mask for b in buffers])

    # Bounds: concatenate if all have bounds, otherwise drop.
    if all(b.bounds is not None for b in buffers):
        new_bounds = cp.concatenate([b.bounds for b in buffers])
    else:
        new_bounds = None
    fixed_size = _common_device_fixed_size_metadata(family, buffers)

    if family in (GeometryFamily.POINT, GeometryFamily.LINESTRING, GeometryFamily.MULTIPOINT):
        # Single level of offsets: geometry_offsets -> coords
        coord_counts = _device_offset_terminal_counts([b.geometry_offsets for b in buffers])
        new_x, new_y = _concat_device_xy_compact(buffers, coord_counts)
        new_geom_offsets = _concat_device_offset_arrays(
            [b.geometry_offsets for b in buffers],
            _device_count_starts(coord_counts),
        )
        return DeviceFamilyGeometryBuffer(
            family=family,
            x=new_x,
            y=new_y,
            geometry_offsets=new_geom_offsets,
            empty_mask=new_empty_mask,
            bounds=new_bounds,
            fixed_size=fixed_size,
        )

    if family is GeometryFamily.POLYGON:
        # Two levels: geometry_offsets -> ring_offsets -> coords
        geom_ring_counts = _device_offset_terminal_counts([b.geometry_offsets for b in buffers])
        ring_counts = _device_offset_values_at_counts(
            [b.ring_offsets for b in buffers],
            geom_ring_counts,
        )
        dense_widths = {b.dense_single_ring_width for b in buffers}
        dense_width = dense_widths.pop() if len(dense_widths) == 1 else None
        shape_buffers = [buffer for buffer in buffers if int(buffer.x.size) > 0]
        axis_aligned_rectangles = bool(
            shape_buffers and all(buffer.axis_aligned_rectangles for buffer in shape_buffers)
        )
        regular_grid_rect = _concat_regular_grid_rect_proofs(buffers)
        new_x, new_y = _concat_device_xy_compact(buffers, ring_counts)
        new_ring_offsets = _concat_device_offset_arrays(
            [b.ring_offsets for b in buffers],
            _device_count_starts(ring_counts),
            active_interval_counts=geom_ring_counts,
        )
        new_geom_offsets = _concat_device_offset_arrays(
            [b.geometry_offsets for b in buffers],
            _device_count_starts(geom_ring_counts),
        )
        return DeviceFamilyGeometryBuffer(
            family=family,
            x=new_x,
            y=new_y,
            geometry_offsets=new_geom_offsets,
            empty_mask=new_empty_mask,
            ring_offsets=new_ring_offsets,
            bounds=new_bounds,
            dense_single_ring_width=dense_width,
            axis_aligned_rectangles=axis_aligned_rectangles,
            regular_grid_rect=regular_grid_rect,
            fixed_size=fixed_size,
        )

    if family is GeometryFamily.MULTILINESTRING:
        # Two levels: geometry_offsets -> part_offsets -> coords
        geom_part_counts = _device_offset_terminal_counts([b.geometry_offsets for b in buffers])
        coord_counts = _device_offset_values_at_counts(
            [b.part_offsets for b in buffers],
            geom_part_counts,
        )
        new_x, new_y = _concat_device_xy_compact(buffers, coord_counts)
        new_part_offsets = _concat_device_offset_arrays(
            [b.part_offsets for b in buffers],
            _device_count_starts(coord_counts),
            active_interval_counts=geom_part_counts,
        )
        new_geom_offsets = _concat_device_offset_arrays(
            [b.geometry_offsets for b in buffers],
            _device_count_starts(geom_part_counts),
        )
        return DeviceFamilyGeometryBuffer(
            family=family,
            x=new_x,
            y=new_y,
            geometry_offsets=new_geom_offsets,
            empty_mask=new_empty_mask,
            part_offsets=new_part_offsets,
            bounds=new_bounds,
            fixed_size=fixed_size,
        )

    if family is GeometryFamily.MULTIPOLYGON:
        # Three levels: geometry_offsets -> part_offsets -> ring_offsets -> coords
        geom_part_counts = _device_offset_terminal_counts([b.geometry_offsets for b in buffers])
        part_ring_counts = _device_offset_values_at_counts(
            [b.part_offsets for b in buffers],
            geom_part_counts,
        )
        ring_coord_counts = _device_offset_values_at_counts(
            [b.ring_offsets for b in buffers],
            part_ring_counts,
        )
        new_x, new_y = _concat_device_xy_compact(buffers, ring_coord_counts)
        new_ring_offsets = _concat_device_offset_arrays(
            [b.ring_offsets for b in buffers],
            _device_count_starts(ring_coord_counts),
            active_interval_counts=part_ring_counts,
        )
        new_part_offsets = _concat_device_offset_arrays(
            [b.part_offsets for b in buffers],
            _device_count_starts(part_ring_counts),
            active_interval_counts=geom_part_counts,
        )
        new_geom_offsets = _concat_device_offset_arrays(
            [b.geometry_offsets for b in buffers],
            _device_count_starts(geom_part_counts),
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
            fixed_size=fixed_size,
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

    new_empty_mask = np.concatenate([b.empty_mask for b in buffers])

    # Bounds: concatenate if all have bounds, otherwise drop.
    if all(b.bounds is not None for b in buffers):
        new_bounds = np.concatenate([b.bounds for b in buffers])
    else:
        new_bounds = None

    if family in (GeometryFamily.POINT, GeometryFamily.LINESTRING, GeometryFamily.MULTIPOINT):
        # Single level of offsets: geometry_offsets -> coords
        coord_counts = [
            int(b.geometry_offsets[-1]) if b.geometry_offsets.size > 0 else 0 for b in buffers
        ]
        new_x, new_y = _concat_xy_by_counts(buffers, coord_counts)
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
        geom_ring_counts = [
            int(b.geometry_offsets[-1]) if b.geometry_offsets.size > 0 else 0 for b in buffers
        ]
        new_x, new_y = _concat_xy_by_counts(buffers, ring_counts)
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
        geom_part_counts = [
            int(b.geometry_offsets[-1]) if b.geometry_offsets.size > 0 else 0 for b in buffers
        ]
        new_x, new_y = _concat_xy_by_counts(buffers, coord_counts)
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
        ring_coord_counts = [
            int(b.ring_offsets[-1]) if b.ring_offsets.size > 0 else 0 for b in buffers
        ]
        part_ring_counts = [
            int(b.part_offsets[-1]) if b.part_offsets.size > 0 else 0 for b in buffers
        ]
        geom_part_counts = [
            int(b.geometry_offsets[-1]) if b.geometry_offsets.size > 0 else 0 for b in buffers
        ]
        new_x, new_y = _concat_xy_by_counts(buffers, ring_coord_counts)
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
    *,
    precomputed_total: int | None = None,
    allocation_capacity: int | None = None,
    active_row_count: object | None = None,
    allocation_reason: str | None = None,
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
    if active_row_count is not None:
        active = cp.arange(int(rows.size), dtype=cp.int64) < cp.asarray(
            active_row_count,
            dtype=cp.int64,
        )
        lengths = cp.where(active, lengths, cp.zeros((), dtype=cp.int32))

    n = int(rows.size)
    new_offsets = cp.empty(n + 1, dtype=cp.int32)
    new_offsets[0] = 0
    if n > 0:
        cp.cumsum(lengths, out=new_offsets[1:])
    if precomputed_total is not None:
        total_length = int(precomputed_total)
    elif allocation_capacity is not None:
        total_length = int(allocation_capacity)
    else:
        if allocation_reason is None:
            raise ValueError(
                "unknown-size offset-slice gather requires an explicit allocation boundary reason"
            )
        total_length = count_scatter_total(
            get_cuda_runtime(),
            lengths,
            new_offsets[:-1],
            reason=allocation_reason,
        )

    if total_length == 0:
        if data.ndim == 2:
            gathered = cp.empty((0, data.shape[1]), dtype=data.dtype)
        else:
            gathered = cp.empty(0, dtype=data.dtype)
        return gathered, new_offsets

    if allocation_capacity is not None and precomputed_total is None:
        if data.ndim == 1 and data.dtype == cp.int32:
            gathered = cp.zeros(total_length, dtype=cp.int32)
            kernel_name = "owned_take_gather_values_i32"
        elif data.ndim == 2 and int(data.shape[1]) == 2 and data.dtype == cp.float64:
            gathered = cp.zeros((total_length, 2), dtype=cp.float64)
            kernel_name = "owned_take_gather_values_f64x2"
        else:
            raise TypeError(
                "capacity offset-slice gather supports int32 vectors or two-column float64 arrays"
            )
        if n == 0:
            return gathered, new_offsets

        runtime = get_cuda_runtime()
        kernel = _owned_take_kernels()[kernel_name]
        block_size = runtime.optimal_block_size(kernel)
        ptr = runtime.pointer
        runtime.launch(
            kernel,
            grid=(n, 1, 1),
            block=(block_size, 1, 1),
            params=(
                (
                    ptr(data),
                    ptr(starts),
                    ptr(lengths),
                    ptr(new_offsets),
                    ptr(gathered),
                    n,
                ),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                ),
            ),
        )
        return gathered, new_offsets

    # Build flat gather indices fully on device using searchsorted.
    # segment_ids[j] = i means position j belongs to row i.
    # searchsorted(new_offsets, j, side='right') - 1 gives the segment.
    positions = cp.arange(total_length, dtype=cp.int64)
    segment_ids = cp.searchsorted(new_offsets, positions, side="right").astype(cp.int64) - 1
    flat_indices = (
        positions - new_offsets[segment_ids].astype(cp.int64) + starts[segment_ids].astype(cp.int64)
    )

    gathered = cp.empty(
        (total_length, data.shape[1]) if data.ndim == 2 else (total_length,),
        dtype=data.dtype,
    )
    if int(flat_indices.size):
        gathered[: int(flat_indices.size)] = data[flat_indices]
    return gathered, new_offsets


@dataclass(frozen=True)
class _DeviceTakeFamilySizePlan:
    """Logical totals and physical capacities for a nested device take.

    Count fields are exact logical totals.  Capacity fields are allocation
    bounds only; a capacity-backed parent must propagate its device terminal
    as the active-row count for the next nested level.
    """

    first_level_count: int | None = None
    second_level_count: int | None = None
    coord_count: int | None = None
    first_level_capacity: int | None = None
    second_level_capacity: int | None = None
    coord_capacity: int | None = None


def _device_take_size_plan_as_capacity(
    plan: _DeviceTakeFamilySizePlan,
) -> _DeviceTakeFamilySizePlan:
    """Demote full-row exact totals when a device mask can zero row spans."""
    return _DeviceTakeFamilySizePlan(
        first_level_capacity=(
            plan.first_level_count
            if plan.first_level_count is not None
            else plan.first_level_capacity
        ),
        second_level_capacity=(
            plan.second_level_count
            if plan.second_level_count is not None
            else plan.second_level_capacity
        ),
        coord_capacity=(plan.coord_count if plan.coord_count is not None else plan.coord_capacity),
    )


@dataclass
class _ExactDeviceRowSelection:
    base: OwnedGeometryArray
    indices: DeviceArray
    active: DeviceArray
    validity: DeviceArray
    tags: DeviceArray
    family_rows: dict[GeometryFamily, DeviceArray]
    family_active: dict[GeometryFamily, DeviceArray]
    family_stats: dict[GeometryFamily, DeviceArray]


def _flatten_exact_device_row_selection(
    owned: OwnedGeometryArray,
    active_mask: DeviceArray | None,
) -> _ExactDeviceRowSelection:
    """Resolve logical row indirection while retaining a device activity mask."""
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        raise RuntimeError("CuPy is required for exact device row physicalization")

    row_count = int(owned.row_count)
    d_active = (
        cp.ones(row_count, dtype=cp.bool_)
        if active_mask is None
        else cp.asarray(active_mask, dtype=cp.bool_)
    )
    if d_active.ndim != 1 or int(d_active.size) != row_count:
        raise ValueError("exact physicalization activity must match logical rows")

    current = owned
    d_indices = cp.arange(row_count, dtype=cp.int64)
    if current._row_active_mask is not None:
        d_active &= cp.asarray(current._row_active_mask, dtype=cp.bool_)
    while current.is_indexed_view:
        base = current._base
        index_map = current._index_map
        if base is None or index_map is None or not hasattr(index_map, "__cuda_array_interface__"):
            raise RuntimeError("exact device physicalization requires device row maps")
        d_indices = cp.asarray(index_map, dtype=cp.int64)[d_indices]
        current = base
        if current._row_active_mask is not None:
            d_active &= cp.asarray(current._row_active_mask, dtype=cp.bool_)[d_indices]

    state = current._ensure_device_state(preserve_indexed_view=True)
    d_source_validity = cp.asarray(state.validity, dtype=cp.bool_)[d_indices]
    d_validity = d_active & d_source_validity
    d_source_tags = cp.asarray(state.tags, dtype=cp.int8)[d_indices]
    d_tags = cp.where(d_validity, d_source_tags, cp.int8(NULL_TAG))
    d_source_family_rows = cp.asarray(state.family_row_offsets, dtype=cp.int64)[d_indices]

    family_rows: dict[GeometryFamily, DeviceArray] = {}
    family_active: dict[GeometryFamily, DeviceArray] = {}
    family_stats: dict[GeometryFamily, DeviceArray] = {}

    def _maximum(values):
        return (
            cp.asarray(0, dtype=cp.int64)
            if int(values.size) == 0
            else cp.max(values).astype(cp.int64, copy=False)
        )

    for family, buffer in state.families.items():
        d_family_active = d_validity & (d_tags == np.int8(FAMILY_TAGS[family]))
        d_family_rows = cp.where(
            d_family_active,
            d_source_family_rows,
            cp.int64(0),
        )
        d_geom_starts = cp.asarray(buffer.geometry_offsets, dtype=cp.int64)[d_family_rows]
        d_geom_ends = cp.asarray(buffer.geometry_offsets, dtype=cp.int64)[d_family_rows + 1]
        d_first_counts = cp.where(
            d_family_active,
            d_geom_ends - d_geom_starts,
            cp.int64(0),
        )
        d_second_counts = cp.zeros(row_count, dtype=cp.int64)
        if family in (
            GeometryFamily.POINT,
            GeometryFamily.LINESTRING,
            GeometryFamily.MULTIPOINT,
        ):
            d_coord_counts = d_first_counts
            d_first_counts = cp.zeros(row_count, dtype=cp.int64)
        elif family is GeometryFamily.POLYGON:
            d_ring_offsets = cp.asarray(buffer.ring_offsets, dtype=cp.int64)
            d_coord_counts = cp.where(
                d_family_active,
                d_ring_offsets[d_geom_ends] - d_ring_offsets[d_geom_starts],
                cp.int64(0),
            )
        elif family is GeometryFamily.MULTILINESTRING:
            d_part_offsets = cp.asarray(buffer.part_offsets, dtype=cp.int64)
            d_coord_counts = cp.where(
                d_family_active,
                d_part_offsets[d_geom_ends] - d_part_offsets[d_geom_starts],
                cp.int64(0),
            )
        elif family is GeometryFamily.MULTIPOLYGON:
            d_part_offsets = cp.asarray(buffer.part_offsets, dtype=cp.int64)
            d_ring_offsets = cp.asarray(buffer.ring_offsets, dtype=cp.int64)
            d_part_starts = d_part_offsets[d_geom_starts]
            d_part_ends = d_part_offsets[d_geom_ends]
            d_second_counts = cp.where(
                d_family_active,
                d_part_ends - d_part_starts,
                cp.int64(0),
            )
            d_coord_counts = cp.where(
                d_family_active,
                d_ring_offsets[d_part_ends] - d_ring_offsets[d_part_starts],
                cp.int64(0),
            )
        else:  # pragma: no cover - exhaustive over GeometryFamily
            raise NotImplementedError(f"exact physicalization for {family.value}")

        family_rows[family] = d_family_rows
        family_active[family] = d_family_active
        family_stats[family] = cp.stack(
            (
                cp.sum(d_family_active, dtype=cp.int64),
                cp.sum(d_first_counts, dtype=cp.int64),
                cp.sum(d_second_counts, dtype=cp.int64),
                cp.sum(d_coord_counts, dtype=cp.int64),
                _maximum(d_first_counts),
                _maximum(d_second_counts),
                _maximum(d_coord_counts),
            )
        )

    return _ExactDeviceRowSelection(
        base=current,
        indices=d_indices,
        active=d_active,
        validity=d_validity,
        tags=d_tags,
        family_rows=family_rows,
        family_active=family_active,
        family_stats=family_stats,
    )


def device_physicalize_owned_row_selections_exact(
    selections: list[tuple[OwnedGeometryArray, DeviceArray | None]],
    *,
    reason: str,
    compact_concrete_prefix: bool = False,
    materialize_all_null: bool = False,
) -> list[OwnedGeometryArray | None]:
    """Gather several logical row selections through one exact-allocation packet.

    This is the physical-layout boundary for multi-root native compositions.
    Coordinates and nested offsets remain on device; only aggregate allocation
    totals cross once so each selected span is copied exactly once. When
    ``compact_concrete_prefix`` is true, every active lane must be concrete and
    active lanes must form a prefix. The result then uses that prefix's exact
    logical row count instead of retaining source capacity. A ``None`` activity
    mask means all logical rows and lets this boundary admit its own mask before
    allocating it. ``materialize_all_null`` returns an all-null device carrier
    instead of ``None`` when a selection has no active geometry families.
    """
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        raise RuntimeError("CuPy is required for exact device row physicalization")

    planning_required_bytes = 0
    planning_rows = 0
    packet_part_count = 0
    for owned, active_mask in selections:
        row_count = int(owned.row_count)
        current = owned
        while current.is_indexed_view:
            if current._base is None:
                raise RuntimeError("exact device physicalization requires a base carrier")
            current = current._base
        state = current._ensure_device_state(preserve_indexed_view=True)
        family_count = len(state.families)
        planning_rows += row_count
        packet_part_count += family_count + int(compact_concrete_prefix)
        # Per row, flattening retains the resolved row map and logical
        # validity/tag arrays, plus one row map and activity mask per family.
        # The additional 64 bytes per family cover the widest nested-offset
        # count pass while those retained vectors are live.  This admission is
        # intentionally conservative and occurs before any of those arrays.
        planning_required_bytes += row_count * (32 + 80 * family_count)
        if active_mask is None:
            planning_required_bytes += row_count * np.dtype(np.bool_).itemsize
    planning_required_bytes += packet_part_count * 7 * np.dtype(np.int64).itemsize
    planning_admission = get_cuda_runtime().admit_device_memory(
        stage="geometry.exact_row_physicalization.plan",
        required_bytes=planning_required_bytes,
        requested_units=planning_rows,
    )
    if not planning_admission.admitted:
        raise MemoryError(
            "exact device row physicalization planning requires "
            f"{planning_required_bytes} device bytes with "
            f"{planning_admission.remaining_bytes} available"
        )

    prepared = [_flatten_exact_device_row_selection(owned, active) for owned, active in selections]
    work_keys = [
        (selection_index, family)
        for selection_index, selection in enumerate(prepared)
        for family in selection.family_stats
    ]
    prefix_stats = []
    if compact_concrete_prefix:
        prefix_stats = [
            cp.concatenate(
                (
                    cp.stack(
                        (
                            cp.sum(selection.active, dtype=cp.int64),
                            cp.sum(selection.validity, dtype=cp.int64),
                        )
                    ),
                    cp.zeros(5, dtype=cp.int64),
                )
            )
            for selection in prepared
        ]
    packet_parts = prefix_stats + [
        prepared[index].family_stats[family] for index, family in work_keys
    ]
    if packet_parts:
        d_packet = cp.concatenate(packet_parts)
        h_packet = np.asarray(
            get_cuda_runtime().copy_device_to_host(d_packet, reason=reason),
            dtype=np.int64,
        ).reshape(len(packet_parts), 7)
    else:
        h_packet = np.empty((0, 7), dtype=np.int64)
    prefix_offset = len(prefix_stats)
    host_stats = {
        key: h_packet[prefix_offset + position]
        for position, key in enumerate(work_keys)
    }

    row_counts: list[int] = []
    required_bytes = 0
    requested_rows = 0
    for selection_index, selection in enumerate(prepared):
        capacity = int(selection.indices.size)
        row_count = capacity
        if compact_concrete_prefix:
            active_count = int(h_packet[selection_index, 0])
            valid_count = int(h_packet[selection_index, 1])
            family_count = sum(
                int(host_stats[(selection_index, family)][0])
                for family in selection.family_stats
            )
            if active_count != valid_count or valid_count != family_count:
                raise ValueError(
                    "compact exact physicalization requires concrete active lanes"
                )
            row_count = active_count
        row_counts.append(row_count)
        requested_rows += row_count

        source_state = selection.base._ensure_device_state(preserve_indexed_view=True)
        # The exact count packet is also the memory-admission boundary.  Count
        # the physical result plus conservative gather scratch before any
        # coordinate-shaped output allocation occurs.
        if materialize_all_null and not any(
            int(host_stats[(selection_index, family)][0])
            for family in selection.family_stats
        ):
            required_bytes += row_count * (
                np.dtype(np.bool_).itemsize
                + np.dtype(np.int8).itemsize
                + np.dtype(np.int32).itemsize
            )
        else:
            required_bytes += row_count * np.dtype(np.int32).itemsize
        if source_state.row_bounds is not None:
            required_bytes += 2 * row_count * 4 * np.dtype(np.float64).itemsize
        for family in selection.family_stats:
            (
                active_count,
                first_total,
                second_total,
                coord_total,
                _max_first,
                _max_second,
                _max_coord,
            ) = (int(value) for value in host_stats[(selection_index, family)])
            if active_count == 0:
                continue
            buffer = source_state.families[family]
            required_bytes += 2 * coord_total * np.dtype(np.float64).itemsize
            required_bytes += (row_count + 1) * np.dtype(np.int32).itemsize
            required_bytes += 2 * row_count * np.dtype(np.bool_).itemsize
            required_bytes += 12 * row_count
            if buffer.bounds is not None:
                required_bytes += 2 * row_count * 4 * np.dtype(np.float64).itemsize
            if family in (GeometryFamily.POLYGON, GeometryFamily.MULTILINESTRING):
                required_bytes += (first_total + 1) * np.dtype(np.int32).itemsize
                required_bytes += 16 * first_total
            elif family is GeometryFamily.MULTIPOLYGON:
                required_bytes += (first_total + 1) * np.dtype(np.int32).itemsize
                required_bytes += (second_total + 1) * np.dtype(np.int32).itemsize
                required_bytes += 16 * first_total + 16 * second_total

    admission = get_cuda_runtime().admit_device_memory(
        stage="geometry.exact_row_physicalization",
        required_bytes=required_bytes,
        requested_units=requested_rows,
    )
    if not admission.admitted:
        raise MemoryError(
            "exact device row physicalization requires "
            f"{required_bytes} device bytes with {admission.remaining_bytes} available"
        )

    results: list[OwnedGeometryArray | None] = []
    for selection_index, selection in enumerate(prepared):
        capacity = int(selection.indices.size)
        row_count = row_counts[selection_index]
        device_families: dict[GeometryFamily, DeviceFamilyGeometryBuffer] = {}
        d_family_row_offsets = cp.full(row_count, -1, dtype=cp.int32)
        segment_bound = 0
        for family, d_family_rows in selection.family_rows.items():
            (
                active_count,
                first_total,
                second_total,
                coord_total,
                max_first,
                max_second,
                max_coord,
            ) = (int(value) for value in host_stats[(selection_index, family)])
            if active_count == 0:
                continue
            if family in (
                GeometryFamily.POINT,
                GeometryFamily.LINESTRING,
                GeometryFamily.MULTIPOINT,
            ):
                size_plan = _DeviceTakeFamilySizePlan(coord_count=coord_total)
                fixed_size = DeviceFixedGeometrySizeMetadata(
                    max_coord_count_per_row=max_coord,
                )
            elif family in (GeometryFamily.POLYGON, GeometryFamily.MULTILINESTRING):
                size_plan = _DeviceTakeFamilySizePlan(
                    first_level_count=first_total,
                    coord_count=coord_total,
                )
                fixed_size = DeviceFixedGeometrySizeMetadata(
                    max_first_level_count_per_row=max_first,
                    max_coord_count_per_row=max_coord,
                )
            else:
                size_plan = _DeviceTakeFamilySizePlan(
                    first_level_count=first_total,
                    second_level_count=second_total,
                    coord_count=coord_total,
                )
                fixed_size = DeviceFixedGeometrySizeMetadata(
                    max_first_level_count_per_row=max_first,
                    max_second_level_count_per_row=max_second,
                    max_coord_count_per_row=max_coord,
                )
            d_family_rows = d_family_rows[:row_count]
            d_family_active = selection.family_active[family][:row_count]
            d_family_row_offsets = cp.where(
                d_family_active,
                cp.arange(row_count, dtype=cp.int32),
                d_family_row_offsets,
            )
            source_state = selection.base._ensure_device_state(preserve_indexed_view=True)
            device_families[family] = _device_take_family_buffer(
                source_state.families[family],
                family,
                d_family_rows,
                selection.base.families.get(family),
                active_row_mask=d_family_active,
                exact_size_plan=size_plan,
                output_fixed_size=fixed_size,
            )
            segment_bound = max(segment_bound, max_coord)

        if not device_families:
            if materialize_all_null:
                result = build_device_resident_owned(
                    device_families={},
                    row_count=row_count,
                    tags=selection.tags[:row_count],
                    validity=selection.validity[:row_count],
                    family_row_offsets=d_family_row_offsets,
                    execution_mode="gpu",
                )
                result._record(
                    DiagnosticKind.MATERIALIZATION,
                    (
                        "exact device row physicalization: "
                        f"{row_count} null rows from {capacity} capacity lanes"
                    ),
                    visible=False,
                )
                results.append(result)
            else:
                results.append(None)
            continue
        result = build_device_resident_owned(
            device_families=device_families,
            row_count=row_count,
            tags=selection.tags[:row_count],
            validity=selection.validity[:row_count],
            family_row_offsets=d_family_row_offsets,
            execution_mode="gpu",
        )
        result_state = result._ensure_device_state(preserve_indexed_view=True)
        source_state = selection.base._ensure_device_state(preserve_indexed_view=True)
        result_state.trusted_unique_family_rows = True
        if source_state.trusted_all_ogc_valid is True:
            result_state.trusted_all_ogc_valid = True
        if source_state.row_bounds is not None:
            d_bounds = cp.asarray(source_state.row_bounds, dtype=cp.float64).reshape(
                selection.base.row_count,
                4,
            )[selection.indices[:row_count]]
            result_state.row_bounds = cp.where(
                selection.validity[:row_count, None],
                d_bounds,
                cp.asarray(cp.nan, dtype=cp.float64),
            )
        result._active_family_row_segment_capacity_bound = segment_bound
        result._record(
            DiagnosticKind.MATERIALIZATION,
            (
                "exact device row physicalization: "
                f"{row_count} rows from {capacity} capacity lanes"
            ),
            visible=False,
        )
        results.append(result)
    return results


def device_physicalize_owned_row_selection_capacity(
    owned: OwnedGeometryArray,
    active_mask: DeviceArray,
) -> OwnedGeometryArray:
    """Gather a masked rowset into a bounded physical device carrier.

    The row capacity and family width metadata size every allocation on the
    host. Dynamic activity remains a device mask, so this transition does not
    export selected counts or nested offset totals. The identity row map is
    injective by construction and each active coordinate span is copied once.
    """
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        raise RuntimeError("CuPy is required for device capacity physicalization")
    if owned.residency is not Residency.DEVICE:
        raise RuntimeError("capacity physicalization requires device geometry")

    row_count = int(owned.row_count)
    d_active = cp.asarray(active_mask, dtype=cp.bool_)
    if d_active.ndim != 1 or int(d_active.size) != row_count:
        raise ValueError("capacity physicalization activity must match logical rows")

    flattened = _flatten_exact_device_row_selection(owned, d_active)
    base_state = flattened.base._ensure_device_state(preserve_indexed_view=True)
    preserve_fixed_widths = all(
        _device_buffer_has_exact_row_width(family, buffer)
        for family, buffer in base_state.families.items()
    )
    if preserve_fixed_widths:
        result = flattened.base._physical_device_take(
            flattened.indices,
            allow_capacity_allocation=True,
        )
        result = device_mask_owned_capacity(result, flattened.active)
    else:
        selected = owned._device_indexed_take(
            cp.arange(row_count, dtype=cp.int64),
            assume_unique_indices=True,
        )._apply_row_activity(
            d_active,
            assume_active_indices_unique=True,
        )
        result = selected.physicalize_device_rows(allow_capacity_allocation=True)
    result._record(
        DiagnosticKind.MATERIALIZATION,
        f"bounded device row physicalization: {row_count} row-capacity lanes",
        visible=False,
    )
    return result


def _host_offset_slice_total(
    offsets: np.ndarray | None,
    rows: np.ndarray | None,
) -> int | None:
    if offsets is None or rows is None:
        return None
    host_offsets = np.asarray(offsets, dtype=np.int64)
    host_rows = np.asarray(rows, dtype=np.int64)
    if host_rows.size == 0:
        return 0
    if host_offsets.size == 0:
        return None
    if int(host_rows.min()) < 0 or int(host_rows.max()) + 1 >= host_offsets.size:
        return None
    starts = host_offsets[host_rows]
    ends = host_offsets[host_rows + 1]
    if bool(np.any(ends < starts)):
        return None
    return int(np.sum(ends - starts, dtype=np.int64))


def _host_nested_offset_total(
    outer_offsets: np.ndarray | None,
    inner_offsets: np.ndarray | None,
    rows: np.ndarray | None,
) -> int | None:
    if outer_offsets is None or inner_offsets is None or rows is None:
        return None
    outer = np.asarray(outer_offsets, dtype=np.int64)
    inner = np.asarray(inner_offsets, dtype=np.int64)
    host_rows = np.asarray(rows, dtype=np.int64)
    if host_rows.size == 0:
        return 0
    if outer.size == 0 or inner.size == 0:
        return None
    if int(host_rows.min()) < 0 or int(host_rows.max()) + 1 >= outer.size:
        return None
    outer_starts = outer[host_rows]
    outer_ends = outer[host_rows + 1]
    if bool(np.any(outer_ends < outer_starts)):
        return None
    if int(outer_starts.min(initial=0)) < 0 or int(outer_ends.max(initial=0)) >= inner.size:
        return None
    inner_starts = inner[outer_starts]
    inner_ends = inner[outer_ends]
    if bool(np.any(inner_ends < inner_starts)):
        return None
    return int(np.sum(inner_ends - inner_starts, dtype=np.int64))


def _host_three_level_offset_total(
    outer_offsets: np.ndarray | None,
    middle_offsets: np.ndarray | None,
    inner_offsets: np.ndarray | None,
    rows: np.ndarray | None,
) -> int | None:
    if outer_offsets is None or middle_offsets is None or inner_offsets is None or rows is None:
        return None
    outer = np.asarray(outer_offsets, dtype=np.int64)
    middle = np.asarray(middle_offsets, dtype=np.int64)
    inner = np.asarray(inner_offsets, dtype=np.int64)
    host_rows = np.asarray(rows, dtype=np.int64)
    if host_rows.size == 0:
        return 0
    if outer.size == 0 or middle.size == 0 or inner.size == 0:
        return None
    if int(host_rows.min()) < 0 or int(host_rows.max()) + 1 >= outer.size:
        return None
    outer_starts = outer[host_rows]
    outer_ends = outer[host_rows + 1]
    if bool(np.any(outer_ends < outer_starts)):
        return None
    if int(outer_starts.min(initial=0)) < 0 or int(outer_ends.max(initial=0)) >= middle.size:
        return None
    middle_starts = middle[outer_starts]
    middle_ends = middle[outer_ends]
    if bool(np.any(middle_ends < middle_starts)):
        return None
    if int(middle_starts.min(initial=0)) < 0 or int(middle_ends.max(initial=0)) >= inner.size:
        return None
    inner_starts = inner[middle_starts]
    inner_ends = inner[middle_ends]
    if bool(np.any(inner_ends < inner_starts)):
        return None
    return int(np.sum(inner_ends - inner_starts, dtype=np.int64))


def _host_fixed_i64_offsets(
    offsets: np.ndarray | None,
    *,
    expected_size: int | None = None,
) -> np.ndarray | None:
    if offsets is None:
        return None
    host_offsets = np.asarray(offsets, dtype=np.int64)
    if expected_size is not None and int(host_offsets.size) != int(expected_size):
        return None
    if host_offsets.ndim != 1 or host_offsets.size == 0:
        return None
    if int(host_offsets[0]) != 0:
        return None
    if bool(np.any(host_offsets[1:] < host_offsets[:-1])):
        return None
    return host_offsets


def _host_fixed_offset_span(
    offsets: np.ndarray | None,
    parent_count: int,
    *,
    terminal_size: int | None = None,
) -> int | None:
    host_offsets = _host_fixed_i64_offsets(
        offsets,
        expected_size=int(parent_count) + 1,
    )
    if host_offsets is None:
        return None
    if terminal_size is not None and int(host_offsets[-1]) != int(terminal_size):
        return None
    if int(parent_count) == 0:
        return 0
    spans = host_offsets[1:] - host_offsets[:-1]
    span = int(spans[0])
    if span < 0 or not bool(np.all(spans == span)):
        return None
    return span


def _host_fixed_value(values: np.ndarray) -> int | None:
    host_values = np.asarray(values, dtype=np.int64)
    if host_values.ndim != 1:
        return None
    if host_values.size == 0:
        return 0
    value = int(host_values[0])
    if value < 0 or not bool(np.all(host_values == value)):
        return None
    return value


def _host_structural_count_proof(
    values: np.ndarray,
) -> tuple[int | None, int]:
    host_values = np.asarray(values, dtype=np.int64)
    if host_values.ndim != 1 or bool(np.any(host_values < 0)):
        raise ValueError("structural counts must be a nonnegative vector")
    maximum = int(host_values.max(initial=0))
    return _host_fixed_value(host_values), maximum


def _host_fixed_geometry_size_metadata(
    family: GeometryFamily,
    host_buffer: FamilyGeometryBuffer | None,
) -> DeviceFixedGeometrySizeMetadata | None:
    if host_buffer is None:
        return None
    row_count = int(host_buffer.row_count)
    if row_count <= 0:
        return None
    coord_size = int(host_buffer.x.size)
    if coord_size != int(host_buffer.y.size):
        return None

    geom_offsets = _host_fixed_i64_offsets(
        host_buffer.geometry_offsets,
        expected_size=row_count + 1,
    )
    if geom_offsets is None or int(geom_offsets[-1]) < 0:
        return None

    if family in (
        GeometryFamily.POINT,
        GeometryFamily.LINESTRING,
        GeometryFamily.MULTIPOINT,
    ):
        if int(geom_offsets[-1]) != coord_size:
            return None
        coord_count, max_coord_count = _host_structural_count_proof(
            geom_offsets[1:] - geom_offsets[:-1]
        )
        return DeviceFixedGeometrySizeMetadata(
            coord_count_per_row=coord_count,
            max_coord_count_per_row=max_coord_count,
        )

    if family is GeometryFamily.POLYGON:
        ring_offsets = _host_fixed_i64_offsets(host_buffer.ring_offsets)
        if ring_offsets is None or int(ring_offsets[-1]) != coord_size:
            return None
        if int(geom_offsets[-1]) != int(ring_offsets.size) - 1 or int(geom_offsets[-1]) >= int(
            ring_offsets.size
        ):
            return None
        ring_count, max_ring_count = _host_structural_count_proof(
            geom_offsets[1:] - geom_offsets[:-1]
        )
        coord_counts = ring_offsets[geom_offsets[1:]] - ring_offsets[geom_offsets[:-1]]
        coord_count, max_coord_count = _host_structural_count_proof(coord_counts)
        return DeviceFixedGeometrySizeMetadata(
            first_level_count_per_row=ring_count,
            coord_count_per_row=coord_count,
            max_first_level_count_per_row=max_ring_count,
            max_coord_count_per_row=max_coord_count,
        )

    if family is GeometryFamily.MULTILINESTRING:
        part_offsets = _host_fixed_i64_offsets(host_buffer.part_offsets)
        if part_offsets is None or int(part_offsets[-1]) != coord_size:
            return None
        if int(geom_offsets[-1]) != int(part_offsets.size) - 1 or int(geom_offsets[-1]) >= int(
            part_offsets.size
        ):
            return None
        part_count, max_part_count = _host_structural_count_proof(
            geom_offsets[1:] - geom_offsets[:-1]
        )
        coord_counts = part_offsets[geom_offsets[1:]] - part_offsets[geom_offsets[:-1]]
        coord_count, max_coord_count = _host_structural_count_proof(coord_counts)
        return DeviceFixedGeometrySizeMetadata(
            first_level_count_per_row=part_count,
            coord_count_per_row=coord_count,
            max_first_level_count_per_row=max_part_count,
            max_coord_count_per_row=max_coord_count,
        )

    if family is GeometryFamily.MULTIPOLYGON:
        part_offsets = _host_fixed_i64_offsets(host_buffer.part_offsets)
        ring_offsets = _host_fixed_i64_offsets(host_buffer.ring_offsets)
        if (
            part_offsets is None
            or ring_offsets is None
            or int(part_offsets[-1]) != int(ring_offsets.size) - 1
            or int(ring_offsets[-1]) != coord_size
        ):
            return None
        if int(geom_offsets[-1]) != int(part_offsets.size) - 1 or int(geom_offsets[-1]) >= int(
            part_offsets.size
        ):
            return None
        part_count, max_part_count = _host_structural_count_proof(
            geom_offsets[1:] - geom_offsets[:-1]
        )
        part_starts = part_offsets[geom_offsets[:-1]]
        part_ends = part_offsets[geom_offsets[1:]]
        if int(part_ends.max(initial=0)) >= int(ring_offsets.size):
            return None
        ring_count, max_ring_count = _host_structural_count_proof(part_ends - part_starts)
        coord_count, max_coord_count = _host_structural_count_proof(
            ring_offsets[part_ends] - ring_offsets[part_starts]
        )
        return DeviceFixedGeometrySizeMetadata(
            first_level_count_per_row=part_count,
            second_level_count_per_row=ring_count,
            coord_count_per_row=coord_count,
            max_first_level_count_per_row=max_part_count,
            max_second_level_count_per_row=max_ring_count,
            max_coord_count_per_row=max_coord_count,
        )

    return None


def _host_device_take_family_size_plan(
    family: GeometryFamily,
    host_buffer: FamilyGeometryBuffer | None,
    host_family_rows: np.ndarray | None,
) -> _DeviceTakeFamilySizePlan:
    if host_buffer is None or host_family_rows is None:
        return _DeviceTakeFamilySizePlan()
    if host_buffer.geometry_offsets.size == 0:
        return _DeviceTakeFamilySizePlan()

    if family in (
        GeometryFamily.POINT,
        GeometryFamily.LINESTRING,
        GeometryFamily.MULTIPOINT,
    ):
        return _DeviceTakeFamilySizePlan(
            coord_count=_host_offset_slice_total(
                host_buffer.geometry_offsets,
                host_family_rows,
            )
        )

    if family is GeometryFamily.POLYGON:
        return _DeviceTakeFamilySizePlan(
            first_level_count=_host_offset_slice_total(
                host_buffer.geometry_offsets,
                host_family_rows,
            ),
            coord_count=_host_nested_offset_total(
                host_buffer.geometry_offsets,
                host_buffer.ring_offsets,
                host_family_rows,
            ),
        )

    if family is GeometryFamily.MULTILINESTRING:
        return _DeviceTakeFamilySizePlan(
            first_level_count=_host_offset_slice_total(
                host_buffer.geometry_offsets,
                host_family_rows,
            ),
            coord_count=_host_nested_offset_total(
                host_buffer.geometry_offsets,
                host_buffer.part_offsets,
                host_family_rows,
            ),
        )

    if family is GeometryFamily.MULTIPOLYGON:
        return _DeviceTakeFamilySizePlan(
            first_level_count=_host_offset_slice_total(
                host_buffer.geometry_offsets,
                host_family_rows,
            ),
            second_level_count=_host_nested_offset_total(
                host_buffer.geometry_offsets,
                host_buffer.part_offsets,
                host_family_rows,
            ),
            coord_count=_host_three_level_offset_total(
                host_buffer.geometry_offsets,
                host_buffer.part_offsets,
                host_buffer.ring_offsets,
                host_family_rows,
            ),
        )

    return _DeviceTakeFamilySizePlan()


def _merge_size_plans(
    host_plan: _DeviceTakeFamilySizePlan,
    device_plan: _DeviceTakeFamilySizePlan,
) -> _DeviceTakeFamilySizePlan:
    return _DeviceTakeFamilySizePlan(
        first_level_count=(
            host_plan.first_level_count
            if host_plan.first_level_count is not None
            else device_plan.first_level_count
        ),
        second_level_count=(
            host_plan.second_level_count
            if host_plan.second_level_count is not None
            else device_plan.second_level_count
        ),
        coord_count=(
            host_plan.coord_count if host_plan.coord_count is not None else device_plan.coord_count
        ),
        first_level_capacity=(
            host_plan.first_level_capacity
            if host_plan.first_level_capacity is not None
            else device_plan.first_level_capacity
        ),
        second_level_capacity=(
            host_plan.second_level_capacity
            if host_plan.second_level_capacity is not None
            else device_plan.second_level_capacity
        ),
        coord_capacity=(
            host_plan.coord_capacity
            if host_plan.coord_capacity is not None
            else device_plan.coord_capacity
        ),
    )


def _device_buffer_fixed_size_metadata(
    family: GeometryFamily,
    device_buffer: DeviceFamilyGeometryBuffer,
) -> DeviceFixedGeometrySizeMetadata | None:
    if device_buffer.fixed_size is not None:
        return device_buffer.fixed_size
    if family is GeometryFamily.POLYGON and device_buffer.dense_single_ring_width is not None:
        return DeviceFixedGeometrySizeMetadata(
            first_level_count_per_row=1,
            coord_count_per_row=int(device_buffer.dense_single_ring_width),
        )
    return None


def _device_fixed_size_metadata_as_bounds(
    fixed_size: DeviceFixedGeometrySizeMetadata | None,
) -> DeviceFixedGeometrySizeMetadata | None:
    """Retain per-row maxima when device activity makes widths variable."""
    if fixed_size is None:
        return None
    return DeviceFixedGeometrySizeMetadata(
        max_first_level_count_per_row=(
            fixed_size.max_first_level_count_per_row
            if fixed_size.max_first_level_count_per_row is not None
            else fixed_size.first_level_count_per_row
        ),
        max_second_level_count_per_row=(
            fixed_size.max_second_level_count_per_row
            if fixed_size.max_second_level_count_per_row is not None
            else fixed_size.second_level_count_per_row
        ),
        max_coord_count_per_row=(
            fixed_size.max_coord_count_per_row
            if fixed_size.max_coord_count_per_row is not None
            else fixed_size.coord_count_per_row
        ),
    )


def _device_buffer_has_exact_row_width(
    family: GeometryFamily,
    device_buffer: DeviceFamilyGeometryBuffer,
) -> bool:
    """Whether every nested level has an exact, not merely bounded, row width."""
    fixed_size = _device_buffer_fixed_size_metadata(family, device_buffer)
    if fixed_size is None:
        return False
    if family in (
        GeometryFamily.LINESTRING,
        GeometryFamily.MULTIPOINT,
    ):
        return fixed_size.coord_count_per_row is not None
    if family in (
        GeometryFamily.POLYGON,
        GeometryFamily.MULTILINESTRING,
    ):
        return (
            fixed_size.first_level_count_per_row is not None
            and fixed_size.coord_count_per_row is not None
        )
    if family is GeometryFamily.MULTIPOLYGON:
        return (
            fixed_size.first_level_count_per_row is not None
            and fixed_size.second_level_count_per_row is not None
            and fixed_size.coord_count_per_row is not None
        )
    return True


def _device_buffer_has_bounded_row_width(
    family: GeometryFamily,
    device_buffer: DeviceFamilyGeometryBuffer,
) -> bool:
    """Whether host-visible per-row maxima bound every nested allocation level."""
    fixed_size = _device_buffer_fixed_size_metadata(family, device_buffer)
    if family is GeometryFamily.POINT:
        return True
    if fixed_size is None or fixed_size.max_coord_count_per_row is None:
        return False
    if (
        family
        in (
            GeometryFamily.POLYGON,
            GeometryFamily.MULTILINESTRING,
            GeometryFamily.MULTIPOLYGON,
        )
        and fixed_size.max_first_level_count_per_row is None
    ):
        return False
    if family is GeometryFamily.MULTIPOLYGON and fixed_size.max_second_level_count_per_row is None:
        return False
    return True


def _common_device_fixed_size_metadata(
    family: GeometryFamily,
    buffers: list[DeviceFamilyGeometryBuffer],
) -> DeviceFixedGeometrySizeMetadata | None:
    proofs = [_device_buffer_fixed_size_metadata(family, buffer) for buffer in buffers]
    if not proofs or any(proof is None for proof in proofs):
        return None

    def _common_fixed(field: str) -> int | None:
        values = [getattr(proof, field) for proof in proofs]
        first = values[0]
        if first is None or any(value != first for value in values[1:]):
            return None
        return int(first)

    def _max_bound(field: str) -> int | None:
        values = [getattr(proof, field) for proof in proofs]
        bounded = [int(value) for value in values if value is not None]
        return max(bounded) if len(bounded) == len(values) else None

    return DeviceFixedGeometrySizeMetadata(
        first_level_count_per_row=_common_fixed("first_level_count_per_row"),
        second_level_count_per_row=_common_fixed("second_level_count_per_row"),
        coord_count_per_row=_common_fixed("coord_count_per_row"),
        max_first_level_count_per_row=_max_bound("max_first_level_count_per_row"),
        max_second_level_count_per_row=_max_bound("max_second_level_count_per_row"),
        max_coord_count_per_row=_max_bound("max_coord_count_per_row"),
    )


def _size_plan_complete(
    family: GeometryFamily,
    plan: _DeviceTakeFamilySizePlan,
) -> bool:
    if family in (
        GeometryFamily.POINT,
        GeometryFamily.LINESTRING,
        GeometryFamily.MULTIPOINT,
    ):
        return plan.coord_count is not None
    if family in (GeometryFamily.POLYGON, GeometryFamily.MULTILINESTRING):
        return plan.first_level_count is not None and plan.coord_count is not None
    if family is GeometryFamily.MULTIPOLYGON:
        return (
            plan.first_level_count is not None
            and plan.second_level_count is not None
            and plan.coord_count is not None
        )
    return True


def _device_take_capacity_multiplier(
    row_count: int,
    host_family_rows: np.ndarray | None,
    *,
    assume_unique_indices: bool = False,
) -> int:
    """Return conservative capacity multiplier for bounded device gathers."""
    if row_count <= 0:
        return 1
    if assume_unique_indices:
        return 1
    duplicate_capable = True
    if host_family_rows is not None:
        h_rows = np.asarray(host_family_rows, dtype=np.int64)
        duplicate_capable = int(np.unique(h_rows).size) != int(h_rows.size)
    return int(row_count) if duplicate_capable else 1


def _device_take_fixed_family_size_plan_from_metadata(
    family: GeometryFamily,
    device_buffer: DeviceFamilyGeometryBuffer,
    row_count: int,
) -> _DeviceTakeFamilySizePlan:
    fixed_size = _device_buffer_fixed_size_metadata(family, device_buffer)
    if fixed_size is None:
        return _DeviceTakeFamilySizePlan()
    first = fixed_size.first_level_count_per_row
    second = fixed_size.second_level_count_per_row
    coords = fixed_size.coord_count_per_row
    first_capacity = fixed_size.max_first_level_count_per_row
    second_capacity = fixed_size.max_second_level_count_per_row
    coord_capacity = fixed_size.max_coord_count_per_row

    def _total(value: int | None) -> int | None:
        return None if value is None else int(row_count) * int(value)

    if family in (
        GeometryFamily.POINT,
        GeometryFamily.LINESTRING,
        GeometryFamily.MULTIPOINT,
    ):
        return _DeviceTakeFamilySizePlan(
            coord_count=_total(coords),
            coord_capacity=_total(coord_capacity),
        )
    if family in (GeometryFamily.POLYGON, GeometryFamily.MULTILINESTRING):
        return _DeviceTakeFamilySizePlan(
            first_level_count=_total(first),
            coord_count=_total(coords),
            first_level_capacity=_total(first_capacity),
            coord_capacity=_total(coord_capacity),
        )
    if family is GeometryFamily.MULTIPOLYGON:
        return _DeviceTakeFamilySizePlan(
            first_level_count=_total(first),
            second_level_count=_total(second),
            coord_count=_total(coords),
            first_level_capacity=_total(first_capacity),
            second_level_capacity=_total(second_capacity),
            coord_capacity=_total(coord_capacity),
        )
    return _DeviceTakeFamilySizePlan()


def _device_take_family_size_plan_from_device(
    family: GeometryFamily,
    device_buffer: DeviceFamilyGeometryBuffer,
    family_rows: DeviceArray,
) -> _DeviceTakeFamilySizePlan:
    row_count = int(family_rows.size)
    if row_count == 0:
        return _DeviceTakeFamilySizePlan(
            first_level_count=0,
            second_level_count=0,
            coord_count=0,
        )

    fixed_plan = _device_take_fixed_family_size_plan_from_metadata(
        family,
        device_buffer,
        row_count,
    )
    if _size_plan_complete(family, fixed_plan):
        return fixed_plan

    if row_count == 1:
        if family in (
            GeometryFamily.POINT,
            GeometryFamily.LINESTRING,
            GeometryFamily.MULTIPOINT,
        ):
            capacity_plan = _DeviceTakeFamilySizePlan(coord_capacity=int(device_buffer.x.size))
            return _merge_size_plans(fixed_plan, capacity_plan)
        if family is GeometryFamily.POLYGON:
            capacity_plan = _DeviceTakeFamilySizePlan(
                first_level_capacity=max(int(device_buffer.ring_offsets.size) - 1, 0),
                coord_capacity=int(device_buffer.x.size),
            )
            return _merge_size_plans(fixed_plan, capacity_plan)
        if family is GeometryFamily.MULTILINESTRING:
            capacity_plan = _DeviceTakeFamilySizePlan(
                first_level_capacity=max(int(device_buffer.part_offsets.size) - 1, 0),
                coord_capacity=int(device_buffer.x.size),
            )
            return _merge_size_plans(fixed_plan, capacity_plan)
        if family is GeometryFamily.MULTIPOLYGON:
            capacity_plan = _DeviceTakeFamilySizePlan(
                first_level_capacity=max(int(device_buffer.part_offsets.size) - 1, 0),
                second_level_capacity=max(int(device_buffer.ring_offsets.size) - 1, 0),
                coord_capacity=int(device_buffer.x.size),
            )
            return _merge_size_plans(fixed_plan, capacity_plan)

    if family is GeometryFamily.POLYGON and device_buffer.ring_offsets is not None:
        dense_width = device_buffer.dense_single_ring_width
        if dense_width is not None:
            dense_plan = _DeviceTakeFamilySizePlan(
                first_level_count=row_count,
                coord_count=row_count * int(dense_width),
            )
            return _merge_size_plans(fixed_plan, dense_plan)
    return fixed_plan


def _complete_device_take_family_size_plan(
    family: GeometryFamily,
    device_buffer: DeviceFamilyGeometryBuffer,
    family_rows: DeviceArray,
    host_buffer: FamilyGeometryBuffer | None,
    host_family_rows: np.ndarray | None,
) -> _DeviceTakeFamilySizePlan:
    host_plan = _host_device_take_family_size_plan(
        family,
        host_buffer,
        host_family_rows,
    )
    if _size_plan_complete(family, host_plan):
        return host_plan

    return _merge_size_plans(
        host_plan,
        _device_take_family_size_plan_from_device(family, device_buffer, family_rows),
    )


def _device_slice_plan(
    offsets: DeviceArray,
    rows: DeviceArray,
    *,
    precomputed_total: int | None = None,
    active_row_count: object | None = None,
    active_row_mask: DeviceArray | None = None,
) -> tuple[DeviceArray, DeviceArray, DeviceArray, int]:
    """Build starts, lengths, output offsets, and total length for row spans."""
    if precomputed_total is None:
        raise ValueError("device slice planning requires an exact total or physical capacity")
    rows = cp.asarray(rows)
    n = int(rows.size)
    new_offsets = cp.empty(n + 1, dtype=cp.int32)
    new_offsets[0] = 0

    if n == 0:
        empty_i32 = cp.empty(0, dtype=cp.int32)
        return empty_i32, empty_i32, new_offsets, 0

    starts = offsets[rows].astype(cp.int32, copy=False)
    ends = offsets[rows + 1].astype(cp.int32, copy=False)
    lengths = (ends - starts).astype(cp.int32, copy=False)
    if active_row_count is not None:
        active = cp.arange(n, dtype=cp.int64) < cp.asarray(
            active_row_count,
            dtype=cp.int64,
        )
        lengths = cp.where(active, lengths, cp.zeros((), dtype=cp.int32))
    if active_row_mask is not None:
        active = cp.asarray(active_row_mask, dtype=cp.bool_)
        if active.ndim != 1 or int(active.size) != n:
            raise ValueError("device slice activity mask must match row capacity")
        lengths = cp.where(active, lengths, cp.zeros((), dtype=cp.int32))
    cp.cumsum(lengths, out=new_offsets[1:])
    total_length = int(precomputed_total)
    return starts, lengths, new_offsets, total_length


def _device_gather_offset_index_ranges(
    offsets: DeviceArray,
    rows: DeviceArray,
    *,
    precomputed_total: int | None = None,
    allocation_capacity: int | None = None,
    active_row_count: object | None = None,
    active_row_mask: DeviceArray | None = None,
) -> tuple[DeviceArray, DeviceArray]:
    """Gather integer index ranges described by offset rows.

    This is the device-side equivalent of gathering ``arange(offsets[-1])``
    with :func:`_device_gather_offset_slices`, but avoids materializing the
    source arange and the per-output ``searchsorted`` mapping.
    """
    starts, lengths, new_offsets, total_length = _device_slice_plan(
        offsets,
        rows,
        precomputed_total=(
            precomputed_total if precomputed_total is not None else allocation_capacity
        ),
        active_row_count=active_row_count,
        active_row_mask=active_row_mask,
    )
    if total_length == 0:
        return cp.empty(0, dtype=cp.int32), new_offsets

    gathered = (
        cp.zeros(total_length, dtype=cp.int32)
        if allocation_capacity is not None and precomputed_total is None
        else cp.empty(total_length, dtype=cp.int32)
    )
    runtime = get_cuda_runtime()
    kernels = _owned_take_kernels()
    kernel = kernels["owned_take_gather_index_ranges_i32"]
    block_size = runtime.optimal_block_size(kernel)
    ptr = runtime.pointer
    params = (
        (
            ptr(starts),
            ptr(lengths),
            ptr(new_offsets),
            ptr(gathered),
            int(starts.size),
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
        ),
    )
    runtime.launch(
        kernel,
        grid=(int(starts.size), 1, 1),
        block=(block_size, 1, 1),
        params=params,
    )
    return gathered, new_offsets


def _device_gather_xy_offset_slices(
    x: DeviceArray,
    y: DeviceArray,
    offsets: DeviceArray,
    rows: DeviceArray,
    *,
    precomputed_total: int | None = None,
    allocation_capacity: int | None = None,
    active_row_count: object | None = None,
    active_row_mask: DeviceArray | None = None,
) -> tuple[DeviceArray, DeviceArray, DeviceArray]:
    """Gather separated x/y coordinate spans without AoS temporaries.

    ``allocation_capacity`` is a native gathered-buffer carrier escape hatch:
    callers that can prove an upper bound from the physical source buffer may
    allocate that capacity and let the returned offsets describe the logical
    coordinate size.  This avoids a host scalar allocation fence for variable
    rowsets that are known to be subsets/permutations of an existing buffer.
    """
    starts, lengths, new_offsets, total_length = _device_slice_plan(
        offsets,
        rows,
        precomputed_total=(
            precomputed_total if precomputed_total is not None else allocation_capacity
        ),
        active_row_count=active_row_count,
        active_row_mask=active_row_mask,
    )
    if total_length == 0:
        return (
            cp.empty(0, dtype=cp.float64),
            cp.empty(0, dtype=cp.float64),
            new_offsets,
        )

    new_x = cp.empty(total_length, dtype=cp.float64)
    new_y = cp.empty(total_length, dtype=cp.float64)
    runtime = get_cuda_runtime()
    kernels = _owned_take_kernels()
    kernel = kernels["owned_take_gather_xy_ranges_f64"]
    block_size = runtime.optimal_block_size(kernel)
    ptr = runtime.pointer
    params = (
        (
            ptr(x),
            ptr(y),
            ptr(starts),
            ptr(lengths),
            ptr(new_offsets),
            ptr(new_x),
            ptr(new_y),
            int(starts.size),
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
        ),
    )
    runtime.launch(
        kernel,
        grid=(int(starts.size), 1, 1),
        block=(block_size, 1, 1),
        params=params,
    )
    return new_x, new_y, new_offsets


def _host_dense_single_ring_width(
    host_buffer: FamilyGeometryBuffer | None,
) -> int | None:
    """Return fixed ring width when existing host structure proves it."""
    if host_buffer is None or host_buffer.ring_offsets is None or host_buffer.row_count <= 0:
        return None
    row_count = int(host_buffer.row_count)
    if (
        int(host_buffer.geometry_offsets.size) != row_count + 1
        or int(host_buffer.ring_offsets.size) != row_count + 1
        or bool(np.any(host_buffer.empty_mask))
    ):
        return None
    geom_counts = host_buffer.geometry_offsets[1:] - host_buffer.geometry_offsets[:-1]
    if not bool(np.all(geom_counts == 1)):
        return None
    ring_widths = host_buffer.ring_offsets[1:] - host_buffer.ring_offsets[:-1]
    first_width = int(ring_widths[0])
    if (
        first_width <= 0
        or not bool(np.all(ring_widths == first_width))
        or int(host_buffer.x.size) != row_count * first_width
    ):
        return None
    return first_width


def _host_axis_aligned_rectangle_batch(
    host_buffer: FamilyGeometryBuffer | None,
) -> bool:
    """Return True when host polygon structure proves an exact box batch."""
    if _host_dense_single_ring_width(host_buffer) != 5:
        return False
    if host_buffer is None or host_buffer.family is not GeometryFamily.POLYGON:
        return False
    row_count = int(host_buffer.row_count)
    if row_count == 0:
        return False
    try:
        x = np.asarray(host_buffer.x, dtype=np.float64).reshape(row_count, 5)
        y = np.asarray(host_buffer.y, dtype=np.float64).reshape(row_count, 5)
    except ValueError:
        return False
    if not (
        np.allclose(x[:, 0], x[:, 4], rtol=0.0, atol=1e-12)
        and np.allclose(y[:, 0], y[:, 4], rtol=0.0, atol=1e-12)
    ):
        return False
    dx = x[:, 1:] - x[:, :-1]
    dy = y[:, 1:] - y[:, :-1]
    axis_aligned = (np.abs(dx) < 1e-12) ^ (np.abs(dy) < 1e-12)
    if not bool(np.all(axis_aligned)):
        return False
    bounds = np.column_stack(
        (
            np.min(x[:, :4], axis=1),
            np.min(y[:, :4], axis=1),
            np.max(x[:, :4], axis=1),
            np.max(y[:, :4], axis=1),
        )
    )
    if not bool(np.all((bounds[:, 0] < bounds[:, 2]) & (bounds[:, 1] < bounds[:, 3]))):
        return False

    x_is_min = np.isclose(x[:, :4], bounds[:, 0:1], rtol=0.0, atol=1e-12)
    x_is_max = np.isclose(x[:, :4], bounds[:, 2:3], rtol=0.0, atol=1e-12)
    y_is_min = np.isclose(y[:, :4], bounds[:, 1:2], rtol=0.0, atol=1e-12)
    y_is_max = np.isclose(y[:, :4], bounds[:, 3:4], rtol=0.0, atol=1e-12)
    corners = (
        x_is_min.astype(np.int8)
        + 2 * x_is_max.astype(np.int8)
        + 4 * y_is_min.astype(np.int8)
        + 8 * y_is_max.astype(np.int8)
    )
    return bool(
        np.all(x_is_min | x_is_max)
        and np.all(y_is_min | y_is_max)
        and np.all(np.sort(corners, axis=1) == np.asarray([5, 6, 9, 10]))
    )


def host_owned_axis_aligned_rectangle_batch(
    owned: OwnedGeometryArray,
) -> bool | None:
    """Classify a logical host-owned batch from canonical polygon buffers.

    ``None`` means the host carrier is not materialized enough to answer
    without a device transfer.  ``False`` is an authoritative structural
    rejection; ``True`` proves every logical row is a valid rectangle polygon.
    """
    if owned.row_count == 0:
        return True
    if owned._validity is None or owned._tags is None or owned._family_row_offsets is None:
        return None

    validity = np.asarray(owned._validity, dtype=bool)
    tags = np.asarray(owned._tags, dtype=np.int8)
    family_rows = np.asarray(owned._family_row_offsets, dtype=np.int64)
    if not bool(np.all(validity)):
        return False

    polygon_tag = np.int8(FAMILY_TAGS[GeometryFamily.POLYGON])
    if not bool(np.all(tags == polygon_tag)):
        return False

    polygon = owned.families.get(GeometryFamily.POLYGON)
    if polygon is None:
        return False
    if not polygon.host_materialized:
        return None
    if not bool(np.all((family_rows >= 0) & (family_rows < int(polygon.row_count)))):
        return False
    return _host_axis_aligned_rectangle_batch(polygon)


def _host_regular_grid_rect_metadata(
    host_buffer: FamilyGeometryBuffer | None,
) -> DeviceRegularGridRectMetadata | None:
    """Return a trusted row-major rectangle-grid proof from host buffers."""
    if not _host_axis_aligned_rectangle_batch(host_buffer):
        return None
    assert host_buffer is not None
    row_count = int(host_buffer.row_count)
    if row_count <= 0:
        return None
    try:
        x = np.asarray(host_buffer.x, dtype=np.float64).reshape(row_count, 5)
        y = np.asarray(host_buffer.y, dtype=np.float64).reshape(row_count, 5)
    except ValueError:
        return None

    bounds = np.column_stack(
        (
            np.min(x[:, :4], axis=1),
            np.min(y[:, :4], axis=1),
            np.max(x[:, :4], axis=1),
            np.max(y[:, :4], axis=1),
        )
    )
    if not np.isfinite(bounds).all():
        return None
    widths = bounds[:, 2] - bounds[:, 0]
    heights = bounds[:, 3] - bounds[:, 1]
    if np.any(widths <= 0.0) or np.any(heights <= 0.0):
        return None
    cell_width = float(widths[0])
    cell_height = float(heights[0])
    tol_scale = max(abs(cell_width), abs(cell_height), 1.0)
    tol = 1e-12 * tol_scale
    if np.any(np.abs(widths - cell_width) > tol):
        return None
    if np.any(np.abs(heights - cell_height) > tol):
        return None

    x_is_min = np.abs(x - bounds[:, 0:1]) <= tol
    x_is_max = np.abs(x - bounds[:, 2:3]) <= tol
    y_is_min = np.abs(y - bounds[:, 1:2]) <= tol
    y_is_max = np.abs(y - bounds[:, 3:4]) <= tol
    if (
        not np.all(x_is_min | x_is_max)
        or not np.all(y_is_min | y_is_max)
        or not np.all(x_is_min[:, :4].sum(axis=1) == 2)
        or not np.all(x_is_max[:, :4].sum(axis=1) == 2)
        or not np.all(y_is_min[:, :4].sum(axis=1) == 2)
        or not np.all(y_is_max[:, :4].sum(axis=1) == 2)
    ):
        return None

    origin_x = float(bounds[:, 0].min())
    origin_y = float(bounds[:, 1].min())
    cols = int(np.rint((bounds[:, 0].max() - origin_x) / cell_width)) + 1
    if cols <= 0:
        return None
    col_index = np.rint((bounds[:, 0] - origin_x) / cell_width).astype(
        np.int64,
        copy=False,
    )
    row_index = np.rint((bounds[:, 1] - origin_y) / cell_height).astype(
        np.int64,
        copy=False,
    )
    rows = _regular_grid_rows_for_size(row_count, cols)
    if np.any(col_index < 0) or np.any(col_index >= cols):
        return None
    if np.any(row_index < 0) or np.any(row_index >= rows):
        return None
    expected = row_index * np.int64(cols) + col_index
    if not np.array_equal(expected, np.arange(row_count, dtype=np.int64)):
        return None

    total_bounds = _regular_grid_total_bounds(
        origin_x=origin_x,
        origin_y=origin_y,
        cell_width=cell_width,
        cell_height=cell_height,
        cols=cols,
        size=row_count,
    )
    if not np.allclose(bounds[:, 2].max(), total_bounds[2], rtol=0.0, atol=tol):
        return None
    if not np.allclose(bounds[:, 3].max(), total_bounds[3], rtol=0.0, atol=tol):
        return None
    return DeviceRegularGridRectMetadata(
        origin_x=origin_x,
        origin_y=origin_y,
        cell_width=cell_width,
        cell_height=cell_height,
        cols=cols,
        rows=rows,
        size=row_count,
        total_bounds=total_bounds,
    )


def _regular_grid_rect_for_host_rows(
    proof: DeviceRegularGridRectMetadata | None,
    host_family_rows: np.ndarray | None,
) -> DeviceRegularGridRectMetadata | None:
    """Preserve a regular-grid proof for contiguous row-major takes."""
    if proof is None or host_family_rows is None:
        return None
    rows = np.asarray(host_family_rows, dtype=np.int64)
    size = int(rows.size)
    if size == 0:
        return DeviceRegularGridRectMetadata(
            origin_x=float(proof.origin_x),
            origin_y=float(proof.origin_y),
            cell_width=float(proof.cell_width),
            cell_height=float(proof.cell_height),
            cols=int(proof.cols),
            rows=0,
            size=0,
            total_bounds=(float("nan"),) * 4,
        )
    cols = int(proof.cols)
    if cols <= 0 or float(proof.cell_width) <= 0.0 or float(proof.cell_height) <= 0.0:
        return None
    start = int(rows[0])
    if start < 0 or start + size > int(proof.size):
        return None
    expected = np.arange(start, start + size, dtype=np.int64)
    if not bool(np.array_equal(rows, expected)):
        return None

    if start % cols != 0:
        remaining_in_row = cols - (start % cols)
        if size > remaining_in_row:
            return None
        out_cols = size
    else:
        out_cols = cols
    if out_cols <= 0:
        return None

    origin_x = float(proof.origin_x) + (start % cols) * float(proof.cell_width)
    origin_y = float(proof.origin_y) + (start // cols) * float(proof.cell_height)
    total_bounds = _regular_grid_total_bounds(
        origin_x=origin_x,
        origin_y=origin_y,
        cell_width=float(proof.cell_width),
        cell_height=float(proof.cell_height),
        cols=out_cols,
        size=size,
    )
    return DeviceRegularGridRectMetadata(
        origin_x=origin_x,
        origin_y=origin_y,
        cell_width=float(proof.cell_width),
        cell_height=float(proof.cell_height),
        cols=out_cols,
        rows=_regular_grid_rows_for_size(size, out_cols),
        size=size,
        total_bounds=total_bounds,
    )


def _device_take_dense_single_ring_polygon_buffer(
    device_buffer: DeviceFamilyGeometryBuffer,
    family_rows: DeviceArray,
    *,
    width: int,
    host_family_rows: np.ndarray | None = None,
    active_row_count: object | None = None,
) -> DeviceFamilyGeometryBuffer:
    """Gather fixed-width one-ring polygon rows with one SoA copy kernel."""
    n = int(family_rows.size)
    new_empty_mask = device_buffer.empty_mask[family_rows]
    new_bounds = device_buffer.bounds[family_rows] if device_buffer.bounds is not None else None
    regular_grid_rect = _regular_grid_rect_for_host_rows(
        device_buffer.regular_grid_rect,
        host_family_rows,
    )
    total_coords = n * int(width)
    if active_row_count is not None:
        d_active = (
            cp.arange(n, dtype=cp.int64)
            < cp.asarray(
                active_row_count,
                dtype=cp.int64,
            )[0]
        )
        new_geom_offsets = cp.empty(n + 1, dtype=cp.int32)
        new_geom_offsets[0] = 0
        cp.cumsum(d_active, dtype=cp.int32, out=new_geom_offsets[1:])
        new_x, new_y, new_ring_offsets = _device_gather_xy_offset_slices(
            device_buffer.x,
            device_buffer.y,
            device_buffer.ring_offsets,
            family_rows.astype(cp.int64, copy=False),
            precomputed_total=total_coords,
            active_row_count=active_row_count,
        )
        return DeviceFamilyGeometryBuffer(
            family=GeometryFamily.POLYGON,
            x=new_x,
            y=new_y,
            geometry_offsets=new_geom_offsets,
            empty_mask=cp.where(d_active, new_empty_mask, cp.bool_(True)),
            ring_offsets=new_ring_offsets,
            bounds=(
                None
                if new_bounds is None
                else cp.where(
                    d_active[:, None],
                    new_bounds,
                    cp.asarray(cp.nan, dtype=cp.float64),
                )
            ),
        )

    new_geom_offsets = cp.arange(n + 1, dtype=cp.int32)
    new_ring_offsets = cp.arange(n + 1, dtype=cp.int32) * width
    if total_coords == 0:
        new_x = cp.empty(0, dtype=cp.float64)
        new_y = cp.empty(0, dtype=cp.float64)
    else:
        rows = family_rows.astype(cp.int64, copy=False)
        new_x = cp.empty(total_coords, dtype=cp.float64)
        new_y = cp.empty(total_coords, dtype=cp.float64)
        runtime = get_cuda_runtime()
        kernel = _owned_take_kernels()["owned_take_gather_dense_xy_f64"]
        grid, block = runtime.launch_config(kernel, total_coords)
        ptr = runtime.pointer
        params = (
            (
                ptr(device_buffer.x),
                ptr(device_buffer.y),
                ptr(rows),
                ptr(new_x),
                ptr(new_y),
                int(width),
                int(total_coords),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
            ),
        )
        runtime.launch(
            kernel,
            grid=grid,
            block=block,
            params=params,
        )
    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.POLYGON,
        x=new_x,
        y=new_y,
        geometry_offsets=new_geom_offsets,
        empty_mask=new_empty_mask,
        ring_offsets=new_ring_offsets,
        bounds=new_bounds,
        dense_single_ring_width=width,
        axis_aligned_rectangles=(bool(device_buffer.axis_aligned_rectangles) and int(width) == 5),
        regular_grid_rect=regular_grid_rect,
        fixed_size=DeviceFixedGeometrySizeMetadata(
            first_level_count_per_row=1,
            coord_count_per_row=int(width),
        ),
    )


def _device_take_family_buffer(
    device_buffer: DeviceFamilyGeometryBuffer,
    family: GeometryFamily,
    family_rows: DeviceArray,
    host_buffer: FamilyGeometryBuffer | None = None,
    *,
    host_family_rows: np.ndarray | None = None,
    allow_capacity_allocation: bool = False,
    assume_unique_indices: bool = False,
    active_row_count: object | None = None,
    active_row_mask: DeviceArray | None = None,
    exact_size_plan: _DeviceTakeFamilySizePlan | None = None,
    output_fixed_size: DeviceFixedGeometrySizeMetadata | None = None,
) -> DeviceFamilyGeometryBuffer:
    """Device-side extract of *family_rows* from a DeviceFamilyGeometryBuffer.

    GPU equivalent of :func:`_take_family_buffer` — all offset gathering uses
    :func:`_device_gather_offset_slices` instead of the serial host loop.

    ``allow_capacity_allocation`` remains accepted for internal caller
    compatibility; physical device takes now always use structural capacity
    when an exact host/fixed-size plan is unavailable.
    """
    d_active_rows = None if active_row_mask is None else cp.asarray(active_row_mask, dtype=cp.bool_)
    new_empty_mask = device_buffer.empty_mask[family_rows]
    if d_active_rows is not None:
        new_empty_mask = cp.where(d_active_rows, new_empty_mask, cp.bool_(True))
    new_bounds = device_buffer.bounds[family_rows] if device_buffer.bounds is not None else None
    if new_bounds is not None and d_active_rows is not None:
        new_bounds = cp.where(
            d_active_rows[:, None],
            new_bounds,
            cp.asarray(cp.nan, dtype=cp.float64),
        )
    fixed_size = _device_buffer_fixed_size_metadata(family, device_buffer)
    row_count = int(family_rows.size)

    if (
        d_active_rows is None
        and family is GeometryFamily.POINT
        and host_buffer is not None
        and int(host_buffer.row_count) == int(device_buffer.x.size)
        and int(host_buffer.row_count) == int(device_buffer.y.size)
    ):
        point_rows = family_rows.astype(cp.int64, copy=False)
        row_count = int(point_rows.size)
        return DeviceFamilyGeometryBuffer(
            family=family,
            x=device_buffer.x[point_rows],
            y=device_buffer.y[point_rows],
            geometry_offsets=cp.arange(row_count + 1, dtype=cp.int32),
            empty_mask=cp.zeros(row_count, dtype=cp.bool_),
            bounds=new_bounds,
            fixed_size=(
                fixed_size
                if fixed_size is not None
                else DeviceFixedGeometrySizeMetadata(coord_count_per_row=1)
            ),
        )

    if family in (GeometryFamily.POINT, GeometryFamily.LINESTRING, GeometryFamily.MULTIPOINT):
        coord_capacity = None
        size_plan = exact_size_plan or _complete_device_take_family_size_plan(
            family, device_buffer, family_rows, host_buffer, host_family_rows
        )
        if d_active_rows is not None and exact_size_plan is None:
            size_plan = _device_take_size_plan_as_capacity(size_plan)
        if not _size_plan_complete(family, size_plan):
            capacity_multiplier = _device_take_capacity_multiplier(
                row_count,
                host_family_rows,
                assume_unique_indices=assume_unique_indices,
            )
            coord_capacity = size_plan.coord_capacity
            if coord_capacity is None:
                coord_capacity = capacity_multiplier * int(device_buffer.x.size)
        new_x, new_y, new_geom_offsets = _device_gather_xy_offset_slices(
            device_buffer.x,
            device_buffer.y,
            device_buffer.geometry_offsets,
            family_rows,
            precomputed_total=size_plan.coord_count,
            allocation_capacity=coord_capacity,
            active_row_count=active_row_count,
            active_row_mask=d_active_rows,
        )
        return DeviceFamilyGeometryBuffer(
            family=family,
            x=new_x,
            y=new_y,
            geometry_offsets=new_geom_offsets,
            empty_mask=new_empty_mask,
            bounds=new_bounds,
            fixed_size=(
                output_fixed_size
                if output_fixed_size is not None
                else (
                    fixed_size
                    if d_active_rows is None
                    else _device_fixed_size_metadata_as_bounds(fixed_size)
                )
            ),
        )

    if family is GeometryFamily.POLYGON:
        dense_width = device_buffer.dense_single_ring_width
        if dense_width is None:
            dense_width = _host_dense_single_ring_width(host_buffer)
        if dense_width is not None and d_active_rows is None:
            return _device_take_dense_single_ring_polygon_buffer(
                device_buffer,
                family_rows,
                width=dense_width,
                host_family_rows=host_family_rows,
                active_row_count=active_row_count,
            )
        ring_capacity = None
        coord_capacity = None
        size_plan = exact_size_plan or _complete_device_take_family_size_plan(
            family, device_buffer, family_rows, host_buffer, host_family_rows
        )
        if d_active_rows is not None and exact_size_plan is None:
            size_plan = _device_take_size_plan_as_capacity(size_plan)
        if not _size_plan_complete(family, size_plan):
            capacity_multiplier = _device_take_capacity_multiplier(
                row_count,
                host_family_rows,
                assume_unique_indices=assume_unique_indices,
            )
            if size_plan.first_level_count is None:
                ring_capacity = size_plan.first_level_capacity
                if ring_capacity is None:
                    ring_capacity = capacity_multiplier * max(
                        int(device_buffer.ring_offsets.size) - 1, 0
                    )
            if size_plan.coord_count is None:
                coord_capacity = size_plan.coord_capacity
                if coord_capacity is None:
                    coord_capacity = capacity_multiplier * int(device_buffer.x.size)
        ring_indices, new_geom_offsets = _device_gather_offset_index_ranges(
            device_buffer.geometry_offsets,
            family_rows,
            precomputed_total=size_plan.first_level_count,
            allocation_capacity=ring_capacity,
            active_row_count=active_row_count,
            active_row_mask=d_active_rows,
        )
        new_x, new_y, new_ring_offsets = _device_gather_xy_offset_slices(
            device_buffer.x,
            device_buffer.y,
            device_buffer.ring_offsets,
            ring_indices,
            precomputed_total=size_plan.coord_count,
            allocation_capacity=coord_capacity,
            active_row_count=(
                new_geom_offsets[-1]
                if (
                    active_row_count is not None
                    or d_active_rows is not None
                    or ring_capacity is not None
                )
                else None
            ),
        )
        return DeviceFamilyGeometryBuffer(
            family=family,
            x=new_x,
            y=new_y,
            geometry_offsets=new_geom_offsets,
            empty_mask=new_empty_mask,
            ring_offsets=new_ring_offsets,
            bounds=new_bounds,
            axis_aligned_rectangles=bool(device_buffer.axis_aligned_rectangles),
            fixed_size=(
                output_fixed_size
                if output_fixed_size is not None
                else (
                    fixed_size
                    if d_active_rows is None
                    else _device_fixed_size_metadata_as_bounds(fixed_size)
                )
            ),
        )

    if family is GeometryFamily.MULTILINESTRING:
        part_capacity = None
        coord_capacity = None
        size_plan = exact_size_plan or _complete_device_take_family_size_plan(
            family, device_buffer, family_rows, host_buffer, host_family_rows
        )
        if d_active_rows is not None and exact_size_plan is None:
            size_plan = _device_take_size_plan_as_capacity(size_plan)
        if not _size_plan_complete(family, size_plan):
            capacity_multiplier = _device_take_capacity_multiplier(
                row_count,
                host_family_rows,
                assume_unique_indices=assume_unique_indices,
            )
            if size_plan.first_level_count is None:
                part_capacity = size_plan.first_level_capacity
                if part_capacity is None:
                    part_capacity = capacity_multiplier * max(
                        int(device_buffer.part_offsets.size) - 1, 0
                    )
            if size_plan.coord_count is None:
                coord_capacity = size_plan.coord_capacity
                if coord_capacity is None:
                    coord_capacity = capacity_multiplier * int(device_buffer.x.size)
        part_indices, new_geom_offsets = _device_gather_offset_index_ranges(
            device_buffer.geometry_offsets,
            family_rows,
            precomputed_total=size_plan.first_level_count,
            allocation_capacity=part_capacity,
            active_row_count=active_row_count,
            active_row_mask=d_active_rows,
        )
        new_x, new_y, new_part_offsets = _device_gather_xy_offset_slices(
            device_buffer.x,
            device_buffer.y,
            device_buffer.part_offsets,
            part_indices,
            precomputed_total=size_plan.coord_count,
            allocation_capacity=coord_capacity,
            active_row_count=(
                new_geom_offsets[-1]
                if (
                    active_row_count is not None
                    or d_active_rows is not None
                    or part_capacity is not None
                )
                else None
            ),
        )
        return DeviceFamilyGeometryBuffer(
            family=family,
            x=new_x,
            y=new_y,
            geometry_offsets=new_geom_offsets,
            empty_mask=new_empty_mask,
            part_offsets=new_part_offsets,
            bounds=new_bounds,
            fixed_size=(
                output_fixed_size
                if output_fixed_size is not None
                else (
                    fixed_size
                    if d_active_rows is None
                    else _device_fixed_size_metadata_as_bounds(fixed_size)
                )
            ),
        )

    if family is GeometryFamily.MULTIPOLYGON:
        part_capacity = None
        ring_capacity = None
        coord_capacity = None
        size_plan = exact_size_plan or _complete_device_take_family_size_plan(
            family, device_buffer, family_rows, host_buffer, host_family_rows
        )
        if d_active_rows is not None and exact_size_plan is None:
            size_plan = _device_take_size_plan_as_capacity(size_plan)
        if not _size_plan_complete(family, size_plan):
            capacity_multiplier = _device_take_capacity_multiplier(
                row_count,
                host_family_rows,
                assume_unique_indices=assume_unique_indices,
            )
            if size_plan.first_level_count is None:
                part_capacity = size_plan.first_level_capacity
                if part_capacity is None:
                    part_capacity = capacity_multiplier * max(
                        int(device_buffer.part_offsets.size) - 1, 0
                    )
            if size_plan.second_level_count is None:
                ring_capacity = size_plan.second_level_capacity
                if ring_capacity is None:
                    ring_capacity = capacity_multiplier * max(
                        int(device_buffer.ring_offsets.size) - 1, 0
                    )
            if size_plan.coord_count is None:
                coord_capacity = size_plan.coord_capacity
                if coord_capacity is None:
                    coord_capacity = capacity_multiplier * int(device_buffer.x.size)
        part_indices, new_geom_offsets = _device_gather_offset_index_ranges(
            device_buffer.geometry_offsets,
            family_rows,
            precomputed_total=size_plan.first_level_count,
            allocation_capacity=part_capacity,
            active_row_count=active_row_count,
            active_row_mask=d_active_rows,
        )
        ring_indices, new_part_offsets = _device_gather_offset_index_ranges(
            device_buffer.part_offsets,
            part_indices,
            precomputed_total=size_plan.second_level_count,
            allocation_capacity=ring_capacity,
            active_row_count=(
                new_geom_offsets[-1]
                if (
                    active_row_count is not None
                    or d_active_rows is not None
                    or part_capacity is not None
                )
                else None
            ),
        )
        new_x, new_y, new_ring_offsets = _device_gather_xy_offset_slices(
            device_buffer.x,
            device_buffer.y,
            device_buffer.ring_offsets,
            ring_indices,
            precomputed_total=size_plan.coord_count,
            allocation_capacity=coord_capacity,
            active_row_count=(
                new_part_offsets[-1]
                if (
                    active_row_count is not None
                    or d_active_rows is not None
                    or ring_capacity is not None
                )
                else None
            ),
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
            fixed_size=(
                output_fixed_size
                if output_fixed_size is not None
                else (
                    fixed_size
                    if d_active_rows is None
                    else _device_fixed_size_metadata_as_bounds(fixed_size)
                )
            ),
        )

    raise NotImplementedError(f"device take not implemented for {family.value}")


def normalize_buffer_sharing_mode(mode: BufferSharingMode | str) -> BufferSharingMode:
    return mode if isinstance(mode, BufferSharingMode) else BufferSharingMode(mode)


def _is_shareable_vector(values: np.ndarray, *, dtype: np.dtype[Any]) -> bool:
    return values.dtype == dtype and values.ndim == 1 and bool(values.flags.c_contiguous)


def _is_shareable_bounds(values: np.ndarray) -> bool:
    return (
        values.dtype == np.float64
        and values.ndim == 2
        and values.shape[1] == 4
        and bool(values.flags.c_contiguous)
    )


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


def _shareable_geoarrow_buffer_view(buffer: GeoArrowBufferView) -> bool:
    return all(
        [
            isinstance(buffer.x, np.ndarray) and _is_shareable_vector(buffer.x, dtype=np.float64),
            isinstance(buffer.y, np.ndarray) and _is_shareable_vector(buffer.y, dtype=np.float64),
            isinstance(buffer.geometry_offsets, np.ndarray)
            and _is_shareable_vector(buffer.geometry_offsets, dtype=np.int32),
            isinstance(buffer.empty_mask, np.ndarray)
            and _is_shareable_vector(buffer.empty_mask, dtype=np.bool_),
            buffer.part_offsets is None
            or (
                isinstance(buffer.part_offsets, np.ndarray)
                and _is_shareable_vector(buffer.part_offsets, dtype=np.int32)
            ),
            buffer.ring_offsets is None
            or (
                isinstance(buffer.ring_offsets, np.ndarray)
                and _is_shareable_vector(buffer.ring_offsets, dtype=np.int32)
            ),
            buffer.bounds is None
            or (isinstance(buffer.bounds, np.ndarray) and _is_shareable_bounds(buffer.bounds)),
        ]
    )


def _shareable_geoarrow_view(view: MixedGeoArrowView) -> bool:
    if not all(
        [
            isinstance(view.validity, np.ndarray)
            and _is_shareable_vector(view.validity, dtype=np.bool_),
            isinstance(view.tags, np.ndarray) and _is_shareable_vector(view.tags, dtype=np.int8),
            isinstance(view.family_row_offsets, np.ndarray)
            and _is_shareable_vector(view.family_row_offsets, dtype=np.int32),
        ]
    ):
        return False
    return all(_shareable_geoarrow_buffer_view(buffer) for buffer in view.families.values())


def _build_shared_geoarrow_owned(
    view: MixedGeoArrowView,
    *,
    residency: Residency,
) -> OwnedGeometryArray:
    cached_family_buffers = view._cached_shared_family_buffers
    if cached_family_buffers is None:
        cached_family_buffers = tuple(
            (
                family,
                FamilyGeometryBuffer(
                    family=family,
                    schema=get_geometry_buffer_schema(family),
                    row_count=int(buffer.empty_mask.size),
                    x=buffer.x,
                    y=buffer.y,
                    geometry_offsets=buffer.geometry_offsets,
                    empty_mask=buffer.empty_mask,
                    part_offsets=buffer.part_offsets,
                    ring_offsets=buffer.ring_offsets,
                    bounds=buffer.bounds,
                ),
            )
            for family, buffer in view.families.items()
        )
        object.__setattr__(view, "_cached_shared_family_buffers", cached_family_buffers)
    families = dict(cached_family_buffers)
    array = OwnedGeometryArray(
        validity=view.validity,
        tags=view.tags,
        family_row_offsets=view.family_row_offsets,
        families=families,
        residency=Residency.HOST,
        geoarrow_backed=True,
        shares_geoarrow_memory=True,
    )
    array._record(
        DiagnosticKind.CREATED,
        "created owned geometry array from shared GeoArrow-style buffers",
        visible=True,
    )
    if residency is Residency.DEVICE:
        array.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="created owned geometry array with device residency requested",
        )
    return array


def _iter_coords(linear: Any) -> list[tuple[float, float]]:
    return [(float(coord[0]), float(coord[1])) for coord in linear.coords]


def _iter_geometry_parts(geometry: Any) -> list[Any]:
    return shapely.get_parts(np.asarray([geometry], dtype=object)).tolist()


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
            coords = [(float(point.x), float(point.y)) for point in _iter_geometry_parts(geometry)]
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
            for part in _iter_geometry_parts(geometry):
                state["part_offsets"].append(len(state["geometry_offsets_payload"]))
                state["geometry_offsets_payload"].extend(_iter_coords(part))
        return

    if family is GeometryFamily.MULTIPOLYGON:
        state["geometry_offsets"].append(len(state["part_offsets"]))
        if not empty:
            for polygon in _iter_geometry_parts(geometry):
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
    - MultiPolygon (type_id 6)
    - Mixed arrays containing only the above four types
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

    # Only handle Point(0), LineString(1), Polygon(3), MultiPolygon(6).
    _SUPPORTED_TYPE_IDS = np.array([0, 1, 3, 6])
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
        6: GeometryFamily.MULTIPOLYGON,
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
                x=x,
                y=y,
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
                x=x,
                y=y,
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
                rings_per_ne = np.bincount(ring_parents, minlength=len(fam_non_empty)).astype(
                    np.int32
                )
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
                x=x,
                y=y,
                geometry_offsets=go,
                empty_mask=fam_empty.copy(),
                ring_offsets=ro,
            )

        elif family is GeometryFamily.MULTIPOLYGON:
            if len(fam_non_empty) > 0:
                polygons, polygon_parents = shapely.get_parts(
                    fam_non_empty,
                    return_index=True,
                )
                rings, ring_parents = shapely.get_rings(
                    polygons,
                    return_index=True,
                )
                ring_coords = shapely.get_coordinates(rings)
                x = ring_coords[:, 0].astype(np.float64, copy=True)
                y = ring_coords[:, 1].astype(np.float64, copy=True)

                ring_num_coords = shapely.get_num_coordinates(rings)
                ro = np.zeros(len(rings) + 1, dtype=np.int32)
                np.cumsum(ring_num_coords, out=ro[1:])

                rings_per_polygon = np.bincount(
                    ring_parents,
                    minlength=len(polygons),
                ).astype(np.int32)
                po = np.zeros(len(polygons) + 1, dtype=np.int32)
                np.cumsum(rings_per_polygon, out=po[1:])

                polygons_per_ne = np.bincount(
                    polygon_parents,
                    minlength=len(fam_non_empty),
                ).astype(np.int32)
            else:
                x = np.empty(0, dtype=np.float64)
                y = np.empty(0, dtype=np.float64)
                ro = np.zeros(1, dtype=np.int32)
                po = np.zeros(1, dtype=np.int32)
                polygons_per_ne = np.empty(0, dtype=np.int32)

            all_polygons_per = np.zeros(fam_count, dtype=np.int32)
            all_polygons_per[~fam_empty] = polygons_per_ne
            go = np.zeros(fam_count + 1, dtype=np.int32)
            np.cumsum(all_polygons_per, out=go[1:])
            families[family] = FamilyGeometryBuffer(
                family=family,
                schema=get_geometry_buffer_schema(family),
                row_count=fam_count,
                x=x,
                y=y,
                geometry_offsets=go,
                empty_mask=fam_empty.copy(),
                part_offsets=po,
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
    array._record(
        DiagnosticKind.CREATED, "created owned geometry array from shapely input", visible=True
    )
    if residency is Residency.DEVICE:
        array.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="created owned geometry array with device residency requested",
        )
    return array


def build_null_owned_array(
    row_count: int,
    *,
    residency: Residency = Residency.HOST,
) -> OwnedGeometryArray:
    """Build an all-null OwnedGeometryArray without materializing Shapely."""
    validity = np.zeros(int(row_count), dtype=np.bool_)
    tags = np.full(int(row_count), NULL_TAG, dtype=np.int8)
    family_row_offsets = np.full(int(row_count), -1, dtype=np.int32)

    if residency is Residency.DEVICE:
        if cp is None:  # pragma: no cover - guarded by GPU callers
            raise RuntimeError("CuPy is required for device-resident null owned arrays")
        return build_device_resident_owned(
            device_families={},
            row_count=int(row_count),
            tags=cp.asarray(tags),
            validity=cp.asarray(validity),
            family_row_offsets=cp.asarray(family_row_offsets),
        )

    array = OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families={},
        residency=Residency.HOST,
    )
    array._record(
        DiagnosticKind.CREATED,
        f"created null owned geometry array, {row_count} rows",
        visible=False,
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
    array._record(
        DiagnosticKind.CREATED, "created owned geometry array from WKB input", visible=True
    )
    return array


def from_geoarrow(
    view: MixedGeoArrowView,
    *,
    residency: Residency = Residency.HOST,
    sharing: BufferSharingMode | str = BufferSharingMode.COPY,
) -> OwnedGeometryArray:
    sharing_mode = normalize_buffer_sharing_mode(sharing)
    if sharing_mode is BufferSharingMode.SHARE and view.shares_memory:
        return _build_shared_geoarrow_owned(view, residency=residency)
    if sharing_mode is BufferSharingMode.AUTO and view.shares_memory:
        return _build_shared_geoarrow_owned(view, residency=residency)
    shareable_view = _shareable_geoarrow_view(view)
    if sharing_mode is BufferSharingMode.SHARE:
        if not shareable_view:
            raise ValueError("GeoArrow view is not shareable in the owned-buffer schema")
        return _build_shared_geoarrow_owned(view, residency=residency)
    if sharing_mode is BufferSharingMode.AUTO and shareable_view:
        return _build_shared_geoarrow_owned(view, residency=residency)
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
        empty_mask, empty_shared = _adopt_vector(
            buffer.empty_mask, dtype=np.bool_, sharing=sharing_mode
        )
        if buffer.part_offsets is None:
            part_offsets = None
            part_shared = True
        else:
            part_offsets, part_shared = _adopt_vector(
                buffer.part_offsets, dtype=np.int32, sharing=sharing_mode
            )
        if buffer.ring_offsets is None:
            ring_offsets = None
            ring_shared = True
        else:
            ring_offsets, ring_shared = _adopt_vector(
                buffer.ring_offsets, dtype=np.int32, sharing=sharing_mode
            )
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
    family_row_offsets, offsets_shared = _adopt_vector(
        view.family_row_offsets, dtype=np.int32, sharing=sharing_mode
    )
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
    When both inputs are device-resident, dispatches to
    :func:`device_concat_owned_scatter` so the result stays on GPU.
    """
    if base.residency is Residency.DEVICE and replacement.residency is Residency.DEVICE:
        return device_concat_owned_scatter(base, replacement, indices)

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
                    base_family_count,
                    dtype=np.int32,
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


def concatenate_owned_arrays(arrays: list[OwnedGeometryArray]) -> OwnedGeometryArray:
    """Concatenate owned geometry arrays without materializing geometry objects."""
    if len(arrays) == 1:
        return arrays[0]
    if (
        arrays
        and cp is not None
        and all(
            array.residency is Residency.DEVICE and array.device_state is not None
            for array in arrays
        )
    ):
        return OwnedGeometryArray.concat(arrays)
    if (
        arrays
        and all(array.residency is Residency.DEVICE for array in arrays)
        and all(set(array.families) == {GeometryFamily.POINT} for array in arrays)
        and all(array.device_state is not None for array in arrays)
        and all(not array.families[GeometryFamily.POINT].host_materialized for array in arrays)
        and all(np.all(array.validity) for array in arrays)
    ):
        runtime = get_cuda_runtime()
        row_count = sum(array.row_count for array in arrays)
        validity = np.ones(row_count, dtype=bool)
        tags = np.full(row_count, FAMILY_TAGS[GeometryFamily.POINT], dtype=np.int8)
        family_row_offsets = np.arange(row_count, dtype=np.int32)
        geometry_offsets = np.arange(row_count + 1, dtype=np.int32)
        empty_mask = np.zeros(row_count, dtype=bool)
        x_device = cp.concatenate(
            [array.device_state.families[GeometryFamily.POINT].x for array in arrays]
        )
        y_device = cp.concatenate(
            [array.device_state.families[GeometryFamily.POINT].y for array in arrays]
        )
        point_buffer = FamilyGeometryBuffer(
            family=GeometryFamily.POINT,
            schema=get_geometry_buffer_schema(GeometryFamily.POINT),
            row_count=row_count,
            x=np.empty(0, dtype=np.float64),
            y=np.empty(0, dtype=np.float64),
            geometry_offsets=np.empty(0, dtype=np.int32),
            empty_mask=np.empty(0, dtype=np.bool_),
            bounds=None,
            host_materialized=False,
        )
        owned = OwnedGeometryArray(
            validity=validity,
            tags=tags,
            family_row_offsets=family_row_offsets,
            families={GeometryFamily.POINT: point_buffer},
            residency=Residency.DEVICE,
            device_state=OwnedGeometryDeviceState(
                validity=runtime.from_host(validity),
                tags=runtime.from_host(tags),
                family_row_offsets=runtime.from_host(family_row_offsets),
                families={
                    GeometryFamily.POINT: DeviceFamilyGeometryBuffer(
                        family=GeometryFamily.POINT,
                        x=x_device,
                        y=y_device,
                        geometry_offsets=runtime.from_host(geometry_offsets),
                        empty_mask=runtime.from_host(empty_mask),
                        bounds=None,
                        fixed_size=DeviceFixedGeometrySizeMetadata(
                            coord_count_per_row=1,
                        ),
                    )
                },
            ),
        )
        owned._record(
            DiagnosticKind.CREATED,
            "concatenated device-resident point geometry buffers without host materialization",
            visible=True,
        )
        return owned

    def _concat_offsets(buffers: list[np.ndarray]) -> np.ndarray:
        if not buffers:
            return np.asarray([0], dtype=np.int32)
        parts = [buffers[0]]
        current = int(buffers[0][-1])
        for offsets in buffers[1:]:
            parts.append(offsets[1:] + current)
            current += int(offsets[-1])
        return np.concatenate(parts).astype(np.int32, copy=False)

    for array in arrays:
        array._ensure_host_state()

    validity = np.concatenate([array.validity for array in arrays])
    tags = np.concatenate([array.tags for array in arrays]).astype(np.int8, copy=False)
    family_row_offsets = np.full(validity.size, -1, dtype=np.int32)
    families: dict[GeometryFamily, FamilyGeometryBuffer] = {}
    for family in GeometryFamily:
        family_chunks = [array.families[family] for array in arrays if family in array.families]
        if not family_chunks:
            continue
        x = (
            np.concatenate([chunk.x for chunk in family_chunks])
            if family_chunks
            else np.asarray([], dtype=np.float64)
        )
        y = (
            np.concatenate([chunk.y for chunk in family_chunks])
            if family_chunks
            else np.asarray([], dtype=np.float64)
        )
        empty_mask = np.concatenate([chunk.empty_mask for chunk in family_chunks]).astype(
            bool, copy=False
        )
        geometry_offsets = _concat_offsets([chunk.geometry_offsets for chunk in family_chunks])
        part_offsets = None
        ring_offsets = None
        if family_chunks[0].part_offsets is not None:
            part_offsets = _concat_offsets(
                [chunk.part_offsets for chunk in family_chunks if chunk.part_offsets is not None]
            )
        if family_chunks[0].ring_offsets is not None:
            ring_offsets = _concat_offsets(
                [chunk.ring_offsets for chunk in family_chunks if chunk.ring_offsets is not None]
            )
        bounds = None
        if family_chunks[0].bounds is not None:
            bounds = np.concatenate(
                [chunk.bounds for chunk in family_chunks if chunk.bounds is not None]
            )
        families[family] = FamilyGeometryBuffer(
            family=family,
            schema=family_chunks[0].schema,
            row_count=int(empty_mask.size),
            x=x,
            y=y,
            geometry_offsets=geometry_offsets,
            empty_mask=empty_mask,
            part_offsets=part_offsets,
            ring_offsets=ring_offsets,
            bounds=bounds,
        )
        family_mask = tags == FAMILY_TAGS[family]
        family_row_offsets[family_mask] = np.arange(int(family_mask.sum()), dtype=np.int32)

    result = OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families=families,
        residency=Residency.HOST,
    )
    result._record(
        DiagnosticKind.CREATED,
        f"concatenated {len(arrays)} owned geometry arrays",
        visible=False,
    )
    return result


def _device_concat_owned_scatter_indexed_view(
    base: OwnedGeometryArray,
    replacement: OwnedGeometryArray,
    indices: DeviceArray,
) -> OwnedGeometryArray:
    """Represent device scatter as a native row-indirection carrier.

    Public scatter semantics are row-aligned: rows listed in ``indices`` come
    from ``replacement`` and all other rows come from ``base``.  The physical
    shape is not a family-buffer gather, though.  For a device rowset we can
    concatenate the two already-device-owned inputs once, then build a device
    index map selecting either the base row or the replacement row for each
    output row.  Variable nested geometry therefore stays as row metadata over
    existing buffers instead of forcing slice-size allocation fences.
    """
    d_indices = cp.asarray(indices, dtype=cp.int64)
    n_out = base.row_count
    replacement_count = replacement.row_count
    if int(d_indices.size) != replacement_count:
        raise ValueError(
            f"indices length ({int(d_indices.size)}) must equal replacement row_count "
            f"({replacement_count})"
        )

    base._ensure_device_state(preserve_indexed_view=True)
    replacement._ensure_device_state(preserve_indexed_view=True)

    d_index_map = cp.arange(n_out, dtype=cp.int64)
    if replacement_count:
        d_index_map[d_indices] = cp.arange(replacement_count, dtype=cp.int64) + np.int64(n_out)

    scatter_base = OwnedGeometryArray.concat([base, replacement])
    result = OwnedGeometryArray._indexed_view(
        scatter_base,
        d_index_map,
        assume_unique_indices=True,
    )
    result._device_scatter_implementation = "device_scatter_row_indirection"
    result._record(
        DiagnosticKind.CREATED,
        (
            f"device row-indirected scatter of {replacement_count} replacement "
            f"rows into {n_out}-row base"
        ),
        visible=False,
    )
    return result


def device_concat_owned_scatter_many(
    base: OwnedGeometryArray,
    replacements: list[tuple[OwnedGeometryArray, np.ndarray | DeviceArray]],
) -> OwnedGeometryArray:
    """Scatter multiple device replacement partitions through one index map.

    Physical shape: row-indirected native assembly.  This is the fused variant
    of repeated ``device_concat_owned_scatter`` calls for partitioned
    constructive outputs.  The base and all replacement buffers are
    concatenated once, then one device row map selects the winning row for
    each public output position.
    """
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        raise RuntimeError("CuPy is required for device scatter assembly")

    normalized: list[tuple[OwnedGeometryArray, DeviceArray]] = []
    for replacement, indices in replacements:
        d_indices = cp.asarray(indices, dtype=cp.int64)
        if d_indices.size != replacement.row_count:
            raise ValueError(
                f"indices length ({d_indices.size}) must equal replacement "
                f"row_count ({replacement.row_count})"
            )
        if replacement.row_count == 0:
            continue
        normalized.append((replacement, d_indices))

    if not normalized:
        return base
    if len(normalized) == 1:
        replacement, indices = normalized[0]
        return device_concat_owned_scatter(base, replacement, indices)

    n_out = base.row_count
    base._ensure_device_state(preserve_indexed_view=True)
    arrays: list[OwnedGeometryArray] = [base]
    d_index_map = cp.arange(n_out, dtype=cp.int64)
    source_offset = np.int64(n_out)
    for replacement, indices in normalized:
        replacement._ensure_device_state(preserve_indexed_view=True)
        d_index_map[indices] = cp.arange(replacement.row_count, dtype=cp.int64) + source_offset
        arrays.append(replacement)
        source_offset = np.int64(int(source_offset) + replacement.row_count)

    scatter_base = OwnedGeometryArray.concat(arrays)
    result = OwnedGeometryArray._indexed_view(
        scatter_base,
        d_index_map,
        assume_unique_indices=True,
    )
    result._device_scatter_implementation = "device_scatter_row_indirection_many"
    result._record(
        DiagnosticKind.CREATED,
        (
            f"device row-indirected fused scatter of {len(normalized)} "
            f"replacement partitions into {n_out}-row base"
        ),
        visible=False,
    )
    return result


def device_scatter_owned_capacity_selection(
    base: OwnedGeometryArray,
    replacement: OwnedGeometryArray,
    selection,
    *,
    active_mask: DeviceArray | None = None,
) -> OwnedGeometryArray:
    """Scatter replacement capacity through a dynamic device selection.

    ``replacement`` has one row per selection-capacity lane. Active lanes map
    to source rows through ``selection.positions``; inactive lanes map to
    scratch destinations beyond the public output. The original and replacement
    carriers are concatenated once, and the result remains row-indirected with
    no logical-count read or variable-width geometry compaction.
    """
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        raise RuntimeError("CuPy is required for device capacity scatter")

    capacity = int(selection.capacity)
    row_count = int(base.row_count)
    if replacement.row_count != capacity:
        raise ValueError("capacity replacement row count must match selection capacity")
    if selection.source_row_count is not None and int(selection.source_row_count) != row_count:
        raise ValueError("capacity selection source row count must match scatter base")

    d_active = selection.active_capacity_mask()
    if active_mask is not None:
        d_requested = cp.asarray(active_mask, dtype=cp.bool_)
        if d_requested.ndim != 1 or int(d_requested.size) != capacity:
            raise ValueError("capacity scatter active mask must match selection capacity")
        d_active &= d_requested

    if _device_concat_requires_exact_physicalization([base, replacement]):
        return device_scatter_owned_capacity_selections_many(
            base,
            [(replacement, selection, active_mask)],
        )

    base._ensure_device_state(preserve_indexed_view=True)
    replacement._ensure_device_state(preserve_indexed_view=True)
    scatter_base = OwnedGeometryArray.concat([base, replacement])

    d_lanes = cp.arange(capacity, dtype=cp.int64)
    d_destinations = cp.where(
        d_active,
        selection.safe_capacity_positions(),
        np.int64(row_count) + d_lanes,
    )
    d_extended_index_map = cp.arange(row_count + capacity, dtype=cp.int64)
    d_extended_index_map[d_destinations] = np.int64(row_count) + d_lanes
    result = OwnedGeometryArray._indexed_view(
        scatter_base,
        d_extended_index_map[:row_count],
        assume_unique_indices=True,
    )
    result._device_scatter_implementation = "device_capacity_selection_scatter"
    result._record(
        DiagnosticKind.CREATED,
        (
            "device row-indirected capacity scatter across "
            f"{capacity} replacement lanes and {row_count} output rows"
        ),
        visible=False,
    )
    return result


def _device_concat_requires_exact_physicalization(
    arrays: list[OwnedGeometryArray],
) -> bool:
    """Whether a multi-root concat lacks one guaranteed contiguous allocation.

    RMM's pool can have enough aggregate free bytes while cached blocks cannot
    satisfy the concat's sequence of output allocations. Only growth admitted
    by both the pool ceiling and the CUDA driver is guaranteed upstream
    capacity in that state. Route the scatter through the exact multi-root
    physicalizer when the complete new carrier exceeds that guarantee; its
    compact allocation packet avoids copying inactive physical capacity and
    the fragmented-pool OOM.
    """
    if cp is None or len(arrays) < 2:
        return False

    roots: list[OwnedGeometryArray] = []
    seen_roots: set[int] = set()
    for array in arrays:
        root = array
        while root.is_indexed_view:
            if root._base is None:
                raise RuntimeError("indexed device geometry is missing its physical base")
            root = root._base
        if id(root) not in seen_roots:
            roots.append(root)
            seen_roots.add(id(root))
    if len(roots) < 2:
        return False

    states = [root._ensure_device_state(preserve_indexed_view=True) for root in roots]
    concat_bytes = sum(
        sum(int(getattr(state, name).nbytes) for state in states)
        for name in ("validity", "tags", "family_row_offsets")
    )
    row_bounds_bytes = sum(
        int(state.row_bounds.nbytes)
        for state in states
        if state.row_bounds is not None
    )
    concat_bytes += row_bounds_bytes

    families = set().union(*(state.families for state in states))
    for family in families:
        for name in (
            "x",
            "y",
            "geometry_offsets",
            "empty_mask",
            "part_offsets",
            "ring_offsets",
            "bounds",
        ):
            concat_bytes += sum(
                int(getattr(state.families[family], name).nbytes)
                for state in states
                if family in state.families
                and getattr(state.families[family], name) is not None
            )

    upstream_growth_bytes = get_cuda_runtime().pool_upstream_growth_bytes()
    return concat_bytes > upstream_growth_bytes


def _device_geometry_has_device_root(array: OwnedGeometryArray) -> bool:
    """Whether an owned carrier resolves through device row maps to a device root."""
    current = array
    while current.is_indexed_view:
        if (
            current._base is None
            or current._index_map is None
            or not hasattr(current._index_map, "__cuda_array_interface__")
        ):
            return False
        current = current._base
    return current.device_state is not None


def device_scatter_owned_capacity_selections_many(
    base: OwnedGeometryArray,
    replacements: list[tuple[OwnedGeometryArray, Any, DeviceArray | None]],
) -> OwnedGeometryArray:
    """Fuse multi-root capacity scatters through one exact physicalization.

    A single-root ``OwnedGeometryArray`` cannot retain row indirection into
    several unrelated geometry buffers. This is the explicit physical-layout
    boundary for that shape: all active logical rows are sized by one sparse
    device packet, copied once, concatenated once, and selected by one row map.
    """
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        raise RuntimeError("CuPy is required for fused device capacity scatter")

    row_count = int(base.row_count)
    normalized: list[tuple[OwnedGeometryArray, Any, DeviceArray]] = []
    exact_selections: list[tuple[OwnedGeometryArray, DeviceArray]] = [
        (base, cp.ones(row_count, dtype=cp.bool_))
    ]
    for replacement, selection, active_mask in replacements:
        capacity = int(selection.capacity)
        if replacement.row_count != capacity:
            raise ValueError("capacity replacement row count must match selection capacity")
        if selection.source_row_count is not None and int(selection.source_row_count) != row_count:
            raise ValueError("capacity selection source row count must match scatter base")
        d_active = selection.active_capacity_mask()
        if active_mask is not None:
            d_requested = cp.asarray(active_mask, dtype=cp.bool_)
            if d_requested.ndim != 1 or d_requested.size != capacity:
                raise ValueError("capacity scatter active mask must match selection capacity")
            d_active &= d_requested
        normalized.append((replacement, selection, d_active))
        exact_selections.append((replacement, d_active))

    physical = device_physicalize_owned_row_selections_exact(
        exact_selections,
        reason="fused multi-root capacity scatter exact allocation packet",
    )
    physical_base = physical[0] if physical[0] is not None else base
    arrays = [physical_base]
    d_index_map = cp.arange(row_count, dtype=cp.int64)
    source_offset = np.int64(row_count)
    emitted = 0
    for (replacement, selection, d_active), physical_replacement in zip(
        normalized,
        physical[1:],
        strict=True,
    ):
        if physical_replacement is None:
            continue
        capacity = replacement.row_count
        d_lanes = cp.arange(capacity, dtype=cp.int64)
        d_destinations = cp.where(
            d_active,
            selection.safe_capacity_positions(),
            np.int64(row_count) + d_lanes,
        )
        d_extended_map = cp.concatenate(
            [d_index_map, cp.zeros(capacity, dtype=cp.int64)]
        )
        d_extended_map[d_destinations] = source_offset + d_lanes
        d_index_map = d_extended_map[:row_count]
        arrays.append(physical_replacement)
        source_offset += np.int64(capacity)
        emitted += 1

    if emitted == 0:
        return physical_base
    scatter_base = OwnedGeometryArray.concat(arrays)
    result = OwnedGeometryArray._indexed_view(
        scatter_base,
        d_index_map,
        assume_unique_indices=True,
    )
    result._device_scatter_implementation = "device_exact_capacity_selection_scatter_many"
    result._record(
        DiagnosticKind.CREATED,
        (
            f"device exact fused capacity scatter of {emitted} replacement "
            f"partitions into {row_count} output rows"
        ),
        visible=False,
    )
    return result


def device_take_owned_family_capacity_selection(
    owned: OwnedGeometryArray,
    selection,
    family: GeometryFamily,
) -> OwnedGeometryArray:
    """Build one homogeneous row-indirected family carrier at selection capacity.

    The selected prefix gathers only row metadata. Coordinates and structural
    offsets remain in the source family buffer and are addressed through the
    gathered family-row offsets. Rejected tail lanes are null. This is the
    native layout for family kernels: logical row capacity over shared physical
    family storage, with no variable-width copy or cardinality fence.
    """
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        raise RuntimeError("CuPy is required for device family capacity take")

    state = owned._ensure_device_state(preserve_indexed_view=True)
    if family not in state.families:
        raise ValueError(f"family {family.value} is absent from device geometry")

    capacity = int(selection.capacity)
    if selection.source_row_count is not None and int(selection.source_row_count) != int(
        owned.row_count
    ):
        raise ValueError("capacity selection source row count mismatch")

    d_active = selection.active_capacity_mask()
    d_positions = cp.asarray(
        selection.partition_capacity_positions(),
        dtype=cp.int64,
    )
    d_source_validity = cp.asarray(state.validity, dtype=cp.bool_)[d_positions]
    d_validity = d_active & d_source_validity
    d_source_family_rows = cp.asarray(
        state.family_row_offsets,
        dtype=cp.int64,
    )[d_positions]
    d_family_rows = cp.where(
        d_validity,
        d_source_family_rows,
        cp.int64(0),
    ).astype(cp.int32, copy=False)
    d_tags = cp.where(
        d_validity,
        cp.int8(FAMILY_TAGS[family]),
        cp.int8(NULL_TAG),
    )

    result = build_device_resident_owned(
        device_families={family: state.families[family]},
        row_count=capacity,
        tags=d_tags,
        validity=d_validity,
        family_row_offsets=d_family_rows,
        execution_mode="gpu",
    )
    result_state = result._ensure_device_state(preserve_indexed_view=True)
    if state.row_bounds is not None:
        d_source_bounds = cp.asarray(state.row_bounds, dtype=cp.float64).reshape(
            owned.row_count,
            4,
        )[d_positions]
        result_state.row_bounds = cp.where(
            d_validity[:, None],
            d_source_bounds,
            cp.asarray(cp.nan, dtype=cp.float64),
        )
    result_state.trusted_all_valid = True if capacity == 0 else False
    result_state.trusted_all_ogc_valid = (
        True if capacity == 0 or state.trusted_all_ogc_valid is True else None
    )
    result_state.trusted_homogeneous_family = family
    result_state.trusted_all_non_empty = None
    result_state.trusted_polygonal_only = (
        True if family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON) else False
    )
    result_state.trusted_unique_family_rows = bool(
        state.trusted_unique_family_rows is True and getattr(selection, "unique", False)
    )
    result_state.trusted_family_domain = (family,)
    result._record(
        DiagnosticKind.CREATED,
        (
            f"device row-indirected family selection retained {capacity} "
            f"{family.value} capacity lanes"
        ),
        visible=False,
    )
    return result


def device_take_owned_capacity_selection(
    owned: OwnedGeometryArray,
    selection,
) -> OwnedGeometryArray:
    """Retain a dynamic selection as a row-indirected geometry capacity.

    ``NativeDeviceSelection`` carries a stable full partition, not just the
    selected prefix.  Addressing that partition keeps every source row unique
    in physical storage while the separate activity mask nulls the inactive
    tail.  This avoids copying the first selected variable-width geometry into
    every inactive lane, which can multiply nested allocation capacity.
    """
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        raise RuntimeError("CuPy is required for device capacity selection")
    if selection.source_row_count is not None and int(selection.source_row_count) != int(
        owned.row_count
    ):
        raise ValueError("capacity selection source row count mismatch")

    unique = bool(getattr(selection, "unique", False))
    result = owned._device_indexed_take(
        cp.asarray(
            selection.partition_capacity_positions(),
            dtype=cp.int64,
        ),
        assume_unique_indices=unique,
    )
    return result._apply_row_activity(
        selection.active_capacity_mask(),
        assume_active_indices_unique=unique,
    )


def device_select_owned_capacity_partitions(
    base: OwnedGeometryArray,
    replacements: list[tuple[OwnedGeometryArray, DeviceArray]],
) -> OwnedGeometryArray:
    """Select row-aligned capacity partitions through one device index map.

    Every replacement has the public row capacity of ``base``. Its mask marks
    lanes that own the logical output. Variable-width buffers are concatenated
    once and remain row-indirected; no partition cardinality is materialized.
    Later replacements win if masks overlap.
    """
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        raise RuntimeError("CuPy is required for device capacity partition assembly")

    row_count = int(base.row_count)
    if not replacements:
        return base
    arrays: list[OwnedGeometryArray] = [base]
    d_index_map = cp.arange(row_count, dtype=cp.int64)
    source_offset = np.int64(row_count)
    for replacement, active_mask in replacements:
        if int(replacement.row_count) != row_count:
            raise ValueError("capacity replacement row count must match base row count")
        d_active = cp.asarray(active_mask, dtype=cp.bool_)
        if d_active.ndim != 1 or d_active.size != row_count:
            raise ValueError("capacity replacement mask must be one-dimensional and row-aligned")
        replacement._ensure_device_state(preserve_indexed_view=True)
        replacement_rows = cp.arange(row_count, dtype=cp.int64) + source_offset
        d_index_map = cp.where(d_active, replacement_rows, d_index_map)
        arrays.append(replacement)
        source_offset = np.int64(int(source_offset) + row_count)

    base._ensure_device_state(preserve_indexed_view=True)
    selection_base = OwnedGeometryArray.concat(arrays)
    result = OwnedGeometryArray._indexed_view(
        selection_base,
        d_index_map,
        assume_unique_indices=True,
    )
    d_validity_proof = cp.zeros(row_count, dtype=cp.bool_)
    base_proof = getattr(base, "_device_ogc_validity_proof", None)
    if base_proof is not None and int(cp.asarray(base_proof).size) == row_count:
        d_validity_proof = cp.asarray(base_proof, dtype=cp.bool_).copy()
    for replacement, active_mask in replacements:
        replacement_proof = getattr(
            replacement,
            "_device_ogc_validity_proof",
            None,
        )
        d_replacement_proof = (
            cp.zeros(row_count, dtype=cp.bool_)
            if replacement_proof is None or int(cp.asarray(replacement_proof).size) != row_count
            else cp.asarray(replacement_proof, dtype=cp.bool_)
        )
        d_validity_proof = cp.where(
            cp.asarray(active_mask, dtype=cp.bool_),
            d_replacement_proof,
            d_validity_proof,
        )
    result._device_ogc_validity_proof = d_validity_proof
    if all(
        array._current_cached_validity_mask() is not None
        and bool(np.all(array._current_cached_validity_mask()))
        and getattr(array, "_cached_is_valid_exact_collinearity_mask", None) is not None
        and bool(
            np.all(
                np.asarray(
                    array._cached_is_valid_exact_collinearity_mask,
                    dtype=bool,
                )
            )
        )
        for array in arrays
    ):
        seed_all_validity_cache(result)
    result._device_scatter_implementation = "device_capacity_partition_selection"
    result._record(
        DiagnosticKind.CREATED,
        (
            f"device row-indirected selection of {len(replacements)} "
            f"capacity partitions across {row_count} rows"
        ),
        visible=False,
    )
    return result


def device_physical_select_owned_capacity_partitions(
    base: OwnedGeometryArray,
    replacements: list[tuple[OwnedGeometryArray, DeviceArray]],
) -> OwnedGeometryArray:
    """Select row-aligned partitions into a bounded physical device carrier.

    Iterative constructive reductions must not retain every prior selection
    source in an indexed-view ancestry graph.  This explicit physical-layout
    transition performs the row selection without exporting cardinality, then
    gathers each selected variable-width row once.  The selection map is
    injective across concatenated sources, so nested allocation is bounded by
    the source carriers rather than multiplied by output-row capacity.
    """
    selected = device_select_owned_capacity_partitions(base, replacements)
    if not selected.is_indexed_view:
        return selected
    result = selected.physicalize_device_rows(allow_capacity_allocation=True)
    result._device_scatter_implementation = "device_capacity_partition_physicalization"
    return result


def device_mask_owned_capacity(
    owned: OwnedGeometryArray,
    active_mask: DeviceArray,
    *,
    preserve_row_bounds: bool = True,
) -> OwnedGeometryArray:
    """Create a null-padded row-capacity view over physical device buffers.

    The geometry buffers are shared. Only row-aligned validity, tags, family
    row offsets, and cached bounds are rewritten, so complementary constructive
    partitions can consume the same physical rows without a sparse take or a
    variable-width coordinate copy.
    """
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        raise RuntimeError("CuPy is required for device capacity masking")
    row_count = int(owned.row_count)
    d_active = cp.asarray(active_mask, dtype=cp.bool_)
    if d_active.ndim != 1 or int(d_active.size) != row_count:
        raise ValueError("capacity mask must be one-dimensional and row-aligned")
    if owned.is_indexed_view:
        result = owned._device_indexed_take(
            cp.arange(row_count, dtype=cp.int64),
            assume_unique_indices=True,
        )._apply_row_activity(
            d_active,
            preserve_row_bounds=preserve_row_bounds,
        )
        result._record(
            DiagnosticKind.CREATED,
            f"device capacity mask retained {row_count} indexed row lanes",
            visible=False,
        )
        return result
    state = owned._ensure_device_state(preserve_indexed_view=True)
    d_validity = cp.asarray(state.validity, dtype=cp.bool_) & d_active
    d_tags = cp.where(
        d_validity,
        cp.asarray(state.tags, dtype=cp.int8),
        cp.int8(-1),
    )
    d_family_rows = cp.where(
        d_validity,
        cp.asarray(state.family_row_offsets, dtype=cp.int32),
        cp.int32(-1),
    )
    d_row_bounds = None
    if preserve_row_bounds and state.row_bounds is not None:
        d_row_bounds = cp.where(
            d_validity[:, None],
            cp.asarray(state.row_bounds, dtype=cp.float64).reshape(row_count, 4),
            cp.asarray(cp.nan, dtype=cp.float64),
        )

    result = OwnedGeometryArray(
        validity=None,
        tags=None,
        family_row_offsets=None,
        families=owned.families,
        residency=Residency.DEVICE,
        device_state=OwnedGeometryDeviceState(
            validity=d_validity,
            tags=d_tags,
            family_row_offsets=d_family_rows,
            families=dict(state.families),
            row_bounds=d_row_bounds,
            trusted_all_valid=True if row_count == 0 else False,
            trusted_all_ogc_valid=(
                True if row_count == 0 or state.trusted_all_ogc_valid is True else None
            ),
            trusted_homogeneous_family=state.trusted_homogeneous_family,
            trusted_all_non_empty=None,
            trusted_all_finite_coordinates=(
                True if state.trusted_all_finite_coordinates is True else None
            ),
            trusted_nonempty_polygonal_positive_area=(
                state.trusted_nonempty_polygonal_positive_area
            ),
            trusted_polygonal_only=state.trusted_polygonal_only,
            trusted_unique_family_rows=state.trusted_unique_family_rows,
            trusted_family_domain=state.trusted_family_domain,
        ),
        _row_count=row_count,
    )
    result._active_family_row_segment_capacity_bound = (
        owned._active_family_row_segment_capacity_bound
    )
    result._record(
        DiagnosticKind.CREATED,
        f"device capacity mask retained {row_count} physical row lanes",
        visible=False,
    )
    return result


def _device_concat_scatter_prefers_row_indirection(
    base: OwnedGeometryArray,
    replacement: OwnedGeometryArray,
) -> bool:
    """Return whether scatter assembly should preserve a row-indirected carrier."""
    for owned in (base, replacement):
        state = owned._ensure_device_state(preserve_indexed_view=True)
        for family, device_buffer in state.families.items():
            if family is GeometryFamily.POINT:
                continue
            if not _device_buffer_has_exact_row_width(family, device_buffer):
                return True
    return False


def device_concat_owned_scatter(
    base: OwnedGeometryArray,
    replacement: OwnedGeometryArray,
    indices: np.ndarray | DeviceArray,
) -> OwnedGeometryArray:
    """Scatter *replacement* rows into *base* without leaving the device."""
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        raise RuntimeError("CuPy is required for device scatter assembly")

    host_indices = None
    if not hasattr(indices, "__cuda_array_interface__"):
        host_indices = np.asarray(indices, dtype=np.int64)
    indices = cp.asarray(indices, dtype=cp.int64)
    n_out = base.row_count
    if int(indices.size) != replacement.row_count:
        raise ValueError(
            f"indices length ({int(indices.size)}) must equal replacement row_count "
            f"({replacement.row_count})"
        )
    if host_indices is None or _device_concat_scatter_prefers_row_indirection(
        base,
        replacement,
    ):
        return _device_concat_owned_scatter_indexed_view(base, replacement, indices)

    base_state = base._ensure_device_state(preserve_indexed_view=True)
    replacement_state = replacement._ensure_device_state(preserve_indexed_view=True)

    out_validity = cp.asarray(base_state.validity).copy()
    out_tags = cp.asarray(base_state.tags).copy()
    out_validity[indices] = replacement_state.validity
    out_tags[indices] = replacement_state.tags
    out_row_bounds = None
    if base_state.row_bounds is not None and replacement_state.row_bounds is not None:
        out_row_bounds = (
            cp.asarray(base_state.row_bounds, dtype=cp.float64)
            .reshape(
                n_out,
                4,
            )
            .copy()
        )
        out_row_bounds[indices] = cp.asarray(
            replacement_state.row_bounds,
            dtype=cp.float64,
        ).reshape(replacement.row_count, 4)

    is_replacement = cp.zeros(n_out, dtype=cp.bool_)
    is_replacement[indices] = True
    inv_map = cp.full(n_out, -1, dtype=cp.int64)
    inv_map[indices] = cp.arange(replacement.row_count, dtype=cp.int64)

    out_family_row_offsets = cp.full(n_out, -1, dtype=cp.int32)
    out_families: dict[GeometryFamily, DeviceFamilyGeometryBuffer] = {}

    host_out_validity = None
    host_out_tags = None
    host_out_family_row_offsets = None
    host_is_replacement = None
    host_inv_map = None
    if (
        host_indices is not None
        and base._validity is not None
        and base._tags is not None
        and base._family_row_offsets is not None
        and replacement._validity is not None
        and replacement._tags is not None
        and replacement._family_row_offsets is not None
    ):
        host_out_validity = np.asarray(base._validity, dtype=np.bool_).copy()
        host_out_tags = np.asarray(base._tags, dtype=np.int8).copy()
        host_out_validity[host_indices] = np.asarray(
            replacement._validity,
            dtype=np.bool_,
        )
        host_out_tags[host_indices] = np.asarray(replacement._tags, dtype=np.int8)
        host_is_replacement = np.zeros(n_out, dtype=bool)
        host_is_replacement[host_indices] = True
        host_inv_map = np.full(n_out, -1, dtype=np.int64)
        host_inv_map[host_indices] = np.arange(replacement.row_count, dtype=np.int64)
        host_out_family_row_offsets = np.full(n_out, -1, dtype=np.int32)

    same_single_family = (
        n_out > 0
        and len(base_state.families) == 1
        and len(replacement_state.families) == 1
        and set(base_state.families) == set(replacement_state.families)
    )
    if same_single_family:
        family = next(iter(base_state.families))
        base_device_buffer = base_state.families[family]
        replacement_device_buffer = replacement_state.families[family]
        base_family_covers_all_rows = int(base_device_buffer.geometry_offsets.size) - 1 == int(
            base.row_count
        )
        replacement_family_covers_all_rows = int(
            replacement_device_buffer.geometry_offsets.size
        ) - 1 == int(replacement.row_count)

    if same_single_family and base_family_covers_all_rows and replacement_family_covers_all_rows:
        family_buffers: list[DeviceFamilyGeometryBuffer] = []
        base_family_count = 0

        base_global = cp.flatnonzero(~is_replacement).astype(cp.int64, copy=False)
        if int(base_global.size) > 0:
            base_rows = cp.asarray(
                base_state.family_row_offsets[base_global],
                dtype=cp.int64,
            )
            base_taken = _device_take_family_buffer(
                base_state.families[family],
                family,
                base_rows,
                base.families.get(family),
                host_family_rows=(
                    None
                    if host_is_replacement is None or base._family_row_offsets is None
                    else np.asarray(
                        base._family_row_offsets[
                            np.flatnonzero(~host_is_replacement).astype(
                                np.int64,
                                copy=False,
                            )
                        ],
                        dtype=np.int64,
                    )
                ),
            )
            family_buffers.append(base_taken)
            base_family_count = int(base_taken.geometry_offsets.size) - 1
            out_family_row_offsets[base_global] = cp.arange(
                base_family_count,
                dtype=cp.int32,
            )
            if host_out_family_row_offsets is not None and host_is_replacement is not None:
                host_base_global = np.flatnonzero(~host_is_replacement).astype(
                    np.int64,
                    copy=False,
                )
                host_out_family_row_offsets[host_base_global] = np.arange(
                    base_family_count,
                    dtype=np.int32,
                )

        repl_global = cp.flatnonzero(is_replacement).astype(cp.int64, copy=False)
        if int(repl_global.size) > 0:
            repl_local = inv_map[repl_global]
            repl_rows = cp.asarray(
                replacement_state.family_row_offsets[repl_local],
                dtype=cp.int64,
            )
            repl_taken = _device_take_family_buffer(
                replacement_state.families[family],
                family,
                repl_rows,
                replacement.families.get(family),
                host_family_rows=(
                    None
                    if (
                        host_is_replacement is None
                        or host_inv_map is None
                        or replacement._family_row_offsets is None
                    )
                    else np.asarray(
                        replacement._family_row_offsets[
                            host_inv_map[
                                np.flatnonzero(host_is_replacement).astype(
                                    np.int64,
                                    copy=False,
                                )
                            ]
                        ],
                        dtype=np.int64,
                    )
                ),
            )
            family_buffers.append(repl_taken)
            repl_family_count = int(repl_taken.geometry_offsets.size) - 1
            out_family_row_offsets[repl_global] = cp.arange(
                base_family_count,
                base_family_count + repl_family_count,
                dtype=cp.int32,
            )
            if host_out_family_row_offsets is not None and host_is_replacement is not None:
                host_repl_global = np.flatnonzero(host_is_replacement).astype(
                    np.int64,
                    copy=False,
                )
                host_out_family_row_offsets[host_repl_global] = np.arange(
                    base_family_count,
                    base_family_count + repl_family_count,
                    dtype=np.int32,
                )

        result = build_device_resident_owned(
            device_families={family: _concat_device_family_buffers(family, family_buffers)},
            row_count=n_out,
            tags=out_tags,
            validity=out_validity,
            family_row_offsets=out_family_row_offsets,
            execution_mode="gpu",
        )
        if result.device_state is not None:
            result.device_state.row_bounds = out_row_bounds
        if (
            host_out_validity is not None
            and host_out_tags is not None
            and host_out_family_row_offsets is not None
        ):
            result._validity = host_out_validity
            result._tags = host_out_tags
            result._family_row_offsets = host_out_family_row_offsets
        result._record(
            DiagnosticKind.CREATED,
            f"device scatter {replacement.row_count} replacement rows into {base.row_count}-row homogeneous base",
            visible=False,
        )
        return result

    all_families = set(base_state.families) | set(replacement_state.families)
    for family in sorted(all_families, key=lambda item: FAMILY_TAGS[item]):
        family_global_indices = cp.flatnonzero(out_tags == FAMILY_TAGS[family]).astype(
            cp.int64,
            copy=False,
        )
        if int(family_global_indices.size) == 0:
            continue

        family_is_replacement = is_replacement[family_global_indices]
        family_buffers: list[DeviceFamilyGeometryBuffer] = []
        base_family_count = 0
        repl_family_count = 0

        if family in base_state.families:
            if host_out_tags is not None and host_is_replacement is not None:
                host_base_global = np.flatnonzero(
                    (host_out_tags == np.int8(FAMILY_TAGS[family])) & ~host_is_replacement
                ).astype(np.int64, copy=False)
                family_has_base_rows = bool(host_base_global.size)
                base_global = family_global_indices[~family_is_replacement]
            else:
                host_base_global = None
                base_global = family_global_indices[~family_is_replacement]
                family_has_base_rows = base_global.size > 0
        else:
            host_base_global = None
            base_global = cp.empty(0, dtype=cp.int64)
            family_has_base_rows = False

        if family_has_base_rows:
            base_rows = cp.asarray(
                base_state.family_row_offsets[base_global],
                dtype=cp.int64,
            )
            host_base_rows = None
            if (
                host_out_tags is not None
                and host_is_replacement is not None
                and base._family_row_offsets is not None
            ):
                if host_base_global is None:
                    host_base_global = np.flatnonzero(
                        (host_out_tags == np.int8(FAMILY_TAGS[family])) & ~host_is_replacement
                    ).astype(np.int64, copy=False)
                host_base_rows = np.asarray(
                    base._family_row_offsets[host_base_global], dtype=np.int64
                )
            base_taken = _device_take_family_buffer(
                base_state.families[family],
                family,
                base_rows,
                base.families.get(family),
                host_family_rows=host_base_rows,
            )
            family_buffers.append(base_taken)
            base_family_count = int(base_taken.geometry_offsets.size) - 1
            out_family_row_offsets[base_global] = cp.arange(
                base_family_count,
                dtype=cp.int32,
            )
            if host_out_family_row_offsets is not None and host_base_rows is not None:
                host_out_family_row_offsets[host_base_global] = np.arange(
                    base_family_count,
                    dtype=np.int32,
                )

        if family in replacement_state.families:
            if host_out_tags is not None and host_is_replacement is not None:
                host_repl_global = np.flatnonzero(
                    (host_out_tags == np.int8(FAMILY_TAGS[family])) & host_is_replacement
                ).astype(np.int64, copy=False)
                family_has_replacement_rows = bool(host_repl_global.size)
                repl_global = family_global_indices[family_is_replacement]
            else:
                host_repl_global = None
                repl_global = family_global_indices[family_is_replacement]
                family_has_replacement_rows = repl_global.size > 0
        else:
            host_repl_global = None
            repl_global = cp.empty(0, dtype=cp.int64)
            family_has_replacement_rows = False

        if family_has_replacement_rows:
            repl_local = inv_map[repl_global]
            repl_rows = cp.asarray(
                replacement_state.family_row_offsets[repl_local],
                dtype=cp.int64,
            )
            host_repl_rows = None
            if (
                host_out_tags is not None
                and host_is_replacement is not None
                and host_inv_map is not None
                and replacement._family_row_offsets is not None
            ):
                if host_repl_global is None:
                    host_repl_global = np.flatnonzero(
                        (host_out_tags == np.int8(FAMILY_TAGS[family])) & host_is_replacement
                    ).astype(np.int64, copy=False)
                host_repl_local = host_inv_map[host_repl_global]
                host_repl_rows = np.asarray(
                    replacement._family_row_offsets[host_repl_local],
                    dtype=np.int64,
                )
            repl_taken = _device_take_family_buffer(
                replacement_state.families[family],
                family,
                repl_rows,
                replacement.families.get(family),
                host_family_rows=host_repl_rows,
            )
            family_buffers.append(repl_taken)
            repl_family_count = int(repl_taken.geometry_offsets.size) - 1
            out_family_row_offsets[repl_global] = cp.arange(
                base_family_count,
                base_family_count + repl_family_count,
                dtype=cp.int32,
            )
            if host_out_family_row_offsets is not None and host_repl_rows is not None:
                host_out_family_row_offsets[host_repl_global] = np.arange(
                    base_family_count,
                    base_family_count + repl_family_count,
                    dtype=np.int32,
                )

        if family_buffers:
            out_families[family] = _concat_device_family_buffers(
                family,
                family_buffers,
            )

    result = build_device_resident_owned(
        device_families=out_families,
        row_count=n_out,
        tags=out_tags,
        validity=out_validity,
        family_row_offsets=out_family_row_offsets,
        execution_mode="gpu",
    )
    if result.device_state is not None:
        result.device_state.row_bounds = out_row_bounds
    result._record(
        DiagnosticKind.CREATED,
        f"device scatter {replacement.row_count} replacement rows into {base.row_count}-row base",
        visible=False,
    )
    return result


def build_device_resident_owned(
    *,
    device_families: dict[GeometryFamily, DeviceFamilyGeometryBuffer],
    row_count: int,
    tags: np.ndarray | DeviceArray,
    validity: np.ndarray | DeviceArray,
    family_row_offsets: np.ndarray | DeviceArray,
    execution_mode: str | None = None,
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
    execution_mode
        Optional execution mode marker. When set to ``"gpu"``, host numpy
        metadata arrays are rejected so GPU-path callers cannot silently
        re-upload metadata through this factory.
    """
    from vibespatial.cuda._runtime import get_cuda_runtime

    runtime = get_cuda_runtime()

    if execution_mode == "gpu":
        host_metadata = [
            name
            for name, value in (
                ("tags", tags),
                ("validity", validity),
                ("family_row_offsets", family_row_offsets),
            )
            if isinstance(value, np.ndarray)
        ]
        if host_metadata:
            caller = inspect.stack()[1]
            raise AssertionError(
                "GPU build_device_resident_owned() received host metadata "
                f"{', '.join(host_metadata)} from {caller.filename}:{caller.lineno}"
            )

    # Build host-side placeholder families with host_materialized=False
    host_families: dict[GeometryFamily, FamilyGeometryBuffer] = {}
    for family in device_families:
        schema = get_geometry_buffer_schema(family)
        host_families[family] = FamilyGeometryBuffer(
            family=family,
            schema=schema,
            row_count=int(device_families[family].geometry_offsets.size - 1),
            x=np.empty(0, dtype=np.float64),
            y=np.empty(0, dtype=np.float64),
            geometry_offsets=np.empty(0, dtype=np.int32),
            empty_mask=np.empty(0, dtype=np.bool_),
            host_materialized=False,
        )

    d_validity = runtime.from_host(validity)
    d_tags = runtime.from_host(tags)
    d_family_row_offsets = runtime.from_host(family_row_offsets)

    host_validity = (
        None
        if hasattr(validity, "__cuda_array_interface__")
        else np.ascontiguousarray(validity, dtype=np.bool_)
    )
    host_tags = (
        None
        if hasattr(tags, "__cuda_array_interface__")
        else np.ascontiguousarray(tags, dtype=np.int8)
    )
    host_family_row_offsets = (
        None
        if hasattr(family_row_offsets, "__cuda_array_interface__")
        else np.ascontiguousarray(family_row_offsets, dtype=np.int32)
    )

    result = OwnedGeometryArray(
        validity=host_validity,
        tags=host_tags,
        family_row_offsets=host_family_row_offsets,
        families=host_families,
        residency=Residency.DEVICE,
        device_state=OwnedGeometryDeviceState(
            validity=d_validity,
            tags=d_tags,
            family_row_offsets=d_family_row_offsets,
            families=device_families,
        ),
        _row_count=int(row_count),
    )
    polygonal_families = {GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON}
    result.device_state.trusted_family_domain = tuple(device_families)
    if set(device_families) <= polygonal_families:
        result.device_state.trusted_polygonal_only = True
    if not device_families:
        result.device_state.trusted_unique_family_rows = True
    if len(device_families) == 1:
        family, device_buffer = next(iter(device_families.items()))
        if _device_family_row_count(device_buffer) == int(row_count):
            result.device_state.trusted_homogeneous_family = family
            result.device_state.trusted_unique_family_rows = True
            result.device_state.trusted_family_domain = (family,)
    result._record(
        DiagnosticKind.CREATED,
        f"device-resident owned array, {row_count} rows, {len(device_families)} families",
        visible=False,
    )
    return result


def build_empty_polygon_rows_device(
    row_count: int,
    *,
    validity: DeviceArray | None = None,
) -> OwnedGeometryArray:
    """Build device empty-polygon rows with device-resident validity."""
    if cp is None:  # pragma: no cover - exercised on CPU-only installs
        raise RuntimeError("CuPy is required for device empty polygon rows")

    row_count = int(row_count)
    polygon_tag = np.int8(FAMILY_TAGS[GeometryFamily.POLYGON])
    d_validity = (
        cp.ones(row_count, dtype=cp.bool_)
        if validity is None
        else cp.asarray(validity, dtype=cp.bool_)
    )
    if int(d_validity.size) != row_count:
        raise ValueError("empty polygon validity must match row_count")
    result = build_device_resident_owned(
        device_families={
            GeometryFamily.POLYGON: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.POLYGON,
                x=cp.empty(0, dtype=cp.float64),
                y=cp.empty(0, dtype=cp.float64),
                geometry_offsets=cp.zeros(row_count + 1, dtype=cp.int32),
                empty_mask=cp.ones(row_count, dtype=cp.bool_),
                ring_offsets=cp.zeros(1, dtype=cp.int32),
                bounds=None,
                fixed_size=DeviceFixedGeometrySizeMetadata(
                    first_level_count_per_row=0,
                    coord_count_per_row=0,
                ),
            )
        },
        row_count=row_count,
        tags=cp.where(d_validity, polygon_tag, np.int8(-1)),
        validity=d_validity,
        family_row_offsets=cp.where(
            d_validity,
            cp.arange(row_count, dtype=cp.int32),
            cp.int32(-1),
        ),
        execution_mode="gpu",
    )
    if validity is None:
        seed_homogeneous_host_metadata(result, GeometryFamily.POLYGON)
        seed_all_validity_cache(result)
    result._active_family_row_segment_capacity_bound = 0
    return result


def device_valid_nonempty_mask(owned: OwnedGeometryArray):
    """Return logical-row valid/nonempty metadata without compact row discovery."""
    if cp is None:  # pragma: no cover - exercised on CPU-only installs
        raise RuntimeError("CuPy is required for device geometry metadata masks")

    state = owned._ensure_device_state(preserve_indexed_view=True)
    row_count = int(owned.row_count)
    d_valid = cp.asarray(state.validity, dtype=cp.bool_)
    d_tags = cp.asarray(state.tags, dtype=cp.int8)
    d_family_rows = cp.asarray(state.family_row_offsets, dtype=cp.int64)
    d_nonempty = cp.zeros(row_count, dtype=cp.bool_)
    for family, buffer in state.families.items():
        if int(buffer.empty_mask.size) == 0:
            continue
        d_family = d_valid & (d_tags == np.int8(FAMILY_TAGS[family]))
        d_safe_rows = cp.where(d_family, d_family_rows, cp.int64(0))
        d_nonempty |= d_family & ~cp.asarray(buffer.empty_mask, dtype=cp.bool_)[d_safe_rows]
    return d_valid & d_nonempty


def forward_result_metadata(
    owned: OwnedGeometryArray,
) -> tuple[np.ndarray | DeviceArray, np.ndarray | DeviceArray, np.ndarray | DeviceArray]:
    """Forward metadata for a device result without forcing host copies.

    When the source already has device metadata, reuse those arrays directly so
    downstream device-resident builders do not pay a D->H->D round-trip.
    Otherwise, preserve the historical host-copy behavior for host-only inputs.
    """
    if owned.device_state is not None:
        return (
            owned.device_state.tags,
            owned.device_state.validity,
            owned.device_state.family_row_offsets,
        )

    return (
        owned.tags.copy(),
        owned.validity.copy(),
        owned.family_row_offsets.copy(),
    )


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
        raise ValueError(f"tile_single_row expects a 1-row array, got {owned.row_count}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if n == 1:
        return owned

    if owned.device_state is not None:
        if cp is None:  # pragma: no cover - device_state requires CuPy
            raise RuntimeError("CuPy is required for device broadcast tiling")
        d_index_map = cp.zeros(int(n), dtype=cp.int64)
        result = OwnedGeometryArray._indexed_view(owned, d_index_map)
        result._device_broadcast_implementation = "device_broadcast_row_indirection"
        result._record(
            DiagnosticKind.CREATED,
            f"device row-indirected broadcast of 1 row to {int(n)} rows",
            visible=False,
        )
        return result

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
        d_state = OwnedGeometryDeviceState(
            validity=d_validity,
            tags=d_tags,
            family_row_offsets=d_fro,
            families=dict(owned.device_state.families),  # shared reference
            trusted_all_finite_coordinates=(
                True
                if owned.device_state.trusted_all_finite_coordinates is True
                else None
            ),
        )
        result = OwnedGeometryArray(
            validity=validity,
            tags=tags,
            family_row_offsets=family_row_offsets,
            families=owned.families,  # shared reference (read-only usage)
            residency=Residency.DEVICE,
            device_state=d_state,
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


def _materialize_device_broadcast(tiled: OwnedGeometryArray) -> OwnedGeometryArray | None:
    """Physically repeat a device broadcast rowset without resolving through host."""
    if cp is None or getattr(tiled, "device_state", None) is None:
        return None
    if (
        getattr(tiled, "_device_broadcast_implementation", None)
        != "device_broadcast_row_indirection"
    ):
        return None
    base = getattr(tiled, "_base", None)
    if base is None or int(base.row_count) != 1:
        return None

    n = int(tiled.row_count)
    base_state = base._ensure_device_state(preserve_indexed_view=True)
    if not base_state.families:
        return build_null_owned_array(n, residency=Residency.DEVICE)
    if len(base_state.families) != 1:
        return None

    family, buf = next(iter(base_state.families.items()))
    new_empty_mask = cp.repeat(cp.asarray(buf.empty_mask, dtype=cp.bool_)[:1], n)
    new_bounds = (
        None
        if buf.bounds is None
        else cp.repeat(cp.asarray(buf.bounds, dtype=cp.float64)[:1], n, axis=0)
    )
    new_x = cp.tile(cp.asarray(buf.x, dtype=cp.float64), n)
    new_y = cp.tile(cp.asarray(buf.y, dtype=cp.float64), n)
    new_part_offsets = None
    new_ring_offsets = None
    fixed_size = _device_buffer_fixed_size_metadata(family, buf)
    dense_single_ring_width = buf.dense_single_ring_width

    if family in (GeometryFamily.POINT, GeometryFamily.LINESTRING, GeometryFamily.MULTIPOINT):
        coord_count = int(buf.x.size)
        new_geom_offsets = cp.arange(n + 1, dtype=cp.int32) * np.int32(coord_count)
    elif family is GeometryFamily.POLYGON:
        if buf.ring_offsets is None:
            return None
        ring_count = max(int(buf.ring_offsets.size) - 1, 0)
        if (
            dense_single_ring_width is None
            and ring_count == 1
            and int(buf.x.size) == int(buf.y.size)
            and int(buf.x.size) > 0
        ):
            dense_single_ring_width = int(buf.x.size)
            fixed_size = DeviceFixedGeometrySizeMetadata(
                first_level_count_per_row=1,
                coord_count_per_row=int(dense_single_ring_width),
            )
        new_geom_offsets = cp.arange(n + 1, dtype=cp.int32) * np.int32(ring_count)
        ring_lengths = (
            cp.asarray(buf.ring_offsets[1:] - buf.ring_offsets[:-1], dtype=cp.int32)
            if ring_count
            else cp.empty(0, dtype=cp.int32)
        )
        tiled_ring_lengths = cp.tile(ring_lengths, n)
        new_ring_offsets = cp.empty(int(tiled_ring_lengths.size) + 1, dtype=cp.int32)
        new_ring_offsets[0] = 0
        if int(tiled_ring_lengths.size):
            cp.cumsum(tiled_ring_lengths, out=new_ring_offsets[1:])
    elif family is GeometryFamily.MULTILINESTRING:
        if buf.part_offsets is None:
            return None
        part_count = max(int(buf.part_offsets.size) - 1, 0)
        new_geom_offsets = cp.arange(n + 1, dtype=cp.int32) * np.int32(part_count)
        part_lengths = (
            cp.asarray(buf.part_offsets[1:] - buf.part_offsets[:-1], dtype=cp.int32)
            if part_count
            else cp.empty(0, dtype=cp.int32)
        )
        tiled_part_lengths = cp.tile(part_lengths, n)
        new_part_offsets = cp.empty(int(tiled_part_lengths.size) + 1, dtype=cp.int32)
        new_part_offsets[0] = 0
        if int(tiled_part_lengths.size):
            cp.cumsum(tiled_part_lengths, out=new_part_offsets[1:])
    elif family is GeometryFamily.MULTIPOLYGON:
        if buf.part_offsets is None or buf.ring_offsets is None:
            return None
        polygon_count = max(int(buf.part_offsets.size) - 1, 0)
        ring_count = max(int(buf.ring_offsets.size) - 1, 0)
        new_geom_offsets = cp.arange(n + 1, dtype=cp.int32) * np.int32(polygon_count)
        polygon_ring_lengths = (
            cp.asarray(buf.part_offsets[1:] - buf.part_offsets[:-1], dtype=cp.int32)
            if polygon_count
            else cp.empty(0, dtype=cp.int32)
        )
        tiled_polygon_ring_lengths = cp.tile(polygon_ring_lengths, n)
        new_part_offsets = cp.empty(
            int(tiled_polygon_ring_lengths.size) + 1,
            dtype=cp.int32,
        )
        new_part_offsets[0] = 0
        if int(tiled_polygon_ring_lengths.size):
            cp.cumsum(tiled_polygon_ring_lengths, out=new_part_offsets[1:])
        ring_coord_lengths = (
            cp.asarray(buf.ring_offsets[1:] - buf.ring_offsets[:-1], dtype=cp.int32)
            if ring_count
            else cp.empty(0, dtype=cp.int32)
        )
        tiled_ring_coord_lengths = cp.tile(ring_coord_lengths, n)
        new_ring_offsets = cp.empty(
            int(tiled_ring_coord_lengths.size) + 1,
            dtype=cp.int32,
        )
        new_ring_offsets[0] = 0
        if int(tiled_ring_coord_lengths.size):
            cp.cumsum(tiled_ring_coord_lengths, out=new_ring_offsets[1:])
    else:
        return None

    device_families = {
        family: DeviceFamilyGeometryBuffer(
            family=family,
            x=new_x,
            y=new_y,
            geometry_offsets=new_geom_offsets,
            empty_mask=new_empty_mask,
            part_offsets=new_part_offsets,
            ring_offsets=new_ring_offsets,
            bounds=new_bounds,
            dense_single_ring_width=dense_single_ring_width,
            axis_aligned_rectangles=bool(buf.axis_aligned_rectangles),
            fixed_size=fixed_size,
        )
    }
    d_validity = cp.repeat(cp.asarray(base_state.validity, dtype=cp.bool_)[:1], n)
    d_tags = cp.repeat(cp.asarray(base_state.tags, dtype=cp.int8)[:1], n)
    d_family_row_offsets = cp.where(
        d_tags == np.int8(FAMILY_TAGS[family]),
        cp.arange(n, dtype=cp.int32),
        cp.int32(-1),
    )
    result = build_device_resident_owned(
        device_families=device_families,
        row_count=n,
        tags=d_tags,
        validity=d_validity,
        family_row_offsets=d_family_row_offsets,
        execution_mode="gpu",
    )
    result._device_broadcast_implementation = "device_broadcast_physicalized"
    result._record(
        DiagnosticKind.CREATED,
        f"device-physicalized broadcast of 1 row to {n} rows",
        visible=False,
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
    device_materialized = _materialize_device_broadcast(tiled)
    if device_materialized is not None:
        return device_materialized

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

        src_geom_start = int(buf.geometry_offsets[0])
        src_geom_end = int(buf.geometry_offsets[1])

        new_part_offsets = None
        new_ring_offsets = None

        if buf.part_offsets is None and buf.ring_offsets is None:
            # Point-like / coordinate-direct families: row -> coordinates.
            src_coord_start = src_geom_start
            src_coord_end = src_geom_end
            src_geometry_width = src_coord_end - src_coord_start
            new_geom_offsets = np.arange(family_n + 1, dtype=np.int32) * src_geometry_width
        elif buf.part_offsets is None and buf.ring_offsets is not None:
            # Polygon: row -> rings -> coordinates.
            src_ring_start = src_geom_start
            src_ring_end = src_geom_end
            src_coord_start = int(buf.ring_offsets[src_ring_start])
            src_coord_end = int(buf.ring_offsets[src_ring_end])
            src_ring_lens = np.diff(buf.ring_offsets[src_ring_start : src_ring_end + 1])
            tiled_ring_lens = np.tile(src_ring_lens, family_n)
            new_ring_offsets = np.empty(len(tiled_ring_lens) + 1, dtype=np.int32)
            new_ring_offsets[0] = 0
            np.cumsum(tiled_ring_lens, out=new_ring_offsets[1:])
            src_geometry_width = src_ring_end - src_ring_start
            new_geom_offsets = np.arange(family_n + 1, dtype=np.int32) * src_geometry_width
        elif buf.part_offsets is not None and buf.ring_offsets is None:
            # MultiLineString: row -> parts -> coordinates.
            src_part_start = src_geom_start
            src_part_end = src_geom_end
            src_coord_start = int(buf.part_offsets[src_part_start])
            src_coord_end = int(buf.part_offsets[src_part_end])
            src_part_lens = np.diff(buf.part_offsets[src_part_start : src_part_end + 1])
            tiled_part_lens = np.tile(src_part_lens, family_n)
            new_part_offsets = np.empty(len(tiled_part_lens) + 1, dtype=np.int32)
            new_part_offsets[0] = 0
            np.cumsum(tiled_part_lens, out=new_part_offsets[1:])
            src_geometry_width = src_part_end - src_part_start
            new_geom_offsets = np.arange(family_n + 1, dtype=np.int32) * src_geometry_width
        else:
            # MultiPolygon: row -> polygons -> rings -> coordinates.
            src_polygon_start = src_geom_start
            src_polygon_end = src_geom_end
            src_ring_start = int(buf.part_offsets[src_polygon_start])
            src_ring_end = int(buf.part_offsets[src_polygon_end])
            src_coord_start = int(buf.ring_offsets[src_ring_start])
            src_coord_end = int(buf.ring_offsets[src_ring_end])

            src_polygon_ring_lens = np.diff(
                buf.part_offsets[src_polygon_start : src_polygon_end + 1]
            )
            tiled_polygon_ring_lens = np.tile(src_polygon_ring_lens, family_n)
            new_part_offsets = np.empty(len(tiled_polygon_ring_lens) + 1, dtype=np.int32)
            new_part_offsets[0] = 0
            np.cumsum(tiled_polygon_ring_lens, out=new_part_offsets[1:])

            src_ring_coord_lens = np.diff(buf.ring_offsets[src_ring_start : src_ring_end + 1])
            tiled_ring_coord_lens = np.tile(src_ring_coord_lens, family_n)
            new_ring_offsets = np.empty(len(tiled_ring_coord_lens) + 1, dtype=np.int32)
            new_ring_offsets[0] = 0
            np.cumsum(tiled_ring_coord_lens, out=new_ring_offsets[1:])

            src_geometry_width = src_polygon_end - src_polygon_start
            new_geom_offsets = np.arange(family_n + 1, dtype=np.int32) * src_geometry_width

        # Replicate coordinates for the single source row.
        src_x = buf.x[src_coord_start:src_coord_end]
        src_y = buf.y[src_coord_start:src_coord_end]
        new_x = np.tile(src_x, family_n)
        new_y = np.tile(src_y, family_n)

        # Replicate empty_mask.
        new_empty_mask = np.repeat(buf.empty_mask[:1], family_n)

        bounds = None
        if buf.bounds is not None:
            bounds = np.repeat(buf.bounds[:1], family_n, axis=0)

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
            bounds=bounds,
        )

    # Build new family_row_offsets that map to 0..family_n-1 (not all 0).
    # Vectorized per-tag cumulative count (VPAT001: no per-element Python loops).
    new_fro = np.full(n, -1, dtype=np.int32)
    for tag_val in np.unique(tiled.tags):
        if tag_val == NULL_TAG:
            continue
        mask = tiled.tags == tag_val
        new_fro[mask] = np.arange(int(mask.sum()), dtype=np.int32)

    result = OwnedGeometryArray(
        validity=tiled.validity.copy(),
        tags=tiled.tags.copy(),
        family_row_offsets=new_fro,
        families=new_families,
        residency=Residency.HOST,
    )
    if tiled.device_state is not None:
        result.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="materialized broadcast-right geometry on device",
        )
    return result
