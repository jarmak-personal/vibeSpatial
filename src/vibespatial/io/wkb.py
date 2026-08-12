from __future__ import annotations

import io
import os
import struct
from dataclasses import dataclass, replace
from importlib.util import find_spec
from types import SimpleNamespace
from typing import Any

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    get_cuda_completion_retainer,
    get_cuda_runtime,
    make_kernel_cache_key,
    pylibcudf_column_from_arrow,
    pylibcudf_column_from_device,
    pylibcudf_to_arrow,
)
from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.cccl_primitives import exclusive_sum
from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema
from vibespatial.geometry.device_array import DeviceGeometryArray
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    TAG_FAMILIES,
    DeviceFamilyGeometryBuffer,
    DeviceFixedGeometrySizeMetadata,
    DiagnosticKind,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
    build_device_resident_owned,
    build_null_owned_array,
    device_family_coordinate_counts,
    from_wkb,
)
from vibespatial.io.wkb_kernels import (
    _WKB_ENCODE_KERNEL_NAMES,
    _WKB_ENCODE_KERNEL_SOURCE,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.fallbacks import (
    record_fallback_event,
    strict_native_mode_enabled,
)
from vibespatial.runtime.materialization import NativeExportBoundary, record_native_export_boundary
from vibespatial.runtime.residency import Residency, TransferTrigger

from .support import IOFormat, IOOperation, IOPathKind, plan_io_support
from .wkb_cpu import iter_geometry_parts

request_warmup(["exclusive_scan_i32", "exclusive_scan_i64"])

from vibespatial.cuda.nvrtc_precompile import (  # noqa: E402
    request_nvrtc_warmup as _request_nvrtc_warmup,
)

WKB_TYPE_IDS: dict[GeometryFamily, int] = {
    GeometryFamily.POINT: 1,
    GeometryFamily.LINESTRING: 2,
    GeometryFamily.POLYGON: 3,
    GeometryFamily.MULTIPOINT: 4,
    GeometryFamily.MULTILINESTRING: 5,
    GeometryFamily.MULTIPOLYGON: 6,
}
WKB_ID_FAMILIES = {value: key for key, value in WKB_TYPE_IDS.items()}
WKB_POINT_RECORD_DTYPE = np.dtype(
    {
        "names": ["byteorder", "type", "x", "y"],
        "formats": ["u1", "<u4", "<f8", "<f8"],
        "offsets": [0, 1, 5, 13],
        "itemsize": 21,
    }
)
DEVICE_WKB_LIST_DECODE_MIN_ROWS = 8_000

_GEOARROW_ENCODING_FAMILIES: dict[str, GeometryFamily] = {
    "point": GeometryFamily.POINT,
    "linestring": GeometryFamily.LINESTRING,
    "polygon": GeometryFamily.POLYGON,
    "multipoint": GeometryFamily.MULTIPOINT,
    "multilinestring": GeometryFamily.MULTILINESTRING,
    "multipolygon": GeometryFamily.MULTIPOLYGON,
}

_SUPPORTED_DEVICE_GEOARROW_PROMOTIONS = {
    frozenset({GeometryFamily.POINT, GeometryFamily.MULTIPOINT}): GeometryFamily.MULTIPOINT,
    frozenset(
        {GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING}
    ): GeometryFamily.MULTILINESTRING,
    frozenset({GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON}): GeometryFamily.MULTIPOLYGON,
}


_request_nvrtc_warmup(
    [
        ("wkb-encode", _WKB_ENCODE_KERNEL_SOURCE, _WKB_ENCODE_KERNEL_NAMES),
    ]
)


@dataclass(frozen=True)
class _GpuWkbDecodeAttempt:
    result: OwnedGeometryArray | None
    fallback_detail: str | None = None


@dataclass(frozen=True)
class _NativeDeviceWriteStatus:
    written: bool
    fallback_detail: str | None = None
    compatibility_detail: str | None = None


@dataclass(frozen=True)
class _NativeDeviceIndexColumn:
    field_name: str
    logical_name: Any
    column: Any
    field: Any
    metadata_index: Any


def _pylibcudf_sink(path) -> str | io.IOBase | None:
    if isinstance(path, io.IOBase):
        return path
    if isinstance(path, bytes):
        return os.fsdecode(path)
    if isinstance(path, (str, os.PathLike)):
        return str(path)
    return None


class _GpuWkbOnInvalidError(ValueError):
    """Raised when the GPU WKB decode path must honor on_invalid='raise'."""


def has_pyarrow_support() -> bool:
    return find_spec("pyarrow") is not None


def has_pylibcudf_support() -> bool:
    return find_spec("pylibcudf") is not None


def _authoritative_host_metadata(
    owned: OwnedGeometryArray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return host metadata sourced from device state when available."""
    if (
        getattr(owned, "_validity", None) is not None
        and getattr(owned, "_tags", None) is not None
        and getattr(owned, "_family_row_offsets", None) is not None
    ):
        return owned._validity, owned._tags, owned._family_row_offsets
    if owned.device_state is not None:
        runtime = get_cuda_runtime()
        return (
            runtime.copy_device_to_host(
                owned.device_state.validity,
                reason="wkb encode validity metadata host boundary",
            ),
            runtime.copy_device_to_host(
                owned.device_state.tags,
                reason="wkb encode family-tags metadata host boundary",
            ),
            runtime.copy_device_to_host(
                owned.device_state.family_row_offsets,
                reason="wkb encode family-row-offset metadata host boundary",
            ),
        )
    return owned.validity, owned.tags, owned.family_row_offsets


def _wkb_encode_kernels():
    runtime = get_cuda_runtime()
    return runtime.compile_kernels(
        cache_key=make_kernel_cache_key("wkb-encode", _WKB_ENCODE_KERNEL_SOURCE),
        source=_WKB_ENCODE_KERNEL_SOURCE,
        kernel_names=_WKB_ENCODE_KERNEL_NAMES,
    )


def _device_family_row_selection(
    owned: OwnedGeometryArray,
    family: GeometryFamily,
) -> tuple[Any, Any]:
    import cupy as cp

    state = _terminal_device_state(owned)
    family_mask = (cp.asarray(state.tags) == np.int8(FAMILY_TAGS[family])) & cp.asarray(
        state.validity
    )
    row_indexes = cp.flatnonzero(family_mask).astype(cp.int32, copy=False)
    family_rows = cp.asarray(state.family_row_offsets)[
        row_indexes.astype(cp.int64, copy=False)
    ].astype(cp.int32, copy=False)
    return row_indexes, family_rows


def _terminal_device_state(owned: OwnedGeometryArray):
    return owned._ensure_device_state(preserve_indexed_view=True)


def _device_wkb_lengths_for_family(
    owned: OwnedGeometryArray,
    family: GeometryFamily,
    family_rows,
):
    import cupy as cp

    state = _terminal_device_state(owned)
    device_buffer = state.families[family]
    if family_rows.size == 0:
        return cp.zeros(0, dtype=cp.int32)

    geometry_offsets = device_buffer.geometry_offsets

    if family is GeometryFamily.POINT:
        lengths = cp.full(family_rows.size, 21, dtype=cp.int32)
    elif family is GeometryFamily.LINESTRING:
        counts = geometry_offsets[family_rows + 1] - geometry_offsets[family_rows]
        lengths = (9 + 16 * counts).astype(cp.int32, copy=False)
    elif family is GeometryFamily.POLYGON:
        ring_offsets = device_buffer.ring_offsets
        ring_start = geometry_offsets[family_rows]
        ring_stop = geometry_offsets[family_rows + 1]
        coord_start = ring_offsets[ring_start]
        coord_stop = ring_offsets[ring_stop]
        lengths = (9 + 4 * (ring_stop - ring_start) + 16 * (coord_stop - coord_start)).astype(
            cp.int32, copy=False
        )
    elif family is GeometryFamily.MULTIPOINT:
        counts = geometry_offsets[family_rows + 1] - geometry_offsets[family_rows]
        lengths = (9 + 21 * counts).astype(cp.int32, copy=False)
    elif family is GeometryFamily.MULTILINESTRING:
        part_offsets = device_buffer.part_offsets
        part_start = geometry_offsets[family_rows]
        part_stop = geometry_offsets[family_rows + 1]
        coord_start = part_offsets[part_start]
        coord_stop = part_offsets[part_stop]
        lengths = (9 + 9 * (part_stop - part_start) + 16 * (coord_stop - coord_start)).astype(
            cp.int32, copy=False
        )
    elif family is GeometryFamily.MULTIPOLYGON:
        part_offsets = device_buffer.part_offsets
        ring_offsets = device_buffer.ring_offsets
        poly_start = geometry_offsets[family_rows]
        poly_stop = geometry_offsets[family_rows + 1]
        ring_start = part_offsets[poly_start]
        ring_stop = part_offsets[poly_stop]
        coord_start = ring_offsets[ring_start]
        coord_stop = ring_offsets[ring_stop]
        lengths = (
            9
            + 9 * (poly_stop - poly_start)
            + 4 * (ring_stop - ring_start)
            + 16 * (coord_stop - coord_start)
        ).astype(cp.int32, copy=False)
    else:  # pragma: no cover - exhaustive today
        raise ValueError(f"Unsupported geometry family for device WKB encode: {family}")
    return lengths


def _launch_device_wkb_write_kernel(
    family: GeometryFamily,
    *,
    owned: OwnedGeometryArray,
    row_indexes,
    family_rows,
    row_offsets,
    payload,
) -> None:
    import cupy as cp

    count = int(row_indexes.size)
    if count == 0:
        return

    runtime = get_cuda_runtime()
    kernels = _wkb_encode_kernels()
    state = _terminal_device_state(owned)
    device_buffer = state.families[family]
    row_indexes = cp.asarray(row_indexes, dtype=cp.int32)
    family_rows = cp.asarray(family_rows, dtype=cp.int32)
    ptr = runtime.pointer
    if family is GeometryFamily.POINT:
        kernel = kernels["write_point_wkb"]
        params = (
            (
                ptr(row_indexes),
                ptr(family_rows),
                ptr(device_buffer.geometry_offsets),
                ptr(device_buffer.x),
                ptr(device_buffer.y),
                ptr(row_offsets),
                ptr(payload),
                count,
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
    elif family is GeometryFamily.LINESTRING:
        kernel = kernels["write_linestring_wkb"]
        params = (
            (
                ptr(row_indexes),
                ptr(family_rows),
                ptr(device_buffer.geometry_offsets),
                ptr(device_buffer.x),
                ptr(device_buffer.y),
                ptr(row_offsets),
                ptr(payload),
                count,
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
    elif family is GeometryFamily.POLYGON:
        kernel = kernels["write_polygon_wkb"]
        params = (
            (
                ptr(row_indexes),
                ptr(family_rows),
                ptr(device_buffer.geometry_offsets),
                ptr(device_buffer.ring_offsets),
                ptr(device_buffer.x),
                ptr(device_buffer.y),
                ptr(row_offsets),
                ptr(payload),
                count,
            ),
            (
                KERNEL_PARAM_PTR,
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
    elif family is GeometryFamily.MULTIPOINT:
        kernel = kernels["write_multipoint_wkb"]
        params = (
            (
                ptr(row_indexes),
                ptr(family_rows),
                ptr(device_buffer.geometry_offsets),
                ptr(device_buffer.x),
                ptr(device_buffer.y),
                ptr(row_offsets),
                ptr(payload),
                count,
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
    elif family is GeometryFamily.MULTILINESTRING:
        kernel = kernels["write_multilinestring_wkb"]
        params = (
            (
                ptr(row_indexes),
                ptr(family_rows),
                ptr(device_buffer.geometry_offsets),
                ptr(device_buffer.part_offsets),
                ptr(device_buffer.x),
                ptr(device_buffer.y),
                ptr(row_offsets),
                ptr(payload),
                count,
            ),
            (
                KERNEL_PARAM_PTR,
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
    elif family is GeometryFamily.MULTIPOLYGON:
        kernel = kernels["write_multipolygon_wkb"]
        params = (
            (
                ptr(row_indexes),
                ptr(family_rows),
                ptr(device_buffer.geometry_offsets),
                ptr(device_buffer.part_offsets),
                ptr(device_buffer.ring_offsets),
                ptr(device_buffer.x),
                ptr(device_buffer.y),
                ptr(row_offsets),
                ptr(payload),
                count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
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
    else:  # pragma: no cover - exhaustive today
        raise ValueError(f"Unsupported geometry family for device WKB encode: {family}")

    grid, block = runtime.launch_config(kernel, count)
    runtime.launch(kernel, grid=grid, block=block, params=params)


def _wkb_upper_bound_bytes(
    state,
    family_selections: dict,
) -> int:
    """Compute a host-side upper bound for total WKB output bytes.

    Uses only ``.shape[0]`` on device arrays (no sync), so this never
    triggers a D->H transfer.  The bound is tight -- it over-estimates
    only by the contribution of invalid (unselected) rows in each family
    buffer, which is typically zero.
    """
    total = 0
    for family, (row_indexes, _family_rows) in family_selections.items():
        n_rows_family = row_indexes.shape[0]
        buf = state.families[family]
        fixed_size = getattr(buf, "fixed_size", None)
        max_coords = (
            None
            if fixed_size is None
            else fixed_size.max_coord_count_per_row
        )
        max_first = (
            None
            if fixed_size is None
            else fixed_size.max_first_level_count_per_row
        )
        max_second = (
            None
            if fixed_size is None
            else fixed_size.max_second_level_count_per_row
        )
        n_coords = (
            buf.x.shape[0]
            if max_coords is None
            else int(max_coords) * n_rows_family
        )

        if family is GeometryFamily.POINT:
            total += 21 * n_rows_family
        elif family is GeometryFamily.LINESTRING:
            total += 9 * n_rows_family + 16 * n_coords
        elif family is GeometryFamily.POLYGON:
            n_rings = (
                buf.ring_offsets.shape[0] - 1
                if max_first is None and buf.ring_offsets is not None
                else int(max_first or 0) * n_rows_family
            )
            total += 9 * n_rows_family + 4 * n_rings + 16 * n_coords
        elif family is GeometryFamily.MULTIPOINT:
            total += 9 * n_rows_family + 21 * n_coords
        elif family is GeometryFamily.MULTILINESTRING:
            n_parts = (
                buf.part_offsets.shape[0] - 1
                if max_first is None and buf.part_offsets is not None
                else int(max_first or 0) * n_rows_family
            )
            total += 9 * n_rows_family + 9 * n_parts + 16 * n_coords
        elif family is GeometryFamily.MULTIPOLYGON:
            n_parts = (
                buf.part_offsets.shape[0] - 1
                if max_first is None and buf.part_offsets is not None
                else int(max_first or 0) * n_rows_family
            )
            n_rings = (
                buf.ring_offsets.shape[0] - 1
                if max_second is None and buf.ring_offsets is not None
                else int(max_second or 0) * n_rows_family
            )
            total += 9 * n_rows_family + 9 * n_parts + 4 * n_rings + 16 * n_coords
    return total


def _encode_owned_wkb_column_device(owned: OwnedGeometryArray):
    import cupy as cp
    import pylibcudf as plc

    state = _terminal_device_state(owned)
    row_count = owned.row_count
    lengths = cp.zeros(row_count, dtype=cp.int32)
    family_selections: dict[GeometryFamily, tuple[Any, Any]] = {}
    valid_row_count = 0

    for family in state.families:
        row_indexes, family_rows = _device_family_row_selection(owned, family)
        family_lengths = _device_wkb_lengths_for_family(owned, family, family_rows)
        if row_indexes.size:
            lengths[row_indexes.astype(cp.int64, copy=False)] = family_lengths
            family_selections[family] = (row_indexes, family_rows)
            valid_row_count += int(row_indexes.size)

    offsets = cp.empty(row_count + 1, dtype=cp.int32)
    if row_count:
        offsets[:-1] = exclusive_sum(lengths)
        offsets[-1] = cp.sum(lengths, dtype=cp.int32)
    else:
        offsets[...] = 0
    # Upper-bound allocation: compute total from host-side buffer shapes
    # (no device sync). Slightly over-estimates if invalid rows contribute
    # coordinates to family buffers but are excluded from encoding.
    total_bytes = _wkb_upper_bound_bytes(state, family_selections) if row_count else 0
    payload = cp.empty(total_bytes, dtype=cp.uint8)

    for family, (row_indexes, family_rows) in family_selections.items():
        _launch_device_wkb_write_kernel(
            family,
            owned=owned,
            row_indexes=row_indexes,
            family_rows=family_rows,
            row_offsets=offsets,
            payload=payload,
        )

    runtime = get_cuda_runtime()
    runtime.synchronize()

    from vibespatial.cuda._runtime import pylibcudf_column_from_device

    offsets_column = pylibcudf_column_from_device(offsets)
    column = plc.Column(
        plc.types.DataType(plc.types.TypeId.STRING),
        row_count,
        plc.gpumemoryview(payload),
        None,
        0,
        0,
        [offsets_column],
    )

    validity_mask, null_count = _device_validity_gpumask(
        owned,
        valid_row_count=valid_row_count,
    )
    if null_count:
        column = column.with_mask(validity_mask, null_count)
    return column


def _geoarrow_family_from_encoding(encoding_name: str) -> GeometryFamily:
    try:
        return _GEOARROW_ENCODING_FAMILIES[encoding_name.lower()]
    except KeyError as exc:  # pragma: no cover - defensive schema guard
        raise ValueError(f"Unsupported GeoArrow encoding: {encoding_name!r}") from exc


def _device_geoarrow_family_set(owned: OwnedGeometryArray) -> frozenset[GeometryFamily]:
    if getattr(owned, "is_indexed_view", False):
        state = owned.device_state
        if state is not None:
            if state.trusted_homogeneous_family is not None:
                return frozenset({state.trusted_homogeneous_family})
            if len(state.families) == 1:
                return frozenset(state.families)
        validity = getattr(owned, "_validity", None)
        tags = getattr(owned, "_tags", None)
        if validity is not None and tags is not None:
            valid_tags = np.asarray(tags[np.asarray(validity, dtype=bool)], dtype=np.int8)
            if valid_tags.size == 0:
                return frozenset()
            return frozenset(TAG_FAMILIES[int(tag)] for tag in np.unique(valid_tags))
        if state is not None:
            return frozenset(state.families)
        return frozenset()
    state = owned.device_state
    if state is not None:
        return frozenset(
            family for family in state.families if _device_family_row_count(owned, family) > 0
        )
    validity, tags, _family_row_offsets = _authoritative_host_metadata(owned)
    valid_tags = np.asarray(tags[validity], dtype=np.int8)
    if valid_tags.size == 0:
        return frozenset()
    return frozenset(TAG_FAMILIES[int(tag)] for tag in np.unique(valid_tags))


def _device_geoarrow_export_family(
    owned: OwnedGeometryArray,
) -> tuple[GeometryFamily, bool]:
    family_set = _device_geoarrow_family_set(owned)
    if not family_set:
        raise ValueError("Cannot encode an all-null geometry array to native GeoArrow")
    if len(family_set) == 1:
        return next(iter(family_set)), False
    promoted = _SUPPORTED_DEVICE_GEOARROW_PROMOTIONS.get(family_set)
    if promoted is not None:
        return promoted, True
    raise ValueError("Native GeoArrow fast path requires a homogeneous geometry family")


def _device_geoarrow_fast_path_reason_owned(owned: OwnedGeometryArray) -> str | None:
    if owned.row_count == 0:
        return "empty geometry column requires upstream GeoArrow constructor semantics"
    try:
        _device_geoarrow_export_family(owned)
    except ValueError as exc:
        if str(exc) == "Cannot encode an all-null geometry array to native GeoArrow":
            return "all-missing geometry column requires upstream GeoArrow constructor semantics"
        return str(exc)
    return None


def _device_full_offsets_from_local(
    owned: OwnedGeometryArray,
    family: GeometryFamily,
    local_offsets,
    *,
    empty_mask=None,
):
    import cupy as cp

    state = _terminal_device_state(owned)
    validity = cp.asarray(state.validity)
    tags = cp.asarray(state.tags)
    family_rows = cp.asarray(state.family_row_offsets, dtype=cp.int64)
    family_valid = validity & (tags == np.int8(FAMILY_TAGS[family]))
    row_count = owned.row_count
    counts = cp.zeros(row_count, dtype=cp.int32)
    if row_count:
        local_counts = (local_offsets[1:] - local_offsets[:-1]).astype(cp.int32, copy=False)
        selected_family_rows = family_rows[family_valid]
        selected_counts = local_counts[selected_family_rows]
        if empty_mask is not None:
            selected_counts = cp.where(
                cp.asarray(empty_mask)[selected_family_rows],
                0,
                selected_counts,
            )
        counts[family_valid] = selected_counts
    full_offsets = cp.empty(row_count + 1, dtype=cp.int32)
    full_offsets[0] = 0
    if row_count:
        full_offsets[1:] = cp.cumsum(counts, dtype=cp.int32)
    return full_offsets


def _device_scalar_to_host_int(device_value, *, reason: str) -> int:
    import cupy as cp

    runtime = get_cuda_runtime()
    d_value = cp.asarray(device_value, dtype=cp.int64).reshape(1)
    host = runtime.copy_device_to_host(d_value, reason=reason)
    return int(np.asarray(host, dtype=np.int64).reshape(1)[0])


def _device_family_row_count(owned: OwnedGeometryArray, family: GeometryFamily) -> int:
    if getattr(owned, "is_indexed_view", False):
        state = owned.device_state
        if (
            state is not None
            and state.trusted_homogeneous_family is family
            and state.trusted_all_valid is True
        ):
            return int(owned.row_count)
        if state is not None and len(state.families) == 1 and family in state.families:
            return int(owned.row_count)
        validity = getattr(owned, "_validity", None)
        tags = getattr(owned, "_tags", None)
        if validity is not None and tags is not None:
            return int(
                np.count_nonzero(
                    np.asarray(validity, dtype=bool)
                    & (np.asarray(tags, dtype=np.int8) == np.int8(FAMILY_TAGS[family]))
                )
            )
        return 0
    state = owned.device_state
    if state is not None and family in state.families:
        offsets = state.families[family].geometry_offsets
        return max(int(offsets.size) - 1, 0)
    if family in owned.families:
        return max(int(owned.families[family].row_count), 0)
    return 0


def _device_structure_valid_row_count(owned: OwnedGeometryArray) -> int | None:
    state = owned.device_state
    if state is None:
        return None
    if state.trusted_all_valid is True:
        return int(owned.row_count)
    if len(state.families) == 1:
        device_buffer = next(iter(state.families.values()))
        if int(getattr(device_buffer.geometry_offsets, "size", 0)) == int(owned.row_count) + 1:
            return int(owned.row_count)
    if getattr(owned, "is_indexed_view", False):
        validity = getattr(owned, "_validity", None)
        if validity is not None:
            return int(np.count_nonzero(np.asarray(validity, dtype=bool)))
        return None
    return sum(_device_family_row_count(owned, family) for family in state.families)


def _device_scatter_xy_offset_slices(
    *,
    source_x,
    source_y,
    source_offsets,
    source_rows,
    target_rows,
    target_offsets,
    out_x,
    out_y,
    total_coords: int,
) -> None:
    import cupy as cp

    if total_coords == 0 or int(source_rows.size) == 0:
        return
    counts = (source_offsets[source_rows + 1] - source_offsets[source_rows]).astype(
        cp.int32, copy=False
    )
    local_offsets = cp.empty(int(source_rows.size) + 1, dtype=cp.int32)
    local_offsets[0] = 0
    local_offsets[1:] = cp.cumsum(counts, dtype=cp.int32)
    positions = cp.arange(total_coords, dtype=cp.int32)
    groups = cp.searchsorted(local_offsets[1:], positions, side="right").astype(
        cp.int64,
        copy=False,
    )
    within = positions - local_offsets[groups]
    source_indices = source_offsets[source_rows[groups]] + within
    dest_indices = target_offsets[target_rows[groups]] + within
    out_x[dest_indices] = source_x[source_indices]
    out_y[dest_indices] = source_y[source_indices]


def _promoted_geoarrow_metadata(
    owned: OwnedGeometryArray,
    export_family: GeometryFamily,
):
    import cupy as cp

    state = _terminal_device_state(owned)
    valid_rows = cp.flatnonzero(state.validity).astype(cp.int64, copy=False)
    valid_count = _device_structure_valid_row_count(owned)
    if valid_count is None:
        valid_count = int(valid_rows.size)
    tags = cp.full(owned.row_count, -1, dtype=cp.int8)
    family_rows = cp.full(owned.row_count, -1, dtype=cp.int32)
    tags[valid_rows] = np.int8(FAMILY_TAGS[export_family])
    family_rows[valid_rows] = cp.arange(valid_count, dtype=cp.int32)
    return state, valid_count, tags, family_rows


def _promote_point_multipoint_geoarrow_owned_device(
    owned: OwnedGeometryArray,
) -> OwnedGeometryArray:
    import cupy as cp

    state, valid_count, promoted_tags, promoted_family_rows = _promoted_geoarrow_metadata(
        owned,
        GeometryFamily.MULTIPOINT,
    )
    point_buffer = state.families.get(GeometryFamily.POINT)
    multipoint_buffer = state.families.get(GeometryFamily.MULTIPOINT)
    if point_buffer is None or multipoint_buffer is None:
        raise ValueError("Point/MultiPoint GeoArrow promotion requires both families")

    counts = cp.zeros(valid_count, dtype=cp.int32)
    point_rows_global = cp.flatnonzero(
        state.validity & (state.tags == np.int8(FAMILY_TAGS[GeometryFamily.POINT]))
    ).astype(cp.int64, copy=False)
    point_rows = state.family_row_offsets[point_rows_global].astype(cp.int64, copy=False)
    point_target_rows = promoted_family_rows[point_rows_global].astype(cp.int64, copy=False)
    point_counts = (
        point_buffer.geometry_offsets[point_rows + 1] - point_buffer.geometry_offsets[point_rows]
    ).astype(cp.int32, copy=False)
    if int(point_target_rows.size):
        counts[point_target_rows] = point_counts

    multipoint_rows_global = cp.flatnonzero(
        state.validity & (state.tags == np.int8(FAMILY_TAGS[GeometryFamily.MULTIPOINT]))
    ).astype(cp.int64, copy=False)
    multipoint_rows = state.family_row_offsets[multipoint_rows_global].astype(cp.int64, copy=False)
    multipoint_target_rows = promoted_family_rows[multipoint_rows_global].astype(
        cp.int64, copy=False
    )
    multipoint_counts = (
        multipoint_buffer.geometry_offsets[multipoint_rows + 1]
        - multipoint_buffer.geometry_offsets[multipoint_rows]
    ).astype(cp.int32, copy=False)
    if int(multipoint_target_rows.size):
        counts[multipoint_target_rows] = multipoint_counts

    geometry_offsets = cp.empty(valid_count + 1, dtype=cp.int32)
    geometry_offsets[0] = 0
    if valid_count:
        geometry_offsets[1:] = cp.cumsum(counts, dtype=cp.int32)
    total_coords = int(point_buffer.x.size) + int(multipoint_buffer.x.size)
    x = cp.empty(total_coords, dtype=cp.float64)
    y = cp.empty(total_coords, dtype=cp.float64)
    _device_scatter_xy_offset_slices(
        source_x=point_buffer.x,
        source_y=point_buffer.y,
        source_offsets=point_buffer.geometry_offsets,
        source_rows=point_rows,
        target_rows=point_target_rows,
        target_offsets=geometry_offsets,
        out_x=x,
        out_y=y,
        total_coords=int(point_buffer.x.size),
    )
    _device_scatter_xy_offset_slices(
        source_x=multipoint_buffer.x,
        source_y=multipoint_buffer.y,
        source_offsets=multipoint_buffer.geometry_offsets,
        source_rows=multipoint_rows,
        target_rows=multipoint_target_rows,
        target_offsets=geometry_offsets,
        out_x=x,
        out_y=y,
        total_coords=int(multipoint_buffer.x.size),
    )
    promoted_buffer = DeviceFamilyGeometryBuffer(
        family=GeometryFamily.MULTIPOINT,
        x=x,
        y=y,
        geometry_offsets=geometry_offsets,
        empty_mask=counts == 0,
    )
    result = build_device_resident_owned(
        device_families={GeometryFamily.MULTIPOINT: promoted_buffer},
        row_count=owned.row_count,
        tags=promoted_tags,
        validity=state.validity,
        family_row_offsets=promoted_family_rows,
        execution_mode="gpu",
    )
    result._record(
        DiagnosticKind.CREATED,
        "device-side GeoArrow Point/MultiPoint promotion to MultiPoint",
        visible=False,
    )
    return result


def _promote_linestring_multilinestring_geoarrow_owned_device(
    owned: OwnedGeometryArray,
) -> OwnedGeometryArray:
    import cupy as cp

    state, valid_count, promoted_tags, promoted_family_rows = _promoted_geoarrow_metadata(
        owned,
        GeometryFamily.MULTILINESTRING,
    )
    linestring_buffer = state.families.get(GeometryFamily.LINESTRING)
    multilinestring_buffer = state.families.get(GeometryFamily.MULTILINESTRING)
    if linestring_buffer is None or multilinestring_buffer is None:
        raise ValueError("LineString/MultiLineString GeoArrow promotion requires both families")

    linestring_rows_global = cp.flatnonzero(
        state.validity & (state.tags == np.int8(FAMILY_TAGS[GeometryFamily.LINESTRING]))
    ).astype(cp.int64, copy=False)
    linestring_rows = state.family_row_offsets[linestring_rows_global].astype(cp.int64, copy=False)
    linestring_target_rows = promoted_family_rows[linestring_rows_global].astype(
        cp.int64, copy=False
    )
    linestring_part_counts = cp.ones(int(linestring_rows.size), dtype=cp.int32)

    multilinestring_rows_global = cp.flatnonzero(
        state.validity & (state.tags == np.int8(FAMILY_TAGS[GeometryFamily.MULTILINESTRING]))
    ).astype(cp.int64, copy=False)
    multilinestring_rows = state.family_row_offsets[multilinestring_rows_global].astype(
        cp.int64, copy=False
    )
    multilinestring_target_rows = promoted_family_rows[multilinestring_rows_global].astype(
        cp.int64, copy=False
    )
    multilinestring_part_counts = (
        multilinestring_buffer.geometry_offsets[multilinestring_rows + 1]
        - multilinestring_buffer.geometry_offsets[multilinestring_rows]
    ).astype(cp.int32, copy=False)

    geometry_counts = cp.zeros(valid_count, dtype=cp.int32)
    if int(linestring_target_rows.size):
        geometry_counts[linestring_target_rows] = linestring_part_counts
    if int(multilinestring_target_rows.size):
        geometry_counts[multilinestring_target_rows] = multilinestring_part_counts
    geometry_offsets = cp.empty(valid_count + 1, dtype=cp.int32)
    geometry_offsets[0] = 0
    if valid_count:
        geometry_offsets[1:] = cp.cumsum(geometry_counts, dtype=cp.int32)

    total_line_parts = int(linestring_rows.size)
    total_mls_parts = max(int(multilinestring_buffer.part_offsets.size) - 1, 0)
    total_parts = total_line_parts + total_mls_parts
    part_lengths = cp.zeros(total_parts, dtype=cp.int32)

    line_part_positions = geometry_offsets[linestring_target_rows]
    line_lengths = (
        linestring_buffer.geometry_offsets[linestring_rows + 1]
        - linestring_buffer.geometry_offsets[linestring_rows]
    ).astype(cp.int32, copy=False)
    if int(line_part_positions.size):
        part_lengths[line_part_positions] = line_lengths

    if total_mls_parts:
        source_part_indexes = cp.arange(total_mls_parts, dtype=cp.int64)
        row_ends = multilinestring_buffer.geometry_offsets[multilinestring_rows + 1]
        groups = cp.searchsorted(row_ends, source_part_indexes, side="right").astype(
            cp.int64, copy=False
        )
        row_starts = multilinestring_buffer.geometry_offsets[multilinestring_rows[groups]]
        within = source_part_indexes - row_starts
        dest_part_indexes = geometry_offsets[multilinestring_target_rows[groups]] + within
        source_part_lengths = (
            multilinestring_buffer.part_offsets[source_part_indexes + 1]
            - multilinestring_buffer.part_offsets[source_part_indexes]
        ).astype(cp.int32, copy=False)
        part_lengths[dest_part_indexes] = source_part_lengths

    part_offsets = cp.empty(total_parts + 1, dtype=cp.int32)
    part_offsets[0] = 0
    if total_parts:
        part_offsets[1:] = cp.cumsum(part_lengths, dtype=cp.int32)

    total_coords = int(linestring_buffer.x.size) + int(multilinestring_buffer.x.size)
    x = cp.empty(total_coords, dtype=cp.float64)
    y = cp.empty(total_coords, dtype=cp.float64)
    _device_scatter_xy_offset_slices(
        source_x=linestring_buffer.x,
        source_y=linestring_buffer.y,
        source_offsets=linestring_buffer.geometry_offsets,
        source_rows=linestring_rows,
        target_rows=line_part_positions.astype(cp.int64, copy=False),
        target_offsets=part_offsets,
        out_x=x,
        out_y=y,
        total_coords=int(linestring_buffer.x.size),
    )
    if total_mls_parts:
        source_part_indexes = cp.arange(total_mls_parts, dtype=cp.int64)
        row_ends = multilinestring_buffer.geometry_offsets[multilinestring_rows + 1]
        groups = cp.searchsorted(row_ends, source_part_indexes, side="right").astype(
            cp.int64, copy=False
        )
        row_starts = multilinestring_buffer.geometry_offsets[multilinestring_rows[groups]]
        within = source_part_indexes - row_starts
        dest_part_indexes = geometry_offsets[multilinestring_target_rows[groups]] + within
        _device_scatter_xy_offset_slices(
            source_x=multilinestring_buffer.x,
            source_y=multilinestring_buffer.y,
            source_offsets=multilinestring_buffer.part_offsets,
            source_rows=source_part_indexes,
            target_rows=dest_part_indexes.astype(cp.int64, copy=False),
            target_offsets=part_offsets,
            out_x=x,
            out_y=y,
            total_coords=int(multilinestring_buffer.x.size),
        )
    promoted_buffer = DeviceFamilyGeometryBuffer(
        family=GeometryFamily.MULTILINESTRING,
        x=x,
        y=y,
        geometry_offsets=geometry_offsets,
        empty_mask=geometry_counts == 0,
        part_offsets=part_offsets,
    )
    result = build_device_resident_owned(
        device_families={GeometryFamily.MULTILINESTRING: promoted_buffer},
        row_count=owned.row_count,
        tags=promoted_tags,
        validity=state.validity,
        family_row_offsets=promoted_family_rows,
        execution_mode="gpu",
    )
    result._record(
        DiagnosticKind.CREATED,
        "device-side GeoArrow LineString/MultiLineString promotion to MultiLineString",
        visible=False,
    )
    return result


def _promote_polygon_multipolygon_geoarrow_owned_device(
    owned: OwnedGeometryArray,
) -> OwnedGeometryArray:
    import cupy as cp

    state, valid_count, promoted_tags, promoted_family_rows = _promoted_geoarrow_metadata(
        owned,
        GeometryFamily.MULTIPOLYGON,
    )
    polygon_buffer = state.families.get(GeometryFamily.POLYGON)
    multipolygon_buffer = state.families.get(GeometryFamily.MULTIPOLYGON)
    if polygon_buffer is None or multipolygon_buffer is None:
        raise ValueError("Polygon/MultiPolygon GeoArrow promotion requires both families")

    polygon_rows_global = cp.flatnonzero(
        state.validity & (state.tags == np.int8(FAMILY_TAGS[GeometryFamily.POLYGON]))
    ).astype(cp.int64, copy=False)
    polygon_rows = state.family_row_offsets[polygon_rows_global].astype(cp.int64, copy=False)
    polygon_target_rows = promoted_family_rows[polygon_rows_global].astype(cp.int64, copy=False)
    polygon_counts = cp.ones(int(polygon_rows.size), dtype=cp.int32)

    multipolygon_rows_global = cp.flatnonzero(
        state.validity & (state.tags == np.int8(FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]))
    ).astype(cp.int64, copy=False)
    multipolygon_rows = state.family_row_offsets[multipolygon_rows_global].astype(
        cp.int64, copy=False
    )
    multipolygon_target_rows = promoted_family_rows[multipolygon_rows_global].astype(
        cp.int64, copy=False
    )
    multipolygon_counts = (
        multipolygon_buffer.geometry_offsets[multipolygon_rows + 1]
        - multipolygon_buffer.geometry_offsets[multipolygon_rows]
    ).astype(cp.int32, copy=False)

    geometry_counts = cp.zeros(valid_count, dtype=cp.int32)
    if int(polygon_target_rows.size):
        geometry_counts[polygon_target_rows] = polygon_counts
    if int(multipolygon_target_rows.size):
        geometry_counts[multipolygon_target_rows] = multipolygon_counts
    geometry_offsets = cp.empty(valid_count + 1, dtype=cp.int32)
    geometry_offsets[0] = 0
    if valid_count:
        geometry_offsets[1:] = cp.cumsum(geometry_counts, dtype=cp.int32)

    total_polygon_parts = int(polygon_rows.size)
    total_mpoly_parts = max(int(multipolygon_buffer.part_offsets.size) - 1, 0)
    total_parts = total_polygon_parts + total_mpoly_parts
    part_ring_counts = cp.zeros(total_parts, dtype=cp.int32)

    polygon_part_positions = geometry_offsets[polygon_target_rows]
    polygon_ring_counts = (
        polygon_buffer.geometry_offsets[polygon_rows + 1]
        - polygon_buffer.geometry_offsets[polygon_rows]
    ).astype(cp.int32, copy=False)
    if int(polygon_part_positions.size):
        part_ring_counts[polygon_part_positions] = polygon_ring_counts

    if total_mpoly_parts:
        source_part_indexes = cp.arange(total_mpoly_parts, dtype=cp.int64)
        row_ends = multipolygon_buffer.geometry_offsets[multipolygon_rows + 1]
        groups = cp.searchsorted(row_ends, source_part_indexes, side="right").astype(
            cp.int64, copy=False
        )
        row_starts = multipolygon_buffer.geometry_offsets[multipolygon_rows[groups]]
        within = source_part_indexes - row_starts
        dest_part_indexes = geometry_offsets[multipolygon_target_rows[groups]] + within
        source_ring_counts = (
            multipolygon_buffer.part_offsets[source_part_indexes + 1]
            - multipolygon_buffer.part_offsets[source_part_indexes]
        ).astype(cp.int32, copy=False)
        part_ring_counts[dest_part_indexes] = source_ring_counts

    part_offsets = cp.empty(total_parts + 1, dtype=cp.int32)
    part_offsets[0] = 0
    if total_parts:
        part_offsets[1:] = cp.cumsum(part_ring_counts, dtype=cp.int32)

    total_polygon_rings = max(int(polygon_buffer.ring_offsets.size) - 1, 0)
    total_mpoly_rings = max(int(multipolygon_buffer.ring_offsets.size) - 1, 0)
    total_rings = total_polygon_rings + total_mpoly_rings
    ring_lengths = cp.zeros(total_rings, dtype=cp.int32)

    if total_polygon_rings:
        source_ring_indexes = cp.arange(total_polygon_rings, dtype=cp.int64)
        row_ends = polygon_buffer.geometry_offsets[polygon_rows + 1]
        groups = cp.searchsorted(row_ends, source_ring_indexes, side="right").astype(
            cp.int64, copy=False
        )
        row_starts = polygon_buffer.geometry_offsets[polygon_rows[groups]]
        within = source_ring_indexes - row_starts
        dest_ring_indexes = part_offsets[polygon_part_positions[groups]] + within
        source_ring_lengths = (
            polygon_buffer.ring_offsets[source_ring_indexes + 1]
            - polygon_buffer.ring_offsets[source_ring_indexes]
        ).astype(cp.int32, copy=False)
        ring_lengths[dest_ring_indexes] = source_ring_lengths

    if total_mpoly_rings:
        source_ring_indexes = cp.arange(total_mpoly_rings, dtype=cp.int64)
        source_part_ends = multipolygon_buffer.part_offsets[
            cp.arange(total_mpoly_parts, dtype=cp.int64) + 1
        ]
        part_groups = cp.searchsorted(source_part_ends, source_ring_indexes, side="right").astype(
            cp.int64, copy=False
        )
        source_part_starts = multipolygon_buffer.part_offsets[part_groups]
        within_part = source_ring_indexes - source_part_starts
        row_ends = multipolygon_buffer.geometry_offsets[multipolygon_rows + 1]
        row_groups = cp.searchsorted(row_ends, part_groups, side="right").astype(
            cp.int64, copy=False
        )
        row_starts = multipolygon_buffer.geometry_offsets[multipolygon_rows[row_groups]]
        within_row = part_groups - row_starts
        dest_part_indexes = geometry_offsets[multipolygon_target_rows[row_groups]] + within_row
        dest_ring_indexes = part_offsets[dest_part_indexes] + within_part
        source_ring_lengths = (
            multipolygon_buffer.ring_offsets[source_ring_indexes + 1]
            - multipolygon_buffer.ring_offsets[source_ring_indexes]
        ).astype(cp.int32, copy=False)
        ring_lengths[dest_ring_indexes] = source_ring_lengths

    ring_offsets = cp.empty(total_rings + 1, dtype=cp.int32)
    ring_offsets[0] = 0
    if total_rings:
        ring_offsets[1:] = cp.cumsum(ring_lengths, dtype=cp.int32)

    total_coords = int(polygon_buffer.x.size) + int(multipolygon_buffer.x.size)
    x = cp.empty(total_coords, dtype=cp.float64)
    y = cp.empty(total_coords, dtype=cp.float64)
    if total_polygon_rings:
        source_ring_indexes = cp.arange(total_polygon_rings, dtype=cp.int64)
        row_ends = polygon_buffer.geometry_offsets[polygon_rows + 1]
        groups = cp.searchsorted(row_ends, source_ring_indexes, side="right").astype(
            cp.int64, copy=False
        )
        row_starts = polygon_buffer.geometry_offsets[polygon_rows[groups]]
        within = source_ring_indexes - row_starts
        dest_ring_indexes = part_offsets[polygon_part_positions[groups]] + within
        _device_scatter_xy_offset_slices(
            source_x=polygon_buffer.x,
            source_y=polygon_buffer.y,
            source_offsets=polygon_buffer.ring_offsets,
            source_rows=source_ring_indexes,
            target_rows=dest_ring_indexes.astype(cp.int64, copy=False),
            target_offsets=ring_offsets,
            out_x=x,
            out_y=y,
            total_coords=int(polygon_buffer.x.size),
        )
    if total_mpoly_rings:
        source_ring_indexes = cp.arange(total_mpoly_rings, dtype=cp.int64)
        source_part_ends = multipolygon_buffer.part_offsets[
            cp.arange(total_mpoly_parts, dtype=cp.int64) + 1
        ]
        part_groups = cp.searchsorted(source_part_ends, source_ring_indexes, side="right").astype(
            cp.int64, copy=False
        )
        source_part_starts = multipolygon_buffer.part_offsets[part_groups]
        within_part = source_ring_indexes - source_part_starts
        row_ends = multipolygon_buffer.geometry_offsets[multipolygon_rows + 1]
        row_groups = cp.searchsorted(row_ends, part_groups, side="right").astype(
            cp.int64, copy=False
        )
        row_starts = multipolygon_buffer.geometry_offsets[multipolygon_rows[row_groups]]
        within_row = part_groups - row_starts
        dest_part_indexes = geometry_offsets[multipolygon_target_rows[row_groups]] + within_row
        dest_ring_indexes = part_offsets[dest_part_indexes] + within_part
        _device_scatter_xy_offset_slices(
            source_x=multipolygon_buffer.x,
            source_y=multipolygon_buffer.y,
            source_offsets=multipolygon_buffer.ring_offsets,
            source_rows=source_ring_indexes,
            target_rows=dest_ring_indexes.astype(cp.int64, copy=False),
            target_offsets=ring_offsets,
            out_x=x,
            out_y=y,
            total_coords=int(multipolygon_buffer.x.size),
        )
    promoted_buffer = DeviceFamilyGeometryBuffer(
        family=GeometryFamily.MULTIPOLYGON,
        x=x,
        y=y,
        geometry_offsets=geometry_offsets,
        empty_mask=geometry_counts == 0,
        part_offsets=part_offsets,
        ring_offsets=ring_offsets,
    )
    result = build_device_resident_owned(
        device_families={GeometryFamily.MULTIPOLYGON: promoted_buffer},
        row_count=owned.row_count,
        tags=promoted_tags,
        validity=state.validity,
        family_row_offsets=promoted_family_rows,
        execution_mode="gpu",
    )
    result._record(
        DiagnosticKind.CREATED,
        "device-side GeoArrow Polygon/MultiPolygon promotion to MultiPolygon",
        visible=False,
    )
    return result


def _promote_supported_geoarrow_owned_device(
    owned: OwnedGeometryArray,
    *,
    export_family: GeometryFamily,
) -> OwnedGeometryArray:
    family_set = _device_geoarrow_family_set(owned)
    if export_family is GeometryFamily.MULTIPOINT and family_set == frozenset(
        {GeometryFamily.POINT, GeometryFamily.MULTIPOINT}
    ):
        return _promote_point_multipoint_geoarrow_owned_device(owned)
    if export_family is GeometryFamily.MULTILINESTRING and family_set == frozenset(
        {GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING}
    ):
        return _promote_linestring_multilinestring_geoarrow_owned_device(owned)
    if export_family is GeometryFamily.MULTIPOLYGON and family_set == frozenset(
        {GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON}
    ):
        return _promote_polygon_multipolygon_geoarrow_owned_device(owned)
    raise ValueError("Native GeoArrow fast path requires a homogeneous geometry family")


def _homogeneous_family_from_device_structure(owned: OwnedGeometryArray) -> GeometryFamily:
    if getattr(owned, "is_indexed_view", False):
        state = owned.device_state
        if state is not None:
            if state.trusted_homogeneous_family is not None:
                return state.trusted_homogeneous_family
            if len(state.families) == 1:
                return next(iter(state.families))
        validity = getattr(owned, "_validity", None)
        tags = getattr(owned, "_tags", None)
        if validity is not None and tags is not None:
            valid_tags = np.asarray(tags[np.asarray(validity, dtype=bool)], dtype=np.int8)
            if valid_tags.size == 0:
                raise ValueError("Cannot encode an all-null geometry array to native GeoArrow")
            unique_tags = np.unique(valid_tags)
            if unique_tags.size != 1:
                raise ValueError("Native GeoArrow fast path requires a homogeneous geometry family")
            return TAG_FAMILIES[int(unique_tags[0])]
        raise ValueError("Native GeoArrow fast path requires indexed-view family metadata")
    state = owned.device_state
    if state is None:
        raise ValueError("device structure is unavailable")
    families = tuple(
        family for family in state.families if _device_family_row_count(owned, family) > 0
    )
    if not families:
        raise ValueError("Cannot encode an all-null geometry array to native GeoArrow")
    if len(families) != 1:
        raise ValueError("Native GeoArrow fast path requires a homogeneous geometry family")
    return families[0]


def _device_validity_gpumask(
    owned: OwnedGeometryArray,
    *,
    valid_row_count: int | None = None,
):
    import cupy as cp
    import pylibcudf as plc

    validity = cp.asarray(_terminal_device_state(owned).validity)
    if valid_row_count is None:
        valid_row_count = _device_structure_valid_row_count(owned)
    null_count = (
        owned.row_count - valid_row_count
        if valid_row_count is not None
        else _device_scalar_to_host_int(
            cp.count_nonzero(~validity),
            reason="native geometry encode validity null-count scalar fence",
        )
    )
    if null_count == 0:
        return None, 0
    validity_bytes = cp.packbits(validity.astype(cp.uint8), bitorder="little")
    return plc.gpumemoryview(validity_bytes.view(cp.uint8)), null_count


def _device_point_values_column(x_device, y_device, *, mask=None, null_count: int = 0):
    import pylibcudf as plc

    from vibespatial.cuda._runtime import pylibcudf_column_from_device

    x_col = pylibcudf_column_from_device(x_device)
    y_col = pylibcudf_column_from_device(y_device)
    if mask is not None and null_count:
        x_col = x_col.with_mask(mask, null_count)
        y_col = y_col.with_mask(mask, null_count)
        return plc.Column.struct_from_children([x_col, y_col]).with_mask(mask, null_count)
    return plc.Column.struct_from_children([x_col, y_col])


def _device_list_column(offsets_device, child_column, size: int):
    import pylibcudf as plc

    from vibespatial.cuda._runtime import pylibcudf_column_from_device

    offsets_col = pylibcudf_column_from_device(offsets_device)
    return plc.Column(
        plc.types.DataType(plc.types.TypeId.LIST),
        int(size),
        None,
        None,
        0,
        0,
        [offsets_col, child_column],
    )


def _encode_owned_geoarrow_column_device(owned: OwnedGeometryArray):
    import cupy as cp

    family, requires_promotion = _device_geoarrow_export_family(owned)
    if requires_promotion:
        owned = _promote_supported_geoarrow_owned_device(
            owned,
            export_family=family,
        )
    state = _terminal_device_state(owned)
    device_buffer = state.families[family]
    mask, null_count = _device_validity_gpumask(owned)

    if family is GeometryFamily.POINT:
        row_count = owned.row_count
        valid_point_rows = _device_family_row_count(owned, GeometryFamily.POINT)
        empty_count = max(valid_point_rows - int(device_buffer.x.size), 0)
        if null_count == 0 and empty_count == 0:
            family_rows = cp.asarray(state.family_row_offsets).astype(cp.int32, copy=False)
            coord_indices = device_buffer.geometry_offsets[family_rows]
            column = _device_point_values_column(
                device_buffer.x[coord_indices],
                device_buffer.y[coord_indices],
            )
            return column, "point"
        x_full = cp.full(row_count, cp.nan, dtype=cp.float64)
        y_full = cp.full(row_count, cp.nan, dtype=cp.float64)
        family_valid = cp.asarray(state.validity) & (
            cp.asarray(state.tags) == np.int8(FAMILY_TAGS[GeometryFamily.POINT])
        )
        valid_rows = cp.flatnonzero(family_valid)
        family_rows = cp.asarray(state.family_row_offsets, dtype=cp.int64)[family_valid]
        non_empty_mask = ~cp.asarray(device_buffer.empty_mask)[family_rows]
        coord_indices = device_buffer.geometry_offsets[family_rows[non_empty_mask]]
        target_rows = valid_rows[non_empty_mask]
        x_full[target_rows] = device_buffer.x[coord_indices]
        y_full[target_rows] = device_buffer.y[coord_indices]
        column = _device_point_values_column(x_full, y_full, mask=mask, null_count=null_count)
        return column, "point"

    point_values = _device_point_values_column(device_buffer.x, device_buffer.y)

    if family is GeometryFamily.LINESTRING:
        full_offsets = _device_full_offsets_from_local(
            owned,
            family,
            device_buffer.geometry_offsets,
            empty_mask=device_buffer.empty_mask,
        )
        column = _device_list_column(full_offsets, point_values, owned.row_count)
        if null_count:
            column = column.with_mask(mask, null_count)
        return column, "linestring"

    if family is GeometryFamily.MULTIPOINT:
        full_offsets = _device_full_offsets_from_local(
            owned,
            family,
            device_buffer.geometry_offsets,
            empty_mask=device_buffer.empty_mask,
        )
        column = _device_list_column(full_offsets, point_values, owned.row_count)
        if null_count:
            column = column.with_mask(mask, null_count)
        return column, "multipoint"

    if family is GeometryFamily.POLYGON:
        rings = _device_list_column(
            device_buffer.ring_offsets, point_values, int(device_buffer.ring_offsets.size - 1)
        )
        full_offsets = _device_full_offsets_from_local(
            owned,
            family,
            device_buffer.geometry_offsets,
            empty_mask=device_buffer.empty_mask,
        )
        column = _device_list_column(full_offsets, rings, owned.row_count)
        if null_count:
            column = column.with_mask(mask, null_count)
        return column, "polygon"

    if family is GeometryFamily.MULTILINESTRING:
        parts = _device_list_column(
            device_buffer.part_offsets, point_values, int(device_buffer.part_offsets.size - 1)
        )
        full_offsets = _device_full_offsets_from_local(
            owned,
            family,
            device_buffer.geometry_offsets,
            empty_mask=device_buffer.empty_mask,
        )
        column = _device_list_column(full_offsets, parts, owned.row_count)
        if null_count:
            column = column.with_mask(mask, null_count)
        return column, "multilinestring"

    if family is GeometryFamily.MULTIPOLYGON:
        rings = _device_list_column(
            device_buffer.ring_offsets, point_values, int(device_buffer.ring_offsets.size - 1)
        )
        polygons = _device_list_column(
            device_buffer.part_offsets, rings, int(device_buffer.part_offsets.size - 1)
        )
        full_offsets = _device_full_offsets_from_local(
            owned,
            family,
            device_buffer.geometry_offsets,
            empty_mask=device_buffer.empty_mask,
        )
        column = _device_list_column(full_offsets, polygons, owned.row_count)
        if null_count:
            column = column.with_mask(mask, null_count)
        return column, "multipolygon"

    raise ValueError(f"Unsupported geometry family for device GeoArrow encode: {family}")


def _apply_geoarrow_child_metadata(column_meta, family: GeometryFamily) -> None:
    if family is GeometryFamily.POINT:
        column_meta.child(0).set_name("x")
        column_meta.child(1).set_name("y")
        return

    point_meta = None
    if family in {GeometryFamily.LINESTRING, GeometryFamily.MULTIPOINT}:
        point_meta = column_meta.child(1)
    elif family in {GeometryFamily.POLYGON, GeometryFamily.MULTILINESTRING}:
        point_meta = column_meta.child(1).child(1)
    elif family is GeometryFamily.MULTIPOLYGON:
        point_meta = column_meta.child(1).child(1).child(1)
    if point_meta is not None:
        point_meta.child(0).set_name("x")
        point_meta.child(1).set_name("y")


def _compression_type_from_name(name: str):
    import pylibcudf as plc

    return getattr(
        plc.io.types.CompressionType, str(name).upper(), plc.io.types.CompressionType.AUTO
    )


def _native_parquet_compression_supported(name: str | None) -> bool:
    normalized = None if name is None else str(name).lower()
    return normalized in {None, "snappy", "lz4", "zstd"}


def _pandas_dtype_for_arrow_type(arrow_type):
    try:
        return arrow_type.to_pandas_dtype()
    except (AttributeError, NotImplementedError, TypeError, ValueError):
        return object


def _empty_pandas_series_for_arrow_field(field, *, name):
    import pandas as pd

    dtype = _pandas_dtype_for_arrow_type(field.type)
    try:
        return pd.Series([], dtype=dtype, name=name)
    except (TypeError, ValueError):
        return pd.Series([], dtype=object, name=name)


def _native_index_field_name(index_name, level: int, column_names) -> str:
    taken = {str(name) for name in column_names}
    if index_name is not None:
        candidate = str(index_name)
        if candidate not in taken:
            return candidate
    base = f"__index_level_{level:d}__"
    candidate = base
    suffix = 1
    while candidate in taken:
        candidate = f"{base}_{suffix:d}"
        suffix += 1
    return candidate


def _range_index_for_native_export(index_plan, fallback_index, row_count: int):
    import pandas as pd

    index = getattr(index_plan, "index", None)
    if isinstance(index, pd.RangeIndex) and len(index) == int(row_count):
        return index
    if index_plan is None and isinstance(fallback_index, pd.RangeIndex):
        if len(fallback_index) == int(row_count):
            return fallback_index
    return None


def _native_device_index_columns(
    *,
    index_plan,
    fallback_index,
    row_count: int,
    index,
    attribute_columns,
    pa,
    plc,
) -> tuple[tuple[_NativeDeviceIndexColumn, ...], Any | None] | None:
    if index is False:
        return (), None

    range_index = _range_index_for_native_export(
        index_plan,
        fallback_index,
        row_count,
    )
    if index is None and range_index is not None:
        return (), range_index

    if index_plan is not None and getattr(index_plan, "nlevels", 1) != 1:
        return None

    import cupy as cp
    import numpy as _np
    import pandas as pd

    labels = None
    logical_name = None
    if range_index is not None:
        logical_name = range_index.name
        labels = cp.arange(int(row_count), dtype=cp.int64) * _np.int64(
            range_index.step
        ) + _np.int64(range_index.start)
    elif index_plan is not None and getattr(index_plan, "device_labels", None) is not None:
        logical_name = getattr(index_plan, "name", None)
        labels = cp.asarray(index_plan.device_labels)
    else:
        return None

    if labels.ndim != 1 or int(labels.size) != int(row_count):
        return None
    dtype = _np.dtype(labels.dtype)
    if not (_np.issubdtype(dtype, _np.number) or _np.issubdtype(dtype, _np.bool_)):
        return None

    field_name = _native_index_field_name(logical_name, 0, attribute_columns)
    from vibespatial.cuda._runtime import pylibcudf_column_from_device

    column = pylibcudf_column_from_device(labels)
    field = pa.field(field_name, column.type().to_arrow())
    metadata_index = pd.Index(
        [],
        dtype=_pandas_dtype_for_arrow_type(field.type),
        name=logical_name,
    )
    return (
        (
            _NativeDeviceIndexColumn(
                field_name=field_name,
                logical_name=logical_name,
                column=column,
                field=field,
                metadata_index=metadata_index,
            ),
        ),
        None,
    )


def _native_pandas_schema_metadata(
    *,
    attribute_fields,
    index_columns: tuple[_NativeDeviceIndexColumn, ...],
    range_index,
    preserve_index,
):
    import pandas as pd
    import pyarrow.pandas_compat as pandas_compat

    columns_to_convert = []
    column_names = []
    column_field_names = []
    types = []
    for field in attribute_fields:
        columns_to_convert.append(_empty_pandas_series_for_arrow_field(field, name=field.name))
        column_names.append(field.name)
        column_field_names.append(field.name)
        types.append(field.type)

    prototype = pd.DataFrame(index=pd.RangeIndex(0), columns=column_field_names)
    index_levels = []
    index_descriptors = []
    if preserve_index is not False:
        if range_index is not None:
            index_levels.append(range_index)
            index_descriptors.append(pandas_compat._get_range_index_descriptor(range_index))
        else:
            for column in index_columns:
                index_levels.append(column.metadata_index)
                index_descriptors.append(column.field_name)
                types.append(column.field.type)

    return pandas_compat.construct_metadata(
        columns_to_convert,
        prototype,
        column_names,
        index_levels,
        index_descriptors,
        preserve_index,
        types,
        column_field_names=column_field_names,
    )


def _try_native_device_attribute_export(
    attribute_frame,
    non_geometry_columns,
    *,
    index,
    index_plan,
    pa,
    plc,
) -> tuple[dict[Any, Any], Any, tuple[str, ...]] | None:
    try:
        from vibespatial.api._native_result_core import NativeAttributeTable
    except Exception:
        return None

    if not isinstance(attribute_frame, NativeAttributeTable):
        return None
    if getattr(attribute_frame, "device_table", None) is None:
        return None

    try:
        columns = attribute_frame.to_pylibcudf_columns(non_geometry_columns)
        attribute_schema = attribute_frame.arrow_schema_for_columns(non_geometry_columns)
        index_result = _native_device_index_columns(
            index_plan=index_plan,
            fallback_index=attribute_frame.index,
            row_count=len(attribute_frame),
            index=index,
            attribute_columns=non_geometry_columns,
            pa=pa,
            plc=plc,
        )
    except (
        AttributeError,
        KeyError,
        ModuleNotFoundError,
        TypeError,
        ValueError,
    ):
        return None

    if index_result is None:
        return None

    index_columns, range_index = index_result
    attribute_fields = list(attribute_schema)
    fields = [*attribute_fields, *(column.field for column in index_columns)]
    metadata = {
        key: value for key, value in (attribute_schema.metadata or {}).items() if key != b"pandas"
    }
    metadata.update(
        _native_pandas_schema_metadata(
            attribute_fields=attribute_fields,
            index_columns=index_columns,
            range_index=range_index,
            preserve_index=index,
        )
    )
    device_columns = dict(zip(non_geometry_columns, columns, strict=True))
    device_columns.update({column.field_name: column.column for column in index_columns})
    return (
        device_columns,
        pa.schema(fields, metadata=metadata or None),
        tuple(column.field_name for column in index_columns),
    )


def _attribute_column_to_plc(arrow_column, col_name, *, plc):
    """Convert attribute column to pylibcudf column, preferring device path.

    If the underlying column data exposes ``__cuda_array_interface__`` (e.g.
    CuPy-backed Pandas columns), build the pylibcudf Column directly from the
    device pointer, avoiding a device-to-host-to-device round-trip.  Otherwise
    fall through to the host Arrow path.
    """
    combined = arrow_column.combine_chunks()
    # For numeric columns backed by CuPy, the pandas-to-arrow conversion
    # materialises a host copy.  Check the *original* pandas values instead.
    # However we only have the Arrow column at this point, so check its buffers
    # for a CUDA array interface (cudf/cupy-backed pyarrow arrays expose one).
    if len(combined) > 0 and hasattr(combined, "buffers"):
        bufs = combined.buffers()
        if (
            bufs
            and len(bufs) > 1
            and bufs[1] is not None
            and hasattr(bufs[1], "__cuda_array_interface__")
        ):
            from vibespatial.cuda._runtime import pylibcudf_column_from_device

            return pylibcudf_column_from_device(bufs[1])
    from vibespatial.cuda._runtime import pylibcudf_current_stream

    return plc.Column.from_arrow(combined, stream=pylibcudf_current_stream())


def _native_host_attribute_table_from_pandas(df, non_geometry_columns, *, index, pa):
    import pandas as pd

    df_attr = pd.DataFrame(
        {column_name: df[column_name] for column_name in non_geometry_columns},
        index=df.index,
        copy=False,
    )
    return pa.Table.from_pandas(df_attr, preserve_index=index)


def _build_native_host_attribute_table_from_frame(attribute_frame, ordered_columns, *, index, pa):
    """Build a host Arrow attribute table from a non-geometry frame."""
    if hasattr(attribute_frame, "to_arrow"):
        try:
            return attribute_frame.to_arrow(index=index, columns=ordered_columns)
        except TypeError:
            pass

    import pandas as pd

    from vibespatial.api._native_result_core import _arrow_compatible_pandas_frame

    df_attr = pd.DataFrame(
        {column_name: attribute_frame[column_name] for column_name in ordered_columns},
        index=attribute_frame.index,
        copy=False,
    )
    df_attr = _arrow_compatible_pandas_frame(df_attr)
    pandas_metadata = pa.Schema.from_pandas(df_attr, preserve_index=index).metadata

    if index not in (None, False):
        return pa.Table.from_pandas(df_attr, preserve_index=index)

    if not ordered_columns:
        return pa.table({}).replace_schema_metadata(pandas_metadata)

    arrays = []
    names = []
    for column_name in ordered_columns:
        values = df_attr[column_name].to_numpy(copy=False)
        if not isinstance(values, np.ndarray) or values.dtype == object:
            return pa.Table.from_pandas(df_attr, preserve_index=index)
        try:
            arrays.append(pa.array(values, from_pandas=True))
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError):
            return pa.Table.from_pandas(df_attr, preserve_index=index)
        names.append(column_name)
    return pa.Table.from_arrays(arrays, names=names).replace_schema_metadata(pandas_metadata)


def _build_native_host_attribute_table(df, non_geometry_columns, *, index, pa):
    """Build the host Arrow table for native device writes.

    Fast path: when there is no explicit index request and all non-geometry
    columns are plain NumPy-backed, build Arrow columns directly and skip the
    heavier DataFrame->Arrow conversion path. Fall back to ``from_pandas`` for
    categoricals, object columns, nullable extension arrays, or index writes.
    """
    return _build_native_host_attribute_table_from_frame(
        df,
        non_geometry_columns,
        index=index,
        pa=pa,
    )


def _write_pylibcudf_parquet_table(
    plc,
    plc_table,
    *,
    sink,
    metadata,
    footer_metadata,
    compression,
    writer_kwargs,
) -> None:
    """Write a device table while honoring explicit row-group boundaries."""
    from vibespatial.cuda._runtime import pylibcudf_current_stream

    stream = pylibcudf_current_stream(plc_table)
    row_group_size = writer_kwargs.get("row_group_size")
    row_count = plc_table.num_rows()
    if row_group_size is not None:
        row_group_size = int(row_group_size)
        if row_group_size <= 0:
            raise ValueError("row_group_size must be greater than zero")

    if row_group_size is not None and row_group_size < row_count:
        builder = plc.io.parquet.ChunkedParquetWriterOptions.builder(
            plc.io.types.SinkInfo([sink])
        )
        builder.metadata(metadata)
        builder.key_value_metadata([footer_metadata])
        builder.write_arrow_schema(False)
        builder.compression(_compression_type_from_name(compression))
        builder.row_group_size_rows(row_group_size)
        if "max_page_size" in writer_kwargs:
            builder.max_page_size_bytes(int(writer_kwargs["max_page_size"]))
        writer = plc.io.parquet.ChunkedParquetWriter.from_options(
            builder.build(),
            stream=stream,
        )
        slice_bounds = [
            bound
            for start in range(0, row_count, row_group_size)
            for bound in (start, min(start + row_group_size, row_count))
        ]
        for chunk in plc.copying.slice(plc_table, slice_bounds, stream=stream):
            writer.write(chunk)
        writer.close([])
        return

    builder = plc.io.parquet.ParquetWriterOptions.builder(
        plc.io.types.SinkInfo([sink]),
        plc_table,
    )
    builder.metadata(metadata)
    builder.key_value_metadata([footer_metadata])
    builder.write_arrow_schema(False)
    builder.compression(_compression_type_from_name(compression))
    if row_group_size is not None:
        builder.row_group_size_rows(row_group_size)
    if "max_page_size" in writer_kwargs:
        builder.max_page_size_bytes(int(writer_kwargs["max_page_size"]))
    plc.io.parquet.write_parquet(builder.build(), stream=stream)


_NATIVE_DEVICE_PARQUET_CHUNK_ROWS = 1_000_000
_NATIVE_WKB_AVAILABLE_MEMORY_DIVISOR = 2
_NATIVE_WKB_MAX_COLUMN_BYTES = int(np.iinfo(np.int32).max)
_NATIVE_WKB_ROW_TEMPORARY_BYTES = 32
_NATIVE_WKB_BBOX_TEMPORARY_BYTES = 64


@dataclass(frozen=True)
class _NativeWkbChunkEstimate:
    row_count: int
    output_bytes: int
    temporary_bytes: int
    composition_bytes: int

    @property
    def live_allocation_bytes(self) -> int:
        return self.output_bytes + self.temporary_bytes + self.composition_bytes


@dataclass(frozen=True)
class _NativeWkbCapacityPlan:
    """Physical allocation plan for one terminal native WKB export."""

    metadata: Any
    max_chunk_rows: int
    max_output_bytes_per_row: int
    max_owned_bytes_per_row: int
    metadata_bytes_per_row: int
    composition: bool
    write_covering_bbox: bool
    ordered_parts: tuple[tuple[int, int, Any], ...] | None = None
    planning_packet_scalars: int = 0

    def estimate(self, row_count: int) -> _NativeWkbChunkEstimate:
        rows = max(int(row_count), 0)
        output_bytes = rows * self.max_output_bytes_per_row
        row_temporary_bytes = (
            _NATIVE_WKB_ROW_TEMPORARY_BYTES
            + self.metadata_bytes_per_row
            + (
                _NATIVE_WKB_BBOX_TEMPORARY_BYTES
                if self.write_covering_bbox
                else 0
            )
        )
        # Reserve one output-sized scratch region for WKB assembly and
        # libcudf compression.  The encoded payload itself is accounted for
        # separately as output_bytes.
        temporary_bytes = output_bytes + rows * row_temporary_bytes
        # Composition may first compact selected fragments and then allocate
        # a second contiguous owned carrier for concat.  Both are live when
        # encoding starts, so admission must cover the doubled geometry shape.
        composition_bytes = (
            2 * rows * self.max_owned_bytes_per_row if self.composition else 0
        )
        return _NativeWkbChunkEstimate(
            row_count=rows,
            output_bytes=output_bytes,
            temporary_bytes=temporary_bytes,
            composition_bytes=composition_bytes,
        )

    def admitted_rows(
        self,
        remaining_rows: int,
        *,
        available_device_bytes: int | None = None,
    ) -> int:
        remaining = max(int(remaining_rows), 0)
        if remaining == 0:
            return 0
        row_limit = min(remaining, self.max_chunk_rows)
        if self.max_output_bytes_per_row:
            row_limit = min(
                row_limit,
                _NATIVE_WKB_MAX_COLUMN_BYTES // self.max_output_bytes_per_row,
            )
        if available_device_bytes is None:
            available_device_bytes = _available_native_wkb_device_bytes()
        if available_device_bytes is not None:
            allocation_budget = max(int(available_device_bytes), 0) // (
                _NATIVE_WKB_AVAILABLE_MEMORY_DIVISOR
            )
            one_row_live_bytes = self.estimate(1).live_allocation_bytes
            if one_row_live_bytes:
                row_limit = min(row_limit, allocation_budget // one_row_live_bytes)
        if row_limit > 0:
            return row_limit

        one_row = self.estimate(1)
        available_detail = (
            "unknown"
            if available_device_bytes is None
            else f"{int(available_device_bytes):,}"
        )
        raise MemoryError(
            "one native WKB export row exceeds the current device allocation "
            f"budget: output_bytes={one_row.output_bytes:,}; "
            f"temporary_bytes={one_row.temporary_bytes:,}; "
            f"composition_bytes={one_row.composition_bytes:,}; "
            f"available_device_bytes={available_detail}"
        )

    def requires_chunking(self, row_count: int) -> bool:
        rows = max(int(row_count), 0)
        return rows > 0 and self.admitted_rows(rows) < rows


def _available_native_wkb_device_bytes() -> int | None:
    """Return bytes allocatable through the active driver and memory pool."""
    try:
        import cupy as cp
    except ModuleNotFoundError:
        return None

    driver_free, driver_total = cp.cuda.Device().mem_info
    pool_free = int(get_cuda_runtime().memory_pool_stats().get("free_bytes", 0))
    return min(int(driver_total), int(driver_free) + pool_free)


def _device_maximum_i64(values):
    import cupy as cp

    values = cp.asarray(values, dtype=cp.int64)
    if int(values.size) == 0:
        return cp.asarray(0, dtype=cp.int64)
    return cp.max(values).astype(cp.int64, copy=False)


def _device_family_size_packet(family: GeometryFamily, buffer):
    """Return first-level, second-level, and coordinate maxima on device."""
    import cupy as cp

    geometry_offsets = cp.asarray(buffer.geometry_offsets, dtype=cp.int64)
    geometry_counts = geometry_offsets[1:] - geometry_offsets[:-1]
    first_maximum = cp.asarray(0, dtype=cp.int64)
    second_maximum = cp.asarray(0, dtype=cp.int64)
    if family in {
        GeometryFamily.POLYGON,
        GeometryFamily.MULTILINESTRING,
        GeometryFamily.MULTIPOLYGON,
    }:
        first_maximum = _device_maximum_i64(geometry_counts)
    if family is GeometryFamily.MULTIPOLYGON:
        part_offsets = cp.asarray(buffer.part_offsets, dtype=cp.int64)
        ring_starts = part_offsets[geometry_offsets[:-1]]
        ring_stops = part_offsets[geometry_offsets[1:]]
        second_maximum = _device_maximum_i64(ring_stops - ring_starts)
    coordinate_maximum = _device_maximum_i64(
        device_family_coordinate_counts(buffer)
    )
    return cp.stack((first_maximum, second_maximum, coordinate_maximum))


def _fixed_or_maximum(fixed_size, field: str) -> int | None:
    if fixed_size is None:
        return None
    maximum = getattr(fixed_size, f"max_{field}", None)
    if maximum is not None:
        return int(maximum)
    fixed = getattr(fixed_size, field, None)
    return None if fixed is None else int(fixed)


def _family_size_proof_complete(family: GeometryFamily, fixed_size) -> bool:
    if _fixed_or_maximum(fixed_size, "coord_count_per_row") is None:
        return False
    if family in {
        GeometryFamily.POLYGON,
        GeometryFamily.MULTILINESTRING,
        GeometryFamily.MULTIPOLYGON,
    } and _fixed_or_maximum(fixed_size, "first_level_count_per_row") is None:
        return False
    return not (
        family is GeometryFamily.MULTIPOLYGON
        and _fixed_or_maximum(fixed_size, "second_level_count_per_row") is None
    )


def _candidate_ordered_composition_parts(composition):
    """Return candidate contiguous parts and device-only certification checks."""
    if (
        not composition.trusted_singular_rows
        or composition.residency is not Residency.DEVICE
        or any(part.collection_position is not None for part in composition.parts)
    ):
        return None, ()

    import cupy as cp

    owned_parts = []
    total_rows = 0
    for part in composition.parts:
        part_owned = part.geometry.owned
        if part_owned is None:
            return None, ()
        part_rows = int(part_owned.row_count)
        if part_rows != int(part.output_rows.shape[0]):
            return None, ()
        if part_rows:
            owned_parts.append(
                (total_rows, total_rows + part_rows, part_owned, part.output_rows)
            )
        total_rows += part_rows
    if total_rows != int(composition.row_count):
        return None, ()

    checks = []
    if not composition.contiguous_row_partitions:
        for start, stop, _part_owned, output_rows in owned_parts:
            rows = cp.asarray(output_rows, dtype=cp.int64)
            internally_contiguous = (
                cp.asarray(True)
                if rows.size == 1
                else cp.all(rows[1:] == rows[:-1] + cp.int64(1))
            )
            checks.append(
                internally_contiguous
                & (rows[0] == cp.int64(start))
                & (rows[-1] == cp.int64(stop - 1))
            )
    return tuple(
        (start, stop, part_owned)
        for start, stop, part_owned, _output_rows in owned_parts
    ), tuple(checks)


def _native_wkb_family_output_bytes(
    family: GeometryFamily,
    *,
    first_count: int,
    second_count: int,
    coordinate_count: int,
) -> int:
    if family is GeometryFamily.POINT:
        return 21
    if family is GeometryFamily.LINESTRING:
        return 9 + 16 * coordinate_count
    if family is GeometryFamily.POLYGON:
        return 9 + 4 * first_count + 16 * coordinate_count
    if family is GeometryFamily.MULTIPOINT:
        return 9 + 21 * coordinate_count
    if family is GeometryFamily.MULTILINESTRING:
        return 9 + 9 * first_count + 16 * coordinate_count
    if family is GeometryFamily.MULTIPOLYGON:
        return (
            9
            + 9 * first_count
            + 4 * second_count
            + 16 * coordinate_count
        )
    raise ValueError(f"Unsupported geometry family for WKB capacity: {family}")


def _device_row_metadata_bytes(state) -> int:
    total = 0
    for values in (
        state.validity,
        state.tags,
        state.family_row_offsets,
        state.row_bounds,
    ):
        if values is None:
            continue
        shape = tuple(int(size) for size in values.shape)
        trailing_count = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
        total += int(values.dtype.itemsize) * trailing_count
    return total


def _native_wkb_capacity_plan_from_metadata(
    metadata,
    *,
    max_chunk_rows: int,
    composition: bool,
    write_covering_bbox: bool,
    ordered_parts=None,
    planning_packet_scalars: int = 0,
) -> _NativeWkbCapacityPlan:
    """Build a reusable admission plan from structural native metadata."""
    summary = metadata.shape_summary
    return _NativeWkbCapacityPlan(
        metadata=metadata,
        max_chunk_rows=max(int(max_chunk_rows), 1),
        max_output_bytes_per_row=max(
            int(summary.get("max_wkb_output_bytes_per_row", 0)),
            0,
        ),
        max_owned_bytes_per_row=max(
            int(summary.get("max_owned_bytes_per_row", 0)),
            0,
        ),
        metadata_bytes_per_row=max(
            int(summary.get("metadata_bytes_per_row", 0)),
            0,
        ),
        composition=bool(composition),
        write_covering_bbox=bool(write_covering_bbox),
        ordered_parts=ordered_parts,
        planning_packet_scalars=max(int(planning_packet_scalars), 0),
    )


def _plan_native_wkb_capacity(
    *,
    owned,
    composition,
    max_chunk_rows: int,
    write_covering_bbox: bool,
) -> _NativeWkbCapacityPlan:
    """Plan terminal WKB allocations from native structure and live memory."""
    import cupy as cp

    from vibespatial.api._native_metadata import NativeGeometryMetadata

    if owned is not None:
        sources = ((0, int(owned.row_count), owned),)
        ordered_parts = None
        layout_checks = ()
        row_count = int(owned.row_count)
    else:
        ordered_parts, layout_checks = _candidate_ordered_composition_parts(
            composition
        )
        sources = (
            ordered_parts
            if ordered_parts is not None
            else tuple(
                (0, int(part.geometry.owned.row_count), part.geometry.owned)
                for part in composition.parts
                if part.geometry.owned is not None
            )
        )
        row_count = int(composition.row_count)

    pending_buffers = []
    seen_buffers: set[int] = set()
    packet_arrays = []
    for _start, _stop, source_owned in sources:
        state = source_owned._ensure_device_state(preserve_indexed_view=True)
        for family, buffer in state.families.items():
            buffer_key = id(buffer)
            if buffer_key in seen_buffers:
                continue
            seen_buffers.add(buffer_key)
            if _family_size_proof_complete(family, buffer.fixed_size):
                continue
            pending_buffers.append((family, buffer))
            packet_arrays.append(_device_family_size_packet(family, buffer))

    packet_arrays.extend(
        cp.asarray(check, dtype=cp.int64).reshape(1) for check in layout_checks
    )
    planning_packet_scalars = sum(int(values.size) for values in packet_arrays)
    packet_values: list[int] = []
    if packet_arrays:
        device_packet = cp.concatenate(
            [cp.asarray(values, dtype=cp.int64).reshape(-1) for values in packet_arrays]
        )
        packet_values = np.asarray(
            get_cuda_runtime().copy_device_to_host(
                device_packet,
                reason=(
                    "native WKB capacity and contiguous composition allocation packet"
                ),
            ),
            dtype=np.int64,
        ).tolist()

    cursor = 0
    for family, buffer in pending_buffers:
        first_maximum, second_maximum, coordinate_maximum = packet_values[
            cursor : cursor + 3
        ]
        cursor += 3
        existing = buffer.fixed_size
        buffer.fixed_size = DeviceFixedGeometrySizeMetadata(
            first_level_count_per_row=(
                None if existing is None else existing.first_level_count_per_row
            ),
            second_level_count_per_row=(
                None if existing is None else existing.second_level_count_per_row
            ),
            coord_count_per_row=(
                None if existing is None else existing.coord_count_per_row
            ),
            max_first_level_count_per_row=max(
                first_maximum,
                _fixed_or_maximum(existing, "first_level_count_per_row") or 0,
            ),
            max_second_level_count_per_row=max(
                second_maximum,
                _fixed_or_maximum(existing, "second_level_count_per_row") or 0,
            ),
            max_coord_count_per_row=max(
                coordinate_maximum,
                _fixed_or_maximum(existing, "coord_count_per_row") or 0,
            ),
        )
        buffer._device_size_bounds_exact = True

    if layout_checks:
        layout_values = packet_values[cursor : cursor + len(layout_checks)]
        if bool(np.all(layout_values)):
            object.__setattr__(composition, "contiguous_row_partitions", True)
        else:
            ordered_parts = None

    maximum_output_bytes = 0
    maximum_owned_bytes = 0
    maximum_metadata_bytes = 0
    coordinate_count = 0
    family_row_count = 0
    unique_structural_buffers: set[int] = set()
    for _start, _stop, source_owned in sources:
        state = source_owned._ensure_device_state(preserve_indexed_view=True)
        row_metadata_bytes = _device_row_metadata_bytes(state)
        maximum_metadata_bytes = max(maximum_metadata_bytes, row_metadata_bytes)
        source_output_bytes = 0
        source_owned_bytes = row_metadata_bytes
        for family, buffer in state.families.items():
            fixed_size = buffer.fixed_size
            first_count = _fixed_or_maximum(
                fixed_size,
                "first_level_count_per_row",
            ) or 0
            second_count = _fixed_or_maximum(
                fixed_size,
                "second_level_count_per_row",
            ) or 0
            family_coordinate_count = _fixed_or_maximum(
                fixed_size,
                "coord_count_per_row",
            ) or 0
            source_output_bytes = max(
                source_output_bytes,
                _native_wkb_family_output_bytes(
                    family,
                    first_count=first_count,
                    second_count=second_count,
                    coordinate_count=family_coordinate_count,
                ),
            )
            source_owned_bytes = max(
                source_owned_bytes,
                row_metadata_bytes
                + 16 * family_coordinate_count
                + 4 * (first_count + second_count + 1)
                + 1,
            )
            buffer_key = id(buffer)
            if buffer_key not in unique_structural_buffers:
                unique_structural_buffers.add(buffer_key)
                coordinate_count += int(buffer.x.size)
                family_row_count += max(int(buffer.geometry_offsets.size) - 1, 0)
        maximum_output_bytes = max(maximum_output_bytes, source_output_bytes)
        maximum_owned_bytes = max(maximum_owned_bytes, source_owned_bytes)

    geometry_view = SimpleNamespace(owned=owned, composition=composition)
    metadata = NativeGeometryMetadata.from_native_geometry(geometry_view)
    metadata = replace(
        metadata,
        shape_summary={
            **metadata.shape_summary,
            "physical_shape": "terminal-native-wkb-export",
            "coordinate_count": coordinate_count,
            "family_row_count": family_row_count,
            "max_wkb_output_bytes_per_row": maximum_output_bytes,
            "max_owned_bytes_per_row": maximum_owned_bytes,
            "metadata_bytes_per_row": maximum_metadata_bytes,
            "planning_packet_scalars": planning_packet_scalars,
            "output_row_count": row_count,
        },
    )
    return _native_wkb_capacity_plan_from_metadata(
        metadata,
        max_chunk_rows=max_chunk_rows,
        composition=composition is not None,
        write_covering_bbox=write_covering_bbox,
        ordered_parts=ordered_parts,
        planning_packet_scalars=planning_packet_scalars,
    )


def _iter_native_wkb_capacity_spans(row_count: int, capacity_plan):
    start = 0
    total_rows = max(int(row_count), 0)
    while start < total_rows:
        admitted_rows = capacity_plan.admitted_rows(total_rows - start)
        stop = start + admitted_rows
        yield start, stop
        start = stop


def _physicalize_native_wkb_composition_span(composition, start: int, stop: int):
    """Assemble one trusted singular logical span without host allocation reads."""
    import cupy as cp

    span_rows = max(int(stop) - int(start), 0)
    selected_parts = []
    selected_destinations = []
    selected_row_count = 0
    for part in composition.parts:
        part_owned = part.geometry.owned
        if part_owned is None:
            raise RuntimeError("native WKB composition part lost owned geometry")
        output_rows = cp.asarray(part.output_rows, dtype=cp.int64)
        selected_lanes = cp.flatnonzero(
            (output_rows >= cp.int64(start)) & (output_rows < cp.int64(stop))
        ).astype(cp.int64, copy=False)
        part_row_count = int(selected_lanes.size)
        if part_row_count == 0:
            continue
        selected = part_owned._device_indexed_take(
            selected_lanes,
            assume_unique_indices=True,
        ).physicalize_device_rows(allow_capacity_allocation=True)
        selected_parts.append(selected)
        selected_destinations.append(
            output_rows[selected_lanes] - cp.int64(start)
        )
        selected_row_count += part_row_count

    if selected_row_count > span_rows:
        raise RuntimeError("trusted singular WKB composition produced duplicate rows")

    arrays = []
    if selected_row_count < span_rows:
        arrays.append(build_null_owned_array(span_rows, residency=Residency.DEVICE))
        index_map = cp.arange(span_rows, dtype=cp.int64)
        source_offset = span_rows
    else:
        index_map = cp.zeros(span_rows, dtype=cp.int64)
        source_offset = 0

    for selected, destinations in zip(
        selected_parts,
        selected_destinations,
        strict=True,
    ):
        part_row_count = int(selected.row_count)
        index_map[destinations] = source_offset + cp.arange(
            part_row_count,
            dtype=cp.int64,
        )
        arrays.append(selected)
        source_offset += part_row_count

    root = OwnedGeometryArray.concat(arrays)
    return OwnedGeometryArray._indexed_view(
        root,
        index_map,
        assume_unique_indices=True,
    )


def _iter_bounded_native_wkb_owned_chunks(*, owned, composition, capacity_plan):
    """Yield capacity-admitted contiguous chunks in logical output order."""
    import cupy as cp

    if owned is not None:
        for start, stop in _iter_native_wkb_capacity_spans(
            owned.row_count,
            capacity_plan,
        ):
            d_rows = cp.arange(start, stop, dtype=cp.int64)
            yield start, stop, owned._device_indexed_take(
                d_rows,
                assume_unique_indices=True,
            )
        return

    ordered_parts = capacity_plan.ordered_parts
    if ordered_parts is None:
        for start, stop in _iter_native_wkb_capacity_spans(
            composition.row_count,
            capacity_plan,
        ):
            yield start, stop, _physicalize_native_wkb_composition_span(
                composition,
                start,
                stop,
            )
        return

    batch_start = 0
    batch_rows = 0
    batch_parts = []
    target_batch_rows = 0
    for _part_start, _part_stop, part_owned in ordered_parts:
        local_start = 0
        part_rows = int(part_owned.row_count)
        while local_start < part_rows:
            if batch_rows == 0:
                target_batch_rows = capacity_plan.admitted_rows(
                    int(composition.row_count) - batch_start
                )
            remaining = target_batch_rows - batch_rows
            take_rows = min(remaining, part_rows - local_start)
            local_stop = local_start + take_rows
            if local_start == 0 and local_stop == part_rows:
                selected_owned = (
                    part_owned.physicalize_device_rows(
                        allow_capacity_allocation=True
                    )
                    if part_owned.is_indexed_view
                    else part_owned
                )
            else:
                d_rows = cp.arange(local_start, local_stop, dtype=cp.int64)
                selected_owned = part_owned._device_indexed_take(
                    d_rows,
                    assume_unique_indices=True,
                ).physicalize_device_rows(allow_capacity_allocation=True)
            batch_parts.append(selected_owned)
            batch_rows += take_rows
            local_start = local_stop
            if batch_rows == target_batch_rows:
                chunk_owned = (
                    batch_parts[0]
                    if len(batch_parts) == 1
                    else OwnedGeometryArray.concat(batch_parts)
                )
                yield batch_start, batch_start + batch_rows, chunk_owned
                batch_start += batch_rows
                batch_rows = 0
                batch_parts = []

    if batch_rows:
        chunk_owned = (
            batch_parts[0]
            if len(batch_parts) == 1
            else OwnedGeometryArray.concat(batch_parts)
        )
        yield batch_start, batch_start + batch_rows, chunk_owned


def _write_geoparquet_native_device_wkb_chunks(
    plc,
    *,
    attribute_schema,
    device_attribute_columns,
    host_table,
    ordered_column_names,
    owned,
    composition,
    geometry_name,
    geometry_crs,
    schema_version,
    write_covering_bbox,
    frame_attrs,
    sink,
    compression,
    writer_kwargs,
    capacity_plan,
) -> None:
    """Encode and write capacity-admitted native WKB row groups on device."""
    import base64
    import json

    import cupy as cp
    import pyarrow as pa

    from vibespatial.api.io.arrow import _create_geometry_metadata, _encode_metadata
    from vibespatial.cuda._runtime import pylibcudf_current_stream

    requested_row_group_size = writer_kwargs.get("row_group_size")
    if requested_row_group_size is not None:
        requested_row_group_size = int(requested_row_group_size)
        if requested_row_group_size <= 0:
            raise ValueError("row_group_size must be greater than zero")
    normalized_geometry_crs = geometry_crs
    if normalized_geometry_crs is not None and not hasattr(
        normalized_geometry_crs,
        "to_json_dict",
    ):
        try:
            from pyproj import CRS

            normalized_geometry_crs = CRS.from_user_input(normalized_geometry_crs)
        except Exception:
            pass

    geometry_encoding_dict = {geometry_name: "WKB"}
    geometry_array = (
        DeviceGeometryArray._from_owned(owned, crs=normalized_geometry_crs)
        if owned is not None
        else DeviceGeometryArray._from_composition(
            composition,
            crs=normalized_geometry_crs,
        )
    )
    geometry_metadata_view = SimpleNamespace(
        array=geometry_array,
        crs=normalized_geometry_crs,
    )
    geo_metadata = _create_geometry_metadata(
        {geometry_name: geometry_metadata_view},
        primary_column=geometry_name,
        schema_version=schema_version,
        geometry_encoding=geometry_encoding_dict,
        write_covering_bbox=write_covering_bbox,
    )
    footer_metadata = {
        (key.decode() if isinstance(key, bytes) else str(key)): (
            value.decode() if isinstance(value, bytes) else str(value)
        )
        for key, value in (attribute_schema.metadata or {}).items()
    }
    footer_metadata["geo"] = _encode_metadata(geo_metadata).decode()
    if frame_attrs:
        footer_metadata["PANDAS_ATTRS"] = json.dumps(frame_attrs)

    geometry_field_metadata = {
        b"ARROW:extension:name": b"geoarrow.wkb",
        b"ARROW:extension:metadata": b"{}",
    }
    if normalized_geometry_crs is not None:
        try:
            crs_json = normalized_geometry_crs.to_json_dict()
        except AttributeError:
            crs_json = None
        if crs_json is not None:
            geometry_field_metadata[b"ARROW:extension:metadata"] = json.dumps(
                {"crs": crs_json}
            ).encode()

    bbox_column_names = ["bbox"] if write_covering_bbox else []
    all_column_names = list(ordered_column_names) + bbox_column_names
    host_fields = {field.name: field for field in attribute_schema}
    schema_fields = []
    for column_name in all_column_names:
        if column_name == geometry_name:
            schema_fields.append(
                pa.field(
                    geometry_name,
                    pa.binary(),
                    nullable=True,
                    metadata=geometry_field_metadata,
                )
            )
        elif column_name == "bbox":
            schema_fields.append(
                pa.field(
                    "bbox",
                    pa.struct(
                        [
                            pa.field("xmin", pa.float64(), nullable=False),
                            pa.field("ymin", pa.float64(), nullable=False),
                            pa.field("xmax", pa.float64(), nullable=False),
                            pa.field("ymax", pa.float64(), nullable=False),
                        ]
                    ),
                    nullable=True,
                )
            )
        else:
            schema_fields.append(host_fields[column_name])
    schema_metadata = dict(attribute_schema.metadata or {})
    schema_metadata[b"geo"] = _encode_metadata(geo_metadata)
    if frame_attrs:
        schema_metadata[b"PANDAS_ATTRS"] = json.dumps(frame_attrs).encode()
    arrow_schema = pa.schema(schema_fields, metadata=schema_metadata)
    footer_metadata["ARROW:schema"] = base64.b64encode(
        arrow_schema.serialize().to_pybytes()
    ).decode()

    attribute_names = [
        column_name
        for column_name in ordered_column_names
        if column_name != geometry_name
    ]
    if device_attribute_columns is not None:
        full_attribute_columns = [
            device_attribute_columns[column_name] for column_name in attribute_names
        ]
    else:
        full_attribute_columns = [
            _attribute_column_to_plc(host_table[column_name], column_name, plc=plc)
            for column_name in attribute_names
        ]
    full_attribute_table = (
        plc.Table(full_attribute_columns) if full_attribute_columns else None
    )
    completion_stream = cp.cuda.get_current_stream()
    stream = pylibcudf_current_stream(
        *([] if full_attribute_table is None else [full_attribute_table])
    )
    writer = None
    try:
        for start, stop, chunk_owned in _iter_bounded_native_wkb_owned_chunks(
            owned=owned,
            composition=composition,
            capacity_plan=capacity_plan,
        ):
            if full_attribute_table is None:
                chunk_attributes = {}
            else:
                sliced_attributes = plc.copying.slice(
                    full_attribute_table,
                    [start, stop],
                    stream=stream,
                )[0]
                chunk_attributes = dict(
                    zip(attribute_names, sliced_attributes.columns(), strict=True)
                )

            chunk_columns = []
            for column_name in ordered_column_names:
                if column_name == geometry_name:
                    chunk_columns.append(_encode_owned_wkb_column_device(chunk_owned))
                else:
                    chunk_columns.append(chunk_attributes[column_name])

            if write_covering_bbox:
                from vibespatial.kernels.core.geometry_analysis import (
                    compute_geometry_bounds_device,
                )

                bounds = cp.asarray(compute_geometry_bounds_device(chunk_owned))
                bbox_children = [
                    pylibcudf_column_from_device(cp.ascontiguousarray(bounds[:, index]))
                    for index in range(4)
                ]
                chunk_columns.append(plc.Column.struct_from_children(bbox_children))

            chunk_table = plc.Table(chunk_columns)
            if writer is None:
                metadata = plc.io.types.TableInputMetadata(chunk_table)
                for index, column_name in enumerate(all_column_names):
                    metadata.column_metadata[index].set_name(column_name)
                    if column_name == geometry_name:
                        metadata.column_metadata[index].set_output_as_binary(True)
                    elif column_name == "bbox":
                        for child_index, child_name in enumerate(
                            ("xmin", "ymin", "xmax", "ymax")
                        ):
                            metadata.column_metadata[index].child(child_index).set_name(
                                child_name
                            )
                builder = plc.io.parquet.ChunkedParquetWriterOptions.builder(
                    plc.io.types.SinkInfo([sink])
                )
                builder.metadata(metadata)
                builder.key_value_metadata([footer_metadata])
                builder.write_arrow_schema(False)
                builder.compression(_compression_type_from_name(compression))
                builder.row_group_size_rows(
                    requested_row_group_size or capacity_plan.max_chunk_rows
                )
                if "max_page_size" in writer_kwargs:
                    builder.max_page_size_bytes(int(writer_kwargs["max_page_size"]))
                writer = plc.io.parquet.ChunkedParquetWriter.from_options(
                    builder.build(),
                    stream=stream,
                )
            writer.write(chunk_table)
            # The chunk table owns the WKB payload and offset buffers consumed
            # asynchronously by libcudf.  Keep those owners alive until the
            # writer stream reaches this point instead of serializing the CUDA
            # context before assembling the next chunk.
            get_cuda_completion_retainer().defer(
                completion_stream,
                (chunk_table, tuple(chunk_columns), chunk_owned),
                lambda _owners: None,
            )
    finally:
        if writer is not None:
            writer.close([])


def _write_geoparquet_native_device_payload(
    attribute_frame,
    geometry_owned,
    path,
    *,
    geometry_composition=None,
    geometry_name,
    geometry_crs,
    index,
    compression,
    geometry_encoding,
    schema_version,
    write_covering_bbox,
    column_order,
    frame_attrs=None,
    index_plan=None,
    **kwargs,
) -> _NativeDeviceWriteStatus:
    import base64
    import json

    import pyarrow as pa

    try:
        import pylibcudf as plc
    except ModuleNotFoundError:
        plc = None

    from vibespatial.api.io._geoarrow import (
        _linestring_type,
        _multilinestring_type,
        _multipoint_type,
        _multipolygon_type,
        _polygon_type,
    )
    from vibespatial.api.io.arrow import _create_geometry_metadata, _encode_metadata
    from vibespatial.io.geoarrow import _geoarrow_field_metadata

    _RECOGNIZED_KWARGS = {"row_group_size", "max_page_size"}
    recognized_kwargs = {k: v for k, v in kwargs.items() if k in _RECOGNIZED_KWARGS}
    unrecognized_kwargs = {k: v for k, v in kwargs.items() if k not in _RECOGNIZED_KWARGS}
    owned = geometry_owned
    composition = geometry_composition
    if owned is None and composition is None:
        return _NativeDeviceWriteStatus(written=False)
    if owned is not None and (
        owned.residency is not Residency.DEVICE or owned.device_state is None
    ):
        return _NativeDeviceWriteStatus(written=False)
    if composition is not None and (
        composition.residency is not Residency.DEVICE
        or not composition.trusted_singular_rows
    ):
        return _NativeDeviceWriteStatus(written=False)
    if composition is not None and geometry_encoding.lower() != "wkb":
        return _NativeDeviceWriteStatus(
            written=False,
            compatibility_detail=(
                "partitioned native GeoParquet export currently requires WKB encoding"
            ),
        )
    if unrecognized_kwargs:
        return _NativeDeviceWriteStatus(
            written=False,
            compatibility_detail=(
                "native device GeoParquet payload writer does not support "
                f"kwargs={sorted(unrecognized_kwargs)}"
            ),
        )
    if not _native_parquet_compression_supported(compression):
        return _NativeDeviceWriteStatus(
            written=False,
            compatibility_detail=(
                "native device GeoParquet payload writer does not support "
                f"compression={compression!r}"
            ),
        )
    sink = _pylibcudf_sink(path)
    if sink is None:
        return _NativeDeviceWriteStatus(
            written=False,
            compatibility_detail=(
                "native device GeoParquet payload writer requires a filesystem path "
                "or Python IO sink"
            ),
        )
    if plc is None or not has_pylibcudf_support():
        return _NativeDeviceWriteStatus(
            written=False,
            fallback_detail=(
                "pylibcudf support is unavailable for the native device GeoParquet payload writer"
            ),
        )

    non_geometry_columns = [column for column in column_order if column != geometry_name]
    device_attribute_columns = None
    device_extra_columns: tuple[str, ...] = ()
    host_table = None
    attribute_schema = None
    device_export = _try_native_device_attribute_export(
        attribute_frame,
        non_geometry_columns,
        index=index,
        index_plan=index_plan,
        pa=pa,
        plc=plc,
    )
    if device_export is not None:
        device_attribute_columns, attribute_schema, device_extra_columns = device_export
    if device_attribute_columns is None:
        host_table = _build_native_host_attribute_table_from_frame(
            attribute_frame,
            non_geometry_columns,
            index=index,
            pa=pa,
        )
        attribute_schema = host_table.schema
    if index is False and attribute_schema.metadata:
        attribute_schema = attribute_schema.with_metadata(
            {key: value for key, value in attribute_schema.metadata.items() if key != b"pandas"}
            or None
        )
    if any(pa.types.is_dictionary(field.type) for field in attribute_schema):
        return _NativeDeviceWriteStatus(
            written=False,
            compatibility_detail=(
                "native device GeoParquet payload writer does not support "
                "dictionary/categorical attribute schema metadata"
            ),
        )
    ordered_column_names = list(column_order)
    if host_table is not None:
        for column_name in host_table.column_names:
            if column_name not in ordered_column_names:
                ordered_column_names.append(column_name)
    else:
        for column_name in device_extra_columns:
            if column_name not in ordered_column_names:
                ordered_column_names.append(column_name)
    requested_row_group_size = recognized_kwargs.get("row_group_size")
    if requested_row_group_size is not None:
        requested_row_group_size = int(requested_row_group_size)
        if requested_row_group_size <= 0:
            raise ValueError("row_group_size must be greater than zero")
    bounded_chunk_rows = min(
        int(requested_row_group_size or _NATIVE_DEVICE_PARQUET_CHUNK_ROWS),
        _NATIVE_DEVICE_PARQUET_CHUNK_ROWS,
    )
    wkb_capacity_plan = None
    capacity_chunk_rows = 0
    native_row_count = int(
        composition.row_count if composition is not None else owned.row_count
    )
    if geometry_encoding.lower() == "wkb":
        wkb_capacity_plan = _plan_native_wkb_capacity(
            owned=owned,
            composition=composition,
            max_chunk_rows=bounded_chunk_rows,
            write_covering_bbox=write_covering_bbox,
        )
        if native_row_count:
            capacity_chunk_rows = wkb_capacity_plan.admitted_rows(native_row_count)
    if (
        wkb_capacity_plan is not None
        and native_row_count > 0
        and (
            composition is not None
            or capacity_chunk_rows < native_row_count
        )
    ):
        _write_geoparquet_native_device_wkb_chunks(
            plc,
            attribute_schema=attribute_schema,
            device_attribute_columns=device_attribute_columns,
            host_table=host_table,
            ordered_column_names=ordered_column_names,
            owned=owned,
            composition=composition,
            geometry_name=geometry_name,
            geometry_crs=geometry_crs,
            schema_version=schema_version,
            write_covering_bbox=write_covering_bbox,
            frame_attrs=frame_attrs,
            sink=sink,
            compression=compression,
            writer_kwargs=recognized_kwargs,
            capacity_plan=wkb_capacity_plan,
        )
        record_dispatch_event(
            surface="vibespatial.io.geoparquet",
            operation="to_parquet",
            implementation="pylibcudf_chunked_device_wkb_parquet_writer",
            reason=(
                "native WKB encoding and Parquet compression stayed within "
                "structural and live-memory allocation capacity"
            ),
            detail=(
                f"rows={native_row_count}; "
                f"admitted_rows={capacity_chunk_rows}; "
                f"coordinates={wkb_capacity_plan.metadata.shape_summary['coordinate_count']}; "
                f"max_output_bytes_per_row={wkb_capacity_plan.max_output_bytes_per_row}; "
                f"max_owned_bytes_per_row={wkb_capacity_plan.max_owned_bytes_per_row}; "
                "workload_shape=capacity_planned_terminal_native_wkb_export"
            ),
            selected=ExecutionMode.GPU,
        )
        return _NativeDeviceWriteStatus(written=True)
    table_columns = []
    geometry_encoding_dict = {}

    for column_name in ordered_column_names:
        if column_name == geometry_name:
            if geometry_encoding.lower() == "geoarrow":
                fast_path_reason = _device_geoarrow_fast_path_reason_owned(owned)
                if fast_path_reason is None:
                    column, encoding_name = _encode_owned_geoarrow_column_device(owned)
                    table_columns.append(column)
                    geometry_encoding_dict[column_name] = encoding_name
                    continue
                record_fallback_event(
                    surface="geopandas.geodataframe.to_parquet",
                    reason=f"device-side GeoArrow fast path unavailable for column {column_name}; falling back to WKB",
                    detail=fast_path_reason,
                    selected=ExecutionMode.CPU,
                    pipeline="io/to_parquet",
                    d2h_transfer=True,
                )
            table_columns.append(_encode_owned_wkb_column_device(owned))
            geometry_encoding_dict[column_name] = "WKB"
        else:
            if device_attribute_columns is not None:
                table_columns.append(device_attribute_columns[column_name])
            else:
                table_columns.append(
                    _attribute_column_to_plc(host_table[column_name], column_name, plc=plc)
                )

    bbox_column_names: list[str] = []
    if write_covering_bbox:
        try:
            import cupy as _cp
        except ModuleNotFoundError:
            _cp = None
        if _cp is None:
            return _NativeDeviceWriteStatus(
                written=False,
                fallback_detail=(
                    "covering bbox export requires CuPy for the native device GeoParquet payload writer"
                ),
            )
        from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device

        bounds = _cp.asarray(compute_geometry_bounds_device(owned))
        d_xmin = _cp.ascontiguousarray(bounds[:, 0])
        d_ymin = _cp.ascontiguousarray(bounds[:, 1])
        d_xmax = _cp.ascontiguousarray(bounds[:, 2])
        d_ymax = _cp.ascontiguousarray(bounds[:, 3])
        bbox_children = [
            pylibcudf_column_from_device(d_xmin),
            pylibcudf_column_from_device(d_ymin),
            pylibcudf_column_from_device(d_xmax),
            pylibcudf_column_from_device(d_ymax),
        ]
        bbox_struct = plc.Column.struct_from_children(bbox_children)
        table_columns.append(bbox_struct)
        bbox_column_names = ["bbox"]

    all_column_names = ordered_column_names + bbox_column_names
    plc_table = plc.Table(table_columns)
    metadata = plc.io.types.TableInputMetadata(plc_table)
    for idx, column_name in enumerate(all_column_names):
        metadata.column_metadata[idx].set_name(column_name)
        if column_name == geometry_name:
            if geometry_encoding_dict[column_name] == "WKB":
                metadata.column_metadata[idx].set_output_as_binary(True)
            else:
                _apply_geoarrow_child_metadata(
                    metadata.column_metadata[idx],
                    _geoarrow_family_from_encoding(geometry_encoding_dict[column_name]),
                )
        elif column_name == "bbox":
            for child_idx, child_name in enumerate(("xmin", "ymin", "xmax", "ymax")):
                metadata.column_metadata[idx].child(child_idx).set_name(child_name)

    normalized_geometry_crs = geometry_crs
    if normalized_geometry_crs is not None and not hasattr(
        normalized_geometry_crs,
        "to_json_dict",
    ):
        try:
            from pyproj import CRS

            normalized_geometry_crs = CRS.from_user_input(normalized_geometry_crs)
        except Exception:
            pass

    geometry_metadata_view = SimpleNamespace(
        array=DeviceGeometryArray._from_owned(owned, crs=normalized_geometry_crs),
        crs=normalized_geometry_crs,
    )
    geo_metadata = _create_geometry_metadata(
        {geometry_name: geometry_metadata_view},
        primary_column=geometry_name,
        schema_version=schema_version,
        geometry_encoding=geometry_encoding_dict,
        write_covering_bbox=write_covering_bbox,
    )
    footer_metadata = {
        (key.decode() if isinstance(key, bytes) else str(key)): (
            value.decode() if isinstance(value, bytes) else str(value)
        )
        for key, value in (attribute_schema.metadata or {}).items()
    }
    footer_metadata["geo"] = _encode_metadata(geo_metadata).decode()
    if frame_attrs:
        footer_metadata["PANDAS_ATTRS"] = json.dumps(frame_attrs)

    point_type = pa.struct(
        [
            pa.field("x", pa.float64(), nullable=False),
            pa.field("y", pa.float64(), nullable=False),
        ]
    )

    def _geometry_field() -> pa.Field:
        if geometry_encoding_dict[geometry_name] == "WKB":
            field_metadata = {}
            if normalized_geometry_crs is not None:
                try:
                    crs_json = normalized_geometry_crs.to_json_dict()
                except AttributeError:
                    crs_json = None
                if crs_json is not None:
                    field_metadata[b"ARROW:extension:metadata"] = json.dumps(
                        {"crs": crs_json}
                    ).encode()
            field_metadata[b"ARROW:extension:name"] = b"geoarrow.wkb"
            if b"ARROW:extension:metadata" not in field_metadata:
                field_metadata[b"ARROW:extension:metadata"] = b"{}"
            return pa.field(
                geometry_name,
                pa.binary(),
                nullable=True,
                metadata=field_metadata,
            )

        family = _geoarrow_family_from_encoding(geometry_encoding_dict[geometry_name])
        if family is GeometryFamily.POINT:
            field_type = point_type
        elif family is GeometryFamily.LINESTRING:
            field_type = _linestring_type(point_type)
        elif family is GeometryFamily.POLYGON:
            field_type = _polygon_type(point_type)
        elif family is GeometryFamily.MULTIPOINT:
            field_type = _multipoint_type(point_type)
        elif family is GeometryFamily.MULTILINESTRING:
            field_type = _multilinestring_type(point_type)
        elif family is GeometryFamily.MULTIPOLYGON:
            field_type = _multipolygon_type(point_type)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported family for native GeoArrow schema: {family}")

        extension_name = f"geoarrow.{geometry_encoding_dict[geometry_name].lower()}"
        return pa.field(
            geometry_name,
            field_type,
            nullable=True,
            metadata=_geoarrow_field_metadata(
                extension_name=extension_name,
                crs=normalized_geometry_crs,
            ),
        )

    schema_fields = []
    host_fields = {field.name: field for field in attribute_schema}
    for column_name in all_column_names:
        if column_name == geometry_name:
            schema_fields.append(_geometry_field())
        elif column_name == "bbox":
            schema_fields.append(
                pa.field(
                    "bbox",
                    pa.struct(
                        [
                            pa.field("xmin", pa.float64(), nullable=False),
                            pa.field("ymin", pa.float64(), nullable=False),
                            pa.field("xmax", pa.float64(), nullable=False),
                            pa.field("ymax", pa.float64(), nullable=False),
                        ]
                    ),
                    nullable=True,
                )
            )
        else:
            schema_fields.append(host_fields[column_name])

    schema_metadata = dict(attribute_schema.metadata or {})
    schema_metadata[b"geo"] = _encode_metadata(geo_metadata)
    if frame_attrs:
        schema_metadata[b"PANDAS_ATTRS"] = json.dumps(frame_attrs).encode()
    arrow_schema = pa.schema(schema_fields, metadata=schema_metadata)
    footer_metadata["ARROW:schema"] = base64.b64encode(
        arrow_schema.serialize().to_pybytes()
    ).decode()

    _write_pylibcudf_parquet_table(
        plc,
        plc_table,
        sink=sink,
        metadata=metadata,
        footer_metadata=footer_metadata,
        compression=compression,
        writer_kwargs=recognized_kwargs,
    )
    record_dispatch_event(
        surface="vibespatial.io.geoparquet",
        operation="to_parquet",
        implementation="pylibcudf_device_parquet_writer",
        reason="device-side GeoParquet write via pylibcudf",
        selected=ExecutionMode.GPU,
    )
    return _NativeDeviceWriteStatus(written=True)


def _write_geoparquet_native_device(
    df,
    path,
    *,
    index,
    compression,
    geometry_encoding,
    schema_version,
    write_covering_bbox,
    geometry_columns,
    **kwargs,
) -> _NativeDeviceWriteStatus:
    import base64
    import json

    import pyarrow as pa

    try:
        import pylibcudf as plc
    except ModuleNotFoundError:
        plc = None

    from vibespatial.api.io._geoarrow import (
        _linestring_type,
        _multilinestring_type,
        _multipoint_type,
        _multipolygon_type,
        _polygon_type,
    )
    from vibespatial.api.io.arrow import _create_metadata, _encode_metadata
    from vibespatial.io.geoarrow import _geoarrow_field_metadata

    # Extract recognized kwargs; only fall back for truly unrecognized ones.
    _RECOGNIZED_KWARGS = {"row_group_size", "max_page_size"}
    recognized_kwargs = {k: v for k, v in kwargs.items() if k in _RECOGNIZED_KWARGS}
    unrecognized_kwargs = {k: v for k, v in kwargs.items() if k not in _RECOGNIZED_KWARGS}
    geometry_columns = list(geometry_columns)
    if not geometry_columns:
        return _NativeDeviceWriteStatus(written=False)

    owned_by_name: dict[Any, OwnedGeometryArray] = {}
    for col in geometry_columns:
        arr = df[col].array
        owned = (
            arr.to_owned() if isinstance(arr, DeviceGeometryArray) else getattr(arr, "_owned", None)
        )
        if owned is None:
            return _NativeDeviceWriteStatus(written=False)
        owned_by_name[col] = owned
    if not all(
        owned.residency is Residency.DEVICE and owned.device_state is not None
        for owned in owned_by_name.values()
    ):
        return _NativeDeviceWriteStatus(written=False)
    if unrecognized_kwargs:
        return _NativeDeviceWriteStatus(
            written=False,
            compatibility_detail=(
                "native device GeoParquet writer does not support "
                f"kwargs={sorted(unrecognized_kwargs)}"
            ),
        )
    if not _native_parquet_compression_supported(compression):
        return _NativeDeviceWriteStatus(
            written=False,
            compatibility_detail=(
                f"native device GeoParquet writer does not support compression={compression!r}"
            ),
        )
    sink = _pylibcudf_sink(path)
    if sink is None:
        return _NativeDeviceWriteStatus(
            written=False,
            compatibility_detail=(
                "native device GeoParquet writer requires a filesystem path or Python IO sink"
            ),
        )
    if plc is None or not has_pylibcudf_support():
        return _NativeDeviceWriteStatus(
            written=False,
            fallback_detail=(
                "pylibcudf support is unavailable for the native device GeoParquet writer"
            ),
        )

    geometry_columns_set = set(geometry_columns)
    non_geometry_columns = [
        column_name for column_name in df.columns if column_name not in geometry_columns_set
    ]
    host_table = _build_native_host_attribute_table(
        df,
        non_geometry_columns,
        index=index,
        pa=pa,
    )
    ordered_column_names = list(df.columns)
    for column_name in host_table.column_names:
        if column_name not in ordered_column_names:
            ordered_column_names.append(column_name)
    table_columns = []
    geometry_encoding_dict = {}

    for column_name in ordered_column_names:
        if column_name in geometry_columns_set:
            owned = owned_by_name[column_name]
            if geometry_encoding.lower() == "geoarrow":
                fast_path_reason = _device_geoarrow_fast_path_reason_owned(owned)
                if fast_path_reason is None:
                    column, encoding_name = _encode_owned_geoarrow_column_device(owned)
                    table_columns.append(column)
                    geometry_encoding_dict[column_name] = encoding_name
                    continue
                record_fallback_event(
                    surface="geopandas.geodataframe.to_parquet",
                    reason=f"device-side GeoArrow fast path unavailable for column {column_name}; falling back to WKB",
                    detail=fast_path_reason,
                    selected=ExecutionMode.CPU,
                    pipeline="io/to_parquet",
                    d2h_transfer=True,
                )
            table_columns.append(_encode_owned_wkb_column_device(owned))
            geometry_encoding_dict[column_name] = "WKB"
        else:
            table_columns.append(
                _attribute_column_to_plc(host_table[column_name], column_name, plc=plc)
            )

    # If write_covering_bbox, compute per-row bounds on device and add a
    # struct column with xmin/ymin/xmax/ymax children.
    bbox_column_names: list[str] = []
    if write_covering_bbox:
        try:
            import cupy as _cp
        except ModuleNotFoundError:
            _cp = None
        if _cp is None:
            # cupy unavailable -- fall back to host path so the bbox column
            # and covering metadata stay consistent.
            return _NativeDeviceWriteStatus(
                written=False,
                fallback_detail=(
                    "covering bbox export requires CuPy for the native device GeoParquet writer"
                ),
            )
        from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device

        # Use the primary geometry column (first) for the covering bbox.
        primary_owned = owned_by_name[geometry_columns[0]]
        bounds = _cp.asarray(compute_geometry_bounds_device(primary_owned))
        d_xmin = _cp.ascontiguousarray(bounds[:, 0])
        d_ymin = _cp.ascontiguousarray(bounds[:, 1])
        d_xmax = _cp.ascontiguousarray(bounds[:, 2])
        d_ymax = _cp.ascontiguousarray(bounds[:, 3])
        bbox_children = [
            pylibcudf_column_from_device(d_xmin),
            pylibcudf_column_from_device(d_ymin),
            pylibcudf_column_from_device(d_xmax),
            pylibcudf_column_from_device(d_ymax),
        ]
        bbox_struct = plc.Column.struct_from_children(bbox_children)
        table_columns.append(bbox_struct)
        bbox_column_names = ["bbox"]

    all_column_names = ordered_column_names + bbox_column_names
    plc_table = plc.Table(table_columns)
    metadata = plc.io.types.TableInputMetadata(plc_table)
    for idx, column_name in enumerate(all_column_names):
        metadata.column_metadata[idx].set_name(column_name)
        if column_name in geometry_columns_set:
            if geometry_encoding_dict[column_name] == "WKB":
                metadata.column_metadata[idx].set_output_as_binary(True)
            else:
                _apply_geoarrow_child_metadata(
                    metadata.column_metadata[idx],
                    _geoarrow_family_from_encoding(geometry_encoding_dict[column_name]),
                )
        elif column_name == "bbox":
            # Set child names for the struct children: xmin, ymin, xmax, ymax
            for child_idx, child_name in enumerate(("xmin", "ymin", "xmax", "ymax")):
                metadata.column_metadata[idx].child(child_idx).set_name(child_name)

    geo_metadata = _create_metadata(
        df,
        schema_version=schema_version,
        geometry_encoding=geometry_encoding_dict,
        write_covering_bbox=write_covering_bbox,
    )
    footer_metadata = {
        (key.decode() if isinstance(key, bytes) else str(key)): (
            value.decode() if isinstance(value, bytes) else str(value)
        )
        for key, value in (host_table.schema.metadata or {}).items()
    }
    footer_metadata["geo"] = _encode_metadata(geo_metadata).decode()
    if df.attrs:
        footer_metadata["PANDAS_ATTRS"] = json.dumps(df.attrs)

    point_type = pa.struct(
        [
            pa.field("x", pa.float64(), nullable=False),
            pa.field("y", pa.float64(), nullable=False),
        ]
    )

    def _geometry_field(column_name: str) -> pa.Field:
        series = df[column_name]
        if geometry_encoding_dict[column_name] == "WKB":
            field_metadata = {}
            if series.crs is not None:
                try:
                    crs_json = series.crs.to_json_dict()
                except AttributeError:
                    crs_json = None
                if crs_json is not None:
                    field_metadata[b"ARROW:extension:metadata"] = json.dumps(
                        {"crs": crs_json}
                    ).encode()
            field_metadata[b"ARROW:extension:name"] = b"geoarrow.wkb"
            if b"ARROW:extension:metadata" not in field_metadata:
                field_metadata[b"ARROW:extension:metadata"] = b"{}"
            return pa.field(
                column_name,
                pa.binary(),
                nullable=True,
                metadata=field_metadata,
            )

        family = _geoarrow_family_from_encoding(geometry_encoding_dict[column_name])
        if family is GeometryFamily.POINT:
            field_type = point_type
        elif family is GeometryFamily.LINESTRING:
            field_type = _linestring_type(point_type)
        elif family is GeometryFamily.POLYGON:
            field_type = _polygon_type(point_type)
        elif family is GeometryFamily.MULTIPOINT:
            field_type = _multipoint_type(point_type)
        elif family is GeometryFamily.MULTILINESTRING:
            field_type = _multilinestring_type(point_type)
        elif family is GeometryFamily.MULTIPOLYGON:
            field_type = _multipolygon_type(point_type)
        else:  # pragma: no cover - exhaustive today
            raise ValueError(f"Unsupported family for native GeoArrow schema: {family}")

        extension_name = f"geoarrow.{geometry_encoding_dict[column_name].lower()}"
        return pa.field(
            column_name,
            field_type,
            nullable=True,
            metadata=_geoarrow_field_metadata(
                extension_name=extension_name,
                crs=series.crs,
            ),
        )

    schema_fields = []
    host_fields = {field.name: field for field in host_table.schema}
    for column_name in all_column_names:
        if column_name in geometry_columns_set:
            schema_fields.append(_geometry_field(column_name))
        elif column_name == "bbox":
            schema_fields.append(
                pa.field(
                    "bbox",
                    pa.struct(
                        [
                            pa.field("xmin", pa.float64(), nullable=False),
                            pa.field("ymin", pa.float64(), nullable=False),
                            pa.field("xmax", pa.float64(), nullable=False),
                            pa.field("ymax", pa.float64(), nullable=False),
                        ]
                    ),
                    nullable=True,
                )
            )
        else:
            schema_fields.append(host_fields[column_name])

    schema_metadata = dict(host_table.schema.metadata or {})
    schema_metadata[b"geo"] = _encode_metadata(geo_metadata)
    if df.attrs:
        schema_metadata[b"PANDAS_ATTRS"] = json.dumps(df.attrs).encode()
    arrow_schema = pa.schema(schema_fields, metadata=schema_metadata)
    footer_metadata["ARROW:schema"] = base64.b64encode(
        arrow_schema.serialize().to_pybytes()
    ).decode()

    _write_pylibcudf_parquet_table(
        plc,
        plc_table,
        sink=sink,
        metadata=metadata,
        footer_metadata=footer_metadata,
        compression=compression,
        writer_kwargs=recognized_kwargs,
    )
    record_dispatch_event(
        surface="vibespatial.io.geoparquet",
        operation="to_parquet",
        implementation="pylibcudf_device_parquet_writer",
        reason="device-side GeoParquet write via pylibcudf",
        selected=ExecutionMode.GPU,
    )
    return _NativeDeviceWriteStatus(written=True)


@dataclass(frozen=True)
class WKBBridgePlan:
    operation: IOOperation
    selected_path: IOPathKind
    canonical_gpu: bool
    device_codec_available: bool
    reason: str


@dataclass(frozen=True)
class WKBPartitionPlan:
    total_rows: int
    valid_rows: int
    null_rows: int
    native_rows: int
    fallback_rows: int
    family_counts: dict[str, int]
    fallback_indexes: tuple[int, ...]
    fallback_reason_counts: dict[str, int]
    reason: str


@dataclass(frozen=True)
class DeviceWKBHeaderScan:
    row_count: int
    valid_count: int
    native_count: int
    fallback_count: int
    validity: Any
    type_ids: Any
    family_tags: Any
    native_mask: Any
    fallback_mask: Any
    point_mask: Any


def plan_wkb_bridge(operation: IOOperation | str) -> WKBBridgePlan:
    normalized = operation if isinstance(operation, IOOperation) else IOOperation(operation)
    plan = plan_io_support(IOFormat.WKB, normalized)
    return WKBBridgePlan(
        operation=normalized,
        selected_path=plan.selected_path,
        canonical_gpu=plan.canonical_gpu,
        device_codec_available=True,
        reason=(
            "WKB remains a compatibility bridge; use staged header scans, family partitions, "
            "output-size scans, and family-local decode or encode today so the same contract "
            "can map onto CCCL primitives on device later."
        ),
    )


def _new_wkb_family_state() -> dict[str, Any]:
    return {
        "row_count": 0,
        "empty_mask": [],
        "geometry_offsets": [],
        "x_payload": [],
        "y_payload": [],
        "part_offsets": [],
        "ring_offsets": [],
    }


def _finalize_wkb_family_buffer(
    family: GeometryFamily, state: dict[str, Any]
) -> FamilyGeometryBuffer:
    x = np.asarray(state["x_payload"], dtype=np.float64)
    y = np.asarray(state["y_payload"], dtype=np.float64)
    geometry_offsets = np.asarray(
        [*state["geometry_offsets"], len(state["x_payload"])], dtype=np.int32
    )
    part_offsets = None
    ring_offsets = None

    if family is GeometryFamily.POLYGON:
        ring_offsets = np.asarray([*state["ring_offsets"], len(state["x_payload"])], dtype=np.int32)
        geometry_offsets = np.asarray(
            [*state["geometry_offsets"], len(state["ring_offsets"])], dtype=np.int32
        )
    elif family is GeometryFamily.MULTILINESTRING:
        part_offsets = np.asarray([*state["part_offsets"], len(state["x_payload"])], dtype=np.int32)
        geometry_offsets = np.asarray(
            [*state["geometry_offsets"], len(state["part_offsets"])], dtype=np.int32
        )
    elif family is GeometryFamily.MULTIPOLYGON:
        part_offsets = np.asarray(
            [*state["part_offsets"], len(state["ring_offsets"])], dtype=np.int32
        )
        ring_offsets = np.asarray([*state["ring_offsets"], len(state["x_payload"])], dtype=np.int32)
        geometry_offsets = np.asarray(
            [*state["geometry_offsets"], len(state["part_offsets"])], dtype=np.int32
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


def _normalize_wkb_value(value: bytes | str | None) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, str):
        return bytes.fromhex(value)
    return value


def _prepare_native_wkb_list_for_device(
    values: list[bytes | str | None] | tuple[bytes | str | None, ...],
) -> tuple[np.ndarray, np.ndarray] | None:
    row_count = len(values)
    lengths = np.empty(row_count, dtype=np.int32)
    payload_parts: list[bytes] = []

    for row_index, value in enumerate(values):
        normalized = _normalize_wkb_value(value)
        if normalized is None:
            lengths[row_index] = 0
            continue
        family, reason = _scan_wkb_value(normalized)
        if reason is not None or family is None:
            return None
        lengths[row_index] = len(normalized)
        payload_parts.append(normalized)

    offsets = np.empty(row_count + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    payload_size = int(offsets[-1])
    if payload_size == 0:
        payload = np.empty(0, dtype=np.uint8)
    else:
        payload = np.frombuffer(
            b"".join(payload_parts),
            dtype=np.uint8,
        )
    return offsets, payload


def _non_null_wkb_input_mask(
    values: list[bytes | str | None] | tuple[bytes | str | None, ...],
) -> np.ndarray:
    return np.asarray(
        [_normalize_wkb_value(value) is not None for value in values],
        dtype=bool,
    )


def _raise_on_invalid_gpu_wkb_decode(
    result: OwnedGeometryArray,
    non_null_mask: np.ndarray,
) -> None:
    validity, _tags, _family_row_offsets = _authoritative_host_metadata(result)
    invalid_rows = np.flatnonzero(non_null_mask & ~np.asarray(validity, dtype=bool))
    if invalid_rows.size == 0:
        return
    first_row = int(invalid_rows[0])
    raise _GpuWkbOnInvalidError(
        f"Invalid WKB geometry encountered during GPU decode at row {first_row}"
    )


def _scan_wkb_value(value: bytes) -> tuple[GeometryFamily | None, str | None]:
    if len(value) < 5:
        return None, "buffer shorter than WKB header"
    byteorder = value[0]
    if byteorder not in {0, 1}:
        return None, f"unsupported WKB byte-order flag {byteorder}"
    byteorder_name = "little" if byteorder == 1 else "big"
    type_id = int.from_bytes(value[1:5], byteorder_name)
    ewkb_z = bool(type_id & 0x80000000)
    ewkb_m = bool(type_id & 0x40000000)
    ewkb_srid = bool(type_id & 0x20000000)
    ewkb_base_type = type_id & 0x1FFFFFFF
    iso_dimension_variant = type_id // 1000 if 1000 <= type_id < 4000 else 0
    iso_base_type = type_id % 1000 if iso_dimension_variant else type_id
    if ewkb_z or ewkb_m or iso_dimension_variant:
        return None, "Z/M/ZM WKB rows fall outside the 2D owned native result model"
    candidate_type_id = ewkb_base_type if ewkb_srid else iso_base_type
    if candidate_type_id == 7:
        return None, "GeometryCollection rows fall outside the 2D owned native result model"
    if ewkb_srid and candidate_type_id in WKB_ID_FAMILIES:
        return None, "EWKB SRID-annotated 2D input routes through the explicit compatibility bridge"
    if byteorder != 1:
        return None, "big-endian 2D WKB input routes through the explicit compatibility bridge"
    family = WKB_ID_FAMILIES.get(candidate_type_id)
    if family is None:
        return None, f"unsupported WKB geometry type id {type_id}"
    return family, None


def _scan_wkb_partition_normalized(
    values: list[bytes | None] | tuple[bytes | None, ...],
) -> tuple[tuple[tuple[GeometryFamily | None, str | None], ...], WKBPartitionPlan]:
    family_counts = {family.value: 0 for family in GeometryFamily}
    fallback_indexes: list[int] = []
    fallback_reason_counts: dict[str, int] = {}
    scan_results: list[tuple[GeometryFamily | None, str | None]] = []
    null_rows = 0
    valid_rows = 0
    native_rows = 0
    for index, value in enumerate(values):
        if value is None:
            null_rows += 1
            scan_results.append((None, None))
            continue
        valid_rows += 1
        family, reason = _scan_wkb_value(value)
        scan_results.append((family, reason))
        if reason is not None or family is None:
            fallback_indexes.append(index)
            fallback_reason_counts[reason or "unknown WKB fallback reason"] = (
                fallback_reason_counts.get(reason or "unknown WKB fallback reason", 0) + 1
            )
            continue
        family_counts[family.value] += 1
        native_rows += 1
    return (
        tuple(scan_results),
        WKBPartitionPlan(
            total_rows=len(values),
            valid_rows=valid_rows,
            null_rows=null_rows,
            native_rows=native_rows,
            fallback_rows=len(fallback_indexes),
            family_counts=family_counts,
            fallback_indexes=tuple(fallback_indexes),
            fallback_reason_counts=fallback_reason_counts,
            reason=(
                "Use one WKB header scan to separate native little-endian 2D families from the "
                "explicit fallback pool before decode or encode work begins."
            ),
        ),
    )


def plan_wkb_partition(
    values: list[bytes | str | None] | tuple[bytes | str | None, ...],
) -> WKBPartitionPlan:
    normalized_values = tuple(_normalize_wkb_value(value) for value in values)
    _scan_results, partition_plan = _scan_wkb_partition_normalized(normalized_values)
    return partition_plan


def _format_wkb_fallback_reason_counts(partition_plan: WKBPartitionPlan) -> str:
    if not partition_plan.fallback_reason_counts:
        return ""
    ordered_reasons = sorted(
        partition_plan.fallback_reason_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return "; ".join(f"{count}x {reason}" for reason, count in ordered_reasons)


def _geometry_family_from_shapely_type(geom_type: str) -> GeometryFamily:
    family = {
        "Point": GeometryFamily.POINT,
        "LineString": GeometryFamily.LINESTRING,
        "Polygon": GeometryFamily.POLYGON,
        "MultiPoint": GeometryFamily.MULTIPOINT,
        "MultiLineString": GeometryFamily.MULTILINESTRING,
        "MultiPolygon": GeometryFamily.MULTIPOLYGON,
    }.get(geom_type)
    if family is None:
        raise NotImplementedError(f"{geom_type} rows fall outside the 2D owned native result model")
    return family


def _append_point_row(state: dict[str, Any], x: float, y: float, *, empty: bool) -> int:
    row = int(state["row_count"])
    state["row_count"] += 1
    state["empty_mask"].append(empty)
    state["geometry_offsets"].append(len(state["x_payload"]))
    if not empty:
        state["x_payload"].append(x)
        state["y_payload"].append(y)
    return row


def _append_coordinate_range(
    state: dict[str, Any],
    coords: np.ndarray,
) -> None:
    if coords.size == 0:
        return
    state["x_payload"].extend(coords[:, 0].tolist())
    state["y_payload"].extend(coords[:, 1].tolist())


def _append_shapely_geometry_state(
    family: GeometryFamily,
    geometry: Any,
    state: dict[str, Any],
) -> int:
    local_row = int(state["row_count"])
    state["row_count"] += 1
    state["empty_mask"].append(bool(geometry.is_empty))
    if family is GeometryFamily.POINT:
        state["geometry_offsets"].append(len(state["x_payload"]))
        if not geometry.is_empty:
            state["x_payload"].append(float(geometry.x))
            state["y_payload"].append(float(geometry.y))
    elif family in {GeometryFamily.LINESTRING, GeometryFamily.MULTIPOINT}:
        state["geometry_offsets"].append(len(state["x_payload"]))
        coords = (
            list(geometry.coords)
            if family is GeometryFamily.LINESTRING
            else [(float(p.x), float(p.y)) for p in iter_geometry_parts(geometry)]
        )
        state["x_payload"].extend([float(x) for x, _ in coords])
        state["y_payload"].extend([float(y) for _, y in coords])
    elif family is GeometryFamily.POLYGON:
        state["geometry_offsets"].append(len(state["ring_offsets"]))
        if not geometry.is_empty:
            rings = [geometry.exterior, *geometry.interiors]
            for ring in rings:
                state["ring_offsets"].append(len(state["x_payload"]))
                coords = list(ring.coords)
                state["x_payload"].extend([float(x) for x, _ in coords])
                state["y_payload"].extend([float(y) for _, y in coords])
    elif family is GeometryFamily.MULTILINESTRING:
        state["geometry_offsets"].append(len(state["part_offsets"]))
        if not geometry.is_empty:
            for part in iter_geometry_parts(geometry):
                state["part_offsets"].append(len(state["x_payload"]))
                coords = list(part.coords)
                state["x_payload"].extend([float(x) for x, _ in coords])
                state["y_payload"].extend([float(y) for _, y in coords])
    elif family is GeometryFamily.MULTIPOLYGON:
        state["geometry_offsets"].append(len(state["part_offsets"]))
        if not geometry.is_empty:
            for polygon in iter_geometry_parts(geometry):
                state["part_offsets"].append(len(state["ring_offsets"]))
                rings = [polygon.exterior, *polygon.interiors]
                for ring in rings:
                    state["ring_offsets"].append(len(state["x_payload"]))
                    coords = list(ring.coords)
                    state["x_payload"].extend([float(x) for x, _ in coords])
                    state["y_payload"].extend([float(y) for _, y in coords])
    return local_row


def _decode_linestring_wkb_payload(value: bytes, state: dict[str, Any]) -> int:
    if len(value) < 9:
        raise ValueError("buffer shorter than LineString header")
    count = int.from_bytes(value[5:9], "little")
    expected = 9 + (count * 16)
    if len(value) != expected:
        raise ValueError("LineString buffer length does not match point count")
    row = int(state["row_count"])
    state["row_count"] += 1
    state["empty_mask"].append(count == 0)
    state["geometry_offsets"].append(len(state["x_payload"]))
    if count:
        coords = np.frombuffer(value, dtype="<f8", offset=9, count=count * 2).reshape(count, 2)
        _append_coordinate_range(state, coords)
    return row


def _decode_polygon_wkb_payload(value: bytes, state: dict[str, Any]) -> int:
    if len(value) < 9:
        raise ValueError("buffer shorter than Polygon header")
    ring_count = int.from_bytes(value[5:9], "little")
    cursor = 9
    row = int(state["row_count"])
    state["row_count"] += 1
    state["empty_mask"].append(ring_count == 0)
    state["geometry_offsets"].append(len(state["ring_offsets"]))
    for _ in range(ring_count):
        if cursor + 4 > len(value):
            raise ValueError("Polygon ring header overruns buffer")
        point_count = int.from_bytes(value[cursor : cursor + 4], "little")
        cursor += 4
        coord_bytes = point_count * 16
        if cursor + coord_bytes > len(value):
            raise ValueError("Polygon ring coordinates overrun buffer")
        state["ring_offsets"].append(len(state["x_payload"]))
        if point_count:
            coords = np.frombuffer(
                value, dtype="<f8", offset=cursor, count=point_count * 2
            ).reshape(point_count, 2)
            _append_coordinate_range(state, coords)
        cursor += coord_bytes
    if cursor != len(value):
        raise ValueError("Polygon buffer has trailing bytes")
    return row


def _decode_multipoint_wkb_payload(value: bytes, state: dict[str, Any]) -> int:
    if len(value) < 9:
        raise ValueError("buffer shorter than MultiPoint header")
    point_count = int.from_bytes(value[5:9], "little")
    cursor = 9
    row = int(state["row_count"])
    state["row_count"] += 1
    state["empty_mask"].append(point_count == 0)
    state["geometry_offsets"].append(len(state["x_payload"]))
    for _ in range(point_count):
        if cursor + 21 > len(value):
            raise ValueError("MultiPoint point record overruns buffer")
        point_record = value[cursor : cursor + 21]
        if point_record[0] != 1 or int.from_bytes(point_record[1:5], "little") != 1:
            raise ValueError("MultiPoint fast path requires nested little-endian point records")
        point_data = np.frombuffer(point_record, dtype=WKB_POINT_RECORD_DTYPE, count=1)
        x = float(point_data["x"][0])
        y = float(point_data["y"][0])
        if not (np.isnan(x) or np.isnan(y)):
            state["x_payload"].append(x)
            state["y_payload"].append(y)
        cursor += 21
    if cursor != len(value):
        raise ValueError("MultiPoint buffer has trailing bytes")
    return row


def _decode_multilinestring_wkb_payload(value: bytes, state: dict[str, Any]) -> int:
    if len(value) < 9:
        raise ValueError("buffer shorter than MultiLineString header")
    part_count = int.from_bytes(value[5:9], "little")
    cursor = 9
    row = int(state["row_count"])
    state["row_count"] += 1
    state["empty_mask"].append(part_count == 0)
    state["geometry_offsets"].append(len(state["part_offsets"]))
    for _ in range(part_count):
        if cursor + 9 > len(value):
            raise ValueError("MultiLineString part header overruns buffer")
        if value[cursor] != 1 or int.from_bytes(value[cursor + 1 : cursor + 5], "little") != 2:
            raise ValueError(
                "MultiLineString fast path requires nested little-endian linestring records"
            )
        point_count = int.from_bytes(value[cursor + 5 : cursor + 9], "little")
        coord_bytes = point_count * 16
        end = cursor + 9 + coord_bytes
        if end > len(value):
            raise ValueError("MultiLineString part coordinates overrun buffer")
        state["part_offsets"].append(len(state["x_payload"]))
        if point_count:
            coords = np.frombuffer(
                value, dtype="<f8", offset=cursor + 9, count=point_count * 2
            ).reshape(point_count, 2)
            _append_coordinate_range(state, coords)
        cursor = end
    if cursor != len(value):
        raise ValueError("MultiLineString buffer has trailing bytes")
    return row


def _decode_multipolygon_wkb_payload(value: bytes, state: dict[str, Any]) -> int:
    if len(value) < 9:
        raise ValueError("buffer shorter than MultiPolygon header")
    polygon_count = int.from_bytes(value[5:9], "little")
    cursor = 9
    row = int(state["row_count"])
    state["row_count"] += 1
    state["empty_mask"].append(polygon_count == 0)
    state["geometry_offsets"].append(len(state["part_offsets"]))
    for _ in range(polygon_count):
        if cursor + 9 > len(value):
            raise ValueError("MultiPolygon polygon header overruns buffer")
        if value[cursor] != 1 or int.from_bytes(value[cursor + 1 : cursor + 5], "little") != 3:
            raise ValueError("MultiPolygon fast path requires nested little-endian polygon records")
        ring_count = int.from_bytes(value[cursor + 5 : cursor + 9], "little")
        cursor += 9
        state["part_offsets"].append(len(state["ring_offsets"]))
        for _ in range(ring_count):
            if cursor + 4 > len(value):
                raise ValueError("MultiPolygon ring header overruns buffer")
            point_count = int.from_bytes(value[cursor : cursor + 4], "little")
            cursor += 4
            coord_bytes = point_count * 16
            if cursor + coord_bytes > len(value):
                raise ValueError("MultiPolygon ring coordinates overrun buffer")
            state["ring_offsets"].append(len(state["x_payload"]))
            if point_count:
                coords = np.frombuffer(
                    value, dtype="<f8", offset=cursor, count=point_count * 2
                ).reshape(point_count, 2)
                _append_coordinate_range(state, coords)
            cursor += coord_bytes
    if cursor != len(value):
        raise ValueError("MultiPolygon buffer has trailing bytes")
    return row


def _decode_point_batch(values: list[bytes], state: dict[str, Any]) -> list[int]:
    if not values:
        return []
    payload = b"".join(values)
    records = np.frombuffer(payload, dtype=WKB_POINT_RECORD_DTYPE)
    x = np.asarray(records["x"], dtype=np.float64)
    y = np.asarray(records["y"], dtype=np.float64)
    # Preserve partial-NaN point coordinates exactly as encoded. Only the
    # canonical NaN/NaN sentinel represents POINT EMPTY in WKB.
    empty_mask = np.isnan(x) & np.isnan(y)
    nonempty = ~empty_mask
    start = len(state["x_payload"])
    coord_starts = start + np.cumsum(nonempty, dtype=np.int32) - nonempty.astype(np.int32)
    local_start = int(state["row_count"])
    row_count = int(records.shape[0])
    state["row_count"] += row_count
    state["empty_mask"].extend(empty_mask.tolist())
    state["geometry_offsets"].extend(coord_starts.tolist())
    if bool(nonempty.any()):
        state["x_payload"].extend(x[nonempty].tolist())
        state["y_payload"].extend(y[nonempty].tolist())
    return list(range(local_start, local_start + row_count))


def _decode_native_wkb(
    values: list[bytes | str | None] | tuple[bytes | str | None, ...],
    *,
    on_invalid: str = "raise",
) -> tuple[OwnedGeometryArray, WKBPartitionPlan]:
    normalized_values = [_normalize_wkb_value(value) for value in values]
    scan_results, partition_plan = _scan_wkb_partition_normalized(normalized_values)
    validity = np.asarray([value is not None for value in normalized_values], dtype=bool)
    tags = np.full(len(normalized_values), -1, dtype=np.int8)
    family_row_offsets = np.full(len(normalized_values), -1, dtype=np.int32)
    states = {family: _new_wkb_family_state() for family in GeometryFamily}
    fallback_rows: list[int] = []
    fallback_values: list[bytes] = []
    point_rows: list[int] = []
    point_values: list[bytes] = []

    for row_index, value in enumerate(normalized_values):
        if value is None:
            continue
        family, scan_reason = scan_results[row_index]
        if scan_reason is not None or family is None:
            fallback_rows.append(row_index)
            fallback_values.append(value)
            continue
        try:
            if family is GeometryFamily.POINT:
                point_rows.append(row_index)
                point_values.append(value)
                continue
            if family is GeometryFamily.LINESTRING:
                local_row = _decode_linestring_wkb_payload(value, states[family])
            elif family is GeometryFamily.POLYGON:
                local_row = _decode_polygon_wkb_payload(value, states[family])
            elif family is GeometryFamily.MULTIPOINT:
                local_row = _decode_multipoint_wkb_payload(value, states[family])
            elif family is GeometryFamily.MULTILINESTRING:
                local_row = _decode_multilinestring_wkb_payload(value, states[family])
            elif family is GeometryFamily.MULTIPOLYGON:
                local_row = _decode_multipolygon_wkb_payload(value, states[family])
            else:
                raise ValueError(f"unsupported WKB family {family.value}")
            tags[row_index] = FAMILY_TAGS[family]
            family_row_offsets[row_index] = local_row
        except Exception:
            if on_invalid == "raise":
                raise
            fallback_rows.append(row_index)
            fallback_values.append(value)

    if point_values:
        point_locals = _decode_point_batch(point_values, states[GeometryFamily.POINT])
        for row_index, local_row in zip(point_rows, point_locals, strict=True):
            tags[row_index] = FAMILY_TAGS[GeometryFamily.POINT]
            family_row_offsets[row_index] = local_row

    if fallback_rows:
        fallback_owned = from_wkb(fallback_values, on_invalid=on_invalid)
        for row_index, geometry in zip(fallback_rows, fallback_owned.to_shapely(), strict=True):
            if geometry is None:
                validity[row_index] = False
                continue
            family = _geometry_family_from_shapely_type(geometry.geom_type)
            local_row = _append_shapely_geometry_state(family, geometry, states[family])
            tags[row_index] = FAMILY_TAGS[family]
            family_row_offsets[row_index] = local_row

    families = {
        family: _finalize_wkb_family_buffer(family, state)
        for family, state in states.items()
        if state["row_count"] > 0
    }
    array = OwnedGeometryArray(
        validity=validity,
        tags=tags.astype(np.int8, copy=False),
        family_row_offsets=family_row_offsets,
        families=families,
    )
    array._record(
        DiagnosticKind.CREATED,
        "created owned geometry array from staged native WKB decode",
        visible=True,
    )
    return array, partition_plan


def _decode_arrow_wkb_point_fast(array) -> OwnedGeometryArray | None:
    validity = np.asarray(array.is_valid().to_numpy(zero_copy_only=False), dtype=bool)
    row_count = int(validity.size)
    tags = np.full(row_count, -1, dtype=np.int8)
    family_row_offsets = np.full(row_count, -1, dtype=np.int32)
    valid_count = int(validity.sum())
    if valid_count == 0:
        return OwnedGeometryArray(
            validity=validity,
            tags=tags,
            family_row_offsets=family_row_offsets,
            families={},
        )

    offset_dtype = np.int64 if "large_binary" in str(array.type) else np.int32
    offsets = np.frombuffer(array.buffers()[1], dtype=offset_dtype, count=row_count + 1)
    lengths = np.diff(offsets)
    valid_lengths = lengths[validity]
    if valid_lengths.size == 0 or not np.all(valid_lengths == WKB_POINT_RECORD_DTYPE.itemsize):
        return None

    payload_size = int(offsets[-1])
    data_buffer = array.buffers()[2]
    if data_buffer is None:
        return None
    records = np.frombuffer(data_buffer, dtype=WKB_POINT_RECORD_DTYPE, count=valid_count, offset=0)
    if records.size != valid_count:
        return None
    if not np.all(records["byteorder"] == 1):
        return None
    if not np.all(records["type"] == WKB_TYPE_IDS[GeometryFamily.POINT]):
        return None
    if payload_size != valid_count * WKB_POINT_RECORD_DTYPE.itemsize:
        return None

    x = np.asarray(records["x"], dtype=np.float64)
    y = np.asarray(records["y"], dtype=np.float64)
    # Preserve partial-NaN point coordinates exactly as encoded. Only the
    # canonical NaN/NaN sentinel represents POINT EMPTY in WKB.
    empty_mask = np.isnan(x) & np.isnan(y)
    nonempty = ~empty_mask
    geometry_offsets = np.empty(valid_count + 1, dtype=np.int32)
    geometry_offsets[0] = 0
    if valid_count:
        geometry_offsets[1:] = np.cumsum(nonempty.astype(np.int32), dtype=np.int32)

    tags[validity] = FAMILY_TAGS[GeometryFamily.POINT]
    family_row_offsets[validity] = np.arange(valid_count, dtype=np.int32)
    families = {
        GeometryFamily.POINT: FamilyGeometryBuffer(
            family=GeometryFamily.POINT,
            schema=get_geometry_buffer_schema(GeometryFamily.POINT),
            row_count=valid_count,
            x=x[nonempty],
            y=y[nonempty],
            geometry_offsets=geometry_offsets,
            empty_mask=empty_mask,
        )
    }
    owned = OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families=families,
    )
    owned._record(
        DiagnosticKind.CREATED,
        "created owned geometry array from raw Arrow WKB point buffers",
        visible=True,
    )
    return owned


def _arrow_binary_offsets(array) -> np.ndarray:
    offset_dtype = np.int64 if "large_binary" in str(array.type) else np.int32
    return np.frombuffer(array.buffers()[1], dtype=offset_dtype, count=len(array) + 1)


def _decode_arrow_wkb_linestring_fast(array) -> OwnedGeometryArray | None:
    validity = np.asarray(array.is_valid().to_numpy(zero_copy_only=False), dtype=bool)
    row_count = int(validity.size)
    tags = np.full(row_count, -1, dtype=np.int8)
    family_row_offsets = np.full(row_count, -1, dtype=np.int32)
    valid_count = int(validity.sum())
    if valid_count == 0:
        return OwnedGeometryArray(
            validity=validity, tags=tags, family_row_offsets=family_row_offsets, families={}
        )

    offsets = _arrow_binary_offsets(array)
    data_buffer = array.buffers()[2]
    if data_buffer is None:
        return None
    data = memoryview(data_buffer)
    geometry_offsets = np.empty(valid_count + 1, dtype=np.int32)
    geometry_offsets[0] = 0
    total_points = 0
    valid_row = 0
    for row_index in range(row_count):
        if not validity[row_index]:
            continue
        start = int(offsets[row_index])
        end = int(offsets[row_index + 1])
        if end - start < 9:
            return None
        if (
            data[start] != 1
            or int.from_bytes(data[start + 1 : start + 5], "little")
            != WKB_TYPE_IDS[GeometryFamily.LINESTRING]
        ):
            return None
        point_count = int.from_bytes(data[start + 5 : start + 9], "little")
        if end - start != 9 + (point_count * 16):
            return None
        total_points += point_count
        valid_row += 1
        geometry_offsets[valid_row] = total_points

    x = np.empty(total_points, dtype=np.float64)
    y = np.empty(total_points, dtype=np.float64)
    empty_mask = np.zeros(valid_count, dtype=bool)
    coord_cursor = 0
    valid_row = 0
    for row_index in range(row_count):
        if not validity[row_index]:
            continue
        start = int(offsets[row_index])
        point_count = int.from_bytes(data[start + 5 : start + 9], "little")
        empty_mask[valid_row] = point_count == 0
        if point_count:
            coords = np.frombuffer(
                data[start + 9 : start + 9 + (point_count * 16)], dtype="<f8", count=point_count * 2
            ).reshape(point_count, 2)
            x[coord_cursor : coord_cursor + point_count] = coords[:, 0]
            y[coord_cursor : coord_cursor + point_count] = coords[:, 1]
            coord_cursor += point_count
        valid_row += 1

    tags[validity] = FAMILY_TAGS[GeometryFamily.LINESTRING]
    family_row_offsets[validity] = np.arange(valid_count, dtype=np.int32)
    owned = OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families={
            GeometryFamily.LINESTRING: FamilyGeometryBuffer(
                family=GeometryFamily.LINESTRING,
                schema=get_geometry_buffer_schema(GeometryFamily.LINESTRING),
                row_count=valid_count,
                x=x,
                y=y,
                geometry_offsets=geometry_offsets,
                empty_mask=empty_mask,
            )
        },
    )
    owned._record(
        DiagnosticKind.CREATED,
        "created owned geometry array from raw Arrow WKB linestring buffers",
        visible=True,
    )
    return owned


def _decode_arrow_wkb_linestring_uniform_fast(array) -> OwnedGeometryArray | None:
    validity = np.asarray(array.is_valid().to_numpy(zero_copy_only=False), dtype=bool)
    row_count = int(validity.size)
    tags = np.full(row_count, -1, dtype=np.int8)
    family_row_offsets = np.full(row_count, -1, dtype=np.int32)
    valid_count = int(validity.sum())
    if valid_count == 0:
        return OwnedGeometryArray(
            validity=validity, tags=tags, family_row_offsets=family_row_offsets, families={}
        )

    offsets = _arrow_binary_offsets(array)
    valid_lengths = np.diff(offsets)[validity]
    record_size = int(valid_lengths[0])
    if record_size < 9 or not np.all(valid_lengths == record_size):
        return None
    payload_size = int(offsets[-1])
    if payload_size != valid_count * record_size:
        return None
    payload_stride = record_size - 9
    if payload_stride % 16 != 0:
        return None
    point_count = payload_stride // 16
    coords_format = ("<f8", (point_count, 2))
    record_dtype = np.dtype(
        {
            "names": ["byteorder", "type", "count", "coords"],
            "formats": ["u1", "<u4", "<u4", coords_format],
            "offsets": [0, 1, 5, 9],
            "itemsize": record_size,
        }
    )
    records = np.frombuffer(array.buffers()[2], dtype=record_dtype, count=valid_count, offset=0)
    if not np.all(records["byteorder"] == 1):
        return None
    if not np.all(records["type"] == WKB_TYPE_IDS[GeometryFamily.LINESTRING]):
        return None
    if not np.all(records["count"] == point_count):
        return None

    if point_count == 0:
        x = np.asarray([], dtype=np.float64)
        y = np.asarray([], dtype=np.float64)
        geometry_offsets = np.zeros(valid_count + 1, dtype=np.int32)
        empty_mask = np.ones(valid_count, dtype=bool)
    else:
        x = np.asarray(records["coords"][:, :, 0].reshape(-1), dtype=np.float64)
        y = np.asarray(records["coords"][:, :, 1].reshape(-1), dtype=np.float64)
        geometry_offsets = np.arange(valid_count + 1, dtype=np.int32) * point_count
        empty_mask = np.zeros(valid_count, dtype=bool)

    tags[validity] = FAMILY_TAGS[GeometryFamily.LINESTRING]
    family_row_offsets[validity] = np.arange(valid_count, dtype=np.int32)
    owned = OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families={
            GeometryFamily.LINESTRING: FamilyGeometryBuffer(
                family=GeometryFamily.LINESTRING,
                schema=get_geometry_buffer_schema(GeometryFamily.LINESTRING),
                row_count=valid_count,
                x=x,
                y=y,
                geometry_offsets=geometry_offsets,
                empty_mask=empty_mask,
            )
        },
    )
    owned._record(
        DiagnosticKind.CREATED,
        "created owned geometry array from uniform raw Arrow WKB linestring buffers",
        visible=True,
    )
    return owned


def _decode_arrow_wkb_polygon_fast(array) -> OwnedGeometryArray | None:
    validity = np.asarray(array.is_valid().to_numpy(zero_copy_only=False), dtype=bool)
    row_count = int(validity.size)
    tags = np.full(row_count, -1, dtype=np.int8)
    family_row_offsets = np.full(row_count, -1, dtype=np.int32)
    valid_count = int(validity.sum())
    if valid_count == 0:
        return OwnedGeometryArray(
            validity=validity, tags=tags, family_row_offsets=family_row_offsets, families={}
        )

    offsets = _arrow_binary_offsets(array)
    data_buffer = array.buffers()[2]
    if data_buffer is None:
        return None
    data = memoryview(data_buffer)
    geometry_offsets = np.empty(valid_count + 1, dtype=np.int32)
    geometry_offsets[0] = 0
    empty_mask = np.zeros(valid_count, dtype=bool)
    total_rings = 0
    total_points = 0
    valid_row = 0
    for row_index in range(row_count):
        if not validity[row_index]:
            continue
        start = int(offsets[row_index])
        end = int(offsets[row_index + 1])
        if end - start < 9:
            return None
        if (
            data[start] != 1
            or int.from_bytes(data[start + 1 : start + 5], "little")
            != WKB_TYPE_IDS[GeometryFamily.POLYGON]
        ):
            return None
        ring_count = int.from_bytes(data[start + 5 : start + 9], "little")
        empty_mask[valid_row] = ring_count == 0
        cursor = start + 9
        for _ in range(ring_count):
            if cursor + 4 > end:
                return None
            point_count = int.from_bytes(data[cursor : cursor + 4], "little")
            cursor += 4 + (point_count * 16)
            if cursor > end:
                return None
            total_points += point_count
        if cursor != end:
            return None
        total_rings += ring_count
        valid_row += 1
        geometry_offsets[valid_row] = total_rings

    ring_offsets = np.empty(total_rings + 1, dtype=np.int32)
    x = np.empty(total_points, dtype=np.float64)
    y = np.empty(total_points, dtype=np.float64)
    ring_cursor = 0
    coord_cursor = 0
    valid_row = 0
    for row_index in range(row_count):
        if not validity[row_index]:
            continue
        start = int(offsets[row_index])
        ring_count = int.from_bytes(data[start + 5 : start + 9], "little")
        cursor = start + 9
        for _ in range(ring_count):
            point_count = int.from_bytes(data[cursor : cursor + 4], "little")
            cursor += 4
            ring_offsets[ring_cursor] = coord_cursor
            if point_count:
                coords = np.frombuffer(
                    data[cursor : cursor + (point_count * 16)], dtype="<f8", count=point_count * 2
                ).reshape(point_count, 2)
                x[coord_cursor : coord_cursor + point_count] = coords[:, 0]
                y[coord_cursor : coord_cursor + point_count] = coords[:, 1]
                coord_cursor += point_count
                cursor += point_count * 16
            ring_cursor += 1
        valid_row += 1
    ring_offsets[ring_cursor] = coord_cursor

    tags[validity] = FAMILY_TAGS[GeometryFamily.POLYGON]
    family_row_offsets[validity] = np.arange(valid_count, dtype=np.int32)
    owned = OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families={
            GeometryFamily.POLYGON: FamilyGeometryBuffer(
                family=GeometryFamily.POLYGON,
                schema=get_geometry_buffer_schema(GeometryFamily.POLYGON),
                row_count=valid_count,
                x=x,
                y=y,
                geometry_offsets=geometry_offsets,
                empty_mask=empty_mask,
                ring_offsets=ring_offsets,
            )
        },
    )
    owned._record(
        DiagnosticKind.CREATED,
        "created owned geometry array from raw Arrow WKB polygon buffers",
        visible=True,
    )
    return owned


def _decode_arrow_wkb_polygon_uniform_fast(array) -> OwnedGeometryArray | None:
    validity = np.asarray(array.is_valid().to_numpy(zero_copy_only=False), dtype=bool)
    row_count = int(validity.size)
    tags = np.full(row_count, -1, dtype=np.int8)
    family_row_offsets = np.full(row_count, -1, dtype=np.int32)
    valid_count = int(validity.sum())
    if valid_count == 0:
        return OwnedGeometryArray(
            validity=validity, tags=tags, family_row_offsets=family_row_offsets, families={}
        )

    offsets = _arrow_binary_offsets(array)
    valid_lengths = np.diff(offsets)[validity]
    record_size = int(valid_lengths[0])
    if record_size < 13 or not np.all(valid_lengths == record_size):
        return None
    payload_size = int(offsets[-1])
    if payload_size != valid_count * record_size:
        return None
    payload_stride = record_size - 13
    if payload_stride % 16 != 0:
        return None
    point_count = payload_stride // 16
    coords_format = ("<f8", (point_count, 2))
    record_dtype = np.dtype(
        {
            "names": ["byteorder", "type", "ring_count", "count", "coords"],
            "formats": ["u1", "<u4", "<u4", "<u4", coords_format],
            "offsets": [0, 1, 5, 9, 13],
            "itemsize": record_size,
        }
    )
    records = np.frombuffer(array.buffers()[2], dtype=record_dtype, count=valid_count, offset=0)
    if not np.all(records["byteorder"] == 1):
        return None
    if not np.all(records["type"] == WKB_TYPE_IDS[GeometryFamily.POLYGON]):
        return None
    if not np.all(records["ring_count"] == 1):
        return None
    if not np.all(records["count"] == point_count):
        return None

    if point_count == 0:
        x = np.asarray([], dtype=np.float64)
        y = np.asarray([], dtype=np.float64)
        ring_offsets = np.zeros(valid_count + 1, dtype=np.int32)
    else:
        x = np.asarray(records["coords"][:, :, 0].reshape(-1), dtype=np.float64)
        y = np.asarray(records["coords"][:, :, 1].reshape(-1), dtype=np.float64)
        ring_offsets = np.arange(valid_count + 1, dtype=np.int32) * point_count
    geometry_offsets = np.arange(valid_count + 1, dtype=np.int32)
    empty_mask = np.zeros(valid_count, dtype=bool)

    tags[validity] = FAMILY_TAGS[GeometryFamily.POLYGON]
    family_row_offsets[validity] = np.arange(valid_count, dtype=np.int32)
    owned = OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families={
            GeometryFamily.POLYGON: FamilyGeometryBuffer(
                family=GeometryFamily.POLYGON,
                schema=get_geometry_buffer_schema(GeometryFamily.POLYGON),
                row_count=valid_count,
                x=x,
                y=y,
                geometry_offsets=geometry_offsets,
                empty_mask=empty_mask,
                ring_offsets=ring_offsets,
            )
        },
    )
    owned._record(
        DiagnosticKind.CREATED,
        "created owned geometry array from uniform raw Arrow WKB polygon buffers",
        visible=True,
    )
    return owned


def _promote_arrow_fast_owned_to_device(
    owned: OwnedGeometryArray | None,
    *,
    detail: str,
) -> OwnedGeometryArray | None:
    from vibespatial.runtime import ExecutionMode, get_requested_mode

    if owned is None:
        return None
    if get_requested_mode() is ExecutionMode.CPU:
        return owned
    runtime = get_cuda_runtime()
    if not runtime.available():
        return owned
    owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason=detail,
    )
    return owned


def _try_uniform_arrow_wkb_fast_decode(array) -> OwnedGeometryArray | None:
    point_fast = _promote_arrow_fast_owned_to_device(
        _decode_arrow_wkb_point_fast(array),
        detail="bulk h2d promotion after uniform Arrow WKB point fast parse",
    )
    if point_fast is not None:
        return point_fast

    linestring_uniform_fast = _promote_arrow_fast_owned_to_device(
        _decode_arrow_wkb_linestring_uniform_fast(array),
        detail="bulk h2d promotion after uniform Arrow WKB linestring fast parse",
    )
    if linestring_uniform_fast is not None:
        return linestring_uniform_fast

    polygon_uniform_fast = _promote_arrow_fast_owned_to_device(
        _decode_arrow_wkb_polygon_uniform_fast(array),
        detail="bulk h2d promotion after uniform Arrow WKB polygon fast parse",
    )
    if polygon_uniform_fast is not None:
        return polygon_uniform_fast

    return None


def _decode_arrow_wkb_multipolygon_fast(array) -> OwnedGeometryArray | None:
    """Decode a WKB Arrow binary column containing only MultiPolygon geometries.

    Two-pass approach (same as the Polygon fast path):
    Pass 1 scans headers to compute total rings, points, and polygon counts.
    Pass 2 copies coordinates into pre-allocated numpy arrays.
    Returns None if any record is not a valid little-endian MultiPolygon.
    """
    validity = np.asarray(array.is_valid().to_numpy(zero_copy_only=False), dtype=bool)
    row_count = int(validity.size)
    tags = np.full(row_count, -1, dtype=np.int8)
    family_row_offsets = np.full(row_count, -1, dtype=np.int32)
    valid_count = int(validity.sum())
    if valid_count == 0:
        return OwnedGeometryArray(
            validity=validity,
            tags=tags,
            family_row_offsets=family_row_offsets,
            families={},
        )

    offsets = _arrow_binary_offsets(array)
    data_buffer = array.buffers()[2]
    if data_buffer is None:
        return None
    data = memoryview(data_buffer)

    geometry_offsets = np.empty(valid_count + 1, dtype=np.int32)
    geometry_offsets[0] = 0
    empty_mask = np.zeros(valid_count, dtype=bool)
    total_polygons = 0
    total_rings = 0
    total_points = 0

    wkb_mp_type = WKB_TYPE_IDS[GeometryFamily.MULTIPOLYGON]
    wkb_poly_type = WKB_TYPE_IDS[GeometryFamily.POLYGON]

    # --- Pass 1: scan structure ---
    valid_row = 0
    for row_index in range(row_count):
        if not validity[row_index]:
            continue
        start = int(offsets[row_index])
        end = int(offsets[row_index + 1])
        if end - start < 9:
            return None
        if data[start] != 1 or int.from_bytes(data[start + 1 : start + 5], "little") != wkb_mp_type:
            return None
        polygon_count = int.from_bytes(data[start + 5 : start + 9], "little")
        empty_mask[valid_row] = polygon_count == 0
        cursor = start + 9
        for _ in range(polygon_count):
            if cursor + 9 > end:
                return None
            if (
                data[cursor] != 1
                or int.from_bytes(data[cursor + 1 : cursor + 5], "little") != wkb_poly_type
            ):
                return None
            ring_count = int.from_bytes(data[cursor + 5 : cursor + 9], "little")
            cursor += 9
            for _ in range(ring_count):
                if cursor + 4 > end:
                    return None
                point_count = int.from_bytes(data[cursor : cursor + 4], "little")
                cursor += 4 + (point_count * 16)
                if cursor > end:
                    return None
                total_points += point_count
            total_rings += ring_count
        if cursor != end:
            return None
        total_polygons += polygon_count
        valid_row += 1
        geometry_offsets[valid_row] = total_polygons

    # --- Pass 2: extract coordinates ---
    part_offsets = np.empty(total_polygons + 1, dtype=np.int32)
    ring_offsets = np.empty(total_rings + 1, dtype=np.int32)
    x = np.empty(total_points, dtype=np.float64)
    y = np.empty(total_points, dtype=np.float64)
    poly_cursor = 0
    ring_cursor = 0
    coord_cursor = 0
    for row_index in range(row_count):
        if not validity[row_index]:
            continue
        start = int(offsets[row_index])
        polygon_count = int.from_bytes(data[start + 5 : start + 9], "little")
        cursor = start + 9
        for _ in range(polygon_count):
            ring_count = int.from_bytes(data[cursor + 5 : cursor + 9], "little")
            cursor += 9
            part_offsets[poly_cursor] = ring_cursor
            for _ in range(ring_count):
                point_count = int.from_bytes(data[cursor : cursor + 4], "little")
                cursor += 4
                ring_offsets[ring_cursor] = coord_cursor
                if point_count:
                    nbytes = point_count * 16
                    coords = np.frombuffer(
                        data[cursor : cursor + nbytes],
                        dtype="<f8",
                        count=point_count * 2,
                    ).reshape(point_count, 2)
                    x[coord_cursor : coord_cursor + point_count] = coords[:, 0]
                    y[coord_cursor : coord_cursor + point_count] = coords[:, 1]
                    coord_cursor += point_count
                    cursor += nbytes
                ring_cursor += 1
            poly_cursor += 1
    part_offsets[poly_cursor] = ring_cursor
    ring_offsets[ring_cursor] = coord_cursor

    tags[validity] = FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]
    family_row_offsets[validity] = np.arange(valid_count, dtype=np.int32)
    owned = OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families={
            GeometryFamily.MULTIPOLYGON: FamilyGeometryBuffer(
                family=GeometryFamily.MULTIPOLYGON,
                schema=get_geometry_buffer_schema(GeometryFamily.MULTIPOLYGON),
                row_count=valid_count,
                x=x,
                y=y,
                geometry_offsets=geometry_offsets,
                empty_mask=empty_mask,
                part_offsets=part_offsets,
                ring_offsets=ring_offsets,
            )
        },
    )
    owned._record(
        DiagnosticKind.CREATED,
        "created owned geometry array from raw Arrow WKB multipolygon buffers",
        visible=True,
    )
    return owned


def _hexify_if_requested(
    values: list[bytes | None], *, hex_output: bool
) -> list[bytes | str | None]:
    if not hex_output:
        return values
    return [None if value is None else value.hex() for value in values]


def _encode_point_wkb_batch(buffer: FamilyGeometryBuffer) -> list[bytes]:
    if buffer.row_count == 0:
        return []
    records = np.empty(buffer.row_count, dtype=WKB_POINT_RECORD_DTYPE)
    records["byteorder"] = 1
    records["type"] = WKB_TYPE_IDS[GeometryFamily.POINT]
    x = np.full(buffer.row_count, np.nan, dtype=np.float64)
    y = np.full(buffer.row_count, np.nan, dtype=np.float64)
    nonempty = ~buffer.empty_mask
    x[nonempty] = buffer.x
    y[nonempty] = buffer.y
    records["x"] = x
    records["y"] = y
    payload = records.tobytes()
    return [payload[index * 21 : (index + 1) * 21] for index in range(buffer.row_count)]


def _pack_linestring_wkb(buffer: FamilyGeometryBuffer, row: int) -> bytes:
    start = int(buffer.geometry_offsets[row])
    end = int(buffer.geometry_offsets[row + 1])
    count = end - start
    payload = bytearray(9 + (count * 16))
    payload[0] = 1
    payload[1:5] = WKB_TYPE_IDS[GeometryFamily.LINESTRING].to_bytes(4, "little")
    payload[5:9] = count.to_bytes(4, "little")
    cursor = 9
    for x, y in zip(buffer.x[start:end], buffer.y[start:end], strict=True):
        struct.pack_into("<dd", payload, cursor, float(x), float(y))
        cursor += 16
    return bytes(payload)


def _pack_polygon_wkb(buffer: FamilyGeometryBuffer, row: int) -> bytes:
    def _ring_needs_closure(coord_start: int, coord_end: int) -> bool:
        if coord_end <= coord_start:
            return False
        return float(buffer.x[coord_start]) != float(buffer.x[coord_end - 1]) or float(
            buffer.y[coord_start]
        ) != float(buffer.y[coord_end - 1])

    ring_start = int(buffer.geometry_offsets[row])
    ring_end = int(buffer.geometry_offsets[row + 1])
    size = 9
    ring_ranges: list[tuple[int, int, bool]] = []
    for ring_index in range(ring_start, ring_end):
        coord_start = int(buffer.ring_offsets[ring_index])
        coord_end = int(buffer.ring_offsets[ring_index + 1])
        needs_closure = _ring_needs_closure(coord_start, coord_end)
        ring_ranges.append((coord_start, coord_end, needs_closure))
        size += 4 + (((coord_end - coord_start) + int(needs_closure)) * 16)
    payload = bytearray(size)
    payload[0] = 1
    payload[1:5] = WKB_TYPE_IDS[GeometryFamily.POLYGON].to_bytes(4, "little")
    payload[5:9] = len(ring_ranges).to_bytes(4, "little")
    cursor = 9
    for coord_start, coord_end, needs_closure in ring_ranges:
        count = (coord_end - coord_start) + int(needs_closure)
        payload[cursor : cursor + 4] = count.to_bytes(4, "little")
        cursor += 4
        for x, y in zip(
            buffer.x[coord_start:coord_end], buffer.y[coord_start:coord_end], strict=True
        ):
            struct.pack_into("<dd", payload, cursor, float(x), float(y))
            cursor += 16
        if needs_closure:
            struct.pack_into(
                "<dd",
                payload,
                cursor,
                float(buffer.x[coord_start]),
                float(buffer.y[coord_start]),
            )
            cursor += 16
    return bytes(payload)


def _pack_multipoint_wkb(buffer: FamilyGeometryBuffer, row: int) -> bytes:
    start = int(buffer.geometry_offsets[row])
    end = int(buffer.geometry_offsets[row + 1])
    count = end - start
    payload = bytearray(9 + (count * 21))
    payload[0] = 1
    payload[1:5] = WKB_TYPE_IDS[GeometryFamily.MULTIPOINT].to_bytes(4, "little")
    payload[5:9] = count.to_bytes(4, "little")
    cursor = 9
    for x, y in zip(buffer.x[start:end], buffer.y[start:end], strict=True):
        payload[cursor] = 1
        payload[cursor + 1 : cursor + 5] = WKB_TYPE_IDS[GeometryFamily.POINT].to_bytes(4, "little")
        struct.pack_into("<dd", payload, cursor + 5, float(x), float(y))
        cursor += 21
    return bytes(payload)


def _pack_multilinestring_wkb(buffer: FamilyGeometryBuffer, row: int) -> bytes:
    part_start = int(buffer.geometry_offsets[row])
    part_end = int(buffer.geometry_offsets[row + 1])
    size = 9
    part_ranges: list[tuple[int, int]] = []
    for part_index in range(part_start, part_end):
        coord_start = int(buffer.part_offsets[part_index])
        coord_end = int(buffer.part_offsets[part_index + 1])
        part_ranges.append((coord_start, coord_end))
        size += 9 + ((coord_end - coord_start) * 16)
    payload = bytearray(size)
    payload[0] = 1
    payload[1:5] = WKB_TYPE_IDS[GeometryFamily.MULTILINESTRING].to_bytes(4, "little")
    payload[5:9] = len(part_ranges).to_bytes(4, "little")
    cursor = 9
    for coord_start, coord_end in part_ranges:
        count = coord_end - coord_start
        payload[cursor] = 1
        payload[cursor + 1 : cursor + 5] = WKB_TYPE_IDS[GeometryFamily.LINESTRING].to_bytes(
            4, "little"
        )
        payload[cursor + 5 : cursor + 9] = count.to_bytes(4, "little")
        cursor += 9
        for x, y in zip(
            buffer.x[coord_start:coord_end], buffer.y[coord_start:coord_end], strict=True
        ):
            struct.pack_into("<dd", payload, cursor, float(x), float(y))
            cursor += 16
    return bytes(payload)


def _pack_multipolygon_wkb(buffer: FamilyGeometryBuffer, row: int) -> bytes:
    def _ring_needs_closure(coord_start: int, coord_end: int) -> bool:
        if coord_end <= coord_start:
            return False
        return float(buffer.x[coord_start]) != float(buffer.x[coord_end - 1]) or float(
            buffer.y[coord_start]
        ) != float(buffer.y[coord_end - 1])

    polygon_start = int(buffer.geometry_offsets[row])
    polygon_end = int(buffer.geometry_offsets[row + 1])
    polygon_specs: list[list[tuple[int, int, bool]]] = []
    size = 9
    for polygon_index in range(polygon_start, polygon_end):
        ring_start = int(buffer.part_offsets[polygon_index])
        ring_end = int(buffer.part_offsets[polygon_index + 1])
        ring_ranges: list[tuple[int, int, bool]] = []
        polygon_size = 9
        for ring_index in range(ring_start, ring_end):
            coord_start = int(buffer.ring_offsets[ring_index])
            coord_end = int(buffer.ring_offsets[ring_index + 1])
            needs_closure = _ring_needs_closure(coord_start, coord_end)
            ring_ranges.append((coord_start, coord_end, needs_closure))
            polygon_size += 4 + (((coord_end - coord_start) + int(needs_closure)) * 16)
        polygon_specs.append(ring_ranges)
        size += polygon_size
    payload = bytearray(size)
    payload[0] = 1
    payload[1:5] = WKB_TYPE_IDS[GeometryFamily.MULTIPOLYGON].to_bytes(4, "little")
    payload[5:9] = len(polygon_specs).to_bytes(4, "little")
    cursor = 9
    for ring_ranges in polygon_specs:
        payload[cursor] = 1
        payload[cursor + 1 : cursor + 5] = WKB_TYPE_IDS[GeometryFamily.POLYGON].to_bytes(
            4, "little"
        )
        payload[cursor + 5 : cursor + 9] = len(ring_ranges).to_bytes(4, "little")
        cursor += 9
        for coord_start, coord_end, needs_closure in ring_ranges:
            count = (coord_end - coord_start) + int(needs_closure)
            payload[cursor : cursor + 4] = count.to_bytes(4, "little")
            cursor += 4
            for x, y in zip(
                buffer.x[coord_start:coord_end], buffer.y[coord_start:coord_end], strict=True
            ):
                struct.pack_into("<dd", payload, cursor, float(x), float(y))
                cursor += 16
            if needs_closure:
                struct.pack_into(
                    "<dd",
                    payload,
                    cursor,
                    float(buffer.x[coord_start]),
                    float(buffer.y[coord_start]),
                )
                cursor += 16
    return bytes(payload)


def _encode_native_wkb(
    array: OwnedGeometryArray,
    *,
    hex_output: bool = False,
) -> tuple[list[bytes | str | None], WKBPartitionPlan]:
    partition_plan = WKBPartitionPlan(
        total_rows=array.row_count,
        valid_rows=int(array.validity.sum()),
        null_rows=int((~array.validity).sum()),
        native_rows=int(array.validity.sum()),
        fallback_rows=0,
        family_counts={
            family.value: int((array.tags == FAMILY_TAGS[family]).sum())
            for family in GeometryFamily
        },
        fallback_indexes=tuple(),
        fallback_reason_counts={},
        reason="Owned buffers already provide family tags and offsets, so encode can go straight to family-local WKB assembly.",
    )
    outputs: list[bytes | None] = [None] * array.row_count
    encoded_by_family: dict[GeometryFamily, list[bytes]] = {}
    for family, buffer in array.families.items():
        if family is GeometryFamily.POINT:
            encoded_by_family[family] = _encode_point_wkb_batch(buffer)
        else:
            batch: list[bytes] = []
            for row in range(buffer.row_count):
                if bool(buffer.empty_mask[row]):
                    payload = bytearray(9)
                    payload[0] = 1
                    payload[1:5] = WKB_TYPE_IDS[family].to_bytes(4, "little")
                    payload[5:9] = (0).to_bytes(4, "little")
                    batch.append(bytes(payload))
                    continue
                if family is GeometryFamily.LINESTRING:
                    batch.append(_pack_linestring_wkb(buffer, row))
                elif family is GeometryFamily.POLYGON:
                    batch.append(_pack_polygon_wkb(buffer, row))
                elif family is GeometryFamily.MULTIPOINT:
                    batch.append(_pack_multipoint_wkb(buffer, row))
                elif family is GeometryFamily.MULTILINESTRING:
                    batch.append(_pack_multilinestring_wkb(buffer, row))
                elif family is GeometryFamily.MULTIPOLYGON:
                    batch.append(_pack_multipolygon_wkb(buffer, row))
            encoded_by_family[family] = batch
    for row_index in range(array.row_count):
        if not bool(array.validity[row_index]):
            continue
        family = TAG_FAMILIES[int(array.tags[row_index])]
        outputs[row_index] = encoded_by_family[family][int(array.family_row_offsets[row_index])]
    return _hexify_if_requested(outputs, hex_output=hex_output), partition_plan


def decode_wkb_owned(
    values: list[bytes | str | None] | tuple[bytes | str | None, ...],
    *,
    on_invalid: str = "raise",
) -> OwnedGeometryArray:
    # Try the GPU-first staged device pipeline before falling back to the
    # host-side bridge. Large native list[bytes] inputs stage directly into
    # contiguous payload/offset buffers to avoid the extra Arrow/pylibcudf
    # bridge overhead on this public decode surface.
    gpu_attempt = _try_gpu_wkb_list_decode(values, on_invalid=on_invalid)
    if gpu_attempt.result is not None:
        record_dispatch_event(
            surface="vibespatial.io.wkb",
            operation="decode",
            implementation="device_wkb_decode",
            reason="GPU WKB decode via the staged device pipeline (list[bytes] input)",
            selected=ExecutionMode.GPU,
        )
        return gpu_attempt.result
    if gpu_attempt.fallback_detail is not None:
        record_fallback_event(
            surface="vibespatial.io.wkb",
            reason="explicit CPU fallback after staged GPU WKB decode could not complete",
            detail=gpu_attempt.fallback_detail,
            selected=ExecutionMode.CPU,
            pipeline="io/wkb_decode",
            d2h_transfer=True,
        )

    # Fall through to host-side staged decode.
    plan = plan_wkb_bridge(IOOperation.DECODE)
    record_dispatch_event(
        surface="vibespatial.io.wkb",
        operation="decode",
        implementation="owned_wkb_bridge",
        reason=plan.reason,
        selected=ExecutionMode.CPU,
    )
    array, partition_plan = _decode_native_wkb(values, on_invalid=on_invalid)
    if partition_plan.fallback_rows:
        fallback_detail = (
            f"{partition_plan.fallback_rows} rows entered the fallback pool during staged decode"
        )
        fallback_reasons = _format_wkb_fallback_reason_counts(partition_plan)
        if fallback_reasons:
            fallback_detail = f"{fallback_detail} ({fallback_reasons})"
        record_fallback_event(
            surface="vibespatial.io.wkb",
            reason="explicit CPU fallback for unsupported or malformed WKB rows",
            detail=fallback_detail,
            selected=ExecutionMode.CPU,
            pipeline="io/wkb_decode",
        )
    return array


def _try_gpu_wkb_list_decode(
    values: list[bytes | str | None] | tuple[bytes | str | None, ...],
    *,
    on_invalid: str = "raise",
) -> _GpuWkbDecodeAttempt:
    """Attempt GPU WKB decode of list[bytes] input.

    Large native lists stage directly into contiguous payload/offset buffers and
    feed the device decode kernels without first materializing a pyarrow /
    pylibcudf bridge. The older bridge remains as a fallback for cases where the
    direct staged path cannot run.
    """
    from vibespatial.runtime import ExecutionMode, get_requested_mode

    if get_requested_mode() is ExecutionMode.CPU:
        return _GpuWkbDecodeAttempt(result=None)
    try:
        runtime = get_cuda_runtime()
        if not runtime.available():
            return _GpuWkbDecodeAttempt(result=None)
    except Exception:
        return _GpuWkbDecodeAttempt(result=None)

    # Avoid GPU staging for small WKB batches where host decode is cheaper.
    if len(values) < DEVICE_WKB_LIST_DECODE_MIN_ROWS:
        return _GpuWkbDecodeAttempt(result=None)

    normalized: list[bytes | None] = [_normalize_wkb_value(v) for v in values]
    non_null_mask = _non_null_wkb_input_mask(normalized)
    arrow_error: str | None = None
    try:
        import pyarrow as pa

        arrow_array = pa.array(normalized, type=pa.binary())
        arrow_attempt = _try_gpu_wkb_arrow_decode(arrow_array, on_invalid=on_invalid)
        if arrow_attempt.result is not None:
            return arrow_attempt
        arrow_error = arrow_attempt.fallback_detail
    except _GpuWkbOnInvalidError:
        raise
    except Exception as exc:
        arrow_error = f"Arrow WKB list bridge failed: {type(exc).__name__}: {exc}"

    staged_records = _prepare_native_wkb_list_for_device(normalized)
    if staged_records is None:
        return _GpuWkbDecodeAttempt(result=None, fallback_detail=arrow_error)

    staged_error: str | None = None

    try:
        from vibespatial.kernels.core.wkb_decode import decode_wkb_device_pipeline

        offsets_host, payload_host = staged_records
        payload_device = runtime.from_host(payload_host)
        offsets_device = runtime.from_host(offsets_host)
        result = decode_wkb_device_pipeline(payload_device, offsets_device, len(values))
        if on_invalid == "raise":
            _raise_on_invalid_gpu_wkb_decode(result, non_null_mask)
        return _GpuWkbDecodeAttempt(result=result)
    except _GpuWkbOnInvalidError:
        raise
    except Exception as exc:
        staged_error = f"staged device decode failed: {type(exc).__name__}: {exc}"

    try:
        import pyarrow as pa

        from .pylibcudf import _decode_pylibcudf_wkb_general_column_to_owned

        # Single bulk allocation: list[bytes|None] -> pa.BinaryArray.
        arrow_array = pa.array(normalized, type=pa.binary())

        # pylibcudf requires string/large_string layout (identical to binary).
        arrow_str = pa.Array.from_buffers(
            pa.string(),
            len(arrow_array),
            arrow_array.buffers(),
            null_count=arrow_array.null_count,
        )

        plc_column = pylibcudf_column_from_arrow(arrow_str)
        result = _decode_pylibcudf_wkb_general_column_to_owned(plc_column)
        if on_invalid == "raise":
            _raise_on_invalid_gpu_wkb_decode(result, non_null_mask)
        return _GpuWkbDecodeAttempt(result=result)
    except _GpuWkbOnInvalidError:
        raise
    except (ImportError, NotImplementedError) as exc:
        detail = "; ".join(
            message
            for message in (
                arrow_error,
                staged_error or "staged device decode did not produce a result",
            )
            if message
        )
        return _GpuWkbDecodeAttempt(
            result=None,
            fallback_detail=(
                f"{detail}; pylibcudf WKB decode bridge unavailable: {type(exc).__name__}: {exc}"
            ),
        )
    except Exception as exc:
        detail = "; ".join(
            message
            for message in (
                arrow_error,
                staged_error or "staged device decode did not produce a result",
            )
            if message
        )
        return _GpuWkbDecodeAttempt(
            result=None,
            fallback_detail=(
                f"{detail}; pylibcudf WKB decode bridge failed: {type(exc).__name__}: {exc}"
            ),
        )


def decode_wkb_arrow_array_owned(
    array,
    *,
    on_invalid: str = "raise",
    allow_fallback: bool = True,
) -> OwnedGeometryArray:
    uniform_fast = _try_uniform_arrow_wkb_fast_decode(array)
    if uniform_fast is not None:
        return uniform_fast

    gpu_attempt = _try_gpu_wkb_arrow_decode(array, on_invalid=on_invalid)
    if gpu_attempt.result is not None:
        return gpu_attempt.result
    if gpu_attempt.fallback_detail is not None:
        if not allow_fallback:
            raise NotImplementedError(gpu_attempt.fallback_detail)
        record_fallback_event(
            surface="vibespatial.io.wkb",
            reason="explicit CPU fallback after GPU Arrow WKB decode could not complete",
            detail=gpu_attempt.fallback_detail,
            selected=ExecutionMode.CPU,
            pipeline="io/wkb_decode",
            d2h_transfer=True,
        )
    linestring_fast = _decode_arrow_wkb_linestring_fast(array)
    if linestring_fast is not None:
        return linestring_fast
    polygon_fast = _decode_arrow_wkb_polygon_fast(array)
    if polygon_fast is not None:
        return polygon_fast
    multipolygon_fast = _decode_arrow_wkb_multipolygon_fast(array)
    if multipolygon_fast is not None:
        return multipolygon_fast
    if not allow_fallback:
        raise NotImplementedError("Arrow WKB decode requires a supported native path")
    values = np.asarray(array.to_numpy(zero_copy_only=False), dtype=object)
    return decode_wkb_owned(list(values), on_invalid=on_invalid)


def _try_gpu_wkb_arrow_decode(
    array,
    *,
    on_invalid: str = "raise",
) -> _GpuWkbDecodeAttempt:
    """Attempt GPU WKB decode of a PyArrow binary/large_binary array via pylibcudf."""
    from vibespatial.runtime import ExecutionMode, get_requested_mode

    if get_requested_mode() is ExecutionMode.CPU:
        return _GpuWkbDecodeAttempt(result=None)
    runtime = get_cuda_runtime()
    if not runtime.available():
        return _GpuWkbDecodeAttempt(result=None)
    non_null_mask = np.asarray(array.is_valid().to_numpy(zero_copy_only=False), dtype=bool)
    try:
        import pyarrow as pa

        from .pylibcudf import _decode_pylibcudf_wkb_general_column_to_owned

        # pylibcudf does not support Arrow binary/large_binary types, but the
        # memory layout is identical to string/large_string (offsets + bytes).
        # Zero-copy reinterpret so plc.Column.from_arrow succeeds.
        if pa.types.is_binary(array.type) or pa.types.is_large_binary(array.type):
            target = pa.string() if pa.types.is_binary(array.type) else pa.large_string()
            array = pa.Array.from_buffers(
                target,
                len(array),
                array.buffers(),
                null_count=array.null_count,
            )

        plc_column = pylibcudf_column_from_arrow(array)
        result = _decode_pylibcudf_wkb_general_column_to_owned(plc_column)
        if on_invalid == "raise":
            _raise_on_invalid_gpu_wkb_decode(result, non_null_mask)
        return _GpuWkbDecodeAttempt(result=result)
    except _GpuWkbOnInvalidError:
        raise
    except (ImportError, NotImplementedError) as exc:
        return _GpuWkbDecodeAttempt(
            result=None,
            fallback_detail=(
                f"GPU Arrow WKB decode bridge unavailable: {type(exc).__name__}: {exc}"
            ),
        )
    except Exception as exc:
        return _GpuWkbDecodeAttempt(
            result=None,
            fallback_detail=(f"GPU Arrow WKB decode bridge failed: {type(exc).__name__}: {exc}"),
        )


def _try_gpu_wkb_encode(
    array: OwnedGeometryArray,
    *,
    hex_output: bool = False,
) -> list[bytes | str | None] | None:
    """Attempt GPU-accelerated WKB encode. Returns list[bytes|str|None] or None on failure."""
    from vibespatial.runtime import ExecutionMode, get_requested_mode

    if get_requested_mode() is ExecutionMode.CPU:
        return None
    # 1. Check runtime available
    try:
        runtime = get_cuda_runtime()
        if not runtime.available():
            return None
    except Exception:
        return None

    # 2. Check minimum row count (GPU overhead not worth it for tiny arrays)
    if array.row_count < 500:
        return None

    # 3. Try GPU encode
    try:
        import pyarrow as pa

        plc_column = _encode_owned_wkb_column_device(array)
        # Single bulk D2H transfer via Arrow
        arrow_col = pylibcudf_to_arrow(plc_column)
        # The plc column is STRING type; cast to binary so raw WKB bytes
        # survive the Arrow conversion without UTF-8 validation issues.
        arrow_bin = arrow_col.cast(pa.binary())
        values = arrow_bin.to_pylist()
        if hex_output:
            values = [v.hex() if isinstance(v, bytes) else v for v in values]
        return values
    except Exception:
        return None


def _try_gpu_wkb_encode_arrow(
    owned: OwnedGeometryArray,
    *,
    field_name: str = "geometry",
    crs: Any | None = None,
) -> tuple | None:
    """GPU WKB encode returning (pa.Field, pa.Array) for zero-copy parquet integration.

    Unlike ``_try_gpu_wkb_encode`` which round-trips through ``.to_pylist()``,
    this casts the device-resident pylibcudf column directly to a ``pa.Array``
    via Arrow IPC -- a single bulk D->H transfer with no per-row Python
    materialisation.  Returns None if GPU is unavailable or encode fails.
    """
    from vibespatial.runtime import ExecutionMode, get_requested_mode

    if get_requested_mode() is ExecutionMode.CPU:
        return None
    try:
        runtime = get_cuda_runtime()
        if not runtime.available():
            return None
    except Exception:
        return None
    if owned.row_count < 500:
        return None
    try:
        import pyarrow as pa

        plc_column = _encode_owned_wkb_column_device(owned)
        # Single bulk D->H via Arrow -- no Python list intermediary
        arrow_col = pylibcudf_to_arrow(plc_column)
        wkb_arr = arrow_col.cast(pa.binary())

        field_metadata = {}
        if crs is not None:
            try:
                crs_json = crs.to_json_dict()
            except AttributeError:
                crs_json = None
            if crs_json is not None:
                import json

                field_metadata[b"ARROW:extension:metadata"] = json.dumps({"crs": crs_json}).encode()
        field_metadata[b"ARROW:extension:name"] = b"geoarrow.wkb"
        field = pa.field(field_name, pa.binary(), nullable=True, metadata=field_metadata)
        return field, wkb_arr
    except Exception:
        return None


def encode_wkb_owned(
    array: OwnedGeometryArray,
    *,
    hex: bool = False,
) -> list[bytes | str | None]:
    plan = plan_wkb_bridge(IOOperation.ENCODE)
    # Try GPU-accelerated encode first
    gpu_result = _try_gpu_wkb_encode(array, hex_output=hex)
    if gpu_result is not None:
        record_dispatch_event(
            surface="vibespatial.io.wkb",
            operation="encode",
            implementation="device_wkb_encode",
            reason="GPU WKB encode via device kernel pipeline",
            selected=ExecutionMode.GPU,
        )
        record_native_export_boundary(
            NativeExportBoundary(
                surface="vibespatial.io.wkb.encode_wkb_owned",
                operation="owned_geometry_to_wkb",
                target="wkb",
                reason="owned geometry exported to host-visible WKB values",
                detail=f"hex={int(hex)}, implementation=device_wkb_encode",
                row_count=array.row_count,
                d2h_transfer=True,
            )
        )
        return gpu_result
    # Fall through to host path
    record_dispatch_event(
        surface="vibespatial.io.wkb",
        operation="encode",
        implementation="owned_wkb_bridge",
        reason=plan.reason,
        selected=ExecutionMode.CPU,
    )
    values, partition_plan = _encode_native_wkb(array, hex_output=hex)
    record_native_export_boundary(
        NativeExportBoundary(
            surface="vibespatial.io.wkb.encode_wkb_owned",
            operation="owned_geometry_to_wkb",
            target="wkb",
            reason="owned geometry exported to host-visible WKB values",
            detail=(
                f"hex={int(hex)}, implementation=owned_wkb_bridge, "
                f"fallback_rows={partition_plan.fallback_rows}"
            ),
            row_count=array.row_count,
            d2h_transfer=array.device_state is not None,
        )
    )
    if partition_plan.fallback_rows:
        record_fallback_event(
            surface="vibespatial.io.wkb",
            reason="explicit CPU fallback for unsupported owned rows during WKB encode",
            detail=f"{partition_plan.fallback_rows} rows entered the fallback pool during staged encode",
            selected=ExecutionMode.CPU,
            pipeline="io/wkb_encode",
            d2h_transfer=True,
        )
    return values


def _homogeneous_family(array: OwnedGeometryArray):
    if array.device_state is not None:
        return _homogeneous_family_from_device_structure(array)
    validity, tags, _family_row_offsets = _authoritative_host_metadata(array)
    valid_tags = tags[validity]
    if valid_tags.size == 0:
        raise ValueError("Cannot encode an all-null geometry array to native GeoArrow")
    unique_tags = np.unique(valid_tags)
    if unique_tags.size != 1:
        raise ValueError("Native GeoArrow fast path requires a homogeneous geometry family")
    return TAG_FAMILIES[int(unique_tags[0])]


def _encode_owned_wkb_array(
    owned: OwnedGeometryArray,
    *,
    field_name: str = "geometry",
    crs: Any | None = None,
    return_mode: bool = False,
    force_device: bool = False,
) -> tuple:
    """Encode OwnedGeometryArray to WKB pyarrow array.

    Tries GPU-accelerated encoding first; falls back to host-side
    row-by-row encoding only when GPU is unavailable.
    """
    # Try GPU path -- keeps coordinates on device, encodes WKB in parallel
    gpu_result = _try_gpu_wkb_encode_arrow(owned, field_name=field_name, crs=crs)
    if gpu_result is not None:
        record_dispatch_event(
            surface="vibespatial.io.wkb",
            operation="encode_to_parquet",
            implementation="device_wkb_encode",
            reason="GPU WKB encode for parquet write -- no host coordinate materialization",
            selected=ExecutionMode.GPU,
        )
        return (*gpu_result, ExecutionMode.GPU) if return_mode else gpu_result

    # Make one final direct device-encode attempt before surfacing a host
    # fallback.  Strict-native writers must not fail just because the generic
    # helper declined GPU encode on a small batch.  A cached device_state is
    # also a valid terminal output-byte source even when public ownership has
    # restored the logical residency flag to HOST.
    if force_device or strict_native_mode_enabled() or (owned.device_state is not None):
        try:
            import pyarrow as pa

            plc_column = _encode_owned_wkb_column_device(owned)
            arrow_col = pylibcudf_to_arrow(plc_column)
            wkb_arr = arrow_col.cast(pa.binary())

            field_metadata = {}
            if crs is not None:
                try:
                    crs_json = crs.to_json_dict()
                except AttributeError:
                    crs_json = None
                if crs_json is not None:
                    import json

                    field_metadata[b"ARROW:extension:metadata"] = json.dumps(
                        {"crs": crs_json}
                    ).encode()
            field_metadata[b"ARROW:extension:name"] = b"geoarrow.wkb"
            field = pa.field(field_name, pa.binary(), nullable=True, metadata=field_metadata)
            record_dispatch_event(
                surface="vibespatial.io.wkb",
                operation="encode_to_parquet",
                implementation="device_wkb_encode",
                reason="direct device-owned WKB encode for parquet write",
                selected=ExecutionMode.GPU,
            )
            return (field, wkb_arr, ExecutionMode.GPU) if return_mode else (field, wkb_arr)
        except Exception:
            pass

    device_source = owned.residency is Residency.DEVICE or owned.device_state is not None
    if device_source:
        record_fallback_event(
            surface="vibespatial.io.wkb",
            reason="GPU WKB encode unavailable; falling back to host-side row-by-row WKB encode",
            detail="device WKB encoder declined the owned carrier",
            selected=ExecutionMode.CPU,
            pipeline="io/wkb_encode",
            d2h_transfer=True,
        )
    else:
        record_dispatch_event(
            surface="vibespatial.io.wkb",
            operation="encode_to_parquet",
            implementation="host_owned_wkb_encode",
            reason="host-owned geometry uses the native typed-buffer WKB terminal encoder",
            selected=ExecutionMode.CPU,
        )
    import pyarrow as pa

    owned._ensure_host_state(preserve_indexed_view=True)
    wkb_list: list[bytes | None] = []
    for row in range(owned.row_count):
        if not bool(owned.validity[row]):
            wkb_list.append(None)
            continue
        family = TAG_FAMILIES[int(owned.tags[row])]
        buf = owned.families[family]
        frow = int(owned.family_row_offsets[row])
        wkb_list.append(_encode_family_row_wkb(family, buf, frow))
    wkb_arr = pa.array(wkb_list, type=pa.binary())
    field_metadata = {}
    if crs is not None:
        try:
            crs_json = crs.to_json_dict()
        except AttributeError:
            crs_json = None
        if crs_json is not None:
            import json

            field_metadata[b"ARROW:extension:metadata"] = json.dumps({"crs": crs_json}).encode()
    field_metadata[b"ARROW:extension:name"] = b"geoarrow.wkb"
    field = pa.field(field_name, pa.binary(), nullable=True, metadata=field_metadata)
    result = (field, wkb_arr)
    return (*result, ExecutionMode.CPU) if return_mode else result


def _encode_family_row_wkb(
    family: GeometryFamily,
    buf: FamilyGeometryBuffer,
    frow: int,
) -> bytes:
    """Encode a single geometry row to WKB from owned coordinate buffers."""
    wkb_type = WKB_TYPE_IDS[family]

    if family is GeometryFamily.POINT:
        if bool(buf.empty_mask[frow]):
            nan = float("nan")
            return struct.pack("<BIdd", 1, wkb_type, nan, nan)
        start = int(buf.geometry_offsets[frow])
        x = float(buf.x[start])
        y = float(buf.y[start])
        return struct.pack("<BIdd", 1, wkb_type, x, y)

    if family is GeometryFamily.LINESTRING:
        s = int(buf.geometry_offsets[frow])
        e = int(buf.geometry_offsets[frow + 1])
        npts = e - s
        header = struct.pack("<BII", 1, wkb_type, npts)
        coords = b"".join(struct.pack("<dd", float(buf.x[i]), float(buf.y[i])) for i in range(s, e))
        return header + coords

    if family is GeometryFamily.POLYGON:
        ring_s = int(buf.geometry_offsets[frow])
        ring_e = int(buf.geometry_offsets[frow + 1])
        nrings = ring_e - ring_s
        header = struct.pack("<BII", 1, wkb_type, nrings)
        rings = b""
        for r in range(ring_s, ring_e):
            cs = int(buf.ring_offsets[r])
            ce = int(buf.ring_offsets[r + 1])
            npts = ce - cs
            rings += struct.pack("<I", npts)
            rings += b"".join(
                struct.pack("<dd", float(buf.x[i]), float(buf.y[i])) for i in range(cs, ce)
            )
        return header + rings

    if family is GeometryFamily.MULTIPOINT:
        s = int(buf.geometry_offsets[frow])
        e = int(buf.geometry_offsets[frow + 1])
        npts = e - s
        header = struct.pack("<BII", 1, wkb_type, npts)
        points = b"".join(
            struct.pack(
                "<BIdd", 1, WKB_TYPE_IDS[GeometryFamily.POINT], float(buf.x[i]), float(buf.y[i])
            )
            for i in range(s, e)
        )
        return header + points

    if family is GeometryFamily.MULTILINESTRING:
        part_s = int(buf.geometry_offsets[frow])
        part_e = int(buf.geometry_offsets[frow + 1])
        nparts = part_e - part_s
        header = struct.pack("<BII", 1, wkb_type, nparts)
        lines = b""
        for p in range(part_s, part_e):
            cs = int(buf.part_offsets[p])
            ce = int(buf.part_offsets[p + 1])
            npts = ce - cs
            lines += struct.pack("<BII", 1, WKB_TYPE_IDS[GeometryFamily.LINESTRING], npts)
            lines += b"".join(
                struct.pack("<dd", float(buf.x[i]), float(buf.y[i])) for i in range(cs, ce)
            )
        return header + lines

    if family is GeometryFamily.MULTIPOLYGON:
        part_s = int(buf.geometry_offsets[frow])
        part_e = int(buf.geometry_offsets[frow + 1])
        nparts = part_e - part_s
        header = struct.pack("<BII", 1, wkb_type, nparts)
        polygons = b""
        for p in range(part_s, part_e):
            ring_s = int(buf.part_offsets[p])
            ring_e = int(buf.part_offsets[p + 1])
            nrings = ring_e - ring_s
            polygons += struct.pack("<BII", 1, WKB_TYPE_IDS[GeometryFamily.POLYGON], nrings)
            for r in range(ring_s, ring_e):
                cs = int(buf.ring_offsets[r])
                ce = int(buf.ring_offsets[r + 1])
                npts = ce - cs
                polygons += struct.pack("<I", npts)
                polygons += b"".join(
                    struct.pack("<dd", float(buf.x[i]), float(buf.y[i])) for i in range(cs, ce)
                )
        return header + polygons

    raise ValueError(f"Unsupported geometry family for WKB encode: {family}")


def encode_owned_wkb_device(
    owned: OwnedGeometryArray,
):
    """Encode OwnedGeometryArray to WKB as a device-resident pylibcudf Column.

    Zero-copy: coordinates stay on device, WKB is produced on device,
    result stays on device.  Raises if GPU is unavailable.
    """
    return _encode_owned_wkb_column_device(owned)
