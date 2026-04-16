from __future__ import annotations

import struct
from dataclasses import dataclass
from importlib.util import find_spec
from types import SimpleNamespace
from typing import Any

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    get_cuda_runtime,
    make_kernel_cache_key,
)
from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.cccl_primitives import exclusive_sum
from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema
from vibespatial.geometry.device_array import DeviceGeometryArray
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    TAG_FAMILIES,
    DiagnosticKind,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
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


_request_nvrtc_warmup([
    ("wkb-encode", _WKB_ENCODE_KERNEL_SOURCE, _WKB_ENCODE_KERNEL_NAMES),
])


@dataclass(frozen=True)
class _GpuWkbDecodeAttempt:
    result: OwnedGeometryArray | None
    fallback_detail: str | None = None


@dataclass(frozen=True)
class _NativeDeviceWriteStatus:
    written: bool
    fallback_detail: str | None = None
    compatibility_detail: str | None = None


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
    if owned.device_state is not None:
        runtime = get_cuda_runtime()
        return (
            runtime.copy_device_to_host(owned.device_state.validity),
            runtime.copy_device_to_host(owned.device_state.tags),
            runtime.copy_device_to_host(owned.device_state.family_row_offsets),
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
) -> tuple[np.ndarray, np.ndarray]:
    validity, tags, family_row_offsets = _authoritative_host_metadata(owned)
    family_mask = (tags == FAMILY_TAGS[family]) & validity
    row_indexes = np.flatnonzero(family_mask).astype(np.int32, copy=False)
    family_rows = family_row_offsets[row_indexes].astype(np.int32, copy=False)
    return row_indexes, family_rows


def _device_wkb_lengths_for_family(owned: OwnedGeometryArray, family: GeometryFamily):
    import cupy as cp

    state = owned._ensure_device_state()
    device_buffer = state.families[family]
    _row_indexes, family_rows_host = _device_family_row_selection(owned, family)
    if family_rows_host.size == 0:
        return cp.zeros(0, dtype=cp.int32), family_rows_host

    family_rows = cp.asarray(family_rows_host)
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
        lengths = (9 + 4 * (ring_stop - ring_start) + 16 * (coord_stop - coord_start)).astype(cp.int32, copy=False)
    elif family is GeometryFamily.MULTIPOINT:
        counts = geometry_offsets[family_rows + 1] - geometry_offsets[family_rows]
        lengths = (9 + 21 * counts).astype(cp.int32, copy=False)
    elif family is GeometryFamily.MULTILINESTRING:
        part_offsets = device_buffer.part_offsets
        part_start = geometry_offsets[family_rows]
        part_stop = geometry_offsets[family_rows + 1]
        coord_start = part_offsets[part_start]
        coord_stop = part_offsets[part_stop]
        lengths = (9 + 9 * (part_stop - part_start) + 16 * (coord_stop - coord_start)).astype(cp.int32, copy=False)
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
            9 + 9 * (poly_stop - poly_start) + 4 * (ring_stop - ring_start) + 16 * (coord_stop - coord_start)
        ).astype(cp.int32, copy=False)
    else:  # pragma: no cover - exhaustive today
        raise ValueError(f"Unsupported geometry family for device WKB encode: {family}")
    return lengths, family_rows_host


def _launch_device_wkb_write_kernel(
    family: GeometryFamily,
    *,
    owned: OwnedGeometryArray,
    row_indexes_host: np.ndarray,
    family_rows_host: np.ndarray,
    row_offsets,
    payload,
) -> None:
    import cupy as cp

    count = int(row_indexes_host.size)
    if count == 0:
        return

    runtime = get_cuda_runtime()
    kernels = _wkb_encode_kernels()
    state = owned._ensure_device_state()
    device_buffer = state.families[family]
    row_indexes = cp.asarray(row_indexes_host)
    family_rows = cp.asarray(family_rows_host)
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
            (ptr(row_indexes), ptr(family_rows), ptr(device_buffer.geometry_offsets), ptr(device_buffer.x), ptr(device_buffer.y), ptr(row_offsets), ptr(payload), count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        )
    elif family is GeometryFamily.POLYGON:
        kernel = kernels["write_polygon_wkb"]
        params = (
            (ptr(row_indexes), ptr(family_rows), ptr(device_buffer.geometry_offsets), ptr(device_buffer.ring_offsets), ptr(device_buffer.x), ptr(device_buffer.y), ptr(row_offsets), ptr(payload), count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        )
    elif family is GeometryFamily.MULTIPOINT:
        kernel = kernels["write_multipoint_wkb"]
        params = (
            (ptr(row_indexes), ptr(family_rows), ptr(device_buffer.geometry_offsets), ptr(device_buffer.x), ptr(device_buffer.y), ptr(row_offsets), ptr(payload), count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        )
    elif family is GeometryFamily.MULTILINESTRING:
        kernel = kernels["write_multilinestring_wkb"]
        params = (
            (ptr(row_indexes), ptr(family_rows), ptr(device_buffer.geometry_offsets), ptr(device_buffer.part_offsets), ptr(device_buffer.x), ptr(device_buffer.y), ptr(row_offsets), ptr(payload), count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        )
    elif family is GeometryFamily.MULTIPOLYGON:
        kernel = kernels["write_multipolygon_wkb"]
        params = (
            (ptr(row_indexes), ptr(family_rows), ptr(device_buffer.geometry_offsets), ptr(device_buffer.part_offsets), ptr(device_buffer.ring_offsets), ptr(device_buffer.x), ptr(device_buffer.y), ptr(row_offsets), ptr(payload), count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
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
    for family, (row_indexes_host, _family_rows_host) in family_selections.items():
        n_rows_family = row_indexes_host.shape[0]
        buf = state.families[family]
        n_coords = buf.x.shape[0]

        if family is GeometryFamily.POINT:
            total += 21 * n_rows_family
        elif family is GeometryFamily.LINESTRING:
            total += 9 * n_rows_family + 16 * n_coords
        elif family is GeometryFamily.POLYGON:
            n_rings = buf.ring_offsets.shape[0] - 1 if buf.ring_offsets is not None else 0
            total += 9 * n_rows_family + 4 * n_rings + 16 * n_coords
        elif family is GeometryFamily.MULTIPOINT:
            total += 9 * n_rows_family + 21 * n_coords
        elif family is GeometryFamily.MULTILINESTRING:
            n_parts = buf.part_offsets.shape[0] - 1 if buf.part_offsets is not None else 0
            total += 9 * n_rows_family + 9 * n_parts + 16 * n_coords
        elif family is GeometryFamily.MULTIPOLYGON:
            n_parts = buf.part_offsets.shape[0] - 1 if buf.part_offsets is not None else 0
            n_rings = buf.ring_offsets.shape[0] - 1 if buf.ring_offsets is not None else 0
            total += 9 * n_rows_family + 9 * n_parts + 4 * n_rings + 16 * n_coords
    return total


def _encode_owned_wkb_column_device(owned: OwnedGeometryArray):
    import cupy as cp
    import pylibcudf as plc

    state = owned._ensure_device_state()
    row_count = owned.row_count
    lengths = cp.zeros(row_count, dtype=cp.int32)
    family_selections: dict[GeometryFamily, tuple[np.ndarray, np.ndarray]] = {}

    for family in state.families:
        row_indexes_host, _family_rows_host = _device_family_row_selection(owned, family)
        family_lengths, family_rows_host = _device_wkb_lengths_for_family(owned, family)
        if row_indexes_host.size:
            lengths[cp.asarray(row_indexes_host)] = family_lengths
            family_selections[family] = (row_indexes_host, family_rows_host)

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

    for family, (row_indexes_host, family_rows_host) in family_selections.items():
        _launch_device_wkb_write_kernel(
            family,
            owned=owned,
            row_indexes_host=row_indexes_host,
            family_rows_host=family_rows_host,
            row_offsets=offsets,
            payload=payload,
        )

    runtime = get_cuda_runtime()
    runtime.synchronize()

    offsets_column = plc.Column.from_cuda_array_interface(offsets)
    column = plc.Column(
        plc.types.DataType(plc.types.TypeId.STRING),
        row_count,
        plc.gpumemoryview(payload),
        None,
        0,
        0,
        [offsets_column],
    )

    validity_mask, null_count = _device_validity_gpumask(owned)
    if null_count:
        column = column.with_mask(validity_mask, null_count)
    return column

def _device_geoarrow_fast_path_reason_owned(owned: OwnedGeometryArray) -> str | None:
    if owned.row_count == 0:
        return "empty geometry column requires upstream GeoArrow constructor semantics"
    try:
        _homogeneous_family(owned)
    except ValueError as exc:
        if str(exc) == "Cannot encode an all-null geometry array to native GeoArrow":
            return "all-missing geometry column requires upstream GeoArrow constructor semantics"
        return str(exc)
    return None


def _device_full_offsets_from_local(owned: OwnedGeometryArray, local_offsets):
    import cupy as cp

    validity = owned._ensure_device_state().validity
    row_count = owned.row_count
    counts = cp.zeros(row_count, dtype=cp.int32)
    if row_count:
        counts[validity] = local_offsets[1:] - local_offsets[:-1]
    full_offsets = cp.empty(row_count + 1, dtype=cp.int32)
    full_offsets[0] = 0
    if row_count:
        full_offsets[1:] = cp.cumsum(counts, dtype=cp.int32)
    return full_offsets


def _device_validity_gpumask(owned: OwnedGeometryArray):
    import cupy as cp
    import pylibcudf as plc

    validity = cp.asarray(owned._ensure_device_state().validity)
    null_count = int(cp.count_nonzero(~validity).item())
    if null_count == 0:
        return None, 0
    validity_bytes = cp.packbits(validity.astype(cp.uint8), bitorder="little")
    return plc.gpumemoryview(validity_bytes.view(cp.uint8)), null_count


def _device_point_values_column(x_device, y_device, *, mask=None, null_count: int = 0):
    import pylibcudf as plc

    x_col = plc.Column.from_cuda_array_interface(x_device)
    y_col = plc.Column.from_cuda_array_interface(y_device)
    if mask is not None and null_count:
        x_col = x_col.with_mask(mask, null_count)
        y_col = y_col.with_mask(mask, null_count)
        return plc.Column.struct_from_children([x_col, y_col]).with_mask(mask, null_count)
    return plc.Column.struct_from_children([x_col, y_col])


def _device_list_column(offsets_device, child_column, size: int):
    import pylibcudf as plc

    offsets_col = plc.Column.from_cuda_array_interface(offsets_device)
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

    family = _homogeneous_family(owned)
    state = owned._ensure_device_state()
    device_buffer = state.families[family]
    mask, null_count = _device_validity_gpumask(owned)

    if family is GeometryFamily.POINT:
        row_count = owned.row_count
        if null_count == 0 and int(cp.count_nonzero(device_buffer.empty_mask).item()) == 0:
            family_rows = cp.asarray(state.family_row_offsets).astype(cp.int32, copy=False)
            coord_indices = device_buffer.geometry_offsets[family_rows]
            column = _device_point_values_column(
                device_buffer.x[coord_indices],
                device_buffer.y[coord_indices],
            )
            return column, "point"
        x_full = cp.full(row_count, cp.nan, dtype=cp.float64)
        y_full = cp.full(row_count, cp.nan, dtype=cp.float64)
        valid_mask = state.validity
        valid_rows = cp.flatnonzero(valid_mask)
        non_empty_mask = ~device_buffer.empty_mask
        if int(cp.count_nonzero(non_empty_mask)) > 0:
            coord_indices = device_buffer.geometry_offsets[:-1][non_empty_mask]
            x_full[valid_rows[non_empty_mask]] = device_buffer.x[coord_indices]
            y_full[valid_rows[non_empty_mask]] = device_buffer.y[coord_indices]
        column = _device_point_values_column(x_full, y_full, mask=mask, null_count=null_count)
        return column, "point"

    point_values = _device_point_values_column(device_buffer.x, device_buffer.y)

    if family is GeometryFamily.LINESTRING:
        full_offsets = _device_full_offsets_from_local(owned, device_buffer.geometry_offsets)
        column = _device_list_column(full_offsets, point_values, owned.row_count)
        if null_count:
            column = column.with_mask(mask, null_count)
        return column, "linestring"

    if family is GeometryFamily.MULTIPOINT:
        full_offsets = _device_full_offsets_from_local(owned, device_buffer.geometry_offsets)
        column = _device_list_column(full_offsets, point_values, owned.row_count)
        if null_count:
            column = column.with_mask(mask, null_count)
        return column, "multipoint"

    if family is GeometryFamily.POLYGON:
        rings = _device_list_column(device_buffer.ring_offsets, point_values, int(device_buffer.ring_offsets.size - 1))
        full_offsets = _device_full_offsets_from_local(owned, device_buffer.geometry_offsets)
        column = _device_list_column(full_offsets, rings, owned.row_count)
        if null_count:
            column = column.with_mask(mask, null_count)
        return column, "polygon"

    if family is GeometryFamily.MULTILINESTRING:
        parts = _device_list_column(device_buffer.part_offsets, point_values, int(device_buffer.part_offsets.size - 1))
        full_offsets = _device_full_offsets_from_local(owned, device_buffer.geometry_offsets)
        column = _device_list_column(full_offsets, parts, owned.row_count)
        if null_count:
            column = column.with_mask(mask, null_count)
        return column, "multilinestring"

    if family is GeometryFamily.MULTIPOLYGON:
        rings = _device_list_column(device_buffer.ring_offsets, point_values, int(device_buffer.ring_offsets.size - 1))
        polygons = _device_list_column(device_buffer.part_offsets, rings, int(device_buffer.part_offsets.size - 1))
        full_offsets = _device_full_offsets_from_local(owned, device_buffer.geometry_offsets)
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

    return getattr(plc.io.types.CompressionType, str(name).upper(), plc.io.types.CompressionType.AUTO)


def _native_parquet_compression_supported(name: str | None) -> bool:
    normalized = None if name is None else str(name).lower()
    return normalized in {None, "snappy", "lz4", "zstd"}


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
        if bufs and len(bufs) > 1 and bufs[1] is not None and hasattr(bufs[1], "__cuda_array_interface__"):
            return plc.Column.from_cuda_array_interface(bufs[1])
    return plc.Column.from_arrow(combined)


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

    df_attr = pd.DataFrame(
        {column_name: attribute_frame[column_name] for column_name in ordered_columns},
        index=attribute_frame.index,
        copy=False,
    )
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


def _write_geoparquet_native_device_payload(
    attribute_frame,
    geometry_owned,
    path,
    *,
    geometry_name,
    geometry_crs,
    index,
    compression,
    geometry_encoding,
    schema_version,
    write_covering_bbox,
    column_order,
    frame_attrs=None,
    **kwargs,
) -> _NativeDeviceWriteStatus:
    import base64
    import json
    import os

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
    if owned is None:
        return _NativeDeviceWriteStatus(written=False)
    if owned.residency is not Residency.DEVICE or owned.device_state is None:
        return _NativeDeviceWriteStatus(written=False)
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
    if not isinstance(path, (str, bytes, os.PathLike)):
        return _NativeDeviceWriteStatus(
            written=False,
            compatibility_detail=(
                "native device GeoParquet payload writer requires a filesystem path sink"
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
    host_table = _build_native_host_attribute_table_from_frame(
        attribute_frame,
        non_geometry_columns,
        index=index,
        pa=pa,
    )
    ordered_column_names = list(column_order)
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
            table_columns.append(_attribute_column_to_plc(host_table[column_name], column_name, plc=plc))

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
            plc.Column.from_cuda_array_interface(d_xmin),
            plc.Column.from_cuda_array_interface(d_ymin),
            plc.Column.from_cuda_array_interface(d_xmax),
            plc.Column.from_cuda_array_interface(d_ymax),
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
                    _homogeneous_family(owned),
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
        for key, value in (host_table.schema.metadata or {}).items()
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

        family = _homogeneous_family(owned)
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
    host_fields = {field.name: field for field in host_table.schema}
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

    schema_metadata = dict(host_table.schema.metadata or {})
    schema_metadata[b"geo"] = _encode_metadata(geo_metadata)
    if frame_attrs:
        schema_metadata[b"PANDAS_ATTRS"] = json.dumps(frame_attrs).encode()
    arrow_schema = pa.schema(schema_fields, metadata=schema_metadata)
    footer_metadata["ARROW:schema"] = base64.b64encode(
        arrow_schema.serialize().to_pybytes()
    ).decode()

    builder = plc.io.parquet.ParquetWriterOptions.builder(
        plc.io.types.SinkInfo([str(path)]),
        plc_table,
    )
    builder.metadata(metadata)
    builder.key_value_metadata([footer_metadata])
    builder.write_arrow_schema(False)
    builder.compression(_compression_type_from_name(compression))
    if "row_group_size" in recognized_kwargs:
        builder.row_group_size_rows(int(recognized_kwargs["row_group_size"]))
    if "max_page_size" in recognized_kwargs:
        builder.max_page_size_bytes(int(recognized_kwargs["max_page_size"]))
    plc.io.parquet.write_parquet(builder.build())
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
    import os

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
    geometry_arrays = [df[col].array for col in geometry_columns]
    if not geometry_arrays or not all(isinstance(arr, DeviceGeometryArray) for arr in geometry_arrays):
        return _NativeDeviceWriteStatus(written=False)

    owned_by_name = {col: df[col].array.to_owned() for col in geometry_columns}
    if not all(owned.residency is Residency.DEVICE and owned.device_state is not None for owned in owned_by_name.values()):
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
                "native device GeoParquet writer does not support "
                f"compression={compression!r}"
            ),
        )
    if not isinstance(path, (str, bytes, os.PathLike)):
        return _NativeDeviceWriteStatus(
            written=False,
            compatibility_detail=(
                "native device GeoParquet writer requires a filesystem path sink"
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
        column_name for column_name in df.columns
        if column_name not in geometry_columns_set
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
            table_columns.append(_attribute_column_to_plc(host_table[column_name], column_name, plc=plc))

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
            plc.Column.from_cuda_array_interface(d_xmin),
            plc.Column.from_cuda_array_interface(d_ymin),
            plc.Column.from_cuda_array_interface(d_xmax),
            plc.Column.from_cuda_array_interface(d_ymax),
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
                    _homogeneous_family(owned_by_name[column_name]),
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

        family = _homogeneous_family(owned_by_name[column_name])
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

    builder = plc.io.parquet.ParquetWriterOptions.builder(
        plc.io.types.SinkInfo([str(path)]),
        plc_table,
    )
    builder.metadata(metadata)
    builder.key_value_metadata([footer_metadata])
    builder.write_arrow_schema(False)
    builder.compression(_compression_type_from_name(compression))
    if "row_group_size" in recognized_kwargs:
        builder.row_group_size_rows(int(recognized_kwargs["row_group_size"]))
    if "max_page_size" in recognized_kwargs:
        builder.max_page_size_bytes(int(recognized_kwargs["max_page_size"]))
    plc.io.parquet.write_parquet(builder.build())
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


def _finalize_wkb_family_buffer(family: GeometryFamily, state: dict[str, Any]) -> FamilyGeometryBuffer:
    x = np.asarray(state["x_payload"], dtype=np.float64)
    y = np.asarray(state["y_payload"], dtype=np.float64)
    geometry_offsets = np.asarray([*state["geometry_offsets"], len(state["x_payload"])], dtype=np.int32)
    part_offsets = None
    ring_offsets = None

    if family is GeometryFamily.POLYGON:
        ring_offsets = np.asarray([*state["ring_offsets"], len(state["x_payload"])], dtype=np.int32)
        geometry_offsets = np.asarray([*state["geometry_offsets"], len(state["ring_offsets"])], dtype=np.int32)
    elif family is GeometryFamily.MULTILINESTRING:
        part_offsets = np.asarray([*state["part_offsets"], len(state["x_payload"])], dtype=np.int32)
        geometry_offsets = np.asarray([*state["geometry_offsets"], len(state["part_offsets"])], dtype=np.int32)
    elif family is GeometryFamily.MULTIPOLYGON:
        part_offsets = np.asarray([*state["part_offsets"], len(state["ring_offsets"])], dtype=np.int32)
        ring_offsets = np.asarray([*state["ring_offsets"], len(state["x_payload"])], dtype=np.int32)
        geometry_offsets = np.asarray([*state["geometry_offsets"], len(state["part_offsets"])], dtype=np.int32)

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
        raise NotImplementedError(
            f"{geom_type} rows fall outside the 2D owned native result model"
        )
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
            coords = np.frombuffer(value, dtype="<f8", offset=cursor, count=point_count * 2).reshape(point_count, 2)
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
            raise ValueError("MultiLineString fast path requires nested little-endian linestring records")
        point_count = int.from_bytes(value[cursor + 5 : cursor + 9], "little")
        coord_bytes = point_count * 16
        end = cursor + 9 + coord_bytes
        if end > len(value):
            raise ValueError("MultiLineString part coordinates overrun buffer")
        state["part_offsets"].append(len(state["x_payload"]))
        if point_count:
            coords = np.frombuffer(value, dtype="<f8", offset=cursor + 9, count=point_count * 2).reshape(point_count, 2)
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
                coords = np.frombuffer(value, dtype="<f8", offset=cursor, count=point_count * 2).reshape(point_count, 2)
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
    array._record(DiagnosticKind.CREATED, "created owned geometry array from staged native WKB decode", visible=True)
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
        return OwnedGeometryArray(validity=validity, tags=tags, family_row_offsets=family_row_offsets, families={})

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
        if data[start] != 1 or int.from_bytes(data[start + 1 : start + 5], "little") != WKB_TYPE_IDS[GeometryFamily.LINESTRING]:
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
            coords = np.frombuffer(data[start + 9 : start + 9 + (point_count * 16)], dtype="<f8", count=point_count * 2).reshape(point_count, 2)
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
        return OwnedGeometryArray(validity=validity, tags=tags, family_row_offsets=family_row_offsets, families={})

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
        return OwnedGeometryArray(validity=validity, tags=tags, family_row_offsets=family_row_offsets, families={})

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
        if data[start] != 1 or int.from_bytes(data[start + 1 : start + 5], "little") != WKB_TYPE_IDS[GeometryFamily.POLYGON]:
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
                coords = np.frombuffer(data[cursor : cursor + (point_count * 16)], dtype="<f8", count=point_count * 2).reshape(point_count, 2)
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
        return OwnedGeometryArray(validity=validity, tags=tags, family_row_offsets=family_row_offsets, families={})

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
            validity=validity, tags=tags,
            family_row_offsets=family_row_offsets, families={},
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
            if data[cursor] != 1 or int.from_bytes(data[cursor + 1 : cursor + 5], "little") != wkb_poly_type:
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
                        dtype="<f8", count=point_count * 2,
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


def _hexify_if_requested(values: list[bytes | None], *, hex_output: bool) -> list[bytes | str | None]:
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
        return (
            float(buffer.x[coord_start]) != float(buffer.x[coord_end - 1])
            or float(buffer.y[coord_start]) != float(buffer.y[coord_end - 1])
        )

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
        for x, y in zip(buffer.x[coord_start:coord_end], buffer.y[coord_start:coord_end], strict=True):
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
        payload[cursor + 1 : cursor + 5] = WKB_TYPE_IDS[GeometryFamily.LINESTRING].to_bytes(4, "little")
        payload[cursor + 5 : cursor + 9] = count.to_bytes(4, "little")
        cursor += 9
        for x, y in zip(buffer.x[coord_start:coord_end], buffer.y[coord_start:coord_end], strict=True):
            struct.pack_into("<dd", payload, cursor, float(x), float(y))
            cursor += 16
    return bytes(payload)


def _pack_multipolygon_wkb(buffer: FamilyGeometryBuffer, row: int) -> bytes:
    def _ring_needs_closure(coord_start: int, coord_end: int) -> bool:
        if coord_end <= coord_start:
            return False
        return (
            float(buffer.x[coord_start]) != float(buffer.x[coord_end - 1])
            or float(buffer.y[coord_start]) != float(buffer.y[coord_end - 1])
        )

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
        payload[cursor + 1 : cursor + 5] = WKB_TYPE_IDS[GeometryFamily.POLYGON].to_bytes(4, "little")
        payload[cursor + 5 : cursor + 9] = len(ring_ranges).to_bytes(4, "little")
        cursor += 9
        for coord_start, coord_end, needs_closure in ring_ranges:
            count = (coord_end - coord_start) + int(needs_closure)
            payload[cursor : cursor + 4] = count.to_bytes(4, "little")
            cursor += 4
            for x, y in zip(buffer.x[coord_start:coord_end], buffer.y[coord_start:coord_end], strict=True):
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
        family_counts={family.value: int((array.tags == FAMILY_TAGS[family]).sum()) for family in GeometryFamily},
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
        import pylibcudf as plc

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

        plc_column = plc.Column.from_arrow(arrow_str)
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
                f"{detail}; pylibcudf WKB decode bridge unavailable: "
                f"{type(exc).__name__}: {exc}"
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
                f"{detail}; pylibcudf WKB decode bridge failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        )


def decode_wkb_arrow_array_owned(array, *, on_invalid: str = "raise") -> OwnedGeometryArray:
    uniform_fast = _try_uniform_arrow_wkb_fast_decode(array)
    if uniform_fast is not None:
        return uniform_fast

    gpu_attempt = _try_gpu_wkb_arrow_decode(array, on_invalid=on_invalid)
    if gpu_attempt.result is not None:
        return gpu_attempt.result
    if gpu_attempt.fallback_detail is not None:
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
        import pylibcudf as plc

        from .pylibcudf import _decode_pylibcudf_wkb_general_column_to_owned

        # pylibcudf does not support Arrow binary/large_binary types, but the
        # memory layout is identical to string/large_string (offsets + bytes).
        # Zero-copy reinterpret so plc.Column.from_arrow succeeds.
        if pa.types.is_binary(array.type) or pa.types.is_large_binary(array.type):
            target = pa.string() if pa.types.is_binary(array.type) else pa.large_string()
            array = pa.Array.from_buffers(
                target, len(array), array.buffers(), null_count=array.null_count,
            )

        plc_column = plc.Column.from_arrow(array)
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
                "GPU Arrow WKB decode bridge unavailable: "
                f"{type(exc).__name__}: {exc}"
            ),
        )
    except Exception as exc:
        return _GpuWkbDecodeAttempt(
            result=None,
            fallback_detail=(
                "GPU Arrow WKB decode bridge failed: "
                f"{type(exc).__name__}: {exc}"
            ),
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
        arrow_col = plc_column.to_arrow()
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
        arrow_col = plc_column.to_arrow()
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
        try:
            import cupy as cp
        except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
            cp = None
    else:
        cp = None
    if cp is not None:
        state = array._ensure_device_state()
        validity = cp.asarray(state.validity)
        valid_count = int(cp.count_nonzero(validity).item())
        if valid_count == 0:
            raise ValueError("Cannot encode an all-null geometry array to native GeoArrow")
        valid_tags = cp.asarray(state.tags)[validity]
        min_tag = int(valid_tags.min().item())
        max_tag = int(valid_tags.max().item())
        if min_tag != max_tag:
            raise ValueError("Native GeoArrow fast path requires a homogeneous geometry family")
        return TAG_FAMILIES[min_tag]
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
        return gpu_result

    # Make one final direct device-encode attempt before surfacing a host
    # fallback.  Strict-native writers must not fail just because the generic
    # helper declined GPU encode on a small batch or because the owned array is
    # still host-resident at the write boundary; _encode_owned_wkb_column_device
    # will materialize device state on demand.
    if strict_native_mode_enabled() or (
        owned.residency is Residency.DEVICE and owned.device_state is not None
    ):
        try:
            import pyarrow as pa

            plc_column = _encode_owned_wkb_column_device(owned)
            arrow_col = plc_column.to_arrow()
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
            return field, wkb_arr
        except Exception:
            pass

    # CPU fallback -- needs host state
    record_fallback_event(
        surface="vibespatial.io.wkb",
        reason="GPU WKB encode unavailable; falling back to host-side row-by-row WKB encode",
        detail="GPU runtime not available or row count below threshold",
        selected=ExecutionMode.CPU,
        pipeline="io/wkb_encode",
        d2h_transfer=True,
    )
    import pyarrow as pa

    owned._ensure_host_state()
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

            field_metadata[b"ARROW:extension:metadata"] = json.dumps(
                {"crs": crs_json}
            ).encode()
    field_metadata[b"ARROW:extension:name"] = b"geoarrow.wkb"
    field = pa.field(field_name, pa.binary(), nullable=True, metadata=field_metadata)
    return field, wkb_arr


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
        coords = b"".join(
            struct.pack("<dd", float(buf.x[i]), float(buf.y[i]))
            for i in range(s, e)
        )
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
                struct.pack("<dd", float(buf.x[i]), float(buf.y[i]))
                for i in range(cs, ce)
            )
        return header + rings

    if family is GeometryFamily.MULTIPOINT:
        s = int(buf.geometry_offsets[frow])
        e = int(buf.geometry_offsets[frow + 1])
        npts = e - s
        header = struct.pack("<BII", 1, wkb_type, npts)
        points = b"".join(
            struct.pack("<BIdd", 1, WKB_TYPE_IDS[GeometryFamily.POINT],
                        float(buf.x[i]), float(buf.y[i]))
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
                struct.pack("<dd", float(buf.x[i]), float(buf.y[i]))
                for i in range(cs, ce)
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
                    struct.pack("<dd", float(buf.x[i]), float(buf.y[i]))
                    for i in range(cs, ce)
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
