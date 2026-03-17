from __future__ import annotations

import struct
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

import numpy as np

from vibespatial.dispatch import record_dispatch_event
from vibespatial.fallbacks import record_fallback_event
from vibespatial.geometry_buffers import GeometryFamily, get_geometry_buffer_schema
from vibespatial.io_support import IOFormat, IOOperation, IOPathKind, plan_io_support
from vibespatial.owned_geometry import (
    DiagnosticKind,
    FAMILY_TAGS,
    TAG_FAMILIES,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
    from_wkb,
)
from vibespatial.device_geometry_array import DeviceGeometryArray
from vibespatial.residency import Residency
from vibespatial.runtime import ExecutionMode
from vibespatial.cuda_runtime import KERNEL_PARAM_I32, KERNEL_PARAM_PTR, get_cuda_runtime, make_kernel_cache_key
from vibespatial.cccl_primitives import exclusive_sum
from vibespatial.cccl_precompile import request_warmup

request_warmup(["exclusive_scan_i32", "exclusive_scan_i64"])

from vibespatial.nvrtc_precompile import request_nvrtc_warmup as _request_nvrtc_warmup  # noqa: E402

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

_WKB_ENCODE_KERNEL_SOURCE = r"""
extern "C" {

__device__ inline void write_u32_le(unsigned char* dst, unsigned int value) {
    dst[0] = (unsigned char)(value & 0xffu);
    dst[1] = (unsigned char)((value >> 8) & 0xffu);
    dst[2] = (unsigned char)((value >> 16) & 0xffu);
    dst[3] = (unsigned char)((value >> 24) & 0xffu);
}

__device__ inline void write_f64_le(unsigned char* dst, double value) {
    unsigned long long bits = *reinterpret_cast<unsigned long long*>(&value);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        dst[i] = (unsigned char)((bits >> (8 * i)) & 0xffull);
    }
}

__global__ void write_point_wkb(
    const int* row_indexes,
    const int* family_rows,
    const double* x,
    const double* y,
    const int* row_offsets,
    unsigned char* payload,
    int count
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= count) return;
    int row = row_indexes[tid];
    int family_row = family_rows[tid];
    unsigned char* out = payload + row_offsets[row];
    out[0] = 1;
    write_u32_le(out + 1, 1u);
    write_f64_le(out + 5, x[family_row]);
    write_f64_le(out + 13, y[family_row]);
}

__global__ void write_linestring_wkb(
    const int* row_indexes,
    const int* family_rows,
    const int* geometry_offsets,
    const double* x,
    const double* y,
    const int* row_offsets,
    unsigned char* payload,
    int count
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= count) return;
    int row = row_indexes[tid];
    int family_row = family_rows[tid];
    int start = geometry_offsets[family_row];
    int stop = geometry_offsets[family_row + 1];
    unsigned char* out = payload + row_offsets[row];
    out[0] = 1;
    write_u32_le(out + 1, 2u);
    write_u32_le(out + 5, (unsigned int)(stop - start));
    out += 9;
    for (int i = start; i < stop; ++i) {
        write_f64_le(out, x[i]);
        write_f64_le(out + 8, y[i]);
        out += 16;
    }
}

__global__ void write_polygon_wkb(
    const int* row_indexes,
    const int* family_rows,
    const int* geometry_offsets,
    const int* ring_offsets,
    const double* x,
    const double* y,
    const int* row_offsets,
    unsigned char* payload,
    int count
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= count) return;
    int row = row_indexes[tid];
    int family_row = family_rows[tid];
    int ring_start = geometry_offsets[family_row];
    int ring_stop = geometry_offsets[family_row + 1];
    unsigned char* out = payload + row_offsets[row];
    out[0] = 1;
    write_u32_le(out + 1, 3u);
    write_u32_le(out + 5, (unsigned int)(ring_stop - ring_start));
    out += 9;
    for (int ring = ring_start; ring < ring_stop; ++ring) {
        int coord_start = ring_offsets[ring];
        int coord_stop = ring_offsets[ring + 1];
        write_u32_le(out, (unsigned int)(coord_stop - coord_start));
        out += 4;
        for (int i = coord_start; i < coord_stop; ++i) {
            write_f64_le(out, x[i]);
            write_f64_le(out + 8, y[i]);
            out += 16;
        }
    }
}

__global__ void write_multipoint_wkb(
    const int* row_indexes,
    const int* family_rows,
    const int* geometry_offsets,
    const double* x,
    const double* y,
    const int* row_offsets,
    unsigned char* payload,
    int count
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= count) return;
    int row = row_indexes[tid];
    int family_row = family_rows[tid];
    int start = geometry_offsets[family_row];
    int stop = geometry_offsets[family_row + 1];
    unsigned char* out = payload + row_offsets[row];
    out[0] = 1;
    write_u32_le(out + 1, 4u);
    write_u32_le(out + 5, (unsigned int)(stop - start));
    out += 9;
    for (int i = start; i < stop; ++i) {
        out[0] = 1;
        write_u32_le(out + 1, 1u);
        write_f64_le(out + 5, x[i]);
        write_f64_le(out + 13, y[i]);
        out += 21;
    }
}

__global__ void write_multilinestring_wkb(
    const int* row_indexes,
    const int* family_rows,
    const int* geometry_offsets,
    const int* part_offsets,
    const double* x,
    const double* y,
    const int* row_offsets,
    unsigned char* payload,
    int count
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= count) return;
    int row = row_indexes[tid];
    int family_row = family_rows[tid];
    int part_start = geometry_offsets[family_row];
    int part_stop = geometry_offsets[family_row + 1];
    unsigned char* out = payload + row_offsets[row];
    out[0] = 1;
    write_u32_le(out + 1, 5u);
    write_u32_le(out + 5, (unsigned int)(part_stop - part_start));
    out += 9;
    for (int part = part_start; part < part_stop; ++part) {
        int coord_start = part_offsets[part];
        int coord_stop = part_offsets[part + 1];
        out[0] = 1;
        write_u32_le(out + 1, 2u);
        write_u32_le(out + 5, (unsigned int)(coord_stop - coord_start));
        out += 9;
        for (int i = coord_start; i < coord_stop; ++i) {
            write_f64_le(out, x[i]);
            write_f64_le(out + 8, y[i]);
            out += 16;
        }
    }
}

__global__ void write_multipolygon_wkb(
    const int* row_indexes,
    const int* family_rows,
    const int* geometry_offsets,
    const int* part_offsets,
    const int* ring_offsets,
    const double* x,
    const double* y,
    const int* row_offsets,
    unsigned char* payload,
    int count
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= count) return;
    int row = row_indexes[tid];
    int family_row = family_rows[tid];
    int poly_start = geometry_offsets[family_row];
    int poly_stop = geometry_offsets[family_row + 1];
    unsigned char* out = payload + row_offsets[row];
    out[0] = 1;
    write_u32_le(out + 1, 6u);
    write_u32_le(out + 5, (unsigned int)(poly_stop - poly_start));
    out += 9;
    for (int poly = poly_start; poly < poly_stop; ++poly) {
        int ring_start = part_offsets[poly];
        int ring_stop = part_offsets[poly + 1];
        out[0] = 1;
        write_u32_le(out + 1, 3u);
        write_u32_le(out + 5, (unsigned int)(ring_stop - ring_start));
        out += 9;
        for (int ring = ring_start; ring < ring_stop; ++ring) {
            int coord_start = ring_offsets[ring];
            int coord_stop = ring_offsets[ring + 1];
            write_u32_le(out, (unsigned int)(coord_stop - coord_start));
            out += 4;
            for (int i = coord_start; i < coord_stop; ++i) {
                write_f64_le(out, x[i]);
                write_f64_le(out + 8, y[i]);
                out += 16;
            }
        }
    }
}

}
"""

_WKB_ENCODE_KERNEL_NAMES = (
    "write_point_wkb",
    "write_linestring_wkb",
    "write_polygon_wkb",
    "write_multipoint_wkb",
    "write_multilinestring_wkb",
    "write_multipolygon_wkb",
)

_request_nvrtc_warmup([
    ("wkb-encode", _WKB_ENCODE_KERNEL_SOURCE, _WKB_ENCODE_KERNEL_NAMES),
])


def has_pyarrow_support() -> bool:
    return find_spec("pyarrow") is not None


def has_pylibcudf_support() -> bool:
    return find_spec("pylibcudf") is not None

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
    family_mask = (owned.tags == FAMILY_TAGS[family]) & owned.validity
    row_indexes = np.flatnonzero(family_mask).astype(np.int32, copy=False)
    family_rows = owned.family_row_offsets[row_indexes].astype(np.int32, copy=False)
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
            (ptr(row_indexes), ptr(family_rows), ptr(device_buffer.x), ptr(device_buffer.y), ptr(row_offsets), ptr(payload), count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
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
    total_bytes = int(cp.asnumpy(offsets[-1])) if row_count else 0
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

    null_count = int((~owned.validity).sum())
    if null_count:
        validity_bytes = np.packbits(owned.validity.astype(np.uint8), bitorder="little")
        validity_mask = cp.asarray(validity_bytes.view(np.uint8))
        column = column.with_mask(plc.gpumemoryview(validity_mask), null_count)
    return column

def _device_geoarrow_fast_path_reason_owned(owned: OwnedGeometryArray) -> str | None:
    if owned.row_count == 0:
        return "empty geometry column requires upstream GeoArrow constructor semantics"
    if not bool(owned.validity.any()):
        return "all-missing geometry column requires upstream GeoArrow constructor semantics"
    for buf in owned.families.values():
        if bool(buf.empty_mask.any()):
            return "empty geometry rows require upstream GeoArrow constructor semantics"
    try:
        _homogeneous_family(owned)
    except ValueError as exc:
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

    null_count = int((~owned.validity).sum())
    if null_count == 0:
        return None, 0
    validity_bytes = np.packbits(owned.validity.astype(np.uint8), bitorder="little")
    return plc.gpumemoryview(cp.asarray(validity_bytes.view(np.uint8))), null_count


def _device_point_values_column(x_device, y_device):
    import pylibcudf as plc

    x_col = plc.Column.from_cuda_array_interface(x_device)
    y_col = plc.Column.from_cuda_array_interface(y_device)
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
        x_full = cp.zeros(row_count, dtype=cp.float64)
        y_full = cp.zeros(row_count, dtype=cp.float64)
        valid_mask = state.validity
        x_full[valid_mask] = device_buffer.x
        y_full[valid_mask] = device_buffer.y
        column = _device_point_values_column(x_full, y_full)
        if null_count:
            column = column.with_mask(mask, null_count)
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
) -> bool:
    import json

    import pandas as pd
    import pyarrow as pa
    import pylibcudf as plc

    from vibespatial.api.io.arrow import _create_metadata, _encode_metadata

    # Extract recognized kwargs; only fall back for truly unrecognized ones.
    _RECOGNIZED_KWARGS = {"row_group_size", "max_page_size"}
    recognized_kwargs = {k: v for k, v in kwargs.items() if k in _RECOGNIZED_KWARGS}
    unrecognized_kwargs = {k: v for k, v in kwargs.items() if k not in _RECOGNIZED_KWARGS}
    if unrecognized_kwargs:
        return False

    geometry_columns = list(geometry_columns)
    geometry_arrays = [df[col].array for col in geometry_columns]
    if not geometry_arrays or not all(isinstance(arr, DeviceGeometryArray) for arr in geometry_arrays):
        return False

    owned_by_name = {col: df[col].array.to_owned() for col in geometry_columns}
    if not all(owned.residency is Residency.DEVICE and owned.device_state is not None for owned in owned_by_name.values()):
        return False
    if not has_pylibcudf_support():
        return False

    geometry_columns_set = set(geometry_columns)
    df_attr = pd.DataFrame(
        {
            col: (None if col in geometry_columns_set else df[col])
            for col in df.columns
        },
        index=df.index,
    )
    host_table = pa.Table.from_pandas(df_attr, preserve_index=index)
    table_columns = []
    geometry_encoding_dict = {}

    for column_name in host_table.column_names:
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
            return False
        from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds

        # Use the primary geometry column (first) for the covering bbox.
        primary_owned = owned_by_name[geometry_columns[0]]
        bounds = compute_geometry_bounds(
            primary_owned,
            dispatch_mode=ExecutionMode.GPU,
        )
        # bounds is (N, 4) float64 numpy: [xmin, ymin, xmax, ymax]
        d_xmin = _cp.asarray(np.ascontiguousarray(bounds[:, 0]))
        d_ymin = _cp.asarray(np.ascontiguousarray(bounds[:, 1]))
        d_xmax = _cp.asarray(np.ascontiguousarray(bounds[:, 2]))
        d_ymax = _cp.asarray(np.ascontiguousarray(bounds[:, 3]))
        bbox_children = [
            plc.Column.from_cuda_array_interface(d_xmin),
            plc.Column.from_cuda_array_interface(d_ymin),
            plc.Column.from_cuda_array_interface(d_xmax),
            plc.Column.from_cuda_array_interface(d_ymax),
        ]
        bbox_struct = plc.Column.struct_from_children(bbox_children)
        table_columns.append(bbox_struct)
        bbox_column_names = ["bbox"]

    all_column_names = list(host_table.column_names) + bbox_column_names
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

    builder = plc.io.parquet.ParquetWriterOptions.builder(
        plc.io.types.SinkInfo([str(path)]),
        plc_table,
    )
    builder.metadata(metadata)
    builder.key_value_metadata([footer_metadata])
    builder.write_arrow_schema(True)
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
    return True

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


def _scan_wkb_value(value: bytes) -> tuple[GeometryFamily | None, str | None]:
    if len(value) < 5:
        return None, "buffer shorter than WKB header"
    byteorder = value[0]
    if byteorder != 1:
        return None, "native WKB fast path currently supports only little-endian input"
    type_id = int.from_bytes(value[1:5], "little")
    family = WKB_ID_FAMILIES.get(type_id)
    if family is None:
        return None, f"unsupported WKB geometry type id {type_id}"
    return family, None


def plan_wkb_partition(
    values: list[bytes | str | None] | tuple[bytes | str | None, ...],
) -> WKBPartitionPlan:
    family_counts = {family.value: 0 for family in GeometryFamily}
    fallback_indexes: list[int] = []
    null_rows = 0
    valid_rows = 0
    native_rows = 0
    for index, value in enumerate(values):
        normalized = _normalize_wkb_value(value)
        if normalized is None:
            null_rows += 1
            continue
        valid_rows += 1
        family, reason = _scan_wkb_value(normalized)
        if reason is not None:
            fallback_indexes.append(index)
            continue
        assert family is not None
        family_counts[family.value] += 1
        native_rows += 1
    return WKBPartitionPlan(
        total_rows=len(values),
        valid_rows=valid_rows,
        null_rows=null_rows,
        native_rows=native_rows,
        fallback_rows=len(fallback_indexes),
        family_counts=family_counts,
        fallback_indexes=tuple(fallback_indexes),
        reason=(
            "Use one WKB header scan to separate native little-endian 2D families from the "
            "explicit fallback pool before decode or encode work begins."
        ),
    )


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
        coords = list(geometry.coords) if family is GeometryFamily.LINESTRING else [(float(p.x), float(p.y)) for p in geometry.geoms]
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
            for part in geometry.geoms:
                state["part_offsets"].append(len(state["x_payload"]))
                coords = list(part.coords)
                state["x_payload"].extend([float(x) for x, _ in coords])
                state["y_payload"].extend([float(y) for _, y in coords])
    elif family is GeometryFamily.MULTIPOLYGON:
        state["geometry_offsets"].append(len(state["part_offsets"]))
        if not geometry.is_empty:
            for polygon in geometry.geoms:
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
    empty_mask = np.isnan(x) | np.isnan(y)
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
    partition_plan = plan_wkb_partition(normalized_values)
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
        family, scan_reason = _scan_wkb_value(value)
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
            family = {
                "Point": GeometryFamily.POINT,
                "LineString": GeometryFamily.LINESTRING,
                "Polygon": GeometryFamily.POLYGON,
                "MultiPoint": GeometryFamily.MULTIPOINT,
                "MultiLineString": GeometryFamily.MULTILINESTRING,
                "MultiPolygon": GeometryFamily.MULTIPOLYGON,
            }[geometry.geom_type]
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
    empty_mask = np.isnan(x) | np.isnan(y)
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
    ring_start = int(buffer.geometry_offsets[row])
    ring_end = int(buffer.geometry_offsets[row + 1])
    size = 9
    ring_ranges: list[tuple[int, int]] = []
    for ring_index in range(ring_start, ring_end):
        coord_start = int(buffer.ring_offsets[ring_index])
        coord_end = int(buffer.ring_offsets[ring_index + 1])
        ring_ranges.append((coord_start, coord_end))
        size += 4 + ((coord_end - coord_start) * 16)
    payload = bytearray(size)
    payload[0] = 1
    payload[1:5] = WKB_TYPE_IDS[GeometryFamily.POLYGON].to_bytes(4, "little")
    payload[5:9] = len(ring_ranges).to_bytes(4, "little")
    cursor = 9
    for coord_start, coord_end in ring_ranges:
        count = coord_end - coord_start
        payload[cursor : cursor + 4] = count.to_bytes(4, "little")
        cursor += 4
        for x, y in zip(buffer.x[coord_start:coord_end], buffer.y[coord_start:coord_end], strict=True):
            struct.pack_into("<dd", payload, cursor, float(x), float(y))
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
    polygon_start = int(buffer.geometry_offsets[row])
    polygon_end = int(buffer.geometry_offsets[row + 1])
    polygon_specs: list[list[tuple[int, int]]] = []
    size = 9
    for polygon_index in range(polygon_start, polygon_end):
        ring_start = int(buffer.part_offsets[polygon_index])
        ring_end = int(buffer.part_offsets[polygon_index + 1])
        ring_ranges: list[tuple[int, int]] = []
        polygon_size = 9
        for ring_index in range(ring_start, ring_end):
            coord_start = int(buffer.ring_offsets[ring_index])
            coord_end = int(buffer.ring_offsets[ring_index + 1])
            ring_ranges.append((coord_start, coord_end))
            polygon_size += 4 + ((coord_end - coord_start) * 16)
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
        for coord_start, coord_end in ring_ranges:
            count = coord_end - coord_start
            payload[cursor : cursor + 4] = count.to_bytes(4, "little")
            cursor += 4
            for x, y in zip(buffer.x[coord_start:coord_end], buffer.y[coord_start:coord_end], strict=True):
                struct.pack_into("<dd", payload, cursor, float(x), float(y))
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
        record_fallback_event(
            surface="vibespatial.io.wkb",
            reason="explicit CPU fallback for unsupported or malformed WKB rows",
            detail=f"{partition_plan.fallback_rows} rows entered the fallback pool during staged decode",
            selected=ExecutionMode.CPU,
        )
    return array


def decode_wkb_arrow_array_owned(array, *, on_invalid: str = "raise") -> OwnedGeometryArray:
    gpu_result = _try_gpu_wkb_arrow_decode(array)
    if gpu_result is not None:
        return gpu_result
    point_fast = _decode_arrow_wkb_point_fast(array)
    if point_fast is not None:
        return point_fast
    linestring_uniform_fast = _decode_arrow_wkb_linestring_uniform_fast(array)
    if linestring_uniform_fast is not None:
        return linestring_uniform_fast
    polygon_uniform_fast = _decode_arrow_wkb_polygon_uniform_fast(array)
    if polygon_uniform_fast is not None:
        return polygon_uniform_fast
    linestring_fast = _decode_arrow_wkb_linestring_fast(array)
    if linestring_fast is not None:
        return linestring_fast
    polygon_fast = _decode_arrow_wkb_polygon_fast(array)
    if polygon_fast is not None:
        return polygon_fast
    values = np.asarray(array.to_numpy(zero_copy_only=False), dtype=object)
    return decode_wkb_owned(list(values), on_invalid=on_invalid)


def _try_gpu_wkb_arrow_decode(array) -> OwnedGeometryArray | None:
    """Attempt GPU WKB decode of a PyArrow binary/large_binary array via pylibcudf."""
    runtime = get_cuda_runtime()
    if not runtime.available():
        return None
    try:
        import pyarrow as pa
        import pylibcudf as plc
        from vibespatial.io_pylibcudf import _decode_pylibcudf_wkb_general_column_to_owned

        # pylibcudf does not support Arrow binary/large_binary types, but the
        # memory layout is identical to string/large_string (offsets + bytes).
        # Zero-copy reinterpret so plc.Column.from_arrow succeeds.
        if pa.types.is_binary(array.type) or pa.types.is_large_binary(array.type):
            target = pa.string() if pa.types.is_binary(array.type) else pa.large_string()
            array = pa.Array.from_buffers(
                target, len(array), array.buffers(), null_count=array.null_count,
            )

        plc_column = plc.Column.from_arrow(array)
        return _decode_pylibcudf_wkb_general_column_to_owned(plc_column)
    except (ImportError, NotImplementedError):
        return None
    except Exception:
        return None


def _try_gpu_wkb_encode(
    array: OwnedGeometryArray,
    *,
    hex_output: bool = False,
) -> list[bytes | str | None] | None:
    """Attempt GPU-accelerated WKB encode. Returns list[bytes|str|None] or None on failure."""
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
        )
    return values

def _homogeneous_family(array: OwnedGeometryArray):
    valid_tags = array.tags[array.validity]
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
    """Encode OwnedGeometryArray to WKB pyarrow array from owned buffers.

    This avoids the np.asarray(DGA) → Shapely materialization path by
    encoding WKB directly from owned coordinate buffers where possible.
    For complex cases, falls back to Shapely's to_wkb.
    """
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
        x = float(buf.x[frow])
        y = float(buf.y[frow])
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

