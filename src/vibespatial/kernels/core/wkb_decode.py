"""GPU WKB decode kernel pipeline (P1a).

Implements a 5-stage GPU pipeline that reads WKB byte streams directly
on device and produces OwnedGeometryArray results without host round-trips.

Stages:
  1. Header scan kernel -- 1 thread per WKB record, reads endian + type tag
  2. Family partition -- CuPy boolean masks to bucket records by geometry type
  3. Size scan + offset computation -- sizing kernels + CCCL exclusive_sum
  4. Family decode kernels -- per-family coordinate extraction
  5. Assembly -- build OwnedGeometryArray via io_pylibcudf helpers

ADR-0033 dispatch tiers:
  Tier 1 (custom NVRTC) for geometry-specific decode
  Tier 3a (CCCL)        for prefix-sum offset computation
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from vibespatial.cccl_precompile import request_warmup
from vibespatial.cccl_primitives import exclusive_sum
from vibespatial.cuda_runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    get_cuda_runtime,
    make_kernel_cache_key,
)
from vibespatial.geometry_buffers import GeometryFamily
from vibespatial.io_pylibcudf import (
    _build_device_mixed_owned,
    _build_device_single_family_owned,
)
from vibespatial.kernel_registry import register_kernel_variant
from vibespatial.owned_geometry import (
    DeviceFamilyGeometryBuffer,
    OwnedGeometryArray,
)
from vibespatial.precision import KernelClass
from vibespatial.runtime import ExecutionMode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CCCL warmup (ADR-0034)
# ---------------------------------------------------------------------------
request_warmup(["exclusive_scan_i32", "exclusive_scan_i64"])

# ---------------------------------------------------------------------------
# Shared CUDA device helpers for WKB byte reading
# ---------------------------------------------------------------------------

_WKB_DECODE_SHARED_HELPERS = r"""
__device__ inline unsigned int read_u32_le(const unsigned char* src) {
    return (unsigned int)src[0]
         | ((unsigned int)src[1] << 8)
         | ((unsigned int)src[2] << 16)
         | ((unsigned int)src[3] << 24);
}

__device__ inline double read_f64_le(const unsigned char* src) {
    unsigned long long bits = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        bits |= ((unsigned long long)src[i]) << (8 * i);
    }
    return *reinterpret_cast<double*>(&bits);
}
"""

# ---------------------------------------------------------------------------
# Kernel source -- all decode kernels in one compilation unit
# ---------------------------------------------------------------------------

_WKB_DECODE_KERNEL_SOURCE = _WKB_DECODE_SHARED_HELPERS + r"""
extern "C" {

/* ---------- Stage 1: Header scan ----------
 * 1 thread per WKB record. Reads 5-9 bytes: 1 endian + 4-byte type tag
 * + optional uint32 count.
 *
 * family_tags: -1 unsupported, 0 point, 1 linestring, 2 polygon,
 *              3 multipoint, 4 multilinestring, 5 multipolygon
 * is_native: 1 if little-endian, 0 otherwise
 * primary_counts: first structural count (point_count for LS, ring_count
 *                 for Polygon, part_count for Multi*, 1 for Point)
 */
__global__ void wkb_header_scan(
    const unsigned char* payload,
    const int* record_offsets,
    signed char* family_tags,
    unsigned char* is_native,
    int* primary_counts,
    int count
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= count) return;

    int start = record_offsets[tid];
    int end   = record_offsets[tid + 1];
    int len   = end - start;

    /* Default: unsupported */
    family_tags[tid] = -1;
    is_native[tid]   = 0;
    primary_counts[tid] = 0;

    if (len < 5) return;

    const unsigned char* rec = payload + start;
    unsigned char byteorder = rec[0];
    if (byteorder != 1) return;  /* big-endian -> fallback */

    is_native[tid] = 1;
    unsigned int type_id = read_u32_le(rec + 1);

    /* Map WKB type id to family tag.  Only accept canonical 2D types. */
    signed char tag = -1;
    int pc = 0;
    switch (type_id) {
        case 1: /* Point */
            tag = 0;
            pc = 1;
            break;
        case 2: /* LineString */
            if (len >= 9) {
                tag = 1;
                pc = (int)read_u32_le(rec + 5);
            }
            break;
        case 3: /* Polygon */
            if (len >= 9) {
                tag = 2;
                pc = (int)read_u32_le(rec + 5);
            }
            break;
        case 4: /* MultiPoint */
            if (len >= 9) {
                tag = 3;
                pc = (int)read_u32_le(rec + 5);
            }
            break;
        case 5: /* MultiLineString */
            if (len >= 9) {
                tag = 4;
                pc = (int)read_u32_le(rec + 5);
            }
            break;
        case 6: /* MultiPolygon */
            if (len >= 9) {
                tag = 5;
                pc = (int)read_u32_le(rec + 5);
            }
            break;
        default:
            /* Z/M/ZM or unknown type -> remains -1 */
            break;
    }

    family_tags[tid] = tag;
    primary_counts[tid] = pc;
}


/* ---------- Stage 3a: Polygon sizing kernel ----------
 * 1 thread per polygon record.  Walks WKB bytes to count total
 * rings and total coordinates per record.
 */
__global__ void wkb_polygon_size_scan(
    const unsigned char* payload,
    const int* record_offsets,
    const int* row_indexes,
    int* total_rings_out,
    int* total_coords_out,
    int count
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= count) return;

    int row = row_indexes[tid];
    int start = record_offsets[row];
    const unsigned char* rec = payload + start;
    int ring_count = (int)read_u32_le(rec + 5);
    total_rings_out[tid] = ring_count;

    int cursor = 9;
    int total_pts = 0;
    for (int r = 0; r < ring_count; ++r) {
        int npts = (int)read_u32_le(rec + cursor);
        cursor += 4 + npts * 16;
        total_pts += npts;
    }
    total_coords_out[tid] = total_pts;
}


/* ---------- Stage 3b: MultiPoint sizing kernel ---------- */
__global__ void wkb_multipoint_size_scan(
    const unsigned char* payload,
    const int* record_offsets,
    const int* row_indexes,
    int* total_coords_out,
    int count
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= count) return;

    int row = row_indexes[tid];
    int start = record_offsets[row];
    const unsigned char* rec = payload + start;
    int part_count = (int)read_u32_le(rec + 5);
    /* Each embedded point is 21 bytes (1 endian + 4 type + 8 x + 8 y). */
    total_coords_out[tid] = part_count;
}


/* ---------- Stage 3c: MultiLineString sizing kernel ---------- */
__global__ void wkb_multilinestring_size_scan(
    const unsigned char* payload,
    const int* record_offsets,
    const int* row_indexes,
    int* total_parts_out,
    int* total_coords_out,
    int count
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= count) return;

    int row = row_indexes[tid];
    int start = record_offsets[row];
    const unsigned char* rec = payload + start;
    int part_count = (int)read_u32_le(rec + 5);
    total_parts_out[tid] = part_count;

    int cursor = 9;
    int total_pts = 0;
    for (int p = 0; p < part_count; ++p) {
        /* Each embedded linestring: 5 header + 4 count + coords. */
        int npts = (int)read_u32_le(rec + cursor + 5);
        cursor += 9 + npts * 16;
        total_pts += npts;
    }
    total_coords_out[tid] = total_pts;
}


/* ---------- Stage 3d: MultiPolygon sizing kernel ---------- */
__global__ void wkb_multipolygon_size_scan(
    const unsigned char* payload,
    const int* record_offsets,
    const int* row_indexes,
    int* total_parts_out,
    int* total_rings_out,
    int* total_coords_out,
    int count
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= count) return;

    int row = row_indexes[tid];
    int start = record_offsets[row];
    const unsigned char* rec = payload + start;
    int poly_count = (int)read_u32_le(rec + 5);
    total_parts_out[tid] = poly_count;

    int cursor = 9;
    int rings = 0;
    int pts = 0;
    for (int p = 0; p < poly_count; ++p) {
        /* Embedded polygon: 5 header + 4 ring_count + rings. */
        int ring_count = (int)read_u32_le(rec + cursor + 5);
        cursor += 9;
        rings += ring_count;
        for (int r = 0; r < ring_count; ++r) {
            int npts = (int)read_u32_le(rec + cursor);
            cursor += 4 + npts * 16;
            pts += npts;
        }
    }
    total_rings_out[tid] = rings;
    total_coords_out[tid] = pts;
}


/* ---------- Stage 4a: Point decode ----------
 * 1 thread per point record.
 */
__global__ void decode_point_wkb(
    const unsigned char* payload,
    const int* record_offsets,
    const int* row_indexes,
    double* x_out,
    double* y_out,
    unsigned char* empty_out,
    int count
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= count) return;

    int row = row_indexes[tid];
    int start = record_offsets[row];
    const unsigned char* rec = payload + start;
    int len = record_offsets[row + 1] - start;

    if (len < 21) {
        /* Malformed -> treat as empty NaN point */
        x_out[tid] = __longlong_as_double(0x7FF8000000000000ULL);
        y_out[tid] = __longlong_as_double(0x7FF8000000000000ULL);
        empty_out[tid] = 1;
        return;
    }

    double xv = read_f64_le(rec + 5);
    double yv = read_f64_le(rec + 13);
    x_out[tid] = xv;
    y_out[tid] = yv;

    /* Check NaN for empty point representation. */
    unsigned long long xbits = *reinterpret_cast<const unsigned long long*>(&xv);
    unsigned long long ybits = *reinterpret_cast<const unsigned long long*>(&yv);
    unsigned long long nan_mask = 0x7FF0000000000000ULL;
    int xnan = ((xbits & nan_mask) == nan_mask) && ((xbits & 0x000FFFFFFFFFFFFFULL) != 0);
    int ynan = ((ybits & nan_mask) == nan_mask) && ((ybits & 0x000FFFFFFFFFFFFFULL) != 0);
    empty_out[tid] = (unsigned char)(xnan | ynan);
}


/* ---------- Stage 4b: LineString decode ----------
 * 1 thread per linestring record.  Reads point count and writes
 * coordinates into pre-allocated output at the correct offset.
 */
__global__ void decode_linestring_wkb(
    const unsigned char* payload,
    const int* record_offsets,
    const int* row_indexes,
    const int* coord_offsets,
    double* x_out,
    double* y_out,
    int count
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= count) return;

    int row = row_indexes[tid];
    int start = record_offsets[row];
    const unsigned char* rec = payload + start;
    int npts = (int)read_u32_le(rec + 5);
    int out_offset = coord_offsets[tid];

    for (int i = 0; i < npts; ++i) {
        int byte_off = 9 + i * 16;
        x_out[out_offset + i] = read_f64_le(rec + byte_off);
        y_out[out_offset + i] = read_f64_le(rec + byte_off + 8);
    }
}


/* ---------- Stage 4c: Polygon decode ----------
 * 1 thread per polygon record.  Walks rings and writes coordinates
 * into flat output arrays at the correct offsets.
 */
__global__ void decode_polygon_wkb(
    const unsigned char* payload,
    const int* record_offsets,
    const int* row_indexes,
    const int* ring_count_offsets,
    const int* coord_offsets,
    int* ring_offsets_out,
    double* x_out,
    double* y_out,
    int count
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= count) return;

    int row = row_indexes[tid];
    int start = record_offsets[row];
    const unsigned char* rec = payload + start;
    int ring_count = (int)read_u32_le(rec + 5);
    int ring_base = ring_count_offsets[tid];
    int coord_base = coord_offsets[tid];

    int cursor = 9;
    int coord_pos = coord_base;
    for (int r = 0; r < ring_count; ++r) {
        int npts = (int)read_u32_le(rec + cursor);
        cursor += 4;
        ring_offsets_out[ring_base + r] = coord_pos;
        for (int i = 0; i < npts; ++i) {
            x_out[coord_pos + i] = read_f64_le(rec + cursor + i * 16);
            y_out[coord_pos + i] = read_f64_le(rec + cursor + i * 16 + 8);
        }
        cursor += npts * 16;
        coord_pos += npts;
    }
}


/* ---------- Stage 4d: MultiPoint decode ----------
 * 1 thread per multipoint record.
 */
__global__ void decode_multipoint_wkb(
    const unsigned char* payload,
    const int* record_offsets,
    const int* row_indexes,
    const int* coord_offsets,
    double* x_out,
    double* y_out,
    int count
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= count) return;

    int row = row_indexes[tid];
    int start = record_offsets[row];
    const unsigned char* rec = payload + start;
    int part_count = (int)read_u32_le(rec + 5);
    int out_offset = coord_offsets[tid];

    int cursor = 9;
    for (int i = 0; i < part_count; ++i) {
        /* Embedded point: 1 endian + 4 type + 8 x + 8 y = 21 bytes */
        x_out[out_offset + i] = read_f64_le(rec + cursor + 5);
        y_out[out_offset + i] = read_f64_le(rec + cursor + 13);
        cursor += 21;
    }
}


/* ---------- Stage 4e: MultiLineString decode ----------
 * 1 thread per multilinestring record.
 */
__global__ void decode_multilinestring_wkb(
    const unsigned char* payload,
    const int* record_offsets,
    const int* row_indexes,
    const int* part_count_offsets,
    const int* coord_offsets,
    int* part_offsets_out,
    double* x_out,
    double* y_out,
    int count
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= count) return;

    int row = row_indexes[tid];
    int start = record_offsets[row];
    const unsigned char* rec = payload + start;
    int part_count = (int)read_u32_le(rec + 5);
    int part_base = part_count_offsets[tid];
    int coord_base = coord_offsets[tid];

    int cursor = 9;
    int coord_pos = coord_base;
    for (int p = 0; p < part_count; ++p) {
        /* Embedded linestring header: 5 bytes (endian+type) + 4 count. */
        int npts = (int)read_u32_le(rec + cursor + 5);
        part_offsets_out[part_base + p] = coord_pos;
        cursor += 9;
        for (int i = 0; i < npts; ++i) {
            x_out[coord_pos + i] = read_f64_le(rec + cursor + i * 16);
            y_out[coord_pos + i] = read_f64_le(rec + cursor + i * 16 + 8);
        }
        cursor += npts * 16;
        coord_pos += npts;
    }
}


/* ---------- Stage 4f: MultiPolygon decode ----------
 * 1 thread per multipolygon record.
 */
__global__ void decode_multipolygon_wkb(
    const unsigned char* payload,
    const int* record_offsets,
    const int* row_indexes,
    const int* poly_count_offsets,
    const int* ring_count_offsets,
    const int* coord_offsets,
    int* part_offsets_out,
    int* ring_offsets_out,
    double* x_out,
    double* y_out,
    int count
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= count) return;

    int row = row_indexes[tid];
    int start = record_offsets[row];
    const unsigned char* rec = payload + start;
    int poly_count = (int)read_u32_le(rec + 5);
    int poly_base = poly_count_offsets[tid];
    int ring_base = ring_count_offsets[tid];
    int coord_base = coord_offsets[tid];

    int cursor = 9;
    int ring_pos = ring_base;
    int coord_pos = coord_base;
    for (int p = 0; p < poly_count; ++p) {
        /* Embedded polygon header: 5 bytes (endian+type) + 4 ring_count. */
        int ring_count = (int)read_u32_le(rec + cursor + 5);
        part_offsets_out[poly_base + p] = ring_pos;
        cursor += 9;
        for (int r = 0; r < ring_count; ++r) {
            int npts = (int)read_u32_le(rec + cursor);
            ring_offsets_out[ring_pos] = coord_pos;
            cursor += 4;
            for (int i = 0; i < npts; ++i) {
                x_out[coord_pos + i] = read_f64_le(rec + cursor + i * 16);
                y_out[coord_pos + i] = read_f64_le(rec + cursor + i * 16 + 8);
            }
            cursor += npts * 16;
            coord_pos += npts;
            ring_pos++;
        }
    }
}

}  /* extern "C" */
"""

_WKB_DECODE_KERNEL_NAMES = (
    "wkb_header_scan",
    "wkb_polygon_size_scan",
    "wkb_multipoint_size_scan",
    "wkb_multilinestring_size_scan",
    "wkb_multipolygon_size_scan",
    "decode_point_wkb",
    "decode_linestring_wkb",
    "decode_polygon_wkb",
    "decode_multipoint_wkb",
    "decode_multilinestring_wkb",
    "decode_multipolygon_wkb",
)


# ---------------------------------------------------------------------------
# Kernel compilation (matches io_wkb._wkb_encode_kernels pattern)
# ---------------------------------------------------------------------------

from vibespatial.nvrtc_precompile import request_nvrtc_warmup as _request_nvrtc_warmup  # noqa: E402

_request_nvrtc_warmup([
    ("wkb-decode", _WKB_DECODE_KERNEL_SOURCE, _WKB_DECODE_KERNEL_NAMES),
])


def _wkb_decode_kernels() -> dict[str, Any]:
    runtime = get_cuda_runtime()
    return runtime.compile_kernels(
        cache_key=make_kernel_cache_key("wkb-decode", _WKB_DECODE_KERNEL_SOURCE),
        source=_WKB_DECODE_KERNEL_SOURCE,
        kernel_names=_WKB_DECODE_KERNEL_NAMES,
    )


# ---------------------------------------------------------------------------
# Family tag -> GeometryFamily mapping (matches FAMILY_TAGS ordering)
# ---------------------------------------------------------------------------

_TAG_TO_FAMILY = {
    0: GeometryFamily.POINT,
    1: GeometryFamily.LINESTRING,
    2: GeometryFamily.POLYGON,
    3: GeometryFamily.MULTIPOINT,
    4: GeometryFamily.MULTILINESTRING,
    5: GeometryFamily.MULTIPOLYGON,
}

# Reverse: GeometryFamily -> kernel family tag (int8)
_FAMILY_TO_TAG = {v: k for k, v in _TAG_TO_FAMILY.items()}


# ---------------------------------------------------------------------------
# Stage 1: Header scan
# ---------------------------------------------------------------------------

def _stage1_header_scan(
    payload_device,
    record_offsets_device,
    record_count: int,
) -> tuple[Any, Any, Any]:
    """Run the header scan kernel. Returns (family_tags, is_native, primary_counts)."""
    import cupy as cp

    runtime = get_cuda_runtime()
    kernels = _wkb_decode_kernels()
    kernel = kernels["wkb_header_scan"]

    family_tags = cp.full(record_count, -1, dtype=cp.int8)
    is_native = cp.zeros(record_count, dtype=cp.uint8)
    primary_counts = cp.zeros(record_count, dtype=cp.int32)

    if record_count == 0:
        return family_tags, is_native, primary_counts

    ptr = runtime.pointer
    grid, block = runtime.launch_config(kernel, record_count)
    runtime.launch(
        kernel,
        grid=grid,
        block=block,
        params=(
            (ptr(payload_device), ptr(record_offsets_device),
             ptr(family_tags), ptr(is_native), ptr(primary_counts),
             record_count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_I32),
        ),
    )

    return family_tags, is_native, primary_counts


# ---------------------------------------------------------------------------
# Stage 2: Family partition
# ---------------------------------------------------------------------------

def _stage2_partition(
    family_tags,
    is_native,
    record_count: int,
) -> dict[GeometryFamily, Any]:
    """Partition row indices by geometry family using CuPy fancy indexing."""
    import cupy as cp

    partitions: dict[GeometryFamily, Any] = {}
    for tag, family in _TAG_TO_FAMILY.items():
        mask = (family_tags == np.int8(tag)) & (is_native == np.uint8(1))
        row_indexes = cp.flatnonzero(mask).astype(cp.int32, copy=False)
        if int(row_indexes.size) > 0:
            partitions[family] = row_indexes

    return partitions


# ---------------------------------------------------------------------------
# Stage 3 + 4: Per-family decode
# ---------------------------------------------------------------------------

def _decode_point_family(
    payload_device,
    record_offsets_device,
    row_indexes,
) -> DeviceFamilyGeometryBuffer:
    """Decode point records on GPU."""
    import cupy as cp

    runtime = get_cuda_runtime()
    kernels = _wkb_decode_kernels()
    kernel = kernels["decode_point_wkb"]

    n = int(row_indexes.size)
    x_out = cp.empty(n, dtype=cp.float64)
    y_out = cp.empty(n, dtype=cp.float64)
    empty_out = cp.zeros(n, dtype=cp.uint8)

    if n > 0:
        ptr = runtime.pointer
        grid, block = runtime.launch_config(kernel, n)
        runtime.launch(
            kernel,
            grid=grid,
            block=block,
            params=(
                (ptr(payload_device), ptr(record_offsets_device),
                 ptr(row_indexes), ptr(x_out), ptr(y_out), ptr(empty_out), n),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_I32),
            ),
        )

    empty_mask = empty_out.astype(cp.bool_, copy=False)
    nonempty = ~empty_mask
    nonempty_counts = nonempty.astype(cp.int32, copy=False)
    geometry_offsets = cp.empty(n + 1, dtype=cp.int32)
    geometry_offsets[0] = 0
    if n > 0:
        offsets_excl = exclusive_sum(nonempty_counts, synchronize=False)
        geometry_offsets[1:] = offsets_excl + nonempty_counts

    x_valid = x_out[nonempty]
    y_valid = y_out[nonempty]

    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.POINT,
        x=x_valid,
        y=y_valid,
        geometry_offsets=geometry_offsets,
        empty_mask=empty_mask,
        bounds=None,
    )


def _decode_linestring_family(
    payload_device,
    record_offsets_device,
    row_indexes,
    primary_counts,
) -> DeviceFamilyGeometryBuffer:
    """Decode linestring records on GPU."""
    import cupy as cp

    runtime = get_cuda_runtime()
    kernels = _wkb_decode_kernels()

    n = int(row_indexes.size)
    point_counts = primary_counts[row_indexes].astype(cp.int32, copy=False)

    # Build geometry offsets via CCCL exclusive_sum (ADR-0033)
    coord_offsets = exclusive_sum(point_counts, synchronize=False) if n > 0 else cp.zeros(0, dtype=cp.int32)
    geometry_offsets = cp.empty(n + 1, dtype=cp.int32)
    geometry_offsets[0] = 0
    if n > 0:
        geometry_offsets[1:] = coord_offsets + point_counts

    total_coords = int(geometry_offsets[-1]) if n > 0 else 0

    x_out = cp.empty(total_coords, dtype=cp.float64)
    y_out = cp.empty(total_coords, dtype=cp.float64)

    if n > 0 and total_coords > 0:
        kernel = kernels["decode_linestring_wkb"]
        ptr = runtime.pointer
        grid, block = runtime.launch_config(kernel, n)
        runtime.launch(
            kernel,
            grid=grid,
            block=block,
            params=(
                (ptr(payload_device), ptr(record_offsets_device),
                 ptr(row_indexes), ptr(coord_offsets),
                 ptr(x_out), ptr(y_out), n),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_I32),
            ),
        )

    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.LINESTRING,
        x=x_out,
        y=y_out,
        geometry_offsets=geometry_offsets,
        empty_mask=point_counts == 0,
        bounds=None,
    )


def _decode_polygon_family(
    payload_device,
    record_offsets_device,
    row_indexes,
) -> DeviceFamilyGeometryBuffer:
    """Decode polygon records on GPU using sizing kernel + decode kernel."""
    import cupy as cp

    runtime = get_cuda_runtime()
    kernels = _wkb_decode_kernels()

    n = int(row_indexes.size)

    # Stage 3: size scan
    total_rings_per = cp.zeros(n, dtype=cp.int32)
    total_coords_per = cp.zeros(n, dtype=cp.int32)

    if n > 0:
        size_kernel = kernels["wkb_polygon_size_scan"]
        ptr = runtime.pointer
        grid, block = runtime.launch_config(size_kernel, n)
        runtime.launch(
            size_kernel,
            grid=grid,
            block=block,
            params=(
                (ptr(payload_device), ptr(record_offsets_device),
                 ptr(row_indexes), ptr(total_rings_per), ptr(total_coords_per), n),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
            ),
        )

    # Geometry offsets = prefix sum of ring counts (rings per polygon, ADR-0033)
    ring_count_offsets = exclusive_sum(total_rings_per, synchronize=False) if n > 0 else cp.zeros(0, dtype=cp.int32)
    geometry_offsets = cp.empty(n + 1, dtype=cp.int32)
    geometry_offsets[0] = 0
    if n > 0:
        geometry_offsets[1:] = ring_count_offsets + total_rings_per
    total_rings = int(geometry_offsets[-1]) if n > 0 else 0
    coord_offsets = exclusive_sum(total_coords_per, synchronize=False) if n > 0 else cp.zeros(0, dtype=cp.int32)
    total_coords = int(coord_offsets[-1] + total_coords_per[-1]) if n > 0 else 0

    # Allocate output
    ring_offsets_out = cp.empty(total_rings + 1, dtype=cp.int32)
    x_out = cp.empty(total_coords, dtype=cp.float64)
    y_out = cp.empty(total_coords, dtype=cp.float64)

    if n > 0 and total_coords > 0:
        decode_kernel = kernels["decode_polygon_wkb"]
        ptr = runtime.pointer
        grid, block = runtime.launch_config(decode_kernel, n)
        runtime.launch(
            decode_kernel,
            grid=grid,
            block=block,
            params=(
                (ptr(payload_device), ptr(record_offsets_device),
                 ptr(row_indexes), ptr(ring_count_offsets), ptr(coord_offsets),
                 ptr(ring_offsets_out), ptr(x_out), ptr(y_out), n),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_I32),
            ),
        )

    # Set the sentinel ring offset
    if total_rings > 0:
        ring_offsets_out[total_rings] = total_coords
    elif total_rings == 0 and n > 0:
        ring_offsets_out[0] = 0

    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.POLYGON,
        x=x_out,
        y=y_out,
        geometry_offsets=geometry_offsets,
        empty_mask=total_rings_per == 0,
        ring_offsets=ring_offsets_out,
        bounds=None,
    )


def _decode_multipoint_family(
    payload_device,
    record_offsets_device,
    row_indexes,
    primary_counts,
) -> DeviceFamilyGeometryBuffer:
    """Decode multipoint records on GPU."""
    import cupy as cp

    runtime = get_cuda_runtime()
    kernels = _wkb_decode_kernels()

    n = int(row_indexes.size)
    part_counts = primary_counts[row_indexes].astype(cp.int32, copy=False)

    coord_offsets = exclusive_sum(part_counts, synchronize=False) if n > 0 else cp.zeros(0, dtype=cp.int32)
    geometry_offsets = cp.empty(n + 1, dtype=cp.int32)
    geometry_offsets[0] = 0
    if n > 0:
        geometry_offsets[1:] = coord_offsets + part_counts
    total_coords = int(geometry_offsets[-1]) if n > 0 else 0

    x_out = cp.empty(total_coords, dtype=cp.float64)
    y_out = cp.empty(total_coords, dtype=cp.float64)

    if n > 0 and total_coords > 0:
        kernel = kernels["decode_multipoint_wkb"]
        ptr = runtime.pointer
        grid, block = runtime.launch_config(kernel, n)
        runtime.launch(
            kernel,
            grid=grid,
            block=block,
            params=(
                (ptr(payload_device), ptr(record_offsets_device),
                 ptr(row_indexes), ptr(coord_offsets),
                 ptr(x_out), ptr(y_out), n),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_I32),
            ),
        )

    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.MULTIPOINT,
        x=x_out,
        y=y_out,
        geometry_offsets=geometry_offsets,
        empty_mask=part_counts == 0,
        bounds=None,
    )


def _decode_multilinestring_family(
    payload_device,
    record_offsets_device,
    row_indexes,
) -> DeviceFamilyGeometryBuffer:
    """Decode multilinestring records on GPU using sizing kernel + decode kernel."""
    import cupy as cp

    runtime = get_cuda_runtime()
    kernels = _wkb_decode_kernels()

    n = int(row_indexes.size)

    # Stage 3: size scan
    total_parts_per = cp.zeros(n, dtype=cp.int32)
    total_coords_per = cp.zeros(n, dtype=cp.int32)

    if n > 0:
        size_kernel = kernels["wkb_multilinestring_size_scan"]
        ptr = runtime.pointer
        grid, block = runtime.launch_config(size_kernel, n)
        runtime.launch(
            size_kernel,
            grid=grid,
            block=block,
            params=(
                (ptr(payload_device), ptr(record_offsets_device),
                 ptr(row_indexes), ptr(total_parts_per), ptr(total_coords_per), n),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
            ),
        )

    # Geometry offsets = prefix sum of part counts (ADR-0033)
    part_count_offsets = exclusive_sum(total_parts_per, synchronize=False) if n > 0 else cp.zeros(0, dtype=cp.int32)
    geometry_offsets = cp.empty(n + 1, dtype=cp.int32)
    geometry_offsets[0] = 0
    if n > 0:
        geometry_offsets[1:] = part_count_offsets + total_parts_per
    total_parts = int(geometry_offsets[-1]) if n > 0 else 0
    coord_offsets = exclusive_sum(total_coords_per, synchronize=False) if n > 0 else cp.zeros(0, dtype=cp.int32)
    total_coords = int(coord_offsets[-1] + total_coords_per[-1]) if n > 0 else 0

    part_offsets_out = cp.empty(total_parts + 1, dtype=cp.int32)
    x_out = cp.empty(total_coords, dtype=cp.float64)
    y_out = cp.empty(total_coords, dtype=cp.float64)

    if n > 0 and total_coords > 0:
        decode_kernel = kernels["decode_multilinestring_wkb"]
        ptr = runtime.pointer
        grid, block = runtime.launch_config(decode_kernel, n)
        runtime.launch(
            decode_kernel,
            grid=grid,
            block=block,
            params=(
                (ptr(payload_device), ptr(record_offsets_device),
                 ptr(row_indexes), ptr(part_count_offsets), ptr(coord_offsets),
                 ptr(part_offsets_out), ptr(x_out), ptr(y_out), n),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_I32),
            ),
        )

    # Set sentinel
    if total_parts > 0:
        part_offsets_out[total_parts] = total_coords
    elif total_parts == 0 and n > 0:
        part_offsets_out[0] = 0

    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.MULTILINESTRING,
        x=x_out,
        y=y_out,
        geometry_offsets=geometry_offsets,
        empty_mask=total_parts_per == 0,
        part_offsets=part_offsets_out,
        bounds=None,
    )


def _decode_multipolygon_family(
    payload_device,
    record_offsets_device,
    row_indexes,
) -> DeviceFamilyGeometryBuffer:
    """Decode multipolygon records on GPU using sizing kernel + decode kernel."""
    import cupy as cp

    runtime = get_cuda_runtime()
    kernels = _wkb_decode_kernels()

    n = int(row_indexes.size)

    # Stage 3: size scan
    total_parts_per = cp.zeros(n, dtype=cp.int32)
    total_rings_per = cp.zeros(n, dtype=cp.int32)
    total_coords_per = cp.zeros(n, dtype=cp.int32)

    if n > 0:
        size_kernel = kernels["wkb_multipolygon_size_scan"]
        ptr = runtime.pointer
        grid, block = runtime.launch_config(size_kernel, n)
        runtime.launch(
            size_kernel,
            grid=grid,
            block=block,
            params=(
                (ptr(payload_device), ptr(record_offsets_device),
                 ptr(row_indexes), ptr(total_parts_per),
                 ptr(total_rings_per), ptr(total_coords_per), n),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_I32),
            ),
        )

    # Geometry offsets = prefix sum of part counts (polygon parts per multipolygon, ADR-0033)
    poly_count_offsets = exclusive_sum(total_parts_per, synchronize=False) if n > 0 else cp.zeros(0, dtype=cp.int32)
    geometry_offsets = cp.empty(n + 1, dtype=cp.int32)
    geometry_offsets[0] = 0
    if n > 0:
        geometry_offsets[1:] = poly_count_offsets + total_parts_per
    total_parts = int(geometry_offsets[-1]) if n > 0 else 0
    ring_count_offsets = exclusive_sum(total_rings_per, synchronize=False) if n > 0 else cp.zeros(0, dtype=cp.int32)
    coord_offsets = exclusive_sum(total_coords_per, synchronize=False) if n > 0 else cp.zeros(0, dtype=cp.int32)

    total_rings = int(ring_count_offsets[-1] + total_rings_per[-1]) if n > 0 else 0
    total_coords = int(coord_offsets[-1] + total_coords_per[-1]) if n > 0 else 0

    part_offsets_out = cp.empty(total_parts + 1, dtype=cp.int32)
    ring_offsets_out = cp.empty(total_rings + 1, dtype=cp.int32)
    x_out = cp.empty(total_coords, dtype=cp.float64)
    y_out = cp.empty(total_coords, dtype=cp.float64)

    if n > 0 and total_coords > 0:
        decode_kernel = kernels["decode_multipolygon_wkb"]
        ptr = runtime.pointer
        grid, block = runtime.launch_config(decode_kernel, n)
        runtime.launch(
            decode_kernel,
            grid=grid,
            block=block,
            params=(
                (ptr(payload_device), ptr(record_offsets_device),
                 ptr(row_indexes), ptr(poly_count_offsets),
                 ptr(ring_count_offsets), ptr(coord_offsets),
                 ptr(part_offsets_out), ptr(ring_offsets_out),
                 ptr(x_out), ptr(y_out), n),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_I32),
            ),
        )

    # Set sentinels
    if total_parts > 0:
        part_offsets_out[total_parts] = total_rings
    elif total_parts == 0 and n > 0:
        part_offsets_out[0] = 0

    if total_rings > 0:
        ring_offsets_out[total_rings] = total_coords
    elif total_rings == 0 and n > 0:
        ring_offsets_out[0] = 0

    return DeviceFamilyGeometryBuffer(
        family=GeometryFamily.MULTIPOLYGON,
        x=x_out,
        y=y_out,
        geometry_offsets=geometry_offsets,
        empty_mask=total_parts_per == 0,
        part_offsets=part_offsets_out,
        ring_offsets=ring_offsets_out,
        bounds=None,
    )


# ---------------------------------------------------------------------------
# Stage 5: Assembly
# ---------------------------------------------------------------------------

def _assemble_single_family(
    family: GeometryFamily,
    family_buffer: DeviceFamilyGeometryBuffer,
    validity_device,
    record_count: int,
) -> OwnedGeometryArray:
    """Assemble single-family result using io_pylibcudf helper."""
    return _build_device_single_family_owned(
        family=family,
        validity_device=validity_device,
        x_device=family_buffer.x,
        y_device=family_buffer.y,
        geometry_offsets_device=family_buffer.geometry_offsets,
        empty_mask_device=family_buffer.empty_mask,
        part_offsets_device=family_buffer.part_offsets,
        ring_offsets_device=family_buffer.ring_offsets,
        detail="created device-resident owned geometry array from GPU WKB decode kernel pipeline",
    )


def _assemble_mixed(
    partitions: dict[GeometryFamily, Any],
    family_buffers: dict[GeometryFamily, DeviceFamilyGeometryBuffer],
    family_tags,
    is_native,
    record_count: int,
) -> OwnedGeometryArray:
    """Assemble mixed-family result using io_pylibcudf helper."""
    import cupy as cp

    valid_mask = is_native.astype(cp.bool_, copy=False)
    tags_device = cp.where(
        valid_mask,
        family_tags.astype(cp.int8, copy=False),
        np.int8(-1),
    ).astype(cp.int8, copy=False)

    family_row_offsets_device = cp.full(record_count, -1, dtype=cp.int32)
    for family, row_indexes in partitions.items():
        n_rows = int(row_indexes.size)
        if n_rows > 0:
            family_row_offsets_device[row_indexes] = cp.arange(n_rows, dtype=cp.int32)

    family_devices: dict[GeometryFamily, DeviceFamilyGeometryBuffer] = {}
    for family in family_buffers:
        family_devices[family] = family_buffers[family]

    return _build_device_mixed_owned(
        validity_device=valid_mask,
        tags_device=tags_device,
        family_row_offsets_device=family_row_offsets_device,
        family_devices=family_devices,
        detail="created device-resident owned geometry array from GPU WKB decode kernel pipeline (mixed)",
    )


# ---------------------------------------------------------------------------
# Public API: decode_wkb_device_pipeline
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "decode_wkb",
    "gpu-cuda-python",
    kernel_class=KernelClass.COARSE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=tuple(family.value for family in GeometryFamily),
    tags=("wkb", "decode", "nvrtc", "cccl"),
)
def decode_wkb_device_pipeline(
    payload_device,
    record_offsets_device,
    record_count: int,
) -> OwnedGeometryArray:
    """GPU WKB decode pipeline.

    Reads WKB byte streams directly on GPU using custom CUDA kernels.
    Returns an OwnedGeometryArray with device-resident geometry buffers.

    Parameters
    ----------
    payload_device : device array (uint8)
        Contiguous WKB byte payload on device.
    record_offsets_device : device array (int32)
        Byte offsets for each record (length = record_count + 1).
    record_count : int
        Number of WKB records.

    Returns
    -------
    OwnedGeometryArray
        Device-resident geometry array.
    """
    import cupy as cp

    runtime = get_cuda_runtime()

    if record_count == 0:
        validity = cp.zeros(0, dtype=cp.bool_)
        return _build_device_single_family_owned(
            family=GeometryFamily.POINT,
            validity_device=validity,
            x_device=cp.empty(0, dtype=cp.float64),
            y_device=cp.empty(0, dtype=cp.float64),
            geometry_offsets_device=cp.zeros(1, dtype=cp.int32),
            empty_mask_device=cp.zeros(0, dtype=cp.bool_),
            detail="created empty device-resident owned geometry array from GPU WKB decode pipeline",
        )

    # Stage 1: Header scan
    family_tags, is_native, primary_counts = _stage1_header_scan(
        payload_device, record_offsets_device, record_count,
    )

    # Stage 2: Family partition
    partitions = _stage2_partition(family_tags, is_native, record_count)

    if not partitions:
        # No supported records found -- return all-null result
        validity = cp.zeros(record_count, dtype=cp.bool_)
        return _build_device_single_family_owned(
            family=GeometryFamily.POINT,
            validity_device=validity,
            x_device=cp.empty(0, dtype=cp.float64),
            y_device=cp.empty(0, dtype=cp.float64),
            geometry_offsets_device=cp.zeros(1, dtype=cp.int32),
            empty_mask_device=cp.zeros(0, dtype=cp.bool_),
            detail="created all-null device-resident owned geometry array from GPU WKB decode pipeline",
        )

    # Stages 3+4: Per-family decode
    family_buffers: dict[GeometryFamily, DeviceFamilyGeometryBuffer] = {}

    for family, row_indexes in partitions.items():
        if family is GeometryFamily.POINT:
            family_buffers[family] = _decode_point_family(
                payload_device, record_offsets_device, row_indexes,
            )
        elif family is GeometryFamily.LINESTRING:
            family_buffers[family] = _decode_linestring_family(
                payload_device, record_offsets_device, row_indexes, primary_counts,
            )
        elif family is GeometryFamily.POLYGON:
            family_buffers[family] = _decode_polygon_family(
                payload_device, record_offsets_device, row_indexes,
            )
        elif family is GeometryFamily.MULTIPOINT:
            family_buffers[family] = _decode_multipoint_family(
                payload_device, record_offsets_device, row_indexes, primary_counts,
            )
        elif family is GeometryFamily.MULTILINESTRING:
            family_buffers[family] = _decode_multilinestring_family(
                payload_device, record_offsets_device, row_indexes,
            )
        elif family is GeometryFamily.MULTIPOLYGON:
            family_buffers[family] = _decode_multipolygon_family(
                payload_device, record_offsets_device, row_indexes,
            )

    # Sync once before reading results to host (Stage 5 assembly reads host arrays)
    runtime.synchronize()

    # Stage 5: Assembly
    families_present = list(family_buffers.keys())
    if len(families_present) == 1:
        single_family = families_present[0]
        single_row_indexes = partitions[single_family]
        n_supported = int(single_row_indexes.size)
        # Check if ALL records are this family (no unsupported/fallback)
        if n_supported == record_count:
            return _assemble_single_family(
                single_family,
                family_buffers[single_family],
                is_native.astype(cp.bool_, copy=False),
                record_count,
            )

    return _assemble_mixed(
        partitions, family_buffers, family_tags, is_native, record_count,
    )


# ---------------------------------------------------------------------------
# Input preparation helper
# ---------------------------------------------------------------------------

def _prepare_device_wkb_input(
    values: list[bytes | None],
) -> tuple[Any, Any, np.ndarray]:
    """Concatenate WKB byte arrays and transfer to device.

    Parameters
    ----------
    values : list of bytes or None
        WKB byte arrays.  None entries produce null rows.

    Returns
    -------
    (payload_device, offsets_device, validity_host)
        payload_device : CuPy uint8 array on device
        offsets_device : CuPy int32 array on device (length = len(values) + 1)
        validity_host : numpy bool array (True = non-null)
    """
    import cupy as cp

    record_count = len(values)
    validity_host = np.array([v is not None for v in values], dtype=np.bool_)

    # Build byte offsets and concatenated payload
    offsets_host = np.empty(record_count + 1, dtype=np.int32)
    offsets_host[0] = 0
    total_bytes = 0
    for i, v in enumerate(values):
        size = len(v) if v is not None else 0
        total_bytes += size
        offsets_host[i + 1] = total_bytes

    if total_bytes > 0:
        payload_host = np.empty(total_bytes, dtype=np.uint8)
        for i, v in enumerate(values):
            if v is not None and len(v) > 0:
                start = offsets_host[i]
                payload_host[start:start + len(v)] = np.frombuffer(v, dtype=np.uint8)
    else:
        payload_host = np.empty(0, dtype=np.uint8)

    payload_device = cp.asarray(payload_host)
    offsets_device = cp.asarray(offsets_host)

    return payload_device, offsets_device, validity_host
