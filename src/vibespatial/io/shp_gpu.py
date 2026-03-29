"""GPU-native Shapefile (.shp) binary decoder.

Reads geometry directly from the SHP binary format, bypassing the
pyogrio -> WKB -> GPU WKB decode roundtrip.  The SHP binary stores
coordinates as little-endian float64, and CUDA is always little-endian,
so coordinate extraction is a pure gather operation with zero byte-swapping.

Architecture
------------
1. **SHX index parsing (CPU)** -- The .shx file is tiny (100-byte header
   + 8 bytes per record).  Parsing it on CPU gives exact byte offsets for
   every record in the .shp file.
2. **Bulk file transfer (GPU)** -- The entire .shp file is read to device
   memory via kvikio (parallel POSIX reads) or CuPy fallback.
3. **NVRTC kernel decode (GPU)** -- Per-record coordinate extraction runs
   as one NVRTC kernel launch per geometry type.  Variable-length types
   (PolyLine, Polygon, MultiPoint) use a two-pass count-scatter pattern:
   pass 1 counts parts/points per record, CCCL exclusive_sum builds offsets,
   pass 2 gathers coordinates to flat output arrays.
4. **Offset assembly (GPU)** -- geometry_offsets, part_offsets, ring_offsets
   built via CuPy cumsum / CCCL exclusive_sum.  Everything stays device-
   resident.
5. **OwnedGeometryArray construction** -- Uses _build_device_single_family_owned
   to produce a fully device-resident geometry array.

SHP binary format reference:
  - All coordinates are little-endian float64 (no byte-swap on CUDA)
  - Record headers are big-endian (parsed from SHX, not re-read on GPU)
  - Shape types: 0=Null, 1=Point, 3=PolyLine, 5=Polygon, 8=MultiPoint

Tier classification (ADR-0033):
    - SHX/SHP header parsing: CPU (small data, one-time)
    - Coordinate gather from SHP binary: Tier 1 (custom NVRTC -- binary offset gather)
    - Offset assembly: Tier 2 (CuPy cumsum) + CCCL exclusive_sum
    - No PrecisionPlan needed: this is I/O parsing producing fp64 storage.
      Same rationale as dbf_gpu.py and csv_gpu.py.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.cccl_primitives import exclusive_sum

if TYPE_CHECKING:
    import cupy as cp

    from vibespatial.geometry.owned import OwnedGeometryArray

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

# ---------------------------------------------------------------------------
# CCCL warmup (ADR-0034)
# ---------------------------------------------------------------------------
request_warmup(["exclusive_scan_i32"])

# ---------------------------------------------------------------------------
# SHP shape type constants
# ---------------------------------------------------------------------------
SHP_NULL = 0
SHP_POINT = 1
SHP_POLYLINE = 3
SHP_POLYGON = 5
SHP_MULTIPOINT = 8


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShpHeader:
    """Parsed SHP/SHX file header."""

    shape_type: int  # 1=Point, 3=PolyLine, 5=Polygon, 8=MultiPoint
    bbox: tuple  # (xmin, ymin, xmax, ymax)
    n_records: int


# ---------------------------------------------------------------------------
# NVRTC kernel sources (Tier 1)
# ---------------------------------------------------------------------------

# Point decode: each thread extracts one Point record's (x, y).
# SHP Point record layout:
#   record_header(8 bytes BE) + type(4 bytes LE i32) + x(8 bytes LE f64) + y(8 bytes LE f64)
# We skip to record_offset + 12 for x, record_offset + 20 for y.
_SHP_DECODE_POINTS_SOURCE = r"""
// Read a little-endian double from a potentially unaligned byte pointer.
// SHP record offsets are 2-byte aligned (16-bit word units from SHX),
// so double reads at offset+12 may not be 8-byte aligned.
__device__ __forceinline__ double read_double(const unsigned char* p) {
    double val;
    memcpy(&val, p, 8);
    return val;
}

// Read a little-endian int32 from a potentially unaligned byte pointer.
__device__ __forceinline__ int read_int32(const unsigned char* p) {
    int val;
    memcpy(&val, p, 4);
    return val;
}

extern "C" __global__ void __launch_bounds__(256, 4)
shp_decode_points(
    const unsigned char* __restrict__ shp_data,
    const long long*     __restrict__ record_offsets,
    double*              __restrict__ out_x,
    double*              __restrict__ out_y,
    const int n_records
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_records) return;

    long long offset = record_offsets[idx];
    // Skip 8-byte record header + 4-byte shape type = 12 bytes
    const unsigned char* p = shp_data + offset + 12;

    // CUDA is little-endian, SHP coordinates are little-endian.
    // Use memcpy to handle potentially unaligned reads safely.
    out_x[idx] = read_double(p);
    out_y[idx] = read_double(p + 8);
}
"""

# Count parts and points for PolyLine/Polygon/MultiPoint records.
# PolyLine/Polygon layout after record header:
#   type(4) + bbox(32) + num_parts(4) + num_points(4) + parts[num_parts] + points[num_points]
# MultiPoint layout after record header:
#   type(4) + bbox(32) + num_points(4) + points[num_points]
_SHP_COUNT_SOURCE = r"""
__device__ __forceinline__ double read_double(const unsigned char* p) {
    double val;
    memcpy(&val, p, 8);
    return val;
}

__device__ __forceinline__ int read_int32(const unsigned char* p) {
    int val;
    memcpy(&val, p, 4);
    return val;
}

extern "C" __global__ void __launch_bounds__(256, 4)
shp_count_parts_points(
    const unsigned char* __restrict__ shp_data,
    const long long*     __restrict__ record_offsets,
    const long long*     __restrict__ content_lengths,
    int*                 __restrict__ out_num_parts,
    int*                 __restrict__ out_num_points,
    int*                 __restrict__ out_is_null,
    const int n_records,
    const int shape_type
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_records) return;

    long long offset = record_offsets[idx];
    long long clen = content_lengths[idx];

    // Read the actual shape type from the record (may be 0 = Null)
    // Offset + 8 (record header) = start of shape content
    int rec_type = read_int32(shp_data + offset + 8);

    if (rec_type == 0 || clen <= 4) {
        // Null shape
        out_num_parts[idx] = 0;
        out_num_points[idx] = 0;
        out_is_null[idx] = 1;
        return;
    }

    out_is_null[idx] = 0;

    if (shape_type == 8) {
        // MultiPoint: offset+8(header) + 4(type) + 32(bbox) = offset + 44
        out_num_parts[idx] = 1;
        out_num_points[idx] = read_int32(shp_data + offset + 44);
    } else {
        // PolyLine (3) or Polygon (5)
        // offset + 8(header) + 4(type) + 32(bbox) = offset + 44
        const unsigned char* p = shp_data + offset + 44;
        out_num_parts[idx] = read_int32(p);
        out_num_points[idx] = read_int32(p + 4);
    }
}
"""

# Gather coordinates from PolyLine/Polygon records to flat output arrays.
# Each thread handles one record, copying all its points to the right
# output position determined by the prefix-summed coordinate offsets.
_SHP_GATHER_COORDS_SOURCE = r"""
__device__ __forceinline__ double read_double(const unsigned char* p) {
    double val;
    memcpy(&val, p, 8);
    return val;
}

__device__ __forceinline__ int read_int32(const unsigned char* p) {
    int val;
    memcpy(&val, p, 4);
    return val;
}

extern "C" __global__ void __launch_bounds__(256, 4)
shp_gather_coords(
    const unsigned char* __restrict__ shp_data,
    const long long*     __restrict__ record_offsets,
    const int*           __restrict__ coord_offsets,
    const int*           __restrict__ num_parts,
    const int*           __restrict__ num_points,
    const int*           __restrict__ is_null,
    double*              __restrict__ out_x,
    double*              __restrict__ out_y,
    const int n_records,
    const int shape_type
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_records) return;

    if (is_null[idx]) return;

    int np = num_points[idx];
    if (np == 0) return;

    long long offset = record_offsets[idx];
    int write_start = coord_offsets[idx];

    // Compute where coordinates begin in this record
    const unsigned char* coords;
    if (shape_type == 8) {
        // MultiPoint: offset + 8(header) + 4(type) + 32(bbox) + 4(num_points) = offset + 48
        coords = shp_data + offset + 48;
    } else {
        // PolyLine/Polygon: offset + 8(header) + 4(type) + 32(bbox) + 4(num_parts) + 4(num_points)
        //                    + 4*num_parts(parts array) = offset + 52 + 4*num_parts
        int nparts = num_parts[idx];
        coords = shp_data + offset + 52 + 4 * nparts;
    }

    // Copy coordinates: SHP stores as [x0, y0, x1, y1, ...] (interleaved)
    // Use memcpy for potentially unaligned reads.
    for (int i = 0; i < np; ++i) {
        out_x[write_start + i] = read_double(coords + i * 16);
        out_y[write_start + i] = read_double(coords + i * 16 + 8);
    }
}
"""

# Gather part indices (ring start positions for Polygon, part starts for PolyLine)
# from each record's parts[] array into a flat output array.
_SHP_GATHER_PARTS_SOURCE = r"""
__device__ __forceinline__ int read_int32(const unsigned char* p) {
    int val;
    memcpy(&val, p, 4);
    return val;
}

extern "C" __global__ void __launch_bounds__(256, 4)
shp_gather_parts(
    const unsigned char* __restrict__ shp_data,
    const long long*     __restrict__ record_offsets,
    const int*           __restrict__ part_offsets,
    const int*           __restrict__ coord_offsets,
    const int*           __restrict__ num_parts,
    const int*           __restrict__ num_points,
    const int*           __restrict__ is_null,
    int*                 __restrict__ out_ring_offsets,
    const int n_records
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_records) return;

    if (is_null[idx]) return;

    int nparts = num_parts[idx];
    if (nparts == 0) return;

    long long offset = record_offsets[idx];
    int write_start = part_offsets[idx];
    int coord_base = coord_offsets[idx];

    // parts[] array starts at: offset + 8(header) + 4(type) + 32(bbox)
    //                          + 4(num_parts) + 4(num_points) = offset + 52
    const unsigned char* parts_base = shp_data + offset + 52;

    // Write each part's starting coordinate index (global, not local)
    for (int p = 0; p < nparts; ++p) {
        out_ring_offsets[write_start + p] = coord_base + read_int32(parts_base + p * 4);
    }
}
"""

_SHP_POINT_NAMES = ("shp_decode_points",)
_SHP_COUNT_NAMES = ("shp_count_parts_points",)
_SHP_GATHER_COORDS_NAMES = ("shp_gather_coords",)
_SHP_GATHER_PARTS_NAMES = ("shp_gather_parts",)

# ---------------------------------------------------------------------------
# NVRTC warmup (ADR-0034 Level 2)
# ---------------------------------------------------------------------------
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup(
    [
        ("shp-decode-points", _SHP_DECODE_POINTS_SOURCE, _SHP_POINT_NAMES),
        ("shp-count-parts-points", _SHP_COUNT_SOURCE, _SHP_COUNT_NAMES),
        ("shp-gather-coords", _SHP_GATHER_COORDS_SOURCE, _SHP_GATHER_COORDS_NAMES),
        ("shp-gather-parts", _SHP_GATHER_PARTS_SOURCE, _SHP_GATHER_PARTS_NAMES),
    ]
)


# ---------------------------------------------------------------------------
# Kernel compilation helpers
# ---------------------------------------------------------------------------


def _point_kernels():
    return compile_kernel_group(
        "shp-decode-points",
        _SHP_DECODE_POINTS_SOURCE,
        _SHP_POINT_NAMES,
    )


def _count_kernels():
    return compile_kernel_group(
        "shp-count-parts-points",
        _SHP_COUNT_SOURCE,
        _SHP_COUNT_NAMES,
    )


def _gather_coords_kernels():
    return compile_kernel_group(
        "shp-gather-coords",
        _SHP_GATHER_COORDS_SOURCE,
        _SHP_GATHER_COORDS_NAMES,
    )


def _gather_parts_kernels():
    return compile_kernel_group(
        "shp-gather-parts",
        _SHP_GATHER_PARTS_SOURCE,
        _SHP_GATHER_PARTS_NAMES,
    )


# ---------------------------------------------------------------------------
# ZIP support: extract SHP/SHX from .shp.zip archives
# ---------------------------------------------------------------------------


def _extract_shp_from_zip(zip_path: Path) -> tuple[bytes, bytes]:
    """Extract .shp and .shx bytes from a zipped shapefile.

    Searches for files ending in .shp and .shx inside the archive.
    Raises FileNotFoundError if either is missing.
    """
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        shp_name = next((n for n in names if n.lower().endswith(".shp")), None)
        shx_name = next((n for n in names if n.lower().endswith(".shx")), None)

        if shp_name is None:
            raise FileNotFoundError(f"No .shp file found in {zip_path}")
        if shx_name is None:
            raise FileNotFoundError(f"No .shx file found in {zip_path}")

        return zf.read(shp_name), zf.read(shx_name)


def _read_shx_index_from_bytes(
    raw: bytes,
) -> tuple[ShpHeader, np.ndarray, np.ndarray]:
    """Parse SHX index from in-memory bytes (same logic as _read_shx_index)."""
    return _parse_shx_bytes(raw)


# ---------------------------------------------------------------------------
# Phase 1: CPU -- Read SHX index + SHP header
# ---------------------------------------------------------------------------


def _read_shx_index(shx_path: Path) -> tuple[ShpHeader, np.ndarray, np.ndarray]:
    """Read SHX file to get record byte offsets and content lengths."""
    return _parse_shx_bytes(shx_path.read_bytes())


def _parse_shx_bytes(raw: bytes) -> tuple[ShpHeader, np.ndarray, np.ndarray]:
    """Parse SHX bytes to get record byte offsets and content lengths.

    The SHX data has a 100-byte header (same format as .shp) followed by
    pairs of (offset, content_length) as big-endian int32 values in
    16-bit words.

    Returns
    -------
    header : ShpHeader
        Parsed file header with shape_type, bbox, and n_records.
    offsets : np.ndarray
        int64 byte offsets for each record in the .shp file.
    content_lengths : np.ndarray
        int64 content lengths (bytes) for each record.
    """
    if len(raw) < 100:
        raise ValueError(f"SHX file too small: {len(raw)} bytes (need >= 100)")

    # File header: first 100 bytes
    # Byte 0-3: file code (BE i32, should be 9994)
    file_code = struct.unpack_from(">i", raw, 0)[0]
    if file_code != 9994:
        raise ValueError(f"Invalid SHX file code: {file_code} (expected 9994)")

    # Byte 24-27: file length in 16-bit words (BE i32)
    file_length_words = struct.unpack_from(">i", raw, 24)[0]
    file_length_bytes = file_length_words * 2

    # Byte 32-35: shape type (LE i32)
    shape_type = struct.unpack_from("<i", raw, 32)[0]

    # Byte 36-67: bounding box (8 LE f64: xmin, ymin, xmax, ymax, zmin, zmax, mmin, mmax)
    xmin, ymin, xmax, ymax = struct.unpack_from("<4d", raw, 36)

    # Number of records: (file_length - 100 header) / 8 bytes per record
    n_records = (file_length_bytes - 100) // 8

    if n_records < 0:
        raise ValueError(f"Invalid SHX file: computed n_records={n_records}")

    header = ShpHeader(
        shape_type=shape_type,
        bbox=(xmin, ymin, xmax, ymax),
        n_records=n_records,
    )

    if n_records == 0:
        return header, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    # Parse record index entries: each is 2 big-endian int32 (offset, content_length)
    # Values are in 16-bit words, so multiply by 2 for byte offsets/lengths.
    index_data = raw[100 : 100 + n_records * 8]
    # Reshape as (n_records, 2) big-endian int32
    idx_array = np.frombuffer(index_data, dtype=np.dtype(">i4")).reshape(n_records, 2)
    # Convert from 16-bit words to bytes
    offsets = idx_array[:, 0].astype(np.int64) * 2
    content_lengths = idx_array[:, 1].astype(np.int64) * 2

    return header, offsets, content_lengths


def _read_shp_header(shp_path: Path) -> ShpHeader:
    """Read just the SHP file header (100 bytes).

    Used as a fallback when no SHX file is available.
    """
    with open(shp_path, "rb") as f:
        raw = f.read(100)

    if len(raw) < 100:
        raise ValueError(f"SHP file too small: {len(raw)} bytes (need >= 100)")

    file_code = struct.unpack_from(">i", raw, 0)[0]
    if file_code != 9994:
        raise ValueError(f"Invalid SHP file code: {file_code} (expected 9994)")

    shape_type = struct.unpack_from("<i", raw, 32)[0]
    xmin, ymin, xmax, ymax = struct.unpack_from("<4d", raw, 36)

    # Cannot compute n_records without SHX
    return ShpHeader(
        shape_type=shape_type,
        bbox=(xmin, ymin, xmax, ymax),
        n_records=-1,
    )


# ---------------------------------------------------------------------------
# Phase 3 + 4: GPU kernel decode + offset assembly
# ---------------------------------------------------------------------------


def _decode_points(
    d_shp: cp.ndarray,
    d_offsets: cp.ndarray,
    n_records: int,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Decode Point records on GPU.

    Parameters
    ----------
    d_shp : cp.ndarray
        Device-resident SHP file bytes.
    d_offsets : cp.ndarray
        int64 record byte offsets (device).
    n_records : int
        Number of records.

    Returns
    -------
    d_x, d_y : cp.ndarray
        Device-resident float64 coordinate arrays, shape (n_records,).
    """
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    d_x = cp.empty(n_records, dtype=cp.float64)
    d_y = cp.empty(n_records, dtype=cp.float64)

    if n_records == 0:
        return d_x, d_y

    kernels = _point_kernels()
    kernel = kernels["shp_decode_points"]
    grid, block = runtime.launch_config(kernel, n_records)
    params = (
        (
            ptr(d_shp),
            ptr(d_offsets),
            ptr(d_x),
            ptr(d_y),
            np.int32(n_records),
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
        ),
    )
    runtime.launch(kernel, grid=grid, block=block, params=params)
    return d_x, d_y


def _count_scatter_complex(
    d_shp: cp.ndarray,
    d_offsets: cp.ndarray,
    d_content_lengths: cp.ndarray,
    n_records: int,
    shape_type: int,
) -> tuple[
    cp.ndarray,
    cp.ndarray,
    cp.ndarray,
    cp.ndarray,
    cp.ndarray,
    int,
    int,
    cp.ndarray | None,
    cp.ndarray,
    cp.ndarray,
]:
    """Two-pass count-scatter for PolyLine/Polygon/MultiPoint records.

    Pass 1: Count num_parts and num_points per record.
    Prefix sum: Build coordinate offsets and part offsets.
    Pass 2: Gather coordinates and (for PolyLine/Polygon) part indices.

    Returns
    -------
    d_x, d_y : cp.ndarray
        Flat coordinate arrays (device, float64).
    d_coord_offsets : cp.ndarray
        Per-record coordinate start indices (device, int32).
    d_part_offsets_prefix : cp.ndarray
        Per-record part start indices (device, int32).
    d_is_null : cp.ndarray
        Per-record null flags (device, int32).
    total_points : int
        Total coordinate count.
    total_parts : int
        Total part count.
    d_ring_offsets_flat : cp.ndarray or None
        Flat ring/part offsets (device, int32) for PolyLine/Polygon, None for MultiPoint.
    d_num_parts : cp.ndarray
        Per-record part counts (device, int32).
    d_num_points : cp.ndarray
        Per-record point counts (device, int32).
    """
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    # ---- Pass 1: count parts and points per record ----
    d_num_parts = cp.empty(n_records, dtype=cp.int32)
    d_num_points = cp.empty(n_records, dtype=cp.int32)
    d_is_null = cp.empty(n_records, dtype=cp.int32)

    if n_records > 0:
        kernels = _count_kernels()
        kernel = kernels["shp_count_parts_points"]
        grid, block = runtime.launch_config(kernel, n_records)
        params = (
            (
                ptr(d_shp),
                ptr(d_offsets),
                ptr(d_content_lengths),
                ptr(d_num_parts),
                ptr(d_num_points),
                ptr(d_is_null),
                np.int32(n_records),
                np.int32(shape_type),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
            ),
        )
        runtime.launch(kernel, grid=grid, block=block, params=params)

    # ---- Prefix sums (CCCL exclusive_sum, no sync needed -- same stream) ----
    d_coord_offsets = exclusive_sum(d_num_points, synchronize=False)
    d_part_offsets_prefix = exclusive_sum(d_num_parts, synchronize=False)

    # Get totals: last element of prefix sum + last element of counts
    # Single sync before host read
    runtime.synchronize()
    total_points = (
        int(d_coord_offsets[n_records - 1].get()) + int(d_num_points[n_records - 1].get())
        if n_records > 0
        else 0
    )
    total_parts = (
        int(d_part_offsets_prefix[n_records - 1].get()) + int(d_num_parts[n_records - 1].get())
        if n_records > 0
        else 0
    )

    # Wait -- exclusive_sum returns length n+1 with the total at index n.
    # Check the actual shape returned by exclusive_sum.
    # exclusive_sum produces output of same length as input. Total = offsets[-1] + counts[-1].
    # Actually let's verify by checking the CCCL exclusive_sum semantics:
    # exclusive_sum(counts) returns offsets where offsets[i] = sum(counts[0:i])
    # So total = offsets[n-1] + counts[n-1], and we already got that above.

    # ---- Pass 2: gather coordinates ----
    d_x = cp.empty(total_points, dtype=cp.float64)
    d_y = cp.empty(total_points, dtype=cp.float64)

    if total_points > 0 and n_records > 0:
        kernels = _gather_coords_kernels()
        kernel = kernels["shp_gather_coords"]
        grid, block = runtime.launch_config(kernel, n_records)
        params = (
            (
                ptr(d_shp),
                ptr(d_offsets),
                ptr(d_coord_offsets),
                ptr(d_num_parts),
                ptr(d_num_points),
                ptr(d_is_null),
                ptr(d_x),
                ptr(d_y),
                np.int32(n_records),
                np.int32(shape_type),
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
                KERNEL_PARAM_I32,
            ),
        )
        runtime.launch(kernel, grid=grid, block=block, params=params)

    # ---- Gather part/ring indices for PolyLine/Polygon ----
    d_ring_offsets_flat = None
    if shape_type in (SHP_POLYLINE, SHP_POLYGON) and total_parts > 0 and n_records > 0:
        # Allocate flat ring offsets: total_parts entries + 1 sentinel
        d_ring_offsets_flat = cp.empty(total_parts + 1, dtype=cp.int32)

        kernels = _gather_parts_kernels()
        kernel = kernels["shp_gather_parts"]
        grid, block = runtime.launch_config(kernel, n_records)
        params = (
            (
                ptr(d_shp),
                ptr(d_offsets),
                ptr(d_part_offsets_prefix),
                ptr(d_coord_offsets),
                ptr(d_num_parts),
                ptr(d_num_points),
                ptr(d_is_null),
                ptr(d_ring_offsets_flat),
                np.int32(n_records),
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
        runtime.launch(kernel, grid=grid, block=block, params=params)

        # Set the sentinel: last element = total_points
        d_ring_offsets_flat[total_parts] = np.int32(total_points)

    return (
        d_x,
        d_y,
        d_coord_offsets,
        d_part_offsets_prefix,
        d_is_null,
        total_points,
        total_parts,
        d_ring_offsets_flat,
        d_num_parts,
        d_num_points,
    )


# ---------------------------------------------------------------------------
# Phase 5: Assembly into OwnedGeometryArray
# ---------------------------------------------------------------------------


def _assemble_points(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    n_records: int,
) -> OwnedGeometryArray:
    """Build OwnedGeometryArray for Point geometries."""
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.io.pylibcudf import _build_device_single_family_owned

    d_validity = cp.ones(n_records, dtype=cp.bool_)
    d_empty_mask = cp.zeros(n_records, dtype=cp.bool_)
    d_geom_offsets = cp.arange(n_records + 1, dtype=cp.int32)

    return _build_device_single_family_owned(
        family=GeometryFamily.POINT,
        validity_device=d_validity,
        x_device=d_x,
        y_device=d_y,
        geometry_offsets_device=d_geom_offsets,
        empty_mask_device=d_empty_mask,
        detail="GPU SHP binary decode (Point)",
    )


def _assemble_multipoint(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    d_coord_offsets: cp.ndarray,
    d_is_null: cp.ndarray,
    n_records: int,
    total_points: int,
) -> OwnedGeometryArray:
    """Build OwnedGeometryArray for MultiPoint geometries."""
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.io.pylibcudf import _build_device_single_family_owned

    d_null_mask = d_is_null.astype(cp.bool_)
    d_validity = ~d_null_mask

    # geometry_offsets: same as coord_offsets but with sentinel
    d_geom_offsets = cp.empty(n_records + 1, dtype=cp.int32)
    d_geom_offsets[:n_records] = d_coord_offsets
    d_geom_offsets[n_records] = np.int32(total_points)

    d_empty_mask = d_null_mask | (d_geom_offsets[1:] == d_geom_offsets[:n_records])

    return _build_device_single_family_owned(
        family=GeometryFamily.MULTIPOINT,
        validity_device=d_validity,
        x_device=d_x,
        y_device=d_y,
        geometry_offsets_device=d_geom_offsets,
        empty_mask_device=d_empty_mask,
        detail="GPU SHP binary decode (MultiPoint)",
    )


def _assemble_linestring(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    d_coord_offsets: cp.ndarray,
    d_ring_offsets_flat: cp.ndarray,
    d_part_offsets_prefix: cp.ndarray,
    d_is_null: cp.ndarray,
    d_num_parts: cp.ndarray,
    n_records: int,
    total_points: int,
    total_parts: int,
) -> OwnedGeometryArray:
    """Build OwnedGeometryArray for PolyLine (LineString) geometries.

    SHP PolyLines can have multiple parts (which map to MultiLineString
    in GIS parlance).  When every record has exactly one part, we use
    the simpler LineString family.  Otherwise, we use MultiLineString.
    """
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.io.pylibcudf import _build_device_single_family_owned

    d_null_mask = d_is_null.astype(cp.bool_)
    d_validity = ~d_null_mask

    # Check if all records have exactly 1 part (simple LineString)
    max_parts = int(d_num_parts.max().get()) if n_records > 0 else 0

    if max_parts <= 1:
        # Simple LineString: geometry_offsets = coord_offsets with sentinel
        d_geom_offsets = cp.empty(n_records + 1, dtype=cp.int32)
        d_geom_offsets[:n_records] = d_coord_offsets
        d_geom_offsets[n_records] = np.int32(total_points)

        d_empty_mask = d_null_mask | (d_geom_offsets[1:] == d_geom_offsets[:n_records])

        return _build_device_single_family_owned(
            family=GeometryFamily.LINESTRING,
            validity_device=d_validity,
            x_device=d_x,
            y_device=d_y,
            geometry_offsets_device=d_geom_offsets,
            empty_mask_device=d_empty_mask,
            detail="GPU SHP binary decode (LineString)",
        )

    # MultiLineString: geometry_offsets = part_offsets_prefix with sentinel
    # part_offsets = ring_offsets_flat with sentinel
    d_geom_offsets = cp.empty(n_records + 1, dtype=cp.int32)
    d_geom_offsets[:n_records] = d_part_offsets_prefix
    d_geom_offsets[n_records] = np.int32(total_parts)

    d_empty_mask = d_null_mask | (d_geom_offsets[1:] == d_geom_offsets[:n_records])

    return _build_device_single_family_owned(
        family=GeometryFamily.MULTILINESTRING,
        validity_device=d_validity,
        x_device=d_x,
        y_device=d_y,
        geometry_offsets_device=d_geom_offsets,
        empty_mask_device=d_empty_mask,
        part_offsets_device=d_ring_offsets_flat,
        detail="GPU SHP binary decode (MultiLineString)",
    )


def _assemble_polygon(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    d_coord_offsets: cp.ndarray,
    d_ring_offsets_flat: cp.ndarray,
    d_part_offsets_prefix: cp.ndarray,
    d_is_null: cp.ndarray,
    d_num_parts: cp.ndarray,
    n_records: int,
    total_points: int,
    total_parts: int,
) -> OwnedGeometryArray:
    """Build OwnedGeometryArray for Polygon geometries.

    SHP Polygons have parts (rings): the first ring is the exterior,
    remaining rings are holes.  Each SHP record maps to one Polygon,
    so geometry_offsets tracks per-record ring counts, and ring_offsets
    tracks per-ring coordinate counts.
    """
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.io.pylibcudf import _build_device_single_family_owned

    d_null_mask = d_is_null.astype(cp.bool_)
    d_validity = ~d_null_mask

    # For Polygon: geometry_offsets = per-record ring count prefix sums
    # ring_offsets = per-ring coordinate count prefix sums
    # geometry_offsets[i] = part_offsets_prefix[i], sentinel = total_parts
    d_geom_offsets = cp.empty(n_records + 1, dtype=cp.int32)
    d_geom_offsets[:n_records] = d_part_offsets_prefix
    d_geom_offsets[n_records] = np.int32(total_parts)

    d_empty_mask = d_null_mask | (d_geom_offsets[1:] == d_geom_offsets[:n_records])

    return _build_device_single_family_owned(
        family=GeometryFamily.POLYGON,
        validity_device=d_validity,
        x_device=d_x,
        y_device=d_y,
        geometry_offsets_device=d_geom_offsets,
        empty_mask_device=d_empty_mask,
        ring_offsets_device=d_ring_offsets_flat,
        detail="GPU SHP binary decode (Polygon)",
    )


# ---------------------------------------------------------------------------
# Phase 5: Public API
# ---------------------------------------------------------------------------


def read_shp_gpu(shp_path: Path | str) -> OwnedGeometryArray:
    """Read an SHP file directly on GPU -- no WKB intermediate.

    Requires both .shp and .shx files.  The .shx index is parsed on CPU
    (tiny: 8 bytes per record), the .shp binary is bulk-transferred to
    device memory, and NVRTC kernels extract coordinates directly from
    the SHP binary format.

    Parameters
    ----------
    shp_path : Path or str
        Path to the .shp file.  The corresponding .shx file must exist
        at the same location with the same stem.

    Returns
    -------
    OwnedGeometryArray
        Device-resident geometry array.

    Raises
    ------
    ImportError
        If CuPy is not available.
    FileNotFoundError
        If the .shx file does not exist.
    ValueError
        If the SHP file contains an unsupported shape type.
    """

    if cp is None:
        raise ImportError("CuPy is required for read_shp_gpu")

    shp_path = Path(shp_path)

    # Handle .shp.zip and .zip archives: extract SHP/SHX to memory.
    if shp_path.suffix.lower() == ".zip" or str(shp_path).lower().endswith(".shp.zip"):
        shp_bytes, shx_bytes = _extract_shp_from_zip(shp_path)
        header, h_offsets, h_content_lengths = _read_shx_index_from_bytes(shx_bytes)

        if header.n_records == 0:
            return _make_empty_owned(header.shape_type)

        n_records = header.n_records
        d_shp = cp.frombuffer(shp_bytes, dtype=cp.uint8).copy()
    else:
        shx_path = shp_path.with_suffix(".shx")

        if not shp_path.exists():
            raise FileNotFoundError(f"SHP file not found: {shp_path}")
        if not shx_path.exists():
            raise FileNotFoundError(f"SHX file not found: {shx_path}")

        # ---- Phase 1: CPU -- parse SHX index ----
        header, h_offsets, h_content_lengths = _read_shx_index(shx_path)

        if header.n_records == 0:
            return _make_empty_owned(header.shape_type)

        n_records = header.n_records

        # ---- Phase 2: GPU -- bulk read SHP file to device ----
        from vibespatial.io.kvikio_reader import read_file_to_device

        shp_size = shp_path.stat().st_size
        file_result = read_file_to_device(shp_path, shp_size)
        d_shp = file_result.device_bytes

    # Transfer record offsets and content lengths to device
    d_offsets = cp.asarray(h_offsets)
    d_content_lengths = cp.asarray(h_content_lengths)

    # ---- Phase 3-5: Decode and assemble based on shape type ----
    if header.shape_type == SHP_POINT:
        d_x, d_y = _decode_points(d_shp, d_offsets, n_records)
        return _assemble_points(d_x, d_y, n_records)

    if header.shape_type == SHP_MULTIPOINT:
        result = _count_scatter_complex(
            d_shp,
            d_offsets,
            d_content_lengths,
            n_records,
            SHP_MULTIPOINT,
        )
        (
            d_x,
            d_y,
            d_coord_offsets,
            d_part_offsets,
            d_is_null,
            total_points,
            total_parts,
            _,
            d_num_parts,
            d_num_points,
        ) = result
        return _assemble_multipoint(
            d_x,
            d_y,
            d_coord_offsets,
            d_is_null,
            n_records,
            total_points,
        )

    if header.shape_type == SHP_POLYLINE:
        result = _count_scatter_complex(
            d_shp,
            d_offsets,
            d_content_lengths,
            n_records,
            SHP_POLYLINE,
        )
        (
            d_x,
            d_y,
            d_coord_offsets,
            d_part_offsets,
            d_is_null,
            total_points,
            total_parts,
            d_ring_flat,
            d_num_parts,
            d_num_points,
        ) = result
        return _assemble_linestring(
            d_x,
            d_y,
            d_coord_offsets,
            d_ring_flat,
            d_part_offsets,
            d_is_null,
            d_num_parts,
            n_records,
            total_points,
            total_parts,
        )

    if header.shape_type == SHP_POLYGON:
        result = _count_scatter_complex(
            d_shp,
            d_offsets,
            d_content_lengths,
            n_records,
            SHP_POLYGON,
        )
        (
            d_x,
            d_y,
            d_coord_offsets,
            d_part_offsets,
            d_is_null,
            total_points,
            total_parts,
            d_ring_flat,
            d_num_parts,
            d_num_points,
        ) = result
        return _assemble_polygon(
            d_x,
            d_y,
            d_coord_offsets,
            d_ring_flat,
            d_part_offsets,
            d_is_null,
            d_num_parts,
            n_records,
            total_points,
            total_parts,
        )

    # Null-only files: all records are Null shape
    if header.shape_type == SHP_NULL:
        return _make_empty_owned(SHP_POINT)

    raise ValueError(
        f"Unsupported SHP shape type {header.shape_type}. "
        f"Supported types: Point (1), PolyLine (3), Polygon (5), MultiPoint (8)."
    )


def _make_empty_owned(shape_type: int) -> OwnedGeometryArray:
    """Create an empty OwnedGeometryArray for the given shape type."""
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.io.pylibcudf import _build_device_single_family_owned

    family_map = {
        SHP_POINT: GeometryFamily.POINT,
        SHP_POLYLINE: GeometryFamily.LINESTRING,
        SHP_POLYGON: GeometryFamily.POLYGON,
        SHP_MULTIPOINT: GeometryFamily.MULTIPOINT,
        SHP_NULL: GeometryFamily.POINT,
    }
    family = family_map.get(shape_type, GeometryFamily.POINT)

    d_validity = cp.empty(0, dtype=cp.bool_)
    d_empty_mask = cp.empty(0, dtype=cp.bool_)
    d_x = cp.empty(0, dtype=cp.float64)
    d_y = cp.empty(0, dtype=cp.float64)
    d_geom_offsets = cp.zeros(1, dtype=cp.int32)

    return _build_device_single_family_owned(
        family=family,
        validity_device=d_validity,
        x_device=d_x,
        y_device=d_y,
        geometry_offsets_device=d_geom_offsets,
        empty_mask_device=d_empty_mask,
        detail=f"GPU SHP binary decode (empty, shape_type={shape_type})",
    )
