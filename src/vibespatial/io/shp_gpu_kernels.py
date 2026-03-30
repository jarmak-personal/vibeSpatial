"""NVRTC kernel sources for GPU Shapefile (.shp) binary decoder."""

from __future__ import annotations

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
