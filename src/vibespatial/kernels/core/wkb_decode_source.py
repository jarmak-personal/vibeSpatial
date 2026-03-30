"""CUDA kernel source for GPU WKB decode pipeline.

Contains NVRTC kernel source strings for the 5-stage WKB decode
pipeline: header scan, family partition sizing, and per-family
coordinate decode kernels.

Extracted from wkb_decode.py -- dispatch logic remains there.
"""
from __future__ import annotations

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
