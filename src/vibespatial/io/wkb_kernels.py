"""NVRTC kernel sources for WKB encode."""

from __future__ import annotations

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
    write_u32_le(out + 1, 1u);
    if (stop == start) {
        unsigned long long nan_bits = 0x7ff8000000000000ull;
        double nan_value = *reinterpret_cast<double*>(&nan_bits);
        write_f64_le(out + 5, nan_value);
        write_f64_le(out + 13, nan_value);
        return;
    }
    write_f64_le(out + 5, x[start]);
    write_f64_le(out + 13, y[start]);
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
