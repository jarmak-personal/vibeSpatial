"""NVRTC kernel sources for GPU OSM PBF reader."""

from __future__ import annotations

_WAY_COORD_GATHER_SOURCE = r"""
// Way coordinate gatherer -- each thread resolves one node reference
// by binary-searching a sorted node ID table and writing the
// corresponding lon/lat to the output arrays.
//
// Input:
//   sorted_node_ids[] -- device-resident int64, sorted ascending
//   sorted_x[]        -- fp64 longitudes parallel to sorted_node_ids
//   sorted_y[]        -- fp64 latitudes parallel to sorted_node_ids
//   way_refs[]        -- flat int64 array of all node refs across all ways
//   n_refs            -- total number of refs
//   n_nodes           -- size of the sorted node table
// Output:
//   out_x[]           -- fp64 longitudes for each ref (NaN if not found)
//   out_y[]           -- fp64 latitudes for each ref (NaN if not found)

extern "C" __global__ void __launch_bounds__(256, 4)
osm_gather_way_coords(
    const long long* __restrict__ sorted_node_ids,
    const double* __restrict__ sorted_x,
    const double* __restrict__ sorted_y,
    const long long* __restrict__ way_refs,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    int n_refs,
    long long n_nodes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_refs) return;

    long long target = way_refs[idx];

    // Binary search for target in sorted_node_ids[0..n_nodes)
    long long lo = 0;
    long long hi = n_nodes;
    while (lo < hi) {
        long long mid = lo + ((hi - lo) >> 1);
        if (sorted_node_ids[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    if (lo < n_nodes && sorted_node_ids[lo] == target) {
        out_x[idx] = sorted_x[lo];
        out_y[idx] = sorted_y[lo];
    } else {
        // Node not found -- write NaN so downstream can detect missing refs
        out_x[idx] = __longlong_as_double(0x7FF8000000000000LL);
        out_y[idx] = __longlong_as_double(0x7FF8000000000000LL);
    }
}
"""

_WAY_COORD_GATHER_NAMES: tuple[str, ...] = (
    "osm_gather_way_coords",
)

_VARINT_DECODE_SOURCE = r"""
// Protobuf varint decoder -- each thread decodes one varint at a known
// byte offset, producing a signed int64 via ZigZag decoding.
//
// Protobuf varint encoding:
//   - 7 data bits per byte, MSB is continuation flag (1 = more bytes)
//   - Signed integers use ZigZag: (n << 1) ^ (n >> 63)
//   - Maximum 10 bytes for int64
//
// Input:
//   data[]       -- raw protobuf bytes (device-resident)
//   positions[]  -- byte offset where each varint starts
//   data_len     -- total byte count of data buffer
// Output:
//   values[]     -- decoded signed int64 values
//   byte_counts[] -- number of bytes consumed per varint (optional, may be NULL)

extern "C" __global__ void __launch_bounds__(256, 4)
decode_varints_zigzag(
    const unsigned char* __restrict__ data,
    const long long* __restrict__ positions,
    long long* __restrict__ values,
    int* __restrict__ byte_counts,
    long long data_len,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    long long pos = positions[idx];
    unsigned long long raw = 0;
    int shift = 0;
    int bytes_read = 0;

    // Decode up to 10 bytes (max for 64-bit varint)
    for (int b = 0; b < 10; ++b) {
        if (pos + b >= data_len) break;
        unsigned long long byte_val = (unsigned long long)data[pos + b];
        raw |= (byte_val & 0x7FULL) << shift;
        shift += 7;
        bytes_read = b + 1;
        if ((byte_val & 0x80ULL) == 0ULL) break;
    }

    // ZigZag decode: (raw >> 1) ^ -(raw & 1)
    long long decoded = (long long)((raw >> 1) ^ (-(raw & 1ULL)));
    values[idx] = decoded;
    if (byte_counts != 0) {
        byte_counts[idx] = bytes_read;
    }
}

// Unsigned varint decoder -- same as above but without ZigZag.
// Used for field tags and lengths in protobuf.
extern "C" __global__ void __launch_bounds__(256, 4)
decode_varints_unsigned(
    const unsigned char* __restrict__ data,
    const long long* __restrict__ positions,
    unsigned long long* __restrict__ values,
    int* __restrict__ byte_counts,
    long long data_len,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    long long pos = positions[idx];
    unsigned long long raw = 0;
    int shift = 0;
    int bytes_read = 0;

    for (int b = 0; b < 10; ++b) {
        if (pos + b >= data_len) break;
        unsigned long long byte_val = (unsigned long long)data[pos + b];
        raw |= (byte_val & 0x7FULL) << shift;
        shift += 7;
        bytes_read = b + 1;
        if ((byte_val & 0x80ULL) == 0ULL) break;
    }

    values[idx] = raw;
    if (byte_counts != 0) {
        byte_counts[idx] = bytes_read;
    }
}
"""

_VARINT_DECODE_NAMES: tuple[str, ...] = (
    "decode_varints_zigzag",
    "decode_varints_unsigned",
)

