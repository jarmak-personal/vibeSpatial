"""GPU byte-classification GeoJSON parser.

Parses GeoJSON FeatureCollection files on GPU using NVRTC kernels for
byte classification, structural scanning, coordinate extraction, and
ASCII-to-float64 parsing.  Property extraction stays on CPU (hybrid
design per vibeSpatial GPU memory policy).

v1 targets homogeneous Polygon files.  Mixed geometry types and chunked
processing for files exceeding GPU memory are deferred to v2.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from vibespatial.cuda_runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.geometry_buffers import GeometryFamily
from vibespatial.io_pylibcudf import _build_device_single_family_owned
from vibespatial.owned_geometry import OwnedGeometryArray

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

# ctypes for int64 kernel params (not in cuda_runtime.py which only has i32)
KERNEL_PARAM_I64 = ctypes.c_longlong

# ---------------------------------------------------------------------------
# Kernel sources (all Tier 1 NVRTC)
# ---------------------------------------------------------------------------

_CLASSIFY_SOURCE = r"""
extern "C" __global__ void classify_bytes(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    long long n
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    unsigned char b = input[idx];
    unsigned char cls;

    switch (b) {
        case '{': cls = 1; break;
        case '}': cls = 2; break;
        case '[': cls = 3; break;
        case ']': cls = 4; break;
        case '"': cls = 5; break;
        case ':': cls = 6; break;
        case ',': cls = 7; break;
        case '-': case '+': case '.':
        case 'e': case 'E':
        case '0': case '1': case '2': case '3': case '4':
        case '5': case '6': case '7': case '8': case '9':
            cls = 8; break;
        default: cls = 0; break;
    }
    output[idx] = cls;
}
"""

_DEPTH_DELTAS_SOURCE = r"""
extern "C" __global__ void compute_depth_deltas(
    const unsigned char* __restrict__ input,
    const unsigned char* __restrict__ quote_parity,
    signed char* __restrict__ deltas,
    long long n
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    signed char d = 0;
    if (quote_parity[idx] == 0) {
        unsigned char b = input[idx];
        if (b == '{' || b == '[') d = 1;
        else if (b == '}' || b == ']') d = -1;
    }
    deltas[idx] = d;
}
"""

_QUOTE_SOURCE = r"""
extern "C" __global__ void quote_toggle(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    long long n
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    unsigned char b = input[idx];
    if (b != '"') {
        output[idx] = 0;
        return;
    }
    // Check for backslash escape: count consecutive backslashes before this quote
    int backslash_count = 0;
    long long j = idx - 1;
    while (j >= 0 && input[j] == '\\') {
        backslash_count++;
        j--;
    }
    // Quote is escaped if preceded by odd number of backslashes
    output[idx] = (backslash_count % 2 == 0) ? 1 : 0;
}
"""

_COORD_KEY_SOURCE = r"""
extern "C" __global__ void find_coord_key(
    const unsigned char* __restrict__ input,
    const unsigned char* __restrict__ quote_parity,
    unsigned char* __restrict__ hits,
    long long n
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > n - 14) {
        if (idx < n) hits[idx] = 0;
        return;
    }

    // Pattern: "coordinates":  (14 bytes)
    const unsigned char pat[14] = {
        '"','c','o','o','r','d','i','n','a','t','e','s','"',':'
    };

    unsigned char match = 1;
    for (int i = 0; i < 14; ++i) {
        if (input[idx + i] != pat[i]) { match = 0; break; }
    }
    // Check quote parity at the colon (idx+13): for a real JSON key the
    // opening and closing quotes cancel, so parity is 0 (even).
    // Inside a string value the parity is 1 (odd).
    if (match && quote_parity[idx + 13] != 0) {
        match = 0;
    }
    hits[idx] = match;
}
"""

_NUM_BOUNDS_SOURCE = r"""
extern "C" __global__ void find_number_boundaries(
    const unsigned char* __restrict__ input,
    const unsigned char* __restrict__ quote_parity,
    unsigned char* __restrict__ is_start,
    unsigned char* __restrict__ is_end,
    long long n
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Skip bytes inside string values (odd parity)
    if (quote_parity[idx] != 0) {
        is_start[idx] = 0;
        is_end[idx] = 0;
        return;
    }

    unsigned char c = input[idx];
    unsigned char prev = (idx > 0) ? input[idx - 1] : '[';
    unsigned char next = (idx < n - 1) ? input[idx + 1] : ']';

    // Number starts: first char of a number, preceded by separator
    // Space included because json.dumps uses ", " as separator
    unsigned char is_first_digit = (c >= '0' && c <= '9') || c == '-' || c == '+';
    unsigned char is_sep_before = (prev == ',' || prev == '[' || prev == ' '
                                   || prev == '\n' || prev == '\r' || prev == '\t');
    is_start[idx] = is_first_digit && is_sep_before;

    // Number ends: last numeric char followed by separator
    unsigned char is_numeric = (c >= '0' && c <= '9') || c == '.' ||
                               c == 'e' || c == 'E' || c == '-' || c == '+';
    unsigned char is_sep_after = (next == ',' || next == ']');
    is_end[idx] = is_numeric && is_sep_after;
}
"""

_PARSE_FLOAT_SOURCE = r"""
extern "C" __global__ void parse_ascii_floats(
    const unsigned char* __restrict__ input,
    const long long* __restrict__ coord_starts,
    const long long* __restrict__ coord_ends,
    double* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    long long start = coord_starts[idx];
    long long end = coord_ends[idx];

    double result = 0.0;
    double frac_mult = 0.0;
    int negative = 0;
    int in_exponent = 0;
    int exp_val = 0;
    int exp_negative = 0;

    for (long long i = start; i < end; ++i) {
        unsigned char c = input[i];
        if (c == '-') {
            if (in_exponent) exp_negative = 1;
            else negative = 1;
        } else if (c == '+') {
            // skip
        } else if (c == '.') {
            frac_mult = 0.1;
        } else if (c == 'e' || c == 'E') {
            in_exponent = 1;
        } else if (c >= '0' && c <= '9') {
            int d = c - '0';
            if (in_exponent) {
                exp_val = exp_val * 10 + d;
            } else if (frac_mult > 0.0) {
                result += d * frac_mult;
                frac_mult *= 0.1;
            } else {
                result = result * 10.0 + d;
            }
        }
    }

    if (negative) result = -result;

    if (in_exponent) {
        double exp_mult = 1.0;
        for (int e = 0; e < exp_val; ++e) exp_mult *= 10.0;
        if (exp_negative) result /= exp_mult;
        else result *= exp_mult;
    }

    output[idx] = result;
}
"""

_COORD_SPAN_END_SOURCE = r"""
extern "C" __global__ void coord_span_end(
    const int* __restrict__ depth,
    const long long* __restrict__ coord_positions,
    long long* __restrict__ coord_ends,
    int n_features,
    long long n_bytes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_features) return;

    // Start scanning after "coordinates": — find the opening bracket
    long long pos = coord_positions[idx] + 14;
    // Skip whitespace to find opening '['
    while (pos < n_bytes && depth[pos] == depth[pos - 1]) {
        pos++;
    }
    if (pos >= n_bytes) {
        coord_ends[idx] = pos;
        return;
    }
    int start_depth = depth[pos];
    // Scan forward until depth drops below start_depth
    pos++;
    while (pos < n_bytes && depth[pos] >= start_depth) {
        pos++;
    }
    coord_ends[idx] = pos;
}
"""

_RING_COUNT_SOURCE = r"""
extern "C" __global__ void count_rings_and_coords(
    const unsigned char* __restrict__ input,
    const int* __restrict__ depth,
    const long long* __restrict__ coord_starts,
    const long long* __restrict__ coord_ends,
    int* __restrict__ ring_counts,
    int* __restrict__ coord_pair_counts,
    int n_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_features) return;

    long long start = coord_starts[idx] + 14;
    long long end = coord_ends[idx];
    if (start >= end) {
        ring_counts[idx] = 0;
        coord_pair_counts[idx] = 0;
        return;
    }

    // Find the depth at the opening '[' of coordinates value
    // Skip to find opening bracket
    while (start < end && input[start] != '[') start++;
    if (start >= end) {
        ring_counts[idx] = 0;
        coord_pair_counts[idx] = 0;
        return;
    }
    int coord_depth = depth[start];  // depth at outer '[' of coordinates

    // For Polygon: coordinates is [[[x,y], ...], [[x,y], ...]]
    // Depth at opening brackets is inclusive (cumsum includes +1 delta).
    // At closing ']', depth includes the -1 delta, so:
    //   ring-closing ']' has depth = coord_depth (was coord_depth+1, minus 1)
    //   pair-closing ']' has depth = coord_depth + 1 (was coord_depth+2, minus 1)
    int rings = 0;
    int pairs = 0;
    int ring_close_depth = coord_depth;
    int pair_close_depth = coord_depth + 1;

    for (long long i = start + 1; i < end; i++) {
        unsigned char c = input[i];
        int d = depth[i];
        if (c == ']' && d == ring_close_depth) {
            rings++;
        }
        if (c == ']' && d == pair_close_depth) {
            pairs++;
        }
    }

    ring_counts[idx] = rings;
    coord_pair_counts[idx] = pairs;
}
"""

_SCATTER_COORDS_SOURCE = r"""
extern "C" __global__ void scatter_ring_offsets(
    const unsigned char* __restrict__ input,
    const int* __restrict__ depth,
    const long long* __restrict__ coord_starts,
    const long long* __restrict__ coord_ends,
    const int* __restrict__ ring_offset_starts,
    const int* __restrict__ coord_pair_offset_starts,
    int* __restrict__ ring_offsets,
    int n_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_features) return;

    long long start = coord_starts[idx] + 14;
    long long end = coord_ends[idx];
    int ring_out = ring_offset_starts[idx];
    int pair_out = coord_pair_offset_starts[idx];

    // Skip to opening bracket
    while (start < end && input[start] != '[') start++;
    if (start >= end) return;
    int coord_depth = depth[start];
    int ring_close_depth = coord_depth;
    int pair_close_depth = coord_depth + 1;

    // Write starting offset for first ring
    ring_offsets[ring_out] = pair_out;

    int rings_seen = 0;
    int pairs_seen = 0;

    for (long long i = start + 1; i < end; i++) {
        unsigned char c = input[i];
        int d = depth[i];
        if (c == ']' && d == pair_close_depth) {
            pairs_seen++;
        }
        if (c == ']' && d == ring_close_depth) {
            rings_seen++;
            // Write end offset for this ring = start offset for next ring
            ring_offsets[ring_out + rings_seen] = pair_out + pairs_seen;
        }
    }
}
"""

_FEATURE_BOUNDARY_SOURCE = r"""
extern "C" __global__ void find_feature_boundaries(
    const unsigned char* __restrict__ input,
    const int* __restrict__ depth,
    unsigned char* __restrict__ is_feature_start,
    unsigned char* __restrict__ is_feature_end,
    long long n
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    unsigned char c = input[idx];
    int d = depth[idx];

    // Features are objects at depth 3: FeatureCollection { depth=1,
    // "features": [ depth=2, Feature { depth=3. Closing } drops to depth=2.
    is_feature_start[idx] = (c == '{' && d == 3) ? 1 : 0;
    is_feature_end[idx] = (c == '}' && d == 2) ? 1 : 0;
}
"""

_MARK_COORD_SPANS_SOURCE = r"""
extern "C" __global__ void mark_coord_spans(
    const long long* __restrict__ coord_positions,
    const long long* __restrict__ coord_ends,
    unsigned char* __restrict__ mask,
    int n_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_features) return;

    long long start = coord_positions[idx] + 14;
    long long end = coord_ends[idx];
    for (long long i = start; i < end; i++) {
        mask[i] = 1;
    }
}
"""

# Kernel name tuples
_DEPTH_DELTAS_NAMES = ("compute_depth_deltas",)
_CLASSIFY_NAMES = ("classify_bytes",)
_QUOTE_NAMES = ("quote_toggle",)
_COORD_KEY_NAMES = ("find_coord_key",)
_NUM_BOUNDS_NAMES = ("find_number_boundaries",)
_PARSE_FLOAT_NAMES = ("parse_ascii_floats",)
_COORD_SPAN_END_NAMES = ("coord_span_end",)
_RING_COUNT_NAMES = ("count_rings_and_coords",)
_SCATTER_COORDS_NAMES = ("scatter_ring_offsets",)
_FEATURE_BOUNDARY_NAMES = ("find_feature_boundaries",)
_MARK_COORD_SPANS_NAMES = ("mark_coord_spans",)

# ---------------------------------------------------------------------------
# NVRTC warmup (ADR-0034 Level 2)
# ---------------------------------------------------------------------------
from vibespatial.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("geojson-depth-deltas", _DEPTH_DELTAS_SOURCE, _DEPTH_DELTAS_NAMES),
    ("geojson-classify", _CLASSIFY_SOURCE, _CLASSIFY_NAMES),
    ("geojson-quote", _QUOTE_SOURCE, _QUOTE_NAMES),
    ("geojson-coord-key", _COORD_KEY_SOURCE, _COORD_KEY_NAMES),
    ("geojson-num-bounds", _NUM_BOUNDS_SOURCE, _NUM_BOUNDS_NAMES),
    ("geojson-parse-float", _PARSE_FLOAT_SOURCE, _PARSE_FLOAT_NAMES),
    ("geojson-coord-span-end", _COORD_SPAN_END_SOURCE, _COORD_SPAN_END_NAMES),
    ("geojson-ring-count", _RING_COUNT_SOURCE, _RING_COUNT_NAMES),
    ("geojson-scatter-coords", _SCATTER_COORDS_SOURCE, _SCATTER_COORDS_NAMES),
    ("geojson-feature-boundary", _FEATURE_BOUNDARY_SOURCE, _FEATURE_BOUNDARY_NAMES),
    ("geojson-mark-spans", _MARK_COORD_SPANS_SOURCE, _MARK_COORD_SPANS_NAMES),
])


# ---------------------------------------------------------------------------
# Kernel compilation helpers
# ---------------------------------------------------------------------------

def _depth_deltas_kernels():
    return compile_kernel_group("geojson-depth-deltas", _DEPTH_DELTAS_SOURCE, _DEPTH_DELTAS_NAMES)


def _classify_kernels():
    return compile_kernel_group("geojson-classify", _CLASSIFY_SOURCE, _CLASSIFY_NAMES)


def _quote_kernels():
    return compile_kernel_group("geojson-quote", _QUOTE_SOURCE, _QUOTE_NAMES)


def _coord_key_kernels():
    return compile_kernel_group("geojson-coord-key", _COORD_KEY_SOURCE, _COORD_KEY_NAMES)


def _num_bounds_kernels():
    return compile_kernel_group("geojson-num-bounds", _NUM_BOUNDS_SOURCE, _NUM_BOUNDS_NAMES)


def _parse_float_kernels():
    return compile_kernel_group("geojson-parse-float", _PARSE_FLOAT_SOURCE, _PARSE_FLOAT_NAMES)


def _coord_span_end_kernels():
    return compile_kernel_group("geojson-coord-span-end", _COORD_SPAN_END_SOURCE, _COORD_SPAN_END_NAMES)


def _ring_count_kernels():
    return compile_kernel_group("geojson-ring-count", _RING_COUNT_SOURCE, _RING_COUNT_NAMES)


def _scatter_coords_kernels():
    return compile_kernel_group("geojson-scatter-coords", _SCATTER_COORDS_SOURCE, _SCATTER_COORDS_NAMES)


def _feature_boundary_kernels():
    return compile_kernel_group("geojson-feature-boundary", _FEATURE_BOUNDARY_SOURCE, _FEATURE_BOUNDARY_NAMES)


def _mark_coord_spans_kernels():
    return compile_kernel_group("geojson-mark-spans", _MARK_COORD_SPANS_SOURCE, _MARK_COORD_SPANS_NAMES)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GeoJSONGpuResult:
    owned: OwnedGeometryArray
    n_features: int
    host_bytes: np.ndarray
    feature_starts: np.ndarray
    feature_ends: np.ndarray

    def properties_loader(self) -> Callable[[], list[dict[str, object]]]:
        host_bytes = self.host_bytes
        feature_starts = self.feature_starts
        feature_ends = self.feature_ends

        def _load() -> list[dict[str, object]]:
            return _extract_properties_cpu(host_bytes, feature_starts, feature_ends)
        return _load

    def extract_properties_dataframe(self):
        import pandas as pd
        props = _extract_properties_cpu(self.host_bytes, self.feature_starts, self.feature_ends)
        if not props:
            return pd.DataFrame()
        return pd.DataFrame(props)


# ---------------------------------------------------------------------------
# CPU property extraction
# ---------------------------------------------------------------------------

def _extract_properties_cpu(
    host_bytes: np.ndarray,
    feature_starts: np.ndarray,
    feature_ends: np.ndarray,
) -> list[dict[str, object]]:
    from vibespatial.io_geojson import _fast_json_loads

    raw = host_bytes.tobytes()
    result: list[dict[str, object]] = []
    for i in range(len(feature_starts)):
        start = int(feature_starts[i])
        end = int(feature_ends[i])
        feature_bytes = raw[start:end]
        try:
            feature = _fast_json_loads(feature_bytes)
            props = feature.get("properties") or {}
            result.append(dict(props))
        except Exception:
            result.append({})
    return result


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _launch_kernel(runtime, kernel, n, params):
    grid, block = runtime.launch_config(kernel, int(n))
    runtime.launch(kernel, grid=grid, block=block, params=params)


def read_geojson_gpu(path: Path) -> GeoJSONGpuResult:
    """Parse a GeoJSON file using GPU byte-classification pipeline.

    Returns a GeoJSONGpuResult with device-resident OwnedGeometryArray
    and host data for lazy CPU property extraction.
    """
    runtime = get_cuda_runtime()

    # S0: Read file to device
    host_bytes = np.fromfile(str(path), dtype=np.uint8)
    d_bytes = cp.asarray(host_bytes)
    n = len(d_bytes)
    n_i64 = np.int64(n)
    ptr = runtime.pointer

    # S1b: Quote toggle → uint8 cumsum → parity (0=outside, 1=inside string)
    # Uses uint8 cumsum (2.16 GB) instead of int32 (8.64 GB). Parity is
    # correct after overflow because 256 is even.
    quote_kernels = _quote_kernels()
    d_quote_toggle = cp.empty(n, dtype=cp.uint8)
    _launch_kernel(runtime, quote_kernels["quote_toggle"], n, (
        (ptr(d_bytes), ptr(d_quote_toggle), n_i64),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I64),
    ))
    d_quote_parity = cp.cumsum(d_quote_toggle, dtype=cp.uint8) & np.uint8(1)
    del d_quote_toggle

    # S2: Nesting depth via fused kernel + prefix sum.
    # Single kernel outputs int8 deltas (+1/-1/0), filtering out brackets
    # inside strings. Avoids materializing d_classes and boolean intermediates.
    dd_kernels = _depth_deltas_kernels()
    d_deltas = cp.empty(n, dtype=cp.int8)
    _launch_kernel(runtime, dd_kernels["compute_depth_deltas"], n, (
        (ptr(d_bytes), ptr(d_quote_parity), ptr(d_deltas), n_i64),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I64),
    ))
    d_depth = cp.cumsum(d_deltas, dtype=cp.int32)
    del d_deltas

    # S3: Find "coordinates": positions (with quote-state filter)
    coord_kernels = _coord_key_kernels()
    d_hits = cp.zeros(n, dtype=cp.uint8)
    _launch_kernel(runtime, coord_kernels["find_coord_key"], n, (
        (ptr(d_bytes), ptr(d_quote_parity), ptr(d_hits), n_i64),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I64),
    ))
    d_coord_positions = cp.flatnonzero(d_hits).astype(cp.int64)
    del d_hits
    n_features = int(len(d_coord_positions))

    if n_features == 0:
        # Empty file — return empty geometry
        d_x = cp.empty(0, dtype=cp.float64)
        d_y = cp.empty(0, dtype=cp.float64)
        d_geom_offsets = cp.zeros(1, dtype=cp.int32)
        d_ring_offsets = cp.zeros(1, dtype=cp.int32)
        owned = _build_device_single_family_owned(
            family=GeometryFamily.POLYGON,
            validity_device=cp.ones(0, dtype=cp.bool_),
            x_device=d_x,
            y_device=d_y,
            geometry_offsets_device=d_geom_offsets,
            empty_mask_device=cp.zeros(0, dtype=cp.bool_),
            ring_offsets_device=d_ring_offsets,
            detail="GPU byte-classification GeoJSON parse (empty)",
        )
        return GeoJSONGpuResult(
            owned=owned,
            n_features=0,
            host_bytes=host_bytes,
            feature_starts=np.empty(0, dtype=np.int64),
            feature_ends=np.empty(0, dtype=np.int64),
        )

    # S3b: Find coordinate span ends
    span_kernels = _coord_span_end_kernels()
    d_coord_ends = cp.empty(n_features, dtype=cp.int64)
    _launch_kernel(runtime, span_kernels["coord_span_end"], n_features, (
        (ptr(d_depth), ptr(d_coord_positions), ptr(d_coord_ends),
         np.int32(n_features), n_i64),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_I32, KERNEL_PARAM_I64),
    ))

    # S3c: Count rings and coordinate pairs per feature
    ring_kernels = _ring_count_kernels()
    d_ring_counts = cp.empty(n_features, dtype=cp.int32)
    d_pair_counts = cp.empty(n_features, dtype=cp.int32)
    _launch_kernel(runtime, ring_kernels["count_rings_and_coords"], n_features, (
        (ptr(d_bytes), ptr(d_depth), ptr(d_coord_positions), ptr(d_coord_ends),
         ptr(d_ring_counts), ptr(d_pair_counts), np.int32(n_features)),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
    ))

    # Prefix sums for offsets
    d_ring_offset_starts = cp.empty(n_features, dtype=cp.int32)
    d_pair_offset_starts = cp.empty(n_features, dtype=cp.int32)
    cp.cumsum(d_ring_counts, out=d_ring_offset_starts)
    cp.cumsum(d_pair_counts, out=d_pair_offset_starts)
    # Convert from inclusive to exclusive prefix sum
    total_rings = int(d_ring_offset_starts[-1].get())
    total_pairs = int(d_pair_offset_starts[-1].get())
    d_ring_offset_starts = cp.concatenate([cp.zeros(1, dtype=cp.int32), d_ring_offset_starts[:-1]])
    d_pair_offset_starts = cp.concatenate([cp.zeros(1, dtype=cp.int32), d_pair_offset_starts[:-1]])

    # GeoArrow Polygon layout:
    # geometry_offsets[i] indexes into ring_offsets (exclusive prefix sum of ring_counts)
    # ring_offsets[j] indexes into coordinates (coord pair offsets per ring)
    d_geometry_offsets = cp.empty(n_features + 1, dtype=cp.int32)
    d_geometry_offsets[0] = 0
    cp.cumsum(d_ring_counts, out=d_geometry_offsets[1:])

    # ring_offsets: total_rings + 1 entries
    d_ring_offsets = cp.empty(total_rings + 1, dtype=cp.int32)
    d_ring_offsets[-1] = total_pairs  # sentinel

    # The ring_offset_starts for the scatter kernel = geometry_offsets[0:n_features]
    d_ring_scatter_starts = d_geometry_offsets[:n_features].copy()

    scatter_kernels = _scatter_coords_kernels()
    _launch_kernel(runtime, scatter_kernels["scatter_ring_offsets"], n_features, (
        (ptr(d_bytes), ptr(d_depth), ptr(d_coord_positions), ptr(d_coord_ends),
         ptr(d_ring_scatter_starts), ptr(d_pair_offset_starts),
         ptr(d_ring_offsets), np.int32(n_features)),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
    ))

    # S4: Find all number boundaries (with quote-state filter)
    nb_kernels = _num_bounds_kernels()
    d_is_start = cp.zeros(n, dtype=cp.uint8)
    d_is_end = cp.zeros(n, dtype=cp.uint8)
    _launch_kernel(runtime, nb_kernels["find_number_boundaries"], n, (
        (ptr(d_bytes), ptr(d_quote_parity), ptr(d_is_start), ptr(d_is_end), n_i64),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I64),
    ))
    del d_quote_parity

    # Filter numbers to only those inside coordinate spans (exclude property numerics).
    # GPU kernel marks coordinate span byte ranges.
    mark_kernels = _mark_coord_spans_kernels()
    d_in_coords = cp.zeros(n, dtype=cp.uint8)
    _launch_kernel(runtime, mark_kernels["mark_coord_spans"], n_features, (
        (ptr(d_coord_positions), ptr(d_coord_ends), ptr(d_in_coords), np.int32(n_features)),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
    ))

    # Apply the mask: zero out starts/ends outside coordinate spans
    d_is_start = d_is_start * d_in_coords
    d_is_end = d_is_end * d_in_coords
    del d_in_coords

    d_starts = cp.flatnonzero(d_is_start).astype(cp.int64)
    d_ends = cp.flatnonzero(d_is_end).astype(cp.int64) + 1  # end is exclusive
    del d_is_start, d_is_end

    n_nums = int(len(d_starts))

    # S5: Parse ASCII floats
    pf_kernels = _parse_float_kernels()
    d_coords = cp.empty(n_nums, dtype=cp.float64)
    if n_nums > 0:
        _launch_kernel(runtime, pf_kernels["parse_ascii_floats"], n_nums, (
            (ptr(d_bytes), ptr(d_starts), ptr(d_ends), ptr(d_coords), np.int32(n_nums)),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        ))
    del d_starts, d_ends

    # S6: Split into x, y (zero-copy views)
    d_x = d_coords[0::2]
    d_y = d_coords[1::2]

    # Verify coordinate count matches expected pairs
    if total_pairs > 0 and len(d_x) != total_pairs:
        # Mismatch — the filtering may have included/excluded some numbers.
        # Truncate or pad to match the expected count from structural analysis.
        if len(d_x) > total_pairs:
            d_x = d_x[:total_pairs].copy()
            d_y = d_y[:total_pairs].copy()
        else:
            # Pad with zeros (shouldn't happen with correct data)
            pad_x = cp.zeros(total_pairs - len(d_x), dtype=cp.float64)
            pad_y = cp.zeros(total_pairs - len(d_y), dtype=cp.float64)
            d_x = cp.concatenate([d_x, pad_x])
            d_y = cp.concatenate([d_y, pad_y])

    # Make x/y contiguous for OwnedGeometryArray
    d_x = cp.ascontiguousarray(d_x)
    d_y = cp.ascontiguousarray(d_y)

    # S8: Find feature boundaries for property extraction
    fb_kernels = _feature_boundary_kernels()
    d_feat_start = cp.zeros(n, dtype=cp.uint8)
    d_feat_end = cp.zeros(n, dtype=cp.uint8)
    _launch_kernel(runtime, fb_kernels["find_feature_boundaries"], n, (
        (ptr(d_bytes), ptr(d_depth), ptr(d_feat_start), ptr(d_feat_end), n_i64),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I64),
    ))
    d_feat_start_pos = cp.flatnonzero(d_feat_start).astype(cp.int64)
    d_feat_end_pos = cp.flatnonzero(d_feat_end).astype(cp.int64) + 1  # include closing brace

    # Free GPU intermediates no longer needed
    del d_bytes, d_depth, d_feat_start, d_feat_end, d_coord_positions, d_coord_ends
    del d_pair_counts, d_pair_offset_starts, d_ring_scatter_starts
    del d_coords

    # D→H copy of feature boundaries for CPU property extraction
    h_feat_starts = cp.asnumpy(d_feat_start_pos)
    h_feat_ends = cp.asnumpy(d_feat_end_pos)
    del d_feat_start_pos, d_feat_end_pos

    # Empty geometries: features with zero rings
    d_empty_mask = (d_geometry_offsets[1:] == d_geometry_offsets[:-1])
    del d_ring_counts

    # S7: Assemble OwnedGeometryArray
    d_validity = ~d_empty_mask
    owned = _build_device_single_family_owned(
        family=GeometryFamily.POLYGON,
        validity_device=d_validity,
        x_device=d_x,
        y_device=d_y,
        geometry_offsets_device=d_geometry_offsets,
        empty_mask_device=d_empty_mask,
        ring_offsets_device=d_ring_offsets,
        detail="GPU byte-classification GeoJSON parse",
    )

    return GeoJSONGpuResult(
        owned=owned,
        n_features=n_features,
        host_bytes=host_bytes,
        feature_starts=h_feat_starts,
        feature_ends=h_feat_ends,
    )
