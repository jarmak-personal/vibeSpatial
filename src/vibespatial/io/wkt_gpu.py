"""GPU WKT reader -- structural analysis, coordinate extraction, and assembly.

GPU-accelerated WKT parser.  Given a device-resident byte array
containing one or more WKT geometries (one per line), this module
performs:

1. **Line splitting** -- detect newline boundaries to delimit individual
   geometry strings.
2. **Parenthesis depth** -- reuse ``gpu_parse.bracket_depth`` with
   ``open_chars="("``, ``close_chars=")"``.
3. **Geometry type classification** -- a custom NVRTC kernel scans the
   start of each geometry string and emits a family tag (POINT=0,
   LINESTRING=1, POLYGON=2, MULTIPOINT=3, MULTILINESTRING=4,
   MULTIPOLYGON=5) plus an EMPTY flag.  Handles case-insensitive
   matching and EWKT ``SRID=NNNN;`` prefixes.
4. **Coordinate extraction** -- locate coordinate regions, extract
   numeric values via gpu_parse primitives, and build per-geometry
   offset arrays.
5. **OwnedGeometryArray assembly** -- pack device-resident coordinates
   and offsets into the standard columnar geometry representation.

All operations run on the GPU with zero host materialization until the
caller explicitly requests results.

Tier classification (ADR-0033):
    - Line splitting: Tier 2 (CuPy element-wise + flatnonzero)
    - Parenthesis depth: delegates to gpu_parse.bracket_depth (Tier 1)
    - Type classification: Tier 1 (custom NVRTC -- text-specific prefix matching)
    - Coordinate region finding: Tier 1 (custom NVRTC -- paren-start scan)
    - Number extraction: delegates to gpu_parse primitives (Tier 1/2)
    - Per-geometry counting: Tier 1 (custom NVRTC -- span-local counting)
    - Ring counting: Tier 1 (custom NVRTC -- depth-aware paren counting)
    - Offset building: Tier 2 (CuPy cumsum) / CCCL exclusive_sum
    - Assembly: follows geojson_gpu.py patterns

Precision (ADR-0002):
    Structural and counting kernels are integer-only byte classification.
    No floating-point coordinate computation occurs in those kernels, so
    no PrecisionPlan is needed (same rationale as gpu_parse/structural.py).
    Coordinate parsing delegates to gpu_parse.parse_ascii_floats which
    always produces fp64 -- storage precision is always fp64 per ADR-0002.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.cuda.cccl_primitives import exclusive_sum
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    DeviceFamilyGeometryBuffer,
    OwnedGeometryArray,
)
from vibespatial.io.gpu_parse.structural import bracket_depth

if TYPE_CHECKING:
    import cupy as cp

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

# ctypes for int64 kernel params (files > 2 GB)
KERNEL_PARAM_I64 = ctypes.c_longlong


# ---------------------------------------------------------------------------
# Kernel sources (Tier 1 NVRTC) -- integer-only byte classification,
# no floating-point computation, so no PrecisionPlan needed.
# ---------------------------------------------------------------------------

_WKT_CLASSIFY_SOURCE = r"""
// WKT geometry type classification kernel.
//
// Each thread examines one geometry start position and classifies it
// by checking byte prefixes.  Case-insensitive: each byte comparison
// tests both upper and lower case via bitwise OR with 0x20 (which
// maps ASCII A-Z to a-z while leaving a-z unchanged).
//
// EWKT support: if the geometry starts with 'S' or 's' (for "SRID="),
// the kernel scans forward past the semicolon before classifying.
//
// Family tags match GeometryFamily enum order:
//   POINT=0, LINESTRING=1, POLYGON=2,
//   MULTIPOINT=3, MULTILINESTRING=4, MULTIPOLYGON=5,
//   unknown/unsupported=-2
//
// Also detects the EMPTY keyword after the type name and any
// optional dimension suffix (Z, M, ZM).

extern "C" __global__ void __launch_bounds__(256, 4)
wkt_classify_geometry_type(
    const unsigned char* __restrict__ input,
    const long long* __restrict__ geom_starts,
    signed char* __restrict__ family_tags,
    unsigned char* __restrict__ empty_flags,
    const int n_geoms,
    const long long n_bytes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_geoms) return;

    long long pos = geom_starts[idx];

    // Skip leading whitespace
    while (pos < n_bytes) {
        unsigned char c = input[pos];
        if (c != ' ' && c != '\t' && c != '\r' && c != '\n') break;
        pos++;
    }

    if (pos >= n_bytes) {
        family_tags[idx] = -2;
        empty_flags[idx] = 0;
        return;
    }

    // EWKT: handle SRID=NNNN; prefix
    // Check for 'S'/'s' followed by 'R'/'r' (start of "SRID=")
    unsigned char first = input[pos] | 0x20;  // to lowercase
    if (first == 's' && pos + 1 < n_bytes && (input[pos + 1] | 0x20) == 'r') {
        // Scan forward past the semicolon
        while (pos < n_bytes && input[pos] != ';') pos++;
        if (pos < n_bytes) pos++;  // skip the semicolon
        // Skip whitespace after semicolon
        while (pos < n_bytes) {
            unsigned char c = input[pos];
            if (c != ' ' && c != '\t' && c != '\r' && c != '\n') break;
            pos++;
        }
        if (pos >= n_bytes) {
            family_tags[idx] = -2;
            empty_flags[idx] = 0;
            return;
        }
    }

    // Now pos points to the geometry type keyword.
    // Helper macro: case-insensitive byte comparison
    #define LC(p) ((p) | 0x20)

    // Read first character (lowercased) to branch into type groups
    unsigned char c0 = LC(input[pos]);
    signed char tag = -2;
    long long type_end = pos;  // will track where the type name ends

    if (c0 == 'p') {
        // Could be POINT or POLYGON
        if (pos + 4 < n_bytes
            && LC(input[pos + 1]) == 'o'
            && LC(input[pos + 2]) == 'i'
            && LC(input[pos + 3]) == 'n'
            && LC(input[pos + 4]) == 't') {
            tag = 0;  // POINT
            type_end = pos + 5;
        } else if (pos + 6 < n_bytes
            && LC(input[pos + 1]) == 'o'
            && LC(input[pos + 2]) == 'l'
            && LC(input[pos + 3]) == 'y'
            && LC(input[pos + 4]) == 'g'
            && LC(input[pos + 5]) == 'o'
            && LC(input[pos + 6]) == 'n') {
            tag = 2;  // POLYGON
            type_end = pos + 7;
        }
    } else if (c0 == 'l') {
        // LINESTRING
        if (pos + 9 < n_bytes
            && LC(input[pos + 1]) == 'i'
            && LC(input[pos + 2]) == 'n'
            && LC(input[pos + 3]) == 'e'
            && LC(input[pos + 4]) == 's'
            && LC(input[pos + 5]) == 't'
            && LC(input[pos + 6]) == 'r'
            && LC(input[pos + 7]) == 'i'
            && LC(input[pos + 8]) == 'n'
            && LC(input[pos + 9]) == 'g') {
            tag = 1;  // LINESTRING
            type_end = pos + 10;
        }
    } else if (c0 == 'm') {
        // MULTI* types: check "MULTI" prefix first
        if (pos + 4 < n_bytes
            && LC(input[pos + 1]) == 'u'
            && LC(input[pos + 2]) == 'l'
            && LC(input[pos + 3]) == 't'
            && LC(input[pos + 4]) == 'i') {
            long long mpos = pos + 5;
            if (mpos < n_bytes) {
                unsigned char mc = LC(input[mpos]);
                if (mc == 'p') {
                    // MULTIPOINT or MULTIPOLYGON
                    if (mpos + 4 < n_bytes
                        && LC(input[mpos + 1]) == 'o'
                        && LC(input[mpos + 2]) == 'i'
                        && LC(input[mpos + 3]) == 'n'
                        && LC(input[mpos + 4]) == 't') {
                        tag = 3;  // MULTIPOINT
                        type_end = mpos + 5;
                    } else if (mpos + 6 < n_bytes
                        && LC(input[mpos + 1]) == 'o'
                        && LC(input[mpos + 2]) == 'l'
                        && LC(input[mpos + 3]) == 'y'
                        && LC(input[mpos + 4]) == 'g'
                        && LC(input[mpos + 5]) == 'o'
                        && LC(input[mpos + 6]) == 'n') {
                        tag = 5;  // MULTIPOLYGON
                        type_end = mpos + 7;
                    }
                } else if (mc == 'l') {
                    // MULTILINESTRING
                    if (mpos + 9 < n_bytes
                        && LC(input[mpos + 1]) == 'i'
                        && LC(input[mpos + 2]) == 'n'
                        && LC(input[mpos + 3]) == 'e'
                        && LC(input[mpos + 4]) == 's'
                        && LC(input[mpos + 5]) == 't'
                        && LC(input[mpos + 6]) == 'r'
                        && LC(input[mpos + 7]) == 'i'
                        && LC(input[mpos + 8]) == 'n'
                        && LC(input[mpos + 9]) == 'g') {
                        tag = 4;  // MULTILINESTRING
                        type_end = mpos + 10;
                    }
                }
            }
        }
    } else if (c0 == 'g') {
        // GEOMETRYCOLLECTION
        if (pos + 17 < n_bytes
            && LC(input[pos + 1]) == 'e'
            && LC(input[pos + 2]) == 'o'
            && LC(input[pos + 3]) == 'm'
            && LC(input[pos + 4]) == 'e'
            && LC(input[pos + 5]) == 't'
            && LC(input[pos + 6]) == 'r'
            && LC(input[pos + 7]) == 'y'
            && LC(input[pos + 8]) == 'c'
            && LC(input[pos + 9]) == 'o'
            && LC(input[pos + 10]) == 'l'
            && LC(input[pos + 11]) == 'l'
            && LC(input[pos + 12]) == 'e'
            && LC(input[pos + 13]) == 'c'
            && LC(input[pos + 14]) == 't'
            && LC(input[pos + 15]) == 'i'
            && LC(input[pos + 16]) == 'o'
            && LC(input[pos + 17]) == 'n') {
            tag = -2;  // unsupported for now
            type_end = pos + 18;
        }
    }

    family_tags[idx] = tag;

    // Detect EMPTY keyword after type name.
    // WKT allows optional dimension suffix (Z, M, ZM) between the
    // type name and EMPTY/opening paren, with or without space:
    //   POINT EMPTY, POINT Z EMPTY, POINTZ EMPTY, POINT ZM EMPTY
    long long ep = type_end;
    // Skip optional dimension suffix: Z, M, ZM (with or without space)
    if (ep < n_bytes) {
        unsigned char dc = LC(input[ep]);
        // Space before dimension suffix
        if (dc == ' ' || dc == '\t') {
            long long sp = ep;
            while (sp < n_bytes && (input[sp] == ' ' || input[sp] == '\t')) sp++;
            if (sp < n_bytes) {
                unsigned char sc = LC(input[sp]);
                if (sc == 'z' || sc == 'm') {
                    ep = sp;
                    dc = sc;
                }
            }
        }
        if (dc == 'z') {
            ep++;
            if (ep < n_bytes && LC(input[ep]) == 'm') ep++;  // ZM
        } else if (dc == 'm') {
            ep++;
        }
    }

    // Skip whitespace before EMPTY or (
    while (ep < n_bytes && (input[ep] == ' ' || input[ep] == '\t')) ep++;

    // Check for EMPTY keyword (case-insensitive)
    unsigned char is_empty = 0;
    if (ep + 4 < n_bytes
        && LC(input[ep]) == 'e'
        && LC(input[ep + 1]) == 'm'
        && LC(input[ep + 2]) == 'p'
        && LC(input[ep + 3]) == 't'
        && LC(input[ep + 4]) == 'y') {
        is_empty = 1;
    }
    empty_flags[idx] = is_empty;

    #undef LC
}
"""

_WKT_CLASSIFY_NAMES: tuple[str, ...] = ("wkt_classify_geometry_type",)


# ---------------------------------------------------------------------------
# Kernel: find the byte offset of the first '(' after each geometry's
# type keyword.  This marks where the coordinate region begins.
# ---------------------------------------------------------------------------

_WKT_FIND_PAREN_STARTS_SOURCE = r"""
// For each geometry, scan forward from its start position past the type
// keyword (and optional dimension suffix / whitespace) to find the first
// opening parenthesis '('.  If the geometry is EMPTY or has no '(',
// emit -1.
//
// The caller passes d_empty_flags so we can skip EMPTY geometries
// immediately without scanning.

extern "C" __global__ void __launch_bounds__(256, 4)
wkt_find_paren_starts(
    const unsigned char* __restrict__ input,
    const long long* __restrict__ geom_starts,
    const unsigned char* __restrict__ empty_flags,
    long long* __restrict__ paren_starts,
    const int n_geoms,
    const long long n_bytes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_geoms) return;

    if (empty_flags[idx]) {
        paren_starts[idx] = -1;
        return;
    }

    long long pos = geom_starts[idx];

    // Scan forward to find first '('
    while (pos < n_bytes && input[pos] != '(') {
        pos++;
    }

    if (pos >= n_bytes) {
        paren_starts[idx] = -1;
    } else {
        paren_starts[idx] = pos;
    }
}
"""

_WKT_FIND_PAREN_STARTS_NAMES: tuple[str, ...] = ("wkt_find_paren_starts",)


# ---------------------------------------------------------------------------
# Kernel: count coordinate *values* (individual floats) per geometry.
# Each thread scans its geometry's coordinate span [paren_start, span_end)
# and counts how many number-start markers fall within that range.
#
# This uses the d_is_num_start mask (from wkt_number_boundaries) rather
# than re-parsing bytes, enabling a simple popcount-style scan.
# ---------------------------------------------------------------------------

_WKT_COUNT_COORDS_SOURCE = r"""
// Count number-start positions within each geometry's coordinate span.
// paren_starts[i] is the byte of the opening '(' (or -1 for EMPTY).
// span_ends[i] is one past the closing ')' from span_boundaries.

extern "C" __global__ void __launch_bounds__(256, 4)
wkt_count_coords_per_geometry(
    const unsigned char* __restrict__ is_num_start,
    const long long* __restrict__ paren_starts,
    const long long* __restrict__ span_ends,
    int* __restrict__ counts,
    const int n_geoms
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_geoms) return;

    long long start = paren_starts[idx];
    if (start < 0) {
        counts[idx] = 0;
        return;
    }
    long long end = span_ends[idx];
    int count = 0;
    for (long long i = start; i < end; i++) {
        count += (int)is_num_start[i];
    }
    counts[idx] = count;
}
"""

_WKT_COUNT_COORDS_NAMES: tuple[str, ...] = ("wkt_count_coords_per_geometry",)


# ---------------------------------------------------------------------------
# Kernel: count rings per geometry for POLYGON / MULTIPOLYGON types.
#
# For POLYGON, rings are delimited by inner parentheses at depth 2.
# We count how many times depth transitions from 1->2 (opening of a ring)
# within the geometry's coordinate span.
#
# For MULTIPOLYGON, parts are at depth 2 and rings at depth 3.  We
# count depth transitions from 2->3.
#
# For other types (POINT, LINESTRING, MULTI{POINT,LINESTRING}), emit 0.
#
# The ring_depth parameter tells us what depth to look for:
#   - POLYGON: ring_depth = 2
#   - MULTIPOLYGON: ring_depth = 3
#   - Others: skip (counts[idx] = 0)
#
# We use a per-geometry approach: each thread scans its span's depth array.
# ---------------------------------------------------------------------------

_WKT_COUNT_RINGS_SOURCE = r"""
// Count rings per geometry by detecting depth transitions.
// family_tags: 0=POINT, 1=LINESTRING, 2=POLYGON,
//              3=MULTIPOINT, 4=MULTILINESTRING, 5=MULTIPOLYGON
// For POLYGON (tag 2), count depth transitions where depth[i]==2 and depth[i-1]<2
// For MULTIPOLYGON (tag 5), count transitions where depth[i]==3 and depth[i-1]<3
// For all others, emit 0.

extern "C" __global__ void __launch_bounds__(256, 4)
wkt_count_rings(
    const int* __restrict__ depth,
    const long long* __restrict__ paren_starts,
    const long long* __restrict__ span_ends,
    const signed char* __restrict__ family_tags,
    int* __restrict__ ring_counts,
    const int n_geoms
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_geoms) return;

    signed char tag = family_tags[idx];
    int ring_depth;
    if (tag == 2) {
        ring_depth = 2;       // POLYGON: rings at depth 2
    } else if (tag == 5) {
        ring_depth = 3;       // MULTIPOLYGON: rings at depth 3
    } else {
        ring_counts[idx] = 0;
        return;
    }

    long long start = paren_starts[idx];
    if (start < 0) {
        ring_counts[idx] = 0;
        return;
    }

    long long end = span_ends[idx];
    int count = 0;

    for (long long i = start + 1; i < end; i++) {
        if (depth[i] == ring_depth && depth[i - 1] < ring_depth) {
            count++;
        }
    }
    ring_counts[idx] = count;
}
"""

_WKT_COUNT_RINGS_NAMES: tuple[str, ...] = ("wkt_count_rings",)


# ---------------------------------------------------------------------------
# Kernel: WKT-aware number boundary detection.
#
# The standard gpu_parse.number_boundaries kernel recognizes '[', ']',
# ',', space, tab, newline, CR as separators.  WKT uses '(' and ')'
# instead of '[' and ']', so we need a variant that includes parentheses
# in the separator set.
#
# This kernel is structurally identical to find_number_boundaries but
# with '(' and ')' added to the separator characters.
# ---------------------------------------------------------------------------

_WKT_NUM_BOUNDS_SOURCE = r"""
extern "C" __global__ void __launch_bounds__(256, 4)
wkt_find_number_boundaries(
    const unsigned char* __restrict__ input,
    const unsigned char* __restrict__ is_coord_region,
    unsigned char* __restrict__ is_start,
    unsigned char* __restrict__ is_end,
    long long n
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Only process bytes inside coordinate regions
    if (is_coord_region[idx] == 0) {
        is_start[idx] = 0;
        is_end[idx] = 0;
        return;
    }

    unsigned char c = input[idx];
    unsigned char prev = (idx > 0) ? input[idx - 1] : '(';
    unsigned char next = (idx < n - 1) ? input[idx + 1] : ')';

    // Number starts: first char of a number, preceded by WKT separator
    // WKT separators: '(' ')' ',' space tab newline CR
    unsigned char is_first_digit = (c >= '0' && c <= '9') || c == '-' || c == '+';
    unsigned char is_sep_before = (prev == ',' || prev == '(' || prev == ')'
                                   || prev == ' ' || prev == '\n'
                                   || prev == '\r' || prev == '\t');
    is_start[idx] = is_first_digit && is_sep_before;

    // Number ends: last numeric char followed by WKT separator
    unsigned char is_numeric = (c >= '0' && c <= '9') || c == '.' ||
                               c == 'e' || c == 'E' || c == '-' || c == '+';
    unsigned char is_sep_after = (next == ',' || next == ')' || next == '('
                                  || next == ' ' || next == '\n'
                                  || next == '\r' || next == '\t');
    is_end[idx] = is_numeric && is_sep_after;
}
"""

_WKT_NUM_BOUNDS_NAMES: tuple[str, ...] = ("wkt_find_number_boundaries",)


# ---------------------------------------------------------------------------
# Kernel: count parts per geometry for MULTI* types.
#
# For MULTIPOINT (tag 3), MULTILINESTRING (tag 4):
#   parts are at depth 2 -- count depth transitions from 1->2
# For MULTIPOLYGON (tag 5):
#   parts are at depth 2 -- count depth transitions from 1->2
# For all others, emit 1 (single-part) or 0 (EMPTY).
# ---------------------------------------------------------------------------

_WKT_COUNT_PARTS_SOURCE = r"""
extern "C" __global__ void __launch_bounds__(256, 4)
wkt_count_parts(
    const int* __restrict__ depth,
    const long long* __restrict__ paren_starts,
    const long long* __restrict__ span_ends,
    const signed char* __restrict__ family_tags,
    const unsigned char* __restrict__ empty_flags,
    int* __restrict__ part_counts,
    const int n_geoms
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_geoms) return;

    signed char tag = family_tags[idx];

    if (empty_flags[idx] || paren_starts[idx] < 0) {
        part_counts[idx] = 0;
        return;
    }

    // Non-multi types have exactly 1 part
    if (tag < 3) {
        part_counts[idx] = 1;
        return;
    }

    // Multi types: count depth transitions from 1->2
    long long start = paren_starts[idx];
    long long end = span_ends[idx];
    int count = 0;

    for (long long i = start + 1; i < end; i++) {
        if (depth[i] == 2 && depth[i - 1] < 2) {
            count++;
        }
    }
    part_counts[idx] = count;
}
"""

_WKT_COUNT_PARTS_NAMES: tuple[str, ...] = ("wkt_count_parts",)


# ---------------------------------------------------------------------------
# Kernel: per-ring coordinate counting for polygon types.
#
# For each ring in a POLYGON or MULTIPOLYGON, count the coordinate
# values (number starts) within that ring's depth span.
# This enables building ring_offsets.
#
# The kernel is launched with one thread per ring (flattened across
# all geometries).  Ring boundaries are identified by scanning depth
# transitions within each geometry's span.
#
# Instead of a complex two-pass approach, we use a simpler strategy:
# assign number-start positions to rings via the depth array.
# For POLYGON: numbers at depth 2 belong to rings.
# For MULTIPOLYGON: numbers at depth 3 belong to rings.
#
# We count ring->coordinate mappings by scanning the coord region
# and tracking which ring each coordinate belongs to.
# ---------------------------------------------------------------------------

_WKT_ASSIGN_RING_COORDS_SOURCE = r"""
// For each geometry that has rings, scan its coordinate span and emit
// per-ring coordinate counts.  ring_offsets_out is pre-allocated to
// have one slot per ring per geometry.  ring_base[i] gives the index
// into ring_offsets_out where geometry i's rings start.
//
// For POLYGON (tag 2): ring boundaries are depth 1->2 transitions.
//   Coords at depth >= 2 belong to the current ring.
// For MULTIPOLYGON (tag 5): ring boundaries are depth 2->3 transitions.
//   Coords at depth >= 3 belong to the current ring.

extern "C" __global__ void __launch_bounds__(256, 4)
wkt_assign_ring_coords(
    const unsigned char* __restrict__ is_num_start,
    const int* __restrict__ depth,
    const long long* __restrict__ paren_starts,
    const long long* __restrict__ span_ends,
    const signed char* __restrict__ family_tags,
    const int* __restrict__ ring_base,
    int* __restrict__ ring_coord_counts,
    const int n_geoms
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_geoms) return;

    signed char tag = family_tags[idx];
    int ring_depth;
    if (tag == 2) {
        ring_depth = 2;
    } else if (tag == 5) {
        ring_depth = 3;
    } else {
        return;
    }

    long long start = paren_starts[idx];
    if (start < 0) return;
    long long end = span_ends[idx];

    int base = ring_base[idx];
    int current_ring = -1;

    for (long long i = start + 1; i < end; i++) {
        // Detect ring boundary: depth transitions up to ring_depth
        if (depth[i] == ring_depth && depth[i - 1] < ring_depth) {
            current_ring++;
        }
        // Count number starts within the current ring
        if (current_ring >= 0 && depth[i] >= ring_depth && is_num_start[i]) {
            ring_coord_counts[base + current_ring]++;
        }
    }
}
"""

_WKT_ASSIGN_RING_COORDS_NAMES: tuple[str, ...] = ("wkt_assign_ring_coords",)


# ---------------------------------------------------------------------------
# NVRTC warmup registration (ADR-0034 Level 2)
# ---------------------------------------------------------------------------
# Register all WKT kernels for background precompilation.
# bracket_depth warmup for WKT parentheses is handled lazily by
# structural.py's _get_depth_kernels on first use (cached by char pair).

from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("wkt-classify-type", _WKT_CLASSIFY_SOURCE, _WKT_CLASSIFY_NAMES),
    ("wkt-find-paren-starts", _WKT_FIND_PAREN_STARTS_SOURCE, _WKT_FIND_PAREN_STARTS_NAMES),
    ("wkt-count-coords", _WKT_COUNT_COORDS_SOURCE, _WKT_COUNT_COORDS_NAMES),
    ("wkt-count-rings", _WKT_COUNT_RINGS_SOURCE, _WKT_COUNT_RINGS_NAMES),
    ("wkt-num-bounds", _WKT_NUM_BOUNDS_SOURCE, _WKT_NUM_BOUNDS_NAMES),
    ("wkt-count-parts", _WKT_COUNT_PARTS_SOURCE, _WKT_COUNT_PARTS_NAMES),
    ("wkt-assign-ring-coords", _WKT_ASSIGN_RING_COORDS_SOURCE, _WKT_ASSIGN_RING_COORDS_NAMES),
])

# CCCL warmup for exclusive_sum used in offset building
from vibespatial.cuda.cccl_precompile import request_warmup  # noqa: E402

request_warmup(["exclusive_scan_i32"])


# ---------------------------------------------------------------------------
# Kernel compilation helpers
# ---------------------------------------------------------------------------

def _classify_type_kernels() -> dict:
    """Compile (or retrieve cached) WKT type classification kernel."""
    return compile_kernel_group(
        "wkt-classify-type", _WKT_CLASSIFY_SOURCE, _WKT_CLASSIFY_NAMES,
    )


def _find_paren_starts_kernels() -> dict:
    """Compile (or retrieve cached) WKT paren-start finder kernel."""
    return compile_kernel_group(
        "wkt-find-paren-starts", _WKT_FIND_PAREN_STARTS_SOURCE, _WKT_FIND_PAREN_STARTS_NAMES,
    )


def _count_coords_kernels() -> dict:
    """Compile (or retrieve cached) WKT coordinate counting kernel."""
    return compile_kernel_group(
        "wkt-count-coords", _WKT_COUNT_COORDS_SOURCE, _WKT_COUNT_COORDS_NAMES,
    )


def _count_rings_kernels() -> dict:
    """Compile (or retrieve cached) WKT ring counting kernel."""
    return compile_kernel_group(
        "wkt-count-rings", _WKT_COUNT_RINGS_SOURCE, _WKT_COUNT_RINGS_NAMES,
    )


def _wkt_num_bounds_kernels() -> dict:
    """Compile (or retrieve cached) WKT number boundary kernel."""
    return compile_kernel_group(
        "wkt-num-bounds", _WKT_NUM_BOUNDS_SOURCE, _WKT_NUM_BOUNDS_NAMES,
    )


def _count_parts_kernels() -> dict:
    """Compile (or retrieve cached) WKT part counting kernel."""
    return compile_kernel_group(
        "wkt-count-parts", _WKT_COUNT_PARTS_SOURCE, _WKT_COUNT_PARTS_NAMES,
    )


def _assign_ring_coords_kernels() -> dict:
    """Compile (or retrieve cached) WKT ring-coordinate assignment kernel."""
    return compile_kernel_group(
        "wkt-assign-ring-coords", _WKT_ASSIGN_RING_COORDS_SOURCE, _WKT_ASSIGN_RING_COORDS_NAMES,
    )


def _launch_kernel(runtime, kernel, n, params):
    """Launch a kernel with occupancy-based grid/block sizing."""
    grid, block = runtime.launch_config(kernel, int(n))
    runtime.launch(kernel, grid=grid, block=block, params=params)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class WktStructuralResult:
    """Result of WKT structural analysis.

    All arrays are device-resident CuPy arrays except ``n_geometries``
    which is a Python int.

    Attributes
    ----------
    d_depth : cp.ndarray
        Per-byte parenthesis depth, int32, shape ``(n_bytes,)``.
    d_geom_starts : cp.ndarray
        Start byte offset of each geometry, int64, shape ``(n_geometries,)``.
    d_family_tags : cp.ndarray
        Geometry family tag per geometry, int8, shape ``(n_geometries,)``.
        Values: 0=POINT, 1=LINESTRING, 2=POLYGON, 3=MULTIPOINT,
        4=MULTILINESTRING, 5=MULTIPOLYGON, -2=unknown/unsupported.
    d_empty_flags : cp.ndarray
        Per-geometry EMPTY flag, uint8, shape ``(n_geometries,)``.
        1 if the geometry uses the EMPTY keyword, 0 otherwise.
    n_geometries : int
        Number of geometries detected.
    """

    d_depth: cp.ndarray
    d_geom_starts: cp.ndarray
    d_family_tags: cp.ndarray
    d_empty_flags: cp.ndarray
    n_geometries: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def wkt_structural_analysis(d_bytes: cp.ndarray) -> WktStructuralResult:
    """Perform structural analysis and geometry type detection on WKT input.

    Given a device-resident byte array containing one or more WKT
    geometry strings separated by newlines, this function:

    1. Detects line boundaries (newline positions) to delimit geometries.
    2. Computes per-byte parenthesis depth using ``bracket_depth``.
    3. Classifies each geometry by type keyword and detects EMPTY.

    The input may contain:

    - Standard WKT: ``POINT(1 2)``
    - EWKT with SRID prefix: ``SRID=4326;POINT(1 2)``
    - Mixed case: ``Point(1 2)``, ``LINESTRING(...)``
    - 3D/M suffixes: ``POINT Z(1 2 3)``, ``POINTZ(1 2 3)``
    - Empty geometries: ``POINT EMPTY``

    Parameters
    ----------
    d_bytes : cp.ndarray
        Device-resident uint8 array of WKT text bytes, shape ``(n,)``.
        Multiple geometries are separated by newline characters
        (``\\n``, 0x0A).  Trailing newlines are handled gracefully.

    Returns
    -------
    WktStructuralResult
        Dataclass containing all structural analysis outputs on device.

    Notes
    -----
    WKT has no string quoting, so ``bracket_depth`` receives an
    all-zeros quote-parity array.  This causes the depth kernel to
    treat every parenthesis as structural.

    The parenthesis depth array uses the same convention as the
    GeoJSON bracket depth:

    - Depth 0: outside all geometry parentheses
    - Depth 1: inside the outermost ``(...)``
    - Depth 2+: nested rings, coordinate groups, etc.

    Examples
    --------
    >>> import cupy as cp
    >>> wkt = b"POINT(1 2)\\nLINESTRING(0 0, 1 1)\\nPOLYGON EMPTY"
    >>> d_bytes = cp.frombuffer(wkt, dtype=cp.uint8)
    >>> result = wkt_structural_analysis(d_bytes)
    >>> result.n_geometries
    3
    >>> result.d_family_tags.get()  # array([0, 1, 2], dtype=int8)
    >>> result.d_empty_flags.get()  # array([0, 0, 1], dtype=uint8)
    """
    import cupy as cp

    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    n = d_bytes.shape[0]
    n_i64 = np.int64(n)

    # ------------------------------------------------------------------
    # Stage 1: Parenthesis depth
    # ------------------------------------------------------------------
    # WKT has no string quoting, so we pass an all-zeros quote parity.
    # This makes bracket_depth treat every '(' and ')' as structural.
    d_quote_parity = cp.zeros(n, dtype=cp.uint8)
    d_depth = bracket_depth(
        d_bytes, d_quote_parity, open_chars="(", close_chars=")",
    )
    del d_quote_parity  # free immediately

    # ------------------------------------------------------------------
    # Stage 2: Detect line boundaries (geometry starts)
    # ------------------------------------------------------------------
    # Each geometry is one line.  Find newline positions, then derive
    # geometry start offsets.  Geometry 0 always starts at byte 0;
    # subsequent geometries start at newline_pos + 1.
    #
    # Tier 2 (CuPy): element-wise comparison + flatnonzero.
    d_is_newline = (d_bytes == ord('\n'))
    d_newline_positions = cp.flatnonzero(d_is_newline).astype(cp.int64)
    del d_is_newline

    n_newlines = d_newline_positions.shape[0]

    # Build geometry start positions: [0, nl[0]+1, nl[1]+1, ...]
    # But only include starts that point to non-empty lines.
    if n_newlines == 0:
        # Single geometry (no newlines) or empty input
        if n == 0:
            return WktStructuralResult(
                d_depth=d_depth,
                d_geom_starts=cp.empty(0, dtype=cp.int64),
                d_family_tags=cp.empty(0, dtype=cp.int8),
                d_empty_flags=cp.empty(0, dtype=cp.uint8),
                n_geometries=0,
            )
        d_geom_starts = cp.zeros(1, dtype=cp.int64)
    else:
        # Starts = [0] + [nl_pos + 1 for each newline that is not the last byte]
        d_after_newlines = d_newline_positions + 1
        # Include byte 0 as the first geometry start
        d_zero = cp.zeros(1, dtype=cp.int64)
        # Filter out starts that are >= n (trailing newline at end of file)
        d_candidate_starts = cp.concatenate([d_zero, d_after_newlines])
        d_valid_mask = d_candidate_starts < n
        d_geom_starts = d_candidate_starts[d_valid_mask]
        del d_after_newlines, d_zero, d_candidate_starts, d_valid_mask

    del d_newline_positions

    # Filter out empty lines (blank or whitespace-only).  A start whose
    # byte is a newline or carriage return indicates a blank line.  We
    # filter these on device to avoid the classify kernel skipping
    # whitespace across line boundaries into the next geometry.
    if d_geom_starts.shape[0] > 0:
        d_start_bytes = d_bytes[d_geom_starts]
        d_non_empty = (d_start_bytes != ord('\n')) & (d_start_bytes != ord('\r'))
        d_geom_starts = d_geom_starts[d_non_empty]
        del d_start_bytes, d_non_empty

    n_geoms = d_geom_starts.shape[0]

    if n_geoms == 0:
        return WktStructuralResult(
            d_depth=d_depth,
            d_geom_starts=d_geom_starts,
            d_family_tags=cp.empty(0, dtype=cp.int8),
            d_empty_flags=cp.empty(0, dtype=cp.uint8),
            n_geometries=0,
        )

    # ------------------------------------------------------------------
    # Stage 3: Classify geometry types
    # ------------------------------------------------------------------
    kernels = _classify_type_kernels()
    d_family_tags = cp.empty(n_geoms, dtype=cp.int8)
    d_empty_flags = cp.empty(n_geoms, dtype=cp.uint8)

    _launch_kernel(
        runtime,
        kernels["wkt_classify_geometry_type"],
        n_geoms,
        (
            (
                ptr(d_bytes),
                ptr(d_geom_starts),
                ptr(d_family_tags),
                ptr(d_empty_flags),
                np.int32(n_geoms),
                n_i64,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I64,
            ),
        ),
    )

    # No sync needed before returning -- all outputs are device arrays
    # and the caller will sync when materializing to host.

    return WktStructuralResult(
        d_depth=d_depth,
        d_geom_starts=d_geom_starts,
        d_family_tags=d_family_tags,
        d_empty_flags=d_empty_flags,
        n_geometries=int(n_geoms),
    )


# ---------------------------------------------------------------------------
# Coordinate extraction pipeline
# ---------------------------------------------------------------------------

def _find_paren_starts(
    d_bytes: cp.ndarray,
    d_geom_starts: cp.ndarray,
    d_empty_flags: cp.ndarray,
    n_geoms: int,
    n_bytes: int,
) -> cp.ndarray:
    """Find the byte offset of the first '(' for each geometry.

    Returns int64 array of shape (n_geoms,).  EMPTY geometries get -1.
    """
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    kernels = _find_paren_starts_kernels()

    d_paren_starts = cp.empty(n_geoms, dtype=cp.int64)
    _launch_kernel(
        runtime,
        kernels["wkt_find_paren_starts"],
        n_geoms,
        (
            (
                ptr(d_bytes),
                ptr(d_geom_starts),
                ptr(d_empty_flags),
                ptr(d_paren_starts),
                np.int32(n_geoms),
                np.int64(n_bytes),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I64,
            ),
        ),
    )
    return d_paren_starts


def _wkt_number_boundaries(
    d_bytes: cp.ndarray,
    d_coord_mask: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray]:
    """WKT-aware number boundary detection.

    Like gpu_parse.number_boundaries but with '(' and ')' in the
    separator set, and integrated coordinate-region masking.

    Returns (d_is_start, d_is_end) uint8 arrays of shape (n_bytes,).
    """
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    n = d_bytes.shape[0]
    kernels = _wkt_num_bounds_kernels()

    d_is_start = cp.zeros(n, dtype=cp.uint8)
    d_is_end = cp.zeros(n, dtype=cp.uint8)

    _launch_kernel(
        runtime,
        kernels["wkt_find_number_boundaries"],
        n,
        (
            (
                ptr(d_bytes),
                ptr(d_coord_mask),
                ptr(d_is_start),
                ptr(d_is_end),
                np.int64(n),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I64,
            ),
        ),
    )
    return d_is_start, d_is_end


def _count_coords_per_geometry(
    d_is_num_start: cp.ndarray,
    d_paren_starts: cp.ndarray,
    d_span_ends: cp.ndarray,
    n_geoms: int,
) -> cp.ndarray:
    """Count coordinate values (individual floats) per geometry.

    Returns int32 array of shape (n_geoms,).
    """
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    kernels = _count_coords_kernels()

    d_counts = cp.zeros(n_geoms, dtype=cp.int32)
    _launch_kernel(
        runtime,
        kernels["wkt_count_coords_per_geometry"],
        n_geoms,
        (
            (
                ptr(d_is_num_start),
                ptr(d_paren_starts),
                ptr(d_span_ends),
                ptr(d_counts),
                np.int32(n_geoms),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        ),
    )
    return d_counts


def _count_rings(
    d_depth: cp.ndarray,
    d_paren_starts: cp.ndarray,
    d_span_ends: cp.ndarray,
    d_family_tags: cp.ndarray,
    n_geoms: int,
) -> cp.ndarray:
    """Count rings per geometry for polygon types.

    Returns int32 array of shape (n_geoms,).  Non-polygon types get 0.
    """
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    kernels = _count_rings_kernels()

    d_ring_counts = cp.zeros(n_geoms, dtype=cp.int32)
    _launch_kernel(
        runtime,
        kernels["wkt_count_rings"],
        n_geoms,
        (
            (
                ptr(d_depth),
                ptr(d_paren_starts),
                ptr(d_span_ends),
                ptr(d_family_tags),
                ptr(d_ring_counts),
                np.int32(n_geoms),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        ),
    )
    return d_ring_counts


def _count_parts(
    d_depth: cp.ndarray,
    d_paren_starts: cp.ndarray,
    d_span_ends: cp.ndarray,
    d_family_tags: cp.ndarray,
    d_empty_flags: cp.ndarray,
    n_geoms: int,
) -> cp.ndarray:
    """Count parts per geometry for multi types.

    Returns int32 array of shape (n_geoms,).
    Non-multi types get 1 (or 0 if EMPTY).
    """
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    kernels = _count_parts_kernels()

    d_part_counts = cp.zeros(n_geoms, dtype=cp.int32)
    _launch_kernel(
        runtime,
        kernels["wkt_count_parts"],
        n_geoms,
        (
            (
                ptr(d_depth),
                ptr(d_paren_starts),
                ptr(d_span_ends),
                ptr(d_family_tags),
                ptr(d_empty_flags),
                ptr(d_part_counts),
                np.int32(n_geoms),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        ),
    )
    return d_part_counts


def _assign_ring_coords(
    d_is_num_start: cp.ndarray,
    d_depth: cp.ndarray,
    d_paren_starts: cp.ndarray,
    d_span_ends: cp.ndarray,
    d_family_tags: cp.ndarray,
    d_ring_base: cp.ndarray,
    total_rings: int,
    n_geoms: int,
) -> cp.ndarray:
    """Count coordinate values per ring for polygon types.

    d_ring_base[i] is the index into the output array where geometry
    i's ring counts start (exclusive prefix sum of ring_counts).

    Returns int32 array of shape (total_rings,) with per-ring value counts.
    """
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    kernels = _assign_ring_coords_kernels()

    d_ring_coord_counts = cp.zeros(total_rings, dtype=cp.int32)
    if n_geoms > 0:
        _launch_kernel(
            runtime,
            kernels["wkt_assign_ring_coords"],
            n_geoms,
            (
                (
                    ptr(d_is_num_start),
                    ptr(d_depth),
                    ptr(d_paren_starts),
                    ptr(d_span_ends),
                    ptr(d_family_tags),
                    ptr(d_ring_base),
                    ptr(d_ring_coord_counts),
                    np.int32(n_geoms),
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
            ),
        )
    return d_ring_coord_counts


def _extract_wkt_coordinates(
    d_bytes: cp.ndarray,
    structural: WktStructuralResult,
) -> tuple[
    cp.ndarray,   # d_x
    cp.ndarray,   # d_y
    cp.ndarray,   # d_coord_value_counts (per geometry)
    cp.ndarray,   # d_ring_counts (per geometry)
    cp.ndarray,   # d_part_counts (per geometry)
    cp.ndarray,   # d_paren_starts
    cp.ndarray,   # d_span_ends
    cp.ndarray,   # d_is_num_start (kept for ring assignment)
]:
    """Extract coordinates from WKT bytes using structural analysis results.

    Returns device-resident x, y coordinate arrays (fp64) and per-geometry
    metadata arrays needed for offset construction.

    All computation stays on GPU -- no host round-trips.
    """
    import cupy as cp

    from vibespatial.io.gpu_parse import parse_ascii_floats

    n_bytes = d_bytes.shape[0]
    n_geoms = structural.n_geometries

    # ------------------------------------------------------------------
    # Stage 1: Find coordinate region boundaries
    # ------------------------------------------------------------------
    # For each geometry, find the opening '(' of its coordinate data.
    d_paren_starts = _find_paren_starts(
        d_bytes, structural.d_geom_starts, structural.d_empty_flags,
        n_geoms, n_bytes,
    )

    # Find span ends using depth-based scanning from gpu_parse.
    # span_boundaries does not handle -1 (EMPTY) positions safely, so
    # we filter to only valid (non-negative) paren starts, run the
    # kernel on those, and scatter results back.
    from vibespatial.io.gpu_parse import mark_spans, span_boundaries

    d_valid_mask = (d_paren_starts >= 0)
    d_valid_indices = cp.flatnonzero(d_valid_mask).astype(cp.int32)
    n_valid = d_valid_indices.shape[0]

    d_span_ends = cp.zeros(n_geoms, dtype=cp.int64)

    if n_valid > 0:
        d_valid_paren_starts = d_paren_starts[d_valid_indices]
        d_valid_span_ends = span_boundaries(
            structural.d_depth, d_valid_paren_starts, n_bytes, skip_bytes=0,
        )
        d_span_ends[d_valid_indices] = d_valid_span_ends
        del d_valid_span_ends

        # Create coordinate region mask using only valid spans
        d_coord_mask = mark_spans(d_valid_paren_starts, d_span_ends[d_valid_indices], n_bytes)
        del d_valid_paren_starts
    else:
        d_coord_mask = cp.zeros(n_bytes, dtype=cp.uint8)

    del d_valid_mask, d_valid_indices

    # ------------------------------------------------------------------
    # Stage 2: Detect number boundaries (WKT-aware)
    # ------------------------------------------------------------------
    d_is_num_start, d_is_num_end = _wkt_number_boundaries(d_bytes, d_coord_mask)
    del d_coord_mask  # free mask after use

    # Convert boundary masks to compact position arrays
    d_num_starts = cp.flatnonzero(d_is_num_start).astype(cp.int64)
    d_num_ends = cp.flatnonzero(d_is_num_end).astype(cp.int64) + 1  # half-open
    del d_is_num_end  # no longer needed; saves 1*n bytes through counting stages

    # ------------------------------------------------------------------
    # Stage 3: Parse numeric values
    # ------------------------------------------------------------------
    n_values = d_num_starts.shape[0]
    if n_values == 0:
        d_x = cp.empty(0, dtype=cp.float64)
        d_y = cp.empty(0, dtype=cp.float64)
        d_coord_value_counts = cp.zeros(n_geoms, dtype=cp.int32)
        d_ring_counts = cp.zeros(n_geoms, dtype=cp.int32)
        d_part_counts = cp.zeros(n_geoms, dtype=cp.int32)
        return (
            d_x, d_y, d_coord_value_counts, d_ring_counts, d_part_counts,
            d_paren_starts, d_span_ends, d_is_num_start,
        )

    d_values = parse_ascii_floats(d_bytes, d_num_starts, d_num_ends)
    del d_num_starts, d_num_ends

    # Split into x, y via zero-copy strided views (2D assumption)
    # d_values is [x0, y0, x1, y1, ...] -- interleaved
    d_x = d_values[0::2]
    d_y = d_values[1::2]

    # ------------------------------------------------------------------
    # Stage 4: Count coordinates, rings, and parts per geometry
    # ------------------------------------------------------------------
    d_coord_value_counts = _count_coords_per_geometry(
        d_is_num_start, d_paren_starts, d_span_ends, n_geoms,
    )

    d_ring_counts = _count_rings(
        structural.d_depth, d_paren_starts, d_span_ends,
        structural.d_family_tags, n_geoms,
    )

    d_part_counts = _count_parts(
        structural.d_depth, d_paren_starts, d_span_ends,
        structural.d_family_tags, structural.d_empty_flags, n_geoms,
    )

    return (
        d_x, d_y, d_coord_value_counts, d_ring_counts, d_part_counts,
        d_paren_starts, d_span_ends, d_is_num_start,
    )


# ---------------------------------------------------------------------------
# Assembly into OwnedGeometryArray
# ---------------------------------------------------------------------------

# WKT family tags align with GeometryFamily enum order:
#   POINT=0, LINESTRING=1, POLYGON=2,
#   MULTIPOINT=3, MULTILINESTRING=4, MULTIPOLYGON=5
_WKT_TAG_TO_FAMILY = {
    0: GeometryFamily.POINT,
    1: GeometryFamily.LINESTRING,
    2: GeometryFamily.POLYGON,
    3: GeometryFamily.MULTIPOINT,
    4: GeometryFamily.MULTILINESTRING,
    5: GeometryFamily.MULTIPOLYGON,
}


def _device_compact_offsets(d_counts: cp.ndarray) -> cp.ndarray:
    """Build (n+1) offset array from per-element counts via exclusive sum.

    Returns int32 array of shape (n+1,) where offsets[0]=0 and
    offsets[i+1] = offsets[i] + counts[i].
    """
    n = d_counts.shape[0]
    d_offsets = cp.empty(n + 1, dtype=cp.int32)
    d_offsets[0] = 0
    if n > 0:
        d_excl = exclusive_sum(d_counts, synchronize=False)
        d_offsets[1:] = d_excl.astype(cp.int32) + d_counts.astype(cp.int32)
    return d_offsets


def _build_point_offsets(n_geoms: int, d_coord_value_counts: cp.ndarray) -> cp.ndarray:
    """Build geometry_offsets for Point: each point is 1 coordinate pair."""
    # For Point, geometry_offsets is simply [0, 1, 2, ..., n_geoms]
    return cp.arange(n_geoms + 1, dtype=cp.int32)


def _build_linestring_offsets(d_coord_value_counts: cp.ndarray) -> cp.ndarray:
    """Build geometry_offsets for LineString from per-geometry value counts.

    Each geometry has coord_value_counts[i] / 2 coordinate pairs.
    Returns int32 array of shape (n_geoms + 1,).
    """
    d_pair_counts = d_coord_value_counts // 2
    return _device_compact_offsets(d_pair_counts)


def _build_polygon_offsets(
    d_ring_counts: cp.ndarray,
    d_ring_coord_counts: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Build geometry_offsets and ring_offsets for Polygon.

    geometry_offsets: (n_geoms+1,) cumulative sum of ring_counts
    ring_offsets: (total_rings+1,) cumulative sum of pairs per ring
    """
    d_geom_offsets = _device_compact_offsets(d_ring_counts)
    d_ring_pair_counts = d_ring_coord_counts // 2
    d_ring_offsets = _device_compact_offsets(d_ring_pair_counts)
    return d_geom_offsets, d_ring_offsets


def _assemble_wkt_homogeneous(
    family: GeometryFamily,
    n_geoms: int,
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    d_coord_value_counts: cp.ndarray,
    d_ring_counts: cp.ndarray,
    d_part_counts: cp.ndarray,
    d_is_num_start: cp.ndarray,
    d_depth: cp.ndarray,
    d_paren_starts: cp.ndarray,
    d_span_ends: cp.ndarray,
    d_family_tags: cp.ndarray,
    d_empty_flags: cp.ndarray,
) -> OwnedGeometryArray:
    """Build OwnedGeometryArray for a homogeneous WKT file (single family)."""
    from vibespatial.io.pylibcudf import _build_device_single_family_owned

    d_empty_mask = (d_coord_value_counts == 0)
    d_validity = ~d_empty_mask

    if family == GeometryFamily.POINT:
        d_geom_offsets = _build_point_offsets(n_geoms, d_coord_value_counts)
        return _build_device_single_family_owned(
            family=GeometryFamily.POINT,
            validity_device=d_validity,
            x_device=cp.ascontiguousarray(d_x),
            y_device=cp.ascontiguousarray(d_y),
            geometry_offsets_device=d_geom_offsets,
            empty_mask_device=d_empty_mask,
            detail="GPU WKT parse (Point)",
        )

    if family == GeometryFamily.LINESTRING:
        d_geom_offsets = _build_linestring_offsets(d_coord_value_counts)
        return _build_device_single_family_owned(
            family=GeometryFamily.LINESTRING,
            validity_device=d_validity,
            x_device=cp.ascontiguousarray(d_x),
            y_device=cp.ascontiguousarray(d_y),
            geometry_offsets_device=d_geom_offsets,
            empty_mask_device=d_empty_mask,
            detail="GPU WKT parse (LineString)",
        )

    if family == GeometryFamily.POLYGON:
        # Build ring-level coordinate counts.
        # ring_base[i] = exclusive prefix sum of ring_counts -- gives the
        # index into the flat ring array where geometry i's rings start.
        d_ring_base_full = _device_compact_offsets(d_ring_counts)
        total_rings = int(d_ring_base_full[-1].get()) if n_geoms > 0 else 0
        # Pass the n-element prefix (without the terminal) as ring_base
        d_ring_base = d_ring_base_full[:-1]
        d_ring_coord_counts = _assign_ring_coords(
            d_is_num_start, d_depth, d_paren_starts, d_span_ends,
            d_family_tags, d_ring_base, total_rings, n_geoms,
        )
        d_geom_offsets, d_ring_offsets = _build_polygon_offsets(
            d_ring_counts, d_ring_coord_counts,
        )
        return _build_device_single_family_owned(
            family=GeometryFamily.POLYGON,
            validity_device=d_validity,
            x_device=cp.ascontiguousarray(d_x),
            y_device=cp.ascontiguousarray(d_y),
            geometry_offsets_device=d_geom_offsets,
            empty_mask_device=d_empty_mask,
            ring_offsets_device=d_ring_offsets,
            detail="GPU WKT parse (Polygon)",
        )

    # Multi* types -- build part offsets + appropriate sub-offsets
    if family in (
        GeometryFamily.MULTIPOINT,
        GeometryFamily.MULTILINESTRING,
        GeometryFamily.MULTIPOLYGON,
    ):
        d_geom_offsets = _device_compact_offsets(d_part_counts)

        if family == GeometryFamily.MULTIPOLYGON:
            # Need ring offsets too
            d_ring_base_full = _device_compact_offsets(d_ring_counts)
            total_rings = int(d_ring_base_full[-1].get()) if n_geoms > 0 else 0
            d_ring_base = d_ring_base_full[:-1]
            d_ring_coord_counts = _assign_ring_coords(
                d_is_num_start, d_depth, d_paren_starts, d_span_ends,
                d_family_tags, d_ring_base, total_rings, n_geoms,
            )
            d_ring_offsets = _device_compact_offsets(d_ring_coord_counts // 2)

            # part_offsets = ring_base_full (maps parts to rings)
            return _build_device_single_family_owned(
                family=GeometryFamily.MULTIPOLYGON,
                validity_device=d_validity,
                x_device=cp.ascontiguousarray(d_x),
                y_device=cp.ascontiguousarray(d_y),
                geometry_offsets_device=d_geom_offsets,
                empty_mask_device=d_empty_mask,
                part_offsets_device=d_ring_base_full,
                ring_offsets_device=d_ring_offsets,
                detail="GPU WKT parse (MultiPolygon)",
            )

        # MULTIPOINT / MULTILINESTRING: geometry_offsets = part boundaries,
        # part_offsets = coordinate pair offsets within each part.
        d_pair_counts = d_coord_value_counts // 2
        d_coord_offsets = _device_compact_offsets(d_pair_counts)
        return _build_device_single_family_owned(
            family=family,
            validity_device=d_validity,
            x_device=cp.ascontiguousarray(d_x),
            y_device=cp.ascontiguousarray(d_y),
            geometry_offsets_device=d_geom_offsets,
            empty_mask_device=d_empty_mask,
            part_offsets_device=d_coord_offsets,
            detail=f"GPU WKT parse ({family.value})",
        )

    msg = f"Unsupported geometry family for WKT assembly: {family}"
    raise ValueError(msg)


def _assemble_wkt_mixed(
    n_geoms: int,
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    d_coord_value_counts: cp.ndarray,
    d_ring_counts: cp.ndarray,
    d_part_counts: cp.ndarray,
    d_family_tags: cp.ndarray,
    d_empty_flags: cp.ndarray,
    d_is_num_start: cp.ndarray,
    d_depth: cp.ndarray,
    d_paren_starts: cp.ndarray,
    d_span_ends: cp.ndarray,
) -> OwnedGeometryArray:
    """Build OwnedGeometryArray for a mixed-type WKT file.

    Partitions geometries by family and builds per-family buffers,
    then assembles them into a single mixed OwnedGeometryArray.
    """
    from vibespatial.geometry.owned import _device_gather_offset_slices
    from vibespatial.io.pylibcudf import _build_device_mixed_owned

    # Map WKT tags to GeometryFamily tags used by OwnedGeometryArray
    # WKT tag matches GeometryFamily enum order, which matches FAMILY_TAGS
    d_oga_tags = d_family_tags.copy().astype(cp.int8)
    # Mark invalid/unknown tags (-2) as -1 (invalid in OGA)
    d_oga_tags[d_oga_tags == -2] = -1
    # Mark EMPTY geometries as -1
    d_oga_tags[d_empty_flags.astype(cp.bool_)] = -1

    d_validity = (d_oga_tags >= 0)

    # Build per-geometry pair counts and coordinate offsets (n+1 array)
    d_pair_counts = d_coord_value_counts // 2
    d_coord_offsets = _device_compact_offsets(d_pair_counts)

    # Build per-family device buffers
    family_devices: dict[GeometryFamily, DeviceFamilyGeometryBuffer] = {}
    family_rows: dict[int, cp.ndarray] = {}

    # Only process families that are present
    unique_tags = cp.unique(d_family_tags[d_family_tags >= 0])
    # Sync needed to read unique_tags on host for the loop
    h_unique_tags = unique_tags.get()

    # Pre-compute 2D coordinate array for gather operations
    coords_2d = (
        cp.column_stack([d_x, d_y])
        if d_x.size > 0
        else cp.empty((0, 2), dtype=cp.float64)
    )

    for tag_val in h_unique_tags:
        tag_int = int(tag_val)
        family = _WKT_TAG_TO_FAMILY.get(tag_int)
        if family is None:
            continue

        rows = cp.flatnonzero(d_family_tags == tag_int).astype(cp.int32)
        if rows.size == 0:
            continue
        family_rows[tag_int] = rows
        n_f = int(rows.size)

        if family == GeometryFamily.POINT:
            pt_starts = d_coord_offsets[rows]
            pt_x = d_x[pt_starts]
            pt_y = d_y[pt_starts]
            family_devices[family] = DeviceFamilyGeometryBuffer(
                family=family,
                x=cp.ascontiguousarray(pt_x),
                y=cp.ascontiguousarray(pt_y),
                geometry_offsets=cp.arange(n_f + 1, dtype=cp.int32),
                empty_mask=cp.zeros(n_f, dtype=cp.bool_),
            )

        elif family == GeometryFamily.LINESTRING:
            gathered, ls_geom_offsets = _device_gather_offset_slices(
                coords_2d, d_coord_offsets, rows,
            )
            family_devices[family] = DeviceFamilyGeometryBuffer(
                family=family,
                x=(
                    cp.ascontiguousarray(gathered[:, 0])
                    if gathered.size else cp.empty(0, dtype=cp.float64)
                ),
                y=(
                    cp.ascontiguousarray(gathered[:, 1])
                    if gathered.size else cp.empty(0, dtype=cp.float64)
                ),
                geometry_offsets=ls_geom_offsets,
                empty_mask=(ls_geom_offsets[1:] == ls_geom_offsets[:-1]),
            )

        elif family == GeometryFamily.POLYGON:
            # Gather polygon coordinates using the global coord offsets.
            # Each polygon geometry's coordinates live at
            # d_coord_offsets[row] to d_coord_offsets[row+1] in the
            # global flat array.
            gathered, sub_coord_offsets = _device_gather_offset_slices(
                coords_2d, d_coord_offsets, rows,
            )
            pg_x = (
                cp.ascontiguousarray(gathered[:, 0])
                if gathered.size else cp.empty(0, dtype=cp.float64)
            )
            pg_y = (
                cp.ascontiguousarray(gathered[:, 1])
                if gathered.size else cp.empty(0, dtype=cp.float64)
            )

            # Build ring offsets for the polygon subset.
            # Use per-geometry ring counts to build geometry_offsets.
            sub_ring_counts = d_ring_counts[rows]
            sub_geom_offsets = _device_compact_offsets(sub_ring_counts)

            # Build ring-level coordinate counts for the polygon subset.
            # We run assign_ring_coords only on polygon geometries by
            # using a subset view.  Build a ring_base for the subset.
            sub_total_rings = int(sub_geom_offsets[-1]) if n_f > 0 else 0
            sub_ring_base = sub_geom_offsets[:-1]
            # Need to call the kernel on the subset.  Create a temporary
            # family_tags array that only has polygon entries.
            d_sub_family_tags = d_family_tags[rows]
            d_sub_paren_starts = d_paren_starts[rows]
            d_sub_span_ends = d_span_ends[rows]
            sub_ring_coord_counts = _assign_ring_coords(
                d_is_num_start, d_depth, d_sub_paren_starts,
                d_sub_span_ends, d_sub_family_tags,
                sub_ring_base, sub_total_rings, n_f,
            )
            sub_ring_offsets = _device_compact_offsets(
                sub_ring_coord_counts // 2,
            )

            family_devices[family] = DeviceFamilyGeometryBuffer(
                family=family,
                x=pg_x,
                y=pg_y,
                geometry_offsets=sub_geom_offsets,
                empty_mask=(sub_geom_offsets[1:] == sub_geom_offsets[:-1]),
                ring_offsets=sub_ring_offsets,
            )

        # Multi* types in mixed files: treat as simplified single-part
        # for initial implementation (stretch goal per bead spec)
        elif family in (
            GeometryFamily.MULTIPOINT,
            GeometryFamily.MULTILINESTRING,
        ):
            gathered, geom_offsets = _device_gather_offset_slices(
                coords_2d, d_coord_offsets, rows,
            )
            family_devices[family] = DeviceFamilyGeometryBuffer(
                family=family,
                x=(
                    cp.ascontiguousarray(gathered[:, 0])
                    if gathered.size else cp.empty(0, dtype=cp.float64)
                ),
                y=(
                    cp.ascontiguousarray(gathered[:, 1])
                    if gathered.size else cp.empty(0, dtype=cp.float64)
                ),
                geometry_offsets=geom_offsets,
                empty_mask=(geom_offsets[1:] == geom_offsets[:-1]),
            )

    # Build tags and family_row_offsets
    d_family_row_offsets = cp.full(n_geoms, -1, dtype=cp.int32)
    for tag_val, rows in family_rows.items():
        d_family_row_offsets[rows] = cp.arange(int(rows.size), dtype=cp.int32)

    return _build_device_mixed_owned(
        validity_device=d_validity,
        tags_device=d_oga_tags,
        family_row_offsets_device=d_family_row_offsets,
        family_devices=family_devices,
        detail="GPU WKT parse (mixed)",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_wkt_gpu(d_bytes: cp.ndarray) -> OwnedGeometryArray:
    """Parse WKT bytes on GPU and return device-resident geometry.

    Given a device-resident byte array containing one or more WKT
    geometry strings separated by newlines, this function performs
    full GPU-accelerated parsing: structural analysis, coordinate
    extraction, and assembly into an ``OwnedGeometryArray``.

    Supported geometry types:

    - ``POINT``, ``LINESTRING``, ``POLYGON`` (full support)
    - ``MULTIPOINT``, ``MULTILINESTRING``, ``MULTIPOLYGON`` (stretch)
    - ``EMPTY`` variants of all types

    Parameters
    ----------
    d_bytes : cp.ndarray
        Device-resident uint8 array of WKT text bytes, shape ``(n,)``.
        Multiple geometries are separated by newline characters
        (``\\n``, 0x0A).

    Returns
    -------
    OwnedGeometryArray
        Device-resident geometry array.  Coordinates are always fp64.
        Structural metadata (offsets, validity) is materialized on both
        host and device per the standard ``_build_device_*_owned``
        pattern.

    Raises
    ------
    ValueError
        If the input contains only unsupported geometry types (e.g.,
        GEOMETRYCOLLECTION) or cannot be parsed.

    Notes
    -----
    Precision (ADR-0002):
        All coordinates are parsed and stored as fp64.  The structural
        analysis and counting kernels are integer-only byte
        classification -- no PrecisionPlan is needed for those stages.

    Tier classification (ADR-0033):
        Uses Tier 1 (custom NVRTC) for geometry-specific scanning and
        Tier 2 (CuPy) for element-wise operations.  Number parsing
        delegates to the gpu_parse primitives.

    Examples
    --------
    >>> import cupy as cp
    >>> wkt = b"POINT(1 2)\\nLINESTRING(0 0, 1 1, 2 0)"
    >>> d_bytes = cp.frombuffer(wkt, dtype=cp.uint8)
    >>> owned = read_wkt_gpu(d_bytes)
    >>> owned.row_count
    2
    """
    import cupy as cp

    # ------------------------------------------------------------------
    # Stage 1: Structural analysis
    # ------------------------------------------------------------------
    structural = wkt_structural_analysis(d_bytes)
    n_geoms = structural.n_geometries

    if n_geoms == 0:
        # Return an empty OwnedGeometryArray
        return _build_empty_owned()

    # ------------------------------------------------------------------
    # Stage 2: Coordinate extraction
    # ------------------------------------------------------------------
    (
        d_x, d_y, d_coord_value_counts, d_ring_counts, d_part_counts,
        d_paren_starts, d_span_ends, d_is_num_start,
    ) = _extract_wkt_coordinates(d_bytes, structural)

    # ------------------------------------------------------------------
    # Stage 3: Determine if homogeneous or mixed
    # ------------------------------------------------------------------
    # Filter to non-empty, valid (tag >= 0) geometries
    d_valid_tags = structural.d_family_tags[structural.d_family_tags >= 0]
    if d_valid_tags.size == 0:
        return _build_empty_owned()

    d_unique_tags = cp.unique(d_valid_tags)
    n_unique = d_unique_tags.shape[0]
    # Need to read n_unique to decide homogeneous vs mixed.  Single sync.
    h_unique_tags = d_unique_tags.get()

    if n_unique == 1:
        # Homogeneous file
        family = _WKT_TAG_TO_FAMILY.get(int(h_unique_tags[0]))
        if family is None:
            msg = f"Unsupported WKT geometry tag: {h_unique_tags[0]}"
            raise ValueError(msg)
        return _assemble_wkt_homogeneous(
            family, n_geoms, d_x, d_y,
            d_coord_value_counts, d_ring_counts, d_part_counts,
            d_is_num_start, structural.d_depth,
            d_paren_starts, d_span_ends,
            structural.d_family_tags, structural.d_empty_flags,
        )

    # Mixed file
    return _assemble_wkt_mixed(
        n_geoms, d_x, d_y,
        d_coord_value_counts, d_ring_counts, d_part_counts,
        structural.d_family_tags, structural.d_empty_flags,
        d_is_num_start, structural.d_depth,
        d_paren_starts, d_span_ends,
    )


def _build_empty_owned() -> OwnedGeometryArray:
    """Build an empty OwnedGeometryArray with zero rows."""
    from vibespatial.io.pylibcudf import _build_device_single_family_owned

    d_validity = cp.empty(0, dtype=cp.bool_)
    d_x = cp.empty(0, dtype=cp.float64)
    d_y = cp.empty(0, dtype=cp.float64)
    d_geom_offsets = cp.zeros(1, dtype=cp.int32)
    d_empty_mask = cp.empty(0, dtype=cp.bool_)

    return _build_device_single_family_owned(
        family=GeometryFamily.POINT,
        validity_device=d_validity,
        x_device=d_x,
        y_device=d_y,
        geometry_offsets_device=d_geom_offsets,
        empty_mask_device=d_empty_mask,
        detail="GPU WKT parse (empty)",
    )
