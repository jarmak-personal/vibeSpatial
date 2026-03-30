"""NVRTC kernel sources for GPU WKT parser."""

from __future__ import annotations

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

