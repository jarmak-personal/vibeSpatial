"""NVRTC kernel sources for GPU GeoJSON parser."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# GeoJSON-specific kernel sources (Tier 1 NVRTC)
# ---------------------------------------------------------------------------

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

_MPOLY_COUNT_SOURCE = r"""
extern "C" __global__ void count_mpoly_levels(
    const unsigned char* __restrict__ input,
    const int* __restrict__ depth,
    const long long* __restrict__ coord_starts,
    const long long* __restrict__ coord_ends,
    const signed char* __restrict__ family_tags,
    int* __restrict__ part_counts,
    int* __restrict__ ring_counts,
    int* __restrict__ coord_pair_counts,
    int n_features,
    signed char mpoly_tag
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_features) return;

    // Only process MultiPolygon features
    if (family_tags[idx] != mpoly_tag) {
        part_counts[idx] = 0;
        ring_counts[idx] = 0;
        coord_pair_counts[idx] = 0;
        return;
    }

    long long start = coord_starts[idx] + 14;
    long long end = coord_ends[idx];
    if (start >= end) {
        part_counts[idx] = 0;
        ring_counts[idx] = 0;
        coord_pair_counts[idx] = 0;
        return;
    }

    while (start < end && input[start] != '[') start++;
    if (start >= end) {
        part_counts[idx] = 0;
        ring_counts[idx] = 0;
        coord_pair_counts[idx] = 0;
        return;
    }
    int coord_depth = depth[start];

    // MultiPolygon: coordinates is [[[[x,y], ...], ...], ...]
    // coord_depth = depth at outermost '['
    // polygon-part-closing ']' has depth = coord_depth
    // ring-closing ']' has depth = coord_depth + 1
    // pair-closing ']' has depth = coord_depth + 2
    int parts = 0;
    int rings = 0;
    int pairs = 0;

    for (long long i = start + 1; i < end; i++) {
        unsigned char c = input[i];
        int d = depth[i];
        if (c == ']') {
            if (d == coord_depth) parts++;
            else if (d == coord_depth + 1) rings++;
            else if (d == coord_depth + 2) pairs++;
        }
    }

    part_counts[idx] = parts;
    ring_counts[idx] = rings;
    coord_pair_counts[idx] = pairs;
}
"""

_MPOLY_SCATTER_SOURCE = r"""
extern "C" __global__ void scatter_mpoly_offsets(
    const unsigned char* __restrict__ input,
    const int* __restrict__ depth,
    const long long* __restrict__ coord_starts,
    const long long* __restrict__ coord_ends,
    const signed char* __restrict__ family_tags,
    const int* __restrict__ part_offset_starts,
    const int* __restrict__ ring_offset_starts,
    const int* __restrict__ pair_offset_starts,
    int* __restrict__ part_offsets,
    int* __restrict__ ring_offsets,
    int n_features,
    signed char mpoly_tag
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_features) return;

    // Only process MultiPolygon features
    if (family_tags[idx] != mpoly_tag) return;

    long long start = coord_starts[idx] + 14;
    long long end = coord_ends[idx];
    int part_out = part_offset_starts[idx];
    int ring_out = ring_offset_starts[idx];
    int pair_out = pair_offset_starts[idx];

    while (start < end && input[start] != '[') start++;
    if (start >= end) return;
    int coord_depth = depth[start];

    // Write starting offsets
    part_offsets[part_out] = ring_out;
    ring_offsets[ring_out] = pair_out;

    int parts_seen = 0;
    int rings_seen = 0;
    int pairs_seen = 0;

    for (long long i = start + 1; i < end; i++) {
        unsigned char c = input[i];
        int d = depth[i];
        if (c == ']' && d == coord_depth + 2) {
            pairs_seen++;
        }
        if (c == ']' && d == coord_depth + 1) {
            rings_seen++;
            ring_offsets[ring_out + rings_seen] = pair_out + pairs_seen;
        }
        if (c == ']' && d == coord_depth) {
            parts_seen++;
            part_offsets[part_out + parts_seen] = ring_out + rings_seen;
        }
    }
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

_TYPE_KEY_SOURCE = r"""
extern "C" __global__ void find_type_key(
    const unsigned char* __restrict__ input,
    const unsigned char* __restrict__ quote_parity,
    const int* __restrict__ depth,
    unsigned char* __restrict__ hits,
    long long n
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > n - 7) {
        if (idx < n) hits[idx] = 0;
        return;
    }

    // Pattern: "type":  (7 bytes)
    const unsigned char pat[7] = {'"','t','y','p','e','"',':'};

    unsigned char match = 1;
    for (int i = 0; i < 7; ++i) {
        if (input[idx + i] != pat[i]) { match = 0; break; }
    }
    // Check quote parity at the colon (idx+6): for a real JSON key the
    // opening and closing quotes cancel, so parity is 0 (even).
    // Inside a string value the parity is 1 (odd).
    if (match && quote_parity[idx + 6] != 0) {
        match = 0;
    }
    // Check depth at geometry level: depth 4 is inside geometry object.
    // Feature-level "type" is at depth 3, root-level at depth 1 — skip those.
    if (match && depth[idx + 6] != 4) {
        match = 0;
    }
    hits[idx] = match;
}
"""

_CLASSIFY_TYPE_SOURCE = r"""
extern "C" __global__ void classify_type_value(
    const unsigned char* __restrict__ input,
    const long long* __restrict__ type_positions,
    signed char* __restrict__ family_tags,
    int n_matches,
    long long n_bytes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_matches) return;

    long long pos = type_positions[idx] + 7;  // skip past "type":

    // Skip whitespace between colon and opening quote
    while (pos < n_bytes && (input[pos] == ' ' || input[pos] == '\n'
           || input[pos] == '\r' || input[pos] == '\t')) {
        pos++;
    }

    // Expect opening quote
    if (pos >= n_bytes || input[pos] != '"') {
        family_tags[idx] = -2;
        return;
    }
    pos++;  // skip the opening quote

    // Classify by prefix matching on the type string value
    if (pos >= n_bytes) {
        family_tags[idx] = -2;
        return;
    }

    if (input[pos] == 'P') {
        if (pos + 2 < n_bytes && input[pos + 1] == 'o' && input[pos + 2] == 'i') {
            family_tags[idx] = 0;  // Point
        } else if (pos + 2 < n_bytes && input[pos + 1] == 'o' && input[pos + 2] == 'l') {
            family_tags[idx] = 2;  // Polygon
        } else {
            family_tags[idx] = -2;
        }
    } else if (input[pos] == 'L') {
        if (pos + 1 < n_bytes && input[pos + 1] == 'i') {
            family_tags[idx] = 1;  // LineString
        } else {
            family_tags[idx] = -2;
        }
    } else if (input[pos] == 'M') {
        // Multi* types: "MultiPoint", "MultiLineString", "MultiPolygon"
        // After 'M' expect "ulti" then discriminate on next char.
        if (pos + 5 < n_bytes
            && input[pos + 1] == 'u' && input[pos + 2] == 'l'
            && input[pos + 3] == 't' && input[pos + 4] == 'i') {
            unsigned char mc = input[pos + 5];
            if (mc == 'P') {
                // "MultiPoint" vs "MultiPolygon": check pos+6
                if (pos + 6 < n_bytes && input[pos + 6] == 'o') {
                    // "MultiPo..." — check pos+7 for 'i'(int) vs 'l'(ygon)
                    if (pos + 7 < n_bytes && input[pos + 7] == 'i') {
                        family_tags[idx] = 3;  // MultiPoint
                    } else if (pos + 7 < n_bytes && input[pos + 7] == 'l') {
                        family_tags[idx] = 5;  // MultiPolygon
                    } else {
                        family_tags[idx] = -2;
                    }
                } else {
                    family_tags[idx] = -2;
                }
            } else if (mc == 'L') {
                family_tags[idx] = 4;  // MultiLineString
            } else {
                family_tags[idx] = -2;
            }
        } else {
            family_tags[idx] = -2;
        }
    } else if (input[pos] == 'G') {
        family_tags[idx] = -2;  // GeometryCollection, unsupported
    } else {
        family_tags[idx] = -2;
    }
}
"""

# GeoJSON-specific kernel name tuples
_COORD_KEY_NAMES = ("find_coord_key",)
_COORD_SPAN_END_NAMES = ("coord_span_end",)
_RING_COUNT_NAMES = ("count_rings_and_coords",)
_MPOLY_COUNT_NAMES = ("count_mpoly_levels",)
_MPOLY_SCATTER_NAMES = ("scatter_mpoly_offsets",)
_SCATTER_COORDS_NAMES = ("scatter_ring_offsets",)
_FEATURE_BOUNDARY_NAMES = ("find_feature_boundaries",)
_TYPE_KEY_NAMES = ("find_type_key",)
_CLASSIFY_TYPE_NAMES = ("classify_type_value",)
