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

_POINT_PAIR_PARSE_SOURCE = r"""
__device__ __forceinline__ int is_ws(unsigned char c) {
    return c == ' ' || c == '\n' || c == '\r' || c == '\t';
}

__device__ __forceinline__ int match_literal(
    const unsigned char* __restrict__ input,
    long long n_bytes,
    long long pos,
    const unsigned char* __restrict__ pat,
    int pat_len
) {
    if (pos < 0 || pos + pat_len > n_bytes) {
        return 0;
    }
    for (int i = 0; i < pat_len; ++i) {
        if (input[pos + i] != pat[i]) {
            return 0;
        }
    }
    return 1;
}

__device__ __forceinline__ int parse_json_float_token(
    const unsigned char* __restrict__ input,
    long long n_bytes,
    long long start,
    double* __restrict__ out_value,
    long long* __restrict__ out_next
) {
    long long pos = start;
    int negative = 0;
    int saw_digit = 0;
    int in_exponent = 0;
    int exp_negative = 0;
    int exp_val = 0;
    double result = 0.0;
    double frac_mult = 0.0;

    while (pos < n_bytes) {
        unsigned char c = input[pos];
        if (c == '-') {
            if (in_exponent) {
                exp_negative = 1;
            } else if (pos == start) {
                negative = 1;
            } else {
                break;
            }
        } else if (c == '+') {
            if (!(in_exponent || pos == start)) {
                break;
            }
        } else if (c == '.') {
            if (frac_mult > 0.0 || in_exponent) {
                break;
            }
            frac_mult = 0.1;
        } else if (c == 'e' || c == 'E') {
            if (in_exponent || !saw_digit) {
                break;
            }
            in_exponent = 1;
        } else if (c >= '0' && c <= '9') {
            int d = c - '0';
            saw_digit = 1;
            if (in_exponent) {
                exp_val = exp_val * 10 + d;
            } else if (frac_mult > 0.0) {
                result += d * frac_mult;
                frac_mult *= 0.1;
            } else {
                result = result * 10.0 + d;
            }
        } else {
            break;
        }
        pos++;
    }

    if (!saw_digit) {
        return 0;
    }

    if (negative) {
        result = -result;
    }

    if (in_exponent) {
        double exp_mult = 1.0;
        for (int e = 0; e < exp_val; ++e) {
            exp_mult *= 10.0;
        }
        result = exp_negative ? (result / exp_mult) : (result * exp_mult);
    }

    *out_value = result;
    *out_next = pos;
    return 1;
}

extern "C" __global__ void find_geometry_key(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ hits,
    long long n
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > n - 10) {
        if (idx < n) hits[idx] = 0;
        return;
    }

    const unsigned char pat[10] = {
        '"','g','e','o','m','e','t','r','y','"'
    };

    unsigned char match = 1;
    for (int i = 0; i < 10; ++i) {
        if (input[idx + i] != pat[i]) {
            match = 0;
            break;
        }
    }
    // Compact point fast path runs before quote-parity. Reject escaped
    // matches so property strings containing \"geometry\" do not qualify.
    if (match && idx > 0 && input[idx - 1] == '\\') {
        match = 0;
    }
    hits[idx] = match;
}

extern "C" __global__ void parse_point_geometry_objects(
    const unsigned char* __restrict__ input,
    const long long* __restrict__ geometry_positions,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    unsigned char* __restrict__ valid,
    int n_features,
    long long n_bytes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_features) return;

    const unsigned char type_key[6] = {'"','t','y','p','e','"'};
    const unsigned char point_value[7] = {'"','P','o','i','n','t','"'};
    const unsigned char coord_key[13] = {
        '"','c','o','o','r','d','i','n','a','t','e','s','"'
    };

    long long pos = geometry_positions[idx] + 10;
    while (pos < n_bytes && is_ws(input[pos])) pos++;
    if (pos >= n_bytes || input[pos] != ':') goto invalid;
    pos++;
    while (pos < n_bytes && is_ws(input[pos])) pos++;
    if (pos >= n_bytes || input[pos] != '{') goto invalid;
    pos++;
    while (pos < n_bytes && is_ws(input[pos])) pos++;

    if (!match_literal(input, n_bytes, pos, type_key, 6)) goto invalid;
    pos += 6;
    while (pos < n_bytes && is_ws(input[pos])) pos++;
    if (pos >= n_bytes || input[pos] != ':') goto invalid;
    pos++;
    while (pos < n_bytes && is_ws(input[pos])) pos++;

    if (!match_literal(input, n_bytes, pos, point_value, 7)) goto invalid;
    pos += 7;
    while (pos < n_bytes && is_ws(input[pos])) pos++;
    if (pos >= n_bytes || input[pos] != ',') goto invalid;
    pos++;
    while (pos < n_bytes && is_ws(input[pos])) pos++;

    if (!match_literal(input, n_bytes, pos, coord_key, 13)) goto invalid;
    pos += 13;
    while (pos < n_bytes && is_ws(input[pos])) pos++;
    if (pos >= n_bytes || input[pos] != ':') goto invalid;
    pos++;
    while (pos < n_bytes && is_ws(input[pos])) pos++;
    if (pos >= n_bytes || input[pos] != '[') goto invalid;
    pos++;
    while (pos < n_bytes && is_ws(input[pos])) pos++;

    double x = 0.0;
    double y = 0.0;
    long long next = pos;
    if (!parse_json_float_token(input, n_bytes, pos, &x, &next)) goto invalid;

    pos = next;
    while (pos < n_bytes && is_ws(input[pos])) pos++;
    if (pos >= n_bytes || input[pos] != ',') goto invalid;
    pos++;
    while (pos < n_bytes && is_ws(input[pos])) pos++;

    if (!parse_json_float_token(input, n_bytes, pos, &y, &next)) goto invalid;
    pos = next;
    while (pos < n_bytes && is_ws(input[pos])) pos++;
    if (pos >= n_bytes || input[pos] != ']') goto invalid;

    valid[idx] = 1;
    out_x[idx] = x;
    out_y[idx] = y;
    return;

invalid:
    valid[idx] = 0;
    out_x[idx] = 0.0;
    out_y[idx] = 0.0;
}

extern "C" __global__ void parse_point_coordinate_pairs(
    const unsigned char* __restrict__ input,
    const long long* __restrict__ coord_positions,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    unsigned char* __restrict__ valid,
    int n_features,
    long long n_bytes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_features) return;

    long long pos = coord_positions[idx] + 14;
    while (pos < n_bytes && is_ws(input[pos])) pos++;
    if (pos >= n_bytes || input[pos] != '[') {
        valid[idx] = 0;
        out_x[idx] = 0.0;
        out_y[idx] = 0.0;
        return;
    }
    pos++;
    while (pos < n_bytes && is_ws(input[pos])) pos++;
    if (pos >= n_bytes || input[pos] == ']') {
        valid[idx] = 0;
        out_x[idx] = 0.0;
        out_y[idx] = 0.0;
        return;
    }

    double x = 0.0;
    double y = 0.0;
    long long next = pos;
    if (!parse_json_float_token(input, n_bytes, pos, &x, &next)) {
        valid[idx] = 0;
        out_x[idx] = 0.0;
        out_y[idx] = 0.0;
        return;
    }

    pos = next;
    while (pos < n_bytes && is_ws(input[pos])) pos++;
    if (pos >= n_bytes || input[pos] != ',') {
        valid[idx] = 0;
        out_x[idx] = 0.0;
        out_y[idx] = 0.0;
        return;
    }
    pos++;
    while (pos < n_bytes && is_ws(input[pos])) pos++;
    if (pos >= n_bytes || input[pos] == ']') {
        valid[idx] = 0;
        out_x[idx] = 0.0;
        out_y[idx] = 0.0;
        return;
    }

    if (!parse_json_float_token(input, n_bytes, pos, &y, &next)) {
        valid[idx] = 0;
        out_x[idx] = 0.0;
        out_y[idx] = 0.0;
        return;
    }

    valid[idx] = 1;
    out_x[idx] = x;
    out_y[idx] = y;
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
    const signed char* __restrict__ family_tags,
    const int* __restrict__ coord_pair_counts,
    const int* __restrict__ ring_offset_starts,
    const int* __restrict__ coord_pair_offset_starts,
    int* __restrict__ ring_offsets,
    int n_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_features) return;

    long long start = coord_starts[idx] + 14;
    long long end = coord_ends[idx];
    signed char family = family_tags[idx];
    int coord_pairs = coord_pair_counts[idx];
    int ring_out = ring_offset_starts[idx];
    int pair_out = coord_pair_offset_starts[idx];

    // Point / LineString / MultiPoint rows still need pseudo-segment
    // boundaries in the shared global ring-offset space so later polygon rows
    // can preserve their true coordinate starts.
    if (family == 0) {
        ring_offsets[ring_out] = pair_out;
        ring_offsets[ring_out + 1] = pair_out + coord_pairs;
        return;
    }
    if (family == 1 || family == 3) {
        for (int i = 0; i <= coord_pairs; ++i) {
            ring_offsets[ring_out + i] = pair_out + i;
        }
        return;
    }
    if (family != 2 && family != 4) {
        return;
    }

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
extern "C" __global__ void find_feature_starts(
    const unsigned char* __restrict__ input,
    const int* __restrict__ depth,
    unsigned char* __restrict__ is_feature_start,
    long long n
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    unsigned char c = input[idx];
    int d = depth[idx];

    // Features are objects at depth 3: FeatureCollection { depth=1,
    // "features": [ depth=2, Feature { depth=3.
    is_feature_start[idx] = (c == '{' && d == 3) ? 1 : 0;
}

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
_FEATURE_BOUNDARY_NAMES = ("find_feature_starts", "find_feature_boundaries")
_TYPE_KEY_NAMES = ("find_type_key",)
_CLASSIFY_TYPE_NAMES = ("classify_type_value",)
_POINT_PAIR_NAMES = (
    "find_geometry_key",
    "parse_point_geometry_objects",
    "parse_point_coordinate_pairs",
)
