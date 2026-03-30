"""NVRTC kernel sources for GPU KML reader."""

from __future__ import annotations

_KML_ASSIGN_GEOM_TYPE_SOURCE = r"""
// Per-Placemark geometry type classification kernel.
//
// Each thread handles one Placemark.  It scans the byte range
// [placemark_start, placemark_end) looking for the first geometry
// type opening tag.  Tag names are case-sensitive in XML/KML.
//
// Handles namespace prefixes: after '<', if we see alphabetic chars
// followed by ':', we skip the prefix and match the local name.
//
// Family tags:
//   0 = Point
//   1 = LineString
//   2 = Polygon
//   6 = MultiGeometry
//  -2 = unknown/none

extern "C" __global__ void __launch_bounds__(256, 4)
kml_assign_geometry_type(
    const unsigned char* __restrict__ input,
    const unsigned char* __restrict__ comment_mask,
    const long long* __restrict__ placemark_starts,
    const long long* __restrict__ placemark_ends,
    signed char* __restrict__ family_tags,
    const int n_placemarks,
    const long long n_bytes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_placemarks) return;

    long long start = placemark_starts[idx];
    long long end = placemark_ends[idx];
    if (end > n_bytes) end = n_bytes;

    family_tags[idx] = -2;  // default: unknown

    // Scan for '<' characters within this Placemark
    for (long long pos = start; pos < end - 1; pos++) {
        if (input[pos] != '<') continue;

        // Skip bytes inside XML comments
        if (comment_mask[pos]) continue;

        // Skip closing tags
        if (input[pos + 1] == '/') continue;

        // Skip XML declarations and processing instructions
        if (input[pos + 1] == '?' || input[pos + 1] == '!') continue;

        // Find the tag name start (skip '<')
        long long name_start = pos + 1;

        // Handle namespace prefix: skip "prefix:" if present
        // Scan for ':' before any whitespace or '>' or '/'
        long long scan = name_start;
        long long colon_pos = -1;
        while (scan < end && scan < name_start + 64) {
            unsigned char sc = input[scan];
            if (sc == ':') { colon_pos = scan; break; }
            if (sc == '>' || sc == '/' || sc == ' ' || sc == '\t'
                || sc == '\n' || sc == '\r') break;
            scan++;
        }
        if (colon_pos > 0) {
            name_start = colon_pos + 1;
        }

        // Now match the local tag name (case-sensitive for XML)
        // We need at least a few bytes to match
        if (name_start >= end) continue;

        // Match "Point"
        if (name_start + 5 <= end
            && input[name_start]     == 'P'
            && input[name_start + 1] == 'o'
            && input[name_start + 2] == 'i'
            && input[name_start + 3] == 'n'
            && input[name_start + 4] == 't') {
            // Verify next char is '>' or whitespace or '/' (not part of longer name)
            if (name_start + 5 >= end) { family_tags[idx] = 0; return; }
            unsigned char after = input[name_start + 5];
            if (after == '>' || after == ' ' || after == '\t'
                || after == '\n' || after == '\r' || after == '/') {
                family_tags[idx] = 0;  // Point
                return;
            }
        }

        // Match "LineString"
        if (name_start + 10 <= end
            && input[name_start]     == 'L'
            && input[name_start + 1] == 'i'
            && input[name_start + 2] == 'n'
            && input[name_start + 3] == 'e'
            && input[name_start + 4] == 'S'
            && input[name_start + 5] == 't'
            && input[name_start + 6] == 'r'
            && input[name_start + 7] == 'i'
            && input[name_start + 8] == 'n'
            && input[name_start + 9] == 'g') {
            if (name_start + 10 >= end) { family_tags[idx] = 1; return; }
            unsigned char after = input[name_start + 10];
            if (after == '>' || after == ' ' || after == '\t'
                || after == '\n' || after == '\r' || after == '/') {
                family_tags[idx] = 1;  // LineString
                return;
            }
        }

        // Match "Polygon"
        if (name_start + 7 <= end
            && input[name_start]     == 'P'
            && input[name_start + 1] == 'o'
            && input[name_start + 2] == 'l'
            && input[name_start + 3] == 'y'
            && input[name_start + 4] == 'g'
            && input[name_start + 5] == 'o'
            && input[name_start + 6] == 'n') {
            if (name_start + 7 >= end) { family_tags[idx] = 2; return; }
            unsigned char after = input[name_start + 7];
            if (after == '>' || after == ' ' || after == '\t'
                || after == '\n' || after == '\r' || after == '/') {
                family_tags[idx] = 2;  // Polygon
                return;
            }
        }

        // Match "MultiGeometry"
        if (name_start + 13 <= end
            && input[name_start]      == 'M'
            && input[name_start + 1]  == 'u'
            && input[name_start + 2]  == 'l'
            && input[name_start + 3]  == 't'
            && input[name_start + 4]  == 'i'
            && input[name_start + 5]  == 'G'
            && input[name_start + 6]  == 'e'
            && input[name_start + 7]  == 'o'
            && input[name_start + 8]  == 'm'
            && input[name_start + 9]  == 'e'
            && input[name_start + 10] == 't'
            && input[name_start + 11] == 'r'
            && input[name_start + 12] == 'y') {
            if (name_start + 13 >= end) { family_tags[idx] = 6; return; }
            unsigned char after = input[name_start + 13];
            if (after == '>' || after == ' ' || after == '\t'
                || after == '\n' || after == '\r' || after == '/') {
                family_tags[idx] = 6;  // MultiGeometry
                return;
            }
        }
    }
}
"""

_KML_ASSIGN_GEOM_TYPE_NAMES: tuple[str, ...] = ("kml_assign_geometry_type",)

_KML_COUNT_COMMAS_SPACES_SOURCE = r"""
// Count commas and spaces per coordinate region.
//
// Each thread handles one coordinate region.  It scans the byte range
// [coord_starts[i], coord_ends[i]) and counts:
//   - commas (0x2C) => commas_out[i]
//   - tuple separators (space/newline/tab/cr) => spaces_out[i]
//     (consecutive separators are collapsed to one count)
//
// These counts let the host infer dimensionality without materializing
// the full coordinate text.

extern "C" __global__ void __launch_bounds__(256, 4)
kml_count_commas_spaces(
    const unsigned char* __restrict__ input,
    const long long* __restrict__ coord_starts,
    const long long* __restrict__ coord_ends,
    int* __restrict__ commas_out,
    int* __restrict__ spaces_out,
    const int n_regions
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_regions) return;

    long long start = coord_starts[idx];
    long long end = coord_ends[idx];

    int commas = 0;
    int sep_count = 0;
    int in_separator = 1;  // treat start as after a separator

    for (long long pos = start; pos < end; pos++) {
        unsigned char c = input[pos];
        if (c == ',') {
            commas++;
            in_separator = 0;
        } else if (c == ' ' || c == '\n' || c == '\r' || c == '\t') {
            if (!in_separator) {
                sep_count++;
                in_separator = 1;
            }
        } else {
            in_separator = 0;
        }
    }

    commas_out[idx] = commas;
    spaces_out[idx] = sep_count;
}
"""

_KML_COUNT_COMMAS_SPACES_NAMES: tuple[str, ...] = ("kml_count_commas_spaces",)

_KML_ASSIGN_COORD_REGIONS_SOURCE = r"""
// Assign coordinate regions to Placemarks.
//
// Each thread handles one coordinate region.  It binary-searches the
// Placemark boundaries to find which Placemark contains this region.
//
// Output: placemark_idx[i] = index of the Placemark containing
//         coordinate region i, or -1 if none.

extern "C" __global__ void __launch_bounds__(256, 4)
kml_assign_coord_regions(
    const long long* __restrict__ coord_starts,
    const long long* __restrict__ placemark_starts,
    const long long* __restrict__ placemark_ends,
    int* __restrict__ placemark_idx,
    const int n_coord_regions,
    const int n_placemarks
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_coord_regions) return;

    long long cs = coord_starts[idx];

    // Binary search: find the last Placemark whose start <= cs
    int lo = 0;
    int hi = n_placemarks - 1;
    int best = -1;

    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (placemark_starts[mid] <= cs) {
            best = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }

    // Verify the coordinate region is within the Placemark's range
    if (best >= 0 && cs < placemark_ends[best]) {
        placemark_idx[idx] = best;
    } else {
        placemark_idx[idx] = -1;
    }
}
"""

_KML_ASSIGN_COORD_REGIONS_NAMES: tuple[str, ...] = ("kml_assign_coord_regions",)

