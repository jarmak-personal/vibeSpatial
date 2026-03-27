"""NVRTC kernels for extract_unique_points: per-geometry coordinate deduplication.

ADR-0033: Tier 1 NVRTC for geometry-specific coordinate range extraction
and unique-pair marking.  Tier 3a CCCL for segmented sort, prefix sum,
and compaction.

ADR-0002: CONSTRUCTIVE class -- fp64 uniform precision (coordinates are
exact subsets of input, no arithmetic).

Two-pass count-scatter architecture:
    Pass 1 (count_coords): Count coordinates per geometry row across all
        6 geometry families.  One thread per row.
    Pass 2 (mark_unique):  After coordinates are gathered and sorted by x
        within each row-segment, mark the first occurrence of each unique
        (x, y) pair.  A coordinate is unique if it differs from its
        predecessor in x or y, or is the first coordinate in its segment.
"""

from __future__ import annotations

from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import FAMILY_TAGS

_FAMILY_POINT = FAMILY_TAGS[GeometryFamily.POINT]
_FAMILY_LINESTRING = FAMILY_TAGS[GeometryFamily.LINESTRING]
_FAMILY_POLYGON = FAMILY_TAGS[GeometryFamily.POLYGON]
_FAMILY_MULTIPOINT = FAMILY_TAGS[GeometryFamily.MULTIPOINT]
_FAMILY_MULTILINESTRING = FAMILY_TAGS[GeometryFamily.MULTILINESTRING]
_FAMILY_MULTIPOLYGON = FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]


# ---------------------------------------------------------------------------
# Kernel source: count coordinates per geometry row
# ---------------------------------------------------------------------------
# One thread per valid row.  Walks the offset hierarchy for the row's
# geometry family and counts coordinates.  For closed rings (polygon,
# multipolygon) we include the closing vertex because Shapely's
# extract_unique_points includes ring-closure coords if they are
# distinct from other vertices.

_COUNT_COORDS_SOURCE = """
#define FAMILY_POINT {family_point}
#define FAMILY_LINESTRING {family_linestring}
#define FAMILY_POLYGON {family_polygon}
#define FAMILY_MULTIPOINT {family_multipoint}
#define FAMILY_MULTILINESTRING {family_multilinestring}
#define FAMILY_MULTIPOLYGON {family_multipolygon}

extern "C" __global__ void __launch_bounds__(256, 4)
count_coords_per_row(
    const int* __restrict__ valid_rows,       /* global row indices */
    const int* __restrict__ family_codes,     /* int32 per global row */
    const int* __restrict__ family_row_off,   /* global->family-local row */
    const int* __restrict__ geom_off,         /* geometry offsets (per-family) */
    const int* __restrict__ part_off,         /* part offsets (may be dummy) */
    const int* __restrict__ ring_off,         /* ring offsets (may be dummy) */
    const unsigned char* __restrict__ empty_mask, /* per family-local row */
    int* __restrict__ coord_counts,           /* output: count per valid row */
    const int n_valid
) {{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_valid) return;

    const int global_row = valid_rows[tid];
    const int family = family_codes[global_row];
    const int fam_row = family_row_off[global_row];

    if (empty_mask[tid]) {{
        coord_counts[tid] = 0;
        return;
    }}

    int count = 0;

    if (family == FAMILY_POINT) {{
        /* Point: geometry_offsets give coordinate range */
        const int cs = geom_off[fam_row];
        const int ce = geom_off[fam_row + 1];
        count = ce - cs;

    }} else if (family == FAMILY_LINESTRING) {{
        const int cs = geom_off[fam_row];
        const int ce = geom_off[fam_row + 1];
        count = ce - cs;

    }} else if (family == FAMILY_POLYGON) {{
        /* geom_off -> ring indices; ring_off -> coordinate indices */
        const int rs = geom_off[fam_row];
        const int re = geom_off[fam_row + 1];
        for (int ri = rs; ri < re; ++ri) {{
            const int cs = ring_off[ri];
            const int ce = ring_off[ri + 1];
            count += ce - cs;
        }}

    }} else if (family == FAMILY_MULTIPOINT) {{
        const int cs = geom_off[fam_row];
        const int ce = geom_off[fam_row + 1];
        count = ce - cs;

    }} else if (family == FAMILY_MULTILINESTRING) {{
        /* geom_off -> part indices; part_off -> coordinate indices */
        const int ps = geom_off[fam_row];
        const int pe = geom_off[fam_row + 1];
        for (int pi = ps; pi < pe; ++pi) {{
            const int cs = part_off[pi];
            const int ce = part_off[pi + 1];
            count += ce - cs;
        }}

    }} else if (family == FAMILY_MULTIPOLYGON) {{
        /* geom_off -> part(polygon) indices; part_off -> ring indices;
           ring_off -> coordinate indices */
        const int ps = geom_off[fam_row];
        const int pe = geom_off[fam_row + 1];
        for (int pi = ps; pi < pe; ++pi) {{
            const int rs = part_off[pi];
            const int re = part_off[pi + 1];
            for (int ri = rs; ri < re; ++ri) {{
                const int cs = ring_off[ri];
                const int ce = ring_off[ri + 1];
                count += ce - cs;
            }}
        }}
    }}

    coord_counts[tid] = count;
}}


/* -----------------------------------------------------------------------
 * Scatter coordinates into flat output arrays.
 * One thread per family-local valid row.  Writes coordinates contiguously
 * starting at coord_offsets[tid].
 *
 * row_id_map[tid] maps the family-local thread index to the merged
 * valid-row position used for segmented sort and unique marking.
 * ----------------------------------------------------------------------- */
extern "C" __global__ void __launch_bounds__(256, 4)
scatter_coords(
    const int* __restrict__ valid_rows,
    const int* __restrict__ family_codes,
    const int* __restrict__ family_row_off,
    const int* __restrict__ geom_off,
    const int* __restrict__ part_off,
    const int* __restrict__ ring_off,
    const unsigned char* __restrict__ empty_mask,
    const double* __restrict__ x_in,
    const double* __restrict__ y_in,
    const int* __restrict__ coord_offsets,   /* per family-local tid */
    const int* __restrict__ row_id_map,      /* tid -> merged valid-row pos */
    double* __restrict__ x_out,
    double* __restrict__ y_out,
    int* __restrict__ row_ids,               /* row id per output coord */
    const int n_valid
) {{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_valid) return;
    if (empty_mask[tid]) return;

    const int global_row = valid_rows[tid];
    const int family = family_codes[global_row];
    const int fam_row = family_row_off[global_row];
    int wp = coord_offsets[tid];
    const int merged_row_id = row_id_map[tid];

    if (family == FAMILY_POINT || family == FAMILY_LINESTRING ||
        family == FAMILY_MULTIPOINT) {{
        const int cs = geom_off[fam_row];
        const int ce = geom_off[fam_row + 1];
        for (int c = cs; c < ce; ++c) {{
            x_out[wp] = x_in[c];
            y_out[wp] = y_in[c];
            row_ids[wp] = merged_row_id;
            ++wp;
        }}

    }} else if (family == FAMILY_POLYGON) {{
        const int rs = geom_off[fam_row];
        const int re = geom_off[fam_row + 1];
        for (int ri = rs; ri < re; ++ri) {{
            const int cs = ring_off[ri];
            const int ce = ring_off[ri + 1];
            for (int c = cs; c < ce; ++c) {{
                x_out[wp] = x_in[c];
                y_out[wp] = y_in[c];
                row_ids[wp] = merged_row_id;
                ++wp;
            }}
        }}

    }} else if (family == FAMILY_MULTILINESTRING) {{
        const int ps = geom_off[fam_row];
        const int pe = geom_off[fam_row + 1];
        for (int pi = ps; pi < pe; ++pi) {{
            const int cs = part_off[pi];
            const int ce = part_off[pi + 1];
            for (int c = cs; c < ce; ++c) {{
                x_out[wp] = x_in[c];
                y_out[wp] = y_in[c];
                row_ids[wp] = merged_row_id;
                ++wp;
            }}
        }}

    }} else if (family == FAMILY_MULTIPOLYGON) {{
        const int ps = geom_off[fam_row];
        const int pe = geom_off[fam_row + 1];
        for (int pi = ps; pi < pe; ++pi) {{
            const int rs = part_off[pi];
            const int re = part_off[pi + 1];
            for (int ri = rs; ri < re; ++ri) {{
                const int cs = ring_off[ri];
                const int ce = ring_off[ri + 1];
                for (int c = cs; c < ce; ++c) {{
                    x_out[wp] = x_in[c];
                    y_out[wp] = y_in[c];
                    row_ids[wp] = merged_row_id;
                    ++wp;
                }}
            }}
        }}
    }}
}}


/* -----------------------------------------------------------------------
 * Mark unique (x, y) pairs within each row-segment.
 *
 * Precondition: x_sorted is sorted within each segment (rows delimited
 * by seg_starts / seg_ends, or equivalently row_ids are monotonically
 * grouped and x is sorted within each group).
 *
 * A coordinate at position i is "unique" if:
 *   - It is the first coordinate in its segment (row_ids[i] != row_ids[i-1]
 *     or i == 0), OR
 *   - x_sorted[i] != x_sorted[i-1] OR y_sorted[i] != y_sorted[i-1]
 *
 * Output: unique_mask[i] = 1 if unique, 0 if duplicate.
 * ----------------------------------------------------------------------- */
extern "C" __global__ void __launch_bounds__(256, 4)
mark_unique_coords(
    const double* __restrict__ x_sorted,
    const double* __restrict__ y_sorted,
    const int* __restrict__ row_ids,       /* row id per coordinate */
    unsigned char* __restrict__ unique_mask,
    const int total_coords
) {{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_coords) return;

    if (i == 0) {{
        unique_mask[i] = 1;
        return;
    }}

    /* Different row -> always unique (first in segment) */
    if (row_ids[i] != row_ids[i - 1]) {{
        unique_mask[i] = 1;
        return;
    }}

    /* Same row: unique iff x or y differs from predecessor */
    /* Use bitwise OR on the inequality to avoid branch divergence */
    const int x_diff = (x_sorted[i] != x_sorted[i - 1]);
    const int y_diff = (y_sorted[i] != y_sorted[i - 1]);
    unique_mask[i] = (unsigned char)(x_diff | y_diff);
}}
"""

_KERNEL_NAMES = ("count_coords_per_row", "scatter_coords", "mark_unique_coords")


def _get_kernel_source() -> str:
    """Return the formatted NVRTC kernel source string."""
    return _COUNT_COORDS_SOURCE.format(
        family_point=_FAMILY_POINT,
        family_linestring=_FAMILY_LINESTRING,
        family_polygon=_FAMILY_POLYGON,
        family_multipoint=_FAMILY_MULTIPOINT,
        family_multilinestring=_FAMILY_MULTILINESTRING,
        family_multipolygon=_FAMILY_MULTIPOLYGON,
    )


def _get_kernel_names() -> tuple[str, ...]:
    """Return the tuple of kernel entry point names."""
    return _KERNEL_NAMES


# Module-level source string for precompilation
KERNEL_SOURCE = _get_kernel_source()
