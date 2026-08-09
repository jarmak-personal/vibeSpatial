"""NVRTC kernels for extract_unique_points: per-geometry coordinate deduplication.

ADR-0033: Tier 1 NVRTC for geometry-specific coordinate range extraction
and unique-pair marking.  Tier 3a CCCL for segmented sort, prefix sum,
and compaction.

ADR-0002: CONSTRUCTIVE class -- fp64 uniform precision (coordinates are
exact subsets of input, no arithmetic).

Capacity-preserving count-scatter architecture:
    Pass 1 (count_coords): Count coordinates per public geometry row.  A
        fixed family launch writes only rows matching that family tag.
    Pass 2 (scatter_coords): Gather coordinates into a row-segmented buffer
        whose allocation retains input-coordinate capacity.
    Pass 3 (mark_unique): Mark the first occurrence of each sorted pair up to
        the device-resident active-coordinate total.
    Pass 4 (scatter_unique): Pack marked coordinates into the output capacity
        without materialising a sparse index or exporting a count.
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
# One thread per public row.  Walks the offset hierarchy for the row's
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
    const unsigned char* __restrict__ validity,
    const int* __restrict__ family_codes,     /* int32 per global row */
    const int* __restrict__ family_row_off,   /* global->family-local row */
    const int* __restrict__ geom_off,         /* geometry offsets (per-family) */
    const int* __restrict__ part_off,         /* part offsets (may be dummy) */
    const int* __restrict__ ring_off,         /* ring offsets (may be dummy) */
    const unsigned char* __restrict__ empty_mask, /* per family-local row */
    int* __restrict__ coord_counts,           /* output: count per public row */
    const int expected_family,
    const int row_count
) {{
    const int global_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_row >= row_count) return;
    if (!validity[global_row]) return;
    const int family = family_codes[global_row];
    if (family != expected_family) return;
    const int fam_row = family_row_off[global_row];

    if (fam_row < 0 || empty_mask[fam_row]) return;

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

    coord_counts[global_row] = count;
}}


/* -----------------------------------------------------------------------
 * Scatter coordinates into flat output arrays.
 * One thread per public row.  A fixed family launch writes only rows matching
 * expected_family, starting at coord_offsets[global_row].
 * ----------------------------------------------------------------------- */
extern "C" __global__ void __launch_bounds__(256, 4)
scatter_coords(
    const unsigned char* __restrict__ validity,
    const int* __restrict__ family_codes,
    const int* __restrict__ family_row_off,
    const int* __restrict__ geom_off,
    const int* __restrict__ part_off,
    const int* __restrict__ ring_off,
    const unsigned char* __restrict__ empty_mask,
    const double* __restrict__ x_in,
    const double* __restrict__ y_in,
    const int* __restrict__ coord_offsets,   /* per public row */
    double* __restrict__ x_out,
    double* __restrict__ y_out,
    int* __restrict__ row_ids,               /* public row per output coord */
    const int expected_family,
    const int row_count
) {{
    const int global_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_row >= row_count) return;
    if (!validity[global_row]) return;
    const int family = family_codes[global_row];
    if (family != expected_family) return;
    const int fam_row = family_row_off[global_row];
    if (fam_row < 0 || empty_mask[fam_row]) return;
    int wp = coord_offsets[global_row];

    if (family == FAMILY_POINT || family == FAMILY_LINESTRING ||
        family == FAMILY_MULTIPOINT) {{
        const int cs = geom_off[fam_row];
        const int ce = geom_off[fam_row + 1];
        for (int c = cs; c < ce; ++c) {{
            x_out[wp] = x_in[c];
            y_out[wp] = y_in[c];
            row_ids[wp] = global_row;
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
                row_ids[wp] = global_row;
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
                row_ids[wp] = global_row;
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
                    row_ids[wp] = global_row;
                    ++wp;
                }}
            }}
        }}
    }}
}}


/* -----------------------------------------------------------------------
 * Emit one collapsed point per active LineString and one per nonempty part
 * of an active MultiLineString. Zero-length cleanup needs unique collapsed
 * part coordinates, not every repeated coordinate in each part.
 * ----------------------------------------------------------------------- */
extern "C" __global__ void __launch_bounds__(256, 4)
count_degenerate_line_candidates(
    const unsigned char* __restrict__ validity,
    const int* __restrict__ family_codes,
    const int* __restrict__ family_row_off,
    const int* __restrict__ geom_off,
    const int* __restrict__ part_off,
    const unsigned char* __restrict__ empty_mask,
    int* __restrict__ candidate_counts,
    const int expected_family,
    const int row_count
) {{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count || !validity[row]) return;
    if (family_codes[row] != expected_family) return;
    const int family_row = family_row_off[row];
    if (family_row < 0 || empty_mask[family_row]) return;

    if (expected_family == FAMILY_LINESTRING) {{
        candidate_counts[row] =
            (geom_off[family_row + 1] > geom_off[family_row]) ? 1 : 0;
        return;
    }}

    int count = 0;
    const int part_start = geom_off[family_row];
    const int part_end = geom_off[family_row + 1];
    for (int part = part_start; part < part_end; ++part) {{
        count += (part_off[part + 1] > part_off[part]) ? 1 : 0;
    }}
    candidate_counts[row] = count;
}}


extern "C" __global__ void __launch_bounds__(256, 4)
scatter_degenerate_line_candidates(
    const unsigned char* __restrict__ validity,
    const int* __restrict__ family_codes,
    const int* __restrict__ family_row_off,
    const int* __restrict__ geom_off,
    const int* __restrict__ part_off,
    const unsigned char* __restrict__ empty_mask,
    const double* __restrict__ x_in,
    const double* __restrict__ y_in,
    const int* __restrict__ candidate_offsets,
    double* __restrict__ x_out,
    double* __restrict__ y_out,
    int* __restrict__ row_ids,
    const int expected_family,
    const int row_count
) {{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count || !validity[row]) return;
    if (family_codes[row] != expected_family) return;
    const int family_row = family_row_off[row];
    if (family_row < 0 || empty_mask[family_row]) return;
    int write_position = candidate_offsets[row];

    if (expected_family == FAMILY_LINESTRING) {{
        const int coord_start = geom_off[family_row];
        if (geom_off[family_row + 1] > coord_start) {{
            x_out[write_position] = x_in[coord_start];
            y_out[write_position] = y_in[coord_start];
            row_ids[write_position] = row;
        }}
        return;
    }}

    const int part_start = geom_off[family_row];
    const int part_end = geom_off[family_row + 1];
    for (int part = part_start; part < part_end; ++part) {{
        const int coord_start = part_off[part];
        if (part_off[part + 1] <= coord_start) continue;
        x_out[write_position] = x_in[coord_start];
        y_out[write_position] = y_in[coord_start];
        row_ids[write_position] = row;
        ++write_position;
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
    const int* __restrict__ active_total,
    const int coordinate_capacity
) {{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= coordinate_capacity) return;
    if (i >= active_total[0]) {{
        unique_mask[i] = 0;
        return;
    }}

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


/* -----------------------------------------------------------------------
 * Scatter marked coordinates using an inclusive unique-prefix array.
 * The output allocation retains input-coordinate capacity; geometry offsets
 * delimit the active packed prefix.
 * ----------------------------------------------------------------------- */
extern "C" __global__ void __launch_bounds__(256, 4)
scatter_unique_coords(
    const double* __restrict__ x_sorted,
    const double* __restrict__ y_sorted,
    const unsigned char* __restrict__ unique_mask,
    const int* __restrict__ unique_prefix,
    double* __restrict__ x_out,
    double* __restrict__ y_out,
    const int coordinate_capacity
) {{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= coordinate_capacity || !unique_mask[i]) return;
    const int destination = unique_prefix[i] - 1;
    x_out[destination] = x_sorted[i];
    y_out[destination] = y_sorted[i];
}}


/* -----------------------------------------------------------------------
 * Compensated centroid of each unique-point row. This matches the existing
 * fp64 MultiPoint centroid reduction without routing through family row
 * compaction. Inactive and empty rows remain NaN.
 * ----------------------------------------------------------------------- */
extern "C" __global__ void __launch_bounds__(256, 4)
mean_unique_coords(
    const double* __restrict__ x_unique,
    const double* __restrict__ y_unique,
    const int* __restrict__ unique_offsets,
    const unsigned char* __restrict__ validity,
    double* __restrict__ x_mean,
    double* __restrict__ y_mean,
    const int row_count
) {{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count || !validity[row]) return;
    const int start = unique_offsets[row];
    const int end = unique_offsets[row + 1];
    const int count = end - start;
    if (count <= 0) return;

    double sum_x = 0.0;
    double sum_y = 0.0;
    double compensation_x = 0.0;
    double compensation_y = 0.0;
    for (int i = start; i < end; ++i) {{
        const double adjusted_x = x_unique[i] - compensation_x;
        const double next_x = sum_x + adjusted_x;
        compensation_x = (next_x - sum_x) - adjusted_x;
        sum_x = next_x;

        const double adjusted_y = y_unique[i] - compensation_y;
        const double next_y = sum_y + adjusted_y;
        compensation_y = (next_y - sum_y) - adjusted_y;
        sum_y = next_y;
    }}
    x_mean[row] = sum_x / (double)count;
    y_mean[row] = sum_y / (double)count;
}}
"""

_KERNEL_NAMES = (
    "count_coords_per_row",
    "scatter_coords",
    "count_degenerate_line_candidates",
    "scatter_degenerate_line_candidates",
    "mark_unique_coords",
    "scatter_unique_coords",
    "mean_unique_coords",
)


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
