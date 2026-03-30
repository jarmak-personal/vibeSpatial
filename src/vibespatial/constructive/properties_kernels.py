"""NVRTC kernel sources for properties."""

from __future__ import annotations

from vibespatial.cuda.device_functions.signed_area import SIGNED_AREA_DEVICE

# ---------------------------------------------------------------------------
# NVRTC kernels for is_closed and is_ccw (Tier 1, ADR-0033)
# ---------------------------------------------------------------------------

# is_closed: compares first and last coordinate per span
_IS_CLOSED_KERNEL_SOURCE = r"""
extern "C" __global__ void is_closed_linestring(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ geometry_offsets,
    const int* __restrict__ rows,
    int* __restrict__ out,
    int n
) {{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int row = rows[i];
    const int start = geometry_offsets[row];
    const int end = geometry_offsets[row + 1];

    if (end - start < 2) {{
        out[i] = 1;
        return;
    }}

    out[i] = (x[start] == x[end - 1] && y[start] == y[end - 1]) ? 1 : 0;
}}

extern "C" __global__ void is_closed_multilinestring(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ part_offsets,
    const int* __restrict__ geometry_offsets,
    const int* __restrict__ rows,
    int* __restrict__ out,
    int n
) {{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int row = rows[i];
    const int ps = geometry_offsets[row];
    const int pe = geometry_offsets[row + 1];

    int closed = 1;
    for (int p = ps; p < pe; p++) {{
        const int cs = part_offsets[p];
        const int ce = part_offsets[p + 1];
        if (ce - cs < 2) continue;
        if (x[cs] != x[ce - 1] || y[cs] != y[ce - 1]) {{
            closed = 0;
            break;
        }}
    }}
    out[i] = closed;
}}
"""
_IS_CLOSED_KERNEL_NAMES = ("is_closed_linestring", "is_closed_multilinestring")
_IS_CLOSED_FP64 = _IS_CLOSED_KERNEL_SOURCE.format()
# is_ccw: shoelace signed area on exterior ring (uses shared vs_ring_signed_area_2x)
_IS_CCW_KERNEL_SOURCE = SIGNED_AREA_DEVICE + r"""
extern "C" __global__ void is_ccw_polygon(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ ring_offsets,
    const int* __restrict__ geometry_offsets,
    const int* __restrict__ rows,
    int* __restrict__ out,
    int n
) {{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int row = rows[i];
    const int ring_idx = geometry_offsets[row];
    const int coord_start = ring_offsets[ring_idx];
    const int coord_end = ring_offsets[ring_idx + 1];

    const int ncoords = coord_end - coord_start;
    if (ncoords < 3) {{
        out[i] = 0;
        return;
    }}

    const double area2 = vs_ring_signed_area_2x(x, y, coord_start, coord_end);
    out[i] = (area2 > 0.0) ? 1 : 0;
}}

extern "C" __global__ void is_ccw_multipolygon(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ ring_offsets,
    const int* __restrict__ part_offsets,
    const int* __restrict__ geometry_offsets,
    const int* __restrict__ rows,
    int* __restrict__ out,
    int n
) {{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int row = rows[i];
    const int poly_start = geometry_offsets[row];
    const int ring_idx = part_offsets[poly_start];
    const int coord_start = ring_offsets[ring_idx];
    const int coord_end = ring_offsets[ring_idx + 1];

    const int ncoords = coord_end - coord_start;
    if (ncoords < 3) {{
        out[i] = 0;
        return;
    }}

    const double area2 = vs_ring_signed_area_2x(x, y, coord_start, coord_end);
    out[i] = (area2 > 0.0) ? 1 : 0;
}}
"""
_IS_CCW_KERNEL_NAMES = ("is_ccw_polygon", "is_ccw_multipolygon")
_IS_CCW_FP64 = _IS_CCW_KERNEL_SOURCE.format()
# ---------------------------------------------------------------------------
# NVRTC offset_diff kernel (Tier 1, ADR-0033)
#
# Replaces CuPy fancy indexing for simple-family offset arithmetic:
#   out[i] = offsets[rows[i] + 1] - offsets[rows[i]]
# Also provides an interior-ring variant that subtracts 1 and clamps to 0.
# ---------------------------------------------------------------------------

_OFFSET_DIFF_KERNEL_SOURCE = r"""
extern "C" __global__ void offset_diff(
    const int* __restrict__ offsets,
    const int* __restrict__ rows,
    int* __restrict__ out,
    int n
) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int row = rows[i];
    out[i] = offsets[row + 1] - offsets[row];
}}

extern "C" __global__ void offset_diff_interior_rings(
    const int* __restrict__ offsets,
    const int* __restrict__ rows,
    int* __restrict__ out,
    int n
) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int row = rows[i];
    int diff = offsets[row + 1] - offsets[row] - 1;
    out[i] = diff > 0 ? diff : 0;
}}
"""
_OFFSET_DIFF_KERNEL_NAMES = ("offset_diff", "offset_diff_interior_rings")
_OFFSET_DIFF_FP64 = _OFFSET_DIFF_KERNEL_SOURCE.format()
# ---------------------------------------------------------------------------
# NVRTC kernel for num_coordinates on POLYGON family (ring indirection)
# Needed because CuPy cannot express the double-indirect offset walk
# without Python loops.
# ---------------------------------------------------------------------------

_NUM_COORDS_KERNEL_SOURCE = r"""
extern "C" __global__ void num_coords_polygon(
    const int* __restrict__ ring_offsets,
    const int* __restrict__ geometry_offsets,
    const int* __restrict__ rows,
    int* __restrict__ out,
    int n
) {{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int row = rows[i];
    const int ring_start = geometry_offsets[row];
    const int ring_end = geometry_offsets[row + 1];

    int total = 0;
    for (int ri = ring_start; ri < ring_end; ri++) {{
        total += ring_offsets[ri + 1] - ring_offsets[ri];
    }}
    out[i] = total;
}}

extern "C" __global__ void num_coords_multilinestring(
    const int* __restrict__ part_offsets,
    const int* __restrict__ geometry_offsets,
    const int* __restrict__ rows,
    int* __restrict__ out,
    int n
) {{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int row = rows[i];
    const int part_start = geometry_offsets[row];
    const int part_end = geometry_offsets[row + 1];

    int total = 0;
    for (int pi = part_start; pi < part_end; pi++) {{
        total += part_offsets[pi + 1] - part_offsets[pi];
    }}
    out[i] = total;
}}

extern "C" __global__ void num_coords_multipolygon(
    const int* __restrict__ ring_offsets,
    const int* __restrict__ part_offsets,
    const int* __restrict__ geometry_offsets,
    const int* __restrict__ rows,
    int* __restrict__ out,
    int n
) {{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int row = rows[i];
    const int poly_start = geometry_offsets[row];
    const int poly_end = geometry_offsets[row + 1];

    int total = 0;
    for (int pi = poly_start; pi < poly_end; pi++) {{
        const int ring_start = part_offsets[pi];
        const int ring_end = part_offsets[pi + 1];
        for (int ri = ring_start; ri < ring_end; ri++) {{
            total += ring_offsets[ri + 1] - ring_offsets[ri];
        }}
    }}
    out[i] = total;
}}
"""
_NUM_COORDS_KERNEL_NAMES = (
    "num_coords_polygon",
    "num_coords_multilinestring",
    "num_coords_multipolygon",
)
_NUM_COORDS_FP64 = _NUM_COORDS_KERNEL_SOURCE.format()
