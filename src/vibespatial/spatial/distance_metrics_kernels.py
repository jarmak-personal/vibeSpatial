"""NVRTC kernel sources for Hausdorff and discrete Frechet distance."""

from __future__ import annotations

from vibespatial.cuda.preamble import PRECISION_PREAMBLE

# ---------------------------------------------------------------------------
# Hausdorff kernel: 1 block per pair, shared-memory max reduction
# ---------------------------------------------------------------------------

_HAUSDORFF_KERNEL_SOURCE = PRECISION_PREAMBLE + r"""
extern "C" __global__ __launch_bounds__(256, 4)
void hausdorff_distance(
    const double* __restrict__ x_a,
    const double* __restrict__ y_a,
    const int* __restrict__ offsets_a,
    const double* __restrict__ x_b,
    const double* __restrict__ y_b,
    const int* __restrict__ offsets_b,
    double* __restrict__ result,
    double center_x,
    double center_y,
    int pair_count
) {{
    const int pair = blockIdx.x;
    if (pair >= pair_count) return;

    const int a_start = offsets_a[pair];
    const int a_end   = offsets_a[pair + 1];
    const int b_start = offsets_b[pair];
    const int b_end   = offsets_b[pair + 1];
    const int na = a_end - a_start;
    const int nb = b_end - b_start;

    if (na == 0 || nb == 0) {{
        if (threadIdx.x == 0) result[pair] = 0.0 / 0.0;  /* NaN */
        return;
    }}

    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    /* Forward direction: max over A of {{ min over B of dist(a,b) }} */
    compute_t local_max_forward = (compute_t)0.0;
    for (int i = tid; i < na; i += stride) {{
        const compute_t ax = CX(x_a[a_start + i]);
        const compute_t ay = CY(y_a[a_start + i]);
        compute_t min_dist_sq = (compute_t)1e38;
        for (int j = 0; j < nb; j++) {{
            const compute_t dx = ax - CX(x_b[b_start + j]);
            const compute_t dy = ay - CY(y_b[b_start + j]);
            const compute_t d_sq = dx * dx + dy * dy;
            if (d_sq < min_dist_sq) min_dist_sq = d_sq;
        }}
        const compute_t min_dist = sqrt((double)min_dist_sq);
        if (min_dist > local_max_forward) local_max_forward = min_dist;
    }}

    /* Backward direction: max over B of {{ min over A of dist(b,a) }} */
    compute_t local_max_backward = (compute_t)0.0;
    for (int j = tid; j < nb; j += stride) {{
        const compute_t bx = CX(x_b[b_start + j]);
        const compute_t by = CY(y_b[b_start + j]);
        compute_t min_dist_sq = (compute_t)1e38;
        for (int i = 0; i < na; i++) {{
            const compute_t dx = bx - CX(x_a[a_start + i]);
            const compute_t dy = by - CY(y_a[a_start + i]);
            const compute_t d_sq = dx * dx + dy * dy;
            if (d_sq < min_dist_sq) min_dist_sq = d_sq;
        }}
        const compute_t min_dist = sqrt((double)min_dist_sq);
        if (min_dist > local_max_backward) local_max_backward = min_dist;
    }}

    double local_max = (double)((local_max_forward > local_max_backward)
                                ? local_max_forward : local_max_backward);

    /* Block-wide max reduction via shared memory */
    __shared__ double sdata[256];
    sdata[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {{
        if (tid < s && sdata[tid + s] > sdata[tid]) {{
            sdata[tid] = sdata[tid + s];
        }}
        __syncthreads();
    }}

    if (tid == 0) result[pair] = sdata[0];
}}
"""

_HAUSDORFF_KERNEL_NAMES = ("hausdorff_distance",)

# ---------------------------------------------------------------------------
# Frechet kernel: 1 thread per pair, rolling-row DP in local memory
# ---------------------------------------------------------------------------

MAX_FRECHET_B = 2048

_FRECHET_KERNEL_SOURCE = PRECISION_PREAMBLE + r"""
#define MAX_FRECHET_B {max_frechet_b}

extern "C" __global__
void frechet_distance(
    const double* __restrict__ x_a,
    const double* __restrict__ y_a,
    const int* __restrict__ offsets_a,
    const double* __restrict__ x_b,
    const double* __restrict__ y_b,
    const int* __restrict__ offsets_b,
    double* __restrict__ result,
    double center_x,
    double center_y,
    int pair_count,
    int max_b_len
) {{
    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= pair_count) return;

    const int a_start = offsets_a[pair];
    const int a_end   = offsets_a[pair + 1];
    const int b_start = offsets_b[pair];
    const int b_end   = offsets_b[pair + 1];
    const int na = a_end - a_start;
    const int nb = b_end - b_start;

    if (na == 0 || nb == 0) {{
        result[pair] = 0.0 / 0.0;  /* NaN */
        return;
    }}

    if (nb > MAX_FRECHET_B) {{
        /* Too large for local arrays -- return NaN as safe fallback */
        result[pair] = 0.0 / 0.0;
        return;
    }}

    /* Rolling DP: previous and current rows, each of length nb.
       For large nb these spill to local memory (L1/L2 cached). */
    double prev_row[MAX_FRECHET_B];
    double curr_row[MAX_FRECHET_B];

    /* Initialize row 0: ca[0][j] = max(ca[0][j-1], d(a_0, b_j)) */
    {{
        const compute_t ax = CX(x_a[a_start]);
        const compute_t ay = CY(y_a[a_start]);
        {{
            const compute_t dx = ax - CX(x_b[b_start]);
            const compute_t dy = ay - CY(y_b[b_start]);
            prev_row[0] = sqrt((double)(dx * dx + dy * dy));
        }}
        for (int j = 1; j < nb; j++) {{
            const compute_t dx = ax - CX(x_b[b_start + j]);
            const compute_t dy = ay - CY(y_b[b_start + j]);
            const double d = sqrt((double)(dx * dx + dy * dy));
            prev_row[j] = (d > prev_row[j - 1]) ? d : prev_row[j - 1];
        }}
    }}

    /* Fill rows 1..na-1 */
    for (int i = 1; i < na; i++) {{
        const compute_t ax = CX(x_a[a_start + i]);
        const compute_t ay = CY(y_a[a_start + i]);

        /* curr_row[0] = max(prev_row[0], d(a_i, b_0)) */
        {{
            const compute_t dx = ax - CX(x_b[b_start]);
            const compute_t dy = ay - CY(y_b[b_start]);
            const double d = sqrt((double)(dx * dx + dy * dy));
            curr_row[0] = (d > prev_row[0]) ? d : prev_row[0];
        }}

        for (int j = 1; j < nb; j++) {{
            const compute_t dx = ax - CX(x_b[b_start + j]);
            const compute_t dy = ay - CY(y_b[b_start + j]);
            const double d = sqrt((double)(dx * dx + dy * dy));

            double min_prev = prev_row[j];          /* ca[i-1][j]   */
            if (curr_row[j - 1] < min_prev) min_prev = curr_row[j - 1]; /* ca[i][j-1]   */
            if (prev_row[j - 1] < min_prev) min_prev = prev_row[j - 1]; /* ca[i-1][j-1] */

            curr_row[j] = (d > min_prev) ? d : min_prev;
        }}

        /* Swap: prev_row = curr_row */
        for (int j = 0; j < nb; j++) prev_row[j] = curr_row[j];
    }}

    result[pair] = prev_row[nb - 1];
}}
"""

_FRECHET_KERNEL_NAMES = ("frechet_distance",)

# ---------------------------------------------------------------------------
# Pre-format kernel sources for fp64 and fp32 variants
# ---------------------------------------------------------------------------

HAUSDORFF_FP64 = _HAUSDORFF_KERNEL_SOURCE.format(compute_type="double")
HAUSDORFF_FP32 = _HAUSDORFF_KERNEL_SOURCE.format(compute_type="float")
FRECHET_FP64 = _FRECHET_KERNEL_SOURCE.format(
    compute_type="double", max_frechet_b=MAX_FRECHET_B,
)
FRECHET_FP32 = _FRECHET_KERNEL_SOURCE.format(
    compute_type="float", max_frechet_b=MAX_FRECHET_B,
)
