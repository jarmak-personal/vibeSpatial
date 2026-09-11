"""Packed STR construction and bounded nearest traversal device code."""

BOUNDS_SOURCE = r"""
#ifndef INFINITY
#define INFINITY __longlong_as_double(0x7ff0000000000000LL)
#endif
extern "C" __global__ __launch_bounds__(256) void compress_bounds(const double* __restrict__ input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    output[i] = i < n / 2 ? __double2float_rd(input[i]) : __double2float_ru(input[i]);
}
extern "C" __global__ __launch_bounds__(256) void packed_leaves(
    const double* __restrict__ ax, const double* __restrict__ ay, const double* __restrict__ bx, const double* __restrict__ by,
    double* bounds, int n, int first, int count, int total, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    double lx = INFINITY, ly = INFINITY, hx = -INFINITY, hy = -INFINITY;
    for (int j = i * width; j < min(n, (i + 1) * width); ++j) {
        lx = fmin(lx, fmin(ax[j], bx[j])); ly = fmin(ly, fmin(ay[j], by[j]));
        hx = fmax(hx, fmax(ax[j], bx[j])); hy = fmax(hy, fmax(ay[j], by[j]));
    }
    int node = first + i;
    bounds[node] = lx; bounds[total + node] = ly;
    bounds[2 * total + node] = hx; bounds[3 * total + node] = hy;
}
extern "C" __global__ __launch_bounds__(256) void str_parents(double* b, const int* __restrict__ children, int first, int count, int total, int leaves) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int node = first + i;
    double lx = INFINITY, ly = INFINITY, hx = -INFINITY, hy = -INFINITY;
    #pragma unroll
    for (int c = 0; c < FANOUT; ++c) {
        int child = children[(node - leaves) * FANOUT + c];
        if (child < 0) continue;
        lx = fmin(lx, b[child]); ly = fmin(ly, b[total + child]);
        hx = fmax(hx, b[2 * total + child]); hy = fmax(hy, b[3 * total + child]);
    }
    b[node] = lx; b[total + node] = ly; b[2 * total + node] = hx; b[3 * total + node] = hy;
}
"""

QUERY_SOURCE = r"""
#ifndef DBL_EPSILON
#define DBL_EPSILON 2.2204460492503131e-16
#endif
__device__ double envelope_lower(const INDEX_T* __restrict__ b, int node, int total,
                                 const double* __restrict__ qb, int q, int nq) {
#if FLOAT_BOUNDS
    float lx = __double2float_rd(qb[q]), ly = __double2float_rd(qb[nq + q]);
    float hx = __double2float_ru(qb[2 * nq + q]), hy = __double2float_ru(qb[3 * nq + q]);
    float dx = fmaxf(0.f, fmaxf(__fsub_rd(b[node], hx), __fsub_rd(lx, b[2 * total + node])));
    float dy = fmaxf(0.f, fmaxf(__fsub_rd(b[total + node], hy), __fsub_rd(ly, b[3 * total + node])));
    return (double)__fadd_rd(__fmul_rd(dx, dx), __fmul_rd(dy, dy));
#else
    double dx = fmax(0., fmax(__dsub_rd(b[node], qb[2 * nq + q]), __dsub_rd(qb[q], b[2 * total + node])));
    double dy = fmax(0., fmax(__dsub_rd(b[total + node], qb[3 * nq + q]), __dsub_rd(qb[nq + q], b[3 * total + node])));
    return __dadd_rd(__dmul_rd(dx, dx), __dmul_rd(dy, dy));
#endif
}
extern "C" __global__ __launch_bounds__(256) void advance_frontier(
    const INDEX_T* __restrict__ b, const int* __restrict__ children, const int* __restrict__ leaves,
    const double* __restrict__ qb, const int* __restrict__ qids, const double* __restrict__ best,
    int* stacks, int* depths, int* left, int* right, bool* active,
    int nq, int total, int first_leaf, int candidate_budget) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nq) return;
    int stack[128], sp = depths[q];
    for (int i = 0; i < sp; ++i) stack[i] = stacks[i * nq + q];
    for (int i = 0; i < SLOTS; ++i) {
        left[i * nq + q] = qids[q]; right[i * nq + q] = 0;
        active[i * nq + q] = false;
    }
    // Padding broadens pruning only. Exact tie membership uses d == best.
    double scale = fabs(qb[q]) + fabs(qb[nq + q]) + fabs(qb[2*nq + q]) + fabs(qb[3*nq + q]);
    double limit = best[q] + 64. * DBL_EPSILON * (scale + best[q] + 1.);
    double limit2 = __dmul_ru(limit, limit);
    int count = 0;
    while (sp && count < candidate_budget) {
        int node = stack[--sp];
        if (envelope_lower(b, node, total, qb, q, nq) > limit2) continue;
        if (node < first_leaf) {
            int at = count++ * nq + q;
            right[at] = leaves[node]; active[at] = true;
        } else {
            int ordered[FANOUT], size = 0;
            double lower[FANOUT];
            for (int c = 0; c < FANOUT; ++c) {
                int child = children[(node - first_leaf) * FANOUT + c];
                if (child < 0) continue;
                double bound = envelope_lower(b, child, total, qb, q, nq);
                if (bound > limit2) continue;
                int pos = size++;
                while (pos && lower[pos - 1] < bound) {
                    lower[pos] = lower[pos - 1]; ordered[pos] = ordered[pos - 1]; --pos;
                }
                lower[pos] = bound; ordered[pos] = child;
            }
            for (int i = 0; i < size; ++i) stack[sp++] = ordered[i];
        }
    }
    for (int i = 0; i < sp; ++i) stacks[i * nq + q] = stack[i];
    depths[q] = sp;
}
extern "C" __global__ __launch_bounds__(256) void consume_distances(
    const double* __restrict__ distance, const bool* __restrict__ active, const int* __restrict__ right, const int* __restrict__ qids,
    double* best, long long* counts, const long long* __restrict__ offsets,
    int* out_q, int* out_t, double* out_d, int nq, int phase, int all_matches) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nq) return;
    if (phase == 0) {
        double value = best[q];
        for (int i = 0; i < SLOTS; ++i) if (active[i*nq+q]) value = fmin(value, distance[i*nq+q]);
        best[q] = value;
        return;
    }
    long long count = counts[q];
    for (int i = 0; i < SLOTS; ++i) {
        int at = i * nq + q;
        if (active[at] && isfinite(distance[at]) && distance[at] == best[q]) {
            if (!all_matches && count) {
                if (phase == 2 && right[at] < out_t[offsets[q]]) out_t[offsets[q]] = right[at];
                continue;
            }
            if (phase == 2) {
                long long output = offsets[q] + count;
                out_q[output] = qids[q]; out_t[output] = right[at]; out_d[output] = distance[at];
            }
            ++count;
        }
    }
    counts[q] = count;
}
"""
