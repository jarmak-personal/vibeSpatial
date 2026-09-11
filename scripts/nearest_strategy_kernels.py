"""Geometry kernels for isolated nearest strategy experiments (not dispatch)."""

from experiment_nearest_hierarchy import SOURCE as REFERENCE_SOURCE
from experimental_strtree_bounds_kernels import SOURCE as BOUNDS_SOURCE

# Keep the reference's exact metric arithmetic and conservative box filter.
COMMON = REFERENCE_SOURCE[:REFERENCE_SOURCE.index('extern "C" __global__ void traverse')]

SOURCE = BOUNDS_SOURCE + r"""// Conservative FP32 coarse distance: enclose the FP64 query coordinate,
// subtract toward -infinity, and round nonnegative squared terms downward.
// Together with outward-rounded boxes this cannot overstate box distance.
__device__ double box_distance2(const float* b, int node, int stride, double x, double y) {
    float xlo = __double2float_rd(x), xhi = __double2float_ru(x);
    float ylo = __double2float_rd(y), yhi = __double2float_ru(y);
    float dx = fmaxf(0.f, fmaxf(__fsub_rd(b[node], xhi), __fsub_rd(xlo, b[2 * stride + node])));
    float dy = fmaxf(0.f, fmaxf(__fsub_rd(b[stride + node], yhi), __fsub_rd(ylo, b[3 * stride + node])));
    return (double)__fadd_rd(__fmul_rd(dx, dx), __fmul_rd(dy, dy));
}
extern "C" __global__ void virtual_bounds(
    const double* ax, const double* ay, const double* bx, const double* by,
    const int* owner, const int* local, const int* counts,
    double* lx, double* ly, double* hx, double* hy, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int j = owner[i];
    double t0 = __ddiv_rd((double)local[i], (double)counts[j]);
    double t1 = __ddiv_ru((double)(local[i] + 1), (double)counts[j]);
    double dx0 = __dsub_rd(bx[j], ax[j]), dx1 = __dsub_ru(bx[j], ax[j]);
    double dy0 = __dsub_rd(by[j], ay[j]), dy1 = __dsub_ru(by[j], ay[j]);
    lx[i] = __dadd_rd(ax[j], __dmul_rd(bx[j] >= ax[j] ? t0 : t1, dx0));
    hx[i] = __dadd_ru(ax[j], __dmul_ru(bx[j] >= ax[j] ? t1 : t0, dx1));
    ly[i] = __dadd_rd(ay[j], __dmul_rd(by[j] >= ay[j] ? t0 : t1, dy0));
    hy[i] = __dadd_ru(ay[j], __dmul_ru(by[j] >= ay[j] ? t1 : t0, dy1));
}
// A seed is only an upper-bound hint; traversal still certifies every answer.
extern "C" __global__ void seed_cells(
    const double* ax, const double* ay, const double* bx, const double* by,
    const INDEX_T* bounds, int* cells, int n, int total, int resolution) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int root = RECURSIVE_STR ? total - 1 : 0;
    double x = .5 * (ax[i] + bx[i]), y = .5 * (ay[i] + by[i]);
    double sx = fmax((double)bounds[2 * total + root] - (double)bounds[root], 1.);
    double sy = fmax((double)bounds[3 * total + root] - (double)bounds[total + root], 1.);
    int ix = min(resolution - 1, max(0, (int)((x - bounds[root]) / sx * resolution)));
    int iy = min(resolution - 1, max(0, (int)((y - bounds[total + root]) / sy * resolution)));
    atomicMin(cells + iy * resolution + ix, i);
}
extern "C" __global__ void packed_traverse(
    const double* ax, const double* ay, const double* bx, const double* by,
    const int* ids, const INDEX_T* b, const double* qx, const double* qy, const int* qids,
    const int* cells, const int* child_map, double* bests, long long* counts, const long long* offsets,
    int* out_q, int* out_t, double* out_d, unsigned long long* visits,
    int n, int first_leaf, int total, int width, int nq, int emit, int resolution) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nq) return;
    double x = qx[q], y = qy[q], best = emit ? bests[q] : INFINITY;
    int root = RECURSIVE_STR ? total - 1 : 0;
    unsigned long long nodes = 0, segments = 0;
    if (!emit && resolution) {
        double sx = fmax((double)b[2 * total + root] - (double)b[root], 1.);
        double sy = fmax((double)b[3 * total + root] - (double)b[total + root], 1.);
        double cellx = (x - b[root]) / sx * resolution, celly = (y - b[total + root]) / sy * resolution;
        // Far/outside points simply retain the unseeded traversal.
        if (cellx >= 0. && cellx < resolution && celly >= 0. && celly < resolution) {
            int ix = (int)cellx, iy = (int)celly;
            for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
                int xx = ix + dx, yy = iy + dy;
                if (xx < 0 || xx >= resolution || yy < 0 || yy >= resolution) continue;
                int i = cells[yy * resolution + xx];
                if (i < n) { best = fmin(best, segment_distance(x, y, ax[i], ay[i], bx[i], by[i])); ++segments; }
            }
        }
    }
    int stack[128], sp = 0; stack[sp++] = root;
    long long matches = 0, output = emit ? offsets[q] : 0;
    while (sp) {
        int node = stack[--sp]; ++nodes;
        double limit = best + 64. * DBL_EPSILON * (fabs(x) + fabs(y) + best + 1.);
        double limit2 = limit * limit;
        if (box_distance2(b, node, total, x, y) > limit2) continue;
        if (RECURSIVE_STR ? node >= first_leaf : node < first_leaf) {
            int children[FANOUT], count = 0;
            double lower[FANOUT];
            #pragma unroll
            for (int c = 0; c < FANOUT; ++c) {
                int child = RECURSIVE_STR ? child_map[(node - first_leaf) * FANOUT + c] : FANOUT * node + 1 + c;
                if (child < 0) continue;
                double bound = box_distance2(b, child, total, x, y);
                if (bound <= limit2) {
                    int pos = count++;
                    while (pos > 0 && lower[pos - 1] < bound) {
                        lower[pos] = lower[pos - 1]; children[pos] = children[pos - 1]; --pos;
                    }
                    lower[pos] = bound; children[pos] = child;
                }
            }
            // Descending lower bounds put the nearest child at stack top.
            for (int c = 0; c < count; ++c) stack[sp++] = children[c];
        } else {
            int start = (RECURSIVE_STR ? node : node - first_leaf) * width;
            for (int j = start; j < min(n, start + width); ++j) {
                ++segments;
                double d = segment_distance(x, y, ax[j], ay[j], bx[j], by[j]);
                if (!emit && d < best) { best = d; matches = 0; }
                if (d == best) {
                    if (emit == 1) { out_q[output] = qids[q]; out_t[output] = ids[j]; out_d[output++] = d; }
                    ++matches;
                }
            }
        }
    }
    if (!emit) { bests[q] = best; counts[q] = matches; visits[2 * q] = nodes; visits[2 * q + 1] = segments; }
    if (emit == 2) counts[q] = matches;
}
"""
