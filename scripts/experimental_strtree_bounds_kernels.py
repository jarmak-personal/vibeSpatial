"""Geometry-independent packed AABB construction shared by STR experiments."""

SOURCE = r"""
#ifndef INFINITY
#define INFINITY __longlong_as_double(0x7ff0000000000000LL)
#endif
extern "C" __global__ void compress_bounds(const double* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    output[i] = i < n / 2 ? __double2float_rd(input[i]) : __double2float_ru(input[i]);
}
extern "C" __global__ void packed_leaves(
    const double* ax, const double* ay, const double* bx, const double* by,
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
extern "C" __global__ void packed_parents(double* b, int first, int count, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int node = first + i;
    double lx = INFINITY, ly = INFINITY, hx = -INFINITY, hy = -INFINITY;
    #pragma unroll
    for (int c = 0; c < FANOUT; ++c) {
        int child = FANOUT * node + 1 + c;
        lx = fmin(lx, b[child]); ly = fmin(ly, b[total + child]);
        hx = fmax(hx, b[2 * total + child]); hy = fmax(hy, b[3 * total + child]);
    }
    b[node] = lx; b[total + node] = ly; b[2 * total + node] = hx; b[3 * total + node] = hy;
}
extern "C" __global__ void str_parents(double* b, const int* children, int first, int count, int total, int leaves) {
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
