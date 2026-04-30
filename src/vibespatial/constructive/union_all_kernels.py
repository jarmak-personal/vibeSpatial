"""CUDA sources for union-all admission and planning helpers."""

BBOX_INTERACTION_KERNEL_NAMES = ("bbox_any_overlap",)

BBOX_INTERACTION_KERNEL_SOURCE = r"""
__device__ bool finite_double(double value) {
    return value == value
        && value <= 1.7976931348623157e308
        && value >= -1.7976931348623157e308;
}

extern "C" __global__
void bbox_any_overlap(
    const double* bounds,
    const bool* validity,
    int* result,
    int row_count
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= row_count || j >= row_count || i >= j) {
        return;
    }
    if (*result != 0 || !validity[i] || !validity[j]) {
        return;
    }

    const double ax0 = bounds[i * 4 + 0];
    const double ay0 = bounds[i * 4 + 1];
    const double ax1 = bounds[i * 4 + 2];
    const double ay1 = bounds[i * 4 + 3];
    const double bx0 = bounds[j * 4 + 0];
    const double by0 = bounds[j * 4 + 1];
    const double bx1 = bounds[j * 4 + 2];
    const double by1 = bounds[j * 4 + 3];
    if (
        !finite_double(ax0) || !finite_double(ay0) ||
        !finite_double(ax1) || !finite_double(ay1) ||
        !finite_double(bx0) || !finite_double(by0) ||
        !finite_double(bx1) || !finite_double(by1)
    ) {
        return;
    }
    if (ax0 <= bx1 && ax1 >= bx0 && ay0 <= by1 && ay1 >= by0) {
        atomicExch(result, 1);
    }
}
"""
