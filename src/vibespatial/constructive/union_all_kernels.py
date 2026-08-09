"""CUDA sources for union-all admission and planning helpers."""

BBOX_INTERACTION_KERNEL_NAMES = (
    "bbox_any_overlap_dense",
    "bbox_any_overlap_sorted",
)

BBOX_INTERACTION_KERNEL_SOURCE = r"""
extern "C" __global__
void bbox_any_overlap_dense(
    const double* __restrict__ xmin,
    const double* __restrict__ ymin,
    const double* __restrict__ xmax,
    const double* __restrict__ ymax,
    int* result,
    int row_count
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= row_count || j >= row_count || i >= j || *result != 0) {
        return;
    }
    if (
        xmin[i] <= xmax[j] && xmax[i] >= xmin[j] &&
        ymin[i] <= ymax[j] && ymax[i] >= ymin[j]
    ) {
        atomicExch(result, 1);
    }
}

extern "C" __global__
__launch_bounds__(256)
void bbox_any_overlap_sorted(
    const double* __restrict__ xmin,
    const double* __restrict__ ymin,
    const double* __restrict__ xmax,
    const double* __restrict__ ymax,
    int* result,
    int row_count
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= row_count || *result != 0) {
        return;
    }

    const double right = xmax[i];
    const double bottom = ymin[i];
    const double top = ymax[i];
    for (int j = i + 1; j < row_count; ++j) {
        if (xmin[j] > right || *result != 0) {
            return;
        }
        if (bottom <= ymax[j] && top >= ymin[j]) {
            atomicExch(result, 1);
            return;
        }
    }
}
"""
