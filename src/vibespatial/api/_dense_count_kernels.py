"""NVRTC kernels for bounded updates of persistent dense count vectors."""

_DENSE_COUNT_UPDATE_KERNEL_NAMES = (
    "dense_count_preflight_u32",
    "dense_count_preflight_u64",
    "dense_count_update_u32",
    "dense_count_update_u64",
)

_DENSE_COUNT_UPDATE_KERNEL_SOURCE = r"""
extern "C" __global__ void __launch_bounds__(256, 4)
dense_count_preflight_u32(
    const long long* __restrict__ codes,
    const long long code_stride,
    const unsigned int* __restrict__ counts,
    const long long count_stride,
    const long long row_count,
    unsigned int* __restrict__ overflow_risk
) {
    const long long stride = (long long)blockDim.x * (long long)gridDim.x;
    const unsigned int safe_limit =
        0xffffffffu - (unsigned int)row_count;
    for (
        long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        row < row_count;
        row += stride
    ) {
        const long long code = codes[row * code_stride];
        if (counts[code * count_stride] > safe_limit) {
            atomicExch(overflow_risk, 1u);
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 4)
dense_count_preflight_u64(
    const long long* __restrict__ codes,
    const long long code_stride,
    const unsigned long long* __restrict__ counts,
    const long long count_stride,
    const long long row_count,
    unsigned int* __restrict__ overflow_risk
) {
    const long long stride = (long long)blockDim.x * (long long)gridDim.x;
    const unsigned long long safe_limit =
        0xffffffffffffffffull - (unsigned long long)row_count;
    for (
        long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        row < row_count;
        row += stride
    ) {
        const long long code = codes[row * code_stride];
        if (counts[code * count_stride] > safe_limit) {
            atomicExch(overflow_risk, 1u);
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 4)
dense_count_update_u32(
    const long long* __restrict__ codes,
    const long long code_stride,
    unsigned int* __restrict__ counts,
    const long long count_stride,
    const long long row_count,
    const unsigned int* __restrict__ overflow_risk
) {
    if (*overflow_risk != 0u) return;
    const long long stride = (long long)blockDim.x * (long long)gridDim.x;
    for (
        long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        row < row_count;
        row += stride
    ) {
        const long long code = codes[row * code_stride];
        atomicAdd(counts + code * count_stride, 1u);
    }
}

extern "C" __global__ void __launch_bounds__(256, 4)
dense_count_update_u64(
    const long long* __restrict__ codes,
    const long long code_stride,
    unsigned long long* __restrict__ counts,
    const long long count_stride,
    const long long row_count,
    const unsigned int* __restrict__ overflow_risk
) {
    if (*overflow_risk != 0u) return;
    const long long stride = (long long)blockDim.x * (long long)gridDim.x;
    for (
        long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        row < row_count;
        row += stride
    ) {
        const long long code = codes[row * code_stride];
        atomicAdd(counts + code * count_stride, 1ull);
    }
}
"""
