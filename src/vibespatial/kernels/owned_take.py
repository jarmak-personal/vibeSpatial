"""NVRTC kernels for owned geometry device-side gather.

These kernels are data-movement primitives for owned geometry buffers.  They do
not perform coordinate arithmetic, so ADR-0002 precision dispatch does not
apply: coordinate storage remains canonical fp64 and the kernels only copy
separated x/y payload spans.
"""

from __future__ import annotations

from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup

_OWNED_TAKE_KERNEL_SOURCE = r"""
extern "C" __global__ void __launch_bounds__(256, 4)
owned_take_gather_index_ranges_i32(
    const int* __restrict__ starts,
    const int* __restrict__ lengths,
    const int* __restrict__ out_offsets,
    int* __restrict__ out,
    const int n_rows
) {
    const int row = blockIdx.x;
    if (row >= n_rows) {
        return;
    }

    const int src_start = starts[row];
    const int length = lengths[row];
    const int dst_start = out_offsets[row];
    for (int j = threadIdx.x; j < length; j += blockDim.x) {
        out[dst_start + j] = src_start + j;
    }
}

extern "C" __global__ void __launch_bounds__(256, 4)
owned_take_gather_xy_ranges_f64(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ starts,
    const int* __restrict__ lengths,
    const int* __restrict__ out_offsets,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    const int n_rows
) {
    const int row = blockIdx.x;
    if (row >= n_rows) {
        return;
    }

    const int src_start = starts[row];
    const int length = lengths[row];
    const int dst_start = out_offsets[row];
    for (int j = threadIdx.x; j < length; j += blockDim.x) {
        const int src = src_start + j;
        const int dst = dst_start + j;
        out_x[dst] = x[src];
        out_y[dst] = y[src];
    }
}

extern "C" __global__ void __launch_bounds__(256, 4)
owned_take_gather_dense_xy_f64(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const long long* __restrict__ rows,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    const int width,
    const int n_coords
) {
    const int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_coords;
         idx += stride) {
        const int row = idx / width;
        const int local = idx - row * width;
        const long long src_row = rows[row];
        const long long src = src_row * (long long)width + (long long)local;
        out_x[idx] = x[src];
        out_y[idx] = y[src];
    }
}
"""

OWNED_TAKE_KERNEL_NAMES = (
    "owned_take_gather_index_ranges_i32",
    "owned_take_gather_xy_ranges_f64",
    "owned_take_gather_dense_xy_f64",
)

request_nvrtc_warmup([
    ("owned-take", _OWNED_TAKE_KERNEL_SOURCE, OWNED_TAKE_KERNEL_NAMES),
])
