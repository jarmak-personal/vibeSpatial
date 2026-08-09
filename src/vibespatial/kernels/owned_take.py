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
owned_take_gather_values_i32(
    const int* __restrict__ values,
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
        out[dst_start + j] = values[src_start + j];
    }
}

extern "C" __global__ void __launch_bounds__(256, 4)
owned_take_gather_values_f64x2(
    const double* __restrict__ values,
    const int* __restrict__ starts,
    const int* __restrict__ lengths,
    const int* __restrict__ out_offsets,
    double* __restrict__ out,
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
        const int src = (src_start + j) * 2;
        const int dst = (dst_start + j) * 2;
        out[dst] = values[src];
        out[dst + 1] = values[src + 1];
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

extern "C" __global__ void __launch_bounds__(256, 4)
owned_concat_compact_xy_f64(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const long long* __restrict__ active_counts,
    const long long* __restrict__ output_starts,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    const int buffer_index,
    const long long capacity
) {
    const long long active_count = active_counts[buffer_index];
    const long long output_start = output_starts[buffer_index];
    const long long stride = (long long)blockDim.x * (long long)gridDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < capacity;
         idx += stride) {
        if (idx < active_count) {
            const long long dst = output_start + idx;
            out_x[dst] = x[idx];
            out_y[dst] = y[idx];
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 4)
owned_concat_compact_offsets_i32(
    const int* __restrict__ offsets,
    const long long* __restrict__ active_interval_counts,
    const long long* __restrict__ output_interval_starts,
    const long long* __restrict__ value_starts,
    int* __restrict__ out_offsets,
    const int buffer_index,
    const long long interval_capacity
) {
    const long long active_count = active_interval_counts[buffer_index];
    const long long output_start = output_interval_starts[buffer_index];
    const long long value_start = value_starts[buffer_index];
    const long long stride = (long long)blockDim.x * (long long)gridDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx <= interval_capacity;
         idx += stride) {
        if (idx <= active_count) {
            out_offsets[output_start + idx] =
                (int)((long long)offsets[idx] + value_start);
        }
    }
}
"""

OWNED_TAKE_KERNEL_NAMES = (
    "owned_take_gather_index_ranges_i32",
    "owned_take_gather_xy_ranges_f64",
    "owned_take_gather_values_i32",
    "owned_take_gather_values_f64x2",
    "owned_take_gather_dense_xy_f64",
    "owned_concat_compact_xy_f64",
    "owned_concat_compact_offsets_i32",
)

request_nvrtc_warmup(
    [
        ("owned-take", _OWNED_TAKE_KERNEL_SOURCE, OWNED_TAKE_KERNEL_NAMES),
    ]
)
