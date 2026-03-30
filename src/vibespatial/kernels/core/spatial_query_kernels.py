from __future__ import annotations

from vibespatial.cuda._runtime import get_cuda_runtime, make_kernel_cache_key
from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.kernels.core.spatial_query_source import (
    _GRID_NEAREST_KERNEL_NAMES,
    _GRID_NEAREST_KERNEL_SOURCE,
    _MORTON_RANGE_KERNEL_NAMES,
    _MORTON_RANGE_KERNEL_SOURCE,
    _SPATIAL_QUERY_KERNEL_NAMES,
    _SPATIAL_QUERY_KERNEL_SOURCE,
)

request_warmup([
    "exclusive_scan_i32", "exclusive_scan_i64",
    "select_i32", "select_i64",
    "radix_sort_i32_i32", "radix_sort_u64_i32",
    "merge_sort_u64_i32",
    "lower_bound_i32", "lower_bound_u64",
    "upper_bound_i32", "upper_bound_u64",
    "segmented_reduce_min_f64",
])

request_nvrtc_warmup([
    ("spatial-query", _SPATIAL_QUERY_KERNEL_SOURCE, _SPATIAL_QUERY_KERNEL_NAMES),
    ("morton-range", _MORTON_RANGE_KERNEL_SOURCE, _MORTON_RANGE_KERNEL_NAMES),
    ("grid-nearest", _GRID_NEAREST_KERNEL_SOURCE, _GRID_NEAREST_KERNEL_NAMES),
])


def _spatial_query_kernels():
    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key("spatial-query", _SPATIAL_QUERY_KERNEL_SOURCE)
    return runtime.compile_kernels(
        cache_key=cache_key,
        source=_SPATIAL_QUERY_KERNEL_SOURCE,
        kernel_names=_SPATIAL_QUERY_KERNEL_NAMES,
    )


def _morton_range_kernels():
    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key("morton-range", _MORTON_RANGE_KERNEL_SOURCE)
    return runtime.compile_kernels(
        cache_key=cache_key,
        source=_MORTON_RANGE_KERNEL_SOURCE,
        kernel_names=_MORTON_RANGE_KERNEL_NAMES,
    )


def _grid_nearest_kernels():
    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key("grid-nearest", _GRID_NEAREST_KERNEL_SOURCE)
    return runtime.compile_kernels(
        cache_key=cache_key,
        source=_GRID_NEAREST_KERNEL_SOURCE,
        kernel_names=_GRID_NEAREST_KERNEL_NAMES,
    )
