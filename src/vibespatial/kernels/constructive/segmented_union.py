"""Segmented union_all kernel: native grouped polygon union.

ADR-0002: CONSTRUCTIVE class -- fp64 by design on all devices.
ADR-0033: Tier classification -- delegates to overlay pipeline (Tier 1 NVRTC
          + Tier 3a CCCL + Tier 2 CuPy) via overlay_union_owned.
ADR-0034: Inherits overlay pipeline precompilation; no new NVRTC source.

Algorithm
---------
CSR offsets are lowered once to compact device grouped metadata. Exact
constructive work then runs over all live groups together through grouped
overlay or group-local pairwise rounds; empty groups are restored by device
scatter. No Python loop dispatches geometry work group by group.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from vibespatial.constructive import segmented_union_cpu as _segmented_union_cpu_module
from vibespatial.constructive.segmented_union_host import (
    group_has_only_polygon_families,
    normalize_group_offsets,
    singleton_indices,
)
from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_I64,
    KERNEL_PARAM_PTR,
    get_cuda_runtime,
    make_kernel_cache_key,
)
from vibespatial.cuda.cccl_primitives import compact_indices
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import FAMILY_TAGS
from vibespatial.runtime import ExecutionMode, combined_residency
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.crossover import estimate_grouped_work_from_owned
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import (
    KernelClass,
    PrecisionMode,
    PrecisionPlan,
    normalize_precision_mode,
)
from vibespatial.runtime.residency import Residency, TransferTrigger
from vibespatial.runtime.robustness import select_robustness_plan

if TYPE_CHECKING:
    from vibespatial.geometry.owned import OwnedGeometryArray

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

_get_empty_owned = _segmented_union_cpu_module.get_empty_owned
_segmented_union_cpu = _segmented_union_cpu_module.segmented_union_cpu

@dataclass(frozen=True)
class _GroupedUnionCoverageFailure:
    failed_selection: Any
    failed_groups: Any
    source_rows: Any
    residuals: Any | None = None
    group_size_max: int | None = None
    source_segment_span_max: int | None = None

    def failed_source_rows_capacity(self):
        return self.failed_selection.gather_capacity(
            self.source_rows,
            fill_value=0,
        )


def _grouped_union_failure_from_mask(
    failed_mask: Any,
    source_rows: Any,
    *,
    output_row_count: int,
    residuals: Any | None = None,
    group_size_max: int | None = None,
    source_segment_span_max: int | None = None,
) -> _GroupedUnionCoverageFailure:
    """Build source-row and group-capacity selections from one failure mask."""
    from vibespatial.api._native_rowset import NativeDeviceSelection

    d_failed = cp.asarray(failed_mask, dtype=cp.bool_)
    d_source_rows = cp.asarray(source_rows, dtype=cp.int64)
    if d_failed.ndim != 1 or int(d_failed.size) != int(d_source_rows.size):
        raise ValueError("grouped union failure mask must align with source rows")
    failed_selection = NativeDeviceSelection.from_mask(d_failed)
    d_active = failed_selection.active_capacity_mask()
    d_failed_sources = failed_selection.gather_capacity(
        d_source_rows,
        fill_value=0,
    )
    capacity = failed_selection.capacity
    d_lanes = cp.arange(capacity, dtype=cp.int64)
    d_destinations = cp.where(
        d_active,
        d_failed_sources,
        cp.int64(output_row_count) + d_lanes,
    )
    d_group_mask_capacity = cp.zeros(
        output_row_count + capacity,
        dtype=cp.bool_,
    )
    d_group_mask_capacity[d_destinations] = True
    failed_groups = NativeDeviceSelection.from_mask(
        d_group_mask_capacity[:output_row_count],
    )
    if residuals is not None and int(residuals.row_count) != int(d_failed.size):
        raise ValueError("grouped union residual capacity must match source rows")
    return _GroupedUnionCoverageFailure(
        failed_selection=failed_selection,
        failed_groups=failed_groups,
        source_rows=d_source_rows,
        residuals=residuals,
        group_size_max=group_size_max,
        source_segment_span_max=source_segment_span_max,
    )


_SEGMENTED_UNION_ROBUST_SNAP_GRID = 1.0e-9
_SEGMENTED_UNION_ROBUST_SNAP_PRE_MAX_COORDS = 4096
_SEGMENTED_UNION_RECT_STRIP_MAX_GROUP_SIZE = 32


_RECTANGLE_STRIP_UNION_KERNEL_SOURCE = r"""
extern "C" __device__ __forceinline__ bool _vs_close(double a, double b) {
    const double diff = fabs(a - b);
    const double scale = fmax(fmax(fabs(a), fabs(b)), 1.0);
    return diff <= scale * 1.0e-12;
}

extern "C" __device__ __forceinline__ double _vs_endpoint(
    const double* __restrict__ bounds,
    const unsigned char* __restrict__ row_supported,
    long long start,
    int span_size,
    int endpoint_index,
    signed char horizontal
) {
    const int active_row = endpoint_index >> 1;
    int seen = 0;
    long long row = start;
    for (int i = 0; i < span_size; ++i) {
        if (row_supported[start + i] == 0) continue;
        if (seen == active_row) {
            row = start + i;
            break;
        }
        ++seen;
    }
    const int slot = endpoint_index & 1;
    const double* b = bounds + row * 4;
    if (horizontal) {
        return slot == 0 ? b[0] : b[2];
    }
    return slot == 0 ? b[1] : b[3];
}

extern "C" __device__ double _vs_kth_endpoint(
    const double* __restrict__ bounds,
    const unsigned char* __restrict__ row_supported,
    long long start,
    int span_size,
    int endpoint_count,
    int kth,
    signed char horizontal
) {
    double selected = 0.0;
    for (int rank = 0; rank <= kth; ++rank) {
        bool found = false;
        double next = 0.0;
        for (int i = 0; i < endpoint_count; ++i) {
            const double value = _vs_endpoint(
                bounds, row_supported, start, span_size, i, horizontal
            );
            if (rank > 0 && (value < selected || _vs_close(value, selected))) {
                continue;
            }
            if (!found || value < next) {
                next = value;
                found = true;
            }
        }
        if (!found) return selected;
        selected = next;
    }
    return selected;
}

extern "C" __global__ void __launch_bounds__(256, 4) validate_rectangle_strip_groups(
    const unsigned char* __restrict__ row_present,
    const unsigned char* __restrict__ row_supported,
    const double* __restrict__ bounds,
    const long long* __restrict__ group_offsets,
    int group_count,
    int max_group_size,
    unsigned char* __restrict__ supported,
    int* __restrict__ active_counts,
    int* __restrict__ endpoint_counts,
    signed char* __restrict__ orientation,
    double* __restrict__ out_bounds
) {
    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= group_count) {
        return;
    }

    const long long start = group_offsets[group];
    const long long end = group_offsets[group + 1];
    const int n = (int)(end - start);
    supported[group] = 0;
    active_counts[group] = 0;
    endpoint_counts[group] = 0;
    orientation[group] = -1;
    if (n <= 0 || n > max_group_size || n > 32) {
        return;
    }

    int first_row = -1;
    int active_count = 0;
    for (int i = 0; i < n; ++i) {
        if (row_present[start + i] != 0 && row_supported[start + i] == 0) {
            return;
        }
        if (row_supported[start + i] != 0) {
            if (first_row < 0) first_row = i;
            ++active_count;
        }
    }
    active_counts[group] = active_count;
    if (active_count == 0) {
        supported[group] = 1;
        double* out = out_bounds + group * 4;
        out[0] = nan("");
        out[1] = nan("");
        out[2] = nan("");
        out[3] = nan("");
        return;
    }

    const double* first = bounds + (start + first_row) * 4;
    const double x0 = first[0];
    const double y0 = first[1];
    const double x1 = first[2];
    const double y1 = first[3];
    bool same_y = true;
    bool same_x = true;
    double min_x = x0;
    double min_y = y0;
    double max_x = x1;
    double max_y = y1;

    for (int i = 0; i < n; ++i) {
        if (row_supported[start + i] == 0) continue;
        const double* b = bounds + (start + i) * 4;
        const double bx0 = b[0];
        const double by0 = b[1];
        const double bx1 = b[2];
        const double by1 = b[3];
        if (!(bx1 > bx0 && by1 > by0)) {
            return;
        }
        same_y = same_y && _vs_close(by0, y0) && _vs_close(by1, y1);
        same_x = same_x && _vs_close(bx0, x0) && _vs_close(bx1, x1);
        min_x = fmin(min_x, bx0);
        min_y = fmin(min_y, by0);
        max_x = fmax(max_x, bx1);
        max_y = fmax(max_y, by1);
    }
    if (!same_y && !same_x) {
        return;
    }
    const signed char horizontal = same_y ? 1 : 0;
    const int endpoint_count = active_count * 2;

    int unique_endpoint_count = 0;
    for (int i = 0; i < endpoint_count; ++i) {
        const double a = _vs_endpoint(
            bounds, row_supported, start, n, i, horizontal
        );
        bool seen = false;
        for (int j = 0; j < i; ++j) {
            if (_vs_close(a, _vs_endpoint(
                    bounds, row_supported, start, n, j, horizontal
                ))) {
                seen = true;
                break;
            }
        }
        if (!seen) ++unique_endpoint_count;
    }
    endpoint_counts[group] = unique_endpoint_count;

    bool used[32];
    for (int i = 0; i < 32; ++i) {
        used[i] = false;
    }
    int first_interval = -1;
    double first_lower = 0.0;
    for (int i = 0; i < n; ++i) {
        if (row_supported[start + i] == 0) continue;
        const double* b = bounds + (start + i) * 4;
        const double lower = horizontal ? b[0] : b[1];
        if (first_interval < 0 || lower < first_lower) {
            first_interval = i;
            first_lower = lower;
        }
    }
    if (first_interval < 0) {
        return;
    }
    used[first_interval] = true;
    const double* first_interval_bounds = bounds + (start + first_interval) * 4;
    double covered_end = horizontal ? first_interval_bounds[2] : first_interval_bounds[3];
    for (int step = 1; step < active_count; ++step) {
        int next_interval = -1;
        double next_lower = 0.0;
        for (int i = 0; i < n; ++i) {
            if (row_supported[start + i] == 0 || used[i]) {
                continue;
            }
            const double* b = bounds + (start + i) * 4;
            const double lower = horizontal ? b[0] : b[1];
            if (next_interval < 0 || lower < next_lower) {
                next_interval = i;
                next_lower = lower;
            }
        }
        if (next_interval < 0 || !(next_lower < covered_end)) {
            return;
        }
        used[next_interval] = true;
        const double* b = bounds + (start + next_interval) * 4;
        const double upper = horizontal ? b[2] : b[3];
        covered_end = fmax(covered_end, upper);
    }

    supported[group] = 1;
    orientation[group] = horizontal ? 1 : 0;
    double* out = out_bounds + group * 4;
    out[0] = min_x;
    out[1] = min_y;
    out[2] = max_x;
    out[3] = max_y;
}

extern "C" __global__ void __launch_bounds__(256, 4) emit_rectangle_strip_union(
    const unsigned char* __restrict__ row_supported,
    const double* __restrict__ bounds,
    const long long* __restrict__ group_offsets,
    const int* __restrict__ active_counts,
    const int* __restrict__ endpoint_counts,
    const int* __restrict__ coordinate_offsets,
    const signed char* __restrict__ orientation,
    int group_count,
    long long coordinate_capacity,
    double* __restrict__ out_x,
    double* __restrict__ out_y
) {
    const long long pos = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= coordinate_capacity || pos >= coordinate_offsets[group_count]) {
        return;
    }

    int lo = 0;
    int hi = group_count;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if ((long long)coordinate_offsets[mid + 1] <= pos) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    const int group = lo;
    if (group >= group_count) return;
    const int local = (int)(pos - (long long)coordinate_offsets[group]);
    const long long start = group_offsets[group];
    const int span_size = (int)(group_offsets[group + 1] - start);
    const int raw_endpoint_count = active_counts[group] * 2;
    const int endpoint_count = endpoint_counts[group];
    const signed char horizontal = orientation[group];
    long long first_row = start;
    for (int i = 0; i < span_size; ++i) {
        if (row_supported[start + i] != 0) {
            first_row = start + i;
            break;
        }
    }
    const double* first = bounds + first_row * 4;

    if (horizontal) {
        const double low_y = first[1];
        const double high_y = first[3];
        int kth = 0;
        double y = low_y;
        if (local == 0) {
            kth = 0;
            y = low_y;
        } else if (local <= endpoint_count) {
            kth = local - 1;
            y = high_y;
        } else {
            kth = (endpoint_count * 2) - local;
            y = low_y;
        }
        out_x[pos] = _vs_kth_endpoint(
            bounds, row_supported, start, span_size, raw_endpoint_count, kth, horizontal
        );
        out_y[pos] = y;
    } else {
        const double low_x = first[0];
        const double high_x = first[2];
        int kth = 0;
        double x = low_x;
        if (local < endpoint_count) {
            kth = local;
            x = low_x;
        } else if (local == endpoint_count) {
            kth = endpoint_count - 1;
            x = high_x;
        } else if (local < endpoint_count * 2) {
            kth = (endpoint_count * 2) - local;
            x = high_x;
        } else {
            kth = 0;
            x = low_x;
        }
        out_x[pos] = x;
        out_y[pos] = _vs_kth_endpoint(
            bounds, row_supported, start, span_size, raw_endpoint_count, kth, horizontal
        );
    }
}
"""


def _empty_group_owned_like(source: OwnedGeometryArray) -> OwnedGeometryArray:
    empty = _get_empty_owned().take(singleton_indices(0))
    if source.residency is Residency.DEVICE and cp is not None:
        empty.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="segmented union empty group matches device-resident input",
        )
    return empty


def _robust_snap_segmented_union_inputs_gpu(
    geometries: OwnedGeometryArray,
    group_offsets: np.ndarray,
    *,
    record: bool,
) -> OwnedGeometryArray | None:
    """Snap grouped-union inputs to a sub-nanometer device grid.

    Grouped dissolves often feed overlay results whose shared seams differ by
    floating-point dust after several constructive stages.  Snap-rounding the
    inputs before reduction closes only those sub-grid seams; the subsequent
    union still does the topology work.
    """
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None
    if geometries.row_count == 0:
        return geometries
    if int(np.diff(group_offsets).max(initial=0)) <= 1:
        return geometries

    from vibespatial.constructive.set_precision import _set_precision_gpu

    try:
        snapped = _set_precision_gpu(
            geometries,
            _SEGMENTED_UNION_ROBUST_SNAP_GRID,
            "pointwise",
        )
    except Exception:
        raise

    if record:
        record_dispatch_event(
            surface="segmented_union_all",
            operation="segmented_union_all_precision_snap",
            implementation="gpu_cupy_pointwise_snap",
            reason="robust grouped union seam snap",
            detail=(f"rows={geometries.row_count}, grid_size={_SEGMENTED_UNION_ROBUST_SNAP_GRID}"),
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
        )
    return snapped


def _owned_coordinate_count(owned: OwnedGeometryArray) -> int:
    if owned.device_state is not None:
        return sum(int(buf.x.size) for buf in owned.device_state.families.values())
    return sum(int(buf.x.size) for buf in owned.families.values())


def _should_pre_snap_segmented_union_inputs(
    geometries: OwnedGeometryArray,
    group_offsets: np.ndarray,
) -> bool:
    if geometries.row_count == 0:
        return False
    if int(np.diff(group_offsets).max(initial=0)) <= 1:
        return False
    return _owned_coordinate_count(geometries) <= _SEGMENTED_UNION_ROBUST_SNAP_PRE_MAX_COORDS


def _segmented_union_work_estimate(
    geometries: OwnedGeometryArray,
    *,
    n_groups: int,
):
    return estimate_grouped_work_from_owned(
        geometries,
        group_count=n_groups,
        output_row_count=n_groups,
        primary_unit_name="segmented-union-coordinate",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def segmented_union_all(
    geometries: OwnedGeometryArray,
    group_offsets: Any,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """Union all geometries within each group.  Returns one geometry per group.

    Parameters
    ----------
    geometries : OwnedGeometryArray
        Input polygons (device- or host-resident).
    group_offsets : array-like
        CSR-style int32/int64 offsets.  Group *i* contains
        ``geometries[group_offsets[i]:group_offsets[i+1]]``.
        Length is ``n_groups + 1``.
    dispatch_mode : ExecutionMode or str
        Execution mode hint (AUTO, GPU, CPU).
    precision : PrecisionMode or str
        Precision mode.  CONSTRUCTIVE kernels stay fp64 per ADR-0002.

    Returns
    -------
    OwnedGeometryArray
        One geometry per group.  May contain MultiPolygon when union
        produces disconnected regions.  Empty groups produce empty Polygon.
    """
    from vibespatial.geometry.owned import from_shapely_geometries

    requested = (
        dispatch_mode if isinstance(dispatch_mode, ExecutionMode) else ExecutionMode(dispatch_mode)
    )
    precision_mode = normalize_precision_mode(precision)

    group_offsets = normalize_group_offsets(group_offsets)
    if group_offsets.ndim != 1:
        raise ValueError("group_offsets must be one-dimensional")
    n_groups = len(group_offsets) - 1
    if n_groups < 0:
        raise ValueError("group_offsets must have length >= 1")
    if int(group_offsets[0]) != 0:
        raise ValueError("group_offsets must start at zero")
    if np.any(group_offsets[1:] < group_offsets[:-1]):
        raise ValueError("group_offsets must be nondecreasing")
    if int(group_offsets[-1]) != geometries.row_count:
        raise ValueError("final group offset must match geometry row count")
    if n_groups == 0:
        return from_shapely_geometries([])

    total_geoms = int(group_offsets[-1])
    work_estimate = _segmented_union_work_estimate(
        geometries,
        n_groups=n_groups,
    )

    # Dispatch selection
    selection = plan_dispatch_selection(
        kernel_name="segmented_union_all",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=total_geoms,
        requested_mode=requested,
        requested_precision=precision_mode,
        current_residency=combined_residency(geometries),
        work_estimate=work_estimate,
    )

    # ADR-0002: CONSTRUCTIVE kernels stay fp64.  Precision plan is computed
    # for observability (dispatch event detail) only.
    precision_plan = selection.precision_plan
    select_robustness_plan(
        kernel_class=KernelClass.CONSTRUCTIVE,
        precision_plan=precision_plan,
    )

    if selection.selected is ExecutionMode.GPU:
        result = _segmented_union_gpu(
            geometries,
            group_offsets,
            n_groups=n_groups,
            precision_plan=precision_plan,
        )
        if result is not None:
            record_dispatch_event(
                surface="segmented_union_all",
                operation="segmented_union_all",
                implementation="gpu_native_grouped_constructive",
                reason=selection.reason,
                detail=(
                    f"groups={n_groups}, total_geoms={total_geoms}, "
                    f"precision={precision_plan.compute_precision.value}, "
                    f"{work_estimate.telemetry_detail()}"
                ),
                requested=selection.requested,
                selected=ExecutionMode.GPU,
            )
            result.record_runtime_selection(selection)
            return result

    # CPU fallback
    result = _segmented_union_cpu(geometries, group_offsets, n_groups=n_groups)
    record_dispatch_event(
        surface="segmented_union_all",
        operation="segmented_union_all",
        implementation="shapely_union_all",
        reason=selection.reason
        if selection.selected is ExecutionMode.CPU
        else "GPU fallback to CPU",
        detail=f"groups={n_groups}, total_geoms={total_geoms}",
        requested=selection.requested,
        selected=ExecutionMode.CPU,
    )
    result.record_runtime_selection(selection)
    return result


# ---------------------------------------------------------------------------
# GPU variant: grouped overlay topology over compact group metadata
# ---------------------------------------------------------------------------


@register_kernel_variant(
    "segmented_union_all",
    "gpu-native-grouped-constructive",
    kernel_class=KernelClass.CONSTRUCTIVE,
    geometry_families=("polygon", "multipolygon"),
    execution_modes=(ExecutionMode.GPU,),
    supports_mixed=True,
    precision_modes=(PrecisionMode.AUTO, PrecisionMode.FP64),
    tags=("constructive", "segmented-union", "gpu", "native-grouped"),
)
def _segmented_union_gpu_variant(
    geometries: OwnedGeometryArray,
    group_offsets: Any,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.GPU,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """GPU variant: grouped overlay topology over all live groups."""
    group_offsets = normalize_group_offsets(group_offsets)
    n_groups = len(group_offsets) - 1
    precision_mode = normalize_precision_mode(precision)
    work_estimate = _segmented_union_work_estimate(
        geometries,
        n_groups=n_groups,
    )
    selection = plan_dispatch_selection(
        kernel_name="segmented_union_all",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=int(group_offsets[-1]),
        requested_mode=dispatch_mode,
        requested_precision=precision_mode,
        current_residency=combined_residency(geometries),
        work_estimate=work_estimate,
    )
    precision_plan = selection.precision_plan
    return _segmented_union_gpu(
        geometries,
        group_offsets,
        n_groups=n_groups,
        precision_plan=precision_plan,
    )


def _segmented_union_gpu(
    geometries: OwnedGeometryArray,
    group_offsets,
    *,
    n_groups: int,
    precision_plan: PrecisionPlan,
) -> OwnedGeometryArray | None:
    """Union polygon geometries through one admitted grouped device plan.

    Unsupported non-polygon families decline before constructive execution.
    Once admitted, grouped topology and repair complete atomically or raise.

    ADR-0002: CONSTRUCTIVE class, fp64 (segment intersection precision).
    ADR-0033: Inherits overlay pipeline tiers (NVRTC + CCCL + CuPy).
    """
    # Validate: GPU overlay requires polygon-family geometries.
    polygon_tags = {FAMILY_TAGS[GeometryFamily.POLYGON], FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]}
    if not group_has_only_polygon_families(geometries, polygon_tags):
        # Non-polygon geometry present: fall back to CPU.
        return None

    working_geometries = geometries
    if _should_pre_snap_segmented_union_inputs(geometries, group_offsets):
        snapped = _robust_snap_segmented_union_inputs_gpu(
            geometries,
            group_offsets,
            record=True,
        )
        if snapped is not None:
            working_geometries = snapped

    result = _segmented_union_gpu_impl(
        working_geometries,
        group_offsets,
        n_groups=n_groups,
        precision_plan=precision_plan,
    )
    return result


def _segmented_union_gpu_impl(
    geometries: OwnedGeometryArray,
    group_offsets: np.ndarray,
    *,
    n_groups: int,
    precision_plan: PrecisionPlan,
) -> OwnedGeometryArray | None:
    if cp is None:  # pragma: no cover - GPU dispatch requires CuPy
        return None

    from vibespatial.geometry.owned import build_empty_polygon_rows_device

    group_sizes = np.diff(group_offsets)
    observed_group_ids = np.flatnonzero(group_sizes > 0).astype(np.int64, copy=False)
    empty_output = build_empty_polygon_rows_device(n_groups)
    if observed_group_ids.size == 0:
        return empty_output

    observed_sizes = group_sizes[observed_group_ids].astype(np.int64, copy=False)
    compact_offsets = np.concatenate(
        [
            np.asarray([0], dtype=np.int64),
            np.cumsum(observed_sizes, dtype=np.int64),
        ]
    )
    result = segmented_union_all_device_grouped(
        geometries,
        cp.asarray(compact_offsets, dtype=cp.int64),
        cp.asarray(observed_group_ids, dtype=cp.int64),
        output_row_count=n_groups,
        precision_plan=precision_plan,
        empty_output=empty_output,
        all_groups_observed=observed_group_ids.size == n_groups,
        group_size_min=int(observed_sizes.min()),
        group_size_max=int(observed_sizes.max()),
    )
    if result is not None:
        from vibespatial.geometry.owned import seed_all_validity_cache

        seed_all_validity_cache(result)
        record_dispatch_event(
            surface="segmented_union_all",
            operation="segmented_union_strategy",
            implementation="gpu_native_grouped_constructive_carrier",
            reason="all multi-group constructive work uses compact device grouped metadata",
            detail=(
                f"groups={n_groups}, observed_groups={observed_group_ids.size}, "
                f"max_group_size={int(observed_sizes.max())}, "
                f"total_geoms={geometries.row_count}"
            ),
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
        )
    return result


def _rectangle_strip_union_kernels():
    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key(
        "segmented-union-rectangle-strip",
        _RECTANGLE_STRIP_UNION_KERNEL_SOURCE,
    )
    return runtime.compile_kernels(
        cache_key=cache_key,
        source=_RECTANGLE_STRIP_UNION_KERNEL_SOURCE,
        kernel_names=(
            "validate_rectangle_strip_groups",
            "emit_rectangle_strip_union",
        ),
    )


def _all_rows_present_from_existing_proof(geometries: OwnedGeometryArray) -> bool:
    """Return whether row validity is structurally known without a device read."""
    if geometries.row_count == 0:
        return True
    validity = getattr(geometries, "_validity", None)
    if validity is not None and int(validity.size) == int(geometries.row_count):
        return bool(np.all(validity))
    if geometries.is_indexed_view and geometries._base is not None:
        base_validity = getattr(geometries._base, "_validity", None)
        if (
            base_validity is not None
            and int(base_validity.size) == int(geometries._base.row_count)
            and bool(np.all(base_validity))
        ):
            return True
    return False


def _device_row_activity_view(
    geometries: OwnedGeometryArray,
    active_mask: Any,
) -> OwnedGeometryArray:
    """Return a non-mutating, row-indirected view of active source capacity."""
    if cp is None:  # pragma: no cover - CPU-only installs do not call this path
        raise RuntimeError("CuPy is required for device row activity")
    d_identity = cp.arange(geometries.row_count, dtype=cp.int64)
    return geometries._device_indexed_take(
        d_identity,
        assume_unique_indices=True,
    )._apply_row_activity(
        cp.asarray(active_mask, dtype=cp.bool_),
        assume_active_indices_unique=True,
    )


def _grouped_rectangle_strip_union_device(
    geometries: OwnedGeometryArray,
    group_offsets: Any,
    group_ids: Any,
    *,
    output_row_count: int,
    empty_output: OwnedGeometryArray,
    all_groups_observed: bool | None = None,
    group_size_min: int | None = None,
    group_size_max: int | None = None,
) -> OwnedGeometryArray | None:
    """Emit exact grouped unions for dense same-span rectangle strips.

    Physical shape: `NativeGrouped` sorted offsets, device row activity, and
    axis-aligned rectangle polygon buffers -> one output-byte shaped kernel
    that writes the exact collinear boundary breakpoints for each grouped
    strip. Unsupported groups remain inactive for the generic grouped carrier.
    """
    if cp is None:  # pragma: no cover - CPU-only installs do not call this path
        return None
    if (
        all_groups_observed is not True
        or group_size_max is None
        or int(group_size_max) <= 1
        or int(group_size_max) > _SEGMENTED_UNION_RECT_STRIP_MAX_GROUP_SIZE
    ):
        return None
    d_group_offsets = cp.asarray(group_offsets, dtype=cp.int64)
    d_group_ids = cp.asarray(group_ids, dtype=cp.int64)
    compact_group_count = int(d_group_ids.size)
    if compact_group_count == 0:
        return empty_output
    if compact_group_count != int(output_row_count):
        return None
    if int(d_group_offsets.size) != compact_group_count + 1:
        return None

    state = geometries._ensure_device_state(preserve_indexed_view=True)
    if GeometryFamily.POLYGON not in state.families:
        return None
    polygon_buffer = state.families[GeometryFamily.POLYGON]
    fixed_size = getattr(polygon_buffer, "fixed_size", None)
    bounded_rectangle_rows = bool(
        fixed_size is not None
        and int(getattr(fixed_size, "max_first_level_count_per_row", 0) or 0) <= 1
        and int(getattr(fixed_size, "max_coord_count_per_row", 0) or 0) <= 5
    )
    if (
        int(getattr(polygon_buffer, "dense_single_ring_width", 0) or 0) != 5
        and not bounded_rectangle_rows
    ) or not bool(getattr(polygon_buffer, "axis_aligned_rectangles", False)):
        return None
    from vibespatial.geometry.owned import device_valid_nonempty_mask

    d_row_present = device_valid_nonempty_mask(geometries).astype(
        cp.uint8,
        copy=False,
    )
    d_tags = cp.asarray(state.tags, dtype=cp.int8)
    d_family_rows = cp.asarray(state.family_row_offsets, dtype=cp.int64)
    polygon_family_capacity = max(
        int(polygon_buffer.geometry_offsets.size) - 1,
        0,
    )
    d_row_supported = (
        (d_row_present != 0)
        & (d_tags == cp.int8(FAMILY_TAGS[GeometryFamily.POLYGON]))
        & (d_family_rows >= 0)
        & (d_family_rows < polygon_family_capacity)
    ).astype(cp.uint8, copy=False)

    bounds = polygon_buffer.bounds
    if (
        bounds is None
        or geometries.is_indexed_view
        or int(getattr(bounds, "shape", (0,))[0]) != geometries.row_count
    ):
        from vibespatial.kernels.core.geometry_analysis import (
            compute_geometry_bounds_device,
        )

        bounds = compute_geometry_bounds_device(
            geometries,
            preserve_indexed_view=True,
        )
    d_bounds = cp.asarray(bounds, dtype=cp.float64).reshape(geometries.row_count, 4)

    d_supported = cp.zeros(compact_group_count, dtype=cp.bool_)
    d_active_counts = cp.zeros(compact_group_count, dtype=cp.int32)
    d_endpoint_counts = cp.zeros(compact_group_count, dtype=cp.int32)
    d_orientation = cp.full(compact_group_count, -1, dtype=cp.int8)
    d_out_bounds = cp.empty((compact_group_count, 4), dtype=cp.float64)
    kernels = _rectangle_strip_union_kernels()
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    validate = kernels["validate_rectangle_strip_groups"]
    grid, block = runtime.launch_config(validate, compact_group_count)
    runtime.launch(
        validate,
        grid=grid,
        block=block,
        params=(
            (
                ptr(d_row_present),
                ptr(d_row_supported),
                ptr(d_bounds),
                ptr(d_group_offsets),
                compact_group_count,
                int(group_size_max),
                ptr(d_supported),
                ptr(d_active_counts),
                ptr(d_endpoint_counts),
                ptr(d_orientation),
                ptr(d_out_bounds),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
            ),
        ),
    )

    d_has_rows = d_supported & (d_active_counts > 0)
    d_coordinate_counts = cp.where(
        d_has_rows,
        (d_endpoint_counts * np.int32(2)) + np.int32(1),
        np.int32(0),
    )
    d_coordinate_offsets = cp.empty(compact_group_count + 1, dtype=cp.int32)
    d_coordinate_offsets[0] = 0
    cp.cumsum(d_coordinate_counts, out=d_coordinate_offsets[1:])
    coordinate_capacity = int(geometries.row_count * 4 + compact_group_count)
    d_x = cp.empty(coordinate_capacity, dtype=cp.float64)
    d_y = cp.empty(coordinate_capacity, dtype=cp.float64)

    emit = kernels["emit_rectangle_strip_union"]
    grid, block = runtime.launch_config(emit, coordinate_capacity)
    runtime.launch(
        emit,
        grid=grid,
        block=block,
        params=(
            (
                ptr(d_row_supported),
                ptr(d_bounds),
                ptr(d_group_offsets),
                ptr(d_active_counts),
                ptr(d_endpoint_counts),
                ptr(d_coordinate_offsets),
                ptr(d_orientation),
                compact_group_count,
                coordinate_capacity,
                ptr(d_x),
                ptr(d_y),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I64,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
            ),
        ),
    )

    from vibespatial.geometry.buffers import get_geometry_buffer_schema
    from vibespatial.geometry.owned import (
        DeviceFamilyGeometryBuffer,
        FamilyGeometryBuffer,
        build_device_resident_owned,
    )

    d_geometry_offsets = cp.arange(compact_group_count + 1, dtype=cp.int32)
    d_empty_mask = ~d_has_rows
    d_validity = cp.ones(compact_group_count, dtype=cp.bool_)
    d_tags = cp.full(
        compact_group_count,
        FAMILY_TAGS[GeometryFamily.POLYGON],
        dtype=cp.int8,
    )
    d_family_row_offsets = cp.arange(compact_group_count, dtype=cp.int32)
    result = build_device_resident_owned(
        device_families={
            GeometryFamily.POLYGON: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.POLYGON,
                x=d_x,
                y=d_y,
                geometry_offsets=d_geometry_offsets,
                empty_mask=d_empty_mask,
                ring_offsets=d_coordinate_offsets,
                bounds=d_out_bounds,
                axis_aligned_rectangles=True,
            )
        },
        row_count=compact_group_count,
        tags=d_tags,
        validity=d_validity,
        family_row_offsets=d_family_row_offsets,
        execution_mode="gpu",
    )
    result.families[GeometryFamily.POLYGON] = FamilyGeometryBuffer(
        family=GeometryFamily.POLYGON,
        schema=get_geometry_buffer_schema(GeometryFamily.POLYGON),
        row_count=compact_group_count,
        x=np.empty(0, dtype=np.float64),
        y=np.empty(0, dtype=np.float64),
        geometry_offsets=np.empty(0, dtype=np.int32),
        empty_mask=np.empty(0, dtype=np.bool_),
        ring_offsets=None,
        bounds=None,
        host_materialized=False,
    )
    if result.device_state is not None:
        result.device_state.trusted_all_valid = True
        result.device_state.trusted_homogeneous_family = GeometryFamily.POLYGON
        result.device_state.trusted_all_non_empty = True
        result.device_state.row_bounds = d_out_bounds
    result._cached_is_valid_mask = np.ones(compact_group_count, dtype=bool)
    from vibespatial.geometry.owned import device_mask_owned_capacity

    if result.device_state is not None:
        result.device_state.trusted_all_valid = None
    result._cached_is_valid_mask = None
    result = device_mask_owned_capacity(result, d_supported)
    result._native_grouped_strip_group_mask = d_supported
    result._native_grouped_union_implementation = "native_grouped_rectangle_strip_union"
    return result


def _segmented_union_device_grouped_pairwise_tree(
    current: OwnedGeometryArray,
    d_current_offsets: Any,
    d_group_ids: Any,
    *,
    output_row_count: int,
    precision_plan: PrecisionPlan,
    empty_output: OwnedGeometryArray,
    all_groups_observed: bool | None = None,
    original_row_count: int | None = None,
    valid_count: int | None = None,
    allow_singleton_identity: bool = False,
    group_size_max: int | None = None,
) -> OwnedGeometryArray | None:
    """Exact grouped pairwise reduction over shrinking device capacities.

    Pair and odd-carry lanes share one device selection each round. Selected
    rows remain row-indirected over each round's constructive output while
    device group offsets shrink the live algebraic workload. Kernels consume
    those views directly; physicalizing a sparse or inactive partition would
    allocate from its root capacity rather than its live geometry.
    """
    from vibespatial.api._native_rowset import NativeDeviceSelection
    from vibespatial.constructive.binary_constructive import (
        _apply_binary_empty_row_semantics_gpu,
        _binary_constructive_gpu,
    )
    from vibespatial.geometry.owned import (
        concat_owned_scatter,
        device_select_owned_capacity_partitions,
        device_take_owned_capacity_selection,
    )

    d_current_offsets = cp.asarray(d_current_offsets, dtype=cp.int64)
    d_group_ids = cp.asarray(d_group_ids, dtype=cp.int64)
    live_group_count = int(d_group_ids.size)
    if int(d_current_offsets.size) != live_group_count + 1:
        return None
    capacity = int(current.row_count)
    if capacity == 0 or live_group_count == 0:
        return empty_output
    d_span_sizes = d_current_offsets[1:] - d_current_offsets[:-1]
    d_group_has_values = d_span_sizes > 0
    if (
        allow_singleton_identity
        and all_groups_observed is True
        and valid_count == original_row_count
        and capacity == output_row_count
        and live_group_count == output_row_count
    ):
        current._native_grouped_union_implementation = "native_grouped_singleton_identity_union"
        return current

    d_active_counts = d_span_sizes
    reduction_width = capacity if group_size_max is None else int(group_size_max)
    max_rounds = 0 if reduction_width <= 1 else int(math.ceil(math.log2(reduction_width)))

    for _round in range(max_rounds):
        d_positions = cp.arange(capacity, dtype=cp.int64)
        d_capacity_active = d_positions < d_current_offsets[-1]
        d_safe_positions = cp.where(d_capacity_active, d_positions, cp.int64(0))
        d_group_local = cp.searchsorted(
            d_current_offsets[1:],
            d_safe_positions,
            side="right",
        ).astype(cp.int64, copy=False)
        d_group_lane = d_safe_positions - d_current_offsets[d_group_local]
        d_pair_counts = d_active_counts // 2
        d_carry_counts = d_active_counts % 2
        d_pair_active = d_capacity_active & (d_group_lane < d_pair_counts[d_group_local])
        d_carry_active = d_capacity_active & (
            (d_carry_counts[d_group_local] != 0) & (d_group_lane == d_pair_counts[d_group_local])
        )
        d_pair_left = d_current_offsets[d_group_local] + (d_group_lane * 2)
        d_carry_source = d_current_offsets[d_group_local] + d_active_counts[d_group_local] - 1
        d_union_active = d_pair_active | d_carry_active
        d_left_source = cp.where(
            d_pair_active,
            d_pair_left,
            cp.where(d_carry_active, d_carry_source, cp.int64(0)),
        )
        left_rows = current._device_indexed_take(
            d_left_source,
            assume_unique_indices=False,
        )._apply_row_activity(
            d_union_active,
        )
        right_rows = current._device_indexed_take(
            cp.where(d_pair_active, d_pair_left + 1, cp.int64(0)),
            assume_unique_indices=False,
        )._apply_row_activity(
            d_pair_active,
        )
        pair_rows = _binary_constructive_gpu(
            "union",
            left_rows,
            right_rows,
            dispatch_mode=ExecutionMode.GPU,
        )
        if pair_rows is None:
            return None
        pair_rows = _apply_binary_empty_row_semantics_gpu(
            "union",
            left_rows,
            right_rows,
            pair_rows,
        )
        if pair_rows.row_count != capacity:
            return None
        # Odd members are algebraic carries, not union(null) operations. The
        # inactive right lane is intentionally null, so binary public null
        # semantics must not be asked to preserve it as though it were a valid
        # empty geometry.
        pair_rows = device_select_owned_capacity_partitions(
            pair_rows,
            [(left_rows, d_carry_active)],
        )

        full_selection = NativeDeviceSelection.from_mask(
            d_union_active,
            source_row_count=capacity,
        )
        next_capacity = max(
            live_group_count,
            min(
                capacity,
                (capacity + live_group_count + 1) // 2,
            ),
        )
        next_selection = NativeDeviceSelection(
            positions=cp.asarray(full_selection.positions, dtype=cp.int64)[:next_capacity],
            logical_count=full_selection.logical_count,
            source_token=full_selection.source_token,
            source_row_count=capacity,
            ordered=True,
            unique=True,
            full_selection_implies_identity=False,
        )
        current = device_take_owned_capacity_selection(
            pair_rows,
            next_selection,
        )
        d_active_counts = d_pair_counts + d_carry_counts
        d_current_offsets = cp.empty(live_group_count + 1, dtype=cp.int64)
        d_current_offsets[0] = 0
        cp.cumsum(d_active_counts, out=d_current_offsets[1:])
        capacity = next_capacity

    compact = current.device_take(
        cp.where(d_group_has_values, d_current_offsets[:-1], cp.int64(0)),
        allow_capacity_allocation=True,
        assume_unique_indices=True,
    )._apply_row_activity(d_group_has_values)
    if compact.row_count != live_group_count:
        return None
    if all_groups_observed is True and live_group_count == output_row_count:
        compact._native_grouped_union_implementation = (
            "native_grouped_device_fixed_span_pairwise_union"
        )
        return compact
    scattered = concat_owned_scatter(
        empty_output,
        compact,
        d_group_ids.astype(cp.int64, copy=False),
    )
    scattered._native_grouped_union_implementation = (
        "native_grouped_device_fixed_span_pairwise_union"
    )
    return scattered


def _grouped_union_constructive_coverage_failure_device(
    candidate: OwnedGeometryArray,
    inputs: OwnedGeometryArray,
    group_offsets: Any,
    group_ids: Any,
    *,
    output_row_count: int,
    stage: str,
    group_size_max: int | None,
    source_segment_span_max: int | None,
) -> _GroupedUnionCoverageFailure | None:
    """Prove grouped coverage with retained input-minus-candidate residuals.

    Physical shape: grouped input rows against bbox-related candidate polygon
    parts, followed by a device fp64 area classifier.  The exact residual is
    also the repair payload, so coverage admission and repair share one
    constructive pass instead of running a full row-aligned DE-9IM proof and
    then reconstructing the same missing geometry.

    Any finite positive area is a failure.  A scale-relative tolerance would
    silently discard valid GEOS-visible slivers, which violates grouped union
    semantics.  Valid zero-area remnants are ignored because the admitted
    source partition contains positive-area polygon rows and polygon union has
    a polygonal output contract.
    """
    if cp is None:  # pragma: no cover - CPU-only installs do not call this path
        return None
    if candidate.row_count != int(output_row_count):
        return None
    if inputs.row_count == 0:
        return _grouped_union_failure_from_mask(
            cp.empty(0, dtype=cp.bool_),
            cp.empty(0, dtype=cp.int64),
            output_row_count=output_row_count,
            group_size_max=group_size_max,
            source_segment_span_max=source_segment_span_max,
        )

    try:
        from vibespatial.constructive.measurement import _area_gpu_device_fp64

        d_group_offsets = cp.asarray(group_offsets, dtype=cp.int64)
        d_group_ids = cp.asarray(group_ids, dtype=cp.int64)
        if int(d_group_offsets.size) != int(d_group_ids.size) + 1:
            return None

        d_positions = cp.arange(inputs.row_count, dtype=cp.int64)
        d_compact_group_rows = cp.searchsorted(
            d_group_offsets[1:],
            d_positions,
            side="right",
        ).astype(cp.int64, copy=False)
        d_source_rows = d_group_ids[d_compact_group_rows].astype(
            cp.int64,
            copy=False,
        )
        if int(d_source_rows.size) != inputs.row_count:
            return None

        from vibespatial.geometry.owned import device_valid_nonempty_mask

        all_rows_failure = _grouped_union_failure_from_mask(
            device_valid_nonempty_mask(inputs),
            d_source_rows,
            output_row_count=output_row_count,
            group_size_max=group_size_max,
            source_segment_span_max=source_segment_span_max,
        )
        residuals = _failed_input_residuals_against_candidate_parts_gpu(
            candidate,
            inputs,
            all_rows_failure,
            dispatch_mode=ExecutionMode.GPU,
            record_event=False,
        )
        if residuals is None or residuals.row_count != inputs.row_count:
            return None

        d_raw_residual_area = cp.abs(cp.asarray(_area_gpu_device_fp64(residuals), dtype=cp.float64))
        d_residual_state = residuals._ensure_device_state(preserve_indexed_view=True)
        d_residual_validity = cp.asarray(
            d_residual_state.validity,
            dtype=cp.bool_,
        )[: inputs.row_count]
        d_residual_area = cp.nan_to_num(
            d_raw_residual_area,
            nan=0.0,
            posinf=cp.inf,
            neginf=cp.inf,
        )
        if int(d_residual_area.size) != inputs.row_count:
            return None
        d_unresolved_area = cp.isinf(d_raw_residual_area) | (
            cp.isnan(d_raw_residual_area) & d_residual_validity
        )
        d_failed_mask = (
            d_residual_validity & (d_residual_area > cp.float64(0.0))
        ) | d_unresolved_area
        return _grouped_union_failure_from_mask(
            d_failed_mask,
            d_source_rows,
            output_row_count=output_row_count,
            residuals=residuals,
            group_size_max=group_size_max,
            source_segment_span_max=source_segment_span_max,
        )
    except Exception:
        raise


def _grouped_union_coverage_failure_device(
    candidate: OwnedGeometryArray,
    inputs: OwnedGeometryArray,
    group_offsets: Any,
    group_ids: Any,
    *,
    output_row_count: int,
    stage: str,
    group_size_max: int | None,
    source_segment_span_max: int | None,
) -> _GroupedUnionCoverageFailure | None:
    """Return sparse device residuals where grouped union needs repair."""
    return _grouped_union_constructive_coverage_failure_device(
        candidate,
        inputs,
        group_offsets,
        group_ids,
        output_row_count=output_row_count,
        stage=stage,
        group_size_max=group_size_max,
        source_segment_span_max=source_segment_span_max,
    )


def _physicalize_polygon_relation_rows_device(
    polygons: OwnedGeometryArray,
    d_rows: Any,
    *,
    row_multiplicity_bound: int,
) -> OwnedGeometryArray:
    """Gather selected Polygon rows into exact device execution storage.

    Dynamic grouped topology commonly leaves polygon parts as row-indirected
    views over a much larger capacity buffer. Segment topology cannot size an
    indexed duplicate gather from that root capacity. This transition expands
    nested offsets with device ``repeat`` primitives, so allocation follows the
    selected rings and coordinates rather than ``selected_rows * root_coords``.
    """
    from vibespatial.geometry.owned import (
        DeviceFamilyGeometryBuffer,
        _device_gather_offset_index_ranges,
        _device_gather_xy_offset_slices,
        build_device_resident_owned,
        seed_all_validity_cache,
        seed_homogeneous_host_metadata,
    )

    d_rows = cp.asarray(d_rows, dtype=cp.int64)
    row_count = int(d_rows.size)
    if row_count == 0:
        from vibespatial.geometry.owned import build_empty_polygon_rows_device

        return build_empty_polygon_rows_device(0)

    state = polygons._ensure_device_state(preserve_indexed_view=True)
    polygon = state.families.get(GeometryFamily.POLYGON)
    if polygon is None or polygon.ring_offsets is None:
        raise RuntimeError("polygon relation physicalization requires Polygon rows")
    row_multiplicity_bound = max(int(row_multiplicity_bound), 1)

    polygon_tag = np.int8(FAMILY_TAGS[GeometryFamily.POLYGON])
    d_family_rows = cp.asarray(state.family_row_offsets, dtype=cp.int64)[d_rows]

    source_geometry_offsets = cp.asarray(polygon.geometry_offsets, dtype=cp.int32)
    source_ring_offsets = cp.asarray(polygon.ring_offsets, dtype=cp.int32)
    source_ring_capacity = int(
        cp.flatnonzero(
            cp.arange(max(int(source_ring_offsets.size) - 1, 0), dtype=cp.int64)
            < source_geometry_offsets[-1].astype(cp.int64, copy=False)
        ).size
    )
    source_coord_capacity = int(
        cp.flatnonzero(
            cp.arange(int(polygon.x.size), dtype=cp.int64)
            < source_ring_offsets[source_geometry_offsets[-1:]].astype(
                cp.int64,
                copy=False,
            )[0]
        ).size
    )
    ring_capacity = source_ring_capacity * row_multiplicity_bound
    d_source_rings_capacity, d_geometry_offsets = _device_gather_offset_index_ranges(
        source_geometry_offsets,
        d_family_rows,
        allocation_capacity=ring_capacity,
    )
    d_active_ring_positions = cp.flatnonzero(
        cp.arange(ring_capacity, dtype=cp.int64)
        < d_geometry_offsets[-1].astype(cp.int64, copy=False)
    )
    d_source_rings = d_source_rings_capacity[d_active_ring_positions]
    ring_count = int(d_source_rings.size)

    coord_capacity = source_coord_capacity * row_multiplicity_bound
    d_x_capacity, d_y_capacity, d_ring_offsets = _device_gather_xy_offset_slices(
        polygon.x,
        polygon.y,
        source_ring_offsets,
        d_source_rings,
        allocation_capacity=coord_capacity,
    )
    d_active_coord_positions = cp.flatnonzero(
        cp.arange(coord_capacity, dtype=cp.int64) < d_ring_offsets[-1].astype(cp.int64, copy=False)
    )
    d_x = d_x_capacity[d_active_coord_positions]
    d_y = d_y_capacity[d_active_coord_positions]

    d_validity = cp.asarray(state.validity, dtype=cp.bool_)[d_rows]
    d_bounds = (
        None
        if polygon.bounds is None
        else cp.asarray(polygon.bounds, dtype=cp.float64)[d_family_rows]
    )
    result = build_device_resident_owned(
        device_families={
            GeometryFamily.POLYGON: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.POLYGON,
                x=d_x,
                y=d_y,
                geometry_offsets=d_geometry_offsets,
                empty_mask=cp.asarray(polygon.empty_mask, dtype=cp.bool_)[d_family_rows],
                ring_offsets=d_ring_offsets[: ring_count + 1],
                bounds=d_bounds,
                axis_aligned_rectangles=bool(polygon.axis_aligned_rectangles),
            )
        },
        row_count=row_count,
        tags=cp.full(row_count, polygon_tag, dtype=cp.int8),
        validity=d_validity,
        family_row_offsets=cp.arange(row_count, dtype=cp.int32),
        execution_mode="gpu",
    )
    seed_homogeneous_host_metadata(result, GeometryFamily.POLYGON)
    if state.trusted_all_ogc_valid is True:
        seed_all_validity_cache(result)
    result._device_physicalization_implementation = (
        "exact_polygon_relation_nested_offset_physicalization"
    )
    return result


def _failed_input_residuals_against_candidate_parts_gpu(
    candidate: OwnedGeometryArray,
    inputs: OwnedGeometryArray,
    failure: _GroupedUnionCoverageFailure,
    *,
    dispatch_mode: ExecutionMode,
    record_event: bool = True,
) -> OwnedGeometryArray | None:
    """Compute failed grouped-union residuals against candidate polygon parts.

    Physical shape: sparse failed input rows x exploded candidate polygon parts.
    The older repair path subtracted each failed row from the full grouped
    MultiPolygon row, duplicating large right-side topology and causing huge
    temporary overlay plans.  This carrier explodes each failed output group
    once, builds a bbox relation to the failed inputs, and materializes
    difference only for related part rows.
    """
    if cp is None:
        return None

    failed_selection = failure.failed_selection
    failed_groups = failure.failed_groups
    d_failed_active = failed_selection.active_capacity_mask()
    d_failed_positions = failed_selection.safe_capacity_positions()
    d_failed_source_rows = cp.asarray(
        failure.failed_source_rows_capacity(),
        dtype=cp.int64,
    )
    d_failed_group_active = failed_groups.active_capacity_mask()
    d_failed_group_ids = failed_groups.partition_capacity_positions()
    failed_count = failed_selection.capacity
    if failed_count == 0:
        return inputs.device_take_capacity(d_failed_positions, d_failed_active)
    if int(d_failed_source_rows.size) != failed_count:
        return None

    from vibespatial.constructive.binary_constructive import (
        _empty_device_constructive_output,
        _explode_polygonal_rows_to_polygon_capacity_gpu,
    )
    from vibespatial.geometry.owned import device_select_owned_capacity_partitions
    from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device
    from vibespatial.overlay.gpu import (
        _build_overlay_execution_plan,
        _materialize_overlay_execution_plan,
    )

    failed_inputs = inputs._device_indexed_take(
        d_failed_positions,
    )._apply_row_activity(d_failed_active)

    def _scatter_to_source_capacity(local_result):
        from vibespatial.geometry.owned import (
            device_scatter_owned_capacity_selection,
        )

        return device_scatter_owned_capacity_selection(
            _empty_device_constructive_output(inputs.row_count),
            local_result,
            failed_selection,
            active_mask=d_failed_active,
        )

    candidate_group_rows = candidate._device_indexed_take(
        d_failed_group_ids,
        assume_unique_indices=True,
    )._apply_row_activity(
        d_failed_group_active,
        assume_active_indices_unique=True,
    )
    grouped_source_segments = None
    grouped_output_segments = None
    if failure.group_size_max is not None and failure.source_segment_span_max is not None:
        grouped_source_segments = int(failure.group_size_max) * int(
            failure.source_segment_span_max
        )
        # Every source segment can be split at both endpoints of an overlap
        # with every other segment. Union boundaries are a subset of those
        # noded fragments, so this remains a strict topology capacity bound.
        grouped_output_segments = (
            0
            if grouped_source_segments == 0
            else 2 * grouped_source_segments * grouped_source_segments
            - grouped_source_segments
        )
    candidate_part_capacity = _explode_polygonal_rows_to_polygon_capacity_gpu(
        candidate_group_rows,
        max_parts_per_row=grouped_source_segments,
        max_rings_per_row=grouped_output_segments,
        max_coords_per_row=(
            None if grouped_output_segments is None else 2 * grouped_output_segments
        ),
    )
    if candidate_part_capacity is None:
        return None
    candidate_parts = candidate_part_capacity.geometry
    d_part_active = candidate_part_capacity.selection.active_capacity_mask()
    d_local_group_rows = cp.asarray(
        candidate_part_capacity.source_rows,
        dtype=cp.int64,
    )
    d_part_group_ids = d_failed_group_ids[d_local_group_rows].astype(
        cp.int64,
        copy=False,
    )
    d_part_active &= d_failed_group_active[d_local_group_rows]

    # The explode carrier is sized from a strict grouped-topology capacity.
    # Residual repair is the explicit consumer that needs a concrete bbox
    # relation, so physicalize the live part prefix once before bounds and
    # relation construction instead of repeatedly scanning inactive capacity.
    d_live_part_rows = compact_indices(
        d_part_active.astype(cp.uint8, copy=False)
    ).values.astype(cp.int64, copy=False)
    if int(d_live_part_rows.size) == 0:
        return _scatter_to_source_capacity(failed_inputs)
    candidate_parts = candidate_parts._device_indexed_take(
        d_live_part_rows,
        assume_unique_indices=True,
    )
    d_part_group_ids = d_part_group_ids[d_live_part_rows]
    d_part_active = cp.ones(int(d_live_part_rows.size), dtype=cp.bool_)

    d_left_bounds = cp.asarray(
        compute_geometry_bounds_device(
            failed_inputs,
            preserve_indexed_view=True,
        ),
        dtype=cp.float64,
    )
    d_part_bounds = cp.asarray(
        compute_geometry_bounds_device(
            candidate_parts,
            preserve_indexed_view=True,
        ),
        dtype=cp.float64,
    )
    if (
        d_left_bounds.ndim != 2
        or d_part_bounds.ndim != 2
        or int(d_left_bounds.shape[1]) != 4
        or int(d_part_bounds.shape[1]) != 4
    ):
        return None

    part_capacity = int(candidate_parts.row_count)
    if part_capacity == 0:
        return _scatter_to_source_capacity(failed_inputs)
    if part_capacity > np.iinfo(np.uint32).max:
        return None

    candidate_part_state = candidate_parts._ensure_device_state(preserve_indexed_view=True)
    candidate_polygon = candidate_part_state.families.get(GeometryFamily.POLYGON)
    if candidate_polygon is None or candidate_polygon.ring_offsets is None:
        return None
    # Build the actual sparse relation instead of allocating
    # ``failed_rows * max_parts_per_group`` lanes. Unknown-width grouped
    # output can have a root-capacity part bound that is orders of magnitude
    # larger than every live group. Tiled device nonzero physicalizes only
    # bbox-related, same-group pairs and keeps all row ids resident.
    relation_left_chunks = []
    relation_part_chunks = []
    relation_tile_slots = 8 * 1024 * 1024
    relation_tile_rows = max(
        1,
        min(failed_count, relation_tile_slots // max(part_capacity, 1)),
    )
    d_part_lanes = cp.arange(part_capacity, dtype=cp.int64)
    for row_start in range(0, failed_count, relation_tile_rows):
        row_stop = min(row_start + relation_tile_rows, failed_count)
        d_tile_rows = cp.arange(row_start, row_stop, dtype=cp.int64)
        d_tile_bounds = d_left_bounds[d_tile_rows]
        d_tile_groups = d_failed_source_rows[d_tile_rows]
        d_related = (
            d_failed_active[d_tile_rows, None]
            & d_part_active[None, :]
            & (d_tile_groups[:, None] == d_part_group_ids[None, :])
            & (d_tile_bounds[:, None, 0] <= d_part_bounds[None, :, 2])
            & (d_tile_bounds[:, None, 2] >= d_part_bounds[None, :, 0])
            & (d_tile_bounds[:, None, 1] <= d_part_bounds[None, :, 3])
            & (d_tile_bounds[:, None, 3] >= d_part_bounds[None, :, 1])
        )
        d_tile_left, d_tile_part = cp.nonzero(d_related)
        relation_left_chunks.append(d_tile_left.astype(cp.int64, copy=False) + np.int64(row_start))
        relation_part_chunks.append(d_part_lanes[d_tile_part])

    d_relation_left = (
        cp.concatenate(relation_left_chunks)
        if relation_left_chunks
        else cp.empty(0, dtype=cp.int64)
    )
    d_relation_part = (
        cp.concatenate(relation_part_chunks)
        if relation_part_chunks
        else cp.empty(0, dtype=cp.int64)
    )
    relation_capacity = int(d_relation_left.size)
    if relation_capacity == 0:
        return _scatter_to_source_capacity(failed_inputs)

    exact_left = failed_inputs
    relation_right = _physicalize_polygon_relation_rows_device(
        candidate_parts,
        d_relation_part,
        row_multiplicity_bound=(
            failure.group_size_max if failure.group_size_max is not None else failed_count
        ),
    )
    same_row_span_summary = None
    if grouped_source_segments is not None:
        same_row_span_summary = (
            int(failure.source_segment_span_max),
            grouped_source_segments * (grouped_source_segments + 1),
            max(failed_count - 1, 0),
        )
    plan = _build_overlay_execution_plan(
        exact_left,
        relation_right,
        dispatch_mode=dispatch_mode,
        _cached_right_segments=None,
        _row_isolated=True,
        _use_same_row_fast_path=True,
        _same_row_span_summary=same_row_span_summary,
        _right_geometry_source_rows=d_relation_left.astype(cp.int32, copy=False),
        _right_segment_source_rows=d_relation_left.astype(cp.int32, copy=False),
    )
    exact_result, _selected = _materialize_overlay_execution_plan(
        plan,
        operation="difference",
        requested=ExecutionMode.GPU,
        preserve_row_count=exact_left.row_count,
    )
    if exact_result is None or exact_result.row_count != exact_left.row_count:
        return None

    # A failed input with no bbox-related candidate part has the exact
    # difference identity ``input - empty == input``. The relation carrier
    # represents that row with inactive right lanes, so restore the identity
    # explicitly before scattering back to source-row capacity. Otherwise a
    # completely missing candidate group appears to have zero residual area.
    d_has_related_part = cp.zeros(failed_count, dtype=cp.bool_)
    d_has_related_part[d_relation_left] = True
    exact_result = device_select_owned_capacity_partitions(
        exact_result,
        [(failed_inputs, d_failed_active & ~d_has_related_part)],
    )

    if record_event:
        record_dispatch_event(
            surface="vibespatial.overlay.dissolve.execute_native_grouped_union",
            operation="grouped_union_residual_difference",
            implementation="native_grouped_union_failed_row_part_relation_difference",
            reason=(
                "grouped union residual repair subtracted only bbox-related "
                "candidate polygon parts instead of full grouped MultiPolygon rows"
            ),
            detail=(
                f"failure_capacity={failed_count}, "
                f"group_capacity={int(d_failed_group_ids.size)}, "
                f"candidate_part_capacity={part_capacity}, "
                f"relation_capacity={relation_capacity}, "
                "active_pairs=device-resident"
            ),
            requested=dispatch_mode,
            selected=ExecutionMode.GPU,
        )
    return _scatter_to_source_capacity(exact_result)


def _grouped_union_residual_capacity_device(
    residuals: OwnedGeometryArray,
    failure: _GroupedUnionCoverageFailure,
    *,
    output_row_count: int,
    empty_output: OwnedGeometryArray,
) -> OwnedGeometryArray | None:
    """Reduce source-aligned residual capacity through all observed groups."""
    from vibespatial.api._native_grouped import NativeGroupedSelection
    from vibespatial.constructive.binary_constructive import (
        _regroup_native_grouped_parts_with_grouped_union_gpu,
    )
    from vibespatial.cuda.cccl_primitives import PairSortStrategy, sort_pairs
    from vibespatial.geometry.owned import (
        OwnedGeometryArray,
        build_empty_polygon_rows_device,
        device_scatter_owned_capacity_selection,
    )

    selection = failure.failed_selection
    capacity = selection.capacity
    d_active = selection.active_capacity_mask()
    selected_residuals = residuals._device_indexed_take(
        selection.partition_capacity_positions(),
        assume_unique_indices=selection.unique,
    )._apply_row_activity(
        d_active,
        assume_active_indices_unique=selection.unique,
    )
    residual_noops = device_scatter_owned_capacity_selection(
        build_empty_polygon_rows_device(capacity),
        selected_residuals,
        selection.as_capacity_prefix(),
        active_mask=d_active,
    )
    residual_noops.device_state.trusted_all_valid = True
    residual_noops.device_state.trusted_polygonal_only = True

    grouped = NativeGroupedSelection(
        selection=selection,
        group_codes=cp.asarray(failure.source_rows, dtype=cp.int32),
        group_count=output_row_count,
    )
    d_group_counts = grouped.reduce_numeric(
        cp.ones(capacity, dtype=cp.int32),
        "count",
    ).values.astype(cp.int64, copy=False)
    d_group_counts += 1
    d_group_counts[0] += cp.int64(capacity) - cp.asarray(selection.logical_count, dtype=cp.int64)[0]
    d_group_offsets = cp.empty(output_row_count + 1, dtype=cp.int64)
    d_group_offsets[0] = 0
    cp.cumsum(d_group_counts, out=d_group_offsets[1:])

    total_capacity = output_row_count + capacity
    if total_capacity > np.iinfo(np.uint32).max:
        raise OverflowError("grouped residual capacity exceeds radix lane width")
    d_residual_groups = selection.gather_capacity(
        failure.source_rows,
        fill_value=0,
    ).astype(cp.int64, copy=False)
    d_all_groups = cp.concatenate(
        [
            cp.arange(output_row_count, dtype=cp.int64),
            d_residual_groups,
        ]
    )
    d_sort_keys = (d_all_groups.astype(cp.uint64, copy=False) << cp.uint64(32)) | cp.arange(
        total_capacity, dtype=cp.uint64
    )
    d_order = sort_pairs(
        d_sort_keys,
        cp.arange(total_capacity, dtype=cp.int32),
        strategy=PairSortStrategy.RADIX,
        synchronize=False,
    ).values.astype(cp.int64, copy=False)
    all_parts = OwnedGeometryArray.concat(
        [
            build_empty_polygon_rows_device(output_row_count),
            residual_noops,
        ]
    )
    all_parts.device_state.trusted_all_valid = True
    all_parts.device_state.trusted_polygonal_only = True
    return _regroup_native_grouped_parts_with_grouped_union_gpu(
        all_parts,
        d_order,
        d_group_offsets,
        cp.arange(output_row_count, dtype=cp.int64),
        output_row_count=output_row_count,
        dispatch_mode=ExecutionMode.GPU,
        allow_direct_disjoint_pack=False,
        use_same_row_fast_path=True,
        empty_output=empty_output,
    )


def _repair_grouped_union_uncovered_rows_device(
    candidate: OwnedGeometryArray,
    inputs: OwnedGeometryArray,
    failure: _GroupedUnionCoverageFailure,
    *,
    output_row_count: int,
    empty_output: OwnedGeometryArray,
    stage: str,
) -> OwnedGeometryArray | None:
    """Repair grouped union through source/group-capacity residual closure."""
    if cp is None:
        return None
    from vibespatial.constructive.binary_constructive import (
        _dispatch_row_aligned_polygon_known_coverage_union_gpu,
    )
    from vibespatial.geometry.owned import (
        device_select_owned_capacity_partitions,
        device_valid_nonempty_mask,
    )

    record_dispatch_event(
        surface="vibespatial.overlay.dissolve.execute_native_grouped_union",
        operation="grouped_union_admission",
        implementation="native_grouped_overlay_union_residual_repair_required",
        reason=(
            "grouped overlay union left positive-area residual capacity; "
            "repairing source-selected rows through one grouped merge carrier"
        ),
        detail=(
            f"rows={inputs.row_count}, groups={output_row_count}, "
            f"failure_capacity={failure.failed_selection.capacity}, stage={stage}"
        ),
        requested=ExecutionMode.GPU,
        selected=ExecutionMode.GPU,
    )

    residuals = getattr(failure, "residuals", None)
    if residuals is not None and residuals.row_count != inputs.row_count:
        residuals = None
    if residuals is None:
        residuals = _failed_input_residuals_against_candidate_parts_gpu(
            candidate,
            inputs,
            failure,
            dispatch_mode=ExecutionMode.GPU,
        )
    if residuals is None or residuals.row_count != inputs.row_count:
        return None

    residual_union = _grouped_union_residual_capacity_device(
        residuals,
        failure,
        output_row_count=output_row_count,
        empty_output=empty_output,
    )
    if residual_union is None or residual_union.row_count != int(output_row_count):
        return None

    d_residual_contributes = device_valid_nonempty_mask(residual_union)
    merged = _dispatch_row_aligned_polygon_known_coverage_union_gpu(
        candidate,
        residual_union,
        dispatch_mode=ExecutionMode.GPU,
        assume_all_valid=True,
    )
    if merged is None or merged.row_count != output_row_count:
        return None
    repaired = device_select_owned_capacity_partitions(
        candidate,
        [(merged, d_residual_contributes)],
    )
    group_selection = failure.failed_groups
    record_dispatch_event(
        surface="vibespatial.overlay.dissolve.execute_native_grouped_union",
        operation="grouped_union",
        implementation="native_grouped_overlay_union_residual_coverage_merge",
        reason=(
            "grouped overlay union merged source-capacity residuals through "
            "one valid-empty grouped coverage carrier"
        ),
        detail=(
            f"groups={output_row_count}, group_capacity={group_selection.capacity}, "
            f"failure_capacity={failure.failed_selection.capacity}, stage={stage}"
        ),
        requested=ExecutionMode.GPU,
        selected=ExecutionMode.GPU,
    )
    # Exact closure is C union (I difference C). Re-running the same difference
    # and area classifier cannot provide an independent proof; it only repeats
    # the full topology plan and adds a host decision. Structural failures in
    # difference, grouped residual reduction, or coverage union return above.
    record_dispatch_event(
        surface="vibespatial.overlay.dissolve.execute_native_grouped_union",
        operation="grouped_union",
        implementation="native_grouped_overlay_union_residual_repair",
        reason=(
            "grouped overlay union kept covered output rows and repaired the "
            "missing failed-row residuals with native constructive closure"
        ),
        detail=(
            f"rows={inputs.row_count}, groups={output_row_count}, "
            f"group_capacity={group_selection.capacity}, "
            f"failure_capacity={failure.failed_selection.capacity}, stage={stage}"
        ),
        requested=ExecutionMode.GPU,
        selected=ExecutionMode.GPU,
    )
    return repaired


def _grouped_subresolution_area_rows(
    d_row_area,
    d_row_nonempty,
    d_row_group_local,
    group_count: int,
):
    """Identify members whose area cannot survive their grouped fp64 sum."""
    d_effective_area = cp.where(
        d_row_nonempty,
        cp.nan_to_num(d_row_area, nan=0.0, posinf=cp.inf, neginf=cp.inf),
        cp.float64(0.0),
    )
    d_group_area = cp.zeros(group_count, dtype=cp.float64)
    cp.add.at(d_group_area, d_row_group_local, d_effective_area)
    d_group_ulp = cp.nextafter(d_group_area, cp.inf) - d_group_area
    d_resolution = d_group_ulp[d_row_group_local] * cp.float64(4.0)
    d_subresolution = d_row_nonempty & (d_effective_area <= d_resolution)
    return d_subresolution, d_group_area


def segmented_union_all_device_grouped(
    geometries: OwnedGeometryArray,
    group_offsets: Any,
    group_ids: Any,
    *,
    output_row_count: int,
    precision_plan: PrecisionPlan,
    empty_output: OwnedGeometryArray,
    all_groups_observed: bool | None = None,
    group_size_min: int | None = None,
    group_size_max: int | None = None,
    nonempty_rows_positive_area: bool = False,
    _skip_rectangle_strip: bool = False,
    _skip_disjoint_pack: bool = False,
    _skip_coverage_area_proof: bool = False,
    _capacity_all_valid_noops: bool = False,
    _source_segment_span_max: int | None = None,
) -> OwnedGeometryArray | None:
    """Exact grouped union from device grouped metadata.

    ``group_offsets`` and ``group_ids`` are the compact ``NativeGrouped``
    carriers: rows are already sorted by group, offsets delimit the compact
    non-empty groups, and group IDs scatter compact results back to public
    output rows. Grouped overlay and exact pairwise reduction both keep row
    pairing, carry propagation, and final scatter in device rowsets.
    """
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None
    if geometries.row_count == 0:
        return empty_output

    if _source_segment_span_max is None:
        from vibespatial.constructive.binary_constructive import (
            _polygon_segment_span_bound,
        )

        _source_segment_span_max = _polygon_segment_span_bound(geometries)
    if _source_segment_span_max is not None:
        geometries._active_family_row_segment_capacity_bound = int(
            _source_segment_span_max
        )
    all_input_rows_present = _capacity_all_valid_noops or _all_rows_present_from_existing_proof(
        geometries
    )
    if (
        all_groups_observed is True
        and group_size_min == 1
        and group_size_max == 1
        and all_input_rows_present
        and geometries.row_count == output_row_count
        and int(cp.asarray(group_ids).size) == output_row_count
    ):
        geometries._native_grouped_union_implementation = "native_grouped_singleton_identity_union"
        return geometries

    # Variable-width and multipart groups are the shape where a dense
    # geometry-pair predicate or pairwise union is most expensive. Node their
    # boundaries once, then use per-group fp64 area identity as the exact
    # coverage proof. Groups with positive-area overlap remain in the exact
    # constructive remainder; admitted groups select the noded result directly.
    if not _skip_coverage_area_proof and _source_segment_span_max is None:
        from vibespatial.constructive.binary_constructive import (
            _dispatch_grouped_polygon_known_coverage_union_gpu,
        )
        from vibespatial.constructive.measurement import _area_gpu_device_fp64
        from vibespatial.geometry.owned import (
            build_empty_polygon_rows_device,
            device_select_owned_capacity_partitions,
            device_valid_nonempty_mask,
        )

        d_group_offsets = cp.asarray(group_offsets, dtype=cp.int64)
        d_group_ids = cp.asarray(group_ids, dtype=cp.int64)
        compact_group_count = int(d_group_ids.size)
        d_positions = cp.arange(geometries.row_count, dtype=cp.int64)
        d_group_local = cp.searchsorted(
            d_group_offsets[1:],
            d_positions,
            side="right",
        ).astype(cp.int64, copy=False)
        d_source_rows = d_group_ids[d_group_local].astype(cp.int32, copy=False)
        coverage = _dispatch_grouped_polygon_known_coverage_union_gpu(
            geometries,
            d_source_rows,
            output_row_count=output_row_count,
            dispatch_mode=ExecutionMode.GPU,
            assume_all_valid=True,
            assume_source_rows_valid=True,
            d_valid_empty_rows=cp.ones(output_row_count, dtype=cp.bool_),
        )
        if coverage is not None and coverage.row_count == output_row_count:
            d_source_area = cp.nan_to_num(
                cp.abs(cp.asarray(_area_gpu_device_fp64(geometries), dtype=cp.float64)),
                nan=0.0,
                posinf=cp.inf,
                neginf=cp.inf,
            )
            d_source_nonempty = device_valid_nonempty_mask(geometries)
            d_subresolution_rows, d_group_source_area = _grouped_subresolution_area_rows(
                d_source_area,
                d_source_nonempty,
                d_group_local,
                compact_group_count,
            )
            d_group_subresolution_count = cp.zeros(compact_group_count, dtype=cp.int32)
            cp.add.at(
                d_group_subresolution_count,
                d_group_local,
                d_subresolution_rows.astype(cp.int32, copy=False),
            )
            d_group_min_source_area = cp.full(compact_group_count, cp.inf, dtype=cp.float64)
            cp.minimum.at(
                d_group_min_source_area,
                d_group_local,
                cp.where(d_source_nonempty, d_source_area, cp.inf),
            )
            d_coverage_area = cp.abs(cp.asarray(_area_gpu_device_fp64(coverage), dtype=cp.float64))[
                d_group_ids
            ]
            d_area_scale = cp.maximum(
                cp.maximum(d_group_source_area, d_coverage_area),
                1.0,
            )
            d_default_area_tolerance = (
                d_area_scale * cp.float64(1.0e-12) + cp.float64(1.0e-9)
            )
            d_area_tolerance = cp.minimum(
                d_default_area_tolerance,
                d_group_min_source_area * cp.float64(0.25),
            )
            d_coverage_groups = (
                (d_group_subresolution_count == 0)
                & cp.isfinite(d_coverage_area)
                & (
                cp.abs(d_group_source_area - d_coverage_area)
                <= d_area_tolerance
                )
            )
            d_remainder_rows = ~d_coverage_groups[d_group_local]
            remainder_geometries = _device_row_activity_view(
                geometries,
                d_remainder_rows,
            )
            remainder = segmented_union_all_device_grouped(
                remainder_geometries,
                d_group_offsets,
                d_group_ids,
                output_row_count=output_row_count,
                precision_plan=precision_plan,
                empty_output=empty_output,
                all_groups_observed=all_groups_observed,
                group_size_min=group_size_min,
                group_size_max=group_size_max,
                nonempty_rows_positive_area=nonempty_rows_positive_area,
                _skip_rectangle_strip=_skip_rectangle_strip,
                _skip_disjoint_pack=_skip_disjoint_pack,
                _skip_coverage_area_proof=True,
                _capacity_all_valid_noops=True,
                _source_segment_span_max=_source_segment_span_max,
            )
            if remainder is None or remainder.row_count != output_row_count:
                return None
            d_output_coverage = cp.zeros(output_row_count, dtype=cp.bool_)
            d_output_coverage[d_group_ids] = d_coverage_groups
            combined = device_select_owned_capacity_partitions(
                remainder,
                [(coverage, d_output_coverage)],
            )
            combined._native_grouped_union_implementation = (
                "native_grouped_noded_coverage_area_partition_union"
            )
            record_dispatch_event(
                surface="vibespatial.overlay.dissolve.execute_native_grouped_union",
                operation="grouped_union",
                implementation="native_grouped_noded_coverage_area_partition_union",
                reason=(
                    "variable-width grouped polygon boundaries were noded once; "
                    "fp64 area identity selected interior-disjoint groups over "
                    "the exact positive-overlap remainder"
                ),
                detail=(
                    f"rows={geometries.row_count}, groups={output_row_count}, "
                    "partition_counts=device-resident"
                ),
                requested=ExecutionMode.GPU,
                selected=ExecutionMode.GPU,
            )
            return combined

    rectangle_strip = (
        None
        if _skip_rectangle_strip
        else _grouped_rectangle_strip_union_device(
            geometries,
            group_offsets,
            group_ids,
            output_row_count=output_row_count,
            empty_output=empty_output,
            all_groups_observed=all_groups_observed,
            group_size_min=group_size_min,
            group_size_max=group_size_max,
        )
    )

    if rectangle_strip is None and not _skip_disjoint_pack:
        from vibespatial.constructive.binary_constructive import (
            _native_grouped_strict_disjoint_mask_gpu,
            _pack_native_grouped_disjoint_polygon_parts_gpu,
        )
        from vibespatial.geometry.owned import (
            build_empty_polygon_rows_device,
            device_select_owned_capacity_partitions,
        )

        d_group_offsets = cp.asarray(group_offsets, dtype=cp.int64)
        d_group_ids = cp.asarray(group_ids, dtype=cp.int64)
        d_identity_order = cp.arange(geometries.row_count, dtype=cp.int64)
        d_direct_groups = _native_grouped_strict_disjoint_mask_gpu(
            geometries,
            d_identity_order,
            d_group_offsets,
            group_size_max=group_size_max,
        )
        if d_direct_groups is not None:
            direct = _pack_native_grouped_disjoint_polygon_parts_gpu(
                geometries,
                d_identity_order,
                d_group_offsets,
                d_group_ids,
                output_row_count=output_row_count,
                group_size_max=group_size_max,
                empty_output=empty_output,
                assume_all_valid=True,
                active_group_mask=d_direct_groups,
                assume_active_groups_disjoint=True,
            )
            if direct is None:
                direct = empty_output
                d_direct_groups = cp.zeros_like(d_direct_groups, dtype=cp.bool_)
            if direct.row_count != output_row_count:
                raise RuntimeError("grouped disjoint partition assembly failed")
            d_positions = cp.arange(geometries.row_count, dtype=cp.int64)
            d_group_local = cp.searchsorted(
                d_group_offsets[1:],
                d_positions,
                side="right",
            ).astype(cp.int64, copy=False)
            d_remainder_rows = ~d_direct_groups[d_group_local]
            d_remainder_active = d_remainder_rows
            if not all_input_rows_present:
                d_source_validity = cp.asarray(
                    geometries._ensure_device_state(
                        preserve_indexed_view=True,
                    ).validity,
                    dtype=cp.bool_,
                )
                d_remainder_active &= d_source_validity
            remainder_geometries = _device_row_activity_view(
                geometries,
                d_remainder_active,
            )
            remainder = segmented_union_all_device_grouped(
                remainder_geometries,
                d_group_offsets,
                d_group_ids,
                output_row_count=output_row_count,
                precision_plan=precision_plan,
                empty_output=empty_output,
                all_groups_observed=all_groups_observed,
                group_size_min=group_size_min,
                group_size_max=group_size_max,
                nonempty_rows_positive_area=nonempty_rows_positive_area,
                _skip_rectangle_strip=True,
                _skip_disjoint_pack=True,
                _skip_coverage_area_proof=_skip_coverage_area_proof,
                _capacity_all_valid_noops=True,
                _source_segment_span_max=_source_segment_span_max,
            )
            if remainder is None or remainder.row_count != output_row_count:
                return None
            d_output_direct = cp.zeros(output_row_count, dtype=cp.bool_)
            d_output_direct[d_group_ids] = d_direct_groups
            combined = device_select_owned_capacity_partitions(
                remainder,
                [(direct, d_output_direct)],
            )
            combined._native_grouped_union_remainder_implementation = getattr(
                remainder,
                "_native_grouped_union_implementation",
                None,
            )
            combined._native_grouped_disjoint_pack_output_mask = d_output_direct
            d_remainder_strip = getattr(
                remainder,
                "_native_grouped_rectangle_strip_output_mask",
                None,
            )
            if d_remainder_strip is not None:
                combined._native_grouped_rectangle_strip_output_mask = (
                    cp.asarray(d_remainder_strip, dtype=cp.bool_) & ~d_output_direct
                )
            combined._native_grouped_union_implementation = (
                "native_grouped_disjoint_pack_partition_union"
            )
            return combined

    if rectangle_strip is not None:
        d_strip_group_mask = cp.asarray(
            rectangle_strip._native_grouped_strip_group_mask,
            dtype=cp.bool_,
        )
        compact_group_count = int(cp.asarray(group_ids).size)
        if int(d_strip_group_mask.size) != compact_group_count:
            return None
        d_group_offsets = cp.asarray(group_offsets, dtype=cp.int64)
        d_group_ids = cp.asarray(group_ids, dtype=cp.int64)
        d_row_positions = cp.arange(geometries.row_count, dtype=cp.int64)
        d_row_group_local = cp.searchsorted(
            d_group_offsets[1:],
            d_row_positions,
            side="right",
        ).astype(cp.int64, copy=False)
        from vibespatial.geometry.owned import (
            build_empty_polygon_rows_device,
            device_select_owned_capacity_partitions,
        )

        d_remainder_rows = ~d_strip_group_mask[d_row_group_local]
        remainder_geometries = _device_row_activity_view(
            geometries,
            d_remainder_rows,
        )
        remainder = segmented_union_all_device_grouped(
            remainder_geometries,
            d_group_offsets,
            d_group_ids,
            output_row_count=output_row_count,
            precision_plan=precision_plan,
            empty_output=empty_output,
            all_groups_observed=all_groups_observed,
            group_size_min=group_size_min,
            group_size_max=group_size_max,
            nonempty_rows_positive_area=nonempty_rows_positive_area,
            _skip_rectangle_strip=True,
            _skip_disjoint_pack=_skip_disjoint_pack,
            _skip_coverage_area_proof=_skip_coverage_area_proof,
            _capacity_all_valid_noops=True,
            _source_segment_span_max=_source_segment_span_max,
        )
        if remainder is None or remainder.row_count != output_row_count:
            return None
        d_output_strip_mask = cp.zeros(output_row_count, dtype=cp.bool_)
        d_output_strip_mask[d_group_ids] = d_strip_group_mask
        combined = device_select_owned_capacity_partitions(
            remainder,
            [(rectangle_strip, d_output_strip_mask)],
        )
        combined._native_grouped_union_remainder_implementation = getattr(
            remainder,
            "_native_grouped_union_implementation",
            None,
        )
        combined._native_grouped_rectangle_strip_output_mask = d_output_strip_mask
        combined._native_grouped_union_implementation = (
            "native_grouped_rectangle_strip_partition_union"
        )
        record_dispatch_event(
            surface="vibespatial.overlay.dissolve.execute_native_grouped_union",
            operation="grouped_union",
            implementation="native_grouped_rectangle_strip_partition_union",
            reason=(
                "device rectangle-strip groups were assembled directly and "
                "selected over the exact remainder carrier"
            ),
            detail=(
                f"rows={geometries.row_count}, groups={output_row_count}, "
                "partition_counts=device-resident"
            ),
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
        )
        return combined

    from vibespatial.geometry.owned import (
        build_empty_polygon_rows_device,
        device_select_owned_capacity_partitions,
    )

    d_group_offsets = cp.asarray(group_offsets, dtype=cp.int64)
    d_group_ids = cp.asarray(group_ids, dtype=cp.int64)
    compact_group_count = int(d_group_ids.size)
    if int(d_group_offsets.size) != compact_group_count + 1:
        return None
    state = geometries._ensure_device_state(preserve_indexed_view=True)
    nonempty_rows_positive_area = bool(
        nonempty_rows_positive_area or state.trusted_nonempty_polygonal_positive_area is True
    )
    d_positions = cp.arange(geometries.row_count, dtype=cp.int64)
    d_group_local_by_row = cp.searchsorted(
        d_group_offsets[1:],
        d_positions,
        side="right",
    ).astype(cp.int64, copy=False)
    all_rows_present = all_input_rows_present
    d_structural_counts = (d_group_offsets[1:] - d_group_offsets[:-1]).astype(
        cp.int64,
        copy=False,
    )
    if all_rows_present:
        current = geometries
        d_valid_counts = d_structural_counts
    else:
        d_validity = cp.asarray(state.validity, dtype=cp.bool_)[: geometries.row_count]
        current = device_select_owned_capacity_partitions(
            build_empty_polygon_rows_device(geometries.row_count),
            [(geometries, d_validity)],
        )
        # CuPy scatter-add supports int32 counters but not signed int64. Group
        # spans are already int32-addressable; widen after the device reduction
        # so subsequent offset arithmetic remains int64.
        d_valid_counts_i32 = cp.zeros(compact_group_count, dtype=cp.int32)
        cp.add.at(
            d_valid_counts_i32,
            d_group_local_by_row,
            d_validity.astype(cp.int32, copy=False),
        )
        d_valid_counts = d_valid_counts_i32.astype(cp.int64, copy=False)
    d_current_offsets = d_group_offsets

    d_output_has_values = cp.zeros(output_row_count, dtype=cp.bool_)
    d_output_has_values[d_group_ids] = d_valid_counts > 0

    def _apply_group_validity(result: OwnedGeometryArray) -> OwnedGeometryArray:
        if all_rows_present:
            return result
        implementation = getattr(
            result,
            "_native_grouped_union_implementation",
            None,
        )
        selected = device_select_owned_capacity_partitions(
            empty_output,
            [(result, d_output_has_values)],
        )
        if implementation is not None:
            selected._native_grouped_union_implementation = implementation
        return selected

    from vibespatial.constructive.binary_constructive import (
        _dispatch_polygon_partition_union_gpu,
        _regroup_native_grouped_parts_with_grouped_union_gpu,
    )
    from vibespatial.constructive.measurement import _area_gpu_device_fp64
    from vibespatial.geometry.owned import device_valid_nonempty_mask

    if nonempty_rows_positive_area:
        d_degenerate_row_mask = cp.zeros(current.row_count, dtype=cp.bool_)
    else:
        d_nonempty = device_valid_nonempty_mask(current)
        d_abs_area = cp.abs(cp.asarray(_area_gpu_device_fp64(current), dtype=cp.float64))
        d_degenerate_row_mask, _ = _grouped_subresolution_area_rows(
            d_abs_area,
            d_nonempty,
            d_group_local_by_row,
            compact_group_count,
        )

    safe_parts = device_select_owned_capacity_partitions(
        build_empty_polygon_rows_device(current.row_count),
        [(current, ~d_degenerate_row_mask)],
    )
    safe_overlay = _regroup_native_grouped_parts_with_grouped_union_gpu(
        safe_parts,
        cp.arange(current.row_count, dtype=cp.int64),
        d_current_offsets,
        d_group_ids,
        output_row_count=output_row_count,
        dispatch_mode=ExecutionMode.GPU,
        allow_direct_disjoint_pack=False,
        use_same_row_fast_path=True,
        same_row_span_summary=(
            None
            if _source_segment_span_max is None or group_size_max is None
            else (
                int(_source_segment_span_max),
                int(_source_segment_span_max) * max(int(group_size_max) - 1, 0),
                max(compact_group_count - 1, 0),
            )
        ),
        empty_output=empty_output,
        group_size_max=group_size_max,
    )

    if _capacity_all_valid_noops and safe_overlay is None:
        safe_overlay = empty_output

    if nonempty_rows_positive_area:
        if safe_overlay is None or safe_overlay.row_count != output_row_count:
            return None
        safe_overlay._native_grouped_union_implementation = "native_grouped_overlay_union_plan"
        return _apply_group_validity(safe_overlay)

    from vibespatial.api._native_rowset import NativeDeviceSelection
    from vibespatial.geometry.owned import device_take_owned_capacity_selection

    degenerate_selection = NativeDeviceSelection.from_mask(d_degenerate_row_mask)
    degenerate_parts = device_take_owned_capacity_selection(
        current,
        degenerate_selection,
    )
    d_degenerate_counts = cp.zeros(compact_group_count, dtype=cp.int32)
    cp.add.at(
        d_degenerate_counts,
        d_group_local_by_row,
        d_degenerate_row_mask.astype(cp.int32, copy=False),
    )
    d_degenerate_offsets = cp.empty(compact_group_count + 1, dtype=cp.int64)
    d_degenerate_offsets[0] = 0
    cp.cumsum(d_degenerate_counts, out=d_degenerate_offsets[1:])
    degenerate_tree = _segmented_union_device_grouped_pairwise_tree(
        degenerate_parts,
        d_degenerate_offsets,
        d_group_ids,
        output_row_count=output_row_count,
        precision_plan=precision_plan,
        empty_output=empty_output,
        all_groups_observed=False,
        allow_singleton_identity=False,
        group_size_max=group_size_max,
    )
    if (
        safe_overlay is not None
        and safe_overlay.row_count == output_row_count
        and degenerate_tree is not None
        and degenerate_tree.row_count == output_row_count
    ):
        d_safe_contributes = device_valid_nonempty_mask(safe_overlay)
        d_degenerate_contributes = device_valid_nonempty_mask(degenerate_tree)
        mixed = _dispatch_polygon_partition_union_gpu(
            safe_overlay,
            degenerate_tree,
            dispatch_mode=ExecutionMode.GPU,
            _partition_disjoint=True,
            _active_rows=d_safe_contributes & d_degenerate_contributes,
        )
        if mixed is None or mixed.row_count != output_row_count:
            raise RuntimeError("degenerate grouped union merge row-count mismatch")
        merged = device_select_owned_capacity_partitions(
            empty_output,
            [
                (safe_overlay, d_safe_contributes & ~d_degenerate_contributes),
                (degenerate_tree, d_degenerate_contributes & ~d_safe_contributes),
                (mixed, d_safe_contributes & d_degenerate_contributes),
            ],
        )
        merged._native_grouped_union_implementation = (
            "native_grouped_overlay_union_plan_mixed_degenerate_pairwise"
        )
        return _apply_group_validity(merged)

    raise RuntimeError("admitted grouped overlay union did not produce a complete native result")
