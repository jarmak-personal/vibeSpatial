"""GPU-accelerated area and length measurement kernels.

Tier 1 NVRTC kernels (ADR-0033) for computing geometric area and length
directly from OwnedGeometryArray coordinate buffers.  ADR-0002 METRIC class
precision dispatch: fp32 + Kahan + coordinate centering on consumer GPUs,
native fp64 on datacenter GPUs.

Zero host/device transfers mid-process.  When data is already device-resident
(vibeFrame path), kernels read directly from DeviceFamilyGeometryBuffer
pointers with no copy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_F64,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    _compile_precision_kernel,
    get_cuda_runtime,
)
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    OwnedGeometryArray,
    device_family_coordinate_counts,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.crossover import estimate_physical_work_from_owned
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass
from vibespatial.runtime.residency import Residency, combined_residency

if TYPE_CHECKING:
    from vibespatial.runtime.precision import PrecisionMode, PrecisionPlan

from vibespatial.constructive.measurement_kernels import (
    _LINESTRING_LENGTH_FP32,
    _LINESTRING_LENGTH_FP64,
    _LINESTRING_LENGTH_NAMES,
    _MULTILINESTRING_LENGTH_FP32,
    _MULTILINESTRING_LENGTH_FP64,
    _MULTILINESTRING_LENGTH_NAMES,
    _MULTIPOLYGON_AREA_FP32,
    _MULTIPOLYGON_AREA_FP64,
    _MULTIPOLYGON_AREA_NAMES,
    _MULTIPOLYGON_LENGTH_FP32,
    _MULTIPOLYGON_LENGTH_FP64,
    _MULTIPOLYGON_LENGTH_NAMES,
    _POLYGON_AREA_COOPERATIVE_FP32,
    _POLYGON_AREA_COOPERATIVE_FP64,
    _POLYGON_AREA_COOPERATIVE_NAMES,
    _POLYGON_AREA_FP32,
    _POLYGON_AREA_FP64,
    _POLYGON_AREA_NAMES,
    _POLYGON_LENGTH_FP32,
    _POLYGON_LENGTH_FP64,
    _POLYGON_LENGTH_NAMES,
)

# Background precompilation (ADR-0034)
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup

request_nvrtc_warmup(
    [
        ("polygon-area-fp64", _POLYGON_AREA_FP64, _POLYGON_AREA_NAMES),
        ("polygon-area-fp32", _POLYGON_AREA_FP32, _POLYGON_AREA_NAMES),
        (
            "polygon-area-cooperative-fp64",
            _POLYGON_AREA_COOPERATIVE_FP64,
            _POLYGON_AREA_COOPERATIVE_NAMES,
        ),
        (
            "polygon-area-cooperative-fp32",
            _POLYGON_AREA_COOPERATIVE_FP32,
            _POLYGON_AREA_COOPERATIVE_NAMES,
        ),
        ("multipolygon-area-fp64", _MULTIPOLYGON_AREA_FP64, _MULTIPOLYGON_AREA_NAMES),
        ("multipolygon-area-fp32", _MULTIPOLYGON_AREA_FP32, _MULTIPOLYGON_AREA_NAMES),
        ("polygon-length-fp64", _POLYGON_LENGTH_FP64, _POLYGON_LENGTH_NAMES),
        ("polygon-length-fp32", _POLYGON_LENGTH_FP32, _POLYGON_LENGTH_NAMES),
        ("multipolygon-length-fp64", _MULTIPOLYGON_LENGTH_FP64, _MULTIPOLYGON_LENGTH_NAMES),
        ("multipolygon-length-fp32", _MULTIPOLYGON_LENGTH_FP32, _MULTIPOLYGON_LENGTH_NAMES),
        ("linestring-length-fp64", _LINESTRING_LENGTH_FP64, _LINESTRING_LENGTH_NAMES),
        ("linestring-length-fp32", _LINESTRING_LENGTH_FP32, _LINESTRING_LENGTH_NAMES),
        (
            "multilinestring-length-fp64",
            _MULTILINESTRING_LENGTH_FP64,
            _MULTILINESTRING_LENGTH_NAMES,
        ),
        (
            "multilinestring-length-fp32",
            _MULTILINESTRING_LENGTH_FP32,
            _MULTILINESTRING_LENGTH_NAMES,
        ),
    ]
)


# ---------------------------------------------------------------------------
# Kernel compilation helpers
# ---------------------------------------------------------------------------


def _compile_kernel(
    name_prefix: str,
    fp64_source: str,
    fp32_source: str,
    kernel_names: tuple[str, ...],
    compute_type: str = "double",
):
    return _compile_precision_kernel(
        name_prefix,
        fp64_source,
        fp32_source,
        kernel_names,
        compute_type,
    )


# ---------------------------------------------------------------------------
# Shared helpers: coordinate statistics from OwnedGeometryArray
# ---------------------------------------------------------------------------


def _device_family_active_coordinate_mask(cp_module, family, host_buffer, device_buffer):
    """Return a device mask for initialized coordinate lanes in a family buffer."""
    coordinate_capacity = int(device_buffer.x.size)
    if coordinate_capacity == 0:
        return cp_module.empty(0, dtype=cp_module.bool_)

    geometry_count = min(
        int(host_buffer.row_count),
        max(int(device_buffer.geometry_offsets.size) - 1, 0),
    )
    coordinate_count = cp_module.asarray(device_buffer.geometry_offsets)[geometry_count]
    if family is GeometryFamily.MULTILINESTRING:
        coordinate_count = cp_module.asarray(device_buffer.part_offsets)[coordinate_count]
    elif family is GeometryFamily.POLYGON:
        coordinate_count = cp_module.asarray(device_buffer.ring_offsets)[coordinate_count]
    elif family is GeometryFamily.MULTIPOLYGON:
        part_count = cp_module.asarray(device_buffer.part_offsets)[coordinate_count]
        coordinate_count = cp_module.asarray(device_buffer.ring_offsets)[part_count]
    return cp_module.arange(coordinate_capacity, dtype=cp_module.int64) < coordinate_count


def _device_family_coordinate_stats(cp_module, family, host_buffer, device_buffer):
    """Return six reductions over active, finite coordinate lanes."""
    d_x = cp_module.asarray(device_buffer.x)
    d_y = cp_module.asarray(device_buffer.y)
    d_active = _device_family_active_coordinate_mask(
        cp_module,
        family,
        host_buffer,
        device_buffer,
    )
    d_x_active = d_active & cp_module.isfinite(d_x)
    d_y_active = d_active & cp_module.isfinite(d_y)
    return (
        cp_module.max(cp_module.where(d_x_active, cp_module.abs(d_x), 0.0)),
        cp_module.max(cp_module.where(d_y_active, cp_module.abs(d_y), 0.0)),
        cp_module.min(cp_module.where(d_x_active, d_x, cp_module.inf)),
        cp_module.min(cp_module.where(d_y_active, d_y, cp_module.inf)),
        cp_module.max(cp_module.where(d_x_active, d_x, -cp_module.inf)),
        cp_module.max(cp_module.where(d_y_active, d_y, -cp_module.inf)),
    )


def _device_family_center_stats(cp_module, family, host_buffer, device_buffer):
    """Return active finite count and paired coordinate bounds."""
    d_x = cp_module.asarray(device_buffer.x)
    d_y = cp_module.asarray(device_buffer.y)
    d_active = _device_family_active_coordinate_mask(
        cp_module,
        family,
        host_buffer,
        device_buffer,
    )
    d_active_finite = d_active & cp_module.isfinite(d_x) & cp_module.isfinite(d_y)
    return cp_module.stack(
        (
            cp_module.sum(d_active_finite, dtype=cp_module.int64),
            cp_module.min(cp_module.where(d_active_finite, d_x, cp_module.inf)),
            cp_module.max(cp_module.where(d_active_finite, d_x, -cp_module.inf)),
            cp_module.min(cp_module.where(d_active_finite, d_y, cp_module.inf)),
            cp_module.max(cp_module.where(d_active_finite, d_y, -cp_module.inf)),
        )
    )


def _host_family_active_coordinates(family, buffer):
    """Return initialized host coordinate prefixes for a family buffer."""
    geometry_count = min(int(buffer.row_count), max(int(buffer.geometry_offsets.size) - 1, 0))
    coordinate_count = int(buffer.geometry_offsets[geometry_count])
    if family is GeometryFamily.MULTILINESTRING:
        coordinate_count = int(buffer.part_offsets[coordinate_count])
    elif family is GeometryFamily.POLYGON:
        coordinate_count = int(buffer.ring_offsets[coordinate_count])
    elif family is GeometryFamily.MULTIPOLYGON:
        part_count = int(buffer.part_offsets[coordinate_count])
        coordinate_count = int(buffer.ring_offsets[part_count])
    x = np.asarray(buffer.x[:coordinate_count])
    y = np.asarray(buffer.y[:coordinate_count])
    finite = np.isfinite(x) & np.isfinite(y)
    return x[finite], y[finite]


def _device_bounds_stats(owned: OwnedGeometryArray):
    """Return device coordinate statistics over logical geometry rows."""
    import cupy as cp

    from vibespatial.kernels.core.geometry_analysis import (
        compute_geometry_bounds_device,
    )

    d_bounds = cp.asarray(
        compute_geometry_bounds_device(
            owned,
            preserve_indexed_view=True,
        ),
        dtype=cp.float64,
    ).reshape(owned.row_count, 4)
    d_finite = cp.all(cp.isfinite(d_bounds), axis=1)
    d_minx = cp.min(cp.where(d_finite, d_bounds[:, 0], cp.inf))
    d_miny = cp.min(cp.where(d_finite, d_bounds[:, 1], cp.inf))
    d_maxx = cp.max(cp.where(d_finite, d_bounds[:, 2], -cp.inf))
    d_maxy = cp.max(cp.where(d_finite, d_bounds[:, 3], -cp.inf))
    return cp.stack(
        (
            cp.sum(d_finite, dtype=cp.int64),
            d_minx,
            d_maxx,
            d_miny,
            d_maxy,
            cp.maximum(
                cp.maximum(cp.abs(d_minx), cp.abs(d_maxx)),
                cp.maximum(cp.abs(d_miny), cp.abs(d_maxy)),
            ),
            cp.minimum(d_minx, d_miny),
            cp.maximum(d_maxx, d_maxy),
        )
    )


def _fp32_center_coords(
    owned: OwnedGeometryArray,
) -> tuple[float, float]:
    """Return ``(center_x, center_y)`` for coordinate centering.

    Scans the first non-empty family in *owned* and computes the midpoint of
    the bounding box.  When device buffers are available the four CuPy
    reductions (min_x, max_x, min_y, max_y) are packed into a single device
    array so that only one named runtime D2H export is issued instead of four.

    The host export is issued outside the family search loop to satisfy ZCOPY002
    (no D2H transfers inside loop bodies).
    """
    if owned.device_state is not None:
        h_stats = get_cuda_runtime().copy_device_to_host(
            _device_bounds_stats(owned),
            reason="geometry measurement center-coordinate scalar export",
        )
        if int(h_stats[0]) == 0:
            return 0.0, 0.0
        return (
            float((h_stats[1] + h_stats[2]) * 0.5),
            float((h_stats[3] + h_stats[4]) * 0.5),
        )

    # Phase 1: find the first non-empty family and compute device stats
    # (no .get() inside the loop).
    d_family_stats = []
    cp_module = None
    host_center: tuple[float, float] | None = None
    for fam, buf in owned.families.items():
        if buf.row_count == 0:
            continue
        ds = owned.device_state
        if ds is not None and fam in ds.families:
            try:
                import cupy as _cp
            except ModuleNotFoundError:
                _cp = None
            if _cp is None:
                continue
            d_buf = ds.families[fam]
            if int(d_buf.x.size) > 0:
                cp_module = _cp
                d_family_stats.append(_device_family_center_stats(_cp, fam, buf, d_buf))
        elif buf.x.size > 0:
            x, y = _host_family_active_coordinates(fam, buf)
            if x.size and y.size:
                host_stats = (
                    1.0,
                    float(x.min()),
                    float(x.max()),
                    float(y.min()),
                    float(y.max()),
                )
                if d_family_stats:
                    d_family_stats.append(cp_module.asarray(host_stats))
                else:
                    host_center = (
                        float((host_stats[1] + host_stats[2]) * 0.5),
                        float((host_stats[3] + host_stats[4]) * 0.5),
                    )
                    break

    # Phase 2: single D2H transfer outside the loop.
    if d_family_stats:
        d_candidates = cp_module.stack(d_family_stats)
        d_has_coordinates = d_candidates[:, 0] > 0
        d_first_active = cp_module.argmax(d_has_coordinates)
        d_selected = d_candidates[d_first_active]
        d_stats = cp_module.where(
            cp_module.any(d_has_coordinates),
            cp_module.stack(
                (
                    (d_selected[1] + d_selected[2]) * 0.5,
                    (d_selected[3] + d_selected[4]) * 0.5,
                )
            ),
            cp_module.zeros(2, dtype=cp_module.float64),
        )
        s = get_cuda_runtime().copy_device_to_host(
            d_stats,
            reason="geometry measurement fp32 center-coordinate scalar export",
        )
        return float(s[0]), float(s[1])
    if host_center is not None:
        return host_center
    return 0.0, 0.0


def _coord_stats_from_owned(
    owned: OwnedGeometryArray,
) -> tuple[float, float, float]:
    """Return ``(max_abs, coord_min, coord_max)`` across all families.

    When device buffers are available the six CuPy reductions per family
    (abs_max_x, abs_max_y, min_x, min_y, max_x, max_y) are collected into
    a single device array across ALL families so that only one named runtime
    D2H export is issued outside the loop, satisfying ZCOPY002.
    """
    if owned.device_state is not None:
        h_stats = get_cuda_runtime().copy_device_to_host(
            _device_bounds_stats(owned),
            reason="geometry measurement coordinate-stats scalar export",
        )
        if int(h_stats[0]) == 0:
            return 0.0, float("inf"), float("-inf")
        return float(h_stats[5]), float(h_stats[6]), float(h_stats[7])

    max_abs: float = 0.0
    coord_min: float = float("inf")
    coord_max: float = float("-inf")

    # Phase 1: collect device-side reduction scalars across all families
    # (no .get() inside the loop).
    device_scalars: list = []  # list of CuPy 0-d arrays
    for fam, buf in owned.families.items():
        if buf.row_count == 0:
            continue
        ds = owned.device_state
        if ds is not None and fam in ds.families:
            try:
                import cupy as _cp
            except ModuleNotFoundError:
                _cp = None
            if _cp is None:
                continue
            d_buf = ds.families[fam]
            if int(d_buf.x.size) > 0:
                device_scalars.extend(_device_family_coordinate_stats(_cp, fam, buf, d_buf))
        elif buf.x.size > 0:
            # Host-resident data: accumulate directly (no D2H).
            x, y = _host_family_active_coordinates(fam, buf)
            if x.size:
                max_abs = max(max_abs, float(np.abs(x).max()))
                coord_min = min(coord_min, float(x.min()))
                coord_max = max(coord_max, float(x.max()))
            if y.size:
                max_abs = max(max_abs, float(np.abs(y).max()))
                coord_min = min(coord_min, float(y.min()))
                coord_max = max(coord_max, float(y.max()))

    # Phase 2: single D2H transfer outside the loop for all device families.
    if device_scalars:
        all_stats = get_cuda_runtime().copy_device_to_host(
            _cp.array(device_scalars),
            reason="geometry measurement coordinate-stats scalar export",
        )
        # Process in groups of 6 (abs_max_x, abs_max_y, min_x, min_y, max_x, max_y)
        for i in range(0, len(all_stats), 6):
            max_abs = max(max_abs, float(all_stats[i]), float(all_stats[i + 1]))
            coord_min = min(coord_min, float(all_stats[i + 2]), float(all_stats[i + 3]))
            coord_max = max(coord_max, float(all_stats[i + 4]), float(all_stats[i + 5]))

    return max_abs, coord_min, coord_max


def _measurement_coordinate_summary(
    owned: OwnedGeometryArray,
) -> tuple[float, float, float, tuple[float, float]]:
    """Return precision statistics and center through one device summary."""
    if owned.device_state is not None:
        h_stats = get_cuda_runtime().copy_device_to_host(
            _device_bounds_stats(owned),
            reason="geometry measurement precision-summary scalar export",
        )
        if int(h_stats[0]) == 0:
            return 0.0, float("inf"), float("-inf"), (0.0, 0.0)
        return (
            float(h_stats[5]),
            float(h_stats[6]),
            float(h_stats[7]),
            (
                float((h_stats[1] + h_stats[2]) * 0.5),
                float((h_stats[3] + h_stats[4]) * 0.5),
            ),
        )
    max_abs, coord_min, coord_max = _coord_stats_from_owned(owned)
    return max_abs, coord_min, coord_max, _fp32_center_coords(owned)


# ---------------------------------------------------------------------------
# GPU implementation: Area
# ---------------------------------------------------------------------------


def _single_family_without_nulls(owned: OwnedGeometryArray) -> GeometryFamily | None:
    # Family kernels emit physical family-row order.  Indexed views require the
    # general scatter below to gather those values through family_row_offsets,
    # even when logical and physical row counts happen to match.
    if owned.is_indexed_view:
        return None
    families = getattr(owned, "families", {})
    if len(families) != 1:
        return None
    family, host_buffer = next(iter(families.items()))
    row_count = getattr(host_buffer, "row_count", None)
    device_state = getattr(owned, "device_state", None)
    if device_state is not None:
        if family in device_state.families:
            device_buffer = device_state.families[family]
            offsets = getattr(device_buffer, "geometry_offsets", None)
            if offsets is not None:
                row_count = int(offsets.size) - 1
    if int(row_count or 0) != int(owned.row_count):
        return None
    return family


def _device_metric_family_selection(cp_module, owned, state, family):
    """Return compact family work plus logical scatter indirection."""
    d_global_rows = cp_module.flatnonzero(
        cp_module.asarray(state.validity, dtype=cp_module.bool_)
        & (cp_module.asarray(state.tags) == FAMILY_TAGS[family])
    ).astype(cp_module.int64, copy=False)
    if int(d_global_rows.size) == 0:
        return d_global_rows, None, None, 0
    d_family_rows = cp_module.asarray(state.family_row_offsets)[d_global_rows].astype(
        cp_module.int32,
        copy=False,
    )
    if owned.is_indexed_view:
        d_source_rows, d_inverse = cp_module.unique(
            d_family_rows,
            return_inverse=True,
        )
        return (
            d_global_rows,
            d_source_rows.astype(cp_module.int32, copy=False),
            d_inverse,
            int(d_source_rows.size),
        )
    return (
        d_global_rows,
        None,
        d_family_rows.astype(cp_module.int64, copy=False),
        int(state.families[family].empty_mask.size),
    )


def _launch_polygon_area_rows(
    runtime,
    device_buffer,
    source_rows,
    out,
    *,
    row_count: int,
    cooperative: bool,
    compute_type: str,
    center_x: float,
    center_y: float,
) -> None:
    if cooperative:
        kernels = _compile_kernel(
            "polygon-area-cooperative",
            _POLYGON_AREA_COOPERATIVE_FP64,
            _POLYGON_AREA_COOPERATIVE_FP32,
            _POLYGON_AREA_COOPERATIVE_NAMES,
            compute_type,
        )
        kernel = kernels["polygon_area_cooperative"]
    else:
        kernels = _compile_kernel(
            "polygon-area",
            _POLYGON_AREA_FP64,
            _POLYGON_AREA_FP32,
            _POLYGON_AREA_NAMES,
            compute_type,
        )
        kernel = kernels["polygon_area"]
    ptr = runtime.pointer
    params = (
        (
            ptr(device_buffer.x),
            ptr(device_buffer.y),
            ptr(device_buffer.ring_offsets),
            ptr(device_buffer.geometry_offsets),
            0 if source_rows is None else ptr(source_rows),
            ptr(out),
            center_x,
            center_y,
            row_count,
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_F64,
            KERNEL_PARAM_F64,
            KERNEL_PARAM_I32,
        ),
    )
    if cooperative:
        grid = (row_count, 1, 1)
        block = (256, 1, 1)
    else:
        grid, block = runtime.launch_config(kernel, row_count)
    runtime.launch(kernel, grid=grid, block=block, params=params)


def _launch_selected_polygon_area(
    cp_module,
    runtime,
    device_buffer,
    source_rows,
    out,
    *,
    row_count: int,
    compute_type: str,
    center_x: float,
    center_y: float,
) -> None:
    if source_rows is None:
        _launch_polygon_area_rows(
            runtime,
            device_buffer,
            None,
            out,
            row_count=row_count,
            cooperative=int(device_buffer.x.size) / max(row_count, 1) >= 64,
            compute_type=compute_type,
            center_x=center_x,
            center_y=center_y,
        )
        return
    coordinate_counts = device_family_coordinate_counts(device_buffer, source_rows)
    cooperative_positions = cp_module.flatnonzero(coordinate_counts >= 64).astype(
        cp_module.int32,
        copy=False,
    )
    serial_positions = cp_module.flatnonzero(coordinate_counts < 64).astype(
        cp_module.int32,
        copy=False,
    )
    for positions, cooperative in (
        (serial_positions, False),
        (cooperative_positions, True),
    ):
        selected_count = int(positions.size)
        if selected_count == 0:
            continue
        selected_source_rows = source_rows[positions]
        selected_out = (
            out
            if selected_count == row_count
            else cp_module.empty(selected_count, dtype=cp_module.float64)
        )
        _launch_polygon_area_rows(
            runtime,
            device_buffer,
            selected_source_rows,
            selected_out,
            row_count=selected_count,
            cooperative=cooperative,
            compute_type=compute_type,
            center_x=center_x,
            center_y=center_y,
        )
        if selected_out is not out:
            out[positions] = selected_out


def _area_host_validity_mask(owned: OwnedGeometryArray) -> np.ndarray:
    """Return public area null placement without materializing all host metadata."""
    cached = getattr(owned, "_validity", None)
    if cached is not None:
        return np.asarray(cached, dtype=bool)
    if owned.residency is Residency.DEVICE and owned.device_state is not None:
        state = owned._ensure_device_state(preserve_indexed_view=True)
        return np.asarray(
            get_cuda_runtime().copy_device_to_host(
                state.validity,
                reason="geometry area validity mask host export",
            ),
            dtype=bool,
        )
    return np.asarray(owned.validity, dtype=bool)


@register_kernel_variant(
    "geometry_area",
    "gpu-cuda-python",
    kernel_class=KernelClass.METRIC,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("polygon", "multipolygon"),
    supports_mixed=True,
    tags=("cuda-python", "metric", "area", "kahan", "centered"),
)
def _area_gpu(
    owned: OwnedGeometryArray,
    precision_plan: PrecisionPlan | None = None,
    center_coords: tuple[float, float] | None = None,
) -> np.ndarray:
    """GPU-accelerated area computation.  Returns float64 array of shape (row_count,)."""
    from vibespatial.runtime.precision import PrecisionMode

    compute_type = "double"
    center_x, center_y = 0.0, 0.0
    if precision_plan is not None and precision_plan.compute_precision is PrecisionMode.FP32:
        compute_type = "float"
        if precision_plan.center_coordinates:
            center_x, center_y = (
                _fp32_center_coords(owned) if center_coords is None else center_coords
            )

    runtime = get_cuda_runtime()
    row_count = owned.row_count
    result = np.zeros(row_count, dtype=np.float64)
    device_state = owned.device_state
    single_family = _single_family_without_nulls(owned)

    if single_family is GeometryFamily.POLYGON:
        buf = owned.families[GeometryFamily.POLYGON]
        n = row_count
        if device_state is not None and GeometryFamily.POLYGON in device_state.families:
            avg_verts = (
                int(device_state.families[GeometryFamily.POLYGON].x.size) / max(n, 1)
                if n > 0
                else 0
            )
        else:
            avg_verts = buf.x.size / max(n, 1) if n > 0 else 0
        use_cooperative = avg_verts >= 64

        if use_cooperative:
            coop_kernels = _compile_kernel(
                "polygon-area-cooperative",
                _POLYGON_AREA_COOPERATIVE_FP64,
                _POLYGON_AREA_COOPERATIVE_FP32,
                _POLYGON_AREA_COOPERATIVE_NAMES,
                compute_type,
            )
            kernel = coop_kernels["polygon_area_cooperative"]
        else:
            kernels = _compile_kernel(
                "polygon-area",
                _POLYGON_AREA_FP64,
                _POLYGON_AREA_FP32,
                _POLYGON_AREA_NAMES,
                compute_type,
            )
            kernel = kernels["polygon_area"]

        needs_free = device_state is None or GeometryFamily.POLYGON not in device_state.families
        if not needs_free:
            ds = device_state.families[GeometryFamily.POLYGON]
            d_x, d_y = ds.x, ds.y
            d_ring = ds.ring_offsets
            d_geom = ds.geometry_offsets
        else:
            d_x = runtime.from_host(buf.x)
            d_y = runtime.from_host(buf.y)
            d_ring = runtime.from_host(buf.ring_offsets.astype(np.int32))
            d_geom = runtime.from_host(buf.geometry_offsets.astype(np.int32))

        d_out = runtime.allocate((n,), np.float64)
        try:
            ptr = runtime.pointer
            params = (
                (
                    ptr(d_x),
                    ptr(d_y),
                    ptr(d_ring),
                    ptr(d_geom),
                    0,
                    ptr(d_out),
                    center_x,
                    center_y,
                    n,
                ),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_I32,
                ),
            )
            if use_cooperative:
                grid = (n, 1, 1)
                block = (256, 1, 1)
            else:
                grid, block = runtime.launch_config(kernel, n)
            runtime.launch(kernel, grid=grid, block=block, params=params)
            return runtime.copy_device_to_host(
                d_out,
                reason="geometry area polygon family-result host export",
            )
        finally:
            runtime.free(d_out)
            if needs_free:
                runtime.free(d_x)
                runtime.free(d_y)
                runtime.free(d_ring)
                runtime.free(d_geom)

    if single_family is GeometryFamily.MULTIPOLYGON:
        buf = owned.families[GeometryFamily.MULTIPOLYGON]
        n = row_count
        kernels = _compile_kernel(
            "multipolygon-area",
            _MULTIPOLYGON_AREA_FP64,
            _MULTIPOLYGON_AREA_FP32,
            _MULTIPOLYGON_AREA_NAMES,
            compute_type,
        )
        kernel = kernels["multipolygon_area"]

        needs_free = (
            device_state is None or GeometryFamily.MULTIPOLYGON not in device_state.families
        )
        if not needs_free:
            ds = device_state.families[GeometryFamily.MULTIPOLYGON]
            d_x, d_y = ds.x, ds.y
            d_ring = ds.ring_offsets
            d_part = ds.part_offsets
            d_geom = ds.geometry_offsets
        else:
            d_x = runtime.from_host(buf.x)
            d_y = runtime.from_host(buf.y)
            d_ring = runtime.from_host(buf.ring_offsets.astype(np.int32))
            d_part = runtime.from_host(buf.part_offsets.astype(np.int32))
            d_geom = runtime.from_host(buf.geometry_offsets.astype(np.int32))

        d_out = runtime.allocate((n,), np.float64)
        try:
            ptr = runtime.pointer
            params = (
                (
                    ptr(d_x),
                    ptr(d_y),
                    ptr(d_ring),
                    ptr(d_part),
                    ptr(d_geom),
                    0,
                    ptr(d_out),
                    center_x,
                    center_y,
                    n,
                ),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_I32,
                ),
            )
            grid, block = runtime.launch_config(kernel, n)
            runtime.launch(kernel, grid=grid, block=block, params=params)
            return runtime.copy_device_to_host(
                d_out,
                reason="geometry area multipolygon family-result host export",
            )
        finally:
            runtime.free(d_out)
            if needs_free:
                runtime.free(d_x)
                runtime.free(d_y)
                runtime.free(d_ring)
                runtime.free(d_part)
                runtime.free(d_geom)

    if device_state is not None and (
        owned.is_indexed_view
        or getattr(owned, "_tags", None) is None
        or getattr(owned, "_family_row_offsets", None) is None
        or getattr(owned, "_validity", None) is None
    ):
        d_result = _area_gpu_device_fp64(
            owned,
            precision_plan=precision_plan,
            preserve_indexed_view=True,
            center_coords=center_coords,
        )
        return np.asarray(
            runtime.copy_device_to_host(
                d_result,
                reason="geometry area device-result host export",
            ),
            dtype=np.float64,
        )

    if single_family is not None:
        return result

    tags = owned.tags
    family_row_offsets = owned.family_row_offsets

    # --- Polygon family ---
    poly_tag = FAMILY_TAGS[GeometryFamily.POLYGON]
    poly_mask = tags == poly_tag
    if np.any(poly_mask) and owned.family_has_rows(GeometryFamily.POLYGON):
        buf = owned.families[GeometryFamily.POLYGON]
        n = buf.row_count

        # Choose cooperative vs simple kernel based on avg vertex count.
        # When device_state has the family, read vertex count from device
        # buffers since host stubs may have empty x arrays.
        if device_state is not None and GeometryFamily.POLYGON in (
            device_state.families if device_state else {}
        ):
            avg_verts = (
                int(device_state.families[GeometryFamily.POLYGON].x.size) / max(n, 1)
                if n > 0
                else 0
            )
        else:
            avg_verts = buf.x.size / max(n, 1) if n > 0 else 0
        use_cooperative = avg_verts >= 64

        if use_cooperative:
            coop_kernels = _compile_kernel(
                "polygon-area-cooperative",
                _POLYGON_AREA_COOPERATIVE_FP64,
                _POLYGON_AREA_COOPERATIVE_FP32,
                _POLYGON_AREA_COOPERATIVE_NAMES,
                compute_type,
            )
            kernel = coop_kernels["polygon_area_cooperative"]
        else:
            kernels = _compile_kernel(
                "polygon-area",
                _POLYGON_AREA_FP64,
                _POLYGON_AREA_FP32,
                _POLYGON_AREA_NAMES,
                compute_type,
            )
            kernel = kernels["polygon_area"]

        global_rows = np.flatnonzero(poly_mask)
        family_rows = family_row_offsets[global_rows]

        # Zero-copy: use device pointers if already resident
        needs_free = device_state is None or GeometryFamily.POLYGON not in (
            device_state.families if device_state else {}
        )
        if not needs_free:
            ds = device_state.families[GeometryFamily.POLYGON]
            d_x, d_y = ds.x, ds.y
            d_ring = ds.ring_offsets
            d_geom = ds.geometry_offsets
        else:
            d_x = runtime.from_host(buf.x)
            d_y = runtime.from_host(buf.y)
            d_ring = runtime.from_host(buf.ring_offsets.astype(np.int32))
            d_geom = runtime.from_host(buf.geometry_offsets.astype(np.int32))

        d_out = runtime.allocate((n,), np.float64)
        try:
            ptr = runtime.pointer
            params = (
                (
                    ptr(d_x),
                    ptr(d_y),
                    ptr(d_ring),
                    ptr(d_geom),
                    0,
                    ptr(d_out),
                    center_x,
                    center_y,
                    n,
                ),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_I32,
                ),
            )
            if use_cooperative:
                # 1 block per geometry; fixed at 256 to match __launch_bounds__(256, 4)
                # and shared memory sized for 8 warps (256 / 32).
                grid = (n, 1, 1)
                block = (256, 1, 1)
            else:
                grid, block = runtime.launch_config(kernel, n)
            runtime.launch(kernel, grid=grid, block=block, params=params)
            family_result = runtime.copy_device_to_host(
                d_out,
                reason="geometry area polygon family-result host export",
            )
            result[global_rows] = family_result[family_rows]
        finally:
            runtime.free(d_out)
            if needs_free:
                runtime.free(d_x)
                runtime.free(d_y)
                runtime.free(d_ring)
                runtime.free(d_geom)

    # --- MultiPolygon family ---
    mpoly_tag = FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]
    mpoly_mask = tags == mpoly_tag
    if np.any(mpoly_mask) and owned.family_has_rows(GeometryFamily.MULTIPOLYGON):
        buf = owned.families[GeometryFamily.MULTIPOLYGON]
        kernels = _compile_kernel(
            "multipolygon-area",
            _MULTIPOLYGON_AREA_FP64,
            _MULTIPOLYGON_AREA_FP32,
            _MULTIPOLYGON_AREA_NAMES,
            compute_type,
        )
        kernel = kernels["multipolygon_area"]
        global_rows = np.flatnonzero(mpoly_mask)
        family_rows = family_row_offsets[global_rows]
        n = buf.row_count

        needs_free = device_state is None or GeometryFamily.MULTIPOLYGON not in (
            device_state.families if device_state else {}
        )
        if not needs_free:
            ds = device_state.families[GeometryFamily.MULTIPOLYGON]
            d_x, d_y = ds.x, ds.y
            d_ring = ds.ring_offsets
            d_part = ds.part_offsets
            d_geom = ds.geometry_offsets
        else:
            d_x = runtime.from_host(buf.x)
            d_y = runtime.from_host(buf.y)
            d_ring = runtime.from_host(buf.ring_offsets.astype(np.int32))
            d_part = runtime.from_host(buf.part_offsets.astype(np.int32))
            d_geom = runtime.from_host(buf.geometry_offsets.astype(np.int32))

        d_out = runtime.allocate((n,), np.float64)
        try:
            ptr = runtime.pointer
            params = (
                (
                    ptr(d_x),
                    ptr(d_y),
                    ptr(d_ring),
                    ptr(d_part),
                    ptr(d_geom),
                    0,
                    ptr(d_out),
                    center_x,
                    center_y,
                    n,
                ),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_I32,
                ),
            )
            grid, block = runtime.launch_config(kernel, n)
            runtime.launch(kernel, grid=grid, block=block, params=params)
            family_result = runtime.copy_device_to_host(
                d_out,
                reason="geometry area multipolygon family-result host export",
            )
            result[global_rows] = family_result[family_rows]
        finally:
            runtime.free(d_out)
            if needs_free:
                runtime.free(d_x)
                runtime.free(d_y)
                runtime.free(d_ring)
                runtime.free(d_part)
                runtime.free(d_geom)

    # Points, LineStrings, MultiPoints, MultiLineStrings: area = 0.0 (already zero-initialized)
    return result


def _area_gpu_device_fp64(
    owned: OwnedGeometryArray,
    *,
    precision_plan: PrecisionPlan | None = None,
    preserve_indexed_view: bool = True,
    center_coords: tuple[float, float] | None = None,
):
    """Compute owned geometry areas into a device-resident float64 array.

    This is an internal residency-preserving helper for GPU assembly paths
    that need a boolean area filter, not a public GeoPandas/NumPy result.
    Public ``area_owned`` keeps the precision-planned host-return contract.
    """
    if owned.row_count == 0:
        runtime = get_cuda_runtime()
        return runtime.allocate((0,), np.float64)

    try:
        import cupy as cp
    except ModuleNotFoundError as exc:  # pragma: no cover - GPU guard
        raise RuntimeError("CuPy is required for device-resident area") from exc

    runtime = get_cuda_runtime()
    from vibespatial.runtime.precision import PrecisionMode

    compute_type = "double"
    center_x, center_y = 0.0, 0.0
    if precision_plan is not None and precision_plan.compute_precision is PrecisionMode.FP32:
        compute_type = "float"
        if precision_plan.center_coordinates:
            center_x, center_y = (
                _fp32_center_coords(owned) if center_coords is None else center_coords
            )

    device_state = owned._ensure_device_state(
        preserve_indexed_view=preserve_indexed_view,
    )
    d_result = cp.zeros(owned.row_count, dtype=cp.float64)

    if GeometryFamily.POLYGON in device_state.families:
        ds = device_state.families[GeometryFamily.POLYGON]
        global_rows, source_rows, inverse, n = _device_metric_family_selection(
            cp,
            owned,
            device_state,
            GeometryFamily.POLYGON,
        )
        if n > 0:
            d_out = runtime.allocate((n,), np.float64)
            try:
                _launch_selected_polygon_area(
                    cp,
                    runtime,
                    ds,
                    source_rows,
                    d_out,
                    row_count=n,
                    compute_type=compute_type,
                    center_x=center_x,
                    center_y=center_y,
                )
                d_result[global_rows] = d_out[inverse]
            finally:
                runtime.free(d_out)

    if GeometryFamily.MULTIPOLYGON in device_state.families:
        ds = device_state.families[GeometryFamily.MULTIPOLYGON]
        global_rows, source_rows, inverse, n = _device_metric_family_selection(
            cp,
            owned,
            device_state,
            GeometryFamily.MULTIPOLYGON,
        )
        if n > 0:
            kernels = _compile_kernel(
                "multipolygon-area",
                _MULTIPOLYGON_AREA_FP64,
                _MULTIPOLYGON_AREA_FP32,
                _MULTIPOLYGON_AREA_NAMES,
                compute_type,
            )
            kernel = kernels["multipolygon_area"]
            d_out = runtime.allocate((n,), np.float64)
            try:
                ptr = runtime.pointer
                params = (
                    (
                        ptr(ds.x),
                        ptr(ds.y),
                        ptr(ds.ring_offsets),
                        ptr(ds.part_offsets),
                        ptr(ds.geometry_offsets),
                        0 if source_rows is None else ptr(source_rows),
                        ptr(d_out),
                        center_x,
                        center_y,
                        n,
                    ),
                    (
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_F64,
                        KERNEL_PARAM_F64,
                        KERNEL_PARAM_I32,
                    ),
                )
                grid, block = runtime.launch_config(kernel, n)
                runtime.launch(kernel, grid=grid, block=block, params=params)

                d_result[global_rows] = d_out[inverse]
            finally:
                runtime.free(d_out)

    d_result[~cp.asarray(device_state.validity)] = cp.nan
    return d_result


def area_expression_owned(
    owned: OwnedGeometryArray,
    *,
    source_token: str | None = None,
):
    """Compute geometry area as a private device expression.

    Physical shape: segmented polygon/multipolygon metric reduction to one
    fp64 device value per source row.  Native-backed public ``geometry.area``
    may export this same vector to a public Series, but native row-flow and
    grouped reducers consume it before that compatibility boundary.
    """
    from vibespatial.api._native_expression import NativeExpression

    values = _area_gpu_device_fp64(owned)
    return NativeExpression(
        operation="geometry.area",
        values=values,
        source_token=source_token,
        source_row_count=owned.row_count,
        dtype="float64",
        precision="fp64",
    )


# ---------------------------------------------------------------------------
# GPU implementation: Length
# ---------------------------------------------------------------------------


@register_kernel_variant(
    "geometry_length",
    "gpu-cuda-python",
    kernel_class=KernelClass.METRIC,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("linestring", "multilinestring", "polygon", "multipolygon"),
    supports_mixed=True,
    tags=("cuda-python", "metric", "length", "kahan", "centered"),
)
def _length_gpu(
    owned: OwnedGeometryArray,
    precision_plan: PrecisionPlan | None = None,
    center_coords: tuple[float, float] | None = None,
) -> np.ndarray:
    """GPU-accelerated length computation.  Returns float64 array of shape (row_count,)."""
    from vibespatial.runtime.precision import PrecisionMode

    compute_type = "double"
    center_x, center_y = 0.0, 0.0
    if precision_plan is not None and precision_plan.compute_precision is PrecisionMode.FP32:
        compute_type = "float"
        if precision_plan.center_coordinates:
            center_x, center_y = (
                _fp32_center_coords(owned) if center_coords is None else center_coords
            )

    runtime = get_cuda_runtime()
    row_count = owned.row_count
    result = np.zeros(row_count, dtype=np.float64)
    device_state = owned.device_state
    if device_state is not None and (
        owned.is_indexed_view
        or getattr(owned, "_tags", None) is None
        or getattr(owned, "_family_row_offsets", None) is None
        or getattr(owned, "_validity", None) is None
    ):
        d_result = _length_gpu_device_fp64(
            owned,
            precision_plan=precision_plan,
            center_coords=center_coords,
        )
        return np.asarray(
            runtime.copy_device_to_host(
                d_result,
                reason="geometry length device-result host export",
            ),
            dtype=np.float64,
        )

    tags = owned.tags
    family_row_offsets = owned.family_row_offsets

    def _launch_ring_length(
        family: GeometryFamily,
        kernel_name: str,
        source_fp64: str,
        source_fp32: str,
        names: tuple[str, ...],
        prefix: str,
        has_part_offsets: bool,
    ):
        tag = FAMILY_TAGS[family]
        mask = tags == tag
        if not np.any(mask) or not owned.family_has_rows(family):
            return
        buf = owned.families[family]

        kernels = _compile_kernel(prefix, source_fp64, source_fp32, names, compute_type)
        kernel = kernels[kernel_name]
        global_rows = np.flatnonzero(mask)
        family_rows = family_row_offsets[global_rows]
        n = buf.row_count

        needs_free = device_state is None or family not in (
            device_state.families if device_state else {}
        )
        allocated = []
        if not needs_free:
            ds = device_state.families[family]
            d_x, d_y = ds.x, ds.y
            d_ring = ds.ring_offsets
            d_geom = ds.geometry_offsets
            d_part = ds.part_offsets if has_part_offsets else None
        else:
            d_x = runtime.from_host(buf.x)
            d_y = runtime.from_host(buf.y)
            d_ring = runtime.from_host(buf.ring_offsets.astype(np.int32))
            d_geom = runtime.from_host(buf.geometry_offsets.astype(np.int32))
            allocated.extend([d_x, d_y, d_ring, d_geom])
            if has_part_offsets:
                d_part = runtime.from_host(buf.part_offsets.astype(np.int32))
                allocated.append(d_part)
            else:
                d_part = None

        d_out = runtime.allocate((n,), np.float64)
        try:
            ptr = runtime.pointer
            if has_part_offsets:
                params = (
                    (
                        ptr(d_x),
                        ptr(d_y),
                        ptr(d_ring),
                        ptr(d_part),
                        ptr(d_geom),
                        0,
                        ptr(d_out),
                        center_x,
                        center_y,
                        n,
                    ),
                    (
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_F64,
                        KERNEL_PARAM_F64,
                        KERNEL_PARAM_I32,
                    ),
                )
            else:
                params = (
                    (
                        ptr(d_x),
                        ptr(d_y),
                        ptr(d_ring),
                        ptr(d_geom),
                        0,
                        ptr(d_out),
                        center_x,
                        center_y,
                        n,
                    ),
                    (
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_F64,
                        KERNEL_PARAM_F64,
                        KERNEL_PARAM_I32,
                    ),
                )
            grid, block = runtime.launch_config(kernel, n)
            runtime.launch(kernel, grid=grid, block=block, params=params)
            family_result = runtime.copy_device_to_host(
                d_out,
                reason=f"geometry length {family.value} ring-result host export",
            )
            result[global_rows] = family_result[family_rows]
        finally:
            runtime.free(d_out)
            for d in allocated:
                runtime.free(d)

    def _launch_line_length(
        family: GeometryFamily,
        kernel_name: str,
        source_fp64: str,
        source_fp32: str,
        names: tuple[str, ...],
        prefix: str,
        has_part_offsets: bool,
    ):
        tag = FAMILY_TAGS[family]
        mask = tags == tag
        if not np.any(mask) or not owned.family_has_rows(family):
            return
        buf = owned.families[family]

        kernels = _compile_kernel(prefix, source_fp64, source_fp32, names, compute_type)
        kernel = kernels[kernel_name]
        global_rows = np.flatnonzero(mask)
        family_rows = family_row_offsets[global_rows]
        n = buf.row_count

        needs_free = device_state is None or family not in (
            device_state.families if device_state else {}
        )
        allocated = []
        if not needs_free:
            ds = device_state.families[family]
            d_x, d_y = ds.x, ds.y
            d_geom = ds.geometry_offsets
            d_part = ds.part_offsets if has_part_offsets else None
        else:
            d_x = runtime.from_host(buf.x)
            d_y = runtime.from_host(buf.y)
            d_geom = runtime.from_host(buf.geometry_offsets.astype(np.int32))
            allocated.extend([d_x, d_y, d_geom])
            if has_part_offsets:
                d_part = runtime.from_host(buf.part_offsets.astype(np.int32))
                allocated.append(d_part)
            else:
                d_part = None

        d_out = runtime.allocate((n,), np.float64)
        try:
            ptr = runtime.pointer
            if has_part_offsets:
                params = (
                    (
                        ptr(d_x),
                        ptr(d_y),
                        ptr(d_part),
                        ptr(d_geom),
                        0,
                        ptr(d_out),
                        center_x,
                        center_y,
                        n,
                    ),
                    (
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_F64,
                        KERNEL_PARAM_F64,
                        KERNEL_PARAM_I32,
                    ),
                )
            else:
                params = (
                    (
                        ptr(d_x),
                        ptr(d_y),
                        ptr(d_geom),
                        0,
                        ptr(d_out),
                        center_x,
                        center_y,
                        n,
                    ),
                    (
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_F64,
                        KERNEL_PARAM_F64,
                        KERNEL_PARAM_I32,
                    ),
                )
            grid, block = runtime.launch_config(kernel, n)
            runtime.launch(kernel, grid=grid, block=block, params=params)
            family_result = runtime.copy_device_to_host(
                d_out,
                reason=f"geometry length {family.value} line-result host export",
            )
            result[global_rows] = family_result[family_rows]
        finally:
            runtime.free(d_out)
            for d in allocated:
                runtime.free(d)

    # Polygon length (all rings)
    _launch_ring_length(
        GeometryFamily.POLYGON,
        "polygon_length",
        _POLYGON_LENGTH_FP64,
        _POLYGON_LENGTH_FP32,
        _POLYGON_LENGTH_NAMES,
        "polygon-length",
        has_part_offsets=False,
    )

    # MultiPolygon length (all rings of all polygon parts)
    _launch_ring_length(
        GeometryFamily.MULTIPOLYGON,
        "multipolygon_length",
        _MULTIPOLYGON_LENGTH_FP64,
        _MULTIPOLYGON_LENGTH_FP32,
        _MULTIPOLYGON_LENGTH_NAMES,
        "multipolygon-length",
        has_part_offsets=True,
    )

    # LineString length
    _launch_line_length(
        GeometryFamily.LINESTRING,
        "linestring_length",
        _LINESTRING_LENGTH_FP64,
        _LINESTRING_LENGTH_FP32,
        _LINESTRING_LENGTH_NAMES,
        "linestring-length",
        has_part_offsets=False,
    )

    # MultiLineString length
    _launch_line_length(
        GeometryFamily.MULTILINESTRING,
        "multilinestring_length",
        _MULTILINESTRING_LENGTH_FP64,
        _MULTILINESTRING_LENGTH_FP32,
        _MULTILINESTRING_LENGTH_NAMES,
        "multilinestring-length",
        has_part_offsets=True,
    )

    # Points and MultiPoints: length = 0.0 (already zero-initialized)
    return result


def _length_gpu_device_fp64(
    owned: OwnedGeometryArray,
    *,
    precision_plan: PrecisionPlan | None = None,
    center_coords: tuple[float, float] | None = None,
):
    """Compute owned geometry lengths into a device-resident float64 array.

    This is an internal NativeExpression helper.  Public ``length_owned`` keeps
    the precision-planned host-return contract.
    """
    if owned.row_count == 0:
        runtime = get_cuda_runtime()
        return runtime.allocate((0,), np.float64)

    try:
        import cupy as cp
    except ModuleNotFoundError as exc:  # pragma: no cover - GPU guard
        raise RuntimeError("CuPy is required for device-resident length") from exc

    runtime = get_cuda_runtime()
    from vibespatial.runtime.precision import PrecisionMode

    compute_type = "double"
    center_x, center_y = 0.0, 0.0
    if precision_plan is not None and precision_plan.compute_precision is PrecisionMode.FP32:
        compute_type = "float"
        if precision_plan.center_coordinates:
            center_x, center_y = (
                _fp32_center_coords(owned) if center_coords is None else center_coords
            )

    device_state = owned._ensure_device_state(preserve_indexed_view=True)
    d_result = cp.zeros(owned.row_count, dtype=cp.float64)

    def _launch_ring_length(
        family: GeometryFamily,
        kernel_name: str,
        source_fp64: str,
        source_fp32: str,
        names: tuple[str, ...],
        prefix: str,
        *,
        has_part_offsets: bool,
    ) -> None:
        if family not in device_state.families:
            return
        ds = device_state.families[family]
        global_rows, source_rows, inverse, n = _device_metric_family_selection(
            cp,
            owned,
            device_state,
            family,
        )
        if n <= 0:
            return
        kernels = _compile_kernel(prefix, source_fp64, source_fp32, names, compute_type)
        kernel = kernels[kernel_name]
        d_out = runtime.allocate((n,), np.float64)
        try:
            ptr = runtime.pointer
            if has_part_offsets:
                params = (
                    (
                        ptr(ds.x),
                        ptr(ds.y),
                        ptr(ds.ring_offsets),
                        ptr(ds.part_offsets),
                        ptr(ds.geometry_offsets),
                        0 if source_rows is None else ptr(source_rows),
                        ptr(d_out),
                        center_x,
                        center_y,
                        n,
                    ),
                    (
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_F64,
                        KERNEL_PARAM_F64,
                        KERNEL_PARAM_I32,
                    ),
                )
            else:
                params = (
                    (
                        ptr(ds.x),
                        ptr(ds.y),
                        ptr(ds.ring_offsets),
                        ptr(ds.geometry_offsets),
                        0 if source_rows is None else ptr(source_rows),
                        ptr(d_out),
                        center_x,
                        center_y,
                        n,
                    ),
                    (
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_F64,
                        KERNEL_PARAM_F64,
                        KERNEL_PARAM_I32,
                    ),
                )
            grid, block = runtime.launch_config(kernel, n)
            runtime.launch(kernel, grid=grid, block=block, params=params)
            d_result[global_rows] = d_out[inverse]
        finally:
            runtime.free(d_out)

    def _launch_line_length(
        family: GeometryFamily,
        kernel_name: str,
        source_fp64: str,
        source_fp32: str,
        names: tuple[str, ...],
        prefix: str,
        *,
        has_part_offsets: bool,
    ) -> None:
        if family not in device_state.families:
            return
        ds = device_state.families[family]
        global_rows, source_rows, inverse, n = _device_metric_family_selection(
            cp,
            owned,
            device_state,
            family,
        )
        if n <= 0:
            return
        kernels = _compile_kernel(prefix, source_fp64, source_fp32, names, compute_type)
        kernel = kernels[kernel_name]
        d_out = runtime.allocate((n,), np.float64)
        try:
            ptr = runtime.pointer
            if has_part_offsets:
                params = (
                    (
                        ptr(ds.x),
                        ptr(ds.y),
                        ptr(ds.part_offsets),
                        ptr(ds.geometry_offsets),
                        0 if source_rows is None else ptr(source_rows),
                        ptr(d_out),
                        center_x,
                        center_y,
                        n,
                    ),
                    (
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_F64,
                        KERNEL_PARAM_F64,
                        KERNEL_PARAM_I32,
                    ),
                )
            else:
                params = (
                    (
                        ptr(ds.x),
                        ptr(ds.y),
                        ptr(ds.geometry_offsets),
                        0 if source_rows is None else ptr(source_rows),
                        ptr(d_out),
                        center_x,
                        center_y,
                        n,
                    ),
                    (
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_F64,
                        KERNEL_PARAM_F64,
                        KERNEL_PARAM_I32,
                    ),
                )
            grid, block = runtime.launch_config(kernel, n)
            runtime.launch(kernel, grid=grid, block=block, params=params)
            d_result[global_rows] = d_out[inverse]
        finally:
            runtime.free(d_out)

    _launch_ring_length(
        GeometryFamily.POLYGON,
        "polygon_length",
        _POLYGON_LENGTH_FP64,
        _POLYGON_LENGTH_FP32,
        _POLYGON_LENGTH_NAMES,
        "polygon-length",
        has_part_offsets=False,
    )
    _launch_ring_length(
        GeometryFamily.MULTIPOLYGON,
        "multipolygon_length",
        _MULTIPOLYGON_LENGTH_FP64,
        _MULTIPOLYGON_LENGTH_FP32,
        _MULTIPOLYGON_LENGTH_NAMES,
        "multipolygon-length",
        has_part_offsets=True,
    )
    _launch_line_length(
        GeometryFamily.LINESTRING,
        "linestring_length",
        _LINESTRING_LENGTH_FP64,
        _LINESTRING_LENGTH_FP32,
        _LINESTRING_LENGTH_NAMES,
        "linestring-length",
        has_part_offsets=False,
    )
    _launch_line_length(
        GeometryFamily.MULTILINESTRING,
        "multilinestring_length",
        _MULTILINESTRING_LENGTH_FP64,
        _MULTILINESTRING_LENGTH_FP32,
        _MULTILINESTRING_LENGTH_NAMES,
        "multilinestring-length",
        has_part_offsets=True,
    )

    d_result[~cp.asarray(device_state.validity)] = cp.nan
    return d_result


def length_expression_owned(
    owned: OwnedGeometryArray,
    *,
    source_token: str | None = None,
):
    """Compute geometry length as a private device expression.

    Physical shape: segmented line/ring metric reduction to one fp64 device
    value per source row.  The sanctioned consumers match area expressions:
    native row-flow and grouped reducers.
    """
    from vibespatial.api._native_expression import NativeExpression

    values = _length_gpu_device_fp64(owned)
    return NativeExpression(
        operation="geometry.length",
        values=values,
        source_token=source_token,
        source_row_count=owned.row_count,
        dtype="float64",
        precision="fp64",
    )


# ---------------------------------------------------------------------------
# CPU fallback: Area (NumPy, NO Shapely)
# ---------------------------------------------------------------------------


@register_kernel_variant(
    "geometry_area",
    "cpu",
    kernel_class=KernelClass.METRIC,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=("polygon", "multipolygon"),
    supports_mixed=True,
    tags=("numpy", "metric", "area"),
)
def _area_cpu(owned: OwnedGeometryArray) -> np.ndarray:
    """CPU area computation using NumPy — no Shapely dependency."""
    # Materialize host buffers from device if needed (stubs have empty x/y
    # and None ring_offsets when host_materialized=False).
    owned._ensure_host_state()
    row_count = owned.row_count
    result = np.zeros(row_count, dtype=np.float64)
    tags = owned.tags
    family_row_offsets = owned.family_row_offsets

    for family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
        tag = FAMILY_TAGS[family]
        mask = tags == tag
        if not np.any(mask) or family not in owned.families:
            continue
        buf = owned.families[family]
        if buf.row_count == 0 or buf.ring_offsets is None:
            continue

        global_rows = np.flatnonzero(mask)
        family_rows = family_row_offsets[global_rows]
        x, y = buf.x, buf.y
        ring_offsets = buf.ring_offsets
        geom_offsets = buf.geometry_offsets
        part_offsets = buf.part_offsets
        is_multi = family is GeometryFamily.MULTIPOLYGON

        for gi, fr in zip(global_rows, family_rows):
            if is_multi:
                first_part = geom_offsets[fr]
                last_part = geom_offsets[fr + 1]
                total = 0.0
                for part in range(first_part, last_part):
                    first_ring = part_offsets[part]
                    last_ring = part_offsets[part + 1]
                    total += _rings_area(x, y, ring_offsets, first_ring, last_ring)
                result[gi] = total
            else:
                first_ring = geom_offsets[fr]
                last_ring = geom_offsets[fr + 1]
                result[gi] = _rings_area(x, y, ring_offsets, first_ring, last_ring)

    return result


def _rings_area(x, y, ring_offsets, first_ring, last_ring):
    """Compute area for a set of rings (exterior + holes)."""
    total = 0.0
    for ring in range(first_ring, last_ring):
        cs = ring_offsets[ring]
        ce = ring_offsets[ring + 1]
        n = ce - cs
        if n < 3:
            continue
        # Strip closure vertex
        if n >= 2:
            dx = x[cs] - x[ce - 1]
            dy = y[cs] - y[ce - 1]
            if dx * dx + dy * dy < 1e-24:
                n -= 1
        if n < 3:
            continue

        rx = x[cs : cs + n]
        ry = y[cs : cs + n]
        rx1 = np.roll(rx, -1)
        ry1 = np.roll(ry, -1)
        signed_area = np.sum(rx * ry1 - rx1 * ry) * 0.5

        if ring == first_ring:
            total += abs(signed_area)
        else:
            total -= abs(signed_area)
    return total


# ---------------------------------------------------------------------------
# CPU fallback: Length (NumPy, NO Shapely)
# ---------------------------------------------------------------------------


@register_kernel_variant(
    "geometry_length",
    "cpu",
    kernel_class=KernelClass.METRIC,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=("linestring", "multilinestring", "polygon", "multipolygon"),
    supports_mixed=True,
    tags=("numpy", "metric", "length"),
)
def _length_cpu(owned: OwnedGeometryArray) -> np.ndarray:
    """CPU length computation using NumPy — no Shapely dependency."""
    # Materialize host buffers from device if needed (stubs have empty x/y
    # and None ring_offsets when host_materialized=False).
    owned._ensure_host_state()
    row_count = owned.row_count
    result = np.zeros(row_count, dtype=np.float64)
    tags = owned.tags
    family_row_offsets = owned.family_row_offsets

    # LineString
    _length_cpu_lines(
        owned, result, tags, family_row_offsets, GeometryFamily.LINESTRING, multi=False
    )
    # MultiLineString
    _length_cpu_lines(
        owned, result, tags, family_row_offsets, GeometryFamily.MULTILINESTRING, multi=True
    )
    # Polygon (all rings)
    _length_cpu_rings(owned, result, tags, family_row_offsets, GeometryFamily.POLYGON, multi=False)
    # MultiPolygon (all rings of all polygon parts)
    _length_cpu_rings(
        owned, result, tags, family_row_offsets, GeometryFamily.MULTIPOLYGON, multi=True
    )

    return result


def _length_cpu_lines(owned, result, tags, family_row_offsets, family: GeometryFamily, multi: bool):
    tag = FAMILY_TAGS[family]
    mask = tags == tag
    if not np.any(mask) or family not in owned.families:
        return
    buf = owned.families[family]
    if buf.row_count == 0:
        return
    global_rows = np.flatnonzero(mask)
    family_rows = family_row_offsets[global_rows]
    x, y = buf.x, buf.y
    geom_offsets = buf.geometry_offsets
    part_offsets = buf.part_offsets

    for gi, fr in zip(global_rows, family_rows):
        if multi:
            fp = geom_offsets[fr]
            lp = geom_offsets[fr + 1]
            total = 0.0
            for p in range(fp, lp):
                cs = part_offsets[p]
                ce = part_offsets[p + 1]
                total += _segment_length_sum(x, y, cs, ce)
            result[gi] = total
        else:
            cs = geom_offsets[fr]
            ce = geom_offsets[fr + 1]
            result[gi] = _segment_length_sum(x, y, cs, ce)


def _length_cpu_rings(owned, result, tags, family_row_offsets, family: GeometryFamily, multi: bool):
    tag = FAMILY_TAGS[family]
    mask = tags == tag
    if not np.any(mask) or family not in owned.families:
        return
    buf = owned.families[family]
    if buf.row_count == 0 or buf.ring_offsets is None:
        return
    global_rows = np.flatnonzero(mask)
    family_rows = family_row_offsets[global_rows]
    x, y = buf.x, buf.y
    ring_offsets = buf.ring_offsets
    geom_offsets = buf.geometry_offsets
    part_offsets = buf.part_offsets

    for gi, fr in zip(global_rows, family_rows):
        if multi:
            fp = geom_offsets[fr]
            lp = geom_offsets[fr + 1]
            total = 0.0
            for p in range(fp, lp):
                fring = part_offsets[p]
                lring = part_offsets[p + 1]
                for ring in range(fring, lring):
                    cs = ring_offsets[ring]
                    ce = ring_offsets[ring + 1]
                    total += _segment_length_sum(x, y, cs, ce)
            result[gi] = total
        else:
            fring = geom_offsets[fr]
            lring = geom_offsets[fr + 1]
            total = 0.0
            for ring in range(fring, lring):
                cs = ring_offsets[ring]
                ce = ring_offsets[ring + 1]
                total += _segment_length_sum(x, y, cs, ce)
            result[gi] = total


def _segment_length_sum(x, y, cs, ce):
    """Sum of Euclidean segment lengths for coords[cs:ce]."""
    n = ce - cs
    if n < 2:
        return 0.0
    dx = np.diff(x[cs:ce])
    dy = np.diff(y[cs:ce])
    return float(np.sum(np.sqrt(dx * dx + dy * dy)))


# ---------------------------------------------------------------------------
# Public dispatch API
# ---------------------------------------------------------------------------


def area_owned(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = "auto",
) -> np.ndarray:
    """Compute area directly from OwnedGeometryArray coordinate buffers.

    GPU path uses ADR-0002 METRIC-class precision dispatch.
    Returns float64 array of shape (row_count,).

    Zero host/device transfers mid-process.  When owned.device_state
    is populated (vibeFrame path), GPU kernels read directly from
    device pointers with no copy.
    """
    from vibespatial.runtime.precision import CoordinateStats

    row_count = owned.row_count
    if row_count == 0:
        return np.empty(0, dtype=np.float64)

    max_abs, coord_min, coord_max, center_coords = _measurement_coordinate_summary(owned)
    span = coord_max - coord_min if np.isfinite(coord_min) else 0.0
    selection = plan_dispatch_selection(
        kernel_name="geometry_area",
        kernel_class=KernelClass.METRIC,
        row_count=row_count,
        work_estimate=estimate_physical_work_from_owned(
            owned,
            output_row_count=row_count,
            output_byte_count=row_count * np.dtype(np.float64).itemsize,
            primary_unit_name="area-ring-coordinate",
        ),
        requested_mode=dispatch_mode,
        requested_precision=precision,
        coordinate_stats=CoordinateStats(max_abs_coord=max_abs, span=span),
        current_residency=combined_residency(owned),
    )

    if selection.selected is ExecutionMode.GPU:
        precision_plan = selection.precision_plan
        result = _area_gpu(
            owned,
            precision_plan=precision_plan,
            center_coords=center_coords,
        )
        if _single_family_without_nulls(owned) is None:
            result[~_area_host_validity_mask(owned)] = np.nan
        record_dispatch_event(
            surface="geopandas.array.area",
            operation="area",
            implementation="gpu_nvrtc_shoelace",
            reason=selection.reason,
            detail=f"rows={row_count}, precision={precision_plan.compute_precision}",
            requested=selection.requested,
            selected=ExecutionMode.GPU,
        )
        return result

    record_dispatch_event(
        surface="geopandas.array.area",
        operation="area",
        implementation="numpy",
        reason=selection.reason,
        detail=f"rows={row_count}",
        requested=selection.requested,
        selected=ExecutionMode.CPU,
    )
    result = _area_cpu(owned)
    result[~owned.validity] = np.nan
    return result


def length_owned(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = "auto",
) -> np.ndarray:
    """Compute length directly from OwnedGeometryArray coordinate buffers.

    For Polygons, measures the perimeter (all rings including holes).
    For LineStrings, measures total segment length.
    Points return 0.0.

    GPU path uses ADR-0002 METRIC-class precision dispatch.
    Returns float64 array of shape (row_count,).

    Zero host/device transfers mid-process.  When owned.device_state
    is populated (vibeFrame path), GPU kernels read directly from
    device pointers with no copy.
    """
    from vibespatial.runtime.precision import CoordinateStats

    row_count = owned.row_count
    if row_count == 0:
        return np.empty(0, dtype=np.float64)

    max_abs, coord_min, coord_max, center_coords = _measurement_coordinate_summary(owned)
    span = coord_max - coord_min if np.isfinite(coord_min) else 0.0
    selection = plan_dispatch_selection(
        kernel_name="geometry_length",
        kernel_class=KernelClass.METRIC,
        row_count=row_count,
        work_estimate=estimate_physical_work_from_owned(
            owned,
            output_row_count=row_count,
            output_byte_count=row_count * np.dtype(np.float64).itemsize,
            primary_unit_name="length-segment",
        ),
        requested_mode=dispatch_mode,
        requested_precision=precision,
        coordinate_stats=CoordinateStats(max_abs_coord=max_abs, span=span),
        current_residency=combined_residency(owned),
    )

    if selection.selected is ExecutionMode.GPU:
        precision_plan = selection.precision_plan
        result = _length_gpu(
            owned,
            precision_plan=precision_plan,
            center_coords=center_coords,
        )
        result[~owned.validity] = np.nan
        record_dispatch_event(
            surface="geopandas.array.length",
            operation="length",
            implementation="gpu_nvrtc_segment_length",
            reason=selection.reason,
            detail=f"rows={row_count}, precision={precision_plan.compute_precision}",
            requested=selection.requested,
            selected=ExecutionMode.GPU,
        )
        return result

    record_dispatch_event(
        surface="geopandas.array.length",
        operation="length",
        implementation="numpy",
        reason=selection.reason,
        detail=f"rows={row_count}",
        requested=selection.requested,
        selected=ExecutionMode.CPU,
    )
    result = _length_cpu(owned)
    result[~owned.validity] = np.nan
    return result
