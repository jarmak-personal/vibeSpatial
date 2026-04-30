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
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass
from vibespatial.runtime.residency import combined_residency

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

request_nvrtc_warmup([
    ("polygon-area-fp64", _POLYGON_AREA_FP64, _POLYGON_AREA_NAMES),
    ("polygon-area-fp32", _POLYGON_AREA_FP32, _POLYGON_AREA_NAMES),
    ("polygon-area-cooperative-fp64", _POLYGON_AREA_COOPERATIVE_FP64, _POLYGON_AREA_COOPERATIVE_NAMES),
    ("polygon-area-cooperative-fp32", _POLYGON_AREA_COOPERATIVE_FP32, _POLYGON_AREA_COOPERATIVE_NAMES),
    ("multipolygon-area-fp64", _MULTIPOLYGON_AREA_FP64, _MULTIPOLYGON_AREA_NAMES),
    ("multipolygon-area-fp32", _MULTIPOLYGON_AREA_FP32, _MULTIPOLYGON_AREA_NAMES),
    ("polygon-length-fp64", _POLYGON_LENGTH_FP64, _POLYGON_LENGTH_NAMES),
    ("polygon-length-fp32", _POLYGON_LENGTH_FP32, _POLYGON_LENGTH_NAMES),
    ("multipolygon-length-fp64", _MULTIPOLYGON_LENGTH_FP64, _MULTIPOLYGON_LENGTH_NAMES),
    ("multipolygon-length-fp32", _MULTIPOLYGON_LENGTH_FP32, _MULTIPOLYGON_LENGTH_NAMES),
    ("linestring-length-fp64", _LINESTRING_LENGTH_FP64, _LINESTRING_LENGTH_NAMES),
    ("linestring-length-fp32", _LINESTRING_LENGTH_FP32, _LINESTRING_LENGTH_NAMES),
    ("multilinestring-length-fp64", _MULTILINESTRING_LENGTH_FP64, _MULTILINESTRING_LENGTH_NAMES),
    ("multilinestring-length-fp32", _MULTILINESTRING_LENGTH_FP32, _MULTILINESTRING_LENGTH_NAMES),
])


# ---------------------------------------------------------------------------
# Kernel compilation helpers
# ---------------------------------------------------------------------------

def _compile_kernel(name_prefix: str, fp64_source: str, fp32_source: str,
                    kernel_names: tuple[str, ...], compute_type: str = "double"):
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
    # Phase 1: find the first non-empty family and compute device stats
    # (no .get() inside the loop).
    d_stats = None
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
                d_x = _cp.asarray(d_buf.x)
                d_y = _cp.asarray(d_buf.y)
                # Batch 4 reductions into 1 device array (transfer deferred)
                d_stats = _cp.array([
                    _cp.min(d_x), _cp.max(d_x),
                    _cp.min(d_y), _cp.max(d_y),
                ])
                break
        elif buf.x.size > 0:
            host_center = (
                float((buf.x.min() + buf.x.max()) * 0.5),
                float((buf.y.min() + buf.y.max()) * 0.5),
            )
            break

    # Phase 2: single D2H transfer outside the loop.
    if d_stats is not None:
        s = get_cuda_runtime().copy_device_to_host(
            d_stats,
            reason="geometry measurement fp32 center-coordinate scalar export",
        )
        return float((s[0] + s[1]) * 0.5), float((s[2] + s[3]) * 0.5)
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
                d_x = _cp.asarray(d_buf.x)
                d_y = _cp.asarray(d_buf.y)
                # 6 reduction scalars per family, deferred to bulk transfer
                device_scalars.extend([
                    _cp.max(_cp.abs(d_x)), _cp.max(_cp.abs(d_y)),
                    _cp.min(d_x), _cp.min(d_y),
                    _cp.max(d_x), _cp.max(d_y),
                ])
        elif buf.x.size > 0:
            # Host-resident data: accumulate directly (no D2H).
            max_abs = max(max_abs, float(np.abs(buf.x).max()), float(np.abs(buf.y).max()))
            coord_min = min(coord_min, float(buf.x.min()), float(buf.y.min()))
            coord_max = max(coord_max, float(buf.x.max()), float(buf.y.max()))

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


# ---------------------------------------------------------------------------
# GPU implementation: Area
# ---------------------------------------------------------------------------

def _single_family_without_nulls(owned: OwnedGeometryArray) -> GeometryFamily | None:
    families = getattr(owned, "families", {})
    if len(families) != 1:
        return None
    family, host_buffer = next(iter(families.items()))
    row_count = getattr(host_buffer, "row_count", None)
    device_state = getattr(owned, "device_state", None)
    if device_state is not None:
        device_buffer = device_state.families.get(family)
        if device_buffer is not None:
            offsets = getattr(device_buffer, "geometry_offsets", None)
            if offsets is not None:
                row_count = int(offsets.size) - 1
    if int(row_count or 0) != int(owned.row_count):
        return None
    return family


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
) -> np.ndarray:
    """GPU-accelerated area computation.  Returns float64 array of shape (row_count,)."""
    from vibespatial.runtime.precision import PrecisionMode

    compute_type = "double"
    center_x, center_y = 0.0, 0.0
    if precision_plan is not None and precision_plan.compute_precision is PrecisionMode.FP32:
        compute_type = "float"
        if precision_plan.center_coordinates:
            center_x, center_y = _fp32_center_coords(owned)

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

        needs_free = (
            device_state is None or GeometryFamily.POLYGON not in device_state.families
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
            device_state is None
            or GeometryFamily.MULTIPOLYGON not in device_state.families
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
        if device_state is not None and GeometryFamily.POLYGON in (device_state.families if device_state else {}):
            avg_verts = int(device_state.families[GeometryFamily.POLYGON].x.size) / max(n, 1) if n > 0 else 0
        else:
            avg_verts = buf.x.size / max(n, 1) if n > 0 else 0
        use_cooperative = avg_verts >= 64

        if use_cooperative:
            coop_kernels = _compile_kernel(
                "polygon-area-cooperative",
                _POLYGON_AREA_COOPERATIVE_FP64, _POLYGON_AREA_COOPERATIVE_FP32,
                _POLYGON_AREA_COOPERATIVE_NAMES, compute_type,
            )
            kernel = coop_kernels["polygon_area_cooperative"]
        else:
            kernels = _compile_kernel("polygon-area", _POLYGON_AREA_FP64, _POLYGON_AREA_FP32,
                                      _POLYGON_AREA_NAMES, compute_type)
            kernel = kernels["polygon_area"]

        global_rows = np.flatnonzero(poly_mask)
        family_rows = family_row_offsets[global_rows]

        # Zero-copy: use device pointers if already resident
        needs_free = device_state is None or GeometryFamily.POLYGON not in (device_state.families if device_state else {})
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
                (ptr(d_x), ptr(d_y), ptr(d_ring), ptr(d_geom),
                 ptr(d_out), center_x, center_y, n),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_F64, KERNEL_PARAM_F64, KERNEL_PARAM_I32),
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
        kernels = _compile_kernel("multipolygon-area", _MULTIPOLYGON_AREA_FP64, _MULTIPOLYGON_AREA_FP32,
                                  _MULTIPOLYGON_AREA_NAMES, compute_type)
        kernel = kernels["multipolygon_area"]
        global_rows = np.flatnonzero(mpoly_mask)
        family_rows = family_row_offsets[global_rows]
        n = buf.row_count

        needs_free = device_state is None or GeometryFamily.MULTIPOLYGON not in (device_state.families if device_state else {})
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
                (ptr(d_x), ptr(d_y), ptr(d_ring), ptr(d_part), ptr(d_geom),
                 ptr(d_out), center_x, center_y, n),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_F64, KERNEL_PARAM_F64,
                 KERNEL_PARAM_I32),
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


def _area_gpu_device_fp64(owned: OwnedGeometryArray):
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
    device_state = owned._ensure_device_state()
    d_result = cp.zeros(owned.row_count, dtype=cp.float64)
    d_tags = cp.asarray(device_state.tags)
    d_family_row_offsets = cp.asarray(device_state.family_row_offsets)
    center_x, center_y = 0.0, 0.0

    if GeometryFamily.POLYGON in device_state.families:
        ds = device_state.families[GeometryFamily.POLYGON]
        n = int(ds.empty_mask.size)
        if n > 0:
            avg_verts = int(ds.x.size) / max(n, 1)
            use_cooperative = avg_verts >= 64
            if use_cooperative:
                kernels = _compile_kernel(
                    "polygon-area-cooperative",
                    _POLYGON_AREA_COOPERATIVE_FP64,
                    _POLYGON_AREA_COOPERATIVE_FP32,
                    _POLYGON_AREA_COOPERATIVE_NAMES,
                    "double",
                )
                kernel = kernels["polygon_area_cooperative"]
            else:
                kernels = _compile_kernel(
                    "polygon-area",
                    _POLYGON_AREA_FP64,
                    _POLYGON_AREA_FP32,
                    _POLYGON_AREA_NAMES,
                    "double",
                )
                kernel = kernels["polygon_area"]

            d_out = runtime.allocate((n,), np.float64)
            try:
                ptr = runtime.pointer
                params = (
                    (
                        ptr(ds.x),
                        ptr(ds.y),
                        ptr(ds.ring_offsets),
                        ptr(ds.geometry_offsets),
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
                if use_cooperative:
                    grid = (n, 1, 1)
                    block = (256, 1, 1)
                else:
                    grid, block = runtime.launch_config(kernel, n)
                runtime.launch(kernel, grid=grid, block=block, params=params)

                global_rows = cp.flatnonzero(
                    d_tags == FAMILY_TAGS[GeometryFamily.POLYGON],
                ).astype(cp.int64, copy=False)
                if int(global_rows.size) > 0:
                    family_rows = d_family_row_offsets[global_rows].astype(
                        cp.int64,
                        copy=False,
                    )
                    d_result[global_rows] = d_out[family_rows]
            finally:
                runtime.free(d_out)

    if GeometryFamily.MULTIPOLYGON in device_state.families:
        ds = device_state.families[GeometryFamily.MULTIPOLYGON]
        n = int(ds.empty_mask.size)
        if n > 0:
            kernels = _compile_kernel(
                "multipolygon-area",
                _MULTIPOLYGON_AREA_FP64,
                _MULTIPOLYGON_AREA_FP32,
                _MULTIPOLYGON_AREA_NAMES,
                "double",
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

                global_rows = cp.flatnonzero(
                    d_tags == FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
                ).astype(cp.int64, copy=False)
                if int(global_rows.size) > 0:
                    family_rows = d_family_row_offsets[global_rows].astype(
                        cp.int64,
                        copy=False,
                    )
                    d_result[global_rows] = d_out[family_rows]
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
    fp64 device value per source row.  The only sanctioned consumers are
    native row-flow and grouped reducers; public ``geometry.area`` continues
    to materialize through ``area_owned``.
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
) -> np.ndarray:
    """GPU-accelerated length computation.  Returns float64 array of shape (row_count,)."""
    from vibespatial.runtime.precision import PrecisionMode

    compute_type = "double"
    center_x, center_y = 0.0, 0.0
    if precision_plan is not None and precision_plan.compute_precision is PrecisionMode.FP32:
        compute_type = "float"
        if precision_plan.center_coordinates:
            center_x, center_y = _fp32_center_coords(owned)

    runtime = get_cuda_runtime()
    row_count = owned.row_count
    result = np.zeros(row_count, dtype=np.float64)

    tags = owned.tags
    family_row_offsets = owned.family_row_offsets
    device_state = owned.device_state

    def _launch_ring_length(family: GeometryFamily, kernel_name: str, source_fp64: str,
                            source_fp32: str, names: tuple[str, ...], prefix: str,
                            has_part_offsets: bool):
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

        needs_free = device_state is None or family not in (device_state.families if device_state else {})
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
                    (ptr(d_x), ptr(d_y), ptr(d_ring), ptr(d_part), ptr(d_geom),
                     ptr(d_out), center_x, center_y, n),
                    (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_F64, KERNEL_PARAM_F64,
                     KERNEL_PARAM_I32),
                )
            else:
                params = (
                    (ptr(d_x), ptr(d_y), ptr(d_ring), ptr(d_geom),
                     ptr(d_out), center_x, center_y, n),
                    (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR, KERNEL_PARAM_F64, KERNEL_PARAM_F64, KERNEL_PARAM_I32),
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

    def _launch_line_length(family: GeometryFamily, kernel_name: str, source_fp64: str,
                            source_fp32: str, names: tuple[str, ...], prefix: str,
                            has_part_offsets: bool):
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

        needs_free = device_state is None or family not in (device_state.families if device_state else {})
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
                    (ptr(d_x), ptr(d_y), ptr(d_part), ptr(d_geom),
                     ptr(d_out), center_x, center_y, n),
                    (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR, KERNEL_PARAM_F64, KERNEL_PARAM_F64, KERNEL_PARAM_I32),
                )
            else:
                params = (
                    (ptr(d_x), ptr(d_y), ptr(d_geom),
                     ptr(d_out), center_x, center_y, n),
                    (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR, KERNEL_PARAM_F64, KERNEL_PARAM_F64, KERNEL_PARAM_I32),
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
        GeometryFamily.POLYGON, "polygon_length",
        _POLYGON_LENGTH_FP64, _POLYGON_LENGTH_FP32,
        _POLYGON_LENGTH_NAMES, "polygon-length", has_part_offsets=False,
    )

    # MultiPolygon length (all rings of all polygon parts)
    _launch_ring_length(
        GeometryFamily.MULTIPOLYGON, "multipolygon_length",
        _MULTIPOLYGON_LENGTH_FP64, _MULTIPOLYGON_LENGTH_FP32,
        _MULTIPOLYGON_LENGTH_NAMES, "multipolygon-length", has_part_offsets=True,
    )

    # LineString length
    _launch_line_length(
        GeometryFamily.LINESTRING, "linestring_length",
        _LINESTRING_LENGTH_FP64, _LINESTRING_LENGTH_FP32,
        _LINESTRING_LENGTH_NAMES, "linestring-length", has_part_offsets=False,
    )

    # MultiLineString length
    _launch_line_length(
        GeometryFamily.MULTILINESTRING, "multilinestring_length",
        _MULTILINESTRING_LENGTH_FP64, _MULTILINESTRING_LENGTH_FP32,
        _MULTILINESTRING_LENGTH_NAMES, "multilinestring-length", has_part_offsets=True,
    )

    # Points and MultiPoints: length = 0.0 (already zero-initialized)
    return result


def _length_gpu_device_fp64(owned: OwnedGeometryArray):
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
    device_state = owned._ensure_device_state()
    d_result = cp.zeros(owned.row_count, dtype=cp.float64)
    d_tags = cp.asarray(device_state.tags)
    d_family_row_offsets = cp.asarray(device_state.family_row_offsets)
    center_x, center_y = 0.0, 0.0

    def _scatter_family_result(family: GeometryFamily, d_out) -> None:
        global_rows = cp.flatnonzero(d_tags == FAMILY_TAGS[family]).astype(
            cp.int64,
            copy=False,
        )
        if int(global_rows.size) == 0:
            return
        family_rows = d_family_row_offsets[global_rows].astype(cp.int64, copy=False)
        d_result[global_rows] = d_out[family_rows]

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
        n = int(ds.empty_mask.size)
        if n <= 0:
            return
        kernels = _compile_kernel(prefix, source_fp64, source_fp32, names, "double")
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
                        ptr(ds.ring_offsets),
                        ptr(ds.geometry_offsets),
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
            _scatter_family_result(family, d_out)
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
        n = int(ds.empty_mask.size)
        if n <= 0:
            return
        kernels = _compile_kernel(prefix, source_fp64, source_fp32, names, "double")
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
            else:
                params = (
                    (
                        ptr(ds.x),
                        ptr(ds.y),
                        ptr(ds.geometry_offsets),
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
                        KERNEL_PARAM_F64,
                        KERNEL_PARAM_F64,
                        KERNEL_PARAM_I32,
                    ),
                )
            grid, block = runtime.launch_config(kernel, n)
            runtime.launch(kernel, grid=grid, block=block, params=params)
            _scatter_family_result(family, d_out)
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

        rx = x[cs:cs + n]
        ry = y[cs:cs + n]
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
    _length_cpu_lines(owned, result, tags, family_row_offsets,
                      GeometryFamily.LINESTRING, multi=False)
    # MultiLineString
    _length_cpu_lines(owned, result, tags, family_row_offsets,
                      GeometryFamily.MULTILINESTRING, multi=True)
    # Polygon (all rings)
    _length_cpu_rings(owned, result, tags, family_row_offsets,
                      GeometryFamily.POLYGON, multi=False)
    # MultiPolygon (all rings of all polygon parts)
    _length_cpu_rings(owned, result, tags, family_row_offsets,
                      GeometryFamily.MULTIPOLYGON, multi=True)

    return result


def _length_cpu_lines(owned, result, tags, family_row_offsets,
                      family: GeometryFamily, multi: bool):
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


def _length_cpu_rings(owned, result, tags, family_row_offsets,
                      family: GeometryFamily, multi: bool):
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

    max_abs, coord_min, coord_max = _coord_stats_from_owned(owned)
    span = coord_max - coord_min if np.isfinite(coord_min) else 0.0
    selection = plan_dispatch_selection(
        kernel_name="geometry_area",
        kernel_class=KernelClass.METRIC,
        row_count=row_count,
        requested_mode=dispatch_mode,
        requested_precision=precision,
        coordinate_stats=CoordinateStats(max_abs_coord=max_abs, span=span),
        current_residency=combined_residency(owned),
    )

    if selection.selected is ExecutionMode.GPU:
        precision_plan = selection.precision_plan
        result = _area_gpu(owned, precision_plan=precision_plan)
        if _single_family_without_nulls(owned) is None:
            result[~owned.validity] = np.nan
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

    max_abs, coord_min, coord_max = _coord_stats_from_owned(owned)
    span = coord_max - coord_min if np.isfinite(coord_min) else 0.0
    selection = plan_dispatch_selection(
        kernel_name="geometry_length",
        kernel_class=KernelClass.METRIC,
        row_count=row_count,
        requested_mode=dispatch_mode,
        requested_precision=precision,
        coordinate_stats=CoordinateStats(max_abs_coord=max_abs, span=span),
        current_residency=combined_residency(owned),
    )

    if selection.selected is ExecutionMode.GPU:
        precision_plan = selection.precision_plan
        result = _length_gpu(owned, precision_plan=precision_plan)
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
