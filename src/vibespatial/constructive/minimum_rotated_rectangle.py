"""GPU-accelerated minimum rotated rectangle (oriented envelope) computation.

Uses rotating calipers on the convex hull to find the minimum-area bounding
rectangle for each geometry.  Output is always Polygon family: each row
produces a closed 5-vertex polygon (the minimum-area rotated rectangle).

Algorithm per geometry:
  1. Compute convex hull (reuses existing GPU convex_hull kernel).
  2. For each edge of the convex hull, compute the perpendicular bounding
     rectangle by rotating coordinates into an edge-aligned frame.
  3. Track the rectangle with minimum area.
  4. Output the 4 corners (+ closure) of the minimum-area rectangle.

Degenerate cases:
  - Point -> degenerate polygon (point repeated 5 times).
  - Collinear / <3 hull vertices -> degenerate rectangle from the line
    segment endpoints.

ADR-0033: Tier 1 NVRTC for rotating calipers (geometry-specific inner loop).
ADR-0002: CONSTRUCTIVE class -- stays fp64 on all devices per policy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

from vibespatial.constructive.minimum_rotated_rectangle_cpu import (
    _min_rect_cpu as _min_rect_cpu,
)
from vibespatial.constructive.minimum_rotated_rectangle_cpu import (
    _minimum_rotated_rectangle_cpu,
)
from vibespatial.constructive.minimum_rotated_rectangle_kernels import (
    _MIN_RECT_KERNEL_NAMES,
    _MIN_RECT_KERNEL_SOURCE,
)
from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    OwnedGeometryArray,
)
from vibespatial.runtime import ExecutionMode, has_gpu_runtime
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.fallbacks import (
    StrictNativeFallbackError,
    record_fallback_event,
    strict_native_mode_enabled,
)
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode
from vibespatial.runtime.residency import Residency

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NVRTC warmup (ADR-0034): register kernels for background precompilation
# ---------------------------------------------------------------------------
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("min-rotated-rect", _MIN_RECT_KERNEL_SOURCE, _MIN_RECT_KERNEL_NAMES),
])


# ---------------------------------------------------------------------------
# GPU implementation
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "minimum_rotated_rectangle",
    "gpu-cuda-python",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=(
        "point", "multipoint", "linestring", "multilinestring",
        "polygon",
    ),
    supports_mixed=False,
    tags=("cuda-python", "constructive", "minimum_rotated_rectangle"),
)
def _min_rect_gpu(owned: OwnedGeometryArray) -> OwnedGeometryArray:
    """GPU minimum rotated rectangle using convex hull + rotating calipers.

    Pipeline:
      1. Compute convex hull on GPU (reuses existing convex_hull_owned).
      2. Launch rotating calipers NVRTC kernel on hull output.
      3. Assemble output as 5-vertex closed Polygon.

    Returns a device-resident Polygon OwnedGeometryArray.
    """
    from .convex_hull import convex_hull_owned

    runtime = get_cuda_runtime()
    row_count = owned.row_count

    # Step 1: Compute convex hull on GPU -- stays device-resident
    hull_owned = convex_hull_owned(owned, dispatch_mode=ExecutionMode.GPU)

    # The hull output is a Polygon OwnedGeometryArray with one ring per row.
    # We need the device buffers for the hull coordinates.
    d_state = hull_owned._ensure_device_state()

    # The hull is always single-family Polygon
    hull_device_buf = d_state.families[GeometryFamily.POLYGON]
    d_hull_x = cp.asarray(hull_device_buf.x)
    d_hull_y = cp.asarray(hull_device_buf.y)
    d_ring_offsets = cp.asarray(hull_device_buf.ring_offsets)
    d_geom_offsets = cp.asarray(hull_device_buf.geometry_offsets)

    # Step 2: Allocate output -- 5 vertices per geometry
    d_out_x = runtime.allocate((row_count * 5,), np.float64)
    d_out_y = runtime.allocate((row_count * 5,), np.float64)

    # Step 3: Launch rotating calipers kernel
    kernels = compile_kernel_group(
        "min-rotated-rect", _MIN_RECT_KERNEL_SOURCE, _MIN_RECT_KERNEL_NAMES,
    )
    kernel = kernels["minimum_rotated_rectangle"]

    ptr = runtime.pointer
    params = (
        (
            ptr(d_hull_x), ptr(d_hull_y),
            ptr(d_ring_offsets), ptr(d_geom_offsets),
            ptr(d_out_x), ptr(d_out_y),
            row_count,
        ),
        (
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
        ),
    )
    grid, block = runtime.launch_config(kernel, row_count)
    runtime.launch(kernel, grid=grid, block=block, params=params)

    # Step 4: Build output Polygon OwnedGeometryArray (5 verts per ring)
    from .point import _build_device_backed_polygon_output

    return _build_device_backed_polygon_output(
        d_out_x,
        d_out_y,
        row_count=row_count,
        bounds=None,
        verts_per_ring=5,
    )


# ---------------------------------------------------------------------------
# Public dispatch API
# ---------------------------------------------------------------------------

def minimum_rotated_rectangle_owned(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """Compute the minimum rotated rectangle (oriented envelope) of each geometry.

    Uses rotating calipers on the convex hull.  Output is always Polygon
    family: each row produces a closed 5-vertex polygon (the minimum-area
    bounding rectangle).

    Parameters
    ----------
    owned : OwnedGeometryArray
        Input geometries (any family).
    dispatch_mode : ExecutionMode or str
        Execution mode selection.  Defaults to AUTO.
    precision : PrecisionMode or str
        Precision mode selection.  Defaults to AUTO.
        CONSTRUCTIVE class: stays fp64 on all devices per ADR-0002.

    Returns
    -------
    OwnedGeometryArray
        Polygon OwnedGeometryArray with one minimum rotated rectangle per
        input row.
    """
    row_count = owned.row_count
    if row_count == 0:
        from vibespatial.geometry.owned import from_shapely_geometries

        return from_shapely_geometries([])

    selection = plan_dispatch_selection(
        kernel_name="minimum_rotated_rectangle",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=row_count,
        requested_mode=dispatch_mode,
        requested_precision=precision,
        current_residency=owned.residency,
    )

    selected_mode = selection.selected
    if strict_native_mode_enabled() and has_gpu_runtime() and cp is not None:
        selected_mode = ExecutionMode.GPU

    if selected_mode is ExecutionMode.GPU and cp is not None:
        precision_plan = selection.precision_plan
        # GPU path supports single-family non-MultiPolygon inputs
        families_with_rows = [
            fam for fam in owned.families
            if owned.family_has_rows(fam)
        ]
        is_single_family = len(families_with_rows) == 1
        has_multipolygon = GeometryFamily.MULTIPOLYGON in families_with_rows

        if is_single_family and not has_multipolygon:
            try:
                result = _min_rect_gpu(owned)
            except StrictNativeFallbackError:
                raise
            except Exception as exc:
                logger.debug(
                    "GPU minimum_rotated_rectangle failed, falling back to CPU",
                    exc_info=True,
                )
                record_fallback_event(
                    surface="geopandas.array.minimum_rotated_rectangle",
                    reason="GPU minimum_rotated_rectangle failed",
                    detail=(
                        f"rows={row_count}, "
                        f"families={','.join(fam.value for fam in families_with_rows)}, "
                        f"precision={precision_plan.compute_precision.value}, "
                        f"error={type(exc).__name__}"
                    ),
                    requested=selection.requested,
                    selected=ExecutionMode.CPU,
                    d2h_transfer=owned.residency is Residency.DEVICE,
                )
            else:
                record_dispatch_event(
                    surface="geopandas.array.minimum_rotated_rectangle",
                    operation="minimum_rotated_rectangle",
                    requested=dispatch_mode,
                    selected=ExecutionMode.GPU,
                    implementation="min_rotated_rect_gpu_nvrtc",
                    reason=selection.reason,
                    detail=f"precision={precision_plan.compute_precision}",
                )
                return result
        else:
            record_fallback_event(
                surface="geopandas.array.minimum_rotated_rectangle",
                reason="GPU minimum_rotated_rectangle does not support this geometry mix",
                detail=(
                    f"rows={row_count}, "
                    f"families={','.join(fam.value for fam in families_with_rows)}, "
                    f"has_multipolygon={has_multipolygon}"
                ),
                requested=selection.requested,
                selected=ExecutionMode.CPU,
                d2h_transfer=owned.residency is Residency.DEVICE,
            )

    result = _minimum_rotated_rectangle_cpu(owned)
    record_dispatch_event(
        surface="geopandas.array.minimum_rotated_rectangle",
        operation="minimum_rotated_rectangle",
        requested=dispatch_mode,
        selected=ExecutionMode.CPU,
        implementation="min_rotated_rect_cpu_shapely",
        reason=selection.reason,
    )
    return result


def minimum_rotated_rectangle_native_tabular_result(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
    crs=None,
    geometry_name: str = "geometry",
    source_rows=None,
    source_tokens: tuple[str, ...] = (),
):
    """Return oriented-envelope output as a private native constructive carrier."""
    from vibespatial.api._native_results import (
        _unary_constructive_owned_to_native_tabular_result,
    )

    result = minimum_rotated_rectangle_owned(
        owned,
        dispatch_mode=dispatch_mode,
        precision=precision,
    )
    return _unary_constructive_owned_to_native_tabular_result(
        result,
        operation="minimum_rotated_rectangle",
        crs=crs,
        geometry_name=geometry_name,
        source_rows=source_rows,
        source_tokens=source_tokens,
    )
