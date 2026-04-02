"""GPU-accelerated set_precision: grid snapping + topology cleanup.

Replaces shapely.set_precision() with a zero-D2H GPU pipeline.

Pipeline stages:
  1. Grid quantization (Tier 2 CuPy): snap coordinates to grid_size multiples
  2. Deduplication (reuse remove_repeated_points): remove coincident points
  3. Topology cleanup (valid_output mode): make_valid for full repair

Three modes:
  - pointwise: snap only, no topology cleanup (fastest)
  - keep_collapsed: snap + dedup, keep degenerate geometries
  - valid_output (default): snap + dedup + make_valid topology repair

ADR-0033: Tier 2 CuPy for element-wise coordinate quantization (round/multiply).
  Reuses existing Tier 1 NVRTC remove_repeated_points kernel for dedup.
  Reuses existing make_valid_owned pipeline for topology repair.
ADR-0002: CONSTRUCTIVE class -- stays fp64 on all devices per policy.
  PrecisionPlan wired through for observability only.
"""

from __future__ import annotations

import logging

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

from vibespatial.constructive.set_precision_cpu import _set_precision_cpu
from vibespatial.geometry.owned import (
    DeviceFamilyGeometryBuffer,
    OwnedGeometryArray,
    build_device_resident_owned,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1: CuPy coordinate quantization (Tier 2 element-wise)
# ---------------------------------------------------------------------------

def _quantize_family_coords(device_buf: DeviceFamilyGeometryBuffer, grid_size: float):
    """Snap all coordinates in a family buffer to grid_size multiples.

    Pure CuPy element-wise: snapped = round(coord / grid_size) * grid_size.
    All operations stay on device -- zero D2H transfers.

    Returns a new DeviceFamilyGeometryBuffer with quantized x/y and
    invalidated bounds (since coordinates moved).
    """
    inv_grid = 1.0 / grid_size

    d_x = cp.asarray(device_buf.x)
    d_y = cp.asarray(device_buf.y)

    # CuPy round-to-nearest, then scale back (matches Shapely/GEOS behaviour)
    d_x_snapped = cp.round(d_x * inv_grid) * grid_size
    d_y_snapped = cp.round(d_y * inv_grid) * grid_size

    return DeviceFamilyGeometryBuffer(
        family=device_buf.family,
        x=d_x_snapped,
        y=d_y_snapped,
        geometry_offsets=device_buf.geometry_offsets,
        empty_mask=device_buf.empty_mask,
        part_offsets=device_buf.part_offsets,
        ring_offsets=device_buf.ring_offsets,
        bounds=None,  # invalidated — coordinates changed
    )


def _quantize_owned(owned: OwnedGeometryArray, grid_size: float) -> OwnedGeometryArray:
    """Quantize all coordinates in an OwnedGeometryArray to grid_size multiples.

    Point/MultiPoint families are quantized like any other family.
    Returns a new device-resident OwnedGeometryArray.
    """
    d_state = owned._ensure_device_state()

    new_device_families = {}
    for family, device_buf in d_state.families.items():
        coord_count = int(device_buf.x.shape[0]) if device_buf.x is not None else 0
        if coord_count == 0:
            new_device_families[family] = device_buf
            continue
        new_device_families[family] = _quantize_family_coords(device_buf, grid_size)

    return build_device_resident_owned(
        device_families=new_device_families,
        row_count=owned.row_count,
        tags=owned.tags.copy(),
        validity=owned.validity.copy(),
        family_row_offsets=owned.family_row_offsets.copy(),
    )


# ---------------------------------------------------------------------------
# GPU dispatch variant (registered)
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "set_precision",
    "gpu-cuda-python",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=(
        "point", "multipoint", "linestring", "multilinestring",
        "polygon", "multipolygon",
    ),
    supports_mixed=True,
    tags=("cuda-python", "constructive", "set_precision"),
)
def _set_precision_gpu(owned: OwnedGeometryArray, grid_size: float, mode: str):
    """GPU set_precision -- returns device-resident OwnedGeometryArray.

    Pipeline:
      pointwise:      quantize only
      keep_collapsed: quantize + remove_repeated_points(tolerance=0)
      valid_output:   quantize + remove_repeated_points(tolerance=0) + make_valid
    """
    # Stage 1: Grid quantization (Tier 2 CuPy element-wise)
    quantized = _quantize_owned(owned, grid_size)

    if mode == "pointwise":
        return quantized

    # Stage 2: Deduplication — reuse existing GPU kernel
    from vibespatial.constructive.remove_repeated_points import (
        remove_repeated_points_owned,
    )

    deduped = remove_repeated_points_owned(
        quantized, tolerance=0.0, dispatch_mode=ExecutionMode.GPU,
    )

    if mode == "keep_collapsed":
        return deduped

    # Stage 3: Topology repair (valid_output) — reuse existing make_valid pipeline
    from vibespatial.constructive.make_valid_pipeline import make_valid_owned

    mv_result = make_valid_owned(owned=deduped, dispatch_mode=ExecutionMode.GPU)
    if mv_result.owned is not None:
        return mv_result.owned
    # make_valid should always produce OGA for device-resident input.
    # If it returns Shapely objects, that indicates a bug in the pipeline --
    # raise instead of silently ingesting a D2H+H2D roundtrip.
    raise RuntimeError(
        "make_valid_owned returned Shapely objects for device-resident input "
        "in set_precision GPU path; this is unexpected and indicates a "
        "pipeline bug"
    )


# ---------------------------------------------------------------------------
# CPU dispatch variant (registered)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Public dispatch API
# ---------------------------------------------------------------------------

_VALID_MODES = {"valid_output", "pointwise", "keep_collapsed"}


def set_precision_owned(
    owned: OwnedGeometryArray,
    grid_size: float,
    mode: str = "valid_output",
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """Set precision (grid snap + topology cleanup) for geometries.

    Snaps all coordinates to multiples of ``grid_size`` and optionally
    removes degenerate features and repairs topology.

    Parameters
    ----------
    owned : OwnedGeometryArray
        Input geometries.
    grid_size : float
        Grid resolution. Coordinates are snapped to the nearest multiple.
        If 0, returns the input unchanged (Shapely-compatible no-op).
    mode : str
        One of ``"valid_output"`` (default), ``"pointwise"``, or
        ``"keep_collapsed"``.
    dispatch_mode : ExecutionMode
        Execution mode selection (AUTO / CPU / GPU).
    precision : PrecisionMode
        Precision mode selection (AUTO / FP32 / FP64).

    Returns
    -------
    OwnedGeometryArray
        New geometry array with snapped coordinates.
    """
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Invalid mode {mode!r}. Must be one of {sorted(_VALID_MODES)}."
        )

    row_count = owned.row_count
    if row_count == 0:
        return owned

    # grid_size=0 is a no-op (Shapely-compatible)
    if grid_size == 0:
        return owned

    selection = plan_dispatch_selection(
        kernel_name="set_precision",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=row_count,
        requested_mode=dispatch_mode,
        requested_precision=precision,
    )

    if selection.selected is ExecutionMode.GPU:
        # Precision plan for observability (ADR-0002 CONSTRUCTIVE -> fp64)
        _precision_plan = selection.precision_plan
        try:
            result = _set_precision_gpu(owned, grid_size, mode)
        except Exception:
            logger.warning(
                "GPU set_precision failed, falling back to CPU",
                exc_info=True,
            )
        else:
            record_dispatch_event(
                surface="geopandas.array.set_precision",
                operation="set_precision",
                implementation="gpu_cupy_quantize",
                reason=f"GPU CuPy quantization + {mode} pipeline",
                detail=(
                    f"rows={row_count}, grid_size={grid_size}, "
                    f"mode={mode}, precision=fp64"
                ),
                requested=dispatch_mode,
                selected=ExecutionMode.GPU,
            )
            return result

    result = _set_precision_cpu(owned, grid_size, mode)
    record_dispatch_event(
        surface="geopandas.array.set_precision",
        operation="set_precision",
        implementation="cpu_shapely",
        reason="CPU fallback via shapely.set_precision",
        detail=f"rows={row_count}, grid_size={grid_size}, mode={mode}",
        requested=dispatch_mode,
        selected=ExecutionMode.CPU,
    )
    return result
