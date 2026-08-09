from __future__ import annotations

from vibespatial.geometry.owned import OwnedGeometryArray
from vibespatial.overlay.microcells import OverlayMicrocellLabels
from vibespatial.runtime import ExecutionMode, RuntimeSelection

from .boundary_graph import (
    build_polygon_output_from_boundary_segments_gpu,
    microcell_boundary_segments_gpu,
)

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    cp = None


def _select_microcell_mask(labels: OverlayMicrocellLabels, operation: str):
    left_inside = cp.asarray(labels.left_inside, dtype=cp.bool_)
    right_inside = cp.asarray(labels.right_inside, dtype=cp.bool_)
    match operation:
        case "intersection":
            return left_inside & right_inside
        case "union":
            return left_inside | right_inside
        case "difference":
            return left_inside & ~right_inside
        case "symmetric_difference":
            return left_inside ^ right_inside
        case "identity":
            return left_inside
        case _:
            raise ValueError(f"unsupported contraction reconstruction operation: {operation}")


def reconstruct_overlay_from_microcells(
    labels: OverlayMicrocellLabels,
    operation: str,
    *,
    row_count: int | None = None,
    dispatch_mode: ExecutionMode | str = ExecutionMode.GPU,
) -> OwnedGeometryArray:
    """Reconstruct selected microcells through an exact native boundary graph.

    Boundary cancellation derives connectivity directly, so no component-id
    carrier or union-find pass is required.
    """
    if cp is None:
        raise RuntimeError("CuPy is required for contraction reconstruction")

    band_row_count = labels.bands.row_count if row_count is None else int(row_count)
    selected_ids = cp.flatnonzero(
        _select_microcell_mask(labels, operation).astype(cp.bool_, copy=False)
    ).astype(cp.int64, copy=False)
    bands = labels.bands
    boundary_rows, start_x, start_y, end_x, end_y = microcell_boundary_segments_gpu(
        cp.asarray(bands.row_indices, dtype=cp.int32)[selected_ids],
        cp.asarray(bands.x_left, dtype=cp.float64)[selected_ids],
        cp.asarray(bands.x_right, dtype=cp.float64)[selected_ids],
        cp.asarray(bands.y_lower_left, dtype=cp.float64)[selected_ids],
        cp.asarray(bands.y_lower_right, dtype=cp.float64)[selected_ids],
        cp.asarray(bands.y_upper_left, dtype=cp.float64)[selected_ids],
        cp.asarray(bands.y_upper_right, dtype=cp.float64)[selected_ids],
    )
    requested = (
        dispatch_mode if isinstance(dispatch_mode, ExecutionMode) else ExecutionMode(dispatch_mode)
    )
    return build_polygon_output_from_boundary_segments_gpu(
        start_x,
        start_y,
        end_x,
        end_y,
        row_indices=boundary_rows,
        row_count=band_row_count,
        runtime_selection=RuntimeSelection(
            requested=requested,
            selected=ExecutionMode.GPU,
            reason="selected microcell boundary graph reconstruction",
        ),
    )
