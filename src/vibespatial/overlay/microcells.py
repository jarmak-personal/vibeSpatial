from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vibespatial.constructive.point import point_owned_from_xy_device
from vibespatial.geometry.owned import OwnedGeometryArray
from vibespatial.kernels.predicates.point_in_polygon import point_in_polygon
from vibespatial.overlay.contraction_types import (
    AlignedOverlayWorkload,
    ContractionMicrocellSummary,
    OverlayContractionSummary,
    RowMicrocellBand,
)
from vibespatial.predicates.binary import (
    _broadcast_right_owned,
    evaluate_geopandas_binary_predicate,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.spatial.indexing import generate_bounds_pairs
from vibespatial.spatial.segment_primitives import (
    SegmentIntersectionKind,
    SegmentTable,
    _extract_segments_gpu,
    _segment_row_spans,
    classify_segment_intersections,
    extract_segments,
    get_cuda_runtime,
    summarize_exact_local_events,
)

from ._host_boundary import overlay_bool_scalar, overlay_device_to_host, overlay_int_scalar

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    cp = None


@dataclass(frozen=True)
class OverlayMicrocellBands:
    row_indices: Any
    interval_indices: Any
    lower_segment_ids: Any
    upper_segment_ids: Any
    x_left: Any
    x_right: Any
    y_lower_left: Any
    y_lower_right: Any
    y_upper_left: Any
    y_upper_right: Any
    representative_x: Any
    representative_y: Any

    @property
    def count(self) -> int:
        return int(self.row_indices.size)

    @property
    def row_count(self) -> int:
        if self.count == 0:
            return 0
        if cp is not None and hasattr(self.row_indices, "__cuda_array_interface__"):
            return (
                overlay_int_scalar(
                    cp.max(self.row_indices),
                    reason="overlay microcells row-count metadata fence",
                )
                + 1
            )
        return int(np.max(self.row_indices, initial=-1)) + 1

    @property
    def representative_points(self) -> OwnedGeometryArray:
        return point_owned_from_xy_device(self.representative_x, self.representative_y)


@dataclass(frozen=True)
class OverlayMicrocellLabels:
    bands: OverlayMicrocellBands
    left_inside: Any
    right_inside: Any

    @property
    def count(self) -> int:
        return self.bands.count


def build_aligned_overlay_workload(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> AlignedOverlayWorkload:
    bbox_pairs = generate_bounds_pairs(left, right)
    left_indices = bbox_pairs.left_indices.astype(np.int64, copy=False)
    right_indices = bbox_pairs.right_indices.astype(np.int64, copy=False)
    if left_indices.size == 0:
        return AlignedOverlayWorkload(
            left_aligned=left.take(left_indices),
            right_aligned=right.take(right_indices),
            left_indices=left_indices,
            right_indices=right_indices,
        )

    left_candidate = left.take(left_indices)
    right_candidate = right.take(right_indices)
    exact_mask = evaluate_geopandas_binary_predicate("intersects", left_candidate, right_candidate)
    if exact_mask is None:
        raise RuntimeError(
            "repo-owned exact intersects predicate unavailable for contraction workload alignment"
        )
    keep = np.flatnonzero(np.asarray(exact_mask, dtype=bool))
    left_indices = left_indices[keep]
    right_indices = right_indices[keep]
    if left_indices.size:
        order = np.lexsort((right_indices, left_indices))
        left_indices = left_indices[order]
        right_indices = right_indices[order]
    return AlignedOverlayWorkload(
        left_aligned=left.take(left_indices),
        right_aligned=right.take(right_indices),
        left_indices=left_indices,
        right_indices=right_indices,
    )


def _segment_y_at_x(
    x0: np.ndarray,
    y0: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    x: float,
) -> np.ndarray:
    dx = x1 - x0
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (x - x0) / dx
        y = y0 + t * (y1 - y0)
    return y


def _extract_segments_for_microcells(
    geometry_array: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str,
) -> SegmentTable:
    mode = dispatch_mode
    if isinstance(mode, ExecutionMode):
        is_gpu = mode is ExecutionMode.GPU
    else:
        is_gpu = str(mode).lower() == "gpu"
    if cp is None or not is_gpu:
        return extract_segments(geometry_array)

    runtime = get_cuda_runtime()
    device_segments = _extract_segments_gpu(geometry_array)
    try:
        row_indices = np.asarray(
            runtime.copy_device_to_host(
                device_segments.row_indices,
                reason="overlay microcells segment row-index host export",
            ),
            dtype=np.int32,
        )
        part_indices = np.asarray(
            runtime.copy_device_to_host(
                device_segments.part_indices,
                reason="overlay microcells segment part-index host export",
            ),
            dtype=np.int32,
        )
        ring_indices = np.asarray(
            runtime.copy_device_to_host(
                device_segments.ring_indices,
                reason="overlay microcells segment ring-index host export",
            ),
            dtype=np.int32,
        )
        segment_indices = np.asarray(
            runtime.copy_device_to_host(
                device_segments.segment_indices,
                reason="overlay microcells segment-index host export",
            ),
            dtype=np.int32,
        )
        x0 = np.asarray(
            runtime.copy_device_to_host(
                device_segments.x0,
                reason="overlay microcells segment x0 host export",
            ),
            dtype=np.float64,
        )
        y0 = np.asarray(
            runtime.copy_device_to_host(
                device_segments.y0,
                reason="overlay microcells segment y0 host export",
            ),
            dtype=np.float64,
        )
        x1 = np.asarray(
            runtime.copy_device_to_host(
                device_segments.x1,
                reason="overlay microcells segment x1 host export",
            ),
            dtype=np.float64,
        )
        y1 = np.asarray(
            runtime.copy_device_to_host(
                device_segments.y1,
                reason="overlay microcells segment y1 host export",
            ),
            dtype=np.float64,
        )
    finally:
        device_segments.free()

    bounds = np.column_stack(
        (
            np.minimum(x0, x1),
            np.minimum(y0, y1),
            np.maximum(x0, x1),
            np.maximum(y0, y1),
        )
    )
    return SegmentTable(
        row_indices=row_indices,
        part_indices=part_indices,
        ring_indices=ring_indices,
        segment_indices=segment_indices,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        bounds=bounds,
    )


def _row_event_x_sets(
    left_segments,
    right_segments,
    point_rows: np.ndarray,
    point_x: np.ndarray,
) -> list[np.ndarray]:
    row_sets = [set() for _ in range(left_segments.row_indices.max(initial=-1) + 1)]

    for row_idx in range(len(row_sets)):
        left_mask = left_segments.row_indices == row_idx
        right_mask = right_segments.row_indices == row_idx
        row_sets[row_idx].update(float(x) for x in left_segments.x0[left_mask])
        row_sets[row_idx].update(float(x) for x in left_segments.x1[left_mask])
        row_sets[row_idx].update(float(x) for x in right_segments.x0[right_mask])
        row_sets[row_idx].update(float(x) for x in right_segments.x1[right_mask])

    if point_x.size > 0:
        for row_idx, x in zip(point_rows, point_x):
            row_sets[int(row_idx)].add(float(x))

    return [np.asarray(sorted(values), dtype=np.float64) for values in row_sets]


def _microcell_summary_from_segments(
    left_segments,
    right_segments,
    x_event_sets: list[np.ndarray],
) -> ContractionMicrocellSummary:
    row_bands: list[RowMicrocellBand] = []

    for row_idx, event_x in enumerate(x_event_sets):
        interval_count = max(int(event_x.size) - 1, 0)
        if interval_count <= 0:
            row_bands.append(
                RowMicrocellBand(
                    row_index=row_idx,
                    event_x=event_x,
                    interval_count=0,
                    max_active_segment_count=0,
                    mean_active_segment_count=0.0,
                    microcell_upper_bound=0,
                )
            )
            continue

        left_mask = left_segments.row_indices == row_idx
        right_mask = right_segments.row_indices == row_idx
        minx = np.concatenate([
            np.minimum(left_segments.x0[left_mask], left_segments.x1[left_mask]),
            np.minimum(right_segments.x0[right_mask], right_segments.x1[right_mask]),
        ])
        maxx = np.concatenate([
            np.maximum(left_segments.x0[left_mask], left_segments.x1[left_mask]),
            np.maximum(right_segments.x0[right_mask], right_segments.x1[right_mask]),
        ])

        start = np.searchsorted(event_x, minx, side="right") - 1
        end = np.searchsorted(event_x, maxx, side="left") - 1
        valid = (start >= 0) & (end >= start) & (start < interval_count)
        if not np.any(valid):
            row_bands.append(
                RowMicrocellBand(
                    row_index=row_idx,
                    event_x=event_x,
                    interval_count=interval_count,
                    max_active_segment_count=0,
                    mean_active_segment_count=0.0,
                    microcell_upper_bound=0,
                )
            )
            continue

        start_valid = start[valid].astype(np.int64, copy=False)
        end_valid = np.minimum(end[valid], interval_count - 1).astype(np.int64, copy=False)
        delta = np.zeros(interval_count + 1, dtype=np.int64)
        np.add.at(delta, start_valid, 1)
        np.add.at(delta, end_valid + 1, -1)
        active = np.cumsum(delta[:-1], dtype=np.int64)
        band_upper = np.maximum(active - 1, 0)
        row_bands.append(
            RowMicrocellBand(
                row_index=row_idx,
                event_x=event_x,
                interval_count=interval_count,
                max_active_segment_count=int(active.max(initial=0)),
                mean_active_segment_count=float(active.mean() if active.size else 0.0),
                microcell_upper_bound=int(band_upper.sum(dtype=np.int64)),
            )
        )

    return ContractionMicrocellSummary(row_bands=tuple(row_bands))


def _row_microcell_bands(
    row_index: int,
    left_segments,
    right_segments,
    point_x: np.ndarray,
    *,
    selection_operation: str | None = None,
) -> list[tuple[int, int, int, float, float, float, float, float, float, float, float, bool, bool]]:
    left_mask = left_segments.row_indices == row_index
    right_mask = right_segments.row_indices == row_index

    x0 = np.concatenate((left_segments.x0[left_mask], right_segments.x0[right_mask]))
    y0 = np.concatenate((left_segments.y0[left_mask], right_segments.y0[right_mask]))
    x1 = np.concatenate((left_segments.x1[left_mask], right_segments.x1[right_mask]))
    y1 = np.concatenate((left_segments.y1[left_mask], right_segments.y1[right_mask]))
    local_segment_ids = np.arange(x0.size, dtype=np.int32)
    left_segment_count = int(np.count_nonzero(left_mask))

    if x0.size == 0:
        return []

    event_x = np.unique(
        np.concatenate((x0, x1, point_x.astype(np.float64, copy=False)))
    )
    if event_x.size < 2:
        return []

    interval_count = int(event_x.size - 1)
    minx = np.minimum(x0, x1)
    maxx = np.maximum(x0, x1)
    start = np.searchsorted(event_x, minx, side="right") - 1
    end = np.searchsorted(event_x, maxx, side="left") - 1
    valid = (
        (start >= 0)
        & (end >= start)
        & (start < interval_count)
        & (maxx > minx)
    )
    if not np.any(valid):
        return []

    starts: list[list[int]] = [[] for _ in range(interval_count)]
    ends: list[list[int]] = [[] for _ in range(interval_count + 1)]
    start_valid = start[valid].astype(np.int64, copy=False)
    end_valid = np.minimum(end[valid], interval_count - 1).astype(np.int64, copy=False)
    ids_valid = local_segment_ids[valid]
    for idx in range(ids_valid.size):
        seg_id = int(ids_valid[idx])
        start_idx = int(start_valid[idx])
        end_idx = int(end_valid[idx])
        starts[start_idx].append(seg_id)
        ends[end_idx + 1].append(seg_id)

    active: set[int] = set()
    bands: list[tuple[int, int, int, float, float, float, float, float, float, float, float, bool, bool]] = []
    for interval_index in range(interval_count):
        for seg_id in starts[interval_index]:
            active.add(seg_id)
        for seg_id in ends[interval_index]:
            active.discard(seg_id)

        if len(active) < 2:
            continue

        x_left = float(event_x[interval_index])
        x_right = float(event_x[interval_index + 1])
        if not np.isfinite(x_left) or not np.isfinite(x_right) or x_right <= x_left:
            continue

        x_mid = 0.5 * (x_left + x_right)
        active_ids = np.asarray(sorted(active), dtype=np.int32)
        y_mid = _segment_y_at_x(x0[active_ids], y0[active_ids], x1[active_ids], y1[active_ids], x_mid)
        if y_mid.size < 2:
            continue
        order = np.argsort(y_mid, kind="stable")
        sorted_ids = active_ids[order]
        sorted_mid_y = y_mid[order]
        y_left = _segment_y_at_x(x0[sorted_ids], y0[sorted_ids], x1[sorted_ids], y1[sorted_ids], x_left)
        y_right = _segment_y_at_x(x0[sorted_ids], y0[sorted_ids], x1[sorted_ids], y1[sorted_ids], x_right)
        is_left = sorted_ids < left_segment_count
        left_parity = np.cumsum(is_left, dtype=np.int32) & 1
        right_parity = np.cumsum(~is_left, dtype=np.int32) & 1

        for band_index in range(sorted_ids.size - 1):
            y0_mid = float(sorted_mid_y[band_index])
            y1_mid = float(sorted_mid_y[band_index + 1])
            if not np.isfinite(y0_mid) or not np.isfinite(y1_mid) or y1_mid <= y0_mid:
                continue
            left_inside = bool(left_parity[band_index])
            right_inside = bool(right_parity[band_index])
            if selection_operation == "intersection" and not (left_inside and right_inside):
                continue
            if selection_operation == "union" and not (left_inside or right_inside):
                continue
            if selection_operation == "difference" and not (left_inside and not right_inside):
                continue
            if selection_operation == "symmetric_difference" and not (left_inside ^ right_inside):
                continue
            if selection_operation == "identity" and not left_inside:
                continue
            rep_y = 0.5 * (y0_mid + y1_mid)
            if not np.isfinite(rep_y):
                continue
            lower_seg = int(sorted_ids[band_index])
            upper_seg = int(sorted_ids[band_index + 1])
            bands.append(
                (
                    row_index,
                    interval_index,
                    lower_seg,
                    upper_seg,
                    x_left,
                    x_right,
                    float(y_left[band_index]),
                    float(y_right[band_index]),
                    float(y_left[band_index + 1]),
                    float(y_right[band_index + 1]),
                    rep_y,
                    left_inside,
                    right_inside,
                )
            )
    return bands


def build_overlay_microcell_bands(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.GPU,
) -> OverlayMicrocellBands:
    if left.row_count != right.row_count:
        raise ValueError("overlay microcell construction requires aligned left/right rows")

    left_segments = _extract_segments_for_microcells(left, dispatch_mode=dispatch_mode)
    right_segments = _extract_segments_for_microcells(right, dispatch_mode=dispatch_mode)
    if left.row_count == 0:
        empty_i32 = np.empty(0, dtype=np.int32)
        empty_f64 = np.empty(0, dtype=np.float64)
        return OverlayMicrocellBands(
            row_indices=empty_i32,
            interval_indices=empty_i32,
            lower_segment_ids=empty_i32,
            upper_segment_ids=empty_i32,
            x_left=empty_f64,
            x_right=empty_f64,
            y_lower_left=empty_f64,
            y_lower_right=empty_f64,
            y_upper_left=empty_f64,
            y_upper_right=empty_f64,
            representative_x=empty_f64,
            representative_y=empty_f64,
        )

    exact = classify_segment_intersections(
        left,
        right,
        dispatch_mode=dispatch_mode,
        _require_same_row=True,
    )
    point_mask = (
        np.asarray(exact.kinds, dtype=np.int8) == int(SegmentIntersectionKind.PROPER)
    ) & np.isfinite(exact.point_x) & np.isfinite(exact.point_y)
    point_rows = exact.left_rows[point_mask].astype(np.int32, copy=False)
    point_x = exact.point_x[point_mask].astype(np.float64, copy=False)

    row_bands: list[tuple[int, int, int, int, float, float, float, float, float, float, float]] = []
    for row_index in range(left.row_count):
        row_point_x = point_x[point_rows == row_index]
        row_bands.extend(
            _row_microcell_bands(row_index, left_segments, right_segments, row_point_x)
        )

    if not row_bands:
        empty_i32 = np.empty(0, dtype=np.int32)
        empty_f64 = np.empty(0, dtype=np.float64)
        return OverlayMicrocellBands(
            row_indices=empty_i32,
            interval_indices=empty_i32,
            lower_segment_ids=empty_i32,
            upper_segment_ids=empty_i32,
            x_left=empty_f64,
            x_right=empty_f64,
            y_lower_left=empty_f64,
            y_lower_right=empty_f64,
            y_upper_left=empty_f64,
            y_upper_right=empty_f64,
            representative_x=empty_f64,
            representative_y=empty_f64,
        )

    data = np.asarray([row[:11] for row in row_bands], dtype=np.float64)
    return OverlayMicrocellBands(
        row_indices=data[:, 0].astype(np.int32, copy=False),
        interval_indices=data[:, 1].astype(np.int32, copy=False),
        lower_segment_ids=data[:, 2].astype(np.int32, copy=False),
        upper_segment_ids=data[:, 3].astype(np.int32, copy=False),
        x_left=data[:, 4],
        x_right=data[:, 5],
        y_lower_left=data[:, 6],
        y_lower_right=data[:, 7],
        y_upper_left=data[:, 8],
        y_upper_right=data[:, 9],
        representative_x=0.5 * (data[:, 4] + data[:, 5]),
        representative_y=data[:, 10],
    )


def label_overlay_microcells(
    bands: OverlayMicrocellBands,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.GPU,
) -> OverlayMicrocellLabels:
    if cp is None:
        raise RuntimeError("CuPy is required for GPU microcell labeling")

    if bands.count == 0:
        return OverlayMicrocellLabels(
            bands=bands,
            left_inside=cp.empty(0, dtype=cp.bool_),
            right_inside=cp.empty(0, dtype=cp.bool_),
        )

    points = bands.representative_points
    row_ids = cp.asarray(bands.row_indices.astype(np.int64, copy=False))
    left_inside = cp.empty(bands.count, dtype=cp.bool_)
    right_inside = cp.empty(bands.count, dtype=cp.bool_)
    unique_rows = cp.unique(row_ids).astype(cp.int64, copy=False)
    unique_rows_h = overlay_device_to_host(
        unique_rows,
        reason="overlay microcells unique-row metadata loop",
        dtype=np.int64,
    )
    for row_id in unique_rows_h:
        d_local = cp.flatnonzero(row_ids == int(row_id)).astype(cp.int64, copy=False)
        point_subset = points.take(d_local)
        d_row = cp.asarray([int(row_id)], dtype=cp.int64)
        left_row = left.take(d_row)
        right_row = right.take(d_row)
        left_broadcast = _broadcast_right_owned(left_row, int(d_local.size))
        right_broadcast = _broadcast_right_owned(right_row, int(d_local.size))
        left_inside[d_local] = point_in_polygon(
            point_subset,
            left_broadcast,
            dispatch_mode=dispatch_mode,
            _return_device=True,
        ).astype(cp.bool_, copy=False)
        right_inside[d_local] = point_in_polygon(
            point_subset,
            right_broadcast,
            dispatch_mode=dispatch_mode,
            _return_device=True,
        ).astype(cp.bool_, copy=False)

    return OverlayMicrocellLabels(
        bands=bands,
        left_inside=left_inside.astype(bool, copy=False),
        right_inside=right_inside.astype(bool, copy=False),
    )


def _segment_y_at_x_device(x0, y0, x1, y1, x):
    dx = x1 - x0
    t = (x - x0) / dx
    y = y0 + t * (y1 - y0)
    return y


def _empty_overlay_microcell_labels_device() -> OverlayMicrocellLabels:
    bands = OverlayMicrocellBands(
        row_indices=np.empty(0, dtype=np.int32),
        interval_indices=np.empty(0, dtype=np.int32),
        lower_segment_ids=np.empty(0, dtype=np.int32),
        upper_segment_ids=np.empty(0, dtype=np.int32),
        x_left=np.empty(0, dtype=np.float64),
        x_right=np.empty(0, dtype=np.float64),
        y_lower_left=np.empty(0, dtype=np.float64),
        y_lower_right=np.empty(0, dtype=np.float64),
        y_upper_left=np.empty(0, dtype=np.float64),
        y_upper_right=np.empty(0, dtype=np.float64),
        representative_x=np.empty(0, dtype=np.float64),
        representative_y=np.empty(0, dtype=np.float64),
    )
    if cp is None:
        return OverlayMicrocellLabels(
            bands=bands,
            left_inside=np.empty(0, dtype=bool),
            right_inside=np.empty(0, dtype=bool),
        )
    return OverlayMicrocellLabels(
        bands=bands,
        left_inside=cp.empty(0, dtype=cp.bool_),
        right_inside=cp.empty(0, dtype=cp.bool_),
    )


def _build_selected_row_microcell_arrays_device(
    row_index: int,
    *,
    x0,
    y0,
    x1,
    y1,
    left_segment_count: int,
    point_x,
    selection_operation: str | None,
) -> dict[str, Any] | None:
    if cp is None or int(x0.size) == 0:
        return None

    event_x = cp.unique(cp.concatenate((x0, x1, point_x.astype(cp.float64, copy=False))))
    if int(event_x.size) < 2:
        return None

    interval_count = int(event_x.size - 1)
    minx = cp.minimum(x0, x1)
    maxx = cp.maximum(x0, x1)
    start = cp.searchsorted(event_x, minx, side="right") - 1
    end = cp.searchsorted(event_x, maxx, side="left") - 1
    valid = (start >= 0) & (end >= start) & (start < interval_count) & (maxx > minx)
    if not overlay_bool_scalar(
        cp.any(valid),
        reason="overlay microcells selected-row valid-band admission fence",
    ):
        return None

    start_valid = start[valid].astype(cp.int64, copy=False)
    end_valid = cp.minimum(end[valid], interval_count - 1).astype(cp.int64, copy=False)
    seg_ids_valid = cp.arange(int(x0.size), dtype=cp.int32)[valid]
    span = (end_valid - start_valid + 1).astype(cp.int32, copy=False)
    total_memberships = overlay_int_scalar(
        cp.sum(span, dtype=cp.int64),
        reason="overlay microcells selected-row membership-count allocation fence",
    )
    if total_memberships == 0:
        return None

    span_offsets = cp.cumsum(span.astype(cp.int64), dtype=cp.int64) - span.astype(cp.int64)
    membership_ids = cp.arange(total_memberships, dtype=cp.int64)
    span_ends = span_offsets + span.astype(cp.int64)
    membership_sources = cp.searchsorted(span_ends, membership_ids, side="right")
    repeated_starts = start_valid[membership_sources]
    repeated_seg_ids = seg_ids_valid[membership_sources]
    repeated_offsets = span_offsets[membership_sources]
    local_rank = membership_ids - repeated_offsets
    interval_ids = repeated_starts + local_rank

    x_mid = 0.5 * (event_x[:-1] + event_x[1:])
    y_mid = _segment_y_at_x_device(
        x0[repeated_seg_ids],
        y0[repeated_seg_ids],
        x1[repeated_seg_ids],
        y1[repeated_seg_ids],
        x_mid[interval_ids],
    )
    order = cp.lexsort(cp.stack((y_mid, interval_ids)))
    interval_sorted = interval_ids[order]
    seg_sorted = repeated_seg_ids[order]
    y_mid_sorted = y_mid[order]
    if int(seg_sorted.size) < 2:
        return None

    same_next = interval_sorted[:-1] == interval_sorted[1:]
    if not overlay_bool_scalar(
        cp.any(same_next),
        reason="overlay microcells selected-row adjacent-band admission fence",
    ):
        return None

    interval_change = cp.empty(int(seg_sorted.size), dtype=cp.bool_)
    interval_change[0] = True
    if int(seg_sorted.size) > 1:
        interval_change[1:] = interval_sorted[1:] != interval_sorted[:-1]
    interval_starts = cp.flatnonzero(interval_change).astype(cp.int64, copy=False)
    interval_ends = cp.concatenate(
        (interval_starts[1:], cp.asarray([int(seg_sorted.size)], dtype=cp.int64))
    )
    interval_counts = interval_ends - interval_starts

    is_left_sorted = (seg_sorted < left_segment_count).astype(cp.int32, copy=False)
    is_right_sorted = (1 - is_left_sorted).astype(cp.int32, copy=False)
    left_cumsum = cp.cumsum(is_left_sorted, dtype=cp.int32)
    right_cumsum = cp.cumsum(is_right_sorted, dtype=cp.int32)
    left_prefix = cp.zeros(int(interval_starts.size), dtype=cp.int32)
    right_prefix = cp.zeros(int(interval_starts.size), dtype=cp.int32)
    if int(interval_starts.size) > 1:
        left_prefix[1:] = left_cumsum[interval_starts[1:] - 1]
        right_prefix[1:] = right_cumsum[interval_starts[1:] - 1]
    interval_membership_ids = cp.arange(int(seg_sorted.size), dtype=cp.int64)
    interval_ends = cp.cumsum(interval_counts, dtype=cp.int64)
    interval_sources = cp.searchsorted(interval_ends, interval_membership_ids, side="right")
    left_counts_per_pos = left_cumsum - left_prefix[interval_sources]
    right_counts_per_pos = right_cumsum - right_prefix[interval_sources]

    lower_pos = cp.flatnonzero(same_next).astype(cp.int64, copy=False)
    left_inside = (left_counts_per_pos[lower_pos] & 1).astype(cp.bool_, copy=False)
    right_inside = (right_counts_per_pos[lower_pos] & 1).astype(cp.bool_, copy=False)
    band_mid_span = (
        y_mid_sorted[1:][same_next] - y_mid_sorted[:-1][same_next]
    ) > 1e-12

    match selection_operation:
        case None:
            band_keep = cp.ones(int(lower_pos.size), dtype=cp.bool_)
        case "intersection":
            band_keep = left_inside & right_inside
        case "union":
            band_keep = left_inside | right_inside
        case "difference":
            band_keep = left_inside & ~right_inside
        case "symmetric_difference":
            band_keep = left_inside ^ right_inside
        case "identity":
            band_keep = left_inside
        case _:
            raise ValueError(f"unsupported selection operation: {selection_operation}")
    band_keep = band_keep & band_mid_span
    if not overlay_bool_scalar(
        cp.any(band_keep),
        reason="overlay microcells selected-row kept-band admission fence",
    ):
        return None

    band_intervals = interval_sorted[:-1][same_next][band_keep].astype(cp.int32, copy=False)
    lower_seg = seg_sorted[:-1][same_next][band_keep].astype(cp.int32, copy=False)
    upper_seg = seg_sorted[1:][same_next][band_keep].astype(cp.int32, copy=False)
    x_left = event_x[band_intervals]
    x_right = event_x[band_intervals.astype(cp.int64) + 1]

    y_lower_left = _segment_y_at_x_device(x0[lower_seg], y0[lower_seg], x1[lower_seg], y1[lower_seg], x_left)
    y_lower_right = _segment_y_at_x_device(x0[lower_seg], y0[lower_seg], x1[lower_seg], y1[lower_seg], x_right)
    y_upper_left = _segment_y_at_x_device(x0[upper_seg], y0[upper_seg], x1[upper_seg], y1[upper_seg], x_left)
    y_upper_right = _segment_y_at_x_device(x0[upper_seg], y0[upper_seg], x1[upper_seg], y1[upper_seg], x_right)
    representative_y = 0.5 * (
        y_mid_sorted[:-1][same_next][band_keep] + y_mid_sorted[1:][same_next][band_keep]
    )

    return {
        "row_index": row_index,
        "interval_indices": band_intervals,
        "lower_segment_ids": lower_seg,
        "upper_segment_ids": upper_seg,
        "x_left": x_left,
        "x_right": x_right,
        "y_lower_left": y_lower_left,
        "y_lower_right": y_lower_right,
        "y_upper_left": y_upper_left,
        "y_upper_right": y_upper_right,
        "representative_x": 0.5 * (x_left + x_right),
        "representative_y": representative_y,
        "left_inside": left_inside[band_keep],
        "right_inside": right_inside[band_keep],
    }


def _build_and_label_selected_overlay_microcells_device(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str,
    selection_operation: str | None,
) -> OverlayMicrocellLabels:
    if cp is None:
        return _empty_overlay_microcell_labels_device()

    left_device = _extract_segments_gpu(left)
    right_device = _extract_segments_gpu(right)
    try:
        intersections = classify_segment_intersections(
            left,
            right,
            dispatch_mode=dispatch_mode,
            _require_same_row=True,
            _cached_right_device_segments=right_device,
        )
        ds = intersections.device_state
        if ds is None:
            raise RuntimeError("GPU segment intersection result did not expose device_state")

        point_mask = cp.isfinite(cp.asarray(ds.point_x)) & cp.isfinite(cp.asarray(ds.point_y))
        point_rows = cp.asarray(ds.left_rows)[point_mask].astype(cp.int32, copy=False)
        point_x = cp.asarray(ds.point_x)[point_mask].astype(cp.float64, copy=False)

        left_row_ids, left_row_starts, left_row_ends = _segment_row_spans(left_device.row_indices)
        right_row_ids, right_row_starts, right_row_ends = _segment_row_spans(right_device.row_indices)
        left_row_ids_h = overlay_device_to_host(
            left_row_ids,
            reason="overlay microcells left row-span id metadata",
            dtype=np.int64,
        )
        left_row_starts_h = overlay_device_to_host(
            left_row_starts,
            reason="overlay microcells left row-span start metadata",
            dtype=np.int64,
        )
        left_row_ends_h = overlay_device_to_host(
            left_row_ends,
            reason="overlay microcells left row-span end metadata",
            dtype=np.int64,
        )
        right_row_ids_h = overlay_device_to_host(
            right_row_ids,
            reason="overlay microcells right row-span id metadata",
            dtype=np.int64,
        )
        right_row_starts_h = overlay_device_to_host(
            right_row_starts,
            reason="overlay microcells right row-span start metadata",
            dtype=np.int64,
        )
        right_row_ends_h = overlay_device_to_host(
            right_row_ends,
            reason="overlay microcells right row-span end metadata",
            dtype=np.int64,
        )
        right_span_by_row = {
            int(row): (int(start), int(end))
            for row, start, end in zip(right_row_ids_h, right_row_starts_h, right_row_ends_h, strict=False)
        }

        chunks: list[dict[str, Any]] = []
        for row, left_start, left_end in zip(left_row_ids_h, left_row_starts_h, left_row_ends_h, strict=False):
            right_span = right_span_by_row.get(int(row))
            if right_span is None:
                continue
            right_start, right_end = right_span
            row_point_x = point_x[point_rows == int(row)]
            chunk = _build_selected_row_microcell_arrays_device(
                int(row),
                x0=cp.concatenate(
                    (cp.asarray(left_device.x0)[left_start:left_end], cp.asarray(right_device.x0)[right_start:right_end])
                ),
                y0=cp.concatenate(
                    (cp.asarray(left_device.y0)[left_start:left_end], cp.asarray(right_device.y0)[right_start:right_end])
                ),
                x1=cp.concatenate(
                    (cp.asarray(left_device.x1)[left_start:left_end], cp.asarray(right_device.x1)[right_start:right_end])
                ),
                y1=cp.concatenate(
                    (cp.asarray(left_device.y1)[left_start:left_end], cp.asarray(right_device.y1)[right_start:right_end])
                ),
                left_segment_count=int(left_end - left_start),
                point_x=row_point_x,
                selection_operation=selection_operation,
            )
            if chunk is not None:
                chunks.append(chunk)

        if not chunks:
            return _empty_overlay_microcell_labels_device()

        def _cat(name: str):
            return cp.concatenate([chunk[name] for chunk in chunks])

        interval_indices = _cat("interval_indices")
        lower_segment_ids = _cat("lower_segment_ids")
        upper_segment_ids = _cat("upper_segment_ids")
        x_left = _cat("x_left")
        x_right = _cat("x_right")
        y_lower_left = _cat("y_lower_left")
        y_lower_right = _cat("y_lower_right")
        y_upper_left = _cat("y_upper_left")
        y_upper_right = _cat("y_upper_right")
        representative_x = _cat("representative_x")
        representative_y = _cat("representative_y")
        left_inside = _cat("left_inside").astype(cp.bool_, copy=False)
        right_inside = _cat("right_inside").astype(cp.bool_, copy=False)
        row_indices = cp.concatenate(
            [
                cp.full(int(chunk["interval_indices"].size), int(chunk["row_index"]), dtype=cp.int32)
                for chunk in chunks
            ]
        )

        bands = OverlayMicrocellBands(
            row_indices=row_indices,
            interval_indices=interval_indices,
            lower_segment_ids=lower_segment_ids,
            upper_segment_ids=upper_segment_ids,
            x_left=x_left,
            x_right=x_right,
            y_lower_left=y_lower_left,
            y_lower_right=y_lower_right,
            y_upper_left=y_upper_left,
            y_upper_right=y_upper_right,
            representative_x=representative_x,
            representative_y=representative_y,
        )
        return OverlayMicrocellLabels(
            bands=bands,
            left_inside=left_inside,
            right_inside=right_inside,
        )
    finally:
        left_device.free()
        right_device.free()


def build_and_label_overlay_microcells(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.GPU,
    selection_operation: str | None = None,
) -> OverlayMicrocellLabels:
    if left.row_count != right.row_count:
        raise ValueError("overlay microcell construction requires aligned left/right rows")
    mode = dispatch_mode
    if isinstance(mode, ExecutionMode):
        is_gpu = mode is ExecutionMode.GPU
    else:
        is_gpu = str(mode).lower() == "gpu"
    if cp is not None and is_gpu:
        return _build_and_label_selected_overlay_microcells_device(
            left,
            right,
            dispatch_mode=dispatch_mode,
            selection_operation=selection_operation,
        )

    left_segments = _extract_segments_for_microcells(left, dispatch_mode=dispatch_mode)
    right_segments = _extract_segments_for_microcells(right, dispatch_mode=dispatch_mode)
    if left.row_count == 0:
        bands = OverlayMicrocellBands(
            row_indices=np.empty(0, dtype=np.int32),
            interval_indices=np.empty(0, dtype=np.int32),
            lower_segment_ids=np.empty(0, dtype=np.int32),
            upper_segment_ids=np.empty(0, dtype=np.int32),
            x_left=np.empty(0, dtype=np.float64),
            x_right=np.empty(0, dtype=np.float64),
            y_lower_left=np.empty(0, dtype=np.float64),
            y_lower_right=np.empty(0, dtype=np.float64),
            y_upper_left=np.empty(0, dtype=np.float64),
            y_upper_right=np.empty(0, dtype=np.float64),
            representative_x=np.empty(0, dtype=np.float64),
            representative_y=np.empty(0, dtype=np.float64),
        )
        return OverlayMicrocellLabels(
            bands=bands,
            left_inside=cp.empty(0, dtype=cp.bool_) if cp is not None else np.empty(0, dtype=bool),
            right_inside=cp.empty(0, dtype=cp.bool_) if cp is not None else np.empty(0, dtype=bool),
        )

    exact = classify_segment_intersections(
        left,
        right,
        dispatch_mode=dispatch_mode,
        _require_same_row=True,
    )
    point_mask = (
        np.asarray(exact.kinds, dtype=np.int8) == int(SegmentIntersectionKind.PROPER)
    ) & np.isfinite(exact.point_x) & np.isfinite(exact.point_y)
    point_rows = exact.left_rows[point_mask].astype(np.int32, copy=False)
    point_x = exact.point_x[point_mask].astype(np.float64, copy=False)

    row_bands: list[tuple[int, int, int, int, float, float, float, float, float, float, float, bool, bool]] = []
    for row_index in range(left.row_count):
        row_point_x = point_x[point_rows == row_index]
        row_bands.extend(
            _row_microcell_bands(
                row_index,
                left_segments,
                right_segments,
                row_point_x,
                selection_operation=selection_operation,
            )
        )

    if not row_bands:
        bands = OverlayMicrocellBands(
            row_indices=np.empty(0, dtype=np.int32),
            interval_indices=np.empty(0, dtype=np.int32),
            lower_segment_ids=np.empty(0, dtype=np.int32),
            upper_segment_ids=np.empty(0, dtype=np.int32),
            x_left=np.empty(0, dtype=np.float64),
            x_right=np.empty(0, dtype=np.float64),
            y_lower_left=np.empty(0, dtype=np.float64),
            y_lower_right=np.empty(0, dtype=np.float64),
            y_upper_left=np.empty(0, dtype=np.float64),
            y_upper_right=np.empty(0, dtype=np.float64),
            representative_x=np.empty(0, dtype=np.float64),
            representative_y=np.empty(0, dtype=np.float64),
        )
        return OverlayMicrocellLabels(
            bands=bands,
            left_inside=cp.empty(0, dtype=cp.bool_) if cp is not None else np.empty(0, dtype=bool),
            right_inside=cp.empty(0, dtype=cp.bool_) if cp is not None else np.empty(0, dtype=bool),
        )

    data = np.asarray([row[:11] for row in row_bands], dtype=np.float64)
    left_inside = np.asarray([row[11] for row in row_bands], dtype=bool)
    right_inside = np.asarray([row[12] for row in row_bands], dtype=bool)
    bands = OverlayMicrocellBands(
        row_indices=data[:, 0].astype(np.int32, copy=False),
        interval_indices=data[:, 1].astype(np.int32, copy=False),
        lower_segment_ids=data[:, 2].astype(np.int32, copy=False),
        upper_segment_ids=data[:, 3].astype(np.int32, copy=False),
        x_left=data[:, 4],
        x_right=data[:, 5],
        y_lower_left=data[:, 6],
        y_lower_right=data[:, 7],
        y_upper_left=data[:, 8],
        y_upper_right=data[:, 9],
        representative_x=0.5 * (data[:, 4] + data[:, 5]),
        representative_y=data[:, 10],
    )
    if cp is not None:
        return OverlayMicrocellLabels(
            bands=bands,
            left_inside=cp.asarray(left_inside, dtype=cp.bool_),
            right_inside=cp.asarray(right_inside, dtype=cp.bool_),
        )
    return OverlayMicrocellLabels(
        bands=bands,
        left_inside=left_inside,
        right_inside=right_inside,
    )


def build_overlay_contraction_summary(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.GPU,
) -> OverlayContractionSummary:
    aligned = build_aligned_overlay_workload(left, right)
    left_segments = _extract_segments_for_microcells(aligned.left_aligned, dispatch_mode=dispatch_mode)
    right_segments = _extract_segments_for_microcells(aligned.right_aligned, dispatch_mode=dispatch_mode)
    exact_events = summarize_exact_local_events(
        aligned.left_aligned,
        aligned.right_aligned,
        dispatch_mode=dispatch_mode,
        _require_same_row=True,
    )
    point_rows = np.repeat(
        np.arange(aligned.row_count, dtype=np.int64),
        exact_events.row_point_intersection_counts.astype(np.int64, copy=False),
    )
    point_x = np.empty(exact_events.point_intersection_count, dtype=np.float64)
    if exact_events.point_intersection_count > 0:
        exact = classify_segment_intersections(
            aligned.left_aligned,
            aligned.right_aligned,
            dispatch_mode=dispatch_mode,
            _require_same_row=True,
        )
        point_mask = np.isfinite(exact.point_x) & np.isfinite(exact.point_y)
        point_rows = exact.left_rows[point_mask].astype(np.int64, copy=False)
        point_x = exact.point_x[point_mask].astype(np.float64, copy=False)

    x_event_sets = _row_event_x_sets(left_segments, right_segments, point_rows, point_x)
    microcells = _microcell_summary_from_segments(left_segments, right_segments, x_event_sets)
    return OverlayContractionSummary(
        workload=aligned,
        left_aligned_segment_count=int(left_segments.count),
        right_aligned_segment_count=int(right_segments.count),
        exact_events=exact_events,
        microcells=microcells,
    )
