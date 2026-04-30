from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from vibespatial.overlay.microcells import OverlayMicrocellLabels

from ._host_boundary import overlay_bool_scalar, overlay_device_to_host, overlay_int_scalar

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    cp = None

logger = logging.getLogger(__name__)
_MAX_EDGE_COMPARE_CELLS = 8_000_000


@dataclass(frozen=True)
class OverlayMicrocellComponents:
    labels: OverlayMicrocellLabels
    component_ids: np.ndarray
    component_count: int

    @property
    def row_indices(self) -> np.ndarray:
        return self.labels.bands.row_indices


def _find(parent: np.ndarray, idx: int) -> int:
    while parent[idx] != idx:
        parent[idx] = parent[parent[idx]]
        idx = int(parent[idx])
    return idx


def _union(parent: np.ndarray, rank: np.ndarray, left: int, right: int) -> None:
    left_root = _find(parent, left)
    right_root = _find(parent, right)
    if left_root == right_root:
        return
    if rank[left_root] < rank[right_root]:
        parent[left_root] = right_root
    elif rank[left_root] > rank[right_root]:
        parent[right_root] = left_root
    else:
        parent[right_root] = left_root
        rank[left_root] += 1


def _to_host_array(arr) -> np.ndarray:
    if cp is not None and hasattr(arr, "__cuda_array_interface__"):
        return overlay_device_to_host(
            arr,
            reason="overlay contract host component assembly metadata",
        )
    return np.asarray(arr)


def _sorted_span_starts_ends_device(values):
    d_values = cp.asarray(values, dtype=cp.int32)
    n = int(d_values.size)
    if n == 0:
        empty = cp.empty(0, dtype=cp.int32)
        return empty, empty
    d_change = cp.empty(n, dtype=cp.bool_)
    d_change[0] = True
    if n > 1:
        d_change[1:] = d_values[1:] != d_values[:-1]
    d_starts = cp.flatnonzero(d_change).astype(cp.int32, copy=False)
    d_ends = cp.concatenate((d_starts[1:], cp.asarray([n], dtype=cp.int32)))
    return d_starts, d_ends


def _build_contraction_edges_device(
    labels: OverlayMicrocellLabels,
    *,
    vertical_tolerance: float = 1e-12,
) -> tuple:
    bands = labels.bands
    row_indices = cp.asarray(bands.row_indices, dtype=cp.int32)
    interval_indices = cp.asarray(bands.interval_indices, dtype=cp.int32)
    x_left = cp.asarray(bands.x_left, dtype=cp.float64)
    x_right = cp.asarray(bands.x_right, dtype=cp.float64)
    y_lower_left = cp.asarray(bands.y_lower_left, dtype=cp.float64)
    y_upper_left = cp.asarray(bands.y_upper_left, dtype=cp.float64)
    y_lower_right = cp.asarray(bands.y_lower_right, dtype=cp.float64)
    y_upper_right = cp.asarray(bands.y_upper_right, dtype=cp.float64)
    left_inside = cp.asarray(labels.left_inside, dtype=cp.bool_)
    right_inside = cp.asarray(labels.right_inside, dtype=cp.bool_)

    count = int(row_indices.size)
    if count == 0:
        empty = cp.empty(0, dtype=cp.int32)
        return empty, empty

    group_breaks = cp.empty(count, dtype=cp.bool_)
    group_breaks[0] = True
    if count > 1:
        group_breaks[1:] = (
            (row_indices[1:] != row_indices[:-1])
            | (interval_indices[1:] != interval_indices[:-1])
        )
    group_starts = cp.flatnonzero(group_breaks).astype(cp.int32, copy=False)
    group_ends = cp.concatenate((group_starts[1:], cp.asarray([count], dtype=cp.int32)))
    if int(group_starts.size) <= 1:
        empty = cp.empty(0, dtype=cp.int32)
        return empty, empty

    group_rows = row_indices[group_starts]
    group_intervals = interval_indices[group_starts]
    group_x_left = x_left[group_starts]
    group_x_right = x_right[group_starts]
    adjacent_group_mask = (
        (group_rows[1:] == group_rows[:-1])
        & (group_intervals[1:] == (group_intervals[:-1] + 1))
        & (cp.abs(group_x_right[:-1] - group_x_left[1:]) <= vertical_tolerance)
    )
    left_group_ids = cp.flatnonzero(adjacent_group_mask).astype(cp.int32, copy=False)
    if int(left_group_ids.size) == 0:
        empty = cp.empty(0, dtype=cp.int32)
        return empty, empty
    right_group_ids = left_group_ids + 1
    left_group_starts = group_starts[left_group_ids]
    left_group_ends = group_ends[left_group_ids]
    right_group_starts = group_starts[right_group_ids]
    right_group_ends = group_ends[right_group_ids]
    left_sizes = (left_group_ends - left_group_starts).astype(cp.int32, copy=False)
    right_sizes = (right_group_ends - right_group_starts).astype(cp.int32, copy=False)
    max_left = overlay_int_scalar(
        cp.max(left_sizes),
        reason="overlay contract left component batch-size fence",
    )
    max_right = overlay_int_scalar(
        cp.max(right_sizes),
        reason="overlay contract right component batch-size fence",
    )
    if max_left <= 0 or max_right <= 0:
        empty = cp.empty(0, dtype=cp.int32)
        return empty, empty

    pair_count = int(left_group_ids.size)
    compare_cells_per_pair = max_left * max_right
    pairs_per_batch = max(1, _MAX_EDGE_COMPARE_CELLS // max(compare_cells_per_pair, 1))

    left_positions = cp.arange(max_left, dtype=cp.int32)
    right_positions = cp.arange(max_right, dtype=cp.int32)
    edge_left_parts: list = []
    edge_right_parts: list = []

    for batch_start in range(0, pair_count, pairs_per_batch):
        batch_end = min(batch_start + pairs_per_batch, pair_count)
        batch_slice = slice(batch_start, batch_end)

        batch_left_starts = left_group_starts[batch_slice]
        batch_left_sizes = left_sizes[batch_slice]
        batch_right_starts = right_group_starts[batch_slice]
        batch_right_sizes = right_sizes[batch_slice]

        left_valid = left_positions[None, :] < batch_left_sizes[:, None]
        right_valid = right_positions[None, :] < batch_right_sizes[:, None]
        left_indices = batch_left_starts[:, None] + left_positions[None, :]
        right_indices = batch_right_starts[:, None] + right_positions[None, :]

        safe_left_indices = cp.where(left_valid, left_indices, 0).astype(cp.int32, copy=False)
        safe_right_indices = cp.where(right_valid, right_indices, 0).astype(cp.int32, copy=False)

        left_lo = y_lower_right[safe_left_indices]
        left_hi = y_upper_right[safe_left_indices]
        right_lo = y_lower_left[safe_right_indices]
        right_hi = y_upper_left[safe_right_indices]
        left_left_inside = left_inside[safe_left_indices]
        left_right_inside = right_inside[safe_left_indices]
        right_left_inside = left_inside[safe_right_indices]
        right_right_inside = right_inside[safe_right_indices]

        valid_pairs = left_valid[:, :, None] & right_valid[:, None, :]
        same_left_inside = left_left_inside[:, :, None] == right_left_inside[:, None, :]
        same_right_inside = left_right_inside[:, :, None] == right_right_inside[:, None, :]
        overlap = (
            cp.minimum(left_hi[:, :, None], right_hi[:, None, :])
            - cp.maximum(left_lo[:, :, None], right_lo[:, None, :])
        ) > vertical_tolerance
        matches = valid_pairs & same_left_inside & same_right_inside & overlap
        if not overlay_bool_scalar(
            cp.any(matches),
            reason="overlay contract component-edge match admission fence",
        ):
            continue

        batch_ids, left_match_ids, right_match_ids = cp.nonzero(matches)
        edge_left_parts.append(
            safe_left_indices[batch_ids, left_match_ids].astype(cp.int32, copy=False)
        )
        edge_right_parts.append(
            safe_right_indices[batch_ids, right_match_ids].astype(cp.int32, copy=False)
        )

    if not edge_left_parts:
        empty = cp.empty(0, dtype=cp.int32)
        return empty, empty

    return (
        cp.concatenate(edge_left_parts).astype(cp.int32, copy=False),
        cp.concatenate(edge_right_parts).astype(cp.int32, copy=False),
    )


def _contract_component_ids_device(
    count: int,
    edge_left,
    edge_right,
    *,
    max_iterations: int = 128,
):
    labels = cp.arange(count, dtype=cp.int32)
    edge_count = int(edge_left.size)
    if count == 0 or edge_count == 0:
        return labels, count

    edge_left = cp.asarray(edge_left, dtype=cp.int32)
    edge_right = cp.asarray(edge_right, dtype=cp.int32)
    converged = False

    for _ in range(min(max_iterations, max(count, 1))):
        next_labels = labels.copy()
        cp.minimum.at(next_labels, edge_left, labels[edge_right])
        cp.minimum.at(next_labels, edge_right, labels[edge_left])
        next_labels = cp.minimum(
            next_labels,
            next_labels[next_labels.astype(cp.int64, copy=False)],
        )
        if overlay_bool_scalar(
            cp.all(next_labels == labels),
            reason="overlay contract component-label convergence fence",
        ):
            labels = next_labels
            converged = True
            break
        labels = next_labels

    if not converged:
        raise RuntimeError(
            "device contraction label propagation did not converge within "
            f"{min(max_iterations, max(count, 1))} iterations"
        )

    roots = labels.astype(cp.int64, copy=False)
    compressed = cp.minimum(roots, roots[roots])
    while not overlay_bool_scalar(
        cp.all(compressed == roots),
        reason="overlay contract component-root compression convergence fence",
    ):
        roots = compressed
        compressed = cp.minimum(roots, roots[roots])

    unique_roots, inverse = cp.unique(compressed, return_inverse=True)
    return inverse.astype(cp.int32, copy=False), int(unique_roots.size)


def _contract_overlay_microcells_host(
    labels: OverlayMicrocellLabels,
    *,
    vertical_tolerance: float = 1e-12,
) -> OverlayMicrocellComponents:
    count = labels.count
    if count == 0:
        return OverlayMicrocellComponents(
            labels=labels,
            component_ids=np.empty(0, dtype=np.int32),
            component_count=0,
        )

    bands = labels.bands
    row_indices = _to_host_array(bands.row_indices).astype(np.int32, copy=False)
    interval_indices = _to_host_array(bands.interval_indices).astype(np.int32, copy=False)
    x_left = _to_host_array(bands.x_left).astype(np.float64, copy=False)
    x_right = _to_host_array(bands.x_right).astype(np.float64, copy=False)
    y_lower_left = _to_host_array(bands.y_lower_left).astype(np.float64, copy=False)
    y_upper_left = _to_host_array(bands.y_upper_left).astype(np.float64, copy=False)
    y_lower_right = _to_host_array(bands.y_lower_right).astype(np.float64, copy=False)
    y_upper_right = _to_host_array(bands.y_upper_right).astype(np.float64, copy=False)
    left_inside = _to_host_array(labels.left_inside).astype(bool, copy=False)
    right_inside = _to_host_array(labels.right_inside).astype(bool, copy=False)

    parent = np.arange(count, dtype=np.int32)
    rank = np.zeros(count, dtype=np.int8)

    row_breaks = np.flatnonzero(np.diff(row_indices)) + 1
    row_starts = np.concatenate(([0], row_breaks))
    row_ends = np.concatenate((row_breaks, [count]))

    for row_start, row_end in zip(row_starts, row_ends, strict=False):
        row_interval_ids = interval_indices[row_start:row_end]
        if row_interval_ids.size == 0:
            continue
        interval_breaks = np.flatnonzero(np.diff(row_interval_ids)) + 1
        interval_starts = row_start + np.concatenate(([0], interval_breaks))
        interval_ends = row_start + np.concatenate((interval_breaks, [row_end - row_start]))

        for local_idx in range(len(interval_starts) - 1):
            current_start = int(interval_starts[local_idx])
            current_end = int(interval_ends[local_idx])
            next_start = int(interval_starts[local_idx + 1])
            next_end = int(interval_ends[local_idx + 1])
            if interval_indices[next_start] != interval_indices[current_start] + 1:
                continue
            if abs(x_right[current_start] - x_left[next_start]) > vertical_tolerance:
                continue

            left_ids = np.arange(current_start, current_end, dtype=np.int32)
            right_ids = np.arange(next_start, next_end, dtype=np.int32)
            left_lo = y_lower_right[current_start:current_end]
            left_hi = y_upper_right[current_start:current_end]
            right_lo = y_lower_left[next_start:next_end]
            right_hi = y_upper_left[next_start:next_end]
            left_left_inside = left_inside[current_start:current_end]
            left_right_inside = right_inside[current_start:current_end]
            right_left_inside = left_inside[next_start:next_end]
            right_right_inside = right_inside[next_start:next_end]

            i = 0
            j = 0
            while i < left_ids.size and j < right_ids.size:
                if left_hi[i] <= right_lo[j] + vertical_tolerance:
                    i += 1
                    continue
                if right_hi[j] <= left_lo[i] + vertical_tolerance:
                    j += 1
                    continue

                if (
                    left_left_inside[i] == right_left_inside[j]
                    and left_right_inside[i] == right_right_inside[j]
                ):
                    overlap = min(left_hi[i], right_hi[j]) - max(left_lo[i], right_lo[j])
                    if overlap > vertical_tolerance:
                        _union(parent, rank, int(left_ids[i]), int(right_ids[j]))

                if left_hi[i] < right_hi[j] - vertical_tolerance:
                    i += 1
                elif right_hi[j] < left_hi[i] - vertical_tolerance:
                    j += 1
                else:
                    i += 1
                    j += 1

    canonical: dict[int, int] = {}
    component_ids = np.empty(count, dtype=np.int32)
    next_component = 0
    for idx in range(count):
        root = _find(parent, idx)
        if root not in canonical:
            canonical[root] = next_component
            next_component += 1
        component_ids[idx] = canonical[root]

    return OverlayMicrocellComponents(
        labels=labels,
        component_ids=component_ids,
        component_count=next_component,
    )


def contract_overlay_microcells(
    labels: OverlayMicrocellLabels,
    *,
    vertical_tolerance: float = 1e-12,
) -> OverlayMicrocellComponents:
    count = labels.count
    if count == 0:
        return OverlayMicrocellComponents(
            labels=labels,
            component_ids=np.empty(0, dtype=np.int32),
            component_count=0,
        )

    if (
        cp is not None
        and hasattr(labels.bands.row_indices, "__cuda_array_interface__")
        and hasattr(labels.left_inside, "__cuda_array_interface__")
        and hasattr(labels.right_inside, "__cuda_array_interface__")
    ):
        try:
            edge_left, edge_right = _build_contraction_edges_device(
                labels,
                vertical_tolerance=vertical_tolerance,
            )
            component_ids, component_count = _contract_component_ids_device(
                count,
                edge_left,
                edge_right,
            )
            return OverlayMicrocellComponents(
                labels=labels,
                component_ids=component_ids,
                component_count=component_count,
            )
        except Exception:
            logger.debug(
                "device microcell contraction failed; falling back to host union-find",
                exc_info=True,
            )

    return _contract_overlay_microcells_host(
        labels,
        vertical_tolerance=vertical_tolerance,
    )
