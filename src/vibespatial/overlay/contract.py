from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vibespatial.overlay.microcells import OverlayMicrocellLabels


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
    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
        cp = None
    if cp is not None and hasattr(arr, "__cuda_array_interface__"):
        return cp.asnumpy(arr)
    return np.asarray(arr)


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
