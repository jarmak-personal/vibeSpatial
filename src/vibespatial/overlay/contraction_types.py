from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vibespatial.geometry.owned import OwnedGeometryArray
from vibespatial.spatial.segment_primitives import SegmentLocalEventSummary


@dataclass(frozen=True)
class AlignedOverlayWorkload:
    left_aligned: OwnedGeometryArray
    right_aligned: OwnedGeometryArray
    left_indices: np.ndarray
    right_indices: np.ndarray

    @property
    def row_count(self) -> int:
        return int(self.left_indices.size)


@dataclass(frozen=True)
class RowMicrocellBand:
    row_index: int
    event_x: np.ndarray
    interval_count: int
    max_active_segment_count: int
    mean_active_segment_count: float
    microcell_upper_bound: int

    @property
    def event_count(self) -> int:
        return int(self.event_x.size)

    def as_dict(self) -> dict[str, object]:
        return {
            "row_index": self.row_index,
            "event_x": self.event_x.tolist(),
            "event_count": self.event_count,
            "interval_count": self.interval_count,
            "max_active_segment_count": self.max_active_segment_count,
            "mean_active_segment_count": self.mean_active_segment_count,
            "microcell_upper_bound": self.microcell_upper_bound,
        }


@dataclass(frozen=True)
class ContractionMicrocellSummary:
    row_bands: tuple[RowMicrocellBand, ...]

    @property
    def row_event_counts(self) -> list[int]:
        return [band.event_count for band in self.row_bands]

    @property
    def row_interval_counts(self) -> list[int]:
        return [band.interval_count for band in self.row_bands]

    @property
    def row_max_active_segment_counts(self) -> list[int]:
        return [band.max_active_segment_count for band in self.row_bands]

    @property
    def row_mean_active_segment_counts(self) -> list[float]:
        return [band.mean_active_segment_count for band in self.row_bands]

    @property
    def row_microcell_upper_bounds(self) -> list[int]:
        return [band.microcell_upper_bound for band in self.row_bands]

    @property
    def max_row_microcell_upper_bound(self) -> int:
        return max(self.row_microcell_upper_bounds, default=0)

    @property
    def total_microcell_upper_bound(self) -> int:
        return int(sum(self.row_microcell_upper_bounds))

    @property
    def max_active_segment_count(self) -> int:
        return max(self.row_max_active_segment_counts, default=0)

    def as_dict(self) -> dict[str, object]:
        return {
            "row_event_counts": self.row_event_counts,
            "row_interval_counts": self.row_interval_counts,
            "row_max_active_segment_counts": self.row_max_active_segment_counts,
            "row_mean_active_segment_counts": self.row_mean_active_segment_counts,
            "row_microcell_upper_bounds": self.row_microcell_upper_bounds,
            "max_row_microcell_upper_bound": self.max_row_microcell_upper_bound,
            "total_microcell_upper_bound": self.total_microcell_upper_bound,
            "max_active_segment_count": self.max_active_segment_count,
        }


@dataclass(frozen=True)
class OverlayContractionSummary:
    workload: AlignedOverlayWorkload
    left_aligned_segment_count: int
    right_aligned_segment_count: int
    exact_events: SegmentLocalEventSummary
    microcells: ContractionMicrocellSummary

    @property
    def pair_count(self) -> int:
        return self.workload.row_count

    def as_dict(self) -> dict[str, object]:
        return {
            "pair_count": self.pair_count,
            "left_indices": self.workload.left_indices.tolist(),
            "right_indices": self.workload.right_indices.tolist(),
            "left_aligned_segment_count": self.left_aligned_segment_count,
            "right_aligned_segment_count": self.right_aligned_segment_count,
            "candidate_pairs": int(self.exact_events.candidate_pairs),
            "point_intersection_count": int(self.exact_events.point_intersection_count),
            "parallel_or_colinear_candidate_count": int(
                self.exact_events.parallel_or_colinear_candidate_count
            ),
            "row_point_intersection_counts": self.exact_events.row_point_intersection_counts.tolist(),
            "exact_event_counts": self.exact_events.exact_event_counts.tolist(),
            "max_exact_events": int(self.exact_events.max_exact_events),
            "microcells": self.microcells.as_dict(),
        }
