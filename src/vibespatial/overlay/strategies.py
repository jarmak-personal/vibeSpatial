"""Overlay strategy detection and routing.

Detects workload shape (N-vs-1, 1-vs-N, N-vs-M) and selects the optimal
overlay strategy.  Currently all strategies route to the existing per-group
code path; future beads (lyy.16, lyy.18, lyy.21) will add strategy-specific
GPU implementations (containment bypass, batched SH clip, batched overlay).

Uses the shared ``WorkloadShape`` enum from ``vibespatial.runtime.crossover``
for broadcast detection (nsf.5), falling back to overlay-specific detection
for ``broadcast_left`` (which the shared enum intentionally omits).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vibespatial.geometry.owned import OwnedGeometryArray
    from vibespatial.runtime.crossover import WorkloadShape


@dataclass(frozen=True)
class OverlayStrategy:
    """Immutable descriptor for the selected overlay execution strategy."""

    name: str  # e.g. "broadcast_right", "per_group", "batched_overlay"
    many_side: str  # "left", "right", or "both" (for N-vs-M)
    reason: str  # human-readable explanation for provenance
    workload_shape: WorkloadShape | None = None  # shared enum, None for overlay-only shapes


def select_overlay_strategy(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    how: str,
    *,
    candidate_pair_count: int = 0,
) -> OverlayStrategy:
    """Detect workload shape and select the optimal overlay strategy.

    Uses the shared ``detect_workload_shape()`` for broadcast-right and
    pairwise detection.  Falls through to overlay-specific logic for
    broadcast-left (1-vs-N) and N-vs-M (per_group), since the shared
    enum intentionally omits these shapes.

    Currently supported strategies:

    - ``"per_group"``: default, process each ``(left_i, right_j)`` group
      independently.
    - ``"broadcast_right"``: N-vs-1 pattern, right has 1 row (placeholder
      for containment bypass).
    - ``"broadcast_left"``: 1-vs-N pattern, left has 1 row.

    Parameters
    ----------
    left : OwnedGeometryArray
        Left geometry array.
    right : OwnedGeometryArray
        Right geometry array.
    how : str
        Overlay operation: ``"intersection"``, ``"union"``, ``"difference"``,
        ``"symmetric_difference"``.
    candidate_pair_count : int
        Number of candidate pairs from the spatial join (informational).

    Returns
    -------
    OverlayStrategy
        The selected strategy descriptor.
    """
    from vibespatial.runtime.crossover import WorkloadShape, detect_workload_shape

    # Use shared workload detection for broadcast-right and pairwise.
    # detect_workload_shape() raises ValueError for mismatched lengths
    # (both > 1), which is valid for overlay (N-vs-M = per_group strategy).
    try:
        shape = detect_workload_shape(left.row_count, right.row_count)
    except ValueError:
        shape = None

    if shape is WorkloadShape.BROADCAST_RIGHT:
        return OverlayStrategy(
            name="broadcast_right",
            many_side="left",
            reason=(
                f"N-vs-1 pattern: {left.row_count} left rows vs 1 right row, "
                f"how={how}, pairs={candidate_pair_count}"
            ),
            workload_shape=shape,
        )

    # broadcast_left: the shared enum intentionally omits BROADCAST_LEFT,
    # so overlay detects it directly.
    if left.row_count == 1 and right.row_count > 1:
        return OverlayStrategy(
            name="broadcast_left",
            many_side="right",
            reason=(
                f"1-vs-N pattern: 1 left row vs {right.row_count} right rows, "
                f"how={how}, pairs={candidate_pair_count}"
            ),
            workload_shape=None,  # no shared enum equivalent
        )

    # PAIRWISE (row-matched) or N-vs-M (mismatched, caught by ValueError
    # above) both route to per_group.
    return OverlayStrategy(
        name="per_group",
        many_side="both",
        reason=(
            f"N-vs-M pattern: {left.row_count} left rows vs "
            f"{right.row_count} right rows, how={how}, "
            f"pairs={candidate_pair_count}"
        ),
        workload_shape=shape,  # WorkloadShape.PAIRWISE or None (N-vs-M)
    )
