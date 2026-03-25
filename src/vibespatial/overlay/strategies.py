"""Overlay strategy detection and routing.

Detects workload shape (N-vs-1, 1-vs-N, N-vs-M) and selects the optimal
overlay strategy.  Currently all strategies route to the existing per-group
code path; future beads (lyy.16, lyy.18, lyy.21) will add strategy-specific
GPU implementations (containment bypass, batched SH clip, batched overlay).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vibespatial.geometry.owned import OwnedGeometryArray


@dataclass(frozen=True)
class OverlayStrategy:
    """Immutable descriptor for the selected overlay execution strategy."""

    name: str  # e.g. "broadcast_right", "per_group", "batched_overlay"
    many_side: str  # "left", "right", or "both" (for N-vs-M)
    reason: str  # human-readable explanation for provenance


def select_overlay_strategy(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    how: str,
    *,
    candidate_pair_count: int = 0,
) -> OverlayStrategy:
    """Detect workload shape and select the optimal overlay strategy.

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
    if right.row_count == 1 and left.row_count > 1:
        return OverlayStrategy(
            name="broadcast_right",
            many_side="left",
            reason=(
                f"N-vs-1 pattern: {left.row_count} left rows vs 1 right row, "
                f"how={how}, pairs={candidate_pair_count}"
            ),
        )

    if left.row_count == 1 and right.row_count > 1:
        return OverlayStrategy(
            name="broadcast_left",
            many_side="right",
            reason=(
                f"1-vs-N pattern: 1 left row vs {right.row_count} right rows, "
                f"how={how}, pairs={candidate_pair_count}"
            ),
        )

    return OverlayStrategy(
        name="per_group",
        many_side="both",
        reason=(
            f"N-vs-M pattern: {left.row_count} left rows vs "
            f"{right.row_count} right rows, how={how}, "
            f"pairs={candidate_pair_count}"
        ),
    )
