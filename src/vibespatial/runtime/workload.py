from __future__ import annotations

from enum import StrEnum


class WorkloadShape(StrEnum):
    """Classification of how left and right geometry arrays relate in size.

    PAIRWISE:        left and right have the same length; element-wise ops.
    BROADCAST_RIGHT: right has length 1, left has length > 1; the single
                     right geometry is broadcast against every left row.
    SCALAR_RIGHT:    right is a scalar (not an array); skips pandas index
                     alignment entirely.

    BROADCAST_LEFT is intentionally omitted — no consumer exists today.
    INDEXED is intentionally omitted — gather-evaluate-scatter is a
    different computation model, not a workload shape.
    """

    PAIRWISE = "pairwise"
    BROADCAST_RIGHT = "broadcast_right"
    SCALAR_RIGHT = "scalar_right"


def detect_workload_shape(
    left_count: int,
    right_count: int | None,
) -> WorkloadShape:
    """Classify the workload shape for a binary operation.

    Parameters
    ----------
    left_count
        Number of rows on the left side.
    right_count
        Number of rows on the right side.  ``None`` means the right
        operand is a scalar geometry (not wrapped in an array).

    Returns
    -------
    WorkloadShape

    Raises
    ------
    ValueError
        If *left_count* and *right_count* are both > 1 and differ in
        length.  Many-to-many operations should use ``gpd.sjoin()``
        instead.
    """
    if right_count is None:
        return WorkloadShape.SCALAR_RIGHT
    if right_count == 1 and left_count > 1:
        return WorkloadShape.BROADCAST_RIGHT
    if left_count == right_count:
        return WorkloadShape.PAIRWISE
    raise ValueError(
        f"Incompatible lengths: left={left_count}, right={right_count}. "
        "Use gpd.sjoin() for many-to-many operations."
    )
