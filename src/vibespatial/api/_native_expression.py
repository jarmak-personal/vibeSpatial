from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vibespatial.api._native_rowset import NativeRowSet


def _is_device_array(values: Any) -> bool:
    return hasattr(values, "__cuda_array_interface__")


def _array_size(values: Any) -> int:
    shape = getattr(values, "shape", None)
    if shape is not None:
        return int(shape[0])
    return len(values)


def _array_namespace(values: Any):
    if _is_device_array(values):
        import cupy as cp

        return cp
    return np


def _comparison_mask(values: Any, op: str, scalar: float):
    xp = _array_namespace(values)
    values_array = xp.asarray(values)
    threshold = np.float64(scalar)
    if not np.isfinite(threshold):
        raise ValueError("NativeExpression scalar comparisons require a finite threshold")

    if op == ">":
        mask = values_array > threshold
    elif op == ">=":
        mask = values_array >= threshold
    elif op == "<":
        mask = values_array < threshold
    elif op == "<=":
        mask = values_array <= threshold
    elif op == "==":
        mask = values_array == threshold
    elif op == "!=":
        mask = values_array != threshold
    else:
        raise ValueError(
            "NativeExpression scalar comparison op must be one of "
            "'>', '>=', '<', '<=', '==', or '!='"
        )

    dtype = np.dtype(getattr(values_array, "dtype", np.float64))
    if np.issubdtype(dtype, np.floating):
        mask = mask & xp.isfinite(values_array)
    return mask


def _threshold_ambiguity_mask(values: Any, scalar: float, epsilon: float):
    xp = _array_namespace(values)
    values_array = xp.asarray(values)
    threshold = np.float64(scalar)
    tolerance = np.float64(epsilon)
    if not np.isfinite(threshold):
        raise ValueError("NativeExpression scalar comparisons require a finite threshold")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("NativeExpression threshold guard epsilon must be finite and >= 0")

    dtype = np.dtype(getattr(values_array, "dtype", np.float64))
    if not np.issubdtype(dtype, np.floating):
        return xp.zeros(values_array.shape, dtype=xp.bool_)
    return xp.isfinite(values_array) & (xp.abs(values_array - threshold) <= tolerance)


def _range_mask(values: Any, lower: float, upper: float, inclusive: str):
    xp = _array_namespace(values)
    values_array = xp.asarray(values)
    lower_threshold = np.float64(lower)
    upper_threshold = np.float64(upper)
    if not np.isfinite(lower_threshold) or not np.isfinite(upper_threshold):
        raise ValueError("NativeExpression range comparisons require finite thresholds")
    if lower_threshold > upper_threshold:
        raise ValueError("NativeExpression range lower threshold must be <= upper")

    if inclusive == "both":
        mask = (values_array >= lower_threshold) & (values_array <= upper_threshold)
    elif inclusive == "left":
        mask = (values_array >= lower_threshold) & (values_array < upper_threshold)
    elif inclusive == "right":
        mask = (values_array > lower_threshold) & (values_array <= upper_threshold)
    elif inclusive == "neither":
        mask = (values_array > lower_threshold) & (values_array < upper_threshold)
    else:
        raise ValueError(
            "NativeExpression range inclusive must be one of "
            "'both', 'left', 'right', or 'neither'"
        )

    dtype = np.dtype(getattr(values_array, "dtype", np.float64))
    if np.issubdtype(dtype, np.floating):
        mask = mask & xp.isfinite(values_array)
    return mask


def _range_ambiguity_mask(values: Any, lower: float, upper: float, epsilon: float):
    xp = _array_namespace(values)
    values_array = xp.asarray(values)
    lower_threshold = np.float64(lower)
    upper_threshold = np.float64(upper)
    tolerance = np.float64(epsilon)
    if not np.isfinite(lower_threshold) or not np.isfinite(upper_threshold):
        raise ValueError("NativeExpression range comparisons require finite thresholds")
    if lower_threshold > upper_threshold:
        raise ValueError("NativeExpression range lower threshold must be <= upper")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("NativeExpression threshold guard epsilon must be finite and >= 0")

    dtype = np.dtype(getattr(values_array, "dtype", np.float64))
    if not np.issubdtype(dtype, np.floating):
        return xp.zeros(values_array.shape, dtype=xp.bool_)
    finite = xp.isfinite(values_array)
    near_lower = xp.abs(values_array - lower_threshold) <= tolerance
    near_upper = xp.abs(values_array - upper_threshold) <= tolerance
    return finite & (near_lower | near_upper)


@dataclass(frozen=True)
class NativeExpressionComparison:
    """Private threshold comparison plus rows that need exact refinement."""

    rowset: NativeRowSet
    ambiguous_rowset: NativeRowSet
    operation: str
    scalar: float | tuple[float, float]
    epsilon: float
    inclusive: str | None = None

    @property
    def is_device(self) -> bool:
        return self.rowset.is_device or self.ambiguous_rowset.is_device

    @property
    def ambiguous(self) -> NativeRowSet:
        return self.ambiguous_rowset

    @property
    def is_unambiguous(self) -> bool:
        return len(self.ambiguous_rowset) == 0


@dataclass(frozen=True)
class NativeExpression:
    """Private expression vector consumed only by admitted native operations."""

    operation: str
    values: Any
    source_token: str | None = None
    source_row_count: int | None = None
    dtype: str | None = None
    precision: str | None = None
    null_policy: str = "nan-false"
    readiness: Any | None = None

    def __post_init__(self) -> None:
        if self.source_row_count is not None and len(self) != int(self.source_row_count):
            raise ValueError(
                "NativeExpression source_row_count must match expression length"
            )
        if self.null_policy != "nan-false":
            raise ValueError("NativeExpression currently admits only nan-false null policy")

    @property
    def is_device(self) -> bool:
        return _is_device_array(self.values)

    def __len__(self) -> int:
        return _array_size(self.values)

    def compare_scalar(self, op: str, scalar: float) -> NativeRowSet:
        """Lower a scalar comparison to private row positions.

        This intentionally returns ``NativeRowSet`` rather than a public boolean
        Series.  Floating nulls/NaNs compare false so invalid geometry rows do
        not enter native row flow.
        """
        xp = _array_namespace(self.values)
        mask = _comparison_mask(self.values, op, scalar)
        positions = xp.nonzero(mask)[0].astype(xp.int64, copy=False)
        return NativeRowSet.from_positions(
            positions,
            source_token=self.source_token,
            source_row_count=self.source_row_count,
            ordered=True,
            unique=True,
        )

    def compare_scalar_guarded(
        self,
        op: str,
        scalar: float,
        *,
        epsilon: float,
    ) -> NativeExpressionComparison:
        """Lower definite comparison rows while exposing threshold-ambiguous rows."""
        xp = _array_namespace(self.values)
        mask = _comparison_mask(self.values, op, scalar)
        ambiguous_mask = _threshold_ambiguity_mask(self.values, scalar, epsilon)
        positions = xp.nonzero(mask & ~ambiguous_mask)[0].astype(xp.int64, copy=False)
        ambiguous_positions = xp.nonzero(ambiguous_mask)[0].astype(xp.int64, copy=False)
        rowset = NativeRowSet.from_positions(
            positions,
            source_token=self.source_token,
            source_row_count=self.source_row_count,
            ordered=True,
            unique=True,
        )
        ambiguous_rowset = NativeRowSet.from_positions(
            ambiguous_positions,
            source_token=self.source_token,
            source_row_count=self.source_row_count,
            ordered=True,
            unique=True,
        )
        return NativeExpressionComparison(
            rowset=rowset,
            ambiguous_rowset=ambiguous_rowset,
            operation=f"{self.operation}{op}",
            scalar=float(scalar),
            epsilon=float(epsilon),
        )

    def compare_range(
        self,
        lower: float,
        upper: float,
        *,
        inclusive: str = "both",
    ) -> NativeRowSet:
        """Lower a scalar range comparison to private row positions."""
        xp = _array_namespace(self.values)
        mask = _range_mask(self.values, lower, upper, inclusive)
        positions = xp.nonzero(mask)[0].astype(xp.int64, copy=False)
        return NativeRowSet.from_positions(
            positions,
            source_token=self.source_token,
            source_row_count=self.source_row_count,
            ordered=True,
            unique=True,
        )

    def compare_range_guarded(
        self,
        lower: float,
        upper: float,
        *,
        inclusive: str = "both",
        epsilon: float,
    ) -> NativeExpressionComparison:
        """Lower definite range rows while exposing boundary-ambiguous rows."""
        xp = _array_namespace(self.values)
        mask = _range_mask(self.values, lower, upper, inclusive)
        ambiguous_mask = _range_ambiguity_mask(self.values, lower, upper, epsilon)
        positions = xp.nonzero(mask & ~ambiguous_mask)[0].astype(xp.int64, copy=False)
        ambiguous_positions = xp.nonzero(ambiguous_mask)[0].astype(xp.int64, copy=False)
        rowset = NativeRowSet.from_positions(
            positions,
            source_token=self.source_token,
            source_row_count=self.source_row_count,
            ordered=True,
            unique=True,
        )
        ambiguous_rowset = NativeRowSet.from_positions(
            ambiguous_positions,
            source_token=self.source_token,
            source_row_count=self.source_row_count,
            ordered=True,
            unique=True,
        )
        return NativeExpressionComparison(
            rowset=rowset,
            ambiguous_rowset=ambiguous_rowset,
            operation=f"{self.operation}.between",
            scalar=(float(lower), float(upper)),
            epsilon=float(epsilon),
            inclusive=inclusive,
        )

    def greater_than(self, scalar: float) -> NativeRowSet:
        return self.compare_scalar(">", scalar)

    def greater_than_guarded(
        self,
        scalar: float,
        *,
        epsilon: float,
    ) -> NativeExpressionComparison:
        return self.compare_scalar_guarded(">", scalar, epsilon=epsilon)

    def greater_equal(self, scalar: float) -> NativeRowSet:
        return self.compare_scalar(">=", scalar)

    def greater_equal_guarded(
        self,
        scalar: float,
        *,
        epsilon: float,
    ) -> NativeExpressionComparison:
        return self.compare_scalar_guarded(">=", scalar, epsilon=epsilon)

    def less_than(self, scalar: float) -> NativeRowSet:
        return self.compare_scalar("<", scalar)

    def less_than_guarded(
        self,
        scalar: float,
        *,
        epsilon: float,
    ) -> NativeExpressionComparison:
        return self.compare_scalar_guarded("<", scalar, epsilon=epsilon)

    def less_equal(self, scalar: float) -> NativeRowSet:
        return self.compare_scalar("<=", scalar)

    def less_equal_guarded(
        self,
        scalar: float,
        *,
        epsilon: float,
    ) -> NativeExpressionComparison:
        return self.compare_scalar_guarded("<=", scalar, epsilon=epsilon)

    def equal_to(self, scalar: float) -> NativeRowSet:
        return self.compare_scalar("==", scalar)

    def not_equal(self, scalar: float) -> NativeRowSet:
        return self.compare_scalar("!=", scalar)

    def between(
        self,
        lower: float,
        upper: float,
        *,
        inclusive: str = "both",
    ) -> NativeRowSet:
        return self.compare_range(lower, upper, inclusive=inclusive)

    def between_guarded(
        self,
        lower: float,
        upper: float,
        *,
        inclusive: str = "both",
        epsilon: float,
    ) -> NativeExpressionComparison:
        return self.compare_range_guarded(
            lower,
            upper,
            inclusive=inclusive,
            epsilon=epsilon,
        )


__all__ = ["NativeExpression", "NativeExpressionComparison"]
