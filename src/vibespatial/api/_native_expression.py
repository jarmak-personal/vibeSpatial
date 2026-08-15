from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vibespatial.api._native_rowset import NativeDeviceSelection, NativeRowSet
from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_I64,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup


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


_NATIVE_EXPRESSION_ROWSET_KERNEL_NAMES = (
    "bool_equal_rowset_i64_kernel",
)

_NATIVE_EXPRESSION_ROWSET_KERNEL_SOURCE = r"""
extern "C" __global__ void bool_equal_rowset_i64_kernel(
    const unsigned char* __restrict__ values,
    long long n,
    int scalar,
    long long* __restrict__ positions,
    int* __restrict__ count
) {
    const long long i = (long long)blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const bool value = values[i] != 0u;
    const bool target = scalar != 0;
    if (value != target) return;
    const int pos = atomicAdd(count, 1);
    positions[pos] = i;
}
"""

request_nvrtc_warmup([
    (
        "native-expression-rowset",
        _NATIVE_EXPRESSION_ROWSET_KERNEL_SOURCE,
        _NATIVE_EXPRESSION_ROWSET_KERNEL_NAMES,
    ),
])


def _native_expression_rowset_kernels():
    return compile_kernel_group(
        "native-expression-rowset",
        _NATIVE_EXPRESSION_ROWSET_KERNEL_SOURCE,
        _NATIVE_EXPRESSION_ROWSET_KERNEL_NAMES,
    )


def _device_bool_equal_rowset(values: Any, scalar: bool):
    import cupy as cp

    d_values = cp.asarray(values, dtype=cp.bool_)
    n = int(d_values.size)
    d_positions = cp.empty(n, dtype=cp.int64)
    d_count = cp.zeros(1, dtype=cp.int32)
    if n == 0:
        return d_positions[:0]

    runtime = get_cuda_runtime()
    kernels = _native_expression_rowset_kernels()
    kernel = kernels["bool_equal_rowset_i64_kernel"]
    ptr = runtime.pointer
    grid, block = runtime.launch_config(kernel, n)
    runtime.launch(
        kernel,
        grid=grid,
        block=block,
        params=(
            (
                ptr(d_values),
                int(n),
                1 if scalar else 0,
                ptr(d_positions),
                ptr(d_count),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I64,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
            ),
        ),
    )
    return d_positions[: int(cp.asarray(d_count)[0])]


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


def _is_scalar_operand(value: Any) -> bool:
    return np.isscalar(value) and not isinstance(value, (str, bytes))


def _validate_expression_operand(
    left: NativeExpression,
    right: NativeExpression,
) -> tuple[str | None, int | None] | None:
    if left.is_device != right.is_device:
        return None
    if (
        left.source_token is not None
        and right.source_token is not None
        and left.source_token != right.source_token
    ):
        return None
    if (
        left.source_row_count is not None
        and right.source_row_count is not None
        and int(left.source_row_count) != int(right.source_row_count)
    ):
        return None
    if len(left) != len(right):
        return None
    source_token = left.source_token if left.source_token is not None else right.source_token
    source_row_count = (
        int(left.source_row_count)
        if left.source_row_count is not None
        else (
            int(right.source_row_count)
            if right.source_row_count is not None
            else len(left)
        )
    )
    return source_token, source_row_count


def _binary_values(left: Any, right: Any, op: str, xp):
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if op == "+":
            return xp.asarray(left) + xp.asarray(right)
        if op == "-":
            return xp.asarray(left) - xp.asarray(right)
        if op == "*":
            return xp.asarray(left) * xp.asarray(right)
        if op == "/":
            return xp.asarray(left) / xp.asarray(right)
        if op == "//":
            return xp.asarray(left) // xp.asarray(right)
        if op == "%":
            return xp.asarray(left) % xp.asarray(right)
    raise ValueError("NativeExpression arithmetic op must be one of '+', '-', '*', '/', '//', or '%'")


def _comparison_values(left: Any, right: Any, op: str, xp):
    if op == ">":
        return xp.asarray(left) > xp.asarray(right)
    if op == ">=":
        return xp.asarray(left) >= xp.asarray(right)
    if op == "<":
        return xp.asarray(left) < xp.asarray(right)
    if op == "<=":
        return xp.asarray(left) <= xp.asarray(right)
    if op == "==":
        return xp.asarray(left) == xp.asarray(right)
    if op == "!=":
        return xp.asarray(left) != xp.asarray(right)
    raise ValueError(
        "NativeExpression comparison op must be one of "
        "'>', '>=', '<', '<=', '==', or '!='"
    )


def _finite_mask_for(values: Any, xp):
    values_array = xp.asarray(values)
    dtype = np.dtype(getattr(values_array, "dtype", np.float64))
    if not np.issubdtype(dtype, np.floating):
        return xp.ones(values_array.shape, dtype=xp.bool_)
    return xp.isfinite(values_array)


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
    certified_position_domain_size: int | None = None

    def __post_init__(self) -> None:
        if self.source_row_count is not None and len(self) != int(self.source_row_count):
            raise ValueError(
                "NativeExpression source_row_count must match expression length"
            )
        if self.null_policy != "nan-false":
            raise ValueError("NativeExpression currently admits only nan-false null policy")
        if (
            self.certified_position_domain_size is not None
            and int(self.certified_position_domain_size) < 0
        ):
            raise ValueError("certified position domain size must be non-negative")

    @property
    def is_device(self) -> bool:
        return _is_device_array(self.values)

    def __len__(self) -> int:
        return _array_size(self.values)

    def binary_arithmetic(
        self,
        op: str,
        other: NativeExpression | float | int | bool,
        *,
        reverse: bool = False,
    ) -> NativeExpression | None:
        """Return a derived private scalar vector for admitted arithmetic.

        Physical shape: row-aligned scalar expression flow.  Device expressions
        stay device-resident; mixed host/device expression arithmetic declines
        so arithmetic composition never hides an implicit transfer.
        """
        if isinstance(other, NativeExpression):
            metadata = _validate_expression_operand(self, other)
            if metadata is None:
                return None
            source_token, source_row_count = metadata
            xp = _array_namespace(self.values)
            left_values, right_values = (
                (other.values, self.values) if reverse else (self.values, other.values)
            )
            right_op = other.operation
        elif _is_scalar_operand(other):
            xp = _array_namespace(self.values)
            source_token = self.source_token
            source_row_count = self.source_row_count
            if isinstance(other, (int, np.integer, bool, np.bool_)) and op != "/":
                try:
                    scalar = np.int64(other)
                except (OverflowError, TypeError, ValueError):
                    return None
            else:
                scalar = np.float64(other)
            if np.issubdtype(np.asarray(scalar).dtype, np.floating) and not np.isfinite(
                scalar
            ):
                return None
            left_values, right_values = (
                (scalar, self.values) if reverse else (self.values, scalar)
            )
            right_op = repr(scalar.item())
        else:
            return None

        values = _binary_values(left_values, right_values, op, xp)
        operation = (
            f"({right_op}{op}{self.operation})"
            if reverse and not isinstance(other, NativeExpression)
            else f"({self.operation}{op}{right_op})"
        )
        return type(self)(
            operation=operation,
            values=values,
            source_token=source_token,
            source_row_count=source_row_count,
            dtype=str(getattr(values, "dtype", "")) or None,
            precision="derived",
            null_policy=self.null_policy,
        )

    def compare(
        self,
        op: str,
        other: NativeExpression | float | int | bool,
        *,
        reverse: bool = False,
    ) -> NativeRowSet | None:
        """Lower an admitted scalar/expression comparison to row positions."""
        if isinstance(other, NativeExpression):
            metadata = _validate_expression_operand(self, other)
            if metadata is None:
                return None
            source_token, source_row_count = metadata
            xp = _array_namespace(self.values)
            left_values, right_values = (
                (other.values, self.values) if reverse else (self.values, other.values)
            )
        elif _is_scalar_operand(other):
            scalar = np.float64(other)
            if not np.isfinite(scalar):
                return None
            xp = _array_namespace(self.values)
            source_token = self.source_token
            source_row_count = self.source_row_count
            left_values, right_values = (
                (scalar, self.values) if reverse else (self.values, scalar)
            )
        else:
            return None

        mask = _comparison_values(left_values, right_values, op, xp)
        mask = mask & _finite_mask_for(left_values, xp) & _finite_mask_for(right_values, xp)
        positions = xp.nonzero(mask)[0].astype(xp.int64, copy=False)
        return NativeRowSet.from_positions(
            positions,
            source_token=source_token,
            source_row_count=source_row_count,
            ordered=True,
            unique=True,
        )

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

    def compare_scalar_selection(
        self,
        op: str,
        scalar: float,
    ) -> NativeDeviceSelection | NativeRowSet:
        """Lower a comparison without compacting a dynamic device count.

        Device expressions return capacity-backed compact-prefix storage.
        Consumers that require a true ``NativeRowSet`` must cross the explicit
        count fence through ``NativeDeviceSelection.compact_rowset``.
        """
        mask = _comparison_mask(self.values, op, scalar)
        if self.is_device:
            return NativeDeviceSelection.from_mask(
                mask,
                source_token=self.source_token,
                source_row_count=self.source_row_count,
            )
        positions = np.flatnonzero(np.asarray(mask, dtype=bool)).astype(
            np.int64,
            copy=False,
        )
        return NativeRowSet.from_positions(
            positions,
            source_token=self.source_token,
            source_row_count=self.source_row_count,
            ordered=True,
            unique=True,
        )

    def equal_to_selection(
        self,
        scalar: float,
    ) -> NativeDeviceSelection | NativeRowSet:
        """Return host rowsets or capacity-backed device selections."""
        return self.compare_scalar_selection("==", scalar)

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
        if self.is_device and np.dtype(getattr(self.values, "dtype", bool)) == np.dtype(bool):
            if scalar is True or scalar is False or isinstance(scalar, (bool, np.bool_)):
                positions = _device_bool_equal_rowset(self.values, bool(scalar))
                return NativeRowSet.from_positions(
                    positions,
                    source_token=self.source_token,
                    source_row_count=self.source_row_count,
                    ordered=True,
                    unique=True,
                )
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

    def __add__(self, other):
        return self.binary_arithmetic("+", other)

    def __radd__(self, other):
        return self.binary_arithmetic("+", other, reverse=True)

    def __sub__(self, other):
        return self.binary_arithmetic("-", other)

    def __rsub__(self, other):
        return self.binary_arithmetic("-", other, reverse=True)

    def __mul__(self, other):
        return self.binary_arithmetic("*", other)

    def __rmul__(self, other):
        return self.binary_arithmetic("*", other, reverse=True)

    def __truediv__(self, other):
        return self.binary_arithmetic("/", other)

    def __rtruediv__(self, other):
        return self.binary_arithmetic("/", other, reverse=True)

    def __floordiv__(self, other):
        return self.binary_arithmetic("//", other)

    def __rfloordiv__(self, other):
        return self.binary_arithmetic("//", other, reverse=True)

    def __mod__(self, other):
        return self.binary_arithmetic("%", other)

    def __rmod__(self, other):
        return self.binary_arithmetic("%", other, reverse=True)

    def __gt__(self, other):
        return self.compare(">", other)

    def __ge__(self, other):
        return self.compare(">=", other)

    def __lt__(self, other):
        return self.compare("<", other)

    def __le__(self, other):
        return self.compare("<=", other)


__all__ = ["NativeExpression", "NativeExpressionComparison"]
