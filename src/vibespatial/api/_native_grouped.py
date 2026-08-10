from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from vibespatial.api._native_rowset import NativeDeviceSelection, NativeIndexPlan
from vibespatial.runtime.materialization import (
    NativeExportBoundary,
    record_native_export_boundary,
)


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


def _array_namespace_for(*values: Any):
    if any(_is_device_array(value) for value in values):
        import cupy as cp

        return cp
    return np


def _as_group_codes(values: Any, *, group_count: int | None):
    if _is_device_array(values):
        if group_count is None:
            raise ValueError("group_count is required for device group codes")
        import cupy as cp

        return cp.asarray(values, dtype=cp.int32)
    return np.asarray(values, dtype=np.int32)


def _resolve_group_count(codes: Any, group_count: int | None) -> int:
    if group_count is not None:
        return int(group_count)
    if _is_device_array(codes):
        raise ValueError("group_count is required for device group codes")
    observed = np.asarray(codes, dtype=np.int32)
    observed = observed[observed >= 0]
    if observed.size == 0:
        return 0
    return int(observed.max()) + 1


def _validate_host_group_codes(codes: np.ndarray, *, group_count: int) -> None:
    if codes.ndim != 1:
        raise ValueError("NativeGrouped dense group codes must be one-dimensional")
    if np.any(codes < -1):
        raise ValueError("NativeGrouped dense group codes must use -1 for dropped null keys")
    if np.any(codes >= int(group_count)):
        raise ValueError("NativeGrouped dense group codes exceed group_count")


def _empty_like_namespace_array(xp, dtype, size: int = 0):
    return xp.empty(int(size), dtype=dtype)


def _observed_sorted_order(codes: Any, *, group_count: int):
    xp = _array_namespace(codes)
    observed_mask = (codes >= 0) & (codes < int(group_count))
    observed_rows = xp.nonzero(observed_mask)[0]
    if _array_size(observed_rows) == 0:
        offsets = xp.asarray([0], dtype=xp.int32)
        return (
            _empty_like_namespace_array(xp, xp.int64),
            offsets,
            _empty_like_namespace_array(xp, xp.int32),
        )

    observed_codes = codes[observed_rows]
    if xp is np:
        order = np.argsort(observed_codes, kind="stable")
    else:
        order = xp.argsort(observed_codes)
    sorted_order = observed_rows[order].astype(xp.int64, copy=False)
    sorted_codes = observed_codes[order].astype(xp.int32, copy=False)
    group_ids, counts = xp.unique(sorted_codes, return_counts=True)
    offsets = xp.concatenate(
        [
            xp.asarray([0], dtype=xp.int32),
            xp.cumsum(counts.astype(xp.int32, copy=False)),
        ],
    )
    return sorted_order, offsets, group_ids.astype(xp.int32, copy=False)


def _bincount_sum(codes: Any, values: Any, *, group_count: int):
    xp = _array_namespace(codes)
    result = xp.bincount(
        codes,
        weights=values,
        minlength=int(group_count),
    )[: int(group_count)]
    if np.issubdtype(values.dtype, np.bool_):
        return result.astype(xp.int64, copy=False)
    if np.issubdtype(values.dtype, np.integer):
        result = result.astype(values.dtype, copy=False)
    return result


def _bincount_count(codes: Any, *, group_count: int):
    xp = _array_namespace(codes)
    return xp.bincount(codes, minlength=int(group_count))[: int(group_count)].astype(
        xp.int64,
        copy=False,
    )


def _is_numeric_like(values: Any) -> bool:
    dtype = getattr(values, "dtype", None)
    if dtype is None:
        dtype = np.asarray(values).dtype
    try:
        normalized = np.dtype(dtype)
    except TypeError:
        return False
    return bool(np.issubdtype(normalized, np.number) or np.issubdtype(normalized, np.bool_))


def _normalized_dtype(values: Any) -> np.dtype:
    dtype = getattr(values, "dtype", None)
    if dtype is None:
        return np.asarray(values).dtype
    return np.dtype(dtype)


def _empty_group_extrema_dtype(values: Any):
    normalized = _normalized_dtype(values)
    if np.issubdtype(normalized, np.integer) or np.issubdtype(normalized, np.bool_):
        return np.dtype(np.float64)
    return normalized


def _extrema_identity(dtype: Any, reducer: str) -> Any:
    normalized = np.dtype(dtype)
    if np.issubdtype(normalized, np.bool_):
        return reducer == "min"
    if np.issubdtype(normalized, np.integer):
        info = np.iinfo(normalized)
        return info.max if reducer == "min" else info.min
    if np.issubdtype(normalized, np.floating):
        return np.inf if reducer == "min" else -np.inf
    raise TypeError("NativeGrouped min/max reducers admit only real numeric or bool values")


def _mark_missing_extrema_groups(xp, reduced: Any, group_ids: Any, group_count: int) -> None:
    has_values = xp.zeros(int(group_count), dtype=xp.bool_)
    has_values[group_ids] = True
    reduced[~has_values] = xp.nan


def _cupy_extrema_scatter(
    xp,
    observed_codes: Any,
    observed_values: Any,
    group_ids: Any,
    *,
    group_count: int,
    reducer: str,
    has_all_groups: bool,
):
    reducer_func = xp.minimum if reducer == "min" else xp.maximum
    dtype = _normalized_dtype(observed_values)
    if not has_all_groups:
        out_dtype = _empty_group_extrema_dtype(observed_values)
        reduced = xp.empty(int(group_count), dtype=out_dtype)
        reduced[...] = _extrema_identity(out_dtype, reducer)
        reducer_func.at(
            reduced,
            observed_codes,
            observed_values.astype(out_dtype, copy=False),
        )
        _mark_missing_extrema_groups(xp, reduced, group_ids, group_count)
        return reduced

    if np.issubdtype(dtype, np.bool_):
        scratch_dtype = xp.int32
        final_dtype = dtype
    elif np.issubdtype(dtype, np.signedinteger) and dtype.itemsize == 8:
        sign_bit = xp.asarray(1 << 63, dtype=xp.uint64)
        encoded = observed_values.view(xp.uint64) ^ sign_bit
        reduced_encoded = xp.empty(int(group_count), dtype=xp.uint64)
        reduced_encoded[...] = _extrema_identity(xp.uint64, reducer)
        reducer_func.at(reduced_encoded, observed_codes, encoded)
        return (reduced_encoded ^ sign_bit).view(xp.int64)
    elif np.issubdtype(dtype, np.signedinteger):
        scratch_dtype = xp.int32
        final_dtype = dtype
    elif np.issubdtype(dtype, np.unsignedinteger):
        scratch_dtype = xp.uint64 if dtype.itemsize == 8 else xp.uint32
        final_dtype = dtype
    elif np.issubdtype(dtype, np.floating):
        scratch_dtype = xp.float64 if dtype.itemsize == 8 else xp.float32
        final_dtype = dtype
    else:
        raise TypeError("NativeGrouped min/max reducers admit only real numeric or bool values")

    reduced = xp.empty(int(group_count), dtype=scratch_dtype)
    reduced[...] = _extrema_identity(scratch_dtype, reducer)
    reducer_func.at(
        reduced,
        observed_codes,
        observed_values.astype(scratch_dtype, copy=False),
    )
    return reduced.astype(final_dtype, copy=False)


def _as_host_series(values: Any) -> pd.Series:
    if _is_device_array(values):
        raise TypeError("NativeGrouped take reducers require host values")
    return values if isinstance(values, pd.Series) else pd.Series(values)


def _device_to_host(values: Any, *, reason: str) -> np.ndarray:
    from vibespatial.cuda._runtime import get_cuda_runtime

    return get_cuda_runtime().copy_device_to_host(values, reason=reason)


@dataclass(frozen=True)
class NativeGroupedReduction:
    """Private grouped reducer result with an explicit pandas export boundary."""

    values: Any
    reducer: str
    group_count: int
    output_index_plan: NativeIndexPlan | None = None

    @property
    def is_device(self) -> bool:
        return _is_device_array(self.values)

    def to_pandas(self, *, name: str | None = None) -> pd.Series:
        if isinstance(self.values, pd.Series):
            from vibespatial.api._native_public_arrays import NativeAttributeColumnArray

            if isinstance(self.values.array, NativeAttributeColumnArray):
                index = (
                    self.output_index_plan.index
                    if self.output_index_plan is not None
                    and self.output_index_plan.index is not None
                    else pd.RangeIndex(self.group_count)
                )
                return self.values.array.to_pandas_series(index=index, name=name)
            series = self.values.copy(deep=False)
            if self.output_index_plan is not None and self.output_index_plan.index is not None:
                index = self.output_index_plan.index
            else:
                index = pd.RangeIndex(self.group_count)
            if not series.index.equals(index):
                series.index = index
            series.name = name
            return series
        if self.is_device:
            record_native_export_boundary(
                NativeExportBoundary(
                    surface="vibespatial.api.NativeGroupedReduction.to_pandas",
                    operation="grouped_reduction_to_pandas",
                    target="pandas-series",
                    reason="native grouped reduction exported to pandas Series",
                    row_count=self.group_count,
                    byte_count=int(self.values.nbytes),
                    detail=f"reducer={self.reducer}",
                    d2h_transfer=True,
                    strict_disallowed=False,
                )
            )
            values = _device_to_host(
                self.values,
                reason=(
                    "vibespatial.api.NativeGroupedReduction.to_pandas::grouped_reduction_to_pandas"
                ),
            )
        else:
            values = np.asarray(self.values)
        if self.output_index_plan is not None and self.output_index_plan.index is not None:
            index = self.output_index_plan.index
        else:
            index = pd.RangeIndex(self.group_count)
        return pd.Series(values, index=index, name=name)


@dataclass(frozen=True)
class NativeGroupedAttributeReduction:
    """Private multi-column grouped reducer result."""

    columns: dict[Any, NativeGroupedReduction]
    group_count: int
    output_index_plan: NativeIndexPlan | None = None

    @property
    def is_device(self) -> bool:
        return any(reduction.is_device for reduction in self.columns.values())

    @property
    def index(self) -> pd.Index:
        if self.output_index_plan is not None and self.output_index_plan.index is not None:
            return self.output_index_plan.index
        return pd.RangeIndex(self.group_count)

    def to_pandas(self) -> pd.DataFrame:
        data: dict[Any, Any] = {}
        device_bytes = sum(
            int(reduction.values.nbytes)
            for reduction in self.columns.values()
            if reduction.is_device
        )
        if device_bytes:
            record_native_export_boundary(
                NativeExportBoundary(
                    surface="vibespatial.api.NativeGroupedAttributeReduction.to_pandas",
                    operation="grouped_attribute_reduction_to_pandas",
                    target="pandas-dataframe",
                    reason="native grouped attribute reductions exported to pandas DataFrame",
                    row_count=self.group_count,
                    byte_count=device_bytes,
                    detail=f"columns={len(self.columns)}",
                    d2h_transfer=True,
                    strict_disallowed=False,
                )
            )
        for name, reduction in self.columns.items():
            values = reduction.values
            if isinstance(values, pd.Series):
                from vibespatial.api._native_public_arrays import (
                    NativeAttributeColumnArray,
                )

                if isinstance(values.array, NativeAttributeColumnArray):
                    data[name] = values.array.to_pandas_series(
                        index=self.index,
                        name=name,
                    )
                    continue
                series = values.copy(deep=False)
                if not series.index.equals(self.index):
                    series.index = self.index
                series.name = name
                data[name] = series
                continue
            if reduction.is_device:
                values = _device_to_host(
                    values,
                    reason=(
                        "vibespatial.api.NativeGroupedAttributeReduction.to_pandas"
                        "::grouped_attribute_reduction_to_pandas"
                    ),
                )
            else:
                values = np.asarray(values)
            data[name] = values
        return pd.DataFrame(data, index=self.index)

    def to_native_attribute_table(self):
        from vibespatial.api._native_result_core import NativeAttributeTable

        if self.columns and all(reduction.is_device for reduction in self.columns.values()):
            try:
                import cupy as cp
                import pylibcudf as plc
            except ModuleNotFoundError:
                pass
            else:
                from vibespatial.cuda._runtime import pylibcudf_column_from_device

                device_columns = [
                    pylibcudf_column_from_device(cp.asarray(reduction.values))
                    for reduction in self.columns.values()
                ]
                return NativeAttributeTable(
                    device_table=plc.Table(device_columns),
                    index_override=self.index,
                    column_override=tuple(self.columns),
                )

        return NativeAttributeTable.from_loader(
            self.to_pandas,
            index_override=self.index,
            columns=tuple(self.columns),
        )


@dataclass(frozen=True)
class NativeGroupedSelection:
    """Capacity-backed grouped view with device-resident selected cardinality."""

    group_codes: Any
    selection: NativeDeviceSelection
    group_count: int
    source_token: str | None = None
    output_index_plan: NativeIndexPlan | None = None

    def __post_init__(self) -> None:
        if not _is_device_array(self.group_codes):
            raise TypeError("NativeGroupedSelection requires device group codes")
        source_row_count = self.source_row_count
        if _array_size(self.group_codes) != source_row_count:
            raise ValueError("NativeGroupedSelection group codes must match selection source rows")
        if (
            self.source_token is not None
            and self.selection.source_token is not None
            and self.source_token != self.selection.source_token
        ):
            raise ValueError("NativeGroupedSelection source token must match selection")
        if int(self.group_count) < 0:
            raise ValueError("NativeGroupedSelection group_count must be nonnegative")

    @property
    def capacity(self) -> int:
        return self.selection.capacity

    @property
    def source_row_count(self) -> int:
        if self.selection.source_row_count is None:
            return self.selection.capacity
        return int(self.selection.source_row_count)

    @property
    def logical_count(self):
        return self.selection.logical_count

    @property
    def is_device(self) -> bool:
        return True

    def _capacity_codes(self):
        import cupy as cp

        group_count = int(self.group_count)
        gathered = cp.asarray(
            self.selection.gather_capacity(
                self.group_codes,
                fill_value=group_count,
            ),
            dtype=cp.int64,
        )
        active = self.selection.active_capacity_mask()
        valid = active & (gathered >= 0) & (gathered < group_count)
        return cp.where(valid, gathered, cp.int64(group_count)), valid

    def reduce_numeric(self, values: Any, reducer: str) -> NativeGroupedReduction:
        """Reduce selected all-valid values without compacting selected rows."""
        normalized = reducer.lower()
        if normalized not in {"sum", "count", "mean", "min", "max", "any", "all"}:
            raise ValueError(
                "NativeGroupedSelection reducer must be sum, count, mean, min, max, any, or all"
            )
        if _array_size(values) != self.source_row_count:
            raise ValueError(
                "NativeGroupedSelection values length must match selection source rows"
            )
        if not _is_numeric_like(values):
            raise TypeError("NativeGroupedSelection reducer admits only numeric or bool values")

        import cupy as cp

        group_count = int(self.group_count)
        codes, valid = self._capacity_codes()
        values_array = cp.asarray(
            self.selection.gather_capacity(values, fill_value=0),
        )
        counts_with_sentinel = cp.bincount(
            codes,
            minlength=group_count + 1,
        ).astype(cp.int64, copy=False)
        counts = counts_with_sentinel[:group_count]
        if normalized == "count":
            reduced = counts
        elif normalized in {"sum", "mean"}:
            if normalized == "sum":
                out_dtype = (
                    cp.int64 if np.issubdtype(values_array.dtype, np.bool_) else values_array.dtype
                )
                sums_with_sentinel = cp.zeros(group_count + 1, dtype=out_dtype)
                cp.add.at(
                    sums_with_sentinel,
                    codes,
                    cp.where(
                        valid,
                        values_array.astype(out_dtype, copy=False),
                        cp.zeros((), dtype=out_dtype),
                    ),
                )
                reduced = sums_with_sentinel[:group_count]
            else:
                sums_with_sentinel = cp.zeros(group_count + 1, dtype=cp.float64)
                cp.add.at(
                    sums_with_sentinel,
                    codes,
                    cp.where(
                        valid,
                        values_array.astype(cp.float64, copy=False),
                        cp.float64(0.0),
                    ),
                )
                sums = sums_with_sentinel[:group_count]
                reduced = cp.full(group_count, cp.nan, dtype=cp.float64)
                has_values = counts > 0
                reduced[has_values] = sums[has_values] / counts[has_values]
        elif normalized in {"any", "all"}:
            if not np.issubdtype(values_array.dtype, np.bool_):
                raise TypeError("NativeGroupedSelection any/all reducers admit only bool values")
            true_counts_with_sentinel = cp.zeros(group_count + 1, dtype=cp.uint64)
            cp.add.at(
                true_counts_with_sentinel,
                codes,
                cp.where(valid & values_array, cp.uint64(1), cp.uint64(0)),
            )
            true_counts = true_counts_with_sentinel[:group_count]
            reduced = (
                true_counts > 0
                if normalized == "any"
                else true_counts == counts.astype(cp.uint64, copy=False)
            )
        else:
            out_dtype = _empty_group_extrema_dtype(values_array)
            identity = _extrema_identity(out_dtype, normalized)
            reduced_with_sentinel = cp.full(
                group_count + 1,
                identity,
                dtype=out_dtype,
            )
            scatter_values = cp.where(
                valid,
                values_array.astype(out_dtype, copy=False),
                cp.asarray(identity, dtype=out_dtype),
            )
            reducer_func = cp.minimum if normalized == "min" else cp.maximum
            reducer_func.at(reduced_with_sentinel, codes, scatter_values)
            reduced = reduced_with_sentinel[:group_count]
            reduced = cp.where(counts > 0, reduced, cp.asarray(cp.nan, dtype=out_dtype))

        return NativeGroupedReduction(
            values=reduced,
            reducer=normalized,
            group_count=group_count,
            output_index_plan=self.output_index_plan,
        )

    def reduce_expression(self, expression: Any, reducer: str) -> NativeGroupedReduction:
        """Reduce a selected private expression over source-capacity rows."""
        from vibespatial.api._native_expression import NativeExpression

        if not isinstance(expression, NativeExpression):
            raise TypeError("NativeGroupedSelection.reduce_expression expects NativeExpression")
        if (
            expression.source_row_count is not None
            and int(expression.source_row_count) != self.source_row_count
        ):
            raise ValueError("NativeGroupedSelection expression source rows must match capacity")
        if (
            expression.source_token is not None
            and self.source_token is not None
            and expression.source_token != self.source_token
        ):
            raise ValueError("NativeGroupedSelection expression source token does not match")
        return self.reduce_numeric(expression.values, reducer)


@dataclass(frozen=True)
class NativeGrouped:
    """Private grouped-execution carrier for segmented reductions."""

    group_codes: Any
    group_offsets: Any
    source_token: str | None = None
    sorted_order: Any | None = None
    group_ids: Any | None = None
    group_count: int | None = None
    row_count: int | None = None
    output_index_plan: NativeIndexPlan | None = None
    null_key_policy: str = "drop"
    sorted_order_is_identity: bool = False
    all_groups_observed: bool | None = None
    group_size_min: int | None = None
    group_size_max: int | None = None
    strictly_disjoint_group_bounds: bool | None = None

    @classmethod
    def from_dense_codes(
        cls,
        group_codes: Any,
        *,
        group_count: int | None = None,
        output_index: pd.Index | None = None,
        source_token: str | None = None,
        null_key_policy: str = "drop",
        all_groups_observed: bool | None = None,
        group_size_min: int | None = None,
        group_size_max: int | None = None,
        strictly_disjoint_group_bounds: bool | None = None,
    ) -> NativeGrouped:
        """Build grouped state from dense row-to-group codes.

        Contract: ``-1`` means a dropped null key, valid groups are dense
        ``[0, group_count)``, and device codes are trusted to satisfy that
        contract so construction does not add scalar D2H validation probes.
        """
        if null_key_policy != "drop":
            raise ValueError("NativeGrouped currently admits only drop null-key policy")
        resolved_group_count = _resolve_group_count(group_codes, group_count)
        codes = _as_group_codes(group_codes, group_count=resolved_group_count)
        if not _is_device_array(codes):
            _validate_host_group_codes(codes, group_count=resolved_group_count)
        if output_index is not None and len(output_index) != resolved_group_count:
            raise ValueError("NativeGrouped output_index length must match group_count")
        sorted_order, group_offsets, group_ids = _observed_sorted_order(
            codes,
            group_count=resolved_group_count,
        )
        if not _is_device_array(group_offsets):
            host_offsets = np.asarray(group_offsets, dtype=np.int64)
            host_group_ids = np.asarray(group_ids, dtype=np.int64)
            counts = host_offsets[1:] - host_offsets[:-1]
            inferred_all_observed = int(host_group_ids.size) == int(resolved_group_count) and bool(
                np.all(counts > 0)
            )
            if all_groups_observed is None:
                all_groups_observed = inferred_all_observed
            if group_size_min is None:
                group_size_min = (
                    0 if not inferred_all_observed else int(counts.min()) if counts.size else 0
                )
            if group_size_max is None:
                group_size_max = int(counts.max(initial=0))
        return cls(
            group_codes=codes,
            group_offsets=group_offsets,
            source_token=source_token,
            sorted_order=sorted_order,
            group_ids=group_ids,
            group_count=resolved_group_count,
            row_count=_array_size(codes),
            output_index_plan=(
                NativeIndexPlan.from_index(output_index)
                if output_index is not None
                else NativeIndexPlan(
                    kind="range",
                    length=resolved_group_count,
                    index=pd.RangeIndex(resolved_group_count),
                )
            ),
            null_key_policy=null_key_policy,
            all_groups_observed=all_groups_observed,
            group_size_min=group_size_min,
            group_size_max=group_size_max,
            strictly_disjoint_group_bounds=strictly_disjoint_group_bounds,
        )

    @classmethod
    def from_dense_sorted_offsets(
        cls,
        group_offsets: Any,
        *,
        row_count: int | None = None,
        output_index: pd.Index | None = None,
        source_token: str | None = None,
        null_key_policy: str = "drop",
        all_groups_observed: bool | None = None,
        group_size_min: int | None = None,
        group_size_max: int | None = None,
        strictly_disjoint_group_bounds: bool | None = None,
    ) -> NativeGrouped:
        """Build identity-sorted grouped state while retaining empty groups.

        Unlike ``from_sorted_offsets``, this constructor does not compact the
        offset or group-id metadata when groups have zero rows. It is the
        physical carrier for public-row-capacity grouped work where an empty
        span is meaningful and output order is already dense.
        """
        if null_key_policy != "drop":
            raise ValueError("NativeGrouped currently admits only drop null-key policy")
        xp = _array_namespace(group_offsets)
        offsets = xp.asarray(group_offsets, dtype=xp.int64)
        if _array_size(offsets) == 0:
            raise ValueError("NativeGrouped dense sorted offsets require length >= 1")
        group_count = _array_size(offsets) - 1
        if output_index is not None and len(output_index) != group_count:
            raise ValueError("NativeGrouped output_index length must match group_count")

        if xp is np:
            if offsets.ndim != 1:
                raise ValueError("NativeGrouped dense sorted offsets must be one-dimensional")
            if int(offsets[0]) != 0:
                raise ValueError("NativeGrouped dense sorted offsets must start at zero")
            if np.any(offsets[1:] < offsets[:-1]):
                raise ValueError("NativeGrouped dense sorted offsets must be nondecreasing")
            resolved_row_count = int(offsets[-1])
            if row_count is not None and int(row_count) != resolved_row_count:
                raise ValueError("NativeGrouped row_count must match final dense sorted offset")
        else:
            if row_count is None:
                raise ValueError("NativeGrouped device dense sorted offsets require row_count")
            resolved_row_count = int(row_count)

        if resolved_row_count:
            positions = xp.arange(resolved_row_count, dtype=xp.int64)
            group_codes = xp.searchsorted(
                offsets[1:],
                positions,
                side="right",
            ).astype(xp.int32, copy=False)
        else:
            group_codes = _empty_like_namespace_array(xp, xp.int32)
        return cls(
            group_codes=group_codes,
            group_offsets=offsets.astype(xp.int32, copy=False),
            source_token=source_token,
            sorted_order=xp.arange(resolved_row_count, dtype=xp.int64),
            group_ids=xp.arange(group_count, dtype=xp.int32),
            group_count=group_count,
            row_count=resolved_row_count,
            output_index_plan=(
                NativeIndexPlan.from_index(output_index)
                if output_index is not None
                else NativeIndexPlan(
                    kind="range",
                    length=group_count,
                    index=pd.RangeIndex(group_count),
                )
            ),
            null_key_policy=null_key_policy,
            sorted_order_is_identity=True,
            all_groups_observed=all_groups_observed,
            group_size_min=group_size_min,
            group_size_max=group_size_max,
            strictly_disjoint_group_bounds=strictly_disjoint_group_bounds,
        )

    @classmethod
    def from_sorted_offsets(
        cls,
        group_offsets: Any,
        *,
        row_count: int | None = None,
        output_index: pd.Index | None = None,
        source_token: str | None = None,
        null_key_policy: str = "drop",
        all_groups_observed: bool | None = None,
        group_size_min: int | None = None,
        group_size_max: int | None = None,
        strictly_disjoint_group_bounds: bool | None = None,
    ) -> NativeGrouped:
        """Build grouped state for rows already sorted by dense group offsets.

        The input offsets describe full dense groups.  The carrier stores the
        compact observed-group offsets expected by native consumers while
        preserving an identity sorted-order proof, so geometry reducers can
        consume already grouped rows without a redundant sort or gather.
        """
        if null_key_policy != "drop":
            raise ValueError("NativeGrouped currently admits only drop null-key policy")
        xp = _array_namespace(group_offsets)
        offsets = xp.asarray(group_offsets, dtype=xp.int64)
        if _array_size(offsets) == 0:
            raise ValueError("NativeGrouped sorted offsets must have length >= 1")
        group_count = _array_size(offsets) - 1
        if output_index is not None and len(output_index) != group_count:
            raise ValueError("NativeGrouped output_index length must match group_count")

        if xp is np:
            if offsets.ndim != 1:
                raise ValueError("NativeGrouped sorted offsets must be one-dimensional")
            if int(offsets[0]) != 0:
                raise ValueError("NativeGrouped sorted offsets must start at zero")
            if np.any(offsets[1:] < offsets[:-1]):
                raise ValueError("NativeGrouped sorted offsets must be nondecreasing")
            resolved_row_count = int(offsets[-1])
            if row_count is not None and int(row_count) != resolved_row_count:
                raise ValueError("NativeGrouped row_count must match final sorted offset")
        else:
            if row_count is None:
                raise ValueError("NativeGrouped device sorted offsets require row_count")
            resolved_row_count = int(row_count)

        counts = offsets[1:] - offsets[:-1]
        if xp is np:
            host_counts = np.asarray(counts, dtype=np.int64)
            inferred_all_observed = bool(np.all(host_counts > 0))
            inferred_min = int(host_counts.min()) if host_counts.size else 0
            inferred_max = int(host_counts.max()) if host_counts.size else 0
            if all_groups_observed is None:
                all_groups_observed = inferred_all_observed
            if group_size_min is None:
                group_size_min = inferred_min
            if group_size_max is None:
                group_size_max = inferred_max
        elif group_size_min is not None and int(group_size_min) > 0:
            all_groups_observed = True if all_groups_observed is None else all_groups_observed
        if all_groups_observed is True:
            group_ids = xp.arange(group_count, dtype=xp.int32)
            compact_offsets = offsets.astype(xp.int32, copy=False)
        else:
            observed_mask = counts > 0
            group_ids = xp.nonzero(observed_mask)[0].astype(xp.int32, copy=False)
            observed_counts = counts[observed_mask].astype(xp.int32, copy=False)
            compact_offsets = xp.concatenate(
                [
                    xp.asarray([0], dtype=xp.int32),
                    xp.cumsum(observed_counts, dtype=xp.int32),
                ],
            )
        if resolved_row_count:
            positions = xp.arange(resolved_row_count, dtype=xp.int64)
            group_codes = xp.searchsorted(
                offsets[1:],
                positions,
                side="right",
            ).astype(xp.int32, copy=False)
        else:
            group_codes = _empty_like_namespace_array(xp, xp.int32)
        sorted_order = xp.arange(resolved_row_count, dtype=xp.int64)
        return cls(
            group_codes=group_codes,
            group_offsets=compact_offsets,
            source_token=source_token,
            sorted_order=sorted_order,
            group_ids=group_ids,
            group_count=group_count,
            row_count=resolved_row_count,
            output_index_plan=(
                NativeIndexPlan.from_index(output_index)
                if output_index is not None
                else NativeIndexPlan(
                    kind="range",
                    length=group_count,
                    index=pd.RangeIndex(group_count),
                )
            ),
            null_key_policy=null_key_policy,
            sorted_order_is_identity=True,
            all_groups_observed=all_groups_observed,
            group_size_min=group_size_min,
            group_size_max=group_size_max,
            strictly_disjoint_group_bounds=strictly_disjoint_group_bounds,
        )

    @property
    def is_device(self) -> bool:
        return _is_device_array(self.group_codes)

    @property
    def resolved_group_count(self) -> int:
        if self.group_count is None:
            return _resolve_group_count(self.group_codes, None)
        return int(self.group_count)

    def reduce_numeric(self, values: Any, reducer: str) -> NativeGroupedReduction:
        """Reduce one all-valid numeric vector by dense group codes."""
        normalized = reducer.lower()
        if normalized not in {
            "sum",
            "count",
            "mean",
            "min",
            "max",
            "first",
            "last",
            "any",
            "all",
        }:
            raise ValueError(
                "NativeGrouped numeric reducer must be sum, count, mean, min, max, first, last, any, or all"
            )
        if _array_size(values) != _array_size(self.group_codes):
            raise ValueError("NativeGrouped reducer values length must match group codes")
        if not _is_numeric_like(values):
            raise TypeError("NativeGrouped numeric reducer admits only numeric or bool values")

        xp = _array_namespace_for(self.group_codes, values)
        values_array = xp.asarray(values)
        codes = xp.asarray(self.group_codes, dtype=xp.int32)
        observed_mask = (codes >= 0) & (codes < self.resolved_group_count)
        observed_codes = codes[observed_mask]
        observed_values = values_array[observed_mask]

        if normalized == "count":
            reduced = _bincount_count(
                observed_codes,
                group_count=self.resolved_group_count,
            )
        elif normalized == "sum":
            reduced = _bincount_sum(
                observed_codes,
                observed_values,
                group_count=self.resolved_group_count,
            )
        elif normalized in {"any", "all"}:
            if not np.issubdtype(_normalized_dtype(values_array), np.bool_):
                raise TypeError("NativeGrouped any/all reducers admit only bool values")
            true_counts = _bincount_sum(
                observed_codes,
                observed_values,
                group_count=self.resolved_group_count,
            )
            if normalized == "any":
                reduced = true_counts > 0
            else:
                counts = _bincount_count(
                    observed_codes,
                    group_count=self.resolved_group_count,
                )
                reduced = true_counts == counts
        elif normalized in {"min", "max"}:
            group_ids = xp.asarray(self.group_ids, dtype=xp.int32)
            has_all_groups = _array_size(group_ids) == self.resolved_group_count
            if xp is np:
                out_dtype = (
                    _normalized_dtype(values_array)
                    if has_all_groups
                    else _empty_group_extrema_dtype(values_array)
                )
                reduced = xp.empty(self.resolved_group_count, dtype=out_dtype)
                reduced[...] = _extrema_identity(out_dtype, normalized)
                reducer_func = xp.minimum if normalized == "min" else xp.maximum
                reducer_func.at(
                    reduced,
                    observed_codes,
                    observed_values.astype(out_dtype, copy=False),
                )
                if not has_all_groups:
                    _mark_missing_extrema_groups(
                        xp,
                        reduced,
                        group_ids,
                        self.resolved_group_count,
                    )
            else:
                reduced = _cupy_extrema_scatter(
                    xp,
                    observed_codes,
                    observed_values,
                    group_ids,
                    group_count=self.resolved_group_count,
                    reducer=normalized,
                    has_all_groups=has_all_groups,
                )
        elif normalized in {"first", "last"}:
            group_ids = xp.asarray(self.group_ids, dtype=xp.int32)
            has_all_groups = _array_size(group_ids) == self.resolved_group_count
            out_dtype = (
                _normalized_dtype(values_array)
                if has_all_groups
                else _empty_group_extrema_dtype(values_array)
            )
            reduced = xp.empty(self.resolved_group_count, dtype=out_dtype)
            if not has_all_groups:
                reduced[...] = xp.nan
            if _array_size(group_ids):
                sorted_order = xp.asarray(self.sorted_order, dtype=xp.int64)
                if normalized == "first":
                    value_positions = xp.asarray(self.group_offsets[:-1], dtype=xp.int64)
                else:
                    value_positions = xp.asarray(self.group_offsets[1:], dtype=xp.int64) - 1
                selected_rows = sorted_order[value_positions]
                reduced[group_ids] = values_array[selected_rows].astype(
                    out_dtype,
                    copy=False,
                )
        else:
            sums = _bincount_sum(
                observed_codes,
                observed_values.astype(xp.float64, copy=False),
                group_count=self.resolved_group_count,
            ).astype(xp.float64, copy=False)
            counts = _bincount_count(
                observed_codes,
                group_count=self.resolved_group_count,
            )
            reduced = xp.empty(self.resolved_group_count, dtype=xp.float64)
            reduced[...] = xp.nan
            has_values = counts > 0
            reduced[has_values] = sums[has_values] / counts[has_values].astype(xp.float64)

        return NativeGroupedReduction(
            values=reduced,
            reducer=normalized,
            group_count=self.resolved_group_count,
            output_index_plan=self.output_index_plan,
        )

    def reduce_expression(self, expression: Any, reducer: str) -> NativeGroupedReduction:
        """Reduce a private scalar expression without public Series materialization."""
        from vibespatial.api._native_expression import NativeExpression

        if not isinstance(expression, NativeExpression):
            raise TypeError("NativeGrouped.reduce_expression expects NativeExpression")
        if expression.source_row_count is not None and int(
            expression.source_row_count
        ) != _array_size(self.group_codes):
            raise ValueError("NativeGrouped expression source row count must match group codes")
        if (
            expression.source_token is not None
            and self.source_token is not None
            and expression.source_token != self.source_token
        ):
            raise ValueError("NativeGrouped expression source token does not match")
        return self.reduce_numeric(expression.values, reducer)

    def reduce_take(self, values: Any, reducer: str) -> NativeGroupedReduction:
        """Select first/last non-null host values by dense group codes."""
        normalized = reducer.lower()
        if normalized not in {"first", "last"}:
            raise ValueError("NativeGrouped take reducer must be first or last")
        if self.is_device:
            raise TypeError("NativeGrouped take reducers require host group codes")
        if _array_size(values) != _array_size(self.group_codes):
            raise ValueError("NativeGrouped reducer values length must match group codes")

        series = _as_host_series(values)
        group_count = self.resolved_group_count
        if self.output_index_plan is not None and self.output_index_plan.index is not None:
            index = self.output_index_plan.index
        else:
            index = pd.RangeIndex(group_count)

        if series.dtype == object:
            result_values = np.empty(group_count, dtype=object)
            result_values[:] = None
            result = pd.Series(result_values, index=index, name=series.name)
        else:
            result = pd.Series(index=index, dtype=series.dtype, name=series.name)

        codes = np.asarray(self.group_codes, dtype=np.int32)
        not_null = ~pd.isna(series).to_numpy(dtype=bool, na_value=True)
        observed_mask = (codes >= 0) & (codes < group_count) & not_null
        if np.any(observed_mask):
            observed_codes = codes[observed_mask]
            observed_rows = np.flatnonzero(observed_mask).astype(np.int64, copy=False)
            order = np.argsort(observed_codes, kind="stable")
            sorted_codes = observed_codes[order]
            sorted_rows = observed_rows[order]
            group_ids, counts = np.unique(sorted_codes, return_counts=True)
            offsets = np.concatenate(
                [
                    np.asarray([0], dtype=np.int64),
                    np.cumsum(counts.astype(np.int64, copy=False)),
                ],
            )
            if normalized == "first":
                value_positions = offsets[:-1]
            else:
                value_positions = offsets[1:] - 1
            selected_rows = sorted_rows[value_positions]
            from vibespatial.api._native_public_arrays import (
                NativeAttributeColumnArray,
            )

            if isinstance(series.array, NativeAttributeColumnArray):
                output_positions = np.full(group_count, -1, dtype=np.int64)
                output_positions[group_ids.astype(np.intp, copy=False)] = selected_rows
                result = pd.Series(
                    series.array.take(
                        output_positions,
                        allow_fill=True,
                        fill_value=None,
                    ),
                    index=index,
                    name=series.name,
                )
            else:
                result.iloc[group_ids.astype(np.intp, copy=False)] = series.iloc[
                    selected_rows
                ].array

        return NativeGroupedReduction(
            values=result,
            reducer=normalized,
            group_count=group_count,
            output_index_plan=self.output_index_plan,
        )

    def reduce_numeric_columns(
        self,
        columns: Mapping[Any, Any],
        reducers: str | Mapping[Any, str],
    ) -> NativeGroupedAttributeReduction:
        """Reduce all-valid numeric columns by dense group codes."""
        if not columns:
            return NativeGroupedAttributeReduction(
                columns={},
                group_count=self.resolved_group_count,
                output_index_plan=self.output_index_plan,
            )
        if isinstance(reducers, str):
            reducer_by_column = {name: reducers for name in columns}
        else:
            reducer_by_column = dict(reducers)
            missing = [name for name in columns if name not in reducer_by_column]
            if missing:
                raise ValueError(
                    "NativeGrouped reducers missing columns: "
                    + ", ".join(str(name) for name in missing)
                )
        reduced: dict[Any, NativeGroupedReduction] = {}
        for name, values in columns.items():
            reduced[name] = self.reduce_numeric(values, reducer_by_column[name])
        return NativeGroupedAttributeReduction(
            columns=reduced,
            group_count=self.resolved_group_count,
            output_index_plan=self.output_index_plan,
        )

    def reduce_take_columns(
        self,
        columns: Mapping[Any, Any],
        reducers: str | Mapping[Any, str],
    ) -> NativeGroupedAttributeReduction:
        """Select first/last host columns by dense group codes."""
        if not columns:
            return NativeGroupedAttributeReduction(
                columns={},
                group_count=self.resolved_group_count,
                output_index_plan=self.output_index_plan,
            )
        if isinstance(reducers, str):
            reducer_by_column = {name: reducers for name in columns}
        else:
            reducer_by_column = dict(reducers)
            missing = [name for name in columns if name not in reducer_by_column]
            if missing:
                raise ValueError(
                    "NativeGrouped reducers missing columns: "
                    + ", ".join(str(name) for name in missing)
                )
        reduced: dict[Any, NativeGroupedReduction] = {}
        for name, values in columns.items():
            reduced[name] = self.reduce_take(values, reducer_by_column[name])
        return NativeGroupedAttributeReduction(
            columns=reduced,
            group_count=self.resolved_group_count,
            output_index_plan=self.output_index_plan,
        )


__all__ = [
    "NativeGrouped",
    "NativeGroupedAttributeReduction",
    "NativeGroupedReduction",
    "NativeGroupedSelection",
]
