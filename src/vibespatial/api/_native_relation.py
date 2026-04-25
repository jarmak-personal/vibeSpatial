from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vibespatial.api._native_grouped import (
    NativeGrouped,
    NativeGroupedAttributeReduction,
    NativeGroupedReduction,
)
from vibespatial.api._native_rowset import NativeRowSet


def _is_device_array(values: Any) -> bool:
    return hasattr(values, "__cuda_array_interface__")


def _array_size(values: Any) -> int:
    shape = getattr(values, "shape", None)
    if shape is not None:
        return int(shape[0])
    return len(values)


def _positions_array(values: Any):
    if _is_device_array(values):
        import cupy as cp

        return cp.asarray(values, dtype=cp.int64)
    return np.asarray(values, dtype=np.int64)


def _valid_positions(values: Any, *, source_row_count: int | None = None):
    positions = _positions_array(values)
    mask = positions >= 0
    if source_row_count is not None:
        mask = mask & (positions < int(source_row_count))
    return positions[mask]


def _unique_positions(values: Any, *, source_row_count: int | None = None):
    positions = _valid_positions(values, source_row_count=source_row_count)
    if _is_device_array(positions):
        import cupy as cp

        return cp.unique(positions)
    return np.unique(positions)


def _unique_positions_first_seen(values: Any, *, source_row_count: int | None = None):
    positions = _valid_positions(values, source_row_count=source_row_count)
    if _array_size(positions) == 0:
        return positions
    if _is_device_array(positions):
        import cupy as cp

        unique, first_indices = cp.unique(positions, return_index=True)
        return unique[cp.argsort(first_indices)]
    unique, first_indices = np.unique(positions, return_index=True)
    return unique[np.argsort(first_indices, kind="stable")]


def _anti_positions(values: Any, *, source_row_count: int):
    matched = _valid_positions(values, source_row_count=source_row_count)
    if _is_device_array(matched):
        import cupy as cp

        keep = cp.ones(int(source_row_count), dtype=cp.bool_)
        if int(matched.size) > 0:
            keep[matched] = False
        return cp.nonzero(keep)[0].astype(cp.int64, copy=False)

    keep = np.ones(int(source_row_count), dtype=bool)
    if matched.size > 0:
        keep[matched] = False
    return np.nonzero(keep)[0].astype(np.int64, copy=False)


def _match_counts(values: Any, *, source_row_count: int):
    matched = _valid_positions(values, source_row_count=source_row_count)
    if _is_device_array(matched):
        import cupy as cp

        return cp.bincount(matched, minlength=int(source_row_count))[
            : int(source_row_count)
        ].astype(cp.int64, copy=False)
    return np.bincount(matched, minlength=int(source_row_count))[
        : int(source_row_count)
    ].astype(np.int64, copy=False)


def _gather_values(values: Any, indices: Any):
    if _is_device_array(values) or _is_device_array(indices):
        import cupy as cp

        return cp.asarray(values)[cp.asarray(indices, dtype=cp.int64)]
    return np.asarray(values)[np.asarray(indices, dtype=np.int64)]


def _resolve_row_count(
    explicit: int | None,
    stored: int | None,
    *,
    side: str,
) -> int:
    row_count = stored if explicit is None else explicit
    if row_count is None:
        raise ValueError(f"{side} source row count is required for this relation view")
    return int(row_count)


@dataclass(frozen=True)
class NativeRelation:
    """Private relation-pair carrier for join-style row flow."""

    left_indices: Any
    right_indices: Any
    left_token: str | None = None
    right_token: str | None = None
    predicate: str | None = None
    distances: Any | None = None
    left_row_count: int | None = None
    right_row_count: int | None = None
    sorted_by_left: bool = False
    left_group_offsets: Any | None = None
    duplicate_policy: str = "preserve"

    @classmethod
    def from_relation_index_result(
        cls,
        result,
        *,
        left_token: str | None = None,
        right_token: str | None = None,
        predicate: str | None = None,
        distances: Any | None = None,
        left_row_count: int | None = None,
        right_row_count: int | None = None,
    ) -> NativeRelation:
        return cls(
            left_indices=result.left_indices,
            right_indices=result.right_indices,
            left_token=left_token,
            right_token=right_token,
            predicate=predicate,
            distances=distances,
            left_row_count=left_row_count,
            right_row_count=right_row_count,
        )

    def __len__(self) -> int:
        left_size = _array_size(self.left_indices)
        right_size = _array_size(self.right_indices)
        if left_size != right_size:
            raise ValueError(
                f"NativeRelation pair length mismatch: {left_size} != {right_size}"
            )
        return left_size

    def left_rowset(self, *, unique: bool = False) -> NativeRowSet:
        if unique:
            return self.left_semijoin_rowset()
        return NativeRowSet.from_positions(
            self.left_indices,
            source_token=self.left_token,
            source_row_count=self.left_row_count,
            ordered=self.sorted_by_left,
            unique=False,
        )

    def right_rowset(self, *, unique: bool = False) -> NativeRowSet:
        if unique:
            return self.right_semijoin_rowset()
        return NativeRowSet.from_positions(
            self.right_indices,
            source_token=self.right_token,
            source_row_count=self.right_row_count,
            ordered=False,
            unique=False,
        )

    def left_semijoin_rowset(self, *, order: str = "sorted") -> NativeRowSet:
        """Rows from the left source that have at least one relation pair."""
        if order == "sorted":
            positions = _unique_positions(
                self.left_indices,
                source_row_count=self.left_row_count,
            )
        elif order == "first":
            positions = _unique_positions_first_seen(
                self.left_indices,
                source_row_count=self.left_row_count,
            )
        else:
            raise ValueError("NativeRelation semijoin order must be 'sorted' or 'first'")
        return NativeRowSet.from_positions(
            positions,
            source_token=self.left_token,
            source_row_count=self.left_row_count,
            ordered=True,
            unique=True,
        )

    def right_semijoin_rowset(self, *, order: str = "sorted") -> NativeRowSet:
        """Rows from the right source that have at least one relation pair."""
        if order == "sorted":
            positions = _unique_positions(
                self.right_indices,
                source_row_count=self.right_row_count,
            )
        elif order == "first":
            positions = _unique_positions_first_seen(
                self.right_indices,
                source_row_count=self.right_row_count,
            )
        else:
            raise ValueError("NativeRelation semijoin order must be 'sorted' or 'first'")
        return NativeRowSet.from_positions(
            positions,
            source_token=self.right_token,
            source_row_count=self.right_row_count,
            ordered=True,
            unique=True,
        )

    def left_antijoin_rowset(
        self,
        *,
        source_row_count: int | None = None,
    ) -> NativeRowSet:
        """Rows from the left source that have no relation pair."""
        row_count = _resolve_row_count(
            source_row_count,
            self.left_row_count,
            side="left",
        )
        positions = _anti_positions(self.left_indices, source_row_count=row_count)
        return NativeRowSet.from_positions(
            positions,
            source_token=self.left_token,
            source_row_count=row_count,
            ordered=True,
            unique=True,
        )

    def right_antijoin_rowset(
        self,
        *,
        source_row_count: int | None = None,
    ) -> NativeRowSet:
        """Rows from the right source that have no relation pair."""
        row_count = _resolve_row_count(
            source_row_count,
            self.right_row_count,
            side="right",
        )
        positions = _anti_positions(self.right_indices, source_row_count=row_count)
        return NativeRowSet.from_positions(
            positions,
            source_token=self.right_token,
            source_row_count=row_count,
            ordered=True,
            unique=True,
        )

    def left_match_counts(self, *, source_row_count: int | None = None):
        row_count = _resolve_row_count(
            source_row_count,
            self.left_row_count,
            side="left",
        )
        return _match_counts(self.left_indices, source_row_count=row_count)

    def right_match_counts(self, *, source_row_count: int | None = None):
        row_count = _resolve_row_count(
            source_row_count,
            self.right_row_count,
            side="right",
        )
        return _match_counts(self.right_indices, source_row_count=row_count)

    def grouped_by_left(self, *, source_row_count: int | None = None) -> NativeGrouped:
        """Group relation pairs by left source row position."""
        row_count = _resolve_row_count(
            source_row_count,
            self.left_row_count,
            side="left",
        )
        return NativeGrouped.from_dense_codes(
            self.left_indices,
            group_count=row_count,
            source_token=self.left_token,
        )

    def grouped_by_right(self, *, source_row_count: int | None = None) -> NativeGrouped:
        """Group relation pairs by right source row position."""
        row_count = _resolve_row_count(
            source_row_count,
            self.right_row_count,
            side="right",
        )
        return NativeGrouped.from_dense_codes(
            self.right_indices,
            group_count=row_count,
            source_token=self.right_token,
        )

    def left_reduce_right_numeric(
        self,
        right_values: Any,
        reducer: str,
        *,
        left_row_count: int | None = None,
    ) -> NativeGroupedReduction:
        """Reduce right-side numeric values into left-row groups."""
        if self.right_row_count is not None and _array_size(right_values) != int(
            self.right_row_count
        ):
            raise ValueError("right_values length must match right_row_count")
        pair_values = _gather_values(right_values, self.right_indices)
        return self.grouped_by_left(source_row_count=left_row_count).reduce_numeric(
            pair_values,
            reducer,
        )

    def left_reduce_right_numeric_columns(
        self,
        right_columns,
        reducers,
        *,
        left_row_count: int | None = None,
    ) -> NativeGroupedAttributeReduction:
        """Reduce right-side numeric columns into left-row groups."""
        if self.right_row_count is not None:
            bad = [
                name
                for name, values in dict(right_columns).items()
                if _array_size(values) != int(self.right_row_count)
            ]
            if bad:
                raise ValueError("right_columns lengths must match right_row_count")
        pair_columns = {
            name: _gather_values(values, self.right_indices)
            for name, values in dict(right_columns).items()
        }
        return self.grouped_by_left(source_row_count=left_row_count).reduce_numeric_columns(
            pair_columns,
            reducers,
        )

    def right_reduce_left_numeric(
        self,
        left_values: Any,
        reducer: str,
        *,
        right_row_count: int | None = None,
    ) -> NativeGroupedReduction:
        """Reduce left-side numeric values into right-row groups."""
        if self.left_row_count is not None and _array_size(left_values) != int(
            self.left_row_count
        ):
            raise ValueError("left_values length must match left_row_count")
        pair_values = _gather_values(left_values, self.left_indices)
        return self.grouped_by_right(source_row_count=right_row_count).reduce_numeric(
            pair_values,
            reducer,
        )

    def right_reduce_left_numeric_columns(
        self,
        left_columns,
        reducers,
        *,
        right_row_count: int | None = None,
    ) -> NativeGroupedAttributeReduction:
        """Reduce left-side numeric columns into right-row groups."""
        if self.left_row_count is not None:
            bad = [
                name
                for name, values in dict(left_columns).items()
                if _array_size(values) != int(self.left_row_count)
            ]
            if bad:
                raise ValueError("left_columns lengths must match left_row_count")
        pair_columns = {
            name: _gather_values(values, self.left_indices)
            for name, values in dict(left_columns).items()
        }
        return self.grouped_by_right(source_row_count=right_row_count).reduce_numeric_columns(
            pair_columns,
            reducers,
        )


__all__ = ["NativeRelation"]
