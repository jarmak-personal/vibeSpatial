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


def _is_pylibcudf_column(values: Any) -> bool:
    type_ = type(values)
    return bool(
        type_.__module__.startswith("pylibcudf.")
        and type_.__name__ == "Column"
        and hasattr(values, "size")
        and hasattr(values, "type")
    )


def _array_size(values: Any) -> int:
    shape = getattr(values, "shape", None)
    if shape is not None:
        return int(shape[0])
    size = getattr(values, "size", None)
    if callable(size):
        return int(size())
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


def _nonzero_positions(mask: Any):
    if _is_device_array(mask):
        import cupy as cp

        return cp.nonzero(mask)[0].astype(cp.int64, copy=False)
    return np.nonzero(mask)[0].astype(np.int64, copy=False)


def _device_bool_column_values(column: Any):
    if not _is_pylibcudf_column(column):
        raise TypeError("expected a pylibcudf Column")
    if int(column.null_count()) != 0 or int(column.offset()) != 0:
        raise ValueError("boolean device column must be all-valid and zero-offset")
    import cupy as cp

    return cp.asarray(column.data()).view(cp.bool_)[: int(column.size())]


def _device_gather_map(indices: Any, *, source_row_count: int | None):
    if not _is_device_array(indices):
        raise ValueError("device column equality requires device relation indices")
    import cupy as cp
    import pylibcudf as plc

    dtype = cp.int32
    if source_row_count is None or int(source_row_count) > np.iinfo(np.int32).max:
        dtype = cp.int64
    return plc.Column.from_cuda_array_interface(cp.asarray(indices, dtype=dtype))


def _dtype_name(values: Any) -> str | None:
    dtype = getattr(values, "dtype", None)
    if dtype is None:
        return None
    return str(np.dtype(dtype))


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

    def left_match_count_expression(
        self,
        *,
        source_row_count: int | None = None,
        operation: str = "relation.left_match_count",
    ):
        """Expose left-row relation match counts as a private expression."""
        row_count = _resolve_row_count(
            source_row_count,
            self.left_row_count,
            side="left",
        )
        counts = self.left_match_counts(source_row_count=row_count)
        from vibespatial.api._native_expression import NativeExpression

        return NativeExpression(
            operation=operation,
            values=counts,
            source_token=self.left_token,
            source_row_count=row_count,
            dtype=_dtype_name(counts),
        )

    def right_match_count_expression(
        self,
        *,
        source_row_count: int | None = None,
        operation: str = "relation.right_match_count",
    ):
        """Expose right-row relation match counts as a private expression."""
        row_count = _resolve_row_count(
            source_row_count,
            self.right_row_count,
            side="right",
        )
        counts = self.right_match_counts(source_row_count=row_count)
        from vibespatial.api._native_expression import NativeExpression

        return NativeExpression(
            operation=operation,
            values=counts,
            source_token=self.right_token,
            source_row_count=row_count,
            dtype=_dtype_name(counts),
        )

    def distance_expression(self, *, operation: str = "relation.distance"):
        """Expose pair distances as a private expression over relation pairs.

        Physical shape: relation-pair scalar flow.  The native input carrier is
        ``NativeRelation`` with a distance vector; the native output carrier is
        ``NativeExpression`` aligned to pair positions, not public rows.
        """
        if self.distances is None:
            raise ValueError("NativeRelation distance expression requires distances")
        pair_count = len(self)
        if _array_size(self.distances) != pair_count:
            raise ValueError("NativeRelation distances length must match pair count")

        from vibespatial.api._native_expression import NativeExpression

        return NativeExpression(
            operation=operation,
            values=self.distances,
            source_row_count=pair_count,
            dtype=_dtype_name(self.distances),
            precision="fp64",
        )

    def distance_rowset(self, op: str, scalar: float) -> NativeRowSet:
        """Return pair positions whose relation distance satisfies ``op``."""
        return self.distance_expression().compare_scalar(op, scalar)

    def filter_pairs(self, pair_rowset: NativeRowSet) -> NativeRelation:
        """Filter relation pairs by a private pair-position rowset."""
        if not isinstance(pair_rowset, NativeRowSet):
            raise TypeError("NativeRelation.filter_pairs expects NativeRowSet")
        pair_count = len(self)
        if (
            pair_rowset.source_row_count is not None
            and int(pair_rowset.source_row_count) != pair_count
        ):
            raise ValueError("NativeRelation pair rowset source_row_count mismatch")
        positions = pair_rowset.positions
        distances = (
            None
            if self.distances is None
            else _gather_values(self.distances, positions)
        )
        return type(self)(
            left_indices=_gather_values(self.left_indices, positions),
            right_indices=_gather_values(self.right_indices, positions),
            left_token=self.left_token,
            right_token=self.right_token,
            predicate=self.predicate,
            distances=distances,
            left_row_count=self.left_row_count,
            right_row_count=self.right_row_count,
            sorted_by_left=self.sorted_by_left and pair_rowset.ordered,
            duplicate_policy=self.duplicate_policy,
        )

    def filter_by_distance(self, op: str, scalar: float) -> NativeRelation:
        """Filter relation pairs by their private distance expression."""
        return self.filter_pairs(self.distance_rowset(op, scalar))

    def filter_by_equal_columns(self, left_columns, right_columns) -> NativeRelation:
        """Filter relation pairs where corresponding left/right columns match.

        Physical shape: relation-pair attribute predicate over an existing
        ``NativeRelation``.  The native inputs are the relation pair vectors and
        all-valid device source columns; the output carrier is another
        ``NativeRelation`` filtered by pair position. CuPy/NumPy numeric arrays
        use direct gathered vector comparison. ``pylibcudf.Column`` inputs use
        libcudf gather/equality so movement-only string, categorical, datetime,
        and numeric/bool join keys can stay device-resident.
        """
        left_columns = dict(left_columns)
        right_columns = dict(right_columns)
        if set(left_columns) != set(right_columns):
            raise ValueError("left_columns and right_columns must have matching keys")
        if not left_columns:
            return self
        if self.left_row_count is not None:
            bad = [
                name
                for name, values in left_columns.items()
                if _array_size(values) != int(self.left_row_count)
            ]
            if bad:
                raise ValueError("left_columns lengths must match left_row_count")
        if self.right_row_count is not None:
            bad = [
                name
                for name, values in right_columns.items()
                if _array_size(values) != int(self.right_row_count)
            ]
            if bad:
                raise ValueError("right_columns lengths must match right_row_count")

        uses_device_columns = any(
            _is_pylibcudf_column(values)
            for values in (*left_columns.values(), *right_columns.values())
        )
        if uses_device_columns:
            return self._filter_by_equal_pylibcudf_columns(left_columns, right_columns)

        keep = None
        for name, left_values in left_columns.items():
            right_values = right_columns[name]
            column_keep = _gather_values(left_values, self.left_indices) == _gather_values(
                right_values,
                self.right_indices,
            )
            keep = column_keep if keep is None else keep & column_keep

        pair_rowset = NativeRowSet.from_positions(
            _nonzero_positions(keep),
            source_row_count=len(self),
            ordered=self.sorted_by_left,
            unique=False,
        )
        return self.filter_pairs(pair_rowset)

    def _filter_by_equal_pylibcudf_columns(
        self,
        left_columns,
        right_columns,
    ) -> NativeRelation:
        if any(
            not _is_pylibcudf_column(values)
            for values in (*left_columns.values(), *right_columns.values())
        ):
            raise ValueError(
                "pylibcudf relation equality requires every join key as a device column"
            )
        if any(
            int(values.null_count()) != 0
            for values in (*left_columns.values(), *right_columns.values())
        ):
            raise ValueError("device relation equality requires all-valid columns")
        try:
            import pylibcudf as plc
        except ModuleNotFoundError as exc:  # pragma: no cover - optional GPU dependency
            raise ValueError("pylibcudf is required for device column equality") from exc

        left_map = _device_gather_map(
            self.left_indices,
            source_row_count=self.left_row_count,
        )
        right_map = _device_gather_map(
            self.right_indices,
            source_row_count=self.right_row_count,
        )
        bool_type = plc.types.DataType(plc.types.TypeId.BOOL8)
        keep_column = None
        for name, left_column in left_columns.items():
            right_column = right_columns[name]
            left_gathered = plc.copying.gather(
                plc.Table([left_column]),
                left_map,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            ).columns()[0]
            right_gathered = plc.copying.gather(
                plc.Table([right_column]),
                right_map,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            ).columns()[0]
            equal = plc.binaryop.binary_operation(
                left_gathered,
                right_gathered,
                plc.binaryop.BinaryOperator.EQUAL,
                bool_type,
            )
            keep_column = (
                equal
                if keep_column is None
                else plc.binaryop.binary_operation(
                    keep_column,
                    equal,
                    plc.binaryop.BinaryOperator.LOGICAL_AND,
                    bool_type,
                )
            )

        keep = _device_bool_column_values(keep_column)
        pair_rowset = NativeRowSet.from_positions(
            _nonzero_positions(keep),
            source_row_count=len(self),
            ordered=self.sorted_by_left,
            unique=False,
        )
        return self.filter_pairs(pair_rowset)

    def left_reduce_distances(
        self,
        reducer: str,
        *,
        left_row_count: int | None = None,
    ) -> NativeGroupedReduction:
        """Reduce relation distances into left-row groups."""
        expression = self.distance_expression()
        return self.grouped_by_left(source_row_count=left_row_count).reduce_expression(
            expression,
            reducer,
        )

    def right_reduce_distances(
        self,
        reducer: str,
        *,
        right_row_count: int | None = None,
    ) -> NativeGroupedReduction:
        """Reduce relation distances into right-row groups."""
        expression = self.distance_expression()
        return self.grouped_by_right(source_row_count=right_row_count).reduce_expression(
            expression,
            reducer,
        )

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
