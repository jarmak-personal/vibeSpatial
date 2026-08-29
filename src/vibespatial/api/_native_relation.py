from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vibespatial.api._native_grouped import (
    NativeGrouped,
    NativeGroupedAttributeReduction,
    NativeGroupedReduction,
    NativeGroupedSelection,
)
from vibespatial.api._native_rowset import NativeDeviceSelection, NativeRowSet


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

    from vibespatial.cuda._runtime import pylibcudf_column_from_device

    dtype = cp.int32
    if source_row_count is None or int(source_row_count) > np.iinfo(np.int32).max:
        dtype = cp.int64
    return pylibcudf_column_from_device(cp.asarray(indices, dtype=dtype))


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
class _NativeRelationFamilySelection:
    """Device-counted span into one shared grouped relation."""

    source_offset: Any
    logical_count: Any
    launch_capacity: int

    def __post_init__(self) -> None:
        if not _is_device_array(self.source_offset) or not _is_device_array(
            self.logical_count
        ):
            raise TypeError("relation family span metadata must be device-resident")
        if tuple(getattr(self.source_offset, "shape", ())) != (1,) or tuple(
            getattr(self.logical_count, "shape", ())
        ) != (1,):
            raise ValueError("relation family span metadata must be one-dimensional")
        if int(self.launch_capacity) <= 0:
            raise ValueError("relation family launch capacity must be positive")


@dataclass(frozen=True)
class _NativeRelationFamilyGroup:
    """One homogeneous family span selected from a grouped relation tile."""

    left_indices: Any
    right_indices: Any
    source_positions: Any
    selection: _NativeRelationFamilySelection
    left_family: Any
    right_family: Any

    @property
    def capacity(self) -> int:
        return self.selection.launch_capacity

    @property
    def source_offset(self):
        return self.selection.source_offset

    @property
    def logical_count(self):
        return self.selection.logical_count

@dataclass(frozen=True)
class NativeRelationFamilyPartition:
    """All-family grouped partition of one device relation capacity.

    Active relation lanes are classified by packed ``(left_tag, right_tag)``
    code and radix-partitioned once. Dense device counts and offsets describe
    every possible family-pair span. Specialized predicate kernels consume
    capacity-backed family views without a host cardinality read or another
    mask scan/scatter partition.
    """

    left_indices: Any
    right_indices: Any
    source_positions: Any
    group_offsets: Any
    group_counts: Any
    family_count: int

    def __post_init__(self) -> None:
        if not all(
            _is_device_array(values)
            for values in (
                self.left_indices,
                self.right_indices,
                self.source_positions,
                self.group_offsets,
                self.group_counts,
            )
        ):
            raise TypeError("relation family partition requires device arrays")
        capacity = _array_size(self.left_indices)
        if _array_size(self.right_indices) != capacity:
            raise ValueError("relation family partition indices must align")
        if _array_size(self.source_positions) != capacity:
            raise ValueError("relation family partition source positions must align")
        group_count = int(self.family_count) ** 2
        if _array_size(self.group_counts) != group_count:
            raise ValueError("relation family partition counts must match family domain")
        if _array_size(self.group_offsets) != group_count + 1:
            raise ValueError("relation family partition offsets must delimit every group")

    @classmethod
    def from_pair_capacity(
        cls,
        left_indices: Any,
        right_indices: Any,
        pair_active: Any,
        left_tags: Any,
        right_tags: Any,
        *,
        family_count: int,
        source_positions: Any | None = None,
    ) -> NativeRelationFamilyPartition:
        """Classify and partition all family pairs in one device pass."""
        if not all(
            _is_device_array(values)
            for values in (
                left_indices,
                right_indices,
                pair_active,
                left_tags,
                right_tags,
            )
        ):
            raise TypeError("relation family partitions require device arrays")
        family_count = int(family_count)
        if family_count <= 0:
            raise ValueError("relation family partition requires a positive family count")

        import cupy as cp

        from vibespatial.cuda.cccl_primitives import PairSortStrategy, sort_pairs

        d_left = cp.asarray(left_indices, dtype=cp.int32)
        d_right = cp.asarray(right_indices, dtype=cp.int32)
        d_pair_active = cp.asarray(pair_active, dtype=cp.bool_)
        if d_left.ndim != 1 or d_right.shape != d_left.shape:
            raise ValueError("relation family partition indices must align")
        if d_pair_active.shape != d_left.shape:
            raise ValueError("relation family partition activity must align")
        d_source_positions = (
            cp.arange(d_left.size, dtype=cp.int32)
            if source_positions is None
            else cp.asarray(source_positions, dtype=cp.int32)
        )
        if d_source_positions.shape != d_left.shape:
            raise ValueError("relation family source positions must align")

        d_left_tags = cp.asarray(left_tags, dtype=cp.int8)
        d_right_tags = cp.asarray(right_tags, dtype=cp.int8)
        if d_left_tags.ndim != 1 or d_right_tags.ndim != 1:
            raise ValueError("relation family partition tags must be one-dimensional")

        d_pair_left_tags = d_left_tags[d_left].astype(cp.int32, copy=False)
        d_pair_right_tags = d_right_tags[d_right].astype(cp.int32, copy=False)
        d_valid_family = (
            d_pair_active
            & (d_pair_left_tags >= 0)
            & (d_pair_left_tags < family_count)
            & (d_pair_right_tags >= 0)
            & (d_pair_right_tags < family_count)
        )
        group_count = family_count**2
        d_family_pair_codes = cp.where(
            d_valid_family,
            d_pair_left_tags * cp.int32(family_count) + d_pair_right_tags,
            cp.int32(group_count),
        ).astype(cp.int32, copy=False)
        capacity = int(d_left.size)
        partitioned = sort_pairs(
            d_family_pair_codes,
            cp.arange(capacity, dtype=cp.int32),
            strategy=PairSortStrategy.RADIX,
            synchronize=False,
        )
        d_order = cp.asarray(partitioned.values, dtype=cp.int32)
        d_group_counts = cp.bincount(
            d_family_pair_codes,
            minlength=group_count + 1,
        )[:group_count].astype(cp.int64, copy=False)
        d_group_offsets = cp.empty(group_count + 1, dtype=cp.int64)
        d_group_offsets[0] = 0
        d_group_offsets[1:] = cp.cumsum(d_group_counts, dtype=cp.int64)
        return cls(
            left_indices=d_left[d_order].astype(cp.int32, copy=False),
            right_indices=d_right[d_order].astype(cp.int32, copy=False),
            source_positions=d_source_positions[d_order].astype(cp.int32, copy=False),
            group_offsets=d_group_offsets,
            group_counts=d_group_counts,
            family_count=family_count,
        )

    @property
    def capacity(self) -> int:
        return _array_size(self.left_indices)

    def family_pair(
        self,
        *,
        left_family: Any,
        right_family: Any,
        left_family_tag: int,
        right_family_tag: int,
        launch_capacity: int,
    ) -> _NativeRelationFamilyGroup:
        """Return metadata for one span of the shared grouped relation."""
        left_family_tag = int(left_family_tag)
        right_family_tag = int(right_family_tag)
        family_count = int(self.family_count)
        if not 0 <= left_family_tag < family_count:
            raise ValueError("left family tag is outside the partition domain")
        if not 0 <= right_family_tag < family_count:
            raise ValueError("right family tag is outside the partition domain")

        group_code = left_family_tag * family_count + right_family_tag
        selection = _NativeRelationFamilySelection(
            source_offset=self.group_offsets[group_code : group_code + 1],
            logical_count=self.group_counts[group_code : group_code + 1],
            launch_capacity=launch_capacity,
        )
        return _NativeRelationFamilyGroup(
            left_indices=self.left_indices,
            right_indices=self.right_indices,
            source_positions=self.source_positions,
            selection=selection,
            left_family=left_family,
            right_family=right_family,
        )


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
    origin: str | None = None

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
            sorted_by_left=bool(getattr(result, "sorted_by_left", False)),
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
        identity = (
            order == "sorted"
            and self.left_row_count is not None
            and _array_size(positions) == int(self.left_row_count)
        )
        return NativeRowSet.from_positions(
            positions,
            source_token=self.left_token,
            source_row_count=self.left_row_count,
            ordered=True,
            unique=True,
            identity=identity,
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
        identity = (
            order == "sorted"
            and self.right_row_count is not None
            and _array_size(positions) == int(self.right_row_count)
        )
        return NativeRowSet.from_positions(
            positions,
            source_token=self.right_token,
            source_row_count=self.right_row_count,
            ordered=True,
            unique=True,
            identity=identity,
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

    def filter_by_distance_selection(
        self,
        op: str,
        scalar: float,
    ) -> NativeRelationSelection | NativeRelation:
        """Filter distances without compacting a device-backed pair relation."""
        selection = self.distance_expression().compare_scalar_selection(op, scalar)
        if isinstance(selection, NativeDeviceSelection):
            return self.filter_pairs_selection(selection)
        return self.filter_pairs(selection)

    def filter_pairs_selection(
        self,
        selection: NativeDeviceSelection,
    ) -> NativeRelationSelection:
        """Keep a dynamic pair filter row-indirected and capacity-backed."""
        return NativeRelationSelection(relation=self, selection=selection)

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
        keep = self._equal_columns_mask(left_columns, right_columns)
        pair_rowset = NativeRowSet.from_positions(
            _nonzero_positions(keep),
            source_row_count=len(self),
            ordered=self.sorted_by_left,
            unique=False,
        )
        return self.filter_pairs(pair_rowset)

    def filter_by_equal_columns_selection(
        self,
        left_columns,
        right_columns,
    ) -> NativeRelationSelection | NativeRelation:
        """Filter equal pair attributes without device cardinality compaction."""
        left_columns = dict(left_columns)
        right_columns = dict(right_columns)
        if set(left_columns) != set(right_columns):
            raise ValueError("left_columns and right_columns must have matching keys")
        if not left_columns:
            return self
        keep = self._equal_columns_mask(left_columns, right_columns)
        if _is_device_array(keep):
            return self.filter_pairs_selection(
                NativeDeviceSelection.from_mask(
                    keep,
                    source_row_count=len(self),
                )
            )
        pair_rowset = NativeRowSet.from_positions(
            _nonzero_positions(keep),
            source_row_count=len(self),
            ordered=self.sorted_by_left,
            unique=False,
        )
        return self.filter_pairs(pair_rowset)

    def _equal_columns_mask(self, left_columns, right_columns):
        left_columns = dict(left_columns)
        right_columns = dict(right_columns)
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
            return self._equal_pylibcudf_columns_mask(left_columns, right_columns)

        keep = None
        for name, left_values in left_columns.items():
            right_values = right_columns[name]
            column_keep = _gather_values(left_values, self.left_indices) == _gather_values(
                right_values,
                self.right_indices,
            )
            keep = column_keep if keep is None else keep & column_keep

        return keep

    def _equal_pylibcudf_columns_mask(
        self,
        left_columns,
        right_columns,
    ):
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

            from vibespatial.cuda._runtime import pylibcudf_current_stream
        except ModuleNotFoundError as exc:  # pragma: no cover - optional GPU dependency
            raise ValueError("pylibcudf is required for device column equality") from exc

        stream = pylibcudf_current_stream(
            *left_columns.values(),
            *right_columns.values(),
        )
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
                stream=stream,
            ).columns()[0]
            right_gathered = plc.copying.gather(
                plc.Table([right_column]),
                right_map,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                stream=stream,
            ).columns()[0]
            equal = plc.binaryop.binary_operation(
                left_gathered,
                right_gathered,
                plc.binaryop.BinaryOperator.EQUAL,
                bool_type,
                stream=stream,
            )
            keep_column = (
                equal
                if keep_column is None
                else plc.binaryop.binary_operation(
                    keep_column,
                    equal,
                    plc.binaryop.BinaryOperator.LOGICAL_AND,
                    bool_type,
                    stream=stream,
                )
            )

        return _device_bool_column_values(keep_column)

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


@dataclass(frozen=True)
class NativeRelationSelection:
    """Capacity-backed relation view with device-resident cardinality.

    Selected positions remain a compact prefix at source-pair capacity.
    Aggregate consumers map inactive lanes to one sentinel group and therefore
    avoid both a host count fence and a compact relation allocation.
    """

    relation: NativeRelation
    selection: NativeDeviceSelection

    def __post_init__(self) -> None:
        pair_count = len(self.relation)
        if (
            self.selection.source_row_count is not None
            and int(self.selection.source_row_count) != pair_count
        ):
            raise ValueError(
                "NativeRelationSelection source row count must match relation pairs"
            )
        if self.selection.capacity > pair_count:
            raise ValueError(
                "NativeRelationSelection capacity cannot exceed relation pair count"
            )
        if (
            self.selection.source_row_count is None
            and self.selection.capacity != pair_count
        ):
            raise ValueError(
                "NativeRelationSelection bounded capacity requires source_row_count"
            )

    @property
    def capacity(self) -> int:
        return self.selection.capacity

    @property
    def logical_count(self):
        return self.selection.logical_count

    @property
    def is_device(self) -> bool:
        return True

    def __len__(self) -> int:
        raise TypeError(
            "NativeRelationSelection logical length is device-resident; use capacity "
            "for native work or explicitly compact the selection"
        )

    def _grouped(self, side: str, *, row_count: int) -> NativeGroupedSelection:
        source = (
            self.relation.left_indices
            if side == "left"
            else self.relation.right_indices
        )
        token = self.relation.left_token if side == "left" else self.relation.right_token
        return NativeGroupedSelection(
            group_codes=source,
            selection=self.selection,
            group_count=int(row_count),
            source_token=token,
        )

    def _match_counts(self, side: str, *, row_count: int):
        source = (
            self.relation.left_indices
            if side == "left"
            else self.relation.right_indices
        )
        return self._grouped(side, row_count=row_count).reduce_numeric(
            source,
            "count",
        ).values

    def left_match_count_expression(
        self,
        *,
        source_row_count: int | None = None,
        operation: str = "relation.selection.left_match_count",
    ):
        from vibespatial.api._native_expression import NativeExpression

        row_count = _resolve_row_count(
            source_row_count,
            self.relation.left_row_count,
            side="left",
        )
        counts = self._match_counts("left", row_count=row_count)
        return NativeExpression(
            operation=operation,
            values=counts,
            source_token=self.relation.left_token,
            source_row_count=row_count,
            dtype=_dtype_name(counts),
        )

    def right_match_count_expression(
        self,
        *,
        source_row_count: int | None = None,
        operation: str = "relation.selection.right_match_count",
    ):
        from vibespatial.api._native_expression import NativeExpression

        row_count = _resolve_row_count(
            source_row_count,
            self.relation.right_row_count,
            side="right",
        )
        counts = self._match_counts("right", row_count=row_count)
        return NativeExpression(
            operation=operation,
            values=counts,
            source_token=self.relation.right_token,
            source_row_count=row_count,
            dtype=_dtype_name(counts),
        )

    def _reduce_distances(
        self,
        side: str,
        reducer: str,
        *,
        row_count: int,
    ) -> NativeGroupedReduction:
        if self.relation.distances is None:
            raise ValueError("selected relation distance reduction requires distances")
        return self._grouped(side, row_count=row_count).reduce_numeric(
            self.relation.distances,
            reducer,
        )

    def left_reduce_distances(
        self,
        reducer: str,
        *,
        left_row_count: int | None = None,
    ) -> NativeGroupedReduction:
        row_count = _resolve_row_count(
            left_row_count,
            self.relation.left_row_count,
            side="left",
        )
        return self._reduce_distances("left", reducer, row_count=row_count)

    def right_reduce_distances(
        self,
        reducer: str,
        *,
        right_row_count: int | None = None,
    ) -> NativeGroupedReduction:
        row_count = _resolve_row_count(
            right_row_count,
            self.relation.right_row_count,
            side="right",
        )
        return self._reduce_distances("right", reducer, row_count=row_count)

    def left_reduce_right_numeric(
        self,
        right_values: Any,
        reducer: str,
        *,
        left_row_count: int | None = None,
    ) -> NativeGroupedReduction:
        """Reduce right-source values by selected left relation rows.

        Physical shape: relation-capacity gather followed by a selection-aware
        grouped reduction. Inactive capacity lanes remain masked by
        ``NativeGroupedSelection``; no logical-pair count or pair compaction is
        exported to the host.
        """
        row_count = _resolve_row_count(
            left_row_count,
            self.relation.left_row_count,
            side="left",
        )
        if self.relation.right_row_count is not None and _array_size(
            right_values
        ) != int(self.relation.right_row_count):
            raise ValueError("right_values length must match right_row_count")
        import cupy as cp

        active_source = self.selection.source_mask()
        safe_right_indices = cp.where(
            active_source,
            cp.asarray(self.relation.right_indices),
            cp.int64(0),
        )
        pair_values = _gather_values(
            right_values,
            safe_right_indices,
        )
        return self._grouped("left", row_count=row_count).reduce_numeric(
            pair_values,
            reducer,
        )

    def right_reduce_left_numeric(
        self,
        left_values: Any,
        reducer: str,
        *,
        right_row_count: int | None = None,
    ) -> NativeGroupedReduction:
        """Reduce left-source values by selected right relation rows."""
        row_count = _resolve_row_count(
            right_row_count,
            self.relation.right_row_count,
            side="right",
        )
        if self.relation.left_row_count is not None and _array_size(
            left_values
        ) != int(self.relation.left_row_count):
            raise ValueError("left_values length must match left_row_count")
        import cupy as cp

        active_source = self.selection.source_mask()
        safe_left_indices = cp.where(
            active_source,
            cp.asarray(self.relation.left_indices),
            cp.int64(0),
        )
        pair_values = _gather_values(
            left_values,
            safe_left_indices,
        )
        return self._grouped("right", row_count=row_count).reduce_numeric(
            pair_values,
            reducer,
        )

    def physicalize_geometries(
        self,
        left_geometry,
        right_geometry,
    ) -> NativeRelationGeometrySelection:
        """Build null-padded pair geometry capacity without a count fence."""
        if (
            self.relation.left_row_count is not None
            and int(left_geometry.row_count) != int(self.relation.left_row_count)
        ):
            raise ValueError("left geometry rows must match relation left_row_count")
        if (
            self.relation.right_row_count is not None
            and int(right_geometry.row_count) != int(self.relation.right_row_count)
        ):
            raise ValueError("right geometry rows must match relation right_row_count")

        active = self.selection.active_capacity_mask()
        left_rows = self.selection.gather_capacity(
            self.relation.left_indices,
            fill_value=0,
        )
        right_rows = self.selection.gather_capacity(
            self.relation.right_indices,
            fill_value=0,
        )
        return NativeRelationGeometrySelection(
            left_geometry=left_geometry.device_take_capacity(left_rows, active),
            right_geometry=right_geometry.device_take_capacity(right_rows, active),
            selection=self.selection,
            broadcast_right_geometry=(
                right_geometry
                if self.relation.right_row_count == 1
                else None
            ),
        )


@dataclass(frozen=True)
class NativeRelationGeometrySelection:
    """Capacity-physicalized pair geometries plus device logical cardinality."""

    left_geometry: Any
    right_geometry: Any
    selection: NativeDeviceSelection
    broadcast_right_geometry: Any | None = None

    def __post_init__(self) -> None:
        if int(self.left_geometry.row_count) != self.selection.capacity:
            raise ValueError("left capacity geometry rows must match selection")
        if int(self.right_geometry.row_count) != self.selection.capacity:
            raise ValueError("right capacity geometry rows must match selection")

    @property
    def capacity(self) -> int:
        return self.selection.capacity

    @property
    def logical_count(self):
        return self.selection.logical_count

    def constructive_native(
        self,
        operation: str,
        **kwargs,
    ) -> NativeRelationConstructiveSelection:
        """Run pair-capacity constructive work with null inactive lanes."""
        import cupy as cp

        from vibespatial.constructive.binary_constructive import (
            binary_constructive_native,
        )
        from vibespatial.runtime import ExecutionMode

        if "dispatch_mode" in kwargs:
            raise TypeError(
                "NativeRelationGeometrySelection constructive dispatch is explicitly GPU"
            )
        result = None
        if operation == "intersection" and self.broadcast_right_geometry is not None:
            from vibespatial.api._native_result_core import GeometryNativeResult
            from vibespatial.constructive.binary_constructive import (
                broadcast_right_polygon_intersection_capacity_gpu,
            )

            broadcast_result = broadcast_right_polygon_intersection_capacity_gpu(
                self.left_geometry,
                self.broadcast_right_geometry,
                dispatch_mode=ExecutionMode.GPU,
            )
            if broadcast_result is not None:
                result = GeometryNativeResult.from_owned(broadcast_result, crs=None)
        if result is None:
            result = binary_constructive_native(
                operation,
                self.left_geometry,
                self.right_geometry,
                dispatch_mode=ExecutionMode.GPU,
                **kwargs,
            )
        if int(result.row_count) != self.capacity:
            raise RuntimeError(
                "capacity constructive result did not preserve physical row count"
            )
        owned = result.owned
        if owned is not None:
            active = self.selection.active_capacity_mask()
            state = owned._ensure_device_state(preserve_indexed_view=True)
            state.validity = cp.asarray(state.validity, dtype=cp.bool_) & active
            state.trusted_all_valid = True if self.capacity == 0 else False
            owned._cached_is_valid_mask = None
            owned._aligned_left_pairs_owned = self.left_geometry
            owned._aligned_right_pairs_owned = self.right_geometry
        return NativeRelationConstructiveSelection(
            geometry=result,
            selection=self.selection,
            operation=operation,
        )


@dataclass(frozen=True)
class NativeRelationConstructiveSelection:
    """Capacity constructive output with device-resident logical cardinality."""

    geometry: Any
    selection: NativeDeviceSelection
    operation: str

    def __post_init__(self) -> None:
        if int(self.geometry.row_count) != self.selection.capacity:
            raise ValueError("capacity constructive rows must match selection")

    @property
    def capacity(self) -> int:
        return self.selection.capacity

    @property
    def logical_count(self):
        return self.selection.logical_count


__all__ = [
    "NativeRelation",
    "NativeRelationConstructiveSelection",
    "NativeRelationGeometrySelection",
    "NativeRelationSelection",
]
