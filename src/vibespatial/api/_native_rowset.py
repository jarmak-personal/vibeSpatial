from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from vibespatial.runtime.materialization import (
    MaterializationBoundary,
    record_materialization_event,
)


def _is_device_array(values: Any) -> bool:
    return hasattr(values, "__cuda_array_interface__")


def _array_size(values: Any) -> int:
    shape = getattr(values, "shape", None)
    if shape is not None:
        return int(shape[0])
    return len(values)


def _array_namespace_for(*values: Any):
    if any(_is_device_array(value) for value in values):
        import cupy as cp

        return cp
    return np


def _as_position_array(values: Any, xp):
    return xp.asarray(values, dtype=xp.int64)


def _host_positions_for_public_index_take(
    row_positions: Any,
    *,
    strict_disallowed: bool,
) -> np.ndarray:
    if _is_device_array(row_positions):
        import cupy as cp

        positions = cp.asarray(row_positions)
        if positions.dtype == cp.bool_ or positions.dtype == bool:
            positions = cp.flatnonzero(positions)
        positions = positions.astype(cp.int64, copy=False)
        item_count = int(getattr(positions, "size", len(positions)))
        itemsize = int(getattr(getattr(positions, "dtype", None), "itemsize", 0))
        record_materialization_event(
            surface="vibespatial.api.NativeIndexPlan.take_public_index",
            boundary=MaterializationBoundary.INTERNAL_HOST_CONVERSION,
            operation="index_plan_take_positions_to_host",
            reason="device row positions were materialized to take host public index labels",
            detail=f"rows={item_count}, bytes={item_count * itemsize}",
            d2h_transfer=True,
            strict_disallowed=strict_disallowed,
        )
        from vibespatial.cuda._runtime import get_cuda_runtime

        host_positions = get_cuda_runtime().copy_device_to_host(
            positions,
            reason=(
                "vibespatial.api.NativeIndexPlan.take_public_index"
                "::index_plan_take_positions_to_host"
            ),
        )
        return np.asarray(host_positions, dtype=np.int64)

    positions = np.asarray(row_positions)
    if positions.dtype == bool:
        positions = np.flatnonzero(positions)
    return np.asarray(positions, dtype=np.int64)


def _validate_compatible_rowsets(
    left: NativeRowSet,
    right: NativeRowSet,
) -> tuple[str | None, int | None]:
    if (
        left.source_token is not None
        and right.source_token is not None
        and left.source_token != right.source_token
    ):
        raise ValueError("NativeRowSet source token mismatch")
    if (
        left.source_row_count is not None
        and right.source_row_count is not None
        and int(left.source_row_count) != int(right.source_row_count)
    ):
        raise ValueError("NativeRowSet source row count mismatch")

    source_token = left.source_token if left.source_token is not None else right.source_token
    source_row_count = (
        int(left.source_row_count)
        if left.source_row_count is not None
        else (
            int(right.source_row_count)
            if right.source_row_count is not None
            else None
        )
    )
    return source_token, source_row_count


def _rowset_mask(rowset: NativeRowSet, row_count: int, xp):
    positions = _as_position_array(rowset.positions, xp)
    if xp is np and positions.size:
        if bool(np.any((positions < 0) | (positions >= row_count))):
            raise ValueError("NativeRowSet positions must be within source_row_count")
    mask = xp.zeros(int(row_count), dtype=xp.bool_)
    mask[positions] = True
    return mask


@dataclass(frozen=True)
class NativeIndexPlan:
    """Private mapping from native row positions to public index labels."""

    kind: str
    length: int
    index: pd.Index | None = None
    name: Any | None = None
    nlevels: int = 1
    has_duplicates: bool = False
    device_labels: Any | None = None

    @classmethod
    def from_index(cls, index: pd.Index) -> NativeIndexPlan:
        if isinstance(index, pd.RangeIndex):
            return cls(
                kind="range",
                length=len(index),
                index=index,
                name=index.name,
                nlevels=index.nlevels,
                has_duplicates=False,
            )
        return cls(
            kind="host-labels",
            length=len(index),
            index=index,
            name=index.name,
            nlevels=index.nlevels,
            has_duplicates=not index.is_unique,
        )

    def validate_length(self, length: int) -> None:
        if int(length) != self.length:
            raise ValueError(
                f"NativeIndexPlan length mismatch: expected {self.length}, got {length}"
            )

    def take_public_index(
        self,
        row_positions,
        *,
        strict_disallowed: bool = True,
    ) -> pd.Index:
        host_positions = _host_positions_for_public_index_take(
            row_positions,
            strict_disallowed=strict_disallowed,
        )
        if self.index is None:
            return self.to_public_index(
                strict_disallowed=strict_disallowed,
            ).take(host_positions)
        return self.index.take(host_positions)

    def take(
        self,
        row_positions,
        *,
        preserve_index: bool = True,
        unique: bool = False,
        strict_disallowed: bool = True,
    ) -> NativeIndexPlan:
        """Return the index plan after taking row positions.

        RangeIndex sources can preserve labels on device by storing the
        computed label vector as a private device carrier. Public pandas index
        objects are built only at an explicit export boundary.
        """
        length = _array_size(row_positions)
        if not preserve_index:
            return type(self).from_index(pd.RangeIndex(length))

        if _is_device_array(row_positions) and self.kind == "range" and isinstance(
            self.index,
            pd.RangeIndex,
        ):
            import cupy as cp

            positions = cp.asarray(row_positions, dtype=cp.int64)
            labels = positions * np.int64(self.index.step) + np.int64(self.index.start)
            return type(self)(
                kind="device-labels",
                length=length,
                index=None,
                name=self.name,
                nlevels=1,
                has_duplicates=not bool(unique),
                device_labels=labels,
            )

        if (
            _is_device_array(row_positions)
            and self.kind == "device-labels"
            and self.device_labels is not None
        ):
            import cupy as cp

            positions = cp.asarray(row_positions, dtype=cp.int64)
            labels = cp.asarray(self.device_labels)[positions]
            return type(self)(
                kind="device-labels",
                length=length,
                index=None,
                name=self.name,
                nlevels=self.nlevels,
                has_duplicates=self.has_duplicates or not bool(unique),
                device_labels=labels,
            )

        return type(self).from_index(
            self.take_public_index(
                row_positions,
                strict_disallowed=strict_disallowed,
            )
        )

    def to_public_index(
        self,
        *,
        surface: str = "vibespatial.api.NativeIndexPlan.to_public_index",
        strict_disallowed: bool = True,
    ) -> pd.Index:
        """Materialize public labels for compatibility export."""
        if self.index is not None:
            return self.index
        if self.device_labels is not None:
            item_count = int(getattr(self.device_labels, "size", self.length))
            itemsize = int(getattr(getattr(self.device_labels, "dtype", None), "itemsize", 0))
            record_materialization_event(
                surface=surface,
                boundary=MaterializationBoundary.INTERNAL_HOST_CONVERSION,
                operation="index_plan_to_host",
                reason="device public index labels were materialized for export",
                detail=f"rows={item_count}, bytes={item_count * itemsize}",
                d2h_transfer=True,
                strict_disallowed=strict_disallowed,
            )
            from vibespatial.cuda._runtime import get_cuda_runtime

            labels = get_cuda_runtime().copy_device_to_host(
                self.device_labels,
                reason=f"{surface}::index_plan_to_host",
            )
            return pd.Index(labels, name=self.name)
        return pd.RangeIndex(self.length, name=self.name)

    @property
    def admits_unique_label_selection(self) -> bool:
        """Whether row-position semijoin can model public unique-label selection."""
        return not self.has_duplicates and self.nlevels == 1


@dataclass(frozen=True)
class NativeRowSet:
    """Private row-flow carrier using device row positions as the currency."""

    positions: Any
    source_token: str | None = None
    source_row_count: int | None = None
    ordered: bool = True
    unique: bool = False
    identity: bool = False

    @classmethod
    def from_positions(
        cls,
        positions: Any,
        *,
        source_token: str | None = None,
        source_row_count: int | None = None,
        ordered: bool = True,
        unique: bool = False,
        identity: bool = False,
    ) -> NativeRowSet:
        return cls(
            positions=positions,
            source_token=source_token,
            source_row_count=source_row_count,
            ordered=ordered,
            unique=unique,
            identity=identity,
        )

    @property
    def is_device(self) -> bool:
        return _is_device_array(self.positions)

    def __len__(self) -> int:
        return _array_size(self.positions)

    def _combine(self, other: NativeRowSet, operation: str) -> NativeRowSet:
        if not isinstance(other, NativeRowSet):
            raise TypeError("NativeRowSet set operations require another NativeRowSet")
        source_token, source_row_count = _validate_compatible_rowsets(self, other)
        xp = _array_namespace_for(self.positions, other.positions)

        if source_row_count is not None:
            left_mask = _rowset_mask(self, source_row_count, xp)
            right_mask = _rowset_mask(other, source_row_count, xp)
            if operation == "intersection":
                mask = left_mask & right_mask
            elif operation == "union":
                mask = left_mask | right_mask
            elif operation == "difference":
                mask = left_mask & ~right_mask
            else:
                raise ValueError("unsupported NativeRowSet set operation")
            positions = xp.nonzero(mask)[0].astype(xp.int64, copy=False)
        elif xp is np:
            left = np.unique(_as_position_array(self.positions, np))
            right = np.unique(_as_position_array(other.positions, np))
            if operation == "intersection":
                positions = np.intersect1d(left, right, assume_unique=True)
            elif operation == "union":
                positions = np.union1d(left, right)
            elif operation == "difference":
                positions = np.setdiff1d(left, right, assume_unique=True)
            else:
                raise ValueError("unsupported NativeRowSet set operation")
        else:
            raise ValueError(
                "NativeRowSet device set operations require source_row_count"
            )

        return type(self).from_positions(
            positions,
            source_token=source_token,
            source_row_count=source_row_count,
            ordered=True,
            unique=True,
        )

    def intersection(self, other: NativeRowSet) -> NativeRowSet:
        return self._combine(other, "intersection")

    def intersect(self, other: NativeRowSet) -> NativeRowSet:
        return self.intersection(other)

    def union(self, other: NativeRowSet) -> NativeRowSet:
        return self._combine(other, "union")

    def difference(self, other: NativeRowSet) -> NativeRowSet:
        return self._combine(other, "difference")

    def to_host_positions(
        self,
        *,
        surface: str = "vibespatial.api.NativeRowSet.to_host_positions",
        strict_disallowed: bool = True,
    ) -> np.ndarray:
        if self.is_device:
            record_materialization_event(
                surface=surface,
                boundary=MaterializationBoundary.INTERNAL_HOST_CONVERSION,
                operation="rowset_to_host",
                reason="device row positions were materialized on host",
                d2h_transfer=True,
                strict_disallowed=strict_disallowed,
            )
            from vibespatial.cuda._runtime import get_cuda_runtime

            return get_cuda_runtime().copy_device_to_host(
                self.positions,
                reason=f"{surface}::rowset_to_host",
            ).astype(np.int64, copy=False)
        return np.asarray(self.positions, dtype=np.int64)


__all__ = ["NativeIndexPlan", "NativeRowSet"]
