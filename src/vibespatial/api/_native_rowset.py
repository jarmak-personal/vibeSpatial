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

    def take_public_index(self, row_positions) -> pd.Index:
        if self.index is None:
            return self.to_public_index().take(row_positions)
        return self.index.take(row_positions)

    def take(
        self,
        row_positions,
        *,
        preserve_index: bool = True,
        unique: bool = False,
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

        return type(self).from_index(self.take_public_index(row_positions))

    def to_public_index(
        self,
        *,
        surface: str = "vibespatial.api.NativeIndexPlan.to_public_index",
    ) -> pd.Index:
        """Materialize public labels for compatibility export."""
        if self.index is not None:
            return self.index
        if self.device_labels is not None:
            import cupy as cp

            item_count = int(getattr(self.device_labels, "size", self.length))
            itemsize = int(getattr(getattr(self.device_labels, "dtype", None), "itemsize", 0))
            record_materialization_event(
                surface=surface,
                boundary=MaterializationBoundary.INTERNAL_HOST_CONVERSION,
                operation="index_plan_to_host",
                reason="device public index labels were materialized for export",
                detail=f"rows={item_count}, bytes={item_count * itemsize}",
                d2h_transfer=True,
            )
            return pd.Index(cp.asnumpy(self.device_labels), name=self.name)
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

    @classmethod
    def from_positions(
        cls,
        positions: Any,
        *,
        source_token: str | None = None,
        source_row_count: int | None = None,
        ordered: bool = True,
        unique: bool = False,
    ) -> NativeRowSet:
        return cls(
            positions=positions,
            source_token=source_token,
            source_row_count=source_row_count,
            ordered=ordered,
            unique=unique,
        )

    @property
    def is_device(self) -> bool:
        return _is_device_array(self.positions)

    def __len__(self) -> int:
        return _array_size(self.positions)

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
            import cupy as cp

            return cp.asnumpy(self.positions).astype(np.int64, copy=False)
        return np.asarray(self.positions, dtype=np.int64)


__all__ = ["NativeIndexPlan", "NativeRowSet"]
