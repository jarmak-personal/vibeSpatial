from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I64,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.runtime.materialization import (
    MaterializationBoundary,
    record_materialization_event,
)

_NATIVE_DEVICE_SELECTION_KERNEL_NAMES = ("scatter_selected_positions_i64",)

_NATIVE_DEVICE_SELECTION_KERNEL_SOURCE = r"""
extern "C" __global__ void scatter_selected_positions_i64(
    const unsigned char* __restrict__ selected,
    const long long* __restrict__ inclusive_offsets,
    long long* __restrict__ positions,
    long long row_count
) {
    const long long row =
        (long long)blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (row >= row_count) return;
    const long long selected_count = inclusive_offsets[row_count - 1LL];
    const long long destination = selected[row] != 0u
        ? inclusive_offsets[row] - 1LL
        : selected_count + row - inclusive_offsets[row];
    positions[destination] = row;
}
"""

request_nvrtc_warmup(
    [
        (
            "native-device-selection",
            _NATIVE_DEVICE_SELECTION_KERNEL_SOURCE,
            _NATIVE_DEVICE_SELECTION_KERNEL_NAMES,
        ),
    ]
)


def _native_device_selection_kernels():
    return compile_kernel_group(
        "native-device-selection",
        _NATIVE_DEVICE_SELECTION_KERNEL_SOURCE,
        _NATIVE_DEVICE_SELECTION_KERNEL_NAMES,
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
        else (int(right.source_row_count) if right.source_row_count is not None else None)
    )
    return source_token, source_row_count


def _rowset_mask(rowset: NativeRowSet, row_count: int, xp):
    positions = _as_position_array(rowset.positions, xp)
    if xp is np and positions.size:
        _validate_host_position_bounds(positions, row_count)
    mask = xp.zeros(int(row_count), dtype=xp.bool_)
    mask[positions] = True
    return mask


def _validate_host_position_bounds(positions: np.ndarray, row_count: int) -> None:
    if bool(np.any((positions < 0) | (positions >= row_count))):
        raise ValueError("NativeRowSet positions must be within source_row_count")


def _validate_rowset_bounds_if_host(rowset: NativeRowSet, row_count: int) -> None:
    if _is_device_array(rowset.positions):
        return
    positions = np.asarray(rowset.positions, dtype=np.int64)
    if positions.size:
        _validate_host_position_bounds(positions, row_count)


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
    source_index: pd.Index | None = None
    take_positions: Any | None = None
    selection_source_token: str | None = None
    selection_source_row_count: int | None = None
    selection_positions: Any | None = None
    selection_grouped: bool = False

    @classmethod
    def from_index(cls, index: pd.Index) -> NativeIndexPlan:
        # MultiIndex intentionally has no single backing array. Native public
        # index carriers are single-level ExtensionArrays, so only probe that
        # structural domain for an embedded plan.
        index_array = index.array if index.nlevels == 1 else None
        embedded_plan = getattr(index_array, "index_plan", None)
        if isinstance(embedded_plan, cls):
            embedded_plan.validate_length(len(index))
            return embedded_plan.with_name(index.name)
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

    def with_name(self, name: Any | None) -> NativeIndexPlan:
        """Return a single-level plan with renamed public index metadata.

        Deferred host-label takes retain their source index, so the public name
        must be applied there as well as on the logical plan. This is metadata
        only and never materializes device labels or row positions.
        """
        if self.nlevels != 1:
            raise ValueError("NativeIndexPlan.with_name requires a single-level index")
        index = None if self.index is None else self.index.rename(name)
        source_index = (
            None if self.source_index is None else self.source_index.rename(name)
        )
        return replace(
            self,
            index=index,
            source_index=source_index,
            name=name,
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

        if (
            _is_device_array(row_positions)
            and self.kind == "range"
            and isinstance(
                self.index,
                pd.RangeIndex,
            )
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
                source_index=self.index,
                take_positions=positions,
            )

        if self.kind == "device-labels" and self.device_labels is not None:
            import cupy as cp

            positions = cp.asarray(row_positions, dtype=cp.int64)
            labels = cp.asarray(self.device_labels)[positions]
            take_positions = (
                None
                if self.take_positions is None
                else cp.asarray(self.take_positions, dtype=cp.int64)[positions]
            )
            return type(self)(
                kind="device-labels",
                length=length,
                index=None,
                name=self.name,
                nlevels=self.nlevels,
                has_duplicates=self.has_duplicates or not bool(unique),
                device_labels=labels,
                source_index=self.source_index,
                take_positions=take_positions,
            )

        if (
            _is_device_array(row_positions)
            and self.kind == "host-labels"
            and self.index is not None
        ):
            import cupy as cp

            return type(self)(
                kind="host-labels-take",
                length=length,
                index=None,
                name=self.name,
                nlevels=self.nlevels,
                has_duplicates=self.has_duplicates or not bool(unique),
                source_index=self.index,
                take_positions=cp.asarray(row_positions, dtype=cp.int64),
            )

        if self.kind == "host-labels-take" and self.source_index is not None:
            if self.take_positions is None:
                return type(self).from_index(
                    self.take_public_index(
                        row_positions,
                        strict_disallowed=strict_disallowed,
                    )
                )
            if _is_device_array(self.take_positions) or _is_device_array(row_positions):
                import cupy as cp

                positions = cp.asarray(row_positions, dtype=cp.int64)
                selected = cp.asarray(self.take_positions, dtype=cp.int64)[positions]
                return type(self)(
                    kind="host-labels-take",
                    length=length,
                    index=None,
                    name=self.name,
                    nlevels=self.nlevels,
                    has_duplicates=self.has_duplicates or not bool(unique),
                    source_index=self.source_index,
                    take_positions=selected,
                )
            source_positions = np.asarray(self.take_positions, dtype=np.int64)[
                np.asarray(row_positions, dtype=np.int64)
            ]
            return type(self)(
                kind="host-labels-take",
                length=length,
                index=None,
                name=self.name,
                nlevels=self.nlevels,
                has_duplicates=self.has_duplicates or not bool(unique),
                source_index=self.source_index,
                take_positions=source_positions,
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
        if self.source_index is not None and self.take_positions is not None:
            host_positions = _host_positions_for_public_index_take(
                self.take_positions,
                strict_disallowed=strict_disallowed,
            )
            return self.source_index.take(host_positions)
        return pd.RangeIndex(self.length, name=self.name)

    def with_selection(
        self,
        row_positions,
        *,
        source_token: str,
        source_row_count: int,
        unique: bool,
        grouped: bool = False,
    ) -> NativeIndexPlan:
        """Attach source-frame row positions for native public label selection."""
        return replace(
            self,
            selection_source_token=source_token,
            selection_source_row_count=int(source_row_count),
            selection_positions=row_positions,
            selection_grouped=bool(grouped),
            has_duplicates=self.has_duplicates or not bool(unique),
        )

    @property
    def admits_unique_label_selection(self) -> bool:
        """Whether row-position semijoin can model public unique-label selection."""
        return not self.has_duplicates and self.nlevels == 1


@dataclass(frozen=True)
class NativeDeviceSelection:
    """Capacity-backed dynamic device selection with a device logical count.

    Physical shape: one predicate bit per source row is scan/scattered into an
    ordered compact prefix of ``positions``. ``positions`` remains allocated at
    source-row capacity, while ``logical_count`` is a one-element device vector.
    Native consumers launch over ``capacity`` and guard work with
    :meth:`active_capacity_mask`; only an explicit compaction/export boundary
    may read the logical count on the host.

    This is deliberately not a ``NativeRowSet``. A rowset's Python length is
    its logical length, while this carrier keeps that length device-resident.
    """

    positions: Any
    logical_count: Any
    source_token: str | None = None
    source_row_count: int | None = None
    ordered: bool = True
    unique: bool = True
    full_selection_implies_identity: bool = False
    geometry_family_domain: tuple[Any, ...] | None = None
    trusted_all_valid_rows: bool | None = None

    def __post_init__(self) -> None:
        if not _is_device_array(self.positions) or not _is_device_array(self.logical_count):
            raise TypeError("NativeDeviceSelection requires device arrays")
        if int(getattr(self.positions, "ndim", 0)) != 1:
            raise ValueError("NativeDeviceSelection positions must be one-dimensional")
        if tuple(getattr(self.logical_count, "shape", ())) != (1,):
            raise ValueError(
                "NativeDeviceSelection logical_count must be a one-element device vector"
            )
        if self.source_row_count is not None and self.capacity > int(self.source_row_count):
            raise ValueError("NativeDeviceSelection capacity cannot exceed source_row_count")

    @classmethod
    def from_mask(
        cls,
        mask: Any,
        *,
        source_token: str | None = None,
        source_row_count: int | None = None,
        geometry_family_domain: tuple[Any, ...] | None = None,
        trusted_all_valid_rows: bool | None = None,
    ) -> NativeDeviceSelection:
        """Build a stable selected-prefix/rejected-tail partition without a fence."""
        if not _is_device_array(mask):
            raise TypeError("NativeDeviceSelection.from_mask requires a device mask")

        import cupy as cp

        d_mask = cp.asarray(mask, dtype=cp.bool_)
        if d_mask.ndim != 1:
            raise ValueError("NativeDeviceSelection mask must be one-dimensional")
        capacity = int(d_mask.size)
        if source_row_count is None:
            source_row_count = capacity
        elif int(source_row_count) != capacity:
            raise ValueError("NativeDeviceSelection mask length must match source_row_count")

        d_positions = cp.empty(capacity, dtype=cp.int64)
        if capacity == 0:
            d_logical_count = cp.zeros(1, dtype=cp.int64)
        else:
            d_offsets = cp.cumsum(d_mask, dtype=cp.int64)
            d_logical_count = d_offsets[-1:]
            runtime = get_cuda_runtime()
            kernel = _native_device_selection_kernels()["scatter_selected_positions_i64"]
            grid, block = runtime.launch_config(kernel, capacity)
            ptr = runtime.pointer
            runtime.launch(
                kernel,
                grid=grid,
                block=block,
                params=(
                    (
                        ptr(d_mask),
                        ptr(d_offsets),
                        ptr(d_positions),
                        capacity,
                    ),
                    (
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_I64,
                    ),
                ),
            )
        return cls(
            positions=d_positions,
            logical_count=d_logical_count,
            source_token=source_token,
            source_row_count=int(source_row_count),
            full_selection_implies_identity=True,
            geometry_family_domain=geometry_family_domain,
            trusted_all_valid_rows=trusted_all_valid_rows,
        )

    @classmethod
    def identity(
        cls,
        row_count: int,
        *,
        source_token: str | None = None,
        geometry_family_domain: tuple[Any, ...] | None = None,
        trusted_all_valid_rows: bool | None = None,
    ) -> NativeDeviceSelection:
        """Build a device identity selection with no cardinality fence."""
        import cupy as cp

        row_count = int(row_count)
        return cls(
            positions=cp.arange(row_count, dtype=cp.int64),
            logical_count=cp.asarray([row_count], dtype=cp.int64),
            source_token=source_token,
            source_row_count=row_count,
            ordered=True,
            unique=True,
            full_selection_implies_identity=True,
            geometry_family_domain=geometry_family_domain,
            trusted_all_valid_rows=trusted_all_valid_rows,
        )

    @classmethod
    def concatenate(
        cls,
        selections: tuple[NativeDeviceSelection, ...] | list[NativeDeviceSelection],
        *,
        source_token: str | None = None,
    ) -> NativeDeviceSelection:
        """Concatenate compact prefixes without reading any logical count.

        Each input addresses a disjoint source partition. Active lanes are
        scattered into a global compact prefix in partition order; inactive
        lanes are assigned unique destinations in the remaining capacity.
        """
        import cupy as cp

        parts = tuple(selections)
        if not parts:
            return cls.identity(0, source_token=source_token)
        if any(part.source_row_count is None for part in parts):
            raise ValueError("concatenated NativeDeviceSelection parts require source_row_count")

        capacities = [part.capacity for part in parts]
        source_counts = [int(part.source_row_count) for part in parts]
        capacity_offsets: list[int] = []
        source_offsets: list[int] = []
        capacity_total = 0
        source_total = 0
        for capacity, source_count in zip(capacities, source_counts, strict=True):
            capacity_offsets.append(capacity_total)
            source_offsets.append(source_total)
            capacity_total += int(capacity)
            source_total += int(source_count)

        d_counts = cp.concatenate(
            [cp.asarray(part.logical_count, dtype=cp.int64) for part in parts]
        )
        d_logical_offsets = cp.cumsum(d_counts, dtype=cp.int64) - d_counts
        d_logical_count = cp.sum(d_counts, dtype=cp.int64).reshape(1)
        if capacity_total == 0:
            return cls(
                positions=cp.empty(0, dtype=cp.int64),
                logical_count=d_logical_count,
                source_token=source_token,
                source_row_count=source_total,
                ordered=all(part.ordered for part in parts),
                unique=all(part.unique for part in parts),
            )

        d_part_ids = cp.concatenate(
            [cp.full(capacity, index, dtype=cp.int64) for index, capacity in enumerate(capacities)]
        )
        d_lanes = cp.concatenate([cp.arange(capacity, dtype=cp.int64) for capacity in capacities])
        d_capacity_offsets = cp.asarray(capacity_offsets, dtype=cp.int64)
        d_part_counts = d_counts[d_part_ids]
        d_part_logical_offsets = d_logical_offsets[d_part_ids]
        d_active = d_lanes < d_part_counts
        d_active_destinations = d_part_logical_offsets + d_lanes
        d_inactive_destinations = (
            d_logical_count[0]
            + d_capacity_offsets[d_part_ids]
            - d_part_logical_offsets
            + d_lanes
            - d_part_counts
        )
        d_destinations = cp.where(
            d_active,
            d_active_destinations,
            d_inactive_destinations,
        )
        d_source_positions = cp.concatenate(
            [
                cp.asarray(part.partition_capacity_positions(), dtype=cp.int64)
                + cp.int64(source_offset)
                for part, source_offset in zip(parts, source_offsets, strict=True)
            ]
        )
        d_positions = cp.empty(capacity_total, dtype=cp.int64)
        d_positions[d_destinations] = d_source_positions
        family_domains = [part.geometry_family_domain for part in parts]
        geometry_family_domain = (
            tuple(
                dict.fromkeys(
                    family
                    for domain in family_domains
                    for family in domain or ()
                )
            )
            if all(domain is not None for domain in family_domains)
            else None
        )
        return cls(
            positions=d_positions,
            logical_count=d_logical_count,
            source_token=source_token,
            source_row_count=source_total,
            ordered=all(part.ordered for part in parts),
            unique=all(part.unique for part in parts),
            full_selection_implies_identity=all(
                part.full_selection_implies_identity for part in parts
            ),
            geometry_family_domain=geometry_family_domain,
            trusted_all_valid_rows=(
                True
                if all(part.trusted_all_valid_rows is True for part in parts)
                else None
            ),
        )

    @property
    def capacity(self) -> int:
        return _array_size(self.positions)

    @property
    def is_device(self) -> bool:
        return True

    def __len__(self) -> int:
        raise TypeError(
            "NativeDeviceSelection logical length is device-resident; use capacity "
            "for native launches or an explicit compaction/export boundary"
        )

    def active_capacity_mask(self):
        """Return a capacity-aligned device mask for the compact prefix."""
        import cupy as cp

        return (
            cp.arange(self.capacity, dtype=cp.int64)
            < cp.asarray(
                self.logical_count,
                dtype=cp.int64,
            )[0]
        )

    def safe_capacity_positions(self, *, fill_value: int = 0):
        """Return capacity positions with inactive lanes replaced on device."""
        import cupy as cp

        active = self.active_capacity_mask()
        return cp.where(
            active,
            cp.asarray(self.positions, dtype=cp.int64),
            cp.asarray(fill_value, dtype=cp.int64),
        )

    def partition_capacity_positions(self):
        """Return the full stable source-row partition for every capacity lane.

        Selections produced by :meth:`from_mask`, :meth:`identity`, and
        :meth:`concatenate` initialize the inactive tail as well as the active
        prefix. Consumers that preserve a separate active mask can therefore
        retain unique source provenance at physical capacity.
        """
        return self.positions

    def gather_capacity(self, values: Any, *, fill_value: Any = 0):
        """Gather selected values into capacity storage without reading count."""
        import cupy as cp

        d_values = cp.asarray(values)
        if self.source_row_count is not None and int(d_values.shape[0]) != int(
            self.source_row_count
        ):
            raise ValueError("NativeDeviceSelection gather source length mismatch")
        active = self.active_capacity_mask()
        safe_positions = self.safe_capacity_positions()
        gathered = d_values[safe_positions]
        active_shape = (self.capacity,) + (1,) * max(gathered.ndim - 1, 0)
        return cp.where(
            active.reshape(active_shape),
            gathered,
            cp.asarray(fill_value, dtype=gathered.dtype),
        )

    def source_mask(self, *, active_mask: Any | None = None):
        """Project active capacity lanes onto the source row domain.

        Inactive lanes are routed to capacity-sized scratch positions after the
        source domain. This keeps the projection device-only even though the
        compact prefix length remains device-resident.
        """
        if self.source_row_count is None:
            raise ValueError("NativeDeviceSelection source_mask requires source_row_count")

        import cupy as cp

        d_active = self.active_capacity_mask()
        if active_mask is not None:
            d_requested = cp.asarray(active_mask, dtype=cp.bool_)
            if d_requested.ndim != 1 or int(d_requested.size) != self.capacity:
                raise ValueError("NativeDeviceSelection source mask must match selection capacity")
            d_active &= d_requested

        source_row_count = int(self.source_row_count)
        d_lanes = cp.arange(self.capacity, dtype=cp.int64)
        d_destinations = cp.where(
            d_active,
            self.safe_capacity_positions(),
            np.int64(source_row_count) + d_lanes,
        )
        d_extended = cp.zeros(
            source_row_count + self.capacity,
            dtype=cp.bool_,
        )
        d_extended[d_destinations] = True
        return d_extended[:source_row_count]

    def as_capacity_prefix(
        self,
        *,
        source_token: str | None = None,
    ) -> NativeDeviceSelection:
        """Rebase a gathered capacity result onto its active compact prefix."""
        import cupy as cp

        return type(self)(
            positions=cp.arange(self.capacity, dtype=cp.int64),
            logical_count=self.logical_count,
            source_token=source_token,
            source_row_count=self.capacity,
            ordered=True,
            unique=True,
            full_selection_implies_identity=True,
            geometry_family_domain=self.geometry_family_domain,
            trusted_all_valid_rows=self.trusted_all_valid_rows,
        )

    def remap_source_positions(
        self,
        source_positions: Any,
        *,
        source_token: str,
        source_row_count: int,
    ) -> NativeDeviceSelection:
        """Map a capacity partition into an enclosing source-row domain.

        ``source_positions`` maps every row in the current source domain to a
        row in the enclosing domain. The complete stable partition is mapped,
        including its inactive tail, so downstream concatenation can preserve
        row indirection without a cardinality fence.
        """
        import cupy as cp

        d_source_positions = cp.asarray(source_positions, dtype=cp.int64)
        if d_source_positions.ndim != 1 or int(d_source_positions.size) != int(
            self.source_row_count if self.source_row_count is not None else -1
        ):
            raise ValueError("selection source-position map must match its source domain")
        return type(self)(
            positions=d_source_positions[
                cp.asarray(self.partition_capacity_positions(), dtype=cp.int64)
            ],
            logical_count=self.logical_count,
            source_token=source_token,
            source_row_count=int(source_row_count),
            ordered=self.ordered,
            unique=self.unique,
            full_selection_implies_identity=False,
            geometry_family_domain=self.geometry_family_domain,
            trusted_all_valid_rows=self.trusted_all_valid_rows,
        )

    def _host_logical_count(
        self,
        *,
        surface: str,
        strict_disallowed: bool,
    ) -> int:
        record_materialization_event(
            surface=surface,
            boundary=MaterializationBoundary.INTERNAL_HOST_CONVERSION,
            operation="device_selection_logical_count_to_host",
            reason=(
                "capacity-backed device selection logical count was materialized "
                "for explicit compaction/export"
            ),
            detail=f"capacity={self.capacity}, bytes=8",
            d2h_transfer=True,
            strict_disallowed=strict_disallowed,
        )
        host_count = get_cuda_runtime().copy_device_to_host(
            self.logical_count,
            reason=f"{surface}::device_selection_logical_count_to_host",
        )
        return int(np.asarray(host_count, dtype=np.int64)[0])

    def compact_rowset(
        self,
        *,
        surface: str = "vibespatial.api.NativeDeviceSelection.compact_rowset",
        strict_disallowed: bool = True,
    ) -> NativeRowSet:
        """Cross the explicit count fence and return a compact rowset."""
        count = self._host_logical_count(
            surface=surface,
            strict_disallowed=strict_disallowed,
        )
        return NativeRowSet.from_positions(
            self.positions[:count],
            source_token=self.source_token,
            source_row_count=self.source_row_count,
            ordered=self.ordered,
            unique=self.unique,
            identity=(
                self.full_selection_implies_identity
                and self.source_row_count is not None
                and count == int(self.source_row_count)
            ),
            geometry_family_domain=self.geometry_family_domain,
            trusted_all_valid_rows=self.trusted_all_valid_rows,
        )

    def to_host_positions(
        self,
        *,
        surface: str = "vibespatial.api.NativeDeviceSelection.to_host_positions",
        strict_disallowed: bool = True,
    ) -> np.ndarray:
        """Materialize the compact prefix at an explicit terminal boundary."""
        rowset = self.compact_rowset(
            surface=surface,
            strict_disallowed=strict_disallowed,
        )
        return rowset.to_host_positions(
            surface=surface,
            strict_disallowed=strict_disallowed,
        )


@dataclass(frozen=True)
class NativeRowSet:
    """Private row-flow carrier using device row positions as the currency."""

    positions: Any
    source_token: str | None = None
    source_row_count: int | None = None
    ordered: bool = True
    unique: bool = False
    identity: bool = False
    geometry_family_domain: tuple[Any, ...] | None = None
    trusted_all_valid_rows: bool | None = None

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
        geometry_family_domain: tuple[Any, ...] | None = None,
        trusted_all_valid_rows: bool | None = None,
    ) -> NativeRowSet:
        return cls(
            positions=positions,
            source_token=source_token,
            source_row_count=source_row_count,
            ordered=ordered,
            unique=unique,
            identity=identity,
            geometry_family_domain=geometry_family_domain,
            trusted_all_valid_rows=trusted_all_valid_rows,
        )

    @property
    def is_device(self) -> bool:
        return _is_device_array(self.positions)

    def __len__(self) -> int:
        return _array_size(self.positions)

    def _with_contract(
        self,
        *,
        source_token: str | None,
        source_row_count: int | None,
        ordered: bool | None = None,
        unique: bool | None = None,
        identity: bool | None = None,
    ) -> NativeRowSet:
        return type(self).from_positions(
            self.positions,
            source_token=source_token,
            source_row_count=source_row_count,
            ordered=self.ordered if ordered is None else ordered,
            unique=self.unique if unique is None else unique,
            identity=self.identity if identity is None else identity,
            geometry_family_domain=self.geometry_family_domain,
            trusted_all_valid_rows=self.trusted_all_valid_rows,
        )

    def _is_full_identity(self, source_row_count: int | None = None) -> bool:
        row_count = self.source_row_count if source_row_count is None else source_row_count
        return self.identity and row_count is not None and len(self) == int(row_count)

    def _combine(self, other: NativeRowSet, operation: str) -> NativeRowSet:
        if not isinstance(other, NativeRowSet):
            raise TypeError("NativeRowSet set operations require another NativeRowSet")
        source_token, source_row_count = _validate_compatible_rowsets(self, other)
        xp = _array_namespace_for(self.positions, other.positions)

        if source_row_count is not None:
            _validate_rowset_bounds_if_host(self, source_row_count)
            _validate_rowset_bounds_if_host(other, source_row_count)

        left_empty = len(self) == 0
        right_empty = len(other) == 0
        left_full_identity = self._is_full_identity(source_row_count)
        right_full_identity = other._is_full_identity(source_row_count)

        def _can_preserve_ordered_unique(rowset: NativeRowSet) -> bool:
            return rowset.ordered and rowset.unique

        def _empty_result() -> NativeRowSet:
            return type(self).from_positions(
                xp.empty(0, dtype=xp.int64),
                source_token=source_token,
                source_row_count=source_row_count,
                ordered=True,
                unique=True,
                identity=False,
                geometry_family_domain=(),
                trusted_all_valid_rows=True,
            )

        def _combine_family_domain() -> tuple[Any, ...] | None:
            left_domain = self.geometry_family_domain
            right_domain = other.geometry_family_domain
            if operation == "difference":
                return left_domain
            if operation == "union":
                if left_domain is None or right_domain is None:
                    return None
                return tuple(dict.fromkeys((*left_domain, *right_domain)))
            if left_domain is None:
                return right_domain
            if right_domain is None:
                return left_domain
            right_set = set(right_domain)
            return tuple(family for family in left_domain if family in right_set)

        def _combine_all_valid_rows() -> bool | None:
            left_valid = self.trusted_all_valid_rows is True
            right_valid = other.trusted_all_valid_rows is True
            if operation == "intersection":
                return True if left_valid or right_valid else None
            if operation == "union":
                return True if left_valid and right_valid else None
            if operation == "difference":
                return True if left_valid else None
            return None

        if operation == "intersection":
            if left_empty:
                return self._with_contract(
                    source_token=source_token,
                    source_row_count=source_row_count,
                    ordered=True,
                    unique=True,
                    identity=False,
                )
            if right_empty:
                return other._with_contract(
                    source_token=source_token,
                    source_row_count=source_row_count,
                    ordered=True,
                    unique=True,
                    identity=False,
                )
            if left_full_identity and right_full_identity:
                return self._with_contract(
                    source_token=source_token,
                    source_row_count=source_row_count,
                    ordered=True,
                    unique=True,
                    identity=True,
                )
            if left_full_identity and _can_preserve_ordered_unique(other):
                return other._with_contract(
                    source_token=source_token,
                    source_row_count=source_row_count,
                    ordered=True,
                    unique=True,
                    identity=False,
                )
            if right_full_identity and _can_preserve_ordered_unique(self):
                return self._with_contract(
                    source_token=source_token,
                    source_row_count=source_row_count,
                    ordered=True,
                    unique=True,
                    identity=False,
                )
        elif operation == "union":
            if left_full_identity:
                return self._with_contract(
                    source_token=source_token,
                    source_row_count=source_row_count,
                    ordered=True,
                    unique=True,
                    identity=True,
                )
            if right_full_identity:
                return other._with_contract(
                    source_token=source_token,
                    source_row_count=source_row_count,
                    ordered=True,
                    unique=True,
                    identity=True,
                )
            if left_empty and _can_preserve_ordered_unique(other):
                return other._with_contract(
                    source_token=source_token,
                    source_row_count=source_row_count,
                    ordered=True,
                    unique=True,
                    identity=other.identity,
                )
            if right_empty and _can_preserve_ordered_unique(self):
                return self._with_contract(
                    source_token=source_token,
                    source_row_count=source_row_count,
                    ordered=True,
                    unique=True,
                    identity=self.identity,
                )
        elif operation == "difference":
            if left_empty or right_full_identity:
                return _empty_result()
            if left_full_identity and right_empty:
                return self._with_contract(
                    source_token=source_token,
                    source_row_count=source_row_count,
                    ordered=True,
                    unique=True,
                    identity=True,
                )
            if right_empty and _can_preserve_ordered_unique(self):
                return self._with_contract(
                    source_token=source_token,
                    source_row_count=source_row_count,
                    ordered=True,
                    unique=True,
                    identity=self.identity,
                )
        else:
            raise ValueError("unsupported NativeRowSet set operation")

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
            raise ValueError("NativeRowSet device set operations require source_row_count")

        return type(self).from_positions(
            positions,
            source_token=source_token,
            source_row_count=source_row_count,
            ordered=True,
            unique=True,
            geometry_family_domain=_combine_family_domain(),
            trusted_all_valid_rows=_combine_all_valid_rows(),
        )

    def intersection(self, other: NativeRowSet) -> NativeRowSet:
        return self._combine(other, "intersection")

    def intersect(self, other: NativeRowSet) -> NativeRowSet:
        return self.intersection(other)

    def union(self, other: NativeRowSet) -> NativeRowSet:
        return self._combine(other, "union")

    def difference(self, other: NativeRowSet) -> NativeRowSet:
        return self._combine(other, "difference")

    def complement(self) -> NativeRowSet:
        """Return source rows absent from this rowset without host export."""
        if self.source_row_count is None:
            raise ValueError("NativeRowSet complement requires source_row_count")

        source_row_count = int(self.source_row_count)
        _validate_rowset_bounds_if_host(self, source_row_count)
        xp = _array_namespace_for(self.positions)
        if self._is_full_identity(source_row_count):
            positions = xp.empty(0, dtype=xp.int64)
        elif len(self) == 0:
            positions = xp.arange(source_row_count, dtype=xp.int64)
        else:
            mask = xp.ones(source_row_count, dtype=xp.bool_)
            mask[_as_position_array(self.positions, xp)] = False
            positions = xp.nonzero(mask)[0].astype(xp.int64, copy=False)

        return type(self).from_positions(
            positions,
            source_token=self.source_token,
            source_row_count=source_row_count,
            ordered=True,
            unique=True,
            identity=len(self) == 0,
        )

    def to_host_positions(
        self,
        *,
        surface: str = "vibespatial.api.NativeRowSet.to_host_positions",
        strict_disallowed: bool = True,
    ) -> np.ndarray:
        if (
            self.identity
            and self.source_row_count is not None
            and len(self) == int(self.source_row_count)
        ):
            return np.arange(int(self.source_row_count), dtype=np.int64)
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

            return (
                get_cuda_runtime()
                .copy_device_to_host(
                    self.positions,
                    reason=f"{surface}::rowset_to_host",
                )
                .astype(np.int64, copy=False)
            )
        return np.asarray(self.positions, dtype=np.int64)


__all__ = ["NativeDeviceSelection", "NativeIndexPlan", "NativeRowSet"]
