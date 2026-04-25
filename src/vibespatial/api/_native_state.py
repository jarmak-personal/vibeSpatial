from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any
from uuid import uuid4
from weakref import ref

from vibespatial.api._native_rowset import NativeIndexPlan, NativeRowSet
from vibespatial.runtime.residency import Residency, combined_residency


@dataclass(frozen=True)
class NativeStreamReadiness:
    """Readiness metadata for private device carriers crossing API boundaries."""

    stream: Any | None = None
    event: Any | None = None
    ready: bool = True


@dataclass(frozen=True)
class NativeFrameState:
    """Private logical frame state beneath exact public GeoPandas objects."""

    attributes: Any
    geometry: Any
    geometry_name: str
    column_order: tuple[Any, ...]
    index_plan: NativeIndexPlan
    row_count: int
    secondary_geometry: tuple[Any, ...] = field(default_factory=tuple)
    attrs: dict[str, Any] = field(default_factory=dict)
    provenance: Any | None = None
    lineage_token: str = field(default_factory=lambda: uuid4().hex)
    residency: Residency = Residency.HOST
    readiness: NativeStreamReadiness = field(default_factory=NativeStreamReadiness)

    @classmethod
    def from_native_tabular_result(cls, result) -> NativeFrameState:
        attributes = result.attributes
        geometry = result.geometry
        row_count = len(attributes)
        if geometry.row_count != row_count:
            raise ValueError("native frame geometry row count must match attributes")
        return cls(
            attributes=attributes,
            geometry=geometry,
            geometry_name=result.geometry_name,
            column_order=tuple(result.column_order),
            index_plan=NativeIndexPlan.from_index(attributes.index),
            row_count=row_count,
            secondary_geometry=tuple(result.secondary_geometry),
            attrs=dict(result.attrs or {}),
            provenance=result.provenance,
            residency=combined_residency(getattr(geometry, "owned", None)),
        )

    def validate_row_count(self, row_count: int) -> None:
        if int(row_count) != self.row_count:
            raise ValueError(
                f"NativeFrameState row count mismatch: expected {self.row_count}, got {row_count}"
            )

    def project_columns(self, columns: tuple[Any, ...]) -> NativeFrameState | None:
        requested = tuple(columns)
        known = set(self.column_order)
        if any(column not in known for column in requested):
            return None
        if self.geometry_name not in requested:
            return None
        projected_attributes = _project_attributes(
            self.attributes,
            tuple(column for column in requested if column != self.geometry_name),
        )
        if projected_attributes is None:
            return None
        return replace(
            self,
            attributes=projected_attributes,
            column_order=requested,
        )

    def rename_columns(self, mapping: dict[Any, Any]) -> NativeFrameState | None:
        if not mapping:
            return self
        known_columns = set(self.column_order)
        rename_map = {
            old_name: new_name
            for old_name, new_name in mapping.items()
            if old_name in known_columns
        }
        if not rename_map:
            return self

        renamed_order = tuple(
            rename_map.get(column, column) for column in self.column_order
        )
        if len(set(renamed_order)) != len(renamed_order):
            return None

        from vibespatial.api._native_result_core import NativeAttributeTable

        attribute_table = NativeAttributeTable.from_value(self.attributes)
        attribute_columns = set(attribute_table.columns)
        renamed_attributes = attribute_table.rename_columns(
            {
                old_name: new_name
                for old_name, new_name in rename_map.items()
                if old_name in attribute_columns
            }
        )
        renamed_secondary_geometry = tuple(
            type(column)(rename_map.get(column.name, column.name), column.geometry)
            for column in self.secondary_geometry
        )
        return replace(
            self,
            attributes=renamed_attributes,
            geometry_name=rename_map.get(self.geometry_name, self.geometry_name),
            column_order=renamed_order,
            secondary_geometry=renamed_secondary_geometry,
        )

    def with_index(self, index: Any) -> NativeFrameState:
        index_plan = NativeIndexPlan.from_index(index)
        if index_plan.length != self.row_count:
            raise ValueError("native frame index length must match row count")
        return replace(self, index_plan=index_plan)

    def assign_attributes(
        self,
        values_by_name: dict[Any, Any],
        *,
        column_order: tuple[Any, ...],
    ) -> NativeFrameState | None:
        requested = tuple(column_order)
        if self.geometry_name not in requested:
            return None
        if len(set(requested)) != len(requested):
            return None
        if self.geometry_name in values_by_name:
            return None

        from vibespatial.api._native_result_core import NativeAttributeTable

        attribute_order = tuple(
            column for column in requested if column != self.geometry_name
        )
        attribute_table = NativeAttributeTable.from_value(self.attributes)
        assigned = {
            name: values
            for name, values in values_by_name.items()
            if name in attribute_order
        }
        updated_attributes = attribute_table.assign_columns(
            assigned,
            columns=attribute_order,
        )
        if updated_attributes is None:
            return None
        return replace(
            self,
            attributes=updated_attributes,
            column_order=requested,
        )

    def take(
        self,
        rowset: NativeRowSet,
        *,
        preserve_index: bool = True,
    ) -> NativeFrameState:
        if rowset.source_row_count is not None and rowset.source_row_count != self.row_count:
            raise ValueError(
                "NativeRowSet source row count does not match NativeFrameState"
            )
        if rowset.source_token is not None and rowset.source_token != self.lineage_token:
            raise ValueError("NativeRowSet source token does not match NativeFrameState")

        if rowset.is_device and preserve_index and self.index_plan.kind in {
            "range",
            "device-labels",
        }:
            normalized = rowset.positions
            attributes = self.attributes.take(normalized, preserve_index=False)
            geometry = self.geometry.take(normalized)
            secondary_geometry = tuple(
                type(column)(column.name, column.geometry.take(normalized))
                for column in self.secondary_geometry
            )
            return type(self)(
                attributes=attributes,
                geometry=geometry,
                geometry_name=self.geometry_name,
                column_order=self.column_order,
                index_plan=self.index_plan.take(
                    normalized,
                    preserve_index=True,
                    unique=rowset.unique,
                ),
                row_count=len(rowset),
                secondary_geometry=secondary_geometry,
                attrs=self.attrs,
                provenance=self.provenance,
                residency=combined_residency(getattr(geometry, "owned", None)),
                readiness=self.readiness,
            )

        taken = self.to_native_tabular_result().take(
            rowset.positions,
            preserve_index=preserve_index,
        )
        return type(self).from_native_tabular_result(taken)

    def to_native_tabular_result(self):
        from vibespatial.api._native_result_core import NativeTabularResult

        attributes = self.attributes
        public_index = self.index_plan.to_public_index(
            surface="vibespatial.api.NativeFrameState.to_native_tabular_result",
        )
        if not attributes.index.equals(public_index):
            attributes = attributes.with_index(public_index)
        return NativeTabularResult(
            attributes=attributes,
            geometry=self.geometry,
            geometry_name=self.geometry_name,
            column_order=self.column_order,
            attrs=self.attrs,
            secondary_geometry=self.secondary_geometry,
            provenance=self.provenance,
        )


@dataclass(frozen=True)
class NativeStateHandle:
    token: str
    generation: int
    row_count: int
    geometry_name: str
    column_order: tuple[Any, ...]
    index_kind: str
    index_name: Any | None
    index_nlevels: int
    index_has_duplicates: bool
    index_length: int
    lineage_token: str
    index_snapshot: Any | None = field(default=None, compare=False)


class NativeStateRegistry:
    """Weak object-to-state registry for sanctioned public object attachment."""

    def __init__(self) -> None:
        self._states: dict[
            int,
            tuple[ref, NativeStateHandle, NativeFrameState],
        ] = {}
        self._generation = 0

    def attach(self, owner: Any, state: NativeFrameState) -> NativeStateHandle:
        self._generation += 1
        owner_id = id(owner)
        index_snapshot = state.index_plan.index
        if index_snapshot is None:
            index_snapshot = getattr(owner, "index", None)
        handle = NativeStateHandle(
            token=uuid4().hex,
            generation=self._generation,
            row_count=state.row_count,
            geometry_name=state.geometry_name,
            column_order=state.column_order,
            index_kind=state.index_plan.kind,
            index_name=state.index_plan.name,
            index_nlevels=state.index_plan.nlevels,
            index_has_duplicates=state.index_plan.has_duplicates,
            index_length=state.index_plan.length,
            lineage_token=state.lineage_token,
            index_snapshot=index_snapshot,
        )
        self._states[owner_id] = (
            ref(owner, lambda _owner_ref, key=owner_id: self._states.pop(key, None)),
            handle,
            state,
        )
        return handle

    def drop(self, owner: Any) -> None:
        self._states.pop(id(owner), None)

    def get(self, owner: Any, handle: NativeStateHandle | None = None) -> NativeFrameState | None:
        entry = self._states.get(id(owner))
        if entry is None:
            return None
        owner_ref, current_handle, state = entry
        if owner_ref() is not owner:
            self.drop(owner)
            return None
        if handle is not None and current_handle != handle:
            return None
        if not _handle_matches_state(current_handle, state):
            self.drop(owner)
            return None
        if not _owner_matches_handle(owner, current_handle):
            self.drop(owner)
            return None
        return state


def _handle_matches_state(handle: NativeStateHandle, state: NativeFrameState) -> bool:
    index_matches = (
        _indexes_equal(handle.index_snapshot, state.index_plan.index)
        if state.index_plan.index is not None
        else handle.index_snapshot is not None
    )
    return (
        handle.row_count == state.row_count
        and handle.geometry_name == state.geometry_name
        and handle.column_order == state.column_order
        and handle.index_kind == state.index_plan.kind
        and handle.index_name == state.index_plan.name
        and handle.index_nlevels == state.index_plan.nlevels
        and handle.index_has_duplicates == state.index_plan.has_duplicates
        and handle.index_length == state.index_plan.length
        and index_matches
        and handle.lineage_token == state.lineage_token
    )


def _owner_matches_handle(owner: Any, handle: NativeStateHandle) -> bool:
    try:
        owner_row_count = len(owner)
    except Exception:
        owner_row_count = handle.row_count
    if int(owner_row_count) != handle.row_count:
        return False

    geometry_name = getattr(owner, "_geometry_column_name", None)
    if geometry_name is not None and geometry_name != handle.geometry_name:
        return False

    columns = getattr(owner, "columns", None)
    if columns is not None and tuple(columns) != handle.column_order:
        return False

    index = getattr(owner, "index", None)
    if index is not None and not _indexes_equal(index, handle.index_snapshot):
        return False
    return True


def _indexes_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    equals = getattr(left, "equals", None)
    if equals is not None:
        try:
            return bool(equals(right))
        except Exception:
            return False
    return left == right


def _project_attributes(attributes: Any, columns: tuple[Any, ...]) -> Any | None:
    from vibespatial.api._native_result_core import NativeAttributeTable

    table = NativeAttributeTable.from_value(attributes)
    current_columns = tuple(table.columns)
    if any(column not in current_columns for column in columns):
        return None
    if columns == current_columns:
        return table
    if table.dataframe is not None:
        return NativeAttributeTable(dataframe=table.dataframe.loc[:, list(columns)])
    if table.arrow_table is not None:
        return NativeAttributeTable(
            arrow_table=table.to_arrow(index=False, columns=columns),
            index_override=table.index,
            column_override=columns,
            to_pandas_kwargs=table.to_pandas_kwargs,
        )
    if table.loader is not None:
        parent = table

        def _load_projected():
            return parent.to_pandas(copy=False).loc[:, list(columns)]

        return NativeAttributeTable.from_loader(
            _load_projected,
            index_override=table.index,
            columns=columns,
            to_pandas_kwargs=table.to_pandas_kwargs,
        )
    return None


_REGISTRY = NativeStateRegistry()


def attach_native_state(owner: Any, state: NativeFrameState) -> NativeStateHandle:
    return _REGISTRY.attach(owner, state)


def attach_native_state_from_native_tabular_result(
    owner: Any,
    result,
) -> NativeStateHandle | None:
    try:
        state = NativeFrameState.from_native_tabular_result(result)
    except Exception:
        return None
    return attach_native_state(owner, state)


def get_native_state(
    owner: Any,
    handle: NativeStateHandle | None = None,
) -> NativeFrameState | None:
    return _REGISTRY.get(owner, handle)


def drop_native_state(owner: Any) -> None:
    _REGISTRY.drop(owner)


__all__ = [
    "NativeFrameState",
    "NativeStateHandle",
    "NativeStateRegistry",
    "NativeStreamReadiness",
    "attach_native_state_from_native_tabular_result",
    "attach_native_state",
    "drop_native_state",
    "get_native_state",
]
