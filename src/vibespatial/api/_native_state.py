from __future__ import annotations

from collections.abc import Sequence
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
    geometry_metadata_cache: Any | None = None
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
        index_plan = getattr(result, "index_plan", None)
        if index_plan is None:
            index_plan = NativeIndexPlan.from_index(attributes.index)
        else:
            index_plan.validate_length(row_count)
        return cls(
            attributes=attributes,
            geometry=geometry,
            geometry_name=result.geometry_name,
            column_order=tuple(result.column_order),
            index_plan=index_plan,
            row_count=row_count,
            secondary_geometry=tuple(result.secondary_geometry),
            attrs=dict(result.attrs or {}),
            provenance=result.provenance,
            geometry_metadata_cache=getattr(result, "geometry_metadata", None),
            residency=combined_residency(geometry),
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
        if requested == self.column_order:
            return self
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

    def _rowset_with_geometry_proofs(self, rowset: NativeRowSet) -> NativeRowSet:
        """Attach inherited source-geometry proofs to a rowset when possible."""
        family_domain = rowset.geometry_family_domain
        trusted_all_valid_rows = rowset.trusted_all_valid_rows
        owned = (
            self.geometry.cached_owned()
            if hasattr(self.geometry, "cached_owned")
            else getattr(self.geometry, "owned", None)
        )
        source_state = getattr(owned, "device_state", None)
        if source_state is not None:
            if family_domain is None and source_state.trusted_family_domain is not None:
                family_domain = source_state.trusted_family_domain
            if family_domain is None and source_state.trusted_homogeneous_family is not None:
                family_domain = (source_state.trusted_homogeneous_family,)
            if (
                family_domain is None
                and source_state.trusted_polygonal_only is True
            ):
                from vibespatial.geometry.buffers import GeometryFamily

                family_domain = (
                    GeometryFamily.POLYGON,
                    GeometryFamily.MULTIPOLYGON,
                )
            if trusted_all_valid_rows is None and source_state.trusted_all_valid is True:
                trusted_all_valid_rows = True
        if (
            family_domain is rowset.geometry_family_domain
            and trusted_all_valid_rows is rowset.trusted_all_valid_rows
        ):
            return rowset
        return NativeRowSet.from_positions(
            rowset.positions,
            source_token=rowset.source_token,
            source_row_count=rowset.source_row_count,
            ordered=rowset.ordered,
            unique=rowset.unique,
            identity=rowset.identity,
            geometry_family_domain=family_domain,
            trusted_all_valid_rows=trusted_all_valid_rows,
        )

    @staticmethod
    def _apply_rowset_geometry_proofs(geometry: Any, rowset: NativeRowSet) -> None:
        """Preserve semantic rowset proofs on the taken geometry carrier."""
        if hasattr(geometry, "apply_rowset_proofs"):
            geometry.apply_rowset_proofs(rowset)
            return
        owned = (
            geometry.cached_owned()
            if hasattr(geometry, "cached_owned")
            else getattr(geometry, "owned", None)
        )
        if owned is None:
            return
        state = getattr(owned, "device_state", None)
        if state is None and getattr(owned, "residency", None) is Residency.DEVICE:
            state = owned._ensure_device_state(preserve_indexed_view=True)
        if state is None:
            return
        if rowset.trusted_all_valid_rows is True:
            state.trusted_all_valid = True
        family_domain = rowset.geometry_family_domain
        if not family_domain:
            return
        state.trusted_family_domain = tuple(family_domain)
        if rowset.trusted_all_valid_rows is True and len(family_domain) == 1:
            state.trusted_homogeneous_family = family_domain[0]
        from vibespatial.geometry.buffers import GeometryFamily

        polygonal_families = {
            GeometryFamily.POLYGON,
            GeometryFamily.MULTIPOLYGON,
        }
        if set(family_domain) <= polygonal_families:
            state.trusted_polygonal_only = True

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

    def with_geometry_crs(self, crs: Any) -> NativeFrameState:
        """Return the same frame state with metadata-only active geometry CRS."""
        return replace(self, geometry=self.geometry.with_crs(crs))

    def with_geometry_result(self, geometry: Any) -> NativeFrameState:
        """Return the same frame state with a replacement active geometry result."""
        if geometry.row_count != self.row_count:
            raise ValueError("native frame geometry row count must match row count")
        return replace(
            self,
            geometry=geometry,
            geometry_metadata_cache=None,
            residency=combined_residency(geometry),
        )

    def with_active_geometry(self, geometry_name: Any, *, crs: Any | None = None) -> NativeFrameState | None:
        """Return the same frame state with another existing geometry active."""
        if geometry_name not in self.column_order:
            return None
        if geometry_name == self.geometry_name:
            geometry = self.geometry if crs is None else self.geometry.with_crs(crs)
            return replace(self, geometry=geometry)

        secondary_by_name = {column.name: column for column in self.secondary_geometry}
        target = secondary_by_name.get(geometry_name)
        if target is None:
            return None

        from vibespatial.api._native_result_core import NativeGeometryColumn

        promoted_geometry = target.geometry if crs is None else target.geometry.with_crs(crs)
        secondary_by_name[self.geometry_name] = NativeGeometryColumn(
            self.geometry_name,
            self.geometry,
        )
        secondary_by_name.pop(geometry_name, None)
        secondary_geometry = tuple(
            secondary_by_name[column]
            for column in self.column_order
            if column in secondary_by_name
        )
        return replace(
            self,
            geometry=promoted_geometry,
            geometry_name=geometry_name,
            secondary_geometry=secondary_geometry,
            geometry_metadata_cache=None,
            residency=combined_residency(promoted_geometry),
        )

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

    def assign_expression_columns(
        self,
        expressions_by_name: dict[Any, Any],
        *,
        column_order: tuple[Any, ...] | None = None,
    ) -> NativeFrameState | None:
        """Attach private numeric expression vectors as native attribute columns."""
        from vibespatial.api._native_expression import NativeExpression

        if not expressions_by_name:
            return self
        for name, expression in expressions_by_name.items():
            if name == self.geometry_name:
                return None
            if not isinstance(expression, NativeExpression):
                return None
            if (
                expression.source_token is not None
                and expression.source_token != self.lineage_token
            ):
                raise ValueError("NativeExpression source token does not match NativeFrameState")
            if (
                expression.source_row_count is not None
                and int(expression.source_row_count) != self.row_count
            ):
                raise ValueError("NativeExpression row count does not match NativeFrameState")

        if column_order is None:
            requested = tuple(
                [
                    column
                    for column in self.column_order
                    if column not in expressions_by_name
                ]
                + list(expressions_by_name)
            )
        else:
            requested = tuple(column_order)
        if self.geometry_name not in requested or len(set(requested)) != len(requested):
            return None
        if any(name not in requested for name in expressions_by_name):
            return None

        attribute_order = tuple(
            column for column in requested if column != self.geometry_name
        )
        from vibespatial.api._native_result_core import NativeAttributeTable

        updated_attributes = NativeAttributeTable.from_value(
            self.attributes,
        ).assign_columns(
            expressions_by_name,
            columns=attribute_order,
        )
        if updated_attributes is None:
            return None
        return replace(
            self,
            attributes=updated_attributes,
            column_order=requested,
        )

    def attribute_expression(self, column: Any, *, operation: str | None = None):
        """Return a private scalar expression backed by an admitted attribute column."""
        from vibespatial.api._native_expression import NativeExpression
        from vibespatial.api._native_result_core import NativeAttributeTable

        columns = NativeAttributeTable.from_value(self.attributes).numeric_column_arrays(
            (column,),
        )
        if columns is None or column not in columns:
            return None
        values = columns[column]
        return NativeExpression(
            operation=operation or f"attribute.{column}",
            values=values,
            source_token=self.lineage_token,
            source_row_count=self.row_count,
            dtype=str(getattr(values, "dtype", "")) or None,
            precision="source",
        )

    def geometry_family_rowset(self, family: Any) -> NativeRowSet | None:
        """Return rows whose current geometry or exploded part matches a family."""
        from vibespatial.geometry.buffers import GeometryFamily
        from vibespatial.geometry.owned import FAMILY_TAGS

        target_family = family if isinstance(family, GeometryFamily) else GeometryFamily(family)
        tag = FAMILY_TAGS[target_family]
        tags = (
            getattr(self.provenance, "part_family_tags", None)
            if self.provenance is not None
            else None
        )
        if tags is None:
            owned = getattr(self.geometry, "owned", None)
            if owned is None:
                return None
            tags = (
                owned._ensure_device_state(preserve_indexed_view=True).tags
                if getattr(owned, "residency", None) is Residency.DEVICE
                else owned.tags
            )

        if hasattr(tags, "__cuda_array_interface__"):
            import cupy as cp

            positions = cp.nonzero(cp.asarray(tags) == cp.int8(tag))[0].astype(
                cp.int64,
                copy=False,
            )
        else:
            import numpy as np

            positions = np.flatnonzero(np.asarray(tags) == np.int8(tag)).astype(
                np.int64,
                copy=False,
            )
        return NativeRowSet.from_positions(
            positions,
            source_token=self.lineage_token,
            source_row_count=self.row_count,
            ordered=True,
            unique=True,
            geometry_family_domain=(target_family,),
            trusted_all_valid_rows=True,
        )

    def take(
        self,
        rowset: NativeRowSet,
        *,
        preserve_index: bool = True,
        index_positions: Any | None = None,
    ) -> NativeFrameState:
        if rowset.source_row_count is not None and rowset.source_row_count != self.row_count:
            raise ValueError(
                "NativeRowSet source row count does not match NativeFrameState"
            )
        if rowset.source_token is not None and rowset.source_token != self.lineage_token:
            raise ValueError("NativeRowSet source token does not match NativeFrameState")

        rowset = self._rowset_with_geometry_proofs(rowset)

        if index_positions is not None and len(index_positions) != len(rowset):
            raise ValueError("index positions must align with NativeRowSet")

        if preserve_index and rowset.identity and len(rowset) == self.row_count:
            self._apply_rowset_geometry_proofs(self.geometry, rowset)
            return self

        if rowset.is_device and preserve_index and self.index_plan.kind in {
            "range",
            "device-labels",
            "host-labels",
            "host-labels-take",
        }:
            normalized = rowset.positions
            attributes = self.attributes.take(normalized, preserve_index=False)
            geometry = self.geometry.take(normalized, unique=rowset.unique)
            self._apply_rowset_geometry_proofs(geometry, rowset)
            secondary_geometry = tuple(
                type(column)(
                    column.name,
                    column.geometry.take(normalized, unique=rowset.unique),
                )
                for column in self.secondary_geometry
            )
            geometry_metadata_cache = (
                None
                if self.geometry_metadata_cache is None
                else self.geometry_metadata_cache.take(normalized)
            )
            provenance = (
                self.provenance.take(normalized)
                if self.provenance is not None and hasattr(self.provenance, "take")
                else self.provenance
            )
            index_plan = self.index_plan.take(
                normalized if index_positions is None else index_positions,
                preserve_index=True,
                unique=rowset.unique,
            )
            if self.index_plan.admits_unique_label_selection:
                index_plan = index_plan.with_selection(
                    normalized,
                    source_token=self.lineage_token,
                    source_row_count=self.row_count,
                    unique=rowset.unique,
                )
            return type(self)(
                attributes=attributes,
                geometry=geometry,
                geometry_name=self.geometry_name,
                column_order=self.column_order,
                index_plan=index_plan,
                row_count=len(rowset),
                secondary_geometry=secondary_geometry,
                attrs=self.attrs,
                provenance=provenance,
                geometry_metadata_cache=geometry_metadata_cache,
                residency=combined_residency(geometry),
                readiness=self.readiness,
            )

        taken = self.to_native_tabular_result().take(
            rowset,
            preserve_index=preserve_index,
        )
        result = type(self).from_native_tabular_result(taken)
        self._apply_rowset_geometry_proofs(result.geometry, rowset)
        return result

    def _geometry_measurement_expression(self, evaluator, *, operation: str):
        """Reduce concrete native geometry parts into one device value per row."""
        owned = getattr(self.geometry, "owned", None)
        if owned is not None:
            return evaluator(owned, source_token=self.lineage_token)

        composition = getattr(self.geometry, "composition", None)
        if composition is None:
            raise TypeError(
                f"NativeFrameState {operation} expression requires native geometry"
            )

        import cupy as cp

        from vibespatial.api._native_expression import NativeExpression

        values = cp.zeros(self.row_count, dtype=cp.float64)
        present = cp.zeros(self.row_count, dtype=cp.bool_)
        for part in composition.parts:
            part_owned = getattr(part.geometry, "owned", None)
            if part_owned is None:
                raise TypeError(
                    f"NativeFrameState {operation} composition parts require owned geometry"
                )
            part_expression = evaluator(part_owned, source_token=self.lineage_token)
            part_values = cp.asarray(part_expression.values, dtype=cp.float64)
            part_rows = cp.asarray(part.output_rows, dtype=cp.int64)
            part_state = part_owned._ensure_device_state(preserve_indexed_view=True)
            part_validity = cp.asarray(part_state.validity, dtype=cp.bool_)
            if int(part_rows.size) != int(part_values.size):
                raise ValueError(
                    f"native geometry {operation} part rows do not match values"
                )
            cp.add.at(values, part_rows[part_validity], part_values[part_validity])
            present[part_rows[part_validity]] = True
        values[~present] = cp.nan
        return NativeExpression(
            operation=f"geometry.{operation}",
            values=values,
            source_token=self.lineage_token,
            source_row_count=self.row_count,
            dtype="float64",
            precision="fp64",
        )

    def geometry_area_expression(self):
        """Return a private device area vector for sanctioned native consumers."""
        from vibespatial.constructive.measurement import area_expression_owned

        return self._geometry_measurement_expression(
            area_expression_owned,
            operation="area",
        )

    def geometry_length_expression(self):
        """Return a private device length vector for sanctioned native consumers."""
        from vibespatial.constructive.measurement import length_expression_owned

        return self._geometry_measurement_expression(
            length_expression_owned,
            operation="length",
        )

    def geometry_validity_expression(self):
        """Return a private device validity vector for sanctioned consumers."""
        owned = getattr(self.geometry, "owned", None)
        if owned is None:
            raise TypeError("NativeFrameState geometry validity expression requires owned geometry")
        from vibespatial.constructive.validity import validity_expression_owned

        return validity_expression_owned(
            owned,
            source_token=self.lineage_token,
            exact_collinearity=True,
        )

    def geometry_predicate_expression(
        self,
        predicate: str,
        other: NativeFrameState,
        *,
        dispatch_mode: Any = "gpu",
        precision: Any = "auto",
    ):
        """Return a private row-aligned predicate vector against another frame."""
        if not isinstance(other, NativeFrameState):
            raise TypeError(
                "NativeFrameState geometry predicate expression requires another NativeFrameState"
            )
        if other.row_count != self.row_count:
            raise ValueError(
                "NativeFrameState geometry predicate expression requires row-aligned frames"
            )
        owned = getattr(self.geometry, "owned", None)
        other_owned = getattr(other.geometry, "owned", None)
        if owned is None or other_owned is None:
            raise TypeError(
                "NativeFrameState geometry predicate expression requires owned geometry"
            )
        from vibespatial.predicates.binary import binary_predicate_expression

        return binary_predicate_expression(
            predicate,
            owned,
            other_owned,
            dispatch_mode=dispatch_mode,
            precision=precision,
            source_token=self.lineage_token,
            operation=f"geometry_predicate.{predicate}",
        )

    def geometry_predicate_expressions(
        self,
        predicates: Sequence[str],
        other: NativeFrameState,
        *,
        dispatch_mode: Any = "gpu",
        precision: Any = "auto",
    ):
        """Return multiple private row-aligned predicate vectors against another frame."""
        if not isinstance(other, NativeFrameState):
            raise TypeError(
                "NativeFrameState geometry predicate expressions require another NativeFrameState"
            )
        if other.row_count != self.row_count:
            raise ValueError(
                "NativeFrameState geometry predicate expressions require row-aligned frames"
            )
        owned = getattr(self.geometry, "owned", None)
        other_owned = getattr(other.geometry, "owned", None)
        if owned is None or other_owned is None:
            raise TypeError(
                "NativeFrameState geometry predicate expressions require owned geometry"
            )
        from vibespatial.predicates.binary import binary_predicate_expressions

        return binary_predicate_expressions(
            predicates,
            owned,
            other_owned,
            dispatch_mode=dispatch_mode,
            precision=precision,
            source_token=self.lineage_token,
            operation_prefix="geometry_predicate",
        )

    def geometry_distance_expression(self, other):
        """Return private row-aligned distances to another native frame."""
        from shapely.geometry.base import BaseGeometry

        scalar_geometry = isinstance(other, BaseGeometry)
        if not isinstance(other, NativeFrameState) and not scalar_geometry:
            raise TypeError(
                "NativeFrameState geometry distance expressions require another "
                "NativeFrameState or one scalar geometry"
            )
        if not scalar_geometry and other.row_count != self.row_count:
            raise ValueError(
                "NativeFrameState geometry distance expressions require row-aligned frames"
            )
        owned = getattr(self.geometry, "owned", None)
        if scalar_geometry:
            from vibespatial.geometry.owned import from_shapely_geometries

            other_owned = from_shapely_geometries([other])
        else:
            other_owned = getattr(other.geometry, "owned", None)
        from vibespatial.spatial.distance_owned import distance_expression_owned

        if owned is not None and other_owned is not None:
            return distance_expression_owned(
                owned,
                other_owned,
                source_token=self.lineage_token,
            )

        left_composition = getattr(self.geometry, "composition", None)
        right_composition = (
            None if scalar_geometry else getattr(other.geometry, "composition", None)
        )
        left_parts = (
            left_composition.ordered_contiguous_device_parts()
            if left_composition is not None
            else None
        )
        right_parts = (
            right_composition.ordered_contiguous_device_parts()
            if right_composition is not None
            else None
        )
        if owned is not None:
            left_parts = ((0, self.row_count, owned),)
        if scalar_geometry and left_parts is not None:
            right_parts = tuple(
                (start, stop, other_owned) for start, stop, _part in left_parts
            )
        elif other_owned is not None:
            right_parts = ((0, other.row_count, other_owned),)
        if left_parts is None or right_parts is None:
            raise TypeError(
                "NativeFrameState geometry distance expression requires owned geometry "
                "or certified contiguous device compositions"
            )

        import cupy as cp

        from vibespatial.api._native_expression import NativeExpression
        from vibespatial.cuda._runtime import get_cuda_runtime

        max_span = max(
            (stop - start for start, stop, _part in (*left_parts, *right_parts)),
            default=0,
        )
        # One persistent fp64 output plus one partition result.  When partition
        # boundaries differ, two int64 row-indirection vectors are also live.
        required_bytes = self.row_count * 8 + max_span * 24
        admission = get_cuda_runtime().admit_device_memory(
            stage="geometry-distance-partitioned",
            required_bytes=required_bytes,
            requested_units=self.row_count,
        )
        if not admission.admitted:
            raise MemoryError(
                "partitioned geometry distance requires "
                f"{required_bytes} device bytes with {admission.remaining_bytes} available"
            )

        values = cp.empty(self.row_count, dtype=cp.float64)
        left_index = right_index = 0
        cursor = 0
        selected_precisions: set[str] = set()
        while left_index < len(left_parts) and right_index < len(right_parts):
            left_start, left_stop, left_part = left_parts[left_index]
            right_start, right_stop, right_part = right_parts[right_index]
            start = max(left_start, right_start)
            stop = min(left_stop, right_stop)
            if stop <= start:
                if left_stop <= right_start:
                    left_index += 1
                    continue
                if right_stop <= left_start:
                    right_index += 1
                    continue
                raise ValueError("native geometry partitions do not overlap monotonically")

            if start == left_start and stop == left_stop:
                left_piece = left_part
            else:
                left_piece = left_part._device_indexed_take(
                    cp.arange(start - left_start, stop - left_start, dtype=cp.int64),
                    assume_unique_indices=True,
                    defer_device_metadata=True,
                )
            if start == right_start and stop == right_stop:
                right_piece = right_part
            else:
                right_piece = right_part._device_indexed_take(
                    cp.arange(start - right_start, stop - right_start, dtype=cp.int64),
                    assume_unique_indices=True,
                    defer_device_metadata=True,
                )

            expression = distance_expression_owned(
                left_piece,
                right_piece,
                source_token=self.lineage_token,
            )
            if expression is None:
                return None
            values[start:stop] = expression.values
            if expression.precision is not None:
                selected_precisions.add(expression.precision)
            cursor = stop
            if left_stop == stop:
                left_index += 1
            if right_stop == stop:
                right_index += 1

        if cursor != self.row_count:
            raise ValueError("native geometry partitions do not cover every aligned row")
        precision = (
            next(iter(selected_precisions))
            if len(selected_precisions) == 1
            else "partitioned"
        )
        return NativeExpression(
            operation="geometry.distance",
            values=values,
            source_token=self.lineage_token,
            source_row_count=self.row_count,
            dtype=str(values.dtype),
            precision=precision,
            null_policy="nan-false",
        )

    def geometry_centroid_x_expression(self):
        """Return a private device centroid-x vector for sanctioned consumers."""
        owned = getattr(self.geometry, "owned", None)
        if owned is None:
            raise TypeError("NativeFrameState geometry centroid expression requires owned geometry")
        from vibespatial.constructive.centroid import centroid_expression_owned

        return centroid_expression_owned(
            owned,
            component="x",
            source_token=self.lineage_token,
        )

    def geometry_centroid_y_expression(self):
        """Return a private device centroid-y vector for sanctioned consumers."""
        owned = getattr(self.geometry, "owned", None)
        if owned is None:
            raise TypeError("NativeFrameState geometry centroid expression requires owned geometry")
        from vibespatial.constructive.centroid import centroid_expression_owned

        return centroid_expression_owned(
            owned,
            component="y",
            source_token=self.lineage_token,
        )

    def geometry_centroid_expressions(self):
        """Return private centroid-x/y expressions without public point export."""
        owned = getattr(self.geometry, "owned", None)
        if owned is None:
            raise TypeError("NativeFrameState geometry centroid expression requires owned geometry")
        from vibespatial.constructive.centroid import centroid_expressions_owned

        return centroid_expressions_owned(
            owned,
            source_token=self.lineage_token,
        )

    def geometry_metadata(self):
        """Return reusable private geometry metadata for sanctioned native consumers."""
        if self.geometry_metadata_cache is not None:
            self.geometry_metadata_cache.validate_row_count(self.row_count)
            return self.geometry_metadata_cache.with_source_token(self.lineage_token)

        owned = getattr(self.geometry, "owned", None)
        if owned is None:
            raise TypeError("NativeFrameState geometry metadata requires owned geometry")
        from vibespatial.api._native_metadata import NativeGeometryMetadata

        return NativeGeometryMetadata.from_owned(
            owned,
            source_token=self.lineage_token,
        )

    def to_native_tabular_result(self):
        from vibespatial.api._native_result_core import NativeTabularResult

        attributes = self.attributes
        if self.index_plan.device_labels is None:
            public_index = self.index_plan.to_public_index(
                surface="vibespatial.api.NativeFrameState.to_native_tabular_result",
                strict_disallowed=False,
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
            geometry_metadata=self.geometry_metadata_cache,
            index_plan=self.index_plan,
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
            if not bool(equals(right)):
                return False
        except Exception:
            return False
        return tuple(getattr(left, "names", (getattr(left, "name", None),))) == tuple(
            getattr(right, "names", (getattr(right, "name", None),)),
        )
    return left == right


def _project_attributes(attributes: Any, columns: tuple[Any, ...]) -> Any | None:
    from vibespatial.api._native_result_core import NativeAttributeTable

    table = NativeAttributeTable.from_value(attributes)
    return table.project_columns(columns)


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
