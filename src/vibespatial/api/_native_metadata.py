from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from vibespatial.api._native_state import NativeStreamReadiness
from vibespatial.runtime.materialization import (
    MaterializationBoundary,
    record_materialization_event,
)
from vibespatial.runtime.residency import Residency


def _is_device_array(values: Any) -> bool:
    return hasattr(values, "__cuda_array_interface__")


def _array_size(values: Any) -> int:
    shape = getattr(values, "shape", None)
    if shape is not None:
        return int(shape[0])
    return len(values)


def _device_or_host_residency(*values: Any) -> Residency:
    return Residency.DEVICE if any(_is_device_array(value) for value in values) else Residency.HOST


def _host_or_device_slots(values: Any) -> tuple[Any | None, Any | None]:
    if values is None:
        return None, None
    return (None, values) if _is_device_array(values) else (values, None)


def _family_names(owned) -> tuple[str, ...]:
    return tuple(family.value for family in getattr(owned, "families", {}))


def _host_bounds_total(bounds: np.ndarray) -> tuple[float, float, float, float] | None:
    if bounds.size == 0 or np.all(np.isnan(bounds[:, 0])):
        return None
    return (
        float(np.nanmin(bounds[:, 0])),
        float(np.nanmin(bounds[:, 1])),
        float(np.nanmax(bounds[:, 2])),
        float(np.nanmax(bounds[:, 3])),
    )


def _normalize_row_positions(row_positions: Any):
    values = getattr(row_positions, "positions", row_positions)
    if _is_device_array(values):
        import cupy as cp

        positions = cp.asarray(values)
        if positions.dtype == cp.bool_ or positions.dtype == bool:
            return cp.flatnonzero(positions).astype(cp.int64, copy=False)
        return positions.astype(cp.int64, copy=False)

    positions = np.asarray(values)
    if positions.dtype == bool:
        positions = np.flatnonzero(positions)
    return np.asarray(positions, dtype=np.int64)


def _gather_optional(values: Any | None, row_positions: Any) -> Any | None:
    if values is None:
        return None
    if _is_device_array(values) or _is_device_array(row_positions):
        import cupy as cp

        return cp.asarray(values)[cp.asarray(row_positions, dtype=cp.int64)]
    return np.asarray(values)[np.asarray(row_positions, dtype=np.int64)]


@dataclass(frozen=True)
class NativeGeometryMetadataPart:
    """Concrete metadata rows mapped into a logical geometry composition."""

    metadata: Any
    output_rows: Any
    geometry: Any | None = None
    collection_position: int | None = None

    def __post_init__(self) -> None:
        if _array_size(self.output_rows) != int(self.metadata.row_count):
            raise ValueError("composition metadata rows must match concrete metadata")

    @property
    def is_device(self) -> bool:
        return bool(self.metadata.is_device or _is_device_array(self.output_rows))


@dataclass(frozen=True)
class NativeGeometryMetadata:
    """Private reusable geometry metadata carrier."""

    row_count: int
    bounds: Any | None = None
    source_token: str | None = None
    total_bounds: tuple[float, float, float, float] | None = None
    validity: Any | None = None
    family_tags: Any | None = None
    family_row_offsets: Any | None = None
    family_names: tuple[str, ...] = ()
    coordinate_stats: Any | None = None
    dimensional_flags: Any | None = None
    shape_summary: dict[str, Any] = field(default_factory=dict)
    composition_parts: tuple[NativeGeometryMetadataPart, ...] = ()
    composition_source_row_count: int | None = None
    composition_row_positions: Any | None = None
    residency: Residency = Residency.HOST
    readiness: NativeStreamReadiness = field(default_factory=NativeStreamReadiness)

    def __post_init__(self) -> None:
        if self.row_count < 0:
            raise ValueError("NativeGeometryMetadata row_count must be non-negative")
        if self.bounds is not None and _array_size(self.bounds) != int(self.row_count):
            raise ValueError("NativeGeometryMetadata bounds length must match row_count")
        for part in self.composition_parts:
            if _array_size(part.output_rows) != int(part.metadata.row_count):
                raise ValueError("composition metadata part row count mismatch")
        if self.composition_parts and self.composition_source_row_count is None:
            raise ValueError("composition metadata requires its source row count")
        if (
            self.composition_row_positions is not None
            and _array_size(self.composition_row_positions) != int(self.row_count)
        ):
            raise ValueError("composition metadata row indirection must match row count")

    @classmethod
    def from_owned(
        cls,
        owned,
        *,
        source_token: str | None = None,
        prefer_device: bool = True,
        materialize_host: bool = False,
    ) -> NativeGeometryMetadata:
        """Build metadata from owned geometry without forcing public objects."""
        bounds = None
        total_bounds = None
        validity = None
        family_tags = None
        family_row_offsets = None

        if prefer_device and getattr(owned, "residency", None) is Residency.DEVICE:
            from vibespatial.kernels.core.geometry_analysis import (
                compute_geometry_bounds_device,
            )

            state = owned._ensure_device_state()
            bounds = state.row_bounds
            if bounds is None:
                bounds = compute_geometry_bounds_device(owned)
            validity = state.validity
            family_tags = state.tags
            family_row_offsets = state.family_row_offsets

        if bounds is None:
            from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds

            host_bounds = compute_geometry_bounds(owned, dispatch_mode="cpu")
            bounds = host_bounds
            total_bounds = _host_bounds_total(host_bounds)
            validity = getattr(owned, "_validity", None)
            family_tags = getattr(owned, "_tags", None)
            family_row_offsets = getattr(owned, "_family_row_offsets", None)
        elif materialize_host:
            host_bounds = cls(
                row_count=int(owned.row_count),
                bounds=bounds,
                source_token=source_token,
            ).bounds_to_host(
                surface="vibespatial.api.NativeGeometryMetadata.from_owned",
                strict_disallowed=True,
            )
            total_bounds = _host_bounds_total(host_bounds)

        residency = _device_or_host_residency(
            bounds,
            validity,
            family_tags,
            family_row_offsets,
        )
        return cls(
            row_count=int(owned.row_count),
            bounds=bounds,
            source_token=source_token,
            total_bounds=total_bounds,
            validity=validity,
            family_tags=family_tags,
            family_row_offsets=family_row_offsets,
            family_names=_family_names(owned),
            residency=residency,
        )

    @classmethod
    def from_cached_owned(
        cls,
        owned,
        *,
        source_token: str | None = None,
        total_bounds: tuple[float, float, float, float] | None = None,
    ) -> NativeGeometryMetadata:
        """Wrap already-cached owned metadata without computing new bounds."""
        bounds = None
        validity = None
        family_tags = None
        family_row_offsets = None

        state = getattr(owned, "device_state", None)
        if state is not None:
            bounds = state.row_bounds
            validity = state.validity
            family_tags = state.tags
            family_row_offsets = state.family_row_offsets
        else:
            from vibespatial.geometry.geometry_analysis_host import assemble_cached_bounds

            bounds = assemble_cached_bounds(owned)
            validity = getattr(owned, "_validity", None)
            family_tags = getattr(owned, "_tags", None)
            family_row_offsets = getattr(owned, "_family_row_offsets", None)
            if total_bounds is None and bounds is not None:
                total_bounds = _host_bounds_total(np.asarray(bounds, dtype=np.float64))

        return cls(
            row_count=int(owned.row_count),
            bounds=bounds,
            source_token=source_token,
            total_bounds=total_bounds,
            validity=validity,
            family_tags=family_tags,
            family_row_offsets=family_row_offsets,
            family_names=_family_names(owned),
            residency=_device_or_host_residency(
                bounds,
                validity,
                family_tags,
                family_row_offsets,
            ),
        )

    @classmethod
    def from_native_geometry(
        cls,
        geometry,
        *,
        source_token: str | None = None,
    ) -> NativeGeometryMetadata:
        """Build logical-row metadata for owned or composed native geometry."""
        owned = getattr(geometry, "owned", None)
        if owned is not None:
            return cls.from_cached_owned(owned, source_token=source_token)

        composition = getattr(geometry, "composition", None)
        if composition is None:
            raise TypeError("native geometry metadata requires owned or composed geometry")

        row_count = int(composition.row_count)
        family_names: set[str] = set()
        metadata_parts = []
        for part in composition.parts:
            part_owned = getattr(part.geometry, "owned", None)
            if part_owned is None:
                raise TypeError("native geometry metadata composition parts must be owned")
            part_metadata = cls.from_cached_owned(part_owned)
            family_names.update(part_metadata.family_names)
            metadata_parts.append(
                NativeGeometryMetadataPart(
                    metadata=part_metadata,
                    output_rows=part.output_rows,
                    geometry=part.geometry,
                    collection_position=part.collection_position,
                )
            )
        return cls(
            row_count=row_count,
            source_token=source_token,
            family_names=tuple(sorted(family_names)),
            shape_summary={
                "physical_shape": "sparse-native-composition",
                "part_count": len(composition.parts),
            },
            composition_parts=tuple(metadata_parts),
            composition_source_row_count=row_count,
            residency=(
                Residency.DEVICE
                if any(part.is_device for part in metadata_parts)
                else Residency.HOST
            ),
        )

    @classmethod
    def from_spatial_index(
        cls,
        flat_index,
        *,
        source_token: str | None = None,
    ) -> NativeGeometryMetadata:
        """Wrap flat-index metadata without touching lazy host properties."""
        bounds = (
            flat_index.device_bounds
            if getattr(flat_index, "device_bounds", None) is not None
            else getattr(flat_index, "_host_bounds", None)
        )
        owned = flat_index.geometry_array
        state = getattr(owned, "device_state", None)
        validity = getattr(state, "validity", None)
        family_tags = getattr(state, "tags", None)
        family_row_offsets = getattr(state, "family_row_offsets", None)
        if validity is None:
            validity = getattr(owned, "_validity", None)
        if family_tags is None:
            family_tags = getattr(owned, "_tags", None)
        if family_row_offsets is None:
            family_row_offsets = getattr(owned, "_family_row_offsets", None)
        return cls(
            row_count=int(owned.row_count),
            bounds=bounds,
            source_token=source_token,
            total_bounds=flat_index.total_bounds,
            validity=validity,
            family_tags=family_tags,
            family_row_offsets=family_row_offsets,
            family_names=_family_names(owned),
            residency=_device_or_host_residency(
                bounds,
                validity,
                family_tags,
                family_row_offsets,
            ),
        )

    @property
    def is_device(self) -> bool:
        return (
            self.residency is Residency.DEVICE
            or _is_device_array(self.bounds)
            or _is_device_array(self.composition_row_positions)
            or any(part.is_device for part in self.composition_parts)
        )

    def validate_row_count(self, row_count: int) -> None:
        if int(row_count) != int(self.row_count):
            raise ValueError(
                f"NativeGeometryMetadata row count mismatch: expected "
                f"{self.row_count}, got {row_count}"
            )

    def with_source_token(self, source_token: str | None) -> NativeGeometryMetadata:
        """Return the same metadata carrier with updated lineage."""
        if self.source_token == source_token:
            return self
        return replace(self, source_token=source_token)

    def take(
        self,
        row_positions: Any,
        *,
        source_token: str | None = None,
    ) -> NativeGeometryMetadata:
        """Gather row-aligned metadata without forcing device buffers to host."""
        positions = _normalize_row_positions(row_positions)
        if self.composition_parts:
            logical_positions = (
                positions
                if self.composition_row_positions is None
                else _gather_optional(self.composition_row_positions, positions)
            )
            return type(self)(
                row_count=_array_size(positions),
                source_token=self.source_token if source_token is None else source_token,
                family_names=self.family_names,
                coordinate_stats=self.coordinate_stats,
                dimensional_flags=self.dimensional_flags,
                shape_summary=dict(self.shape_summary),
                composition_parts=self.composition_parts,
                composition_source_row_count=self.composition_source_row_count,
                composition_row_positions=logical_positions,
                residency=self.residency,
                readiness=self.readiness,
            )
        bounds = _gather_optional(self.bounds, positions)
        validity = _gather_optional(self.validity, positions)
        family_tags = _gather_optional(self.family_tags, positions)
        family_row_offsets = _gather_optional(self.family_row_offsets, positions)
        total_bounds = (
            _host_bounds_total(np.asarray(bounds, dtype=np.float64))
            if bounds is not None and not _is_device_array(bounds)
            else None
        )
        return type(self)(
            row_count=_array_size(positions),
            bounds=bounds,
            source_token=self.source_token if source_token is None else source_token,
            total_bounds=total_bounds,
            validity=validity,
            family_tags=family_tags,
            family_row_offsets=family_row_offsets,
            family_names=self.family_names,
            coordinate_stats=self.coordinate_stats,
            dimensional_flags=self.dimensional_flags,
            shape_summary=dict(self.shape_summary),
            composition_parts=(),
            composition_source_row_count=None,
            composition_row_positions=None,
            residency=_device_or_host_residency(
                bounds,
                validity,
                family_tags,
                family_row_offsets,
                self.coordinate_stats,
                self.dimensional_flags,
            ),
            readiness=self.readiness,
        )

    @classmethod
    def concat(
        cls,
        metadatas: list[NativeGeometryMetadata],
        *,
        source_token: str | None = None,
    ) -> NativeGeometryMetadata | None:
        if not metadatas:
            return None

        def concat_optional(name: str):
            values = [getattr(metadata, name) for metadata in metadatas]
            if any(value is None for value in values):
                return None
            if any(_is_device_array(value) for value in values):
                import cupy as cp

                return cp.concatenate([cp.asarray(value) for value in values])
            return np.concatenate([np.asarray(value) for value in values])

        bounds = concat_optional("bounds")
        validity = concat_optional("validity")
        family_tags = concat_optional("family_tags")
        family_row_offsets = concat_optional("family_row_offsets")
        total_bounds = None
        if all(metadata.total_bounds is not None for metadata in metadatas):
            total_bounds = (
                min(metadata.total_bounds[0] for metadata in metadatas),
                min(metadata.total_bounds[1] for metadata in metadatas),
                max(metadata.total_bounds[2] for metadata in metadatas),
                max(metadata.total_bounds[3] for metadata in metadatas),
            )
        elif bounds is not None and not _is_device_array(bounds):
            total_bounds = _host_bounds_total(np.asarray(bounds, dtype=np.float64))

        family_names = tuple(
            dict.fromkeys(
                family_name
                for metadata in metadatas
                for family_name in metadata.family_names
            )
        )
        return cls(
            row_count=sum(int(metadata.row_count) for metadata in metadatas),
            bounds=bounds,
            source_token=source_token,
            total_bounds=total_bounds,
            validity=validity,
            family_tags=family_tags,
            family_row_offsets=family_row_offsets,
            family_names=family_names,
            residency=_device_or_host_residency(
                bounds,
                validity,
                family_tags,
                family_row_offsets,
            ),
        )

    def bounds_to_host(
        self,
        *,
        surface: str = "vibespatial.api.NativeGeometryMetadata.bounds_to_host",
        strict_disallowed: bool = True,
    ) -> np.ndarray:
        if self.bounds is None and self.composition_parts:
            return self.physicalize_logical_rows().bounds_to_host(
                surface=surface,
                strict_disallowed=strict_disallowed,
            )
        if self.bounds is None:
            return np.empty((int(self.row_count), 4), dtype=np.float64)
        if _is_device_array(self.bounds):
            item_count = int(np.prod(getattr(self.bounds, "shape", (self.row_count, 4))))
            itemsize = int(getattr(getattr(self.bounds, "dtype", None), "itemsize", 8))
            record_materialization_event(
                surface=surface,
                boundary=MaterializationBoundary.INTERNAL_HOST_CONVERSION,
                operation="native_geometry_metadata_bounds_to_host",
                reason="device geometry metadata bounds were materialized on host",
                detail=f"rows={self.row_count}, bytes={item_count * itemsize}",
                d2h_transfer=True,
                strict_disallowed=strict_disallowed,
            )
            from vibespatial.cuda._runtime import get_cuda_runtime

            return get_cuda_runtime().copy_device_to_host(
                self.bounds,
                reason=f"{surface}::native_geometry_metadata_bounds_to_host",
            ).astype(np.float64, copy=False)
        return np.asarray(self.bounds, dtype=np.float64)

    def physicalize_logical_rows(self) -> NativeGeometryMetadata:
        """Reduce sparse composition metadata into dense logical-row columns."""
        if not self.composition_parts:
            return self

        import cupy as cp

        bounds = cp.empty((self.row_count, 4), dtype=cp.float64)
        bounds[:, 0:2] = cp.inf
        bounds[:, 2:4] = -cp.inf
        validity = cp.zeros(self.row_count, dtype=cp.bool_)
        bounded = cp.zeros(self.row_count, dtype=cp.bool_)
        selected_source_rows = (
            cp.arange(self.row_count, dtype=cp.int64)
            if self.composition_row_positions is None
            else cp.asarray(self.composition_row_positions, dtype=cp.int64)
        )
        from vibespatial.api._native_result_core import _composition_part_take_relation

        for part in self.composition_parts:
            relation = _composition_part_take_relation(
                part.output_rows,
                selected_source_rows,
            )
            concrete_positions = relation.left_indices
            if _array_size(concrete_positions) == 0:
                continue
            part_metadata = part.metadata.take(concrete_positions)
            if part_metadata.composition_parts:
                part_metadata = part_metadata.physicalize_logical_rows()
            if part_metadata.bounds is None:
                part_owned = getattr(part.geometry, "owned", None)
                if part_owned is None:
                    raise TypeError(
                        "composition metadata bounds require concrete owned geometry"
                    )
                selected_geometry = part.geometry.take(
                    concrete_positions,
                    defer_device_metadata=True,
                )
                selected_owned = getattr(selected_geometry, "owned", None)
                if selected_owned is None:
                    raise TypeError(
                        "composition metadata bounds require selected owned geometry"
                    )
                part_metadata = type(self).from_owned(selected_owned)
            part_bounds = cp.asarray(part_metadata.bounds, dtype=cp.float64)
            part_validity = (
                cp.ones(part_metadata.row_count, dtype=cp.bool_)
                if part_metadata.validity is None
                else cp.asarray(part_metadata.validity, dtype=cp.bool_)
            )
            part_rows = cp.asarray(relation.right_indices, dtype=cp.int64)
            validity[part_rows[part_validity]] = True
            finite = part_validity & cp.all(cp.isfinite(part_bounds), axis=1)
            finite_rows = part_rows[finite]
            cp.minimum.at(bounds[:, 0], finite_rows, part_bounds[finite, 0])
            cp.minimum.at(bounds[:, 1], finite_rows, part_bounds[finite, 1])
            cp.maximum.at(bounds[:, 2], finite_rows, part_bounds[finite, 2])
            cp.maximum.at(bounds[:, 3], finite_rows, part_bounds[finite, 3])
            bounded[finite_rows] = True
        bounds[~bounded] = cp.nan
        return replace(
            self,
            bounds=bounds,
            validity=validity,
            composition_parts=(),
            composition_source_row_count=None,
            composition_row_positions=None,
            shape_summary={
                **self.shape_summary,
                "physical_shape": "dense-logical-rows",
            },
            residency=Residency.DEVICE,
        )


@dataclass(frozen=True)
class NativeSpatialIndex:
    """Private reusable spatial-index execution carrier."""

    kind: str
    row_count: int
    geometry: Any
    order: Any | None = None
    morton_keys: Any | None = None
    metadata: NativeGeometryMetadata | None = None
    regular_grid: Any | None = None
    total_bounds: tuple[float, float, float, float] | None = None
    source_token: str | None = None
    index_parameters: dict[str, Any] = field(default_factory=dict)
    residency: Residency = Residency.HOST
    readiness: NativeStreamReadiness = field(default_factory=NativeStreamReadiness)

    def __post_init__(self) -> None:
        if self.row_count < 0:
            raise ValueError("NativeSpatialIndex row_count must be non-negative")
        if self.order is not None and _array_size(self.order) != int(self.row_count):
            raise ValueError("NativeSpatialIndex order length must match row_count")
        if self.morton_keys is not None and _array_size(self.morton_keys) != int(
            self.row_count
        ):
            raise ValueError("NativeSpatialIndex morton_keys length must match row_count")
        if self.metadata is not None and self.metadata.row_count != int(self.row_count):
            raise ValueError("NativeSpatialIndex metadata row_count mismatch")

    @classmethod
    def from_flat_index(
        cls,
        flat_index,
        *,
        source_token: str | None = None,
    ) -> NativeSpatialIndex:
        """Wrap the current flat index as a first-class native carrier."""
        order = (
            flat_index.device_order
            if getattr(flat_index, "device_order", None) is not None
            else getattr(flat_index, "_host_order", None)
        )
        morton_keys = (
            flat_index.device_morton_keys
            if getattr(flat_index, "device_morton_keys", None) is not None
            else getattr(flat_index, "_host_morton_keys", None)
        )
        metadata = NativeGeometryMetadata.from_spatial_index(
            flat_index,
            source_token=source_token,
        )
        residency = _device_or_host_residency(metadata.bounds, order, morton_keys)
        return cls(
            kind="flat-morton",
            row_count=int(flat_index.geometry_array.row_count),
            geometry=flat_index.geometry_array,
            order=order,
            morton_keys=morton_keys,
            metadata=metadata,
            regular_grid=flat_index.regular_grid,
            total_bounds=flat_index.total_bounds,
            source_token=source_token,
            index_parameters={
                "regular_grid": flat_index.regular_grid is not None,
                "device_bounds": getattr(flat_index, "device_bounds", None) is not None,
                "device_order": getattr(flat_index, "device_order", None) is not None,
            },
            residency=residency,
        )

    @property
    def is_device(self) -> bool:
        return self.residency is Residency.DEVICE

    def validate_row_count(self, row_count: int) -> None:
        if int(row_count) != int(self.row_count):
            raise ValueError(
                f"NativeSpatialIndex row count mismatch: expected "
                f"{self.row_count}, got {row_count}"
            )

    def to_flat_index(self):
        """Return a transitional flat-index view without rebuilding index state."""
        if self.metadata is None or self.metadata.bounds is None:
            raise ValueError("NativeSpatialIndex query requires metadata bounds")

        from vibespatial.spatial.indexing import FlatSpatialIndex

        host_bounds, device_bounds = _host_or_device_slots(self.metadata.bounds)
        host_order, device_order = _host_or_device_slots(self.order)
        host_morton_keys, device_morton_keys = _host_or_device_slots(self.morton_keys)
        total_bounds = self.total_bounds or self.metadata.total_bounds
        if total_bounds is None:
            total_bounds = (float("nan"),) * 4

        return FlatSpatialIndex(
            geometry_array=self.geometry,
            _host_order=host_order,
            _host_morton_keys=host_morton_keys,
            _host_bounds=host_bounds,
            total_bounds=total_bounds,
            regular_grid=self.regular_grid,
            device_morton_keys=device_morton_keys,
            device_order=device_order,
            device_bounds=device_bounds,
        )

    def query_relation(
        self,
        query_geometry: Any,
        *,
        predicate: str | None = None,
        sort: bool = False,
        distance: float | np.ndarray | None = None,
        query_token: str | None = None,
        query_row_count: int | None = None,
        return_device: bool = True,
        return_metadata: bool = False,
        tree_shapely: np.ndarray | None = None,
        query_shapely: np.ndarray | None = None,
        precomputed_query_bounds: np.ndarray | None = None,
    ):
        """Query this native index and return relation-pair row flow.

        Physical shape: candidate/predicate pair generation over a reusable
        ``NativeSpatialIndex``.  Native input carriers are ``NativeSpatialIndex``
        plus owned query geometry or ``NativeFrameState``.  The native output
        carrier is ``NativeRelation`` with query rows on the left and indexed
        rows on the right.
        """
        query_owned = _query_owned_geometry(query_geometry)
        resolved_query_token = (
            getattr(query_geometry, "lineage_token", None)
            if query_token is None
            else query_token
        )
        resolved_query_row_count = (
            getattr(query_owned, "row_count", None)
            if query_row_count is None
            else query_row_count
        )
        if resolved_query_row_count is None:
            raise ValueError("NativeSpatialIndex.query_relation requires query row count")

        from vibespatial.api._native_relation import (
            NativeRelation,
            NativeRelationSelection,
        )
        from vibespatial.spatial.query import query_spatial_index

        query_result, execution = query_spatial_index(
            self.geometry,
            self.to_flat_index(),
            query_owned,
            predicate=predicate,
            sort=sort,
            distance=distance,
            output_format="indices",
            return_metadata=True,
            return_device=return_device,
            tree_shapely=tree_shapely,
            query_shapely=query_shapely,
            precomputed_query_bounds=precomputed_query_bounds,
        )
        if isinstance(query_result, NativeRelationSelection):
            query_relation = query_result.relation
            relation = NativeRelation(
                left_indices=query_relation.left_indices,
                right_indices=query_relation.right_indices,
                left_token=resolved_query_token,
                right_token=self.source_token,
                predicate=predicate,
                distances=query_relation.distances,
                left_row_count=int(resolved_query_row_count),
                right_row_count=int(self.row_count),
                sorted_by_left=query_relation.sorted_by_left,
                duplicate_policy=query_relation.duplicate_policy,
            )
            selected_relation = NativeRelationSelection(
                relation=relation,
                selection=query_result.selection,
            )
            return (
                (selected_relation, execution)
                if return_metadata
                else selected_relation
            )

        left_indices, right_indices = _relation_pair_arrays_from_query_result(
            query_result,
            query_row_count=int(resolved_query_row_count),
        )
        relation = NativeRelation(
            left_indices=left_indices,
            right_indices=right_indices,
            left_token=resolved_query_token,
            right_token=self.source_token,
            predicate=predicate,
            left_row_count=int(resolved_query_row_count),
            right_row_count=int(self.row_count),
            sorted_by_left=bool(
                sort or getattr(query_result, "sorted_by_left", False)
            ),
        )
        return (relation, execution) if return_metadata else relation

    def _query_device_reduction(
        self,
        query_owned: Any,
        *,
        predicate: str | None,
        precomputed_query_bounds: Any | None,
        reduction: str,
    ):
        if not (
            self.is_device
            and _is_device_array(self.order)
            and _is_device_array(self.morton_keys)
            and self.metadata is not None
            and self.metadata.bounds is not None
        ):
            return None, None

        from vibespatial.kernels.core.geometry_analysis import (
            compute_geometry_bounds_device,
        )
        from vibespatial.spatial.spatial_index_device import (
            spatial_index_device_left_match_counts,
            spatial_index_device_left_semijoin_rows,
            spatial_index_device_right_match_counts,
            spatial_index_device_right_semijoin_rows,
        )

        query_bounds = (
            precomputed_query_bounds
            if precomputed_query_bounds is not None
            else compute_geometry_bounds_device(
                query_owned,
                preserve_indexed_view=True,
            )
        )
        reducer = {
            "exists": spatial_index_device_left_semijoin_rows,
            "right_exists": spatial_index_device_right_semijoin_rows,
            "count": spatial_index_device_left_match_counts,
            "right_count": spatial_index_device_right_match_counts,
        }[reduction]
        return reducer(
            self.to_flat_index(),
            query_owned,
            self.geometry,
            query_bounds,
            predicate=predicate,
        )

    def _query_device_pair_reduction(
        self,
        query_owned: Any,
        aligned_tree_owned: Any,
        aligned_flat_index: Any,
        *,
        predicate: str,
        precomputed_query_bounds: Any | None,
    ):
        if not (
            self.is_device
            and _is_device_array(self.order)
            and _is_device_array(self.morton_keys)
            and self.metadata is not None
            and self.metadata.bounds is not None
        ):
            return None, None

        from vibespatial.kernels.core.geometry_analysis import (
            compute_geometry_bounds_device,
        )
        from vibespatial.spatial.spatial_index_device import (
            spatial_index_device_right_pair_match_counts,
        )

        query_bounds = (
            precomputed_query_bounds
            if precomputed_query_bounds is not None
            else compute_geometry_bounds_device(
                query_owned,
                preserve_indexed_view=True,
            )
        )
        return spatial_index_device_right_pair_match_counts(
            self.to_flat_index(),
            aligned_flat_index,
            query_owned,
            self.geometry,
            aligned_tree_owned,
            query_bounds,
            predicate=predicate,
        )

    def query_left_semijoin(
        self,
        query_geometry: Any,
        *,
        predicate: str | None = None,
        distance: float | np.ndarray | None = None,
        query_token: str | None = None,
        query_row_count: int | None = None,
        return_metadata: bool = False,
        precomputed_query_bounds: Any | None = None,
    ):
        """Return matched query rows without materializing relation pairs.

        Physical shape: reusable ``NativeSpatialIndex`` Morton ranges feed
        range-sliced candidate-refine tiles and reduce existential matches into
        a capacity-backed ``NativeDeviceSelection``. This preserves the device
        logical count for consumers that discard duplicate relation pairs.
        """
        from vibespatial.api._native_rowset import NativeDeviceSelection

        query_owned = _query_owned_geometry(query_geometry)
        resolved_query_token = (
            getattr(query_geometry, "lineage_token", None)
            if query_token is None
            else query_token
        )
        resolved_query_row_count = (
            getattr(query_owned, "row_count", None)
            if query_row_count is None
            else query_row_count
        )
        if resolved_query_row_count is None:
            raise ValueError("NativeSpatialIndex.query_left_semijoin requires query row count")

        direct_rows = None
        execution = None
        if distance is None:
            direct_rows, execution = self._query_device_reduction(
                query_owned,
                predicate=predicate,
                precomputed_query_bounds=precomputed_query_bounds,
                reduction="exists",
            )

        if direct_rows is None:
            from vibespatial.api._native_relation import NativeRelationSelection

            relation, execution = self.query_relation(
                query_geometry,
                predicate=predicate,
                distance=distance,
                query_token=resolved_query_token,
                query_row_count=int(resolved_query_row_count),
                return_device=True,
                return_metadata=True,
                precomputed_query_bounds=precomputed_query_bounds,
            )
            if isinstance(relation, NativeRelationSelection):
                rowset = relation.left_match_count_expression(
                    source_row_count=int(resolved_query_row_count),
                ).compare_scalar_selection(">", 0)
            else:
                rowset = relation.left_semijoin_rowset(order="sorted")
        else:
            if not isinstance(direct_rows, NativeDeviceSelection):
                raise TypeError("native device semijoin must return a device selection")
            rowset = replace(
                direct_rows,
                source_token=resolved_query_token,
                source_row_count=int(resolved_query_row_count),
            )
        return (rowset, execution) if return_metadata else rowset

    def query_left_antijoin(
        self,
        query_geometry: Any,
        *,
        predicate: str | None = None,
        distance: float | np.ndarray | None = None,
        query_token: str | None = None,
        query_row_count: int | None = None,
        return_metadata: bool = False,
        precomputed_query_bounds: Any | None = None,
    ):
        """Return unmatched query rows without materializing relation pairs."""
        matched, execution = self.query_left_semijoin(
            query_geometry,
            predicate=predicate,
            distance=distance,
            query_token=query_token,
            query_row_count=query_row_count,
            return_metadata=True,
            precomputed_query_bounds=precomputed_query_bounds,
        )
        from vibespatial.api._native_rowset import NativeDeviceSelection

        if isinstance(matched, NativeDeviceSelection):
            rowset = NativeDeviceSelection.from_mask(
                ~matched.source_mask(),
                source_token=matched.source_token,
                source_row_count=matched.source_row_count,
                geometry_family_domain=matched.geometry_family_domain,
                trusted_all_valid_rows=matched.trusted_all_valid_rows,
            )
        else:
            rowset = matched.complement()
        return (rowset, execution) if return_metadata else rowset

    def query_right_semijoin(
        self,
        query_geometry: Any,
        *,
        predicate: str | None = None,
        distance: float | np.ndarray | None = None,
        query_row_count: int | None = None,
        return_metadata: bool = False,
        precomputed_query_bounds: Any | None = None,
    ):
        """Return matched indexed rows without materializing relation pairs."""
        from vibespatial.api._native_rowset import NativeDeviceSelection

        query_owned = _query_owned_geometry(query_geometry)
        resolved_query_row_count = (
            getattr(query_owned, "row_count", None)
            if query_row_count is None
            else query_row_count
        )
        if resolved_query_row_count is None:
            raise ValueError("NativeSpatialIndex.query_right_semijoin requires query row count")

        direct_rows = None
        execution = None
        if distance is None:
            direct_rows, execution = self._query_device_reduction(
                query_owned,
                predicate=predicate,
                precomputed_query_bounds=precomputed_query_bounds,
                reduction="right_exists",
            )

        if direct_rows is None:
            from vibespatial.api._native_relation import NativeRelationSelection

            relation, execution = self.query_relation(
                query_geometry,
                predicate=predicate,
                distance=distance,
                query_row_count=int(resolved_query_row_count),
                return_device=True,
                return_metadata=True,
                precomputed_query_bounds=precomputed_query_bounds,
            )
            if isinstance(relation, NativeRelationSelection):
                rowset = relation.right_match_count_expression(
                    source_row_count=int(self.row_count),
                ).compare_scalar_selection(">", 0)
            else:
                rowset = relation.right_semijoin_rowset(order="sorted")
        else:
            if not isinstance(direct_rows, NativeDeviceSelection):
                raise TypeError("native device semijoin must return a device selection")
            rowset = replace(
                direct_rows,
                source_token=self.source_token,
                source_row_count=int(self.row_count),
            )
        return (rowset, execution) if return_metadata else rowset

    def query_left_match_count_expression(
        self,
        query_geometry: Any,
        *,
        predicate: str | None = None,
        distance: float | np.ndarray | None = None,
        query_token: str | None = None,
        query_row_count: int | None = None,
        return_metadata: bool = False,
        precomputed_query_bounds: Any | None = None,
    ):
        """Return exact per-query match counts without a relation carrier."""
        from vibespatial.api._native_expression import NativeExpression

        query_owned = _query_owned_geometry(query_geometry)
        resolved_query_token = (
            getattr(query_geometry, "lineage_token", None)
            if query_token is None
            else query_token
        )
        resolved_query_row_count = (
            getattr(query_owned, "row_count", None)
            if query_row_count is None
            else query_row_count
        )
        if resolved_query_row_count is None:
            raise ValueError(
                "NativeSpatialIndex.query_left_match_count_expression requires "
                "query row count"
            )

        direct_counts = None
        execution = None
        if distance is None:
            direct_counts, execution = self._query_device_reduction(
                query_owned,
                predicate=predicate,
                precomputed_query_bounds=precomputed_query_bounds,
                reduction="count",
            )

        if direct_counts is None:
            relation, execution = self.query_relation(
                query_geometry,
                predicate=predicate,
                distance=distance,
                query_token=resolved_query_token,
                query_row_count=int(resolved_query_row_count),
                return_device=True,
                return_metadata=True,
                precomputed_query_bounds=precomputed_query_bounds,
            )
            expression = relation.left_match_count_expression()
        else:
            expression = NativeExpression(
                operation="spatial_index.left_match_count",
                values=direct_counts,
                source_token=resolved_query_token,
                source_row_count=int(resolved_query_row_count),
                dtype="int64",
            )
        return (expression, execution) if return_metadata else expression

    def query_right_match_count_expression(
        self,
        query_geometry: Any,
        *,
        predicate: str | None = None,
        distance: float | np.ndarray | None = None,
        query_row_count: int | None = None,
        return_metadata: bool = False,
        precomputed_query_bounds: Any | None = None,
    ):
        """Return exact per-indexed-row match counts without pair export."""
        from vibespatial.api._native_expression import NativeExpression

        query_owned = _query_owned_geometry(query_geometry)
        resolved_query_row_count = (
            getattr(query_owned, "row_count", None)
            if query_row_count is None
            else query_row_count
        )
        if resolved_query_row_count is None:
            raise ValueError(
                "NativeSpatialIndex.query_right_match_count_expression requires "
                "query row count"
            )

        direct_counts = None
        execution = None
        if distance is None:
            direct_counts, execution = self._query_device_reduction(
                query_owned,
                predicate=predicate,
                precomputed_query_bounds=precomputed_query_bounds,
                reduction="right_count",
            )

        if direct_counts is None:
            relation, execution = self.query_relation(
                query_geometry,
                predicate=predicate,
                distance=distance,
                query_row_count=int(resolved_query_row_count),
                return_device=True,
                return_metadata=True,
                precomputed_query_bounds=precomputed_query_bounds,
            )
            expression = relation.right_match_count_expression(
                source_row_count=int(self.row_count),
            )
        else:
            expression = NativeExpression(
                operation="spatial_index.right_match_count",
                values=direct_counts,
                source_token=self.source_token,
                source_row_count=int(self.row_count),
                dtype="int64",
            )
        return (expression, execution) if return_metadata else expression

    def query_right_pair_match_count_expressions(
        self,
        query_geometry: Any,
        aligned_tree_geometry: Any,
        aligned_flat_index: Any,
        *,
        predicate: str,
        query_row_count: int | None = None,
        return_metadata: bool = False,
        precomputed_query_bounds: Any | None = None,
    ):
        """Return first and shared aligned indexed-row match counts."""
        from vibespatial.api._native_expression import NativeExpression

        query_owned = _query_owned_geometry(query_geometry)
        aligned_tree_owned = _query_owned_geometry(aligned_tree_geometry)
        resolved_query_row_count = (
            getattr(query_owned, "row_count", None)
            if query_row_count is None
            else query_row_count
        )
        if resolved_query_row_count is None:
            raise ValueError(
                "NativeSpatialIndex.query_right_pair_match_count_expressions "
                "requires query row count"
            )
        direct_counts, execution = self._query_device_pair_reduction(
            query_owned,
            aligned_tree_owned,
            aligned_flat_index,
            predicate=predicate,
            precomputed_query_bounds=precomputed_query_bounds,
        )
        if direct_counts is None:
            return (None, execution) if return_metadata else None

        operations = (
            "spatial_index.right_pair_match_count.left",
            "spatial_index.right_pair_match_count.right",
            "spatial_index.right_pair_match_count.shared",
        )
        if len(direct_counts) == 2:
            operations = (operations[0], operations[2])
        expressions = tuple(
            NativeExpression(
                operation=operation,
                values=values,
                source_token=self.source_token,
                source_row_count=int(self.row_count),
                dtype="int64",
            )
            for operation, values in zip(operations, direct_counts, strict=True)
        )
        return (expressions, execution) if return_metadata else expressions


def _query_owned_geometry(value: Any):
    geometry = getattr(value, "geometry", None)
    if geometry is not None and getattr(geometry, "owned", None) is not None:
        return geometry.owned
    if getattr(value, "owned", None) is not None:
        return value.owned
    return value


def _relation_pair_arrays_from_query_result(
    query_result: Any,
    *,
    query_row_count: int,
) -> tuple[Any, Any]:
    left_idx = getattr(query_result, "d_left_idx", None)
    right_idx = getattr(query_result, "d_right_idx", None)
    if left_idx is not None and right_idx is not None:
        return left_idx, right_idx

    if getattr(query_result, "ndim", None) == 1:
        if query_row_count != 1:
            raise ValueError(
                "scalar spatial query results require query_row_count == 1"
            )
        if _is_device_array(query_result):
            import cupy as cp

            return (
                cp.zeros(int(query_result.size), dtype=cp.int32),
                cp.asarray(query_result, dtype=cp.int32),
            )
        right = np.asarray(query_result, dtype=np.int32)
        return np.zeros(int(right.size), dtype=np.int32), right

    return (
        query_result[0].astype(np.int32, copy=False),
        query_result[1].astype(np.int32, copy=False),
    )


__all__ = ["NativeGeometryMetadata", "NativeSpatialIndex"]
