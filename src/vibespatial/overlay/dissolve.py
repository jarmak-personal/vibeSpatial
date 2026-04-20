from __future__ import annotations

import gc
import logging
import statistics
from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import GeometryCollection

from vibespatial.api._native_results import (
    GeometryNativeResult,
    NativeTabularResult,
    _coerce_constructive_export_frame,
    _grouped_constructive_to_native_tabular_result,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.config import (
    OVERLAY_GPU_FAILURE_THRESHOLD,
    OVERLAY_GROUPED_BOX_GPU_THRESHOLD,
    OVERLAY_GROUPED_COVERAGE_EDGE_THRESHOLD,
    OVERLAY_UNION_ALL_GPU_THRESHOLD,
    SPATIAL_EPSILON,
)
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.fallbacks import record_fallback_event, strict_native_mode_enabled
from vibespatial.runtime.fusion import IntermediateDisposition, PipelineStep, StepKind, plan_fusion
from vibespatial.runtime.provenance import provenance_rewrites_enabled, record_rewrite_event

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    cp = None


if TYPE_CHECKING:
    from collections.abc import Sequence

    from vibespatial.geometry.owned import OwnedGeometryArray
    from vibespatial.runtime import ExecutionMode


_EMPTY = GeometryCollection()
_BUFFERED_TWO_POINT_EXACT_UNION_MAX_UNIQUE_ROWS = 256
_BUFFERED_TWO_POINT_SMALL_PARTIAL_UNION_MAX_ROWS = 2
_BUFFERED_LINE_EXACT_CPU_MAX_ROWS = 2_048
logger = logging.getLogger(__name__)


def _collect_polygonal_parts(geometry) -> list[object]:
    if geometry is None or shapely.is_empty(geometry):
        return []
    geom_type = geometry.geom_type
    if geom_type == "Polygon":
        return [geometry]
    if geom_type == "MultiPolygon":
        return list(shapely.get_parts(geometry))
    if geom_type != "GeometryCollection":
        return []

    parts: list[object] = []
    for part in shapely.get_parts(geometry):
        parts.extend(_collect_polygonal_parts(part))
    return parts


def _canonicalize_polygonal_make_valid_geometry(geometry):
    if geometry is None or shapely.is_empty(geometry):
        return geometry
    if geometry.geom_type in {"Polygon", "MultiPolygon"}:
        return geometry

    polygonal_parts = _collect_polygonal_parts(geometry)
    if not polygonal_parts:
        return geometry
    if len(polygonal_parts) == 1:
        return polygonal_parts[0]

    merged = shapely.union_all(np.asarray(polygonal_parts, dtype=object))
    if merged is None or shapely.is_empty(merged):
        return merged
    if merged.geom_type in {"Polygon", "MultiPolygon"}:
        return merged

    repaired = shapely.make_valid(merged)
    repaired_parts = _collect_polygonal_parts(repaired)
    if not repaired_parts:
        return repaired
    if len(repaired_parts) == 1:
        return repaired_parts[0]
    return shapely.union_all(np.asarray(repaired_parts, dtype=object))


def _canonicalize_polygonal_make_valid_values(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    return np.asarray(
        [_canonicalize_polygonal_make_valid_geometry(value) for value in values],
        dtype=object,
    )


def _recompute_invalid_grouped_union_owned_rows(
    reduced: OwnedGeometryArray,
    *,
    ordered_owned: OwnedGeometryArray,
    offsets: np.ndarray,
    group_count: int,
) -> OwnedGeometryArray:
    from vibespatial.constructive.validity import is_valid_owned
    from vibespatial.geometry.owned import (
        concat_owned_scatter,
        from_shapely_geometries,
        seed_all_validity_cache,
    )

    invalid_mask = ~np.asarray(is_valid_owned(reduced), dtype=bool)
    if not bool(np.all(reduced.validity)):
        invalid_mask = invalid_mask.copy()
        invalid_mask[~reduced.validity] = False
    if not invalid_mask.any():
        seed_all_validity_cache(reduced)
        return reduced

    invalid_rows = np.flatnonzero(invalid_mask).astype(np.int64, copy=False)
    repaired_values: list[object] = []
    for row in invalid_rows:
        start = int(offsets[row])
        stop = int(offsets[row + 1])
        members = np.asarray(
            ordered_owned.take(np.arange(start, stop, dtype=np.int64)).to_shapely(),
            dtype=object,
        )
        members = members[
            [geom is not None and not shapely.is_empty(geom) for geom in members]
        ]
        if members.size == 0:
            repaired_values.append(_EMPTY)
            continue

        merged = shapely.union_all(members)
        if merged is not None and merged.geom_type == "GeometryCollection":
            merged = _canonicalize_polygonal_make_valid_geometry(merged)
        if merged is not None and not shapely.is_valid(merged):
            merged = _canonicalize_polygonal_make_valid_geometry(shapely.make_valid(merged))
        repaired_values.append(merged)

    repaired_subset = from_shapely_geometries(
        repaired_values,
        residency=reduced.residency,
    )
    repaired = concat_owned_scatter(reduced, repaired_subset, invalid_rows)
    seed_all_validity_cache(repaired)
    record_dispatch_event(
        surface="geopandas.geodataframe.dissolve",
        operation="grouped_union_boundary_repair",
        implementation="shapely.union_all_subset",
        reason="recomputed invalid grouped GPU unions from original per-group members",
        detail=(
            f"groups={group_count}, "
            f"repaired={int(np.count_nonzero(invalid_mask))}"
        ),
        selected=ExecutionMode.CPU,
    )
    return repaired


class DissolveUnionMethod(StrEnum):
    UNARY = "unary"
    COVERAGE = "coverage"
    DISJOINT_SUBSET = "disjoint_subset"


class DissolvePrimitive(StrEnum):
    ENCODE_KEYS = "encode_keys"
    STABLE_SORT = "stable_sort"
    RUN_LENGTH_ENCODE = "run_length_encode"
    REDUCE_BY_KEY = "reduce_by_key"
    GROUPED_UNION = "grouped_union"
    GATHER = "gather"
    SCATTER = "scatter"
    ASSEMBLE_FRAME = "assemble_frame"


@dataclass(frozen=True)
class DissolveStage:
    name: str
    primitive: DissolvePrimitive
    purpose: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    cccl_mapping: tuple[str, ...]
    disposition: IntermediateDisposition
    preserves_group_order: bool = False
    geometry_producing: bool = False


@dataclass(frozen=True)
class DissolvePipelinePlan:
    method: DissolveUnionMethod
    stages: tuple[DissolveStage, ...]
    fusion_steps: tuple[PipelineStep, ...]
    reason: str


@dataclass(frozen=True)
class GroupedUnionResult:
    geometries: np.ndarray | None
    group_count: int
    non_empty_groups: int
    empty_groups: int
    method: DissolveUnionMethod
    owned: OwnedGeometryArray | None = None


@dataclass(frozen=True)
class DissolveBenchmark:
    dataset: str
    rows: int
    groups: int
    pipeline_elapsed_seconds: float
    baseline_elapsed_seconds: float
    iterations: int = 1

    @property
    def speedup_vs_baseline(self) -> float:
        if self.pipeline_elapsed_seconds == 0.0:
            return float("inf")
        return self.baseline_elapsed_seconds / self.pipeline_elapsed_seconds


def _grouped_union_geometry_result(
    grouped_union: GroupedUnionResult,
    *,
    geometry_name: str,
    crs,
) -> GeometryNativeResult:
    from vibespatial.api.geoseries import GeoSeries

    if grouped_union.owned is not None:
        return _repair_grouped_union_owned_if_needed(
            grouped_union.owned,
            group_count=grouped_union.group_count,
            geometry_name=geometry_name,
            crs=crs,
        )

    return GeometryNativeResult.from_geoseries(
        GeoSeries(grouped_union.geometries, name=geometry_name, crs=crs)
    )


def _grouped_constructive_result(
    grouped_union: GroupedUnionResult,
    *,
    frame,
    aggregated_data: pd.DataFrame,
    as_index: bool,
) -> NativeTabularResult:
    return _grouped_constructive_to_native_tabular_result(
        geometry=_grouped_union_geometry_result(
            grouped_union,
            geometry_name=frame.geometry.name,
            crs=frame.crs,
        ),
        geometry_name=frame.geometry.name,
        as_index=as_index,
        attributes=aggregated_data,
    )


class LazyDissolvedFrame:
    """Predicate-first grouped dissolve view with on-demand materialization."""

    def __init__(
        self,
        *,
        frame,
        aggregated_data: pd.DataFrame,
        group_positions: list[np.ndarray],
        row_group_codes: np.ndarray | None,
        method: DissolveUnionMethod,
        grid_size: float | None,
        as_index: bool,
        owned: OwnedGeometryArray | None,
    ) -> None:
        self._frame = frame
        self._aggregated_data = aggregated_data
        self._group_positions = [np.asarray(pos, dtype=np.int32) for pos in group_positions]
        self._row_group_codes = None if row_group_codes is None else np.asarray(row_group_codes, dtype=np.int32)
        self._method = method
        self._grid_size = grid_size
        self._as_index = as_index
        self._owned = owned
        self._geometry_name = frame.geometry.name
        self._materialized = None
        self._group_bounds = None

    def __len__(self) -> int:
        return len(self._group_positions)

    def __getitem__(self, key):
        return self.attributes[key]

    @property
    def attributes(self) -> pd.DataFrame:
        if self._as_index:
            return self._aggregated_data.copy()
        return self._aggregated_data.reset_index()

    @property
    def group_index(self) -> pd.Index:
        return self._aggregated_data.index

    @property
    def geometry(self):
        return self.materialize()[self._geometry_name]

    def _materialized_geometry_column(self, grouped_union: GroupedUnionResult):
        if grouped_union.owned is not None:
            from vibespatial.io.geoarrow import geoseries_from_owned

            return geoseries_from_owned(
                grouped_union.owned,
                name=self._geometry_name,
                crs=self._frame.crs,
            )
        return grouped_union.geometries

    def to_native_result(self) -> NativeTabularResult:
        self._ensure_group_positions()
        if self._row_group_codes is not None:
            grouped_union = execute_grouped_union_codes(
                np.asarray(self._frame.geometry.array, dtype=object),
                self._row_group_codes,
                group_count=len(self._aggregated_data.index),
                method=self._method,
                grid_size=self._grid_size,
                owned=self._owned,
            )
        else:
            grouped_union = None
        if grouped_union is None:
            grouped_union = execute_grouped_union(
                np.asarray(self._frame.geometry.array, dtype=object),
                self._group_positions,
                method=self._method,
                grid_size=self._grid_size,
                owned=self._owned,
            )
        return _grouped_constructive_result(
            grouped_union,
            frame=self._frame,
            aggregated_data=self._aggregated_data,
            as_index=self._as_index,
        )

    def materialize(self):
        if self._materialized is not None:
            return self._materialized
        self._materialized = _coerce_constructive_export_frame(
            self.to_native_result().to_geodataframe(),
            geometry_name=self._geometry_name,
            frame_type=type(self._frame),
        )
        return self._materialized

    def to_geodataframe(self):
        return self.materialize()

    def intersects(self, other) -> pd.Series:
        query_values, scalar = _coerce_lazy_query(other)
        if not scalar:
            record_dispatch_event(
                surface="geopandas.geodataframe.dissolve_lazy",
                operation="intersects",
                implementation="materialized_fallback",
                reason="row-wise lazy intersects currently supports scalar queries only",
                detail=f"groups={len(self)}",
                selected=ExecutionMode.CPU,
            )
            return self.geometry.intersects(other, align=False)
        return self._scalar_group_any_predicate("intersects", query_values[0], use_bbox=True)

    def contains(self, other) -> pd.Series:
        query_values, scalar = _coerce_lazy_query(other)
        if (
            not scalar
            or not _is_point_family_query(query_values[0])
            or self._method is not DissolveUnionMethod.COVERAGE
        ):
            record_dispatch_event(
                surface="geopandas.geodataframe.dissolve_lazy",
                operation="contains",
                implementation="materialized_fallback",
                reason="lazy contains fast path only preserves exact semantics for scalar point queries on coverage groups",
                detail=f"groups={len(self)}",
                selected=ExecutionMode.CPU,
            )
            return self.geometry.contains(other, align=False)
        return self._scalar_group_contains_point_coverage(query_values[0])

    def _scalar_group_any_predicate(self, predicate: str, query, *, use_bbox: bool) -> pd.Series:
        self._ensure_group_positions()
        result = np.zeros(len(self._group_positions), dtype=bool)
        if len(self._group_positions) == 0:
            return pd.Series(result, index=self._aggregated_data.index, copy=False)

        candidate_groups = np.arange(len(self._group_positions), dtype=np.int32)
        if use_bbox:
            group_bounds = self._ensure_group_bounds()
            query_bounds = np.asarray(shapely.bounds(np.asarray([query], dtype=object)), dtype=np.float64)[0]
            overlaps = _bbox_overlaps(group_bounds, query_bounds)
            candidate_groups = np.flatnonzero(overlaps).astype(np.int32, copy=False)
            if candidate_groups.size == 0:
                return pd.Series(result, index=self._aggregated_data.index, copy=False)

        member_positions = np.concatenate([self._group_positions[int(i)] for i in candidate_groups if self._group_positions[int(i)].size])
        if member_positions.size == 0:
            return pd.Series(result, index=self._aggregated_data.index, copy=False)

        member_lengths = np.asarray(
            [self._group_positions[int(i)].size for i in candidate_groups if self._group_positions[int(i)].size],
            dtype=np.int32,
        )
        non_empty_candidate_groups = candidate_groups[member_lengths > 0]
        if member_lengths.size == 0:
            return pd.Series(result, index=self._aggregated_data.index, copy=False)

        predicate_values = getattr(self._frame.geometry.iloc[member_positions], predicate)(query, align=False)
        member_hits = np.asarray(predicate_values, dtype=bool)
        offsets = np.concatenate(
            [np.asarray([0], dtype=np.int32), np.cumsum(member_lengths[:-1], dtype=np.int32)]
        )
        reduced = np.maximum.reduceat(member_hits.astype(np.int8, copy=False), offsets).astype(bool, copy=False)
        result[non_empty_candidate_groups.astype(np.intp, copy=False)] = reduced
        return pd.Series(result, index=self._aggregated_data.index, copy=False)

    def _scalar_group_contains_point_coverage(self, query) -> pd.Series:
        self._ensure_group_positions()
        result = np.zeros(len(self._group_positions), dtype=bool)
        if len(self._group_positions) == 0:
            return pd.Series(result, index=self._aggregated_data.index, copy=False)

        group_bounds = self._ensure_group_bounds()
        query_bounds = np.asarray(shapely.bounds(np.asarray([query], dtype=object)), dtype=np.float64)[0]
        candidate_groups = np.flatnonzero(_bbox_overlaps(group_bounds, query_bounds)).astype(np.int32, copy=False)
        if candidate_groups.size == 0:
            return pd.Series(result, index=self._aggregated_data.index, copy=False)

        non_empty_candidate_groups = np.asarray(
            [int(i) for i in candidate_groups if self._group_positions[int(i)].size],
            dtype=np.int32,
        )
        if non_empty_candidate_groups.size == 0:
            return pd.Series(result, index=self._aggregated_data.index, copy=False)
        member_lengths = np.asarray(
            [self._group_positions[int(i)].size for i in non_empty_candidate_groups],
            dtype=np.int32,
        )
        member_positions = np.concatenate([self._group_positions[int(i)] for i in non_empty_candidate_groups])
        offsets = np.concatenate(
            [np.asarray([0], dtype=np.int32), np.cumsum(member_lengths[:-1], dtype=np.int32)]
        )

        member_contains = np.asarray(
            self._frame.geometry.iloc[member_positions].contains(query, align=False),
            dtype=bool,
        )
        contains_any = np.maximum.reduceat(
            member_contains.astype(np.int8, copy=False),
            offsets,
        ).astype(bool, copy=False)
        result[non_empty_candidate_groups.astype(np.intp, copy=False)] = contains_any

        unresolved_groups = non_empty_candidate_groups[~contains_any]
        if unresolved_groups.size == 0:
            return pd.Series(result, index=self._aggregated_data.index, copy=False)

        unresolved_lengths = np.asarray(
            [self._group_positions[int(i)].size for i in unresolved_groups],
            dtype=np.int32,
        )
        unresolved_positions = np.concatenate([self._group_positions[int(i)] for i in unresolved_groups])
        unresolved_offsets = np.concatenate(
            [np.asarray([0], dtype=np.int32), np.cumsum(unresolved_lengths[:-1], dtype=np.int32)]
        )
        member_covers = np.asarray(
            self._frame.geometry.iloc[unresolved_positions].covers(query, align=False),
            dtype=bool,
        )
        cover_any = np.maximum.reduceat(
            member_covers.astype(np.int8, copy=False),
            unresolved_offsets,
        ).astype(bool, copy=False)
        ambiguous_groups = unresolved_groups[cover_any]
        if ambiguous_groups.size:
            materialized = np.asarray(self.geometry.contains(query, align=False), dtype=bool)
            result[ambiguous_groups.astype(np.intp, copy=False)] = materialized[
                ambiguous_groups.astype(np.intp, copy=False)
            ]
        return pd.Series(result, index=self._aggregated_data.index, copy=False)

    def _ensure_group_bounds(self) -> np.ndarray:
        if self._group_bounds is not None:
            return self._group_bounds
        self._ensure_group_positions()
        values = np.asarray(self._frame.geometry.array, dtype=object)
        row_bounds = np.asarray(shapely.bounds(values), dtype=np.float64)
        group_bounds = np.full((len(self._group_positions), 4), np.nan, dtype=np.float64)
        for group_index, positions in enumerate(self._group_positions):
            if positions.size == 0:
                continue
            block = row_bounds[positions]
            valid = np.isfinite(block).all(axis=1)
            if not np.any(valid):
                continue
            block = block[valid]
            group_bounds[group_index, 0] = float(block[:, 0].min())
            group_bounds[group_index, 1] = float(block[:, 1].min())
            group_bounds[group_index, 2] = float(block[:, 2].max())
            group_bounds[group_index, 3] = float(block[:, 3].max())
        self._group_bounds = group_bounds
        return group_bounds

    def _ensure_group_positions(self) -> None:
        if self._group_positions:
            return
        if self._row_group_codes is None:
            return
        self._group_positions = _group_positions_from_codes(
            self._row_group_codes,
            len(self._aggregated_data.index),
        )


def plan_dissolve_pipeline(method: DissolveUnionMethod | str = DissolveUnionMethod.UNARY) -> DissolvePipelinePlan:
    normalized = method if isinstance(method, DissolveUnionMethod) else DissolveUnionMethod(method)
    stages = (
        DissolveStage(
            name="encode_groups",
            primitive=DissolvePrimitive.ENCODE_KEYS,
            purpose="Encode dissolve group keys into compact integer labels without materializing per-row Python objects.",
            inputs=("group_keys",),
            outputs=("group_codes",),
            cccl_mapping=("transform",),
            disposition=IntermediateDisposition.EPHEMERAL,
        ),
        DissolveStage(
            name="stable_sort_rows",
            primitive=DissolvePrimitive.STABLE_SORT,
            purpose="Stable-sort rows by encoded group so grouped unions and attribute aggregation see deterministic row order.",
            inputs=("group_codes", "geometry_rows", "attribute_rows"),
            outputs=("sorted_group_codes", "sorted_geometry_rows", "sorted_attribute_rows"),
            cccl_mapping=("stable_sort", "gather"),
            disposition=IntermediateDisposition.EPHEMERAL,
            preserves_group_order=True,
        ),
        DissolveStage(
            name="segment_groups",
            primitive=DissolvePrimitive.RUN_LENGTH_ENCODE,
            purpose="Build group spans once so geometry union and attribute reduction can share the same segmentation.",
            inputs=("sorted_group_codes",),
            outputs=("group_offsets", "unique_group_codes"),
            cccl_mapping=("run_length_encode", "prefix_sum"),
            disposition=IntermediateDisposition.EPHEMERAL,
            preserves_group_order=True,
        ),
        DissolveStage(
            name="aggregate_attributes",
            primitive=DissolvePrimitive.REDUCE_BY_KEY,
            purpose="Reduce non-geometry columns per group while leaving pandas-level agg semantics above the geometry kernel seam.",
            inputs=("sorted_group_codes", "sorted_attribute_rows"),
            outputs=("aggregated_attributes",),
            cccl_mapping=("reduce_by_key",),
            disposition=IntermediateDisposition.EPHEMERAL,
            preserves_group_order=True,
        ),
        DissolveStage(
            name="union_group_geometries",
            primitive=DissolvePrimitive.GROUPED_UNION,
            purpose="Union each group independently so future GPU kernels can map grouped constructive work onto per-group overlays.",
            inputs=("group_offsets", "sorted_geometry_rows"),
            outputs=("group_geometries",),
            cccl_mapping=("DeviceSelect", "reduce_by_key", "scatter"),
            disposition=IntermediateDisposition.EPHEMERAL,
            preserves_group_order=True,
            geometry_producing=True,
        ),
        DissolveStage(
            name="assemble_result_frame",
            primitive=DissolvePrimitive.ASSEMBLE_FRAME,
            purpose="Attach grouped geometries to aggregated attributes and emit the final GeoDataFrame surface.",
            inputs=("aggregated_attributes", "group_geometries"),
            outputs=("dissolved_frame",),
            cccl_mapping=("gather", "scatter"),
            disposition=IntermediateDisposition.PERSIST,
            preserves_group_order=True,
            geometry_producing=True,
        ),
    )
    fusion_steps = (
        PipelineStep(name="group_codes", kind=StepKind.ORDERING, output_name="group_codes"),
        PipelineStep(name="sorted_rows", kind=StepKind.ORDERING, output_name="sorted_geometry_rows"),
        PipelineStep(name="group_offsets", kind=StepKind.ORDERING, output_name="group_offsets"),
        PipelineStep(name="aggregated_attributes", kind=StepKind.DERIVED, output_name="aggregated_attributes"),
        PipelineStep(name="group_geometries", kind=StepKind.GEOMETRY, output_name="group_geometries"),
        PipelineStep(
            name="dissolved_frame",
            kind=StepKind.GEOMETRY,
            output_name="dissolved_frame",
            reusable_output=True,
        ),
    )
    return DissolvePipelinePlan(
        method=normalized,
        stages=stages,
        fusion_steps=fusion_steps,
        reason=(
            "Dissolve should encode groups once, keep row order stable within each group, and stage grouped unions "
            "separately from attribute aggregation so a future CUDA path can lean on CCCL sorting, run-length encode, "
            "and reduce-by-key primitives instead of Python group iteration."
        ),
    )


def fusion_plan_for_dissolve(method: DissolveUnionMethod | str = DissolveUnionMethod.UNARY):
    return plan_fusion(plan_dissolve_pipeline(method).fusion_steps)


def _keys_equal(left: Any, right: Any) -> bool:
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(_keys_equal(a, b) for a, b in zip(left, right, strict=True))
    if pd.isna(left) and pd.isna(right):
        return True
    return bool(left == right)


def _lookup_group_positions(indices_items: list[tuple[Any, Any]], key: Any) -> np.ndarray:
    for candidate, positions in indices_items:
        if _keys_equal(candidate, key):
            return np.asarray(positions, dtype=np.int32)
    return np.asarray([], dtype=np.int32)


def _normalize_group_positions(index: pd.Index, indices_items: list[tuple[Any, Any]]) -> list[np.ndarray]:
    if isinstance(index, pd.MultiIndex):
        keys = list(index.tolist())
    else:
        keys = list(index)
    return [_lookup_group_positions(indices_items, key) for key in keys]


def _coerce_lazy_query(other) -> tuple[np.ndarray, bool]:
    try:
        from vibespatial.api.geoseries import GeoSeries
    except Exception:  # pragma: no cover - defensive import seam
        GeoSeries = ()

    if isinstance(other, GeoSeries):
        return np.asarray(other.array, dtype=object), False
    if isinstance(other, np.ndarray):
        if other.ndim == 0 or other.size == 1:
            return np.asarray(other, dtype=object).reshape(1), True
        return np.asarray(other, dtype=object), False
    if isinstance(other, (list, tuple)):
        values = np.asarray(other, dtype=object)
        if values.ndim == 0 or values.size == 1:
            return values.reshape(1), True
        return values, False
    return np.asarray([other], dtype=object), True


def _is_point_family_query(query) -> bool:
    if query is None:
        return False
    type_id = int(np.asarray(shapely.get_type_id(np.asarray([query], dtype=object)), dtype=np.int32)[0])
    return type_id in (0, 4)  # Point / MultiPoint


def _bbox_overlaps(group_bounds: np.ndarray, query_bounds: np.ndarray) -> np.ndarray:
    valid = np.isfinite(group_bounds).all(axis=1)
    if not np.any(valid):
        return np.zeros(group_bounds.shape[0], dtype=bool)
    overlaps = (
        (group_bounds[:, 0] <= query_bounds[2])
        & (group_bounds[:, 2] >= query_bounds[0])
        & (group_bounds[:, 1] <= query_bounds[3])
        & (group_bounds[:, 3] >= query_bounds[1])
    )
    return valid & overlaps


def _build_row_group_codes(
    frame,
    *,
    by,
    level,
    aggregated_index: pd.Index,
) -> np.ndarray | None:
    if len(frame) == 0:
        return np.asarray([], dtype=np.int32)
    try:
        if level is None:
            if by is None:
                row_keys = pd.Index(np.zeros(len(frame), dtype=np.int64))
            elif isinstance(by, str) and by in frame.columns:
                row_keys = pd.Index(frame[by])
            elif (
                isinstance(by, (list, tuple))
                and len(by) > 0
                and all(isinstance(key, str) and key in frame.columns for key in by)
            ):
                row_keys = pd.MultiIndex.from_frame(frame[list(by)])
            else:
                by_array = np.asarray(by, dtype=object)
                if by_array.ndim == 1 and by_array.shape[0] == len(frame):
                    row_keys = pd.Index(by_array)
                else:
                    return None
        else:
            index = frame.index
            if isinstance(level, (list, tuple)):
                if not isinstance(index, pd.MultiIndex):
                    return None
                row_keys = index.droplevel([name for name in index.names if name not in level])
            else:
                row_keys = index.get_level_values(level)
        codes = aggregated_index.get_indexer(row_keys)
    except Exception:
        return None
    return np.asarray(codes, dtype=np.int32)


def _group_offsets_from_sorted_codes(sorted_codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if sorted_codes.size == 0:
        return np.asarray([0], dtype=np.int32), np.asarray([], dtype=np.int32)
    start_mask = np.empty(sorted_codes.size, dtype=bool)
    start_mask[0] = True
    if sorted_codes.size > 1:
        start_mask[1:] = sorted_codes[1:] != sorted_codes[:-1]
    starts = np.flatnonzero(start_mask).astype(np.int32, copy=False)
    unique_codes = sorted_codes[starts]
    offsets = np.concatenate(
        [starts, np.asarray([sorted_codes.size], dtype=np.int32)],
    )
    return offsets, unique_codes


def _group_non_empty_counts(row_group_codes: np.ndarray, group_count: int) -> tuple[int, int]:
    if group_count == 0:
        return 0, 0
    observed = row_group_codes[row_group_codes >= 0]
    if observed.size == 0:
        return 0, group_count
    counts = np.bincount(observed.astype(np.int32, copy=False), minlength=group_count)
    non_empty = int(np.count_nonzero(counts))
    return non_empty, int(group_count - non_empty)


def _group_positions_from_codes(row_group_codes: np.ndarray, group_count: int) -> list[np.ndarray]:
    return [
        np.flatnonzero(row_group_codes == group_index).astype(np.int32, copy=False)
        for group_index in range(group_count)
    ]


def _owned_supports_polygonal_grouped_union(owned: OwnedGeometryArray) -> bool:
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import FAMILY_TAGS

    valid_tags = owned.tags[owned.validity]
    polygon_tags = np.asarray(
        [
            FAMILY_TAGS[GeometryFamily.POLYGON],
            FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
        ],
        dtype=valid_tags.dtype if valid_tags.size else np.int8,
    )
    return bool(valid_tags.size == 0 or np.all(np.isin(valid_tags, polygon_tags)))


def _union_block(values: np.ndarray, method: DissolveUnionMethod, grid_size: float | None) -> object:
    if values.size == 0:
        return _EMPTY
    if method is DissolveUnionMethod.COVERAGE:
        return shapely.coverage_union_all(values)
    if method is DissolveUnionMethod.DISJOINT_SUBSET:
        return shapely.disjoint_subset_union_all(values)
    return shapely.union_all(values, grid_size=grid_size)


def _repair_grouped_union_owned_if_needed(
    owned: OwnedGeometryArray,
    *,
    group_count: int,
    geometry_name: str,
    crs,
) -> GeometryNativeResult:
    """Repair invalid grouped-union outputs while preserving the native seam."""
    from vibespatial.api.geoseries import GeoSeries
    from vibespatial.constructive.make_valid_pipeline import make_valid_owned
    from vibespatial.constructive.validity import is_valid_owned
    from vibespatial.geometry.owned import (
        concat_owned_scatter,
        from_shapely_geometries,
        seed_all_validity_cache,
    )

    invalid_mask = ~np.asarray(is_valid_owned(owned), dtype=bool)
    if not bool(np.all(owned.validity)):
        invalid_mask = invalid_mask.copy()
        invalid_mask[~owned.validity] = False
    if not invalid_mask.any():
        seed_all_validity_cache(owned)
        return GeometryNativeResult.from_owned(owned, crs=crs)

    requested_mode = ExecutionMode.GPU if strict_native_mode_enabled() else ExecutionMode.AUTO
    mv_result = make_valid_owned(owned=owned, dispatch_mode=requested_mode)

    if strict_native_mode_enabled() and (
        mv_result.selected is not ExecutionMode.GPU or mv_result.owned is None
    ):
        record_fallback_event(
            surface="geopandas.geodataframe.dissolve",
            reason="grouped union boundary repair could not stay native",
            detail=(
                f"groups={group_count}, repaired={int(np.count_nonzero(invalid_mask))}, "
                f"selected={mv_result.selected.value}"
            ),
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.CPU,
            pipeline="dissolve.grouped_union_boundary_repair",
            d2h_transfer=True,
        )

    if mv_result.owned is not None:
        repaired_owned = mv_result.owned
        remaining_invalid = ~np.asarray(is_valid_owned(repaired_owned), dtype=bool)
        if not bool(np.all(repaired_owned.validity)):
            remaining_invalid = remaining_invalid.copy()
            remaining_invalid[~repaired_owned.validity] = False
        if remaining_invalid.any():
            remaining_rows = np.flatnonzero(remaining_invalid).astype(np.int64, copy=False)
            if strict_native_mode_enabled():
                record_fallback_event(
                    surface="geopandas.geodataframe.dissolve",
                    reason="grouped union boundary repair requires host compatibility cleanup",
                    detail=(
                        f"groups={group_count}, "
                        f"remaining_invalid={int(np.count_nonzero(remaining_invalid))}"
                    ),
                    requested=ExecutionMode.GPU,
                    selected=ExecutionMode.CPU,
                    pipeline="dissolve.grouped_union_boundary_repair",
                    d2h_transfer=True,
                )
            repaired_values = np.asarray(
                repaired_owned.take(remaining_rows).to_shapely(),
                dtype=object,
            )
            repaired_values = np.asarray(
                shapely.make_valid(repaired_values),
                dtype=object,
            )
            repaired_values = _canonicalize_polygonal_make_valid_values(repaired_values)
            repaired_subset = from_shapely_geometries(
                repaired_values.tolist(),
                residency=repaired_owned.residency,
            )
            repaired_owned = concat_owned_scatter(
                repaired_owned,
                repaired_subset,
                remaining_rows,
            )
            record_dispatch_event(
                surface="geopandas.geodataframe.dissolve",
                operation="grouped_union_boundary_repair",
                implementation="shapely.make_valid_subset",
                reason="grouped union boundary repair for residual invalid dissolved rows",
                detail=(
                    f"groups={group_count}, "
                    f"repaired={int(np.count_nonzero(remaining_invalid))}"
                ),
                selected=ExecutionMode.CPU,
            )
        seed_all_validity_cache(repaired_owned)
        return GeometryNativeResult.from_owned(repaired_owned, crs=crs)

    record_dispatch_event(
        surface="geopandas.geodataframe.dissolve",
        operation="grouped_union_boundary_repair",
        implementation="shapely.make_valid_fallback",
        reason="grouped union boundary repair fell back to host compatibility cleanup",
        detail=(
            f"groups={group_count}, "
            f"repaired={int(np.count_nonzero(invalid_mask))}"
        ),
        selected=ExecutionMode.CPU,
    )
    return GeometryNativeResult.from_geoseries(
        GeoSeries(
            _canonicalize_polygonal_make_valid_values(
                np.asarray(mv_result.geometries, dtype=object)
            ),
            name=geometry_name,
            crs=crs,
        )
    )


def _greedy_bbox_disjoint_coloring(bounds: np.ndarray) -> np.ndarray | None:
    row_count = int(bounds.shape[0])
    if row_count == 0:
        return np.empty(0, dtype=np.int32)
    if bounds.ndim != 2 or bounds.shape[1] != 4:
        return None

    overlaps = (
        (bounds[:, 0][:, None] <= bounds[:, 2][None, :])
        & (bounds[:, 2][:, None] >= bounds[:, 0][None, :])
        & (bounds[:, 1][:, None] <= bounds[:, 3][None, :])
        & (bounds[:, 3][:, None] >= bounds[:, 1][None, :])
    )
    np.fill_diagonal(overlaps, False)

    degrees = overlaps.sum(axis=1, dtype=np.int32)
    max_degree = int(degrees.max(initial=0))
    if max_degree == 0:
        return np.zeros(row_count, dtype=np.int32)

    order = np.argsort(-degrees, kind="stable")
    colors = np.full(row_count, -1, dtype=np.int32)
    for node in order:
        used = np.unique(colors[overlaps[node] & (colors >= 0)])
        if used.size == 0:
            colors[node] = 0
            continue
        available = np.ones(int(used[-1]) + 2, dtype=bool)
        available[used.astype(np.intp, copy=False)] = False
        colors[node] = int(np.flatnonzero(available)[0])
    return colors


def _reorder_small_partial_union_groups_by_overlap(
    expanded_bounds: np.ndarray,
    color_rows: Sequence[np.ndarray],
) -> list[np.ndarray]:
    """Order small disjoint color groups to reduce first-round exact union cost.

    The buffered two-point dissolve rewrite ultimately exact-unions one
    single-row partial polygon per color group. For tiny partial counts, the
    first pairing order materially affects overlay complexity. Use the already
    computed expanded source-line bounds as a cheap proxy: groups with fewer
    inter-group expanded-bbox overlaps are paired first.
    """
    if len(color_rows) < 4 or len(color_rows) > 8:
        return list(color_rows)

    remaining = list(range(len(color_rows)))
    ordered_indices: list[int] = []

    while len(remaining) > 1:
        best_pair: tuple[int, int] | None = None
        best_score: tuple[int, int, int, int] | None = None

        for pos, left_index in enumerate(remaining[:-1]):
            left_bounds = expanded_bounds[color_rows[left_index]]
            for right_index in remaining[pos + 1:]:
                right_bounds = expanded_bounds[color_rows[right_index]]
                overlap_count = int(
                    np.count_nonzero(
                        (left_bounds[:, 0][:, None] <= right_bounds[:, 2][None, :])
                        & (left_bounds[:, 2][:, None] >= right_bounds[:, 0][None, :])
                        & (left_bounds[:, 1][:, None] <= right_bounds[:, 3][None, :])
                        & (left_bounds[:, 3][:, None] >= right_bounds[:, 1][None, :])
                    )
                )
                score = (
                    overlap_count,
                    abs(int(color_rows[left_index].size) - int(color_rows[right_index].size)),
                    left_index,
                    right_index,
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_pair = (left_index, right_index)

        assert best_pair is not None
        ordered_indices.extend(best_pair)
        remaining = [index for index in remaining if index not in best_pair]

    ordered_indices.extend(remaining)
    return [np.asarray(color_rows[index], dtype=np.int64, copy=False) for index in ordered_indices]


def _union_small_partial_rows_gpu(partials: Sequence[OwnedGeometryArray]) -> OwnedGeometryArray | None:
    """Exact GPU union for a tiny list of single-row partial polygons.

    The generic union_all tree reducer pays unnecessary localization and
    reduction overhead when the buffered two-point dissolve rewrite has
    already collapsed the corridor to a few partial rows. Keep this final
    merge on the exact binary overlay path instead.
    """
    from vibespatial.constructive.binary_constructive import binary_constructive_owned

    if not partials:
        return None
    if len(partials) == 1:
        return partials[0]
    if len(partials) > _BUFFERED_TWO_POINT_SMALL_PARTIAL_UNION_MAX_ROWS:
        return None
    if any(partial.row_count != 1 for partial in partials):
        return None

    current = list(partials)
    try:
        while len(current) > 1:
            next_round: list[OwnedGeometryArray] = []
            limit = len(current) - 1
            for i in range(0, limit, 2):
                next_round.append(
                    binary_constructive_owned(
                        "union",
                        current[i],
                        current[i + 1],
                        dispatch_mode=ExecutionMode.GPU,
                    )
                )
            if len(current) % 2:
                next_round.append(current[-1])
            current = next_round
    except Exception:
        logger.debug(
            "small buffered-line partial union failed; falling back to generic union_all",
            exc_info=True,
        )
        return None
    return current[0]


def _reduce_partial_rows_gpu(partials: Sequence[OwnedGeometryArray]) -> OwnedGeometryArray | None:
    """Reduce a tiny list of single-row partial polygons on the exact GPU path."""
    from vibespatial.constructive.union_all import _tree_reduce_global, union_all_gpu_owned
    from vibespatial.geometry.owned import OwnedGeometryArray

    if not partials:
        return None
    if len(partials) == 1:
        return partials[0]

    reduced = _union_small_partial_rows_gpu(partials)
    if reduced is not None:
        return reduced

    merged = OwnedGeometryArray.concat(list(partials))
    if merged.row_count <= 8:
        try:
            return _tree_reduce_global(merged, "union")
        except Exception:
            logger.debug(
                "tiny partial tree reduction failed; falling back to generic union_all",
                exc_info=True,
            )
    return union_all_gpu_owned(
        merged,
        dispatch_mode=ExecutionMode.GPU,
    )


def _rectangle_bounds(values: np.ndarray) -> np.ndarray | None:
    if values.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    if not np.all(shapely.get_type_id(values) == 3):
        return None
    if np.any(shapely.is_empty(values)):
        return None
    if not np.all(shapely.get_num_coordinates(values) == 5):
        return None
    bounds = np.asarray(shapely.bounds(values), dtype=np.float64)
    if bounds.ndim != 2 or bounds.shape[1] != 4:
        return None
    expected_area = (bounds[:, 2] - bounds[:, 0]) * (bounds[:, 3] - bounds[:, 1])
    if not np.allclose(np.asarray(shapely.area(values), dtype=np.float64), expected_area):
        return None
    return bounds


def _owned_rectangle_bounds_device(owned: OwnedGeometryArray):
    if cp is None:
        return None

    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import FAMILY_TAGS
    from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device

    device_state = owned._ensure_device_state()
    if set(device_state.families) != {GeometryFamily.POLYGON}:
        return None

    polygon_tag = np.int8(FAMILY_TAGS[GeometryFamily.POLYGON])
    d_validity = cp.asarray(device_state.validity).astype(cp.bool_, copy=False)
    d_tags = cp.asarray(device_state.tags).astype(cp.int8, copy=False)
    if not bool(cp.all(d_validity)) or not bool(cp.all(d_tags == polygon_tag)):
        return None

    polygon_buffer = device_state.families[GeometryFamily.POLYGON]
    if polygon_buffer.ring_offsets is None or bool(cp.any(polygon_buffer.empty_mask)):
        return None

    d_geom_starts = cp.asarray(polygon_buffer.geometry_offsets[:-1]).astype(cp.int32, copy=False)
    d_geom_ends = cp.asarray(polygon_buffer.geometry_offsets[1:]).astype(cp.int32, copy=False)
    if not bool(cp.all((d_geom_ends - d_geom_starts) == 1)):
        return None

    d_ring_offsets = cp.asarray(polygon_buffer.ring_offsets).astype(cp.int32, copy=False)
    d_coord_starts = d_ring_offsets[d_geom_starts]
    d_coord_ends = d_ring_offsets[d_geom_ends]
    if not bool(cp.all((d_coord_ends - d_coord_starts) == 5)):
        return None

    compute_geometry_bounds_device(owned)
    d_bounds = cp.asarray(owned.device_state.row_bounds).reshape(owned.row_count, 4)

    d_offsets = d_coord_starts[:, None] + cp.arange(5, dtype=cp.int32)[None, :]
    d_x = cp.asarray(polygon_buffer.x)[d_offsets]
    d_y = cp.asarray(polygon_buffer.y)[d_offsets]

    d_xmin = d_bounds[:, 0][:, None]
    d_ymin = d_bounds[:, 1][:, None]
    d_xmax = d_bounds[:, 2][:, None]
    d_ymax = d_bounds[:, 3][:, None]
    on_corners = (
        ((d_x == d_xmin) | (d_x == d_xmax))
        & ((d_y == d_ymin) | (d_y == d_ymax))
    )
    closed = (d_x[:, 0] == d_x[:, 4]) & (d_y[:, 0] == d_y[:, 4])
    twice_area = cp.abs(cp.sum(d_x[:, :-1] * d_y[:, 1:] - d_y[:, :-1] * d_x[:, 1:], axis=1))
    expected_twice_area = 2.0 * (d_bounds[:, 2] - d_bounds[:, 0]) * (d_bounds[:, 3] - d_bounds[:, 1])
    # Bounds kernels can use a lower-precision plan on consumer GPUs; keep
    # the bow-tie rejection but allow the small fp rounding error that shows
    # up on larger regular grids.
    if not bool(
        cp.all(
            cp.all(on_corners, axis=1)
            & closed
            & cp.isclose(twice_area, expected_twice_area, rtol=1.0e-10, atol=SPATIAL_EPSILON)
        )
    ):
        return None
    return d_bounds


def execute_grouped_box_union_gpu_owned_codes(
    row_group_codes: np.ndarray,
    *,
    group_count: int,
    owned: OwnedGeometryArray,
) -> GroupedUnionResult | None:
    if cp is None:
        return None

    try:
        d_bounds = _owned_rectangle_bounds_device(owned)
    except RuntimeError:
        return None
    if d_bounds is None:
        return None

    d_codes = cp.asarray(row_group_codes, dtype=cp.int32)
    d_observed_mask = d_codes >= 0
    if not bool(cp.any(d_observed_mask)):
        merged = np.full(group_count, _EMPTY, dtype=object)
        return GroupedUnionResult(
            geometries=merged,
            group_count=group_count,
            non_empty_groups=0,
            empty_groups=group_count,
            method=DissolveUnionMethod.COVERAGE,
        )

    from vibespatial.constructive.envelope import _build_device_boxes_from_bounds

    d_observed_codes = d_codes[d_observed_mask]
    d_observed_bounds = d_bounds[d_observed_mask]
    d_counts = cp.bincount(d_observed_codes, minlength=group_count)
    if not bool(cp.all(d_counts[:group_count] > 0)):
        return None
    if d_observed_codes.size > 1 and not bool(cp.all(d_observed_codes[1:] >= d_observed_codes[:-1])):
        return None

    d_xmin = cp.full(group_count, cp.inf, dtype=cp.float64)
    d_ymin = cp.full(group_count, cp.inf, dtype=cp.float64)
    d_xmax = cp.full(group_count, -cp.inf, dtype=cp.float64)
    d_ymax = cp.full(group_count, -cp.inf, dtype=cp.float64)
    cp.minimum.at(d_xmin, d_observed_codes, d_observed_bounds[:, 0])
    cp.minimum.at(d_ymin, d_observed_codes, d_observed_bounds[:, 1])
    cp.maximum.at(d_xmax, d_observed_codes, d_observed_bounds[:, 2])
    cp.maximum.at(d_ymax, d_observed_codes, d_observed_bounds[:, 3])

    d_width = d_observed_bounds[:, 2] - d_observed_bounds[:, 0]
    d_height = d_observed_bounds[:, 3] - d_observed_bounds[:, 1]
    d_area_sum = cp.zeros(group_count, dtype=cp.float64)
    cp.add.at(d_area_sum, d_observed_codes, d_width * d_height)
    d_bbox_area = (d_xmax - d_xmin) * (d_ymax - d_ymin)
    if not bool(cp.all(cp.isclose(d_area_sum, d_bbox_area, rtol=1.0e-12, atol=SPATIAL_EPSILON))):
        return None

    d_group_ymin = d_ymin[d_observed_codes]
    d_group_ymax = d_ymax[d_observed_codes]
    d_group_xmin = d_xmin[d_observed_codes]
    d_group_xmax = d_xmax[d_observed_codes]
    d_full_height = cp.isclose(
        d_observed_bounds[:, 1],
        d_group_ymin,
        rtol=1.0e-12,
        atol=SPATIAL_EPSILON,
    ) & cp.isclose(
        d_observed_bounds[:, 3],
        d_group_ymax,
        rtol=1.0e-12,
        atol=SPATIAL_EPSILON,
    )
    d_full_width = cp.isclose(
        d_observed_bounds[:, 0],
        d_group_xmin,
        rtol=1.0e-12,
        atol=SPATIAL_EPSILON,
    ) & cp.isclose(
        d_observed_bounds[:, 2],
        d_group_xmax,
        rtol=1.0e-12,
        atol=SPATIAL_EPSILON,
    )
    same_group = d_observed_codes[1:] == d_observed_codes[:-1]
    x_contiguous = bool(
        cp.all(
            (~same_group)
            | cp.isclose(
                d_observed_bounds[:-1, 2],
                d_observed_bounds[1:, 0],
                rtol=1.0e-12,
                atol=SPATIAL_EPSILON,
            )
        )
    )
    y_contiguous = bool(
        cp.all(
            (~same_group)
            | cp.isclose(
                d_observed_bounds[:-1, 3],
                d_observed_bounds[1:, 1],
                rtol=1.0e-12,
                atol=SPATIAL_EPSILON,
            )
        )
    )
    horizontal_strip = bool(cp.all(d_full_height)) and x_contiguous
    vertical_strip = bool(cp.all(d_full_width)) and y_contiguous
    if not (horizontal_strip or vertical_strip):
        return None

    d_group_bounds = cp.stack((d_xmin, d_ymin, d_xmax, d_ymax), axis=1)
    reduced = _build_device_boxes_from_bounds(d_group_bounds, row_count=group_count)

    non_empty_groups, empty_groups = _group_non_empty_counts(row_group_codes, group_count)
    return GroupedUnionResult(
        geometries=None,
        group_count=group_count,
        non_empty_groups=non_empty_groups,
        empty_groups=empty_groups,
        method=DissolveUnionMethod.COVERAGE,
        owned=reduced,
    )


def execute_grouped_box_union_gpu(
    values: np.ndarray,
    group_positions: list[np.ndarray],
) -> GroupedUnionResult | None:
    if cp is None:
        return None
    bounds = _rectangle_bounds(values)
    if bounds is None:
        return None
    if values.size == 0:
        return GroupedUnionResult(
            geometries=np.asarray([], dtype=object),
            group_count=0,
            non_empty_groups=0,
            empty_groups=0,
            method=DissolveUnionMethod.COVERAGE,
        )

    group_codes = np.full(values.size, -1, dtype=np.int32)
    for group_index, positions in enumerate(group_positions):
        if positions.size:
            group_codes[np.asarray(positions, dtype=np.int32)] = group_index
    observed_mask = group_codes >= 0
    if not np.any(observed_mask):
        merged = np.full(len(group_positions), _EMPTY, dtype=object)
        return GroupedUnionResult(
            geometries=merged,
            group_count=len(group_positions),
            non_empty_groups=0,
            empty_groups=len(group_positions),
            method=DissolveUnionMethod.COVERAGE,
        )

    # Sort on host — downstream reduceat ops are numpy-only, so a
    # H->D->sort->D->H round-trip through CuPy would waste bandwidth.
    observed_codes = group_codes[observed_mask].astype(np.int32, copy=False)
    observed_bounds = bounds[observed_mask].astype(np.float64, copy=False)
    sorted_rows = np.argsort(observed_codes, kind="stable")
    sorted_codes = observed_codes[sorted_rows]
    sorted_bounds = observed_bounds[sorted_rows]

    start_mask = np.empty((int(sorted_codes.size),), dtype=bool)
    start_mask[0] = True
    if int(sorted_codes.size) > 1:
        start_mask[1:] = sorted_codes[1:] != sorted_codes[:-1]
    starts = np.flatnonzero(start_mask).astype(np.int32, copy=False)
    unique_codes = sorted_codes[starts]
    xmin = np.minimum.reduceat(sorted_bounds[:, 0], starts)
    ymin = np.minimum.reduceat(sorted_bounds[:, 1], starts)
    xmax = np.maximum.reduceat(sorted_bounds[:, 2], starts)
    ymax = np.maximum.reduceat(sorted_bounds[:, 3], starts)
    merged = np.full(len(group_positions), _EMPTY, dtype=object)
    boxes = shapely.box(xmin, ymin, xmax, ymax)
    merged[unique_codes.astype(int)] = boxes
    non_empty_groups = int(np.count_nonzero([positions.size > 0 for positions in group_positions]))
    return GroupedUnionResult(
        geometries=merged,
        group_count=len(group_positions),
        non_empty_groups=non_empty_groups,
        empty_groups=len(group_positions) - non_empty_groups,
        method=DissolveUnionMethod.COVERAGE,
    )


def execute_grouped_box_union_gpu_codes(
    values: np.ndarray,
    row_group_codes: np.ndarray,
    *,
    group_count: int,
) -> GroupedUnionResult | None:
    if cp is None:
        return None
    bounds = _rectangle_bounds(values)
    if bounds is None:
        return None
    if values.size == 0:
        return GroupedUnionResult(
            geometries=np.asarray([], dtype=object),
            group_count=0,
            non_empty_groups=0,
            empty_groups=0,
            method=DissolveUnionMethod.COVERAGE,
        )

    observed_mask = row_group_codes >= 0
    if not np.any(observed_mask):
        merged = np.full(group_count, _EMPTY, dtype=object)
        return GroupedUnionResult(
            geometries=merged,
            group_count=group_count,
            non_empty_groups=0,
            empty_groups=group_count,
            method=DissolveUnionMethod.COVERAGE,
        )

    observed_codes = row_group_codes[observed_mask].astype(np.int32, copy=False)
    observed_bounds = bounds[observed_mask].astype(np.float64, copy=False)
    sorted_rows = np.argsort(observed_codes, kind="stable")
    sorted_codes = observed_codes[sorted_rows]
    sorted_bounds = observed_bounds[sorted_rows]
    offsets, unique_codes = _group_offsets_from_sorted_codes(sorted_codes)
    starts = offsets[:-1]
    xmin = np.minimum.reduceat(sorted_bounds[:, 0], starts)
    ymin = np.minimum.reduceat(sorted_bounds[:, 1], starts)
    xmax = np.maximum.reduceat(sorted_bounds[:, 2], starts)
    ymax = np.maximum.reduceat(sorted_bounds[:, 3], starts)
    merged = np.full(group_count, _EMPTY, dtype=object)
    merged[unique_codes.astype(np.intp, copy=False)] = shapely.box(xmin, ymin, xmax, ymax)
    non_empty_groups, empty_groups = _group_non_empty_counts(row_group_codes, group_count)
    return GroupedUnionResult(
        geometries=merged,
        group_count=group_count,
        non_empty_groups=non_empty_groups,
        empty_groups=empty_groups,
        method=DissolveUnionMethod.COVERAGE,
    )


def execute_grouped_coverage_union_gpu(
    values: Sequence[object | None] | np.ndarray | None,
    group_positions: list[np.ndarray],
    *,
    owned: OwnedGeometryArray | None = None,
) -> GroupedUnionResult | None:
    """Union coverage groups on GPU via owned coverage reduction per group.

    This replaces the grouped host-side ``coverage_union_all`` loop for
    polygon-family coverages that are not narrow enough for the rectangle
    bounds fast path.
    """
    if cp is None:
        return None

    if owned is None:
        try:
            owned = getattr(values, "_owned", None) if values is not None else None
        except Exception:
            owned = None
    if owned is None:
        return None
    if not _owned_supports_polygonal_grouped_union(owned):
        return None

    host_values: np.ndarray | None = None

    def _values() -> np.ndarray | None:
        nonlocal host_values
        if values is None:
            return None
        if host_values is None:
            host_values = np.asarray(values, dtype=object)
        return host_values

    from vibespatial.constructive.union_all import coverage_union_all_gpu_owned

    group_owned_results: list[OwnedGeometryArray] = []
    all_groups_observed = all(len(positions) > 0 for positions in group_positions)
    merged = None if all_groups_observed else np.empty(len(group_positions), dtype=object)
    non_empty_groups = 0
    empty_groups = 0

    for group_index, positions in enumerate(group_positions):
        positions = np.asarray(positions, dtype=np.intp)
        if positions.size == 0:
            if merged is not None:
                merged[group_index] = _EMPTY
            empty_groups += 1
            continue
        non_empty_groups += 1
        if positions.size == 1:
            single = owned.take(positions)
            group_owned_results.append(single)
            if merged is not None:
                materialized = _values()
                if materialized is not None:
                    merged[group_index] = materialized[int(positions[0])]
                else:
                    single_geoms = single.to_shapely()
                    merged[group_index] = single_geoms[0] if single_geoms else _EMPTY
            continue
        group_owned = owned.take(positions)
        reduced = coverage_union_all_gpu_owned(group_owned)
        if reduced is None:
            return None
        group_owned_results.append(reduced)
        if merged is not None:
            reduced_geoms = reduced.to_shapely()
            merged[group_index] = reduced_geoms[0] if reduced_geoms else _EMPTY

    if all_groups_observed and len(group_owned_results) == len(group_positions):
        return GroupedUnionResult(
            geometries=None,
            group_count=len(group_positions),
            non_empty_groups=non_empty_groups,
            empty_groups=empty_groups,
            method=DissolveUnionMethod.COVERAGE,
            owned=type(owned).concat(group_owned_results),
        )

    return GroupedUnionResult(
        geometries=merged if merged is not None else np.empty(0, dtype=object),
        group_count=len(group_positions),
        non_empty_groups=non_empty_groups,
        empty_groups=empty_groups,
        method=DissolveUnionMethod.COVERAGE,
    )


def execute_grouped_disjoint_subset_union_gpu(
    values: Sequence[object | None] | np.ndarray | None,
    group_positions: list[np.ndarray],
    *,
    owned: OwnedGeometryArray | None = None,
) -> GroupedUnionResult | None:
    """Union disjoint-subset groups on GPU via owned disjoint-subset reduction."""
    if cp is None:
        return None

    if owned is None:
        try:
            owned = getattr(values, "_owned", None) if values is not None else None
        except Exception:
            owned = None
    if owned is None:
        return None
    if not _owned_supports_polygonal_grouped_union(owned):
        return None

    host_values: np.ndarray | None = None

    def _values() -> np.ndarray | None:
        nonlocal host_values
        if values is None:
            return None
        if host_values is None:
            host_values = np.asarray(values, dtype=object)
        return host_values

    from vibespatial.constructive.union_all import disjoint_subset_union_all_owned

    group_owned_results: list[OwnedGeometryArray] = []
    all_groups_observed = all(len(positions) > 0 for positions in group_positions)
    merged = None if all_groups_observed else np.empty(len(group_positions), dtype=object)
    non_empty_groups = 0
    empty_groups = 0

    for group_index, positions in enumerate(group_positions):
        positions = np.asarray(positions, dtype=np.intp)
        if positions.size == 0:
            if merged is not None:
                merged[group_index] = _EMPTY
            empty_groups += 1
            continue
        non_empty_groups += 1
        if positions.size == 1:
            single = owned.take(positions)
            group_owned_results.append(single)
            if merged is not None:
                materialized = _values()
                if materialized is not None:
                    merged[group_index] = materialized[int(positions[0])]
                else:
                    single_geoms = single.to_shapely()
                    merged[group_index] = single_geoms[0] if single_geoms else _EMPTY
            continue
        group_owned = owned.take(positions)
        reduced = disjoint_subset_union_all_owned(group_owned)
        if reduced is None:
            return None
        group_owned_results.append(reduced)
        if merged is not None:
            reduced_geoms = reduced.to_shapely()
            merged[group_index] = reduced_geoms[0] if reduced_geoms else _EMPTY

    if all_groups_observed and len(group_owned_results) == len(group_positions):
        return GroupedUnionResult(
            geometries=None,
            group_count=len(group_positions),
            non_empty_groups=non_empty_groups,
            empty_groups=empty_groups,
            method=DissolveUnionMethod.DISJOINT_SUBSET,
            owned=type(owned).concat(group_owned_results),
        )

    return GroupedUnionResult(
        geometries=merged if merged is not None else np.empty(0, dtype=object),
        group_count=len(group_positions),
        non_empty_groups=non_empty_groups,
        empty_groups=empty_groups,
        method=DissolveUnionMethod.DISJOINT_SUBSET,
    )


def execute_grouped_coverage_edge_union(
    values: np.ndarray,
    group_positions: list[np.ndarray],
) -> GroupedUnionResult | None:
    """Topology-free grouped coverage dissolve via shared-edge elimination.

    This implements the essential `#13` shape for polygonal coverages:
    eliminate shared interior edges inside each dissolve group, then
    reconstruct polygon boundaries from the surviving edges.
    """
    if values.size == 0:
        return GroupedUnionResult(
            geometries=np.asarray([], dtype=object),
            group_count=0,
            non_empty_groups=0,
            empty_groups=0,
            method=DissolveUnionMethod.COVERAGE,
        )

    group_codes = np.full(values.size, -1, dtype=np.int32)
    for group_index, positions in enumerate(group_positions):
        if len(positions):
            group_codes[np.asarray(positions, dtype=np.int32)] = group_index

    valid_mask = np.asarray(
        [geom is not None and not shapely.is_empty(geom) for geom in values],
        dtype=bool,
    )
    if not np.any(valid_mask):
        merged = np.full(len(group_positions), _EMPTY, dtype=object)
        return GroupedUnionResult(
            geometries=merged,
            group_count=len(group_positions),
            non_empty_groups=0,
            empty_groups=len(group_positions),
            method=DissolveUnionMethod.COVERAGE,
        )

    valid_values = values[valid_mask]
    valid_rows = np.flatnonzero(valid_mask).astype(np.int32, copy=False)
    parts, geom_part_index = shapely.get_parts(valid_values, return_index=True)
    if parts.size == 0:
        return None
    part_type_ids = shapely.get_type_id(parts)
    if not np.all(part_type_ids == 3):  # Polygon
        return None

    rings, ring_part_index = shapely.get_rings(parts, return_index=True)
    if rings.size == 0:
        return None

    ring_lengths = np.asarray(shapely.get_num_coordinates(rings), dtype=np.int64)
    edge_counts = ring_lengths - 1
    total_edges = int(edge_counts.sum())
    if total_edges <= 0:
        return None

    ring_coords = np.asarray(shapely.get_coordinates(rings), dtype=np.float64)
    ring_offsets = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum(ring_lengths[:-1], dtype=np.int64)]
    )
    edge_offsets = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum(edge_counts[:-1], dtype=np.int64)]
    )
    ring_row_index = valid_rows[geom_part_index[ring_part_index]]
    edge_group_codes = np.repeat(group_codes[ring_row_index], edge_counts)

    base = np.arange(total_edges, dtype=np.int64)
    repeated_edge_offsets = np.repeat(edge_offsets, edge_counts)
    repeated_ring_offsets = np.repeat(ring_offsets, edge_counts)
    start_idx = repeated_ring_offsets + (base - repeated_edge_offsets)
    end_idx = start_idx + 1

    start_xy = ring_coords[start_idx]
    end_xy = ring_coords[end_idx]
    start_x_bits = start_xy[:, 0].view(np.uint64)
    start_y_bits = start_xy[:, 1].view(np.uint64)
    end_x_bits = end_xy[:, 0].view(np.uint64)
    end_y_bits = end_xy[:, 1].view(np.uint64)

    swap = (
        (start_x_bits > end_x_bits)
        | ((start_x_bits == end_x_bits) & (start_y_bits > end_y_bits))
    )
    key_x0 = np.where(swap, end_x_bits, start_x_bits)
    key_y0 = np.where(swap, end_y_bits, start_y_bits)
    key_x1 = np.where(swap, start_x_bits, end_x_bits)
    key_y1 = np.where(swap, start_y_bits, end_y_bits)

    order = np.lexsort((key_y1, key_x1, key_y0, key_x0, edge_group_codes))
    sorted_groups = edge_group_codes[order]
    sorted_x0 = key_x0[order]
    sorted_y0 = key_y0[order]
    sorted_x1 = key_x1[order]
    sorted_y1 = key_y1[order]

    start_mask = np.empty(order.size, dtype=bool)
    start_mask[0] = True
    start_mask[1:] = (
        (sorted_groups[1:] != sorted_groups[:-1])
        | (sorted_x0[1:] != sorted_x0[:-1])
        | (sorted_y0[1:] != sorted_y0[:-1])
        | (sorted_x1[1:] != sorted_x1[:-1])
        | (sorted_y1[1:] != sorted_y1[:-1])
    )
    run_starts = np.flatnonzero(start_mask).astype(np.int64, copy=False)
    run_lengths = np.diff(
        np.concatenate([run_starts, np.asarray([order.size], dtype=np.int64)])
    )
    boundary_orders = order[run_starts[run_lengths % 2 == 1]]

    if boundary_orders.size == 0:
        merged = np.full(len(group_positions), _EMPTY, dtype=object)
        return GroupedUnionResult(
            geometries=merged,
            group_count=len(group_positions),
            non_empty_groups=0,
            empty_groups=len(group_positions),
            method=DissolveUnionMethod.COVERAGE,
        )

    boundary_groups = edge_group_codes[boundary_orders]
    boundary_start = start_xy[boundary_orders]
    boundary_end = end_xy[boundary_orders]
    flat_coords = np.empty((boundary_orders.size * 2, 2), dtype=np.float64)
    flat_coords[0::2] = boundary_start
    flat_coords[1::2] = boundary_end
    line_indices = np.repeat(np.arange(boundary_orders.size, dtype=np.int32), 2)
    boundary_lines = shapely.linestrings(flat_coords, indices=line_indices)

    observed_boundary_groups = np.unique(boundary_groups).astype(np.int32, copy=False)
    group_inverse = np.searchsorted(observed_boundary_groups, boundary_groups).astype(np.int32, copy=False)
    grouped_lines = shapely.multilinestrings(boundary_lines, indices=group_inverse)
    grouped_areas = shapely.build_area(grouped_lines)

    merged = np.full(len(group_positions), _EMPTY, dtype=object)
    merged[observed_boundary_groups.astype(np.intp, copy=False)] = np.asarray(grouped_areas, dtype=object)
    non_empty_groups = int(np.count_nonzero([len(positions) > 0 for positions in group_positions]))
    empty_groups = len(group_positions) - non_empty_groups

    return GroupedUnionResult(
        geometries=merged,
        group_count=len(group_positions),
        non_empty_groups=non_empty_groups,
        empty_groups=empty_groups,
        method=DissolveUnionMethod.COVERAGE,
    )


def execute_grouped_coverage_edge_union_codes(
    values: np.ndarray,
    row_group_codes: np.ndarray,
    *,
    group_count: int,
) -> GroupedUnionResult | None:
    if values.size == 0:
        return GroupedUnionResult(
            geometries=np.asarray([], dtype=object),
            group_count=0,
            non_empty_groups=0,
            empty_groups=0,
            method=DissolveUnionMethod.COVERAGE,
        )

    valid_mask = np.asarray(
        [geom is not None and not shapely.is_empty(geom) for geom in values],
        dtype=bool,
    )
    if not np.any(valid_mask):
        merged = np.full(group_count, _EMPTY, dtype=object)
        return GroupedUnionResult(
            geometries=merged,
            group_count=group_count,
            non_empty_groups=0,
            empty_groups=group_count,
            method=DissolveUnionMethod.COVERAGE,
        )

    valid_values = values[valid_mask]
    valid_group_codes = row_group_codes[valid_mask].astype(np.int32, copy=False)
    valid_rows = np.flatnonzero(valid_mask).astype(np.int32, copy=False)
    parts, geom_part_index = shapely.get_parts(valid_values, return_index=True)
    if parts.size == 0:
        return None
    part_type_ids = shapely.get_type_id(parts)
    if not np.all(part_type_ids == 3):
        return None

    rings, ring_part_index = shapely.get_rings(parts, return_index=True)
    if rings.size == 0:
        return None

    ring_lengths = np.asarray(shapely.get_num_coordinates(rings), dtype=np.int64)
    edge_counts = ring_lengths - 1
    total_edges = int(edge_counts.sum())
    if total_edges <= 0:
        return None

    ring_coords = np.asarray(shapely.get_coordinates(rings), dtype=np.float64)
    ring_offsets = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum(ring_lengths[:-1], dtype=np.int64)]
    )
    edge_offsets = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum(edge_counts[:-1], dtype=np.int64)]
    )
    ring_row_index = valid_rows[geom_part_index[ring_part_index]]
    edge_group_codes = np.repeat(valid_group_codes[ring_row_index], edge_counts)

    base = np.arange(total_edges, dtype=np.int64)
    repeated_edge_offsets = np.repeat(edge_offsets, edge_counts)
    repeated_ring_offsets = np.repeat(ring_offsets, edge_counts)
    start_idx = repeated_ring_offsets + (base - repeated_edge_offsets)
    end_idx = start_idx + 1

    start_xy = ring_coords[start_idx]
    end_xy = ring_coords[end_idx]
    start_x_bits = start_xy[:, 0].view(np.uint64)
    start_y_bits = start_xy[:, 1].view(np.uint64)
    end_x_bits = end_xy[:, 0].view(np.uint64)
    end_y_bits = end_xy[:, 1].view(np.uint64)

    swap = (
        (start_x_bits > end_x_bits)
        | ((start_x_bits == end_x_bits) & (start_y_bits > end_y_bits))
    )
    key_x0 = np.where(swap, end_x_bits, start_x_bits)
    key_y0 = np.where(swap, end_y_bits, start_y_bits)
    key_x1 = np.where(swap, start_x_bits, end_x_bits)
    key_y1 = np.where(swap, start_y_bits, end_y_bits)

    order = np.lexsort((key_y1, key_x1, key_y0, key_x0, edge_group_codes))
    sorted_groups = edge_group_codes[order]
    sorted_x0 = key_x0[order]
    sorted_y0 = key_y0[order]
    sorted_x1 = key_x1[order]
    sorted_y1 = key_y1[order]

    start_mask = np.empty(order.size, dtype=bool)
    start_mask[0] = True
    start_mask[1:] = (
        (sorted_groups[1:] != sorted_groups[:-1])
        | (sorted_x0[1:] != sorted_x0[:-1])
        | (sorted_y0[1:] != sorted_y0[:-1])
        | (sorted_x1[1:] != sorted_x1[:-1])
        | (sorted_y1[1:] != sorted_y1[:-1])
    )
    run_starts = np.flatnonzero(start_mask).astype(np.int64, copy=False)
    run_lengths = np.diff(
        np.concatenate([run_starts, np.asarray([order.size], dtype=np.int64)])
    )
    boundary_orders = order[run_starts[run_lengths % 2 == 1]]

    if boundary_orders.size == 0:
        merged = np.full(group_count, _EMPTY, dtype=object)
        return GroupedUnionResult(
            geometries=merged,
            group_count=group_count,
            non_empty_groups=0,
            empty_groups=group_count,
            method=DissolveUnionMethod.COVERAGE,
        )

    boundary_groups = edge_group_codes[boundary_orders]
    boundary_start = start_xy[boundary_orders]
    boundary_end = end_xy[boundary_orders]
    flat_coords = np.empty((boundary_orders.size * 2, 2), dtype=np.float64)
    flat_coords[0::2] = boundary_start
    flat_coords[1::2] = boundary_end
    line_indices = np.repeat(np.arange(boundary_orders.size, dtype=np.int32), 2)
    boundary_lines = shapely.linestrings(flat_coords, indices=line_indices)

    observed_boundary_groups = np.unique(boundary_groups).astype(np.int32, copy=False)
    group_inverse = np.searchsorted(observed_boundary_groups, boundary_groups).astype(np.int32, copy=False)
    grouped_lines = shapely.multilinestrings(boundary_lines, indices=group_inverse)
    grouped_areas = shapely.build_area(grouped_lines)

    merged = np.full(group_count, _EMPTY, dtype=object)
    merged[observed_boundary_groups.astype(np.intp, copy=False)] = np.asarray(grouped_areas, dtype=object)
    non_empty_groups, empty_groups = _group_non_empty_counts(row_group_codes, group_count)
    return GroupedUnionResult(
        geometries=merged,
        group_count=group_count,
        non_empty_groups=non_empty_groups,
        empty_groups=empty_groups,
        method=DissolveUnionMethod.COVERAGE,
    )


def execute_grouped_disjoint_subset_union_codes(
    values: np.ndarray,
    row_group_codes: np.ndarray,
    *,
    group_count: int,
) -> GroupedUnionResult | None:
    observed_mask = row_group_codes >= 0
    merged = np.full(group_count, _EMPTY, dtype=object)
    if not np.any(observed_mask):
        return GroupedUnionResult(
            geometries=merged,
            group_count=group_count,
            non_empty_groups=0,
            empty_groups=group_count,
            method=DissolveUnionMethod.DISJOINT_SUBSET,
        )

    observed_codes = row_group_codes[observed_mask].astype(np.int32, copy=False)
    observed_values = values[observed_mask]
    order = np.argsort(observed_codes, kind="stable")
    sorted_codes = observed_codes[order]
    sorted_values = observed_values[order]
    offsets, unique_codes = _group_offsets_from_sorted_codes(sorted_codes)

    for group_index, start, stop in zip(
        unique_codes.astype(np.intp, copy=False),
        offsets[:-1],
        offsets[1:],
        strict=True,
    ):
        merged[int(group_index)] = shapely.disjoint_subset_union_all(sorted_values[int(start):int(stop)])

    non_empty_groups, empty_groups = _group_non_empty_counts(row_group_codes, group_count)
    return GroupedUnionResult(
        geometries=merged,
        group_count=group_count,
        non_empty_groups=non_empty_groups,
        empty_groups=empty_groups,
        method=DissolveUnionMethod.DISJOINT_SUBSET,
    )


def _gpu_union_group(group_geoms: np.ndarray) -> object:
    """Union a group of polygon geometries on GPU via tree-reduce (ADR-0017).

    Uses overlay_union_owned in log₂(n) rounds, processing all pairs per round.
    Falls back to shapely for non-polygon or degenerate inputs.
    """
    from vibespatial.geometry.owned import from_shapely_geometries
    from vibespatial.runtime import ExecutionMode

    from .gpu import overlay_union_owned

    if group_geoms.size == 0:
        return _EMPTY
    if group_geoms.size == 1:
        return group_geoms[0]

    # Filter valid, non-empty polygon geometries
    valid = [g for g in group_geoms if g is not None and not shapely.is_empty(g)]
    if len(valid) == 0:
        return _EMPTY
    if len(valid) == 1:
        return valid[0]

    # Check all are polygon-family
    type_ids = shapely.get_type_id(np.asarray(valid, dtype=object))
    polygon_family = np.isin(type_ids, [3, 6])  # Polygon=3, MultiPolygon=6
    if not np.all(polygon_family):
        return shapely.union_all(np.asarray(valid, dtype=object))

    # Tree-reduce: union pairs, then union the results
    current = list(valid)
    while len(current) > 1:
        next_round: list[object] = []
        for i in range(0, len(current), 2):
            if i + 1 < len(current):
                left_owned = from_shapely_geometries([current[i]])
                right_owned = from_shapely_geometries([current[i + 1]])
                result = overlay_union_owned(left_owned, right_owned, dispatch_mode=ExecutionMode.GPU)
                result_geoms = result.to_shapely()
                if result_geoms:
                    next_round.append(result_geoms[0])
                else:
                    next_round.append(_EMPTY)
            else:
                next_round.append(current[i])
        current = next_round
    return current[0]


def execute_grouped_union_codes(
    geometries: Sequence[object | None] | np.ndarray,
    row_group_codes: np.ndarray,
    *,
    group_count: int,
    method: DissolveUnionMethod | str = DissolveUnionMethod.UNARY,
    grid_size: float | None = None,
    owned: OwnedGeometryArray | None = None,
) -> GroupedUnionResult | None:
    normalized = method if isinstance(method, DissolveUnionMethod) else DissolveUnionMethod(method)
    values: np.ndarray | None = None

    def _values() -> np.ndarray:
        nonlocal values
        if values is None:
            values = np.asarray(geometries, dtype=object)
        return values

    geometry_count = int(owned.row_count) if owned is not None else int(len(geometries))
    if geometry_count != row_group_codes.size:
        raise ValueError("row_group_codes length must match geometries length")

    owned_supports_segmented_union = False
    if owned is not None:
        from vibespatial.geometry.buffers import GeometryFamily
        from vibespatial.geometry.owned import FAMILY_TAGS

        valid_tags = owned.tags[owned.validity]
        polygon_tags = np.asarray(
            [
                FAMILY_TAGS[GeometryFamily.POLYGON],
                FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
            ],
            dtype=valid_tags.dtype if valid_tags.size else np.int8,
        )
        owned_supports_segmented_union = bool(
            valid_tags.size == 0 or np.all(np.isin(valid_tags, polygon_tags))
        )

    if (
        owned is not None
        and owned_supports_segmented_union
        and normalized is DissolveUnionMethod.UNARY
        and grid_size is None
    ):
        observed_mask = row_group_codes >= 0
        if not np.any(observed_mask):
            merged = np.full(group_count, _EMPTY, dtype=object)
            return GroupedUnionResult(
                geometries=merged,
                group_count=group_count,
                non_empty_groups=0,
                empty_groups=group_count,
                method=normalized,
            )
        from vibespatial.kernels.constructive.segmented_union import segmented_union_all

        observed_codes = row_group_codes[observed_mask].astype(np.int32, copy=False)
        observed_rows = np.flatnonzero(observed_mask).astype(np.int64, copy=False)
        order = np.argsort(observed_codes, kind="stable")
        sorted_codes = observed_codes[order]
        sorted_rows = observed_rows[order]
        offsets, unique_codes = _group_offsets_from_sorted_codes(sorted_codes)
        ordered_owned = owned.take(sorted_rows)
        if group_count == 1:
            from vibespatial.constructive.union_all import (
                _spatially_localize_polygon_union_inputs,
            )

            ordered_owned = _spatially_localize_polygon_union_inputs(ordered_owned)
        reduced = segmented_union_all(ordered_owned, offsets)
        reduced = _recompute_invalid_grouped_union_owned_rows(
            reduced,
            ordered_owned=ordered_owned,
            offsets=offsets,
            group_count=int(unique_codes.size),
        )
        if unique_codes.size == group_count and np.array_equal(
            unique_codes,
            np.arange(group_count, dtype=unique_codes.dtype),
        ):
            non_empty_groups, empty_groups = _group_non_empty_counts(row_group_codes, group_count)
            return GroupedUnionResult(
                geometries=None,
                group_count=group_count,
                non_empty_groups=non_empty_groups,
                empty_groups=empty_groups,
                method=normalized,
                owned=reduced,
            )
        reduced_geometries = np.asarray(reduced.to_shapely(), dtype=object)
        merged = np.full(group_count, _EMPTY, dtype=object)
        merged[unique_codes.astype(np.intp, copy=False)] = reduced_geometries
        non_empty_groups, empty_groups = _group_non_empty_counts(row_group_codes, group_count)
        return GroupedUnionResult(
            geometries=merged,
            group_count=group_count,
            non_empty_groups=non_empty_groups,
            empty_groups=empty_groups,
            method=normalized,
        )

    if (
        owned is not None
        and normalized is DissolveUnionMethod.COVERAGE
        and grid_size is None
    ):
        accelerated = execute_grouped_box_union_gpu_owned_codes(
            row_group_codes,
            group_count=group_count,
            owned=owned,
        )
        if accelerated is not None:
            return accelerated

    if (
        normalized is DissolveUnionMethod.COVERAGE
        and geometry_count >= OVERLAY_GROUPED_BOX_GPU_THRESHOLD
    ):
        accelerated = execute_grouped_box_union_gpu_codes(
            _values(),
            row_group_codes,
            group_count=group_count,
        )
        if accelerated is not None:
            return accelerated
    if (
        normalized is DissolveUnionMethod.COVERAGE
        and geometry_count >= OVERLAY_GROUPED_COVERAGE_EDGE_THRESHOLD
    ):
        accelerated = execute_grouped_coverage_edge_union_codes(
            _values(),
            row_group_codes,
            group_count=group_count,
        )
        if accelerated is not None:
            return accelerated
    if (
        normalized is DissolveUnionMethod.COVERAGE
        and geometry_count >= OVERLAY_UNION_ALL_GPU_THRESHOLD
    ):
        accelerated = execute_grouped_disjoint_subset_union_codes(
            _values(),
            row_group_codes,
            group_count=group_count,
        )
        if accelerated is not None:
            return accelerated
    if (
        normalized is DissolveUnionMethod.COVERAGE
    ):
        if owned is not None:
            accelerated = execute_grouped_coverage_union_gpu(
                None,
                _group_positions_from_codes(row_group_codes, group_count),
                owned=owned,
            )
            if accelerated is not None:
                return accelerated
    if normalized is DissolveUnionMethod.DISJOINT_SUBSET:
        if owned is not None:
            accelerated = execute_grouped_disjoint_subset_union_gpu(
                None,
                _group_positions_from_codes(row_group_codes, group_count),
                owned=owned,
            )
            if accelerated is not None:
                return accelerated
        exact = execute_grouped_disjoint_subset_union_codes(
            _values(),
            row_group_codes,
            group_count=group_count,
        )
        if owned is not None and exact.geometries is not None:
            from vibespatial.geometry.owned import from_shapely_geometries

            exact = GroupedUnionResult(
                geometries=exact.geometries,
                group_count=exact.group_count,
                non_empty_groups=exact.non_empty_groups,
                empty_groups=exact.empty_groups,
                method=exact.method,
                owned=from_shapely_geometries(
                    exact.geometries.tolist(),
                    residency=owned.residency,
                ),
            )
        return exact
    return None


def execute_grouped_union(
    geometries: Sequence[object | None] | np.ndarray,
    group_positions: list[np.ndarray],
    *,
    method: DissolveUnionMethod | str = DissolveUnionMethod.UNARY,
    grid_size: float | None = None,
    owned: OwnedGeometryArray | None = None,
) -> GroupedUnionResult:
    normalized = method if isinstance(method, DissolveUnionMethod) else DissolveUnionMethod(method)
    if normalized is DissolveUnionMethod.COVERAGE and owned is not None:
        accelerated = execute_grouped_coverage_union_gpu(
            None,
            group_positions,
            owned=owned,
        )
        if accelerated is not None:
            return accelerated
    if normalized is DissolveUnionMethod.DISJOINT_SUBSET and owned is not None:
        accelerated = execute_grouped_disjoint_subset_union_gpu(
            None,
            group_positions,
            owned=owned,
        )
        if accelerated is not None:
            return accelerated
    values = np.asarray(geometries, dtype=object)
    # GPU box union fast path
    if (
        normalized is DissolveUnionMethod.COVERAGE
        and int(values.size) >= OVERLAY_GROUPED_BOX_GPU_THRESHOLD
    ):
        accelerated = execute_grouped_box_union_gpu(values, group_positions)
        if accelerated is not None:
            return accelerated
    if (
        normalized is DissolveUnionMethod.COVERAGE
        and int(values.size) >= OVERLAY_GROUPED_COVERAGE_EDGE_THRESHOLD
    ):
        accelerated = execute_grouped_coverage_edge_union(
            values,
            group_positions,
        )
        if accelerated is not None:
            return accelerated
        accelerated = execute_grouped_coverage_union_gpu(
            values,
            group_positions,
            owned=owned,
        )
        if accelerated is not None:
            return accelerated
    # GPU polygon union via tree-reduce (ADR-0017)
    # Only use for large groups where overlay JIT overhead is amortized.
    # Per-pair overlay_union has ~200ms JIT startup; only worthwhile when
    # group has enough polygons that GPU parallelism outweighs the overhead.
    _GPU_UNION_MIN_GROUP_SIZE = 100
    use_gpu_union = (
        cp is not None
        and int(values.size) >= OVERLAY_GROUPED_BOX_GPU_THRESHOLD
        and grid_size is None
    )
    merged = np.empty(len(group_positions), dtype=object)
    non_empty_groups = 0
    empty_groups = 0
    for group_index, positions in enumerate(group_positions):
        block = values[positions]
        if use_gpu_union and block.size >= _GPU_UNION_MIN_GROUP_SIZE:
            merged[group_index] = _gpu_union_group(block)
        else:
            merged[group_index] = _union_block(block, normalized, grid_size)
        if positions.size:
            non_empty_groups += 1
        else:
            empty_groups += 1
    return GroupedUnionResult(
        geometries=merged,
        group_count=len(group_positions),
        non_empty_groups=non_empty_groups,
        empty_groups=empty_groups,
        method=normalized,
    )


def _prepare_grouped_dissolve(
    frame,
    *,
    by,
    aggfunc,
    level,
    sort: bool,
    observed: bool,
    dropna: bool,
    normalized_method: DissolveUnionMethod,
    agg_kwargs: dict[str, Any],
) -> tuple[pd.DataFrame, list[np.ndarray], OwnedGeometryArray | None, np.ndarray | None]:
    if by is None and level is None:
        by = np.zeros(len(frame), dtype="int64")

    groupby_kwargs = {
        "by": by,
        "level": level,
        "sort": sort,
        "observed": observed,
        "dropna": dropna,
    }

    data = frame.drop(labels=frame.geometry.name, axis=1)
    aggregated_data = data.groupby(**groupby_kwargs).agg(aggfunc, **agg_kwargs)
    aggregated_data.columns = aggregated_data.columns.to_flat_index()

    row_group_codes = _build_row_group_codes(
        frame,
        by=by,
        level=level,
        aggregated_index=aggregated_data.index,
    )
    if row_group_codes is None:
        grouped_geometry = frame.groupby(group_keys=False, **groupby_kwargs)[frame.geometry.name]
        indices_items = list(grouped_geometry.indices.items())
        group_positions = _normalize_group_positions(aggregated_data.index, indices_items)
    else:
        group_positions = []
    owned = getattr(frame.geometry.values, "_owned", None)
    if (
        owned is None
        and normalized_method is DissolveUnionMethod.COVERAGE
        and int(len(frame)) >= OVERLAY_UNION_ALL_GPU_THRESHOLD
    ):
        try:
            owned = frame.geometry.values.to_owned()
        except (AttributeError, NotImplementedError):
            owned = None
    return aggregated_data, group_positions, owned, row_group_codes


def _max_group_size(
    row_group_codes: np.ndarray | None,
    group_positions: list[np.ndarray],
    group_count: int,
) -> int:
    if group_count == 0:
        return 0
    if row_group_codes is not None:
        observed = row_group_codes[row_group_codes >= 0]
        if observed.size == 0:
            return 0
        counts = np.bincount(observed.astype(np.int32, copy=False), minlength=group_count)
        return int(counts.max(initial=0))
    if not group_positions:
        return 0
    return max((int(len(positions)) for positions in group_positions), default=0)


def _provenance_source_owned(tag) -> OwnedGeometryArray | None:
    if tag is None:
        return None

    source = getattr(tag, "source_array", None)
    if source is None:
        return None

    owned = getattr(source, "_owned", None)
    if owned is not None:
        return owned

    to_owned = getattr(source, "to_owned", None)
    if callable(to_owned):
        try:
            return to_owned()
        except Exception:
            return None

    values = getattr(source, "values", None)
    return getattr(values, "_owned", None)


def _dedupe_two_point_linestring_rows_gpu(
    lines: OwnedGeometryArray,
) -> np.ndarray | None:
    from vibespatial.geometry.buffers import GeometryFamily
    if GeometryFamily.LINESTRING not in lines.families or len(lines.families) != 1:
        return None

    line_buffer = lines.families[GeometryFamily.LINESTRING]
    if line_buffer.host_materialized:
        offsets = np.asarray(line_buffer.geometry_offsets, dtype=np.int32)
        if offsets.shape != (lines.row_count + 1,) or not np.all((offsets[1:] - offsets[:-1]) == 2):
            return None

        coord_starts = offsets[:-1]
        x = np.asarray(line_buffer.x, dtype=np.float64)
        y = np.asarray(line_buffer.y, dtype=np.float64)
        x0 = x[coord_starts]
        y0 = y[coord_starts]
        x1 = x[coord_starts + 1]
        y1 = y[coord_starts + 1]
        swap = (x0 > x1) | ((x0 == x1) & (y0 > y1))
        ax = np.where(swap, x1, x0)
        ay = np.where(swap, y1, y0)
        bx = np.where(swap, x0, x1)
        by = np.where(swap, y0, y1)
        order = np.lexsort((by, bx, ay, ax))
        sorted_ax = ax[order]
        sorted_ay = ay[order]
        sorted_bx = bx[order]
        sorted_by = by[order]
        unique_mask = np.empty(order.size, dtype=bool)
        unique_mask[0] = True
        unique_mask[1:] = (
            (sorted_ax[1:] != sorted_ax[:-1])
            | (sorted_ay[1:] != sorted_ay[:-1])
            | (sorted_bx[1:] != sorted_bx[:-1])
            | (sorted_by[1:] != sorted_by[:-1])
        )
        return np.sort(order[unique_mask]).astype(np.int64, copy=False)

    if cp is not None:
        try:
            from vibespatial.runtime.residency import Residency, TransferTrigger

            lines.move_to(
                Residency.DEVICE,
                trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                reason="buffered-line dissolve rewrite needs device-resident line endpoints for duplicate elimination",
            )
            state = lines._ensure_device_state()
            line_buf = state.families[GeometryFamily.LINESTRING]
            offsets = cp.asarray(line_buf.geometry_offsets)
            if int(offsets.size) == lines.row_count + 1 and bool(cp.all((offsets[1:] - offsets[:-1]) == 2).item()):
                coord_starts = offsets[:-1]
                d_x = cp.asarray(line_buf.x)
                d_y = cp.asarray(line_buf.y)
                x0 = d_x[coord_starts]
                y0 = d_y[coord_starts]
                x1 = d_x[coord_starts + 1]
                y1 = d_y[coord_starts + 1]

                swap = (x0 > x1) | ((x0 == x1) & (y0 > y1))
                ax = cp.where(swap, x1, x0)
                ay = cp.where(swap, y1, y0)
                bx = cp.where(swap, x0, x1)
                by = cp.where(swap, y0, y1)
                order = cp.lexsort((by, bx, ay, ax)).astype(cp.int64, copy=False)
                if int(order.size) > 0:
                    sorted_ax = ax[order]
                    sorted_ay = ay[order]
                    sorted_bx = bx[order]
                    sorted_by = by[order]
                    unique_mask = cp.empty(order.size, dtype=cp.bool_)
                    unique_mask[0] = True
                    if int(order.size) > 1:
                        unique_mask[1:] = (
                            (sorted_ax[1:] != sorted_ax[:-1])
                            | (sorted_ay[1:] != sorted_ay[:-1])
                            | (sorted_bx[1:] != sorted_bx[:-1])
                            | (sorted_by[1:] != sorted_by[:-1])
                        )
                    unique_rows = cp.sort(order[unique_mask]).astype(cp.int64, copy=False)
                    return cp.asnumpy(unique_rows)
        except Exception:
            pass

    lines._ensure_host_state()
    line_buf = lines.families[GeometryFamily.LINESTRING]
    offsets = np.asarray(line_buf.geometry_offsets, dtype=np.int32)
    if offsets.shape != (lines.row_count + 1,) or not np.all((offsets[1:] - offsets[:-1]) == 2):
        return None

    coord_starts = offsets[:-1]
    x = np.asarray(line_buf.x, dtype=np.float64)
    y = np.asarray(line_buf.y, dtype=np.float64)
    x0 = x[coord_starts]
    y0 = y[coord_starts]
    x1 = x[coord_starts + 1]
    y1 = y[coord_starts + 1]
    swap = (x0 > x1) | ((x0 == x1) & (y0 > y1))
    ax = np.where(swap, x1, x0)
    ay = np.where(swap, y1, y0)
    bx = np.where(swap, x0, x1)
    by = np.where(swap, y0, y1)
    order = np.lexsort((by, bx, ay, ax))
    sorted_ax = ax[order]
    sorted_ay = ay[order]
    sorted_bx = bx[order]
    sorted_by = by[order]
    unique_mask = np.empty(order.size, dtype=bool)
    unique_mask[0] = True
    unique_mask[1:] = (
        (sorted_ax[1:] != sorted_ax[:-1])
        | (sorted_ay[1:] != sorted_ay[:-1])
        | (sorted_bx[1:] != sorted_bx[:-1])
        | (sorted_by[1:] != sorted_by[:-1])
    )
    return np.sort(order[unique_mask]).astype(np.int64, copy=False)


def _maybe_execute_buffered_two_point_line_exact_union_rewrite(
    *,
    normalized_method: DissolveUnionMethod,
    grid_size: float | None,
    row_group_codes: np.ndarray | None,
    group_count: int,
    tag,
) -> GroupedUnionResult | None:
    if (
        normalized_method is not DissolveUnionMethod.UNARY
        or grid_size is not None
        or row_group_codes is None
        or group_count != 1
        or not provenance_rewrites_enabled()
        or cp is None
    ):
        return None

    if tag is None or tag.operation != "buffer":
        return None

    source_types = tag.source_geom_types
    if not source_types or source_types != frozenset({"linestring"}):
        return None

    try:
        distance_value = float(tag.get_param("distance", 0.0))
    except (TypeError, ValueError):
        return None
    if distance_value <= 0.0 or bool(tag.get_param("single_sided", False)):
        return None

    observed_rows = np.flatnonzero(row_group_codes >= 0).astype(np.int64, copy=False)
    if observed_rows.size < OVERLAY_UNION_ALL_GPU_THRESHOLD:
        return None
    if observed_rows.size <= _BUFFERED_LINE_EXACT_CPU_MAX_ROWS:
        return None

    source_owned = _provenance_source_owned(tag)
    if source_owned is None:
        return None
    if observed_rows.size != source_owned.row_count:
        source_owned = source_owned.take(observed_rows)

    from vibespatial.constructive.linestring import (
        linestring_buffer_owned_array,
        supports_two_point_linestring_buffer_fast_path,
    )
    from vibespatial.constructive.union_all import (
        disjoint_subset_union_all_owned,
        union_all_gpu_owned,
    )
    from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device

    quad_segs_param = tag.get_param("quad_segs", 16)
    quad_segs = 16 if quad_segs_param is None else int(quad_segs_param)
    cap_style_param = tag.get_param("cap_style", "round")
    join_style_param = tag.get_param("join_style", "round")
    cap_style = "round" if cap_style_param is None else str(cap_style_param)
    join_style = "round" if join_style_param is None else str(join_style_param)
    if not supports_two_point_linestring_buffer_fast_path(
        source_owned,
        quad_segs=quad_segs,
        cap_style=cap_style,
        join_style=join_style,
        single_sided=False,
    ):
        return None

    unique_rows = _dedupe_two_point_linestring_rows_gpu(source_owned)
    if unique_rows is None or unique_rows.size == 0 or unique_rows.size >= source_owned.row_count:
        return None
    # Keep the exact rewrite for materially deduped groups while capping the
    # reduced exact-union size. Corridor-style networks often collapse far
    # below the original row count but still land just above the generic
    # grouped-union crossover.
    if unique_rows.size > _BUFFERED_TWO_POINT_EXACT_UNION_MAX_UNIQUE_ROWS:
        return None
    # This rewrite only pays off when deduplication is material; otherwise the
    # exact GPU union can lose to the disjoint-subset host engine on lightly
    # duplicated groups.
    if unique_rows.size * 4 > source_owned.row_count:
        return None

    unique_owned = source_owned.take(unique_rows.astype(np.int64, copy=False))
    expanded_bounds = np.asarray(
        cp.asnumpy(cp.asarray(compute_geometry_bounds_device(unique_owned), dtype=cp.float64)),
        dtype=np.float64,
    )
    if int(expanded_bounds.shape[0]) != unique_owned.row_count:
        return None
    expanded_bounds[:, 0] -= distance_value
    expanded_bounds[:, 1] -= distance_value
    expanded_bounds[:, 2] += distance_value
    expanded_bounds[:, 3] += distance_value

    colors = _greedy_bbox_disjoint_coloring(expanded_bounds)
    color_rows: list[np.ndarray] = []
    if colors is not None:
        color_rows = [
            np.flatnonzero(colors == color).astype(np.int64, copy=False)
            for color in range(int(colors.max(initial=-1)) + 1)
        ]
        color_rows = [rows for rows in color_rows if rows.size]
        color_rows = _reorder_small_partial_union_groups_by_overlap(
            expanded_bounds,
            color_rows,
        )
    color_count = len(color_rows)
    if (
        color_rows
        and 1 < color_count < unique_owned.row_count
        and color_count <= 128
        and color_count * 4 <= unique_owned.row_count
    ):
        partials: list[OwnedGeometryArray] = []
        for group_rows in color_rows:
            buffered_color = linestring_buffer_owned_array(
                unique_owned.take(group_rows),
                distance_value,
                quad_segs=quad_segs,
                cap_style=cap_style,
                join_style=join_style,
                dispatch_mode=ExecutionMode.GPU,
            )
            partial = disjoint_subset_union_all_owned(
                buffered_color,
                dispatch_mode=ExecutionMode.GPU,
            )
            if partial is None:
                partials = []
                break
            partials.append(partial)
        if partials:
            reduced = _reduce_partial_rows_gpu(partials)
        else:
            buffered_unique = linestring_buffer_owned_array(
                unique_owned,
                distance_value,
                quad_segs=quad_segs,
                cap_style=cap_style,
                join_style=join_style,
                dispatch_mode=ExecutionMode.GPU,
            )
            reduced = union_all_gpu_owned(
                buffered_unique,
                dispatch_mode=ExecutionMode.GPU,
            )
    else:
        buffered_unique = linestring_buffer_owned_array(
            unique_owned,
            distance_value,
            quad_segs=quad_segs,
            cap_style=cap_style,
            join_style=join_style,
            dispatch_mode=ExecutionMode.GPU,
        )
        reduced = union_all_gpu_owned(
            buffered_unique,
            dispatch_mode=ExecutionMode.GPU,
        )

    record_rewrite_event(
        rule_name="R9_dissolve_buffered_two_point_lines_exact_union",
        surface="geopandas.geodataframe.dissolve",
        original_operation="dissolve(method=unary)",
        rewritten_operation="buffer(unique(two_point_lines)).union_all_gpu",
        reason="single-group buffered two-point lines dissolve rewrites to deduped source-line buffering plus exact GPU union",
        detail=(
            f"rows={source_owned.row_count}, unique_rows={unique_rows.size}, "
            f"color_subsets={max(color_count, 0)}, "
            f"buffer_distance={distance_value}, quad_segs={quad_segs}"
        ),
    )
    record_dispatch_event(
        surface="geopandas.geodataframe.dissolve",
        operation="dissolve",
        implementation="buffered_two_point_line_exact_union_gpu",
        reason="deduped single-group buffered-line dissolve rewrite",
        detail=(
            f"rows={source_owned.row_count}, unique_rows={unique_rows.size}, "
            f"color_subsets={max(color_count, 0)}, "
            f"buffer_distance={distance_value}"
        ),
        selected=ExecutionMode.GPU,
    )

    non_empty_groups, empty_groups = _group_non_empty_counts(row_group_codes, group_count)
    return GroupedUnionResult(
        geometries=None,
        group_count=group_count,
        non_empty_groups=non_empty_groups,
        empty_groups=empty_groups,
        method=normalized_method,
        owned=reduced,
    )


def _maybe_execute_buffered_two_point_line_host_disjoint_subset_rewrite(
    *,
    normalized_method: DissolveUnionMethod,
    row_group_codes: np.ndarray | None,
    group_count: int,
    tag,
) -> GroupedUnionResult | None:
    from vibespatial.runtime.residency import Residency

    if (
        normalized_method is not DissolveUnionMethod.DISJOINT_SUBSET
        or row_group_codes is None
        or group_count != 1
        or not provenance_rewrites_enabled()
    ):
        return None

    if tag is None or tag.operation != "buffer":
        return None

    source_types = tag.source_geom_types
    if not source_types or source_types != frozenset({"linestring"}):
        return None

    source_owned = _provenance_source_owned(tag)
    if source_owned is None:
        return None
    if source_owned.residency is Residency.DEVICE:
        return None

    observed_rows = np.flatnonzero(row_group_codes >= 0).astype(np.int64, copy=False)
    if observed_rows.size < OVERLAY_UNION_ALL_GPU_THRESHOLD:
        return None
    if observed_rows.size != source_owned.row_count:
        source_owned = source_owned.take(observed_rows)

    unique_rows = _dedupe_two_point_linestring_rows_gpu(source_owned)
    if unique_rows is None or unique_rows.size == 0 or unique_rows.size >= source_owned.row_count:
        return None

    quad_segs_param = tag.get_param("quad_segs", 16)
    quad_segs = 16 if quad_segs_param is None else int(quad_segs_param)
    cap_style_param = tag.get_param("cap_style", "round")
    join_style_param = tag.get_param("join_style", "round")
    cap_style = "round" if cap_style_param is None else str(cap_style_param)
    join_style = "round" if join_style_param is None else str(join_style_param)
    distance_value = float(tag.get_param("distance", 0.0))
    if distance_value <= 0.0 or bool(tag.get_param("single_sided", False)):
        return None

    source_unique_owned = source_owned.take(unique_rows.astype(np.int64, copy=False))
    unique_lines = np.asarray(source_unique_owned.to_shapely(), dtype=object)
    buffered_unique = shapely.buffer(
        unique_lines,
        distance_value,
        quad_segs=quad_segs,
        cap_style=cap_style,
        join_style=join_style,
        single_sided=False,
    )
    reduced = shapely.disjoint_subset_union_all(np.asarray(buffered_unique, dtype=object))
    merged = np.asarray([reduced], dtype=object)

    record_rewrite_event(
        rule_name="R10_dissolve_buffered_two_point_lines_to_source_disjoint_subset",
        surface="geopandas.geodataframe.dissolve",
        original_operation="dissolve(method=disjoint_subset)",
        rewritten_operation="disjoint_subset_union_all(buffer(unique(two_point_lines)))",
        reason="single-group buffered two-point lines dissolve rebuilds the exact host disjoint-subset input from deduped source lines",
        detail=(
            f"rows={source_owned.row_count}, unique_rows={unique_rows.size}, "
            f"buffer_distance={distance_value}, quad_segs={quad_segs}"
        ),
    )
    record_dispatch_event(
        surface="geopandas.geodataframe.dissolve",
        operation="dissolve",
        implementation="buffered_two_point_line_source_disjoint_subset",
        reason="source-aware disjoint-subset rewrite for large duplicated buffered-line groups",
        detail=(
            f"rows={source_owned.row_count}, unique_rows={unique_rows.size}, "
            f"buffer_distance={distance_value}"
        ),
        selected=ExecutionMode.CPU,
    )

    non_empty_groups, empty_groups = _group_non_empty_counts(row_group_codes, group_count)
    return GroupedUnionResult(
        geometries=merged,
        group_count=group_count,
        non_empty_groups=non_empty_groups,
        empty_groups=empty_groups,
        method=normalized_method,
    )


def _maybe_execute_buffered_line_exact_cpu_rewrite(
    frame,
    *,
    normalized_method: DissolveUnionMethod,
    grid_size: float | None,
    row_group_codes: np.ndarray | None,
    group_count: int,
    tag,
) -> GroupedUnionResult | None:
    if (
        normalized_method is not DissolveUnionMethod.UNARY
        or grid_size is not None
        or row_group_codes is None
        or group_count != 1
        or tag is None
        or tag.operation != "buffer"
        or not provenance_rewrites_enabled()
    ):
        return None

    source_types = tag.source_geom_types
    if not source_types or not source_types <= frozenset({"linestring", "multilinestring"}):
        return None

    try:
        distance_value = float(tag.get_param("distance", 0.0))
    except (TypeError, ValueError):
        return None
    if distance_value <= 0.0 or bool(tag.get_param("single_sided", False)):
        return None

    observed_rows = np.flatnonzero(row_group_codes >= 0).astype(np.int64, copy=False)
    if (
        observed_rows.size < OVERLAY_UNION_ALL_GPU_THRESHOLD
        or observed_rows.size > _BUFFERED_LINE_EXACT_CPU_MAX_ROWS
    ):
        return None

    geometry_owned = getattr(frame.geometry.values, "_owned", None)
    from vibespatial.runtime.residency import Residency

    result_residency = (
        Residency.DEVICE
        if (
            geometry_owned is not None
            and geometry_owned.residency is Residency.DEVICE
            and cp is not None
        )
        else Residency.HOST
    )

    record_fallback_event(
        surface="geopandas.geodataframe.dissolve",
        reason="small buffered-line dissolve uses exact GEOS union until the GPU union path is both exact and faster",
        detail=(
            f"rows={observed_rows.size}, max_rows={_BUFFERED_LINE_EXACT_CPU_MAX_ROWS}, "
            f"buffer_distance={distance_value}"
        ),
        requested=ExecutionMode.AUTO,
        selected=ExecutionMode.CPU,
        pipeline="dissolve.buffered_line_exact_union",
        d2h_transfer=geometry_owned is not None and geometry_owned.residency is Residency.DEVICE,
    )

    quad_segs_param = tag.get_param("quad_segs", 16)
    quad_segs = 16 if quad_segs_param is None else int(quad_segs_param)
    cap_style_param = tag.get_param("cap_style", "round")
    join_style_param = tag.get_param("join_style", "round")
    cap_style = "round" if cap_style_param is None else str(cap_style_param)
    join_style = "round" if join_style_param is None else str(join_style_param)

    merged = None
    source_owned = _provenance_source_owned(tag)
    if source_owned is not None:
        if observed_rows.size != source_owned.row_count:
            source_owned = source_owned.take(observed_rows)
        unique_rows = _dedupe_two_point_linestring_rows_gpu(source_owned)
        if (
            unique_rows is not None
            and 0 < unique_rows.size < source_owned.row_count
        ):
            unique_lines = np.asarray(
                source_owned.take(unique_rows.astype(np.int64, copy=False)).to_shapely(),
                dtype=object,
            )
            buffered_unique = shapely.buffer(
                unique_lines,
                distance_value,
                quad_segs=quad_segs,
                cap_style=cap_style,
                join_style=join_style,
                single_sided=False,
            )
            merged = shapely.union_all(np.asarray(buffered_unique, dtype=object))

    if merged is None:
        if geometry_owned is not None:
            members = np.asarray(
                geometry_owned.take(observed_rows).to_shapely(),
                dtype=object,
            )
        else:
            values = np.asarray(frame.geometry.values, dtype=object)
            members = values[observed_rows]
        members = members[
            [geom is not None and not shapely.is_empty(geom) for geom in members]
        ]
        merged = _EMPTY if members.size == 0 else shapely.union_all(members)

    if merged is not None and merged.geom_type == "GeometryCollection":
        merged = _canonicalize_polygonal_make_valid_geometry(merged)
    if merged is not None and not shapely.is_valid(merged):
        merged = _canonicalize_polygonal_make_valid_geometry(shapely.make_valid(merged))

    from vibespatial.geometry.owned import from_shapely_geometries

    owned = from_shapely_geometries([merged], residency=result_residency)
    record_dispatch_event(
        surface="geopandas.geodataframe.dissolve",
        operation="dissolve",
        implementation="buffered_line_exact_cpu_union",
        reason="performance/correctness guard for small buffered-line dissolve groups",
        detail=(
            f"rows={observed_rows.size}, buffer_distance={distance_value}, "
            f"result_residency={result_residency.value}"
        ),
        requested=ExecutionMode.AUTO,
        selected=ExecutionMode.CPU,
    )

    non_empty_groups, empty_groups = _group_non_empty_counts(row_group_codes, group_count)
    return GroupedUnionResult(
        geometries=None,
        group_count=group_count,
        non_empty_groups=non_empty_groups,
        empty_groups=empty_groups,
        method=normalized_method,
        owned=owned,
    )


def _maybe_rewrite_buffered_line_dissolve_method(
    frame,
    *,
    normalized_method: DissolveUnionMethod,
    grid_size: float | None,
    row_group_codes: np.ndarray | None,
    group_positions: list[np.ndarray],
    group_count: int,
) -> DissolveUnionMethod:
    from vibespatial.runtime.residency import Residency

    if (
        normalized_method is not DissolveUnionMethod.UNARY
        or grid_size is not None
        or not provenance_rewrites_enabled()
        or shapely.geos_version < (3, 12, 0)
    ):
        return normalized_method

    geometry_values = frame.geometry.values
    tag = getattr(geometry_values, "_provenance", None)
    if tag is None or tag.operation != "buffer":
        return normalized_method

    source_types = tag.source_geom_types
    if not source_types or not source_types <= frozenset({"linestring", "multilinestring"}):
        return normalized_method

    distance = tag.get_param("distance", 0.0)
    try:
        distance_value = float(distance)
    except (TypeError, ValueError):
        return normalized_method
    if distance_value <= 0.0 or bool(tag.get_param("single_sided", False)):
        return normalized_method

    max_group_size = _max_group_size(row_group_codes, group_positions, group_count)
    if max_group_size < OVERLAY_UNION_ALL_GPU_THRESHOLD:
        return normalized_method

    geometry_owned = getattr(geometry_values, "_owned", None)
    if geometry_owned is not None and geometry_owned.residency is Residency.DEVICE:
        if strict_native_mode_enabled():
            return normalized_method
        source_owned = _provenance_source_owned(tag)
        if source_owned is None:
            return normalized_method
        if _dedupe_two_point_linestring_rows_gpu(source_owned) is not None:
            return normalized_method

    record_rewrite_event(
        rule_name="R8_dissolve_buffered_lines_to_disjoint_subset",
        surface="geopandas.geodataframe.dissolve",
        original_operation="dissolve(method=unary)",
        rewritten_operation="dissolve(method=disjoint_subset)",
        reason="buffered line unary dissolve rewrites to the exact disjoint-subset union engine for large grouped workloads",
        detail=(
            f"rows={len(frame)}, groups={group_count}, max_group_size={max_group_size}, "
            f"buffer_distance={distance_value}"
        ),
    )
    return DissolveUnionMethod.DISJOINT_SUBSET


def evaluate_geopandas_dissolve_native(
    frame,
    *,
    by,
    aggfunc,
    as_index: bool,
    level,
    sort: bool,
    observed: bool,
    dropna: bool,
    method: str,
    grid_size: float | None,
    agg_kwargs: dict[str, Any],
) -> NativeTabularResult:
    from vibespatial.runtime.execution_trace import execution_trace

    with execution_trace("dissolve"):
        normalized_method = DissolveUnionMethod(method)
        if normalized_method is not DissolveUnionMethod.UNARY and grid_size is not None:
            raise ValueError(f"grid_size is not supported for method '{method}'.")
        aggregated_data, group_positions, owned, row_group_codes = _prepare_grouped_dissolve(
            frame,
            by=by,
            aggfunc=aggfunc,
            level=level,
            sort=sort,
            observed=observed,
            dropna=dropna,
            normalized_method=normalized_method,
            agg_kwargs=agg_kwargs,
        )
        provenance_tag = getattr(frame.geometry.values, "_provenance", None)
        grouped_union = _maybe_execute_buffered_two_point_line_exact_union_rewrite(
            normalized_method=normalized_method,
            grid_size=grid_size,
            row_group_codes=row_group_codes,
            group_count=len(aggregated_data.index),
            tag=provenance_tag,
        )
        if grouped_union is None:
            grouped_union = _maybe_execute_buffered_line_exact_cpu_rewrite(
                frame,
                normalized_method=normalized_method,
                grid_size=grid_size,
                row_group_codes=row_group_codes,
                group_count=len(aggregated_data.index),
                tag=provenance_tag,
            )
        if grouped_union is None:
            normalized_method = _maybe_rewrite_buffered_line_dissolve_method(
                frame,
                normalized_method=normalized_method,
                grid_size=grid_size,
                row_group_codes=row_group_codes,
                group_positions=group_positions,
                group_count=len(aggregated_data.index),
            )
        if grouped_union is None:
            grouped_union = _maybe_execute_buffered_two_point_line_host_disjoint_subset_rewrite(
                normalized_method=normalized_method,
                row_group_codes=row_group_codes,
                group_count=len(aggregated_data.index),
                tag=provenance_tag,
            )
        if grouped_union is None and row_group_codes is not None:
            grouped_union = execute_grouped_union_codes(
                frame.geometry.array,
                row_group_codes,
                group_count=len(aggregated_data.index),
                method=normalized_method,
                grid_size=grid_size,
                owned=owned,
            )
        if grouped_union is None:
            if row_group_codes is not None and not group_positions:
                group_positions = _group_positions_from_codes(
                    row_group_codes,
                    len(aggregated_data.index),
                )
            grouped_union = execute_grouped_union(
                frame.geometry.array,
                group_positions,
                method=normalized_method,
                grid_size=grid_size,
                owned=owned,
            )
        return _grouped_constructive_result(
            grouped_union,
            frame=frame,
            aggregated_data=aggregated_data,
            as_index=as_index,
        )


def evaluate_geopandas_dissolve(
    frame,
    *,
    by,
    aggfunc,
    as_index: bool,
    level,
    sort: bool,
    observed: bool,
    dropna: bool,
    method: str,
    grid_size: float | None,
    agg_kwargs: dict[str, Any],
):
    native_result = evaluate_geopandas_dissolve_native(
        frame,
        by=by,
        aggfunc=aggfunc,
        as_index=as_index,
        level=level,
        sort=sort,
        observed=observed,
        dropna=dropna,
        method=method,
        grid_size=grid_size,
        agg_kwargs=agg_kwargs,
    )
    exported = _coerce_constructive_export_frame(
        native_result.to_geodataframe(),
        geometry_name=frame.geometry.name,
        frame_type=type(frame),
    )
    return exported


def evaluate_geopandas_lazy_dissolve(
    frame,
    *,
    by,
    aggfunc,
    as_index: bool,
    level,
    sort: bool,
    observed: bool,
    dropna: bool,
    method: str,
    grid_size: float | None,
    agg_kwargs: dict[str, Any],
) -> LazyDissolvedFrame:
    normalized_method = DissolveUnionMethod(method)
    if normalized_method is not DissolveUnionMethod.UNARY and grid_size is not None:
        raise ValueError(f"grid_size is not supported for method '{method}'.")
    aggregated_data, group_positions, owned, _row_group_codes = _prepare_grouped_dissolve(
        frame,
        by=by,
        aggfunc=aggfunc,
        level=level,
        sort=sort,
        observed=observed,
        dropna=dropna,
        normalized_method=normalized_method,
        agg_kwargs=agg_kwargs,
    )
    return LazyDissolvedFrame(
        frame=frame,
        aggregated_data=aggregated_data,
        group_positions=group_positions,
        row_group_codes=_row_group_codes,
        method=normalized_method,
        grid_size=grid_size,
        as_index=as_index,
        owned=owned,
    )


def _run_pipeline_once(frame, *, by, method):
    """Execute the pipeline path once and return (result, elapsed)."""
    start = perf_counter()
    result = evaluate_geopandas_dissolve(
        frame,
        by=by,
        aggfunc="first",
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method=str(method),
        grid_size=None,
        agg_kwargs={},
    )
    elapsed = perf_counter() - start
    return result, elapsed


def _run_baseline_once(frame, *, by, method):
    """Execute the baseline path once and return elapsed time."""
    start = perf_counter()
    baseline = frame.copy()
    data = baseline.drop(labels=baseline.geometry.name, axis=1)
    aggregated_data = data.groupby(by=by, sort=True, observed=False, dropna=True).agg("first")
    aggregated_geometry = type(frame)(
        baseline.groupby(group_keys=False, by=by, sort=True, observed=False, dropna=True)[baseline.geometry.name].agg(
            lambda block: block.union_all(method=str(method))
        ),
        geometry=baseline.geometry.name,
        crs=baseline.crs,
    )
    aggregated_geometry = aggregated_geometry.join(aggregated_data)
    return perf_counter() - start


def benchmark_dissolve_pipeline(
    frame,
    *,
    by,
    method: DissolveUnionMethod | str = DissolveUnionMethod.UNARY,
    dataset: str = "dissolve",
    iterations: int = 5,
    warmup: int = 1,
):
    """Benchmark the dissolve pipeline against the baseline groupby path.

    Uses *warmup* discarded runs followed by *iterations* timed runs per
    path.  Reports the **median** elapsed time to resist outlier noise.
    GC is disabled during timed sections to avoid non-deterministic pauses.

    Set *iterations=1* and *warmup=0* for a quick smoke-test (the old
    behaviour).
    """
    from vibespatial.cuda.cccl_precompile import ensure_pipelines_warm

    # Drain any CCCL background compilation so it does not interfere
    # with timing.
    ensure_pipelines_warm()

    # -- Warmup (results discarded) ------------------------------------
    for _ in range(warmup):
        _run_pipeline_once(frame, by=by, method=method)
        _run_baseline_once(frame, by=by, method=method)

    # Collect garbage once before timed section so both paths start from
    # a similar heap state.
    gc.collect()

    # -- Timed iterations: pipeline ------------------------------------
    pipeline_times: list[float] = []
    result = None
    gc.disable()
    try:
        for _ in range(iterations):
            r, elapsed = _run_pipeline_once(frame, by=by, method=method)
            pipeline_times.append(elapsed)
            if result is None:
                result = r
    finally:
        gc.enable()

    gc.collect()

    # -- Timed iterations: baseline ------------------------------------
    baseline_times: list[float] = []
    gc.disable()
    try:
        for _ in range(iterations):
            baseline_times.append(_run_baseline_once(frame, by=by, method=method))
    finally:
        gc.enable()

    pipeline_median = statistics.median(pipeline_times)
    baseline_median = statistics.median(baseline_times)

    return DissolveBenchmark(
        dataset=dataset,
        rows=len(frame),
        groups=int(result.shape[0]) if result is not None else 0,
        pipeline_elapsed_seconds=pipeline_median,
        baseline_elapsed_seconds=baseline_median,
        iterations=iterations,
    )


def union_all_owned(owned: OwnedGeometryArray) -> OwnedGeometryArray:
    """Union all geometries in *owned* into a single geometry.

    Bypasses the GeoDataFrame/groupby machinery used by
    ``evaluate_geopandas_dissolve`` and calls ``shapely.union_all``
    directly on the materialized geometries, then wraps the result back
    into an ``OwnedGeometryArray``.  This eliminates the GeoDataFrame
    construction and group-encoding overhead for the common single-group
    dissolve case (ADR-0005: minimize unnecessary host materialization).
    """
    from vibespatial.geometry.owned import from_shapely_geometries

    if owned.row_count == 0:
        return from_shapely_geometries([_EMPTY])
    if owned.row_count == 1:
        return owned

    # Materialize once to shapely and union directly -- no GeoDataFrame.
    geoms = owned.to_shapely()
    valid = [g for g in geoms if g is not None and not shapely.is_empty(g)]
    if not valid:
        return from_shapely_geometries([_EMPTY])
    merged = shapely.make_valid(shapely.union_all(np.asarray(valid, dtype=object)))
    return from_shapely_geometries([merged])


# ---------------------------------------------------------------------------
# GPU-accelerated union_all via device-resident tree-reduce (ADR-0017)
# ---------------------------------------------------------------------------



def union_all_gpu(
    owned: OwnedGeometryArray,
    *,
    grid_size: float | None = None,
    dispatch_mode: ExecutionMode | str = "auto",
    return_owned: bool = False,
) -> object:
    """Union all geometries in *owned* into a single result.

    When *return_owned* is False (default), returns a Shapely geometry
    (per GeoPandas union_all API contract).  When True, returns a
    single-row ``OwnedGeometryArray`` keeping the result device-resident
    (ADR-0005 zero-copy).

    Uses GPU tree-reduce via overlay_union_owned when beneficial:
    - Keeps all intermediate results as OwnedGeometryArray (no D->H->D per round)
    - O(log N) pairwise overlay rounds, O(1) JIT cost (precompiled)
    - Falls back to Shapely for small N or non-polygon families
    """
    from vibespatial.geometry.owned import from_shapely_geometries
    from vibespatial.runtime import ExecutionMode
    from vibespatial.runtime.adaptive import plan_dispatch_selection
    from vibespatial.runtime.dispatch import record_dispatch_event
    from vibespatial.runtime.precision import KernelClass

    if isinstance(dispatch_mode, str):
        dispatch_mode = ExecutionMode(dispatch_mode)

    row_count = owned.row_count
    if row_count == 0:
        if return_owned:
            return from_shapely_geometries([GeometryCollection()])
        return GeometryCollection()

    if row_count == 1:
        if return_owned:
            return owned if owned.validity[0] else from_shapely_geometries([GeometryCollection()])
        geoms = owned.to_shapely()
        return geoms[0] if geoms[0] is not None else GeometryCollection()

    # Dispatch decision
    selection = plan_dispatch_selection(
        kernel_name="union_all",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=row_count,
        requested_mode=dispatch_mode,
        current_residency=owned.residency,
    )

    if (
        selection.selected is ExecutionMode.GPU
        and row_count >= OVERLAY_UNION_ALL_GPU_THRESHOLD
        and grid_size is None  # grid_size not supported on GPU path
    ):
        # Validate input: GPU overlay requires polygon-family geometries
        polygon_tags = set()
        from vibespatial.geometry.buffers import GeometryFamily
        from vibespatial.geometry.owned import FAMILY_TAGS
        for fam in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
            polygon_tags.add(FAMILY_TAGS[fam])
        valid_tags = np.isin(owned.tags[owned.validity], list(polygon_tags))
        if np.all(valid_tags):
            result = _union_all_tree_reduce_gpu(owned, return_owned=return_owned)
            if result is not None:
                impl = "gpu_tree_reduce_overlay_owned" if return_owned else "gpu_tree_reduce_overlay"
                record_dispatch_event(
                    surface="union_all",
                    operation="union_all",
                    implementation=impl,
                    reason=f"tree-reduce via overlay_union_owned, {int(np.ceil(np.log2(row_count)))} rounds",
                    detail=f"rows={row_count}",
                    selected=ExecutionMode.GPU,
                )
                return result

    # CPU fallback: Shapely union_all
    record_dispatch_event(
        surface="union_all",
        operation="union_all",
        implementation="shapely",
        reason="CPU fallback",
        detail=f"rows={row_count}, grid_size={grid_size}",
        selected=ExecutionMode.CPU,
    )
    geoms = owned.to_shapely()
    valid = [g for g in geoms if g is not None and not shapely.is_empty(g)]
    if not valid:
        if return_owned:
            return from_shapely_geometries([GeometryCollection()])
        return GeometryCollection()
    merged = shapely.make_valid(
        shapely.union_all(np.asarray(valid, dtype=object), grid_size=grid_size)
    )
    if return_owned:
        return from_shapely_geometries([merged])
    return merged


def _union_all_tree_reduce_gpu(
    owned: OwnedGeometryArray,
    *,
    return_owned: bool = False,
) -> object | None:
    """GPU tree-reduce: union N geometries in log2(N) rounds.

    When *return_owned* is False (default), materializes the final result to
    a Shapely geometry (single D->H).  When True, returns the single-row
    ``OwnedGeometryArray`` directly -- no final D->H.

    ADR-0002: CONSTRUCTIVE class, fp64 (segment intersection precision).
    ADR-0005: Device-resident intermediates; single D->H only when needed.
    ADR-0033: Inherits overlay pipeline tiers (NVRTC + CCCL + CuPy).

    Resilience: if a GPU overlay raises a CUDA error (e.g. ILLEGAL_ADDRESS
    from degenerate half-edge topology), the pair falls back to Shapely CPU
    union.  If the CUDA context itself becomes unusable (multiple consecutive
    GPU failures suggesting context corruption), the entire remaining
    reduction switches to CPU to avoid cascading failures.
    """
    import math

    from vibespatial.geometry.owned import from_shapely_geometries
    from vibespatial.runtime import ExecutionMode

    from .gpu import overlay_union_owned

    # Single bulk D->H to identify non-empty rows (1 transfer, not N).
    geoms = owned.to_shapely()
    keep = np.array([
        i for i in range(owned.row_count)
        if owned.validity[i] and geoms[i] is not None and not shapely.is_empty(geoms[i])
    ], dtype=np.intp)

    if keep.size == 0:
        if return_owned:
            return from_shapely_geometries([GeometryCollection()])
        return GeometryCollection()
    if keep.size == 1:
        if return_owned:
            return owned.take(keep)
        g = geoms[keep[0]]
        return g

    current = [owned.take(np.array([idx], dtype=np.intp)) for idx in keep]

    # Tree-reduce: each round halves the geometry count.
    # Safety: cap rounds to prevent infinite loop from unexpected behavior,
    # and track consecutive GPU failures to detect context corruption.
    max_rounds = int(math.ceil(math.log2(max(len(current), 2)))) + 2
    rounds = 0
    consecutive_gpu_failures = 0
    while len(current) > 1 and rounds < max_rounds:
        next_round: list[OwnedGeometryArray] = []
        for i in range(0, len(current), 2):
            if i + 1 < len(current):
                gpu_ok = False
                if consecutive_gpu_failures < OVERLAY_GPU_FAILURE_THRESHOLD:
                    try:
                        result = overlay_union_owned(
                            current[i], current[i + 1],
                            dispatch_mode=ExecutionMode.GPU,
                        )
                        next_round.append(result)
                        gpu_ok = True
                        consecutive_gpu_failures = 0
                    except Exception:
                        consecutive_gpu_failures += 1

                if not gpu_ok:
                    # CPU fallback: materialize both sides and use Shapely.
                    # Use try/except for to_shapely() since it may fail if
                    # the CUDA context is corrupted (device-resident data).
                    try:
                        left_g = current[i].to_shapely()[0]
                    except Exception:
                        left_g = _EMPTY
                    try:
                        right_g = current[i + 1].to_shapely()[0]
                    except Exception:
                        right_g = _EMPTY
                    try:
                        merged = shapely.union(left_g, right_g)
                        if merged is not None and not shapely.is_valid(merged):
                            merged = shapely.make_valid(merged)
                    except Exception:
                        merged = _EMPTY
                    next_round.append(from_shapely_geometries(
                        [merged if merged is not None else _EMPTY],
                    ))
            else:
                next_round.append(current[i])
        # Release previous round's intermediates promptly.
        del current
        current = next_round
        rounds += 1
        # Phase 25 memory: release GPU pool memory between tree-reduce
        # rounds so overlay intermediates (split events, half-edge graphs,
        # face tables) don't accumulate across rounds.
        try:
            from vibespatial.cuda._runtime import get_cuda_runtime
            get_cuda_runtime().free_pool_memory()
        except Exception:
            pass  # best-effort cleanup

    if return_owned:
        return current[0]

    # Materialize final result to Shapely geometry
    final_geoms = current[0].to_shapely()
    result = final_geoms[0] if final_geoms else GeometryCollection()
    if result is not None and not shapely.is_valid(result):
        result = shapely.make_valid(result)
    return result


def union_all_gpu_owned(owned, *, grid_size=None, dispatch_mode="auto"):
    """Union all geometries into a single-row OwnedGeometryArray (device-resident).

    Convenience wrapper: calls ``union_all_gpu`` with ``return_owned=True``.
    """
    return union_all_gpu(owned, grid_size=grid_size, dispatch_mode=dispatch_mode, return_owned=True)
