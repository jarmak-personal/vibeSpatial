from __future__ import annotations

import gc
import logging
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import GeometryCollection

from vibespatial.api._native_grouped import NativeGrouped, NativeGroupedAttributeReduction
from vibespatial.api._native_result_core import NativeAttributeTable, NativeGeometryProvenance
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

from ._host_boundary import overlay_bool_scalar, overlay_device_to_host, overlay_int_scalar

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
_COVERAGE_REWRITE_MAX_CANDIDATE_PAIRS = 250_000
_COVERAGE_REWRITE_NEGATIVE_PROBE_PAIRS = 1
_NATIVE_GROUPED_ATTRIBUTE_REDUCERS = frozenset(
    {"sum", "count", "mean", "min", "max", "first", "last", "any", "all"}
)
_NATIVE_GROUPED_TAKE_REDUCERS = frozenset({"first", "last"})
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


def _device_grouped_union_repair_needed(owned: OwnedGeometryArray) -> bool | None:
    """Return whether grouped-union output needs repair without host metadata."""
    cached = owned._current_cached_validity_mask()
    if cached is not None:
        return not bool(np.all(cached))
    if owned.device_state is None or cp is None:
        return None
    from vibespatial.constructive.validity import validity_expression_owned

    state = owned._ensure_device_state()
    expression = validity_expression_owned(owned)
    d_validity = cp.asarray(state.validity, dtype=cp.bool_)
    d_valid_flags = cp.asarray(expression.values, dtype=cp.bool_)
    d_repair_needed = (~d_valid_flags) & d_validity
    repair_count = overlay_int_scalar(
        cp.count_nonzero(d_repair_needed),
        reason="dissolve grouped-union repair-needed count fence",
    )
    return repair_count > 0


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

    repair_needed = _device_grouped_union_repair_needed(reduced)
    if repair_needed is not None:
        if not repair_needed:
            seed_all_validity_cache(reduced)
            return reduced
        if reduced.device_state is not None:
            from vibespatial.constructive.make_valid_pipeline import make_valid_owned

            requested_mode = (
                ExecutionMode.GPU
                if strict_native_mode_enabled()
                else ExecutionMode.AUTO
            )
            try:
                mv_result = make_valid_owned(
                    owned=reduced,
                    dispatch_mode=requested_mode,
                )
            except Exception:
                if strict_native_mode_enabled():
                    raise
            else:
                if (
                    mv_result.owned is not None
                    and mv_result.selected is ExecutionMode.GPU
                ):
                    repaired_owned = mv_result.owned
                    remaining_repair_needed = _device_grouped_union_repair_needed(
                        repaired_owned,
                    )
                    if remaining_repair_needed is False:
                        seed_all_validity_cache(repaired_owned)
                        record_dispatch_event(
                            surface="geopandas.geodataframe.dissolve",
                            operation="grouped_union_boundary_repair",
                            implementation="gpu_make_valid_subset",
                            reason=(
                                "repaired invalid grouped GPU union rows through "
                                "native make-valid"
                            ),
                            detail=(
                                f"groups={group_count}, "
                                f"repaired={int(mv_result.repaired_rows.size)}"
                            ),
                            selected=ExecutionMode.GPU,
                        )
                        return repaired_owned

    invalid_mask = ~np.asarray(is_valid_owned(reduced), dtype=bool)
    if invalid_mask.any() and not bool(np.all(reduced.validity)):
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
class _GroupedGeometryResult:
    geometry: GeometryNativeResult
    repaired: bool = False


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


def _grouped_union_geometry_payload(
    grouped_union: GroupedUnionResult,
    *,
    geometry_name: str,
    crs,
) -> _GroupedGeometryResult:
    if grouped_union.owned is not None:
        return _repair_grouped_union_owned_if_needed(
            grouped_union.owned,
            group_count=grouped_union.group_count,
            geometry_name=geometry_name,
            crs=crs,
        )

    values = np.asarray(grouped_union.geometries, dtype=object)
    try:
        from vibespatial.geometry.owned import from_shapely_geometries

        owned = from_shapely_geometries(values.tolist())
    except (NotImplementedError, TypeError, ValueError):
        from vibespatial.api.geoseries import GeoSeries

        geometry = GeometryNativeResult.from_geoseries(
            GeoSeries(values, name=geometry_name, crs=crs)
        )
    else:
        geometry = GeometryNativeResult.from_owned(owned, crs=crs)

    return _GroupedGeometryResult(
        geometry,
        repaired=False,
    )


def _grouped_union_geometry_result(
    grouped_union: GroupedUnionResult,
    *,
    geometry_name: str,
    crs,
) -> GeometryNativeResult:
    return _grouped_union_geometry_payload(
        grouped_union,
        geometry_name=geometry_name,
        crs=crs,
    ).geometry


def _grouped_constructive_provenance(
    grouped_union: GroupedUnionResult,
    *,
    frame,
    repaired: bool,
) -> NativeGeometryProvenance:
    operation = f"grouped_{grouped_union.method.value}_union"
    if repaired:
        operation = f"{operation}_repair"
    source_rows: Any
    if grouped_union.owned is not None:
        from vibespatial.runtime.residency import Residency

        if grouped_union.owned.residency is Residency.DEVICE and cp is not None:
            source_rows = cp.arange(grouped_union.group_count, dtype=cp.int64)
        else:
            source_rows = np.arange(grouped_union.group_count, dtype=np.int64)
    else:
        source_rows = np.arange(grouped_union.group_count, dtype=np.int64)

    source_tokens: tuple[str, ...] = ()
    try:
        from vibespatial.api._native_state import get_native_state

        state = get_native_state(frame)
        if state is not None:
            source_tokens = (state.lineage_token,)
    except Exception:
        source_tokens = ()

    return NativeGeometryProvenance(
        operation=operation,
        row_count=grouped_union.group_count,
        source_rows=source_rows,
        source_tokens=source_tokens,
    )


def _grouped_constructive_result(
    grouped_union: GroupedUnionResult,
    *,
    frame,
    aggregated_data: pd.DataFrame,
    as_index: bool,
) -> NativeTabularResult:
    geometry_payload = _grouped_union_geometry_payload(
        grouped_union,
        geometry_name=frame.geometry.name,
        crs=frame.crs,
    )
    return _grouped_constructive_to_native_tabular_result(
        geometry=geometry_payload.geometry,
        geometry_name=frame.geometry.name,
        as_index=as_index,
        attributes=aggregated_data,
        provenance=_grouped_constructive_provenance(
            grouped_union,
            frame=frame,
            repaired=geometry_payload.repaired,
        ),
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


def _native_dissolve_group_index_and_codes(
    frame,
    *,
    by,
    level,
    observed: bool,
    sort: bool,
    dropna: bool,
    normalized_method: DissolveUnionMethod | None = None,
    grid_size: float | None = None,
    allow_device_key_codes: bool = False,
) -> tuple[pd.Index, Any, tuple[Any, ...]] | None:
    """Build the narrow group contract used by native dissolve reducers."""
    if level is not None:
        return None
    if by is None:
        return (
            pd.Index(np.asarray([0], dtype=np.int64)),
            np.zeros(len(frame), dtype=np.int32),
            tuple(),
        )
    if allow_device_key_codes:
        device_contract = _native_device_plain_group_index_and_codes(
            frame,
            by=by,
            normalized_method=normalized_method,
            grid_size=grid_size,
            observed=observed,
            sort=sort,
            dropna=dropna,
        )
        if device_contract is not None:
            return device_contract
    if isinstance(by, (list, tuple)):
        return _native_multi_plain_group_index_and_codes(
            frame,
            by=tuple(by),
            observed=observed,
            sort=sort,
            dropna=dropna,
        )
    if not isinstance(by, str) or by not in frame.columns or by == frame.geometry.name:
        return None

    key = frame[by]
    if not isinstance(key, pd.Series):
        return None
    if isinstance(key.dtype, pd.CategoricalDtype):
        return _native_categorical_group_index_and_codes(
            key,
            observed=observed,
            sort=sort,
            dropna=dropna,
        )
    return _native_plain_group_index_and_codes(
        key,
        sort=sort,
        dropna=dropna,
    )


def _native_device_plain_group_index_and_codes(
    frame,
    *,
    by,
    normalized_method: DissolveUnionMethod | None,
    grid_size: float | None = None,
    observed: bool,
    sort: bool,
    dropna: bool,
) -> tuple[pd.Index, Any, tuple[Any, ...]] | None:
    """Encode admitted device-backed group keys on device.

    Physical shape: row-aligned device key vector(s), optionally dictionary
    codes plus validity -> dense group codes for a segmented `NativeGrouped`
    consumer.  The only host boundary is the compact group-label export needed
    for the public dissolve index.
    """
    if isinstance(by, str):
        key_columns = (by,)
    elif isinstance(by, (list, tuple)):
        key_columns = tuple(by)
    else:
        return None

    if (
        cp is None
        or not sort
        or grid_size is not None
        or normalized_method
        not in {
            DissolveUnionMethod.UNARY,
            DissolveUnionMethod.COVERAGE,
            DissolveUnionMethod.DISJOINT_SUBSET,
        }
        or not key_columns
        or len(set(key_columns)) != len(key_columns)
        or any(
            not isinstance(column, str)
            or column not in frame.columns
            or column == frame.geometry.name
            for column in key_columns
        )
        or getattr(frame.geometry.values, "_provenance", None) is not None
    ):
        return None

    try:
        from vibespatial.api._native_state import get_native_state
        from vibespatial.runtime.residency import Residency

        state = get_native_state(frame)
    except Exception:
        return None
    if state is None:
        return None

    owned = getattr(state.geometry, "owned", None)
    if (
        owned is None
        or owned.residency is not Residency.DEVICE
        or not _owned_supports_polygonal_grouped_union(owned)
    ):
        return None

    key_payload = _device_dissolve_key_columns(
        NativeAttributeTable.from_value(state.attributes),
        key_columns,
        frame=frame,
    )
    if key_payload is None:
        return None
    if len(key_columns) == 1 and key_payload[0].categorical_dtype is not None:
        return _device_categorical_group_index_and_codes(
            key_payload[0],
            name=key_columns[0],
            observed=observed,
            dropna=dropna,
        )
    has_categorical = any(column.categorical_dtype is not None for column in key_payload)
    value_arrays = tuple(column.values for column in key_payload)
    dtypes = tuple(np.dtype(values.dtype) for values in value_arrays)
    if any(
        not (
            np.issubdtype(dtype, np.integer)
            or np.issubdtype(dtype, np.bool_)
        )
        for dtype in dtypes
    ):
        return None
    row_count = int(value_arrays[0].size)
    if any(int(values.size) != row_count for values in value_arrays):
        return None
    if has_categorical and not observed:
        return _device_multi_unobserved_product_index_and_codes(
            key_payload,
            key_columns=key_columns,
            dropna=dropna,
        )

    if len(key_columns) == 1:
        key_column = key_payload[0]
        values_array = key_column.values
        try:
            if key_column.valid_mask is None:
                labels, inverse = cp.unique(values_array, return_inverse=True)
            else:
                valid_values = values_array[key_column.valid_mask]
                labels, inverse_valid = cp.unique(valid_values, return_inverse=True)
                inverse = cp.full(row_count, -1, dtype=cp.int32)
                inverse[key_column.valid_mask] = inverse_valid.astype(
                    cp.int32,
                    copy=False,
                )
                if not dropna:
                    inverse[~key_column.valid_mask] = int(labels.size)
        except Exception:
            return None
        host_labels = _materialize_device_group_labels(
            labels,
            operation="device_group_key_labels_to_host",
            reason="device dissolve group labels were materialized for public output index",
        )
        output_index = _device_group_label_index(
            host_labels,
            name=key_columns[0],
            pandas_dtype=key_column.pandas_dtype,
            include_null=key_column.valid_mask is not None and not dropna,
        )
        return output_index, inverse.astype(cp.int32, copy=False), key_columns

    if row_count == 0:
        host_columns = [
            np.asarray([], dtype=dtype)
            for dtype in dtypes
        ]
        host_valid_columns = [None for _ in host_columns]
        return (
            _device_multi_group_label_index(
                host_columns,
                host_valid_columns,
                key_payload,
                key_columns,
            ),
            cp.empty(0, dtype=cp.int32),
            key_columns,
        )

    try:
        ordered_rows = _device_multi_key_order(key_payload, dropna=dropna)
        if ordered_rows is None:
            return None
        sorted_values = tuple(column.values[ordered_rows] for column in key_payload)
        sorted_valid = tuple(
            None if column.valid_mask is None else column.valid_mask[ordered_rows]
            for column in key_payload
        )
        starts = cp.empty(row_count, dtype=cp.bool_)
        starts[...] = False
        ordered_count = int(ordered_rows.size)
        if ordered_count:
            starts[:ordered_count][0] = True
        if ordered_count > 1:
            changed = _device_multi_key_changed(sorted_values, sorted_valid)
            starts[:ordered_count][1:] = changed
        start_positions = cp.nonzero(starts)[0]
        group_ids_sorted = cp.cumsum(starts.astype(cp.int32)) - 1
        inverse = cp.full(row_count, -1, dtype=cp.int32)
        inverse[ordered_rows] = group_ids_sorted[:ordered_count].astype(
            cp.int32,
            copy=False,
        )
        unique_source_rows = ordered_rows[start_positions]
        label_columns = tuple(column.values[unique_source_rows] for column in key_payload)
        label_valid_columns = tuple(
            None if column.valid_mask is None else column.valid_mask[unique_source_rows]
            for column in key_payload
        )
    except Exception:
        return None

    if any(valid is not None for valid in label_valid_columns) and not dropna:
        host_columns, host_valid_columns = _materialize_device_group_label_columns_with_validity(
            label_columns,
            label_valid_columns,
            operation="device_multi_group_key_labels_to_host",
            reason="device dissolve multi-key labels were materialized for public output index",
        )
    else:
        host_columns = _materialize_device_group_label_columns(
            label_columns,
            operation="device_multi_group_key_labels_to_host",
            reason="device dissolve multi-key labels were materialized for public output index",
        )
        host_valid_columns = [None for _ in host_columns]
    output_index = _device_multi_group_label_index(
        host_columns,
        host_valid_columns,
        key_payload,
        key_columns,
    )
    return output_index, inverse, key_columns


@dataclass(frozen=True)
class _DeviceDissolveKeyColumn:
    values: Any
    valid_mask: Any | None
    pandas_dtype: Any | None
    categorical_dtype: pd.CategoricalDtype | None = None


def _device_dissolve_key_columns(
    attributes: NativeAttributeTable,
    key_columns: tuple[Any, ...],
    *,
    frame,
) -> tuple[_DeviceDissolveKeyColumn, ...] | None:
    if cp is None or attributes.device_table is None:
        return None
    positions = {
        name: index
        for index, name in enumerate(tuple(attributes.column_override or ()))
    }
    if len(positions) != len(tuple(attributes.column_override or ())):
        return None
    if any(column not in positions for column in key_columns):
        return None
    if not hasattr(attributes.device_table, "columns"):
        return None
    source_columns = attributes.device_table.columns()
    payload = []
    for column in key_columns:
        try:
            series = frame[column]
        except Exception:
            return None
        if not isinstance(series, pd.Series):
            return None
        view = _device_dissolve_key_column_view(
            source_columns[positions[column]],
            series=series,
        )
        if view is None:
            return None
        payload.append(view)
    return tuple(payload)


def _device_dissolve_key_column_view(
    column,
    *,
    series: pd.Series,
) -> _DeviceDissolveKeyColumn | None:
    try:
        import pyarrow as pa
    except ModuleNotFoundError:
        return None
    if int(column.offset()) != 0:
        return None
    if isinstance(series.dtype, pd.CategoricalDtype):
        return _device_categorical_key_column_view(column, series.dtype)
    arrow_type = column.type().to_arrow()
    if not (pa.types.is_integer(arrow_type) or pa.types.is_boolean(arrow_type)):
        return None
    try:
        dtype = np.dtype(column.type().typestr)
        values = cp.asarray(column.data()).view(dtype)[: int(column.size())]
    except Exception:
        return None
    null_count = int(column.null_count())
    if null_count == 0:
        return _DeviceDissolveKeyColumn(
            values=values,
            valid_mask=None,
            pandas_dtype=None,
        )
    valid_mask = _device_column_valid_mask(column)
    if valid_mask is None:
        return None
    return _DeviceDissolveKeyColumn(
        values=values,
        valid_mask=valid_mask,
        pandas_dtype=_nullable_pandas_dtype_for_device_key(dtype),
    )


def _device_categorical_key_column_view(
    column,
    dtype: pd.CategoricalDtype,
) -> _DeviceDissolveKeyColumn | None:
    try:
        if int(column.num_children()) < 1:
            return None
        codes_column = column.child(0)
        codes_dtype = np.dtype(codes_column.type().typestr)
        if not np.issubdtype(codes_dtype, np.integer):
            return None
        values = cp.asarray(codes_column.data()).view(codes_dtype)[: int(column.size())]
    except Exception:
        return None
    valid_mask = None
    if int(column.null_count()) > 0:
        valid_mask = _device_column_valid_mask(column)
        if valid_mask is None:
            return None
    return _DeviceDissolveKeyColumn(
        values=values,
        valid_mask=valid_mask,
        pandas_dtype=None,
        categorical_dtype=dtype,
    )


def _device_column_valid_mask(column):
    try:
        mask_bytes = cp.asarray(column.null_mask()).view(cp.uint8)
        row_offsets = cp.arange(int(column.size()), dtype=cp.int64) + int(column.offset())
        return ((mask_bytes[row_offsets >> 3] >> (row_offsets & 7)) & 1).astype(
            cp.bool_,
            copy=False,
        )
    except Exception:
        return None


def _nullable_pandas_dtype_for_device_key(dtype: np.dtype) -> str | None:
    normalized = np.dtype(dtype)
    if np.issubdtype(normalized, np.bool_):
        return "boolean"
    if np.issubdtype(normalized, np.signedinteger):
        return f"Int{normalized.itemsize * 8}"
    if np.issubdtype(normalized, np.unsignedinteger):
        return f"UInt{normalized.itemsize * 8}"
    return None


def _device_group_label_index(
    labels: np.ndarray,
    *,
    name: Any,
    pandas_dtype: Any | None,
    include_null: bool,
) -> pd.Index:
    if pandas_dtype is None:
        return pd.Index(labels, name=name)
    values: Any = labels
    if include_null:
        values = [*labels.tolist(), pd.NA]
    return pd.Index(pd.array(values, dtype=pandas_dtype), name=name)


def _device_categorical_group_index_and_codes(
    key_column: _DeviceDissolveKeyColumn,
    *,
    name: Any,
    observed: bool,
    dropna: bool,
) -> tuple[pd.CategoricalIndex, Any, tuple[Any, ...]] | None:
    dtype = key_column.categorical_dtype
    if dtype is None:
        return None
    try:
        category_count = len(dtype.categories)
        raw_codes = cp.asarray(key_column.values, dtype=cp.int32)
        row_count = int(raw_codes.size)
        valid_mask = key_column.valid_mask
        include_null = valid_mask is not None and not dropna
        if observed:
            valid_codes = raw_codes if valid_mask is None else raw_codes[valid_mask]
            observed_codes = cp.unique(valid_codes)
            host_codes = _materialize_device_group_labels(
                observed_codes,
                operation="device_categorical_group_key_codes_to_host",
                reason=(
                    "device dissolve categorical group codes were materialized "
                    "for public output index"
                ),
            )
            output_codes = host_codes.astype(np.int64, copy=False)
            observed_count = int(observed_codes.size)
            inverse = cp.full(row_count, -1, dtype=cp.int32)
            if observed_count:
                if valid_mask is None:
                    inverse = cp.searchsorted(observed_codes, raw_codes).astype(
                        cp.int32,
                        copy=False,
                    )
                else:
                    inverse[valid_mask] = cp.searchsorted(
                        observed_codes,
                        raw_codes[valid_mask],
                    ).astype(cp.int32, copy=False)
            if include_null:
                output_codes = np.concatenate(
                    [output_codes, np.asarray([-1], dtype=np.int64)],
                )
                inverse[~valid_mask] = observed_count
        else:
            output_codes = np.arange(category_count, dtype=np.int64)
            inverse = raw_codes.astype(cp.int32, copy=True)
            if include_null:
                output_codes = np.concatenate(
                    [output_codes, np.asarray([-1], dtype=np.int64)],
                )
                inverse[~valid_mask] = category_count
            elif valid_mask is not None:
                inverse[~valid_mask] = -1
        output_index = pd.CategoricalIndex(
            pd.Categorical.from_codes(
                output_codes,
                categories=dtype.categories,
                ordered=dtype.ordered,
            ),
            name=name,
        )
    except Exception:
        return None
    return output_index, inverse.astype(cp.int32, copy=False), (name,)


def _device_multi_key_order(
    columns: tuple[_DeviceDissolveKeyColumn, ...],
    *,
    dropna: bool,
):
    row_count = int(columns[0].values.size)
    if dropna:
        row_mask = cp.ones(row_count, dtype=cp.bool_)
        for column in columns:
            if column.valid_mask is not None:
                row_mask = row_mask & column.valid_mask
        row_indices = cp.nonzero(row_mask)[0]
    else:
        row_indices = cp.arange(row_count, dtype=cp.int64)
    if int(row_indices.size) == 0:
        return row_indices

    sort_keys = []
    for column in reversed(columns):
        values = column.values[row_indices]
        if column.valid_mask is None:
            sort_keys.append(values)
            continue
        valid = column.valid_mask[row_indices]
        safe_values = cp.where(valid, values, cp.zeros((), dtype=values.dtype))
        sort_keys.append(safe_values)
        sort_keys.append(~valid)
    return row_indices[cp.lexsort(cp.stack(tuple(sort_keys)))]


def _device_multi_key_changed(
    sorted_values: tuple[Any, ...],
    sorted_valid: tuple[Any | None, ...],
):
    changed = None
    for values, valid in zip(sorted_values, sorted_valid, strict=True):
        if valid is None:
            column_changed = values[1:] != values[:-1]
        else:
            both_valid = valid[1:] & valid[:-1]
            both_null = (~valid[1:]) & (~valid[:-1])
            column_equal = both_null | (both_valid & (values[1:] == values[:-1]))
            column_changed = ~column_equal
        changed = column_changed if changed is None else changed | column_changed
    return changed


def _device_multi_group_label_index(
    host_columns: list[np.ndarray],
    host_valid_columns: list[np.ndarray | None],
    key_payload: tuple[_DeviceDissolveKeyColumn, ...],
    key_columns: tuple[Any, ...],
) -> pd.MultiIndex:
    arrays = []
    for labels, valid, payload in zip(
        host_columns,
        host_valid_columns,
        key_payload,
        strict=True,
    ):
        if payload.categorical_dtype is not None:
            codes: Any = labels.astype(np.int64, copy=False)
            if valid is not None:
                codes = np.where(valid.astype(bool, copy=False), codes, -1)
            arrays.append(
                pd.CategoricalIndex(
                    pd.Categorical.from_codes(
                        codes,
                        categories=payload.categorical_dtype.categories,
                        ordered=payload.categorical_dtype.ordered,
                    )
                )
            )
            continue
        if payload.pandas_dtype is None:
            arrays.append(labels)
            continue
        values = pd.array(labels, dtype=payload.pandas_dtype)
        if valid is not None:
            values[~valid.astype(bool, copy=False)] = pd.NA
        arrays.append(values)
    return pd.MultiIndex.from_arrays(arrays, names=list(key_columns))


def _device_multi_unobserved_product_index_and_codes(
    key_payload: tuple[_DeviceDissolveKeyColumn, ...],
    *,
    key_columns: tuple[Any, ...],
    dropna: bool,
) -> tuple[pd.MultiIndex, Any, tuple[Any, ...]] | None:
    row_count = int(key_payload[0].values.size)
    levels: list[pd.Index] = []
    code_columns = []
    level_sizes: list[int] = []
    valid_rows = cp.ones(row_count, dtype=cp.bool_)
    try:
        for payload, name in zip(key_payload, key_columns, strict=True):
            values = cp.asarray(payload.values, dtype=cp.int32)
            if payload.categorical_dtype is not None:
                dtype = payload.categorical_dtype
                category_count = len(dtype.categories)
                include_null = payload.valid_mask is not None and not dropna
                level_codes = np.arange(category_count, dtype=np.int64)
                if include_null:
                    level_codes = np.concatenate(
                        [level_codes, np.asarray([-1], dtype=np.int64)]
                    )
                levels.append(
                    pd.CategoricalIndex(
                        pd.Categorical.from_codes(
                            level_codes,
                            categories=dtype.categories,
                            ordered=dtype.ordered,
                        ),
                        name=name,
                    )
                )
                level_size = category_count + int(include_null)
                if payload.valid_mask is None:
                    column_codes = values
                elif dropna:
                    valid_rows = valid_rows & payload.valid_mask
                    column_codes = values
                else:
                    column_codes = cp.where(
                        payload.valid_mask,
                        values,
                        cp.asarray(category_count, dtype=cp.int32),
                    )
                code_columns.append(column_codes.astype(cp.int64, copy=False))
                level_sizes.append(level_size)
                continue

            if payload.valid_mask is None:
                labels = cp.unique(payload.values)
                column_codes = cp.searchsorted(labels, payload.values).astype(
                    cp.int64,
                    copy=False,
                )
                include_null = False
            else:
                valid_values = payload.values[payload.valid_mask]
                labels = cp.unique(valid_values)
                valid_code_values = cp.searchsorted(labels, valid_values).astype(
                    cp.int64,
                    copy=False,
                )
                column_codes = cp.full(row_count, -1, dtype=cp.int64)
                column_codes[payload.valid_mask] = valid_code_values
                if dropna:
                    valid_rows = valid_rows & payload.valid_mask
                    include_null = False
                else:
                    include_null = True
                    column_codes[~payload.valid_mask] = labels.size
            host_labels = _materialize_device_group_labels(
                labels,
                operation="device_multi_product_group_key_labels_to_host",
                reason=(
                    "device dissolve multi-key product labels were materialized "
                    "for public output index"
                ),
            )
            levels.append(
                _device_group_label_index(
                    host_labels,
                    name=name,
                    pandas_dtype=payload.pandas_dtype,
                    include_null=include_null,
                )
            )
            code_columns.append(column_codes)
            level_sizes.append(labels.size + int(include_null))
        mixed_codes = cp.zeros(row_count, dtype=cp.int64)
        for column_codes, level_size in zip(code_columns, level_sizes, strict=True):
            mixed_codes = mixed_codes * int(level_size) + column_codes
        if dropna:
            row_group_codes = cp.full(row_count, -1, dtype=cp.int32)
            row_group_codes[valid_rows] = mixed_codes[valid_rows].astype(
                cp.int32,
                copy=False,
            )
        else:
            row_group_codes = mixed_codes.astype(cp.int32, copy=False)
        output_index = pd.MultiIndex.from_product(levels, names=list(key_columns))
    except Exception:
        return None
    return output_index, row_group_codes, key_columns


def _materialize_device_group_label_columns_with_validity(
    label_columns: tuple[Any, ...],
    valid_columns: tuple[Any | None, ...],
    *,
    operation: str,
    reason: str,
) -> tuple[list[np.ndarray], list[np.ndarray | None]]:
    from vibespatial.runtime.materialization import (
        MaterializationBoundary,
        record_materialization_event,
    )

    if not label_columns:
        return [], []
    group_count = int(getattr(label_columns[0], "size", len(label_columns[0])))
    total_bytes = 0
    for labels in label_columns:
        item_count = int(getattr(labels, "size", len(labels)))
        itemsize = int(getattr(getattr(labels, "dtype", None), "itemsize", 0))
        total_bytes += item_count * itemsize
    for valid in valid_columns:
        if valid is None:
            continue
        item_count = int(getattr(valid, "size", len(valid)))
        itemsize = int(getattr(getattr(valid, "dtype", None), "itemsize", 0))
        total_bytes += item_count * itemsize
    record_materialization_event(
        surface="vibespatial.overlay.dissolve._native_device_plain_group_index_and_codes",
        boundary=MaterializationBoundary.INTERNAL_HOST_CONVERSION,
        operation=operation,
        reason=reason,
        detail=f"groups={group_count}, columns={len(label_columns)}, bytes={total_bytes}",
        d2h_transfer=True,
        strict_disallowed=False,
    )
    host_labels = [
        overlay_device_to_host(
            labels,
            reason=(
                "vibespatial.overlay.dissolve._native_device_plain_group_index_and_codes"
                f"::{operation}"
            ),
        )
        for labels in label_columns
    ]
    host_valid = [
        None
        if valid is None
        else overlay_device_to_host(
            valid,
            reason=(
                "vibespatial.overlay.dissolve._native_device_plain_group_index_and_codes"
                f"::{operation}_validity"
            ),
        )
        for valid in valid_columns
    ]
    return host_labels, host_valid


def _materialize_device_group_label_columns(
    label_columns: tuple[Any, ...],
    *,
    operation: str,
    reason: str,
) -> list[np.ndarray]:
    from vibespatial.runtime.materialization import (
        MaterializationBoundary,
        record_materialization_event,
    )

    if not label_columns:
        return []
    group_count = int(getattr(label_columns[0], "size", len(label_columns[0])))
    total_bytes = 0
    for labels in label_columns:
        item_count = int(getattr(labels, "size", len(labels)))
        itemsize = int(getattr(getattr(labels, "dtype", None), "itemsize", 0))
        total_bytes += item_count * itemsize
    record_materialization_event(
        surface="vibespatial.overlay.dissolve._native_device_plain_group_index_and_codes",
        boundary=MaterializationBoundary.INTERNAL_HOST_CONVERSION,
        operation=operation,
        reason=reason,
        detail=f"groups={group_count}, columns={len(label_columns)}, bytes={total_bytes}",
        d2h_transfer=True,
        strict_disallowed=False,
    )
    return [
        overlay_device_to_host(
            labels,
            reason=(
                "vibespatial.overlay.dissolve._native_device_plain_group_index_and_codes"
                f"::{operation}"
            ),
        )
        for labels in label_columns
    ]


def _materialize_device_group_labels(
    labels: Any,
    *,
    operation: str,
    reason: str,
) -> np.ndarray:
    from vibespatial.runtime.materialization import (
        MaterializationBoundary,
        record_materialization_event,
    )

    item_count = int(getattr(labels, "size", len(labels)))
    itemsize = int(getattr(getattr(labels, "dtype", None), "itemsize", 0))
    record_materialization_event(
        surface="vibespatial.overlay.dissolve._native_device_plain_group_index_and_codes",
        boundary=MaterializationBoundary.INTERNAL_HOST_CONVERSION,
        operation=operation,
        reason=reason,
        detail=f"groups={item_count}, bytes={item_count * itemsize}",
        d2h_transfer=True,
        strict_disallowed=False,
    )
    return overlay_device_to_host(
        labels,
        reason=(
            "vibespatial.overlay.dissolve._native_device_plain_group_index_and_codes"
            f"::{operation}"
        ),
    )


def _native_plain_group_index_and_codes(
    key: pd.Series,
    *,
    sort: bool,
    dropna: bool,
) -> tuple[pd.Index, np.ndarray, tuple[Any, ...]] | None:
    """Encode a single non-categorical dissolve key without pandas groupby."""
    if pd.api.types.is_object_dtype(key.dtype) and not _is_admitted_object_group_key(
        key
    ):
        return None
    try:
        codes, uniques = pd.factorize(
            key,
            sort=sort,
            use_na_sentinel=dropna,
        )
        output_index = pd.Index(uniques, name=key.name)
        if pd.api.types.is_object_dtype(key.dtype) and len(output_index):
            output_index = pd.Index(output_index.tolist(), name=key.name)
    except Exception:
        return None
    if np.any(codes < 0):
        if not dropna:
            return None
        missing_codes = codes[codes < 0]
        if np.any(missing_codes != -1):
            return None
    return output_index, np.asarray(codes, dtype=np.int32), (key.name,)


def _native_multi_plain_group_index_and_codes(
    frame,
    *,
    by: tuple[Any, ...],
    observed: bool,
    sort: bool,
    dropna: bool,
) -> tuple[pd.MultiIndex, np.ndarray, tuple[Any, ...]] | None:
    """Encode admitted non-categorical multi-column dissolve keys."""
    if not by or len(set(by)) != len(by):
        return None
    if any(
        not isinstance(column, str)
        or column not in frame.columns
        or column == frame.geometry.name
        for column in by
    ):
        return None
    try:
        key_frame = frame.loc[:, list(by)]
    except Exception:
        return None
    has_categorical = any(isinstance(dtype, pd.CategoricalDtype) for dtype in key_frame.dtypes)
    object_columns = tuple(
        column
        for column, dtype in key_frame.dtypes.items()
        if pd.api.types.is_object_dtype(dtype)
    )
    if any(
        not _is_admitted_object_group_key(key_frame[column])
        for column in object_columns
    ):
        return None
    if (
        has_categorical
        and not observed
        and not dropna
        and not sort
        and bool(key_frame.isna().any().any())
    ):
        return _native_unobserved_categorical_null_multi_group_index_and_codes(
            key_frame,
            by=by,
        )

    try:
        valid_rows = ~key_frame.isna().any(axis=1) if dropna else pd.Series(True, index=key_frame.index)
        valid_mask = valid_rows.to_numpy(dtype=bool, na_value=False)
        admitted_frame = key_frame.loc[valid_rows]
        row_index = _native_multi_key_row_index(key_frame)
        observed_index = row_index[valid_mask].drop_duplicates()
        if sort:
            observed_index = observed_index.sort_values()
        if has_categorical and not observed:
            levels = []
            for column in by:
                series = admitted_frame[column]
                if isinstance(series.dtype, pd.CategoricalDtype):
                    category = series.array
                    level_values = category.categories
                    if not dropna and bool(series.isna().any()):
                        level_values = pd.Categorical(
                            list(category.categories) + [np.nan],
                            categories=category.categories,
                            ordered=category.ordered,
                        )
                    levels.append(
                        pd.CategoricalIndex(
                            level_values,
                            categories=category.categories,
                            ordered=category.ordered,
                            name=column,
                        )
                    )
                    continue
                levels.append(_native_multi_key_level(series, sort=sort))
            output_index = pd.MultiIndex.from_product(levels, names=list(by))
            if not sort:
                remaining = output_index[observed_index.get_indexer(output_index) < 0]
                output_index = observed_index.append(remaining)
        else:
            output_index = observed_index
        codes = output_index.get_indexer(row_index)
    except Exception:
        return None

    if dropna:
        codes = np.where(valid_mask, codes, -1)
    if np.any(codes < 0):
        missing_codes = codes[codes < 0]
        if not dropna or np.any(missing_codes != -1):
            return None
    return output_index, np.asarray(codes, dtype=np.int32), by


def _native_unobserved_categorical_null_multi_group_index_and_codes(
    key_frame: pd.DataFrame,
    *,
    by: tuple[Any, ...],
) -> tuple[pd.MultiIndex, np.ndarray, tuple[Any, ...]] | None:
    """Encode pandas' sort=False unobserved categorical-null product."""
    try:
        from pandas.core.sorting import (
            compress_group_index,
            decons_obs_group_ids,
            get_group_index,
        )

        levels: list[pd.Index] = []
        codes: list[np.ndarray] = []
        observed_flags: list[bool] = []
        names = list(by)
        for column in by:
            series = key_frame[column]
            raw_codes, uniques = pd.factorize(
                series,
                sort=False,
                use_na_sentinel=False,
            )
            if isinstance(series.dtype, pd.CategoricalDtype):
                category = series.array
                values = list(uniques)
                present = {value for value in values if not pd.isna(value)}
                values.extend(
                    value for value in category.categories if value not in present
                )
                level = pd.CategoricalIndex(
                    pd.Categorical(
                        values,
                        categories=category.categories,
                        ordered=category.ordered,
                    ),
                    name=column,
                )
                observed_flags.append(False)
            else:
                level = pd.Index(uniques, name=column)
                observed_flags.append(True)
            levels.append(level)
            codes.append(np.asarray(raw_codes, dtype=np.intp))

        observed_indices = [
            index for index, observed in enumerate(observed_flags) if observed
        ]
        unobserved_indices = [
            index for index, observed in enumerate(observed_flags) if not observed
        ]
        if not observed_indices or not unobserved_indices:
            return None

        observed_index, observed_ids = _native_observed_index_and_ids(
            levels=[levels[index] for index in observed_indices],
            codes=[codes[index] for index in observed_indices],
            names=[names[index] for index in observed_indices],
            sort=False,
            get_group_index=get_group_index,
            compress_group_index=compress_group_index,
            decons_obs_group_ids=decons_obs_group_ids,
        )
        unobserved_index, unobserved_ids = _native_unobserved_index_and_ids(
            levels=[levels[index] for index in unobserved_indices],
            codes=[codes[index] for index in unobserved_indices],
            names=[names[index] for index in unobserved_indices],
            get_group_index=get_group_index,
        )

        result_index_codes = np.concatenate(
            [
                np.tile(unobserved_index.codes, len(observed_index)),
                np.repeat(observed_index.codes, len(unobserved_index), axis=1),
            ],
            axis=0,
        )
        _, level_order = np.unique(
            unobserved_indices + observed_indices,
            return_index=True,
        )
        result_index = pd.MultiIndex(
            levels=list(unobserved_index.levels) + list(observed_index.levels),
            codes=result_index_codes,
            names=list(unobserved_index.names) + list(observed_index.names),
        ).reorder_levels(level_order)

        row_group_codes = len(unobserved_index) * observed_ids + unobserved_ids
        row_group_codes, observed_group_positions = compress_group_index(
            row_group_codes,
            sort=False,
        )
        take_order = np.concatenate(
            [
                observed_group_positions,
                np.delete(
                    np.arange(len(result_index), dtype=np.intp),
                    observed_group_positions,
                ),
            ]
        )
        return (
            result_index.take(take_order),
            np.asarray(row_group_codes, dtype=np.int32),
            by,
        )
    except Exception:
        return None


def _native_observed_index_and_ids(
    *,
    levels: list[pd.Index],
    codes: list[np.ndarray],
    names: list[Any],
    sort: bool,
    get_group_index,
    compress_group_index,
    decons_obs_group_ids,
) -> tuple[pd.MultiIndex, np.ndarray]:
    shape = tuple(len(level) for level in levels)
    group_index = get_group_index(codes, shape, sort=True, xnull=True)
    observed_ids, observed_group_ids = compress_group_index(group_index, sort=sort)
    index_codes = decons_obs_group_ids(
        observed_ids,
        observed_group_ids,
        shape,
        codes,
        xnull=True,
    )
    return (
        pd.MultiIndex(
            levels=levels,
            codes=index_codes,
            names=names,
            verify_integrity=False,
        ),
        np.asarray(observed_ids, dtype=np.intp),
    )


def _native_unobserved_index_and_ids(
    *,
    levels: list[pd.Index],
    codes: list[np.ndarray],
    names: list[Any],
    get_group_index,
) -> tuple[pd.MultiIndex, np.ndarray]:
    shape = tuple(len(level) for level in levels)
    row_group_ids = get_group_index(codes, shape, sort=True, xnull=True)
    return (
        pd.MultiIndex.from_product(levels, names=names),
        np.asarray(row_group_ids, dtype=np.intp),
    )


def _is_admitted_object_group_key(series: pd.Series) -> bool:
    """Return whether an object key has scalar pandas group semantics."""
    if not pd.api.types.is_object_dtype(series.dtype):
        return True
    values = series[~series.isna()]
    if values.empty:
        return True
    inferred = pd.api.types.infer_dtype(values, skipna=True)
    return inferred in {
        "boolean",
        "floating",
        "integer",
        "mixed-integer",
        "mixed-integer-float",
        "string",
        "unicode",
    }


def _native_multi_key_array(series: pd.Series):
    if pd.api.types.is_object_dtype(series.dtype):
        return pd.Index(series.tolist(), name=series.name)
    return series


def _native_multi_key_level(series: pd.Series, *, sort: bool) -> pd.Index:
    if pd.api.types.is_object_dtype(series.dtype):
        level = pd.Index(series.drop_duplicates().tolist(), name=series.name)
    else:
        level = pd.Index(series.drop_duplicates(), name=series.name)
    if sort:
        level = level.sort_values()
    return level


def _native_multi_key_row_index(key_frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays(
        [
            _native_multi_key_array(key_frame[column])
            for column in key_frame.columns
        ],
        names=list(key_frame.columns),
    )


def _native_categorical_group_index_and_codes(
    key: pd.Series,
    *,
    observed: bool,
    sort: bool,
    dropna: bool,
) -> tuple[pd.CategoricalIndex, np.ndarray, tuple[Any, ...]]:
    category = key.array
    raw_codes = np.asarray(category.codes, dtype=np.int32)
    category_count = len(category.categories)
    category_codes = np.arange(category_count, dtype=np.int32)
    valid_codes = raw_codes[raw_codes >= 0]
    include_null = bool(not dropna and np.any(raw_codes < 0))
    if observed:
        if sort:
            observed_mask = np.zeros(category_count, dtype=bool)
            observed_mask[valid_codes] = True
            output_codes = category_codes[observed_mask]
            if include_null:
                output_codes = np.concatenate(
                    [output_codes, np.asarray([-1], dtype=np.int32)],
                )
        elif valid_codes.size:
            first_seen: list[int] = []
            seen: set[int] = set()
            for raw_code in raw_codes:
                code = int(raw_code)
                if code < 0 and not include_null:
                    continue
                if code not in seen:
                    seen.add(code)
                    first_seen.append(code)
            output_codes = np.asarray(first_seen, dtype=np.int32)
        elif include_null:
            output_codes = np.asarray([-1], dtype=np.int32)
        else:
            output_codes = np.asarray([], dtype=np.int32)
    elif sort:
        output_codes = category_codes
        if include_null:
            output_codes = np.concatenate(
                [output_codes, np.asarray([-1], dtype=np.int32)],
            )
    else:
        observed_mask = np.zeros(category_count, dtype=bool)
        first_seen = []
        seen: set[int] = set()
        for raw_code in raw_codes:
            code = int(raw_code)
            if code < 0 and not include_null:
                continue
            if code not in seen:
                seen.add(code)
                first_seen.append(code)
                if code >= 0:
                    observed_mask[code] = True
        output_codes = np.concatenate(
            [
                np.asarray(first_seen, dtype=np.int32),
                category_codes[~observed_mask],
            ],
        ).astype(np.int32, copy=False)

    code_to_position = np.full(category_count, -1, dtype=np.int32)
    category_output_mask = output_codes >= 0
    category_output_codes = output_codes[category_output_mask]
    if category_output_codes.size:
        code_to_position[category_output_codes] = np.flatnonzero(
            category_output_mask,
        ).astype(np.int32, copy=False)
    null_positions = np.flatnonzero(output_codes < 0)
    null_position = int(null_positions[0]) if null_positions.size else None
    row_group_codes = np.full(raw_codes.shape, -1, dtype=np.int32)
    valid_mask = raw_codes >= 0
    row_group_codes[valid_mask] = code_to_position[raw_codes[valid_mask]]
    if null_position is not None:
        row_group_codes[raw_codes < 0] = null_position
    output_index = pd.CategoricalIndex(
        pd.Categorical.from_codes(
            output_codes.astype(np.int64, copy=False),
            categories=category.categories,
            ordered=category.ordered,
        ),
        name=key.name,
    )
    return output_index, row_group_codes, (key.name,)


def _native_dissolve_reducers(
    *,
    aggfunc,
    agg_kwargs: dict[str, Any],
    data_columns: pd.Index,
    group_key_columns: tuple[Any, ...],
) -> dict[Any, str] | None:
    if agg_kwargs or len(set(data_columns)) != len(data_columns):
        return None
    if isinstance(aggfunc, str):
        reducer = aggfunc.lower()
        if reducer not in _NATIVE_GROUPED_ATTRIBUTE_REDUCERS:
            return None
        return {
            column: reducer
            for column in data_columns
            if column not in group_key_columns
        }
    if not isinstance(aggfunc, Mapping):
        return None

    reducers: dict[Any, str] = {}
    known_columns = set(data_columns)
    for column, reducer in aggfunc.items():
        if column not in known_columns or not isinstance(reducer, str):
            return None
        normalized = reducer.lower()
        if normalized not in _NATIVE_GROUPED_ATTRIBUTE_REDUCERS:
            return None
        reducers[column] = normalized
    return reducers


def _native_attribute_table_for_dissolve(frame, data: pd.DataFrame) -> NativeAttributeTable:
    from vibespatial.api._native_state import get_native_state

    state = get_native_state(frame)
    if state is not None:
        return NativeAttributeTable.from_value(state.attributes)
    return NativeAttributeTable(dataframe=data)


def _reduce_native_grouped_dissolve_attributes(
    attributes: NativeAttributeTable,
    grouped: NativeGrouped,
    reducers: Mapping[Any, str],
) -> tuple[NativeGroupedAttributeReduction | NativeAttributeTable, bool] | None:
    numeric_columns = attributes.numeric_column_arrays(tuple(reducers))
    if numeric_columns is not None:
        return grouped.reduce_numeric_columns(numeric_columns, reducers), False
    device_take = attributes.grouped_device_take_columns(grouped, reducers)
    if device_take is not None:
        return device_take, True

    reduced_columns = {}
    used_take_reducer = False
    for column, reducer in reducers.items():
        numeric_column = attributes.numeric_column_arrays((column,))
        if numeric_column is not None:
            reduced_columns[column] = grouped.reduce_numeric(
                numeric_column[column],
                reducer,
            )
            continue
        if reducer in _NATIVE_GROUPED_TAKE_REDUCERS:
            take_column = attributes.host_column_series((column,))
            if take_column is not None:
                reduced_columns[column] = grouped.reduce_take(
                    take_column[column],
                    reducer,
                )
                used_take_reducer = True
                continue
        return None

    return (
        NativeGroupedAttributeReduction(
            columns=reduced_columns,
            group_count=grouped.resolved_group_count,
            output_index_plan=grouped.output_index_plan,
        ),
        used_take_reducer,
    )


def _try_prepare_native_grouped_dissolve_attributes(
    frame,
    data: pd.DataFrame,
    *,
    by,
    aggfunc,
    level,
    observed: bool,
    sort: bool,
    dropna: bool,
    normalized_method: DissolveUnionMethod = DissolveUnionMethod.UNARY,
    grid_size: float | None = None,
    allow_device_key_codes: bool = False,
    agg_kwargs: dict[str, Any],
) -> tuple[NativeAttributeTable, Any] | None:
    group_contract = _native_dissolve_group_index_and_codes(
        frame,
        by=by,
        level=level,
        observed=observed,
        sort=sort,
        dropna=dropna,
        normalized_method=normalized_method,
        grid_size=grid_size,
        allow_device_key_codes=allow_device_key_codes,
    )
    if group_contract is None:
        return None
    output_index, row_group_codes, group_key_columns = group_contract
    reducers = _native_dissolve_reducers(
        aggfunc=aggfunc,
        agg_kwargs=agg_kwargs,
        data_columns=data.columns,
        group_key_columns=group_key_columns,
    )
    if reducers is None:
        return None

    attributes = _native_attribute_table_for_dissolve(frame, data)
    grouped = NativeGrouped.from_dense_codes(
        row_group_codes,
        group_count=len(output_index),
        output_index=output_index,
    )
    reduced_result = _reduce_native_grouped_dissolve_attributes(
        attributes,
        grouped,
        reducers,
    )
    if reduced_result is None:
        return None
    reduced, used_take_reducer = reduced_result
    reduced_is_device = (
        reduced.device_table is not None
        if isinstance(reduced, NativeAttributeTable)
        else reduced.is_device
    )
    record_dispatch_event(
        surface="geopandas.geodataframe.dissolve",
        operation="dissolve_attribute_aggregation",
        implementation=(
            "native_grouped_attribute_reducers"
            if used_take_reducer
            else "native_grouped_numeric_reducers"
        ),
        reason="admitted dissolve attributes reduced through NativeGrouped",
        detail=(
            f"rows={len(frame)}, groups={len(output_index)}, "
            f"columns={len(reducers)}, reducers={sorted(set(reducers.values()))}"
        ),
        selected=ExecutionMode.GPU if reduced_is_device else ExecutionMode.CPU,
    )
    reduced_attributes = (
        reduced
        if isinstance(reduced, NativeAttributeTable)
        else reduced.to_native_attribute_table()
    )
    return reduced_attributes, row_group_codes


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


def _host_known_observed_valid_count(
    owned: OwnedGeometryArray,
    row_group_codes: Any,
    *,
    group_count: int,
) -> int | None:
    if _is_device_array(row_group_codes):
        return None
    host_codes = np.asarray(row_group_codes, dtype=np.int32)
    if int(host_codes.size) != int(owned.row_count):
        return None
    observed = (host_codes >= 0) & (host_codes < int(group_count))
    if not bool(np.any(observed)):
        return 0
    cached = owned._current_cached_validity_mask()
    if cached is not None and int(cached.size) == int(owned.row_count):
        return int(np.count_nonzero(observed & np.asarray(cached, dtype=bool)))
    if getattr(owned, "_validity", None) is not None:
        validity = np.asarray(owned._validity, dtype=bool)
        if int(validity.size) == int(owned.row_count):
            return int(np.count_nonzero(observed & validity))
    return None


def _row_code_count(row_group_codes: Any) -> int:
    shape = getattr(row_group_codes, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[0])
    size = getattr(row_group_codes, "size", None)
    if size is not None:
        return int(size)
    return len(row_group_codes)


def _is_device_array(value: Any) -> bool:
    return hasattr(value, "__cuda_array_interface__")


def _materialize_device_group_codes(row_group_codes: Any) -> np.ndarray:
    from vibespatial.runtime.materialization import (
        MaterializationBoundary,
        record_materialization_event,
    )

    rows = _row_code_count(row_group_codes)
    itemsize = int(getattr(getattr(row_group_codes, "dtype", None), "itemsize", 0))
    record_materialization_event(
        surface="vibespatial.overlay.dissolve.execute_grouped_union_codes",
        boundary=MaterializationBoundary.INTERNAL_HOST_CONVERSION,
        operation="device_group_codes_to_host",
        reason="device dense group codes were materialized for host-only grouped union fallback",
        detail=f"rows={rows}, bytes={rows * itemsize}",
        d2h_transfer=True,
        strict_disallowed=True,
    )
    if cp is None:
        raise RuntimeError("CuPy is required to materialize device group codes")
    return overlay_device_to_host(
        row_group_codes,
        reason="overlay dissolve device group-code host fallback export",
        dtype=np.int32,
    )


def _tag_grouped_convex_hull_source(
    reduced: OwnedGeometryArray,
    *,
    ordered_owned: OwnedGeometryArray,
    offsets: np.ndarray,
) -> None:
    """Attach source groups for the dissolve -> convex_hull physical rewrite."""
    if not provenance_rewrites_enabled():
        return
    try:
        from vibespatial.runtime.residency import Residency

        if ordered_owned.residency is not Residency.DEVICE:
            return
        reduced._grouped_convex_hull_source = (
            ordered_owned,
            np.asarray(offsets, dtype=np.int64).copy(),
        )
    except Exception:
        logger.debug("failed to tag grouped convex-hull provenance", exc_info=True)


def _group_positions_from_codes(row_group_codes: np.ndarray, group_count: int) -> list[np.ndarray]:
    return [
        np.flatnonzero(row_group_codes == group_index).astype(np.int32, copy=False)
        for group_index in range(group_count)
    ]


def _owned_supports_polygonal_grouped_union(owned: OwnedGeometryArray) -> bool:
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import FAMILY_TAGS

    polygonal_families = {
        GeometryFamily.POLYGON,
        GeometryFamily.MULTIPOLYGON,
    }
    if owned.device_state is not None:
        return set(owned.device_state.families).issubset(polygonal_families)

    valid_tags = owned.tags[owned.validity]
    polygon_tags = np.asarray(
        [
            FAMILY_TAGS[GeometryFamily.POLYGON],
            FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
        ],
        dtype=valid_tags.dtype if valid_tags.size else np.int8,
    )
    return bool(valid_tags.size == 0 or np.all(np.isin(valid_tags, polygon_tags)))


def _execute_device_native_grouped_union(
    grouped: NativeGrouped,
    *,
    method: DissolveUnionMethod,
    owned: OwnedGeometryArray,
) -> GroupedUnionResult | None:
    """Reduce device `NativeGrouped` polygon rows without host group assembly.

    Physical shape: device sorted row order + device grouped offsets -> one
    grouped overlay union. Native input carriers are `NativeGrouped` and
    `OwnedGeometryArray`; native output is `GroupedUnionResult` with an owned
    device geometry payload for downstream `NativeTabularResult` lowering.
    """
    from vibespatial.runtime.residency import Residency

    if cp is None or owned.residency is not Residency.DEVICE:
        return None
    if method is not DissolveUnionMethod.UNARY:
        return None
    if not grouped.is_device:
        return None
    if grouped.sorted_order is None or grouped.group_offsets is None or grouped.group_ids is None:
        return None
    if not _owned_supports_polygonal_grouped_union(owned):
        return None

    group_count = grouped.resolved_group_count
    if group_count < 0:
        return None

    sorted_order = cp.asarray(grouped.sorted_order, dtype=cp.int64)
    if int(sorted_order.size) != owned.row_count:
        return None

    group_offsets = cp.asarray(grouped.group_offsets, dtype=cp.int64)
    group_ids = cp.asarray(grouped.group_ids, dtype=cp.int64)
    if int(group_offsets.size) != int(group_ids.size) + 1:
        return None

    all_rows_valid = _owned_all_rows_valid_host_proof(owned)
    ordered_owned = owned.take(sorted_order)
    total_rows = int(sorted_order.size)
    if total_rows == 0:
        from vibespatial.constructive.binary_constructive import (
            _empty_device_constructive_output,
        )

        empty = _empty_device_constructive_output(group_count)
        return GroupedUnionResult(
            geometries=None,
            group_count=group_count,
            non_empty_groups=0,
            empty_groups=group_count,
            method=method,
            owned=empty,
        )

    positions = cp.arange(total_rows, dtype=cp.int64)
    compact_group_positions = cp.searchsorted(
        group_offsets[1:],
        positions,
        side="right",
    ).astype(cp.int64, copy=False)
    source_rows = group_ids[compact_group_positions].astype(cp.int32, copy=False)

    cached_validity = ordered_owned._current_cached_validity_mask()
    if all_rows_valid:
        valid_count = total_rows
        valid_mask = None
    elif cached_validity is not None and int(cached_validity.size) == total_rows:
        valid_count = int(np.count_nonzero(cached_validity))
        valid_mask = None
    else:
        state = ordered_owned._ensure_device_state()
        valid_mask = cp.asarray(state.validity)[:total_rows].astype(
            cp.bool_,
            copy=False,
        )
        valid_count = overlay_int_scalar(
            cp.count_nonzero(valid_mask),
            reason="overlay dissolve native grouped-union valid-row count fence",
        )
    if valid_count == 0:
        from vibespatial.constructive.binary_constructive import (
            _empty_device_constructive_output,
        )

        empty = _empty_device_constructive_output(group_count)
        return GroupedUnionResult(
            geometries=None,
            group_count=group_count,
            non_empty_groups=0,
            empty_groups=group_count,
            method=method,
            owned=empty,
        )
    if valid_count == total_rows:
        valid_owned = ordered_owned
        valid_source_rows = source_rows
    else:
        if valid_mask is None:
            valid_mask = cp.asarray(cached_validity, dtype=cp.bool_)
        valid_positions = cp.flatnonzero(valid_mask).astype(cp.int64, copy=False)
        valid_owned = ordered_owned.take(valid_positions)
        valid_source_rows = source_rows[valid_positions]

    from vibespatial.constructive.binary_constructive import (
        _regroup_intersection_parts_with_grouped_union_gpu,
    )

    try:
        reduced = _regroup_intersection_parts_with_grouped_union_gpu(
            valid_owned,
            valid_source_rows,
            output_row_count=group_count,
            dispatch_mode=ExecutionMode.GPU,
        )
    except Exception:
        logger.debug("device native grouped union failed", exc_info=True)
        return None
    if reduced is None or reduced.row_count != group_count:
        return None
    from vibespatial.geometry.owned import seed_all_validity_cache

    seed_all_validity_cache(reduced)

    if valid_count == total_rows:
        non_empty_groups = int(group_ids.size)
    else:
        counts = cp.bincount(valid_source_rows, minlength=group_count)
        non_empty_groups = overlay_int_scalar(
            cp.count_nonzero(counts > 0),
            reason="overlay dissolve native grouped-union nonempty-group count fence",
        )
    record_dispatch_event(
        surface="vibespatial.overlay.dissolve.execute_native_grouped_union",
        operation="grouped_union",
        implementation="native_grouped_device_overlay_union",
        reason="device NativeGrouped rows reduced without host group-code assembly",
        detail=(
            f"rows={owned.row_count}, groups={group_count}, "
            f"non_empty_groups={non_empty_groups}"
        ),
        requested=ExecutionMode.GPU,
        selected=ExecutionMode.GPU,
    )
    return GroupedUnionResult(
        geometries=None,
        group_count=group_count,
        non_empty_groups=non_empty_groups,
        empty_groups=group_count - non_empty_groups,
        method=method,
        owned=reduced,
    )


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
) -> _GroupedGeometryResult:
    """Repair invalid grouped-union outputs while preserving the native seam."""
    from vibespatial.api.geoseries import GeoSeries
    from vibespatial.constructive.make_valid_pipeline import make_valid_owned
    from vibespatial.constructive.validity import is_valid_owned
    from vibespatial.geometry.owned import (
        concat_owned_scatter,
        from_shapely_geometries,
        seed_all_validity_cache,
    )

    repair_needed = _device_grouped_union_repair_needed(owned)
    if repair_needed is False:
        seed_all_validity_cache(owned)
        return _GroupedGeometryResult(
            GeometryNativeResult.from_owned(owned, crs=crs),
            repaired=False,
        )

    if owned.device_state is not None:
        requested_mode = ExecutionMode.GPU if strict_native_mode_enabled() else ExecutionMode.AUTO
        try:
            mv_result = make_valid_owned(owned=owned, dispatch_mode=requested_mode)
        except Exception:
            if strict_native_mode_enabled():
                raise
        else:
            if mv_result.owned is not None and mv_result.selected is ExecutionMode.GPU:
                seed_all_validity_cache(mv_result.owned)
                return _GroupedGeometryResult(
                    GeometryNativeResult.from_owned(mv_result.owned, crs=crs),
                    repaired=bool(mv_result.repaired_rows.size),
                )
            if mv_result.owned is None:
                return _GroupedGeometryResult(
                    GeometryNativeResult.from_geoseries(
                        GeoSeries(
                            _canonicalize_polygonal_make_valid_values(
                                np.asarray(mv_result.geometries, dtype=object)
                            ),
                            name=geometry_name,
                            crs=crs,
                        ),
                    ),
                    repaired=bool(mv_result.repaired_rows.size),
                )

    invalid_mask = ~np.asarray(is_valid_owned(owned), dtype=bool)
    if invalid_mask.any() and not bool(np.all(owned.validity)):
        invalid_mask = invalid_mask.copy()
        invalid_mask[~owned.validity] = False
    if not invalid_mask.any():
        seed_all_validity_cache(owned)
        return _GroupedGeometryResult(
            GeometryNativeResult.from_owned(owned, crs=crs),
            repaired=False,
        )

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
        if remaining_invalid.any() and not bool(np.all(repaired_owned.validity)):
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
        return _GroupedGeometryResult(
            GeometryNativeResult.from_owned(repaired_owned, crs=crs),
            repaired=True,
        )

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
    return _GroupedGeometryResult(
        GeometryNativeResult.from_geoseries(
            GeoSeries(
                _canonicalize_polygonal_make_valid_values(
                    np.asarray(mv_result.geometries, dtype=object)
                ),
                name=geometry_name,
                crs=crs,
            ),
        ),
        repaired=True,
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
                        _skip_polygon_contraction=True,
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
            return _tree_reduce_global(merged, "union", skip_polygon_contraction=True)
        except Exception:
            logger.debug(
                "tiny partial tree reduction failed; falling back to generic union_all",
                exc_info=True,
            )
    return union_all_gpu_owned(
        merged,
        dispatch_mode=ExecutionMode.GPU,
        _skip_polygon_contraction=True,
    )


def _reduce_buffered_line_polygons_gpu(buffered: OwnedGeometryArray) -> OwnedGeometryArray:
    """Reduce line-buffer polygons without the generic union_all heuristics.

    Buffered-line dissolve has already proven the physical shape: one
    device-resident source-line group reduced into one corridor. The generic
    union_all entrypoint spends extra time on bbox decomposition/color probes
    and may choose the contraction overlay path, both of which are poor fits
    for small dense corridor masks.
    """
    from vibespatial.constructive.union_all import (
        _spatially_localize_polygon_union_inputs,
        _tree_reduce_global,
    )

    if buffered.row_count <= 1:
        return buffered
    buffered = _spatially_localize_polygon_union_inputs(buffered)
    return _tree_reduce_global(
        buffered,
        "union",
        skip_polygon_contraction=True,
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

    polygon_buffer = device_state.families[GeometryFamily.POLYGON]
    if polygon_buffer.ring_offsets is None:
        return None

    d_geom_starts = cp.asarray(polygon_buffer.geometry_offsets[:-1]).astype(cp.int32, copy=False)
    d_geom_ends = cp.asarray(polygon_buffer.geometry_offsets[1:]).astype(cp.int32, copy=False)

    d_ring_offsets = cp.asarray(polygon_buffer.ring_offsets).astype(cp.int32, copy=False)
    d_coord_starts = d_ring_offsets[d_geom_starts]
    d_coord_ends = d_ring_offsets[d_geom_ends]
    if not overlay_bool_scalar(
        cp.all(d_validity)
        & cp.all(d_tags == polygon_tag)
        & ~cp.any(polygon_buffer.empty_mask)
        & cp.all((d_geom_ends - d_geom_starts) == 1)
        & cp.all((d_coord_ends - d_coord_starts) == 5),
        reason="dissolve rectangle bounds structural scalar fence",
    ):
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
    if not overlay_bool_scalar(
        cp.all(
            cp.all(on_corners, axis=1)
            & closed
            & cp.isclose(twice_area, expected_twice_area, rtol=1.0e-10, atol=SPATIAL_EPSILON)
        ),
        reason="dissolve rectangle bounds axis-aligned area scalar fence",
    ):
        return None
    return d_bounds


def _host_polygon_segment_count_hint(owned: OwnedGeometryArray) -> int | None:
    """Return polygon segment count from existing structure metadata, never D2H."""
    from vibespatial.geometry.buffers import GeometryFamily

    state = owned.device_state
    if state is None or set(state.families) != {GeometryFamily.POLYGON}:
        return None
    device_buffer = state.families.get(GeometryFamily.POLYGON)
    if device_buffer is not None:
        dense_width = getattr(device_buffer, "dense_single_ring_width", None)
        if dense_width is not None:
            return int(owned.row_count) * max(int(dense_width) - 1, 0)
    buffer = owned.families.get(GeometryFamily.POLYGON)
    if buffer is None or buffer.ring_offsets is None:
        return None
    ring_offsets = np.asarray(buffer.ring_offsets, dtype=np.int64)
    if ring_offsets.ndim != 1 or ring_offsets.size < 2:
        return None
    ring_sizes = ring_offsets[1:] - ring_offsets[:-1]
    if np.any(ring_sizes < 1):
        return None
    return int(np.maximum(ring_sizes - 1, 0).sum())


def _owned_all_rows_valid_host_proof(owned: OwnedGeometryArray) -> bool:
    """Return true when existing host/cache metadata proves every row valid."""
    cached = owned._current_cached_validity_mask()
    if cached is not None and int(cached.size) == int(owned.row_count):
        return bool(np.all(cached))
    validity = getattr(owned, "_validity", None)
    if validity is not None and int(validity.size) == int(owned.row_count):
        return bool(np.all(validity))
    return False


def _group_codes_cover_all_rows_and_groups(
    row_group_codes: Any,
    *,
    group_count: int,
    row_count: int,
) -> bool:
    """Prove dense group codes have no dropped rows and observe every group."""
    if group_count < 0:
        return False
    if group_count == 0:
        return row_count == 0
    if _is_device_array(row_group_codes):
        grouped = NativeGrouped.from_dense_codes(
            row_group_codes,
            group_count=group_count,
        )
        if grouped.sorted_order is None or grouped.group_ids is None:
            return False
        return (
            int(grouped.sorted_order.size) == int(row_count)
            and int(grouped.group_ids.size) == int(group_count)
        )

    codes = np.asarray(row_group_codes, dtype=np.int32)
    if int(codes.size) != int(row_count):
        return False
    if not bool(np.all((codes >= 0) & (codes < int(group_count)))):
        return False
    return int(np.unique(codes).size) == int(group_count)


def _group_codes_observed_in_range_row_count(
    row_group_codes: Any,
    *,
    group_count: int,
    row_count: int,
) -> int | None:
    """Count rows with dense in-range group codes without materializing codes."""
    if group_count < 0:
        return None
    if _is_device_array(row_group_codes):
        grouped = NativeGrouped.from_dense_codes(
            row_group_codes,
            group_count=group_count,
        )
        if grouped.sorted_order is None:
            return None
        observed_count = int(grouped.sorted_order.size)
        if observed_count > int(row_count):
            return None
        return observed_count

    codes = np.asarray(row_group_codes, dtype=np.int32)
    if int(codes.size) != int(row_count):
        return None
    observed = (codes >= 0) & (codes < int(group_count))
    return int(np.count_nonzero(observed))


def _execute_low_fan_in_all_valid_coverage_union_gpu_owned_codes(
    row_group_codes: Any,
    *,
    group_count: int,
    owned: OwnedGeometryArray,
) -> GroupedUnionResult | None:
    """Exact grouped coverage union for low-fan-in all-valid device codes.

    Physical shape: `NativeGrouped` proves observed device groups, an existing
    all-valid cache proves row validity, and host polygon offsets provide only
    an allocation size hint.  The reducer itself stays segment/group shaped on
    device and avoids scalar D2H admissions before the exact coverage union.
    """
    if cp is None or not _is_device_array(row_group_codes):
        return None
    if group_count <= 0 or int(owned.row_count) == 0:
        return None
    if int(owned.row_count) > (2 * int(group_count)):
        return None
    if not _owned_all_rows_valid_host_proof(owned):
        return None
    if not _owned_supports_polygonal_grouped_union(owned):
        return None

    total_segments_hint = _host_polygon_segment_count_hint(owned)
    if total_segments_hint is None:
        return None

    if not _group_codes_cover_all_rows_and_groups(
        row_group_codes,
        group_count=group_count,
        row_count=owned.row_count,
    ):
        return None

    from vibespatial.constructive.binary_constructive import (
        _dispatch_grouped_polygon_known_coverage_union_gpu,
    )

    reduced = _dispatch_grouped_polygon_known_coverage_union_gpu(
        owned,
        cp.asarray(row_group_codes, dtype=cp.int32),
        output_row_count=group_count,
        dispatch_mode=ExecutionMode.GPU,
        assume_all_valid=True,
        assume_source_rows_valid=True,
        total_segments_hint=total_segments_hint,
    )
    if reduced is None or reduced.row_count != int(group_count):
        return None
    from vibespatial.geometry.owned import seed_all_validity_cache

    seed_all_validity_cache(reduced)
    record_dispatch_event(
        surface="vibespatial.overlay.dissolve.execute_grouped_union_codes",
        operation="grouped_coverage_union",
        implementation="native_low_fan_in_grouped_coverage_union",
        reason="low-fan-in all-valid device coverage groups reduced without scalar admissions",
        detail=f"rows={owned.row_count}, groups={group_count}",
        requested=ExecutionMode.GPU,
        selected=ExecutionMode.GPU,
    )
    return GroupedUnionResult(
        geometries=None,
        group_count=int(group_count),
        non_empty_groups=int(group_count),
        empty_groups=0,
        method=DissolveUnionMethod.COVERAGE,
        owned=reduced,
    )


def execute_grouped_box_union_gpu_owned_codes(
    row_group_codes: Any,
    *,
    group_count: int,
    owned: OwnedGeometryArray,
) -> GroupedUnionResult | None:
    if cp is None:
        return None
    if _row_code_count(row_group_codes) != int(owned.row_count):
        raise ValueError("row_group_codes length must match owned geometry row count")

    try:
        d_bounds = _owned_rectangle_bounds_device(owned)
    except RuntimeError:
        return None
    if d_bounds is None:
        return None

    d_codes = cp.asarray(row_group_codes, dtype=cp.int32)
    d_observed_mask = d_codes >= 0
    if not overlay_bool_scalar(
        cp.any(d_observed_mask),
        reason="dissolve grouped-box observed-row scalar fence",
    ):
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
    x_order = cp.lexsort(
        cp.stack(
            [
                d_observed_bounds[:, 0],
                d_observed_codes.astype(cp.float64, copy=False),
            ],
        ),
    )
    x_sorted_codes = d_observed_codes[x_order]
    x_sorted_bounds = d_observed_bounds[x_order]
    same_group = x_sorted_codes[1:] == x_sorted_codes[:-1]
    d_x_contiguous = cp.all(
        (~same_group)
        | cp.isclose(
            x_sorted_bounds[:-1, 2],
            x_sorted_bounds[1:, 0],
            rtol=1.0e-12,
            atol=SPATIAL_EPSILON,
        )
    )
    y_order = cp.lexsort(
        cp.stack(
            [
                d_observed_bounds[:, 1],
                d_observed_codes.astype(cp.float64, copy=False),
            ],
        ),
    )
    y_sorted_codes = d_observed_codes[y_order]
    y_sorted_bounds = d_observed_bounds[y_order]
    same_group = y_sorted_codes[1:] == y_sorted_codes[:-1]
    d_y_contiguous = cp.all(
        (~same_group)
        | cp.isclose(
            y_sorted_bounds[:-1, 3],
            y_sorted_bounds[1:, 1],
            rtol=1.0e-12,
            atol=SPATIAL_EPSILON,
        )
    )
    d_horizontal_strip = cp.all(d_full_height) & d_x_contiguous
    d_vertical_strip = cp.all(d_full_width) & d_y_contiguous
    if not overlay_bool_scalar(
        cp.all(d_counts[:group_count] > 0)
        & cp.all(cp.isclose(d_area_sum, d_bbox_area, rtol=1.0e-12, atol=SPATIAL_EPSILON))
        & (d_horizontal_strip | d_vertical_strip),
        reason="dissolve grouped-box strip-coverage scalar fence",
    ):
        return None

    d_group_bounds = cp.stack((d_xmin, d_ymin, d_xmax, d_ymax), axis=1)
    reduced = _build_device_boxes_from_bounds(d_group_bounds, row_count=group_count)
    from vibespatial.geometry.owned import seed_all_validity_cache

    seed_all_validity_cache(reduced)

    return GroupedUnionResult(
        geometries=None,
        group_count=group_count,
        non_empty_groups=group_count,
        empty_groups=0,
        method=DissolveUnionMethod.COVERAGE,
        owned=reduced,
    )


def _regular_grid_rectangles_are_strictly_group_disjoint_device(
    d_bounds,
    d_codes,
    *,
    group_count: int,
) -> bool:
    """Certify sparse regular-grid rectangle groups for MultiPolygon assembly.

    The shape is conservative: all observed rectangles must be equal-size
    cells on a fp64 grid, each cell id must be unique, and no horizontal or
    vertical neighbor may share a dissolve group. Under those constraints the
    input coverage has no same-group touching/overlap, so grouped disjoint
    assembly is exact without all-pairs checks.
    """
    row_count = int(getattr(d_bounds, "shape", (0,))[0])
    if row_count <= 1:
        return False
    d_codes = cp.asarray(d_codes, dtype=cp.int32)

    d_widths = d_bounds[:, 2] - d_bounds[:, 0]
    d_heights = d_bounds[:, 3] - d_bounds[:, 1]
    d_width = d_widths[0]
    d_height = d_heights[0]
    if not overlay_bool_scalar(
        cp.all((d_codes >= 0) & (d_codes < int(group_count)))
        & (d_width > 0.0)
        & (d_height > 0.0),
        reason="dissolve regular-grid domain-positive-size scalar fence",
    ):
        return False

    d_min_x = cp.min(d_bounds[:, 0])
    d_min_y = cp.min(d_bounds[:, 1])
    d_ix_float = (d_bounds[:, 0] - d_min_x) / d_width
    d_iy_float = (d_bounds[:, 1] - d_min_y) / d_height
    d_ix = cp.rint(d_ix_float).astype(cp.int64, copy=False)
    d_iy = cp.rint(d_iy_float).astype(cp.int64, copy=False)
    d_uniform_cell_size = (
        cp.all(cp.isclose(d_widths, d_width, rtol=1.0e-12, atol=SPATIAL_EPSILON))
        & cp.all(cp.isclose(d_heights, d_height, rtol=1.0e-12, atol=SPATIAL_EPSILON))
    )
    d_integral_cell_index = (
        cp.all(
            cp.isclose(
                d_ix_float,
                d_ix.astype(cp.float64, copy=False),
                rtol=1.0e-12,
                atol=SPATIAL_EPSILON,
            )
        )
        & cp.all(
            cp.isclose(
                d_iy_float,
                d_iy.astype(cp.float64, copy=False),
                rtol=1.0e-12,
                atol=SPATIAL_EPSILON,
            )
        )
    )

    d_cols = cp.max(d_ix) + cp.asarray(1, dtype=cp.int64)
    d_linear = d_iy * d_cols + d_ix
    order = cp.argsort(d_linear)
    d_sorted_linear = d_linear[order]
    d_sorted_ix = d_ix[order]
    d_sorted_iy = d_iy[order]
    d_sorted_codes = d_codes[order]

    d_unique_cells = cp.all(d_sorted_linear[1:] > d_sorted_linear[:-1])

    d_horizontal_same_group = (
        (d_sorted_iy[1:] == d_sorted_iy[:-1])
        & (d_sorted_ix[1:] == (d_sorted_ix[:-1] + 1))
        & (d_sorted_codes[1:] == d_sorted_codes[:-1])
    )

    def _has_same_group_target(d_targets, d_target_valid):
        d_positions = cp.searchsorted(d_sorted_linear, d_targets)
        d_found = d_target_valid & (d_positions < row_count)
        d_safe_positions = cp.minimum(
            d_positions,
            cp.asarray(row_count - 1, dtype=d_positions.dtype),
        )
        d_same_group = (
            d_found
            & (d_sorted_linear[d_safe_positions] == d_targets)
            & (d_sorted_codes[d_safe_positions] == d_sorted_codes)
        )
        return cp.any(d_same_group)

    d_has_vertical_same_group = _has_same_group_target(
        d_sorted_linear + d_cols,
        cp.ones(row_count, dtype=cp.bool_),
    )
    d_has_diagonal_left_same_group = _has_same_group_target(
        d_sorted_linear + d_cols - 1,
        d_sorted_ix > 0,
    )
    d_has_diagonal_right_same_group = _has_same_group_target(
        d_sorted_linear + d_cols + 1,
        d_sorted_ix < (d_cols - 1),
    )
    if not overlay_bool_scalar(
        d_uniform_cell_size
        & d_integral_cell_index
        & d_unique_cells
        & ~cp.any(d_horizontal_same_group)
        & ~d_has_vertical_same_group
        & ~d_has_diagonal_left_same_group
        & ~d_has_diagonal_right_same_group,
        reason="dissolve regular-grid disjoint-neighbor scalar fence",
    ):
        return False
    return True


def execute_grouped_disjoint_subset_union_gpu_owned_codes(
    row_group_codes: Any,
    *,
    group_count: int,
    owned: OwnedGeometryArray,
    method: DissolveUnionMethod = DissolveUnionMethod.DISJOINT_SUBSET,
) -> GroupedUnionResult | None:
    """Assemble strictly disjoint polygon groups into MultiPolygon rows.

    This path performs no topology union.  It is exact only when every group is
    already valid and every pair of polygons in a group has strictly separated
    bounding boxes.  Touching or overlapping groups decline to the exact
    fallback instead of returning adjacent parts as an invalid MultiPolygon.
    """
    if cp is None:
        return None
    if _row_code_count(row_group_codes) != int(owned.row_count):
        raise ValueError("row_group_codes length must match owned geometry row count")
    if owned.device_state is None:
        return None

    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import (
        FAMILY_TAGS,
        DeviceFamilyGeometryBuffer,
        build_device_resident_owned,
        seed_all_validity_cache,
        seed_homogeneous_host_metadata,
    )

    device_state = owned._ensure_device_state()
    if set(device_state.families) != {GeometryFamily.POLYGON}:
        return None
    cached_validity = getattr(owned, "_cached_is_valid_mask", None)
    cached_validity_ok = (
        cached_validity is not None
        and int(getattr(cached_validity, "size", 0)) == int(owned.row_count)
        and bool(np.all(cached_validity))
    )

    polygon_tag = np.int8(FAMILY_TAGS[GeometryFamily.POLYGON])
    d_validity = cp.asarray(device_state.validity).astype(cp.bool_, copy=False)
    d_tags = cp.asarray(device_state.tags).astype(cp.int8, copy=False)

    polygon_buffer = device_state.families[GeometryFamily.POLYGON]
    if polygon_buffer.ring_offsets is None:
        return None

    d_codes = cp.asarray(row_group_codes, dtype=cp.int32)
    d_observed_mask = (d_codes >= 0) & (d_codes < int(group_count))

    d_rows = cp.arange(int(owned.row_count), dtype=cp.int64)
    d_observed_rows = d_rows[d_observed_mask]
    d_observed_codes = d_codes[d_observed_mask]
    d_counts = cp.bincount(d_observed_codes, minlength=int(group_count))[
        : int(group_count)
    ].astype(cp.int32, copy=False)
    # Preserve exact GeoPandas geometry typing by declining singleton or empty
    # groups for now.  Mixed Polygon/MultiPolygon grouped output is a separate
    # scatter problem; this bulk path owns only multi-member groups.
    d_admissible = cp.all(d_validity) & cp.all(d_tags == polygon_tag)
    if polygon_buffer.empty_mask is not None:
        d_admissible &= ~cp.any(polygon_buffer.empty_mask)
    d_admissible &= cp.any(d_observed_mask)
    d_admissible &= cp.all(d_counts > 1)
    if not overlay_bool_scalar(
        d_admissible,
        reason="dissolve disjoint-subset admissibility scalar fence",
    ):
        return None

    d_bounds = None
    small_pairwise_groups = overlay_bool_scalar(
        cp.all(d_counts <= 8),
        reason="dissolve disjoint-subset small-pairwise scalar fence",
    )
    if not small_pairwise_groups:
        try:
            d_all_bounds = _owned_rectangle_bounds_device(owned)
        except RuntimeError:
            d_all_bounds = None
        if d_all_bounds is None:
            return None
        d_bounds = d_all_bounds[d_observed_mask]
        if not _regular_grid_rectangles_are_strictly_group_disjoint_device(
            d_bounds,
            d_observed_codes,
            group_count=group_count,
        ):
            return None
    elif not cached_validity_ok:
        return None

    order = cp.lexsort(
        cp.stack(
            [
                d_observed_rows,
                d_observed_codes.astype(cp.int64, copy=False),
            ],
        ),
    )
    d_sorted_rows = d_observed_rows[order].astype(cp.int64, copy=False)
    sorted_owned = owned.take(d_sorted_rows)
    sorted_state = sorted_owned._ensure_device_state()
    sorted_polygon = sorted_state.families.get(GeometryFamily.POLYGON)
    if sorted_polygon is None or sorted_polygon.ring_offsets is None:
        return None

    d_group_offsets = cp.concatenate(
        [
            cp.asarray([0], dtype=cp.int32),
            cp.cumsum(d_counts, dtype=cp.int32),
        ],
    )
    if small_pairwise_groups:
        from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device

        sorted_bounds = cp.asarray(
            compute_geometry_bounds_device(sorted_owned),
            dtype=cp.float64,
        )
        max_group_size = 8
        slot_ids = cp.arange(max_group_size, dtype=cp.int32)
        slot_valid = slot_ids[None, :] < d_counts[:, None]
        slot_positions = d_group_offsets[:-1, None] + slot_ids[None, :]
        slot_positions = cp.minimum(
            slot_positions,
            cp.asarray(max(int(sorted_owned.row_count) - 1, 0), dtype=cp.int32),
        )
        slot_bounds = sorted_bounds[slot_positions]
        a = slot_bounds[:, :, None, :]
        b = slot_bounds[:, None, :, :]
        pair_valid = (
            slot_valid[:, :, None]
            & slot_valid[:, None, :]
            & (slot_ids[None, :, None] < slot_ids[None, None, :])
        )
        separated = (
            (a[..., 2] < b[..., 0])
            | (b[..., 2] < a[..., 0])
            | (a[..., 3] < b[..., 1])
            | (b[..., 3] < a[..., 1])
        )
        if not overlay_bool_scalar(
            cp.all((~pair_valid) | separated),
            reason="dissolve disjoint-subset pair-separation scalar fence",
        ):
            return None

    d_empty_mask = cp.zeros(int(group_count), dtype=cp.bool_)
    d_multipolygon = DeviceFamilyGeometryBuffer(
        family=GeometryFamily.MULTIPOLYGON,
        x=sorted_polygon.x,
        y=sorted_polygon.y,
        geometry_offsets=d_group_offsets,
        empty_mask=d_empty_mask,
        part_offsets=sorted_polygon.geometry_offsets,
        ring_offsets=sorted_polygon.ring_offsets,
    )
    multipolygon_tag = np.int8(FAMILY_TAGS[GeometryFamily.MULTIPOLYGON])
    reduced = build_device_resident_owned(
        device_families={GeometryFamily.MULTIPOLYGON: d_multipolygon},
        row_count=int(group_count),
        tags=cp.full(int(group_count), multipolygon_tag, dtype=cp.int8),
        validity=cp.ones(int(group_count), dtype=cp.bool_),
        family_row_offsets=cp.arange(int(group_count), dtype=cp.int32),
        execution_mode="gpu",
    )
    seed_homogeneous_host_metadata(reduced, GeometryFamily.MULTIPOLYGON)
    seed_all_validity_cache(reduced)

    return GroupedUnionResult(
        geometries=None,
        group_count=int(group_count),
        non_empty_groups=int(group_count),
        empty_groups=0,
        method=method,
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

    from vibespatial.constructive.union_all import coverage_union_all_gpu_owned

    group_owned_results: list[OwnedGeometryArray] = []
    observed_group_ids: list[int] = []
    non_empty_groups = 0
    empty_groups = 0

    for group_index, positions in enumerate(group_positions):
        positions = np.asarray(positions, dtype=np.intp)
        if positions.size == 0:
            empty_groups += 1
            continue
        non_empty_groups += 1
        observed_group_ids.append(group_index)
        if positions.size == 1:
            single = owned.take(positions)
            group_owned_results.append(single)
            continue
        group_owned = owned.take(positions)
        reduced = coverage_union_all_gpu_owned(group_owned)
        if reduced is None:
            return None
        group_owned_results.append(reduced)

    native_result = _grouped_owned_results_to_native_grouped_union_result(
        group_owned_results,
        observed_group_ids,
        group_count=len(group_positions),
        non_empty_groups=non_empty_groups,
        empty_groups=empty_groups,
        method=DissolveUnionMethod.COVERAGE,
    )
    if native_result is not None:
        return native_result

    merged = np.full(len(group_positions), _EMPTY, dtype=object)
    for group_index, group_owned in zip(
        observed_group_ids,
        group_owned_results,
        strict=True,
    ):
        group_geoms = group_owned.to_shapely()
        merged[group_index] = group_geoms[0] if group_geoms else _EMPTY
    return GroupedUnionResult(
        geometries=merged,
        group_count=len(group_positions),
        non_empty_groups=non_empty_groups,
        empty_groups=empty_groups,
        method=DissolveUnionMethod.COVERAGE,
    )


def _grouped_owned_results_to_native_grouped_union_result(
    group_owned_results: list[OwnedGeometryArray],
    observed_group_ids: list[int],
    *,
    group_count: int,
    non_empty_groups: int,
    empty_groups: int,
    method: DissolveUnionMethod,
) -> GroupedUnionResult | None:
    """Assemble grouped owned reductions into full grouped output rows.

    Physical shape: observed grouped reductions plus sparse group ids -> full
    grouped constructive output.  When sparse outputs are device-resident, empty
    rows are prebuilt as device empty polygons and observed rows are scattered
    into place without a host geometry array.
    """
    if len(group_owned_results) != len(observed_group_ids):
        return None
    if non_empty_groups != len(observed_group_ids):
        return None
    if int(group_count) != non_empty_groups + empty_groups:
        return None

    from vibespatial.geometry.owned import seed_all_validity_cache
    from vibespatial.runtime.residency import Residency

    if not group_owned_results:
        if cp is None:
            return None
        owned_result = _empty_polygon_rows_device(group_count)
        seed_all_validity_cache(owned_result)
        return GroupedUnionResult(
            geometries=None,
            group_count=group_count,
            non_empty_groups=non_empty_groups,
            empty_groups=empty_groups,
            method=method,
            owned=owned_result,
        )

    if (
        empty_groups == 0
        and observed_group_ids == list(range(group_count))
    ):
        owned_result = type(group_owned_results[0]).concat(group_owned_results)
        seed_all_validity_cache(owned_result)
        return GroupedUnionResult(
            geometries=None,
            group_count=group_count,
            non_empty_groups=non_empty_groups,
            empty_groups=empty_groups,
            method=method,
            owned=owned_result,
        )

    if cp is None:
        return None
    if any(result.residency is not Residency.DEVICE for result in group_owned_results):
        return None

    replacement = type(group_owned_results[0]).concat(group_owned_results)
    if replacement.row_count != len(observed_group_ids):
        return None

    from vibespatial.geometry.owned import concat_owned_scatter

    owned_result = concat_owned_scatter(
        _empty_polygon_rows_device(group_count),
        replacement,
        np.asarray(observed_group_ids, dtype=np.int64),
    )
    seed_all_validity_cache(owned_result)
    return GroupedUnionResult(
        geometries=None,
        group_count=group_count,
        non_empty_groups=non_empty_groups,
        empty_groups=empty_groups,
        method=method,
        owned=owned_result,
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

    from vibespatial.constructive.union_all import disjoint_subset_union_all_owned

    group_owned_results: list[OwnedGeometryArray] = []
    observed_group_ids: list[int] = []
    non_empty_groups = 0
    empty_groups = 0

    for group_index, positions in enumerate(group_positions):
        positions = np.asarray(positions, dtype=np.intp)
        if positions.size == 0:
            empty_groups += 1
            continue
        non_empty_groups += 1
        observed_group_ids.append(group_index)
        if positions.size == 1:
            single = owned.take(positions)
            group_owned_results.append(single)
            continue
        group_owned = owned.take(positions)
        reduced = disjoint_subset_union_all_owned(group_owned)
        if reduced is None:
            return None
        group_owned_results.append(reduced)

    native_result = _grouped_owned_results_to_native_grouped_union_result(
        group_owned_results,
        observed_group_ids,
        group_count=len(group_positions),
        non_empty_groups=non_empty_groups,
        empty_groups=empty_groups,
        method=DissolveUnionMethod.DISJOINT_SUBSET,
    )
    if native_result is not None:
        return native_result

    merged = np.full(len(group_positions), _EMPTY, dtype=object)
    for group_index, group_owned in zip(
        observed_group_ids,
        group_owned_results,
        strict=True,
    ):
        group_geoms = group_owned.to_shapely()
        merged[group_index] = group_geoms[0] if group_geoms else _EMPTY
    return GroupedUnionResult(
        geometries=merged,
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


def _empty_polygon_rows_device(row_count: int) -> OwnedGeometryArray:
    """Build valid empty polygon rows without leaving the device."""
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        raise RuntimeError("CuPy is required for device empty polygon rows")

    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import (
        FAMILY_TAGS,
        DeviceFamilyGeometryBuffer,
        build_device_resident_owned,
        seed_homogeneous_host_metadata,
    )

    row_count = int(row_count)
    polygon_tag = np.int8(FAMILY_TAGS[GeometryFamily.POLYGON])
    result = build_device_resident_owned(
        device_families={
            GeometryFamily.POLYGON: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.POLYGON,
                x=cp.empty(0, dtype=cp.float64),
                y=cp.empty(0, dtype=cp.float64),
                geometry_offsets=cp.zeros(row_count + 1, dtype=cp.int32),
                empty_mask=cp.ones(row_count, dtype=cp.bool_),
                ring_offsets=cp.zeros(1, dtype=cp.int32),
                bounds=None,
            )
        },
        row_count=row_count,
        tags=cp.full(row_count, polygon_tag, dtype=cp.int8),
        validity=cp.ones(row_count, dtype=cp.bool_),
        family_row_offsets=cp.arange(row_count, dtype=cp.int32),
        execution_mode="gpu",
    )
    seed_homogeneous_host_metadata(result, GeometryFamily.POLYGON)
    return result


def _scatter_empty_polygon_groups_device(
    reduced: OwnedGeometryArray,
    counts,
) -> OwnedGeometryArray:
    """Replace unobserved grouped-output rows with valid empty polygon rows."""
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return reduced

    empty_rows = cp.flatnonzero(cp.asarray(counts) == 0).astype(cp.int64, copy=False)
    if int(empty_rows.size) == 0:
        return reduced

    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import (
        FAMILY_TAGS,
        DeviceFamilyGeometryBuffer,
        build_device_resident_owned,
    )

    state = reduced._ensure_device_state()
    n_empty = int(empty_rows.size)
    polygon = GeometryFamily.POLYGON
    polygon_tag = np.int8(FAMILY_TAGS[polygon])

    out_validity = cp.asarray(state.validity, dtype=cp.bool_).copy()
    out_tags = cp.asarray(state.tags, dtype=cp.int8).copy()
    out_family_rows = cp.asarray(state.family_row_offsets, dtype=cp.int32).copy()

    polygon_buffer = state.families.get(polygon)
    if polygon_buffer is None:
        polygon_row_count = 0
        new_polygon_buffer = DeviceFamilyGeometryBuffer(
            family=polygon,
            x=cp.empty(0, dtype=cp.float64),
            y=cp.empty(0, dtype=cp.float64),
            geometry_offsets=cp.zeros(n_empty + 1, dtype=cp.int32),
            empty_mask=cp.ones(n_empty, dtype=cp.bool_),
            ring_offsets=cp.zeros(1, dtype=cp.int32),
            bounds=None,
        )
    else:
        polygon_row_count = int(polygon_buffer.geometry_offsets.size) - 1
        last_geom_offset = polygon_buffer.geometry_offsets[-1:]
        appended_geom_offsets = cp.repeat(last_geom_offset, n_empty)
        new_bounds = None
        if polygon_buffer.bounds is not None:
            empty_bounds = cp.full((n_empty, 4), cp.nan, dtype=cp.float64)
            new_bounds = cp.concatenate((polygon_buffer.bounds, empty_bounds))
        new_polygon_buffer = DeviceFamilyGeometryBuffer(
            family=polygon,
            x=polygon_buffer.x,
            y=polygon_buffer.y,
            geometry_offsets=cp.concatenate(
                (polygon_buffer.geometry_offsets, appended_geom_offsets)
            ),
            empty_mask=cp.concatenate(
                (polygon_buffer.empty_mask, cp.ones(n_empty, dtype=cp.bool_))
            ),
            ring_offsets=polygon_buffer.ring_offsets,
            bounds=new_bounds,
        )

    out_validity[empty_rows] = True
    out_tags[empty_rows] = polygon_tag
    out_family_rows[empty_rows] = cp.arange(
        polygon_row_count,
        polygon_row_count + n_empty,
        dtype=cp.int32,
    )
    device_families = dict(state.families)
    device_families[polygon] = new_polygon_buffer
    result = build_device_resident_owned(
        device_families=device_families,
        row_count=reduced.row_count,
        tags=out_tags,
        validity=out_validity,
        family_row_offsets=out_family_rows,
        execution_mode="gpu",
    )
    result.runtime_history.extend(reduced.runtime_history)
    return result


def execute_grouped_coverage_edge_union_gpu_owned_codes(
    row_group_codes,
    *,
    group_count: int,
    owned: OwnedGeometryArray,
) -> GroupedUnionResult | None:
    """Topology-free grouped coverage dissolve from device native carriers."""
    if cp is None:
        return None
    from vibespatial.runtime.residency import Residency

    if owned.residency is not Residency.DEVICE:
        return None
    if group_count < 0 or not _owned_supports_polygonal_grouped_union(owned):
        return None
    if int(row_group_codes.size) != owned.row_count:
        return None

    if group_count == 0:
        from vibespatial.constructive.binary_constructive import (
            _empty_device_constructive_output,
        )

        return GroupedUnionResult(
            geometries=None,
            group_count=0,
            non_empty_groups=0,
            empty_groups=0,
            method=DissolveUnionMethod.COVERAGE,
            owned=_empty_device_constructive_output(0),
        )

    host_valid_count = _host_known_observed_valid_count(
        owned,
        row_group_codes,
        group_count=group_count,
    )
    device_grouped_shape: NativeGrouped | None = None
    if _is_device_array(row_group_codes):
        device_grouped_shape = NativeGrouped.from_dense_codes(
            row_group_codes,
            group_count=group_count,
        )
        host_group_counts = (
            int(device_grouped_shape.group_ids.size),
            int(group_count) - int(device_grouped_shape.group_ids.size),
        )
    else:
        host_group_counts = _group_non_empty_counts(
            np.asarray(row_group_codes, dtype=np.int32),
            group_count,
        )
    d_codes = cp.asarray(row_group_codes, dtype=cp.int32)
    all_rows_valid = _owned_all_rows_valid_host_proof(owned)
    if not all_rows_valid:
        observed_in_range_count = None
    elif device_grouped_shape is not None:
        observed_in_range_count = int(device_grouped_shape.sorted_order.size)
    else:
        observed_in_range_count = _group_codes_observed_in_range_row_count(
            row_group_codes,
            group_count=group_count,
            row_count=owned.row_count,
        )
    all_valid_all_rows = (
        observed_in_range_count is not None
        and int(observed_in_range_count) == int(owned.row_count)
    )
    all_valid_all_groups = all_valid_all_rows and host_group_counts[0] == int(group_count)
    total_segments_hint = (
        _host_polygon_segment_count_hint(owned)
        if all_valid_all_rows
        else None
    )
    state = owned._ensure_device_state()
    d_valid = cp.asarray(state.validity, dtype=cp.bool_)
    d_observed_valid = d_valid & (d_codes >= 0) & (d_codes < np.int32(group_count))
    if observed_in_range_count is not None:
        valid_count = int(observed_in_range_count)
    elif all_valid_all_groups:
        valid_count = int(owned.row_count)
    elif host_valid_count is not None:
        valid_count = host_valid_count
    else:
        valid_count = overlay_int_scalar(
            cp.count_nonzero(d_observed_valid),
            reason="overlay dissolve grouped coverage-edge valid-row count fence",
        )
    if valid_count == 0:
        reduced = _empty_polygon_rows_device(group_count)
        return GroupedUnionResult(
            geometries=None,
            group_count=group_count,
            non_empty_groups=0,
            empty_groups=group_count,
            method=DissolveUnionMethod.COVERAGE,
            owned=reduced,
        )

    if valid_count == owned.row_count:
        valid_owned = owned
        valid_source_rows = d_codes
    else:
        valid_positions = cp.flatnonzero(d_observed_valid).astype(cp.int64, copy=False)
        valid_owned = owned.take(valid_positions)
        valid_source_rows = d_codes[valid_positions]
        if total_segments_hint is None:
            total_segments_hint = _host_polygon_segment_count_hint(valid_owned)

    from vibespatial.constructive.binary_constructive import (
        _dispatch_grouped_polygon_known_coverage_union_gpu,
    )

    reduced = _dispatch_grouped_polygon_known_coverage_union_gpu(
        valid_owned,
        valid_source_rows,
        output_row_count=group_count,
        dispatch_mode=ExecutionMode.GPU,
        assume_all_valid=True,
        assume_source_rows_valid=True,
        total_segments_hint=total_segments_hint,
    )
    if reduced is None or reduced.row_count != group_count:
        return None

    if all_valid_all_groups:
        counts = None
        non_empty_groups = int(group_count)
    else:
        counts = cp.bincount(valid_source_rows, minlength=group_count)
        non_empty_groups = (
            host_group_counts[0]
            if host_group_counts is not None
            else overlay_int_scalar(
                cp.count_nonzero(counts > 0),
                reason="overlay dissolve grouped coverage-edge nonempty-group count fence",
            )
        )
    if counts is not None and non_empty_groups != group_count:
        reduced = _scatter_empty_polygon_groups_device(reduced, counts)
    from vibespatial.geometry.owned import seed_all_validity_cache

    seed_all_validity_cache(reduced)
    record_dispatch_event(
        surface="vibespatial.overlay.dissolve.execute_grouped_union_codes",
        operation="grouped_coverage_union",
        implementation="native_grouped_coverage_edge_union",
        reason="coverage dissolve eliminated shared edges by device group without host geometry assembly",
        detail=(
            f"rows={owned.row_count}, groups={group_count}, "
            f"non_empty_groups={non_empty_groups}"
        ),
        requested=ExecutionMode.GPU,
        selected=ExecutionMode.GPU,
    )
    return GroupedUnionResult(
        geometries=None,
        group_count=group_count,
        non_empty_groups=non_empty_groups,
        empty_groups=group_count - non_empty_groups,
        method=DissolveUnionMethod.COVERAGE,
        owned=reduced,
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
    row_group_codes_device = _is_device_array(row_group_codes)
    values: np.ndarray | None = None

    def _values() -> np.ndarray:
        nonlocal values
        if values is None:
            values = np.asarray(geometries, dtype=object)
        return values

    geometry_count = int(owned.row_count) if owned is not None else int(len(geometries))
    if geometry_count != row_group_codes.size:
        raise ValueError("row_group_codes length must match geometries length")

    if (
        owned is not None
        and normalized is DissolveUnionMethod.UNARY
        and grid_size is None
        and _owned_supports_polygonal_grouped_union(owned)
    ):
        native_grouped = NativeGrouped.from_dense_codes(
            row_group_codes,
            group_count=group_count,
        )
        accelerated = execute_native_grouped_union(
            native_grouped,
            _geometries=geometries,
            method=normalized,
            grid_size=grid_size,
            owned=owned,
        )
        if accelerated is not None:
            return accelerated

    if (
        owned is not None
        and normalized is DissolveUnionMethod.COVERAGE
        and grid_size is None
    ):
        accelerated = _execute_low_fan_in_all_valid_coverage_union_gpu_owned_codes(
            row_group_codes,
            group_count=group_count,
            owned=owned,
        )
        if accelerated is not None:
            return accelerated
        if geometry_count <= 2 * max(int(group_count), 1):
            accelerated = execute_grouped_coverage_edge_union_gpu_owned_codes(
                row_group_codes,
                group_count=group_count,
                owned=owned,
            )
            if accelerated is not None:
                return accelerated
        accelerated = execute_grouped_box_union_gpu_owned_codes(
            row_group_codes,
            group_count=group_count,
            owned=owned,
        )
        if accelerated is not None:
            return accelerated

    if (
        owned is not None
        and normalized
        in {
            DissolveUnionMethod.COVERAGE,
            DissolveUnionMethod.DISJOINT_SUBSET,
        }
        and grid_size is None
    ):
        accelerated = execute_grouped_disjoint_subset_union_gpu_owned_codes(
            row_group_codes,
            group_count=group_count,
            owned=owned,
            method=normalized,
        )
        if accelerated is not None:
            return accelerated

    if (
        owned is not None
        and normalized is DissolveUnionMethod.COVERAGE
        and grid_size is None
    ):
        accelerated = execute_grouped_coverage_edge_union_gpu_owned_codes(
            row_group_codes,
            group_count=group_count,
            owned=owned,
        )
        if accelerated is not None:
            return accelerated

    if row_group_codes_device:
        row_group_codes = _materialize_device_group_codes(row_group_codes)

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


def execute_native_grouped_union(
    grouped: NativeGrouped,
    *,
    _geometries: Sequence[object | None] | np.ndarray,
    method: DissolveUnionMethod | str = DissolveUnionMethod.UNARY,
    grid_size: float | None = None,
    owned: OwnedGeometryArray | None = None,
) -> GroupedUnionResult | None:
    """Execute admitted grouped geometry union from a `NativeGrouped` carrier."""
    normalized = method if isinstance(method, DissolveUnionMethod) else DissolveUnionMethod(method)
    if owned is None or grid_size is not None:
        return None
    if normalized is DissolveUnionMethod.COVERAGE:
        accelerated = execute_grouped_box_union_gpu_owned_codes(
            grouped.group_codes,
            group_count=grouped.resolved_group_count,
            owned=owned,
        )
        if accelerated is not None:
            return accelerated
        return execute_grouped_coverage_edge_union_gpu_owned_codes(
            grouped.group_codes,
            group_count=grouped.resolved_group_count,
            owned=owned,
        )
    if grouped.is_device:
        return _execute_device_native_grouped_union(
            grouped,
            method=normalized,
            owned=owned,
        )
    if (
        normalized is not DissolveUnionMethod.UNARY
        or not _owned_supports_polygonal_grouped_union(owned)
    ):
        return None

    group_count = grouped.resolved_group_count
    group_ids = np.asarray(grouped.group_ids, dtype=np.int32)
    offsets = np.asarray(grouped.group_offsets, dtype=np.int32)
    if group_ids.size == 0:
        merged = np.full(group_count, _EMPTY, dtype=object)
        return GroupedUnionResult(
            geometries=merged,
            group_count=group_count,
            non_empty_groups=0,
            empty_groups=group_count,
            method=normalized,
        )

    from vibespatial.kernels.constructive.segmented_union import segmented_union_all

    sorted_rows = np.asarray(grouped.sorted_order, dtype=np.int64)
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
        group_count=int(group_ids.size),
    )
    non_empty_groups = int(group_ids.size)
    empty_groups = int(group_count - non_empty_groups)
    if group_ids.size == group_count and np.array_equal(
        group_ids,
        np.arange(group_count, dtype=group_ids.dtype),
    ):
        _tag_grouped_convex_hull_source(
            reduced,
            ordered_owned=ordered_owned,
            offsets=offsets,
        )
        return GroupedUnionResult(
            geometries=None,
            group_count=group_count,
            non_empty_groups=non_empty_groups,
            empty_groups=empty_groups,
            method=normalized,
            owned=reduced,
        )

    from vibespatial.runtime.residency import Residency

    if reduced.residency is Residency.DEVICE and cp is not None:
        from vibespatial.geometry.owned import concat_owned_scatter

        full_reduced = concat_owned_scatter(
            _empty_polygon_rows_device(group_count),
            reduced,
            group_ids.astype(np.int64, copy=False),
        )
        return GroupedUnionResult(
            geometries=None,
            group_count=group_count,
            non_empty_groups=non_empty_groups,
            empty_groups=empty_groups,
            method=normalized,
            owned=full_reduced,
        )

    reduced_geometries = np.asarray(reduced.to_shapely(), dtype=object)
    merged = np.full(group_count, _EMPTY, dtype=object)
    merged[group_ids.astype(np.intp, copy=False)] = reduced_geometries
    return GroupedUnionResult(
        geometries=merged,
        group_count=group_count,
        non_empty_groups=non_empty_groups,
        empty_groups=empty_groups,
        method=normalized,
    )


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
    grid_size: float | None,
    allow_device_key_codes: bool = True,
    agg_kwargs: dict[str, Any],
) -> tuple[
    pd.DataFrame | NativeAttributeTable,
    list[np.ndarray],
    OwnedGeometryArray | None,
    Any | None,
]:
    groupby_by = by
    if groupby_by is None and level is None:
        groupby_by = np.zeros(len(frame), dtype="int64")

    groupby_kwargs = {
        "by": groupby_by,
        "level": level,
        "sort": sort,
        "observed": observed,
        "dropna": dropna,
    }

    data = frame.drop(labels=frame.geometry.name, axis=1)
    native_prepared = _try_prepare_native_grouped_dissolve_attributes(
        frame,
        data,
        by=by,
        aggfunc=aggfunc,
        level=level,
        observed=observed,
        sort=sort,
        dropna=dropna,
        normalized_method=normalized_method,
        grid_size=grid_size,
        allow_device_key_codes=allow_device_key_codes,
        agg_kwargs=agg_kwargs,
    )
    if native_prepared is None:
        aggregated_data = data.groupby(**groupby_kwargs).agg(aggfunc, **agg_kwargs)
        aggregated_data.columns = aggregated_data.columns.to_flat_index()
        row_group_codes = _build_row_group_codes(
            frame,
            by=groupby_by,
            level=level,
            aggregated_index=aggregated_data.index,
        )
    else:
        aggregated_data, row_group_codes = native_prepared

    if row_group_codes is None:
        grouped_geometry = frame.groupby(group_keys=False, **groupby_kwargs)[frame.geometry.name]
        indices_items = list(grouped_geometry.indices.items())
        group_positions = _normalize_group_positions(aggregated_data.index, indices_items)
    else:
        group_positions = []
    owned = getattr(frame.geometry.values, "_owned", None)
    if owned is None:
        try:
            from vibespatial.api._native_state import get_native_state

            state = get_native_state(frame)
        except Exception:
            state = None
        if state is not None:
            state_owned = getattr(state.geometry, "owned", None)
            if state_owned is not None and int(state_owned.row_count) == int(len(frame)):
                owned = state_owned
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


def _owned_supports_polygonal_coverage_certification(owned: OwnedGeometryArray) -> bool:
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import FAMILY_TAGS

    polygonal_families = {
        GeometryFamily.POLYGON,
        GeometryFamily.MULTIPOLYGON,
    }
    if owned.device_state is not None:
        return set(owned.device_state.families).issubset(polygonal_families)

    valid_tags = owned.tags[owned.validity]
    if valid_tags.size == 0:
        return False
    polygon_tags = np.asarray(
        [
            FAMILY_TAGS[GeometryFamily.POLYGON],
            FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
        ],
        dtype=valid_tags.dtype,
    )
    return bool(np.all(np.isin(valid_tags, polygon_tags)))


def _certify_grouped_polygon_coverage_gpu(
    owned: OwnedGeometryArray,
    row_group_codes: np.ndarray,
    *,
    group_count: int,
) -> bool | None:
    """Return True when GPU evidence proves unary dissolve can use coverage.

    Coverage union is equivalent to unary union only when members in each
    dissolve group have no positive-area overlap.  This certification keeps
    candidate generation and pair gathering on device, then uses row-wise GPU
    intersections for same-group candidate pairs.  A ``None`` result means the
    certifier could not prove safety cheaply, so callers must keep exact unary.
    """
    if cp is None:
        return None
    from vibespatial.runtime import has_gpu_runtime
    from vibespatial.runtime.residency import Residency

    if (
        not has_gpu_runtime()
        or owned.row_count != row_group_codes.size
        or owned.residency is not Residency.DEVICE
        or not _owned_supports_polygonal_coverage_certification(owned)
    ):
        return None

    observed_mask = row_group_codes >= 0
    if not bool(np.any(observed_mask)):
        return True
    if _max_group_size(row_group_codes, [], group_count) <= 1:
        return True

    try:
        from vibespatial.spatial.indexing import build_flat_spatial_index
        from vibespatial.spatial.query import query_spatial_index

        flat_index = build_flat_spatial_index(owned)
        query_result, execution = query_spatial_index(
            owned,
            flat_index,
            owned,
            predicate=None,
            sort=False,
            output_format="indices",
            return_metadata=True,
            return_device=True,
        )
        if execution.selected is not ExecutionMode.GPU:
            return None

        if hasattr(query_result, "d_left_idx"):
            d_left = query_result.d_left_idx
            d_right = query_result.d_right_idx
        else:
            indices = np.asarray(query_result, dtype=np.int64)
            if indices.size == 0:
                return True
            if indices.ndim != 2 or indices.shape[0] != 2:
                return None
            d_left = cp.asarray(indices[0], dtype=cp.int32)
            d_right = cp.asarray(indices[1], dtype=cp.int32)

        if int(d_left.size) == 0:
            return True

        d_codes = cp.asarray(row_group_codes, dtype=cp.int32)
        d_left_codes = d_codes[d_left]
        d_right_codes = d_codes[d_right]
        same_group = (
            (d_left < d_right)
            & (d_left_codes >= 0)
            & (d_left_codes == d_right_codes)
        )
        candidate_count = overlay_int_scalar(
            cp.count_nonzero(same_group),
            reason="overlay dissolve coverage-certification candidate-count fence",
        )
        if candidate_count == 0:
            return True
        if candidate_count > _COVERAGE_REWRITE_MAX_CANDIDATE_PAIRS:
            return None

        candidate_left = d_left[same_group].astype(cp.int64, copy=False)
        candidate_right = d_right[same_group].astype(cp.int64, copy=False)

        from vibespatial.constructive.binary_constructive import _binary_constructive_gpu
        from vibespatial.constructive.measurement import area_owned

        def _has_positive_overlap(
            left_rows,
            right_rows,
        ) -> bool | None:
            intersections = _binary_constructive_gpu(
                "intersection",
                owned.take(left_rows),
                owned.take(right_rows),
                dispatch_mode=ExecutionMode.GPU,
            )
            if intersections is None:
                return None
            overlap_areas = np.asarray(
                area_owned(intersections, dispatch_mode=ExecutionMode.GPU),
                dtype=np.float64,
            )
            finite_areas = overlap_areas[np.isfinite(overlap_areas)]
            if finite_areas.size == 0:
                return False
            return bool(np.any(finite_areas > SPATIAL_EPSILON))

        if candidate_count > _COVERAGE_REWRITE_NEGATIVE_PROBE_PAIRS:
            probe_positions = cp.linspace(
                0,
                candidate_count - 1,
                _COVERAGE_REWRITE_NEGATIVE_PROBE_PAIRS,
                dtype=cp.int64,
            )
            probe_result = _has_positive_overlap(
                candidate_left[probe_positions],
                candidate_right[probe_positions],
            )
            if probe_result is None:
                return None
            if probe_result:
                return False

        overlap_result = _has_positive_overlap(candidate_left, candidate_right)
        if overlap_result is None:
            return None
        return not overlap_result
    except Exception:
        logger.debug("grouped polygon coverage certification failed", exc_info=True)
        return None


def _maybe_rewrite_grouped_polygon_coverage_dissolve_method(
    *,
    normalized_method: DissolveUnionMethod,
    grid_size: float | None,
    row_group_codes: np.ndarray | None,
    group_count: int,
    owned: OwnedGeometryArray | None,
) -> DissolveUnionMethod:
    if (
        normalized_method is not DissolveUnionMethod.UNARY
        or grid_size is not None
        or row_group_codes is None
        or owned is None
        or int(row_group_codes.size) < OVERLAY_UNION_ALL_GPU_THRESHOLD
        or not provenance_rewrites_enabled()
    ):
        return normalized_method

    if _max_group_size(row_group_codes, [], group_count) <= 1:
        return normalized_method

    from vibespatial.runtime.residency import Residency

    if (
        owned.residency is Residency.DEVICE
        and int(row_group_codes.size) < OVERLAY_GROUPED_BOX_GPU_THRESHOLD
    ):
        # The unary->coverage certifier proves no positive-area overlap, but
        # not that shared boundaries are noded identically enough for the
        # grouped edge-elimination reducer to match exact unary union output.
        return normalized_method

    certified = _certify_grouped_polygon_coverage_gpu(
        owned,
        row_group_codes,
        group_count=group_count,
    )
    if certified is not True:
        return normalized_method

    record_rewrite_event(
        rule_name="R11_dissolve_unary_polygon_coverage_to_coverage",
        surface="geopandas.geodataframe.dissolve",
        original_operation="dissolve(method=unary)",
        rewritten_operation="dissolve(method=coverage)",
        reason="GPU certification proved grouped polygon inputs have no same-group positive-area overlaps",
        detail=f"rows={row_group_codes.size}, groups={group_count}",
    )
    return DissolveUnionMethod.COVERAGE


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
            if int(offsets.size) == lines.row_count + 1 and overlay_bool_scalar(
                cp.all((offsets[1:] - offsets[:-1]) == 2),
                reason="overlay dissolve two-point line rewrite admissibility fence",
            ):
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
                    return overlay_device_to_host(
                        unique_rows,
                        reason="overlay dissolve two-point line unique-row export",
                        dtype=np.int64,
                    )
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

    source_owned = _provenance_source_owned(tag)
    if source_owned is None:
        return None
    from vibespatial.runtime.residency import Residency

    if source_owned.residency is not Residency.DEVICE:
        return None
    if observed_rows.size != source_owned.row_count:
        source_owned = source_owned.take(observed_rows)

    from vibespatial.constructive.linestring import (
        linestring_buffer_owned_array,
        supports_two_point_linestring_buffer_fast_path,
    )
    from vibespatial.constructive.union_all import disjoint_subset_union_all_owned
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
    if unique_rows is None or unique_rows.size == 0:
        return None

    deduped = unique_rows.size < source_owned.row_count
    unique_owned = (
        source_owned.take(unique_rows.astype(np.int64, copy=False))
        if deduped
        else source_owned
    )
    # This rewrite is the device-resident exact dissolve path for simple line
    # buffers. Cap the reduced union size directly instead of routing small
    # device-backed groups through an exact host rescue.
    if unique_owned.row_count > _BUFFERED_TWO_POINT_EXACT_UNION_MAX_UNIQUE_ROWS:
        return None

    expanded_bounds = overlay_device_to_host(
        cp.asarray(compute_geometry_bounds_device(unique_owned), dtype=cp.float64),
        reason="overlay dissolve buffered-line expanded-bounds host rewrite export",
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
            reduced = _reduce_buffered_line_polygons_gpu(buffered_unique)
    else:
        buffered_unique = linestring_buffer_owned_array(
            unique_owned,
            distance_value,
            quad_segs=quad_segs,
            cap_style=cap_style,
            join_style=join_style,
            dispatch_mode=ExecutionMode.GPU,
        )
        reduced = _reduce_buffered_line_polygons_gpu(buffered_unique)

    record_rewrite_event(
        rule_name="R9_dissolve_buffered_two_point_lines_exact_union",
        surface="geopandas.geodataframe.dissolve",
        original_operation="dissolve(method=unary)",
        rewritten_operation="buffer(device_two_point_lines).union_all_gpu",
        reason="device-resident single-group buffered two-point lines dissolve rewrites to source-line buffering plus exact GPU union",
        detail=(
            f"rows={source_owned.row_count}, unique_rows={unique_owned.row_count}, "
            f"deduped={deduped}, "
            f"color_subsets={max(color_count, 0)}, "
            f"buffer_distance={distance_value}, quad_segs={quad_segs}"
        ),
    )
    record_dispatch_event(
        surface="geopandas.geodataframe.dissolve",
        operation="dissolve",
        implementation="buffered_two_point_line_exact_union_gpu",
        reason="device-resident single-group buffered-line dissolve rewrite",
        detail=(
            f"rows={source_owned.row_count}, unique_rows={unique_owned.row_count}, "
            f"deduped={deduped}, "
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

    from vibespatial.runtime.residency import Residency

    geometry_owned = getattr(frame.geometry.values, "_owned", None)
    source_owned = _provenance_source_owned(tag)
    if (
        source_owned is not None
        and source_owned.residency is Residency.DEVICE
        and source_types == frozenset({"linestring"})
        and cp is not None
    ):
        try:
            from vibespatial.constructive.linestring import (
                supports_two_point_linestring_buffer_fast_path,
            )

            quad_segs_param = tag.get_param("quad_segs", 16)
            quad_segs = 16 if quad_segs_param is None else int(quad_segs_param)
            cap_style_param = tag.get_param("cap_style", "round")
            join_style_param = tag.get_param("join_style", "round")
            cap_style = "round" if cap_style_param is None else str(cap_style_param)
            join_style = "round" if join_style_param is None else str(join_style_param)
            candidate_owned = (
                source_owned.take(observed_rows)
                if observed_rows.size != source_owned.row_count
                else source_owned
            )
            if supports_two_point_linestring_buffer_fast_path(
                candidate_owned,
                quad_segs=quad_segs,
                cap_style=cap_style,
                join_style=join_style,
                single_sided=False,
            ):
                return None
        except Exception:
            return None

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

    from vibespatial.geometry.owned import from_shapely_geometries, seed_all_validity_cache

    owned = from_shapely_geometries([merged], residency=result_residency)
    # The exact GEOS path above already canonicalizes invalid or collection
    # output before re-entering owned storage; avoid immediately re-running the
    # full OGC validity kernel in the grouped result wrapper.
    seed_all_validity_cache(owned)
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
            grid_size=grid_size,
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
        if grouped_union is None:
            normalized_method = _maybe_rewrite_grouped_polygon_coverage_dissolve_method(
                normalized_method=normalized_method,
                grid_size=grid_size,
                row_group_codes=row_group_codes,
                group_count=len(aggregated_data.index),
                owned=owned,
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
        grid_size=grid_size,
        allow_device_key_codes=False,
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
