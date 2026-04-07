from __future__ import annotations

import gc
import statistics
from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import GeometryCollection

from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.config import (
    OVERLAY_GPU_FAILURE_THRESHOLD,
    OVERLAY_GROUPED_BOX_GPU_THRESHOLD,
    OVERLAY_UNION_ALL_GPU_THRESHOLD,
)
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.fusion import IntermediateDisposition, PipelineStep, StepKind, plan_fusion

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    cp = None


if TYPE_CHECKING:
    from collections.abc import Sequence

    from vibespatial.geometry.owned import OwnedGeometryArray
    from vibespatial.runtime import ExecutionMode


_EMPTY = GeometryCollection()


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
    geometries: np.ndarray
    group_count: int
    non_empty_groups: int
    empty_groups: int
    method: DissolveUnionMethod


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

    def materialize(self):
        if self._materialized is not None:
            return self._materialized
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
        geometry_frame = type(self._frame)(
            {self._geometry_name: grouped_union.geometries},
            geometry=self._geometry_name,
            index=self._aggregated_data.index,
            crs=self._frame.crs,
        )
        aggregated = geometry_frame.join(self._aggregated_data)
        if not self._as_index:
            aggregated = aggregated.reset_index()
        self._materialized = aggregated
        return aggregated

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


def _union_block(values: np.ndarray, method: DissolveUnionMethod, grid_size: float | None) -> object:
    if values.size == 0:
        return _EMPTY
    if method is DissolveUnionMethod.COVERAGE:
        return shapely.coverage_union_all(values)
    if method is DissolveUnionMethod.DISJOINT_SUBSET:
        return shapely.disjoint_subset_union_all(values)
    return shapely.union_all(values, grid_size=grid_size)


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
    values: np.ndarray,
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

    valid_mask = np.asarray([geom is not None and not shapely.is_empty(geom) for geom in values], dtype=bool)
    if np.any(valid_mask):
        type_ids = shapely.get_type_id(values[valid_mask])
        if not np.all(np.isin(type_ids, [3, 6])):  # Polygon / MultiPolygon
            return None

    if owned is None:
        try:
            owned = getattr(values, "_owned", None)
        except Exception:
            owned = None
    if owned is None:
        return None

    from vibespatial.constructive.union_all import coverage_union_all_gpu_owned

    merged = np.empty(len(group_positions), dtype=object)
    non_empty_groups = 0
    empty_groups = 0

    for group_index, positions in enumerate(group_positions):
        positions = np.asarray(positions, dtype=np.intp)
        if positions.size == 0:
            merged[group_index] = _EMPTY
            empty_groups += 1
            continue
        non_empty_groups += 1
        if positions.size == 1:
            merged[group_index] = values[int(positions[0])]
            continue
        group_owned = owned.take(positions)
        reduced = coverage_union_all_gpu_owned(group_owned)
        reduced_geoms = reduced.to_shapely()
        merged[group_index] = reduced_geoms[0] if reduced_geoms else _EMPTY

    return GroupedUnionResult(
        geometries=merged,
        group_count=len(group_positions),
        non_empty_groups=non_empty_groups,
        empty_groups=empty_groups,
        method=DissolveUnionMethod.COVERAGE,
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
        reduced = segmented_union_all(owned.take(sorted_rows), offsets)
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
        and geometry_count >= OVERLAY_UNION_ALL_GPU_THRESHOLD
    ):
        accelerated = execute_grouped_coverage_edge_union_codes(
            _values(),
            row_group_codes,
            group_count=group_count,
        )
        if accelerated is not None:
            return accelerated
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
        and int(values.size) >= OVERLAY_UNION_ALL_GPU_THRESHOLD
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
        if row_group_codes is not None:
            grouped_union = execute_grouped_union_codes(
                frame.geometry.array,
                row_group_codes,
                group_count=len(aggregated_data.index),
                method=normalized_method,
                grid_size=grid_size,
                owned=owned,
            )
        else:
            grouped_union = None
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

        geometry_frame = type(frame)(
            {frame.geometry.name: grouped_union.geometries},
            geometry=frame.geometry.name,
            index=aggregated_data.index,
            crs=frame.crs,
        )
        aggregated = geometry_frame.join(aggregated_data)
        if not as_index:
            aggregated = aggregated.reset_index()
        return aggregated


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
