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
_GROUPED_BOX_GPU_THRESHOLD = 50_000


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
                try:
                    left_owned = from_shapely_geometries([current[i]])
                    right_owned = from_shapely_geometries([current[i + 1]])
                    result = overlay_union_owned(left_owned, right_owned, dispatch_mode=ExecutionMode.GPU)
                    result_geoms = result.to_shapely()
                    if result_geoms:
                        next_round.append(result_geoms[0])
                    else:
                        next_round.append(_EMPTY)
                except Exception:
                    # Fallback to shapely for this pair if GPU union fails
                    next_round.append(shapely.union(current[i], current[i + 1]))
            else:
                next_round.append(current[i])
        current = next_round
    return current[0]


def execute_grouped_union(
    geometries: Sequence[object | None] | np.ndarray,
    group_positions: list[np.ndarray],
    *,
    method: DissolveUnionMethod | str = DissolveUnionMethod.UNARY,
    grid_size: float | None = None,
) -> GroupedUnionResult:
    normalized = method if isinstance(method, DissolveUnionMethod) else DissolveUnionMethod(method)
    values = np.asarray(geometries, dtype=object)
    # GPU box union fast path
    if normalized is DissolveUnionMethod.COVERAGE and int(values.size) >= _GROUPED_BOX_GPU_THRESHOLD:
        accelerated = execute_grouped_box_union_gpu(values, group_positions)
        if accelerated is not None:
            return accelerated
    # GPU polygon union via tree-reduce (ADR-0017)
    # Only use for large groups where overlay JIT overhead is amortized.
    # Per-pair overlay_union has ~200ms JIT startup; only worthwhile when
    # group has enough polygons that GPU parallelism outweighs the overhead.
    _GPU_UNION_MIN_GROUP_SIZE = 100
    use_gpu_union = (
        cp is not None
        and int(values.size) >= _GROUPED_BOX_GPU_THRESHOLD
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

        grouped_geometry = frame.groupby(group_keys=False, **groupby_kwargs)[frame.geometry.name]
        indices_items = list(grouped_geometry.indices.items())
        group_positions = _normalize_group_positions(aggregated_data.index, indices_items)
        grouped_union = execute_grouped_union(
            np.asarray(frame.geometry.array, dtype=object),
            group_positions,
            method=normalized_method,
            grid_size=grid_size,
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

_UNION_ALL_GPU_THRESHOLD = 50


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
        and row_count >= _UNION_ALL_GPU_THRESHOLD
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
            try:
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
            except Exception:
                pass  # fall through to CPU

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
    """GPU tree-reduce: union N geometries in log₂(N) rounds.

    When *return_owned* is False (default), materializes the final result to
    a Shapely geometry (single D→H).  When True, returns the single-row
    ``OwnedGeometryArray`` directly — no final D→H.

    ADR-0002: CONSTRUCTIVE class, fp64 (segment intersection precision).
    ADR-0005: Device-resident intermediates; single D→H only when needed.
    ADR-0033: Inherits overlay pipeline tiers (NVRTC + CCCL + CuPy).
    """
    from vibespatial.geometry.owned import from_shapely_geometries
    from vibespatial.runtime import ExecutionMode

    from .gpu import overlay_union_owned

    # Single bulk D→H to identify non-empty rows (1 transfer, not N).
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

    # Tree-reduce: each round halves the geometry count
    while len(current) > 1:
        next_round: list[OwnedGeometryArray] = []
        for i in range(0, len(current), 2):
            if i + 1 < len(current):
                try:
                    result = overlay_union_owned(
                        current[i], current[i + 1],
                        dispatch_mode=ExecutionMode.GPU,
                    )
                    next_round.append(result)
                except Exception:
                    left_g = current[i].to_shapely()[0]
                    right_g = current[i + 1].to_shapely()[0]
                    merged = shapely.union(left_g, right_g)
                    next_round.append(from_shapely_geometries([merged]))
            else:
                next_round.append(current[i])
        current = next_round

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
