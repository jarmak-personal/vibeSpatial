from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter

import numpy as np

from vibespatial.constructive.make_valid_pipeline_cpu import (
    build_make_valid_warmup_owned,
    make_valid_cpu_baseline,
    make_valid_cpu_is_valid,
    make_valid_cpu_repair,
    values_support_owned_make_valid,
)
from vibespatial.constructive.make_valid_pipeline_kernels import (
    _VALIDITY_KERNEL_NAMES,
    _VALIDITY_KERNEL_SOURCE_FP32,
    _VALIDITY_KERNEL_SOURCE_FP64,
    _format_validity_kernel_source,
)
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.runtime import ExecutionMode, has_gpu_runtime
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.crossover import (
    PhysicalWorkEstimate,
    estimate_segment_pair_work_from_owned,
)
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.fallbacks import record_fallback_event
from vibespatial.runtime.fusion import IntermediateDisposition, PipelineStep, StepKind, plan_fusion
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass
from vibespatial.runtime.residency import Residency, combined_residency


def _owned_can_skip_host_make_valid(owned) -> bool:
    from vibespatial.geometry.buffers import GeometryFamily

    polygonal_families = {GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON}
    if not any(family in owned.families for family in polygonal_families):
        return True
    return _owned_all_polygon_rectangles(owned)


class MakeValidPrimitive(StrEnum):
    VALIDITY_MASK = "validity_mask"
    COMPACT_INVALID = "compact_invalid"
    SEGMENTIZE_INVALID = "segmentize_invalid"
    POLYGONIZE_REPAIR = "polygonize_repair"
    SCATTER_REPAIRED = "scatter_repaired"
    EMIT_GEOMETRY = "emit_geometry"


@dataclass(frozen=True)
class MakeValidStage:
    name: str
    primitive: MakeValidPrimitive
    purpose: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    cccl_mapping: tuple[str, ...]
    disposition: IntermediateDisposition
    geometry_producing: bool = False


@dataclass(frozen=True)
class MakeValidPlan:
    method: str
    keep_collapsed: bool
    stages: tuple[MakeValidStage, ...]
    fusion_steps: tuple[PipelineStep, ...]
    reason: str


@dataclass(frozen=True)
class MakeValidResult:
    row_count: int
    valid_rows: object
    repaired_rows: object
    null_rows: object
    method: str
    keep_collapsed: bool
    owned: object | None = None
    native_geometry: object | None = None
    selected: ExecutionMode = ExecutionMode.CPU
    _geometries: np.ndarray | None = None

    @property
    def geometries(self) -> np.ndarray:
        """Lazily materialize Shapely geometries from owned array (ADR-0005).

        When the GPU repair path produces a device-resident result, Shapely
        objects are only created when a caller actually accesses this property,
        avoiding a D->H transfer for callers that consume .owned directly.
        """
        if self._geometries is not None:
            return self._geometries
        if self.native_geometry is not None:
            import pandas as pd

            result = np.asarray(
                self.native_geometry.to_geoseries(
                    index=pd.RangeIndex(int(self.row_count)),
                    name="geometry",
                ),
                dtype=object,
            )
            object.__setattr__(self, "_geometries", result)
            return result
        if self.owned is not None:
            result = np.asarray(self.owned.to_shapely(), dtype=object)
            object.__setattr__(self, "_geometries", result)
            return result
        raise RuntimeError("MakeValidResult has no geometries source")

    def to_native_tabular_result(
        self,
        *,
        crs=None,
        geometry_name: str = "geometry",
        index=None,
    ):
        """Lower make-valid geometry output into the private native carrier."""
        import pandas as pd

        from vibespatial.api._native_result_core import (
            GeometryNativeResult,
            NativeAttributeTable,
            NativeGeometryProvenance,
            NativeTabularResult,
        )

        if index is None:
            index = pd.RangeIndex(int(self.row_count))
        elif len(index) != int(self.row_count):
            raise ValueError("make_valid native result index length must match row count")

        geometry_metadata = None
        if self.native_geometry is not None:
            geometry = self.native_geometry.with_crs(crs)
        elif self.owned is not None:
            geometry = GeometryNativeResult.from_owned(self.owned, crs=crs)
            from vibespatial.api._native_metadata import NativeGeometryMetadata

            geometry_metadata = NativeGeometryMetadata.from_cached_owned(self.owned)
        else:
            geometry = GeometryNativeResult.from_values(
                self.geometries,
                crs=crs,
                index=index,
                name=geometry_name,
            )

        source_rows = np.arange(int(self.row_count), dtype=np.int64)
        if self.owned is not None and self.owned.residency is Residency.DEVICE:
            try:
                import cupy as cp

                source_rows = cp.arange(int(self.row_count), dtype=cp.int64)
            except ModuleNotFoundError:
                pass
        device_result = self.owned is not None and self.owned.residency is Residency.DEVICE
        if device_result:
            try:
                import cupy as cp

                repaired_mask = cp.zeros(int(self.row_count), dtype=cp.bool_)
                repaired_mask[cp.asarray(self.repaired_rows, dtype=cp.int64)] = True
            except ModuleNotFoundError:
                repaired_mask = np.zeros(int(self.row_count), dtype=bool)
                if self.repaired_rows.size:
                    repaired_mask[np.asarray(self.repaired_rows, dtype=np.int64)] = True
        else:
            repaired_mask = np.zeros(int(self.row_count), dtype=bool)
            if self.repaired_rows.size:
                repaired_mask[np.asarray(self.repaired_rows, dtype=np.int64)] = True

        return NativeTabularResult(
            attributes=NativeAttributeTable(dataframe=pd.DataFrame(index=index)),
            geometry=geometry,
            geometry_name=geometry_name,
            column_order=(geometry_name,),
            provenance=NativeGeometryProvenance(
                operation="make_valid",
                row_count=int(self.row_count),
                source_rows=source_rows,
                repaired_mask=repaired_mask,
            ),
            geometry_metadata=geometry_metadata,
        )


@dataclass(frozen=True)
class MakeValidBenchmark:
    dataset: str
    rows: int
    repaired_rows: int
    compact_elapsed_seconds: float
    baseline_elapsed_seconds: float

    @property
    def speedup_vs_baseline(self) -> float:
        if self.compact_elapsed_seconds == 0.0:
            return float("inf")
        return self.baseline_elapsed_seconds / self.compact_elapsed_seconds


@dataclass(frozen=True)
class _DeviceValidityRows:
    valid_rows: object
    repaired_rows: object
    null_rows: object


def _make_valid_linework_native_geometry(
    source_owned,
    repaired_owned,
    repaired_rows,
    *,
    method: str,
):
    if method != "linework" or repaired_owned is None:
        return None
    if repaired_owned.residency is not Residency.DEVICE:
        return None

    from vibespatial.constructive.boundary_remnants import (
        polygon_make_valid_linework_composition_device,
    )

    return polygon_make_valid_linework_composition_device(
        source_owned,
        repaired_owned,
        repaired_rows,
        crs=None,
    )


def _try_device_validity_expression_rows(
    owned,
    *,
    row_count: int,
) -> _DeviceValidityRows | None:
    """Return compact row sets from the device validity expression.

    Physical shape: row-aligned predicate expression -> compact device rowsets.
    Full validity, compact row positions, and family metadata remain resident.
    """
    if owned.device_state is None or not has_gpu_runtime():
        return None
    try:
        import cupy as cp
    except ModuleNotFoundError:
        return None

    from vibespatial.constructive.validity import validity_expression_owned

    state = owned._ensure_device_state(preserve_indexed_view=True)
    expression = validity_expression_owned(owned)

    d_validity = cp.asarray(state.validity, dtype=cp.bool_)
    d_valid_flags = cp.asarray(expression.values, dtype=cp.bool_)
    d_repair_needed = (~d_valid_flags) & d_validity
    repaired_rows = cp.flatnonzero(d_repair_needed).astype(
        cp.int32,
        copy=False,
    )

    trusted_no_null_rows = (
        state.trusted_all_valid is True or getattr(state, "trusted_all_non_empty", None) is True
    )
    if not trusted_no_null_rows:
        cached_host_validity = getattr(owned, "_validity", None)
        trusted_no_null_rows = (
            cached_host_validity is not None
            and int(cached_host_validity.size) == int(row_count)
            and bool(np.all(cached_host_validity))
        )

    null_rows = cp.empty(0, dtype=cp.int32)
    if not trusted_no_null_rows:
        null_rows = cp.flatnonzero(~d_validity).astype(
            cp.int32,
            copy=False,
        )

    valid_rows = cp.flatnonzero(d_valid_flags & d_validity).astype(
        cp.int32,
        copy=False,
    )

    return _DeviceValidityRows(
        valid_rows=valid_rows,
        repaired_rows=repaired_rows,
        null_rows=null_rows,
    )


def _make_valid_no_repair_result_from_device_rows(
    device_rows: _DeviceValidityRows,
    *,
    owned,
    row_count: int,
    method: str,
    keep_collapsed: bool,
    dispatch_mode: ExecutionMode | str,
) -> MakeValidResult | None:
    """Return a device-resident no-op result when validity proves no repairs."""
    if device_rows.repaired_rows.size:
        return None
    if not device_rows.null_rows.size:
        owned._cached_is_valid_mask = np.ones(row_count, dtype=bool)

    record_dispatch_event(
        surface="geopandas.array.make_valid",
        operation="make_valid",
        implementation="validity_expression_no_repair",
        reason="Device validity expression proved no non-null rows require repair",
        detail=f"rows={row_count}, method={method}, repaired=0",
        requested=dispatch_mode,
        selected=ExecutionMode.GPU,
    )
    return MakeValidResult(
        row_count=row_count,
        valid_rows=device_rows.valid_rows,
        repaired_rows=np.asarray([], dtype=np.int32),
        null_rows=device_rows.null_rows,
        method=method,
        keep_collapsed=keep_collapsed,
        owned=owned,
        selected=ExecutionMode.GPU,
    )


def _try_device_validity_no_repair_result(
    owned,
    *,
    row_count: int,
    method: str,
    keep_collapsed: bool,
    dispatch_mode: ExecutionMode | str,
) -> MakeValidResult | None:
    device_rows = _try_device_validity_expression_rows(owned, row_count=row_count)
    if device_rows is None:
        return None
    return _make_valid_no_repair_result_from_device_rows(
        device_rows,
        owned=owned,
        row_count=row_count,
        method=method,
        keep_collapsed=keep_collapsed,
        dispatch_mode=dispatch_mode,
    )


def _make_valid_result_from_device_rows(
    device_rows: _DeviceValidityRows,
    *,
    owned,
    row_count: int,
    method: str,
    keep_collapsed: bool,
    dispatch_mode: ExecutionMode | str,
) -> MakeValidResult | None:
    """Repair compact invalid rows selected by device validity expression."""
    no_repair = _make_valid_no_repair_result_from_device_rows(
        device_rows,
        owned=owned,
        row_count=row_count,
        method=method,
        keep_collapsed=keep_collapsed,
        dispatch_mode=dispatch_mode,
    )
    if no_repair is not None:
        return no_repair

    gpu_result = _make_valid_gpu_repair(
        owned,
        device_rows.repaired_rows,
        method=method,
        keep_collapsed=keep_collapsed,
    )
    if gpu_result is None or gpu_result.repaired_owned is None:
        return None

    result_owned = gpu_result.repaired_owned

    record_dispatch_event(
        surface="geopandas.array.make_valid",
        operation="make_valid",
        implementation="validity_expression+gpu_repair",
        reason=(
            "Device validity expression compacted invalid rows and GPU repair "
            "scattered repaired rows without full host validity metadata"
        ),
        detail=f"rows={row_count}, method={method}, repaired={device_rows.repaired_rows.size}",
        requested=dispatch_mode,
        selected=ExecutionMode.GPU,
    )
    native_geometry = _make_valid_linework_native_geometry(
        owned,
        result_owned,
        device_rows.repaired_rows,
        method=method,
    )
    return MakeValidResult(
        row_count=row_count,
        valid_rows=device_rows.valid_rows,
        repaired_rows=device_rows.repaired_rows,
        null_rows=device_rows.null_rows,
        method=method,
        keep_collapsed=keep_collapsed,
        owned=result_owned,
        native_geometry=native_geometry,
        selected=ExecutionMode.GPU,
    )


request_nvrtc_warmup(
    [
        ("make-valid-detect-fp64", _VALIDITY_KERNEL_SOURCE_FP64, _VALIDITY_KERNEL_NAMES),
        ("make-valid-detect-fp32", _VALIDITY_KERNEL_SOURCE_FP32, _VALIDITY_KERNEL_NAMES),
    ]
)


def _compile_validity_kernels(compute_type: str = "double"):
    from vibespatial.cuda._runtime import get_cuda_runtime, make_kernel_cache_key

    source = _format_validity_kernel_source(compute_type)
    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key(f"make-valid-detect-{compute_type}", source)
    return runtime.compile_kernels(
        cache_key=cache_key,
        source=source,
        kernel_names=_VALIDITY_KERNEL_NAMES,
    )


def _owned_all_polygon_rectangles(owned) -> bool:
    """Return True when every valid row is an exact axis-aligned rectangle."""
    if owned is None or owned.row_count == 0:
        return False
    if not has_gpu_runtime():
        return False

    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.kernels.constructive.polygon_rect_intersection import (
        _device_logical_rectangle_bounds,
        _device_rectangle_bounds,
    )

    if set(owned.families) != {GeometryFamily.POLYGON}:
        return False

    state = owned._ensure_device_state(preserve_indexed_view=True)
    polygon_buf = state.families.get(GeometryFamily.POLYGON)
    physical_rows = (
        max(int(polygon_buf.geometry_offsets.size) - 1, 0) if polygon_buf is not None else 0
    )
    if polygon_buf is None or physical_rows <= 0:
        return False
    if getattr(owned, "is_indexed_view", False) or physical_rows != owned.row_count:
        if not bool(getattr(polygon_buf, "axis_aligned_rectangles", False)):
            return False
        rect_bounds = _device_logical_rectangle_bounds(
            polygon_buf,
            state,
            owned.row_count,
        )
    else:
        rect_bounds = _device_rectangle_bounds(polygon_buf, owned.row_count)
    if rect_bounds is None:
        return False

    cached_validity = owned._current_cached_validity_mask()
    if cached_validity is not None:
        return bool(np.all(cached_validity))
    if getattr(owned, "_validity", None) is not None:
        return bool(np.all(owned._validity))
    if state is not None:
        # Rectangle structure proves topology, but the no-op result also needs
        # a non-null proof. Without one, let the device validity expression
        # produce aligned valid/null rowsets instead of synchronizing a scalar.
        return state.trusted_all_valid is True
    return bool(np.all(owned.validity))


def plan_make_valid_pipeline(
    *, method: str = "linework", keep_collapsed: bool = True
) -> MakeValidPlan:
    stages = (
        MakeValidStage(
            name="compute_validity_mask",
            primitive=MakeValidPrimitive.VALIDITY_MASK,
            purpose="Compute validity and null masks so only invalid rows flow into repair work.",
            inputs=("geometry_rows",),
            outputs=("valid_mask", "null_mask"),
            cccl_mapping=("transform",),
            disposition=IntermediateDisposition.EPHEMERAL,
        ),
        MakeValidStage(
            name="compact_invalid_rows",
            primitive=MakeValidPrimitive.COMPACT_INVALID,
            purpose="Compact invalid rows into one dense repair batch instead of sending valid rows through constructive work.",
            inputs=("geometry_rows", "valid_mask", "null_mask"),
            outputs=("invalid_rows", "invalid_index"),
            cccl_mapping=("DeviceSelect", "gather"),
            disposition=IntermediateDisposition.EPHEMERAL,
        ),
        MakeValidStage(
            name="repair_invalid_topology",
            primitive=MakeValidPrimitive.POLYGONIZE_REPAIR
            if method == "linework"
            else MakeValidPrimitive.SEGMENTIZE_INVALID,
            purpose="Repair only the compacted invalid rows using the selected make-valid strategy.",
            inputs=("invalid_rows",),
            outputs=("repaired_rows",),
            cccl_mapping=("transform", "scatter"),
            disposition=IntermediateDisposition.EPHEMERAL,
            geometry_producing=True,
        ),
        MakeValidStage(
            name="scatter_repaired_rows",
            primitive=MakeValidPrimitive.SCATTER_REPAIRED,
            purpose="Scatter repaired invalid rows back into original row order while preserving valid rows untouched.",
            inputs=("repaired_rows", "invalid_index"),
            outputs=("output_rows",),
            cccl_mapping=("scatter",),
            disposition=IntermediateDisposition.EPHEMERAL,
            geometry_producing=True,
        ),
        MakeValidStage(
            name="emit_geometry",
            primitive=MakeValidPrimitive.EMIT_GEOMETRY,
            purpose="Emit final geometry rows for GeoSeries and overlay preprocessing surfaces.",
            inputs=("output_rows",),
            outputs=("geometry_buffers",),
            cccl_mapping=("gather",),
            disposition=IntermediateDisposition.PERSIST,
            geometry_producing=True,
        ),
    )
    fusion_steps = (
        PipelineStep(name="valid_mask", kind=StepKind.FILTER, output_name="valid_mask"),
        PipelineStep(name="invalid_rows", kind=StepKind.FILTER, output_name="invalid_rows"),
        PipelineStep(name="repaired_rows", kind=StepKind.GEOMETRY, output_name="repaired_rows"),
        PipelineStep(
            name="geometry_buffers",
            kind=StepKind.GEOMETRY,
            output_name="geometry_buffers",
            reusable_output=True,
        ),
    )
    return MakeValidPlan(
        method=method,
        keep_collapsed=keep_collapsed,
        stages=stages,
        fusion_steps=fusion_steps,
        reason=(
            "make_valid should compact invalid rows first and only run constructive repair on the invalid subset so future "
            "GPU implementations can use DeviceSelect-style compaction instead of paying topology-repair cost on already-valid rows."
        ),
    )


def fusion_plan_for_make_valid(*, method: str = "linework", keep_collapsed: bool = True):
    return plan_fusion(
        plan_make_valid_pipeline(method=method, keep_collapsed=keep_collapsed).fusion_steps
    )


@register_kernel_variant(
    "make_valid",
    "gpu-nvrtc",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("polygon", "multipolygon"),
    supports_mixed=True,
    tags=("nvrtc", "constructive", "make_valid", "compact-invalid"),
)
def _make_valid_gpu_repair(owned, repaired_rows, *, method, keep_collapsed):
    """GPU repair through the shared native topology pipeline."""
    from .make_valid_gpu import gpu_repair_invalid_polygons

    return gpu_repair_invalid_polygons(
        owned,
        repaired_rows,
        method=method,
        keep_collapsed=keep_collapsed,
    )


def make_valid_owned(
    values=None,
    *,
    method: str = "linework",
    keep_collapsed: bool = True,
    owned=None,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
) -> MakeValidResult:
    """Validate and repair geometries using compact-invalid-row pattern (ADR-0019).

    Parameters
    ----------
    values : array-like of shapely geometries, optional
        When *owned* is provided, *values* may be None -- Shapely objects
        will only be materialized if GPU validity checks find invalid rows
        that require repair (lazy materialization per ADR-0005).
    method : repair method ("linework" or "structure")
    keep_collapsed : whether to keep collapsed geometries
    owned : optional pre-built OwnedGeometryArray (avoids shapely->owned conversion
            when data is already device-resident, eliminating D->H transfer for
            the validity check per ADR-0005)
    dispatch_mode : requested execution mode (AUTO/GPU/CPU)
    """
    row_count = owned.row_count if owned is not None else (len(values) if values is not None else 0)

    trusted_deferred_valid = (
        owned is not None
        and getattr(owned, "trusted_all_valid", None) is True
        and getattr(owned, "trusted_all_ogc_valid", None) is True
    )

    # Use "make_valid_repair" kernel name to match the crossover override
    # (2,000 rows) instead of the generic CONSTRUCTIVE default (50,000).
    # When owned has device_state the data is already on GPU, so the
    # transfer-free repair path is profitable at very low row counts.
    work_estimate = (
        PhysicalWorkEstimate.from_rows(row_count)
        if trusted_deferred_valid
        else
        estimate_segment_pair_work_from_owned(
            owned,
            output_row_count=row_count,
            output_byte_count=row_count * 32,
            primary_unit_name="make-valid-topology-segment-pair",
        )
        if owned is not None
        else PhysicalWorkEstimate.from_rows(row_count)
    )
    selection = plan_dispatch_selection(
        kernel_name="make_valid_repair",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=row_count,
        work_estimate=work_estimate,
        requested_mode=dispatch_mode,
        current_residency=combined_residency(owned),
    )

    if trusted_deferred_valid:
        record_dispatch_event(
            surface="geopandas.array.make_valid",
            operation="make_valid",
            implementation="trusted_deferred_ogc_valid_no_repair",
            reason=(
                "Deferred native geometry carrier guarantees non-null OGC-valid "
                "output; make_valid preserves the carrier without physicalization"
            ),
            detail=f"rows={row_count}, method={method}, repaired=0",
            requested=dispatch_mode,
            selected=selection.selected,
        )
        return MakeValidResult(
            row_count=row_count,
            valid_rows=np.arange(row_count, dtype=np.int32),
            repaired_rows=np.asarray([], dtype=np.int32),
            null_rows=np.asarray([], dtype=np.int32),
            method=method,
            keep_collapsed=keep_collapsed,
            owned=owned,
            selected=selection.selected,
        )

    # Defer Shapely materialization: when owned is provided, we may not need
    # values at all (zero-transfer fast path).  Materialize lazily.
    geometries = None
    null_mask = None

    def _ensure_geometries():
        nonlocal geometries, null_mask
        if geometries is not None:
            return geometries
        if values is not None:
            geometries = np.asarray(values, dtype=object)
        elif owned is not None:
            geometries = np.asarray(owned.to_shapely(), dtype=object)
        else:
            raise ValueError("Either values or owned must be provided")
        # Preserve vectorized null_mask from ~owned.validity when already set
        if null_mask is None:
            null_mask = np.asarray([g is None for g in geometries], dtype=bool)
        return geometries

    # ADR-0019 compact-invalid-row: detect validity first, repair only invalids.
    # When an OwnedGeometryArray is provided (data already on device), use
    # is_valid_owned for full OGC validity detection without Shapely.
    # If all rows pass, skip Shapely entirely (zero-transfer fast path, ADR-0005).
    #
    # When only Shapely values are provided (no owned), the CPU path is correct:
    # uploading the entire array just to repair 5% on GPU is slower than CPU
    # Shapely.  The GPU path only fires when data is already device-resident.
    gpu_detection_used = False
    native_repair_declined = False
    fallback_recorded = False
    if owned is not None:
        cached_validity = owned._current_cached_validity_mask()
        if cached_validity is not None:
            null_mask = ~owned.validity
            cached_validity = np.asarray(cached_validity, dtype=bool)
            repaired_rows = np.flatnonzero(
                (~null_mask) & (~cached_validity),
            ).astype(np.int32)
            if repaired_rows.size == 0:
                record_dispatch_event(
                    surface="geopandas.array.make_valid",
                    operation="make_valid",
                    implementation="cached_validity_no_repair",
                    reason=("Owned geometry validity cache proved no non-null rows require repair"),
                    detail=f"rows={row_count}, method={method}, repaired=0",
                    requested=dispatch_mode,
                    selected=selection.selected,
                )
                return MakeValidResult(
                    row_count=row_count,
                    valid_rows=np.flatnonzero(~null_mask).astype(np.int32),
                    repaired_rows=np.asarray([], dtype=np.int32),
                    null_rows=np.flatnonzero(null_mask).astype(np.int32),
                    method=method,
                    keep_collapsed=keep_collapsed,
                    owned=owned,
                    selected=selection.selected,
                )

        device_state = getattr(owned, "device_state", None)
        if (
            device_state is not None
            and device_state.trusted_all_valid is True
            and device_state.trusted_all_ogc_valid is True
        ):
            owned._cached_is_valid_mask = np.ones(row_count, dtype=bool)
            record_dispatch_event(
                surface="geopandas.array.make_valid",
                operation="make_valid",
                implementation="trusted_ogc_valid_no_repair",
                reason=(
                    "Owned geometry device metadata proved all rows non-null "
                    "and OGC-valid; make_valid is a no-op"
                ),
                detail=f"rows={row_count}, method={method}, repaired=0",
                requested=dispatch_mode,
                selected=selection.selected,
            )
            return MakeValidResult(
                row_count=row_count,
                valid_rows=np.arange(row_count, dtype=np.int32),
                repaired_rows=np.asarray([], dtype=np.int32),
                null_rows=np.asarray([], dtype=np.int32),
                method=method,
                keep_collapsed=keep_collapsed,
                owned=owned,
                selected=selection.selected,
            )

    if owned is not None and selection.selected is ExecutionMode.GPU:
        if _owned_all_polygon_rectangles(owned):
            record_dispatch_event(
                surface="geopandas.array.make_valid",
                operation="make_valid",
                implementation="rectangle_valid_fast_path",
                reason="All valid rows are exact axis-aligned rectangles; repair is a no-op",
                detail=f"rows={row_count}, method={method}",
                requested=dispatch_mode,
                selected=ExecutionMode.GPU,
            )
            return MakeValidResult(
                row_count=row_count,
                valid_rows=np.arange(row_count, dtype=np.int32),
                repaired_rows=np.asarray([], dtype=np.int32),
                null_rows=np.asarray([], dtype=np.int32),
                method=method,
                keep_collapsed=keep_collapsed,
                owned=owned,
                selected=ExecutionMode.GPU,
            )
        device_rows = None
        if owned.device_state is not None:
            device_rows = _try_device_validity_expression_rows(
                owned,
                row_count=row_count,
            )
            if device_rows is not None:
                device_result = _make_valid_result_from_device_rows(
                    device_rows,
                    owned=owned,
                    row_count=row_count,
                    method=method,
                    keep_collapsed=keep_collapsed,
                    dispatch_mode=dispatch_mode,
                )
                if device_result is not None:
                    return device_result
                native_repair_declined = True

        if native_repair_declined:
            # The native repair result is atomic. Recompute validity from the
            # whole host operation instead of exporting and patching residual
            # rows from a partially repaired carrier.
            record_fallback_event(
                surface="geopandas.array.make_valid",
                reason=(
                    "Native make-valid did not produce a complete aligned result; "
                    "falling back to whole-operation shapely.make_valid"
                ),
                detail=(
                    f"rows={row_count}, "
                    f"repair_candidates={device_rows.repaired_rows.size}, "
                    f"method={method}"
                ),
                requested=dispatch_mode,
                selected=ExecutionMode.CPU,
                pipeline="make_valid_owned",
                d2h_transfer=True,
            )
            fallback_recorded = True
            _ensure_geometries()
            valid_mask = make_valid_cpu_is_valid(geometries)
        else:
            # Compute null_mask from owned validity (no Shapely needed).
            null_mask = ~owned.validity
            from vibespatial.constructive.validity import is_valid_owned

            # Once the public make_valid boundary has selected GPU, keep the
            # internal validity pass on GPU as well instead of replanning below
            # a smaller unary crossover.
            if owned.device_state is None:
                owned._ensure_device_state()
            gpu_mask = is_valid_owned(owned, dispatch_mode=ExecutionMode.GPU)
            gpu_mask[null_mask] = False
            gpu_detection_used = True
            gpu_invalid_rows = np.flatnonzero(~gpu_mask & ~null_mask)
            if gpu_invalid_rows.size == 0:
                record_dispatch_event(
                    surface="geopandas.array.make_valid",
                    operation="make_valid",
                    implementation="is_valid_owned_ogc",
                    reason="Full OGC validity check: all rows valid, zero repair needed",
                    detail=f"rows={row_count}, method={method}",
                    requested=dispatch_mode,
                    selected=ExecutionMode.GPU,
                )
                return MakeValidResult(
                    row_count=row_count,
                    valid_rows=np.flatnonzero(gpu_mask).astype(np.int32),
                    repaired_rows=np.asarray([], dtype=np.int32),
                    null_rows=np.flatnonzero(null_mask).astype(np.int32),
                    method=method,
                    keep_collapsed=keep_collapsed,
                    owned=owned,
                    selected=ExecutionMode.GPU,
                )
            valid_mask = gpu_mask
    else:
        if owned is not None:
            null_mask = ~owned.validity
        _ensure_geometries()
        # CPU-only mode: GPU ring checks not available, fall back to Shapely
        valid_mask = make_valid_cpu_is_valid(geometries)

    # Identify rows needing repair (no Shapely needed for this).
    valid_mask[null_mask] = False
    repaired_mask = (~null_mask) & (~valid_mask)
    repaired_rows = np.flatnonzero(repaired_mask).astype(np.int32)
    selected = ExecutionMode.GPU if gpu_detection_used else ExecutionMode.CPU
    result = None  # Shapely array; only populated when CPU path is used
    result_owned = None  # device-resident result; populated by GPU path

    if repaired_rows.size:
        # Try GPU repair path first — stays device-resident, no D->H (ADR-0005)
        gpu_repair_done = False
        if (
            owned is not None
            and selection.selected is ExecutionMode.GPU
            and not native_repair_declined
        ):
            gpu_result = _make_valid_gpu_repair(
                owned,
                repaired_rows,
                method=method,
                keep_collapsed=keep_collapsed,
            )
            if gpu_result is not None:
                result_owned = gpu_result.repaired_owned
                gpu_repair_done = result_owned is not None
                if gpu_repair_done:
                    selected = ExecutionMode.GPU

        if not gpu_repair_done:
            # GPU repair failed — fall back to CPU. Now we need Shapely.
            if selection.selected is ExecutionMode.GPU and not fallback_recorded:
                record_fallback_event(
                    surface="geopandas.array.make_valid",
                    reason=(
                        "Native make-valid did not produce a complete aligned result; "
                        "falling back to whole-operation shapely.make_valid"
                    ),
                    detail=f"rows={row_count}, repaired={repaired_rows.size}, method={method}",
                    requested=dispatch_mode,
                    selected=ExecutionMode.CPU,
                    pipeline="make_valid_owned",
                    d2h_transfer=True,
                )
            _ensure_geometries()
            result = make_valid_cpu_repair(
                geometries,
                repaired_rows,
                method=method,
                keep_collapsed=keep_collapsed,
            )
            if not gpu_detection_used:
                selected = ExecutionMode.CPU
    else:
        # No repair needed — preserve device residency when available,
        # otherwise materialize geometries for host-only callers.
        result_owned = owned
        if result_owned is None:
            _ensure_geometries()
            result = geometries

    impl = "gpu_ring_validity_check" if gpu_detection_used else "shapely_is_valid"
    if repaired_rows.size:
        impl += "+gpu_repair" if selected is ExecutionMode.GPU else "+shapely_make_valid"
    record_dispatch_event(
        surface="geopandas.array.make_valid",
        operation="make_valid",
        implementation=impl,
        reason=f"make_valid dispatch: detection={'GPU' if gpu_detection_used else 'CPU'}, "
        f"repair={selected.value}, {repaired_rows.size} rows repaired",
        detail=f"rows={row_count}, method={method}, repaired={repaired_rows.size}",
        requested=dispatch_mode,
        selected=selected,
    )
    native_geometry = None
    if selected is ExecutionMode.GPU and repaired_rows.size:
        native_geometry = _make_valid_linework_native_geometry(
            owned,
            result_owned,
            repaired_rows,
            method=method,
        )
    return MakeValidResult(
        row_count=row_count,
        valid_rows=np.flatnonzero(valid_mask).astype(np.int32),
        repaired_rows=repaired_rows,
        null_rows=np.flatnonzero(null_mask).astype(np.int32),
        method=method,
        keep_collapsed=keep_collapsed,
        owned=result_owned,
        native_geometry=native_geometry,
        selected=selected,
        _geometries=result,
    )


def evaluate_geopandas_make_valid(
    values,
    *,
    method: str = "linework",
    keep_collapsed: bool = True,
    prebuilt_owned=None,
) -> MakeValidResult:
    """Run make_valid and return the full MakeValidResult.

    Returns MakeValidResult so callers can access .owned for device-resident
    fast paths and .selected for dispatch event accuracy.
    """
    from vibespatial.runtime.execution_trace import execution_trace

    with execution_trace("make_valid"):
        owned = prebuilt_owned
        input_values = values
        if (
            owned is None
            and values is not None
            and has_gpu_runtime()
            and values_support_owned_make_valid(values)
        ):
            from vibespatial.geometry.owned import from_shapely_geometries

            geometries = np.asarray(values, dtype=object)
            owned_candidate = from_shapely_geometries(
                geometries.tolist(),
                residency=Residency.DEVICE,
            )
            if _owned_can_skip_host_make_valid(owned_candidate):
                owned = owned_candidate
                input_values = None
        return make_valid_owned(
            input_values,
            method=method,
            keep_collapsed=keep_collapsed,
            owned=owned,
        )


_gpu_make_valid_warmed = False


def _warmup_gpu_make_valid_pipeline():
    """Block until the full GPU make_valid pipeline is compiled.

    1. Import make_valid_gpu + overlay/gpu to trigger module-scope
       request_nvrtc_warmup / request_warmup calls.
    2. Call ensure_warm() on both precompilers to block until all
       NVRTC kernels and CCCL specs are compiled (or loaded from
       the disk cubin cache).
    3. Run a tiny throwaway repair to flush any remaining first-call
       paths (Numba operator JIT, CuPy kernel caching, etc.).
    """
    global _gpu_make_valid_warmed
    if _gpu_make_valid_warmed:
        return
    _gpu_make_valid_warmed = True

    # Step 1+2: import modules and block on precompilation
    from vibespatial.cuda.cccl_precompile import CCCLPrecompiler
    from vibespatial.cuda.nvrtc_precompile import NVRTCPrecompiler

    from . import make_valid_gpu  # noqa: F401

    NVRTCPrecompiler.get().ensure_warm(timeout=60.0)
    CCCLPrecompiler.get().ensure_warm(timeout=60.0)

    # Step 3: throwaway repair at representative scale to flush remaining
    # per-size-class JIT paths.  CCCL make_* callables pre-allocate temp
    # storage for a specific max input size; a 1-polygon warmup only covers
    # that tiny size class.  Using ~100 self-intersecting polygons ensures
    # the CCCL temp storage, CuPy kernel cache, and overlay pipeline
    # internal caches are all sized for realistic workloads.
    make_valid_owned(owned=build_make_valid_warmup_owned())


def benchmark_make_valid(
    values,
    *,
    method: str = "linework",
    keep_collapsed: bool = True,
    dataset: str = "make-valid",
    owned=None,
):
    geometries = np.asarray(values, dtype=object)

    # Warm up the full GPU make_valid pipeline before the timed measurement.
    # CCCL make_* callables cache temp storage per input-size class, so the
    # precompile warmup (which uses a small dataset) doesn't cover the real
    # workload.  A throwaway call with the actual data ensures all CCCL
    # temp storage, CuPy kernel caches, and overlay pipeline internal
    # state are sized for this specific workload.
    if owned is not None:
        from vibespatial.runtime import has_gpu_runtime

        if has_gpu_runtime():
            _warmup_gpu_make_valid_pipeline()
            make_valid_owned(geometries, method=method, keep_collapsed=keep_collapsed, owned=owned)

    start = perf_counter()
    compact = make_valid_owned(
        geometries, method=method, keep_collapsed=keep_collapsed, owned=owned
    )
    compact_elapsed = perf_counter() - start

    start = perf_counter()
    # Benchmark baseline: intentional Shapely call for comparison timing
    make_valid_cpu_baseline(geometries, method=method, keep_collapsed=keep_collapsed)
    baseline_elapsed = perf_counter() - start

    return MakeValidBenchmark(
        dataset=dataset,
        rows=len(geometries),
        repaired_rows=int(compact.repaired_rows.size),
        compact_elapsed_seconds=compact_elapsed,
        baseline_elapsed_seconds=baseline_elapsed,
    )
