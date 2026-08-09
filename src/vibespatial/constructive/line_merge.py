"""GPU-accelerated line_merge: merge connected LineStrings into longer chains.

Replaces ``shapely.line_merge(data, directed=directed)`` with a fully
device-resident NVRTC implementation.

Architecture (ADR-0033 tier classification):
- Tier 1 NVRTC: per-geometry endpoint graph construction and chain-following.
  One thread per geometry.  Graph data fits in thread-local arrays for
  typical MultiLineStrings (up to 256 parts).
- Tier 3a CCCL: exclusive_sum for output offset computation.
- Tier 2 CuPy: metadata array construction, element-wise operations.

Precision (ADR-0002): CONSTRUCTIVE class -- stays fp64 on all devices.
Coordinates are exact subsets of input (no arithmetic on coordinate values).
PrecisionPlan wired through dispatch for observability.

Supports:
- MultiLineString: merge connected parts into longer chains
- LineString: returned as-is (single segment, trivial merge)
- directed=True: only merge if endpoint == startpoint of next segment
- directed=False: bidirectional endpoint matching (default)
- Disconnected components: output MultiLineString with multiple parts
- Rings (closed chains): detected via unvisited segments after open chains

Source coordinate and part capacities size outputs; device offsets carry the
logical lengths without a host allocation fence.
"""

from __future__ import annotations

import logging

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

from vibespatial.constructive.line_merge_cpu import _line_merge_cpu
from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.cccl_primitives import exclusive_sum
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    OwnedGeometryArray,
    build_device_resident_owned,
    from_shapely_geometries,
)
from vibespatial.kernels.constructive.line_merge import (
    _LINE_MERGE_KERNEL_SOURCE,
    LINE_MERGE_KERNEL_NAMES,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.crossover import estimate_part_pair_work_from_owned
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import (
    KernelClass,
    PrecisionMode,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ADR-0034: CCCL + NVRTC warmup at module scope
# ---------------------------------------------------------------------------

# CCCL: exclusive_sum used twice (coord offsets, part offsets)
request_warmup(
    [
        "exclusive_scan_i32",
    ]
)

request_nvrtc_warmup(
    [
        ("line-merge-fp64", _LINE_MERGE_KERNEL_SOURCE, LINE_MERGE_KERNEL_NAMES),
    ]
)


def _compile_kernels():
    """Compile and cache NVRTC kernels."""
    return compile_kernel_group(
        "line-merge-fp64",
        _LINE_MERGE_KERNEL_SOURCE,
        LINE_MERGE_KERNEL_NAMES,
    )


# ---------------------------------------------------------------------------
# GPU implementation
# ---------------------------------------------------------------------------


@register_kernel_variant(
    "line_merge",
    "gpu-cuda-python",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("linestring", "multilinestring"),
    supports_mixed=True,
    tags=("cuda-python", "constructive", "line_merge"),
)
def _line_merge_gpu(
    owned: OwnedGeometryArray,
    *,
    directed: bool = False,
) -> OwnedGeometryArray:
    """GPU line_merge -- returns device-resident MultiLineString/LineString OGA.

    Algorithm:
        1. Identify LineString and MultiLineString rows
        2. Launch count kernel: per-geometry endpoint graph + chain walk
           to count output coordinates and parts
        3. CCCL exclusive_sum for output offsets
        4. Launch scatter kernel: same graph walk, writes coordinates
        5. Build output OGA (LineString for single-part, MultiLineString for multi)
    """
    runtime = get_cuda_runtime()
    d_state = owned._ensure_device_state()
    row_count = owned.row_count
    ptr = runtime.pointer

    if row_count == 0:
        return _build_empty_output(d_state.validity, row_count=0)

    kernels = _compile_kernels()

    # Identify rows belonging to LineString or MultiLineString families
    ls_tag = FAMILY_TAGS[GeometryFamily.LINESTRING]
    mls_tag = FAMILY_TAGS[GeometryFamily.MULTILINESTRING]

    d_validity = cp.asarray(d_state.validity, dtype=cp.bool_)
    d_tags = cp.asarray(d_state.tags, dtype=cp.int8)
    d_eligible = d_validity & ((d_tags == ls_tag) | (d_tags == mls_tag))
    d_family_codes = cp.where(d_tags == ls_tag, 1, 0).astype(cp.int32)
    d_fam_local_rows = cp.where(
        d_eligible,
        cp.asarray(d_state.family_row_offsets, dtype=cp.int32),
        cp.int32(0),
    )

    # Get family buffers (use dummy empty arrays if family not present)
    has_mls = GeometryFamily.MULTILINESTRING in d_state.families
    has_ls = GeometryFamily.LINESTRING in d_state.families

    if has_mls:
        d_mls = d_state.families[GeometryFamily.MULTILINESTRING]
        mls_row_count = int(d_mls.geometry_offsets.shape[0]) - 1
    else:
        mls_row_count = 0

    if has_ls:
        d_ls = d_state.families[GeometryFamily.LINESTRING]
        ls_row_count = int(d_ls.geometry_offsets.shape[0]) - 1
    else:
        ls_row_count = 0

    # Create dummy device arrays for missing families
    d_dummy = runtime.allocate((1,), np.float64)
    d_dummy_i = runtime.allocate((1,), np.int32, zero=True)

    mls_x = d_mls.x if has_mls else d_dummy
    mls_y = d_mls.y if has_mls else d_dummy
    mls_geom_off = d_mls.geometry_offsets if has_mls else d_dummy_i
    mls_part_off = d_mls.part_offsets if has_mls else d_dummy_i

    ls_x = d_ls.x if has_ls else d_dummy
    ls_y = d_ls.y if has_ls else d_dummy
    ls_geom_off = d_ls.geometry_offsets if has_ls else d_dummy_i

    directed_int = 1 if directed else 0

    # Allocate count output arrays
    d_coord_counts = runtime.allocate((row_count,), np.int32, zero=True)
    d_part_counts = runtime.allocate((row_count,), np.int32, zero=True)

    # --- Pass 1: Count ---
    count_params = (
        (
            ptr(mls_x),
            ptr(mls_y),
            ptr(mls_geom_off),
            ptr(mls_part_off),
            mls_row_count,
            ptr(ls_x),
            ptr(ls_y),
            ptr(ls_geom_off),
            ls_row_count,
            ptr(d_family_codes),
            ptr(d_fam_local_rows),
            ptr(d_eligible),
            directed_int,
            ptr(d_coord_counts),
            ptr(d_part_counts),
            row_count,
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
        ),
    )
    grid, block = runtime.launch_config(kernels["line_merge_count"], row_count)
    runtime.launch(kernels["line_merge_count"], grid=grid, block=block, params=count_params)

    # --- Prefix sums for output offsets ---
    d_coord_offsets = exclusive_sum(d_coord_counts, synchronize=False)
    d_part_offsets = exclusive_sum(d_part_counts, synchronize=False)

    coordinate_capacity = (int(d_mls.x.size) if has_mls else 0) + (
        int(d_ls.x.size) if has_ls else 0
    )
    part_capacity = (max(int(d_mls.part_offsets.size) - 1, 0) if has_mls else 0) + (
        ls_row_count if has_ls else 0
    )

    # --- Allocate source-bounded output capacity ---
    d_out_x = runtime.allocate((coordinate_capacity,), np.float64)
    d_out_y = runtime.allocate((coordinate_capacity,), np.float64)
    d_out_part_off = runtime.allocate((part_capacity,), np.int32, zero=True)

    # --- Pass 2: Scatter ---
    scatter_params = (
        (
            ptr(mls_x),
            ptr(mls_y),
            ptr(mls_geom_off),
            ptr(mls_part_off),
            mls_row_count,
            ptr(ls_x),
            ptr(ls_y),
            ptr(ls_geom_off),
            ls_row_count,
            ptr(d_family_codes),
            ptr(d_fam_local_rows),
            ptr(d_eligible),
            directed_int,
            ptr(d_coord_offsets),
            ptr(d_part_offsets),
            ptr(d_out_x),
            ptr(d_out_y),
            ptr(d_out_part_off),
            row_count,
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
        ),
    )
    grid, block = runtime.launch_config(kernels["line_merge_scatter"], row_count)
    runtime.launch(kernels["line_merge_scatter"], grid=grid, block=block, params=scatter_params)

    # --- Build output OGA ---
    # All output rows are MultiLineString (even single-part results, for
    # consistency).  Shapely's line_merge returns LineString for single chains
    # and MultiLineString for disconnected components, but MultiLineString
    # with one part is equivalent and simpler to produce in bulk.
    #
    # The geometry_offsets for MultiLineString map row -> parts range in
    # the part_offsets array.  We need to compute per-row geometry_offsets
    # from d_part_counts.

    d_out_geom_offsets = cp.empty(row_count + 1, dtype=cp.int32)
    d_out_geom_offsets[0] = 0
    cp.cumsum(d_part_counts, out=d_out_geom_offsets[1:])

    d_total_coords = d_coord_offsets[-1:] + d_coord_counts[-1:]
    d_total_parts = d_part_offsets[-1:] + d_part_counts[-1:]
    d_full_part_off = cp.empty(part_capacity + 1, dtype=cp.int32)
    if part_capacity:
        d_part_lanes = cp.arange(part_capacity, dtype=cp.int32)
        d_full_part_off[:part_capacity] = cp.where(
            d_part_lanes < d_total_parts[0],
            cp.asarray(d_out_part_off, dtype=cp.int32),
            d_total_coords[0],
        )
    d_full_part_off[part_capacity] = d_total_coords[0]

    d_out_empty = (~d_eligible) | (cp.asarray(d_part_counts) == 0)
    out_tags = cp.where(
        d_validity,
        np.int8(FAMILY_TAGS[GeometryFamily.MULTILINESTRING]),
        np.int8(-1),
    )
    out_family_row_offsets = cp.where(
        d_validity,
        cp.arange(row_count, dtype=cp.int32),
        cp.int32(-1),
    )

    device_families = {
        GeometryFamily.MULTILINESTRING: DeviceFamilyGeometryBuffer(
            family=GeometryFamily.MULTILINESTRING,
            x=d_out_x,
            y=d_out_y,
            geometry_offsets=d_out_geom_offsets,
            part_offsets=d_full_part_off,
            empty_mask=d_out_empty,
        ),
    }

    return build_device_resident_owned(
        device_families=device_families,
        row_count=row_count,
        tags=out_tags,
        validity=d_validity,
        family_row_offsets=out_family_row_offsets,
        execution_mode="gpu",
    )


def _build_empty_output(validity, *, row_count: int) -> OwnedGeometryArray:
    """Build an all-empty MultiLineString OGA."""
    runtime = get_cuda_runtime()

    d_validity = cp.asarray(validity, dtype=cp.bool_)
    out_tags = cp.full(row_count, FAMILY_TAGS[GeometryFamily.MULTILINESTRING], dtype=cp.int8)
    out_family_row_offsets = cp.arange(row_count, dtype=cp.int32)
    out_tags = cp.where(d_validity, out_tags, np.int8(-1))
    out_family_row_offsets = cp.where(
        d_validity,
        out_family_row_offsets,
        cp.int32(-1),
    )

    d_x = runtime.allocate((0,), np.float64)
    d_y = runtime.allocate((0,), np.float64)
    d_geom_off = cp.zeros(row_count + 1, dtype=cp.int32)
    d_part_off = cp.zeros(1, dtype=cp.int32)
    d_empty = cp.ones(row_count, dtype=cp.bool_)

    device_families = {
        GeometryFamily.MULTILINESTRING: DeviceFamilyGeometryBuffer(
            family=GeometryFamily.MULTILINESTRING,
            x=d_x,
            y=d_y,
            geometry_offsets=d_geom_off,
            part_offsets=d_part_off,
            empty_mask=d_empty,
        ),
    }

    return build_device_resident_owned(
        device_families=device_families,
        row_count=row_count,
        tags=out_tags,
        validity=d_validity,
        family_row_offsets=out_family_row_offsets,
        execution_mode="gpu",
    )


# ---------------------------------------------------------------------------
# Public dispatch API
# ---------------------------------------------------------------------------


def line_merge_owned(
    owned: OwnedGeometryArray,
    *,
    directed: bool = False,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """Merge connected LineStrings within each geometry.

    For MultiLineString inputs, merges parts that share endpoints into
    longer LineString chains.  Disconnected components produce a
    MultiLineString in the output.

    Parameters
    ----------
    owned : OwnedGeometryArray
        Input geometries (LineString or MultiLineString).
    directed : bool, default False
        If True, only merge segments where endpoint == startpoint of
        next segment.  If False, bidirectional endpoint matching.
    dispatch_mode : ExecutionMode or str, default AUTO
        Execution mode hint.
    precision : PrecisionMode or str, default AUTO
        Precision mode.  CONSTRUCTIVE class stays fp64 per ADR-0002;
        wired for observability.

    Returns
    -------
    OwnedGeometryArray
        Merged geometries.  Single-chain results are MultiLineString
        with one part; disconnected results have multiple parts.
    """
    row_count = owned.row_count
    if row_count == 0:
        return from_shapely_geometries([])

    selection = plan_dispatch_selection(
        kernel_name="line_merge",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=row_count,
        work_estimate=estimate_part_pair_work_from_owned(
            owned,
            output_row_count=row_count,
            primary_unit_name="line-merge-endpoint-pair",
        ),
        requested_mode=dispatch_mode,
        requested_precision=precision,
        current_residency=owned.residency,
    )

    if selection.selected is ExecutionMode.GPU:
        precision_plan = selection.precision_plan
        result = _line_merge_gpu(owned, directed=directed)
        record_dispatch_event(
            surface="geopandas.array.line_merge",
            operation="line_merge",
            implementation="line_merge_gpu_nvrtc",
            reason=selection.reason,
            detail=(
                f"rows={row_count}, directed={directed}, "
                f"precision={precision_plan.compute_precision.value}"
            ),
            requested=selection.requested,
            selected=ExecutionMode.GPU,
        )
        return result

    result = _line_merge_cpu(owned, directed=directed)
    record_dispatch_event(
        surface="geopandas.array.line_merge",
        operation="line_merge",
        implementation="line_merge_cpu_shapely",
        reason=selection.reason,
        detail=f"rows={row_count}, directed={directed}",
        requested=selection.requested,
        selected=ExecutionMode.CPU,
    )
    return result


def line_merge_native_tabular_result(
    owned: OwnedGeometryArray,
    *,
    directed: bool = False,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
    crs=None,
    geometry_name: str = "geometry",
    source_rows=None,
    source_tokens: tuple[str, ...] = (),
    attrs: dict[str, object] | None = None,
):
    from vibespatial.api._native_results import (
        _unary_constructive_owned_to_native_tabular_result,
    )

    result = line_merge_owned(
        owned,
        directed=directed,
        dispatch_mode=dispatch_mode,
        precision=precision,
    )
    return _unary_constructive_owned_to_native_tabular_result(
        result,
        operation="line_merge",
        crs=crs,
        geometry_name=geometry_name,
        source_rows=source_rows,
        source_tokens=source_tokens,
        attrs=attrs,
    )
