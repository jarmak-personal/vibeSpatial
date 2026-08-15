from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from fractions import Fraction
from time import perf_counter

import numpy as np

from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.cccl_primitives import (
    compact_indices,
    exclusive_sum,
    lower_bound,
    sort_pairs,
    upper_bound,
)

request_warmup(
    [
        "select_i32",
        "select_i64",
        "exclusive_scan_i32",
        "exclusive_scan_i64",
        "lower_bound_f64",
        "lower_bound_i64",
        "upper_bound_f64",
        "upper_bound_i64",
    ]
)
from vibespatial.cuda._runtime import (  # noqa: E402
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    DeviceArray,
    compile_kernel_group,
    get_cuda_completion_retainer,
    get_cuda_runtime,
)
from vibespatial.geometry.buffers import GeometryFamily  # noqa: E402
from vibespatial.geometry.owned import (  # noqa: E402
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    DeviceFixedGeometrySizeMetadata,
    OwnedGeometryArray,
    build_device_resident_owned,
)
from vibespatial.runtime import ExecutionMode, RuntimeSelection  # noqa: E402
from vibespatial.runtime.adaptive import AdaptivePlan, plan_dispatch_selection  # noqa: E402
from vibespatial.runtime.config import SEGMENT_TILE_SIZE  # noqa: E402
from vibespatial.runtime.crossover import PhysicalWorkEstimate  # noqa: E402
from vibespatial.runtime.dispatch import record_dispatch_event  # noqa: E402
from vibespatial.runtime.hotpath_trace import hotpath_stage  # noqa: E402
from vibespatial.runtime.kernel_registry import register_kernel_variant  # noqa: E402
from vibespatial.runtime.precision import (  # noqa: E402
    KernelClass,
    PrecisionMode,
    PrecisionPlan,
)
from vibespatial.runtime.residency import Residency, combined_residency  # noqa: E402
from vibespatial.runtime.robustness import RobustnessPlan, select_robustness_plan  # noqa: E402
from vibespatial.spatial.segment_primitives_kernels import (  # noqa: E402
    _SAME_ROW_CANDIDATE_KERNEL_NAMES,
    _SAME_ROW_CANDIDATE_KERNEL_SOURCE,
    _SEGMENT_CLASSIFY_KERNEL_NAMES,
    _SEGMENT_EXTRACT_KERNEL_NAMES,
    _SWEEP_CANDIDATE_KERNEL_NAMES,
    _SWEEP_CANDIDATE_KERNEL_SOURCE,
    CLASSIFY_SOURCE_FP32,
    CLASSIFY_SOURCE_FP64,
    EXTRACT_SOURCE_FP32,
    EXTRACT_SOURCE_FP64,
    format_classify_source,
    format_extract_source,
)

_FLOAT_EPSILON = np.finfo(np.float64).eps
_ORIENTATION_ERRBOUND = (3.0 + 16.0 * _FLOAT_EPSILON) * _FLOAT_EPSILON
_SEGMENT_EXTRACTION_CAPACITY_MAX_SLOTS = 2 * 1024 * 1024

# ---------------------------------------------------------------------------
# Family type codes matching GeometryFamily enum order (0-based)
# ---------------------------------------------------------------------------
_FAMILY_LINESTRING = FAMILY_TAGS[GeometryFamily.LINESTRING]
_FAMILY_POLYGON = FAMILY_TAGS[GeometryFamily.POLYGON]
_FAMILY_MULTILINESTRING = FAMILY_TAGS[GeometryFamily.MULTILINESTRING]
_FAMILY_MULTIPOLYGON = FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]


def _segment_device_to_host(device_array: object, *, reason: str) -> np.ndarray:
    return np.asarray(get_cuda_runtime().copy_device_to_host(device_array, reason=reason))


def _segment_int_scalar(value: object, *, reason: str) -> int:
    import cupy as cp

    return int(_segment_device_to_host(cp.asarray(value).reshape(1), reason=reason)[0])


# ---------------------------------------------------------------------------
# Kernel sources extracted to segment_primitives_kernels.py
# Imported above: _SEGMENT_EXTRACT_KERNEL_NAMES, _SEGMENT_CLASSIFY_KERNEL_NAMES,
#   _CANDIDATE_SCATTER_KERNEL_NAMES, _CANDIDATE_SCATTER_KERNEL_SOURCE,
#   format_extract_source, format_classify_source,
#   EXTRACT_SOURCE_FP64, EXTRACT_SOURCE_FP32,
#   CLASSIFY_SOURCE_FP64, CLASSIFY_SOURCE_FP32.
# ---------------------------------------------------------------------------

# Data types
# ---------------------------------------------------------------------------


class SegmentIntersectionKind(IntEnum):
    DISJOINT = 0
    PROPER = 1
    TOUCH = 2
    OVERLAP = 3


@dataclass(frozen=True)
class SegmentTable:
    row_indices: np.ndarray
    part_indices: np.ndarray
    ring_indices: np.ndarray
    segment_indices: np.ndarray
    x0: np.ndarray
    y0: np.ndarray
    x1: np.ndarray
    y1: np.ndarray
    bounds: np.ndarray

    @property
    def count(self) -> int:
        return int(self.row_indices.size)


@dataclass(frozen=True)
class DeviceSegmentTable:
    """GPU-resident segment table in SoA layout."""

    row_indices: DeviceArray
    segment_indices: DeviceArray
    x0: DeviceArray
    y0: DeviceArray
    x1: DeviceArray
    y1: DeviceArray
    count: int
    max_segments_per_row: int | None = None
    part_indices: DeviceArray | None = None
    ring_indices: DeviceArray | None = None

    def free(self) -> None:
        """Release all device allocations held by this table.

        Consolidates the 7--9 individual ``runtime.free()`` calls that
        previously had to be duplicated at every cleanup site.
        """
        runtime = get_cuda_runtime()
        runtime.free(self.x0)
        runtime.free(self.y0)
        runtime.free(self.x1)
        runtime.free(self.y1)
        runtime.free(self.row_indices)
        runtime.free(self.segment_indices)
        if self.part_indices is not None:
            runtime.free(self.part_indices)
        if self.ring_indices is not None:
            runtime.free(self.ring_indices)


def device_segment_table_as_linestrings(
    segments: DeviceSegmentTable,
) -> OwnedGeometryArray:
    """Expose segment rows as a compact fixed-width device geometry carrier.

    Segment coordinates remain device-only. The returned rows are independent
    two-coordinate LineStrings whose bounds can participate in the canonical
    native spatial-index relation without interpreting the source geometry's
    nested offsets again.
    """
    import cupy as cp

    row_count = int(segments.count)
    d_x = cp.empty(row_count * 2, dtype=cp.float64)
    d_y = cp.empty(row_count * 2, dtype=cp.float64)
    d_x[0::2] = cp.asarray(segments.x0, dtype=cp.float64)
    d_x[1::2] = cp.asarray(segments.x1, dtype=cp.float64)
    d_y[0::2] = cp.asarray(segments.y0, dtype=cp.float64)
    d_y[1::2] = cp.asarray(segments.y1, dtype=cp.float64)
    d_bounds = cp.stack(
        (
            cp.minimum(d_x[0::2], d_x[1::2]),
            cp.minimum(d_y[0::2], d_y[1::2]),
            cp.maximum(d_x[0::2], d_x[1::2]),
            cp.maximum(d_y[0::2], d_y[1::2]),
        ),
        axis=1,
    )
    result = build_device_resident_owned(
        device_families={
            GeometryFamily.LINESTRING: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.LINESTRING,
                x=d_x,
                y=d_y,
                geometry_offsets=cp.arange(
                    0,
                    (row_count + 1) * 2,
                    2,
                    dtype=cp.int32,
                ),
                empty_mask=cp.zeros(row_count, dtype=cp.bool_),
                bounds=d_bounds,
                fixed_size=DeviceFixedGeometrySizeMetadata(
                    coord_count_per_row=2,
                    max_coord_count_per_row=2,
                ),
            )
        },
        row_count=row_count,
        tags=cp.full(
            row_count,
            FAMILY_TAGS[GeometryFamily.LINESTRING],
            dtype=cp.int8,
        ),
        validity=cp.ones(row_count, dtype=cp.bool_),
        family_row_offsets=cp.arange(row_count, dtype=cp.int32),
        execution_mode="gpu",
    )
    state = result._ensure_device_state(preserve_indexed_view=True)
    state.row_bounds = d_bounds
    state.trusted_all_valid = True
    state.trusted_all_non_empty = True
    state.trusted_homogeneous_family = GeometryFamily.LINESTRING
    state.trusted_unique_family_rows = True
    state.trusted_family_domain = (GeometryFamily.LINESTRING,)
    result._active_family_row_segment_capacity_bound = 1
    return result


@dataclass(frozen=True)
class DeviceBroadcastSegmentRelation:
    """One physical segment table related to many logical geometry rows.

    The segment coordinates and source metadata remain singular. Consumers
    derive a logical segment id from ``logical_row * physical_count + lookup``
    instead of repeating the table for every row.
    """

    physical_segments: DeviceSegmentTable
    logical_row_count: int

    def __post_init__(self) -> None:
        if int(self.logical_row_count) < 0:
            raise ValueError("broadcast segment logical_row_count must be non-negative")

    @property
    def physical_count(self) -> int:
        return int(self.physical_segments.count)

    @property
    def logical_count(self) -> int:
        return self.physical_count * int(self.logical_row_count)


@dataclass
class SegmentIntersectionResult:
    """Segment intersection results with lazy host materialization.

    When produced by the GPU pipeline, all 14 result arrays live in
    ``device_state`` and host numpy arrays are lazily copied on first
    property access.  GPU-only consumers (e.g. ``build_gpu_split_events``)
    that read only ``device_state``, ``candidate_pairs``, ``count``,
    ``runtime_selection``, ``precision_plan``, and ``robustness_plan``
    never trigger device-to-host copies.
    """

    candidate_pairs: int
    runtime_selection: RuntimeSelection
    precision_plan: PrecisionPlan
    robustness_plan: RobustnessPlan
    device_state: SegmentIntersectionDeviceState | None = None
    _count: int = 0
    # Host arrays — lazily materialized from device_state on first access.
    _left_rows: np.ndarray | None = None
    _left_segments: np.ndarray | None = None
    _left_lookup: np.ndarray | None = None
    _right_rows: np.ndarray | None = None
    _right_segments: np.ndarray | None = None
    _right_lookup: np.ndarray | None = None
    _kinds: np.ndarray | None = None
    _point_x: np.ndarray | None = None
    _point_y: np.ndarray | None = None
    _overlap_x0: np.ndarray | None = None
    _overlap_y0: np.ndarray | None = None
    _overlap_x1: np.ndarray | None = None
    _overlap_y1: np.ndarray | None = None
    _ambiguous_rows: np.ndarray | None = None

    def _ensure_host(self) -> None:
        """Lazily copy host arrays from device_state on first access."""
        if self._left_rows is not None:
            return
        ds = self.device_state
        if ds is None:
            return
        runtime = get_cuda_runtime()
        self._left_rows = np.asarray(
            runtime.copy_device_to_host(
                ds.left_rows,
                reason="segment intersections left-row host export",
            ),
            dtype=np.int32,
        )
        self._left_segments = np.asarray(
            runtime.copy_device_to_host(
                ds.left_segments,
                reason="segment intersections left-segment host export",
            ),
            dtype=np.int32,
        )
        self._left_lookup = np.asarray(
            runtime.copy_device_to_host(
                ds.left_lookup,
                reason="segment intersections left-lookup host export",
            ),
            dtype=np.int32,
        )
        self._right_rows = np.asarray(
            runtime.copy_device_to_host(
                ds.right_rows,
                reason="segment intersections right-row host export",
            ),
            dtype=np.int32,
        )
        self._right_segments = np.asarray(
            runtime.copy_device_to_host(
                ds.right_segments,
                reason="segment intersections right-segment host export",
            ),
            dtype=np.int32,
        )
        self._right_lookup = np.asarray(
            runtime.copy_device_to_host(
                ds.right_lookup,
                reason="segment intersections right-lookup host export",
            ),
            dtype=np.int32,
        )
        self._kinds = np.asarray(
            runtime.copy_device_to_host(
                ds.kinds,
                reason="segment intersections kind-code host export",
            ),
            dtype=np.int8,
        )
        self._point_x = np.asarray(
            runtime.copy_device_to_host(
                ds.point_x,
                reason="segment intersections point-x host export",
            ),
            dtype=np.float64,
        )
        self._point_y = np.asarray(
            runtime.copy_device_to_host(
                ds.point_y,
                reason="segment intersections point-y host export",
            ),
            dtype=np.float64,
        )
        self._overlap_x0 = np.asarray(
            runtime.copy_device_to_host(
                ds.overlap_x0,
                reason="segment intersections overlap-x0 host export",
            ),
            dtype=np.float64,
        )
        self._overlap_y0 = np.asarray(
            runtime.copy_device_to_host(
                ds.overlap_y0,
                reason="segment intersections overlap-y0 host export",
            ),
            dtype=np.float64,
        )
        self._overlap_x1 = np.asarray(
            runtime.copy_device_to_host(
                ds.overlap_x1,
                reason="segment intersections overlap-x1 host export",
            ),
            dtype=np.float64,
        )
        self._overlap_y1 = np.asarray(
            runtime.copy_device_to_host(
                ds.overlap_y1,
                reason="segment intersections overlap-y1 host export",
            ),
            dtype=np.float64,
        )
        self._ambiguous_rows = np.asarray(
            runtime.copy_device_to_host(
                ds.ambiguous_rows,
                reason="segment intersections ambiguous-row host export",
            ),
            dtype=np.int32,
        )

    @property
    def left_rows(self) -> np.ndarray:
        self._ensure_host()
        return self._left_rows  # type: ignore[return-value]

    @property
    def left_segments(self) -> np.ndarray:
        self._ensure_host()
        return self._left_segments  # type: ignore[return-value]

    @property
    def left_lookup(self) -> np.ndarray:
        self._ensure_host()
        return self._left_lookup  # type: ignore[return-value]

    @property
    def right_rows(self) -> np.ndarray:
        self._ensure_host()
        return self._right_rows  # type: ignore[return-value]

    @property
    def right_segments(self) -> np.ndarray:
        self._ensure_host()
        return self._right_segments  # type: ignore[return-value]

    @property
    def right_lookup(self) -> np.ndarray:
        self._ensure_host()
        return self._right_lookup  # type: ignore[return-value]

    @property
    def kinds(self) -> np.ndarray:
        self._ensure_host()
        return self._kinds  # type: ignore[return-value]

    @property
    def point_x(self) -> np.ndarray:
        self._ensure_host()
        return self._point_x  # type: ignore[return-value]

    @property
    def point_y(self) -> np.ndarray:
        self._ensure_host()
        return self._point_y  # type: ignore[return-value]

    @property
    def overlap_x0(self) -> np.ndarray:
        self._ensure_host()
        return self._overlap_x0  # type: ignore[return-value]

    @property
    def overlap_y0(self) -> np.ndarray:
        self._ensure_host()
        return self._overlap_y0  # type: ignore[return-value]

    @property
    def overlap_x1(self) -> np.ndarray:
        self._ensure_host()
        return self._overlap_x1  # type: ignore[return-value]

    @property
    def overlap_y1(self) -> np.ndarray:
        self._ensure_host()
        return self._overlap_y1  # type: ignore[return-value]

    @property
    def ambiguous_rows(self) -> np.ndarray:
        self._ensure_host()
        return self._ambiguous_rows  # type: ignore[return-value]

    @property
    def count(self) -> int:
        if self._count > 0:
            return self._count
        if self.device_state is not None and self.device_state.left_rows is not None:
            return int(self.device_state.left_rows.size)
        if self._left_rows is not None:
            return int(self._left_rows.size)
        return 0

    def kind_names(self) -> list[str]:
        return [SegmentIntersectionKind(int(value)).name.lower() for value in self.kinds]


@dataclass(frozen=True)
class PagedSegmentIntersectionResult:
    """Device-classified intersection pages beyond one contiguous relation budget."""

    pages: tuple[SegmentIntersectionResult, ...]
    candidate_pairs: int
    runtime_selection: RuntimeSelection
    precision_plan: PrecisionPlan
    robustness_plan: RobustnessPlan
    compact_non_disjoint: bool = False
    classified_count: int | None = None
    page_count: int | None = None

    @property
    def count(self) -> int:
        if self.classified_count is not None:
            return int(self.classified_count)
        return sum(page.count for page in self.pages)

    def kind_names(self) -> list[str]:
        return [name for page in self.pages for name in page.kind_names()]


@dataclass(frozen=True)
class SegmentIntersectionBenchmark:
    rows_left: int
    rows_right: int
    candidate_pairs: int
    disjoint_pairs: int
    proper_pairs: int
    touch_pairs: int
    overlap_pairs: int
    ambiguous_pairs: int
    elapsed_seconds: float


@dataclass(frozen=True)
class SegmentLocalEventSummary:
    """Per-row exact local-event summary derived from segment intersections."""

    runtime_selection: RuntimeSelection
    precision_plan: PrecisionPlan
    robustness_plan: RobustnessPlan
    candidate_pairs: int
    point_intersection_count: int
    parallel_or_colinear_candidate_count: int
    row_point_intersection_counts: np.ndarray
    exact_event_counts: np.ndarray
    exact_interval_upper_bounds: np.ndarray

    @property
    def max_exact_events(self) -> int:
        return int(self.exact_event_counts.max(initial=0))


@dataclass(frozen=True)
class SegmentIntersectionDeviceState:
    left_rows: DeviceArray
    left_segments: DeviceArray
    left_lookup: DeviceArray
    right_rows: DeviceArray
    right_segments: DeviceArray
    right_lookup: DeviceArray
    kinds: DeviceArray
    point_x: DeviceArray
    point_y: DeviceArray
    overlap_x0: DeviceArray
    overlap_y0: DeviceArray
    overlap_x1: DeviceArray
    overlap_y1: DeviceArray
    ambiguous_rows: DeviceArray


@dataclass(frozen=True)
class SegmentIntersectionCandidates:
    left_rows: np.ndarray
    left_segments: np.ndarray
    left_lookup: np.ndarray
    right_rows: np.ndarray
    right_segments: np.ndarray
    right_lookup: np.ndarray
    pairs_examined: int
    tile_size: int

    @property
    def count(self) -> int:
        return int(self.left_rows.size)


@dataclass(frozen=True)
class DeviceSegmentIntersectionCandidates:
    """GPU-resident candidate pairs from sweep-based spatial join."""

    left_rows: DeviceArray
    left_segments: DeviceArray
    left_lookup: DeviceArray
    right_rows: DeviceArray
    right_segments: DeviceArray
    right_lookup: DeviceArray
    count: int


@dataclass(frozen=True)
class DeviceSegmentIntersectionCandidatePages:
    """Marker for candidate pages consumed without contiguous concatenation."""

    count: int


def _compact_non_disjoint_segment_intersection_page(
    result: SegmentIntersectionResult,
) -> SegmentIntersectionResult:
    """Compact one classified page to rows that can emit split events."""
    import cupy as cp

    state = result.device_state
    if state is None:
        return result
    d_live_rows = cp.flatnonzero(
        cp.asarray(state.kinds, dtype=cp.int8) != cp.int8(SegmentIntersectionKind.DISJOINT)
    ).astype(cp.int64, copy=False)
    if int(d_live_rows.size) == result.count:
        return result

    d_ambiguous = cp.asarray(state.ambiguous_rows, dtype=cp.int64)
    if int(d_ambiguous.size) > 0 and int(d_live_rows.size) > 0:
        d_inverse = cp.full(result.count, -1, dtype=cp.int64)
        d_inverse[d_live_rows] = cp.arange(d_live_rows.size, dtype=cp.int64)
        d_remapped_ambiguous = d_inverse[d_ambiguous]
        d_remapped_ambiguous = d_remapped_ambiguous[d_remapped_ambiguous >= 0].astype(
            cp.int32, copy=False
        )
    else:
        d_remapped_ambiguous = cp.empty(0, dtype=cp.int32)

    def _take(values):
        return cp.asarray(values)[d_live_rows]

    return SegmentIntersectionResult(
        candidate_pairs=result.candidate_pairs,
        runtime_selection=result.runtime_selection,
        precision_plan=result.precision_plan,
        robustness_plan=result.robustness_plan,
        device_state=SegmentIntersectionDeviceState(
            left_rows=_take(state.left_rows),
            left_segments=_take(state.left_segments),
            left_lookup=_take(state.left_lookup),
            right_rows=_take(state.right_rows),
            right_segments=_take(state.right_segments),
            right_lookup=_take(state.right_lookup),
            kinds=_take(state.kinds),
            point_x=_take(state.point_x),
            point_y=_take(state.point_y),
            overlap_x0=_take(state.overlap_x0),
            overlap_y0=_take(state.overlap_y0),
            overlap_x1=_take(state.overlap_x1),
            overlap_y1=_take(state.overlap_y1),
            ambiguous_rows=d_remapped_ambiguous,
        ),
        _count=int(d_live_rows.size),
    )


def concatenate_paged_segment_intersections_device(
    result: PagedSegmentIntersectionResult,
) -> SegmentIntersectionResult:
    """Globally concatenate compact classified pages for ordered consumers."""
    import cupy as cp

    pages = tuple(page for page in result.pages if page.count > 0)
    if not pages:
        return _empty_segment_intersection_result(
            runtime_selection=result.runtime_selection,
            precision_plan=result.precision_plan,
            robustness_plan=result.robustness_plan,
        )
    states = tuple(page.device_state for page in pages)
    if any(state is None for state in states):
        raise RuntimeError("paged device intersection result contains a host page")

    def _concat(field):
        return cp.concatenate(tuple(cp.asarray(getattr(state, field)) for state in states))

    ambiguous_parts = []
    row_offset = 0
    for page, state in zip(pages, states, strict=True):
        d_ambiguous = cp.asarray(state.ambiguous_rows, dtype=cp.int64)
        if d_ambiguous.size > 0:
            ambiguous_parts.append(d_ambiguous + cp.int64(row_offset))
        row_offset += page.count
    d_ambiguous_rows = (
        cp.concatenate(tuple(ambiguous_parts)).astype(cp.int32, copy=False)
        if ambiguous_parts
        else cp.empty(0, dtype=cp.int32)
    )
    return SegmentIntersectionResult(
        candidate_pairs=result.candidate_pairs,
        runtime_selection=result.runtime_selection,
        precision_plan=result.precision_plan,
        robustness_plan=result.robustness_plan,
        device_state=SegmentIntersectionDeviceState(
            left_rows=_concat("left_rows"),
            left_segments=_concat("left_segments"),
            left_lookup=_concat("left_lookup"),
            right_rows=_concat("right_rows"),
            right_segments=_concat("right_segments"),
            right_lookup=_concat("right_lookup"),
            kinds=_concat("kinds"),
            point_x=_concat("point_x"),
            point_y=_concat("point_y"),
            overlap_x0=_concat("overlap_x0"),
            overlap_y0=_concat("overlap_y0"),
            overlap_x1=_concat("overlap_x1"),
            overlap_y1=_concat("overlap_y1"),
            ambiguous_rows=d_ambiguous_rows,
        ),
        _count=sum(page.count for page in pages),
    )


# ---------------------------------------------------------------------------
# NVRTC compilation and warmup
# ---------------------------------------------------------------------------

from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup(
    [
        ("segment-extract-fp64", EXTRACT_SOURCE_FP64, _SEGMENT_EXTRACT_KERNEL_NAMES),
        ("segment-extract-fp32", EXTRACT_SOURCE_FP32, _SEGMENT_EXTRACT_KERNEL_NAMES),
        ("segment-classify-fp64", CLASSIFY_SOURCE_FP64, _SEGMENT_CLASSIFY_KERNEL_NAMES),
        ("segment-classify-fp32", CLASSIFY_SOURCE_FP32, _SEGMENT_CLASSIFY_KERNEL_NAMES),
        (
            "segment-same-row-candidates",
            _SAME_ROW_CANDIDATE_KERNEL_SOURCE,
            _SAME_ROW_CANDIDATE_KERNEL_NAMES,
        ),
        (
            "segment-sweep-candidates",
            _SWEEP_CANDIDATE_KERNEL_SOURCE,
            _SWEEP_CANDIDATE_KERNEL_NAMES,
        ),
    ]
)


def _extract_kernels(compute_type: str = "double"):
    source = format_extract_source(compute_type)
    name = f"segment-extract-{compute_type.replace('double', 'fp64').replace('float', 'fp32')}"
    return compile_kernel_group(name, source, _SEGMENT_EXTRACT_KERNEL_NAMES)


def _classify_kernels(compute_type: str = "double"):
    source = format_classify_source(compute_type)
    name = f"segment-classify-{compute_type.replace('double', 'fp64').replace('float', 'fp32')}"
    return compile_kernel_group(name, source, _SEGMENT_CLASSIFY_KERNEL_NAMES)


def _same_row_candidate_kernels():
    return compile_kernel_group(
        "segment-same-row-candidates",
        _SAME_ROW_CANDIDATE_KERNEL_SOURCE,
        _SAME_ROW_CANDIDATE_KERNEL_NAMES,
    )


def _sweep_candidate_kernels():
    return compile_kernel_group(
        "segment-sweep-candidates",
        _SWEEP_CANDIDATE_KERNEL_SOURCE,
        _SWEEP_CANDIDATE_KERNEL_NAMES,
    )


@dataclass
class _PendingSegmentFamily:
    family: GeometryFamily
    buffer: DeviceArray
    valid_rows: DeviceArray
    empty_mask: DeviceArray
    geometry_offsets: DeviceArray
    part_offsets: DeviceArray
    ring_offsets: DeviceArray
    segment_counts: DeviceArray | None
    segment_offsets: DeviceArray
    row_count: int
    total_segments: int | None
    segment_capacity: int | None = None
    segment_capacity_per_row: int | None = None


def _host_segment_total_for_family(
    geometry_array: OwnedGeometryArray,
    family: GeometryFamily,
    expected_rows: int,
) -> int | None:
    """Return a host-proven segment total when structural offsets are present.

    Device-resident owned arrays often still carry host routing metadata and
    structural offsets from ingestion or an admitted host-known take.  Segment
    extraction needs device per-row counts for scatter offsets, but allocation
    only needs the family total.  Reusing already-known offsets avoids a scalar
    D2H fence without changing the device execution shape.
    """
    validity = geometry_array._validity
    tags = geometry_array._tags
    family_row_offsets = geometry_array._family_row_offsets
    buffer = geometry_array.families.get(family)
    if (
        validity is None
        or tags is None
        or family_row_offsets is None
        or buffer is None
        or buffer.geometry_offsets.size == 0
        or buffer.empty_mask.size == 0
    ):
        return None
    row_count = geometry_array.row_count
    if (
        int(validity.size) != row_count
        or int(tags.size) != row_count
        or int(family_row_offsets.size) != row_count
    ):
        return None

    family_mask = np.asarray(validity, dtype=bool) & (
        np.asarray(tags, dtype=np.int8) == np.int8(FAMILY_TAGS[family])
    )
    if int(np.count_nonzero(family_mask)) != int(expected_rows):
        return None
    if expected_rows == 0:
        return 0

    family_rows = np.asarray(family_row_offsets[family_mask], dtype=np.int64)
    if family_rows.size == 0:
        return 0
    if int(family_rows.min(initial=0)) < 0:
        return None
    max_row = int(family_rows.max(initial=-1))
    if max_row + 1 >= int(buffer.geometry_offsets.size):
        return None
    if max_row >= int(buffer.empty_mask.size):
        return None

    active_rows = family_rows[~np.asarray(buffer.empty_mask[family_rows], dtype=bool)]
    if active_rows.size == 0:
        return 0

    geom_offsets = np.asarray(buffer.geometry_offsets, dtype=np.int64)

    if family is GeometryFamily.LINESTRING:
        lengths = geom_offsets[active_rows + 1] - geom_offsets[active_rows]
        return int(np.maximum(lengths - 1, 0).sum())

    if family is GeometryFamily.POLYGON:
        ring_offsets = buffer.ring_offsets
        if ring_offsets is None or ring_offsets.size == 0:
            return None
        return _host_nested_segment_total(
            geom_offsets,
            np.asarray(ring_offsets, dtype=np.int64),
            active_rows,
        )

    if family is GeometryFamily.MULTILINESTRING:
        part_offsets = buffer.part_offsets
        if part_offsets is None or part_offsets.size == 0:
            return None
        return _host_nested_segment_total(
            geom_offsets,
            np.asarray(part_offsets, dtype=np.int64),
            active_rows,
        )

    if family is GeometryFamily.MULTIPOLYGON:
        part_offsets = buffer.part_offsets
        ring_offsets = buffer.ring_offsets
        if (
            part_offsets is None
            or part_offsets.size == 0
            or ring_offsets is None
            or ring_offsets.size == 0
        ):
            return None
        return _host_multipolygon_segment_total(
            geom_offsets,
            np.asarray(part_offsets, dtype=np.int64),
            np.asarray(ring_offsets, dtype=np.int64),
            active_rows,
        )

    return None


def _host_max_segment_count_for_family(
    geometry_array: OwnedGeometryArray,
    family: GeometryFamily,
    expected_rows: int,
) -> int | None:
    """Return an exact host-known maximum row span when offsets are retained."""
    validity = geometry_array._validity
    tags = geometry_array._tags
    family_row_offsets = geometry_array._family_row_offsets
    buffer = geometry_array.families.get(family)
    if (
        validity is None
        or tags is None
        or family_row_offsets is None
        or buffer is None
        or buffer.geometry_offsets.size == 0
        or buffer.empty_mask.size == 0
    ):
        return None
    row_count = geometry_array.row_count
    if (
        int(validity.size) != row_count
        or int(tags.size) != row_count
        or int(family_row_offsets.size) != row_count
    ):
        return None

    family_mask = np.asarray(validity, dtype=bool) & (
        np.asarray(tags, dtype=np.int8) == np.int8(FAMILY_TAGS[family])
    )
    if int(np.count_nonzero(family_mask)) != int(expected_rows):
        return None
    if expected_rows == 0:
        return 0

    family_rows = np.asarray(family_row_offsets[family_mask], dtype=np.int64)
    if family_rows.size == 0:
        return 0
    if int(family_rows.min(initial=0)) < 0:
        return None
    max_row = int(family_rows.max(initial=-1))
    if max_row + 1 >= int(buffer.geometry_offsets.size):
        return None
    if max_row >= int(buffer.empty_mask.size):
        return None

    active_rows = family_rows[~np.asarray(buffer.empty_mask[family_rows], dtype=bool)]
    if active_rows.size == 0:
        return 0
    geometry_offsets = np.asarray(buffer.geometry_offsets, dtype=np.int64)

    if family is GeometryFamily.LINESTRING:
        lengths = geometry_offsets[active_rows + 1] - geometry_offsets[active_rows]
        return int(np.maximum(lengths - 1, 0).max(initial=0))

    if family in (GeometryFamily.POLYGON, GeometryFamily.MULTILINESTRING):
        leaf_offsets = (
            buffer.ring_offsets
            if family is GeometryFamily.POLYGON
            else buffer.part_offsets
        )
        if leaf_offsets is None or leaf_offsets.size == 0:
            return None
        starts = geometry_offsets[active_rows]
        ends = geometry_offsets[active_rows + 1]
        leaf_offsets = np.asarray(leaf_offsets, dtype=np.int64)
        if int(starts.min(initial=0)) < 0 or int(ends.max(initial=0)) >= int(
            leaf_offsets.size
        ):
            return None
        leaf_counts = np.maximum(np.diff(leaf_offsets) - 1, 0)
        prefix = np.empty(leaf_counts.size + 1, dtype=np.int64)
        prefix[0] = 0
        np.cumsum(leaf_counts, out=prefix[1:])
        return int((prefix[ends] - prefix[starts]).max(initial=0))

    if family is GeometryFamily.MULTIPOLYGON:
        part_offsets = buffer.part_offsets
        ring_offsets = buffer.ring_offsets
        if (
            part_offsets is None
            or part_offsets.size == 0
            or ring_offsets is None
            or ring_offsets.size == 0
        ):
            return None
        starts = geometry_offsets[active_rows]
        ends = geometry_offsets[active_rows + 1]
        part_offsets = np.asarray(part_offsets, dtype=np.int64)
        ring_offsets = np.asarray(ring_offsets, dtype=np.int64)
        if (
            int(starts.min(initial=0)) < 0
            or int(ends.max(initial=0)) >= int(part_offsets.size)
            or int(part_offsets.min(initial=0)) < 0
            or int(part_offsets.max(initial=0)) >= int(ring_offsets.size)
        ):
            return None
        ring_counts = np.maximum(np.diff(ring_offsets) - 1, 0)
        ring_prefix = np.empty(ring_counts.size + 1, dtype=np.int64)
        ring_prefix[0] = 0
        np.cumsum(ring_counts, out=ring_prefix[1:])
        part_counts = ring_prefix[part_offsets[1:]] - ring_prefix[part_offsets[:-1]]
        part_prefix = np.empty(part_counts.size + 1, dtype=np.int64)
        part_prefix[0] = 0
        np.cumsum(part_counts, out=part_prefix[1:])
        return int((part_prefix[ends] - part_prefix[starts]).max(initial=0))

    return 0


def _host_nested_segment_total(
    geometry_offsets: np.ndarray,
    leaf_offsets: np.ndarray,
    rows: np.ndarray,
) -> int | None:
    starts = geometry_offsets[rows]
    ends = geometry_offsets[rows + 1]
    if starts.size == 0:
        return 0
    if int(starts.min(initial=0)) < 0 or int(ends.max(initial=0)) >= int(leaf_offsets.size):
        return None
    leaf_segment_counts = np.maximum(np.diff(leaf_offsets) - 1, 0)
    prefix = np.empty(leaf_segment_counts.size + 1, dtype=np.int64)
    prefix[0] = 0
    np.cumsum(leaf_segment_counts, out=prefix[1:])
    return int((prefix[ends] - prefix[starts]).sum())


def _host_multipolygon_segment_total(
    geometry_offsets: np.ndarray,
    part_offsets: np.ndarray,
    ring_offsets: np.ndarray,
    rows: np.ndarray,
) -> int | None:
    part_starts = geometry_offsets[rows]
    part_ends = geometry_offsets[rows + 1]
    if part_starts.size == 0:
        return 0
    if int(part_starts.min(initial=0)) < 0 or int(part_ends.max(initial=0)) >= int(
        part_offsets.size
    ):
        return None
    if int(part_offsets.min(initial=0)) < 0 or int(part_offsets.max(initial=0)) >= int(
        ring_offsets.size
    ):
        return None

    ring_segment_counts = np.maximum(np.diff(ring_offsets) - 1, 0)
    ring_prefix = np.empty(ring_segment_counts.size + 1, dtype=np.int64)
    ring_prefix[0] = 0
    np.cumsum(ring_segment_counts, out=ring_prefix[1:])

    part_segment_counts = ring_prefix[part_offsets[1:]] - ring_prefix[part_offsets[:-1]]
    part_prefix = np.empty(part_segment_counts.size + 1, dtype=np.int64)
    part_prefix[0] = 0
    np.cumsum(part_segment_counts, out=part_prefix[1:])
    return int((part_prefix[part_ends] - part_prefix[part_starts]).sum())


def _device_structural_segment_total_for_family(
    family: GeometryFamily,
    buffer,
) -> int | None:
    """Return segment totals proved purely by device allocation metadata.

    Variable nested buffers cannot infer exact segment cardinality from
    ``coord_count - leaf_count`` because exact constructive outputs may carry
    zero-coordinate structural leaves.  In that shape the count kernel is the
    source of truth and the allocation total must come from its device counts.
    """
    if (
        family is GeometryFamily.POLYGON
        and getattr(buffer, "dense_single_ring_width", None) is not None
    ):
        width = int(buffer.dense_single_ring_width)
        if width > 1:
            return (int(buffer.geometry_offsets.size) - 1) * (width - 1)
    return None


def _device_segment_capacity_for_family(
    family: GeometryFamily,
    buffer,
    *,
    indexed_view: bool,
    row_count: int,
    unique_family_rows: bool = False,
    active_family_row_multiplicity_bound: int | None = None,
    active_family_row_segment_capacity_bound: int | None = None,
) -> tuple[int, int | None]:
    """Return a device-resident upper bound for capacity scatter.

    For non-indexed buffers each segment-producing leaf contributes at most one
    segment per coordinate.  ``x.size`` is therefore a safe exact allocation
    upper bound even when a malformed or empty leaf would make
    ``coord_count - leaf_count`` unsafe.  Indexed views repeat base rows; the
    safe per-row upper bound comes from trusted maximum row-width metadata.
    Only carriers without that structural proof use the full base coordinate
    allocation. When the whole indexed-view capacity would be too large,
    callers batch that per-row capacity instead of exporting the exact total.
    """
    if family not in (
        GeometryFamily.LINESTRING,
        GeometryFamily.POLYGON,
        GeometryFamily.MULTILINESTRING,
        GeometryFamily.MULTIPOLYGON,
    ):
        raise ValueError(f"unsupported segment-producing family: {family!r}")
    unit_capacity = max(int(buffer.x.size), 0)
    if not indexed_view:
        return unit_capacity, None
    if active_family_row_segment_capacity_bound is not None:
        per_row_capacity = int(active_family_row_segment_capacity_bound)
        total_capacity = row_count * per_row_capacity
        if total_capacity <= _SEGMENT_EXTRACTION_CAPACITY_MAX_SLOTS:
            return total_capacity, None
        return None, per_row_capacity
    if active_family_row_multiplicity_bound is not None:
        total_capacity = unit_capacity * int(active_family_row_multiplicity_bound)
        if total_capacity <= _SEGMENT_EXTRACTION_CAPACITY_MAX_SLOTS:
            return total_capacity, None
        return None, unit_capacity
    fixed_size = getattr(buffer, "fixed_size", None)
    max_coord_count = (
        None if fixed_size is None else getattr(fixed_size, "max_coord_count_per_row", None)
    )
    if unique_family_rows:
        if unit_capacity <= _SEGMENT_EXTRACTION_CAPACITY_MAX_SLOTS:
            return unit_capacity, None
        return None, unit_capacity if max_coord_count is None else int(max_coord_count)
    if max_coord_count is not None:
        unit_capacity = max(int(max_coord_count), 0)
    row_count = max(int(row_count), 0)
    total_capacity = row_count * unit_capacity
    if total_capacity <= _SEGMENT_EXTRACTION_CAPACITY_MAX_SLOTS:
        return total_capacity, None
    return None, unit_capacity


# Kernel 1 dispatch: GPU Segment Extraction
# ---------------------------------------------------------------------------


def _extract_segments_gpu(
    geometry_array: OwnedGeometryArray,
    compute_type: str = "double",
) -> DeviceSegmentTable:
    """Extract all segments from a geometry array entirely on GPU.

    Uses the count-scatter pattern:
    1. Count segments per valid geometry row
    2. Exclusive prefix sum for write offsets
    3. Scatter segment endpoints to output SoA arrays
    """
    import cupy as cp

    runtime = get_cuda_runtime()
    d_state = geometry_array._ensure_device_state(preserve_indexed_view=True)

    # The count_segments / scatter_segments kernels declare family_codes as
    # ``const int*`` (int32), but d_state.tags is int8.  Passing an int8
    # pointer to a kernel that reads 4-byte ints causes every thread to read
    # a garbage family code, producing zero segment counts and (when the
    # underlying memory layout changes) an illegal-address fault.
    d_family_codes = (
        d_state.tags.astype(cp.int32) if d_state.tags.dtype != cp.int32 else d_state.tags
    )
    d_family_row_offsets = d_state.family_row_offsets

    # We need unified offset arrays across all families.
    # Build concatenated offset arrays: for each valid row, we need the
    # correct family's offsets. We concatenate all family offset arrays
    # and build an offset base per family so the kernel can index correctly.
    #
    # Strategy: since the kernel accesses offsets by family_row_offsets[global_row],
    # which gives the row index within that family's buffer, and each family
    # has its own device offset arrays, we need to provide per-family offset
    # pointers. The simplest approach: one kernel launch per family. But that
    # loses the benefit of a single bulk launch.
    #
    # Better: build a unified offset table on device by concatenating family
    # offsets with base pointers. However, the kernel design above already
    # takes family code as input and does the right thing per family.
    # The problem is that different families store their offsets in different
    # device arrays. We need to either:
    #   (a) Pass all family offset arrays as separate kernel params, or
    #   (b) Build unified offset arrays by concatenating and adjusting.
    #
    # For maximum simplicity and GPU-residency, we use approach (a):
    # launch per-family kernels. With only 4 families this is 4 launches
    # max, all on the same stream (no sync needed between them).

    # However, approach (a) with separate kernels is cleaner with the count-scatter
    # pattern since each family produces different counts. Let's use a different
    # strategy: per-family count-scatter with a final concat.

    # Compile extraction kernels once (SHA1-cached), not per-family.
    kernels = _extract_kernels(compute_type)

    all_row_idx = []
    all_seg_idx = []
    all_part_idx = []
    all_ring_idx = []
    all_x0 = []
    all_y0 = []
    all_x1 = []
    all_y1 = []
    total_segments = 0
    max_segments_per_row = 0
    pending_families: list[_PendingSegmentFamily] = []
    scatter_scratch = []

    for family_enum, family_tag in [
        (GeometryFamily.LINESTRING, _FAMILY_LINESTRING),
        (GeometryFamily.POLYGON, _FAMILY_POLYGON),
        (GeometryFamily.MULTILINESTRING, _FAMILY_MULTILINESTRING),
        (GeometryFamily.MULTIPOLYGON, _FAMILY_MULTIPOLYGON),
    ]:
        if family_enum not in d_state.families:
            continue
        d_buf = d_state.families[family_enum]

        # Valid rows for this family. Keep the mask and selected row ids on
        # device; segment extraction is a hot overlay primitive and must not
        # materialize row metadata just to decide per-family launch spans.
        fam_valid_mask = d_state.validity & (d_state.tags == family_tag)
        d_fam_valid = cp.flatnonzero(fam_valid_mask).astype(cp.int32, copy=False)
        n_fam = int(d_fam_valid.size)
        if n_fam == 0:
            continue

        d_fam_row_off = d_family_row_offsets[d_fam_valid].astype(cp.int64, copy=False)
        dense_polygon_width = (
            family_enum is GeometryFamily.POLYGON
            and d_buf.dense_single_ring_width is not None
            and int(d_buf.dense_single_ring_width) > 1
        )
        family_row_segment_bound = _host_max_segment_count_for_family(
            geometry_array,
            family_enum,
            n_fam,
        )
        if family_row_segment_bound is None:
            inherited_bound = getattr(
                geometry_array,
                "_active_family_row_segment_capacity_bound",
                None,
            )
            fixed_size = getattr(d_buf, "fixed_size", None)
            fixed_coord_bound = (
                None
                if fixed_size is None
                else getattr(fixed_size, "max_coord_count_per_row", None)
            )
            if dense_polygon_width:
                family_row_segment_bound = int(d_buf.dense_single_ring_width) - 1
            elif inherited_bound is not None:
                family_row_segment_bound = int(inherited_bound)
            elif fixed_coord_bound is not None:
                family_row_segment_bound = int(fixed_coord_bound)
            else:
                family_row_segment_bound = int(d_buf.x.size)
        max_segments_per_row = max(
            max_segments_per_row,
            int(family_row_segment_bound),
        )
        if dense_polygon_width:
            d_fam_empty = cp.zeros(n_fam, dtype=cp.uint8)
        else:
            d_fam_empty = d_buf.empty_mask[d_fam_row_off].astype(cp.uint8, copy=True)

        # Part and ring offsets (use zeros if not available)
        d_geom_off = d_buf.geometry_offsets
        d_part_off = (
            d_buf.part_offsets if d_buf.part_offsets is not None else d_buf.geometry_offsets
        )
        d_ring_off = (
            d_buf.ring_offsets if d_buf.ring_offsets is not None else d_buf.geometry_offsets
        )

        ptr = runtime.pointer

        if dense_polygon_width:
            # Fixed-width one-ring polygons prove their segment count from
            # metadata: a closed ring with W coords has W - 1 edges. Avoid the
            # count kernel and scalar total-size D2H fence for this common
            # rectangle/buffer-like shape.
            segments_per_row = int(d_buf.dense_single_ring_width) - 1
            fam_total = n_fam * segments_per_row
            d_seg_counts = None
            d_seg_offsets = cp.arange(n_fam, dtype=cp.int32) * np.int32(
                segments_per_row,
            )
            fam_capacity = None
            fam_capacity_per_row = None
        else:
            # Step 1: Count segments
            d_seg_counts = runtime.allocate((n_fam,), np.int32, zero=True)
            count_kernel = kernels["count_segments"]

            count_params = (
                (
                    ptr(d_fam_valid),
                    ptr(d_family_codes),
                    ptr(d_family_row_offsets),
                    ptr(d_geom_off),
                    ptr(d_part_off),
                    ptr(d_ring_off),
                    ptr(d_fam_empty),
                    ptr(d_seg_counts),
                    n_fam,
                ),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                ),
            )
            grid, block = runtime.launch_config(count_kernel, n_fam)
            runtime.launch(count_kernel, grid=grid, block=block, params=count_params)

            # Step 2: Exclusive prefix sum for write offsets
            d_seg_offsets = exclusive_sum(d_seg_counts, synchronize=False)
            fam_total = _host_segment_total_for_family(
                geometry_array,
                family_enum,
                n_fam,
            )
            if fam_total is None and not geometry_array.is_indexed_view:
                fam_total = _device_structural_segment_total_for_family(
                    family_enum,
                    d_buf,
                )
            fam_capacity = None
            fam_capacity_per_row = None
            if fam_total is None:
                fam_capacity, fam_capacity_per_row = _device_segment_capacity_for_family(
                    family_enum,
                    d_buf,
                    indexed_view=geometry_array.is_indexed_view,
                    row_count=n_fam,
                    unique_family_rows=d_state.trusted_unique_family_rows is True,
                    active_family_row_multiplicity_bound=(
                        geometry_array._active_family_row_multiplicity_bound
                        if geometry_array._row_active_mask is not None
                        else None
                    ),
                    active_family_row_segment_capacity_bound=(
                        geometry_array._active_family_row_segment_capacity_bound
                        if geometry_array._row_active_mask is not None
                        else None
                    ),
                )

        pending_families.append(
            _PendingSegmentFamily(
                family=family_enum,
                buffer=d_buf,
                valid_rows=d_fam_valid,
                empty_mask=d_fam_empty,
                geometry_offsets=d_geom_off,
                part_offsets=d_part_off,
                ring_offsets=d_ring_off,
                segment_counts=d_seg_counts,
                segment_offsets=d_seg_offsets,
                row_count=n_fam,
                total_segments=fam_total,
                segment_capacity=fam_capacity,
                segment_capacity_per_row=fam_capacity_per_row,
            )
        )

    ptr = runtime.pointer
    for pending in pending_families:
        family_enum = pending.family
        d_buf = pending.buffer
        d_fam_valid = pending.valid_rows
        d_fam_empty = pending.empty_mask
        d_geom_off = pending.geometry_offsets
        d_part_off = pending.part_offsets
        d_ring_off = pending.ring_offsets
        d_seg_counts = pending.segment_counts
        d_seg_offsets = pending.segment_offsets
        n_fam = pending.row_count
        capacity_scatter = pending.total_segments is None and pending.segment_capacity is not None
        batched_capacity_scatter = (
            pending.total_segments is None and pending.segment_capacity_per_row is not None
        )
        if batched_capacity_scatter:
            unit_capacity = int(pending.segment_capacity_per_row or 0)
            if unit_capacity <= 0 or n_fam <= 0:
                runtime.free(d_fam_valid)
                runtime.free(d_fam_empty)
                runtime.free(d_seg_counts)
                runtime.free(d_seg_offsets)
                continue
            scatter_kernel = kernels["scatter_segments"]
            family_total = 0
            d_selected_rows = cp.arange(n_fam, dtype=cp.int32)
            selected_count = n_fam
            tier_start = 0
            preferred_tier_width = 8
            while selected_count > 0 and tier_start < unit_capacity:
                tier_width = min(
                    unit_capacity - tier_start,
                    max(1, _SEGMENT_EXTRACTION_CAPACITY_MAX_SLOTS // selected_count),
                    preferred_tier_width,
                )
                tier_capacity = selected_count * tier_width
                d_batch_valid = d_fam_valid[d_selected_rows]
                d_batch_empty = d_fam_empty[d_selected_rows]
                d_batch_offsets = cp.arange(selected_count, dtype=cp.int32) * np.int32(
                    tier_width
                )

                d_out_row = cp.full(tier_capacity, -1, dtype=cp.int32)
                d_out_seg = cp.full(tier_capacity, -1, dtype=cp.int32)
                d_out_part = cp.full(tier_capacity, -1, dtype=cp.int32)
                d_out_ring = cp.full(tier_capacity, -1, dtype=cp.int32)
                d_out_x0 = cp.full(tier_capacity, cp.nan, dtype=cp.float64)
                d_out_y0 = cp.full(tier_capacity, cp.nan, dtype=cp.float64)
                d_out_x1 = cp.full(tier_capacity, cp.nan, dtype=cp.float64)
                d_out_y1 = cp.full(tier_capacity, cp.nan, dtype=cp.float64)

                scatter_params = (
                    (
                        ptr(d_batch_valid),
                        ptr(d_family_codes),
                        ptr(d_family_row_offsets),
                        ptr(d_geom_off),
                        ptr(d_part_off),
                        ptr(d_ring_off),
                        ptr(d_batch_empty),
                        ptr(d_buf.x),
                        ptr(d_buf.y),
                        ptr(d_batch_offsets),
                        ptr(d_out_row),
                        ptr(d_out_seg),
                        ptr(d_out_part),
                        ptr(d_out_ring),
                        ptr(d_out_x0),
                        ptr(d_out_y0),
                        ptr(d_out_x1),
                        ptr(d_out_y1),
                        selected_count,
                        tier_start,
                        tier_width,
                    ),
                    (
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_I32,
                        KERNEL_PARAM_I32,
                        KERNEL_PARAM_I32,
                    ),
                )
                grid, block = runtime.launch_config(scatter_kernel, selected_count)
                runtime.launch(
                    scatter_kernel,
                    grid=grid,
                    block=block,
                    params=scatter_params,
                )
                live_compact = compact_indices(
                    (d_out_row >= 0).astype(cp.uint8, copy=False),
                )
                if live_compact.count > 0:
                    live = live_compact.values
                    all_row_idx.append(d_out_row[live])
                    all_seg_idx.append(d_out_seg[live])
                    all_part_idx.append(d_out_part[live])
                    all_ring_idx.append(d_out_ring[live])
                    all_x0.append(d_out_x0[live])
                    all_y0.append(d_out_y0[live])
                    all_x1.append(d_out_x1[live])
                    all_y1.append(d_out_y1[live])
                    family_total += int(live_compact.count)

                tier_start += tier_width
                if tier_start >= unit_capacity:
                    break
                d_selected_rows = compact_indices(
                    (
                        cp.asarray(d_seg_counts, dtype=cp.int32) > tier_start
                    ).astype(cp.uint8, copy=False)
                ).values
                selected_count = d_selected_rows.shape[0]
                preferred_tier_width = min(
                    preferred_tier_width * 2,
                    unit_capacity - tier_start,
                )

            total_segments += family_total
            scatter_scratch.extend(
                (d_fam_valid, d_fam_empty, d_seg_counts, d_seg_offsets),
            )
            continue

        fam_total = int(
            pending.total_segments
            if pending.total_segments is not None
            else (pending.segment_capacity or 0)
        )
        if fam_total == 0:
            runtime.free(d_fam_valid)
            runtime.free(d_fam_empty)
            runtime.free(d_seg_counts)
            runtime.free(d_seg_offsets)
            continue

        # Step 3: Allocate and scatter
        if capacity_scatter:
            d_out_row = cp.full(fam_total, -1, dtype=cp.int32)
            d_out_seg = cp.full(fam_total, -1, dtype=cp.int32)
            d_out_part = cp.full(fam_total, -1, dtype=cp.int32)
            d_out_ring = cp.full(fam_total, -1, dtype=cp.int32)
            d_out_x0 = cp.full(fam_total, cp.nan, dtype=cp.float64)
            d_out_y0 = cp.full(fam_total, cp.nan, dtype=cp.float64)
            d_out_x1 = cp.full(fam_total, cp.nan, dtype=cp.float64)
            d_out_y1 = cp.full(fam_total, cp.nan, dtype=cp.float64)
        else:
            d_out_row = runtime.allocate((fam_total,), np.int32)
            d_out_seg = runtime.allocate((fam_total,), np.int32)
            d_out_part = runtime.allocate((fam_total,), np.int32)
            d_out_ring = runtime.allocate((fam_total,), np.int32)
            d_out_x0 = runtime.allocate((fam_total,), np.float64)
            d_out_y0 = runtime.allocate((fam_total,), np.float64)
            d_out_x1 = runtime.allocate((fam_total,), np.float64)
            d_out_y1 = runtime.allocate((fam_total,), np.float64)

        scatter_kernel = kernels["scatter_segments"]
        scatter_params = (
            (
                ptr(d_fam_valid),
                ptr(d_family_codes),
                ptr(d_family_row_offsets),
                ptr(d_geom_off),
                ptr(d_part_off),
                ptr(d_ring_off),
                ptr(d_fam_empty),
                ptr(d_buf.x),
                ptr(d_buf.y),
                ptr(d_seg_offsets),
                ptr(d_out_row),
                ptr(d_out_seg),
                ptr(d_out_part),
                ptr(d_out_ring),
                ptr(d_out_x0),
                ptr(d_out_y0),
                ptr(d_out_x1),
                ptr(d_out_y1),
                n_fam,
                0,
                fam_total,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
            ),
        )
        grid, block = runtime.launch_config(scatter_kernel, n_fam)
        runtime.launch(scatter_kernel, grid=grid, block=block, params=scatter_params)

        if capacity_scatter:
            live_compact = compact_indices((d_out_row >= 0).astype(cp.uint8, copy=False))
            if live_compact.count == 0:
                runtime.free(d_fam_valid)
                runtime.free(d_fam_empty)
                runtime.free(d_seg_counts)
                runtime.free(d_seg_offsets)
                continue
            live = live_compact.values
            d_out_row = d_out_row[live]
            d_out_seg = d_out_seg[live]
            d_out_part = d_out_part[live]
            d_out_ring = d_out_ring[live]
            d_out_x0 = d_out_x0[live]
            d_out_y0 = d_out_y0[live]
            d_out_x1 = d_out_x1[live]
            d_out_y1 = d_out_y1[live]
            fam_total = int(live_compact.count)

        all_row_idx.append(d_out_row)
        all_seg_idx.append(d_out_seg)
        all_part_idx.append(d_out_part)
        all_ring_idx.append(d_out_ring)
        all_x0.append(d_out_x0)
        all_y0.append(d_out_y0)
        all_x1.append(d_out_x1)
        all_y1.append(d_out_y1)
        total_segments += fam_total

        # Keep scatter inputs alive until the asynchronous kernel has
        # completed.  Returning these buffers to the pool immediately lets
        # later allocations overwrite row/offset metadata while the scatter
        # kernel is still reading it, which corrupts downstream overlay
        # segment tables after warmup has populated the memory pool.
        scatter_scratch.extend(
            (d_fam_valid, d_fam_empty, d_seg_counts, d_seg_offsets),
        )

    if total_segments == 0 or not all_row_idx:
        if scatter_scratch:
            get_cuda_completion_retainer().defer(
                cp.cuda.get_current_stream(),
                tuple(scatter_scratch),
                lambda _owners: None,
            )
        return DeviceSegmentTable(
            row_indices=runtime.allocate((0,), np.int32),
            segment_indices=runtime.allocate((0,), np.int32),
            x0=runtime.allocate((0,), np.float64),
            y0=runtime.allocate((0,), np.float64),
            x1=runtime.allocate((0,), np.float64),
            y1=runtime.allocate((0,), np.float64),
            count=0,
            max_segments_per_row=0,
            part_indices=runtime.allocate((0,), np.int32),
            ring_indices=runtime.allocate((0,), np.int32),
        )

    # Concatenate per-family results on device (CuPy Tier 2)
    if len(all_row_idx) == 1:
        if scatter_scratch:
            get_cuda_completion_retainer().defer(
                cp.cuda.get_current_stream(),
                tuple(scatter_scratch),
                lambda _owners: None,
            )
        return DeviceSegmentTable(
            row_indices=all_row_idx[0],
            segment_indices=all_seg_idx[0],
            x0=all_x0[0],
            y0=all_y0[0],
            x1=all_x1[0],
            y1=all_y1[0],
            count=total_segments,
            max_segments_per_row=max_segments_per_row,
            part_indices=all_part_idx[0],
            ring_indices=all_ring_idx[0],
        )

    d_rows = cp.concatenate(all_row_idx)
    d_segments = cp.concatenate(all_seg_idx)
    d_order_keys = (
        d_rows.astype(cp.uint64, copy=False) << cp.uint64(32)
    ) | d_segments.astype(cp.uint32, copy=False).astype(cp.uint64, copy=False)
    d_order = sort_pairs(
        d_order_keys,
        cp.arange(total_segments, dtype=cp.int32),
        synchronize=False,
    ).values
    concatenated = DeviceSegmentTable(
        row_indices=d_rows[d_order],
        segment_indices=d_segments[d_order],
        x0=cp.concatenate(all_x0)[d_order],
        y0=cp.concatenate(all_y0)[d_order],
        x1=cp.concatenate(all_x1)[d_order],
        y1=cp.concatenate(all_y1)[d_order],
        count=total_segments,
        max_segments_per_row=max_segments_per_row,
        part_indices=cp.concatenate(all_part_idx)[d_order],
        ring_indices=cp.concatenate(all_ring_idx)[d_order],
    )
    get_cuda_completion_retainer().defer(
        cp.cuda.get_current_stream(),
        (
            *scatter_scratch,
            *all_row_idx,
            *all_seg_idx,
            *all_part_idx,
            *all_ring_idx,
            *all_x0,
            *all_y0,
            *all_x1,
            *all_y1,
            d_rows,
            d_segments,
            d_order_keys,
            d_order,
        ),
        lambda _owners: None,
    )
    return concatenated


# ---------------------------------------------------------------------------
# Kernel 2: GPU Spatial-Index Candidate Generation (sort-based sweep)
# ---------------------------------------------------------------------------
# O(n log n) candidate generation using radix sort + binary search sweep.
#
# Algorithm:
# 1. Compute x-midpoints for all segments on both sides
# 2. Sort both sides by x-midpoint using CCCL radix_sort
# 3. For each left segment, binary search in right's sorted x-midpoints
#    to find the range of rights whose x-midpoint overlaps the left's
#    x-extent. Then filter by y-overlap.
# 4. Output candidate pair indices.
#
# This replaces the O(n^2) tiled brute-force approach.
# ---------------------------------------------------------------------------

# Peak bytes per raw candidate pair during scatter+MBR-filter:
#   2 x int32 pair arrays = 8 bytes
#   8 x float64 gathered bounds = 64 bytes
#   1 x bool overlap mask = 1 byte
#   1 x uint8 cast = 1 byte
#   ~8 bytes CuPy temporaries during boolean expression evaluation
#   ~4 bytes compact_indices output (worst case)
# Total ~86 bytes.  Use 120 for safety headroom and pool fragmentation.
_BYTES_PER_RAW_PAIR = 120

# Absolute floor: never create batches smaller than 1M pairs.
_MIN_BATCH_PAIRS = 1 * 1024 * 1024


_MAX_BATCH_PAIRS_CAP = 8 * 1024 * 1024  # 8M pairs hard cap (~960 MB peak)
_SAME_ROW_WARP_MAX_RIGHT_SEGMENTS_PER_ROW = 2048
_PREFERRED_INITIAL_SWEEP_TIER_WIDTH = 256


@dataclass(frozen=True)
class _SegmentCandidateCapacityPlan:
    right_capacity: int
    left_batch_size: int
    max_batch_pairs: int


def _compute_max_batch_pairs() -> int:
    """Return the maximum number of raw candidate pairs per batch.

    Uses the stable per-query allocation envelope when available, falling back
    to allocator or driver statistics. Applies a hard cap of 8M pairs to
    prevent OOM from fragmentation and CuPy advanced-indexing temporaries.
    """
    import cupy as cp

    from vibespatial.cuda._runtime import get_cuda_runtime

    try:
        runtime = get_cuda_runtime()
        remaining = getattr(runtime, "query_memory_remaining_bytes", None)
        if callable(remaining):
            free_bytes = int(remaining())
        else:
            stats = runtime.memory_pool_stats()
            if "free_bytes" in stats:
                free_bytes = int(stats["free_bytes"])
            else:
                free_bytes, _ = cp.cuda.Device().mem_info
    except Exception:
        return _MAX_BATCH_PAIRS_CAP

    # Use 25% of the admitted query memory, capped at _MAX_BATCH_PAIRS_CAP.
    usable_bytes = free_bytes // 4
    max_pairs = usable_bytes // _BYTES_PER_RAW_PAIR

    return min(max(max_pairs, _MIN_BATCH_PAIRS), _MAX_BATCH_PAIRS_CAP)


def _candidate_capacity_plan(
    right_capacity: int,
    *,
    max_batch_pairs: int | None = None,
) -> _SegmentCandidateCapacityPlan | None:
    """Plan fixed-capacity candidate batches from host-known table shape."""
    right_capacity = int(right_capacity)
    if right_capacity <= 0:
        return None
    max_batch_pairs = min(
        _compute_max_batch_pairs(),
        int(max_batch_pairs) if max_batch_pairs is not None else _MAX_BATCH_PAIRS_CAP,
    )
    if right_capacity > max_batch_pairs:
        return None
    return _SegmentCandidateCapacityPlan(
        right_capacity=right_capacity,
        left_batch_size=max(1, max_batch_pairs // right_capacity),
        max_batch_pairs=max_batch_pairs,
    )


def _segment_row_spans(row_indices):
    import cupy as cp

    n = int(row_indices.size)
    if n == 0:
        empty = cp.empty(0, dtype=cp.int32)
        return empty, empty, empty

    d_rows = cp.asarray(row_indices, dtype=cp.int32)
    d_change = cp.empty(n, dtype=cp.bool_)
    d_change[0] = True
    if n > 1:
        d_change[1:] = d_rows[1:] != d_rows[:-1]
    d_starts = cp.flatnonzero(d_change).astype(cp.int32)
    d_ends = cp.concatenate((d_starts[1:], cp.asarray([n], dtype=cp.int32)))
    d_row_ids = d_rows[d_starts]
    return d_row_ids, d_starts, d_ends


def _empty_device_segment_candidates(runtime) -> DeviceSegmentIntersectionCandidates:
    empty_d = runtime.allocate((0,), np.int32)
    return DeviceSegmentIntersectionCandidates(
        left_rows=empty_d,
        left_segments=empty_d,
        left_lookup=runtime.allocate((0,), np.int32),
        right_rows=empty_d,
        right_segments=empty_d,
        right_lookup=runtime.allocate((0,), np.int32),
        count=0,
    )


class _DeviceCandidatePageAccumulator:
    """Buffer lookup pages up to a budget, then consume pages incrementally."""

    def __init__(
        self,
        *,
        left: DeviceSegmentTable,
        right: DeviceSegmentTable,
        contiguous_pair_budget: int,
        page_consumer,
    ) -> None:
        self.left = left
        self.right = right
        self.contiguous_pair_budget = max(int(contiguous_pair_budget), 1)
        self.page_consumer = page_consumer
        self.lookup_pages = []
        self.total_count = 0
        self.paged = False

    def _candidate_page(self, d_left_lookup, d_right_lookup):
        count = int(d_left_lookup.size)
        return DeviceSegmentIntersectionCandidates(
            left_rows=self.left.row_indices[d_left_lookup],
            left_segments=self.left.segment_indices[d_left_lookup],
            left_lookup=d_left_lookup,
            right_rows=self.right.row_indices[d_right_lookup],
            right_segments=self.right.segment_indices[d_right_lookup],
            right_lookup=d_right_lookup,
            count=count,
        )

    def _consume_lookup_page(self, page) -> None:
        d_left_lookup, d_right_lookup = page
        if int(d_left_lookup.size) == 0:
            return
        self.page_consumer(self._candidate_page(d_left_lookup, d_right_lookup))

    def append(self, d_left_lookup, d_right_lookup) -> None:
        count = int(d_left_lookup.size)
        if count == 0:
            return
        self.total_count += count
        if self.page_consumer is not None and self.total_count > self.contiguous_pair_budget:
            if not self.paged:
                self.paged = True
                for page in self.lookup_pages:
                    self._consume_lookup_page(page)
                self.lookup_pages.clear()
            self._consume_lookup_page((d_left_lookup, d_right_lookup))
            return
        self.lookup_pages.append((d_left_lookup, d_right_lookup))

    def finish(self, runtime):
        import cupy as cp

        if self.paged:
            return DeviceSegmentIntersectionCandidatePages(count=self.total_count)
        if not self.lookup_pages:
            return _empty_device_segment_candidates(runtime)
        if len(self.lookup_pages) == 1:
            d_left_lookup, d_right_lookup = self.lookup_pages[0]
        else:
            d_left_lookup = cp.concatenate(tuple(page[0] for page in self.lookup_pages))
            d_right_lookup = cp.concatenate(tuple(page[1] for page in self.lookup_pages))
        return self._candidate_page(d_left_lookup, d_right_lookup)


def _same_row_capacity_scatter_candidates(
    *,
    runtime,
    kernels,
    left: DeviceSegmentTable,
    right: DeviceSegmentTable,
    d_left_rows,
    d_right_row_starts,
    d_right_row_ends,
    max_right_span: int,
    max_batch_pairs: int | None = None,
    page_consumer=None,
    upper_left_rows=None,
    upper_right_rows=None,
) -> DeviceSegmentIntersectionCandidates | DeviceSegmentIntersectionCandidatePages:
    """Generate same-row candidates from a proof-bounded fixed slot carrier."""
    import cupy as cp

    max_right_span = int(max_right_span)
    plan = _candidate_capacity_plan(
        max_right_span,
        max_batch_pairs=max_batch_pairs,
    )
    if plan is None:
        return _empty_device_segment_candidates(runtime)

    scatter_kernel = kernels["scatter_same_row_overlap_candidates_capacity"]
    ptr = runtime.pointer
    use_upper_rows = upper_left_rows is not None and upper_right_rows is not None
    d_upper_left_rows = (
        cp.asarray(upper_left_rows, dtype=cp.int32)
        if use_upper_rows
        else cp.empty(1, dtype=cp.int32)
    )
    d_upper_right_rows = (
        cp.asarray(upper_right_rows, dtype=cp.int32)
        if use_upper_rows
        else cp.empty(1, dtype=cp.int32)
    )
    accumulator = _DeviceCandidatePageAccumulator(
        left=left,
        right=right,
        contiguous_pair_budget=plan.max_batch_pairs,
        page_consumer=page_consumer,
    )
    left_batch_size = int(plan.left_batch_size)

    for left_start in range(0, int(left.count), left_batch_size):
        batch_size = min(left_batch_size, int(left.count) - left_start)
        capacity = int(batch_size) * int(plan.right_capacity)
        if capacity <= 0:
            continue
        d_left_lookup = cp.full(capacity, -1, dtype=cp.int32)
        d_right_lookup = cp.full(capacity, -1, dtype=cp.int32)
        total_threads = batch_size * 32
        scatter_grid, scatter_block = runtime.launch_config(
            scatter_kernel,
            total_threads,
        )
        runtime.launch(
            scatter_kernel,
            grid=scatter_grid,
            block=scatter_block,
            params=(
                (
                    ptr(d_left_rows),
                    ptr(left.x0),
                    ptr(left.y0),
                    ptr(left.x1),
                    ptr(left.y1),
                    ptr(d_right_row_starts),
                    ptr(d_right_row_ends),
                    ptr(right.x0),
                    ptr(right.y0),
                    ptr(right.x1),
                    ptr(right.y1),
                    ptr(d_upper_left_rows),
                    ptr(d_upper_right_rows),
                    int(use_upper_rows),
                    ptr(d_left_lookup),
                    ptr(d_right_lookup),
                    left_start,
                    batch_size,
                    int(plan.right_capacity),
                ),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_I32,
                ),
            ),
        )
        live_compact = compact_indices((d_left_lookup >= 0).astype(cp.uint8))
        if live_compact.count <= 0:
            continue
        live = live_compact.values
        accumulator.append(d_left_lookup[live], d_right_lookup[live])

    return accumulator.finish(runtime)


def _generate_candidates_gpu_same_row_warp(
    left: DeviceSegmentTable,
    right: DeviceSegmentTable,
    *,
    _allow_swap: bool = True,
    span_summary: tuple[int, int, int] | None = None,
    strict_upper_source_rows: tuple[DeviceArray, DeviceArray] | None = None,
    max_batch_pairs: int | None = None,
    page_consumer=None,
) -> DeviceSegmentIntersectionCandidates | DeviceSegmentIntersectionCandidatePages | None:
    import cupy as cp

    if span_summary is None:
        return None

    runtime = get_cuda_runtime()
    left_row_ids, left_row_starts, left_row_ends = _segment_row_spans(left.row_indices)
    right_row_ids, right_row_starts, right_row_ends = _segment_row_spans(right.row_indices)
    if left_row_ids.size == 0 or right_row_ids.size == 0:
        return None

    max_left_span = int(span_summary[0])
    max_right_span = int(span_summary[1])
    max_row_id = int(span_summary[2])
    if max_right_span <= 0:
        return _empty_device_segment_candidates(runtime)
    if max_right_span > _SAME_ROW_WARP_MAX_RIGHT_SEGMENTS_PER_ROW:
        if (
            strict_upper_source_rows is None
            and _allow_swap
            and max_left_span <= _SAME_ROW_WARP_MAX_RIGHT_SEGMENTS_PER_ROW
        ):
            def _consume_swapped_page(page):
                page_consumer(
                    DeviceSegmentIntersectionCandidates(
                        left_rows=page.right_rows,
                        left_segments=page.right_segments,
                        left_lookup=page.right_lookup,
                        right_rows=page.left_rows,
                        right_segments=page.left_segments,
                        right_lookup=page.left_lookup,
                        count=page.count,
                    )
                )

            swapped = _generate_candidates_gpu_same_row_warp(
                right,
                left,
                _allow_swap=False,
                span_summary=(max_right_span, max_left_span, max_row_id),
                max_batch_pairs=max_batch_pairs,
                page_consumer=(_consume_swapped_page if page_consumer is not None else None),
            )
            if swapped is None:
                return None
            if isinstance(swapped, DeviceSegmentIntersectionCandidatePages):
                return swapped
            return DeviceSegmentIntersectionCandidates(
                left_rows=swapped.right_rows,
                left_segments=swapped.right_segments,
                left_lookup=swapped.right_lookup,
                right_rows=swapped.left_rows,
                right_segments=swapped.left_segments,
                right_lookup=swapped.left_lookup,
                count=swapped.count,
            )
        return None

    d_right_row_starts = cp.full(max_row_id + 1, -1, dtype=cp.int32)
    d_right_row_ends = cp.full(max_row_id + 1, -1, dtype=cp.int32)
    d_right_row_starts[right_row_ids] = right_row_starts
    d_right_row_ends[right_row_ids] = right_row_ends

    d_left_rows = cp.asarray(left.row_indices, dtype=cp.int32)
    use_upper_rows = strict_upper_source_rows is not None
    if use_upper_rows:
        d_upper_left_rows = cp.asarray(strict_upper_source_rows[0], dtype=cp.int32)
        d_upper_right_rows = cp.asarray(strict_upper_source_rows[1], dtype=cp.int32)
        if int(d_upper_left_rows.size) != left.count or int(d_upper_right_rows.size) != right.count:
            raise ValueError("strict upper source-row arrays must match segment counts")
    else:
        d_upper_left_rows = cp.empty(1, dtype=cp.int32)
        d_upper_right_rows = cp.empty(1, dtype=cp.int32)
    kernels = _same_row_candidate_kernels()
    return _same_row_capacity_scatter_candidates(
        runtime=runtime,
        kernels=kernels,
        left=left,
        right=right,
        d_left_rows=d_left_rows,
        d_right_row_starts=d_right_row_starts,
        d_right_row_ends=d_right_row_ends,
        max_right_span=max_right_span,
        max_batch_pairs=max_batch_pairs,
        page_consumer=page_consumer,
        upper_left_rows=d_upper_left_rows if use_upper_rows else None,
        upper_right_rows=d_upper_right_rows if use_upper_rows else None,
    )


def _device_counted_sweep_candidates(
    *,
    left: DeviceSegmentTable,
    right: DeviceSegmentTable,
    range_start,
    range_end,
    range_capacity: int,
    sorted_right_idx,
    left_minx,
    left_maxx,
    left_miny,
    left_maxy,
    right_minx,
    right_maxx,
    right_miny,
    right_maxy,
    left_rows_all,
    right_rows_all,
    require_same_row: bool,
    outlier_mask_bool,
    accumulator: _DeviceCandidatePageAccumulator,
    upper_left_rows=None,
    upper_right_rows=None,
):
    """Count once, then scatter sparse neighbor tiers through device rowsets."""
    import cupy as cp

    range_capacity = int(range_capacity)
    if left.count <= 0 or range_capacity <= 0:
        return

    runtime = get_cuda_runtime()
    max_batch_pairs = min(
        _compute_max_batch_pairs(),
        accumulator.contiguous_pair_budget,
    )
    per_left_chunk = min(range_capacity, max_batch_pairs)
    d_range_start = cp.asarray(range_start, dtype=cp.int64)
    d_range_end = cp.asarray(range_end, dtype=cp.int64)
    d_sorted_right_idx = cp.asarray(sorted_right_idx, dtype=cp.int32)
    d_left_rows = (
        cp.asarray(left_rows_all, dtype=cp.int32)
        if require_same_row
        else cp.zeros(1, dtype=cp.int32)
    )
    d_right_rows = (
        cp.asarray(right_rows_all, dtype=cp.int32)
        if require_same_row
        else cp.zeros(1, dtype=cp.int32)
    )
    d_outlier_mask = (
        cp.asarray(outlier_mask_bool, dtype=cp.bool_)
        if outlier_mask_bool is not None
        else cp.zeros(1, dtype=cp.bool_)
    )
    use_upper_rows = upper_left_rows is not None and upper_right_rows is not None
    d_upper_left_rows = (
        cp.asarray(upper_left_rows, dtype=cp.int32)
        if use_upper_rows
        else cp.zeros(1, dtype=cp.int32)
    )
    d_upper_right_rows = (
        cp.asarray(upper_right_rows, dtype=cp.int32)
        if use_upper_rows
        else cp.zeros(1, dtype=cp.int32)
    )
    kernels = _sweep_candidate_kernels()
    count_kernel = kernels["count_sweep_overlap_candidates"]
    scatter_kernel = kernels["scatter_sweep_overlap_candidates"]
    ptr = runtime.pointer
    shared_prefix = (
        ptr(d_range_start),
        ptr(d_range_end),
        ptr(d_sorted_right_idx),
        ptr(left_minx),
        ptr(left_maxx),
        ptr(left_miny),
        ptr(left_maxy),
        ptr(right_minx),
        ptr(right_maxx),
        ptr(right_miny),
        ptr(right_maxy),
        ptr(d_left_rows),
        ptr(d_right_rows),
        int(require_same_row),
        ptr(d_outlier_mask),
        int(outlier_mask_bool is not None),
        ptr(d_upper_left_rows),
        ptr(d_upper_right_rows),
        int(use_upper_rows),
    )
    shared_types = (
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_PTR,
        KERNEL_PARAM_I32,
    )
    left_count = int(left.count)
    for range_offset in range(0, range_capacity, per_left_chunk):
        range_chunk = min(per_left_chunk, range_capacity - range_offset)
        d_counts = cp.zeros(left_count, dtype=cp.int32)
        launch_block = (128, 1, 1)
        with hotpath_stage("segment.candidates.counted_sweep.count", category="filter"):
            runtime.launch(
                count_kernel,
                grid=(left_count, 1, 1),
                block=launch_block,
                params=(
                    (
                        *shared_prefix,
                        0,
                        left_count,
                        range_offset,
                        range_chunk,
                        ptr(d_counts),
                    ),
                    (
                        *shared_types,
                        KERNEL_PARAM_I32,
                        KERNEL_PARAM_I32,
                        KERNEL_PARAM_I32,
                        KERNEL_PARAM_I32,
                        KERNEL_PARAM_PTR,
                    ),
                ),
            )

        d_selected_left = cp.arange(left_count, dtype=cp.int32)
        selected_count = left_count
        tier_start = 0
        preferred_tier_width = min(
            _PREFERRED_INITIAL_SWEEP_TIER_WIDTH,
            max(1, max_batch_pairs // selected_count),
        )
        while selected_count > 0 and tier_start < range_chunk:
            tier_width = min(
                range_chunk - tier_start,
                max(1, max_batch_pairs // selected_count),
                preferred_tier_width,
            )
            output_capacity = selected_count * tier_width
            d_out_left = cp.full(output_capacity, -1, dtype=cp.int32)
            d_out_right = cp.full(output_capacity, -1, dtype=cp.int32)
            with hotpath_stage(
                "segment.candidates.counted_sweep.scatter",
                category="emit",
            ):
                runtime.launch(
                    scatter_kernel,
                    grid=(selected_count, 1, 1),
                    block=(32, 1, 1),
                    params=(
                        (
                            *shared_prefix,
                            ptr(d_selected_left),
                            selected_count,
                            range_offset,
                            range_chunk,
                            tier_start,
                            tier_width,
                            ptr(d_out_left),
                            ptr(d_out_right),
                        ),
                        (
                            *shared_types,
                            KERNEL_PARAM_PTR,
                            KERNEL_PARAM_I32,
                            KERNEL_PARAM_I32,
                            KERNEL_PARAM_I32,
                            KERNEL_PARAM_I32,
                            KERNEL_PARAM_I32,
                            KERNEL_PARAM_PTR,
                            KERNEL_PARAM_PTR,
                        ),
                    ),
                )
            with hotpath_stage(
                "segment.candidates.counted_sweep.physicalize",
                category="compact",
            ):
                d_live = compact_indices(
                    (d_out_right >= 0).astype(cp.uint8, copy=False)
                ).values
            if d_live.shape[0] > 0:
                accumulator.append(d_out_left[d_live], d_out_right[d_live])

            tier_start += tier_width
            if tier_start >= range_chunk:
                break
            with hotpath_stage(
                "segment.candidates.counted_sweep.overflow_rows",
                category="compact",
            ):
                d_selected_left = compact_indices(
                    (d_counts > tier_start).astype(cp.uint8, copy=False)
                ).values
            selected_count = d_selected_left.shape[0]
            preferred_tier_width = min(preferred_tier_width * 2, range_chunk - tier_start)


def _generate_candidates_gpu(
    left: DeviceSegmentTable,
    right: DeviceSegmentTable,
    *,
    require_same_row: bool = False,
    use_same_row_fast_path: bool = True,
    strict_upper_source_rows: tuple[DeviceArray, DeviceArray] | None = None,
    same_row_span_summary: tuple[int, int, int] | None = None,
    candidate_page_budget: int | None = None,
    page_consumer=None,
) -> DeviceSegmentIntersectionCandidates | DeviceSegmentIntersectionCandidatePages:
    """GPU-native O(n log n) candidate generation via sort-sweep."""
    import cupy as cp

    runtime = get_cuda_runtime()

    if left.count == 0 or right.count == 0:
        empty_d = runtime.allocate((0,), np.int32)
        return DeviceSegmentIntersectionCandidates(
            left_rows=empty_d,
            left_segments=empty_d,
            left_lookup=runtime.allocate((0,), np.int32),
            right_rows=empty_d,
            right_segments=empty_d,
            right_lookup=runtime.allocate((0,), np.int32),
            count=0,
        )

    if require_same_row and use_same_row_fast_path and same_row_span_summary is not None:
        with hotpath_stage("segment.candidates.same_row_fast_path", category="filter"):
            same_row_candidates = _generate_candidates_gpu_same_row_warp(
                left,
                right,
                span_summary=same_row_span_summary,
                strict_upper_source_rows=strict_upper_source_rows,
                max_batch_pairs=candidate_page_budget,
                page_consumer=page_consumer,
            )
        if same_row_candidates is not None:
            return same_row_candidates

    # Compute segment bounds on device (CuPy Tier 2 element-wise)
    with hotpath_stage("segment.candidates.compute_bounds", category="setup"):
        left_minx = cp.minimum(left.x0, left.x1)
        left_maxx = cp.maximum(left.x0, left.x1)
        left_miny = cp.minimum(left.y0, left.y1)
        left_maxy = cp.maximum(left.y0, left.y1)

        right_minx = cp.minimum(right.x0, right.x1)
        right_maxx = cp.maximum(right.x0, right.x1)
        right_miny = cp.minimum(right.y0, right.y1)
        right_maxy = cp.maximum(right.y0, right.y1)
        left_rows_all = cp.asarray(left.row_indices, dtype=cp.int32) if require_same_row else None
        right_rows_all = cp.asarray(right.row_indices, dtype=cp.int32) if require_same_row else None
        upper_left_rows = None
        upper_right_rows = None
        if strict_upper_source_rows is not None:
            upper_left_rows = cp.asarray(strict_upper_source_rows[0], dtype=cp.int32)
            upper_right_rows = cp.asarray(strict_upper_source_rows[1], dtype=cp.int32)
            if int(upper_left_rows.size) != left.count or int(upper_right_rows.size) != right.count:
                raise ValueError("strict upper source-row arrays must match segment counts")

    # Assign right segments to their cheaper normalized sweep axis. Both axis
    # plans retain full row capacity and sort inactive lanes to an infinity
    # tail, avoiding a device-count fence between classification and sorting.
    # Every segment is active in exactly one plan, and each plan is refined by
    # the full 2D MBR test, so the candidate relation remains exact.
    right_half_w_x = (right_maxx - right_minx) * 0.5
    right_half_w_y = (right_maxy - right_miny) * 0.5
    d_extent_x = cp.maximum(
        cp.maximum(cp.max(left_maxx), cp.max(right_maxx))
        - cp.minimum(cp.min(left_minx), cp.min(right_minx)),
        cp.asarray(1.0e-12, dtype=cp.float64),
    )
    d_extent_y = cp.maximum(
        cp.maximum(cp.max(left_maxy), cp.max(right_maxy))
        - cp.minimum(cp.min(left_miny), cp.min(right_miny)),
        cp.asarray(1.0e-12, dtype=cp.float64),
    )
    d_use_y_sweep = (right_half_w_y / d_extent_y) < (right_half_w_x / d_extent_x)

    contiguous_pair_budget = min(
        _compute_max_batch_pairs(),
        (
            int(candidate_page_budget)
            if candidate_page_budget is not None
            else _MAX_BATCH_PAIRS_CAP
        ),
    )
    candidate_accumulator = _DeviceCandidatePageAccumulator(
        left=left,
        right=right,
        contiguous_pair_budget=contiguous_pair_budget,
        page_consumer=page_consumer,
    )

    right_indices = cp.arange(right.count, dtype=cp.int32)
    for axis_name, axis_mask, sweep_left_min, sweep_left_max, sweep_right_min, sweep_right_max in (
        (
            "x",
            ~d_use_y_sweep,
            left_minx,
            left_maxx,
            right_minx,
            right_maxx,
        ),
        (
            "y",
            d_use_y_sweep,
            left_miny,
            left_maxy,
            right_miny,
            right_maxy,
        ),
    ):
        with hotpath_stage(
            f"segment.candidates.partition_{axis_name}_axis",
            category="mask",
        ):
            partition_right_idx = right_indices
            partition_right_min = sweep_right_min
            partition_right_max = sweep_right_max
            partition_right_mid = cp.where(
                axis_mask,
                (partition_right_min + partition_right_max) * 0.5,
                cp.inf,
            )
            d_search_hw = cp.max(
                cp.where(
                    axis_mask,
                    (partition_right_max - partition_right_min) * 0.5,
                    0.0,
                )
            )

        with hotpath_stage(
            f"segment.candidates.sort_{axis_name}_midpoints",
            category="sort",
        ):
            sort_result = sort_pairs(
                partition_right_mid,
                partition_right_idx,
                synchronize=False,
            )

        with hotpath_stage(
            f"segment.candidates.search_{axis_name}_ranges",
            category="filter",
        ):
            range_start = lower_bound(
                sort_result.keys,
                sweep_left_min - d_search_hw,
                synchronize=False,
            )
            range_end = upper_bound(
                sort_result.keys,
                sweep_left_max + d_search_hw,
                synchronize=False,
            )

        with hotpath_stage(
            "segment.candidates.device_counted_sweep",
            category="filter",
        ):
            _device_counted_sweep_candidates(
                left=left,
                right=right,
                range_start=range_start,
                range_end=range_end,
                range_capacity=right.count,
                sorted_right_idx=sort_result.values,
                left_minx=left_minx,
                left_maxx=left_maxx,
                left_miny=left_miny,
                left_maxy=left_maxy,
                right_minx=right_minx,
                right_maxx=right_maxx,
                right_miny=right_miny,
                right_maxy=right_maxy,
                left_rows_all=left_rows_all,
                right_rows_all=right_rows_all,
                require_same_row=require_same_row,
                outlier_mask_bool=None,
                accumulator=candidate_accumulator,
                upper_left_rows=upper_left_rows,
                upper_right_rows=upper_right_rows,
            )

    with hotpath_stage("segment.candidates.assemble_output", category="emit"):
        return candidate_accumulator.finish(runtime)


# ---------------------------------------------------------------------------
# Legacy CPU segment extraction (kept for CPU fallback)
# ---------------------------------------------------------------------------


def _valid_global_rows(geometry_array: OwnedGeometryArray, family_name: str) -> np.ndarray:
    tag = FAMILY_TAGS[family_name]
    return np.flatnonzero(geometry_array.validity & (geometry_array.tags == tag)).astype(
        np.int32, copy=False
    )


def _append_segments_for_span(
    *,
    row_index: int,
    part_index: int,
    ring_index: int,
    segment_counter: int,
    x: np.ndarray,
    y: np.ndarray,
    start: int,
    end: int,
    row_indices: list[int],
    part_indices: list[int],
    ring_indices: list[int],
    segment_indices: list[int],
    x0: list[float],
    y0: list[float],
    x1: list[float],
    y1: list[float],
    bounds: list[tuple[float, float, float, float]],
) -> int:
    if end - start < 2:
        return segment_counter

    xs0 = x[start : end - 1]
    ys0 = y[start : end - 1]
    xs1 = x[start + 1 : end]
    ys1 = y[start + 1 : end]
    count = int(xs0.size)
    if count == 0:
        return segment_counter

    row_indices.extend([row_index] * count)
    part_indices.extend([part_index] * count)
    ring_indices.extend([ring_index] * count)
    segment_indices.extend(range(segment_counter, segment_counter + count))
    x0.extend(xs0.tolist())
    y0.extend(ys0.tolist())
    x1.extend(xs1.tolist())
    y1.extend(ys1.tolist())
    bounds.extend(
        zip(
            np.minimum(xs0, xs1).tolist(),
            np.minimum(ys0, ys1).tolist(),
            np.maximum(xs0, xs1).tolist(),
            np.maximum(ys0, ys1).tolist(),
            strict=True,
        )
    )
    return segment_counter + count


def extract_segments(geometry_array: OwnedGeometryArray) -> SegmentTable:
    """Extract segments from geometry array on CPU (legacy path)."""
    geometry_array._ensure_host_state()
    row_indices: list[int] = []
    part_indices: list[int] = []
    ring_indices: list[int] = []
    segment_indices: list[int] = []
    x0: list[float] = []
    y0: list[float] = []
    x1: list[float] = []
    y1: list[float] = []
    bounds: list[tuple[float, float, float, float]] = []

    for family_name, buffer in geometry_array.families.items():
        if family_name not in {"linestring", "polygon", "multilinestring", "multipolygon"}:
            continue

        global_rows = _valid_global_rows(geometry_array, family_name)
        for family_row, row_index in enumerate(
            global_rows.tolist()  # zcopy:ok(CPU-only legacy path over host np.ndarray)
        ):
            if bool(buffer.empty_mask[family_row]):
                continue

            segment_counter = 0
            if family_name == "linestring":
                start = int(buffer.geometry_offsets[family_row])
                end = int(buffer.geometry_offsets[family_row + 1])
                segment_counter = _append_segments_for_span(
                    row_index=row_index,
                    part_index=0,
                    ring_index=0,
                    segment_counter=segment_counter,
                    x=buffer.x,
                    y=buffer.y,
                    start=start,
                    end=end,
                    row_indices=row_indices,
                    part_indices=part_indices,
                    ring_indices=ring_indices,
                    segment_indices=segment_indices,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    bounds=bounds,
                )
                del segment_counter
                continue

            if family_name == "polygon":
                ring_start = int(buffer.geometry_offsets[family_row])
                ring_end = int(buffer.geometry_offsets[family_row + 1])
                for ring_local, ring_index in enumerate(range(ring_start, ring_end)):
                    coord_start = int(buffer.ring_offsets[ring_index])
                    coord_end = int(buffer.ring_offsets[ring_index + 1])
                    segment_counter = _append_segments_for_span(
                        row_index=row_index,
                        part_index=0,
                        ring_index=ring_local,
                        segment_counter=segment_counter,
                        x=buffer.x,
                        y=buffer.y,
                        start=coord_start,
                        end=coord_end,
                        row_indices=row_indices,
                        part_indices=part_indices,
                        ring_indices=ring_indices,
                        segment_indices=segment_indices,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        bounds=bounds,
                    )
                continue

            if family_name == "multilinestring":
                part_start = int(buffer.geometry_offsets[family_row])
                part_end = int(buffer.geometry_offsets[family_row + 1])
                for part_local, part_index in enumerate(range(part_start, part_end)):
                    coord_start = int(buffer.part_offsets[part_index])
                    coord_end = int(buffer.part_offsets[part_index + 1])
                    segment_counter = _append_segments_for_span(
                        row_index=row_index,
                        part_index=part_local,
                        ring_index=-1,
                        segment_counter=segment_counter,
                        x=buffer.x,
                        y=buffer.y,
                        start=coord_start,
                        end=coord_end,
                        row_indices=row_indices,
                        part_indices=part_indices,
                        ring_indices=ring_indices,
                        segment_indices=segment_indices,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        bounds=bounds,
                    )
                continue

            polygon_start = int(buffer.geometry_offsets[family_row])
            polygon_end = int(buffer.geometry_offsets[family_row + 1])
            for polygon_local, polygon_index in enumerate(range(polygon_start, polygon_end)):
                ring_start = int(buffer.part_offsets[polygon_index])
                ring_end = int(buffer.part_offsets[polygon_index + 1])
                for ring_local, ring_index in enumerate(range(ring_start, ring_end)):
                    coord_start = int(buffer.ring_offsets[ring_index])
                    coord_end = int(buffer.ring_offsets[ring_index + 1])
                    segment_counter = _append_segments_for_span(
                        row_index=row_index,
                        part_index=polygon_local,
                        ring_index=ring_local,
                        segment_counter=segment_counter,
                        x=buffer.x,
                        y=buffer.y,
                        start=coord_start,
                        end=coord_end,
                        row_indices=row_indices,
                        part_indices=part_indices,
                        ring_indices=ring_indices,
                        segment_indices=segment_indices,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        bounds=bounds,
                    )

    if not row_indices:
        empty_i32 = np.asarray([], dtype=np.int32)
        empty_f64 = np.asarray([], dtype=np.float64)
        return SegmentTable(
            row_indices=empty_i32,
            part_indices=empty_i32,
            ring_indices=empty_i32,
            segment_indices=empty_i32,
            x0=empty_f64,
            y0=empty_f64,
            x1=empty_f64,
            y1=empty_f64,
            bounds=np.empty((0, 4), dtype=np.float64),
        )

    return SegmentTable(
        row_indices=np.asarray(row_indices, dtype=np.int32),
        part_indices=np.asarray(part_indices, dtype=np.int32),
        ring_indices=np.asarray(ring_indices, dtype=np.int32),
        segment_indices=np.asarray(segment_indices, dtype=np.int32),
        x0=np.asarray(x0, dtype=np.float64),
        y0=np.asarray(y0, dtype=np.float64),
        x1=np.asarray(x1, dtype=np.float64),
        y1=np.asarray(y1, dtype=np.float64),
        bounds=np.asarray(bounds, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Legacy CPU candidate generation (kept for CPU fallback)
# ---------------------------------------------------------------------------


def generate_segment_candidates(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    tile_size: int = SEGMENT_TILE_SIZE,
) -> SegmentIntersectionCandidates:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")

    left_segments = extract_segments(left)
    right_segments = extract_segments(right)
    return _generate_segment_candidates_from_tables(
        left_segments, right_segments, tile_size=tile_size
    )


def _generate_segment_candidates_from_tables(
    left_segments: SegmentTable,
    right_segments: SegmentTable,
    *,
    tile_size: int = SEGMENT_TILE_SIZE,
) -> SegmentIntersectionCandidates:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")

    left_rows_out: list[np.ndarray] = []
    left_segment_out: list[np.ndarray] = []
    left_lookup_out: list[np.ndarray] = []
    right_rows_out: list[np.ndarray] = []
    right_segment_out: list[np.ndarray] = []
    right_lookup_out: list[np.ndarray] = []
    pairs_examined = 0

    for left_start in range(0, left_segments.count, tile_size):
        left_bounds = left_segments.bounds[left_start : left_start + tile_size]
        left_rows = left_segments.row_indices[left_start : left_start + tile_size]
        left_ids = left_segments.segment_indices[left_start : left_start + tile_size]
        for right_start in range(0, right_segments.count, tile_size):
            right_bounds = right_segments.bounds[right_start : right_start + tile_size]
            right_rows = right_segments.row_indices[right_start : right_start + tile_size]
            right_ids = right_segments.segment_indices[right_start : right_start + tile_size]
            pairs_examined += int(left_bounds.shape[0] * right_bounds.shape[0])
            intersects = (
                (left_bounds[:, None, 0] <= right_bounds[None, :, 2])
                & (left_bounds[:, None, 2] >= right_bounds[None, :, 0])
                & (left_bounds[:, None, 1] <= right_bounds[None, :, 3])
                & (left_bounds[:, None, 3] >= right_bounds[None, :, 1])
            )
            left_local, right_local = np.nonzero(intersects)
            if left_local.size == 0:
                continue
            left_rows_out.append(left_rows[left_local].astype(np.int32, copy=False))
            left_segment_out.append(left_ids[left_local].astype(np.int32, copy=False))
            left_lookup_out.append((left_start + left_local).astype(np.int32, copy=False))
            right_rows_out.append(right_rows[right_local].astype(np.int32, copy=False))
            right_segment_out.append(right_ids[right_local].astype(np.int32, copy=False))
            right_lookup_out.append((right_start + right_local).astype(np.int32, copy=False))

    if not left_rows_out:
        empty = np.asarray([], dtype=np.int32)
        return SegmentIntersectionCandidates(
            left_rows=empty,
            left_segments=empty,
            left_lookup=empty,
            right_rows=empty,
            right_segments=empty,
            right_lookup=empty,
            pairs_examined=pairs_examined,
            tile_size=tile_size,
        )
    return SegmentIntersectionCandidates(
        left_rows=np.concatenate(left_rows_out),
        left_segments=np.concatenate(left_segment_out),
        left_lookup=np.concatenate(left_lookup_out),
        right_rows=np.concatenate(right_rows_out),
        right_segments=np.concatenate(right_segment_out),
        right_lookup=np.concatenate(right_lookup_out),
        pairs_examined=pairs_examined,
        tile_size=tile_size,
    )


# ---------------------------------------------------------------------------
# CPU exact arithmetic helpers (kept for CPU fallback)
# ---------------------------------------------------------------------------


def _orient2d_fast(
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    abx = bx - ax
    aby = by - ay
    acx = cx - ax
    acy = cy - ay
    term1 = abx * acy
    term2 = aby * acx
    det = term1 - term2
    errbound = _ORIENTATION_ERRBOUND * (np.abs(term1) + np.abs(term2))
    return det, np.abs(det) <= errbound


def _line_intersection_point(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    dx: float,
    dy: float,
) -> tuple[float, float]:
    denominator = (ax - bx) * (cy - dy) - (ay - by) * (cx - dx)
    if denominator == 0.0:
        return float("nan"), float("nan")
    left_det = ax * by - ay * bx
    right_det = cx * dy - cy * dx
    x = (left_det * (cx - dx) - (ax - bx) * right_det) / denominator
    y = (left_det * (cy - dy) - (ay - by) * right_det) / denominator
    return float(x), float(y)


def _fraction(value: float) -> Fraction:
    return Fraction.from_float(float(value))


def _exact_orientation_sign(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
) -> int:
    det = (_fraction(bx) - _fraction(ax)) * (_fraction(cy) - _fraction(ay)) - (
        _fraction(by) - _fraction(ay)
    ) * (_fraction(cx) - _fraction(ax))
    return int(det > 0) - int(det < 0)


def _point_on_segment_exact(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> bool:
    if _exact_orientation_sign(ax, ay, bx, by, px, py) != 0:
        return False
    pxf = _fraction(px)
    pyf = _fraction(py)
    axf = _fraction(ax)
    ayf = _fraction(ay)
    bxf = _fraction(bx)
    byf = _fraction(by)
    return min(axf, bxf) <= pxf <= max(axf, bxf) and min(ayf, byf) <= pyf <= max(ayf, byf)


def _exact_intersection_point(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    dx: float,
    dy: float,
) -> tuple[float, float]:
    axf = _fraction(ax)
    ayf = _fraction(ay)
    bxf = _fraction(bx)
    byf = _fraction(by)
    cxf = _fraction(cx)
    cyf = _fraction(cy)
    dxf = _fraction(dx)
    dyf = _fraction(dy)
    denominator = (axf - bxf) * (cyf - dyf) - (ayf - byf) * (cxf - dxf)
    if denominator == 0:
        return float("nan"), float("nan")
    left_det = axf * byf - ayf * bxf
    right_det = cxf * dyf - cyf * dxf
    x = (left_det * (cxf - dxf) - (axf - bxf) * right_det) / denominator
    y = (left_det * (cyf - dyf) - (ayf - byf) * right_det) / denominator
    return float(x), float(y)


def _unique_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique: list[tuple[float, float]] = []
    seen: set[tuple[Fraction, Fraction]] = set()
    for x, y in points:
        key = (_fraction(x), _fraction(y))
        if key in seen:
            continue
        seen.add(key)
        unique.append((float(x), float(y)))
    return unique


def _sort_collinear_points(
    points: list[tuple[float, float]],
    *,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> list[tuple[float, float]]:
    use_x = abs(bx - ax) >= abs(by - ay)

    def _key(point: tuple[float, float]) -> tuple[Fraction, Fraction]:
        x, y = point
        if use_x:
            return (_fraction(x), _fraction(y))
        return (_fraction(y), _fraction(x))

    return sorted(points, key=_key)


def _classify_exact_pair(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    dx: float,
    dy: float,
) -> tuple[SegmentIntersectionKind, tuple[float, float], tuple[float, float, float, float]]:
    a_is_point = _fraction(ax) == _fraction(bx) and _fraction(ay) == _fraction(by)
    c_is_point = _fraction(cx) == _fraction(dx) and _fraction(cy) == _fraction(dy)

    if a_is_point and c_is_point:
        if _fraction(ax) == _fraction(cx) and _fraction(ay) == _fraction(cy):
            return SegmentIntersectionKind.TOUCH, (float(ax), float(ay)), (float("nan"),) * 4
        return SegmentIntersectionKind.DISJOINT, (float("nan"), float("nan")), (float("nan"),) * 4

    if a_is_point:
        if _point_on_segment_exact(ax, ay, cx, cy, dx, dy):
            return SegmentIntersectionKind.TOUCH, (float(ax), float(ay)), (float("nan"),) * 4
        return SegmentIntersectionKind.DISJOINT, (float("nan"), float("nan")), (float("nan"),) * 4

    if c_is_point:
        if _point_on_segment_exact(cx, cy, ax, ay, bx, by):
            return SegmentIntersectionKind.TOUCH, (float(cx), float(cy)), (float("nan"),) * 4
        return SegmentIntersectionKind.DISJOINT, (float("nan"), float("nan")), (float("nan"),) * 4

    o1 = _exact_orientation_sign(ax, ay, bx, by, cx, cy)
    o2 = _exact_orientation_sign(ax, ay, bx, by, dx, dy)
    o3 = _exact_orientation_sign(cx, cy, dx, dy, ax, ay)
    o4 = _exact_orientation_sign(cx, cy, dx, dy, bx, by)

    if o1 * o2 < 0 and o3 * o4 < 0:
        point = _exact_intersection_point(ax, ay, bx, by, cx, cy, dx, dy)
        return SegmentIntersectionKind.PROPER, point, (float("nan"),) * 4

    if o1 == 0 and o2 == 0 and o3 == 0 and o4 == 0:
        shared = _unique_points(
            [
                point
                for point in ((ax, ay), (bx, by), (cx, cy), (dx, dy))
                if _point_on_segment_exact(point[0], point[1], ax, ay, bx, by)
                and _point_on_segment_exact(point[0], point[1], cx, cy, dx, dy)
            ]
        )
        if not shared:
            return (
                SegmentIntersectionKind.DISJOINT,
                (float("nan"), float("nan")),
                (float("nan"),) * 4,
            )
        shared = _sort_collinear_points(shared, ax=ax, ay=ay, bx=bx, by=by)
        if len(shared) == 1:
            x, y = shared[0]
            return SegmentIntersectionKind.TOUCH, (x, y), (float("nan"),) * 4
        (sx0, sy0), (sx1, sy1) = shared[0], shared[-1]
        return SegmentIntersectionKind.OVERLAP, (float("nan"), float("nan")), (sx0, sy0, sx1, sy1)

    if o1 == 0 and _point_on_segment_exact(cx, cy, ax, ay, bx, by):
        return SegmentIntersectionKind.TOUCH, (float(cx), float(cy)), (float("nan"),) * 4
    if o2 == 0 and _point_on_segment_exact(dx, dy, ax, ay, bx, by):
        return SegmentIntersectionKind.TOUCH, (float(dx), float(dy)), (float("nan"),) * 4
    if o3 == 0 and _point_on_segment_exact(ax, ay, cx, cy, dx, dy):
        return SegmentIntersectionKind.TOUCH, (float(ax), float(ay)), (float("nan"),) * 4
    if o4 == 0 and _point_on_segment_exact(bx, by, cx, cy, dx, dy):
        return SegmentIntersectionKind.TOUCH, (float(bx), float(by)), (float("nan"),) * 4

    return SegmentIntersectionKind.DISJOINT, (float("nan"), float("nan")), (float("nan"),) * 4


def _classify_exact_rows(
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    rows: np.ndarray,
    kinds: np.ndarray,
    point_x: np.ndarray,
    point_y: np.ndarray,
    overlap_x0: np.ndarray,
    overlap_y0: np.ndarray,
    overlap_x1: np.ndarray,
    overlap_y1: np.ndarray,
) -> None:
    for row in rows.tolist():
        kind, point, overlap = _classify_exact_pair(
            float(ax[row]),
            float(ay[row]),
            float(bx[row]),
            float(by[row]),
            float(cx[row]),
            float(cy[row]),
            float(dx[row]),
            float(dy[row]),
        )
        kinds[row] = int(kind)
        point_x[row], point_y[row] = point
        overlap_x0[row], overlap_y0[row], overlap_x1[row], overlap_y1[row] = overlap


# ---------------------------------------------------------------------------
# Dispatch wiring
# ---------------------------------------------------------------------------


def _select_segment_runtime(
    dispatch_mode: ExecutionMode | str,
    *,
    candidate_count: int,
    current_residency: Residency,
) -> AdaptivePlan:
    return plan_dispatch_selection(
        kernel_name="segment_classify",
        kernel_class=KernelClass.PREDICATE,
        row_count=candidate_count,
        requested_mode=dispatch_mode,
        requested_precision=PrecisionMode.AUTO,
        current_residency=current_residency,
        work_estimate=PhysicalWorkEstimate.for_candidate_pairs(
            row_count=candidate_count,
            candidate_pair_count=candidate_count,
            primary_unit_name="segment-candidate-pair",
        ),
    )


# ---------------------------------------------------------------------------
# GPU variant: full pipeline (extract -> candidates -> classify)
# ---------------------------------------------------------------------------


@register_kernel_variant(
    "segment_intersection",
    "gpu-cuda-python",
    kernel_class=KernelClass.PREDICATE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("linestring", "polygon", "multilinestring", "multipolygon"),
    supports_mixed=True,
    precision_modes=(PrecisionMode.AUTO, PrecisionMode.FP32, PrecisionMode.FP64),
    preferred_residency=Residency.DEVICE,
    tags=("cuda-python",),
)
def _empty_segment_intersection_result(
    *,
    runtime_selection: RuntimeSelection,
    precision_plan: PrecisionPlan,
    robustness_plan: RobustnessPlan,
) -> SegmentIntersectionResult:
    """Construct an empty SegmentIntersectionResult with host arrays."""
    empty_i32 = np.asarray([], dtype=np.int32)
    empty_f64 = np.asarray([], dtype=np.float64)
    return SegmentIntersectionResult(
        candidate_pairs=0,
        runtime_selection=runtime_selection,
        precision_plan=precision_plan,
        robustness_plan=robustness_plan,
        _count=0,
        _left_rows=empty_i32,
        _left_segments=empty_i32,
        _left_lookup=empty_i32,
        _right_rows=empty_i32,
        _right_segments=empty_i32,
        _right_lookup=empty_i32,
        _kinds=empty_i32,
        _point_x=empty_f64,
        _point_y=empty_f64,
        _overlap_x0=empty_f64,
        _overlap_y0=empty_f64,
        _overlap_x1=empty_f64,
        _overlap_y1=empty_f64,
        _ambiguous_rows=empty_i32,
    )


def _classify_segment_intersections_gpu(
    *,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    candidate_pairs: SegmentIntersectionCandidates
    | DeviceSegmentIntersectionCandidates
    | None = None,
    left_segments: SegmentTable | DeviceSegmentTable | None = None,
    right_segments: SegmentTable | DeviceSegmentTable | None = None,
    pairs: SegmentIntersectionCandidates | DeviceSegmentIntersectionCandidates | None = None,
    runtime_selection: RuntimeSelection,
    precision_plan: PrecisionPlan,
    robustness_plan: RobustnessPlan,
    tile_size: int = SEGMENT_TILE_SIZE,
    _cached_right_device_segments: DeviceSegmentTable | None = None,
    _require_same_row: bool = False,
    _use_same_row_fast_path: bool = True,
    _collect_ambiguous_rows: bool = True,
    _strict_upper_source_rows: tuple[DeviceArray, DeviceArray] | None = None,
    _same_row_single_group: bool = False,
    _same_row_span_summary: tuple[int, int, int] | None = None,
    _compact_paged_non_disjoint: bool = False,
    _candidate_page_budget: int | None = None,
    _classified_page_consumer: Callable[[SegmentIntersectionResult], None] | None = None,
) -> SegmentIntersectionResult | PagedSegmentIntersectionResult:
    """Full GPU-native segment intersection pipeline.

    Kernel 1: GPU segment extraction (NVRTC count-scatter)
    Kernel 2: GPU candidate generation (sort-sweep with CCCL radix sort)
    Kernel 3: GPU classification with Shewchuk adaptive refinement

    Parameters
    ----------
    left_segments : DeviceSegmentTable, optional
        Pre-extracted left-side segments. When provided, skips
        ``_extract_segments_gpu(left)`` entirely.
    _cached_right_device_segments : DeviceSegmentTable, optional
        Pre-extracted right-side segments.  When provided, skips
        ``_extract_segments_gpu(right)`` entirely.  Used by
        ``spatial_overlay_owned`` to avoid re-extracting the same
        corridor geometry N times in an N-vs-1 overlay loop (lyy.15).
    """
    import cupy as cp

    runtime = get_cuda_runtime()

    # Determine compute type from precision plan
    compute_type = "float" if precision_plan.compute_precision is PrecisionMode.FP32 else "double"

    # --- Kernel 1: Extract segments on GPU ---
    with hotpath_stage("segment.classify.extract_left_segments", category="setup"):
        try:
            d_left_segs = (
                left_segments
                if isinstance(left_segments, DeviceSegmentTable)
                else _extract_segments_gpu(left, compute_type)
            )
        except Exception as exc:
            raise RuntimeError(
                f"segment left extraction failed: {type(exc).__name__}: {exc}"
            ) from exc
    with hotpath_stage("segment.classify.extract_right_segments", category="setup"):
        try:
            d_right_segs = (
                right_segments
                if isinstance(right_segments, DeviceSegmentTable)
                else (
                    _cached_right_device_segments
                    if _cached_right_device_segments is not None
                    else _extract_segments_gpu(right, compute_type)
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"segment right extraction failed: {type(exc).__name__}: {exc}"
            ) from exc

    if d_left_segs.count == 0 or d_right_segs.count == 0:
        return _empty_segment_intersection_result(
            runtime_selection=runtime_selection,
            precision_plan=precision_plan,
            robustness_plan=robustness_plan,
        )
    same_row_span_summary = _same_row_span_summary
    if (
        same_row_span_summary is None
        and _require_same_row
        and (_same_row_single_group or (left.row_count == 1 and right.row_count == 1))
    ):
        same_row_span_summary = (int(d_left_segs.count), int(d_right_segs.count), 0)
    if same_row_span_summary is None and _require_same_row:
        # A whole-table segment count is a conservative upper bound for every
        # row span.  When it is already within the same-row warp carrier limit,
        # avoid a separate device max-span probe.
        if (
            min(int(d_left_segs.count), int(d_right_segs.count))
            <= _SAME_ROW_WARP_MAX_RIGHT_SEGMENTS_PER_ROW
        ):
            same_row_span_summary = (
                int(d_left_segs.count),
                int(d_right_segs.count),
                max(int(left.row_count), int(right.row_count), 1) - 1,
            )

    # --- Kernel 2: Generate candidates on GPU ---
    with hotpath_stage("segment.classify.generate_candidates", category="filter"):
        precomputed_candidates = candidate_pairs if candidate_pairs is not None else pairs
        classified_pages = []
        classified_count = 0
        classified_page_count = 0

        def _classify_candidate_page(page):
            nonlocal classified_count, classified_page_count
            classified = _classify_segment_intersections_gpu(
                left=left,
                right=right,
                candidate_pairs=page,
                left_segments=d_left_segs,
                right_segments=d_right_segs,
                runtime_selection=runtime_selection,
                precision_plan=precision_plan,
                robustness_plan=robustness_plan,
                tile_size=tile_size,
                _cached_right_device_segments=d_right_segs,
                _require_same_row=_require_same_row,
                _use_same_row_fast_path=_use_same_row_fast_path,
                _collect_ambiguous_rows=_collect_ambiguous_rows,
                _strict_upper_source_rows=_strict_upper_source_rows,
                _same_row_single_group=_same_row_single_group,
                _same_row_span_summary=same_row_span_summary,
                _compact_paged_non_disjoint=False,
            )
            if isinstance(classified, PagedSegmentIntersectionResult):
                raise RuntimeError("candidate page classification nested paging")
            if _compact_paged_non_disjoint:
                classified = _compact_non_disjoint_segment_intersection_page(classified)
            classified_count += classified.count
            classified_page_count += 1
            if _classified_page_consumer is None:
                classified_pages.append(classified)
            else:
                _classified_page_consumer(classified)

        try:
            if isinstance(precomputed_candidates, DeviceSegmentIntersectionCandidates):
                d_candidates = precomputed_candidates
            elif isinstance(precomputed_candidates, SegmentIntersectionCandidates):
                d_candidates = DeviceSegmentIntersectionCandidates(
                    left_rows=runtime.from_host(precomputed_candidates.left_rows),
                    left_segments=runtime.from_host(precomputed_candidates.left_segments),
                    left_lookup=runtime.from_host(precomputed_candidates.left_lookup),
                    right_rows=runtime.from_host(precomputed_candidates.right_rows),
                    right_segments=runtime.from_host(precomputed_candidates.right_segments),
                    right_lookup=runtime.from_host(precomputed_candidates.right_lookup),
                    count=precomputed_candidates.count,
                )
            else:
                d_candidates = _generate_candidates_gpu(
                    d_left_segs,
                    d_right_segs,
                    require_same_row=_require_same_row,
                    use_same_row_fast_path=_use_same_row_fast_path,
                    strict_upper_source_rows=_strict_upper_source_rows,
                    same_row_span_summary=same_row_span_summary,
                    candidate_page_budget=_candidate_page_budget,
                    page_consumer=_classify_candidate_page,
                )
        except Exception as exc:
            raise RuntimeError(
                f"segment candidate generation failed: {type(exc).__name__}: {exc}"
            ) from exc

    if isinstance(d_candidates, DeviceSegmentIntersectionCandidatePages):
        paged_result = PagedSegmentIntersectionResult(
            pages=tuple(classified_pages),
            candidate_pairs=classified_count,
            runtime_selection=runtime_selection,
            precision_plan=precision_plan,
            robustness_plan=robustness_plan,
            compact_non_disjoint=_compact_paged_non_disjoint,
            classified_count=classified_count,
            page_count=classified_page_count,
        )
        record_dispatch_event(
            surface="vibespatial.segment_primitives",
            operation="classify_segment_intersections",
            implementation="paged_candidate_classification_gpu",
            reason=(
                "candidate relation exceeded the contiguous classification "
                "budget and was refined in bounded device pages"
            ),
            detail=(
                f"candidate_pairs={classified_count}; "
                f"pages={classified_page_count}; "
                f"classified_rows={paged_result.count}; "
                f"compact_non_disjoint={_compact_paged_non_disjoint}"
            ),
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
        )
        if not _compact_paged_non_disjoint and _classified_page_consumer is None:
            return concatenate_paged_segment_intersections_device(paged_result)
        return paged_result

    if d_candidates.count == 0:
        return _empty_segment_intersection_result(
            runtime_selection=runtime_selection,
            precision_plan=precision_plan,
            robustness_plan=robustness_plan,
        )

    n_pairs = d_candidates.count

    # --- Kernel 3: Classify segment pairs on GPU ---
    device_kinds = runtime.allocate((n_pairs,), np.int8)
    device_point_x = runtime.allocate((n_pairs,), np.float64)
    device_point_y = runtime.allocate((n_pairs,), np.float64)
    device_overlap_x0 = runtime.allocate((n_pairs,), np.float64)
    device_overlap_y0 = runtime.allocate((n_pairs,), np.float64)
    device_overlap_x1 = runtime.allocate((n_pairs,), np.float64)
    device_overlap_y1 = runtime.allocate((n_pairs,), np.float64)

    kernels = _classify_kernels(compute_type)
    classify_kernel = kernels["classify_segment_pairs_v2"]
    ptr = runtime.pointer

    classify_params = (
        (
            ptr(d_candidates.left_lookup),
            ptr(d_candidates.right_lookup),
            ptr(d_left_segs.x0),
            ptr(d_left_segs.y0),
            ptr(d_left_segs.x1),
            ptr(d_left_segs.y1),
            ptr(d_right_segs.x0),
            ptr(d_right_segs.y0),
            ptr(d_right_segs.x1),
            ptr(d_right_segs.y1),
            ptr(device_kinds),
            ptr(device_point_x),
            ptr(device_point_y),
            ptr(device_overlap_x0),
            ptr(device_overlap_y0),
            ptr(device_overlap_x1),
            ptr(device_overlap_y1),
            n_pairs,
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
        ),
    )
    with hotpath_stage("segment.classify.launch_kernel", category="refine"):
        try:
            grid, block = runtime.launch_config(classify_kernel, n_pairs)
            runtime.launch(classify_kernel, grid=grid, block=block, params=classify_params)
        except Exception as exc:
            raise RuntimeError(
                f"segment classify kernel launch failed: {type(exc).__name__}: {exc}"
            ) from exc

    if _collect_ambiguous_rows:
        # Preserve the CPU-visible ambiguous-row contract on device: rows whose
        # fast orientation filter was numerically ambiguous or degenerate still
        # count as ambiguous even though exact refinement happens fully on GPU.
        d_left_lookup = cp.asarray(d_candidates.left_lookup)
        d_right_lookup = cp.asarray(d_candidates.right_lookup)
        ax = cp.asarray(d_left_segs.x0)[d_left_lookup]
        ay = cp.asarray(d_left_segs.y0)[d_left_lookup]
        bx = cp.asarray(d_left_segs.x1)[d_left_lookup]
        by = cp.asarray(d_left_segs.y1)[d_left_lookup]
        cx = cp.asarray(d_right_segs.x0)[d_right_lookup]
        cy = cp.asarray(d_right_segs.y0)[d_right_lookup]
        dx = cp.asarray(d_right_segs.x1)[d_right_lookup]
        dy = cp.asarray(d_right_segs.y1)[d_right_lookup]

        def _orient2d_fast_device(
            ax,
            ay,
            bx,
            by,
            cx,
            cy,
        ):
            abx = bx - ax
            aby = by - ay
            acx = cx - ax
            acy = cy - ay
            term1 = abx * acy
            term2 = aby * acx
            det = term1 - term2
            errbound = _ORIENTATION_ERRBOUND * (cp.abs(term1) + cp.abs(term2))
            return det, cp.abs(det) <= errbound

        with hotpath_stage("segment.classify.ambiguous_rows", category="refine"):
            try:
                o1, a1 = _orient2d_fast_device(ax, ay, bx, by, cx, cy)
                o2, a2 = _orient2d_fast_device(ax, ay, bx, by, dx, dy)
                o3, a3 = _orient2d_fast_device(cx, cy, dx, dy, ax, ay)
                o4, a4 = _orient2d_fast_device(cx, cy, dx, dy, bx, by)

                zero_left = (ax == bx) & (ay == by)
                zero_right = (cx == dx) & (cy == dy)
                sign1 = cp.sign(o1).astype(cp.int8, copy=False)
                sign2 = cp.sign(o2).astype(cp.int8, copy=False)
                sign3 = cp.sign(o3).astype(cp.int8, copy=False)
                sign4 = cp.sign(o4).astype(cp.int8, copy=False)

                ambiguous_mask = (
                    a1
                    | a2
                    | a3
                    | a4
                    | zero_left
                    | zero_right
                    | (sign1 == 0)
                    | (sign2 == 0)
                    | (sign3 == 0)
                    | (sign4 == 0)
                )
                d_ambiguous_rows = compact_indices(ambiguous_mask.astype(cp.uint8)).values
            except Exception as exc:
                raise RuntimeError(
                    f"segment ambiguous-row detection failed: {type(exc).__name__}: {exc}"
                ) from exc
    else:
        d_ambiguous_rows = runtime.allocate((0,), np.int32)

    # Sync GPU before returning device-primary result.
    with hotpath_stage("segment.classify.synchronize", category="emit"):
        runtime.synchronize()

    # Device-primary: host arrays are lazily materialized on first access.
    return SegmentIntersectionResult(
        candidate_pairs=n_pairs,
        runtime_selection=runtime_selection,
        precision_plan=precision_plan,
        robustness_plan=robustness_plan,
        device_state=SegmentIntersectionDeviceState(
            left_rows=d_candidates.left_rows,
            left_segments=d_candidates.left_segments,
            left_lookup=d_candidates.left_lookup,
            right_rows=d_candidates.right_rows,
            right_segments=d_candidates.right_segments,
            right_lookup=d_candidates.right_lookup,
            kinds=device_kinds,
            point_x=device_point_x,
            point_y=device_point_y,
            overlap_x0=device_overlap_x0,
            overlap_y0=device_overlap_y0,
            overlap_x1=device_overlap_x1,
            overlap_y1=device_overlap_y1,
            ambiguous_rows=d_ambiguous_rows,
        ),
        _count=n_pairs,
    )


# ---------------------------------------------------------------------------
# CPU variant (Shapely-based fallback)
# ---------------------------------------------------------------------------


@register_kernel_variant(
    "segment_intersection",
    "cpu",
    kernel_class=KernelClass.PREDICATE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=("linestring", "polygon", "multilinestring", "multipolygon"),
    supports_mixed=True,
    precision_modes=(PrecisionMode.AUTO, PrecisionMode.FP64),
    tags=("shapely",),
)
def _classify_segment_intersections_cpu(
    *,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    candidate_pairs: SegmentIntersectionCandidates | None = None,
    left_segments: SegmentTable | None = None,
    right_segments: SegmentTable | None = None,
    pairs: SegmentIntersectionCandidates | None = None,
    runtime_selection: RuntimeSelection,
    precision_plan: PrecisionPlan,
    robustness_plan: RobustnessPlan,
    tile_size: int = SEGMENT_TILE_SIZE,
) -> SegmentIntersectionResult:
    """CPU fallback using numpy vectorized orientation + exact Fraction arithmetic."""
    left_segs = left_segments if left_segments is not None else extract_segments(left)
    right_segs = right_segments if right_segments is not None else extract_segments(right)
    cands = (
        candidate_pairs or pairs
        if (candidate_pairs is not None or pairs is not None)
        else _generate_segment_candidates_from_tables(left_segs, right_segs, tile_size=tile_size)
    )
    return _classify_segment_intersections_from_tables(
        left_segments=left_segs,
        right_segments=right_segs,
        pairs=cands,
        runtime_selection=runtime_selection,
        precision_plan=precision_plan,
        robustness_plan=robustness_plan,
    )


def _classify_segment_intersections_from_tables(
    *,
    left_segments: SegmentTable,
    right_segments: SegmentTable,
    pairs: SegmentIntersectionCandidates,
    runtime_selection: RuntimeSelection,
    precision_plan: PrecisionPlan,
    robustness_plan: RobustnessPlan,
) -> SegmentIntersectionResult:
    if pairs.count == 0:
        return _empty_segment_intersection_result(
            runtime_selection=runtime_selection,
            precision_plan=precision_plan,
            robustness_plan=robustness_plan,
        )

    left_lookup = pairs.left_lookup
    right_lookup = pairs.right_lookup

    ax = left_segments.x0[left_lookup]
    ay = left_segments.y0[left_lookup]
    bx = left_segments.x1[left_lookup]
    by = left_segments.y1[left_lookup]
    cx = right_segments.x0[right_lookup]
    cy = right_segments.y0[right_lookup]
    dx = right_segments.x1[right_lookup]
    dy = right_segments.y1[right_lookup]

    o1, a1 = _orient2d_fast(ax, ay, bx, by, cx, cy)
    o2, a2 = _orient2d_fast(ax, ay, bx, by, dx, dy)
    o3, a3 = _orient2d_fast(cx, cy, dx, dy, ax, ay)
    o4, a4 = _orient2d_fast(cx, cy, dx, dy, bx, by)

    zero_left = (ax == bx) & (ay == by)
    zero_right = (cx == dx) & (cy == dy)
    sign1 = np.sign(o1).astype(np.int8, copy=False)
    sign2 = np.sign(o2).astype(np.int8, copy=False)
    sign3 = np.sign(o3).astype(np.int8, copy=False)
    sign4 = np.sign(o4).astype(np.int8, copy=False)

    ambiguous_mask = (
        a1
        | a2
        | a3
        | a4
        | zero_left
        | zero_right
        | (sign1 == 0)
        | (sign2 == 0)
        | (sign3 == 0)
        | (sign4 == 0)
    )
    proper_mask = (~ambiguous_mask) & (sign1 * sign2 < 0) & (sign3 * sign4 < 0)

    count = int(pairs.count)
    kinds = np.full(count, int(SegmentIntersectionKind.DISJOINT), dtype=np.int8)
    point_x = np.full(count, np.nan, dtype=np.float64)
    point_y = np.full(count, np.nan, dtype=np.float64)
    overlap_x0 = np.full(count, np.nan, dtype=np.float64)
    overlap_y0 = np.full(count, np.nan, dtype=np.float64)
    overlap_x1 = np.full(count, np.nan, dtype=np.float64)
    overlap_y1 = np.full(count, np.nan, dtype=np.float64)

    kinds[proper_mask] = int(SegmentIntersectionKind.PROPER)
    proper_rows = np.flatnonzero(proper_mask)
    for row in proper_rows.tolist():
        point_x[row], point_y[row] = _line_intersection_point(
            float(ax[row]),
            float(ay[row]),
            float(bx[row]),
            float(by[row]),
            float(cx[row]),
            float(cy[row]),
            float(dx[row]),
            float(dy[row]),
        )

    ambiguous_rows = np.flatnonzero(ambiguous_mask).astype(np.int32, copy=False)
    if ambiguous_rows.size:
        _classify_exact_rows(
            ax,
            ay,
            bx,
            by,
            cx,
            cy,
            dx,
            dy,
            ambiguous_rows,
            kinds,
            point_x,
            point_y,
            overlap_x0,
            overlap_y0,
            overlap_x1,
            overlap_y1,
        )

    return SegmentIntersectionResult(
        candidate_pairs=int(pairs.count),
        runtime_selection=runtime_selection,
        precision_plan=precision_plan,
        robustness_plan=robustness_plan,
        _left_rows=pairs.left_rows.copy(),
        _left_segments=pairs.left_segments.copy(),
        _left_lookup=pairs.left_lookup.copy(),
        _right_rows=pairs.right_rows.copy(),
        _right_segments=pairs.right_segments.copy(),
        _right_lookup=pairs.right_lookup.copy(),
        _kinds=kinds,
        _point_x=point_x,
        _point_y=point_y,
        _overlap_x0=overlap_x0,
        _overlap_y0=overlap_y0,
        _overlap_x1=overlap_x1,
        _overlap_y1=overlap_y1,
        _ambiguous_rows=ambiguous_rows,
    )


# ---------------------------------------------------------------------------
# Public API entry point with dispatch
# ---------------------------------------------------------------------------


def classify_segment_intersections(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    candidate_pairs: SegmentIntersectionCandidates | None = None,
    tile_size: int = SEGMENT_TILE_SIZE,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
    _cached_left_device_segments: DeviceSegmentTable | None = None,
    _cached_right_device_segments: DeviceSegmentTable | None = None,
    _require_same_row: bool = False,
    _use_same_row_fast_path: bool = True,
    _collect_ambiguous_rows: bool = True,
    _strict_upper_source_rows: tuple[DeviceArray, DeviceArray] | None = None,
    _same_row_single_group: bool = False,
    _same_row_span_summary: tuple[int, int, int] | None = None,
    _compact_paged_non_disjoint: bool = False,
    _candidate_page_budget: int | None = None,
    _classified_page_consumer: Callable[[SegmentIntersectionResult], None] | None = None,
) -> SegmentIntersectionResult | PagedSegmentIntersectionResult:
    """Classify all segment-segment intersections between two geometry arrays.

    Parameters
    ----------
    left, right : OwnedGeometryArray
        Input geometry arrays (linestring, polygon, or multi-variants).
    candidate_pairs : SegmentIntersectionCandidates, optional
        Pre-computed candidate pairs. If None, candidates are generated
        internally (GPU-native O(n log n) when GPU mode, tiled CPU otherwise).
    tile_size : int
        Tile size for CPU candidate generation (ignored in GPU mode).
    dispatch_mode : ExecutionMode
        Force GPU, CPU, or AUTO dispatch.
    precision : PrecisionMode
        Force fp32, fp64, or AUTO precision.
    _cached_left_device_segments : DeviceSegmentTable, optional
        Pre-extracted left-side device segments for reuse.
    _cached_right_device_segments : DeviceSegmentTable, optional
        Pre-extracted right-side device segments for reuse (lyy.15).

    Returns
    -------
    SegmentIntersectionResult
        Classification of all candidate segment pairs.
    """
    stage = "estimate_candidate_count"
    try:
        # Estimate candidate count for dispatch decision
        # Use a rough heuristic: total coords across both arrays
        total_coords = sum(
            buf.x.size
            for buf in left.families.values()
            if buf.family
            in {
                GeometryFamily.LINESTRING,
                GeometryFamily.POLYGON,
                GeometryFamily.MULTILINESTRING,
                GeometryFamily.MULTIPOLYGON,
            }
        ) + sum(
            buf.x.size
            for buf in right.families.values()
            if buf.family
            in {
                GeometryFamily.LINESTRING,
                GeometryFamily.POLYGON,
                GeometryFamily.MULTILINESTRING,
                GeometryFamily.MULTIPOLYGON,
            }
        )
        estimated_candidates = max(total_coords, 1)

        stage = "select_runtime"
        runtime_selection = _select_segment_runtime(
            dispatch_mode,
            candidate_count=estimated_candidates,
            current_residency=combined_residency(left, right),
        )
        if precision is PrecisionMode.AUTO:
            precision_plan = runtime_selection.precision_plan
        else:
            runtime_selection = plan_dispatch_selection(
                kernel_name="segment_classify",
                kernel_class=KernelClass.PREDICATE,
                row_count=estimated_candidates,
                requested_mode=dispatch_mode,
                requested_precision=precision,
                current_residency=combined_residency(left, right),
                work_estimate=PhysicalWorkEstimate.for_candidate_pairs(
                    row_count=estimated_candidates,
                    candidate_pair_count=estimated_candidates,
                    primary_unit_name="segment-candidate-pair",
                ),
            )
            precision_plan = runtime_selection.precision_plan

        stage = "select_robustness"
        robustness_plan = select_robustness_plan(
            kernel_class=KernelClass.PREDICATE,
            precision_plan=precision_plan,
        )

        if runtime_selection.selected is ExecutionMode.GPU:
            stage = "gpu_dispatch"
            return _classify_segment_intersections_gpu(
                left=left,
                right=right,
                candidate_pairs=candidate_pairs,
                left_segments=_cached_left_device_segments,
                runtime_selection=runtime_selection,
                precision_plan=precision_plan,
                robustness_plan=robustness_plan,
                tile_size=tile_size,
                _cached_right_device_segments=_cached_right_device_segments,
                _require_same_row=_require_same_row,
                _use_same_row_fast_path=_use_same_row_fast_path,
                _collect_ambiguous_rows=_collect_ambiguous_rows,
                _strict_upper_source_rows=_strict_upper_source_rows,
                _same_row_single_group=_same_row_single_group,
                _same_row_span_summary=_same_row_span_summary,
                _compact_paged_non_disjoint=_compact_paged_non_disjoint,
                _candidate_page_budget=_candidate_page_budget,
                _classified_page_consumer=_classified_page_consumer,
            )

        stage = "cpu_dispatch"
        return _classify_segment_intersections_cpu(
            left=left,
            right=right,
            candidate_pairs=candidate_pairs,
            runtime_selection=runtime_selection,
            precision_plan=precision_plan,
            robustness_plan=robustness_plan,
            tile_size=tile_size,
        )
    except Exception as exc:
        raise RuntimeError(
            f"classify_segment_intersections failed at {stage}: {type(exc).__name__}: {exc}"
        ) from exc


def summarize_exact_local_events(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    candidate_pairs: SegmentIntersectionCandidates | None = None,
    tile_size: int = SEGMENT_TILE_SIZE,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
    _cached_right_device_segments: DeviceSegmentTable | None = None,
    _require_same_row: bool = False,
) -> SegmentLocalEventSummary:
    """Summarize per-row exact local-event counts for overlay-style workloads.

    This is a reusable bridge between segment intersection classification and
    later topology stages.  It combines segment endpoints with exact
    point-intersection outputs to produce stable row-local exact-event counts
    and interval upper bounds without teaching that logic separately in each
    overlay consumer.
    """
    intersections = classify_segment_intersections(
        left,
        right,
        candidate_pairs=candidate_pairs,
        tile_size=tile_size,
        dispatch_mode=dispatch_mode,
        precision=precision,
        _cached_right_device_segments=_cached_right_device_segments,
        _require_same_row=_require_same_row,
    )
    left_segments = extract_segments(left)
    right_segments = extract_segments(right)
    row_count = max(left.row_count, right.row_count)

    xy_events_by_row: list[set[tuple[str, str]]] = [set() for _ in range(row_count)]
    for row_idx in range(row_count):
        left_mask = left_segments.row_indices == row_idx
        right_mask = right_segments.row_indices == row_idx
        xy_events_by_row[row_idx].update(
            (float(x).hex(), float(y).hex())
            for x, y in zip(left_segments.x0[left_mask], left_segments.y0[left_mask])
        )
        xy_events_by_row[row_idx].update(
            (float(x).hex(), float(y).hex())
            for x, y in zip(left_segments.x1[left_mask], left_segments.y1[left_mask])
        )
        xy_events_by_row[row_idx].update(
            (float(x).hex(), float(y).hex())
            for x, y in zip(right_segments.x0[right_mask], right_segments.y0[right_mask])
        )
        xy_events_by_row[row_idx].update(
            (float(x).hex(), float(y).hex())
            for x, y in zip(right_segments.x1[right_mask], right_segments.y1[right_mask])
        )

    point_mask = np.isfinite(intersections.point_x) & np.isfinite(intersections.point_y)
    point_rows = intersections.left_rows[point_mask].astype(np.int64, copy=False)
    point_x = intersections.point_x[point_mask]
    point_y = intersections.point_y[point_mask]
    for row_idx, x, y in zip(point_rows, point_x, point_y):
        xy_events_by_row[int(row_idx)].add((float(x).hex(), float(y).hex()))

    exact_event_counts = np.asarray([len(events) for events in xy_events_by_row], dtype=np.int64)
    return SegmentLocalEventSummary(
        runtime_selection=intersections.runtime_selection,
        precision_plan=intersections.precision_plan,
        robustness_plan=intersections.robustness_plan,
        candidate_pairs=int(intersections.candidate_pairs),
        point_intersection_count=int(point_mask.sum()),
        parallel_or_colinear_candidate_count=int(
            np.count_nonzero(
                ~np.isfinite(intersections.point_x)
                & ~np.isfinite(intersections.overlap_x0)
                & (intersections.kinds != int(SegmentIntersectionKind.DISJOINT))
            )
        ),
        row_point_intersection_counts=np.bincount(point_rows, minlength=row_count).astype(
            np.int64, copy=False
        ),
        exact_event_counts=exact_event_counts,
        exact_interval_upper_bounds=np.maximum(exact_event_counts - 1, 0),
    )


def benchmark_segment_intersections(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    tile_size: int = SEGMENT_TILE_SIZE,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
) -> SegmentIntersectionBenchmark:
    started = perf_counter()
    result = classify_segment_intersections(
        left, right, tile_size=tile_size, dispatch_mode=dispatch_mode
    )
    elapsed = perf_counter() - started
    return SegmentIntersectionBenchmark(
        rows_left=left.row_count,
        rows_right=right.row_count,
        candidate_pairs=result.candidate_pairs,
        disjoint_pairs=int(np.count_nonzero(result.kinds == int(SegmentIntersectionKind.DISJOINT))),
        proper_pairs=int(np.count_nonzero(result.kinds == int(SegmentIntersectionKind.PROPER))),
        touch_pairs=int(np.count_nonzero(result.kinds == int(SegmentIntersectionKind.TOUCH))),
        overlap_pairs=int(np.count_nonzero(result.kinds == int(SegmentIntersectionKind.OVERLAP))),
        ambiguous_pairs=int(result.ambiguous_rows.size),
        elapsed_seconds=elapsed,
    )
