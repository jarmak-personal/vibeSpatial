from __future__ import annotations

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

request_warmup([
    "select_i32",
    "select_i64",
    "exclusive_scan_i32",
    "exclusive_scan_i64",
    "lower_bound_f64",
    "lower_bound_i64",
    "upper_bound_f64",
    "upper_bound_i64",
])
from vibespatial.cuda._runtime import (  # noqa: E402
    KERNEL_PARAM_I32,
    KERNEL_PARAM_I64,
    KERNEL_PARAM_PTR,
    DeviceArray,
    compile_kernel_group,
    count_scatter_total,
    count_scatter_totals,
    get_cuda_runtime,
    maybe_trim_pool_memory,
)
from vibespatial.geometry.buffers import GeometryFamily  # noqa: E402
from vibespatial.geometry.owned import FAMILY_TAGS, OwnedGeometryArray  # noqa: E402
from vibespatial.runtime import ExecutionMode, RuntimeSelection  # noqa: E402
from vibespatial.runtime.adaptive import AdaptivePlan, plan_dispatch_selection  # noqa: E402
from vibespatial.runtime.config import SEGMENT_TILE_SIZE  # noqa: E402
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
    _CANDIDATE_SCATTER_KERNEL_NAMES,
    _CANDIDATE_SCATTER_KERNEL_SOURCE,
    _SAME_ROW_CANDIDATE_KERNEL_NAMES,
    _SAME_ROW_CANDIDATE_KERNEL_SOURCE,
    _SEGMENT_CLASSIFY_KERNEL_NAMES,
    _SEGMENT_EXTRACT_KERNEL_NAMES,
    CLASSIFY_SOURCE_FP32,
    CLASSIFY_SOURCE_FP64,
    EXTRACT_SOURCE_FP32,
    EXTRACT_SOURCE_FP64,
    format_classify_source,
    format_extract_source,
)

_FLOAT_EPSILON = np.finfo(np.float64).eps
_ORIENTATION_ERRBOUND = (3.0 + 16.0 * _FLOAT_EPSILON) * _FLOAT_EPSILON

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
            ), dtype=np.int32,
        )
        self._left_segments = np.asarray(
            runtime.copy_device_to_host(
                ds.left_segments,
                reason="segment intersections left-segment host export",
            ), dtype=np.int32,
        )
        self._left_lookup = np.asarray(
            runtime.copy_device_to_host(
                ds.left_lookup,
                reason="segment intersections left-lookup host export",
            ), dtype=np.int32,
        )
        self._right_rows = np.asarray(
            runtime.copy_device_to_host(
                ds.right_rows,
                reason="segment intersections right-row host export",
            ), dtype=np.int32,
        )
        self._right_segments = np.asarray(
            runtime.copy_device_to_host(
                ds.right_segments,
                reason="segment intersections right-segment host export",
            ), dtype=np.int32,
        )
        self._right_lookup = np.asarray(
            runtime.copy_device_to_host(
                ds.right_lookup,
                reason="segment intersections right-lookup host export",
            ), dtype=np.int32,
        )
        self._kinds = np.asarray(
            runtime.copy_device_to_host(
                ds.kinds,
                reason="segment intersections kind-code host export",
            ), dtype=np.int8,
        )
        self._point_x = np.asarray(
            runtime.copy_device_to_host(
                ds.point_x,
                reason="segment intersections point-x host export",
            ), dtype=np.float64,
        )
        self._point_y = np.asarray(
            runtime.copy_device_to_host(
                ds.point_y,
                reason="segment intersections point-y host export",
            ), dtype=np.float64,
        )
        self._overlap_x0 = np.asarray(
            runtime.copy_device_to_host(
                ds.overlap_x0,
                reason="segment intersections overlap-x0 host export",
            ), dtype=np.float64,
        )
        self._overlap_y0 = np.asarray(
            runtime.copy_device_to_host(
                ds.overlap_y0,
                reason="segment intersections overlap-y0 host export",
            ), dtype=np.float64,
        )
        self._overlap_x1 = np.asarray(
            runtime.copy_device_to_host(
                ds.overlap_x1,
                reason="segment intersections overlap-x1 host export",
            ), dtype=np.float64,
        )
        self._overlap_y1 = np.asarray(
            runtime.copy_device_to_host(
                ds.overlap_y1,
                reason="segment intersections overlap-y1 host export",
            ), dtype=np.float64,
        )
        self._ambiguous_rows = np.asarray(
            runtime.copy_device_to_host(
                ds.ambiguous_rows,
                reason="segment intersections ambiguous-row host export",
            ), dtype=np.int32,
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


# ---------------------------------------------------------------------------
# NVRTC compilation and warmup
# ---------------------------------------------------------------------------

from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("segment-extract-fp64", EXTRACT_SOURCE_FP64, _SEGMENT_EXTRACT_KERNEL_NAMES),
    ("segment-extract-fp32", EXTRACT_SOURCE_FP32, _SEGMENT_EXTRACT_KERNEL_NAMES),
    ("segment-classify-fp64", CLASSIFY_SOURCE_FP64, _SEGMENT_CLASSIFY_KERNEL_NAMES),
    ("segment-classify-fp32", CLASSIFY_SOURCE_FP32, _SEGMENT_CLASSIFY_KERNEL_NAMES),
    ("segment-candidate-scatter", _CANDIDATE_SCATTER_KERNEL_SOURCE, _CANDIDATE_SCATTER_KERNEL_NAMES),
    ("segment-same-row-candidates", _SAME_ROW_CANDIDATE_KERNEL_SOURCE, _SAME_ROW_CANDIDATE_KERNEL_NAMES),
])


def _extract_kernels(compute_type: str = "double"):
    source = format_extract_source(compute_type)
    name = f"segment-extract-{compute_type.replace('double', 'fp64').replace('float', 'fp32')}"
    return compile_kernel_group(name, source, _SEGMENT_EXTRACT_KERNEL_NAMES)


def _classify_kernels(compute_type: str = "double"):
    source = format_classify_source(compute_type)
    name = f"segment-classify-{compute_type.replace('double', 'fp64').replace('float', 'fp32')}"
    return compile_kernel_group(name, source, _SEGMENT_CLASSIFY_KERNEL_NAMES)


def _candidate_scatter_kernels():
    return compile_kernel_group(
        "segment-candidate-scatter",
        _CANDIDATE_SCATTER_KERNEL_SOURCE,
        _CANDIDATE_SCATTER_KERNEL_NAMES,
    )


def _same_row_candidate_kernels():
    return compile_kernel_group(
        "segment-same-row-candidates",
        _SAME_ROW_CANDIDATE_KERNEL_SOURCE,
        _SAME_ROW_CANDIDATE_KERNEL_NAMES,
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
    if (
        int(part_starts.min(initial=0)) < 0
        or int(part_ends.max(initial=0)) >= int(part_offsets.size)
    ):
        return None
    if int(part_offsets.min(initial=0)) < 0 or int(part_offsets.max(initial=0)) >= int(ring_offsets.size):
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
    """Return segment totals proved by device buffer shape metadata.

    Nested geometry buffers do not need to inspect offset values to know total
    segment cardinality: non-empty leaves are stored as coordinate spans and
    empty rows contribute no leaves.  Therefore total segments are total
    coordinates minus total line/ring leaves.  Both array sizes are host-known
    allocation metadata, not device scalar reads.
    """
    coord_count = int(buffer.x.size)

    if family is GeometryFamily.POLYGON and buffer.ring_offsets is not None:
        ring_count = int(buffer.ring_offsets.size) - 1
        if ring_count >= 0 and coord_count >= ring_count:
            return coord_count - ring_count
        return None

    if family is GeometryFamily.MULTILINESTRING and buffer.part_offsets is not None:
        part_count = int(buffer.part_offsets.size) - 1
        if part_count >= 0 and coord_count >= part_count:
            return coord_count - part_count
        return None

    if family is GeometryFamily.MULTIPOLYGON and buffer.ring_offsets is not None:
        ring_count = int(buffer.ring_offsets.size) - 1
        if ring_count >= 0 and coord_count >= ring_count:
            return coord_count - ring_count
        return None

    return None


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
    d_state = geometry_array._ensure_device_state()

    # The count_segments / scatter_segments kernels declare family_codes as
    # ``const int*`` (int32), but d_state.tags is int8.  Passing an int8
    # pointer to a kernel that reads 4-byte ints causes every thread to read
    # a garbage family code, producing zero segment counts and (when the
    # underlying memory layout changes) an illegal-address fault.
    d_family_codes = d_state.tags.astype(cp.int32) if d_state.tags.dtype != cp.int32 else d_state.tags
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
    pending_families: list[_PendingSegmentFamily] = []

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
        if dense_polygon_width:
            d_fam_empty = cp.zeros(n_fam, dtype=cp.uint8)
        else:
            d_fam_empty = d_buf.empty_mask[d_fam_row_off].astype(cp.uint8, copy=True)

        # Part and ring offsets (use zeros if not available)
        d_geom_off = d_buf.geometry_offsets
        d_part_off = d_buf.part_offsets if d_buf.part_offsets is not None else d_buf.geometry_offsets
        d_ring_off = d_buf.ring_offsets if d_buf.ring_offsets is not None else d_buf.geometry_offsets

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
        else:
            # Step 1: Count segments
            d_seg_counts = runtime.allocate((n_fam,), np.int32, zero=True)
            count_kernel = kernels["count_segments"]

            count_params = (
                (ptr(d_fam_valid), ptr(d_family_codes), ptr(d_family_row_offsets),
                 ptr(d_geom_off), ptr(d_part_off), ptr(d_ring_off),
                 ptr(d_fam_empty), ptr(d_seg_counts), n_fam),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
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
            if fam_total is None:
                fam_total = _device_structural_segment_total_for_family(
                    family_enum,
                    d_buf,
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
            )
        )

    counted_families = [
        pending
        for pending in pending_families
        if pending.total_segments is None and pending.segment_counts is not None
    ]
    if counted_families:
        totals = count_scatter_totals(
            runtime,
            [
                (pending.segment_counts, pending.segment_offsets)
                for pending in counted_families
                if pending.segment_counts is not None
            ],
            reason="segment extraction total-segments allocation fence",
        )
        for pending, fam_total in zip(counted_families, totals, strict=True):
            pending.total_segments = int(fam_total)

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
        fam_total = int(pending.total_segments or 0)
        if fam_total == 0:
            runtime.free(d_fam_valid)
            runtime.free(d_fam_empty)
            runtime.free(d_seg_counts)
            runtime.free(d_seg_offsets)
            continue

        # Step 3: Allocate and scatter
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
            (ptr(d_fam_valid), ptr(d_family_codes), ptr(d_family_row_offsets),
             ptr(d_geom_off), ptr(d_part_off), ptr(d_ring_off),
             ptr(d_fam_empty), ptr(d_buf.x), ptr(d_buf.y),
             ptr(d_seg_offsets),
             ptr(d_out_row), ptr(d_out_seg), ptr(d_out_part), ptr(d_out_ring),
             ptr(d_out_x0), ptr(d_out_y0), ptr(d_out_x1), ptr(d_out_y1),
             n_fam),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_I32),
        )
        grid, block = runtime.launch_config(scatter_kernel, n_fam)
        runtime.launch(scatter_kernel, grid=grid, block=block, params=scatter_params)

        all_row_idx.append(d_out_row)
        all_seg_idx.append(d_out_seg)
        all_part_idx.append(d_out_part)
        all_ring_idx.append(d_out_ring)
        all_x0.append(d_out_x0)
        all_y0.append(d_out_y0)
        all_x1.append(d_out_x1)
        all_y1.append(d_out_y1)
        total_segments += fam_total

        # Free temp buffers (output arrays are kept)
        runtime.free(d_fam_valid)
        runtime.free(d_fam_empty)
        runtime.free(d_seg_counts)
        runtime.free(d_seg_offsets)

    if total_segments == 0 or not all_row_idx:
        return DeviceSegmentTable(
            row_indices=runtime.allocate((0,), np.int32),
            segment_indices=runtime.allocate((0,), np.int32),
            x0=runtime.allocate((0,), np.float64),
            y0=runtime.allocate((0,), np.float64),
            x1=runtime.allocate((0,), np.float64),
            y1=runtime.allocate((0,), np.float64),
            count=0,
            part_indices=runtime.allocate((0,), np.int32),
            ring_indices=runtime.allocate((0,), np.int32),
        )

    # Concatenate per-family results on device (CuPy Tier 2)
    if len(all_row_idx) == 1:
        return DeviceSegmentTable(
            row_indices=all_row_idx[0],
            segment_indices=all_seg_idx[0],
            x0=all_x0[0],
            y0=all_y0[0],
            x1=all_x1[0],
            y1=all_y1[0],
            count=total_segments,
            part_indices=all_part_idx[0],
            ring_indices=all_ring_idx[0],
        )

    return DeviceSegmentTable(
        row_indices=cp.concatenate(all_row_idx),
        segment_indices=cp.concatenate(all_seg_idx),
        x0=cp.concatenate(all_x0),
        y0=cp.concatenate(all_y0),
        x1=cp.concatenate(all_x1),
        y1=cp.concatenate(all_y1),
        count=total_segments,
        part_indices=cp.concatenate(all_part_idx),
        ring_indices=cp.concatenate(all_ring_idx),
    )


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
_BOUNDED_CAPACITY_SCATTER_MAX_PAIRS = 1 * 1024 * 1024
_SAME_ROW_WARP_MAX_RIGHT_SEGMENTS_PER_ROW = 2048


def _compute_max_batch_pairs() -> int:
    """Return the maximum number of raw candidate pairs per batch.

    Uses actual RMM/CuPy pool free blocks when available, falling back
    to CUDA mem_info.  Applies a hard cap of 8M pairs to prevent OOM
    from pool fragmentation and CuPy advanced-indexing temporaries.
    """
    import cupy as cp

    from vibespatial.cuda._runtime import get_cuda_runtime

    # Try to get actual pool-level free memory (more accurate than CUDA mem_info
    # because RMM reserves large blocks from CUDA up front).
    try:
        runtime = get_cuda_runtime()
        stats = runtime.memory_pool_stats()
        if "free_bytes" in stats:
            free_bytes = stats["free_bytes"]
        else:
            free_bytes, _ = cp.cuda.Device().mem_info
    except Exception:
        return _MAX_BATCH_PAIRS_CAP

    # Use 25% of available pool memory, capped at _MAX_BATCH_PAIRS_CAP.
    usable_bytes = free_bytes // 4
    max_pairs = usable_bytes // _BYTES_PER_RAW_PAIR

    return min(max(max_pairs, _MIN_BATCH_PAIRS), _MAX_BATCH_PAIRS_CAP)


def _bounded_candidate_capacity(*dimensions: int) -> int | None:
    """Return a host-known candidate capacity when bounded scatter is admissible."""
    capacity = 1
    for dimension in dimensions:
        dimension = int(dimension)
        if dimension <= 0:
            return 0
        capacity *= dimension
        if capacity > _BOUNDED_CAPACITY_SCATTER_MAX_PAIRS:
            return None
    if capacity > _compute_max_batch_pairs():
        return None
    return capacity


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


def _generate_candidates_gpu_same_row_warp(
    left: DeviceSegmentTable,
    right: DeviceSegmentTable,
    *,
    _allow_swap: bool = True,
) -> DeviceSegmentIntersectionCandidates | None:
    import cupy as cp

    runtime = get_cuda_runtime()

    left_row_ids, left_row_starts, left_row_ends = _segment_row_spans(left.row_indices)
    right_row_ids, right_row_starts, right_row_ends = _segment_row_spans(right.row_indices)
    if left_row_ids.size == 0 or right_row_ids.size == 0:
        return None

    d_span_summary = cp.empty(4, dtype=cp.int32)
    d_span_summary[0] = cp.max(left_row_ends - left_row_starts)
    d_span_summary[1] = cp.max(right_row_ends - right_row_starts)
    d_span_summary[2] = cp.max(left_row_ids)
    d_span_summary[3] = cp.max(right_row_ids)
    span_summary = _segment_device_to_host(
        d_span_summary,
        reason="segment same-row span summary scalar fence",
    )
    max_left_span = int(span_summary[0])
    max_right_span = int(span_summary[1])
    if max_right_span > _SAME_ROW_WARP_MAX_RIGHT_SEGMENTS_PER_ROW:
        if _allow_swap and max_left_span <= _SAME_ROW_WARP_MAX_RIGHT_SEGMENTS_PER_ROW:
            swapped = _generate_candidates_gpu_same_row_warp(
                right,
                left,
                _allow_swap=False,
            )
            if swapped is None:
                return None
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

    max_row_id = int(max(span_summary[2], span_summary[3]))
    d_right_row_starts = cp.full(max_row_id + 1, -1, dtype=cp.int32)
    d_right_row_ends = cp.full(max_row_id + 1, -1, dtype=cp.int32)
    d_right_row_starts[right_row_ids] = right_row_starts
    d_right_row_ends[right_row_ids] = right_row_ends

    d_left_rows = cp.asarray(left.row_indices, dtype=cp.int32)
    kernels = _same_row_candidate_kernels()
    ptr = runtime.pointer
    n_left = left.count

    d_counts = cp.empty(n_left, dtype=cp.int32)
    total_threads = n_left * 32
    count_kernel = kernels["count_same_row_overlap_candidates"]
    count_grid, count_block = runtime.launch_config(count_kernel, total_threads)
    runtime.launch(
        count_kernel,
        grid=count_grid,
        block=count_block,
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
                ptr(d_counts),
                n_left,
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
                KERNEL_PARAM_I32,
            ),
        ),
    )

    d_counts64 = d_counts.astype(cp.int64, copy=False)
    d_offsets = exclusive_sum(d_counts64, synchronize=False)
    bounded_capacity = _bounded_candidate_capacity(n_left, max_right_span)
    if bounded_capacity is None:
        total_candidates = count_scatter_total(
            runtime,
            d_counts64,
            d_offsets,
            reason="segment same-row candidate total allocation fence",
        )
    else:
        total_candidates = int(bounded_capacity)
    if total_candidates == 0:
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

    if bounded_capacity is None:
        d_left_lookup = cp.empty(total_candidates, dtype=cp.int32)
        d_right_lookup = cp.empty(total_candidates, dtype=cp.int32)
    else:
        d_left_lookup = cp.full(total_candidates, -1, dtype=cp.int32)
        d_right_lookup = cp.full(total_candidates, -1, dtype=cp.int32)

    scatter_kernel = kernels["scatter_same_row_overlap_candidates"]
    scatter_grid, scatter_block = runtime.launch_config(scatter_kernel, total_threads)
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
                ptr(d_offsets),
                ptr(d_left_lookup),
                ptr(d_right_lookup),
                n_left,
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
                KERNEL_PARAM_I32,
            ),
        ),
    )
    if bounded_capacity is not None:
        live_compact = compact_indices((d_left_lookup >= 0).astype(cp.uint8))
        if live_compact.count == 0:
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
        live = live_compact.values
        d_left_lookup = d_left_lookup[live]
        d_right_lookup = d_right_lookup[live]

    out_left_rows = left.row_indices[d_left_lookup]
    out_left_segs = left.segment_indices[d_left_lookup]
    out_right_rows = right.row_indices[d_right_lookup]
    out_right_segs = right.segment_indices[d_right_lookup]

    return DeviceSegmentIntersectionCandidates(
        left_rows=out_left_rows,
        left_segments=out_left_segs,
        left_lookup=d_left_lookup,
        right_rows=out_right_rows,
        right_segments=out_right_segs,
        right_lookup=d_right_lookup,
        count=int(d_left_lookup.size),
    )


def _main_sweep_scatter_and_filter(
    *,
    runtime,
    left: DeviceSegmentTable,
    right: DeviceSegmentTable,
    d_cand_offsets,
    range_start,
    range_end,
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
    total_raw_candidates: int,
):
    """Scatter raw candidates and MBR-filter, batching to fit in VRAM.

    When ``total_raw_candidates`` fits in a single batch this is identical
    to the original unbatched code path — no performance regression for
    small inputs.  For large candidate counts that would OOM, left segments
    are partitioned into batches whose raw pair count fits within 50% of
    free VRAM.

    Returns (main_final_left, main_final_right) as CuPy int32 arrays.
    """
    import cupy as cp

    max_batch_pairs = _compute_max_batch_pairs()

    # Fast path: everything fits in one batch — no overhead.
    if total_raw_candidates <= max_batch_pairs:
        return _scatter_and_filter_single(
            runtime=runtime,
            left=left,
            right=right,
            d_cand_offsets=d_cand_offsets,
            range_start=range_start,
            range_end=range_end,
            sorted_right_idx=sorted_right_idx,
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
            outlier_mask_bool=outlier_mask_bool,
            left_start=0,
            left_end=left.count,
            offset_base=0,
            batch_raw_count=total_raw_candidates,
        )

    # --- Batched path: partition left segments into VRAM-bounded chunks ---
    # Batch boundaries are found entirely on device using lower_bound on
    # the monotonically non-decreasing d_cand_offsets, then only the small
    # boundary array is transferred to host for loop control flow.
    n_left = left.count

    # Build search targets: multiples of max_batch_pairs up to total.
    n_batches_est = (total_raw_candidates + max_batch_pairs - 1) // max_batch_pairs
    d_targets = cp.arange(1, n_batches_est, dtype=cp.int64) * max_batch_pairs
    d_inner = lower_bound(d_cand_offsets, d_targets, synchronize=False)

    # Full boundary array: [0, *inner_boundaries, n_left]
    d_boundaries = cp.concatenate([
        cp.zeros(1, dtype=cp.intp),
        d_inner.astype(cp.intp),
        cp.full(1, n_left, dtype=cp.intp),
    ])
    # Remove duplicate boundaries (collapsed empty batches).
    d_boundaries = cp.unique(d_boundaries)

    # Gather offset values at each boundary for batch-size computation.
    # Clamp indices to valid range for the offsets array (n_left entries).
    d_boundary_clamped = cp.minimum(d_boundaries, n_left - 1)
    d_boundary_offsets = d_cand_offsets[d_boundary_clamped]

    # Single bulk transfer of the small boundaries + offset-values arrays.
    runtime.synchronize()
    h_boundaries = runtime.copy_device_to_host(
        d_boundaries,
        reason="segment candidate batch-boundary host export",
    )
    h_boundary_offsets = runtime.copy_device_to_host(
        d_boundary_offsets,
        reason="segment candidate batch-boundary-offset host export",
    )

    # Process each batch, keeping filtered results on device.  Filtered
    # output is a small fraction of raw candidates (typically 5-20% after
    # MBR filtering) so accumulating on device is safe — the raw pair
    # temporaries that drive OOM are freed inside _scatter_and_filter_single.
    result_left_parts: list = []
    result_right_parts: list = []

    n_bounds = len(h_boundaries)
    for b in range(n_bounds - 1):
        b_lo = int(h_boundaries[b])
        b_hi = int(h_boundaries[b + 1])
        if b_lo >= b_hi:
            continue

        b_offset_base = int(h_boundary_offsets[b])
        if b_hi < n_left:
            b_raw_count = int(h_boundary_offsets[b + 1]) - b_offset_base
        else:
            b_raw_count = total_raw_candidates - b_offset_base

        if b_raw_count == 0:
            continue

        # Only trim the pool when explicitly enabled. Forced GC between
        # normal batches dominates warmed overlay/clip workloads.
        maybe_trim_pool_memory(runtime)

        b_left, b_right = _scatter_and_filter_single(
            runtime=runtime,
            left=left,
            right=right,
            d_cand_offsets=d_cand_offsets,
            range_start=range_start,
            range_end=range_end,
            sorted_right_idx=sorted_right_idx,
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
            outlier_mask_bool=outlier_mask_bool,
            left_start=b_lo,
            left_end=b_hi,
            offset_base=b_offset_base,
            batch_raw_count=b_raw_count,
        )
        if b_left.size > 0:
            result_left_parts.append(b_left)
            result_right_parts.append(b_right)

    if not result_left_parts:
        return cp.empty(0, dtype=cp.int32), cp.empty(0, dtype=cp.int32)
    if len(result_left_parts) == 1:
        return result_left_parts[0], result_right_parts[0]
    return (
        cp.concatenate(result_left_parts),
        cp.concatenate(result_right_parts),
    )


def _main_sweep_capacity_scatter_and_filter(
    *,
    runtime,
    left: DeviceSegmentTable,
    right: DeviceSegmentTable,
    range_start,
    range_end,
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
):
    """Scatter sweep candidates in host-sized fixed-capacity batches.

    Physical shape: segment candidate-refine.  The batch capacity is
    `left_batch_size * right.count`, so allocation sizing is proved from host
    table cardinalities rather than a device scalar sum.  Each row writes only
    its binary-search range, then sentinel slots are compacted on device.
    """
    import cupy as cp

    max_batch_pairs = _compute_max_batch_pairs()
    right_capacity = int(right.count)
    if right_capacity <= 0 or right_capacity > max_batch_pairs:
        return None

    left_batch_size = max(1, max_batch_pairs // right_capacity)
    result_left_parts: list = []
    result_right_parts: list = []
    for left_start in range(0, int(left.count), int(left_batch_size)):
        left_end = min(left_start + left_batch_size, int(left.count))
        batch_left, batch_right = _scatter_and_filter_capacity_single(
            runtime=runtime,
            left=left,
            right=right,
            range_start=range_start,
            range_end=range_end,
            sorted_right_idx=sorted_right_idx,
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
            outlier_mask_bool=outlier_mask_bool,
            left_start=left_start,
            left_end=left_end,
            right_capacity=right_capacity,
        )
        if batch_left.size > 0:
            result_left_parts.append(batch_left)
            result_right_parts.append(batch_right)

    if not result_left_parts:
        return cp.empty(0, dtype=cp.int32), cp.empty(0, dtype=cp.int32)
    if len(result_left_parts) == 1:
        return result_left_parts[0], result_right_parts[0]
    return (
        cp.concatenate(result_left_parts),
        cp.concatenate(result_right_parts),
    )


def _filter_scattered_candidate_pairs(
    *,
    left: DeviceSegmentTable,
    right: DeviceSegmentTable,
    d_left_pair,
    d_right_pair,
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
):
    import cupy as cp

    d_lminx = left_minx[d_left_pair]
    d_lmaxx = left_maxx[d_left_pair]
    d_lminy = left_miny[d_left_pair]
    d_lmaxy = left_maxy[d_left_pair]
    d_rminx = right_minx[d_right_pair]
    d_rmaxx = right_maxx[d_right_pair]
    d_rminy = right_miny[d_right_pair]
    d_rmaxy = right_maxy[d_right_pair]

    main_overlap = (
        (d_lminx <= d_rmaxx) & (d_lmaxx >= d_rminx) &
        (d_lminy <= d_rmaxy) & (d_lmaxy >= d_rminy)
    )
    if require_same_row:
        try:
            main_overlap &= left_rows_all[d_left_pair] == right_rows_all[d_right_pair]
        except Exception as exc:
            left_pair_max = (
                _segment_int_scalar(
                    cp.max(d_left_pair),
                    reason="segment same-row debug left-pair-max scalar fence",
                )
                if int(d_left_pair.size)
                else -1
            )
            right_pair_max = (
                _segment_int_scalar(
                    cp.max(d_right_pair),
                    reason="segment same-row debug right-pair-max scalar fence",
                )
                if int(d_right_pair.size)
                else -1
            )
            raise RuntimeError(
                "same-row main candidate filter failed: "
                f"left_rows={int(left_rows_all.size)}, "
                f"right_rows={int(right_rows_all.size)}, "
                f"left_pair_count={int(d_left_pair.size)}, "
                f"right_pair_count={int(d_right_pair.size)}, "
                f"left_pair_max={left_pair_max}, "
                f"right_pair_max={right_pair_max}"
            ) from exc
    del d_lminx, d_lmaxx, d_lminy, d_lmaxy
    del d_rminx, d_rmaxx, d_rminy, d_rmaxy

    if outlier_mask_bool is not None:
        main_overlap &= ~outlier_mask_bool[d_right_pair]
    main_compact = compact_indices(main_overlap.astype(cp.uint8))
    if main_compact.count > 0:
        main_keep = main_compact.values
        return d_left_pair[main_keep], d_right_pair[main_keep]
    return cp.empty(0, dtype=cp.int32), cp.empty(0, dtype=cp.int32)


def _scatter_and_filter_capacity_single(
    *,
    runtime,
    left: DeviceSegmentTable,
    right: DeviceSegmentTable,
    range_start,
    range_end,
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
    left_start: int,
    left_end: int,
    right_capacity: int,
):
    import cupy as cp

    batch_size = int(left_end) - int(left_start)
    if batch_size <= 0 or right_capacity <= 0:
        return cp.empty(0, dtype=cp.int32), cp.empty(0, dtype=cp.int32)

    capacity = batch_size * int(right_capacity)
    d_left_pair = cp.full(capacity, -1, dtype=cp.int32)
    d_right_pair = cp.full(capacity, -1, dtype=cp.int32)
    scatter_fn = _candidate_scatter_kernels()["scatter_candidate_pairs_capacity_batch"]
    ptr = runtime.pointer
    d_range_start_i32 = cp.asarray(range_start, dtype=cp.int32)
    d_range_end_i32 = cp.asarray(range_end, dtype=cp.int32)
    grid, block = runtime.launch_config(scatter_fn, batch_size)
    runtime.launch(
        scatter_fn,
        grid=grid,
        block=block,
        params=(
            (
                ptr(d_range_start_i32),
                ptr(d_range_end_i32),
                ptr(sorted_right_idx),
                ptr(d_left_pair),
                ptr(d_right_pair),
                int(left_start),
                int(batch_size),
                int(right_capacity),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
            ),
        ),
    )
    live_compact = compact_indices((d_left_pair >= 0).astype(cp.uint8))
    if live_compact.count == 0:
        return cp.empty(0, dtype=cp.int32), cp.empty(0, dtype=cp.int32)
    live = live_compact.values
    return _filter_scattered_candidate_pairs(
        left=left,
        right=right,
        d_left_pair=d_left_pair[live],
        d_right_pair=d_right_pair[live],
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
        outlier_mask_bool=outlier_mask_bool,
    )


def _scatter_and_filter_single(
    *,
    runtime,
    left: DeviceSegmentTable,
    right: DeviceSegmentTable,
    d_cand_offsets,
    range_start,
    range_end,
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
    left_start: int,
    left_end: int,
    offset_base: int,
    batch_raw_count: int,
    capacity_upper_bound: bool = False,
):
    """Scatter candidate pairs for left segments [left_start, left_end)
    and apply MBR overlap filter.  Returns (filtered_left, filtered_right).

    When left_start==0, left_end==n_left, and offset_base==0 this is
    identical to the original unbatched code path.
    """
    import cupy as cp

    batch_size = left_end - left_start
    if batch_raw_count <= 0:
        return cp.empty(0, dtype=cp.int32), cp.empty(0, dtype=cp.int32)

    if capacity_upper_bound:
        d_left_pair = cp.full(batch_raw_count, -1, dtype=cp.int32)
        d_right_pair = cp.full(batch_raw_count, -1, dtype=cp.int32)
    else:
        d_left_pair = runtime.allocate((batch_raw_count,), np.int32)
        d_right_pair = runtime.allocate((batch_raw_count,), np.int32)

    scatter_kernels = _candidate_scatter_kernels()
    ptr = runtime.pointer

    # range_start/range_end are CuPy uint arrays; cast to int32 for kernel
    d_range_start_i32 = cp.asarray(range_start, dtype=cp.int32)
    d_range_end_i32 = cp.asarray(range_end, dtype=cp.int32)

    if left_start == 0 and left_end == left.count:
        # Unbatched path: use the original kernel (no offset arithmetic).
        scatter_fn = scatter_kernels["scatter_candidate_pairs"]
        grid, block = runtime.launch_config(scatter_fn, left.count)
        runtime.launch(
            scatter_fn,
            grid=grid, block=block,
            params=(
                (ptr(d_cand_offsets), ptr(d_range_start_i32), ptr(d_range_end_i32),
                 ptr(sorted_right_idx), ptr(d_left_pair), ptr(d_right_pair),
                 left.count),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_I32),
            ),
        )
    else:
        # Batched path: use the batch-aware kernel variant.
        scatter_fn = scatter_kernels["scatter_candidate_pairs_batch"]
        grid, block = runtime.launch_config(scatter_fn, batch_size)
        runtime.launch(
            scatter_fn,
            grid=grid, block=block,
            params=(
                (ptr(d_cand_offsets), ptr(d_range_start_i32), ptr(d_range_end_i32),
                 ptr(sorted_right_idx), ptr(d_left_pair), ptr(d_right_pair),
                 left_start, batch_size, offset_base),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_I32, KERNEL_PARAM_I32, KERNEL_PARAM_I64),
            ),
        )

    if capacity_upper_bound:
        live_compact = compact_indices((d_left_pair >= 0).astype(cp.uint8))
        if live_compact.count == 0:
            return cp.empty(0, dtype=cp.int32), cp.empty(0, dtype=cp.int32)
        live = live_compact.values
        d_left_pair = d_left_pair[live]
        d_right_pair = d_right_pair[live]

    return _filter_scattered_candidate_pairs(
        left=left,
        right=right,
        d_left_pair=d_left_pair,
        d_right_pair=d_right_pair,
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
        outlier_mask_bool=outlier_mask_bool,
    )


def _generate_candidates_gpu(
    left: DeviceSegmentTable,
    right: DeviceSegmentTable,
    *,
    require_same_row: bool = False,
    use_same_row_fast_path: bool = True,
) -> DeviceSegmentIntersectionCandidates:
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

    if require_same_row and use_same_row_fast_path:
        with hotpath_stage("segment.candidates.same_row_fast_path", category="filter"):
            same_row_candidates = _generate_candidates_gpu_same_row_warp(left, right)
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
        left_rows_all = (
            cp.asarray(left.row_indices, dtype=cp.int32)
            if require_same_row
            else None
        )
        right_rows_all = (
            cp.asarray(right.row_indices, dtype=cp.int32)
            if require_same_row
            else None
        )

    # Sort right segments by x-midpoint for sweep-based candidate search.
    # Algorithm: sort right segments by x-midpoint, then for each left segment
    # binary-search for the range of rights whose midpoint falls within
    # [left_minx - max_right_halfwidth, left_maxx + max_right_halfwidth].
    # Then filter surviving candidates by full MBR y-overlap.
    # Complexity: O(n log n + k) where k is the number of candidate pairs.
    right_xmid = (right_minx + right_maxx) * 0.5
    # NOTE (P5/LOW): cp.arange allocates a 4MB array at 1M scale just to
    # provide [0..n-1] indices.  A counting_iterator would be zero-alloc,
    # but sort_pairs calls _validate_vector("values", values) which
    # requires a 1D DeviceArray (ndim check).  Both the CuPy argsort
    # fallback and the CCCL radix_sort path index into the values array,
    # so accepting an iterator would require invasive changes to
    # sort_pairs + its strategy dispatch.  Not worth the churn for a
    # one-shot 4MB allocation.
    with hotpath_stage("segment.candidates.sort_right_midpoints", category="sort"):
        right_indices = cp.arange(right.count, dtype=cp.int32)
        sort_result = sort_pairs(right_xmid, right_indices, synchronize=False)
        sorted_right_xmid = sort_result.keys
        sorted_right_idx = sort_result.values

    # -----------------------------------------------------------------------
    # P95 half-width strategy for search window sizing.
    #
    # Problem: using cp.max(right_half_w) makes the binary-search window
    # as wide as the single largest right segment. If even one segment
    # spans the full coordinate space, every left window covers ALL rights
    # -> O(n^2) candidates -> OOM.
    #
    # Solution: use the 95th percentile of right half-widths for the main
    # sweep (tight windows for 95% of segments), then handle the <=5%
    # outlier right segments in a separate brute-force MBR pass.
    # -----------------------------------------------------------------------
    right_half_w = (right_maxx - right_minx) * 0.5

    with hotpath_stage("segment.candidates.search_window", category="filter"):
        if right.count < 20:
            # Too few segments for P95 to matter -- use global max.
            d_search_hw = cp.max(right_half_w)
            has_outliers = False
            outlier_mask_bool = None
        else:
            # Use partition (O(n)) instead of full sort (O(n log n)) to get P95.
            p95_idx = int(right.count * 95) // 100  # floor index
            partitioned_hw = cp.partition(right_half_w, p95_idx)
            d_p95_hw = partitioned_hw[p95_idx]       # CuPy scalar on device
            d_search_hw = d_p95_hw

            # Keep outlier admission on device.  A no-outlier workload compacts
            # to zero later without a host scalar branch here.
            has_outliers = True
            outlier_mask_bool = right_half_w > d_search_hw

    # --- Main sweep: binary search using P95 (or max) half-width ---
    with hotpath_stage("segment.candidates.binary_search", category="filter"):
        search_lo = left_minx - d_search_hw
        search_hi = left_maxx + d_search_hw

        # Binary search in sorted_right_xmid (same-stream ordering guarantees
        # sort_pairs completes before lower_bound/upper_bound read its output)
        range_start = lower_bound(sorted_right_xmid, search_lo, synchronize=False)
        range_end = upper_bound(sorted_right_xmid, search_hi, synchronize=False)

        # Two-pass: count candidates per left, prefix-sum, scatter.
        # Use int64 for counts/offsets: at 100k+ segments, total candidates can
        # exceed int32 max (~2.1B), causing prefix-sum overflow and negative batch sizes.
    with hotpath_stage("segment.candidates.prefix_candidate_counts", category="sort"):
        d_cand_counts = cp.asarray(range_end, dtype=cp.int64) - cp.asarray(range_start, dtype=cp.int64)
        d_cand_counts = cp.maximum(d_cand_counts, 0)  # clamp negatives

        d_cand_offsets = exclusive_sum(d_cand_counts, synchronize=False)
        bounded_capacity = _bounded_candidate_capacity(left.count, right.count)
        capacity_batch_used = False
        if bounded_capacity is None:
            capacity_batch_result = _main_sweep_capacity_scatter_and_filter(
                runtime=runtime,
                left=left,
                right=right,
                range_start=range_start,
                range_end=range_end,
                sorted_right_idx=sorted_right_idx,
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
                outlier_mask_bool=outlier_mask_bool,
            )
            if capacity_batch_result is None:
                total_raw_candidates = count_scatter_total(
                    runtime,
                    d_cand_counts,
                    d_cand_offsets,
                    reason="segment candidate total allocation fence",
                )
            else:
                main_final_left, main_final_right = capacity_batch_result
                total_raw_candidates = int(main_final_left.size)
                capacity_batch_used = True
        else:
            total_raw_candidates = int(bounded_capacity)

    if bounded_capacity is None and not capacity_batch_used:
        # Guard: if total raw candidates exceed what the GPU can classify
        # (each surviving pair requires ~49 bytes of output arrays), raise early
        # instead of crashing mid-way through batched scatter.
        try:
            free_bytes, _ = cp.cuda.Device().mem_info
        except Exception:
            free_bytes = 8 * 1024**3  # conservative 8 GB
        # 49 bytes per pair for classification output + 8 bytes for candidate indices
        max_feasible_pairs = free_bytes // 57
        if total_raw_candidates > max_feasible_pairs:
            raise MemoryError(
                f"Segment intersection candidate count ({total_raw_candidates:,}) exceeds "
                f"GPU memory capacity ({free_bytes / 1e9:.1f} GB free, max ~{max_feasible_pairs:,} "
                f"feasible pairs). Reduce input scale or use CPU dispatch."
            )

    if capacity_batch_used:
        pass
    elif total_raw_candidates > 0:
        with hotpath_stage("segment.candidates.main_sweep_filter", category="filter"):
            if bounded_capacity is None:
                main_final_left, main_final_right = _main_sweep_scatter_and_filter(
                    runtime=runtime,
                    left=left,
                    right=right,
                    d_cand_offsets=d_cand_offsets,
                    range_start=range_start,
                    range_end=range_end,
                    sorted_right_idx=sorted_right_idx,
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
                    outlier_mask_bool=outlier_mask_bool,
                    total_raw_candidates=total_raw_candidates,
                )
            else:
                main_final_left, main_final_right = _scatter_and_filter_single(
                    runtime=runtime,
                    left=left,
                    right=right,
                    d_cand_offsets=d_cand_offsets,
                    range_start=range_start,
                    range_end=range_end,
                    sorted_right_idx=sorted_right_idx,
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
                    outlier_mask_bool=outlier_mask_bool,
                    left_start=0,
                    left_end=left.count,
                    offset_base=0,
                    batch_raw_count=total_raw_candidates,
                    capacity_upper_bound=True,
                )
    else:
        main_final_left = cp.empty(0, dtype=cp.int32)
        main_final_right = cp.empty(0, dtype=cp.int32)

    # --- Outlier pass: brute-force MBR test for right segs with hw > P95 ---
    if has_outliers:
        # Identify outlier right segment indices (boolean mask -> compact).
        # These are right segments whose half-width exceeds P95, meaning
        # the main sweep's narrower window may have missed them.
        with hotpath_stage("segment.candidates.outlier_pass", category="filter"):
            outlier_mask_u8 = outlier_mask_bool.astype(cp.uint8)
            outlier_compact = compact_indices(outlier_mask_u8)

            if outlier_compact.count > 0:
                outlier_right_idx = outlier_compact.values  # indices into right arrays
                n_outliers = outlier_compact.count
                n_left = left.count

                # Batched brute-force: process outlier rights in chunks to avoid
                # materializing O(n_left * n_outliers) pairs which would OOM at
                # scale (e.g. 1M left × 50K outliers = 50B elements = 200+ GB).
                # Batch size adapts to available VRAM (same policy as main sweep).
                _MAX_EXPANDED_PAIRS = _compute_max_batch_pairs()
                _OUTLIER_BATCH = max(1, _MAX_EXPANDED_PAIRS // max(n_left, 1))
                batch_left_parts = []
                batch_right_parts = []
                d_left_arange = cp.arange(n_left, dtype=cp.int32)

                for batch_start in range(0, n_outliers, _OUTLIER_BATCH):
                    batch_end = min(batch_start + _OUTLIER_BATCH, n_outliers)
                    batch_right = outlier_right_idx[batch_start:batch_end]
                    batch_size = batch_end - batch_start

                    # Expand: n_left × batch_size pairs (bounded to ~1M per batch)
                    ol_left_idx = cp.repeat(d_left_arange, batch_size)
                    ol_right_idx = cp.tile(batch_right, n_left)

                    # Vectorized MBR overlap test
                    ol_overlap = (
                        (left_minx[ol_left_idx] <= right_maxx[ol_right_idx]) &
                        (left_maxx[ol_left_idx] >= right_minx[ol_right_idx]) &
                        (left_miny[ol_left_idx] <= right_maxy[ol_right_idx]) &
                        (left_maxy[ol_left_idx] >= right_miny[ol_right_idx])
                    )
                    if require_same_row:
                        try:
                            ol_overlap &= (
                                left_rows_all[ol_left_idx] == right_rows_all[ol_right_idx]
                            )
                        except Exception as exc:
                            raise RuntimeError(
                                "same-row outlier candidate filter failed: "
                                f"left_rows={left_rows_all.shape[0]}, "
                                f"right_rows={right_rows_all.shape[0]}, "
                                f"left_pair_count={ol_left_idx.size}, "
                                f"right_pair_count={ol_right_idx.size}"
                            ) from exc
                    ol_compact = compact_indices(ol_overlap.astype(cp.uint8))

                    if ol_compact.count > 0:
                        ol_keep = ol_compact.values
                        batch_left_parts.append(ol_left_idx[ol_keep])
                        batch_right_parts.append(ol_right_idx[ol_keep])

                if batch_left_parts:
                    outlier_final_left = cp.concatenate(batch_left_parts) if len(batch_left_parts) > 1 else batch_left_parts[0]
                    outlier_final_right = cp.concatenate(batch_right_parts) if len(batch_right_parts) > 1 else batch_right_parts[0]
                else:
                    outlier_final_left = cp.empty(0, dtype=cp.int32)
                    outlier_final_right = cp.empty(0, dtype=cp.int32)
            else:
                outlier_final_left = cp.empty(0, dtype=cp.int32)
                outlier_final_right = cp.empty(0, dtype=cp.int32)
    else:
        outlier_final_left = cp.empty(0, dtype=cp.int32)
        outlier_final_right = cp.empty(0, dtype=cp.int32)

    # --- Merge main + outlier candidates on device ---
    with hotpath_stage("segment.candidates.assemble_output", category="emit"):
        if outlier_final_left.size > 0 and main_final_left.size > 0:
            final_left = cp.concatenate([main_final_left, outlier_final_left])
            final_right = cp.concatenate([main_final_right, outlier_final_right])
        elif outlier_final_left.size > 0:
            final_left = outlier_final_left
            final_right = outlier_final_right
        else:
            final_left = main_final_left
            final_right = main_final_right

        total_candidates = int(final_left.size)
    if total_candidates == 0:
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

    # Build output candidate arrays
    with hotpath_stage("segment.candidates.gather_rows", category="emit"):
        out_left_rows = left.row_indices[final_left]
        out_left_segs = left.segment_indices[final_left]
        out_right_rows = right.row_indices[final_right]
        out_right_segs = right.segment_indices[final_right]

    return DeviceSegmentIntersectionCandidates(
        left_rows=out_left_rows,
        left_segments=out_left_segs,
        left_lookup=final_left,
        right_rows=out_right_rows,
        right_segments=out_right_segs,
        right_lookup=final_right,
        count=total_candidates,
    )


# ---------------------------------------------------------------------------
# Legacy CPU segment extraction (kept for CPU fallback)
# ---------------------------------------------------------------------------

def _valid_global_rows(geometry_array: OwnedGeometryArray, family_name: str) -> np.ndarray:
    tag = FAMILY_TAGS[family_name]
    return np.flatnonzero(geometry_array.validity & (geometry_array.tags == tag)).astype(np.int32, copy=False)


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
        for family_row, row_index in enumerate(global_rows.tolist()):  # zcopy:ok(CPU-only legacy path: global_rows is np.ndarray from np.flatnonzero)
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
    return _generate_segment_candidates_from_tables(left_segments, right_segments, tile_size=tile_size)


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
            return SegmentIntersectionKind.DISJOINT, (float("nan"), float("nan")), (float("nan"),) * 4
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
    candidate_pairs: SegmentIntersectionCandidates | DeviceSegmentIntersectionCandidates | None = None,
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
) -> SegmentIntersectionResult:
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

    # --- Kernel 2: Generate candidates on GPU ---
    with hotpath_stage("segment.classify.generate_candidates", category="filter"):
        precomputed_candidates = candidate_pairs if candidate_pairs is not None else pairs
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
                )
        except Exception as exc:
            raise RuntimeError(
                f"segment candidate generation failed: {type(exc).__name__}: {exc}"
            ) from exc

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
) -> SegmentIntersectionResult:
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
            buf.x.size for buf in left.families.values()
            if buf.family in {GeometryFamily.LINESTRING, GeometryFamily.POLYGON,
                              GeometryFamily.MULTILINESTRING, GeometryFamily.MULTIPOLYGON}
        ) + sum(
            buf.x.size for buf in right.families.values()
            if buf.family in {GeometryFamily.LINESTRING, GeometryFamily.POLYGON,
                              GeometryFamily.MULTILINESTRING, GeometryFamily.MULTIPOLYGON}
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
            f"classify_segment_intersections failed at {stage}: "
            f"{type(exc).__name__}: {exc}"
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
        row_point_intersection_counts=np.bincount(point_rows, minlength=row_count).astype(np.int64, copy=False),
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
    result = classify_segment_intersections(left, right, tile_size=tile_size, dispatch_mode=dispatch_mode)
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
