"""Parent-aware reduction of point matches against multipart query rows.

Physical shape
--------------
Logical ``MultiPolygon contains Point`` matches are lowered to Polygon-part
candidate/refine relations.  Exact part hits are mapped back to packed
``(point row, parent query row)`` keys, radix sorted, deduplicated, and reduced
to the indexed-row result shape.  Two aligned point indexes intersect their
sorted parent-key streams so pickup and dropoff points may match different
parts of the same parent.

This is a private reducer beneath the existing public spatial-index aggregate
API.  It does not export relation pairs.  Dynamic exact relations retain their
ordinary query-stage admission; every additional key/sort/reduction temporary
is admitted from host-known capacity before allocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - CPU-only installs
    cp = None

from vibespatial.cuda._runtime import get_cuda_runtime
from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.cccl_primitives import (
    PairSortStrategy,
    lower_bound,
    sort_pairs,
)
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    build_device_resident_owned,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.hotpath_trace import (
    attach_work_amplification,
    hotpath_stage,
    hotpath_timing_enabled,
)
from vibespatial.spatial.query_types import (
    CandidateRelationCapacityError,
    SpatialQueryExecution,
)

request_warmup(["lower_bound_u64", "radix_sort_u64_i32"])


# This is deliberately a conservative bootstrap admission, not the steady-state
# device planner.  R3 evidence found a 1,404--18,933 part maximum in every SF100
# Q11 zone frame, while ordinary multipart rows stay well below this boundary.
# The work-amplification packet records the observed maximum so a later selector
# can replace this floor with candidate-weighted evidence.
_COMPONENT_PARENT_MIN_MAX_PARTS = 1_024
_COMPONENT_PARENT_MIN_AVERAGE_EXTRA_PART_LANES = 500_000
_U64_SENTINEL = (1 << 64) - 1


@dataclass(frozen=True, slots=True)
class _MultipartComponentCapacity:
    geometry: object
    parent_rows: object
    max_parts_per_parent: int
    parent_count: int
    readiness: object

    @property
    def capacity(self) -> int:
        return int(self.geometry.row_count)


@dataclass(frozen=True, slots=True)
class _SortedParentKeys:
    keys: object
    unique: object
    owners: tuple[object, ...]

    @property
    def capacity(self) -> int:
        return int(self.keys.size)


@dataclass(slots=True)
class _CudaStageIntervals:
    """Non-overlapping CUDA-event intervals for profiling-only attribution."""

    events: list[object]
    labels: list[str]

    @classmethod
    def start(cls) -> _CudaStageIntervals | None:
        if cp is None or not hotpath_timing_enabled():
            return None
        event = cp.cuda.Event()
        event.record(cp.cuda.get_current_stream())
        return cls(events=[event], labels=[])

    def checkpoint(self, label: str) -> None:
        event = cp.cuda.Event()
        event.record(cp.cuda.get_current_stream())
        self.events.append(event)
        self.labels.append(label)

    def finish(self) -> dict[str, float]:
        if not self.labels:
            return {}
        self.events[-1].synchronize()
        return {
            label: max(
                0.0,
                float(cp.cuda.get_elapsed_time(start, end)) / 1000.0,
            )
            for label, start, end in zip(
                self.labels,
                self.events[:-1],
                self.events[1:],
                strict=True,
            )
        }


def _attach_cuda_stage_intervals(
    intervals: _CudaStageIntervals | None,
    metadata_by_label: dict[str, dict[str, object] | None],
) -> None:
    if intervals is None:
        return
    for label, elapsed_seconds in intervals.finish().items():
        metadata = metadata_by_label.get(label)
        if metadata is not None:
            metadata.update(
                {
                    "gpu_event_elapsed_seconds": elapsed_seconds,
                    "gpu_timing_source": "cuda_event_non_overlapping",
                }
            )
            attach_work_amplification(
                metadata,
                operation=f"component_parent_{label}_timing",
                metric_family="timing",
                sums={"gpu_event_elapsed_seconds": elapsed_seconds},
                maxima={"gpu_event_elapsed_seconds": elapsed_seconds},
                physical_shape="component_parent_stage_interval",
                consumer_kind=label,
                semantic_contract={
                    "cuda_events_non_overlapping": True,
                    "profiling_only_synchronization": True,
                    "candidate_output_reused": metadata.get(
                        "candidate_output_reused",
                        False,
                    ),
                },
            )


_PARENT_KEY_WORKSPACE_DTYPES = {
    "component_rows": "int32",
    "point_rows": "int32",
    "locations": "uint8",
    "active": "bool",
    "safe_components": "int32",
    "parents": "uint64",
    "parent_keys": "uint64",
    "ordered_keys": "uint64",
    "order": "int32",
    "sorted_keys": "uint64",
    "sorted_order": "int32",
    "sorted_active": "bool",
    "sorted_locations": "uint8",
    "unique": "bool",
}
_PARENT_KEY_PERSISTENT_BYTES_PER_LANE = 57
_PARENT_KEY_TOTAL_ADMISSION_BYTES_PER_LANE = 160
_REDUCTION_COUNT_PERSISTENT_BYTES_PER_TREE = 24
_REDUCTION_TOTAL_ADMISSION_BYTES_PER_TREE = 48


def _parent_key_admission_bytes(*, capacity: int, persistent_capacity: int) -> int:
    capacity = max(int(capacity), 0)
    growth = max(capacity - int(persistent_capacity), 0)
    transient_per_lane = (
        _PARENT_KEY_TOTAL_ADMISSION_BYTES_PER_LANE
        - _PARENT_KEY_PERSISTENT_BYTES_PER_LANE
    )
    return (
        capacity * transient_per_lane
        + growth * _PARENT_KEY_PERSISTENT_BYTES_PER_LANE
    )


def _reduction_admission_bytes(
    *,
    tree_count: int,
    persistent_tree_capacity: int,
    candidate_capacity: int,
) -> int:
    tree_count = max(int(tree_count), 0)
    growth = max(tree_count - int(persistent_tree_capacity), 0)
    transient_per_tree = (
        _REDUCTION_TOTAL_ADMISSION_BYTES_PER_TREE
        - _REDUCTION_COUNT_PERSISTENT_BYTES_PER_TREE
    )
    return (
        tree_count * transient_per_tree
        + growth * _REDUCTION_COUNT_PERSISTENT_BYTES_PER_TREE
        + max(int(candidate_capacity), 0) * 64
    )


@dataclass(slots=True)
class _ParentKeyWorkspaceSlot:
    capacity: int = 0
    buffers: dict[str, object] | None = None

    def ensure(self, capacity: int) -> bool:
        capacity = int(capacity)
        if capacity <= self.capacity:
            return False
        self.buffers = {
            name: cp.empty(capacity, dtype=dtype)
            for name, dtype in _PARENT_KEY_WORKSPACE_DTYPES.items()
        }
        self.buffers["order"][:] = cp.arange(capacity, dtype=cp.int32)
        self.capacity = capacity
        return True

    def view(self, name: str, size: int):
        assert self.buffers is not None
        return self.buffers[name][: int(size)]

    @property
    def persistent_bytes(self) -> int:
        if self.buffers is None:
            return 0
        return sum(int(value.nbytes) for value in self.buffers.values())


@dataclass(slots=True)
class _ComponentParentWorkspace:
    left: _ParentKeyWorkspaceSlot
    right: _ParentKeyWorkspaceSlot
    tree_capacity: int = 0
    reduction_counts: tuple[object, object, object] | None = None
    readiness: object | None = None

    @classmethod
    def empty(cls) -> _ComponentParentWorkspace:
        return cls(
            left=_ParentKeyWorkspaceSlot(),
            right=_ParentKeyWorkspaceSlot(),
        )

    def slot(self, side: str, capacity: int) -> tuple[_ParentKeyWorkspaceSlot, bool]:
        slot = self.left if side == "left" else self.right
        return slot, slot.ensure(capacity)

    def counts(self, tree_count: int) -> tuple[tuple[object, object, object], bool]:
        tree_count = int(tree_count)
        grew = tree_count > self.tree_capacity
        if grew:
            self.reduction_counts = tuple(
                cp.empty(tree_count, dtype=cp.uint64) for _ in range(3)
            )
            self.tree_capacity = tree_count
        assert self.reduction_counts is not None
        return tuple(values[:tree_count] for values in self.reduction_counts), grew

    @property
    def persistent_bytes(self) -> int:
        count_bytes = (
            0
            if self.reduction_counts is None
            else sum(int(value.nbytes) for value in self.reduction_counts)
        )
        return self.left.persistent_bytes + self.right.persistent_bytes + count_bytes


def _require_memory(*, stage: str, required_bytes: int, requested_units: int) -> None:
    admission = get_cuda_runtime().admit_device_memory(
        stage=stage,
        required_bytes=max(int(required_bytes), 0),
        requested_units=max(int(requested_units), 0),
    )
    if admission.admitted:
        return
    gib = 1024**3
    raise CandidateRelationCapacityError(
        f"{stage} requires {admission.required_bytes / gib:.2f} GiB but the "
        f"active query envelope has {admission.remaining_bytes / gib:.2f} GiB; "
        "the component-parent reducer has already selected native execution "
        "and will not retry a different provider"
    )


def _component_cache_lock(query_owned) -> RLock:
    """Return the one owner-lifetime lock for the bounded derived cache."""
    cache = query_owned.__dict__
    lock = cache.get("_component_parent_cache_lock")
    if lock is None:
        lock = cache.setdefault("_component_parent_cache_lock", RLock())
    return lock


def _component_execution_lock(query_owned) -> RLock:
    """Serialize owner-local workspace leases across CUDA streams."""
    cache = query_owned.__dict__
    lock = cache.get("_component_parent_execution_lock")
    if lock is None:
        lock = cache.setdefault("_component_parent_execution_lock", RLock())
    return lock


def _component_workspace(query_owned) -> _ComponentParentWorkspace:
    cache = query_owned.__dict__
    workspace = cache.get("_component_parent_workspace")
    if not isinstance(workspace, _ComponentParentWorkspace):
        workspace = _ComponentParentWorkspace.empty()
        cache["_component_parent_workspace"] = workspace
    return workspace


def _write_component_parent_ordered_keys(
    point_rows,
    component_rows,
    parents,
    *,
    parent_count: int,
    component_capacity: int,
    parent_keys,
    ordered_keys,
) -> None:
    """Write exact uint64 parent and stable-component packed keys."""
    cp.copyto(parent_keys, point_rows, casting="unsafe")
    cp.multiply(parent_keys, cp.uint64(parent_count), out=parent_keys)
    cp.add(parent_keys, parents, out=parent_keys)
    cp.multiply(
        parent_keys,
        cp.uint64(component_capacity),
        out=ordered_keys,
    )
    # CuPy promotes uint64 + int32 to float64. Reuse parent_keys as the exact
    # uint64 component addend after the parent key has been consumed.
    cp.copyto(parent_keys, component_rows, casting="unsafe")
    cp.add(ordered_keys, parent_keys, out=ordered_keys)


def _admitted_point_index(native_index) -> bool:
    """Admit only dense, valid, non-empty Point carriers without indirection."""
    from vibespatial.spatial.point_partition import wait_for_point_partition

    wait_for_point_partition(native_index.readiness)
    geometry = native_index.geometry
    if getattr(geometry, "is_indexed_view", False):
        return False
    state = geometry._ensure_device_state(preserve_indexed_view=True)
    if (
        state.trusted_homogeneous_family is not GeometryFamily.POINT
        or state.trusted_all_valid is not True
        or state.trusted_all_non_empty is not True
    ):
        return False
    buffer = state.families.get(GeometryFamily.POINT)
    return bool(
        buffer is not None
        and int(buffer.geometry_offsets.size) == int(native_index.row_count) + 1
    )


def _admitted_multipolygon_components(
    query_owned,
    *,
    native_index,
    aligned_native_index,
    tree_count: int,
) -> _MultipartComponentCapacity | None:
    """Return a conservative homogeneous MultiPolygon part carrier.

    The component classifier preserves stable part order and the first
    non-exterior point location used by the authoritative MultiPolygon kernel.
    That remains exact for both valid and invalid multipart containers rather
    than assuming component-interior OR semantics. Null/empty or row-indirected
    inputs decline here; the existing reducer remains authoritative for those
    shapes.
    """
    if (
        cp is None
        or query_owned.is_indexed_view
        or not _admitted_point_index(native_index)
        or not _admitted_point_index(aligned_native_index)
    ):
        return None
    from vibespatial.spatial.point_partition import (
        record_point_partition_readiness,
        wait_for_point_partition,
    )

    # Exactly one derived carrier is retained per immutable query owner.  The
    # owner lifetime is the eviction boundary; publication is serialized and
    # carries an event so another CUDA stream cannot observe partial state.
    with _component_cache_lock(query_owned):
        cached = getattr(query_owned, "_component_parent_capacity_cache", None)
        if isinstance(cached, _MultipartComponentCapacity):
            wait_for_point_partition(cached.readiness)
            extra_parts = max(cached.capacity - cached.parent_count, 0)
            average_extra_part_lanes = (
                int(tree_count) * extra_parts // max(cached.parent_count, 1)
            )
            if (
                cached.max_parts_per_parent >= _COMPONENT_PARENT_MIN_MAX_PARTS
                and average_extra_part_lanes
                >= _COMPONENT_PARENT_MIN_AVERAGE_EXTRA_PART_LANES
            ):
                return cached
            return None
        state = query_owned._ensure_device_state(preserve_indexed_view=False)
        if (
            state.trusted_homogeneous_family is not GeometryFamily.MULTIPOLYGON
            or state.trusted_all_valid is not True
        ):
            return None
        buffer = state.families.get(GeometryFamily.MULTIPOLYGON)
        if (
            buffer is None
            or buffer.part_offsets is None
            or buffer.ring_offsets is None
            or int(buffer.geometry_offsets.size) != int(query_owned.row_count) + 1
        ):
            return None

        row_count = int(query_owned.row_count)
        part_capacity = max(int(buffer.part_offsets.size) - 1, 0)
        if part_capacity == 0 or part_capacity > (1 << 31) - 1:
            return None
        coordinate_capacity = int(buffer.x.size)
        # One complete pre-submission envelope: persistent component routing;
        # all heavy-tail/carrier temporaries; the prepared eight-bin part-y
        # directory (112 persistent + 64 scan bytes/part); conservative scan
        # scratch; and the worst-case eight memberships per source coordinate.
        # Coordinates and topology offsets themselves remain shared with the
        # immutable owner.
        _require_memory(
            stage="spatial.component_parent.derived_component_cache",
            required_bytes=(
                part_capacity * 320
                + row_count * 32
                + coordinate_capacity * 32
            ),
            requested_units=part_capacity,
        )
        d_geometry_offsets = cp.asarray(buffer.geometry_offsets, dtype=cp.int64)
        d_family_rows = cp.asarray(state.family_row_offsets, dtype=cp.int64)
        d_identity = cp.all(d_family_rows == cp.arange(row_count, dtype=cp.int64))
        d_part_counts = d_geometry_offsets[1:] - d_geometry_offsets[:-1]
        d_max_parts = (
            cp.max(d_part_counts)
            if row_count
            else cp.asarray(0, dtype=cp.int64)
        )
        proof = cp.stack(
            (
                d_identity.astype(cp.int64, copy=False),
                d_max_parts.astype(cp.int64, copy=False),
            )
        )
        identity, max_parts = get_cuda_runtime().copy_device_to_host(
            proof,
            reason="component-parent heavy-tail admission planning packet",
        )
        max_parts = int(max_parts)
        if not bool(identity) or max_parts < _COMPONENT_PARENT_MIN_MAX_PARTS:
            return None

        extra_parts = max(part_capacity - row_count, 0)
        average_extra_part_lanes = (
            int(tree_count) * extra_parts // max(row_count, 1)
        )
        if (
            average_extra_part_lanes
            < _COMPONENT_PARENT_MIN_AVERAGE_EXTRA_PART_LANES
        ):
            return None
        d_part_rows = cp.arange(part_capacity, dtype=cp.int64)
        d_logical_parts = d_geometry_offsets[-1]
        d_active = d_part_rows < d_logical_parts
        d_safe_parts = cp.minimum(
            d_part_rows,
            cp.maximum(d_logical_parts - 1, cp.int64(0)),
        )
        d_parent_rows = cp.searchsorted(
            d_geometry_offsets[1:],
            d_safe_parts,
            side="right",
        ).astype(cp.uint64, copy=False)
        d_parent_rows = cp.where(d_active, d_parent_rows, cp.uint64(0))

        polygon_buffer = DeviceFamilyGeometryBuffer(
            family=GeometryFamily.POLYGON,
            x=buffer.x,
            y=buffer.y,
            geometry_offsets=buffer.part_offsets,
            empty_mask=~d_active,
            ring_offsets=buffer.ring_offsets,
            bounds=None,
        )
        parts = build_device_resident_owned(
            device_families={GeometryFamily.POLYGON: polygon_buffer},
            row_count=part_capacity,
            tags=cp.full(
                part_capacity,
                FAMILY_TAGS[GeometryFamily.POLYGON],
                dtype=cp.int8,
            ),
            validity=d_active,
            family_row_offsets=cp.arange(part_capacity, dtype=cp.int32),
            execution_mode="gpu",
        )
        part_state = parts._ensure_device_state(preserve_indexed_view=True)
        part_state.trusted_homogeneous_family = GeometryFamily.POLYGON
        part_state.trusted_all_valid = True
        part_state.trusted_all_non_empty = True
        part_state.trusted_polygonal_only = True
        part_state.trusted_family_domain = (GeometryFamily.POLYGON,)
        from vibespatial.predicates.point_location_index import (
            prepare_polygon_part_y_index,
        )

        prepare_polygon_part_y_index(parts, GeometryFamily.POLYGON)
        result = _MultipartComponentCapacity(
            geometry=parts,
            parent_rows=d_parent_rows,
            max_parts_per_parent=max_parts,
            parent_count=row_count,
            readiness=record_point_partition_readiness(),
        )
        query_owned._component_parent_capacity_cache = result
        return result


def _classified_parent_keys(
    native_index,
    components: _MultipartComponentCapacity,
    workspace: _ComponentParentWorkspace,
    *,
    parent_count: int,
    tree_count: int,
    side: str,
) -> _SortedParentKeys:
    """Classify component candidates and retain first-hit parent keys.

    The authoritative MultiPolygon point kernel scans components in stable
    order and stops at the first boundary/interior classification. Sorting the
    raw Polygon locations by ``(point, parent, component)`` reproduces that
    rule exactly. This matters for invalid overlapping components, where
    reducing component ``contains`` booleans with OR is not GEOS-compatible.
    """
    from vibespatial.kernels.core.geometry_analysis import (
        compute_geometry_bounds_device,
    )
    from vibespatial.predicates.point_location_index import (
        prepare_point_region_y_indexes,
    )
    from vibespatial.predicates.point_relations import (
        _classify_indexed_point_region,
        _resolve_indexed_point_precision_plan,
    )
    from vibespatial.spatial.spatial_index_device import spatial_index_device_query

    intervals = _CudaStageIntervals.start()
    candidate_workspace_supplied = False
    candidate_workspace_grew = False

    def _candidate_output(capacity: int):
        nonlocal candidate_workspace_grew, candidate_workspace_supplied
        slot = workspace.left if side == "left" else workspace.right
        _require_memory(
            stage=f"spatial.component_parent.{side}_key_sort",
            required_bytes=_parent_key_admission_bytes(
                capacity=capacity,
                persistent_capacity=slot.capacity,
            ),
            requested_units=capacity,
        )
        slot, grew = workspace.slot(side, capacity)
        candidate_workspace_supplied = True
        candidate_workspace_grew = candidate_workspace_grew or grew
        return (
            slot.view("component_rows", capacity),
            slot.view("point_rows", capacity),
        )

    metadata_by_label: dict[str, dict[str, object] | None] = {}
    with hotpath_stage(
        f"spatial.component_parent.{side}.candidate_generation",
        category="filter",
    ) as candidate_metadata:
        metadata_by_label["candidate_generation"] = candidate_metadata
        prepare_point_region_y_indexes(components.geometry, native_index.geometry)
        component_bounds = compute_geometry_bounds_device(
            components.geometry,
            preserve_indexed_view=True,
        )
        candidates, candidate_execution = spatial_index_device_query(
            native_index.to_flat_index(),
            component_bounds,
            native_index=native_index,
            allow_bbox_superset=True,
            candidate_output=_candidate_output,
        )
        if intervals is not None:
            intervals.checkpoint("candidate_generation")
    if candidates is None:
        if candidate_execution.selected is not ExecutionMode.GPU:
            raise RuntimeError(
                "component-parent candidate generation declined after native "
                "selection; refusing a post-submission retry"
            )
        empty = cp.empty(0, dtype=cp.uint64)
        result = _SortedParentKeys(
            keys=empty,
            unique=cp.empty(0, dtype=cp.bool_),
            owners=(native_index, components, component_bounds, candidate_execution),
        )
        _attach_cuda_stage_intervals(intervals, metadata_by_label)
        return result
    candidates.validate_error_flag()
    capacity = int(candidates.total_pairs)
    if capacity == 0:
        empty = cp.empty(0, dtype=cp.uint64)
        result = _SortedParentKeys(
            keys=empty,
            unique=cp.empty(0, dtype=cp.bool_),
            owners=(native_index, components, component_bounds, candidate_execution),
        )
        _attach_cuda_stage_intervals(intervals, metadata_by_label)
        return result
    if capacity > (1 << 31) - 1:
        raise CandidateRelationCapacityError(
            "component-parent candidate relation exceeds radix-sort int32 capacity"
        )
    if parent_count <= 0 or tree_count > _U64_SENTINEL // parent_count:
        raise OverflowError("component-parent packed key exceeds uint64 capacity")
    if (
        components.capacity <= 0
        or tree_count * parent_count
        > _U64_SENTINEL // components.capacity
    ):
        raise OverflowError("component-parent ordered key exceeds uint64 capacity")
    # Complete simultaneous-live envelope for raw locations, mapping arrays,
    # ordered/parent keys, permutation values, radix outputs and temporary
    # storage, and sorted classification masks. The provider separately owns
    # and admits the candidate relation and exact-classification workspace.
    if not candidate_workspace_supplied:
        slot = workspace.left if side == "left" else workspace.right
        _require_memory(
            stage=f"spatial.component_parent.{side}_key_sort",
            required_bytes=_parent_key_admission_bytes(
                capacity=capacity,
                persistent_capacity=slot.capacity,
            ),
            requested_units=capacity,
        )
    slot, workspace_grew = workspace.slot(side, capacity)
    workspace_grew = workspace_grew or candidate_workspace_grew
    if candidate_metadata is not None:
        candidate_metadata.update(
            {
                "workspace_capacity": slot.capacity,
                "workspace_reused": not workspace_grew,
                "workspace_persistent_bytes": workspace.persistent_bytes,
                "candidate_output_reused": candidate_workspace_supplied,
            }
        )
    with hotpath_stage(
        f"spatial.component_parent.{side}.exact_classification",
        category="refine",
    ) as classification_metadata:
        metadata_by_label["exact_classification"] = classification_metadata
        d_component_rows = slot.view("component_rows", capacity)
        d_point_rows = slot.view("point_rows", capacity)
        if not candidate_workspace_supplied:
            cp.copyto(d_component_rows, candidates.d_left, casting="unsafe")
            cp.copyto(d_point_rows, candidates.d_right, casting="unsafe")
        d_locations = slot.view("locations", capacity)
        d_locations = _classify_indexed_point_region(
            native_index.geometry,
            components.geometry,
            d_point_rows,
            d_component_rows,
            region_family=GeometryFamily.POLYGON,
            precision_plan=_resolve_indexed_point_precision_plan(None),
            return_device=True,
            relation_out=d_locations,
        )
        if intervals is not None:
            intervals.checkpoint("exact_classification")
    with hotpath_stage(
        f"spatial.component_parent.{side}.parent_key_construction",
        category="emit",
    ) as construction_metadata:
        metadata_by_label["parent_key_construction"] = construction_metadata
        d_active = slot.view("active", capacity)
        cp.not_equal(d_locations, cp.uint8(0), out=d_active)
        d_safe_components = slot.view("safe_components", capacity)
        cp.copyto(d_safe_components, d_component_rows)
        cp.copyto(d_safe_components, cp.int32(0), where=~d_active)
        d_parents = slot.view("parents", capacity)
        cp.take(components.parent_rows, d_safe_components, out=d_parents)
        d_parent_keys = slot.view("parent_keys", capacity)
        d_ordered_keys = slot.view("ordered_keys", capacity)
        _write_component_parent_ordered_keys(
            d_point_rows,
            d_component_rows,
            d_parents,
            parent_count=parent_count,
            component_capacity=components.capacity,
            parent_keys=d_parent_keys,
            ordered_keys=d_ordered_keys,
        )
        cp.copyto(d_ordered_keys, cp.uint64(_U64_SENTINEL), where=~d_active)
        if intervals is not None:
            intervals.checkpoint("parent_key_construction")
    with hotpath_stage(
        f"spatial.component_parent.{side}.radix_sort",
        category="sort",
    ) as sort_metadata:
        metadata_by_label["radix_sort"] = sort_metadata
        sorted_result = sort_pairs(
            d_ordered_keys,
            slot.view("order", capacity),
            strategy=PairSortStrategy.RADIX,
            synchronize=False,
            out_keys=slot.view("sorted_keys", capacity),
            out_values=slot.view("sorted_order", capacity),
        )
        d_sorted_ordered = cp.asarray(sorted_result.keys, dtype=cp.uint64)
        if intervals is not None:
            intervals.checkpoint("radix_sort")
    with hotpath_stage(
        f"spatial.component_parent.{side}.deduplication",
        category="filter",
    ) as dedup_metadata:
        metadata_by_label["deduplication"] = dedup_metadata
        d_sorted_active = slot.view("sorted_active", capacity)
        cp.not_equal(
            d_sorted_ordered,
            cp.uint64(_U64_SENTINEL),
            out=d_sorted_active,
        )
        d_sorted_parent = d_sorted_ordered
        cp.floor_divide(
            d_sorted_parent,
            cp.uint64(components.capacity),
            out=d_sorted_parent,
        )
        cp.copyto(
            d_sorted_parent,
            cp.uint64(_U64_SENTINEL),
            where=~d_sorted_active,
        )
        d_sorted_locations = slot.view("sorted_locations", capacity)
        cp.take(d_locations, sorted_result.values, out=d_sorted_locations)
        d_unique = slot.view("unique", capacity)
        cp.equal(d_sorted_locations, cp.uint8(2), out=d_unique)
        cp.logical_and(d_unique, d_sorted_active, out=d_unique)
        if capacity > 1:
            cp.not_equal(
                d_sorted_parent[1:],
                d_sorted_parent[:-1],
                out=d_sorted_active[:-1],
            )
            cp.logical_and(
                d_unique[1:],
                d_sorted_active[:-1],
                out=d_unique[1:],
            )
        if intervals is not None:
            intervals.checkpoint("deduplication")
    result = _SortedParentKeys(
        keys=d_sorted_parent,
        unique=d_unique,
        owners=(
            native_index,
            components,
            component_bounds,
            candidates,
            candidate_execution,
            d_locations,
            d_active,
            d_component_rows,
            d_point_rows,
            d_safe_components,
            d_parents,
            d_parent_keys,
            d_ordered_keys,
            sorted_result,
            d_sorted_ordered,
            d_sorted_locations,
            slot,
        ),
    )
    _attach_cuda_stage_intervals(intervals, metadata_by_label)
    return result


def _profiled_sorted_parent_keys(
    native_index,
    components: _MultipartComponentCapacity,
    workspace: _ComponentParentWorkspace,
    *,
    parent_count: int,
    tree_count: int,
    side: str,
) -> _SortedParentKeys:
    with hotpath_stage(
        f"spatial.component_parent.{side}.parent_key_pipeline",
        category="sort",
    ) as stage_metadata:
        result = _classified_parent_keys(
            native_index,
            components,
            workspace,
            parent_count=parent_count,
            tree_count=tree_count,
            side=side,
        )
        attach_work_amplification(
            stage_metadata,
            operation="component_parent_key_sort",
            metric_family="relation",
            sums={
                "component_candidate_capacity": result.capacity,
                "parent_query_rows": int(parent_count),
                "indexed_rows": int(tree_count),
            },
            maxima={"key_capacity": result.capacity},
            unavailable=("logical_unique_parent_keys",),
            physical_shape="component_locations_to_ordered_parent_keys",
            consumer_kind=side,
            semantic_contract={
                "device_logical_counts_read": False,
                "inactive_capacity_uses_uint64_sentinel": True,
                "first_non_exterior_component_preserved": True,
                "duplicate_parent_keys_removed": True,
            },
        )
        return result


def _reduce_sorted_parent_keys(
    left: _SortedParentKeys,
    right: _SortedParentKeys,
    workspace: _ComponentParentWorkspace,
    *,
    parent_count: int,
    tree_count: int,
):
    # Boolean compaction can retain every candidate in the all-interior case.
    # Admit both active streams, tree-row gathers, lower-bound positions, and
    # shared-key masks/temporaries at their full capacities before allocation.
    _require_memory(
        stage="spatial.component_parent.intersection_reduce",
        required_bytes=_reduction_admission_bytes(
            tree_count=tree_count,
            persistent_tree_capacity=workspace.tree_capacity,
            candidate_capacity=left.capacity + right.capacity,
        ),
        requested_units=left.capacity,
    )
    (left_counts, right_counts, shared_counts), _workspace_grew = workspace.counts(
        tree_count
    )
    left_counts.fill(cp.uint64(0))
    right_counts.fill(cp.uint64(0))
    shared_counts.fill(cp.uint64(0))
    left_active = left.keys[left.unique]
    right_active = right.keys[right.unique]
    left_tree_rows = left_active // cp.uint64(parent_count)
    right_tree_rows = right_active // cp.uint64(parent_count)
    cp.add.at(left_counts, left_tree_rows, cp.uint64(1))
    cp.add.at(right_counts, right_tree_rows, cp.uint64(1))
    if int(right_active.size):
        positions = lower_bound(
            right_active,
            left_active,
            synchronize=False,
        ).astype(cp.int64, copy=False)
        safe_positions = cp.minimum(positions, cp.int64(right_active.size - 1))
        shared = (
            (positions < int(right_active.size))
            & (right_active[safe_positions] == left_active)
        )
        shared_keys = left_active[shared]
        shared_tree_rows = shared_keys // cp.uint64(parent_count)
        cp.add.at(shared_counts, shared_tree_rows, cp.uint64(1))
    else:
        positions = None
        safe_positions = None
        shared = None
        shared_keys = None
        shared_tree_rows = None

    from vibespatial.spatial.point_partition import retain_point_partition_completion

    results = (
        left_counts.astype(cp.int64, copy=False),
        right_counts.astype(cp.int64, copy=False),
        shared_counts.astype(cp.int64, copy=False),
    )
    retain_point_partition_completion(
        left,
        right,
        left_active,
        right_active,
        left_tree_rows,
        right_tree_rows,
        positions,
        safe_positions,
        shared,
        shared_keys,
        shared_tree_rows,
        left_counts,
        right_counts,
        shared_counts,
        *results,
    )
    return results


def _profiled_reduce_sorted_parent_keys(
    left: _SortedParentKeys,
    right: _SortedParentKeys,
    workspace: _ComponentParentWorkspace,
    *,
    parent_count: int,
    tree_count: int,
):
    with hotpath_stage(
        "spatial.component_parent.pair_intersection_reduce",
        category="emit",
    ) as stage_metadata:
        intervals = _CudaStageIntervals.start()
        result = _reduce_sorted_parent_keys(
            left,
            right,
            workspace,
            parent_count=parent_count,
            tree_count=tree_count,
        )
        if intervals is not None:
            intervals.checkpoint("pair_intersection_reduce")
            _attach_cuda_stage_intervals(
                intervals,
                {"pair_intersection_reduce": stage_metadata},
            )
        attach_work_amplification(
            stage_metadata,
            operation="component_parent_pair_intersection_reduce",
            metric_family="relation",
            sums={
                "left_parent_key_capacity": left.capacity,
                "right_parent_key_capacity": right.capacity,
                "terminal_rows": int(tree_count),
            },
            maxima={
                "parent_key_capacity": max(left.capacity, right.capacity),
            },
            unavailable=(
                "logical_left_parent_keys",
                "logical_right_parent_keys",
                "logical_shared_parent_keys",
            ),
            physical_shape="two_sorted_parent_key_streams_to_aligned_counts",
            consumer_kind="query_pair_aggregate",
            semantic_contract={
                "device_logical_counts_read": False,
                "different_components_same_parent_match": True,
                "public_pair_arrays_exported": False,
            },
        )
        if stage_metadata is not None:
            stage_metadata.update(
                {
                    "workspace_tree_capacity": workspace.tree_capacity,
                    "workspace_persistent_bytes": workspace.persistent_bytes,
                }
            )
        return result


def try_component_parent_pair_match_counts(
    native_index,
    aligned_native_index,
    query_owned,
    *,
    predicate: str,
):
    """Try the parent-aware component shape for aligned point indexes."""
    if cp is None or predicate not in {"contains", "contains_properly"}:
        return None
    if int(native_index.row_count) != int(aligned_native_index.row_count):
        return None
    tree_count = int(native_index.row_count)
    components = _admitted_multipolygon_components(
        query_owned,
        native_index=native_index,
        aligned_native_index=aligned_native_index,
        tree_count=tree_count,
    )
    if components is None:
        return None

    from vibespatial.spatial.point_partition import (
        record_point_partition_readiness,
        retain_point_partition_completion,
        wait_for_point_partition,
    )

    with _component_execution_lock(query_owned):
        workspace = _component_workspace(query_owned)
        if workspace.readiness is not None:
            wait_for_point_partition(workspace.readiness)
        try:
            parent_count = int(query_owned.row_count)
            left_keys = _profiled_sorted_parent_keys(
                native_index,
                components,
                workspace,
                parent_count=parent_count,
                tree_count=tree_count,
                side="left",
            )
            right_keys = _profiled_sorted_parent_keys(
                aligned_native_index,
                components,
                workspace,
                parent_count=parent_count,
                tree_count=tree_count,
                side="right",
            )
            values = _profiled_reduce_sorted_parent_keys(
                left_keys,
                right_keys,
                workspace,
                parent_count=parent_count,
                tree_count=tree_count,
            )
        finally:
            workspace.readiness = record_point_partition_readiness()
    execution = SpatialQueryExecution(
        requested=ExecutionMode.GPU,
        selected=ExecutionMode.GPU,
        implementation="owned_gpu_component_parent_pair_match_count",
        reason=(
            "heavy-tail MultiPolygon query rows were lowered to Polygon-part "
            "relations and reduced by exact parent keys; "
            f"max_parts_per_parent={components.max_parts_per_parent}"
        ),
    )
    retain_point_partition_completion(
        native_index,
        aligned_native_index,
        query_owned,
        components,
        left_keys,
        right_keys,
        values,
        workspace,
    )
    return values, execution
