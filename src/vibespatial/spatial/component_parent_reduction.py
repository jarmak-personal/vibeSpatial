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
from vibespatial.runtime.hotpath_trace import attach_work_amplification, hotpath_stage
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
    )
    if candidates is None:
        if candidate_execution.selected is not ExecutionMode.GPU:
            raise RuntimeError(
                "component-parent candidate generation declined after native "
                "selection; refusing a post-submission retry"
            )
        empty = cp.empty(0, dtype=cp.uint64)
        return _SortedParentKeys(
            keys=empty,
            unique=cp.empty(0, dtype=cp.bool_),
            owners=(native_index, components, component_bounds, candidate_execution),
        )
    candidates.validate_error_flag()
    capacity = int(candidates.total_pairs)
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
    _require_memory(
        stage=f"spatial.component_parent.{side}_key_sort",
        required_bytes=capacity * 160,
        requested_units=capacity,
    )
    d_component_rows = cp.asarray(candidates.d_left, dtype=cp.int64)
    d_point_rows = cp.asarray(candidates.d_right, dtype=cp.int64)
    d_locations = _classify_indexed_point_region(
        native_index.geometry,
        components.geometry,
        d_point_rows,
        d_component_rows,
        region_family=GeometryFamily.POLYGON,
        precision_plan=_resolve_indexed_point_precision_plan(None),
        return_device=True,
    )
    d_active = d_locations != cp.uint8(0)
    d_safe_components = cp.where(d_active, d_component_rows, cp.int64(0))
    d_parents = components.parent_rows[d_safe_components]
    d_parent_keys = (
        d_point_rows.astype(cp.uint64, copy=False) * cp.uint64(parent_count)
        + d_parents
    )
    d_ordered_keys = (
        d_parent_keys * cp.uint64(components.capacity)
        + d_component_rows.astype(cp.uint64, copy=False)
    )
    d_ordered_keys = cp.where(
        d_active,
        d_ordered_keys,
        cp.uint64(_U64_SENTINEL),
    )
    sorted_result = sort_pairs(
        d_ordered_keys,
        cp.arange(capacity, dtype=cp.int32),
        strategy=PairSortStrategy.RADIX,
        synchronize=False,
    )
    d_sorted_ordered = cp.asarray(sorted_result.keys, dtype=cp.uint64)
    d_sorted_active = d_sorted_ordered != cp.uint64(_U64_SENTINEL)
    d_sorted_parent = cp.where(
        d_sorted_active,
        d_sorted_ordered // cp.uint64(components.capacity),
        cp.uint64(_U64_SENTINEL),
    )
    d_sorted_locations = d_locations[
        cp.asarray(sorted_result.values, dtype=cp.int32)
    ]
    d_unique = d_sorted_active & (d_sorted_locations == cp.uint8(2))
    if capacity > 1:
        d_unique[1:] &= d_sorted_parent[1:] != d_sorted_parent[:-1]
    return _SortedParentKeys(
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
        ),
    )


def _profiled_sorted_parent_keys(
    native_index,
    components: _MultipartComponentCapacity,
    *,
    parent_count: int,
    tree_count: int,
    side: str,
) -> _SortedParentKeys:
    with hotpath_stage(
        "spatial.component_parent.key_sort",
        category="sort",
    ) as stage_metadata:
        result = _classified_parent_keys(
            native_index,
            components,
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
    *,
    parent_count: int,
    tree_count: int,
):
    # Boolean compaction can retain every candidate in the all-interior case.
    # Admit both active streams, tree-row gathers, lower-bound positions, and
    # shared-key masks/temporaries at their full capacities before allocation.
    reduction_bytes = tree_count * 48
    reduction_bytes += (left.capacity + right.capacity) * 64
    _require_memory(
        stage="spatial.component_parent.intersection_reduce",
        required_bytes=reduction_bytes,
        requested_units=left.capacity,
    )
    left_counts = cp.zeros(tree_count, dtype=cp.uint64)
    right_counts = cp.zeros(tree_count, dtype=cp.uint64)
    shared_counts = cp.zeros(tree_count, dtype=cp.uint64)
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
    *,
    parent_count: int,
    tree_count: int,
):
    with hotpath_stage(
        "spatial.component_parent.pair_intersection_reduce",
        category="emit",
    ) as stage_metadata:
        result = _reduce_sorted_parent_keys(
            left,
            right,
            parent_count=parent_count,
            tree_count=tree_count,
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

    parent_count = int(query_owned.row_count)
    left_keys = _profiled_sorted_parent_keys(
        native_index,
        components,
        parent_count=parent_count,
        tree_count=tree_count,
        side="left",
    )
    right_keys = _profiled_sorted_parent_keys(
        aligned_native_index,
        components,
        parent_count=parent_count,
        tree_count=tree_count,
        side="right",
    )
    values = _profiled_reduce_sorted_parent_keys(
        left_keys,
        right_keys,
        parent_count=parent_count,
        tree_count=tree_count,
    )
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
    from vibespatial.spatial.point_partition import retain_point_partition_completion

    retain_point_partition_completion(
        native_index,
        aligned_native_index,
        query_owned,
        components,
        left_keys,
        right_keys,
        values,
    )
    return values, execution
