#!/usr/bin/env python3
"""Q11 parent control versus exact component-to-parent device reduction."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import sys
import tempfile
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

CAPSULE = Path(__file__).resolve().parent
REPO = CAPSULE.parents[2]
R2 = CAPSULE.parent / "2026-08-25-r2"
DATA = Path("/home/picard/datasets/spatialbench/v0.1.0/sf100-geoparquet")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(R2))

os.environ.setdefault("VIBESPATIAL_STRICT_NATIVE", "1")

from q11_component_first import (  # noqa: E402
    _component_frames,
    _first_trip_batch,
    _query_objects,
)
from q12_dense_experiment import (  # noqa: E402
    _dataset_identity,
    _environment_identity,
    _source_tree_identity,
)


def _dispatch_summary(events) -> list[dict[str, object]]:
    counts = Counter(
        (
            event.surface,
            event.operation,
            event.implementation,
            event.selected,
            event.reason,
        )
        for event in events
    )
    return [
        {
            "surface": key[0],
            "operation": key[1],
            "implementation": key[2],
            "selected": key[3],
            "reason": key[4],
            "count": count,
        }
        for key, count in counts.most_common()
    ]


class _CudaStages:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    @contextlib.contextmanager
    def stage(self, name: str):
        import cupy as cp

        start = cp.cuda.Event()
        stop = cp.cuda.Event()
        wall_start = time.perf_counter()
        used_before = int(cp.get_default_memory_pool().used_bytes())
        start.record()
        try:
            yield
        finally:
            stop.record()
            stop.synchronize()
            used_after = int(cp.get_default_memory_pool().used_bytes())
            self.records.append(
                {
                    "name": name,
                    "gpu_seconds": float(cp.cuda.get_elapsed_time(start, stop)) / 1000.0,
                    "wall_seconds": time.perf_counter() - wall_start,
                    "pool_used_before_bytes": used_before,
                    "pool_used_after_bytes": used_after,
                }
            )


def _component_parent_ids(frame, *, parent_count: int):
    import cupy as cp

    from vibespatial.api.geo_base import _native_expression_from_public_series

    expression = _native_expression_from_public_series(frame["parent_zone_row"])
    if expression is None or not expression.is_device:
        raise RuntimeError("component parent lineage is not device-resident")
    result = cp.asarray(expression.values, dtype=cp.int64)
    if result.ndim != 1 or int(result.size) != len(frame):
        raise RuntimeError("component parent lineage is not row-aligned")
    if int(result.size):
        lo = int(result.min().item())
        hi = int(result.max().item())
        if lo < 0 or hi >= parent_count:
            raise RuntimeError("component parent lineage is outside parent capacity")
    return result


def _device_relation(index, geometry):
    from vibespatial.spatial.query_types import DeviceSpatialJoinResult

    result = index.query(
        geometry,
        predicate="contains",
        sort=False,
        output_format="indices",
        return_device=True,
    )
    if not isinstance(result, DeviceSpatialJoinResult):
        raise RuntimeError("component relation did not select device execution")
    return result


def _native_relation(index, geometry):
    from vibespatial.runtime import ExecutionMode

    owned = geometry.values.owned
    relation, execution = index._native_spatial_index_for_query().query_relation(
        owned,
        predicate="contains",
        query_row_count=owned.row_count,
        return_device=True,
        return_metadata=True,
    )
    if execution is None or execution.selected is not ExecutionMode.GPU:
        raise RuntimeError("component relation did not select native GPU execution")
    return relation


def _fixed_parent_keys(relation, parent_ids, *, parent_count: int):
    import cupy as cp

    from vibespatial.api._native_relation import NativeRelationSelection
    from vibespatial.cuda.cccl_primitives import PairSortStrategy, sort_pairs

    if isinstance(relation, NativeRelationSelection):
        active = relation.selection.source_mask()
        pairs = relation.relation
    else:
        pairs = relation
        active = cp.ones(len(pairs), dtype=cp.bool_)
    component_rows = cp.asarray(pairs.left_indices, dtype=cp.int64)
    point_rows = cp.asarray(pairs.right_indices, dtype=cp.int64)
    safe_component_rows = cp.where(active, component_rows, cp.int64(0))
    parents = parent_ids[safe_component_rows]
    keys = point_rows * cp.uint64(parent_count) + parents.astype(cp.uint64)
    sentinel = cp.uint64(cp.iinfo(cp.uint64).max)
    keys = cp.where(active, keys, sentinel)
    if int(keys.size) > cp.iinfo(cp.int32).max:
        raise RuntimeError("component relation capacity exceeds radix-sort index width")
    sorted_result = sort_pairs(
        keys,
        cp.arange(keys.size, dtype=cp.int32),
        strategy=PairSortStrategy.RADIX,
        synchronize=False,
    )
    sorted_keys = cp.asarray(sorted_result.keys, dtype=cp.uint64)
    unique = sorted_keys != sentinel
    if int(sorted_keys.size) > 1:
        unique[1:] &= sorted_keys[1:] != sorted_keys[:-1]
    return sorted_keys, unique, int(keys.size)


def _fixed_component_frame_count(
    pickup,
    dropoff,
    components,
    parent_ids,
    *,
    parent_count: int,
    stages: _CudaStages,
):
    import cupy as cp

    from vibespatial.cuda.cccl_primitives import lower_bound

    with stages.stage("pickup_native_component_query"):
        left_relation = _native_relation(pickup.sindex, components.geometry)
    with stages.stage("pickup_fixed_parent_key_sort"):
        left_keys, left_unique, left_capacity = _fixed_parent_keys(
            left_relation,
            parent_ids,
            parent_count=parent_count,
        )
    del left_relation
    with stages.stage("dropoff_native_component_query"):
        right_relation = _native_relation(dropoff.sindex, components.geometry)
    with stages.stage("dropoff_fixed_parent_key_sort"):
        right_keys, right_unique, right_capacity = _fixed_parent_keys(
            right_relation,
            parent_ids,
            parent_count=parent_count,
        )
    del right_relation
    with stages.stage("fixed_parent_key_intersection_reduce"):
        left_counts = cp.zeros(len(pickup), dtype=cp.uint64)
        right_counts = cp.zeros(len(pickup), dtype=cp.uint64)
        shared_counts = cp.zeros(len(pickup), dtype=cp.uint64)
        left_safe = cp.where(left_unique, left_keys, cp.uint64(0))
        right_safe = cp.where(right_unique, right_keys, cp.uint64(0))
        cp.add.at(
            left_counts,
            left_safe // cp.uint64(parent_count),
            left_unique.astype(cp.uint64, copy=False),
        )
        cp.add.at(
            right_counts,
            right_safe // cp.uint64(parent_count),
            right_unique.astype(cp.uint64, copy=False),
        )
        if right_capacity:
            positions = lower_bound(
                right_keys,
                left_keys,
                synchronize=False,
            ).astype(cp.int64, copy=False)
            safe_positions = cp.minimum(positions, cp.int64(right_capacity - 1))
            shared = (
                left_unique
                & (positions < right_capacity)
                & right_unique[safe_positions]
                & (right_keys[safe_positions] == left_keys)
            )
            cp.add.at(
                shared_counts,
                left_safe // cp.uint64(parent_count),
                shared.astype(cp.uint64, copy=False),
            )
        count = (left_counts * right_counts).sum() - shared_counts.sum()
    return count, {
        "left_relation_capacity": left_capacity,
        "right_relation_capacity": right_capacity,
    }


def _parent_keys(relation, parent_ids, *, parent_count: int):
    import cupy as cp

    component_rows = cp.asarray(relation.d_left_idx, dtype=cp.int64)
    point_rows = cp.asarray(relation.d_right_idx, dtype=cp.int64)
    parents = parent_ids[component_rows]
    keys = point_rows * cp.int64(parent_count) + parents
    return cp.unique(keys)


def _component_frame_count(
    pickup,
    dropoff,
    components,
    parent_ids,
    *,
    parent_count: int,
    stages: _CudaStages,
) -> tuple[object, dict[str, int]]:
    import cupy as cp

    with stages.stage("pickup_component_query"):
        left_relation = _device_relation(pickup.sindex, components.geometry)
    left_component_hits = int(left_relation.d_left_idx.size)
    with stages.stage("pickup_parent_key_map_deduplicate"):
        left_keys = _parent_keys(
            left_relation,
            parent_ids,
            parent_count=parent_count,
        )
    del left_relation
    with stages.stage("dropoff_component_query"):
        right_relation = _device_relation(dropoff.sindex, components.geometry)
    right_component_hits = int(right_relation.d_left_idx.size)
    with stages.stage("dropoff_parent_key_map_deduplicate"):
        right_keys = _parent_keys(
            right_relation,
            parent_ids,
            parent_count=parent_count,
        )
    del right_relation
    with stages.stage("parent_key_intersection_reduce"):
        shared_keys = cp.intersect1d(left_keys, right_keys, assume_unique=True)
        left_points = left_keys // cp.int64(parent_count)
        right_points = right_keys // cp.int64(parent_count)
        shared_points = shared_keys // cp.int64(parent_count)
        # CuPy's histogram-backed bincount may allocate multi-GiB temporary
        # storage for millions of bins. Exact parent keys are unique, so a
        # fixed input-sized scatter is the correct physical reduction shape.
        left_counts = cp.zeros(len(pickup), dtype=cp.uint64)
        right_counts = cp.zeros(len(pickup), dtype=cp.uint64)
        shared_counts = cp.zeros(len(pickup), dtype=cp.uint64)
        cp.add.at(left_counts, left_points, cp.uint64(1))
        cp.add.at(right_counts, right_points, cp.uint64(1))
        cp.add.at(shared_counts, shared_points, cp.uint64(1))
        count = (left_counts * right_counts).sum() - shared_counts.sum()
    return count, {
        "left_component_hits": left_component_hits,
        "right_component_hits": right_component_hits,
        "left_parent_keys": int(left_keys.size),
        "right_parent_keys": int(right_keys.size),
        "shared_parent_keys": int(shared_keys.size),
    }


def _run_parent(queries, pickup, dropoff, zones_parent, stages: _CudaStages):
    count = 0
    for zones in zones_parent:
        with stages.stage("parent_query_pair_aggregate"):
            result = queries._zone_pair_aggregates(pickup, dropoff, zones)
            contribution = (
                result["left_count"] * result["right_count"]
            ).sum() - result["shared_count"].sum()
        count += int(contribution)
    return count, []


def _run_component(
    pickup,
    dropoff,
    zones_parent,
    zones_component,
    stages: _CudaStages,
):
    import cupy as cp

    parent_count = sum(len(frame) for frame in zones_parent)
    parent_ids = [
        _component_parent_ids(frame, parent_count=parent_count)
        for frame in zones_component
    ]
    cp.cuda.get_current_stream().synchronize()
    device_counts = []
    frame_metrics = []
    for components, ids in zip(zones_component, parent_ids, strict=True):
        count, metrics = _component_frame_count(
            pickup,
            dropoff,
            components,
            ids,
            parent_count=parent_count,
            stages=stages,
        )
        device_counts.append(count)
        frame_metrics.append(metrics)
    with stages.stage("terminal_scalar_reduce_export"):
        total = int(cp.stack(device_counts).sum().item())
    return total, frame_metrics


def _native_component_frame(zones, stages: _CudaStages):
    import cupy as cp

    from vibespatial.api import GeoSeries
    from vibespatial.constructive.binary_constructive import (
        _explode_polygonal_rows_to_polygon_capacity_gpu,
    )
    from vibespatial.geometry.device_array import DeviceGeometryArray

    with stages.stage("native_component_explode"):
        owned = zones.geometry.values.owned
        parts = _explode_polygonal_rows_to_polygon_capacity_gpu(owned)
        if parts is None:
            raise RuntimeError("native polygon component expansion declined")
        components = GeoSeries(
            DeviceGeometryArray._from_owned(parts.geometry, crs=zones.crs),
            crs=zones.crs,
        )
        parent_ids = cp.asarray(parts.source_rows, dtype=cp.int64)
    return components, parent_ids, parts


def _run_native_component(
    pickup,
    dropoff,
    zones_parent,
    stages: _CudaStages,
):
    import cupy as cp

    device_counts = []
    frame_metrics = []
    owners = []
    for zones in zones_parent:
        components, ids, parts = _native_component_frame(zones, stages)
        count, metrics = _component_frame_count(
            pickup,
            dropoff,
            components,
            ids,
            parent_count=len(zones),
            stages=stages,
        )
        metrics["component_capacity"] = len(components)
        device_counts.append(count)
        frame_metrics.append(metrics)
        owners.append((components, ids, parts))
    with stages.stage("terminal_scalar_reduce_export"):
        total = int(cp.stack(device_counts).sum().item())
    return total, frame_metrics, owners


def _run_native_fixed_component(
    pickup,
    dropoff,
    zones_parent,
    stages: _CudaStages,
):
    import cupy as cp

    device_counts = []
    frame_metrics = []
    owners = []
    for zones in zones_parent:
        components, ids, parts = _native_component_frame(zones, stages)
        count, metrics = _fixed_component_frame_count(
            pickup,
            dropoff,
            components,
            ids,
            parent_count=len(zones),
            stages=stages,
        )
        metrics["component_capacity"] = len(components)
        device_counts.append(count)
        frame_metrics.append(metrics)
        owners.append((components, ids, parts))
    with stages.stage("terminal_scalar_reduce_export"):
        total = int(cp.stack(device_counts).sum().item())
    return total, frame_metrics, owners


def _identity(variant: str, zone_frames: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "complete_contemporaneous",
        "captured_at": datetime.now(UTC).isoformat(),
        "source": _source_tree_identity(Path(__file__).resolve()),
        "dataset": _dataset_identity(DATA),
        "environment": _environment_identity(),
        "measurement": {
            "query": "SF100 Q11 first 4M-row trip batch",
            "variant": variant,
            "zone_frames": zone_frames,
            "process_isolation": "one command/process per arm",
            "warmup_runs": 0,
            "repeat_runs": 1,
            "statistic": "single cold observation",
            "clock": "time.perf_counter wall time with synchronized stage events",
            "excluded_scope": "trip/zone IO and host component fixture decomposition",
            "included_scope": (
                "public spatial queries, device parent mapping/deduplication, "
                "shared-parent intersection, reduction, and terminal scalar export"
            ),
            "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        },
    }


def run(variant: str, *, zone_frames: int) -> dict[str, object]:
    import cupy as cp

    import vibespatial
    from vibespatial.runtime.fallbacks import clear_fallback_events, get_fallback_events
    from vibespatial.runtime.materialization import (
        clear_materialization_events,
        get_materialization_events,
    )

    gpd, paths, queries = _query_objects()
    zones_parent = queries._zone_frames(paths)[:zone_frames]
    with tempfile.TemporaryDirectory(prefix="vs-r3-q11-components-") as raw_directory:
        zones_component = (
            _component_frames(gpd, zones_parent, Path(raw_directory))
            if variant == "component-parent"
            else None
        )
        trips = _first_trip_batch(queries, paths)
        pickup = trips.set_geometry("t_pickuploc").geometry
        dropoff = trips.set_geometry("t_dropoffloc").geometry
        trip_rows = len(trips)
        parent_zone_rows = sum(len(frame) for frame in zones_parent)
        component_zone_rows = (
            sum(len(frame) for frame in zones_component)
            if zones_component is not None
            else None
        )
        clear_fallback_events()
        clear_materialization_events()
        vibespatial.clear_dispatch_events()
        cp.cuda.get_current_stream().synchronize()
        stages = _CudaStages()
        pool_before = int(cp.get_default_memory_pool().used_bytes())
        started = time.perf_counter()
        native_owners = None
        if variant == "parent":
            count, frame_metrics = _run_parent(
                queries,
                pickup,
                dropoff,
                zones_parent,
                stages,
            )
        elif variant == "component-parent":
            count, frame_metrics = _run_component(
                pickup,
                dropoff,
                zones_parent,
                zones_component,
                stages,
            )
            native_owners = None
        elif variant == "native-component-parent":
            count, frame_metrics, native_owners = _run_native_component(
                pickup,
                dropoff,
                zones_parent,
                stages,
            )
        else:
            count, frame_metrics, native_owners = _run_native_fixed_component(
                pickup,
                dropoff,
                zones_parent,
                stages,
            )
        cp.cuda.get_current_stream().synchronize()
        wall_seconds = time.perf_counter() - started
        pool_after = int(cp.get_default_memory_pool().used_bytes())
        fallback_events = get_fallback_events(clear=True)
        materialization_events = get_materialization_events(clear=True)
        dispatch_events = vibespatial.get_dispatch_events(clear=True)
        del dropoff, pickup, trips, zones_component, zones_parent
        if native_owners is not None:
            del native_owners
        gc.collect()
        cp.cuda.get_current_stream().synchronize()
    return {
        "schema_version": 1,
        "experiment": "q11_parent_aware_component",
        "variant": variant,
        "zone_frames": zone_frames,
        "trip_rows": trip_rows,
        "parent_zone_rows": parent_zone_rows,
        "component_zone_rows": component_zone_rows,
        "cross_zone_trip_count": count,
        "wall_seconds": wall_seconds,
        "stage_records": stages.records,
        "frame_metrics": frame_metrics,
        "pool_used_before_bytes": pool_before,
        "pool_used_after_bytes": pool_after,
        "fallback_event_count": len(fallback_events),
        "fallback_events": [event.to_dict() for event in fallback_events],
        "materialization_event_count": len(materialization_events),
        "materialization_events": [event.to_dict() for event in materialization_events],
        "dispatch_summary": _dispatch_summary(dispatch_events),
        "evidence_identity": _identity(variant, zone_frames),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=(
            "parent",
            "component-parent",
            "native-component-parent",
            "native-fixed-component-parent",
        ),
        required=True,
    )
    parser.add_argument("--zone-frames", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.variant, zone_frames=args.zone_frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
