#!/usr/bin/env python3
"""Q10 parent control versus component-first grouped attribute reduction."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

CAPSULE = Path(__file__).resolve().parent
REPO = CAPSULE.parents[2]
R2 = CAPSULE.parent / "2026-08-25-r2"
DATA = Path("/home/picard/datasets/spatialbench/v0.1.0/sf100-geoparquet")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(R2))
sys.path.insert(0, str(CAPSULE))

os.environ.setdefault("VIBESPATIAL_STRICT_NATIVE", "1")

from q11_component_first import _component_frames, _query_objects  # noqa: E402
from q11_parent_aware_component import (  # noqa: E402
    _component_parent_ids,
    _CudaStages,
    _dispatch_summary,
)
from q12_dense_experiment import (  # noqa: E402
    _dataset_identity,
    _environment_identity,
    _source_tree_identity,
)


def _device_values(series, *, dtype):
    import cupy as cp

    from vibespatial.api.geo_base import _native_expression_from_public_series

    expression = _native_expression_from_public_series(series)
    if expression is None or not expression.is_device:
        raise RuntimeError("query aggregate column is not device-resident")
    return cp.asarray(expression.values, dtype=dtype)


def _first_trip_batch(queries, paths):
    columns = [
        "t_tripkey",
        "t_pickuploc",
        "t_pickuptime",
        "t_dropofftime",
        "t_distance",
    ]
    return next(
        iter(
            queries._spatial_frames(
                paths["trip"],
                columns,
                "t_pickuploc",
                batch_rows=4_000_000,
            )
        )
    )


def _aggregate_device_columns(result):
    import cupy as cp

    return (
        _device_values(result["total_pickups"], dtype=cp.uint64),
        _device_values(result["distance_sum"], dtype=cp.float64),
        _device_values(result["duration_sum"], dtype=cp.float64),
    )


def _parent_frame(queries, trips, zones, distances, durations, stages):
    with stages.stage("parent_query_aggregate"):
        result = queries._zone_aggregates(
            trips,
            zones,
            distances,
            durations,
        )
        columns = _aggregate_device_columns(result)
    return columns


def _component_frame(
    queries,
    trips,
    components,
    distances,
    durations,
    parent_ids,
    *,
    parent_base: int,
    parent_rows: int,
    stages,
):
    import cupy as cp

    with stages.stage("component_query_aggregate"):
        result = queries._zone_aggregates(
            trips,
            components,
            distances,
            durations,
        )
        component_count, component_distance, component_duration = (
            _aggregate_device_columns(result)
        )
    with stages.stage("component_parent_attribute_reduce"):
        local_parent = parent_ids - cp.int64(parent_base)
        counts = cp.zeros(parent_rows, dtype=cp.uint64)
        distance = cp.zeros(parent_rows, dtype=cp.float64)
        duration = cp.zeros(parent_rows, dtype=cp.float64)
        cp.add.at(counts, local_parent, component_count)
        cp.add.at(distance, local_parent, component_distance)
        cp.add.at(duration, local_parent, component_duration)
    return counts, distance, duration


def _host_digest(frames, stages) -> dict[str, object]:
    import cupy as cp

    with stages.stage("terminal_parent_aggregate_export"):
        counts = cp.asnumpy(cp.concatenate([item[0] for item in frames]))
        distance = cp.asnumpy(cp.concatenate([item[1] for item in frames]))
        duration = cp.asnumpy(cp.concatenate([item[2] for item in frames]))
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(counts).tobytes())
    digest.update(np.ascontiguousarray(distance).tobytes())
    digest.update(np.ascontiguousarray(duration).tobytes())
    return {
        "sha256": digest.hexdigest(),
        "count_sum": int(counts.sum()),
        "distance_sum": float(distance.sum()),
        "duration_sum": float(duration.sum()),
        "counts": counts,
        "distance": distance,
        "duration": duration,
    }


def _identity(variant: str, zone_frames: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "complete_contemporaneous",
        "captured_at": datetime.now(UTC).isoformat(),
        "source": _source_tree_identity(Path(__file__).resolve()),
        "dataset": _dataset_identity(DATA),
        "environment": _environment_identity(),
        "measurement": {
            "query": "SF100 Q10 first 4M-row trip batch",
            "variant": variant,
            "zone_frames": zone_frames,
            "process_isolation": "one command/process per arm",
            "warmup_runs": 0,
            "repeat_runs": 1,
            "statistic": "single cold observation",
            "clock": "time.perf_counter wall time with synchronized stage events",
            "excluded_scope": "trip/zone IO and host component fixture decomposition",
            "included_scope": (
                "public query_aggregate, device component-to-parent grouped "
                "reduction, and terminal aggregate export"
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
    with tempfile.TemporaryDirectory(prefix="vs-r3-q10-components-") as raw_directory:
        zones_component = _component_frames(gpd, zones_parent, Path(raw_directory))
        trips = _first_trip_batch(queries, paths)
        trips["t_distance"] = trips["t_distance"].astype(float)
        durations = queries._duration_seconds(
            trips,
            "t_pickuptime",
            "t_dropofftime",
        )
        distances = trips["t_distance"]
        parent_count = sum(len(frame) for frame in zones_parent)
        component_ids = [
            _component_parent_ids(frame, parent_count=parent_count)
            for frame in zones_component
        ]
        clear_fallback_events()
        clear_materialization_events()
        vibespatial.clear_dispatch_events()
        cp.cuda.get_current_stream().synchronize()
        stages = _CudaStages()
        started = time.perf_counter()
        result_frames = []
        parent_base = 0
        for parents, components, ids in zip(
            zones_parent,
            zones_component,
            component_ids,
            strict=True,
        ):
            if variant == "parent":
                result = _parent_frame(
                    queries,
                    trips,
                    parents,
                    distances,
                    durations,
                    stages,
                )
            else:
                result = _component_frame(
                    queries,
                    trips,
                    components,
                    distances,
                    durations,
                    ids,
                    parent_base=parent_base,
                    parent_rows=len(parents),
                    stages=stages,
                )
            result_frames.append(result)
            parent_base += len(parents)
        digest = _host_digest(result_frames, stages)
        cp.cuda.get_current_stream().synchronize()
        wall_seconds = time.perf_counter() - started
        fallback_events = get_fallback_events(clear=True)
        materialization_events = get_materialization_events(clear=True)
        dispatch_events = vibespatial.get_dispatch_events(clear=True)
        trip_rows = len(trips)
        parent_zone_rows = sum(len(frame) for frame in zones_parent)
        component_zone_rows = sum(len(frame) for frame in zones_component)
        del (
            component_ids,
            distances,
            durations,
            result_frames,
            trips,
            zones_component,
            zones_parent,
        )
        gc.collect()
        cp.cuda.get_current_stream().synchronize()
    return {
        "schema_version": 1,
        "experiment": "q10_parent_aware_component",
        "variant": variant,
        "zone_frames": zone_frames,
        "trip_rows": trip_rows,
        "parent_zone_rows": parent_zone_rows,
        "component_zone_rows": component_zone_rows,
        "wall_seconds": wall_seconds,
        "stage_records": stages.records,
        "result": {
            "sha256": digest["sha256"],
            "count_sum": digest["count_sum"],
            "distance_sum": digest["distance_sum"],
            "duration_sum": digest["duration_sum"],
        },
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
        choices=("parent", "component-parent"),
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
