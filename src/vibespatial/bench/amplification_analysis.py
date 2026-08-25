"""Offline Level-0 work-amplification analysis for benchmark artifacts.

The analyzer deliberately operates on dictionaries and performs no file IO.  It
only promotes counters whose names state a physical meaning; public ``rows_in``
and ``rows_out`` fields are never used as candidate, relation, or fragment
counts.
"""

from __future__ import annotations

import math
import re
from copy import deepcopy
from typing import Any

from .amplification import Metric, Record, make_ratio

_DETAIL_COUNTER = re.compile(
    r"(?:^|[,;]\s*)(capacity|rows|parts|groups|bytes)\s*=\s*(\d+)(?=\s*(?:[,;]|$))"
)
_MEMORY_SIGNAL_BYTES = 64 * 1024 * 1024

_FAMILY_KEYS: dict[str, dict[str, tuple[str, str | None]]] = {
    "relation": {
        "pairs_examined": ("pairs_examined", "pairs"),
        "candidate_pairs": ("candidate_pairs", "pairs"),
        "refined_pairs": ("refined_pairs", "pairs"),
        "exact_pairs": ("exact_pairs", "pairs"),
        "relation_pairs": ("relation_pairs", "pairs"),
        "unique_left_rows": ("unique_left_rows", "rows"),
        "unique_right_rows": ("unique_right_rows", "rows"),
        "deduped_candidate_rows": ("deduped_candidate_rows", "rows"),
        "semijoin_rows": ("unique_left_rows", "rows"),
        "terminal_rows": ("terminal_rows", "rows"),
        "pair_bytes": ("pair_bytes", "bytes"),
        "relation_pair_bytes": ("pair_bytes", "bytes"),
        "terminal_bytes": ("terminal_bytes", "bytes"),
    },
    "constructive": {
        "source_segments": ("source_segments", "segments"),
        "split_events": ("split_events", "events"),
        "intersection_events": ("intersection_events", "events"),
        "emitted_fragments": ("emitted_fragments", "fragments"),
        "fragment_coordinates": ("fragment_coordinates", "coordinates"),
        "retained_output_parts": ("retained_output_parts", "parts"),
        "output_parts": ("retained_output_parts", "parts"),
        "output_coordinates": ("output_coordinates", "coordinates"),
        "constructive_bytes": ("constructive_bytes", "bytes"),
        "peak_live_bytes": ("peak_live_bytes", "bytes"),
        "output_bytes": ("output_bytes", "bytes"),
    },
    "capacity": {
        "capacity": ("capacity_slots", "slots"),
        "admitted_slots": ("admitted_slots", "slots"),
        "capacity_slots": ("capacity_slots", "slots"),
        "allocated_slots": ("capacity_slots", "slots"),
        "logical_slots": ("logical_slots", "slots"),
        "used_slots": ("logical_slots", "slots"),
        "admitted_bytes": ("admitted_bytes", "bytes"),
        "allocated_bytes": ("allocated_bytes", "bytes"),
        "logical_bytes": ("logical_bytes", "bytes"),
        "used_bytes": ("logical_bytes", "bytes"),
        # The pipeline profiler samples process/device-pool state at a stage
        # boundary. It is useful context, but it is not an attributable stage
        # peak and must never be promoted to ``peak_live_bytes``.
        "peak_device_memory_bytes": ("observed_device_memory_bytes", "bytes"),
        "peak_live_bytes": ("peak_live_bytes", "bytes"),
        "rmm_peak_allocation_bytes": ("peak_live_bytes", "bytes"),
        "rmm_total_allocation_bytes": ("total_allocated_bytes", "bytes"),
        "rmm_allocation_count": ("allocation_count", "allocations"),
        "largest_admitted_allocation_bytes": (
            "largest_admitted_allocation_bytes",
            "bytes",
        ),
        "peak_vram_bytes": ("observed_peak_vram_bytes", "bytes"),
        "pool_live_bytes": ("observed_pool_live_bytes", "bytes"),
        "allocation_fence_count": ("allocation_fence_count", "fences"),
    },
    "refinement": {
        "bbox_candidates": ("bbox_candidates", "candidates"),
        "candidate_parts": ("candidate_parts", "parts"),
        "candidate_lanes": ("candidate_lanes", "lanes"),
        "active_bins": ("active_bins", "bins"),
        "edge_evaluations": ("edge_evaluations", "evaluations"),
        "orientation_evaluations": ("orientation_evaluations", "evaluations"),
        "exact_evaluations": ("exact_evaluations", "evaluations"),
        "ambiguous_lanes": ("ambiguous_lanes", "lanes"),
        "exact_refinements": ("exact_refinements", "refinements"),
        "survivors": ("survivors", "rows"),
        "early_terminated": ("early_terminated", "lanes"),
        "work_p50": ("work_p50", "evaluations"),
        "work_p95": ("work_p95", "evaluations"),
        "work_p99": ("work_p99", "evaluations"),
        "work_max": ("work_max", "evaluations"),
    },
    "rebuild": {
        "build_count": ("build_count", "builds"),
        "index_build_count": ("build_count", "builds"),
        "build_seconds": ("build_seconds", "seconds"),
        "build_time_seconds": ("build_seconds", "seconds"),
        "native_index_build_seconds": ("build_seconds", "seconds"),
        "persistent_bytes": ("persistent_bytes", "bytes"),
        "cache_hits": ("cache_hits", "hits"),
        "cache_misses": ("cache_misses", "misses"),
        "consumer_count": ("consumer_count", "consumers"),
        "avoidable_rebuild_seconds": ("avoidable_rebuild_seconds", "seconds"),
    },
    "group_compression": {
        "input_rows": ("input_rows", "rows"),
        "group_count": ("output_groups", "groups"),
        "output_groups": ("output_groups", "groups"),
        "max_group_size": ("max_group_size", "rows"),
        "input_segments": ("input_segments", "segments"),
        "input_coordinates": ("input_coordinates", "coordinates"),
        "pre_reduction_fragments": ("pre_reduction_fragments", "fragments"),
        "pre_reduction_coordinates": ("pre_reduction_coordinates", "coordinates"),
        "output_parts": ("output_parts", "parts"),
        "output_coordinates": ("output_coordinates", "coordinates"),
        "group_size_p50": ("group_size_p50", "rows"),
        "group_size_p95": ("group_size_p95", "rows"),
        "group_size_p99": ("group_size_p99", "rows"),
    },
}

_PHYSICAL_SHAPES = {
    "relation": "relation_pairs",
    "constructive": "constructive_fragments",
    "capacity": "capacity",
    "refinement": "refinement_work",
    "rebuild": "reusable_preparation",
    "group_compression": "grouped_work",
    "timing": "timed_stage",
}

_NEEDED_METRICS = {
    "relation": ("candidate_pairs", "refined_pairs", "relation_pairs", "terminal_rows", "pair_bytes", "terminal_bytes"),
    "constructive": ("split_events", "fragment_coordinates", "output_coordinates", "peak_live_bytes", "output_bytes"),
    "capacity": ("capacity_slots", "logical_slots", "allocated_bytes", "logical_bytes", "peak_live_bytes"),
    "refinement": ("candidate_lanes", "exact_evaluations", "ambiguous_lanes", "survivors", "early_terminated"),
    "rebuild": ("build_count", "build_seconds", "cache_hits", "cache_misses", "consumer_count", "persistent_bytes"),
    "group_compression": ("input_rows", "output_groups", "pre_reduction_coordinates", "output_coordinates"),
}


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value < 0:
        return None
    return value


def _metric(name: str, value: int | float, unit: str | None, source: str) -> dict[str, Any]:
    return Metric(
        name=name,
        value=value,
        status="exact",
        source=source,
        unit=unit,
    ).to_dict()


def _unavailable_metric(name: str, *, source: str) -> dict[str, Any]:
    return Metric(
        name=name,
        value=None,
        status="unavailable",
        reason=f"not available at Level 0 ({source})",
    ).to_dict()


def _ratio(name: str, numerator: dict[str, Any], denominator: dict[str, Any]) -> dict[str, Any]:
    return make_ratio(
        name,
        numerator_metric=numerator["name"],
        numerator_value=numerator["value"],
        denominator_metric=denominator["name"],
        denominator_value=denominator["value"],
    ).to_dict()


def _metric_by_name(metrics: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((metric for metric in metrics if metric["name"] == name), None)


def _derived_ratios(family: str, metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: tuple[tuple[str, str, str], ...]
    if family == "relation":
        pairs = (
            ("coarse_to_exact", "candidate_pairs", "refined_pairs"),
            ("relation_to_terminal", "relation_pairs", "terminal_rows"),
            ("relation_byte_amplification", "pair_bytes", "terminal_bytes"),
        )
    elif family == "constructive":
        pairs = (
            ("event_to_output", "split_events", "output_coordinates"),
            ("fragment_to_output", "fragment_coordinates", "output_coordinates"),
            ("peak_live_to_output", "peak_live_bytes", "output_bytes"),
        )
    elif family == "capacity":
        pairs = (
            ("slot_utilization", "logical_slots", "capacity_slots"),
            ("byte_utilization", "logical_bytes", "allocated_bytes"),
        )
    elif family == "refinement":
        pairs = (
            ("exact_work_per_survivor", "exact_evaluations", "survivors"),
            (
                "candidate_parts_per_lane",
                "candidate_parts_considered",
                "candidate_lanes",
            ),
            ("edge_visits_per_lane", "edge_visits", "candidate_lanes"),
            ("edge_visits_per_survivor", "edge_visits", "survivors"),
            ("ambiguity_fraction", "ambiguous_lanes", "candidate_lanes"),
            ("early_exit_fraction", "early_terminated", "candidate_lanes"),
        )
    elif family == "rebuild":
        pairs = (("consumer_reuse", "consumer_count", "build_count"),)
    elif family == "group_compression":
        pairs = (
            ("rows_per_output_group", "input_rows", "output_groups"),
            ("group_geometry_amplification", "pre_reduction_coordinates", "output_coordinates"),
        )
    else:
        pairs = ()
    ratios = []
    for name, numerator_name, denominator_name in pairs:
        numerator = _metric_by_name(metrics, numerator_name)
        denominator = _metric_by_name(metrics, denominator_name)
        if (
            numerator is not None
            and denominator is not None
            and numerator.get("status") != "unavailable"
            and denominator.get("status") != "unavailable"
            and _number(numerator.get("value")) is not None
            and _number(denominator.get("value")) is not None
        ):
            ratios.append(_ratio(name, numerator, denominator))
    return ratios


def _record(
    *,
    operation: str,
    stage: str,
    family: str,
    device: str,
    measurement_boundary: str,
    boundary_role: str,
    metrics: list[dict[str, Any]],
    semantic_contract: dict[str, Any] | None = None,
    physical_shape: str | None = None,
    consumer_kind: str = "unspecified",
    instrumentation_level: int = 0,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "operation": operation,
        "stage": stage,
        "metric_family": family,
        "physical_shape": physical_shape or _PHYSICAL_SHAPES[family],
        "source_lineage": [{"alias": "source_0", "role": "artifact_stage"}],
        "consumer_kind": consumer_kind,
        "device": device,
        "measurement_boundary": measurement_boundary,
        "boundary_role": boundary_role,
        "instrumentation_level": instrumentation_level,
        "metrics": metrics,
        "ratios": _derived_ratios(family, metrics),
        "semantic_contract": {
            "lineage_status": "unavailable",
            **(semantic_contract or {}),
        },
    }
    # Keep this offline analyzer coupled to the authoritative versioned schema,
    # including zero-denominator representation and JSON-native serialization.
    return Record.from_dict(payload).to_dict()


def _parse_detail_counters(detail: Any) -> dict[str, int]:
    if isinstance(detail, dict):
        result = {}
        for key in ("capacity", "rows", "parts", "groups", "bytes"):
            value = _number(detail.get(key))
            if value is not None and value >= 0 and float(value).is_integer():
                result[key] = int(value)
        return result
    if not isinstance(detail, str):
        return {}
    return {match.group(1): int(match.group(2)) for match in _DETAIL_COUNTER.finditer(detail)}


def _records_for_stage(
    stage_payload: dict[str, Any],
    metadata: dict[str, Any],
    *,
    operation: str,
    stage_name: str,
    elapsed_seconds: int | float | None,
    device: str,
    measurement_boundary: str,
    boundary_role: str,
) -> list[dict[str, Any]]:
    elapsed = _number(elapsed_seconds)
    records: list[dict[str, Any]] = []
    packet = metadata.get("work_amplification")
    if isinstance(packet, dict):
        version = packet.get("schema_version")
        if version != 1:
            raise ValueError(
                f"unsupported hotpath work-amplification packet schema_version {version!r}"
            )
        family = packet.get("metric_family")
        if family not in _PHYSICAL_SHAPES or family == "timing":
            raise ValueError(f"invalid hotpath work-amplification metric_family {family!r}")
        packet_metrics: list[dict[str, Any]] = []
        seen_packet_metrics: set[str] = set()
        for reducer in ("sum", "max"):
            values = packet.get(reducer, {})
            if not isinstance(values, dict):
                raise TypeError(f"work-amplification packet {reducer} must be an object")
            for name, value in values.items():
                normalized = _number(value)
                if not isinstance(name, str) or normalized is None:
                    raise ValueError(
                        f"invalid work-amplification packet {reducer} metric {name!r}"
                    )
                if name in seen_packet_metrics:
                    raise ValueError(
                        f"duplicate work-amplification packet metric {name!r}"
                    )
                seen_packet_metrics.add(name)
                packet_metrics.append(
                    Metric(
                        name=name,
                        value=normalized,
                        status="exact",
                        source=f"metadata.work_amplification.{reducer}.{name}",
                        metadata={"reducer": reducer},
                    ).to_dict()
                )
        unavailable = packet.get("unavailable", [])
        if not isinstance(unavailable, list):
            raise TypeError("work-amplification packet unavailable must be an array")
        for name in unavailable:
            if not isinstance(name, str) or not name or name in seen_packet_metrics:
                continue
            seen_packet_metrics.add(name)
            packet_metrics.append(
                _unavailable_metric(
                    name,
                    source="metadata.work_amplification.unavailable",
                )
            )
        if elapsed is not None:
            packet_metrics.append(
                _metric("stage_elapsed_seconds", elapsed, "seconds", "stage.elapsed_seconds")
            )
        if packet_metrics:
            semantic_contract = packet.get("semantic_contract", {})
            if not isinstance(semantic_contract, dict):
                raise TypeError(
                    "work-amplification packet semantic_contract must be an object"
                )
            instrumentation_level = packet.get("instrumentation_level", 0)
            if isinstance(instrumentation_level, bool) or not isinstance(
                instrumentation_level, int
            ):
                raise TypeError(
                    "work-amplification packet instrumentation_level must be an integer"
                )
            records.append(
                _record(
                    operation=str(packet.get("operation") or operation),
                    stage=stage_name,
                    family=family,
                    device=str(packet.get("device") or device),
                    measurement_boundary=str(
                        packet.get("measurement_boundary") or measurement_boundary
                    ),
                    boundary_role=str(packet.get("boundary_role") or boundary_role),
                    metrics=packet_metrics,
                    semantic_contract={
                        "counter_packet": True,
                        "mixed_context": bool(packet.get("mixed_context", False)),
                        **semantic_contract,
                    },
                    physical_shape=(
                        str(packet["physical_shape"])
                        if packet.get("physical_shape")
                        else None
                    ),
                    consumer_kind=str(packet.get("consumer_kind") or "unspecified"),
                    instrumentation_level=instrumentation_level,
                )
            )
    for family, aliases in _FAMILY_KEYS.items():
        metrics = []
        seen = set()
        for source_name, (name, unit) in aliases.items():
            value = _number(metadata.get(source_name))
            if value is None or name in seen:
                continue
            seen.add(name)
            metrics.append(_metric(name, value, unit, f"metadata.{source_name}"))
        if family == "rebuild" and isinstance(metadata.get("cache_hit"), bool):
            cache_hit = bool(metadata["cache_hit"])
            seen.update(("cache_hits", "cache_misses"))
            metrics.extend(
                (
                    _metric("cache_hits", int(cache_hit), "hits", "metadata.cache_hit"),
                    _metric("cache_misses", int(not cache_hit), "misses", "metadata.cache_hit"),
                )
            )
        if family == "rebuild":
            for container_name in ("build", "cache"):
                container = metadata.get(container_name)
                if not isinstance(container, dict):
                    continue
                for source_name, (name, unit) in aliases.items():
                    value = _number(container.get(source_name))
                    if value is None or name in seen:
                        continue
                    seen.add(name)
                    metrics.append(
                        _metric(
                            name,
                            value,
                            unit,
                            f"metadata.{container_name}.{source_name}",
                        )
                    )
        if not metrics:
            continue
        if elapsed is not None:
            metrics.append(_metric("stage_elapsed_seconds", elapsed, "seconds", "stage.elapsed_seconds"))
        records.append(
            _record(
                operation=operation,
                stage=stage_name,
                family=family,
                device=device,
                measurement_boundary=measurement_boundary,
                boundary_role=boundary_role,
                metrics=metrics,
                semantic_contract={"level0_explicit_keys_only": True},
            )
        )

    events = metadata.get("materialization_events", stage_payload.get("materialization_events", ()))
    if isinstance(events, list):
        for event_index, event in enumerate(events):
            if not isinstance(event, dict):
                continue
            counters = _parse_detail_counters(event.get("detail"))
            if not counters:
                continue
            # A capacity and its same-event logical row count form a valid
            # utilization observation. Rows/parts/groups or bytes alone do
            # not state amplification. In particular, materialization-event
            # ``bytes`` commonly describes the source geometry carrier's
            # nbytes; it is not a transfer, allocation, or peak-live measure.
            if "capacity" not in counters:
                continue
            metrics = [
                _metric(
                    "capacity_slots",
                    counters["capacity"],
                    "slots",
                    f"materialization_events[{event_index}].detail.capacity",
                )
            ]
            if "rows" in counters:
                metrics.append(
                    _metric(
                        "logical_slots",
                        counters["rows"],
                        "slots",
                        f"materialization_events[{event_index}].detail.rows",
                    )
                )
            if "bytes" in counters:
                metrics.append(
                    _metric(
                        "reported_boundary_bytes",
                        counters["bytes"],
                        "bytes",
                        f"materialization_events[{event_index}].detail.bytes",
                    )
                )
            if elapsed is not None:
                metrics.append(_metric("stage_elapsed_seconds", elapsed, "seconds", "stage.elapsed_seconds"))
            records.append(
                _record(
                    operation=operation,
                    stage=stage_name,
                    family="capacity",
                    device=device,
                    measurement_boundary=measurement_boundary,
                    boundary_role=str(event.get("boundary") or boundary_role),
                    metrics=metrics,
                    semantic_contract={
                        "event_index": event_index,
                        "event_boundary": event.get("boundary", "unspecified"),
                        "same_event_pairing_only": True,
                        "reported_boundary_bytes_are_not_peak_or_transfer_bytes": True,
                    },
                )
            )

    if not records and elapsed is not None:
        records.append(
            _record(
                operation=operation,
                stage=stage_name,
                family="timing",
                device=device,
                measurement_boundary=measurement_boundary,
                boundary_role=boundary_role,
                metrics=[_metric("stage_elapsed_seconds", elapsed, "seconds", "stage.elapsed_seconds")],
                semantic_contract={"physical_metrics": "unavailable"},
            )
        )
    return records


def _attach_stage_records(
    stage: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    pipeline: bool,
) -> None:
    if pipeline:
        metadata = stage.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata["work_amplification"] = records
    else:
        stage["work_amplification"] = records


def _walk_pipeline(artifact: dict[str, Any], flat: list[dict[str, Any]]) -> bool:
    found = False
    for result in artifact.get("results", ()):
        if not isinstance(result, dict) or not isinstance(result.get("stages"), list):
            continue
        workflow_wall = _number(result.get("elapsed_seconds"))
        for trace_wrapper in result["stages"]:
            if not isinstance(trace_wrapper, dict):
                continue
            trace = trace_wrapper.get("trace", trace_wrapper)
            if not isinstance(trace, dict) or not isinstance(trace.get("stages"), list):
                continue
            found = True
            operation = str(trace.get("operation") or result.get("pipeline") or "unknown")
            if workflow_wall is None:
                workflow_wall = _number(trace.get("total_elapsed_seconds"))
            for stage in trace["stages"]:
                if not isinstance(stage, dict):
                    continue
                metadata = stage.get("metadata") if isinstance(stage.get("metadata"), dict) else {}
                stage_records = _records_for_stage(
                    stage,
                    metadata,
                    operation=operation,
                    stage_name=str(stage.get("name", "unknown")),
                    elapsed_seconds=stage.get("elapsed_seconds"),
                    device=str(stage.get("device", "unknown")),
                    measurement_boundary="pipeline_trace",
                    boundary_role=str(metadata.get("profile_boundary", "unspecified")),
                )
                for record in stage_records:
                    record["semantic_contract"]["workflow_wall_seconds"] = workflow_wall
                _attach_stage_records(stage, stage_records, pipeline=True)
                flat.extend(stage_records)
    return found


def _timing_seconds(vibespatial: dict[str, Any]) -> int | float | None:
    timing = vibespatial.get("timing")
    if not isinstance(timing, dict):
        return None
    for key in ("median_seconds", "mean_seconds", "min_seconds"):
        value = _number(timing.get(key))
        if value is not None:
            return value
    return None


def _shootout_results(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(artifact.get("vibespatial"), dict):
        return [artifact]
    return [item for item in artifact.get("results", ()) if isinstance(item, dict) and isinstance(item.get("vibespatial"), dict)]


def _walk_shootout(
    artifact: dict[str, Any],
    flat: list[dict[str, Any]],
    observer_effects: list[dict[str, Any]],
) -> bool:
    results = _shootout_results(artifact)
    for result in results:
        vibespatial = result["vibespatial"]
        profile = vibespatial.get("profile")
        if not isinstance(profile, dict):
            continue
        operation = str(result.get("script") or artifact.get("script") or "shootout")
        profile_wall = _number(profile.get("elapsed_seconds"))
        timed_wall = _timing_seconds(vibespatial)
        observer = None
        if profile_wall is not None and timed_wall is not None:
            ratio = None if timed_wall == 0 else profile_wall / timed_wall
            gross = ratio is not None and ratio > 2.0
            observer = {
                "operation": operation,
                "timed_execution_seconds": timed_wall,
                "post_timing_profile_seconds": profile_wall,
                "post_profile_to_timed_ratio": ratio,
                "execution_scopes": ["timed_execution", "post_timing_profile"],
                "wall_attribution": "unreliable" if gross else "usable_with_observer_caveat",
                "is_workload_amplification": False,
                "gross_perturbation_guard": "research reliability check only; not production policy",
            }
            if gross:
                observer["needed_metrics"] = ["corrected_counterfactual_profile"]
            observer_effects.append(observer)
            profile["work_amplification_observer_effect"] = observer

        for collection_name in ("timed_stages", "hotpath"):
            stages = profile.get(collection_name)
            if not isinstance(stages, list):
                continue
            if (
                collection_name == "hotpath"
                and profile.get("profile_mode") == "counters"
                and bool(profile.get("timed_stages"))
            ):
                # Counter packets are attached to their enclosing statements;
                # the global summary is a convenience roll-up of the same
                # evidence and would otherwise duplicate every metric.
                continue
            for stage in stages:
                if not isinstance(stage, dict):
                    continue
                if collection_name == "timed_stages":
                    stage_name = str(stage.get("name") or f"statement_{stage.get('index', 'unknown')}")
                    metadata = stage
                    device = str(stage.get("actual_backend", "unknown"))
                    boundary = "shootout_post_timing_statement"
                else:
                    stage_name = str(stage.get("name", "unknown"))
                    nested_metadata = stage.get("metadata")
                    metadata = dict(stage)
                    if isinstance(nested_metadata, dict):
                        metadata.update(nested_metadata)
                    device = str(stage.get("device", "unknown"))
                    boundary = "shootout_post_timing_hotpath"
                stage_records = _records_for_stage(
                    stage,
                    metadata,
                    operation=operation,
                    stage_name=stage_name,
                    elapsed_seconds=(
                        None
                        if collection_name == "hotpath"
                        and stage.get("timing_mode") == "counter"
                        else stage.get("elapsed_seconds")
                    ),
                    device=device,
                    measurement_boundary=boundary,
                    boundary_role="compute",
                )
                for record in stage_records:
                    record["semantic_contract"]["workflow_wall_seconds"] = profile_wall
                    record["semantic_contract"]["workflow_operation"] = operation
                    record["semantic_contract"]["hotpath_timers_may_overlap"] = collection_name == "hotpath"
                    if observer is not None:
                        record["semantic_contract"]["observer_wall_attribution"] = observer["wall_attribution"]
                _attach_stage_records(stage, stage_records, pipeline=False)
                flat.extend(stage_records)
                if (
                    collection_name == "timed_stages"
                    and profile.get("profile_mode") == "counters"
                    and isinstance(stage.get("hotpath"), list)
                ):
                    for nested_stage in stage["hotpath"]:
                        if not isinstance(nested_stage, dict):
                            continue
                        nested_metadata = nested_stage.get("metadata")
                        nested_payload = dict(nested_stage)
                        if isinstance(nested_metadata, dict):
                            nested_payload.update(nested_metadata)
                        nested_records = _records_for_stage(
                            nested_stage,
                            nested_payload,
                            operation=operation,
                            stage_name=str(nested_stage.get("name", "unknown")),
                            elapsed_seconds=stage.get("elapsed_seconds"),
                            device=device,
                            measurement_boundary=(
                                "shootout_counter_enclosing_statement"
                            ),
                            boundary_role="compute",
                        )
                        for nested_record in nested_records:
                            nested_record["semantic_contract"][
                                "workflow_operation"
                            ] = operation
                            nested_record["semantic_contract"][
                                "workflow_wall_seconds"
                            ] = profile_wall
                            nested_record["semantic_contract"][
                                "elapsed_scope"
                            ] = "enclosing_statement"
                            nested_record["semantic_contract"][
                                "nested_counter_timing_unavailable"
                            ] = True
                            if observer is not None:
                                nested_record["semantic_contract"][
                                    "observer_wall_attribution"
                                ] = observer["wall_attribution"]
                        _attach_stage_records(
                            nested_stage,
                            nested_records,
                            pipeline=False,
                        )
                        flat.extend(nested_records)
    return bool(results)


def _attach_sf100_query(
    query: dict[str, Any],
    *,
    engine: str,
    flat: list[dict[str, Any]],
) -> bool:
    if "query" not in query:
        return False
    elapsed_key = "time_seconds" if "time_seconds" in query else "seconds"
    elapsed = _number(query.get(elapsed_key))
    if elapsed is None:
        return False
    metrics = [
        _metric(
            "stage_elapsed_seconds",
            elapsed,
            "seconds",
            f"query.{elapsed_key}",
        )
    ]
    output_key = "row_count" if "row_count" in query else "rows"
    output_rows = _number(query.get(output_key))
    if output_rows is not None:
        metrics.append(
            _metric(
                "public_output_rows",
                output_rows,
                "rows",
                f"query.{output_key}",
            )
        )
    record = _record(
        operation=str(query["query"]),
        stage=str(query["query"]),
        family="timing",
        device=engine,
        measurement_boundary="sf100_query_timing",
        boundary_role="workflow",
        metrics=metrics,
        semantic_contract={
            "public_output_rows_are_not_physical_work": True,
            "physical_metrics": "unavailable",
            "workflow_wall_seconds": elapsed,
        },
    )
    query_records = [record]
    flat.append(record)
    telemetry_runs = query.get("telemetry_runs", [])
    if isinstance(telemetry_runs, list):
        for run_index, telemetry in enumerate(telemetry_runs):
            if not isinstance(telemetry, dict):
                continue
            telemetry_records = _records_for_stage(
                telemetry,
                telemetry,
                operation=str(query["query"]),
                stage_name=f"{query['query']}.telemetry_run_{run_index + 1}",
                elapsed_seconds=elapsed,
                device=engine,
                measurement_boundary="sf100_query_telemetry",
                boundary_role="workflow",
            )
            for telemetry_record in telemetry_records:
                telemetry_record["semantic_contract"]["workflow_wall_seconds"] = elapsed
                telemetry_record["semantic_contract"][
                    "telemetry_wall_includes_observer_overhead"
                ] = True
            query_records.extend(telemetry_records)
            flat.extend(telemetry_records)
            point_region = telemetry.get("point_region_profile")
            if isinstance(point_region, dict):
                profile_wall = _number(point_region.get("profile_wall_seconds"))
                groups = point_region.get("groups", [])
                if isinstance(groups, list):
                    for group_index, group in enumerate(groups):
                        if not isinstance(group, dict):
                            continue
                        counters = group.get("counters", {})
                        if not isinstance(counters, dict):
                            counters = {}
                        profile_metrics: list[dict[str, Any]] = []
                        metric_sources = {
                            "candidates": ("candidate_lanes", "lanes"),
                            "valid_candidates": ("valid_candidate_lanes", "lanes"),
                            "parts_considered": ("candidate_parts_considered", "parts"),
                            "active_parts": ("active_candidate_parts", "parts"),
                            "edges_visited": ("edge_visits", "edges"),
                            "orient2d_calls": ("exact_evaluations", "evaluations"),
                            "zero_active_candidates": ("early_terminated", "lanes"),
                            "boundary_results": ("boundary_results", "results"),
                            "interior_results": ("survivors", "results"),
                            "exterior_results": ("exterior_results", "results"),
                            "max_parts_considered": ("max_parts_considered", "parts"),
                            "max_active_parts": ("max_active_parts", "parts"),
                            "max_edges_visited": ("max_edges_visited", "edges"),
                        }
                        for source_name, (metric_name, unit) in metric_sources.items():
                            value = _number(counters.get(source_name))
                            if value is not None:
                                profile_metrics.append(
                                    _metric(
                                        metric_name,
                                        value,
                                        unit,
                                        f"point_region_profile.groups[{group_index}].counters.{source_name}",
                                    )
                                )
                        kernel_seconds = _number(group.get("kernel_seconds"))
                        if kernel_seconds is not None:
                            profile_metrics.append(
                                _metric(
                                    "stage_elapsed_seconds",
                                    kernel_seconds,
                                    "seconds",
                                    f"point_region_profile.groups[{group_index}].kernel_seconds",
                                )
                            )
                        if profile_metrics:
                            profile_record = _record(
                                operation="point_region_profile",
                                stage=f"{query['query']}.point_region_group_{group_index + 1}",
                                family="refinement",
                                device=engine,
                                measurement_boundary="sf100_point_region_profile",
                                boundary_role="compute",
                                metrics=profile_metrics,
                                physical_shape="prepared_part_y_index_candidate_lanes",
                                consumer_kind="exact_point_region_classification",
                                instrumentation_level=1,
                                semantic_contract={
                                    "workflow_operation": str(query["query"]),
                                    "workflow_wall_seconds": (
                                        profile_wall if profile_wall is not None else elapsed
                                    ),
                                    "profile_wall_includes_observer_overhead": True,
                                    "family": group.get("family"),
                                    "geometry_count": group.get("geometry_count"),
                                    "part_count": group.get("part_count"),
                                    "edge_membership_count": group.get(
                                        "edge_membership_count"
                                    ),
                                    "sample_limit": point_region.get("sample_limit"),
                                    "samples_per_launch": point_region.get(
                                        "samples_per_launch"
                                    ),
                                },
                            )
                            query_records.append(profile_record)
                            flat.append(profile_record)
                preparations = point_region.get("index_preparation", [])
                if isinstance(preparations, list):
                    for prep_index, preparation in enumerate(preparations):
                        if not isinstance(preparation, dict):
                            continue
                        preparation_metrics: list[dict[str, Any]] = []
                        for source_name, unit in (
                            ("build_count", "builds"),
                            ("cache_hits", "hits"),
                            ("build_wall_seconds", "seconds"),
                        ):
                            value = _number(preparation.get(source_name))
                            if value is not None:
                                metric_name = (
                                    "build_seconds"
                                    if source_name == "build_wall_seconds"
                                    else source_name
                                )
                                preparation_metrics.append(
                                    _metric(
                                        metric_name,
                                        value,
                                        unit,
                                        f"point_region_profile.index_preparation[{prep_index}].{source_name}",
                                    )
                                )
                        build_count = _number(preparation.get("build_count"))
                        cache_hits = _number(preparation.get("cache_hits"))
                        if build_count is not None and cache_hits is not None:
                            preparation_metrics.append(
                                _metric(
                                    "consumer_count",
                                    build_count + cache_hits,
                                    "consumers",
                                    f"point_region_profile.index_preparation[{prep_index}].build_count+cache_hits",
                                )
                            )
                        if preparation_metrics:
                            preparation_record = _record(
                                operation="prepare_point_region_y_index",
                                stage=f"{query['query']}.point_region_prepare_{prep_index + 1}",
                                family="rebuild",
                                device=engine,
                                measurement_boundary="sf100_point_region_profile",
                                boundary_role="setup",
                                metrics=preparation_metrics,
                                physical_shape="reusable_polygon_part_y_index",
                                consumer_kind="point_region_exact_refinement",
                                instrumentation_level=1,
                                semantic_contract={
                                    "workflow_operation": str(query["query"]),
                                    "workflow_wall_seconds": (
                                        profile_wall if profile_wall is not None else elapsed
                                    ),
                                    "profile_wall_includes_observer_overhead": True,
                                    "family": preparation.get("family"),
                                },
                            )
                            query_records.append(preparation_record)
                            flat.append(preparation_record)
            hotpaths = telemetry.get("hotpath")
            if not isinstance(hotpaths, list):
                continue
            for hotpath in hotpaths:
                if not isinstance(hotpath, dict):
                    continue
                nested_metadata = hotpath.get("metadata")
                metadata = dict(hotpath)
                if isinstance(nested_metadata, dict):
                    metadata.update(nested_metadata)
                hotpath_records = _records_for_stage(
                    hotpath,
                    metadata,
                    operation=str(query["query"]),
                    stage_name=str(hotpath.get("name", "unknown")),
                    elapsed_seconds=(
                        None
                        if hotpath.get("timing_mode") == "counter"
                        else hotpath.get("elapsed_seconds")
                    ),
                    device=engine,
                    measurement_boundary="sf100_hotpath_counter",
                    boundary_role="compute",
                )
                for hotpath_record in hotpath_records:
                    hotpath_record["semantic_contract"][
                        "workflow_wall_seconds"
                    ] = elapsed
                    hotpath_record["semantic_contract"][
                        "hotpath_timers_may_overlap"
                    ] = hotpath.get("timing_mode") != "counter"
                query_records.extend(hotpath_records)
                flat.extend(hotpath_records)
    query["work_amplification"] = query_records
    return True


def _walk_sf100(artifact: dict[str, Any], flat: list[dict[str, Any]]) -> bool:
    found = False
    results = artifact.get("results", ())
    if not isinstance(results, list):
        return False
    for item in results:
        if not isinstance(item, dict):
            continue
        if "query" in item:
            found |= _attach_sf100_query(
                item,
                engine=str(artifact.get("engine", "unknown")),
                flat=flat,
            )
            continue
        if not isinstance(item.get("results"), list):
            continue
        if "scale_factor" not in item and "engine" not in item:
            continue
        for query in item["results"]:
            if isinstance(query, dict):
                found |= _attach_sf100_query(
                    query,
                    engine=str(item.get("engine", "unknown")),
                    flat=flat,
                )
    return found


def _record_elapsed(record: dict[str, Any]) -> float:
    metric = _metric_by_name(record["metrics"], "stage_elapsed_seconds")
    return float(metric["value"]) if metric is not None else 0.0


def _substantive_signal(record: dict[str, Any]) -> tuple[list[str], float]:
    family = record["metric_family"]
    why = []
    magnitude = 0.0
    for ratio in record["ratios"]:
        value = ratio.get("value")
        if value is None:
            continue
        name = ratio["name"]
        if name in {"slot_utilization", "byte_utilization"} and value <= 0.5:
            why.append(f"{name}={value:.3g}")
            magnitude = max(magnitude, 1.0 / max(value, 1e-12))
        elif name not in {"ambiguity_fraction", "early_exit_fraction", "consumer_reuse"} and value >= 2.0:
            why.append(f"{name}={value:.3g}")
            magnitude = max(magnitude, float(value))
    raw_metrics = {metric["name"]: metric["value"] for metric in record["metrics"]}

    def metric_value(name: str) -> int | float:
        value = _number(raw_metrics.get(name))
        return 0 if value is None else value

    build_count = metric_value("build_count")
    if family == "rebuild" and build_count > 1:
        why.append(f"build_count={build_count}")
        magnitude = max(magnitude, float(build_count))
    memory = max(
        (
            float(metric_value(name))
            for name in (
                "pair_bytes",
                "peak_live_bytes",
                "allocated_bytes",
                "persistent_bytes",
            )
        ),
        default=0.0,
    )
    if memory >= _MEMORY_SIGNAL_BYTES:
        why.append(f"material_memory_bytes={int(memory)}")
        magnitude = max(magnitude, memory / _MEMORY_SIGNAL_BYTES)
    allocation_fence_count = metric_value("allocation_fence_count")
    if allocation_fence_count > 1:
        why.append(f"allocation_fence_count={allocation_fence_count}")
        magnitude = max(magnitude, float(allocation_fence_count))
    return why, magnitude


def _rank_findings(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings = []
    for record_index, record in enumerate(records):
        if record["metric_family"] == "timing":
            continue
        if record["semantic_contract"].get("mixed_context"):
            continue
        elapsed = _record_elapsed(record)
        wall = _number(record["semantic_contract"].get("workflow_wall_seconds"))
        wall_fraction = elapsed / wall if wall not in (None, 0) else None
        if elapsed <= 1.0 and (wall_fraction is None or wall_fraction < 0.05):
            continue
        why, magnitude = _substantive_signal(record)
        if not why:
            continue
        present = {metric["name"] for metric in record["metrics"]}
        needed = [name for name in _NEEDED_METRICS.get(record["metric_family"], ()) if name not in present]
        if record["semantic_contract"].get("observer_wall_attribution") == "unreliable":
            needed.append("corrected_counterfactual_profile")
            why.append("post-timing profiling materially perturbed wall time")
        findings.append(
            {
                "record_index": record_index,
                "operation": record["operation"],
                "stage": record["stage"],
                "metric_family": record["metric_family"],
                "stage_elapsed_seconds": elapsed,
                "workflow_wall_fraction": wall_fraction,
                "why": why,
                "needed_metrics": needed,
                "priority_score": elapsed * max(magnitude, 1.0),
            }
        )
    findings.sort(key=lambda finding: (-finding["priority_score"], -finding["stage_elapsed_seconds"], finding["operation"], finding["stage"]))
    for rank, finding in enumerate(findings, 1):
        finding["rank"] = rank
    return findings


def analyze_work_amplification(artifact: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy and enrich a benchmark artifact with Level-0 evidence.

    The returned envelope contains ``artifact``, flattened ``records``, ranked
    ``findings``, and separate shootout ``observer_effects``.  The input is not
    mutated.
    """

    if not isinstance(artifact, dict):
        raise TypeError("artifact must be a dictionary")
    enriched = deepcopy(artifact)
    records: list[dict[str, Any]] = []
    observer_effects: list[dict[str, Any]] = []
    _walk_pipeline(enriched, records)
    _walk_shootout(enriched, records, observer_effects)
    _walk_sf100(enriched, records)
    return {
        "artifact": enriched,
        "records": records,
        "findings": _rank_findings(records),
        "observer_effects": observer_effects,
    }


analyze_artifact = analyze_work_amplification


__all__ = ["analyze_artifact", "analyze_work_amplification"]
