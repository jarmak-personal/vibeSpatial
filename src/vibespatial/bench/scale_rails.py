from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScaleRailObservation:
    """One post-warmup observation for a monotonically increasing input tier."""

    name: str
    input_rows: int
    selected_rows: int
    batch_count: int
    elapsed_seconds: float
    allocation_bytes: int
    allocation_events: int
    materializations: int
    synchronizations: int
    terminal_d2h_bytes: int
    fallback_count: int = 0
    prepared_builds: int = 0
    prepared_reuses: int = 0
    prepared_declines: int = 0
    exact_refinement_work: int = 0


@dataclass(frozen=True)
class ScaleRailViolation:
    metric: str
    previous_tier: str
    current_tier: str
    previous_normalized: float
    current_normalized: float


@dataclass(frozen=True)
class ScaleRailResult:
    passed: bool
    observations: tuple[ScaleRailObservation, ...]
    violations: tuple[ScaleRailViolation, ...]


def _normalized_metrics(observation: ScaleRailObservation) -> dict[str, float]:
    rows = max(int(observation.input_rows), 1)
    selected_rows = max(int(observation.selected_rows), 1)
    batches = max(int(observation.batch_count), 1)
    return {
        "seconds_per_input_row": float(observation.elapsed_seconds) / rows,
        "allocation_bytes_per_input_row": float(observation.allocation_bytes) / rows,
        "allocation_events_per_input_row": float(observation.allocation_events) / rows,
        "materializations_per_batch": float(observation.materializations) / batches,
        "synchronizations_per_batch": float(observation.synchronizations) / batches,
        "terminal_d2h_bytes_per_selected_row": (
            float(observation.terminal_d2h_bytes) / selected_rows
        ),
        "prepared_declines_per_batch": float(observation.prepared_declines) / batches,
        "prepared_builds_per_batch": float(observation.prepared_builds) / batches,
        "exact_refinement_work_per_selected_row": (
            float(observation.exact_refinement_work) / selected_rows
        ),
    }


def evaluate_scale_rail(
    observations,
    *,
    maximum_adjacent_growth: float = 1.15,
    minimum_tiers: int = 4,
) -> ScaleRailResult:
    """Check that normalized work remains stable across increasing tiers.

    The rail intentionally evaluates work and state slopes, not named scale
    factors or device products. Fallbacks fail unconditionally. Terminal D2H
    may remain constant or grow with the public output, but it must never exceed
    the selected payload implied by the caller's recorded row counts.
    """
    ordered = tuple(sorted(observations, key=lambda item: item.input_rows))
    if len(ordered) < int(minimum_tiers):
        raise ValueError(f"scale rail requires at least {minimum_tiers} tiers")
    if any(item.input_rows <= 0 or item.batch_count <= 0 for item in ordered):
        raise ValueError("scale rail rows and batch counts must be positive")
    if any(
        current.input_rows <= previous.input_rows
        for previous, current in zip(ordered, ordered[1:], strict=False)
    ):
        raise ValueError("scale rail input rows must be strictly increasing")
    growth = float(maximum_adjacent_growth)
    if growth < 1.0:
        raise ValueError("maximum_adjacent_growth must be at least 1.0")

    violations: list[ScaleRailViolation] = []
    for observation in ordered:
        if observation.fallback_count:
            violations.append(
                ScaleRailViolation(
                    metric="fallback_count",
                    previous_tier=observation.name,
                    current_tier=observation.name,
                    previous_normalized=0.0,
                    current_normalized=float(observation.fallback_count),
                )
            )
        if (
            observation.batch_count > 1
            and observation.prepared_builds > 1
            and observation.prepared_builds >= observation.batch_count
            and observation.prepared_builds >= observation.prepared_reuses
        ):
            violations.append(
                ScaleRailViolation(
                    metric="prepared_rebuilds_without_reuse",
                    previous_tier=observation.name,
                    current_tier=observation.name,
                    previous_normalized=1.0,
                    current_normalized=(
                        float(observation.prepared_builds)
                        / max(float(observation.prepared_reuses), 1.0)
                    ),
                )
            )
    for previous, current in zip(ordered, ordered[1:], strict=False):
        previous_metrics = _normalized_metrics(previous)
        current_metrics = _normalized_metrics(current)
        for metric, current_value in current_metrics.items():
            previous_value = previous_metrics[metric]
            if previous_value == 0.0:
                regressed = current_value > 0.0
            else:
                regressed = current_value > previous_value * growth
            if regressed:
                violations.append(
                    ScaleRailViolation(
                        metric=metric,
                        previous_tier=previous.name,
                        current_tier=current.name,
                        previous_normalized=previous_value,
                        current_normalized=current_value,
                    )
                )
    return ScaleRailResult(
        passed=not violations,
        observations=ordered,
        violations=tuple(violations),
    )


def _work_packet(stage: dict) -> dict:
    metadata = stage.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    packet = metadata.get("work_amplification")
    return packet if isinstance(packet, dict) else {}


def _spatialbench_query_payload(payload: dict, query: str) -> tuple[dict, dict]:
    suites = payload.get("results")
    if not isinstance(suites, list):
        raise ValueError("SpatialBench artifact is missing result suites")
    suite = next(
        (
            item
            for item in suites
            if isinstance(item, dict) and item.get("engine") == "vibespatial"
        ),
        None,
    )
    if suite is None:
        raise ValueError("SpatialBench artifact has no vibeSpatial suite")
    results = suite.get("results")
    result = next(
        (
            item
            for item in results or ()
            if isinstance(item, dict) and item.get("query") == query
        ),
        None,
    )
    if result is None:
        raise ValueError(f"SpatialBench artifact has no {query!r} result")
    if result.get("status") != "success":
        raise ValueError(f"SpatialBench {query!r} result did not succeed")
    return suite, result


def spatialbench_scale_observation(
    artifact,
    *,
    query: str,
    run_index: int = -1,
) -> ScaleRailObservation:
    """Build one rail observation from runner telemetry, not synthetic inputs."""
    path = Path(artifact)
    payload = json.loads(path.read_text())
    _suite, result = _spatialbench_query_payload(payload, query)
    telemetry_runs = result.get("telemetry_runs")
    if not isinstance(telemetry_runs, list) or not telemetry_runs:
        raise ValueError(
            "SpatialBench rail artifacts require --profile-telemetry and "
            "VIBESPATIAL_HOTPATH_TRACE=counter"
        )
    telemetry = telemetry_runs[run_index]
    hotpath = telemetry.get("hotpath")
    if not isinstance(hotpath, list):
        raise ValueError("SpatialBench telemetry is missing bounded hotpath packets")

    input_rows = 0
    batch_count = 0
    prepared_builds = 0
    prepared_reuses = 0
    prepared_declines = 0
    exact_refinement_work = 0
    diagnostic_synchronizations = 0
    for stage in hotpath:
        packet = _work_packet(stage)
        sums = packet.get("sum") if isinstance(packet.get("sum"), dict) else {}
        if stage.get("name") == "spatialbench.scan_decode":
            input_rows += int(sums.get("input_rows", 0))
            recorded_batches = int(sums.get("input_batches", 0))
            if recorded_batches == 0:
                # Artifacts produced before the explicit input_batches packet
                # include one final StopIteration stage call.
                recorded_batches = max(int(stage.get("calls", 1)) - 1, 1)
            batch_count += recorded_batches
        if packet.get("operation") == "prepare_point_region_y_index":
            prepared_builds += int(sums.get("build_count", 0))
            prepared_reuses += int(sums.get("cache_hits", 0))
            prepared_declines += int(sums.get("declined_preparations", 0))
        if str(stage.get("name", "")).startswith("predicate.point_region."):
            exact_refinement_work += int(sums.get("candidate_lanes", 0))
        diagnostic_synchronizations += int(
            sums.get("diagnostic_synchronizations", 0)
        )

    if input_rows <= 0 or batch_count <= 0:
        raise ValueError(
            "SpatialBench telemetry lacks scan input_rows/input_batches counters"
        )
    elapsed = result.get("time_seconds")
    if elapsed is None:
        raise ValueError("SpatialBench result is missing elapsed time")
    selected_rows = max(
        int(result.get("row_count") or 0),
        exact_refinement_work,
        1,
    )
    allocation_bytes = telemetry.get("rmm_total_allocation_bytes")
    allocation_events = telemetry.get("rmm_allocation_count")
    if allocation_bytes is None or allocation_events is None:
        raise ValueError("SpatialBench telemetry is missing RMM allocation totals")
    d2h_count = int(telemetry.get("d2h_transfer_count") or 0)

    return ScaleRailObservation(
        name=path.stem,
        input_rows=input_rows,
        selected_rows=selected_rows,
        batch_count=batch_count,
        elapsed_seconds=float(elapsed),
        allocation_bytes=int(allocation_bytes),
        allocation_events=int(allocation_events),
        materializations=int(telemetry.get("materialization_event_count") or 0),
        synchronizations=d2h_count + diagnostic_synchronizations,
        # The runner does not classify individual transfers by boundary, so
        # count every D2H byte as terminal. This is a conservative rail bound.
        terminal_d2h_bytes=int(telemetry.get("d2h_transfer_bytes") or 0),
        fallback_count=int(telemetry.get("fallback_event_count") or 0),
        prepared_builds=prepared_builds,
        prepared_reuses=prepared_reuses,
        prepared_declines=prepared_declines,
        exact_refinement_work=exact_refinement_work,
    )


def load_spatialbench_scale_observations(
    artifacts,
    *,
    query: str,
    run_index: int = -1,
) -> tuple[ScaleRailObservation, ...]:
    """Load monotonically increasing observations from runner JSON artifacts."""
    return tuple(
        spatialbench_scale_observation(
            artifact,
            query=query,
            run_index=run_index,
        )
        for artifact in artifacts
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a four-tier SpatialBench telemetry rail"
    )
    parser.add_argument("--query", required=True)
    parser.add_argument("--run-index", type=int, default=-1)
    parser.add_argument("--maximum-adjacent-growth", type=float, default=1.15)
    parser.add_argument("artifacts", nargs="+")
    args = parser.parse_args(argv)
    observations = load_spatialbench_scale_observations(
        args.artifacts,
        query=args.query,
        run_index=args.run_index,
    )
    result = evaluate_scale_rail(
        observations,
        maximum_adjacent_growth=args.maximum_adjacent_growth,
    )
    print(
        json.dumps(
            {
                "query": args.query,
                "passed": result.passed,
                "observations": [item.__dict__ for item in result.observations],
                "violations": [item.__dict__ for item in result.violations],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.passed else 1


__all__ = [
    "ScaleRailObservation",
    "ScaleRailResult",
    "ScaleRailViolation",
    "evaluate_scale_rail",
    "load_spatialbench_scale_observations",
    "spatialbench_scale_observation",
]


if __name__ == "__main__":
    raise SystemExit(main())
