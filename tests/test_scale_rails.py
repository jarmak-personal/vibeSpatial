from __future__ import annotations

import json

import pytest

from vibespatial.bench.scale_rails import (
    ScaleRailObservation,
    evaluate_scale_rail,
    main,
    spatialbench_scale_observation,
)


def _observation(name: str, rows: int, *, allocation_multiplier: int = 24):
    batches = max(rows // 1_000, 1)
    return ScaleRailObservation(
        name=name,
        input_rows=rows,
        selected_rows=rows // 10,
        batch_count=batches,
        elapsed_seconds=rows / 2_000_000,
        allocation_bytes=rows * allocation_multiplier,
        allocation_events=batches * 4,
        materializations=1,
        synchronizations=batches,
        d2h_bytes=800,
    )


def test_scale_rail_accepts_four_linear_work_tiers() -> None:
    result = evaluate_scale_rail(
        [_observation(name, rows) for name, rows in (
            ("sf1", 10_000),
            ("sf10", 100_000),
            ("sf100", 1_000_000),
            ("sf1000", 10_000_000),
        )]
    )

    assert result.passed
    assert result.violations == ()


def test_scale_rail_rejects_superlinear_allocation_and_fallback() -> None:
    observations = [
        _observation("tier-1", 10_000),
        _observation("tier-2", 100_000),
        _observation("tier-3", 1_000_000),
        _observation("tier-4", 10_000_000, allocation_multiplier=48),
    ]
    observations[-1] = ScaleRailObservation(
        **{
            **observations[-1].__dict__,
            "fallback_count": 1,
        }
    )

    result = evaluate_scale_rail(observations)

    assert not result.passed
    assert {violation.metric for violation in result.violations} == {
        "allocation_bytes_per_input_row",
        "fallback_count",
    }


def test_scale_rail_rejects_growing_d2h_and_refinement_work() -> None:
    observations = [
        _observation("tier-1", 10_000),
        _observation("tier-2", 100_000),
        _observation("tier-3", 1_000_000),
        _observation("tier-4", 10_000_000),
    ]
    observations[-1] = ScaleRailObservation(
        **{
            **observations[-1].__dict__,
            "d2h_bytes": 16_000_000,
            "exact_refinement_work": 20_000_000,
        }
    )

    result = evaluate_scale_rail(observations)

    assert not result.passed
    assert {
        "d2h_bytes_per_input_row",
        "exact_refinement_work_per_input_row",
    }.issubset({violation.metric for violation in result.violations})


def test_scale_rail_requires_four_strictly_increasing_tiers() -> None:
    with pytest.raises(ValueError, match="at least 4"):
        evaluate_scale_rail([_observation("one", 1_000)])
    with pytest.raises(ValueError, match="strictly increasing"):
        evaluate_scale_rail(
            [_observation(str(index), 1_000) for index in range(4)]
        )


def test_scale_rail_rejects_per_batch_prepared_rebuilds() -> None:
    observations = [
        _observation("tier-1", 10_000),
        _observation("tier-2", 100_000),
        _observation("tier-3", 1_000_000),
        _observation("tier-4", 10_000_000),
    ]
    observations[-1] = ScaleRailObservation(
        **{
            **observations[-1].__dict__,
            "prepared_builds": observations[-1].batch_count,
            "prepared_reuses": 0,
        }
    )

    result = evaluate_scale_rail(observations)

    assert not result.passed
    assert "prepared_rebuilds_without_reuse" in {
        violation.metric for violation in result.violations
    }


def test_scale_rail_allows_one_reusable_build_after_no_build_tier() -> None:
    observations = [
        _observation("tier-1", 10_000),
        _observation("tier-2", 100_000),
        _observation("tier-3", 1_000_000),
        _observation("tier-4", 10_000_000),
    ]
    observations[1] = ScaleRailObservation(
        **{
            **observations[1].__dict__,
            "prepared_builds": 1,
            "prepared_reuses": observations[1].batch_count - 1,
        }
    )

    result = evaluate_scale_rail(observations)

    assert result.passed


def _write_spatialbench_artifact(path, *, input_rows: int, batch_count: int) -> None:
    payload = {
        "benchmark": "spatialbench",
        "results": [
            {
                "engine": "vibespatial",
                "scale_factor": 1.0,
                "results": [
                    {
                        "query": "q6",
                        "time_seconds": input_rows / 2_000_000,
                        "row_count": 100,
                        "status": "success",
                        "telemetry_runs": [
                            {
                                "rmm_total_allocation_bytes": input_rows * 24,
                                "rmm_allocation_count": batch_count * 4,
                                "d2h_transfer_count": batch_count,
                                "d2h_transfer_bytes": 800,
                                "fallback_event_count": 0,
                                "materialization_event_count": 1,
                                "hotpath": [
                                    {
                                        "name": "spatialbench.scan_decode",
                                        "metadata": {
                                            "work_amplification": {
                                                "sum": {
                                                    "input_rows": input_rows,
                                                    "input_batches": batch_count,
                                                }
                                            }
                                        },
                                    },
                                    {
                                        "name": (
                                            "predicate.point_location_part_y_index.prepare"
                                        ),
                                        "metadata": {
                                            "work_amplification": {
                                                "operation": (
                                                    "prepare_point_region_y_index"
                                                ),
                                                "sum": {
                                                    "build_count": 1,
                                                    "cache_hits": batch_count - 1,
                                                    "declined_preparations": 0,
                                                },
                                            }
                                        },
                                    },
                                    {
                                        "name": "spatial.query_aggregate.query_input",
                                        "metadata": {
                                            "work_amplification": {
                                                "operation": (
                                                    "select_spatial_aggregate_query_carrier"
                                                ),
                                                "max": {
                                                    "ancestral_coordinate_capacity": 2_000_000,
                                                    "compact_coordinate_capacity": 500_000,
                                                    "preparation_minimum_coordinates": 1_000_000,
                                                },
                                            }
                                        },
                                    },
                                    {
                                        "name": "predicate.point_region.contains",
                                        "metadata": {
                                            "work_amplification": {
                                                "sum": {
                                                    "candidate_lanes": input_rows // 10
                                                }
                                            }
                                        },
                                    },
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }
    path.write_text(json.dumps(payload))


def test_spatialbench_artifact_wires_real_telemetry_into_scale_rail(
    tmp_path,
    capsys,
) -> None:
    paths = []
    for index, rows in enumerate((10_000, 100_000, 1_000_000, 10_000_000), 1):
        path = tmp_path / f"tier-{index}.json"
        _write_spatialbench_artifact(
            path,
            input_rows=rows,
            batch_count=rows // 1_000,
        )
        paths.append(path)

    observation = spatialbench_scale_observation(paths[0], query="q6")
    assert observation.input_rows == 10_000
    assert observation.batch_count == 10
    assert observation.prepared_builds == 1
    assert observation.prepared_reuses == 9
    assert observation.exact_refinement_work == 1_000
    assert observation.ancestral_coordinate_capacity == 2_000_000
    assert observation.compact_coordinate_capacity == 500_000
    assert observation.preparation_minimum_coordinates == 1_000_000

    assert main(["--query", "q6", *(str(path) for path in paths)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is True
    assert len(report["observations"]) == 4


def test_q6_scale_rail_requires_large_ancestor_small_derivative_proof() -> None:
    observations = [
        _observation(name, rows)
        for name, rows in (
            ("sf1", 10_000),
            ("sf10", 100_000),
            ("sf100", 1_000_000),
            ("sf1000", 10_000_000),
        )
    ]

    result = evaluate_scale_rail(
        observations,
        require_reuse_admission_shape=True,
    )

    assert not result.passed
    assert {violation.metric for violation in result.violations} == {
        "missing_reuse_admission_shape"
    }
