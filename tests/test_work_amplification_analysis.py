from __future__ import annotations

from copy import deepcopy

import pytest

from vibespatial.bench.amplification import Record
from vibespatial.bench.amplification_analysis import analyze_work_amplification


def _metrics(record):
    return {metric["name"]: metric for metric in record["metrics"]}


def _ratios(record):
    return {ratio["name"]: ratio for ratio in record["ratios"]}


def _pipeline_artifact(*stages, elapsed_seconds=10.0):
    return {
        "results": [
            {
                "pipeline": "overlay-relation-constructive",
                "elapsed_seconds": elapsed_seconds,
                "stages": [
                    {
                        "trace": {
                            "operation": "overlay-relation-constructive",
                            "total_elapsed_seconds": elapsed_seconds,
                            "stages": list(stages),
                        }
                    }
                ],
            }
        ]
    }


def test_pipeline_relation_metrics_are_exact_derived_and_attached_without_mutation():
    stage = {
        "name": "refine_relation",
        "device": "gpu",
        "elapsed_seconds": 2.5,
        "rows_in": 999_999,
        "rows_out": 1,
        "metadata": {
            "candidate_pairs": 100,
            "refined_pairs": 0,
            "relation_pairs": 20,
            "terminal_rows": 5,
            "pair_bytes": 320,
            "terminal_bytes": 40,
        },
    }
    artifact = _pipeline_artifact(stage)
    original = deepcopy(artifact)

    analysis = analyze_work_amplification(artifact)

    assert artifact == original
    enriched_stage = analysis["artifact"]["results"][0]["stages"][0]["trace"]["stages"][0]
    records = enriched_stage["metadata"]["work_amplification"]
    assert records == analysis["records"]
    relation = next(record for record in records if record["metric_family"] == "relation")
    metrics = _metrics(relation)
    assert metrics["candidate_pairs"]["status"] == "exact"
    assert metrics["stage_elapsed_seconds"]["unit"] == "seconds"
    assert "rows_in" not in metrics
    assert "rows_out" not in metrics
    assert _ratios(relation)["relation_to_terminal"]["value"] == 4
    assert _ratios(relation)["relation_byte_amplification"]["value"] == 8
    zero_ratio = _ratios(relation)["coarse_to_exact"]
    assert zero_ratio["status"] == "invalid"
    assert zero_ratio["value"] is None
    assert zero_ratio["denominator_zero"] is True
    assert Record.from_dict(relation).to_dict() == relation


def test_generic_stage_rows_never_become_physical_counts_or_findings():
    artifact = _pipeline_artifact(
        {
            "name": "public_projection",
            "device": "gpu",
            "elapsed_seconds": 9.0,
            "rows_in": 1_000_000_000,
            "rows_out": 1,
            "metadata": {},
        }
    )

    analysis = analyze_work_amplification(artifact)

    assert len(analysis["records"]) == 1
    assert analysis["records"][0]["metric_family"] == "timing"
    assert set(_metrics(analysis["records"][0])) == {"stage_elapsed_seconds"}
    assert analysis["findings"] == []


def test_observed_device_memory_is_context_not_attributable_peak():
    artifact = _pipeline_artifact(
        {
            "name": "read_input",
            "device": "gpu",
            "elapsed_seconds": 2.0,
            "metadata": {"peak_device_memory_bytes": 2 * 1024**3},
        },
        elapsed_seconds=2.0,
    )

    analysis = analyze_work_amplification(artifact)

    capacity = next(
        record for record in analysis["records"] if record["metric_family"] == "capacity"
    )
    metrics = _metrics(capacity)
    assert metrics["observed_device_memory_bytes"]["value"] == 2 * 1024**3
    assert "peak_live_bytes" not in metrics
    assert analysis["findings"] == []


def test_shootout_profile_is_separate_from_timing_and_parses_same_event_capacity():
    artifact = {
        "script": "retail.py",
        "results": [
            {
                "script": "retail.py",
                "vibespatial": {
                    "timing": {"median_seconds": 1.0},
                    "profile": {
                        "mode": "post_timing_profile",
                        "elapsed_seconds": 3.0,
                        "timed_stages": [
                            {
                                "index": 4,
                                "actual_backend": "gpu",
                                "elapsed_seconds": 2.0,
                                "materialization_events": [
                                    {
                                        "boundary": "user-export",
                                        "detail": "kind=relation, capacity=100, rows=10, bytes=800",
                                    }
                                ],
                            }
                        ],
                        "hotpath": [
                            {
                                "name": "nested.kernel",
                                "elapsed_seconds": 1.5,
                            }
                        ],
                    },
                },
            }
        ],
    }

    analysis = analyze_work_amplification(artifact)

    observer = analysis["observer_effects"][0]
    assert observer["post_profile_to_timed_ratio"] == 3
    assert observer["wall_attribution"] == "unreliable"
    assert observer["is_workload_amplification"] is False
    profile = analysis["artifact"]["results"][0]["vibespatial"]["profile"]
    statement_records = profile["timed_stages"][0]["work_amplification"]
    capacity = next(record for record in statement_records if record["metric_family"] == "capacity")
    assert _ratios(capacity)["slot_utilization"]["value"] == pytest.approx(0.1)
    assert _metrics(capacity)["reported_boundary_bytes"]["value"] == 800
    assert (
        capacity["semantic_contract"][
            "reported_boundary_bytes_are_not_peak_or_transfer_bytes"
        ]
        is True
    )
    assert capacity["semantic_contract"]["same_event_pairing_only"] is True
    hotpath_record = profile["hotpath"][0]["work_amplification"][0]
    assert hotpath_record["semantic_contract"]["hotpath_timers_may_overlap"] is True
    assert all(effect not in analysis["records"] for effect in analysis["observer_effects"])
    finding = next(item for item in analysis["findings"] if item["stage"] == "statement_4")
    assert "corrected_counterfactual_profile" in finding["needed_metrics"]


def test_counter_hotpath_packet_becomes_schema_record_without_fake_timing():
    artifact = {
        "script": "q11.py",
        "vibespatial": {
            "timing": {"median_seconds": 12.0},
            "profile": {
                "profile_mode": "counters",
                "elapsed_seconds": 12.1,
                "timed_stages": [],
                "hotpath": [
                    {
                        "name": "spatial.point_partition.reduce",
                        "category": "filter",
                        "calls": 4,
                        "elapsed_seconds": 0.0,
                        "timing_mode": "counter",
                        "metadata": {
                            "work_amplification": {
                                "schema_version": 1,
                                "operation": "query_pair_aggregate",
                                "metric_family": "relation",
                                "sum": {
                                    "candidate_pairs": 40,
                                    "refined_pairs": 10,
                                },
                                "max": {"live_pair_capacity": 16},
                                "unavailable": ["survivors"],
                                "semantic_contract": {"predicate": "within"},
                            }
                        },
                    }
                ],
            },
        },
    }

    analysis = analyze_work_amplification(artifact)

    relation = next(
        record for record in analysis["records"] if record["metric_family"] == "relation"
    )
    metrics = _metrics(relation)
    assert metrics["candidate_pairs"]["value"] == 40
    assert metrics["live_pair_capacity"]["metadata"]["reducer"] == "max"
    assert metrics["survivors"]["status"] == "unavailable"
    assert "stage_elapsed_seconds" not in metrics
    assert _ratios(relation)["coarse_to_exact"]["value"] == 4
    assert relation["semantic_contract"]["counter_packet"] is True
    assert relation["semantic_contract"]["predicate"] == "within"
    assert Record.from_dict(relation).to_dict() == relation


def test_counter_packet_uses_enclosing_statement_once_when_available():
    packet_stage = {
        "name": "spatial.point_partition.reduce",
        "category": "filter",
        "calls": 4,
        "elapsed_seconds": 0.0,
        "timing_mode": "counter",
        "metadata": {
            "work_amplification": {
                "schema_version": 1,
                "operation": "query_pair_aggregate",
                "metric_family": "relation",
                "sum": {"candidate_pairs": 40, "refined_pairs": 10},
                "max": {},
                "unavailable": ["survivors"],
            }
        },
    }
    artifact = {
        "script": "q11.py",
        "vibespatial": {
            "timing": {"median_seconds": 6.0},
            "profile": {
                "profile_mode": "counters",
                "elapsed_seconds": 6.1,
                "timed_stages": [
                    {
                        "index": 3,
                        "elapsed_seconds": 5.0,
                        "hotpath": [deepcopy(packet_stage)],
                    }
                ],
                "hotpath": [deepcopy(packet_stage)],
            },
        },
    }

    analysis = analyze_work_amplification(artifact)

    relations = [
        record for record in analysis["records"] if record["metric_family"] == "relation"
    ]
    assert len(relations) == 1
    assert _metrics(relations[0])["stage_elapsed_seconds"]["value"] == 5.0
    assert relations[0]["semantic_contract"]["elapsed_scope"] == "enclosing_statement"
    assert relations[0]["semantic_contract"]["nested_counter_timing_unavailable"] is True
    assert relations[0]["semantic_contract"]["workflow_operation"] == "q11.py"


def test_counter_packet_omits_ratio_with_unavailable_denominator():
    packet_stage = {
        "name": "relation.reduce",
        "category": "filter",
        "calls": 1,
        "metadata": {
            "work_amplification": {
                "schema_version": 1,
                "operation": "reduce_relation",
                "metric_family": "relation",
                "physical_shape": "candidate_pairs",
                "consumer_kind": "existence",
                "sum": {"candidate_pairs": 12},
                "max": {},
                "unavailable": ["refined_pairs"],
            }
        },
    }
    artifact = {
        "type": "shootout",
        "script": "example.py",
        "status": "pass",
        "geopandas": {"timing": {"median_seconds": 2.0}},
        "vibespatial": {
            "timing": {"median_seconds": 1.0},
            "profile": {
                "available": True,
                "profile_mode": "counters",
                "elapsed_seconds": 1.0,
                "timed_stages": [
                    {
                        "index": 0,
                        "elapsed_seconds": 0.5,
                        "actual_backend": "gpu",
                        "hotpath": [packet_stage],
                    }
                ],
            },
        },
        "metadata": {"scale": "1M"},
    }

    analysis = analyze_work_amplification(artifact)
    relation = next(
        record
        for record in analysis["records"]
        if record["operation"] == "reduce_relation"
    )
    assert relation["ratios"] == []


def test_ranking_ignores_unavailable_memory_metrics():
    artifact = _pipeline_artifact(
        {
            "name": "rebuild",
            "device": "gpu",
            "elapsed_seconds": 2.0,
            "metadata": {
                "work_amplification": {
                    "schema_version": 1,
                    "operation": "prepare_index",
                    "metric_family": "rebuild",
                    "sum": {"build_count": 1},
                    "max": {},
                    "unavailable": ["persistent_bytes"],
                }
            },
        },
        elapsed_seconds=2.0,
    )

    analysis = analyze_work_amplification(artifact)
    assert analysis["findings"] == []


def test_materialization_counters_from_different_events_are_not_coupled():
    artifact = {
        "vibespatial": {
            "timing": {"mean_seconds": 2.0},
            "profile": {
                "elapsed_seconds": 2.0,
                "timed_stages": [
                    {
                        "index": 0,
                        "elapsed_seconds": 1.5,
                        "materialization_events": [
                            {"detail": "capacity=100"},
                            {"detail": "rows=5"},
                        ],
                    }
                ],
                "hotpath": [],
            },
        }
    }

    analysis = analyze_work_amplification(artifact)
    event_records = analysis["artifact"]["vibespatial"]["profile"]["timed_stages"][0]["work_amplification"]

    assert len(event_records) == 1
    assert "slot_utilization" not in _ratios(event_records[0])
    assert set(_metrics(event_records[0])) >= {"capacity_slots", "stage_elapsed_seconds"}


def test_sf100_timing_records_public_rows_without_ranking_them_as_amplification():
    artifact = {
        "benchmark": "spatialbench",
        "results": [
            {
                "engine": "vibespatial",
                "scale_factor": 100,
                "results": [
                    {
                        "query": "q11",
                        "time_seconds": 252.41,
                        "row_count": 1,
                        "status": "success",
                    }
                ],
            }
        ],
    }

    analysis = analyze_work_amplification(artifact)

    record = analysis["records"][0]
    assert _metrics(record)["stage_elapsed_seconds"]["value"] == 252.41
    assert _metrics(record)["public_output_rows"]["value"] == 1
    assert record["semantic_contract"]["public_output_rows_are_not_physical_work"] is True
    assert record["semantic_contract"]["physical_metrics"] == "unavailable"
    assert analysis["findings"] == []
    assert analysis["artifact"]["results"][0]["results"][0]["work_amplification"] == [record]


def test_flat_sf100_shape_and_alternate_timing_keys_are_supported():
    artifact = {
        "engine": "vibespatial",
        "results": [{"query": "q2", "seconds": 4.5, "rows": 0, "status": "success"}],
    }

    analysis = analyze_work_amplification(artifact)

    record = analysis["records"][0]
    assert _metrics(record)["stage_elapsed_seconds"]["source"] == "query.seconds"
    assert _metrics(record)["public_output_rows"]["value"] == 0
    assert Record.from_dict(record).to_dict() == record


def test_sf100_telemetry_and_counter_packets_remain_separate_from_public_rows():
    artifact = {
        "engine": "vibespatial",
        "results": [
            {
                "query": "q11",
                "seconds": 10.0,
                "rows": 1,
                "telemetry_runs": [
                    {
                        "rmm_peak_allocation_bytes": 256 * 1024**2,
                        "rmm_total_allocation_bytes": 2 * 1024**3,
                        "hotpath": [
                            {
                                "name": "point.reduce",
                                "timing_mode": "counter",
                                "elapsed_seconds": 0.0,
                                "metadata": {
                                    "work_amplification": {
                                        "schema_version": 1,
                                        "operation": "query_pair_aggregate",
                                        "metric_family": "relation",
                                        "sum": {
                                            "candidate_pairs": 100,
                                            "refined_pairs": 10,
                                        },
                                        "max": {},
                                        "unavailable": ["terminal_rows"],
                                    }
                                },
                            }
                        ],
                    }
                ],
            }
        ],
    }

    analysis = analyze_work_amplification(artifact)

    capacity = next(
        record for record in analysis["records"] if record["metric_family"] == "capacity"
    )
    relation = next(
        record for record in analysis["records"] if record["metric_family"] == "relation"
    )
    assert _metrics(capacity)["peak_live_bytes"]["value"] == 256 * 1024**2
    assert capacity["semantic_contract"]["telemetry_wall_includes_observer_overhead"] is True
    assert _ratios(relation)["coarse_to_exact"]["value"] == 10
    assert "stage_elapsed_seconds" not in _metrics(relation)
    assert all("public_output_rows" not in _metrics(item) for item in (capacity, relation))


def test_sf100_point_region_profile_emits_refinement_and_rebuild_records():
    artifact = {
        "engine": "vibespatial",
        "results": [
            {
                "query": "q11",
                "seconds": 10.0,
                "rows": 1,
                "telemetry_runs": [
                    {
                        "point_region_profile": {
                            "profile_wall_seconds": 10.2,
                            "sample_limit": 64,
                            "samples_per_launch": 8,
                            "groups": [
                                {
                                    "family": "multipolygon",
                                    "geometry_count": 5,
                                    "part_count": 7,
                                    "edge_membership_count": 100,
                                    "kernel_seconds": 6.0,
                                    "counters": {
                                        "candidates": 1000,
                                        "valid_candidates": 1000,
                                        "parts_considered": 10000,
                                        "active_parts": 800,
                                        "edges_visited": 50000,
                                        "orient2d_calls": 400,
                                        "zero_active_candidates": 200,
                                        "boundary_results": 0,
                                        "interior_results": 100,
                                        "exterior_results": 900,
                                        "max_parts_considered": 50,
                                        "max_active_parts": 4,
                                        "max_edges_visited": 500,
                                    },
                                }
                            ],
                            "index_preparation": [
                                {
                                    "family": "multipolygon",
                                    "build_count": 1,
                                    "cache_hits": 9,
                                    "build_wall_seconds": 0.2,
                                }
                            ],
                        }
                    }
                ],
            }
        ],
    }

    analysis = analyze_work_amplification(artifact)
    refinement = next(
        record
        for record in analysis["records"]
        if record["operation"] == "point_region_profile"
    )
    rebuild = next(
        record
        for record in analysis["records"]
        if record["operation"] == "prepare_point_region_y_index"
    )
    assert _metrics(refinement)["candidate_parts_considered"]["value"] == 10000
    assert _metrics(refinement)["edge_visits"]["value"] == 50000
    assert _ratios(refinement)["exact_work_per_survivor"]["value"] == 4
    assert _ratios(refinement)["candidate_parts_per_lane"]["value"] == 10
    assert _ratios(refinement)["edge_visits_per_lane"]["value"] == 50
    assert _ratios(refinement)["edge_visits_per_survivor"]["value"] == 500
    assert _ratios(refinement)["early_exit_fraction"]["value"] == 0.2
    assert _metrics(rebuild)["build_seconds"]["value"] == 0.2
    assert _ratios(rebuild)["consumer_reuse"]["value"] == 10


def test_ranking_requires_both_wall_threshold_and_substantive_signal():
    artifact = _pipeline_artifact(
        {
            "name": "cheap_amplified",
            "device": "gpu",
            "elapsed_seconds": 0.01,
            "metadata": {"candidate_pairs": 1000, "refined_pairs": 1},
        },
        {
            "name": "material_amplified",
            "device": "gpu",
            "elapsed_seconds": 0.6,
            "metadata": {"candidate_pairs": 100, "refined_pairs": 10},
        },
        {
            "name": "slow_but_unmeasured",
            "device": "gpu",
            "elapsed_seconds": 4.0,
            "metadata": {"pairs_examined": 1_000_000},
        },
        elapsed_seconds=10.0,
    )

    analysis = analyze_work_amplification(artifact)

    assert [finding["stage"] for finding in analysis["findings"]] == ["material_amplified"]
    assert analysis["findings"][0]["workflow_wall_fraction"] == pytest.approx(0.06)


def test_rejects_non_dictionary_but_accepts_unknown_dictionary():
    with pytest.raises(TypeError, match="artifact must be a dictionary"):
        analyze_work_amplification([])
    assert analyze_work_amplification({}) == {
        "artifact": {},
        "records": [],
        "findings": [],
        "observer_effects": [],
    }
