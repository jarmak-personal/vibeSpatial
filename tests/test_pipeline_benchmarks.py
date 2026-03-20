from __future__ import annotations

import json

from scripts.check_pipeline_regressions import compare_results
from vibespatial.bench.pipeline import benchmark_pipeline_suite, suite_to_json


def test_pipeline_smoke_suite_runs_active_pipelines() -> None:
    results = benchmark_pipeline_suite(suite="smoke")
    by_name = {(result.pipeline, result.scale): result for result in results}

    assert ("join-heavy", 1000) in by_name
    assert ("constructive", 1000) in by_name
    assert ("predicate-heavy", 1000) in by_name
    assert ("zero-transfer", 1000) in by_name
    assert by_name[("join-heavy", 1000)].status == "ok"
    assert by_name[("constructive", 1000)].output_rows >= 0
    assert by_name[("predicate-heavy", 1000)].selected_runtime in {"cpu", "hybrid", "gpu"}
    assert by_name[("zero-transfer", 1000)].status in {"ok", "deferred"}
    if by_name[("zero-transfer", 1000)].status == "ok":
        assert by_name[("zero-transfer", 1000)].transfer_count == 0
        assert by_name[("zero-transfer", 1000)].materialization_count == 0
    stage_trace = by_name[("join-heavy", 1000)].stages[0]
    stage = stage_trace["stages"][0]
    assert "transfer_count_total" in stage["metadata"]
    assert "materialization_count_total" in stage["metadata"]
    payload = json.loads(suite_to_json(results, suite="smoke", repeat=1))
    assert payload["metadata"]["suite"] == "smoke"
    assert payload["metadata"]["repeat"] == 1


def test_pipeline_smoke_suite_can_run_geopandas_predicate_baseline() -> None:
    results = benchmark_pipeline_suite(suite="smoke", pipelines=("predicate-heavy-geopandas",))
    assert len(results) == 1
    result = results[0]
    assert result.pipeline == "predicate-heavy-geopandas"
    assert result.scale == 1000
    assert result.status == "ok"
    assert result.selected_runtime == "cpu"
    trace = result.stages[0]
    stage_names = [stage["name"] for stage in trace["stages"]]
    assert stage_names == ["read_geojson", "load_polygons", "point_in_polygon", "filter_points", "write_output"]
    read_stage = trace["stages"][0]
    assert read_stage["metadata"]["requested_engine"] == "pyogrio"
    assert read_stage["metadata"]["actual_engine"] in {"pyogrio", "default"}


def test_vegetation_corridor_smoke() -> None:
    results = benchmark_pipeline_suite(suite="smoke", pipelines=("vegetation-corridor",))
    assert len(results) == 1
    result = results[0]
    assert result.pipeline == "vegetation-corridor"
    assert result.scale == 1000
    assert result.status == "ok"
    assert result.selected_runtime in {"cpu", "hybrid", "gpu"}
    trace = result.stages[0]
    stage_names = [stage["name"] for stage in trace["stages"]]
    assert "read_lines" in stage_names
    assert "buffer_lines" in stage_names
    assert "dissolve_corridor" in stage_names
    assert "intersect_vegetation" in stage_names
    assert "write_output" in stage_names


def test_vegetation_corridor_geopandas_smoke() -> None:
    results = benchmark_pipeline_suite(suite="smoke", pipelines=("vegetation-corridor-geopandas",))
    assert len(results) == 1
    result = results[0]
    assert result.pipeline == "vegetation-corridor-geopandas"
    assert result.status == "ok"
    assert result.selected_runtime == "cpu"


def test_parcel_zoning_smoke() -> None:
    results = benchmark_pipeline_suite(suite="smoke", pipelines=("parcel-zoning",))
    assert len(results) == 1
    result = results[0]
    assert result.pipeline == "parcel-zoning"
    assert result.status == "ok"
    trace = result.stages[0]
    stage_names = [stage["name"] for stage in trace["stages"]]
    assert "clip_to_study_area" in stage_names
    assert "overlay_intersect" in stage_names


def test_parcel_zoning_geopandas_smoke() -> None:
    results = benchmark_pipeline_suite(suite="smoke", pipelines=("parcel-zoning-geopandas",))
    assert len(results) == 1
    result = results[0]
    assert result.pipeline == "parcel-zoning-geopandas"
    assert result.status == "ok"
    assert result.selected_runtime == "cpu"


def test_flood_exposure_smoke() -> None:
    results = benchmark_pipeline_suite(suite="smoke", pipelines=("flood-exposure",))
    assert len(results) == 1
    result = results[0]
    assert result.pipeline == "flood-exposure"
    assert result.status == "ok"
    trace = result.stages[0]
    stage_names = [stage["name"] for stage in trace["stages"]]
    assert "make_valid" in stage_names
    assert "sjoin_intersects" in stage_names


def test_flood_exposure_geopandas_smoke() -> None:
    results = benchmark_pipeline_suite(suite="smoke", pipelines=("flood-exposure-geopandas",))
    assert len(results) == 1
    result = results[0]
    assert result.pipeline == "flood-exposure-geopandas"
    assert result.status == "ok"
    assert result.selected_runtime == "cpu"


def test_network_service_area_smoke() -> None:
    results = benchmark_pipeline_suite(suite="smoke", pipelines=("network-service-area",))
    assert len(results) == 1
    result = results[0]
    assert result.pipeline == "network-service-area"
    assert result.status == "ok"
    trace = result.stages[0]
    stage_names = [stage["name"] for stage in trace["stages"]]
    assert "buffer_network" in stage_names
    assert "dissolve_service_area" in stage_names
    assert "clip_to_admin" in stage_names


def test_network_service_area_geopandas_smoke() -> None:
    results = benchmark_pipeline_suite(suite="smoke", pipelines=("network-service-area-geopandas",))
    assert len(results) == 1
    result = results[0]
    assert result.pipeline == "network-service-area-geopandas"
    assert result.status == "ok"
    assert result.selected_runtime == "cpu"


def test_site_suitability_smoke() -> None:
    results = benchmark_pipeline_suite(suite="smoke", pipelines=("site-suitability",))
    assert len(results) == 1
    result = results[0]
    assert result.pipeline == "site-suitability"
    assert result.status == "ok"
    trace = result.stages[0]
    stage_names = [stage["name"] for stage in trace["stages"]]
    assert "overlay_difference" in stage_names
    assert "buffer_transit" in stage_names
    assert "sjoin_proximity" in stage_names


def test_site_suitability_geopandas_smoke() -> None:
    results = benchmark_pipeline_suite(suite="smoke", pipelines=("site-suitability-geopandas",))
    assert len(results) == 1
    result = results[0]
    assert result.pipeline == "site-suitability-geopandas"
    assert result.status == "ok"
    assert result.selected_runtime == "cpu"


def test_pipeline_regression_check_flags_wall_clock_and_transfers() -> None:
    baseline = {
        "results": [
            {
                "pipeline": "join-heavy",
                "scale": 100000,
                "status": "ok",
                "elapsed_seconds": 1.0,
                "transfer_count": 1,
                "materialization_count": 1,
                "peak_device_memory_bytes": 100,
            }
        ]
    }
    current = {
        "results": [
            {
                "pipeline": "join-heavy",
                "scale": 100000,
                "status": "ok",
                "elapsed_seconds": 1.2,
                "transfer_count": 2,
                "materialization_count": 2,
                "peak_device_memory_bytes": 130,
            }
        ]
    }

    findings = compare_results(current, baseline)
    metrics = {finding.metric for finding in findings}

    assert "wall_clock" in metrics
    assert "transfer_count" in metrics
    assert "materialization_count" in metrics
    assert "peak_device_memory_bytes" in metrics
