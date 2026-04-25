from __future__ import annotations

import json

import pytest
from shapely.geometry import GeometryCollection, LineString, Point

import vibespatial.bench.pipeline as pipeline_module
from vibespatial.bench.compare import compare_results
from vibespatial.bench.pipeline import (
    PipelineBenchmarkResult,
    _actual_array_device_label,
    _from_shapely_safe,
    _profile_predicate_pipeline,
    benchmark_pipeline_suite,
    render_gpu_sparkline_report,
    suite_to_json,
)
from vibespatial.bench.runner import _extract_gpu_util
from vibespatial.runtime import has_gpu_runtime


def test_pipeline_smoke_suite_runs_active_pipelines() -> None:
    pytest.importorskip("pylibcudf")
    results = benchmark_pipeline_suite(suite="smoke")
    by_name = {(result.pipeline, result.scale): result for result in results}

    assert ("join-heavy", 1000) in by_name
    assert ("relation-semijoin", 1000) in by_name
    assert ("constructive", 1000) in by_name
    assert ("predicate-heavy", 1000) in by_name
    assert ("zero-transfer", 1000) in by_name
    assert by_name[("join-heavy", 1000)].status == "ok"
    relation_semijoin = by_name[("relation-semijoin", 1000)]
    assert relation_semijoin.status in {"ok", "deferred", "failed"}
    if relation_semijoin.status != "deferred":
        assert 0 < relation_semijoin.output_rows < 1000
        relation_stages = {
            stage["name"]: stage
            for stage in relation_semijoin.stages[0]["stages"]
        }
        assert relation_stages["sjoin_relation"]["metadata"]["pair_storage"] in {
            "device",
            "host",
        }
        assert (
            relation_stages["semijoin_rowset"]["metadata"][
                "runtime_d2h_transfer_count_delta"
            ]
            == 0
        )
        assert (
            relation_stages["subset_rows"]["metadata"][
                "runtime_d2h_transfer_count_delta"
            ]
            == 0
        )
        assert (
            relation_stages["subset_rows"]["metadata"]["materialization_count_delta"]
            == 0
        )
    assert by_name[("constructive", 1000)].output_rows >= 0
    assert by_name[("predicate-heavy", 1000)].selected_runtime in {"cpu", "hybrid", "gpu"}
    zero_transfer = by_name[("zero-transfer", 1000)]
    assert zero_transfer.status in {"ok", "deferred", "failed"}
    if zero_transfer.status != "deferred":
        assert 0 < zero_transfer.output_rows < 1000
        predicate_stage = next(
            stage
            for stage in zero_transfer.stages[0]["stages"]
            if stage["name"] == "predicate_filter"
        )
        assert predicate_stage["metadata"]["predicate_bounds"] == (
            0.0,
            0.0,
            400.0,
            400.0,
        )
    if zero_transfer.status == "ok":
        assert zero_transfer.transfer_count == 0
        assert zero_transfer.materialization_count == 0
    elif zero_transfer.status == "failed":
        assert (zero_transfer.runtime_d2h_transfer_count or 0) > 0 or (
            zero_transfer.materialization_count > 0
        )
        assert zero_transfer.owned_transfer_count == 0
        stage_d2h = sum(
            stage["metadata"].get("runtime_d2h_transfer_count_delta", 0)
            for stage in zero_transfer.stages[0]["stages"]
        )
        assert stage_d2h == zero_transfer.runtime_d2h_transfer_count
        stage_d2h_seconds = sum(
            stage["metadata"].get("runtime_d2h_transfer_seconds_delta", 0.0)
            for stage in zero_transfer.stages[0]["stages"]
        )
        assert stage_d2h_seconds == pytest.approx(
            zero_transfer.runtime_d2h_transfer_seconds or 0.0
        )
    stage_trace = by_name[("join-heavy", 1000)].stages[0]
    stage = stage_trace["stages"][0]
    assert "transfer_count_total" in stage["metadata"]
    assert "owned_transfer_count_total" in stage["metadata"]
    assert "runtime_d2h_transfer_count_total" in stage["metadata"]
    assert "runtime_d2h_transfer_seconds_total" in stage["metadata"]
    assert "gpu_device_name" not in stage["metadata"]
    assert "gpu_event_elapsed_seconds" not in stage["metadata"]
    assert "materialization_count_total" in stage["metadata"]
    payload = json.loads(suite_to_json(results, suite="smoke", repeat=1))
    assert payload["metadata"]["suite"] == "smoke"
    assert payload["metadata"]["repeat"] == 1
    assert payload["metadata"]["profile_mode"] == "lean"
    result_payload = {
        (item["pipeline"], item["scale"]): item
        for item in payload["results"]
    }
    join_payload = result_payload[("join-heavy", 1000)]
    assert "owned_transfer_count" in join_payload
    assert "runtime_d2h_transfer_count" in join_payload
    assert "runtime_d2h_transfer_seconds" in join_payload
    assert join_payload["profile_mode"] == "lean"


def test_relation_bridge_consumer_pipeline_smoke() -> None:
    results = benchmark_pipeline_suite(
        suite="smoke",
        pipelines=("relation-bridge-consumer",),
    )
    assert len(results) == 1
    result = results[0]
    assert result.pipeline == "relation-bridge-consumer"
    assert result.scale == 1000
    assert result.status == "ok"

    trace = result.stages[0]
    stage_by_name = {stage["name"]: stage for stage in trace["stages"]}
    assert set(stage_by_name) == {
        "native_state_seed",
        "sjoin_relation_export",
        "native_semijoin_consumer",
        "public_joined_export_consumer",
    }
    assert (
        stage_by_name["native_semijoin_consumer"]["metadata"][
            "materialization_count_delta"
        ]
        == 0
    )
    assert (
        stage_by_name["native_semijoin_consumer"]["metadata"]["admissibility"]
        == "unique_label_semijoin"
    )
    assert (
        stage_by_name["native_semijoin_consumer"]["metadata"][
            "preserve_public_index"
        ]
        is True
    )
    expected_index_kinds = {"device-labels"} if has_gpu_runtime() else {
        "range",
        "host-labels",
    }
    assert (
        stage_by_name["native_semijoin_consumer"]["metadata"]["native_index_kind"]
        in expected_index_kinds
    )
    expected_pair_storage = "device" if has_gpu_runtime() else "host"
    assert (
        stage_by_name["sjoin_relation_export"]["metadata"]["pair_storage"]
        == expected_pair_storage
    )
    assert (
        stage_by_name["sjoin_relation_export"]["metadata"]["device_pair_request"]
        == "requested"
    )
    assert (
        stage_by_name["public_joined_export_consumer"]["metadata"][
            "materialization_count_delta"
        ]
        >= 1
    )
    assert stage_by_name["public_joined_export_consumer"]["metadata"]["results_match"] is True
    assert trace["metadata"]["admissible_shape"] == (
        "native-backed public relation export -> unique-label semijoin native frame"
    )


def test_grouped_reducer_pipeline_smoke() -> None:
    results = benchmark_pipeline_suite(
        suite="smoke",
        pipelines=("grouped-reducer",),
    )
    assert len(results) == 1
    result = results[0]
    assert result.pipeline == "grouped-reducer"
    assert result.scale == 1000
    if result.status == "deferred":
        return
    assert result.status == "ok"

    trace = result.stages[0]
    stage_by_name = {stage["name"]: stage for stage in trace["stages"]}
    assert set(stage_by_name) == {
        "build_dense_codes",
        "native_sum",
        "public_groupby_reference",
    }
    assert stage_by_name["native_sum"]["metadata"]["result_storage"] == "device"
    assert stage_by_name["native_sum"]["metadata"]["runtime_d2h_transfer_count_delta"] == 0
    assert stage_by_name["native_sum"]["metadata"]["materialization_count_delta"] == 0
    assert stage_by_name["public_groupby_reference"]["metadata"]["results_match"] is True
    assert (
        stage_by_name["public_groupby_reference"]["metadata"][
            "materialization_count_delta"
        ]
        >= 1
    )
    assert trace["metadata"]["admissible_shape"] == "dense-code NativeGrouped numeric sum"


def test_small_grouped_constructive_reduce_pipeline_smoke() -> None:
    results = benchmark_pipeline_suite(
        suite="smoke",
        pipelines=("small-grouped-constructive-reduce",),
    )
    assert len(results) == 1
    result = results[0]
    assert result.pipeline == "small-grouped-constructive-reduce"
    assert result.scale == 1000
    if result.status == "deferred":
        return
    assert result.status == "ok"

    trace = result.stages[0]
    stage_by_name = {stage["name"]: stage for stage in trace["stages"]}
    assert set(stage_by_name) == {
        "build_device_grouped_polygons",
        "native_grouped_union",
        "shapely_reference",
    }
    assert (
        stage_by_name["native_grouped_union"]["metadata"]["result_storage"]
        == "device"
    )
    assert (
        stage_by_name["native_grouped_union"]["metadata"][
            "used_many_small_batch"
        ]
        is True
    )
    assert (
        stage_by_name["native_grouped_union"]["metadata"][
            "materialization_count_delta"
        ]
        == 0
    )
    assert (
        stage_by_name["native_grouped_union"]["metadata"][
            "runtime_d2h_transfer_count_delta"
        ]
        <= 3
    )
    assert (
        stage_by_name["native_grouped_union"]["metadata"][
            "runtime_d2h_transfer_bytes_delta"
        ]
        <= 64
    )
    assert stage_by_name["shapely_reference"]["metadata"]["results_match"] is True
    assert (
        trace["metadata"]["admissible_shape"]
        == "owned device polygons + dense group offsets -> batched grouped constructive reduce"
    )


def test_relation_attribute_reducer_pipeline_smoke() -> None:
    results = benchmark_pipeline_suite(
        suite="smoke",
        pipelines=("relation-attribute-reducer",),
    )
    assert len(results) == 1
    result = results[0]
    assert result.pipeline == "relation-attribute-reducer"
    assert result.scale == 1000
    if result.status == "deferred":
        return
    assert result.status == "ok"

    trace = result.stages[0]
    stage_by_name = {stage["name"]: stage for stage in trace["stages"]}
    assert set(stage_by_name) == {
        "build_relation_inputs",
        "native_attribute_reduce",
        "public_groupby_reference",
    }
    assert stage_by_name["build_relation_inputs"]["metadata"]["pair_storage"] == "device"
    assert (
        stage_by_name["native_attribute_reduce"]["metadata"]["result_storage"]
        == "device"
    )
    assert (
        stage_by_name["native_attribute_reduce"]["metadata"][
            "runtime_d2h_transfer_count_delta"
        ]
        == 0
    )
    assert (
        stage_by_name["native_attribute_reduce"]["metadata"][
            "materialization_count_delta"
        ]
        == 0
    )
    assert stage_by_name["public_groupby_reference"]["metadata"]["results_match"] is True
    assert (
        stage_by_name["public_groupby_reference"]["metadata"][
            "materialization_count_delta"
        ]
        >= 1
    )
    assert trace["metadata"]["admissible_shape"] == (
        "device NativeRelation -> grouped right numeric attributes by left rows"
    )


def test_pipeline_profile_mode_controls_expensive_gpu_monitors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict] = []

    class _FakeProfiler:
        def __init__(self, **kwargs) -> None:
            captured.append(kwargs)

    monkeypatch.setattr(pipeline_module, "StageProfiler", _FakeProfiler)

    previous = pipeline_module._set_pipeline_profile_mode("lean")
    try:
        pipeline_module._stage_profiler(
            operation="pipeline.test",
            dataset="smoke",
            requested_runtime="gpu",
            selected_runtime="gpu",
            retain_gpu_trace=False,
            include_gpu_sparklines=False,
        )
    finally:
        pipeline_module._set_pipeline_profile_mode(previous)

    lean_kwargs = captured[-1]
    assert lean_kwargs["gpu_sampler"].available is False
    assert lean_kwargs["gpu_event_timer_factory"] is pipeline_module._NoopGpuEventTimer
    assert lean_kwargs["retain_gpu_trace"] is False
    assert lean_kwargs["include_gpu_sparklines"] is False

    previous = pipeline_module._set_pipeline_profile_mode("audit")
    try:
        pipeline_module._stage_profiler(
            operation="pipeline.test",
            dataset="smoke",
            requested_runtime="gpu",
            selected_runtime="gpu",
            retain_gpu_trace=True,
            include_gpu_sparklines=True,
        )
    finally:
        pipeline_module._set_pipeline_profile_mode(previous)

    audit_kwargs = captured[-1]
    assert "gpu_sampler" not in audit_kwargs
    assert "gpu_event_timer_factory" not in audit_kwargs
    assert audit_kwargs["retain_gpu_trace"] is True
    assert audit_kwargs["include_gpu_sparklines"] is True
    assert pipeline_module._resolve_pipeline_profile_mode(
        "lean",
        include_gpu_sparklines=True,
    ) == "audit"
    assert (
        pipeline_module._deferred_raster_pipeline(1000, profile_mode="audit").profile_mode
        == "audit"
    )


def test_pipeline_audit_counts_runtime_materialization_events() -> None:
    from vibespatial.runtime.materialization import (
        MaterializationBoundary,
        clear_materialization_events,
        record_materialization_event,
    )

    clear_materialization_events()
    audit = pipeline_module._OwnedAudit()

    record_materialization_event(
        surface="test.pipeline",
        boundary=MaterializationBoundary.USER_EXPORT,
        operation="export",
        reason="test materialization",
    )

    assert audit.materialization_count == 1
    assert audit.snapshot()[1] == 1
    clear_materialization_events()


def test_stage_profiler_attaches_materialization_event_context() -> None:
    from vibespatial.runtime.materialization import (
        MaterializationBoundary,
        clear_materialization_events,
        record_materialization_event,
    )

    clear_materialization_events()
    profiler = pipeline_module._stage_profiler(
        operation="pipeline.zero-transfer",
        dataset="full",
        requested_runtime="gpu",
        selected_runtime="gpu",
    )

    with profiler.stage(
        "read_input",
        category="setup",
        device="gpu",
    ):
        record_materialization_event(
            surface="vibespatial.api.NativeTabularResult.to_geodataframe",
            boundary=MaterializationBoundary.USER_EXPORT,
            operation="native_tabular_to_geodataframe",
            reason="test export",
        )

    trace = profiler.finish()
    metadata = trace.stages[0].metadata
    event = metadata["materialization_events"][0]

    assert metadata["materialization_count_delta"] == 1
    assert event["pipeline"] == "pipeline.zero-transfer"
    assert event["dataset"] == "full"
    assert event["stage"] == "read_input"
    assert event["stage_category"] == "setup"
    assert event["surface"] == "vibespatial.api.NativeTabularResult.to_geodataframe"
    clear_materialization_events()


def test_stage_profiler_attaches_runtime_d2h_transfer_context() -> None:
    import numpy as np

    from vibespatial.cuda._runtime import (
        _notify_runtime_d2h_transfer,
        reset_d2h_transfer_count,
    )

    reset_d2h_transfer_count()
    profiler = pipeline_module._stage_profiler(
        operation="pipeline.zero-transfer",
        dataset="scale-1",
        requested_runtime="gpu",
        selected_runtime="gpu",
    )

    device_like = np.empty(4, dtype=np.int32)
    with profiler.stage(
        "read_input",
        category="setup",
        device="gpu",
    ):
        _notify_runtime_d2h_transfer(
            device_like,
            trigger="unit-test",
            reason="test runtime transfer",
            elapsed_seconds=0.125,
        )

    trace = profiler.finish()
    metadata = trace.stages[0].metadata
    event = metadata["runtime_d2h_transfer_events"][0]

    assert metadata["runtime_d2h_transfer_count_delta"] == 1
    assert metadata["runtime_d2h_transfer_bytes_delta"] == device_like.nbytes
    assert event["pipeline"] == "pipeline.zero-transfer"
    assert event["dataset"] == "scale-1"
    assert event["stage"] == "read_input"
    assert event["stage_category"] == "setup"
    assert event["trigger"] == "unit-test"
    assert event["reason"] == "test runtime transfer"
    assert event["item_count"] == 4
    assert event["bytes_transferred"] == device_like.nbytes
    reset_d2h_transfer_count()


def test_benchmark_pipeline_suite_precompiles_full_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[float] = []

    def _fake_precompile_all(timeout: float = 120.0):
        calls.append(timeout)
        return {"cccl": {}, "nvrtc": {}, "cccl_cold": [], "nvrtc_cold": []}

    monkeypatch.setattr(
        "vibespatial.cuda.cccl_precompile.precompile_all",
        _fake_precompile_all,
    )
    monkeypatch.setattr(pipeline_module, "pipeline_scales", lambda _suite: ())

    assert benchmark_pipeline_suite(suite="smoke", pipelines=("join-heavy",)) == []
    assert calls == [120.0]


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


def test_actual_array_device_label_requires_cuda_array_interface() -> None:
    class _FakeCpuArray:
        device = "cpu"

    class _FakeGpuArray:
        __cuda_array_interface__ = {
            "shape": (1,),
            "strides": None,
            "typestr": "<i4",
            "data": (0, False),
            "version": 3,
        }

    assert _actual_array_device_label(_FakeCpuArray()) == "cpu"
    assert _actual_array_device_label(_FakeGpuArray()) == "gpu"


def test_predicate_pipeline_reports_cpu_planner_when_runtime_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(pipeline_module, "has_gpu_runtime", lambda: False)

    result = _profile_predicate_pipeline(scale=8)

    # The benchmark rail reports actual execution in selected_runtime. With a
    # real GPU still present underneath, lower layers can remain hybrid even if
    # the pipeline planner is forced to CPU here.
    assert result.planner_selected_runtime == "cpu"
    assert result.selected_runtime in {"cpu", "hybrid"}
    trace = result.stages[0]
    assert trace["metadata"]["planner_selected_runtime"] == "cpu"


def test_predicate_pipeline_reads_precomputed_geojson_bytes() -> None:
    result = _profile_predicate_pipeline(scale=8)

    trace = result.stages[0]
    read_stage = trace["stages"][0]

    assert read_stage["name"] == "read_geojson"
    assert read_stage["metadata"]["source_kind"] == "bytes"
    assert "GeoJSON bytes ingest" in result.notes


def test_extract_gpu_util_ignores_nvml_only_cpu_traces() -> None:
    stages = (
        {
            "stages": [
                {
                    "name": "cpu_stage",
                    "device": "cpu",
                    "metadata": {
                        "gpu_device_name": "fake-gpu",
                        "gpu_utilization_pct_avg": 12.0,
                        "gpu_utilization_pct_max": 24.0,
                        "gpu_memory_utilization_pct_avg": 30.0,
                        "gpu_vram_used_bytes_max": 100,
                        "gpu_vram_total_bytes": 1000,
                        "gpu_util_sparkline": "||",
                    },
                }
            ]
        },
    )

    assert _extract_gpu_util(stages, selected_runtime="cpu") is None


def test_extract_gpu_util_requires_gpu_labeled_stage() -> None:
    stages = (
        {
            "stages": [
                {
                    "name": "gpu_stage",
                    "device": "gpu",
                    "metadata": {
                        "gpu_device_name": "fake-gpu",
                        "gpu_utilization_pct_avg": 12.0,
                        "gpu_utilization_pct_max": 24.0,
                        "gpu_memory_utilization_pct_avg": 30.0,
                        "gpu_vram_used_bytes_max": 100,
                        "gpu_vram_total_bytes": 1000,
                        "gpu_util_sparkline": "||",
                    },
                }
            ]
        },
    )

    gpu_util = _extract_gpu_util(stages, selected_runtime="gpu")

    assert gpu_util is not None
    assert gpu_util.device_name == "fake-gpu"


def test_render_gpu_sparkline_report_prefers_cuda_event_timing_for_gpu_stage() -> None:
    result = PipelineBenchmarkResult(
        pipeline="join-heavy",
        scale=1000,
        status="ok",
        elapsed_seconds=0.0123,
        selected_runtime="gpu",
        planner_selected_runtime="gpu",
        output_rows=1,
        transfer_count=0,
        materialization_count=0,
        fallback_event_count=0,
        peak_device_memory_bytes=None,
        stages=(
            {
                "stages": [
                    {
                        "name": "sjoin_query",
                        "device": "gpu",
                        "elapsed_seconds": 0.0123,
                        "metadata": {
                            "elapsed_display": "12.3ms",
                            "gpu_event_elapsed_seconds": 0.000245,
                            "gpu_event_elapsed_display": "245us",
                            "gpu_util_sparkline": "0% |▁| 4%",
                            "gpu_memory_util_sparkline": "0% |▁| 0%",
                            "gpu_vram_sparkline": "10MiB |▁| 10MiB",
                            "gpu_substage_timings": {
                                "coerce_left_s": 0.000002,
                                "strategy": "fused",
                            },
                        },
                    }
                ]
            },
        ),
    )

    report = render_gpu_sparkline_report([result])

    assert "stage=sjoin_query gpu=245us wall=12.3ms" in report
    assert "coerce_left_s=2us" in report
    assert "strategy=fused" in report


def test_from_shapely_safe_flattens_supported_geometry_collections() -> None:
    owned = _from_shapely_safe([
        GeometryCollection([Point(0, 0), LineString([(0, 0), (1, 1)])]),
        GeometryCollection([]),
        None,
    ])

    restored = owned.to_shapely()

    assert len(restored) == 2
    assert restored[0].equals(Point(0, 0))
    assert restored[1].equals(LineString([(0, 0), (1, 1)]))


def test_vegetation_corridor_smoke() -> None:
    pytest.importorskip("pylibcudf")
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
    pytest.importorskip("pylibcudf")
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
    pytest.importorskip("pylibcudf")
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
    pytest.importorskip("pylibcudf")
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
    pytest.importorskip("pylibcudf")
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


def test_provenance_rewrite_pipeline_smoke() -> None:
    results = benchmark_pipeline_suite(suite="smoke", pipelines=("provenance-rewrite",))
    assert len(results) == 1
    result = results[0]
    assert result.pipeline == "provenance-rewrite"
    assert result.scale == 1000
    assert result.status == "ok"
    assert result.rewrite_event_count > 0

    trace = result.stages[0]
    stage_names = [stage["name"] for stage in trace["stages"]]
    assert "generate_points" in stage_names
    assert "buffer_intersects_rewrite" in stage_names
    assert "buffer_intersects_naive" in stage_names
    assert "compare" in stage_names

    # Rewrite stage should have rewrite_count > 0
    rewrite_stage = next(s for s in trace["stages"] if s["name"] == "buffer_intersects_rewrite")
    assert rewrite_stage["metadata"]["rewrite_count"] > 0

    # Naive stage should have rewrite_count == 0
    naive_stage = next(s for s in trace["stages"] if s["name"] == "buffer_intersects_naive")
    assert naive_stage["metadata"]["rewrite_count"] == 0

    # Results should match
    compare_stage = next(s for s in trace["stages"] if s["name"] == "compare")
    assert compare_stage["metadata"]["results_match"] is True
