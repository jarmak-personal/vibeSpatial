from __future__ import annotations

import inspect
import json
from pathlib import Path

import vibespatial.bench.runner as runner_module
from vibespatial.bench.catalog import (
    _OPERATION_REGISTRY,
    OperationParameterSpec,
    benchmark_operation,
    ensure_operations_loaded,
    get_operation,
    list_operations,
    resolve_operation_args,
)
from vibespatial.bench.cli import main as vsbench_main
from vibespatial.bench.runner import run_operation
from vibespatial.bench.schema import (
    BenchmarkResult,
    GpuUtilSummary,
    KernelTimingSummary,
    SuiteResult,
    TimingSummary,
    TransferSummary,
    benchmark_result_from_dict,
    timing_from_samples,
)
from vibespatial.bench.suites import SUITES


def test_operation_parameter_spec_parses_supported_types() -> None:
    assert OperationParameterSpec("count", "int", "count").parse("7") == 7
    assert OperationParameterSpec("ratio", "float", "ratio").parse("0.25") == 0.25
    assert OperationParameterSpec("enabled", "bool", "enabled").parse("true") is True
    assert OperationParameterSpec(
        "kind",
        "choice",
        "kind",
        choices=("line", "polygon"),
    ).parse("polygon") == "polygon"
    assert OperationParameterSpec(
        "rect",
        "float_list",
        "rect",
        arity=4,
    ).parse("1,2,3,4") == (1.0, 2.0, 3.0, 4.0)


def test_resolve_operation_args_applies_defaults_and_rejects_unknown_keys() -> None:
    ensure_operations_loaded()
    spec = get_operation("clip-rect")

    resolved = resolve_operation_args(spec, ["kind=polygon", "rect=1,2,3,4"])

    assert resolved["kind"] == "polygon"
    assert resolved["rect"] == (1.0, 2.0, 3.0, 4.0)

    try:
        resolve_operation_args(spec, ["unknown=1"])
    except ValueError as exc:
        assert "unknown operation arg" in str(exc)
    else:
        raise AssertionError("expected resolve_operation_args to reject unknown keys")


def test_operation_spec_to_dict_includes_parameters() -> None:
    ensure_operations_loaded()
    spec = get_operation("bounds-pairs")

    payload = spec.to_dict()

    assert any(param["name"] == "dataset" for param in payload["parameters"])
    assert any(param["name"] == "tile_size" for param in payload["parameters"])
    dataset_param = next(param for param in payload["parameters"] if param["name"] == "dataset")
    assert dataset_param["choices"] == ["uniform", "skewed", "both"]
    assert payload["public_api"] is False


def test_spatial_query_spec_exposes_overlap_ratio_parameter() -> None:
    ensure_operations_loaded()
    spec = get_operation("spatial-query")

    payload = spec.to_dict()

    assert any(param["name"] == "overlap_ratio" for param in payload["parameters"])


def test_gpu_dissolve_spec_defaults_to_public_coverage_method() -> None:
    ensure_operations_loaded()
    spec = get_operation("gpu-dissolve")

    payload = spec.to_dict()

    method = next(param for param in payload["parameters"] if param["name"] == "method")
    assert method["default"] == "coverage"
    assert method["choices"] == ["coverage", "unary", "disjoint_subset"]


def test_list_operations_hides_internal_diagnostics_by_default() -> None:
    ensure_operations_loaded()

    public_names = {spec.name for spec in list_operations()}
    all_names = {spec.name for spec in list_operations(include_internal=True)}

    assert "bounds" in public_names
    assert "gpu-pip" not in public_names
    assert "bounds-pairs" not in public_names
    assert "gpu-pip" in all_names
    assert "bounds-pairs" in all_names


def test_vsbench_list_operations_include_internal_flag(capsys) -> None:
    exit_code = vsbench_main(["list", "operations", "--json"])
    captured = capsys.readouterr()
    public_payload = json.loads(captured.out)
    public_names = {item["name"] for item in public_payload}

    exit_code_internal = vsbench_main(["list", "operations", "--json", "--include-internal"])
    captured_internal = capsys.readouterr()
    all_payload = json.loads(captured_internal.out)
    all_names = {item["name"] for item in all_payload}

    assert exit_code == 0
    assert exit_code_internal == 0
    assert "bounds" in public_names
    assert "gpu-pip" not in public_names
    assert "gpu-pip" in all_names


def test_predefined_suites_reference_public_operations_only() -> None:
    ensure_operations_loaded()

    private_suite_ops = {
        op_name
        for suite in SUITES.values()
        for op_name in suite.operations
        if op_name in _OPERATION_REGISTRY and not _OPERATION_REGISTRY[op_name].public_api
    }
    stale_suite_ops = {
        op_name
        for suite in SUITES.values()
        for op_name in suite.operations
        if op_name not in _OPERATION_REGISTRY
    }

    assert private_suite_ops == set()
    assert stale_suite_ops == set()
    assert all(not suite.kernels for suite in SUITES.values())


def test_public_benchmark_operations_do_not_force_private_paths() -> None:
    ensure_operations_loaded()

    forbidden_tokens = (
        "load_owned(",
        "decode_wkb_owned",
        "encode_wkb_owned",
        "overlay_intersection_owned",
        "benchmark_bounds_pairs",
        "benchmark_segment_",
        "build_flat_spatial_index",
        "query_spatial_index",
        "from vibespatial.kernels",
        "vibespatial.testing.mixed_layouts",
    )
    violations: dict[str, list[str]] = {}
    for spec in list_operations():
        if not spec.callable.__module__.startswith("vibespatial.bench.operations"):
            continue
        source = inspect.getsource(spec.callable)
        hits = [token for token in forbidden_tokens if token in source]
        if hits:
            violations[spec.name] = hits

    assert violations == {}


def test_public_benchmark_scripts_do_not_force_private_paths() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    checked = [
        *(
            path
            for path in (repo_root / "benchmarks" / "shootout").rglob("*.py")
            if not path.name.startswith("_")
        ),
        *(
            path
            for path in (repo_root / "scripts").glob("benchmark_*.py")
        ),
    ]
    forbidden_tokens = (
        "read_osm_pbf",
        "from vibespatial import benchmark_",
        "vibespatial.bench.io_benchmark_rails",
        "from vibespatial.spatial",
        "from vibespatial.predicates",
        "from vibespatial.constructive",
        "from vibespatial.overlay",
        "from vibespatial.kernels",
        "build_owned_spatial_index",
        "evaluate_binary_predicate",
        "_decode_pylibcudf",
        "_decode_geoparquet",
        "load_owned(",
        ".to_owned(",
        "from_shapely_geometries",
        "clip_by_rect_owned",
        "point_buffer_owned",
        "read_geoparquet_owned",
        "read_geojson_owned",
        "read_shapefile_owned",
    )

    violations = {
        str(path.relative_to(repo_root)): [token for token in forbidden_tokens if token in path.read_text()]
        for path in checked
    }
    violations = {path: hits for path, hits in violations.items() if hits}

    assert violations == {}


def test_run_operation_records_operation_args_in_metadata() -> None:
    @benchmark_operation(
        name="test-opargs-runner",
        description="test helper",
        category="misc",
        parameters=(
            OperationParameterSpec(
                name="tile_size",
                value_type="int",
                description="Tile size",
                default=64,
            ),
        ),
    )
    def _bench(
        *,
        scale: int,
        repeat: int,
        compare: str | None,
        precision: str,
        input_format: str,
        nvtx: bool,
        gpu_sparkline: bool,
        trace: bool,
        **kwargs: object,
    ) -> BenchmarkResult:
        return BenchmarkResult(
            operation="test-opargs-runner",
            tier=1,
            scale=scale,
            geometry_type="point",
            precision=precision,
            status="pass",
            status_reason="ok",
            timing=timing_from_samples([0.001]),
            metadata={"tile_size_seen": kwargs["tile_size"]},
        )

    try:
        result = run_operation(
            "test-opargs-runner",
            scale=10,
            repeat=1,
            operation_args={"tile_size": 128},
        )
    finally:
        _OPERATION_REGISTRY.pop("test-opargs-runner", None)

    assert result.metadata["tile_size_seen"] == 128
    assert result.metadata["operation_args"] == {"tile_size": 128}


def test_run_operation_aggregates_repeat_samples() -> None:
    samples = iter((0.0005, 0.003, 0.001, 0.002))
    baseline_samples = iter((0.005, 0.030, 0.010, 0.020))

    @benchmark_operation(
        name="test-repeat-runner",
        description="test helper",
        category="misc",
    )
    def _bench(
        *,
        scale: int,
        repeat: int,
        compare: str | None,
        precision: str,
        input_format: str,
        nvtx: bool,
        gpu_sparkline: bool,
        trace: bool,
    ) -> BenchmarkResult:
        elapsed = next(samples)
        baseline_elapsed = next(baseline_samples)
        return BenchmarkResult(
            operation="test-repeat-runner",
            tier=1,
            scale=scale,
            geometry_type="point",
            precision=precision,
            status="pass",
            status_reason="ok",
            timing=timing_from_samples([elapsed]),
            baseline_name="baseline",
            baseline_timing=timing_from_samples([baseline_elapsed]),
            speedup=baseline_elapsed / elapsed,
        )

    try:
        result = run_operation("test-repeat-runner", scale=10, repeat=3)
    finally:
        _OPERATION_REGISTRY.pop("test-repeat-runner", None)

    assert result.timing.sample_count == 3
    assert result.timing.median_seconds == 0.002
    assert result.baseline_timing is not None
    assert result.baseline_timing.sample_count == 3
    assert result.baseline_timing.median_seconds == 0.020
    assert result.speedup == 10.0
    assert result.metadata["repeat"] == 3
    assert result.metadata["sample_seconds"] == [0.003, 0.001, 0.002]
    assert result.metadata["baseline_sample_seconds"] == [0.030, 0.010, 0.020]


def test_vsbench_run_accepts_generic_operation_args(capsys) -> None:
    @benchmark_operation(
        name="test-opargs-cli",
        description="test helper",
        category="misc",
        parameters=(
            OperationParameterSpec(
                name="enabled",
                value_type="bool",
                description="Enabled flag",
                default=False,
            ),
        ),
    )
    def _bench(
        *,
        scale: int,
        repeat: int,
        compare: str | None,
        precision: str,
        input_format: str,
        nvtx: bool,
        gpu_sparkline: bool,
        trace: bool,
        **kwargs: object,
    ) -> BenchmarkResult:
        return BenchmarkResult(
            operation="test-opargs-cli",
            tier=1,
            scale=scale,
            geometry_type="point",
            precision=precision,
            status="pass",
            status_reason="ok",
            timing=timing_from_samples([0.001]),
            metadata={"enabled_seen": kwargs["enabled"]},
        )

    try:
        exit_code = vsbench_main(
            ["run", "test-opargs-cli", "--arg", "enabled=true", "--json", "--quiet"]
        )
    finally:
        _OPERATION_REGISTRY.pop("test-opargs-cli", None)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["metadata"]["enabled_seen"] is True
    assert payload["metadata"]["operation_args"] == {"enabled": True}


def test_vsbench_list_operations_json_includes_parameter_schema(capsys) -> None:
    exit_code = vsbench_main(["list", "operations", "--json", "--category", "constructive"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    clip_rect = next(item for item in payload if item["name"] == "clip-rect")

    assert exit_code == 0
    assert clip_rect["parameters"] == [
        {
            "name": "kind",
            "type": "choice",
            "description": "Geometry workload to benchmark",
            "default": "line",
            "choices": ["line", "polygon"],
        },
        {
            "name": "rect",
            "type": "float_list",
            "description": "Custom clip rectangle as xmin,ymin,xmax,ymax",
            "arity": 4,
        },
    ]


def test_vsbench_run_accepts_exact_rows_override(capsys) -> None:
    @benchmark_operation(
        name="test-opargs-rows",
        description="test helper",
        category="misc",
    )
    def _bench(
        *,
        scale: int,
        repeat: int,
        compare: str | None,
        precision: str,
        input_format: str,
        nvtx: bool,
        gpu_sparkline: bool,
        trace: bool,
        **kwargs: object,
    ) -> BenchmarkResult:
        return BenchmarkResult(
            operation="test-opargs-rows",
            tier=1,
            scale=scale,
            geometry_type="point",
            precision=precision,
            status="pass",
            status_reason="ok",
            timing=timing_from_samples([0.001]),
        )

    try:
        exit_code = vsbench_main(
            ["run", "test-opargs-rows", "--rows", "1234", "--json", "--quiet"]
        )
    finally:
        _OPERATION_REGISTRY.pop("test-opargs-rows", None)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["scale"] == 1234


def test_benchmark_result_from_dict_round_trips_optional_sections() -> None:
    result = BenchmarkResult(
        operation="test-roundtrip",
        tier=2,
        scale=1000,
        geometry_type="point",
        precision="fp64",
        status="pass",
        status_reason="ok",
        timing=TimingSummary(
            mean_seconds=0.1,
            median_seconds=0.1,
            min_seconds=0.09,
            max_seconds=0.11,
            stddev_seconds=0.01,
            sample_count=3,
        ),
        baseline_name="geopandas",
        baseline_timing=timing_from_samples([0.3, 0.4, 0.5]),
        speedup=4.0,
        transfers=TransferSummary(
            d2h_count=1,
            h2d_count=2,
            total_bytes=128,
            total_seconds=0.001,
            offramps=1,
        ),
        gpu_util=GpuUtilSummary(
            device_name="test-gpu",
            sm_utilization_pct_avg=25.0,
            sm_utilization_pct_max=50.0,
            memory_utilization_pct_avg=10.0,
            vram_used_bytes_max=1024,
            vram_total_bytes=2048,
            sparkline="|",
        ),
        kernel_timing=KernelTimingSummary(
            gpu_time_seconds=0.01,
            cpu_time_seconds=0.02,
            bandwidth_gb_per_second=123.0,
            bandwidth_pct_of_peak=12.0,
            l2_cache_flushed=True,
            throttle_detected=False,
            convergence_met=True,
        ),
        tier_gate_threshold=1.1,
        tier_gate_passed=True,
        input_format="geojson",
        read_seconds=0.012,
        stages=({"name": "stage", "device": "gpu"},),
        metadata={"k": "v"},
    )

    restored = benchmark_result_from_dict(result.to_dict())

    assert restored == result


def test_vsbench_suite_defaults_to_isolated_subprocesses(monkeypatch, capsys) -> None:
    calls: list[dict[str, object]] = []

    def _fake_run_suite(level: str, **kwargs: object) -> SuiteResult:
        calls.append({"level": level, **kwargs})
        return SuiteResult(
            suite_name=level,
            results=[
                BenchmarkResult(
                    operation="bounds",
                    tier=1,
                    scale=1000,
                    geometry_type="point",
                    precision="auto",
                    status="pass",
                    status_reason="ok",
                    timing=timing_from_samples([0.001]),
                )
            ],
            metadata={"isolated": kwargs["isolated"]},
        )

    monkeypatch.setattr(runner_module, "run_suite", _fake_run_suite)

    exit_code = vsbench_main(["suite", "smoke", "--json", "--quiet"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["metadata"]["isolated"] is True
    assert calls[0]["isolated"] is True
    assert calls[0]["item_timeout"] == 600


def test_vsbench_suite_accepts_in_process_opt_out(monkeypatch, capsys) -> None:
    calls: list[dict[str, object]] = []

    def _fake_run_suite(level: str, **kwargs: object) -> SuiteResult:
        calls.append({"level": level, **kwargs})
        return SuiteResult(suite_name=level, results=[])

    monkeypatch.setattr(runner_module, "run_suite", _fake_run_suite)

    exit_code = vsbench_main(
        ["suite", "smoke", "--in-process", "--item-timeout", "7", "--json", "--quiet"]
    )

    capsys.readouterr()
    assert exit_code == 0
    assert calls[0]["isolated"] is False
    assert calls[0]["item_timeout"] == 7


def test_isolated_benchmark_command_parses_child_suite(monkeypatch) -> None:
    child_suite = SuiteResult(
        suite_name="pipeline:join-heavy",
        results=[
            BenchmarkResult(
                operation="join-heavy",
                tier=1,
                scale=1000,
                geometry_type="mixed",
                precision="auto",
                status="pass",
                status_reason="ok",
                timing=timing_from_samples([0.001]),
            )
        ],
    )

    monkeypatch.setattr(
        runner_module,
        "_run_child_process",
        lambda command, *, item_timeout: (0, child_suite.to_json(), "", False),
    )
    monkeypatch.setattr(runner_module, "_gpu_compute_apps", lambda: [])

    results = runner_module._run_isolated_benchmark_command(
        ["python", "-m", "vibespatial.bench.cli", "pipeline", "join-heavy"],
        operation="join-heavy",
        tier=1,
        scale=1000,
        geometry_type="mixed",
        precision="auto",
        item_timeout=10,
    )

    assert len(results) == 1
    assert results[0].operation == "join-heavy"
    assert results[0].metadata["isolated_subprocess"] is True
    assert results[0].metadata["subprocess_returncode"] == 0


def test_isolated_benchmark_command_reports_timeout(monkeypatch) -> None:
    monkeypatch.setattr(
        runner_module,
        "_run_child_process",
        lambda command, *, item_timeout: (None, "", "still running", True),
    )
    monkeypatch.setattr(
        runner_module,
        "_gpu_compute_apps",
        lambda: [{"pid": "123", "process_name": "python", "used_memory_mib": "10"}],
    )

    results = runner_module._run_isolated_benchmark_command(
        ["python", "-m", "vibespatial.bench.cli", "run", "bounds"],
        operation="bounds",
        tier=1,
        scale=1000,
        geometry_type="unknown",
        precision="auto",
        item_timeout=1,
    )

    assert len(results) == 1
    assert results[0].status == "error"
    assert "timed out" in results[0].status_reason
    assert results[0].metadata["subprocess_timeout_seconds"] == 1
    assert results[0].metadata["gpu_compute_apps_after_subprocess"]
