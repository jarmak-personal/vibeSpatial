from __future__ import annotations

import json

from vibespatial.bench.catalog import (
    _OPERATION_REGISTRY,
    OperationParameterSpec,
    benchmark_operation,
    ensure_operations_loaded,
    get_operation,
    resolve_operation_args,
)
from vibespatial.bench.cli import main as vsbench_main
from vibespatial.bench.runner import run_operation
from vibespatial.bench.schema import BenchmarkResult, timing_from_samples


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


def test_spatial_query_spec_exposes_overlap_ratio_parameter() -> None:
    ensure_operations_loaded()
    spec = get_operation("spatial-query")

    payload = spec.to_dict()

    assert any(param["name"] == "overlap_ratio" for param in payload["parameters"])


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
