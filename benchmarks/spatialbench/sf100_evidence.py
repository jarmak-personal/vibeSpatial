#!/usr/bin/env python3
"""Validate the frozen SF100 comparator and candidate result artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

QUERY_NAMES = tuple(f"q{index}" for index in range(1, 13))
ALLOWED_COLUMN_KINDS = {"datetime", "float", "int", "string"}
DEFAULT_COMPARATOR = Path(
    "benchmark_results/spatialbench/sf100/accepted-geopandas-comparator.json"
)


class EvidenceError(RuntimeError):
    """Raised when an SF100 evidence packet is incomplete or inconsistent."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"JSON artifact must contain an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise EvidenceError(f"cannot hash evidence file {path}: {exc}") from exc
    return digest.hexdigest()


def _repo_path(repo_root: Path, relative_path: str) -> Path:
    path = (repo_root / relative_path).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise EvidenceError(f"evidence path escapes repository: {relative_path}") from exc
    return path


def _verify_file_identity(repo_root: Path, packet: dict[str, Any], label: str) -> Path:
    relative_path = packet.get("path")
    expected_hash = packet.get("sha256")
    if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
        raise EvidenceError(f"{label} must provide string path and sha256 fields")
    path = _repo_path(repo_root, relative_path)
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        raise EvidenceError(
            f"{label} hash mismatch for {relative_path}: "
            f"expected {expected_hash}, found {actual_hash}"
        )
    return path


def _suite_by_engine(artifact: dict[str, Any], engine: str) -> dict[str, Any]:
    matches = [
        suite
        for suite in artifact.get("results", [])
        if isinstance(suite, dict) and suite.get("engine") == engine
    ]
    if len(matches) != 1:
        raise EvidenceError(
            f"artifact must contain exactly one {engine!r} suite; found {len(matches)}"
        )
    return matches[0]


def _query_map(suite: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = suite.get("results")
    if not isinstance(rows, list):
        raise EvidenceError("benchmark suite is missing its results list")
    mapped: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("query"), str):
            raise EvidenceError("benchmark query entries must be JSON objects with a query")
        query = row["query"]
        if query in mapped:
            raise EvidenceError(f"duplicate benchmark result for {query}")
        mapped[query] = row
    if set(mapped) != set(QUERY_NAMES):
        missing = sorted(set(QUERY_NAMES) - set(mapped))
        extra = sorted(set(mapped) - set(QUERY_NAMES))
        raise EvidenceError(f"query set mismatch: missing={missing}, extra={extra}")
    return mapped


def _median(values: Any, label: str) -> float:
    if not isinstance(values, list) or len(values) != 3:
        raise EvidenceError(f"{label} must contain exactly three measured samples")
    try:
        return float(statistics.median(float(value) for value in values))
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{label} contains a non-numeric sample") from exc


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.reader(source)
            header = next(reader)
            rows = list(reader)
    except (OSError, StopIteration, csv.Error) as exc:
        raise EvidenceError(f"cannot read result CSV {path}: {exc}") from exc
    if len(header) != len(set(header)):
        raise EvidenceError(f"result CSV has duplicate columns: {path}")
    if any(len(row) != len(header) for row in rows):
        raise EvidenceError(f"result CSV has a ragged row: {path}")
    return header, rows


def validate_comparator(comparator_path: Path, repo_root: Path) -> dict[str, Any]:
    """Validate every derivation input and output oracle in the comparator."""
    comparator = _load_json(comparator_path)
    if comparator.get("schema_version") != 1:
        raise EvidenceError("unsupported comparator schema_version")
    if comparator.get("kind") != "spatialbench_sf100_geopandas_optimized":
        raise EvidenceError("unexpected comparator kind")
    if comparator.get("status") != "accepted_immutable":
        raise EvidenceError("comparator is not marked accepted_immutable")

    identity_sources = comparator.get("identity_sources")
    if not isinstance(identity_sources, list) or len(identity_sources) != 3:
        raise EvidenceError("comparator must identify its three provenance sources")
    for index, source in enumerate(identity_sources):
        _verify_file_identity(repo_root, source, f"identity source {index}")

    workload = comparator.get("workload")
    if not isinstance(workload, dict) or workload.get("query_count") != 12:
        raise EvidenceError("comparator workload must declare all 12 queries")
    for index, source in enumerate(workload.get("sources", [])):
        _verify_file_identity(repo_root, source, f"workload source {index}")
    if len(workload.get("sources", [])) != 3:
        raise EvidenceError("comparator must identify exactly three query source files")

    environment = comparator.get("environment")
    if not isinstance(environment, dict):
        raise EvidenceError("comparator environment packet is missing")
    _verify_file_identity(repo_root, environment.get("lock", {}), "package lock")
    packages = environment.get("packages")
    if not isinstance(packages, dict) or not packages:
        raise EvidenceError("comparator package versions are missing")
    for package, expected_version in packages.items():
        try:
            actual_version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise EvidenceError(f"required comparator package is not installed: {package}") from exc
        if actual_version != expected_version:
            raise EvidenceError(
                f"comparator package mismatch for {package}: "
                f"expected {expected_version}, found {actual_version}"
            )

    derivation = comparator.get("derivation")
    if not isinstance(derivation, dict) or derivation.get("replacement_query") != "q5":
        raise EvidenceError("comparator must declare the checked-in Q5 replacement")
    base_path = _verify_file_identity(repo_root, derivation.get("base", {}), "base timing")
    replacement_path = _verify_file_identity(
        repo_root, derivation.get("replacement", {}), "Q5 replacement"
    )
    base_suite = _suite_by_engine(_load_json(base_path), "geopandas_optimized")
    base_queries = _query_map(base_suite)
    replacement = _load_json(replacement_path).get("q5")
    if not isinstance(replacement, dict):
        raise EvidenceError("Q5 replacement artifact has no q5 evidence packet")

    measurement = comparator.get("measurement")
    expected_measurement = {
        "isolated_process_per_engine_query": True,
        "warmup_runs": 1,
        "measured_runs": 3,
        "statistic": "median",
        "scale_factor": 100,
        "timed_boundary": "scan_compute_public_result",
        "result_serialization_timed": False,
    }
    if measurement != expected_measurement:
        raise EvidenceError("comparator measurement contract is incomplete or changed")

    queries = comparator.get("queries")
    if not isinstance(queries, dict) or set(queries) != set(QUERY_NAMES):
        raise EvidenceError("comparator query packet must contain exactly q1-q12")

    expected_total = 0.0
    for query in QUERY_NAMES:
        packet = queries[query]
        if not isinstance(packet, dict):
            raise EvidenceError(f"comparator {query} packet must be an object")
        source = base_queries[query]
        expected_seconds = float(source["time_seconds"])
        expected_runs = source["run_times_seconds"]
        if query == "q5":
            expected_seconds = float(replacement["optimized_geopandas_seconds"])
            expected_runs = replacement["optimized_geopandas_runs_seconds"]
        if float(packet.get("time_seconds", -1)) != expected_seconds:
            raise EvidenceError(f"{query} time does not match its checked-in source evidence")
        if packet.get("run_times_seconds") != expected_runs:
            raise EvidenceError(f"{query} samples do not match their checked-in source evidence")
        if _median(expected_runs, f"{query} samples") != expected_seconds:
            raise EvidenceError(f"{query} declared time is not its sample median")
        if packet.get("row_count") != source.get("row_count"):
            raise EvidenceError(f"{query} row count differs from the accepted base artifact")

        columns = packet.get("columns")
        if not isinstance(columns, list) or not columns:
            raise EvidenceError(f"{query} comparison schema is missing")
        names = [column.get("name") for column in columns if isinstance(column, dict)]
        kinds = [column.get("kind") for column in columns if isinstance(column, dict)]
        if len(names) != len(columns) or len(names) != len(set(names)):
            raise EvidenceError(f"{query} comparison schema has invalid column names")
        if any(kind not in ALLOWED_COLUMN_KINDS for kind in kinds):
            raise EvidenceError(f"{query} comparison schema has an unsupported column kind")

        oracle_path = _verify_file_identity(repo_root, packet.get("oracle", {}), f"{query} oracle")
        header, rows = _read_csv(oracle_path)
        if header != names:
            raise EvidenceError(f"{query} oracle columns differ from the comparison schema")
        if len(rows) != packet["row_count"]:
            raise EvidenceError(f"{query} oracle row count differs from the timing artifact")
        expected_total += expected_seconds

    accepted_total = float(comparator.get("total_time_seconds", -1))
    if not math.isclose(expected_total, accepted_total, rel_tol=0.0, abs_tol=1e-9):
        raise EvidenceError(
            f"comparator total mismatch: derived {expected_total:.2f}, "
            f"declared {accepted_total:.2f}"
        )
    if accepted_total != 8086.0:
        raise EvidenceError("accepted SF100 GeoPandas total must remain 8086.00 seconds")
    acceptance = comparator.get("acceptance")
    expected_acceptance = {
        "minimum_suite_speedup": 10.0,
        "maximum_candidate_total_seconds": 808.6,
        "q9_per_query_speedup_exempt": True,
        "all_queries_must_pass_correctness": True,
    }
    if acceptance != expected_acceptance:
        raise EvidenceError("comparator acceptance contract is incomplete or changed")
    return comparator


def _dataset_inventory(dataset_dir: Path) -> tuple[int, int, str]:
    rows: list[str] = []
    total_bytes = 0
    try:
        files = sorted(path for path in dataset_dir.rglob("*") if path.is_file())
    except OSError as exc:
        raise EvidenceError(f"cannot enumerate dataset {dataset_dir}: {exc}") from exc
    for path in files:
        size = path.stat().st_size
        total_bytes += size
        rows.append(f"{path.relative_to(dataset_dir).as_posix()}\t{size}\n")
    digest = hashlib.sha256("".join(rows).encode()).hexdigest()
    return len(files), total_bytes, digest


def _result_manifest(result_dir: Path) -> str:
    digest = hashlib.sha256()
    for query in QUERY_NAMES:
        path = result_dir / f"vibespatial_{query}_result.csv"
        digest.update(f"{path.name}\t{_sha256(path)}\n".encode())
    return digest.hexdigest()


def _validate_dataset(comparator: dict[str, Any], dataset_dir: Path) -> None:
    packet = comparator.get("dataset")
    if not isinstance(packet, dict):
        raise EvidenceError("comparator dataset identity is missing")
    manifest_path = dataset_dir / "geoparquet_manifest.json"
    manifest_hash = _sha256(manifest_path)
    if manifest_hash != packet.get("manifest_sha256"):
        raise EvidenceError(
            f"dataset manifest mismatch: expected {packet.get('manifest_sha256')}, "
            f"found {manifest_hash}"
        )
    count, total_bytes, inventory_hash = _dataset_inventory(dataset_dir)
    expected = (
        packet.get("file_count"),
        packet.get("total_bytes"),
        packet.get("file_inventory_sha256"),
    )
    if (count, total_bytes, inventory_hash) != expected:
        raise EvidenceError(
            "dataset inventory mismatch: "
            f"expected={expected}, found={(count, total_bytes, inventory_hash)}"
        )


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError as exc:
        raise EvidenceError(f"cannot identify comparator CPU: {exc}") from exc
    raise EvidenceError("cannot identify comparator CPU from /proc/cpuinfo")


def _validate_host(comparator: dict[str, Any]) -> None:
    host = comparator.get("host")
    if not isinstance(host, dict):
        raise EvidenceError("comparator host identity is missing")
    actual = {"hostname": platform.node(), "cpu_model": _cpu_model()}
    expected = {"hostname": host.get("hostname"), "cpu_model": host.get("cpu_model")}
    if actual != expected:
        raise EvidenceError(f"comparator host mismatch: expected={expected}, found={actual}")


def _parse_int(value: str, label: str) -> int | None:
    if value == "":
        return None
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise EvidenceError(f"{label} is not an integer: {value!r}") from exc
    if not parsed.is_finite() or parsed != parsed.to_integral_value():
        raise EvidenceError(f"{label} is not an exact integer: {value!r}")
    return int(parsed)


def _parse_float(value: str, label: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise EvidenceError(f"{label} is not a float: {value!r}") from exc


def _compare_cell(expected: str, actual: str, kind: str, label: str, rtol: float, atol: float) -> None:
    if kind == "int":
        equal = _parse_int(expected, label) == _parse_int(actual, label)
    elif kind == "float":
        left = _parse_float(expected, label)
        right = _parse_float(actual, label)
        if left is None or right is None:
            equal = left is right
        elif math.isnan(left) or math.isnan(right):
            equal = math.isnan(left) and math.isnan(right)
        else:
            equal = math.isclose(left, right, rel_tol=rtol, abs_tol=atol)
    else:
        equal = expected == actual
    if not equal:
        raise EvidenceError(f"{label} mismatch: expected {expected!r}, found {actual!r}")


def verify_candidate(
    comparator_path: Path,
    candidate_path: Path,
    result_dir: Path,
    dataset_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Fail closed unless candidate timing, data, and all twelve outputs pass."""
    comparator = validate_comparator(comparator_path, repo_root)
    _validate_dataset(comparator, dataset_dir)
    _validate_host(comparator)

    candidate = _load_json(candidate_path)
    if candidate.get("benchmark") != "spatialbench":
        raise EvidenceError("candidate is not a SpatialBench artifact")
    suite = _suite_by_engine(candidate, "vibespatial")
    if float(suite.get("scale_factor", -1)) != 100.0:
        raise EvidenceError("candidate scale factor must be SF100")
    queries = _query_map(suite)

    candidate_total = 0.0
    results: dict[str, str] = {}
    correctness = comparator.get("correctness", {})
    rtol = float(correctness.get("rtol", 1e-6))
    atol = float(correctness.get("atol", 1e-9))
    for query in QUERY_NAMES:
        timing = queries[query]
        if timing.get("status") != "success":
            raise EvidenceError(f"{query} did not succeed: {timing.get('status')}")
        if timing.get("warmup_runs") != 1 or timing.get("statistic") != "median":
            raise EvidenceError(f"{query} does not use one warmup and median timing")
        run_times = timing.get("run_times_seconds")
        median = _median(run_times, f"candidate {query} samples")
        declared = float(timing.get("time_seconds", -1))
        if not math.isclose(median, declared, rel_tol=0.0, abs_tol=0.005):
            raise EvidenceError(
                f"candidate {query} time {declared} is not the median of {run_times}"
            )
        packet = comparator["queries"][query]
        if timing.get("row_count") != packet["row_count"]:
            raise EvidenceError(
                f"candidate {query} row count mismatch: expected {packet['row_count']}, "
                f"found {timing.get('row_count')}"
            )

        oracle_path = _repo_path(repo_root, packet["oracle"]["path"])
        candidate_csv = result_dir / f"vibespatial_{query}_result.csv"
        oracle_header, oracle_rows = _read_csv(oracle_path)
        candidate_header, candidate_rows = _read_csv(candidate_csv)
        if candidate_header != oracle_header:
            raise EvidenceError(
                f"candidate {query} columns mismatch: "
                f"expected={oracle_header}, found={candidate_header}"
            )
        if len(candidate_rows) != len(oracle_rows):
            raise EvidenceError(
                f"candidate {query} row count mismatch in CSV: "
                f"expected={len(oracle_rows)}, found={len(candidate_rows)}"
            )
        kinds = [column["kind"] for column in packet["columns"]]
        for row_index, (expected_row, actual_row) in enumerate(
            zip(oracle_rows, candidate_rows, strict=True)
        ):
            for column_index, kind in enumerate(kinds):
                _compare_cell(
                    expected_row[column_index],
                    actual_row[column_index],
                    kind,
                    f"{query} row {row_index} column {oracle_header[column_index]!r}",
                    rtol,
                    atol,
                )
        candidate_total += declared
        results[query] = "pass"

    declared_total = float(suite.get("total_time", -1))
    if not math.isclose(candidate_total, declared_total, rel_tol=0.0, abs_tol=0.01):
        raise EvidenceError(
            f"candidate suite total mismatch: summed {candidate_total:.2f}, "
            f"declared {declared_total:.2f}"
        )
    required_speedup = float(comparator["acceptance"]["minimum_suite_speedup"])
    maximum_seconds = comparator["total_time_seconds"] / required_speedup
    if candidate_total > maximum_seconds:
        raise EvidenceError(
            f"candidate misses {required_speedup:.1f}x suite gate: "
            f"{candidate_total:.2f}s > {maximum_seconds:.2f}s"
        )
    return {
        "schema_version": 1,
        "status": "pass",
        "comparator": str(comparator_path),
        "comparator_sha256": _sha256(comparator_path),
        "candidate": str(candidate_path),
        "candidate_sha256": _sha256(candidate_path),
        "candidate_result_manifest_sha256": _result_manifest(result_dir),
        "scale_factor": 100,
        "dataset_manifest_sha256": comparator["dataset"]["manifest_sha256"],
        "dataset_file_inventory_sha256": comparator["dataset"][
            "file_inventory_sha256"
        ],
        "host": comparator["host"],
        "correct_queries": 12,
        "query_results": results,
        "candidate_total_seconds": round(candidate_total, 2),
        "comparator_total_seconds": comparator["total_time_seconds"],
        "suite_speedup": comparator["total_time_seconds"] / candidate_total,
        "minimum_suite_speedup": required_speedup,
        "q9_per_query_speedup_exempt": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check-comparator")
    check.add_argument("--comparator", type=Path, default=DEFAULT_COMPARATOR)
    verify = subparsers.add_parser("verify-candidate")
    verify.add_argument("--comparator", type=Path, default=DEFAULT_COMPARATOR)
    verify.add_argument("--candidate", type=Path, required=True)
    verify.add_argument("--result-dir", type=Path, required=True)
    verify.add_argument("--dataset-dir", type=Path, required=True)
    verify.add_argument("--report", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    comparator_path = (
        args.comparator
        if args.comparator.is_absolute()
        else repo_root / args.comparator
    )
    try:
        if args.command == "check-comparator":
            comparator = validate_comparator(comparator_path, repo_root)
            report = {
                "status": "pass",
                "comparator_sha256": _sha256(comparator_path),
                "queries": len(comparator["queries"]),
                "total_time_seconds": comparator["total_time_seconds"],
            }
        else:
            report = verify_candidate(
                comparator_path=comparator_path,
                candidate_path=args.candidate,
                result_dir=args.result_dir,
                dataset_dir=args.dataset_dir,
                repo_root=repo_root,
            )
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(report, indent=2) + "\n")
    except EvidenceError as exc:
        print(f"SF100 evidence verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
