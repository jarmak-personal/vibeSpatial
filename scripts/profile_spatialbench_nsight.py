#!/usr/bin/env python3
"""Capture and compare query-scoped Nsight Systems traces for SpatialBench."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ALL_QUERIES = tuple(f"q{i}" for i in range(1, 13))
STATS_REPORTS = (
    "cuda_api_sum",
    "cuda_gpu_kern_sum",
    "cuda_gpu_mem_time_sum",
    "cuda_gpu_mem_size_sum",
    "nvtx_sum",
    "osrt_sum",
)
SYNC_APIS = {
    "cudaDeviceSynchronize",
    "cudaEventSynchronize",
    "cudaStreamSynchronize",
    "cuCtxSynchronize",
    "cuEventSynchronize",
    "cuStreamSynchronize",
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _run_text(command: list[str], *, cwd: Path = REPO_ROOT) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return {"command": command, "returncode": None, "stdout": "", "stderr": str(exc)}
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _parse_queries(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return ALL_QUERIES
    queries = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    invalid = sorted(set(queries) - set(ALL_QUERIES))
    if invalid or not queries:
        raise argparse.ArgumentTypeError(
            f"queries must be 'all' or a comma-separated subset of {','.join(ALL_QUERIES)}"
        )
    return queries


def _source_manifest() -> dict[str, Any]:
    revision = _run_text(["git", "rev-parse", "HEAD"])
    status = _run_text(["git", "status", "--short"])
    listed = _run_text(
        ["git", "ls-files", "-co", "--exclude-standard", "src", "benchmarks", "scripts"]
    )
    digest = hashlib.sha256()
    count = 0
    if listed["returncode"] == 0:
        for relative in sorted(line for line in listed["stdout"].splitlines() if line):
            path = REPO_ROOT / relative
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            digest.update(relative.encode())
            digest.update(b"\0")
            digest.update(bytes.fromhex(_sha256_file(path)))
            count += 1
    return {
        "revision": revision["stdout"].strip() or None,
        "git_status": status["stdout"].splitlines(),
        "source_manifest_sha256": digest.hexdigest(),
        "source_manifest_file_count": count,
    }


def _gpu_metrics_probe(nsys: str) -> dict[str, Any]:
    probe = _run_text([nsys, "profile", "--gpu-metrics-devices=help"])
    output = f"{probe['stdout']}\n{probe['stderr']}".strip()
    unavailable_markers = (
        "Insufficient privilege",
        "None of the installed GPUs are supported",
        "No GPU Metrics Devices found",
    )
    available = probe["returncode"] == 0 and not any(
        marker in output for marker in unavailable_markers
    )
    return {
        "available": available,
        "returncode": probe["returncode"],
        "output": output,
    }


def _preflight(nsys: str) -> dict[str, Any]:
    status = _run_text([nsys, "status", "--environment"])
    version = _run_text([nsys, "--version"])
    gpu_metrics = _gpu_metrics_probe(nsys)
    nvidia_smi = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,compute_cap,power.limit",
            "--format=csv,noheader,nounits",
        ]
    )
    lscpu = _run_text(["lscpu", "--json"])
    status_output = f"{status['stdout']}\n{status['stderr']}"
    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nsys_path": shutil.which(nsys) or nsys,
        "nsys_version": version["stdout"].strip() or version["stderr"].strip(),
        "nsys_environment": status_output.strip(),
        "cpu_sampling_available": "CPU Profiling Environment (process-tree): OK"
        in status_output,
        "gpu_metrics": gpu_metrics,
        "nvidia_smi": nvidia_smi,
        "lscpu": lscpu,
        "source": _source_manifest(),
    }


def _host_performance_state(data_dir: Path, output_dir: Path) -> dict[str, Any]:
    governor_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    driver_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver")
    return {
        "data_mount": _run_text(
            ["findmnt", "--json", "-T", str(data_dir), "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"]
        ),
        "output_mount": _run_text(
            ["findmnt", "--json", "-T", str(output_dir), "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"]
        ),
        "block_devices": _run_text(
            ["lsblk", "--json", "-o", "NAME,MODEL,TRAN,TYPE,SIZE,ROTA,MOUNTPOINTS"]
        ),
        "data_filesystem": _run_text(["df", "-B1", str(data_dir)]),
        "output_filesystem": _run_text(["df", "-B1", str(output_dir)]),
        "dataset_size": _run_text(["du", "-sb", str(data_dir)]),
        "memory": _run_text(["free", "-b"]),
        "load": _run_text(["uptime"]),
        "cpu_scaling_governor": governor_path.read_text().strip()
        if governor_path.exists()
        else None,
        "cpu_scaling_driver": driver_path.read_text().strip() if driver_path.exists() else None,
    }


def _gpu_metrics_enabled(mode: str, preflight: dict[str, Any]) -> bool:
    available = bool(preflight["gpu_metrics"]["available"])
    if mode == "on" and not available:
        raise RuntimeError(
            "GPU metrics were required but Nsight reported them unavailable: "
            + preflight["gpu_metrics"]["output"]
        )
    return available and mode != "off"


def _build_profile_command(
    *,
    nsys: str,
    python: str,
    data_dir: Path,
    query: str,
    query_dir: Path,
    warmup_runs: int,
    gpu_metrics: bool,
    cuda_memory_usage: bool,
) -> list[str]:
    trace_base = query_dir / "trace"
    result_path = query_dir / "result.json"
    command = [
        nsys,
        "profile",
        "--capture-range=nvtx",
        f"--nvtx-capture=spatialbench.{query}.measured",
        "--capture-range-end=stop",
        "--trace=cuda,nvtx,osrt,cublas,cudnn,python-gil",
        "--sample=process-tree",
        "--cpuctxsw=process-tree",
        "--python-sampling=true",
        "--python-backtrace=cuda",
        f"--cuda-memory-usage={'true' if cuda_memory_usage else 'false'}",
        "--export=sqlite",
        "--force-overwrite=true",
        f"--output={trace_base}",
    ]
    if gpu_metrics:
        command.extend(
            [
                "--gpu-metrics-devices=cuda-visible",
                "--gpu-metrics-frequency=1000",
            ]
        )
    command.extend(
        [
            python,
            str(Path(__file__).resolve()),
            "_run-query",
            "--data-dir",
            str(data_dir),
            "--query",
            query,
            "--warmup-runs",
            str(warmup_runs),
            "--result",
            str(result_path),
        ]
    )
    return command


def _capture_query(
    *,
    nsys: str,
    python: str,
    data_dir: Path,
    query: str,
    output_dir: Path,
    warmup_runs: int,
    timeout: int,
    gpu_metrics: bool,
    cuda_memory_usage: bool,
    strict_native: bool,
) -> dict[str, Any]:
    query_dir = output_dir / query
    query_dir.mkdir(parents=True, exist_ok=True)
    prewarm_command: list[str] | None = None
    if warmup_runs:
        prewarm_dir = query_dir / "prewarm"
        prewarm_dir.mkdir(parents=True, exist_ok=True)
        prewarm_command = [
            python,
            str(Path(__file__).resolve()),
            "_run-query",
            "--data-dir",
            str(data_dir),
            "--query",
            query,
            "--warmup-runs",
            str(warmup_runs - 1),
            "--result",
            str(prewarm_dir / "result.json"),
        ]
    command = _build_profile_command(
        nsys=nsys,
        python=python,
        data_dir=data_dir,
        query=query,
        query_dir=query_dir,
        # Keep the profiler's device reservation out of the warmup allocator
        # lifetime. This is required for near-capacity queries on 24 GiB cards.
        warmup_runs=0,
        gpu_metrics=gpu_metrics,
        cuda_memory_usage=cuda_memory_usage,
    )
    _write_json(
        query_dir / "command.json",
        {
            "prewarm_command": prewarm_command,
            "profile_command": command,
            "warmup_isolation": "disposable-process",
        },
    )
    environment = os.environ.copy()
    environment["VIBESPATIAL_HOTPATH_TRACE"] = "1"
    environment["VIBESPATIAL_HOTPATH_NVTX"] = "1"
    if strict_native:
        environment["VIBESPATIAL_STRICT_NATIVE"] = "1"
    else:
        environment.pop("VIBESPATIAL_STRICT_NATIVE", None)

    started = time.monotonic()
    status = "success"
    error = None
    try:
        if prewarm_command is not None:
            prewarm = subprocess.run(
                prewarm_command,
                cwd=REPO_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            (query_dir / "prewarm.stdout.txt").write_text(prewarm.stdout)
            (query_dir / "prewarm.stderr.txt").write_text(prewarm.stderr)
            if prewarm.returncode != 0:
                raise RuntimeError(f"prewarm returned {prewarm.returncode}")
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        returncode = completed.returncode
        (query_dir / "capture.stdout.txt").write_text(completed.stdout)
        (query_dir / "capture.stderr.txt").write_text(completed.stderr)
        if returncode != 0:
            status = "error"
            error = f"nsys returned {returncode}"
    except subprocess.TimeoutExpired as exc:
        returncode = None
        status = "timeout"
        error = f"capture exceeded {timeout} seconds"
        (query_dir / "capture.stdout.txt").write_text(exc.stdout or "")
        (query_dir / "capture.stderr.txt").write_text(exc.stderr or "")

    sqlite_path = query_dir / "trace.sqlite"
    result_path = query_dir / "result.json"
    if status == "success" and (not sqlite_path.exists() or not result_path.exists()):
        status = "error"
        error = "capture did not produce trace.sqlite and result.json"

    summary = {
        "query": query,
        "status": status,
        "error": error,
        "capture_elapsed_seconds": time.monotonic() - started,
        "returncode": returncode,
        "gpu_metrics_enabled": gpu_metrics,
        "cuda_memory_usage_enabled": cuda_memory_usage,
        "warmup_runs": warmup_runs,
        "warmup_isolation": "disposable-process",
    }
    if status == "success":
        summary.update(_summarize_sqlite(sqlite_path, result_path, query=query))
        _write_stats(nsys, query_dir)
        summary["artifact_bytes"] = sum(
            path.stat().st_size for path in query_dir.iterdir() if path.is_file()
        )
    _write_json(query_dir / "summary.json", summary)
    return summary


def _write_stats(nsys: str, query_dir: Path) -> None:
    command = [
        nsys,
        "stats",
        "--force-overwrite=true",
        "--report",
        ",".join(STATS_REPORTS),
        "--format",
        "csv",
        "--output",
        str(query_dir / "stats"),
        str(query_dir / "trace.nsys-rep"),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    (query_dir / "stats.stdout.txt").write_text(completed.stdout)
    (query_dir / "stats.stderr.txt").write_text(completed.stderr)


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}


def _named_durations(
    connection: sqlite3.Connection,
    table: str,
    name_column: str,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    query = f"""
        SELECT COALESCE(strings.value, '[unknown]') AS name,
               COUNT(*) AS calls,
               SUM(events.end - events.start) AS total_ns
        FROM {table} AS events
        LEFT JOIN StringIds AS strings ON strings.id = events.{name_column}
        GROUP BY name
        ORDER BY total_ns DESC
        LIMIT ?
    """
    return [
        {"name": name, "calls": int(calls), "total_ns": int(total_ns or 0)}
        for name, calls, total_ns in connection.execute(query, (limit,))
    ]


def _event_intervals(
    connection: sqlite3.Connection,
    tables: Iterable[str],
    existing: set[str],
) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for table in tables:
        if table not in existing:
            continue
        intervals.extend(
            (int(start), int(end))
            for start, end in connection.execute(f"SELECT start, end FROM {table}")
            if end is not None and int(end) >= int(start)
        )
    return intervals


def _merge_intervals(
    intervals: Iterable[tuple[int, int]],
    *,
    lower: int,
    upper: int,
) -> list[tuple[int, int]]:
    clipped = sorted(
        (max(lower, start), min(upper, end))
        for start, end in intervals
        if end > lower and start < upper
    )
    merged: list[tuple[int, int]] = []
    for start, end in clipped:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _measured_range(
    connection: sqlite3.Connection,
    existing: set[str],
    query: str,
    fallback_intervals: list[tuple[int, int]],
) -> tuple[int, int]:
    label = f"spatialbench.{query}.measured"
    if "NVTX_EVENTS" in existing:
        row = connection.execute(
            """
            SELECT events.start, events.end
            FROM NVTX_EVENTS AS events
            LEFT JOIN StringIds AS strings ON strings.id = events.textId
            WHERE COALESCE(events.text, strings.value) = ? AND events.end IS NOT NULL
            ORDER BY events.start LIMIT 1
            """,
            (label,),
        ).fetchone()
        if row is not None:
            return int(row[0]), int(row[1])
    if fallback_intervals:
        return min(start for start, _ in fallback_intervals), max(end for _, end in fallback_intervals)
    return 0, 0


def _summarize_sqlite(sqlite_path: Path, result_path: Path, *, query: str) -> dict[str, Any]:
    result = json.loads(result_path.read_text())
    connection = sqlite3.connect(sqlite_path)
    try:
        existing = _table_names(connection)
        gpu_tables = (
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "CUPTI_ACTIVITY_KIND_MEMCPY",
            "CUPTI_ACTIVITY_KIND_MEMSET",
        )
        intervals = _event_intervals(connection, gpu_tables, existing)
        range_start, range_end = _measured_range(connection, existing, query, intervals)
        merged = _merge_intervals(intervals, lower=range_start, upper=range_end)
        measured_ns = max(0, range_end - range_start)
        busy_ns = sum(end - start for start, end in merged)
        gaps: list[int] = []
        cursor = range_start
        for start, end in merged:
            gaps.append(max(0, start - cursor))
            cursor = end
        if range_end >= cursor:
            gaps.append(range_end - cursor)

        top_kernels = (
            _named_durations(connection, "CUPTI_ACTIVITY_KIND_KERNEL", "demangledName")
            if "CUPTI_ACTIVITY_KIND_KERNEL" in existing
            else []
        )
        top_cuda_apis = (
            _named_durations(connection, "CUPTI_ACTIVITY_KIND_RUNTIME", "nameId")
            if "CUPTI_ACTIVITY_KIND_RUNTIME" in existing
            else []
        )
        top_osrt = (
            _named_durations(connection, "OSRT_API", "nameId")
            if "OSRT_API" in existing
            else []
        )
        sync_ns = 0
        if "CUPTI_ACTIVITY_KIND_RUNTIME" in existing:
            placeholders = ",".join("?" for _ in SYNC_APIS)
            sync_ns = int(
                connection.execute(
                    f"""
                    SELECT COALESCE(SUM(events.end-events.start), 0)
                    FROM CUPTI_ACTIVITY_KIND_RUNTIME AS events
                    JOIN StringIds AS strings ON strings.id = events.nameId
                    WHERE strings.value IN ({placeholders})
                    """,
                    tuple(sorted(SYNC_APIS)),
                ).fetchone()[0]
            )
        kernel_sum_ns = sum(item["total_ns"] for item in top_kernels)
        kernel_count = sum(item["calls"] for item in top_kernels)
        if "CUPTI_ACTIVITY_KIND_KERNEL" in existing:
            kernel_sum_ns, kernel_count = connection.execute(
                "SELECT COALESCE(SUM(end-start), 0), COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL"
            ).fetchone()

        memcpy_count = 0
        memcpy_sum_ns = 0
        memcpy_bytes = 0
        if "CUPTI_ACTIVITY_KIND_MEMCPY" in existing:
            memcpy_columns = _table_columns(connection, "CUPTI_ACTIVITY_KIND_MEMCPY")
            byte_expression = "COALESCE(SUM(bytes), 0)" if "bytes" in memcpy_columns else "0"
            memcpy_sum_ns, memcpy_count, memcpy_bytes = connection.execute(
                f"""
                SELECT COALESCE(SUM(end-start), 0), COUNT(*), {byte_expression}
                FROM CUPTI_ACTIVITY_KIND_MEMCPY
                """
            ).fetchone()

        return {
            "result": result,
            "trace": {
                "measured_range_ns": measured_ns,
                "gpu_busy_union_ns": busy_ns,
                "gpu_busy_fraction": (busy_ns / measured_ns) if measured_ns else None,
                "gpu_idle_union_ns": max(0, measured_ns - busy_ns),
                "largest_gpu_idle_gap_ns": max(gaps, default=0),
                "kernel_sum_ns": int(kernel_sum_ns or 0),
                "kernel_launch_count": int(kernel_count or 0),
                "memcpy_sum_ns": int(memcpy_sum_ns or 0),
                "memcpy_count": int(memcpy_count or 0),
                "memcpy_bytes": int(memcpy_bytes or 0),
                "cuda_sync_api_ns": sync_ns,
                "top_kernels": top_kernels,
                "top_cuda_apis": top_cuda_apis,
                "top_os_runtime_calls": top_osrt,
                "sqlite_tables": sorted(existing),
            },
        }
    finally:
        connection.close()


def _run_query(args: argparse.Namespace) -> int:
    # Delay benchmark and CUDA imports so setup/warmup are excluded from capture.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import cupy as cp
    import nvtx

    from benchmarks.spatialbench.run_benchmark import (
        VibeSpatialBenchmark,
        get_data_paths,
        normalize,
    )
    from vibespatial.runtime.hotpath_trace import reset_hotpath_trace, summarize_hotpath_trace

    result_path = Path(args.result).resolve()
    data_paths = get_data_paths(args.data_dir)
    if not data_paths:
        raise RuntimeError(f"no SpatialBench inputs found under {args.data_dir}")

    benchmark = VibeSpatialBenchmark(data_paths)
    completed = False
    benchmark.setup()
    try:
        for _ in range(args.warmup_runs):
            _, warmup_result = benchmark.execute_query(args.query, None)
            cp.cuda.get_current_stream().synchronize()
            del warmup_result
            gc.collect()

        reset_hotpath_trace()
        with nvtx.annotate(f"spatialbench.{args.query}.measured", color="green"):
            started = time.perf_counter()
            row_count, result = benchmark.execute_query(args.query, None)
            cp.cuda.get_current_stream().synchronize()
            elapsed = time.perf_counter() - started

        normalized_path = result_path.with_name("normalized.csv")
        normalize(benchmark.dump_frame(result, None)).to_csv(normalized_path, index=False)
        normalized_sha256 = hashlib.sha256(normalized_path.read_bytes()).hexdigest()
        payload = {
            "schema_version": 1,
            "query": args.query,
            "status": "success",
            "elapsed_seconds": elapsed,
            "row_count": row_count,
            "warmup_runs": args.warmup_runs,
            "normalized_result_sha256": normalized_sha256,
            "hotpath_summary": summarize_hotpath_trace(),
        }
        _write_json(result_path, payload)
        completed = True
    finally:
        benchmark.teardown()
    if completed:
        # Match multiprocessing.Process worker semantics used by SpatialBench.
        # Large CUDA/RMM query results can fault during interpreter-global
        # destructor ordering even after a correct result has been written.
        # This process owns no reusable state after its artifacts are closed.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    return 0


def _suite_summary(output_dir: Path, preflight: dict[str, Any], captures: list[dict[str, Any]]) -> None:
    successful = [capture for capture in captures if capture["status"] == "success"]
    wall = sum(float(item["result"]["elapsed_seconds"]) for item in successful)
    measured_ns = sum(int(item["trace"]["measured_range_ns"]) for item in successful)
    busy_ns = sum(int(item["trace"]["gpu_busy_union_ns"]) for item in successful)
    payload = {
        "schema_version": 1,
        "type": "spatialbench-nsight-suite",
        "environment": preflight,
        "query_count": len(captures),
        "passed": len(successful),
        "failed": len(captures) - len(successful),
        "measured_wall_seconds": wall,
        "weighted_gpu_busy_fraction": (busy_ns / measured_ns) if measured_ns else None,
        "queries": captures,
    }
    _write_json(output_dir / "suite-summary.json", payload)


def _write_sha256s(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        rows.append(f"{_sha256_file(path)}  {path.relative_to(output_dir)}")
    (output_dir / "SHA256SUMS").write_text("\n".join(rows) + "\n")


def _capture(args: argparse.Namespace) -> int:
    nsys = shutil.which(args.nsys)
    if nsys is None:
        raise RuntimeError(f"Nsight Systems CLI not found: {args.nsys}")
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise RuntimeError(f"output directory is not empty: {output_dir}; use --force to overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight = _preflight(nsys)
    preflight["host_performance_state"] = _host_performance_state(data_dir, output_dir)
    metrics_enabled = _gpu_metrics_enabled(args.gpu_metrics, preflight)
    preflight["capture_contract"] = {
        "queries": list(args.queries),
        "warmup_runs": args.warmup_runs,
        "timeout_seconds": args.timeout,
        "strict_native": args.strict_native,
        "gpu_metrics_requested": args.gpu_metrics,
        "gpu_metrics_enabled": metrics_enabled,
        "cuda_memory_usage_enabled": args.cuda_memory_usage,
    }
    _write_json(output_dir / "environment.json", preflight)
    if not preflight["cpu_sampling_available"] and not args.allow_partial:
        raise RuntimeError(
            "Nsight CPU sampling/context-switch profiling is unavailable. "
            "Use --allow-partial for a CUDA/NVTX-only trace or adjust perf_event_paranoid "
            "before collecting a full CPU-bottleneck comparison."
        )

    captures: list[dict[str, Any]] = []
    for query in args.queries:
        print(f"[{query}] capturing", flush=True)
        capture = _capture_query(
            nsys=nsys,
            python=shutil.which(args.python) or str(Path(args.python).resolve()),
            data_dir=data_dir,
            query=query,
            output_dir=output_dir,
            warmup_runs=args.warmup_runs,
            timeout=args.timeout,
            gpu_metrics=metrics_enabled,
            cuda_memory_usage=args.cuda_memory_usage,
            strict_native=args.strict_native,
        )
        captures.append(capture)
        print(f"[{query}] {capture['status']}", flush=True)
        _suite_summary(output_dir, preflight, captures)
        if capture["status"] != "success" and not args.keep_going:
            _write_sha256s(output_dir)
            return 1
    _write_sha256s(output_dir)
    return 0 if all(item["status"] == "success" for item in captures) else 1


def _format_float(value: float | None, digits: int = 2) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def _comparison_payload(
    baseline: dict[str, Any], candidate: dict[str, Any], baseline_label: str, candidate_label: str
) -> dict[str, Any]:
    baseline_queries = {item["query"]: item for item in baseline["queries"]}
    candidate_queries = {item["query"]: item for item in candidate["queries"]}
    comparisons = []
    for query in ALL_QUERIES:
        left = baseline_queries.get(query)
        right = candidate_queries.get(query)
        if not left or not right or left["status"] != "success" or right["status"] != "success":
            comparisons.append({"query": query, "status": "missing"})
            continue
        left_wall = float(left["result"]["elapsed_seconds"])
        right_wall = float(right["result"]["elapsed_seconds"])
        result_hash_match = (
            left["result"].get("normalized_result_sha256")
            == right["result"].get("normalized_result_sha256")
        )
        comparisons.append(
            {
                "query": query,
                "status": "success",
                "result_hash_match": result_hash_match,
                "baseline_wall_seconds": left_wall,
                "candidate_wall_seconds": right_wall,
                "wall_speedup": left_wall / right_wall if right_wall else None,
                "baseline_gpu_busy_fraction": left["trace"]["gpu_busy_fraction"],
                "candidate_gpu_busy_fraction": right["trace"]["gpu_busy_fraction"],
                "baseline_kernel_sum_seconds": left["trace"]["kernel_sum_ns"] / 1e9,
                "candidate_kernel_sum_seconds": right["trace"]["kernel_sum_ns"] / 1e9,
                "baseline_kernel_launches": left["trace"]["kernel_launch_count"],
                "candidate_kernel_launches": right["trace"]["kernel_launch_count"],
                "baseline_cuda_sync_seconds": left["trace"]["cuda_sync_api_ns"] / 1e9,
                "candidate_cuda_sync_seconds": right["trace"]["cuda_sync_api_ns"] / 1e9,
                "baseline_largest_gpu_gap_ms": left["trace"]["largest_gpu_idle_gap_ns"] / 1e6,
                "candidate_largest_gpu_gap_ms": right["trace"]["largest_gpu_idle_gap_ns"] / 1e6,
            }
        )
    baseline_contract = baseline["environment"].get("capture_contract", {})
    candidate_contract = candidate["environment"].get("capture_contract", {})
    comparable_contract_fields = (
        "warmup_runs",
        "strict_native",
        "gpu_metrics_enabled",
        "cuda_memory_usage_enabled",
    )
    contract_mismatches = [
        field
        for field in comparable_contract_fields
        if baseline_contract.get(field) != candidate_contract.get(field)
    ]
    compatibility = {
        "source_manifest_match": (
            baseline["environment"]["source"].get("source_manifest_sha256")
            == candidate["environment"]["source"].get("source_manifest_sha256")
        ),
        "capture_contract_mismatches": contract_mismatches,
        "nsys_version_match": (
            baseline["environment"].get("nsys_version")
            == candidate["environment"].get("nsys_version")
        ),
        "result_hash_mismatches": [
            item["query"]
            for item in comparisons
            if item.get("status") == "success" and not item["result_hash_match"]
        ],
    }
    return {
        "schema_version": 1,
        "type": "spatialbench-nsight-comparison",
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "baseline_source": baseline["environment"]["source"],
        "candidate_source": candidate["environment"]["source"],
        "compatibility": compatibility,
        "queries": comparisons,
    }


def _comparison_markdown(payload: dict[str, Any]) -> str:
    left = payload["baseline_label"]
    right = payload["candidate_label"]
    lines = [
        "# SpatialBench Nsight Comparison",
        "",
        "Instrumented capture times are comparable only when source fingerprints and capture "
        "contracts match. GPU busy is the union of kernel, memcpy, and memset intervals inside "
        "the measured NVTX range.",
        "",
        f"Source manifest match: **{payload['compatibility']['source_manifest_match']}**; "
        f"Nsight version match: **{payload['compatibility']['nsys_version_match']}**; "
        f"capture-contract mismatches: **"
        f"{payload['compatibility']['capture_contract_mismatches'] or 'none'}**.",
        "",
        f"| Query | {left} wall | {right} wall | Speedup | {left} GPU busy | "
        f"{right} GPU busy | Kernel launches ({left}/{right}) | CUDA sync ({left}/{right}) | "
        f"Largest GPU gap ({left}/{right}) | Result hash |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["queries"]:
        if item["status"] != "success":
            lines.append(f"| {item['query']} | -- | -- | -- | -- | -- | -- | -- | -- | -- |")
            continue
        lines.append(
            f"| {item['query']} | {item['baseline_wall_seconds']:.3f}s | "
            f"{item['candidate_wall_seconds']:.3f}s | {_format_float(item['wall_speedup'])}x | "
            f"{100 * item['baseline_gpu_busy_fraction']:.1f}% | "
            f"{100 * item['candidate_gpu_busy_fraction']:.1f}% | "
            f"{item['baseline_kernel_launches']}/{item['candidate_kernel_launches']} | "
            f"{item['baseline_cuda_sync_seconds']:.3f}s/{item['candidate_cuda_sync_seconds']:.3f}s | "
            f"{item['baseline_largest_gpu_gap_ms']:.1f}ms/"
            f"{item['candidate_largest_gpu_gap_ms']:.1f}ms | "
            f"{'match' if item['result_hash_match'] else 'MISMATCH'} |"
        )
    return "\n".join(lines) + "\n"


def _compare(args: argparse.Namespace) -> int:
    baseline = json.loads(Path(args.baseline).read_text())
    candidate = json.loads(Path(args.candidate).read_text())
    payload = _comparison_payload(baseline, candidate, args.baseline_label, args.candidate_label)
    compatibility = payload["compatibility"]
    blocking_mismatches = []
    if not compatibility["source_manifest_match"]:
        blocking_mismatches.append("source manifest")
    if compatibility["capture_contract_mismatches"]:
        blocking_mismatches.append(
            "capture contract: " + ", ".join(compatibility["capture_contract_mismatches"])
        )
    if compatibility["result_hash_mismatches"]:
        blocking_mismatches.append(
            "normalized result hashes: " + ", ".join(compatibility["result_hash_mismatches"])
        )
    if blocking_mismatches and not args.allow_mismatch:
        raise RuntimeError(
            "profiles are not comparable (" + "; ".join(blocking_mismatches) + ")"
        )
    output = Path(args.output).resolve()
    _write_json(output.with_suffix(".json"), payload)
    output.with_suffix(".md").write_text(_comparison_markdown(payload))
    print(output.with_suffix(".md"))
    return 0


def _preflight_command(args: argparse.Namespace) -> int:
    nsys = shutil.which(args.nsys)
    if nsys is None:
        raise RuntimeError(f"Nsight Systems CLI not found: {args.nsys}")
    payload = _preflight(nsys)
    if args.output:
        _write_json(Path(args.output).resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="check Nsight and profiling permissions")
    preflight.add_argument("--nsys", default="nsys")
    preflight.add_argument("--output")
    preflight.set_defaults(func=_preflight_command)

    capture = subparsers.add_parser("capture", help="capture one trace per SF100 query")
    capture.add_argument("--data-dir", required=True)
    capture.add_argument("--output-dir", required=True)
    capture.add_argument("--queries", type=_parse_queries, default=ALL_QUERIES)
    capture.add_argument("--warmup-runs", type=int, default=1)
    capture.add_argument("--timeout", type=int, default=1800)
    capture.add_argument("--python", default=sys.executable)
    capture.add_argument("--nsys", default="nsys")
    capture.add_argument("--gpu-metrics", choices=("auto", "on", "off"), default="auto")
    capture.add_argument(
        "--cuda-memory-usage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="trace CUDA allocation lifetime; disabled by default because it perturbs 24GB admission",
    )
    capture.add_argument("--strict-native", action=argparse.BooleanOptionalAction, default=True)
    capture.add_argument("--allow-partial", action="store_true")
    capture.add_argument("--keep-going", action=argparse.BooleanOptionalAction, default=True)
    capture.add_argument("--force", action="store_true")
    capture.set_defaults(func=_capture)

    compare = subparsers.add_parser("compare", help="compare two suite-summary.json files")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--baseline-label", default="RTX 4090")
    compare.add_argument("--candidate-label", default="H200")
    compare.add_argument("--output", required=True, help="output basename for .json and .md")
    compare.add_argument("--allow-mismatch", action="store_true")
    compare.set_defaults(func=_compare)

    inner = subparsers.add_parser("_run-query", help=argparse.SUPPRESS)
    inner.add_argument("--data-dir", required=True)
    inner.add_argument("--query", choices=ALL_QUERIES, required=True)
    inner.add_argument("--warmup-runs", type=int, default=1)
    inner.add_argument("--result", required=True)
    inner.set_defaults(func=_run_query)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if getattr(args, "warmup_runs", 0) < 0:
        raise SystemExit("--warmup-runs must be non-negative")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
