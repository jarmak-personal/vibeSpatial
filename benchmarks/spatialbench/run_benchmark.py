#!/usr/bin/env python3
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.

"""
SpatialBench Benchmark Runner

This script runs spatial benchmarks comparing SedonaDB, DuckDB, and GeoPandas
on the SpatialBench queries at a specified scale factor.
"""

import argparse
import gc
import json
import multiprocessing
import os
import queue
import signal
import statistics
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Query modules live beside this runner in the vibeSpatial repository. Use
# append (not insert) so installed packages still take precedence.
QUERY_MODULE_DIR = Path(__file__).resolve().parent
sys.path.append(str(QUERY_MODULE_DIR))

# Constants
QUERY_COUNT = 12
TABLES = ["building", "customer", "driver", "trip", "vehicle", "zone"]
DURATION_SUFFIX = "_seconds"
# When --result-dir is set, the worker serializes the (already computed) result
# after queueing its timing. If the process is still alive at the query timeout, we
# briefly probe the queue: a result already there means the query finished and only
# serialization is running, which then gets its own grace window rather than being
# charged to the query timeout.
SERIALIZE_PROBE_SECONDS = 5
SERIALIZE_GRACE_SECONDS = 120


@dataclass
class BenchmarkResult:
    """Result of a single query benchmark."""
    query: str
    engine: str
    time_seconds: float | None
    row_count: int | None
    status: str  # "success", "error", "timeout"
    error_message: str | None = None
    run_times_seconds: list[float] = field(default_factory=list)
    statistic: str | None = None
    warmup_runs: int = 0
    telemetry_runs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""
    engine: str
    scale_factor: float
    results: list[BenchmarkResult] = field(default_factory=list)
    total_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    version: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "version": self.version,
            "scale_factor": self.scale_factor,
            "timestamp": self.timestamp,
            "total_time": self.total_time,
            "results": [
                {
                    "query": r.query,
                    "time_seconds": r.time_seconds,
                    "row_count": r.row_count,
                    "status": r.status,
                    "error_message": r.error_message,
                    "run_times_seconds": r.run_times_seconds,
                    "statistic": r.statistic,
                    "warmup_runs": r.warmup_runs,
                    "telemetry_runs": r.telemetry_runs,
                }
                for r in self.results
            ],
        }


class QueryTimeoutError(Exception):
    """Raised when a query times out."""
    pass


class _ProcessVramSampler:
    """Sample this process's NVML-reported device memory off the timed thread."""

    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = interval_seconds
        self.peak_bytes: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
        except Exception:
            return

        pid = os.getpid()

        def _sample() -> None:
            while not self._stop.is_set():
                try:
                    total = 0
                    for device_index in range(pynvml.nvmlDeviceGetCount()):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                        total += sum(
                            int(process.usedGpuMemory)
                            for process in processes
                            if process.pid == pid
                        )
                    self.peak_bytes = max(self.peak_bytes or 0, total)
                except Exception:
                    pass
                self._stop.wait(self.interval_seconds)

        self._thread = threading.Thread(target=_sample, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(self.interval_seconds * 2, 0.2))


def _event_dict(event: Any) -> dict[str, Any]:
    to_dict = getattr(event, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    values = getattr(event, "__dict__", None)
    return dict(values) if isinstance(values, dict) else {"event": repr(event)}


def _begin_vibespatial_telemetry(engine_class: type) -> dict[str, Any] | None:
    if engine_class.__name__ != "VibeSpatialBenchmark":
        return None

    from rmm import statistics as rmm_statistics

    import vibespatial
    from vibespatial.cuda._runtime import (
        get_cuda_runtime,
        reset_d2h_transfer_count,
    )
    from vibespatial.runtime.hotpath_trace import reset_hotpath_trace

    runtime = get_cuda_runtime()
    rmm_statistics.enable_statistics()
    rmm_statistics.push_statistics()
    reset_d2h_transfer_count()
    reset_hotpath_trace()
    vibespatial.clear_fallback_events()
    vibespatial.clear_materialization_events()
    sampler = _ProcessVramSampler()
    sampler.start()
    return {
        "runtime": runtime,
        "rmm_statistics": rmm_statistics,
        "sampler": sampler,
    }


def _end_vibespatial_telemetry(state: dict[str, Any] | None) -> dict[str, Any]:
    if state is None:
        return {}

    import vibespatial
    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        get_d2h_transfer_profile,
    )
    from vibespatial.runtime.hotpath_trace import (
        hotpath_trace_mode,
        summarize_hotpath_trace,
    )

    sampler = state["sampler"]
    sampler.stop()
    count, transferred_bytes, transfer_seconds = get_d2h_transfer_profile()
    transfer_events = get_d2h_transfer_events(clear=True)
    fallback_events = vibespatial.get_fallback_events(clear=True)
    materialization_events = vibespatial.get_materialization_events(clear=True)
    allocation_stats = state["rmm_statistics"].pop_statistics()
    pool_stats = state["runtime"].memory_pool_stats()
    hotpath_mode = hotpath_trace_mode()
    return {
        "peak_vram_bytes": sampler.peak_bytes,
        "rmm_peak_allocation_bytes": (
            None if allocation_stats is None else int(allocation_stats.peak_bytes)
        ),
        "rmm_total_allocation_bytes": (
            None if allocation_stats is None else int(allocation_stats.total_bytes)
        ),
        "rmm_allocation_count": (
            None if allocation_stats is None else int(allocation_stats.total_count)
        ),
        "pool_reserved_bytes": pool_stats.get("reserved_bytes"),
        "pool_live_bytes": pool_stats.get("used_bytes"),
        "largest_admitted_allocation_bytes": pool_stats.get(
            "largest_admitted_allocation_bytes"
        ),
        "d2h_transfer_count": count,
        "d2h_transfer_bytes": transferred_bytes,
        "d2h_transfer_seconds": transfer_seconds,
        "d2h_transfer_events": [
            _event_dict(event) for event in transfer_events[:40]
        ],
        "d2h_transfer_events_truncated": len(transfer_events) > 40,
        "fallback_event_count": len(fallback_events),
        "fallback_events": [_event_dict(event) for event in fallback_events[:40]],
        "fallback_events_truncated": len(fallback_events) > 40,
        "materialization_event_count": len(materialization_events),
        "materialization_events": [
            _event_dict(event) for event in materialization_events[:40]
        ],
        "materialization_events_truncated": len(materialization_events) > 40,
        "hotpath_trace_mode": hotpath_mode,
        "hotpath": (
            summarize_hotpath_trace()[:120] if hotpath_mode != "off" else []
        ),
    }


def _run_query_in_process(
    result_queue: multiprocessing.Queue,
    engine_class: type,
    data_paths: dict[str, str],
    query_name: str,
    query_sql: str | None,
    dump_csv: str | None = None,
    measured_runs: int = 1,
    warmup_runs: int = 0,
    statistic: str = "mean",
    profile_telemetry: bool = False,
    profile_point_region: bool = False,
):
    """Worker function to run a query in a separate process.

    This allows us to forcefully terminate queries that hang or consume
    too much memory, which SIGALRM cannot do for native code.

    If dump_csv is set, the *timed* result is reused (no re-execution) to write the
    normalized result there for the correctness verify job. The timing result is put
    on the queue before dumping, so a dump failure never affects the timing; the
    verify job reports a missing csv as a failed correctness check.
    """
    try:
        # For Spatial Polars, ensure the package is imported first to register namespace
        if engine_class.__name__ == "SpatialPolarsBenchmark":
            import spatial_polars as _sp  # noqa: F401

        benchmark = engine_class(data_paths)
        benchmark.setup()
        try:
            for _ in range(warmup_runs):
                _warmup_row_count, warmup_result = benchmark.execute_query(
                    query_name,
                    query_sql,
                )
                del warmup_result
                gc.collect()

            run_times: list[float] = []
            telemetry_runs: list[dict[str, Any]] = []
            row_count = None
            result = None
            for run_number in range(measured_runs):
                telemetry_state = (
                    _begin_vibespatial_telemetry(engine_class)
                    if profile_telemetry
                    else None
                )
                point_profile = None
                if profile_point_region and engine_class.__name__ == "VibeSpatialBenchmark":
                    from vibespatial.predicates.point_region_profile import (
                        profile_point_region as create_point_region_profile,
                    )

                    point_profile = create_point_region_profile(
                        label=f"spatialbench-{query_name}-run-{run_number + 1}"
                    )
                    point_profile.__enter__()
                try:
                    start_time = time.perf_counter()
                    measured_row_count, measured_result = benchmark.execute_query(
                        query_name,
                        query_sql,
                    )
                    run_times.append(time.perf_counter() - start_time)
                    point_profile_snapshot = (
                        point_profile.snapshot() if point_profile is not None else None
                    )
                finally:
                    if point_profile is not None:
                        point_profile.__exit__(None, None, None)
                telemetry = (
                    _end_vibespatial_telemetry(telemetry_state)
                    if profile_telemetry
                    else {}
                )
                if point_profile_snapshot is not None:
                    telemetry["point_region_profile"] = point_profile_snapshot
                if telemetry:
                    telemetry_runs.append(telemetry)
                if row_count is not None and measured_row_count != row_count:
                    raise RuntimeError(
                        f"{query_name} row count changed across measured runs: "
                        f"{row_count} != {measured_row_count}"
                    )
                row_count = measured_row_count
                if run_number + 1 < measured_runs:
                    del measured_result
                    gc.collect()
                else:
                    result = measured_result

            elapsed = (
                statistics.median(run_times)
                if statistic == "median"
                else statistics.fmean(run_times)
            )
            result_queue.put({
                "status": "success",
                "time_seconds": round(elapsed, 2),
                "row_count": row_count,
                "error_message": None,
                "run_times_seconds": [round(value, 2) for value in run_times],
                "statistic": statistic,
                "warmup_runs": warmup_runs,
                "telemetry_runs": telemetry_runs,
            })
            if dump_csv:
                try:
                    # Write to a temp file and atomically replace, so a process killed
                    # mid-serialization never leaves a partial csv at the final path.
                    tmp_csv = f"{dump_csv}.tmp"
                    normalize(benchmark.dump_frame(result, query_sql)).to_csv(tmp_csv, index=False)
                    os.replace(tmp_csv, dump_csv)
                except Exception as e:
                    print(f"  [dump] failed to write {dump_csv}: {e}", file=sys.stderr, flush=True)
        finally:
            benchmark.teardown()
    except Exception as e:
        result_queue.put({
            "status": "error",
            "time_seconds": None,
            "row_count": None,
            "error_message": str(e),
            "run_times_seconds": [],
            "statistic": statistic,
            "warmup_runs": warmup_runs,
            "telemetry_runs": [],
        })


# ── Result dumping (for correctness verification) ──
# The benchmark run optionally writes each query's normalized result to a csv so a
# downstream, engine-free CI job (verify_results.py) can compare it to the committed
# ground-truth answer. The dump reuses the *timed* run's result (no extra query
# execution) and writes csv only (never parquet), so no pyarrow filesystem is
# touched inside a SedonaDB worker. pandas/numpy are imported lazily so the parent
# never pulls them in before forking a SedonaDB worker (which bundles its own Arrow
# and segfaults if pyarrow/pandas load first).


def normalize(df):
    """Coerce an engine result to the canonical, engine-neutral form.

    Matches the normalization used to generate the committed answers: durations
    become float seconds (with a ``_seconds`` suffix), decimals become floats, and
    timestamps become microsecond datetimes. Object columns of Decimal / timedelta
    (which DuckDB's fetchall yields) are handled too.
    """
    import datetime as _dt
    import decimal

    import pandas as pd

    df = df.reset_index(drop=True).copy()
    rename = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_timedelta64_dtype(s):
            df[col] = s.dt.total_seconds().astype(float)
            if not str(col).endswith(DURATION_SUFFIX):
                rename[col] = f"{col}{DURATION_SUFFIX}"
        elif s.dtype == object:
            non_null = s.dropna()
            if len(non_null) and all(isinstance(v, decimal.Decimal) for v in non_null):
                df[col] = s.astype(float)
            elif len(non_null) and all(isinstance(v, _dt.timedelta) for v in non_null):
                df[col] = s.map(lambda v: v.total_seconds() if isinstance(v, _dt.timedelta) else v).astype(float)
                if not str(col).endswith(DURATION_SUFFIX):
                    rename[col] = f"{col}{DURATION_SUFFIX}"
        elif pd.api.types.is_datetime64_any_dtype(s):
            df[col] = s.astype("datetime64[us]")
    return df.rename(columns=rename)


def _to_pandas_frame(result):
    """Coerce whatever an engine's query returns into a plain pandas DataFrame."""
    import pandas as pd

    if isinstance(result, pd.DataFrame):
        return pd.DataFrame(result)  # strip GeoDataFrame subclass; keep values
    if hasattr(result, "to_pandas"):  # polars (spatial_polars, pycanopy)
        return result.to_pandas()
    return pd.DataFrame(result)


def get_data_paths(data_dir: str) -> dict[str, str]:
    """Get paths to all data tables.

    Supports two data formats:
    1. Directory format: table_name/*.parquet (e.g., building/building.1.parquet)
    2. Single file format: table_name.parquet (e.g., building.parquet)

    Returns directory paths for directories containing parquet files.
    Both DuckDB, pandas, and SedonaDB can read all parquet files from a directory.
    """
    data_path = Path(data_dir)
    paths = {}

    for table in TABLES:
        table_path = data_path / table
        # Check for directory format first (from HF: building/building.1.parquet)
        if table_path.is_dir():
            parquet_files = list(table_path.glob("*.parquet"))
            if parquet_files:
                # Return directory path - DuckDB, pandas, and SedonaDB all support reading
                # all parquet files from a directory
                paths[table] = str(table_path)
            else:
                paths[table] = str(table_path)
        # Then check for single file format (building.parquet)
        elif (data_path / f"{table}.parquet").exists():
            paths[table] = str(data_path / f"{table}.parquet")
        # Finally check for any matching parquet files
        else:
            matches = list(data_path.glob(f"{table}*.parquet"))
            if matches:
                paths[table] = str(matches[0])

    return paths


class BaseBenchmark(ABC):
    """Base class for benchmark runners."""

    def __init__(self, data_paths: dict[str, str], engine_name: str):
        self.data_paths = data_paths
        self.engine_name = engine_name

    @abstractmethod
    def setup(self) -> None:
        """Initialize the benchmark environment."""
        pass

    @abstractmethod
    def teardown(self) -> None:
        """Cleanup the benchmark environment."""
        pass

    @abstractmethod
    def execute_query(self, query_name: str, query: str | None) -> tuple[int, Any]:
        """Execute a query and return (row_count, result)."""
        pass

    def dump_frame(self, result, query: str | None):
        """Convert a *timed* query result into a pandas DataFrame for dumping.

        Reuses the result from execute_query (no re-execution). DuckDB overrides
        this because its execute_query returns raw rows without column names.
        """
        return _to_pandas_frame(result)

class DuckDBBenchmark(BaseBenchmark):
    """DuckDB benchmark runner."""

    def __init__(self, data_paths: dict[str, str]):
        super().__init__(data_paths, "duckdb")
        self._conn = None
        self._last_columns = None

    def setup(self) -> None:
        import duckdb
        self._conn = duckdb.connect()
        self._conn.execute("LOAD spatial;")
        self._conn.execute("SET enable_external_file_cache = false;")
        for table, path in self.data_paths.items():
            # DuckDB needs glob pattern for directories, add /*.parquet if path is a directory
            parquet_path = path
            if Path(path).is_dir():
                parquet_path = str(Path(path) / "*.parquet")
            self._conn.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{parquet_path}')")

    def teardown(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def execute_query(self, query_name: str, query: str | None) -> tuple[int, Any]:
        rel = self._conn.execute(query)
        # Capture column names (cheap, from the cursor) so the timed fetchall result
        # can be turned into a DataFrame later without re-running the query.
        self._last_columns = [d[0] for d in rel.description]
        result = rel.fetchall()
        return len(result), result

    def dump_frame(self, result, query: str | None):
        import pandas as pd
        return pd.DataFrame(result, columns=self._last_columns)


class GeoPandasBenchmark(BaseBenchmark):
    """GeoPandas benchmark runner."""

    def __init__(self, data_paths: dict[str, str]):
        super().__init__(data_paths, "geopandas")
        self._queries = None

    def setup(self) -> None:
        import importlib.util
        geopandas_path = QUERY_MODULE_DIR / "geopandas_queries.py"
        spec = importlib.util.spec_from_file_location("geopandas_queries", geopandas_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._queries = {f"q{i}": getattr(module, f"q{i}") for i in range(1, QUERY_COUNT + 1)}

    def teardown(self) -> None:
        self._queries = None

    def execute_query(self, query_name: str, query: str | None) -> tuple[int, Any]:
        if query_name not in self._queries:
            raise ValueError(f"Query {query_name} not found")
        result = self._queries[query_name](self.data_paths)
        return len(result), result


class PublicApiModuleBenchmark(BaseBenchmark):
    """Runner for a public-API Python query module."""

    module_filename: str

    def __init__(self, data_paths: dict[str, str], engine_name: str):
        super().__init__(data_paths, engine_name)
        self._queries = None

    def setup(self) -> None:
        import importlib.util

        query_path = QUERY_MODULE_DIR / self.module_filename
        module_name = f"{self.engine_name}_queries"
        spec = importlib.util.spec_from_file_location(module_name, query_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._queries = {
            f"q{i}": getattr(module, f"q{i}")
            for i in range(1, QUERY_COUNT + 1)
        }

    def teardown(self) -> None:
        self._queries = None

    def execute_query(self, query_name: str, query: str | None) -> tuple[int, Any]:
        if query_name not in self._queries:
            raise ValueError(f"Query {query_name} not found")
        result = self._queries[query_name](self.data_paths)
        return len(result), result


class OptimizedGeoPandasBenchmark(PublicApiModuleBenchmark):
    """Hand-optimized implementation using only public GeoPandas APIs."""

    module_filename = "geopandas_optimized_queries.py"

    def __init__(self, data_paths: dict[str, str]):
        super().__init__(data_paths, "geopandas_optimized")


class VibeSpatialBenchmark(PublicApiModuleBenchmark):
    """Optimized implementation using only public vibeSpatial APIs."""

    module_filename = "vibespatial_queries.py"

    def __init__(self, data_paths: dict[str, str]):
        super().__init__(data_paths, "vibespatial")


class SedonaDBBenchmark(BaseBenchmark):
    """SedonaDB benchmark runner."""

    def __init__(self, data_paths: dict[str, str]):
        super().__init__(data_paths, "sedonadb")
        self._sedona = None

    def setup(self) -> None:
        import sedonadb
        self._sedona = sedonadb.connect()
        for table, path in self.data_paths.items():
            # SedonaDB needs glob pattern for directories
            parquet_path = path
            if Path(path).is_dir():
                parquet_path = str(Path(path) / "*.parquet")
            self._sedona.read_parquet(parquet_path).to_view(table, overwrite=True)

    def teardown(self) -> None:
        self._sedona = None

    def execute_query(self, query_name: str, query: str | None) -> tuple[int, Any]:
        result = self._sedona.sql(query).to_pandas()
        return len(result), result


class SpatialPolarsBenchmark(BaseBenchmark):
    """Spatial Polars benchmark runner."""

    def __init__(self, data_paths: dict[str, str]):
        super().__init__(data_paths, "spatial_polars")
        self._queries = None

    def setup(self) -> None:
        # spatial_polars package is already imported in _run_query_in_process
        # to register .spatial namespace before any module loading

        # Load query functions directly from the module
        import importlib.util
        query_file = QUERY_MODULE_DIR / "spatial_polars.py"
        spec = importlib.util.spec_from_file_location("spatial_polars_queries", query_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._queries = {f"q{i}": getattr(module, f"q{i}") for i in range(1, QUERY_COUNT + 1)}

    def teardown(self) -> None:
        self._queries = None

    def execute_query(self, query_name: str, query: str | None) -> tuple[int, Any]:
        if query_name not in self._queries:
            raise ValueError(f"Query {query_name} not found")
        result = self._queries[query_name](self.data_paths)
        return len(result), result


class PyCanopyBenchmark(BaseBenchmark):
    """PyCanopy benchmark runner."""

    def __init__(self, data_paths: dict[str, str]):
        super().__init__(data_paths, "pycanopy")
        self._queries = None

    def setup(self) -> None:
        import importlib.util
        query_file = QUERY_MODULE_DIR / "pycanopy_queries.py"
        spec = importlib.util.spec_from_file_location("pycanopy_queries", query_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._queries = {f"q{i}": getattr(module, f"q{i}") for i in range(1, QUERY_COUNT + 1)}

    def teardown(self) -> None:
        self._queries = None

    def execute_query(self, query_name: str, query: str | None) -> tuple[int, Any]:
        if query_name not in self._queries:
            raise ValueError(f"Query {query_name} not found")
        result = self._queries[query_name](self.data_paths)
        return len(result), result


def get_sql_queries(dialect: str) -> dict[str, str]:
    """Get SQL queries for a specific dialect from print_queries.py."""
    from print_queries import DuckDBSpatialBenchBenchmark, SedonaDBSpatialBenchBenchmark

    dialects = {
        "duckdb": DuckDBSpatialBenchBenchmark,
        "sedonadb": SedonaDBSpatialBenchBenchmark,
    }
    return dialects[dialect]().queries()


def run_query_isolated(
    engine_class: type,
    engine_name: str,
    data_paths: dict[str, str],
    query_name: str,
    query_sql: str | None,
    timeout: int,
    dump_csv: Path | None = None,
    measured_runs: int = 1,
    warmup_runs: int = 0,
    statistic: str = "mean",
    profile_telemetry: bool = False,
    profile_point_region: bool = False,
) -> BenchmarkResult:
    """Run a single query in an isolated subprocess with hard timeout.

    This is more robust than SIGALRM because:
    1. Native code (C++/Rust) can be forcefully terminated
    2. Memory-hungry queries don't affect the main process
    3. Crashed queries don't invalidate the benchmark runner

    If dump_csv is set, the timed result is also written there (reused, not re-run)
    for the correctness verify job.
    """
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_run_query_in_process,
        args=(result_queue, engine_class, data_paths, query_name, query_sql,
              str(dump_csv) if dump_csv else None, measured_runs, warmup_runs,
              statistic, profile_telemetry, profile_point_region),
    )

    def _kill():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=2)

    def _from_queue(result_data):
        return BenchmarkResult(
            query=query_name,
            engine=engine_name,
            time_seconds=result_data["time_seconds"],
            row_count=result_data["row_count"],
            status=result_data["status"],
            error_message=result_data["error_message"],
            run_times_seconds=result_data.get("run_times_seconds", []),
            statistic=result_data.get("statistic"),
            warmup_runs=result_data.get("warmup_runs", 0),
            telemetry_runs=result_data.get("telemetry_runs", []),
        )

    process.start()
    total_timeout = timeout * (warmup_runs + measured_runs)
    deadline = time.monotonic() + total_timeout
    result_data = None
    while time.monotonic() < deadline:
        try:
            result_data = result_queue.get(
                timeout=min(SERIALIZE_PROBE_SECONDS, deadline - time.monotonic())
            )
            break
        except queue.Empty:
            if not process.is_alive():
                break

    if result_data is not None:
        # The query has finished. Only optional result serialization remains, and it
        # gets a separate grace window. Draining the queue before join is essential:
        # telemetry payloads can exceed the pipe buffer and otherwise deadlock the
        # child feeder thread against a parent blocked in join().
        process.join(timeout=SERIALIZE_GRACE_SECONDS)
        if process.is_alive():
            _kill()
        return _from_queue(result_data)

    if process.is_alive():
        _kill()
        return BenchmarkResult(
            query=query_name,
            engine=engine_name,
            time_seconds=total_timeout,
            row_count=None,
            status="timeout",
            error_message=(
                f"Query {query_name} timed out after {total_timeout} seconds "
                "across warmup and measured runs (process killed)"
            ),
        )

    # Process exited without a readable result payload.
    try:
        return _from_queue(result_queue.get_nowait())
    except Exception:
        # Process died without putting result in queue
        return BenchmarkResult(
            query=query_name,
            engine=engine_name,
            time_seconds=None,
            row_count=None,
            status="error",
            error_message=f"Query {query_name} crashed (process exit code: {process.exitcode})",
        )


def run_benchmark(
    engine: str,
    data_paths: dict[str, str],
    queries: list[str] | None,
    timeout: int,
    scale_factor: float,
    runs: int = 3,
    warmup_runs: int = 0,
    statistic: str = "mean",
    profile_telemetry: bool = False,
    profile_point_region: bool = False,
    output_file: str | None = None,
    result_dir: Path | None = None,
) -> BenchmarkSuite:
    """Generic benchmark runner for any engine.

    Each query runs in an isolated subprocess to ensure:
    - Hard timeout enforcement (process can be killed)
    - Memory isolation (one query can't OOM the runner)
    - Crash isolation (one query crash doesn't affect others)

    Warmups and measured repetitions execute in the same isolated query
    process. Raw measured times and the requested summary statistic are
    persisted with the result.

    If output_file is provided, results are saved incrementally after each
    query so that partial results survive if the runner crashes mid-way.
    """

    from importlib.metadata import version as pkg_version

    # Engine configurations
    configs = {
        "duckdb": {
            "class": DuckDBBenchmark,
            "version_getter": lambda: __import__("duckdb").__version__,
            "queries_getter": lambda: get_sql_queries("duckdb"),
        },
        "geopandas": {
            "class": GeoPandasBenchmark,
            "version_getter": lambda: pkg_version("geopandas"),
            "queries_getter": lambda: {f"q{i}": None for i in range(1, QUERY_COUNT + 1)},
        },
        "geopandas_optimized": {
            "class": OptimizedGeoPandasBenchmark,
            "version_getter": lambda: pkg_version("geopandas"),
            "queries_getter": lambda: {f"q{i}": None for i in range(1, QUERY_COUNT + 1)},
        },
        "vibespatial": {
            "class": VibeSpatialBenchmark,
            "version_getter": lambda: pkg_version("vibespatial"),
            "queries_getter": lambda: {f"q{i}": None for i in range(1, QUERY_COUNT + 1)},
        },
        "sedonadb": {
            "class": SedonaDBBenchmark,
            "version_getter": lambda: pkg_version("sedonadb"),
            "queries_getter": lambda: get_sql_queries("sedonadb"),
        },
        "spatial_polars": {
            "class": SpatialPolarsBenchmark,
            "version_getter": lambda: pkg_version("spatial-polars"),
            "queries_getter": lambda: {f"q{i}": None for i in range(1, QUERY_COUNT + 1)},
        },
        "pycanopy": {
            "class": PyCanopyBenchmark,
            "version_getter": lambda: pkg_version("pycanopy"),
            "queries_getter": lambda: {f"q{i}": None for i in range(1, QUERY_COUNT + 1)},
        },
    }

    config = configs[engine]
    version = config["version_getter"]()

    # Format engine name for display
    display_name = engine.replace("_", " ").title()

    print(f"\n{'=' * 60}")
    print(f"Running {display_name} Benchmark")
    print(f"{'=' * 60}")
    print(f"{display_name} version: {version}")
    if runs > 1 or warmup_runs:
        print(
            f"Runs per query: {runs} measured + {warmup_runs} warmup "
            f"({statistic} will be reported)"
        )

    suite = BenchmarkSuite(engine=engine, scale_factor=scale_factor, version=version)
    all_queries = config["queries_getter"]()
    engine_class = config["class"]

    # Determine which queries will be run
    query_items = [
        (qname, qsql) for qname, qsql in all_queries.items()
        if not queries or qname in queries
    ]

    # Remove any stale dump from a previous run of these queries so a failed or
    # skipped capture can't leave behind another scale's result (dump filenames are
    # not scale-scoped, so a reused --result-dir could otherwise mix scales).
    if result_dir is not None:
        for query_name, _ in query_items:
            stale = result_dir / f"{engine}_{query_name}_result.csv"
            if stale.exists():
                stale.unlink()

    # Pre-populate all queries as "not_started" so even a total crash
    # (e.g. OOM killing the runner) leaves a file showing what was attempted
    for query_name, _ in query_items:
        suite.results.append(BenchmarkResult(
            query=query_name,
            engine=engine,
            time_seconds=None,
            row_count=None,
            status="not_started",
            error_message=None,
        ))
    if output_file:
        save_results([suite], output_file)

    # Install a SIGTERM handler so we flush results if the runner is shutting down
    def _sigterm_handler(signum, frame):
        print(f"\nReceived signal {signum}, saving partial results...", flush=True)
        if output_file:
            save_results([suite], output_file)
        sys.exit(128 + signum)

    prev_handler = signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        for idx, (query_name, query_sql) in enumerate(query_items):
            print(f"  Running {query_name}...", end=" ", flush=True)

            # First run — also dump the (reused) result for the verify job.
            dump_csv = result_dir / f"{engine}_{query_name}_result.csv" if result_dir else None
            result = run_query_isolated(
                engine_class=engine_class,
                engine_name=engine,
                data_paths=data_paths,
                query_name=query_name,
                query_sql=query_sql,
                timeout=timeout,
                dump_csv=dump_csv,
                measured_runs=runs,
                warmup_runs=warmup_runs,
                statistic=statistic,
                profile_telemetry=profile_telemetry,
                profile_point_region=profile_point_region,
            )

            if result.status == "success":
                samples = ", ".join(f"{value:.2f}" for value in result.run_times_seconds)
                print(
                    f"{result.time_seconds}s {result.statistic} "
                    f"({len(result.run_times_seconds)} runs: [{samples}], "
                    f"{result.row_count} rows)"
                )
            else:
                print(f"{result.status.upper()}: {result.error_message}")

            # Replace the pre-populated "not_started" entry with the actual result
            suite.results[idx] = result
            if result.status == "success":
                suite.total_time += result.time_seconds

            # Save partial results after each query so they survive crashes
            if output_file:
                save_results([suite], output_file)
    finally:
        signal.signal(signal.SIGTERM, prev_handler)

    return suite


def print_summary(results: list[BenchmarkSuite]) -> None:
    """Print a summary comparison table."""
    print(f"\n{'=' * 80}")
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    all_queries = sorted(
        {r.query for suite in results for r in suite.results},
        key=lambda x: int(x[1:])
    )

    data = {
        suite.engine: {
            r.query: f"{r.time_seconds:.2f}s" if r.status == "success" else r.status.upper()
            for r in suite.results
        }
        for suite in results
    }

    engines = [s.engine for s in results]
    header = f"{'Query':<10}" + "".join(f"{e:<15}" for e in engines)
    print(header)
    print("-" * len(header))

    for query in all_queries:
        row = f"{query:<10}" + "".join(f"{data.get(e, {}).get(query, 'N/A'):<15}" for e in engines)
        print(row)

    print("-" * len(header))
    print(f"{'Total':<10}" + "".join(f"{s.total_time:.2f}s{'':<9}" for s in results))


def save_results(results: list[BenchmarkSuite], output_file: str) -> None:
    """Save results to JSON file."""
    output = {
        "benchmark": "spatialbench",
        "version": "0.1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "results": [suite.to_dict() for suite in results],
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run SpatialBench benchmarks across SQL and public dataframe engines"
    )
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to directory containing benchmark data (parquet files)")
    parser.add_argument("--engines", type=str, default="geopandas_optimized,vibespatial",
                        help="Comma-separated list of engines to benchmark")
    parser.add_argument("--queries", type=str, default=None,
                        help="Comma-separated list of queries to run (e.g., q1,q2,q3)")
    parser.add_argument("--timeout", type=int, default=10,
                        help="Query timeout in seconds (default: 10)")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of measured runs per query (default: 3)")
    parser.add_argument("--warmup-runs", type=int, default=0,
                        help="Untimed warmup runs in each isolated query process")
    parser.add_argument("--statistic", choices=("mean", "median"), default="mean",
                        help="Summary statistic for measured query times")
    parser.add_argument("--profile-telemetry", action="store_true",
                        help="Collect vibeSpatial VRAM, RMM, D2H, and fallback telemetry")
    parser.add_argument(
        "--profile-point-region",
        action="store_true",
        help="Collect bounded exact point/region physical-work counters",
    )
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                        help="Output file for results")
    parser.add_argument("--scale-factor", type=float, default=1,
                        help="Scale factor of the data (for reporting only)")
    parser.add_argument("--result-dir", type=str, default=None,
                        help="If set, write each query's normalized result to "
                             "<result-dir>/<engine>_<query>_result.csv. The downstream "
                             "correctness verify job compares these to the committed answers "
                             "(and skips scale factors that have none), and the same dumps "
                             "bootstrap the answers for a new scale factor.")

    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs cannot be negative")

    engines = [e.strip().lower() for e in args.engines.split(",")]
    valid_engines = {"geopandas_optimized", "vibespatial"}

    for e in engines:
        if e not in valid_engines:
            print(f"Error: Unknown engine '{e}'. Valid options: {valid_engines}")
            sys.exit(1)

    queries = [q.strip().lower() for q in args.queries.split(",")] if args.queries else None

    data_paths = get_data_paths(args.data_dir)
    if not data_paths:
        print(f"Error: No data files found in {args.data_dir}")
        sys.exit(1)

    print("Data paths:")
    for table, path in data_paths.items():
        print(f"  {table}: {path}")

    # Dump normalized results (for the downstream verify job, and to bootstrap answers
    # for a new scale factor). Dumping reuses each timed run, so it is unconditional
    # when --result-dir is given; the verify job independently skips scale factors that
    # have no committed answers.
    result_dir = None
    if args.result_dir:
        result_dir = Path(args.result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        print(f"Dumping normalized results to {result_dir}")

    results = [
        run_benchmark(
            engine,
            data_paths,
            queries,
            args.timeout,
            args.scale_factor,
            runs=args.runs,
            warmup_runs=args.warmup_runs,
            statistic=args.statistic,
            profile_telemetry=args.profile_telemetry,
            profile_point_region=args.profile_point_region,
            output_file=args.output,
            result_dir=result_dir,
        )
        for engine in engines
    ]

    print_summary(results)
    save_results(results, args.output)

    # If result capture was requested, a query that ran successfully must have left a
    # dump. A missing one means serialization failed; fail loudly rather than let a
    # bootstrapping run (no committed answers to verify against) pass silently.
    if result_dir is not None:
        dump_failures = [
            f"{suite.engine}/{r.query}"
            for suite in results
            for r in suite.results
            if r.status == "success"
            and not (result_dir / f"{suite.engine}_{r.query}_result.csv").exists()
        ]
        if dump_failures:
            print(f"\nError: --result-dir was set but these successful queries produced no "
                  f"result dump (capture failed): {', '.join(dump_failures)}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    # Use 'spawn' on macOS to avoid issues with forking and native code
    # On Linux (GitHub Actions), 'fork' is default and usually works fine
    import platform
    if platform.system() == 'Darwin':
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
    main()
