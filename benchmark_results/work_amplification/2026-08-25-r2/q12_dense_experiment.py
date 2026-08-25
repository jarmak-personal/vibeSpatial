#!/usr/bin/env python3
"""Reproduce the R2 SF100 Q12 indexed-vs-dense comparison with identity."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

CAPSULE = Path(__file__).resolve().parent
REPO = CAPSULE.parents[2]
DEFAULT_DATA = Path("/home/picard/datasets/spatialbench/v0.1.0/sf100-geoparquet")
IMPLEMENTATION = (
    REPO
    / "benchmark_results"
    / "experiments"
    / "2026-08-21-q11-q12-physical-shapes"
    / "experiment.py"
)
IMPLEMENTATION_SHA256 = (
    "9886b6a2fcdd766feb347b8c2b1acb93069ffcf7ed0286f834619e4a301830ef"
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _run_text(*args: str) -> str:
    return subprocess.run(
        args,
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_tree_identity(*extra_paths: Path) -> dict[str, object]:
    paths = [
        *sorted(
            path
            for path in (REPO / "src" / "vibespatial").rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        ),
        REPO / "benchmarks" / "spatialbench" / "run_benchmark.py",
        REPO / "benchmarks" / "spatialbench" / "vibespatial_queries.py",
        REPO / "pyproject.toml",
        REPO / "uv.lock",
        IMPLEMENTATION,
        Path(__file__).resolve(),
        *extra_paths,
    ]
    hashes = {
        str(path.relative_to(REPO)): _sha256_file(path)
        for path in paths
        if path.is_file()
    }
    encoded = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    status = _run_text("git", "status", "--porcelain=v1").splitlines()
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=REPO,
        check=False,
        capture_output=True,
    ).stdout
    return {
        "git_head": _run_text("git", "rev-parse", "HEAD"),
        "git_status": status,
        "git_diff_binary_sha256": _sha256_bytes(diff),
        "relevant_source_manifest_sha256": _sha256_bytes(encoded),
        "critical_file_sha256": {
            name: hashes[name]
            for name in (
                "benchmarks/spatialbench/run_benchmark.py",
                "benchmarks/spatialbench/vibespatial_queries.py",
                "pyproject.toml",
                "uv.lock",
                str(IMPLEMENTATION.relative_to(REPO)),
                str(Path(__file__).resolve().relative_to(REPO)),
                *(str(path.resolve().relative_to(REPO)) for path in extra_paths),
            )
        },
    }


def _dataset_identity(data: Path) -> dict[str, object]:
    manifest = data / "geoparquet_manifest.json"
    inventory_lines = []
    total_bytes = 0
    for path in sorted(item for item in data.rglob("*") if item.is_file()):
        size = path.stat().st_size
        total_bytes += size
        inventory_lines.append(f"{path.relative_to(data)}\t{size}\n")
    manifest_payload = json.loads(manifest.read_text())
    return {
        "path": str(data),
        "scale_factor": 100,
        "format": manifest_payload["format"],
        "geometry_encoding": manifest_payload["geometry_encoding"],
        "manifest_sha256": _sha256_file(manifest),
        "file_inventory_sha256": _sha256_bytes("".join(inventory_lines).encode()),
        "file_inventory_fingerprint_definition": (
            "sha256 of sorted tab-separated relative path and byte size; "
            "not a content digest"
        ),
        "file_count": len(inventory_lines),
        "total_bytes": total_bytes,
        "tables": manifest_payload["tables"],
    }


def _environment_identity() -> dict[str, object]:
    packages = {}
    for name in (
        "cupy-cuda12x",
        "cudf-cu12",
        "geopandas",
        "numpy",
        "pandas",
        "pyarrow",
        "pylibcudf-cu12",
        "vibespatial",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cpu_model = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    meminfo = Path("/proc/meminfo")
    memory_total = None
    if meminfo.is_file():
        memory_total = next(
            (
                line.split(":", 1)[1].strip()
                for line in meminfo.read_text().splitlines()
                if line.startswith("MemTotal:")
            ),
            None,
        )
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_model": cpu_model,
        "memory_total": memory_total,
        "python": sys.version,
        "python_executable": sys.executable,
        "package_versions": packages,
        "nvidia_smi": _run_text(
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,power.limit",
            "--format=csv,noheader,nounits",
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "strict_native": os.environ.get("VIBESPATIAL_STRICT_NATIVE", "1"),
        "precompile": os.environ.get("VIBESPATIAL_PRECOMPILE"),
    }


def _identity(data: Path, lane: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "complete_contemporaneous",
        "captured_at": datetime.now(UTC).isoformat(),
        "source": _source_tree_identity(),
        "dataset": _dataset_identity(data),
        "environment": _environment_identity(),
        "measurement": {
            "query": "SF100 Q12",
            "lane": lane,
            "process_isolation": "one command/process per arm",
            "warmup_runs": 0,
            "repeat_runs": 1,
            "statistic": "single cold observation",
            "clock": "time.perf_counter wall time",
            "timed_scope": "queries.q12(paths)",
            "excluded_scope": "query object and data-path setup",
            "distance_tolerance": {"rtol": 1e-6, "atol": 1e-9},
            "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        },
    }


def _load_implementation(data: Path):
    actual_sha256 = _sha256_file(IMPLEMENTATION)
    if actual_sha256 != IMPLEMENTATION_SHA256:
        raise RuntimeError(
            "Q12 implementation source changed: "
            f"expected {IMPLEMENTATION_SHA256}, found {actual_sha256}"
        )
    spec = importlib.util.spec_from_file_location("r2_q12_implementation", IMPLEMENTATION)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {IMPLEMENTATION}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DATA = data
    module.RAW = CAPSULE
    return module


def _write_identified(source: Path, output: Path, identity: dict[str, object]) -> None:
    payload = json.loads(source.read_text())
    payload["evidence_identity"] = identity
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")


def run_indexed(data: Path, output: Path) -> None:
    identity = _identity(data, "indexed_public_nearest")
    implementation = _load_implementation(data)
    implementation.run_q12_baseline()
    _write_identified(CAPSULE / "q12_baseline.json", output, identity)


def run_dense(data: Path, output: Path) -> None:
    reference = CAPSULE / "q12_baseline_result.csv"
    if not reference.is_file():
        raise FileNotFoundError(
            f"{reference} is missing; run the indexed arm first in this capsule"
        )
    identity = _identity(data, "dense_bbox_filter_public_exact_distance")
    identity["measurement"]["reference_result"] = {
        "path": str(reference.relative_to(REPO)),
        "sha256": _sha256_file(reference),
    }
    implementation = _load_implementation(data)
    implementation.run_q12_full_dense()
    _write_identified(CAPSULE / "q12_full_dense_filter.json", output, identity)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lane", choices=("indexed", "dense"))
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.lane == "indexed":
        output = args.output or CAPSULE / "q12_indexed_current.json"
        run_indexed(args.data.resolve(), output.resolve())
    else:
        output = args.output or CAPSULE / "q12_dense_filter_current.json"
        run_dense(args.data.resolve(), output.resolve())


if __name__ == "__main__":
    main()
