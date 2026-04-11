from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from vibespatial import (
    benchmark_geoarrow_bridge,
    benchmark_geoparquet_planner,
    benchmark_geoparquet_scan_engine,
    benchmark_native_geometry_codec,
    benchmark_wkb_bridge,
    build_geoparquet_metadata_summary,
)
from vibespatial.bench.io_benchmark_rails import benchmark_io_arrow_suite, io_suite_to_json


def build_covering_summary(*, scale: int, selectivity: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    row_group_count = max(125, scale // 8_000)
    rows_per_group = max(1, scale // row_group_count)
    grid_width = max(12, int(np.ceil(np.sqrt(row_group_count))))
    cell = 1_000.0 / grid_width
    centers_x = []
    centers_y = []
    for index in range(row_group_count):
        row = index // grid_width
        col = index % grid_width
        centers_x.append((col + 0.5) * cell)
        centers_y.append((row + 0.5) * cell)
    centers_x = np.asarray(centers_x, dtype=np.float64)
    centers_y = np.asarray(centers_y, dtype=np.float64)
    spread = cell * 0.45
    widths = rng.uniform(spread * 0.6, spread, row_group_count)
    heights = rng.uniform(spread * 0.6, spread, row_group_count)
    xmin = centers_x - widths
    ymin = centers_y - heights
    xmax = centers_x + widths
    ymax = centers_y + heights
    row_group_rows = np.full(row_group_count, rows_per_group, dtype=np.int64)

    target_width = 1_000.0 * np.sqrt(max(min(selectivity, 1.0), 0.0001))
    bbox = (
        500.0 - target_width / 2.0,
        500.0 - target_width / 2.0,
        500.0 + target_width / 2.0,
        500.0 + target_width / 2.0,
    )
    summary = build_geoparquet_metadata_summary(
        source="covering_bbox",
        row_group_rows=row_group_rows,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
    )
    return summary, bbox


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark IO adapter strategies.")
    parser.add_argument("--suite", choices=("smoke", "ci", "all"), default=None)
    parser.add_argument("--format", default="geoparquet")
    parser.add_argument("--scale", type=int, default=1_000_000)
    parser.add_argument("--selectivity", type=float, default=0.1)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--operation", default="decode")
    parser.add_argument("--geometry-type", default="point")
    parser.add_argument("--geometry-encoding", default="geoarrow")
    parser.add_argument("--chunk-rows", type=int, default=None)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.suite is not None:
        repeat = 1 if args.repeat is None else args.repeat
        payload = io_suite_to_json(
            benchmark_io_arrow_suite(suite=args.suite, repeat=repeat),
            suite=args.suite,
            repeat=repeat,
            scope="io-arrow",
        )
        if args.json_out is not None:
            args.json_out.write_text(payload, encoding="utf-8")
        print(payload)
        return 0

    repeat = 20 if args.repeat is None else args.repeat
    if args.format == "geoparquet":
        if args.operation == "planner":
            summary, bbox = build_covering_summary(scale=args.scale, selectivity=args.selectivity)
            benchmarks = benchmark_geoparquet_planner(summary, bbox, repeat=repeat)
            payload = {
                "format": args.format,
                "operation": args.operation,
                "scale": args.scale,
                "selectivity": args.selectivity,
                "row_group_count": summary.row_group_count,
                "total_rows": summary.total_rows,
                "benchmarks": [asdict(item) for item in benchmarks],
            }
            print(json.dumps(payload, indent=2))
            return 0
        benchmark = benchmark_geoparquet_scan_engine(
            geometry_type=args.geometry_type,
            rows=args.scale,
            geometry_encoding=args.geometry_encoding,
            chunk_rows=args.chunk_rows,
            backend=args.backend,
            repeat=repeat,
        )
        payload = {
            "format": args.format,
            "operation": args.operation,
            "geometry_type": args.geometry_type,
            "geometry_encoding": args.geometry_encoding,
            "rows": args.scale,
            "chunk_rows": args.chunk_rows,
            "backend": args.backend,
            "benchmark": asdict(benchmark),
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.format == "geoarrow":
        benchmarks = benchmark_geoarrow_bridge(
            operation=args.operation,
            geometry_type=args.geometry_type,
            rows=args.scale,
            repeat=repeat,
        )
        payload = {
            "format": args.format,
            "operation": args.operation,
            "geometry_type": args.geometry_type,
            "rows": args.scale,
            "benchmarks": [asdict(item) for item in benchmarks],
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.format == "native-geometry":
        benchmarks = benchmark_native_geometry_codec(
            operation=args.operation,
            geometry_type=args.geometry_type,
            rows=args.scale,
            repeat=repeat,
        )
        payload = {
            "format": args.format,
            "operation": args.operation,
            "geometry_type": args.geometry_type,
            "rows": args.scale,
            "benchmarks": [asdict(item) for item in benchmarks],
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.format == "wkb":
        benchmarks = benchmark_wkb_bridge(
            operation=args.operation,
            geometry_type=args.geometry_type,
            rows=args.scale,
            repeat=repeat,
        )
        payload = {
            "format": args.format,
            "operation": args.operation,
            "geometry_type": args.geometry_type,
            "rows": args.scale,
            "benchmarks": [asdict(item) for item in benchmarks],
        }
        print(json.dumps(payload, indent=2))
        return 0

    raise SystemExit(f"Unsupported benchmark format: {args.format}")


if __name__ == "__main__":
    raise SystemExit(main())
