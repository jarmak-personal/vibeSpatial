from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from vibespatial import benchmark_geojson_ingest, benchmark_shapefile_ingest
from vibespatial.io_benchmark_rails import benchmark_io_file_suite, io_suite_to_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark hybrid file-format IO strategies.")
    parser.add_argument("--suite", choices=("smoke", "ci", "all"), default=None)
    parser.add_argument("--format", default="geojson")
    parser.add_argument("--geometry-type", default="point")
    parser.add_argument("--scale", type=int, default=100_000)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.suite is not None:
        repeat = 1 if args.repeat is None else args.repeat
        payload = io_suite_to_json(
            benchmark_io_file_suite(suite=args.suite, repeat=repeat),
            suite=args.suite,
            repeat=repeat,
            scope="io-file",
        )
        if args.json_out is not None:
            args.json_out.write_text(payload, encoding="utf-8")
        print(payload)
        return 0

    repeat = 5 if args.repeat is None else args.repeat
    if args.format == "geojson":
        benchmarks = benchmark_geojson_ingest(
            geometry_type=args.geometry_type,
            rows=args.scale,
            repeat=repeat,
        )
    elif args.format == "shapefile":
        benchmarks = benchmark_shapefile_ingest(
            geometry_type=args.geometry_type,
            rows=args.scale,
            repeat=repeat,
        )
    else:
        raise SystemExit(f"Unsupported benchmark format: {args.format}")
    payload = {
        "format": args.format,
        "geometry_type": args.geometry_type,
        "rows": args.scale,
        "benchmarks": [asdict(item) for item in benchmarks],
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
