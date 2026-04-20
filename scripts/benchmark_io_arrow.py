from __future__ import annotations

import argparse
import json
from pathlib import Path

from vibespatial.bench.catalog import ensure_operations_loaded
from vibespatial.bench.runner import run_operation
from vibespatial.bench.schema import BenchmarkResult

_PUBLIC_FORMATS = ("parquet", "geojson", "shapefile", "gpkg")
_SUITE_SCALES = {
    "smoke": (10_000,),
    "ci": (100_000,),
    "all": (10_000, 100_000, 1_000_000),
}


def _status_counts(results: list[BenchmarkResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return counts


def _suite_payload(*, suite: str, repeat: int) -> dict[str, object]:
    ensure_operations_loaded()
    results: list[BenchmarkResult] = []
    for input_format in _PUBLIC_FORMATS:
        for scale in _SUITE_SCALES[suite]:
            results.append(
                run_operation(
                    "io-arrow",
                    scale=scale,
                    repeat=repeat,
                    compare="shapely",
                    input_format=input_format,
                )
            )
    return {
        "schema_version": 2,
        "suite": suite,
        "metadata": {
            "scope": "io-arrow",
            "repeat": repeat,
            "public_api": True,
            "statuses": _status_counts(results),
            "notes": [
                "Runs the registered public io-arrow operation only.",
                "Internal GeoArrow/WKB/GeoParquet component rails live in tests/test_io_benchmark_rails.py.",
            ],
        },
        "results": [result.to_dict() for result in results],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark public Arrow/WKB IO APIs.")
    parser.add_argument("--suite", choices=tuple(_SUITE_SCALES), default=None)
    parser.add_argument("--format", choices=_PUBLIC_FORMATS, default="parquet")
    parser.add_argument("--scale", type=int, default=100_000)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.suite is not None:
        repeat = 1 if args.repeat is None else args.repeat
        payload = _suite_payload(suite=args.suite, repeat=repeat)
        text = json.dumps(payload, indent=2)
        if args.json_out is not None:
            args.json_out.write_text(text, encoding="utf-8")
        print(text)
        return 0

    repeat = 3 if args.repeat is None else args.repeat
    ensure_operations_loaded()
    result = run_operation(
        "io-arrow",
        scale=args.scale,
        repeat=repeat,
        compare="shapely",
        input_format=args.format,
    )
    print(result.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
