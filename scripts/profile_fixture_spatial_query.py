from __future__ import annotations

import argparse
import json

from vibespatial.fixture_profiles import (
    ensure_named_fixture,
    profile_fixture_nearest,
    profile_fixture_query,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile public GeoPandas spatial query/nearest against cached GeoParquet fixtures.")
    parser.add_argument("--fixture", required=True, help="Tree fixture name or parquet path.")
    parser.add_argument("--operation", choices=("query", "nearest"), default="query")
    parser.add_argument("--query-mode", choices=("self", "fixture", "translated-self"), default="self")
    parser.add_argument("--query-fixture", help="Optional query fixture name or parquet path.")
    parser.add_argument("--predicate", default="intersects")
    parser.add_argument("--output-format", choices=("indices", "dense", "sparse"), default="indices")
    parser.add_argument("--sort", action="store_true")
    parser.add_argument("--max-distance", type=float)
    parser.add_argument("--return-distance", action="store_true")
    parser.add_argument("--return-all", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclusive", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--ensure", action="store_true", help="Create the requested fixture name before profiling.")
    args = parser.parse_args(argv)

    if args.ensure and not args.fixture.endswith(".parquet"):
        ensure_named_fixture(args.fixture)
    if args.ensure and args.query_fixture and not args.query_fixture.endswith(".parquet"):
        ensure_named_fixture(args.query_fixture)

    traces = []
    for _ in range(args.repeat):
        if args.operation == "query":
            trace = profile_fixture_query(
                args.fixture,
                query_mode=args.query_mode,
                query_fixture=args.query_fixture,
                predicate=args.predicate,
                sort=args.sort,
                output_format=args.output_format,
            )
        else:
            trace = profile_fixture_nearest(
                args.fixture,
                query_mode=args.query_mode,
                query_fixture=args.query_fixture,
                max_distance=args.max_distance,
                return_all=args.return_all,
                return_distance=args.return_distance,
                exclusive=args.exclusive,
            )
        traces.append(trace.to_dict())

    print(
        json.dumps(
            {
                "fixture": args.fixture,
                "operation": args.operation,
                "query_mode": args.query_mode,
                "query_fixture": args.query_fixture,
                "predicate": args.predicate,
                "output_format": args.output_format,
                "sort": args.sort,
                "repeat": args.repeat,
                "traces": traces,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
