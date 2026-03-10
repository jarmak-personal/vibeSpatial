from __future__ import annotations

import argparse
import json

from vibespatial import benchmark_clip_by_rect, from_shapely_geometries
from vibespatial.testing.synthetic import SyntheticSpec, generate_lines, generate_polygons


def build_dataset(kind: str, rows: int):
    if kind == "line":
        dataset = generate_lines(SyntheticSpec("line", "grid", count=rows, seed=0))
    elif kind == "polygon":
        dataset = generate_polygons(SyntheticSpec("polygon", "regular-grid", count=rows, seed=0))
    else:
        raise ValueError(f"unsupported dataset kind: {kind}")
    return from_shapely_geometries(list(dataset.geometries))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark owned rectangle clip fast paths.")
    parser.add_argument("--kind", choices=("line", "polygon"), default="line")
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--rect", nargs=4, type=float, default=(100.0, 100.0, 700.0, 700.0))
    args = parser.parse_args(argv)

    values = build_dataset(args.kind, args.rows)
    benchmark = benchmark_clip_by_rect(values, *args.rect, dataset=f"{args.kind}-{args.rows}")
    print(
        json.dumps(
            {
                "dataset": benchmark.dataset,
                "rows": benchmark.rows,
                "candidate_rows": benchmark.candidate_rows,
                "fast_rows": benchmark.fast_rows,
                "fallback_rows": benchmark.fallback_rows,
                "owned_elapsed_seconds": benchmark.owned_elapsed_seconds,
                "shapely_elapsed_seconds": benchmark.shapely_elapsed_seconds,
                "speedup_vs_shapely": benchmark.speedup_vs_shapely,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
