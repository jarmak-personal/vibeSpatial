from __future__ import annotations

import argparse
import json

from vibespatial import benchmark_bounds_pairs, from_shapely_geometries
from vibespatial.testing.synthetic import SyntheticSpec, generate_points


def build_dataset(kind: str, rows: int):
    distribution = "uniform" if kind == "uniform" else "clustered"
    dataset = generate_points(
        SyntheticSpec(geometry_type="point", distribution=distribution, count=rows, seed=0)
    )
    return from_shapely_geometries(list(dataset.geometries))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark tiled bounds-pair generation.")
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--tile-size", type=int, default=256)
    args = parser.parse_args(argv)

    results = []
    for dataset in ("uniform", "skewed"):
        owned = build_dataset(dataset, args.rows)
        benchmark = benchmark_bounds_pairs(owned, dataset=dataset, tile_size=args.tile_size)
        results.append(
            {
                "dataset": benchmark.dataset,
                "rows": benchmark.rows,
                "tile_size": benchmark.tile_size,
                "elapsed_seconds": benchmark.elapsed_seconds,
                "pairs_examined": benchmark.pairs_examined,
                "candidate_pairs": benchmark.candidate_pairs,
            }
        )
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
