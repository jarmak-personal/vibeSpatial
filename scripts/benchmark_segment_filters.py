from __future__ import annotations

import argparse
import json

from shapely.affinity import translate

from vibespatial import benchmark_segment_filter, from_shapely_geometries
from vibespatial.testing.synthetic import SyntheticSpec, generate_polygons


def build_datasets(rows: int):
    base = list(
        generate_polygons(
            SyntheticSpec(geometry_type="polygon", distribution="star", count=rows, seed=0)
        ).geometries
    )
    shifted = [translate(geometry, xoff=5.0, yoff=0.0) for geometry in base]
    return from_shapely_geometries(base), from_shapely_geometries(shifted)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark segment-level MBR filtering.")
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--tile-size", type=int, default=512)
    args = parser.parse_args(argv)

    left, right = build_datasets(args.rows)
    benchmark = benchmark_segment_filter(left, right, tile_size=args.tile_size)
    print(
        json.dumps(
            {
                "rows_left": benchmark.rows_left,
                "rows_right": benchmark.rows_right,
                "naive_segment_pairs": benchmark.naive_segment_pairs,
                "filtered_segment_pairs": benchmark.filtered_segment_pairs,
                "elapsed_seconds": benchmark.elapsed_seconds,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
