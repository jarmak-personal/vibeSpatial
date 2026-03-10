from __future__ import annotations

import argparse
import json

from shapely.affinity import translate

from vibespatial import benchmark_segment_intersections, from_shapely_geometries
from vibespatial.testing.synthetic import SyntheticSpec, generate_lines


def build_datasets(rows: int):
    base = list(
        generate_lines(
            SyntheticSpec(geometry_type="line", distribution="grid", count=rows, seed=0)
        ).geometries
    )
    shifted = [translate(geometry, xoff=0.5, yoff=0.5) for geometry in base]
    return from_shapely_geometries(base), from_shapely_geometries(shifted)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark segment intersection primitives.")
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--tile-size", type=int, default=512)
    args = parser.parse_args(argv)

    left, right = build_datasets(args.rows)
    benchmark = benchmark_segment_intersections(left, right, tile_size=args.tile_size)
    print(
        json.dumps(
            {
                "rows_left": benchmark.rows_left,
                "rows_right": benchmark.rows_right,
                "candidate_pairs": benchmark.candidate_pairs,
                "disjoint_pairs": benchmark.disjoint_pairs,
                "proper_pairs": benchmark.proper_pairs,
                "touch_pairs": benchmark.touch_pairs,
                "overlap_pairs": benchmark.overlap_pairs,
                "ambiguous_pairs": benchmark.ambiguous_pairs,
                "elapsed_seconds": benchmark.elapsed_seconds,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
