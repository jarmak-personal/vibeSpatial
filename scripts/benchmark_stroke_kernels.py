from __future__ import annotations

import argparse

from shapely.geometry import LineString, Point

from vibespatial import benchmark_offset_curve, benchmark_point_buffer


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark repo-owned stroke kernels against Shapely baselines.")
    parser.add_argument("--kind", choices=("point-buffer", "offset-curve"), default="point-buffer")
    parser.add_argument("--rows", type=int, default=1000)
    args = parser.parse_args()

    if args.kind == "point-buffer":
        values = [Point(float(index), float(index) * 0.25) for index in range(args.rows)]
        benchmark = benchmark_point_buffer(values, distance=1.0, quad_segs=8)
    else:
        values = [
            LineString([(float(index), 0.0), (float(index), 2.0), (float(index) + 2.0, 2.0)])
            for index in range(args.rows)
        ]
        benchmark = benchmark_offset_curve(values, distance=1.0, join_style="mitre")

    print(
        {
            "dataset": benchmark.dataset,
            "rows": benchmark.rows,
            "fast_rows": benchmark.fast_rows,
            "fallback_rows": benchmark.fallback_rows,
            "owned_elapsed_seconds": benchmark.owned_elapsed_seconds,
            "shapely_elapsed_seconds": benchmark.shapely_elapsed_seconds,
            "speedup_vs_shapely": benchmark.speedup_vs_shapely,
        }
    )


if __name__ == "__main__":
    main()
