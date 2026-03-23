"""NVBench kernel benchmark: polygon intersection.

Requires cuda-bench: pip install cuda-bench[cu12]

Usage (standalone):
    python bench_polygon_intersection.py --scale 10000 --output-json results.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NVBench polygon intersection benchmark")
    parser.add_argument("--scale", type=int, default=10_000)
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bandwidth", action="store_true")
    args = parser.parse_args(argv)

    try:
        import cuda.bench as bench
    except ImportError:
        sys.exit("cuda-bench not installed. Install with: pip install cuda-bench[cu12]")

    import numpy as np
    from shapely.geometry import box

    from vibespatial import from_shapely_geometries
    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    rng = np.random.default_rng(42)
    n = args.scale
    left_geoms = []
    right_geoms = []
    for _ in range(n):
        x, y = rng.uniform(0, 100, 2)
        w, h = rng.uniform(1, 10, 2)
        left_geoms.append(box(x, y, x + w, y + h))
        x2, y2 = rng.uniform(0, 100, 2)
        w2, h2 = rng.uniform(1, 10, 2)
        right_geoms.append(box(x2, y2, x2 + w2, y2 + h2))

    left = from_shapely_geometries(left_geoms)
    right = from_shapely_geometries(right_geoms)

    # Warmup
    polygon_intersection(left, right)

    def intersection_bench(state: bench.State) -> None:
        elem_count = state.get_int64("NumElements")

        def launcher(launch: bench.Launch) -> None:
            polygon_intersection(left, right)

        state.add_element_count(elem_count)
        state.exec(launcher, sync=True)

    b = bench.register(intersection_bench)
    b.add_int64_axis("NumElements", [n])

    bench.run_all_benchmarks(
        ["--json", str(args.output_json)]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
