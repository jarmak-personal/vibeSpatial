"""NVBench kernel benchmark: point-in-polygon.

Requires cuda-bench: pip install cuda-bench[cu12]

Usage (standalone):
    python bench_pip.py --scale 100000 --output-json results.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NVBench PiP kernel benchmark")
    parser.add_argument("--scale", type=int, default=100_000)
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bandwidth", action="store_true")
    args = parser.parse_args(argv)

    try:
        import cuda.bench as bench
    except ImportError:
        sys.exit("cuda-bench not installed. Install with: pip install cuda-bench[cu12]")

    import numpy as np

    from vibespatial import ExecutionMode, from_shapely_geometries
    from vibespatial.kernels.predicates.point_in_polygon import point_in_polygon
    from vibespatial.testing.synthetic import SyntheticSpec, generate_points, generate_polygons

    polygon_base_count = max(args.scale // 8, 1)
    points = np.asarray(
        list(generate_points(SyntheticSpec("point", "grid", count=args.scale, seed=0)).geometries),
        dtype=object,
    )
    polygons = np.resize(
        np.asarray(
            list(generate_polygons(
                SyntheticSpec("polygon", "regular-grid", count=polygon_base_count, seed=1, vertices=5)
            ).geometries),
            dtype=object,
        ),
        args.scale,
    ).astype(object, copy=False)

    points_owned = from_shapely_geometries(points.tolist())
    polygons_owned = from_shapely_geometries(polygons.tolist())

    # Warmup
    point_in_polygon(points_owned, polygons_owned, dispatch_mode=ExecutionMode.GPU)

    def pip_bench(state: bench.State) -> None:
        n = state.get_int64("NumElements")

        def launcher(launch: bench.Launch) -> None:
            point_in_polygon(points_owned, polygons_owned, dispatch_mode=ExecutionMode.GPU)

        state.add_element_count(n)
        state.exec(launcher, sync=True)

    b = bench.register(pip_bench)
    b.add_int64_axis("NumElements", [args.scale])

    bench.run_all_benchmarks(
        ["--json", str(args.output_json)]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
