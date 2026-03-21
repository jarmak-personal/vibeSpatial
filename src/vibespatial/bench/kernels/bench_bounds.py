"""NVBench kernel benchmark: geometry bounds computation.

Requires cuda-bench: pip install cuda-bench[cu12]

Usage (standalone):
    python bench_bounds.py --scale 100000 --output-json results.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NVBench bounds kernel benchmark")
    parser.add_argument("--scale", type=int, default=100_000)
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bandwidth", action="store_true")
    args = parser.parse_args(argv)

    try:
        import cuda.bench as bench
    except ImportError:
        sys.exit("cuda-bench not installed. Install with: pip install cuda-bench[cu12]")

    from shapely.geometry import Point

    from vibespatial import from_shapely_geometries
    from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds

    points = [Point(float(i), float((i * 7) % 1000)) for i in range(args.scale)]
    owned = from_shapely_geometries(points)

    # Warmup
    compute_geometry_bounds(owned)

    def bounds_bench(state: bench.State) -> None:
        n = state.get_int64("NumElements")

        def launcher(launch: bench.Launch) -> None:
            compute_geometry_bounds(owned)

        state.add_element_count(n)
        if args.bandwidth:
            state.add_global_memory_reads(n * 16)  # xy coords per point
            state.add_global_memory_writes(n * 32)  # 4 doubles per bounds
        state.exec(launcher)

    b = bench.register(bounds_bench)
    b.add_int64_axis("NumElements", [args.scale])

    bench.run_all_benchmarks(
        ["--json", str(args.output_json)] + (sys.argv[1:] if argv is None else [])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
