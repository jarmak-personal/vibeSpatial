"""NVBench kernel benchmark: Morton key computation and sort.

Requires cuda-bench: pip install cuda-bench[cu12]

Usage (standalone):
    python bench_morton_sort.py --scale 100000 --output-json results.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NVBench Morton sort benchmark")
    parser.add_argument("--scale", type=int, default=100_000)
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bandwidth", action="store_true")
    args = parser.parse_args(argv)

    try:
        import cuda.bench as bench
    except ImportError:
        sys.exit("cuda-bench not installed. Install with: pip install cuda-bench[cu12]")

    from vibespatial import from_shapely_geometries
    from vibespatial.kernels.core.geometry_analysis import (
        compute_geometry_bounds,
        compute_morton_keys,
    )
    from vibespatial.testing.synthetic import SyntheticSpec, generate_points

    dataset = generate_points(SyntheticSpec("point", "grid", count=args.scale, seed=0))
    owned = from_shapely_geometries(list(dataset.geometries))
    bounds = compute_geometry_bounds(owned)

    # Warmup
    compute_morton_keys(bounds)

    def morton_bench(state: bench.State) -> None:
        n = state.get_int64("NumElements")

        def launcher(launch: bench.Launch) -> None:
            compute_morton_keys(bounds)

        state.add_element_count(n)
        if args.bandwidth:
            state.add_global_memory_reads(n * 32)  # 4 doubles per bounds row
            state.add_global_memory_writes(n * 4)  # 1 uint32 per key
        state.exec(launcher)

    b = bench.register(morton_bench)
    b.add_int64_axis("NumElements", [args.scale])

    bench.run_all_benchmarks(
        ["--json", str(args.output_json)] + (sys.argv[1:] if argv is None else [])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
