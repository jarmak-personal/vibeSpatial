"""NVBench kernel benchmark: segment intersection classification.

Requires cuda-bench: pip install cuda-bench[cu12]

Usage (standalone):
    python bench_segment_intersection.py --scale 500 --output-json results.json

The benchmark exercises the full GPU-native pipeline:
  Kernel 1: GPU segment extraction (NVRTC count-scatter)
  Kernel 2: GPU candidate generation (sort-sweep with CCCL radix sort)
  Kernel 3: GPU classification with Shewchuk adaptive refinement
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NVBench segment intersection benchmark")
    parser.add_argument("--scale", type=int, default=500)
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--dispatch", default="auto", choices=["auto", "gpu", "cpu"])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bandwidth", action="store_true")
    args = parser.parse_args(argv)

    try:
        import cuda.bench as bench
    except ImportError:
        sys.exit("cuda-bench not installed. Install with: pip install cuda-bench[cu12]")

    from shapely.affinity import translate

    from vibespatial import (
        classify_segment_intersections,
        from_shapely_geometries,
    )
    from vibespatial.runtime import ExecutionMode
    from vibespatial.testing.synthetic import SyntheticSpec, generate_lines

    base = list(generate_lines(SyntheticSpec("line", "grid", count=args.scale, seed=0)).geometries)
    shifted = [translate(g, xoff=0.5, yoff=0.5) for g in base]
    left = from_shapely_geometries(base)
    right = from_shapely_geometries(shifted)

    dispatch_mode = ExecutionMode(args.dispatch)

    # Warmup: full pipeline including GPU extraction + candidate gen + classify
    classify_segment_intersections(
        left, right,
        dispatch_mode=dispatch_mode,
        precision=args.precision,
    )

    def seg_bench(state: bench.State) -> None:
        n = state.get_int64("NumElements")

        def launcher(launch: bench.Launch) -> None:
            classify_segment_intersections(
                left, right,
                dispatch_mode=dispatch_mode,
                precision=args.precision,
            )

        state.add_element_count(n)
        state.exec(launcher, sync=True)

    b = bench.register(seg_bench)
    b.add_int64_axis("NumElements", [args.scale])

    bench.run_all_benchmarks(
        ["--json", str(args.output_json)]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
