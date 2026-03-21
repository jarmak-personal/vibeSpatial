"""NVBench kernel benchmark: rectangle clipping.

Requires cuda-bench: pip install cuda-bench[cu12]

Usage (standalone):
    python bench_clip_rect.py --scale 10000 --output-json results.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NVBench clip rect benchmark")
    parser.add_argument("--scale", type=int, default=10_000)
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bandwidth", action="store_true")
    args = parser.parse_args(argv)

    try:
        import cuda.bench as bench
    except ImportError:
        sys.exit("cuda-bench not installed. Install with: pip install cuda-bench[cu12]")

    from vibespatial import from_shapely_geometries
    from vibespatial.constructive.clip_rect import clip_by_rect_owned
    from vibespatial.testing.synthetic import SyntheticSpec, generate_lines

    dataset = generate_lines(SyntheticSpec("line", "grid", count=args.scale, seed=0))
    owned = from_shapely_geometries(list(dataset.geometries))
    rect = (100.0, 100.0, 700.0, 700.0)

    # Warmup
    clip_by_rect_owned(owned, *rect)

    def clip_bench(state: bench.State) -> None:
        n = state.get_int64("NumElements")

        def launcher(launch: bench.Launch) -> None:
            clip_by_rect_owned(owned, *rect)

        state.add_element_count(n)
        state.exec(launcher)

    b = bench.register(clip_bench)
    b.add_int64_axis("NumElements", [args.scale])

    bench.run_all_benchmarks(
        ["--json", str(args.output_json)] + (sys.argv[1:] if argv is None else [])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
