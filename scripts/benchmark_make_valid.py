from __future__ import annotations

import argparse

from shapely.geometry import Polygon

from vibespatial import benchmark_make_valid


def _valid_polygon(offset: float) -> Polygon:
    return Polygon([(offset, 0.0), (offset, 1.0), (offset + 1.0, 1.0), (offset + 1.0, 0.0)])


def _invalid_polygon(offset: float) -> Polygon:
    return Polygon([(offset, 0.0), (offset + 1.0, 1.0), (offset + 1.0, 2.0), (offset + 1.0, 1.0), (offset, 0.0)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark compact-invalid-row make_valid against the baseline path.")
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--invalid-every", type=int, default=20)
    args = parser.parse_args()

    values = [
        _invalid_polygon(float(index)) if index % max(args.invalid_every, 1) == 0 else _valid_polygon(float(index))
        for index in range(args.rows)
    ]
    benchmark = benchmark_make_valid(values)
    print(
        {
            "dataset": benchmark.dataset,
            "rows": benchmark.rows,
            "repaired_rows": benchmark.repaired_rows,
            "compact_elapsed_seconds": benchmark.compact_elapsed_seconds,
            "baseline_elapsed_seconds": benchmark.baseline_elapsed_seconds,
            "speedup_vs_baseline": benchmark.speedup_vs_baseline,
        }
    )


if __name__ == "__main__":
    main()
