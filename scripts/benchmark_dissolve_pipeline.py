from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from shapely.geometry import Point

import geopandas
from vibespatial import benchmark_dissolve_pipeline


def _build_frame(rows: int, groups: int) -> geopandas.GeoDataFrame:
    xs = np.linspace(0.0, float(rows - 1), rows, dtype=np.float64)
    ys = xs * 0.5
    labels = np.arange(rows, dtype=np.int32) % max(groups, 1)
    return geopandas.GeoDataFrame(
        {
            "group": pd.Categorical(labels),
            "value": np.arange(rows, dtype=np.int64),
            "geometry": [Point(float(x), float(y)) for x, y in zip(xs, ys, strict=True)],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the grouped dissolve pipeline against the baseline path.",
    )
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--groups", type=int, default=128)
    parser.add_argument(
        "--iterations", type=int, default=5,
        help="Number of timed iterations (reports median). Default: 5.",
    )
    parser.add_argument(
        "--warmup", type=int, default=1,
        help="Number of warmup runs discarded before timing. Default: 1.",
    )
    args = parser.parse_args()

    frame = _build_frame(args.rows, args.groups)
    benchmark = benchmark_dissolve_pipeline(
        frame,
        by="group",
        dataset="points",
        iterations=args.iterations,
        warmup=args.warmup,
    )
    print(
        json.dumps(
            {
                "dataset": benchmark.dataset,
                "rows": benchmark.rows,
                "groups": benchmark.groups,
                "iterations": benchmark.iterations,
                "pipeline_elapsed_seconds": round(benchmark.pipeline_elapsed_seconds, 6),
                "baseline_elapsed_seconds": round(benchmark.baseline_elapsed_seconds, 6),
                "speedup_vs_baseline": round(benchmark.speedup_vs_baseline, 2),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
