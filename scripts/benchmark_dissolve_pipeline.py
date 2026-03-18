from __future__ import annotations

import argparse

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
    parser = argparse.ArgumentParser(description="Benchmark the grouped dissolve pipeline against the baseline path.")
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--groups", type=int, default=128)
    args = parser.parse_args()

    frame = _build_frame(args.rows, args.groups)
    benchmark = benchmark_dissolve_pipeline(frame, by="group", dataset="points")
    print(
        {
            "dataset": benchmark.dataset,
            "rows": benchmark.rows,
            "groups": benchmark.groups,
            "pipeline_elapsed_seconds": benchmark.pipeline_elapsed_seconds,
            "baseline_elapsed_seconds": benchmark.baseline_elapsed_seconds,
            "speedup_vs_baseline": benchmark.speedup_vs_baseline,
        }
    )


if __name__ == "__main__":
    main()
