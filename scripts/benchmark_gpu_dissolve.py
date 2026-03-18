from __future__ import annotations

import argparse
import json
from time import perf_counter

import numpy as np
from shapely.geometry import box

import geopandas
from vibespatial.dissolve_pipeline import execute_grouped_box_union_gpu


def _build_frame(rows: int, groups: int) -> geopandas.GeoDataFrame:
    rows_per_group = rows // groups
    geometries = []
    group_values = []
    values = []
    for group in range(groups):
        base_y = float(group * 10.0)
        for index in range(rows_per_group):
            base_x = float(index)
            geometries.append(box(base_x, base_y, base_x + 1.0, base_y + 1.0))
            group_values.append(group)
            values.append(index)
    return geopandas.GeoDataFrame(
        {"group": group_values, "value": values, "geometry": geometries},
        crs="EPSG:3857",
    )


def _group_positions(frame: geopandas.GeoDataFrame) -> list[np.ndarray]:
    grouped = frame.groupby("group", sort=True, observed=False, dropna=True)[frame.geometry.name]
    return [np.asarray(positions, dtype=np.int32) for _, positions in grouped.indices.items()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark grouped GPU dissolve for rectangle coverages.")
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--groups", type=int, default=100)
    args = parser.parse_args()

    frame = _build_frame(args.rows, args.groups)
    values = np.asarray(frame.geometry.array, dtype=object)
    group_positions = _group_positions(frame)

    started = perf_counter()
    execute_grouped_box_union_gpu(
        values,
        group_positions,
    )
    accelerated_cold_elapsed = perf_counter() - started

    started = perf_counter()
    accelerated_warm = execute_grouped_box_union_gpu(
        values,
        group_positions,
    )
    accelerated_warm_elapsed = perf_counter() - started

    started = perf_counter()
    baseline = [
        geopandas.GeoSeries(values[positions]).union_all(method="unary")
        for positions in group_positions
    ]
    baseline_elapsed = perf_counter() - started

    print(
        json.dumps(
            {
                "rows": args.rows,
                "groups": args.groups,
                "accelerated_cold_elapsed_seconds": accelerated_cold_elapsed,
                "accelerated_warm_elapsed_seconds": accelerated_warm_elapsed,
                "baseline_elapsed_seconds": baseline_elapsed,
                "speedup_vs_baseline": (
                    baseline_elapsed / accelerated_warm_elapsed if accelerated_warm_elapsed else None
                ),
                "result_rows": int(len(accelerated_warm.geometries)),
                "baseline_rows": int(len(baseline)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
