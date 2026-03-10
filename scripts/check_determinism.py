from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from statistics import median
from time import perf_counter

import geopandas
import numpy as np
import pandas as pd
import shapely

from vibespatial import DeterminismMode, KernelClass, evaluate_geopandas_dissolve, select_determinism_plan
from vibespatial.testing.synthetic import SyntheticSpec, generate_polygons


@dataclass(frozen=True)
class DeterminismProbeResult:
    rows: int
    groups: int
    repeats: int
    unique_fingerprints: int
    fingerprints: tuple[str, ...]
    deterministic_median_seconds: float
    default_median_seconds: float
    overhead_factor: float
    plan: dict

    @property
    def bitwise_identical(self) -> bool:
        return self.unique_fingerprints == 1


def build_dissolve_frame(rows: int, groups: int) -> geopandas.GeoDataFrame:
    dataset = generate_polygons(
        SyntheticSpec(
            "polygon",
            "regular-grid",
            count=rows,
            seed=7,
            vertices=5,
            hole_probability=0.0,
        )
    )
    values = np.asarray(list(dataset.geometries), dtype=object)
    return geopandas.GeoDataFrame(
        {
            "group": pd.Categorical(np.arange(rows, dtype=np.int32) % max(groups, 1)),
            "weight": np.arange(rows, dtype=np.int64),
            "geometry": values,
        },
        geometry="geometry",
        crs="EPSG:4326",
    )


def _fingerprint_dissolve_output(frame: geopandas.GeoDataFrame) -> str:
    geometries = np.asarray(frame.geometry.array, dtype=object)
    areas = np.asarray(shapely.area(geometries), dtype=np.float64)
    weights = np.asarray(frame["weight"], dtype=np.int64)
    digest = hashlib.sha256()
    for geometry, area, weight in zip(geometries.tolist(), areas.tolist(), weights.tolist(), strict=True):
        digest.update(b"<null>" if geometry is None else geometry.wkb)
        digest.update(np.float64(area).tobytes())
        digest.update(np.int64(weight).tobytes())
    return digest.hexdigest()


def _run_dissolve_area_pipeline(frame: geopandas.GeoDataFrame) -> tuple[str, float]:
    started = perf_counter()
    dissolved = evaluate_geopandas_dissolve(
        frame,
        by="group",
        aggfunc="sum",
        as_index=True,
        level=None,
        sort=True,
        observed=False,
        dropna=True,
        method="unary",
        grid_size=None,
        agg_kwargs={},
    )
    _ = np.asarray(shapely.area(np.asarray(dissolved.geometry.array, dtype=object)), dtype=np.float64)
    elapsed = perf_counter() - started
    return _fingerprint_dissolve_output(dissolved), elapsed


def probe_determinism(*, rows: int, groups: int, repeats: int) -> DeterminismProbeResult:
    frame = build_dissolve_frame(rows, groups)
    deterministic_times: list[float] = []
    fingerprints: list[str] = []
    for _ in range(repeats):
        fingerprint, elapsed = _run_dissolve_area_pipeline(frame)
        fingerprints.append(fingerprint)
        deterministic_times.append(elapsed)

    default_times: list[float] = []
    for _ in range(min(repeats, 10)):
        _, elapsed = _run_dissolve_area_pipeline(frame)
        default_times.append(elapsed)

    plan = select_determinism_plan(kernel_class=KernelClass.CONSTRUCTIVE, requested=DeterminismMode.DETERMINISTIC)
    deterministic_median = float(median(deterministic_times))
    default_median = float(median(default_times))
    overhead = deterministic_median / default_median if default_median > 0.0 else float("inf")
    return DeterminismProbeResult(
        rows=rows,
        groups=groups,
        repeats=repeats,
        unique_fingerprints=len(set(fingerprints)),
        fingerprints=tuple(fingerprints[:5]),
        deterministic_median_seconds=deterministic_median,
        default_median_seconds=default_median,
        overhead_factor=overhead,
        plan=asdict(plan),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check deterministic dissolve + area reproducibility.")
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--groups", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args(argv)

    result = probe_determinism(rows=args.rows, groups=args.groups, repeats=args.repeats)
    payload = {
        "rows": result.rows,
        "groups": result.groups,
        "repeats": result.repeats,
        "bitwise_identical": result.bitwise_identical,
        "unique_fingerprints": result.unique_fingerprints,
        "sample_fingerprints": list(result.fingerprints),
        "deterministic_median_seconds": result.deterministic_median_seconds,
        "default_median_seconds": result.default_median_seconds,
        "overhead_factor": result.overhead_factor,
        "plan": result.plan,
    }
    print(json.dumps(payload, indent=2))
    return 0 if result.bitwise_identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
