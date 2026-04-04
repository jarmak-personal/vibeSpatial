from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import time
from pathlib import Path

import pandas as pd
from shapely.geometry import Polygon

import vibespatial as geopandas
from vibespatial.api.tools import overlay as overlay_module
from vibespatial.runtime.hotpath_trace import reset_hotpath_trace, summarize_hotpath_trace

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_OVERLAY_DATA = ROOT / "tests" / "upstream" / "geopandas" / "tests" / "data" / "overlay"


def _load_nybb_inputs() -> tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame]:
    left = geopandas.read_file(
        str(ROOT / "tests" / "upstream" / "geopandas" / "tests" / "data" / "nybb_16a.zip")
    )
    right = geopandas.read_file(str(UPSTREAM_OVERLAY_DATA / "nybb_qgis" / "polydf2.shp"))
    return left, right


def _load_geometry_name_inputs(
    *, other_geometry: bool,
) -> tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame]:
    s1 = geopandas.GeoSeries(
        [
            Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
        ]
    )
    s2 = geopandas.GeoSeries(
        [
            Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
            Polygon([(3, 3), (5, 3), (5, 5), (3, 5)]),
        ]
    )
    df1 = geopandas.GeoDataFrame({"col1": [1, 2], "geometry": s1})
    df2 = geopandas.GeoDataFrame({"col2": [1, 2], "geometry": s2})
    df3 = df1.rename(columns={"geometry": "polygons"}).set_geometry("polygons")
    if other_geometry:
        df3["geometry"] = df1.centroid.geometry
    return df3, df2


def _profile(callable_) -> tuple[object, float, str]:
    os.environ["VIBESPATIAL_HOTPATH_TRACE"] = "1"
    profiler = cProfile.Profile()
    reset_hotpath_trace()
    started = time.perf_counter()
    result = profiler.runcall(callable_)
    elapsed = time.perf_counter() - started

    stats = pstats.Stats(profiler)
    hotspots: list[tuple[float, int, int, str, str, int]] = []
    interesting = (
        str(ROOT / "src" / "vibespatial" / "api" / "tools" / "overlay.py"),
        str(ROOT / "src" / "vibespatial" / "constructive" / "binary_constructive.py"),
        str(ROOT / "src" / "vibespatial" / "overlay"),
        str(ROOT / "src" / "vibespatial" / "spatial"),
        "site-packages/pandas",
    )
    for (filename, line_no, func_name), (cc, nc, _tt, ct, _callers) in stats.stats.items():
        if any(token in filename for token in interesting):
            hotspots.append((ct, nc, cc, filename, func_name, line_no))
    hotspots.sort(reverse=True)

    lines = ["cumulative_s  total_calls  prim_calls  location"]
    for ct, nc, cc, filename, func_name, line_no in hotspots[:30]:
        rel = filename.replace(f"{ROOT}/", "")
        lines.append(f"{ct:12.3f}  {nc:11d}  {cc:10d}  {rel}:{line_no}({func_name})")
    trace_lines = ["hotpath_trace_s  calls  stage"]
    for entry in summarize_hotpath_trace()[:30]:
        trace_lines.append(
            f"{float(entry['elapsed_seconds']):15.3f}  {int(entry['calls']):5d}  {entry['name']}"
        )
    return result, elapsed, "\n".join(lines + [""] + trace_lines)


def _run_nybb_public(how: str) -> tuple[object, float, str]:
    left, right = _load_nybb_inputs()
    return _profile(lambda: geopandas.overlay(left, right, how=how))


def _run_nybb_internal(how: str) -> tuple[object, float, str]:
    left, right = _load_nybb_inputs()
    fn = {
        "intersection": overlay_module._overlay_intersection,
        "difference": overlay_module._overlay_difference,
        "identity": overlay_module._overlay_identity,
        "symmetric_difference": overlay_module._overlay_symmetric_diff,
        "union": overlay_module._overlay_union,
    }[how]
    return _profile(lambda: fn(left, right))


def _run_geometry_name_public(how: str, *, other_geometry: bool) -> tuple[object, float, str]:
    left, right = _load_geometry_name_inputs(other_geometry=other_geometry)
    return _profile(lambda: geopandas.overlay(left, right, how=how))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile strict overlay hotspots.")
    parser.add_argument("--scenario", choices=("nybb", "geometry-name"), default="nybb")
    parser.add_argument(
        "--layer",
        choices=("public", "internal"),
        default="public",
        help="Internal is only supported for the nybb scenario.",
    )
    parser.add_argument(
        "--how",
        choices=("intersection", "difference", "identity", "symmetric_difference", "union"),
        default="union",
    )
    parser.add_argument("--other-geometry", action="store_true")
    args = parser.parse_args(argv)

    if args.scenario == "nybb":
        if args.layer == "public":
            result, elapsed, stats = _run_nybb_public(args.how)
        else:
            result, elapsed, stats = _run_nybb_internal(args.how)
    elif args.layer != "public":
        raise SystemExit("internal layer is only supported for --scenario nybb")
    else:
        result, elapsed, stats = _run_geometry_name_public(
            args.how,
            other_geometry=args.other_geometry,
        )

    rows = len(result[0] if isinstance(result, tuple) else result)
    print(
        pd.Series(
            {
                "scenario": args.scenario,
                "layer": args.layer,
                "how": args.how,
                "other_geometry": args.other_geometry,
                "elapsed_s": round(elapsed, 3),
                "rows": int(rows),
            }
        ).to_string()
    )
    print()
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
