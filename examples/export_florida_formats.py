#!/usr/bin/env python
"""Export the Florida buildings GeoJSON into repo-relevant IO formats.

This script reads the cached Florida GeoJSON once through vibeSpatial's
native GeoJSON boundary, then writes a set of persisted formats that matter
for real-dataset IO regression work:

- GeoParquet (WKB)
- GeoParquet (GeoArrow)
- Feather
- FlatGeobuf
- GeoPackage
- ESRI Shapefile
- GeoJSONSeq
- CSV with WKT geometry

The output manifest records row count, source schema, per-format timing, and
output sizes so the generated corpus can be reused in future IO tests and
benchmarks.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pyogrio

import vibespatial as vs
from vibespatial.api._native_results import NativeTabularResult
from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events
from vibespatial.runtime.fallbacks import clear_fallback_events, get_fallback_events

SOURCE_PATH = Path("examples/data/Florida.geojson")
OUTPUT_DIR = Path("examples/data/florida_formats")
SHAPEFILE_RENAME = {"capture_dates_range": "cap_dates"}
SHAPEFILE_SIDECAR_SUFFIXES = {
    ".cpg",
    ".dbf",
    ".prj",
    ".qix",
    ".qmd",
    ".sbn",
    ".sbx",
    ".shp",
    ".shx",
}


def _list_output_paths(path: Path) -> list[Path]:
    if path.suffix.lower() == ".shp":
        return [
            candidate
            for candidate in sorted(path.parent.glob(f"{path.stem}.*"))
            if candidate.suffix.lower() in SHAPEFILE_SIDECAR_SUFFIXES
        ]
    return [path] if path.exists() else []


def _remove_output(path: Path) -> None:
    for candidate in _list_output_paths(path):
        if candidate.is_dir():
            shutil.rmtree(candidate)
        elif candidate.exists():
            candidate.unlink()


def _output_size_bytes(path: Path) -> int:
    targets = _list_output_paths(path)
    return sum(candidate.stat().st_size for candidate in targets if candidate.exists())


def _driver_supported(name: str) -> bool:
    mode = pyogrio.list_drivers().get(name)
    return mode is not None and "w" in mode


def _rename_for_shapefile(payload: NativeTabularResult) -> NativeTabularResult:
    renamed = payload.attributes.rename_columns(SHAPEFILE_RENAME)
    column_order = tuple(SHAPEFILE_RENAME.get(column, column) for column in payload.column_order)
    return replace(
        payload,
        attributes=renamed,
        column_order=column_order,
    )


def _export_csv_wkt(payload: NativeTabularResult, path: Path) -> None:
    gdf = payload.to_geodataframe()
    csv_df = pd.DataFrame(gdf.drop(columns=[payload.geometry_name]).copy())
    csv_df["geometry"] = gdf.geometry.to_wkt()
    csv_df.to_csv(path, index=False)


def _manifest_entry(*, fmt: str, path: Path, elapsed_s: float, status: str, detail: str = "") -> dict:
    return {
        "format": fmt,
        "path": str(path),
        "elapsed_seconds": round(elapsed_s, 3),
        "size_bytes": _output_size_bytes(path) if status == "ok" else None,
        "status": status,
        "detail": detail,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Directory for converted outputs (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite any existing converted outputs.",
    )
    parser.add_argument(
        "--formats",
        nargs="*",
        default=[
            "parquet-wkb",
            "parquet-geoarrow",
            "feather",
            "fgb",
            "gpkg",
            "shapefile",
            "geojsonseq",
            "csv-wkt",
        ],
        help="Subset of formats to export.",
    )
    args = parser.parse_args()

    if not SOURCE_PATH.exists():
        raise SystemExit(
            f"Missing {SOURCE_PATH}. Run examples/nearby_buildings.py first to cache the dataset."
        )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    clear_dispatch_events()
    clear_fallback_events()
    t0 = time.monotonic()
    payload = vs.read_geojson_native(SOURCE_PATH)
    read_elapsed = time.monotonic() - t0

    print(f"Read {SOURCE_PATH} into native payload in {read_elapsed:.3f}s")
    print(f"Rows: {payload.geometry.row_count:,}")
    print(f"Columns: {list(payload.attributes.columns)} + [{payload.geometry_name}]")

    targets: list[tuple[str, Path, callable]] = [
        (
            "parquet-wkb",
            output_dir / "Florida.wkb.parquet",
            lambda p: p.to_parquet(output_dir / "Florida.wkb.parquet", geometry_encoding="WKB"),
        ),
        (
            "parquet-geoarrow",
            output_dir / "Florida.geoarrow.parquet",
            lambda p: p.to_parquet(output_dir / "Florida.geoarrow.parquet", geometry_encoding="geoarrow"),
        ),
        (
            "feather",
            output_dir / "Florida.feather",
            lambda p: p.to_feather(output_dir / "Florida.feather"),
        ),
        (
            "fgb",
            output_dir / "Florida.fgb",
            lambda p: vs.write_vector_file(p, output_dir / "Florida.fgb", driver="FlatGeobuf"),
        ),
        (
            "gpkg",
            output_dir / "Florida.gpkg",
            lambda p: vs.write_vector_file(p, output_dir / "Florida.gpkg", driver="GPKG"),
        ),
        (
            "shapefile",
            output_dir / "Florida.shp",
            lambda p: vs.write_vector_file(
                _rename_for_shapefile(p),
                output_dir / "Florida.shp",
                driver="ESRI Shapefile",
            ),
        ),
        (
            "geojsonseq",
            output_dir / "Florida.geojsonseq",
            lambda p: vs.write_vector_file(p, output_dir / "Florida.geojsonseq", driver="GeoJSONSeq"),
        ),
        (
            "csv-wkt",
            output_dir / "Florida.wkt.csv",
            lambda p: _export_csv_wkt(p, output_dir / "Florida.wkt.csv"),
        ),
    ]

    results: list[dict] = []
    selected = set(args.formats)

    for fmt, path, writer in targets:
        if fmt not in selected:
            continue

        if fmt == "fgb" and not _driver_supported("FlatGeobuf"):
            results.append(_manifest_entry(fmt=fmt, path=path, elapsed_s=0.0, status="skipped", detail="FlatGeobuf driver unavailable"))
            continue
        if fmt == "gpkg" and not _driver_supported("GPKG"):
            results.append(_manifest_entry(fmt=fmt, path=path, elapsed_s=0.0, status="skipped", detail="GPKG driver unavailable"))
            continue
        if fmt == "shapefile" and not _driver_supported("ESRI Shapefile"):
            results.append(_manifest_entry(fmt=fmt, path=path, elapsed_s=0.0, status="skipped", detail="ESRI Shapefile driver unavailable"))
            continue
        if fmt == "geojsonseq" and not _driver_supported("GeoJSONSeq"):
            results.append(_manifest_entry(fmt=fmt, path=path, elapsed_s=0.0, status="skipped", detail="GeoJSONSeq driver unavailable"))
            continue

        if path.exists() or (path.suffix.lower() == ".shp" and _list_output_paths(path)):
            if not args.force:
                print(f"Skipping existing {fmt}: {path}")
                results.append(_manifest_entry(fmt=fmt, path=path, elapsed_s=0.0, status="exists"))
                continue
            _remove_output(path)

        print(f"Writing {fmt} -> {path}")
        t0 = time.monotonic()
        try:
            writer(payload)
        except Exception as exc:  # pragma: no cover - exercised on real dataset runs
            elapsed = time.monotonic() - t0
            print(f"  FAILED in {elapsed:.3f}s: {type(exc).__name__}: {exc}")
            results.append(
                _manifest_entry(
                    fmt=fmt,
                    path=path,
                    elapsed_s=elapsed,
                    status="failed",
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        elapsed = time.monotonic() - t0
        size_gb = _output_size_bytes(path) / 1e9
        print(f"  ok in {elapsed:.3f}s, size={size_gb:.3f} GB")
        results.append(_manifest_entry(fmt=fmt, path=path, elapsed_s=elapsed, status="ok"))

    manifest = {
        "source": str(SOURCE_PATH),
        "rows": int(payload.geometry.row_count),
        "attribute_columns": list(payload.attributes.columns),
        "geometry_name": payload.geometry_name,
        "source_read_elapsed_seconds": round(read_elapsed, 3),
        "source_dispatch_events": [event.to_dict() for event in get_dispatch_events()],
        "source_fallback_events": [event.to_dict() for event in get_fallback_events()],
        "outputs": results,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
