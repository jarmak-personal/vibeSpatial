"""Format-aware fixture loading for vsbench benchmarks.

Loads cached fixtures from disk using GPU-native readers (vibespatial path)
or CPU readers (GeoPandas/pyogrio baseline path). Read time is tracked
separately so benchmarks can report it in metadata.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import vibespatial.api as geopandas
from vibespatial.bench.fixtures import (
    BenchmarkFixtureSpec,
    InputFormat,
    ensure_fixture_format,
    fixture_path_for_format,
)


def _preferred_geoparquet_backend() -> str:
    from vibespatial.runtime import has_gpu_runtime

    return "gpu" if has_gpu_runtime() else "cpu"


def load_owned(
    spec: BenchmarkFixtureSpec,
    fmt: InputFormat | str,
    *,
    fixture_dir: Path | None = None,
) -> tuple[Any, float]:
    """Load a fixture as ``OwnedGeometryArray``, returning ``(array, read_seconds)``.

    Uses GPU-native readers where available. Falls back to
    ``from_shapely_geometries`` for formats without a GPU reader.
    """
    fmt = InputFormat(fmt) if not isinstance(fmt, InputFormat) else fmt
    path = fixture_path_for_format(spec, fmt, fixture_dir=fixture_dir)
    if not path.exists():
        ensure_fixture_format(spec, fmt, fixture_dir=fixture_dir)

    start = perf_counter()
    match fmt:
        case InputFormat.PARQUET:
            from vibespatial.io.geoparquet import read_geoparquet_owned

            owned = read_geoparquet_owned(path, backend=_preferred_geoparquet_backend())
        case InputFormat.GEOJSON:
            from vibespatial.io.geojson import read_geojson_owned

            batch = read_geojson_owned(str(path), track_properties=False)
            owned = batch.geometry
        case InputFormat.SHAPEFILE:
            from vibespatial.io.file import read_shapefile_owned

            batch = read_shapefile_owned(str(path))
            owned = batch.geometry
        case InputFormat.GPKG:
            from vibespatial.geometry.owned import from_shapely_geometries

            frame = geopandas.read_file(str(path), driver="GPKG")
            owned = from_shapely_geometries(frame.geometry.tolist())
    read_seconds = perf_counter() - start
    return owned, read_seconds


def load_geodataframe(
    spec: BenchmarkFixtureSpec,
    fmt: InputFormat | str,
    *,
    fixture_dir: Path | None = None,
) -> tuple[geopandas.GeoDataFrame, float]:
    """Load a fixture as ``GeoDataFrame`` for baseline comparisons.

    Returns ``(geodataframe, read_seconds)``. Uses CPU-only readers
    (PyArrow for parquet, pyogrio for everything else).
    """
    fmt = InputFormat(fmt) if not isinstance(fmt, InputFormat) else fmt
    path = fixture_path_for_format(spec, fmt, fixture_dir=fixture_dir)
    if not path.exists():
        ensure_fixture_format(spec, fmt, fixture_dir=fixture_dir)

    start = perf_counter()
    match fmt:
        case InputFormat.PARQUET:
            import pyarrow.parquet as pq

            table = pq.read_table(str(path))
            frame = geopandas.GeoDataFrame.from_arrow(table)
        case _:
            frame = geopandas.read_file(str(path))
    read_seconds = perf_counter() - start
    return frame, read_seconds


def load_public_geodataframe(
    spec: BenchmarkFixtureSpec,
    fmt: InputFormat | str,
    *,
    fixture_dir: Path | None = None,
) -> tuple[geopandas.GeoDataFrame, float]:
    """Load a fixture through the public GeoPandas-compatible API."""
    fmt = InputFormat(fmt) if not isinstance(fmt, InputFormat) else fmt
    path = fixture_path_for_format(spec, fmt, fixture_dir=fixture_dir)
    if not path.exists():
        ensure_fixture_format(spec, fmt, fixture_dir=fixture_dir)

    start = perf_counter()
    match fmt:
        case InputFormat.PARQUET:
            frame = geopandas.read_parquet(path)
        case _:
            frame = geopandas.read_file(str(path))
    read_seconds = perf_counter() - start
    return frame, read_seconds
