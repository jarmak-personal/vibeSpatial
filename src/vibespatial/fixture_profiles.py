from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from shapely.affinity import translate

import vibespatial.api as geopandas
from vibespatial import clear_dispatch_events, get_dispatch_events
from vibespatial.benchmark_fixtures import ensure_fixture, fixture_path, get_fixture_spec


def _nvtx_range(label: str, color: str):
    try:
        import nvtx
    except Exception:
        return nullcontext()
    return nvtx.annotate(label, color=color)


def _result_pair_count(result: Any, *, output_format: str, return_distance: bool = False) -> int:
    if return_distance:
        result = result[0]
    if output_format == "indices":
        if getattr(result, "ndim", 1) == 1:
            return int(result.size)
        return int(result.shape[1])
    if output_format == "dense":
        return int(np.count_nonzero(result))
    if output_format == "sparse":
        return int(result.nnz)
    return 0


@dataclass(frozen=True)
class FixtureProfileResult:
    operation: str
    tree_fixture: str
    query_fixture: str | None
    query_mode: str
    predicate: str | None
    output_format: str
    tree_rows: int
    query_rows: int
    read_tree_elapsed_seconds: float
    read_query_elapsed_seconds: float
    build_sindex_elapsed_seconds: float
    operation_elapsed_seconds: float
    total_elapsed_seconds: float
    result_pairs: int
    dispatch_events: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "tree_fixture": self.tree_fixture,
            "query_fixture": self.query_fixture,
            "query_mode": self.query_mode,
            "predicate": self.predicate,
            "output_format": self.output_format,
            "tree_rows": self.tree_rows,
            "query_rows": self.query_rows,
            "read_tree_elapsed_seconds": self.read_tree_elapsed_seconds,
            "read_query_elapsed_seconds": self.read_query_elapsed_seconds,
            "build_sindex_elapsed_seconds": self.build_sindex_elapsed_seconds,
            "operation_elapsed_seconds": self.operation_elapsed_seconds,
            "total_elapsed_seconds": self.total_elapsed_seconds,
            "result_pairs": self.result_pairs,
            "dispatch_events": list(self.dispatch_events),
        }


def _resolve_fixture_path(name_or_path: str | Path, *, fixture_dir: str | Path | None = None) -> Path:
    path = Path(name_or_path)
    if path.suffix == ".parquet" or path.exists():
        return path
    if fixture_dir is not None:
        candidate = Path(fixture_dir) / f"{path.name}.parquet"
        if candidate.exists():
            return candidate
    return fixture_path(get_fixture_spec(str(name_or_path)), fixture_dir=fixture_dir)


def _load_query_geometry(
    tree_frame,
    *,
    query_mode: str,
    query_fixture: str | Path | None,
    fixture_dir: str | Path | None,
) -> tuple[Any, float, str | None]:
    if query_mode == "self":
        return tree_frame.geometry, 0.0, None
    if query_mode == "translated-self":
        started = perf_counter()
        values = np.asarray(
            [translate(geometry, xoff=10_000.0, yoff=10_000.0) for geometry in tree_frame.geometry.to_numpy()],
            dtype=object,
        )
        query = geopandas.GeoSeries(values, crs=tree_frame.crs)
        return query, perf_counter() - started, None
    if query_mode == "fixture":
        if query_fixture is None:
            raise ValueError("query_fixture is required when query_mode='fixture'")
        path = _resolve_fixture_path(query_fixture, fixture_dir=fixture_dir)
        with _nvtx_range("fixture.read_query", "blue"):
            started = perf_counter()
            frame = geopandas.read_parquet(path)
            elapsed = perf_counter() - started
        return frame.geometry, elapsed, str(path)
    raise ValueError(f"Unsupported query_mode: {query_mode}")


def profile_fixture_query(
    tree_fixture: str | Path,
    *,
    query_mode: str = "self",
    query_fixture: str | Path | None = None,
    predicate: str | None = "intersects",
    sort: bool = False,
    output_format: str = "indices",
    fixture_dir: str | Path | None = None,
) -> FixtureProfileResult:
    clear_dispatch_events()
    tree_path = _resolve_fixture_path(tree_fixture, fixture_dir=fixture_dir)
    started_total = perf_counter()
    with _nvtx_range("fixture.read_tree", "blue"):
        started = perf_counter()
        tree_frame = geopandas.read_parquet(tree_path)
        read_tree_elapsed = perf_counter() - started

    query_geometry, read_query_elapsed, query_fixture_path = _load_query_geometry(
        tree_frame,
        query_mode=query_mode,
        query_fixture=query_fixture,
        fixture_dir=fixture_dir,
    )

    with _nvtx_range("fixture.build_sindex", "purple"):
        started = perf_counter()
        sindex = tree_frame.sindex
        build_sindex_elapsed = perf_counter() - started

    with _nvtx_range("fixture.query", "green"):
        started = perf_counter()
        result = sindex.query(query_geometry, predicate=predicate, sort=sort, output_format=output_format)
        operation_elapsed = perf_counter() - started

    with _nvtx_range("fixture.result_shape", "orange"):
        result_pairs = _result_pair_count(result, output_format=output_format)

    events = tuple(event.to_dict() for event in get_dispatch_events(clear=True))
    return FixtureProfileResult(
        operation="query",
        tree_fixture=str(tree_path),
        query_fixture=query_fixture_path,
        query_mode=query_mode,
        predicate=predicate,
        output_format=output_format,
        tree_rows=int(len(tree_frame)),
        query_rows=int(len(query_geometry)),
        read_tree_elapsed_seconds=read_tree_elapsed,
        read_query_elapsed_seconds=read_query_elapsed,
        build_sindex_elapsed_seconds=build_sindex_elapsed,
        operation_elapsed_seconds=operation_elapsed,
        total_elapsed_seconds=perf_counter() - started_total,
        result_pairs=result_pairs,
        dispatch_events=events,
    )


def profile_fixture_nearest(
    tree_fixture: str | Path,
    *,
    query_mode: str = "self",
    query_fixture: str | Path | None = None,
    max_distance: float | None = None,
    return_all: bool = True,
    return_distance: bool = False,
    exclusive: bool = False,
    fixture_dir: str | Path | None = None,
) -> FixtureProfileResult:
    clear_dispatch_events()
    tree_path = _resolve_fixture_path(tree_fixture, fixture_dir=fixture_dir)
    started_total = perf_counter()
    with _nvtx_range("fixture.read_tree", "blue"):
        started = perf_counter()
        tree_frame = geopandas.read_parquet(tree_path)
        read_tree_elapsed = perf_counter() - started

    query_geometry, read_query_elapsed, query_fixture_path = _load_query_geometry(
        tree_frame,
        query_mode=query_mode,
        query_fixture=query_fixture,
        fixture_dir=fixture_dir,
    )

    with _nvtx_range("fixture.build_sindex", "purple"):
        started = perf_counter()
        sindex = tree_frame.sindex
        build_sindex_elapsed = perf_counter() - started

    with _nvtx_range("fixture.nearest", "green"):
        started = perf_counter()
        result = sindex.nearest(
            query_geometry,
            return_all=return_all,
            max_distance=max_distance,
            return_distance=return_distance,
            exclusive=exclusive,
        )
        operation_elapsed = perf_counter() - started

    with _nvtx_range("fixture.result_shape", "orange"):
        result_pairs = _result_pair_count(result, output_format="indices", return_distance=return_distance)

    events = tuple(event.to_dict() for event in get_dispatch_events(clear=True))
    return FixtureProfileResult(
        operation="nearest",
        tree_fixture=str(tree_path),
        query_fixture=query_fixture_path,
        query_mode=query_mode,
        predicate=None,
        output_format="indices",
        tree_rows=int(len(tree_frame)),
        query_rows=int(len(query_geometry)),
        read_tree_elapsed_seconds=read_tree_elapsed,
        read_query_elapsed_seconds=read_query_elapsed,
        build_sindex_elapsed_seconds=build_sindex_elapsed,
        operation_elapsed_seconds=operation_elapsed,
        total_elapsed_seconds=perf_counter() - started_total,
        result_pairs=result_pairs,
        dispatch_events=events,
    )


def ensure_named_fixture(name: str, *, fixture_dir: str | Path | None = None, force: bool = False) -> Path:
    return ensure_fixture(name, fixture_dir=fixture_dir, force=force)
