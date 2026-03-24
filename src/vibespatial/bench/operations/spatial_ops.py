"""Spatial operation benchmarks: bounds, spatial-query, bounds-pairs."""
from __future__ import annotations

from typing import Any

from vibespatial.bench.catalog import benchmark_operation
from vibespatial.bench.schema import (
    BenchmarkResult,
    timing_from_samples,
)


@benchmark_operation(
    name="bounds",
    description="Geometry bounds computation (GPU vs CPU vs Shapely)",
    category="spatial",
    geometry_types=("point",),
    default_scale=100_000,
    tier=5,
    legacy_script="benchmark_gpu_bounds.py",
    tags=("gpu", "cpu", "compare"),
)
def bench_bounds(
    *,
    scale: int,
    repeat: int,
    compare: str | None,
    precision: str,
    input_format: str = "parquet",
    **kwargs: Any,
) -> BenchmarkResult:
    from time import perf_counter

    from vibespatial import ExecutionMode, has_gpu_runtime
    from vibespatial.bench.fixture_loader import load_owned
    from vibespatial.bench.fixtures import InputFormat, resolve_fixture_spec
    from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds

    spec = resolve_fixture_spec("point", "grid", scale)
    owned, read_seconds = load_owned(spec, InputFormat(input_format))

    # CPU timing
    cpu_times: list[float] = []
    for _ in range(max(1, repeat)):
        start = perf_counter()
        compute_geometry_bounds(owned, dispatch_mode=ExecutionMode.CPU)
        cpu_times.append(perf_counter() - start)

    # GPU timing
    gpu_times: list[float] = []
    speedup = None
    baseline_timing = None

    if has_gpu_runtime():
        compute_geometry_bounds(owned, dispatch_mode=ExecutionMode.GPU)
        for _ in range(max(1, repeat)):
            start = perf_counter()
            compute_geometry_bounds(owned, dispatch_mode=ExecutionMode.GPU)
            gpu_times.append(perf_counter() - start)

    if gpu_times:
        timing = timing_from_samples(gpu_times)
        baseline_timing = timing_from_samples(cpu_times)
        if timing.median_seconds > 0:
            speedup = baseline_timing.median_seconds / timing.median_seconds
    else:
        timing = timing_from_samples(cpu_times)

    return BenchmarkResult(
        operation="bounds",
        tier=1,
        scale=scale,
        geometry_type="point",
        precision=precision,
        status="pass",
        status_reason="ok",
        timing=timing,
        baseline_name="cpu" if gpu_times else None,
        baseline_timing=baseline_timing,
        speedup=speedup,
        input_format=input_format,
        read_seconds=read_seconds,
        metadata={"repeat": repeat},
    )


@benchmark_operation(
    name="spatial-query",
    description="Spatial index build + query benchmark (vs Shapely STRtree)",
    category="spatial",
    geometry_types=("polygon",),
    default_scale=100_000,
    tier=3,
    legacy_script="benchmark_spatial_query.py",
    tags=("gpu", "index", "compare"),
)
def bench_spatial_query(
    *,
    scale: int,
    repeat: int,
    compare: str | None,
    precision: str,
    input_format: str = "parquet",
    **kwargs: Any,
) -> BenchmarkResult:
    from time import perf_counter

    import numpy as np
    import shapely

    from vibespatial.bench.fixture_loader import load_owned
    from vibespatial.bench.fixtures import InputFormat, resolve_fixture_spec
    from vibespatial.spatial.indexing import build_flat_spatial_index
    from vibespatial.spatial.query import query_spatial_index

    spec = resolve_fixture_spec("polygon", "regular-grid", scale)
    owned, read_seconds = load_owned(spec, InputFormat(input_format))

    # Build a small set of query geometries that overlap the tree.
    # Use 1% of the tree as query inputs (shifted by half a cell).
    query_count = max(10, scale // 100)
    tree_bounds = np.asarray(shapely.bounds(owned.to_shapely()[:query_count]), dtype=np.float64)
    half_w = float((tree_bounds[0, 2] - tree_bounds[0, 0]) * 0.5)
    half_h = float((tree_bounds[0, 3] - tree_bounds[0, 1]) * 0.5)
    query_geoms = np.asarray(
        [shapely.box(b[0] + half_w, b[1] + half_h, b[2] + half_w, b[3] + half_h) for b in tree_bounds],
        dtype=object,
    )

    # --- Shapely STRtree baseline (construction + query) ---
    shapely_geoms = np.asarray(owned.to_shapely(), dtype=object)
    # Warmup
    _tree = shapely.STRtree(shapely_geoms)
    _tree.query(query_geoms, predicate="intersects")

    shapely_times: list[float] = []
    for _ in range(max(1, repeat)):
        start = perf_counter()
        tree = shapely.STRtree(shapely_geoms)
        tree.query(query_geoms, predicate="intersects")
        shapely_times.append(perf_counter() - start)

    # --- vibeSpatial (construction + query) ---
    # Warmup
    flat_index = build_flat_spatial_index(owned)
    query_spatial_index(owned, flat_index, query_geoms, predicate="intersects")

    gpu_times: list[float] = []
    for _ in range(max(1, repeat)):
        start = perf_counter()
        flat_index = build_flat_spatial_index(owned)
        query_spatial_index(owned, flat_index, query_geoms, predicate="intersects")
        gpu_times.append(perf_counter() - start)

    timing = timing_from_samples(gpu_times)
    baseline_timing = timing_from_samples(shapely_times)
    speedup = (
        baseline_timing.median_seconds / timing.median_seconds
        if timing.median_seconds > 0
        else None
    )

    return BenchmarkResult(
        operation="spatial-query",
        tier=1,
        scale=scale,
        geometry_type="polygon",
        precision=precision,
        status="pass",
        status_reason="ok",
        timing=timing,
        baseline_name="shapely_strtree",
        baseline_timing=baseline_timing,
        speedup=speedup,
        input_format=input_format,
        read_seconds=read_seconds,
        metadata={
            "repeat": repeat,
            "query_count": query_count,
        },
    )


@benchmark_operation(
    name="bounds-pairs",
    description="Tiled bounds-pair generation benchmark",
    category="spatial",
    geometry_types=("point",),
    default_scale=10_000,
    tier=5,
    legacy_script="benchmark_bounds_pairs.py",
    tags=("gpu",),
    max_scale=100_000,
)
def bench_bounds_pairs(
    *,
    scale: int,
    repeat: int,
    compare: str | None,
    precision: str,
    input_format: str = "parquet",
    **kwargs: Any,
) -> BenchmarkResult:
    from vibespatial import benchmark_bounds_pairs
    from vibespatial.bench.fixture_loader import load_owned
    from vibespatial.bench.fixtures import InputFormat, resolve_fixture_spec

    spec = resolve_fixture_spec("point", "uniform", scale)
    owned, read_seconds = load_owned(spec, InputFormat(input_format))
    result = benchmark_bounds_pairs(owned, dataset="uniform", tile_size=256)

    timing = timing_from_samples([result.elapsed_seconds])

    return BenchmarkResult(
        operation="bounds-pairs",
        tier=1,
        scale=scale,
        geometry_type="point",
        precision=precision,
        status="pass",
        status_reason="ok",
        timing=timing,
        input_format=input_format,
        read_seconds=read_seconds,
        metadata={
            "pairs_examined": result.pairs_examined,
            "candidate_pairs": result.candidate_pairs,
            "tile_size": result.tile_size,
        },
    )
