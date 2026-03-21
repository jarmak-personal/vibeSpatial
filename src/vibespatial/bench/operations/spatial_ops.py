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
    nvtx: bool = False,
    gpu_sparkline: bool = False,
    trace: bool = False,
    **kwargs: Any,
) -> BenchmarkResult:
    from time import perf_counter

    from shapely.geometry import Point

    from vibespatial import ExecutionMode, from_shapely_geometries, has_gpu_runtime
    from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds

    points = [Point(float(i), float((i * 7) % 1000)) for i in range(scale)]
    owned = from_shapely_geometries(points)

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
        # Warmup
        compute_geometry_bounds(owned, dispatch_mode=ExecutionMode.GPU)
        for _ in range(max(1, repeat)):
            start = perf_counter()
            compute_geometry_bounds(owned, dispatch_mode=ExecutionMode.GPU)
            gpu_times.append(perf_counter() - start)

    # Determine primary timing and baseline
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
        metadata={"repeat": repeat},
    )


@benchmark_operation(
    name="spatial-query",
    description="Spatial index build + query benchmark",
    category="spatial",
    geometry_types=("polygon",),
    default_scale=100_000,
    tier=3,
    legacy_script="benchmark_spatial_query.py",
    tags=("gpu", "index"),
)
def bench_spatial_query(
    *,
    scale: int,
    repeat: int,
    compare: str | None,
    precision: str,
    nvtx: bool = False,
    gpu_sparkline: bool = False,
    trace: bool = False,
    **kwargs: Any,
) -> BenchmarkResult:
    from time import perf_counter

    from vibespatial import from_shapely_geometries
    from vibespatial.spatial.indexing import build_flat_spatial_index
    from vibespatial.testing.synthetic import SyntheticSpec, generate_polygons

    dataset = generate_polygons(
        SyntheticSpec(
            geometry_type="polygon",
            distribution="regular-grid",
            count=scale,
            seed=0,
        )
    )
    owned = from_shapely_geometries(list(dataset.geometries))

    times: list[float] = []
    # Warmup
    build_flat_spatial_index(owned)
    for _ in range(max(1, repeat)):
        start = perf_counter()
        build_flat_spatial_index(owned)
        times.append(perf_counter() - start)

    timing = timing_from_samples(times)

    return BenchmarkResult(
        operation="spatial-query",
        tier=1,
        scale=scale,
        geometry_type="polygon",
        precision=precision,
        status="pass",
        status_reason="ok",
        timing=timing,
        metadata={"repeat": repeat},
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
)
def bench_bounds_pairs(
    *,
    scale: int,
    repeat: int,
    compare: str | None,
    precision: str,
    **kwargs: Any,
) -> BenchmarkResult:
    from vibespatial import benchmark_bounds_pairs, from_shapely_geometries
    from vibespatial.testing.synthetic import SyntheticSpec, generate_points

    dataset = generate_points(
        SyntheticSpec("point", "uniform", count=scale, seed=0)
    )
    owned = from_shapely_geometries(list(dataset.geometries))
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
        metadata={
            "pairs_examined": result.pairs_examined,
            "candidate_pairs": result.candidate_pairs,
            "tile_size": result.tile_size,
        },
    )
