"""Spatial operation benchmarks: bounds, spatial-query, bounds-pairs."""
from __future__ import annotations

from typing import Any

from vibespatial.bench.catalog import OperationParameterSpec, benchmark_operation
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

    import shapely

    from vibespatial.bench.fixture_loader import load_public_geodataframe
    from vibespatial.bench.fixtures import InputFormat, resolve_fixture_spec

    spec = resolve_fixture_spec("point", "grid", scale)
    frame, read_seconds = load_public_geodataframe(spec, InputFormat(input_format))
    geometry = frame.geometry

    _ = geometry.bounds
    public_times: list[float] = []
    for _ in range(max(1, repeat)):
        start = perf_counter()
        _ = geometry.bounds
        public_times.append(perf_counter() - start)

    speedup = None
    baseline_timing = None
    baseline_name = None

    timing = timing_from_samples(public_times)
    if compare == "shapely" or compare is None:
        values = geometry.to_numpy()
        baseline_times: list[float] = []
        for _ in range(max(1, repeat)):
            start = perf_counter()
            shapely.bounds(values)
            baseline_times.append(perf_counter() - start)
        baseline_timing = timing_from_samples(baseline_times)
        baseline_name = "shapely"
        if timing.median_seconds > 0:
            speedup = baseline_timing.median_seconds / timing.median_seconds

    return BenchmarkResult(
        operation="bounds",
        tier=1,
        scale=scale,
        geometry_type="point",
        precision=precision,
        status="pass",
        status_reason="ok",
        timing=timing,
        baseline_name=baseline_name,
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
    parameters=(
        OperationParameterSpec(
            name="overlap_ratio",
            value_type="float",
            description="Fraction of query geometries that overlap the tree",
            default=0.2,
        ),
    ),
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
    from shapely.affinity import translate

    from vibespatial.bench.fixture_loader import load_public_geodataframe
    from vibespatial.bench.fixtures import InputFormat, resolve_fixture_spec

    fmt = InputFormat(input_format)
    spec = resolve_fixture_spec("polygon", "regular-grid", scale)
    gdf, read_seconds = load_public_geodataframe(spec, fmt)

    overlap_ratio = float(kwargs.get("overlap_ratio", 0.2))
    if not 0.0 <= overlap_ratio <= 1.0:
        raise ValueError(f"overlap_ratio must be within [0, 1], got {overlap_ratio}")

    tree_geoms = np.asarray(gdf.geometry.to_numpy(), dtype=object)
    query_geoms = tree_geoms.copy()
    cutoff = int(scale * overlap_ratio)
    if cutoff < scale:
        query_geoms[cutoff:] = np.asarray(
            [translate(geometry, xoff=10_000.0, yoff=10_000.0) for geometry in query_geoms[cutoff:]],
            dtype=object,
        )

    # --- Shapely STRtree baseline (construction + query) ---
    shapely_geoms = tree_geoms
    # Warmup
    _tree = shapely.STRtree(shapely_geoms)
    _tree.query(query_geoms, predicate="intersects")

    shapely_times: list[float] = []
    for _ in range(max(1, repeat)):
        start = perf_counter()
        tree = shapely.STRtree(shapely_geoms)
        tree.query(query_geoms, predicate="intersects")
        shapely_times.append(perf_counter() - start)

    # --- vibeSpatial public API (construction + query) ---
    gdf.sindex.query(query_geoms, predicate="intersects")

    gpu_times: list[float] = []
    for _ in range(max(1, repeat)):
        start = perf_counter()
        gdf.sindex.query(query_geoms, predicate="intersects")
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
            "query_count": len(query_geoms),
            "overlap_ratio": overlap_ratio,
        },
    )


@benchmark_operation(
    name="bounds-pairs",
    description="Tiled bounds-pair generation benchmark",
    category="spatial",
    geometry_types=("point",),
    default_scale=10_000,
    tier=5,
    parameters=(
        OperationParameterSpec(
            name="dataset",
            value_type="choice",
            description="Synthetic point distribution to benchmark",
            default="uniform",
            choices=("uniform", "skewed", "both"),
        ),
        OperationParameterSpec(
            name="tile_size",
            value_type="int",
            description="Tile size for tiled bounds-pair candidate generation",
            default=256,
        ),
    ),
    tags=("gpu",),
    max_scale=100_000,
    public_api=False,
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
    from time import perf_counter

    from vibespatial import benchmark_bounds_pairs
    from vibespatial.bench.fixture_loader import load_geodataframe, load_owned
    from vibespatial.bench.fixtures import InputFormat, resolve_fixture_spec

    def _run_single_dataset(name: str) -> tuple[dict[str, Any], float, float, int]:
        distribution = "uniform" if name == "uniform" else "clustered"
        spec = resolve_fixture_spec("point", distribution, scale)
        owned, read_s = load_owned(spec, fmt)
        result = benchmark_bounds_pairs(owned, dataset=name, tile_size=tile_size)

        baseline_elapsed = 0.0
        if compare == "shapely" or compare is None:
            import shapely

            gdf, _ = load_geodataframe(spec, fmt)
            geom_arr = gdf.geometry.to_numpy()
            start = perf_counter()
            tree = shapely.STRtree(geom_arr)
            tree.query(geom_arr, predicate="intersects")
            baseline_elapsed = perf_counter() - start

        return (
            {
                "dataset": result.dataset,
                "pairs_examined": result.pairs_examined,
                "candidate_pairs": result.candidate_pairs,
                "tile_size": result.tile_size,
                "elapsed_seconds": result.elapsed_seconds,
                "baseline_elapsed_seconds": baseline_elapsed,
            },
            result.elapsed_seconds,
            baseline_elapsed,
            read_s,
        )

    fmt = InputFormat(input_format)
    dataset = kwargs.get("dataset", "uniform")
    tile_size = kwargs.get("tile_size", 256)
    datasets = ("uniform", "skewed") if dataset == "both" else (dataset,)

    dataset_results: list[dict[str, Any]] = []
    elapsed_total = 0.0
    baseline_total = 0.0
    read_seconds = 0.0
    for dataset_name in datasets:
        metadata, elapsed_seconds, baseline_elapsed_seconds, read_s = _run_single_dataset(dataset_name)
        dataset_results.append(metadata)
        elapsed_total += elapsed_seconds
        baseline_total += baseline_elapsed_seconds
        read_seconds += read_s

    timing = timing_from_samples([elapsed_total])

    baseline_timing = None
    speedup = None
    baseline_name = None
    if compare == "shapely" or compare is None:
        baseline_timing = timing_from_samples([baseline_total])
        baseline_name = "shapely-strtree"
        if timing.median_seconds > 0:
            speedup = baseline_timing.median_seconds / timing.median_seconds

    return BenchmarkResult(
        operation="bounds-pairs",
        tier=1,
        scale=scale,
        geometry_type="point",
        precision=precision,
        status="pass",
        status_reason="ok",
        timing=timing,
        baseline_name=baseline_name,
        baseline_timing=baseline_timing,
        speedup=speedup,
        input_format=input_format,
        read_seconds=read_seconds,
        metadata={
            "dataset": dataset,
            "datasets": dataset_results,
            "tile_size": tile_size,
            "pairs_examined": sum(item["pairs_examined"] for item in dataset_results),
            "candidate_pairs": sum(item["candidate_pairs"] for item in dataset_results),
        },
    )
