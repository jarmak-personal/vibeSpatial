"""Predicate operation benchmarks: gpu-pip, binary-predicates, gpu-predicates, point-predicates."""
from __future__ import annotations

from typing import Any

from vibespatial.bench.catalog import benchmark_operation
from vibespatial.bench.schema import (
    BenchmarkResult,
    timing_from_samples,
)


@benchmark_operation(
    name="gpu-pip",
    description="Point-in-polygon GPU kernel (cold + warm vs CPU vs Shapely)",
    category="predicate",
    geometry_types=("point", "polygon"),
    default_scale=100_000,
    tier=4,
    legacy_script="benchmark_gpu_pip.py",
    tags=("gpu", "compare"),
)
def bench_gpu_pip(
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

    import numpy as np

    from vibespatial import ExecutionMode, from_shapely_geometries, has_gpu_runtime
    from vibespatial.kernels.predicates.point_in_polygon import point_in_polygon
    from vibespatial.testing.synthetic import SyntheticSpec, generate_points, generate_polygons

    if not has_gpu_runtime():
        return BenchmarkResult(
            operation="gpu-pip",
            tier=1,
            scale=scale,
            geometry_type="point+polygon",
            precision=precision,
            status="skip",
            status_reason="GPU runtime not available",
            timing=timing_from_samples([]),
        )

    polygon_base_count = max(scale // 8, 1)
    points = np.asarray(
        list(generate_points(SyntheticSpec("point", "grid", count=scale, seed=0)).geometries),
        dtype=object,
    )
    polygons = np.resize(
        np.asarray(
            list(generate_polygons(
                SyntheticSpec("polygon", "regular-grid", count=polygon_base_count, seed=1, vertices=5)
            ).geometries),
            dtype=object,
        ),
        scale,
    ).astype(object, copy=False)

    points_owned = from_shapely_geometries(points.tolist())
    polygons_owned = from_shapely_geometries(polygons.tolist())

    # Baseline: Shapely
    baseline_timing = None
    speedup = None
    baseline_name = None
    if compare == "shapely" or compare is None:
        import shapely

        shapely_times: list[float] = []
        for _ in range(max(1, repeat)):
            start = perf_counter()
            shapely.covers(polygons, points)
            shapely_times.append(perf_counter() - start)
        baseline_timing = timing_from_samples(shapely_times)
        baseline_name = "shapely"

    # Warmup (cold run)
    point_in_polygon(points_owned, polygons_owned, dispatch_mode=ExecutionMode.GPU)

    # Timed GPU runs
    gpu_times: list[float] = []
    for _ in range(max(1, repeat)):
        start = perf_counter()
        point_in_polygon(points_owned, polygons_owned, dispatch_mode=ExecutionMode.GPU)
        gpu_times.append(perf_counter() - start)

    timing = timing_from_samples(gpu_times)
    if baseline_timing and timing.median_seconds > 0:
        speedup = baseline_timing.median_seconds / timing.median_seconds

    return BenchmarkResult(
        operation="gpu-pip",
        tier=1,
        scale=scale,
        geometry_type="point+polygon",
        precision=precision,
        status="pass",
        status_reason="ok",
        timing=timing,
        baseline_name=baseline_name,
        baseline_timing=baseline_timing,
        speedup=speedup,
        metadata={"repeat": repeat, "polygon_base_count": polygon_base_count},
    )


@benchmark_operation(
    name="binary-predicates",
    description="Binary predicate dispatch (intersects, contains, etc.)",
    category="predicate",
    geometry_types=("polygon",),
    default_scale=10_000,
    tier=3,
    legacy_script="benchmark_binary_predicates.py",
    tags=("gpu", "cpu", "compare"),
)
def bench_binary_predicates(
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

    import numpy as np
    from shapely.affinity import translate

    from vibespatial import from_shapely_geometries
    from vibespatial.predicates.binary import evaluate_binary_predicate
    from vibespatial.testing.synthetic import SyntheticSpec, generate_polygons

    dataset = generate_polygons(
        SyntheticSpec("polygon", "regular-grid", count=scale, seed=0)
    )
    left_geoms = list(dataset.geometries)
    right_geoms = [translate(g, xoff=0.3, yoff=0.3) for g in left_geoms]

    left_owned = from_shapely_geometries(left_geoms)
    right_owned = from_shapely_geometries(right_geoms)

    predicate = "intersects"

    # Warmup
    evaluate_binary_predicate(left_owned, right_owned, predicate)

    times: list[float] = []
    for _ in range(max(1, repeat)):
        start = perf_counter()
        evaluate_binary_predicate(left_owned, right_owned, predicate)
        times.append(perf_counter() - start)

    timing = timing_from_samples(times)

    # Shapely baseline
    baseline_timing = None
    speedup = None
    baseline_name = None
    if compare == "shapely" or compare is None:
        import shapely

        left_arr = np.asarray(left_geoms, dtype=object)
        right_arr = np.asarray(right_geoms, dtype=object)

        shapely_times: list[float] = []
        for _ in range(max(1, repeat)):
            start = perf_counter()
            shapely.intersects(left_arr, right_arr)
            shapely_times.append(perf_counter() - start)
        baseline_timing = timing_from_samples(shapely_times)
        baseline_name = "shapely"
        if timing.median_seconds > 0:
            speedup = baseline_timing.median_seconds / timing.median_seconds

    return BenchmarkResult(
        operation="binary-predicates",
        tier=1,
        scale=scale,
        geometry_type="polygon",
        precision=precision,
        status="pass",
        status_reason="ok",
        timing=timing,
        baseline_name=baseline_name,
        baseline_timing=baseline_timing,
        speedup=speedup,
        metadata={"repeat": repeat, "predicate": predicate},
    )


@benchmark_operation(
    name="gpu-predicates",
    description="GPU binary predicates (contains) on point-centric workloads",
    category="predicate",
    geometry_types=("polygon", "point"),
    default_scale=100_000,
    tier=3,
    legacy_script="benchmark_gpu_predicates.py",
    tags=("gpu", "compare"),
)
def bench_gpu_predicates(
    *,
    scale: int,
    repeat: int,
    compare: str | None,
    precision: str,
    **kwargs: Any,
) -> BenchmarkResult:
    from time import perf_counter

    import numpy as np
    import shapely
    from shapely.affinity import translate

    from vibespatial import evaluate_binary_predicate, from_shapely_geometries
    from vibespatial.testing.synthetic import SyntheticSpec, generate_points, generate_polygons

    polygons = np.asarray(
        list(generate_polygons(
            SyntheticSpec("polygon", "regular-grid", count=scale, seed=0)
        ).geometries),
        dtype=object,
    )
    points = np.asarray(
        list(generate_points(
            SyntheticSpec("point", "grid", count=scale, seed=1)
        ).geometries),
        dtype=object,
    )
    half = scale // 2
    if half < scale:
        points[half:] = np.asarray(
            [translate(g, xoff=10_000.0, yoff=10_000.0) for g in points[half:]],
            dtype=object,
        )

    left_owned = from_shapely_geometries(polygons.tolist())
    right_owned = from_shapely_geometries(points.tolist())

    # Warmup
    evaluate_binary_predicate(left_owned, right_owned, "contains")

    times: list[float] = []
    for _ in range(max(1, repeat)):
        start = perf_counter()
        evaluate_binary_predicate(left_owned, right_owned, "contains")
        times.append(perf_counter() - start)

    timing = timing_from_samples(times)

    baseline_timing = None
    speedup = None
    baseline_name = None
    if compare == "shapely" or compare is None:
        shapely_times: list[float] = []
        for _ in range(max(1, repeat)):
            start = perf_counter()
            shapely.contains(polygons, points)
            shapely_times.append(perf_counter() - start)
        baseline_timing = timing_from_samples(shapely_times)
        baseline_name = "shapely"
        if timing.median_seconds > 0:
            speedup = baseline_timing.median_seconds / timing.median_seconds

    return BenchmarkResult(
        operation="gpu-predicates",
        tier=1,
        scale=scale,
        geometry_type="polygon+point",
        precision=precision,
        status="pass",
        status_reason="ok",
        timing=timing,
        baseline_name=baseline_name,
        baseline_timing=baseline_timing,
        speedup=speedup,
        metadata={"repeat": repeat, "predicate": "contains"},
    )


@benchmark_operation(
    name="point-predicates",
    description="Point predicate bootstrap kernels (bounds + exact)",
    category="predicate",
    geometry_types=("point", "polygon"),
    default_scale=10_000,
    tier=4,
    legacy_script="benchmark_point_predicates.py",
    tags=("gpu", "cpu"),
)
def bench_point_predicates(
    *,
    scale: int,
    repeat: int,
    compare: str | None,
    precision: str,
    **kwargs: Any,
) -> BenchmarkResult:
    from time import perf_counter

    from vibespatial import ExecutionMode, from_shapely_geometries
    from vibespatial.kernels.predicates import point_within_bounds
    from vibespatial.testing.synthetic import SyntheticSpec, generate_points, generate_polygons

    points_data = generate_points(
        SyntheticSpec("point", "uniform", count=scale, seed=7)
    ).geometries
    polygon_data = generate_polygons(
        SyntheticSpec("polygon", "regular-grid", count=max(1, scale // 10), seed=3)
    ).geometries
    tiled = [polygon_data[i % len(polygon_data)] for i in range(scale)]

    points = from_shapely_geometries(list(points_data))
    polygons = from_shapely_geometries(tiled)

    # Warmup
    point_within_bounds(points, polygons, dispatch_mode=ExecutionMode.CPU)

    times: list[float] = []
    for _ in range(max(1, repeat)):
        start = perf_counter()
        point_within_bounds(points, polygons, dispatch_mode=ExecutionMode.CPU)
        times.append(perf_counter() - start)

    timing = timing_from_samples(times)

    return BenchmarkResult(
        operation="point-predicates",
        tier=1,
        scale=scale,
        geometry_type="point+polygon",
        precision=precision,
        status="pass",
        status_reason="ok",
        timing=timing,
        metadata={"repeat": repeat},
    )
