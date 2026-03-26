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
    input_format: str = "parquet",
    **kwargs: Any,
) -> BenchmarkResult:
    from time import perf_counter

    from vibespatial import ExecutionMode, has_gpu_runtime
    from vibespatial.bench.fixture_loader import load_geodataframe, load_owned
    from vibespatial.bench.fixtures import InputFormat, resolve_fixture_spec
    from vibespatial.kernels.predicates.point_in_polygon import point_in_polygon

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
            input_format=input_format,
        )

    fmt = InputFormat(input_format)
    points_spec = resolve_fixture_spec("point", "grid", scale)
    polygon_base = max(scale // 8, 1)
    polygons_spec = resolve_fixture_spec("polygon", "regular-grid", polygon_base, vertices=5)

    points_owned, read_s1 = load_owned(points_spec, fmt)
    polygons_owned, read_s2 = load_owned(polygons_spec, fmt)
    read_seconds = read_s1 + read_s2

    # Baseline: Shapely
    baseline_timing = None
    speedup = None
    baseline_name = None
    if compare == "shapely" or compare is None:
        import shapely

        gdf_points, _ = load_geodataframe(points_spec, fmt)
        gdf_polys, _ = load_geodataframe(polygons_spec, fmt)
        points_arr = gdf_points.geometry.to_numpy()
        polys_arr = gdf_polys.geometry.to_numpy()
        if len(polys_arr) < scale:
            import numpy as np

            polys_arr = np.resize(polys_arr, scale)

        shapely_times: list[float] = []
        for _ in range(max(1, repeat)):
            start = perf_counter()
            shapely.covers(polys_arr, points_arr)
            shapely_times.append(perf_counter() - start)
        baseline_timing = timing_from_samples(shapely_times)
        baseline_name = "shapely"

    # Warmup
    point_in_polygon(points_owned, polygons_owned, dispatch_mode=ExecutionMode.GPU)

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
        input_format=input_format,
        read_seconds=read_seconds,
        metadata={"repeat": repeat, "polygon_base_count": polygon_base},
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
    input_format: str = "parquet",
    **kwargs: Any,
) -> BenchmarkResult:
    from time import perf_counter

    from vibespatial.bench.fixture_loader import load_geodataframe, load_owned
    from vibespatial.bench.fixtures import (
        InputFormat,
        ensure_shifted_fixture,
        resolve_fixture_spec,
    )
    from vibespatial.predicates.binary import evaluate_binary_predicate

    fmt = InputFormat(input_format)
    left_spec = resolve_fixture_spec("polygon", "regular-grid", scale)
    right_spec, _ = ensure_shifted_fixture(left_spec, xoff=0.3, yoff=0.3, fmt=fmt)

    left_owned, read_s1 = load_owned(left_spec, fmt)
    right_owned, read_s2 = load_owned(right_spec, fmt)
    read_seconds = read_s1 + read_s2

    predicate = "intersects"

    evaluate_binary_predicate(predicate, left_owned, right_owned)

    times: list[float] = []
    for _ in range(max(1, repeat)):
        start = perf_counter()
        evaluate_binary_predicate(predicate, left_owned, right_owned)
        times.append(perf_counter() - start)

    timing = timing_from_samples(times)

    baseline_timing = None
    speedup = None
    baseline_name = None
    if compare == "shapely" or compare is None:
        import shapely

        gdf_left, _ = load_geodataframe(left_spec, fmt)
        gdf_right, _ = load_geodataframe(right_spec, fmt)
        left_arr = gdf_left.geometry.to_numpy()
        right_arr = gdf_right.geometry.to_numpy()

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
        input_format=input_format,
        read_seconds=read_seconds,
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
    input_format: str = "parquet",
    **kwargs: Any,
) -> BenchmarkResult:
    from time import perf_counter

    from vibespatial import evaluate_binary_predicate
    from vibespatial.bench.fixture_loader import load_geodataframe, load_owned
    from vibespatial.bench.fixtures import InputFormat, resolve_fixture_spec

    fmt = InputFormat(input_format)
    polys_spec = resolve_fixture_spec("polygon", "regular-grid", scale)
    points_spec = resolve_fixture_spec("point", "grid", scale, seed=1)

    left_owned, read_s1 = load_owned(polys_spec, fmt)
    right_owned, read_s2 = load_owned(points_spec, fmt)
    read_seconds = read_s1 + read_s2

    evaluate_binary_predicate("contains", left_owned, right_owned)

    times: list[float] = []
    for _ in range(max(1, repeat)):
        start = perf_counter()
        evaluate_binary_predicate("contains", left_owned, right_owned)
        times.append(perf_counter() - start)

    timing = timing_from_samples(times)

    baseline_timing = None
    speedup = None
    baseline_name = None
    if compare == "shapely" or compare is None:
        import shapely

        gdf_left, _ = load_geodataframe(polys_spec, fmt)
        gdf_right, _ = load_geodataframe(points_spec, fmt)
        left_arr = gdf_left.geometry.to_numpy()
        right_arr = gdf_right.geometry.to_numpy()

        shapely_times: list[float] = []
        for _ in range(max(1, repeat)):
            start = perf_counter()
            shapely.contains(left_arr, right_arr)
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
        input_format=input_format,
        read_seconds=read_seconds,
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
    input_format: str = "parquet",
    **kwargs: Any,
) -> BenchmarkResult:
    from time import perf_counter

    from vibespatial import ExecutionMode
    from vibespatial.bench.fixture_loader import load_geodataframe, load_owned
    from vibespatial.bench.fixtures import InputFormat, resolve_fixture_spec
    from vibespatial.kernels.predicates import point_within_bounds

    fmt = InputFormat(input_format)
    points_spec = resolve_fixture_spec("point", "grid", scale)
    polys_spec = resolve_fixture_spec("polygon", "regular-grid", scale)

    points_owned, read_s1 = load_owned(points_spec, fmt)
    polys_owned, read_s2 = load_owned(polys_spec, fmt)
    read_seconds = read_s1 + read_s2

    # Guard: ensure both arrays have rows before calling the kernel.
    if points_owned.row_count == 0 or polys_owned.row_count == 0:
        return BenchmarkResult(
            operation="point-predicates",
            tier=1,
            scale=scale,
            geometry_type="point+polygon",
            precision=precision,
            status="error",
            status_reason=f"empty input: {points_owned.row_count} points, {polys_owned.row_count} polygons",
            timing=timing_from_samples([]),
            input_format=input_format,
            read_seconds=read_seconds,
        )

    # Align row counts: point_within_bounds requires equal-length inputs.
    # If the fixtures produce different counts (e.g. grid rounding), truncate
    # to the shorter one.
    min_rows = min(points_owned.row_count, polys_owned.row_count)
    if points_owned.row_count != polys_owned.row_count:
        import numpy as np

        if points_owned.row_count > min_rows:
            points_owned = points_owned.take(np.arange(min_rows, dtype=np.intp))
        if polys_owned.row_count > min_rows:
            polys_owned = polys_owned.take(np.arange(min_rows, dtype=np.intp))

    # Warmup — point_within_bounds has known bugs with some OGA layouts
    try:
        point_within_bounds(points_owned, polys_owned, dispatch_mode=ExecutionMode.CPU)
    except (IndexError, ValueError) as exc:
        return BenchmarkResult(
            operation="point-predicates",
            tier=1,
            scale=scale,
            geometry_type="point+polygon",
            precision=precision,
            status="error",
            status_reason=f"point_within_bounds bug: {exc}",
            timing=timing_from_samples([]),
            input_format=input_format,
            read_seconds=read_seconds,
        )

    times: list[float] = []
    for _ in range(max(1, repeat)):
        start = perf_counter()
        point_within_bounds(points_owned, polys_owned, dispatch_mode=ExecutionMode.CPU)
        times.append(perf_counter() - start)

    timing = timing_from_samples(times)

    # Baseline: Shapely within on the same data (pairwise point-in-polygon)
    baseline_timing = None
    speedup = None
    baseline_name = None
    if compare == "shapely" or compare is None:
        import shapely

        gdf_points, _ = load_geodataframe(points_spec, fmt)
        gdf_polys, _ = load_geodataframe(polys_spec, fmt)
        points_arr = gdf_points.geometry.to_numpy()
        polys_arr = gdf_polys.geometry.to_numpy()

        # Align to same row count
        if len(points_arr) != len(polys_arr):
            min_len = min(len(points_arr), len(polys_arr))
            points_arr = points_arr[:min_len]
            polys_arr = polys_arr[:min_len]

        shapely_times: list[float] = []
        for _ in range(max(1, repeat)):
            start = perf_counter()
            shapely.within(points_arr, polys_arr)
            shapely_times.append(perf_counter() - start)
        baseline_timing = timing_from_samples(shapely_times)
        baseline_name = "shapely"
        if timing.median_seconds > 0:
            speedup = baseline_timing.median_seconds / timing.median_seconds

    return BenchmarkResult(
        operation="point-predicates",
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
        input_format=input_format,
        read_seconds=read_seconds,
        metadata={"repeat": repeat},
    )
