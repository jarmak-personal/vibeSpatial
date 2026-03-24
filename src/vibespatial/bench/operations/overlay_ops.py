"""Overlay operation benchmarks: gpu-overlay, segment-filters, segment-intersections, segment-primitives."""
from __future__ import annotations

from typing import Any

from vibespatial.bench.catalog import benchmark_operation
from vibespatial.bench.schema import (
    BenchmarkResult,
    timing_from_samples,
)


@benchmark_operation(
    name="gpu-overlay",
    description="GPU polygon overlay intersection",
    category="overlay",
    geometry_types=("polygon",),
    default_scale=1_000,
    tier=2,
    legacy_script="benchmark_gpu_overlay.py",
    tags=("gpu",),
    max_scale=10_000,
)
def bench_gpu_overlay(
    *,
    scale: int,
    repeat: int,
    compare: str | None,
    precision: str,
    input_format: str = "parquet",
    **kwargs: Any,
) -> BenchmarkResult:
    from time import perf_counter

    from vibespatial import has_gpu_runtime
    from vibespatial.bench.fixture_loader import load_geodataframe, load_owned
    from vibespatial.bench.fixtures import (
        InputFormat,
        ensure_shifted_fixture,
        resolve_fixture_spec,
    )
    from vibespatial.overlay.gpu import overlay_intersection_owned

    if not has_gpu_runtime():
        return BenchmarkResult(
            operation="gpu-overlay",
            tier=1,
            scale=scale,
            geometry_type="polygon",
            precision=precision,
            status="skip",
            status_reason="GPU runtime not available",
            timing=timing_from_samples([]),
            input_format=input_format,
        )

    fmt = InputFormat(input_format)
    left_spec = resolve_fixture_spec("polygon", "regular-grid", scale)
    # Shift by a fraction of the cell size to guarantee overlap between
    # the left and right polygon grids.  A fixed 0.3 offset may not
    # overlap at every scale, so we derive the shift from the grid
    # cell dimensions.
    import math

    side = max(1, math.ceil(math.sqrt(scale)))
    xmin, ymin, xmax, ymax = left_spec.bounds
    cell_w = (xmax - xmin) / side
    cell_h = (ymax - ymin) / side
    shift_x = cell_w * 0.3
    shift_y = cell_h * 0.3
    right_spec, _ = ensure_shifted_fixture(left_spec, xoff=shift_x, yoff=shift_y, fmt=fmt)

    left_owned, read_s1 = load_owned(left_spec, fmt)
    right_owned, read_s2 = load_owned(right_spec, fmt)
    read_seconds = read_s1 + read_s2

    # Warmup with error guard: overlay on non-overlapping inputs can
    # crash deep in the kernel with opaque IndexError / reshape errors.
    try:
        overlay_intersection_owned(left_owned, right_owned)
    except Exception as exc:
        return BenchmarkResult(
            operation="gpu-overlay",
            tier=1,
            scale=scale,
            geometry_type="polygon",
            precision=precision,
            status="error",
            status_reason=f"overlay warmup failed: {exc}",
            timing=timing_from_samples([]),
            input_format=input_format,
            read_seconds=read_seconds,
        )

    times: list[float] = []
    for _ in range(max(1, repeat)):
        start = perf_counter()
        overlay_intersection_owned(left_owned, right_owned)
        times.append(perf_counter() - start)

    timing = timing_from_samples(times)

    # Baseline: Shapely pairwise intersection on the same data
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
            shapely.intersection(left_arr, right_arr)
            shapely_times.append(perf_counter() - start)
        baseline_timing = timing_from_samples(shapely_times)
        baseline_name = "shapely"
        if timing.median_seconds > 0:
            speedup = baseline_timing.median_seconds / timing.median_seconds

    return BenchmarkResult(
        operation="gpu-overlay",
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
        metadata={"repeat": repeat},
    )


@benchmark_operation(
    name="segment-filters",
    description="Segment-level MBR filtering benchmark",
    category="overlay",
    geometry_types=("polygon",),
    default_scale=200,
    tier=3,
    legacy_script="benchmark_segment_filters.py",
    tags=("gpu",),
    max_scale=5_000,
)
def bench_segment_filters(
    *,
    scale: int,
    repeat: int,
    compare: str | None,
    precision: str,
    input_format: str = "parquet",
    **kwargs: Any,
) -> BenchmarkResult:
    from time import perf_counter

    from vibespatial import benchmark_segment_filter
    from vibespatial.bench.fixture_loader import load_geodataframe, load_owned
    from vibespatial.bench.fixtures import (
        InputFormat,
        ensure_shifted_fixture,
        resolve_fixture_spec,
    )

    fmt = InputFormat(input_format)
    left_spec = resolve_fixture_spec("polygon", "regular-grid", scale)
    right_spec, _ = ensure_shifted_fixture(left_spec, xoff=5.0, yoff=0.0, fmt=fmt)

    left_owned, read_s1 = load_owned(left_spec, fmt)
    right_owned, read_s2 = load_owned(right_spec, fmt)
    read_seconds = read_s1 + read_s2

    result = benchmark_segment_filter(left_owned, right_owned, tile_size=512)

    timing = timing_from_samples([result.elapsed_seconds])

    # Baseline: Shapely STRtree candidate filtering (the CPU equivalent
    # of segment-level MBR pair generation)
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
            tree = shapely.STRtree(left_arr)
            tree.query(right_arr, predicate="intersects")
            shapely_times.append(perf_counter() - start)
        baseline_timing = timing_from_samples(shapely_times)
        baseline_name = "shapely-strtree"
        if timing.median_seconds > 0:
            speedup = baseline_timing.median_seconds / timing.median_seconds

    return BenchmarkResult(
        operation="segment-filters",
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
        metadata={
            "naive_segment_pairs": result.naive_segment_pairs,
            "filtered_segment_pairs": result.filtered_segment_pairs,
        },
    )


@benchmark_operation(
    name="segment-intersections",
    description="Segment intersection classification benchmark",
    category="overlay",
    geometry_types=("line",),
    default_scale=200,
    tier=3,
    legacy_script="benchmark_segment_intersections.py",
    tags=("gpu",),
    max_scale=5_000,
)
def bench_segment_intersections(
    *,
    scale: int,
    repeat: int,
    compare: str | None,
    precision: str,
    input_format: str = "parquet",
    **kwargs: Any,
) -> BenchmarkResult:
    from time import perf_counter

    from vibespatial import classify_segment_intersections, generate_segment_candidates
    from vibespatial.bench.fixture_loader import load_geodataframe, load_owned
    from vibespatial.bench.fixtures import (
        InputFormat,
        ensure_shifted_fixture,
        resolve_fixture_spec,
    )

    fmt = InputFormat(input_format)
    left_spec = resolve_fixture_spec("line", "random-walk", scale)
    right_spec, _ = ensure_shifted_fixture(left_spec, xoff=0.5, yoff=0.5, fmt=fmt)

    left_owned, read_s1 = load_owned(left_spec, fmt)
    right_owned, read_s2 = load_owned(right_spec, fmt)
    read_seconds = read_s1 + read_s2

    candidates = generate_segment_candidates(left_owned, right_owned)

    classify_segment_intersections(left_owned, right_owned, candidate_pairs=candidates)

    times: list[float] = []
    for _ in range(max(1, repeat)):
        start = perf_counter()
        classify_segment_intersections(left_owned, right_owned, candidate_pairs=candidates)
        times.append(perf_counter() - start)

    timing = timing_from_samples(times)

    # Baseline: Shapely pairwise intersection on the same line geometries
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
            shapely.intersection(left_arr, right_arr)
            shapely_times.append(perf_counter() - start)
        baseline_timing = timing_from_samples(shapely_times)
        baseline_name = "shapely"
        if timing.median_seconds > 0:
            speedup = baseline_timing.median_seconds / timing.median_seconds

    return BenchmarkResult(
        operation="segment-intersections",
        tier=1,
        scale=scale,
        geometry_type="line",
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
    name="segment-primitives",
    description="Segment intersection primitive benchmark",
    category="overlay",
    geometry_types=("line",),
    default_scale=200,
    tier=3,
    legacy_script="benchmark_segment_primitives.py",
    tags=("gpu",),
    max_scale=5_000,
)
def bench_segment_primitives(
    *,
    scale: int,
    repeat: int,
    compare: str | None,
    precision: str,
    input_format: str = "parquet",
    **kwargs: Any,
) -> BenchmarkResult:
    from time import perf_counter

    from vibespatial import benchmark_segment_intersections
    from vibespatial.bench.fixture_loader import load_geodataframe, load_owned
    from vibespatial.bench.fixtures import (
        InputFormat,
        ensure_shifted_fixture,
        resolve_fixture_spec,
    )

    fmt = InputFormat(input_format)
    left_spec = resolve_fixture_spec("line", "random-walk", scale)
    right_spec, _ = ensure_shifted_fixture(left_spec, xoff=0.5, yoff=0.5, fmt=fmt)

    left_owned, read_s1 = load_owned(left_spec, fmt)
    right_owned, read_s2 = load_owned(right_spec, fmt)
    read_seconds = read_s1 + read_s2

    result = benchmark_segment_intersections(left_owned, right_owned, tile_size=512)

    timing = timing_from_samples([result.elapsed_seconds])

    # Baseline: Shapely pairwise intersection (the CPU equivalent of
    # the full segment intersection primitive pipeline)
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
            shapely.intersection(left_arr, right_arr)
            shapely_times.append(perf_counter() - start)
        baseline_timing = timing_from_samples(shapely_times)
        baseline_name = "shapely"
        if timing.median_seconds > 0:
            speedup = baseline_timing.median_seconds / timing.median_seconds

    return BenchmarkResult(
        operation="segment-primitives",
        tier=1,
        scale=scale,
        geometry_type="line",
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
            "candidate_pairs": result.candidate_pairs,
            "proper_pairs": result.proper_pairs,
        },
    )
