"""Constructive operation benchmarks: clip-rect, gpu-constructive, make-valid, gpu-dissolve, stroke-kernels."""
from __future__ import annotations

import math
from typing import Any

from vibespatial.bench.catalog import OperationParameterSpec, benchmark_operation
from vibespatial.bench.schema import (
    BenchmarkResult,
    timing_from_samples,
)


def _clip_rect_from_bounds(bounds: Any, fraction: float = 0.6) -> tuple[float, float, float, float]:
    """Derive a clip rectangle from public total-bounds output.

    Returns the central ``fraction`` of the total bounding box, guaranteeing
    overlap with the input data.  Falls back to (0, 0, 1000, 1000) inner 60%
    if bounds cannot be computed.
    """
    xmin, ymin, xmax, ymax = (float(value) for value in bounds)
    if math.isnan(xmin) or math.isnan(ymin) or math.isnan(xmax) or math.isnan(ymax):
        # Fallback: use default fixture bounds (0..1000) with the given fraction
        margin = (1.0 - fraction) / 2.0
        return (1000.0 * margin, 1000.0 * margin, 1000.0 * (1.0 - margin), 1000.0 * (1.0 - margin))
    margin = (1.0 - fraction) / 2.0
    dx = (xmax - xmin) * margin
    dy = (ymax - ymin) * margin
    return (xmin + dx, ymin + dy, xmax - dx, ymax - dy)


def _clip_rect_from_owned(owned: Any, fraction: float = 0.6) -> tuple[float, float, float, float]:
    from vibespatial.kernels.core.geometry_analysis import compute_total_bounds

    return _clip_rect_from_bounds(compute_total_bounds(owned), fraction=fraction)


@benchmark_operation(
    name="clip-rect",
    description="Rectangle clipping fast-path benchmark",
    category="constructive",
    geometry_types=("line", "polygon"),
    default_scale=5_000,
    tier=4,
    parameters=(
        OperationParameterSpec(
            name="kind",
            value_type="choice",
            description="Geometry workload to benchmark",
            default="line",
            choices=("line", "polygon"),
        ),
        OperationParameterSpec(
            name="rect",
            value_type="float_list",
            description="Custom clip rectangle as xmin,ymin,xmax,ymax",
            arity=4,
        ),
    ),
    tags=("gpu", "compare"),
    max_scale=50_000,
)
def bench_clip_rect(
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

    kind = kwargs.get("kind", "line")
    if kind == "polygon":
        spec = resolve_fixture_spec("polygon", "regular-grid", scale)
    else:
        spec = resolve_fixture_spec("line", "random-walk", scale)
    gdf, read_seconds = load_public_geodataframe(spec, InputFormat(input_format))

    # Guard: if fixture produced 0 rows, report as error.
    if len(gdf) == 0:
        return BenchmarkResult(
            operation="clip-rect",
            tier=1,
            scale=scale,
            geometry_type=kind,
            precision=precision,
            status="error",
            status_reason="fixture produced 0 rows",
            timing=timing_from_samples([]),
            input_format=input_format,
            read_seconds=read_seconds,
        )

    # Derive clip rectangle from actual data bounds (central 60%) to
    # guarantee overlap regardless of scale or fixture distribution.
    rect = tuple(kwargs["rect"]) if "rect" in kwargs else _clip_rect_from_bounds(gdf.total_bounds, fraction=0.6)

    try:
        gdf.geometry.clip_by_rect(*rect)
    except (IndexError, ValueError) as exc:
        return BenchmarkResult(
            operation="clip-rect",
            tier=1,
            scale=scale,
            geometry_type=kind,
            precision=precision,
            status="error",
            status_reason=f"clip_by_rect crashed: {exc}",
            timing=timing_from_samples([]),
            input_format=input_format,
            read_seconds=read_seconds,
        )

    times: list[float] = []
    for _ in range(max(1, repeat)):
        start = perf_counter()
        clipped = gdf.geometry.clip_by_rect(*rect)
        times.append(perf_counter() - start)

    timing = timing_from_samples(times)

    speedup = None
    baseline_timing = None
    baseline_name = None
    if compare == "shapely" or compare is None:
        values = gdf.geometry.to_numpy()
        baseline_times: list[float] = []
        for _ in range(max(1, repeat)):
            start = perf_counter()
            shapely.clip_by_rect(values, *rect)
            baseline_times.append(perf_counter() - start)
        baseline_timing = timing_from_samples(baseline_times)
        baseline_name = "shapely"
        if timing.median_seconds > 0:
            speedup = baseline_timing.median_seconds / timing.median_seconds

    return BenchmarkResult(
        operation="clip-rect",
        tier=1,
        scale=scale,
        geometry_type=kind,
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
            "kind": kind,
            "rect": list(rect),
            "result_rows": int(len(clipped)),
        },
    )


@benchmark_operation(
    name="gpu-constructive",
    description="GPU constructive operations (clip + buffer pipeline)",
    category="constructive",
    geometry_types=("polygon",),
    default_scale=10_000,
    tier=4,
    tags=("gpu", "compare"),
    public_api=False,
)
def bench_gpu_constructive(
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
    from vibespatial.bench.fixtures import InputFormat, resolve_fixture_spec
    from vibespatial.constructive.clip_rect import clip_by_rect_owned

    if not has_gpu_runtime():
        return BenchmarkResult(
            operation="gpu-constructive",
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
    spec = resolve_fixture_spec("polygon", "regular-grid", scale)
    owned, read_seconds = load_owned(spec, fmt)

    # Guard: if fixture produced 0 rows, report as error.
    if owned.row_count == 0:
        return BenchmarkResult(
            operation="gpu-constructive",
            tier=1,
            scale=scale,
            geometry_type="polygon",
            precision=precision,
            status="error",
            status_reason="fixture produced 0 rows",
            timing=timing_from_samples([]),
            input_format=input_format,
            read_seconds=read_seconds,
        )

    # Derive clip rectangle from actual data bounds (central 60%) to
    # guarantee overlap regardless of scale or fixture distribution.
    rect = _clip_rect_from_owned(owned, fraction=0.6)

    # Warmup run with guard for kernel-level crashes.
    try:
        warmup_result = clip_by_rect_owned(owned, *rect)
    except (IndexError, ValueError) as exc:
        return BenchmarkResult(
            operation="gpu-constructive",
            tier=1,
            scale=scale,
            geometry_type="polygon",
            precision=precision,
            status="error",
            status_reason=f"clip warmup crashed: {exc}",
            timing=timing_from_samples([]),
            input_format=input_format,
            read_seconds=read_seconds,
        )

    # Guard: if the clip produces 0 rows even with data-derived rect,
    # report as error rather than crashing downstream.
    if warmup_result.owned_result is not None and warmup_result.owned_result.row_count == 0:
        return BenchmarkResult(
            operation="gpu-constructive",
            tier=1,
            scale=scale,
            geometry_type="polygon",
            precision=precision,
            status="error",
            status_reason="clip produced 0 rows with data-derived rect",
            timing=timing_from_samples([]),
            input_format=input_format,
            read_seconds=read_seconds,
        )

    times: list[float] = []
    for _ in range(max(1, repeat)):
        start = perf_counter()
        clip_by_rect_owned(owned, *rect)
        times.append(perf_counter() - start)

    timing = timing_from_samples(times)

    # Baseline: Shapely clip_by_rect on the same data
    baseline_timing = None
    speedup = None
    baseline_name = None
    if compare == "shapely" or compare is None:
        import shapely

        gdf, _ = load_geodataframe(spec, fmt)
        geom_arr = gdf.geometry.to_numpy()

        shapely_times: list[float] = []
        for _ in range(max(1, repeat)):
            start = perf_counter()
            shapely.clip_by_rect(geom_arr, *rect)
            shapely_times.append(perf_counter() - start)
        baseline_timing = timing_from_samples(shapely_times)
        baseline_name = "shapely"
        if timing.median_seconds > 0:
            speedup = baseline_timing.median_seconds / timing.median_seconds

    return BenchmarkResult(
        operation="gpu-constructive",
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
        metadata={"repeat": repeat, "rect": list(rect)},
    )


@benchmark_operation(
    name="make-valid",
    description="make_valid benchmark on fixture with ~5% invalid polygons",
    category="constructive",
    geometry_types=("polygon",),
    default_scale=10_000,
    tier=4,
    tags=("gpu", "compare"),
)
def bench_make_valid(
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
    from vibespatial.bench.fixtures import InputFormat, ensure_invalids_fixture

    fmt = InputFormat(input_format)
    spec, _ = ensure_invalids_fixture(scale, fmt=fmt)

    gdf, read_seconds = load_public_geodataframe(spec, fmt)

    gdf.geometry.make_valid()
    times: list[float] = []
    for _ in range(max(1, repeat)):
        start = perf_counter()
        result = gdf.geometry.make_valid()
        times.append(perf_counter() - start)

    timing = timing_from_samples(times)

    baseline_times: list[float] = []
    values = gdf.geometry.to_numpy()
    for _ in range(max(1, repeat)):
        start = perf_counter()
        shapely.make_valid(values)
        baseline_times.append(perf_counter() - start)
    baseline_timing = timing_from_samples(baseline_times)
    speedup = (
        baseline_timing.median_seconds / timing.median_seconds
        if timing.median_seconds > 0
        else None
    )

    return BenchmarkResult(
        operation="make-valid",
        tier=1,
        scale=scale,
        geometry_type="polygon",
        precision=precision,
        status="pass",
        status_reason="ok",
        timing=timing,
        baseline_name="baseline",
        baseline_timing=baseline_timing,
        speedup=speedup,
        input_format=input_format,
        read_seconds=read_seconds,
        metadata={"repaired_rows": int(len(result))},
    )


@benchmark_operation(
    name="gpu-dissolve",
    description="Grouped GPU dissolve for rectangle coverages",
    category="constructive",
    geometry_types=("polygon",),
    default_scale=50_000,
    tier=2,
    parameters=(
        OperationParameterSpec(
            name="groups",
            value_type="int",
            description="Number of dissolve groups in the grouped-box fixture",
            default=100,
        ),
        OperationParameterSpec(
            name="method",
            value_type="choice",
            description="Public GeoDataFrame.dissolve union method",
            default="coverage",
            choices=("coverage", "unary", "disjoint_subset"),
        ),
    ),
    tags=("gpu",),
)
def bench_gpu_dissolve(
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
    from vibespatial.bench.fixtures import InputFormat, ensure_grouped_boxes_fixture

    fmt = InputFormat(input_format)
    groups = kwargs.get("groups", 100)
    method = str(kwargs.get("method", "coverage"))
    spec, _ = ensure_grouped_boxes_fixture(scale, groups=groups, fmt=fmt)
    frame, read_seconds = load_public_geodataframe(spec, fmt)

    frame.dissolve(by="group", method=method)

    times: list[float] = []
    last_result = None
    for _ in range(max(1, repeat)):
        start = perf_counter()
        last_result = frame.dissolve(by="group", method=method)
        times.append(perf_counter() - start)

    timing = timing_from_samples(times)

    # Baseline: Shapely unary_union per group (the CPU equivalent of dissolve)
    baseline_timing = None
    speedup = None
    baseline_name = None
    baseline_elapsed_seconds = None
    baseline_rows = None
    if compare == "shapely" or compare is None:
        values = frame.geometry.to_numpy()
        grouped = frame.groupby("group", sort=True, observed=False, dropna=True)[
            frame.geometry.name
        ]
        group_positions = [
            positions
            for _, positions in grouped.indices.items()
        ]
        shapely_times: list[float] = []
        baseline = None
        baseline_func = {
            "coverage": shapely.coverage_union_all,
            "disjoint_subset": shapely.disjoint_subset_union_all,
        }.get(method, shapely.union_all)
        for _ in range(max(1, repeat)):
            start = perf_counter()
            baseline = [
                baseline_func(values[positions])
                for positions in group_positions
                if positions.size > 0
            ]
            shapely_times.append(perf_counter() - start)
        if baseline is not None:
            baseline_rows = len(baseline)
        if shapely_times:
            baseline_elapsed_seconds = shapely_times[-1]
        baseline_timing = timing_from_samples(shapely_times)
        baseline_name = f"shapely-{method}"
        if timing.median_seconds > 0:
            speedup = baseline_timing.median_seconds / timing.median_seconds

    return BenchmarkResult(
        operation="gpu-dissolve",
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
            "repeat": repeat,
            "groups": groups,
            "method": method,
            "public_elapsed_seconds": timing.median_seconds,
            "baseline_elapsed_seconds": baseline_elapsed_seconds,
            "result_rows": int(len(last_result)) if last_result is not None else None,
            "baseline_rows": baseline_rows,
        },
    )


@benchmark_operation(
    name="stroke-kernels",
    description="Point buffer and offset curve stroke kernels",
    category="constructive",
    geometry_types=("point", "line"),
    default_scale=1_000,
    tier=4,
    parameters=(
        OperationParameterSpec(
            name="kind",
            value_type="choice",
            description="Stroke kernel workload to benchmark",
            default="point-buffer",
            choices=("point-buffer", "offset-curve"),
        ),
    ),
    tags=("gpu", "compare"),
    max_scale=50_000,
)
def bench_stroke_kernels(
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

    kind = kwargs.get("kind", "point-buffer")
    geometry_type = "point" if kind == "point-buffer" else "line"
    distribution = "grid" if geometry_type == "point" else "random-walk"
    spec = resolve_fixture_spec(geometry_type, distribution, scale)
    gdf, read_seconds = load_public_geodataframe(spec, InputFormat(input_format))

    if kind == "offset-curve":
        gdf.geometry.offset_curve(1.0, join_style="mitre")
    else:
        gdf.geometry.buffer(1.0, quad_segs=8)

    times: list[float] = []
    for _ in range(max(1, repeat)):
        start = perf_counter()
        if kind == "offset-curve":
            result = gdf.geometry.offset_curve(1.0, join_style="mitre")
        else:
            result = gdf.geometry.buffer(1.0, quad_segs=8)
        times.append(perf_counter() - start)

    timing = timing_from_samples(times)

    values = gdf.geometry.to_numpy()
    baseline_times: list[float] = []
    if kind == "offset-curve":
        for _ in range(max(1, repeat)):
            start = perf_counter()
            shapely.offset_curve(values, distance=1.0, join_style="mitre")
            baseline_times.append(perf_counter() - start)
    else:
        for _ in range(max(1, repeat)):
            start = perf_counter()
            shapely.buffer(values, distance=1.0, quad_segs=8)
            baseline_times.append(perf_counter() - start)

    baseline_timing = timing_from_samples(baseline_times)
    speedup = (
        baseline_timing.median_seconds / timing.median_seconds
        if timing.median_seconds > 0
        else None
    )

    return BenchmarkResult(
        operation="stroke-kernels",
        tier=1,
        scale=scale,
        geometry_type=geometry_type,
        precision=precision,
        status="pass",
        status_reason="ok",
        timing=timing,
        baseline_name="shapely" if baseline_timing else None,
        baseline_timing=baseline_timing,
        speedup=speedup,
        input_format=input_format,
        read_seconds=read_seconds,
        metadata={
            "kind": kind,
            "result_rows": int(len(result)),
        },
    )
