"""IO operation benchmarks: io-arrow, io-file, gpu-decode, mixed-layouts."""
from __future__ import annotations

from typing import Any

from vibespatial.bench.catalog import benchmark_operation
from vibespatial.bench.schema import (
    BenchmarkResult,
    timing_from_samples,
)


@benchmark_operation(
    name="io-arrow",
    description="GeoArrow / WKB encode roundtrip benchmark",
    category="io",
    geometry_types=("point", "polygon"),
    default_scale=100_000,
    tier=1,
    tags=("io",),
)
def bench_io_arrow(
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

    fmt = InputFormat(input_format)
    spec = resolve_fixture_spec("point", "grid", scale)
    frame, read_seconds = load_public_geodataframe(spec, fmt)

    frame.geometry.to_wkb()

    times: list[float] = []
    for _ in range(max(1, repeat)):
        start = perf_counter()
        frame.geometry.to_wkb()
        times.append(perf_counter() - start)

    timing = timing_from_samples(times)

    # Baseline: Shapely to_wkb encoding on the same data
    baseline_timing = None
    speedup = None
    baseline_name = None
    if compare == "shapely" or compare is None:
        geom_arr = frame.geometry.to_numpy()

        shapely_times: list[float] = []
        for _ in range(max(1, repeat)):
            start = perf_counter()
            shapely.to_wkb(geom_arr)
            shapely_times.append(perf_counter() - start)
        baseline_timing = timing_from_samples(shapely_times)
        baseline_name = "shapely"
        if timing.median_seconds > 0:
            speedup = baseline_timing.median_seconds / timing.median_seconds

    return BenchmarkResult(
        operation="io-arrow",
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
        metadata={"repeat": repeat, "format": "public-wkb-encode"},
    )


@benchmark_operation(
    name="io-file",
    description="File-format IO benchmark (GeoJSON, Shapefile)",
    category="io",
    geometry_types=("point",),
    default_scale=100_000,
    tier=1,
    tags=("io",),
)
def bench_io_file(
    *,
    scale: int,
    repeat: int,
    compare: str | None,
    precision: str,
    input_format: str = "parquet",
    **kwargs: Any,
) -> BenchmarkResult:
    from time import perf_counter

    import vibespatial.api as geopandas
    from vibespatial.bench.fixtures import (
        InputFormat,
        ensure_fixture_format,
        resolve_fixture_spec,
    )

    # This benchmark measures the read itself — input_format IS the operation
    fmt = InputFormat(input_format)
    spec = resolve_fixture_spec("point", "grid", scale)
    path = ensure_fixture_format(spec, fmt)

    def _read_public():
        if fmt == InputFormat.PARQUET:
            return geopandas.read_parquet(path)
        return geopandas.read_file(str(path))

    _read_public()

    times: list[float] = []
    frame = None
    for _ in range(max(1, repeat)):
        start = perf_counter()
        frame = _read_public()
        times.append(perf_counter() - start)

    timing = timing_from_samples(times)

    # Time explicit CPU/container readers as a reference denominator. Importing
    # ``geopandas`` inside this repo resolves to the compatibility shim, so do
    # not label that as a real GeoPandas baseline.
    baseline_timing = None
    speedup = None
    baseline_name = None
    if compare in (None, "geopandas"):
        try:
            if fmt == InputFormat.PARQUET:
                import pyarrow.parquet as pq

                def _read_reference():
                    return pq.read_table(path)

                baseline_name = "pyarrow-parquet"
            else:
                import pyogrio

                def _read_reference():
                    return pyogrio.read_arrow(path)

                baseline_name = f"pyogrio-arrow-{input_format}"
        except ImportError:
            baseline_name = None
        else:
            baseline_times: list[float] = []
            for _ in range(max(1, repeat)):
                start = perf_counter()
                _read_reference()
                baseline_times.append(perf_counter() - start)
            baseline_timing = timing_from_samples(baseline_times)
            if timing.median_seconds > 0:
                speedup = baseline_timing.median_seconds / timing.median_seconds

    return BenchmarkResult(
        operation="io-file",
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
        read_seconds=timing.median_seconds,
        metadata={"format": input_format, "result_rows": len(frame) if frame is not None else None},
    )


@benchmark_operation(
    name="gpu-decode",
    description="GPU-accelerated WKB/GeoParquet decode benchmark",
    category="io",
    geometry_types=("point",),
    default_scale=100_000,
    tier=1,
    tags=("gpu", "io"),
    public_api=False,
)
def bench_gpu_decode(
    *,
    scale: int,
    repeat: int,
    compare: str | None,
    precision: str,
    input_format: str = "parquet",
    **kwargs: Any,
) -> BenchmarkResult:
    from time import perf_counter

    from vibespatial import decode_wkb_owned
    from vibespatial.bench.fixture_loader import load_owned
    from vibespatial.bench.fixtures import InputFormat, resolve_fixture_spec
    from vibespatial.io.wkb import encode_wkb_owned

    spec = resolve_fixture_spec("point", "grid", scale)
    owned, _ = load_owned(spec, InputFormat(input_format))
    wkb_data = encode_wkb_owned(owned)

    decode_wkb_owned(wkb_data)

    times: list[float] = []
    for _ in range(max(1, repeat)):
        start = perf_counter()
        decode_wkb_owned(wkb_data)
        times.append(perf_counter() - start)

    timing = timing_from_samples(times)

    # Baseline: Shapely from_wkb decoding on the same WKB data
    baseline_timing = None
    speedup = None
    baseline_name = None
    if compare == "shapely" or compare is None:
        import shapely

        # Convert WKB data to format Shapely expects (list of bytes)
        wkb_list = list(wkb_data)

        shapely_times: list[float] = []
        for _ in range(max(1, repeat)):
            start = perf_counter()
            shapely.from_wkb(wkb_list)
            shapely_times.append(perf_counter() - start)
        baseline_timing = timing_from_samples(shapely_times)
        baseline_name = "shapely"
        if timing.median_seconds > 0:
            speedup = baseline_timing.median_seconds / timing.median_seconds

    return BenchmarkResult(
        operation="gpu-decode",
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
        metadata={"repeat": repeat, "format": "wkb-decode"},
    )


@benchmark_operation(
    name="mixed-layouts",
    description="Mixed geometry layout benchmark (tagged vs separate)",
    category="io",
    geometry_types=("mixed",),
    default_scale=100_000,
    tier=3,
    tags=("layout",),
    public_api=False,
)
def bench_mixed_layouts(
    *,
    scale: int,
    repeat: int,
    compare: str | None,
    precision: str,
    input_format: str = "parquet",
    **kwargs: Any,
) -> BenchmarkResult:
    from vibespatial.testing.mixed_layouts import benchmark_matrix

    results = benchmark_matrix((scale,), seed=0)
    if not results:
        return BenchmarkResult(
            operation="mixed-layouts",
            tier=1,
            scale=scale,
            geometry_type="mixed",
            precision=precision,
            status="error",
            status_reason="no results from benchmark_matrix",
            timing=timing_from_samples([]),
            input_format=input_format,
        )

    r = results[0]
    timing = timing_from_samples([r.tagged_prep_ms / 1000.0])

    return BenchmarkResult(
        operation="mixed-layouts",
        tier=1,
        scale=scale,
        geometry_type="mixed",
        precision=precision,
        status="pass",
        status_reason="ok",
        timing=timing,
        input_format=input_format,
        metadata={
            "dataset_name": r.dataset_name,
            "warp_purity": r.tagged_warp_purity,
            "tagged_ms": r.tagged_prep_ms,
            "separate_ms": r.separate_prep_ms,
            "recommended_default": r.recommended_default,
        },
    )
