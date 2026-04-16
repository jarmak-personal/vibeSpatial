from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from vibespatial import (
    benchmark_geoarrow_bridge,
    benchmark_native_geometry_codec,
    benchmark_shapefile_ingest,
    benchmark_wkb_bridge,
    build_geoparquet_metadata_summary,
    decode_wkb_owned,
    from_shapely_geometries,
    from_wkb,
    has_pylibcudf_support,
    read_geoparquet_owned,
)
from vibespatial.geometry.owned import FamilyGeometryBuffer, OwnedGeometryArray
from vibespatial.io.geoparquet_planner import select_row_groups
from vibespatial.testing.synthetic import (
    SyntheticSpec,
    generate_lines,
    generate_mixed_geometries,
    generate_points,
    generate_polygons,
)

_GEOPARQUET_BENCH_WARMUP_READS = 2
_GEOPARQUET_SCAN_BENCH_MIN_REPEAT = 3
_SHAPEFILE_BENCH_MIN_REPEAT = 3


@dataclass(frozen=True)
class IOBenchmarkCase:
    case_id: str
    family: str
    format_name: str
    geometry_profile: str
    scale: int
    metric: str
    unit: str
    enforced: bool
    status: str
    target_floor: float | None
    actual_value: float | None
    baseline_label: str | None
    candidate_label: str | None
    baseline_rows_per_second: float | None
    candidate_rows_per_second: float | None
    rows_input: int
    rows_decoded: int
    bytes_scanned: int | None
    copies_made: int | None
    fallback_pool_share: float | None
    notes: str


def _estimate_family_bytes(buffer: FamilyGeometryBuffer) -> int:
    total = 0
    for name in ("empty_mask", "geometry_offsets", "x", "y", "part_offsets", "ring_offsets"):
        value = getattr(buffer, name, None)
        if value is None:
            continue
        total += int(getattr(value, "nbytes", 0))
    return total


def _estimate_owned_bytes(array: OwnedGeometryArray) -> int:
    total = int(array.validity.nbytes + array.tags.nbytes + array.family_row_offsets.nbytes)
    for buffer in array.families.values():
        total += _estimate_family_bytes(buffer)
    return total


def _benchmark_lookup(results: list[object], key: str, *, attr: str = "implementation") -> object:
    for result in results:
        if getattr(result, attr) == key:
            return result
    raise KeyError(f"Benchmark result {key!r} was not present")


def _speedup_case(
    *,
    case_id: str,
    family: str,
    format_name: str,
    geometry_profile: str,
    scale: int,
    target_floor: float,
    enforced: bool,
    baseline_label: str,
    candidate_label: str,
    baseline_rows_per_second: float,
    candidate_rows_per_second: float,
    rows_decoded: int,
    bytes_scanned: int | None,
    copies_made: int | None,
    fallback_pool_share: float | None = None,
    notes: str = "",
) -> IOBenchmarkCase:
    actual = (
        candidate_rows_per_second / baseline_rows_per_second
        if baseline_rows_per_second and baseline_rows_per_second > 0.0
        else None
    )
    status = "pass" if actual is not None and actual >= target_floor else "fail"
    if not enforced and status == "fail":
        status = "informational"
    return IOBenchmarkCase(
        case_id=case_id,
        family=family,
        format_name=format_name,
        geometry_profile=geometry_profile,
        scale=scale,
        metric="speedup",
        unit="x",
        enforced=enforced,
        status=status,
        target_floor=target_floor,
        actual_value=actual,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        baseline_rows_per_second=baseline_rows_per_second,
        candidate_rows_per_second=candidate_rows_per_second,
        rows_input=scale,
        rows_decoded=rows_decoded,
        bytes_scanned=bytes_scanned,
        copies_made=copies_made,
        fallback_pool_share=fallback_pool_share,
        notes=notes,
    )


def _fraction_case(
    *,
    case_id: str,
    family: str,
    format_name: str,
    geometry_profile: str,
    scale: int,
    target_floor: float,
    actual_value: float,
    enforced: bool,
    rows_decoded: int,
    notes: str,
) -> IOBenchmarkCase:
    status = "pass" if actual_value <= target_floor else "fail"
    if not enforced and status == "fail":
        status = "informational"
    return IOBenchmarkCase(
        case_id=case_id,
        family=family,
        format_name=format_name,
        geometry_profile=geometry_profile,
        scale=scale,
        metric="decoded_row_fraction",
        unit="fraction",
        enforced=enforced,
        status=status,
        target_floor=target_floor,
        actual_value=actual_value,
        baseline_label=None,
        candidate_label="metadata_prune",
        baseline_rows_per_second=None,
        candidate_rows_per_second=None,
        rows_input=scale,
        rows_decoded=rows_decoded,
        bytes_scanned=None,
        copies_made=0,
        fallback_pool_share=0.0,
        notes=notes,
    )


def _unavailable_case(
    *,
    case_id: str,
    family: str,
    format_name: str,
    geometry_profile: str,
    scale: int,
    notes: str,
    enforced: bool,
) -> IOBenchmarkCase:
    return IOBenchmarkCase(
        case_id=case_id,
        family=family,
        format_name=format_name,
        geometry_profile=geometry_profile,
        scale=scale,
        metric="speedup",
        unit="x",
        enforced=enforced,
        status="unavailable",
        target_floor=None,
        actual_value=None,
        baseline_label=None,
        candidate_label=None,
        baseline_rows_per_second=None,
        candidate_rows_per_second=None,
        rows_input=scale,
        rows_decoded=0,
        bytes_scanned=None,
        copies_made=None,
        fallback_pool_share=None,
        notes=notes,
    )


def _sample_owned(geometry_type: str, rows: int, *, seed: int = 0) -> OwnedGeometryArray:
    if geometry_type == "point":
        dataset = generate_points(SyntheticSpec("point", "uniform", count=rows, seed=seed))
    elif geometry_type == "line":
        dataset = generate_lines(SyntheticSpec("line", "grid", count=rows, seed=seed, vertices=8))
    elif geometry_type == "polygon":
        dataset = generate_polygons(
            SyntheticSpec("polygon", "regular-grid", count=rows, seed=seed, vertices=6)
        )
    else:
        raise ValueError(f"Unsupported geometry type: {geometry_type}")
    return from_shapely_geometries(list(dataset.geometries))


def _sample_geodataframe(geometry_type: str, rows: int, *, seed: int = 0):
    if geometry_type == "point":
        return generate_points(SyntheticSpec("point", "uniform", count=rows, seed=seed)).to_geodataframe()
    if geometry_type == "polygon":
        return generate_polygons(
            SyntheticSpec("polygon", "regular-grid", count=rows, seed=seed, vertices=6)
        ).to_geodataframe()
    raise ValueError(f"Unsupported geometry type: {geometry_type}")


def _consume_first_gpu_stage(frame) -> None:
    from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device

    compute_geometry_bounds_device(frame.geometry.values.to_owned())


def _benchmark_geojson_public_pipeline(*, rows: int, repeat: int, seed: int = 0) -> tuple[float, float] | None:
    import vibespatial.api as geopandas
    from vibespatial.runtime._runtime import has_gpu_runtime

    if not has_gpu_runtime():
        return None

    gdf = _sample_geodataframe("point", rows, seed=seed)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "sample.geojson"
        gdf.to_file(path, driver="GeoJSON")

        _consume_first_gpu_stage(geopandas.read_file(path, engine="pyogrio"))
        _consume_first_gpu_stage(geopandas.read_file(path))

        start = perf_counter()
        for _ in range(repeat):
            frame = geopandas.read_file(path, engine="pyogrio")
            _consume_first_gpu_stage(frame)
        baseline_elapsed = (perf_counter() - start) / repeat

        start = perf_counter()
        for _ in range(repeat):
            frame = geopandas.read_file(path)
            _consume_first_gpu_stage(frame)
        candidate_elapsed = (perf_counter() - start) / repeat

    baseline_rows_per_second = rows / baseline_elapsed if baseline_elapsed else float("inf")
    candidate_rows_per_second = rows / candidate_elapsed if candidate_elapsed else float("inf")
    return baseline_rows_per_second, candidate_rows_per_second


def _benchmark_shapefile_public_pipeline(*, rows: int, repeat: int, seed: int = 0) -> tuple[float, float] | None:
    import vibespatial.api as geopandas
    from vibespatial.runtime._runtime import has_gpu_runtime

    if not has_gpu_runtime():
        return None

    gdf = _sample_geodataframe("point", rows, seed=seed)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "sample.shp"
        gdf.to_file(path, driver="ESRI Shapefile")

        _consume_first_gpu_stage(geopandas.read_file(path, engine="pyogrio"))
        _consume_first_gpu_stage(geopandas.read_file(path))

        start = perf_counter()
        for _ in range(repeat):
            frame = geopandas.read_file(path, engine="pyogrio")
            _consume_first_gpu_stage(frame)
        baseline_elapsed = (perf_counter() - start) / repeat

        start = perf_counter()
        for _ in range(repeat):
            frame = geopandas.read_file(path)
            _consume_first_gpu_stage(frame)
        candidate_elapsed = (perf_counter() - start) / repeat

    baseline_rows_per_second = rows / baseline_elapsed if baseline_elapsed else float("inf")
    candidate_rows_per_second = rows / candidate_elapsed if candidate_elapsed else float("inf")
    return baseline_rows_per_second, candidate_rows_per_second


def _benchmark_pyogrio_vector_public_pipeline(
    *,
    driver: str,
    suffix: str,
    rows: int,
    repeat: int,
    candidate_engine: str | None = "pyogrio",
    seed: int = 0,
) -> tuple[float, float] | None:
    import pyogrio

    import vibespatial.api as geopandas
    from vibespatial.runtime._runtime import has_gpu_runtime

    if not has_gpu_runtime():
        return None

    gdf = _sample_geodataframe("point", rows, seed=seed)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / f"sample{suffix}"
        gdf.to_file(path, driver=driver)

        def _read_candidate():
            if candidate_engine is None:
                return geopandas.read_file(path)
            return geopandas.read_file(path, engine=candidate_engine)

        _consume_first_gpu_stage(pyogrio.read_dataframe(path))
        _consume_first_gpu_stage(_read_candidate())

        start = perf_counter()
        for _ in range(repeat):
            frame = pyogrio.read_dataframe(path)
            _consume_first_gpu_stage(frame)
        baseline_elapsed = (perf_counter() - start) / repeat

        start = perf_counter()
        for _ in range(repeat):
            frame = _read_candidate()
            _consume_first_gpu_stage(frame)
        candidate_elapsed = (perf_counter() - start) / repeat

    baseline_rows_per_second = rows / baseline_elapsed if baseline_elapsed else float("inf")
    candidate_rows_per_second = rows / candidate_elapsed if candidate_elapsed else float("inf")
    return baseline_rows_per_second, candidate_rows_per_second


def _build_covering_summary(*, scale: int, selectivity: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    row_group_count = max(125, scale // 8_000)
    rows_per_group = max(1, scale // row_group_count)
    grid_width = max(12, int(np.ceil(np.sqrt(row_group_count))))
    cell = 1_000.0 / grid_width
    centers_x = []
    centers_y = []
    for index in range(row_group_count):
        row = index // grid_width
        col = index % grid_width
        centers_x.append((col + 0.5) * cell)
        centers_y.append((row + 0.5) * cell)
    centers_x = np.asarray(centers_x, dtype=np.float64)
    centers_y = np.asarray(centers_y, dtype=np.float64)
    spread = cell * 0.45
    widths = rng.uniform(spread * 0.6, spread, row_group_count)
    heights = rng.uniform(spread * 0.6, spread, row_group_count)
    xmin = centers_x - widths
    ymin = centers_y - heights
    xmax = centers_x + widths
    ymax = centers_y + heights
    row_group_rows = np.full(row_group_count, rows_per_group, dtype=np.int64)
    target_width = 1_000.0 * np.sqrt(max(min(selectivity, 1.0), 0.0001))
    bbox = (
        500.0 - target_width / 2.0,
        500.0 - target_width / 2.0,
        500.0 + target_width / 2.0,
        500.0 + target_width / 2.0,
    )
    summary = build_geoparquet_metadata_summary(
        source="covering_bbox",
        row_group_rows=row_group_rows,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
    )
    return summary, bbox


def _benchmark_geoparquet_scan(
    *,
    geometry_type: str,
    rows: int,
    backend: str,
    repeat: int,
    geometry_encoding: str = "geoarrow",
    chunk_rows: int | None = None,
    compression: str | None = None,
    seed: int = 0,
) -> tuple[float, int]:
    gdf = _sample_geodataframe(geometry_type, rows, seed=seed)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "sample.parquet"
        gdf.to_parquet(path, geometry_encoding=geometry_encoding, compression=compression)
        file_bytes = path.stat().st_size
        for _ in range(_GEOPARQUET_BENCH_WARMUP_READS):
            read_geoparquet_owned(path, backend=backend, chunk_rows=chunk_rows)
        start = perf_counter()
        for _ in range(repeat):
            read_geoparquet_owned(path, backend=backend, chunk_rows=chunk_rows)
        elapsed = (perf_counter() - start) / repeat
    rows_per_second = rows / elapsed if elapsed else float("inf")
    return rows_per_second, file_bytes


def _benchmark_mixed_wkb_decode(*, rows: int, repeat: int, seed: int = 0) -> tuple[float, float, int, int]:
    dataset = generate_mixed_geometries(
        SyntheticSpec(
            "mixed",
            "mixed",
            count=rows,
            seed=seed,
            mix_ratios=(("point", 0.55), ("line", 0.25), ("polygon", 0.20)),
        )
    )
    owned = from_shapely_geometries(list(dataset.geometries))
    values = owned.to_wkb()
    bytes_scanned = sum(len(value) for value in values if isinstance(value, bytes))

    start = perf_counter()
    for _ in range(repeat):
        from_wkb(values)
    host_elapsed = (perf_counter() - start) / repeat

    start = perf_counter()
    for _ in range(repeat):
        decoded = decode_wkb_owned(values)
    native_elapsed = (perf_counter() - start) / repeat

    fallback_pool_share = 0.0
    if decoded.row_count:
        fallback_pool_share = 0.0
    return (
        rows / host_elapsed if host_elapsed else float("inf"),
        rows / native_elapsed if native_elapsed else float("inf"),
        bytes_scanned,
        int(round(fallback_pool_share * rows)),
    )


def _stable_geoparquet_scan_repeat(repeat: int) -> int:
    return max(repeat, _GEOPARQUET_SCAN_BENCH_MIN_REPEAT)


def _stable_shapefile_repeat(repeat: int) -> int:
    return max(repeat, _SHAPEFILE_BENCH_MIN_REPEAT)


def benchmark_io_arrow_suite(*, suite: str = "all", repeat: int = 1) -> list[IOBenchmarkCase]:
    if suite not in {"smoke", "ci", "all"}:
        raise ValueError(f"Unsupported suite: {suite}")

    point_scales = {"smoke": [10_000], "ci": [100_000], "all": [10_000, 100_000, 1_000_000]}[suite]
    polygon_scales = {"smoke": [10_000], "ci": [20_000], "all": [20_000, 100_000]}[suite]
    geoparquet_scales = {"smoke": [1_000_000], "ci": [1_000_000], "all": [1_000_000, 10_000_000]}[suite]
    results: list[IOBenchmarkCase] = []

    for rows in point_scales:
        owned = _sample_owned("point", rows)
        aligned_bytes = _estimate_owned_bytes(owned)
        encode_results = benchmark_geoarrow_bridge(
            operation="encode", geometry_type="point", rows=rows, repeat=repeat
        )
        copy = _benchmark_lookup(encode_results, "copy", attr="sharing")
        share = _benchmark_lookup(encode_results, "share", attr="sharing")
        results.append(
            _speedup_case(
                case_id=f"geoarrow-bridge-encode-point-{rows}",
                family="arrow",
                format_name="geoarrow_aligned_export",
                geometry_profile="point-heavy",
                scale=rows,
                target_floor=5.0,
                enforced=True,
                baseline_label="copy",
                candidate_label="share",
                baseline_rows_per_second=rows / copy.elapsed_seconds,
                candidate_rows_per_second=rows / share.elapsed_seconds,
                rows_decoded=rows,
                bytes_scanned=aligned_bytes,
                copies_made=0 if share.shares_memory else 1,
                notes="Aligned GeoArrow export should share buffers instead of copying.",
            )
        )

        decode_results = benchmark_geoarrow_bridge(
            operation="decode", geometry_type="point", rows=rows, repeat=repeat
        )
        copy = _benchmark_lookup(decode_results, "copy", attr="sharing")
        auto = _benchmark_lookup(decode_results, "auto", attr="sharing")
        results.append(
            _speedup_case(
                case_id=f"geoarrow-bridge-decode-point-{rows}",
                family="arrow",
                format_name="geoarrow_aligned_import",
                geometry_profile="point-heavy",
                scale=rows,
                target_floor=5.0,
                enforced=True,
                baseline_label="copy",
                candidate_label="auto",
                baseline_rows_per_second=rows / copy.elapsed_seconds,
                candidate_rows_per_second=rows / auto.elapsed_seconds,
                rows_decoded=rows,
                bytes_scanned=aligned_bytes,
                copies_made=0 if auto.shares_memory else 1,
                notes="Aligned GeoArrow import should adopt buffers zero-copy in auto mode.",
            )
        )

        native_decode = benchmark_native_geometry_codec(
            operation="decode", geometry_type="point", rows=rows, repeat=repeat
        )
        host = _benchmark_lookup(native_decode, "host_bridge")
        native = _benchmark_lookup(native_decode, "native_owned")
        results.append(
            _speedup_case(
                case_id=f"native-geometry-decode-point-{rows}",
                family="arrow",
                format_name="geoarrow_native_decode",
                geometry_profile="point-heavy",
                scale=rows,
                target_floor=4.0,
                enforced=True,
                baseline_label="host_bridge",
                candidate_label="native_owned",
                baseline_rows_per_second=host.rows_per_second,
                candidate_rows_per_second=native.rows_per_second,
                rows_decoded=rows,
                bytes_scanned=aligned_bytes,
                copies_made=0,
                notes="Family-specialized native decode should beat the host GeoArrow bridge.",
            )
        )

        native_encode = benchmark_native_geometry_codec(
            operation="encode", geometry_type="point", rows=rows, repeat=repeat
        )
        host = _benchmark_lookup(native_encode, "host_bridge")
        native = _benchmark_lookup(native_encode, "native_owned")
        results.append(
            _speedup_case(
                case_id=f"native-geometry-encode-point-{rows}",
                family="arrow",
                format_name="geoarrow_native_encode",
                geometry_profile="point-heavy",
                scale=rows,
                target_floor=4.0,
                enforced=True,
                baseline_label="host_bridge",
                candidate_label="native_owned",
                baseline_rows_per_second=host.rows_per_second,
                candidate_rows_per_second=native.rows_per_second,
                rows_decoded=rows,
                bytes_scanned=aligned_bytes,
                copies_made=0,
                notes="Family-specialized native encode should avoid the host GeoArrow bridge.",
            )
        )

        wkb_decode = benchmark_wkb_bridge(
            operation="decode", geometry_type="point", rows=rows, repeat=repeat
        )
        host = _benchmark_lookup(wkb_decode, "host_bridge")
        native = _benchmark_lookup(wkb_decode, "native_owned")
        results.append(
            _speedup_case(
                case_id=f"wkb-decode-point-{rows}",
                family="compat",
                format_name="wkb_decode",
                geometry_profile="point-heavy",
                scale=rows,
                target_floor=4.0,
                enforced=True,
                baseline_label="host_bridge",
                candidate_label="native_owned",
                baseline_rows_per_second=host.rows_per_second,
                candidate_rows_per_second=native.rows_per_second,
                rows_decoded=rows,
                bytes_scanned=sum(len(value) for value in owned.to_wkb() if isinstance(value, bytes)),
                copies_made=1,
                fallback_pool_share=native.fallback_rows / rows if rows else 0.0,
                notes="WKB decode should use staged native planning instead of per-row host decode.",
            )
        )

        wkb_encode = benchmark_wkb_bridge(
            operation="encode", geometry_type="point", rows=rows, repeat=repeat
        )
        host = _benchmark_lookup(wkb_encode, "host_bridge")
        native = _benchmark_lookup(wkb_encode, "native_owned")
        results.append(
            _speedup_case(
                case_id=f"wkb-encode-point-{rows}",
                family="compat",
                format_name="wkb_encode",
                geometry_profile="point-heavy",
                scale=rows,
                target_floor=3.0,
                enforced=True,
                baseline_label="host_bridge",
                candidate_label="native_owned",
                baseline_rows_per_second=host.rows_per_second,
                candidate_rows_per_second=native.rows_per_second,
                rows_decoded=rows,
                bytes_scanned=aligned_bytes,
                copies_made=1,
                fallback_pool_share=native.fallback_rows / rows if rows else 0.0,
                notes="WKB encode should stay in the staged native bridge for supported families.",
            )
        )

    for rows in polygon_scales:
        owned = _sample_owned("polygon", rows)
        encoded_bytes = sum(len(value) for value in owned.to_wkb() if isinstance(value, bytes))
        native_decode = benchmark_native_geometry_codec(
            operation="decode", geometry_type="polygon", rows=rows, repeat=repeat
        )
        host = _benchmark_lookup(native_decode, "host_bridge")
        native = _benchmark_lookup(native_decode, "native_owned")
        results.append(
            _speedup_case(
                case_id=f"native-geometry-decode-polygon-{rows}",
                family="arrow",
                format_name="geoarrow_native_decode",
                geometry_profile="polygon-heavy",
                scale=rows,
                target_floor=4.0,
                enforced=True,
                baseline_label="host_bridge",
                candidate_label="native_owned",
                baseline_rows_per_second=host.rows_per_second,
                candidate_rows_per_second=native.rows_per_second,
                rows_decoded=rows,
                bytes_scanned=_estimate_owned_bytes(owned),
                copies_made=0,
                notes="Polygon-native GeoArrow decode should stay on the family-specialized path.",
            )
        )

        wkb_decode = benchmark_wkb_bridge(
            operation="decode", geometry_type="polygon", rows=rows, repeat=repeat
        )
        host = _benchmark_lookup(wkb_decode, "host_bridge")
        native = _benchmark_lookup(wkb_decode, "native_owned")
        results.append(
            _speedup_case(
                case_id=f"wkb-decode-polygon-{rows}",
                family="compat",
                format_name="wkb_decode",
                geometry_profile="polygon-heavy",
                scale=rows,
                target_floor=4.0,
                enforced=True,
                baseline_label="host_bridge",
                candidate_label="native_owned",
                baseline_rows_per_second=host.rows_per_second,
                candidate_rows_per_second=native.rows_per_second,
                rows_decoded=rows,
                bytes_scanned=encoded_bytes,
                copies_made=1,
                fallback_pool_share=native.fallback_rows / rows if rows else 0.0,
                notes="Polygon WKB decode should stay on the staged native bridge.",
            )
        )

        wkb_encode = benchmark_wkb_bridge(
            operation="encode", geometry_type="polygon", rows=rows, repeat=repeat
        )
        host = _benchmark_lookup(wkb_encode, "host_bridge")
        native = _benchmark_lookup(wkb_encode, "native_owned")
        results.append(
            _speedup_case(
                case_id=f"wkb-encode-polygon-{rows}",
                family="compat",
                format_name="wkb_encode",
                geometry_profile="polygon-heavy",
                scale=rows,
                target_floor=3.0,
                enforced=True,
                baseline_label="host_bridge",
                candidate_label="native_owned",
                baseline_rows_per_second=host.rows_per_second,
                candidate_rows_per_second=native.rows_per_second,
                rows_decoded=rows,
                bytes_scanned=_estimate_owned_bytes(owned),
                copies_made=1,
                fallback_pool_share=native.fallback_rows / rows if rows else 0.0,
                notes="Polygon WKB encode should stay on the staged native bridge.",
            )
        )

    for rows in geoparquet_scales:
        scan_repeat = _stable_geoparquet_scan_repeat(repeat)
        summary, bbox = _build_covering_summary(scale=rows, selectivity=0.1)
        prune_result = select_row_groups(summary, bbox, strategy="auto")
        decoded_rows = int(round(prune_result.decoded_row_fraction * summary.total_rows))
        results.append(
            _fraction_case(
                case_id=f"geoparquet-selective-point-{rows}",
                family="parquet",
                format_name="geoparquet_selective_scan",
                geometry_profile="point-heavy",
                scale=rows,
                target_floor=0.15,
                actual_value=prune_result.decoded_row_fraction,
                enforced=True,
                rows_decoded=decoded_rows,
                notes=(
                    "Metadata-first row-group pruning should keep decoded rows at or below "
                    "15% for the canonical 10% selectivity bbox."
                ),
            )
        )

        cpu_rows_per_second, bytes_scanned = _benchmark_geoparquet_scan(
            geometry_type="point",
            rows=rows,
            backend="cpu",
            repeat=scan_repeat,
            compression=None,
        )
        if has_pylibcudf_support():
            gpu_rows_per_second, gpu_bytes = _benchmark_geoparquet_scan(
                geometry_type="point",
                rows=rows,
                backend="gpu",
                repeat=scan_repeat,
                chunk_rows=250_000,
                compression=None,
            )
            results.append(
                _speedup_case(
                    case_id=f"geoparquet-scan-point-{rows}",
                    family="parquet",
                    format_name="geoparquet_scan",
                    geometry_profile="point-heavy",
                    scale=rows,
                    target_floor=2.0,
                    enforced=True,
                    baseline_label="pyarrow_cpu",
                    candidate_label="pylibcudf_gpu",
                    baseline_rows_per_second=cpu_rows_per_second,
                    candidate_rows_per_second=gpu_rows_per_second,
                    rows_decoded=rows,
                    bytes_scanned=gpu_bytes,
                    copies_made=1,
                    notes="Supported GeoParquet scans should stay on the GPU through owned assembly.",
                )
            )
        else:
            results.append(
                _unavailable_case(
                    case_id=f"geoparquet-scan-point-{rows}",
                    family="parquet",
                    format_name="geoparquet_scan",
                    geometry_profile="point-heavy",
                    scale=rows,
                    enforced=True,
                    notes="pylibcudf is not installed, so the GPU GeoParquet scan rail is unavailable.",
                )
            )

    if suite != "smoke":
        mixed_rows = 100_000 if suite == "all" else 20_000
        host_rows_per_second, native_rows_per_second, bytes_scanned, fallback_rows = _benchmark_mixed_wkb_decode(
            rows=mixed_rows, repeat=repeat
        )
        results.append(
            _speedup_case(
                case_id=f"wkb-decode-mixed-{mixed_rows}",
                family="compat",
                format_name="wkb_decode",
                geometry_profile="mixed-family",
                scale=mixed_rows,
                target_floor=1.0,
                enforced=False,
                baseline_label="host_bridge",
                candidate_label="native_owned",
                baseline_rows_per_second=host_rows_per_second,
                candidate_rows_per_second=native_rows_per_second,
                rows_decoded=mixed_rows,
                bytes_scanned=bytes_scanned,
                copies_made=1,
                fallback_pool_share=fallback_rows / mixed_rows if mixed_rows else 0.0,
                notes="Mixed-family WKB decode is tracked for drift, but remains informational until a mixed-family floor is published.",
            )
        )

    return results


def benchmark_io_file_suite(*, suite: str = "all", repeat: int = 1) -> list[IOBenchmarkCase]:
    if suite not in {"smoke", "ci", "all"}:
        raise ValueError(f"Unsupported suite: {suite}")

    point_scales = {"smoke": [10_000], "ci": [100_000], "all": [10_000, 100_000, 1_000_000]}[suite]
    line_scales = {"smoke": [10_000], "ci": [100_000], "all": [10_000, 100_000, 1_000_000]}[suite]
    polygon_scales = {"smoke": [5_000], "ci": [50_000], "all": [5_000, 50_000, 250_000]}[suite]
    geojson_scales = {"smoke": [10_000], "ci": [10_000], "all": [10_000, 100_000]}[suite]
    shapefile_public_scales = {"smoke": [10_000], "ci": [10_000], "all": [10_000, 100_000]}[suite]
    vector_container_scales = {"smoke": [10_000], "ci": [10_000], "all": [10_000, 100_000]}[suite]
    results: list[IOBenchmarkCase] = []

    for rows in point_scales:
        benchmarks = benchmark_shapefile_ingest(
            geometry_type="point",
            rows=rows,
            repeat=_stable_shapefile_repeat(repeat),
        )
        host = _benchmark_lookup(benchmarks, "pyogrio_host")
        owned = _benchmark_lookup(benchmarks, "shapefile_owned_native")
        results.append(
            _speedup_case(
                case_id=f"shapefile-point-{rows}",
                family="file",
                format_name="shapefile_ingest",
                geometry_profile="point-heavy",
                scale=rows,
                target_floor=1.5,
                enforced=True,
                baseline_label="pyogrio_host",
                candidate_label="shapefile_owned_native",
                baseline_rows_per_second=host.rows_per_second,
                candidate_rows_per_second=owned.rows_per_second,
                rows_decoded=rows,
                bytes_scanned=None,
                copies_made=1,
                notes="Point Shapefile ingest should beat the host baseline through direct SHP GPU decode.",
            )
        )

    for rows in line_scales:
        benchmarks = benchmark_shapefile_ingest(
            geometry_type="line",
            rows=rows,
            repeat=_stable_shapefile_repeat(repeat),
        )
        host = _benchmark_lookup(benchmarks, "pyogrio_host")
        owned = _benchmark_lookup(benchmarks, "shapefile_owned_native")
        results.append(
            _speedup_case(
                case_id=f"shapefile-line-{rows}",
                family="file",
                format_name="shapefile_ingest",
                geometry_profile="line-heavy",
                scale=rows,
                target_floor=1.5,
                enforced=True,
                baseline_label="pyogrio_host",
                candidate_label="shapefile_owned_native",
                baseline_rows_per_second=host.rows_per_second,
                candidate_rows_per_second=owned.rows_per_second,
                rows_decoded=rows,
                bytes_scanned=None,
                copies_made=1,
                notes="Line-heavy Shapefile ingest should stay on the direct SHP GPU path.",
            )
        )

    for rows in polygon_scales:
        benchmarks = benchmark_shapefile_ingest(
            geometry_type="polygon",
            rows=rows,
            repeat=_stable_shapefile_repeat(repeat),
        )
        host = _benchmark_lookup(benchmarks, "pyogrio_host")
        owned = _benchmark_lookup(benchmarks, "shapefile_owned_native")
        results.append(
            _speedup_case(
                case_id=f"shapefile-polygon-{rows}",
                family="file",
                format_name="shapefile_ingest",
                geometry_profile="polygon-heavy",
                scale=rows,
                target_floor=1.1,
                enforced=True,
                baseline_label="pyogrio_host",
                candidate_label="shapefile_owned_native",
                baseline_rows_per_second=host.rows_per_second,
                candidate_rows_per_second=owned.rows_per_second,
                rows_decoded=rows,
                bytes_scanned=None,
                copies_made=1,
                notes="Polygon Shapefile ingest should clear the modest floor through direct SHP GPU decode.",
            )
        )

    for rows in geojson_scales:
        pipeline = _benchmark_geojson_public_pipeline(rows=rows, repeat=repeat)
        if pipeline is None:
            results.append(
                _unavailable_case(
                    case_id=f"geojson-point-pipeline-{rows}",
                    family="file",
                    format_name="geojson_public_pipeline",
                    geometry_profile="point-heavy",
                    scale=rows,
                    enforced=True,
                    notes=(
                        "GeoJSON public pipeline rail requires a visible GPU because it "
                        "measures read_file plus the first downstream GPU consumer."
                    ),
                )
            )
        else:
            baseline_rows_per_second, candidate_rows_per_second = pipeline
            results.append(
                _speedup_case(
                    case_id=f"geojson-point-pipeline-{rows}",
                    family="file",
                    format_name="geojson_public_pipeline",
                    geometry_profile="point-heavy",
                    scale=rows,
                    target_floor=1.0,
                    enforced=True,
                    baseline_label="pyogrio_host_plus_first_gpu_stage",
                    candidate_label="public_auto_pipeline",
                    baseline_rows_per_second=baseline_rows_per_second,
                    candidate_rows_per_second=candidate_rows_per_second,
                    rows_decoded=rows,
                    bytes_scanned=None,
                    copies_made=1,
                    notes=(
                        "GeoJSON auto-routing should reach parity or better at 10k+ on the "
                        "public read_file path once the first downstream GPU consumer is "
                        "included, so read-then-promote churn does not dominate."
                    ),
                )
            )

    for rows in shapefile_public_scales:
        pipeline = _benchmark_shapefile_public_pipeline(rows=rows, repeat=repeat)
        if pipeline is None:
            results.append(
                _unavailable_case(
                    case_id=f"shapefile-point-pipeline-{rows}",
                    family="file",
                    format_name="shapefile_public_pipeline",
                    geometry_profile="point-heavy",
                    scale=rows,
                    enforced=True,
                    notes=(
                        "Shapefile public pipeline rail requires a visible GPU because it "
                        "measures read_file plus the first downstream GPU consumer."
                    ),
                )
            )
        else:
            baseline_rows_per_second, candidate_rows_per_second = pipeline
            results.append(
                _speedup_case(
                    case_id=f"shapefile-point-pipeline-{rows}",
                    family="file",
                    format_name="shapefile_public_pipeline",
                    geometry_profile="point-heavy",
                    scale=rows,
                    target_floor=1.0,
                    enforced=True,
                    baseline_label="pyogrio_host_plus_first_gpu_stage",
                    candidate_label="public_auto_pipeline",
                    baseline_rows_per_second=baseline_rows_per_second,
                    candidate_rows_per_second=candidate_rows_per_second,
                    rows_decoded=rows,
                    bytes_scanned=None,
                    copies_made=1,
                    notes=(
                        "Shapefile auto-routing should reach parity or better at 10k+ on the "
                        "public read_file path once the first downstream GPU consumer is "
                        "included."
                    ),
                )
            )

    container_cases = (
        ("GPKG", ".gpkg", "geopackage_public_pipeline", "pyogrio", "public_engine_pyogrio_native_boundary"),
        ("FlatGeobuf", ".fgb", "flatgeobuf_public_pipeline", None, "public_auto_pipeline"),
    )
    for rows in vector_container_scales:
        for driver, suffix, format_name, candidate_engine, candidate_label in container_cases:
            pipeline = _benchmark_pyogrio_vector_public_pipeline(
                driver=driver,
                suffix=suffix,
                rows=rows,
                repeat=repeat,
                candidate_engine=candidate_engine,
            )
            if pipeline is None:
                results.append(
                    _unavailable_case(
                        case_id=f"{format_name}-{rows}",
                        family="file",
                        format_name=format_name,
                        geometry_profile="point-heavy",
                        scale=rows,
                        enforced=True,
                        notes=(
                            f"{driver} public pipeline rail requires a visible GPU because "
                            "it measures read_file plus the first downstream GPU consumer."
                        ),
                    )
                )
            else:
                baseline_rows_per_second, candidate_rows_per_second = pipeline
                results.append(
                    _speedup_case(
                        case_id=f"{format_name}-{rows}",
                        family="file",
                        format_name=format_name,
                        geometry_profile="point-heavy",
                        scale=rows,
                        target_floor=1.0,
                        enforced=True,
                        baseline_label="pyogrio_host_plus_first_gpu_stage",
                        candidate_label=candidate_label,
                        baseline_rows_per_second=baseline_rows_per_second,
                        candidate_rows_per_second=candidate_rows_per_second,
                        rows_decoded=rows,
                        bytes_scanned=None,
                        copies_made=1,
                        notes=(
                            f"{driver} explicit pyogrio reads should stay at parity or better "
                            "on the public read_file path once the first downstream GPU "
                            "consumer is included."
                            if candidate_engine == "pyogrio"
                            else (
                                f"{driver} auto-routing should stay at parity or better on "
                                "the public read_file path once the first downstream GPU "
                                "consumer is included."
                            )
                        ),
                    )
                )

    return results


def io_suite_to_json(
    results: list[IOBenchmarkCase],
    *,
    suite: str,
    repeat: int,
    scope: str,
) -> str:
    passed = sum(1 for result in results if result.status == "pass")
    failed = sum(1 for result in results if result.status == "fail")
    informational = sum(1 for result in results if result.status == "informational")
    unavailable = sum(1 for result in results if result.status == "unavailable")
    payload = {
        "metadata": {
            "suite": suite,
            "repeat": repeat,
            "scope": scope,
            "statuses": {
                "pass": passed,
                "fail": failed,
                "informational": informational,
                "unavailable": unavailable,
            },
            "notes": [
                "10M-scale coverage remains a manual deep-run path for the cheapest point-heavy cases.",
                "copies_made is a conservative path-level estimate, not a byte-for-byte allocator trace.",
                "GeoJSON rails now target the public read_file pipeline with the first downstream GPU consumer, not isolated parser throughput.",
            ],
        },
        "results": [asdict(result) for result in results],
    }
    return json.dumps(payload, indent=2)
