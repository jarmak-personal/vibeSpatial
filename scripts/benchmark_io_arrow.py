from __future__ import annotations

import argparse
import json
import platform
import struct
from pathlib import Path
from time import perf_counter

import numpy as np

from vibespatial.bench.catalog import ensure_operations_loaded
from vibespatial.bench.runner import run_operation
from vibespatial.bench.schema import BenchmarkResult

_PUBLIC_FORMATS = ("parquet", "geojson", "shapefile", "gpkg")
_SUITE_SCALES = {
    "smoke": (10_000,),
    "ci": (100_000,),
    "all": (10_000, 100_000, 1_000_000),
}
_WKB_STEADY_STATE_SCALE = 100_000


def _wkb_prefix(order: int) -> str:
    return "<" if order == 1 else ">"


def _wkb_header(order: int, type_id: int) -> bytes:
    return bytes((order,)) + struct.pack(f"{_wkb_prefix(order)}I", type_id)


def _wkb_point(order: int, x: float, y: float) -> bytes:
    return _wkb_header(order, 1) + struct.pack(f"{_wkb_prefix(order)}dd", x, y)


def _wkb_line(order: int, coords: tuple[tuple[float, float], ...]) -> bytes:
    prefix = _wkb_prefix(order)
    return (
        _wkb_header(order, 2)
        + struct.pack(f"{prefix}I", len(coords))
        + b"".join(struct.pack(f"{prefix}dd", x, y) for x, y in coords)
    )


def _wkb_polygon(order: int, rings: tuple[tuple[tuple[float, float], ...], ...]) -> bytes:
    prefix = _wkb_prefix(order)
    return (
        _wkb_header(order, 3)
        + struct.pack(f"{prefix}I", len(rings))
        + b"".join(
            struct.pack(f"{prefix}I", len(ring))
            + b"".join(struct.pack(f"{prefix}dd", x, y) for x, y in ring)
            for ring in rings
        )
    )


def _wkb_multi(order: int, type_id: int, children: tuple[bytes, ...]) -> bytes:
    return (
        _wkb_header(order, type_id)
        + struct.pack(f"{_wkb_prefix(order)}I", len(children))
        + b"".join(children)
    )


def _wkb_benchmark_record(
    family: str, order: int, *, mixed: bool = False
) -> tuple[bytes, int, int, int]:
    line_a = tuple((float(index), float(index % 3)) for index in range(8))
    line_b = tuple((float(index), float(5 + index % 4)) for index in range(8))
    ring_a = (
        (0.0, 0.0),
        (4.0, 0.0),
        (4.0, 4.0),
        (0.0, 4.0),
        (0.0, 0.0),
    )
    ring_b = (
        (10.0, 10.0),
        (13.0, 10.0),
        (13.0, 13.0),
        (10.0, 13.0),
        (10.0, 10.0),
    )
    if family == "point":
        return _wkb_point(order, 1.25, -2.5), 1, 0, 0
    if family == "linestring":
        return _wkb_line(order, line_a), len(line_a), 0, 0
    if family == "polygon":
        return _wkb_polygon(order, (ring_a,)), len(ring_a), 0, 1

    child_orders = (order, 1 - order) if mixed else (order, order)
    if family == "multipoint":
        return (
            _wkb_multi(
                order,
                4,
                (
                    _wkb_point(child_orders[0], 1.0, 2.0),
                    _wkb_point(child_orders[1], 3.0, 4.0),
                ),
            ),
            2,
            2,
            0,
        )
    if family == "multilinestring":
        return (
            _wkb_multi(
                order,
                5,
                (_wkb_line(child_orders[0], line_a), _wkb_line(child_orders[1], line_b)),
            ),
            len(line_a) + len(line_b),
            2,
            0,
        )
    if family == "multipolygon":
        return (
            _wkb_multi(
                order,
                6,
                (
                    _wkb_polygon(child_orders[0], (ring_a,)),
                    _wkb_polygon(child_orders[1], (ring_b,)),
                ),
            ),
            len(ring_a) + len(ring_b),
            2,
            2,
        )
    raise ValueError(f"unsupported WKB benchmark family: {family}")


def _device_owned_bytes(owned) -> int:
    state = owned.device_state
    if state is None:
        return 0
    total = sum(
        int(value.nbytes) for value in (state.validity, state.tags, state.family_row_offsets)
    )
    for buffer in state.families.values():
        for name in ("x", "y", "geometry_offsets", "empty_mask", "part_offsets", "ring_offsets"):
            value = getattr(buffer, name, None)
            if value is not None:
                total += int(value.nbytes)
    return total


def _median(samples: list[float]) -> float:
    return float(np.median(np.asarray(samples, dtype=np.float64)))


def _wkb_endian_results(
    *, scale: int, repeat: int
) -> tuple[list[dict[str, object]], dict[str, object]]:
    import pyarrow
    import shapely

    from vibespatial import has_gpu_runtime

    metadata: dict[str, object] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pyarrow": pyarrow.__version__,
        "shapely": shapely.__version__,
    }
    if not has_gpu_runtime():
        return [], {**metadata, "status": "unavailable", "reason": "CUDA runtime unavailable"}

    import cupy as cp
    import pylibcudf

    from vibespatial.bench.wkb_codec import decode_device_wkb, scan_device_wkb
    from vibespatial.cuda._runtime import get_d2h_transfer_events, reset_d2h_transfer_count

    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    gpu_name = props["name"]
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()
    metadata.update(
        {
            "status": "available",
            "gpu": gpu_name,
            "cupy": cp.__version__,
            "pylibcudf": getattr(pylibcudf, "__version__", "unknown"),
            "scale": scale,
            "repeat": repeat,
            "timing": "warm synchronized repeat median",
            "comparator": "Shapely from_wkb on the identical immutable in-memory records",
        }
    )
    families = (
        "point",
        "linestring",
        "polygon",
        "multipoint",
        "multilinestring",
        "multipolygon",
    )
    profiles = [(family, order, False) for family in families for order in (1, 0)]
    profiles.extend((family, 1, True) for family in families[3:])
    results: list[dict[str, object]] = []
    for family, order, mixed in profiles:
        record, coords_per_row, parts_per_row, rings_per_row = _wkb_benchmark_record(
            family,
            order,
            mixed=mixed,
        )
        payload = cp.asarray(np.frombuffer(record * scale, dtype=np.uint8))
        offsets = cp.arange(scale + 1, dtype=cp.int64) * len(record)
        host_records = np.empty(scale, dtype=object)
        host_records[:] = [record]

        scan_device_wkb(payload, offsets, scale)
        decoded = decode_device_wkb(payload, offsets, scale)
        shapely.from_wkb(host_records)
        cp.cuda.get_current_stream().synchronize()

        scan_samples: list[float] = []
        for _ in range(repeat):
            started = perf_counter()
            scan_device_wkb(payload, offsets, scale)
            cp.cuda.get_current_stream().synchronize()
            scan_samples.append(perf_counter() - started)

        reset_d2h_transfer_count()
        get_d2h_transfer_events(clear=True)
        total_samples: list[float] = []
        for _ in range(repeat):
            started = perf_counter()
            decoded = decode_device_wkb(payload, offsets, scale)
            cp.cuda.get_current_stream().synchronize()
            total_samples.append(perf_counter() - started)
        d2h_events = get_d2h_transfer_events(clear=True)

        host_samples: list[float] = []
        for _ in range(repeat):
            started = perf_counter()
            shapely.from_wkb(host_records)
            host_samples.append(perf_counter() - started)

        total_seconds = _median(total_samples)
        host_seconds = _median(host_samples)
        payload_bytes = len(record) * scale
        results.append(
            {
                "case_id": f"wkb-{family}-{'mixed' if mixed else ('le' if order else 'be')}-{scale}",
                "family": family,
                "endian": "mixed" if mixed else ("little" if order else "big"),
                "rows": scale,
                "payload_bytes": payload_bytes,
                "coordinate_count": coords_per_row * scale,
                "part_count": parts_per_row * scale,
                "ring_count": rings_per_row * scale,
                "structural_scan_seconds": _median(scan_samples),
                "total_decode_seconds": total_seconds,
                "host_comparator_seconds": host_seconds,
                "rows_per_second": scale / total_seconds,
                "payload_gb_per_second": payload_bytes / total_seconds / 1e9,
                "coordinates_per_second": coords_per_row * scale / total_seconds,
                "host_speedup": host_seconds / total_seconds,
                "output_bytes": _device_owned_bytes(decoded),
                "d2h_bytes": sum(event.bytes_transferred for event in d2h_events),
                "d2h_reasons": sorted({event.reason for event in d2h_events}),
                "fallback_rows": 0,
                "host_speedup_floor": 4.0,
                "big_to_little_floor": 0.8,
                "enforced": scale >= _WKB_STEADY_STATE_SCALE,
                "status": (
                    "pass"
                    if host_seconds / total_seconds >= 4.0
                    else ("fail" if scale >= _WKB_STEADY_STATE_SCALE else "informational")
                ),
            }
        )

    by_family_endian = {(item["family"], item["endian"]): item for item in results}
    for item in results:
        if item["endian"] == "big":
            little = by_family_endian[(item["family"], "little")]
            ratio = float(item["rows_per_second"]) / float(little["rows_per_second"])
            item["big_to_little_throughput"] = ratio
            if ratio < 0.8:
                item["status"] = (
                    "fail" if scale >= _WKB_STEADY_STATE_SCALE else "informational"
                )
    return results, metadata


def _status_counts(results: list[BenchmarkResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return counts


def _suite_payload(*, suite: str, repeat: int) -> dict[str, object]:
    ensure_operations_loaded()
    results: list[BenchmarkResult] = []
    for input_format in _PUBLIC_FORMATS:
        for scale in _SUITE_SCALES[suite]:
            results.append(
                run_operation(
                    "io-arrow",
                    scale=scale,
                    repeat=repeat,
                    compare="shapely",
                    input_format=input_format,
                )
            )
    codec_results, codec_metadata = _wkb_endian_results(
        scale=_WKB_STEADY_STATE_SCALE,
        repeat=max(3, repeat),
    )
    return {
        "schema_version": 2,
        "suite": suite,
        "metadata": {
            "scope": "io-arrow",
            "repeat": repeat,
            "public_api": True,
            "statuses": _status_counts(results),
            "notes": [
                "Runs the registered public io-arrow operation only.",
                "WKB endian codec results are synchronized device microbenchmarks with an identical Shapely comparator.",
                "Other internal GeoArrow/WKB/GeoParquet component rails live in tests/test_io_benchmark_rails.py.",
            ],
        },
        "results": [result.to_dict() for result in results],
        "wkb_endian_metadata": codec_metadata,
        "wkb_endian_results": codec_results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark public Arrow/WKB IO APIs.")
    parser.add_argument("--suite", choices=tuple(_SUITE_SCALES), default=None)
    parser.add_argument("--format", choices=_PUBLIC_FORMATS, default="parquet")
    parser.add_argument("--scale", type=int, default=100_000)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--wkb-endian",
        action="store_true",
        help="run only the family/endian device WKB codec benchmark",
    )
    args = parser.parse_args(argv)

    if args.wkb_endian:
        repeat = 3 if args.repeat is None else max(1, args.repeat)
        results, metadata = _wkb_endian_results(scale=args.scale, repeat=repeat)
        payload = {
            "schema_version": 1,
            "metadata": metadata,
            "wkb_endian_results": results,
        }
        text = json.dumps(payload, indent=2)
        if args.json_out is not None:
            args.json_out.write_text(text, encoding="utf-8")
        print(text)
        return 0

    if args.suite is not None:
        repeat = 1 if args.repeat is None else args.repeat
        payload = _suite_payload(suite=args.suite, repeat=repeat)
        text = json.dumps(payload, indent=2)
        if args.json_out is not None:
            args.json_out.write_text(text, encoding="utf-8")
        print(text)
        return 0

    repeat = 3 if args.repeat is None else args.repeat
    ensure_operations_loaded()
    result = run_operation(
        "io-arrow",
        scale=args.scale,
        repeat=repeat,
        compare="shapely",
        input_format=args.format,
    )
    print(result.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
