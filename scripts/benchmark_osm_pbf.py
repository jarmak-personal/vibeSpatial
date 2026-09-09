"""Same-data OSM PBF / native GeoArrow GeoParquet ingestion comparison.

Comparators are immutable identity-checked artifacts. Each timed backend runs
in its own process; conversion and correctness fingerprints are outside timing.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

LAYERS = ("points", "lines", "multilinestrings", "multipolygons", "other_relations")

def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(payload):
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import shapely

    digest = hashlib.sha256()
    attrs = payload.attributes.to_arrow(index=False).combine_chunks()
    schema = pa.schema([pa.field(field.name, pa.string() if pa.types.is_large_string(field.type) else field.type) for field in attrs.schema])
    attrs = attrs.cast(schema)
    digest.update(str(attrs.schema.remove_metadata()).encode())
    for name in attrs.column_names:
        values = attrs[name].chunk(0)
        digest.update(values.is_valid().to_numpy(zero_copy_only=False).tobytes())
        fill = "" if str(values.type) == "string" else 0
        values = pc.fill_null(values, fill)
        for buffer in values.buffers()[1:]:
            if buffer is not None:
                digest.update(buffer)
    geometries = np.asarray(payload.geometry.to_geoseries(index=None, name="geometry"))
    # Compare polygon topology with canonical ring starts/orientation. Invalid
    # rings are preserved: GEOS topological equals can reject identical inputs.
    # Other families retain coordinate and collection member order exactly.
    for begin in range(0, len(geometries), 100_000):
        chunk = geometries[begin:begin+100_000].copy()
        polygons = np.isin(shapely.get_type_id(chunk), [3, 6])
        chunk[polygons] = shapely.normalize(chunk[polygons])
        digest.update(b"".join(shapely.to_wkb(chunk, byte_order=1)))
    return digest.hexdigest()


def _identity(path, repeat):
    packages = {}
    for name in ("pyogrio", "pyarrow", "shapely", "numpy", "pandas", "pylibcudf-cu12", "pylibcudf-cu13", "nvidia-libnvcomp-cu12", "nvidia-libnvcomp-cu13"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "source_sha256": _sha(path), "source_bytes": path.stat().st_size,
        "host": platform.node(), "platform": platform.platform(),
        "cpu": {row["field"]: row["data"] for row in json.loads(subprocess.check_output(["lscpu", "-J"], text=True))["lscpu"] if row["field"] in ("Architecture:", "CPU(s):", "Model name:", "Thread(s) per core:", "Core(s) per socket:", "Socket(s):")},
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip(),
        "packages": packages, "repeat": repeat, "statistic": "median",
        "measurement": "filesystem-cache-warm; imports excluded; CUDA synchronized; native payload and public frame boundaries",
        "encoding": "GeoParquet 1.1 native GeoArrow, snappy, row groups 1M; GeometryCollection has no native encoding",
        "benchmark_sha256": _sha(__file__),
    }


def _worker(args):
    if args.worker == "native" and args.pool_limit:
        os.environ["VIBESPATIAL_GPU_POOL_LIMIT"] = str(args.pool_limit)
    import cupy as cp
    import rmm.statistics as rstats

    from vibespatial.cuda._runtime import get_cuda_runtime
    from vibespatial.io.file import _read_osm_pbf_pyogrio_layer_native
    from vibespatial.io.geoparquet import read_geoparquet_native
    from vibespatial.io.osm_pbf_native import read_osm_pbf_native
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    runtime = get_cuda_runtime()
    runtime._ensure_context()
    def read():
        if args.worker == "gdal":
            return _read_osm_pbf_pyogrio_layer_native(args.path, layer=args.layer)
        if args.worker == "parquet":
            return read_geoparquet_native(args.cache / f"{args.layer}.parquet", backend="gpu")
        return read_osm_pbf_native(args.path, layer=args.layer)
    result = {"pool_limit_bytes": args.pool_limit if args.worker == "native" else None}
    fingerprint = None
    for boundary in ("native", "public"):
        samples, peaks, final_bytes, paging = [], [], [], []
        for iteration in range(args.repeat+1):
            gc.collect()
            clear_dispatch_events()
            rstats.push_statistics()
            start = perf_counter()
            payload = read()
            public = payload.to_geodataframe() if boundary == "public" else None
            cp.cuda.get_current_stream().synchronize()
            elapsed = perf_counter()-start
            counters = rstats.pop_statistics()
            if iteration == 0:
                result[f"{boundary}_first_seconds"] = elapsed
                if fingerprint is None:
                    fingerprint = _fingerprint(payload)
                    result["rows"] = payload.geometry.row_count
            else:
                samples.append(elapsed)
                peaks.append(counters.peak_bytes)
                final_bytes.append(counters.current_bytes)
                paging.append(sum("cuda_managed" in event.implementation for event in get_dispatch_events()))
            del public, payload
        result[boundary] = {"seconds": statistics.median(samples), "samples": samples,
                            "peak_device_bytes": max(peaks), "live_device_bytes": max(final_bytes),
                            "managed_workspace_events": paging}
    result["fingerprint"] = fingerprint
    result["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    args.output.write_text(json.dumps(result, indent=2)+"\n")


def _run_worker(args, backend, layer):
    output = args.cache / f"{backend}-{layer}.json"
    command = [sys.executable, __file__, str(args.path), "--cache", str(args.cache),
               "--repeat", str(args.repeat), "--worker", backend, "--layer", layer, "--output", str(output)]
    if backend == "native" and args.pool_limit:
        command += ["--pool-limit", str(args.pool_limit)]
    subprocess.run(command, check=True)
    return json.loads(output.read_text())


def _prepare_comparator(args, identity):
    from vibespatial.io.file import _read_osm_pbf_pyogrio_layer_native
    from vibespatial.io.geoparquet import read_geoparquet_native, write_geoparquet

    checkpoint = args.cache / "preparation.json"
    result = (json.loads(checkpoint.read_text()) if checkpoint.exists() else
              {"identity": identity, "created": datetime.now(UTC).isoformat(), "layers": {}})
    if result["identity"] != identity:
        raise ValueError("Incomplete comparator identity changed; select a new --cache directory")
    for layer in LAYERS:
        if layer in result["layers"]:
            continue
        print(f"Preparing {layer} comparator", flush=True)
        if layer == "other_relations":
            gdal = _run_worker(args, "gdal", layer)
            result["layers"][layer] = {"fingerprint": gdal["fingerprint"], "parquet_sha256": None,
                                        "gdal": gdal, "parquet": None,
                                        "parquet_note": "GeoParquet 1.1 has no native GeoArrow GeometryCollection encoding"}
            checkpoint.write_text(json.dumps(result, indent=2)+"\n")
            continue
        payload = _read_osm_pbf_pyogrio_layer_native(args.path, layer=layer)
        fingerprint = _fingerprint(payload)
        path = args.cache / f"{layer}.parquet"
        write_geoparquet(payload, path, index=False, geometry_encoding="geoarrow",
                         schema_version="1.1.0", compression="snappy", row_group_size=1_000_000)
        del payload
        parquet = read_geoparquet_native(path, backend="gpu")
        if _fingerprint(parquet) != fingerprint:
            raise ValueError(f"{layer}: native GeoParquet conversion changed the GDAL oracle")
        del parquet
        gc.collect()
        result["layers"][layer] = {"fingerprint": fingerprint, "parquet_sha256": _sha(path),
                                     "gdal": _run_worker(args, "gdal", layer),
                                     "parquet": _run_worker(args, "parquet", layer)}
        checkpoint.write_text(json.dumps(result, indent=2)+"\n")
    (args.cache / "baseline.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--worker", choices=("gdal", "parquet", "native"), help=argparse.SUPPRESS)
    parser.add_argument("--layer", choices=LAYERS, help=argparse.SUPPRESS)
    parser.add_argument("--pool-limit", type=int, help="Candidate RMM device pool limit in bytes; comparators stay unchanged")
    args = parser.parse_args()
    args.path = args.path.resolve()
    args.cache = args.cache.resolve()
    args.cache.mkdir(parents=True, exist_ok=True)
    if args.repeat < 1:
        parser.error("repeat must be positive")
    if args.worker:
        _worker(args)
        return
    identity = _identity(args.path, args.repeat)
    baseline_path = args.cache / "baseline.json"
    reused = baseline_path.exists()
    if reused:
        baseline = json.loads(baseline_path.read_text())
        if baseline["identity"] != identity:
            raise ValueError("Comparator identity changed; select a new --cache directory")
        for layer, packet in baseline["layers"].items():
            if packet["parquet_sha256"] is not None and _sha(args.cache / f"{layer}.parquet") != packet["parquet_sha256"]:
                raise ValueError("Cached GeoParquet oracle has changed")
    else:
        baseline = _prepare_comparator(args, identity)
    current = {}
    for layer in LAYERS:
        current[layer] = _run_worker(args, "native", layer)
        if current[layer]["fingerprint"] != baseline["layers"][layer]["fingerprint"]:
            raise ValueError(f"{layer}: candidate differs from same-data GDAL oracle")
    result = {"identity": identity, "baseline_path": str(baseline_path), "baseline_reused": reused,
              "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "candidate_sources": {str(p): _sha(p) for p in sorted(Path("src/vibespatial/io").glob("osm_pbf*.py"))},
              "baseline": baseline["layers"], "current": current}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    for layer in LAYERS:
        print(layer, {backend: round((current[layer] if backend == "native" else baseline["layers"][layer][backend])["public"]["seconds"], 4)
                      for backend in ("gdal", "native", "parquet") if backend == "native" or baseline["layers"][layer][backend] is not None}, flush=True)


if __name__ == "__main__":
    main()
