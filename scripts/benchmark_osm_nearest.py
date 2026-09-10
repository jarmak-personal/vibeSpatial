"""Same-data public nearest comparison: OSM building points to roads/segments.

CPU preparation/oracles are deliberately outside the candidate GPU path. Every
backend runs in a separate process. Public NumPy result export is timed; parity
and optional cProfile collection are not. Comparator artifacts are immutable.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
CRS = "EPSG:26986"
SEED = 20260910
FILTERS = {"lines": "highway IS NOT NULL", "multipolygons": "building IS NOT NULL AND building <> 'no'"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def segment_lines(lines):
    """Split consecutive coordinates, retaining parent rows and zero-length edges."""
    import numpy as np
    import shapely

    coords, parents = shapely.get_coordinates(lines, return_index=True)
    starts = np.flatnonzero(parents[:-1] == parents[1:])
    segments = shapely.linestrings(np.stack((coords[starts], coords[starts + 1]), axis=1))
    return segments, parents[starts]


def canonical(indices, distances):
    import numpy as np

    indices, distances = np.asarray(indices, dtype=np.int64), np.asarray(distances, dtype=np.float64)
    if indices.ndim != 2 or indices.shape[0] != 2 or distances.shape != (indices.shape[1],):
        raise ValueError("nearest output must have shape (2, n) with n distances")
    order = np.lexsort((indices[1], indices[0]))
    return indices[:, order], distances[order]


def compare(candidate, oracle):
    """All-tie pair identity is exact; metric comparison permits 1 micrometre."""
    import numpy as np

    ci, cd = canonical(candidate[0], candidate[1])
    oi, od = canonical(oracle[0], oracle[1])
    same_pairs = np.array_equal(ci, oi)
    valid_distances = bool(np.all(np.isfinite(cd) & (cd >= 0)) and np.all(np.isfinite(od) & (od >= 0)))
    same_distances = valid_distances and same_pairs and np.allclose(cd, od, rtol=1e-10, atol=1e-6)
    return {"passed": bool(same_distances), "pairs_equal": same_pairs,
            "finite_nonnegative_distances": valid_distances,
            "candidate_pairs": len(cd), "oracle_pairs": len(od),
            "max_distance_error_m": float(np.max(np.abs(cd - od), initial=0)) if same_pairs and valid_distances else None,
            "rtol": 1e-10, "atol_m": 1e-6}


def prepare(args):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyogrio
    import pyproj
    import shapely

    identity = {"source_sha256": sha(args.path), "source_bytes": args.path.stat().st_size,
                "crs": CRS, "filters": FILTERS, "seed": SEED,
                "script_sha256": sha(__file__),
                "packages": {n: importlib.metadata.version(n) for n in ("pyogrio", "pyproj", "shapely", "numpy", "pyarrow")}}
    manifest_path = args.cache / "fixture.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"] != identity:
            raise ValueError("Fixture identity changed; use a new --cache directory")
        for name, packet in manifest["files"].items():
            if sha(args.cache / f"{name}.parquet") != packet["sha256"]:
                raise ValueError(f"Fixture changed: {name}")
        return manifest
    start = perf_counter()
    transform = pyproj.Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    selected = {}
    stages = {}
    for layer, where in FILTERS.items():
        stage = perf_counter()
        metadata, table = pyogrio.read_arrow(args.path, layer=layer, where=where, columns=["osm_id"])
        geometries = shapely.from_wkb(table[metadata["geometry_name"] or "wkb_geometry"].to_numpy())
        valid = ~shapely.is_missing(geometries) & ~shapely.is_empty(geometries)
        geometries = shapely.transform(geometries[valid], transform.transform, interleaved=False)
        selected[layer] = geometries
        stages[f"read_filter_project_{layer}_seconds"] = perf_counter() - stage
    roads = selected.pop("lines")
    if not np.all(shapely.get_type_id(roads) == 1):
        raise ValueError("OSM lines fixture must contain only LineStrings")
    stage = perf_counter()
    points = shapely.point_on_surface(selected.pop("multipolygons"))
    if not np.all(np.isfinite(shapely.get_coordinates(points))):
        raise ValueError("Nonfinite building representative points")
    stages["representative_points_seconds"] = perf_counter() - stage
    # Random sampling across the whole state; every scale is a nested prefix.
    points = points[np.random.default_rng(SEED).permutation(len(points))]
    stage = perf_counter()
    segments, parents = segment_lines(roads)
    stages["segment_roads_seconds"] = perf_counter() - stage
    files = {}
    for name, geometries, parent in (("roads", roads, np.arange(len(roads))),
                                     ("segments", segments, parents),
                                     ("points", points, np.arange(len(points)))):
        table = pa.table({"wkb": pa.array(shapely.to_wkb(geometries, byte_order=1), type=pa.binary()),
                          "parent_row": parent})
        path = args.cache / f"{name}.parquet"
        pq.write_table(table, path, compression="snappy")
        files[name] = {"rows": len(geometries), "coordinates": int(shapely.count_coordinates(geometries)),
                       "sha256": sha(path)}
    manifest = {"identity": identity, "created": datetime.now(UTC).isoformat(), "files": files,
                "preparation_seconds": perf_counter() - start, "stages": stages,
                "contract": "All highway-tagged line ways (including paths); building-tagged areas except no. "
                            "Point-on-surface after projection. No topology repair. Parent rows reference filtered roads. "
                            "Two-vertex segments retain zero-length edges. No segment-to-road deduplication in timing."}
    write_json(manifest_path, manifest)
    return manifest


def measurement_identity(args, manifest):
    names = ("shapely", "numpy", "pyarrow", "pyogrio", "pyproj")
    cpu = json.loads(subprocess.check_output(["lscpu", "-J"], text=True))["lscpu"]
    return {"fixture": manifest["identity"], "files": manifest["files"], "python": sys.version,
            "packages": {name: importlib.metadata.version(name) for name in names},
            "geos": __import__("shapely").geos_version_string,
            "lock_sha256": sha(ROOT / "uv.lock"), "host": platform.node(), "platform": platform.platform(),
            "cpu": {row["field"]: row["data"] for row in cpu if row["field"] in
                    ("Architecture:", "CPU(s):", "Model name:", "Thread(s) per core:", "Core(s) per socket:", "Socket(s):")},
            "storage": str(args.cache), "repeat": args.repeat, "timeout_seconds": args.timeout,
            "scales": args.scales, "layouts": args.layouts,
            "measurement": "Imports and file reads excluded. One cold-process trial, then repeat fresh-input trials; "
                           "each times WKB ingress, sindex accessor, first query, one repeated query on same index. "
                           "CUDA synchronized before/after stages. Public pair/distance export included. "
                           "Median of warm-process trials. No max_distance; all ties; k=1.",
            "script_sha256": sha(__file__)}


def worker(args):
    import numpy as np
    import pyarrow.parquet as pq
    import shapely

    tree_wkb = pq.read_table(args.cache / f"{args.layout}.parquet", columns=["wkb"])["wkb"].to_numpy()
    query_wkb = pq.read_table(args.cache / "points.parquet", columns=["wkb"])["wkb"].to_numpy()
    if args.rows:
        query_wkb = query_wkb[:args.rows]
    is_vs = args.worker == "vibespatial"
    if is_vs:
        import cupy as cp

        import vibespatial as gpd
        from vibespatial.cuda._runtime import (
            get_cuda_runtime,
            get_d2h_transfer_events,
            get_d2h_transfer_profile,
            reset_d2h_transfer_count,
        )
        from vibespatial.runtime.dispatch import get_dispatch_events
        from vibespatial.runtime.fallbacks import get_fallback_events
        from vibespatial.runtime.materialization import get_materialization_events

        gpd.set_execution_mode(args.mode)
        get_cuda_runtime()._ensure_context()
        synchronize = cp.cuda.runtime.deviceSynchronize
    else:
        def synchronize():
            pass
    result = {"backend": args.worker, "mode": args.mode if is_vs else "cpu", "tree_rows": len(tree_wkb), "query_rows": len(query_wkb),
              "status": "running", "trials": []}
    write_json(args.output, result)
    for iteration in range(args.repeat + 1):
        gc.collect()
        stages = {}
        result["active_trial"] = stages
        for stage in ("ingress", "index_access", "first_query", "warm_query"):
            result["active_stage"] = stage
            result["active_iteration"] = iteration
            write_json(args.output, result)
            if is_vs:
                get_dispatch_events(clear=True)
                get_fallback_events(clear=True)
                get_materialization_events(clear=True)
                reset_d2h_transfer_count()
            synchronize()
            start = perf_counter()
            if stage == "ingress":
                if is_vs:
                    tree_values = gpd.GeoSeries.from_wkb(tree_wkb, crs=CRS)
                    query_values = gpd.GeoSeries.from_wkb(query_wkb, crs=CRS)
                else:
                    tree_values, query_values = shapely.from_wkb(tree_wkb), shapely.from_wkb(query_wkb)
            elif stage == "index_access":
                tree = tree_values.sindex if is_vs else shapely.STRtree(tree_values)
            else:
                if is_vs:
                    output = tree.nearest(query_values, return_all=True, return_distance=True)
                else:
                    output = tree.query_nearest(query_values, all_matches=True, return_distance=True)
            synchronize()
            seconds = perf_counter() - start
            packet = {"seconds": seconds}
            if is_vs:
                count, size, transfer_seconds = get_d2h_transfer_profile()
                packet.update({"dispatch": [e.to_dict() for e in get_dispatch_events()],
                               "fallbacks": [e.to_dict() for e in get_fallback_events()],
                               "materializations": [e.to_dict() for e in get_materialization_events()],
                               "d2h_count": count, "d2h_bytes": size, "d2h_seconds": transfer_seconds,
                               "d2h_reasons": sorted({e.reason for e in get_d2h_transfer_events()}),
                               "event_detail_limit": 512})
            stages[stage] = packet
            # Validate every first and repeated output, outside measurement.
            if stage in ("first_query", "warm_query"):
                indices, distances = canonical(*output)
                if args.oracle.exists():
                    with np.load(args.oracle) as expected:
                        parity = compare((indices, distances), (expected["indices"], expected["distances"]))
                elif is_vs:
                    raise ValueError("Candidate requires a saved same-data Shapely oracle")
                else:
                    np.savez(args.oracle, indices=indices, distances=distances)
                    parity = compare((indices, distances), (indices, distances))
                stages[stage]["parity"] = parity
                if not parity["passed"]:
                    np.savez(args.output.with_suffix(".mismatch.npz"), indices=indices, distances=distances)
                del output
            result["active_trial"] = stages
            write_json(args.output, result)
        stages["input_types"] = [type(tree_values.array).__name__, type(query_values.array).__name__] if is_vs else ["ndarray", "ndarray"]
        result["trials"].append(stages)
        if iteration == args.repeat and is_vs and args.profile:
            import cProfile

            result["active_stage"] = "profile"
            write_json(args.output, result)
            profiler = cProfile.Profile()
            profiler.enable()
            output = tree.nearest(query_values, return_all=True, return_distance=True)
            synchronize()
            profiler.disable()
            profiler.dump_stats(str(args.output.with_suffix(".prof")))
            del output
        del tree, tree_values, query_values
    result.pop("active_stage")
    result.pop("active_iteration")
    result.pop("active_trial")
    result["status"] = "ok" if all(t[s]["parity"]["passed"] for t in result["trials"] for s in ("first_query", "warm_query")) else "parity_failed"
    warm = result["trials"][1:]
    result["median_seconds"] = {s: statistics.median(t[s]["seconds"] for t in warm)
                                for s in ("ingress", "index_access", "first_query", "warm_query")}
    result["median_seconds"]["build_plus_query"] = statistics.median(t["index_access"]["seconds"] + t["first_query"]["seconds"] for t in warm)
    result["median_seconds"]["ingress_build_query"] = statistics.median(sum(t[s]["seconds"] for s in ("ingress", "index_access", "first_query")) for t in warm)
    result["oracle_sha256"] = sha(args.oracle)
    write_json(args.output, result)


def run_worker(args, backend, layout, scale):
    key = f"{layout}-{scale}"
    output = args.cache / f"{backend}-{key}.json" if backend == "shapely" else args.output.parent / f"vibespatial-{key}.json"
    command = [sys.executable, str(Path(__file__).resolve()), str(args.path), "--cache", str(args.cache),
               "--output", str(output), "--worker", backend, "--layout", layout,
               "--rows", str(0 if scale == "all" else int(scale)), "--repeat", str(args.repeat),
               "--oracle", str(args.cache / f"oracle-{key}.npz"), "--mode", args.mode]
    if args.profile:
        command.append("--profile")
    print(f"Running {backend} {key}", flush=True)
    write_json(output, {"backend": backend, "status": "starting"})
    env = os.environ.copy()
    # Continuous event-file IO would contaminate timings; stage event queues
    # remain enabled and are reported. Record other runtime overrides verbatim.
    env.pop("VIBESPATIAL_EVENT_LOG", None)
    with output.with_suffix(".log").open("w") as log:
        try:
            completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
            status = "error" if completed.returncode else None
        except subprocess.TimeoutExpired:
            status = "timeout"
    result = json.loads(output.read_text()) if output.exists() else {}
    if status:
        result.update(status=status, timeout_seconds=args.timeout, log=str(output.with_suffix(".log")))
    elif not result or result.get("status") in {"running", "starting"}:
        raise ValueError(f"Worker did not complete: {output}")
    write_json(output, result)
    print(f"  {result['status']}: {result.get('median_seconds', {})}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scales", nargs="+", default=["1000", "10000", "100000", "all"])
    parser.add_argument("--layouts", nargs="+", choices=["roads", "segments"], default=["roads", "segments"])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--mode", choices=["auto", "gpu"], default="auto", help="vS public execution policy; CPU comparator is unchanged")
    parser.add_argument("--timeout", type=float, default=180, help="Per backend/scale worker seconds, including imports and validation")
    parser.add_argument("--profile", action="store_true", help="Additional untimed candidate query saved as .prof")
    parser.add_argument("--worker", choices=["shapely", "vibespatial"], help=argparse.SUPPRESS)
    parser.add_argument("--layout", choices=["roads", "segments"], help=argparse.SUPPRESS)
    parser.add_argument("--rows", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--oracle", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeat < 1 or args.timeout <= 0 or any(s != "all" and (not s.isdigit() or int(s) < 1) for s in args.scales):
        parser.error("repeat, timeout, and numeric scales must be positive")
    args.path, args.cache, args.output = args.path.resolve(), args.cache.resolve(), args.output.resolve()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    manifest = prepare(args)
    identity = measurement_identity(args, manifest)
    baseline_path = args.cache / "baseline.json"
    reused = baseline_path.exists()
    baseline = json.loads(baseline_path.read_text()) if reused else {"identity": identity, "created": datetime.now(UTC).isoformat(), "cases": {}}
    if baseline["identity"] != identity:
        raise ValueError("Comparator identity changed; select a new --cache directory")
    if reused:
        for key, packet in baseline["cases"].items():
            if sha(args.cache / f"oracle-{key}.npz") != packet["oracle_sha256"]:
                raise ValueError(f"Comparator oracle changed: {key}")
    report = {"identity": identity, "fixture": manifest, "baseline_path": str(baseline_path),
              "baseline_reused": reused, "baseline_created": baseline["created"], "created": datetime.now(UTC).isoformat(),
              "mode": args.mode,
              "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD", "--", "src"], cwd=ROOT)).hexdigest(),
              "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip(),
              "runtime_environment": {k: v for k, v in os.environ.items() if k.startswith(("VIBESPATIAL_", "CUDA_", "CUPY_"))},
              "cases": {}}
    for layout in args.layouts:
        for scale in args.scales:
            key = f"{layout}-{scale}"
            if key not in baseline["cases"]:
                if reused:
                    raise ValueError(f"Incomplete immutable baseline: {key}")
                cpu = run_worker(args, "shapely", layout, scale)
                if cpu["status"] != "ok":
                    raise ValueError(f"Shapely comparator failed: {key}; see worker log")
                baseline["cases"][key] = cpu
            cpu = baseline["cases"][key]
            candidate = run_worker(args, "vibespatial", layout, scale)
            case = {"shapely": cpu, "vibespatial": candidate}
            if candidate["status"] == "ok":
                case["vs_over_shapely"] = {stage: candidate["median_seconds"][stage] / cpu["median_seconds"][stage]
                                           for stage in ("build_plus_query", "warm_query", "ingress_build_query")}
            report["cases"][key] = case
            write_json(args.output, report)
    if not reused:
        write_json(baseline_path, baseline)
    print(f"Report: {args.output}")
    if any(case["vibespatial"]["status"] != "ok" for case in report["cases"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
