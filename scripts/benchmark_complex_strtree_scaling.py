"""Scale holed polygon nearest through the public API with immutable CPU oracles."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path

from benchmark_osm_nearest import canonical, compare, sha, write_json
from experiment_nearest_strategies import experiment_environment


def fixture(rows, vertices, seed=1702):
    import numpy as np
    import shapely

    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2 * np.pi, vertices, endpoint=False)
    unit = np.stack((np.cos(theta), np.sin(theta)), axis=1)
    result = []
    for _ in range(2):
        centres = rng.uniform(-10000, 10000, (rows, 2))
        shell = shapely.linearrings(centres[:, None, :] + unit[None, :, :] * 4.0)
        hole = shapely.linearrings(centres[:, None, :] + unit[None, ::-1, :])
        result.append(shapely.to_wkb(shapely.polygons(shell, holes=hole[:, None])))
    return result


def timed(fn, gpu):
    if gpu:
        import cupy as cp

        cp.cuda.runtime.deviceSynchronize()
    start = time.perf_counter()
    value = fn()
    if gpu:
        cp.cuda.runtime.deviceSynchronize()
    return value, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument(
        "--vertices", type=int, required=True, help="Vertices in each of shell and hole"
    )
    parser.add_argument("--reuse-comparator", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.rows < 1 or args.vertices < 4:
        parser.error("Use positive rows, at least four vertices, and a new output")
    import numpy as np
    import shapely

    twkb, qwkb = fixture(args.rows, args.vertices)
    checksum = hashlib.sha256()
    for wkbs in (twkb, qwkb):
        for wkb in wkbs:
            checksum.update(len(wkb).to_bytes(8, "little"))
            checksum.update(wkb)
    env = experiment_environment()
    identity = dict(
        rows=args.rows,
        vertices_per_ring=args.vertices,
        rings_per_polygon=2,
        fixture_sha256=checksum.hexdigest(),
        generator_sha256=hashlib.sha256(inspect.getsource(fixture).encode()).hexdigest(),
        environment={
            k: env[k] for k in ("python", "host", "platform", "packages", "geos", "lock_sha256")
        },
        boundary="Same WKB ingress, index accessor, first and reused host pair/distance output. One first-process and one fresh-input trial; imports, fixture, oracle work excluded.",
    )
    paths = [
        Path(__file__),
        *Path("src/vibespatial/kernels/spatial").glob("*.py"),
        *(
            Path(p)
            for p in (
                "src/vibespatial/api/sindex.py",
                "src/vibespatial/api/_native_metadata.py",
                "src/vibespatial/cuda/_runtime.py",
                "src/vibespatial/spatial/nearest.py",
                "src/vibespatial/spatial/point_distance_kernels.py",
                "src/vibespatial/cuda/device_functions/segment_distance.py",
                "src/vibespatial/cuda/device_functions/point_segment_distance.py",
                "src/vibespatial/cuda/cccl_primitives.py",
                "src/vibespatial/runtime/cccl_warmup_specs.py",
                "src/vibespatial/predicates/binary.py",
                "src/vibespatial/predicates/polygon.py",
                "src/vibespatial/predicates/point_relations.py",
                "src/vibespatial/spatial/spatial_index_device.py",
            )
        ),
    ]
    report = dict(
        status="running",
        identity=identity,
        environment=env,
        source_sha256={str(p): sha(p) for p in paths},
        cases={},
    )
    for p in paths:
        args.output.with_name(args.output.stem + "." + p.name).write_text(p.read_text())
    expected = None
    if args.reuse_comparator:
        old = json.loads(args.reuse_comparator.read_text())
        if old["identity"] != identity or old["cases"]["shapely"]["status"] != "passed":
            raise ValueError("Comparator identity/status mismatch")
        oracle = Path(old["oracle_path"])
        if sha(oracle) != old["oracle_sha256"]:
            raise ValueError("Comparator oracle changed")
        with np.load(oracle) as data:
            expected = data["indices"], data["distances"]
        report.update(
            oracle_path=str(oracle),
            oracle_sha256=sha(oracle),
            reused_comparator=str(args.reuse_comparator),
        )
        report["cases"]["shapely"] = old["cases"]["shapely"]
    for backend in ("gpu",) if expected is not None else ("shapely", "gpu"):
        gpu = backend == "gpu"
        if gpu:
            import vibespatial as vs

            vs.set_execution_mode("gpu")
        case = dict(status="running", trials=[])
        report["cases"][backend] = case
        for _ in range(2):
            record = {}
            case["trials"].append(record)
            constructor = vs.GeoSeries.from_wkb if gpu else shapely.from_wkb
            report["active_stage"] = backend + " ingress"
            write_json(args.output, report)
            (tree, query), record["ingress_s"] = timed(
                lambda constructor=constructor: (constructor(twkb), constructor(qwkb)), gpu
            )
            index, record["build_s"] = timed(
                lambda tree=tree, gpu=gpu: tree.sindex if gpu else shapely.STRtree(tree), gpu
            )
            for stage in ("first_query_s", "warm_query_s"):
                report["active_stage"] = backend + " " + stage
                write_json(args.output, report)
                result, record[stage] = timed(
                    lambda index=index, query=query, gpu=gpu: (
                        index.nearest(query, return_all=True, return_distance=True)
                        if gpu
                        else index.query_nearest(query, all_matches=True, return_distance=True)
                    ),
                    gpu,
                )
                if expected is None:
                    expected = canonical(*result)
                    oracle = args.output.with_suffix(".oracle.npz")
                    np.savez(oracle, indices=expected[0], distances=expected[1])
                    report.update(oracle_path=str(oracle), oracle_sha256=sha(oracle))
                record[stage + "_parity"] = compare(result, expected)
                write_json(args.output, report)
                if not record[stage + "_parity"]["passed"]:
                    raise AssertionError(record[stage + "_parity"])
            if gpu:
                record["backends"] = index.backend_info
            del tree, query, index, result
            print(backend, record, flush=True)
        case["status"] = "passed"
        write_json(args.output, report)
    report["status"] = "passed"
    report.pop("active_stage")
    write_json(args.output, report)


if __name__ == "__main__":
    main()
