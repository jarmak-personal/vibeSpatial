"""Identity-qualified CPU/generalized-GPU STR comparison on independent fixtures."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import statistics
from pathlib import Path

from benchmark_osm_nearest import canonical, compare, sha, write_json
from experiment_generalized_strtree import GeneralizedSTRtree
from experiment_nearest_hierarchy import timed
from experiment_nearest_strategies import experiment_environment


def make_fixture(tree_family, query_family, n, q, seed=19871):
    import numpy as np
    import shapely

    rng = np.random.default_rng(seed)

    def generate(family, count):
        xy = rng.uniform(-10000., 10000., (count, 2))
        if family == "point":
            return shapely.points(xy)
        if family == "line":
            return shapely.linestrings(np.stack((xy, xy + rng.normal(0., 5., xy.shape)), axis=1))
        if family == "polygon":
            return shapely.box(xy[:,0], xy[:,1], xy[:,0] + 4., xy[:,1] + 4.)
        if family == "multipart":
            a = shapely.box(xy[:,0], xy[:,1], xy[:,0] + 2., xy[:,1] + 2.)
            b = shapely.box(xy[:,0] + 5., xy[:,1], xy[:,0] + 7., xy[:,1] + 2.)
            return shapely.multipolygons(np.stack((a,b), axis=1))
        out = np.empty(count, dtype=object)
        for i, kind in enumerate(("point", "line", "polygon", "multipart")):
            out[i::4] = generate(kind, len(out[i::4]))
        return out

    return generate(tree_family, n), generate(query_family, q)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    families = ("point", "line", "polygon", "multipart", "mixed")
    parser.add_argument("--tree-family", choices=families, required=True)
    parser.add_argument("--query-family", choices=families, required=True)
    parser.add_argument("--tree-rows", type=int, default=100000)
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--public-sindex", action="store_true", help="Measure production GeoSeries.sindex.nearest")
    parser.add_argument("--reuse-comparator", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or min(args.rows, args.tree_rows, args.repeat) < 1:
        parser.error("Use positive counts and a new output")
    import numpy as np
    import shapely

    tree, points = make_fixture(args.tree_family, args.query_family, args.tree_rows, args.rows)
    twkb, qwkb = shapely.to_wkb(tree), shapely.to_wkb(points)
    fixture_hash = hashlib.sha256()
    for wkbs in (twkb, qwkb):
        for wkb in wkbs:
            fixture_hash.update(len(wkb).to_bytes(8, "little"))
            fixture_hash.update(wkb)
    environment = experiment_environment()
    identity = dict(fixture_sha256=fixture_hash.hexdigest(), generator_sha256=hashlib.sha256(inspect.getsource(make_fixture).encode()).hexdigest(),
        tree_family=args.tree_family, query_family=args.query_family, tree_rows=args.tree_rows, rows=args.rows,
        repeat=args.repeat, python=environment["python"], host=environment["host"], platform=environment["platform"],
        shapely=environment["packages"]["shapely"], geos=environment["geos"], lock_sha256=environment["lock_sha256"],
        boundary="Same WKB ingress/build/first/reused host int64+float64 output. One first trial, then repeat fresh inputs. Imports, fixture/file/oracle work excluded.")
    source_paths = [Path(__file__), *(Path("scripts")/s for s in ("experiment_generalized_strtree.py", "experimental_strtree_kernels.py",
        "experimental_strtree_bounds.py", "experimental_strtree_bounds_kernels.py", "experiment_nearest_hierarchy.py", "nearest_strategy_kernels.py", "benchmark_osm_nearest.py")),
        *(Path("src/vibespatial/spatial")/s for s in ("nearest.py", "point_distance.py", "point_distance_kernels.py", "segment_distance.py", "segment_distance_kernels.py"))]
    source_paths.extend(Path("src/vibespatial/kernels/spatial").glob("*.py"))
    source_paths.extend(Path(p) for p in ("src/vibespatial/api/sindex.py", "src/vibespatial/api/_native_metadata.py", "src/vibespatial/cuda/_runtime.py", "src/vibespatial/cuda/device_functions/segment_distance.py",
                "src/vibespatial/cuda/device_functions/point_segment_distance.py",
                "src/vibespatial/cuda/cccl_primitives.py",
                "src/vibespatial/runtime/cccl_warmup_specs.py",
                "src/vibespatial/predicates/binary.py",
                "src/vibespatial/predicates/polygon.py",
                "src/vibespatial/predicates/point_relations.py",
                "src/vibespatial/spatial/spatial_index_device.py", "src/vibespatial/spatial/index_backends.py", "src/vibespatial/geometry/equality.py"))
    report = dict(status="running", identity=identity, environment=environment,
        source_sha256={str(p):sha(p) for p in source_paths}, cases={}, options=dict(batch_size=args.batch_size, slots=args.slots, public_sindex=args.public_sindex))
    for p in source_paths:
        args.output.with_name(args.output.stem + "." + p.name).write_text(p.read_text())
    expected = None
    if args.reuse_comparator:
        previous = json.loads(args.reuse_comparator.read_text())
        if previous["identity"] != identity or previous["cases"]["shapely"]["status"] != "passed":
            raise ValueError("Comparator identity or status mismatch")
        op = Path(previous["oracle_path"])
        if sha(op) != previous["oracle_sha256"]:
            raise ValueError("Comparator oracle changed")
        with np.load(op) as oracle:
            expected = oracle["indices"], oracle["distances"]
        report["cases"]["shapely"] = previous["cases"]["shapely"]
        report["reused_comparator"] = str(args.reuse_comparator)
        report.update(oracle_path=str(op), oracle_sha256=sha(op))
    write_json(args.output, report)
    for backend in (("gpu",) if expected is not None else ("shapely", "gpu")):
        gpu = backend == "gpu"
        if gpu:
            import vibespatial as vs
            vs.set_execution_mode("gpu")
        case = {"status":"running", "trials":[]}
        report["cases"][backend] = case
        for _ in range(args.repeat + 1):
            record = {}
            case["trials"].append(record)
            case["active_stage"] = "ingress"
            write_json(args.output, report)
            constructor = vs.GeoSeries.from_wkb if gpu else shapely.from_wkb
            (lines, query), record["ingress_s"] = timed(lambda constructor=constructor:(constructor(twkb), constructor(qwkb)), gpu=gpu)
            case["active_stage"] = "build"
            write_json(args.output, report)
            index, record["build_s"] = timed(lambda lines=lines, gpu=gpu: (lines.sindex if args.public_sindex else GeneralizedSTRtree(lines, batch_size=args.batch_size, slots=args.slots)) if gpu else shapely.STRtree(lines), gpu=gpu)
            for stage in ("first_query_s", "warm_query_s"):
                case["active_stage"] = stage
                write_json(args.output, report)
                result, record[stage] = timed(lambda index=index, query=query, gpu=gpu: index.nearest(query, return_all=True, return_distance=True) if gpu and args.public_sindex else index.query_nearest(query, all_matches=True, return_distance=True), gpu=gpu)
                if expected is None:
                    expected = canonical(*result)
                    oracle_path = args.output.with_suffix(".oracle.npz")
                    np.savez(oracle_path, indices=expected[0], distances=expected[1])
                    report.update(oracle_path=str(oracle_path), oracle_sha256=sha(oracle_path))
                record[stage + "_parity"] = compare(result, expected)
                write_json(args.output, report)
                if not record[stage + "_parity"]["passed"]:
                    report["status"] = case["status"] = "parity_failed"
                    write_json(args.output, report)
                    raise AssertionError(record[stage + "_parity"])
            if gpu and args.public_sindex:
                record["backends"] = index.backend_info
            if gpu and not args.public_sindex:
                record["work"] = index.last_work
                record["index_bytes"] = index.bounds.nbytes + index.children.nbytes + index.ids.nbytes
            del lines, query, index, result
        case["status"] = "passed"
        case.pop("active_stage")
        case["median_s"] = {k:statistics.median(t[k] for t in case["trials"][1:]) for k in ("ingress_s", "build_s", "first_query_s", "warm_query_s")}
        print(backend, case["median_s"], flush=True)
    report["status"] = "passed"
    write_json(args.output, report)


if __name__ == "__main__":
    main()
