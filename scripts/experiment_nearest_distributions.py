"""Independent distribution checks for nearest strategy performance evidence."""
from __future__ import annotations

import argparse
import hashlib
import statistics
from pathlib import Path

from benchmark_osm_nearest import canonical, compare, sha, write_json
from experiment_nearest_grid import GridHierarchy
from experiment_nearest_hierarchy import timed
from experiment_nearest_strategies import StrategyHierarchy, experiment_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution", choices=("uniform", "clustered", "long"), required=True)
    parser.add_argument("--tree-rows", type=int, default=1000000)
    parser.add_argument("--rows", type=int, default=1000000)
    parser.add_argument("--fixture-rows", type=int, help="Generate this many points, then use a --rows prefix")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or min(args.rows, args.tree_rows, args.repeat) < 1:
        parser.error("Use a new output and positive counts")
    import numpy as np
    import shapely

    rng = np.random.default_rng(90217)
    n, q = args.tree_rows, args.fixture_rows or args.rows
    if args.rows > q:
        parser.error("Query prefix exceeds generated fixture")
    starts = rng.uniform(-1e4, 1e4, size=(n, 2))
    query = rng.uniform(-1e4, 1e4, size=(q, 2))
    if args.distribution == "clustered":
        centres = rng.uniform(-1e4, 1e4, size=(16, 2))
        starts = centres[rng.integers(16, size=n)] + rng.normal(0, 50, size=(n, 2))
        query = centres[rng.integers(16, size=q)] + rng.normal(0, 50, size=(q, 2))
    ends = starts + rng.normal(0, 10, size=(n, 2))
    if args.distribution == "long":
        ends = rng.uniform(-1000, 1000, size=(n, 2))
        starts = -ends
        query = rng.uniform(-900, 900, size=(q, 2))
    coords = np.stack((starts, ends), axis=1)
    query = query[:args.rows]
    twkb, qwkb = shapely.to_wkb(shapely.linestrings(coords)), shapely.to_wkb(shapely.points(query))
    root = Path(__file__).parent
    sources = {name: sha(root / name) for name in (Path(__file__).name, "experiment_nearest_strategies.py",
               "experiment_nearest_grid.py", "experimental_strtree_bounds.py", "experimental_strtree_bounds_kernels.py", "nearest_strategy_kernels.py", "experiment_nearest_hierarchy.py", "benchmark_osm_nearest.py")}
    report = {"distribution": args.distribution, "tree_rows": n, "query_rows": args.rows, "fixture_rows": q, "seed": 90217,
              "environment": experiment_environment(),
              "fixture_sha256": hashlib.sha256(coords.tobytes() + query.tobytes()).hexdigest(),
              "source_sha256": sources, "status": "running", "cases": {},
              "measurement": "Shared CPU fixture preparation excluded. Identical WKB ingress, completed build, first/warm query. "
                             "First trial per backend recorded separately, then repeat fresh-input trials; backends share one sequential worker. "
                             "GPU synchronized; int64/float64 host export included; oracle validation excluded."}
    write_json(args.output, report)
    for name in sources:
        args.output.with_name(args.output.stem + "." + name).write_text((root / name).read_text())
    cases = {"shapely": None, "str-leaf4": {"leaf_width": 4}, "str-leaf1": {"leaf_width": 1},
             "str-leaf4-fp32": {"leaf_width": 4, "bounds_precision": "fp32-outward"},
             "str-leaf1-fp32": {"leaf_width": 1, "bounds_precision": "fp32-outward"},
             "grid4096": {"grid": True}}
    if args.distribution == "long":
        cases = {"shapely": None, "str-leaf4": {"leaf_width": 4},
                 "str-virtual25": {"leaf_width": 4, "max_span": 25.}, "grid256": {"grid": True, "seed_resolution": 256}}
    expected = None
    for name, options in cases.items():
        gpu = options is not None
        if gpu:
            import vibespatial as vs
            vs.set_execution_mode("gpu")
            constructor = vs.GeoSeries.from_wkb
            options = dict(options)
            grid = options.pop("grid", False)
            seed = options.pop("seed_resolution", 4096 if grid else 512)
            factory = GridHierarchy if grid else StrategyHierarchy
        else:
            constructor = shapely.from_wkb
            factory = shapely.STRtree
        case = {"status": "running", "trials": []}
        report["cases"][name] = case
        for _trial in range(args.repeat + 1):
            record = {}
            case["trials"].append(record)
            case["active_stage"] = "ingress"
            write_json(args.output, report)
            (lines, points), record["ingress_s"] = timed(lambda constructor=constructor: (constructor(twkb), constructor(qwkb)), gpu=gpu)
            case["active_stage"] = "build"
            write_json(args.output, report)
            if gpu:
                try:
                    tree, record["build_s"] = timed(lambda factory=factory, lines=lines, seed=seed, options=options:
                        factory(lines, packing="str-recursive", fanout=8, query_order="morton", seed_resolution=seed, **options), gpu=True)
                except ValueError as error:
                    if not grid or not str(error).startswith("Grid admission:"):
                        raise
                    case.update(status="admission_declined", reason=str(error))
                    write_json(args.output, report)
                    break
            else:
                tree, record["build_s"] = timed(lambda lines=lines: shapely.STRtree(lines))
            for stage in ("first_query_s", "warm_query_s"):
                case["active_stage"] = stage
                write_json(args.output, report)
                if gpu:
                    result, record[stage] = timed(lambda tree=tree, points=points: tree.query_nearest(points), gpu=True)
                else:
                    result, record[stage] = timed(lambda tree=tree, points=points: tree.query_nearest(points, all_matches=True, return_distance=True))
                if expected is None:
                    expected = canonical(*result)
                    np.savez(args.output.with_suffix(".oracle.npz"), indices=expected[0], distances=expected[1])
                record[stage + "_parity"] = compare(result, expected)
                write_json(args.output, report)
                if not record[stage + "_parity"]["passed"]:
                    case["status"] = "parity_failed"
                    write_json(args.output, report)
                    raise AssertionError(record[stage + "_parity"])
            if gpu:
                record["mean_visits"], record["mean_segments"] = tree.cp.asnumpy(tree.last_visits).mean(axis=0).tolist()
                record["index_bytes"] = sum(a.nbytes for a in (tree.ax, tree.ay, tree.bx, tree.by, tree.ids, tree.bounds, tree.cells, tree.children))
                record["indexed_primitives"] = tree.n
            del tree, lines, points, result
        if case["status"] == "admission_declined":
            print(name, case["reason"], flush=True)
            continue
        case["status"] = "passed"
        case.pop("active_stage")
        case["median_s"] = {key: statistics.median(t[key] for t in case["trials"][1:])
                            for key in ("ingress_s", "build_s", "first_query_s", "warm_query_s")}
        print(name, case["median_s"], flush=True)
    report["status"] = "passed" if all(c["status"] == "passed" for c in report["cases"].values()) else "admission_declined"
    write_json(args.output, report)


if __name__ == "__main__":
    main()
