"""Compare nearest search structures using the validated MA WKB fixtures.

Experiments only: finite nonempty 2D LineStrings and points. GPU segment
expansion, Morton or spatial-tile packing, 2/4/8-way traversal, coherent query
ordering, optional grid seeds, and device parent-road tie deduplication.
No approximate output or unverified radius cutoffs. FP64 reference policy.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import statistics
from pathlib import Path

from benchmark_osm_nearest import compare, sha, write_json
from experiment_nearest_hierarchy import SegmentHierarchy, owned, timed
from experimental_strtree_bounds import PackedBoundsBuilder
from nearest_strategy_kernels import COMMON, SOURCE


def experiment_environment():
    import importlib.metadata
    import platform
    import subprocess
    import sys
    from datetime import UTC, datetime

    import shapely

    return {"created": datetime.now(UTC).isoformat(), "python": sys.version, "host": platform.node(),
            "platform": platform.platform(), "geos": shapely.geos_version_string,
            "packages": {name: importlib.metadata.version(name) for name in ("shapely", "numpy", "pyarrow", "cupy-cuda12x")},
            "lock_sha256": sha(Path(__file__).resolve().parents[1] / "uv.lock"),
            "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True).strip(),
            "cpu": subprocess.check_output(["lscpu", "-J"], text=True),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}


class StrategyHierarchy(PackedBoundsBuilder, SegmentHierarchy):
    def __init__(self, lines, *, leaf_width=8, fanout=2, packing="morton",
                 query_order="input", seed_resolution=0, parent_roads=False, max_span=0., bounds_precision="fp64"):
        import cupy as cp

        from vibespatial.cuda._runtime import (
            KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR,
            get_cuda_runtime,
            make_kernel_cache_key,
        )
        from vibespatial.geometry.buffers import GeometryFamily
        from vibespatial.runtime._runtime import select_runtime
        from vibespatial.runtime.precision import KernelClass, PrecisionMode, select_precision_plan

        if fanout not in (2, 4, 8) or not 1 <= leaf_width < 2**24:
            raise ValueError("Expected fanout 2/4/8 and leaf width in 1..2**24-1")
        if packing not in ("morton", "str", "str-recursive") or query_order not in ("input", "morton"):
            raise ValueError("Unknown packing or query order")
        if not 0 <= seed_resolution <= 4096:
            raise ValueError("Seed resolution must be in 0..4096")
        if not math.isfinite(max_span) or max_span < 0:
            raise ValueError("Virtual span must be finite and nonnegative")
        if bounds_precision not in ("fp64", "fp32-outward"):
            raise ValueError("Unknown index precision")
        self.cp, self.runtime = cp, get_cuda_runtime()
        self.ptr, self.i32 = KERNEL_PARAM_PTR, KERNEL_PARAM_I32
        self.precision_plan = select_precision_plan(runtime_selection=select_runtime("gpu"),
                                                   kernel_class=KernelClass.METRIC, requested=PrecisionMode.FP64)
        self.bounds_precision_plan = select_precision_plan(runtime_selection=select_runtime("gpu"),
            kernel_class=KernelClass.COARSE, requested=PrecisionMode.FP64 if bounds_precision == "fp64" else PrecisionMode.FP32)
        self.width, self.fanout = leaf_width, fanout
        self.packing, self.query_order = packing, query_order
        self.seed_resolution, self.parent_roads = seed_resolution, parent_roads
        self.max_span = max_span
        index_type = "float" if bounds_precision == "fp32-outward" else "double"
        source = f"#define FANOUT {fanout}\n#define RECURSIVE_STR {int(packing == 'str-recursive')}\n#define INDEX_T {index_type}\n" + COMMON + SOURCE
        self.kernels = self.runtime.compile_kernels(
            cache_key=make_kernel_cache_key("nearest_strategy", source), source=source,
            kernel_names=("compress_bounds", "virtual_bounds", "packed_leaves", "packed_parents", "str_parents", "seed_cells", "packed_traverse"),
            options=("--fmad=false",),
        )
        state = owned(lines).device_state
        if state is None or set(state.families) != {GeometryFamily.LINESTRING}:
            raise ValueError("Experiment requires device-backed nonempty LineStrings")
        buf = state.families[GeometryFamily.LINESTRING]
        sizes = cp.diff(buf.geometry_offsets)
        if not bool(cp.all(sizes >= 2)) or not bool(cp.all(cp.isfinite(buf.x) & cp.isfinite(buf.y))):
            raise ValueError("Experiment requires finite lines with at least two vertices")
        if not parent_roads and not bool(cp.all(sizes == 2)):
            raise ValueError("Use parent_roads for non-segment input")
        self.n = len(buf.x) - len(lines)
        if not 0 < self.n < 2**24:
            raise ValueError("Experiment supports 1..2**24-1 segments")
        parent_starts = buf.geometry_offsets[:-1] - cp.arange(len(lines), dtype=cp.int32)
        parent_flags = cp.zeros(self.n, dtype=cp.int32)
        parent_flags[parent_starts] = 1
        parent = cp.cumsum(parent_flags, dtype=cp.int32) - 1
        starts = cp.arange(self.n, dtype=cp.int32) + parent
        ax, ay, bx, by = buf.x[starts], buf.y[starts], buf.x[starts + 1], buf.y[starts + 1]
        self.original_segments = self.n
        target_ids = parent if parent_roads else cp.arange(self.n, dtype=cp.int32)
        bounds_coords = ax, ay, bx, by
        if max_span:
            owner, bounds_coords = self.subdivide_bounds(bounds_coords, max_span)
            ax, ay, bx, by = (a[owner] for a in (ax, ay, bx, by))
            target_ids = target_ids[owner]
        cx, cy = .5 * (bounds_coords[0] + bounds_coords[2]), .5 * (bounds_coords[1] + bounds_coords[3])
        order = self.pack(cx, cy)
        self.ids = target_ids[order]
        self.ax, self.ay, self.bx, self.by = ax[order], ay[order], bx[order], by[order]
        bounds_coords = tuple(a[order] for a in bounds_coords)
        if packing == "str-recursive":
            self.build_recursive(bounds_coords)
        else:
            leaves = 1
            while leaves * self.width < self.n:
                leaves *= fanout
            self.first_leaf = (leaves - 1) // (fanout - 1)
            self.total = self.first_leaf + leaves
            self.bounds = cp.empty((4, self.total), dtype=cp.float64)
            self.children = cp.empty(0, dtype=cp.int32)
            self.run("packed_leaves", leaves, bounds_coords + (self.bounds,),
                     (self.n, self.first_leaf, leaves, self.total, self.width))
            count = leaves // fanout
            while count:
                first = (count - 1) // (fanout - 1)
                self.run("packed_parents", count, (self.bounds,), (first, count, self.total))
                count //= fanout
        if bounds_precision == "fp32-outward":
            compressed = cp.empty(self.bounds.shape, dtype=cp.float32)
            self.run("compress_bounds", 4 * self.total, (self.bounds, compressed), (4 * self.total,))
            self.bounds = compressed
        self.cells = cp.full(seed_resolution**2, 2**31 - 1, dtype=cp.int32)
        if seed_resolution:
            self.run("seed_cells", self.n, bounds_coords + (self.bounds, self.cells),
                     (self.n, self.total, seed_resolution))

    def subdivide_bounds(self, coords, span):
        from vibespatial.cuda.cccl_primitives import exclusive_sum

        cp = self.cp
        ax, ay, bx, by = coords
        pieces = cp.maximum(1., cp.ceil(cp.maximum(cp.abs(bx - ax), cp.abs(by - ay)) / span))
        if not bool(cp.all(cp.isfinite(pieces) & (pieces < 2**24))):
            raise ValueError("Virtual segment subdivision exceeds admission")
        pieces = pieces.astype(cp.int32)
        offsets = exclusive_sum(cp.concatenate((pieces.astype(cp.int64), cp.zeros(1, dtype=cp.int64))))
        total = int(offsets[-1].item())
        if total >= 2**24:
            raise ValueError(f"Virtual segment subdivision has {total} references; experimental limit is 2**24-1")
        flags = cp.zeros(total, dtype=cp.int32)
        flags[offsets[:-1]] = 1
        owner = cp.cumsum(flags, dtype=cp.int32) - 1
        local = (cp.arange(total, dtype=cp.int32) - offsets[owner]).astype(cp.int32)
        bounds = tuple(cp.empty(total, dtype=cp.float64) for _ in range(4))
        self.run("virtual_bounds", total, coords + (owner, local, pieces) + bounds, (total,))
        self.n = total
        return owner, bounds

    def query_nearest(self, points):
        from vibespatial.cuda.cccl_primitives import exclusive_sum, sort_pairs
        from vibespatial.geometry.buffers import GeometryFamily

        cp = self.cp
        state = owned(points).device_state
        if state is None or set(state.families) != {GeometryFamily.POINT}:
            raise ValueError("Experiment requires device-backed nonempty points")
        buf, nq = state.families[GeometryFamily.POINT], len(points)
        if not 0 < nq < 2**27 or len(buf.x) != nq or not bool(cp.all(cp.isfinite(buf.x) & cp.isfinite(buf.y))):
            raise ValueError("Experiment requires finite nonempty points, count < 2**27")
        qids = cp.arange(nq, dtype=cp.int32)
        qx, qy = buf.x, buf.y
        if self.query_order == "morton":
            qids = sort_pairs(self.morton(qx, qy), qids).values
            qx, qy = qx[qids], qy[qids]
        best, counts = cp.empty(nq, dtype=cp.float64), cp.zeros(nq + 1, dtype=cp.int64)
        visits = cp.empty((nq, 2), dtype=cp.uint64)
        dummy_i, dummy_d = cp.empty(0, dtype=cp.int32), cp.empty(0, dtype=cp.float64)
        common = (self.ax, self.ay, self.bx, self.by, self.ids, self.bounds, qx, qy, qids, self.cells, self.children, best, counts)
        sizes = (self.n, self.first_leaf, self.total, self.width, nq)
        self.run("packed_traverse", nq, common + (counts, dummy_i, dummy_i, dummy_d, visits),
                 sizes + (0, self.seed_resolution))
        if self.max_span:
            # A virtual box bounds only part of the refined segment. References
            # visited before the minimum tightens may be pruned on emission.
            # Count with the final minimum so count/scatter visit identical refs.
            self.run("packed_traverse", nq, common + (counts, dummy_i, dummy_i, dummy_d, visits),
                     sizes + (2, self.seed_resolution))
        offsets = exclusive_sum(counts)
        total = int(offsets[-1].item())
        out_q, out_t, out_d = cp.empty(total, dtype=cp.int32), cp.empty(total, dtype=cp.int32), cp.empty(total, dtype=cp.float64)
        self.run("packed_traverse", nq, common + (offsets, out_q, out_t, out_d, visits),
                 sizes + (1, self.seed_resolution))
        self.last_visits = visits
        if self.parent_roads or self.max_span:
            pairs = (out_q.astype(cp.uint64) << 32) | out_t.astype(cp.uint64)
            sorted_pairs = sort_pairs(pairs, out_d)
            keep = cp.concatenate((cp.ones(1, dtype=cp.bool_), sorted_pairs.keys[1:] != sorted_pairs.keys[:-1]))
            pairs, out_d = sorted_pairs.keys[keep], sorted_pairs.values[keep]
            out_q, out_t = (pairs >> 32).astype(cp.int32), (pairs & cp.uint64(0xffffffff)).astype(cp.int32)
        return cp.asnumpy(cp.stack((out_q, out_t))).astype("int64", copy=False), cp.asnumpy(out_d)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/vibespatial-osm/nearest-v2-baseline"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--layout", choices=("roads", "segments"), default="segments")
    parser.add_argument("--packing", choices=("morton", "str", "str-recursive"), default="morton")
    parser.add_argument("--fanout", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--leaf-width", type=int, default=8)
    parser.add_argument("--query-order", choices=("input", "morton"), default="input")
    parser.add_argument("--seed-resolution", type=int, default=0)
    parser.add_argument("--max-span", type=float, default=0., help="Virtual bounding span; zero disables subdivision")
    parser.add_argument("--bounds-precision", choices=("fp64", "fp32-outward"), default="fp64")
    args = parser.parse_args()
    if args.output.exists() or args.rows < 1 or args.repeat < 1:
        parser.error("Use a new output path and positive rows/repeat")
    import numpy as np
    import pyarrow.parquet as pq

    import vibespatial as vs

    vs.set_execution_mode("gpu")
    manifest = json.loads((args.cache / "fixture.json").read_text())
    baseline = json.loads((args.cache / "baseline.json").read_text())
    for layout in (args.layout, "points"):
        if sha(args.cache / f"{layout}.parquet") != manifest["files"][layout]["sha256"]:
            raise ValueError(f"Changed fixture {layout}")
    oracle_path = args.cache / f"oracle-{args.layout}-all.npz"
    if sha(oracle_path) != baseline["cases"][f"{args.layout}-all"]["oracle_sha256"]:
        raise ValueError("Changed oracle")
    with np.load(oracle_path) as oracle:
        mask = oracle["indices"][0] < args.rows
        expected = oracle["indices"][:, mask], oracle["distances"][mask]
    twkb = pq.read_table(args.cache / f"{args.layout}.parquet", columns=["wkb"])["wkb"].to_numpy()
    qwkb = pq.read_table(args.cache / "points.parquet", columns=["wkb"])["wkb"].to_numpy()[:args.rows]
    if len(qwkb) != args.rows:
        raise ValueError("Requested rows exceed fixture")
    root = Path(__file__).parent
    source_names = (Path(__file__).name, "nearest_strategy_kernels.py", "experiment_nearest_hierarchy.py",
                    "benchmark_osm_nearest.py", "experimental_strtree_bounds.py", "experimental_strtree_bounds_kernels.py", Path(inspect.getfile(StrategyHierarchy)).name)
    sources = {name: sha(root / name) for name in source_names}
    options = dict(leaf_width=args.leaf_width, fanout=args.fanout, packing=args.packing,
                   query_order=args.query_order, seed_resolution=args.seed_resolution, parent_roads=args.layout == "roads",
                   max_span=args.max_span, bounds_precision=args.bounds_precision)
    result = {"status": "running", "backend": StrategyHierarchy.__name__, "rows": args.rows, "layout": args.layout, "options": options,
              "environment": experiment_environment(),
              "fixture_sha256": sha(args.cache / "fixture.json"), "oracle_sha256": sha(oracle_path),
              "source_sha256": sources, "trials": [],
              "measurement": "Same ingress/build/first/warm boundaries as E3; one first-process then repeat fresh-input trials. "
                             "GPU synchronized; int64/float64 host outputs included. Imports/file reads/oracle checks excluded."}
    write_json(args.output, result)
    for name in sources:
        args.output.with_name(args.output.stem + "." + name).write_text((root / name).read_text())
    for _trial in range(args.repeat + 1):
        record = {}
        result["trials"].append(record)
        (lines, points), record["ingress_s"] = timed(lambda: (vs.GeoSeries.from_wkb(twkb), vs.GeoSeries.from_wkb(qwkb)), gpu=True)
        tree, record["build_s"] = timed(lambda lines=lines: StrategyHierarchy(lines, **options), gpu=True)
        write_json(args.output, result)
        for stage in ("first_query_s", "warm_query_s"):
            output, record[stage] = timed(lambda tree=tree, points=points: tree.query_nearest(points), gpu=True)
            record[stage + "_parity"] = compare(output, expected)
            write_json(args.output, result)
            if not record[stage + "_parity"]["passed"]:
                result["status"] = "parity_failed"
                write_json(args.output, result)
                np.savez(args.output.with_suffix(".mismatch.npz"), indices=output[0], distances=output[1])
                raise AssertionError(record[stage + "_parity"])
        visits = tree.cp.asnumpy(tree.last_visits)
        record["mean_nodes"], record["mean_segments"] = visits.mean(axis=0).tolist()
        record["visit_unit"] = getattr(tree, "visit_unit", "node")
        record["indexed_primitives"] = tree.n
        record["original_segments"] = getattr(tree, "original_segments", tree.n)
        record["index_bytes"] = sum(a.nbytes for a in (tree.ax, tree.ay, tree.bx, tree.by, tree.ids, tree.bounds, tree.cells, tree.children))
        del tree, points, lines, output
    result["median_s"] = {key: statistics.median(t[key] for t in result["trials"][1:])
                          for key in ("ingress_s", "build_s", "first_query_s", "warm_query_s")}
    result["status"] = "passed"
    write_json(args.output, result)
    print(json.dumps(result["median_s"], indent=2), flush=True)


if __name__ == "__main__":
    main()
