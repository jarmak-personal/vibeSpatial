"""Experimental point/segment hierarchy; intentionally outside public dispatch.

Physical shape: Morton-sorted segments, packed binary bounds, independent
query traversals, and count/scan/scatter tie output. Only finite 2D points and
two-vertex lines are admitted. Explicit FP64 PrecisionPlan is a reference
experiment, not the production consumer-GPU precision policy. CPU Shapely is
used only for fixture/oracle work; all index/search computation stays on GPU.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from benchmark_osm_nearest import canonical, compare, sha, write_json

SOURCE = r"""
#define INFINITY __longlong_as_double(0x7ff0000000000000LL)
#define DBL_EPSILON 2.2204460492503131e-16
extern "C" __global__ void leaves(
    const double* ax, const double* ay, const double* bx, const double* by,
    double* bounds, int n, int base, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= base) return;
    double lo_x = INFINITY, lo_y = INFINITY, hi_x = -INFINITY, hi_y = -INFINITY;
    for (int j = i * width; j < min(n, (i + 1) * width); ++j) {
        lo_x = fmin(lo_x, fmin(ax[j], bx[j]));
        lo_y = fmin(lo_y, fmin(ay[j], by[j]));
        hi_x = fmax(hi_x, fmax(ax[j], bx[j]));
        hi_y = fmax(hi_y, fmax(ay[j], by[j]));
    }
    int stride = 2 * base, node = base + i;
    bounds[node] = lo_x; bounds[stride + node] = lo_y;
    bounds[2 * stride + node] = hi_x; bounds[3 * stride + node] = hi_y;
}
extern "C" __global__ void parents(double* b, int start, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= start) return;
    int node = start + i, left = 2 * node;
    b[node] = fmin(b[left], b[left + 1]);
    b[stride + node] = fmin(b[stride + left], b[stride + left + 1]);
    b[2 * stride + node] = fmax(b[2 * stride + left], b[2 * stride + left + 1]);
    b[3 * stride + node] = fmax(b[3 * stride + left], b[3 * stride + left + 1]);
}
__device__ double box_distance2(const double* b, int node, int stride, double x, double y) {
    double dx = fmax(0., fmax(__dsub_rd(b[node], x), __dsub_rd(x, b[2 * stride + node])));
    double dy = fmax(0., fmax(__dsub_rd(b[stride + node], y), __dsub_rd(y, b[3 * stride + node])));
    return __dadd_rd(__dmul_rd(dx, dx), __dmul_rd(dy, dy));
}
__device__ double segment_distance(double x, double y, double ax, double ay, double bx, double by) {
    double vx = bx - ax, vy = by - ay, wx = x - ax, wy = y - ay;
    double length2 = vx * vx + vy * vy;
    double projection = length2 == 0. ? 0. : (wx * vx + wy * vy) / length2;
    if (projection <= 0.) return sqrt(wx * wx + wy * wy);
    if (projection >= 1.) {
        wx = x - bx; wy = y - by;
        return sqrt(wx * wx + wy * wy);
    }
    // Perpendicular distance; preserve the oracle's binary64 operation order.
    return fabs(((ay - y) * vx - (ax - x) * vy) / length2) * sqrt(length2);
}
extern "C" __global__ void traverse(
    const double* ax, const double* ay, const double* bx, const double* by,
    const int* ids, const double* b, const double* qx, const double* qy,
    double* bests, long long* counts, const long long* offsets,
    int* out_q, int* out_t, double* out_d, unsigned long long* visits,
    int n, int base, int width, int nq, int emit) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nq) return;
    double x = qx[q], y = qy[q], best = emit ? bests[q] : INFINITY;
    int stack[32], sp = 0;
    stack[sp++] = 1;
    long long matches = 0, output = emit ? offsets[q] : 0;
    unsigned long long nodes = 0, segments = 0;
    while (sp) {
        int node = stack[--sp];
        ++nodes;
        // This pad only broadens traversal; equality below never uses a tolerance.
        double limit = best + 64. * DBL_EPSILON * (fabs(x) + fabs(y) + best + 1.);
        if (box_distance2(b, node, 2 * base, x, y) > limit * limit) continue;
        if (node < base) {
            int near = 2 * node, far = near + 1;
            if (box_distance2(b, near, 2 * base, x, y) > box_distance2(b, far, 2 * base, x, y)) {
                int swap = near; near = far; far = swap;
            }
            // Host admission proves depth <= 30; binary DFS needs depth + 1 slots.
            stack[sp++] = far; stack[sp++] = near;
        } else {
            int start = (node - base) * width;
            for (int j = start; j < min(n, start + width); ++j) {
                ++segments;
                double d = segment_distance(x, y, ax[j], ay[j], bx[j], by[j]);
                if (!emit && d < best) { best = d; matches = 0; }
                if (d == best) {
                    if (emit) { out_q[output] = q; out_t[output] = ids[j]; out_d[output++] = d; }
                    ++matches;
                }
            }
        }
    }
    if (!emit) { bests[q] = best; counts[q] = matches; visits[2 * q] = nodes; visits[2 * q + 1] = segments; }
}
"""


def owned(series):
    values = series.array
    return values.to_owned() if hasattr(values, "to_owned") else values._owned


class SegmentHierarchy:
    """Reference-only hierarchy. Input lifetime is retained by device arrays."""

    def __init__(self, lines, *, leaf_width=8):
        import cupy as cp

        from vibespatial.cuda._runtime import (
            KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR,
            get_cuda_runtime,
            make_kernel_cache_key,
        )
        from vibespatial.cuda.cccl_primitives import sort_pairs
        from vibespatial.geometry.buffers import GeometryFamily
        from vibespatial.runtime._runtime import select_runtime
        from vibespatial.runtime.precision import KernelClass, PrecisionMode, select_precision_plan

        self.cp, self.runtime = cp, get_cuda_runtime()
        self.ptr, self.i32 = KERNEL_PARAM_PTR, KERNEL_PARAM_I32
        self.precision_plan = select_precision_plan(
            runtime_selection=select_runtime("gpu"), kernel_class=KernelClass.METRIC,
            requested=PrecisionMode.FP64,
        )
        self.kernels = self.runtime.compile_kernels(
            cache_key=make_kernel_cache_key("experimental_segment_hierarchy", SOURCE),
            source=SOURCE, kernel_names=("leaves", "parents", "traverse"), options=("--fmad=false",),
        )
        state = owned(lines).device_state
        if set(state.families) != {GeometryFamily.LINESTRING}:
            raise ValueError("Experiment requires only nonempty two-vertex LineStrings")
        buf = state.families[GeometryFamily.LINESTRING]
        if not bool(cp.all(cp.diff(buf.geometry_offsets) == 2)):
            raise ValueError("Experiment requires exactly two vertices per line")
        if not bool(cp.all(cp.isfinite(buf.x) & cp.isfinite(buf.y))):
            raise ValueError("Experiment requires finite coordinates")
        self.n, self.width = len(lines), leaf_width
        if not self.n or not 1 <= leaf_width < 2**27 or self.n >= 2**27:
            raise ValueError("Experiment requires segment count and leaf width in 1..2**27-1")
        self.base = 1 << (((self.n + leaf_width - 1) // leaf_width) - 1).bit_length()
        self.ids = sort_pairs(self.morton((buf.x[::2] + buf.x[1::2]) * .5,
                                         (buf.y[::2] + buf.y[1::2]) * .5),
                              cp.arange(self.n, dtype=cp.int32)).values
        self.ax, self.ay = buf.x[2 * self.ids], buf.y[2 * self.ids]
        self.bx, self.by = buf.x[2 * self.ids + 1], buf.y[2 * self.ids + 1]
        self.bounds = cp.empty((4, 2 * self.base), dtype=cp.float64)
        self.run("leaves", self.base, (self.ax, self.ay, self.bx, self.by, self.bounds),
                 (self.n, self.base, self.width))
        level = self.base // 2
        while level:
            self.run("parents", level, (self.bounds,), (level, 2 * self.base))
            level //= 2

    def morton(self, x, y):
        cp = self.cp

        def spread(v):
            v = v & cp.uint32(0x0000ffff)
            v = (v | (v << 8)) & cp.uint32(0x00ff00ff)
            v = (v | (v << 4)) & cp.uint32(0x0f0f0f0f)
            v = (v | (v << 2)) & cp.uint32(0x33333333)
            return (v | (v << 1)) & cp.uint32(0x55555555)

        ix = ((x - x.min()) / cp.maximum(x.max() - x.min(), 1.) * 65535.).astype(cp.uint32)
        iy = ((y - y.min()) / cp.maximum(y.max() - y.min(), 1.) * 65535.).astype(cp.uint32)
        return spread(ix) | (spread(iy) << 1)

    def run(self, name, n, arrays, scalars):
        kernel = self.kernels[name]
        grid, block = self.runtime.launch_config(kernel, n)
        self.runtime.launch(kernel, grid=grid, block=block,
                            params=(tuple(a.data.ptr for a in arrays) + scalars,
                                    (self.ptr,) * len(arrays) + (self.i32,) * len(scalars)))

    def query_nearest(self, points):
        cp = self.cp
        from vibespatial.cuda.cccl_primitives import exclusive_sum
        from vibespatial.geometry.buffers import GeometryFamily

        state = owned(points).device_state
        if set(state.families) != {GeometryFamily.POINT}:
            raise ValueError("Experiment requires nonempty points")
        buf = state.families[GeometryFamily.POINT]
        nq = len(points)
        if len(buf.x) != nq or not bool(cp.all(cp.isfinite(buf.x) & cp.isfinite(buf.y))):
            raise ValueError("Experiment requires finite nonempty points")
        best = cp.empty(nq, dtype=cp.float64)
        counts = cp.zeros(nq + 1, dtype=cp.int64)
        visits = cp.empty((nq, 2), dtype=cp.uint64)
        dummy_i = cp.empty(0, dtype=cp.int32)
        dummy_d = cp.empty(0, dtype=cp.float64)
        common = (self.ax, self.ay, self.bx, self.by, self.ids, self.bounds, buf.x, buf.y, best, counts)
        sizes = (self.n, self.base, self.width, nq)
        self.run("traverse", nq, common + (counts, dummy_i, dummy_i, dummy_d, visits), sizes + (0,))
        offsets = exclusive_sum(counts)
        total = int(offsets[-1].item())
        out_q, out_t = cp.empty(total, dtype=cp.int32), cp.empty(total, dtype=cp.int32)
        out_d = cp.empty(total, dtype=cp.float64)
        self.run("traverse", nq, common + (offsets, out_q, out_t, out_d, visits), sizes + (1,))
        self.last_visits = visits
        # Match the public NumPy intp pair-array boundary on this 64-bit host.
        return (cp.asnumpy(cp.stack((out_q, out_t))).astype("int64", copy=False), cp.asnumpy(out_d))


def timed(fn, *, gpu=False):
    if gpu:
        import cupy as cp
        cp.cuda.runtime.deviceSynchronize()
    start = perf_counter()
    value = fn()
    if gpu:
        cp.cuda.runtime.deviceSynchronize()
    return value, perf_counter() - start


def smoke():
    import numpy as np
    import shapely

    import vibespatial as vs

    vs.set_execution_mode("gpu")
    rng = np.random.default_rng(42)
    # CPU-only oracle fixtures: duplicate edges, degeneracies, endpoints,
    # exact ties, near ties, far queries, translated coordinates, and long edges.
    edges = np.concatenate((rng.normal(size=(300, 2, 2)), np.array([
        [[-1, 0], [1, 0]], [[-1, 0], [1, 0]], [[0, 0], [0, 0]],
        [[-1, 0.00015], [1, 0.00015]], [[-1e6, 0], [1e6, 0]],
    ])))
    queries = np.concatenate((rng.normal(size=(1000, 2)), [[0, 0], [0, 1], [1e7, 1e7]]))
    results = []
    for shift in (0., 1e7):
        lines = shapely.linestrings(edges + shift)
        points = shapely.points(queries + shift)
        oracle = shapely.STRtree(lines).query_nearest(points, all_matches=True, return_distance=True)
        for width in (1, 8, 32):
            tree = SegmentHierarchy(vs.GeoSeries.from_wkb(shapely.to_wkb(lines)), leaf_width=width)
            candidate = tree.query_nearest(vs.GeoSeries.from_wkb(shapely.to_wkb(points)))
            parity = compare(candidate, oracle)
            results.append({"shift": shift, "leaf_width": width, **parity})
    print(json.dumps(results, indent=2), flush=True)
    if not all(r["passed"] for r in results):
        raise AssertionError("Hierarchy failed adversarial smoke parity")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/vibespatial-osm/nearest-v2-baseline"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--leaf-width", type=int, default=8)
    parser.add_argument("--backend", choices=("hierarchy", "shapely", "fixed-k"), default="hierarchy")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return
    if args.output is None or args.output.exists():
        parser.error("Provide a new --output path; experiment evidence is immutable")
    if args.rows < 1 or args.repeat < 1:
        parser.error("--rows and --repeat must be positive")
    import numpy as np
    import pyarrow.parquet as pq

    fixture = json.loads((args.cache / "fixture.json").read_text())
    for layout in ("segments", "points"):
        if sha(args.cache / f"{layout}.parquet") != fixture["files"][layout]["sha256"]:
            raise ValueError(f"Changed fixture: {layout}")
    tree_wkb = pq.read_table(args.cache / "segments.parquet", columns=["wkb"])["wkb"].to_numpy()
    query_wkb = pq.read_table(args.cache / "points.parquet", columns=["wkb"])["wkb"].to_numpy()[:args.rows]
    if len(query_wkb) != args.rows:
        raise ValueError("Requested more queries than the fixture contains")
    oracle_path = args.cache / "oracle-segments-all.npz"
    baseline = json.loads((args.cache / "baseline.json").read_text())
    if sha(oracle_path) != baseline["cases"]["segments-all"]["oracle_sha256"]:
        raise ValueError("Changed oracle; restore the validated comparator")
    with np.load(oracle_path) as expected:
        mask = expected["indices"][0] < args.rows
        oracle = expected["indices"][:, mask], expected["distances"][mask]
    result = {"backend": args.backend, "rows": args.rows, "segments": len(tree_wkb),
              "leaf_width": args.leaf_width, "script_sha256": sha(__file__),
              "fixture": fixture, "oracle_sha256": sha(oracle_path), "status": "running", "trials": []}
    result["environment"] = {
        "created": datetime.now(UTC).isoformat(), "python": sys.version,
        "host": platform.node(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("shapely", "numpy", "pyarrow", "cupy-cuda12x")},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "lock_sha256": sha(Path(__file__).resolve().parents[1] / "uv.lock"),
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True).strip(),
        "cpu": subprocess.check_output(["lscpu", "-J"], text=True),
    }
    result["measurement"] = (
        "File reads, imports, fixture/oracle validation excluded. First-process trial then repeat fresh-input trials. "
        "Each measures WKB ingress, completed index construction, first query, repeated query. "
        "CUDA synchronized. Host pairs/distances included; correctness and visit telemetry excluded. "
        "Hierarchy is experimental FP64 point/segment execution, not public dispatch. "
        "Fixed-k is exactly-one diagnostic; bounds preparation excluded. Disk compilation cache may be warm."
    )
    write_json(args.output, result)
    args.output.with_suffix(".source.py").write_text(Path(__file__).read_text())
    gpu = args.backend != "shapely"
    if gpu:
        import vibespatial as vs
        vs.set_execution_mode("gpu")
        constructor = vs.GeoSeries.from_wkb
    else:
        import shapely
        constructor = shapely.from_wkb
    for trial in range(args.repeat + 1):
        record = {}
        result["trials"].append(record)
        (lines, points), record["ingress_s"] = timed(lambda: (constructor(tree_wkb), constructor(query_wkb)), gpu=gpu)
        write_json(args.output, result)
        if args.backend == "shapely":
            tree, record["build_s"] = timed(lambda lines=lines: shapely.STRtree(lines))
            def query(tree=tree, points=points):
                return tree.query_nearest(points, all_matches=True, return_distance=True)
        elif args.backend == "hierarchy":
            tree, record["build_s"] = timed(lambda lines=lines: SegmentHierarchy(lines, leaf_width=args.leaf_width), gpu=True)
            def query(tree=tree, points=points):
                return tree.query_nearest(points)
        else:
            from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device
            from vibespatial.spatial.spatial_index_knn_device import spatial_index_knn_device
            tree, record["build_s"] = timed(lambda lines=lines: lines.sindex._native_spatial_index_for_query(), gpu=True)
            q, t = owned(points), owned(lines)
            qb, tb = compute_geometry_bounds_device(q), compute_geometry_bounds_device(t)

            def query(q=q, t=t, qb=qb, tb=tb, tree=tree):
                output = spatial_index_knn_device(q, t, qb, tb, native_spatial_index=tree, k=1, return_all=False)
                if output is None:
                    raise RuntimeError("Fixed-k engine declined the workload")
                left, right, distances = output.to_host()
                return np.stack((left, right)), distances
        write_json(args.output, result)
        for stage in ("first_query_s", "warm_query_s"):
            output, record[stage] = timed(query, gpu=gpu)
            if args.backend == "fixed-k":
                ci, cd = canonical(*output)
                oi, od = oracle
                best_rows = np.searchsorted(oi[0], ci[0])
                parity = {"all_match_contract": False, "minimum_distances_equal": bool(
                    len(cd) == args.rows and np.allclose(cd, od[best_rows], atol=1e-6, rtol=1e-10))}
            else:
                parity = compare(output, oracle)
            record[stage + "_parity"] = parity
            write_json(args.output, result)
            if not parity.get("passed", parity.get("minimum_distances_equal", False)):
                np.savez(args.output.with_suffix(".mismatch.npz"), indices=output[0], distances=output[1])
                raise AssertionError(f"Parity failed: {parity}")
        if args.backend == "hierarchy":
            visits = tree.cp.asnumpy(tree.last_visits)
            record["first_pass_mean_nodes"] = float(visits[:, 0].mean())
            record["first_pass_mean_segments"] = float(visits[:, 1].mean())
            record["retained_index_bytes"] = sum(a.nbytes for a in
                                                 (tree.ax, tree.ay, tree.bx, tree.by, tree.ids, tree.bounds))
        del tree, lines, points, output, query
    result["median_s"] = {key: statistics.median(t[key] for t in result["trials"][1:])
                          for key in ("ingress_s", "build_s", "first_query_s", "warm_query_s")}
    result["status"] = "passed" if args.backend != "fixed-k" else "diagnostic_only"
    write_json(args.output, result)
    print(json.dumps(result["median_s"], indent=2), flush=True)


if __name__ == "__main__":
    main()
