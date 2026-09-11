"""Direct uniform-grid nearest experiment; no hierarchy in its search path.

Every segment is replicated into all cells overlapped by its bounding box.
Expanding cell rings certify a minimum against the unvisited-cell boundary.
Two bounded-box passes count/emit ties once per segment using a canonical cell.
This tests the build/memory/query tradeoff against a seeded STR hierarchy.
"""
from __future__ import annotations

from experiment_nearest_hierarchy import SegmentHierarchy, owned
from nearest_strategy_kernels import COMMON

SOURCE = r"""
__device__ int grid_cell(double x, double origin, double span, int r) {
    double cell = (x - origin) / span * r;
    return (int)fmin((double)(r - 1), fmax(0., floor(cell)));
}
extern "C" __global__ void grid_query(
    const double* ax, const double* ay, const double* bx, const double* by, const int* ids,
    const double* extent, const int* entries, const int* offsets, const int* minx, const int* miny,
    const double* qx, const double* qy, const int* qids, double* bests, long long* counts,
    const long long* output_offsets, int* out_q, int* out_t, double* out_d,
    unsigned long long* visits, int nq, int r, int phase) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nq) return;
    double x = qx[q], y = qy[q], best = phase ? bests[q] : INFINITY;
    double ox = extent[0], oy = extent[1], sx = extent[2], sy = extent[3];
    double wx = sx / r, wy = sy / r;
    int cx = grid_cell(x, ox, sx, r), cy = grid_cell(y, oy, sy, r);
    unsigned long long cells = 0, evaluations = 0;
    if (!phase) {
        for (int ring = 0; ring < r; ++ring) {
            int lx = max(0, cx - ring), hx = min(r - 1, cx + ring);
            int ly = max(0, cy - ring), hy = min(r - 1, cy + ring);
            for (int edge = 0; edge < (ring ? 8 * ring : 1); ++edge) {
                int xx = cx, yy = cy;
                if (ring) {
                    int side = edge / (2 * ring), pos = edge % (2 * ring);
                    if (side == 0) { xx = cx - ring + pos; yy = cy - ring; }
                    if (side == 1) { xx = cx + ring; yy = cy - ring + pos; }
                    if (side == 2) { xx = cx + ring - pos; yy = cy + ring; }
                    if (side == 3) { xx = cx - ring; yy = cy + ring - pos; }
                }
                if (xx < 0 || xx >= r || yy < 0 || yy >= r) continue;
                int cell = yy * r + xx; ++cells;
                for (int at = offsets[cell]; at < offsets[cell + 1]; ++at) {
                    int i = entries[at]; ++evaluations;
                    best = fmin(best, segment_distance(x, y, ax[i], ay[i], bx[i], by[i]));
                }
            }
            // Outside the global grid there is no geometry. Interior faces
            // bound every unvisited cell, including for outside-grid queries.
            double unvisited = INFINITY;
            if (lx > 0) unvisited = fmin(unvisited, fabs(x - (ox + lx * wx)));
            if (hx < r - 1) unvisited = fmin(unvisited, fabs(ox + (hx + 1) * wx - x));
            if (ly > 0) unvisited = fmin(unvisited, fabs(y - (oy + ly * wy)));
            if (hy < r - 1) unvisited = fmin(unvisited, fabs(oy + (hy + 1) * wy - y));
            double limit = best + 64. * DBL_EPSILON * (fabs(x) + fabs(y) + best + 1.);
            if (unvisited > limit) break;
        }
        bests[q] = best; visits[2 * q] = cells; visits[2 * q + 1] = evaluations;
        return;
    }
    double limit = best + 64. * DBL_EPSILON * (fabs(x) + fabs(y) + best + 1.);
    int lx = grid_cell(x - limit, ox, sx, r), hx = grid_cell(x + limit, ox, sx, r);
    int ly = grid_cell(y - limit, oy, sy, r), hy = grid_cell(y + limit, oy, sy, r);
    long long matches = 0, output = phase == 2 ? output_offsets[q] : 0;
    for (int yy = ly; yy <= hy; ++yy) for (int xx = lx; xx <= hx; ++xx) {
        int cell = yy * r + xx;
        for (int at = offsets[cell]; at < offsets[cell + 1]; ++at) {
            int i = entries[at];
            // Canonical first cell in the intersection of segment's cell
            // rectangle and this query rectangle. Do not circle-prune cells:
            // their owner cell need not contain the closest segment point.
            if (xx != max(minx[i], lx) || yy != max(miny[i], ly)) continue;
            double d = segment_distance(x, y, ax[i], ay[i], bx[i], by[i]);
            if (d == best) {
                if (phase == 2) { out_q[output] = qids[q]; out_t[output] = ids[i]; out_d[output++] = d; }
                ++matches;
            }
        }
    }
    if (phase == 1) counts[q] = matches;
}
"""


class GridHierarchy(SegmentHierarchy):
    def __init__(self, lines, *, seed_resolution=1024, parent_roads=False, query_order="input", max_span=0., bounds_precision="fp64", **unused):
        import cupy as cp

        from vibespatial.cuda._runtime import (
            KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR,
            get_cuda_runtime,
            make_kernel_cache_key,
        )
        from vibespatial.cuda.cccl_primitives import exclusive_sum, sort_pairs
        from vibespatial.geometry.buffers import GeometryFamily
        from vibespatial.runtime._runtime import select_runtime
        from vibespatial.runtime.precision import KernelClass, PrecisionMode, select_precision_plan

        del unused
        if max_span:
            raise ValueError("Virtual bounds are only implemented in the hierarchy experiment")
        if bounds_precision != "fp64":
            raise ValueError("The direct grid experiment uses FP64 bounds")
        if not 1 <= seed_resolution <= 4096 or query_order not in ("input", "morton"):
            raise ValueError("Grid requires resolution in 1..4096 and input/morton query order")
        self.cp, self.runtime = cp, get_cuda_runtime()
        self.ptr, self.i32 = KERNEL_PARAM_PTR, KERNEL_PARAM_I32
        self.precision_plan = select_precision_plan(runtime_selection=select_runtime("gpu"),
                                                   kernel_class=KernelClass.METRIC, requested=PrecisionMode.FP64)
        self.r, self.parent_roads, self.query_order = seed_resolution, parent_roads, query_order
        self.visit_unit = "cell"
        source = COMMON + SOURCE
        self.kernels = self.runtime.compile_kernels(cache_key=make_kernel_cache_key("nearest_grid", source),
                                                    source=source, kernel_names=("grid_query",), options=("--fmad=false",))
        state = owned(lines).device_state
        if state is None or set(state.families) != {GeometryFamily.LINESTRING}:
            raise ValueError("Grid requires device-backed LineStrings")
        buf = state.families[GeometryFamily.LINESTRING]
        sizes = cp.diff(buf.geometry_offsets)
        if not bool(cp.all(sizes >= 2)) or not bool(cp.all(cp.isfinite(buf.x) & cp.isfinite(buf.y))):
            raise ValueError("Grid requires finite nonempty lines")
        if not parent_roads and not bool(cp.all(sizes == 2)):
            raise ValueError("Use parent_roads for non-segment input")
        self.n = len(buf.x) - len(lines)
        if not 0 < self.n < 2**24:
            raise ValueError("Grid supports 1..2**24-1 segments")
        parent_flags = cp.zeros(self.n, dtype=cp.int32)
        parent_flags[buf.geometry_offsets[:-1] - cp.arange(len(lines), dtype=cp.int32)] = 1
        parent = cp.cumsum(parent_flags, dtype=cp.int32) - 1
        starts = cp.arange(self.n, dtype=cp.int32) + parent
        self.ids = parent if parent_roads else cp.arange(self.n, dtype=cp.int32)
        self.ax, self.ay = buf.x[starts], buf.y[starts]
        self.bx, self.by = buf.x[starts + 1], buf.y[starts + 1]
        ox, oy = buf.x.min(), buf.y.min()
        sx, sy = cp.maximum(buf.x.max() - ox, 1.), cp.maximum(buf.y.max() - oy, 1.)
        self.bounds = cp.stack((ox, oy, sx, sy))
        r = self.r

        def cell(value, origin, span, direction):
            normalized = (value - origin) / span * r
            return cp.clip(cp.floor(cp.nextafter(normalized, direction)), 0, r - 1).astype(cp.int32)

        lx, ly = cell(cp.minimum(self.ax, self.bx), ox, sx, -cp.inf), cell(cp.minimum(self.ay, self.by), oy, sy, -cp.inf)
        hx, hy = cell(cp.maximum(self.ax, self.bx), ox, sx, cp.inf), cell(cp.maximum(self.ay, self.by), oy, sy, cp.inf)
        widths = (hx - lx + 1).astype(cp.int64)
        counts = widths * (hy - ly + 1)
        offsets = exclusive_sum(cp.concatenate((counts, cp.zeros(1, dtype=cp.int64))))
        entries = int(offsets[-1].item())
        if entries >= 64_000_000:
            raise ValueError(f"Grid admission: {entries} replicated entries exceeds experimental 64M limit")
        flags = cp.zeros(entries, dtype=cp.int32)
        flags[offsets[:-1]] = 1
        owner = cp.cumsum(flags, dtype=cp.int32) - 1
        local = cp.arange(entries, dtype=cp.int32) - offsets[owner]
        keys = ((ly[owner] + local // widths[owner]) * r + lx[owner] + local % widths[owner]).astype(cp.int32)
        pairs = sort_pairs(keys, owner)
        # Sorted runs already encode the CSR offsets. A dense CUB histogram
        # at 4096**2 bins requested 48 GiB scratch on this GPU; reuse runs.
        ends = cp.flatnonzero(cp.concatenate((pairs.keys[1:] != pairs.keys[:-1], cp.ones(1, dtype=cp.bool_))))
        begins = cp.concatenate((cp.zeros(1, dtype=ends.dtype), ends[:-1] + 1))
        cell_counts = cp.zeros(r * r + 1, dtype=cp.int32)
        cell_counts[pairs.keys[ends]] = (ends - begins + 1).astype(cp.int32)
        cell_offsets = exclusive_sum(cell_counts)
        self.cells = cp.empty(entries + r * r + 1 + 2 * self.n, dtype=cp.int32)
        self.entries = self.cells[:entries]
        self.offsets = self.cells[entries:entries + r * r + 1]
        self.minx, self.miny = self.cells[-2 * self.n:-self.n], self.cells[-self.n:]
        self.entries[:], self.offsets[:], self.minx[:], self.miny[:] = pairs.values, cell_offsets, lx, ly
        self.children = cp.empty(0, dtype=cp.int32)

    def query_nearest(self, points):
        from vibespatial.cuda.cccl_primitives import exclusive_sum, sort_pairs
        from vibespatial.geometry.buffers import GeometryFamily

        cp = self.cp
        state = owned(points).device_state
        if state is None or set(state.families) != {GeometryFamily.POINT}:
            raise ValueError("Grid requires device-backed points")
        buf, nq = state.families[GeometryFamily.POINT], len(points)
        if not 0 < nq < 2**27 or len(buf.x) != nq or not bool(cp.all(cp.isfinite(buf.x) & cp.isfinite(buf.y))):
            raise ValueError("Grid requires finite nonempty points")
        qx, qy, qids = buf.x, buf.y, cp.arange(nq, dtype=cp.int32)
        if self.query_order == "morton":
            qids = sort_pairs(self.morton(qx, qy), qids).values
            qx, qy = qx[qids], qy[qids]
        best, counts = cp.empty(nq, dtype=cp.float64), cp.zeros(nq + 1, dtype=cp.int64)
        visits = cp.empty((nq, 2), dtype=cp.uint64)
        dummy_i, dummy_d = cp.empty(0, dtype=cp.int32), cp.empty(0, dtype=cp.float64)
        common = (self.ax, self.ay, self.bx, self.by, self.ids, self.bounds, self.entries, self.offsets,
                  self.minx, self.miny, qx, qy, qids, best, counts)
        for phase in (0, 1):
            self.run("grid_query", nq, common + (counts, dummy_i, dummy_i, dummy_d, visits), (nq, self.r, phase))
        offsets = exclusive_sum(counts)
        total = int(offsets[-1].item())
        out_q, out_t, out_d = cp.empty(total, dtype=cp.int32), cp.empty(total, dtype=cp.int32), cp.empty(total, dtype=cp.float64)
        self.run("grid_query", nq, common + (offsets, out_q, out_t, out_d, visits), (nq, self.r, 2))
        self.last_visits = visits
        if self.parent_roads:
            pairs = (out_q.astype(cp.uint64) << 32) | out_t.astype(cp.uint64)
            sorted_pairs = sort_pairs(pairs, out_d)
            keep = cp.concatenate((cp.ones(1, dtype=cp.bool_), sorted_pairs.keys[1:] != sorted_pairs.keys[:-1]))
            pairs, out_d = sorted_pairs.keys[keep], sorted_pairs.values[keep]
            out_q, out_t = (pairs >> 32).astype(cp.int32), (pairs & cp.uint64(0xffffffff)).astype(cp.int32)
        return cp.asnumpy(cp.stack((out_q, out_t))).astype("int64", copy=False), cp.asnumpy(out_d)


if __name__ == "__main__":
    import experiment_nearest_strategies as runner

    runner.StrategyHierarchy = GridHierarchy
    runner.main()
