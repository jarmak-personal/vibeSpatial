"""Local all-family STR experiment; does not change production dispatch.

Shared bounds packing, bounded resumable candidate tiles, existing GPU
refiners, and three-pass exact-tie output. See the generalized STR ledger.
"""
from __future__ import annotations

from types import SimpleNamespace

from experiment_nearest_hierarchy import owned
from experimental_strtree_bounds import PackedBoundsBuilder
from experimental_strtree_bounds_kernels import SOURCE as BOUNDS_SOURCE
from experimental_strtree_kernels import SOURCE as QUERY_SOURCE


class GeneralizedSTRtree(PackedBoundsBuilder):
    def __init__(self, geometries, *, bounds_precision="fp32-outward", batch_size=65536, slots=8, refiner_factory=None, seed_first=False):
        import cupy as cp

        from vibespatial.cuda._runtime import (
            KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR,
            get_cuda_runtime,
            make_kernel_cache_key,
        )
        from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device
        from vibespatial.runtime._runtime import select_runtime
        from vibespatial.runtime.precision import KernelClass, PrecisionMode, select_precision_plan

        if bounds_precision not in ("fp64", "fp32-outward") or not 1 <= batch_size <= 262144 or slots not in (1, 4, 8, 16):
            raise ValueError("Invalid bounds precision, batch size or candidate slot count")
        self.cp, self.runtime = cp, get_cuda_runtime()
        self.ptr, self.i32 = KERNEL_PARAM_PTR, KERNEL_PARAM_I32
        self.geometry = owned(geometries)
        self.batch_size, self.slots = batch_size, slots
        self.seed_first = seed_first
        if self.geometry.row_count >= 2**24:
            raise ValueError("Experimental row limit is 2**24-1")
        self.width, self.fanout, self.packing = 1, 8, "str-recursive"
        self.precision_plan = select_precision_plan(runtime_selection=select_runtime("gpu"),
            kernel_class=KernelClass.METRIC, requested=PrecisionMode.FP64)
        self.bounds_precision_plan = select_precision_plan(runtime_selection=select_runtime("gpu"),
            kernel_class=KernelClass.COARSE, requested=PrecisionMode.FP64 if bounds_precision == "fp64" else PrecisionMode.FP32)
        self.refiner = None if refiner_factory is None else refiner_factory(self.geometry, self.precision_plan)
        float_bounds = bounds_precision == "fp32-outward"
        source = (f"#define FANOUT 8\n#define SLOTS {slots}\n#define FLOAT_BOUNDS {int(float_bounds)}\n"
                  f"#define INDEX_T {'float' if float_bounds else 'double'}\n" + BOUNDS_SOURCE + QUERY_SOURCE)
        self.kernels = self.runtime.compile_kernels(cache_key=make_kernel_cache_key("generalized-str-experiment", source),
            source=source, kernel_names=("packed_leaves", "str_parents", "compress_bounds", "advance_frontier", "consume_distances"),
            options=("--fmad=false",))
        bounds = cp.asarray(compute_geometry_bounds_device(self.geometry, precision=PrecisionMode.FP64))
        active = cp.flatnonzero(cp.all(cp.isfinite(bounds), axis=1)).astype(cp.int32)
        self.n = len(active)
        self.ids, self.bounds, self.children = active, cp.empty((4, 0)), cp.empty(0, dtype=cp.int32)
        if not self.n:
            return
        b = bounds[active]
        order = self.pack(.5 * (b[:, 0] + b[:, 2]), .5 * (b[:, 1] + b[:, 3]))
        self.ids = active[order]
        coords = tuple(cp.ascontiguousarray(b[order, i]) for i in range(4))
        self.build_recursive(coords)
        if float_bounds:
            compressed = cp.empty(self.bounds.shape, dtype=cp.float32)
            self.run("compress_bounds", 4 * self.total, (self.bounds, compressed), (4 * self.total,))
            self.bounds = compressed

    def query_relation(self, geometries):
        from vibespatial.api._native_relation import NativeRelation
        from vibespatial.cuda.cccl_primitives import exclusive_sum, sort_pairs
        from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device
        from vibespatial.runtime.precision import PrecisionMode
        from vibespatial.runtime.robustness import NumericalErrorEnvelope
        from vibespatial.spatial.nearest import (
            _compute_mixed_distances_gpu_device,
            _NearestMetricPrecisionContext,
        )
        from vibespatial.spatial.point_distance import compute_distance_center_device

        cp, query = self.cp, owned(geometries)
        refine = None if self.refiner is None else self.refiner.for_query(query)
        if query.row_count >= 2**24:
            raise ValueError("Experimental query row limit is 2**24-1")
        chunks = []
        self.last_work = dict(waves=0, candidate_capacity=0, candidate_pairs=0, node_visits=0, max_batch_rows=0)
        bounds = cp.asarray(compute_geometry_bounds_device(query, precision=PrecisionMode.FP64))
        qids = cp.flatnonzero(cp.all(cp.isfinite(bounds), axis=1)).astype(cp.int32)
        if self.n and len(qids):
            b = bounds[qids]
            order = sort_pairs(self.morton(.5 * (b[:, 0] + b[:, 2]), .5 * (b[:, 1] + b[:, 3])),
                               cp.arange(len(qids), dtype=cp.int32)).values
            qids = qids[order]
            context = _NearestMetricPrecisionContext(self.precision_plan, None, NumericalErrorEnvelope.exact(quantity="distance"))
            center = compute_distance_center_device(query, self.geometry)
            for start in range(0, len(qids), self.batch_size):
                ids = cp.ascontiguousarray(qids[start:start + self.batch_size])
                nq = len(ids)
                qb = cp.ascontiguousarray(bounds[ids].T)
                best = cp.full(nq, cp.inf, dtype=cp.float64)
                stacks = cp.empty((128, nq), dtype=cp.int32)
                depths = cp.empty(nq, dtype=cp.int32)
                left, right = cp.empty((2, self.slots * nq), dtype=cp.int32)
                active = cp.empty(self.slots * nq, dtype=cp.bool_)
                visits = cp.zeros(nq, dtype=cp.uint64)
                counts = cp.zeros(nq + 1, dtype=cp.int64)
                offsets = counts
                out_q, out_t, out_d = cp.empty(0, dtype=cp.int32), cp.empty(0, dtype=cp.int32), cp.empty(0, dtype=cp.float64)
                for phase in (0, 1, 2):
                    if phase == 2:
                        offsets = exclusive_sum(counts)
                        total = int(offsets[-1].item())
                        out_q, out_t, out_d = cp.empty(total, dtype=cp.int32), cp.empty(total, dtype=cp.int32), cp.empty(total, dtype=cp.float64)
                    counts.fill(0)
                    depths.fill(1)
                    stacks[0].fill(self.total - 1)
                    first_wave = True
                    while True:
                        self.run("advance_frontier", nq, (self.bounds, self.children, self.ids, qb, ids, best,
                            stacks, depths, left, right, active, visits), (nq, self.total, self.first_leaf,
                            1 if self.seed_first and phase == 0 and first_wave else self.slots))
                        first_wave = False
                        count = int(cp.count_nonzero(active).item())
                        if not count:
                            break
                        candidates = SimpleNamespace(d_left=left, d_right=right, total_pairs=len(left))
                        result = (refine(left, right, active) if refine is not None else
                            _compute_mixed_distances_gpu_device(query, self.geometry, None, None,
                                candidates, precision_context=context, pair_active=active, center_device=center))
                        if result is None or result[1]:
                            raise NotImplementedError("Device refinement declined; no CPU fallback in this experiment")
                        self.run("consume_distances", nq, (result[0], active, right, ids, best, counts, offsets,
                            out_q, out_t, out_d), (nq, phase))
                        self.last_work["waves"] += 1
                        self.last_work["candidate_capacity"] += len(left)
                        self.last_work["candidate_pairs"] += count
                self.last_work["node_visits"] += int(visits.sum().item())
                self.last_work["max_batch_rows"] = max(nq, self.last_work["max_batch_rows"])
                chunks.append((out_q, out_t, out_d))
        columns = [cp.concatenate([c[i] for c in chunks]) if chunks else cp.empty(0, dtype=cp.float64 if i == 2 else cp.int32) for i in range(3)]
        return NativeRelation(left_indices=columns[0], right_indices=columns[1], distances=columns[2],
            left_row_count=query.row_count, right_row_count=self.geometry.row_count, predicate="nearest",
            duplicate_policy="unique", origin="experimental_generalized_strtree")

    def query_nearest(self, geometries, *, all_matches=True, return_distance=True):
        cp = self.cp
        relation = self.query_relation(geometries)
        left, right, distance = relation.left_indices, relation.right_indices, relation.distances
        if not all_matches and len(left):
            from vibespatial.cuda.cccl_primitives import sort_pairs

            order = sort_pairs(left, cp.arange(len(left), dtype=cp.int32)).values
            left, right, distance = left[order], right[order], distance[order]
            keep = cp.concatenate((cp.ones(1, dtype=cp.bool_), left[1:] != left[:-1]))
            left, right, distance = left[keep], right[keep], distance[keep]
        indices = cp.asnumpy(cp.stack((left, right))).astype("int64", copy=False)
        return (indices, cp.asnumpy(distance)) if return_distance else indices
