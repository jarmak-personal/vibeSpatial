"""Reusable all-family STR nearest backend; device candidate/refine to NativeRelation."""

from __future__ import annotations

import math
from threading import RLock
from types import SimpleNamespace

from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.kernels.spatial.packed_str_kernels import BOUNDS_SOURCE, QUERY_SOURCE
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode
from vibespatial.runtime.residency import Residency

request_warmup([
    "exclusive_scan_i64", "radix_sort_f64_i32", "radix_sort_i64_i32",
    "radix_sort_u32_i32", "radix_sort_i64_i64",
])


class _PackedBoundsBuilder:
    def build_recursive(self, bounds_coords):
        cp, fanout = self.cp, self.fanout
        levels = [math.ceil(self.n / self.width)]
        while levels[-1] > 1:
            levels.append(math.ceil(levels[-1] / fanout))
        self.first_leaf, self.total = levels[0], sum(levels)
        self.bounds = cp.empty((4, self.total), dtype=cp.float64)
        self.children = cp.full((self.total - levels[0]) * fanout, -1, dtype=cp.int32)
        self.run(
            "packed_leaves",
            levels[0],
            bounds_coords + (self.bounds,),
            (self.n, 0, levels[0], self.total, self.width),
        )
        start = 0
        for count, parent_count in zip(levels[:-1], levels[1:], strict=True):
            b = self.bounds[:, start : start + count]
            order = self.pack(0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]), width=fanout)
            next_start = start + count
            first_child = (next_start - levels[0]) * fanout
            self.children[first_child : first_child + count] = order + start
            self.run(
                "str_parents",
                parent_count,
                (self.bounds, self.children),
                (next_start, parent_count, self.total, levels[0]),
            )
            start = next_start

    def pack(self, x, y, *, width=None):
        from vibespatial.cuda.cccl_primitives import sort_pairs

        cp, n = self.cp, len(x)
        width = self.width if width is None else width
        if self.packing == "morton":
            return sort_pairs(self.morton(x, y), cp.arange(n, dtype=cp.int32)).values
        # Equal-population x slices, then y within each slice. The recursive
        # builder also calls this for parent boxes; leaf-only STR does not.
        order_x = sort_pairs(x, cp.arange(n, dtype=cp.int32)).values
        slices = math.ceil(math.sqrt(math.ceil(n / width)))
        slice_size = math.ceil(n / (slices * width)) * width
        # Sort original float64 y values, then stably group by integer slice.
        # This avoids coordinate quantization and host geometry loops.
        order_y = sort_pairs(y[order_x], cp.arange(n, dtype=cp.int32)).values
        slice_ids = (order_y // slice_size).astype(cp.int32)
        return order_x[
            order_y[
                sort_pairs(
                    slice_ids.astype(cp.int64) * n + cp.arange(n, dtype=cp.int64),
                    cp.arange(n, dtype=cp.int32),
                ).values
            ]
        ]

    def morton(self, x, y):
        cp = self.cp

        def spread(v):
            v = v & cp.uint32(0x0000FFFF)
            v = (v | (v << 8)) & cp.uint32(0x00FF00FF)
            v = (v | (v << 4)) & cp.uint32(0x0F0F0F0F)
            v = (v | (v << 2)) & cp.uint32(0x33333333)
            return (v | (v << 1)) & cp.uint32(0x55555555)

        ix = ((x - x.min()) / cp.maximum(x.max() - x.min(), 1.0) * 65535.0).astype(cp.uint32)
        iy = ((y - y.min()) / cp.maximum(y.max() - y.min(), 1.0) * 65535.0).astype(cp.uint32)
        return spread(ix) | (spread(iy) << 1)

    def run(self, name, n, arrays, scalars):
        kernel = self.kernels[name]
        grid, block = self.runtime.launch_config(kernel, n)
        self.runtime.launch(
            kernel,
            grid=grid,
            block=block,
            params=(
                tuple(a.data.ptr for a in arrays) + scalars,
                (self.ptr,) * len(arrays) + (self.i32,) * len(scalars),
            ),
        )


_KERNEL_NAMES = (
    "packed_leaves",
    "str_parents",
    "compress_bounds",
    "advance_frontier",
    "consume_distances",
)
_SOURCES = {
    precision: (
        f"#define FANOUT 8\n#define SLOTS 8\n#define FLOAT_BOUNDS {int(precision == 'fp32')}\n"
        f"#define INDEX_T {'float' if precision == 'fp32' else 'double'}\n"
        + BOUNDS_SOURCE
        + QUERY_SOURCE
    )
    for precision in ("fp32", "fp64")
}
request_nvrtc_warmup([(f"packed-str-{p}", s, _KERNEL_NAMES) for p, s in _SOURCES.items()])


class PackedSTRtree(_PackedBoundsBuilder):
    """Bounded candidate tiles over immutable owned rows, retaining original IDs.

    The tree holds envelope state and optional segment acceleration. Scratch is
    query-local; a lock and CUDA event protect the one-entry prepared-query cache
    across host threads and streams. No geometry or relation exports occur here.
    """

    def __init__(self, geometry, *, bounds=None, precision_plan):
        import cupy as cp

        from vibespatial.cuda._runtime import (
            KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR,
            get_cuda_runtime,
            make_kernel_cache_key,
        )
        from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device

        self.cp, self.runtime = cp, get_cuda_runtime()
        self.ptr, self.i32 = KERNEL_PARAM_PTR, KERNEL_PARAM_I32
        self.geometry = geometry.physicalize_device_rows(allow_capacity_allocation=True)
        self.precision_plan = precision_plan
        self.batch_size, self.slots = 65536, 8
        self.width, self.fanout, self.packing = 1, 8, "str-recursive"
        self._lock, self._ready = RLock(), cp.cuda.Event()
        self._segment_refiner = None
        # 8-way DFS requires at most 1 + 7*8 entries under this bound.
        # Node IDs, sort values and candidate tiles remain signed int32.
        if self.geometry.row_count >= 2**24:
            raise ValueError("Packed STR supports fewer than 2**24 indexed rows")
        precision = precision_plan.compute_precision.value
        source = _SOURCES[precision]
        self.kernels = self.runtime.compile_kernels(
            cache_key=make_kernel_cache_key(f"packed-str-{precision}", source),
            source=source,
            kernel_names=_KERNEL_NAMES,
        )
        if bounds is None:
            bounds = compute_geometry_bounds_device(self.geometry, precision=PrecisionMode.FP64)
        bounds = cp.asarray(bounds, dtype=cp.float64)
        active = cp.flatnonzero(cp.all(cp.isfinite(bounds), axis=1)).astype(cp.int32)
        self.n = len(active)
        self.ids, self.bounds, self.children = active, cp.empty((4, 0)), cp.empty(0, dtype=cp.int32)
        if self.n:
            b = bounds[active]
            order = self.pack(0.5 * b[:, 0] + 0.5 * b[:, 2], 0.5 * b[:, 1] + 0.5 * b[:, 3])
            self.ids = active[order]
            self.build_recursive(tuple(cp.ascontiguousarray(b[order, i]) for i in range(4)))
            if precision_plan.compute_precision is PrecisionMode.FP32:
                compressed = cp.empty(self.bounds.shape, dtype=cp.float32)
                self.run(
                    "compress_bounds", 4 * self.total, (self.bounds, compressed), (4 * self.total,)
                )
                self.bounds = compressed
        self._ready.record()

    def query_relation(self, query, *, return_all=True, max_distance=None, exclusive=False):
        """Return exact minimum-distance ties as a sorted device NativeRelation."""
        with self._lock:
            self.cp.cuda.get_current_stream().wait_event(self._ready)
            try:
                return self._query_relation(
                    query, return_all=return_all, max_distance=max_distance, exclusive=exclusive
                )
            finally:
                self._ready.record()

    def _query_relation(self, query, *, return_all, max_distance, exclusive):
        from vibespatial.api._native_relation import NativeRelation
        from vibespatial.cuda.cccl_primitives import exclusive_sum, sort_pairs
        from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device
        from vibespatial.spatial.nearest import (
            _compute_mixed_distances_gpu_device,
            _plan_device_resident_metric_precision,
        )
        from vibespatial.spatial.point_distance import compute_distance_center_device

        cp = self.cp
        if max_distance is not None and not max_distance > 0:
            raise ValueError("max_distance must be greater than 0")
        if query.row_count >= 2**24:
            raise ValueError("Packed STR supports fewer than 2**24 query rows")
        query = query.physicalize_device_rows(allow_capacity_allocation=True)
        chunks = []
        bounds = cp.asarray(compute_geometry_bounds_device(query, precision=PrecisionMode.FP64))
        qids = cp.flatnonzero(cp.all(cp.isfinite(bounds), axis=1)).astype(cp.int32)
        if self.n and len(qids):
            b = bounds[qids]
            order = sort_pairs(
                self.morton(0.5 * b[:, 0] + 0.5 * b[:, 2], 0.5 * b[:, 1] + 0.5 * b[:, 3]),
                cp.arange(len(qids), dtype=cp.int32),
            ).values
            qids = qids[order]
            # Nearest tie membership requires FP64 terminal refinement even when
            # the COARSE envelope plan uses outward FP32 bounds.
            context = _plan_device_resident_metric_precision(
                query, self.geometry, min(len(qids), self.batch_size) * self.slots
            ).refinement_context()
            center = compute_distance_center_device(query, self.geometry)
            refine = self._refiner_for_query(query, context)
            for start in range(0, len(qids), self.batch_size):
                ids = cp.ascontiguousarray(qids[start : start + self.batch_size])
                nq = len(ids)
                qb = cp.ascontiguousarray(bounds[ids].T)
                best = cp.full(
                    nq, cp.inf if max_distance is None else max_distance, dtype=cp.float64
                )
                stacks = cp.empty((128, nq), dtype=cp.int32)
                depths = cp.empty(nq, dtype=cp.int32)
                left, right = cp.empty((2, self.slots * nq), dtype=cp.int32)
                active = cp.empty(self.slots * nq, dtype=cp.bool_)
                counts = cp.zeros(nq + 1, dtype=cp.int64)
                offsets = counts
                out_q, out_t = cp.empty((2, 0), dtype=cp.int32)
                out_d = cp.empty(0, dtype=cp.float64)
                for phase in (0, 1, 2):
                    if phase == 2:
                        offsets = exclusive_sum(counts)
                        total = int(
                            self.runtime.copy_device_to_host(
                                offsets[-1:], reason="packed STR tie-output allocation fence"
                            )[0]
                        )
                        out_q, out_t = cp.empty((2, total), dtype=cp.int32)
                        out_d = cp.empty(total, dtype=cp.float64)
                    counts.fill(0)
                    depths.fill(1)
                    stacks[0].fill(self.total - 1)
                    first_wave = True
                    while True:
                        self.run(
                            "advance_frontier",
                            nq,
                            (
                                self.bounds,
                                self.children,
                                self.ids,
                                qb,
                                ids,
                                best,
                                stacks,
                                depths,
                                left,
                                right,
                                active,
                            ),
                            (
                                nq,
                                self.total,
                                self.first_leaf,
                                1 if phase == 0 and first_wave else self.slots,
                            ),
                        )
                        first_wave = False
                        if not bool(
                            self.runtime.copy_device_to_host(
                                cp.any(active).reshape(1),
                                reason="packed STR candidate convergence fence",
                            )[0]
                        ):
                            break
                        candidates = SimpleNamespace(
                            d_left=left, d_right=right, total_pairs=len(left)
                        )
                        result = (
                            refine(left, right, active)
                            if refine is not None
                            else _compute_mixed_distances_gpu_device(
                                query,
                                self.geometry,
                                None,
                                None,
                                candidates,
                                precision_context=context,
                                pair_active=active,
                                center_device=center,
                            )
                        )
                        if result is None or result[1]:
                            raise NotImplementedError(
                                "Packed STR requires an admitted device distance refiner"
                            )
                        distances = result[0]
                        if exclusive:
                            distances = self._exclude_equal(query, left, right, active, distances)
                        self.run(
                            "consume_distances",
                            nq,
                            (
                                distances,
                                active,
                                right,
                                ids,
                                best,
                                counts,
                                offsets,
                                out_q,
                                out_t,
                                out_d,
                            ),
                            (nq, phase, int(return_all)),
                        )
                chunks.append((out_q, out_t, out_d))
        columns = [
            cp.concatenate([c[i] for c in chunks])
            if chunks
            else cp.empty(0, dtype=cp.float64 if i == 2 else cp.int32)
            for i in range(3)
        ]
        if len(columns[0]):
            order = sort_pairs(
                columns[0].astype(cp.int64) * self.geometry.row_count + columns[1],
                cp.arange(len(columns[0]), dtype=cp.int64),
            ).values
            columns = [c[order] for c in columns]
        return NativeRelation(
            left_indices=columns[0],
            right_indices=columns[1],
            distances=columns[2],
            left_row_count=query.row_count,
            right_row_count=self.geometry.row_count,
            predicate="nearest",
            duplicate_policy="unique",
            sorted_by_left=True,
            origin="packed_str_nearest",
        )

    def _refiner_for_query(self, query, context):
        from vibespatial.geometry.owned import ensure_device_geometry_size_bounds
        from vibespatial.kernels.spatial.segment_bvh import KINDS, SegmentBVHRefiner

        if not (set(query.families) & set(KINDS) and set(self.geometry.families) & set(KINDS)):
            return None

        # Reuse per-row structural bounds, deriving missing proofs from device
        # offsets once. Simple/null padding must not hide an expensive outlier.
        def complex_boundaries(owned):
            return ensure_device_geometry_size_bounds(
                owned, reason="nearest segment refinement shape admission"
            ) >= 32

        if not (complex_boundaries(self.geometry) or complex_boundaries(query)):
            return None
        if self._segment_refiner is None:
            self._segment_refiner = SegmentBVHRefiner(
                self.geometry, context.coarse_plan, bounds_plan=self.precision_plan
            )
        return self._segment_refiner.for_query(query, context=context)

    def _exclude_equal(self, query, left, right, active, distances):
        from vibespatial.geometry.equality import _geom_equals_topological_gpu_device
        from vibespatial.runtime.adaptive import plan_dispatch_selection

        cp = self.cp
        positions = cp.flatnonzero(active & (distances == 0.0)).astype(cp.int64)
        if len(positions):
            lhs = query.device_take(left[positions]).physicalize_device_rows(
                allow_capacity_allocation=True
            )
            rhs = self.geometry.device_take(right[positions]).physicalize_device_rows(
                allow_capacity_allocation=True
            )
            plan = plan_dispatch_selection(
                kernel_name="nearest_exclusive_equals",
                kernel_class=KernelClass.PREDICATE,
                row_count=len(positions),
                requested_mode=ExecutionMode.GPU,
            )
            equal = _geom_equals_topological_gpu_device(lhs, rhs, plan)
            if equal is None:
                raise NotImplementedError("Nearest exclusion requires native topological equality")
            distances[positions] = cp.where(equal, cp.inf, distances[positions])
        return distances


@register_kernel_variant(
    "packed_str_index",
    "gpu-cuda-python",
    kernel_class=KernelClass.COARSE,
    geometry_families=(
        "point",
        "linestring",
        "polygon",
        "multipoint",
        "multilinestring",
        "multipolygon",
    ),
    execution_modes=(ExecutionMode.GPU,),
    supports_mixed=True,
    precision_modes=(PrecisionMode.AUTO, PrecisionMode.FP32, PrecisionMode.FP64),
    preferred_residency=Residency.DEVICE,
    tags=("cuda-python", "native-spatial-index"),
)
def packed_str_index(geometry, *, bounds=None, precision_plan):
    """Build reusable packed bounds behind an admitted NativeSpatialIndex contract."""
    return PackedSTRtree(geometry, bounds=bounds, precision_plan=precision_plan)
