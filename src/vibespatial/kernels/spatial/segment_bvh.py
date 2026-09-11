"""Cached segment work and per-row BVHs for expensive candidate refinement."""

from __future__ import annotations

from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.device_functions.segment_distance import SEGMENT_DISTANCE_DEVICE
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import FAMILY_TAGS
from vibespatial.kernels.spatial.segment_bvh_kernels import SOURCE
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode
from vibespatial.runtime.residency import Residency

KINDS = {
    GeometryFamily.LINESTRING: 0,
    GeometryFamily.MULTILINESTRING: 1,
    GeometryFamily.POLYGON: 2,
    GeometryFamily.MULTIPOLYGON: 3,
}


_KERNEL_NAMES = ("row_bvh_build", "cooperative_segment_refine")
_SOURCES = {
    p: (
        f"#define INDEX_T {'float' if p == 'fp32' else 'double'}\n"
        f"#define FLOAT_BOUNDS {int(p == 'fp32')}\n"
        + (
            "#define LOWER(x) __double2float_rd(x)\n#define UPPER(x) __double2float_ru(x)\n"
            if p == "fp32"
            else "#define LOWER(x) (x)\n#define UPPER(x) (x)\n"
        )
        + SEGMENT_DISTANCE_DEVICE
        + SOURCE
    )
    for p in ("fp32", "fp64")
}
request_nvrtc_warmup([(f"segment-bvh-{p}", s, _KERNEL_NAMES) for p, s in _SOURCES.items()])


request_warmup(
    ["exclusive_scan_i32", "exclusive_scan_i64", "radix_sort_i32_i32", "upper_bound_i32"]
)


class RowSegmentBVH:
    def __init__(self, geometry, *, bounds_plan, hierarchy=True):
        import cupy as cp

        from vibespatial.cuda._runtime import get_cuda_runtime, make_kernel_cache_key
        from vibespatial.cuda.cccl_primitives import exclusive_sum, sort_pairs, upper_bound
        from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device
        from vibespatial.runtime.residency import Residency, TransferTrigger
        from vibespatial.spatial.segment_primitives import _extract_segments_gpu

        geometry.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="segment hierarchy consumes owned device buffers",
        )
        self.geometry, self.state = geometry, geometry._ensure_device_state()
        # Coordinate capacity bounds segment count before the extractor's
        # int32 scans run. Array sizes are metadata; no coordinates leave device.
        if sum(b.x.size for f, b in self.state.families.items() if f in KINDS) >= 2**29:
            raise ValueError("Segment BVH supports fewer than 2**29 non-point coordinates")
        self.row_bounds = cp.ascontiguousarray(compute_geometry_bounds_device(geometry))
        self.runtime = get_cuda_runtime()
        precision = bounds_plan.compute_precision.value
        source = _SOURCES[precision]
        self.kernels = self.runtime.compile_kernels(
            cache_key=make_kernel_cache_key(f"segment-bvh-{precision}", source),
            source=source,
            kernel_names=_KERNEL_NAMES,
        )
        segments = _extract_segments_gpu(geometry)
        if segments.count >= 2**29:
            raise ValueError("Segment BVH supports fewer than 2**29 extracted segments")
        rows = cp.asarray(segments.row_indices).astype(cp.int32, copy=False)
        order = sort_pairs(rows, cp.arange(segments.count, dtype=cp.int32)).values
        self.coords = tuple(
            cp.ascontiguousarray(cp.asarray(getattr(segments, k))[order])
            for k in ("x0", "y0", "x1", "y1")
        )
        counts = cp.bincount(rows, minlength=geometry.row_count).astype(cp.int32)
        maximum = (
            int(
                self.runtime.copy_device_to_host(
                    counts.max().reshape(1), reason="segment BVH per-row capacity admission"
                )[0]
            )
            if len(counts)
            else 0
        )
        if maximum >= 2**24:
            raise ValueError("Per-row segment limit is 2**24-1")
        self.offsets = exclusive_sum(cp.concatenate((counts, cp.zeros(1, dtype=cp.int32))))
        capacity = cp.maximum(counts, 1) - 1
        for shift in (1, 2, 4, 8, 16):
            capacity |= capacity >> shift
        self.capacity = capacity + 1
        sizes = 2 * self.capacity - 1
        self.nodes = exclusive_sum(
            cp.concatenate((sizes.astype(cp.int64), cp.zeros(1, dtype=cp.int64)))
        )
        self.total = int(
            self.runtime.copy_device_to_host(
                self.nodes[-1:], reason="segment BVH node allocation fence"
            )[0]
        )
        if self.total >= 2**29:
            raise ValueError("Segment BVH four-plane bounds offsets exceed int32 capacity")
        self.nodes = self.nodes.astype(cp.int32)
        self.bounds = cp.empty(
            (4, self.total if hierarchy else 0),
            dtype=cp.float32 if precision == "fp32" else cp.float64,
        )
        if hierarchy and self.total:
            owners = (upper_bound(self.nodes, cp.arange(self.total, dtype=cp.int32)) - 1).astype(
                cp.int32
            )
            depth = (max(1, maximum) - 1).bit_length()
            for stage in range(depth + 1):
                self.run(
                    "row_bvh_build",
                    self.total,
                    (owners, self.nodes, self.capacity, self.offsets, *self.coords, self.bounds),
                    (self.total, stage),
                )
        self.bytes = sum(
            a.nbytes for a in (*self.coords, self.offsets, self.nodes, self.capacity, self.bounds)
        )

    def run(self, name, count, arrays, scalars):
        from vibespatial.cuda._runtime import KERNEL_PARAM_I32, KERNEL_PARAM_PTR

        kernel = self.kernels[name]
        grid, block = self.runtime.launch_config(kernel, count)
        self.runtime.launch(
            kernel,
            grid=grid,
            block=block,
            params=(
                tuple(self.runtime.pointer(a) for a in arrays) + scalars,
                (KERNEL_PARAM_PTR,) * len(arrays) + (KERNEL_PARAM_I32,) * len(scalars),
            ),
        )

    def family(self, family):
        b = self.state.families[family]
        go = b.geometry_offsets
        po = (
            b.part_offsets
            if family in (GeometryFamily.MULTILINESTRING, GeometryFamily.MULTIPOLYGON)
            else go
        )
        ro = (
            b.ring_offsets
            if family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON)
            else go
        )
        return self.state.tags, self.state.family_row_offsets, go, po, ro, b.x, b.y


class SegmentBVHRefiner:
    def __init__(self, geometry, precision_plan, *, bounds_plan):

        from vibespatial.runtime.precision import PrecisionMode

        if precision_plan.compute_precision is not PrecisionMode.FP64:
            raise ValueError("Exact nearest refinement requires its FP64 precision plan")
        self.tree = RowSegmentBVH(geometry, bounds_plan=bounds_plan)
        self.bounds_plan = bounds_plan
        self.query = None

    def for_query(self, geometry, *, context):
        if self.query is None or self.query.geometry is not geometry:
            self.query = RowSegmentBVH(geometry, bounds_plan=self.bounds_plan, hierarchy=False)
        self.context = context
        return self.refine

    def refine(self, left, right, active):
        import cupy as cp

        q, t = self.query, self.tree
        result = cp.full(len(left), cp.inf, dtype=cp.float64)
        for lf in set(q.geometry.families) & set(KINDS):
            for rf in set(t.geometry.families) & set(KINDS):
                if not len(left):
                    continue
                t.run(
                    "cooperative_segment_refine",
                    len(left) * 32,
                    (
                        left,
                        right,
                        active,
                        *q.family(lf),
                        *t.family(rf),
                        q.row_bounds,
                        t.row_bounds,
                        q.offsets,
                        *q.coords,
                        t.offsets,
                        *t.coords,
                        t.nodes,
                        t.capacity,
                        t.bounds,
                        result,
                    ),
                    (len(left), t.total, FAMILY_TAGS[lf], KINDS[lf], FAMILY_TAGS[rf], KINDS[rf]),
                )
        result = cp.sqrt(result)
        if set(q.geometry.families) - set(KINDS) or set(t.geometry.families) - set(KINDS):
            from types import SimpleNamespace

            from vibespatial.spatial.nearest import _compute_mixed_distances_gpu_device
            from vibespatial.spatial.point_distance import compute_distance_center_device

            qtags, ttags = q.state.tags[left], t.state.tags[right]
            qnonpoint = cp.zeros(len(left), dtype=cp.bool_)
            tnonpoint = cp.zeros(len(left), dtype=cp.bool_)
            for family in KINDS:
                qnonpoint |= qtags == FAMILY_TAGS[family]
                tnonpoint |= ttags == FAMILY_TAGS[family]
            nonpoint = qnonpoint & tnonpoint
            other = _compute_mixed_distances_gpu_device(
                q.geometry,
                t.geometry,
                None,
                None,
                SimpleNamespace(d_left=left, d_right=right, total_pairs=len(left)),
                precision_context=self.context,
                pair_active=active & ~nonpoint,
                center_device=compute_distance_center_device(q.geometry, t.geometry),
            )
            if other is None or other[1]:
                return None
            result = cp.where(nonpoint, result, other[0])
        return result, False


@register_kernel_variant(
    "segment_bvh",
    "gpu-cuda-python",
    kernel_class=KernelClass.COARSE,
    geometry_families=("linestring", "multilinestring", "polygon", "multipolygon"),
    execution_modes=(ExecutionMode.GPU,),
    supports_mixed=True,
    precision_modes=(PrecisionMode.AUTO, PrecisionMode.FP32, PrecisionMode.FP64),
    preferred_residency=Residency.DEVICE,
    tags=("cuda-python", "native-spatial-index"),
)
def segment_bvh(geometry, *, precision_plan):
    """Build per-row segment bounds; exact metric consumers use a separate FP64 plan."""
    return RowSegmentBVH(geometry, bounds_plan=precision_plan)
