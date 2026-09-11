"""Cooperative per-geometry segment BVH experiment; no public dispatch changes."""
from __future__ import annotations

from experimental_segment_bvh_kernels import SOURCE

from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import FAMILY_TAGS

KINDS = {GeometryFamily.LINESTRING:0, GeometryFamily.MULTILINESTRING:1,
         GeometryFamily.POLYGON:2, GeometryFamily.MULTIPOLYGON:3}


class RowSegmentBVH:
    def __init__(self, geometry, *, hierarchy=True):
        import cupy as cp

        from vibespatial.cuda._runtime import get_cuda_runtime, make_kernel_cache_key
        from vibespatial.cuda.cccl_primitives import exclusive_sum, sort_pairs, upper_bound
        from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device
        from vibespatial.runtime.residency import Residency, TransferTrigger
        from vibespatial.spatial.segment_distance_kernels import _SEGMENT_DISTANCE_KERNEL_SOURCE
        from vibespatial.spatial.segment_primitives import _extract_segments_gpu

        if not set(geometry.families) <= set(KINDS):
            raise NotImplementedError("Segment experiment admits the four non-point families")
        geometry.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                         reason="experimental segment hierarchy consumes owned device buffers")
        self.geometry, self.state = geometry, geometry._ensure_device_state()
        self.row_bounds = cp.ascontiguousarray(compute_geometry_bounds_device(geometry))
        self.runtime = get_cuda_runtime()
        source = _SEGMENT_DISTANCE_KERNEL_SOURCE + SOURCE
        self.kernels = self.runtime.compile_kernels(cache_key=make_kernel_cache_key("segment-bvh-experiment-fp64",source),
            source=source,kernel_names=("row_bvh_build","cooperative_segment_refine"),options=("--fmad=false",))
        segments = _extract_segments_gpu(geometry)
        rows = cp.asarray(segments.row_indices).astype(cp.int32,copy=False)
        order = sort_pairs(rows,cp.arange(segments.count,dtype=cp.int32)).values
        self.coords = tuple(cp.ascontiguousarray(cp.asarray(getattr(segments,k))[order]) for k in ("x0","y0","x1","y1"))
        counts = cp.bincount(rows,minlength=geometry.row_count).astype(cp.int32)
        if len(counts) and int(counts.max().item()) >= 2**24:
            raise ValueError("Experimental per-row segment limit is 2**24-1")
        self.offsets = exclusive_sum(cp.concatenate((counts,cp.zeros(1,dtype=cp.int32))))
        capacity = cp.maximum(counts,1)-1
        for shift in (1,2,4,8,16):
            capacity |= capacity >> shift
        self.capacity = capacity+1
        sizes = 2*self.capacity-1
        self.nodes = exclusive_sum(cp.concatenate((sizes,cp.zeros(1,dtype=cp.int32))))
        self.total = int(self.nodes[-1].item())
        self.bounds = cp.empty((4,self.total if hierarchy else 0),dtype=cp.float32)
        if hierarchy and self.total:
            owners = (upper_bound(self.nodes,cp.arange(self.total,dtype=cp.int32))-1).astype(cp.int32)
            depth = int(self.capacity.max().item()).bit_length()-1
            for stage in range(depth+1):
                self.run("row_bvh_build",self.total,(owners,self.nodes,self.capacity,self.offsets,*self.coords,self.bounds),
                         (self.total,stage))
        self.bytes = sum(a.nbytes for a in (*self.coords,self.offsets,self.nodes,self.capacity,self.bounds))

    def run(self,name,count,arrays,scalars):
        from vibespatial.cuda._runtime import KERNEL_PARAM_I32, KERNEL_PARAM_PTR

        kernel = self.kernels[name]
        grid,block = self.runtime.launch_config(kernel,count)
        self.runtime.launch(kernel,grid=grid,block=block,
            params=(tuple(self.runtime.pointer(a) for a in arrays)+scalars,
                    (KERNEL_PARAM_PTR,)*len(arrays)+(KERNEL_PARAM_I32,)*len(scalars)))

    def family(self,family):
        b=self.state.families[family]
        go=b.geometry_offsets
        po=b.part_offsets if family in (GeometryFamily.MULTILINESTRING,GeometryFamily.MULTIPOLYGON) else go
        ro=b.ring_offsets if family in (GeometryFamily.POLYGON,GeometryFamily.MULTIPOLYGON) else go
        return self.state.tags,self.state.family_row_offsets,go,po,ro,b.x,b.y


class SegmentBVHRefiner:
    def __init__(self,geometry,precision_plan,*,hierarchy=True,tile_segments=0):
        import cupy as cp

        from vibespatial.runtime.precision import PrecisionMode

        if precision_plan.compute_precision is not PrecisionMode.FP64:
            raise ValueError("Reference refiner requires its explicit FP64 plan")
        self.tree=RowSegmentBVH(geometry,hierarchy=hierarchy)
        self.hierarchy=hierarchy
        self.tile_segments=tile_segments
        self.query=None
        self.work=cp.zeros(2,dtype=cp.uint64)

    def for_query(self,geometry):
        if self.query is None or self.query.geometry is not geometry:
            self.query=RowSegmentBVH(geometry,hierarchy=False)
        self.work.fill(0)
        return self.refine

    def refine(self,left,right,active):
        import cupy as cp

        q,t=self.query,self.tree
        result=cp.full(len(left),cp.inf,dtype=cp.float64)
        offsets=cp.empty(0,dtype=cp.int64)
        phases=[(2,len(left))]
        if self.tile_segments:
            from vibespatial.cuda.cccl_primitives import exclusive_sum

            counts=cp.where(active,(cp.maximum(q.offsets[left+1]-q.offsets[left]-32,0)+self.tile_segments-1)//self.tile_segments,0).astype(cp.int64)
            offsets=exclusive_sum(cp.concatenate((counts,cp.zeros(1,dtype=cp.int64))))
            total=int(offsets[-1].item())
            phases=[(0,len(left)),(1,total)]
        for lf in q.geometry.families:
            for rf in t.geometry.families:
                for phase,total in phases:
                    if not total:
                        continue
                    t.run("cooperative_segment_refine",total*32,
                        (left,right,active,*q.family(lf),*t.family(rf),q.row_bounds,t.row_bounds,q.offsets,*q.coords,t.offsets,*t.coords,
                         t.nodes,t.capacity,t.bounds,result,self.work,offsets),
                        (len(left),t.total,FAMILY_TAGS[lf],KINDS[lf],FAMILY_TAGS[rf],KINDS[rf],int(self.hierarchy),phase,self.tile_segments,total))
        return cp.sqrt(result),False
