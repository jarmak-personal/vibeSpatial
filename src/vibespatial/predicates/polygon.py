from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vibespatial.api._native_state import NativeStreamReadiness
from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    cuda_stream_identity,
    get_cuda_runtime,
)
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import FAMILY_TAGS, OwnedGeometryArray
from vibespatial.predicates.polygon_kernels import (
    _POLYGON_PREDICATES_KERNEL_NAMES,
    _POLYGON_PREDICATES_KERNEL_SOURCE,
)
from vibespatial.runtime.residency import Residency

# ---------------------------------------------------------------------------
# DE-9IM bitmask layout
# ---------------------------------------------------------------------------
#   bit 0: II  (Interior ∩ Interior)
#   bit 1: IB  (Interior ∩ Boundary)
#   bit 2: IE  (Interior ∩ Exterior)
#   bit 3: BI  (Boundary ∩ Interior)
#   bit 4: BB  (Boundary ∩ Boundary)
#   bit 5: BE  (Boundary ∩ Exterior)
#   bit 6: EI  (Exterior ∩ Interior)
#   bit 7: EB  (Exterior ∩ Boundary)
#   bit 8: EE  (Exterior ∩ Exterior)

DE9IM_II = 1 << 0
DE9IM_IB = 1 << 1
DE9IM_IE = 1 << 2
DE9IM_BI = 1 << 3
DE9IM_BB = 1 << 4
DE9IM_BE = 1 << 5
DE9IM_EI = 1 << 6
DE9IM_EB = 1 << 7
DE9IM_EE = 1 << 8

# The automatic lowering is intentionally narrower than the primitive.  The
# complete-stage experiment established wins from 10K source rows through 10M
# for targets with 4-64 exterior vertices.  Shapes outside that measured region
# retain the general exact polygon predicate until evidence expands it.
_CONVEX_GROUPED_MIN_SOURCE_ROWS = 10_000
_CONVEX_GROUPED_MAX_MASK_COORDINATES = 65  # 64 vertices plus exact closure
_CONVEX_GROUPED_MAX_AVERAGE_SOURCE_COORDINATES = 8.0
_CONVEX_GROUPED_MAX_SOURCE_COORDINATES = 65
_CONVEX_GROUPED_MIN_LOGICAL_DENSITY = 0.5

request_nvrtc_warmup(
    [
        ("polygon-predicates", _POLYGON_PREDICATES_KERNEL_SOURCE, _POLYGON_PREDICATES_KERNEL_NAMES),
    ]
)


def _polygon_predicates_kernels():
    return compile_kernel_group(
        "polygon-predicates", _POLYGON_PREDICATES_KERNEL_SOURCE, _POLYGON_PREDICATES_KERNEL_NAMES
    )


@dataclass(frozen=True)
class NativeConvexMaskCertificate:
    """Typed device-capable certificate for one convex polygonal row."""

    source_token: str
    owner_id: int
    device_state_id: int
    row_mapping_id: int
    family: GeometryFamily
    family_row: int
    coordinate_count: int
    values: Any
    convex_simple: bool
    residency: Residency
    readiness: NativeStreamReadiness

    def validate(
        self,
        owner: OwnedGeometryArray,
        state: Any,
        buffer: Any,
        *,
        family: GeometryFamily,
        family_row: int,
    ) -> None:
        expected_token = f"owned-device-state:{id(state)}"
        if (
            self.source_token != expected_token
            or self.owner_id != id(owner)
            or self.device_state_id != id(state)
            or self.row_mapping_id != id(state.family_row_offsets)
            or self.family is not family
            or self.family_row != int(family_row)
            or self.coordinate_count != int(buffer.x.size)
            or self.residency is not Residency.DEVICE
        ):
            raise RuntimeError("stale convex polygon certificate lineage")


@dataclass(frozen=True)
class NativeSimplePolygonalSourceCertificate:
    """Typed family-row certificate for simple positive-area source rings."""

    source_token: str
    owner_id: int
    device_state_id: int
    row_mapping_id: int
    family: GeometryFamily
    family_row_count: int
    coordinate_count: int
    values: Any
    all_certified: bool
    residency: Residency
    readiness: NativeStreamReadiness

    def validate(
        self,
        owner: OwnedGeometryArray,
        state: Any,
        buffer: Any,
        *,
        family: GeometryFamily,
    ) -> None:
        expected_token = f"owned-device-state:{id(state)}"
        if (
            self.source_token != expected_token
            or self.owner_id != id(owner)
            or self.device_state_id != id(state)
            or self.row_mapping_id != id(state.family_row_offsets)
            or self.family is not family
            or self.family_row_count != int(buffer.empty_mask.size)
            or self.coordinate_count != int(buffer.x.size)
            or self.residency is not Residency.DEVICE
        ):
            raise RuntimeError("stale simple polygonal source certificate lineage")


def _record_certificate_readiness() -> NativeStreamReadiness:
    import cupy as cp

    stream = cp.cuda.get_current_stream()
    event = cp.cuda.Event(disable_timing=True)
    event.record(stream)
    return NativeStreamReadiness(stream=stream, event=event, ready=False)


def _wait_for_certificate(readiness: NativeStreamReadiness) -> None:
    if readiness.ready or readiness.event is None:
        return
    import cupy as cp

    consumer = cp.cuda.get_current_stream()
    if readiness.stream is not None and (
        cuda_stream_identity(readiness.stream) == cuda_stream_identity(consumer)
    ):
        return
    consumer.wait_event(readiness.event)


def certify_single_polygonal_convex_no_holes_gpu(
    mask_owned: OwnedGeometryArray,
    *,
    mask_family: GeometryFamily,
    family_row: int = 0,
) -> NativeConvexMaskCertificate:
    """Return a cached exact-sign convex/no-hole certificate.

    The one-byte device result is an explicit planning packet.  It crosses the
    boundary once per immutable OwnedGeometryArray generation and is then
    reused by every public predicate consumer of that mask.
    """
    state = mask_owned._ensure_device_state(preserve_indexed_view=True)
    buffer = state.families[mask_family]
    key = ("convex-mask", mask_family, int(family_row))
    cached = state.polygon_certificates.get(key)
    if cached is not None:
        if not isinstance(cached, NativeConvexMaskCertificate):
            raise RuntimeError("convex polygon certificate cache type mismatch")
        cached.validate(
            mask_owned,
            state,
            buffer,
            family=mask_family,
            family_row=family_row,
        )
        _wait_for_certificate(cached.readiness)
        return cached

    runtime = get_cuda_runtime()
    admission = runtime.admit_device_memory(
        stage="polygon.convex_mask_certificate",
        required_bytes=1,
        requested_units=1,
    )
    if not admission.admitted:
        raise MemoryError(
            "convex mask certificate requires one device byte with "
            f"{admission.remaining_bytes} available"
        )
    ptr = runtime.pointer
    d_out = runtime.allocate((1,), np.uint8)
    args = [ptr(buffer.geometry_offsets)]
    types = [KERNEL_PARAM_PTR]
    if mask_family is GeometryFamily.MULTIPOLYGON:
        args.append(ptr(buffer.part_offsets))
        types.append(KERNEL_PARAM_PTR)
    args.extend(
        [
            ptr(buffer.ring_offsets),
            ptr(buffer.empty_mask),
            ptr(buffer.x),
            ptr(buffer.y),
            int(family_row),
            ptr(d_out),
        ]
    )
    types.extend(
        [
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR,
        ]
    )
    kernel_name = (
        "certify_single_polygon_convex_no_holes"
        if mask_family is GeometryFamily.POLYGON
        else "certify_single_multipolygon_convex_no_holes"
    )
    runtime.launch(
        _polygon_predicates_kernels()[kernel_name],
        grid=(1, 1, 1),
        block=(1, 1, 1),
        params=(tuple(args), tuple(types)),
    )
    readiness = _record_certificate_readiness()
    host = runtime.copy_device_to_host(
        d_out,
        reason="polygon predicate convex-mask certification planning packet",
    )
    certificate = NativeConvexMaskCertificate(
        source_token=f"owned-device-state:{id(state)}",
        owner_id=id(mask_owned),
        device_state_id=id(state),
        row_mapping_id=id(state.family_row_offsets),
        family=mask_family,
        family_row=int(family_row),
        coordinate_count=int(buffer.x.size),
        values=d_out,
        convex_simple=bool(np.asarray(host, dtype=np.uint8)[0]),
        residency=Residency.DEVICE,
        readiness=readiness,
    )
    state.polygon_certificates[key] = certificate
    return certificate


def certify_polygonal_sources_simple_no_holes_gpu(
    source_owned: OwnedGeometryArray,
    *,
    source_family: GeometryFamily,
) -> NativeSimplePolygonalSourceCertificate:
    """Certify every physical source row for the convex vertex theorem.

    The sufficient source domain is deliberately narrow: one simple,
    non-empty, positive-area ring (and one part for MultiPolygon).  Anything
    else declines the complete-batch fast path and retains the general exact
    predicate.  This makes invalid geometries fail closed without making a
    vertex-only claim about arbitrary OGC topology.
    """
    import cupy as cp

    state = source_owned._ensure_device_state(preserve_indexed_view=True)
    buffer = state.families[source_family]
    family_row_count = int(buffer.empty_mask.size)
    key = ("simple-source-no-holes", source_family, -1)
    cached = state.polygon_certificates.get(key)
    if cached is not None:
        if not isinstance(cached, NativeSimplePolygonalSourceCertificate):
            raise RuntimeError("simple polygonal source certificate cache type mismatch")
        cached.validate(source_owned, state, buffer, family=source_family)
        _wait_for_certificate(cached.readiness)
        return cached

    required_bytes = family_row_count + np.dtype(np.int32).itemsize
    runtime = get_cuda_runtime()
    admission = runtime.admit_device_memory(
        stage="polygon.simple_source_certificate",
        required_bytes=required_bytes,
        requested_units=family_row_count,
    )
    if not admission.admitted:
        raise MemoryError(
            "simple polygonal source certificate requires "
            f"{required_bytes} device bytes with {admission.remaining_bytes} available"
        )
    d_rows = cp.empty(family_row_count, dtype=cp.bool_)
    d_all = cp.ones(1, dtype=cp.int32)
    if family_row_count:
        ptr = runtime.pointer
        args = [ptr(buffer.geometry_offsets)]
        types = [KERNEL_PARAM_PTR]
        if source_family is GeometryFamily.MULTIPOLYGON:
            args.append(ptr(buffer.part_offsets))
            types.append(KERNEL_PARAM_PTR)
        args.extend(
            [
                ptr(buffer.ring_offsets),
                ptr(buffer.empty_mask),
                ptr(buffer.x),
                ptr(buffer.y),
                family_row_count,
                ptr(d_rows),
                ptr(d_all),
            ]
        )
        types.extend(
            [
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
            ]
        )
        kernel_name = (
            "certify_polygon_sources_simple_no_holes"
            if source_family is GeometryFamily.POLYGON
            else "certify_multipolygon_sources_simple_no_holes"
        )
        kernel = _polygon_predicates_kernels()[kernel_name]
        grid, block = runtime.launch_config(kernel, family_row_count)
        runtime.launch(
            kernel,
            grid=grid,
            block=block,
            params=(tuple(args), tuple(types)),
        )
    readiness = _record_certificate_readiness()
    host = runtime.copy_device_to_host(
        d_all,
        reason="polygon predicate simple-source certification planning packet",
    )
    certificate = NativeSimplePolygonalSourceCertificate(
        source_token=f"owned-device-state:{id(state)}",
        owner_id=id(source_owned),
        device_state_id=id(state),
        row_mapping_id=id(state.family_row_offsets),
        family=source_family,
        family_row_count=family_row_count,
        coordinate_count=int(buffer.x.size),
        values=d_rows,
        all_certified=bool(np.asarray(host, dtype=np.int32)[0]),
        residency=Residency.DEVICE,
        readiness=readiness,
    )
    state.polygon_certificates[key] = certificate
    return certificate


def _polygonal_coordinate_offsets_device(buffer: Any, family: GeometryFamily):
    """Resolve one authoritative int64 coordinate span per family row."""
    import cupy as cp

    offset_count = int(buffer.empty_mask.size) + 1
    d_offsets = cp.empty(offset_count, dtype=cp.int64)
    if offset_count == 0:
        return d_offsets
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    args = [ptr(buffer.geometry_offsets)]
    types = [KERNEL_PARAM_PTR]
    if family is GeometryFamily.MULTIPOLYGON:
        args.append(ptr(buffer.part_offsets))
        types.append(KERNEL_PARAM_PTR)
    args.extend([ptr(buffer.ring_offsets), ptr(d_offsets), offset_count])
    types.extend([KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32])
    kernel_name = (
        "polygon_coordinate_offsets_i64"
        if family is GeometryFamily.POLYGON
        else "multipolygon_coordinate_offsets_i64"
    )
    kernel = _polygon_predicates_kernels()[kernel_name]
    grid, block = runtime.launch_config(kernel, offset_count)
    runtime.launch(
        kernel,
        grid=grid,
        block=block,
        params=(tuple(args), tuple(types)),
    )
    return d_offsets


def compute_polygonal_covered_by_single_convex_grouped_gpu(
    query_owned: OwnedGeometryArray,
    mask_owned: OwnedGeometryArray,
    *,
    query_family: GeometryFamily,
    mask_family: GeometryFamily,
):
    """Classify all family coordinates, then reduce by source geometry.

    This is the measured direct/broadcast convex-containment shape.  It emits
    no candidate relation and constructs no point geometry or dense group-code
    vector.  The authoritative family coordinate offsets are consumed directly
    by the bounded segmented ``ALL`` reducer.
    """
    import cupy as cp

    # The measured carrier is dense and physical.  Indexed source views may
    # expose a small logical subset over an arbitrarily larger physical base;
    # certifying the base would both amplify work and return rows outside the
    # active logical selection.  The general predicate already handles these
    # views exactly, so keep them outside this selector until an active-row
    # certificate/reducer is measured.
    if query_owned.is_indexed_view or mask_owned.is_indexed_view:
        return None

    query_state = query_owned._ensure_device_state(preserve_indexed_view=True)
    query_buffer = query_state.families[query_family]
    family_row_count = int(query_buffer.empty_mask.size)
    logical_row_count = int(query_owned.row_count)
    if logical_row_count < _CONVEX_GROUPED_MIN_SOURCE_ROWS or family_row_count == 0:
        return None

    physical_density = family_row_count / logical_row_count
    if physical_density < _CONVEX_GROUPED_MIN_LOGICAL_DENSITY:
        return None

    coordinate_count = int(query_buffer.x.size)
    average_source_coordinates = coordinate_count / family_row_count
    if average_source_coordinates > _CONVEX_GROUPED_MAX_AVERAGE_SOURCE_COORDINATES:
        return None
    fixed_size = query_buffer.fixed_size
    max_source_coordinates = (
        fixed_size.max_coord_count_per_row
        if fixed_size is not None
        else query_buffer.dense_single_ring_width
    )
    if (
        max_source_coordinates is None
        or int(max_source_coordinates) > _CONVEX_GROUPED_MAX_SOURCE_COORDINATES
    ):
        return None

    mask_state = mask_owned._ensure_device_state(preserve_indexed_view=True)
    mask_buffer = mask_state.families[mask_family]
    mask_coordinate_count = int(mask_buffer.x.size)
    if mask_coordinate_count > _CONVEX_GROUPED_MAX_MASK_COORDINATES:
        return None

    offset_bytes = (family_row_count + 1) * np.dtype(np.int64).itemsize
    mask_bounds_bytes = (
        0
        if mask_buffer.bounds is not None
        else int(mask_buffer.empty_mask.size) * 4 * np.dtype(np.float64).itemsize
    )
    required_bytes = (
        coordinate_count  # raw coordinate classifications
        + family_row_count  # simple-source row certificates
        + np.dtype(np.int32).itemsize  # simple-source aggregate
        + family_row_count  # grouped boolean output
        + family_row_count * np.dtype(np.int32).itemsize  # dense group IDs
        + logical_row_count  # mapped public result
        + offset_bytes
        + mask_bounds_bytes
        + 1  # convex-mask certificate
    )
    admission = get_cuda_runtime().admit_device_memory(
        stage="polygon-convex-grouped-containment",
        required_bytes=required_bytes,
        requested_units=family_row_count,
    )
    if not admission.admitted:
        raise MemoryError(
            "convex grouped containment requires complete-stage admission for "
            f"{required_bytes} device bytes with {admission.remaining_bytes} available"
        )

    certificate = certify_single_polygonal_convex_no_holes_gpu(
        mask_owned,
        mask_family=mask_family,
    )
    if not certificate.convex_simple:
        return None

    source_certificate = certify_polygonal_sources_simple_no_holes_gpu(
        query_owned,
        source_family=query_family,
    )
    if not source_certificate.all_certified:
        return None

    d_offsets = _polygonal_coordinate_offsets_device(query_buffer, query_family)
    if int(d_offsets.size) != family_row_count + 1:
        return None

    from vibespatial.kernels.predicates.point_in_polygon import (
        _wrap_device_result_with_keepalive,
        classify_point_coordinates_single_region_device,
    )
    from vibespatial.runtime.adaptive import get_cached_snapshot

    compute_type = (
        "double"
        if get_cached_snapshot().device_profile.favors_native_fp64
        else "float"
    )
    d_vertex_covered = classify_point_coordinates_single_region_device(
        query_buffer.x,
        query_buffer.y,
        mask_owned,
        region_family=mask_family,
        compute_type=compute_type,
    )
    from vibespatial.api._native_grouped import NativeGrouped

    reduction = NativeGrouped.reduce_boolean_all_from_sorted_offsets(
        d_vertex_covered,
        d_offsets,
        row_count=coordinate_count,
        group_count=family_row_count,
        source_token=source_certificate.source_token,
    )
    d_result = cp.asarray(reduction.values, dtype=cp.bool_)
    return _wrap_device_result_with_keepalive(
        d_result,
        d_vertex_covered,
        d_offsets,
        reduction,
        source_certificate,
        certificate,
    )


_DE9IM_EVAL_KERNEL_NAMES = (
    "transpose_de9im_kernel",
    "evaluate_de9im_kernel",
    "transpose_de9im_grouped_kernel",
    "evaluate_de9im_grouped_kernel",
)

_DE9IM_EVAL_KERNEL_SOURCE = r"""
extern "C" __global__ void transpose_de9im_kernel(
    const unsigned short* __restrict__ masks,
    unsigned short* __restrict__ out,
    int n
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned short m = masks[i];
    unsigned short value = 0;
    value |= (unsigned short)(m & 1u);      // II
    value |= (unsigned short)(m & 256u);    // EE
    if (m & 2u) value |= 8u;                // IB -> BI
    if (m & 8u) value |= 2u;                // BI -> IB
    if (m & 4u) value |= 64u;               // IE -> EI
    if (m & 64u) value |= 4u;               // EI -> IE
    if (m & 32u) value |= 128u;             // BE -> EB
    if (m & 128u) value |= 32u;             // EB -> BE
    value |= (unsigned short)(m & 16u);     // BB
    out[i] = value;
}

extern "C" __global__ void evaluate_de9im_kernel(
    const unsigned short* __restrict__ masks,
    unsigned char* __restrict__ out,
    int n,
    int predicate_code
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned short m = masks[i];
    const unsigned short contact = (unsigned short)(1u | 2u | 8u | 16u);
    bool value = false;
    switch (predicate_code) {
        case 0:  // intersects
            value = (m & contact) != 0u;
            break;
        case 1:  // touches
            value = ((m & (2u | 8u | 16u)) != 0u) && ((m & 1u) == 0u);
            break;
        case 2:  // covers
            value = ((m & contact) != 0u) && ((m & (64u | 128u)) == 0u);
            break;
        case 3:  // covered_by
            value = ((m & contact) != 0u) && ((m & (4u | 32u)) == 0u);
            break;
        case 4:  // contains
            value = ((m & 1u) == 1u) && ((m & (64u | 128u)) == 0u);
            break;
        case 5:  // within
            value = ((m & 1u) == 1u) && ((m & (4u | 32u)) == 0u);
            break;
        case 6:  // overlaps
            value = ((m & (1u | 4u | 64u)) == (1u | 4u | 64u));
            break;
        case 7:  // disjoint
            value = ((m & contact) == 0u);
            break;
        case 8:  // contains_properly
            value = ((m & 1u) == 1u) && ((m & (64u | 128u | 16u)) == 0u);
            break;
        case 9:  // equals
            value = ((m & 1u) == 1u) && ((m & (4u | 32u | 64u | 128u)) == 0u);
            break;
        default:
            value = false;
            break;
    }
    out[i] = value ? 1u : 0u;
}

extern "C" __global__ void transpose_de9im_grouped_kernel(
    unsigned short* masks,
    const long long* source_offset,
    const int* logical_count
) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int offset = (int)source_offset[0];
    const int count = logical_count[0];
    for (int local_i = lane; local_i < count; local_i += stride) {
        const int i = offset + local_i;
        const unsigned short m = masks[i];
        unsigned short value = 0;
        value |= (unsigned short)(m & 1u);
        value |= (unsigned short)(m & 256u);
        if (m & 2u) value |= 8u;
        if (m & 8u) value |= 2u;
        if (m & 4u) value |= 64u;
        if (m & 64u) value |= 4u;
        if (m & 32u) value |= 128u;
        if (m & 128u) value |= 32u;
        value |= (unsigned short)(m & 16u);
        masks[i] = value;
    }
}

extern "C" __global__ void evaluate_de9im_grouped_kernel(
    const unsigned short* masks,
    unsigned char* out,
    const long long* source_offset,
    const int* logical_count,
    int predicate_code
) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int offset = (int)source_offset[0];
    const int count = logical_count[0];
    for (int local_i = lane; local_i < count; local_i += stride) {
        const int i = offset + local_i;
        const unsigned short m = masks[i];
        const unsigned short contact = (unsigned short)(1u | 2u | 8u | 16u);
        bool value = false;
        switch (predicate_code) {
            case 0: value = (m & contact) != 0u; break;
            case 1: value = ((m & (2u | 8u | 16u)) != 0u) && ((m & 1u) == 0u); break;
            case 2: value = ((m & contact) != 0u) && ((m & (64u | 128u)) == 0u); break;
            case 3: value = ((m & contact) != 0u) && ((m & (4u | 32u)) == 0u); break;
            case 4: value = ((m & 1u) == 1u) && ((m & (64u | 128u)) == 0u); break;
            case 5: value = ((m & 1u) == 1u) && ((m & (4u | 32u)) == 0u); break;
            case 6: value = ((m & (1u | 4u | 64u)) == (1u | 4u | 64u)); break;
            case 7: value = (m & contact) == 0u; break;
            case 8: value = ((m & 1u) == 1u) && ((m & (64u | 128u | 16u)) == 0u); break;
            case 9: value = ((m & 1u) == 1u) && ((m & (4u | 32u | 64u | 128u)) == 0u); break;
            default: value = false; break;
        }
        out[i] = value ? 1u : 0u;
    }
}
"""

request_nvrtc_warmup(
    [
        ("polygon-de9im-eval", _DE9IM_EVAL_KERNEL_SOURCE, _DE9IM_EVAL_KERNEL_NAMES),
    ]
)


def _de9im_eval_kernels():
    return compile_kernel_group(
        "polygon-de9im-eval",
        _DE9IM_EVAL_KERNEL_SOURCE,
        _DE9IM_EVAL_KERNEL_NAMES,
    )


def evaluate_de9im_grouped_device(
    d_masks,
    predicate: str,
    *,
    source_offset,
    logical_count,
    launch_capacity: int,
    out,
):
    """Evaluate one device-counted DE-9IM span into shared tile output."""
    predicate_code = _PREDICATE_DEVICE_CODES.get(predicate)
    if predicate_code is None:
        raise ValueError(f"unsupported grouped DE-9IM predicate: {predicate}")
    runtime = get_cuda_runtime()
    kernel = _de9im_eval_kernels()["evaluate_de9im_grouped_kernel"]
    ptr = runtime.pointer
    grid, block = runtime.launch_config(kernel, int(launch_capacity))
    runtime.launch(
        kernel,
        grid=grid,
        block=block,
        params=(
            (
                ptr(d_masks),
                ptr(out),
                ptr(source_offset),
                ptr(logical_count),
                int(predicate_code),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        ),
    )
    return out


# ---------------------------------------------------------------------------
# Predicate evaluation from DE-9IM bitmask
# ---------------------------------------------------------------------------
# Each predicate is defined by a (required_set, required_unset) pair of
# bitmasks.  The predicate is TRUE when:
#   (mask & required_set) == required_set  AND  (mask & required_unset) == 0

_PREDICATE_RULES: dict[str, tuple[int, int]] = {
    # intersects: at least one of II, IB, BI, BB is set
    "intersects": (0, 0),  # handled specially below
    # contains: II set, EI and EB unset
    "contains": (DE9IM_II, DE9IM_EI | DE9IM_EB),
    # within: II set, IE and BE unset
    "within": (DE9IM_II, DE9IM_IE | DE9IM_BE),
    # covers: at least one of II/IB/BI/BB set, EI and EB unset
    "covers": (0, DE9IM_EI | DE9IM_EB),  # handled specially
    # covered_by: at least one of II/IB/BI/BB set, IE and BE unset
    "covered_by": (0, DE9IM_IE | DE9IM_BE),  # handled specially
    # touches: II unset, at least one of IB/BI/BB set
    "touches": (0, 0),  # handled specially
    # overlaps (same-dim = 2D polygon): II, IE, EI all set
    "overlaps": (DE9IM_II | DE9IM_IE | DE9IM_EI, 0),
    # disjoint: II, IB, BI, BB all unset
    "disjoint": (0, DE9IM_II | DE9IM_IB | DE9IM_BI | DE9IM_BB),
    # contains_properly: contains (II set, EI/EB unset) AND BB unset
    "contains_properly": (DE9IM_II, DE9IM_EI | DE9IM_EB | DE9IM_BB),
    # equals: II set, IE/BE/EI/EB all unset (T*F**FFF*)
    "equals": (DE9IM_II, DE9IM_IE | DE9IM_BE | DE9IM_EI | DE9IM_EB),
}

_CONTACT_MASK = DE9IM_II | DE9IM_IB | DE9IM_BI | DE9IM_BB
_PREDICATE_DEVICE_CODES: dict[str, int] = {
    "intersects": 0,
    "touches": 1,
    "covers": 2,
    "covered_by": 3,
    "contains": 4,
    "within": 5,
    "overlaps": 6,
    "disjoint": 7,
    "contains_properly": 8,
    "equals": 9,
}


def evaluate_predicate_from_de9im(masks: np.ndarray, predicate: str) -> np.ndarray:
    """Evaluate a spatial predicate from DE-9IM bitmasks.

    Parameters
    ----------
    masks : uint16 array of DE-9IM bitmasks
    predicate : one of the supported predicate names

    Returns
    -------
    bool array
    """
    m = masks.astype(np.uint16, copy=False)

    if predicate == "intersects":
        return (m & _CONTACT_MASK).astype(bool)

    if predicate == "touches":
        has_contact = (m & (DE9IM_IB | DE9IM_BI | DE9IM_BB)).astype(bool)
        no_ii = ~(m & DE9IM_II).astype(bool)
        return has_contact & no_ii

    if predicate == "covers":
        has_contact = (m & _CONTACT_MASK).astype(bool)
        no_ext = ~(m & (DE9IM_EI | DE9IM_EB)).astype(bool)
        return has_contact & no_ext

    if predicate == "covered_by":
        has_contact = (m & _CONTACT_MASK).astype(bool)
        no_ext = ~(m & (DE9IM_IE | DE9IM_BE)).astype(bool)
        return has_contact & no_ext

    rule = _PREDICATE_RULES.get(predicate)
    if rule is None:
        raise ValueError(f"Unsupported predicate for DE-9IM evaluation: {predicate}")

    required_set, required_unset = rule
    result = np.ones(len(m), dtype=bool)
    if required_set:
        result &= (m & required_set) == required_set
    if required_unset:
        result &= (m & required_unset) == 0
    return result


def evaluate_predicate_from_de9im_device(d_masks: object, predicate: str) -> object:
    """Evaluate a spatial predicate from device-resident DE-9IM bitmasks."""
    import cupy as cp

    predicate_code = _PREDICATE_DEVICE_CODES.get(predicate)
    if predicate_code is None:
        raise ValueError(f"Unsupported predicate for DE-9IM evaluation: {predicate}")

    d_masks = cp.asarray(d_masks)
    if d_masks.dtype != cp.uint16:
        d_masks = d_masks.astype(cp.uint16, copy=False)
    n = int(d_masks.size)
    d_out = cp.empty(n, dtype=cp.bool_)
    if n == 0:
        return d_out

    runtime = get_cuda_runtime()
    kernels = _de9im_eval_kernels()
    kernel = kernels["evaluate_de9im_kernel"]
    ptr = runtime.pointer
    grid, block = runtime.launch_config(kernel, n)
    runtime.launch(
        kernel,
        grid=grid,
        block=block,
        params=(
            (ptr(d_masks), ptr(d_out), n, int(predicate_code)),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32, KERNEL_PARAM_I32),
        ),
    )
    return d_out


def compute_rect_bounds_polygon_mask_predicates_gpu(
    mask_owned: OwnedGeometryArray,
    rect_bounds: object,
    *,
    mask_family: GeometryFamily,
    return_device: bool = False,
) -> tuple[object, object] | tuple[np.ndarray, np.ndarray] | None:
    """Evaluate rectangle-bounds ``intersects`` and ``covered_by`` predicates.

    Physical shape: a device rowset of axis-aligned rectangle bounds against one
    polygon mask row.  The source rectangles do not need to be materialized as
    polygon geometry for this predicate refine; the constructive stage can still
    consume the original source rows for exact output assembly.
    """
    if mask_family is not GeometryFamily.POLYGON:
        return None

    import cupy as cp

    from vibespatial.runtime.residency import Residency, TransferTrigger

    if not (mask_owned.is_indexed_view and mask_owned.residency is Residency.DEVICE):
        mask_owned.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="rectangle-bounds polygon-mask predicate GPU: mask POLYGON",
        )

    mask_state = mask_owned._ensure_device_state(preserve_indexed_view=True)
    mask_buf = mask_state.families.get(GeometryFamily.POLYGON)
    if mask_buf is None or mask_buf.ring_offsets is None:
        return None

    d_bounds = cp.ascontiguousarray(
        cp.asarray(rect_bounds, dtype=cp.float64).reshape(-1, 4),
    )
    row_count = int(d_bounds.shape[0])
    runtime = get_cuda_runtime()
    d_intersects = runtime.allocate((row_count,), np.bool_)
    d_covered_by = runtime.allocate((row_count,), np.bool_)
    if row_count == 0:
        if return_device:
            return d_intersects, d_covered_by
        runtime.free(d_intersects)
        runtime.free(d_covered_by)
        return np.empty(0, dtype=np.bool_), np.empty(0, dtype=np.bool_)

    kernels = _polygon_predicates_kernels()
    kernel = kernels["rect_bounds_polygon_mask_predicates"]
    ptr = runtime.pointer
    params = (
        (
            ptr(d_bounds),
            ptr(mask_state.validity),
            ptr(mask_state.tags),
            ptr(mask_state.family_row_offsets),
            ptr(mask_buf.geometry_offsets),
            ptr(mask_buf.ring_offsets),
            ptr(mask_buf.empty_mask),
            ptr(mask_buf.x),
            ptr(mask_buf.y),
            FAMILY_TAGS[GeometryFamily.POLYGON],
            ptr(d_intersects),
            ptr(d_covered_by),
            row_count,
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
        ),
    )
    grid, block = runtime.launch_config(kernel, row_count)
    runtime.launch(kernel, grid=grid, block=block, params=params)
    if return_device:
        return d_intersects, d_covered_by

    runtime.synchronize()
    h_intersects = np.empty(row_count, dtype=np.bool_)
    h_covered_by = np.empty(row_count, dtype=np.bool_)
    runtime.copy_device_to_host(
        d_intersects,
        h_intersects,
        reason="rectangle-bounds polygon-mask intersects result host export",
    )
    runtime.copy_device_to_host(
        d_covered_by,
        h_covered_by,
        reason="rectangle-bounds polygon-mask covered_by result host export",
    )
    runtime.free(d_intersects)
    runtime.free(d_covered_by)
    return h_intersects, h_covered_by


# ---------------------------------------------------------------------------
# Kernel dispatch
# ---------------------------------------------------------------------------

# Maps (left_family, right_family) → kernel name.
_KERNEL_MAP: dict[tuple[GeometryFamily, GeometryFamily], str] = {
    # Polygon × Polygon
    (GeometryFamily.POLYGON, GeometryFamily.POLYGON): "polygon_polygon_de9im_from_owned",
    (
        GeometryFamily.MULTIPOLYGON,
        GeometryFamily.MULTIPOLYGON,
    ): "multipolygon_multipolygon_de9im_from_owned",
    (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON): "polygon_multipolygon_de9im_from_owned",
    (GeometryFamily.MULTIPOLYGON, GeometryFamily.POLYGON): "polygon_multipolygon_de9im_from_owned",
    # Line × Line
    (GeometryFamily.LINESTRING, GeometryFamily.LINESTRING): "ls_ls_de9im_from_owned",
    (GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING): "ls_mls_de9im_from_owned",
    (GeometryFamily.MULTILINESTRING, GeometryFamily.LINESTRING): "ls_mls_de9im_from_owned",
    (GeometryFamily.MULTILINESTRING, GeometryFamily.MULTILINESTRING): "mls_mls_de9im_from_owned",
    # Line × Polygon
    (GeometryFamily.LINESTRING, GeometryFamily.POLYGON): "ls_pg_de9im_from_owned",
    (GeometryFamily.LINESTRING, GeometryFamily.MULTIPOLYGON): "ls_mpg_de9im_from_owned",
    (GeometryFamily.MULTILINESTRING, GeometryFamily.POLYGON): "mls_pg_de9im_from_owned",
    (GeometryFamily.MULTILINESTRING, GeometryFamily.MULTIPOLYGON): "mls_mpg_de9im_from_owned",
    # Polygon × Line (dispatched by swapping to Line × Polygon)
    (GeometryFamily.POLYGON, GeometryFamily.LINESTRING): "ls_pg_de9im_from_owned",
    (GeometryFamily.POLYGON, GeometryFamily.MULTILINESTRING): "mls_pg_de9im_from_owned",
    (GeometryFamily.MULTIPOLYGON, GeometryFamily.LINESTRING): "ls_mpg_de9im_from_owned",
    (GeometryFamily.MULTIPOLYGON, GeometryFamily.MULTILINESTRING): "mls_mpg_de9im_from_owned",
}

_COVERED_BY_SINGLE_MASK_NO_HOLES_KERNEL_MAP: dict[tuple[GeometryFamily, GeometryFamily], str] = {
    (
        GeometryFamily.POLYGON,
        GeometryFamily.POLYGON,
    ): "polygon_polygon_covered_by_mask_no_holes_coop",
    (
        GeometryFamily.MULTIPOLYGON,
        GeometryFamily.POLYGON,
    ): "multipolygon_polygon_covered_by_mask_no_holes_coop",
    (
        GeometryFamily.POLYGON,
        GeometryFamily.MULTIPOLYGON,
    ): "polygon_multipolygon_covered_by_mask_no_holes_coop",
    (GeometryFamily.MULTIPOLYGON, GeometryFamily.MULTIPOLYGON): (
        "multipolygon_multipolygon_covered_by_mask_no_holes_coop"
    ),
}

_COVERED_BY_PAIR_ROWS_NO_HOLES_KERNEL_MAP: dict[tuple[GeometryFamily, GeometryFamily], str] = {
    (
        GeometryFamily.POLYGON,
        GeometryFamily.POLYGON,
    ): "polygon_polygon_covered_by_pair_rows_no_holes_coop",
    (GeometryFamily.MULTIPOLYGON, GeometryFamily.POLYGON): (
        "multipolygon_polygon_covered_by_pair_rows_no_holes_coop"
    ),
    (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON): (
        "polygon_multipolygon_covered_by_pair_rows_no_holes_coop"
    ),
    (GeometryFamily.MULTIPOLYGON, GeometryFamily.MULTIPOLYGON): (
        "multipolygon_multipolygon_covered_by_pair_rows_no_holes_coop"
    ),
}

_POLYGONAL_INTERSECTS_KERNEL_MAP: dict[tuple[GeometryFamily, GeometryFamily], str] = {
    (GeometryFamily.POLYGON, GeometryFamily.POLYGON): "polygon_polygon_intersects_from_owned",
    (GeometryFamily.MULTIPOLYGON, GeometryFamily.MULTIPOLYGON): (
        "multipolygon_multipolygon_intersects_from_owned"
    ),
    (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON): (
        "polygon_multipolygon_intersects_from_owned"
    ),
}

_POLYGONAL_INTERSECTS_SWAP_PAIRS: dict[
    tuple[GeometryFamily, GeometryFamily],
    tuple[GeometryFamily, GeometryFamily],
] = {
    (GeometryFamily.MULTIPOLYGON, GeometryFamily.POLYGON): (
        GeometryFamily.POLYGON,
        GeometryFamily.MULTIPOLYGON,
    ),
}


_LINE_FAMILIES = frozenset({GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING})
_POLYGON_FAMILIES = frozenset({GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON})

# Pairs that require swapping (A,B) to (B,A) before kernel dispatch.
# The kernel is written with a specific left/right layout, so we swap
# and transpose the DE-9IM result.
_SWAP_PAIRS: dict[tuple[GeometryFamily, GeometryFamily], tuple[GeometryFamily, GeometryFamily]] = {
    # MPG×PG → PG×MPG
    (GeometryFamily.MULTIPOLYGON, GeometryFamily.POLYGON): (
        GeometryFamily.POLYGON,
        GeometryFamily.MULTIPOLYGON,
    ),
    # MLS×LS → LS×MLS
    (GeometryFamily.MULTILINESTRING, GeometryFamily.LINESTRING): (
        GeometryFamily.LINESTRING,
        GeometryFamily.MULTILINESTRING,
    ),
    # PG×LS → LS×PG
    (GeometryFamily.POLYGON, GeometryFamily.LINESTRING): (
        GeometryFamily.LINESTRING,
        GeometryFamily.POLYGON,
    ),
    # PG×MLS → MLS×PG
    (GeometryFamily.POLYGON, GeometryFamily.MULTILINESTRING): (
        GeometryFamily.MULTILINESTRING,
        GeometryFamily.POLYGON,
    ),
    # MPG×LS → LS×MPG
    (GeometryFamily.MULTIPOLYGON, GeometryFamily.LINESTRING): (
        GeometryFamily.LINESTRING,
        GeometryFamily.MULTIPOLYGON,
    ),
    # MPG×MLS → MLS×MPG
    (GeometryFamily.MULTIPOLYGON, GeometryFamily.MULTILINESTRING): (
        GeometryFamily.MULTILINESTRING,
        GeometryFamily.MULTIPOLYGON,
    ),
}


def _build_side_args(ptr, state, buf, family):
    """Build kernel args + types for one side of a DE-9IM kernel call."""
    P = KERNEL_PARAM_PTR
    I32 = KERNEL_PARAM_I32

    args = [
        ptr(state.validity),
        ptr(state.tags),
        ptr(state.family_row_offsets),
        ptr(buf.geometry_offsets),
    ]
    types = [P, P, P, P]

    # Multi-families need part_offsets before ring/coord offsets.
    if family in (GeometryFamily.MULTILINESTRING, GeometryFamily.MULTIPOLYGON):
        args.append(ptr(buf.part_offsets))
        types.append(P)

    # Polygon families need ring_offsets.
    if family in _POLYGON_FAMILIES:
        args.append(ptr(buf.ring_offsets))
        types.append(P)

    args.extend([ptr(buf.empty_mask), ptr(buf.x), ptr(buf.y), FAMILY_TAGS[family]])
    types.extend([P, P, P, I32])
    return args, types


def compute_polygonal_covered_by_single_mask_no_holes_gpu(
    query_owned: OwnedGeometryArray,
    mask_owned: OwnedGeometryArray,
    left_indices: np.ndarray | None = None,
    *,
    query_family: GeometryFamily,
    mask_family: GeometryFamily,
    d_left: object | None = None,
    d_pair_count: object | None = None,
    pair_capacity: int | None = None,
    return_device: bool = False,
) -> np.ndarray | None:
    """Evaluate ``query covered_by mask`` for one polygonal mask on device.

    Convex no-hole masks use a cheaper one-sided proof in the kernel.
    Concave, multipart, and hole-bearing masks fall through to the exact
    polygon DE-9IM device path instead of a host-side capability branch.
    """
    kernel_name = _COVERED_BY_SINGLE_MASK_NO_HOLES_KERNEL_MAP.get((query_family, mask_family))
    if kernel_name is None:
        return None

    from vibespatial.runtime.residency import Residency, TransferTrigger

    if not (query_owned.is_indexed_view and query_owned.residency is Residency.DEVICE):
        query_owned.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason=f"covered_by no-hole mask GPU: query {query_family.name}",
        )
    if not (mask_owned.is_indexed_view and mask_owned.residency is Residency.DEVICE):
        mask_owned.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason=f"covered_by no-hole mask GPU: mask {mask_family.name}",
        )

    query_state = query_owned._ensure_device_state(preserve_indexed_view=True)
    mask_state = mask_owned._ensure_device_state(preserve_indexed_view=True)
    query_buf = query_state.families[query_family]
    mask_buf = mask_state.families[mask_family]

    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    if d_pair_count is not None:
        if pair_capacity is None:
            if d_left is None:
                raise ValueError(
                    "compute_polygonal_covered_by_single_mask_no_holes_gpu "
                    "with d_pair_count requires pair_capacity or d_left"
                )
            pair_capacity = int(d_left.shape[0])
        pair_count = int(pair_capacity)
    elif d_left is not None:
        pair_count = int(d_left.shape[0])
    elif left_indices is not None:
        pair_count = int(left_indices.size)
    else:
        raise ValueError(
            "compute_polygonal_covered_by_single_mask_no_holes_gpu requires "
            "either d_left or left_indices"
        )

    own_d_left = d_left is None
    if own_d_left:
        d_left = runtime.from_host(np.ascontiguousarray(left_indices, dtype=np.int32))
    d_out = runtime.allocate((pair_count,), np.bool_)

    try:
        kernels = _polygon_predicates_kernels()
        P = KERNEL_PARAM_PTR
        I32 = KERNEL_PARAM_I32
        left_args, left_types = _build_side_args(ptr, query_state, query_buf, query_family)
        right_args, right_types = _build_side_args(ptr, mask_state, mask_buf, mask_family)
        tail_args = [ptr(d_left), ptr(d_out), ptr(d_pair_count), pair_count, 0]
        tail_types = [P, P, P, I32, I32]
        all_args = tuple(left_args + right_args + tail_args)
        all_types = tuple(left_types + right_types + tail_types)
        if kernel_name.endswith("_coop"):
            block_size = runtime.optimal_block_size(kernels[kernel_name])
            grid, block = (pair_count, 1, 1), (block_size, 1, 1)
        else:
            grid, block = runtime.launch_config(kernels[kernel_name], pair_count)
        runtime.launch(
            kernels[kernel_name],
            grid=grid,
            block=block,
            params=(all_args, all_types),
        )
        if return_device:
            return d_out

        runtime.synchronize()
        h_out = np.empty(pair_count, dtype=np.bool_)
        runtime.copy_device_to_host(
            d_out,
            h_out,
            reason=f"polygon predicate {kernel_name} result host export",
        )
        return h_out
    finally:
        if own_d_left:
            runtime.free(d_left)
        if not return_device:
            runtime.free(d_out)


def compute_polygonal_covered_by_pair_rows_no_holes_gpu(
    query_owned: OwnedGeometryArray,
    mask_owned: OwnedGeometryArray,
    *,
    query_family: GeometryFamily,
    mask_family: GeometryFamily,
    d_left: object,
    d_right: object,
    d_pair_count: object | None = None,
    pair_capacity: int | None = None,
    return_device: bool = False,
) -> np.ndarray | None:
    """Evaluate polygonal ``query covered_by mask`` relation pairs on device.

    This preserves the relation-pair physical shape for callers that already
    hold device row arrays.  Convex no-hole masks use the cheaper one-sided
    proof; concave, multipart, and hole-bearing masks remain exact by falling
    through inside the kernel to the polygon DE-9IM predicate path.
    """
    kernel_name = _COVERED_BY_PAIR_ROWS_NO_HOLES_KERNEL_MAP.get(
        (query_family, mask_family),
    )
    if kernel_name is None:
        return None

    from vibespatial.runtime.residency import Residency, TransferTrigger

    if not (query_owned.is_indexed_view and query_owned.residency is Residency.DEVICE):
        query_owned.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason=f"covered_by pair-row GPU: query {query_family.name}",
        )
    if not (mask_owned.is_indexed_view and mask_owned.residency is Residency.DEVICE):
        mask_owned.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason=f"covered_by pair-row GPU: mask {mask_family.name}",
        )

    query_state = query_owned._ensure_device_state(preserve_indexed_view=True)
    mask_state = mask_owned._ensure_device_state(preserve_indexed_view=True)
    query_buf = query_state.families[query_family]
    mask_buf = mask_state.families[mask_family]

    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    if d_pair_count is not None:
        if pair_capacity is None:
            pair_capacity = int(d_left.shape[0])
        pair_count = int(pair_capacity)
    else:
        pair_count = int(d_left.shape[0])
    if pair_count != int(d_right.shape[0]):
        raise ValueError(
            "compute_polygonal_covered_by_pair_rows_no_holes_gpu requires "
            "matching d_left and d_right lengths"
        )
    d_out = runtime.allocate((pair_count,), np.bool_)

    try:
        kernels = _polygon_predicates_kernels()
        P = KERNEL_PARAM_PTR
        I32 = KERNEL_PARAM_I32
        left_args, left_types = _build_side_args(
            ptr,
            query_state,
            query_buf,
            query_family,
        )
        right_args, right_types = _build_side_args(
            ptr,
            mask_state,
            mask_buf,
            mask_family,
        )
        tail_args = [
            ptr(d_left),
            ptr(d_right),
            ptr(d_out),
            ptr(d_pair_count),
            pair_count,
        ]
        tail_types = [P, P, P, P, I32]
        block_size = runtime.optimal_block_size(kernels[kernel_name])
        grid, block = (pair_count, 1, 1), (block_size, 1, 1)
        runtime.launch(
            kernels[kernel_name],
            grid=grid,
            block=block,
            params=(
                tuple(left_args + right_args + tail_args),
                tuple(left_types + right_types + tail_types),
            ),
        )
        if return_device:
            return d_out

        runtime.synchronize()
        h_out = np.empty(pair_count, dtype=np.bool_)
        runtime.copy_device_to_host(
            d_out,
            h_out,
            reason=f"polygon predicate {kernel_name} pair-row result host export",
        )
        return h_out
    finally:
        if not return_device:
            runtime.free(d_out)


def compute_polygonal_intersects_gpu(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
    left_indices: np.ndarray | None = None,
    right_indices: np.ndarray | None = None,
    *,
    query_family: GeometryFamily,
    tree_family: GeometryFamily,
    d_left: object | None = None,
    d_right: object | None = None,
    d_pair_count: object | None = None,
    pair_capacity: int | None = None,
    d_pair_offset: object | None = None,
    launch_capacity: int | None = None,
    d_out: object | None = None,
    return_device: bool = False,
) -> np.ndarray | None:
    """Evaluate polygonal ``intersects`` as a boolean relation-pair refine.

    This is narrower than a full DE-9IM matrix.  It returns the boolean
    predicate directly and lets the kernel early-exit on boundary contact or
    containment, preserving device relation pairs for downstream consumers.
    """
    key = (query_family, tree_family)
    swap = False
    if key in _POLYGONAL_INTERSECTS_SWAP_PAIRS:
        swap = True
        key = _POLYGONAL_INTERSECTS_SWAP_PAIRS[key]

    kernel_name = _POLYGONAL_INTERSECTS_KERNEL_MAP.get(key)
    if kernel_name is None:
        return None

    from vibespatial.runtime.residency import Residency, TransferTrigger

    if not (query_owned.is_indexed_view and query_owned.residency is Residency.DEVICE):
        query_owned.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason=f"polygonal intersects GPU: query {query_family.name}",
        )
    if not (tree_owned.is_indexed_view and tree_owned.residency is Residency.DEVICE):
        tree_owned.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason=f"polygonal intersects GPU: tree {tree_family.name}",
        )

    if swap:
        eff_query_owned, eff_tree_owned = tree_owned, query_owned
        eff_query_family, eff_tree_family = tree_family, query_family
        eff_left, eff_right = right_indices, left_indices
        eff_d_left, eff_d_right = d_right, d_left
    else:
        eff_query_owned, eff_tree_owned = query_owned, tree_owned
        eff_query_family, eff_tree_family = query_family, tree_family
        eff_left, eff_right = left_indices, right_indices
        eff_d_left, eff_d_right = d_left, d_right

    query_state = eff_query_owned._ensure_device_state(preserve_indexed_view=True)
    tree_state = eff_tree_owned._ensure_device_state(preserve_indexed_view=True)
    query_buf = query_state.families[eff_query_family]
    tree_buf = tree_state.families[eff_tree_family]

    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    if d_pair_count is not None:
        if pair_capacity is None:
            if eff_d_left is None:
                raise ValueError(
                    "compute_polygonal_intersects_gpu with d_pair_count "
                    "requires pair_capacity or d_left"
                )
            pair_capacity = int(eff_d_left.shape[0])
        pair_count = int(pair_capacity)
    elif eff_d_left is not None:
        pair_count = int(eff_d_left.shape[0])
    elif eff_left is not None:
        pair_count = int(eff_left.size)
    else:
        raise ValueError("compute_polygonal_intersects_gpu requires either d_left or left_indices")

    own_d_left = eff_d_left is None
    own_d_right = eff_d_right is None
    if own_d_left:
        eff_d_left = runtime.from_host(np.ascontiguousarray(eff_left, dtype=np.int32))
    if own_d_right:
        eff_d_right = runtime.from_host(np.ascontiguousarray(eff_right, dtype=np.int32))
    own_d_out = d_out is None
    if own_d_out:
        d_out = runtime.allocate((pair_count,), np.bool_)

    try:
        kernels = _polygon_predicates_kernels()
        P = KERNEL_PARAM_PTR
        I32 = KERNEL_PARAM_I32

        left_args, left_types = _build_side_args(
            ptr,
            query_state,
            query_buf,
            eff_query_family,
        )
        right_args, right_types = _build_side_args(
            ptr,
            tree_state,
            tree_buf,
            eff_tree_family,
        )
        tail_args = [
            ptr(eff_d_left),
            ptr(eff_d_right),
            ptr(d_out),
            ptr(d_pair_offset),
            ptr(d_pair_count),
            pair_count,
        ]
        tail_types = [P, P, P, P, P, I32]
        launch_items = pair_count if launch_capacity is None else int(launch_capacity)
        grid, block = runtime.launch_config(kernels[kernel_name], launch_items)
        runtime.launch(
            kernels[kernel_name],
            grid=grid,
            block=block,
            params=(
                tuple(left_args + right_args + tail_args),
                tuple(left_types + right_types + tail_types),
            ),
        )
        if return_device:
            return d_out

        runtime.synchronize()
        h_out = np.empty(pair_count, dtype=np.bool_)
        runtime.copy_device_to_host(
            d_out,
            h_out,
            reason=f"polygon predicate {kernel_name} intersects result host export",
        )
        return h_out
    finally:
        if own_d_left:
            runtime.free(eff_d_left)
        if own_d_right:
            runtime.free(eff_d_right)
        if own_d_out and not return_device:
            runtime.free(d_out)


def compute_polygon_de9im_gpu(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
    left_indices: np.ndarray | None = None,
    right_indices: np.ndarray | None = None,
    *,
    query_family: GeometryFamily,
    tree_family: GeometryFamily,
    d_left: object | None = None,
    d_right: object | None = None,
    d_pair_count: object | None = None,
    pair_capacity: int | None = None,
    d_pair_offset: object | None = None,
    launch_capacity: int | None = None,
    d_mask: object | None = None,
    return_device: bool = False,
) -> np.ndarray | None:
    """Compute DE-9IM bitmasks for geometry candidate pairs on GPU.

    Supports all combinations of LINESTRING, MULTILINESTRING, POLYGON,
    and MULTIPOLYGON families.

    When *d_left* / *d_right* are provided (device-resident CuPy int32
    arrays), they are used directly instead of uploading *left_indices* /
    *right_indices* from host — avoiding a redundant host->device transfer
    when candidates are already on device.

    Parameters
    ----------
    left_indices, right_indices : np.ndarray or None
        Host index arrays.  May be ``None`` when *d_left* / *d_right* are
        provided — ``pair_count`` is derived from ``d_left.shape[0]``.
    return_device : bool
        When True, return the result as a device-resident CuPy uint16 array
        (caller takes ownership).  The D->H copy is skipped entirely.
        Default False preserves backward compatibility (returns host
        np.ndarray).
    d_pair_count : cupy.ndarray or None
        Optional device scalar naming how many leading entries in *d_left* /
        *d_right* are live.  This preserves candidate-refine rowsets whose
        count is produced by a device compaction kernel without a host scalar
        sizing fence.
    pair_capacity : int or None
        Launch capacity for *d_left* / *d_right* when *d_pair_count* is used.

    Returns
    -------
    np.ndarray | cupy.ndarray | None
        uint16 DE-9IM bitmask array, or None if the family pair is
        not supported.
    """
    key = (query_family, tree_family)
    swap = False
    if key in _SWAP_PAIRS:
        swap = True
        key = _SWAP_PAIRS[key]

    kernel_name = _KERNEL_MAP.get(key)
    if kernel_name is None:
        return None

    from vibespatial.runtime.residency import Residency, TransferTrigger

    if not (query_owned.is_indexed_view and query_owned.residency is Residency.DEVICE):
        query_owned.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason=f"de9im GPU: query {query_family.name}",
        )
    if not (tree_owned.is_indexed_view and tree_owned.residency is Residency.DEVICE):
        tree_owned.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason=f"de9im GPU: tree {tree_family.name}",
        )

    if swap:
        eff_query_owned, eff_tree_owned = tree_owned, query_owned
        eff_query_family, eff_tree_family = tree_family, query_family
        eff_left, eff_right = right_indices, left_indices
        eff_d_left, eff_d_right = d_right, d_left
    else:
        eff_query_owned, eff_tree_owned = query_owned, tree_owned
        eff_query_family, eff_tree_family = query_family, tree_family
        eff_left, eff_right = left_indices, right_indices
        eff_d_left, eff_d_right = d_left, d_right

    query_state = eff_query_owned._ensure_device_state(preserve_indexed_view=True)
    tree_state = eff_tree_owned._ensure_device_state(preserve_indexed_view=True)
    query_buf = query_state.families[eff_query_family]
    tree_buf = tree_state.families[eff_tree_family]

    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    # Derive pair_count: prefer device arrays (avoids need for host
    # placeholder), then fall back to host indices.
    if d_pair_count is not None:
        if pair_capacity is None:
            if eff_d_left is None:
                raise ValueError(
                    "compute_polygon_de9im_gpu with d_pair_count requires pair_capacity or d_left"
                )
            pair_capacity = int(eff_d_left.shape[0])
        pair_count = int(pair_capacity)
    elif eff_d_left is not None:
        pair_count = eff_d_left.shape[0]
    elif eff_left is not None:
        pair_count = eff_left.size
    else:
        raise ValueError("compute_polygon_de9im_gpu requires either d_left or left_indices")

    # Use device-resident arrays when provided; otherwise upload from host.
    own_d_left = eff_d_left is None
    own_d_right = eff_d_right is None
    if own_d_left:
        eff_d_left = runtime.from_host(np.ascontiguousarray(eff_left, dtype=np.int32))
    if own_d_right:
        eff_d_right = runtime.from_host(np.ascontiguousarray(eff_right, dtype=np.int32))
    own_d_mask = d_mask is None
    if own_d_mask:
        d_mask = runtime.allocate((pair_count,), np.uint16)

    try:
        kernels = _polygon_predicates_kernels()
        P = KERNEL_PARAM_PTR
        I32 = KERNEL_PARAM_I32

        left_args, left_types = _build_side_args(ptr, query_state, query_buf, eff_query_family)
        right_args, right_types = _build_side_args(ptr, tree_state, tree_buf, eff_tree_family)
        tail_args = [
            ptr(eff_d_left),
            ptr(eff_d_right),
            ptr(d_mask),
            ptr(d_pair_offset),
            ptr(d_pair_count),
            pair_count,
        ]
        tail_types = [P, P, P, P, P, I32]

        all_args = tuple(left_args + right_args + tail_args)
        all_types = tuple(left_types + right_types + tail_types)

        launch_items = pair_count if launch_capacity is None else int(launch_capacity)
        grid, block = runtime.launch_config(kernels[kernel_name], launch_items)
        runtime.launch(
            kernels[kernel_name],
            grid=grid,
            block=block,
            params=(all_args, all_types),
        )
        if return_device:
            # Return device-resident CuPy array — caller takes ownership.
            # No sync needed: CuPy ops on the same stream are ordered.
            if swap:
                if d_pair_offset is None:
                    d_mask = _transpose_de9im_device(d_mask)
                else:
                    transpose_kernel = _de9im_eval_kernels()[
                        "transpose_de9im_grouped_kernel"
                    ]
                    transpose_grid, transpose_block = runtime.launch_config(
                        transpose_kernel,
                        launch_items,
                    )
                    runtime.launch(
                        transpose_kernel,
                        grid=transpose_grid,
                        block=transpose_block,
                        params=(
                            (
                                ptr(d_mask),
                                ptr(d_pair_offset),
                                ptr(d_pair_count),
                            ),
                            (
                                KERNEL_PARAM_PTR,
                                KERNEL_PARAM_PTR,
                                KERNEL_PARAM_PTR,
                            ),
                        ),
                    )
            return d_mask

        runtime.synchronize()
        h_mask = np.empty(pair_count, dtype=np.uint16)
        runtime.copy_device_to_host(
            d_mask,
            h_mask,
            reason=f"polygon predicate {kernel_name} de9im-mask host export",
        )

        if swap:
            h_mask = _transpose_de9im(h_mask)

        return h_mask

    finally:
        if own_d_left:
            runtime.free(eff_d_left)
        if own_d_right:
            runtime.free(eff_d_right)
        if own_d_mask and not return_device:
            runtime.free(d_mask)


def _transpose_de9im(masks: np.ndarray) -> np.ndarray:
    """Transpose DE-9IM bitmasks (swap A and B roles)."""
    m = masks.astype(np.uint16, copy=True)
    out = np.zeros_like(m)
    # II stays, EE stays.
    out |= m & DE9IM_II
    out |= m & DE9IM_EE
    # Swap IB ↔ BI.
    out |= np.where(m & DE9IM_IB, DE9IM_BI, 0).astype(np.uint16)
    out |= np.where(m & DE9IM_BI, DE9IM_IB, 0).astype(np.uint16)
    # Swap IE ↔ EI.
    out |= np.where(m & DE9IM_IE, DE9IM_EI, 0).astype(np.uint16)
    out |= np.where(m & DE9IM_EI, DE9IM_IE, 0).astype(np.uint16)
    # Swap BE ↔ EB.
    out |= np.where(m & DE9IM_BE, DE9IM_EB, 0).astype(np.uint16)
    out |= np.where(m & DE9IM_EB, DE9IM_BE, 0).astype(np.uint16)
    # BB stays.
    out |= m & DE9IM_BB
    return out


def _transpose_de9im_device(d_masks: object) -> object:
    """Transpose DE-9IM bitmasks on device with the native predicate kernel.

    Device-resident mirror of ``_transpose_de9im`` that keeps data on GPU.
    """
    import cupy as cp

    d_masks = cp.asarray(d_masks)
    if d_masks.dtype != cp.uint16:
        d_masks = d_masks.astype(cp.uint16, copy=False)
    out = cp.empty_like(d_masks)
    n = int(d_masks.size)
    if n == 0:
        return out

    runtime = get_cuda_runtime()
    kernels = _de9im_eval_kernels()
    kernel = kernels["transpose_de9im_kernel"]
    ptr = runtime.pointer
    grid, block = runtime.launch_config(kernel, n)
    runtime.launch(
        kernel,
        grid=grid,
        block=block,
        params=(
            (ptr(d_masks), ptr(out), n),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        ),
    )
    return out


def supported_predicate_families() -> frozenset[tuple[GeometryFamily, GeometryFamily]]:
    """Return the set of family pairs supported by polygon predicate kernels."""
    return frozenset(_KERNEL_MAP.keys())
