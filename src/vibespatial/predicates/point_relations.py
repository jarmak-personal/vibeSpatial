from __future__ import annotations

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import FAMILY_TAGS, OwnedGeometryArray
from vibespatial.predicates.point_relations_kernels import (
    _MULTIPOINT_BINARY_RELATIONS_KERNEL_SOURCE,
    _MULTIPOINT_KERNEL_NAMES,
    _POINT_BINARY_RELATIONS_KERNEL_NAMES,
    _POINT_BINARY_RELATIONS_KERNEL_SOURCE,
)

POINT_LOCATION_OUTSIDE = np.uint8(0)
POINT_LOCATION_BOUNDARY = np.uint8(1)
POINT_LOCATION_INTERIOR = np.uint8(2)

request_nvrtc_warmup([
    ("point-binary-relations", _POINT_BINARY_RELATIONS_KERNEL_SOURCE, _POINT_BINARY_RELATIONS_KERNEL_NAMES),
    ("multipoint-binary-relations", _MULTIPOINT_BINARY_RELATIONS_KERNEL_SOURCE, _MULTIPOINT_KERNEL_NAMES),
])


def _point_binary_relation_kernels():
    return compile_kernel_group("point-binary-relations", _POINT_BINARY_RELATIONS_KERNEL_SOURCE, _POINT_BINARY_RELATIONS_KERNEL_NAMES)


def _multipoint_relation_kernels():
    return compile_kernel_group("multipoint-binary-relations", _MULTIPOINT_BINARY_RELATIONS_KERNEL_SOURCE, _MULTIPOINT_KERNEL_NAMES)


def _is_device_array(value) -> bool:
    return hasattr(value, "__cuda_array_interface__")


def _identity_rows(count: int, *, device: bool):
    if device:
        import cupy as cp

        return cp.arange(count, dtype=cp.int32)
    return np.arange(count, dtype=np.int32)


def _false_like_bool(reference):
    if _is_device_array(reference):
        import cupy as cp

        return cp.zeros(reference.shape[0], dtype=cp.bool_)
    return np.zeros(reference.shape[0], dtype=bool)


def _point_relation_to_predicate_array(
    predicate: str,
    relation,
    *,
    point_on_left: bool,
):
    outside = relation == POINT_LOCATION_OUTSIDE
    boundary = relation == POINT_LOCATION_BOUNDARY
    interior = relation == POINT_LOCATION_INTERIOR
    if predicate == "intersects":
        return ~outside
    if predicate == "disjoint":
        return outside
    if predicate == "touches":
        return boundary
    if predicate in {"crosses", "overlaps"}:
        return _false_like_bool(relation)
    if point_on_left:
        if predicate == "within":
            return interior
        if predicate == "covered_by":
            return ~outside
        return _false_like_bool(relation)
    if predicate == "contains":
        return interior
    if predicate == "covers":
        return ~outside
    if predicate == "contains_properly":
        return interior
    return _false_like_bool(relation)


def _point_equals_to_predicate_array(predicate: str, relation):
    equal = relation == POINT_LOCATION_INTERIOR
    if predicate in {
        "intersects",
        "contains",
        "within",
        "covers",
        "covered_by",
        "contains_properly",
        "equals",
    }:
        return equal
    if predicate == "disjoint":
        return ~equal
    return _false_like_bool(relation)


# ---------------------------------------------------------------------------
# Unified kernel launch -- replaces the three nearly-identical functions
# _launch_rows_kernel, _launch_indexed_kernel, _launch_indexed_mp_kernel.
# ---------------------------------------------------------------------------

def _launch_kernel(
    kernel_dict_fn,
    kernel_name: str,
    candidate_rows: np.ndarray,
    args: tuple[int, ...],
    arg_types: tuple[object, ...],
    *,
    extra_device_allocs: list | None = None,
    return_device: bool = False,
) -> np.ndarray:
    """Launch a point or multipoint binary-relation kernel.

    Parameters
    ----------
    kernel_dict_fn : callable
        One of ``_point_binary_relation_kernels`` or ``_multipoint_relation_kernels``.
    kernel_name : str
        Name of the CUDA kernel to launch.
    candidate_rows : np.ndarray
        Row indices (int32) to pass as the first kernel argument.
    args : tuple
        Device pointer / scalar arguments between candidate_rows and (out, count).
    arg_types : tuple
        KERNEL_PARAM_* type tags matching *args*.
    extra_device_allocs : list or None
        Additional device allocations to free after launch (e.g. uploaded
        mapped FRO arrays).  With ``return_device=True``, ownership of
        ``device_out`` transfers to the caller as a CuPy array.
    """
    n_items = int(candidate_rows.size)
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    returning_device = False
    device_rows_temp = None
    if _is_device_array(candidate_rows):
        import cupy as cp

        device_rows = cp.asarray(candidate_rows)
        if device_rows.dtype != cp.int32:
            device_rows = device_rows.astype(cp.int32, copy=False)
            device_rows_temp = device_rows
    else:
        device_rows = runtime.from_host(candidate_rows.astype(np.int32, copy=False))
        device_rows_temp = device_rows
    device_out = runtime.allocate((n_items,), np.uint8)
    try:
        kernel = kernel_dict_fn()[kernel_name]
        params = (
            (ptr(device_rows), *args, ptr(device_out), n_items),
            (KERNEL_PARAM_PTR, *arg_types, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        )
        grid, block = runtime.launch_config(kernel, n_items)
        runtime.launch(kernel, grid=grid, block=block, params=params)
        runtime.synchronize()
        if return_device:
            returning_device = True
            return device_out
        out = np.empty(n_items, dtype=np.uint8)
        runtime.copy_device_to_host(device_out, out)
        return out
    finally:
        runtime.free(device_rows_temp)
        if not returning_device:
            runtime.free(device_out)
        if extra_device_allocs:
            for alloc in extra_device_allocs:
                runtime.free(alloc)


# ---------------------------------------------------------------------------
# Non-indexed public API -- use candidate_rows and the owned array's
# device-side family_row_offsets directly.
# ---------------------------------------------------------------------------

def classify_point_equals_gpu(
    candidate_rows: np.ndarray,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> np.ndarray:
    if candidate_rows.size == 0:
        return np.empty(0, dtype=np.uint8)
    left_state = left._ensure_device_state()
    right_state = right._ensure_device_state()
    left_buffer = left_state.families[GeometryFamily.POINT]
    right_buffer = right_state.families[GeometryFamily.POINT]
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    return _launch_kernel(
        _point_binary_relation_kernels,
        "point_equals_compacted",
        candidate_rows,
        (
            ptr(left_state.family_row_offsets),
            ptr(left_buffer.geometry_offsets),
            ptr(left_buffer.empty_mask),
            ptr(left_buffer.x),
            ptr(left_buffer.y),
            ptr(right_state.family_row_offsets),
            ptr(right_buffer.geometry_offsets),
            ptr(right_buffer.empty_mask),
            ptr(right_buffer.x),
            ptr(right_buffer.y),
        ),
        (KERNEL_PARAM_PTR,) * 10,
    )


def classify_point_line_gpu(
    candidate_rows: np.ndarray,
    points: OwnedGeometryArray,
    lines: OwnedGeometryArray,
    *,
    line_family: GeometryFamily,
) -> np.ndarray:
    if candidate_rows.size == 0:
        return np.empty(0, dtype=np.uint8)
    point_state = points._ensure_device_state()
    line_state = lines._ensure_device_state()
    point_buffer = point_state.families[GeometryFamily.POINT]
    line_buffer = line_state.families[line_family]
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    kernel_name = (
        "point_on_linestring_compacted"
        if line_family is GeometryFamily.LINESTRING
        else "point_on_multilinestring_compacted"
    )
    args = [
        ptr(point_state.family_row_offsets),
        ptr(point_buffer.geometry_offsets),
        ptr(point_buffer.empty_mask),
        ptr(point_buffer.x),
        ptr(point_buffer.y),
        ptr(line_state.family_row_offsets),
        ptr(line_buffer.geometry_offsets),
    ]
    if line_family is not GeometryFamily.LINESTRING:
        args.append(ptr(line_buffer.part_offsets))
    args.extend([
        ptr(line_buffer.empty_mask),
        ptr(line_buffer.x),
        ptr(line_buffer.y),
    ])
    return _launch_kernel(
        _point_binary_relation_kernels, kernel_name,
        candidate_rows, tuple(args), (KERNEL_PARAM_PTR,) * len(args),
    )


def classify_point_region_gpu(
    candidate_rows: np.ndarray,
    points: OwnedGeometryArray,
    regions: OwnedGeometryArray,
    *,
    region_family: GeometryFamily,
) -> np.ndarray:
    if candidate_rows.size == 0:
        return np.empty(0, dtype=np.uint8)
    point_state = points._ensure_device_state()
    region_state = regions._ensure_device_state()
    point_buffer = point_state.families[GeometryFamily.POINT]
    region_buffer = region_state.families[region_family]
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    kernel_name = (
        "point_in_polygon_polygon_compacted_state"
        if region_family is GeometryFamily.POLYGON
        else "point_in_polygon_multipolygon_compacted_state"
    )
    args = [
        ptr(point_state.family_row_offsets),
        ptr(point_buffer.geometry_offsets),
        ptr(point_buffer.empty_mask),
        ptr(point_buffer.x),
        ptr(point_buffer.y),
        ptr(region_state.family_row_offsets),
        ptr(region_buffer.empty_mask),
        ptr(region_buffer.geometry_offsets),
    ]
    if region_family is not GeometryFamily.POLYGON:
        args.append(ptr(region_buffer.part_offsets))
    args.extend([
        ptr(region_buffer.ring_offsets),
        ptr(region_buffer.x),
        ptr(region_buffer.y),
    ])
    return _launch_kernel(
        _point_binary_relation_kernels, kernel_name,
        candidate_rows, tuple(args), (KERNEL_PARAM_PTR,) * len(args),
    )


# ---------------------------------------------------------------------------
# Indexed variants: separate left/right index arrays into original owned
# geometry arrays.  Avoids the expensive take() buffer copy by pre-gathering
# family_row_offsets on host and uploading the mapped arrays.
# ---------------------------------------------------------------------------

_POINT_TAG_INDEXED = FAMILY_TAGS[GeometryFamily.POINT]
_LINE_FAMILIES_INDEXED = (GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING)
_REGION_FAMILIES_INDEXED = (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON)
_LINE_TAGS_INDEXED = tuple(FAMILY_TAGS[f] for f in _LINE_FAMILIES_INDEXED)
_REGION_TAGS_INDEXED = tuple(FAMILY_TAGS[f] for f in _REGION_FAMILIES_INDEXED)


def _prepare_indexed_fro(owned, indices, runtime):
    """Map indices through family_row_offsets and return a device array."""
    if _is_device_array(indices):
        import cupy as cp

        state = owned._ensure_device_state()
        return state.family_row_offsets[indices].astype(cp.int32, copy=False)
    mapped = owned.family_row_offsets[indices].astype(np.int32, copy=False)
    return runtime.from_host(mapped)


def _classify_indexed_point_equals(
    left_owned: OwnedGeometryArray,
    right_owned: OwnedGeometryArray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    *,
    return_device: bool = False,
) -> np.ndarray:
    n = int(left_indices.size)
    left_state = left_owned._ensure_device_state()
    right_state = right_owned._ensure_device_state()
    left_buffer = left_state.families[GeometryFamily.POINT]
    right_buffer = right_state.families[GeometryFamily.POINT]
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    device_left_fro = _prepare_indexed_fro(left_owned, left_indices, runtime)
    device_right_fro = _prepare_indexed_fro(right_owned, right_indices, runtime)
    identity_rows = _identity_rows(n, device=return_device)
    return _launch_kernel(
        _point_binary_relation_kernels,
        "point_equals_compacted", identity_rows,
        (
            ptr(device_left_fro),
            ptr(left_buffer.geometry_offsets),
            ptr(left_buffer.empty_mask),
            ptr(left_buffer.x),
            ptr(left_buffer.y),
            ptr(device_right_fro),
            ptr(right_buffer.geometry_offsets),
            ptr(right_buffer.empty_mask),
            ptr(right_buffer.x),
            ptr(right_buffer.y),
        ),
        (KERNEL_PARAM_PTR,) * 10,
        extra_device_allocs=[device_left_fro, device_right_fro],
        return_device=return_device,
    )


def _classify_indexed_point_line(
    point_owned: OwnedGeometryArray,
    line_owned: OwnedGeometryArray,
    point_indices: np.ndarray,
    line_indices: np.ndarray,
    *,
    line_family: GeometryFamily,
    return_device: bool = False,
) -> np.ndarray:
    n = int(point_indices.size)
    point_state = point_owned._ensure_device_state()
    line_state = line_owned._ensure_device_state()
    point_buffer = point_state.families[GeometryFamily.POINT]
    line_buffer = line_state.families[line_family]
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    device_point_fro = _prepare_indexed_fro(point_owned, point_indices, runtime)
    device_line_fro = _prepare_indexed_fro(line_owned, line_indices, runtime)
    identity_rows = _identity_rows(n, device=return_device)
    kernel_name = (
        "point_on_linestring_compacted"
        if line_family is GeometryFamily.LINESTRING
        else "point_on_multilinestring_compacted"
    )
    args = [
        ptr(device_point_fro),
        ptr(point_buffer.geometry_offsets),
        ptr(point_buffer.empty_mask),
        ptr(point_buffer.x),
        ptr(point_buffer.y),
        ptr(device_line_fro),
        ptr(line_buffer.geometry_offsets),
    ]
    if line_family is not GeometryFamily.LINESTRING:
        args.append(ptr(line_buffer.part_offsets))
    args.extend([
        ptr(line_buffer.empty_mask),
        ptr(line_buffer.x),
        ptr(line_buffer.y),
    ])
    return _launch_kernel(
        _point_binary_relation_kernels, kernel_name,
        identity_rows, tuple(args), (KERNEL_PARAM_PTR,) * len(args),
        extra_device_allocs=[device_point_fro, device_line_fro],
        return_device=return_device,
    )


def _classify_indexed_point_region(
    point_owned: OwnedGeometryArray,
    region_owned: OwnedGeometryArray,
    point_indices: np.ndarray,
    region_indices: np.ndarray,
    *,
    region_family: GeometryFamily,
    return_device: bool = False,
) -> np.ndarray:
    n = int(point_indices.size)
    point_state = point_owned._ensure_device_state()
    region_state = region_owned._ensure_device_state()
    point_buffer = point_state.families[GeometryFamily.POINT]
    region_buffer = region_state.families[region_family]
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    device_point_fro = _prepare_indexed_fro(point_owned, point_indices, runtime)
    device_region_fro = _prepare_indexed_fro(region_owned, region_indices, runtime)
    identity_rows = _identity_rows(n, device=return_device)
    kernel_name = (
        "point_in_polygon_polygon_compacted_state"
        if region_family is GeometryFamily.POLYGON
        else "point_in_polygon_multipolygon_compacted_state"
    )
    args = [
        ptr(device_point_fro),
        ptr(point_buffer.geometry_offsets),
        ptr(point_buffer.empty_mask),
        ptr(point_buffer.x),
        ptr(point_buffer.y),
        ptr(device_region_fro),
        ptr(region_buffer.empty_mask),
        ptr(region_buffer.geometry_offsets),
    ]
    if region_family is not GeometryFamily.POLYGON:
        args.append(ptr(region_buffer.part_offsets))
    args.extend([
        ptr(region_buffer.ring_offsets),
        ptr(region_buffer.x),
        ptr(region_buffer.y),
    ])
    return _launch_kernel(
        _point_binary_relation_kernels, kernel_name,
        identity_rows, tuple(args), (KERNEL_PARAM_PTR,) * len(args),
        extra_device_allocs=[device_point_fro, device_region_fro],
        return_device=return_device,
    )


def classify_point_predicates_indexed(
    predicate: str,
    left_owned: OwnedGeometryArray,
    right_owned: OwnedGeometryArray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
) -> np.ndarray:
    """Evaluate point-family predicates using indexed access into original owned arrays.

    Avoids the expensive take() buffer copy by pre-gathering family_row_offsets
    on the host and passing them directly to existing GPU kernels.

    Returns a boolean array of length ``left_indices.size``.
    """
    from .binary import (
        _apply_relation_rows,
        _point_equals_to_predicate,
        _point_relation_to_predicate,
    )

    n = left_indices.size
    if n == 0:
        return np.empty(0, dtype=bool)

    out = np.zeros(n, dtype=bool)
    left_tags = left_owned.tags[left_indices]
    right_tags = right_owned.tags[right_indices]

    # Point x point
    pp_mask = (left_tags == _POINT_TAG_INDEXED) & (right_tags == _POINT_TAG_INDEXED)
    if pp_mask.any():
        idx = np.flatnonzero(pp_mask)
        relation = _classify_indexed_point_equals(
            left_owned, right_owned, left_indices[idx], right_indices[idx],
        )
        _apply_relation_rows(out, idx, _point_equals_to_predicate(predicate, relation))

    # Point x line and line x point
    for line_family, line_tag in zip(_LINE_FAMILIES_INDEXED, _LINE_TAGS_INDEXED, strict=True):
        pl_mask = (left_tags == _POINT_TAG_INDEXED) & (right_tags == line_tag)
        if pl_mask.any():
            idx = np.flatnonzero(pl_mask)
            relation = _classify_indexed_point_line(
                left_owned, right_owned, left_indices[idx], right_indices[idx],
                line_family=line_family,
            )
            _apply_relation_rows(out, idx, _point_relation_to_predicate(predicate, relation, point_on_left=True))

        lp_mask = (left_tags == line_tag) & (right_tags == _POINT_TAG_INDEXED)
        if lp_mask.any():
            idx = np.flatnonzero(lp_mask)
            relation = _classify_indexed_point_line(
                right_owned, left_owned, right_indices[idx], left_indices[idx],
                line_family=line_family,
            )
            _apply_relation_rows(out, idx, _point_relation_to_predicate(predicate, relation, point_on_left=False))

    # Point x region and region x point
    for region_family, region_tag in zip(_REGION_FAMILIES_INDEXED, _REGION_TAGS_INDEXED, strict=True):
        pr_mask = (left_tags == _POINT_TAG_INDEXED) & (right_tags == region_tag)
        if pr_mask.any():
            idx = np.flatnonzero(pr_mask)
            relation = _classify_indexed_point_region(
                left_owned, right_owned, left_indices[idx], right_indices[idx],
                region_family=region_family,
            )
            _apply_relation_rows(out, idx, _point_relation_to_predicate(predicate, relation, point_on_left=True))

        rp_mask = (region_tag == left_tags) & (right_tags == _POINT_TAG_INDEXED)
        if rp_mask.any():
            idx = np.flatnonzero(rp_mask)
            relation = _classify_indexed_point_region(
                right_owned, left_owned, right_indices[idx], left_indices[idx],
                region_family=region_family,
            )
            _apply_relation_rows(out, idx, _point_relation_to_predicate(predicate, relation, point_on_left=False))

    # Multipoint x anything and anything x multipoint
    mp_tag = FAMILY_TAGS[GeometryFamily.MULTIPOINT]
    mp_left_mask = left_tags == mp_tag
    mp_right_mask = right_tags == mp_tag

    if mp_left_mask.any() or mp_right_mask.any():
        _dispatch_multipoint_pairs(
            predicate, out,
            left_owned, right_owned,
            left_indices, right_indices,
            left_tags, right_tags,
            mp_left_mask, mp_right_mask,
            _apply_relation_rows,
        )

    return out


def classify_point_predicates_indexed_device(
    predicate: str,
    left_owned: OwnedGeometryArray,
    right_owned: OwnedGeometryArray,
    left_indices,
    right_indices,
    *,
    left_tags=None,
    right_tags=None,
):
    """Evaluate point-vs-point/line/region indexed predicates on device.

    This is the device-resident companion to
    :func:`classify_point_predicates_indexed`.  It intentionally excludes
    multipoint rows for now because those kernels need subset-style reverse
    checks that still compose through host masks.  Callers must route batches
    containing multipoints through the existing host-returning path.
    """
    import cupy as cp

    left_indices = cp.asarray(left_indices, dtype=cp.int32)
    right_indices = cp.asarray(right_indices, dtype=cp.int32)
    n = int(left_indices.size)
    if n == 0:
        return cp.empty(0, dtype=cp.bool_)

    left_state = left_owned._ensure_device_state()
    right_state = right_owned._ensure_device_state()
    left_tags = (
        cp.asarray(left_tags, dtype=cp.int8)
        if left_tags is not None
        else left_state.tags[left_indices]
    )
    right_tags = (
        cp.asarray(right_tags, dtype=cp.int8)
        if right_tags is not None
        else right_state.tags[right_indices]
    )

    mp_tag = FAMILY_TAGS[GeometryFamily.MULTIPOINT]
    if bool(cp.any((left_tags == mp_tag) | (right_tags == mp_tag))):
        raise ValueError(
            "device indexed point predicate classification does not support multipoint rows"
        )

    out = cp.zeros(n, dtype=cp.bool_)

    pp_mask = (left_tags == _POINT_TAG_INDEXED) & (right_tags == _POINT_TAG_INDEXED)
    if bool(cp.any(pp_mask)):
        idx = cp.flatnonzero(pp_mask).astype(cp.int32, copy=False)
        relation = _classify_indexed_point_equals(
            left_owned,
            right_owned,
            left_indices[idx],
            right_indices[idx],
            return_device=True,
        )
        out[idx] = _point_equals_to_predicate_array(predicate, relation)

    for line_family, line_tag in zip(_LINE_FAMILIES_INDEXED, _LINE_TAGS_INDEXED, strict=True):
        pl_mask = (left_tags == _POINT_TAG_INDEXED) & (right_tags == line_tag)
        if bool(cp.any(pl_mask)):
            idx = cp.flatnonzero(pl_mask).astype(cp.int32, copy=False)
            relation = _classify_indexed_point_line(
                left_owned,
                right_owned,
                left_indices[idx],
                right_indices[idx],
                line_family=line_family,
                return_device=True,
            )
            out[idx] = _point_relation_to_predicate_array(
                predicate,
                relation,
                point_on_left=True,
            )

        lp_mask = (left_tags == line_tag) & (right_tags == _POINT_TAG_INDEXED)
        if bool(cp.any(lp_mask)):
            idx = cp.flatnonzero(lp_mask).astype(cp.int32, copy=False)
            relation = _classify_indexed_point_line(
                right_owned,
                left_owned,
                right_indices[idx],
                left_indices[idx],
                line_family=line_family,
                return_device=True,
            )
            out[idx] = _point_relation_to_predicate_array(
                predicate,
                relation,
                point_on_left=False,
            )

    for region_family, region_tag in zip(_REGION_FAMILIES_INDEXED, _REGION_TAGS_INDEXED, strict=True):
        pr_mask = (left_tags == _POINT_TAG_INDEXED) & (right_tags == region_tag)
        if bool(cp.any(pr_mask)):
            idx = cp.flatnonzero(pr_mask).astype(cp.int32, copy=False)
            relation = _classify_indexed_point_region(
                left_owned,
                right_owned,
                left_indices[idx],
                right_indices[idx],
                region_family=region_family,
                return_device=True,
            )
            out[idx] = _point_relation_to_predicate_array(
                predicate,
                relation,
                point_on_left=True,
            )

        rp_mask = (left_tags == region_tag) & (right_tags == _POINT_TAG_INDEXED)
        if bool(cp.any(rp_mask)):
            idx = cp.flatnonzero(rp_mask).astype(cp.int32, copy=False)
            relation = _classify_indexed_point_region(
                right_owned,
                left_owned,
                right_indices[idx],
                left_indices[idx],
                region_family=region_family,
                return_device=True,
            )
            out[idx] = _point_relation_to_predicate_array(
                predicate,
                relation,
                point_on_left=False,
            )

    return out


# ---------------------------------------------------------------------------
# Multipoint support -- launch helpers, predicate conversion, and dispatch.
# Tier 1 per ADR-0033: geometry-specific inner loops (multipoint coord iteration).
# ---------------------------------------------------------------------------

# Bit flags in multipoint kernel output
_MP_ANY_OUTSIDE = np.uint8(1)
_MP_ANY_BOUNDARY = np.uint8(2)
_MP_ANY_INTERIOR = np.uint8(4)


def _multipoint_bits_to_predicate(
    predicate: str,
    bits: np.ndarray,
    *,
    mp_on_left: bool,
    target_family: GeometryFamily | None = None,
) -> np.ndarray:
    """Convert packed multipoint relation bits to boolean predicate results.

    Bits: 0x1 = any_outside, 0x2 = any_boundary, 0x4 = any_interior.
    Each bit records how the multipoint's coordinates relate to the target.

    **Key asymmetry:**  The bits tell us *"for each MP coord, its location in
    the target."*  This directly gives ``within`` / ``covered_by`` (all MP
    coords inside target) and the symmetric predicates (intersects / disjoint
    / touches).  For ``contains`` / ``covers`` / ``contains_properly`` we need
    the *reverse* -- whether the target fits inside the multipoint.

    * MP x Point ``contains``: at least one MP coord equals the point ->
      ``any_interior`` (the kernel records "equal" as interior).
    * MP x Line/Polygon ``contains``: always False (0-D cannot contain >=1-D).
    * MP x MP ``contains``: handled by the dispatch running the kernel in
      reverse and calling this function with a swapped predicate.
    """
    any_outside = (bits & _MP_ANY_OUTSIDE).astype(bool)
    any_boundary = (bits & _MP_ANY_BOUNDARY).astype(bool)
    any_interior = (bits & _MP_ANY_INTERIOR).astype(bool)
    any_hit = any_boundary | any_interior
    n = bits.shape[0]

    # --- Symmetric predicates ---
    if predicate == "intersects":
        return any_hit
    if predicate == "disjoint":
        return ~any_hit
    if predicate == "touches":
        return any_boundary & ~any_interior

    # --- within / covered_by: is the MP inside the target? ---
    # Condition: every MP coord must be inside (or on boundary of) the target.
    if mp_on_left:
        if predicate == "within":
            return any_interior & ~any_outside
        if predicate == "covered_by":
            return any_hit & ~any_outside

        # contains / covers / contains_properly: is the target inside the MP?
        if predicate in {"contains", "covers", "contains_properly"}:
            tf = target_family
            if tf is GeometryFamily.POINT or tf is GeometryFamily.MULTIPOINT:
                # MP contains point iff point matches at least one MP coord.
                # For MPxMP, the dispatch handles the reverse check.
                return any_interior
            # MP can't contain a line or polygon -- 0-D vs >=1-D.
            return np.zeros(n, dtype=bool)

    else:
        # MP is on the right (tree side), target on left.

        # contains / covers: does the target contain every MP coord?
        if predicate == "contains":
            return any_interior & ~any_outside
        if predicate == "covers":
            return any_hit & ~any_outside
        if predicate == "contains_properly":
            return any_interior & ~any_outside

        # within / covered_by: is the target within the MP?
        if predicate in {"within", "covered_by"}:
            tf = target_family
            if tf is GeometryFamily.POINT or tf is GeometryFamily.MULTIPOINT:
                return any_interior
            return np.zeros(n, dtype=bool)

    return np.zeros(n, dtype=bool)


# ---------------------------------------------------------------------------
# Indexed multipoint classify functions
# ---------------------------------------------------------------------------

def _classify_indexed_mp_point(
    mp_owned: OwnedGeometryArray,
    pt_owned: OwnedGeometryArray,
    mp_indices: np.ndarray,
    pt_indices: np.ndarray,
) -> np.ndarray:
    """MULTIPOINT x POINT relation bits."""
    n = mp_indices.size
    mp_state = mp_owned._ensure_device_state()
    pt_state = pt_owned._ensure_device_state()
    mp_buffer = mp_state.families[GeometryFamily.MULTIPOINT]
    pt_buffer = pt_state.families[GeometryFamily.POINT]
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    device_mp_fro = _prepare_indexed_fro(mp_owned, mp_indices, runtime)
    device_pt_fro = _prepare_indexed_fro(pt_owned, pt_indices, runtime)
    identity_rows = np.arange(n, dtype=np.int32)
    return _launch_kernel(
        _multipoint_relation_kernels,
        "multipoint_point_relation_compacted", identity_rows,
        (
            ptr(device_mp_fro),
            ptr(mp_buffer.geometry_offsets),
            ptr(mp_buffer.empty_mask),
            ptr(mp_buffer.x),
            ptr(mp_buffer.y),
            ptr(device_pt_fro),
            ptr(pt_buffer.geometry_offsets),
            ptr(pt_buffer.empty_mask),
            ptr(pt_buffer.x),
            ptr(pt_buffer.y),
        ),
        (KERNEL_PARAM_PTR,) * 10,
        extra_device_allocs=[device_mp_fro, device_pt_fro],
    )


def _classify_indexed_mp_line(
    mp_owned: OwnedGeometryArray,
    line_owned: OwnedGeometryArray,
    mp_indices: np.ndarray,
    line_indices: np.ndarray,
    *,
    line_family: GeometryFamily,
) -> np.ndarray:
    """MULTIPOINT x LINESTRING/MULTILINESTRING relation bits."""
    n = mp_indices.size
    mp_state = mp_owned._ensure_device_state()
    line_state = line_owned._ensure_device_state()
    mp_buffer = mp_state.families[GeometryFamily.MULTIPOINT]
    line_buffer = line_state.families[line_family]
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    device_mp_fro = _prepare_indexed_fro(mp_owned, mp_indices, runtime)
    device_line_fro = _prepare_indexed_fro(line_owned, line_indices, runtime)
    identity_rows = np.arange(n, dtype=np.int32)
    kernel_name = (
        "multipoint_linestring_relation_compacted"
        if line_family is GeometryFamily.LINESTRING
        else "multipoint_multilinestring_relation_compacted"
    )
    args = [
        ptr(device_mp_fro),
        ptr(mp_buffer.geometry_offsets),
        ptr(mp_buffer.empty_mask),
        ptr(mp_buffer.x),
        ptr(mp_buffer.y),
        ptr(device_line_fro),
        ptr(line_buffer.geometry_offsets),
    ]
    if line_family is not GeometryFamily.LINESTRING:
        args.append(ptr(line_buffer.part_offsets))
    args.extend([
        ptr(line_buffer.empty_mask),
        ptr(line_buffer.x),
        ptr(line_buffer.y),
    ])
    return _launch_kernel(
        _multipoint_relation_kernels, kernel_name,
        identity_rows, tuple(args), (KERNEL_PARAM_PTR,) * len(args),
        extra_device_allocs=[device_mp_fro, device_line_fro],
    )


def _classify_indexed_mp_region(
    mp_owned: OwnedGeometryArray,
    region_owned: OwnedGeometryArray,
    mp_indices: np.ndarray,
    region_indices: np.ndarray,
    *,
    region_family: GeometryFamily,
) -> np.ndarray:
    """MULTIPOINT x POLYGON/MULTIPOLYGON relation bits."""
    n = mp_indices.size
    mp_state = mp_owned._ensure_device_state()
    region_state = region_owned._ensure_device_state()
    mp_buffer = mp_state.families[GeometryFamily.MULTIPOINT]
    region_buffer = region_state.families[region_family]
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    device_mp_fro = _prepare_indexed_fro(mp_owned, mp_indices, runtime)
    device_region_fro = _prepare_indexed_fro(region_owned, region_indices, runtime)
    identity_rows = np.arange(n, dtype=np.int32)
    kernel_name = (
        "multipoint_polygon_relation_compacted"
        if region_family is GeometryFamily.POLYGON
        else "multipoint_multipolygon_relation_compacted"
    )
    args = [
        ptr(device_mp_fro),
        ptr(mp_buffer.geometry_offsets),
        ptr(mp_buffer.empty_mask),
        ptr(mp_buffer.x),
        ptr(mp_buffer.y),
        ptr(device_region_fro),
        ptr(region_buffer.empty_mask),
        ptr(region_buffer.geometry_offsets),
    ]
    if region_family is not GeometryFamily.POLYGON:
        args.append(ptr(region_buffer.part_offsets))
    args.extend([
        ptr(region_buffer.ring_offsets),
        ptr(region_buffer.x),
        ptr(region_buffer.y),
    ])
    return _launch_kernel(
        _multipoint_relation_kernels, kernel_name,
        identity_rows, tuple(args), (KERNEL_PARAM_PTR,) * len(args),
        extra_device_allocs=[device_mp_fro, device_region_fro],
    )


def _classify_indexed_mp_mp(
    left_owned: OwnedGeometryArray,
    right_owned: OwnedGeometryArray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
) -> np.ndarray:
    """MULTIPOINT x MULTIPOINT relation bits (left MP vs right MP)."""
    n = left_indices.size
    left_state = left_owned._ensure_device_state()
    right_state = right_owned._ensure_device_state()
    left_buffer = left_state.families[GeometryFamily.MULTIPOINT]
    right_buffer = right_state.families[GeometryFamily.MULTIPOINT]
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    device_left_fro = _prepare_indexed_fro(left_owned, left_indices, runtime)
    device_right_fro = _prepare_indexed_fro(right_owned, right_indices, runtime)
    identity_rows = np.arange(n, dtype=np.int32)
    return _launch_kernel(
        _multipoint_relation_kernels,
        "multipoint_multipoint_relation_compacted", identity_rows,
        (
            ptr(device_left_fro),
            ptr(left_buffer.geometry_offsets),
            ptr(left_buffer.empty_mask),
            ptr(left_buffer.x),
            ptr(left_buffer.y),
            ptr(device_right_fro),
            ptr(right_buffer.geometry_offsets),
            ptr(right_buffer.empty_mask),
            ptr(right_buffer.x),
            ptr(right_buffer.y),
        ),
        (KERNEL_PARAM_PTR,) * 10,
        extra_device_allocs=[device_left_fro, device_right_fro],
    )


def _dispatch_multipoint_pairs(
    predicate: str,
    out: np.ndarray,
    left_owned: OwnedGeometryArray,
    right_owned: OwnedGeometryArray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    left_tags: np.ndarray,
    right_tags: np.ndarray,
    mp_left_mask: np.ndarray,
    mp_right_mask: np.ndarray,
    _apply_relation_rows,
) -> None:
    """Dispatch multipoint pairs to the appropriate GPU kernel and convert results."""
    pt_tag = _POINT_TAG_INDEXED

    # --- MULTIPOINT on left ---

    # MP x point
    mask = mp_left_mask & (right_tags == pt_tag)
    if mask.any():
        idx = np.flatnonzero(mask)
        bits = _classify_indexed_mp_point(
            left_owned, right_owned, left_indices[idx], right_indices[idx],
        )
        _apply_relation_rows(out, idx, _multipoint_bits_to_predicate(
            predicate, bits, mp_on_left=True, target_family=GeometryFamily.POINT,
        ))

    # MP x line families
    for lf, lt in zip(_LINE_FAMILIES_INDEXED, _LINE_TAGS_INDEXED, strict=True):
        mask = mp_left_mask & (right_tags == lt)
        if mask.any():
            idx = np.flatnonzero(mask)
            bits = _classify_indexed_mp_line(
                left_owned, right_owned, left_indices[idx], right_indices[idx],
                line_family=lf,
            )
            _apply_relation_rows(out, idx, _multipoint_bits_to_predicate(
                predicate, bits, mp_on_left=True, target_family=lf,
            ))

    # MP x region families
    for rf, rt in zip(_REGION_FAMILIES_INDEXED, _REGION_TAGS_INDEXED, strict=True):
        mask = mp_left_mask & (right_tags == rt)
        if mask.any():
            idx = np.flatnonzero(mask)
            bits = _classify_indexed_mp_region(
                left_owned, right_owned, left_indices[idx], right_indices[idx],
                region_family=rf,
            )
            _apply_relation_rows(out, idx, _multipoint_bits_to_predicate(
                predicate, bits, mp_on_left=True, target_family=rf,
            ))

    # MP x MP: the kernel checks each LEFT-MP coord against RIGHT-MP.
    # For contains/covers/contains_properly we also need the reverse check
    # (each RIGHT-MP coord against LEFT-MP) to verify the right side is a
    # subset of the left side.
    mask = mp_left_mask & mp_right_mask
    if mask.any():
        idx = np.flatnonzero(mask)
        li, ri = left_indices[idx], right_indices[idx]
        bits_fwd = _classify_indexed_mp_mp(left_owned, right_owned, li, ri)

        if predicate in {"contains", "covers", "contains_properly"}:
            # Reverse: check each right-MP coord against left-MP.
            bits_rev = _classify_indexed_mp_mp(right_owned, left_owned, ri, li)
            # "contains" = right subset of left = ~any_outside in reverse
            result = _multipoint_bits_to_predicate(
                predicate, bits_rev, mp_on_left=False,
                target_family=GeometryFamily.MULTIPOINT,
            )
        elif predicate in {"within", "covered_by"}:
            # Forward bits already tell us if left is subset of right.
            result = _multipoint_bits_to_predicate(
                predicate, bits_fwd, mp_on_left=True,
                target_family=GeometryFamily.MULTIPOINT,
            )
        else:
            # Symmetric predicates -- forward bits suffice.
            result = _multipoint_bits_to_predicate(
                predicate, bits_fwd, mp_on_left=True,
                target_family=GeometryFamily.MULTIPOINT,
            )
        _apply_relation_rows(out, idx, result)

    # --- MULTIPOINT on right only (left is not multipoint) ---

    # point x MP
    mask = (left_tags == pt_tag) & mp_right_mask
    if mask.any():
        idx = np.flatnonzero(mask)
        bits = _classify_indexed_mp_point(
            right_owned, left_owned, right_indices[idx], left_indices[idx],
        )
        _apply_relation_rows(out, idx, _multipoint_bits_to_predicate(
            predicate, bits, mp_on_left=False, target_family=GeometryFamily.POINT,
        ))

    # line x MP
    for lf, lt in zip(_LINE_FAMILIES_INDEXED, _LINE_TAGS_INDEXED, strict=True):
        mask = (left_tags == lt) & mp_right_mask
        if mask.any():
            idx = np.flatnonzero(mask)
            bits = _classify_indexed_mp_line(
                right_owned, left_owned, right_indices[idx], left_indices[idx],
                line_family=lf,
            )
            _apply_relation_rows(out, idx, _multipoint_bits_to_predicate(
                predicate, bits, mp_on_left=False, target_family=lf,
            ))

    # region x MP
    for rf, rt in zip(_REGION_FAMILIES_INDEXED, _REGION_TAGS_INDEXED, strict=True):
        mask = (left_tags == rt) & mp_right_mask
        if mask.any():
            idx = np.flatnonzero(mask)
            bits = _classify_indexed_mp_region(
                right_owned, left_owned, right_indices[idx], left_indices[idx],
                region_family=rf,
            )
            _apply_relation_rows(out, idx, _multipoint_bits_to_predicate(
                predicate, bits, mp_on_left=False, target_family=rf,
            ))
