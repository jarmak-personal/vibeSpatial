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
from vibespatial.predicates.polygon_kernels import (
    _POLYGON_PREDICATES_KERNEL_NAMES,
    _POLYGON_PREDICATES_KERNEL_SOURCE,
)

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

request_nvrtc_warmup([
    ("polygon-predicates", _POLYGON_PREDICATES_KERNEL_SOURCE, _POLYGON_PREDICATES_KERNEL_NAMES),
])


def _polygon_predicates_kernels():
    return compile_kernel_group("polygon-predicates", _POLYGON_PREDICATES_KERNEL_SOURCE, _POLYGON_PREDICATES_KERNEL_NAMES)


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


# ---------------------------------------------------------------------------
# Kernel dispatch
# ---------------------------------------------------------------------------

# Maps (left_family, right_family) → kernel name.
_KERNEL_MAP: dict[tuple[GeometryFamily, GeometryFamily], str] = {
    # Polygon × Polygon
    (GeometryFamily.POLYGON, GeometryFamily.POLYGON): "polygon_polygon_de9im_from_owned",
    (GeometryFamily.MULTIPOLYGON, GeometryFamily.MULTIPOLYGON): "multipolygon_multipolygon_de9im_from_owned",
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
    (GeometryFamily.POLYGON, GeometryFamily.POLYGON): "polygon_polygon_covered_by_mask_no_holes",
    (GeometryFamily.MULTIPOLYGON, GeometryFamily.POLYGON): "multipolygon_polygon_covered_by_mask_no_holes",
    (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON): "polygon_multipolygon_covered_by_mask_no_holes",
    (GeometryFamily.MULTIPOLYGON, GeometryFamily.MULTIPOLYGON): (
        "multipolygon_multipolygon_covered_by_mask_no_holes"
    ),
}


_LINE_FAMILIES = frozenset({GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING})
_POLYGON_FAMILIES = frozenset({GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON})

# Pairs that require swapping (A,B) to (B,A) before kernel dispatch.
# The kernel is written with a specific left/right layout, so we swap
# and transpose the DE-9IM result.
_SWAP_PAIRS: dict[tuple[GeometryFamily, GeometryFamily], tuple[GeometryFamily, GeometryFamily]] = {
    # MPG×PG → PG×MPG
    (GeometryFamily.MULTIPOLYGON, GeometryFamily.POLYGON):
        (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON),
    # MLS×LS → LS×MLS
    (GeometryFamily.MULTILINESTRING, GeometryFamily.LINESTRING):
        (GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING),
    # PG×LS → LS×PG
    (GeometryFamily.POLYGON, GeometryFamily.LINESTRING):
        (GeometryFamily.LINESTRING, GeometryFamily.POLYGON),
    # PG×MLS → MLS×PG
    (GeometryFamily.POLYGON, GeometryFamily.MULTILINESTRING):
        (GeometryFamily.MULTILINESTRING, GeometryFamily.POLYGON),
    # MPG×LS → LS×MPG
    (GeometryFamily.MULTIPOLYGON, GeometryFamily.LINESTRING):
        (GeometryFamily.LINESTRING, GeometryFamily.MULTIPOLYGON),
    # MPG×MLS → MLS×MPG
    (GeometryFamily.MULTIPOLYGON, GeometryFamily.MULTILINESTRING):
        (GeometryFamily.MULTILINESTRING, GeometryFamily.MULTIPOLYGON),
}


def _build_side_args(ptr, state, buf, family):
    """Build kernel args + types for one side of a DE-9IM kernel call."""
    P = KERNEL_PARAM_PTR
    I32 = KERNEL_PARAM_I32

    args = [
        ptr(state.validity), ptr(state.tags), ptr(state.family_row_offsets),
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

    query_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason=f"covered_by no-hole mask GPU: query {query_family.name}",
    )
    mask_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason=f"covered_by no-hole mask GPU: mask {mask_family.name}",
    )

    query_state = query_owned._ensure_device_state()
    mask_state = mask_owned._ensure_device_state()
    query_buf = query_state.families[query_family]
    mask_buf = mask_state.families[mask_family]

    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    if d_left is not None:
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
        tail_args = [ptr(d_left), ptr(d_out), pair_count, 0]
        tail_types = [P, P, I32, I32]
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

    query_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason=f"de9im GPU: query {query_family.name}",
    )
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

    query_state = eff_query_owned._ensure_device_state()
    tree_state = eff_tree_owned._ensure_device_state()
    query_buf = query_state.families[eff_query_family]
    tree_buf = tree_state.families[eff_tree_family]

    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    # Derive pair_count: prefer device arrays (avoids need for host
    # placeholder), then fall back to host indices.
    if eff_d_left is not None:
        pair_count = eff_d_left.shape[0]
    elif eff_left is not None:
        pair_count = eff_left.size
    else:
        raise ValueError(
            "compute_polygon_de9im_gpu requires either d_left or left_indices"
        )

    # Use device-resident arrays when provided; otherwise upload from host.
    own_d_left = eff_d_left is None
    own_d_right = eff_d_right is None
    if own_d_left:
        eff_d_left = runtime.from_host(np.ascontiguousarray(eff_left, dtype=np.int32))
    if own_d_right:
        eff_d_right = runtime.from_host(np.ascontiguousarray(eff_right, dtype=np.int32))
    d_mask = runtime.allocate((pair_count,), np.uint16)

    try:
        kernels = _polygon_predicates_kernels()
        P = KERNEL_PARAM_PTR
        I32 = KERNEL_PARAM_I32

        left_args, left_types = _build_side_args(ptr, query_state, query_buf, eff_query_family)
        right_args, right_types = _build_side_args(ptr, tree_state, tree_buf, eff_tree_family)
        tail_args = [ptr(eff_d_left), ptr(eff_d_right), ptr(d_mask), pair_count]
        tail_types = [P, P, P, I32]

        all_args = tuple(left_args + right_args + tail_args)
        all_types = tuple(left_types + right_types + tail_types)

        grid, block = runtime.launch_config(kernels[kernel_name], pair_count)
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
                d_mask = _transpose_de9im_device(d_mask)
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
        if not return_device:
            runtime.free(d_mask)


def _transpose_de9im(masks: np.ndarray) -> np.ndarray:
    """Transpose DE-9IM bitmasks (swap A and B roles)."""
    m = masks.astype(np.uint16, copy=True)
    out = np.zeros_like(m)
    # II stays, EE stays.
    out |= (m & DE9IM_II)
    out |= (m & DE9IM_EE)
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
    out |= (m & DE9IM_BB)
    return out


def _transpose_de9im_device(d_masks: object) -> object:
    """Transpose DE-9IM bitmasks on device (CuPy — Tier 2 element-wise).

    Device-resident mirror of ``_transpose_de9im`` that keeps data on GPU.
    """
    import cupy as cp

    m = d_masks.astype(cp.uint16, copy=False)
    out = cp.zeros_like(m)
    # II stays, EE stays.
    out |= m & DE9IM_II
    out |= m & DE9IM_EE
    # Swap IB <-> BI.
    out |= cp.where(m & DE9IM_IB, DE9IM_BI, 0).astype(cp.uint16)
    out |= cp.where(m & DE9IM_BI, DE9IM_IB, 0).astype(cp.uint16)
    # Swap IE <-> EI.
    out |= cp.where(m & DE9IM_IE, DE9IM_EI, 0).astype(cp.uint16)
    out |= cp.where(m & DE9IM_EI, DE9IM_IE, 0).astype(cp.uint16)
    # Swap BE <-> EB.
    out |= cp.where(m & DE9IM_BE, DE9IM_EB, 0).astype(cp.uint16)
    out |= cp.where(m & DE9IM_EB, DE9IM_BE, 0).astype(cp.uint16)
    # BB stays.
    out |= m & DE9IM_BB
    return out


def supported_predicate_families() -> frozenset[tuple[GeometryFamily, GeometryFamily]]:
    """Return the set of family pairs supported by polygon predicate kernels."""
    return frozenset(_KERNEL_MAP.keys())
