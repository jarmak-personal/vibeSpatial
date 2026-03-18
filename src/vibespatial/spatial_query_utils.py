from __future__ import annotations

from typing import Any

import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry

from vibespatial.binary_predicates import evaluate_binary_predicate
from vibespatial.geometry_buffers import GeometryFamily, get_geometry_buffer_schema
from vibespatial.owned_geometry import (
    FAMILY_TAGS,
    TAG_FAMILIES,
    DiagnosticKind,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
    from_shapely_geometries,
)
from vibespatial.residency import Residency
from vibespatial.runtime import ExecutionMode, RuntimeSelection, has_gpu_runtime
from vibespatial.spatial_query_types import (
    _POLYGON_DE9IM_PREDICATES,
    SUPPORTED_GEOM_TYPES,
)


def _as_geometry_array(geometry: Any) -> tuple[np.ndarray | None, bool]:
    if isinstance(geometry, OwnedGeometryArray):
        return np.asarray(geometry.to_shapely(), dtype=object), False
    if isinstance(geometry, np.ndarray):
        if geometry.ndim == 0:
            return np.asarray([geometry.item()], dtype=object), True
        return np.asarray(geometry, dtype=object), False
    if isinstance(geometry, BaseGeometry):
        return np.asarray([geometry], dtype=object), True
    if geometry is None:
        return None, True
    if hasattr(geometry, "_data"):
        return np.asarray(geometry._data, dtype=object), False
    if hasattr(geometry, "values") and hasattr(geometry.values, "_data"):
        return np.asarray(geometry.values._data, dtype=object), False
    values = np.asarray(geometry, dtype=object)
    if values.ndim == 0:
        return np.asarray([values.item()], dtype=object), True
    return values, False


def supports_owned_spatial_input(geometry: Any) -> bool:
    values, _ = _as_geometry_array(geometry)
    if values is None:
        return True
    for value in values:
        if value is None:
            continue
        if not hasattr(value, "geom_type") or not hasattr(value, "is_empty"):
            return False
        if value.geom_type not in SUPPORTED_GEOM_TYPES:
            return False
        # GeometryCollection passes the type check but cannot be serialized
        # into owned geometry buffers — reject it so the caller falls back
        # to the STRtree host path.
        if value.geom_type == "GeometryCollection":
            return False
    return True


def _to_owned_points_fast(values: np.ndarray) -> OwnedGeometryArray | None:
    """Vectorized fast-path for all-Point arrays using shapely.get_coordinates().

    Returns None if any geometry is not a simple Point (i.e. MultiPoint,
    LineString, etc.) so the caller falls back to the generic Python loop.
    Uses shapely.get_type_id() (vectorized C) to detect all-Point arrays,
    then shapely.get_coordinates() (~1ms for 100k points) instead of the
    per-object Python loop in from_shapely_geometries() (~500ms-2s).
    """
    n = len(values)
    if n == 0:
        return None

    # Check for None/missing values.
    none_mask = shapely.is_missing(values)
    has_none = bool(none_mask.any())

    # Check that all non-None geometries are simple Points (type_id == 0).
    if has_none:
        non_none_mask = ~none_mask
        if not non_none_mask.any():
            # All None -- fall back to generic path.
            return None
        type_ids = shapely.get_type_id(values)
        valid_type_ids = type_ids[non_none_mask]
        if not np.all(valid_type_ids == 0):
            return None
        # Check for empty points.
        empty_flags = shapely.is_empty(values)
        valid_empty = empty_flags[non_none_mask]
    else:
        type_ids = shapely.get_type_id(values)
        if not np.all(type_ids == 0):
            return None
        empty_flags = shapely.is_empty(values)
        valid_empty = empty_flags

    # Extract coordinates vectorized -- ~1ms for 100k points.
    # For non-empty points this returns an (M, 2) array of coords.
    # We need to handle None and empty points specially.
    validity = ~none_mask
    point_tag = FAMILY_TAGS[GeometryFamily.POINT]
    tags = np.full(n, -1, dtype=np.int8)
    family_row_offsets = np.full(n, -1, dtype=np.int32)

    valid_indices = np.flatnonzero(validity)
    n_valid = valid_indices.size
    tags[valid_indices] = point_tag
    family_row_offsets[valid_indices] = np.arange(n_valid, dtype=np.int32)

    # Build empty_mask for the point family buffer.
    # valid_empty is already indexed to only the valid (non-None) geometries,
    # so it has length n_valid and maps 1:1 to the family rows.
    empty_mask = np.asarray(valid_empty, dtype=bool)

    # Extract coordinates for valid, non-empty points.
    non_empty_valid_mask = validity & ~empty_flags
    non_empty_geoms = values[non_empty_valid_mask]
    if non_empty_geoms.size > 0:
        coords = shapely.get_coordinates(non_empty_geoms)
        x_coords = np.ascontiguousarray(coords[:, 0])
        y_coords = np.ascontiguousarray(coords[:, 1])
    else:
        x_coords = np.asarray([], dtype=np.float64)
        y_coords = np.asarray([], dtype=np.float64)

    # Build geometry_offsets: [0, 1, 2, ..., n_valid] for simple points,
    # but empty points contribute 0 coords.
    # For each valid point: non-empty contributes 1 coord, empty contributes 0.
    coord_counts = np.ones(n_valid, dtype=np.int32)
    coord_counts[empty_mask] = 0
    geometry_offsets = np.empty(n_valid + 1, dtype=np.int32)
    geometry_offsets[0] = 0
    np.cumsum(coord_counts, out=geometry_offsets[1:])

    point_buffer = FamilyGeometryBuffer(
        family=GeometryFamily.POINT,
        schema=get_geometry_buffer_schema(GeometryFamily.POINT),
        row_count=n_valid,
        x=x_coords,
        y=y_coords,
        geometry_offsets=geometry_offsets,
        empty_mask=empty_mask,
    )

    array = OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families={GeometryFamily.POINT: point_buffer},
        residency=Residency.HOST,
    )
    array._record(DiagnosticKind.CREATED, "created owned geometry array from shapely points (vectorized fast-path)", visible=True)
    return array


def _to_owned(values: np.ndarray | None) -> OwnedGeometryArray:
    if values is None:
        return from_shapely_geometries([])
    # Fast-path: all-Point arrays avoid the Python per-object loop.
    fast = _to_owned_points_fast(values)
    if fast is not None:
        return fast
    sanitized = []
    for value in values.tolist():
        if value is None:
            sanitized.append(None)
            continue
        if value.geom_type not in SUPPORTED_GEOM_TYPES:
            sanitized.append(None)
            continue
        # GeometryCollection is accepted by supports_owned_spatial_input so
        # it doesn't block the owned path, but the owned geometry buffers
        # cannot serialize it — treat as empty/None for candidate gen.
        if value.geom_type == "GeometryCollection":
            sanitized.append(None)
            continue
        sanitized.append(value)
    return from_shapely_geometries(sanitized)


def _sort_indices(indices: np.ndarray) -> np.ndarray:
    if indices.ndim == 1:
        return np.sort(indices)
    sort_indexer = np.lexsort((indices[1], indices[0]))
    return indices[:, sort_indexer]


def _indices_to_dense(indices: np.ndarray, tree_size: int, query_size: int, scalar: bool) -> np.ndarray:
    if scalar:
        dense = np.zeros(tree_size, dtype=bool)
        dense[indices] = True
        return dense
    dense = np.zeros((tree_size, query_size), dtype=bool)
    dense[indices[1], indices[0]] = True
    return dense


def _indices_to_sparse(indices: np.ndarray, tree_size: int, query_size: int, scalar: bool):
    from scipy import sparse

    if scalar:
        return sparse.coo_array(
            (np.ones(len(indices), dtype=np.bool_), indices.reshape(1, -1)),
            shape=(tree_size,),
            dtype=np.bool_,
        )
    return sparse.coo_array(
        (np.ones(indices.shape[1], dtype=np.bool_), indices[::-1]),
        shape=(tree_size, query_size),
        dtype=np.bool_,
    )


def _expand_bounds(bounds: np.ndarray, distances: np.ndarray) -> np.ndarray:
    expanded = bounds.copy()
    expanded[:, 0] -= distances
    expanded[:, 1] -= distances
    expanded[:, 2] += distances
    expanded[:, 3] += distances
    return expanded


def _gpu_bounds_dispatch_mode() -> ExecutionMode:
    return ExecutionMode.GPU if has_gpu_runtime() else ExecutionMode.CPU


def _filter_predicate_pairs(
    predicate: str | None,
    query_values: np.ndarray,
    tree_values: np.ndarray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    *,
    distance: float | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, RuntimeSelection]:
    if predicate is None:
        return (
            left_indices,
            right_indices,
            RuntimeSelection(
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.CPU,
                reason="bbox-only query does not invoke exact predicate refinement",
            ),
        )
    if left_indices.size == 0:
        return (
            left_indices,
            right_indices,
            RuntimeSelection(
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.CPU,
                reason="no candidate pairs reached exact predicate refinement",
            ),
        )
    if predicate == "dwithin":
        exact = shapely.dwithin(
            query_values[left_indices],
            tree_values[right_indices],
            distance if np.isscalar(distance) else np.asarray(distance)[left_indices],
        )
        keep = np.asarray(exact, dtype=bool)
        selection = RuntimeSelection(
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.CPU,
            reason="dwithin exact refinement currently uses the Shapely host path",
        )
    else:
        exact = evaluate_binary_predicate(
            predicate,
            query_values[left_indices],
            tree_values[right_indices],
            dispatch_mode=ExecutionMode.AUTO,
            null_behavior="false",
        )
        keep = np.asarray(exact.values, dtype=bool)
        selection = exact.runtime_selection
    return left_indices[keep], right_indices[keep], selection


def _filter_predicate_pairs_owned(
    predicate: str | None,
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
    left_indices: np.ndarray | None,
    right_indices: np.ndarray | None,
    *,
    query_shapely: np.ndarray | None = None,
    tree_shapely: np.ndarray | None = None,
    device_candidates: object | None = None,
) -> tuple[np.ndarray, np.ndarray, RuntimeSelection]:
    """Exact predicate refinement using indexed access into original arrays.

    Avoids the expensive OwnedGeometryArray.take() buffer copy by evaluating
    predicates directly using indexed access into the original geometry arrays.

    When *device_candidates* is a :class:`_DeviceCandidates`, the device-
    resident index arrays are used directly for GPU kernel dispatch, avoiding
    redundant host→device transfers.  If *left_indices* / *right_indices* are
    ``None``, they are lazily materialised from *device_candidates* only when
    the host arrays are actually needed (Shapely fallback).

    Tag classification is performed on-device when device candidates are
    available (ADR-0005: no mid-pipeline D→H transfers).

    For GPU-supported predicate pairs (point vs point/line/region), creates
    compact 1:1 owned arrays via take() for the GPU kernel path.
    For all other pairs, evaluates via Shapely using direct fancy indexing
    into the original Shapely arrays — no buffer copy, no redundant bounds
    computation.

    When *query_shapely* / *tree_shapely* are provided, the Shapely
    materialisation step is skipped (the caller already has the arrays).
    """
    _dc = device_candidates
    _has_device = _dc is not None and hasattr(_dc, "d_left")

    # Determine total_pairs without materialising host indices.
    if left_indices is not None:
        _total = left_indices.size
    elif _has_device:
        _total = _dc.total_pairs
    else:
        _total = 0

    # Fast exit: no predicate or no pairs — materialise host indices lazily.
    if predicate is None or _total == 0:
        if left_indices is None or right_indices is None:
            if _has_device:
                left_indices, right_indices = _dc.to_host()
            else:
                empty = np.asarray([], dtype=np.int32)
                left_indices = left_indices if left_indices is not None else empty
                right_indices = right_indices if right_indices is not None else empty
        if predicate is None:
            return (
                left_indices,
                right_indices,
                RuntimeSelection(
                    requested=ExecutionMode.AUTO,
                    selected=ExecutionMode.CPU,
                    reason="bbox-only query does not invoke exact predicate refinement",
                ),
            )
        return (
            left_indices,
            right_indices,
            RuntimeSelection(
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.CPU,
                reason="no candidate pairs reached exact predicate refinement",
            ),
        )

    # Tag classification — prefer device-side when candidates are device-resident
    # (ADR-0005: no eager D→H for tag lookup).
    if _has_device and has_gpu_runtime():
        import cupy as _cp
        d_query_tags = _cp.asarray(query_owned.tags)  # tiny: 1 byte per row
        d_tree_tags = _cp.asarray(tree_owned.tags)
        d_left_tags = d_query_tags[_dc.d_left]
        d_right_tags = d_tree_tags[_dc.d_right]

        point_tag = FAMILY_TAGS[GeometryFamily.POINT]
        mp_tag = FAMILY_TAGS[GeometryFamily.MULTIPOINT]
        line_tags_arr = _cp.array([FAMILY_TAGS[f] for f in (GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING)], dtype=_cp.int8)
        region_tags_arr = _cp.array([FAMILY_TAGS[f] for f in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON)], dtype=_cp.int8)
        all_non_de9im = _cp.concatenate([_cp.array([point_tag, mp_tag], dtype=_cp.int8), line_tags_arr, region_tags_arr])

        d_left_is_point = (d_left_tags == point_tag) | (d_left_tags == mp_tag)
        d_right_is_point = (d_right_tags == point_tag) | (d_right_tags == mp_tag)
        d_gpu_pair_mask = (
            (d_left_is_point & _cp.isin(d_right_tags, all_non_de9im))
            | (d_right_is_point & _cp.isin(d_left_tags, all_non_de9im))
        )

        all_gpu = bool(_cp.all(d_gpu_pair_mask))
        any_gpu = bool(_cp.any(d_gpu_pair_mask))

        # Materialise host arrays only when needed below.
        # Tag arrays stay on device; host copies are deferred.
        left_tags = None  # will be set from device if needed
        right_tags = None
        gpu_pair_mask = None  # will be set from device if needed

        # Helper to lazily get host-side tags and mask.
        def _ensure_host_tags():
            nonlocal left_tags, right_tags, gpu_pair_mask, left_indices, right_indices
            if left_indices is None or right_indices is None:
                left_indices, right_indices = _dc.to_host()
            if left_tags is None:
                left_tags = _cp.asnumpy(d_left_tags)
                right_tags = _cp.asnumpy(d_right_tags)
                gpu_pair_mask = _cp.asnumpy(d_gpu_pair_mask)

        # Convert device tag arrays to host-compatible numpy references for
        # downstream code that indexes with them.
        line_tags = np.array([FAMILY_TAGS[f] for f in (GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING)], dtype=np.int8)
        region_tags = np.array([FAMILY_TAGS[f] for f in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON)], dtype=np.int8)
    else:
        # No device candidates — materialise host indices eagerly.
        if left_indices is None or right_indices is None:
            if _has_device:
                left_indices, right_indices = _dc.to_host()
            else:
                empty = np.asarray([], dtype=np.int32)
                left_indices = left_indices if left_indices is not None else empty
                right_indices = right_indices if right_indices is not None else empty

        left_tags = query_owned.tags[left_indices]
        right_tags = tree_owned.tags[right_indices]

        point_tag = FAMILY_TAGS[GeometryFamily.POINT]
        mp_tag = FAMILY_TAGS[GeometryFamily.MULTIPOINT]
        line_tags = np.array([FAMILY_TAGS[f] for f in (GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING)], dtype=np.int8)
        region_tags = np.array([FAMILY_TAGS[f] for f in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON)], dtype=np.int8)
        all_non_de9im = np.concatenate([np.array([point_tag, mp_tag], dtype=np.int8), line_tags, region_tags])

        left_is_point = (left_tags == point_tag) | (left_tags == mp_tag)
        right_is_point = (right_tags == point_tag) | (right_tags == mp_tag)
        gpu_pair_mask = (
            (left_is_point & np.isin(right_tags, all_non_de9im))
            | (right_is_point & np.isin(left_tags, all_non_de9im))
        )

        all_gpu = bool(np.all(gpu_pair_mask))
        any_gpu = bool(np.any(gpu_pair_mask)) and has_gpu_runtime()

        _ensure_host_tags = lambda: None  # noqa: E731 — no-op, already on host

    # Fast path: ALL pairs support GPU predicate evaluation.
    # Uses indexed access into original owned arrays — no take() buffer copy.
    if all_gpu and has_gpu_runtime():
        _ensure_host_tags()
        from vibespatial.point_binary_relations import classify_point_predicates_indexed
        keep = classify_point_predicates_indexed(
            predicate, query_owned, tree_owned, left_indices, right_indices,
        )
        return (
            left_indices[keep],
            right_indices[keep],
            RuntimeSelection(
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.GPU,
                reason=f"GPU indexed point-family {predicate} refinement (no take copy)",
            ),
        )

    # --- GPU DE-9IM predicate path ---
    # When all non-GPU pairs are line/polygon families, use the DE-9IM kernel.
    _ensure_host_tags()
    non_point_tags = np.concatenate([line_tags, region_tags])
    de9im_pair_mask = (
        np.isin(left_tags, non_point_tags) & np.isin(right_tags, non_point_tags)
    )
    non_gpu_mask = ~gpu_pair_mask
    all_non_gpu_have_de9im = bool(np.all(de9im_pair_mask | gpu_pair_mask))

    if all_non_gpu_have_de9im and has_gpu_runtime() and predicate in _POLYGON_DE9IM_PREDICATES:
        keep = np.zeros(left_indices.size, dtype=bool)

        # Handle point-pairs via indexed access (no take copy).
        if any_gpu:
            from vibespatial.point_binary_relations import classify_point_predicates_indexed
            gpu_idx = np.flatnonzero(gpu_pair_mask)
            gpu_left = left_indices[gpu_idx]
            gpu_right = right_indices[gpu_idx]
            keep[gpu_idx] = classify_point_predicates_indexed(
                predicate, query_owned, tree_owned, gpu_left, gpu_right,
            )

        # Handle line/polygon pairs via DE-9IM kernel.
        de9im_idx = np.flatnonzero(non_gpu_mask & de9im_pair_mask)
        if de9im_idx.size > 0:
            de9im_left = left_indices[de9im_idx]
            de9im_right = right_indices[de9im_idx]
            de9im_left_tags = left_tags[de9im_idx]
            de9im_right_tags = right_tags[de9im_idx]

            # When device candidates are available, extract device sub-arrays
            # via CuPy fancy indexing to avoid re-uploading indices.
            _dc = device_candidates
            _use_device_idx = _dc is not None and hasattr(_dc, "d_left")

            # Group by (left_family, right_family) to dispatch correct kernel.
            de9im_masks = np.zeros(de9im_idx.size, dtype=np.uint16)
            unique_tag_pairs = set(zip(de9im_left_tags.tolist(), de9im_right_tags.tolist()))
            for (lt, rt) in unique_tag_pairs:
                sub_mask = (de9im_left_tags == lt) & (de9im_right_tags == rt)
                sub_idx = np.flatnonzero(sub_mask)
                if sub_idx.size == 0:
                    continue
                lf = TAG_FAMILIES.get(lt)
                rf = TAG_FAMILIES.get(rt)
                if lf is None or rf is None:
                    continue

                # Build device sub-arrays when candidates are device-resident.
                d_sub_left = None
                d_sub_right = None
                if _use_device_idx:
                    import cupy as cp
                    # de9im_idx[sub_idx] maps back to the full candidate array
                    full_sub_idx = de9im_idx[sub_idx]
                    d_full_idx = cp.asarray(full_sub_idx.astype(np.int32))
                    d_sub_left = _dc.d_left[d_full_idx]
                    d_sub_right = _dc.d_right[d_full_idx]

                from vibespatial.polygon_predicates import (
                    compute_polygon_de9im_gpu,
                    evaluate_predicate_from_de9im,
                )
                sub_result = compute_polygon_de9im_gpu(
                    query_owned, tree_owned,
                    de9im_left[sub_idx], de9im_right[sub_idx],
                    query_family=lf, tree_family=rf,
                    d_left=d_sub_left, d_right=d_sub_right,
                )
                if sub_result is not None:
                    de9im_masks[sub_idx] = sub_result

            keep[de9im_idx] = evaluate_predicate_from_de9im(de9im_masks, predicate)

        return (
            left_indices[keep],
            right_indices[keep],
            RuntimeSelection(
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.GPU,
                reason=f"GPU DE-9IM {predicate} refinement for non-point pairs",
            ),
        )

    # Non-GPU path: evaluate via Shapely with direct indexed access.
    # This avoids the expensive .take() buffer copy by indexing into
    # the original Shapely arrays with the candidate pair indices.
    if query_shapely is None:
        query_shapely = np.asarray(query_owned.to_shapely(), dtype=object)
    if tree_shapely is None:
        tree_shapely = np.asarray(tree_owned.to_shapely(), dtype=object)

    if not any_gpu:
        # All pairs go through Shapely — direct indexed evaluation.
        exact_values = getattr(shapely, predicate)(
            query_shapely[left_indices],
            tree_shapely[right_indices],
        )
        keep = np.asarray(exact_values, dtype=bool)
        return (
            left_indices[keep],
            right_indices[keep],
            RuntimeSelection(
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.CPU,
                reason=f"direct indexed Shapely {predicate} refinement (no buffer copy)",
            ),
        )

    # Mixed path: some pairs support GPU, others need Shapely.
    keep = np.zeros(left_indices.size, dtype=bool)
    gpu_idx = np.flatnonzero(gpu_pair_mask)
    cpu_idx = np.flatnonzero(~gpu_pair_mask)

    # GPU portion via indexed access (no take copy).
    from vibespatial.point_binary_relations import classify_point_predicates_indexed
    gpu_left = left_indices[gpu_idx]
    gpu_right = right_indices[gpu_idx]
    keep[gpu_idx] = classify_point_predicates_indexed(
        predicate, query_owned, tree_owned, gpu_left, gpu_right,
    )

    # CPU portion via direct Shapely indexing.
    cpu_left = left_indices[cpu_idx]
    cpu_right = right_indices[cpu_idx]
    cpu_exact = getattr(shapely, predicate)(
        query_shapely[cpu_left],
        tree_shapely[cpu_right],
    )
    keep[cpu_idx] = np.asarray(cpu_exact, dtype=bool)

    return (
        left_indices[keep],
        right_indices[keep],
        RuntimeSelection(
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.GPU,
            reason=f"mixed GPU/Shapely {predicate} refinement (GPU for point pairs, direct Shapely for remainder)",
        ),
    )


def _format_query_indices(
    indices: np.ndarray,
    *,
    tree_size: int,
    query_size: int,
    scalar: bool,
    sort: bool,
    output_format: str,
) -> Any:
    formatted = _sort_indices(indices) if sort else indices
    if output_format == "indices":
        # ADR-0036: spatial kernels produce integer index arrays only.
        if __debug__:
            assert isinstance(formatted, np.ndarray), (
                f"ADR-0036: expected ndarray, got {type(formatted).__name__}"
            )
            assert np.issubdtype(formatted.dtype, np.integer), (
                f"ADR-0036: expected integer dtype, got {formatted.dtype}"
            )
        return formatted
    if output_format == "dense":
        return _indices_to_dense(formatted, tree_size, query_size, scalar)
    if output_format == "sparse":
        return _indices_to_sparse(formatted, tree_size, query_size, scalar)
    if output_format == "count":
        if formatted.ndim == 1:
            return int(np.unique(formatted).size) if formatted.size > 0 else 0
        return int(np.unique(formatted[0]).size) if formatted.shape[1] > 0 else 0
    raise ValueError(
        f"Invalid output_format: '{output_format}'. Use one of 'indices', 'sparse', 'dense', 'count'."
    )
