"""Device-native lower-dimensional remnants for polygon constructive results."""

from __future__ import annotations

from vibespatial.api._native_results import (
    _geometry_composition_from_owned_parts_at_capacity,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime._runtime import has_gpu_runtime
from vibespatial.runtime.residency import Residency


def _valid_nonempty_device_mask(owned):
    """Return a capacity-aligned valid, non-empty device mask."""
    if owned.residency is not Residency.DEVICE or not has_gpu_runtime():
        return None

    import cupy as cp

    from vibespatial.geometry.owned import FAMILY_TAGS

    state = owned._ensure_device_state(preserve_indexed_view=True)
    d_keep = cp.asarray(state.validity, dtype=cp.bool_).copy()
    if state.trusted_all_non_empty is not True:
        d_tags = cp.asarray(state.tags, dtype=cp.int8)
        d_family_rows = cp.asarray(state.family_row_offsets, dtype=cp.int64)
        for family, buffer in state.families.items():
            d_empty = cp.asarray(buffer.empty_mask, dtype=cp.bool_)
            if d_empty.size == 0:
                continue
            d_family = d_keep & (d_tags == cp.int8(FAMILY_TAGS[family]))
            d_in_bounds = d_family & (d_family_rows >= 0) & (d_family_rows < cp.int64(d_empty.size))
            d_safe_rows = cp.where(d_in_bounds, d_family_rows, 0)
            d_keep &= ~d_family | (d_in_bounds & ~d_empty[d_safe_rows])
    return d_keep


def _native_valid_nonempty_device_mask(native_geometry):
    """Return a logical-row nonempty mask for owned or composed geometry."""
    from vibespatial.api._native_result_core import GeometryNativeResult

    if isinstance(native_geometry, GeometryNativeResult):
        return native_geometry.valid_nonempty_mask_device()
    return _valid_nonempty_device_mask(native_geometry)


def _difference_native_parts_at_output_rows(
    native_geometry,
    area_owned,
    *,
    row_count: int,
):
    """Difference concrete native parts against aligned area rows on device."""
    import cupy as cp

    from vibespatial.api._native_result_core import GeometryNativeResult
    from vibespatial.constructive.binary_constructive import _binary_constructive_gpu
    from vibespatial.geometry.owned import (
        device_mask_owned_capacity,
        device_select_owned_capacity_partitions,
    )

    if isinstance(native_geometry, GeometryNativeResult):
        if native_geometry.owned is not None:
            source_parts = (
                (
                    native_geometry.owned,
                    cp.arange(native_geometry.row_count, dtype=cp.int64),
                ),
            )
        elif native_geometry.composition is not None:
            source_parts = tuple(
                (
                    part.geometry.owned,
                    cp.asarray(part.output_rows, dtype=cp.int64),
                )
                for part in native_geometry.composition.parts
                if part.geometry.owned is not None
            )
        else:
            return None
    else:
        source_parts = ((native_geometry, cp.arange(row_count, dtype=cp.int64)),)

    result_parts = []
    for part_owned, d_output_rows in source_parts:
        if part_owned is None or int(part_owned.row_count) != int(d_output_rows.size):
            return None
        part_area = area_owned._device_indexed_take(
            d_output_rows,
            assume_unique_indices=True,
        )
        d_area_keep = _valid_nonempty_device_mask(part_area)
        if d_area_keep is None:
            return None
        difference = _binary_constructive_gpu(
            "difference",
            device_mask_owned_capacity(part_owned, d_area_keep),
            device_mask_owned_capacity(part_area, d_area_keep),
            dispatch_mode=ExecutionMode.GPU,
        )
        if difference is None or int(difference.row_count) != int(part_owned.row_count):
            return None
        if isinstance(difference, GeometryNativeResult):
            if difference.owned is not None:
                result_parts.append(
                    (
                        device_select_owned_capacity_partitions(
                            part_owned,
                            [(difference.owned, d_area_keep)],
                        ),
                        d_output_rows,
                    )
                )
                continue
            if difference.composition is None:
                return None
            for nested in difference.composition.parts:
                nested_owned = nested.geometry.owned
                if nested_owned is None:
                    return None
                d_nested_rows = cp.asarray(nested.output_rows, dtype=cp.int64)
                result_parts.append((nested_owned, d_output_rows[d_nested_rows]))
            continue
        result_parts.append(
            (
                device_select_owned_capacity_partitions(
                    part_owned,
                    [(difference, d_area_keep)],
                ),
                d_output_rows,
            )
        )
    return tuple(result_parts)


def _polygon_pair_boundary_remnant_components_capacity_device(
    left_owned,
    right_owned,
    area_owned,
    *,
    remnant_mask=None,
):
    """Build candidate-shaped boundary remnants and capacity-aligned masks."""
    if not has_gpu_runtime():
        return None
    if any(
        owned is None or owned.residency is not Residency.DEVICE
        for owned in (left_owned, right_owned, area_owned)
    ):
        return None
    row_count = int(area_owned.row_count)
    if int(left_owned.row_count) != row_count or int(right_owned.row_count) != row_count:
        return None

    import cupy as cp

    from vibespatial.constructive.binary_constructive import _binary_constructive_gpu
    from vibespatial.constructive.boundary import boundary_owned
    from vibespatial.geometry.owned import (
        device_physicalize_owned_row_selection_capacity,
    )

    if remnant_mask is not None:
        d_remnant = cp.asarray(remnant_mask, dtype=cp.bool_)
        if int(d_remnant.size) != row_count:
            return None
        left_boundary_input = device_physicalize_owned_row_selection_capacity(
            left_owned,
            d_remnant,
        )
        right_boundary_input = device_physicalize_owned_row_selection_capacity(
            right_owned,
            d_remnant,
        )
    else:
        left_boundary_input = left_owned
        right_boundary_input = right_owned

    left_boundary = boundary_owned(
        left_boundary_input,
        dispatch_mode=ExecutionMode.GPU,
    )
    right_boundary = boundary_owned(
        right_boundary_input,
        dispatch_mode=ExecutionMode.GPU,
    )
    boundary_overlap = _binary_constructive_gpu(
        "intersection",
        left_boundary,
        right_boundary,
        dispatch_mode=ExecutionMode.GPU,
    )
    if boundary_overlap is None or int(boundary_overlap.row_count) != row_count:
        return None
    d_overlap_keep = _native_valid_nonempty_device_mask(boundary_overlap)
    if d_overlap_keep is None:
        return None

    boundary_parts = _difference_native_parts_at_output_rows(
        boundary_overlap,
        area_owned,
        row_count=row_count,
    )
    if boundary_parts is None:
        return None, None, d_overlap_keep, None

    d_area_keep = _valid_nonempty_device_mask(area_owned)
    if d_area_keep is None:
        return None, None, d_overlap_keep, None
    d_boundary_keep = cp.zeros(row_count, dtype=cp.bool_)
    for boundary_part, d_part_rows in boundary_parts:
        d_part_keep = _valid_nonempty_device_mask(boundary_part)
        if d_part_keep is None:
            return None
        part_count = int(boundary_part.row_count)
        d_lanes = cp.arange(part_count, dtype=cp.int64)
        d_destinations = cp.where(
            d_part_keep,
            d_part_rows,
            cp.int64(row_count) + d_lanes,
        )
        d_scatter = cp.zeros(row_count + part_count, dtype=cp.bool_)
        d_scatter[d_destinations] = d_part_keep
        d_boundary_keep |= d_scatter[:row_count]
    return boundary_parts, d_area_keep, d_overlap_keep, d_boundary_keep


def polygon_pair_boundary_remnant_mask_capacity_device(
    left_owned,
    right_owned,
    area_owned,
    *,
    keep_area_mask,
    remnant_mask=None,
):
    """Classify dropped lower-dimensional pair remnants at row capacity.

    Physical shape: source boundaries lower to same-row segment candidate
    relations, exact intersection refinement, dynamic constructive assembly,
    and a row-capacity nonempty reduction. Rows without retained area consume
    the boundary-overlap mask directly; retained-area rows consume the exact
    boundary-minus-area remnant mask.
    """
    components = _polygon_pair_boundary_remnant_components_capacity_device(
        left_owned,
        right_owned,
        area_owned,
        remnant_mask=remnant_mask,
    )
    if components is None:
        return None

    import cupy as cp

    _, d_area_nonempty, d_overlap_keep, d_boundary_keep = components
    d_keep_area = cp.asarray(keep_area_mask, dtype=cp.bool_)
    if int(d_keep_area.size) != int(area_owned.row_count):
        return None
    if d_boundary_keep is None:
        d_supported = ~d_area_nonempty
        return (~d_area_nonempty & d_overlap_keep), d_supported
    return cp.where(d_area_nonempty, d_boundary_keep, d_overlap_keep), cp.ones(
        int(area_owned.row_count),
        dtype=cp.bool_,
    )


def polygon_pair_boundary_remnants_capacity_device(
    left_owned,
    right_owned,
    area_owned,
    *,
    crs,
    remnant_mask=None,
):
    """Compose area and boundary remnants at aligned pair capacity."""
    native_parts = getattr(
        area_owned,
        "_polygon_intersection_lower_dimensional_parts",
        None,
    )
    if native_parts is not None:
        import cupy as cp

        row_count = int(area_owned.row_count)
        d_area_keep = _valid_nonempty_device_mask(area_owned)
        if d_area_keep is None:
            return None
        d_boundary_keep = cp.zeros(row_count, dtype=cp.bool_)
        for part in native_parts:
            if part.residency is not Residency.DEVICE or int(part.row_count) != row_count:
                return None
            d_part_keep = _valid_nonempty_device_mask(part)
            if d_part_keep is None:
                return None
            d_boundary_keep |= cp.asarray(d_part_keep, dtype=cp.bool_)
        d_output_rows = cp.arange(row_count, dtype=cp.int64)
        geometry = _geometry_composition_from_owned_parts_at_capacity(
            (
                (area_owned, d_output_rows),
                *((part, d_output_rows) for part in native_parts),
            ),
            row_count=row_count,
            crs=crs,
        )
        if geometry is None:
            return None
        return geometry, d_area_keep | d_boundary_keep

    components = _polygon_pair_boundary_remnant_components_capacity_device(
        left_owned,
        right_owned,
        area_owned,
        remnant_mask=remnant_mask,
    )
    if components is None:
        return None

    import cupy as cp

    boundary_parts, d_area_keep, _, d_boundary_keep = components
    if boundary_parts is None or d_area_keep is None or d_boundary_keep is None:
        return None
    row_count = int(area_owned.row_count)
    d_output_rows = cp.arange(row_count, dtype=cp.int64)
    geometry = _geometry_composition_from_owned_parts_at_capacity(
        (
            (area_owned, d_output_rows),
            *boundary_parts,
        ),
        row_count=row_count,
        crs=crs,
    )
    if geometry is None:
        return None
    return geometry, d_area_keep | d_boundary_keep


def polygon_make_valid_linework_composition_device(
    source_owned,
    repaired_owned,
    repaired_rows,
    *,
    crs=None,
):
    """Compose repaired polygonal area with collapsed input boundary linework."""
    if not has_gpu_runtime():
        return None
    if any(
        owned is None or owned.residency is not Residency.DEVICE
        for owned in (source_owned, repaired_owned)
    ):
        return None
    if int(source_owned.row_count) != int(repaired_owned.row_count):
        return None

    import cupy as cp

    from vibespatial.constructive.binary_constructive import _binary_constructive_gpu
    from vibespatial.constructive.boundary import boundary_owned
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import FAMILY_TAGS, device_mask_owned_capacity

    d_repaired_rows = cp.asarray(repaired_rows, dtype=cp.int64)
    if int(d_repaired_rows.size) == 0:
        return None
    source_state = source_owned._ensure_device_state(preserve_indexed_view=True)
    repaired_state = repaired_owned._ensure_device_state(preserve_indexed_view=True)
    d_source_tags = cp.asarray(source_state.tags, dtype=cp.int8)
    d_repaired_tags = cp.asarray(repaired_state.tags, dtype=cp.int8)
    polygon_tags = cp.asarray(
        [
            FAMILY_TAGS[GeometryFamily.POLYGON],
            FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
        ],
        dtype=cp.int8,
    )
    row_count = int(repaired_owned.row_count)
    d_repaired_mask = cp.zeros(row_count, dtype=cp.bool_)
    d_repaired_mask[d_repaired_rows] = True
    d_polygon_repair_mask = (
        d_repaired_mask
        & cp.isin(d_source_tags, polygon_tags)
        & cp.isin(d_repaired_tags, polygon_tags)
    )

    source_repair = device_mask_owned_capacity(
        source_owned,
        d_polygon_repair_mask,
    )
    area_repair = device_mask_owned_capacity(
        repaired_owned,
        d_polygon_repair_mask,
    )
    source_boundary = boundary_owned(
        source_repair,
        dispatch_mode=ExecutionMode.GPU,
    )
    area_boundary = boundary_owned(
        area_repair,
        dispatch_mode=ExecutionMode.GPU,
    )
    remnants = _binary_constructive_gpu(
        "difference",
        source_boundary,
        area_boundary,
        dispatch_mode=ExecutionMode.GPU,
    )
    if remnants is None or int(remnants.row_count) != row_count:
        return None

    d_remnant_keep = _valid_nonempty_device_mask(remnants)
    d_area_keep = _valid_nonempty_device_mask(repaired_owned)
    if d_remnant_keep is None or d_area_keep is None:
        return None
    area_part = device_mask_owned_capacity(repaired_owned, d_area_keep)
    remnant_part = device_mask_owned_capacity(
        remnants,
        d_remnant_keep & d_polygon_repair_mask,
    )
    d_output_rows = cp.arange(row_count, dtype=cp.int64)
    return _geometry_composition_from_owned_parts_at_capacity(
        (
            (area_part, d_output_rows),
            (remnant_part, d_output_rows),
        ),
        row_count=row_count,
        crs=crs,
    )


__all__ = [
    "polygon_make_valid_linework_composition_device",
    "polygon_pair_boundary_remnant_mask_capacity_device",
    "polygon_pair_boundary_remnants_capacity_device",
]
