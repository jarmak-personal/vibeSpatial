"""Device endpoint graph to nested polygon offsets, without host geometry."""
from __future__ import annotations

import cupy as cp

from vibespatial.io.osm_pbf_native import _host_metadata, _launch, _offsets


def selected_indices(mask):
    positions = _offsets(mask)
    count = int(_host_metadata(positions[-1:], "compact row count")[0])
    out = cp.empty(count, dtype=cp.int64)
    _launch("pbf_select_indices", mask.size, mask.astype(cp.int32), positions, out, mask.size)
    return out


def counts_by_row(rows, size, values=1):
    counts = cp.zeros(size, dtype=cp.uint64)
    cp.add.at(counts, rows, cp.asarray(values, dtype=cp.uint64))
    return counts.view(cp.int64)


def assemble_rings(members, member_offsets, relation_offsets, x, y, nrelations):
    """Return ring coordinates, offsets and source-relation rows in source order."""
    nm = members.shape[0]
    if not nm or not x.size:
        return cp.empty(0, cp.float64), cp.empty(0, cp.float64), cp.zeros(1, cp.int64), cp.empty(0, cp.int64)
    endpoints = cp.stack((member_offsets[:-1], cp.maximum(member_offsets[1:]-1, 0)), axis=1).reshape(-1)
    endpoints = cp.minimum(endpoints, x.size-1)
    starts, ends = endpoints[::2], endpoints[1::2]
    eligible = (members[:,1] == 1) & (members[:,2] != 3) & (members[:,5] >= 2)
    eligible &= (x[starts] != x[ends]) | (y[starts] != y[ends])
    states = selected_indices(cp.repeat(eligible, 2))
    # Integer 100-nanodegree lattice inherited from the OSM driver's node index.
    ex = cp.rint(x[endpoints[states]]*1e7).astype(cp.int64)
    ey = cp.rint(y[endpoints[states]]*1e7).astype(cp.int64)
    rows = members[states//2, 3]
    permutation = cp.lexsort(cp.stack((states, ex, ey, rows))).astype(cp.int64)
    order = states[permutation]
    keys = cp.stack((rows[permutation], ex[permutation], ey[permutation]), axis=1)
    starts_group = cp.empty(states.size, dtype=cp.bool_)
    if states.size:
        starts_group[0] = True
        starts_group[1:] = cp.any(keys[1:] != keys[:-1], axis=1)
    group_starts = selected_indices(starts_group)
    groups = cp.empty(group_starts.size+1, dtype=cp.int64)
    groups[:-1], groups[-1] = group_starts, states.size
    inverse = cp.full(2*nm, -1, dtype=cp.int64)
    inverse[order] = _offsets(starts_group)[1:]-1
    next_state = cp.empty(2*nm, dtype=cp.int64)
    branches = cp.zeros(nrelations, dtype=cp.int32)
    _launch("pbf_endpoint_successors", 2*nm, groups, order, inverse, members, x, y,
            member_offsets, next_state, branches, nm)
    visited = cp.empty(nm, dtype=cp.int32)
    cursors = group_starts.copy()
    _launch("pbf_branch_rings", nrelations, groups, order, inverse, members,
            relation_offsets, branches, x, y, member_offsets, next_state, visited, cursors, nrelations)
    del keys, order, inverse, rows, ex, ey, endpoints, branches, visited, groups, cursors
    del starts, ends, eligible, states, permutation, starts_group, group_starts
    labels = cp.arange(2*nm, dtype=cp.int64)
    jumps = next_state.copy()
    for _ in range((2*nm).bit_length()):
        safe = cp.maximum(jumps, 0)
        labels = cp.where(jumps < 0, -1, cp.minimum(labels, labels[safe]))
        jumps = cp.where(jumps < 0, -1, jumps[safe])
    states = 2*cp.arange(nm, dtype=cp.int64) + (labels[::2] & 1)
    member_labels = labels[states]
    valid = member_labels >= 0
    heads = selected_indices(valid & (member_labels == states))
    if not heads.size:
        return cp.empty(0, cp.float64), cp.empty(0, cp.float64), cp.zeros(1, cp.int64), cp.empty(0, cp.int64)
    ring_ids = cp.searchsorted(heads, cp.maximum(member_labels, 0)//2).astype(cp.int64)
    contributions = cp.where(valid, members[:, 5]-1, 0)
    selected = selected_indices(valid)
    ring_sizes = counts_by_row(ring_ids[selected], heads.size, contributions[selected])+1
    # GDAL preserves directly closed members, but drops a reconstructed
    # cycle unless it has at least four coordinates including closure.
    closed_head = (x[member_offsets[heads]] == x[member_offsets[heads+1]-1]) & (y[member_offsets[heads]] == y[member_offsets[heads+1]-1])
    admitted = (ring_sizes >= 4) | closed_head
    valid[selected] &= admitted[ring_ids[selected]]
    heads = heads[selected_indices(admitted)]
    ring_sizes = ring_sizes[selected_indices(admitted)]
    if not heads.size:
        return cp.empty(0, cp.float64), cp.empty(0, cp.float64), cp.zeros(1, cp.int64), cp.empty(0, cp.int64)
    ring_ids = cp.searchsorted(heads, cp.maximum(member_labels, 0)//2).astype(cp.int64)
    selected = selected_indices(valid)
    contributions = cp.where(valid, members[:, 5]-1, 0)
    # List ranking on the oriented cycles, cut at each canonical head.
    predecessor = cp.full(nm, -1, dtype=cp.int64)
    successor = next_state[states[selected]]//2
    predecessor[successor] = selected
    predecessor[heads] = -1
    prefixes = contributions.copy()
    jumps = predecessor
    for _ in range(nm.bit_length()):
        safe = cp.maximum(jumps, 0)
        prefixes = prefixes + cp.where(jumps < 0, 0, prefixes[safe])
        jumps = cp.where(jumps < 0, -1, jumps[safe])
    offsets = _offsets(ring_sizes)
    ncoords = int(_host_metadata(offsets[-1:], "assembled ring coordinate count")[0])
    ox, oy = cp.empty(ncoords, cp.float64), cp.empty(ncoords, cp.float64)
    _launch("pbf_emit_rings", nm, x, y, member_offsets, cp.where(valid, states, -1),
            ring_ids, prefixes, offsets, ox, oy, nm, warp=True)
    return ox, oy, offsets, members[heads, 3]


def nest_rings(x, y, offsets, rows, nrelations, *, precision_plan=None):
    """Choose enclosing rings with OGR-compatible binary64 boundary tests."""
    from vibespatial.runtime._runtime import ExecutionMode, RuntimeSelection
    from vibespatial.runtime.dispatch import record_dispatch_event
    from vibespatial.runtime.precision import KernelClass, PrecisionMode, select_precision_plan

    if precision_plan is None:
        precision_plan = select_precision_plan(
            runtime_selection=RuntimeSelection(ExecutionMode.GPU, ExecutionMode.GPU, "OSM polygon construction"),
            kernel_class=KernelClass.CONSTRUCTIVE, requested=PrecisionMode.FP64,
        )
    if precision_plan.compute_precision is not PrecisionMode.FP64:
        raise ValueError("OSM topology construction requires binary64 compatibility semantics")
    record_dispatch_event(
        surface="vibespatial.io.osm_pbf", operation="ring_nesting", implementation="nvrtc_fp64_ring_nesting",
        selected=ExecutionMode.GPU, reason=precision_plan.reason,
        detail=f"shape=relation_consume; rings={rows.size}; coordinates={x.size}; compute={precision_plan.compute_precision.value}",
    )
    nr = rows.size
    if not nr:
        return x, y, offsets, cp.zeros(1, cp.int64), cp.zeros(1, cp.int64), cp.empty(0, cp.int64)
    bounds = cp.empty((nr, 6), dtype=cp.int64)
    _launch("pbf_ring_bounds", nr, x, y, offsets, bounds, nr, warp=True)
    parent = cp.empty(nr, dtype=cp.int64)
    row_offsets = _offsets(counts_by_row(rows, nrelations))
    invalid_rows = counts_by_row(rows, nrelations, ((offsets[1:]-offsets[:-1]) < 4).astype(cp.int64))
    _launch("pbf_ring_parent", nr, x, y, offsets, rows, row_offsets, bounds, invalid_rows, parent, nr, warp=True)
    depth = (parent >= 0).astype(cp.int64)
    jumps = parent.copy()
    for _ in range(nr.bit_length()):
        safe = cp.maximum(jumps, 0)
        depth += cp.where(jumps < 0, 0, depth[safe])
        jumps = cp.where(jumps < 0, -1, jumps[safe])
    holes = (depth & 1) != 0
    owners = cp.where(holes, parent, cp.arange(nr, dtype=cp.int64))
    order = cp.lexsort(cp.stack((cp.arange(nr), holes, owners, rows))).astype(cp.int64)
    outer = selected_indices(~holes)
    polygon_rows = rows[outer]
    selected_rows = selected_indices(counts_by_row(polygon_rows, nrelations) > 0)
    geometry_offsets = _offsets(counts_by_row(polygon_rows, nrelations)[selected_rows])
    polygon_ids = cp.searchsorted(outer, owners)
    part_offsets = _offsets(counts_by_row(polygon_ids, outer.size))
    new_offsets = _offsets((offsets[1:]-offsets[:-1])[order])
    ox, oy = cp.empty_like(x), cp.empty_like(y)
    _launch("pbf_reorder_rings", nr, x, y, offsets, order, new_offsets, ox, oy, nr, warp=True)
    return ox, oy, new_offsets, part_offsets, geometry_offsets, selected_rows
