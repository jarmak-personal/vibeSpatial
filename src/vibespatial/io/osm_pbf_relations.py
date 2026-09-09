"""Bounded GPU relation expansion into native multipart and collection carriers."""
from __future__ import annotations

from dataclasses import dataclass

import cupy as cp

from vibespatial.io.osm_pbf_memory import relation_workspace
from vibespatial.io.osm_pbf_native import (
    _BatchPlan,
    _check_error,
    _host_metadata,
    _launch,
    _offsets,
    _prepare_device_batch,
    _resolve_source,
)
from vibespatial.io.osm_pbf_rings import counts_by_row, selected_indices


@dataclass
class _Catalogue:
    plans: list
    ranges: list
    way_indices: tuple
    relation_indices: tuple
    way_ranges: object
    identity: tuple


def catalogue(source, raw_budget):
    plans, ranges, way_indices, relation_indices, way_ranges = [], [], [], [], []
    for indices in source.batches(raw_budget=raw_budget):
        batch = _prepare_device_batch(source, indices)
        batch.select("lines")
        ndense = batch.dense.shape[0]
        base = len(ranges)
        if ndense:
            dranges = cp.empty((ndense, 4), dtype=cp.int64)
            _launch("pbf_decode_dense", ndense*3, batch.data, batch.meta, batch.dense,
                    None, None, None, None, None, dranges, batch.error, ndense, 0, warp=True)
            host = _host_metadata(dranges[:, :2].copy(), "node block ID ranges")
            ranges.extend(tuple(row) for row in host)
        wranges = cp.empty((len(indices), 2), dtype=cp.int64)
        _launch("pbf_way_ranges", len(indices), batch.meta, batch.ways, wranges, len(indices), warp=True)
        host_ways = _host_metadata(wranges, "way block ID ranges")
        for index, meta, bounds in zip(indices, batch.host_meta, host_ways, strict=True):
            if meta[6]:
                way_indices.append(index)
                way_ranges.append(tuple(bounds))
            if meta[13]:
                relation_indices.append(index)
        node_indices = tuple(i for i, m in zip(indices, batch.host_meta, strict=True) if m[7])
        plans.append(_BatchPlan(indices, node_indices, 0, 0, base, ndense, (), (), tuple(int(m[7]) for m in batch.host_meta if m[7])))
        _check_error(batch.error)
        del batch
    if any(a[1] >= b[0] for a, b in zip(ranges, ranges[1:])):
        raise NotImplementedError("Native PBF relation lookup requires ordered node blocks")
    return _Catalogue(plans, ranges, tuple(way_indices), tuple(relation_indices), cp.asarray(way_ranges, dtype=cp.int64).reshape(-1, 2), source.identity)


def _relation_batch(source, indices, layer):
    batch = _prepare_device_batch(source, indices)
    batch.index(relations=True)
    n = batch.relations.shape[0]
    batch.keep = cp.empty(n, dtype=cp.int32)
    _launch("pbf_parse_relations", n, batch.data, batch.meta, batch.strings, batch.relations,
            batch.keep, batch.error, n, layer)
    _check_error(batch.error)
    offsets = _offsets(batch.relations[:, 6] * batch.keep)
    nm = int(_host_metadata(offsets[-1:], "relation member count")[0])
    members = cp.zeros((nm, 10), dtype=cp.int64)
    _launch("pbf_decode_members", n, batch.data, batch.meta, batch.strings, batch.relations,
            batch.keep, offsets, members, batch.error, n, layer)
    _check_error(batch.error)
    ids = cp.where((members[:, 1] == 1) & (members[:, 2] != 3), members[:, 0], cp.iinfo(cp.int64).max)
    order = cp.argsort(ids).astype(cp.int64)
    return batch, members, offsets, ids[order], order


def _needed_ways(cat, ids):
    low = cp.searchsorted(ids, cat.way_ranges[:, 0], side="left")
    high = cp.searchsorted(ids, cat.way_ranges[:, 1], side="right")
    needed = _host_metadata(high > low, "referenced way block selection")
    return tuple(index for index, keep in zip(cat.way_indices, needed, strict=True) if keep)


def _match_members(source, cat, batch, members, ids, order, layer, raw_budget):
    needed = _needed_ways(cat, ids)
    from vibespatial.io.osm_pbf_native import _available_bytes
    for indices in source.batches(needed, raw_budget=raw_budget):
        work = 64*sum(source.blobs[i].raw_size for i in indices)
        with relation_workspace(work, _available_bytes(), kind="blob"):
            ways = _prepare_device_batch(source, indices)
            ways.select("lines")
            _launch("pbf_match_member_ways", ways.ways.shape[0], ways.ways, ids, order,
                    members, ways.error, ways.ways.shape[0], members.shape[0])
            _check_error(ways.error)
            del ways
    # Matching uses a biased atomic claim; normalize only after every Blob
    # has passed its duplicate-ID validation fence.
    members[:, 5] = cp.where(members[:, 1] == 1, cp.where(members[:, 5] >= 3, members[:, 5]-1, 0), members[:, 5])
    if layer == 2:
        missing = counts_by_row(members[:, 3], batch.relations.shape[0],
                               ((members[:, 1] == 1) & (members[:, 2] != 3) & (members[:, 5] == 0)).astype(cp.int64))
        members[:, 5] = cp.where(missing[members[:, 3]] > 0, 0, members[:, 5])
    return needed


def _expand(source, cat, batch, members, ids, order, layer, raw_budget):
    needed = _needed_ways(cat, ids)
    offsets = _offsets(members[:, 5])
    count = int(_host_metadata(offsets[-1:], "relation coordinate count")[0])
    x, y = cp.empty(count, cp.float64), cp.empty(count, cp.float64)
    _launch("pbf_member_nodes", members.shape[0], members, offsets, x, members.shape[0])
    from vibespatial.io.osm_pbf_native import _available_bytes
    for indices in source.batches(needed, raw_budget=raw_budget):
        work = 64*sum(source.blobs[i].raw_size for i in indices)
        with relation_workspace(work, _available_bytes(), kind="blob"):
            ways = _prepare_device_batch(source, indices)
            ways.select("lines")
            _launch("pbf_expand_member_refs", ways.ways.shape[0], ways.data, ways.ways, ids, order,
                    members, offsets, x, ways.ways.shape[0], members.shape[0], warp=True)
            _check_error(ways.error)
            del ways
    with relation_workspace(64*raw_budget, _available_bytes(), kind="blob"):
        missing = _resolve_source(source, cat.plans, cat.ranges, x, y, raw_budget, allow_missing=True)
    if missing:
        _launch("pbf_compact_member_coords", members.shape[0], x, y, offsets, members,
                None, None, None, members.shape[0], 0, warp=True)
        if layer == 2:
            missing_rel = counts_by_row(members[:, 3], batch.relations.shape[0],
                                       ((members[:, 1] == 1) & (members[:, 2] != 3) & (members[:, 5] == 0)).astype(cp.int64))
            members[:, 5] = cp.where(missing_rel[members[:, 3]] > 0, 0, members[:, 5])
        new_offsets = _offsets(members[:, 5])
        count = int(_host_metadata(new_offsets[-1:], "resolved member coordinate count")[0])
        ox, oy = cp.empty(count, cp.float64), cp.empty(count, cp.float64)
        _launch("pbf_compact_member_coords", members.shape[0], x, y, offsets, members,
                new_offsets, ox, oy, members.shape[0], 1, warp=True)
        x, y, offsets = ox, oy, new_offsets
    return x, y, offsets, needed


def _owned(family, rows, x, y, offsets, *, parts=None, rings=None):
    from vibespatial.io.pylibcudf import _build_device_single_family_owned

    counts = [rows, x.size, offsets.size-1]
    counts.extend(value.size-1 for value in (parts, rings) if value is not None)
    if max(counts) > 2147483647:
        raise OverflowError("OSM relation geometry exceeds the owned int32 offset capacity")
    return _build_device_single_family_owned(
        family=family, validity_device=cp.ones(rows, dtype=cp.bool_), x_device=x, y_device=y,
        geometry_offsets_device=offsets.astype(cp.int32),
        part_offsets_device=None if parts is None else parts.astype(cp.int32),
        ring_offsets_device=None if rings is None else rings.astype(cp.int32),
        empty_mask_device=cp.zeros(rows, dtype=cp.bool_), all_valid=True,
        detail="OSM PBF device relation/member assembly",
    )


def _multipart_geometry(batch, members, offsets, x, y, layer):
    from vibespatial.api._native_result_core import (
        GeometryNativeResult,
        NativeGeometryComposition,
        NativeGeometryCompositionPart,
    )
    from vibespatial.geometry.buffers import GeometryFamily

    n = batch.relations.shape[0]
    selected = selected_indices(counts_by_row(members[:, 3], n, (members[:, 5] > 0).astype(cp.int64)) > 0)
    row_map = cp.full(n, -1, dtype=cp.int64)
    row_map[selected] = cp.arange(selected.size, dtype=cp.int64)
    active = selected_indices(members[:, 5] > 0)
    if layer == 3:
        go = _offsets(counts_by_row(members[active, 3], n)[selected])
        po = _offsets(members[active, 5])
        owned = _owned(GeometryFamily.MULTILINESTRING, selected.size, x, y, go, parts=po)
        return GeometryNativeResult(owned=owned, crs="EPSG:4326"), selected
    # A vector member-order carrier avoids a Python part per member position.
    pieces = []
    for family, kind in ((GeometryFamily.POINT, 0), (GeometryFamily.LINESTRING, 1), (GeometryFamily.POLYGON, 2)):
        member_kind = cp.where(members[:, 1] == 0, 0, cp.where(members[:, 8] != 0, 2, 1))
        chosen = selected_indices((members[:, 5] > 0) & (member_kind == kind))
        if not chosen.size:
            continue
        ro = _offsets(members[chosen, 5])
        count = int(_host_metadata(ro[-1:], "collection family coordinate count")[0])
        ox, oy = cp.empty(count, cp.float64), cp.empty(count, cp.float64)
        _launch("pbf_reorder_rings", chosen.size, x, y, offsets, chosen, ro, ox, oy, chosen.size, warp=True)
        go = cp.arange(chosen.size+1, dtype=cp.int64) if kind == 2 else ro
        owned = _owned(family, chosen.size, ox, oy, go, rings=ro if kind == 2 else None)
        pieces.append(NativeGeometryCompositionPart(
            geometry=GeometryNativeResult(owned=owned, crs="EPSG:4326"),
            output_rows=row_map[members[chosen, 3]], collection_positions=members[chosen, 4],
        ))
    geometry = GeometryNativeResult.from_composition(
        NativeGeometryComposition(parts=tuple(pieces), row_count=selected.size, crs="EPSG:4326"), crs="EPSG:4326",
    )
    return geometry, selected


def _relation_result(source, cat, batch, members, relation_offsets, ids, order, kind, *, raw_budget, geometry_only, tags):
    import pyarrow as pa

    from vibespatial.api._native_result_core import NativeAttributeTable, NativeTabularResult
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.io.osm_pbf_attributes import AttributeOutput, measure_attributes
    from vibespatial.io.osm_pbf_rings import assemble_rings, nest_rings

    x, y, offsets, needed = _expand(source, cat, batch, members, ids, order, kind, raw_budget)
    suppressed = cp.empty(0, dtype=cp.int64)
    if kind == 2:
        from vibespatial.api._native_result_core import GeometryNativeResult

        closed = cp.zeros(members.shape[0], dtype=cp.bool_)
        active = selected_indices(members[:, 5] > 0)
        closed[active] = (x[offsets[active]] == x[offsets[active+1]-1]) & (y[offsets[active]] == y[offsets[active+1]-1])
        suppress_rows = selected_indices(closed & (members[:, 1] == 1) & (members[:, 2] == 1))
        suppressed = members[suppress_rows, 0]
        rx, ry, ro, rr = assemble_rings(members, offsets, relation_offsets, x, y, batch.relations.shape[0])
        del x, y
        ox, oy, ring_offsets, part_offsets, go, selected = nest_rings(rx, ry, ro, rr, batch.relations.shape[0])
        del rx, ry
        owned = _owned(GeometryFamily.MULTIPOLYGON, selected.size, ox, oy, go, parts=part_offsets, rings=ring_offsets)
        geometry = GeometryNativeResult(owned=owned, crs="EPSG:4326")
    else:
        geometry, selected = _multipart_geometry(batch, members, offsets, x, y, kind)
    batch.keep.fill(0)
    batch.keep[selected] = 1
    batch.positions = _offsets(batch.keep)
    batch.rows = selected.size
    if geometry_only or not batch.rows:
        attrs = NativeAttributeTable(arrow_table=pa.table({"_": pa.nulls(batch.rows)}).select([]))
    else:
        schema_kind = 2 if kind == 2 else 3
        attr_offsets, sizes, nulls = measure_attributes(batch, False, id_only=tags is False, kind=schema_kind, relation=True)
        builder = AttributeOutput(False, batch.rows, sizes, nulls, id_only=tags is False, kind=schema_kind, relation=True)
        builder.emit(batch, attr_offsets, sizes, 0)
        attrs = builder.finish()
        if kind == 2 and tags is not False:
            attrs = _inherit_polygon_tags(source, cat, batch, members, selected, attrs, raw_budget)
    _check_error(batch.error)
    return NativeTabularResult(attributes=attrs, geometry=geometry, geometry_name="geometry",
                               column_order=(*attrs.columns, "geometry")), suppressed


def relation_chunks(source, cat, layer, *, raw_budget, geometry_only, tags):
    from dataclasses import replace

    from vibespatial.io.osm_pbf_native import _available_bytes

    kind = 2 if layer == "multipolygons" else 3 if layer == "multilinestrings" else 4
    for indices in source.batches(cat.relation_indices, raw_budget=raw_budget):
        work = 64*sum(source.blobs[i].raw_size for i in indices)
        with relation_workspace(work, _available_bytes(), kind="blob"):
            base, all_members, member_offsets, all_ids, all_order = _relation_batch(source, indices, kind)
        _match_members(source, cat, base, all_members, all_ids, all_order, kind, raw_budget)
        costs = counts_by_row(all_members[:, 3], base.relations.shape[0], 48*all_members[:, 5]+512)
        # This packet describes output/work allocation sizes, never coordinates,
        # tags or member IDs. Window decisions are logarithmic in relation rows.
        prefixes = _host_metadata(cp.stack((_offsets(costs), member_offsets)), "relation expanded work prefixes")
        start = 0
        while start < base.relations.shape[0]:
            budget = min(256 << 20, max(_available_bytes()-(16 << 20), 0)//2)
            stop = max(start+1, int(prefixes[0].searchsorted(prefixes[0, start]+budget, side="right"))-1)
            stop = min(stop, base.relations.shape[0])
            work_bytes = int(prefixes[0, stop]-prefixes[0, start])
            begin, end = int(prefixes[1, start]), int(prefixes[1, stop])
            with relation_workspace(work_bytes, budget):
                members = all_members[begin:end].copy()
                members[:, 3] -= start
                batch = replace(base, relations=base.relations[start:stop], keep=base.keep[start:stop].copy(), ids=None)
                offsets = member_offsets[start:stop+1]-begin
                ids = cp.where((members[:, 1] == 1) & (members[:, 2] != 3), members[:, 0], cp.iinfo(cp.int64).max)
                order = cp.argsort(ids).astype(cp.int64)
                result = _relation_result(source, cat, batch, members, offsets, ids[order], order, kind,
                                          raw_budget=raw_budget, geometry_only=geometry_only, tags=tags)
                del batch, members, offsets, ids, order
            yield result
            start = stop
        del base, all_members, all_ids, all_order, member_offsets


def read_relation_layer(path, *, layer, geometry_only=False, tags="ways", raw_budget=256 << 20, _catalogue=None):
    from dataclasses import replace

    from vibespatial.api._native_results import NativeReadProvenance
    from vibespatial.io.osm_pbf_inflate import PbfSource
    from vibespatial.io.osm_pbf_memory import concatenate_results
    from vibespatial.io.osm_pbf_native import _available_bytes, read_osm_pbf_native

    parts, excluded = [], []
    with PbfSource(path) as source:
        available = max(_available_bytes()-(16 << 20), 0)
        largest = max((b.raw_size for b in source.blobs), default=0)
        raw_budget = max(largest, min(raw_budget, available//64))
        with relation_workspace(64*raw_budget, available, kind="blob"):
            cat = catalogue(source, raw_budget) if _catalogue is None else _catalogue
        if cat.identity != source.identity:
            raise ValueError("OSM PBF source changed during read")
        for result, suppressed in relation_chunks(source, cat, layer, raw_budget=raw_budget, geometry_only=geometry_only, tags=tags):
            if result.geometry.row_count:
                parts.append(result)
            if suppressed.size:
                excluded.append(suppressed)
        source.validate_identity()
    if layer == "multipolygons":
        suppression = cp.sort(cp.concatenate(excluded)) if excluded else cp.empty(0, cp.int64)
        standalone = read_osm_pbf_native(path, layer="polygonways", geometry_only=geometry_only, tags=tags,
                                        raw_budget=raw_budget, _excluded_way_ids=suppression)
        parts.append(standalone)
    if not parts:
        # Keep the declared relation schema for empty admitted layers.
        from vibespatial.api._native_result_core import GeometryNativeResult, NativeTabularResult
        from vibespatial.geometry.buffers import GeometryFamily
        from vibespatial.io.osm_pbf_attributes import AttributeOutput
        kind = 2 if layer == "multipolygons" else 3
        ncol = (2 if kind == 2 else 1) if tags is False else 25 if kind == 2 else 4
        if geometry_only:
            import pyarrow as pa

            from vibespatial.api._native_result_core import NativeAttributeTable
            attrs = NativeAttributeTable(arrow_table=pa.table({}))
        else:
            attrs = AttributeOutput(False, 0, [0]*ncol, [0]*ncol, id_only=tags is False, kind=kind, relation=True).finish()
        geometry = GeometryNativeResult(owned=_owned(GeometryFamily.MULTILINESTRING, 0, cp.empty(0, cp.float64), cp.empty(0, cp.float64), cp.zeros(1, cp.int64), parts=cp.zeros(1, cp.int64)), crs="EPSG:4326")
        parts.append(NativeTabularResult(attributes=attrs, geometry=geometry, geometry_name="geometry", column_order=(*attrs.columns, "geometry")))
    result = concatenate_results(parts)
    cp.cuda.get_current_stream().synchronize()
    return replace(result, provenance=NativeReadProvenance(surface="read_file", format_name="OSM-PBF", source=str(path), backend="nvcomp-nvrtc"))


def _inherit_polygon_tags(source, cat, batch, members, selected, attrs, raw_budget):
    """GDAL's type-only polygon relation inherits the first tagged outer way."""
    import pylibcudf as plc

    from vibespatial.api._native_result_core import NativeAttributeTable
    from vibespatial.cuda._runtime import pylibcudf_column_from_device, pylibcudf_current_stream
    from vibespatial.io.osm_pbf_attributes import AttributeOutput, measure_attributes
    from vibespatial.io.osm_pbf_native import _available_bytes

    n = batch.relations.shape[0]
    sentinel = cp.iinfo(cp.int64).max
    first = cp.full(n, sentinel, dtype=cp.uint64)
    candidates = (members[:, 1] == 1) & (members[:, 2] == 1) & (members[:, 5] > 0) & (members[:, 8] != 0) & (members[:, 9] != 0)
    cp.minimum.at(first, members[:, 3], cp.where(candidates, cp.arange(members.shape[0], dtype=cp.int64), sentinel).astype(cp.uint64))
    first = first.view(cp.int64)[selected]
    target_rows = selected_indices((first != sentinel) & (batch.relations[selected, 17] != 0))
    if not target_rows.size:
        return attrs
    wanted = members[first[target_rows], 0]
    needed = _needed_ways(cat, cp.sort(wanted))
    replace_columns = [i for i, name in enumerate(attrs.columns) if name not in ("osm_id", "osm_way_id", "type")]
    current = attrs.device_table
    for indices in source.batches(needed, raw_budget=raw_budget):
        work = 64*sum(source.blobs[i].raw_size for i in indices)
        with relation_workspace(work, _available_bytes(), kind="blob"):
            ways = _prepare_device_batch(source, indices)
            ways.select("lines")
            way_order = cp.argsort(ways.ways[:, 3]).astype(cp.int64)
            way_ids = ways.ways[way_order, 3]
            locations = cp.searchsorted(way_ids, wanted)
            safe = cp.minimum(locations, way_ids.size-1)
            matches = selected_indices((locations < way_ids.size) & (way_ids[safe] == wanted))
            if not matches.size:
                continue
            input_rows = way_order[locations[matches]]
            ways.keep.fill(0)
            ways.keep[input_rows] = 1
            ways.positions = _offsets(ways.keep)
            ways.rows = int(_host_metadata(ways.positions[-1:], "inherited way row count")[0])
            offsets, sizes, nulls = measure_attributes(ways, False, kind=2)
            builder = AttributeOutput(False, ways.rows, sizes, nulls, kind=2)
            builder.emit(ways, offsets, sizes, 0)
            inherited = builder.finish()
            _check_error(ways.error)
            gather_rows = ways.positions[input_rows].astype(cp.int32)
            # A tagged way can feed many relation rows. Admit the expanded UTF-8
            # bytes, including both the gather and full destination scatter copy.
            expanded = cp.stack([(builder.offsets[i][gather_rows+1].astype(cp.int64)
                                  - builder.offsets[i][gather_rows]).sum()
                                 for i in replace_columns])
            char_sizes = _host_metadata(expanded, "inherited attribute allocation bytes")
            target = plc.Table([current.columns()[i] for i in replace_columns])
            current_chars = [c.data().size if c.data() is not None else 0 for c in target.columns()]
            if any(int(a)+b > cp.iinfo(cp.int32).max for a, b in zip(char_sizes, current_chars, strict=True)):
                raise OverflowError("Inherited PBF string column exceeds libcudf int32 capacity")
            estimate = (2*sum(int(v) for v in char_sizes)+sum(current_chars)
                        + 8*(2*matches.size+selected.size)*len(replace_columns))
            budget = _available_bytes()
            resource = None
            if estimate > budget:
                import rmm

                from vibespatial.runtime import ExecutionMode
                from vibespatial.runtime.dispatch import record_dispatch_event
                resource = rmm.mr.ManagedMemoryResource()
                record_dispatch_event(
                    surface="vibespatial.io.osm_pbf", operation="inherited_attributes",
                    implementation="cuda_managed_inherited_attributes", selected=ExecutionMode.GPU,
                    reason="Expanded inherited tags and their scatter copy exceed the device workspace",
                    detail=f"estimated_transient_bytes={estimate}, available_device_bytes={budget}",
                )
            gather_map = pylibcudf_column_from_device(gather_rows)
            source_table = plc.Table([inherited.device_table.columns()[i] for i in replace_columns])
            gathered = plc.copying.gather(source_table, gather_map, plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                                         stream=pylibcudf_current_stream(source_table, gather_map), mr=resource)
            scatter_map = pylibcudf_column_from_device(target_rows[matches].astype(cp.int32))
            replaced = plc.copying.scatter(gathered, scatter_map, target,
                                           stream=pylibcudf_current_stream(gathered, scatter_map, target), mr=resource)
            columns = current.columns()
            for index, column in zip(replace_columns, replaced.columns(), strict=True):
                columns[index] = column
            current = plc.Table(columns)
            del ways, builder, inherited, gathered, replaced
    return NativeAttributeTable(device_table=current, column_override=attrs.column_override, schema_override=attrs.schema_override)


def _element_column(result, layer):
    from dataclasses import replace

    import pyarrow as pa
    import pylibcudf as plc

    from vibespatial.api._native_result_core import NativeAttributeTable
    from vibespatial.cuda._runtime import pylibcudf_current_stream

    table = result.attributes.device_table
    if table is None:
        if result.geometry.row_count:
            raise TypeError("Native OSM element annotation requires device attributes")
        table = plc.Table.from_arrow(result.attributes.to_arrow(index=False))
    if layer == "multipolygons":
        way_id = table.columns()[list(result.attributes.columns).index("osm_way_id")]
        mask = plc.unary.is_valid(way_id, stream=pylibcudf_current_stream(way_id))
        element = plc.copying.copy_if_else(
            plc.Scalar.from_arrow(pa.scalar("way")), plc.Scalar.from_arrow(pa.scalar("relation")),
            mask, stream=pylibcudf_current_stream(mask),
        )
    else:
        value = "node" if layer == "points" else "way" if layer == "lines" else "relation"
        element = plc.Column.from_scalar(plc.Scalar.from_arrow(pa.scalar(value)), result.geometry.row_count,
                                         stream=pylibcudf_current_stream(table))
    names = (*result.attributes.columns, "osm_element")
    schema = result.attributes.schema_override.append(pa.field("osm_element", pa.string()))
    attrs = NativeAttributeTable(device_table=plc.Table([*table.columns(), element]), column_override=names, schema_override=schema)
    return replace(result, attributes=attrs, column_order=(*names, "geometry"))


def read_all_layers(path, *, geometry_only, tags, raw_budget):
    from vibespatial.api._native_results import NativeReadProvenance
    from vibespatial.io.osm_pbf_inflate import PbfSource
    from vibespatial.io.osm_pbf_memory import concatenate_results
    from vibespatial.io.osm_pbf_native import _available_bytes, read_osm_pbf_native

    results = []
    with PbfSource(path) as source:
        available = max(_available_bytes()-(16 << 20), 0)
        largest = max((b.raw_size for b in source.blobs), default=0)
        raw_budget = max(largest, min(raw_budget, available//64))
        with relation_workspace(64*raw_budget, available, kind="blob"):
            cat = catalogue(source, raw_budget)
        for layer in ("points", "lines", "multilinestrings", "multipolygons", "other_relations"):
            if layer in ("points", "lines"):
                result = read_osm_pbf_native(path, layer=layer, geometry_only=geometry_only, tags=tags, raw_budget=raw_budget)
            else:
                result = read_relation_layer(path, layer=layer, geometry_only=geometry_only, tags=tags, raw_budget=raw_budget, _catalogue=cat)
            results.append(result if geometry_only else _element_column(result, layer))
        source.validate_identity()
    from dataclasses import replace
    result = concatenate_results(results)
    cp.cuda.get_current_stream().synchronize()
    return replace(result,
                   provenance=NativeReadProvenance(surface="read_file", format_name="OSM-PBF", source=str(path), backend="nvcomp-nvrtc"))
