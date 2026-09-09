"""Counted native string-column assembly for standard OSM point/line layers."""
from __future__ import annotations

import ctypes

import cupy as cp

_POINT_COLUMNS = ("osm_id", "name", "barrier", "highway", "ref", "address", "is_in", "place", "man_made", "other_tags")
_LINE_COLUMNS = ("osm_id", "name", "highway", "waterway", "aerialway", "barrier", "man_made", "railway", "other_tags")

_POLYGON_COLUMNS = ("osm_id", "osm_way_id", "name", "type", "aeroway", "amenity", "admin_level", "barrier", "boundary", "building", "craft", "geological", "historic", "land_area", "landuse", "leisure", "man_made", "military", "natural", "office", "place", "shop", "sport", "tourism", "other_tags")
_RELATION_COLUMNS = ("osm_id", "name", "type", "other_tags")
_SCHEMAS = (_POINT_COLUMNS, _LINE_COLUMNS, _POLYGON_COLUMNS, _RELATION_COLUMNS)


def measure_attributes(batch, point, *, id_only=False, kind=None, relation=False):
    from vibespatial.io.osm_pbf_native import _host_metadata, _launch

    kind = (0 if point else 1) if kind is None else kind
    ncols = (2 if kind == 2 else 1) if id_only else len(_SCHEMAS[kind])
    lengths = cp.empty((ncols, batch.rows), dtype=cp.int64)
    if point:
        batch.ids = cp.empty(batch.rows, dtype=cp.int64)
        _launch("pbf_decode_dense", batch.dense.shape[0]*3, batch.data, batch.meta,
                batch.dense, batch.ids, None, None, batch.keep, batch.positions,
                None, batch.error, batch.dense.shape[0], 1, warp=True)
    _launch("pbf_attributes", batch.keep.size, batch.data, batch.meta, batch.strings,
            batch.relations if relation else batch.ways, batch.spans, batch.keep, batch.positions, batch.ids,
            lengths, None, None, None, batch.error, batch.keep.size, batch.rows,
            kind, 20 if relation else 16, int(kind == 2 and not relation), ncols, 0, ctypes.c_longlong(0))
    offsets = cp.empty((ncols, batch.rows+1), dtype=cp.int64)
    offsets[:, 0] = 0
    cp.cumsum(cp.maximum(lengths, 0), axis=1, out=offsets[:, 1:])
    packet = cp.concatenate((offsets[:, -1], cp.sum(lengths < 0, axis=1), batch.error.astype(cp.int64)))
    values = _host_metadata(packet, "attribute byte counts, null counts and validation").tolist()
    if values[-1]:
        from vibespatial.io.osm_pbf_native import _check_error
        _check_error(batch.error)
    return offsets, values[:ncols], values[ncols:2*ncols]


class AttributeOutput:
    def __init__(self, point, rows, totals, nulls, *, id_only=False, kind=None, relation=False):
        self.kind = (0 if point else 1) if kind is None else kind
        self.relation = relation
        self.columns = _SCHEMAS[self.kind]
        if id_only:
            self.columns = self.columns[:2 if self.kind == 2 else 1]
        self.id_only = id_only
        self.rows = rows
        self.point = point
        self.nulls = nulls
        self.totals = totals
        if any(total > 2147483647 for total in totals):
            raise OverflowError("OSM string column exceeds libcudf int32 capacity")
        self.chars = [cp.empty(total, dtype=cp.uint8) for total in totals]
        self.offsets = [cp.empty(rows+1, dtype=cp.int32) for _ in totals]
        self.masks = [cp.zeros((rows+31)//32, dtype=cp.uint32) for _ in totals]
        self.z_order = cp.empty(rows, dtype=cp.int32) if self.kind == 1 and not id_only else None
        self.bases = [0] * len(totals)

    def emit(self, batch, offsets, sizes, row_base):
        from vibespatial.io.osm_pbf_native import _launch

        pointers = cp.asarray([
            (chars.data.ptr, off.data.ptr, mask.data.ptr, base)
            for chars, off, mask, base in zip(self.chars, self.offsets, self.masks, self.bases, strict=True)
        ], dtype=cp.uint64)
        _launch("pbf_attributes", batch.keep.size, batch.data, batch.meta, batch.strings,
                batch.relations if self.relation else batch.ways, batch.spans, batch.keep, batch.positions, batch.ids,
                None, offsets, pointers, self.z_order, batch.error, batch.keep.size,
                batch.rows, self.kind, 20 if self.relation else 16, int(self.kind == 2 and not self.relation),
                len(self.columns), 1, ctypes.c_longlong(row_base))
        self.bases = [a+b for a, b in zip(self.bases, sizes, strict=True)]

    def finish(self):
        import pyarrow as pa
        import pylibcudf as plc

        from vibespatial.api._native_result_core import NativeAttributeTable
        from vibespatial.cuda._runtime import pylibcudf_column_from_device

        columns = []
        names = list(self.columns)
        for chars, offsets, mask, total, nulls in zip(
            self.chars, self.offsets, self.masks, self.totals, self.nulls, strict=True,
        ):
            offsets[-1] = total
            columns.append(plc.Column(
                plc.types.DataType(plc.types.TypeId.STRING), self.rows,
                plc.gpumemoryview(chars), plc.gpumemoryview(mask), nulls, 0,
                [pylibcudf_column_from_device(offsets)],
            ))
        schema = [pa.field(name, pa.string()) for name in names]
        if self.z_order is not None:
            names.insert(-1, "z_order")
            columns.insert(-1, pylibcudf_column_from_device(self.z_order))
            schema.insert(-1, pa.field("z_order", pa.int32()))
        return NativeAttributeTable(device_table=plc.Table(columns), column_override=tuple(names), schema_override=pa.schema(schema))
