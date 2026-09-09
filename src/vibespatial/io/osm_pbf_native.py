"""Byte-bounded GPU PBF ingress into the private native tabular boundary.

The physical shape is block/byte scan, segmented delta decode and dynamic
output assembly. Counts are the only device data exported before the terminal
boundary. The two-pass reader allocates the final result once and replays
bounded batches, rather than concatenating a second copy of the result.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import cupy as cp
import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_I64,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.cccl_primitives import exclusive_sum
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.io.osm_pbf_native_kernels import _PBF_NAMES, _PBF_SOURCE
from vibespatial.runtime.hotpath_trace import hotpath_stage

request_nvrtc_warmup([("osm-pbf-native-fp64", _PBF_SOURCE, _PBF_NAMES)])
request_warmup(["exclusive_scan_i64"])

def _launch(name, n, *args, warp=False):
    if not n:
        return
    # A protobuf record chain is serial. Isolate its control lane so a batch
    # occupies many SMs instead of packing all Blob parsers into two CTAs.
    lanes = 32 if warp or name in ("pbf_count_blocks", "pbf_index_blocks") else 1
    runtime = get_cuda_runtime()
    kernels = compile_kernel_group("osm-pbf-native-fp64", _PBF_SOURCE, _PBF_NAMES)
    values, types = [], []
    for value in args:
        if isinstance(value, int):
            values.append(ctypes.c_int(value))
            types.append(KERNEL_PARAM_I32)
        elif isinstance(value, ctypes.c_longlong):
            values.append(value)
            types.append(KERNEL_PARAM_I64)
        else:
            values.append(runtime.pointer(value))
            types.append(KERNEL_PARAM_PTR)
    grid, block = runtime.launch_config(kernels[name], n * lanes)
    runtime.launch(kernels[name], grid=grid, block=block, params=(tuple(values), tuple(types)))


def _host_metadata(data, reason):
    return get_cuda_runtime().copy_device_to_host(data, reason=f"OSM PBF {reason} metadata")


def _check_error(error, *, allow_missing=False):
    code = int(_host_metadata(error, "structural validation")[0])
    if code == 2:
        raise NotImplementedError("PBF native decode requires ordered DenseNodes and unsplit packed fields")
    if code == 4:
        raise NotImplementedError("Native PBF requires unique referenced way IDs")
    if code == 3 and allow_missing:
        return True
    if code == 3:
        raise NotImplementedError("Native PBF line decode encountered missing node references")
    if code:
        raise ValueError("Malformed OSM PBF protobuf payload")


def _available_bytes():
    runtime = get_cuda_runtime()
    # A caller or another process can acquire VRAM after the query envelope
    # was captured. The async pool quota alone does not bound physical growth.
    free, _ = cp.cuda.runtime.memGetInfo()
    return min(runtime.query_memory_remaining_bytes(), runtime.pool_upstream_growth_bytes(), int(free))


def _offsets(counts):
    offsets = cp.empty(counts.size + 1, dtype=cp.int64)
    offsets[:-1] = counts
    offsets[-1] = 0
    return exclusive_sum(offsets, synchronize=False)


@dataclass
class _Batch:
    data: object
    offsets: object
    meta: object
    host_meta: np.ndarray
    error: object
    strings: object = None
    ways: object = None
    dense: object = None
    spans: object = None
    keep: object = None
    positions: object = None
    ids: object = None
    ranges: object = None
    relations: object = None
    rows: int = 0

    def index(self, *, relations=False):
        nstr, nway, ndense, nnode = self.host_meta[:, 5:9].sum(axis=0).tolist()
        self.strings = cp.empty((nstr, 2), dtype=cp.int64)
        self.ways = cp.zeros((nway, 16), dtype=cp.int64)
        self.dense = cp.zeros((ndense, 14), dtype=cp.int64)
        if relations:
            self.relations = cp.zeros((int(self.host_meta[:, 13].sum()), 20), dtype=cp.int64)
        _launch("pbf_index_blocks", len(self.host_meta), self.data, self.offsets, self.meta,
                self.strings, self.ways, self.dense, self.relations, self.error, len(self.host_meta))
        _check_error(self.error)
        return nway, ndense, nnode

    def select(self, layer, excluded=None):
        nway, ndense, nnode = self.index()
        if layer == "points":
            self.spans = cp.empty((nnode, 3), dtype=cp.int64)
            self.keep = cp.empty(nnode, dtype=cp.int32)
            _launch("pbf_dense_tags", ndense, self.data, self.meta, self.strings,
                    self.dense, self.spans, self.keep, self.error, ndense, warp=True)
        else:
            self.keep = cp.empty(nway, dtype=cp.int32)
            _launch("pbf_parse_ways", nway, self.data, self.meta, self.strings,
                    self.ways, self.keep, self.error, nway)
        if layer == "polygonways":
            _launch("pbf_select_area_ways", nway, self.ways, self.keep, excluded,
                    nway, 0 if excluded is None else excluded.size)
        self.positions = _offsets(self.keep)
        packet = cp.concatenate((self.positions[-1:], self.error.astype(cp.int64)))
        rows, code = _host_metadata(packet, "selected row count and validation").tolist()
        if code:
            _check_error(self.error)
        self.rows = rows

    def decode_points(self, x, y, row_base=0):
        self.ids = cp.empty(self.rows, dtype=cp.int64)
        # IDs are batch-local; coordinate views provide the final destination.
        _launch("pbf_decode_dense", self.dense.shape[0] * 3, self.data, self.meta,
                self.dense, self.ids, x[row_base:], y[row_base:], self.keep,
                self.positions, None, self.error, self.dense.shape[0], 1, warp=True)


def _prepare_device_batch(source, indices):
    data, offsets, checksums, error = source.inflate(indices)
    if error is None:
        error = cp.zeros(1, dtype=cp.int32)
    _launch("pbf_validate_inflate", len(indices), data, offsets, checksums, error, len(indices), warp=True)
    # Reject corrupt compressed data before structural traversal.
    _check_error(error)
    meta = cp.zeros((len(indices), 15), dtype=cp.int64)
    _launch("pbf_count_blocks", len(indices), data, offsets, meta, error, len(indices))
    host_meta = _host_metadata(meta, "block sizes")
    _check_error(error)
    for column, count_column in ((9, 5), (10, 6), (11, 7), (12, 8), (14, 13)):
        meta[:, column] = _offsets(meta[:, count_column])[:-1]
    return _Batch(data, offsets, meta, host_meta, error)


@dataclass(frozen=True)
class _BatchPlan:
    indices: tuple[int, ...]
    node_indices: tuple[int, ...]
    rows: int
    refs: int
    dense_base: int
    dense_count: int
    sizes: tuple[int, ...]
    nulls: tuple[int, ...]
    node_dense_counts: tuple[int, ...] = ()


def _plan_source(source, layer, attributes, raw_budget, *, id_only=False, excluded=None):
    from vibespatial.io.osm_pbf_attributes import measure_attributes

    plans, ranges = [], []
    chunks = [] if layer == "points" else None
    chunk_bytes = 0
    last_way_id = None
    for indices in source.batches(raw_budget=raw_budget):
        batch = _prepare_device_batch(source, indices)
        batch.select(layer, excluded)
        if layer == "polygonways" and batch.ways.shape[0]:
            ids = batch.ways[:, 3]
            certificate = _host_metadata(cp.stack((ids[0], ids[-1], cp.all(ids[1:] > ids[:-1]))),
                                         "standalone polygon way ordering certificate")
            if not certificate[2] or (last_way_id is not None and certificate[0] <= last_way_id):
                raise NotImplementedError("Native PBF standalone polygons require ordered unique way IDs")
            last_way_id = int(certificate[1])
        node_indices = tuple(i for i, m in zip(indices, batch.host_meta, strict=True) if m[7])
        refs = batch.rows
        base = len(ranges)
        if layer != "points":
            ref_offsets = _offsets(batch.ways[:, 6] * batch.keep)
            refs = int(_host_metadata(ref_offsets[-1:], "reference count")[0])
            dranges = cp.empty((batch.dense.shape[0], 4), dtype=cp.int64)
            _launch("pbf_decode_dense", batch.dense.shape[0]*3, batch.data, batch.meta,
                    batch.dense, None, None, None, None, None, dranges, batch.error,
                    batch.dense.shape[0], 0, warp=True)
            host_ranges = _host_metadata(dranges[:, :2].copy(), "node block ID ranges")
            ranges.extend(tuple(row) for row in host_ranges)
        sizes, nulls = (), ()
        if attributes and batch.rows:
            attr_offsets, sizes, nulls = measure_attributes(batch, layer == "points", id_only=id_only, kind=2 if layer == "polygonways" else None)
        _check_error(batch.error)
        if chunks is not None and batch.rows:
            from vibespatial.io.osm_pbf_attributes import AttributeOutput

            output_bytes = batch.rows * 16 + sum(sizes) + len(sizes) * (4*(batch.rows+1) + 4*((batch.rows+31)//32))
            remaining = _available_bytes()
            # Keep room for the next decode workspace AND the eventual compact
            # concatenation. If it cannot be proved, release all chunks and use
            # the same counted replay plan; no out-of-memory retry is needed.
            if remaining < chunk_bytes + 2*output_bytes + 64*raw_budget:
                chunks.clear()
                chunks = None
            else:
                cx, cy = cp.empty(batch.rows, dtype=cp.float64), cp.empty(batch.rows, dtype=cp.float64)
                batch.decode_points(cx, cy)
                attrs = None
                if attributes:
                    builder = AttributeOutput(True, batch.rows, sizes, nulls, id_only=id_only)
                    builder.emit(batch, attr_offsets, sizes, 0)
                    attrs = builder.finish()
                chunks.append((cx, cy, attrs))
                chunk_bytes += output_bytes
                _check_error(batch.error)
                del cx, cy, attrs
                if attributes:
                    del builder
        plans.append(_BatchPlan(indices, node_indices, batch.rows, refs, base,
                                batch.dense.shape[0], tuple(sizes), tuple(nulls),
                                tuple(int(m[7]) for m in batch.host_meta if m[7])))
        del batch
        if attributes and sizes:
            del attr_offsets
    if layer != "points" and any(a[1] >= b[0] for a, b in zip(ranges, ranges[1:])):
        raise NotImplementedError("Native PBF line lookup requires ordered node blocks")
    return plans, ranges, chunks


def _emit_source(source, plans, layer, x, y, geometry_offsets, attrs, raw_budget, excluded=None):
    id_only = attrs.id_only if attrs is not None else False
    from vibespatial.io.osm_pbf_attributes import measure_attributes

    row_base = ref_base = 0
    for plan in plans:
        if not plan.rows:
            continue
        emitted_rows = emitted_refs = 0
        emitted_sizes = [0] * len(plan.sizes)
        for indices in source.batches(plan.indices, raw_budget=raw_budget):
            batch = _prepare_device_batch(source, indices)
            batch.select(layer, excluded)
            if emitted_rows + batch.rows > plan.rows:
                raise ValueError("OSM PBF row count changed during read")
            if not batch.rows:
                continue
            if attrs is not None:
                attr_offsets, sizes, _ = measure_attributes(batch, layer == "points", id_only=id_only, kind=2 if layer == "polygonways" else None)
                emitted_sizes = [a+b for a, b in zip(emitted_sizes, sizes, strict=True)]
                if any(a>b for a, b in zip(emitted_sizes, plan.sizes, strict=True)):
                    raise ValueError("OSM PBF attribute sizes changed during read")
                attrs.emit(batch, attr_offsets, sizes, row_base)
            refs = batch.rows
            if layer == "points":
                batch.decode_points(x, y, row_base)
            else:
                offsets = _offsets(batch.ways[:, 6] * batch.keep)
                refs = int(_host_metadata(offsets[-1:], "emission reference count")[0])
                if emitted_refs + refs > plan.refs:
                    raise ValueError("OSM PBF reference count changed during read")
                _launch("pbf_emit_refs", batch.ways.shape[0], batch.data, batch.ways,
                        batch.keep, batch.positions, offsets, x, geometry_offsets,
                        batch.ways.shape[0], ctypes.c_longlong(row_base), ctypes.c_longlong(ref_base), warp=True)
            emitted_rows += batch.rows
            emitted_refs += refs
            row_base += batch.rows
            ref_base += refs
            _check_error(batch.error)
            del batch
            if attrs is not None:
                del attr_offsets
        if emitted_rows != plan.rows or emitted_refs != plan.refs or tuple(emitted_sizes) != plan.sizes:
            raise ValueError("OSM PBF source changed during read")
    geometry_offsets[-1] = ref_base


def _resolve_source(source, plans, ranges, x, y, raw_budget, *, allow_missing=False):
    if not x.size:
        return
    first = cp.asarray([r[0] for r in ranges], dtype=cp.int64)
    last = cp.asarray([r[1] for r in ranges], dtype=cp.int64)
    # Reuse final y storage as intrusive links. The head directory is capped
    # at 8 MiB for ordinary datasets; a single shard needs 8 bytes per block.
    shards = 128
    while shards > 1 and len(ranges) * shards * 8 > 8 << 20:
        shards >>= 1
    heads = cp.full((len(ranges), shards), -1, dtype=cp.int64)
    error = cp.zeros(1, dtype=cp.int32)
    _launch("pbf_link_refs", x.size, x, y, first, last, heads, error,
            ctypes.c_longlong(x.size), len(ranges), shards)
    missing = bool(_check_error(error, allow_missing=allow_missing))
    active = _host_metadata(cp.any(heads >= 0, axis=1), "referenced node block selection")
    selected_blocks, range_ids = [], {}
    for plan in plans:
        base = plan.dense_base
        for index, count in zip(plan.node_indices, plan.node_dense_counts, strict=True):
            if active[base:base+count].any():
                selected_blocks.append(index)
                range_ids[index] = tuple(range(base, base+count))
            base += count
    for indices in source.batches(selected_blocks, raw_budget=raw_budget):
        batch = _prepare_device_batch(source, indices)
        _, ndense, nnode = batch.index()
        ids = cp.empty(nnode, dtype=cp.int64)
        nx, ny = cp.empty(nnode, dtype=cp.float64), cp.empty(nnode, dtype=cp.float64)
        _launch("pbf_decode_dense", ndense*3, batch.data, batch.meta, batch.dense,
                ids, nx, ny, None, None, None, batch.error, ndense, 0, warp=True)
        directory_rows = cp.asarray([row for index in indices for row in range_ids[index]], dtype=cp.int64)
        _launch("pbf_resolve_refs", ndense*shards, ids, nx, ny, batch.dense,
                heads, x, y, error, ndense, shards, directory_rows)
        _check_error(batch.error)
        missing |= bool(_check_error(error, allow_missing=allow_missing))
        del batch, ids, nx, ny, directory_rows
    return missing


def read_osm_pbf_native(path, *, layer, geometry_only=False, tags="ways", raw_budget=256 << 20, _excluded_way_ids=None):
    """Read any admitted standard OSM layer into device-native carriers."""
    import pyarrow as pa

    from vibespatial.api._native_result_core import (
        GeometryNativeResult,
        NativeAttributeTable,
        NativeReadProvenance,
        NativeTabularResult,
    )
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.io.osm_pbf_attributes import AttributeOutput
    from vibespatial.io.osm_pbf_inflate import PbfSource
    from vibespatial.io.osm_pbf_memory import relation_workspace
    from vibespatial.io.pylibcudf import _build_device_single_family_owned

    if layer == "all":
        from vibespatial.io.osm_pbf_relations import read_all_layers
        return read_all_layers(path, geometry_only=geometry_only, tags=tags, raw_budget=raw_budget)
    if layer in ("multipolygons", "multilinestrings", "other_relations"):
        from vibespatial.io.osm_pbf_relations import read_relation_layer
        return read_relation_layer(path, layer=layer, geometry_only=geometry_only, tags=tags, raw_budget=raw_budget)
    if layer not in ("points", "lines", "polygonways"):
        raise NotImplementedError("Unsupported native OSM PBF layer")
    get_cuda_runtime()._ensure_context()
    attributes = not geometry_only
    id_only = tags is False
    with PbfSource(path) as source:
        largest_blob = max((b.raw_size for b in source.blobs), default=0)
        available = max(_available_bytes() - (16 << 20), 0)
        raw_budget = max(largest_blob, min(raw_budget, available // 64))
        with relation_workspace(64*raw_budget, available, kind="blob"), hotpath_stage("osm/plan", category="setup"):
            plans, ranges, chunks = _plan_source(source, layer, attributes, raw_budget, id_only=id_only, excluded=_excluded_way_ids)
        rows, refs = sum(p.rows for p in plans), sum(p.refs for p in plans)
        if max(rows, refs) > 2147483647:
            raise OverflowError("OSM geometry exceeds the owned int32 offset capacity")
        ncol = ((2 if layer == "polygonways" else 1) if id_only else 10 if layer == "points" else 25 if layer == "polygonways" else 9) if attributes else 0
        char_bytes = sum(sum(p.sizes) for p in plans)
        output_bytes = 16*refs + 11*rows + 4 + char_bytes + ncol*(4*(rows+1) + 4*((rows+31)//32))
        if layer == "polygonways":
            output_bytes += rows*8
        if attributes and layer == "lines" and not id_only:
            output_bytes += rows*4
        with relation_workspace(output_bytes, _available_bytes(), kind="output"):
            attrs = None
            if chunks:
                x = cp.concatenate([c[0] for c in chunks])
                y = cp.concatenate([c[1] for c in chunks])
                offsets = cp.arange(rows+1, dtype=cp.int32)
                if attributes:
                    import pylibcudf as plc

                    from vibespatial.cuda._runtime import pylibcudf_current_stream

                    tables = [c[2] for c in chunks]
                    table = plc.concatenate.concatenate(
                        [t.device_table for t in tables],
                        stream=pylibcudf_current_stream(*(t.device_table for t in tables)),
                    )
                    attribute_table = NativeAttributeTable(
                        device_table=table, column_override=tables[0].column_override,
                        schema_override=tables[0].schema_override,
                    )
                else:
                    attribute_table = NativeAttributeTable(arrow_table=pa.table({"_": pa.nulls(rows)}).select([]))
                del chunks
            else:
                directory_bytes = 24*len(ranges) + (8 << 20)
                free_workspace = _available_bytes() - output_bytes - directory_bytes - (16 << 20)
                raw_budget = max(largest_blob, min(raw_budget, max(free_workspace, 0)//64))
                if attributes:
                    sizes = [sum(p.sizes[c] for p in plans if p.sizes) for c in range(ncol)]
                    nulls = [sum(p.nulls[c] for p in plans if p.nulls) for c in range(ncol)]
                    attrs = AttributeOutput(layer == "points", rows, sizes, nulls, id_only=id_only, kind=2 if layer == "polygonways" else None)
                x, y = cp.empty(refs, dtype=cp.float64), cp.empty(refs, dtype=cp.float64)
                offsets = cp.arange(rows+1, dtype=cp.int32) if layer == "points" else cp.empty(rows+1, dtype=cp.int32)
                with relation_workspace(64*raw_budget, _available_bytes(), kind="blob"), hotpath_stage("osm/emit", category="emit"):
                    _emit_source(source, plans, layer, x, y, offsets, attrs, raw_budget, _excluded_way_ids)
                if layer != "points":
                    with relation_workspace(64*raw_budget+directory_bytes, _available_bytes(), kind="blob"), hotpath_stage("osm/resolve", category="refine"):
                        _resolve_source(source, plans, ranges, x, y, raw_budget)
                attribute_table = attrs.finish() if attrs is not None else NativeAttributeTable(
                    arrow_table=pa.table({"_": pa.nulls(rows)}).select([]),
                )
            source.validate_identity()
            owned = _build_device_single_family_owned(
                family=GeometryFamily.POINT if layer == "points" else GeometryFamily.MULTIPOLYGON if layer == "polygonways" else GeometryFamily.LINESTRING,
                validity_device=cp.ones(rows, dtype=cp.bool_), x_device=x, y_device=y,
                geometry_offsets_device=cp.arange(rows+1, dtype=cp.int32) if layer == "polygonways" else offsets,
                part_offsets_device=cp.arange(rows+1, dtype=cp.int32) if layer == "polygonways" else None,
                ring_offsets_device=offsets if layer == "polygonways" else None,
                empty_mask_device=cp.zeros(rows, dtype=cp.bool_),
                all_valid=True, detail="OSM PBF counted replay with device tags and bounded reference lookup",
            )
    # Owned geometry currently has no cross-stream readiness carrier. Complete
    # this synchronous IO boundary before handing buffers to another stream.
    cp.cuda.get_current_stream().synchronize()
    return NativeTabularResult(
        attributes=attribute_table, geometry=GeometryNativeResult(owned=owned, crs="EPSG:4326"),
        geometry_name="geometry", column_order=(*attribute_table.columns, "geometry"),
        provenance=NativeReadProvenance(surface="read_file", format_name="OSM-PBF",
                                        source=str(path), backend="nvcomp-nvrtc"),
    )
