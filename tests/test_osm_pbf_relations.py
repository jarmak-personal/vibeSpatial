"""Native PBF ring, multipart and ordered collection compatibility canaries."""
from __future__ import annotations

import numpy as np
import pyogrio
import pytest
import shapely

from tests.test_osm_gpu import (
    _build_dense_nodes,
    _build_osm_header,
    _build_pbf_block,
    _build_primitive_block_with_stringtable,
    _build_primitive_group,
    _build_primitive_group_with_relations,
    _build_primitive_group_with_ways,
    _build_relation,
    _build_way,
)
from vibespatial import has_gpu_runtime

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required")]


def relation_fixture():
    strings = [b"", b"type", b"multipolygon", b"route", b"restriction", b"outer", b"inner", b"subarea", b"name", b"test", b"landuse", b"forest"]
    xy = np.array([(0,0),(10,0),(10,10),(0,10),(2,2),(4,2),(4,4),(2,4),(20,0),(22,0),(22,2),(20,2)], dtype=np.int64)
    xy = (xy + [-70, 40])*10_000_000
    def delta(values):
        return np.diff(values, prepend=0).tolist()
    nodes = _build_dense_nodes([1]*len(xy), delta(xy[:,1]), delta(xy[:,0]))
    ways = [
        _build_way(100, delta([1,2,3])),
        _build_way(101, delta([1,4,3])),  # reversed chain fragment
        _build_way(102, delta([5,6,7,8,5])),
        _build_way(103, delta([9,10,11,12,9]), keys=[10], vals=[11]),
    ]
    relations = [
        _build_relation(1000, [(101,1,5),(100,1,5),(102,1,6),(103,1,5)], keys=[1,8], vals=[2,9]),
        _build_relation(1001, [(101,1,0),(103,1,0),(1,0,0),(99999,1,0)], keys=[1,8], vals=[3,9]),
        _build_relation(1002, [(1,0,0),(103,1,0),(2,0,0),(101,1,0),(100,1,7),(1000,2,0)], keys=[1,8], vals=[4,9]),
    ]
    block = _build_primitive_block_with_stringtable([
        _build_primitive_group(nodes), _build_primitive_group_with_ways(ways),
        _build_primitive_group_with_relations(relations),
    ], stringtable_entries=strings)
    return _build_osm_header()+_build_pbf_block("OSMData", block)


@pytest.mark.parametrize("layer", ["multipolygons", "multilinestrings", "other_relations"])
def test_relation_assembly_oracle(tmp_path, layer):
    from vibespatial.io.osm_pbf_inflate import PbfSource
    from vibespatial.io.osm_pbf_relations import catalogue, relation_chunks

    path = tmp_path/"relation.osm.pbf"
    path.write_bytes(relation_fixture())
    with PbfSource(path) as source:
        cat = catalogue(source, 32 << 20)
        results = list(relation_chunks(source, cat, layer, raw_budget=32 << 20, geometry_only=False, tags=True))
    assert len(results) == 1
    result = results[0][0]
    frame = result.to_geodataframe()
    meta, table = pyogrio.read_arrow(path, layer=layer)
    expected = shapely.from_wkb(table[meta['geometry_name'] or 'wkb_geometry'].to_numpy())
    actual = np.asarray(frame.geometry)
    assert len(actual) == len(expected)
    assert shapely.equals(actual, expected).all(), (shapely.to_wkt(actual), shapely.to_wkt(expected))
    if layer == "other_relations":
        assert shapely.equals_exact(actual, expected, tolerance=0).all()
    attrs = result.attributes.to_arrow(index=False)
    for name in attrs.column_names:
        assert attrs[name].combine_chunks().equals(table[name].combine_chunks()), name


@pytest.mark.parametrize("layer", [None, "all", "multipolygons", "multilinestrings", "other_relations"])
def test_public_native_admits_all_shapes(tmp_path, monkeypatch, layer):
    from vibespatial.io.file import read_vector_file_native

    path = tmp_path/"all.osm.pbf"
    path.write_bytes(relation_fixture())
    monkeypatch.setenv("VIBESPATIAL_STRICT_NATIVE", "1")
    def no_gdal(*args, **kwargs):
        raise AssertionError("PBF ingress must decode relations on the device")
    monkeypatch.setattr(pyogrio, "read_arrow", no_gdal)
    result = read_vector_file_native(path, **({} if layer is None else {"layer": layer}))
    assert result.provenance.backend == "nvcomp-nvrtc"
    assert result.attributes.is_device_backed
    assert result.geometry.row_count == (3 if layer in (None, "all") else 1)
    if layer in (None, "all"):
        assert result.attributes.to_arrow(index=False)["osm_element"].to_pylist() == ["relation"]*3


def test_collection_member_order_survives_row_operations(tmp_path):
    import cupy as cp

    from vibespatial.api._native_metadata import NativeGeometryMetadata
    from vibespatial.io.osm_pbf_native import read_osm_pbf_native

    path = tmp_path/"collection.osm.pbf"
    path.write_bytes(relation_fixture())
    result = read_osm_pbf_native(path, layer="other_relations")
    expected = np.asarray(result.to_geodataframe().geometry)
    taken = result.geometry.take(cp.asarray([0, 0], dtype=cp.int64))
    actual = np.asarray(taken.to_geoseries(index=None, name="geometry"))
    assert shapely.equals_exact(actual, np.repeat(expected, 2), 0).all()
    metadata = NativeGeometryMetadata.from_native_geometry(taken)
    assert all(part.collection_positions is not None for part in metadata.composition_parts)
    masked = taken.mask_capacity(cp.asarray([False, True]))
    actual = np.asarray(masked.to_geoseries(index=None, name="geometry"))
    assert actual[0] is None
    assert shapely.equals_exact(actual[1], expected[0], 0)


def test_single_large_relation_uses_explicit_managed_workspace(tmp_path, monkeypatch):
    import vibespatial.io.osm_pbf_native as native

    path = tmp_path/"paged.osm.pbf"
    path.write_bytes(relation_fixture())
    calls = 0
    def available():
        nonlocal calls
        calls += 1
        return (512 if calls == 1 else 16) << 20
    monkeypatch.setattr(native, "_available_bytes", available)
    result = native.read_osm_pbf_native(path, layer="other_relations")
    assert result.geometry.row_count == 1
    actual = np.asarray(result.to_geodataframe().geometry)
    meta, table = pyogrio.read_arrow(path, layer="other_relations")
    expected = shapely.from_wkb(table[meta["geometry_name"] or "wkb_geometry"].to_numpy())
    assert shapely.equals_exact(actual, expected, 0).all()


def test_collection_export_from_nondefault_stream(tmp_path):
    import cupy as cp

    from vibespatial.io.osm_pbf_native import read_osm_pbf_native

    path = tmp_path/"stream.osm.pbf"
    path.write_bytes(relation_fixture())
    with cp.cuda.Stream(non_blocking=True):
        result = read_osm_pbf_native(path, layer="other_relations")
    actual = np.asarray(result.to_geodataframe().geometry)
    meta, table = pyogrio.read_arrow(path, layer="other_relations")
    expected = shapely.from_wkb(table[meta["geometry_name"] or "wkb_geometry"].to_numpy())
    assert shapely.equals_exact(actual, expected, 0).all()


@pytest.mark.parametrize("layer", ["points", "lines", "multilinestrings", "multipolygons", "other_relations", "all"])
@pytest.mark.parametrize("geometry_only,tags", [(True, True), (False, False)])
def test_empty_layer_preserves_projection(tmp_path, layer, geometry_only, tags):
    from vibespatial.io.osm_pbf_native import read_osm_pbf_native

    path = tmp_path/"empty.osm.pbf"
    path.write_bytes(_build_osm_header())
    result = read_osm_pbf_native(path, layer=layer, geometry_only=geometry_only, tags=tags)
    assert result.geometry.row_count == 0
    assert tuple(result.attributes.columns) == (() if geometry_only else
           ("osm_id", "osm_element", "osm_way_id") if layer == "all" else
           ("osm_id", "osm_way_id") if layer == "multipolygons" else ("osm_id",))


def test_duplicate_slanted_rings_follow_binary64_nesting(tmp_path):
    from vibespatial.io.osm_pbf_native import read_osm_pbf_native

    # Reproduces rounded-midpoint classification in OGR's boundary test.
    outer = [(-704630000,416350000),(-704610000,416350000),(-704610000,416360000),(-704630000,416360000)]
    inner = [(-704623207,416354294),(-704625023,416353989),(-704625261,416354780),
             (-704624010,416354990),(-704623983,416354901),(-704623418,416354996)]
    xy = np.array(outer+inner, dtype=np.int64)
    def delta(values):
        return np.diff(values, prepend=0).tolist()
    dense = _build_dense_nodes([1]*len(xy), delta(xy[:,1]), delta(xy[:,0]))
    ways = [_build_way(100, delta([1,2,3,4,1])), _build_way(101, delta([5,6,7,8,9,10,5])),
            _build_way(102, delta([5,6,7,8,9,10,5]))]
    relation = _build_relation(1000, [(100,1,3),(101,1,4),(102,1,4)], keys=[1], vals=[2])
    block = _build_primitive_block_with_stringtable([
        _build_primitive_group(dense), _build_primitive_group_with_ways(ways),
        _build_primitive_group_with_relations([relation]),
    ], stringtable_entries=[b"",b"type",b"multipolygon",b"outer",b"inner"])
    path = tmp_path/"duplicate.osm.pbf"
    path.write_bytes(_build_osm_header()+_build_pbf_block("OSMData",block))
    result = read_osm_pbf_native(path, layer="multipolygons")
    actual = np.asarray(result.to_geodataframe().geometry)
    meta, table = pyogrio.read_arrow(path, layer="multipolygons")
    expected = shapely.from_wkb(table[meta["geometry_name"] or "wkb_geometry"].to_numpy())
    assert shapely.equals_exact(actual, expected, 0, normalize=True).all()
    assert shapely.get_num_geometries(actual).tolist() == [2]


def test_final_attribute_concat_can_page_without_changing_collection_order(tmp_path, monkeypatch):
    import vibespatial.io.osm_pbf_native as native
    from vibespatial.io.osm_pbf_memory import concatenate_results

    path = tmp_path/"concat.osm.pbf"
    path.write_bytes(relation_fixture())
    result = native.read_osm_pbf_native(path, layer="other_relations")
    expected = np.asarray(result.to_geodataframe().geometry)
    monkeypatch.setattr(native, "_available_bytes", lambda: 0)
    combined = concatenate_results([result, result])
    actual = np.asarray(combined.to_geodataframe().geometry)
    assert shapely.equals_exact(actual, np.repeat(expected, 2), 0).all()
    assert combined.attributes.to_arrow(index=False)["osm_id"].to_pylist() == ["1002", "1002"]


@pytest.mark.parametrize("lengths", [(2,4), (4,2), (0,4), (0,0)])
def test_duplicate_referenced_way_ids_decline_before_expansion(tmp_path, lengths):
    from vibespatial.io.osm_pbf_native import read_osm_pbf_native

    nodes = _build_dense_nodes([1]*4, [400000000,100,100,100], [-700000000,100,100,100])
    ways = [_build_way(100, [1]*size) for size in lengths]
    relation = _build_relation(1000, [(100,1,0)], keys=[1], vals=[2])
    block = _build_primitive_block_with_stringtable([
        _build_primitive_group(nodes), _build_primitive_group_with_ways(ways),
        _build_primitive_group_with_relations([relation]),
    ], stringtable_entries=[b"", b"type", b"route"])
    path = tmp_path/"duplicate-ids.osm.pbf"
    path.write_bytes(_build_osm_header()+_build_pbf_block("OSMData",block))
    with pytest.raises(NotImplementedError, match="unique referenced way IDs"):
        read_osm_pbf_native(path, layer="multilinestrings")



def test_degenerate_closed_ring_keeps_relation_rings_as_separate_parts():
    import cupy as cp

    from vibespatial.io.osm_pbf_rings import nest_rings

    xy = cp.asarray([(0,0),(10,0),(10,10),(0,10),(0,0),(2,2),(3,2),(2,2)], dtype=cp.float64)
    result = nest_rings(xy[:,0].copy(), xy[:,1].copy(), cp.asarray([0,5,8]), cp.asarray([0,0]), 1)
    # organizePolygons declines hole assignment for the entire input when a
    # directly closed member cannot meet the four-coordinate ring contract.
    assert cp.asnumpy(result[3]).tolist() == [0,1,2]
    assert cp.asnumpy(result[4]).tolist() == [0,2]


@pytest.mark.parametrize("layer", ["points", "lines", "multipolygons", "other_relations"])
def test_target_crs_transforms_coordinates(tmp_path, layer):
    import cupy as cp
    from pyproj import Transformer

    from tests.test_osm_pbf_native import _fixture
    from vibespatial.io.file import read_vector_file_native
    from vibespatial.io.osm_pbf_native import read_osm_pbf_native

    path = tmp_path/"reproject.osm.pbf"
    path.write_bytes(_fixture() if layer in ("points", "lines") else relation_fixture())
    original = read_osm_pbf_native(path, layer=layer)
    xy = shapely.get_coordinates(np.asarray(original.to_geodataframe().geometry))
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        result = read_vector_file_native(path, layer=layer, target_crs="EPSG:3857")
    assert stream.done
    actual = shapely.get_coordinates(np.asarray(result.to_geodataframe().geometry))
    x, y = Transformer.from_crs(4326,3857,always_xy=True).transform(xy[:,0],xy[:,1])
    np.testing.assert_allclose(actual, np.column_stack((x,y)), rtol=1e-10, atol=1e-7)
    assert result.geometry.crs == "EPSG:3857"


@pytest.mark.parametrize("members_count", [8, 20_000])
def test_shared_junction_with_many_open_members_has_no_rings(members_count):
    import cupy as cp

    from vibespatial.io.osm_pbf_rings import assemble_rings

    members = cp.zeros((members_count, 10), dtype=cp.int64)
    members[:,1] = 1
    members[:,2] = 1
    members[:,5] = 2
    x = cp.empty(2*members_count, dtype=cp.float64)
    y = cp.empty_like(x)
    x[::2], y[::2] = -70, 40
    x[1::2] = -69+cp.arange(members_count)/members_count
    y[1::2] = 41
    offsets = cp.arange(members_count+1, dtype=cp.int64)*2
    ox, oy, rings, rows = assemble_rings(members, offsets, cp.asarray([0,members_count]), x, y, 1)
    assert not ox.size and not oy.size and not rows.size
    assert cp.asnumpy(rings).tolist() == [0]



def test_inherited_tags_admit_expanded_fanout(tmp_path, monkeypatch):
    import vibespatial.io.osm_pbf_native as native
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    rows = 64
    label = b"a"*8192
    nodes = _build_dense_nodes([1]*4, [400000000,0,1000,0], [-700000000,1000,0,-1000])
    ways = [_build_way(100, [1,1,1,1,-3], keys=[4,6], vals=[5,7])]
    relations = [_build_relation(1000+i, [(100,1,3)], keys=[1], vals=[2]) for i in range(rows)]
    block = _build_primitive_block_with_stringtable([
        _build_primitive_group(nodes), _build_primitive_group_with_ways(ways),
        _build_primitive_group_with_relations(relations),
    ], stringtable_entries=[b"",b"type",b"multipolygon",b"outer",b"landuse",b"forest",b"name",label])
    path = tmp_path/"inherit.osm.pbf"
    path.write_bytes(_build_osm_header()+_build_pbf_block("OSMData", block))
    monkeypatch.setattr(native, "_available_bytes", lambda: 0)
    clear_dispatch_events()
    result = native.read_osm_pbf_native(path, layer="multipolygons")
    events = get_dispatch_events()
    assert any(e.implementation == "cuda_managed_inherited_attributes" for e in events)
    actual = result.attributes.to_arrow(index=False)
    assert actual["name"].to_pylist() == [label.decode()]*rows
    assert actual["landuse"].to_pylist() == ["forest"]*rows
    _, expected = pyogrio.read_arrow(path, layer="multipolygons")
    for name in actual.column_names:
        assert actual[name].combine_chunks().equals(expected[name].combine_chunks()), name


def test_unordered_standalone_way_ids_decline(tmp_path):
    from vibespatial.io.osm_pbf_native import read_osm_pbf_native

    nodes = _build_dense_nodes([1]*4, [400000000,0,1000,0], [-700000000,1000,0,-1000])
    ways = [_build_way(i, [1,1,1,1,-3], keys=[1], vals=[2]) for i in (101,100)]
    block = _build_primitive_block_with_stringtable([
        _build_primitive_group(nodes), _build_primitive_group_with_ways(ways),
    ], stringtable_entries=[b"",b"landuse",b"forest"])
    path = tmp_path/"unordered.osm.pbf"
    path.write_bytes(_build_osm_header()+_build_pbf_block("OSMData", block))
    with pytest.raises(NotImplementedError, match="ordered unique way IDs"):
        read_osm_pbf_native(path, layer="multipolygons")
