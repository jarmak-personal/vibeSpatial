"""Native PBF count/replay, tag semantics, and byte-codec canaries."""
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
    _build_primitive_group_with_ways,
    _build_way,
    _encode_length_delimited,
    _encode_varint_field,
)
from vibespatial import has_gpu_runtime

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required")]


def _fixture(compress=True, count=259):
    strings = [b"", b"name", b'quoted "x" \\y', b"source", b"survey", b"highway", b"residential", b"note", b"is_in", b"layer", b"-2", b"bridge", b"yes", b"building", b"area", b"no"]
    kv = []
    for i in range(count):
        if i % 7 == 0:
            kv.extend([1, 2, 8, 0, 0])  # empty value (SID 0) then delimiter
        elif i % 11 == 0:
            kv.extend([7, 2, 0])  # note is significant for points, despite ignored export
        else:
            kv.append(0)
    dense = _build_dense_nodes([1000] + [1]*(count-1), [420000000]+[1]*(count-1), [-710000000]+[-1]*(count-1), kv)
    nodes = _build_primitive_block_with_stringtable([_build_primitive_group(dense)], stringtable_entries=strings)
    ways = [
        _build_way(12345678901, [1000, 1, count-2], keys=[5, 9, 11, 1], vals=[6, 10, 12, 2]),
        _build_way(12345678902, [1002, 2, -2], keys=[13, 14], vals=[12, 15]),
        _build_way(12345678903, [1000, 1], keys=[7], vals=[2]),
        _build_way(12345678904, [1000, 1, -1], keys=[13], vals=[12]),
        _build_way(12345678905, [1000, 1]),  # untagged is absent
    ]
    way_block = _build_primitive_block_with_stringtable([_build_primitive_group_with_ways(ways)], stringtable_entries=strings)
    return _build_osm_header() + _build_pbf_block("OSMData", nodes, compress) + _build_pbf_block("OSMData", way_block, compress)


def _assert_oracle(path, payload, layer):
    import cupy as cp

    meta, table = pyogrio.read_arrow(path, layer=layer)
    attributes = payload.attributes.to_arrow(index=False)
    for name in attributes.column_names:
        assert attributes[name].combine_chunks().equals(table[name].combine_chunks()), name
    family = next(iter(payload.geometry.owned.device_state.families.values()))
    actual = np.column_stack((cp.asnumpy(family.x), cp.asnumpy(family.y)))
    expected_geom = shapely.from_wkb(table[meta["geometry_name"] or "wkb_geometry"].to_numpy())
    np.testing.assert_array_equal(actual, shapely.get_coordinates(expected_geom))
    np.testing.assert_array_equal(cp.asnumpy(family.geometry_offsets), np.r_[0, np.cumsum(shapely.get_num_coordinates(expected_geom))])


@pytest.mark.parametrize("layer", ["points", "lines"])
@pytest.mark.parametrize("compress", [False, True])
def test_standard_layers_match_gdal_all_columns(tmp_path, layer, compress):
    from vibespatial.io.osm_pbf_native import read_osm_pbf_native

    path = tmp_path / "sample.osm.pbf"
    path.write_bytes(_fixture(compress))
    payload = read_osm_pbf_native(path, layer=layer)
    _assert_oracle(path, payload, layer)
    assert payload.attributes.is_device_backed
    assert payload.geometry.owned.device_state is not None
    assert all(not family.host_materialized for family in payload.geometry.owned.families.values())


@pytest.mark.parametrize("layer", ["points", "lines"])
def test_counted_replay_with_small_batches(tmp_path, monkeypatch, layer):
    import vibespatial.io.osm_pbf_native as native

    path = tmp_path / "sample.osm.pbf"
    path.write_bytes(_fixture())
    calls = 0
    def available():
        nonlocal calls
        calls += 1
        return (512 if calls == 1 else 64) << 20
    monkeypatch.setattr(native, "_available_bytes", available)
    payload = native.read_osm_pbf_native(path, layer=layer)
    _assert_oracle(path, payload, layer)


@pytest.mark.parametrize("layer", ["points", "lines"])
@pytest.mark.parametrize("geometry_only,tags", [(True, "ways"), (False, False)])
def test_projection(tmp_path, layer, geometry_only, tags):
    from vibespatial.io.osm_pbf_native import read_osm_pbf_native

    path = tmp_path / "sample.osm.pbf"
    path.write_bytes(_fixture())
    payload = read_osm_pbf_native(path, layer=layer, geometry_only=geometry_only, tags=tags)
    assert tuple(payload.attributes.columns) == (() if geometry_only else ("osm_id",))
    _assert_oracle(path, payload, layer)


def test_signed_offsets_are_int64_not_zigzag(tmp_path):
    from vibespatial.io.osm_pbf_native import read_osm_pbf_native

    dense = _build_dense_nodes([3], [2], [4], [1, 2, 0])
    block = _build_primitive_block_with_stringtable([_build_primitive_group(dense)], stringtable_entries=[b"", b"name", b"test"], granularity=1000)
    block += _encode_varint_field(19, (-1234567) & ((1 << 64)-1))
    block += _encode_varint_field(20, 4567890)
    path = tmp_path / "offset.osm.pbf"
    path.write_bytes(_build_osm_header() + _build_pbf_block("OSMData", block))
    _assert_oracle(path, read_osm_pbf_native(path, layer="points"), "points")


def test_zlib_checksum_is_validated_before_parse(tmp_path):
    from vibespatial.io.osm_pbf_native import read_osm_pbf_native

    path = tmp_path / "corrupt.osm.pbf"
    data = bytearray(_fixture())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="Malformed"):
        read_osm_pbf_native(path, layer="points")


def test_ordinary_nodes_decline_explicitly(tmp_path):
    from vibespatial.io.osm_pbf_native import read_osm_pbf_native

    block = _build_primitive_block_with_stringtable([_encode_length_delimited(1, b"\x08\x02")], stringtable_entries=[b""])
    path = tmp_path / "ordinary.osm.pbf"
    path.write_bytes(_build_osm_header() + _build_pbf_block("OSMData", block))
    with pytest.raises(NotImplementedError, match="DenseNodes"):
        read_osm_pbf_native(path, layer="points")


def test_public_reader_uses_native_bytes(tmp_path, monkeypatch):
    from vibespatial.io.file import read_vector_file_native

    path = tmp_path / "sample.osm.pbf"
    path.write_bytes(_fixture())
    def no_host_parse(*args, **kwargs):
        raise AssertionError("native PBF ingress must not call GDAL")
    monkeypatch.setattr(pyogrio, "read_arrow", no_host_parse)
    monkeypatch.setenv("VIBESPATIAL_STRICT_NATIVE", "1")
    result = read_vector_file_native(path, layer="lines")
    assert result.provenance.backend == "nvcomp-nvrtc"
    assert result.geometry.row_count == 3


def test_pbf_metadata_only_transfers(tmp_path, monkeypatch):
    from vibespatial.cuda._runtime import get_cuda_runtime
    from vibespatial.io.osm_pbf_native import read_osm_pbf_native

    path = tmp_path / "sample.osm.pbf"
    path.write_bytes(_fixture())
    runtime = get_cuda_runtime()
    original = runtime.copy_device_to_host
    transfers = []
    def copy(values, **kwargs):
        transfers.append((values.nbytes, kwargs.get("reason", "")))
        return original(values, **kwargs)
    monkeypatch.setattr(runtime, "copy_device_to_host", copy)
    result = read_osm_pbf_native(path, layer="lines")
    assert result.attributes.is_device_backed
    assert transfers and all(reason.startswith("OSM PBF ") and "metadata" in reason for _, reason in transfers)
    assert max(size for size, _ in transfers) < 1024


def test_node_at_null_island_is_preserved(tmp_path):
    from vibespatial.io.osm_pbf_native import read_osm_pbf_native

    # GDAL 3.11's node index reserves (0,0) as a missing sentinel. It is a
    # legitimate coordinate; use the mechanical geometry oracle for this case.
    dense = _build_dense_nodes([1,1,1], [0,10_000_000,10_000_000], [0,10_000_000,10_000_000])
    way = _build_way(100, [1,1,1], keys=[1], vals=[2])
    block = _build_primitive_block_with_stringtable([
        _build_primitive_group(dense), _build_primitive_group_with_ways([way]),
    ], stringtable_entries=[b"",b"highway",b"residential"])
    path = tmp_path/"origin.osm.pbf"
    path.write_bytes(_build_osm_header()+_build_pbf_block("OSMData", block))
    result = read_osm_pbf_native(path, layer="lines")
    actual = np.asarray(result.to_geodataframe().geometry)
    assert shapely.equals_exact(actual[0], shapely.LineString([(0,0),(1,1),(2,2)]), 0)


@pytest.mark.parametrize("suffix", [b"\x00", b"\xff"*4, b"\x00\x00\x00\x02\x0a", b"\x00\x00\x00\x01\x00"])
def test_truncated_or_invalid_framing_is_rejected(tmp_path, suffix):
    from vibespatial.io.osm_pbf_inflate import PbfSource

    path = tmp_path/"bad.osm.pbf"
    path.write_bytes(_build_osm_header()+suffix)
    with pytest.raises(ValueError), PbfSource(path):
        pass


def test_source_replacement_is_detected(tmp_path):
    from vibespatial.io.osm_pbf_inflate import PbfSource

    path = tmp_path/"source.osm.pbf"
    path.write_bytes(_fixture())
    with PbfSource(path) as source:
        replacement = tmp_path/"replacement.osm.pbf"
        replacement.write_bytes(_fixture())
        replacement.replace(path)
        with pytest.raises(ValueError, match="changed"):
            source.validate_identity()
