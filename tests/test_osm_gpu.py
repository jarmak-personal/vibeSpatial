"""Tests for GPU OSM PBF reader: protobuf + geometry construction on GPU.

Tests cover:
- Protobuf varint encoding/decoding (CPU helpers + GPU kernel)
- ZigZag encoding/decoding
- Delta decoding via cumulative sum
- Coordinate scaling from nanodegrees to degrees
- End-to-end: minimal PBF file -> Point OwnedGeometryArray
- Block index parsing
- Blob decompression (raw and zlib)
- Way extraction: CPU field parsing, GPU coordinate gathering
- Way classification: open -> LineString, closed -> Polygon
- Mixed nodes + ways in single PBF file
- End-to-end: read_osm_pbf() returns both nodes and ways

Test data is constructed programmatically as raw protobuf bytes -- no
external dependencies (osmium, etc.) required.
"""
from __future__ import annotations

import struct
import tempfile
import zlib
from pathlib import Path

import numpy as np
import pytest

from vibespatial.geometry.buffers import GeometryFamily

try:
    import cupy as cp

    HAS_GPU = True
except (ImportError, ModuleNotFoundError):
    HAS_GPU = False

needs_gpu = pytest.mark.skipif(not HAS_GPU, reason="GPU not available")


# ---------------------------------------------------------------------------
# Protobuf encoding helpers -- build raw protobuf bytes for test data
# ---------------------------------------------------------------------------


def _encode_varint(value: int) -> bytes:
    """Encode an unsigned integer as a protobuf varint."""
    result = bytearray()
    while value > 0x7F:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value & 0x7F)
    return bytes(result)


def _encode_zigzag(value: int) -> int:
    """ZigZag-encode a signed integer."""
    return (value << 1) ^ (value >> 63) if value >= 0 else ((-value - 1) << 1) | 1


def _encode_sint64(value: int) -> bytes:
    """Encode a signed int64 as a ZigZag varint."""
    return _encode_varint(_encode_zigzag(value))


def _encode_field_tag(field_number: int, wire_type: int) -> bytes:
    """Encode a protobuf field tag."""
    return _encode_varint((field_number << 3) | wire_type)


def _encode_length_delimited(field_number: int, data: bytes) -> bytes:
    """Encode a length-delimited protobuf field."""
    return _encode_field_tag(field_number, 2) + _encode_varint(len(data)) + data


def _encode_varint_field(field_number: int, value: int) -> bytes:
    """Encode a varint protobuf field."""
    return _encode_field_tag(field_number, 0) + _encode_varint(value)


def _encode_packed_sint64(values: list[int]) -> bytes:
    """Encode a packed repeated sint64 field (just the packed bytes, no tag)."""
    result = bytearray()
    for v in values:
        result.extend(_encode_sint64(v))
    return bytes(result)


def _encode_packed_uint32(values: list[int]) -> bytes:
    """Encode a packed repeated uint32 field (just the packed bytes, no tag)."""
    result = bytearray()
    for v in values:
        result.extend(_encode_varint(v))
    return bytes(result)


# ---------------------------------------------------------------------------
# Build a minimal valid PBF file
# ---------------------------------------------------------------------------


def _build_dense_nodes(
    id_deltas: list[int],
    lat_deltas: list[int],
    lon_deltas: list[int],
    keys_vals: list[int] | None = None,
) -> bytes:
    """Build a DenseNodes protobuf message.

    Parameters
    ----------
    keys_vals
        Optional packed uint32 array for DenseNodes tags.
        Uses the interleaved key_sid/val_sid/0 encoding:
        ``[key1, val1, 0, key2, val2, key3, val3, 0, ...]``
        where ``0`` delimits between nodes.
    """
    id_packed = _encode_packed_sint64(id_deltas)
    lat_packed = _encode_packed_sint64(lat_deltas)
    lon_packed = _encode_packed_sint64(lon_deltas)
    result = (
        _encode_length_delimited(1, id_packed)    # field 1: id
        + _encode_length_delimited(8, lat_packed)  # field 8: lat
        + _encode_length_delimited(9, lon_packed)  # field 9: lon
    )
    if keys_vals is not None:
        kv_packed = _encode_packed_uint32(keys_vals)
        result += _encode_length_delimited(10, kv_packed)  # field 10: keys_vals
    return result


def _build_primitive_group(dense_nodes: bytes) -> bytes:
    """Build a PrimitiveGroup containing DenseNodes."""
    return _encode_length_delimited(2, dense_nodes)  # field 2: dense


def _build_primitive_block(
    primitive_group: bytes,
    granularity: int = 100,
    lat_offset: int = 0,
    lon_offset: int = 0,
) -> bytes:
    """Build a PrimitiveBlock protobuf message."""
    # Stringtable (required, can be empty)
    stringtable = _encode_length_delimited(1, b"")  # field 1: stringtable

    result = stringtable
    # field 2: primitivegroup
    result += _encode_length_delimited(2, primitive_group)
    # field 17: granularity
    result += _encode_varint_field(17, granularity)
    # field 19: lat_offset (ZigZag-encoded)
    if lat_offset != 0:
        result += _encode_varint_field(19, _encode_zigzag(lat_offset))
    # field 20: lon_offset (ZigZag-encoded)
    if lon_offset != 0:
        result += _encode_varint_field(20, _encode_zigzag(lon_offset))

    return result


def _build_blob(raw_data: bytes, compress: bool = True) -> bytes:
    """Build a Blob protobuf message."""
    if compress:
        compressed = zlib.compress(raw_data)
        return (
            _encode_varint_field(2, len(raw_data))              # field 2: raw_size
            + _encode_length_delimited(3, compressed)           # field 3: zlib_data
        )
    else:
        return _encode_length_delimited(1, raw_data)            # field 1: raw


def _build_blob_header(block_type: str, blob_size: int) -> bytes:
    """Build a BlobHeader protobuf message."""
    return (
        _encode_length_delimited(1, block_type.encode("utf-8"))  # field 1: type
        + _encode_varint_field(3, blob_size)                     # field 3: datasize
    )


def _build_pbf_block(block_type: str, payload: bytes, compress: bool = True) -> bytes:
    """Build a complete (BlobHeader size prefix + BlobHeader + Blob) block."""
    blob = _build_blob(payload, compress=compress)
    blob_header = _build_blob_header(block_type, len(blob))
    return struct.pack(">I", len(blob_header)) + blob_header + blob


def _build_osm_header() -> bytes:
    """Build a minimal OSMHeader block."""
    # Minimal HeaderBlock: just required_features
    header_block = _encode_length_delimited(
        4, b"OsmSchema-V0.6",  # field 4: required_features
    )
    return _build_pbf_block("OSMHeader", header_block)


def _build_way(
    way_id: int,
    ref_deltas: list[int],
    keys: list[int] | None = None,
    vals: list[int] | None = None,
) -> bytes:
    """Build a Way protobuf message.

    Parameters
    ----------
    way_id
        The Way ID.
    ref_deltas
        Delta-encoded sint64 node references.
    keys, vals
        Optional stringtable indices for tags.
    """
    result = _encode_varint_field(1, way_id)  # field 1: id
    if keys:
        result += _encode_length_delimited(2, _encode_packed_uint32(keys))
    if vals:
        result += _encode_length_delimited(3, _encode_packed_uint32(vals))
    result += _encode_length_delimited(8, _encode_packed_sint64(ref_deltas))
    return result


def _build_primitive_group_with_ways(ways_bytes: list[bytes]) -> bytes:
    """Build a PrimitiveGroup containing Way messages."""
    result = b""
    for way_data in ways_bytes:
        result += _encode_length_delimited(3, way_data)  # field 3: ways
    return result


def _build_primitive_block_with_stringtable(
    primitive_groups: list[bytes],
    stringtable_entries: list[bytes] | None = None,
    granularity: int = 100,
    lat_offset: int = 0,
    lon_offset: int = 0,
) -> bytes:
    """Build a PrimitiveBlock with an explicit stringtable and multiple groups."""
    # Build stringtable
    st_content = b""
    if stringtable_entries:
        for entry in stringtable_entries:
            st_content += _encode_length_delimited(1, entry)  # field 1: s (repeated)
    stringtable = _encode_length_delimited(1, st_content)

    result = stringtable
    for group in primitive_groups:
        result += _encode_length_delimited(2, group)
    result += _encode_varint_field(17, granularity)
    if lat_offset != 0:
        result += _encode_varint_field(19, _encode_zigzag(lat_offset))
    if lon_offset != 0:
        result += _encode_varint_field(20, _encode_zigzag(lon_offset))
    return result


def _build_test_pbf(
    id_deltas: list[int],
    lat_deltas: list[int],
    lon_deltas: list[int],
    granularity: int = 100,
    lat_offset: int = 0,
    lon_offset: int = 0,
    compress: bool = True,
) -> bytes:
    """Build a complete minimal PBF file with one OSMData block."""
    # OSM Header block
    header = _build_osm_header()

    # Build the data block
    dense = _build_dense_nodes(id_deltas, lat_deltas, lon_deltas)
    group = _build_primitive_group(dense)
    primitive_block = _build_primitive_block(
        group,
        granularity=granularity,
        lat_offset=lat_offset,
        lon_offset=lon_offset,
    )
    data_block = _build_pbf_block("OSMData", primitive_block, compress=compress)

    return header + data_block


def _build_test_pbf_with_ways(
    node_id_deltas: list[int],
    node_lat_deltas: list[int],
    node_lon_deltas: list[int],
    ways: list[tuple[int, list[int]]],
    stringtable_entries: list[bytes] | None = None,
    granularity: int = 100,
    compress: bool = True,
) -> bytes:
    """Build a PBF file with both DenseNodes and Ways.

    Parameters
    ----------
    ways
        List of (way_id, absolute_refs) tuples.  Refs are converted to
        delta encoding automatically.
    """
    header = _build_osm_header()

    # Build DenseNodes group
    dense = _build_dense_nodes(node_id_deltas, node_lat_deltas, node_lon_deltas)
    dense_group = _build_primitive_group(dense)

    # Build Ways group: convert absolute refs to delta-encoded
    way_messages = []
    for way_id, absolute_refs in ways:
        ref_deltas = [absolute_refs[0]]
        for i in range(1, len(absolute_refs)):
            ref_deltas.append(absolute_refs[i] - absolute_refs[i - 1])
        way_messages.append(_build_way(way_id, ref_deltas))
    ways_group = _build_primitive_group_with_ways(way_messages)

    primitive_block = _build_primitive_block_with_stringtable(
        [dense_group, ways_group],
        stringtable_entries=stringtable_entries,
        granularity=granularity,
    )
    data_block = _build_pbf_block("OSMData", primitive_block, compress=compress)

    return header + data_block


def _write_temp_pbf(content: bytes) -> Path:
    """Write bytes to a temporary .osm.pbf file and return the path."""
    f = tempfile.NamedTemporaryFile(suffix=".osm.pbf", delete=False)
    f.write(content)
    f.close()
    return Path(f.name)


# ---------------------------------------------------------------------------
# Tests: CPU protobuf helpers
# ---------------------------------------------------------------------------


class TestVarintEncoding:
    """Test varint encode/decode round-trip on CPU."""

    def test_small_values(self):
        from vibespatial.io.osm_gpu import _decode_varint

        for val in [0, 1, 42, 127]:
            encoded = _encode_varint(val)
            decoded, consumed = _decode_varint(encoded, 0)
            assert decoded == val
            assert consumed == len(encoded)

    def test_multibyte_values(self):
        from vibespatial.io.osm_gpu import _decode_varint

        for val in [128, 255, 300, 16384, 2**20, 2**32, 2**50]:
            encoded = _encode_varint(val)
            decoded, consumed = _decode_varint(encoded, 0)
            assert decoded == val
            assert consumed == len(encoded)

    def test_zigzag_round_trip(self):
        """ZigZag encoding: 0->0, -1->1, 1->2, -2->3, 2->4, etc."""
        values = [0, -1, 1, -2, 2, -100, 100, -2**30, 2**30]
        for val in values:
            zz = _encode_zigzag(val)
            # Manual ZigZag decode
            decoded = (zz >> 1) ^ -(zz & 1)
            assert decoded == val, f"ZigZag round-trip failed for {val}"


class TestBlockIndexParsing:
    """Test CPU-side PBF block index parsing."""

    def test_header_and_data_blocks(self):
        from vibespatial.io.osm_gpu import _parse_block_index

        pbf = _build_test_pbf(
            id_deltas=[1, 1, 1],
            lat_deltas=[10, 20, 30],
            lon_deltas=[40, 50, 60],
        )
        path = _write_temp_pbf(pbf)
        try:
            blocks = _parse_block_index(path)
            assert len(blocks) == 2
            assert blocks[0].block_type == "OSMHeader"
            assert blocks[1].block_type == "OSMData"
            assert blocks[1].blob_size > 0
        finally:
            path.unlink()

    def test_empty_file(self):
        from vibespatial.io.osm_gpu import _parse_block_index

        path = _write_temp_pbf(b"")
        try:
            blocks = _parse_block_index(path)
            assert blocks == []
        finally:
            path.unlink()


class TestBlobDecompression:
    """Test CPU-side blob decompression."""

    def test_zlib_compressed(self):
        from vibespatial.io.osm_gpu import _decompress_blob

        payload = b"Hello, OSM world! " * 100
        blob = _build_blob(payload, compress=True)
        result = _decompress_blob(blob)
        assert result == payload

    def test_raw_uncompressed(self):
        from vibespatial.io.osm_gpu import _decompress_blob

        payload = b"Raw data"
        blob = _build_blob(payload, compress=False)
        result = _decompress_blob(blob)
        assert result == payload


class TestDenseNodesExtraction:
    """Test CPU-side extraction of DenseNodes fields from PrimitiveBlock."""

    def test_single_block(self):
        from vibespatial.io.osm_gpu import _extract_dense_nodes_blocks

        id_deltas = [10, 1, 1]
        lat_deltas = [407000000, 100, 200]
        lon_deltas = [-740000000, -50, -100]

        dense = _build_dense_nodes(id_deltas, lat_deltas, lon_deltas)
        group = _build_primitive_group(dense)
        pblock = _build_primitive_block(group, granularity=100)

        results = _extract_dense_nodes_blocks([pblock])
        assert len(results) == 1
        assert results[0].granularity == 100
        assert results[0].lat_offset == 0
        assert results[0].lon_offset == 0
        # Verify varint byte content is non-empty
        assert len(results[0].id_bytes) > 0
        assert len(results[0].lat_bytes) > 0
        assert len(results[0].lon_bytes) > 0

    def test_custom_granularity_and_offsets(self):
        from vibespatial.io.osm_gpu import _extract_dense_nodes_blocks

        dense = _build_dense_nodes([1], [100], [200])
        group = _build_primitive_group(dense)
        pblock = _build_primitive_block(
            group, granularity=1000, lat_offset=500000000, lon_offset=-1000000000,
        )

        results = _extract_dense_nodes_blocks([pblock])
        assert len(results) == 1
        assert results[0].granularity == 1000
        assert results[0].lat_offset == 500000000
        assert results[0].lon_offset == -1000000000


# ---------------------------------------------------------------------------
# Tests: GPU varint decoding
# ---------------------------------------------------------------------------


class TestGpuVarintDecode:
    """Test GPU-side varint decoding kernel."""

    @needs_gpu
    def test_simple_zigzag_values(self):
        """Decode a sequence of known ZigZag-encoded varints."""
        from vibespatial.io.osm_gpu import (
            _count_varints,
            _gpu_decode_varints_zigzag,
            _locate_varint_positions,
        )

        # Encode known values
        values = [0, 1, -1, 42, -42, 1000, -1000, 2**20, -(2**20)]
        packed = _encode_packed_sint64(values)

        n = _count_varints(packed)
        assert n == len(values)

        positions = _locate_varint_positions(packed, n)
        d_result = _gpu_decode_varints_zigzag(packed, positions)
        result = cp.asnumpy(d_result)

        np.testing.assert_array_equal(result, values)

    @needs_gpu
    def test_large_values(self):
        """Decode large int64 values near the boundaries."""
        from vibespatial.io.osm_gpu import (
            _count_varints,
            _gpu_decode_varints_zigzag,
            _locate_varint_positions,
        )

        values = [2**50, -(2**50), 2**62, -(2**62), 0]
        packed = _encode_packed_sint64(values)
        n = _count_varints(packed)
        positions = _locate_varint_positions(packed, n)
        d_result = _gpu_decode_varints_zigzag(packed, positions)
        result = cp.asnumpy(d_result)

        np.testing.assert_array_equal(result, values)

    @needs_gpu
    def test_empty_input(self):
        """Empty byte array produces empty output."""
        from vibespatial.io.osm_gpu import _gpu_decode_varints_zigzag

        positions = np.empty(0, dtype=np.int64)
        d_result = _gpu_decode_varints_zigzag(b"", positions)
        assert d_result.shape[0] == 0

    @needs_gpu
    def test_single_zero(self):
        """Single zero varint."""
        from vibespatial.io.osm_gpu import (
            _count_varints,
            _gpu_decode_varints_zigzag,
            _locate_varint_positions,
        )

        packed = _encode_packed_sint64([0])
        n = _count_varints(packed)
        positions = _locate_varint_positions(packed, n)
        d_result = _gpu_decode_varints_zigzag(packed, positions)
        assert cp.asnumpy(d_result)[0] == 0


# ---------------------------------------------------------------------------
# Tests: GPU delta decoding and coordinate scaling
# ---------------------------------------------------------------------------


class TestGpuDeltaDecode:
    """Test GPU-side delta decode + coordinate scaling."""

    @needs_gpu
    def test_cumsum_produces_absolute_ids(self):
        """Delta-encoded IDs: cumsum of [10, 1, 2, 3] -> [10, 11, 13, 16]."""
        from vibespatial.io.osm_gpu import DenseNodesBlock, _gpu_delta_decode_and_scale

        # Each block's deltas are cumsum'd independently
        # IDs: 10, 11, 13, 16
        id_deltas = [10, 1, 2, 3]
        lat_deltas = [100, 200, 300, 400]
        lon_deltas = [500, 600, 700, 800]

        block = DenseNodesBlock(
            id_bytes=_encode_packed_sint64(id_deltas),
            lat_bytes=_encode_packed_sint64(lat_deltas),
            lon_bytes=_encode_packed_sint64(lon_deltas),
            granularity=100,
            lat_offset=0,
            lon_offset=0,
        )

        result = _gpu_delta_decode_and_scale([block])
        assert result is not None
        d_ids, d_lat, d_lon = result

        ids = cp.asnumpy(d_ids)
        np.testing.assert_array_equal(ids, [10, 11, 13, 16])

    @needs_gpu
    def test_coordinate_scaling(self):
        """Verify nanodegree -> degree conversion with default granularity."""
        from vibespatial.io.osm_gpu import DenseNodesBlock, _gpu_delta_decode_and_scale

        # With granularity=100, lat in nanodegrees = cumsum * 100 * 1e-9
        # If we want lat = 40.7 degrees:
        #   raw nanodegree = 40.7 / (100 * 1e-9) = 40.7 / 1e-7 = 407000000
        # So delta = 407000000 for the first node
        lat_nano_value = 407000000   # -> 40.7 degrees
        lon_nano_value = -740000000  # -> -74.0 degrees

        block = DenseNodesBlock(
            id_bytes=_encode_packed_sint64([1]),
            lat_bytes=_encode_packed_sint64([lat_nano_value]),
            lon_bytes=_encode_packed_sint64([lon_nano_value]),
            granularity=100,
            lat_offset=0,
            lon_offset=0,
        )

        result = _gpu_delta_decode_and_scale([block])
        assert result is not None
        _, d_lat, d_lon = result

        lat = cp.asnumpy(d_lat)
        lon = cp.asnumpy(d_lon)

        np.testing.assert_allclose(lat, [40.7], rtol=1e-10)
        np.testing.assert_allclose(lon, [-74.0], rtol=1e-10)

    @needs_gpu
    def test_coordinate_scaling_with_offsets(self):
        """Verify coordinate scaling with non-zero lat/lon offsets."""
        from vibespatial.io.osm_gpu import DenseNodesBlock, _gpu_delta_decode_and_scale

        # lat = delta * granularity * 1e-9 + lat_offset * 1e-9
        # With granularity=100, lat_offset=1000000000 (0.1 degree offset)
        # delta=407000000: lat = 407000000 * 100 * 1e-9 + 1000000000 * 1e-9
        #                     = 40.7 + 1.0 = 41.7
        block = DenseNodesBlock(
            id_bytes=_encode_packed_sint64([1]),
            lat_bytes=_encode_packed_sint64([407000000]),
            lon_bytes=_encode_packed_sint64([-740000000]),
            granularity=100,
            lat_offset=10000000000,   # +10.0 degrees
            lon_offset=-5000000000,   # -5.0 degrees
        )

        result = _gpu_delta_decode_and_scale([block])
        assert result is not None
        _, d_lat, d_lon = result

        lat = cp.asnumpy(d_lat)
        lon = cp.asnumpy(d_lon)

        np.testing.assert_allclose(lat, [40.7 + 10.0], rtol=1e-10)
        np.testing.assert_allclose(lon, [-74.0 + (-5.0)], rtol=1e-10)

    @needs_gpu
    def test_multiple_nodes_delta(self):
        """Multiple nodes with delta encoding: verify absolute coordinates."""
        from vibespatial.io.osm_gpu import DenseNodesBlock, _gpu_delta_decode_and_scale

        # Three nodes at lat = 40.0, 40.1, 40.3
        # nanodegree values: 400000000, 401000000, 403000000
        # deltas (at granularity=100): 4000000, 10000, 20000
        # (since nanodeg = cumsum(delta) * granularity = cumsum(delta) * 100)
        # Actually: raw = value / (gran * 1e-9) = value / 1e-7
        # 40.0 / 1e-7 = 400000000
        # 40.1 / 1e-7 = 401000000 -> delta = 1000000
        # 40.3 / 1e-7 = 403000000 -> delta = 2000000

        block = DenseNodesBlock(
            id_bytes=_encode_packed_sint64([100, 1, 1]),
            lat_bytes=_encode_packed_sint64([400000000, 1000000, 2000000]),
            lon_bytes=_encode_packed_sint64([-740000000, -100000, -200000]),
            granularity=100,
            lat_offset=0,
            lon_offset=0,
        )

        result = _gpu_delta_decode_and_scale([block])
        assert result is not None
        d_ids, d_lat, d_lon = result

        ids = cp.asnumpy(d_ids)
        lat = cp.asnumpy(d_lat)
        lon = cp.asnumpy(d_lon)

        np.testing.assert_array_equal(ids, [100, 101, 102])
        np.testing.assert_allclose(lat, [40.0, 40.1, 40.3], rtol=1e-10)
        np.testing.assert_allclose(
            lon, [-74.0, -74.01, -74.03], rtol=1e-10,
        )

    @needs_gpu
    def test_multi_block_delta_reset(self):
        """Delta encoding resets at block boundaries -- cumsum is per-block."""
        from vibespatial.io.osm_gpu import DenseNodesBlock, _gpu_delta_decode_and_scale

        # Block 1: ids 100, 101 (deltas: 100, 1)
        # Block 2: ids 200, 201 (deltas: 200, 1)  -- NOT cumsum from block 1
        block1 = DenseNodesBlock(
            id_bytes=_encode_packed_sint64([100, 1]),
            lat_bytes=_encode_packed_sint64([400000000, 1000000]),
            lon_bytes=_encode_packed_sint64([-740000000, -100000]),
            granularity=100,
            lat_offset=0,
            lon_offset=0,
        )
        block2 = DenseNodesBlock(
            id_bytes=_encode_packed_sint64([200, 1]),
            lat_bytes=_encode_packed_sint64([500000000, 2000000]),
            lon_bytes=_encode_packed_sint64([-600000000, 300000]),
            granularity=100,
            lat_offset=0,
            lon_offset=0,
        )

        result = _gpu_delta_decode_and_scale([block1, block2])
        assert result is not None
        d_ids, d_lat, d_lon = result

        ids = cp.asnumpy(d_ids)
        lat = cp.asnumpy(d_lat)
        lon = cp.asnumpy(d_lon)

        # Block 1: ids [100, 101], lat [40.0, 40.1], lon [-74.0, -74.01]
        # Block 2: ids [200, 201], lat [50.0, 50.2], lon [-60.0, -59.97]
        np.testing.assert_array_equal(ids, [100, 101, 200, 201])
        np.testing.assert_allclose(lat, [40.0, 40.1, 50.0, 50.2], rtol=1e-10)
        np.testing.assert_allclose(
            lon, [-74.0, -74.01, -60.0, -59.97], rtol=1e-10,
        )


# ---------------------------------------------------------------------------
# Tests: End-to-end PBF -> Point OwnedGeometryArray
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """End-to-end tests: PBF file -> OsmGpuResult -> coordinates."""

    @needs_gpu
    def test_single_node(self):
        """One node at known coordinates."""
        from vibespatial.io.osm_gpu import read_osm_pbf_nodes

        # Node at (lat=51.5, lon=-0.1) -- London
        # nanodeg = value / (100 * 1e-9) = value / 1e-7
        lat_nano = 515000000     # 51.5 degrees
        lon_nano = -1000000      # -0.1 degrees

        pbf = _build_test_pbf(
            id_deltas=[42],
            lat_deltas=[lat_nano],
            lon_deltas=[lon_nano],
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf_nodes(path)

            assert result.n_nodes == 1
            assert result.nodes is not None
            assert result.node_ids is not None

            node_id = cp.asnumpy(result.node_ids)
            assert node_id[0] == 42

            # Check coordinates from device state
            ds = result.nodes.device_state
            assert ds is not None
            assert GeometryFamily.POINT in ds.families
            pt = ds.families[GeometryFamily.POINT]

            x = cp.asnumpy(pt.x)  # longitude
            y = cp.asnumpy(pt.y)  # latitude

            np.testing.assert_allclose(x, [-0.1], rtol=1e-9)
            np.testing.assert_allclose(y, [51.5], rtol=1e-9)
        finally:
            path.unlink()

    @needs_gpu
    def test_three_nodes(self):
        """Three nodes with delta encoding."""
        from vibespatial.io.osm_gpu import read_osm_pbf_nodes

        # Three nodes:
        # Node 1: id=100, lat=40.0, lon=-74.0
        # Node 2: id=101, lat=40.1, lon=-74.01
        # Node 3: id=102, lat=40.3, lon=-74.03
        pbf = _build_test_pbf(
            id_deltas=[100, 1, 1],
            lat_deltas=[400000000, 1000000, 2000000],
            lon_deltas=[-740000000, -100000, -200000],
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf_nodes(path)

            assert result.n_nodes == 3
            ids = cp.asnumpy(result.node_ids)
            np.testing.assert_array_equal(ids, [100, 101, 102])

            ds = result.nodes.device_state
            pt = ds.families[GeometryFamily.POINT]
            x = cp.asnumpy(pt.x)
            y = cp.asnumpy(pt.y)

            np.testing.assert_allclose(y, [40.0, 40.1, 40.3], rtol=1e-10)
            np.testing.assert_allclose(x, [-74.0, -74.01, -74.03], rtol=1e-10)
        finally:
            path.unlink()

    @needs_gpu
    def test_uncompressed_blob(self):
        """PBF with uncompressed (raw) blob data."""
        from vibespatial.io.osm_gpu import read_osm_pbf_nodes

        pbf = _build_test_pbf(
            id_deltas=[1],
            lat_deltas=[0],
            lon_deltas=[0],
            compress=False,
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf_nodes(path)
            assert result.n_nodes == 1
        finally:
            path.unlink()

    @needs_gpu
    def test_file_not_found(self):
        """Non-existent file raises FileNotFoundError."""
        from vibespatial.io.osm_gpu import read_osm_pbf_nodes

        with pytest.raises(FileNotFoundError):
            read_osm_pbf_nodes("/nonexistent/path.osm.pbf")

    @needs_gpu
    def test_geometry_array_properties(self):
        """OwnedGeometryArray has correct residency and family structure."""
        from vibespatial.io.osm_gpu import read_osm_pbf_nodes
        from vibespatial.runtime.residency import Residency

        pbf = _build_test_pbf(
            id_deltas=[1, 2, 3],
            lat_deltas=[100, 200, 300],
            lon_deltas=[400, 500, 600],
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf_nodes(path)
            owned = result.nodes

            assert owned.residency == Residency.DEVICE
            assert GeometryFamily.POINT in owned.families
            assert owned._row_count == 3

            # Tags should all be POINT
            tags = owned.tags
            from vibespatial.geometry.owned import FAMILY_TAGS

            assert all(t == FAMILY_TAGS[GeometryFamily.POINT] for t in tags)

            # Validity should all be True
            assert all(owned.validity)
        finally:
            path.unlink()

    @needs_gpu
    def test_negative_deltas(self):
        """Nodes with negative delta values (ZigZag encoding)."""
        from vibespatial.io.osm_gpu import read_osm_pbf_nodes

        # Nodes moving south and east
        # Node 1: id=1000, lat=50.0, lon=10.0
        # Node 2: id=999 (delta=-1), lat=49.0 (delta=-10000000), lon=11.0 (delta=10000000)
        pbf = _build_test_pbf(
            id_deltas=[1000, -1],
            lat_deltas=[500000000, -10000000],
            lon_deltas=[100000000, 10000000],
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf_nodes(path)
            assert result.n_nodes == 2

            ids = cp.asnumpy(result.node_ids)
            np.testing.assert_array_equal(ids, [1000, 999])

            ds = result.nodes.device_state
            pt = ds.families[GeometryFamily.POINT]
            y = cp.asnumpy(pt.y)
            x = cp.asnumpy(pt.x)

            np.testing.assert_allclose(y, [50.0, 49.0], rtol=1e-10)
            np.testing.assert_allclose(x, [10.0, 11.0], rtol=1e-10)
        finally:
            path.unlink()

    @needs_gpu
    def test_custom_granularity(self):
        """Non-default granularity scales coordinates differently."""
        from vibespatial.io.osm_gpu import read_osm_pbf_nodes

        # granularity=1 means 1 nanodegree per unit
        # To get lat=1.0: delta = 1.0 / (1 * 1e-9) = 1000000000
        pbf = _build_test_pbf(
            id_deltas=[1],
            lat_deltas=[1000000000],
            lon_deltas=[2000000000],
            granularity=1,
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf_nodes(path)
            assert result.n_nodes == 1

            ds = result.nodes.device_state
            pt = ds.families[GeometryFamily.POINT]
            y = cp.asnumpy(pt.y)
            x = cp.asnumpy(pt.x)

            np.testing.assert_allclose(y, [1.0], rtol=1e-10)
            np.testing.assert_allclose(x, [2.0], rtol=1e-10)
        finally:
            path.unlink()


# ---------------------------------------------------------------------------
# Tests: Varint position locator
# ---------------------------------------------------------------------------


class TestVarintPositionLocator:
    """Test the CPU-side varint position locator."""

    def test_simple_positions(self):
        from vibespatial.io.osm_gpu import _count_varints, _locate_varint_positions

        # Three single-byte varints: 0x00, 0x01, 0x7F
        data = bytes([0x00, 0x01, 0x7F])
        n = _count_varints(data)
        assert n == 3
        positions = _locate_varint_positions(data, n)
        np.testing.assert_array_equal(positions, [0, 1, 2])

    def test_multibyte_varints(self):
        from vibespatial.io.osm_gpu import _count_varints, _locate_varint_positions

        # First varint: 300 = 0xAC 0x02 (2 bytes)
        # Second varint: 1 = 0x01 (1 byte)
        data = bytes([0xAC, 0x02, 0x01])
        n = _count_varints(data)
        assert n == 2
        positions = _locate_varint_positions(data, n)
        np.testing.assert_array_equal(positions, [0, 2])

    def test_empty_data(self):
        from vibespatial.io.osm_gpu import _count_varints, _locate_varint_positions

        n = _count_varints(b"")
        assert n == 0
        positions = _locate_varint_positions(b"", 0)
        assert len(positions) == 0


# ---------------------------------------------------------------------------
# Tests: CPU Way field extraction
# ---------------------------------------------------------------------------


class TestWayExtraction:
    """Test CPU-side Way field extraction from PrimitiveBlocks."""

    def test_single_way_extraction(self):
        """Extract a single Way with known refs and tags."""
        from vibespatial.io.osm_gpu import _extract_way_blocks

        # Build a Way: id=1000, refs=[10, 20, 30] (deltas: 10, 10, 10)
        way = _build_way(1000, [10, 10, 10])
        group = _build_primitive_group_with_ways([way])
        pblock = _build_primitive_block_with_stringtable([group])

        results = _extract_way_blocks([pblock])
        assert len(results) == 1
        wb = results[0]
        assert wb.way_ids == [1000]
        assert wb.refs_per_way == [[10, 20, 30]]

    def test_multiple_ways(self):
        """Extract multiple Ways from a single PrimitiveBlock."""
        from vibespatial.io.osm_gpu import _extract_way_blocks

        # Way 1: refs [10, 20] (deltas: 10, 10)
        # Way 2: refs [30, 40, 50] (deltas: 30, 10, 10)
        way1 = _build_way(100, [10, 10])
        way2 = _build_way(200, [30, 10, 10])
        group = _build_primitive_group_with_ways([way1, way2])
        pblock = _build_primitive_block_with_stringtable([group])

        results = _extract_way_blocks([pblock])
        assert len(results) == 1
        wb = results[0]
        assert wb.way_ids == [100, 200]
        assert wb.refs_per_way == [[10, 20], [30, 40, 50]]

    def test_way_with_tags(self):
        """Extract Way tags (keys/vals are stringtable indices)."""
        from vibespatial.io.osm_gpu import _extract_way_blocks

        way = _build_way(500, [1, 1], keys=[1, 2], vals=[3, 4])
        group = _build_primitive_group_with_ways([way])
        pblock = _build_primitive_block_with_stringtable(
            [group],
            stringtable_entries=[b"", b"highway", b"building", b"residential", b"yes"],
        )

        results = _extract_way_blocks([pblock])
        assert len(results) == 1
        wb = results[0]
        assert wb.tag_keys_per_way == [[1, 2]]
        assert wb.tag_vals_per_way == [[3, 4]]
        assert wb.stringtable[1] == b"highway"
        assert wb.stringtable[4] == b"yes"

    def test_way_delta_decoding(self):
        """Verify delta decoding of Way node refs on CPU."""
        from vibespatial.io.osm_gpu import _extract_way_blocks

        # Absolute refs: [100, 105, 110, 100] (closed ring)
        # Deltas: [100, 5, 5, -10]
        way = _build_way(42, [100, 5, 5, -10])
        group = _build_primitive_group_with_ways([way])
        pblock = _build_primitive_block_with_stringtable([group])

        results = _extract_way_blocks([pblock])
        wb = results[0]
        assert wb.refs_per_way == [[100, 105, 110, 100]]

    def test_no_ways_in_block(self):
        """Block with only DenseNodes produces no WayBlocks."""
        from vibespatial.io.osm_gpu import _extract_way_blocks

        dense = _build_dense_nodes([1, 1], [100, 200], [300, 400])
        group = _build_primitive_group(dense)
        pblock = _build_primitive_block(group)

        results = _extract_way_blocks([pblock])
        assert results == []


# ---------------------------------------------------------------------------
# Tests: GPU node lookup table
# ---------------------------------------------------------------------------


class TestNodeLookup:
    """Test GPU-side node lookup table construction."""

    @needs_gpu
    def test_sorted_by_id(self):
        """Node lookup table is sorted by node ID."""
        from vibespatial.io.osm_gpu import _build_node_lookup

        # Unsorted node IDs
        d_ids = cp.array([30, 10, 20], dtype=cp.int64)
        d_x = cp.array([3.0, 1.0, 2.0], dtype=cp.float64)
        d_y = cp.array([33.0, 11.0, 22.0], dtype=cp.float64)

        d_sorted_ids, d_sorted_x, d_sorted_y = _build_node_lookup(d_ids, d_x, d_y)

        sorted_ids = cp.asnumpy(d_sorted_ids)
        sorted_x = cp.asnumpy(d_sorted_x)
        sorted_y = cp.asnumpy(d_sorted_y)

        np.testing.assert_array_equal(sorted_ids, [10, 20, 30])
        np.testing.assert_array_equal(sorted_x, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(sorted_y, [11.0, 22.0, 33.0])


# ---------------------------------------------------------------------------
# Tests: GPU Way coordinate gathering
# ---------------------------------------------------------------------------


class TestWayCoordGathering:
    """Test GPU binary search kernel for Way coordinate resolution."""

    @needs_gpu
    def test_basic_gathering(self):
        """Binary search resolves known node refs to coordinates."""
        from vibespatial.io.osm_gpu import _gpu_gather_way_coords

        # Sorted node table
        d_sorted_ids = cp.array([10, 20, 30, 40, 50], dtype=cp.int64)
        d_sorted_x = cp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=cp.float64)
        d_sorted_y = cp.array([11.0, 22.0, 33.0, 44.0, 55.0], dtype=cp.float64)

        # Way refs: look up nodes 20, 40, 10
        d_refs = cp.array([20, 40, 10], dtype=cp.int64)

        d_out_x, d_out_y = _gpu_gather_way_coords(
            d_sorted_ids, d_sorted_x, d_sorted_y, d_refs,
        )

        out_x = cp.asnumpy(d_out_x)
        out_y = cp.asnumpy(d_out_y)

        np.testing.assert_array_equal(out_x, [2.0, 4.0, 1.0])
        np.testing.assert_array_equal(out_y, [22.0, 44.0, 11.0])

    @needs_gpu
    def test_missing_node_produces_nan(self):
        """Node ref not in lookup table produces NaN."""
        from vibespatial.io.osm_gpu import _gpu_gather_way_coords

        d_sorted_ids = cp.array([10, 20, 30], dtype=cp.int64)
        d_sorted_x = cp.array([1.0, 2.0, 3.0], dtype=cp.float64)
        d_sorted_y = cp.array([11.0, 22.0, 33.0], dtype=cp.float64)

        # Ref 99 does not exist
        d_refs = cp.array([20, 99], dtype=cp.int64)

        d_out_x, d_out_y = _gpu_gather_way_coords(
            d_sorted_ids, d_sorted_x, d_sorted_y, d_refs,
        )

        out_x = cp.asnumpy(d_out_x)
        out_y = cp.asnumpy(d_out_y)

        np.testing.assert_equal(out_x[0], 2.0)
        assert np.isnan(out_x[1])
        assert np.isnan(out_y[1])

    @needs_gpu
    def test_empty_refs(self):
        """Empty ref array produces empty output."""
        from vibespatial.io.osm_gpu import _gpu_gather_way_coords

        d_sorted_ids = cp.array([10], dtype=cp.int64)
        d_sorted_x = cp.array([1.0], dtype=cp.float64)
        d_sorted_y = cp.array([11.0], dtype=cp.float64)
        d_refs = cp.empty(0, dtype=cp.int64)

        d_out_x, d_out_y = _gpu_gather_way_coords(
            d_sorted_ids, d_sorted_x, d_sorted_y, d_refs,
        )
        assert d_out_x.shape[0] == 0
        assert d_out_y.shape[0] == 0


# ---------------------------------------------------------------------------
# Tests: Way classification (LineString vs Polygon)
# ---------------------------------------------------------------------------


class TestWayClassification:
    """Test Way classification into LineString and Polygon."""

    @needs_gpu
    def test_closed_way_is_polygon(self):
        """Closed Way (first ref == last ref) classified as Polygon."""
        from vibespatial.io.osm_gpu import read_osm_pbf

        # 4 nodes forming a square
        # Node 1: id=1, lat=0, lon=0
        # Node 2: id=2, lat=0, lon=1
        # Node 3: id=3, lat=1, lon=1
        # Node 4: id=4, lat=1, lon=0
        node_id_deltas = [1, 1, 1, 1]
        node_lat_deltas = [0, 0, 10000000, 10000000]
        node_lon_deltas = [0, 10000000, 10000000, 0]

        # Way refs: [1, 2, 3, 4, 1] (closed ring)
        ways = [(100, [1, 2, 3, 4, 1])]

        pbf = _build_test_pbf_with_ways(
            node_id_deltas, node_lat_deltas, node_lon_deltas, ways,
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf(path)
            assert result.n_ways == 1
            assert result.ways is not None

            # Should be classified as Polygon
            ds = result.ways.device_state
            assert ds is not None
            assert GeometryFamily.POLYGON in ds.families
        finally:
            path.unlink()

    @needs_gpu
    def test_open_way_is_linestring(self):
        """Open Way (first ref != last ref) classified as LineString."""
        from vibespatial.io.osm_gpu import read_osm_pbf

        # 3 nodes forming a line
        node_id_deltas = [1, 1, 1]
        node_lat_deltas = [0, 10000000, 20000000]
        node_lon_deltas = [0, 10000000, 20000000]

        # Way refs: [1, 2, 3] (open)
        ways = [(100, [1, 2, 3])]

        pbf = _build_test_pbf_with_ways(
            node_id_deltas, node_lat_deltas, node_lon_deltas, ways,
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf(path)
            assert result.n_ways == 1
            assert result.ways is not None

            # Should be classified as LineString
            ds = result.ways.device_state
            assert ds is not None
            assert GeometryFamily.LINESTRING in ds.families
        finally:
            path.unlink()

    @needs_gpu
    def test_mixed_open_and_closed_ways(self):
        """PBF with both open and closed ways produces mixed geometry."""
        from vibespatial.io.osm_gpu import read_osm_pbf

        # 5 nodes
        node_id_deltas = [1, 1, 1, 1, 1]
        node_lat_deltas = [0, 10000000, 20000000, 30000000, 40000000]
        node_lon_deltas = [0, 10000000, 20000000, 30000000, 40000000]

        ways = [
            (100, [1, 2, 3]),       # open -> LineString
            (200, [1, 2, 3, 1]),    # closed -> Polygon
        ]

        pbf = _build_test_pbf_with_ways(
            node_id_deltas, node_lat_deltas, node_lon_deltas, ways,
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf(path)
            assert result.n_ways == 2
            assert result.ways is not None

            ds = result.ways.device_state
            assert ds is not None
            # Mixed: should have both families
            assert GeometryFamily.LINESTRING in ds.families
            assert GeometryFamily.POLYGON in ds.families
        finally:
            path.unlink()


# ---------------------------------------------------------------------------
# Tests: End-to-end read_osm_pbf (nodes + ways)
# ---------------------------------------------------------------------------


class TestReadOsmPbf:
    """End-to-end tests for the combined nodes + ways reader."""

    @needs_gpu
    def test_nodes_and_ways(self):
        """read_osm_pbf returns both nodes and ways."""
        from vibespatial.io.osm_gpu import read_osm_pbf

        # 4 nodes at known coordinates
        # lat = delta * 100 * 1e-9
        # Node 1: lat=0.0, lon=0.0
        # Node 2: lat=0.001, lon=0.001
        # Node 3: lat=0.002, lon=0.002
        # Node 4: lat=0.003, lon=0.003
        node_id_deltas = [1, 1, 1, 1]
        node_lat_deltas = [0, 10000, 10000, 10000]
        node_lon_deltas = [0, 10000, 10000, 10000]

        # One open way: [1, 2, 3]
        ways = [(500, [1, 2, 3])]

        pbf = _build_test_pbf_with_ways(
            node_id_deltas, node_lat_deltas, node_lon_deltas, ways,
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf(path)

            # Nodes
            assert result.n_nodes == 4
            assert result.nodes is not None
            assert result.node_ids is not None

            # Ways
            assert result.n_ways == 1
            assert result.ways is not None
            assert result.way_ids is not None
            way_ids = cp.asnumpy(result.way_ids)
            assert way_ids[0] == 500
        finally:
            path.unlink()

    @needs_gpu
    def test_way_coordinates_match_nodes(self):
        """Way coordinates are correctly resolved from the node table."""
        from vibespatial.io.osm_gpu import read_osm_pbf

        # Node 10: lat=40.0, lon=-74.0
        # Node 20: lat=41.0, lon=-73.0
        # Node 30: lat=42.0, lon=-72.0
        node_id_deltas = [10, 10, 10]
        node_lat_deltas = [400000000, 10000000, 10000000]
        node_lon_deltas = [-740000000, 10000000, 10000000]

        # Way uses nodes [10, 20, 30]
        ways = [(1000, [10, 20, 30])]

        pbf = _build_test_pbf_with_ways(
            node_id_deltas, node_lat_deltas, node_lon_deltas, ways,
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf(path)
            assert result.n_ways == 1

            ds = result.ways.device_state
            line_buf = ds.families[GeometryFamily.LINESTRING]
            x = cp.asnumpy(line_buf.x)
            y = cp.asnumpy(line_buf.y)

            # x = longitude, y = latitude
            np.testing.assert_allclose(y, [40.0, 41.0, 42.0], rtol=1e-9)
            np.testing.assert_allclose(x, [-74.0, -73.0, -72.0], rtol=1e-9)
        finally:
            path.unlink()

    @needs_gpu
    def test_polygon_way_coordinates(self):
        """Closed Way produces Polygon with correct ring coordinates."""
        from vibespatial.io.osm_gpu import read_osm_pbf

        # Triangle: nodes 1, 2, 3 at known positions, closed ring [1, 2, 3, 1]
        # Node 1: lat=0.0, lon=0.0
        # Node 2: lat=0.0, lon=1.0
        # Node 3: lat=1.0, lon=0.0
        node_id_deltas = [1, 1, 1]
        node_lat_deltas = [0, 0, 10000000]
        node_lon_deltas = [0, 10000000, 0]

        ways = [(42, [1, 2, 3, 1])]

        pbf = _build_test_pbf_with_ways(
            node_id_deltas, node_lat_deltas, node_lon_deltas, ways,
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf(path)
            assert result.n_ways == 1

            ds = result.ways.device_state
            poly_buf = ds.families[GeometryFamily.POLYGON]
            x = cp.asnumpy(poly_buf.x)
            y = cp.asnumpy(poly_buf.y)

            # 4 coordinates (closed ring: first == last)
            assert len(x) == 4
            np.testing.assert_allclose(x[0], x[3], atol=1e-12)
            np.testing.assert_allclose(y[0], y[3], atol=1e-12)

            # ring_offsets should be [0, 4]
            ring_offsets = cp.asnumpy(poly_buf.ring_offsets)
            np.testing.assert_array_equal(ring_offsets, [0, 4])

            # geometry_offsets (1 ring per polygon): [0, 1]
            geom_offsets = cp.asnumpy(poly_buf.geometry_offsets)
            np.testing.assert_array_equal(geom_offsets, [0, 1])
        finally:
            path.unlink()

    @needs_gpu
    def test_nodes_only_no_ways(self):
        """PBF with only nodes produces None for ways."""
        from vibespatial.io.osm_gpu import read_osm_pbf

        pbf = _build_test_pbf(
            id_deltas=[1, 1],
            lat_deltas=[100, 200],
            lon_deltas=[300, 400],
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf(path)
            assert result.n_nodes == 2
            assert result.nodes is not None
            assert result.n_ways == 0
            assert result.ways is None
            assert result.way_ids is None
        finally:
            path.unlink()

    @needs_gpu
    def test_file_not_found_read_osm_pbf(self):
        """read_osm_pbf raises FileNotFoundError for missing files."""
        from vibespatial.io.osm_gpu import read_osm_pbf

        with pytest.raises(FileNotFoundError):
            read_osm_pbf("/nonexistent/path.osm.pbf")

    @needs_gpu
    def test_way_ids_on_device(self):
        """Way IDs are device-resident int64 arrays."""
        from vibespatial.io.osm_gpu import read_osm_pbf

        node_id_deltas = [1, 1, 1]
        node_lat_deltas = [0, 10000, 20000]
        node_lon_deltas = [0, 10000, 20000]

        ways = [
            (777, [1, 2, 3]),
            (888, [1, 3, 2, 1]),
        ]

        pbf = _build_test_pbf_with_ways(
            node_id_deltas, node_lat_deltas, node_lon_deltas, ways,
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf(path)
            assert result.way_ids is not None

            way_ids = cp.asnumpy(result.way_ids)
            # In mixed mode, LineStrings come first then Polygons
            # Way 777 is open (LineString), Way 888 is closed (Polygon)
            assert 777 in way_ids
            assert 888 in way_ids
        finally:
            path.unlink()


# ---------------------------------------------------------------------------
# Relation protobuf building helpers
# ---------------------------------------------------------------------------


def _build_relation(
    relation_id: int,
    members: list[tuple[int, int, int]],  # (absolute_id, member_type, role_sid)
    keys: list[int] | None = None,
    vals: list[int] | None = None,
) -> bytes:
    """Build a Relation protobuf message.

    Parameters
    ----------
    relation_id
        The Relation ID.
    members
        List of (absolute_member_id, member_type, role_sid) tuples.
        member_type: 0=NODE, 1=WAY, 2=RELATION
        role_sid: stringtable index for the role string
    keys, vals
        Optional stringtable indices for tags.
    """
    result = _encode_varint_field(1, relation_id)  # field 1: id
    if keys:
        result += _encode_length_delimited(2, _encode_packed_uint32(keys))
    if vals:
        result += _encode_length_delimited(3, _encode_packed_uint32(vals))

    # Build delta-encoded member IDs (sint64)
    if members:
        # roles_sid (field 8): packed uint32
        roles = [m[2] for m in members]
        result += _encode_length_delimited(8, _encode_packed_uint32(roles))

        # memids (field 9): packed sint64, delta-encoded
        absolute_ids = [m[0] for m in members]
        deltas = [absolute_ids[0]]
        for i in range(1, len(absolute_ids)):
            deltas.append(absolute_ids[i] - absolute_ids[i - 1])
        result += _encode_length_delimited(9, _encode_packed_sint64(deltas))

        # types (field 10): packed int32
        types = [m[1] for m in members]
        result += _encode_length_delimited(10, _encode_packed_uint32(types))

    return result


def _build_primitive_group_with_relations(relations_bytes: list[bytes]) -> bytes:
    """Build a PrimitiveGroup containing Relation messages."""
    result = b""
    for rel_data in relations_bytes:
        result += _encode_length_delimited(4, rel_data)  # field 4: relations
    return result


def _build_test_pbf_with_relations(
    node_id_deltas: list[int],
    node_lat_deltas: list[int],
    node_lon_deltas: list[int],
    ways: list[tuple[int, list[int]]],
    relations: list[tuple[int, list[tuple[int, int, int]]]],
    stringtable_entries: list[bytes] | None = None,
    granularity: int = 100,
    compress: bool = True,
) -> bytes:
    """Build a PBF file with DenseNodes, Ways, and Relations.

    Parameters
    ----------
    ways
        List of (way_id, absolute_refs) tuples.
    relations
        List of (relation_id, [(member_id, member_type, role_sid), ...]) tuples.
    stringtable_entries
        The string table entries for the block. Must include role strings
        (e.g., b"outer", b"inner") at the indices referenced by role_sid.
    """
    header = _build_osm_header()

    # Build DenseNodes group
    dense = _build_dense_nodes(node_id_deltas, node_lat_deltas, node_lon_deltas)
    dense_group = _build_primitive_group(dense)

    # Build Ways group: convert absolute refs to delta-encoded
    way_messages = []
    for way_id, absolute_refs in ways:
        ref_deltas = [absolute_refs[0]]
        for i in range(1, len(absolute_refs)):
            ref_deltas.append(absolute_refs[i] - absolute_refs[i - 1])
        way_messages.append(_build_way(way_id, ref_deltas))
    ways_group = _build_primitive_group_with_ways(way_messages)

    # Build Relations group
    relation_messages = []
    for rel_id, members in relations:
        relation_messages.append(_build_relation(rel_id, members))
    relations_group = _build_primitive_group_with_relations(relation_messages)

    primitive_block = _build_primitive_block_with_stringtable(
        [dense_group, ways_group, relations_group],
        stringtable_entries=stringtable_entries,
        granularity=granularity,
    )
    data_block = _build_pbf_block("OSMData", primitive_block, compress=compress)

    return header + data_block


# ---------------------------------------------------------------------------
# Tests: CPU Relation field extraction
# ---------------------------------------------------------------------------


class TestRelationExtraction:
    """Test CPU-side Relation field extraction from PrimitiveBlocks."""

    def test_single_relation_extraction(self):
        """Extract a single Relation with Way members."""
        from vibespatial.io.osm_gpu import _extract_relation_blocks

        # stringtable: [b"", b"outer", b"inner", b"type", b"multipolygon"]
        st = [b"", b"outer", b"inner", b"type", b"multipolygon"]

        # Relation 5000 with two Way members (outer and inner)
        rel = _build_relation(5000, [
            (100, 1, 1),  # Way 100, role "outer" (st index 1)
            (200, 1, 2),  # Way 200, role "inner" (st index 2)
        ])
        group = _build_primitive_group_with_relations([rel])
        pblock = _build_primitive_block_with_stringtable(
            [group],
            stringtable_entries=st,
        )

        results = _extract_relation_blocks([pblock])
        assert len(results) == 1
        rb = results[0]
        assert rb.relation_ids == [5000]
        assert len(rb.members_per_relation) == 1
        members = rb.members_per_relation[0]
        assert len(members) == 2
        assert members[0].member_id == 100
        assert members[0].member_type == 1  # WAY
        assert members[0].role == "outer"
        assert members[1].member_id == 200
        assert members[1].member_type == 1  # WAY
        assert members[1].role == "inner"

    def test_multiple_relations(self):
        """Extract multiple Relations from a single block."""
        from vibespatial.io.osm_gpu import _extract_relation_blocks

        st = [b"", b"outer", b"inner"]

        rel1 = _build_relation(1000, [(10, 1, 1)])
        rel2 = _build_relation(2000, [(20, 1, 1), (30, 1, 2)])
        group = _build_primitive_group_with_relations([rel1, rel2])
        pblock = _build_primitive_block_with_stringtable(
            [group],
            stringtable_entries=st,
        )

        results = _extract_relation_blocks([pblock])
        assert len(results) == 1
        rb = results[0]
        assert rb.relation_ids == [1000, 2000]
        assert len(rb.members_per_relation[0]) == 1
        assert len(rb.members_per_relation[1]) == 2

    def test_delta_encoded_memids(self):
        """Verify delta decoding of member IDs."""
        from vibespatial.io.osm_gpu import _extract_relation_blocks

        st = [b"", b"outer"]

        # Members: Way 100, Way 300, Way 250 (delta: 100, 200, -50)
        rel = _build_relation(42, [
            (100, 1, 1),
            (300, 1, 1),
            (250, 1, 1),
        ])
        group = _build_primitive_group_with_relations([rel])
        pblock = _build_primitive_block_with_stringtable(
            [group],
            stringtable_entries=st,
        )

        results = _extract_relation_blocks([pblock])
        rb = results[0]
        members = rb.members_per_relation[0]
        assert members[0].member_id == 100
        assert members[1].member_id == 300
        assert members[2].member_id == 250

    def test_no_relations_in_block(self):
        """Block with only DenseNodes produces no RelationBlocks."""
        from vibespatial.io.osm_gpu import _extract_relation_blocks

        dense = _build_dense_nodes([1, 1], [100, 200], [300, 400])
        group = _build_primitive_group(dense)
        pblock = _build_primitive_block(group)

        results = _extract_relation_blocks([pblock])
        assert results == []


# ---------------------------------------------------------------------------
# Tests: Way chaining for MultiPolygon assembly
# ---------------------------------------------------------------------------


class TestWayChaining:
    """Test CPU-side Way chaining into closed rings."""

    def test_single_closed_way(self):
        """A single closed Way produces one ring."""
        from vibespatial.io.osm_gpu import _chain_ways_to_rings

        # Already closed
        rings = _chain_ways_to_rings([[1, 2, 3, 4, 1]])
        assert len(rings) == 1
        assert rings[0][0] == rings[0][-1]
        assert rings[0] == [1, 2, 3, 4, 1]

    def test_two_ways_forming_one_ring(self):
        """Two open Ways whose endpoints match form one closed ring."""
        from vibespatial.io.osm_gpu import _chain_ways_to_rings

        # Way A: [1, 2, 3], Way B: [3, 4, 1]
        rings = _chain_ways_to_rings([[1, 2, 3], [3, 4, 1]])
        assert len(rings) == 1
        assert rings[0][0] == rings[0][-1]
        assert len(rings[0]) == 5  # [1, 2, 3, 4, 1]

    def test_reversed_way_direction(self):
        """Two Ways that need direction reversal to form a ring."""
        from vibespatial.io.osm_gpu import _chain_ways_to_rings

        # Way A: [1, 2, 3], Way B: [1, 4, 3] (reversed: [3, 4, 1])
        rings = _chain_ways_to_rings([[1, 2, 3], [1, 4, 3]])
        assert len(rings) == 1
        assert rings[0][0] == rings[0][-1]

    def test_three_ways_forming_one_ring(self):
        """Three open Ways chain into one closed ring."""
        from vibespatial.io.osm_gpu import _chain_ways_to_rings

        # Way A: [1, 2], Way B: [2, 3], Way C: [3, 1]
        rings = _chain_ways_to_rings([[1, 2], [2, 3], [3, 1]])
        assert len(rings) == 1
        assert rings[0][0] == rings[0][-1]
        assert len(rings[0]) == 4  # [1, 2, 3, 1]

    def test_two_separate_closed_rings(self):
        """Two separate closed Ways produce two rings."""
        from vibespatial.io.osm_gpu import _chain_ways_to_rings

        rings = _chain_ways_to_rings([
            [1, 2, 3, 1],
            [10, 20, 30, 10],
        ])
        assert len(rings) == 2
        assert rings[0][0] == rings[0][-1]
        assert rings[1][0] == rings[1][-1]

    def test_empty_input(self):
        """Empty input produces no rings."""
        from vibespatial.io.osm_gpu import _chain_ways_to_rings

        assert _chain_ways_to_rings([]) == []

    def test_multiple_outer_rings_from_open_ways(self):
        """Multiple disjoint open Way groups produce multiple rings."""
        from vibespatial.io.osm_gpu import _chain_ways_to_rings

        # Group 1: [1, 2] + [2, 1] -> ring [1, 2, 1]
        # Group 2: [10, 20] + [20, 10] -> ring [10, 20, 10]
        # But these need 4 vertices minimum for valid ring; let's use bigger
        # Group 1: [1, 2, 3] + [3, 4, 1] -> ring [1, 2, 3, 4, 1]
        # Group 2: [10, 20, 30] + [30, 40, 10] -> ring [10, 20, 30, 40, 10]
        rings = _chain_ways_to_rings([
            [1, 2, 3], [3, 4, 1],
            [10, 20, 30], [30, 40, 10],
        ])
        assert len(rings) == 2
        for ring in rings:
            assert ring[0] == ring[-1]
            assert len(ring) >= 4


# ---------------------------------------------------------------------------
# Tests: GPU Relation MultiPolygon assembly
# ---------------------------------------------------------------------------


class TestRelationMultiPolygon:
    """Test GPU-side MultiPolygon assembly from Relations."""

    @needs_gpu
    def test_simple_multipolygon_one_outer(self):
        """A relation with one closed outer Way produces a single-part MultiPolygon."""
        from vibespatial.io.osm_gpu import read_osm_pbf

        # 4 nodes forming a square
        # Node 1: lat=0, lon=0; Node 2: lat=0, lon=1
        # Node 3: lat=1, lon=1; Node 4: lat=1, lon=0
        node_id_deltas = [1, 1, 1, 1]
        node_lat_deltas = [0, 0, 10000000, 10000000]
        node_lon_deltas = [0, 10000000, 10000000, 0]

        # Way 100: closed ring [1, 2, 3, 4, 1]
        ways = [(100, [1, 2, 3, 4, 1])]

        # Relation 5000: Way 100 as outer
        # stringtable: [b"", b"outer", b"inner", b"type", b"multipolygon"]
        st = [b"", b"outer", b"inner", b"type", b"multipolygon"]
        relations = [(5000, [(100, 1, 1)])]  # Way 100, type=WAY(1), role_sid=1("outer")

        pbf = _build_test_pbf_with_relations(
            node_id_deltas, node_lat_deltas, node_lon_deltas,
            ways, relations, stringtable_entries=st,
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf(path)
            assert result.n_relations == 1
            assert result.relations is not None
            assert result.relation_ids is not None

            rel_ids = cp.asnumpy(result.relation_ids)
            assert rel_ids[0] == 5000

            ds = result.relations.device_state
            assert ds is not None
            assert GeometryFamily.MULTIPOLYGON in ds.families

            mpoly_buf = ds.families[GeometryFamily.MULTIPOLYGON]

            # geometry_offsets: [0, 1] (one polygon part)
            geom_offsets = cp.asnumpy(mpoly_buf.geometry_offsets)
            assert geom_offsets[0] == 0
            assert geom_offsets[1] == 1

            # part_offsets: [0, 1] (one ring in the one part)
            part_offsets = cp.asnumpy(mpoly_buf.part_offsets)
            assert part_offsets[0] == 0
            assert part_offsets[1] == 1

            # ring_offsets: [0, 5] (5 coordinates: 4 vertices + closing point)
            ring_offsets = cp.asnumpy(mpoly_buf.ring_offsets)
            assert ring_offsets[0] == 0
            assert ring_offsets[1] == 5

            # Verify coordinates
            x = cp.asnumpy(mpoly_buf.x)
            y = cp.asnumpy(mpoly_buf.y)
            assert len(x) == 5
            # First and last should match (closed ring)
            np.testing.assert_allclose(x[0], x[4], atol=1e-12)
            np.testing.assert_allclose(y[0], y[4], atol=1e-12)
        finally:
            path.unlink()

    @needs_gpu
    def test_multipolygon_with_inner_ring(self):
        """A relation with outer + inner Way produces a polygon with hole."""
        from vibespatial.io.osm_gpu import read_osm_pbf

        # Outer: large square, nodes 1-4
        # Inner: small square, nodes 5-8
        node_id_deltas = [1, 1, 1, 1, 1, 1, 1, 1]
        node_lat_deltas = [0, 0, 100000000, 100000000,
                           10000000, 10000000, 30000000, 30000000]
        node_lon_deltas = [0, 100000000, 100000000, 0,
                           10000000, 30000000, 30000000, 10000000]

        ways = [
            (100, [1, 2, 3, 4, 1]),  # outer ring
            (200, [5, 6, 7, 8, 5]),  # inner ring (hole)
        ]

        st = [b"", b"outer", b"inner"]
        relations = [(9000, [
            (100, 1, 1),  # Way 100 as outer
            (200, 1, 2),  # Way 200 as inner
        ])]

        pbf = _build_test_pbf_with_relations(
            node_id_deltas, node_lat_deltas, node_lon_deltas,
            ways, relations, stringtable_entries=st,
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf(path)
            assert result.n_relations == 1

            ds = result.relations.device_state
            mpoly_buf = ds.families[GeometryFamily.MULTIPOLYGON]

            # 1 geometry, 1 part (polygon), 2 rings (outer + inner)
            geom_offsets = cp.asnumpy(mpoly_buf.geometry_offsets)
            part_offsets = cp.asnumpy(mpoly_buf.part_offsets)
            ring_offsets = cp.asnumpy(mpoly_buf.ring_offsets)

            assert geom_offsets[-1] == 1  # 1 polygon part
            assert part_offsets[-1] == 2  # 2 rings total
            assert len(ring_offsets) == 3  # [0, 5, 10]

            # Each ring has 5 coordinates
            assert ring_offsets[1] - ring_offsets[0] == 5
            assert ring_offsets[2] - ring_offsets[1] == 5
        finally:
            path.unlink()

    @needs_gpu
    def test_multipolygon_two_outer_rings(self):
        """A relation with two outer Ways produces a 2-part MultiPolygon."""
        from vibespatial.io.osm_gpu import read_osm_pbf

        # Two separate triangles
        node_id_deltas = [1, 1, 1, 1, 1, 1]
        # Triangle 1: (0,0), (0,1), (1,0)
        # Triangle 2: (10,10), (10,11), (11,10)
        node_lat_deltas = [0, 0, 10000000,
                           90000000, 0, 10000000]
        node_lon_deltas = [0, 10000000, 0,
                           90000000, 10000000, 0]

        ways = [
            (100, [1, 2, 3, 1]),  # triangle 1
            (200, [4, 5, 6, 4]),  # triangle 2
        ]

        st = [b"", b"outer"]
        relations = [(7000, [
            (100, 1, 1),  # outer
            (200, 1, 1),  # outer
        ])]

        pbf = _build_test_pbf_with_relations(
            node_id_deltas, node_lat_deltas, node_lon_deltas,
            ways, relations, stringtable_entries=st,
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf(path)
            assert result.n_relations == 1

            ds = result.relations.device_state
            mpoly_buf = ds.families[GeometryFamily.MULTIPOLYGON]

            geom_offsets = cp.asnumpy(mpoly_buf.geometry_offsets)
            part_offsets = cp.asnumpy(mpoly_buf.part_offsets)

            # 2 polygon parts
            assert geom_offsets[-1] == 2

            # Each part has 1 ring
            ring_count_part1 = part_offsets[1] - part_offsets[0]
            ring_count_part2 = part_offsets[2] - part_offsets[1]
            assert ring_count_part1 == 1
            assert ring_count_part2 == 1
        finally:
            path.unlink()

    @needs_gpu
    def test_multipolygon_chained_ways(self):
        """A relation whose outer ring is split across two Ways."""
        from vibespatial.io.osm_gpu import read_osm_pbf

        # 4 nodes forming a square
        node_id_deltas = [1, 1, 1, 1]
        node_lat_deltas = [0, 0, 10000000, 10000000]
        node_lon_deltas = [0, 10000000, 10000000, 0]

        # Two open ways that together form a closed ring:
        # Way 100: [1, 2, 3] (half the square)
        # Way 200: [3, 4, 1] (other half)
        ways = [
            (100, [1, 2, 3]),
            (200, [3, 4, 1]),
        ]

        st = [b"", b"outer"]
        relations = [(8000, [
            (100, 1, 1),  # outer
            (200, 1, 1),  # outer
        ])]

        pbf = _build_test_pbf_with_relations(
            node_id_deltas, node_lat_deltas, node_lon_deltas,
            ways, relations, stringtable_entries=st,
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf(path)
            assert result.n_relations == 1

            ds = result.relations.device_state
            mpoly_buf = ds.families[GeometryFamily.MULTIPOLYGON]

            # Should have chained into one ring with 5 coordinates
            ring_offsets = cp.asnumpy(mpoly_buf.ring_offsets)
            total_coords = ring_offsets[-1]
            assert total_coords == 5  # [1, 2, 3, 4, 1]

            # Verify the ring is closed
            x = cp.asnumpy(mpoly_buf.x)
            y = cp.asnumpy(mpoly_buf.y)
            np.testing.assert_allclose(x[0], x[4], atol=1e-12)
            np.testing.assert_allclose(y[0], y[4], atol=1e-12)
        finally:
            path.unlink()

    @needs_gpu
    def test_multipolygon_reversed_way(self):
        """A relation whose outer ring needs Way direction reversal."""
        from vibespatial.io.osm_gpu import read_osm_pbf

        # 4 nodes forming a square
        node_id_deltas = [1, 1, 1, 1]
        node_lat_deltas = [0, 0, 10000000, 10000000]
        node_lon_deltas = [0, 10000000, 10000000, 0]

        # Way 100: [1, 2, 3] (forward)
        # Way 200: [1, 4, 3] (reversed -- last ref matches Way 100 last ref)
        ways = [
            (100, [1, 2, 3]),
            (200, [1, 4, 3]),
        ]

        st = [b"", b"outer"]
        relations = [(8001, [
            (100, 1, 1),
            (200, 1, 1),
        ])]

        pbf = _build_test_pbf_with_relations(
            node_id_deltas, node_lat_deltas, node_lon_deltas,
            ways, relations, stringtable_entries=st,
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf(path)
            assert result.n_relations == 1

            ds = result.relations.device_state
            mpoly_buf = ds.families[GeometryFamily.MULTIPOLYGON]

            # Should have 5 coordinates (closed ring)
            ring_offsets = cp.asnumpy(mpoly_buf.ring_offsets)
            total_coords = ring_offsets[-1]
            assert total_coords == 5

            x = cp.asnumpy(mpoly_buf.x)
            y = cp.asnumpy(mpoly_buf.y)
            np.testing.assert_allclose(x[0], x[4], atol=1e-12)
            np.testing.assert_allclose(y[0], y[4], atol=1e-12)
        finally:
            path.unlink()

    @needs_gpu
    def test_multipolygon_coordinate_correctness(self):
        """Verify exact coordinates in a MultiPolygon from a Relation."""
        from vibespatial.io.osm_gpu import read_osm_pbf

        # Node 1: lat=40.0, lon=-74.0  (nanodeg: 400000000, -740000000)
        # Node 2: lat=40.0, lon=-73.0  (nanodeg delta: 0, 10000000)
        # Node 3: lat=41.0, lon=-73.0  (nanodeg delta: 10000000, 0)
        node_id_deltas = [1, 1, 1]
        node_lat_deltas = [400000000, 0, 10000000]
        node_lon_deltas = [-740000000, 10000000, 0]

        ways = [(100, [1, 2, 3, 1])]

        st = [b"", b"outer"]
        relations = [(6000, [(100, 1, 1)])]

        pbf = _build_test_pbf_with_relations(
            node_id_deltas, node_lat_deltas, node_lon_deltas,
            ways, relations, stringtable_entries=st,
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf(path)
            assert result.n_relations == 1

            ds = result.relations.device_state
            mpoly_buf = ds.families[GeometryFamily.MULTIPOLYGON]

            x = cp.asnumpy(mpoly_buf.x)  # longitude
            y = cp.asnumpy(mpoly_buf.y)  # latitude

            np.testing.assert_allclose(y, [40.0, 40.0, 41.0, 40.0], rtol=1e-9)
            np.testing.assert_allclose(x, [-74.0, -73.0, -73.0, -74.0], rtol=1e-9)
        finally:
            path.unlink()

    @needs_gpu
    def test_empty_role_defaults_to_outer(self):
        """Members with empty role ('') are treated as outer."""
        from vibespatial.io.osm_gpu import read_osm_pbf

        node_id_deltas = [1, 1, 1, 1]
        node_lat_deltas = [0, 0, 10000000, 10000000]
        node_lon_deltas = [0, 10000000, 10000000, 0]

        ways = [(100, [1, 2, 3, 4, 1])]

        # stringtable: [b""] -- index 0 is empty string
        st = [b""]
        # role_sid=0 maps to "" which defaults to "outer"
        relations = [(5500, [(100, 1, 0)])]

        pbf = _build_test_pbf_with_relations(
            node_id_deltas, node_lat_deltas, node_lon_deltas,
            ways, relations, stringtable_entries=st,
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf(path)
            assert result.n_relations == 1
        finally:
            path.unlink()

    @needs_gpu
    def test_missing_way_member_skipped(self):
        """Relation referencing a Way not in the dataset skips that member."""
        from vibespatial.io.osm_gpu import read_osm_pbf

        node_id_deltas = [1, 1, 1, 1]
        node_lat_deltas = [0, 0, 10000000, 10000000]
        node_lon_deltas = [0, 10000000, 10000000, 0]

        # Only Way 100 exists, not Way 999
        ways = [(100, [1, 2, 3, 4, 1])]

        st = [b"", b"outer"]
        # Relation references Way 100 (exists) and Way 999 (missing)
        relations = [(5001, [
            (100, 1, 1),
            (999, 1, 1),  # missing Way
        ])]

        pbf = _build_test_pbf_with_relations(
            node_id_deltas, node_lat_deltas, node_lon_deltas,
            ways, relations, stringtable_entries=st,
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf(path)
            # Should still produce a valid relation from Way 100
            assert result.n_relations == 1
        finally:
            path.unlink()

    @needs_gpu
    def test_relation_referencing_missing_relation_still_resolves(self):
        """A parent relation with an unresolvable child still resolves from its Way members."""
        from vibespatial.io.osm_gpu import read_osm_pbf

        node_id_deltas = [1, 1, 1, 1]
        node_lat_deltas = [0, 0, 10000000, 10000000]
        node_lon_deltas = [0, 10000000, 10000000, 0]

        ways = [(100, [1, 2, 3, 4, 1])]

        st = [b"", b"outer"]
        # Way 100 outer + Relation 9999 (not in dataset -- gracefully skipped)
        relations = [(5002, [
            (100, 1, 1),    # Way 100, outer
            (9999, 2, 1),   # Relation 9999, not found -- skipped
        ])]

        pbf = _build_test_pbf_with_relations(
            node_id_deltas, node_lat_deltas, node_lon_deltas,
            ways, relations, stringtable_entries=st,
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf(path)
            assert result.n_relations == 1
        finally:
            path.unlink()

    @needs_gpu
    def test_recursive_relation_resolved(self):
        """A parent relation referencing a child relation merges child's rings."""
        from vibespatial.io.osm_gpu import read_osm_pbf

        # 8 nodes forming two separate squares:
        # Square 1 (child): nodes 1-4 at (0,0),(1,0),(1,1),(0,1)
        # Square 2 (parent's own): nodes 5-8 at (2,0),(3,0),(3,1),(2,1)
        node_id_deltas = [1, 1, 1, 1, 1, 1, 1, 1]
        # lat in nanodegrees/granularity: 0, 0, 10M, 10M, 0, 0, 10M, 10M
        node_lat_deltas = [0, 0, 10000000, 0, -10000000, 0, 10000000, 0]
        # lon: 0, 10M, 0, -10M, 10M, 10M, 0, -10M  (= 0,1,1,0, 2,3,3,2)
        node_lon_deltas = [0, 10000000, 0, -10000000, 20000000, 10000000, 0, -10000000]

        # Way 100 = closed ring for square 1: nodes 1->2->3->4->1
        # Way 200 = closed ring for square 2: nodes 5->6->7->8->5
        ways = [
            (100, [1, 2, 3, 4, 1]),
            (200, [5, 6, 7, 8, 5]),
        ]

        st = [b"", b"outer"]
        # Child relation 5001: outer = Way 100 (square 1)
        # Parent relation 5002: outer = Way 200 (square 2) + outer = Relation 5001
        relations = [
            (5001, [(100, 1, 1)]),              # child: Way 100 as outer
            (5002, [(200, 1, 1), (5001, 2, 1)]),  # parent: Way 200 + child Relation 5001
        ]

        pbf = _build_test_pbf_with_relations(
            node_id_deltas, node_lat_deltas, node_lon_deltas,
            ways, relations, stringtable_entries=st,
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf(path)
            # Both relations should resolve
            assert result.n_relations == 2

            # Parent relation (5002) should be a MultiPolygon with 2 outer parts
            # (one from Way 200, one from child Relation 5001's Way 100)
            rel_ids = cp.asnumpy(result.relation_ids)
            # Check both relation IDs are present
            assert 5001 in rel_ids
            assert 5002 in rel_ids
        finally:
            path.unlink()


# ---------------------------------------------------------------------------
# Tests: End-to-end PBF with nodes + ways + relations
# ---------------------------------------------------------------------------


class TestEndToEndWithRelations:
    """End-to-end test: PBF with all three element types."""

    @needs_gpu
    def test_nodes_ways_relations(self):
        """read_osm_pbf returns nodes, ways, and relations."""
        from vibespatial.io.osm_gpu import read_osm_pbf

        # 4 nodes forming a square + 2 extra for a linestring
        node_id_deltas = [1, 1, 1, 1, 1, 1]
        node_lat_deltas = [0, 10000, 10000, 10000, 10000, 10000]
        node_lon_deltas = [0, 10000, 10000, 10000, 10000, 10000]

        # Way 100: closed ring -> Polygon when standalone
        # Way 200: open linestring [5, 6]
        ways = [
            (100, [1, 2, 3, 4, 1]),
            (200, [5, 6]),
        ]

        st = [b"", b"outer"]
        # Relation 9000: Way 100 as outer
        relations = [(9000, [(100, 1, 1)])]

        pbf = _build_test_pbf_with_relations(
            node_id_deltas, node_lat_deltas, node_lon_deltas,
            ways, relations, stringtable_entries=st,
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf(path)

            # Should have nodes, ways, and relations
            assert result.n_nodes == 6
            assert result.nodes is not None

            assert result.n_ways == 2
            assert result.ways is not None

            assert result.n_relations == 1
            assert result.relations is not None
            assert result.relation_ids is not None

            rel_ids = cp.asnumpy(result.relation_ids)
            assert rel_ids[0] == 9000
        finally:
            path.unlink()

    @needs_gpu
    def test_no_relations_returns_zero(self):
        """PBF without relations has n_relations=0."""
        from vibespatial.io.osm_gpu import read_osm_pbf

        pbf = _build_test_pbf(
            id_deltas=[1, 1],
            lat_deltas=[100, 200],
            lon_deltas=[300, 400],
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf(path)
            assert result.n_relations == 0
            assert result.relations is None
            assert result.relation_ids is None
        finally:
            path.unlink()


# ---------------------------------------------------------------------------
# Multi-block PBF builder helpers (for streaming pipeline tests)
# ---------------------------------------------------------------------------


def _build_multi_block_pbf(
    blocks_data: list[tuple[list[int], list[int], list[int], int]],
    compress: bool = True,
) -> bytes:
    """Build a PBF file with multiple OSMData blocks.

    Each entry in *blocks_data* is
    ``(id_deltas, lat_deltas, lon_deltas, granularity)``.
    Each produces a separate OSMData block in the file, so the streaming
    pipeline must process them in separate batches.
    """
    header = _build_osm_header()
    result = header

    for id_deltas, lat_deltas, lon_deltas, granularity in blocks_data:
        dense = _build_dense_nodes(id_deltas, lat_deltas, lon_deltas)
        group = _build_primitive_group(dense)
        pblock = _build_primitive_block(group, granularity=granularity)
        data_block = _build_pbf_block("OSMData", pblock, compress=compress)
        result += data_block

    return result


def _build_multi_block_pbf_with_ways(
    node_blocks: list[tuple[list[int], list[int], list[int]]],
    way_block_data: list[tuple[list[tuple[int, list[int]]], list[bytes] | None]],
    granularity: int = 100,
    compress: bool = True,
) -> bytes:
    """Build a PBF file with multiple OSMData blocks containing nodes and ways.

    *node_blocks*: list of (id_deltas, lat_deltas, lon_deltas) per block.
    *way_block_data*: list of (ways, stringtable_entries) per block.
      Each ``ways`` is a list of (way_id, absolute_refs) tuples.
    Nodes and ways are interleaved into separate blocks.
    """
    header = _build_osm_header()
    result = header

    # Emit node-only blocks first
    for id_deltas, lat_deltas, lon_deltas in node_blocks:
        dense = _build_dense_nodes(id_deltas, lat_deltas, lon_deltas)
        group = _build_primitive_group(dense)
        pblock = _build_primitive_block(group, granularity=granularity)
        data_block = _build_pbf_block("OSMData", pblock, compress=compress)
        result += data_block

    # Emit way-only blocks
    for ways, st_entries in way_block_data:
        way_messages = []
        for way_id, absolute_refs in ways:
            ref_deltas = [absolute_refs[0]]
            for i in range(1, len(absolute_refs)):
                ref_deltas.append(absolute_refs[i] - absolute_refs[i - 1])
            way_messages.append(_build_way(way_id, ref_deltas))
        ways_group = _build_primitive_group_with_ways(way_messages)
        pblock = _build_primitive_block_with_stringtable(
            [ways_group],
            stringtable_entries=st_entries,
            granularity=granularity,
        )
        data_block = _build_pbf_block("OSMData", pblock, compress=compress)
        result += data_block

    return result


# ---------------------------------------------------------------------------
# Tests: Streaming block pipeline
# ---------------------------------------------------------------------------


class TestStreamingPipeline:
    """Tests for the streaming block decode pipeline.

    Verifies that processing blocks in small batches produces identical
    results to processing them all at once, and that the pipeline
    handles multi-block PBF files correctly.
    """

    @needs_gpu
    def test_multi_block_nodes_match_single_block(self):
        """Multiple blocks produce the same result as one combined block."""
        from vibespatial.io.osm_gpu import read_osm_pbf_nodes

        # Block 1: nodes 100, 101 at (40.0, -74.0), (40.1, -74.01)
        # Block 2: nodes 200, 201 at (50.0, -60.0), (50.2, -59.97)
        multi_pbf = _build_multi_block_pbf([
            ([100, 1], [400000000, 1000000], [-740000000, -100000], 100),
            ([200, 1], [500000000, 2000000], [-600000000, 300000], 100),
        ])
        multi_path = _write_temp_pbf(multi_pbf)

        try:
            result = read_osm_pbf_nodes(multi_path)

            assert result.n_nodes == 4

            ids = cp.asnumpy(result.node_ids)
            np.testing.assert_array_equal(ids, [100, 101, 200, 201])

            ds = result.nodes.device_state
            pt = ds.families[GeometryFamily.POINT]
            y = cp.asnumpy(pt.y)  # latitude
            x = cp.asnumpy(pt.x)  # longitude

            # Block 1 coordinates
            np.testing.assert_allclose(y[:2], [40.0, 40.1], rtol=1e-10)
            np.testing.assert_allclose(x[:2], [-74.0, -74.01], rtol=1e-10)

            # Block 2 coordinates (independent delta reset)
            np.testing.assert_allclose(y[2:], [50.0, 50.2], rtol=1e-10)
            np.testing.assert_allclose(x[2:], [-60.0, -59.97], rtol=1e-10)
        finally:
            multi_path.unlink()

    @needs_gpu
    def test_three_blocks_with_varying_sizes(self):
        """Three blocks with 1, 2, and 3 nodes respectively."""
        from vibespatial.io.osm_gpu import read_osm_pbf_nodes

        multi_pbf = _build_multi_block_pbf([
            # Block 1: 1 node (id=10, lat=1.0, lon=2.0)
            ([10], [10000000], [20000000], 100),
            # Block 2: 2 nodes (id=20, 21, lat=3.0, 3.1, lon=4.0, 4.1)
            ([20, 1], [30000000, 1000000], [40000000, 1000000], 100),
            # Block 3: 3 nodes (id=30, 31, 32)
            ([30, 1, 1], [50000000, 100000, 200000], [60000000, 100000, 200000], 100),
        ])
        path = _write_temp_pbf(multi_pbf)

        try:
            result = read_osm_pbf_nodes(path)
            assert result.n_nodes == 6

            ids = cp.asnumpy(result.node_ids)
            np.testing.assert_array_equal(ids, [10, 20, 21, 30, 31, 32])
        finally:
            path.unlink()

    @needs_gpu
    def test_single_block_unchanged(self):
        """Single-block PBF is handled identically by the streaming pipeline."""
        from vibespatial.io.osm_gpu import read_osm_pbf_nodes

        pbf = _build_test_pbf(
            id_deltas=[42, 1, 1],
            lat_deltas=[400000000, 1000000, 2000000],
            lon_deltas=[-740000000, -100000, -200000],
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf_nodes(path)
            assert result.n_nodes == 3

            ids = cp.asnumpy(result.node_ids)
            np.testing.assert_array_equal(ids, [42, 43, 44])

            ds = result.nodes.device_state
            pt = ds.families[GeometryFamily.POINT]
            y = cp.asnumpy(pt.y)
            np.testing.assert_allclose(y, [40.0, 40.1, 40.3], rtol=1e-10)
        finally:
            path.unlink()

    @needs_gpu
    def test_multi_block_read_osm_pbf_with_ways(self):
        """read_osm_pbf works correctly when nodes and ways are in separate blocks."""
        from vibespatial.io.osm_gpu import read_osm_pbf

        # Build a multi-block PBF:
        # Block 1: 3 nodes (id=1,2,3 at known positions)
        # Block 2: 1 way referencing those nodes
        node_blocks = [
            ([1, 1, 1], [0, 10000000, 20000000], [0, 10000000, 20000000]),
        ]
        way_block_data = [
            ([(500, [1, 2, 3])], None),
        ]

        pbf = _build_multi_block_pbf_with_ways(node_blocks, way_block_data)
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf(path)

            assert result.n_nodes == 3
            assert result.nodes is not None

            assert result.n_ways == 1
            assert result.ways is not None
            assert result.way_ids is not None

            way_ids = cp.asnumpy(result.way_ids)
            assert way_ids[0] == 500
        finally:
            path.unlink()

    @needs_gpu
    def test_batch_size_boundary(self):
        """5 blocks exercises batch boundary (batch_size=4: 4+1 blocks)."""
        from vibespatial.io.osm_gpu import _STREAM_BATCH_SIZE, read_osm_pbf_nodes

        # Create exactly _STREAM_BATCH_SIZE + 1 blocks
        n_blocks = _STREAM_BATCH_SIZE + 1
        blocks_data = []
        expected_ids = []
        base_id = 100
        for i in range(n_blocks):
            node_id = base_id + i * 100
            expected_ids.append(node_id)
            blocks_data.append(
                ([node_id], [10000000 * (i + 1)], [20000000 * (i + 1)], 100),
            )

        multi_pbf = _build_multi_block_pbf(blocks_data)
        path = _write_temp_pbf(multi_pbf)

        try:
            result = read_osm_pbf_nodes(path)
            assert result.n_nodes == n_blocks

            ids = cp.asnumpy(result.node_ids)
            np.testing.assert_array_equal(ids, expected_ids)
        finally:
            path.unlink()

    @needs_gpu
    def test_empty_blocks_skipped(self):
        """Blocks with no DenseNodes are gracefully skipped."""
        from vibespatial.io.osm_gpu import read_osm_pbf

        # Build a PBF with:
        # Block 1: nodes only (id=1)
        # Block 2: ways only (no nodes) -- the streaming node pipeline skips it
        node_blocks = [
            ([1, 1, 1], [0, 10000000, 20000000], [0, 10000000, 20000000]),
        ]
        way_block_data = [
            ([(100, [1, 2, 3])], None),
        ]

        pbf = _build_multi_block_pbf_with_ways(node_blocks, way_block_data)
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf(path)
            assert result.n_nodes == 3
            assert result.n_ways == 1
        finally:
            path.unlink()

    @needs_gpu
    def test_multi_block_different_granularities(self):
        """Blocks with different granularities are scaled correctly."""
        from vibespatial.io.osm_gpu import read_osm_pbf_nodes

        # Block 1: granularity=100, delta=10000000 -> lat = 10000000 * 100 * 1e-9 = 1.0
        # Block 2: granularity=1000, delta=1000000 -> lat = 1000000 * 1000 * 1e-9 = 1.0
        # Both should produce lat=1.0 despite different granularities
        multi_pbf = _build_multi_block_pbf([
            ([1], [10000000], [20000000], 100),
            ([2], [1000000], [2000000], 1000),
        ])
        path = _write_temp_pbf(multi_pbf)

        try:
            result = read_osm_pbf_nodes(path)
            assert result.n_nodes == 2

            ds = result.nodes.device_state
            pt = ds.families[GeometryFamily.POINT]
            y = cp.asnumpy(pt.y)
            x = cp.asnumpy(pt.x)

            # Both blocks produce the same lat/lon
            np.testing.assert_allclose(y, [1.0, 1.0], rtol=1e-10)
            np.testing.assert_allclose(x, [2.0, 2.0], rtol=1e-10)
        finally:
            path.unlink()

    def test_read_and_decompress_batch(self):
        """_read_and_decompress_batch reads only specified blocks."""
        from vibespatial.io.osm_gpu import (
            _parse_block_index,
            _read_and_decompress_batch,
        )

        # Build a multi-block PBF with 3 data blocks
        multi_pbf = _build_multi_block_pbf([
            ([1], [100], [200], 100),
            ([2], [300], [400], 100),
            ([3], [500], [600], 100),
        ])
        path = _write_temp_pbf(multi_pbf)

        try:
            block_index = _parse_block_index(path)
            data_blocks = [bi for bi in block_index if bi.block_type == "OSMData"]
            assert len(data_blocks) == 3

            # Read only the first block
            batch1 = _read_and_decompress_batch(path, data_blocks[:1])
            assert len(batch1) == 1

            # Read blocks 2 and 3
            batch23 = _read_and_decompress_batch(path, data_blocks[1:])
            assert len(batch23) == 2

            # Each should be non-empty decompressed data
            assert all(len(b) > 0 for b in batch1)
            assert all(len(b) > 0 for b in batch23)
        finally:
            path.unlink()

    def test_stream_batch_size_is_positive(self):
        """Smoke test: _STREAM_BATCH_SIZE is a positive integer."""
        from vibespatial.io.osm_gpu import _STREAM_BATCH_SIZE

        assert isinstance(_STREAM_BATCH_SIZE, int)
        assert _STREAM_BATCH_SIZE >= 1


# ---------------------------------------------------------------------------
# Tests: Tag / attribute extraction
# ---------------------------------------------------------------------------


class TestDenseNodeTagDecoding:
    """Test DenseNodes tag extraction from the keys_vals packed field."""

    def test_decode_single_node_single_tag(self):
        """Single node with one tag pair -> one dict with one entry."""
        from vibespatial.io.osm_gpu import _decode_dense_node_tags

        # keys_vals = [key_sid=1, val_sid=2, 0] -- one node, one tag
        st = [b"", b"name", b"Main St"]
        result = _decode_dense_node_tags([1, 2, 0], st, n_nodes=1)
        assert len(result) == 1
        assert result[0] == {"name": "Main St"}

    def test_decode_multiple_nodes_with_varying_tags(self):
        """Three nodes: first has 2 tags, second has 0, third has 1."""
        from vibespatial.io.osm_gpu import _decode_dense_node_tags

        st = [b"", b"highway", b"residential", b"name", b"Oak Ave"]
        # Node 1: highway=residential, name=Oak Ave
        # Node 2: (no tags)
        # Node 3: highway=residential
        keys_vals = [1, 2, 3, 4, 0, 0, 1, 2, 0]
        result = _decode_dense_node_tags(keys_vals, st, n_nodes=3)
        assert len(result) == 3
        assert result[0] == {"highway": "residential", "name": "Oak Ave"}
        assert result[1] == {}
        assert result[2] == {"highway": "residential"}

    def test_decode_no_tags_all_zeros(self):
        """All nodes have no tags -> all zeros -> all empty dicts."""
        from vibespatial.io.osm_gpu import _decode_dense_node_tags

        # 3 nodes, no tags: [0, 0, 0]
        result = _decode_dense_node_tags([0, 0, 0], [b""], n_nodes=3)
        assert len(result) == 3
        assert all(d == {} for d in result)

    def test_decode_empty_keys_vals(self):
        """Empty keys_vals array -> n_nodes empty dicts."""
        from vibespatial.io.osm_gpu import _decode_dense_node_tags

        result = _decode_dense_node_tags([], [b""], n_nodes=5)
        assert len(result) == 5
        assert all(d == {} for d in result)

    def test_decode_unicode_tag_values(self):
        """Tags with non-ASCII characters decode correctly."""
        from vibespatial.io.osm_gpu import _decode_dense_node_tags

        st = [b"", b"name", "Caf\u00e9 R\u00f6sti".encode()]
        result = _decode_dense_node_tags([1, 2, 0], st, n_nodes=1)
        assert result[0] == {"name": "Caf\u00e9 R\u00f6sti"}


class TestWayTagDecoding:
    """Test Way tag extraction from WayBlock stringtable indices."""

    def test_decode_way_with_tags(self):
        """Way with tags -> decoded dict."""
        from vibespatial.io.osm_gpu import WayBlock, _decode_way_tags

        st = [b"", b"highway", b"building", b"residential", b"yes"]
        wb = WayBlock(
            way_ids=[100],
            refs_per_way=[[1, 2, 3]],
            tag_keys_per_way=[[1, 2]],       # highway, building
            tag_vals_per_way=[[3, 4]],        # residential, yes
            stringtable=st,
        )
        result = _decode_way_tags(wb)
        assert len(result) == 1
        assert result[0] == {"highway": "residential", "building": "yes"}

    def test_decode_way_without_tags(self):
        """Way with no tags -> empty dict."""
        from vibespatial.io.osm_gpu import WayBlock, _decode_way_tags

        wb = WayBlock(
            way_ids=[200],
            refs_per_way=[[1, 2, 3]],
            tag_keys_per_way=[[]],
            tag_vals_per_way=[[]],
            stringtable=[b""],
        )
        result = _decode_way_tags(wb)
        assert len(result) == 1
        assert result[0] == {}

    def test_decode_multiple_ways_mixed_tags(self):
        """Multiple ways: first with tags, second without."""
        from vibespatial.io.osm_gpu import WayBlock, _decode_way_tags

        st = [b"", b"highway", b"primary"]
        wb = WayBlock(
            way_ids=[100, 200],
            refs_per_way=[[1, 2], [3, 4]],
            tag_keys_per_way=[[1], []],
            tag_vals_per_way=[[2], []],
            stringtable=st,
        )
        result = _decode_way_tags(wb)
        assert len(result) == 2
        assert result[0] == {"highway": "primary"}
        assert result[1] == {}


class TestRelationTagDecoding:
    """Test Relation tag extraction from RelationBlock."""

    def test_decode_relation_with_tags(self):
        """Relation with tags -> decoded dict."""
        from vibespatial.io.osm_gpu import RelationBlock, _decode_relation_tags

        st = [b"", b"outer", b"type", b"multipolygon", b"name", b"Park"]
        rb = RelationBlock(
            relation_ids=[5000],
            members_per_relation=[[]],
            stringtable=st,
            granularity=100,
            lat_offset=0,
            lon_offset=0,
            tag_keys_per_relation=[[2, 4]],      # type, name
            tag_vals_per_relation=[[3, 5]],       # multipolygon, Park
        )
        result = _decode_relation_tags(rb)
        assert len(result) == 1
        assert result[0] == {"type": "multipolygon", "name": "Park"}

    def test_decode_relation_without_tags(self):
        """Relation with no tags -> empty dict."""
        from vibespatial.io.osm_gpu import RelationBlock, _decode_relation_tags

        rb = RelationBlock(
            relation_ids=[6000],
            members_per_relation=[[]],
            stringtable=[b""],
            granularity=100,
            lat_offset=0,
            lon_offset=0,
            tag_keys_per_relation=None,
            tag_vals_per_relation=None,
        )
        result = _decode_relation_tags(rb)
        assert len(result) == 1
        assert result[0] == {}


class TestTagsToDataframe:
    """Test the _tags_to_dataframe helper."""

    def test_empty_tags(self):
        """Empty list -> empty DataFrame."""
        from vibespatial.io.osm_gpu import _tags_to_dataframe

        df = _tags_to_dataframe([])
        assert len(df) == 0

    def test_uniform_tags(self):
        """All elements have the same keys."""
        from vibespatial.io.osm_gpu import _tags_to_dataframe

        tags = [
            {"highway": "primary", "name": "Main St"},
            {"highway": "secondary", "name": "Oak Ave"},
        ]
        df = _tags_to_dataframe(tags)
        assert len(df) == 2
        assert "highway" in df.columns
        assert "name" in df.columns
        assert df["highway"].iloc[0] == "primary"
        assert df["name"].iloc[1] == "Oak Ave"

    def test_sparse_tags(self):
        """Different elements have different keys -> NaN for missing."""
        import pandas as pd

        from vibespatial.io.osm_gpu import _tags_to_dataframe

        tags = [
            {"highway": "primary"},
            {"building": "yes"},
            {},
        ]
        df = _tags_to_dataframe(tags)
        assert len(df) == 3
        assert "highway" in df.columns
        assert "building" in df.columns
        # Missing values become NaN
        assert pd.isna(df["highway"].iloc[1])
        assert pd.isna(df["building"].iloc[0])
        assert pd.isna(df["highway"].iloc[2])


class TestDenseNodeTagsEndToEnd:
    """End-to-end: PBF with DenseNodes tags -> OsmGpuResult.node_tags."""

    @needs_gpu
    def test_nodes_with_tags(self):
        """Nodes with tags extracted via read_osm_pbf_nodes."""
        from vibespatial.io.osm_gpu import read_osm_pbf_nodes

        # stringtable: [b"", b"amenity", b"cafe", b"name", b"Joe's"]
        # Node 1: amenity=cafe, name=Joe's
        # Node 2: (no tags)
        # keys_vals: [1, 2, 3, 4, 0, 0]  (node1: key1=val2, key3=val4; node2: empty)
        keys_vals = [1, 2, 3, 4, 0, 0]

        dense = _build_dense_nodes(
            id_deltas=[1, 1],
            lat_deltas=[400000000, 1000000],
            lon_deltas=[-740000000, 100000],
            keys_vals=keys_vals,
        )
        group = _build_primitive_group(dense)
        pblock = _build_primitive_block_with_stringtable(
            [group],
            stringtable_entries=[b"", b"amenity", b"cafe", b"name", b"Joe's"],
        )
        data_block = _build_pbf_block("OSMData", pblock)
        pbf = _build_osm_header() + data_block
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf_nodes(path)
            assert result.n_nodes == 2
            assert result.node_tags is not None
            assert len(result.node_tags) == 2
            assert result.node_tags[0] == {"amenity": "cafe", "name": "Joe's"}
            assert result.node_tags[1] == {}
        finally:
            path.unlink()

    @needs_gpu
    def test_nodes_without_tags_returns_none(self):
        """Nodes with no keys_vals field -> node_tags is None."""
        from vibespatial.io.osm_gpu import read_osm_pbf_nodes

        pbf = _build_test_pbf(
            id_deltas=[1, 1],
            lat_deltas=[100, 200],
            lon_deltas=[300, 400],
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf_nodes(path)
            assert result.n_nodes == 2
            assert result.node_tags is None
        finally:
            path.unlink()


class TestWayTagsEndToEnd:
    """End-to-end: PBF with Way tags -> OsmGpuResult.way_tags."""

    @needs_gpu
    def test_ways_with_tags(self):
        """Ways with tags extracted via read_osm_pbf."""
        from vibespatial.io.osm_gpu import read_osm_pbf

        # 3 nodes
        node_id_deltas = [1, 1, 1]
        node_lat_deltas = [0, 10000, 20000]
        node_lon_deltas = [0, 10000, 20000]

        # Way 100 with tags: highway=primary (st[1]=highway, st[2]=primary)
        way = _build_way(100, [1, 1, 1], keys=[1], vals=[2])
        ways_group = _build_primitive_group_with_ways([way])

        dense = _build_dense_nodes(node_id_deltas, node_lat_deltas, node_lon_deltas)
        dense_group = _build_primitive_group(dense)

        pblock = _build_primitive_block_with_stringtable(
            [dense_group, ways_group],
            stringtable_entries=[b"", b"highway", b"primary"],
        )
        data_block = _build_pbf_block("OSMData", pblock)
        pbf = _build_osm_header() + data_block
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf(path)
            assert result.n_ways == 1
            assert result.way_tags is not None
            assert len(result.way_tags) == 1
            assert result.way_tags[0] == {"highway": "primary"}
        finally:
            path.unlink()

    @needs_gpu
    def test_ways_without_tags_returns_none(self):
        """Ways with no tags -> way_tags is None."""
        from vibespatial.io.osm_gpu import read_osm_pbf

        node_id_deltas = [1, 1, 1]
        node_lat_deltas = [0, 10000, 20000]
        node_lon_deltas = [0, 10000, 20000]

        ways = [(100, [1, 2, 3])]
        pbf = _build_test_pbf_with_ways(
            node_id_deltas, node_lat_deltas, node_lon_deltas, ways,
        )
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf(path)
            assert result.n_ways == 1
            assert result.way_tags is None
        finally:
            path.unlink()

    @needs_gpu
    def test_multiple_tags_per_way(self):
        """Way with multiple tag key/value pairs."""
        from vibespatial.io.osm_gpu import read_osm_pbf

        node_id_deltas = [1, 1, 1]
        node_lat_deltas = [0, 10000, 20000]
        node_lon_deltas = [0, 10000, 20000]

        # st: [b"", b"highway", b"building", b"primary", b"yes"]
        way = _build_way(100, [1, 1, 1], keys=[1, 2], vals=[3, 4])
        ways_group = _build_primitive_group_with_ways([way])

        dense = _build_dense_nodes(node_id_deltas, node_lat_deltas, node_lon_deltas)
        dense_group = _build_primitive_group(dense)

        pblock = _build_primitive_block_with_stringtable(
            [dense_group, ways_group],
            stringtable_entries=[b"", b"highway", b"building", b"primary", b"yes"],
        )
        data_block = _build_pbf_block("OSMData", pblock)
        pbf = _build_osm_header() + data_block
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf(path)
            assert result.n_ways == 1
            assert result.way_tags is not None
            assert result.way_tags[0] == {"highway": "primary", "building": "yes"}
        finally:
            path.unlink()


class TestRelationTagsEndToEnd:
    """End-to-end: PBF with Relation tags -> OsmGpuResult.relation_tags."""

    @needs_gpu
    def test_relation_with_tags(self):
        """Relation with tags extracted via read_osm_pbf."""
        from vibespatial.io.osm_gpu import read_osm_pbf

        # Build a simple multipolygon relation with tags
        # 4 nodes for a closed ring
        node_id_deltas = [1, 1, 1, 1]
        node_lat_deltas = [0, 10000000, 0, -10000000]
        node_lon_deltas = [0, 0, 10000000, 0]

        # Way 50: closed ring [1, 2, 3, 4, 1]
        ways = [(50, [1, 2, 3, 4, 1])]

        # Relation 9000: Way 50 as outer, tags: type=multipolygon, name=TestPark
        # st: [b"", b"outer", b"type", b"multipolygon", b"name", b"TestPark"]
        rel = _build_relation(
            9000,
            members=[(50, 1, 1)],  # Way 50, type=WAY, role="outer" (st[1])
            keys=[2, 4],           # type, name
            vals=[3, 5],           # multipolygon, TestPark
        )
        rel_group = _build_primitive_group_with_relations([rel])

        # Build Ways group
        way_messages = []
        for way_id, absolute_refs in ways:
            ref_deltas = [absolute_refs[0]]
            for i in range(1, len(absolute_refs)):
                ref_deltas.append(absolute_refs[i] - absolute_refs[i - 1])
            way_messages.append(_build_way(way_id, ref_deltas))
        ways_group = _build_primitive_group_with_ways(way_messages)

        dense = _build_dense_nodes(node_id_deltas, node_lat_deltas, node_lon_deltas)
        dense_group = _build_primitive_group(dense)

        pblock = _build_primitive_block_with_stringtable(
            [dense_group, ways_group, rel_group],
            stringtable_entries=[
                b"", b"outer", b"type", b"multipolygon", b"name", b"TestPark",
            ],
        )
        data_block = _build_pbf_block("OSMData", pblock)
        pbf = _build_osm_header() + data_block
        path = _write_temp_pbf(pbf)

        try:
            result = read_osm_pbf(path)
            assert result.n_relations > 0
            assert result.relation_tags is not None
            assert len(result.relation_tags) == result.n_relations
            # The first (only) resolved relation should have our tags
            assert result.relation_tags[0] == {
                "type": "multipolygon",
                "name": "TestPark",
            }
        finally:
            path.unlink()


class TestRelationParsingWithTags:
    """Test that _parse_single_relation captures tag key/val fields."""

    def test_relation_keys_vals_captured(self):
        """Relation protobuf with keys/vals fields -> parsed correctly."""
        from vibespatial.io.osm_gpu import _parse_single_relation

        st = [b"", b"outer", b"type", b"multipolygon"]
        rel_bytes = _build_relation(
            1000,
            members=[(10, 1, 1)],  # one Way member
            keys=[2],              # "type"
            vals=[3],              # "multipolygon"
        )
        rel_id, members, tag_keys, tag_vals = _parse_single_relation(rel_bytes, st)
        assert rel_id == 1000
        assert len(members) == 1
        assert tag_keys == [2]
        assert tag_vals == [3]

    def test_relation_no_tags(self):
        """Relation without tags -> empty key/val lists."""
        from vibespatial.io.osm_gpu import _parse_single_relation

        st = [b"", b"outer"]
        rel_bytes = _build_relation(2000, members=[(10, 1, 1)])
        rel_id, members, tag_keys, tag_vals = _parse_single_relation(rel_bytes, st)
        assert rel_id == 2000
        assert tag_keys == []
        assert tag_vals == []


class TestDenseNodesTagExtraction:
    """Test that _extract_dense_nodes_blocks captures keys_vals and stringtable."""

    def test_keys_vals_and_stringtable_captured(self):
        """DenseNodes block with keys_vals -> DenseNodesBlock has both."""
        from vibespatial.io.osm_gpu import _extract_dense_nodes_blocks

        # keys_vals: node1 has tag key=1, val=2; node2 has no tags
        keys_vals = [1, 2, 0, 0]
        dense = _build_dense_nodes([1, 1], [100, 200], [300, 400], keys_vals=keys_vals)
        group = _build_primitive_group(dense)
        pblock = _build_primitive_block_with_stringtable(
            [group],
            stringtable_entries=[b"", b"amenity", b"cafe"],
        )

        results = _extract_dense_nodes_blocks([pblock])
        assert len(results) == 1
        block = results[0]
        assert block.keys_vals_bytes != b""
        assert block.stringtable is not None
        assert len(block.stringtable) == 3
        assert block.stringtable[1] == b"amenity"

    def test_no_keys_vals_field(self):
        """DenseNodes without keys_vals -> empty bytes, stringtable still parsed."""
        from vibespatial.io.osm_gpu import _extract_dense_nodes_blocks

        dense = _build_dense_nodes([1], [100], [200])
        group = _build_primitive_group(dense)
        pblock = _build_primitive_block_with_stringtable(
            [group],
            stringtable_entries=[b"", b"test"],
        )

        results = _extract_dense_nodes_blocks([pblock])
        assert len(results) == 1
        block = results[0]
        assert block.keys_vals_bytes == b""
        assert block.stringtable is not None
