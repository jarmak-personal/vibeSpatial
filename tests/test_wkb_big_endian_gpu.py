from __future__ import annotations

import struct

import numpy as np
import pytest
import shapely

import vibespatial.api as geopandas
from vibespatial import decode_wkb_owned, has_gpu_runtime
from vibespatial.cuda._runtime import (
    get_d2h_transfer_events,
    pylibcudf_column_from_arrow,
    reset_d2h_transfer_count,
)
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.io.pylibcudf import _decode_pylibcudf_wkb_general_column_to_owned
from vibespatial.io.wkb import decode_wkb_arrow_array_owned
from vibespatial.io.wkb_decode_status import WKBDecodeStatus
from vibespatial.kernels.core.wkb_decode import (
    WKBDeviceDecodeDeclined,
    decode_wkb_device_pipeline,
    scan_wkb_device_structural_plan,
    summarize_wkb_device_plan,
)
from vibespatial.runtime import set_requested_mode
from vibespatial.testing import strict_native_environment

pytestmark = pytest.mark.skipif(not has_gpu_runtime(), reason="CUDA runtime required")


def _prefix(order: int) -> str:
    return "<" if order == 1 else ">"


def _header(order: int, type_id: int) -> bytes:
    return bytes((order,)) + struct.pack(f"{_prefix(order)}I", type_id)


def _point(order: int, x: float = 1.25, y: float = -2.5) -> bytes:
    return _header(order, 1) + struct.pack(f"{_prefix(order)}dd", x, y)


def _point_bits(order: int, x_bits: int, y_bits: int) -> bytes:
    prefix = _prefix(order)
    return _header(order, 1) + struct.pack(f"{prefix}QQ", x_bits, y_bits)


def _linestring(order: int, coords=((0.0, 0.0), (2.0, 3.0))) -> bytes:
    prefix = _prefix(order)
    return (
        _header(order, 2)
        + struct.pack(f"{prefix}I", len(coords))
        + b"".join(struct.pack(f"{prefix}dd", x, y) for x, y in coords)
    )


def _polygon(
    order: int,
    rings=(((0.0, 0.0), (3.0, 0.0), (0.0, 3.0), (0.0, 0.0)),),
) -> bytes:
    prefix = _prefix(order)
    return (
        _header(order, 3)
        + struct.pack(f"{prefix}I", len(rings))
        + b"".join(
            struct.pack(f"{prefix}I", len(ring))
            + b"".join(struct.pack(f"{prefix}dd", x, y) for x, y in ring)
            for ring in rings
        )
    )


def _multi(order: int, type_id: int, children: tuple[bytes, ...]) -> bytes:
    return (
        _header(order, type_id)
        + struct.pack(f"{_prefix(order)}I", len(children))
        + b"".join(children)
    )


def _family_records(order: int) -> dict[GeometryFamily, bytes]:
    return {
        GeometryFamily.POINT: _point(order),
        GeometryFamily.LINESTRING: _linestring(order),
        GeometryFamily.POLYGON: _polygon(order),
        GeometryFamily.MULTIPOINT: _multi(
            order,
            4,
            (_point(order, 1.0, 2.0), _point(order, 3.0, 4.0)),
        ),
        GeometryFamily.MULTILINESTRING: _multi(
            order,
            5,
            (_linestring(order), _linestring(order, ((4.0, 5.0), (6.0, 7.0)))),
        ),
        GeometryFamily.MULTIPOLYGON: _multi(
            order,
            6,
            (_polygon(order), _polygon(order, (((5.0, 5.0), (7.0, 5.0), (5.0, 7.0), (5.0, 5.0)),))),
        ),
    }


def _device_records(records: list[bytes]):
    import cupy as cp

    lengths = np.asarray([len(record) for record in records], dtype=np.int64)
    offsets = np.empty(len(records) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    payload = np.frombuffer(b"".join(records), dtype=np.uint8)
    return cp.asarray(payload), cp.asarray(offsets)


def _decode_device_records(records: list[bytes], **kwargs):
    payload, offsets = _device_records(records)
    return decode_wkb_device_pipeline(payload, offsets, len(records), **kwargs)


def _pylibcudf_wkb_column(records: list[bytes]):
    import pyarrow as pa

    binary = pa.array(records, type=pa.binary())
    string = pa.Array.from_buffers(
        pa.string(),
        len(binary),
        binary.buffers(),
        null_count=binary.null_count,
    )
    return pylibcudf_column_from_arrow(string)


def test_all_six_families_decode_little_and_big_endian_exactly() -> None:
    records = [record for order in (1, 0) for record in _family_records(order).values()]
    expected = shapely.from_wkb(np.asarray(records, dtype=object))

    with set_requested_mode("gpu"):
        decoded = decode_wkb_owned(records)
    actual = np.asarray(decoded.to_shapely(), dtype=object)

    assert decoded.residency.value == "device"
    assert shapely.equals_exact(actual, expected, tolerance=0.0).all()


@pytest.mark.parametrize("type_id,child_builder", [(4, _point), (5, _linestring), (6, _polygon)])
@pytest.mark.parametrize(
    "root_order,first_order,second_order",
    [
        (root_order, first_order, second_order)
        for root_order in (0, 1)
        for first_order in (0, 1)
        for second_order in (0, 1)
    ],
)
def test_multi_geometry_children_honor_independent_byte_order(
    type_id,
    child_builder,
    root_order,
    first_order,
    second_order,
) -> None:
    record = _multi(
        root_order,
        type_id,
        (child_builder(first_order), child_builder(second_order)),
    )
    decoded = _decode_device_records([record])

    actual = np.asarray(decoded.to_shapely(), dtype=object)
    expected = shapely.from_wkb(np.asarray([record], dtype=object))
    assert shapely.equals_exact(actual, expected, tolerance=0.0).all()
    summary = summarize_wkb_device_plan(decoded._wkb_structural_plan)
    expected_mixed = first_order != root_order or second_order != root_order
    assert summary["native_mixed_endian_rows"] == int(expected_mixed)
    assert summary["native_big_endian_rows"] == int(not expected_mixed and root_order == 0)
    assert summary["native_little_endian_rows"] == int(not expected_mixed and root_order == 1)


def test_coordinate_decode_preserves_ieee754_bits() -> None:
    cases = [
        (0x8000000000000000, 0x0000000000000001),
        (0x7FF0000000000000, 0xFFF0000000000000),
        (0x7FF8000000001234, 0x3FF0000000000000),
    ]
    records = [_point_bits(order, x_bits, y_bits) for order in (0, 1) for x_bits, y_bits in cases]
    decoded = _decode_device_records(records)
    decoded._ensure_host_state(preserve_indexed_view=True)
    point = decoded.families[GeometryFamily.POINT]

    assert point.x.view(np.uint64).tolist() == [x for _order in (0, 1) for x, _y in cases]
    assert point.y.view(np.uint64).tolist() == [y for _order in (0, 1) for _x, y in cases]


def test_null_root_empty_and_nested_empty_records_preserve_structure() -> None:
    nan_a = 0x7FF8000000000001
    nan_b = 0xFFF8000000000002
    empty_point = _point_bits(0, nan_a, nan_b)
    records = [
        empty_point,
        _linestring(0, ()),
        _polygon(0, ()),
        _multi(0, 4, (empty_point, _point(1))),
        _multi(0, 5, (_linestring(1, ()), _linestring(0))),
        _multi(0, 6, (_polygon(1, ()), _polygon(0))),
        _multi(0, 4, ()),
        _multi(0, 5, ()),
        _multi(0, 6, ()),
    ]
    payload, offsets = _device_records(records)
    import cupy as cp

    payload_with_null = cp.concatenate((payload, cp.asarray([], dtype=cp.uint8)))
    offsets_with_null = cp.concatenate((offsets, offsets[-1:]))
    validity = cp.asarray([True] * len(records) + [False])
    decoded = decode_wkb_device_pipeline(
        payload_with_null,
        offsets_with_null,
        len(records) + 1,
        validity_device=validity,
    )

    encoded = decoded.to_wkb()
    assert encoded[-1] is None
    expected = shapely.from_wkb(np.asarray(records, dtype=object))
    actual = shapely.from_wkb(np.asarray(encoded[:-1], dtype=object))
    assert shapely.equals_exact(actual, expected, tolerance=0.0).all()
    exported = np.asarray(decoded.to_shapely()[:-1], dtype=object)
    assert shapely.equals_exact(exported, expected, tolerance=0.0).all()


@pytest.mark.parametrize(
    "record,status",
    [
        (b"\x02\x00\x00\x00\x01" + b"\x00" * 16, WKBDecodeStatus.INVALID_BYTE_ORDER),
        (_point(0)[:-1], WKBDecodeStatus.TRUNCATED_OR_MALFORMED),
        (_header(0, 2) + struct.pack(">I", 0xFFFFFFFF), WKBDecodeStatus.COUNT_OR_OFFSET_OVERFLOW),
        (
            _header(1, 0x20000001) + struct.pack("<I", 4326) + struct.pack("<dd", 1.0, 2.0),
            WKBDecodeStatus.EWKB_SRID,
        ),
        (_header(0, 1001) + b"\x00" * 24, WKBDecodeStatus.DIMENSIONAL_WKB),
        (_header(0, 7) + struct.pack(">I", 0), WKBDecodeStatus.GEOMETRY_COLLECTION),
        (_header(0, 8), WKBDecodeStatus.UNSUPPORTED_FAMILY),
        (_multi(0, 4, (_linestring(1),)), WKBDecodeStatus.EMBEDDED_FAMILY_MISMATCH),
        (_point(0) + b"\x00", WKBDecodeStatus.TRAILING_BYTES),
        (_linestring(0, ((1.0, 2.0),)), WKBDecodeStatus.SEMANTIC_INVALID),
    ],
)
def test_structural_scan_returns_precise_decline_status(record, status) -> None:
    payload, offsets = _device_records([record])
    plan = scan_wkb_device_structural_plan(payload, offsets, 1)
    assert int(plan.statuses.get()[0]) == int(status)


def test_structural_scan_rejects_offset_beyond_payload_before_device_read() -> None:
    import cupy as cp

    payload = cp.asarray(np.frombuffer(_point(0), dtype=np.uint8))
    offsets = cp.asarray([0, len(_point(0)) + 4096], dtype=cp.int64)
    plan = scan_wkb_device_structural_plan(
        payload,
        offsets,
        1,
        validity_device=cp.asarray([True]),
    )
    assert int(plan.statuses.get()[0]) == int(WKBDecodeStatus.TRUNCATED_OR_MALFORMED)


def test_declined_rows_raise_before_family_decode_and_never_fake_success() -> None:
    record = _header(0, 7) + struct.pack(">I", 0)
    with pytest.raises(WKBDeviceDecodeDeclined, match="GeometryCollection"):
        _decode_device_records([record])


@pytest.mark.parametrize("on_invalid", ["warn", "ignore"])
def test_semantic_invalid_rows_follow_on_invalid_policy(on_invalid) -> None:
    records = [_point(0), _linestring(0, ((1.0, 2.0),))]
    warning = pytest.warns(UserWarning, match="point array") if on_invalid == "warn" else None
    if warning is None:
        decoded = _decode_device_records(records, on_invalid=on_invalid)
    else:
        with warning:
            decoded = _decode_device_records(records, on_invalid=on_invalid)
    decoded._ensure_host_state(preserve_indexed_view=True)
    assert decoded.validity.tolist() == [True, False]


def test_strict_native_big_endian_decode_has_no_fallback_or_internal_materialization() -> None:
    geopandas.clear_dispatch_events()
    geopandas.clear_fallback_events()
    geopandas.clear_materialization_events()
    records = [_point(0), _linestring(0), _polygon(0)]

    with strict_native_environment(execution_mode="auto"):
        decoded = decode_wkb_owned(records)

    assert decoded.residency.value == "device"
    assert geopandas.get_fallback_events(clear=True) == []
    assert geopandas.get_materialization_events(clear=True) == []
    events = geopandas.get_dispatch_events(clear=True)
    assert any(event.implementation == "device_wkb_decode" for event in events)


@pytest.mark.parametrize("order", [0, 1])
def test_pylibcudf_dense_point_shape_uses_final_buffer_decode(
    monkeypatch: pytest.MonkeyPatch,
    order: int,
) -> None:
    from vibespatial.kernels.core import wkb_decode as wkb_decode_module

    calls: list[int] = []
    original = wkb_decode_module._decode_dense_point_family

    def recording_dense_decode(payload, offsets, row_count, *, byte_order):
        calls.append(byte_order)
        return original(
            payload,
            offsets,
            row_count,
            byte_order=byte_order,
        )

    monkeypatch.setattr(
        wkb_decode_module,
        "_decode_dense_point_family",
        recording_dense_decode,
    )
    records = [_point(order, float(index), -float(index)) for index in range(8)]

    decoded = _decode_pylibcudf_wkb_general_column_to_owned(
        _pylibcudf_wkb_column(records)
    )

    assert calls == [order]
    assert decoded.__dict__.get("_wkb_structural_plan") is None
    assert decoded.__dict__["_wkb_structural_summary"][
        "native_big_endian_rows" if order == 0 else "native_little_endian_rows"
    ] == len(records)
    expected = shapely.from_wkb(np.asarray(records, dtype=object))
    assert shapely.equals_exact(
        np.asarray(decoded.to_shapely(), dtype=object),
        expected,
        tolerance=0.0,
    ).all()


def test_pylibcudf_mixed_families_use_exact_summary_work_totals() -> None:
    records = [
        _family_records(order)[family]
        for order in (0, 1)
        for family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON)
    ]

    decoded = _decode_pylibcudf_wkb_general_column_to_owned(
        _pylibcudf_wkb_column(records)
    )

    summary = decoded.__dict__["_wkb_structural_summary"]
    assert sum(summary["family_part_counts"].values()) == summary["part_count"]
    assert sum(summary["family_ring_counts"].values()) == summary["ring_count"]
    assert (
        sum(summary["family_coordinate_counts"].values())
        == summary["coordinate_count"]
    )
    assert summary["family_counts"][GeometryFamily.POLYGON.value] == 2
    assert summary["family_counts"][GeometryFamily.MULTIPOLYGON.value] == 2
    state = decoded._ensure_device_state(preserve_indexed_view=True)
    polygon = state.families[GeometryFamily.POLYGON]
    multipolygon = state.families[GeometryFamily.MULTIPOLYGON]
    assert int(polygon.x.data.ptr) + int(polygon.x.nbytes) == int(
        multipolygon.x.data.ptr
    )
    assert int(polygon.y.data.ptr) + int(polygon.y.nbytes) == int(
        multipolygon.y.data.ptr
    )
    expected = shapely.from_wkb(np.asarray(records, dtype=object))
    assert shapely.equals_exact(
        np.asarray(decoded.to_shapely(), dtype=object),
        expected,
        tolerance=0.0,
    ).all()


def test_sparse_compatibility_decodes_only_declined_rows_and_scatter_merges() -> None:
    ewkb = bytes.fromhex("0101000020e610000000000000000008400000000000001040")
    column = _pylibcudf_wkb_column([_point(0, 1.0, 2.0), ewkb, _point(1, 5.0, 6.0)])
    geopandas.clear_fallback_events()
    geopandas.clear_dispatch_events()

    decoded = _decode_pylibcudf_wkb_general_column_to_owned(column)
    actual = np.asarray(decoded.to_shapely(), dtype=object)

    assert shapely.equals_exact(
        actual,
        shapely.from_wkb(
            np.asarray([_point(0, 1.0, 2.0), ewkb, _point(1, 5.0, 6.0)], dtype=object)
        ),
        tolerance=0.0,
    ).all()
    fallbacks = geopandas.get_fallback_events(clear=True)
    assert len(fallbacks) == 1
    assert "1 declined rows" in fallbacks[0].detail
    events = geopandas.get_dispatch_events(clear=True)
    assert any(
        event.implementation == "device_wkb_decode_with_sparse_compatibility_merge"
        for event in events
    )


def test_strict_native_decline_stops_before_payload_compatibility_transfer() -> None:
    ewkb = bytes.fromhex("0101000020e610000000000000000008400000000000001040")
    column = _pylibcudf_wkb_column([_point(0), ewkb])
    reset_d2h_transfer_count()
    get_d2h_transfer_events(clear=True)

    with strict_native_environment(), pytest.raises(WKBDeviceDecodeDeclined, match="EWKB"):
        _decode_pylibcudf_wkb_general_column_to_owned(column)

    events = get_d2h_transfer_events(clear=True)
    assert events
    assert {event.reason for event in events} <= {
        "WKB decode bounded aggregate telemetry packet",
        "WKB decode bounded decline status packet",
        "WKB decode bounded decline row packet",
    }
    assert sum(event.bytes_transferred for event in events) <= 381


@pytest.mark.parametrize("carrier", ["binary", "large_binary", "binary_view"])
def test_arrow_binary_carriers_and_logical_slices_decode_on_device(carrier) -> None:
    import pyarrow as pa

    carrier_type = {
        "binary": pa.binary(),
        "large_binary": pa.large_binary(),
        "binary_view": pa.binary_view(),
    }[carrier]
    array = pa.array(
        [_point(1, 0.0, 0.0), _point(0, 1.0, 2.0), _point(1, 3.0, 4.0)], type=carrier_type
    )
    sliced = array.slice(1, 2)

    decoded = decode_wkb_arrow_array_owned(
        sliced,
        allow_fallback=False,
        requested_mode="gpu",
    )

    assert decoded.residency.value == "device"
    expected = shapely.from_wkb(np.asarray(sliced.to_pylist(), dtype=object))
    assert shapely.equals_exact(
        np.asarray(decoded.to_shapely(), dtype=object),
        expected,
        tolerance=0.0,
    ).all()


def test_native_metadata_reuses_decode_seeded_device_classification() -> None:
    from vibespatial.api._native_metadata import NativeGeometryMetadata

    decoded = _decode_device_records([_point(0), _linestring(1), _polygon(0)])
    state = decoded.device_state
    metadata = NativeGeometryMetadata.from_cached_owned(decoded)

    assert state is not None
    assert metadata.validity is state.validity
    assert metadata.family_tags is state.tags
    assert metadata.family_row_offsets is state.family_row_offsets
    assert metadata.residency.value == "device"
