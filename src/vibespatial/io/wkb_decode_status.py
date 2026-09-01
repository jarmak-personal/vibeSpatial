"""Stable device status codes for canonical 2D WKB admission."""

from __future__ import annotations

from enum import IntEnum


class WKBDecodeStatus(IntEnum):
    """Row-level result of the byte-authoritative WKB structural scan."""

    NULL = 0
    NATIVE_LITTLE_ENDIAN = 1
    NATIVE_BIG_ENDIAN = 2
    NATIVE_MIXED_ENDIAN = 3
    INVALID_BYTE_ORDER = 10
    TRUNCATED_OR_MALFORMED = 11
    COUNT_OR_OFFSET_OVERFLOW = 12
    EWKB_SRID = 13
    DIMENSIONAL_WKB = 14
    GEOMETRY_COLLECTION = 15
    UNSUPPORTED_FAMILY = 16
    EMBEDDED_FAMILY_MISMATCH = 17
    TRAILING_BYTES = 18
    SEMANTIC_INVALID = 19


NATIVE_WKB_STATUSES = frozenset(
    {
        WKBDecodeStatus.NATIVE_LITTLE_ENDIAN,
        WKBDecodeStatus.NATIVE_BIG_ENDIAN,
        WKBDecodeStatus.NATIVE_MIXED_ENDIAN,
    }
)

WKB_STATUS_REASONS = {
    WKBDecodeStatus.NULL: "null",
    WKBDecodeStatus.NATIVE_LITTLE_ENDIAN: "native little endian",
    WKBDecodeStatus.NATIVE_BIG_ENDIAN: "native big endian",
    WKBDecodeStatus.NATIVE_MIXED_ENDIAN: "native mixed embedded endian",
    WKBDecodeStatus.INVALID_BYTE_ORDER: "invalid WKB byte-order flag",
    WKBDecodeStatus.TRUNCATED_OR_MALFORMED: "truncated or malformed WKB record",
    WKBDecodeStatus.COUNT_OR_OFFSET_OVERFLOW: "WKB count, cursor, or offset overflow",
    WKBDecodeStatus.EWKB_SRID: "EWKB SRID or flag encoding",
    WKBDecodeStatus.DIMENSIONAL_WKB: "OGC Z/M/ZM WKB",
    WKBDecodeStatus.GEOMETRY_COLLECTION: "GeometryCollection",
    WKBDecodeStatus.UNSUPPORTED_FAMILY: "unsupported WKB family",
    WKBDecodeStatus.EMBEDDED_FAMILY_MISMATCH: "embedded WKB family mismatch",
    WKBDecodeStatus.TRAILING_BYTES: "WKB record has trailing bytes",
    WKBDecodeStatus.SEMANTIC_INVALID: "point array must contain 0 or >1 elements",
}


def is_native_wkb_status(status: int | WKBDecodeStatus) -> bool:
    return WKBDecodeStatus(int(status)) in NATIVE_WKB_STATUSES
