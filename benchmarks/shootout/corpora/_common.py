"""Shared identity and correctness helpers for external-corpus shootouts."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from functools import cache
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import shapely

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
_MANIFEST = json.loads((_HERE / "vsbench-workload.json").read_text(encoding="utf-8"))


def get_scale() -> int:
    """Return the maximum rows admitted by this discovery run."""
    raw = os.environ.get("VSBENCH_SCALE", "10000").strip().upper()
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    for suffix, multiplier in multipliers.items():
        if raw.endswith(suffix):
            return int(float(raw[:-1]) * multiplier)
    return int(raw)


def _data_root() -> Path:
    override = os.environ.get(_MANIFEST["data_root_env"])
    if override:
        return Path(override).expanduser().resolve()
    return (_REPO_ROOT / _MANIFEST["default_data_root"]).resolve()


@cache
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_asset(key: str) -> Path:
    """Return one verified local asset or fail before the timed region."""
    asset = _MANIFEST["assets"][key]
    path = _data_root() / asset["local_path"]
    if not path.is_file():
        raise FileNotFoundError(
            f"missing external corpus asset {key!r}: {path}; "
            "run `uv run python scripts/manage_external_corpora.py fetch`"
        )
    size = path.stat().st_size
    if size != asset["size_bytes"]:
        raise RuntimeError(
            f"external corpus asset {key!r} has size {size}, expected {asset['size_bytes']}"
        )
    actual = _sha256(path)
    if actual != asset["sha256"]:
        raise RuntimeError(
            f"external corpus asset {key!r} has SHA-256 {actual}, expected {asset['sha256']}"
        )
    return path


_RESULT_FINGERPRINT_VERSION = "vsbench-result-v3"
_RESULT_METRIC_SIGNIFICANT_DIGITS = 7
_RESULT_COORDINATE_SIGNIFICANT_DIGITS = 11
_FINGERPRINT_DIAGNOSTICS_ENV = "VSBENCH_FINGERPRINT_DIAGNOSTICS"
_FINGERPRINT_DUMP_ENV = "VSBENCH_FINGERPRINT_DUMP_DIR"


def _pack(tag: bytes, payload: bytes = b"") -> bytes:
    return tag + struct.pack(">Q", len(payload)) + payload


def _canonical_float(
    value: float,
    *,
    tolerant: bool,
    significant_digits: int = _RESULT_METRIC_SIGNIFICANT_DIGITS,
) -> bytes:
    if pd.isna(value):
        return _pack(b"n")
    if not tolerant:
        return _pack(b"f", struct.pack(">d", value))
    if value == 0.0:
        value = 0.0
    return _pack(
        b"f",
        format(value, f".{significant_digits}g").encode("ascii"),
    )


def _canonical_value(value: Any, *, tolerant_floats: bool = False) -> bytes:
    """Encode ordinary pandas scalar/nested values without lossy coercion."""
    if value is None or value is pd.NA or value is pd.NaT:
        return _pack(b"n")

    as_py = getattr(value, "as_py", None)
    if callable(as_py):
        value = as_py()
        if value is None:
            return _pack(b"n")

    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (bytes, bytearray, memoryview)):
        try:
            value = item()
        except ValueError:
            pass

    if isinstance(value, bool):
        return _pack(b"b", b"1" if value else b"0")
    if isinstance(value, int):
        return _pack(b"i", str(value).encode("ascii"))
    if isinstance(value, float):
        return _canonical_float(value, tolerant=tolerant_floats)
    if isinstance(value, Decimal):
        sign, digits, exponent = value.as_tuple()
        payload = f"{sign}:{''.join(map(str, digits))}:{exponent}".encode("ascii")
        return _pack(b"d", payload)
    if isinstance(value, str):
        return _pack(b"s", value.encode("utf-8"))
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _pack(b"y", bytes(value))
    if isinstance(value, datetime):
        return _pack(b"D", value.isoformat().encode("utf-8"))
    if isinstance(value, date):
        return _pack(b"a", value.isoformat().encode("ascii"))
    if isinstance(value, time):
        return _pack(b"t", value.isoformat().encode("utf-8"))
    if isinstance(value, timedelta):
        return _pack(b"T", str(value.total_seconds()).encode("ascii"))
    if isinstance(value, Mapping):
        items = sorted(
            (
                _canonical_value(key, tolerant_floats=tolerant_floats),
                _canonical_value(item, tolerant_floats=tolerant_floats),
            )
            for key, item in value.items()
        )
        return _pack(
            b"m",
            b"".join(_pack(b"k", key) + _pack(b"v", item) for key, item in items),
        )
    if isinstance(value, tuple):
        return _pack(
            b"u",
            b"".join(
                _pack(
                    b"e",
                    _canonical_value(v, tolerant_floats=tolerant_floats),
                )
                for v in value
            ),
        )
    if isinstance(value, list):
        return _pack(
            b"l",
            b"".join(
                _pack(
                    b"e",
                    _canonical_value(v, tolerant_floats=tolerant_floats),
                )
                for v in value
            ),
        )
    if isinstance(value, (set, frozenset)):
        return _pack(
            b"q",
            b"".join(
                _pack(b"e", item)
                for item in sorted(
                    _canonical_value(v, tolerant_floats=tolerant_floats)
                    for v in value
                )
            ),
        )

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _canonical_value(
            tolist(),
            tolerant_floats=tolerant_floats,
        )

    raise TypeError(
        "unsupported result value for correctness fingerprint: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _canonical_geometry_wkb(payload: bytes) -> bytes:
    """Encode WKB structure exactly while quantizing coordinate values."""
    geometry = shapely.from_wkb(payload)
    normalized = shapely.normalize(geometry)
    view = memoryview(shapely.to_wkb(normalized, hex=False))

    def read(offset: int, size: int) -> tuple[memoryview, int]:
        end = offset + size
        if end > len(view):
            raise TypeError("truncated geometry WKB in correctness fingerprint")
        return view[offset:end], end

    def parse(offset: int) -> tuple[bytes, int]:
        endian_raw, offset = read(offset, 1)
        byte_order = int(endian_raw[0])
        if byte_order not in (0, 1):
            raise TypeError("invalid geometry WKB byte order in correctness fingerprint")
        endian = ">" if byte_order == 0 else "<"
        type_raw, offset = read(offset, 4)
        raw_type = struct.unpack(f"{endian}I", type_raw)[0]

        has_z = bool(raw_type & 0x80000000)
        has_m = bool(raw_type & 0x40000000)
        has_srid = bool(raw_type & 0x20000000)
        type_code = raw_type & 0x1FFFFFFF
        if type_code >= 3000:
            has_z = True
            has_m = True
            base_type = type_code - 3000
        elif type_code >= 2000:
            has_m = True
            base_type = type_code - 2000
        elif type_code >= 1000:
            has_z = True
            base_type = type_code - 1000
        else:
            base_type = type_code
        if base_type not in range(1, 8):
            raise TypeError(
                "unsupported geometry WKB type in correctness fingerprint: "
                f"{base_type}"
            )

        parts = [
            _pack(b"t", str(base_type).encode("ascii")),
            _pack(b"z", b"1" if has_z else b"0"),
            _pack(b"m", b"1" if has_m else b"0"),
        ]
        if has_srid:
            srid_raw, offset = read(offset, 4)
            srid = struct.unpack(f"{endian}I", srid_raw)[0]
            parts.append(_pack(b"s", str(srid).encode("ascii")))
        else:
            parts.append(_pack(b"s"))

        dimensions = 2 + int(has_z) + int(has_m)

        def coordinates(count: int, position: int) -> tuple[bytes, int]:
            encoded: list[bytes] = []
            for _ in range(count * dimensions):
                raw, position = read(position, 8)
                coordinate = struct.unpack(f"{endian}d", raw)[0]
                encoded.append(
                    _canonical_float(
                        coordinate,
                        tolerant=True,
                        significant_digits=_RESULT_COORDINATE_SIGNIFICANT_DIGITS,
                    )
                )
            return _pack(b"x", b"".join(encoded)), position

        def count(position: int) -> tuple[int, int]:
            raw, position = read(position, 4)
            return struct.unpack(f"{endian}I", raw)[0], position

        if base_type == 1:
            encoded, offset = coordinates(1, offset)
            parts.append(encoded)
        elif base_type == 2:
            coordinate_count, offset = count(offset)
            parts.append(_pack(b"n", str(coordinate_count).encode("ascii")))
            encoded, offset = coordinates(coordinate_count, offset)
            parts.append(encoded)
        elif base_type == 3:
            ring_count, offset = count(offset)
            parts.append(_pack(b"n", str(ring_count).encode("ascii")))
            for _ in range(ring_count):
                coordinate_count, offset = count(offset)
                parts.append(_pack(b"n", str(coordinate_count).encode("ascii")))
                encoded, offset = coordinates(coordinate_count, offset)
                parts.append(encoded)
        else:
            geometry_count, offset = count(offset)
            parts.append(_pack(b"n", str(geometry_count).encode("ascii")))
            for _ in range(geometry_count):
                encoded, offset = parse(offset)
                parts.append(_pack(b"e", encoded))
        return _pack(b"w", b"".join(parts)), offset

    encoded, consumed = parse(0)
    if consumed != len(view):
        raise TypeError("trailing geometry WKB bytes in correctness fingerprint")
    return encoded


def _null_dtype_semantics(dtype: Any) -> str:
    name = str(getattr(dtype, "name", dtype))
    if name.startswith("vibespatial_"):
        logical_name = name.removeprefix("vibespatial_")
        if logical_name == "attribute":
            raise TypeError(
                "opaque all-null attribute schema cannot be fingerprinted exactly"
            )
        return logical_name
    return name


def _logical_schema_descriptors(
    values,
    carrier_dtypes,
    *,
    semantic_overrides: dict[int, str] | None = None,
) -> list[bytes]:
    """Describe public logical fields, independent of pandas carrier classes."""
    try:
        logical_types = [
            pa.array(
                values.iloc[:, position].array.to_numpy(),
                from_pandas=True,
            ).type
            for position in range(values.shape[1])
        ]
    except pa.ArrowException as exc:
        raise TypeError(
            "unsupported result value or schema for correctness fingerprint: "
            f"{exc}"
        ) from exc
    descriptors: list[bytes] = []
    for position, (logical_type, dtype) in enumerate(
        zip(logical_types, carrier_dtypes, strict=True)
    ):
        override = (semantic_overrides or {}).get(position)
        kind = str(getattr(dtype, "kind", ""))
        dtype_name = str(getattr(dtype, "name", dtype))
        native_carrier = dtype_name.startswith("vibespatial_")
        if native_carrier:
            dtype_name = dtype_name.removeprefix("vibespatial_")
        if override is not None:
            logical_name = override
        elif pa.types.is_null(logical_type):
            logical_name = f"null[{_null_dtype_semantics(dtype)}]"
        elif native_carrier and dtype_name != "attribute":
            # Native carriers are an execution detail. Describe their public
            # scalar width so, for example, vibespatial_uint64 and pandas'
            # ordinary uint64 compare as the same logical result schema.
            logical_name = dtype_name.lower()
        elif kind in "biufcMm":
            # Preserve public extension semantics such as nullable `Int64`;
            # unlike a native carrier, that distinction is part of the result.
            logical_name = dtype_name
        elif pa.types.is_float32(logical_type):
            logical_name = "float32"
        elif pa.types.is_float64(logical_type):
            logical_name = "float64"
        else:
            logical_name = str(logical_type)
        payload: list[bytes] = [
            _pack(b"t", logical_name.encode("utf-8")),
        ]
        categories = getattr(dtype, "categories", None)
        if categories is not None:
            payload.append(_pack(b"c", _canonical_value(list(categories))))
            payload.append(_pack(b"o", b"1" if bool(dtype.ordered) else b"0"))
        descriptors.append(_pack(b"d", b"".join(payload)))
    return descriptors


def _carrier_dtype_detail(dtype: Any) -> dict[str, str]:
    type_name = f"{type(dtype).__module__}.{type(dtype).__qualname__}"
    name = getattr(dtype, "name", None)
    storage = getattr(dtype, "storage", None)
    return {
        "type": type_name,
        "name": str(name) if name is not None else "",
        "storage": str(storage) if storage is not None else "",
        "repr": repr(dtype),
    }


def _diagnostic_value(value: Any) -> dict[str, Any]:
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (bytes, bytearray, memoryview)):
        try:
            value = item()
        except ValueError:
            pass
    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
        details = {
            "kind": "bytes",
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        if len(payload) <= 128:
            details["hex"] = payload.hex()
        return details
    if isinstance(value, float):
        return {"kind": "float", "repr": repr(value), "hex": value.hex()}
    rendered = repr(value)
    return {
        "kind": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": rendered[:160],
    }


def _geometry_column_positions(frame, public_values) -> list[int]:
    """Return geometry column positions from public dtype names."""
    if not callable(getattr(frame, "to_wkb", None)):
        return []
    positions: list[int] = []
    for position, dtype in enumerate(frame.dtypes):
        name = str(getattr(dtype, "name", dtype)).lower()
        if "geometry" in name:
            positions.append(position)
            continue
        before = frame.iloc[:, position]
        after = public_values.iloc[:, position]
        if len(before) and len(after):
            original = next((value for value in before.array if value is not None), None)
            encoded = next((value for value in after.array if value is not None), None)
            if encoded is not None and isinstance(encoded, bytes) and not isinstance(
                original, bytes
            ):
                positions.append(position)
    return positions


def _component_sha256(parts) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(_pack(b"p", part))
    return digest.hexdigest()


def _emit_fingerprint_diagnostics(
    frame,
    public_values,
    index_frame,
    *,
    geometry_name,
    crs_text,
    order_sensitive: bool,
) -> None:
    """Print component hashes that localize cross-environment mismatches."""
    if os.environ.get(_FINGERPRINT_DIAGNOSTICS_ENV) != "1":
        return

    dtypes = list(frame.dtypes)
    index_dtypes = list(index_frame.dtypes)
    geometry_positions = _geometry_column_positions(frame, public_values)
    logical_dtypes = _logical_schema_descriptors(
        public_values,
        dtypes,
        semantic_overrides={position: "geometry" for position in geometry_positions},
    )
    logical_index_dtypes = _logical_schema_descriptors(index_frame, index_dtypes)
    components = {
        "columns": _component_sha256([_canonical_value(list(frame.columns))]),
        "dtypes": _component_sha256(logical_dtypes),
        "index_schema": _component_sha256(
            [
                type(frame.index).__name__.encode("utf-8"),
                _canonical_value(list(frame.index.names)),
                *logical_index_dtypes,
            ]
        ),
        "index_values": _component_sha256(
            _canonical_value(tuple(row))
            for row in index_frame.itertuples(index=False, name=None)
        ),
        "geometry_metadata": _component_sha256(
            [
                _canonical_value(geometry_name),
                _canonical_value(crs_text),
                _canonical_value(geometry_positions),
            ]
        ),
    }
    for position in range(public_values.shape[1]):
        series = public_values.iloc[:, position]
        components[f"column_{position}_values"] = _component_sha256(
            _canonical_value(value) for value in series.array
        )

    details = {
        "column_names": [repr(column) for column in frame.columns],
        "logical_dtypes": [part.hex() for part in logical_dtypes],
        "carrier_dtypes": [_carrier_dtype_detail(dtype) for dtype in dtypes],
        "index_type": type(frame.index).__name__,
        "index_names": [repr(name) for name in frame.index.names],
        "index_samples": [
            [repr(value) for value in row]
            for row in list(index_frame.itertuples(index=False, name=None))[:3]
        ],
        "logical_index_dtypes": [part.hex() for part in logical_index_dtypes],
        "carrier_index_dtypes": [
            _carrier_dtype_detail(dtype) for dtype in index_dtypes
        ],
        "geometry_name": repr(geometry_name),
        "crs_sha256": hashlib.sha256((crs_text or "").encode("utf-8")).hexdigest(),
        "column_samples": [
            [
                _diagnostic_value(value)
                for value in public_values.iloc[:, position].array[:3]
            ]
            for position in range(public_values.shape[1])
        ],
    }
    payload = {
        "version": _RESULT_FINGERPRINT_VERSION,
        "order": "ordered" if order_sensitive else "unordered",
        "components": components,
        "details": details,
    }
    print(
        "SHOOTOUT_FINGERPRINT_COMPONENTS: "
        + json.dumps(payload, sort_keys=True, separators=(",", ":"))
    )


def _dump_fingerprint_values(public_values) -> None:
    """Persist public values for an explicitly requested cross-engine diff."""
    dump_root = os.environ.get(_FINGERPRINT_DUMP_ENV)
    if not dump_root:
        return
    root = Path(dump_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    script_name = Path(sys.argv[1]).stem if len(sys.argv) > 1 else "result"
    engine = f"pandas-{pd.__version__}-pyarrow-{pa.__version__}"
    columns = [
        pd.Series(
            public_values.iloc[:, position].array.to_numpy(),
            index=public_values.index,
        )
        for position in range(public_values.shape[1])
    ]
    materialized = pd.concat(columns, axis=1)
    materialized.columns = public_values.columns
    materialized.to_parquet(
        root / f"{script_name}-{engine}.parquet",
        index=True,
    )


def fingerprint(frame, *, order_sensitive: bool = True) -> str:
    """Hash the complete public result under explicit row-order semantics.

    The digest covers column order/names/dtypes, index type/names/dtypes and
    values, active geometry/CRS metadata, every attribute value, geometry
    topology and their row association. Constructive serialization order is
    normalized. Floating metrics use the repository fp64 contract of seven
    significant digits; coordinates use a tighter eleven digits. Exact
    component hashes remain available through diagnostics. Unsupported values
    raise instead of weakening the correctness oracle.
    """
    if not isinstance(frame, (pd.DataFrame, pd.Series)):
        raise TypeError("result fingerprint requires a pandas Series or DataFrame")
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()

    to_wkb = getattr(frame, "to_wkb", None)
    public_values = to_wkb(hex=False) if callable(to_wkb) else frame
    geometry_positions = _geometry_column_positions(frame, public_values)
    logical_dtypes = _logical_schema_descriptors(
        public_values,
        list(frame.dtypes),
        semantic_overrides={position: "geometry" for position in geometry_positions},
    )

    digest = hashlib.sha256()
    semantics = "ordered" if order_sensitive else "unordered"
    digest.update(_pack(b"v", _RESULT_FINGERPRINT_VERSION.encode("ascii")))
    digest.update(_pack(b"o", semantics.encode("ascii")))
    digest.update(_pack(b"c", _canonical_value(list(frame.columns))))
    digest.update(_pack(b"d", b"".join(logical_dtypes)))

    index = frame.index
    digest.update(_pack(b"I", type(index).__name__.encode("utf-8")))
    digest.update(_pack(b"N", _canonical_value(list(index.names))))
    index_frame = index.to_frame(index=False)
    logical_index_dtypes = _logical_schema_descriptors(
        index_frame, list(index_frame.dtypes)
    )
    digest.update(
        _pack(
            b"J",
            b"".join(logical_index_dtypes),
        )
    )

    geometry = getattr(frame, "geometry", None) if callable(to_wkb) else None
    geometry_name = getattr(geometry, "name", None)
    digest.update(_pack(b"g", _canonical_value(geometry_name)))
    crs = getattr(frame, "crs", None) if geometry_name is not None else None
    crs_text = crs.to_wkt() if crs is not None else None
    digest.update(_pack(b"r", _canonical_value(crs_text)))
    digest.update(
        _pack(
            b"G",
            _canonical_value(geometry_positions),
        )
    )

    _emit_fingerprint_diagnostics(
        frame,
        public_values,
        index_frame,
        geometry_name=geometry_name,
        crs_text=crs_text,
        order_sensitive=order_sensitive,
    )
    _dump_fingerprint_values(public_values)
    row_digests: list[bytes] = []
    geometry_position_set = set(geometry_positions)
    index_rows = index_frame.itertuples(index=False, name=None)
    value_rows = public_values.itertuples(index=False, name=None)
    for index_values, values in zip(index_rows, value_rows, strict=True):
        row = hashlib.sha256()
        row.update(_pack(b"i", _canonical_value(tuple(index_values))))
        encoded_values: list[bytes] = []
        for position, value in enumerate(values):
            if position in geometry_position_set and isinstance(
                value, (bytes, bytearray, memoryview)
            ):
                encoded = _canonical_geometry_wkb(bytes(value))
            else:
                encoded = _canonical_value(value, tolerant_floats=True)
            encoded_values.append(_pack(b"e", encoded))
        row.update(_pack(b"v", _pack(b"u", b"".join(encoded_values))))
        row_digest = row.digest()
        if order_sensitive:
            digest.update(_pack(b"w", row_digest))
        else:
            row_digests.append(row_digest)
    if not order_sensitive:
        for row_digest in sorted(row_digests):
            digest.update(_pack(b"w", row_digest))

    return f"{_RESULT_FINGERPRINT_VERSION}:{semantics}:sha256={digest.hexdigest()}"
