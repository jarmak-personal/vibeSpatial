"""Tests for GPU property extraction from GeoJSON Feature objects.

Tests the ``vibespatial.io.gpu_parse.properties`` module which extracts
numeric, boolean, and null property values directly on GPU.  Each test
builds a minimal GeoJSON FeatureCollection byte string, computes the
structural arrays (quote_parity, bracket_depth) using the gpu_parse
primitives, derives feature boundaries from the depth array, and then
calls the property extraction functions.

The tests exercise the internal helper functions directly
(``_find_property_keys``, ``_classify_values``, ``_extract_boolean_column``,
``_colon_positions_from_key_hits``) as well as the top-level
``extract_gpu_properties`` and ``infer_property_schema`` entry points.

All assertions operate on host-side values obtained via ``cp.asnumpy()``.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

try:
    import cupy as cp

    HAS_GPU = True
except (ImportError, ModuleNotFoundError):
    HAS_GPU = False

needs_gpu = pytest.mark.skipif(not HAS_GPU, reason="GPU not available")

# The actual bracket depth at which property keys appear in a standard
# GeoJSON FeatureCollection:
#   FeatureCollection{1} > features[2] > Feature{3} > properties{4}
# Property keys and their colons are at depth 4.
PROPERTY_DEPTH = 4


def test_gpu_properties_has_no_raw_cupy_scalar_syncs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "src" / "vibespatial" / "io" / "gpu_parse" / "properties.py"
    tree = ast.parse(path.read_text(), filename=str(path))

    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "item":
            offenders.append(f"{path.relative_to(repo_root)}:{node.lineno}: .item()")
            continue
        if (
            isinstance(func, ast.Name)
            and func.id in {"bool", "int", "float"}
            and node.args
            and isinstance(node.args[0], ast.Call)
            and isinstance(node.args[0].func, ast.Attribute)
            and isinstance(node.args[0].func.value, ast.Name)
            and node.args[0].func.value.id == "cp"
        ):
            offenders.append(
                f"{path.relative_to(repo_root)}:{node.lineno}: {func.id}(cp.*)"
            )

    assert offenders == []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_device(data: bytes) -> cp.ndarray:
    """Convert a Python bytes object to a device-resident uint8 array."""
    return cp.asarray(np.frombuffer(data, dtype=np.uint8))


def _structural_arrays(data: bytes):
    """Compute quote_parity and bracket_depth for a byte string.

    Returns (d_bytes, d_qp, d_depth) all device-resident.
    """
    from vibespatial.io.gpu_parse import bracket_depth, quote_parity

    d_bytes = _to_device(data)
    d_qp = quote_parity(d_bytes)
    d_depth = bracket_depth(d_bytes, d_qp)
    return d_bytes, d_qp, d_depth


def _feature_boundaries(d_bytes, d_depth):
    """Find feature start/end positions from structural arrays.

    Features in GeoJSON FeatureCollection are objects at depth 3.
    Feature start: '{' where depth == 3
    Feature end: '}' where depth == 2  (closing brace drops depth from 3 to 2)

    Returns (d_feature_starts, d_feature_ends) as int64 device arrays.
    d_feature_ends points one byte PAST the closing '}' (exclusive end).
    """
    h_bytes = cp.asnumpy(d_bytes)
    h_depth = cp.asnumpy(d_depth)

    n = len(h_bytes)
    starts = []
    ends = []

    for i in range(n):
        if h_bytes[i] == ord("{") and h_depth[i] == 3:
            starts.append(i)
        elif h_bytes[i] == ord("}") and h_depth[i] == 2:
            ends.append(i + 1)

    d_starts = cp.array(starts, dtype=cp.int64) if starts else cp.empty(0, dtype=cp.int64)
    d_ends = cp.array(ends, dtype=cp.int64) if ends else cp.empty(0, dtype=cp.int64)
    return d_starts, d_ends


def _build_feature_collection(*features: str) -> bytes:
    """Build a minimal GeoJSON FeatureCollection from Feature JSON strings.

    Each feature string should be a complete Feature object, e.g.:
        '{"type":"Feature","geometry":null,"properties":{"pop":999}}'

    Returns the full FeatureCollection as bytes.
    """
    inner = ",".join(features)
    return f'{{"type":"FeatureCollection","features":[{inner}]}}'.encode("ascii")


def _minimal_feature(properties_json: str) -> str:
    """Build a minimal Feature with null geometry and given properties JSON."""
    return f'{{"type":"Feature","geometry":null,"properties":{properties_json}}}'


# ===========================================================================
# _find_property_keys tests
# ===========================================================================


class TestFindPropertyKeys:
    """Tests for _find_property_keys at the correct bracket depth."""

    @needs_gpu
    def test_single_key_single_feature(self):
        """Finds a single property key in one feature."""
        from vibespatial.io.gpu_parse.properties import _find_property_keys

        data = _build_feature_collection(
            _minimal_feature('{"population":999}'),
        )
        d_bytes, d_qp, d_depth = _structural_arrays(data)

        positions = _find_property_keys(
            d_bytes, d_qp, d_depth, "population",
            property_depth=PROPERTY_DEPTH,
        )
        h_pos = cp.asnumpy(positions)

        assert len(h_pos) == 1
        # Position should point to the opening quote of "population"
        pos = int(h_pos[0])
        assert data[pos : pos + 1] == b'"'
        assert data[pos + 1 : pos + 11] == b"population"

    @needs_gpu
    def test_key_in_multiple_features(self):
        """Finds the same key in each of three features."""
        from vibespatial.io.gpu_parse.properties import _find_property_keys

        data = _build_feature_collection(
            _minimal_feature('{"val":1}'),
            _minimal_feature('{"val":2}'),
            _minimal_feature('{"val":3}'),
        )
        d_bytes, d_qp, d_depth = _structural_arrays(data)

        positions = _find_property_keys(
            d_bytes, d_qp, d_depth, "val",
            property_depth=PROPERTY_DEPTH,
        )
        h_pos = cp.asnumpy(positions)

        assert len(h_pos) == 3
        # Each position should point to the opening quote of "val"
        for p in h_pos:
            pos = int(p)
            assert data[pos : pos + 1] == b'"'
            assert data[pos + 1 : pos + 4] == b"val"

    @needs_gpu
    def test_key_not_found(self):
        """Returns empty array when key does not exist."""
        from vibespatial.io.gpu_parse.properties import _find_property_keys

        data = _build_feature_collection(
            _minimal_feature('{"other":1}'),
        )
        d_bytes, d_qp, d_depth = _structural_arrays(data)

        positions = _find_property_keys(
            d_bytes, d_qp, d_depth, "missing",
            property_depth=PROPERTY_DEPTH,
        )
        h_pos = cp.asnumpy(positions)

        assert len(h_pos) == 0

    @needs_gpu
    def test_key_at_wrong_depth_ignored(self):
        """Keys at other depths (e.g., Feature-level 'type') are not found."""
        from vibespatial.io.gpu_parse.properties import _find_property_keys

        data = _build_feature_collection(
            _minimal_feature('{"type":"Point"}'),
        )
        d_bytes, d_qp, d_depth = _structural_arrays(data)

        # "type" at depth 3 (Feature level) should not be found at depth 4
        positions = _find_property_keys(
            d_bytes, d_qp, d_depth, "type",
            property_depth=PROPERTY_DEPTH,
        )
        h_pos = cp.asnumpy(positions)

        # Should find the "type" inside properties at depth 4
        assert len(h_pos) == 1
        pos = int(h_pos[0])
        # Verify this is inside the properties object, not the Feature "type" key
        # The Feature "type":"Feature" is at depth 3, not 4
        assert data[pos + 1 : pos + 5] == b"type"

    @needs_gpu
    def test_multiple_keys_in_same_feature(self):
        """Finds each distinct key when searching for them individually."""
        from vibespatial.io.gpu_parse.properties import _find_property_keys

        data = _build_feature_collection(
            _minimal_feature('{"a":1,"b":2,"c":3}'),
        )
        d_bytes, d_qp, d_depth = _structural_arrays(data)

        for key_name in ("a", "b", "c"):
            positions = _find_property_keys(
                d_bytes, d_qp, d_depth, key_name,
                property_depth=PROPERTY_DEPTH,
            )
            h_pos = cp.asnumpy(positions)
            assert len(h_pos) == 1, f"Expected 1 hit for key '{key_name}'"

    @needs_gpu
    def test_single_char_key(self):
        """Single-character property key found correctly."""
        from vibespatial.io.gpu_parse.properties import _find_property_keys

        data = _build_feature_collection(
            _minimal_feature('{"x":7}'),
        )
        d_bytes, d_qp, d_depth = _structural_arrays(data)

        positions = _find_property_keys(
            d_bytes, d_qp, d_depth, "x",
            property_depth=PROPERTY_DEPTH,
        )
        h_pos = cp.asnumpy(positions)
        assert len(h_pos) == 1

    @needs_gpu
    def test_long_key_name(self):
        """Long property key name found correctly."""
        from vibespatial.io.gpu_parse.properties import _find_property_keys

        key = "this_is_a_very_long_property_name"
        data = _build_feature_collection(
            _minimal_feature(f'{{"{key}":123}}'),
        )
        d_bytes, d_qp, d_depth = _structural_arrays(data)

        positions = _find_property_keys(
            d_bytes, d_qp, d_depth, key,
            property_depth=PROPERTY_DEPTH,
        )
        h_pos = cp.asnumpy(positions)
        assert len(h_pos) == 1


# ===========================================================================
# _colon_positions_from_key_hits tests
# ===========================================================================


class TestColonPositions:
    """Tests for _colon_positions_from_key_hits."""

    @needs_gpu
    def test_basic_offset_computation(self):
        """Colon offset is len(key_name) + 2 from key position."""
        from vibespatial.io.gpu_parse.properties import _colon_positions_from_key_hits

        # For key "val": pattern is "val": (6 bytes)
        # Opening quote at position 10 -> colon at 10 + 3 + 2 = 15
        d_key_pos = cp.array([10, 50], dtype=cp.int64)
        d_colon_pos = _colon_positions_from_key_hits(d_key_pos, "val")
        h_colon = cp.asnumpy(d_colon_pos)

        # colon_offset = len("val") + 2 = 5
        assert h_colon[0] == 15
        assert h_colon[1] == 55

    @needs_gpu
    def test_single_char_key_offset(self):
        """Colon offset for single-char key is 3."""
        from vibespatial.io.gpu_parse.properties import _colon_positions_from_key_hits

        d_key_pos = cp.array([0], dtype=cp.int64)
        d_colon_pos = _colon_positions_from_key_hits(d_key_pos, "x")
        h_colon = cp.asnumpy(d_colon_pos)
        # "x": -> offset = 1 + 2 = 3
        assert h_colon[0] == 3

    @needs_gpu
    def test_actual_colon_position_in_data(self):
        """Verify colon positions point to actual ':' bytes in real data."""
        from vibespatial.io.gpu_parse.properties import (
            _colon_positions_from_key_hits,
            _find_property_keys,
        )

        data = _build_feature_collection(
            _minimal_feature('{"population":999}'),
        )
        d_bytes, d_qp, d_depth = _structural_arrays(data)

        d_key_pos = _find_property_keys(
            d_bytes, d_qp, d_depth, "population",
            property_depth=PROPERTY_DEPTH,
        )
        d_colon_pos = _colon_positions_from_key_hits(d_key_pos, "population")

        h_colon = cp.asnumpy(d_colon_pos)
        assert len(h_colon) == 1
        # The byte at the colon position should be ':'
        colon_byte = data[int(h_colon[0])]
        assert colon_byte == ord(":"), f"Expected ':', got '{chr(colon_byte)}'"


# ===========================================================================
# _classify_values tests
# ===========================================================================


class TestClassifyValues:
    """Tests for the classify_property_values NVRTC kernel."""

    @needs_gpu
    def test_numeric_value(self):
        """Classifies a numeric value (digit after colon) as VTYPE_NUMBER."""
        from vibespatial.io.gpu_parse.properties import (
            VTYPE_NUMBER,
            _classify_values,
            _colon_positions_from_key_hits,
            _find_property_keys,
        )

        data = _build_feature_collection(
            _minimal_feature('{"val":42}'),
        )
        d_bytes, d_qp, d_depth = _structural_arrays(data)

        d_key_pos = _find_property_keys(
            d_bytes, d_qp, d_depth, "val",
            property_depth=PROPERTY_DEPTH,
        )
        d_colon_pos = _colon_positions_from_key_hits(d_key_pos, "val")
        d_types = _classify_values(d_bytes, d_colon_pos)

        h_types = cp.asnumpy(d_types)
        assert len(h_types) == 1
        assert h_types[0] == VTYPE_NUMBER

    @needs_gpu
    def test_negative_numeric_value(self):
        """Classifies negative number (starts with '-') as VTYPE_NUMBER."""
        from vibespatial.io.gpu_parse.properties import (
            VTYPE_NUMBER,
            _classify_values,
            _colon_positions_from_key_hits,
            _find_property_keys,
        )

        data = _build_feature_collection(
            _minimal_feature('{"temp":-42.7}'),
        )
        d_bytes, d_qp, d_depth = _structural_arrays(data)

        d_key_pos = _find_property_keys(
            d_bytes, d_qp, d_depth, "temp",
            property_depth=PROPERTY_DEPTH,
        )
        d_colon_pos = _colon_positions_from_key_hits(d_key_pos, "temp")
        d_types = _classify_values(d_bytes, d_colon_pos)

        assert cp.asnumpy(d_types)[0] == VTYPE_NUMBER

    @needs_gpu
    def test_boolean_true(self):
        """Classifies 'true' as VTYPE_BOOLEAN."""
        from vibespatial.io.gpu_parse.properties import (
            VTYPE_BOOLEAN,
            _classify_values,
            _colon_positions_from_key_hits,
            _find_property_keys,
        )

        data = _build_feature_collection(
            _minimal_feature('{"ok":true}'),
        )
        d_bytes, d_qp, d_depth = _structural_arrays(data)

        d_key_pos = _find_property_keys(
            d_bytes, d_qp, d_depth, "ok",
            property_depth=PROPERTY_DEPTH,
        )
        d_colon_pos = _colon_positions_from_key_hits(d_key_pos, "ok")
        d_types = _classify_values(d_bytes, d_colon_pos)

        assert cp.asnumpy(d_types)[0] == VTYPE_BOOLEAN

    @needs_gpu
    def test_boolean_false(self):
        """Classifies 'false' as VTYPE_BOOLEAN."""
        from vibespatial.io.gpu_parse.properties import (
            VTYPE_BOOLEAN,
            _classify_values,
            _colon_positions_from_key_hits,
            _find_property_keys,
        )

        data = _build_feature_collection(
            _minimal_feature('{"ok":false}'),
        )
        d_bytes, d_qp, d_depth = _structural_arrays(data)

        d_key_pos = _find_property_keys(
            d_bytes, d_qp, d_depth, "ok",
            property_depth=PROPERTY_DEPTH,
        )
        d_colon_pos = _colon_positions_from_key_hits(d_key_pos, "ok")
        d_types = _classify_values(d_bytes, d_colon_pos)

        assert cp.asnumpy(d_types)[0] == VTYPE_BOOLEAN

    @needs_gpu
    def test_null_value(self):
        """Classifies 'null' as VTYPE_NULL."""
        from vibespatial.io.gpu_parse.properties import (
            VTYPE_NULL,
            _classify_values,
            _colon_positions_from_key_hits,
            _find_property_keys,
        )

        data = _build_feature_collection(
            _minimal_feature('{"val":null}'),
        )
        d_bytes, d_qp, d_depth = _structural_arrays(data)

        d_key_pos = _find_property_keys(
            d_bytes, d_qp, d_depth, "val",
            property_depth=PROPERTY_DEPTH,
        )
        d_colon_pos = _colon_positions_from_key_hits(d_key_pos, "val")
        d_types = _classify_values(d_bytes, d_colon_pos)

        assert cp.asnumpy(d_types)[0] == VTYPE_NULL

    @needs_gpu
    def test_string_value(self):
        """Classifies a quoted string value as VTYPE_STRING."""
        from vibespatial.io.gpu_parse.properties import (
            VTYPE_STRING,
            _classify_values,
            _colon_positions_from_key_hits,
            _find_property_keys,
        )

        data = _build_feature_collection(
            _minimal_feature('{"name":"test"}'),
        )
        d_bytes, d_qp, d_depth = _structural_arrays(data)

        d_key_pos = _find_property_keys(
            d_bytes, d_qp, d_depth, "name",
            property_depth=PROPERTY_DEPTH,
        )
        d_colon_pos = _colon_positions_from_key_hits(d_key_pos, "name")
        d_types = _classify_values(d_bytes, d_colon_pos)

        assert cp.asnumpy(d_types)[0] == VTYPE_STRING

    @needs_gpu
    def test_complex_object_value(self):
        """Classifies a nested object as VTYPE_COMPLEX."""
        from vibespatial.io.gpu_parse.properties import (
            VTYPE_COMPLEX,
            _classify_values,
            _colon_positions_from_key_hits,
            _find_property_keys,
        )

        data = _build_feature_collection(
            _minimal_feature('{"meta":{"nested":true}}'),
        )
        d_bytes, d_qp, d_depth = _structural_arrays(data)

        d_key_pos = _find_property_keys(
            d_bytes, d_qp, d_depth, "meta",
            property_depth=PROPERTY_DEPTH,
        )
        d_colon_pos = _colon_positions_from_key_hits(d_key_pos, "meta")
        d_types = _classify_values(d_bytes, d_colon_pos)

        assert cp.asnumpy(d_types)[0] == VTYPE_COMPLEX

    @needs_gpu
    def test_complex_array_value(self):
        """Classifies an array value as VTYPE_COMPLEX."""
        from vibespatial.io.gpu_parse.properties import (
            VTYPE_COMPLEX,
            _classify_values,
            _colon_positions_from_key_hits,
            _find_property_keys,
        )

        data = _build_feature_collection(
            _minimal_feature('{"tags":[1,2,3]}'),
        )
        d_bytes, d_qp, d_depth = _structural_arrays(data)

        d_key_pos = _find_property_keys(
            d_bytes, d_qp, d_depth, "tags",
            property_depth=PROPERTY_DEPTH,
        )
        d_colon_pos = _colon_positions_from_key_hits(d_key_pos, "tags")
        d_types = _classify_values(d_bytes, d_colon_pos)

        assert cp.asnumpy(d_types)[0] == VTYPE_COMPLEX

    @needs_gpu
    def test_whitespace_before_value(self):
        """Whitespace between colon and value is correctly skipped."""
        from vibespatial.io.gpu_parse.properties import (
            VTYPE_NUMBER,
            _classify_values,
            _colon_positions_from_key_hits,
            _find_property_keys,
        )

        data = _build_feature_collection(
            _minimal_feature('{"val": 42}'),
        )
        d_bytes, d_qp, d_depth = _structural_arrays(data)

        d_key_pos = _find_property_keys(
            d_bytes, d_qp, d_depth, "val",
            property_depth=PROPERTY_DEPTH,
        )
        d_colon_pos = _colon_positions_from_key_hits(d_key_pos, "val")
        d_types = _classify_values(d_bytes, d_colon_pos)

        assert cp.asnumpy(d_types)[0] == VTYPE_NUMBER

    @needs_gpu
    def test_mixed_types_across_features(self):
        """Classifies different types for the same key across features."""
        from vibespatial.io.gpu_parse.properties import (
            VTYPE_BOOLEAN,
            VTYPE_NULL,
            VTYPE_NUMBER,
            _classify_values,
            _colon_positions_from_key_hits,
            _find_property_keys,
        )

        data = _build_feature_collection(
            _minimal_feature('{"val":42}'),
            _minimal_feature('{"val":null}'),
            _minimal_feature('{"val":true}'),
        )
        d_bytes, d_qp, d_depth = _structural_arrays(data)

        d_key_pos = _find_property_keys(
            d_bytes, d_qp, d_depth, "val",
            property_depth=PROPERTY_DEPTH,
        )
        d_colon_pos = _colon_positions_from_key_hits(d_key_pos, "val")
        d_types = _classify_values(d_bytes, d_colon_pos)

        h_types = cp.asnumpy(d_types)
        assert len(h_types) == 3
        assert h_types[0] == VTYPE_NUMBER
        assert h_types[1] == VTYPE_NULL
        assert h_types[2] == VTYPE_BOOLEAN

    @needs_gpu
    def test_all_types_in_single_feature(self):
        """Classifies all supported types from different keys in one feature."""
        from vibespatial.io.gpu_parse.properties import (
            VTYPE_BOOLEAN,
            VTYPE_NULL,
            VTYPE_NUMBER,
            VTYPE_STRING,
            _classify_values,
            _colon_positions_from_key_hits,
            _find_property_keys,
        )

        data = _build_feature_collection(
            _minimal_feature(
                '{"count":42,"active":true,"name":"test","x":null}'
            ),
        )
        d_bytes, d_qp, d_depth = _structural_arrays(data)

        expected = {
            "count": VTYPE_NUMBER,
            "active": VTYPE_BOOLEAN,
            "name": VTYPE_STRING,
            "x": VTYPE_NULL,
        }

        for key_name, expected_type in expected.items():
            d_key_pos = _find_property_keys(
                d_bytes, d_qp, d_depth, key_name,
                property_depth=PROPERTY_DEPTH,
            )
            d_colon_pos = _colon_positions_from_key_hits(d_key_pos, key_name)
            d_types = _classify_values(d_bytes, d_colon_pos)
            actual = cp.asnumpy(d_types)[0]
            assert actual == expected_type, (
                f"Key '{key_name}': expected type {expected_type}, got {actual}"
            )


# ===========================================================================
# _extract_boolean_column tests
# ===========================================================================


class TestExtractBooleanColumn:
    """Tests for the extract_booleans NVRTC kernel."""

    @needs_gpu
    def test_true_value(self):
        """Boolean 'true' extracted as uint8 value 1."""
        from vibespatial.io.gpu_parse.properties import (
            _colon_positions_from_key_hits,
            _extract_boolean_column,
            _find_property_keys,
        )

        data = _build_feature_collection(
            _minimal_feature('{"active":true}'),
        )
        d_bytes, d_qp, d_depth = _structural_arrays(data)

        d_key_pos = _find_property_keys(
            d_bytes, d_qp, d_depth, "active",
            property_depth=PROPERTY_DEPTH,
        )
        d_colon_pos = _colon_positions_from_key_hits(d_key_pos, "active")
        d_bools = _extract_boolean_column(d_bytes, d_colon_pos)

        h_bools = cp.asnumpy(d_bools)
        assert h_bools.dtype == np.uint8
        assert len(h_bools) == 1
        assert h_bools[0] == 1

    @needs_gpu
    def test_false_value(self):
        """Boolean 'false' extracted as uint8 value 0."""
        from vibespatial.io.gpu_parse.properties import (
            _colon_positions_from_key_hits,
            _extract_boolean_column,
            _find_property_keys,
        )

        data = _build_feature_collection(
            _minimal_feature('{"active":false}'),
        )
        d_bytes, d_qp, d_depth = _structural_arrays(data)

        d_key_pos = _find_property_keys(
            d_bytes, d_qp, d_depth, "active",
            property_depth=PROPERTY_DEPTH,
        )
        d_colon_pos = _colon_positions_from_key_hits(d_key_pos, "active")
        d_bools = _extract_boolean_column(d_bytes, d_colon_pos)

        assert cp.asnumpy(d_bools)[0] == 0

    @needs_gpu
    def test_mixed_true_false_across_features(self):
        """Multiple features produce correct columnar boolean array."""
        from vibespatial.io.gpu_parse.properties import (
            _colon_positions_from_key_hits,
            _extract_boolean_column,
            _find_property_keys,
        )

        data = _build_feature_collection(
            _minimal_feature('{"flag":true}'),
            _minimal_feature('{"flag":false}'),
            _minimal_feature('{"flag":true}'),
        )
        d_bytes, d_qp, d_depth = _structural_arrays(data)

        d_key_pos = _find_property_keys(
            d_bytes, d_qp, d_depth, "flag",
            property_depth=PROPERTY_DEPTH,
        )
        d_colon_pos = _colon_positions_from_key_hits(d_key_pos, "flag")
        d_bools = _extract_boolean_column(d_bytes, d_colon_pos)

        h_bools = cp.asnumpy(d_bools)
        assert len(h_bools) == 3
        np.testing.assert_array_equal(h_bools, [1, 0, 1])

    @needs_gpu
    def test_five_features_alternating(self):
        """Five features with alternating true/false values."""
        from vibespatial.io.gpu_parse.properties import (
            _colon_positions_from_key_hits,
            _extract_boolean_column,
            _find_property_keys,
        )

        features = []
        for i in range(5):
            val = "true" if i % 2 == 0 else "false"
            features.append(_minimal_feature(f'{{"b":{val}}}'))

        data = _build_feature_collection(*features)
        d_bytes, d_qp, d_depth = _structural_arrays(data)

        d_key_pos = _find_property_keys(
            d_bytes, d_qp, d_depth, "b",
            property_depth=PROPERTY_DEPTH,
        )
        d_colon_pos = _colon_positions_from_key_hits(d_key_pos, "b")
        d_bools = _extract_boolean_column(d_bytes, d_colon_pos)

        h_bools = cp.asnumpy(d_bools)
        assert len(h_bools) == 5
        np.testing.assert_array_equal(h_bools, [1, 0, 1, 0, 1])

    @needs_gpu
    def test_whitespace_before_boolean(self):
        """Whitespace between colon and boolean is handled by kernel."""
        from vibespatial.io.gpu_parse.properties import (
            _colon_positions_from_key_hits,
            _extract_boolean_column,
            _find_property_keys,
        )

        data = _build_feature_collection(
            _minimal_feature('{"ok": true}'),
        )
        d_bytes, d_qp, d_depth = _structural_arrays(data)

        d_key_pos = _find_property_keys(
            d_bytes, d_qp, d_depth, "ok",
            property_depth=PROPERTY_DEPTH,
        )
        d_colon_pos = _colon_positions_from_key_hits(d_key_pos, "ok")
        d_bools = _extract_boolean_column(d_bytes, d_colon_pos)

        assert cp.asnumpy(d_bools)[0] == 1

    @needs_gpu
    def test_multiple_boolean_properties(self):
        """Multiple boolean properties extracted independently."""
        from vibespatial.io.gpu_parse.properties import (
            _colon_positions_from_key_hits,
            _extract_boolean_column,
            _find_property_keys,
        )

        data = _build_feature_collection(
            _minimal_feature('{"a":true,"b":false}'),
        )
        d_bytes, d_qp, d_depth = _structural_arrays(data)

        for key, expected in [("a", 1), ("b", 0)]:
            d_key_pos = _find_property_keys(
                d_bytes, d_qp, d_depth, key,
                property_depth=PROPERTY_DEPTH,
            )
            d_colon_pos = _colon_positions_from_key_hits(d_key_pos, key)
            d_bools = _extract_boolean_column(d_bytes, d_colon_pos)
            assert cp.asnumpy(d_bools)[0] == expected, (
                f"Key '{key}': expected {expected}, got {cp.asnumpy(d_bools)[0]}"
            )


# ===========================================================================
# Boolean extraction with type filtering (simulating extract_gpu_properties)
# ===========================================================================


class TestBooleanExtractionWithTypeFilter:
    """Tests for boolean extraction using the classify+filter pattern.

    This mirrors the logic in extract_gpu_properties: find keys, classify
    value types, filter to boolean positions, then extract booleans.
    """

    @needs_gpu
    def test_boolean_filtered_from_mixed_types(self):
        """Only boolean-typed entries are passed to the boolean kernel."""
        from vibespatial.io.gpu_parse.properties import (
            VTYPE_BOOLEAN,
            _classify_values,
            _colon_positions_from_key_hits,
            _extract_boolean_column,
            _find_property_keys,
        )

        # Three features: boolean, null, boolean
        data = _build_feature_collection(
            _minimal_feature('{"flag":true}'),
            _minimal_feature('{"flag":null}'),
            _minimal_feature('{"flag":false}'),
        )
        d_bytes, d_qp, d_depth = _structural_arrays(data)

        d_key_pos = _find_property_keys(
            d_bytes, d_qp, d_depth, "flag",
            property_depth=PROPERTY_DEPTH,
        )
        d_colon_pos = _colon_positions_from_key_hits(d_key_pos, "flag")
        d_vtypes = _classify_values(d_bytes, d_colon_pos)

        # Filter to boolean-only positions
        d_is_bool = (d_vtypes == VTYPE_BOOLEAN)
        d_bool_colon_pos = d_colon_pos[d_is_bool]
        d_bools = _extract_boolean_column(d_bytes, d_bool_colon_pos)

        h_bools = cp.asnumpy(d_bools)
        # Only 2 boolean values (features 0 and 2)
        assert len(h_bools) == 2
        assert h_bools[0] == 1  # true
        assert h_bools[1] == 0  # false


# ===========================================================================
# Schema inference tests (infer_property_schema)
# ===========================================================================


class TestSchemaInference:
    """Tests for infer_property_schema().

    Note: The schema inference function walks host-side bytes looking for
    property keys at the specified depth.  Its key detection relies on
    finding quote bytes where quote_parity == 0 at the property depth.
    """

    @needs_gpu
    def test_empty_features(self):
        """Schema inference with zero features returns empty dict."""
        from vibespatial.io.gpu_parse.properties import infer_property_schema

        data = b'{"type":"FeatureCollection","features":[]}'
        d_bytes, d_qp, d_depth = _structural_arrays(data)
        d_feat_starts = cp.empty(0, dtype=cp.int64)
        d_feat_ends = cp.empty(0, dtype=cp.int64)

        schema = infer_property_schema(
            d_bytes, d_qp, d_depth, d_feat_starts, d_feat_ends,
            property_depth=PROPERTY_DEPTH,
        )
        assert schema == {}

    @needs_gpu
    def test_empty_properties_object(self):
        """Feature with empty properties produces empty schema."""
        from vibespatial.io.gpu_parse.properties import infer_property_schema

        data = _build_feature_collection(
            _minimal_feature("{}"),
        )
        d_bytes, d_qp, d_depth = _structural_arrays(data)
        d_feat_starts, d_feat_ends = _feature_boundaries(d_bytes, d_depth)

        schema = infer_property_schema(
            d_bytes, d_qp, d_depth, d_feat_starts, d_feat_ends,
            property_depth=PROPERTY_DEPTH,
        )
        assert schema == {}


# ===========================================================================
# Top-level extract_gpu_properties tests
# ===========================================================================


class TestExtractGpuProperties:
    """Tests for the top-level extract_gpu_properties entry point."""

    @needs_gpu
    def test_no_features_returns_empty(self):
        """Zero features produce empty result dict."""
        from vibespatial.io.gpu_parse.properties import extract_gpu_properties

        data = b'{"type":"FeatureCollection","features":[]}'
        d_bytes, d_qp, d_depth = _structural_arrays(data)
        d_feat_starts = cp.empty(0, dtype=cp.int64)
        d_feat_ends = cp.empty(0, dtype=cp.int64)

        result = extract_gpu_properties(
            d_bytes, d_feat_starts, d_feat_ends, d_qp, d_depth,
            property_depth=PROPERTY_DEPTH,
        )
        assert result == {}

    @needs_gpu
    def test_empty_properties_returns_empty(self):
        """Feature with empty properties object produces empty result."""
        from vibespatial.io.gpu_parse.properties import extract_gpu_properties

        data = _build_feature_collection(
            _minimal_feature("{}"),
        )
        d_bytes, d_qp, d_depth = _structural_arrays(data)
        d_feat_starts, d_feat_ends = _feature_boundaries(d_bytes, d_depth)

        result = extract_gpu_properties(
            d_bytes, d_feat_starts, d_feat_ends, d_qp, d_depth,
            property_depth=PROPERTY_DEPTH,
        )
        assert result == {}

    @needs_gpu
    def test_mixed_numeric_boolean_count_fences_are_runtime_observable(self):
        """Type-count branch gates should be named runtime D2H fences."""
        from vibespatial.cuda._runtime import (
            get_d2h_transfer_events,
            reset_d2h_transfer_count,
        )
        from vibespatial.io.gpu_parse.properties import extract_gpu_properties

        data = _build_feature_collection(
            _minimal_feature('{"score":1.5,"flag":true}'),
            _minimal_feature('{"score":null,"flag":false}'),
            _minimal_feature('{"score":2.5,"flag":null}'),
        )
        d_bytes, d_qp, d_depth = _structural_arrays(data)
        d_feat_starts, d_feat_ends = _feature_boundaries(d_bytes, d_depth)

        reset_d2h_transfer_count()
        result = extract_gpu_properties(
            d_bytes, d_feat_starts, d_feat_ends, d_qp, d_depth,
            property_depth=PROPERTY_DEPTH,
        )
        reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

        assert "geojson properties numeric-value count scalar fence" in reasons
        assert "geojson properties boolean-value count scalar fence" in reasons
        np.testing.assert_allclose(
            cp.asnumpy(result["score"]),
            np.asarray([1.5, np.nan, 2.5]),
            equal_nan=True,
        )
        assert cp.asnumpy(result["flag"]).tolist() == [1, 0, 0]


# ===========================================================================
# Feature boundary helper tests
# ===========================================================================


class TestFeatureBoundaryHelper:
    """Tests for the _feature_boundaries test helper."""

    @needs_gpu
    def test_single_feature_boundaries(self):
        """Single feature produces one start/end pair."""
        data = _build_feature_collection(
            _minimal_feature('{"val":1}'),
        )
        d_bytes, _, d_depth = _structural_arrays(data)
        d_starts, d_ends = _feature_boundaries(d_bytes, d_depth)

        h_starts = cp.asnumpy(d_starts)
        h_ends = cp.asnumpy(d_ends)

        assert len(h_starts) == 1
        assert len(h_ends) == 1
        assert data[int(h_starts[0])] == ord("{")
        assert data[int(h_ends[0]) - 1] == ord("}")

    @needs_gpu
    def test_three_feature_boundaries(self):
        """Three features produce three start/end pairs."""
        data = _build_feature_collection(
            _minimal_feature('{"a":1}'),
            _minimal_feature('{"b":2}'),
            _minimal_feature('{"c":3}'),
        )
        d_bytes, _, d_depth = _structural_arrays(data)
        d_starts, d_ends = _feature_boundaries(d_bytes, d_depth)

        h_starts = cp.asnumpy(d_starts)
        h_ends = cp.asnumpy(d_ends)

        assert len(h_starts) == 3
        assert len(h_ends) == 3

        for i in range(3):
            assert data[int(h_starts[i])] == ord("{")
            assert data[int(h_ends[i]) - 1] == ord("}")

        assert h_starts[0] < h_starts[1] < h_starts[2]

    @needs_gpu
    def test_feature_span_contains_properties(self):
        """Each feature span contains the property data."""
        data = _build_feature_collection(
            _minimal_feature('{"pop":42}'),
        )
        d_bytes, _, d_depth = _structural_arrays(data)
        d_starts, d_ends = _feature_boundaries(d_bytes, d_depth)

        start = int(cp.asnumpy(d_starts)[0])
        end = int(cp.asnumpy(d_ends)[0])
        feature_bytes = data[start:end]

        assert b'"pop"' in feature_bytes
        assert b"42" in feature_bytes

    @needs_gpu
    def test_empty_feature_collection(self):
        """Empty features array produces empty boundaries."""
        data = b'{"type":"FeatureCollection","features":[]}'
        d_bytes, _, d_depth = _structural_arrays(data)
        d_starts, d_ends = _feature_boundaries(d_bytes, d_depth)

        assert cp.asnumpy(d_starts).shape[0] == 0
        assert cp.asnumpy(d_ends).shape[0] == 0


# ===========================================================================
# VTYPE constant tests
# ===========================================================================


class TestVtypeConstants:
    """Verify the value type constant values match kernel output codes."""

    def test_constants_defined(self):
        """All VTYPE constants are accessible and have expected values."""
        from vibespatial.io.gpu_parse.properties import (
            VTYPE_BOOLEAN,
            VTYPE_COMPLEX,
            VTYPE_NULL,
            VTYPE_NUMBER,
            VTYPE_STRING,
        )

        assert VTYPE_STRING == 0
        assert VTYPE_BOOLEAN == 1
        assert VTYPE_NULL == 2
        assert VTYPE_NUMBER == 3
        assert VTYPE_COMPLEX == 4
