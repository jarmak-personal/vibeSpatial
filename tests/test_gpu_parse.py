"""Tests for GPU text-parsing primitives in vibespatial.io.gpu_parse.

Tests each primitive in isolation and in composition, verifying correct
structural analysis, numeric parsing, pattern matching, span detection,
and the full pipeline from raw bytes to parsed coordinates.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    import cupy as cp

    HAS_GPU = True
except (ImportError, ModuleNotFoundError):
    HAS_GPU = False

needs_gpu = pytest.mark.skipif(not HAS_GPU, reason="GPU not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_device(data: bytes) -> cp.ndarray:
    """Convert a Python bytes object to a device-resident uint8 array."""
    return cp.asarray(np.frombuffer(data, dtype=np.uint8))


# ===========================================================================
# Structural tests: quote_parity
# ===========================================================================


class TestQuoteParity:
    """Tests for structural.quote_parity()."""

    @needs_gpu
    def test_simple_json(self):
        """Simple JSON: {"a": 1} -- parity toggles at quote boundaries."""
        from vibespatial.io.gpu_parse import quote_parity

        data = b'{"a": 1}'
        d_bytes = _to_device(data)
        d_qp = quote_parity(d_bytes)
        result = cp.asnumpy(d_qp)

        # Position-by-position for {"a": 1}
        # Position: 0:{  1:"  2:a  3:"  4::  5:_  6:1  7:}
        # Toggle:   0    1    0    1    0    0    0    0
        # Cumsum:   0    1    1    2    2    2    2    2
        # Parity:   0    1    1    0    0    0    0    0
        #
        # The quote byte itself gets parity 1 (the cumsum is inclusive).
        # The closing quote byte gets parity 0 (cumsum is even).
        expected = np.array([0, 1, 1, 0, 0, 0, 0, 0], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    @needs_gpu
    def test_nested_json(self):
        """Nested JSON: {"a": {"b": [1, 2]}} -- parity tracks inner keys."""
        from vibespatial.io.gpu_parse import quote_parity

        data = b'{"a": {"b": [1, 2]}}'
        d_bytes = _to_device(data)
        d_qp = quote_parity(d_bytes)
        result = cp.asnumpy(d_qp)

        # Characters:  { " a " :   { " b " :   [ 1 ,   2 ] } }
        # Indices:     0 1 2 3 4 5 6 7 8 9 ...
        # Toggle:      0 1 0 1 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0
        # Cumsum:      0 1 1 2 2 2 2 3 3 4 4 4 4 4 4 4 4 4 4 4
        # Parity:      0 1 1 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0

        # Verify key positions: inside-quote chars have parity 1
        assert result[2] == 1, "'a' is inside quotes"
        assert result[8] == 1, "'b' is inside quotes"
        # Structural chars outside quotes have parity 0
        assert result[0] == 0, "'{' at start outside quotes"
        assert result[4] == 0, "':' outside quotes"
        assert result[12] == 0, "'[' outside quotes"

    @needs_gpu
    def test_escaped_quotes(self):
        r"""Escaped quotes: {"a": "he said \"hello\""} -- parity stays correct."""
        from vibespatial.io.gpu_parse import quote_parity

        data = b'{"a": "he said \\"hello\\""}'
        d_bytes = _to_device(data)
        d_qp = quote_parity(d_bytes)
        result = cp.asnumpy(d_qp)

        # The backslash-escaped quotes inside the string value should NOT
        # toggle parity. The string value bytes should all remain parity=1.
        # The closing unescaped " at position -2 (before }) should toggle
        # back to 0.
        assert result[len(data) - 1] == 0, "'}' at end is outside quotes"
        assert result[0] == 0, "'{' at start is outside quotes"

        # The 'h' in 'hello' should be inside the string value
        hello_pos = data.index(b"hello")
        assert result[hello_pos] == 1, "'h' of 'hello' is inside quotes"

    @needs_gpu
    def test_double_backslash_before_quote(self):
        r"""Double backslash: {"a": "path\\\"end"} -- even backslashes don't prevent toggle."""
        from vibespatial.io.gpu_parse import quote_parity

        # In the raw bytes: "path\\\\"end"
        # Two backslashes (\\\\) = even count, so the following " IS a real quote toggle
        data = b'{"a": "path\\\\\\"end"}'
        d_bytes = _to_device(data)
        d_qp = quote_parity(d_bytes)
        result = cp.asnumpy(d_qp)

        # The final '}' should be outside quotes (parity 0)
        assert result[len(data) - 1] == 0, "closing '}' is outside quotes"
        # The first '{' should be outside quotes
        assert result[0] == 0, "opening '{' is outside quotes"

    @needs_gpu
    def test_empty_input(self):
        """Zero-length byte array returns zero-length result."""
        from vibespatial.io.gpu_parse import quote_parity

        d_bytes = cp.empty(0, dtype=cp.uint8)
        d_qp = quote_parity(d_bytes)
        assert d_qp.shape == (0,)

    @needs_gpu
    def test_no_quotes(self):
        """Input with no quotes: all positions have parity 0."""
        from vibespatial.io.gpu_parse import quote_parity

        data = b"[1, 2, 3]"
        d_bytes = _to_device(data)
        d_qp = quote_parity(d_bytes)
        result = cp.asnumpy(d_qp)

        np.testing.assert_array_equal(result, np.zeros(len(data), dtype=np.uint8))

    @needs_gpu
    def test_multiple_strings(self):
        """Multiple string values: parity toggles correctly across them."""
        from vibespatial.io.gpu_parse import quote_parity

        data = b'{"x":"a","y":"b"}'
        d_bytes = _to_device(data)
        d_qp = quote_parity(d_bytes)
        result = cp.asnumpy(d_qp)

        # Comma between key-value pairs should be outside quotes
        comma_pos = data.index(b",")
        assert result[comma_pos] == 0, "comma between pairs is outside quotes"


# ===========================================================================
# Structural tests: bracket_depth
# ===========================================================================


class TestBracketDepth:
    """Tests for structural.bracket_depth()."""

    @needs_gpu
    def test_simple_json(self):
        """Simple JSON: {"a": 1} -- depth progression."""
        from vibespatial.io.gpu_parse import bracket_depth, quote_parity

        data = b'{"a": 1}'
        d_bytes = _to_device(data)
        d_qp = quote_parity(d_bytes)
        d_depth = bracket_depth(d_bytes, d_qp)
        result = cp.asnumpy(d_depth)

        # { " a " : _ 1 }
        # 1 1 1 1 1 1 1 0
        assert result[0] == 1, "depth at '{' is 1 (inclusive)"
        assert result[len(data) - 1] == 0, "depth at '}' is 0 (inclusive)"
        # Interior positions are depth 1
        assert result[4] == 1, "':' inside object is depth 1"

    @needs_gpu
    def test_nested_json(self):
        """Nested JSON: {"a": {"b": [1, 2]}} -- depth reaches 3."""
        from vibespatial.io.gpu_parse import bracket_depth, quote_parity

        data = b'{"a": {"b": [1, 2]}}'
        d_bytes = _to_device(data)
        d_qp = quote_parity(d_bytes)
        d_depth = bracket_depth(d_bytes, d_qp)
        result = cp.asnumpy(d_depth)

        # The '[' inside the nested object should reach depth 3
        bracket_pos = data.index(b"[")
        assert result[bracket_pos] == 3, "nested '[' has depth 3"

        # The '1' inside the array should also be at depth 3
        one_pos = data.index(b"1")
        assert result[one_pos] == 3

        # Final '}' brings depth back to 0
        assert result[len(data) - 1] == 0

    @needs_gpu
    def test_brackets_inside_strings_ignored(self):
        """Brackets inside quoted strings do not affect depth."""
        from vibespatial.io.gpu_parse import bracket_depth, quote_parity

        data = b'{"key": "[not a bracket]"}'
        d_bytes = _to_device(data)
        d_qp = quote_parity(d_bytes)
        d_depth = bracket_depth(d_bytes, d_qp)
        result = cp.asnumpy(d_depth)

        # The '[' inside the string value should NOT increase depth.
        # Depth should stay at 1 for the string content.
        bracket_in_string = data.index(b"[")
        assert result[bracket_in_string] == 1, (
            "bracket inside string should not change depth"
        )
        # Final '}' goes to 0
        assert result[len(data) - 1] == 0

    @needs_gpu
    def test_custom_brackets_wkt(self):
        """Custom bracket chars for WKT: open_chars='(', close_chars=')'."""
        from vibespatial.io.gpu_parse import bracket_depth

        data = b"POLYGON((0 0, 1 1, 1 0, 0 0))"
        d_bytes = _to_device(data)
        # WKT has no quotes, but we still need a parity mask (all zeros)
        d_qp = cp.zeros(len(data), dtype=cp.uint8)
        d_depth = bracket_depth(
            d_bytes, d_qp, open_chars="(", close_chars=")"
        )
        result = cp.asnumpy(d_depth)

        # First '(' at POLYGON( -> depth 1
        first_open = data.index(b"(")
        assert result[first_open] == 1

        # Second '(' -> depth 2
        second_open = data.index(b"(", first_open + 1)
        assert result[second_open] == 2

        # Coordinate data inside ((...)) -> depth 2
        coord_pos = data.index(b"0 0") + 1  # the space
        assert result[coord_pos] == 2

        # After last ')' -> depth 0
        assert result[len(data) - 1] == 0

    @needs_gpu
    def test_mismatched_lengths_raises(self):
        """ValueError when open_chars and close_chars have different lengths."""
        from vibespatial.io.gpu_parse import bracket_depth

        d_bytes = cp.zeros(10, dtype=cp.uint8)
        d_qp = cp.zeros(10, dtype=cp.uint8)

        with pytest.raises(ValueError, match="same length"):
            bracket_depth(d_bytes, d_qp, open_chars="({", close_chars=")")

    @needs_gpu
    def test_too_many_bracket_chars_raises(self):
        """ValueError when bracket chars exceed maximum of 8."""
        from vibespatial.io.gpu_parse import bracket_depth

        d_bytes = cp.zeros(10, dtype=cp.uint8)
        d_qp = cp.zeros(10, dtype=cp.uint8)

        with pytest.raises(ValueError, match="Maximum 8"):
            bracket_depth(
                d_bytes, d_qp,
                open_chars="({[<abcde",
                close_chars=")}]>ABCDE",
            )

    @needs_gpu
    def test_empty_input(self):
        """Empty input returns empty depth array."""
        from vibespatial.io.gpu_parse import bracket_depth, quote_parity

        d_bytes = cp.empty(0, dtype=cp.uint8)
        d_qp = quote_parity(d_bytes)
        d_depth = bracket_depth(d_bytes, d_qp)
        assert d_depth.shape == (0,)


# ===========================================================================
# Numeric tests: number_boundaries
# ===========================================================================


class TestNumberBoundaries:
    """Tests for numeric.number_boundaries()."""

    @needs_gpu
    def test_basic_integers(self):
        """Detect start/end of simple integers in array context."""
        from vibespatial.io.gpu_parse import number_boundaries, quote_parity

        # [42, 7]
        # 0123456
        data = b"[42, 7]"
        d_bytes = _to_device(data)
        d_qp = quote_parity(d_bytes)
        d_start, d_end = number_boundaries(d_bytes, d_qp)

        starts = cp.asnumpy(d_start)
        ends = cp.asnumpy(d_end)

        # "42" starts at position 1, ends at position 2 (next is ',')
        assert starts[1] == 1, "start of 42"
        assert ends[2] == 1, "end of 42"
        # "7" starts at position 5 (preceded by space), ends at position 5 (next is ']')
        assert starts[5] == 1, "start of 7"
        assert ends[5] == 1, "end of 7"

    @needs_gpu
    def test_negative_and_decimal(self):
        """Negative and decimal numbers detected correctly."""
        from vibespatial.io.gpu_parse import number_boundaries, quote_parity

        data = b"[-80.12, 27.5]"
        d_bytes = _to_device(data)
        d_qp = quote_parity(d_bytes)
        d_start, d_end = number_boundaries(d_bytes, d_qp)

        starts = cp.asnumpy(d_start)
        ends = cp.asnumpy(d_end)

        # -80.12 starts at position 1 (the '-')
        assert starts[1] == 1, "start of -80.12"
        # -80.12 ends at position 6 (the '2'), since next char is ','
        assert ends[6] == 1, "end of -80.12"

    @needs_gpu
    def test_numbers_inside_strings_filtered(self):
        """Numbers inside quoted strings should not be detected."""
        from vibespatial.io.gpu_parse import number_boundaries, quote_parity

        data = b'{"id": "42", "val": 7}'
        d_bytes = _to_device(data)
        d_qp = quote_parity(d_bytes)
        d_start, d_end = number_boundaries(d_bytes, d_qp)

        starts = cp.asnumpy(d_start)

        # "42" inside string value should not be detected
        # Find the position of '4' in "42"
        str_42_pos = data.index(b'"42"') + 1  # position of '4'
        assert starts[str_42_pos] == 0, "number inside string should not be detected"

        # The bare 7 outside quotes should be detected
        bare_7_pos = data.index(b" 7}") + 1  # position of '7'
        assert starts[bare_7_pos] == 1, "number outside string should be detected"

    @needs_gpu
    def test_gdal_whitespace(self):
        """GDAL whitespace pattern: '0.0 ]' with space before bracket."""
        from vibespatial.io.gpu_parse import number_boundaries, quote_parity

        data = b"[0.0 ]"
        d_bytes = _to_device(data)
        d_qp = quote_parity(d_bytes)
        d_start, d_end = number_boundaries(d_bytes, d_qp)

        starts = cp.asnumpy(d_start)
        ends = cp.asnumpy(d_end)

        # 0.0 starts at position 1, ends at position 3 (before the space)
        assert starts[1] == 1, "start of 0.0"
        assert ends[3] == 1, "end of 0.0 (space is separator)"

    @needs_gpu
    def test_scientific_notation_boundaries(self):
        """Scientific notation: e/E characters are numeric for boundary detection."""
        from vibespatial.io.gpu_parse import number_boundaries, quote_parity

        # [1.5e2, 3.14E-2]
        # 0123456789...
        data = b"[1.5e2, 3.14E-2]"
        d_bytes = _to_device(data)
        d_qp = quote_parity(d_bytes)
        d_start, d_end = number_boundaries(d_bytes, d_qp)

        starts = cp.asnumpy(d_start)
        ends = cp.asnumpy(d_end)

        # "1.5e2" starts at position 1
        assert starts[1] == 1, "start of 1.5e2"
        # "1.5e2" ends at position 5 ('2'), since next char is ','
        assert ends[5] == 1, "end of 1.5e2"
        # "3.14E-2" starts at position 8
        assert starts[8] == 1, "start of 3.14E-2"
        # "3.14E-2" ends at position 14 ('2'), since next char is ']'
        assert ends[14] == 1, "end of 3.14E-2"

    @needs_gpu
    def test_empty_input(self):
        """Empty input returns zero-length boundary arrays."""
        from vibespatial.io.gpu_parse import number_boundaries

        d_bytes = cp.empty(0, dtype=cp.uint8)
        d_qp = cp.empty(0, dtype=cp.uint8)
        d_start, d_end = number_boundaries(d_bytes, d_qp)

        assert d_start.shape == (0,)
        assert d_end.shape == (0,)


# ===========================================================================
# Numeric tests: parse_ascii_floats
# ===========================================================================


class TestParseAsciiFloats:
    """Tests for numeric.parse_ascii_floats()."""

    @needs_gpu
    def test_positive_integer(self):
        """Positive integer: 42 -> 42.0."""
        from vibespatial.io.gpu_parse import parse_ascii_floats

        data = b"[42]"
        d_bytes = _to_device(data)
        # Token "42" is at [1, 3) -- half-open
        d_starts = cp.array([1], dtype=cp.int64)
        d_ends = cp.array([3], dtype=cp.int64)

        result = cp.asnumpy(parse_ascii_floats(d_bytes, d_starts, d_ends))
        assert result[0] == pytest.approx(42.0)

    @needs_gpu
    def test_negative_number(self):
        """Negative number: -80.12345 -> -80.12345."""
        from vibespatial.io.gpu_parse import parse_ascii_floats

        data = b"[-80.12345]"
        d_bytes = _to_device(data)
        # Token "-80.12345" at [1, 10)
        d_starts = cp.array([1], dtype=cp.int64)
        d_ends = cp.array([10], dtype=cp.int64)

        result = cp.asnumpy(parse_ascii_floats(d_bytes, d_starts, d_ends))
        assert result[0] == pytest.approx(-80.12345)

    @needs_gpu
    def test_scientific_notation(self):
        """Scientific notation: 1.5e2 -> 150.0, 3.14E-2 -> 0.0314."""
        from vibespatial.io.gpu_parse import parse_ascii_floats

        data = b"[1.5e2,3.14E-2]"
        d_bytes = _to_device(data)
        # "1.5e2" at [1, 5+1=6)  -- actually [1,6) but let's compute carefully
        # data[1:6] = b"1.5e2"  that's 5 bytes, positions 1,2,3,4,5
        d_starts = cp.array([1, 7], dtype=cp.int64)
        d_ends = cp.array([6, 14], dtype=cp.int64)

        result = cp.asnumpy(parse_ascii_floats(d_bytes, d_starts, d_ends))
        assert result[0] == pytest.approx(150.0)
        assert result[1] == pytest.approx(0.0314)

    @needs_gpu
    def test_leading_plus_sign(self):
        """Leading plus sign: +1.5 -> 1.5."""
        from vibespatial.io.gpu_parse import parse_ascii_floats

        data = b"[+1.5]"
        d_bytes = _to_device(data)
        # "+1.5" at [1, 5)
        d_starts = cp.array([1], dtype=cp.int64)
        d_ends = cp.array([5], dtype=cp.int64)

        result = cp.asnumpy(parse_ascii_floats(d_bytes, d_starts, d_ends))
        assert result[0] == pytest.approx(1.5)

    @needs_gpu
    def test_coordinate_precision(self):
        """High-precision coordinate roundtrip within 1e-10."""
        from vibespatial.io.gpu_parse import parse_ascii_floats

        # 13-digit precision coordinate
        coord = b"-80.92302345678"
        data = b"[" + coord + b"]"
        d_bytes = _to_device(data)
        d_starts = cp.array([1], dtype=cp.int64)
        d_ends = cp.array([1 + len(coord)], dtype=cp.int64)

        result = cp.asnumpy(parse_ascii_floats(d_bytes, d_starts, d_ends))
        assert abs(result[0] - (-80.92302345678)) < 1e-10

    @needs_gpu
    def test_multiple_values(self):
        """Parse multiple floats in one call."""
        from vibespatial.io.gpu_parse import parse_ascii_floats

        data = b"[1.5,-2.3,0.0]"
        d_bytes = _to_device(data)
        # "1.5" at [1, 4), "-2.3" at [5, 9), "0.0" at [10, 13)
        d_starts = cp.array([1, 5, 10], dtype=cp.int64)
        d_ends = cp.array([4, 9, 13], dtype=cp.int64)

        result = cp.asnumpy(parse_ascii_floats(d_bytes, d_starts, d_ends))
        np.testing.assert_allclose(result, [1.5, -2.3, 0.0], atol=1e-15)

    @needs_gpu
    def test_zero_numbers(self):
        """Empty starts/ends arrays produce empty output."""
        from vibespatial.io.gpu_parse import parse_ascii_floats

        data = b"[1.5]"
        d_bytes = _to_device(data)
        d_starts = cp.empty(0, dtype=cp.int64)
        d_ends = cp.empty(0, dtype=cp.int64)

        result = parse_ascii_floats(d_bytes, d_starts, d_ends)
        assert result.shape == (0,)
        assert result.dtype == cp.float64


# ===========================================================================
# Numeric tests: parse_ascii_ints
# ===========================================================================


class TestParseAsciiInts:
    """Tests for numeric.parse_ascii_ints()."""

    @needs_gpu
    def test_positive_int(self):
        """Positive integer: 12345 -> 12345."""
        from vibespatial.io.gpu_parse import parse_ascii_ints

        data = b"[12345]"
        d_bytes = _to_device(data)
        d_starts = cp.array([1], dtype=cp.int64)
        d_ends = cp.array([6], dtype=cp.int64)

        result = cp.asnumpy(parse_ascii_ints(d_bytes, d_starts, d_ends))
        assert result[0] == 12345

    @needs_gpu
    def test_negative_int(self):
        """Negative integer: -42 -> -42."""
        from vibespatial.io.gpu_parse import parse_ascii_ints

        data = b"[-42]"
        d_bytes = _to_device(data)
        d_starts = cp.array([1], dtype=cp.int64)
        d_ends = cp.array([4], dtype=cp.int64)

        result = cp.asnumpy(parse_ascii_ints(d_bytes, d_starts, d_ends))
        assert result[0] == -42

    @needs_gpu
    def test_zero(self):
        """Zero: 0 -> 0."""
        from vibespatial.io.gpu_parse import parse_ascii_ints

        data = b"[0]"
        d_bytes = _to_device(data)
        d_starts = cp.array([1], dtype=cp.int64)
        d_ends = cp.array([2], dtype=cp.int64)

        result = cp.asnumpy(parse_ascii_ints(d_bytes, d_starts, d_ends))
        assert result[0] == 0

    @needs_gpu
    def test_leading_zeros(self):
        """Leading zeros: 007 -> 7."""
        from vibespatial.io.gpu_parse import parse_ascii_ints

        data = b"[007]"
        d_bytes = _to_device(data)
        d_starts = cp.array([1], dtype=cp.int64)
        d_ends = cp.array([4], dtype=cp.int64)

        result = cp.asnumpy(parse_ascii_ints(d_bytes, d_starts, d_ends))
        assert result[0] == 7

    @needs_gpu
    def test_multiple_ints(self):
        """Parse multiple integers at once."""
        from vibespatial.io.gpu_parse import parse_ascii_ints

        data = b"[100,-200,0,42]"
        d_bytes = _to_device(data)
        # "100" at [1,4), "-200" at [5,9), "0" at [10,11), "42" at [12,14)
        d_starts = cp.array([1, 5, 10, 12], dtype=cp.int64)
        d_ends = cp.array([4, 9, 11, 14], dtype=cp.int64)

        result = cp.asnumpy(parse_ascii_ints(d_bytes, d_starts, d_ends))
        np.testing.assert_array_equal(result, [100, -200, 0, 42])

    @needs_gpu
    def test_empty_input(self):
        """Zero numbers produce zero-length int64 output."""
        from vibespatial.io.gpu_parse import parse_ascii_ints

        data = b"[]"
        d_bytes = _to_device(data)
        d_starts = cp.empty(0, dtype=cp.int64)
        d_ends = cp.empty(0, dtype=cp.int64)

        result = parse_ascii_ints(d_bytes, d_starts, d_ends)
        assert result.shape == (0,)
        assert result.dtype == cp.int64


# ===========================================================================
# Numeric tests: extract_number_positions
# ===========================================================================


class TestExtractNumberPositions:
    """Tests for numeric.extract_number_positions()."""

    @needs_gpu
    def test_without_mask(self):
        """All numbers returned when no mask is provided."""
        from vibespatial.io.gpu_parse import (
            extract_number_positions,
            number_boundaries,
            quote_parity,
        )

        data = b"[1.5, -2.3]"
        d_bytes = _to_device(data)
        d_qp = quote_parity(d_bytes)
        d_start, d_end = number_boundaries(d_bytes, d_qp)
        d_starts, d_ends = extract_number_positions(d_start, d_end)

        starts = cp.asnumpy(d_starts)
        ends = cp.asnumpy(d_ends)

        assert len(starts) == 2, "two numbers detected"
        assert len(ends) == 2

        # Verify start positions point to beginning of each number
        assert data[starts[0] : starts[0] + 1] in (b"1",)
        assert data[starts[1] : starts[1] + 1] in (b"-",)

    @needs_gpu
    def test_with_mask(self):
        """Only numbers in masked region returned."""
        from vibespatial.io.gpu_parse import (
            extract_number_positions,
            number_boundaries,
            quote_parity,
        )

        # Two arrays: [1.5, 2.5] and [3.5, 4.5]
        data = b"[1.5, 2.5],[3.5, 4.5]"
        d_bytes = _to_device(data)
        d_qp = quote_parity(d_bytes)
        d_start, d_end = number_boundaries(d_bytes, d_qp)

        # Mask: only the second array [3.5, 4.5]
        d_mask = cp.zeros(len(data), dtype=cp.uint8)
        second_bracket = data.index(b"[", 1)  # find second '['
        d_mask[second_bracket:] = 1

        d_starts, d_ends = extract_number_positions(d_start, d_end, d_mask=d_mask)

        starts = cp.asnumpy(d_starts)
        assert len(starts) == 2, "only two numbers from masked region"
        # Both starts should be in the second half of the data
        assert all(s >= second_bracket for s in starts)

    @needs_gpu
    def test_empty_no_numbers(self):
        """No numbers in input produces zero-length arrays."""
        from vibespatial.io.gpu_parse import (
            extract_number_positions,
            number_boundaries,
            quote_parity,
        )

        data = b'{"key": "value"}'
        d_bytes = _to_device(data)
        d_qp = quote_parity(d_bytes)
        d_start, d_end = number_boundaries(d_bytes, d_qp)
        d_starts, d_ends = extract_number_positions(d_start, d_end)

        assert d_starts.shape[0] == 0
        assert d_ends.shape[0] == 0

    @needs_gpu
    def test_ends_are_exclusive(self):
        """Verify end positions are exclusive (start + token_length)."""
        from vibespatial.io.gpu_parse import (
            extract_number_positions,
            number_boundaries,
            parse_ascii_floats,
            quote_parity,
        )

        data = b"[42]"
        d_bytes = _to_device(data)
        d_qp = quote_parity(d_bytes)
        d_start, d_end = number_boundaries(d_bytes, d_qp)
        d_starts, d_ends = extract_number_positions(d_start, d_end)

        # Use the extracted positions to parse and verify
        result = cp.asnumpy(
            parse_ascii_floats(d_bytes, d_starts, d_ends)
        )
        assert result[0] == pytest.approx(42.0)


# ===========================================================================
# Pattern tests: pattern_match
# ===========================================================================


class TestPatternMatch:
    """Tests for pattern.pattern_match()."""

    @needs_gpu
    def test_exact_match(self):
        """Find b'"coordinates":' in simple GeoJSON."""
        from vibespatial.io.gpu_parse import pattern_match, quote_parity

        data = b'{"coordinates": [1.5, -2.3]}'
        d_bytes = _to_device(data)
        d_qp = quote_parity(d_bytes)

        pattern = b'"coordinates":'
        d_hits = pattern_match(d_bytes, pattern, d_qp)
        hits = cp.asnumpy(d_hits)

        # Pattern should match at position 1 (after the opening '{')
        hit_positions = np.flatnonzero(hits)
        assert len(hit_positions) == 1
        assert hit_positions[0] == 1

    @needs_gpu
    def test_pattern_inside_string_filtered(self):
        """Pattern inside string value should NOT match when quote parity is used.

        The JSON key pattern '"coordinates":' appears as a real key and is
        also embedded inside a string value.  Only the real key should match.
        """
        from vibespatial.io.gpu_parse import pattern_match, quote_parity

        # The real key is at position 1: "coordinates": [1.5]
        # The string value contains the text "coordinates": but parity-filtered out
        data = b'{"coordinates": [1.5], "note": "has \\"coordinates\\": in it"}'
        d_bytes = _to_device(data)
        d_qp = quote_parity(d_bytes)

        pattern = b'"coordinates":'
        d_hits = pattern_match(d_bytes, pattern, d_qp)
        hits = cp.asnumpy(d_hits)

        hit_positions = np.flatnonzero(hits)
        # The real key at position 1 matches; the embedded occurrence inside the
        # string value is suppressed by quote-parity filtering.
        assert len(hit_positions) == 1, (
            f"expected exactly 1 match, got {len(hit_positions)} at {hit_positions}"
        )
        assert hit_positions[0] == 1, "match is at the real JSON key"

    @needs_gpu
    def test_pattern_without_quote_filter(self):
        """Pattern inside string DOES match when no quote_parity is passed."""
        from vibespatial.io.gpu_parse import pattern_match

        data = b'{"note": "abc"}'
        d_bytes = _to_device(data)

        pattern = b"abc"
        # No quote parity passed -- all matches returned
        d_hits = pattern_match(d_bytes, pattern)
        hits = cp.asnumpy(d_hits)

        hit_positions = np.flatnonzero(hits)
        # "abc" is inside the string value, but without quote filtering it should match
        assert len(hit_positions) == 1

    @needs_gpu
    def test_multiple_matches(self):
        """Multiple occurrences of the same pattern."""
        from vibespatial.io.gpu_parse import pattern_match

        data = b"abcXabcXabc"
        d_bytes = _to_device(data)

        pattern = b"abc"
        d_hits = pattern_match(d_bytes, pattern)
        hits = cp.asnumpy(d_hits)

        hit_positions = np.flatnonzero(hits)
        np.testing.assert_array_equal(hit_positions, [0, 4, 8])

    @needs_gpu
    def test_pattern_at_end_no_match(self):
        """Pattern that would extend past end of buffer does not match."""
        from vibespatial.io.gpu_parse import pattern_match

        data = b"ab"
        d_bytes = _to_device(data)

        pattern = b"abc"
        d_hits = pattern_match(d_bytes, pattern)
        hits = cp.asnumpy(d_hits)

        # No match because pattern is longer than input at every position
        assert np.sum(hits) == 0

    @needs_gpu
    def test_empty_pattern_raises(self):
        """Empty pattern raises ValueError."""
        from vibespatial.io.gpu_parse import pattern_match

        d_bytes = _to_device(b"abc")
        with pytest.raises(ValueError, match="non-empty"):
            pattern_match(d_bytes, b"")

    @needs_gpu
    def test_pattern_too_long_raises(self):
        """Pattern exceeding 256 bytes raises ValueError."""
        from vibespatial.io.gpu_parse import pattern_match

        d_bytes = _to_device(b"x" * 300)
        with pytest.raises(ValueError, match="exceeds maximum"):
            pattern_match(d_bytes, b"x" * 257)


# ===========================================================================
# Pattern tests: span_boundaries
# ===========================================================================


class TestSpanBoundaries:
    """Tests for pattern.span_boundaries()."""

    @needs_gpu
    def test_find_closing_bracket(self):
        """Find matching closing bracket for a nested structure."""
        from vibespatial.io.gpu_parse import (
            bracket_depth,
            quote_parity,
            span_boundaries,
        )

        data = b'{"coordinates": [[1, 2], [3, 4]]}'
        d_bytes = _to_device(data)
        d_qp = quote_parity(d_bytes)
        d_depth = bracket_depth(d_bytes, d_qp)

        # Start from position of '"coordinates"' key
        key_pos = data.index(b'"coordinates"')
        d_starts = cp.array([key_pos], dtype=cp.int64)

        # skip_bytes = len('"coordinates": ') = 16
        skip = len(b'"coordinates": ')
        d_ends = span_boundaries(d_depth, d_starts, len(data), skip_bytes=skip)
        ends = cp.asnumpy(d_ends)

        # The kernel scans until depth drops below start_depth.
        # start_depth = depth at the '[' at position 16, which is 2.
        # The outer ']' at position 31 has depth 1 (< 2), so the scan stops.
        # The returned end is the position of the closing bracket (31).
        outer_close = data.rindex(b"]")
        assert ends[0] == outer_close

    @needs_gpu
    def test_skip_bytes(self):
        """skip_bytes correctly advances past key pattern."""
        from vibespatial.io.gpu_parse import (
            bracket_depth,
            quote_parity,
            span_boundaries,
        )

        data = b'{"key": [1, 2, 3]}'
        d_bytes = _to_device(data)
        d_qp = quote_parity(d_bytes)
        d_depth = bracket_depth(d_bytes, d_qp)

        key_pos = data.index(b'"key"')
        d_starts = cp.array([key_pos], dtype=cp.int64)

        # Skip past '"key": ' (7 bytes) to reach the '[' at position 8
        d_ends = span_boundaries(d_depth, d_starts, len(data), skip_bytes=7)
        ends = cp.asnumpy(d_ends)

        # The kernel scans from position key_pos+7=8 (the '['), records
        # start_depth=depth[8]=2, then scans until depth < 2.
        # The ']' at position 16 has depth 1 < 2, so end = 16.
        closing_bracket = data.rindex(b"]")
        assert ends[0] == closing_bracket


# ===========================================================================
# Pattern tests: mark_spans
# ===========================================================================


class TestMarkSpans:
    """Tests for pattern.mark_spans()."""

    @needs_gpu
    def test_single_span(self):
        """Single span marks correct byte range."""
        from vibespatial.io.gpu_parse import mark_spans

        d_starts = cp.array([5], dtype=cp.int64)
        d_ends = cp.array([10], dtype=cp.int64)
        d_mask = mark_spans(d_starts, d_ends, 20)

        mask = cp.asnumpy(d_mask)

        # Positions 5-9 should be 1, everything else 0
        expected = np.zeros(20, dtype=np.uint8)
        expected[5:10] = 1
        np.testing.assert_array_equal(mask, expected)

    @needs_gpu
    def test_multiple_spans(self):
        """Multiple non-overlapping spans."""
        from vibespatial.io.gpu_parse import mark_spans

        d_starts = cp.array([2, 10, 18], dtype=cp.int64)
        d_ends = cp.array([5, 15, 20], dtype=cp.int64)
        d_mask = mark_spans(d_starts, d_ends, 25)

        mask = cp.asnumpy(d_mask)

        expected = np.zeros(25, dtype=np.uint8)
        expected[2:5] = 1
        expected[10:15] = 1
        expected[18:20] = 1
        np.testing.assert_array_equal(mask, expected)

    @needs_gpu
    def test_empty_spans(self):
        """Zero spans produces all-zero mask."""
        from vibespatial.io.gpu_parse import mark_spans

        d_starts = cp.empty(0, dtype=cp.int64)
        d_ends = cp.empty(0, dtype=cp.int64)
        d_mask = mark_spans(d_starts, d_ends, 50)

        mask = cp.asnumpy(d_mask)
        np.testing.assert_array_equal(mask, np.zeros(50, dtype=np.uint8))

    @needs_gpu
    def test_adjacent_spans(self):
        """Adjacent spans (end of one = start of next) produce continuous mask."""
        from vibespatial.io.gpu_parse import mark_spans

        d_starts = cp.array([0, 5], dtype=cp.int64)
        d_ends = cp.array([5, 10], dtype=cp.int64)
        d_mask = mark_spans(d_starts, d_ends, 10)

        mask = cp.asnumpy(d_mask)
        # All positions 0-9 should be 1
        np.testing.assert_array_equal(mask, np.ones(10, dtype=np.uint8))


# ===========================================================================
# Composition test: full pipeline
# ===========================================================================


class TestFullPipeline:
    """End-to-end composition test using all primitives together."""

    @needs_gpu
    def test_geojson_point_pipeline(self):
        """Full pipeline: bytes -> coordinates for a GeoJSON Point.

        Compose: quote_parity -> bracket_depth -> pattern_match ->
        span_boundaries -> mark_spans -> number_boundaries ->
        extract_number_positions -> parse_ascii_floats
        """
        from vibespatial.io.gpu_parse import (
            bracket_depth,
            extract_number_positions,
            mark_spans,
            number_boundaries,
            parse_ascii_floats,
            pattern_match,
            quote_parity,
            span_boundaries,
        )

        geojson = (
            b'{"type":"Feature","geometry":'
            b'{"type":"Point","coordinates":[1.5,-2.3]},'
            b'"properties":{}}'
        )

        d_bytes = _to_device(geojson)
        n = len(geojson)

        # Stage 1: structural analysis
        d_qp = quote_parity(d_bytes)
        d_depth = bracket_depth(d_bytes, d_qp)

        # Stage 2: locate "coordinates": pattern
        pattern = b'"coordinates":'
        d_hits = pattern_match(d_bytes, pattern, d_qp)
        d_positions = cp.flatnonzero(d_hits).astype(cp.int64)
        assert cp.asnumpy(d_positions).shape[0] == 1, "exactly one coordinates key"

        # Stage 3: find span of coordinate array
        skip = len(pattern)
        d_span_ends = span_boundaries(d_depth, d_positions, n, skip_bytes=skip)
        d_span_starts = d_positions + skip
        d_mask = mark_spans(d_span_starts, d_span_ends, n)

        # Stage 4: extract numbers within spans
        d_is_start, d_is_end = number_boundaries(d_bytes, d_qp)
        d_starts, d_ends = extract_number_positions(
            d_is_start, d_is_end, d_mask=d_mask
        )
        d_values = parse_ascii_floats(d_bytes, d_starts, d_ends)

        # Verify
        values = cp.asnumpy(d_values)
        assert len(values) == 2, "Point has exactly 2 coordinates"
        assert values[0] == pytest.approx(1.5)
        assert values[1] == pytest.approx(-2.3)

    @needs_gpu
    def test_geojson_multipoint_pipeline(self):
        """Full pipeline for a GeoJSON with multiple coordinate pairs.

        Verifies the pipeline works for nested coordinate arrays.
        """
        from vibespatial.io.gpu_parse import (
            bracket_depth,
            extract_number_positions,
            mark_spans,
            number_boundaries,
            parse_ascii_floats,
            pattern_match,
            quote_parity,
            span_boundaries,
        )

        geojson = (
            b'{"type":"Feature","geometry":'
            b'{"type":"LineString","coordinates":[[1.0,2.0],[3.0,4.0],[5.0,6.0]]},'
            b'"properties":{}}'
        )

        d_bytes = _to_device(geojson)
        n = len(geojson)

        d_qp = quote_parity(d_bytes)
        d_depth = bracket_depth(d_bytes, d_qp)

        pattern = b'"coordinates":'
        d_hits = pattern_match(d_bytes, pattern, d_qp)
        d_positions = cp.flatnonzero(d_hits).astype(cp.int64)

        skip = len(pattern)
        d_span_ends = span_boundaries(d_depth, d_positions, n, skip_bytes=skip)
        d_span_starts = d_positions + skip
        d_mask = mark_spans(d_span_starts, d_span_ends, n)

        d_is_start, d_is_end = number_boundaries(d_bytes, d_qp)
        d_starts, d_ends = extract_number_positions(
            d_is_start, d_is_end, d_mask=d_mask
        )
        d_values = parse_ascii_floats(d_bytes, d_starts, d_ends)

        values = cp.asnumpy(d_values)
        assert len(values) == 6, "LineString with 3 points = 6 coordinates"
        np.testing.assert_allclose(
            values, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], atol=1e-15
        )

    @needs_gpu
    def test_pipeline_negative_coordinates(self):
        """Pipeline handles negative coordinates (common in real-world GIS data)."""
        from vibespatial.io.gpu_parse import (
            bracket_depth,
            extract_number_positions,
            mark_spans,
            number_boundaries,
            parse_ascii_floats,
            pattern_match,
            quote_parity,
            span_boundaries,
        )

        geojson = (
            b'{"type":"Feature","geometry":'
            b'{"type":"Point","coordinates":[-80.123,27.456]},'
            b'"properties":{}}'
        )

        d_bytes = _to_device(geojson)
        n = len(geojson)

        d_qp = quote_parity(d_bytes)
        d_depth = bracket_depth(d_bytes, d_qp)

        pattern = b'"coordinates":'
        d_hits = pattern_match(d_bytes, pattern, d_qp)
        d_positions = cp.flatnonzero(d_hits).astype(cp.int64)

        skip = len(pattern)
        d_span_ends = span_boundaries(d_depth, d_positions, n, skip_bytes=skip)
        d_span_starts = d_positions + skip
        d_mask = mark_spans(d_span_starts, d_span_ends, n)

        d_is_start, d_is_end = number_boundaries(d_bytes, d_qp)
        d_starts, d_ends = extract_number_positions(
            d_is_start, d_is_end, d_mask=d_mask
        )
        d_values = parse_ascii_floats(d_bytes, d_starts, d_ends)

        values = cp.asnumpy(d_values)
        assert len(values) == 2
        assert values[0] == pytest.approx(-80.123)
        assert values[1] == pytest.approx(27.456)

    @needs_gpu
    def test_pipeline_ignores_numeric_property_values(self):
        """Pipeline extracts only coordinates, not numeric property values.

        The coordinate span mask ensures that numbers in "properties"
        are excluded from coordinate extraction.
        """
        from vibespatial.io.gpu_parse import (
            bracket_depth,
            extract_number_positions,
            mark_spans,
            number_boundaries,
            parse_ascii_floats,
            pattern_match,
            quote_parity,
            span_boundaries,
        )

        geojson = (
            b'{"type":"Feature","geometry":'
            b'{"type":"Point","coordinates":[10.0,20.0]},'
            b'"properties":{"population":999999,"area":42.5}}'
        )

        d_bytes = _to_device(geojson)
        n = len(geojson)

        d_qp = quote_parity(d_bytes)
        d_depth = bracket_depth(d_bytes, d_qp)

        pattern = b'"coordinates":'
        d_hits = pattern_match(d_bytes, pattern, d_qp)
        d_positions = cp.flatnonzero(d_hits).astype(cp.int64)

        skip = len(pattern)
        d_span_ends = span_boundaries(d_depth, d_positions, n, skip_bytes=skip)
        d_span_starts = d_positions + skip
        d_mask = mark_spans(d_span_starts, d_span_ends, n)

        d_is_start, d_is_end = number_boundaries(d_bytes, d_qp)
        d_starts, d_ends = extract_number_positions(
            d_is_start, d_is_end, d_mask=d_mask
        )
        d_values = parse_ascii_floats(d_bytes, d_starts, d_ends)

        values = cp.asnumpy(d_values)
        # Only the 2 coordinate values, NOT the property values
        assert len(values) == 2, (
            f"expected 2 coordinates, got {len(values)}: {values}"
        )
        assert values[0] == pytest.approx(10.0)
        assert values[1] == pytest.approx(20.0)
