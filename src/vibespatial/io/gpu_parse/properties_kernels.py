"""NVRTC kernel sources for GPU property extraction primitives."""

from __future__ import annotations

_CLASSIFY_VALUE_SOURCE = r"""
// After each property key+colon, examine the first non-whitespace byte
// of the value to determine its JSON type.
//
// Output codes:
//   0 = STRING  (starts with '"')
//   1 = BOOLEAN (starts with 't' or 'f')
//   2 = NULL    (starts with 'n')
//   3 = NUMBER  (starts with digit or '-')
//   4 = COMPLEX (starts with '[' or '{')
//  -1 = ERROR   (unexpected byte or end-of-input)

extern "C" __global__ void __launch_bounds__(256, 4)
classify_property_values(
    const unsigned char* __restrict__ input,
    const long long* __restrict__ colon_positions,
    signed char* __restrict__ value_types,
    int n_keys,
    long long n_bytes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_keys) return;

    // Start scanning after the colon
    long long pos = colon_positions[idx] + 1;

    // Skip whitespace: space, tab, newline, carriage return
    while (pos < n_bytes) {
        unsigned char c = input[pos];
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') break;
        pos++;
    }

    if (pos >= n_bytes) {
        value_types[idx] = -1;
        return;
    }

    unsigned char c = input[pos];
    if (c == '"') {
        value_types[idx] = 0;  // STRING
    } else if (c == 't' || c == 'f') {
        value_types[idx] = 1;  // BOOLEAN
    } else if (c == 'n') {
        value_types[idx] = 2;  // NULL
    } else if ((c >= '0' && c <= '9') || c == '-') {
        value_types[idx] = 3;  // NUMBER
    } else if (c == '[' || c == '{') {
        value_types[idx] = 4;  // COMPLEX
    } else {
        value_types[idx] = -1;  // ERROR
    }
}
"""

_PROPERTY_NUM_BOUNDS_SOURCE = r"""
// Property-specific number boundary detection.
// Identical to the generic find_number_boundaries kernel but adds ':'
// to the separator set for start detection.  In JSON property values
// like "population":42, the number starts immediately after ':'.
// The generic kernel only recognizes ',', '[', ']', space, newline,
// tab, and carriage return as separators.

extern "C" __global__ void __launch_bounds__(256, 4)
find_property_number_boundaries(
    const unsigned char* __restrict__ input,
    const unsigned char* __restrict__ quote_parity,
    unsigned char* __restrict__ is_start,
    unsigned char* __restrict__ is_end,
    long long n
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Skip bytes inside string values (odd parity)
    if (quote_parity[idx] != 0) {
        is_start[idx] = 0;
        is_end[idx] = 0;
        return;
    }

    unsigned char c = input[idx];
    unsigned char prev = (idx > 0) ? input[idx - 1] : '[';
    unsigned char next = (idx < n - 1) ? input[idx + 1] : ']';

    // Number starts: first char of a number, preceded by separator.
    // Includes ':' for JSON property values ("key":42).
    unsigned char is_first_digit = (c >= '0' && c <= '9') || c == '-' || c == '+';
    unsigned char is_sep_before = (prev == ',' || prev == '[' || prev == ':'
                                   || prev == ' ' || prev == '\n'
                                   || prev == '\r' || prev == '\t');
    is_start[idx] = is_first_digit && is_sep_before;

    // Number ends: last numeric char followed by separator.
    // Includes '}' for JSON property values (42}).
    unsigned char is_numeric = (c >= '0' && c <= '9') || c == '.' ||
                               c == 'e' || c == 'E' || c == '-' || c == '+';
    unsigned char is_sep_after = (next == ',' || next == ']' || next == '}'
                                  || next == ' ' || next == '\n'
                                  || next == '\r' || next == '\t');
    is_end[idx] = is_numeric && is_sep_after;
}
"""

_PROPERTY_NUM_BOUNDS_NAMES = ("find_property_number_boundaries",)

_EXTRACT_BOOL_SOURCE = r"""
// For boolean-typed property values, read "true" or "false" from the
// byte stream.  Output: 1 for true, 0 for false.
//
// colon_positions[idx] points to the ':' after the property key.
// We skip whitespace to find the value start, then check the first byte.

extern "C" __global__ void __launch_bounds__(256, 4)
extract_booleans(
    const unsigned char* __restrict__ input,
    const long long* __restrict__ colon_positions,
    unsigned char* __restrict__ output,
    int n_keys,
    long long n_bytes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_keys) return;

    long long pos = colon_positions[idx] + 1;

    // Skip whitespace
    while (pos < n_bytes) {
        unsigned char c = input[pos];
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') break;
        pos++;
    }

    if (pos >= n_bytes) {
        output[idx] = 0;
        return;
    }

    // 't' -> true (1), 'f' -> false (0)
    output[idx] = (input[pos] == 't') ? 1 : 0;
}
"""

# Kernel name tuples
_CLASSIFY_VALUE_NAMES = ("classify_property_values",)
_EXTRACT_BOOL_NAMES = ("extract_booleans",)
