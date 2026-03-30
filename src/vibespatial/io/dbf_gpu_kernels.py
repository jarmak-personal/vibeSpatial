"""NVRTC kernel sources for GPU DBF (dBASE III) reader."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# NVRTC kernel sources (Tier 1)
# ---------------------------------------------------------------------------

# Numeric field extraction: each thread parses one (record, field) pair.
# The field's byte span is at a fixed offset within each record, so
# indexing is pure arithmetic with no search.  The ASCII-to-float64
# state machine is the same approach as gpu_parse/numeric.py.
_DBF_EXTRACT_NUMERIC_SOURCE = r"""
extern "C" __global__ void __launch_bounds__(256, 4)
dbf_extract_numeric(
    const unsigned char* __restrict__ data,
    double*              __restrict__ output,
    const int record_count,
    const int record_length,
    const int field_offset,
    const int field_length,
    const int header_length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= record_count) return;

    // +1 skips the deletion flag byte at the start of each record
    int start = header_length + idx * record_length + field_offset + 1;
    int end   = start + field_length;

    // Strip leading spaces and trailing spaces
    while (start < end && data[start] == ' ') start++;
    while (end > start && data[end - 1] == ' ') end--;

    // Empty or all-spaces field -> NaN
    if (start >= end) {
        output[idx] = __longlong_as_double(0x7FF8000000000000LL); // NaN
        return;
    }

    // Check for '*' fill (null indicator in some DBF writers)
    if (data[start] == '*') {
        output[idx] = __longlong_as_double(0x7FF8000000000000LL); // NaN
        return;
    }

    double result = 0.0;
    double frac_mult = 0.0;
    int negative = 0;
    int in_exponent = 0;
    int exp_val = 0;
    int exp_negative = 0;

    for (int i = start; i < end; ++i) {
        unsigned char c = data[i];
        if (c == '-') {
            if (in_exponent) exp_negative = 1;
            else negative = 1;
        } else if (c == '+') {
            // skip
        } else if (c == '.') {
            frac_mult = 0.1;
        } else if (c == 'e' || c == 'E') {
            in_exponent = 1;
        } else if (c >= '0' && c <= '9') {
            int d = c - '0';
            if (in_exponent) {
                exp_val = exp_val * 10 + d;
            } else if (frac_mult > 0.0) {
                result += d * frac_mult;
                frac_mult *= 0.1;
            } else {
                result = result * 10.0 + d;
            }
        }
        // Skip any other characters (e.g., commas in some locales)
    }

    if (negative) result = -result;

    if (in_exponent) {
        double exp_mult = 1.0;
        for (int e = 0; e < exp_val; ++e) exp_mult *= 10.0;
        if (exp_negative) result /= exp_mult;
        else result *= exp_mult;
    }

    output[idx] = result;
}
"""

# Date field extraction: YYYYMMDD (8 bytes fixed) -> int32 as YYYYMMDD integer.
# Each thread handles one record.
_DBF_EXTRACT_DATE_SOURCE = r"""
extern "C" __global__ void __launch_bounds__(256, 4)
dbf_extract_date(
    const unsigned char* __restrict__ data,
    int*                 __restrict__ output,
    const int record_count,
    const int record_length,
    const int field_offset,
    const int header_length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= record_count) return;

    // +1 skips the deletion flag byte
    int start = header_length + idx * record_length + field_offset + 1;

    // Read 8 ASCII digits: YYYYMMDD
    int value = 0;
    int all_spaces = 1;
    for (int i = 0; i < 8; ++i) {
        unsigned char c = data[start + i];
        if (c != ' ' && c != 0) all_spaces = 0;
        if (c >= '0' && c <= '9') {
            value = value * 10 + (c - '0');
        }
        // Non-digit, non-space: treat as null
    }

    // Empty date field -> 0 (sentinel for null)
    output[idx] = all_spaces ? 0 : value;
}
"""

_DBF_NUMERIC_NAMES = ("dbf_extract_numeric",)
_DBF_DATE_NAMES = ("dbf_extract_date",)
