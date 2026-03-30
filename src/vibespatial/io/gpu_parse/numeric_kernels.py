"""NVRTC kernel sources for GPU numeric parsing primitives."""

from __future__ import annotations

_NUM_BOUNDS_SOURCE = r"""
extern "C" __global__ void find_number_boundaries(
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

    // Number starts: first char of a number, preceded by separator
    // Space included because json.dumps uses ", " as separator
    unsigned char is_first_digit = (c >= '0' && c <= '9') || c == '-' || c == '+';
    unsigned char is_sep_before = (prev == ',' || prev == '[' || prev == ' '
                                   || prev == '\n' || prev == '\r' || prev == '\t');
    is_start[idx] = is_first_digit && is_sep_before;

    // Number ends: last numeric char followed by separator
    // Space/newline included because GDAL/OGR writes "0.0 ]" with
    // whitespace between the last coordinate value and closing bracket.
    unsigned char is_numeric = (c >= '0' && c <= '9') || c == '.' ||
                               c == 'e' || c == 'E' || c == '-' || c == '+';
    unsigned char is_sep_after = (next == ',' || next == ']' || next == ' '
                                  || next == '\n' || next == '\r' || next == '\t');
    is_end[idx] = is_numeric && is_sep_after;
}
"""

_PARSE_FLOAT_SOURCE = r"""
extern "C" __global__ void parse_ascii_floats(
    const unsigned char* __restrict__ input,
    const long long* __restrict__ coord_starts,
    const long long* __restrict__ coord_ends,
    double* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    long long start = coord_starts[idx];
    long long end = coord_ends[idx];

    double result = 0.0;
    double frac_mult = 0.0;
    int negative = 0;
    int in_exponent = 0;
    int exp_val = 0;
    int exp_negative = 0;

    for (long long i = start; i < end; ++i) {
        unsigned char c = input[i];
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

_PARSE_INT_SOURCE = r"""
extern "C" __global__ void parse_ascii_ints(
    const unsigned char* __restrict__ input,
    const long long* __restrict__ token_starts,
    const long long* __restrict__ token_ends,
    long long* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    long long start = token_starts[idx];
    long long end = token_ends[idx];

    long long result = 0;
    int negative = 0;

    for (long long i = start; i < end; ++i) {
        unsigned char c = input[i];
        if (c == '-') {
            negative = 1;
        } else if (c == '+') {
            // skip
        } else if (c >= '0' && c <= '9') {
            result = result * 10 + (c - '0');
        } else {
            // Non-digit, non-sign: stop accumulating
            break;
        }
    }

    if (negative) result = -result;

    output[idx] = result;
}
"""

# Kernel name tuples
_NUM_BOUNDS_NAMES = ("find_number_boundaries",)
_PARSE_FLOAT_NAMES = ("parse_ascii_floats",)
_PARSE_INT_NAMES = ("parse_ascii_ints",)
