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
__device__ __forceinline__ double decimal_pow10_abs(int exp10) {
    double scale = 1.0;
    double base = 10.0;
    int e = exp10;
    while (e > 0) {
        if (e & 1) {
            scale *= base;
        }
        base *= base;
        e >>= 1;
    }
    return scale;
}

__device__ __forceinline__ double decimal_apply_scale(double value, int exp10) {
    if (exp10 > 0) {
        return value * decimal_pow10_abs(exp10);
    }
    if (exp10 < 0) {
        return value / decimal_pow10_abs(-exp10);
    }
    return value;
}

__device__ __forceinline__ double decimal_mantissa_to_double(
    unsigned long long mantissa,
    int decimal_exp
) {
    const unsigned long long fp64_exact_integer_limit = 9007199254740992ULL;
    if (mantissa <= fp64_exact_integer_limit) {
        return decimal_apply_scale((double)mantissa, decimal_exp);
    }

    const unsigned long long chunk_base = 1000000000ULL;
    unsigned long long high = mantissa / chunk_base;
    unsigned long long low = mantissa - high * chunk_base;

    if (high == 0ULL) {
        return decimal_apply_scale((double)low, decimal_exp);
    }
    if (low == 0ULL) {
        return decimal_apply_scale((double)high, decimal_exp + 9);
    }

    int high_exp = decimal_exp + 9;
    double high_scale = high_exp >= 0
        ? decimal_pow10_abs(high_exp)
        : 1.0 / decimal_pow10_abs(-high_exp);
    double low_term = decimal_apply_scale((double)low, decimal_exp);
    return fma((double)high, high_scale, low_term);
}

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

    int negative = 0;
    int after_decimal = 0;
    int in_exponent = 0;
    int exp_val = 0;
    int exp_negative = 0;
    int saw_digit = 0;
    int significant_digits = 0;
    int decimal_exp = 0;
    int truncated_nonzero = 0;
    unsigned long long mantissa = 0ULL;

    for (long long i = start; i < end; ++i) {
        unsigned char c = input[i];
        if (c == '-') {
            if (in_exponent) exp_negative = 1;
            else negative = 1;
        } else if (c == '+') {
            // skip
        } else if (c == '.') {
            if (!in_exponent) {
                after_decimal = 1;
            }
        } else if (c == 'e' || c == 'E') {
            in_exponent = 1;
        } else if (c >= '0' && c <= '9') {
            int d = c - '0';
            if (in_exponent) {
                exp_val = exp_val * 10 + d;
            } else {
                saw_digit = 1;
                if (significant_digits == 0 && d == 0) {
                    if (after_decimal) {
                        decimal_exp -= 1;
                    }
                } else if (significant_digits < 19) {
                    mantissa = mantissa * 10ULL + (unsigned long long)d;
                    significant_digits += 1;
                    if (after_decimal) {
                        decimal_exp -= 1;
                    }
                } else if (!after_decimal) {
                    decimal_exp += 1;
                    if (d != 0) {
                        truncated_nonzero = 1;
                    }
                } else if (d != 0) {
                    truncated_nonzero = 1;
                }
            }
        }
    }

    if (in_exponent) {
        decimal_exp += exp_negative ? -exp_val : exp_val;
    }
    double result = saw_digit ? decimal_mantissa_to_double(mantissa, decimal_exp) : 0.0;
    (void)truncated_nonzero;
    if (negative) result = -result;

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
