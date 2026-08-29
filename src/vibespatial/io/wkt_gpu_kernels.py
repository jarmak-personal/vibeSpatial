"""NVRTC kernel sources for GPU WKT parser."""

from __future__ import annotations

_WKT_CLASSIFY_SOURCE = r"""
// WKT geometry type classification kernel.
//
// Each thread examines one geometry start position and classifies it
// by checking byte prefixes.  Case-insensitive: each byte comparison
// tests both upper and lower case via bitwise OR with 0x20 (which
// maps ASCII A-Z to a-z while leaving a-z unchanged).
//
// EWKT support: an exact case-insensitive "SRID=" prefix, one or more decimal
// digits, and a same-row semicolon may precede the geometry keyword.
//
// Family tags match GeometryFamily enum order:
//   POINT=0, LINESTRING=1, POLYGON=2,
//   MULTIPOINT=3, MULTILINESTRING=4, MULTIPOLYGON=5,
//   unknown/unsupported=-2
//
// Also detects the EMPTY keyword after the type name and any
// optional dimension suffix (Z, M, ZM).

extern "C" __global__ void __launch_bounds__(256, 4)
wkt_classify_geometry_type(
    const unsigned char* __restrict__ input,
    const long long* __restrict__ geom_starts,
    signed char* __restrict__ family_tags,
    unsigned char* __restrict__ empty_flags,
    const int n_geoms,
    const long long n_bytes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_geoms) return;

    long long pos = geom_starts[idx];
    const long long geom_end =
        (idx + 1 < n_geoms) ? geom_starts[idx + 1] : n_bytes;

    // Skip leading whitespace
    while (pos < n_bytes) {
        unsigned char c = input[pos];
        if (c != ' ' && c != '\t' && c != '\r' && c != '\n') break;
        pos++;
    }

    if (pos >= n_bytes) {
        family_tags[idx] = -2;
        empty_flags[idx] = 0;
        return;
    }

    // EWKT: handle only exact SRID=NNNN; prefixes. Any spelling beginning
    // with "sr" that fails this grammar is malformed, and scans never cross
    // the current geometry row looking for a later semicolon.
    unsigned char first = input[pos] | 0x20;  // to lowercase
    if (first == 's' && pos + 1 < geom_end && (input[pos + 1] | 0x20) == 'r') {
        const bool exact_prefix =
            pos + 4 < geom_end
            && (input[pos + 2] | 0x20) == 'i'
            && (input[pos + 3] | 0x20) == 'd'
            && input[pos + 4] == '=';
        long long srid_pos = pos + 5;
        const long long digit_start = srid_pos;
        if (exact_prefix) {
            while (
                srid_pos < geom_end
                && input[srid_pos] >= '0'
                && input[srid_pos] <= '9'
            ) {
                srid_pos++;
            }
        }
        if (
            !exact_prefix
            || srid_pos == digit_start
            || srid_pos >= geom_end
            || input[srid_pos] != ';'
        ) {
            family_tags[idx] = -4;
            empty_flags[idx] = 0;
            return;
        }
        pos = srid_pos + 1;
        // Skip whitespace after semicolon
        while (pos < geom_end) {
            unsigned char c = input[pos];
            if (c != ' ' && c != '\t' && c != '\r' && c != '\n') break;
            pos++;
        }
        if (pos >= geom_end) {
            family_tags[idx] = -4;
            empty_flags[idx] = 0;
            return;
        }
    }

    // Now pos points to the geometry type keyword.
    // Helper macro: case-insensitive byte comparison
    #define LC(p) ((p) | 0x20)

    // Read first character (lowercased) to branch into type groups
    unsigned char c0 = LC(input[pos]);
    signed char tag = -2;
    long long type_end = pos;  // will track where the type name ends

    if (c0 == 'p') {
        // Could be POINT or POLYGON
        if (pos + 4 < n_bytes
            && LC(input[pos + 1]) == 'o'
            && LC(input[pos + 2]) == 'i'
            && LC(input[pos + 3]) == 'n'
            && LC(input[pos + 4]) == 't') {
            tag = 0;  // POINT
            type_end = pos + 5;
        } else if (pos + 6 < n_bytes
            && LC(input[pos + 1]) == 'o'
            && LC(input[pos + 2]) == 'l'
            && LC(input[pos + 3]) == 'y'
            && LC(input[pos + 4]) == 'g'
            && LC(input[pos + 5]) == 'o'
            && LC(input[pos + 6]) == 'n') {
            tag = 2;  // POLYGON
            type_end = pos + 7;
        }
    } else if (c0 == 'l') {
        // LINESTRING
        if (pos + 9 < n_bytes
            && LC(input[pos + 1]) == 'i'
            && LC(input[pos + 2]) == 'n'
            && LC(input[pos + 3]) == 'e'
            && LC(input[pos + 4]) == 's'
            && LC(input[pos + 5]) == 't'
            && LC(input[pos + 6]) == 'r'
            && LC(input[pos + 7]) == 'i'
            && LC(input[pos + 8]) == 'n'
            && LC(input[pos + 9]) == 'g') {
            tag = 1;  // LINESTRING
            type_end = pos + 10;
        }
    } else if (c0 == 'm') {
        // MULTI* types: check "MULTI" prefix first
        if (pos + 4 < n_bytes
            && LC(input[pos + 1]) == 'u'
            && LC(input[pos + 2]) == 'l'
            && LC(input[pos + 3]) == 't'
            && LC(input[pos + 4]) == 'i') {
            long long mpos = pos + 5;
            if (mpos < n_bytes) {
                unsigned char mc = LC(input[mpos]);
                if (mc == 'p') {
                    // MULTIPOINT or MULTIPOLYGON
                    if (mpos + 4 < n_bytes
                        && LC(input[mpos + 1]) == 'o'
                        && LC(input[mpos + 2]) == 'i'
                        && LC(input[mpos + 3]) == 'n'
                        && LC(input[mpos + 4]) == 't') {
                        tag = 3;  // MULTIPOINT
                        type_end = mpos + 5;
                    } else if (mpos + 6 < n_bytes
                        && LC(input[mpos + 1]) == 'o'
                        && LC(input[mpos + 2]) == 'l'
                        && LC(input[mpos + 3]) == 'y'
                        && LC(input[mpos + 4]) == 'g'
                        && LC(input[mpos + 5]) == 'o'
                        && LC(input[mpos + 6]) == 'n') {
                        tag = 5;  // MULTIPOLYGON
                        type_end = mpos + 7;
                    }
                } else if (mc == 'l') {
                    // MULTILINESTRING
                    if (mpos + 9 < n_bytes
                        && LC(input[mpos + 1]) == 'i'
                        && LC(input[mpos + 2]) == 'n'
                        && LC(input[mpos + 3]) == 'e'
                        && LC(input[mpos + 4]) == 's'
                        && LC(input[mpos + 5]) == 't'
                        && LC(input[mpos + 6]) == 'r'
                        && LC(input[mpos + 7]) == 'i'
                        && LC(input[mpos + 8]) == 'n'
                        && LC(input[mpos + 9]) == 'g') {
                        tag = 4;  // MULTILINESTRING
                        type_end = mpos + 10;
                    }
                }
            }
        }
    } else if (c0 == 'g') {
        // GEOMETRYCOLLECTION
        if (pos + 17 < n_bytes
            && LC(input[pos + 1]) == 'e'
            && LC(input[pos + 2]) == 'o'
            && LC(input[pos + 3]) == 'm'
            && LC(input[pos + 4]) == 'e'
            && LC(input[pos + 5]) == 't'
            && LC(input[pos + 6]) == 'r'
            && LC(input[pos + 7]) == 'y'
            && LC(input[pos + 8]) == 'c'
            && LC(input[pos + 9]) == 'o'
            && LC(input[pos + 10]) == 'l'
            && LC(input[pos + 11]) == 'l'
            && LC(input[pos + 12]) == 'e'
            && LC(input[pos + 13]) == 'c'
            && LC(input[pos + 14]) == 't'
            && LC(input[pos + 15]) == 'i'
            && LC(input[pos + 16]) == 'o'
            && LC(input[pos + 17]) == 'n') {
            tag = -2;  // unsupported for now
            type_end = pos + 18;
        }
    }

    family_tags[idx] = tag;

    // Detect EMPTY keyword after type name.
    // WKT allows optional dimension suffix (Z, M, ZM) between the
    // type name and EMPTY/opening paren, with or without space:
    //   POINT EMPTY, POINT Z EMPTY, POINTZ EMPTY, POINT ZM EMPTY
    long long ep = type_end;
    unsigned char type_terminated = 0;
    if (tag >= 0 && type_end < geom_end) {
        const unsigned char tc = input[type_end];
        const unsigned char tlc = LC(tc);
        type_terminated = (
            tc == '(' || tc == ' ' || tc == '\t' || tc == '\r' || tc == '\n'
            || tlc == 'z' || tlc == 'm'
        );
    }
    unsigned char has_dimension = 0;
    // Skip optional dimension suffix: Z, M, ZM (with or without space)
    if (ep < n_bytes) {
        unsigned char dc = LC(input[ep]);
        // Space before dimension suffix
        if (dc == ' ' || dc == '\t') {
            long long sp = ep;
            while (sp < n_bytes && (input[sp] == ' ' || input[sp] == '\t')) sp++;
            if (sp < n_bytes) {
                unsigned char sc = LC(input[sp]);
                if (sc == 'z' || sc == 'm') {
                    ep = sp;
                    dc = sc;
                    has_dimension = 1;
                }
            }
        }
        if (dc == 'z') {
            has_dimension = 1;
            ep++;
            if (ep < n_bytes && LC(input[ep]) == 'm') ep++;  // ZM
        } else if (dc == 'm') {
            has_dimension = 1;
            ep++;
        }
    }

    // Skip whitespace before EMPTY or (
    while (ep < n_bytes && (input[ep] == ' ' || input[ep] == '\t')) ep++;

    // Check for EMPTY keyword (case-insensitive)
    unsigned char is_empty = 0;
    if (ep + 4 < geom_end
        && LC(input[ep]) == 'e'
        && LC(input[ep + 1]) == 'm'
        && LC(input[ep + 2]) == 'p'
        && LC(input[ep + 3]) == 't'
        && LC(input[ep + 4]) == 'y') {
        is_empty = 1;
    }
    empty_flags[idx] = is_empty;

    // A recognized family prefix is not sufficient: the keyword must end at
    // a WKT delimiter/dimension suffix, and the suffix parser must land on an
    // opening parenthesis or an exact EMPTY token. This rejects spellings such
    // as POINTXYZ(...) and POINT FOO(...), rather than truncating the type.
    if (
        tag >= 0
        && (
            !type_terminated
            || (!is_empty && (ep >= geom_end || input[ep] != '('))
        )
    ) {
        family_tags[idx] = -4;
        empty_flags[idx] = 0;
    }

    // EMPTY is a complete geometry spelling. GEOS rejects any non-whitespace
    // suffix, so encode that syntax failure in the family-tag export instead
    // of silently returning an empty owned row.
    if (is_empty) {
        for (long long tail = ep + 5; tail < geom_end; tail++) {
            const unsigned char tc = input[tail];
            if (tc != ' ' && tc != '\t' && tc != '\r' && tc != '\n') {
                family_tags[idx] = -4;
                break;
            }
        }
    }

    // The owned WKT layout is deliberately two-dimensional.  Treat Z/M/ZM
    // input as unsupported instead of silently pairing an interleaved 3D/4D
    // coordinate stream as x/y values.  Public constructors can then take
    // their observable compatibility fallback.
    if (has_dimension) family_tags[idx] = -2;

    #undef LC
}
"""

_WKT_CLASSIFY_NAMES: tuple[str, ...] = ("wkt_classify_geometry_type",)

_WKT_CAPACITY_SOURCE = r"""
// Capacity-backed WKT parsing.  Pass one validates/counts into fixed row
// capacity; after one host planning packet, pass two scatters directly into
// exact family buffers.  No device select/flatnonzero is used.

__device__ __forceinline__ bool wkt_space(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

__device__ __forceinline__ bool wkt_sep(unsigned char c) {
    return wkt_space(c) || c == ',' || c == '(' || c == ')';
}

__device__ __forceinline__ bool wkt_number_grammar(
    const unsigned char* input,
    long long start,
    long long end
) {
    long long pos = start;
    if (pos < end && (input[pos] == '+' || input[pos] == '-')) {
        pos++;
    }
    int mantissa_digits = 0;
    while (pos < end && input[pos] >= '0' && input[pos] <= '9') {
        pos++;
        mantissa_digits++;
    }
    if (pos < end && input[pos] == '.') {
        pos++;
        while (pos < end && input[pos] >= '0' && input[pos] <= '9') {
            pos++;
            mantissa_digits++;
        }
    }
    if (mantissa_digits == 0) return false;
    if (pos < end && (input[pos] == 'e' || input[pos] == 'E')) {
        pos++;
        if (pos < end && (input[pos] == '+' || input[pos] == '-')) {
            pos++;
        }
        int exponent_digits = 0;
        while (pos < end && input[pos] >= '0' && input[pos] <= '9') {
            pos++;
            exponent_digits++;
        }
        if (exponent_digits == 0) return false;
    }
    return pos == end;
}

// Exact decimal-vs-dyadic comparison used to refine a close fp64 estimate to
// strtod round-to-nearest-even semantics. IEEE-754 binary64
// conversion needs at most 768 significant decimal digits; later digits are
// represented by a sticky bit.  Finite binary64 inputs require fewer than 96
// base-2^32 limbs after powers-of-five scaling.
#define WKT_BIG_LIMBS 96
#define WKT_MAX_DECIMAL_DIGITS 768

struct WktBigUInt {
    unsigned int limb[WKT_BIG_LIMBS];
    int size;
};

__device__ __forceinline__ void wkt_big_zero(WktBigUInt* value) {
    #pragma unroll 1
    for (int i = 0; i < WKT_BIG_LIMBS; i++) value->limb[i] = 0;
    value->size = 0;
}

__device__ __forceinline__ void wkt_big_from_u64(
    WktBigUInt* value,
    unsigned long long input
) {
    wkt_big_zero(value);
    if (!input) return;
    value->limb[0] = (unsigned int)input;
    value->limb[1] = (unsigned int)(input >> 32);
    value->size = value->limb[1] ? 2 : 1;
}

__device__ __forceinline__ void wkt_big_copy(
    WktBigUInt* output,
    const WktBigUInt* input
) {
    output->size = input->size;
    #pragma unroll 1
    for (int i = 0; i < WKT_BIG_LIMBS; i++) output->limb[i] = input->limb[i];
}

__device__ __forceinline__ void wkt_big_mul_small(
    WktBigUInt* value,
    const unsigned int multiplier
) {
    unsigned long long carry = 0;
    #pragma unroll 1
    for (int i = 0; i < value->size; i++) {
        const unsigned long long product =
            (unsigned long long)value->limb[i] * multiplier + carry;
        value->limb[i] = (unsigned int)product;
        carry = product >> 32;
    }
    if (carry && value->size < WKT_BIG_LIMBS) {
        value->limb[value->size++] = (unsigned int)carry;
    }
}

__device__ __forceinline__ void wkt_big_add_small(
    WktBigUInt* value,
    const unsigned int addend
) {
    unsigned long long carry = addend;
    int i = 0;
    while (carry && i < value->size) {
        const unsigned long long sum = (unsigned long long)value->limb[i] + carry;
        value->limb[i++] = (unsigned int)sum;
        carry = sum >> 32;
    }
    if (carry && value->size < WKT_BIG_LIMBS) {
        value->limb[value->size++] = (unsigned int)carry;
    }
}

__device__ __forceinline__ void wkt_big_append_digit(
    WktBigUInt* value,
    const unsigned int digit
) {
    if (value->size == 0) {
        if (digit) {
            value->limb[0] = digit;
            value->size = 1;
        }
        return;
    }
    wkt_big_mul_small(value, 10);
    wkt_big_add_small(value, digit);
}

__device__ __forceinline__ int wkt_big_bit_length(const WktBigUInt* value) {
    if (!value->size) return 0;
    return (value->size - 1) * 32 + 32 - __clz(value->limb[value->size - 1]);
}

__device__ __forceinline__ unsigned int wkt_big_shifted_word(
    const WktBigUInt* value,
    const int shift,
    const int output_word
) {
    const int word_shift = shift >> 5;
    const int bit_shift = shift & 31;
    const int source = output_word - word_shift;
    unsigned long long word = 0;
    if (source >= 0 && source < value->size) {
        word = (unsigned long long)value->limb[source] << bit_shift;
    }
    if (bit_shift && source > 0 && source - 1 < value->size) {
        word |= (unsigned long long)value->limb[source - 1] >> (32 - bit_shift);
    }
    return (unsigned int)word;
}

__device__ __forceinline__ int wkt_big_compare_shifted(
    const WktBigUInt* left,
    int left_shift,
    const WktBigUInt* right,
    int right_shift
) {
    const int common = min(left_shift, right_shift);
    left_shift -= common;
    right_shift -= common;
    const int left_bits = wkt_big_bit_length(left) + left_shift;
    const int right_bits = wkt_big_bit_length(right) + right_shift;
    if (left_bits != right_bits) return left_bits < right_bits ? -1 : 1;
    const int words = (left_bits + 31) >> 5;
    for (int i = words - 1; i >= 0; i--) {
        const unsigned int lhs = wkt_big_shifted_word(left, left_shift, i);
        const unsigned int rhs = wkt_big_shifted_word(right, right_shift, i);
        if (lhs != rhs) return lhs < rhs ? -1 : 1;
    }
    return 0;
}

struct WktDecimal {
    WktBigUInt coefficient;
    int exponent10;
    int significant_digits;
    bool negative;
    bool sticky;
};

__device__ __forceinline__ bool wkt_parse_decimal_fast_exact(
    const unsigned char* chars,
    long long begin,
    const long long end,
    double* output
) {
    bool negative = false;
    if (begin < end && (chars[begin] == '+' || chars[begin] == '-')) {
        negative = chars[begin] == '-';
        begin++;
    }
    unsigned long long coefficient = 0;
    bool decimal = false;
    bool overflow = false;
    int fractional_digits = 0;
    int explicit_exponent = 0;
    int exponent_sign = 1;
    for (long long pos = begin; pos < end; pos++) {
        const unsigned char c = chars[pos];
        if (c == '.') {
            decimal = true;
            continue;
        }
        if (c == 'e' || c == 'E') {
            pos++;
            if (pos < end && (chars[pos] == '+' || chars[pos] == '-')) {
                exponent_sign = chars[pos] == '-' ? -1 : 1;
                pos++;
            }
            for (; pos < end; pos++) {
                explicit_exponent =
                    min(1000000, explicit_exponent * 10 + (int)(chars[pos] - '0'));
            }
            break;
        }
        const unsigned int digit = (unsigned int)(c - '0');
        if (decimal) fractional_digits++;
        if (coefficient > (0xffffffffffffffffULL - digit) / 10ULL) {
            overflow = true;
        } else if (!overflow) {
            coefficient = coefficient * 10ULL + digit;
        }
    }
    if (overflow) return false;
    int exponent10 = exponent_sign * explicit_exponent - fractional_digits;
    while (coefficient && coefficient % 10ULL == 0) {
        coefficient /= 10ULL;
        exponent10++;
    }
    if (coefficient == 0) {
        *output = negative ? -0.0 : 0.0;
        return true;
    }
    constexpr unsigned long long max_exact_integer = 1ULL << 53;
    if (coefficient > max_exact_integer) return false;
    double magnitude;
    if (exponent10 >= 0) {
        unsigned long long integer = coefficient;
        for (int i = 0; i < exponent10; i++) {
            if (integer > max_exact_integer / 10ULL) return false;
            integer *= 10ULL;
        }
        magnitude = (double)integer;
    } else {
        const int divisor_power = -exponent10;
        // 10^15 and the numerator are exactly representable integers, so the
        // IEEE division performs the one required round-to-nearest-even step.
        if (divisor_power > 15) return false;
        unsigned long long divisor = 1;
        for (int i = 0; i < divisor_power; i++) divisor *= 10ULL;
        magnitude = (double)coefficient / (double)divisor;
    }
    *output = negative ? -magnitude : magnitude;
    return true;
}

__device__ __forceinline__ double wkt_decimal_close_estimate(
    const unsigned char* chars,
    long long begin,
    const long long end
) {
    if (begin < end && (chars[begin] == '+' || chars[begin] == '-')) begin++;
    unsigned long long mantissa = 0;
    int stored = 0;
    int significant = 0;
    int fractional = 0;
    int explicit_exponent = 0;
    int exponent_sign = 1;
    bool decimal = false;
    bool started = false;
    for (long long pos = begin; pos < end; pos++) {
        const unsigned char c = chars[pos];
        if (c == '.') { decimal = true; continue; }
        if (c == 'e' || c == 'E') {
            pos++;
            if (pos < end && (chars[pos] == '+' || chars[pos] == '-')) {
                exponent_sign = chars[pos] == '-' ? -1 : 1;
                pos++;
            }
            for (; pos < end; pos++) {
                explicit_exponent = min(
                    1000000,
                    explicit_exponent * 10 + (int)(chars[pos] - '0')
                );
            }
            break;
        }
        const unsigned int digit = (unsigned int)(c - '0');
        if (decimal) fractional++;
        started = started || digit != 0;
        if (!started) continue;
        significant++;
        if (stored < 18) {
            mantissa = mantissa * 10ULL + digit;
            stored++;
        }
    }
    if (!mantissa) return 0.0;
    int exponent10 = exponent_sign * explicit_exponent - fractional
        + max(significant - stored, 0);
    if (exponent10 < -308) {
        const int shift = min(stored - 1, -308 - exponent10);
        double divisor = 1.0;
        for (int i = 0; i < shift; i++) divisor *= 10.0;
        return ((double)mantissa / divisor)
            * pow(10.0, (double)(exponent10 + shift));
    }
    return (double)mantissa * pow(10.0, (double)exponent10);
}

__device__ __forceinline__ WktDecimal wkt_parse_decimal_exact(
    const unsigned char* chars,
    long long begin,
    const long long end
) {
    WktDecimal result;
    wkt_big_zero(&result.coefficient);
    result.exponent10 = 0;
    result.significant_digits = 0;
    result.negative = false;
    result.sticky = false;
    if (begin < end && (chars[begin] == '+' || chars[begin] == '-')) {
        result.negative = chars[begin] == '-';
        begin++;
    }
    bool decimal = false;
    bool significant = false;
    int fractional_digits = 0;
    int discarded_digits = 0;
    int explicit_exponent = 0;
    int exponent_sign = 1;
    for (long long pos = begin; pos < end; pos++) {
        const unsigned char c = chars[pos];
        if (c == '.') {
            decimal = true;
            continue;
        }
        if (c == 'e' || c == 'E') {
            pos++;
            if (pos < end && (chars[pos] == '+' || chars[pos] == '-')) {
                exponent_sign = chars[pos] == '-' ? -1 : 1;
                pos++;
            }
            for (; pos < end; pos++) {
                const int digit = chars[pos] - '0';
                explicit_exponent = min(1000000, explicit_exponent * 10 + digit);
            }
            break;
        }
        const unsigned int digit = (unsigned int)(c - '0');
        if (decimal) fractional_digits++;
        significant = significant || digit != 0;
        if (!significant) continue;
        if (result.significant_digits < WKT_MAX_DECIMAL_DIGITS) {
            wkt_big_append_digit(&result.coefficient, digit);
            result.significant_digits++;
        } else {
            discarded_digits++;
            result.sticky = result.sticky || digit != 0;
        }
    }
    result.exponent10 =
        exponent_sign * explicit_exponent - fractional_digits + discarded_digits;
    return result;
}

__device__ __forceinline__ void wkt_binary64_components(
    const unsigned long long bits,
    unsigned long long* significand,
    int* exponent2
) {
    const unsigned long long fraction = bits & 0x000fffffffffffffULL;
    const int exponent = (int)((bits >> 52) & 0x7ffULL);
    if (exponent == 0) {
        *significand = fraction;
        *exponent2 = -1074;
    } else {
        *significand = (1ULL << 52) | fraction;
        *exponent2 = exponent - 1023 - 52;
    }
}

__device__ __forceinline__ void wkt_binary64_midpoint(
    const unsigned long long lower_bits,
    const unsigned long long upper_bits,
    unsigned long long* midpoint_significand,
    int* midpoint_exponent2
) {
    unsigned long long lower_sig;
    unsigned long long upper_sig;
    int lower_exp;
    int upper_exp;
    wkt_binary64_components(lower_bits, &lower_sig, &lower_exp);
    if (upper_bits == 0x7ff0000000000000ULL) {
        // Mathematical successor of max-finite for overflow rounding is 2^1024.
        upper_sig = 1;
        upper_exp = 1024;
    } else {
        wkt_binary64_components(upper_bits, &upper_sig, &upper_exp);
    }
    const int common_exp = min(lower_exp, upper_exp);
    *midpoint_significand =
        (lower_sig << (lower_exp - common_exp))
        + (upper_sig << (upper_exp - common_exp));
    *midpoint_exponent2 = common_exp - 1;
}

__device__ __forceinline__ int wkt_compare_decimal_to_dyadic(
    const WktDecimal* decimal,
    const unsigned long long dyadic_significand,
    const int dyadic_exponent2
) {
    WktBigUInt left;
    WktBigUInt right;
    wkt_big_copy(&left, &decimal->coefficient);
    wkt_big_from_u64(&right, dyadic_significand);
    int left_shift = 0;
    int right_shift = dyadic_exponent2;
    if (decimal->exponent10 >= 0) {
        for (int i = 0; i < decimal->exponent10; i++) wkt_big_mul_small(&left, 5);
        left_shift = decimal->exponent10;
    } else {
        const int power = -decimal->exponent10;
        for (int i = 0; i < power; i++) wkt_big_mul_small(&right, 5);
        right_shift += power;
    }
    const int comparison =
        wkt_big_compare_shifted(&left, left_shift, &right, right_shift);
    return comparison == 0 && decimal->sticky ? 1 : comparison;
}

extern "C" __global__ void __launch_bounds__(256, 4)
wkt_capacity_scatter_geom_starts(
    const unsigned char* __restrict__ is_geom_start,
    const int* __restrict__ start_prefix,
    long long* __restrict__ geom_starts,
    const int row_capacity,
    const long long n_bytes
) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_bytes || !is_geom_start[idx]) return;
    const int row = start_prefix[idx] - 1;
    if (row >= 0 && row < row_capacity) geom_starts[row] = idx;
}

extern "C" __global__ void __launch_bounds__(256, 4)
wkt_capacity_count_validate(
    const unsigned char* __restrict__ input,
    const long long* __restrict__ geom_starts,
    const signed char* __restrict__ family_tags,
    const unsigned char* __restrict__ empty_flags,
    signed char* __restrict__ status,
    int* __restrict__ pair_counts,
    int* __restrict__ token_counts,
    int* __restrict__ polygon_ring_counts,
    int* __restrict__ multiline_part_counts,
    int* __restrict__ multipolygon_part_counts,
    int* __restrict__ multipolygon_ring_counts,
    const int row_capacity,
    const long long n_bytes
) {
    const int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= row_capacity) return;
    const long long begin = geom_starts[row];
    const long long end = row + 1 < row_capacity
        ? min(geom_starts[row + 1], n_bytes)
        : n_bytes;
    status[row] = 0;
    pair_counts[row] = 0;
    token_counts[row] = 0;
    polygon_ring_counts[row] = 0;
    multiline_part_counts[row] = 0;
    multipolygon_part_counts[row] = 0;
    multipolygon_ring_counts[row] = 0;
    if (begin >= n_bytes || begin >= end) return;

    signed char tag = family_tags[row];
    if (tag < 0) {
        status[row] = tag;
        return;
    }
    if (empty_flags[row]) return;

    long long first_open = begin;
    while (first_open < end && input[first_open] != '(') first_open++;
    if (first_open >= end) {
        status[row] = -4;
        return;
    }

    int depth = 0;
    int token_count = 0;
    int space_count = 0;
    // Significant-token state: 0=start, 1=open, 2=x, 3=y, 4=comma, 5=close.
    // Validate separators when encountered so a comma cannot disappear at a
    // sequence boundary merely because no later numeric token consumes it.
    int last_kind = 0;
    int ring_pairs = 0;
    int part_pairs = 0;
    int part_rings = 0;
    int wrapped_multipoints = 0;
    bool structural_bad = false;
    bool shape_bad = false;
    bool completed = false;

    for (long long pos = first_open; pos < end;) {
        const unsigned char c = input[pos];
        if (completed) {
            if (!wkt_space(c)) structural_bad = true;
            pos++;
            continue;
        }
        if (c == '(') {
            if (last_kind == 2 || last_kind == 3 || last_kind == 5) {
                shape_bad = true;
            }
            depth++;
            if (tag == 2 && depth == 2) {
                polygon_ring_counts[row]++;
                ring_pairs = 0;
            } else if (tag == 3 && depth == 2) {
                wrapped_multipoints++;
                part_pairs = 0;
            } else if (tag == 4 && depth == 2) {
                multiline_part_counts[row]++;
                part_pairs = 0;
            } else if (tag == 5 && depth == 2) {
                multipolygon_part_counts[row]++;
                part_rings = 0;
            } else if (tag == 5 && depth == 3) {
                multipolygon_ring_counts[row]++;
                part_rings++;
                ring_pairs = 0;
            }
            last_kind = 1;
            space_count = 0;
            pos++;
            continue;
        }
        if (c == ')') {
            if (depth <= 0) {
                structural_bad = true;
                pos++;
                continue;
            }
            if (last_kind == 1 || last_kind == 2 || last_kind == 4) {
                shape_bad = true;
            }
            if ((tag == 2 && depth == 2) || (tag == 5 && depth == 3)) {
                if (ring_pairs < 4) shape_bad = true;
            }
            if ((tag == 3 || tag == 4) && depth == 2) {
                if ((tag == 3 && part_pairs != 1) || (tag == 4 && part_pairs < 2)) {
                    shape_bad = true;
                }
            }
            if (tag == 5 && depth == 2 && part_rings < 1) shape_bad = true;
            depth--;
            if (depth == 0) completed = true;
            last_kind = 5;
            space_count = 0;
            pos++;
            continue;
        }
        if (c == ',') {
            if (last_kind != 3 && last_kind != 5) shape_bad = true;
            last_kind = 4;
            space_count = 0;
            pos++;
            continue;
        }
        if (wkt_space(c)) {
            space_count++;
            pos++;
            continue;
        }

        const long long token_start = pos;
        while (pos < end && !wkt_sep(input[pos])) pos++;
        if (!wkt_number_grammar(input, token_start, pos)) shape_bad = true;
        const bool is_x = (token_count & 1) == 0;
        if (is_x) {
            if (last_kind != 1 && last_kind != 4) shape_bad = true;
            last_kind = 2;
        } else {
            if (last_kind != 2 || space_count == 0) shape_bad = true;
            last_kind = 3;
        }
        const int expected_depth = tag == 2 ? 2 : (tag == 4 ? 2 : (tag == 5 ? 3 : 1));
        if (tag == 3) {
            if (depth != 1 && depth != 2) shape_bad = true;
        } else if (depth != expected_depth) {
            shape_bad = true;
        }
        if (!is_x) {
            if ((tag == 2 && depth == 2) || (tag == 5 && depth == 3)) {
                ring_pairs++;
            }
            if ((tag == 3 || tag == 4) && depth == 2) part_pairs++;
        }
        token_count++;
        space_count = 0;
    }

    if (!completed || depth != 0) structural_bad = true;
    if (token_count & 1) status[row] = -3;
    const int pairs = token_count >> 1;
    token_counts[row] = token_count;
    pair_counts[row] = pairs;
    if (tag == 0 && pairs != 1) shape_bad = true;
    if (tag == 1 && pairs < 2) shape_bad = true;
    if (tag == 2 && polygon_ring_counts[row] < 1) shape_bad = true;
    if (tag == 3 && (pairs < 1 || (wrapped_multipoints && wrapped_multipoints != pairs))) {
        shape_bad = true;
    }
    if (tag == 4 && multiline_part_counts[row] < 1) shape_bad = true;
    if (tag == 5 && multipolygon_part_counts[row] < 1) shape_bad = true;
    if (structural_bad) status[row] = -4;
    if (shape_bad) status[row] = -5;
    if (token_count & 1) status[row] = -3;
    // Preserve GEOS' specific public diagnostic for a one-point LineString.
    if (tag == 1 && pairs == 1 && !structural_bad && !(token_count & 1)) {
        status[row] = -7;
    }
}

extern "C" __global__ void __launch_bounds__(256, 4)
wkt_capacity_scatter_numeric_tokens(
    const unsigned char* __restrict__ input,
    const long long* __restrict__ geom_starts,
    const int* __restrict__ token_offsets_by_row,
    long long* __restrict__ out_starts,
    long long* __restrict__ out_ends,
    const int row_capacity,
    const long long n_bytes
) {
    const int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= row_capacity || geom_starts[row] >= n_bytes) return;
    const long long end = row + 1 < row_capacity
        ? min(geom_starts[row + 1], n_bytes)
        : n_bytes;
    long long pos = geom_starts[row];
    while (pos < end && input[pos] != '(') pos++;
    int token = token_offsets_by_row[row];
    while (pos < end) {
        if (wkt_sep(input[pos])) {
            pos++;
            continue;
        }
        out_starts[token] = pos;
        while (pos < end && !wkt_sep(input[pos])) pos++;
        out_ends[token++] = pos;
    }
}

extern "C" __global__ void
wkt_capacity_refine_fp64(
    const unsigned char* __restrict__ input,
    const long long* __restrict__ token_starts,
    const long long* __restrict__ token_ends,
    double* __restrict__ parsed_values,
    const int token_capacity
) {
    const int token = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (token >= token_capacity) return;
    const long long begin = token_starts[token];
    const long long end = token_ends[token];
    if (begin >= end) return;
    double fast_exact;
    if (wkt_parse_decimal_fast_exact(input, begin, end, &fast_exact)) {
        parsed_values[token] = fast_exact;
        return;
    }
    const WktDecimal decimal = wkt_parse_decimal_exact(input, begin, end);
    union { double value; unsigned long long bits; } estimate;
    const unsigned long long sign_bit = decimal.negative ? (1ULL << 63) : 0;
    if (decimal.coefficient.size == 0) {
        estimate.bits = sign_bit;
        parsed_values[token] = estimate.value;
        return;
    }

    // Scientific exponents outside the binary64 boundary decades prove the
    // result without powers-of-five scaling. Values below 1e-324 are strictly
    // below half the minimum subnormal; values at or above 1e309 are strictly
    // beyond the finite overflow midpoint. Keep decades -324 and 308 on the
    // exact path because they contain the subnormal and max-finite boundaries.
    const int scientific_exponent =
        decimal.exponent10 + decimal.significant_digits - 1;
    if (scientific_exponent < -324) {
        estimate.bits = sign_bit;
        parsed_values[token] = estimate.value;
        return;
    }
    if (scientific_exponent > 308) {
        estimate.bits = sign_bit | 0x7ff0000000000000ULL;
        parsed_values[token] = estimate.value;
        return;
    }

    estimate.value = wkt_decimal_close_estimate(input, begin, end);

    unsigned long long magnitude = estimate.bits & 0x7fffffffffffffffULL;
    if (magnitude >= 0x7ff0000000000000ULL) magnitude = 0x7fefffffffffffffULL;
    // The bounded mantissa path provides a close estimate. Exact midpoint
    // tests both prove and correct it; ordinary inputs exit immediately.
    for (int iteration = 0; iteration < 64; iteration++) {
        bool moved = false;
        const bool odd = (magnitude & 1ULL) != 0;
        if (magnitude > 0) {
            unsigned long long midpoint_sig;
            int midpoint_exp;
            wkt_binary64_midpoint(
                magnitude - 1,
                magnitude,
                &midpoint_sig,
                &midpoint_exp
            );
            const int lower_cmp = wkt_compare_decimal_to_dyadic(
                &decimal,
                midpoint_sig,
                midpoint_exp
            );
            if (lower_cmp < 0 || (lower_cmp == 0 && odd)) {
                magnitude--;
                moved = true;
            }
        }
        if (moved) continue;

        unsigned long long midpoint_sig;
        int midpoint_exp;
        const unsigned long long upper = magnitude == 0x7fefffffffffffffULL
            ? 0x7ff0000000000000ULL
            : magnitude + 1;
        wkt_binary64_midpoint(magnitude, upper, &midpoint_sig, &midpoint_exp);
        const int upper_cmp = wkt_compare_decimal_to_dyadic(
            &decimal,
            midpoint_sig,
            midpoint_exp
        );
        if (upper_cmp > 0 || (upper_cmp == 0 && odd)) {
            magnitude = upper;
            moved = true;
        }
        if (!moved) break;
        if (magnitude == 0x7ff0000000000000ULL) break;
    }
    estimate.bits = sign_bit | magnitude;
    parsed_values[token] = estimate.value;
}

extern "C" __global__ void __launch_bounds__(256, 4)
wkt_capacity_validate_ring_closure(
    const unsigned char* __restrict__ input,
    const long long* __restrict__ geom_starts,
    const signed char* __restrict__ family_tags,
    const unsigned char* __restrict__ empty_flags,
    const int* __restrict__ token_offsets_by_row,
    const double* __restrict__ parsed_values,
    signed char* __restrict__ status,
    const int row_capacity,
    const long long n_bytes
) {
    const int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= row_capacity || status[row] != 0 || empty_flags[row]) return;
    const signed char tag = family_tags[row];
    if (tag != 2 && tag != 5) return;
    const long long end = row + 1 < row_capacity
        ? min(geom_starts[row + 1], n_bytes)
        : n_bytes;
    long long pos = geom_starts[row];
    while (pos < end && input[pos] != '(') pos++;
    int depth = 0;
    int token = token_offsets_by_row[row];
    int ring_pairs = 0;
    double first_x = 0.0;
    double first_y = 0.0;
    double last_x = 0.0;
    double last_y = 0.0;
    while (pos < end) {
        const unsigned char c = input[pos];
        if (c == '(') {
            depth++;
            if ((tag == 2 && depth == 2) || (tag == 5 && depth == 3)) {
                ring_pairs = 0;
            }
            pos++;
            continue;
        }
        if (c == ')') {
            if ((tag == 2 && depth == 2) || (tag == 5 && depth == 3)) {
                if (ring_pairs > 0 &&
                    (first_x != last_x || first_y != last_y)) {
                    status[row] = -6;
                }
            }
            depth--;
            pos++;
            continue;
        }
        if (wkt_sep(c)) {
            pos++;
            continue;
        }
        while (pos < end && !wkt_sep(input[pos])) pos++;
        const int ordinal = token - token_offsets_by_row[row];
        const double value = parsed_values[token++];
        if (ordinal & 1) {
            const double x = parsed_values[token - 2];
            if (ring_pairs == 0) {
                first_x = x;
                first_y = value;
            }
            last_x = x;
            last_y = value;
            ring_pairs++;
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 4)
wkt_capacity_scatter_family(
    const unsigned char* __restrict__ input,
    const long long* __restrict__ geom_starts,
    const signed char* __restrict__ family_tags,
    const unsigned char* __restrict__ empty_flags,
    const int* __restrict__ family_row_offsets,
    const int* __restrict__ coordinate_offsets,
    const int* __restrict__ first_offsets,
    const int* __restrict__ second_offsets,
    const int* __restrict__ token_offsets_by_row,
    const double* __restrict__ parsed_values,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    unsigned char* __restrict__ out_empty,
    int* __restrict__ out_geometry,
    int* __restrict__ out_first,
    int* __restrict__ out_second,
    const int target_tag,
    const int row_count,
    const long long n_bytes
) {
    const int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= row_count || family_tags[row] != target_tag) return;
    const int family_row = family_row_offsets[row];
    out_empty[family_row] = empty_flags[row];
    const int coord_base = coordinate_offsets[row];
    const int first_base = first_offsets ? first_offsets[row] : 0;
    const int second_base = second_offsets ? second_offsets[row] : 0;
    if (target_tag == 2 || target_tag == 4 || target_tag == 5) {
        out_geometry[family_row] = first_base;
        out_geometry[family_row + 1] = first_offsets[row + 1];
    } else {
        out_geometry[family_row] = coord_base;
        out_geometry[family_row + 1] = coordinate_offsets[row + 1];
    }
    if (empty_flags[row]) return;

    const long long begin = geom_starts[row];
    const long long end = row + 1 < row_count ? geom_starts[row + 1] : n_bytes;
    long long pos = begin;
    while (pos < end && input[pos] != '(') pos++;
    int depth = 0;
    int token_count = 0;
    const int token_base = token_offsets_by_row[row];
    int first_index = 0;
    int second_index = 0;

    while (pos < end) {
        const unsigned char c = input[pos];
        if (c == '(') {
            depth++;
            if (target_tag == 2 && depth == 2) {
                out_first[first_base + first_index++] = coord_base + (token_count >> 1);
            } else if (target_tag == 4 && depth == 2) {
                out_first[first_base + first_index++] = coord_base + (token_count >> 1);
            } else if (target_tag == 5 && depth == 2) {
                out_first[first_base + first_index++] = second_base + second_index;
            } else if (target_tag == 5 && depth == 3) {
                out_second[second_base + second_index++] = coord_base + (token_count >> 1);
            }
            pos++;
            continue;
        }
        if (c == ')') {
            if (target_tag == 2 && depth == 2) {
                out_first[first_base + first_index] = coord_base + (token_count >> 1);
            } else if (target_tag == 4 && depth == 2) {
                out_first[first_base + first_index] = coord_base + (token_count >> 1);
            } else if (target_tag == 5 && depth == 3) {
                out_second[second_base + second_index] = coord_base + (token_count >> 1);
            } else if (target_tag == 5 && depth == 2) {
                out_first[first_base + first_index] = second_base + second_index;
            }
            depth--;
            pos++;
            continue;
        }
        if (wkt_sep(c)) {
            pos++;
            continue;
        }
        while (pos < end && !wkt_sep(input[pos])) pos++;
        const double value = parsed_values[token_base + token_count];
        const int pair = coord_base + (token_count >> 1);
        if ((token_count & 1) == 0) out_x[pair] = value;
        else out_y[pair] = value;
        token_count++;
    }
}
"""

_WKT_CAPACITY_NAMES: tuple[str, ...] = (
    "wkt_capacity_scatter_geom_starts",
    "wkt_capacity_count_validate",
    "wkt_capacity_scatter_numeric_tokens",
    "wkt_capacity_refine_fp64",
    "wkt_capacity_validate_ring_closure",
    "wkt_capacity_scatter_family",
)
