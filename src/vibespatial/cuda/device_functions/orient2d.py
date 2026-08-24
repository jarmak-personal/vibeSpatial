"""Shared CUDA device function: Shewchuk orient2d adaptive predicate."""

from __future__ import annotations

__all__ = ["ORIENT2D_DEVICE"]

ORIENT2D_DEVICE: str = r"""
/* ------------------------------------------------------------------ */
/* Shewchuk error-free primitives (fp64 exact predicates)             */
/* Shared via vibespatial.cuda.device_functions.orient2d              */
/* ------------------------------------------------------------------ */

/* Shewchuk two-product error-free transformation (GPU implementation).
   Given a, b: computes (p, e) such that a*b = p + e exactly.
   Uses Dekker's algorithm with FMA where available. */
__device__ inline void vs_two_product(double a, double b, double &p, double &e) {{
    p = a * b;
    e = fma(a, b, -p);
}}

/* Shewchuk two-sum error-free transformation.
   Given a, b: computes (s, e) such that a+b = s + e exactly. */
__device__ inline void vs_two_sum(double a, double b, double &s, double &e) {{
    s = a + b;
    double bv = s - a;
    double av = s - bv;
    double br = b - bv;
    double ar = a - av;
    e = ar + br;
}}

/* Error-free subtraction.  The tail is essential when the rounded coordinate
   difference loses low bits before the determinant products are formed. */
__device__ inline void vs_two_diff(double a, double b, double &x, double &tail) {{
    x = a - b;
    const double b_virtual = a - x;
    const double a_virtual = x + b_virtual;
    const double b_roundoff = b_virtual - b;
    const double a_roundoff = a - a_virtual;
    tail = a_roundoff + b_roundoff;
}}

/* Add one scalar to a nonoverlapping expansion, eliminating zero components. */
__device__ inline int vs_grow_expansion_zeroelim(
    int expansion_length,
    const double* expansion,
    double scalar,
    double* output
) {{
    double accumulator = scalar;
    int output_length = 0;
    for (int index = 0; index < expansion_length; ++index) {{
        double sum, roundoff;
        vs_two_sum(accumulator, expansion[index], sum, roundoff);
        if (roundoff != 0.0) output[output_length++] = roundoff;
        accumulator = sum;
    }}
    if (accumulator != 0.0 || output_length == 0) {{
        output[output_length++] = accumulator;
    }}
    return output_length;
}}

/* Add the exact two-component product a*b to an expansion.  The caller owns
   32-component expansion/scratch arrays; orient2d adds only eight products. */
__device__ inline int vs_add_product_to_expansion(
    int expansion_length,
    double* expansion,
    double* scratch,
    double a,
    double b
) {{
    double product, roundoff;
    vs_two_product(a, b, product, roundoff);
    int scratch_length = vs_grow_expansion_zeroelim(
        expansion_length, expansion, roundoff, scratch);
    return vs_grow_expansion_zeroelim(
        scratch_length, scratch, product, expansion);
}}

#define VS_BIG_COORD_LIMBS 33
#define VS_BIG_PRODUCT_LIMBS 66

__device__ inline void vs_big_clear(
    unsigned long long* value,
    int length
) {{
    for (int index = 0; index < length; ++index) value[index] = 0ull;
}}

__device__ inline bool vs_big_is_zero(
    const unsigned long long* value,
    int length
) {{
    for (int index = 0; index < length; ++index) {{
        if (value[index] != 0ull) return false;
    }}
    return true;
}}

__device__ inline int vs_big_compare(
    const unsigned long long* left,
    const unsigned long long* right,
    int length
) {{
    for (int index = length - 1; index >= 0; --index) {{
        if (left[index] > right[index]) return 1;
        if (left[index] < right[index]) return -1;
    }}
    return 0;
}}

__device__ inline void vs_big_add(
    const unsigned long long* left,
    const unsigned long long* right,
    unsigned long long* output,
    int length
) {{
    unsigned long long carry = 0ull;
    for (int index = 0; index < length; ++index) {{
        const unsigned long long partial = left[index] + right[index];
        const unsigned long long carry0 = partial < left[index];
        const unsigned long long total = partial + carry;
        const unsigned long long carry1 = total < partial;
        output[index] = total;
        carry = carry0 | carry1;
    }}
}}

/* Magnitude subtraction with the precondition left >= right. */
__device__ inline void vs_big_subtract(
    const unsigned long long* left,
    const unsigned long long* right,
    unsigned long long* output,
    int length
) {{
    unsigned long long borrow = 0ull;
    for (int index = 0; index < length; ++index) {{
        const unsigned long long partial = left[index] - right[index];
        const unsigned long long borrow0 = left[index] < right[index];
        const unsigned long long total = partial - borrow;
        const unsigned long long borrow1 = partial < borrow;
        output[index] = total;
        borrow = borrow0 | borrow1;
    }}
}}

/* Every finite binary64 number is an integer multiple of 2^-1074.  Encode
   that integer exactly as a bounded 2098-bit signed magnitude. */
__device__ inline void vs_big_from_double(
    double value,
    unsigned long long* magnitude,
    bool* negative
) {{
    vs_big_clear(magnitude, VS_BIG_COORD_LIMBS);
    const unsigned long long bits =
        (unsigned long long)__double_as_longlong(value);
    const unsigned long long fraction = bits & 0x000fffffffffffffull;
    const int exponent = (int)((bits >> 52) & 0x7ffull);
    const unsigned long long mantissa = exponent == 0
        ? fraction
        : fraction | 0x0010000000000000ull;
    *negative = mantissa != 0ull && (bits >> 63) != 0ull;
    if (mantissa == 0ull) return;
    const int shift = exponent == 0 ? 0 : exponent - 1;
    const int limb = shift >> 6;
    const int bit = shift & 63;
    magnitude[limb] = mantissa << bit;
    if (bit > 11) magnitude[limb + 1] = mantissa >> (64 - bit);
}}

__device__ inline void vs_big_signed_difference(
    double left_value,
    double right_value,
    unsigned long long* output,
    bool* negative
) {{
    unsigned long long left[VS_BIG_COORD_LIMBS];
    unsigned long long right[VS_BIG_COORD_LIMBS];
    bool left_negative = false;
    bool right_negative = false;
    vs_big_from_double(left_value, left, &left_negative);
    vs_big_from_double(right_value, right, &right_negative);
    if (left_negative != right_negative) {{
        vs_big_add(left, right, output, VS_BIG_COORD_LIMBS);
        *negative = left_negative;
        return;
    }}
    const int comparison = vs_big_compare(left, right, VS_BIG_COORD_LIMBS);
    if (comparison >= 0) {{
        vs_big_subtract(left, right, output, VS_BIG_COORD_LIMBS);
        *negative = left_negative;
    }} else {{
        vs_big_subtract(right, left, output, VS_BIG_COORD_LIMBS);
        *negative = !left_negative;
    }}
    if (vs_big_is_zero(output, VS_BIG_COORD_LIMBS)) *negative = false;
}}

__device__ inline void vs_big_multiply(
    const unsigned long long* left,
    const unsigned long long* right,
    unsigned long long* output
) {{
    vs_big_clear(output, VS_BIG_PRODUCT_LIMBS);
    for (int left_index = 0; left_index < VS_BIG_COORD_LIMBS; ++left_index) {{
        if (left[left_index] == 0ull) continue;
        for (int right_index = 0; right_index < VS_BIG_COORD_LIMBS; ++right_index) {{
            if (right[right_index] == 0ull) continue;
            const unsigned long long low = left[left_index] * right[right_index];
            const unsigned long long high = __umul64hi(
                left[left_index], right[right_index]);
            int output_index = left_index + right_index;
            const unsigned long long old_low = output[output_index];
            output[output_index] = old_low + low;
            unsigned long long carry = output[output_index] < old_low;
            ++output_index;
            const unsigned long long old_high = output[output_index];
            unsigned long long total = old_high + high;
            const unsigned long long high_carry = total < old_high;
            const unsigned long long before_carry = total;
            total += carry;
            const unsigned long long low_carry = total < before_carry;
            output[output_index] = total;
            carry = high_carry | low_carry;
            while (carry && ++output_index < VS_BIG_PRODUCT_LIMBS) {{
                const unsigned long long old = output[output_index];
                output[output_index] = old + 1ull;
                carry = output[output_index] == 0ull;
            }}
        }}
    }}
}}

/* Exact sign fallback over the entire finite binary64 exponent domain. */
__device__ __noinline__ int vs_orient2d_big_exact(
    double ax, double ay,
    double bx, double by,
    double cx, double cy
) {{
    if (!isfinite(ax) || !isfinite(ay) || !isfinite(bx)
        || !isfinite(by) || !isfinite(cx) || !isfinite(cy)) return 0;
    unsigned long long first[VS_BIG_COORD_LIMBS];
    unsigned long long second[VS_BIG_COORD_LIMBS];
    unsigned long long left_product[VS_BIG_PRODUCT_LIMBS];
    unsigned long long right_product[VS_BIG_PRODUCT_LIMBS];
    bool first_negative = false;
    bool second_negative = false;
    vs_big_signed_difference(ax, cx, first, &first_negative);
    vs_big_signed_difference(by, cy, second, &second_negative);
    const bool left_zero = vs_big_is_zero(first, VS_BIG_COORD_LIMBS)
        || vs_big_is_zero(second, VS_BIG_COORD_LIMBS);
    const bool left_negative = first_negative != second_negative;
    vs_big_multiply(first, second, left_product);

    vs_big_signed_difference(ay, cy, first, &first_negative);
    vs_big_signed_difference(bx, cx, second, &second_negative);
    const bool right_zero = vs_big_is_zero(first, VS_BIG_COORD_LIMBS)
        || vs_big_is_zero(second, VS_BIG_COORD_LIMBS);
    const bool right_negative = first_negative != second_negative;
    vs_big_multiply(first, second, right_product);

    if (left_zero && right_zero) return 0;
    if (left_zero) return right_negative ? 1 : -1;
    if (right_zero) return left_negative ? -1 : 1;
    if (left_negative != right_negative) return left_negative ? -1 : 1;
    const int comparison = vs_big_compare(
        left_product, right_product, VS_BIG_PRODUCT_LIMBS);
    if (comparison == 0) return 0;
    return comparison > 0
        ? (left_negative ? -1 : 1)
        : (left_negative ? 1 : -1);
}}

__device__ inline bool vs_product_expansion_is_safe(double left, double right) {{
    if (!isfinite(left) || !isfinite(right)) return false;
    if (left == 0.0 || right == 0.0) return true;
    const int product_exponent = ilogb(fabs(left)) + ilogb(fabs(right));
    return product_exponent >= -895 && product_exponent <= 895;
}}

/* Keep the expansion stack frame off the overwhelmingly common stage-A path.
   This function is intentionally not inlined. */
__device__ __noinline__ int vs_orient2d_exact(
    double ax, double ay,
    double bx, double by,
    double cx, double cy
) {{
    double acx, acxtail;
    double bcx, bcxtail;
    double acy, acytail;
    double bcy, bcytail;
    vs_two_diff(ax, cx, acx, acxtail);
    vs_two_diff(bx, cx, bcx, bcxtail);
    vs_two_diff(ay, cy, acy, acytail);
    vs_two_diff(by, cy, bcy, bcytail);

    if (
        !vs_product_expansion_is_safe(acx, bcy)
        || !vs_product_expansion_is_safe(acx, bcytail)
        || !vs_product_expansion_is_safe(acxtail, bcy)
        || !vs_product_expansion_is_safe(acxtail, bcytail)
        || !vs_product_expansion_is_safe(acy, bcx)
        || !vs_product_expansion_is_safe(acy, bcxtail)
        || !vs_product_expansion_is_safe(acytail, bcx)
        || !vs_product_expansion_is_safe(acytail, bcxtail)
    ) return vs_orient2d_big_exact(ax, ay, bx, by, cx, cy);

    /* Exact determinant of the stored binary64 inputs:
         (acx + acxtail) * (bcy + bcytail)
       - (acy + acytail) * (bcx + bcxtail). */
    double expansion[32];
    double scratch[32];
    int expansion_length = 1;
    expansion[0] = 0.0;
    expansion_length = vs_add_product_to_expansion(
        expansion_length, expansion, scratch, acx, bcy);
    expansion_length = vs_add_product_to_expansion(
        expansion_length, expansion, scratch, acx, bcytail);
    expansion_length = vs_add_product_to_expansion(
        expansion_length, expansion, scratch, acxtail, bcy);
    expansion_length = vs_add_product_to_expansion(
        expansion_length, expansion, scratch, acxtail, bcytail);
    expansion_length = vs_add_product_to_expansion(
        expansion_length, expansion, scratch, acy, -bcx);
    expansion_length = vs_add_product_to_expansion(
        expansion_length, expansion, scratch, acy, -bcxtail);
    expansion_length = vs_add_product_to_expansion(
        expansion_length, expansion, scratch, acytail, -bcx);
    expansion_length = vs_add_product_to_expansion(
        expansion_length, expansion, scratch, acytail, -bcxtail);

    for (int index = expansion_length - 1; index >= 0; --index) {{
        if (expansion[index] > 0.0) return 1;
        if (expansion[index] < 0.0) return -1;
    }}
    return 0;
}}

/* Shewchuk orient2d adaptive predicate.
   Returns the sign of det = (bx-ax)*(cy-ay) - (by-ay)*(cx-ax).
   The inexpensive stage-A error bound handles the common non-degenerate case;
   error-free arithmetic is reserved for determinants inside that bound.
   Returns: +1, 0, or -1  */
__device__ int vs_orient2d(
    double ax, double ay,
    double bx, double by,
    double cx, double cy
) {{
    const double acx = ax - cx;
    const double bcx = bx - cx;
    const double acy = ay - cy;
    const double bcy = by - cy;

    const double detleft_fast = acx * bcy;
    const double detright_fast = acy * bcx;
    const double det_fast = detleft_fast - detright_fast;
    const double detsum = fabs(detleft_fast) + fabs(detright_fast);

    /* Shewchuk's IEEE-754 stage-A bound:
       (3 + 16 * epsilon) * epsilon, epsilon = 2^-53.  A determinant
       outside this interval has an authoritative sign without expansion
       arithmetic. */
    const double ccwerrbound_a = 3.3306690738754716e-16;
    const double errbound = ccwerrbound_a * detsum;
    if (det_fast > errbound) return 1;
    if (det_fast < -errbound) return -1;

    return vs_orient2d_exact(ax, ay, bx, by, cx, cy);
}}

"""
