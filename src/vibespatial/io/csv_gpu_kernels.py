"""NVRTC kernel sources for GPU CSV reader."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Kernel sources (Tier 1 NVRTC) -- integer-only byte classification,
# no floating-point computation, so no PrecisionPlan needed.
# ---------------------------------------------------------------------------

_CSV_QUOTE_TOGGLE_SOURCE = r"""
extern "C" __global__ void __launch_bounds__(256, 4)
csv_quote_toggle(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    long long n
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    // CSV uses "" (doubled quotes) for escaping, NOT backslash escaping.
    // Simply emit 1 at every " character.  Two consecutive quotes produce
    // two toggles that cancel each other in the cumulative-sum parity.
    output[idx] = (input[idx] == '"') ? (unsigned char)1 : (unsigned char)0;
}
"""

_CSV_QUOTE_TOGGLE_NAMES: tuple[str, ...] = ("csv_quote_toggle",)

_CSV_FIND_ROW_ENDS_SOURCE = r"""
extern "C" __global__ void __launch_bounds__(256, 4)
csv_find_row_ends(
    const unsigned char* __restrict__ input,
    const unsigned char* __restrict__ quote_parity,
    unsigned char* __restrict__ is_row_end,
    long long n
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // A row ends at \n that is outside quotes (parity == 0).
    // For \r\n sequences, only the \n is marked as the row end;
    // the \r is ignored (it will be stripped during field extraction).
    unsigned char b = input[idx];
    is_row_end[idx] = (b == '\n' && quote_parity[idx] == 0)
                      ? (unsigned char)1
                      : (unsigned char)0;
}
"""

_CSV_FIND_ROW_ENDS_NAMES: tuple[str, ...] = ("csv_find_row_ends",)

_CSV_FIND_DELIMITERS_SOURCE = r"""
extern "C" __global__ void __launch_bounds__(256, 4)
csv_find_delimiters(
    const unsigned char* __restrict__ input,
    const unsigned char* __restrict__ quote_parity,
    unsigned char* __restrict__ is_delimiter,
    long long n,
    int delimiter
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Mark delimiter characters that are outside quoted fields.
    unsigned char b = input[idx];
    is_delimiter[idx] = (b == (unsigned char)delimiter && quote_parity[idx] == 0)
                        ? (unsigned char)1
                        : (unsigned char)0;
}
"""

_CSV_FIND_DELIMITERS_NAMES: tuple[str, ...] = ("csv_find_delimiters",)

# ---------------------------------------------------------------------------
# Hex-to-binary WKB decode kernel (Tier 1 NVRTC)
#
# Two entry points for count-scatter pattern:
#   csv_hex_wkb_count  -- per-row binary byte count + validity
#   csv_hex_wkb_decode -- scatter hex-decoded bytes into output buffer
#
# Integer-only byte classification, no PrecisionPlan needed.
# ---------------------------------------------------------------------------

_CSV_HEX_WKB_SOURCE = r"""
// Convert a hex ASCII character to its 4-bit value.
// Returns -1 for invalid characters.
__device__ __forceinline__ int hex_nibble(unsigned char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// Returns 1 if the character is ASCII whitespace (space, tab, \r, \n).
__device__ __forceinline__ int is_ws(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

extern "C" __global__ void __launch_bounds__(256, 4)
csv_hex_wkb_count(
    const unsigned char* __restrict__ data,
    const long long*     __restrict__ starts,
    const long long*     __restrict__ ends,
    int*                 __restrict__ counts,
    unsigned char*       __restrict__ valid,
    int n_rows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    long long s = starts[row];
    long long e = ends[row];

    // Skip leading whitespace.
    while (s < e && is_ws(data[s])) s++;
    // Skip trailing whitespace.
    while (e > s && is_ws(data[e - 1])) e--;

    long long hex_len = e - s;
    if (hex_len <= 0 || (hex_len & 1) != 0) {
        // Empty or odd-length: invalid hex WKB.
        counts[row] = 0;
        valid[row] = 0;
        return;
    }

    // Quick validation: check all characters are valid hex.
    int ok = 1;
    for (long long i = s; i < e; i++) {
        if (hex_nibble(data[i]) < 0) { ok = 0; break; }
    }
    if (!ok) {
        counts[row] = 0;
        valid[row] = 0;
        return;
    }

    counts[row] = (int)(hex_len >> 1);
    valid[row] = 1;
}

extern "C" __global__ void __launch_bounds__(256, 4)
csv_hex_wkb_decode(
    const unsigned char* __restrict__ data,
    const long long*     __restrict__ starts,
    const long long*     __restrict__ ends,
    const int*           __restrict__ offsets,
    const unsigned char* __restrict__ valid,
    unsigned char*       __restrict__ output,
    int n_rows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    if (!valid[row]) return;

    long long s = starts[row];
    long long e = ends[row];

    // Skip whitespace (same logic as count kernel).
    while (s < e && is_ws(data[s])) s++;
    while (e > s && is_ws(data[e - 1])) e--;

    int out_off = offsets[row];
    long long hex_len = e - s;
    int n_bytes = (int)(hex_len >> 1);

    for (int i = 0; i < n_bytes; i++) {
        int hi = hex_nibble(data[s + 2 * i]);
        int lo = hex_nibble(data[s + 2 * i + 1]);
        output[out_off + i] = (unsigned char)((hi << 4) | lo);
    }
}
"""

_CSV_HEX_WKB_NAMES: tuple[str, ...] = ("csv_hex_wkb_count", "csv_hex_wkb_decode")
