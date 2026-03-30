"""NVRTC kernel sources for GPU structural scanning primitives."""

from __future__ import annotations

_QUOTE_TOGGLE_SOURCE = r"""
extern "C" __global__ void quote_toggle(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    long long n
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    unsigned char b = input[idx];
    if (b != '"') {
        output[idx] = 0;
        return;
    }
    // Check for backslash escape: count consecutive backslashes before this quote
    int backslash_count = 0;
    long long j = idx - 1;
    while (j >= 0 && input[j] == '\\') {
        backslash_count++;
        j--;
    }
    // Quote is escaped if preceded by odd number of backslashes
    output[idx] = (backslash_count % 2 == 0) ? 1 : 0;
}
"""

_QUOTE_TOGGLE_NAMES: tuple[str, ...] = ("quote_toggle",)

_DEPTH_DELTAS_TEMPLATE = r"""
extern "C" __global__ void compute_depth_deltas(
    const unsigned char* __restrict__ input,
    const unsigned char* __restrict__ quote_parity,
    signed char* __restrict__ deltas,
    long long n
) {{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    signed char d = 0;
    if (quote_parity[idx] == 0) {{
        unsigned char b = input[idx];
        {open_checks}
        {close_checks}
    }}
    deltas[idx] = d;
}}
"""

_DEPTH_DELTAS_NAMES: tuple[str, ...] = ("compute_depth_deltas",)
