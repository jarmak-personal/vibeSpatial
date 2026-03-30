"""GPU property extraction primitives for GeoJSON Feature objects.

Extracts numeric, boolean, and null property values directly on GPU,
avoiding the per-feature ``orjson.loads()`` CPU bottleneck.  String
properties are intentionally NOT handled -- the caller falls back to
CPU for those columns.

Pipeline overview::

    # Inputs: d_bytes, d_quote_parity, d_depth, feature boundaries
    #
    # 1. Locate property key positions via pattern_match + depth filter
    # 2. Classify value types (string/bool/null/number/complex)
    # 3. For numeric columns: extract floats via number_boundaries + parse
    # 4. For boolean columns: NVRTC kernel reads "true"/"false"
    # 5. For null columns: mark in validity mask
    # 6. Schema inference on first N features (only D->H transfer)

The only host round-trip is schema inference (reading unique key names
and their types from a small sample).  All column extraction runs
entirely on GPU.

Kernel classification (ADR-0033):
    - classify_property_values: Tier 1 (byte-level pattern at specific positions)
    - extract_booleans: Tier 1 (byte-level "true"/"false" recognition)
    - All other operations: Tier 2 (CuPy element-wise / gpu_parse reuse)

Precision (ADR-0002):
    - Integer-only byte classification kernels -- no PrecisionPlan needed.
    - Numeric property values are parsed to fp64 via parse_ascii_floats
      (storage is always fp64 per ADR-0002).
"""
from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.io.gpu_parse.numeric import (
    extract_number_positions,
    parse_ascii_floats,
)
from vibespatial.io.gpu_parse.pattern import (
    mark_spans,
    pattern_match,
    span_boundaries,
)
from vibespatial.io.gpu_parse.properties_kernels import (
    _CLASSIFY_VALUE_NAMES,
    _CLASSIFY_VALUE_SOURCE,
    _EXTRACT_BOOL_NAMES,
    _EXTRACT_BOOL_SOURCE,
    _PROPERTY_NUM_BOUNDS_NAMES,
    _PROPERTY_NUM_BOUNDS_SOURCE,
)

if TYPE_CHECKING:
    import cupy as cp

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

# ctypes for int64 kernel params
KERNEL_PARAM_I64 = ctypes.c_longlong

# ---------------------------------------------------------------------------
# Value type constants (matches kernel output)
# ---------------------------------------------------------------------------
VTYPE_STRING: int = 0
VTYPE_BOOLEAN: int = 1
VTYPE_NULL: int = 2
VTYPE_NUMBER: int = 3
VTYPE_COMPLEX: int = 4  # nested object or array -- skip

# ---------------------------------------------------------------------------
# NVRTC kernel sources (Tier 1)
# ---------------------------------------------------------------------------

# Kernel name tuples
# ---------------------------------------------------------------------------
# NVRTC warmup (ADR-0034 Level 2)
# ---------------------------------------------------------------------------
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("gpu-parse-classify-value", _CLASSIFY_VALUE_SOURCE, _CLASSIFY_VALUE_NAMES),
    ("gpu-parse-extract-bool", _EXTRACT_BOOL_SOURCE, _EXTRACT_BOOL_NAMES),
    ("gpu-parse-prop-num-bounds", _PROPERTY_NUM_BOUNDS_SOURCE, _PROPERTY_NUM_BOUNDS_NAMES),
])


# ---------------------------------------------------------------------------
# Kernel compilation helpers
# ---------------------------------------------------------------------------

def _classify_value_kernels():
    return compile_kernel_group(
        "gpu-parse-classify-value",
        _CLASSIFY_VALUE_SOURCE,
        _CLASSIFY_VALUE_NAMES,
    )


def _extract_bool_kernels():
    return compile_kernel_group(
        "gpu-parse-extract-bool",
        _EXTRACT_BOOL_SOURCE,
        _EXTRACT_BOOL_NAMES,
    )


def _property_num_bounds_kernels():
    return compile_kernel_group(
        "gpu-parse-prop-num-bounds",
        _PROPERTY_NUM_BOUNDS_SOURCE,
        _PROPERTY_NUM_BOUNDS_NAMES,
    )


# ---------------------------------------------------------------------------
# Kernel launch helper
# ---------------------------------------------------------------------------

def _launch_kernel(runtime, kernel, n, params):
    """Launch a kernel with occupancy-based grid/block sizing."""
    grid, block = runtime.launch_config(kernel, int(n))
    runtime.launch(kernel, grid=grid, block=block, params=params)


# ---------------------------------------------------------------------------
# Property-specific number boundary detection
# ---------------------------------------------------------------------------

def _property_number_boundaries(
    d_bytes: cp.ndarray,
    d_quote_parity: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Identify start and end positions of numeric tokens in property values.

    Identical to ``number_boundaries`` from the numeric module, but adds
    ``:`` to the start-separator set and ``}`` to the end-separator set
    so that JSON property values like ``"population":42`` are correctly
    detected.

    Parameters
    ----------
    d_bytes : cp.ndarray
        Device-resident uint8 byte array.
    d_quote_parity : cp.ndarray
        Device-resident uint8 quote parity mask.

    Returns
    -------
    d_is_start : cp.ndarray
        Device-resident uint8 per-byte start mask.
    d_is_end : cp.ndarray
        Device-resident uint8 per-byte end mask.
    """
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    n = len(d_bytes)
    n_i64 = np.int64(n)

    kernels = _property_num_bounds_kernels()

    d_is_start = cp.zeros(n, dtype=cp.uint8)
    d_is_end = cp.zeros(n, dtype=cp.uint8)

    _launch_kernel(runtime, kernels["find_property_number_boundaries"], n, (
        (ptr(d_bytes), ptr(d_quote_parity), ptr(d_is_start), ptr(d_is_end), n_i64),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I64),
    ))

    return d_is_start, d_is_end


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_property_keys(
    d_bytes: cp.ndarray,
    d_quote_parity: cp.ndarray,
    d_depth: cp.ndarray,
    key_name: str,
    *,
    property_depth: int = 5,
) -> cp.ndarray:
    """Find positions of a specific property key at the correct depth.

    For GeoJSON FeatureCollection documents, property keys are at depth 5:
    FeatureCollection{1} > features[2] > Feature{3} > properties{4} > key{5}.

    Parameters
    ----------
    d_bytes : cp.ndarray
        Device-resident uint8 byte array.
    d_quote_parity : cp.ndarray
        Device-resident uint8 quote parity mask.
    d_depth : cp.ndarray
        Device-resident int32 bracket depth array.
    key_name : str
        The property key to search for (without quotes/colon).
    property_depth : int
        The bracket depth at which property keys appear.  Default 5
        for standard GeoJSON FeatureCollection.

    Returns
    -------
    cp.ndarray
        Device-resident int64 positions where the pattern starts.
        Each position points to the opening quote of the key.
    """
    # Build the pattern: "key_name":
    pattern = f'"{key_name}":'.encode("ascii")
    d_hits = pattern_match(d_bytes, pattern, d_quote_parity)

    # The colon is at offset len(pattern)-1 from the hit position.
    # Filter by depth at the colon position: must equal property_depth.
    colon_offset = len(pattern) - 1
    n = d_bytes.shape[0]

    # Compact gather: extract hit positions first (tiny array), then
    # check depth only at those positions.  Avoids allocating a full
    # n-element shifted depth array (4n bytes).
    d_hit_positions = cp.flatnonzero(d_hits).astype(cp.int64)
    del d_hits
    if d_hit_positions.shape[0] == 0:
        return d_hit_positions

    d_colon_positions = d_hit_positions + colon_offset
    # Clamp to valid range (pattern_match already zeroes near-end
    # hits, but guard against edge cases)
    d_colon_positions = cp.minimum(d_colon_positions, cp.int64(n - 1))
    d_depth_at_hits = d_depth[d_colon_positions]
    d_depth_ok = d_depth_at_hits == property_depth
    return d_hit_positions[d_depth_ok]


def _colon_positions_from_key_hits(
    d_key_positions: cp.ndarray,
    key_name: str,
) -> cp.ndarray:
    """Compute the byte offset of the ':' for each key hit.

    The pattern is ``"key_name":``, so the colon is at offset
    ``len(key_name) + 2`` (quote + key + quote + colon = len+3,
    but the hit position is at the opening quote, and the colon
    is len(key_name)+2 bytes later).
    """
    colon_offset = len(key_name) + 2  # opening-quote + key + closing-quote
    return d_key_positions + colon_offset


def _classify_values(
    d_bytes: cp.ndarray,
    d_colon_positions: cp.ndarray,
) -> cp.ndarray:
    """Classify the JSON value type after each colon position.

    Returns
    -------
    cp.ndarray
        Device-resident int8 array of value type codes.
    """
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    n_keys = d_colon_positions.shape[0]
    n_bytes = d_bytes.shape[0]

    kernels = _classify_value_kernels()
    d_types = cp.empty(n_keys, dtype=cp.int8)

    _launch_kernel(runtime, kernels["classify_property_values"], n_keys, (
        (
            ptr(d_bytes),
            ptr(d_colon_positions),
            ptr(d_types),
            np.int32(n_keys),
            np.int64(n_bytes),
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_I64,
        ),
    ))

    return d_types


def _extract_numeric_column(
    d_bytes: cp.ndarray,
    d_quote_parity: cp.ndarray,
    d_colon_positions: cp.ndarray,
    d_depth: cp.ndarray,
    n_features: int,
) -> cp.ndarray:
    """Extract a float64 value for each feature from a numeric property column.

    Uses span_boundaries to find the extent of each numeric value,
    then property_number_boundaries + parse_ascii_floats within those spans.

    Returns
    -------
    cp.ndarray
        Device-resident float64 array of shape ``(n_features,)``.
    """
    n_bytes = d_bytes.shape[0]

    # Find span end for each value (from colon to next structural delimiter).
    # skip_bytes=1 to skip past the colon itself.
    d_span_ends = span_boundaries(
        d_depth, d_colon_positions, n_bytes, skip_bytes=1,
    )

    # Create a region mask covering the value spans
    d_value_starts = d_colon_positions + 1
    d_mask = mark_spans(d_value_starts, d_span_ends, n_bytes)

    # Detect number boundaries within those spans.
    # Uses property-specific kernel that includes ':' as a start separator
    # and '}' as an end separator (the generic kernel misses "key":42 patterns).
    d_is_start, d_is_end = _property_number_boundaries(d_bytes, d_quote_parity)
    d_num_starts, d_num_ends = extract_number_positions(
        d_is_start, d_is_end, d_mask,
    )

    # Parse the numbers
    d_values = parse_ascii_floats(d_bytes, d_num_starts, d_num_ends)

    # We expect exactly one number per feature.  If there are fewer
    # (e.g., some features have null for this key), the caller handles
    # alignment via the per-feature mask.  If there are more (shouldn't
    # happen for scalar properties), take only the first n_features.
    if d_values.shape[0] > n_features:
        d_values = d_values[:n_features]

    return d_values


def _extract_boolean_column(
    d_bytes: cp.ndarray,
    d_colon_positions: cp.ndarray,
) -> cp.ndarray:
    """Extract boolean values for a property column.

    Returns
    -------
    cp.ndarray
        Device-resident uint8 array: 1 = true, 0 = false.
    """
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    n_keys = d_colon_positions.shape[0]
    n_bytes = d_bytes.shape[0]

    kernels = _extract_bool_kernels()
    d_bools = cp.empty(n_keys, dtype=cp.uint8)

    _launch_kernel(runtime, kernels["extract_booleans"], n_keys, (
        (
            ptr(d_bytes),
            ptr(d_colon_positions),
            ptr(d_bools),
            np.int32(n_keys),
            np.int64(n_bytes),
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_I64,
        ),
    ))

    return d_bools


# ---------------------------------------------------------------------------
# Schema inference
# ---------------------------------------------------------------------------

def infer_property_schema(
    d_bytes: cp.ndarray,
    d_quote_parity: cp.ndarray,
    d_depth: cp.ndarray,
    d_feature_starts: cp.ndarray,
    d_feature_ends: cp.ndarray,
    *,
    sample_size: int = 100,
    property_depth: int = 5,
) -> dict[str, int]:
    """Infer property names and their types from a sample of features.

    Scans the first ``sample_size`` features for property keys at the
    specified depth, classifies each key's value type, and returns a
    schema mapping property name to value type code.

    This is the ONLY acceptable D->H transfer in the property extraction
    pipeline.

    Parameters
    ----------
    d_bytes : cp.ndarray
        Device-resident uint8 byte array.
    d_quote_parity : cp.ndarray
        Device-resident uint8 quote parity mask.
    d_depth : cp.ndarray
        Device-resident int32 bracket depth array.
    d_feature_starts : cp.ndarray
        Device-resident int64 array of feature start byte offsets.
    d_feature_ends : cp.ndarray
        Device-resident int64 array of feature end byte offsets.
    sample_size : int
        Number of features to sample for schema detection.
    property_depth : int
        Bracket depth of property keys (default 5 for GeoJSON).

    Returns
    -------
    dict[str, int]
        Mapping of property name to value type code (VTYPE_* constants).
        String columns are included (VTYPE_STRING) so the caller knows
        which columns need CPU fallback.
    """
    n_features = d_feature_starts.shape[0]
    n_sample = min(sample_size, n_features)

    if n_sample == 0:
        return {}

    # Restrict scanning to the first n_sample features.
    # Create a byte mask covering the sample feature spans.
    d_sample_starts = d_feature_starts[:n_sample]
    d_sample_ends = d_feature_ends[:n_sample]
    # Find all bytes at property_depth that are opening quotes of keys.
    # Property keys in JSON are strings at the right depth.
    # Strategy: find all '"' bytes that are outside strings (qp==0),
    # at the correct depth, and preceded by '{', ',', or whitespace.
    # We detect key-colon patterns: look for '":' sequences at the
    # right depth within the sample region.

    # Simpler approach: scan the sample bytes on host.
    # The sample is tiny (first 100 features, maybe ~10KB), so this
    # D->H transfer is acceptable per the bead specification.
    sample_end_byte = int(d_sample_ends[n_sample - 1].item())
    sample_start_byte = int(d_sample_starts[0].item())

    # Transfer only the sample region to host
    h_bytes = cp.asnumpy(d_bytes[sample_start_byte:sample_end_byte])
    h_qp = cp.asnumpy(d_quote_parity[sample_start_byte:sample_end_byte])
    h_depth = cp.asnumpy(d_depth[sample_start_byte:sample_end_byte])

    # Walk the sample bytes to find property key names
    schema: dict[str, int] = {}
    local_len = len(h_bytes)
    i = 0
    while i < local_len:
        b = h_bytes[i]
        d = h_depth[i]
        qp = h_qp[i]

        # Look for opening quote of a key at the right depth, outside strings
        if b == ord('"') and qp == 1 and d == property_depth:
            # Find the closing quote
            j = i + 1
            while j < local_len and h_bytes[j] != ord('"'):
                j += 1
            if j >= local_len:
                break

            key_name = h_bytes[i + 1:j].tobytes().decode("ascii", errors="replace")

            # Find the colon after the closing quote
            k = j + 1
            while k < local_len and h_bytes[k] in (ord(' '), ord('\t'), ord('\n'), ord('\r')):
                k += 1

            if k < local_len and h_bytes[k] == ord(':'):
                # Classify the value after the colon
                v = k + 1
                while v < local_len and h_bytes[v] in (ord(' '), ord('\t'), ord('\n'), ord('\r')):
                    v += 1

                if v < local_len:
                    vc = h_bytes[v]
                    if vc == ord('"'):
                        vtype = VTYPE_STRING
                    elif vc == ord('t') or vc == ord('f'):
                        vtype = VTYPE_BOOLEAN
                    elif vc == ord('n'):
                        vtype = VTYPE_NULL
                    elif (vc >= ord('0') and vc <= ord('9')) or vc == ord('-'):
                        vtype = VTYPE_NUMBER
                    elif vc == ord('[') or vc == ord('{'):
                        vtype = VTYPE_COMPLEX
                    else:
                        vtype = -1

                    if vtype >= 0 and key_name not in schema:
                        schema[key_name] = vtype

            i = j + 1
        else:
            i += 1

    return schema


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------

def extract_gpu_properties(
    d_bytes: cp.ndarray,
    d_feature_starts: cp.ndarray,
    d_feature_ends: cp.ndarray,
    d_quote_parity: cp.ndarray,
    d_depth: cp.ndarray,
    *,
    property_depth: int = 5,
    sample_size: int = 100,
) -> dict[str, cp.ndarray]:
    """Extract numeric/boolean properties on GPU.

    Performs schema inference on a small sample (the only D->H transfer),
    then extracts each non-string property column entirely on GPU.

    Parameters
    ----------
    d_bytes : cp.ndarray
        Device-resident uint8 byte array of the full file.
    d_feature_starts : cp.ndarray
        Device-resident int64 feature start byte offsets.
    d_feature_ends : cp.ndarray
        Device-resident int64 feature end byte offsets.
    d_quote_parity : cp.ndarray
        Device-resident uint8 quote parity mask.
    d_depth : cp.ndarray
        Device-resident int32 bracket depth array.
    property_depth : int
        Bracket depth of property keys (default 5 for GeoJSON).
    sample_size : int
        Number of features to sample for schema inference.

    Returns
    -------
    dict[str, cp.ndarray]
        Mapping of property name to device array of values.

        - Numeric properties: float64 array of shape ``(n_features,)``
        - Boolean properties: uint8 array (1=true, 0=false) of shape ``(n_features,)``
        - Null-only properties: excluded from output (all-null columns are dropped)
        - String properties: NOT included (caller falls back to CPU)
        - Complex properties: NOT included (nested objects/arrays skipped)

        For columns with mixed types across features (e.g., some features
        have a numeric value and others have null), the output array has
        NaN for missing numeric values and 0 for missing boolean values.
    """
    n_features = d_feature_starts.shape[0]
    if n_features == 0:
        return {}

    # Step 1: Infer schema from sample (D->H transfer, acceptable)
    schema = infer_property_schema(
        d_bytes, d_quote_parity, d_depth,
        d_feature_starts, d_feature_ends,
        sample_size=sample_size,
        property_depth=property_depth,
    )

    if not schema:
        return {}

    result: dict[str, cp.ndarray] = {}

    for key_name, vtype in schema.items():
        if vtype == VTYPE_STRING or vtype == VTYPE_COMPLEX:
            # Skip string and complex columns -- caller handles via CPU
            continue

        if vtype == VTYPE_NULL:
            # All-null column in sample -- skip entirely
            continue

        # Find all positions of this key across ALL features (on GPU)
        d_key_positions = _find_property_keys(
            d_bytes, d_quote_parity, d_depth, key_name,
            property_depth=property_depth,
        )

        n_found = d_key_positions.shape[0]
        if n_found == 0:
            continue

        # Compute colon positions
        d_colon_pos = _colon_positions_from_key_hits(d_key_positions, key_name)

        # Classify value types for ALL found keys (not just sample)
        d_vtypes = _classify_values(d_bytes, d_colon_pos)

        if vtype == VTYPE_NUMBER:
            # Build a mask for features that actually have numeric values
            d_is_numeric = (d_vtypes == VTYPE_NUMBER)
            n_numeric = int(cp.sum(d_is_numeric).item())

            if n_numeric == 0:
                continue

            # Extract numeric positions (only where type is NUMBER)
            d_numeric_colon_pos = d_colon_pos[d_is_numeric]

            d_values = _extract_numeric_column(
                d_bytes, d_quote_parity, d_numeric_colon_pos,
                d_depth, n_numeric,
            )

            if n_found == n_features and n_numeric == n_features:
                # Every feature has this key with a numeric value
                result[key_name] = d_values
            else:
                # Some features have null or missing -- build full array with NaN
                d_full = cp.full(n_features, np.nan, dtype=cp.float64)
                if n_found == n_features:
                    # Key present in all features but some are null/other type
                    d_numeric_indices = cp.flatnonzero(d_is_numeric)
                    if d_values.shape[0] == d_numeric_indices.shape[0]:
                        d_full[d_numeric_indices] = d_values
                else:
                    # Key not present in all features -- the positions
                    # correspond to the subset that has the key.
                    # Map key positions back to feature indices using
                    # searchsorted on feature_starts.
                    d_feature_indices = cp.searchsorted(
                        d_feature_starts, d_key_positions, side="right",
                    ) - 1
                    d_numeric_global = d_feature_indices[d_is_numeric]
                    if d_values.shape[0] == d_numeric_global.shape[0]:
                        d_full[d_numeric_global] = d_values
                result[key_name] = d_full

        elif vtype == VTYPE_BOOLEAN:
            d_is_bool = (d_vtypes == VTYPE_BOOLEAN)
            n_bool = int(cp.sum(d_is_bool).item())

            if n_bool == 0:
                continue

            d_bool_colon_pos = d_colon_pos[d_is_bool]
            d_values = _extract_boolean_column(d_bytes, d_bool_colon_pos)

            if n_found == n_features and n_bool == n_features:
                result[key_name] = d_values
            else:
                # Mixed: some features have null or missing
                d_full = cp.zeros(n_features, dtype=cp.uint8)
                if n_found == n_features:
                    d_bool_indices = cp.flatnonzero(d_is_bool)
                    if d_values.shape[0] == d_bool_indices.shape[0]:
                        d_full[d_bool_indices] = d_values
                else:
                    d_feature_indices = cp.searchsorted(
                        d_feature_starts, d_key_positions, side="right",
                    ) - 1
                    d_bool_global = d_feature_indices[d_is_bool]
                    if d_values.shape[0] == d_bool_global.shape[0]:
                        d_full[d_bool_global] = d_values
                result[key_name] = d_full

    return result
