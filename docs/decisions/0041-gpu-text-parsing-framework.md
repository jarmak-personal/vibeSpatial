---
id: ADR-0041
status: accepted
date: 2026-03-26
deciders:
  - claude-opus
  - vibeSpatial maintainers
tags:
  - io
  - gpu-primitives
  - framework
  - kernel-strategy
  - parsing
---

# ADR-0041: GPU Text Parsing Framework

## Context

ADR-0038 introduced GPU byte-classification parsing for GeoJSON, achieving
a 32x speedup over pyogrio for coordinate extraction.  That implementation
proved several key techniques: per-byte quote-parity via uint8 cumsum,
bracket-depth via int8 delta kernel + int32 prefix sum, pattern-matched
structural marker detection, span-based region masking, and per-token
ASCII-to-float64 parsing.

As vibeSpatial adds GPU readers for additional text formats (WKT, CSV,
KML, GML), the risk of kernel duplication becomes significant.  The
GeoJSON reader alone uses 12 NVRTC kernels.  A naive WKT reader would
duplicate the quote-state logic, number boundary detection, and float
parsing kernels with minor variations.  A CSV reader would duplicate the
quote-toggle and number-extraction logic but need different row/column
boundary semantics.

The need for a composable framework is clear: formats differ in their
structural delimiters, nesting semantics, and content layout, but the
underlying byte-classification and numeric extraction operations are
fundamentally the same.  A shared primitive library enables new format
parsers to be built by composition rather than by copying and modifying
existing kernel code.

### Design constraints

1. **Pure Python shipping** -- vibeSpatial ships NVRTC source strings,
   not compiled `.cu` files.  The framework must generate kernel source
   at runtime and compile via the existing `compile_kernel_group()` /
   NVRTC infrastructure.
2. **Format-agnostic primitives** -- Each primitive must work across
   formats without hard-coding format-specific constants (e.g., bracket
   characters, pattern bytes, quote-escape conventions).
3. **Device-resident throughout** -- All intermediate and output arrays
   remain on GPU.  The only acceptable D-to-H transfer is for small
   metadata (e.g., CSV header row, unique tag counts for
   homogeneous/mixed branching).
4. **Composable, not monolithic** -- Format parsers compose primitives
   in format-specific sequences.  The framework does not impose a fixed
   pipeline order.

## Decision

### Package structure

The framework lives in `src/vibespatial/io/gpu_parse/` with three
modules, each containing a focused set of primitives:

```
vibespatial/io/gpu_parse/
    __init__.py        # Re-exports all public primitives
    structural.py      # Quote-state and bracket-depth computation
    numeric.py         # Number boundary detection and ASCII-to-number parsing
    pattern.py         # Byte-pattern matching and span detection
```

### Primitive taxonomy

| Module | Primitive | Underlying Kernel(s) | Input | Output | Memory |
|--------|-----------|---------------------|-------|--------|--------|
| structural | `quote_parity` | `quote_toggle` | uint8 bytes | uint8 parity (0/1) | uint8 cumsum (4x savings vs int32) |
| structural | `bracket_depth` | `compute_depth_deltas` (generated per char set) | uint8 bytes + uint8 parity | int32 depth | int8 deltas before int32 cumsum |
| numeric | `number_boundaries` | `find_number_boundaries` | uint8 bytes + uint8 parity | uint8 is_start, uint8 is_end | Per-byte masks |
| numeric | `extract_number_positions` | (CuPy flatnonzero) | uint8 masks + optional uint8 region mask | int64 starts, int64 ends | Compact position arrays |
| numeric | `parse_ascii_floats` | `parse_ascii_floats` | uint8 bytes + int64 starts/ends | float64 values | One thread per token |
| numeric | `parse_ascii_ints` | `parse_ascii_ints` | uint8 bytes + int64 starts/ends | int64 values | One thread per token |
| pattern | `pattern_match` | `pattern_match_kernel` (generated per pattern) | uint8 bytes + optional uint8 parity | uint8 hits (0/1) | Per-byte mask |
| pattern | `span_boundaries` | `span_boundaries_kernel` | int32 depth + int64 starts | int64 ends | One thread per span |
| pattern | `mark_spans` | `mark_spans_kernel` | int64 starts + int64 ends | uint8 mask | Per-byte mask, one thread per span |

All primitives accept and return CuPy device arrays.  No primitive
performs a D-to-H transfer.

### Composition pattern

A GPU text parser composes these primitives in a pipeline that flows
from raw bytes to extracted numeric values.  The canonical stages are:

```
Stage 1: Structural analysis
    d_bytes             -> quote_parity()     -> d_quote_parity
    d_bytes + d_qp      -> bracket_depth()    -> d_depth

Stage 2: Marker detection
    d_bytes + d_qp      -> pattern_match()    -> d_hits
    d_hits               -> cp.flatnonzero()   -> d_positions

Stage 3: Span definition
    d_depth + d_positions -> span_boundaries() -> d_span_ends
    d_positions + d_ends  -> mark_spans()      -> d_region_mask

Stage 4: Number extraction
    d_bytes + d_qp       -> number_boundaries() -> d_is_start, d_is_end
    d_is_start/end + mask -> extract_number_positions() -> d_starts, d_ends
    d_bytes + d_starts/ends -> parse_ascii_floats() -> d_values

Stage 5: Coordinate split (zero-copy)
    d_values[0::2] -> d_x
    d_values[1::2] -> d_y

Stage 6: Format-specific assembly
    d_x, d_y, offsets -> OwnedGeometryArray
```

Each format composes these stages differently:

**GeoJSON** (`geojson_gpu.py`):
```
quote_parity -> bracket_depth({[, }])
-> pattern_match("coordinates":) -> span_boundaries(skip=14) -> mark_spans
-> number_boundaries -> extract_number_positions(mask) -> parse_ascii_floats
-> x/y split -> family-aware assembly (homogeneous or mixed)
```
GeoJSON also uses format-specific kernels for type detection
(`find_type_key`, `classify_type_value`), ring/part counting
(`count_rings_and_coords`, `count_mpoly_levels`), feature boundary
detection (`find_feature_boundaries`), and offset scattering
(`scatter_ring_offsets`, `scatter_mpoly_offsets`).  These remain in
`geojson_gpu.py` because they encode GeoJSON-specific structural
semantics (depth levels, key patterns, nesting conventions).

**WKT** (`wkt_gpu.py`):
```
bracket_depth(open="(", close=")")    # no quote_parity needed
-> line splitting (CuPy newline detection)
-> type classification (custom NVRTC prefix matching)
-> paren-start detection -> span-local counting
-> number_boundaries (WKT-specific: space as separator, no quotes)
-> parse_ascii_floats -> x/y split
-> per-family offset building -> assembly
```
WKT skips `quote_parity` entirely because WKT has no string quoting.
It passes an all-zeros parity array to `bracket_depth`.  WKT uses
parentheses `()` instead of brackets `{[}]` for nesting, which is
handled by the parameterizable `bracket_depth(open_chars="(",
close_chars=")")`.

**CSV** (`csv_gpu.py`):
```
csv_quote_toggle (format-specific: no backslash escaping)
-> uint8 cumsum parity
-> row boundary detection (newline + parity filter)
-> delimiter detection (configurable char + parity filter)
-> column count verification (CuPy bincount)
-> header parsing (small D->H for column names only)
```
CSV required a custom `csv_quote_toggle` kernel because CSV uses
doubled-quote escaping (`""`) rather than backslash escaping (`\"`).
The doubled-quote convention naturally cancels in the cumulative-sum
parity computation without special handling, making the CSV toggle
kernel simpler than the JSON version.  CSV does not use
`bracket_depth`, `pattern_match`, or `span_boundaries` because its
structure is flat (row/column grid) rather than hierarchically nested.

**KML/GML** (future):
```
quote_parity -> bracket_depth(open="<", close=">")
-> pattern_match(<coordinates>) -> span_boundaries -> mark_spans
-> number_boundaries -> extract_number_positions(mask) -> parse_ascii_floats
-> x/y(/z) split -> assembly
```
XML-based formats would use angle brackets for depth and tag-based
patterns for marker detection.  The existing primitives support this
without modification.

### Memory model

The framework's memory optimizations apply to all format parsers
that compose these primitives:

**uint8 parity trick.**  Quote state is a single-bit value (inside
or outside a string).  Rather than accumulating toggle counts in
int32 (4 bytes per position), the framework uses `cp.cumsum(toggle,
dtype=cp.uint8) & 1`.  The uint8 cumsum overflows at 256, but
because 256 is even, the low bit (parity) is preserved correctly.
For a 2 GB input file, this saves 6.48 GB of device memory (uint8
at 2 GB vs int32 at 8.64 GB).

**int8 deltas before int32 cumsum.**  The `bracket_depth` kernel
emits int8 deltas (+1, -1, or 0) rather than computing depth
directly.  The int8 delta array uses 1 byte per position.  Only the
final `cp.cumsum(deltas, dtype=cp.int32)` materializes the 4-byte
depth array.  This avoids an intermediate boolean mask and `cp.where`
that would each require additional per-byte allocations.

**Zero-copy strided views for x/y split.**  After `parse_ascii_floats`
produces a flat float64 array of interleaved coordinates, the x and
y arrays are extracted as strided views: `d_x = d_coords[0::2]`,
`d_y = d_coords[1::2]`.  These are CuPy views backed by the same
device buffer -- no allocation or copy occurs.  A contiguous copy is
made only when required by downstream kernels that need contiguous
input.

**Device-resident output.**  The final output is an
`OwnedGeometryArray` with coordinates, offsets, and validity masks
on device.  No bulk D-to-H transfer occurs during parsing.  The only
host materializations are small metadata reads (unique tag counts for
homogeneous/mixed branching, total coordinate counts for offset array
sizing).

### Kernel generation and caching

Two primitives generate NVRTC source at runtime:

**`pattern_match`** generates a kernel per byte pattern.  The pattern
bytes are embedded as a compile-time constant array in the generated
CUDA source, enabling the compiler to optimize the comparison loop
(unrolling, constant propagation).  The kernel source also
conditionally includes a quote-parity check parameter based on
whether `d_quote_parity` is provided.

```python
# Cache key: (pattern_bytes, check_quote, quote_check_offset)
# NVRTC cache key: SHA1 hash of generated source via make_kernel_cache_key()
_pattern_kernel_cache: dict[bytes, dict] = {}
```

**`bracket_depth`** generates a depth-delta kernel per open/close
character set.  The open and close characters are baked into the
NVRTC source as equality checks:

```python
# For JSON: open_chars="{[", close_chars="}]"
# Generated: if (b == '{' || b == '[') d = 1;
#            else if (b == '}' || b == ']') d = -1;

# Cache key: (open_chars, close_chars) tuple
_depth_kernel_cache: dict[tuple[str, str], dict[str, object]] = {}
```

Both caches are module-level dictionaries.  The underlying NVRTC
compilation is also cached by SHA1 hash of the source string via
`compile_kernel_group()`, so even if the Python-level cache misses,
the NVRTC disk cache may hit.

Static kernels (`quote_toggle`, `find_number_boundaries`,
`parse_ascii_floats`, `parse_ascii_ints`, `span_boundaries_kernel`,
`mark_spans_kernel`) are compiled once via `compile_kernel_group()`
and cached by their fixed prefix + SHA1 hash.

### NVRTC warmup registration

All static kernels and the most common parameterized kernel variants
are registered for background precompilation at module import time
via `request_nvrtc_warmup()` (ADR-0034 Level 2):

| Module | Warmup entries |
|--------|---------------|
| structural | `quote_toggle`, `compute_depth_deltas` for `{[`/`}]` |
| numeric | `find_number_boundaries`, `parse_ascii_floats`, `parse_ascii_ints` |
| pattern | `span_boundaries_kernel`, `mark_spans_kernel` |

Pattern-match kernels cannot be warmed up at module scope because
they are generated per pattern.  Format-specific modules (e.g.,
`geojson_gpu.py`) register their own pattern warmups.

### Tier classification (ADR-0033)

All framework kernels are Tier 1 (custom NVRTC) or Tier 2 (CuPy
built-in):

| Kernel | Tier | Rationale |
|--------|------|-----------|
| `quote_toggle` | Tier 1 | Custom byte classification with backslash escape handling |
| `compute_depth_deltas` | Tier 1 | Generated per-char-set byte classification |
| `find_number_boundaries` | Tier 1 | Custom byte classification with separator detection |
| `parse_ascii_floats` | Tier 1 | Custom per-token state machine |
| `parse_ascii_ints` | Tier 1 | Custom per-token accumulator |
| `pattern_match_kernel` | Tier 1 | Generated per-pattern byte comparison |
| `span_boundaries_kernel` | Tier 1 | Custom per-span depth scan |
| `mark_spans_kernel` | Tier 1 | Custom per-span region fill |
| `cp.cumsum` | Tier 2 | CuPy built-in prefix sum |
| `cp.flatnonzero` | Tier 2 | CuPy built-in compaction |

No Tier 3 (CCCL) primitives are used in the framework itself,
although format-specific modules (e.g., `wkt_gpu.py`) may use CCCL
`exclusive_sum` for offset construction.

### Precision (ADR-0002)

All structural and counting kernels are integer-only byte
classification.  No floating-point coordinate computation occurs in
those kernels, so no PrecisionPlan is needed.

`parse_ascii_floats` always produces fp64 output.  Storage precision
is always fp64 per ADR-0002.  The kernel's character-by-character
state machine supports sign, integer part, fractional part, and
scientific notation exponent, matching the JSON/CSV/WKT numeric
literal specifications.

`parse_ascii_ints` always produces int64 output for integer fields
(feature IDs, SRID values, integer attributes).

### Guidelines for adding a new format parser

A new format parser should follow this process:

**Step 1: Identify structural delimiters.**  Does the format use
quoted strings?  What characters define nesting?  Examples:

| Format | Quotes | Nesting chars | Notes |
|--------|--------|---------------|-------|
| JSON | `"` with `\` escape | `{` `[` / `}` `]` | Backslash-escape aware |
| WKT | None | `(` / `)` | No quoting |
| CSV | `"` with `""` escape | None (flat) | Doubled-quote escaping |
| KML | `"` with `\` escape | `<` / `>` | XML conventions |

**Step 2: Compose structural primitives.**  Call `quote_parity()` if
the format uses quoted strings (skip for WKT).  Call
`bracket_depth()` with the format's nesting characters if the format
has hierarchical structure (skip for CSV).  If the format has a
non-standard quoting convention (CSV doubled-quote), write a
format-specific quote-toggle kernel and use the same uint8 cumsum
parity technique.

**Step 3: Locate content regions.**  Use `pattern_match()` to find
structural markers (JSON keys, XML tags, WKT keywords).  Use
`span_boundaries()` to find the extent of each marker's governed
region.  Use `mark_spans()` to create a per-byte region mask.  For
flat formats (CSV), use format-specific row/column boundary detection
instead.

**Step 4: Extract numbers.**  Call `number_boundaries()` to classify
start/end positions of numeric tokens.  Call
`extract_number_positions()` with the region mask from Step 3 to
filter to relevant numbers only.  Call `parse_ascii_floats()` or
`parse_ascii_ints()` to convert tokens to numeric values.  If the
format uses a different numeric separator convention (e.g., WKT uses
space between coordinates), write a format-specific boundary kernel
or extend the existing one.

**Step 5: Build offsets and assemble.**  Compute per-geometry offset
arrays (coordinate offsets, ring offsets, part offsets) from the
extracted structural metadata.  Use CuPy `cumsum` or CCCL
`exclusive_sum` for prefix-sum offset construction.  Assemble the
final `OwnedGeometryArray` using the existing
`_build_device_single_family_owned()` or `_build_device_mixed_owned()`
helpers.

**Step 6: Register NVRTC warmup.**  Register all format-specific
NVRTC kernels via `request_nvrtc_warmup()` at module scope.  The
framework's shared kernels are already registered by their respective
modules; only format-specific kernels need additional registration.

### What we chose NOT to do

**Full GPU JSON DOM construction.**  A general-purpose JSON parser on
GPU would build a DOM tree (parent pointers, sibling links, value
type tags) for every node in the document.  For a 2 GB GeoJSON file
with 7.2M features, this would require tens of gigabytes of device
memory for structural metadata that is discarded immediately after
coordinate extraction.  The targeted byte-classification approach
extracts only the data needed (coordinates, types, boundaries) at a
fraction of the memory cost.

**GPU string processing for properties.**  GeoJSON property values
are mixed-type strings, integers, and nested objects.  Processing
these on GPU would require variable-length string handling, type
dispatch, and dictionary construction -- operations where the CPU
(via orjson) is faster and more flexible.  The hybrid design keeps
geometry on GPU and properties on CPU (ADR-0038).

**simdjson-style vectorized classification.**  simdjson uses x86 SIMD
intrinsics (AVX2, SSE4.2) for 64-byte-at-a-time structural
classification with lookup tables.  Adapting this to GPU would
require warp-level shuffle-based lookup tables and would add
complexity without clear benefit: the GPU's massive thread
parallelism already saturates memory bandwidth with the simpler
per-byte kernel approach.  The per-byte kernels achieve memory
bandwidth saturation on current hardware (verified at ~900 GB/s on
RTX 4090).

**Unified quote-toggle kernel for all formats.**  CSV's
doubled-quote escaping (`""`) has fundamentally different semantics
from JSON's backslash escaping (`\"`).  Attempting to unify these
into a single parameterized kernel would add branching complexity
without benefit, since each format's toggle logic is a single simple
kernel.  The framework accepts that quote-toggle is the one primitive
that may need format-specific variants.

## Consequences

### Positive

- **New formats by composition.** The WKT GPU reader was built by
  composing `bracket_depth(open="(", close=")")` with the numeric
  primitives, reusing 100% of the number extraction pipeline.  Only
  format-specific kernels (type classification, line splitting,
  paren-start detection, ring/part counting) needed to be written.

- **Shared memory optimizations.** The uint8 parity trick, int8
  delta optimization, and zero-copy strided views benefit every format
  parser that uses the framework.  A new format parser gets these
  optimizations for free without re-implementing them.

- **Pure Python shipping.** All kernel source is embedded as Python
  strings in the framework modules.  No `.cu` files, no nvcc build
  step, no platform-specific compiled artifacts.  NVRTC compilation
  happens at runtime and is cached by SHA1 hash on disk.

- **Parameterizable without code duplication.** The
  `bracket_depth(open_chars, close_chars)` and `pattern_match(pattern)`
  primitives generate format-specific NVRTC source at runtime.  Adding
  support for XML angle brackets or WKT parentheses requires zero new
  kernel source files -- just different parameter values.

- **Consistent kernel lifecycle.** All framework kernels follow the
  same compilation, caching, warmup, and launch patterns established
  by `compile_kernel_group()` and `request_nvrtc_warmup()`.  Format
  authors do not need to learn a new kernel management API.

### Negative

- **First-use compilation latency for new patterns.** The
  `pattern_match` and `bracket_depth` primitives compile NVRTC kernels
  on first use of each unique pattern or character set.  For patterns
  not covered by warmup registration, this adds ~0.5-2s of latency on
  first invocation.  Subsequent invocations hit the SHA1-based disk
  cache.  Mitigation: format-specific modules should register their
  patterns via `request_nvrtc_warmup()` at import time.

- **CSV needed a format-specific quote-toggle kernel.** The
  `quote_parity()` primitive in `structural.py` implements
  backslash-escape-aware toggling for JSON.  CSV uses doubled-quote
  escaping, which has simpler semantics (every `"` toggles; doubled
  quotes cancel naturally in the cumsum).  The CSV reader
  (`csv_gpu.py`) implements its own `csv_quote_toggle` kernel rather
  than using the framework's `quote_parity()`.  This is an accepted
  divergence: the quoting convention is the one dimension where
  formats genuinely differ at the kernel level.

- **Module-level kernel caches lack thread-safety locks.**  The
  `_pattern_kernel_cache` and `_depth_kernel_cache` are plain
  dictionaries without `threading.Lock` protection.  Under
  free-threaded Python (PEP 703), concurrent first-use compilation of
  the same pattern could result in redundant compilation (but not
  incorrect results, because `compile_kernel_group` is idempotent).
  If free-threaded builds become a target, these caches should be
  guarded by locks.

- **Number boundary heuristic is JSON/CSV-centric.**  The
  `find_number_boundaries` kernel uses separator characters (`,`,
  `[`, `]`, space, newline) that match JSON and CSV conventions.  WKT
  uses space as a coordinate separator and comma as a point separator,
  which partially overlaps but may require a format-specific variant
  for edge cases.  The WKT reader addresses this with its own
  boundary kernel (`wkt_find_number_boundaries`) that uses
  WKT-appropriate separators.

## References

- ADR-0038: GPU byte-classification GeoJSON parser (the approach this
  framework generalizes)
- ADR-0002: Dual-precision dispatch (fp64 storage for I/O)
- ADR-0033: GPU primitive dispatch rules (tier classification)
- ADR-0034: CCCL/NVRTC precompile warmup strategy
- Framework: `src/vibespatial/io/gpu_parse/` (structural, numeric,
  pattern modules)
- GeoJSON consumer: `src/vibespatial/io/geojson_gpu.py`
- WKT consumer: `src/vibespatial/io/wkt_gpu.py`
- CSV consumer: `src/vibespatial/io/csv_gpu.py`
