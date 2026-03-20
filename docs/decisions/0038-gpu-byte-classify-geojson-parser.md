---
id: ADR-0038
status: accepted
date: 2026-03-17
deciders:
  - claude-opus
  - vibeSpatial maintainers
tags:
  - io
  - gpu-primitives
  - geojson
  - kernel-strategy
  - performance
---

# ADR-0038: GPU Byte-Classification GeoJSON Parser

## Context

Reading large GeoJSON files is the dominant bottleneck in vibeSpatial's
end-to-end workflow.  The Florida.geojson benchmark (2.16 GB, 7.2M
polygons) takes 57.7s via pyogrio, which uses CPU-bound JSON parsing
internally.  For GPU-first spatial analytics, this I/O cost dwarfs all
subsequent GPU operations (reproject 0.4s, spatial query 0.5s).

A POC (`examples/poc_gpu_geojson.py`) demonstrated that GPU byte
classification + ASCII-to-fp64 parsing can extract coordinates in ~1.7s
(34x faster than pyogrio).  This ADR covers wiring that approach into
vibeSpatial's I/O architecture.

### Design constraints

1. **Geometry on GPU, properties on CPU** — vibeSpatial's GPU memory
   policy reserves device memory for geometry operations.  Property
   data (strings, mixed types) stays on the host.
2. **Homogeneous Polygon files** — v1 targets single-family polygon
   datasets (the dominant GIS use case).  Mixed geometry types are
   deferred to v2.
3. **No chunking** — files must fit in GPU memory (~3x file size peak).
   Chunked processing for files exceeding GPU memory is deferred to v2.

## Decision

### Hybrid GPU/CPU pipeline

The parser uses 10 NVRTC kernels (all Tier 1 per ADR-0033) for
geometry extraction, with CPU-side orjson for property extraction:

```
GPU pipeline (1.8s for 2.16 GB):
  S0   np.fromfile → cp.asarray                    [PCIe transfer]
  S1b  quote_toggle → uint8 cumsum → parity         [string awareness]
  S2   compute_depth_deltas → int32 cumsum → depth   [structural depth]
  S3   find_coord_key → flatnonzero → positions      [pattern match]
  S3b  coord_span_end                                [per-feature depth scan]
  S3c  count_rings_and_coords + scatter_ring_offsets  [GeoArrow offsets]
  S4   find_number_boundaries + mark_coord_spans      [coord-only numbers]
  S5   parse_ascii_floats → d_coords                  [ASCII → fp64]
  S6   x = d_coords[0::2], y = d_coords[1::2]        [zero-copy views]
  S7   _build_device_single_family_owned              [OwnedGeometryArray]
  S8   find_feature_boundaries → D→H copy             [for CPU properties]

CPU property extraction (9.2s, lazy):
  For each feature: slice host_bytes → orjson.loads → extract "properties"
```

### Memory optimization

The initial implementation hit OOM on a 24 GB GPU because `cp.cumsum`
of 2.16 billion int32 values requires 8.64 GB.  Two optimizations
resolved this:

1. **uint8 parity** — Quote state only needs even/odd (0/1), not the
   full cumsum value.  `cp.cumsum(toggle, dtype=cp.uint8) & 1` uses
   2.16 GB instead of 8.64 GB.  Parity is correct after uint8 overflow
   because 256 is even.

2. **Fused depth delta kernel** — A single `compute_depth_deltas` kernel
   takes raw bytes + quote parity and outputs int8 deltas directly,
   avoiding materialization of `d_classes` (2.16 GB), `outside_string`
   (2.16 GB), and boolean intermediates from `cp.where`.

Peak GPU memory is ~3x file size (~6.5 GB for 2.16 GB file).

### Integration points

- **`io_geojson.py`** — New `"gpu-byte-classify"` strategy in
  `plan_geojson_ingest()`, routed in `read_geojson_owned()`.
- **`io_file.py`** — `_try_gpu_read_file()` auto-routes GeoJSON files
  >10 MB to the GPU path before falling back to pyogrio.
- **NVRTC warmup** — All 10 kernels registered via
  `request_nvrtc_warmup()` per ADR-0034.  First-run compilation adds
  ~12s; subsequent runs are cached.

### Depth semantics

CuPy's `cumsum` produces an inclusive prefix sum, so depth values at
brackets include the bracket's own delta:

- At opening `[`: depth = parent_depth + 1
- At closing `]`: depth = parent_depth (was +1, now -1 applied)

This means ring-closing `]` has `depth == coord_depth` (not
`coord_depth + 1`) and pair-closing `]` has `depth == coord_depth + 1`.
Feature boundaries are at depth 3 (open) / depth 2 (close), not 2/1,
because FeatureCollection adds an extra nesting level.

## Consequences

### Measured performance (RTX 4090, i9-13900k)

| Step | GeoPandas | vibeSpatial | Speedup |
|---|---|---|---|
| Read GeoJSON | 57.7s | 11.7s | **4.9x** |
| Reproject to UTM | 8.2s | 0.4s | 21x |
| Select within 1km | 0.2s | 0.5s | — |
| Write GeoParquet | 0.2s | 0.1s | 2x |
| **End-to-end** | **66.3s** | **12.7s** | **5.2x** |

GPU geometry parse: **1.8s** (32x vs pyogrio).
CPU property extraction: **9.2s** (lazy, only when accessed).

### Property extraction is the remaining bottleneck

The 9.2s CPU property loop is 7.2M calls to `orjson.loads()` on ~35-byte
property objects.  The time is dominated by Python interpreter overhead
(function call dispatch, dict construction, list.append), not JSON
parsing throughput.

**Strategies evaluated and rejected:**

| Strategy | Time | Why it doesn't help |
|---|---|---|
| ThreadPool orjson (8w) | 10.0s | GIL held during Python object construction |
| Multiprocessing (8w) | 12.8s | 2.16 GB raw bytes copied to each worker via IPC |
| Bulk orjson (single parse) | 17.9s | 1 GB FeatureCollection → 7.2M Python dicts = catastrophic |
| pylibcudf batched | 8.6s | CPU loop to extract+re-serialize property substrings dominates |
| Strip coords + bulk parse | >14 min | Coordinates are only 54% of file; 1 GB remainder still too large |

**Key finding:** For this dataset, coordinate bytes are 54% of the file
(not ~90% as initially estimated).  Structural JSON overhead (`"type":`,
`"Feature"`, `"geometry":`, braces, commas) accounts for 34%.  Actual
property data is only ~12% of file bytes.

### Deferred: native property parser

A Rust/C extension that walks feature bytes and extracts property values
directly into columnar arrays (bypassing Python dict construction)
could plausibly achieve ~1s for property extraction.  This is deferred
because:

1. **Pure Python policy** — vibeSpatial's current codebase is pure
   Python + CUDA kernels (via NVRTC strings).  Adding a compiled
   Rust/C extension changes the build story and CI matrix.
2. **Diminishing returns** — The 9.2s property cost only matters when
   properties are accessed.  Geometry-only workflows (the primary GPU
   target) see only 1.8s read time.
3. **Lazy evaluation** — Properties are loaded via a closure; the cost
   is deferred until `batch.properties` or `GeoDataFrame` construction.
   Workflows that filter first (spatial query → subset → access
   properties) parse far fewer features.

If property extraction becomes a measured bottleneck in production
workflows, the recommended path is a purpose-built columnar JSON
property extractor — either as a Rust pyo3 extension or as additional
NVRTC kernels that output property strings into a device buffer for
`pylibcudf.io.json.read_json_from_string_column`.

## Alternatives Considered

### pylibcudf `get_json_object` on full file

Using `plc.json.get_json_object(file_column, "$.features[*].properties")`
to extract all properties on GPU.  Rejected because it requires the
entire 2.16 GB file as a single GPU string column, and the JSONPath
evaluation on 7.2M features causes OOM even on a 24 GB card.

### cuDF `read_json` on the full file

cuDF's JSON reader could parse the entire FeatureCollection on GPU,
producing a columnar table with both geometry and property columns.
Rejected because: (a) it would parse coordinates redundantly (we
already extract them faster with our kernels), (b) GeoJSON's nested
coordinate arrays don't map cleanly to cuDF's flat column model, and
(c) cuDF is a much heavier dependency than pylibcudf.

### Host-only fast path (simdjson / orjson bulk)

Keep everything on CPU but use SIMD-accelerated JSON parsing.  Already
benchmarked: simdjson and orjson bulk parse are slower than the
per-feature loop for this workload shape (many small features).

## References

- POC: `examples/poc_gpu_geojson.py`
- Property extraction POC: `examples/poc_property_extraction.py`
- Geometry stripping POC: `examples/poc_strip_geometry.py`
- Implementation: `src/vibespatial/io/geojson_gpu.py`
- Tests: `tests/test_geojson_gpu.py`
- ADR-0002: Precision policy (fp64 for I/O)
- ADR-0033: Kernel tier classification
- ADR-0034: NVRTC warmup
