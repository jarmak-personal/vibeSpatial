# Native Big-Endian WKB And GeoParquet Implementation Plan

Status: implemented and verified on SF1, SF10, and SF100; landing workflow follows this report

Source investigation: `reports/WKB-geoparquet.md`

Mandate: big-endian 2D OGC WKB GPU acceleration is required for completion.
It is not an optional follow-up and must work through the reusable WKB decoder,
not a SpatialBench-specific conversion or query branch.

## Implementation And Verification Result

The mandate is complete on the authorized RTX 4090 workstation. The shared
Arrow/GeoParquet decoder now performs one byte-authoritative structural scan,
uses stable per-row status codes, decodes all six canonical 2D families in
little, big, or independently mixed embedded byte order, builds exact
device-resident owned buffers, and retains only one fixed 376-byte aggregate
telemetry packet on successful native ingress. Dimensional WKB rejects the 2D
owned carrier before compatibility merge so Z/M ordinates cannot be silently
discarded.

The metadata-only transcode accepts `binary`, `large_binary`, and `binary_view`,
publishes standard WKB `binary`, carries supported decimal/timestamp metadata,
validates bytes/values/schema/footer before atomic publication, and releases
source tables before later large allocations. GeoParquet family metadata is
never treated as proof of byte layout.

The default RMM stack now uses `CudaAsyncMemoryResource` with a release
threshold and allocation limiter. This resolved repeat-run fragmentation at
SF100: Q11 repeats remain stable with about 80 MB live after the query instead
of retaining a fragmented roughly 19.9 GB legacy pool reservation.

SF1000 was deliberately not inspected, prepared, or executed on this machine.

### Dataset inventory

| Scale | WKB rows | Payload bytes | Little endian | Big endian | Coordinates |
|---:|---:|---:|---:|---:|---:|
| SF1 | 12,176,095 | 1,652,659,182 | 12,020,000 | 156,095 | 99,326,114 |
| SF10 | 120,541,148 | 4,557,163,400 | 120,086,438 | 454,710 | 246,797,486 |
| SF100 | 1,201,186,386 | 30,671,576,381 | 1,200,152,877 | 1,033,509 | 1,540,760,192 |

The available building and trip shards are little endian; zones are big endian
and contain Polygon/MultiPolygon rows. No executed scale contained a declined
or mixed-embedded production row. Synthetic tests cover every mixed embedded
combination.

### SpatialBench correctness and performance

Every WKB Q1-Q12 output matched the current GeoArrow result at SF1, SF10, and
SF100 (`rtol=atol=1e-12`). SF100 additionally matched the frozen independent
GeoPandas reference (`rtol=1e-6`, `atol=1e-9`). Repeat-3 medians after one warm
run were:

| Scale | WKB total | GeoArrow total | WKB Q11 | Result hash |
|---:|---:|---:|---:|---|
| SF1 | 5.86s | 6.75s | 0.86s | `b0ca8257854173e285e7d355e77027df766ded8ace6e3f68423856350e4fb948` |
| SF10 | 43.24s | 46.87s | 5.34s | `39daf447ae09c29059bb0dc0f468f6b3a2ba149365f6daecc4e14d90a65229cd` |
| SF100 | 258.78s | 254.90s | 78.12s | `c7e20edae9b6a7978a28c7722fa7f2db7fb921d101a2f8f7ecf251494aa57d99` |

The SF100 WKB total is 1.5% above GeoArrow and stays within the 5% control
rail. Artifacts are under
`benchmark_results/spatialbench/wkb-big-endian-2026-09-01/`, including the
`sf{1,10,100}-wkb-final-rmm-async-repeat3.json` and matching GeoArrow files.

### Codec performance and safety

The 100K synchronized codec benchmark (`io-arrow-smoke-final.json`) passed all
15 family/endian cases with zero fallback and zero payload/coordinate D2H.

| Family | LE host speedup | BE host speedup | BE/LE throughput |
|---|---:|---:|---:|
| Point | 11.98x | 11.08x | 1.051 |
| LineString | 18.35x | 18.96x | 1.033 |
| Polygon | 15.62x | 15.87x | 1.009 |
| MultiPoint | 19.32x | 21.89x | 1.062 |
| MultiLineString | 27.13x | 25.92x | 0.939 |
| MultiPolygon | 30.34x | 30.28x | 1.023 |

All cases exceed the 4x host floor and all big-endian variants exceed the 0.8
relative-throughput floor. Kernel inspection on the RTX 4090 found 20-40
registers, 256 maximum threads, 376 bytes shared memory only in the summary
kernel, and no local-memory spills. CUDA memcheck passed 35 malformed and
mixed-endian cases with `ERROR SUMMARY: 0 errors`.

### Tests and repository health

- Focused WKB endian suite: 51 passed.
- WKB/GeoParquet I/O selection: 157 passed, 128 deselected.
- GPU dispatch: 110 passed.
- Codec, memory-pool, admission, and benchmark rails: 92 passed.
- Full repository run before the final focused fixes: 7,960 passed, 430
  skipped, 7 xfailed. Eight change-surface failures (architecture, benchmark
  routing, generated docs, and five bounded-telemetry expectations) were fixed
  and passed focused reruns. The remaining 11 failures are unrelated vendored
  Feather/vibeproj compatibility cases.
- Ruff, architecture lint, documentation checks, import guard, property
  dashboard, and the focused regression reruns pass after the final edits.

### Mandatory 1M pipeline profile

`uv run python scripts/benchmark_pipelines.py --suite full --repeat 1
--gpu-sparkline` passed. The repeat used for the concise timing ledger reported:

| Pipeline | 1M stages (wall time) |
|---|---|
| join-heavy | read_points 3.33ms; read_polygons 5.75ms; build_index 0.26ms; sjoin_query 0.63ms; assemble_join_rows 0.42ms; dissolve_groups 2.22ms; write_output 16.13ms |
| relation-semijoin | read_inputs 11.75ms; build_index 0.20ms; sjoin_relation 0.54ms; semijoin_rowset 0.37ms; subset_rows 0.86ms; write_output 3.33ms |
| small-grouped-constructive-reduce | build_device_grouped_polygons 37.84ms; native_grouped_union 46.67ms; reference_check 0.03ms |
| grouped-capacity-partitions | build fixtures 47.09ms; mixed_strip_exact_union 62.99ms; positive_degenerate_union 58.41ms; reference_check 0.02ms |
| grouped-disjoint-constructive-reduce | build groups 61.89ms; grouped subset 1.20ms; reference_check 0.02ms |
| grouped-difference-constructive | build inputs 24.56ms; grouped difference 9.14ms; reference_check 0.03ms |
| constructive-output-native | build boxes 3.52ms; intersection 3.45ms; area expression 1.08ms; expression consumers 1.51ms; reference_check 0.03ms |
| overlay-relation-constructive | build inputs 6.24ms; index 0.10ms; candidates 0.68ms; refine 0.22ms; intersection 2.62ms; projection 0.94ms; reference_check 0.03ms |
| constructive | read_points 3.65ms; clip_points 0.48ms; buffer_points 1.44ms; write_output 16.00ms |
| predicate-heavy | read_geojson 69.82ms; load_polygons 6.79ms; point_in_polygon 0.36ms; filter_points 0.35ms; write_output 1.46ms |
| zero-transfer | read_input 6.54ms; predicate_filter 0.41ms; subset_rows 1.02ms; write_output 2.95ms |

No stage exceeded 70ms and no mask, partition, candidate-filter, or subsetting
stage approached the 1s investigation threshold.

## Outcome

Deliver one byte-derived, GPU-native WKB ingress pipeline that:

- safely admits canonical 2D little- and big-endian WKB;
- honors the independent byte-order flag of every embedded record in multi-
  geometries;
- decodes Point, LineString, Polygon, MultiPoint, MultiLineString, and
  MultiPolygon directly into device-resident owned geometry buffers;
- keeps null, empty, family, row order, and invalid-input semantics exact;
- classifies EWKB, Z/M/ZM, GeometryCollection, malformed, truncated, and
  otherwise unsupported rows before decode;
- never treats GeoParquet `geometry_types` as proof of byte order or binary
  layout;
- keeps supported big-endian input native under strict-native mode;
- makes every unsupported decline observable and rejects it under strict-native
  mode; and
- removes the current big-endian compatibility cost from WKB GeoParquet reads.

The work also completes the adjacent metadata-only transcode contract:

- accept Arrow `binary`, `large_binary`, and `binary_view` WKB input;
- write standards-compatible Parquet BYTE_ARRAY / Arrow `binary` geometry
  fields without changing WKB payload bytes;
- preserve non-geometry values and logical schemas exactly, or fail explicitly
  before writing when the selected writer cannot represent the source logical
  type; and
- validate source/output row counts, schemas, metadata, and geometry byte
  identity mechanically.

## Scope And Non-Goals

In scope:

- OGC canonical 2D WKB type ids 1 through 6;
- root little endian, root big endian, and independently mixed endian embedded
  Point, LineString, and Polygon records;
- homogeneous and mixed-family WKB columns;
- PyArrow binary carriers and pylibcudf string/binary physical views;
- public Arrow WKB ingress and pylibcudf GeoParquet ingress;
- device-resident `OwnedGeometryArray` and seeded `NativeGeometryMetadata`
  output;
- explicit compatibility handling for rows outside the native 2D contract;
- metadata-only legacy-WKB Parquet to GeoParquet transcode; and
- SF-scale correctness and end-to-end profiling.

Out of scope unless a separate accepted contract expands them:

- native GeometryCollection assembly;
- EWKB SRID flags and PostGIS dimensional flag encodings;
- OGC Z, M, or ZM coordinates;
- non-linear and extended geometry families;
- changing GeoArrow from the canonical prepared interchange format; and
- benchmark-name, scale-factor, file-name, or dataset-specific branches.

Endian distribution is an input property, not a planning assumption. Inventory
the actual bytes at every tested scale. A table or shard being little endian at
SF1 and big endian at SF1000 must require no code or metadata change.

## Design Decisions

### 1. Admission follows bytes

GeoParquet metadata may constrain or cross-check family planning, but it may
never admit a WKB record. The Arrow validity bitmap, record length, root header,
type id, dimensional encoding, structural counts, embedded headers, and record
bounds are authoritative.

Remove the `_metadata_declares_native_wkb(...)` decoder bypass. Missing or empty
`geometry_types` must not be interpreted as a positive native proof.

No decoder may translate an unsupported or rejected input record into a valid
empty geometry or a successful all-null carrier. The admission result must be
consumed before family decode and final assembly.

### 2. Direct endian-aware decode is the primary path

The read path will decode integer fields and coordinate payloads according to
the byte-order flag attached to the record that owns those fields. This avoids
an additional full-payload normalization write on one-shot reads.

A GPU WKB normalization mode may be added later for a reusable prepared
artifact, where a one-time big-to-little-endian rewrite can amortize across many
reads. It must be an explicit preparation choice with byte/value validation,
not an admission workaround.

Host compatibility decode is retained only for rows outside the declared native
contract. It is not an implementation of required big-endian support.

### 3. Family and endian specialization remains explicit

Do not add one generic branch-heavy decoder. Partition structural work by
geometry family and byte order. For multi-geometries, emit component tasks with
their own byte order, then partition those component tasks before coordinate
decode.

Homogeneous little-endian fast paths must not regress. Homogeneous big-endian
inputs should use equally specialized kernels. Truly mixed columns share the
same structural scan but feed family-local work queues.

### 4. Invalid structure is fail-closed

Every structural read must prove that its bytes lie within the owning record
before the value is used for sizing or allocation. Counts use overflow-checked
64-bit arithmetic during planning and must fit the owned-buffer offset contract
before narrowing.

Malformed lengths, impossible counts, invalid byte-order flags, embedded family
mismatches, truncated coordinates, integer overflow, and cursor overflow are
classified on device. No malformed payload may trigger an out-of-bounds read,
oversized allocation, CUDA error, or apparently valid geometry.

### 5. Null and empty semantics are distinct

- Arrow validity is authoritative for null rows, regardless of bytes in a null
  slot.
- Point EMPTY is the canonical NaN/NaN point representation. A partial NaN
  coordinate is not silently converted to empty.
- Zero structural counts preserve the family-specific empty representation for
  line, polygon, and multi families.
- Unsupported and malformed rows are neither null nor empty; they retain a
  decline/error classification until runtime policy resolves them.

## ADR-0046 Physical Workload Shape

The public result is row-aligned, but WKB execution is byte-, component-, and
coordinate-shaped.

| Contract | Decision |
|---|---|
| Public semantics | Decode an Arrow/GeoParquet WKB column with exact row order, nulls, empties, families, and coordinates. |
| Admissibility | Canonical 2D types 1-6 with valid root and embedded structure; byte order 0 or 1 at every applicable header. |
| Native input | Device WKB payload bytes, record offsets, Arrow validity, row count, and optional family metadata as a non-authoritative hint. |
| Physical work | Root records, payload bytes, embedded component headers, rings, coordinates, output offsets, and temporary bytes. |
| Temporary layout | Per-row status/count plan, family/endian row queues, embedded component task queues, ring/part counts, and output offsets. |
| Primitive shape | NVRTC structural validation and byte decode; CuPy element-wise masks/gathers; CCCL exclusive scans and reusable partition/sort primitives where benchmarked. |
| Native output | Device-resident `OwnedGeometryArray` plus decode-seeded `NativeGeometryMetadata`. |
| Precision | Integer/byte planning is exact; decoded coordinate storage remains fp64. No coordinate `PrecisionPlan` downcast is allowed. |
| Transfer boundary | No payload or coordinate D2H on successful native decode. At most one named bounded status/total packet if exact allocation requires host-known totals. |
| Export boundary | Shapely, GeoSeries, pandas, and Arrow host materialization remain explicit terminal or compatibility boundaries. |

### Saturation variants

- Many small records: grid-stride root scan, family/endian partition, and bulk
  family decode.
- Homogeneous points and simple lines: preserve lightweight specialized paths
  after the shared safety proof.
- Polygon-heavy batches: coordinate/ring tasks perform bulk decode rather than
  one thread serially copying every coordinate of a row.
- Large single or highly skewed records: route to a cooperative block/task
  variant after structural counts identify skew. Do not leave one thread to
  traverse and copy an unbounded geometry.
- Mixed-family or mixed-endian batches: partition once from the structural plan
  and launch uniform family/endian kernels; preserve original rows through a
  device row map.
- Sparse unsupported rows: compact only declined rows into the compatibility
  pool and merge results by original row position. Strict-native fails before
  compatibility materialization.

Dispatch and benchmark reporting must include rows, payload bytes, family
counts, component counts, ring counts, coordinate counts, endian counts,
declined rows, output bytes, and peak temporary bytes. Row count alone is not an
acceptable steady-state dispatch estimate.

## Device Plan And Status Model

Extend the current header scan into one reusable structural plan. Keep all
row-level arrays device-resident through decode and metadata assembly.

Minimum per-row state:

- Arrow validity;
- native/declined/error status;
- root byte order;
- family tag;
- primary structural count;
- coordinate, ring, part, and embedded-record counts;
- original row position;
- empty flag; and
- reason code for declined or malformed input.

Required reason classes:

- native little endian;
- native big endian;
- native mixed embedded endian;
- null;
- empty;
- invalid byte-order flag;
- truncated or malformed record;
- count/cursor/offset overflow;
- EWKB SRID or flag encoding;
- OGC Z/M/ZM type;
- GeometryCollection or unsupported family; and
- semantic invalidity governed by the public `on_invalid` contract.

Telemetry aggregates these reason codes without exporting per-row state. The
read event must distinguish native little-endian rows, native big-endian rows,
native mixed-embedded rows, and declined rows.

## Target Files And Responsibilities

- `src/vibespatial/kernels/core/wkb_decode_source.py`
  - endian-aware byte readers;
  - structural validation/count kernels;
  - family/endian task emission; and
  - coordinate/ring/part scatter kernels.
- `src/vibespatial/kernels/core/wkb_decode.py`
  - physical-plan orchestration;
  - CCCL scan/partition lifecycle;
  - skew variant selection;
  - owned-buffer assembly; and
  - warmup/kernel registration.
- `src/vibespatial/io/pylibcudf.py`
  - authoritative validity/offset/payload views;
  - removal of metadata-only admission;
  - GeoParquet decode integration; and
  - compatibility-pool handoff.
- `src/vibespatial/io/wkb.py`
  - shared status/reason contract;
  - public Arrow/list dispatch policy;
  - `on_invalid` behavior;
  - partial compatibility merge; and
  - telemetry text.
- `src/vibespatial/io/geoparquet.py`
  - WKB decoder wiring;
  - `binary_view` metadata-only transcode normalization;
  - exact schema/footer contract; and
  - strict-native read behavior.
- `tests/test_io_arrow.py` and focused GPU codec tests
  - mechanical WKB fixtures, oracle comparison, dispatch events, strict-native,
    transcode, schema, and safety coverage.
- `scripts/benchmark_io_arrow.py`
  - endian/family/mix microbenchmarks and machine-readable stage evidence.
- IO architecture, native inventory, and kernel registry documentation
  - native big-endian contract, remaining declines, and benchmark evidence.

Do not create a second decoder owned by GeoParquet. Arrow constructors,
GeoParquet reads, and direct WKB bridges must converge on the same structural
plan and family kernels.

## Milestones

### M0. Freeze the failure with red tests

- Add a GPU GeoParquet regression proving declared `geometry_types` cannot
  bypass actual WKB admission.
- Cover missing, empty, correct, incorrect, and mixed family metadata.
- Prove the current all-null success is rejected.
- Add strict-native coverage: supported big-endian is temporarily an explicit
  missing-native failure until M3/M4, never a successful empty result.
- Add `binary_view` transcode and timestamp/decimal schema-fidelity regressions.
- Record actual family/endian distributions for each available SpatialBench
  scale without turning them into production assumptions.

Exit: tests fail for the known reasons on the unmodified implementation and
mechanically reproduce the report.

### M1. Land the WKB admission firewall

- Remove metadata-only WKB admission.
- Make Arrow validity and byte-derived structural status authoritative.
- Replace all-null unsupported assembly with an explicit decline/error result.
- Ensure mixed supported/unsupported rows cannot produce invalid tags or family
  row offsets.
- Preserve existing little-endian native performance paths after the scan.

Exit: no unsupported byte stream enters decode, strict-native rejects every
decline, and auto mode remains correct through the explicit compatibility path.

### M2. Implement the endian-aware structural plan

- Add device helpers for u32/f64 reads in both byte orders.
- Validate complete root structure and every embedded WKB header.
- Compute exact family, part, ring, coordinate, component, and output-byte
  counts with overflow and record-bound checks.
- Emit family/endian row and component task descriptors.
- Aggregate a bounded completion/status packet only if allocation or exception
  semantics require it.
- Reuse the plan for decode; do not rescan payload structure independently in
  each family path.

Exit: every supported record has a complete device structural proof; malformed
and unsupported records have stable reason codes and cannot reach sizing or
decode.

### M3. Add native big-endian simple-family decode

- Implement Point, LineString, and Polygon decode for root byte order 0 and 1.
- Keep little- and big-endian work in uniform queues/kernels.
- Decode coordinates into fp64 SoA owned buffers.
- Preserve Point EMPTY NaN/NaN, partial NaN, zero-count empties, polygon ring
  offsets, closure, and row order exactly.
- Seed `NativeGeometryMetadata` from structural-plan validity, family, and
  emptiness state.

Exit: both endian variants for all simple families match the mechanical oracle,
run with zero fallback, and pass strict-native.

### M4. Add native multi-family and mixed-embedded decode

- Implement MultiPoint, MultiLineString, and MultiPolygon using embedded
  component tasks whose byte order is independent of the root and siblings.
- Cover LE root/LE child, LE root/BE child, BE root/LE child, BE root/BE child,
  and mixed sibling orders.
- Use coordinate/component-shaped scatter and a cooperative skew variant rather
  than one thread copying every coordinate of a multi record.
- Restore output rows through device row maps without host ordering work.

Exit: all six native families, homogeneous/mixed columns, and every root/child
endian combination match the oracle with zero fallback under strict-native.

### M5. Complete fallback-pool and observability behavior

- Compact only genuinely unsupported rows after structural classification.
- Preserve native rows on device while compatibility policy handles declined
  rows.
- Merge native and compatibility results by original row position.
- Fail before compatibility materialization in strict-native mode.
- Record native LE, native BE, native mixed-embedded, and reason-specific decline
  counts.
- Confirm GPU-selected success has no payload/coordinate D2H and no hidden
  Shapely or object-dtype materialization.

Exit: mixed native/unsupported batches preserve exact values and order; events
describe the actual execution path without double-counting or silent fallback.

### M6. Complete metadata-only transcode fidelity

- Admit `binary`, `large_binary`, and `binary_view` geometry fields.
- Normalize the published Arrow geometry field to `binary` while preserving
  every WKB byte.
- Carry exact decimal precision/scale and timestamp timezone/adjusted-to-UTC
  semantics through pylibcudf writer metadata.
- For any source logical type the writer cannot represent exactly, fail before
  destination publication with a precise unsupported-schema error.
- Write to a temporary destination and publish atomically only after footer,
  schema, row-count, metadata, value, and WKB-identity validation.
- Add a manifest/result summary that distinguishes standardization-only WKB from
  GPU-ready GeoArrow/normalized preparation.

Exit: the complete supported schema is exact except the intentional
`binary_view` to `binary` geometry normalization, and no failed transcode leaves
an apparently valid destination.

### M7. Benchmark and tune the reusable physical shape

- Extend IO codec benchmarks across family, endian, payload bytes, geometry
  complexity, homogeneous/mixed layout, null fraction, empty fraction, and
  skew.
- Measure structural scan, partition, size scan, decode, assembly, fallback
  merge, total wall, rows/s, payload GB/s, coordinates/s, launches, D2H bytes,
  and peak temporary memory.
- Profile on the local RTX 4090 and at least one datacenter GPU when available.
- Preserve demand-driven NVRTC and CCCL warmup declarations.
- Check register pressure, occupancy, wave quantization, and large-single versus
  many-small saturation before accepting micro-optimizations.

Exit: big-endian support meets the performance gates below without regressing
little-endian or GeoArrow paths.

### M8. End-to-end SpatialBench and landing gates

- Prepare a preserved-WKB GeoParquet dataset with the metadata-only transcode.
- Record byte-order/family/schema inventories for every shard.
- Run all Q1-Q12 against independent same-data references, not only queries
  previously affected by polygon decode.
- Start with SF1 correctness. Run SF100 when available. Run SF1000 only after
  all smaller correctness, memory, and profiling gates pass and only on a
  machine explicitly provisioned/authorized for that scale.
- Compare preserved WKB GeoParquet with the native GeoArrow control and explain
  every material stage difference.
- Run the mandatory full pipeline sparkline and resolve every unexplained
  CPU-heavy stage before landing.
- Update IO architecture/current-behavior docs, native inventory, benchmarks,
  and the source investigation with final evidence.

Exit: all acceptance criteria pass, the commit handoff contains 1M stage times,
and the normal `$commit` plus push workflow completes.

## Correctness Test Matrix

The oracle must be mechanical. Use Shapely/GEOS only as the host reference and a
small fixture builder for byte-order combinations Shapely cannot emit directly.
Never hand-code expected coordinate buffers independently of the fixture source.

| Dimension | Required cases |
|---|---|
| Family | Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon |
| Root endian | little, big |
| Embedded endian | all little, all big, root/child opposite, mixed siblings |
| Column composition | homogeneous family, promoted simple/multi mix, all-family mix |
| Validity | all valid, interleaved nulls, null slot with non-empty physical bytes |
| Empty | empty point, zero-point line, zero-ring polygon, zero-part multis, mixed empty/non-empty |
| Coordinates | finite, negative zero, infinities where accepted by oracle, partial NaN, large magnitude, subnormal, exact bit-pattern checks |
| Structure | holes, multiple rings, multiple parts, closed/unclosed rings as encoded, skewed counts, large single geometry |
| Declines | invalid byte order, truncated header/count/coordinate, impossible count, overflow, EWKB SRID, Z/M/ZM, GeometryCollection, embedded family mismatch |
| Carrier | binary, large_binary, binary_view, sliced/offset carrier behavior under the existing contract |
| Runtime | auto, explicit GPU, explicit CPU compatibility, strict-native |

Assertions must cover:

- public geometry equality and exceptions/warnings;
- exact validity, tags, family-row offsets, geometry/part/ring offsets, emptiness,
  coordinate values, and row ordering;
- no fallback for supported LE/BE/mixed-embedded 2D records;
- reason-specific events for every decline;
- no all-null success for unsupported input;
- zero payload/coordinate D2H on native success;
- no CUDA error after malformed-input tests; and
- deterministic repeat behavior after warmup.

Run targeted malformed and mixed-endian cases under `compute-sanitizer` when
available to catch out-of-bounds reads that result comparison alone may miss.

## Performance Gates

All timings use warm runs, identical inputs, synchronized operation boundaries,
and machine metadata. Immutable comparator data is prepared once and reused.

- WKB decode must meet the existing IO acceleration floor: at least `4x` the
  faster current host/compatibility baseline at the documented reference scale.
- Big-endian throughput for the same physical shape must be at least `80%` of
  little-endian throughput unless a profile proves an inherent byte-swap cost
  and an ADR records a different floor.
- Existing little-endian point, line, polygon, and mixed-family benchmarks may
  not regress by more than `5%` in repeat-3 median wall time or throughput.
- Native GeoArrow import/export may not regress by more than `5%`.
- GeoParquet unfiltered native scan retains the enforced local `2x` floor over
  the host comparator.
- Supported big-endian WKB reads record zero fallback rows and zero geometry
  payload/coordinate D2H.
- Memory remains bounded by input payload, exact owned output, and documented
  temporary work queues; no stage may allocate from an unvalidated WKB count.
- At 1M scale, mask construction, partition, candidate filtering, and buffer
  subsetting stages expected to be lightweight must remain below `1s`; any
  exception requires root-cause resolution or an ADR proving inherent cost.

Do not claim a win from query totals alone. Save stage times, device counters,
fallback events, transfer reasons/bytes, peak VRAM, and exact result hashes.

## Verification Commands

During implementation, start narrow and expand only after the local milestone
passes:

```bash
uv run ruff check
uv run pytest tests/test_io_arrow.py -q -k "wkb or geoparquet or transcode"
uv run pytest tests/test_io_gpu_dispatch.py -q
uv run pytest tests/test_io_arrow.py
uv run python scripts/benchmark_io_arrow.py --suite smoke
uv run python scripts/check_docs.py --check
```

Before completion:

```bash
VIBESPATIAL_STRICT_NATIVE=1 uv run pytest tests/test_io_arrow.py -q -k "wkb or geoparquet"
uv run pytest
uv run python scripts/benchmark_io_arrow.py --suite smoke
uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline
uv run python scripts/property_dashboard.py
```

GPU-sensitive commands must run with verified device visibility. Capture:

```bash
nvidia-smi -L
ls /dev/nvidia*
printenv CUDA_VISIBLE_DEVICES
```

The pre-push hook owns the heavy cached GPU health gate. Do not substitute a
sandboxed no-GPU result for the focused GPU verification above.

## Acceptance Criteria

- [x] GeoParquet metadata never bypasses byte-derived WKB admission.
- [x] Missing and empty `geometry_types` are not positive native proofs.
- [x] Canonical 2D big-endian Point, LineString, Polygon, MultiPoint,
  MultiLineString, and MultiPolygon decode natively on GPU.
- [x] Every embedded multi-geometry record honors its own byte-order flag.
- [x] Mixed LE/BE families and mixed embedded byte order remain native and
  preserve row order.
- [x] Null, empty, partial-NaN, malformed, unsupported, and `on_invalid`
  semantics match the mechanical oracle.
- [x] Supported big-endian input succeeds under strict-native with zero
  fallback and zero payload/coordinate D2H.
- [x] Unsupported input declines before decode and strict-native rejects it.
- [x] No unsupported input can return an apparently successful empty/all-null
  native result.
- [x] Structural counts and cursors are bounds- and overflow-checked before
  allocation or decode.
- [x] Native decode seeds reusable geometry metadata without a second scan.
- [x] `binary`, `large_binary`, and `binary_view` transcode to standard WKB
  GeoParquet with exact payload bytes.
- [x] Non-geometry values and logical schemas are preserved exactly, or the
  transcode fails before publication for an unsupported writer contract.
- [x] Little-endian, GeoArrow, and mixed-family performance rails do not
  regress.
- [x] Big-endian decode meets the stated throughput and GeoParquet scan floors.
- [x] SF1 and every larger executed scale match independent Q1-Q12 references.
- [x] The mandatory full pipeline profile has no unexplained CPU-heavy stage;
  1M stage names and times are included in the handoff/commit evidence.
- [x] Architecture, native inventory, tests, benchmarks, and report evidence
  describe the landed behavior accurately.
- [x] `$commit` pre-land review passes and the resulting commit is pushed as
  the terminal landing action for this implementation.

## Risks And Required Responses

| Risk | Required response |
|---|---|
| Metadata shortcut reappears for performance | Keep byte-derived plan authoritative; metadata can only narrow work after validation. |
| Endian branches cause warp divergence | Partition family/component tasks by endian before decode. |
| Multi record hides opposite-endian children | Validate and tag every embedded WKB header; cover mixed siblings mechanically. |
| Malformed counts trigger OOB or huge allocation | Record-bound checks, 64-bit overflow checks, fail-closed status, sanitizer cases. |
| Generic mixed decoder regresses homogeneous paths | Preserve specialized queues/kernels and benchmark each family independently. |
| Row-shaped kernels collapse on skew | Coordinate/component task queues plus cooperative large-single variant. |
| Partial fallback forces full-column host decode | Compact declined rows only and merge by device row map. |
| Schema footer claims types the physical writer changed | Compare physical/logical schemas after write; fail before atomic publication on mismatch. |
| Big-endian speed is achieved by hidden normalization | Record stages and device bytes; direct decode is the read-path contract. |
| SpatialBench-specific optimization leaks into production | Admission uses bytes, Arrow carriers, work estimates, and runtime policy only. |
| SF1000 hides correctness or capacity defects | Require SF1 and intermediate gates first; run only on an authorized provisioned machine. |

## Handoff Evidence

The completed implementation handoff must contain:

- exact revision and worktree status;
- GPU, driver, CUDA, pylibcudf, CuPy, PyArrow, Shapely, and GeoPandas versions;
- source/output dataset identities and WKB endian/family distributions;
- targeted and full test commands/results;
- per-family/endian benchmark tables and immutable comparator references;
- fallback, dispatch, D2H, launch, and peak-memory summaries;
- full 1M pipeline stage names and times;
- SF query result hashes and reference comparison summaries;
- any intentionally unsupported WKB reason classes; and
- the final commit and pushed remote revision.
