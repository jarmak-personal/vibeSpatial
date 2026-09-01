# Arrow And GeoParquet IO

<!-- DOC_HEADER:START
Scope: Arrow, GeoParquet, and WKB IO boundary around owned geometry buffers and GPU-native decode paths.
Read If: You are changing Arrow, GeoParquet, WKB adapters, or owned-buffer IO decode and encode.
STOP IF: Your task already has the specific IO adapter open and only needs local implementation detail.
Source Of Truth: IO architecture for Arrow, GeoParquet, and WKB owned-buffer bridges.
Body Budget: 242/260 lines
Document: docs/architecture/io-arrow.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-7 | Intent |
| 8-18 | Request Signals |
| 19-25 | Open First |
| 26-31 | Verify |
| 32-37 | Risks |
| 38-53 | Decision |
| 54-75 | Performance Notes |
| 76-228 | Current Behavior |
| 229-242 | Measured Local Baseline |
DOC_HEADER:END -->

## Intent

Define the repo-owned Arrow, GeoParquet, and WKB IO boundary around owned
geometry buffers while keeping GPU-native formats as the design center.

## Request Signals

- io arrow
- geoparquet
- wkb
- geoarrow
- parquet
- arrow bridge
- io decode
- io encode

## Open First

- docs/architecture/io-arrow.md
- src/vibespatial/io/geoarrow.py
- src/vibespatial/io/geoparquet.py
- src/vibespatial/io/wkb.py

## Verify

- `uv run pytest tests/test_io_arrow.py`
- `uv run python scripts/benchmark_io_arrow.py --suite smoke`
- `uv run python scripts/check_docs.py --check`

## Risks

- Repeatedly rebuilding Shapely-heavy intermediate state in the Arrow path destroys throughput.
- Silent host decode hides missing GPU paths.
- WKB compatibility bridge becoming the de facto layout instead of GeoArrow.

## Decision

- Treat GeoArrow as the canonical geometry interchange surface for owned
  buffers.
- Route GeoPandas `to_arrow`, `from_arrow`, `to_parquet`, and `read_parquet`
  through repo-owned adapters instead of calling vendored helpers directly.
- Keep a real optional `pylibcudf` GeoParquet scan path for unfiltered scans,
  but fall back explicitly when that runtime or a GPU-side bbox filter path is
  unavailable.
- Model bbox pushdown at the adapter layer from GeoParquet covering metadata or
  point encoding so later GPU scanners can reuse the same decision logic.
- Treat WKB as a compatibility bridge, not a canonical layout, and keep its
  encode/decode path explicit.
- Adopt aligned GeoArrow buffers zero-copy and normalize only when the incoming
  layout does not match the canonical owned schema.

## Performance Notes

- Arrow and GeoParquet should converge on owned buffers instead of repeatedly
  rebuilding shapely-heavy intermediate state.
- The fastest long-term path is device-side GeoArrow and WKB codecs plus a GPU
  Parquet scanner; today the repo-owned adapters make the fallback visible
  instead of silently hiding a host path.
- GeoParquet scans without bbox filters can already target a `pylibcudf`
  reader when that dependency is present.
- Covering-based bbox pruning should stay outside geometry decode so row-group
  selection can reject work before expensive geometry materialization.
- The current planner compares loop and vectorized row-group pruning and uses
  the vectorized strategy once row-group counts are large enough to justify it.
- GeoArrow import and export should prefer shared buffer views over eager host
  copies whenever dtypes and shapes already match owned-buffer requirements.
- Host geometry objects should stay lazily materialized; GeoArrow adoption must
  not construct Shapely objects unless a caller explicitly requests them.
- GeoParquet scans should decode native GeoArrow family columns directly into
  owned buffers after scan instead of bouncing through Shapely.
- `read_parquet_batches` should yield whole-row-group public frames while
  retaining device attributes and owned geometry until explicit export.

## Current Behavior

- `GeoDataFrame.to_arrow`, `GeoDataFrame.from_arrow`, `GeoSeries.to_arrow`,
  `GeoSeries.from_arrow`, `GeoDataFrame.to_parquet`, and `geopandas.read_parquet`
  now dispatch through repo-owned wrappers.
- Owned GeoArrow and WKB bridge helpers exist as first-class repo APIs.
- Dispatch and fallback events make the current host/device choice observable.
- Repo-owned WKB bridges now use a staged native path for supported families:
  - one byte-authoritative structural scan validates root and embedded headers,
    record bounds, counts, families, and independent byte order
  - point, linestring, polygon, multipoint, multilinestring, and multipolygon
    rows use endian-specialized GPU decode into device owned buffers
  - homogeneous Arrow WKB point, uniform-linestring, and uniform-polygon
    batches now take raw-buffer fast paths ahead of the generic GPU bridge and
    bulk-promote to device when a GPU runtime is available
  - malformed, EWKB, dimensional, GeometryCollection, or otherwise unsupported
    rows compact into an explicit sparse compatibility pool; strict-native
    rejects before compatibility materialization
- `geopandas.read_parquet(..., bbox=...)` now builds a repo-owned metadata
  summary when pyarrow metadata is available, selects row groups before the
  table read, and passes those row groups into the host read path instead of
  decoding the full dataset first.
- Repo-owned GeoArrow bridges now distinguish:
  - `copy`: always normalize into fresh owned buffers
  - `auto`: share aligned buffers, normalize only when required
  - `share`: require a fully aligned layout and fail otherwise
- Repo-owned GeoParquet export now also accepts grouped native constructive
  results as an explicit terminal boundary, so grouped dissolve-style outputs
  can write directly without first materializing an intermediate GeoDataFrame.
- Geometry-only native results can also write directly to GeoParquet, so
  geometry-producing pipelines do not need to rebuild a temporary GeoDataFrame
  just to hit the writer boundary.
- Row-preserving native clip results can also write directly to GeoParquet, so
  constructive filter pipelines do not need to materialize a public
  GeoDataFrame before the terminal write boundary.
- Point-only row-preserving clip results now also lower directly into the
  shared native tabular boundary, so simple clip producer paths no longer need
  to materialize a temporary public spatial object before Arrow-family export.
- More generally, default `clip(..., keep_geom_type=False)` producer paths now
  lower row-preserving non-point results directly into the shared native
  tabular boundary too; only the stricter `keep_geom_type=True` compatibility
  cases still need the public clip materializer today.
- Deferred overlay constructive results can also write directly to GeoParquet,
  so union, identity, and symmetric-difference native paths no longer need to
  collapse into an intermediate public frame just to hit the writer boundary.
- Overlay pairwise and left-row constructive fragments now also project
  attributes directly into the shared native attribute-table boundary, so
  union/intersection/difference producer paths no longer need to build pandas
  attribute fragments before native terminal export.
- Deferred spatial-join export results can also write directly to GeoParquet,
  so native join pairs and join-context assembly can stay deferred until the
  terminal write boundary instead of rebuilding a public frame first.
- Join export now also produces Arrow-backed native attribute payloads before
  the sink boundary, so `sjoin` and `sjoin_nearest` no longer need to build a
  joined pandas frame just to cross into Arrow-family writers.
- Native join, clip, and overlay exports now converge on a shared
  `NativeTabularResult` boundary of attribute columns plus native geometry
  before any terminal sink runs.
- That same shared boundary now lowers directly to Arrow too, so GeoParquet
  and other Arrow-family sinks do not need to rebuild a temporary
  GeoDataFrame-shaped export just to cross the terminal format boundary.
- The low-level GeoPandas Arrow helper for `geometry_encoding="geoarrow"`
  now also delegates to that shared native tabular boundary, so GeoArrow
  export no longer keeps a separate helper-local DeviceGeometryArray
  materialization branch alongside the repo-owned adapter.
- Repo-owned GeoArrow export keeps promotable single/multi mixes on the native
  adapter instead of the host bridge; device-backed supported mixes promote to
  native multi-family GeoArrow encodings.
- The shared native boundary now also owns Parquet and Feather terminal
  emission, so Arrow-family write sinks no longer depend on GeoDataFrame
  assembly when a native result is already available.
- `NativeTabularResult` now accepts a shared attribute payload abstraction, so
  Arrow-family sinks can lower Arrow-backed attribute tables directly instead
  of requiring pandas frames as the only internal attribute representation.
- Native GeoParquet payload writes now also keep Arrow-backed attributes plus
  owned geometry on the device writer until the sink actually declines a
  feature, so the public `to_parquet` boundary no longer eagerly materializes
  a temporary `GeoSeries` just to discover that the native writer would have
  accepted the payload.
- Shared native Arrow export now follows the same rule for WKB and GeoArrow
  payloads: when owned geometry is already available, it encodes directly from
  the owned buffers instead of rebuilding a temporary `GeoSeries`; public
  host-backed GeoSeries/GeoDataFrame GeoArrow export does so for supported
  families and preserves GeoPandas errors for unsupported mixed families.
- Public host-originated `GeoDataFrame.to_arrow` and `GeoSeries.to_arrow`
  exports now record an explicit GeoArrow compatibility-writer boundary, while
  device-backed CPU misses stay in `io_write` coverage as real acceleration
  gaps instead of being hidden by the compatibility bucket.
- Public `GeoDataFrame.to_arrow` and `GeoSeries.to_arrow` own the terminal
  `NativeExportBoundary` event when they call the repo-owned Arrow adapter; direct
  low-level adapter calls still record their own boundary.
- Terminal GeoParquet compatibility decisions such as non-filesystem sinks or
  non-native compression now record an explicit CPU dispatch at the sink
  boundary instead of a fallback event, so strict-native mode still rejects
  hidden mid-pipeline fallback without forbidding explicit compatibility export.
- Repo-owned `read_geoparquet_owned(...)` and public `read_parquet_batches(...)`
  now provide the scan seam: pylibcudf/pyarrow selection, decoded-byte row-group
  planning, direct GeoArrow decode, device-backed projected attributes, and
  owned-buffer concatenation or bounded public batch delivery.
- `read_geoparquet_native(...)` now seeds `NativeGeometryMetadata` on its
  `NativeTabularResult` from decode-time validity/family classification and
  GeoParquet total bounds without forcing eager per-row bounds recomputation;
  chunked native reads concatenate that metadata along with geometry and
  attributes.
- Owned WKB/GeoArrow-style geometry results that cross
  `to_native_tabular_result(...)` also seed cached `NativeGeometryMetadata`, so
  immediate native consumers can reuse classification state instead of
  rediscovering it after decode.
- Public WKB and GeoArrow native imports from `GeoDataFrame.from_arrow` and
  `GeoSeries.from_arrow` now attach metadata-seeded `NativeFrameState`, so
  immediate native consumers can use decoded owned geometry without a second
  decode/materialization.
- The `pylibcudf` GeoParquet device path now decodes all native GeoArrow
  families (`point`, `linestring`, `polygon`, `multipoint`,
  `multilinestring`, `multipolygon`) into device-resident owned buffers without
  forcing host family payload materialization first.
- The `pylibcudf` GeoParquet device path now also decodes canonical WKB point,
  linestring, polygon, multipoint, multilinestring, and multipolygon columns
  into device-resident owned buffers without a Shapely round-trip.
- Legacy binary-WKB Parquet can be transcoded metadata-only to GeoParquet 1.1:
  `binary`, `large_binary`, and `binary_view` geometry normalize to Arrow
  `binary`; pylibcudf preserves exact WKB and typed attributes, validates the
  complete result, then publishes atomically.
- `NativePartitionedParquetSink` orders bounded device-clustered batches onto one
  writer stream; its file-identity-bound sidecar routes equality reads to exact row groups.
  Empty filtered WKB scans synthesize device offsets instead of using the host decoder.
- Mixed canonical WKB columns now keep the same GPU-first contract too:
  point-only, linestring-only, and point/linestring columns still use the
  lightweight `pylibcudf` helpers, while heavier or broader family mixes route
  through the staged GPU WKB decode pipeline after the same header scan.
- Canonical 2D little endian, big endian, and mixed embedded-endian WKB are
  native. EWKB SRID, Z/M/ZM, GeometryCollection, invalid order, malformed
  structure, and families outside the owned model retain stable decline codes.
- GeoParquet `geometry_types` is a planning hint only and never bypasses the
  byte-derived structural proof. Successful native reads export only one
  bounded aggregate telemetry packet, never payload or coordinate bytes.
- Repo-owned native GeoArrow codecs provide family-specialized homogeneous IO:
  - point, linestring, polygon, multilinestring, and multipolygon extension
    arrays decode through dedicated family builders
  - device-backed homogeneous exports rebuild Arrow arrays from the device codec
  - low-level/native-tabular unsupported host- and device-backed mixes drop to
    the repo-owned WKB bridge instead of forcing `construct_wkb_array(...)`
  - public unsupported mixed GeoArrow exports preserve GeoPandas `ValueError`
    until partition-and-restore mixed codecs land
  - successful homogeneous native export no longer records a fallback event on
    the public GeoPandas Arrow surface
- The verified `pylibcudf` transport matrix is now checked in explicitly:
  local paths, `bytes`, `BytesIO`, `DeviceBuffer`, multi-source scans,
  row-group selection, filters, and `ChunkedParquetReader` are confirmed in
  [`pylibcudf-capabilities.md`](/home/picard/repos/vibeSpatial/docs/architecture/pylibcudf-capabilities.md).
- Local partitioned-directory GeoParquet reads and normalized `file://` public
  reads are now explicitly verified to stay on the `pylibcudf` scan backend.

## Measured Local Baseline

RTX 4090 synchronized repeat-3 measurements at `100K` identical records show
the endian-aware decoder at `11.1x-34.4x` the Shapely comparator across all six
families. Big-endian throughput is `0.98x-1.05x` little endian for identical
physical shapes. The benchmark and machine-readable evidence come from:

```bash
uv run python scripts/benchmark_io_arrow.py --wkb-endian --scale 100000 --repeat 3
```

The steady-state rail requires `>=4x` host speedup, big endian at `>=80%` of
little endian, zero fallback, and zero payload/coordinate D2H. Ten-thousand-row
results remain informational because fixed launch/allocation latency dominates.
