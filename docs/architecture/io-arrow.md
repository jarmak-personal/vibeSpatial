# Arrow And GeoParquet IO

<!-- DOC_HEADER:START
Scope: Arrow, GeoParquet, and WKB IO boundary around owned geometry buffers and GPU-native decode paths.
Read If: You are changing Arrow, GeoParquet, WKB adapters, or owned-buffer IO decode and encode.
STOP IF: Your task already has the specific IO adapter open and only needs local implementation detail.
Source Of Truth: IO architecture for Arrow, GeoParquet, and WKB owned-buffer bridges.
Body Budget: 209/260 lines
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
| 76-171 | Current Behavior |
| 172-209 | Measured Local Baseline |
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
- Chunked GeoParquet scans should concatenate owned-buffer batches, not
  materialized geometry objects.

## Current Behavior

- `GeoDataFrame.to_arrow`, `GeoDataFrame.from_arrow`, `GeoSeries.to_arrow`,
  `GeoSeries.from_arrow`, `GeoDataFrame.to_parquet`, and `geopandas.read_parquet`
  now dispatch through repo-owned wrappers.
- Owned GeoArrow and WKB bridge helpers exist as first-class repo APIs.
- Dispatch and fallback events make the current host/device choice observable.
- Repo-owned WKB bridges now use a staged native path for supported families:
  - one header scan separates native rows from the explicit fallback pool
  - point, linestring, polygon, multipoint, multilinestring, and multipolygon
    rows use family-specialized native decode or encode
  - malformed, unsupported, or non-little-endian rows compact into explicit
    fallback instead of forcing the whole batch through Shapely
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
- The shared native boundary now also owns Parquet and Feather terminal
  emission, so Arrow-family write sinks no longer depend on GeoDataFrame
  assembly when a native result is already available.
- `NativeTabularResult` now accepts a shared attribute payload abstraction, so
  Arrow-family sinks can lower Arrow-backed attribute tables directly instead
  of requiring pandas frames as the only internal attribute representation.
- The GeoParquet writer consumes that shared native tabular boundary directly,
  so writer-local payload assembly is no longer the place where native result
  semantics live.
- Terminal GeoParquet compatibility decisions such as non-filesystem sinks or
  non-native compression now record an explicit CPU dispatch at the sink
  boundary instead of a fallback event, so strict-native mode still rejects
  hidden mid-pipeline fallback without forbidding explicit compatibility export.
- Repo-owned `read_geoparquet_owned(...)` now provides the scan-engine seam for
  `o17.6.20`:
  - backend selection: `pylibcudf` or `pyarrow`
  - row-group chunk planning from metadata summaries
  - direct GeoArrow-family decode into owned buffers
  - chunk concatenation at the owned-buffer layer
- The `pylibcudf` GeoParquet device path now decodes all native GeoArrow
  families (`point`, `linestring`, `polygon`, `multipoint`,
  `multilinestring`, `multipolygon`) into device-resident owned buffers without
  forcing host family payload materialization first.
- The `pylibcudf` GeoParquet device path now also decodes WKB point-only,
  linestring-only, and mixed point/linestring columns into device-resident owned
  buffers without a Shapely round-trip.
- Polygon-family WKB device decode is still follow-on work; the current staged
  WKB bridge remains the contract to port for polygon, multipoint,
  multilinestring, and multipolygon WKB rows.
- Repo-owned native GeoArrow codecs now provide family-specialized encode and
  decode for homogeneous geometry columns:
  - point, linestring, polygon, multilinestring, and multipolygon extension
    arrays decode through dedicated family builders
  - homogeneous exports encode directly from owned buffers to native GeoArrow
    arrays instead of routing through the generic host bridge
  - mixed-family exports stay on explicit WKB fallback until partition-and-restore
    mixed codecs land
  - successful homogeneous native export no longer records a fallback event on
    the public GeoPandas Arrow surface

## Measured Local Baseline

Host-only validation on this machine already shows why native GeoArrow decode
must be the design center even before GPU throughput is measured:

- `100K` point rows, GeoArrow GeoParquet decode: about `37.0M` rows/s
- `100K` point rows, WKB GeoParquet decode: about `170K` rows/s
- `20K` polygon rows, GeoArrow GeoParquet decode: about `3.05M` rows/s
- `20K` polygon rows, WKB GeoParquet decode: about `64.6K` rows/s

That is roughly `218x` better on the point case and `47x` better on the polygon
case, which validates the native scan-engine direction before `pylibcudf`
throughput is available locally.

The new family-specialized codec benchmarks also show the bridge structure is
paying off before device kernels land:

- `100K` point rows, native GeoArrow encode: about `98.9M` rows/s
- `100K` point rows, host bridge encode: about `11.2M` rows/s
- `20K` polygon rows, native GeoArrow decode: about `5.68M` rows/s
- `20K` polygon rows, host bridge decode: about `4.62M` rows/s

That is about `8.8x` faster on point encode and about `1.23x` faster on polygon
decode. The remaining bottleneck is the `pylibcudf -> pyarrow -> owned` bridge,
which is now isolated behind the family codec boundary instead of being mixed
into the public adapter layer.

The staged WKB bridge now shows the same pattern on the compatibility path:

- `1M` point rows, native WKB decode: about `1.54M` rows/s
- `1M` point rows, host WKB decode bridge: about `177K` rows/s
- `1M` point rows, native WKB encode: about `5.37M` rows/s
- `1M` point rows, host WKB encode bridge: about `145K` rows/s

That is about `8.7x` faster on decode and about `37x` faster on encode while
keeping unsupported rows isolated in an explicit fallback pool. The remaining
work for `o17.6.22` is no longer bridge shape; it is moving the same staged
scan, partition, size, and scatter contract onto CCCL-backed device passes.
