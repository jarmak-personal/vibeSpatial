

## Summary

The new metadata-only Parquet-to-GeoParquet transcode makes a practical fast
step 0 for legacy WKB datasets: it scans and rewrites Parquet with pylibcudf,
adds standard GeoParquet metadata, and preserves geometry bytes without a CPU
geometry decode. SpatialBench exposes an important contract boundary in that
design: its trip points are little-endian WKB, while its building and zone
polygons are big-endian WKB.

The transcode is correct and fast for both because it treats WKB as opaque
bytes. The original GPU WKB decoder was little-endian-only, however. More
importantly, the GeoParquet reader used declared `geometry_types` as sufficient
proof that a WKB column could enter that decoder. GeoParquet family metadata
does not declare byte order. Big-endian polygons therefore entered the
little-endian pipeline and were returned as all-null geometry instead of
declining to the explicit compatibility path.

This report records the original defect and the now-completed native big-endian
extension, plus two adjacent transcode findings. It is not a request for a
SpatialBench-specific branch: selection must follow WKB bytes, Arrow types, and
the normal runtime policy.

## Observed behavior

A small end-to-end control performed:

```text
original legacy-WKB Parquet
  -> pylibcudf metadata-only WKB GeoParquet transcode
  -> footer/schema and byte-identity validation
  -> public vibeSpatial SpatialBench queries
```

Preparation succeeded without geometry decode. Full validation confirmed equal
row counts, compatible schemas, standard GeoParquet metadata, and byte-identical
WKB values for every geometry column.

Before safe WKB admission, polygon-heavy queries failed or returned incorrect
empty results:

- Q2 failed while selecting a known matching zone from an all-null geometry
  carrier.
- Q4 returned 0 rows instead of 258.
- Q6 and Q10 failed during reductions over empty decoded geometry.

The failure was not corrupted input or transcode output. Source and destination
polygon WKB were byte-identical, and non-geometry zone columns were equal.

## Root cause

The WKB GeoParquet decode selector had two paths:

1. scan the actual WKB headers, admit supported little-endian 2D records, and
   decline unsupported records observably; or
2. when GeoParquet metadata declared supported geometry families, enter the
   little-endian device decoder directly.

Path 2 is unsound. GeoParquet `geometry_types` constrains geometry family, not
WKB byte order, dimensional flags, EWKB headers, or byte order of embedded
records. The little-endian kernel correctly rejected the big-endian records,
but the direct path assembled that rejection as an all-null result.

The safe correction is to scan the WKB byte stream before device admission even
when family metadata is present. With that change, the same big-endian inputs
take the existing explicit compatibility route and Q2, Q4, Q6, and Q10 complete
with the expected row counts.

## Measured control

These are single engineering runs from one consistent environment, not release
medians. They show the cost and correctness shape rather than a publishable
performance comparison.

### Fast preparation

| Table | Rows | Metadata-only transcode |
|---|---:|---:|
| Building | 20,000 | 0.52s |
| Trip | 6,000,000 | 0.44s across two shards |
| Zone | 156,095 | 4.48s across six shards |

An exact WKB scan of the prepared dataset completed in 3.27s and found no
payload differences. This confirms that standard WKB GeoParquet preparation
can be inexpensive; it does not imply that all preserved WKB is immediately
eligible for the current native decoder.

### Read-path consequence

| Query | Preserved WKB GeoParquet | Native GeoArrow control | Result rows |
|---|---:|---:|---:|
| Q2 | 29.05s | 3.26s | 1 |
| Q6 | 30.58s | 3.25s | 3 |
| Q10 | 29.79s | 3.80s | 100 |

The preserved-WKB runs are correct after safe admission but pay the explicit
big-endian compatibility cost. The native-GeoArrow control avoids that cost.
This distinguishes two useful preparation contracts:

- **Standardization-only prep:** add GeoParquet metadata, preserve WKB bytes,
  and finish quickly.
- **GPU-ready prep:** additionally normalize unsupported WKB representation or
  encode native GeoArrow so later reads stay on the device fast path.

## Generalized implementation direction

### Required correctness fix

Always inspect WKB headers before entering an endian-specific decoder. Declared
families may reduce family planning work, but they cannot prove byte order or
record layout. An unsupported byte stream must produce an observable decline or
compatibility event, never an empty native result.

### Implemented native big-endian support

Big-endian WKB is now in the native v1 contract and is implemented in the
reusable GPU WKB decoder rather than in conversion or benchmark code. The decoder:

1. read each root record's byte-order flag;
2. decode integer counts, type tags, and coordinates with that order;
3. honor the byte-order flag on every embedded Point, LineString, or Polygon
   record inside multi-geometries;
4. retain the current family-specialized and mixed-family device pipelines;
5. classify dimensional, EWKB, invalid, and unsupported records separately;
6. preserve empty and null semantics; and
7. expose telemetry for native little-endian, native big-endian, and declined
   rows.

Do not rewrite bytes merely to make an unsafe decoder appear eligible. A
GPU-side normalization stage is reasonable only if selected from reuse and cost:
for a one-shot read, direct endian-aware decode is preferable; for a reusable
prepared artifact, normalization or native GeoArrow may amortize better.

## Adjacent findings

### Arrow `binary_view` input

The metadata-only transcode originally admitted `binary` and `large_binary` but
rejected Arrow `binary_view`. SpatialBench zone WKB is exposed as `binary_view`.
The generalized transcode should accept it and publish the geometry field as
standard Arrow `binary`, matching the pylibcudf writer's physical BYTE_ARRAY
output. A focused GPU test covers this normalization.

### Logical schema fidelity

The transcode preserves attribute values, but one control observed physical
schema widening:

```text
timestamp[ms]       -> timestamp[ms, tz=UTC]
decimal128(15, 5)   -> decimal128(18, 5)
```

The source logical types remain in the serialized Arrow footer schema, but
PyArrow resolves the rewritten Parquet fields to the widened types above. This
did not explain the geometry failures and did not change the tested query
values, but a production transcode should either preserve these logical types
exactly or document and test the allowed widening. Decimal precision can likely
be carried through pylibcudf column metadata; timestamp adjusted-to-UTC behavior
may require writer support or an explicit contract.

## Verification completed

The integrated implementation accepts and normalizes WKB `binary_view`, keeps
family metadata subordinate to the byte-derived structural plan, and decodes
canonical 2D big-endian WKB natively for all six families. Synthetic coverage
includes independently mixed embedded byte order and malformed records; CUDA
memcheck reports zero errors.

Fresh preserved-WKB Q1-Q12 runs match GeoArrow at SF1, SF10, and SF100. SF100
also matches the frozen independent GeoPandas reference. Repeat-3 totals are
5.86s, 43.24s, and 258.78s respectively, with zero WKB fallback. Detailed
inventories, per-family codec rails, hashes, stage timings, and artifact paths
are recorded in `reports/wkb-geoparquet-big-endian-implementation-plan.md`.
SF1000 was not inspected or executed on this workstation.

## Acceptance criteria

- Big-endian WKB never enters a little-endian-only decoder based solely on
  GeoParquet family metadata.
- Unsupported big-endian input declines observably and returns correct results
  through the configured compatibility policy.
- If native big-endian support is added, point, line, polygon, and all multi-
  families match a mechanical oracle for root and embedded byte-order variants.
- Mixed little- and big-endian rows, nulls, and empty geometry are covered.
- `binary`, `large_binary`, and `binary_view` legacy WKB fields transcode to
  standards-compliant WKB GeoParquet.
- Geometry bytes and non-geometry values are preserved by metadata-only prep.
- Logical timestamp and decimal behavior is preserved or explicitly specified.
- Strict-native mode rejects unsupported records rather than returning an
  apparently successful empty result.
- A fresh prepared SF1 run matches the independent query references before an
  SF1000 end-to-end timing is published.
