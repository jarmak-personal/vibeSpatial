# Synthetic Data

<!-- DOC_HEADER:START
Scope: Synthetic geometry dataset generator workflow, shapes, and verification guidance.
Read If: You are adding benchmark data, regression corpora, or synthetic geometry fixtures.
STOP IF: Your task already has a routed generator API file open and needs no workflow guidance.
Source Of Truth: Phase-0 synthetic data generator contract for tests and benchmarks.
Body Budget: 91/220 lines
Document: docs/testing/synthetic-data.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-5 | Preamble |
| 6-10 | Intent |
| 11-20 | Request Signals |
| 21-27 | Open First |
| 28-32 | Verify |
| 33-38 | Risks |
| 39-59 | Current Contract |
| 60-70 | Outputs |
| 71-82 | Pytest Integration |
| 83-91 | Verification |
DOC_HEADER:END -->

Use the synthetic generator for benchmarks, smoke tests, and future regression
corpora instead of checked-in external datasets.

## Intent

Describe the repo-local synthetic geometry generator, its current output
contract, and the verification path for extending it.

## Request Signals

- synthetic data
- generator
- benchmark data
- regression corpus
- fixture data
- seeded geometry
- license-free

## Open First

- docs/testing/synthetic-data.md
- src/vibespatial/testing/synthetic.py
- tests/test_synthetic_data.py
- docs/testing/performance-tiers.md

## Verify

- `uv run pytest tests/test_synthetic_data.py`
- `uv run python scripts/check_docs.py --check`

## Risks

- Large preset sizes can exhaust memory if generated eagerly in broad test runs.
- Optional GeoParquet export depends on `pyarrow` and should stay out of default smoke paths.
- Synthetic shapes can drift away from benchmark policy if dataset families are added ad hoc.

## Current Contract

The bootstrap generator currently provides deterministic Shapely-backed datasets
for:

- points: `uniform`, `clustered`, `grid`, `along-line`
- lines: `random-walk`, `grid`, `river`
- polygons: `regular-grid`, `convex-hull`, `star`
- multi geometries: `MultiPoint`, `MultiLineString`, `MultiPolygon`
- mixed arrays: configurable point/line/polygon ratios
- invalid shapes: bowtie-like, duplicate-vertex, repeated-segment, and `NaN`
  coordinate cases

Scale presets are defined in `SCALE_PRESETS`:

- `1K`
- `10K`
- `100K`
- `1M`
- `10M`

## Outputs

- `SyntheticDataset.to_geoseries()`
- `SyntheticDataset.to_geodataframe()`
- `SyntheticDataset.write_geojson(path)`
- `SyntheticDataset.write_geoparquet(path)` when parquet dependencies exist

The current bootstrap implementation is Shapely-first so benchmarks and tests
can start immediately. Owned device-oriented geometry buffers should become the
primary output once Phase 2 geometry-buffer work lands.

## Pytest Integration

The repo-level `synthetic_dataset` fixture accepts a `SyntheticSpec` and
returns a generated dataset for `point`, `line`, or `polygon` families. Use it
for narrow tests; larger benchmark suites should call the generator directly so
they can document scale and distribution choices explicitly.

GPU kernel tests should cover null, empty, and mixed-geometry cases in the same
file so early kernels do not accidentally specialize to clean homogeneous
inputs. Outside the vendored upstream tree, do not check in external data files
under `tests/`; build repo-local fixtures from this generator instead.

## Verification

Use this narrow gate when changing the synthetic generator:

```bash
uv run pytest tests/test_synthetic_data.py
uv run pytest tests/test_runtime_harness.py tests/test_geopandas_shim.py
uv run python scripts/check_docs.py --check
```
