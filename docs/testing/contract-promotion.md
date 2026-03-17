<!-- DOC_HEADER:START
Scope: Upstream GeoPandas contract promotion gates, test-group ordering, and promotion workflow.
Read If: You are promoting, inspecting, or verifying upstream contract test groups.
STOP IF: Your task already has the specific test group open and only needs local pass criteria.
Source Of Truth: Contract promotion workflow for upstream GeoPandas test groups.
Body Budget: 142/260 lines
Document: docs/testing/contract-promotion.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-5 | Intent |
| 6-13 | Request Signals |
| 14-19 | Open First |
| 20-24 | Verify |
| 25-30 | Risks |
| 31-37 | Phase 7 Order |
| 38-51 | Current Promotion Gate |
| 52-82 | Query Promotion Gate |
| 83-113 | Constructive Promotion Gate |
| 114-137 | IO Promotion Gate |
| 138-142 | Workflow |
DOC_HEADER:END -->

## Intent

Define how upstream GeoPandas contract slices move from passive inventory to
active promotion gates, starting with the hottest extension-array surfaces.

## Request Signals

- contract promotion
- upstream tests
- promotion gate
- test promotion
- vendored tests

## Open First

- docs/testing/contract-promotion.md
- tests/upstream/
- scripts/contract_promotion.py

## Verify

- `uv run python scripts/contract_promotion.py check vibeSpatial-o17.7.1`
- `uv run python scripts/check_docs.py --check`

## Risks

- Promoting a contract group before smoke commands and tracked pass criteria are both explicit.
- Mistaking contract promotion for GPU-performance completion.
- Adding promotion gates that depend on optional runtime dependencies not available in CI.

## Phase 7 Order

1. `vibeSpatial-o17.7.1`: `test_extension_array.py` and `test_array.py`
2. `vibeSpatial-o17.7.2`: `test_sindex.py`, `test_geom_methods.py`, and `test_sjoin.py`
3. `vibeSpatial-o17.7.3`: `test_overlay.py`, `test_clip.py`, and `test_geodataframe.py`
4. `vibeSpatial-o17.7.4`: file, Arrow, GeoArrow, and SQL IO slices with optional dependency gates

## Current Promotion Gate

`vibeSpatial-o17.7.1` is promoted with these smoke commands:

```bash
uv run pytest tests/upstream/geopandas/tests/test_extension_array.py -q
uv run pytest tests/upstream/geopandas/tests/test_array.py -q
```

Tracked pass criteria:

- `test_extension_array.py` stays green aside from intentional skips and documented readonly xfails.
- `test_array.py` stays green as the semantic companion slice for geometry-array behavior.

## Query Promotion Gate

`vibeSpatial-o17.7.2` is promoted with these smoke commands:

```bash
uv run pytest tests/upstream/geopandas/tests/test_sindex.py -q
uv run pytest tests/upstream/geopandas/tests/test_geom_methods.py -q
uv run pytest tests/upstream/geopandas/tools/tests/test_sjoin.py -q
```

Tracked pass criteria:

- `test_sindex.py` stays green aside from the existing unordered-result xfails.
- `test_geom_methods.py` stays green with dependency-aware skips only.
- `test_sjoin.py` stays green aside from the existing documented xfail and skip.

Performance watch commands:

```bash
uv run python scripts/benchmark_bounds_pairs.py --rows 20000 --tile-size 256
uv run python scripts/benchmark_spatial_query.py --rows 20000 --overlap-ratio 0.2
```

Current host baseline on this machine:

- bounds-pair benchmark at `20K` rows: about `0.71s`
- spatial query benchmark at `20K` rows and `20%` overlap:
  repo-owned query `~1.91s`, direct Shapely query `~0.0042s`

That gap is why Phase 7 promotion should not be mistaken for GPU-performance completion. The contract is promoted; the GPU-dominant implementation target is still ahead of us.

## Constructive Promotion Gate

`vibeSpatial-o17.7.3` is promoted with these smoke commands:

```bash
uv run pytest tests/upstream/geopandas/tests/test_overlay.py -q
uv run pytest tests/upstream/geopandas/tools/tests/test_clip.py -q
uv run pytest tests/upstream/geopandas/tests/test_geodataframe.py -q
```

Tracked pass criteria:

- `test_overlay.py` stays green with the current constructive skips only.
- `test_clip.py` stays green against the owned clip and fallback surface.
- `test_geodataframe.py` stays green with dependency-aware skips only.

Performance watch commands:

```bash
uv run python scripts/benchmark_clip_rect.py --kind polygon --rows 5000
uv run python scripts/benchmark_make_valid.py --rows 10000 --invalid-every 20
```

Current host baseline on this machine:

- overlay slice: `128 passed`, `2 skipped`
- clip slice: `89 passed`
- geodataframe slice: `126 passed`, `1 skipped`
- clip benchmark at `polygon-5000`: owned `~0.200s`, Shapely `~0.00116s`
- make_valid benchmark at `10K` rows / every `20th` invalid: compact path `~1.19x` faster than baseline

## IO Promotion Gate

`vibeSpatial-o17.7.4` is promoted with these smoke commands:

```bash
uv run pytest tests/upstream/geopandas/io/tests/test_file.py -q
uv run pytest tests/upstream/geopandas/io/tests/test_arrow.py -q
uv run pytest tests/upstream/geopandas/io/tests/test_geoarrow.py -q
uv run pytest tests/upstream/geopandas/io/tests/test_sql.py -q
```

Dependency-aware rules:

- `test_file.py` should pass with optional-driver skips.
- `test_arrow.py` and `test_geoarrow.py` may exit skip-only when `pyarrow` is absent.
- `test_sql.py` may exit skip-only when PostGIS drivers or a live database are absent.

Current host baseline on this machine:

- file IO slice: `134 passed`, `160 skipped`, `2 xfailed`, `1 xpassed`
- Arrow slice: skip-only because `pyarrow` is not installed
- GeoArrow slice: skip-only because `pyarrow` is not installed
- SQL slice: `2 passed`, `40 skipped`

## Workflow

- Inspect a promotion group with `uv run python scripts/contract_promotion.py show vibeSpatial-o17.7.1`.
- Run the smoke gate with `uv run python scripts/contract_promotion.py check vibeSpatial-o17.7.1`.
- Only mark a contract group promoted after its smoke commands and tracked pass criteria are both explicit.
