# Upstream Test Inventory

<!-- DOC_HEADER:START
Scope: Ranked upstream GeoPandas contract slices, dependencies, and runtime weight.
Read If: You are choosing which vendored test groups to promote or implement next.
STOP IF: Your task is limited to a single already-routed upstream module.
Source Of Truth: Phase-0 upstream test inventory for implementation planning.
Body Budget: 106/220 lines
Document: docs/testing/upstream-inventory.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-6 | Preamble |
| 7-15 | Intent |
| 16-25 | Request Signals |
| 26-32 | Open First |
| 33-38 | Verify |
| 39-44 | Risks |
| 45-59 | Ranking |
| 60-75 | Notes |
| 76-88 | Recommended Promotion Order |
| 89-95 | Runtime Tiers |
| 96-106 | Verification |
DOC_HEADER:END -->

This inventory ranks the vendored GeoPandas contract slices by expected
implementation leverage, optional dependency pressure, and collected pytest
weight. Use it to choose which upstream groups to promote next.

## Intent

Map the highest-value upstream test groups to:

- actual collected node weight when available
- required or optional dependencies
- expected runtime tier for local smoke versus broader coverage
- the implementation-order epic they most directly unlock

## Request Signals

- upstream tests
- inventory
- contract slices
- collect-only
- runtime weight
- dependency groups
- promotion order

## Open First

- docs/testing/upstream-inventory.md
- tests/upstream/README.md
- pyproject.toml
- tests/upstream/geopandas/conftest.py

## Verify

- `uv run python scripts/check_docs.py --check`
- `uv run python scripts/intake.py "inventory upstream tests by weight dependency group and runtime"`
- `uv run pytest --collect-only -q tests/upstream/geopandas/tests/test_extension_array.py`

## Risks

- Collected node counts do not measure wall-clock runtime directly.
- Optional-dependency groups can appear artificially light when collection is skipped.
- Promoting high-weight slices too early can hide narrower kernel milestones.

## Ranking

| Rank | Slice | Collected nodes | Dependency group | Runtime weight | Target epic |
|---|---|---:|---|---|---|
| 1 | `tests/test_extension_array.py` | 504 | default | heavy | Phase 6a / `o17.6.1`, `o17.6.4`, Phase 7 / `o17.7.1` |
| 2 | `tests/test_sindex.py` + `tools/tests/test_sjoin.py` | 459 | default, `pyproj` for some cases, GEOS>=3.10 for `dwithin` | heavy | Phase 3 / `o17.3.*`, Phase 4 / `o17.4.3`, Phase 7 / `o17.7.2` |
| 3 | `tests/test_geodataframe.py` | 405 | default, `pyproj` for CRS/sjoin variants | heavy | Phase 6a / `o17.6.4`, Phase 7 / `o17.7.3` |
| 4 | `io/tests/test_file.py` + `io/tests/test_file_geom_types_drivers.py` | 409 | `pyogrio` default, `fiona` optional | heavy | Phase 6b / `o17.6.11`, `o17.6.12`, `o17.6.13`, Phase 7 / `o17.7.4` |
| 5 | `tests/test_geom_methods.py` | 335 | default, `pyproj` and newer GEOS for selected methods | heavy | Phase 4 / `o17.4.1`, `o17.4.2`; Phase 5 / `o17.5.*` |
| 6 | `tests/test_overlay.py` + `tools/tests/test_clip.py` | 219 | default, `pyproj`; some file-backed comparisons need I/O stack | medium-heavy | Phase 5 / `o17.5.1` to `o17.5.5`, Phase 7 / `o17.7.3` |
| 7 | `tests/test_array.py` | 114 | default, `pyproj` for CRS transforms | medium | Phase 2 / `o17.2.1` to `o17.2.4`, Phase 6a / `o17.6.1`, Phase 7 / `o17.7.1` |
| 8 | `io/tests/test_arrow.py` + `io/tests/test_geoarrow.py` | skipped without `pyarrow`; 0 collected here | `pyarrow` optional, `fsspec`/`requests`/`aiohttp` for remote cases, `pyproj` for CRS metadata paths | heavy when enabled | Phase 6b / `o17.6.2`, `o17.6.7`, `o17.6.8`, `o17.6.9`, Phase 7 / `o17.7.4` |
| 9 | `io/tests/test_sql.py` | 42 | `sqlalchemy` + `psycopg[binary]` optional, live PostGIS fixture expectations | medium | Phase 6b / SQL fallback adapters, Phase 7 / `o17.7.4` |
| 10 | `io/tests/test_infer_schema.py` + `io/tests/test_pickle.py` | 27 | default | light | Phase 6b fallback and compatibility bridges |

## Notes

- Collected node counts came from `uv run pytest --collect-only -q` against the
  vendored files on this repo checkout.
- `test_arrow.py` and `test_geoarrow.py` are currently gated entirely by the
  missing `pyarrow` extra, so their weight is known conceptually but not yet
  collected in this environment.
- `test_file.py` and `test_file_geom_types_drivers.py` are disproportionately
  large because they cross product engines, drivers, geometry mixes, and file
  suffixes.
- `test_extension_array.py` is the largest single compatibility slice and is
  the best denominator for Phase 6a progress because it stresses pandas
  alignment, dtype behavior, reshaping, casting, and groupby semantics.
- `test_sindex.py` plus `test_sjoin.py` form the clearest bridge from Phase 3
  indexing kernels into user-visible GeoPandas behavior.

## Recommended Promotion Order

1. `test_config.py` remains the smoke gate for vendored import health.
2. `test_array.py` should be the first geometry-buffer contract slice promoted
   once Phase 2 starts because it is core, medium-sized, and default-deps only.
3. `test_extension_array.py` should be the first major Phase 6a promotion
   target because it is large but still free of heavyweight I/O extras.
4. `test_sindex.py` and `test_sjoin.py` should anchor Phase 3 and 4 promotion.
5. `test_overlay.py`, `test_clip.py`, and `test_geom_methods.py` should stay
   behind Phase 4 and 5 kernel milestones.
6. Arrow/GeoArrow and SQL slices should remain explicitly optional until the
   `upstream-optional` dependency group is installed in CI or local benches.

## Runtime Tiers

- `light`: under 50 collected nodes or no external system dependency
- `medium`: 50 to 150 collected nodes, mostly default dependencies
- `medium-heavy`: 150 to 300 nodes or moderate fixture/data pressure
- `heavy`: 300+ nodes, strong parametrization, or optional/external stack needs

## Verification

Use these commands to refresh the inventory when vendored tests or dependency
groups change:

```bash
uv run pytest --collect-only -q tests/upstream/geopandas/tests/test_extension_array.py
uv run pytest --collect-only -q tests/upstream/geopandas/tests/test_sindex.py tests/upstream/geopandas/tools/tests/test_sjoin.py
uv run pytest --collect-only -q tests/upstream/geopandas/tests/test_overlay.py tests/upstream/geopandas/tools/tests/test_clip.py
uv run pytest --collect-only -q tests/upstream/geopandas/io/tests/test_file.py tests/upstream/geopandas/io/tests/test_file_geom_types_drivers.py
```
