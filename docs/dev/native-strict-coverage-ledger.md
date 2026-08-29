# Native Strict Coverage Ledger

<!-- DOC_HEADER:START
Scope: Disposition ledger for broad strict-native upstream GeoPandas coverage during the Native consolidation hold.
Read If: You are reconciling strict-native failures, changing WKB/WKT admission, deciding whether an upstream failure is in the native contract, or closing M1 of the Native consolidation plan.
STOP IF: You only need a focused operation implementation or ordinary non-strict upstream compatibility result.
Source Of Truth: Current classification and resolution evidence for every failure in the broad strict-native grouped sweep.
Body Budget: 99/180 lines
Document: docs/dev/native-strict-coverage-ledger.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-9 | Intent |
| 10-20 | Request Signals |
| 21-28 | Open First |
| 29-35 | Verify |
| 36-46 | Risks |
| 47-62 | Baseline |
| 63-80 | Disposition Ledger |
| 81-99 | Final Acceptance |
DOC_HEADER:END -->

## Intent

Own every failure from the broad grouped strict-native upstream sweep. This
ledger distinguishes an admitted native defect from a deliberately unsupported
contract and from a broken upstream harness. A focused canary cannot close a
row: the broad sweep and the Native inventory must agree.

## Request Signals

- strict-native upstream failure
- native coverage percentage
- WKB or WKT invalid input
- GeoArrow metadata
- dimensional geometry support
- NTv2 transform
- harness defect
- Native consolidation M1

## Open First

- `docs/dev/native-consolidation-execution-plan.md`
- `docs/dev/native-full-coverage-prd.md`
- `docs/dev/native-format-inventory.md`
- `docs/testing/native-coverage.md`
- `docs/testing/upstream-inventory.md`

## Verify

- `VIBESPATIAL_STRICT_NATIVE=1 uv run python scripts/upstream_native_coverage.py --grouped --group-by file --json`
- `uv run pytest tests/test_io_arrow.py tests/test_io_gpu_dispatch.py tests/test_wkt_gpu.py -q`
- `uv run pytest tests/test_dissolve_pipeline.py -q`
- `uv run python scripts/check_docs.py --check`

## Risks

- Calling every strict decline a defect would silently expand the current 2D
  geometry and transform contract.
- Calling an admitted null, schema, or invalid-input failure unsupported would
  hide a real public compatibility defect.
- A stale percentage can claim closure after focused fixes while the grouped
  sweep still fails.
- GeometryCollection compatibility must remain an explicit public boundary;
  it is not a concrete owned device family today.

## Baseline

The August 27, 2026 pre-fix grouped sweep produced 2,215 passed, 78 failed,
410 skipped, and 6 xfailed, for a 96.35% native pass rate. The 78 failures were
mechanically reviewed by test identity and failure reason. Counts below sum to
78; they are the starting ledger, not the post-fix acceptance result.

| Class | Count | Meaning |
|---|---:|---|
| correctness defect | 7 | Public contract was admitted but returned the wrong result or invalid-input behavior. |
| metadata/schema defect | 6 | Geometry was correct but exported GeoArrow/WKB field metadata was incomplete. |
| missing admitted capability | 12 | Current 2D public contract should support the input without a strict decline. |
| intentional unsupported contract | 50 | The test requires dimensional or transform support outside the current repo-owned native contract. |
| optional dependency | 0 | No failure was caused by a legitimately absent optional dependency. |
| harness defect | 3 | The test or external fixture expectation is broken independently of native execution. |

## Disposition Ledger

| Failure family | Starting count | Disposition and evidence | State |
|---|---:|---|---|
| Point dissolve `level`, `sort`, `dropna`, and sparse categorical products | 3 | `NativeGrouped` Point union consumes sorted observed groups directly, preserves the full categorical product, and keeps empty-group masks device-resident until terminal GeometryCollection typing. | fixed; final broad sweep confirmed |
| Invalid one-point LineString WKB/WKT `on_invalid` | 4 | Native scanners identify the semantic invalidity, implement raise/warn/ignore, and null invalid rows without mislabeled fallback. | fixed; final broad sweep confirmed |
| XY WKB GeoArrow field metadata | 6 | Every WKB field carries `ARROW:extension:name=geoarrow.wkb` and extension metadata, including CRS JSON where present. | fixed; final broad sweep confirmed |
| Nullable/bytes WKB and WKT constructors | 7 | Nullable scalars and byte strings are admitted; validity stays separate from placeholder parse storage. | fixed; final broad sweep confirmed |
| Empty WKB and empty factorization | 2 | Empty/all-null inputs create repo-owned nullable geometry arrays without fallback. | fixed; final broad sweep confirmed |
| GeometryCollection WKT ingress | 2 | Direct GeometryCollection WKT is handled at an observable public compatibility boundary because there is no concrete owned GeometryCollection family. | fixed at explicit compatibility boundary; final broad sweep confirmed |
| Mixed line/polygon `build_area` | 1 | The clean landing tree still reaches the existing union fallback for mixed line/polygon input. A separate native topology-assembly workstream is preserved outside this candidate until its zero-copy and upstream-contract gates pass. | open admitted capability |
| XYZ GeoArrow/WKB export | 36 | The owned coordinate contract is 2D. A strict decline is correct until a separate dimensionality plan adds native Z storage and codec semantics. | intentional unsupported |
| ISO Point Z WKB | 1 | Same 2D coordinate-carrier limit; do not silently discard Z. | intentional unsupported |
| M/ZM `has_m`, M point construction, and `get_coordinates(include_m=True)` | 3 | Native M/ZM storage and public dimensional semantics do not exist; strict decline is the correct observable result. | intentional unsupported |
| NTv2 grid transforms | 10 | vibeProj does not implement the required grid-shift transform family. This is transform breadth, not a hidden CPU fallback candidate. | intentional unsupported |
| GDAL CA remote fixture cases | 2 | The upstream remote fixture URL/certificate path is broken independently of backend selection. Keep outside the native pass denominator until the harness is repaired upstream or vendored deterministically. | harness defect |
| Feather complex-type expectation | 1 | The copied expectation is stale against the active Arrow/GeoPandas contract. Repair only through the refreshable harness, never by hand-editing vendored tests. | harness defect |

## Final Acceptance

The August 29 landing-tree grouped file sweep reports 2,239 passed, 54 failed,
410 skipped, and 6 xfailed, for a 97.39% native pass rate. Its JSON packet is
`/tmp/native-consolidation-landing-strict.json` with SHA-256
`554d7e7286fdd3b63467e61eeb6321a41a36f0a2936b87b81756adf63200574d`.
The session exposed and fixed two admitted NYBB interval-component
validity failures and one pickle reconstruction field omitted by the new
selection-provenance contract. Focused tests, the complete 126-test
geodataframe file, and the final broad rerun pass. Every admitted correctness,
schema, null, family, dtype, and index case except `build_area` passes. The 54
remaining failures match the ledger exactly: 36 XYZ GeoArrow/WKB, one ISO
Point Z, three M/ZM, ten NTv2, three harness cases, and one open admitted
`build_area` capability. There are zero unclassified failures. Durable full
JSON publication remains tied to the final clean candidate revision.

M1 remains open only for `build_area`. Ring-local winding passes its exact
physical-shape canary and mandatory profile; its evidence lives in
`native-physical-shape-ledger.md`.
