# Constructive Result Unification Execution Plan

<!-- DOC_HEADER:START
Scope: Execution plan for unifying constructive results across overlay, clip, dissolve, and planner-selected fused execution families.
Read If: You are planning or executing the constructive-result refactor, overlay architecture rewrite, or planner-driven fusion work.
STOP IF: You already have the active milestone open and only need local implementation detail.
Source Of Truth: Program plan for making NativeTabularResult the canonical constructive boundary and using planner-selected workload families for acceleration.
Body Budget: 884/900 lines
Document: docs/dev/constructive-result-unification-execution-plan.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-13 | Intent |
| 14-29 | Request Signals |
| 30-45 | Open First |
| 46-56 | Verify |
| 57-75 | Risks |
| 76-107 | Mission |
| 108-156 | Target Shape |
| 157-175 | Non-Goals |
| 176-192 | Working Principles |
| 193-204 | Tracking |
| 205-320 | Milestone M0: Baseline, Contract Freeze, And Deletion Inventory |
| 321-376 | Milestone M1: Shared Result Core And Canonical Contract Extraction |
| 377-454 | Milestone M2: Explicit Export Boundary Rewrite |
| ... | (6 additional sections omitted; open document body for full map) |
DOC_HEADER:END -->

## Intent

Turn the overlay/clip/dissolve architecture discussion into an execution plan
with tracked milestones, explicit deletion targets, fusion guidance, and
verification gates.

This plan is for doing the larger refactor now, not for incremental patching of
the current wrapper stack. The goal is to make the constructive result boundary
correct once and then build planner-selected acceleration on top of that stable
shape.

## Request Signals

- constructive result
- `NativeTabularResult`
- overlay refactor
- clip refactor
- dissolve refactor
- device-native result boundary
- execution plan
- milestone plan
- fusion plan
- workload family
- true operation fast path
- overlay architecture
- delete constructive wrappers

## Open First

- `docs/dev/constructive-result-unification-execution-plan.md`
- `docs/decisions/0042-device-native-result-boundary.md`
- `docs/decisions/0016-overlay-reconstruction-plan.md`
- `docs/architecture/overlay-reconstruction.md`
- `docs/architecture/clip-fast-paths.md`
- `docs/architecture/dissolve.md`
- `docs/architecture/fusion.md`
- `docs/architecture/runtime.md`
- `src/vibespatial/api/_native_results.py`
- `src/vibespatial/api/tools/overlay.py`
- `src/vibespatial/api/tools/clip.py`
- `src/vibespatial/overlay/dissolve.py`
- `src/vibespatial/runtime/fusion.py`

## Verify

- `uv run python scripts/check_docs.py --check`
- `uv run pytest tests/test_overlay_api.py tests/test_clip_rect.py tests/test_dissolve_pipeline.py tests/test_gpu_dissolve.py -q`
- `uv run pytest tests/upstream/geopandas/tests/test_overlay.py -k geometry_not_named_geometry`
- `uv run pytest tests/upstream/geopandas/tests/test_dissolve.py -k dissolve_multi_agg`
- `uv run pytest tests/upstream/geopandas/tests/test_pandas_methods.py -k groupby_metadata`
- `uv run python scripts/health.py --tier contract --check`
- `uv run python scripts/health.py --tier gpu --check`
- `uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline`

## Risks

- Replacing wrapper types without hardening the export boundary first can just
  move the same metadata bugs into a new file.
- Reworking clip and dissolve separately from overlay can recreate the current
  divergence under new type names.
- Overfusing generic overlay into a mega-kernel would hide reusable structures,
  erase diagnostics, and likely make correctness work harder rather than
  faster.
- Leaving `GeoDataFrame` composition in place anywhere on the hot path would
  preserve the main architectural failure mode even if native result types are
  renamed.
- Planner-selected fast paths can regress semantics if they are not forced back
  through the same canonical result contract.
- The push can devolve into “make tests pass” unless each milestone also
  deletes obsolete result wrappers and reduces host-side assembly.
- End-to-end speed can still regress if GPU coverage rises only through helper
  stages while public constructive workflows remain host-dominated.

## Mission

Make `NativeTabularResult` the canonical constructive result across overlay,
clip, and dissolve.

The refactor is successful only if all of the following are true:

- overlay, clip, and dissolve native paths converge on one result boundary
- GeoPandas export becomes one explicit terminal boundary instead of repeated
  intermediate composition
- planner-selected workload families become the main mechanism for “true
  operation” fast paths
- fused execution happens inside those planner-selected families, not as one
  generic overlay mega-kernel
- the current metadata/export failures disappear because the contract is
  simpler, not because more compatibility glue was layered on top

This plan treats the current constructive wrapper pyramid as transitional debt:

- `PairwiseConstructiveResult`
- `LeftConstructiveResult`
- `SymmetricDifferenceConstructiveResult`
- `PairwiseConstructiveFragment`
- `LeftConstructiveFragment`
- `ConcatConstructiveResult`
- `GroupedConstructiveResult`
- `ClipNativeResult`

These types may survive temporarily during migration, but the plan is not
complete until they are deleted or reduced to thin compatibility shims with no
architectural significance.

## Target Shape

The target architecture is:

- one canonical constructive result: `NativeTabularResult`
- one geometry-only carrier: `GeometryNativeResult`
- one explicit compatibility boundary: `to_geodataframe()`, `to_arrow()`,
  `to_parquet()`, `to_feather()`
- many execution families selected by a planner before heavy work starts

The planner should select among workload families such as:

- `clip_rewrite`
- `broadcast_right_intersection`
- `broadcast_right_difference`
- `containment_bypass + batched_sh_clip + remainder_overlay`
- `coverage_union`
- `grouped_union`
- `generic_reconstruction`

The planner output should describe:

- operation
- workload shape
- geometry family mix
- topology class
- semantic flags such as `keep_geom_type`
- result shape
- execution family
- fusion opportunities

Within a selected family, fused execution is encouraged where the stages are
ephemeral and device-local. Fused execution is not the goal by itself; it is a
mechanism for reducing launch count, intermediate traffic, and host re-entry
once the planner has already identified the true operation shape.

The canonical result contract must own:

- primary geometry column
- secondary geometry columns
- active geometry name
- CRS
- column order
- attrs / provenance that survive export

No constructive path should build its final answer by concatenating
`GeoDataFrame`s or by repeatedly calling `to_geodataframe()` on intermediate
parts.

## Non-Goals

This push is not about:

- preserving the current constructive wrapper taxonomy
- building a whole-program lazy graph runtime
- writing one universal fused overlay kernel
- changing public APIs for its own sake
- accepting host-side composition because “there are no users yet”
- making the tests green by papering over export bugs with more pandas surgery

This push is also not done when:

- overlay still composes results through intermediate `GeoDataFrame`s
- clip or dissolve still use a different native result model
- the planner still reasons in terms of public surface names instead of
  workload families
- fused paths exist but export still falls back through the old host boundary

## Working Principles

- Make the result boundary authoritative before optimizing around it.
- Delete wrappers instead of shuffling them into more files.
- Keep one explicit compatibility boundary and make it correct.
- Choose execution family before heavy work starts; do not re-plan mid-kernel.
- Use fused stage clusters inside a workload family, not across every
  algorithmic boundary.
- Persist reusable structures such as indexes, sorted half-edges, and grouped
  offsets when the pipeline needs them again.
- Keep stable-order semantics explicit anywhere GeoPandas parity depends on
  deterministic row ordering.
- Treat `NativeTabularResult` as the public internal contract for constructive
  work. Everything else lowers into it.
- Verify every milestone with upstream contract tests, health gates, and the
  mandatory end-to-end profile.

## Tracking

- [x] M0. Baseline, contract freeze, and deletion inventory
- [x] M1. Shared result core and canonical contract extraction
- [x] M2. Explicit export boundary rewrite
- [x] M3. Overlay cutover to canonical native tabular results
- [x] M4. Clip cutover to canonical native tabular results
- [x] M5. Dissolve cutover to canonical native tabular results
- [x] M6. Planner-selected workload families
- [x] M7. Fused stage clusters and specialized kernel family acceleration
- [x] M8. Wrapper deletion, convergence, and final verification

## Milestone M0: Baseline, Contract Freeze, And Deletion Inventory

### Goal

Freeze the target architecture, capture before-state evidence, and make the
wrapper deletion target explicit before implementation starts.

### Primary Surfaces

- `docs/dev/constructive-result-unification-execution-plan.md`
- `docs/decisions/0042-device-native-result-boundary.md`
- `src/vibespatial/api/_native_results.py`
- `scripts/health.py`
- benchmark and upstream overlay/dissolve verification rails

### Checklist

- [x] Capture the current overlay, clip, and dissolve native result families.
- [x] Write down which wrapper types are transitional and must be deleted.
- [x] Capture the current upstream failure set for:
  `geometry_not_named_geometry`, `dissolve_multi_agg`, and `groupby_metadata`.
- [x] Capture current `contract` and `gpu` health outputs as the before-state.
- [x] Capture the current full end-to-end pipeline profile on the target GPU.
- [x] Confirm that the canonical target is `NativeTabularResult`, not a new
  sibling abstraction.
- [x] Confirm that fused execution will be planner-selected by workload family,
  not implemented as a single universal overlay kernel.

### Baseline Snapshot Captured 2026-04-14

- Target machine visibility:
  GPU `0` is `NVIDIA GeForce RTX 4090`; `/dev/nvidia0`, `/dev/nvidiactl`,
  `/dev/nvidia-uvm`, `/dev/nvidia-uvm-tools`, and `/dev/nvidia-modeset` are
  visible; `CUDA_VISIBLE_DEVICES` is unset.
- Property dashboard before-state:
  `6/6` clean, total distance `0.00`.
- Current constructive wrapper inventory:
  `_native_results.py` currently defines
  `PairwiseConstructiveResult`,
  `LeftConstructiveResult`,
  `SymmetricDifferenceConstructiveResult`,
  `PairwiseConstructiveFragment`,
  `LeftConstructiveFragment`,
  `ConcatConstructiveResult`, and
  `GroupedConstructiveResult`;
  `api/tools/clip.py` defines `ClipNativeResult`.
- Current hot-file sizes:
  `api/tools/overlay.py` `5786` lines,
  `api/_native_results.py` `2584` lines,
  `api/tools/clip.py` `3024` lines,
  `overlay/dissolve.py` `2956` lines.
- Focused upstream failure slice:
  `test_geometry_not_named_geometry` currently fails for
  `[union-True]`, `[intersection-True]`, and `[identity-True]`;
  `test_dissolve_multi_agg` fails on `Pandas4Warning`;
  `test_groupby_metadata` fails for all four
  `[geometry|geom] x [None|EPSG:4326]` cases.
- Export-seam diagnosis from the focused slice:
  overlay failures route through `ConcatConstructiveResult` /
  `to_geodataframe()` and `GeoDataFrame.crs`;
  dissolve and groupby failures route through
  `_materialize_attribute_geometry_frame()` and its
  `reindex(..., copy=False)` frame rebuild path.
- Contract health before-state:
  `Repo Health — contract: FAIL`;
  required surfaces passing `7/8`;
  required red is `overlay` at `236/260`;
  optional red is `performance_rails` at `25/26`.
- GPU health before-state:
  `Repo Health — gpu: FAIL`;
  property summary `6/6` clean, distance `0.00`;
  GPU acceleration `20.08%`
  (`4349` GPU dispatches / `21654` total dispatches);
  CPU dispatches `16717`;
  fallback dispatches `16`.
- Full upstream GPU-sensitive sweep inside the GPU tier:
  `1954` passed, `423` skipped, `14` xfailed, `8` failed in `114.61s`.
  The eight real failures are the same focused export/metadata slice captured
  above.
- 1M pipeline baseline summary:
  `join-heavy` total `110.6ms`, planner-selected runtime `gpu`, actual runtime
  `hybrid`; `dissolve_groups` at `60.15ms` is the dominant CPU stage.
  `constructive` total `65.9ms`, actual runtime `hybrid`; `write_output` at
  `59.07ms` dominates.
  `predicate-heavy` total `104.8ms`, actual runtime `gpu`; `read_geojson` at
  `76.28ms` dominates.
  `zero-transfer` total `40.57ms`, actual runtime `gpu`; `read_input` at
  `19.02ms` and `write_output` at `11.88ms` dominate.
- 10k full shootout baseline summary:
  `benchmark_results/2026-04-14-constructive-result-unification-m0/shootout_suite_10k.json`
  captures the full directory run at `scale=10k`, `repeat=3`, `warmup=true`.
  All `10/10` workflows passed and fingerprint-matched; `4/10` are already at
  GeoPandas parity or better and `6/10` are still below parity.
- 10k shootout parity leaders:
  `site_suitability` `2.847x` (`654.34ms` GeoPandas vs `228.00ms` vibeSpatial),
  `redevelopment_screening` `1.783x` (`685.98ms` vs `399.89ms`),
  `network_service_area` `1.374x` (`91.55ms` vs `66.28ms`), and
  `nearby_buildings` `1.021x` (`97.37ms` vs `95.08ms`).
- 10k shootout below-parity workloads:
  `flood_exposure` `0.994x` (`36.12ms` vs `36.26ms`),
  `transit_service_gap` `0.937x` (`230.01ms` vs `244.38ms`),
  `parcel_zoning` `0.856x` (`63.64ms` vs `75.01ms`),
  `corridor_flood_priority` `0.795x` (`158.52ms` vs `199.24ms`),
  `vegetation_corridor` `0.679x` (`292.94ms` vs `431.79ms`), and
  `accessibility_redevelopment` `0.265x` (`215.91ms` vs `826.97ms`).
- M0 decisions locked:
  the canonical target is `NativeTabularResult`;
  acceleration work should be planner-selected by workload family with fused
  stage clusters inside those families, not a single generic overlay mega-kernel.

### Exit Criteria

- the target contract is written down unambiguously
- the deletion inventory is explicit
- the before-state is documented well enough to measure structural progress

## Milestone M1: Shared Result Core And Canonical Contract Extraction

### Goal

Separate the shared constructive result core from relation-join and legacy
wrapper glue, then make the core contract explicit enough that overlay, clip,
and dissolve can all target it directly.

### Primary Surfaces

- `src/vibespatial/api/_native_results.py`
- shared result helpers and tests

### Checklist

- [x] Isolate the generic result primitives:
  `NativeAttributeTable`, `GeometryNativeResult`, `NativeGeometryColumn`,
  `NativeTabularResult`.
- [x] Move relation-join-specific logic out of the constructive center of
  gravity where possible.
- [x] Make the column-order and geometry-column invariants explicit in the
  shared result core.
- [x] Add direct tests for:
  primary geometry name, secondary geometry columns, CRS, attrs, and column
  order.
- [x] Make `to_native_tabular_result()` explicit and exhaustive for all
  supported constructive result families during migration.
- [x] Add a small shared builder layer for:
  pairwise projection, left-row-preserving projection, grouped projection, and
  native concat.

### Completion Notes Captured 2026-04-15

- Extracted the shared contract into
  `src/vibespatial/api/_native_result_core.py` so the generic primitives and
  explicit GeoPandas export boundary can stand alone.
- `src/vibespatial/api/_native_results.py` now imports that core and acts as
  the relation-join / wrapper-lowering layer instead of defining the contract
  inline.
- Added direct contract coverage in `tests/test_native_result_core.py` for
  active geometry name, secondary geometry columns, CRS, attrs, late attribute
  columns, and geometry-column-order validation.
- Kept `to_native_tabular_result()` as the explicit migration choke point and
  documented the shared builder layer that lowers pairwise, left-preserving,
  grouped, and concat families into `NativeTabularResult`.
- The current `Pandas4Warning` from `reindex(..., copy=False)` still exists in
  the extracted core and is intentionally left for M2, where the export
  boundary rewrite replaces that path rather than hiding it.

### Exit Criteria

- the shared result core can stand on its own without overlay-specific naming
- the core contract is test-covered directly
- later milestones can lower into the same shared builders instead of
  re-inventing export logic

## Milestone M2: Explicit Export Boundary Rewrite

### Goal

Rewrite the GeoPandas export boundary once so geometry-name restoration, CRS,
column order, and multi-geometry handling stop depending on brittle pandas
rebuild tricks.

### Primary Surfaces

- `src/vibespatial/api/_native_results.py`
- `src/vibespatial/api/geodataframe.py`

### Checklist

- [x] Rewrite `_materialize_attribute_geometry_frame()` around the canonical
  contract instead of post-hoc frame surgery.
- [x] Remove reliance on `reindex(..., copy=False)` plus manual class mutation
  for correctness.
- [x] Make active-geometry restoration explicit and test it directly.
- [x] Preserve secondary geometry columns without silently clobbering the
  active geometry state.
- [x] Add direct regression tests for:
  geometry-not-named-geometry, CRS access, groupby metadata, and column order.
- [x] Confirm Arrow / Feather / GeoParquet export still works from the new
  boundary without forcing `GeoDataFrame` materialization first.

### Exit Criteria

- the metadata/export seam is singular and trustworthy
- the known upstream metadata failures are either already green or clearly
  isolated above the export layer

### Completion Notes Captured 2026-04-15

- Rewrote `_materialize_attribute_geometry_frame()` in
  `src/vibespatial/api/_native_result_core.py` to build the final ordered
  payload directly from attributes plus explicit geometry columns, instead of
  rebuilding a frame through `reindex(..., copy=False)`.
- Tightened the core contract so attribute columns cannot overlap primary or
  secondary geometry names, and removed the remaining `Pandas4Warning` from
  loader-backed column renames in the shared core.
- Reworked constructive wrapper lowering in
  `src/vibespatial/api/_native_results.py` so projected frames split into
  true attributes plus explicit secondary geometry columns before Arrow
  storage. The lowering path now preserves legacy overwrite semantics where
  pairwise overlay intentionally clobbers a temporary `geometry` column, while
  left-row-preserving exports keep secondary geometry columns unless the final
  active geometry name would collide.
- Fixed two boundary-specific structural bugs exposed by the rewrite:
  `NativeAttributeTable.concat()` now preserves row counts for zero-attribute
  tables, and projected / left-preserving output-order reconstruction now
  normalizes integer-labeled columns against the actual Arrow-backed schema
  instead of reintroducing stale raw labels.
- Symmetric-difference lowering now computes one final merged column order
  after concat, so the active geometry column lands in the correct final
  position instead of being interleaved between attribute blocks.
- Added direct regression coverage in `tests/test_native_result_core.py` for
  the no-`Pandas4Warning` export boundary and zero-column attribute concat,
  and updated `tests/test_overlay_api.py` so the no-materialization assertions
  track the new pairwise / left lowering choke points instead of the old
  attribute helper.
- Focused upstream regressions are green:
  `test_geometry_not_named_geometry`,
  `test_dissolve_multi_agg`,
  `test_groupby_metadata`,
  upstream clip donut / keep-geom-type setup via symmetric difference, and
  upstream Arrow column-order preservation.
- Contract and GPU ratchets are clean against the committed baselines:
  contract baseline comparison returns `exit_code 0` with no regressions;
  GPU baseline comparison also returns `exit_code 0` with no regressions.
  Property distance stayed at `0.0` (`6/6` clean).
- Mandatory 1M profile after the rewrite shows no new host-side stall:
  `join-heavy` remains dominated by `dissolve_groups` (`44.92ms`),
  `constructive` by `write_output` (`61.84ms`),
  `predicate-heavy` by `read_geojson` (`77.71ms`), and
  `zero-transfer` by `read_input` (`17.36ms`) plus `write_output` (`14.31ms`).

## Milestone M3: Overlay Cutover To Canonical Native Tabular Results

### Goal

Make overlay native paths return `NativeTabularResult` directly and stop using
constructive fragments and wrapper composition as the primary execution model.

### Primary Surfaces

- `src/vibespatial/api/tools/overlay.py`
- `src/vibespatial/overlay/`
- `src/vibespatial/api/_native_results.py`
- `tests/test_overlay_api.py`

### Checklist

- [x] Make intersection lower directly to `NativeTabularResult`.
- [x] Make difference lower directly to `NativeTabularResult`.
- [x] Make identity compose canonical native tabular results, not
  `GeoDataFrame`-producing fragments.
- [x] Make symmetric difference compose canonical native tabular results, not
  wrapper objects.
- [x] Make union compose canonical native tabular results, not fragment trees.
- [x] Reduce `overlay()` to validation, policy, planning, dispatch logging, and
  one final export boundary.
- [x] Keep no-materialization write paths green for Arrow / Feather /
  GeoParquet.
- [x] Remove overlay hot-path dependence on the constructive wrapper classes.

### Exit Criteria

- all overlay native entrypoints return `NativeTabularResult`
- overlay no longer relies on repeated `to_geodataframe()` composition
- overlay export-without-materialization tests still pass

### Completion Notes Captured 2026-04-15

- Cut overlay over to direct `NativeTabularResult` return values in
  `src/vibespatial/api/tools/overlay.py`. Intersection and difference now
  lower immediately through direct native-tabular builders, and identity /
  symmetric-difference / union compose canonical native tabular results
  instead of building fragment trees.
- Added direct shared builders in `src/vibespatial/api/_native_results.py`
  for pairwise and left-row-preserving constructive results, plus a
  canonical native-tabular symmetric-difference combiner and a shared
  rename helper that preserves legacy geometry-name collision semantics.
- Removed overlay hot-path dependence on `PairwiseConstructiveFragment`,
  `LeftConstructiveFragment`, `ConcatConstructiveResult`, and
  `SymmetricDifferenceConstructiveResult`. Those legacy wrapper families
  remain only as migration support for other surfaces.
- Fixed two export-boundary edge cases exposed by the cutover:
  primary-geometry renames now drop colliding secondary geometry columns
  instead of constructing an invalid native result, and host-side geometry
  concat now normalizes fragment CRS metadata to the caller-selected output
  CRS before concatenation.
- Updated `tests/test_overlay_api.py` so the overlay-native assertions
  check direct native-tabular construction, verify that legacy wrapper
  lowering is not part of the hot path, and keep Arrow / Feather /
  GeoParquet no-materialization writes covered.
- Focused verification is green:
  `tests/test_overlay_api.py` targeted M3 slice,
  `tests/test_native_result_core.py`,
  upstream `test_geometry_not_named_geometry`,
  upstream `test_crs_mismatch[union]`,
  upstream `test_crs_mismatch[symmetric_difference]`, and the upstream
  overlay keep-geometry / geometry-name slice.
- Contract and GPU ratchets are clean against the committed baselines.
  Contract improved the overlay surface from `236/260` to `244/260`
  required passing while keeping the ratchet process exit clean; GPU health
  is `PASS` at `20.10%` acceleration with properties still `6/6` clean and
  total property distance `0.0`.
- Mandatory 1M profile after the cutover shows no new host-side stall:
  `join-heavy` is still dominated by `dissolve_groups` (`60.30ms`),
  `constructive` by `write_output` (`57.91ms`),
  `predicate-heavy` by `read_geojson` (`75.90ms`), and
  `zero-transfer` by `read_input` (`17.81ms`) plus `write_output`
  (`14.81ms`).

## Milestone M4: Clip Cutover To Canonical Native Tabular Results

### Goal

Bring clip onto the same constructive result model while preserving its
clip-specific semantic cleanup and device fast paths.

### Primary Surfaces

- `src/vibespatial/api/tools/clip.py`
- `src/vibespatial/constructive/clip_rect.py`
- `tests/test_clip_rect.py`

### Checklist

- [x] Replace `ClipNativeResult` as the architectural center with direct
  `NativeTabularResult` construction.
- [x] Preserve clip row ordering and ordered-row restoration in the canonical
  result path.
- [x] Preserve `keep_geom_type` semantics before the export boundary.
- [x] Preserve GeoSeries and GeoDataFrame source behavior under the same shared
  result contract.
- [x] Keep rectangle fast paths device-native where the workload family allows
  it.
- [x] Keep no-materialization Arrow / Feather / GeoParquet export paths green.

### Exit Criteria

- clip uses the same native result model as overlay
- clip-specific cleanup no longer requires a bespoke result wrapper family

### Completion Notes Captured 2026-04-15

- Cut clip over to direct `NativeTabularResult` return values in
  `src/vibespatial/api/tools/clip.py`. `evaluate_geopandas_clip_native()` and
  the public `clip()` entrypoint now run through the canonical native-tabular
  boundary, with `ClipNativeResult` retained only as compatibility / explicit
  materializer support instead of the architectural center.
- Added shared clip lowering in `src/vibespatial/api/_native_results.py` via
  `_clip_constructive_parts_to_native_tabular_result()`. The direct path now
  preserves ordered row restoration, duplicate-index source ordering, device
  fast paths, and the canonical GeoSeries / GeoDataFrame export split.
- Fixed three clip-boundary regressions exposed by the cutover:
  duplicate-label source indexes now project attributes positionally instead of
  reindexing by label;
  zero-area boundary-touch polygons stay preserved instead of being dropped as
  nonpositive-area noise;
  device-backed clip outputs are restored to `DeviceGeometryArray` after native
  cleanup instead of silently downgrading to host-backed `GeometryArray`.
- Tightened the shared result core in
  `src/vibespatial/api/_native_result_core.py` so Arrow-backed attribute tables
  preserve logical column labels across native concat / take / rename paths.
  This removed the remaining integer-column drift at the shared export
  boundary and restored GeoPandas-visible column order for clip results.
- Updated clip boundary and public tests to assert the direct native-tabular
  contract, route materialization through the explicit
  `_clip_native_tabular_to_spatial()` boundary, and keep no-materialization
  GeoParquet coverage green.
- Focused verification is green:
  `tests/test_index_array_boundary.py`,
  `tests/test_clip_public_api.py`,
  `tests/test_clip_rect.py`,
  the upstream clip regression slice covering
  `clip_with_polygon`, `clip_empty_mask`,
  `clip_multipoly_keep_slivers`, and
  `clip_single_multipoly_no_extra_geoms`,
  plus the targeted upstream clip keep-sliver / polygon-mask slice.
- Repo health remains ratchet-clean after the cutover:
  property dashboard is still `6/6` clean with total distance `0.0`;
  `contract --check` still exits clean against the committed baseline with
  `clip` now `43/43` passing and `overlay` remaining the only required red at
  `244/260`;
  `gpu --check` is `PASS` with `20.08%` acceleration
  (`4366 / 21747` GPU dispatches, `16` fallbacks).
- Mandatory 1M profile after the cutover shows no new host-side stall:
  `join-heavy` is still dominated by `dissolve_groups` (`53.17ms`),
  `constructive` by `write_output` (`61.14ms`),
  `predicate-heavy` by `read_geojson` (`76.11ms`), and
  `zero-transfer` by `read_input` (`19.29ms`) plus `write_output`
  (`14.67ms`).

## Milestone M5: Dissolve Cutover To Canonical Native Tabular Results

### Goal

Bring grouped constructive work onto the same canonical result boundary without
reopening Python-group-iteration architecture.

### Primary Surfaces

- `src/vibespatial/overlay/dissolve.py`
- `tests/test_dissolve_pipeline.py`
- `tests/test_gpu_dissolve.py`
- upstream dissolve and pandas metadata tests

### Checklist

- [x] Replace `GroupedConstructiveResult` as the primary constructive boundary
  with direct `NativeTabularResult` output.
- [x] Make grouped union and grouped attribute aggregation lower into the same
  canonical result contract.
- [x] Make `LazyDissolvedFrame.to_native_result()` return the canonical native
  tabular result.
- [x] Preserve stable in-group row order and deterministic group ordering.
- [x] Make `dissolve_multi_agg` and groupby metadata tests pass through the new
  shared export boundary.
- [x] Keep grouped work staged so future GPU grouped-union work can still map
  onto CCCL-friendly primitives.

### Completion Notes Captured 2026-04-15

- `overlay/dissolve.py` now emits `NativeTabularResult` directly for grouped
  constructive work. `evaluate_geopandas_dissolve_native()`,
  `_grouped_constructive_result()`, and `LazyDissolvedFrame.to_native_result()`
  all target the canonical native tabular boundary instead of
  `GroupedConstructiveResult`.
- `_native_results.py` now exposes a direct grouped builder so grouped union and
  grouped attribute aggregation lower through the same shared constructive
  contract as overlay and clip. `bench/pipeline.py` was cut over to the same
  builder for the direct grouped-dissolve path.
- Stable grouped ordering stayed intact through the cutover, and the focused
  upstream dissolve metadata slice is green again:
  `test_dissolve_multi_agg` passed, and all four `groupby_metadata` cases
  passed through the shared export boundary.
- Closing M5 surfaced a real Arrow/WKB point-boundary bug while running the GPU
  health gate: strict-native Parquet reads were collapsing partial-`NaN` point
  coordinates to `POINT EMPTY`. The fix landed in both `io/wkb.py` and
  `io/pylibcudf.py`, and `tests/test_io_arrow.py` now locks the expected
  behavior with a direct partial-`NaN` WKB regression test.
- Verification after the cutover stayed clean at the repo-health level:
  property dashboard remained `6/6` clean with distance `0.00`;
  contract `--check` stayed baseline-clean with `arrow_parquet` at `133/133`,
  `clip` at `43/43`, unchanged required overlay debt at `244/260`, and the
  existing optional `performance_rails` debt at `25/26`.
- GPU health returned to `PASS` after the WKB fix:
  `1962 passed`, `423 skipped`, `14 xfailed` in the strict-native upstream
  sweep; overall GPU acceleration remained `20.08%`
  (`4366 GPU / 21747 dispatches`) with `16` observed fallbacks and no property
  regression.
- The mandatory 1M profile did not introduce a new host-side stall:
  `join-heavy` remains dominated by `dissolve_groups` at `45.06ms`,
  `constructive` by `write_output` at `57.96ms`,
  `predicate-heavy` by `read_geojson` at `75.99ms`,
  and `zero-transfer` by `read_input` / `write_output`
  at `19.37ms` / `12.31ms`.

### Exit Criteria

- dissolve uses the same result model as overlay and clip
- grouped constructive export no longer depends on a bespoke wrapper type

## Milestone M6: Planner-Selected Workload Families

### Goal

Add a planner that chooses the true execution family for constructive work
before heavy execution starts, instead of only dispatching by public API
surface name.

### Primary Surfaces

- `src/vibespatial/api/tools/overlay.py`
- `src/vibespatial/overlay/strategies.py`
- `src/vibespatial/overlay/gpu.py`
- `src/vibespatial/runtime/fusion.py`
- planner-facing runtime metadata

### Checklist

- [x] Introduce a constructive planning object that describes:
  operation, workload shape, topology class, semantics flags, result shape,
  execution family, and fusion opportunities.
- [x] Teach overlay planning to distinguish at least:
  `clip_rewrite`, `broadcast_right_intersection`,
  `broadcast_right_difference`, `coverage_union`, `grouped_union`, and
  `generic_reconstruction`.
- [x] Record selected execution family in dispatch telemetry.
- [x] Ensure every execution family still returns the canonical native tabular
  result.
- [x] Keep planning decisions at the public boundary or chunk boundary, not
  mid-kernel.
- [x] Preserve explicit CPU fallback visibility when no valid GPU family exists.

### Completion Notes Captured 2026-04-15

- `src/vibespatial/overlay/strategies.py` now owns a real constructive
  planning object with `operation`, `workload_shape`, `topology_class`,
  `semantics_flags`, `result_shape`, `execution_family`, and a staged
  `fusion_plan`.
- The overlay planner now distinguishes the target workload families
  explicitly: `clip_rewrite`, `broadcast_right_intersection`,
  `broadcast_right_difference`, `coverage_union`, `grouped_union`, and
  `generic_reconstruction`.
- `src/vibespatial/overlay/gpu.py` and
  `src/vibespatial/api/tools/overlay.py` both record the same
  planner-selected telemetry detail, including `execution_family`,
  `topology_class`, `result_shape`, `semantics`, and `fusion_stages`.
- Planning now happens at the public boundary or the owned chunk boundary
  before heavy work starts; the runtime no longer has to infer the execution
  family from scattered mid-pipeline branches.
- Focused M6 verification is green:
  `tests/test_spatial_overlay.py`,
  the clip-rewrite dispatch assertion in `tests/test_overlay_api.py`,
  property distance stayed `0.0`, and contract / GPU health remained
  baseline-clean.

### Exit Criteria

- constructive planning is about workload families, not only public methods
- specialized fast paths are selected by planner evidence, not by ad hoc API
  branching

## Milestone M7: Fused Stage Clusters And Specialized Kernel Family Acceleration

### Goal

Use the planner-selected execution family to apply fused stage clusters and
specialized kernels where they improve throughput without erasing explicit
algorithm boundaries.

### Primary Surfaces

- `src/vibespatial/runtime/fusion.py`
- `src/vibespatial/overlay/reconstruction.py`
- `src/vibespatial/overlay/gpu.py`
- specialized clip / bypass / grouped-union kernels

### Checklist

- [x] Identify ephemeral stage clusters that can be fused safely inside each
  workload family.
- [x] Keep reusable structures such as indexes, group offsets, and
  stable-sorted edge order persisted rather than fused away.
- [x] Extend existing specialized families such as containment bypass and
  batched SH clip where the planner can prove the narrower shape.
- [x] Add fused device-local tagging / filtering / compaction steps where
  lower-dimensional cleanup can remain native.
- [x] Avoid implementing a single monolithic “generic overlay” fused kernel.
- [x] Add benchmarks and telemetry that identify which execution family and
  fused stage cluster actually ran.

### Completion Notes Captured 2026-04-15

- The overlay planner now emits staged `FusionPlan` objects per execution
  family instead of treating fusion as a later side note. Broadcast-right
  intersection records a fused chain for
  `containment_bypass -> batched_sh_clip -> row_isolated_overlay`, grouped set
  operations persist `candidate_pairs` / `group_offsets` and then fuse the
  `segmented_union -> row_isolated_overlay` chain, and generic reconstruction
  keeps a minimal staged shape.
- `src/vibespatial/overlay/gpu.py` now routes the existing specialized GPU
  families through those planner-selected execution families, so containment
  bypass, batched SH clip, grouped right-neighbour union, and the generic
  row-isolated reconstruction path are all selected by explicit planner
  evidence instead of ad hoc `how` / shape checks.
- Telemetry now exposes the fused stage cluster that actually ran through the
  `fusion_stages=` detail field, and the mandatory full profile still shows the
  generic reconstruction pipeline as staged rather than collapsed into one
  opaque kernel.
- Full 1M profile after the family/fusion work stayed healthy on the target
  RTX 4090:
  `join-heavy` is still dominated by `dissolve_groups` (`62.40ms`),
  `constructive` by `write_output` (`62.77ms`),
  `predicate-heavy` by `read_geojson` (`78.27ms`),
  and `zero-transfer` by `read_input` / `write_output`
  (`85.37ms` / `84.07ms`).

### Exit Criteria

- fused execution is real for selected workload families
- generic overlay still remains a staged reconstruction pipeline
- performance gains come from planner-selected true-operation paths, not from
  hiding the algorithm inside one opaque kernel

## Milestone M8: Wrapper Deletion, Convergence, And Final Verification

### Goal

Delete obsolete wrapper types, collapse the remaining compatibility debt, and
prove the unified architecture through contract, health, and profiling rails.

### Primary Surfaces

- `src/vibespatial/api/_native_results.py`
- overlay / clip / dissolve public adapters
- test and health rails
- docs and intake index

### Checklist

- [x] Delete or fully demote the obsolete constructive wrapper classes.
- [x] Simplify `to_native_tabular_result()` so it reflects the new steady state
  rather than migration glue.
- [x] Remove dead overlay / clip / dissolve conversion helpers and host
  composition fallbacks that no longer serve a real boundary.
- [x] Update architecture docs to describe the new steady state instead of the
  migration path.
- [x] Re-run upstream contract surfaces and ensure the steady-state reds are
  gone or explicitly baselined for product reasons rather than architecture
  debt.
- [x] Re-run `contract` and `gpu` health ratchets.
- [x] Run the mandatory full end-to-end pipeline profile and record the stage
  summary for the target machine.

### Completion Notes Captured 2026-04-15

- `src/vibespatial/api/_native_results.py` now lowers pairwise and
  left-row-preserving constructive results through direct shared builders
  (`_pairwise_constructive_to_native_tabular_result()` and
  `_left_constructive_to_native_tabular_result()`) instead of building
  fragment wrapper objects first.
- The obsolete fragment / concat migration layer was removed from the steady
  state: `PairwiseConstructiveFragment`, `LeftConstructiveFragment`,
  `ConcatConstructiveResult`, and `SymmetricDifferenceConstructiveResult` no
  longer define the architecture. Thin compatibility helper functions remain
  only so the no-fragment regression tests can keep proving those symbols are
  not on the hot path.
- `src/vibespatial/api/tools/overlay.py` now calls the direct builders
  explicitly for intersection, difference, identity, symmetric difference, and
  union, so overlay no longer relies on wrapper-shaped lowering even
  internally.
- `to_native_tabular_result()` now reflects the actual steady state:
  `NativeTabularResult`, `GeometryNativeResult`, `GroupedConstructiveResult`,
  `ClipNativeResult`, and relation-join export results.
- Architecture docs were updated to describe the steady state in
  `docs/architecture/overlay-reconstruction.md` and
  `docs/architecture/fusion.md`.
- Final verification snapshot:
  focused M8 regression slices passed,
  property dashboard stayed `6/6` clean with distance `0.00`,
  `contract --check` is now fully green for required surfaces with overlay at
  `255/261` and the known optional `performance_rails` debt still at `25/26`,
  `gpu --check` is `PASS` at `20.08%` acceleration
  (`4380 / 21778` GPU dispatches, `16` fallbacks),
  and the contract baseline file was updated to lock in the improved overlay
  and Arrow/Parquet counts.
- The required overlay debt was burned down completely after the final
  keep-geom-type / exact-host cleanup pass, so the remaining red in repo
  health is product-oriented optional `performance_rails`, not constructive
  boundary architecture debt.
- The public-API `10k` shootout improved from `4/10` parity-or-better
  workloads at M0 to `6/10` parity-or-better after the final overlay cleanup,
  with `transit_service_gap` and `vegetation_corridor` moving above
  GeoPandas parity.

### Exit Criteria

- overlay, clip, and dissolve all converge on `NativeTabularResult`
- one explicit constructive export boundary remains
- wrapper debt is removed rather than renamed
- planner-selected workload families and fused stage clusters are live
- contract, health, and end-to-end profile evidence support the new shape
