# Issue 11: Bounded Exact Fixed-k Nearest Tracker

**Source:** `reports/issue-11`
**Status:** Complete; implementation landed on `origin/main`
**Owner:** Codex
**Machine constraint:** Do not run SF1000 on this machine. Use SF1, SF10, and
SF100 only when required by the applicable correctness or performance gate.

## Objective

Replace the current full-candidate fixed-`k` nearest GPU plan with an exact,
capacity-bounded implementation that reuses native spatial-index state, keeps
progressive search state on device, returns a `NativeRelation`, and never
silently executes CPU work inside the supported GPU path.

## Physical Shape Contract

- **Public contract:** `SpatialIndex.nearest(..., k=N)` for fixed positive `k`,
  `return_all=False`, and `exclusive=False`; bounded `max_distance` is exact.
- **Physical shape:** capacity-bounded candidate-refine with segmented fixed-k
  reduction and progressive indexed radius expansion.
- **Work units:** active query rows, query tiles, index ranges, candidate pairs,
  retained relation rows, distance evaluations, output rows, and temporary
  bytes.
- **Native inputs:** `NativeSpatialIndex`, query `OwnedGeometryArray`, and a
  device-resident active-row selection compatible with `NativeRowSet`.
- **Native output:** `NativeRelation` with at most `valid_query_count * k`
  rows and optional fp64 distances.
- **Staging layout:** bounded COO candidate buffers, dense count/offset arrays,
  fixed per-query retained top-k state, active-row masks/positions, and a
  reusable scratch workspace.
- **Export boundary:** only the existing public nearest return boundary may
  materialize host index/distance arrays.
- **Precision:** fp64/conservatively outward bounds; METRIC `PrecisionPlan`
  for distance calculation; ambiguous ranking/finalization decisions refined
  to fp64.
- **Saturation:** query tiling for many-query workloads and resumable
  target/index-range streaming when one query's fanout exceeds pair capacity.

## Design Decision

Two structurally different solutions were evaluated:

1. Progressive radius search over the retained flat/native index with bounded
   query and target/index-range streaming.
2. A new hierarchical best-first nearest traversal with per-query priority
   queues.

The implementation will start with option 1 because it reuses the current
`NativeSpatialIndex`, candidate query, distance strategies, and native relation
substrate while still bounding peak memory independently of query-shard
cardinality. Query-only tiling is explicitly insufficient: dense single-query
fanout must stream target/index ranges or decline deterministically before
allocation. If the mandatory profile shows repeated index traversal has the
wrong physical performance shape, stop and replace it with option 2 rather
than tuning around the problem.

## Tracker

### 1. Contract and regression harness

- [x] Record the public semantics and ADR-0046 physical workload shape.
- [x] Record the SF1/SF10/SF100-only machine constraint.
- [x] Add a direct fixed-`k` GPU test module with a brute-force Shapely oracle.
- [x] Add deterministic ordering tests using `(distance, target_row)`.
- [x] Add empty, missing, degenerate, and `k > valid_target_count` cases.
- [x] Add bounded and unbounded distance cases.
- [x] Add multiple query-tile and multiple radius-expansion cases.
- [x] Add a dense single-query case that exceeds one candidate tile.
- [x] Add large-offset and near-kth precision cases for fp32/fp64 plans.
- [x] Add explicit rejection/observable decline tests for `return_all=True` and
  `exclusive=True`.

### 2. Native dispatch and index reuse

- [x] Pass the cached `NativeSpatialIndex` through `nearest_relation()`.
- [x] Preserve active query positions on device across progressive iterations.
- [x] Reuse the dispatch-owned `PrecisionPlan` rather than recomputing it.
- [x] Keep the supported GPU result in `NativeRelation` until public export.
- [x] Prove the flat/native index is built once and reused by the fixed-k path.
- [x] Ensure unsupported semantics produce one observable fallback/decline
  event and strict-native failure, without silent CPU execution.

### 3. Capacity-bounded candidate workspace

- [x] Define a stage-specific workspace model covering retained results,
  candidate indices, distances, counts/offsets, reorder/reduction buffers, and
  CCCL/NVRTC scratch simultaneously.
- [x] Use live `DeviceMemoryAdmission` to choose pair and query capacity.
- [x] Produce candidate segments with dense counts/offsets retained on device.
- [x] Support inner/outer search radii so repeated expansions do not recompute
  exact distances for previously visited pairs.
- [x] Stream target/index ranges when one query can exceed pair capacity.
- [x] Reuse capacity-sized buffers across tiles and radius iterations.
- [x] Prove no allocation is sized from full `query_count * target_count`.
- [x] Raise an actionable capacity error before allocation when even the
  minimum valid workspace cannot be admitted.

### 4. Exact retained top-k

- [x] Evaluate CCCL segmented lexicographic reduction against a custom bounded
  merge: existing ADR-0033 primitive evidence plus the 35.19-second exact SF100
  Q12 run favored CCCL; the profile did not justify a new custom kernel.
- [x] Implement deterministic total ordering by `(distance, target_row)`.
- [x] Merge each candidate chunk into at most `k` retained rows per query.
- [x] Prevent duplicate `(query, target)` entries across radius expansions.
- [x] Preserve valid counts when fewer than `k` targets are available.
- [x] Keep the compact output relation bounded by `valid_query_count * k`.

### 5. Progressive scheduler and exactness

- [x] Seed initial radii from retained index extent/density metadata.
- [x] Search only unresolved query rows after each iteration.
- [x] Finalize only when the exact/refined kth distance is within the searched
  radius.
- [x] Double radii only for unresolved rows.
- [x] Terminate at the global extent when fewer than `k` valid targets exist.
- [x] Execute bounded `max_distance` requests in one pass at the ceiling.
- [x] Keep active masks, radii, counts, and retained state device-resident.
- [x] Limit loop control to bounded scalar transfers; prohibit array-sized
  D2H/H2D traffic inside the engine loop.

### 6. Precision, warmup, and telemetry

- [x] Keep candidate-producing bounds fp64 or conservatively outward-rounded.
- [x] Apply the selected METRIC precision plan to distance computation.
- [x] Refine ambiguous kth-boundary distances in fp64 before ordering or
  finalization.
- [x] Register every required CCCL/NVRTC warmup specialization.
- [x] Ensure warm execution emits no unknown-specialization warning.
- [x] Record admitted capacity, tile count, radius iterations, candidate
  counts, workspace peak, allocations, transfers, and materializations.
- [x] Add canaries for allocation count and bulk host-transfer regressions.

### 7. Verification

- [x] Run targeted fixed-k, spatial-index, native-carrier, precision,
  fallback, and strict-native tests.
- [x] Run the upstream GeoPandas spatial-index contract tests.
- [x] Run `uv run ruff check`.
- [x] Run `uv run python scripts/check_docs.py --check`.
- [x] Run deterministic repository checks required by the pre-commit hook.
- [x] Run the mandatory full end-to-end profile with GPU sparkline output.
- [x] Review every profile stage and resolve unexplained CPU-heavy work.
- [x] Validate exact ordered output at SF1.
- [x] Skip SF10 because SF1 and SF100 both passed without an intermediate
  scaling diagnostic.
- [x] Validate exact ordered output at SF100.
- [x] Do not run SF1000 on this machine.
- [x] Record benchmark identity, correctness fingerprints, fallbacks,
  transfers, capacity failures, and stage timings.

### 8. Review and landing

- [x] Run the CUDA optimizer review on the completed implementation.
- [x] Run precision-compliance review and update its ledger if applicable.
- [x] Run GPU code review and resolve every blocking finding.
- [x] Run the mandatory pre-land review through the commit workflow.
- [x] Mark every completed tracker item and record any intentionally
  non-applicable item with evidence.
- [x] Commit the reviewed diff.
- [x] Run `git pull --rebase` and push successfully.

## Completion Criteria

- [x] Supported public fixed-k nearest returns exact, deterministically ordered
  results.
- [x] Peak candidate storage is bounded by admitted workspace capacity rather
  than full shard cardinality.
- [x] No benchmark name, dataset scale, or device identity influences engine
  selection.
- [x] Unsupported semantics decline observably and never silently execute CPU
  work in strict-native mode.
- [x] Native index/query/relation carriers remain device-resident through the
  engine pipeline.
- [x] SF1 and SF100 match authoritative exact answers; SF10 is used only when
  needed; SF1000 is not run on this machine.
- [x] Telemetry explains capacity, iterations, candidates, allocation peak,
  transfers, and materialization boundaries.
- [x] Cold and warm paths have registered specializations and no unknown
  warmup warning.
- [x] The reviewed commit is present on the remote branch.

## Evidence Log

| Date | Evidence | Result |
|---|---|---|
| 2026-08-31 | Initial source investigation and RTX 4090 probe | Confirmed full-candidate physical shape, lost native-index reuse, and incorrect `exclusive=True` row-identity filtering |
| 2026-08-31 | `tests/test_spatial_index_knn_device.py` | 14 passed; covers oracle correctness, deterministic ties, bounded/unbounded search, missing/empty rows, forced target streaming, constrained memory, cache reuse, precision refinement, and observable declines |
| 2026-08-31 | Spatial/runtime/native regression selection | 860 passed, 1 optional SciPy skip; generic candidate output and deferred-free changes remained green |
| 2026-08-31 | Upstream GeoPandas spatial-index contract | 221 passed, 59 optional/version skips |
| 2026-08-31 | CCCL precompile/primitives | 85 passed; `segmented_sort_asc_i32` is registered and warm execution emitted no unknown-spec warning |
| 2026-09-01 | [Tracked profile evidence](issue-11-profile-evidence.md), source digest `f048d61c37e358a38b7b609fc768cda766dcabd5d70896d1015b8381d68e664e` | Complete identity and all 51 active 1M stage names/times recorded; zero fallbacks, zero compute materializations, 20,528 bytes compute D2H; slowest stage 73.21 ms; no stage exceeded 1 second |
| 2026-09-01 | Nearest-relation producer profile | Exact-source 1M canary: build 27.46 ms, native distance consume 1.37 ms, attribute filter 0.29 ms, right relation 2.57 ms; zero fallback, zero compute materialization, 16 bytes compute D2H |
| 2026-08-31 | SF1 Q12, strict native | 100 rows in 14.06 s; ordered key SHA-256 `9c9708d3...`; exact key/order match to independent GeoPandas and max absolute numeric delta `2.84e-14` |
| 2026-09-01 | SF100 Q12, strict native, exact source | 100 rows in 34.48 s; result SHA-256 `4c9b9f08cf2a800d64412a4c0444162a81a088286e677963662844e0ac52d00d`; exact key/order match to frozen GeoPandas oracle and max absolute numeric delta `1.42e-14` |
| 2026-09-01 | SF100 Q12 telemetry | Zero fallback, 11.89 GB peak VRAM, 6.01 GB operation-local RMM peak, 46,602 RMM allocations, 7.66 MB tracked D2H; no SF10 diagnostic or SF1000 run was needed/performed |
| 2026-09-01 | Full package suite | 7,883 passed; the 11 task-related kNN/canary failures were corrected; 10 remaining upstream Arrow/CRS failures are outside the diff and reflect optional dependency behavior |
| 2026-09-01 | Final issue-focused regression matrix | 897 passed, 1 optional SciPy skip after telemetry-finalization remediation |
| 2026-09-01 | Pre-land deterministic checks | Ruff, docs, architecture, zero-copy, performance-pattern, maintainability, and import-guard checks all passed with zero baseline violations |
| 2026-09-01 | First independent pre-land review | Fix required: one-target candidate workspace bypass, process-historical memory peak, and incomplete tracked profile identity/stage evidence |
| 2026-09-01 | Review remediation | One-target output now preserves supplied workspace pointers; fixed-k telemetry uses a nested operation-local RMM scope; tracked evidence binds complete identities and every active 1M stage; focused remediation tests passed |
| 2026-09-01 | Second independent pre-land review | Original three findings resolved; fix required for fp32 false negatives at the `max_distance` threshold and missing nonpositive public-distance validation |
| 2026-09-01 | Exactness/API remediation | Staged plans now refine threshold-ambiguous distances before exact filtering; public nearest rejects zero/negative `max_distance`; concrete GPU reproduction and validation tests passed |
| 2026-09-01 | Third independent pre-land review | Prior five findings resolved; fix required for finite fp64 distances that overflow to non-finite fp32 and for two stale raw-artifact command paths |
| 2026-09-01 | Overflow/evidence remediation | Non-finite coarse distances now fail closed into fp64 refinement; the `1e39` GPU reproduction passes; tracked commands name the then-current hash-bound artifacts |
| 2026-09-01 | Fourth independent pre-land review | All code findings resolved; fix required only for stale SF100 allocation count and `final2` labels in this tracker |
| 2026-09-01 | Evidence consistency remediation | Tracker matches the hash-bound evidence: 46,602 SF100 allocations and exact artifact labels |
| 2026-09-01 | Fifth independent pre-land review | Fix required for omitted terminating loop-control fences and early-return D2H telemetry finalization |
| 2026-09-01 | Telemetry finalization remediation | Transfer/materialization deltas now finalize in the operation wrapper on every result path, and every loop-control fence is counted |
| 2026-09-01 | Sixth independent pre-land review | LAND with no findings; all prior findings, final4 artifacts, exact source identity, randomized oracles, and focused GPU tests verified |
| 2026-09-01 | Reviewed implementation commit | `94b17da` (`Implement bounded exact fixed-k nearest`) created after the mandatory LAND verdict |
| 2026-09-01 | Remote landing | `git pull --rebase && git push` completed successfully; `origin/main` advanced from `bc022fe` to `94b17da` after the contract and GPU health pre-push gate refreshed its cache |
