# Native Tabular Execution Ledger

<!-- DOC_HEADER:START
Scope: Producer-to-consumer ledger for Native tabular, relation, rowset, grouped, attribute, and index execution boundaries.
Read If: You are removing pandas-shaped work from a Native workflow, choosing pylibcudf versus CCCL, or classifying an internal host conversion.
STOP IF: You only need an SF100 query record or geometry-kernel physical-shape detail.
Source Of Truth: Current disposition of private tabular execution boundaries under Native carriers.
Body Budget: 220/240 lines
Document: docs/dev/native-tabular-execution-ledger.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-9 | Intent |
| 10-18 | Request Signals |
| 19-26 | Open First |
| 27-34 | Verify |
| 35-43 | Risks |
| 44-52 | Rules |
| 53-68 | Ordered Ledger |
| 69-77 | Profile Reconciliation |
| 78-121 | Core 10K Boundary Classification |
| 122-159 | External-Corpus Boundary Classification |
| 160-205 | M3 Acceptance Evidence |
| 206-220 | Handoff |
DOC_HEADER:END -->

## Intent

Keep pylibcudf beneath Native carriers and prevent relation pairs, row maps,
attributes, keys, and index labels from becoming pandas-shaped intermediate
work. This is the M2 evidence ledger for
`docs/dev/native-consolidation-execution-plan.md`.

## Request Signals

- Native tabular execution
- pylibcudf consumer
- relation sort or dedup
- pair or row-position export
- index assembly
- internal host conversion

## Open First

- `docs/dev/native-consolidation-execution-plan.md`
- `docs/decisions/0044-private-native-execution-substrate.md`
- `docs/decisions/0046-gpu-physical-workload-shape-contracts.md`
- `docs/dev/native-physical-shape-ledger.md`
- `docs/dev/pylibcudf-sf100-query-ledger.md`

## Verify

- `uv run pytest tests/test_private_native_substrate.py -q`
- `uv run pytest tests/test_index_array_boundary.py -q`
- `uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline`
- run CMAB and Power at 10K, repeat three, with ordered-v3 fingerprints and
  statement profiles before changing a status to complete

## Risks

- Attaching transient source state globally can create stale attribute lineage.
- A device sort is not a win when public pair or row-map export still dominates.
- Treating terminal index construction as compute debt can overcomplicate the
  public compatibility boundary without improving wall time.
- General dtype admission without null/order tests can silently change pandas
  semantics.

## Rules

- Public pandas, GeoPandas, Arrow, and index construction is terminal only.
- Stable order, null equality, NaN equality, duplicate policy, suffixes, and
  source-index labels are semantic inputs, not cleanup after compute.
- An eight-byte scalar fence may be admitted; a row-count-sized host array may
  not be mislabeled as a fence.
- Lower counters do not close a row unless end-to-end wall time also improves.

## Ordered Ledger

| Priority and boundary | Shape and contract | Starting backend/evidence | Selected primitive or fence | Status and canary |
|---|---|---|---|---|
| 1. Public `sjoin`/`sjoin_nearest` source frames -> joined attributes | `NativeRelation(P)` gathers two source tables; nullable/string columns are movement-only; suffixes, `on_attribute`, stable pair order, distance, and exact left index labels must survive | CMAB 10K started at 2,058,428 pairs, 0.166 s join, 0.762 s pandas sort/dedup, and 33.88 MiB D2H. Power 10K started at 4,870 pairs and 0.040 s pandas sort/dedup. Neither ordinary source had an attached frame state. | Transient source lowering from existing owned geometry, `NativeIndexPlan`, and one Arrow-to-pylibcudf attribute conversion; device relation gather returns `NativeFrameState`. Explicit public index-column access and terminal public index assembly may materialize; internal consumers do not. | **Accepted at 10K.** Public `on_attribute`, suffix, nullable string/Int64, named RangeIndex, NumPy-compatible right-index dtype, distance, and no-pair-export canaries are green. CMAB and Power preserve ordered-v3 fingerprints. |
| 2. Joined frame -> stable sort -> first/last/unique per key | Stable equal rows, direction-independent null/NaN placement, `NullEquality.EQUAL`, and `keep=first/last/False` are mandatory. | Recursive device-part resolution avoids Arrow/host combine. Sort/distinct now admit memory before keys, order, gather, or workspace allocate. | `stable_sorted_order`, NaN-to-null normalization, `distinct_indices`, then device gather into `NativeRowSet`. | **Accepted for explicit device dtype policies.** Nullable numeric, string, categorical, decimal, and temporal canaries cover stable order, all keep modes, null/NaN equality, zero D2H/materialization, and observable decline. Pure null and unsupported types fail closed. |
| 2b. Device frame -> bounded top-k | Lexicographic order, missing-last, signed zero, duplicate labels, and pandas `keep=first/last/all` tie order must match without a full-table sort under primary-key skew. | The old primary-threshold candidate could retain and sort all rows; row-indirected keys also gathered before admission. | Admit before recursive key gather; iteratively compact strict winners and refine only the boundary-equal span; stable-sort selected/output rows through one `NativeRowSet`. | **Accepted for fixed-width pandas top-k dtypes.** Multipart, nullable integer/bool/timestamp, float NaN/inf/signed-zero, empty/full, strict decline, and exact multi-key tie canaries pass. The exact top-100 crossover is 0.399x pandas at 10K, 0.840x at 100K, and 4.988x at 1M. |
| 3. Selected frame -> GeoParquet/Arrow writer | Preserve the device selection map, nullable attributes, and exact source index labels through terminal write; no host row-position normalization. | Initial CMAB exported a 2,058,428-row writer map; after relation shaping it still exported 10,000 positions/80,000 bytes through `NativeIndexPlan.take_public_index`. | Preserve the device index plan beside the exact public index shell; resolve composed device attribute parts recursively. GeoParquet writes attributes and final index labels through pylibcudf; Arrow copies only final labels at its terminal user-export boundary. | **Accepted at CMAB 10K.** Chained join -> sort -> dedup -> Arrow/Parquet canaries are exact. Final GeoParquet writer stage has zero D2H, materialization, or fallback; no pair/index/row-position normalization occurs. |
| 4. `NativeDeviceSelection` -> native consumer or eager public frame | Capacity-backed positions plus one dynamic logical count; native consumers do not require Python length. | Final core profile records 24 `NativeTabularSelection` and five `query_any` logical-count reads. Each is eight bytes. | Retain `NativeDeviceSelection` across native consumers. One named scalar count is admitted only when an eager public object requires length. | **Accepted as bounded.** Final core evidence has 29 eight-byte fences and no row-sized selection export. |
| 5. Geometry composition -> concrete singular proof | Prove that each logical row maps to one concrete owned row without rebuilding geometry or exporting row maps. | Final core profile records 15 two-byte singular-owned and 11 one-byte partitioned-composition proof packets. | Propagate trusted singular/coverage invariants when producers already prove them; otherwise retain the compact packet. | **Accepted as bounded.** Proof packets are constant-size metadata, not tabular compute debt. |
| 6. Clip/overlay terminal source-row projection | Device `source_rows` select source attributes and index labels for exact public output. | Final core profile has one 764-row/6,112-byte export for a source whose attributes and index are already public host data. | Preserve device row indirection for native attributes; permit the host row map only while assembling an eager public clip result from host-public attributes. | **Classified terminal for the observed shape.** Device-attribute canaries avoid this export; eliminating the host-public case requires broader transient lowering. |
| 7. Dissolve/grouped labels -> output index | Group keys preserve `sort`, `dropna`, observed categories, and label dtype; nullable values use pandas `first`/`last` skip-null semantics and retain empty groups. | Final core profile has one four-label/16-byte terminal label export. | Pylibcudf `nth_element(EXCLUDE)` reduces nullable string/categorical/temporal values, then a nullable device gather restores dense group order; mixed numeric and take reducers combine as device tables. | **Accepted for first/last breadth.** Sparse, all-null, empty, row-indirected, mixed-reducer, strict-native, dtype/index, zero-D2H, and admission-decline canaries are green. Broader nullable reductions remain open. |
| 8. Device index labels -> public/Arrow index | Exact name, dtype, duplicate labels, RangeIndex semantics, and MultiIndex levels/codes. | `NativeIndexPlan.to_public_index` may export device labels; device writer index-column helpers cover admitted Arrow types. | Keep writer-compatible labels device-native; otherwise classify the public index or Arrow metadata construction as terminal. | **Admitted terminal boundary with breadth gaps.** Never infer uniqueness from labels. |
| 9. Device frame -> restricted public query/eval | Exact signed-int64 and bool comparisons and signed-int64 `+`, `-`, `*` assignment; quoted/commented `@`, backticks, missing names, and pandas syntax errors retain public semantics. | The public wrappers capture caller scope, lower a bounded AST, include recursive row-view gather in admission, and avoid duplicate fallback events. | Pylibcudf expressions produce `NativeDeviceSelection`/`NativeRowSet`; eval returns a device column in `NativeFrameState`. Unsupported valid expressions decline observably and strict mode rejects them. | **Accepted for the restricted grammar.** Exact values above `2**53`, metadata, attrs, admission, strict decline, invalid-expression, and cross-stream canaries pass. Query is 0.386x/0.710x/4.230x pandas at 10K/100K/1M. |
| 10. Device frame + bounded dimension -> merge/join | Stable inner same-name equality merge with unique right keys; exact aligned unique-index left join; pandas suffix, cardinality, null, order, index, attrs, and `columns.name` semantics. | Public validation is classified before GPU work. Admission covers carriers, row maps, gather, output metadata, and right H2D. Row-indirection readiness follows projected/renamed tables across streams. | Pylibcudf equality join plus output-shaped device gathers returns `NativeFrameState`; exact aligned index join avoids label export. Array/nested keys, geometry-bearing right frames, and broader join shapes decline observably. | **Accepted for the bounded shape.** Nullable string, cardinality, invalid-argument, metadata, strict, admission, and actual two-stream query-to-merge canaries pass. Merge is 0.521x/1.109x/6.903x pandas at 10K/100K/1M. |

## Profile Reconciliation

The starting core 10K statement profile reported 69 internal host conversions,
179 user exports, and 631 D2H events. The final 14-workflow rerun now records 66
internal conversions, all reconciled below: 29 logical-count fences, 26
composition proofs, nine public index exports, one clip source-row export, and
one dissolve label export. External profiles remain a separate inventory, and
user-requested exports remain separate from compute debt.

## Core 10K Boundary Classification

The current 14-workflow core profile (`core-10k-query-merge-profile.json`,
SHA256 `b4d9bd67d61488470c2773a93c86ef85e96b7c4ee51f30969f936c55bce30efb`)
records exactly 66
`internal-host-conversion` events. Every event is D2H and admitted at the public
boundary (`strict_disallowed=False`). The table accounts for every event; `S<n>`
is the statement index within the named workflow.

| Operation and reason | Exact stages | Count and bytes | Disposition |
|---|---|---|---|
| `device_selection_logical_count_to_host`; explicit public compaction/export | accessibility redevelopment S11/S12/S23/S28; corridor flood priority S8/S10/S11; emergency response catchments S4/S16; habitat corridor compliance S5/S12; insurance flood screening S3/S9; network service area S5; parcel zoning S6/S10; redevelopment screening S8/S10/S15; site suitability S7/S10; transit service gap S4/S8; vegetation corridor S8 | 24 events, 8 B each, 192 B | **Named bounded fence.** Eager public result length only. |
| `device_selection_logical_count_to_host`; spatial-index boolean-filter compaction | redevelopment screening S13; retail trade area screening S15/S17; site suitability S13; transit service gap S7 | 5 events, 8 B each, 40 B | **Named bounded fence.** One scalar logical count per public filter. |
| `singular_partitioned_certification`; terminal composition multiplicity proof | accessibility redevelopment S11/S12; corridor flood priority S8; emergency response catchments S4; habitat corridor compliance S5; insurance flood screening S3; network service area S5; parcel zoning S6; redevelopment screening S8; site suitability S7; transit service gap S4 | 11 events, 1 B each, 11 B | **Named bounded fence.** One boolean proof packet, independent of rows/parts. |
| `singular_owned_certification`; terminal composition multiplicity/coverage proof | accessibility redevelopment S16/S23/S28; corridor flood priority S10; emergency response catchments S16; habitat corridor compliance S12/S14/S15; insurance flood screening S5/S9; parcel zoning S9; redevelopment screening S10; site suitability S10; transit service gap S7/S8 | 15 events, 2 B each, 30 B | **Named bounded fence.** Two-boolean proof packet, independent of rows/parts. |
| `index_plan_to_host`; device public index labels materialized by `NativeTabularResult.to_geodataframe` | accessibility redevelopment S12: 764 rows/6,112 B; emergency response catchments S4: 700/5,600 B; habitat corridor compliance S5: 700/5,600 B and S12: 686/5,488 B; insurance flood screening S3: 700/5,600 B; parcel zoning S6: 3,844/30,752 B; redevelopment screening S8: 5,184/41,472 B; site suitability S7: 3,844/30,752 B; transit service gap S4: 700/5,600 B | 9 events, 136,976 B | **Terminal public export.** Exact ordinary pandas index construction for public clip results. |
| `clip_terminal_source_rows_to_host`; clip source attribute rows | accessibility redevelopment S11: 764 rows/6,112 B | 1 event, 6,112 B | **Terminal public export.** Positions select already-public host attributes and source index while assembling the eager clip result. |
| `device_group_key_labels_to_host`; dissolve labels for public output index | redevelopment screening S15: 4 groups/16 B | 1 event, 16 B | **Terminal public export.** Exact single-level group labels become the eager dissolve index. |

The reconciliation is 55 bounded events/273 B plus 11 terminal events/143,104
B, for 66 events/143,377 B. No event in this materialization class is
unbounded compute debt. This does not classify unrelated runtime transfers that
have no `internal-host-conversion` event.

No small coherent M4 elimination is warranted from this artifact. The nine
index events are the explicit `lazy_public_index=False` clip compatibility
boundary; avoiding them would expose `NativeIndex` extension dtype semantics to
ordinary public results. The clip row export is the row map required to slice
host-public attributes and labels; removing it requires full transient device
attribute lowering, not a local row-map change. Dissolve's device labels feed a
`NativeGrouped` contract whose reducers currently require a concrete pandas
output index; changing that correctly also requires categorical, nullable, and
MultiIndex breadth. Trusted singular-proof propagation may remove some 1-2 B
fences later, but it spans many producers and is not justified as a narrow fix.

The current six-workflow external reruns at 10K, 100K, and requested 1M each
record exactly three internal conversions: CMAB and Power terminal public join
indices plus one 8-byte OSM explode count fence. All 18 contracts pass with
zero fallback. The 627-row WKT compatibility control explicitly selects its
host dissolve plan and returns three rows; it has no internal conversion. The
only >1 s synchronized stage is deferred GPU multipart-union topology realized
at OSM enrichment export, already classified as device compute rather than
pandas composition. M4 therefore has no unexplained current host boundary.

## External-Corpus Boundary Classification

The current CMAB 10K profile
(`/tmp/cmab-flat-sindex-auto-10k-repeat3.json`) contains the following complete
transfer/materialization inventory. Rows that combine a trace copy and a
materialization event describe the same boundary, not two byte charges.

| Stage | Observed reason | Events and bytes | Disposition |
|---|---|---|---|
| Statement 3, spatial join | `spatial index regular-grid summary scalar fence` | 1 D2H, 8 items, 64 B | **Named bounded fence.** Fixed-size grid summary only. |
| Statement 3, spatial join | `flat spatial index device total-bounds scalar fence` | 1 D2H, 5 items, 40 B | **Named bounded fence.** Fixed-size extent packet for the row-indirected device index. |
| Statement 3, spatial join | `device spatial-index tree extent planning fence` | 1 D2H, 3 items, 24 B | **Named bounded fence.** Fixed-size tree extent only. |
| Statement 3, spatial join | `device spatial-index refined-pair allocation fence` | 1 D2H, 2 items, 16 B | **Named bounded fence.** Allocation cardinality only. |
| Statement 3, spatial join | `relation_join_public_index_to_host`; terminal public relation join requires exact pandas index semantics | 1 D2H plus 1 materialization, 2,058,428 labels, 16,467,424 B | **Terminal public export.** Exact eager public `sjoin` index construction. |
| Statement 5, GeoParquet write | native GeoDataFrame exported to GeoParquet writer | 1 materialization, 590,004 B logical payload, 0 D2H | **Terminal public export.** Device-native pylibcudf writer boundary. |

These rows reconcile to five runtime D2H copies and 16,467,568 B, two
materialization events, and no H2D copy or fallback. Named fences are 144 B;
the accepted public-index export is 16,467,424 B. Partial device slicing, flat
spatial indexing, stable sort/dedup, and the writer contribute zero D2H.

The final Power 10K row-indirection profile is
`/tmp/power-slice-indirection-10k-repeat3.json`. Its fail-closed
`power-nearest-v1` contract passes after the nearest tie-membership, writer
row-indirection, and device geometry-slice fixes.

| Stage | Observed reason | Events and bytes | Disposition |
|---|---|---|---|
| Statements 2-3, public `iloc` slices | Device row-indirected geometry and flat spatial-index construction | Zero D2H, H2D, or materialization in both stages | **Accepted native composition.** Slices retain source row provenance and build bounds/order from device-resident rows. |
| Statement 4, nearest join | `nearest sorted-x tie-pair allocation fence` | 1 D2H, 2 items, 8 B | **Named bounded fence.** Tie-pair allocation cardinality only. |
| Statement 4, nearest join | `relation_join_public_index_to_host`; terminal public relation join requires exact pandas index semantics | 1 D2H plus 1 materialization, 4,870 labels, 38,960 B | **Terminal public export.** Exact eager public `sjoin_nearest` index construction. |

Power therefore records two runtime D2H copies and 38,968 B, zero H2D, zero
fallback, and no slice or writer transfer. The bytes are one 38,960 B terminal
public-index export and one 8 B allocation fence. Stable sort/dedup, both
geometry slices, flat spatial-index construction, and the device GeoParquet
writer contribute zero D2H or materialization.

## M3 Acceptance Evidence

The August 27, 2026 repeat-three runs used validated GeoPandas comparators and
ordered-v3 fingerprints:

- CMAB 10K (`/tmp/native-cmab-multipart-fix-final.json`) is exact at 0.0976 s
  versus 0.6776 s, or 6.94x. Its statement profile records 0.1206 s join and
  0.0170 s GPU stable sort/dedup with zero D2H/materializations. Total D2H fell
  from 171,719,616 to 17,337,512 bytes; the remaining large transfer is the
  terminal 16,467,424-byte public left index, not relation or attribute shaping.
- The flat-index rerun (`/tmp/cmab-flat-sindex-auto-10k-repeat3.json`) is exact
  at 0.0764 s versus 0.6959 s, or 9.10x in the artifact. Partial device slices,
  flat index construction, stable sort/dedup, and the pylibcudf GeoParquet
  writer have zero D2H/materialization. Total profile D2H is 16,467,568 bytes,
  all but 144 bytes of which is the terminal public join index.
- Final Power 10K in `/tmp/power-slice-indirection-10k-repeat3.json` passes
  `power-nearest-v1` at 0.0598 s versus 0.0425 s, or 0.711x. Its two runtime
  D2H copies total 38,968 B: a 38,960 B terminal public-index export and one
  8 B allocation fence. Both `iloc` stages and the writer have zero transfer or
  materialization.
- An exact same-input shaping microbenchmark puts the crossover in the tens of
  thousands: at 4,870 relation rows native sort/dedup is 8.31 ms versus pandas
  1.99 ms; at 49,678 rows it is 12.90 ms versus 16.01 ms. Small Power work is a
  legitimate CPU-fast case; do not tune the device shape merely to force a 10K
  win.
- The current-source restricted query and bounded unique-right merge probe is
  exact at 10K, 100K, and 1M. Query is 0.386x, 0.710x, and 4.230x pandas;
  merge is 0.521x, 1.109x, and 6.903x. The CPU legitimately wins the smallest
  work. Merge crosses by 100K and both device-native paths win materially at
  1M. This is crossover evidence, not an accepted shootout baseline.

The nearest tie-membership defect found by the 20K/100K probes is fixed in both
the sorted-X and grid paths. Direct comparisons now match all selected IDs,
index labels, attributes, row order, and row counts at 20K, 100K, and the full
corpus. The fail-closed `power-nearest-v1` verifier also requires exact column
order, logical dtypes, index type/names/dtypes, and CRS. Only projected point
coordinates and `distance_m` are numeric: coordinate tolerance is two fp64
ULPs at the EPSG:3857 world-coordinate bound (7.450580596923828e-09 m), and
distance tolerance is `2*sqrt(2)*coordinate_atol` plus two fp64 ULPs at the
100,000 m search bound (2.110252808590375e-08 m). Tests reject the first value
beyond either bound. The global ordered-v3 mismatch remains visible because it
uses a broader generic numeric policy; no Power timing is comparable unless the
workload-specific contract matches. The final 100K corpus run is exact at
0.1408 s versus 0.3129 s (2.22x), and the requested-1M run is exact at 0.2232 s
versus 0.6405 s (2.87x).

## Handoff

Transient relation-source lowering, multipart pylibcudf sort/dedup, device
geometry slicing, flat spatial indexing, writer row indirection, and all 66
final core internal conversions are classified. No narrow M4 code change is
justified by the remaining core events. Nullable distinct, grouped first/last,
bounded top-k, restricted query/eval, and bounded merge/join now have explicit
policy, admission, null/order, validation, stream-readiness, and public
composition contracts. The final same-source audit
(`full-pipeline-final-reviewed-source.json`, SHA256
`a3553cb2c3be3a32e2a77230db0c9932fb0721eb3135e61efd2567e7c4a879c2`)
passes 22 active 100K/1M lanes with zero fallback or compute materialization;
40 bounded planning packets total 41,056 bytes and the max 1M stage is 73.70 ms.
Landing-tree SF100 passes 12/12 at 18.955x. M4 and local M5 evidence are
complete; clean-revision delivery evidence remains.
