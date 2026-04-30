# Native Format Inventory

<!-- DOC_HEADER:START
Scope: Inventory of GPU-capable library surfaces mapped to current and target Native* carriers.
Read If: You are choosing or updating a Native* adoption slice, classifying host transfers, or adding carrier canary coverage.
STOP IF: You only need the high-level program plan or one operation-local implementation detail.
Source Of Truth: Phase-0 inventory for Native* carrier adoption across GPU-capable vibeSpatial surfaces.
Body Budget: 220/220 lines
Document: docs/dev/native-format-inventory.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-11 | Intent |
| 12-19 | Request Signals |
| 20-26 | Open First |
| 27-32 | Verify |
| 33-40 | Risks |
| 41-53 | Classification |
| 54-73 | Inventory |
| 74-183 | Immediate Gaps |
| 184-212 | Canary Map |
| 213-220 | Update Rules |
DOC_HEADER:END -->

## Intent

Track GPU-capable library surfaces by their current private execution carrier,
target `Native*` carrier, transfer classification, and canary coverage.

This is the working inventory for Phase 0 of the library-wide Native* plan.
It is not an implementation spec for any one operation. Use it to choose the
next slice and to avoid proving a rule on only one function.

## Request Signals

- Native* inventory
- carrier coverage
- GPU operation map
- host transfer classification
- native canary coverage

## Open First

- docs/dev/native-format-library-plan.md
- docs/dev/native-format-inventory.md
- docs/decisions/0044-private-native-execution-substrate.md
- docs/decisions/0046-gpu-physical-workload-shape-contracts.md

## Verify

- `uv run python scripts/check_docs.py --check`
- `uv run python scripts/intake.py "Native format inventory"`
- `uv run pytest tests/test_pipeline_benchmarks.py -k "native or relation or grouped" -q`

## Risks

- Marking a surface covered before a canary checks transfer shape can hide
  pandas or Shapely materialization.
- A carrier name alone does not prove the physical workload shape is correct.
- Some fences are legitimate scalar admissions; classify them by byte budget
  instead of deleting them blindly.

## Classification

- `export`: intentional public compatibility or user IO boundary
- `fence`: small scalar or metadata transfer required to admit a GPU shape
- `debt`: avoidable host transfer, pandas assembly, Shapely round trip, or
  public object construction inside an otherwise native workflow
- `covered`: has a Native* canary or strict-native regression
- `partial`: native pieces exist but composition still exports too early
- runtime D2H fences in admitted smoke canaries must carry operation-level
  reasons, not generic CUDA helper names
- source guards reject raw `cp.asnumpy`, unnamed runtime D2H copies, and active
  raw device `.get()` materialization in `src/vibespatial`

## Inventory

| Surface | Current GPU Shape | Target Carrier | Transfer Class | Coverage |
|---|---|---|---|---|
| GeoParquet/GeoArrow read | file/Arrow to `OwnedGeometryArray` and `NativeTabularResult` seeded with cached geometry metadata; public GeoParquet, Feather, Arrow table, and Arrow series reads attach metadata-seeded `NativeFrameState` for immediate native consumers; `bbox` filters keep device row positions and defer RangeIndex labels through `NativeIndexPlan`; native-backed GeoParquet/Feather writers record terminal export boundaries even when the public frame has native geometry backing but no full `NativeFrameState` | `NativeFrameState`, `NativeAttributeTable`, `NativeGeometryMetadata` | export at public dataframe/write boundary only; decode seeds existing classification/total-bounds metadata without recomputing row bounds | covered |
| GeoJSON/Shapefile/native file read/write | parser output to owned geometry/native tabular payload; GeoJSON dynamic-size, validity, property type-count, unsupported-type, and family-domain scalar fences are runtime-counted and avoid raw CuPy host export; OSM way classification/counts/reorder reuse CPU-parsed refs/tags while coordinate gather stays device-side; public GeoJSON string, Python geo-interface, geo-dict, and vector-file exports are terminal native export boundaries for native-backed objects; read-file datetime compatibility rewrites, lazy GeoJSON property expansion, GeoJSON mask-filtered Arrow/WKB reads, and named-index vector-file writes preserve deferred attributes until the terminal export | `NativeFrameState` with deferred attributes | format metadata fences; public export only; GeoJSON/geo-dict/vector-file exports are terminal `NativeExportBoundary` events | covered |
| WKB/GeoArrow encode/decode | owned buffers to Arrow/WKB bridge; public Arrow/Feather readers attach metadata-seeded `NativeFrameState`; native tabular, low-level owned, and public host-owned WKB/GeoArrow exports encode from owned buffers instead of temporary GeoSeries/public arrays; unsupported low-level and native-tabular GeoArrow geometry mixes use the repo-owned WKB bridge instead of upstream WKB constructor fallback, while public GeoSeries/GeoDataFrame unsupported mixed GeoArrow exports preserve GeoPandas `ValueError`; GeoSeries, GeoDataFrame, and direct native-tabular Arrow exports record terminal native export boundaries only after successful export; device GeoArrow encode misses are observable before host terminal export and strict-native rejects them; WKB decode sizing probes use named scalar fences; device GeoArrow family, validity, point-empty gates, and supported single/multi promotions use owned structure proofs without runtime D2H; pylibcudf GeoArrow decode reuses column child sizes before named scalar fences | terminal `NativeExportBoundary` plus metadata carrier | export unless immediately consumed; native consumer handoff covered for public Arrow/Feather ingress; scalar fences and explicit Arrow/GeoParquet/Feather/Shapely/display/GeoJSON/WKB/WKT exports are runtime events | covered |
| Bounds and total bounds | device row bounds or host ndarray; native-backed public row bounds and total bounds record terminal DataFrame/NumPy export boundaries | `NativeGeometryMetadata` | grid summary fence <=64 B; native metadata consumers reuse device row bounds before terminal export; public bounds/total-bounds are terminal exports when user-requested | covered |
| Flat spatial index | `NativeSpatialIndex` reuses flat-index bounds/order and public sindex caches native wrappers by lineage token; device and fused Hilbert builds keep row bounds/order resident and batch extent fences before public formatting | `NativeSpatialIndex` | regular-grid, Hilbert extent, or total-bounds scalar fence <=64 B; native query consumers use `NativeSpatialIndex` bounds/order carriers; public index formatting is terminal export | covered |
| Spatial query/sjoin | attached native frame state and public sindex/query can drive `NativeSpatialIndex` relation queries; scalar and multi-row public `sindex.query` device candidates stay relation-shaped until the explicit terminal index-array export; relation export wrappers feed rowsets/reducers and lower joined rows to `NativeTabularResult`/`NativeFrameState`; joined-row native exports can request all-valid device attribute storage and append distance columns; public sjoin non-empty/empty relation lowering and all-valid device-compatible `on_attribute` filtering stay device-resident when source state or public join-key columns can supply device attributes; dense/sparse public query exports format relation pairs through `NativeExportBoundary`; empty geometries are filtered before owned/native pair generation and pair rows are remapped to original frame positions; zero-cardinality device candidates and relation joins assemble from known shape without pair host export | `NativeRelation`, `NativeRowSet`, and native joined-row tabular payloads | public joined dataframe assembly is terminal export with validated state attached; Shapely-origin regular-grid certification fence <=160 B; scalar and multi-row device pair host formatting is explicit terminal export | covered |
| Point-in-polygon and binary predicates | staged PIP can now return a device `NativeExpression`; admitted row-aligned binary predicates over all-valid owned point/point, multipoint point-family, point-region, and non-point DE-9IM inputs can also return a device `NativeExpression` that feeds `NativeRowSet` and native frame takes without public bool Series export; `NativeFrameState` exposes row-aligned single- and multi-predicate expression helpers that preserve left-frame lineage for sanctioned rowset/take consumers; row-aligned multi-predicate DE-9IM requests share one device refine pass and emit multiple `NativeExpression` vectors without candidate-row export; covered-by single-mask polygonal probes can return a device vector for clip rowset consumers before the legacy host wrapper exports public bool masks; row-aligned point/point public predicates use device coordinate equality without row-bounds export; point-region predicates that need interior/boundary distinction consume the device relation kernel directly; scalar public `relate` and `relate_pattern` operands broadcast through owned native views when the row family is admitted; native-backed public binary predicate Series exports record terminal boundaries and mark predicate result transfers as terminal exports; return-device spatial-query refinement can filter point-family, multipoint-family, and non-point DE-9IM relation pairs without host pair, tag, mask, or DE-9IM-mask export, and admitted owned point-family domains can avoid the family-admission scalar fence entirely; indexed point-relation and covered-by single-mask branch routing use host metadata or named admission fences | `NativeExpression` for scores and row-aligned predicate vectors, `NativeRowSet` for filters, `NativeRelation` for pairs | row-aligned predicate vectors feed native rowset/frame consumers before terminal export; relation-pair predicate refinement preserves device pairs for native consumers; public bool Series is terminal export when user-requested | covered for row-aligned PIP expression -> rowset, row-aligned binary predicate expression -> rowset/native take including point/point and multipoint point-family, `NativeFrameState` predicate expression wrappers -> rowset/native take, shared-pass DE-9IM multi-predicate expressions, covered-by single-mask expression -> clip rowsets, row-aligned point/point public predicate terminal export, point-region relation-kernel expression -> rowset, device DE-9IM and multipoint relation-pair refinement -> device pairs, scalar-broadcast public relate/relate-pattern strict-native admission, public relate strict-native decline propagation, and native-backed public binary predicate Series export accounting |
| Nearest/distance | admitted point-nearest producers keep inner/left/right pair arrays and distances device-resident for expression, filter, and grouped distance consumers; return-device nearest kNN computes and expands row bounds on device for device candidate generation, with host-known or named extent fences for unbounded search; row-aligned owned distance over admitted point/lineal/polygonal family pairs can emit a device `NativeExpression` for rowsets and grouped reducers; spatial nearest/KNN/distance public exports are operation-named runtime D2H boundaries and native-backed sindex nearest formatting records terminal exports | `NativeRelation` with distance `NativeExpression`; row-aligned distance `NativeExpression` | named dynamic-output count fence <=8 B per producer; public nearest/KNN/distance formatting is explicit terminal export; row-aligned distance expression stays device-resident until a sanctioned consumer or terminal public Series export | covered for relation distance expression consumers, nearest relation producers, return-device kNN device-bounds candidate generation, native nearest attribute filters, public nearest terminal exports, and row-aligned owned distance expression -> rowset/grouped reducers |
| Area/length/centroid metrics | public Series or point geometry for broad API, private area/length/centroid-component expression canary with composed rowsets and guarded threshold ambiguity rowsets; native-backed scalar geometry properties record terminal public Series exports | `NativeExpression` | metric filters/reducers consume `NativeExpression` before terminal export; public Series or centroid point materialization is terminal export when user-requested | covered for area/length/centroid components, guarded metric thresholds, and native-backed public scalar property export accounting |
| Groupby numeric/bool reductions | `NativeGrouped` for dense codes and relation attributes, with device reductions exportable as `NativeAttributeTable`; admitted dissolve keys include categorical and ordinary nullable string/object/numeric single-key groups plus scalar object, categorical, and typed nullable multi-key groups; all-valid device string/datetime/categorical columns admit grouped `first`/`last` take reducers; device grouped reducer pandas exports use `NativeExportBoundary` | `NativeGrouped` plus `NativeAttributeTable` | native reducers consume dense codes and device values before terminal export; explicit grouped reducer pandas output is terminal export; nullable non-numeric device reductions remain movement-only | covered |
| Dissolve/union-all | grouped geometry kernels, device `NativeGrouped` unary polygon reducers, sparse host-code grouped unary reducers that scatter observed device groups into device empty polygon rows without Shapely row export, large regular-grid rectangle grouped disjoint assembly with batched named scalar certification, native make-valid repair before invalid grouped-union host recompute, polygon coverage edge reduction with device metadata scatter for unobserved categorical empty groups, host-known coverage-edge code/validity mirrors, all-valid full/observed-group coverage-edge proofs, low-fan-in all-valid device coverage groups that enter exact grouped coverage union without scalar admission probes, all-valid host/cache proofs for device grouped-union admission, global union/coverage-union valid-nonempty filtering from structural device metadata, global disjoint-subset assembly sized from structural offsets, and grouped constructive results lowering to `NativeTabularResult`/`NativeFrameState`; optional bbox component/coloring optimizers only consume already-materialized host bounds and decline on device-only rows instead of exporting row-bounds matrices; admitted nullable/categorical/object/numeric group-key contracts use native dense codes, while custom/unhashable keys stay on pandas policy | `NativeGrouped`, constructive native result, metadata | Shapely reference/export only; grouped outputs re-enter owned/native geometry before public export; unary-to-coverage is admitted only after coverage/noding proof; grouped/global disjoint admission, validity, and non-empty fences are byte-sized and operation-named when not structurally proved | covered for host grouped output -> owned native tabular lowering, sparse grouped unary device empty-row scatter, coverage-edge host-code occupancy sizing, coverage-edge all-valid zero-D2H admission, low-fan-in coverage zero-D2H admission including dropped rows, device grouped-union all-valid host/cache zero-D2H admission, unobserved-group device metadata scatter without variable-width take sizing, global union empty-row structural filtering with device empty output, grouped/global disjoint raw-sync accounting, optional bbox optimizer host-boundary decline, and union-all coverage validity count fences |
| Clip/overlay constructive workflows | owned geometry outputs, native tabular builders, row-aligned, part-source, relation-pair, repair, and part-family provenance, cached geometry metadata, constructive-output expression consumers, overlay assembly allocations sized from native cardinalities or host-known capacity, split-event and face-edge assembly sized from host-known capacities with device live-row compaction/CSR delimiters, grouped pair-position expansion from host-known pair counts, grouped exact-difference source-row expansion from device CSR offsets, aligned single-pair grouped difference as rowwise native work, grouped-union fallback before sequential exact fallback, broadcast-right chunk row-position restoration, containment-bypass named count fences, public clip/overlay admission probes with operation-named scalar fences, non-empty overlay row filters as device row views, mask-cover clip passthrough using `NativeFrameState` rowset takes and owned structural non-empty proofs, face assembly seeding all-valid proofs when preserved rows are dense, and polygon-mask clip consuming shared-pass and single-mask predicate expressions as native rowsets before row-position assembly | constructive native result plus `NativeFrameState`, `NativeGeometryProvenance`, and `NativeGeometryMetadata` | public keep-geom/type, attribute probes, unsupported selected-face/topology declines, and final public scatter index exports are explicit runtime-counted boundaries; admitted overlay allocation sizing, grouped pair expansion, grouped difference source rows, single-pair grouped difference, grouped-union fallback, many-vs-one chunk positions, split-event compaction, face-edge gather, public admission probes, cached-validity constructive proof gates, non-empty row filtering, mask-cover clip passthrough, and clip predicate split rowsets stay device-resident except byte-sized named scalar or row-position fences; binary constructive scalar gates are named runtime fences only when no cached/host proof exists | covered for owned pairwise, relation-pair geometry-only, public clip/overlay admission scalar-fence accounting, containment-bypass count accounting, mask-cover clip native passthrough rowsets, cached-validity proof gates including dense preserved-row face assembly, polygon-mask predicate expression -> rowset splits without full bool-mask export, and lineal/polygonal/mixed part output -> provenance/metadata -> expression consumers |
| Row-aligned unary constructive outputs | owned buffer, offset, envelope, boundary, hull, minimum-rotated-rectangle, affine transform, reverse, normalize, orient, simplify, segmentize, remove-repeated-points, set-precision, exterior/interiors, extract-unique-points, representative-point, centroid, minimum-bounding-circle, line-merge, interpolate, minimum-clearance-line, and clip-rect outputs lower to geometry-only native tabular results with provenance and metadata; boundary/interiors reuse host structural sizing, line-buffer admission uses host mirrors or named scalar fences, grouped hull rewrite uses named group-domain/nonempty fences, and point-buffer GPU admission validates device metadata with named scalar fences only when host metadata has not already proven point-only/non-empty inputs | constructive native result plus metadata | public GeoSeries construction is export; unsupported non-row-aligned outputs decline before native lowering; row-aligned GPU admission fences stay byte-sized; clip-rect native scatter consumes device row maps and line output count fences are named | covered for owned row-aligned output -> provenance/metadata, empty lineal boundary endpoint assembly, and zero-transfer boundary/interiors/hull canaries |
| Validity/make-valid | device repair/property kernels with public compatibility wrappers, row-aligned validity vectors that feed private rowsets, device make-valid admission through validity expressions, compact invalid-row GPU repair, adjacent-hole topology repair via grouped hole union plus exterior difference, and make-valid geometry-only native lowering | `NativeExpression` for properties, `NativeTabularResult` plus `NativeGeometryProvenance`/metadata for repairs | named scalar, null-row, invalid-row, ring-closure, duplicate-vertex, orientation, repaired-ring, touching-hole area/validity admission, and compact family-row fences allowed; public bool validity Series is terminal export only | covered for validity expression -> rowset, make-valid no-repair and repair expression gates, compact device family-row repair mapping, compact grouped-union repair gating, adjacent-hole topology repair without CPU fallback, and make-valid owned output -> native tabular |
| Public dataframe composition | registry-attached private state through exact operations, including cached geometry metadata, device attribute projection/assignment, nullable/categorical/string/datetime movement-only device attribute policies, exact boolean filters, `.loc`/`.iloc` row takes, row drops/dropna/reindex repeats, non-geometry duplicate-row drops, exact attribute-only fillna/replace/where/mask/astype, full and selected reset-index relabels, index/CRS relabels, existing active-geometry switches, exact GeoSeries CRS/copy/take/head/tail/drop/reindex/sample/sort-index/`__getitem__`/`.loc`/`.iloc`/metadata-relabel row transitions including duplicate-label and RangeIndex sort relabels, exact GeoDataFrame row sorts with duplicate-label and RangeIndex relabels, concat over appended indexes, public expression export accounting, declared point/lineal/polygonal plus mixed-row-family geometry explode, public GeometryCollection explode ingress, display repr export accounting, scalar property Series and bounds export accounting, NumPy array-protocol geometry export accounting, and GeoJSON/WKB/WKT/vector-file terminal export accounting for native-backed GeoDataFrame/GeoSeries objects | `NativeFrameState` | unknown pandas operations invalidate; exact filters/takes/sorts/projections/concat/relabels/fills should stay native; owned concat/take/scatter use structural row counts, mixed-family rowset presence, host mirrors, or named scalar fences; variable-width geometry takes may require bounded output-size fences; explicit native host bridges, display exports, scalar property/bounds/NumPy exports, and text/binary/file geometry exports are strictness-labeled and runtime-counted | covered for broad pandas expression/mutator invalidation boundaries, exact row/index/CRS/active-geometry transitions, dropna row/column composition including row `ignore_index` relabels, exact attribute-only fillna/replace/where/mask/astype, selected MultiIndex reset-index composition, exact GeoSeries CRS/copy/take/head/tail/drop/reindex/sample/sort-index/`__getitem__`/`.loc`/`.iloc`/metadata-relabel composition including duplicate-label and RangeIndex sort relabels, exact GeoDataFrame sort-values/sort-index/drop-duplicates duplicate-label and ignore-index relabels, exact concat, declared geometry explode, owned device take/scatter family-presence rowset proofs, native display repr boundaries, scalar property Series, bounds, NumPy array-protocol export boundaries, and GeoJSON/WKB/WKT/vector-file terminal export boundaries |

## Immediate Gaps

- `NativeSpatialIndex` can produce `NativeRelation`; public `sindex.query`
  exports indices, dense arrays, and sparse arrays from that relation for
  admitted owned inputs, keeping scalar and multi-row device pair arrays
  resident until the named terminal relation-index export; public nearest can format admissible
  `NativeRelation` pairs; and sjoin reuses cached native index state by lineage.
- `NativeExpression` covers area, length, centroid component filters, grouped
  summaries, composed `NativeRowSet` filters, and guarded threshold lowerings
  that return definite rows plus an ambiguous rowset for exact refinement;
  row-aligned point-in-polygon predicate results can also stay device-resident as a `NativeExpression` and lower directly to a rowset without public bool Series export; `NativeFrameState` can produce row-aligned predicate expressions from another native frame while preserving left-frame lineage; row-aligned point/point public predicates stay on device through coordinate equality and mark the final bool-vector copy as a terminal export;
  relation distance vectors and per-source relation match counts now have the
  same sanctioned expression consumer pattern, and admitted public point-nearest
  producers can keep inner/left/right pair arrays device-resident through the
  indexed point path before explicit public export. `NativeRelation` can also
  filter relation pairs by all-valid device-compatible shared attributes; row-aligned owned distance feeds device rowsets/grouped reducers; CCCL selected-count wrappers and profile-only pair-kind summaries are operation-named scalar fences.
  Nullable shared attribute filters are explicit observable
  decline/fallback boundaries under strict-native device requests. Public
  joined GeoDataFrame assembly is terminal export with attached native state;
  multipoint relation-pair refinement and row-aligned multipoint predicate
  expressions now stay device-resident through the indexed point relation kernels.
- Spatial join paths can produce native relations, relation attribute reducers
  can produce device `NativeAttributeTable`, and deferred/public join exports
  lower full joined rows to `NativeTabularResult`/`NativeFrameState`. All-valid
  attributes, distances, and non-empty/empty public sjoin lowering stay
  device-resident until GeoDataFrame export with validated native state attached.
  Polygon-box query certification uses named scalar fences for single-ring,
  coordinate-count, and axis-alignment proofs before the terminal bounds export.
- Public column projections, scalar and exact multi-column numeric/bool assignments, non-geometry
  reset-index insertion, `set_index` relabels, row drops, row/column dropna, exact attribute fillna/replace/where/mask/astype, reindex repeated
  labels, and metadata-only index/column relabels including `rename(index=...)` can now preserve
  all-valid `NativeAttributeTable` device payloads through `NativeFrameState`;
  admitted device attribute appends and numeric/bool reset-index group keys stay
  device-resident, nullable numeric/bool/categorical/string/datetime policies
  are movement-only unless compute is explicitly admitted, and unsupported
  appended dtypes export observably;
  sanctioned scalar, multi-column, and native-expression assignments, exact
  `.loc`/`.iloc` row takes, exact boolean Series filters, exact row sorts by value or index
  including duplicate-label indexes and `ignore_index=True` relabels, and homogeneous
  `concat` with RangeIndex output or exact appended public indexes preserve
  state, while arbitrary pandas expression
  and mutation surfaces such as `query`, `eval`,
  geometry-changing fill/replace/where/mask/astype,
  `apply`, `update`, `merge`, and `join` are explicit invalidation
  boundaries. Public active-geometry `explode` can preserve native state for
  declared point/lineal/polygonal and mixed-row-family part shapes by repeating
  attributes from device `source_rows`. Public GeometryCollection explode can
  re-ingest the already compatibility-expanded single-family parts into a
  device `NativeFrameState` with source-row and part-family provenance.
  Exact non-geometry `drop_duplicates(ignore_index=True)` now preserves native state as a rowset take plus RangeIndex relabel, `rename_axis(axis=1)` preserves native state as a column-axis metadata relabel, `set_crs` preserves native state as a metadata-only CRS relabel, and `set_geometry` over an existing native geometry column promotes the active geometry without rebuilding buffers. Native-backed GeoSeries imported from Arrow preserve their one-column `NativeFrameState` through exact copy, CRS relabels, positional takes, head/tail windows, drop/reindex/sample row selections, exact sort-index row selections including duplicate-label and `ignore_index=True` relabels, slices, boolean masks, label-aligned unique-index selections, `.loc`/`.iloc` row selections, and metadata-only `rename`/`rename_axis`/`set_axis` relabels. These have zero-D2H canaries.
  Downstream expression consumers stay zero-D2H. Private
  relation-pair constructive geometry can also lower from `NativeRelation` plus
  source `NativeFrameState` carriers into `NativeTabularResult` while preserving
  device left/right provenance for downstream geometry consumers.
- Constructive outputs can now hand owned pairwise geometry through
  `NativeTabularResult` into `NativeFrameState`, `NativeExpression`,
  `NativeRowSet`, and `NativeGrouped` consumers before public export while
  carrying row-aligned or part-source `NativeGeometryProvenance` and cached
  `NativeGeometryMetadata`. Pairwise constructive family classification now
  consumes device metadata and emits only a small family-domain scalar fence
  instead of full owned host metadata copies. The current canary keeps provenance
  device-resident and still has only bounded scalar fences plus the bounded
  polygon take fence. Owned row-aligned unary constructive outputs now lower to
  geometry-only native tabular results with provenance/metadata; boundary,
  interiors, line-buffer admission, shortest-line, minimum-clearance-line,
  clip-rect line outputs, `get_geometry`, and part expansion avoid raw CuPy
  scalar syncs via host sizing/mirrors, row-aligned assembly, or named fences.
  Part validity/output-size fences, proof scalars, source rows, and family tags
  stay named and device-resident through grouped/native consumers. Mixed-row-family
  explode composes family-specific part outputs by source row and carries output
  family tags in provenance.
  Grouped constructive results now expose direct `NativeTabularResult`/
  `NativeFrameState` lowering, and device `NativeGrouped` unary polygon
  reducers can feed grouped overlay union without host group-code assembly.
  Grouped repair output marks repaired grouped-union provenance at the native
  tabular boundary, and make-valid/clip lower row-aligned repair masks through
  `NativeGeometryProvenance` so takes and concatenations keep invalid-member
  repair lineage native.
- Overlay assembly allocation fences now reuse host-known native cardinalities, compacted output sizes,
  conservative coordinate capacity, host-known pair counts, split-event capacity, or face-edge capacity. Device
  offsets and live-row compaction delimit referenced coordinates/events/edges;
  host-indexed public takes/scatters carry sizing mirrors through nested gathers;
  bounded segment candidates use that shape, same-row warp admission batches span
  metadata in one named fence, and very large sweeps keep a named scalar fence.
  Grouped overlay non-empty filtering uses a device row view instead of host
  metadata materialization, grouped-disjoint dissolve uses batched named scalar
  fences, coverage-edge grouped union reuses host row-code/validity mirrors or all-valid full-group proofs, low-fan-in all-valid coverage groups use exact grouped coverage union without scalar admission probes, and unobserved categorical groups are filled by device metadata scatter without variable-width take sizing,
  contraction microcell rectangles reduce through grouped device union, and
  rectangle-intersection fast-path admission, boundary splitting, and all-rectangle
  few-right/many-vs-one classification reuse fixed-width polygon metadata before
  falling back to named scalar certification fences. Grouped exact difference can now keep
  bbox pair arrays, CSR group offsets, and right-geometry source-row expansion
  device-resident through the row-isolated topology planner; broadcast-right
  many-vs-one chunk positions and warning masks also restore row order by
  device inverse permutation. The admitted single-batch difference scatter now
  derives both neighbor and no-neighbor rowsets on device and feeds owned
  geometry scatter without a unique-left host export. Public non-empty cleanup
  and pandas-facing row export remain explicit terminal boundaries. The small
  grouped constructive canary now budgets
  native grouped union at 5 named D2H / 56B and rejects anonymous count-scatter
  or overlay assembly runtime D2H events; segmented union and public clip/overlay admission scalar fences are operation-named.
- GeoParquet native reads and WKB/GeoArrow tabular lowerings now seed or attach
  `NativeGeometryMetadata` so native consumers reuse decode-time geometry
  state instead of recomputing row bounds. All-valid device
  string/datetime/categorical attributes can now run grouped `first`/`last`
  take reducers directly from `NativeGrouped` offsets; nullable non-numeric
  reductions remain explicit export/fallback work. Device OGC validity now emits
  row-aligned `NativeExpression` values for rowsets, no-repair make-valid
  admission, compact repair, and terminal low-level owned WKB/GeoArrow exports.

## Canary Map

| Canary | Carrier Rule |
|---|---|
| `native-area-expression` pipeline | metric vectors, including centroid components and guarded threshold ambiguity rowsets, stay `NativeExpression` until composed rowset/grouped consumers |
| point-in-polygon and binary predicate expression tests | row-aligned predicate bool vectors, including `NativeFrameState` predicate helpers, point-region relation-kernel predicates, multipoint point-family expressions, and shared-pass DE-9IM multi-predicate vectors, stay `NativeExpression` until rowset consumers; row-aligned point/point public predicates avoid row-bounds export and mark final bool-vector copies as terminal exports; return-device point-family, multipoint-family, and DE-9IM relation-pair predicates filter device pairs without host pair, tag, mask, or DE-9IM-mask export |
| `native-metadata-index` pipeline | device bounds and spatial index state wrap as carriers, query directly to `NativeRelation`, and carry cached metadata through `NativeFrameState` rowset take |
| `relation-semijoin` pipeline | relation row flow can avoid joined public row assembly |
| relation joined-row native lowering tests | deferred sjoin export can lower full joined rows to `NativeTabularResult`/`NativeFrameState` without public GeoDataFrame materialization |
| public sjoin relation canaries | non-empty/empty device candidates, all-valid string/bool/numeric `on_attribute` relation filters, and relation joins stay device-resident until terminal export |
| `relation-attribute-reducer` pipeline | relation pairs can feed grouped attribute reducers and device attribute tables |
| `relation-distance-expression` pipeline | relation distances and per-source match counts can feed expressions, rowsets, filtered relations, and grouped distance reducers |
| `nearest-relation-producer` pipeline | public nearest producers can emit device `NativeRelation` distances, filter by device-compatible attributes, and remap right-join pairs before reference export |
| `row-aligned-distance-expression` tests | element-wise distance over admitted point/lineal/polygonal family pairs stays `NativeExpression` until rowset and grouped consumers |
| `grouped-reducer` pipeline | dense group codes can reduce numeric and all-valid boolean device values |
| `constructive-output-native` pipeline | pairwise constructive owned geometry can become `NativeFrameState`, carry row-aligned provenance and cached geometry metadata, and feed expression rowsets/grouped reducers |
| relation constructive native tests | relation-pair constructive geometry lowers through `NativeRelation` and source `NativeFrameState` carriers without public pair formatting |
| point/lineal/polygonal part native tests | dynamic part output carries device source-row and part-family provenance into grouped native expression consumers |
| row-aligned unary constructive tests | geometry-only unary outputs lower to `NativeTabularResult` with source provenance and cached metadata before public export |
| clip-rect native scatter test | device row maps feed row-restoring native scatter without full row-map host export |
| make-valid repair provenance tests | repaired rows lower to row-aligned native repair masks that survive take and concat |
| validity expression tests | public validity semantics stay as a device `NativeExpression` until rowset consumers and no-repair make-valid admission |
| grouped constructive native lowering tests | grouped constructive outputs, including large regular-grid rectangle disjoint groups and repaired grouped-union outputs, lower to `NativeTabularResult`/`NativeFrameState` with cached metadata and provenance before public export |
| device grouped unary union test | device `NativeGrouped` polygon rows reduce through grouped overlay union without host group-code materialization; sparse host-code grouped unary results scatter observed device rows into device empty polygon outputs before public export; all-valid cached rows avoid grouped-union validity/non-empty scalar fences; all-valid coverage-edge and low-fan-in dropped-row coverage groups route to exact grouped coverage union without scalar D2H admission probes; global union/coverage-union filters empty rows by device metadata and returns device empty output for all-empty input; unobserved categorical coverage groups append device empty polygon rows without owned take sizing fences |
| overlay assembly fence budget | output-ring, family-count, boundary-coordinate, hole-coordinate, coordinate-gather, expand-by-count, many-vs-one chunk-position, and clip predicate rowset assembly avoid public-shaped runtime D2H in admitted canaries |
| allocation-sizing canaries | host-known takes, fixed-width polygon device takes, mixed-family device take rowsets, public GeometryArray takes, host-indexed device scatters, device scatter family-presence rowsets, host-mirrored device rowsets, flat spatial-index device bounds, and optional union bbox grouping avoid full size/bounds copies; device-only nested takes batch totals; segment extraction uses host/device structural totals with batched fallback totals; bounded segment candidates use host-known capacity; atomic edges use split-event cardinality |
| strict-native public tests | public compatibility paths must expose or reject materialization; relate and global-reduction wrappers propagate strict-native declines, wrapper-level reduction failures are observable before Shapely, and admitted scalar-broadcast relate operands stay on the native path; tree-reduction GPU failure and pairwise strict declines are observable before CPU reduction; unary constructive, simplify, remove-repeated-points, and distance-metric GPU failures are observable before CPU geometry work; exact sort-values/sort-index/drop-duplicates RangeIndex relabels, existing-geometry active switches, and GeoSeries CRS/copy/take/head/tail/drop/reindex/sample/sort-index/`__getitem__`/`.loc`/`.iloc`/metadata-relabel composition preserve `NativeFrameState` with device-backed zero-D2H canaries |
| public geometry explode native test | declared point/lineal/polygonal, mixed-row-family, and GeometryCollection public explode shapes attach `NativeFrameState` with repeated or compatibility-expanded attributes, source-row provenance, part-family provenance, and family-rowset consumers |

## Update Rules

- Add an inventory row when a GPU-capable public surface appears.
- Mark `covered` only after a test or benchmark canary enforces the carrier
  shape and transfer classification.
- Keep byte budgets on fences explicit.
- Promote `partial` to `covered` only when the immediate downstream consumer
  composes through Native* without public object reconstruction.
