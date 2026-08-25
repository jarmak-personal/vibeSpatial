# Grouped Constructive Distributive Execution

<!-- DOC_HEADER:START
Scope: Exact many-features by few-grouped-coverages execution, provenance continuity, and distributive constructive reduction.
Read If: You are changing grouped overlay followed by dissolve, buffer existential queries, or constructive fusion.
STOP IF: You only need an isolated overlay or dissolve kernel detail.
Source Of Truth: Physical-shape plan and measured evidence for distributive grouped constructive workflows.
Body Budget: 215/220 lines
Document: docs/dev/grouped-constructive-distributive-execution-plan.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-13 | Intent |
| 14-21 | Request Signals |
| 22-28 | Open First |
| 29-35 | Verify |
| 36-44 | Risks |
| 45-69 | Measured Diagnosis |
| 70-104 | Physical Shape |
| 105-135 | Correctness Contract |
| 136-153 | Implementation |
| 154-201 | Evidence And Gates |
| 202-215 | Deferred Automatic Fusion |
DOC_HEADER:END -->

## Intent

Optimize the reusable workload shape in which many source polygons are
intersected with a few region groups and the fragments are then unioned by the
region group. The implementation must use public APIs, preserve exact geometry
semantics, remain memory-bounded, and avoid a query-specific kernel.

Status: the public workflow and native provenance continuity are implemented.
Automatic eager-overlay-to-dissolve fusion is intentionally deferred pending a
second independent workload that needs it.

## Request Signals

- grouped overlay
- dissolve after intersection
- repeated constructive pages
- buffer existential query
- reduce before construct

## Open First

- `docs/decisions/0017-dissolve-grouped-union-pipeline.md`
- `docs/architecture/overlay-reconstruction.md`
- `src/vibespatial/api/_native_result_core.py`
- `benchmarks/shootout/redevelopment_screening.py`

## Verify

- `uv run pytest tests/test_spatial_query.py -k buffer_rewrite -q`
- `uv run pytest tests/test_overlay_api.py -k grouped_intersection_distributes -q`
- `uv run python scripts/check_docs.py --check`
- `uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline`

## Risks

- Retaining operation provenance across a row-changing transform can certify a
  false rewrite against stale source rows.
- Coverage union is invalid when source interiors overlap.
- Preserving fragment attributes or counts makes the distributive lowering
  semantically invalid.
- Removing pages without reducing relation cardinality can exhaust GPU memory.

## Measured Diagnosis

The original 1M redevelopment workflow used 10,000-row application pages,
clipped four grouped zone coverages for every page, constructed every
parcel-zone fragment, coverage-dissolved every page, wrote and reread every
page, then coverage-dissolved the page outputs.

An exact statement profile on the RTX 4090 measured:

- 311.996 s total;
- 249.644 s in the overlap-heavy polygon existential transit query;
- 58.637 s in the paged grouped constructive branch; and
- 3.444 s in the initial parcel-exclusion difference.

The existential algorithm already had a bounded point-buffer rewrite. A
semantics-preserving GeoDataFrame column projection dropped the buffer operation
provenance from `NativeFrameState`, so the public query missed that algorithm
and scanned the dense polygon relation. Preserving identity-safe operation
provenance reduced the stage to 2.651 s and the unchanged workflow to 64.726 s.

The remaining page workflow still constructed a relation that the terminal
grouped union did not need. A full unpaged overlay of 414,447 candidates by four
complex grouped zones exhausted the 24 GiB 4090, confirming that merely removing
application paging has the wrong memory shape.

## Physical Shape

For source polygons `A_i` and all zone polygons in group `g`, `B_gj`, the
terminal geometry is:

```text
union_i,j (A_i intersection B_gj)
```

Set intersection distributes over union, so the exact lower-work form is:

```text
union(A_i) intersection union(B_gj)
```

The public physical plan is therefore:

```text
source rows
  -> exact existential row selection
  -> one certified source coverage

zone rows + right-side group key
  -> exact grouped zone union

source coverage x grouped zone coverage
  -> bounded few-right overlay
  -> grouped coverage reduction
  -> terminal GeoParquet export
```

This replaces relation cardinality proportional to source rows with a relation
proportional to the number of retained groups. It also reduces geometry before
the allocation boundary that made the full raw overlay fail.

## Correctness Contract

The distributive lowering is admissible only when all of the following hold:

- The terminal geometry is a union grouped only by right-side keys.
- No left attribute, fragment count, fragment order, or fragment-level
  aggregation survives the reduction.
- Right-side non-key attributes are either absent or have an explicitly
  equivalent grouped aggregation.
- The source reduction is an exact set union. `method="coverage"` is admissible
  only with the ordinary coverage precondition: valid polygonal inputs with
  non-overlapping interiors.
- The zone reduction is an exact union for each group.
- Null geometries follow dissolve semantics and are skipped; valid empty
  geometries remain identity elements.
- The overlay keeps the same dimensional policy. The redevelopment workflow
  uses the public polygon default, so lower-dimensional boundary remnants are
  dropped on both sides of the rewrite.
- CRS, validity repair, and planar-coordinate semantics are unchanged.
- Group-key ordering and the final public schema remain identical.

The redevelopment source rows are clipped cells from an interior-disjoint
parcel grid. Difference removes subsets of those cells and cannot introduce
interior overlap between different source rows. The coverage certificate is
therefore structural, not inferred from the benchmark output.

The rewrite is not legal for workflows that retain parcel identifiers, compute
per-parcel statistics, count intersections, depend on first/last fragment
attributes, preserve lower-dimensional fragments differently, or use
overlapping source polygons with coverage union.

## Implementation

1. `GeometryNativeResult` carries optional operation provenance across
   identity-preserving native frame reconstruction and column projection.
2. Geometry assignment captures the public array provenance in the native
   geometry result. Reconstructed public geometry restores that tag without a
   host materialization.
3. Row-changing device takes clear untransformed operation provenance. Keeping
   a point-buffer tag after removing or reordering source points could produce
   a false existential certificate.
4. `SpatialIndex.query_any()` can consequently select its existing bounded
   point-buffer existential path after ordinary public column projection.
5. The redevelopment public query now expresses the distributive plan with
   `dissolve(method="coverage")`, grouped `dissolve()`, and `overlay()`; it does
   not call private carriers or benchmark-only kernels.
6. Explicit intermediate Parquet page barriers are removed. Terminal output
   remains an ordinary public GeoParquet write.

## Evidence And Gates

RTX 4090, 1M, strict-native, identical persisted fixtures:

| Shape | Total / phase | Correctness |
|---|---:|---|
| Original paged workflow before provenance continuity | 311.996 s | canonical fingerprint |
| Unchanged paged workflow after provenance continuity | 64.726 s | canonical fingerprint |
| Full raw unpaged overlay | OOM | fail-closed; no retry |
| Distributive constructive phase | 4.860 s | canonical fingerprint |
| Distributive preparation plus constructive phase | 21.317 s | canonical fingerprint |
| Rewritten public script, timed harness | 10.082 s | canonical fingerprint; zero fallback/offramps |
| Rewritten public script, isolated statement profile | 11.407 s | canonical fingerprint; zero fallback/offramps |

Canonical 1M fingerprint:

```text
rows=4 bounds=(599.05, 309.85, 850.0, 738.62) convex_hull_area=279078.10
```

The authoritative 1M profile assigns 2.635 s to the buffer existential and
5.025 s to the reduced grouped constructive statement. The timed result is
30.9x faster than the original vibeSpatial workflow and 6.4x faster than the
same paged workflow after the provenance fix alone.

At 10K, the rewritten script measures 0.366 s versus 0.709 s for its refreshed
GeoPandas comparator (1.94x) and matches both the exact fingerprint and the
latest validated pre-change vibeSpatial timing. The refreshed 1M GeoPandas leg
timed out at 895 s in its pre-existing transit spatial semijoin, before the
constructive branch, and emitted no correctness fingerprint. It therefore
provides neither a valid point timing nor evidence about CPU execution of the
distributive branch; no numeric 1M cross-library speedup is claimed.

The current one-warmup SF100 regression pass is 12/12 accurate and totals
456.68 s versus the prior 481.06 s (5.1% faster). Eleven normalized query
results are byte-identical; Q6 differs only by sub-ULP floating-point reduction
order and passes the established `rtol=1e-6`, `atol=1e-9` oracle.

The mandatory full pipeline profile also passes. Every successful 1M pipeline
has zero compute D2H transfers, zero compute materializations, and zero
fallbacks. The largest 1M stage is the 71.3 ms
`grouped-capacity-partitions/mixed_strip_exact_union`; no stage approaches the
one-second CPU-heavy investigation threshold.

Required landing gates remain focused provenance/query tests, public 10K and 1M
fingerprints, the broader spatial query and native-carrier suites, shootout
balance, SF100 accuracy/performance, and the mandatory full pipeline profile.

## Deferred Automatic Fusion

An optimizer could eventually defer an eager public overlay into a native
constructive expression and let a following grouped dissolve apply this law
automatically. That requires a carrier representing unresolved fragments,
attribute lineage, group-key lineage, dimensional policy, validity policy,
readiness, and a terminal materializer.

Do not add that abstraction from one workload. The current public APIs already
express the lower-work plan clearly, and explicit Parquet writes are semantic
materialization boundaries that no optimizer may silently cross. Reopen
automatic fusion only when another independent workflow demonstrates the same
overlay-then-grouped-union shape and cannot reasonably express the reduced
public plan.
