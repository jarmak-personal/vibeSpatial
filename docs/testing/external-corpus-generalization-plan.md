# External Corpus Performance Generalization Plan

<!-- DOC_HEADER:START
Scope: External real-data discovery, public workflow measurement, corpus promotion, and performance-generalization gates.
Read If: You are adding an external dataset, interpreting external-corpus evidence, or optimizing a public workflow found by the corpus sweep.
STOP IF: You only need an operation-local kernel benchmark or an existing immutable comparator timing.
Source Of Truth: External-corpus benchmark portfolio and public automatic-acceleration policy.
Body Budget: 232/250 lines
Document: docs/testing/external-corpus-generalization-plan.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-17 | Intent |
| 18-26 | Request Signals |
| 27-34 | Open First |
| 35-43 | Risks |
| 44-66 | Success Contract |
| 67-95 | Measurement Unit |
| 96-129 | Portfolio |
| 130-171 | Discovery Funnel |
| 172-194 | Evidence Contract |
| 195-209 | Initial Capsule |
| 210-225 | Promotion And Remediation |
| 226-232 | Verify |
DOC_HEADER:END -->

## Intent

Use diverse external vector datasets to discover where complete public
vibeSpatial workflows lose time. Geometry throughput is only one possible
cause. IO, pandas-shaped composition, index semantics, relation assembly,
grouping, sorting, null handling, host/device movement, and terminal output are
equally important benchmark surfaces.

The portfolio is deliberately broad. A corpus earns initial testing when it
adds even a plausible execution shape. It earns permanent regression status
only after measurements show that the shape is distinct and reproducible.

This plan implements ADR-0043's rule: workflows are canaries; reusable physical
shapes are the optimization unit.

## Request Signals

- external corpus
- real-data benchmark
- public workflow performance
- automatic acceleration
- pandas composition
- corpus promotion

## Open First

- `docs/testing/external-corpus-generalization-plan.md`
- `docs/decisions/0043-public-api-physical-plan-coverage.md`
- `docs/testing/pipeline-benchmarks.md`
- `benchmarks/shootout/corpora/vsbench-workload.json`
- `src/vibespatial/bench/shootout.py`

## Risks

- Dataset-specific optimization can masquerade as general acceleration.
- Geometry-only timing can hide dominant frame composition and export costs.
- Mutable remote assets invalidate correctness and timing comparisons.
- Repeated comparator runs waste time and introduce denominator drift.
- Scaling candidate-heavy joins without inspecting amplification can exhaust
  memory without adding useful evidence.

## Success Contract

- Timed code uses ordinary GeoPandas-compatible public APIs only.
- Default `auto` dispatch is measured. Scripts may not select a private kernel,
  index, algorithm, precision tier, or execution carrier.
- Users must receive acceleration from normal data and API semantics. They must
  not need vibeSpatial expertise, precomputed private state, or benchmark hints.
- Dataset preparation may normalize an external format to a documented
  GeoParquet artifact, but the conversion is immutable, hashed, and outside the
  timed boundary.
- Correctness fingerprints must match the same-data GeoPandas oracle before
  wall-time comparisons are valid. The ordered v3 digest covers exact schema,
  index, IDs, nulls, geometry topology, row association, and row order.
  Constructive geometry serialization order is normalized; fp64 metrics use
  seven significant digits and coordinates use a tighter eleven. Unsupported
  result values fail the workflow closed.
- A fast kernel does not make a workflow fast if frame composition or export
  dominates end to end.
- A specialized internal fast path is admissible only when an evidence-derived
  selector chooses it automatically and declines before taxing simple inputs.
- Existing 10K, 1M, full pipeline, and SF100 gates remain mandatory when a
  discovered bottleneck leads to runtime changes.

## Measurement Unit

The measurement boundary is a complete public workflow:

```text
read or construct
  -> project/filter/align
  -> spatial relation or constructive operation
  -> sort/group/reduce/reshape
  -> public GeoParquet output
```

Each artifact records:

- end-to-end and statement-level wall time;
- selected versus actual backend;
- GPU hotpath time and composition overhead;
- fallback and off-ramp events;
- runtime D2H/H2D counts, bytes, and reasons;
- materialization boundary counts and reasons;
- peak device memory when audit profiling is enabled;
- input rows and the available physical-work counters;
- output correctness fingerprint;
- source revision, local file SHA-256, workload hash, package environment,
  machine identity, warmup, repeat, scale, and timeout.

Download and SHA verification are never timed. Static GeoPandas evidence is
measured once per identity packet and reused fail-closed.

## Portfolio

### Discovery Now

- OSM description-tag MultiPolygons: nested attributes, GeoParquet WKB,
  projection, explode, metrics, grouped geometry reduction, and output.
- OSM polygon selection: tabular Parquet plus WKT decode, frame construction,
  filtering, explode, grouped reduction, and output.
- OSM power grid: tabular point construction, reprojection, nearest relation,
  relation sorting/deduplication, and output.
- CMAB spatial join: attributed numeric columns, variable-distance buffers,
  candidate-heavy spatial join, attribute restriction, relation shaping, and
  output.
- GeoLife spatial join: multi-million-row Parquet ingest, pandas filtering and
  sorting, point construction, grouped geometry reduction, and output.

### Acquire Next

- VIDA Google-Microsoft buildings for 10M, 100M, and 1B polygon scaling.
- Overture administrative boundaries and places for hierarchical regions,
  point/region joins, nearest, and mixed attributes.
- APRIL TIGER polygons and 16.9M lines for polygon/polygon and polygon/line
  filter-refine workflows.
- RayJoin public county, ZIP, block-group, water, lake, and park sources after
  converting the original geometries rather than its specialized CDB format.
- GeoArrow data for encoding, CRS, dimensionality, antimeridian, and mixed
  geometry contracts.
- Synthetic 100M GPS data as a permissive scaling stress only after its schema,
  license, and generator provenance pass review.

SpatialBench remains the multi-stage SQL-derived workflow suite. External
corpora test whether the same physical improvements generalize to independent
data and compositions.

## Discovery Funnel

### D0: Metadata Audit

Pin repository revision, file path, SHA-256, byte size, license, schema,
geometry encoding, CRS, row groups, rows, and geometry family. Reject mutable
latest URLs and assets without usable redistribution terms.

### D1: 1K Capability Sweep

Run each workflow once without warmup and with full statement profiling. The
goal is to find correctness failures, unsupported public composition, hidden
fallback, format incompatibility, or fixed overhead. Cold timings are
diagnostic and are not published as steady-state parity evidence.

### D2: 10K Characterization

Run repeat-three medians with warmup. Save the first immutable GeoPandas
comparator and current vibeSpatial evidence. Classify each slow workflow as:

- input/codec or output writer;
- public frame composition;
- relation assembly or index semantics;
- missing native carrier continuity;
- work amplification or candidate explosion;
- grouped reduction;
- geometry kernel floor;
- external denominator or unavoidable format cost.

### D3: Scaling Ladder

Promote informative workflows through 1M, 10M, 100M, and 1B when the source
supports those sizes. Stop a lane when its physical shape is already clear,
capacity is exceeded, or larger scale adds cost without information. Capacity
failures remain recorded evidence.

### D4: Permanent Ratchet

A workflow enters the routine suite only when it has a unique coverage role, a
stable acquisition path, a same-data oracle, manageable gate cost, and a named
owner physical shape. Other corpora remain opt-in research lanes.

## Evidence Contract

The manifest under `benchmarks/shootout/corpora/` is the first capsule's source
of truth. The acquisition script verifies every local asset before a timed
process starts. The manifest and all capsule Python files form an isolated
workload hash, so unrelated shootout edits do not invalidate its comparator.

Reported results separate:

- measured versus reused GeoPandas evidence;
- cold discovery probes versus steady-state repeat-three medians;
- correctness, execution coverage, and physical-plan performance;
- terminal user exports from internal host conversions;
- expected capacity limits from correctness or execution failures.

No aggregate speedup is reported until every included workflow passes its
fingerprint. Failed workflows remain visible rather than being removed from the
denominator.

Changing the fingerprint contract invalidates the whole capsule identity.
Results from older identities remain archival context only, even when an
isolated repair later passed; they cannot be combined into a current suite.

## Initial Capsule

The first pinned download is about 80 MiB and spans five independent sources.
It intentionally contains both geometry-native and tabular-coordinate inputs:

- 1,934 Algeria MultiPolygons with nested Arrow attributes;
- 627 American Oceania OSM land-use polygons stored as WKT;
- 164,374 global OSM substations stored as longitude/latitude columns;
- 104,640 Beijing CMAB attributed building rectangles;
- 2,507,357 GeoLife spatiotemporal rectangles.

The small polygon files are geometry-complex controls rather than scale tests.
CMAB and GeoLife provide the first 10K/100K/1M frame and relation ladders.
Larger files are added only after the initial workflows and identities settle.

## Promotion And Remediation

When a workflow exposes a bottleneck:

1. reproduce it at the smallest informative scale;
2. quantify geometry time versus orchestration and materialization;
3. identify at least two structurally different remedies;
4. select a reusable physical workload shape, not the dataset or script name;
5. define semantic admission and constant-time decline rules;
6. implement beneath existing public APIs with no user hint;
7. verify the external canary plus 10K, 1M, pipeline, and relevant SF100 gates;
8. reject the remedy if simple/common workflows pay measurable planning tax.

Dataset-specific branches, filename detection, script detection, public tuning
knobs, and private benchmark APIs are prohibited.

## Verify

- `uv run python scripts/manage_external_corpora.py verify`
- `uv run pytest tests/test_bench_shootout.py -k 'workload_identity or external_corpus_fingerprint or versioned_result_fingerprints' -q`
- `uv run vsbench shootout benchmarks/shootout/corpora --scale 1k --repeat 1 --no-warmup --profile-mode full --json --output benchmark_results/external-corpora/discovery-1k.json`
- `uv run vsbench shootout benchmarks/shootout/corpora --scale 10k --repeat 3 --profile-mode full --json --output benchmark_results/external-corpora/discovery-10k.json`
- `uv run python scripts/check_docs.py --check`
