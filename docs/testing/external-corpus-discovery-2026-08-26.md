# External Corpus Discovery: 2026-08-26

<!-- DOC_HEADER:START
Scope: Archived initial external-corpus measurements and reusable bottleneck classifications.
Read If: You are choosing the next general performance remediation or extending the external corpus suite.
STOP IF: You only need the portfolio policy or acquisition commands.
Source Of Truth: Identity-qualified archive of the initial RTX 4090 discovery.
Body Budget: 179/180 lines
Document: docs/testing/external-corpus-discovery-2026-08-26.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-7 | Intent |
| 8-15 | Request Signals |
| 16-22 | Open First |
| 23-28 | Verify |
| 29-35 | Risks |
| 36-62 | Measurement Identity |
| 63-80 | Archived Results |
| 81-153 | Profiles |
| 154-169 | Classification |
| 170-179 | Next Tests |
DOC_HEADER:END -->

## Intent

Archive the first external-corpus measurements and classify the reusable
public execution gaps they exposed. These timings are not current evidence.

## Request Signals

- external corpus results
- GeoLife grouped tracks
- CMAB influence join
- OSM power nearest
- nested GeoParquet failure

## Open First

- `docs/testing/external-corpus-discovery-2026-08-26.md`
- `docs/testing/external-corpus-generalization-plan.md`
- `benchmarks/shootout/corpora/README.md`
- `benchmarks/shootout/corpora/vsbench-workload.json`

## Verify

- `uv run python scripts/manage_external_corpora.py verify`
- `uv run python scripts/check_docs.py --check`
- `uv run pytest tests/test_bench_shootout.py -k 'workload_identity or external_corpus_fingerprint or versioned_result_fingerprints' -q`

## Risks

- Full statement profiling adds synchronization and cannot replace wall medians.
- The small OSM files are compatibility controls, not scaling evidence.
- CMAB output amplification makes naive 100K extrapolation unsafe.
- Local raw artifacts are ignored; the tracked report must preserve context.

## Measurement Identity

**Evidence status: archived.** The original row-count/bounds/hull fingerprint
could miss wrong attributes, relation partners, distances, row association,
and ordering. These timings cannot establish current correctness.

- Date: 2026-08-26
- Host: `picard-4090`, Intel Core i9-13900K, local NVMe, RTX 4090
- vibeSpatial source parent: `52f6fc96537cf99e2eb289ae69049c484836423a`
- Candidate state: dirty with the external-corpus harness in this change
- GeoPandas comparator: GeoPandas 1.1.4, pandas 3.0.5, PyArrow 25.0.1,
  Shapely 2.1.2; candidate: vibeSpatial 0.5.2, pandas 3.0.1, PyArrow 23.0.1
- Timing: repeat-three median with warmup except the diagnostic 1K sweep
- Correctness: legacy incomplete fingerprint; superseded by ordered v3
- Dispatch: default automatic public APIs; no private selectors or hints

Artifacts span incompatible identities: discovery-10K/scaling `d8862c76...`,
nested repair `960edfb4...`, and later individual runs `f1eaae4c...`.
Discovery-10K includes a failed nested-IO row; none may be combined or reused.

Current capsule `5cf13540c71dfa91e70d84aee43a1a590b674c97338cb4473aaeb44446e9051d`
passed all six v3 fingerprints in a local repeat-one 10K validation. Those
single samples establish correctness only, not current performance claims.

Ignored raw artifacts are under `benchmark_results/external-corpora/`; source
revisions and hashes are tracked in the workload manifest.

## Archived Results

| Workflow | Admitted rows | GeoPandas | vibeSpatial | Speedup | Legacy fp |
|---|---:|---:|---:|---:|---|
| CMAB influence join | 10K | 0.6874 s | 0.9101 s | 0.755x | match |
| GeoLife grouped tracks | 10K | 0.1065 s | 0.1142 s | 0.933x | match |
| GeoLife grouped tracks | 100K | 0.1498 s | 0.1668 s | 0.898x | match |
| GeoLife grouped tracks | 1M | 0.6417 s | 0.5177 s | 1.240x | match |
| OSM MultiPolygon enrichment | all 1,934 | 0.1697 s | 0.5664 s | 0.300x | match |
| OSM nested GeoParquet IO | all 1,934 | 0.0283 s | 0.0292 s | 0.970x | match |
| OSM WKT land-use summary | all 627 | 0.2466 s | 0.2622 s | 0.940x | match |
| Power substation nearest | 10K | 0.0378 s | 0.0535 s | 0.706x | match |
| Power substation nearest | 100K | 0.3287 s | 0.1778 s | 1.848x | match |

This table combines historical per-workflow artifacts and is not a current
suite result. A later isolated nested-IO run reported a match after the
schema-fidelity repair, but it used a different capsule identity.

## Profiles

Full profiling synchronizes statements and is not directly comparable to the
repeat-three wall median. Stage times below are used to rank costs within the
profile.

### CMAB At 10K

The GPU spatial join took 0.168 s. The subsequent ordinary public
`sort_values(...).drop_duplicates(...)` took 0.791 s. The writer took 0.064 s.
The join produced enough relation amplification that 100K would mainly test
capacity. This lane stops at 10K until relation shaping is addressed.

The profile recorded 15 runtime D2H events and 33.9 MB. Most bytes came from
terminal relation-pair export and host row-position normalization before the
pandas-shaped selection.

### GeoLife Scaling

The first implementation selected CPU and spent 2.490 s in 1M grouped
`dissolve`. The remediated public path builds 2D Points directly into owned
SoA buffers, then uses exact device `(group, x, y)` radix partitioning,
coordinate deduplication, and Point/MultiPoint offset assembly. At 1M,
end-to-end wall fell from 3.115 s to 0.518 s and dissolve profiled at 0.238 s.

The historical automatic path was neutral-to-positive across scale: 0.933x at
10K, 0.898x at 100K, and 1.240x at 1M versus refreshed GeoPandas comparators.
Its legacy fingerprints reported matches but do not satisfy v3 correctness.

### Power Nearest Scaling

At 100K, nearest relation generation took 0.064 s, relation sorting and
deduplication took 0.060 s, output took 0.058 s, GeoDataFrame construction took
0.038 s, and tabular input took 0.024 s. Despite composition costs, the same
public workflow crossed from 0.706x at 10K to 1.848x at 100K.

Historically, no user action changed between scales, so this lane remains a
useful candidate for a repeat-three v3 no-regression control.

### OSM Storage Shapes

The original WKT measurement filtered for `"ok"` although the corpus uses
`"clean"`, so it timed an empty post-ingest workflow. A later legacy run
processed 627 rows into 698 parts and three groups and reported a match under
the superseded fingerprint: GeoPandas was 0.2466 s and vibeSpatial 0.2622 s.

The native WKT parser uses a row-parallel fixed-capacity scan over each geometry
to validate structure and count family output sizes. Numeric token spans feed
a device-only correctly rounded fp64 parser; one compact validation/allocation
packet precedes direct family scatter. On
this skewed 3.28 MB corpus, warm parsing fell from about 89.9 ms to 32.8 ms
versus Shapely's 32.9 ms, so auto still selects host at 627 rows. The former
100K gate is gone: 4K small polygons are 1.64x faster on GPU and 8K points are
1.82x; admission uses rows, bytes, and skew.

Device-owned `explode()` exports only its source-row index map. A grouped-union
shape guard prevents residency from forcing dense pairwise topology: native
ingest plus automatic host exact union is about 0.271 s, versus 10.73 s when
every downstream operation is forced to GPU. The latter is diagnostic only.

Projected MultiPolygon enrichment passed after column projection, but its full
profile attributed 1.754 s to output and 0.382 s to GeoParquet input. The
repeat-three end-to-end median was 0.566 s, confirming significant profiling
synchronization overhead but still a real 0.300x public-workflow result.

The initial scan failed because libcudf intentionally omitted embedded Arrow
schema restoration and returned unnamed nested struct children. The native
carrier now recursively restores the authoritative Arrow field types at public
materialization, while device Parquet writes propagate nested child names and
nullability into libcudf metadata. The later public read/write/read round trip
reported a legacy match for all 1,934 rows with no fallback. Its schema repair
remains covered by tests, and the current repeat-one v3 capsule now passes.

## Classification

The first sweep found three reusable gaps rather than polygon-specific gaps:

1. **Grouped simple-geometry collection.** Point-group dissolve now lowers to
   exact segmented Point/MultiPoint assembly and is a permanent scaling ratchet.
2. **Native relation shaping.** Sort, stable first-per-key, deduplication, and
   final attribute/index assembly can dominate an otherwise fast join.
3. **Small complex-geometry IO/export fixed cost.** Device output can lose by a
   large factor even when computation stays on GPU.

Tabular `pd.read_parquet` and public GeoDataFrame construction are also visible
costs, but the Power 100K win shows they do not necessarily prevent material
automatic acceleration. Optimize them only through a general public boundary,
not a corpus-specific loader.

## Next Tests

1. Preserve GeoLife at 10K, 100K, and 1M as the grouped-point scaling ratchet.
2. Preserve Power 100K as the successful public nearest control.
3. Add a relation-selection canary with low and high output amplification to
   separate sort/dedup cost from exact spatial work.
4. Preserve nested list-of-struct round-trip as a correctness control.
5. Add one GeoArrow contract corpus and one VIDA 10M building partition.
6. Reuse the saved GeoPandas legs after workload identity stabilizes; do not
   remeasure them for vibeSpatial-only changes.
