# SpatialBench Q1-Q12

This directory is the vibeSpatial source of truth for the public-API query
implementations used for the SF100 results. It is separate from
`benchmarks/shootout/`, whose synthetic 10K/1M workflows are regression tests
and are not the SpatialBench Q1-Q12 suite.

## Files

- `public_api_queries.py`: shared SQL-derived query contract and helpers.
- `geoparquet_public_api_queries.py`: streaming GeoParquet Q1-Q12 plans.
- `vibespatial_queries.py`: vibeSpatial-specific public-API physical plans.
- `geopandas_optimized_queries.py`: optimized GeoPandas comparison entrypoint.
- `prepare_geoparquet.py`: one-time WKB-to-GeoParquet 1.1 conversion.
- `run_benchmark.py`: isolated public-engine Q1-Q12 runner used by the evidence.

The files were imported from SpatialBench revisions `321c3f7`, `230ebd5`, and
`3414c8a`. The local modules support both package imports and SpatialBench's
standalone file loader.

## Reproduction contract

Use GeoParquet 1.1 with native GeoArrow geometry, prepared once outside the
timed benchmark. Both engines must read the same converted dataset. Run each
engine/query in an isolated process with one untimed warmup and report the
median of three measured runs. Timed work includes scan, compute, and public
result construction; result serialization is excluded.

The frozen SF100 environment and commands are recorded in
`benchmark_results/spatialbench/sf100/2026-08-14-final-median/provenance.json`.
The query-level semantics, timings, memory budgets, and correctness evidence
are in `docs/dev/pylibcudf-sf100-query-ledger.md`.

Prepare legacy SpatialBench WKB shards once, outside benchmark timing:

```bash
uv run python benchmarks/spatialbench/prepare_geoparquet.py \
  <sf100-wkb-root> <sf100-geoparquet-root>
```

Then run the repository-local benchmark harness:

```bash
uv run python benchmarks/spatialbench/run_benchmark.py \
  --data-dir <sf100-geoparquet> \
  --engines geopandas_optimized,vibespatial \
  --queries q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12 \
  --scale-factor 100 \
  --warmup-runs 1 \
  --runs 3 \
  --statistic median \
  --timeout 7200 \
  --result-dir <results>
```

Compare every result with the committed SpatialBench answers or the same-data
optimized GeoPandas outputs. A suite total must never hide a failed query.

## Frozen SF100 acceptance evidence

The immutable optimized-GeoPandas comparator is
`benchmark_results/spatialbench/sf100/accepted-geopandas-comparator.json`.
Its 8,086.00-second total is mechanically derived from the accepted original
twelve-query artifact by replacing only Q5 with the later checked-in
same-contract median and samples. The packet binds the workload sources,
package lock and versions, host, dataset identity, derivation inputs, and all
twelve normalized oracle CSVs by SHA-256.

Validate the frozen packet without running a query:

```bash
uv run python -m benchmarks.spatialbench.sf100_evidence check-comparator
```

For M5, run only the changed vibeSpatial implementation and reuse the frozen
GeoPandas comparator. Use a new result directory because the runner removes
stale files for the selected engine and queries:

```bash
env VIBESPATIAL_STRICT_NATIVE=1 UV_CACHE_DIR=/tmp/uv-cache \
  uv run python -m benchmarks.spatialbench.run_benchmark \
  --data-dir /home/picard/datasets/spatialbench/v0.1.0/sf100-geoparquet \
  --engines vibespatial \
  --queries q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12 \
  --scale-factor 100 --warmup-runs 1 --runs 3 --statistic median \
  --timeout 7200 \
  --output <acceptance-dir>/sf100_vibespatial_repeat3.json \
  --result-dir <acceptance-dir>/results
```

Verify identity, the measurement contract, every ordered result cell, and the
10x suite gate. This command fails closed on a changed source artifact, lock,
package version, host, dataset, missing query/output, schema/order mismatch,
exact-value drift, numeric tolerance failure, or timing-contract mismatch:

```bash
uv run python -m benchmarks.spatialbench.sf100_evidence verify-candidate \
  --candidate <acceptance-dir>/sf100_vibespatial_repeat3.json \
  --result-dir <acceptance-dir>/results \
  --dataset-dir /home/picard/datasets/spatialbench/v0.1.0/sf100-geoparquet \
  --report <acceptance-dir>/sf100_acceptance.json
```
