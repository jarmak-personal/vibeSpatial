---
name: performance-baselining
description: "PROACTIVELY USE THIS SKILL when measuring performance, creating or updating a benchmark baseline, running public shootouts, checking 10K/1M/SF100 regressions, or preparing performance evidence for review/commit. It separates immutable comparator baselines from current vibeSpatial measurements, reuses validated GeoPandas results, and defines when a baseline must be refreshed. Trigger on: baseline, benchmark, shootout, performance regression, profiling evidence, before/after, GeoPandas comparison, 10K, 1M, 10M, SF100."
---

# Performance Baselining

Treat a comparator baseline as versioned evidence, not a command that must run
for every candidate revision. Rerun the changed implementation; reuse an
unchanged comparator only after validating its provenance and correctness
fingerprint.

## Decide What To Run

1. Identify the changed implementation. Rerun that implementation.
2. Identify the comparison artifact and its identity packet.
3. Reuse the comparator when every identity field still matches.
4. Refresh the comparator once when any identity field changed or is absent.
5. Compare current correctness to the cached oracle/fingerprint before using
   cached timing.

Do not rerun static GeoPandas merely because vibeSpatial source changed. Do not
reuse old vibeSpatial timings as the candidate result.

## Baseline Identity Contract

Reuse a comparator only when all applicable fields match:

- workload/query source and shared fixture hash;
- dataset identity, scale, and authoritative fingerprint;
- comparator package lock and Python environment;
- machine identity when wall time is compared;
- warmup, repeat/statistic, timeout, and measurement boundary;
- relevant data conversion/encoding and storage location.

If an artifact lacks the required identity, refresh it once. Never silently
trust a legacy artifact. A different GPU alone does not invalidate a CPU-only
GeoPandas baseline, but a different CPU/host does when reporting wall speedup.

## Public Shootouts

Create or intentionally refresh a baseline:

```bash
uv run vsbench shootout benchmarks/shootout --scale 10k --repeat 3 \
  --json --output benchmark_results/baselines/shootout-10k.json
```

For later vibeSpatial revisions, rerun only vibeSpatial and reuse the validated
GeoPandas leg:

```bash
uv run vsbench shootout benchmarks/shootout --scale 10k --repeat 3 \
  --reuse-geopandas benchmark_results/baselines/shootout-10k.json \
  --json --output benchmark_results/current/shootout-10k.json
```

`vsbench` rejects reuse when the workload/measurement hash, scale, execution
settings, host identity, comparator-environment fingerprint, or correctness
fingerprint is missing or stale. Never bypass that validation. A refreshed
artifact becomes the next reusable baseline only after all fingerprints pass.

## SF100 And External Datasets

Reuse the comparator artifact only when its provenance records the same
SQL/query source, dataset fingerprint, encoding, CPU host, GeoPandas
environment, and measurement contract. Rerun current vibeSpatial in isolated
query processes and compare its outputs against the cached same-data oracle.
Refresh GeoPandas only after one of those comparator inputs changes or when the
user explicitly requests a new baseline.

Do not substitute SF1 correctness for SF100 same-data correctness when the
SF100 output is available. Do not compare totals until every query passes.

## Reporting

Report separately:

- baseline source, date, identity, and whether it was reused or refreshed;
- current source revision/worktree identity and measured times;
- correctness/fingerprint status for every workload;
- per-workload deltas and aggregate/geomean only after all workloads pass;
- fallbacks, offramps, compute transfers, capacity failures, and deferred lanes.

Never describe a cached comparator as newly measured. Never rerun a static
comparator to make a candidate gate look more complete.

## Verification

- `uv run pytest tests/test_bench_shootout.py -q`
- `uv run ruff check src/vibespatial/bench tests/test_bench_shootout.py`
- `uv run python scripts/check_docs.py --check`
