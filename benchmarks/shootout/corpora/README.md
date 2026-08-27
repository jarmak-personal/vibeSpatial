# External Corpus Shootouts

These shootouts exercise ordinary GeoPandas-compatible public workflows over
immutable external data. They are discovery canaries, not benchmark-specific
implementation targets.

Fetch and verify the first capsule:

```bash
uv run python scripts/manage_external_corpora.py fetch
uv run python scripts/manage_external_corpora.py verify
```

Run a profiled discovery sweep:

```bash
uv run vsbench shootout benchmarks/shootout/corpora \
  --scale 10k --repeat 3 --profile-mode full --json \
  --output benchmark_results/external-corpora/discovery-10k.json
```

`VSBENCH_SCALE` limits admitted rows after each source read. Downloads and
SHA-256 verification happen before the timed region. Override the ignored
local data directory with `VSBENCH_CORPUS_ROOT`.

The workload manifest is also the nearest `vsbench` identity root. Any change
to a workflow, shared helper, or pinned dataset identity invalidates cached
GeoPandas evidence for this capsule. Unrelated shootout changes do not.

Current capsule identity:
`5cf13540c71dfa91e70d84aee43a1a590b674c97338cb4473aaeb44446e9051d`.
A local repeat-one 10K validation passed all six v3 fingerprints on 2026-08-26.
Its single samples are correctness evidence, not stable timing baselines.
Earlier measurements are archived in the discovery report and must not be
reused because they predate the complete result fingerprint.

Rules:

- timed work uses public GeoPandas-compatible APIs only;
- no private vibeSpatial imports, planner hints, or algorithm selectors;
- default automatic dispatch is the performance contract;
- ordered SHA-256 fingerprints cover exact schema, index, IDs, nulls, geometry
  topology, and row association; constructive serialization order is
  normalized, metrics use the fp64 seven-significant-digit contract, and
  coordinates use a tighter eleven digits before timings are compared;
- static GeoPandas legs are reused after the identity checks pass;
- a failure is evidence and must not be hidden by changing the workload.
