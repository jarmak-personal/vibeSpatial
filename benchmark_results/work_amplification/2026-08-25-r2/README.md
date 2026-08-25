# Work-amplification R2 evidence

Current-revision counterfactuals and the component-first Q11 falsifier collected
on `picard-4090`, rooted at revision `e8e7f22` and exact dirty source identity
`708b2b41`. The custom Q11/Q12 outputs now embed contemporaneous source,
worktree, dataset, machine/environment, and measurement identity.

- `RESULTS.md` is the human-readable decision record.
- `CUSTOM_ARTIFACT_IDENTITY.json` indexes the fresh identified Q11/Q12
  decision artifacts and isolates the older one-zone smoke files as
  retrospective-only evidence.
- `redevelopment_paged_1m_current.json` is the current public paged
  constructive control; the missing comparator is intentional and only the
  vibeSpatial candidate/fingerprint is used.
- `q12_indexed_current.json` and `q12_dense_filter_current.json` are fresh
  identified current-worktree Q12 arms.
- `q11_parent_z5.json` and `q11_component_z5.json` are lean one-batch/all-zone
  comparisons.
- `q11_parent_z5_profile.json` and `q11_component_z5_profile.json` are their
  separate schema-2 attribution runs.
- one-zone and 1K files are smoke/protected-shape evidence.
- `shootout_10k_final.json` is the final 14/14 exact public regression floor
  using the validated static comparator.
- `shootout_1m_final.json` is the current candidate-only 1M diagnostic: 13
  strict-native workflows complete in 158.427s total and the corridor workflow
  retains its explicit strict-native off-ramp. Its intentionally invalid
  comparator supplies no correctness or speedup claim.
- `pipelines_full_final.json` and `.html` retain the mandatory full audit: 22
  successful pipelines, 2 deferred raster pipelines, and zero compute D2H,
  compute materialization, or fallback.
- `sf100_vibespatial_lean_cold1.json` and
  `sf100_vibespatial_counter_telemetry_cold1.json` are the current 12-query
  SF100 runs at 464.88s and 464.32s. `SF100_RUN_IDENTITY.json` binds their
  source, data, environment, commands, and normalized-result directories.
- `q11_component_first.py` and `redevelopment_paged_control.py` preserve the
  measured experiment entrypoints; neither changes production dispatch.
- `q11_component_first_identified.py` runs the preserved Q11 implementation in
  an isolated child process and embeds complete identity without changing the
  timed scope.
- `q12_dense_experiment.py` is the missing identified Q12 entrypoint. It pins
  the exact dense implementation source and embeds source/worktree, dataset,
  environment, and measurement identity in every fresh JSON output.

Reproduce Q12 in two isolated cold processes, indexed first because it writes
the retained points and correctness reference consumed by the dense arm:

```bash
uv run python benchmark_results/work_amplification/2026-08-25-r2/q12_dense_experiment.py indexed
uv run python benchmark_results/work_amplification/2026-08-25-r2/q12_dense_experiment.py dense
```

The commands reproduce the current-worktree measurements and retain
`q12_baseline_result.csv`,
`q12_dense_filter_result.csv`, and `q12_retained_points.npz` in this capsule.

Fresh Q11 arms use the identified wrapper. For the decision pair:

```bash
uv run python benchmark_results/work_amplification/2026-08-25-r2/q11_component_first_identified.py --variant parent --zone-frames 5 --output benchmark_results/work_amplification/2026-08-25-r2/q11_parent_z5.json
uv run python benchmark_results/work_amplification/2026-08-25-r2/q11_component_first_identified.py --variant component --zone-frames 5 --output benchmark_results/work_amplification/2026-08-25-r2/q11_component_z5.json
```

Append `--profile` and use the corresponding `*_profile.json` output for the
separate attribution arms.

The paged constructive control imports the shared shootout fixture module
`_data`, so its isolated vibeSpatial subprocess requires the shootout directory
on `PYTHONPATH`. The canonical current-source command is:

```bash
PYTHONPATH=/home/picard/repos/vibeSpatial/benchmarks/shootout \
  uv run vsbench shootout \
  benchmark_results/work_amplification/2026-08-25-r2/redevelopment_paged_control.py \
  --scale 1M --repeat 1 --no-warmup --baseline-python /bin/true \
  --timeout 600 --profile-mode off --json \
  --output benchmark_results/work_amplification/2026-08-25-r2/redevelopment_paged_1m_current.json
```

The invalid `/bin/true` comparator used for vibeSpatial-only controls causes a
non-zero command exit after the current result is safely written. It is not a
vibeSpatial workload failure and supplies no comparator timing claim.
