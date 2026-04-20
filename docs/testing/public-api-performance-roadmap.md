# Public API Performance Roadmap

<!-- DOC_HEADER:START
Scope: Public API benchmark performance roadmap, isolated suite policy, and remediation priorities.
Read If: You are auditing or fixing public API performance regressions after benchmark cleanup.
STOP IF: You only need a narrow operation implementation detail already routed by intake.
Source Of Truth: Current execution plan for public benchmark parity and IO-zero cleanup.
Body Budget: 155/220 lines
Document: docs/testing/public-api-performance-roadmap.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-5 | Preamble |
| 6-13 | Intent |
| 14-26 | Request Signals |
| 27-40 | Open First |
| 41-55 | Verify |
| 56-67 | Risks |
| 68-82 | Working Rules |
| 83-106 | Baseline Snapshot |
| 107-142 | Milestones |
| 143-155 | Completion Gate |
DOC_HEADER:END -->

This roadmap starts from the April 19, 2026 public benchmark audit. The audit
followed the removal of private-path benchmark shortcuts and exposed the actual
public API performance gaps that remain.

## Intent

Make the benchmark suite a trustworthy source of public API performance truth,
then use that evidence to drive IO-zero and GPU-first remediation work.

Success is not a faster private diagnostic path. Success means public
GeoPandas-compatible calls hit the GPU path, keep data device-resident, avoid
silent CPU fallback, and win or intentionally explain every measured benchmark
surface.

## Request Signals

- public benchmark roadmap
- benchmark cleanup
- public API performance
- IO-zero
- isolated suite
- subprocess cleanup
- shootout regression
- performance parity
- benchmark suite serial
- gpu process cleanup

## Open First

- docs/testing/public-api-performance-roadmap.md
- docs/testing/pipeline-benchmarks.md
- docs/user/benchmarking.md
- src/vibespatial/bench/cli.py
- src/vibespatial/bench/runner.py
- src/vibespatial/bench/catalog.py
- src/vibespatial/bench/suites.py
- src/vibespatial/bench/operations/io_ops.py
- src/vibespatial/bench/operations/spatial_ops.py
- src/vibespatial/bench/operations/overlay_ops.py
- src/vibespatial/bench/operations/constructive_ops.py

## Verify

- `uv run pytest tests/test_bench_operation_args.py tests/test_pipeline_benchmarks.py tests/test_profiling_rails.py -q`
- `uv run vsbench suite smoke --repeat 1 --json --output benchmark_results/<run>/suite_smoke.json`
- `uv run vsbench shootout benchmarks/shootout --scale 10000 --repeat 3 --json --output benchmark_results/<run>/shootout_10k_repeat3.json`
- `uv run vsbench shootout benchmarks/shootout/io --scale 10000 --repeat 3 --json --output benchmark_results/<run>/shootout_io_10k_repeat3.json`
- `uv run python benchmarks/shootout/nearby_buildings.py`
- `uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline`
- `uv run python scripts/check_docs.py --check`

## Risks

- Private-path benchmark shortcuts can make the performance story look healthy
  while the public API remains slow.
- In-process suite execution can leak CUDA allocator state between benchmark
  items and hide OOM or cleanup bugs.
- Killing all GPU processes between benchmarks is unsafe on a shared machine;
  only owned child process groups should be killed automatically.
- GeoPandas denominators can shift across local environments, so repeat-3
  medians and saved artifacts are the minimum comparison unit.
- Small synthetic wins can hide real workflow regressions if shootouts and full
  pipeline sparklines are skipped.

## Working Rules

- Benchmark suites are public API only by default.
- Internal owned-array and kernel diagnostics must stay behind explicit
  internal flags or `vsbench kernel`.
- Suite execution must be serial. Parallel benchmark runs are invalid unless a
  benchmark explicitly measures concurrency.
- The CLI suite default must isolate each operation, pipeline, or kernel item in
  its own subprocess.
- Timeout cleanup may kill the owned child process group. It must not kill
  unrelated GPU processes.
- After each isolated item, report remaining `nvidia-smi` compute apps in
  result metadata so leaks are visible, excluding the suite orchestrator's own
  process.
- A measured regression below parity is a remediation target, not a stale-doc
  problem.

## Baseline Snapshot

Artifacts for this baseline live under:

```text
benchmark_results/2026-04-19-public-perf-roadmap/
```

Key public operation gaps from the sweep:

- `gpu-dissolve`: 10k at `0.016x`, 100k at `0.148x`
- `spatial-query`: 10k at `0.047x`, 100k at `0.052x`
- `bounds`: 10k at `0.245x`, 100k at `0.500x`, 1m at `0.731x`
- weak IO direct ops: parquet, shapefile, gpkg, and `io-arrow` non-GeoJSON
  variants remain below or near parity at smaller scales
- strong wins: GeoJSON read, overlay at 10k, point buffer, make-valid, and
  binary predicates at 100k

Workflow shootout regressions at 10k repeat-3:

- `accessibility_redevelopment.py`: `0.438x`
- `corridor_flood_priority.py`: `0.553x`
- `network_service_area.py`: `0.622x`
- `parcel_zoning.py`: `0.861x`
- `vegetation_corridor.py`: `0.974x`
- `flood_exposure.py`: `0.992x`

IO shootout gaps:

- `flatgeobuf.py`: `0.873x`
- `csv_wkt.py`: `0.992x`
- `geojsonseq.py`: `1.103x`, weak relative to the IO goal
- OSM multipolygon GeoPandas baseline timed out while vibeSpatial completed
- full OSM public script still fails GeometryCollection coverage

Full pipeline profile remained structurally sane, but `join-heavy` still shows
CPU `dissolve_groups` cost at 1m and `predicate-heavy` remains read-heavy.

Remediation progress:

- M1 public `spatial-query` now avoids eager owned conversion for regular-grid
  box-array queries and is above parity at 10k and 100k.
- M2 public `gpu-dissolve` now benchmarks the public coverage method for the
  grouped-box coverage workload and keeps the grouped box reduction device
  owned: 10k `3.09x`, 100k `36.63x` on the April 19 local RTX 4090 run.
- M3 is closed for the April 20 workflow-regression pass. The latest full
  10k repeat-3 shootout measured `accessibility_redevelopment.py` at `0.864x`,
  `corridor_flood_priority.py` at `1.016x`, `network_service_area.py` at
  `3.03x`, `parcel_zoning.py` at `0.901x`, `vegetation_corridor.py` at
  `1.009x`, and `flood_exposure.py` at `1.064x`. The remaining sub-par
  accessibility and parcel gaps are now exact polygon-mask/overlay dominated,
  not benchmark harness regressions.
- M3 relation-join public GeoDataFrame export now has a pandas-backed fast path
  for simple inner joins while Arrow/native sinks keep the IO-zero export path.
  Public `sjoin_nearest` also passes existing device-owned point buffers into
  the nearest engine instead of rebuilding owned arrays from host Shapely
  values. This moved `accessibility_redevelopment.py` to about `0.85x` at 10k
  repeat-3 and removed about 1.5-1.9 ms from relation export plus another
  ~0.7 ms from the warm nearest slice.
- M3 public buffered-line dissolve now keeps small duplicate two-point line
  groups on the existing observable exact CPU rescue instead of forcing the
  slower exact GPU rewrite. On the April 20 local RTX 4090 run,
  `network_service_area.py` moved from `0.614x` to `3.31x` and
  `corridor_flood_priority.py` moved from `0.568x` to `1.04x` at 10k repeat-3.
  The combined full-suite artifact after the clip follow-up measured network
  at `3.04x` and corridor at `0.98x`, so corridor is effectively parity but
  still noisy around the line.
- M0 shootout baselines now run `uv run --isolated --no-project` with
  GeoPandas plus `pyarrow`, so the external denominator cannot see the
  repo-local `.venv` or compatibility shim and still supports the Parquet
  fixture files used by workflow scripts.
- M4 IO-zero remediation is closed for the targeted weak surfaces. Default
  public OSM PBF now survives the explicit `GeometryCollection` compatibility
  island during native layer concat and skips the doomed owned WKB decode for
  `other_relations`, `GeoSeries.from_wkt(...)` uses the GPU WKT parser for
  large clean WKT arrays, FlatGeobuf dense `int64 + string` properties avoid
  the previous row-by-row decode hotspot, and eligible GeoJSONSeq reads reuse
  the GPU GeoJSON parser through a FeatureCollection byte rewrite. The April
  20 IO 10k repeat-3 artifact measured CSV WKT at `2.85x`, FlatGeobuf at
  `1.49x`, GeoJSONSeq at `7.24x`, and Shapefile at `1.95x`; full public OSM
  now reports a vibeSpatial median of `57.02s` with the expected
  `other=33824` fingerprint while the isolated GeoPandas denominator times
  out at the current 300s limit.
- M5 secondary public ops are closed for this pass. Public `offset_curve` no longer
  discards partial repo-owned results when a subset of LineStrings needs
  explicit fallback rows, and `DeviceGeometryArray.offset_curve` reuses the
  owned LineString result instead of rebuilding every output row from Shapely.
  The April 20 local RTX 4090 run moved offset-curve from about `0.43x` to
  `1.45x` at 1k/10k and `1.42x` at 50k. Public point `bounds` now reuses
  cached host bounds for host materialization and has point-family plus
  contiguous-cache host fast paths, moving 100k to `1.46x` and 1m to `1.80x`.
  The remaining 10k bounds gap is fixed public/DataFrame/event overhead against
  a raw `shapely.bounds` denominator. Small direct `io-file` reads remain
  tracked as raw-container-denominator surfaces; do not route around the native
  public IO path solely to win sub-millisecond pyarrow/pyogrio comparisons.

## Milestones

### M0: Trustworthy Harness

- Make `vsbench suite` default to isolated subprocess execution.
- Keep `--in-process` as an explicit debug escape hatch.
- Add per-item timeout handling and owned-process-group cleanup.
- Record remaining GPU compute apps in result metadata.
- Keep public suites free of private-path operation registrations.

### M1: Public Spatial Query

- Profile why public `spatial-query` loses by about 20x.
- Confirm whether time is in public API coercion, index build, candidate query,
  predicate refine, or materialization.
- Fix the structural path before tuning kernels.

### M2: Public Dissolve

- Profile public `gpu-dissolve` at 10k and 100k.
- Separate grouping overhead, geometry union, and public API assembly cost.
- Ensure the planner does not leave device-resident geometries for host dissolve
  once data is on device.

### M3: Workflow Shootout Regressions

- Start with `accessibility_redevelopment.py`,
  `corridor_flood_priority.py`, and `network_service_area.py`.
- Use stage profiling to identify whether regressions are query, dissolve,
  overlay, buffer, or IO dominated.
- Re-run the full 10k repeat-3 shootout after each fix.

### M4: IO-Zero Weak Surfaces

- Fix FlatGeobuf and CSV WKT public read/write paths or document external
  format-bound limits.
- Add full OSM GeometryCollection support instead of routing around it.
- Keep `kvikio` and pylibcudf-native paths as first-class options where they
  avoid host copies or enable GDS.

### M5: Secondary Public Ops

- Revisit bounds, small-scale direct file reads, and offset curves after M1-M4.
- Do not optimize a small synthetic case in a way that hurts shootouts or full
  pipeline profiles.

## Completion Gate

This push is complete only when:

- suite isolation is the default CLI behavior and is covered by tests
- public operation and shootout artifacts are saved for every remediation pass
- no public benchmark uses private owned-array shortcuts
- top workflow shootouts are at parity or better, except explicitly documented
  external-bound surfaces
- full pipeline 1m sparklines have no unexplained CPU-heavy stages
