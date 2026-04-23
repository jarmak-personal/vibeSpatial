# Public API Performance Roadmap

<!-- DOC_HEADER:START
Scope: Public API benchmark performance roadmap, isolated suite policy, and remediation priorities.
Read If: You are auditing or fixing public API performance regressions after benchmark cleanup.
STOP IF: You only need a narrow operation implementation detail already routed by intake.
Source Of Truth: Current execution plan for public benchmark parity and IO-zero cleanup.
Body Budget: 274/300 lines
Document: docs/testing/public-api-performance-roadmap.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-6 | Preamble |
| 7-16 | Intent |
| 17-29 | Request Signals |
| 30-43 | Open First |
| 44-53 | Verify |
| 54-68 | Risks |
| 69-85 | Working Rules |
| 86-197 | Baseline Snapshot |
| 198-262 | Milestones |
| 263-274 | Completion Gate |
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
- API coverage can look healthy while reusable physical-plan shapes such as
  semijoin, anti-join, many-few overlay, and grouped dissolve remain slow.

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
- M3 public buffered-line dissolve now treats device-resident two-point line
  buffers as a GPU physical shape: rebuild buffers from source lines on device,
  reduce with exact GPU union, and do not route through the small exact CPU
  rescue. Multi-vertex line-buffer dissolve remains on the observable exact
  CPU rescue at 10k scale because the current pairwise GPU reducer is slower;
  the real GPU gap is an n-ary corridor-union physical operator, not another
  workflow-specific route.
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
- M6 starts from the April 21 physical-plan audit. Four new real-world public
  shootouts matched GeoPandas fingerprints at 10k repeat-3, so correctness
  generalized. Performance did not: emergency response catchments measured
  `0.015x`, retail trade-area screening `0.060x`, insurance flood screening
  `0.273x`, and habitat compliance `0.229x`. Treat these as canaries for
  reusable physical shapes, not as workflow-specific optimization targets.
- M6 artifacts now include statement-level `timed_stages`,
  `stage_totals_by_tag`, `stage_totals_by_backend`, `hotpath_total_seconds`,
  and `composition_overhead_seconds`. The April 22 10k repeat-3 canaries
  confirmed the expected shape after removing two residency-breaking dispatch
  heuristics: emergency response improved to `0.646x` with `52.6ms` hotpath
  inside `151.0ms` profile time, and retail trade-area screening held `2.12x`
  with `90.2ms` hotpath inside `295.8ms` profile time. Both canaries now show
  all-GPU execution traces with zero fallbacks and zero offramps. The next
  remediation target is reducing public composition/materialization overhead
  for reusable grouped-reduce, spatial-join, clip, and many-few-overlay shapes,
  not IO or hidden CPU fallback.

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

### M6: Real-World Physical Plan Coverage

- Wire `vsbench shootout` artifacts to expose stage timing, actual backend,
  fallback events, materialization/transfer counts, and top hotpath stages.
- Tag real-world shootouts by physical shapes: semijoin, anti-semijoin,
  many-few overlay, mask clip, grouped geometry reduce, and
  area-filter-after-overlay.
- Add operation-vs-operation floor checks for each dominant workflow stage so
  regressions can be separated into stage-floor gaps versus composition
  overhead.
- Profile emergency response and retail trade-area first because they expose
  the largest reusable shape gaps.
- Fix shared physical shapes before workflow-specific scripts. A workflow fix
  is incomplete unless it improves or explains the named reusable shape.
- Treat high `composition_overhead_ratio` with GPU-only `trace_summary` as a
  first-class regression signal. It means kernels are not the limiting factor;
  public API statement composition, scalar synchronization, host-visible
  indexing, or geometry-frame assembly is.

## Completion Gate

This push is complete only when:

- suite isolation is the default CLI behavior and is covered by tests
- public operation and shootout artifacts are saved for every remediation pass
- no public benchmark uses private owned-array shortcuts
- top workflow shootouts are at parity or better, except explicitly documented
  external-bound surfaces
- real-world shootouts have physical-plan artifacts that explain every
  sub-par result by reusable shape
- full pipeline 1m sparklines have no unexplained CPU-heavy stages
