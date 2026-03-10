# Binary Predicate Refine Pipeline

<!-- DOC_HEADER:START -->
> [!IMPORTANT]
> This block is auto-generated. Edit metadata in `docs/doc_headers.json`.
> Refresh with `uv run python scripts/check_docs.py --refresh` and validate with `uv run python scripts/check_docs.py --check`.

**Scope:** Exact binary-predicate refine strategy, coarse-filter staging, and GeoPandas adapter policy.
**Read If:** You are changing exact predicate kernels, binary predicate dispatch, or GeoPandas predicate integration.
**STOP IF:** You already have the binary-predicate engine open and only need local implementation detail.
**Source Of Truth:** Phase-4 exact binary-predicate architecture policy before join assembly.
**Body Budget:** 137/220 lines
**Document:** `docs/architecture/binary-predicates.md`

**Section Map (Body Lines)**
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-13 | Request Signals |
| 14-21 | Open First |
| 22-32 | Verify |
| 33-38 | Risks |
| 39-43 | Intent |
| 44-73 | Decision |
| 74-91 | Current Surface |
| 92-104 | CCCL Mapping |
| 105-118 | Host Crossover |
| 119-137 | Consequences |
<!-- DOC_HEADER:END -->

## Request Signals

- exact predicate
- binary predicate
- intersects
- within
- contains
- covers
- cccl
- geopandas predicate

## Open First

- docs/architecture/binary-predicates.md
- docs/architecture/robustness.md
- src/vibespatial/binary_predicates.py
- src/vibespatial/kernels/predicates/binary_refine.py
- src/vibespatial/_vendor/geopandas/array.py

## Verify

- `uv run pytest tests/test_binary_predicates.py tests/test_geopandas_binary_predicates.py`
- `uv run pytest tests/test_geopandas_dispatch.py -q`
- `uv run pytest tests/upstream/geopandas/tests/test_geom_methods.py -k "contains or within or intersects or covers or covered_by or touches or crosses"`
- `uv run pytest tests/test_gpu_binary_predicates.py -q --run-gpu`
- `VIBESPATIAL_STRICT_NATIVE=1 uv run pytest tests/upstream/geopandas/tests/test_array.py -k "test_predicates_vector_scalar or test_predicates_vector_vector or test_chaining or test_raise_on_bad_sizes" -q`
- `uv run python scripts/benchmark_gpu_predicates.py --scale 100000`
- `uv run python scripts/check_architecture_lints.py --all`
- `uv run python scripts/check_docs.py --check`

## Risks

- A monolithic exact kernel would hide the compaction boundaries needed for CCCL and later GPU fallback staging.
- Host-side small-array overhead can erase any value from coarse filtering if the adapter never short-circuits.
- Predicate semantics will drift from GeoPandas if the vendored array path and the internal kernel path diverge.

## Intent

Define the first exact binary-predicate refine contract and route the vendored
GeoPandas binary predicate surface through it.

## Decision

Use a staged exact-refine engine:

- coarse bounding-box relation first
- compact surviving candidate rows
- run exact predicate only on the compacted candidate set
- restore row order directly into the final boolean output

This stage pattern applies to `intersects`, `within`, `contains`, `covers`,
`covered_by`, `touches`, `crosses`, `contains_properly`, `overlaps`, and
`disjoint`.

The current implementation now has a real but narrow CUDA path:

- repo-owned exact predicate kernels register CPU and GPU variants
- GPU refine currently covers point-centric candidate rows only:
  point/point, point/line, point/polygon, and inverse orientations
- uniform point<->polygon and point<->multipolygon batches now reuse the
  point-in-polygon device bounds mask so candidate-row compaction stays on the
  GPU instead of round-tripping through host `flatnonzero`
- explicit GPU requests still fail loudly when candidate rows require an
  unsupported geometry-pair refine
- `auto` mode records explicit CPU fallback on owned-array inputs when the
  candidate geometry mix exceeds current GPU support
- the vendored GeoPandas `GeometryArray` binary predicate path now routes
  supported predicates through the repo-owned exact engine and records a real
  dispatch event for owned CPU vs owned GPU selection instead of emitting an
  immediate fallback event

## Current Surface

- `evaluate_binary_predicate(...)`: stable internal exact-refine engine
- point-centric GPU refine now keeps owned-array inputs in buffer form until a
  real Shapely fallback is required, avoiding the previous eager materialization
- `evaluate_geopandas_binary_predicate(...)`: GeoPandas-facing adapter that
  routes supported predicates through the owned exact-refine engine and lets the
  runtime choose owned CPU vs owned GPU execution without a fallback event
- `*_exact(...)` predicate wrappers in `src/vibespatial/kernels/predicates/binary_refine.py`

Coarse bounding-box relations:

- `intersects` family: intersecting bounds are candidates
- `contains` family: containing bounds are candidates
- `within` family: contained bounds are candidates
- `disjoint`: non-intersecting bounds are immediate `True`, intersecting bounds
  are candidates

## CCCL Mapping

The intended GPU implementation should preserve the same stages:

- bounds generation and relation mask: transform-style compare over min/max columns
- candidate compaction: CCCL `DeviceSelect`
- candidate row reorder: gather/scatter on compacted row ids
- exact predicate pass: geometry-family-specific kernels over the compacted rows
- result restoration: scatter exact results back into the full output buffer

This is why the current host path is structured around masks, compaction, and
explicit candidate-row tracking instead of direct all-row exact calls.

## Host Crossover

Current host measurements show direct vectorized Shapely still beats the staged
CPU refine engine on the workloads measured so far, including a `20K`-row
`intersects` run with `80%` coarse rejects.

So the vendored GeoPandas adapter stays on direct host execution for now.

The staged exact-refine engine remains the owned-kernel contract for:

- explicit kernel work in `src/vibespatial`
- future non-point GPU predicate variants
- performance experiments that justify a later host crossover

## Consequences

- exact predicate semantics are centralized behind one internal API
- point-centric GPU refine is now fast enough to beat the repo CPU staged engine
  materially on large aligned batches
- the point/polygon `contains` benchmark now reports both cold and warm GPU
  timings because first-call NVRTC/JIT cost is large enough to obscure the
  steady-state kernel throughput
- warm point/polygon `contains` is now materially faster than direct vectorized
  Shapely on the aligned box workload used by the repo benchmark, while cold
  first-call latency is still dominated by compilation
- targeted upstream GeoPandas predicate tests now exercise repo-owned predicate routing
- strict-native GeoPandas predicate tests can now stay on the owned predicate
  path for the covered binary methods instead of failing immediately on a
  fallback event
- the future GPU path can replace the exact pass and compaction internals without
  changing the adapter contract
- `o17.4.3` can build join/query assembly on top of one predicate engine instead
  of many ad hoc Shapely calls
