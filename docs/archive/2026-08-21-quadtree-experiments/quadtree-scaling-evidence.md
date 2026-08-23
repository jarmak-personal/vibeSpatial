# Archived Adaptive Point-Quadtree Scaling Evidence

<!-- DOC_HEADER:START
Scope: Archived RTX 4090 synthetic scale evidence for the rejected adaptive point-quadtree provider.
Read If: You are reviewing the synthetic evidence that preceded the full SF100 physical-shape experiment.
STOP IF: You need current production selection behavior or current performance claims.
Source Of Truth: Historical measurement summary; immutable raw experiment artifacts remain authoritative.
Body Budget: 114/130 lines
Document: docs/archive/2026-08-21-quadtree-experiments/quadtree-scaling-evidence.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-6 | Preamble |
| 7-13 | Intent |
| 14-21 | Request Signals |
| 22-30 | Open First |
| 31-37 | Verify |
| 38-47 | Risks |
| 48-62 | Measurement Contract |
| 63-84 | Results |
| 85-98 | Decisions |
| 99-114 | Production Follow-up |
DOC_HEADER:END -->

Status: archived evidence. These synthetic scale results motivated the
experiment but did not translate into a winning warmed SF100 Q11 execution
region; they are not production-selection evidence.

## Intent

Measure how the source-derived vS point quadtree scales beyond its first 1M
canary. Separate reusable index construction from warm public execution,
compare the existing Morton and production-auto controls, and refuse scales
whose observed memory slope cannot fit the active device safely.

## Request Signals

- adaptive quadtree scaling
- 10M, 100M, or 1B point-region benchmark
- point-region memory slope
- Morton versus quadtree
- dense point-grid scale failure

## Open First

- `docs/dev/libcuspatial-quadtree-pip-benchmark-plan.md`
- `docs/dev/point-region-execution-evidence.md`
- `docs/decisions/0046-gpu-physical-workload-shape-contracts.md`
- `scripts/profile_point_region.py`
- `src/vibespatial/spatial/point_quadtree_index.py`
- `src/vibespatial/spatial/point_grid_index.py`

## Verify

- `uv run python scripts/check_docs.py --check`
- `uv run ruff check scripts/profile_point_region.py`
- rerun the script with the parameters recorded in each raw artifact before
  changing a result

## Risks

- The quadtree lane is benchmark-forced through a public operator; it is not a
  production automatic-selection claim.
- This skew canary isolates a favorable adaptive-partition shape and does not
  replace the protected corpus or H200 gate.
- Timed wall ends when the public native result returns. Mechanical oracle
  exports and their D2H transfers are outside the timed region.
- A projected memory result is an admission decision, not a measured timing.

## Measurement Contract

Evidence was collected on 2026-08-20 on an RTX 4090 with 25,250,627,584 bytes
of VRAM. The final production source fingerprint is
`a344f24f45446c221480e771934006fa796ae2ea39530f99b1b5140446fb9719` at Git
revision `5422e3c0ddf1b0d36b02a70f07be1e4d86a2516f`, including the untracked
quadtree sources. Inputs pass through public GeoArrow GeoParquet IO and timing
enters only through `SpatialIndex.query_pair_aggregate(predicate="contains")`.

The fixture places all but four points in `[0, 1] x [0, 1]`, stretches the
global extent to +/-1024, and queries 64 disjoint boxes. Every completed run
matched both vectorized Shapely count oracles and recorded zero fallbacks.
Cold is the first call with index construction; warm is the median of calls two
and three. Memory is allocator live and process peak after the third call.

## Results

| Points | Auto warm | Morton warm | Quadtree cold | Quadtree warm | Warm throughput | Live / peak | Quadtree index |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10K | 2.98 ms | 2.96 ms | 43.47 ms | 1.83 ms | 5.45 Mpts/s | 0.002 / 0.003 GiB | 0.08 MiB |
| 1M | 191.32 ms | 14.32 ms | 42.60 ms | 2.01 ms | 496 Mpts/s | 0.206 / 0.265 GiB | 7.68 MiB |
| 10M | OOM in grid build | 126.76 ms | 92.78 ms | 6.86 ms | 1.46 Bpts/s | 2.014 / 2.601 GiB | 76.35 MiB |
| 50M | not run | not run | 321.80 ms | 31.73 ms | 1.58 Bpts/s | 10.071 / 12.539 GiB | 381.52 MiB |

At 10M the dense-grid build passed its vS admission estimate, then CuPy CUB
histogram scratch requested one 12.000 GiB allocation and exceeded the 21.165
GiB RMM ceiling. The raw stderr and exit status are preserved with the runs.
Quadtree warm execution is 7.1x faster than Morton at 1M and 18.5x at 10M.

Observed 1M, 10M, and 50M memory fits give 216.20 live and 268.29 peak bytes
per point plus small intercepts. The resulting admissions are:

| Requested scale | Projected live / peak | RTX 4090 decision | H200 implication |
|---:|---:|---|---|
| 100M | 20.14 / 25.04 GiB | not admitted; peak exceeds physical VRAM | memory-feasible hypothesis, not measured |
| 1B | 201.35 / 249.92 GiB | not admitted | does not fit the measured 143,771 MiB H200 either |

## Decisions

- Retain the quadtree as a serious V2 candidate: after saturation its warm
  curve is close to linear and its persistent index converges to 8 bytes per
  point.
- Do not raise the 16-bit depth blindly. Leaf count stays at 1,028 while maximum
  occupancy grows from 1,024 at 1M to 48,841 at 50M; the current kernel remains
  fast, but this is a visible resolution ceiling to test on other shapes.
- Fix dense-grid construction admission before claiming production-auto scale
  safety; retained arrays omit the dominant CUB histogram scratch request.
- Run 100M on H200/H100-class memory when available. Reaching 1B under this
  public paired shape requires streaming or sharding rather than a larger
  in-core quadtree.

## Production Follow-up

The original auto results above remain immutable prototype evidence. The
production repair replaced `cupy.bincount` with sorted-key lower-bound counts
and added complete preflight/token/guard contracts. On the same RTX 4090 and
the same 10M extent-skew fixture, current automatic grid now completes in
56.15 s cold and 262.11 ms warm with zero fallback instead of OOMing. Across
five calls, fixed production quadtree completes in 135.06 ms cold and 10.76 ms
steady-state; forced Morton takes 195.62 ms cold and 126.67 ms steady-state.
Quadtree is 31.0% faster cold and 11.78x faster warm. Both count oracles match
for every run. The automatic result remains grid because production policy
does not override a fully admitted provider with a benchmark-derived winner
model.

Current raw artifacts are under
`benchmark_results/point_region/production_quadtree/`.
