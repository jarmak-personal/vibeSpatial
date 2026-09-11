# STRtree Backend Landing Ledger

<!-- DOC_HEADER:START
Scope: Landing-time lazy spatial index admission, cache promotion and validation evidence.
Read If: You are separating nearest bounds from flat-index construction or reviewing the final landing checks.
STOP IF: You need the earlier generalized STR algorithm experiments.
Source Of Truth: Chronological nearest backend admission corrections during landing.
Body Budget: 87/160 lines
Document: docs/dev/strtree-backend-landing-ledger.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-8 | Intent |
| 9-15 | Request Signals |
| 16-23 | Open First |
| 24-28 | Verify |
| 29-34 | Risks |
| 35-44 | E66 / Initial Commit and Host Environment |
| 45-62 | E67 / Backend Admission Finding |
| 63-87 | E68-E72 / Final Correction and Evidence |
DOC_HEADER:END -->

## Intent

Record the landing-time correction that separates nearest bounds from full
flat-index construction. Continue the STRtree integration ledger without
rewriting its historical measurements or review findings.

## Request Signals

- nearest lazy index admission
- geometry-bounds native index
- flat backend promotion
- nearest pipeline scalar fences

## Open First

- docs/dev/strtree-integration-ledger.md
- docs/architecture/spatial-index-backends.md
- src/vibespatial/api/sindex.py
- src/vibespatial/api/_native_metadata.py
- tests/test_pipeline_benchmarks.py

## Verify

- `uv run pytest tests/test_packed_str_index.py tests/test_pipeline_benchmarks.py tests/test_geopandas_fallbacks.py --run-gpu -q`
- `uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline`

## Risks

- Bounds alone must never masquerade as a complete flat-Morton layout.
- Index promotion must preserve geometry lineage, stream readiness and caches.
- Raising a transfer budget can conceal unnecessary backend construction.

## E66 / Initial Commit and Host Environment

Commit `c069a50` passed the consolidated review and seven deterministic gates.
The first push failed after an NVIDIA package upgrade installed 580.178.04
userspace while the running kernel retained 580.173.02. Default NVML and CUDA
initialization failed independently of vibeSpatial. Matching 580.173.02 libraries
from an official Ubuntu package restored GPU detection and computation when
selected through a process-local library path. No host installation or hook
was changed. The ordinary pre-push checks then ran with the GPU visible.

## E67 / Backend Admission Finding

The contract gate exposed an obsolete public nearest dispatch assertion and
an eight-byte producer budget inherited from the prior nearest engine.
The inner producer used the documented seven STR fences, totaling 14 bytes.
The right producer also downloaded five flat-index planning scalars (40 bytes).
Review traced that extra packet to constructing a complete flat-Morton layout
solely to obtain a NativeSpatialIndex carrier. Packed STR consumes geometry
and feature bounds, so this construction was unnecessary nearest-only work.

A direct 1K bounds-only probe preserved every pair and distance while reducing
the right producer from 54 to 14 bytes and omitting Morton keys/order.
The correction introduces explicit bounds-native state with lazy promotion
for consumers that need a complete flat layout. Both pipeline directions now
reject the extra flat summary and validate every permitted STR fence by name,
item count and byte size. Public dispatch tests require the GPU producer and
the native relation export, while checking explicit CPU selection without GPU.

## E68-E72 / Final Correction and Evidence

The native carrier now distinguishes `geometry-bounds` from complete flat state.
Six regressions cover host/WKB ingress, both nearest directions, orphan reuse,
promotion/cache retention, direct native consumers and cross-stream promotion.
The last case exposed a missing producer-event wait before public flat
construction; promotion now establishes that dependency before the build.

Final native/contract suite: 525 pass, one optional SciPy skip. Upstream
sindex/sjoin: 416 pass, 59 skips, one existing xfail. Native/IO selection: 30
pass. All 83 packed STR cases pass memcheck with zero errors, including the
two new stream cases. The earlier independent libcudf initcheck finding remains
documented; no unfiltered clean-initcheck claim is made.

All 28 refreshed public cases match their immutable CPU pair/distance oracles.
Holed polygons: 100K reused 121.225 ms; 1M reused 677.595 ms. Full MA segments
reused 753.922 ms. These remain screening trials with explicit first-query and
WKB ingress costs. E71 full profile: 22 runnable cases, two raster deferrals,
102 stages reviewed, zero fallbacks. Maximum stage is 84.817 ms; no stage over
5 ms exceeds 1.35x its E59 timing. The results document retains every 1M stage.

Current results and verification are in `docs/testing/strtree-integration-results.md`
and `docs/testing/strtree-backend-landing-evidence.json`. The prior integration
evidence remains unchanged so its measurements and resolved findings remain
auditable alongside the landing correction.
