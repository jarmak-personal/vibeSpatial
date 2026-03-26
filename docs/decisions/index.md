# Decision Log

<!-- DOC_HEADER:START
Scope: Architecture decision log and index of accepted or superseded ADRs.
Read If: You are resolving, revisiting, or querying a design decision.
STOP IF: You already have the specific ADR open and do not need the index.
Source Of Truth: Decision log for architecture choices.
Body Budget: 80/220 lines
Document: docs/decisions/index.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-4 | Preamble |
| 5-8 | Intent |
| 9-16 | Request Signals |
| 17-25 | Open First |
| 26-30 | Verify |
| 31-35 | Risks |
| 36-80 | Decisions |
DOC_HEADER:END -->

Use this index to find accepted architecture decisions.

## Intent

Track architecture decisions in a stable, agent-discoverable format.

## Request Signals

- adr
- decision log
- architecture decision
- design decision
- superseded design

## Open First

- docs/decisions/index.md
- scripts/new_decision.py
- docs/architecture/runtime.md
- docs/decisions/0001-mixed-geometries.md
- docs/decisions/0002-dual-precision-dispatch.md
- docs/decisions/0003-null-empty-geometry-contract.md

## Verify

- `uv run pytest tests/test_decision_log.py`
- `uv run python scripts/check_docs.py --check`

## Risks

- Decisions can drift from implemented code if follow-up changes do not update the log.
- Buried or unindexed ADRs make agents re-litigate settled design choices.

## Decisions

| ADR | Status | Date | Title |
|---|---|---|---|
<!-- DECISION_ROWS -->
| `ADR-0001` | accepted | 2026-03-10 | [Mixed Geometry Storage And Execution](docs/decisions/0001-mixed-geometries.md) |
| `ADR-0002` | accepted | 2026-03-11 | [Dual Precision Dispatch](docs/decisions/0002-dual-precision-dispatch.md) |
| `ADR-0003` | accepted | 2026-03-11 | [Null And Empty Geometry Contract](docs/decisions/0003-null-empty-geometry-contract.md) |
| `ADR-0004` | accepted | 2026-03-11 | [Predicate And Overlay Robustness Strategy](docs/decisions/0004-robustness-strategy.md) |
| `ADR-0005` | accepted | 2026-03-11 | [Device Residency And Transfer Visibility](docs/decisions/0005-device-residency-policy.md) |
| `ADR-0006` | accepted | 2026-03-11 | [Per-Kernel Dispatch Crossover Policy](docs/decisions/0006-dispatch-crossover-policy.md) |
| `ADR-0007` | accepted | 2026-03-11 | [Probe-First Adaptive Runtime](docs/decisions/0007-probe-first-adaptive-runtime.md) |
| `ADR-0008` | accepted | 2026-03-11 | [Owned Geometry Buffer Schema](docs/decisions/0008-owned-geometry-buffer-schema.md) |
| `ADR-0009` | accepted | 2026-03-11 | [Staged Fusion Strategy](docs/decisions/0009-fusion-strategy.md) |
| `ADR-0010` | accepted | 2026-03-11 | [Staged Point Predicate Pipeline](docs/decisions/0010-point-predicate-pipeline.md) |
| `ADR-0011` | accepted | 2026-03-11 | [Staged Binary Predicate Refine Pipeline](docs/decisions/0011-binary-predicate-refine-pipeline.md) |
| `ADR-0012` | accepted | 2026-03-11 | [Spatial Query And Join Assembly](docs/decisions/0012-spatial-query-assembly.md) |
| `ADR-0013` | accepted | 2026-03-11 | [Explicit CPU Fallback Events](docs/decisions/0013-explicit-fallback-events.md) |
| `ADR-0014` | accepted | 2026-03-11 | [Staged Segment Intersection Primitives](docs/decisions/0014-segment-intersection-primitives.md) |
| `ADR-0015` | accepted | 2026-03-11 | [Rectangle Clip As First Constructive Fast Path](docs/decisions/0015-rectangle-clip-fast-path.md) |
| `ADR-0016` | accepted | 2026-03-11 | [Shared Overlay Reconstruction Plan](docs/decisions/0016-overlay-reconstruction-plan.md) |
| `ADR-0017` | accepted | 2026-03-11 | [Dissolve Grouped Union Pipeline](docs/decisions/0017-dissolve-grouped-union-pipeline.md) |
| `ADR-0018` | accepted | 2026-03-11 | [Stroke Kernel Seam](docs/decisions/0018-stroke-kernel-seam.md) |
| `ADR-0019` | accepted | 2026-03-11 | [Compact Invalid Row Make Valid](docs/decisions/0019-compact-invalid-row-make-valid.md) |
| `ADR-0020` | accepted | 2026-03-11 | [Public API Dispatch Boundary](docs/decisions/0020-public-api-dispatch-boundary.md) |
| `ADR-0021` | accepted | 2026-03-11 | [IO Support Matrix](docs/decisions/0021-io-support-matrix.md) |
| `ADR-0022` | accepted | 2026-03-11 | [GeoArrow, GeoParquet, and WKB Bridges](docs/decisions/0022-geoarrow-geoparquet-bridges.md) |
| `ADR-0023` | accepted | 2026-03-11 | [Hybrid File Format Adapters](docs/decisions/0023-hybrid-file-format-adapters.md) |
| `ADR-0024` | accepted | 2026-03-11 | [Staged GPU-Native IO Execution Model](docs/decisions/0024-staged-gpu-native-io-execution.md) |
| `ADR-0025` | accepted | 2026-03-11 | [Zero-Copy GeoArrow Adoption](docs/decisions/0025-zero-copy-geoarrow-adoption.md) |
| `ADR-0026` | accepted | 2026-03-11 | [GeoParquet Scan Engine](docs/decisions/0026-geoparquet-scan-engine.md) |
| `ADR-0027` | accepted | 2026-03-11 | [Family-Specialized GeoArrow Codecs](docs/decisions/0027-family-specialized-geoarrow-codecs.md) |
| `ADR-0028` | accepted | 2026-03-11 | [Staged WKB Bridge](docs/decisions/0028-staged-wkb-bridge.md) |
| `ADR-0029` | accepted | 2026-03-11 | [Staged GeoJSON Ingest](docs/decisions/0029-staged-geojson-ingest.md) |
| `ADR-0030` | accepted | 2026-03-11 | [Batch-First Shapefile Ingest](docs/decisions/0030-batch-first-shapefile-ingest.md) |
| `ADR-0031` | accepted | 2026-03-11 | [Determinism And Reproducibility Policy](docs/decisions/0031-determinism-and-reproducibility.md) |
| `ADR-0032` | accepted | 2026-03-11 | [ADR-0032: Point-in-Polygon GPU Utilization Diagnosis](docs/decisions/0032-point-in-polygon-gpu-utilization-diagnosis.md) |
| `ADR-0033` | accepted | 2026-03-12 | [ADR-0033: GPU Primitive Dispatch Rules](docs/decisions/0033-gpu-primitive-dispatch-rules.md) |
| `ADR-0034` | accepted | 2026-03-14 | [ADR-0034: CCCL make_* Pre-Compilation and Warmup Strategy](docs/decisions/0034-cccl-precompile-warmup-strategy.md) |
| `ADR-0035` | deferred | 2026-03-14 | [ADR-0035: Geodesic Distance Kernel and CRS-Aware Dispatch](docs/decisions/0035-geodesic-distance-kernel.md) |
| `ADR-0036` | accepted | 2026-03-16 | [Index-Array Boundary Attribute Model](docs/decisions/0036-index-array-boundary-attribute-model.md) |
| `ADR-0037` | deferred | 2026-03-16 | [ADR-0037: GPU Voronoi Diagram Kernel](docs/decisions/0037-gpu-voronoi-diagram-kernel.md) |
| `ADR-0038` | accepted | 2026-03-17 | [ADR-0038: GPU Byte-Classification GeoJSON Parser](docs/decisions/0038-gpu-byte-classify-geojson-parser.md) |
| `ADR-0039` | accepted | 2026-03-20 | [ADR-0039: Provenance Tagging and Rewrite System](docs/decisions/0039-provenance-tagging-rewrites.md) |
| `ADR-0040` | accepted | 2026-03-25 | [ADR-0040: Tiered GPU Memory Pool (RMM)](docs/decisions/0040-tiered-gpu-memory-pool.md) |
