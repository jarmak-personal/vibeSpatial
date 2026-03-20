# Adaptive Runtime

<!-- DOC_HEADER:START
Scope: Probe-first adaptive planning, monitoring inputs, and chunk-boundary runtime decisions.
Read If: You are changing variant selection, NVML monitoring, adaptive chunking, or runtime planning.
STOP IF: Your task already has a settled adaptive-runtime contract and only needs implementation detail.
Source Of Truth: Phase-1 adaptive runtime policy before broader kernel work lands.
Body Budget: 76/240 lines
Document: docs/architecture/adaptive-runtime.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-4 | Preamble |
| 5-9 | Intent |
| 10-18 | Request Signals |
| 19-26 | Open First |
| 27-31 | Verify |
| 32-37 | Risks |
| 38-44 | Canonical Rule |
| 45-51 | Required Layers |
| 52-66 | Decision Scope |
| 67-76 | Upgrade Path |
DOC_HEADER:END -->

Use a probe-first planner now, and leave room for a fuller controller later.

## Intent

Define how runtime adaptation works before owned buffers and kernel families
expand, without overcommitting to a live feedback controller too early.

## Request Signals

- adaptive runtime
- nvml
- variant registry
- probe and adapt
- chunk planning
- saturation monitoring

## Open First

- docs/architecture/adaptive-runtime.md
- docs/architecture/runtime.md
- src/vibespatial/runtime/adaptive.py
- src/vibespatial/runtime/kernel_registry.py
- docs/decisions/0007-probe-first-adaptive-runtime.md

## Verify

- `uv run pytest tests/test_adaptive_runtime.py`
- `uv run python scripts/check_docs.py --check`

## Risks

- A full live controller would create more machinery than value before real kernels and chunked workloads exist.
- Overfitting variant choice too early can freeze bad metadata into the registry contract.
- Hard-coding NVML into call sites would make later telemetry upgrades expensive.

## Canonical Rule

- Adaptive planning happens before execution and, for streaming work, at chunk boundaries.
- The first landing is a planner, not a continuous controller.
- Telemetry is optional. When monitoring is unavailable, planning falls back to static heuristics and declared metadata.
- Explicit `cpu`, `gpu`, and precision overrides remain authoritative.

## Required Layers

- telemetry snapshot: GPU availability plus optional NVML saturation and memory signals
- variant registry: typed metadata, not just variant names
- planner input: kernel class, row count, geometry mix, residency, and requested mode
- planner output: selected runtime, variant, precision plan, chunk size hint, and reason log

## Decision Scope

The planner may adapt:

- kernel variant
- chunk size hint
- precision path through the existing precision-policy contract
- `auto` runtime target through the existing crossover policy

The planner must not:

- switch mid-kernel
- override explicit user pins
- depend on continuous background polling

## Upgrade Path

This design is intentionally a stepping stone.

- Today: one-shot planning plus optional re-plan after the first chunk.
- Later: richer telemetry, runtime history, and tighter re-plan cadence.

Moving from the planner to a live controller should only replace internal
policy and telemetry sources. Kernel call sites, registry metadata, and plan
objects should stay stable.
