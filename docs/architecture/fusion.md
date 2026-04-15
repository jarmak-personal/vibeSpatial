# Fusion Strategy

<!-- DOC_HEADER:START
Scope: Kernel-fusion policy, stage boundaries, and intermediate elimination rules.
Read If: You are changing pipeline execution, fused chains, or intermediate materialization behavior.
STOP IF: Your task already has a settled fusion contract and only needs implementation detail.
Source Of Truth: Phase-2 fusion strategy before broad multi-kernel pipelines land.
Body Budget: 103/240 lines
Document: docs/architecture/fusion.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-4 | Preamble |
| 5-9 | Intent |
| 10-18 | Request Signals |
| 19-26 | Open First |
| 27-31 | Verify |
| 32-37 | Risks |
| 38-43 | Candidate Approaches |
| 44-63 | Evaluation |
| 64-73 | Decision |
| 74-86 | Default Fusible Chains |
| 87-95 | Persisted Intermediates |
| 96-103 | Runtime Interaction |
DOC_HEADER:END -->

Use a lightweight staged operator DAG for fusion, not a whole-program lazy graph.

## Intent

Define how chained kernels should eliminate intermediate buffers without
committing the repo to an overbuilt execution engine too early.

## Request Signals

- fusion
- intermediate elimination
- kernel chain
- operator dag
- materialization boundary
- reusable index

## Open First

- docs/architecture/fusion.md
- docs/architecture/residency.md
- docs/architecture/adaptive-runtime.md
- src/vibespatial/runtime/fusion.py
- docs/decisions/0009-fusion-strategy.md

## Verify

- `uv run pytest tests/test_fusion_policy.py`
- `uv run python scripts/check_docs.py --check`

## Risks

- A full lazy graph would add scheduling complexity before the repo has enough kernels to justify it.
- Only relying on hand-written fused kernels would leave too much performance on the table and duplicate logic.
- Fusing across the wrong boundaries can hide reusable structures or break residency and diagnostics guarantees.

## Candidate Approaches

- full lazy evaluation graph
- explicit fused kernel variants only
- lightweight staged operator DAG

## Evaluation

Full lazy graph:

- strongest long-term optimizer story
- highest implementation and debugging cost
- wrong first move for the current kernel inventory

Explicit fused kernels only:

- easiest to reason about locally
- too rigid for mixed workloads and adaptive-runtime integration
- forces every valuable chain to be rewritten by hand

Lightweight staged DAG:

- enough structure to batch launches and eliminate ephemeral intermediates
- keeps explicit boundaries for materialization, reusable indexes, and diagnostics
- fits the current probe-first runtime better than a continuous optimizer

## Decision

Use a lightweight staged operator DAG.

- fuse device-local ephemeral chains
- persist reusable structures such as spatial indexes
- stop at explicit host materialization boundaries
- keep fusion transparent to users
- allow later kernels to provide specialized fused variants within the same staged contract

## Default Fusible Chains

- bounds -> SFC key -> sort
- predicate -> filter -> compact
- clip fast path -> predicate mask -> emit geometry slice
- overlay broadcast-right intersection ->
  containment bypass -> batched SH clip -> row-isolated remainder overlay
- grouped difference / symmetric difference ->
  candidate pairs -> group offsets -> segmented union -> row-isolated overlay

These chains may stay in one fused stage when they do not emit reusable
structures and do not cross a host boundary.

## Persisted Intermediates

Do not fuse away:

- spatial indexes
- reusable partition metadata
- explicit materialized host outputs
- buffers that are referenced by more than one downstream branch

## Runtime Interaction

- fusion planning happens before execution or at chunk boundaries
- overlay family planning now records both `execution_family` and
  `fusion_stages` in dispatch telemetry so the chosen staged shape is visible
  in health / profiling runs
- adaptive runtime may choose a fused stage shape, but not rewire the graph mid-kernel
- residency and fallback diagnostics must stay visible at stage boundaries
