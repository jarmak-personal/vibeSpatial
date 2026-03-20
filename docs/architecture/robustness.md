# Robustness Strategy

<!-- DOC_HEADER:START
Scope: Exactness guarantees, fallback policy, and topology-preservation strategy for predicates and overlay.
Read If: You are designing predicate math, intersection logic, or overlay correctness guarantees.
STOP IF: Your task already has a settled robustness contract and only needs implementation detail.
Source Of Truth: Phase-1 robustness policy for predicate and constructive kernels.
Body Budget: 107/240 lines
Document: docs/architecture/robustness.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-4 | Preamble |
| 5-9 | Intent |
| 10-19 | Request Signals |
| 20-27 | Open First |
| 28-32 | Verify |
| 33-38 | Risks |
| 39-47 | Canonical Rule |
| 48-53 | Guarantee Levels |
| 54-60 | Default Kernel-Class Policy |
| 61-75 | Predicate Strategy |
| 76-88 | Constructive Strategy |
| 89-102 | GPU Strategy |
| 103-107 | Rejected Defaults |
DOC_HEADER:END -->

Predicate correctness and topology preservation require more than choosing fp32 or fp64.

## Intent

Define the shared robustness strategy for predicates and constructive geometry
before exact-kernel work begins.

## Request Signals

- robustness
- exact predicate
- orientation
- incircle
- topology preservation
- overlay robustness
- degeneracy

## Open First

- docs/architecture/robustness.md
- src/vibespatial/runtime/robustness.py
- docs/architecture/precision.md
- docs/architecture/nulls.md
- docs/decisions/0004-robustness-strategy.md

## Verify

- `uv run pytest tests/test_robustness_policy.py`
- `uv run python scripts/check_docs.py --check`

## Risks

- Wrong orientation or intersection signs will silently poison every later predicate and overlay kernel.
- Exact-style fallback that diverges per row can destroy GPU efficiency if it is not staged carefully.
- Snap or grid policies can preserve topology while still changing geometry semantics if applied indiscriminately.

## Canonical Rule

- Predicate kernels must never return the wrong sign.
- Constructive kernels must preserve topology, not just approximate coordinates.
- Precision dispatch alone is insufficient to guarantee robustness.

Robustness is a separate contract layered on top of the
precision and null/empty contracts.

## Guarantee Levels

- `exact`: sign or topology-critical results must match the mathematically correct outcome
- `bounded-error`: metric-style results may deviate within the documented numeric bound
- `best-effort`: permitted only for coarse kernels that do not determine topology

## Default Kernel-Class Policy

- `coarse`: bounded-error is acceptable
- `metric`: bounded-error is acceptable
- `predicate`: exact guarantee required
- `constructive`: exact guarantee with topology preservation required

## Predicate Strategy

For orientation, incircle, and segment-side decisions:

- use fast-path arithmetic first
- reject clearly separated cases cheaply
- escalate ambiguous cases to an exact-style fallback

Chosen fallback policy:

- fp32 predicate pipelines escalate ambiguous rows to expansion-arithmetic style exact predicates
- fp64 predicate pipelines may use selective fp64 or expansion fallback, but they still owe an exact sign guarantee

Pure epsilon-based sign tests are rejected as the final decision mechanism.

## Constructive Strategy

Constructive kernels need more than exact predicate signs.

Default policy:

- use exact or exact-style predicate decisions for topology-critical branching
- reconstruct intersection points with a topology-preserving fallback
- prefer topology preservation over fastest coordinate reconstruction

For now, the design contract assumes rational-reconstruction or equivalent
exact-style fallback for intersection points when the fast path is ambiguous.

## GPU Strategy

GPU-unfriendly exact arithmetic should be staged, not sprayed across every row.

Required pattern:

- bulk fast path for clearly separated rows
- compact ambiguous rows
- run exact-style fallback only on the compacted ambiguity set
- restore output order explicitly

This keeps warp divergence localized to the minority of degenerate or nearly
degenerate rows.

## Rejected Defaults

- fixed global epsilons as the final predicate decision rule
- naive “fp64 is robust enough” reasoning
- pure snap-rounding as the only overlay correctness policy
