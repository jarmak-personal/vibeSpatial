---
id: ADR-0048
status: accepted
date: 2026-08-23
deciders:
  - vibeSpatial maintainers
tags:
  - architecture
  - gpu
  - precision
  - robustness
---

# Bounded-Accuracy Spatial Execution

## Context

vibeSpatial stores canonical coordinates as fp64 and promises exact predicate
semantics by default. Consumer GPUs can nevertheless execute centered fp32
arithmetic substantially faster, especially for metric work. Exact predicate
kernels already use the useful pattern: compute a cheap decision, identify the
numerically ambiguous cases, and refine only those cases with fp64 or expansion
arithmetic.

The missing abstraction was the uncertainty between those stages. Individual
kernels used fixed epsilons or local error variables, so downstream consumers
could not distinguish a proven decision from an unresolved one. Fixed epsilons
are particularly unsafe for topology: the former point-in-region tolerance
classified points up to `1e-7` coordinate units outside sloped boundaries as
inside.

Users also have legitimate workloads whose measurement fidelity is much lower
than fp64. A future opt-in mode may exchange a declared spatial error for more
throughput, but a dtype switch cannot express that contract. Predicate error is
distance from a decision boundary; metric error is a numeric interval around a
result. CRS units and topology make those distinct from `precision="fp32"`.

The RTX 4090 study in
`docs/dev/bounded-accuracy-experiment-review.md` measured useful fp32 gains but
did not prove conservative bounds across the supported domain. The Native*
feature hold also precludes adding an unrelated public policy surface now.

## Decision

Adopt a shared numerical-error-envelope execution contract and defer the public
bounded-accuracy API.

An error envelope is an immutable native planning carrier containing:

- a conservative host- or device-resident bound
- the physical quantity it bounds, such as orientation sign or distance
- the compute precision that produced it
- the derivation used to reproduce and audit the bound

Exact-mode consumers compare their decision margin with the envelope. Decisions
outside it are final; decisions inside it are ambiguous and must use the
operation's existing exact fallback. An envelope never weakens semantics by
itself. A zero envelope represents an already exact or fully refined result.

Predicate kernels may implement the same contract inside CUDA without
materializing a row-shaped carrier. For orientation, centered fp32 computes a
determinant and a conservative roundoff envelope per edge; ambiguous signs call
the existing adaptive exact orientation function in the same kernel. Fixed
absolute boundary epsilons are not exact predicate implementations. A nonzero
fp64 operand or product that cannot be represented as a normal fp32 value is
also ambiguous; this fail-closed rule covers device subnormal flushing.
Exact orientation preserves binary64 subtraction tails and uses a fixed-limb
determinant sign when product exponents would overflow or underflow expansion
arithmetic. The common stage-A path does not allocate that fallback frame.

Metric planners carry the envelope explicitly between coarse distance and
ordering/threshold refinement. The bound may remain a device scalar so native
consumers do not add a host planning fence.

The future public concept is tentatively `AccuracyBudget`, separate from both
`PrecisionPlan` and `RobustnessPlan`:

- exact is the zero-configuration default
- users authorize boundary or absolute metric tolerances, not data types
- the runtime may skip exact refinement only when a conservative proof fits the
  active budget
- V1 is limited to projected CRS coordinate units and point-region predicates,
  point-family distance, and `dwithin`
- null, empty, index, ordering, and type behavior remain exact
- geographic, missing-CRS, constructive, and persistent low-precision geometry
  behavior decline until separately specified
- opt-in is per call or task-local immutable context, never mutable global state
- approximate results and dispatch evidence carry provenance; they do not
  silently become canonical exact inputs

Public `AccuracyBudget` implementation requires a separate feature-hold
decision plus cross-device proof that conservative envelopes and complete-stage
performance win. This ADR authorizes the internal exact-mode substrate now; it
does not expose approximation.

## Consequences

- Exact predicates can use fp32 throughput without accepting fp32 decisions near
  topology boundaries.
- Metrics and predicates share one vocabulary for ambiguity while retaining
  operation-specific derivations and fallbacks.
- Error-bound computation, selective refinement, readiness, and producer
  retention are part of a kernel's complete physical shape and benchmarks.
- Empirical maximum error remains useful test evidence but cannot replace a
  conservative derivation.
- Some kernels will stay fp64 when deriving or consuming an envelope costs more
  than it saves.
- Public approximation remains unavailable until the execution plan's accuracy,
  CRS, cross-device, and performance gates pass.

## Alternatives Considered

- **Expose `fp32=True` or `coarse=True`.** Rejected because arithmetic type does
  not bound spatial error and encourages users to infer guarantees that do not
  exist.
- **Keep per-kernel epsilons.** Rejected because absolute tolerances are not
  scale-aware, cannot be composed, and caused exact point-region false
  positives.
- **Always execute fp64.** Rejected because selective exact refinement retains
  exact semantics while recovering meaningful consumer-GPU throughput.
- **Adopt the public budget from empirical RTX 4090 results.** Rejected because
  observed maxima are not proofs and do not establish H100/H200 behavior, CRS
  semantics, or safe API composition.
- **Approximate canonical geometry storage.** Rejected because error would
  silently accumulate across operations and violate the owned fp64 storage
  contract.
