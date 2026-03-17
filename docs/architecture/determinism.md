# Determinism And Reproducibility

<!-- DOC_HEADER:START
Scope: Deterministic versus performance-first runtime policy, reproducibility guarantees, and reduction-order rules.
Read If: You are changing reproducibility behavior, reduction order, or deterministic mode for GPU-oriented kernels.
STOP IF: Your task already has the determinism policy module open and only needs local implementation detail.
Source Of Truth: Phase-2 determinism and reproducibility policy before broader GPU reductions land.
Body Budget: 112/220 lines
Document: docs/architecture/determinism.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-4 | Preamble |
| 5-10 | Intent |
| 11-19 | Request Signals |
| 20-27 | Open First |
| 28-33 | Verify |
| 34-42 | Risks |
| 43-54 | Canonical Rule |
| 55-71 | What Changes In Deterministic Mode |
| 72-89 | Operations Affected |
| 90-101 | Performance Budget |
| 102-112 | Current Baseline |
DOC_HEADER:END -->

`vibeSpatial` defaults to maximum throughput, not maximum reproducibility.

## Intent

Define when GPU-oriented operations may be nondeterministic, what deterministic
mode guarantees, and how future kernels should implement reproducible reductions
without pretending cross-device bitwise equality exists.

## Request Signals

- determinism
- reproducibility
- bitwise identical
- reduction order
- atomics
- stable output

## Open First

- docs/architecture/determinism.md
- src/vibespatial/determinism.py
- scripts/check_determinism.py
- docs/architecture/runtime.md
- docs/architecture/precision.md

## Verify

- `uv run pytest tests/test_determinism_policy.py -q`
- `uv run python scripts/check_determinism.py --rows 512 --groups 32 --repeats 100`
- `uv run python scripts/check_docs.py --check`

## Risks

- Claiming cross-device reproducibility would be false; different architectures
  and driver stacks may choose different legal implementations.
- Hidden nondeterminism in reductions, scans, or floating atomics makes
  debugging and scientific verification much harder.
- Forcing deterministic order everywhere would waste performance on operations
  that are not actually sensitivity-bound.

## Canonical Rule

- Default mode is performance-first.
- Deterministic mode is explicit and opt-in.
- Deterministic mode guarantees bitwise-identical output only for the same
  input, same device architecture, and same driver/runtime stack.
- Cross-device reproducibility is not guaranteed.

The runtime flag is:

- `VIBESPATIAL_DETERMINISM=default|deterministic`

## What Changes In Deterministic Mode

For affected kernels, deterministic mode requires:

- stable output order
- fixed reduction order
- fixed scan order
- no floating-point atomics as the final accumulation mechanism
- explicit restore-order after compaction or partitioning

Preferred implementation patterns:

- CCCL or CUB stable sort before group reduction
- fixed tree-reduction shapes
- sorted-key gather or reduce-by-key for grouped aggregates
- staged ambiguity compaction followed by deterministic restore

## Operations Affected

The main affected categories are:

- metric reductions
  - area totals
  - length totals
  - grouped sums and counts
- structured constructive work
  - dissolve
  - overlay area totals
  - any grouped union pipeline that reduces or restores rows
- query aggregation surfaces
  - spatial join count/sum style kernels

Pure geometry-local coarse work is usually not determinism-sensitive unless it
changes emit order.

## Performance Budget

Deterministic mode is allowed to cost more.

Repo policy:

- up to `2x` overhead is acceptable for reduction-heavy metric and constructive kernels
- up to `1.5x` overhead is acceptable for order-sensitive coarse or predicate paths

Each affected kernel or pipeline should publish its measured overhead once a GPU
implementation exists.

## Current Baseline

The current dissolve baseline is CPU-hosted and already stable in row order, so
the reproducibility probe in `scripts/check_determinism.py` is mainly proving
the contract and artifact shape today:

- repeated dissolve + area output hashing
- same-device bitwise check
- default vs deterministic elapsed comparison

That is the proof surface future GPU reductions should keep green as they land.
