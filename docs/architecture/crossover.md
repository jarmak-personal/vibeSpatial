# Dispatch Crossover Policy

<!-- DOC_HEADER:START
Scope: Per-kernel auto-dispatch thresholds and CPU/GPU crossover policy.
Read If: You are changing auto mode, size thresholds, or CPU/GPU dispatch heuristics.
STOP IF: Your task already has a settled crossover policy and only needs implementation detail.
Source Of Truth: Phase-1 crossover policy before adaptive runtime lands.
Body Budget: 58/240 lines
Document: docs/architecture/crossover.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-4 | Preamble |
| 5-9 | Intent |
| 10-18 | Request Signals |
| 19-25 | Open First |
| 26-30 | Verify |
| 31-36 | Risks |
| 37-43 | Canonical Rule |
| 44-53 | Provisional Thresholds |
| 54-58 | Measurement Rule |
DOC_HEADER:END -->

`auto` mode should use per-kernel row thresholds, not one global size rule.

## Intent

Define the minimum dataset sizes where `auto` should prefer GPU execution for
each kernel class before adaptive runtime work lands.

## Request Signals

- crossover
- dispatch threshold
- auto mode
- size gate
- small data
- gpu dispatch

## Open First

- docs/architecture/crossover.md
- docs/architecture/runtime.md
- src/vibespatial/crossover.py
- docs/decisions/0006-dispatch-crossover-policy.md

## Verify

- `uv run pytest tests/test_runtime_policy.py`
- `uv run python scripts/check_docs.py --check`

## Risks

- A single threshold would hide the real differences between bounds, predicates, and overlay work.
- Dispatching small inputs to GPU can lose badly to single-threaded GeoPandas.
- Leaving crossover implicit invites ad hoc size checks inside individual kernels.

## Canonical Rule

- Thresholds are defined per kernel class.
- Explicit `gpu` bypasses the threshold and must attempt device execution.
- Explicit `cpu` bypasses the threshold and stays on host.
- `auto` uses CPU below the threshold and GPU at or above it.

## Provisional Thresholds

- `coarse`: `1K`
- `metric`: `5K`
- `predicate`: `10K`
- `constructive`: `50K`

These are policy constants, not folklore. Update them when kernel
implementations materially change.

## Measurement Rule

- Benchmarks should compare `auto` against single-threaded GeoPandas on the same workload family.
- `auto` must not regress below the single-threaded GeoPandas baseline for the thresholded sizes it claims to cover.
- `o17.2.10` should eventually replace fixed crossover constants with adaptive runtime inputs where justified.
