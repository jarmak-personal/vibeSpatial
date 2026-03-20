# Precision Strategy

<!-- DOC_HEADER:START
Scope: Dual fp32/fp64 compute strategy, runtime precision dispatch, and canonical storage policy.
Read If: You are designing kernel arithmetic, precision selection, or numerical policy.
STOP IF: Your task already has a settled precision plan and only needs implementation detail.
Source Of Truth: Phase-1 precision dispatch policy before owned kernel expansion.
Body Budget: 105/240 lines
Document: docs/architecture/precision.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-4 | Preamble |
| 5-9 | Intent |
| 10-19 | Request Signals |
| 20-26 | Open First |
| 27-31 | Verify |
| 32-37 | Risks |
| 38-47 | Canonical Rule |
| 48-55 | Precision Modes |
| 56-62 | Kernel Classes |
| 63-79 | Default Policy |
| 80-95 | What Staged fp32 Means |
| 96-105 | Buffer And Signature Implications |
DOC_HEADER:END -->

Use dual compute precision with one canonical storage precision.

## Intent

Define how `vibeSpatial` chooses between staged fp32 execution and native fp64
execution before owned geometry buffers and kernels expand.

## Request Signals

- precision
- fp32
- fp64
- numerical
- runtime precision
- ulp
- coordinate centering

## Open First

- docs/architecture/precision.md
- src/vibespatial/runtime/precision.py
- docs/architecture/runtime.md
- docs/decisions/0002-dual-precision-dispatch.md

## Verify

- `uv run pytest tests/test_precision_policy.py`
- `uv run python scripts/check_docs.py --check`

## Risks

- Canonical fp32 storage would trade away too much accuracy for geographic and projected coordinates.
- Native fp64 everywhere would erase the performance advantage on consumer GPUs.
- A single global precision switch would hide kernel-class differences that matter for correctness.

## Canonical Rule

- Canonical owned geometry storage is `fp64`.
- Compute precision is selected at dispatch time.
- The dispatch contract is `auto | fp32 | fp64`.

This separates authoritative coordinate storage from execution strategy in the
same way the repo separates canonical mixed-geometry storage from execution-time
partitioning.

## Precision Modes

- `auto`: choose the plan from device profile, kernel class, and coordinate
  characteristics
- `fp32`: force staged fp32 execution with centering and compensation where
  required
- `fp64`: force native fp64 execution

## Kernel Classes

- `coarse`: bounds, simple filters, sort keys, and other cheap geometry-local work
- `metric`: area, length, centroid-like reductions, and other accumulation-heavy kernels
- `predicate`: orientation tests, point-in-polygon, binary predicates, and exact-refine pipelines
- `constructive`: clip, intersection, union, difference, and other geometry-producing kernels

## Default Policy

On CPU:

- use native fp64 semantics

On datacenter-style GPUs with favorable fp64 throughput:

- default to native fp64 for all kernel classes

On consumer-style GPUs with weak fp64 throughput:

- `coarse`: staged fp32 with coordinate centering when large absolute magnitudes make cancellation likely
- `metric`: staged fp32 with centered coordinates and compensated accumulation
- `predicate`: staged fp32 for coarse work plus selective fp64 refinement for sensitive cases
- `constructive`: stay on native fp64 until robustness work proves a cheaper safe path

## What Staged fp32 Means

Staged fp32 is not naive fp32.

Required techniques:

- local coordinate centering or shifting before sensitive arithmetic
- compensated accumulation for reduction-like kernels
- selective fp64 refinement for predicate-style kernels

Rejected as defaults:

- canonical fp32 coordinate storage
- permanent dual fp32/fp64 buffer copies
- pure fp32 constructive kernels without a later robustness proof

## Buffer And Signature Implications

`o17.2.1` and later kernel work should assume:

- buffers store authoritative coordinates in fp64
- dispatch chooses a `PrecisionPlan`
- kernel call sites take a precision mode or a resolved precision plan, not
  ad hoc booleans
- temporary centered fp32 work buffers are execution-local artifacts, not part
  of the canonical buffer contract
