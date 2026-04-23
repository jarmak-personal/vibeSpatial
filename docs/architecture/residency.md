# Residency Policy

<!-- DOC_HEADER:START
Scope: Device residency defaults, transfer visibility rules, and zero-copy interop policy.
Read If: You are designing buffer movement, interop adapters, or host/device materialization behavior.
STOP IF: Your task already has a settled residency contract and only needs implementation detail.
Source Of Truth: Phase-1 residency and transfer policy before owned geometry buffers land.
Body Budget: 99/240 lines
Document: docs/architecture/residency.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-4 | Preamble |
| 5-9 | Intent |
| 10-18 | Request Signals |
| 19-25 | Open First |
| 26-30 | Verify |
| 31-36 | Risks |
| 37-58 | Canonical Rule |
| 59-64 | Transfer Rules |
| 65-71 | Zero-Copy Rule |
| 72-84 | Pipeline Rule |
| 85-99 | Diagnostics Surface |
DOC_HEADER:END -->

Owned geometry buffers are lazy-resident and move only at explicit boundaries.

## Intent

Define when geometry and attribute buffers stay on device, when transfers are
allowed, and how zero-copy interop should behave before owned buffers land.

## Request Signals

- residency
- device transfer
- zero-copy
- host materialization
- interop
- fallback visibility

## Open First

- docs/architecture/residency.md
- docs/architecture/runtime.md
- src/vibespatial/runtime/residency.py
- docs/decisions/0005-device-residency-policy.md

## Verify

- `uv run pytest tests/test_runtime_policy.py`
- `uv run python scripts/check_docs.py --check`

## Risks

- Silent host copies can erase GPU wins while still looking correct in tests.
- Early buffer work can lock in unnecessary copies if residency is underspecified.
- Interop wrappers can accidentally materialize on host unless zero-copy rules are explicit.

## Canonical Rule

- Geometry and attribute buffers are lazy-resident.
- After first use on device, owned buffers are device-resident by default.
- In `auto` mode, device-resident workloads stay on device; crossover planning
  may still choose among GPU variants, but it should not demote back to CPU
  based on row-count heuristics alone.
- Buffers stay where they were created until a user-visible API or explicit
  runtime request needs a move.
- Device-first owned outputs may keep lightweight host metadata
  (validity/tags/offset vectors) while deferring heavy coordinate payload
  materialization until an explicit host boundary.
- Cached derived buffers such as per-row bounds may exist on both host and
  device when that avoids recomputation across repeated kernel launches.

Host materialization boundaries are explicit:

- `to_pandas`
- `to_numpy`
- `values`
- `__repr__`

## Transfer Rules

- Non-user transfers must remain observable in logs, diagnostics, or test output.
- Unsupported GPU paths may transfer to host only through explicit fallback machinery.
- Runtime policy may request a move, but it cannot hide the transfer.

## Zero-Copy Rule

- Prefer zero-copy interop with cuDF, CuPy, Arrow, and GeoArrow-compatible
  buffers when ownership and layout constraints permit.
- Interop views should not trigger a copy by default.
- Copying for interop must be treated as a visible transfer event, not a silent helper detail.

## Pipeline Rule

- A device-resident pipeline should incur zero non-user transfers once buffers
  have reached device residency.
- Device-resident pipelines should treat `auto` as GPU-sticky until an explicit
  host materialization or visible fallback boundary occurs.
- Constructive chains such as point clip followed by point buffer should keep
  coordinate payloads on device through intermediate owned outputs.
- Re-entering point-only `clip_by_rect` from a device-backed owned array should
  materialize only the kept point rows needed for the public result, not the
  full source batch.
- The transfer audit surface should prove this rule.

## Diagnostics Surface

Owned geometry arrays should make the following visible:

- current residency
- whether device mirrors have been allocated
- transfer events and their trigger
- explicit materialization events
- runtime selection or fallback reasons recorded at the buffer boundary

The bootstrap implementation exposes this through an event log on the owned
array object instead of a separate daemon or profiler surface. Runtime-level
CUDA D2H copies are also counted and synchronously timed separately, because
owned-array residency events are semantic boundaries and do not cover every
internal device-to-host copy.
