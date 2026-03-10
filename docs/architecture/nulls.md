# Null And Empty Geometry Semantics

<!-- DOC_HEADER:START -->
> [!IMPORTANT]
> This block is auto-generated. Edit metadata in `docs/doc_headers.json`.
> Refresh with `uv run python scripts/check_docs.py --refresh` and validate with `uv run python scripts/check_docs.py --check`.

**Scope:** Null versus empty geometry semantics, validity rules, and kernel-state policy.
**Read If:** You are defining missing-geometry behavior, validity handling, or empty-geometry contracts.
**STOP IF:** Your task already has a settled null and empty contract and only needs implementation detail.
**Source Of Truth:** Phase-1 null and empty geometry policy before owned kernel expansion.
**Body Budget:** 87/240 lines
**Document:** `docs/architecture/nulls.md`

**Section Map (Body Lines)**
| Body Lines | Section |
|---|---|
| 1-4 | Preamble |
| 5-9 | Intent |
| 10-18 | Request Signals |
| 19-25 | Open First |
| 26-30 | Verify |
| 31-36 | Risks |
| 37-49 | Canonical States |
| 50-57 | Core Rules |
| 58-73 | Empty Geometry Rules |
| 74-80 | Batch Execution Rules |
| 81-87 | Buffer Implications |
<!-- DOC_HEADER:END -->

Null geometries and empty geometries are distinct states in `vibeSpatial`.

## Intent

Define the runtime and kernel contract for missing versus empty geometries
before owned geometry buffers and kernels expand.

## Request Signals

- null geometry
- empty geometry
- missing geometry
- validity bitmap
- null propagation
- empty semantics

## Open First

- docs/architecture/nulls.md
- src/vibespatial/nulls.py
- docs/architecture/runtime.md
- docs/decisions/0003-null-empty-geometry-contract.md

## Verify

- `uv run pytest tests/test_null_policy.py`
- `uv run python scripts/check_docs.py --check`

## Risks

- Treating null and empty as the same state breaks GeoPandas and Shapely semantics.
- Scalar branching for null or empty cases can create avoidable warp divergence.
- Buffer layouts that omit validity information will force slow-path recovery later.

## Canonical States

Every geometry slot is in exactly one of these states:

- `null`: missing geometry value
- `empty`: valid geometry object with zero coordinates
- `value`: non-empty geometry

Buffers must represent both missingness and emptiness:

- nulls use a validity bitmap
- empties use valid rows with zero-length coordinate or offset spans

## Core Rules

- Null handling follows Arrow semantics.
- Unary operations propagate nulls.
- Binary predicates propagate nulls.
- Joins and aggregations exclude null geometries from candidate generation.
- Empty geometries are valid inputs, not missing data.

## Empty Geometry Rules

- bounds of empty geometries are `NaN`
- area and length of empty geometries are `0`
- predicates involving an empty geometry return defined boolean results
- empties must never be rewritten to nulls

For the current default contract:

- `empty.intersects(X) -> false`
- `empty.within(X) -> false`
- `empty.contains(X) -> false`

Kernel-specific docs may extend this table, but they must not collapse empty
and null into the same result state.

## Batch Execution Rules

- Null and empty handling should use predicated execution or mask-and-fill patterns.
- Warp-wide scalar fallback branches are not the default mechanism.
- Output buffers must preserve the distinction between propagated nulls and
  defined empty-derived values.

## Buffer Implications

`o17.2.1` and later work should assume:

- validity is orthogonal to geometry family tags
- zero-length spans are meaningful and must survive partitioning or permutation
- restoration of row order cannot lose null or empty state
