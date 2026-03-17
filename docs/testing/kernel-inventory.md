# Kernel Inventory

<!-- DOC_HEADER:START
Scope: Scaffolded kernel inventory, benchmark stubs, and generated test surfaces.
Read If: You are generating, reviewing, or extending owned kernel scaffolds.
STOP IF: You already have the target kernel module and generated files open.
Source Of Truth: Kernel scaffold inventory and generated surface map for repo-owned kernels.
Body Budget: 36/220 lines
Document: docs/testing/kernel-inventory.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-4 | Preamble |
| 5-8 | Intent |
| 9-14 | Request Signals |
| 15-20 | Open First |
| 21-25 | Verify |
| 26-30 | Risks |
| 31-36 | Scaffolds |
DOC_HEADER:END -->

Generated kernel scaffolds land here first so agents can audit what exists and which tier gate applies.

## Intent

Track scaffolded kernel modules, tests, and benchmark stubs.

## Request Signals

- kernel scaffold
- benchmark stub
- kernel inventory

## Open First

- docs/testing/kernel-inventory.md
- scripts/generate_kernel_scaffold.py
- docs/testing/performance-tiers.md

## Verify

- `uv run python scripts/generate_kernel_scaffold.py --check point_bounds`
- `uv run python scripts/check_docs.py --check`

## Risks

- Scaffold drift can leave tests, benchmarks, and manifests out of sync.
- Tier metadata becomes meaningless if generated benchmarks do not match policy.

## Scaffolds

| Kernel | Module | Tier | Geometry Types | Source | Test | Benchmark |
|---|---|---|---|---|---|---|
<!-- KERNEL_INVENTORY:ROWS -->
| `point_bounds` | `vibespatial.kernels.predicates` | Tier 4 | `point, polygon` | `src/vibespatial/kernels/predicates/point_bounds.py` | `tests/test_point_bounds.py` | `benchmarks/bench_point_bounds.py` |
