# Kernel Inventory

<!-- DOC_HEADER:START
Scope: Scaffolded kernel inventory, benchmark stubs, and generated test surfaces.
Read If: You are generating, reviewing, or extending owned kernel scaffolds.
STOP IF: You already have the target kernel module and generated files open.
Source Of Truth: Kernel scaffold inventory and generated surface map for repo-owned kernels.
Body Budget: 42/220 lines
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
| 31-42 | Scaffolds |
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
| `point_bounds` | `vibespatial.kernels.predicates` | Tier 4 | `point, polygon` | `src/vibespatial/kernels/predicates/point_bounds.py` | `tests/test_point_bounds.py` | `vsbench run point-predicates` |
| `segment_intersection` | `vibespatial.spatial.segment_primitives` | Tier 1 | `linestring, polygon, multilinestring, multipolygon` | `src/vibespatial/spatial/segment_primitives.py` | `tests/test_segment_primitives.py` | `vsbench run segment-intersection` |
| `binary_constructive` | `vibespatial.constructive.binary_constructive` | Tier 3 | `point, polygon, multipolygon` | `src/vibespatial/constructive/binary_constructive.py` | `tests/test_binary_constructive.py` | `vsbench run constructive` |
| `envelope` | `vibespatial.constructive.envelope` | Tier 1 | `point, multipoint, linestring, multilinestring, polygon, multipolygon` | `src/vibespatial/constructive/envelope.py` | `tests/test_envelope.py` | `vsbench run constructive` |
| `geometry_simplify` | `vibespatial.constructive.simplify` | Tier 1 | `linestring, multilinestring, polygon, multipolygon` | `src/vibespatial/constructive/simplify.py` | `tests/test_zero_copy_pipeline.py` | `vsbench run constructive` |
| `make_valid` | `vibespatial.constructive.make_valid_pipeline` | Tier 3 | `polygon, multipolygon, linestring, multilinestring, point, multipoint` | `src/vibespatial/constructive/make_valid_pipeline.py` | `tests/test_make_valid_pipeline.py` | `vsbench run constructive` |
| `normalize` | `vibespatial.constructive.normalize` | Tier 1 | `point, multipoint, linestring, multilinestring, polygon, multipolygon` | `src/vibespatial/constructive/normalize.py` | `tests/test_normalize.py` | `vsbench run constructive` |
