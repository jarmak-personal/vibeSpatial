# Archived Device Execution Planning Implementation Plan

<!-- DOC_HEADER:START
Scope: Archived design exploration for a complete capability-driven private device execution planner.
Read If: You are auditing why ADR-0047 was superseded or researching previously proposed calibration and planning contracts.
STOP IF: You are implementing current point-region or adaptive-runtime work; use the evidence-first plan instead.
Source Of Truth: Historical design record only; not implementation authority.
Body Budget: 700/730 lines
Document: docs/archive/2026-08-18-device-planning/device-execution-planning-implementation-plan.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-8 | Preamble |
| 9-19 | Intent |
| 20-35 | Request Signals |
| 36-53 | Open First |
| 54-63 | Verify |
| 64-79 | Risks |
| 80-97 | Mission |
| 98-141 | Risk Classification And Safety Model |
| 142-168 | Scope And Non-Goals |
| 169-193 | Current State And Gaps |
| 194-222 | Target Architecture |
| 223-370 | Core Data Contracts |
| 371-445 | Planner Pipeline |
| 446-466 | Calibration Policy |
| ... | (7 additional sections omitted; open document body for full map) |
DOC_HEADER:END -->

> **Archived 2026-08-18.** This design exploration is not implementation
> authority. Independent hardware and YAGNI reviews found that it both
> duplicated the existing adaptive runtime and underspecified the multi-stage
> execution and resource-safety contract it would need. The active direction is
> `docs/dev/evidence-first-point-region-execution-plan.md`.

## Intent

Implement ADR-0047 as a complete, reusable private runtime facility. The layer
must let one operation delegate all hardware-sensitive GPU choices to a common
planner without requiring immediate adoption by every existing kernel.

This plan sits above operation-specific execution plans and below ADR-0046
physical workload contracts. It does not choose public semantics or invent
workload shape. It converts an already-admitted physical shape into a concrete
plan for the active logical CUDA device.

## Request Signals

- device execution planning
- architecture-independent GPU tuning
- device capability profile
- kernel resource profile
- variant cost model
- bounded calibration
- cross-device portability
- H100 / A100 / RTX 4090 / RTX 3090
- MIG planning
- tile size
- cooperative width
- occupancy and grid planning
- complete device planner adoption

## Open First

- `docs/decisions/0047-device-execution-planning.md`
- `docs/decisions/0046-gpu-physical-workload-shape-contracts.md`
- `docs/decisions/0007-probe-first-adaptive-runtime.md`
- `docs/decisions/0002-dual-precision-dispatch.md`
- `docs/decisions/0040-tiered-gpu-memory-pool.md`
- `docs/architecture/adaptive-runtime.md`
- `docs/architecture/precision.md`
- `docs/archive/2026-08-18-device-planning/adaptive-point-region-refinement-plan.md`
- `src/vibespatial/runtime/adaptive.py`
- `src/vibespatial/runtime/precision.py`
- `src/vibespatial/runtime/kernel_registry.py`
- `src/vibespatial/runtime/workload.py`
- `src/vibespatial/cuda/_runtime.py`
- `tests/test_adaptive_runtime.py`
- `tests/test_precision_policy.py`

## Verify

- `uv run ruff check`
- `uv run pytest tests/test_device_execution_planning.py -q`
- `uv run pytest tests/test_adaptive_runtime.py tests/test_precision_policy.py -q`
- `uv run pytest tests/test_runtime_policy.py tests/test_gpu_memory_pool.py -q`
- `uv run python scripts/check_architecture_lints.py --all`
- `uv run python scripts/check_docs.py --check`
- `uv run python scripts/benchmark_pipelines.py --suite full --repeat 1 --gpu-sparkline`

## Risks

- A device planner can become a second broad query planner if it starts
  deciding public semantics or operation order.
- Product names and compute capability can become hidden performance heuristics
  even when represented as profile fields.
- Calibration can add more work than it saves, synchronize the stream, or
  pollute production latency.
- Function-resource metadata can become stale when kernel source, precision,
  compiler options, or dynamic shared memory changes.
- One global threshold can hide workload skew just as easily as one global row
  threshold.
- Rich plans can tempt call sites to override individual fields after planning,
  destroying coherence and reproducibility.
- A planner that is never used end to end is documentation, not infrastructure.

## Mission

Provide one private API that can select and explain the complete GPU execution
strategy for an admitted physical workload:

```text
public operation semantics
        -> ADR-0046 physical shape and work estimate
        -> device execution planner
        -> precision + variant + partition + launch + scratch plan
        -> native execution and result carrier
        -> explicit public export boundary
```

The implementation is successful when adaptive point-region refinement can use
the planner without containing product-specific thresholds, fp64-ratio policy,
hard-coded tile capacities, or operation-local calibration storage.

## Risk Classification And Safety Model

This is high-risk infrastructure because its blast radius grows with every
adopter. The design must make a poor prediction a bounded performance miss,
not a correctness, memory-safety, or availability failure.

Non-negotiable safety rules:

- **Admission precedes ranking.** Semantic, precision, robustness, resource,
  integer-width, and launch-coverage checks construct the admissible set. The
  cost model and calibration only rank that set.
- **Permanent exact baseline.** Every family retains a portable exact variant
  with conservative memory and launch policy. Unknown or inconsistent state
  selects it.
- **Fail closed locally.** Missing attributes, stale calibration, low
  confidence, planner exceptions, allocation pressure, or resource mismatch
  decline specialization. They do not silently move an explicit GPU request to
  CPU.
- **Immutable validated plan.** Launchers validate coverage, bounds, scratch
  admission, readiness, and explicit pins. They consume a plan atomically and
  cannot mix fields from different alternatives.
- **No calibration authority over correctness.** Timing never relaxes exactness
  or admits a precision path that has not independently passed its oracle.
- **Bounded observation.** Planning and calibration have explicit time, launch,
  synchronization, and byte budgets under ADR-0045.
- **Stable decisions.** Confidence margins and hysteresis prevent noisy samples
  from switching variants at every chunk.
- **Per-family activation.** A successful first adopter does not activate the
  planner for unrelated kernels.

Rollout states are private runtime policy:

1. `shadow`: compute and record the proposed plan while executing the existing
   baseline.
2. `validated`: permit specialization in tests and explicit family canaries.
3. `active`: enable the family by default only for validated capability and
   workload classes.
4. `baseline`: force the portable family variant when evidence expires or a
   regression guard trips.

These states are not new public tuning APIs. Existing execution and precision
overrides remain the user authority. Planner policy version, family activation,
and evidence expiry are internal and observable in dispatch diagnostics.

## Scope And Non-Goals

The planner owns:

- static logical-device capabilities
- dynamic allocator and optional utilization state
- compiled-kernel resource characteristics
- precision-plan selection
- admissible variant comparison
- cooperative-width and work-bucket policy
- tile and scratch sizing
- block/grid/dynamic-shared-memory policy
- bounded calibration and cache invalidation
- plan explanation and profiling metadata

The planner does not own:

- public GeoPandas method semantics
- geometry validity or predicate definitions
- relation, rowset, grouped, or frame lineage
- broad workflow planning
- arbitrary operation reordering
- continuous background control
- cross-device work distribution
- public hardware tuning knobs
- correctness fallbacks hidden behind a faster variant

## Current State And Gaps

The repo already has important pieces:

- `DevicePrecisionProfile` records the fp64-to-fp32 throughput ratio.
- `_detect_device_profile()` obtains that ratio from the active CUDA device.
- `PrecisionPlan` selects native fp64 or staged fp32 by kernel class.
- `AdaptivePlan` owns runtime selection, variant, precision, chunk hints, and
  diagnostics.
- NVRTC compiles CUBIN for the active compute capability.
- `optimal_block_size()` uses CUDA occupancy for one compiled kernel.
- `PhysicalWorkEstimate` exposes shape-level cardinalities.
- the allocator reports budgets, admissions, and pressure.

The missing contracts are:

- a device profile broader than precision
- a dynamic snapshot separated from static identity
- introspection of each compiled variant's actual resource use
- a cost/admission model joining device, kernel, and physical work
- coherent tile, cooperative-width, and grid selection
- calibration records with stable cache keys and confidence
- an immutable complete plan accepted by launchers
- simulated and real cross-device plan tests

## Target Architecture

```text
                       stable/session cached
CUDA driver probes --------------------------> DeviceExecutionProfile
allocator + NVML ----------------------------> DeviceExecutionSnapshot
compiled CUBIN/function attrs ---------------> KernelResourceProfile

ADR-0046 work estimate ----------------------> PhysicalWorkProfile
operation variant registry ------------------> DeviceVariantSpec[]
calibration cache ----------------------------> CalibrationRecord?

                                 all inputs
                                     |
                                     v
                           plan_device_execution()
                                     |
                                     v
                           DeviceExecutionPlan
                 precision / variant / tiles / launch / scratch
                                     |
                                     v
                         operation-native executor
```

Profiles describe facts. Variant specifications describe capabilities and cost
formulas. Plans describe one immutable decision. No profile should contain a
selected operation variant.

## Core Data Contracts

Names are provisional, but their separation is required.

### `DeviceIdentity`

Stable identity for caching and diagnostics:

- CUDA device ordinal for the active process
- device UUID when available
- logical partition or MIG identity when available
- compute capability
- driver and runtime versions

The identity is not a performance policy. Device name may be retained only for
human-readable diagnostics.

### `DeviceExecutionProfile`

Session-stable hardware and feature facts:

- identity
- SM count and warp size
- maximum threads per SM and block
- maximum resident blocks and warps per SM
- register file size per SM
- shared memory per SM and per block
- L2 size
- global memory capacity visible to the logical device
- fp64-to-fp32 throughput ratio reported by the driver
- memory clock and bus-width-derived nominal bandwidth when available
- support flags for asynchronous allocation, cooperative launch, cluster
  launch, distributed shared memory, and other optional features

Feature flags may use compute capability to prove instruction availability.
They must not use compute capability to infer throughput when a direct
attribute or calibration exists.

### `DeviceExecutionSnapshot`

Dynamic, cheaply refreshable state:

- allocator ceiling and query reserve
- live, reserved, and available growth bytes
- largest known admissible allocation
- memory-pressure state
- optional SM and memory utilization
- active stream identity and readiness policy
- snapshot timestamp and provenance

The snapshot must not require synchronization with active compute merely to
plan. Missing NVML data is normal and must not disable planning.

### `KernelResourceProfile`

Facts obtained after compilation for one kernel variant:

- kernel group, entry point, and code hash
- compiled architecture and compiler options
- precision variant
- registers per thread
- static shared memory
- local memory or spill indicators
- maximum threads per block
- occupancy-selected block size
- active blocks per SM at supported block/shared-memory choices
- optional cluster or cooperative-launch constraints

The key includes every source or launch parameter that can change resource
use. A function pointer alone is not a persistent cache key.

### `PhysicalWorkProfile`

An extension or normalized view of `PhysicalWorkEstimate`:

- ADR-0046 shape family
- primitive unit counts
- output and scratch estimates
- workload residency
- streaming/chunk state
- reuse expectation for persistent metadata
- optional histograms or quantiles for variable work
- skew metrics such as coefficient of variation and maximum-to-median ratio
- consumer/result shape: relation, selection, grouped reduction, dynamic
  geometry, or terminal export

Large histograms remain device-resident. The planner may consume a compact
fixed-size summary packet, but it must not export per-row or per-candidate
statistics.

### `DeviceVariantSpec`

Registration for a materially distinct execution alternative:

- stable semantic family and variant name
- supported physical shapes and result carriers
- required device features
- supported precision and robustness plans
- minimum and maximum cooperative widths
- workload admissibility predicate
- work and scratch formulas
- launch-policy factory
- calibration family
- exactness and determinism metadata
- portable-baseline marker

Variants are different physical executions, not aliases for the same kernel.
Examples include lane-per-item, warp-per-item, block-per-item, and staged
count/scan/scatter. Compiler precision specializations may be separate variants
when their execution pipeline differs.

### `CalibrationRecord`

Bounded empirical evidence:

- cache key and schema version
- device identity and logical capacity
- driver/runtime and compiler versions
- kernel code/resource hashes
- precision and robustness plan
- normalized workload-shape bucket
- candidate variants and synchronized CUDA-event durations
- sample size and probe overhead
- winning variant and confidence margin
- creation time, observation count, and invalidation reason

A calibration record is advisory. Admissibility, correctness, explicit pins,
and current memory limits always override it.

### `DeviceExecutionPlan`

Immutable output consumed by operation launchers:

- physical shape and selected variant
- `PrecisionPlan` and robustness plan
- cooperative width or work mapping
- work-bucket boundaries
- tile units and byte ceiling
- block size, grid cap/policy, and dynamic shared memory
- scratch arena estimate and admission token
- persistent metadata decision and amortization estimate
- calibration provenance and confidence
- allowed replan boundary
- complete reason and diagnostics

The plan must be validated once before launch. A caller may decline the plan
and request another admitted plan, but must not mutate fields independently.

## Planner Pipeline

### 1. Semantic And Shape Admission

The operation supplies an already-validated semantic contract, native carriers,
and physical work profile. Device planning never weakens public semantics.

### 2. Refresh Dynamic State

Read allocator and optional utilization state without synchronizing device
work. Static capabilities and compiled resource profiles come from caches.

### 3. Filter Variants

Remove variants that fail:

- device feature requirements
- precision or robustness guarantees
- physical shape or carrier requirements
- kernel resource limits
- scratch or output memory admission
- explicit execution/precision requests

At least one portable exact variant must remain for an advertised strict-native
shape.

### 4. Select Precision

Use the existing `PrecisionPlan` contract and the actual fp64 ratio. The
operation supplies coordinate statistics and predicate/metric/constructive
class. The device planner owns the selected plan in its final output.

### 5. Estimate Variant Cost

The first cost model should be analytic and intentionally small. It may compare:

- primitive operations per item or cooperative group
- expected inactive warp lanes from work-size distributions
- estimated bytes read/written
- count/scan/scatter and sorting overhead
- number of waves and partial final-wave waste
- expected occupancy from compiled resources
- persistent preparation cost amortized over expected reuse
- scratch pressure and required tile count

The model produces a ranking and uncertainty, not fictional nanosecond-perfect
predictions.

### 6. Apply Calibration

Use a valid calibration record when its shape bucket and resource hashes match.
Schedule a new bounded probe only when ranking uncertainty and remaining work
justify it. Calibration must use CUDA events on the active stream and stay
inside the operation's transient work budget.

### 7. Size Tiles And Launches

Tile sizing is the minimum of:

- operation structural maximum
- allocator budget after reserve and persistent state
- variant scratch/output formula
- index and integer-width limits
- latency or streaming chunk target

Grid policy uses actual SM count and active blocks per SM. Grid-stride kernels
should cap blocks near complete waves instead of launching an arbitrary grid
just over a wave boundary. Cooperative launches additionally respect the
resident-grid limit.

### 8. Emit Plan And Evidence

Return the immutable plan and one structured dispatch event. The event records
compact facts and reasons, not full histograms or device arrays.

## Calibration Policy

Calibration is intentionally narrow:

- never required for correctness
- never performed for a single admissible variant
- never performed when projected savings cannot repay probe cost
- never based on a public workflow or query identifier
- never allowed to allocate beyond the selected tile budget
- never allowed to introduce D2H result materialization
- never persisted without full invalidation metadata

Three modes are useful internally:

- `disabled`: static profile and analytic model only
- `observe`: time the already-selected production variant and update evidence
- `compare`: run a bounded representative comparison when amortized

`compare` is planner policy, not a public environment setting. Tests may force
it through private fixtures.

## Cache And Invalidation

Maintain separate caches for:

- static logical-device profile
- compiled kernel resources
- calibration records

Invalidate calibration when any of these change:

- device UUID or logical partition
- driver, CUDA runtime, NVRTC, or relevant CCCL version
- kernel source, compiler options, precision, or robustness hash
- resource profile
- planner schema or cost-model version
- material change in power/partition capacity if observable

Allocator pressure does not invalidate calibration; it changes the current
tile plan. Calibration keys must not contain public dataset or query names.

## Integration With Existing Runtime

`AdaptivePlan` remains the public-boundary execution plan. Add an optional
`device_execution` field and a complete-planning helper such as:

```python
adaptive = plan_adaptive_execution(...)
device_plan = plan_device_execution(
    adaptive_plan=adaptive,
    work_profile=work_profile,
    variants=registered_variants,
    kernel_resources=resources,
)
```

For complete adopters, `adaptive.precision_plan`, variant, and chunk decisions
must agree with `device_plan`; ideally the outer fields become projections of
the inner immutable plan. Transitional adopters may continue without the new
field.

Kernel launchers accept a plan or a narrow launch view derived from it. They do
not call global detection helpers independently.

## Adoption Levels

### Level 0: Existing adaptive path

Current runtime selection, precision, variant registry, and occupancy launch.
No migration required merely because ADR-0047 exists.

### Level 1: Capability-visible

The operation records device and kernel resource profiles and uses planner
memory admission, but retains one variant.

### Level 2: Static adaptive

Multiple variants use analytic cost and capability selection without empirical
calibration.

### Level 3: Complete usage

The operation supplies full work distributions, all material variants,
precision and robustness choices, bounded calibration, coherent tile/launch
planning, and structured evidence. No operation-local hardware policy remains.

Adaptive point-region refinement must reach Level 3. Library-wide Level 3
adoption is explicitly not required by this plan.

## Implementation Packages

### D0. Freeze contracts and baselines

- Add this plan and ADR-0047.
- Capture current planner latency, dispatch output, and representative public
  workflow profiles.
- Inventory direct device-property reads and hard-coded variant thresholds.
- Define cross-device fixture profiles for consumer, datacenter, constrained
  memory, and MIG-like logical partitions.
- Add the family rollout-state contract and require shadow mode before active
  selection.

Exit: contracts and before-state evidence are reviewable without a GPU.

### D1. Static and dynamic device profiles

- Add `DeviceIdentity`, `DeviceExecutionProfile`, and
  `DeviceExecutionSnapshot`.
- Extend CUDA runtime probing with direct attributes.
- Keep `DevicePrecisionProfile` as a compatibility projection.
- Cache static facts per active logical context.
- Refresh dynamic allocator state without device synchronization.

Exit: tests prove direct capabilities, conservative missing-attribute defaults,
and no product-name policy.

### D2. Compiled-kernel resource profiles

- Query registers, local memory, shared memory, maximum threads, and occupancy
  after CUBIN load.
- Key profiles by stable kernel and compilation hashes.
- Expose active-block calculations for candidate block/shared-memory choices.
- Add spill/resource diagnostics to warmup and review output.

Exit: a compiled variant can be rejected before launch when its resource shape
is impossible or predictably poor.

### D3. Complete plan and static cost model

- Add `DeviceVariantSpec`, `PhysicalWorkProfile`, and
  `DeviceExecutionPlan`.
- Extend the kernel registry without breaking existing registrations.
- Implement variant filtering, precision ownership, scratch admission, tile
  sizing, and wave-aware grid policy.
- Add structured reason logs and deterministic plan serialization for tests.

Exit: simulated 4090-like and H100-like profiles choose different coherent
plans for the same synthetic physical workload when economics differ.

### D4. Calibration and evidence cache

- Add bounded CUDA-event timing utilities that preserve stream ordering.
- Add `CalibrationRecord`, confidence rules, and invalidation keys.
- Implement disabled/observe/compare internal modes.
- Bound probe time and bytes through ADR-0045 budgets.
- Keep disk persistence optional until schema and invalidation tests are stable;
  session caching is sufficient for the first adopter.

Exit: calibration changes variant choice only with representative, reusable,
and sufficiently confident evidence.

### D5. Runtime integration and guardrails

- Attach the complete plan to `AdaptivePlan` for opted-in operations.
- Add launch views and validation so callers cannot mutate coherent fields.
- Add architecture lints against device-name performance branches and direct
  operation-local capability probes.
- Make explicit user execution/precision requests authoritative.
- Preserve one dispatch-event owner.
- Validate every accepted plan again at the launcher boundary and atomically
  return to the exact baseline on any pre-launch inconsistency.
- Add regression-triggered family demotion to baseline for subsequent calls;
  never change variants while a tile is executing.

Exit: a complete adopter has one planning entry and no duplicate policy.

### D6. First complete adopter

- Integrate the adaptive point-region refinement plan at Level 3.
- Validate real consumer and datacenter GPU selection.
- Use its edge-work histograms to exercise lane/warp/block planning and
  precision differences.
- Prove the runtime layer contains no PIP-specific concepts.

Exit: the adopter improves reusable shape canaries and public APIs on both
device classes without query-specific code.

### D7. Selective broader adoption

- Route future multi-variant kernel families through complete planning.
- Migrate existing families only when profile evidence shows hardware policy
  or fixed thresholds are limiting them.
- Keep simple single-variant operations on the lighter path.

Exit: adoption remains value-driven rather than a mechanical library rewrite.

## Testing Strategy

### CPU-only contract tests

- construct synthetic device, kernel, and workload profiles
- verify admissibility and explicit-pin precedence
- verify memory/tile formulas and overflow handling
- verify stable reason logs and plan serialization
- verify cache invalidation keys
- verify absent optional telemetry
- verify no device-name or benchmark-name policy

### GPU functional tests

- compare runtime attributes with CUDA-reported values
- validate function-resource profiles after compilation
- validate occupancy and cooperative-launch constraints
- validate stream-correct CUDA-event timing
- validate zero bulk D2H during planning
- validate allocator-pressure tile reduction

### Cross-device performance rails

Require at least one consumer-class and one datacenter-class device for mature
Level-3 adopters. Compare each device against its own current baseline; do not
require identical variants or speedups.

Record:

- selected precision and physical variant
- work buckets and cooperative widths
- tile and grid policy
- registers, occupancy, shared memory, and scratch
- calibration overhead and confidence
- device compute, memory, and allocator profile
- kernel and end-to-end wall time
- D2H, synchronization, materialization, and fallback events

## Performance And Safety Gates

- Planning without calibration must remain sub-millisecond after caches are
  warm for ordinary calls.
- Calibration cost must be bounded and amortized by predicted remaining work.
- No planner step may export row-, candidate-, or edge-sized device data.
- No planner allocation may bypass the active memory resource or reserve.
- The portable exact variant must remain available on every advertised device.
- Unknown capability combinations and expired evidence must execute the
  portable baseline, not the most optimistic compatible variant.
- A new adaptive family must not regress its simple/uniform baseline merely to
  improve a skewed case; it must retain or select the baseline path.
- The mandatory full end-to-end profile must show no new hidden compute
  materialization or CPU-heavy planning stage.

## Documentation And Handoff

When a kernel family adopts the planner, document:

- its ADR-0046 physical shape
- registered variants and exact admissibility
- work statistics supplied to planning
- native inputs and outputs
- precision and robustness behavior
- calibration family and cache key
- cross-device evidence
- retained portable baseline

Do not document frozen device thresholds in operation plans. Document the work
statistics and alternatives; the device planner owns the thresholds.
