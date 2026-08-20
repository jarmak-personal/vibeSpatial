---
id: ADR-0047
status: superseded
date: 2026-08-18
deciders:
  - vibeSpatial maintainers
tags:
  - architecture
  - gpu
  - runtime
  - dispatch
  - performance
  - portability
---

# Capability-Driven Device Execution Planning

## Supersession

This decision was superseded before implementation on 2026-08-18. Independent
principal GPU-hardware and YAGNI reviews agreed that the proposed layer was
premature: it duplicated existing adaptive-runtime ownership while modeling a
single launch too narrowly for the multi-stage execution graphs it intended to
control.

The preserved design remains useful research, but it is not implementation
authority. Active work follows
`docs/dev/evidence-first-point-region-execution-plan.md`: measure the existing
public path, prove one exact alternative on consumer and datacenter hardware,
extend current runtime owners minimally, and reconsider generic planning only
after a second kernel family demonstrates the same contract.

## Context

ADR-0007 introduced a probe-first adaptive runtime. ADR-0046 later made the
physical workload shape the required design unit for GPU work. Together they
answer two important questions: when GPU execution is admissible, and what
primitive work the operation contains.

They do not yet answer a third question completely: how the same admitted
physical shape should execute on materially different CUDA devices.

The current runtime detects the actual fp64-to-fp32 throughput ratio and uses
it through `PrecisionPlan`. NVRTC compiles for the active compute capability,
and the launch helper asks CUDA's occupancy API for a block size. These are
useful capabilities, but they do not form a complete device execution plan.
Variant thresholds, cooperative width, tile capacity, grid sizing, shared
memory use, prepared-index size, and calibration policy remain operation-local
or fixed.

That gap matters because a consumer GPU and a datacenter GPU can favor
different implementations of the same exact algorithm. A consumer device may
need staged fp32 plus selective fp64 refinement, while a datacenter device may
prefer native fp64. A device with more bandwidth or shared memory may keep a
direct kernel profitable over a larger work range. A device with less memory,
a MIG partition, or a constrained allocator needs smaller tiles even when its
product family normally has greater capacity.

If these choices are frozen from one development GPU, a reusable physical
shape can still be accidentally tuned to one machine. Product-name branches,
compute-capability folklore, and benchmark-specific thresholds do not solve
that problem.

## Decision

Adopt capability-driven device execution planning as a private runtime layer
between physical workload planning and kernel launch.

The decision sequence becomes:

1. Preserve public semantics and check operation admissibility.
2. Identify the physical workload shape under ADR-0046.
3. Measure or estimate shape-level work units and output requirements.
4. Capture the active logical device, allocator, and compiled-kernel
   capabilities.
5. Select precision, physical variant, cooperative width, tile size, launch
   geometry, and scratch budget as one coherent `DeviceExecutionPlan`.
6. Execute through native carriers and preserve the explicit public export
   boundary.
7. Record enough evidence to explain and reproduce the selection.

The planner is private. It must not introduce a public dataframe planner, a
new user-facing GPU object, or public device-specific tuning APIs. Existing
explicit execution and precision requests remain authoritative.

### Capability Sources

Planning must use capabilities and measured behavior rather than product
names. Inputs may include:

- logical-device identity, including MIG or other partition identity
- compute capability as a feature and compilation target, not a performance
  proxy by itself
- SM count, warp size, thread and block limits
- register-file, shared-memory, and L2 capacities
- reported fp64-to-fp32 throughput ratio
- allocator ceiling, reserve, live bytes, largest allocatable block, and
  admission state
- compiled-kernel registers, static/dynamic shared memory, maximum threads,
  and occupancy limits
- optional utilization and memory-pressure telemetry
- bounded calibration results for the exact kernel variant and workload shape

When an attribute is unavailable, the planner must use a conservative declared
default and make that fact observable. It must not infer missing performance
properties from a marketing name.

### Planning Contracts

The complete planning contract consists of:

- a static `DeviceExecutionProfile`
- a dynamic `DeviceExecutionSnapshot`
- a compiled `KernelResourceProfile`
- an ADR-0046 `PhysicalWorkEstimate` plus optional device-resident shape
  histograms
- registered `DeviceVariantSpec` alternatives with exact admissibility and
  scratch formulas
- an optional, bounded `CalibrationRecord`
- one immutable `DeviceExecutionPlan`

The final plan owns the selected precision plan, kernel variant, work
partition, cooperative width, tile capacity, block/grid policy, dynamic shared
memory, scratch budget, replan boundary, and reason log. Call sites must not
independently recompute those choices after accepting the plan.

`AdaptivePlan` may carry a `DeviceExecutionPlan` while older operations
continue using their existing fields. Complete library-wide adoption is not a
precondition for making the layer production-ready. The implementation must,
however, support complete usage by an operation without operation-local
hardware policy.

### Calibration

Static attributes are necessary but insufficient for irregular geometry
kernels. The planner may perform bounded first-use or first-chunk calibration
when all of the following are true:

- two or more exact variants are admissible
- remaining work is large enough to amortize the probe
- the probe uses representative physical work, not a benchmark name
- probe output is discarded or verified against the authoritative result
- the probe introduces no hidden host materialization or unbounded allocation

Calibration results are keyed by logical-device identity, driver/runtime
versions, compiled-kernel hash, precision variant, and relevant workload-shape
class. Stale or low-confidence records are ignored. Calibration must not run on
every call, and absence of calibration must leave a correct static plan.

### Portability And Specialization

Portable CUDA variants remain the baseline. Architecture-specific variants,
such as a Hopper-only asynchronous shared-memory path, are allowed only behind
capability admission and must have an exact portable alternative.

The planner may produce different execution plans and performance on different
devices. It may not produce different public semantics. Cross-device bitwise
identity is governed separately by ADR-0031; predicate truth and topology
remain exact regardless of plan.

### Adoption Policy

The device planner is mandatory for new work that exposes multiple materially
different GPU execution variants or whose safe precision, cooperative width,
or memory shape depends on the device. Existing kernels may migrate
incrementally when profiling shows value.

The first complete adopter will be adaptive exact point-region refinement.
That project is evidence for the planner contract, not the reason for baking
point-in-polygon concepts into the runtime layer.

### Safety Boundary

Device execution planning is a high-blast-radius subsystem. A planner defect
can repeatedly select a slow path, oversize temporary work, amplify
synchronization, or reuse stale calibration across every adopting operation.
Precision or robustness mistakes can additionally become correctness defects.

The planner therefore selects only among variants whose semantic, precision,
robustness, resource, and coverage contracts have already been proven. Cost or
calibration evidence may rank admissible variants; it may never make an unsafe
variant admissible.

Every adaptive family retains a portable exact baseline. Missing capabilities,
stale cache state, low calibration confidence, planner exceptions, inconsistent
resource metadata, and memory pressure fail closed to that baseline. Failure
to plan must not silently fall back to CPU in strict-native mode.

Adoption proceeds through observe-only shadow planning, family-scoped opt-in,
validated default selection, and only then broader reuse. One operation's
successful adoption does not authorize automatic activation elsewhere.

Plans are immutable and validated at the launcher boundary. Work coverage,
integer limits, memory admission, stream readiness, and explicit user pins are
checked independently of the performance cost model. Replanning occurs only at
declared safe boundaries and uses confidence margins or hysteresis to prevent
variant oscillation.

This ADR amends ADR-0002, ADR-0006, ADR-0007, ADR-0033, and ADR-0046. It
builds on ADR-0040, ADR-0044, and ADR-0045.

## Consequences

- Device-specific execution choices become inspectable runtime policy rather
  than constants scattered through operation modules.
- Development on one GPU no longer licenses thresholds for every target GPU.
- Kernel families can share profiling, calibration, cache invalidation,
  memory admission, and explanation infrastructure.
- Planning becomes more complex and needs strict latency, synchronization,
  and metadata budgets.
- The portable baseline and shadow/rollback machinery remain permanent safety
  infrastructure, not temporary scaffolding removed after tuning.
- A complete adopter must register meaningful alternatives and work estimates;
  wrapping one fixed kernel in a plan object is not compliance.
- Real-device performance gates must cover at least one consumer and one
  datacenter profile before a cross-device adaptive family is declared mature.
- Operations with one obvious portable variant may continue using the lighter
  existing adaptive path.

## Alternatives Considered

- **Tune globally on the primary development GPU.** Rejected because precision,
  bandwidth, cache, shared memory, occupancy, and capacity economics differ
  substantially across supported devices.
- **Maintain product-name threshold tables.** Rejected because board variants,
  clocks, power limits, MIG partitions, drivers, and allocator constraints make
  names unreliable execution contracts.
- **Use compute capability as the entire policy.** Rejected because compute
  capability describes features, not the active device's complete performance
  or memory envelope.
- **Autotune every operation call.** Rejected because duplicate work, launch
  overhead, and cache instability would dominate transient and small workloads.
- **Require immediate library-wide migration.** Rejected because it would turn
  a useful runtime layer into a broad refactor gate. The planner must be
  complete and incrementally adoptable.
- **Keep all policy inside each kernel family.** Rejected because it repeats
  capability probing, calibration, caching, memory admission, and explanation
  logic while making cross-device review impractical.
