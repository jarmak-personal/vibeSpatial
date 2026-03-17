# Mixed Geometries

<!-- DOC_HEADER:START
Scope: Mixed-geometry storage and execution strategy for GPU-oriented geometry buffers.
Read If: You are designing owned geometry buffers, mixed-type dispatch, or geometry-family layout policy.
STOP IF: Your task is limited to a single kernel implementation detail that already has a chosen layout.
Source Of Truth: Phase-1 design decision for mixed-geometry handling before buffer implementation.
Body Budget: 158/260 lines
Document: docs/architecture/mixed-geometries.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-5 | Preamble |
| 6-10 | Intent |
| 11-20 | Request Signals |
| 21-27 | Open First |
| 28-33 | Verify |
| 34-39 | Risks |
| 40-48 | Candidates |
| 49-62 | Decision |
| 63-72 | Why |
| 73-89 | Benchmark Method |
| 90-111 | Results |
| 112-124 | Thresholds |
| 125-135 | Buffer Implications |
| 136-145 | Rejections |
| ... | (2 additional sections omitted; open document body for full map) |
DOC_HEADER:END -->

GeoPandas permits mixed geometry families in one array. This doc defines the
default storage and execution strategy before owned geometry buffers land.

## Intent

Choose a mixed-geometry handling strategy that minimizes divergence without
making the canonical buffer model overly fragmented or wasteful.

## Request Signals

- mixed geometry
- tagged union
- sort partition
- soa
- geometry families
- buffer layout
- divergence

## Open First

- docs/architecture/mixed-geometries.md
- src/vibespatial/testing/mixed_layouts.py
- scripts/benchmark_mixed_layouts.py
- docs/implementation-order.md

## Verify

- `uv run pytest tests/test_mixed_layouts.py`
- `uv run python scripts/benchmark_mixed_layouts.py --scales 100000 1000000`
- `uv run python scripts/check_docs.py --check`

## Risks

- Canonical storage and execution strategy are easy to conflate; they should not be the same decision.
- GeometryCollections remain a pathological mixed-type case and should not force the common fast path.
- A storage model that optimizes memory at ingest can still lose badly once divergent kernels dominate runtime.

## Candidates

The four candidate approaches are:

- separate typed arrays: split by family, execute family-specific kernels, then rejoin
- tagged union: one mixed array with family tags and offsets, execute in original order
- sort-partition: keep one logical array, but sort or partition to homogeneous chunks for execution
- promote to common type: coerce everything to one family such as polygon

## Decision

Use a hybrid strategy:

- canonical storage: dense tagged representation with family tags and child-relative offsets
- execution default for truly mixed inputs: sort-partition by coarse family (`point`, `line`, `polygon`)
- execution fast path for near-homogeneous inputs: direct tagged execution without repartition
- reject common-type promotion as a default strategy
- do not make permanently separated typed arrays the canonical user-visible storage

This means candidate `B` wins as the storage model and candidate `C` wins as
the mixed-execution model. Candidate `A` remains a useful internal cache or
kernel-local staging shape, not the primary persisted array contract.

## Why

- Tagged storage preserves original ordering and keeps API semantics simple.
- Sort-partition execution removes the worst warp divergence when the array is
  materially mixed.
- Permanently separate typed arrays force split and rejoin logic into every
  consumer and complicate row-wise pandas alignment.
- Promotion to polygon-like common type is semantically wrong for points and
  lines and can distort both memory and algorithm choice.

## Benchmark Method

The current benchmark in `scripts/benchmark_mixed_layouts.py` measures:

- metadata and payload byte estimates from synthetic geometry families
- partitioning or reorder cost at `100K` and `1M` rows
- warp-purity proxy on the original order as a divergence signal

Representative mixes:

- point-dominated: `90 / 8 / 2`
- polygon-dominated: `5 / 15 / 80`
- mixed: `40 / 30 / 30`

These are design-stage layout benchmarks, not final kernel throughput numbers.
Actual GPU throughput still needs validation in later benchmark rails.

## Results

Measured on this repo checkout with the synthetic payload model:

| Dataset | Scale | Tagged purity | Tagged prep ms | Sort-partition prep ms | Tagged payload MB | Sort-partition MB | Recommendation |
|---|---:|---:|---:|---:|---:|---:|---|
| point-dominated | 100K | 0.900 | 0.16 | 0.71 | 3.43 | 4.73 | direct tagged execution |
| point-dominated | 1M | 0.900 | 1.74 | 7.89 | 34.28 | 47.28 | direct tagged execution |
| polygon-dominated | 100K | 0.800 | 0.15 | 0.38 | 19.84 | 21.14 | tagged with optional late partitioning |
| polygon-dominated | 1M | 0.800 | 1.57 | 7.02 | 198.40 | 211.40 | tagged with optional late partitioning |
| mixed | 100K | 0.436 | 0.10 | 0.27 | 11.58 | 12.88 | sort-partition execution |
| mixed | 1M | 0.435 | 1.08 | 5.82 | 115.80 | 128.80 | sort-partition execution |

Interpretation:

- Metadata overhead is not the deciding factor between tagged and separated layouts.
- Divergence risk becomes the real problem once warp purity drops well below
  the `0.70` to `0.80` range.
- Sort-partition adds modest metadata and reorder cost relative to the payload
  sizes at `100K` and `1M`, which makes it a good execution-time trade when the
  mix is genuinely heterogeneous.

## Thresholds

Use these provisional thresholds until adaptive runtime work lands:

- dominant-family share `>= 88%`: execute directly from tagged storage
- dominant-family share `70%` to `< 88%`: default to tagged execution, allow
  kernel-specific late partitioning if profiling shows divergence pain
- dominant-family share `< 70%` and row count `>= 10K`: partition before execution
- row count `< 10K`: prefer tagged execution unless a kernel proves otherwise

`o17.2.10` should eventually replace these fixed thresholds with observed
runtime-driven switching.

## Buffer Implications

`o17.2.1` should assume:

- one logical mixed array may contain multiple geometry families
- the canonical metadata must include at least family tag and family-relative offset
- partitioned execution should be able to materialize permutation buffers
  without copying full payload data
- row-order restoration must be cheap and explicit
- GeometryCollections can stay on a slow path or explicit fallback path early

## Rejections

Reject as defaults:

- sparse-union-like promotion of all rows to the same payload shape
- permanent split-by-family storage as the only canonical representation

Both can still exist as specialized internal views, but neither should define
the Phase 2 buffer contract.

## Next Consumers

- `o17.2.1` should use this doc as the storage-layout decision input.
- `o17.2.10` should treat the thresholds here as the first adaptive-policy baseline.
- `o17.6.1` should plan to preserve pandas row order even when execution partitions by family.

## Verification

```bash
uv run pytest tests/test_mixed_layouts.py
uv run python scripts/benchmark_mixed_layouts.py --scales 100000 1000000
uv run python scripts/check_docs.py --check
```
