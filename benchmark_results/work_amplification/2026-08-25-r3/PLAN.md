# Work-Amplification R3 Experiment Plan

Status: executed. Decisions and the broad queue are in `RESULTS.md` and
`OPPORTUNITY_MAP.md`.

This capsule executes the next evidence-gated investigations from
`docs/dev/work-amplification-evidence.md`. Experiments use current vibeSpatial
public operations over identical SF100 batches. Private device access is
permitted only inside a forced counterfactual recorder; no production selector
or API is added until an alternative wins complete public wall time.

## Q11 Parent-Aware Component Lowering

The parent control uses public `SpatialIndex.query_pair_aggregate` against the
original MultiPolygon rows. The alternative explodes valid MultiPolygons into
Polygon components outside timing, then uses public device-returning
`SpatialIndex.query` for the two aligned point indexes. Exact component hits
are mapped to stable parent rows on device. Unique `(point, parent)` keys are
formed independently for pickup and dropoff, intersected for shared parent
membership, and reduced to the ordinary Q11 scalar.

This is an exact set re-expression, not a component-count approximation:

- holes remain attached to their Polygon components;
- `(point, parent)` keys are deduplicated before counting;
- endpoints in different components of one parent intersect at parent level;
- null/empty parents are excluded from admission, while stable tri-state part
  order preserves exact invalid overlapping-component semantics;
- lean decision arms and instrumented attribution arms remain separate.

The alternative advances only if complete query + map + deduplicate +
intersection + reduction wall beats the parent control on the five real zone
frames, with identical terminal Q11 count, zero fallback, no host pair export,
and bounded peak-live memory. A one-frame smoke precedes the five-frame run.

If it advances, rerun the same parent-key lowering on Q10's single-endpoint
grouped aggregates. Otherwise archive it before production work.

## Constructive Follow-Ups

After the point-region decision, measure vegetation seam stitching and habitat
tile clipping. First add only missing output-coordinate and attributable-stage
counters. A counterfactual must preserve public attributes, geometry type,
ordering policy, nulls, and exact fingerprint. No rewrite graduates from a
compression ratio alone.

## Broad Review

Finally, rerun current public 10K, 1M, full-pipeline, and isolated SF100 rails
with validated static comparators. Rank opportunities by recoverable public
wall, then peak-live memory, recurrence, semantic confidence, and risk. The
result is a new opportunity map, not automatic device policy.
