---
name: dispatch-wiring
description: "PROACTIVELY USE THIS SKILL when wiring a new operation into the vibeSpatial Python dispatch stack: public GeoPandas APIs, GeometryArray, GeoBase, DeviceGeometryArray, OwnedGeometryArray helpers, CPU fallback observability, strict-native behavior, and dispatch wrappers. Also use it for ADR-0044 private native substrate and ADR-0046 physical workload shape wiring: NativeFrameState, NativeRowSet, NativeRelation, NativeGrouped, NativeSpatialIndex, NativeGeometryMetadata, NativeExpression, export boundaries, shape-level work estimates, and avoiding pandas-shaped hot paths. Trigger on: \"wire dispatch\", \"add API\", \"add method\", \"GeometryArray\", \"DGA method\", \"CPU fallback\", \"dispatch wrapper\", \"public API\", \"coerce\", \"OwnedGeometryArray\", \"NativeRelation\", \"NativeRowSet\", \"NativeSpatialIndex\", \"physical workload shape\"."
---

# Dispatch Wiring — vibeSpatial

Use this skill for Python-side dispatch wiring outside the kernel itself.
Target task: **$ARGUMENTS**

## Open First

- `src/vibespatial/api/geometry_array.py`
- `src/vibespatial/api/geo_base.py`
- `src/vibespatial/runtime/adaptive.py`
- `src/vibespatial/runtime/dispatch.py`
- `src/vibespatial/runtime/fallbacks.py`
- `docs/architecture/api-dispatch.md`
- `docs/decisions/0044-private-native-execution-substrate.md`
- `docs/decisions/0046-gpu-physical-workload-shape-contracts.md`
- the nearest current implementation for the operation family

Start from the closest existing operation. Do not invent a new dispatch shape
unless the existing ones fail for a concrete reason.

## Current Model

Usual flow:

```text
public API helper
-> GeometryArray / GeoBase surface
-> owned helper or evaluate_geopandas_* wrapper
-> plan_dispatch_selection(...)
-> GPU path or explicit CPU fallback
-> observable dispatch / fallback event
```

Native substrate flow:

```text
public compatibility ingress
-> admitted local lowering
-> private native carrier(s)
-> sanctioned native consumer(s)
-> explicit public export boundary
```

Current repo truth:

- `plan_dispatch_selection()` returns an `AdaptivePlan`, not a bare
  `RuntimeSelection`.
- Reuse `selection.precision_plan`; do not recompute precision separately when
  the plan already owns it.
- Dispatch-event ownership belongs at the layer that actually decides GPU vs
  CPU. Do not double-log.
- Public dispatch shape and physical workload shape are different. Preserve the
  public API shape, but choose or reuse a device-native physical shape from
  ADR-0046.

## Physical Shape and Native Carrier Rules

- Before wiring, name the physical workload shape and native output carrier.
- `WorkloadShape` cardinality helpers are not the ADR-0046 taxonomy. Do not
  treat pairwise/broadcast/scalar as sufficient physical-shape proof.
- Preserve `NativeFrameState`, `NativeRowSet`, `NativeRelation`,
  `NativeGrouped`, `NativeSpatialIndex`, `NativeGeometryMetadata`, and
  `NativeExpression` when lineage, index semantics, and admissibility remain
  valid.
- Spatial indexes and geometry metadata are execution state. Rebuilding them
  because a public object boundary changed is a performance bug unless the
  lineage or parameters changed.
- Feed shape-level estimates into dispatch when available: coordinates,
  vertices, segments, candidate pairs, relation pairs, groups, output rows,
  output bytes, and expected temporary bytes.
- Public GeoDataFrame, GeoSeries, pandas, and Shapely objects are ingress,
  fallback, debug, or export surfaces. They are not the hot internal execution
  currency for GPU-selected native paths.

## Decision Tree

1. **Simple unary/property operation**
   - Add or update the `GeometryArray` method.
   - Route to the nearest `*_owned()` helper.
   - Use this for most metric/property surfaces.

2. **Host-surface orchestration or richer fallback behavior**
   - Use an `evaluate_geopandas_*` helper.
   - Keep policy, alignment, and materialization in the wrapper.
   - Use this for operations that already have a host-oriented wrapper shape.

3. **Binary predicate**
   - Follow `GeometryArray._binary_method()` and `src/vibespatial/predicates/`.
   - Keep predicate-family coercion and family checks in the predicate layer.

4. **Binary constructive**
   - Follow `_constructive_or_fallback()` and
     `binary_constructive_owned()`.
   - Expect more explicit workload/routing logic than a simple unary helper.

5. **DeviceGeometryArray surface**
   - Add only when there is real device-native value.
   - Do not mirror every `GeometryArray` method by default.

## Public Surface Rules

- Keep public GeoPandas-facing delegation in `src/vibespatial/api/geo_base.py`.
- `_delegate_property`, `_delegate_geo_method`, and `_binary_op` remain the
  normal public entry points.
- Preserve GeoPandas alignment and return-shape expectations on binary methods.
- Keep CRS, null, and row-count behavior aligned with the nearest existing API.

## Owned Helper Rules

- Accept `dispatch_mode` and `precision` when the public surface already
  supports them.
- Call `plan_dispatch_selection()` with the best residency/workload/coordinate
  information available. Prefer shape-level work estimates over row count when
  the API supports them.
- Use `selection.runtime_selection` and `selection.precision_plan` from the
  returned `AdaptivePlan`.
- Record a dispatch event in the helper only if the helper owns the runtime
  choice.
- Record a fallback event only at the explicit CPU fallback boundary.
- If the array layer or wrapper already records the event, the helper must not
  record it again.

## Coercion Rules

There is no universal coercion helper in `geometry/owned.py`.

Use the coercion shape that matches the surface:

- Predicate-style family-checked coercion:
  `src/vibespatial/predicates/support.py::coerce_geometry_array`
- Generic `GeometryArray` / `GeoSeries` inputs: prefer `.to_owned()`
- Scalar or host geometry sequences: use `from_shapely_geometries(...)`
- `DeviceGeometryArray` binary methods: use `_coerce_other_to_owned()`

Do not widen predicate-specific helpers into global coercion utilities without
an explicit design reason.

## DeviceGeometryArray Rules

- Return `DeviceGeometryArray._from_owned(...)` for geometry outputs that stay
  device-native.
- If a DGA method must materialize host state, emit the explicit shapely/host
  fallback event at that boundary.
- Keep DGA methods thin. Reuse owned helpers or existing wrappers rather than
  forking logic.

## Non-Negotiables

- No silent CPU fallback across an advertised native boundary.
- No duplicate dispatch events.
- No new public dispatch shape when an existing local pattern already fits.
- No pandas-shaped internal execution when an admissible native carrier exists.
- No generic coercion abstraction unless at least two families truly need it.
- Start from the nearest current operation, not from stale skill examples.

## Good Anchors

Use these as current patterns, not as copy-paste templates:

- simple owned metric/property path:
  `src/vibespatial/constructive/measurement.py`
- host wrapper with explicit event ownership:
  `src/vibespatial/api/geometry_array.py`
- binary constructive dispatch:
  `src/vibespatial/constructive/binary_constructive.py`
- make-valid pipeline:
  `src/vibespatial/constructive/make_valid_pipeline.py`
- DGA surface:
  `src/vibespatial/geometry/device_array.py`

## Verify

Run the narrowest checks that cover the changed surface, then expand:

```bash
uv run pytest tests/test_runtime_policy.py -q
uv run pytest tests/test_geopandas_dispatch.py -q
uv run python scripts/check_docs.py --check
```

If runtime, kernel, pipeline, IO, or predicate code paths changed, also obey
the repo-wide end-to-end profile gate from `AGENTS.md`.
