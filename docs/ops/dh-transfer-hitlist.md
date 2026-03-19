# D/H Transfer Elimination Hitlist

## Goal

Eliminate unnecessary device-to-host and host-to-device transfers to make
vibeSpatial a pure GPU library.

## Rules

- **Each fix lands in its own commit.** This makes review easier and lets us
  confirm no regressions per change.
- Run the narrow verification gate for the edited surface after each fix.
- Run `uv run python scripts/check_zero_copy.py --all` after each fix to
  confirm the ratchet baseline decreases.
- Current baseline: **48 known violations** (as of 2026-03-19).
- Items 1-5 landed in commit df557a7 (unique_tag_pairs utility).

---

## TIER 1 -- HOT PATH (per-geometry / per-predicate, highest impact)

- [x] **1. binary_predicates.py:766-771** -- D->H `cp.asnumpy(d_cand_rows)`,
      `d_cand_ltags`, `d_cand_rtags` pulled to host for Python
      `set(zip(...tolist()))` tag grouping.
      *Fix: GPU radix sort + unique on tag pairs.*

- [x] **2. binary_predicates.py:656** -- D->H `set(zip(left_tags.tolist(),
      right_tags.tolist()))` Python set on host.
      *Fix: Same GPU unique tag-pair kernel as #1.*

- [x] **3. spatial_nearest.py:1771, 2017** -- D->H identical
      `set(zip(...tolist()))` pattern.
      *Fix: GPU unique tag-pair extraction.*

- [x] **4. spatial_query_utils.py:494** -- D->H same `set(zip(...tolist()))`
      on DE-9IM tags.
      *Fix: GPU unique tag-pair extraction.*

- [x] **5. distance_owned.py:261** -- D->H same
      `set(zip(valid_left_tags.tolist(), ...))`.
      *Fix: GPU unique tag-pair extraction.*

- [x] **6. point_in_polygon.py:1036, 1158, 2116** -- D->H candidate rows
      pulled to host for Python binning logic.
      *Fix: Device-side binning kernel.*

- [x] **7. point_in_polygon.py:1056, 1125, 1175, 1246** -- H->D per-bin
      candidate subsets re-uploaded **in a loop**.
      *Fix: Keep bins on device.*

- [x] **8. point_in_polygon.py:1428-1429, 1553-1554** -- H->D mask + output
      arrays uploaded per dense kernel launch.
      *Fix: Allocate on device directly.*

- [x] **9. point_in_polygon.py:1471, 2077, 2176, 2196** -- D->H dense result
      array copied to host (unless `_return_device=True`).
      *Fix: Make device-return the default internal path.*

- [x] **10. overlay_gpu.py:1564** -- D->H
      `list(cp.asnumpy(...).astype(int))` cycle source rows.
      *Fix: Keep on device, use GPU scatter.*

- [x] **11. overlay_gpu.py:1599** -- D->H
      `list(cp.asnumpy(cp.flatnonzero(d_hole_mask)))`.
      *Fix: Device-side hole indexing.*

- [x] **12. overlay_gpu.py:1808, 1866, 1895** -- D->H->Python loop
      `for ext_idx in host_array.tolist():`.
      *Fix: GPU gather kernel.*

- [x] **13. overlay_gpu.py:1182-1183, 1210-1211** -- D->H
      `coords[:, 0].tolist()` in coordinate assembly (Python list.extend
      per geometry).
      *Fix: GPU coordinate gather.*

- [x] **14. segment_primitives.py:323-332** -- D->H `xs0.tolist()` etc.
      segment coords pulled to Python lists.
      *Fix: Keep segments as device arrays.*

- [x] **15. segment_primitives.py:355, 795, 995, 1171** -- D->H->loop
      `for row in rows.tolist():` Python loops over device-computed indices.
      *Fix: GPU scatter/gather.*

---

## TIER 2 -- WARM PATH (per-operation, amortized but avoidable)

- [x] **16. device_geometry_array.py:1776-1779** -- D->H
      `float(min_xy[0].item())` x4 for total_bounds.
      *Fix: Single kernel returning 4 scalars.*

- [x] **17. spatial_query_utils.py:366-367** -- H->D
      `cp.asarray(query_owned.tags)` uploaded every query.
      *Fix: Keep tags device-resident on OwnedGeometryArray.*

- [x] **18. spatial_query_utils.py:399-401** -- D->H
      `cp.asnumpy(d_left_tags)`, `d_right_tags`, `d_gpu_pair_mask`.
      *Fix: Filter on device, avoid host materialization.*

- [x] **19. spatial_nearest.py:1975-1976** -- H->D `cp.asarray(left_idx)`,
      `cp.asarray(right_idx)`.
      *N/A: Transfers are structurally required -- only reached in CPU-candidate
      path where indices genuinely start on host. GPU path uses device arrays directly.*

- [x] **20. wkb_decode.py:637-1246 (9 sites)** -- D->H
      `int(row_indexes.size)` scalar sync per geometry family (up to 6x).
      *Fix: Consolidate sizing into single kernel.*

- [x] **21. wkb_decode.py:724, 1019, 1023-1024** -- D->H
      `int(geometry_offsets[-1])` offset terminators for allocation.
      *Fix: Fused allocation kernel.*

- [x] **22. overlay_gpu.py:1570, 1628, 1711, 1734** -- D->H scalar
      `int(cp.asnumpy(...)[0])` extractions for offset counts.
      *Fix: Batch scalar reads.*

- [x] **23. overlay_gpu.py:2674-2675** -- D->H `cp.asnumpy(d_lc)`,
      `cp.asnumpy(d_rc)` coverage matrices.
      *Fix: Keep coverage on device.*

- [x] **24. dissolve_pipeline.py:279-284** -- H->D->D->H upload codes/bounds,
      sort on device, pull back sorted results.
      *Fix: Keep sorted results on device.*

- [x] **25. cccl_primitives.py:205, 627** -- D->H `int(count.get()[0])`
      compaction result counts.
      *Fix: Use device-side count.*

- [x] **26. io_pylibcudf.py (40+ sites)** -- D->H scalar
      `int(cp.asnumpy(offsets[n]))` throughout.
      *N/A: Cold-path I/O (52 syncs, ~310us worst case, runs once per data load.
      Most are unbatchable data-dependent loop bounds or sequential offset chains.
      File is already excluded from zero-copy lint via io_ prefix.)*

---

## TIER 3 -- API LAYER (Shapely round-trips breaking device residency)

- [x] **27. api/geometry_array.py:598** -- H->D
      `from_shapely_geometries(self._data.tolist())` the lazy owned init.
      *Fix: Native device construction (WKB/GeoArrow direct to device).*

- [x] **28. api/geometry_array.py:679** -- CPU
      `shapely.get_type_id(self._data)` forces host geom_type check.
      *Fix: Device-side type tags on OwnedGeometryArray.*

- [x] **29. api/geometry_array.py:701, 710** -- CPU fallback
      `shapely.area()`, `shapely.length()` when `_owned is None`.
      *N/A: Already fixed — area_owned/length_owned fast paths already exist
      at lines 761-773 using measurement_kernels.*

- [x] **30. api/geometry_array.py:1573** -- CPU `shapely.bounds(self._data)`
      always host.
      *Fix: Added _owned fast path using compute_geometry_bounds.*

- [x] **31. api/geometry_array.py:871, 793, 1161** -- D->H
      `result_owned.to_shapely()` / `np.asarray(owned, dtype=object)` GPU
      results force-materialized back.
      *N/A: Structural — GeoPandas requires Shapely arrays for GeometryArray._data.
      Future: cache _owned on returned GeometryArray to avoid re-serialization.*

- [x] **32. api/geometry_array.py:1009, 1042, 1239, 1253** -- CPU fallback
      `shapely.dwithin()`, `shapely.clip_by_rect()`, `shapely.union_all()`,
      `shapely.intersection_all()`.
      *N/A: dwithin/clip_by_rect already have GPU paths (fallbacks are safety nets).
      union_all GPU exists in DeviceGeometryArray (wire to GeometryArray: future).
      intersection_all needs new GPU kernel (low priority, rarely used).*

- [x] **33. api/sindex.py:458** -- H->D
      `from_shapely_geometries(geometry.tolist())` query input conversion.
      *Fix: Accept device arrays directly.*

- [x] **34. api/sindex.py:56-64** -- D->H
      `np.asarray(self._geometry_array._data, dtype=object)` STRtree fallback.
      *N/A: Lazy STRtree construction already avoids materialization for GPU queries.
      Remaining gap: `crosses` predicate lacks DE-9IM refinement (small future fix).*

- [x] **35. stroke_kernels.py:533-534, 751, 786, 811** -- H->D
      `from_shapely_geometries(geometries.tolist())` stroke results re-uploaded.
      *Fix: Keep stroke output as device arrays.*

- [x] **36. clip_rect.py:421, 1892** -- H->D
      `from_shapely_geometries(shapely_values.tolist())` clip results
      re-uploaded.
      *Fix: Device-side clip kernel.*

- [x] **37. distance_owned.py:162, 170** -- H->D
      `from_shapely_geometries(left.tolist())` distance input upload.
      *Fix: Accept device arrays; skip Shapely intermediate.*

- [x] **38. profile_rails.py:97, 107, 228, 238** -- H->D
      `from_shapely_geometries(values.tolist())` profiler input conversion.
      *Fix: Profile with device-native inputs.*

- [x] **39. normalize_gpu.py:255** -- H->D
      `from_shapely_geometries(result.tolist())`.
      *N/A: Structural — CPU fallback path only. GPU normalize path exists
      and is complete (NVRTC kernels, ADR-0002 compliant). Transfer only
      reached when GPU unavailable or input < 500 rows.*

---

## TIER 4 -- COLD PATH (I/O boundaries, acceptable per ADR-0005)

These are at materialization boundaries and are generally acceptable:

- io_geojson_gpu.py -- initial byte upload, feature offset extraction
- io_wkb.py -- WKB payload upload, bounds setup
- io_pylibcudf.py -- cuDF column metadata extraction (scalar reads in T2)
- make_valid_gpu.py -- fallback CPU path (intentional, commented)
- overlay_gpu.py:1783-1788 -- final coordinate materialization (ADR-0005)
- indexing.py -- spatial index serialization/deserialization

---

## Recurring Anti-Patterns

1. **`set(zip(tags.tolist(), tags.tolist()))`** -- 5 files. #1 pattern to
   kill. A GPU `unique_pairs` kernel eliminates all of them.

2. **`from_shapely_geometries(array.tolist())`** -- 15+ sites. Foundational
   Shapely->device serialization bottleneck. Direct WKB-to-device or
   GeoArrow-to-device path eliminates the Python list intermediate.

3. **`for row in device_array.tolist():`** -- Python loops over
   device-computed indices in segment_primitives and overlay_gpu.

4. **Scalar extraction via `int(cp.asnumpy(arr[-1]))` or `.item()`** --
   dozens of sync points. Batch reads or fused sizing kernels.

5. **`shapely.*()` as fallback in API layer** -- `get_type_id`, `bounds`,
   `area`, `length`, `is_valid` force host computation even when device data
   exists.

## Recommended Attack Order

1. GPU unique tag-pair kernel (kills anti-pattern #1 across 5 files)
2. Device-side binning in point_in_polygon (kills per-bin H->D loop)
3. Device-resident result returns (make `_return_device=True` default)
4. Device-side type tags (eliminate `shapely.get_type_id()`)
5. Direct WKB/GeoArrow->device construction (eliminate `from_shapely_geometries(x.tolist())`)
6. GPU segment/overlay coordinate gather (kill Python loops)
7. Batch scalar reads (consolidate `int(arr[-1])` sync points)
