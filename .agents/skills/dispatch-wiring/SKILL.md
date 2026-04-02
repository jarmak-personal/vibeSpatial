---
name: dispatch-wiring
description: "PROACTIVELY USE THIS SKILL when wiring a new operation into the vibeSpatial Python dispatch stack — adding a public API method, connecting to GeometryArray, writing the dispatch wrapper, adding CPU fallbacks, handling OwnedGeometryArray coercion, or updating DGA/GeoPandas surface methods. This is the Python-side complement to $new-kernel-checklist (which covers the kernel itself). Trigger on: \"wire dispatch\", \"add API\", \"add method\", \"GeometryArray\", \"DGA method\", \"CPU fallback\", \"dispatch wrapper\", \"public API\", \"coerce\", \"OwnedGeometryArray\"."
---

# Dispatch Wiring Guide — vibeSpatial

You are wiring a new operation into vibeSpatial's Python dispatch stack.
This skill covers everything OUTSIDE the kernel: from the GeoPandas-
compatible public API down to the dispatch wrapper that calls the GPU/CPU
implementation.

Target operation: **$ARGUMENTS**

---

## The Dispatch Stack (10 layers)

```
GeoSeries.area / .within(other) / .buffer(distance)     [Layer 1: Public API]
  |
_delegate_property() / _binary_op() / _delegate_geo_method()  [Layer 2: Delegation]
  |
GeometryArray.area / ._binary_method() / .buffer()       [Layer 3: GeometryArray]
  |
  +-- if self._owned: dispatch to owned-path              [Layer 4: Owned routing]
  |     |
  |     area_owned() / evaluate_geopandas_binary_predicate()  [Layer 5: Dispatch wrapper]
  |       |
  |       plan_dispatch_selection() -> RuntimeSelection   [Layer 6: Runtime selection]
  |       select_precision_plan() -> PrecisionPlan        [Layer 7: Precision planning]
  |       |
  |       +-- GPU: _area_gpu(owned, precision_plan)       [Layer 8: GPU kernel]
  |       +-- CPU: _area_cpu(owned)                       [Layer 9: CPU fallback]
  |
  +-- else: shapely.area(self._data)                      [Layer 10: Shapely fallback]
```

---

## Step 1: Classify Your Operation

| Type | Examples | Entry Pattern | Return Type |
|------|----------|---------------|-------------|
| **Property** | area, length, bounds | `_delegate_property()` | `Series[float64]` |
| **Unary method** | centroid, make_valid, simplify, normalize | `_delegate_geo_method()` | `GeoSeries` |
| **Binary predicate** | within, contains, intersects, touches | `_binary_op()` | `Series[bool]` |
| **Binary constructive** | intersection, union, difference | `_binary_op()` | `GeoSeries` |
| **Parameterized** | buffer(distance), offset_curve(distance) | `_delegate_geo_method()` | `GeoSeries` |

---

## Step 2: GeometryArray Method

**File:** `src/vibespatial/api/geometry_array.py`

This is where the owned-path routing happens. Add your method here.

### Pattern A: Property (area, length)

```python
@property
def area(self):
    if self._owned is not None:
        from vibespatial.constructive.measurement import area_owned
        return area_owned(self._owned)
    return shapely.area(self._data)
```

### Pattern B: Unary method returning geometry (centroid, make_valid)

```python
def centroid(self):
    if self._owned is not None:
        from vibespatial.constructive.point import centroid_owned
        result_owned = centroid_owned(self._owned)
        return GeometryArray.from_owned(result_owned, crs=self.crs)
    return GeometryArray(shapely.centroid(self._data), crs=self.crs)
```

### Pattern C: Binary predicate (within, intersects)

Binary predicates route through `_binary_method()` which already handles
the dispatch. To add a new predicate:

1. Register it in `src/vibespatial/predicates/binary.py` in
   `supports_binary_predicate()`.
2. Ensure the DE-9IM evaluation path in
   `_evaluate_gpu_de9im_candidates()` handles it.
3. No changes needed to `GeometryArray._binary_method()` — it already
   dispatches all supported predicates.

### Pattern D: Parameterized method (buffer, simplify)

```python
def simplify(self, tolerance, preserve_topology=True):
    if self._owned is not None:
        from vibespatial.constructive.simplify import simplify_owned
        result_owned, selected = simplify_owned(
            self._owned, tolerance, preserve_topology=preserve_topology,
        )
        record_dispatch_event(
            surface="geopandas.array.simplify",
            operation="simplify",
            selected=selected,
        )
        if result_owned is not None:
            return GeometryArray.from_owned(result_owned, crs=self.crs)
    return GeometryArray(
        shapely.simplify(self._data, tolerance, preserve_topology=preserve_topology),
        crs=self.crs,
    )
```

### Checklist for GeometryArray

- [ ] Method added to `GeometryArray` class
- [ ] Routes through `self._owned` when available
- [ ] Falls back to `shapely.*()` when `_owned` is None
- [ ] Lazy import of implementation module (avoid circular imports)
- [ ] `record_dispatch_event()` called for observability
- [ ] Returns correct type (numpy array for metrics, GeometryArray for geometry)

---

## Step 3: Dispatch Wrapper (the `*_owned()` function)

This is the bridge between GeometryArray and the kernel. Lives in the
implementation module (e.g., `constructive/measurement.py`).

### Pattern: Metric operation

```python
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.precision import (
    KernelClass, PrecisionMode, select_precision_plan,
)
from vibespatial.runtime._runtime import ExecutionMode

def area_owned(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> np.ndarray:
    row_count = owned.row_count

    selection = plan_dispatch_selection(
        kernel_name="geometry_area",
        kernel_class=KernelClass.METRIC,
        row_count=row_count,
        requested_mode=dispatch_mode,
    )

    if selection.selected is ExecutionMode.GPU:
        precision_plan = select_precision_plan(
            runtime_selection=selection,
            kernel_class=KernelClass.METRIC,
            requested=precision,
        )
        try:
            return _area_gpu(owned, precision_plan=precision_plan)
        except Exception:
            pass

    return _area_cpu(owned)
```

### Pattern: Constructive operation (returns geometry)

```python
def simplify_owned(
    owned: OwnedGeometryArray,
    tolerance: float,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> tuple[OwnedGeometryArray | None, ExecutionMode]:
    selection = plan_dispatch_selection(
        kernel_name="geometry_simplify",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=owned.row_count,
        requested_mode=dispatch_mode,
    )

    if selection.selected is ExecutionMode.GPU:
        precision_plan = select_precision_plan(
            runtime_selection=selection,
            kernel_class=KernelClass.CONSTRUCTIVE,
            requested=precision,
        )
        try:
            result = _simplify_gpu(owned, tolerance, precision_plan)
            return result, ExecutionMode.GPU
        except Exception:
            pass

    return _simplify_cpu(owned, tolerance), ExecutionMode.CPU
```

### Checklist for dispatch wrapper

- [ ] Accepts `dispatch_mode` and `precision` parameters
- [ ] Calls `plan_dispatch_selection()` with correct `kernel_name` and `kernel_class`
- [ ] Calls `select_precision_plan()` before GPU path
- [ ] GPU path in try/except with CPU fallback
- [ ] CPU fallback uses Shapely (via OwnedGeometryArray → Shapely conversion)
- [ ] Null rows handled (result[~owned.validity] = NaN or None)

---

## Step 4: CPU Fallback Implementation

Every GPU operation MUST have a CPU fallback. Pattern:

```python
@register_kernel_variant(
    "geometry_simplify",
    "cpu",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=("polygon", "linestring", "multipolygon", "multilinestring"),
    supports_mixed=True,
    tags=("shapely",),
)
def _simplify_cpu(owned: OwnedGeometryArray, tolerance: float) -> OwnedGeometryArray:
    shapely_geoms = owned.to_shapely()
    results = shapely.simplify(shapely_geoms, tolerance)
    return OwnedGeometryArray.from_shapely(results)
```

- [ ] Registered with `@register_kernel_variant` (variant="cpu")
- [ ] Uses `owned.to_shapely()` for input conversion
- [ ] Uses `OwnedGeometryArray.from_shapely()` for output conversion
- [ ] Handles null/empty propagation

---

## Step 5: Public API Surface

### GeoSeries method (GeoPandas compatibility)

**File:** `src/vibespatial/api/geo_base.py`

Most methods are already wired via delegation helpers. If adding a new one:

```python
# For properties:
@property
def my_property(self):
    return _delegate_property("my_property", self)

# For unary methods:
def my_method(self, param):
    return _delegate_geo_method("my_method", self, param=param)

# For binary operations:
def my_binary(self, other, align=None):
    return _binary_op("my_binary", self, other, align)
```

- [ ] Method signature matches GeoPandas API
- [ ] Docstring matches GeoPandas (or links to it)
- [ ] `align` parameter present for binary ops (GeoPandas contract)

### DeviceGeometryArray surface

**File:** `src/vibespatial/geometry/device_array.py`

If the operation should be callable on device-resident arrays without
materializing to Shapely, add it to `DeviceGeometryArray`:

```python
@property
def area(self):
    from vibespatial.constructive.measurement import area_owned
    return area_owned(self._owned)
```

- [ ] Method added to DGA if applicable
- [ ] Routes directly to `*_owned()` (never materializes Shapely)
- [ ] Same signature as GeometryArray method

---

## Step 6: OwnedGeometryArray Coercion

When accepting external geometry inputs (e.g., binary operations where
the right side comes from the user), use the coercion utilities:

```python
from vibespatial.geometry.owned import coerce_geometry_array

# Coerce any input (Shapely list, GeoSeries, numpy array) to OwnedGeometryArray
owned = coerce_geometry_array(
    input_data,
    arg_name="right",
    expected_families=(GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON),
)
```

**Coercion handles:**
- `list[shapely.Geometry]` → OwnedGeometryArray
- `np.ndarray` of Shapely objects → OwnedGeometryArray
- `GeoSeries` → extract `.values._owned` or coerce `.values._data`
- Already an `OwnedGeometryArray` → pass through

- [ ] External inputs coerced via `coerce_geometry_array()`
- [ ] `expected_families` specified to catch type mismatches early
- [ ] `arg_name` set for clear error messages

---

## Step 7: Dispatch Event Recording

Record what happened for observability:

```python
from vibespatial.runtime.dispatch import record_dispatch_event

record_dispatch_event(
    surface="geopandas.array.simplify",
    operation="simplify",
    requested=dispatch_mode,
    selected=selection.selected,
    implementation="simplify_gpu_kernel",
    reason=selection.reason,
)
```

- [ ] Event recorded for both GPU and CPU paths
- [ ] Surface matches GeoPandas API name
- [ ] Implementation identifies which variant ran

---

## Quick Reference: Files to Touch

| What | File |
|------|------|
| GeometryArray method | `src/vibespatial/api/geometry_array.py` |
| Public API (GeoSeries) | `src/vibespatial/api/geo_base.py` |
| DeviceGeometryArray | `src/vibespatial/geometry/device_array.py` |
| Dispatch wrapper | `src/vibespatial/{module}/{operation}.py` |
| Binary predicate routing | `src/vibespatial/predicates/binary.py` |
| Predicate support check | `src/vibespatial/predicates/support.py` |
| Dispatch planning | `src/vibespatial/runtime/adaptive.py` |
| Precision planning | `src/vibespatial/runtime/precision.py` |
| Dispatch events | `src/vibespatial/runtime/dispatch.py` |
| Kernel variant registry | `src/vibespatial/runtime/kernel_registry.py` |
| OwnedGeometryArray | `src/vibespatial/geometry/owned.py` |
| Coercion utilities | `src/vibespatial/geometry/owned.py` |

---

## Common Mistakes

1. **Forgetting the Shapely fallback** — Every path through GeometryArray
   must work when `_owned is None`. The `else: shapely.*()` branch is not
   optional.

2. **Materializing to host in GPU path** — If your dispatch wrapper calls
   `owned.to_shapely()` in the GPU branch, you have a bug. That's a D->H
   transfer. Use device buffers directly.

3. **Not recording dispatch events** — Silent fallbacks are the #1 source
   of performance regressions. Always record what ran and why.

4. **Wrong return type** — Properties return numpy arrays. Geometry methods
   return `GeometryArray`. Binary predicates return numpy bool arrays.
   Getting this wrong breaks the GeoPandas contract.

5. **Missing null propagation** — Null rows (`~owned.validity`) must
   produce NaN (metrics), None (predicates), or null geometry (constructive).
   Check this explicitly.

6. **Circular imports** — Use lazy imports (`from X import Y` inside the
   method body) for implementation modules. The GeometryArray module is
   imported early; implementation modules import heavy CUDA dependencies.
