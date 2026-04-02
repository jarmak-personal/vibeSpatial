---
name: gis-domain
description: "Use this skill to verify GIS correctness of spatial operations, check predicate semantics, validate geometry conventions (winding, closure, DE-9IM), understand edge cases (degenerate geometries, null/empty handling), or answer \"does this make sense?\" questions about spatial algorithms. This is the domain knowledge oracle for autonomous agents working on vibeSpatial. Trigger on: \"is this correct\", \"should this return\", \"what does X mean in GIS\", \"DE-9IM\", \"winding\", \"orientation\", \"edge case\", \"degenerate\", \"predicate semantics\", \"topology\", \"validity\"."
---

# GIS Domain Knowledge Oracle — vibeSpatial

Use this reference to verify whether a spatial operation produces the
correct result, to understand edge cases, or to validate design decisions.

Question: **$ARGUMENTS**

---

## 1. Binary Predicate Semantics (DE-9IM)

The 9 standard binary predicates are defined by the Dimensionally Extended
9-Intersection Model (DE-9IM). Each predicate tests a specific relationship
between the interior (I), boundary (B), and exterior (E) of two geometries.

### Quick Reference

| Predicate | Plain English | Key Rule |
|-----------|--------------|----------|
| `intersects(a, b)` | Any shared points | `NOT disjoint` |
| `disjoint(a, b)` | No shared points | Interior, boundary, exterior all separate |
| `within(a, b)` | a entirely inside b | Interior of a inside interior of b; boundary of a may touch boundary of b |
| `contains(a, b)` | b entirely inside a | `within(b, a)` |
| `touches(a, b)` | Boundaries touch, interiors don't | Shared points exist but only on boundaries |
| `crosses(a, b)` | Geometries cross | Interior intersection has lower dimension than max of inputs |
| `overlaps(a, b)` | Partial overlap, same dimension | Interior intersection has same dimension as both inputs |
| `covers(a, b)` | No point of b outside a | Like contains but boundary-on-boundary is OK |
| `covered_by(a, b)` | No point of a outside b | `covers(b, a)` |
| `contains_properly(a, b)` | b in interior of a | No boundary contact at all |

### Predicate Edge Cases

**Point on polygon boundary:**
- `within(point, polygon)` = **True** (boundary contact allowed)
- `contains(polygon, point)` = **True**
- `touches(point, polygon)` = **True** (boundary-only intersection)
- `contains_properly(polygon, point)` = **False** (point is on boundary, not interior)
- `intersects(point, polygon)` = **True**

**Point on polygon vertex:**
Same as point-on-boundary — vertices are part of the boundary.

**Line endpoint touching polygon boundary:**
- `touches(line, polygon)` = **True** (if only endpoint touches)
- `intersects(line, polygon)` = **True**
- `crosses(line, polygon)` = **False** (no interior penetration)

**Identical geometries:**
- `within(a, a)` = **True**
- `contains(a, a)` = **True**
- `overlaps(a, a)` = **False** for points/lines (same dim but identical = not "partial")
- `equals(a, a)` = **True**

**Empty geometries:**
- `intersects(empty, anything)` = **False**
- `disjoint(empty, anything)` = **True**
- All other predicates with empty input = **False**

**Null geometries:**
- Any predicate with null input = **None** (propagate null)

### Predicate Symmetry

| Symmetric | Asymmetric |
|-----------|------------|
| intersects, disjoint, touches, crosses, overlaps | within/contains, covers/covered_by, contains_properly |

For asymmetric predicates: `within(a, b) == contains(b, a)`.

---

## 2. Geometry Validity and Conventions

### Ring Orientation (Winding)

- **Exterior rings**: Counter-Clockwise (CCW) = positive signed area
- **Interior rings (holes)**: Clockwise (CW) = negative signed area
- **Shoelace formula**: `signed_area = 0.5 * sum(x[i]*y[i+1] - x[i+1]*y[i])`

A positive shoelace result means CCW. This is the OGC/ISO convention used
by Shapely, PostGIS, and vibeSpatial.

### Ring Closure

Rings MUST be closed: first coordinate == last coordinate. A triangle
requires 4 coordinate pairs: `[(0,0), (1,0), (0,1), (0,0)]`.

Minimum valid ring: 4 coordinates (3 unique + closing).

### Polygon Validity Rules (OGC)

1. Exterior ring is CCW, holes are CW
2. All rings are closed
3. All rings are simple (no self-intersection)
4. Holes are contained within the exterior ring
5. No two rings cross (holes may touch exterior at a single point)
6. The interior is connected (no cuts that split it)

### Invalid Geometry Examples

| Invalid Geometry | Description | Result of `make_valid()` |
|-----------------|-------------|--------------------------|
| Bowtie polygon | Self-intersecting exterior ring | Split into two triangles (MultiPolygon) |
| Touching hole | Hole boundary touches exterior at >1 point | Split into valid parts |
| Inverted hole | Hole has CCW winding (same as exterior) | Reverse hole orientation |
| Unclosed ring | First != last coordinate | Close ring by appending first point |
| Spike/cut | Zero-area protrusion or cut | Remove degenerate parts |

### LineString Validity

- Minimum 2 points
- `is_closed`: first point == last point (forms a LinearRing)
- Self-intersection is allowed (unlike rings)

### Point Validity

- Always valid unless null or empty
- Empty point: valid geometry with no coordinates

---

## 3. Measurement Semantics

### Area

- **Polygon**: Absolute value of shoelace area (exterior - holes)
- **MultiPolygon**: Sum of individual polygon areas
- **Point/LineString**: 0.0 (dimensionally zero)
- **Empty geometry**: 0.0
- **Null geometry**: NaN (propagate)
- **Units**: Square units of the CRS (not geodetic!)

### Length

- **LineString**: Sum of Euclidean segment lengths
- **MultiLineString**: Sum of part lengths
- **Polygon**: Perimeter (exterior ring + hole rings)
- **Point**: 0.0
- **Empty**: 0.0
- **Null**: NaN
- **Units**: Linear units of the CRS

### Distance

- **Euclidean**: Minimum distance between closest points
- **Same geometry**: 0.0
- **Disjoint**: Positive value
- **Intersecting**: 0.0
- **Null input**: NaN
- **Empty input**: NaN (or infinity depending on convention)
- **dwithin(a, b, d)**: `distance(a, b) <= d`

### Centroid

- **Polygon**: Area-weighted centroid (not center of bounding box)
- **LineString**: Length-weighted midpoint
- **Point**: The point itself
- **MultiPolygon**: Area-weighted across parts
- **Empty**: Empty point
- **Null**: Null
- Centroid may be OUTSIDE the geometry (e.g., C-shaped polygon)

### Bounds

- Returns `(minx, miny, maxx, maxy)`
- **Empty geometry**: `(NaN, NaN, NaN, NaN)`
- **Null geometry**: `(NaN, NaN, NaN, NaN)`
- **Point**: `(x, y, x, y)` (zero-extent box)

---

## 4. Constructive Operation Semantics

### Intersection / Union / Difference / Symmetric Difference

These are set operations on the point sets of two geometries:

| Operation | Result | Points in result |
|-----------|--------|-----------------|
| `intersection(a, b)` | A AND B | Points in both a and b |
| `union(a, b)` | A OR B | Points in a or b (or both) |
| `difference(a, b)` | A AND NOT B | Points in a but not b |
| `symmetric_difference(a, b)` | A XOR B | Points in a or b but not both |

**Return type depends on input and result dimension:**
- Polygon intersect Polygon → Polygon, MultiPolygon, LineString, Point, or GeometryCollection
- Line intersect Line → Point, MultiPoint, LineString, or GeometryCollection
- Point intersect anything → Point or empty

**Edge cases:**
- `intersection(a, a)` = `a` (identity)
- `union(a, a)` = `a`
- `difference(a, a)` = empty
- `intersection(a, disjoint_b)` = empty
- Result may be mixed type (GeometryCollection) for non-trivial overlaps

### Buffer

- `buffer(geom, distance)`: All points within `distance` of `geom`
- `buffer(geom, 0)`: For polygons, equivalent to `make_valid` (repair trick)
- `buffer(point, r)`: Circle approximation (polygon with `quad_segs` segments per quarter)
- `buffer(line, r)`: Corridor around the line
- `buffer(polygon, -r)`: Erode inward (may produce empty if too thin)
- Negative buffer on point/line = empty
- **cap_style**: round (1), flat (2), square (3)
- **join_style**: round (1), mitre (2), bevel (3)

### Clip (clip_by_rect)

- Fast path for axis-aligned rectangle clipping
- Sutherland-Hodgman algorithm
- Output preserves ring orientation
- Geometry fully inside clip box: returned unchanged
- Geometry fully outside clip box: empty geometry
- Partial overlap: clipped polygon (possibly multi-part)

### Dissolve

- Group-by aggregation of geometries
- `dissolve(by=column)`: Union all geometries sharing the same group key
- Interior boundaries between adjacent polygons are removed
- Produces MultiPolygon when unioned parts are disjoint

### Make Valid

- Repairs invalid geometries following OGC rules
- Self-intersecting rings → split at intersection points → reassemble
- Inverted holes → reverse winding
- Unclosed rings → close by appending first coordinate
- Spike removal → collapse degenerate parts
- Output is always valid (may change geometry type)

---

## 5. Null and Empty Handling

This is the #1 source of subtle bugs. Null and empty are DIFFERENT.

| | Null | Empty |
|---|------|-------|
| **Meaning** | "No value" / missing | "Valid geometry with no points" |
| **In pandas** | None / NaN | `shapely.Point()` (empty Point) |
| **Predicate result** | None (propagate) | False (except disjoint=True) |
| **Metric result** | NaN | 0.0 |
| **Bounds result** | (NaN, NaN, NaN, NaN) | (NaN, NaN, NaN, NaN) |
| **In OwnedGeometryArray** | `validity[i] = False` | `validity[i] = True`, zero-span offsets |
| **Buffer** | None | Empty |
| **Union with valid** | None | The valid geometry |
| **Intersection with valid** | None | Empty |

### Propagation Rules

- **Metric ops (area, length, distance)**: Null → NaN, Empty → 0.0
- **Predicates**: Null → None, Empty → False (except disjoint)
- **Constructive**: Null → None, Empty → Empty (or context-dependent)
- **Aggregation (dissolve)**: Nulls skipped, empties included

---

## 6. Coordinate System Considerations

### vibeSpatial Assumptions

- All kernel computations are **planar** (Euclidean)
- CRS is stored at GeoSeries/GeoDataFrame level, not per-geometry
- Kernels are CRS-agnostic — they operate on raw (x, y) coordinates
- Distance, area, length are in **CRS units** (meters for UTM, degrees for WGS84)

### When CRS Matters

- **Area in degrees** (WGS84/EPSG:4326): Meaningless for accurate
  measurement. Must project to equal-area CRS first. vibeSpatial does
  NOT do this automatically.
- **Distance in degrees**: 1 degree != constant meters. For accurate
  distance, project to local UTM or use geodetic formulas. vibeSpatial
  uses Euclidean distance only.
- **Antimeridian (180/-180 longitude)**: Geometries crossing the
  antimeridian may have incorrectly computed bounds, area, and predicates.
  Split at antimeridian first.
- **Polar regions**: Euclidean assumptions break down near poles.

### Coordinate Precision

- **fp64 storage**: ~15 significant decimal digits
- **fp64 geographic**: Sub-millimeter precision at any location on Earth
- **fp32 geographic**: ~1 meter precision (insufficient for parcel-level work)
- **fp32 with centering**: Sub-meter precision when coordinates are
  centered around the compute region

---

## 7. Degeneracy Corpus

vibeSpatial maintains a deterministic corpus of degenerate geometries
for testing edge cases. When writing or verifying operations, test
against these:

**File:** `src/vibespatial/testing/degeneracy.py`

| Case | What It Tests |
|------|--------------|
| `shared_vertex_lines` | Two lines meeting at endpoint → `touches` = True |
| `collinear_overlap_lines` | Lines on same line with overlap → `overlaps` = True |
| `donut_window_polygon` | Polygon with hole, clipped by window crossing hole |
| `duplicate_vertex_polygon` | Repeated adjacent vertices → stable under overlay |
| `bowtie_invalid_polygon` | Self-intersecting → requires `make_valid` first |
| `touching_hole_invalid_polygon` | Hole touching shell → invalid |
| `null_and_empty_polygon_rows` | Null/empty rows disappear cleanly |

---

## 8. Oracle Testing Contract

When verifying correctness, compare against Shapely (the reference
implementation):

```python
# Geometry equality
shapely_result.equals_exact(gpu_result, tolerance=1e-9)

# Float equality
math.isclose(gpu_value, shapely_value, rel_tol=1e-7, abs_tol=1e-9)

# Null handling
# Both null → equal
# One null, one not → NOT equal
# Both empty, same type → equal
```

**Standard tolerances:**
- `rtol=1e-7` (relative) for fp64 compute
- `atol=1e-9` (absolute) for fp64 compute
- For fp32 compute: `rtol=1e-3` to `1e-4` depending on kernel class

---

## 9. Robustness Guarantees by Kernel Class

| Class | Guarantee | What It Means |
|-------|-----------|---------------|
| COARSE | Bounded error | Result is approximate but within known error bounds |
| METRIC | Bounded error | Accumulation uses Kahan summation to control drift |
| PREDICATE | Exact | Correct boolean result even for degenerate inputs |
| CONSTRUCTIVE | Exact | Output topology is valid; no missed intersections |

**Exact predicates** use adaptive precision:
1. Fast fp64 test with error bound check
2. If `|orientation| <= error_bound`: mark ambiguous
3. Re-evaluate ambiguous cases with higher precision (expansion arithmetic)
4. Error bound formula: `error = (3.0 + 16.0 * eps) * eps * magnitude`

---

## 10. "Does This Make Sense?" Quick Checks

Use these rules to sanity-check operation results:

- **area(polygon) < 0**: Wrong. Area is absolute value. If your kernel
  returns negative area, you have a winding bug.
- **centroid outside polygon**: Valid! C-shaped, U-shaped, and ring-shaped
  polygons can have exterior centroids.
- **buffer(polygon, 0) != polygon**: Valid if polygon was invalid. buffer(0)
  is a repair operation.
- **intersection(a, b) larger than a or b**: Wrong. Intersection is always
  <= both inputs.
- **union(a, b) smaller than a or b**: Wrong. Union is always >= both inputs.
- **within(a, b) AND within(b, a)**: Means a equals b (geometrically).
- **distance(a, b) = 0 but NOT intersects(a, b)**: Wrong. Zero distance
  implies intersection (or touching).
- **touches(a, b) AND intersects(a, b)**: Always true. Touches implies
  intersects.
- **touches(a, b) AND overlaps(a, b)**: Wrong. These are mutually exclusive.
- **contains(a, b) AND disjoint(a, b)**: Wrong. Mutually exclusive.
- **is_valid(make_valid(g))**: Must always be True.
- **area(union(a,b)) <= area(a) + area(b)**: Always true (equality when
  disjoint).
