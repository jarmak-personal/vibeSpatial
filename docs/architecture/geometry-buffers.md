# Geometry Buffer Schema

<!-- DOC_HEADER:START
Scope: Canonical owned geometry buffer schema for points, lines, polygons, and multiparts.
Read If: You are designing adapters, kernels, or memory layout for owned geometry arrays.
STOP IF: Your task already has a settled buffer schema and only needs implementation detail.
Source Of Truth: Phase-2 owned geometry buffer layout contract.
Body Budget: 158/260 lines
Document: docs/architecture/geometry-buffers.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-4 | Preamble |
| 5-9 | Intent |
| 10-18 | Request Signals |
| 19-28 | Open First |
| 29-33 | Verify |
| 34-39 | Risks |
| 40-47 | Canonical Rule |
| 48-59 | Shared Buffers |
| 60-96 | Family Schemas |
| 97-102 | Mixed-Geometry Integration |
| 103-117 | Adapter Surface |
| 118-124 | Offset Rules |
| 125-151 | Indexed-View Pattern |
| 152-158 | Execution Boundaries |
DOC_HEADER:END -->

Use separated fp64 coordinate buffers plus hierarchical offsets as the owned geometry core.

## Intent

Define the concrete buffer schema for the six primary geometry families before
adapter and kernel work begins.

## Request Signals

- geometry buffer
- offsets
- geoarrow layout
- owned array
- coordinate buffers
- multipart schema

## Open First

- docs/architecture/geometry-buffers.md
- docs/architecture/mixed-geometries.md
- docs/architecture/precision.md
- src/vibespatial/geometry/buffers.py
- src/vibespatial/geometry/owned.py
- src/vibespatial/kernels/core/geometry_analysis.py
- docs/decisions/0008-owned-geometry-buffer-schema.md

## Verify

- `uv run pytest tests/test_geometry_buffers.py`
- `uv run python scripts/check_docs.py --check`

## Risks

- Choosing an eager object-like layout now would force expensive rewrites once kernels want contiguous payloads.
- Overfitting the schema to one geometry family would make mixed arrays and multipart kernels awkward later.
- Mixing canonical storage concerns with execution-local staging would blur residency and precision boundaries.

## Canonical Rule

- Canonical owned storage uses separated `x` and `y` coordinate buffers.
- Canonical coordinate precision is `fp64`.
- Nulls use a validity bitmap.
- Empties use valid rows with zero-length spans.
- Multipart structure is represented with prefix-offset buffers, not nested Python objects.

## Shared Buffers

Every schema includes:

- `validity`: row-level bitmask
- `x`: contiguous fp64 x coordinates
- `y`: contiguous fp64 y coordinates
- optional derived `bounds` cache

`bounds` is not part of the authoritative payload. It is a cache that later
kernels may materialize or invalidate independently.

## Family Schemas

### Point

- `geometry_offsets`: row -> coordinate
- valid non-empty rows own exactly one coordinate pair
- empty rows own zero coordinate pairs

### LineString

- `geometry_offsets`: row -> coordinate
- payload slice: `x[start:end]`, `y[start:end]`

### Polygon

- `geometry_offsets`: row -> ring
- `ring_offsets`: ring -> coordinate
- payload hierarchy: row -> ring -> coordinate

### MultiPoint

- `geometry_offsets`: row -> coordinate
- same physical shape as `LineString`, different semantics

### MultiLineString

- `geometry_offsets`: row -> part
- `part_offsets`: part -> coordinate
- payload hierarchy: row -> line part -> coordinate

### MultiPolygon

- `geometry_offsets`: row -> polygon part
- `part_offsets`: polygon part -> ring
- `ring_offsets`: ring -> coordinate
- payload hierarchy: row -> polygon part -> ring -> coordinate

## Mixed-Geometry Integration

- The mixed-array contract from `o17.2.12` remains canonical.
- Mixed arrays should store a coarse family tag plus a family-relative row offset.
- Family payload buffers must stay reusable without copying coordinate payloads during sort-partition execution.

## Adapter Surface

The owned-array bootstrap surface currently supports:

- Shapely geometry sequences -> owned arrays
- WKB sequences -> owned arrays
- GeoArrow-style buffer views -> owned arrays
- owned arrays -> Shapely
- owned arrays -> WKB
- owned arrays -> GeoArrow-style buffer views

The current GeoArrow path is a typed buffer-view contract, not a full `pyarrow`
extension-array integration. That narrower surface is enough for CPU
validation, buffer inspection, and later IO work to target.

## Offset Rules

- Offsets are prefix arrays with length `N + 1`.
- Empty valid geometries use equal adjacent offsets.
- Nullness is never encoded by offset shape.
- `int32` is the default offset dtype for Phase 2; revisit only when measured scale requires wider offsets.

## Indexed-View Pattern

`OwnedGeometryArray` supports an indexed-view optimisation for operations
that expand few unique geometries to many output rows (e.g., sjoin).

- `_base`: the compact `OwnedGeometryArray` holding only unique rows.
- `_index_map`: int64 array mapping each logical row to a `_base` row.
- `is_indexed_view`: property returning `True` when `_base` and `_index_map`
  are set.
- `families` dict: shared by reference with `_base` (no coordinate copy).
- `_resolve()`: materialises the view into a flat contiguous array by
  physically gathering coordinates. Called automatically before kernel
  dispatch (`_ensure_device_state`) and host materialisation
  (`_ensure_host_state`). After resolution, `_base` and `_index_map` are
  set to `None`.

Threshold policy constants control when `take()` produces an indexed view
instead of a physical copy:

- `_INDEXED_VIEW_MIN_ROWS = 1000`: minimum output size to consider.
- `_INDEXED_VIEW_RATIO_THRESHOLD = 0.5`: unique/total ratio below which
  the indexed view is used.

Stacking indexed views is prevented: if `self` is already an indexed view,
`take()` composes the two index maps and builds a single view over the
original physical base.

## Execution Boundaries

- Centered fp32 work buffers from the precision policy are execution-local artifacts, not canonical storage.
- Permutation buffers for mixed execution are execution-local artifacts, not canonical storage.
- Residency attaches to the owned buffer object as a whole, not separately to every offset array.
- Buffer-boundary diagnostics belong to the owned array object and must survive
  transfers and explicit fallback decisions.
