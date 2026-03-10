---
id: ADR-0036
status: accepted
date: 2026-03-16
deciders:
  - claude-opus
  - vibeSpatial maintainers
tags:
  - architecture
  - dataframe
  - attributes
  - cudf
  - performance
---

# Index-Array Boundary Attribute Model

## Context

vibeSpatial accelerates geometry operations on GPU while attribute (non-geometry)
columns remain on the host as a pandas DataFrame. Every spatial operation — sjoin,
overlay, dissolve, clip — follows this pattern today: run geometry kernels on
device, transfer result index arrays to host, assemble attribute columns with
pandas.

As the library matures, the question arises: should attribute columns move to GPU
as well? Options considered were:

1. **cuDF DataFrame as the substrate.** Replace pandas with cuDF for all attribute
   storage. Geometry registers as a cuDF extension column. Single device-resident
   container.
2. **pylibcudf columns for hot-path attribute ops.** Keep geometry separate but
   accelerate merge/groupby/reindex on device using pylibcudf primitives directly.
3. **Encode geometry as GeoArrow ListColumns in cuDF.** Store everything — geometry
   and attributes — as cuDF columns using GeoArrow nested list encoding.
4. **Keep attributes on host.** Spatial kernels produce index arrays. Attribute
   assembly stays pandas.

Investigation revealed that:

- cuDF does not support custom extension types. There is no registration mechanism
  for `OwnedGeometryArray` as a cuDF column dtype. Options 1 and 3 require either
  cuSpatial-style workarounds (two containers pretending to be one) or forced
  conversion between OwnedGeometryArray's mixed-family compute layout and
  GeoArrow's nested list storage layout on every kernel invocation.
- cuSpatial tried option 1 and ended up with a GeoDataFrame that was neither a
  pandas DataFrame nor a cuDF DataFrame, with operations that sometimes worked and
  sometimes did not. Users were confused about what they were holding.
- The attribute operations that spatial pipelines actually need (gather-by-index,
  merge-by-key, groupby-agg, concat, column add/drop) are fast on pandas at the
  row counts where spatial operations are the bottleneck. At 1M rows with 10-20
  attribute columns, pandas merge on host takes single-digit milliseconds — noise
  relative to GPU geometry kernel time.
- The index arrays that cross the device-host boundary are small. 1M int32 index
  pairs = ~8MB. PCIe transfer is microseconds. The synchronization point, not the
  data volume, is the cost — and that cost is inherent regardless of where
  attributes live.
- For combined spatial + attribute queries (e.g., "select intersecting geometry
  where name = X"), the optimal execution plan filters attributes on host first to
  reduce the geometry set sent to GPU. Keeping attributes on host enables this
  naturally. Moving attributes to GPU would require either uploading string columns
  to device (expensive, wasteful) or maintaining redundant copies.
- GPU VRAM is a finite resource better spent on geometry buffers, spatial indices,
  and kernel working memory than on attribute columns that pandas handles
  adequately.
- Keeping attributes as a real pandas DataFrame means users get the full pandas
  API with no compatibility gaps, surprises, or behavioral differences.

## Decision

Adopt the **index-array boundary model**: geometry is device-resident, attributes
are host-resident, and the interface between the two domains is index arrays.

### Execution contract

Spatial kernels accept `OwnedGeometryArray` inputs and produce one of:

- index arrays (row indices that matched a predicate or join condition)
- geometry buffers (new `OwnedGeometryArray` from constructive operations)

Spatial kernels never read, write, or reference attribute columns. They have no
knowledge of the DataFrame that wraps them.

Attribute assembly — `df.iloc[indices]`, `df.merge(...)`, `df.groupby().agg()` —
is always pandas, always on host. The GeoDataFrame coordinates both sides:

```
GPU domain                    Host domain
-----                         -----
OwnedGeometryArray            pandas DataFrame
  |                              |
spatial kernel                   (waiting, or attribute pre-filter)
  |
index arrays --(transfer)-->  index arrays
                                 |
                              df.iloc / df.merge / df.groupby
                                 |
OwnedGeometryArray.take(idx)  pandas DataFrame (result)
  |                              |
  +--------- GeoDataFrame ------+
```

### No cuDF dependency

vibeSpatial does not depend on cuDF for attribute operations. pylibcudf remains an
optional dependency for columnar I/O (Parquet, Arrow) only.

### cudf.pandas compatibility

The index-array boundary model is compatible with `cudf.pandas` as an external
accelerator. If a user activates `cudf.pandas`, pandas attribute operations are
transparently proxied to cuDF on GPU. This is additive — vibeSpatial does not need
to know or care whether the underlying DataFrame is pandas or a cudf.pandas proxy.

The geometry column (`DeviceGeometryArray`, a pandas `ExtensionArray` subclass)
falls back to the pandas code path under cudf.pandas because cuDF cannot recognize
the custom dtype. This is correct behavior: geometry operations are already
GPU-accelerated through vibeSpatial's own kernels, not through cuDF.

Compatibility with cudf.pandas should be tested but is not a design constraint.
Edge cases around proxy wrapping of `DeviceGeometryArray` during mixed-column
operations (e.g., `.iloc` on a frame with both cuDF-proxied attribute columns and
an ExtensionArray geometry column) may require targeted fixes.

### Arrow tables in I/O pipelines

I/O paths should keep attribute data as Arrow tables as long as possible and defer
`.to_pandas()` conversion to the point where a GeoDataFrame is actually
constructed. For pipelines that chain `read -> spatial_op -> write` without user
inspection of attributes, the Arrow-to-pandas conversion may be avoidable entirely.

This is an optimization opportunity, not a hard requirement. The current
`.to_pandas()` call in `io_file.py` is correct; it is simply earlier than
necessary.

### When to reconsider

If end-to-end profiling (the mandatory profile gate in AGENTS.md) reveals that
host-side attribute operations consume >15% of wall time at target scale (1M+
rows), the decision should be revisited. The likely remediation at that point is
targeted pylibcudf acceleration of the specific bottleneck operation (e.g.,
merge-by-key for sjoin), not wholesale migration to cuDF.

## Consequences

- The GeoDataFrame remains a real pandas DataFrame with a GPU-accelerated geometry
  column. Users get the full pandas API for attribute work with no compatibility
  gaps.
- GPU VRAM is reserved for geometry, spatial indices, and kernel working memory.
  Attribute columns do not consume device resources.
- Combined spatial + attribute queries naturally optimize by filtering attributes
  on host first, reducing the geometry set sent to GPU.
- There is no cuDF version coupling for attribute operations. The only RAPIDS
  dependency is pylibcudf for I/O, which the project already carries.
- The index-array boundary creates a clean, testable seam: spatial kernel tests
  validate index output, attribute assembly tests validate pandas operations, and
  integration tests validate the combination.
- cudf.pandas users get transparent attribute acceleration without vibeSpatial
  changes, though edge cases with `DeviceGeometryArray` may need fixes.

## Alternatives Considered

- **cuDF DataFrame with geometry extension column.** Not feasible: cuDF has no
  custom extension type registration. Would require cuSpatial-style dual-container
  workaround, producing a DataFrame that is neither pandas nor cuDF.
- **pylibcudf column bag for attributes.** Feasible but creates a mini-DataFrame
  library. The five operations needed today (gather, merge, groupby, concat,
  column ops) would grow over time as users expect more DataFrame functionality.
  Maintenance burden is not justified by the marginal performance gain at current
  target scales.
- **GeoArrow nested list encoding in cuDF.** Geometry participates in cuDF frame
  operations, but requires conversion between OwnedGeometryArray's mixed-family
  compute layout and GeoArrow's storage layout on every spatial kernel invocation.
  The conversion cost defeats the purpose.
- **Full GPU residency for both geometry and attributes.** Wastes VRAM on data
  that pandas handles in microseconds. Forces GPU-side string operations that are
  slower than host for typical GIS attribute workloads (place names, codes,
  categorical fields). Prevents the natural "filter attributes first, then spatial"
  optimization.
