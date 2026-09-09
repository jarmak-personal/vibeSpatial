"""Physical-work admission for bounded PBF relation batches.

Ordinary relations are partitioned by expanded coordinates and member graph
bytes. A single relation that exceeds the workspace budget uses CUDA managed
scratch: kernels still execute on the GPU and the CUDA driver can page its
working set into host backing memory. No geometry is evaluated on the host.
"""
from __future__ import annotations

from contextlib import contextmanager

import cupy as cp


@contextmanager
def relation_workspace(work_bytes, budget, *, kind="relation"):
    if work_bytes <= budget:
        yield
        return
    from vibespatial.runtime import ExecutionMode
    from vibespatial.runtime.dispatch import record_dispatch_event

    record_dispatch_event(
        surface="vibespatial.io.osm_pbf", operation=f"{kind}_workspace",
        implementation=f"cuda_managed_{kind}_workspace", selected=ExecutionMode.GPU,
        reason=f"The {kind} work unit exceeds the device workspace; CUDA pages its working set",
        detail=f"estimated_work_bytes={work_bytes}, device_workspace_bytes={budget}",
    )
    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    try:
        with cp.cuda.using_allocator(pool.malloc):
            yield
    finally:
        # Live result allocations retain their pool owner.
        cp.cuda.get_current_stream().synchronize()
        pool.free_all_blocks()


def concatenate_results(results):
    """Assemble the final native columns with an explicit paging admission.

    Input geometry buffers are retained through a native composition. libcudf
    writes final attribute columns once; when that transient copy exceeds the
    remaining device envelope, its allocation uses CUDA managed backing.
    """
    import pyarrow as pa
    import pylibcudf as plc
    import rmm

    from vibespatial.api._native_result_core import (
        GeometryNativeResult,
        NativeAttributeTable,
        NativeGeometryComposition,
        NativeTabularResult,
    )
    from vibespatial.cuda._runtime import pylibcudf_current_stream
    from vibespatial.io.osm_pbf_native import _available_bytes

    if len(results) == 1:
        return results[0]
    rows = sum(result.geometry.row_count for result in results)
    fields = {}
    chars = 0
    for result in results:
        for field in (result.attributes.schema_override if result.attributes.schema_override is not None
                      else result.attributes.arrow_table.schema):
            fields.setdefault(field.name, field)
        table = result.attributes.device_table
        if table is not None:
            for column in table.columns():
                if column.type().id() == plc.types.TypeId.STRING:
                    chars += column.data().size if column.data() is not None else 0
    # Includes final offset/mask storage and temporary typed null columns.
    estimate = chars + 10*rows*len(fields) + 8*rows
    budget = _available_bytes()
    spill = estimate > budget
    resource = rmm.mr.ManagedMemoryResource() if spill else None
    if spill:
        from vibespatial.runtime import ExecutionMode
        from vibespatial.runtime.dispatch import record_dispatch_event
        record_dispatch_event(
            surface="vibespatial.io.osm_pbf", operation="attribute_concatenate",
            implementation="cuda_managed_attribute_concatenate", selected=ExecutionMode.GPU,
            reason="Final column assembly uses CUDA paging to avoid a second resident copy of the attributes",
            detail=f"estimated_transient_bytes={estimate}, available_device_bytes={budget}",
        )
    if fields:
        aligned = []
        for result in results:
            source = result.attributes.device_table
            if source is None:
                if result.geometry.row_count:
                    raise TypeError("Native OSM concat requires device attributes")
                source = plc.Table.from_arrow(result.attributes.to_arrow(index=False))
            columns = dict(zip(result.attributes.columns, source.columns(), strict=True))
            out = []
            for name, field in fields.items():
                column = columns.get(name)
                if column is None:
                    scalar = plc.Scalar.from_arrow(pa.scalar(None, type=field.type))
                    column = plc.Column.from_scalar(scalar, result.geometry.row_count,
                                                     stream=pylibcudf_current_stream(), mr=resource)
                out.append(column)
            aligned.append(plc.Table(out))
        table = plc.concatenate.concatenate(aligned, stream=pylibcudf_current_stream(*aligned), mr=resource)
        attrs = NativeAttributeTable(device_table=table, column_override=tuple(fields), schema_override=pa.schema(fields.values()))
    else:
        attrs = NativeAttributeTable(arrow_table=pa.table({"_": pa.nulls(rows)}).select([]))
    with relation_workspace(estimate, budget):
        geometry = GeometryNativeResult.from_composition(
            NativeGeometryComposition.concat([result.geometry for result in results], crs="EPSG:4326"), crs="EPSG:4326",
        )
    return NativeTabularResult(attributes=attrs, geometry=geometry, geometry_name="geometry", column_order=(*attrs.columns, "geometry"))
