from __future__ import annotations

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    cp = None

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    get_cuda_runtime,
    make_kernel_cache_key,
)
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.geometry_analysis_host import (
    compute_geometry_bounds_cpu_scalar as _compute_geometry_bounds_cpu_scalar_host,
)
from vibespatial.geometry.geometry_analysis_host import (
    compute_geometry_bounds_cpu_vectorized as _compute_geometry_bounds_cpu_vectorized_host,
)
from vibespatial.geometry.geometry_analysis_host import (
    compute_morton_keys_cpu,
    compute_offset_spans_cpu,
    compute_total_bounds_from_bounds,
)
from vibespatial.geometry.owned import (
    DeviceFamilyGeometryBuffer,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
)
from vibespatial.kernels.core.geometry_analysis_source import (
    _BOUNDS_COOPERATIVE_KERNEL_NAMES,
    _BOUNDS_KERNEL_NAMES,
    _COOPERATIVE_BOUNDS_THRESHOLD,
    _format_bounds_kernel_source,
    _format_cooperative_bounds_kernel_source,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_kernel_dispatch
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.fallbacks import record_fallback_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import (
    KernelClass,
    PrecisionMode,
    normalize_precision_mode,
    select_precision_plan,
)
from vibespatial.runtime.residency import Residency, TransferTrigger

_BOUNDS_KERNEL_SOURCE = _format_bounds_kernel_source("double")
_BOUNDS_COOPERATIVE_KERNEL_SOURCE = _format_cooperative_bounds_kernel_source("double")


def _compute_geometry_bounds_cpu_scalar(geometry_array: OwnedGeometryArray):
    return _compute_geometry_bounds_cpu_scalar_host(geometry_array)


@register_kernel_variant(
    "compute_geometry_bounds",
    "cpu-vectorized",
    kernel_class=KernelClass.COARSE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=tuple(family.value for family in GeometryFamily),
    supports_mixed=True,
    tags=("vectorized",),
)
def _compute_geometry_bounds_cpu_vectorized(geometry_array: OwnedGeometryArray):
    return _compute_geometry_bounds_cpu_vectorized_host(geometry_array)



from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

if cp is not None:
    request_nvrtc_warmup([
        ("geometry-bounds", _BOUNDS_KERNEL_SOURCE, _BOUNDS_KERNEL_NAMES),
        ("geometry-bounds-cooperative", _BOUNDS_COOPERATIVE_KERNEL_SOURCE, _BOUNDS_COOPERATIVE_KERNEL_NAMES),
    ])


def _bounds_kernels(compute_type: str = "double"):
    source = _format_bounds_kernel_source(compute_type)
    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key(f"geometry-bounds-{compute_type}", source)
    return runtime.compile_kernels(
        cache_key=cache_key,
        source=source,
        kernel_names=_BOUNDS_KERNEL_NAMES,
    )


def _bounds_cooperative_kernels(compute_type: str = "double"):
    source = _format_cooperative_bounds_kernel_source(compute_type)
    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key(f"geometry-bounds-cooperative-{compute_type}", source)
    return runtime.compile_kernels(
        cache_key=cache_key,
        source=source,
        kernel_names=_BOUNDS_COOPERATIVE_KERNEL_NAMES,
    )


def _family_kernel_name(family: GeometryFamily) -> str:
    if family in {GeometryFamily.POINT, GeometryFamily.LINESTRING, GeometryFamily.MULTIPOINT}:
        return "bounds_simple"
    if family is GeometryFamily.POLYGON:
        return "bounds_polygon"
    if family is GeometryFamily.MULTILINESTRING:
        return "bounds_multilinestring"
    if family is GeometryFamily.MULTIPOLYGON:
        return "bounds_multipolygon"
    raise NotImplementedError(f"unsupported family: {family.value}")


def _launch_family_bounds_kernel(
    family: GeometryFamily,
    device_buffer: DeviceFamilyGeometryBuffer,
    *,
    row_count: int,
    compute_type: str = "double",
) -> None:
    if row_count == 0:
        return
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    kernel = _bounds_kernels(compute_type)[_family_kernel_name(family)]
    if family in {GeometryFamily.POINT, GeometryFamily.LINESTRING, GeometryFamily.MULTIPOINT}:
        params = (
            (
                ptr(device_buffer.x),
                ptr(device_buffer.y),
                ptr(device_buffer.geometry_offsets),
                ptr(device_buffer.empty_mask),
                ptr(device_buffer.bounds),
                row_count,
            ),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        )
    elif family is GeometryFamily.POLYGON:
        params = (
            (
                ptr(device_buffer.x),
                ptr(device_buffer.y),
                ptr(device_buffer.geometry_offsets),
                ptr(device_buffer.ring_offsets),
                ptr(device_buffer.empty_mask),
                ptr(device_buffer.bounds),
                row_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
    elif family is GeometryFamily.MULTILINESTRING:
        params = (
            (
                ptr(device_buffer.x),
                ptr(device_buffer.y),
                ptr(device_buffer.geometry_offsets),
                ptr(device_buffer.part_offsets),
                ptr(device_buffer.empty_mask),
                ptr(device_buffer.bounds),
                row_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
    else:
        params = (
            (
                ptr(device_buffer.x),
                ptr(device_buffer.y),
                ptr(device_buffer.geometry_offsets),
                ptr(device_buffer.part_offsets),
                ptr(device_buffer.ring_offsets),
                ptr(device_buffer.empty_mask),
                ptr(device_buffer.bounds),
                row_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
    grid, block = runtime.launch_config(kernel, row_count)
    runtime.launch(kernel, grid=grid, block=block, params=params)


def _avg_coords_per_geometry(buffer: FamilyGeometryBuffer) -> float:
    """Return average coordinate count per geometry for cooperative dispatch heuristic."""
    if buffer.row_count == 0:
        return 0.0
    return float(len(buffer.x)) / float(buffer.row_count)


def _launch_family_bounds_cooperative(
    family: GeometryFamily,
    device_buffer: DeviceFamilyGeometryBuffer,
    *,
    row_count: int,
    compute_type: str = "double",
) -> None:
    """Launch cooperative (block-per-geometry) bounds kernel for polygon/multipolygon."""
    if row_count == 0:
        return
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    cooperative_kernels = _bounds_cooperative_kernels(compute_type)
    if family is GeometryFamily.POLYGON:
        kernel = cooperative_kernels["bounds_polygon_cooperative"]
        params = (
            (
                ptr(device_buffer.x),
                ptr(device_buffer.y),
                ptr(device_buffer.geometry_offsets),
                ptr(device_buffer.ring_offsets),
                ptr(device_buffer.empty_mask),
                ptr(device_buffer.bounds),
                row_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
    elif family is GeometryFamily.MULTIPOLYGON:
        kernel = cooperative_kernels["bounds_multipolygon_cooperative"]
        params = (
            (
                ptr(device_buffer.x),
                ptr(device_buffer.y),
                ptr(device_buffer.geometry_offsets),
                ptr(device_buffer.part_offsets),
                ptr(device_buffer.ring_offsets),
                ptr(device_buffer.empty_mask),
                ptr(device_buffer.bounds),
                row_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
    else:
        raise ValueError(f"cooperative bounds not supported for family {family.value}")
    # 1 block per geometry; fixed at 256 to match __launch_bounds__(256, 4)
    # and shared memory sized for 8 warps (256 / 32).
    grid = (row_count, 1, 1)
    block = (256, 1, 1)
    runtime.launch(kernel, grid=grid, block=block, params=params)


def _compute_geometry_bounds_gpu_impl(
    geometry_array: OwnedGeometryArray,
    compute_type: str = "double",
):
    if cp is None:  # pragma: no cover - exercised on CPU-only installs
        raise RuntimeError("CuPy is not installed; GPU bounds execution is unavailable")
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    state = geometry_array._ensure_device_state()
    temp_bounds: list[tuple[GeometryFamily, DeviceFamilyGeometryBuffer, object]] = []
    _cooperative_families = frozenset({GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON})
    try:
        for family, device_buffer in state.families.items():
            if device_buffer.bounds is None:
                host_buffer = geometry_array.families[family]
                device_buffer.bounds = runtime.allocate((host_buffer.row_count, 4), cp.float64)
                temp_bounds.append((family, device_buffer, device_buffer.bounds))
                # Use cooperative (block-per-geometry) kernel when family supports it
                # and geometries are complex enough to benefit from warp-level reduction.
                use_cooperative = (
                    family in _cooperative_families
                    and _avg_coords_per_geometry(host_buffer) >= _COOPERATIVE_BOUNDS_THRESHOLD
                )
                if use_cooperative:
                    _launch_family_bounds_cooperative(
                        family, device_buffer,
                        row_count=host_buffer.row_count,
                        compute_type=compute_type,
                    )
                else:
                    _launch_family_bounds_kernel(
                        family, device_buffer,
                        row_count=host_buffer.row_count,
                        compute_type=compute_type,
                    )
        out_bounds = runtime.allocate((geometry_array.row_count, 4), cp.float64)
        try:
            kernel = _bounds_kernels(compute_type)["scatter_mixed_bounds"]
            params = (
                (
                    ptr(state.validity),
                    ptr(state.tags),
                    ptr(state.family_row_offsets),
                    0 if GeometryFamily.POINT not in state.families else ptr(state.families[GeometryFamily.POINT].bounds),
                    0 if GeometryFamily.LINESTRING not in state.families else ptr(state.families[GeometryFamily.LINESTRING].bounds),
                    0 if GeometryFamily.POLYGON not in state.families else ptr(state.families[GeometryFamily.POLYGON].bounds),
                    0 if GeometryFamily.MULTIPOINT not in state.families else ptr(state.families[GeometryFamily.MULTIPOINT].bounds),
                    0 if GeometryFamily.MULTILINESTRING not in state.families else ptr(state.families[GeometryFamily.MULTILINESTRING].bounds),
                    0 if GeometryFamily.MULTIPOLYGON not in state.families else ptr(state.families[GeometryFamily.MULTIPOLYGON].bounds),
                    ptr(out_bounds),
                    geometry_array.row_count,
                ),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                ),
            )
            grid, block = runtime.launch_config(kernel, geometry_array.row_count)
            runtime.launch(kernel, grid=grid, block=block, params=params)
            runtime.synchronize()
            bounds = runtime.copy_device_to_host(out_bounds)
        except Exception:
            runtime.free(out_bounds)
            raise
        # Cache per-row device bounds instead of freeing — avoids
        # recomputation for subsequent device-side bbox queries (dwithin).
        state.row_bounds = out_bounds
        geometry_array.cache_bounds(bounds)
        for family, _, device_bounds in temp_bounds:
            geometry_array.cache_device_bounds(family, device_bounds)
        return bounds
    except Exception:
        for _, device_buffer, device_bounds in temp_bounds:
            device_buffer.bounds = None
            runtime.free(device_bounds)
        raise


@register_kernel_variant(
    "compute_geometry_bounds",
    "gpu-cuda-python",
    kernel_class=KernelClass.COARSE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=tuple(family.value for family in GeometryFamily),
    supports_mixed=True,
    preferred_residency=Residency.DEVICE,
    tags=("cuda-python", "family-specialized"),
)
def _compute_geometry_bounds_gpu(
    geometry_array: OwnedGeometryArray,
    compute_type: str = "double",
):
    return _compute_geometry_bounds_gpu_impl(geometry_array, compute_type=compute_type)


def compute_geometry_bounds(
    geometry_array: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.CPU,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
):
    normalize_precision_mode(precision)
    row_count = geometry_array.row_count
    geometry_families = tuple(sorted(family.value for family in geometry_array.families))

    plan = plan_kernel_dispatch(
        kernel_name="compute_geometry_bounds",
        kernel_class=KernelClass.COARSE,
        row_count=row_count,
        geometry_families=geometry_families,
        mixed_geometry=len(geometry_families) > 1,
        current_residency=geometry_array.residency,
        requested_mode=dispatch_mode,
        requested_precision=precision,
    )
    selection = plan.runtime_selection
    geometry_array.record_runtime_selection(selection)

    if selection.selected is ExecutionMode.GPU:
        precision_plan = select_precision_plan(
            runtime_selection=selection,
            kernel_class=KernelClass.COARSE,
            requested=precision,
        )
        # Bounds always use fp64 compute: they are memory-bound (not compute-bound)
        # and fp32 rounding can shrink bounds, causing false negatives in spatial filtering.
        # The precision plan is still consulted for observability/diagnostics.
        try:
            geometry_array.move_to(
                Residency.DEVICE,
                trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                reason="compute_geometry_bounds selected GPU execution",
            )
            result = _compute_geometry_bounds_gpu(geometry_array, compute_type="double")
        except Exception:
            record_fallback_event(
                surface="geopandas.array.bounds",
                reason="GPU bounds kernel failed; falling back to CPU vectorized bounds",
                detail=f"rows={row_count}",
                requested=ExecutionMode.GPU,
                selected=ExecutionMode.CPU,
            )
        else:
            record_dispatch_event(
                surface="geopandas.array.bounds",
                operation="bounds",
                implementation="gpu_nvrtc_bounds",
                reason="GPU NVRTC bounds kernel",
                detail=f"rows={row_count}, precision={precision_plan.compute_precision}",
                selected=ExecutionMode.GPU,
            )
            return result

    if geometry_array.residency is Residency.DEVICE or any(
        not buffer.host_materialized for buffer in geometry_array.families.values()
    ):
        geometry_array.move_to(
            Residency.HOST,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="compute_geometry_bounds selected CPU execution",
        )
    record_dispatch_event(
        surface="geopandas.array.bounds",
        operation="bounds",
        implementation="cpu_vectorized",
        reason="CPU fallback",
        detail=f"rows={row_count}",
        selected=ExecutionMode.CPU,
    )
    return _compute_geometry_bounds_cpu_vectorized(geometry_array)


@register_kernel_variant(
    "compute_total_bounds",
    "cpu",
    kernel_class=KernelClass.COARSE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=tuple(family.value for family in GeometryFamily),
    supports_mixed=True,
)
def compute_total_bounds(
    geometry_array: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.CPU,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> tuple[float, float, float, float]:
    normalize_precision_mode(precision)
    bounds = compute_geometry_bounds(geometry_array, dispatch_mode=dispatch_mode, precision=precision)
    return compute_total_bounds_from_bounds(bounds)


@register_kernel_variant(
    "compute_offset_spans",
    "cpu",
    kernel_class=KernelClass.COARSE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=tuple(family.value for family in GeometryFamily),
    supports_mixed=True,
)
def compute_offset_spans(
    geometry_array: OwnedGeometryArray,
    *,
    level: str = "geometry",
    dispatch_mode: ExecutionMode = ExecutionMode.CPU,
) -> dict[GeometryFamily, object]:
    del dispatch_mode
    return compute_offset_spans_cpu(geometry_array, level=level)


@register_kernel_variant(
    "compute_morton_keys",
    "cpu",
    kernel_class=KernelClass.COARSE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=tuple(family.value for family in GeometryFamily),
    supports_mixed=True,
)
def compute_morton_keys(
    geometry_array: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.CPU,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
    bits: int = 16,
):
    del dispatch_mode
    normalize_precision_mode(precision)
    if bits != 16:
        raise ValueError("only 16-bit morton keys are currently supported")
    bounds = compute_geometry_bounds(geometry_array, precision=precision)
    total = compute_total_bounds(geometry_array, precision=precision)
    minx, miny, maxx, maxy = total
    return compute_morton_keys_cpu(bounds, total, geometry_array.row_count, bits=bits)
