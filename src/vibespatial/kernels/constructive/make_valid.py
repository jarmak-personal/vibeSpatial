from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from vibespatial.constructive.make_valid_pipeline import MakeValidResult, make_valid_owned
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass

MakeValidInput = Sequence[object | None] | np.ndarray | object


@register_kernel_variant(
    "make_valid",
    "cpu",
    kernel_class=KernelClass.CONSTRUCTIVE,
    geometry_families=("polygon", "multipolygon", "linestring", "multilinestring"),
    execution_modes=(ExecutionMode.CPU,),
    supports_mixed=True,
    tags=("constructive", "make-valid", "owned"),
)
def make_valid_kernel(
    values: MakeValidInput,
    *,
    method: str = "linework",
    keep_collapsed: bool = True,
) -> MakeValidResult:
    return make_valid_owned(values, method=method, keep_collapsed=keep_collapsed)


@register_kernel_variant(
    "make_valid",
    "gpu-cuda-python",
    kernel_class=KernelClass.CONSTRUCTIVE,
    geometry_families=("polygon", "multipolygon"),
    execution_modes=(ExecutionMode.GPU,),
    supports_mixed=True,
    tags=("constructive", "make-valid", "gpu", "owned"),
)
def make_valid_gpu_kernel(
    values: MakeValidInput,
    *,
    method: str = "linework",
    keep_collapsed: bool = True,
    owned=None,
) -> MakeValidResult:
    """GPU-accelerated make_valid using NVRTC repair kernels (ADR-0019 + ADR-0033).

    When an OwnedGeometryArray with device_state is provided, runs the full
    GPU repair pipeline: ring closure, duplicate removal, orientation fix,
    self-intersection splitting, and re-polygonization.  Falls back to CPU
    Shapely path for non-polygon families and when GPU repair fails.
    """
    return make_valid_owned(values, method=method, keep_collapsed=keep_collapsed, owned=owned)
