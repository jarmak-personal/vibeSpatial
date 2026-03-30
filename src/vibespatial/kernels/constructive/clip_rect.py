from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np  # hygiene:ok — used for type alias, not device computation

from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode

if TYPE_CHECKING:
    from vibespatial.constructive.clip_rect import RectClipResult

ClipInput = Sequence[object | None] | np.ndarray | object


@register_kernel_variant(
    "clip_by_rect",
    "cpu",
    kernel_class=KernelClass.CONSTRUCTIVE,
    geometry_families=("point", "multipoint", "linestring", "multilinestring", "polygon", "multipolygon"),
    execution_modes=(ExecutionMode.CPU,),
    supports_mixed=True,
    tags=("constructive", "rect-clip", "owned"),
)
def clip_by_rect_kernel(
    values: ClipInput,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> RectClipResult:
    from vibespatial.constructive.clip_rect import clip_by_rect_owned  # lazy

    return clip_by_rect_owned(
        values,
        xmin,
        ymin,
        xmax,
        ymax,
        dispatch_mode=dispatch_mode,
        precision=precision,
    )
