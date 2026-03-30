"""Shared CUDA C++ ``__device__`` function source strings.

Each sibling ``.py`` module exports one or more string constants containing
self-contained ``__device__`` functions as CUDA C++ source.  Kernel authors
prepend these strings to their kernel source::

    from vibespatial.cuda.device_functions.orient2d import ORIENT2D_DEVICE

    _MY_KERNEL = ORIENT2D_DEVICE + r\"\"\"
    __global__ void my_kernel(...) { ... }
    \"\"\"

See ``README.md`` in this package for conventions and guidelines.
"""

# Re-export for convenience
from vibespatial.cuda.device_functions.orient2d import ORIENT2D_DEVICE  # noqa: F401
from vibespatial.cuda.device_functions.point_on_segment import (  # noqa: F401
    POINT_ON_SEGMENT_DEVICE,
    POINT_ON_SEGMENT_KIND_DEVICE,
)
from vibespatial.cuda.device_functions.signed_area import SIGNED_AREA_DEVICE  # noqa: F401
