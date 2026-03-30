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
