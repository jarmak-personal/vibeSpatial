"""Shared CUDA C++ ``__device__`` function source strings.

Each sibling ``.py`` module exports one or more string constants containing
self-contained ``__device__`` functions as CUDA C++ source.  Kernel authors
prepend these strings to their kernel source::

    from vibespatial.cuda.device_functions.orientation import VS_ORIENT2D

    _MY_KERNEL = VS_ORIENT2D + r\"\"\"
    __global__ void my_kernel(...) { ... }
    \"\"\"

See ``README.md`` in this package for conventions and guidelines.
"""
