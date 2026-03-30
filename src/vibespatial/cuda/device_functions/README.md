# cuda/device_functions/ -- Shared CUDA C++ device function strings

This package holds shared CUDA C++ `__device__` function source strings
that are reused across multiple NVRTC kernels.

## Conventions

- Each `.py` file exports one or more **string constants** containing
  `__device__` functions as raw CUDA C++ source (use `r"""..."""`).
- Consumers prepend them to kernel source:
  `_MY_KERNEL = SHARED_FUNC + r"""..."""`
- Each function string is **self-contained** -- no external `#include`
  directives; all dependencies are inlined.
- Function names use the **`vs_`** prefix (vibeSpatial) to avoid symbol
  collisions when multiple device-function strings are concatenated.
- Specialized fast-path variants (e.g. point-in-ring vs generic
  point-in-polygon) belong here as centralized shared code, **not**
  inlined in individual operation kernel files.
- Changes here affect **every kernel** that includes the string.
  After modifying a device function, re-run the full kernel test suite.
