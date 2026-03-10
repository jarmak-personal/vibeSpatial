# Installation

## Basic install (CPU)

```bash
pip install vibespatial
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add vibespatial
```

## GPU support

GPU acceleration requires an NVIDIA GPU with CUDA 12+ drivers. Install the
GPU extras:

```bash
pip install vibespatial[gpu]
```

This pulls in CuPy, CUDA CCCL, and pylibcudf.

## Development install

```bash
git clone https://github.com/vibeSpatial/vibeSpatial.git
cd vibeSpatial
uv sync
```

To include GPU dependencies:

```bash
uv sync --group gpu-optional
```

## Verify GPU availability

```python
import vibespatial

sel = vibespatial.select_runtime()
print(sel)
# RuntimeSelection(requested=auto, selected=gpu, reason=...)
```

If no GPU is available, `selected` will be `cpu` and all operations
will use the CPU fallback path transparently.
