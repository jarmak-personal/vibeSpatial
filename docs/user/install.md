# Installation

## Basic install (CPU)

```bash
pip install vibespatial
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add vibespatial
```

This gives you a fully functional GeoPandas drop-in. All operations work
via CPU fallback.

## GPU support

GPU acceleration requires an NVIDIA GPU with compatible drivers. Install
the extra matching your CUDA version:

```bash
pip install vibespatial[cu12]    # CUDA 12
pip install vibespatial[cu13]    # CUDA 13
```

This pulls in CuPy, cuda-python, CUDA CCCL, pylibcudf, and nvidia-ml-py.

> **Note:** The CUDA toolkit version installed by the extras must not exceed
> your NVIDIA driver version. Check with `nvidia-smi` — the "CUDA Version"
> shown is the maximum supported.

## Development install

```bash
git clone https://github.com/jarmak-personal/vibeSpatial.git
cd vibeSpatial
uv sync                          # CPU only
uv sync --extra cu12             # with CUDA 12 GPU deps
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
