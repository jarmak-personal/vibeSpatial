from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class GpuTelemetrySample:
    sm_utilization_pct: float
    memory_utilization_pct: float
    used_bytes: int
    total_bytes: int
    device_name: str = "unknown"


class NvmlGpuSampler:
    def __init__(self) -> None:
        self._available = False
        self._pynvml = None
        self._handle = None
        try:
            pynvml = import_module("pynvml")
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._pynvml = pynvml
            self._available = True
        except Exception:
            self._available = False
            self._pynvml = None
            self._handle = None

    @property
    def available(self) -> bool:
        return self._available

    def sample(self) -> GpuTelemetrySample | None:
        if not self._available or self._pynvml is None or self._handle is None:
            return None
        try:
            utilization = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            memory = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            name = self._pynvml.nvmlDeviceGetName(self._handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            return GpuTelemetrySample(
                sm_utilization_pct=float(utilization.gpu),
                memory_utilization_pct=float(utilization.memory),
                used_bytes=int(memory.used),
                total_bytes=int(memory.total),
                device_name=str(name),
            )
        except Exception:
            return None


_NvmlGpuSampler = NvmlGpuSampler
