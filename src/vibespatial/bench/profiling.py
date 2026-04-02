from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from importlib import import_module
from threading import Event, Thread
from time import perf_counter
from typing import Any

from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.gpu_sampling import GpuTelemetrySample, NvmlGpuSampler

_NVTX_COLORS = {
    "setup": "blue",
    "sort": "purple",
    "filter": "green",
    "refine": "red",
    "emit": "orange",
}


_SPARKLINE_BARS = " ▁▂▃▄▅▆▇█"


def _format_percent(value: float) -> str:
    return f"{value:.0f}%"


def _format_bytes_mib(value: int) -> str:
    return f"{value / (1024 * 1024):.0f}MiB"


def _sparkline(values: list[float], *, width: int = 28) -> str:
    if not values:
        return ""
    if len(values) > width:
        buckets: list[float] = []
        for start in range(0, len(values), max(1, len(values) // width)):
            chunk = values[start : start + max(1, len(values) // width)]
            buckets.append(sum(chunk) / len(chunk))
        values = buckets[:width]
    minimum = min(values)
    maximum = max(values)
    if maximum <= minimum:
        return _SPARKLINE_BARS[1] * len(values)
    pieces = []
    for value in values:
        normalized = (value - minimum) / (maximum - minimum)
        index = min(len(_SPARKLINE_BARS) - 1, max(1, round(normalized * (len(_SPARKLINE_BARS) - 1))))
        pieces.append(_SPARKLINE_BARS[index])
    return "".join(pieces)


_NvmlGpuSampler = NvmlGpuSampler


@dataclass
class _StageGpuTelemetryCollector:
    sampler: Any
    interval_seconds: float = 0.001
    retain_trace: bool = False
    include_sparklines: bool = False
    _stop: Event = field(default_factory=Event)
    _thread: Thread | None = None
    _samples: list[tuple[float, GpuTelemetrySample]] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return getattr(self.sampler, "available", False)

    def start(self) -> None:
        if not self.available:
            return

        def _run() -> None:
            origin = perf_counter()
            while not self._stop.is_set():
                sample = self.sampler.sample()
                if sample is not None:
                    self._samples.append((perf_counter() - origin, sample))
                self._stop.wait(self.interval_seconds)

        self._thread = Thread(target=_run, name="vibespatial-gpu-profiler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(0.1, self.interval_seconds * 4))

    def summarize(self) -> dict[str, Any]:
        if not self._samples:
            return {}
        first = self._samples[0][1]
        last = self._samples[-1][1]
        sm_values = [sample.sm_utilization_pct for _, sample in self._samples]
        mem_values = [sample.memory_utilization_pct for _, sample in self._samples]
        used_values = [sample.used_bytes for _, sample in self._samples]
        summary = {
            "gpu_device_name": first.device_name,
            "gpu_sampling_interval_ms": round(self.interval_seconds * 1000, 3),
            "gpu_sample_count": len(self._samples),
            "gpu_utilization_pct_start": first.sm_utilization_pct,
            "gpu_utilization_pct_end": last.sm_utilization_pct,
            "gpu_utilization_pct_avg": sum(sm_values) / len(sm_values),
            "gpu_utilization_pct_max": max(sm_values),
            "gpu_memory_utilization_pct_start": first.memory_utilization_pct,
            "gpu_memory_utilization_pct_end": last.memory_utilization_pct,
            "gpu_memory_utilization_pct_avg": sum(mem_values) / len(mem_values),
            "gpu_memory_utilization_pct_max": max(mem_values),
            "gpu_vram_used_bytes_start": first.used_bytes,
            "gpu_vram_used_bytes_end": last.used_bytes,
            "gpu_vram_used_bytes_max": max(used_values),
            "gpu_vram_total_bytes": first.total_bytes,
        }
        if self.retain_trace:
            summary["gpu_trace"] = [
                {
                    "t_ms": round(offset_seconds * 1000, 3),
                    "gpu_util_pct": sample.sm_utilization_pct,
                    "mem_util_pct": sample.memory_utilization_pct,
                    "vram_used_bytes": sample.used_bytes,
                }
                for offset_seconds, sample in self._samples
            ]
        if self.include_sparklines:
            summary["gpu_util_sparkline"] = (
                f"{_format_percent(min(sm_values))} |{_sparkline(sm_values)}| {_format_percent(max(sm_values))}"
            )
            summary["gpu_memory_util_sparkline"] = (
                f"{_format_percent(min(mem_values))} |{_sparkline(mem_values)}| {_format_percent(max(mem_values))}"
            )
            summary["gpu_vram_sparkline"] = (
                f"{_format_bytes_mib(min(used_values))} |{_sparkline([float(v) for v in used_values])}| "
                f"{_format_bytes_mib(max(used_values))}"
            )
        return summary


@dataclass
class StageMeasurement:
    rows_out: int | None = None
    detail: str = ""
    device: ExecutionMode | str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProfileStageTrace:
    name: str
    category: str
    device: str
    elapsed_seconds: float
    rows_in: int | None = None
    rows_out: int | None = None
    detail: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "device": self.device,
            "elapsed_seconds": self.elapsed_seconds,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "detail": self.detail,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ProfileTrace:
    operation: str
    dataset: str
    requested_runtime: str
    selected_runtime: str
    total_elapsed_seconds: float
    nvtx_enabled: bool
    stages: tuple[ProfileStageTrace, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "dataset": self.dataset,
            "requested_runtime": self.requested_runtime,
            "selected_runtime": self.selected_runtime,
            "total_elapsed_seconds": self.total_elapsed_seconds,
            "nvtx_enabled": self.nvtx_enabled,
            "stages": [stage.to_dict() for stage in self.stages],
            "metadata": self.metadata,
        }


def _maybe_nvtx_context(label: str, category: str, *, enabled: bool):
    if not enabled:
        return nullcontext()
    try:
        nvtx = import_module("nvtx")
    except ImportError:
        return nullcontext()
    color = _NVTX_COLORS.get(category, "blue")
    return nvtx.annotate(label, color=color)


class StageProfiler:
    def __init__(
        self,
        *,
        operation: str,
        dataset: str,
        requested_runtime: ExecutionMode | str,
        selected_runtime: ExecutionMode | str,
        enable_nvtx: bool = False,
        gpu_sampler: Any | None = None,
        gpu_sample_interval_seconds: float = 0.001,
        retain_gpu_trace: bool = False,
        include_gpu_sparklines: bool = False,
    ) -> None:
        self.operation = operation
        self.dataset = dataset
        self.requested_runtime = (
            requested_runtime.value if isinstance(requested_runtime, ExecutionMode) else str(requested_runtime)
        )
        self.selected_runtime = (
            selected_runtime.value if isinstance(selected_runtime, ExecutionMode) else str(selected_runtime)
        )
        self.enable_nvtx = enable_nvtx
        self._gpu_sampler = gpu_sampler if gpu_sampler is not None else _NvmlGpuSampler()
        self._gpu_sample_interval_seconds = gpu_sample_interval_seconds
        self._retain_gpu_trace = retain_gpu_trace
        self._include_gpu_sparklines = include_gpu_sparklines
        self._started = perf_counter()
        self._stages: list[ProfileStageTrace] = []

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        category: str,
        device: ExecutionMode | str,
        rows_in: int | None = None,
        detail: str = "",
        metadata: dict[str, Any] | None = None,
    ):
        measurement = StageMeasurement(detail=detail, device=device, metadata=dict(metadata or {}))
        label = f"{self.operation}:{name}"
        selected_device = device.value if isinstance(device, ExecutionMode) else str(device)
        gpu_collector = _StageGpuTelemetryCollector(
            sampler=self._gpu_sampler,
            interval_seconds=self._gpu_sample_interval_seconds,
            retain_trace=self._retain_gpu_trace,
            include_sparklines=self._include_gpu_sparklines,
        )
        gpu_collector.start()
        started = perf_counter()
        with _maybe_nvtx_context(label, category, enabled=self.enable_nvtx):
            yield measurement
        elapsed = perf_counter() - started
        gpu_collector.stop()
        measurement.metadata.update(gpu_collector.summarize())
        self._stages.append(
            ProfileStageTrace(
                name=name,
                category=category,
                device=measurement.device.value if isinstance(measurement.device, ExecutionMode) else (measurement.device or selected_device),
                elapsed_seconds=elapsed,
                rows_in=rows_in,
                rows_out=measurement.rows_out,
                detail=measurement.detail,
                metadata=measurement.metadata,
            )
        )

    def finish(self, *, metadata: dict[str, Any] | None = None) -> ProfileTrace:
        return ProfileTrace(
            operation=self.operation,
            dataset=self.dataset,
            requested_runtime=self.requested_runtime,
            selected_runtime=self.selected_runtime,
            total_elapsed_seconds=perf_counter() - self._started,
            nvtx_enabled=self.enable_nvtx,
            stages=tuple(self._stages),
            metadata=dict(metadata or {}),
        )
