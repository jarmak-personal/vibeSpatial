"""Bounded, opt-in measurements for exact point/region GPU refinement.

This module is deliberately private.  Public spatial APIs remain the workload
entry point; a profiling session only substitutes instrumented copies of the
prepared point-location kernels and exports one fixed-size control packet when
the session is closed or snapshotted.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    get_cuda_runtime,
)

_COUNTER_NAMES = (
    "candidates",
    "valid_candidates",
    "parts_considered",
    "active_parts",
    "edges_visited",
    "orient2d_calls",
    "zero_active_candidates",
    "zero_edge_candidates",
    "boundary_results",
    "interior_results",
    "exterior_results",
    "max_parts_considered",
    "max_active_parts",
    "max_edges_visited",
    "sample_reservations",
    "sampled_candidates",
)
_HISTOGRAM_BINS = 4097
_DEFAULT_SAMPLE_LIMIT = 65_536
_SAMPLES_PER_LAUNCH = 128


@dataclass
class _ProfileGroup:
    key: tuple[int, str]
    family: str
    geometry_count: int
    part_count: int
    bin_count: int
    coverage_grid_width: int
    edge_membership_count: int
    index_device_bytes: int
    summary: Any
    parts_histogram: Any
    edges_histogram: Any
    sample_plan: Any
    launches: int = 0
    start_events: list[Any] = field(default_factory=list)
    end_events: list[Any] = field(default_factory=list)


_ACTIVE_SESSION: ContextVar[PointRegionProfileSession | None] = ContextVar(
    "vibespatial_point_region_profile", default=None
)


def current_point_region_profile() -> PointRegionProfileSession | None:
    """Return the current private profiling session, if one is active."""
    return _ACTIVE_SESSION.get()


class PointRegionProfileSession:
    """Collect bounded physical-work evidence around public spatial calls."""

    def __init__(
        self,
        *,
        label: str,
        sample_limit: int = _DEFAULT_SAMPLE_LIMIT,
        force_prepared_index: bool = False,
    ) -> None:
        if sample_limit < 0:
            raise ValueError("sample_limit must be non-negative")
        self.label = str(label)
        self.sample_limit = min(int(sample_limit), _DEFAULT_SAMPLE_LIMIT)
        self.force_prepared_index = bool(force_prepared_index)
        self._token: Token | None = None
        self._groups: dict[tuple[int, str], _ProfileGroup] = {}
        self._preparation: dict[tuple[int, str], dict[str, Any]] = {}
        self._started_at = 0.0

    def __enter__(self) -> PointRegionProfileSession:
        if self._token is not None:
            raise RuntimeError("point-region profiling session is already active")
        self._started_at = perf_counter()
        self._token = _ACTIVE_SESSION.set(self)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        assert self._token is not None
        _ACTIVE_SESSION.reset(self._token)
        self._token = None

    def note_index_cache_hit(self, prepared) -> None:
        key = (id(prepared), prepared.family.value)
        record = self._preparation.setdefault(
            key,
            {
                "family": prepared.family.value,
                "nominal_vram_class_gib": int(prepared.nominal_vram_class_gib),
                "target_bin_count": int(prepared.target_bin_count),
                "admitted_bin_count": int(prepared.bin_count),
                "coverage_grid_width": int(prepared.coverage_grid_width),
                "coverage_decline_reason": prepared.coverage_decline_reason,
                "decline_reason": prepared.decline_reason,
                "edge_membership_count": int(prepared.edge_membership_count),
                "persistent_bytes": int(prepared.device_bytes),
                "peak_build_bytes": int(prepared.peak_build_bytes),
                "build_count": 0,
                "cache_hits": 0,
                "build_wall_seconds": 0.0,
            },
        )
        record["cache_hits"] += 1

    def note_index_build(self, prepared, wall_seconds: float) -> None:
        key = (id(prepared), prepared.family.value)
        record = self._preparation.setdefault(
            key,
            {
                "family": prepared.family.value,
                "nominal_vram_class_gib": int(prepared.nominal_vram_class_gib),
                "target_bin_count": int(prepared.target_bin_count),
                "admitted_bin_count": int(prepared.bin_count),
                "coverage_grid_width": int(prepared.coverage_grid_width),
                "coverage_decline_reason": prepared.coverage_decline_reason,
                "decline_reason": prepared.decline_reason,
                "edge_membership_count": int(prepared.edge_membership_count),
                "persistent_bytes": int(prepared.device_bytes),
                "peak_build_bytes": int(prepared.peak_build_bytes),
                "build_count": 0,
                "cache_hits": 0,
                "build_wall_seconds": 0.0,
            },
        )
        record["build_count"] += 1
        record["build_wall_seconds"] += float(wall_seconds)

    def launch_arguments(self, prepared) -> tuple[Any, Any, Any, Any]:
        """Return persistent device counters for one prepared index."""
        import cupy as cp

        key = (id(prepared), prepared.family.value)
        group = self._groups.get(key)
        if group is None:
            group = _ProfileGroup(
                key=key,
                family=prepared.family.value,
                geometry_count=int(prepared.geometry_count),
                part_count=int(prepared.part_count),
                bin_count=int(prepared.bin_count),
                coverage_grid_width=int(prepared.coverage_grid_width),
                edge_membership_count=int(prepared.edge_membership_count),
                index_device_bytes=int(prepared.device_bytes),
                summary=cp.zeros(len(_COUNTER_NAMES), dtype=cp.uint64),
                parts_histogram=cp.zeros(_HISTOGRAM_BINS, dtype=cp.uint64),
                edges_histogram=cp.zeros(_HISTOGRAM_BINS, dtype=cp.uint64),
                sample_plan=cp.zeros(1, dtype=cp.uint64),
            )
            self._groups[key] = group
        return (
            group.summary,
            group.parts_histogram,
            group.edges_histogram,
            group.sample_plan,
        )

    def begin_launch(self, prepared, *, logical_count, candidate_count: int) -> None:
        import cupy as cp

        from .point_location_index import point_location_part_y_index_profile_kernels

        key = (id(prepared), prepared.family.value)
        group = self._groups[key]
        runtime = get_cuda_runtime()
        reserve_kernel = point_location_part_y_index_profile_kernels(prepared.bin_count)[
            "reserve_point_region_profile_samples"
        ]
        grid, block = runtime.launch_config(reserve_kernel, 1)
        runtime.launch(
            reserve_kernel,
            grid=grid,
            block=block,
            params=(
                (
                    runtime.pointer(group.summary),
                    runtime.pointer(group.sample_plan),
                    runtime.pointer(logical_count),
                    self.sample_limit,
                    int(candidate_count),
                ),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_I32,
                ),
            ),
        )
        event = cp.cuda.Event()
        event.record()
        group.start_events.append(event)

    def end_launch(self, prepared) -> None:
        import cupy as cp

        key = (id(prepared), prepared.family.value)
        group = self._groups[key]
        event = cp.cuda.Event()
        event.record()
        group.end_events.append(event)
        group.launches += 1

    @staticmethod
    def _percentile(histogram: np.ndarray, percentile: float) -> int | None:
        total = int(histogram.sum())
        if total == 0:
            return None
        target = max(1, int(np.ceil(total * percentile)))
        index = int(np.searchsorted(np.cumsum(histogram), target, side="left"))
        return index

    def snapshot(self) -> dict[str, Any]:
        """Synchronize once and export fixed-size counter packets to host."""
        import cupy as cp

        runtime = get_cuda_runtime()
        groups: list[dict[str, Any]] = []
        for group in self._groups.values():
            if group.end_events:
                group.end_events[-1].synchronize()
            summary = np.empty(len(_COUNTER_NAMES), dtype=np.uint64)
            parts_histogram = np.empty(_HISTOGRAM_BINS, dtype=np.uint64)
            edges_histogram = np.empty(_HISTOGRAM_BINS, dtype=np.uint64)
            runtime.copy_device_to_host(
                group.summary,
                summary,
                reason="point-region profiler fixed summary packet",
            )
            runtime.copy_device_to_host(
                group.parts_histogram,
                parts_histogram,
                reason="point-region profiler bounded parts histogram",
            )
            runtime.copy_device_to_host(
                group.edges_histogram,
                edges_histogram,
                reason="point-region profiler bounded edges histogram",
            )
            kernel_ms = sum(
                float(cp.cuda.get_elapsed_time(start, end))
                for start, end in zip(
                    group.start_events, group.end_events, strict=True
                )
            )
            counters = {
                name: int(value)
                for name, value in zip(_COUNTER_NAMES, summary, strict=True)
            }
            groups.append(
                {
                    "family": group.family,
                    "geometry_count": group.geometry_count,
                    "part_count": group.part_count,
                    "bin_count": group.bin_count,
                    "coverage_grid_width": group.coverage_grid_width,
                    "mean_parts_per_geometry": (
                        group.part_count / group.geometry_count
                        if group.geometry_count
                        else 0.0
                    ),
                    "edge_membership_count": group.edge_membership_count,
                    "mean_edge_memberships_per_part_bin": (
                        group.edge_membership_count
                        / (group.part_count * group.bin_count)
                        if group.part_count
                        else 0.0
                    ),
                    "index_device_bytes": group.index_device_bytes,
                    "launches": group.launches,
                    "kernel_seconds": kernel_ms / 1000.0,
                    "counters": counters,
                    "parts_considered_percentiles": {
                        "p50": self._percentile(parts_histogram, 0.50),
                        "p95": self._percentile(parts_histogram, 0.95),
                        "p99": self._percentile(parts_histogram, 0.99),
                    },
                    "edges_visited_percentiles": {
                        "p50": self._percentile(edges_histogram, 0.50),
                        "p95": self._percentile(edges_histogram, 0.95),
                        "p99": self._percentile(edges_histogram, 0.99),
                    },
                    "parts_histogram_ge_4096": int(parts_histogram[-1]),
                    "edges_histogram_ge_4096": int(edges_histogram[-1]),
                }
            )
        return {
            "schema_version": 4,
            "label": self.label,
            "profile_wall_seconds": perf_counter() - self._started_at,
            "sample_limit": self.sample_limit,
            "samples_per_launch": _SAMPLES_PER_LAUNCH,
            "forced_prepared_index": self.force_prepared_index,
            "groups": groups,
            "index_preparation": list(self._preparation.values()),
            "memory_pool": runtime.memory_pool_stats(),
        }


def profile_point_region(
    *,
    label: str,
    sample_limit: int = _DEFAULT_SAMPLE_LIMIT,
    force_prepared_index: bool = False,
) -> PointRegionProfileSession:
    """Create a private profiler to wrap calls made through public APIs."""
    return PointRegionProfileSession(
        label=label,
        sample_limit=sample_limit,
        force_prepared_index=force_prepared_index,
    )
