"""vsbench shootout — run a user script with geopandas then vibespatial."""
from __future__ import annotations

import io
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import TimingSummary, timing_from_samples

_TIMED_START_MARKER = "# --- timed work starts here ---"
_TIMED_END_MARKER = "# --- timed work ends here ---"

# ---------------------------------------------------------------------------
# Harness script — executed inside the subprocess
# ---------------------------------------------------------------------------

_HARNESS_CODE = """\
import ast
import faulthandler
import io
import json
import os
from pathlib import Path
import runpy
import sys
import time
import warnings

warnings.filterwarnings("ignore")

script = sys.argv[1]
result_path = sys.argv[2]
repeat = int(sys.argv[3])
do_warmup = sys.argv[4] == "1"
do_pipeline_warm = sys.argv[5] == "1"
dump_after = int(sys.argv[6])
do_profile = sys.argv[7] == "1"

if dump_after > 0:
    faulthandler.dump_traceback_later(dump_after, repeat=False)

TIMED_START_MARKER = "# --- timed work starts here ---"
TIMED_END_MARKER = "# --- timed work ends here ---"


def _load_script_sections(script_path):
    text = Path(script_path).read_text(encoding="utf-8")
    start = text.find(TIMED_START_MARKER)
    end = text.find(TIMED_END_MARKER)
    if start == -1 or end == -1 or end < start:
        return None
    timed_start = start + len(TIMED_START_MARKER)
    timed_source = text[timed_start:end]
    timed_line_offset = text[:timed_start].count("\\n") + 1
    return (
        compile(text[:start], script_path, "exec"),
        compile(timed_source, script_path, "exec"),
        compile(text[end + len(TIMED_END_MARKER):], script_path, "exec"),
        timed_source,
        timed_line_offset,
    )


def _run_timed_sections(script_path, sections, *, run_name):
    preamble_code, timed_code, postamble_code, _timed_source, _timed_line_offset = sections
    globals_dict = {
        "__name__": run_name,
        "__file__": script_path,
        "__package__": None,
        "__cached__": None,
    }
    exec(preamble_code, globals_dict)
    start = time.perf_counter()
    exec(timed_code, globals_dict)
    elapsed = time.perf_counter() - start
    strict_native = os.environ.pop("VIBESPATIAL_STRICT_NATIVE", None)
    try:
        exec(postamble_code, globals_dict)
    finally:
        if strict_native is not None:
            os.environ["VIBESPATIAL_STRICT_NATIVE"] = strict_native
    return elapsed


def _as_dict_event(event):
    to_dict = getattr(event, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return {
        key: value
        for key, value in getattr(event, "__dict__", {}).items()
        if not key.startswith("_")
    }


def _trace_step_to_dict(step):
    selected = getattr(step, "selected", "")
    return {
        "surface": getattr(step, "surface", ""),
        "operation": getattr(step, "operation", ""),
        "selected": getattr(selected, "value", str(selected)),
        "implementation": getattr(step, "implementation", ""),
    }


def _trace_transfer_to_dict(transfer):
    return {
        "direction": getattr(transfer, "direction", ""),
        "trigger": getattr(transfer, "trigger", ""),
        "reason": getattr(transfer, "reason", ""),
        "source": getattr(transfer, "source", "semantic"),
        "item_count": int(getattr(transfer, "item_count", 0) or 0),
        "bytes_transferred": int(getattr(transfer, "bytes_transferred", 0) or 0),
        "elapsed_seconds": float(getattr(transfer, "elapsed_seconds", 0.0) or 0.0),
    }


def _statement_tags(source):
    lower = source.lower()
    tags = []

    def add(tag):
        if tag not in tags:
            tags.append(tag)

    if "read_parquet(" in lower or "read_file(" in lower or "read_feather(" in lower:
        add("io_read")
    if "to_parquet(" in lower or "to_file(" in lower or "to_feather(" in lower:
        add("io_write")
    if "gpd.clip(" in lower or ".clip(" in lower:
        add("mask_clip")
    if ".buffer(" in lower:
        add("buffer")
        add("mask_construction")
    if "gpd.sjoin(" in lower or ".sjoin(" in lower:
        add("spatial_join")
    if "gpd.overlay(" in lower or ".overlay(" in lower:
        add("many_few_overlay")
    if ".dissolve(" in lower:
        add("grouped_geometry_reduce")
    if ".area" in lower:
        add("area_filter_after_overlay")
    if ".convex_hull" in lower:
        add("convex_hull")
    if ".geom_type" in lower or ".isin(" in lower or ".loc[" in lower or ".iloc[" in lower:
        add("tabular_filter")
    if ".copy(" in lower:
        add("copy")
    if "notify_dispatch(" in lower:
        add("trace_dispatch")
    if "notify_transfer(" in lower:
        add("trace_transfer")
    if "record_fallback_event(" in lower:
        add("fallback")
    if "hotpath_stage(" in lower:
        add("hotpath")
    if not tags:
        add("python")
    return tags


def _compile_timed_statements(script_path, timed_source, timed_line_offset):
    try:
        tree = ast.parse(timed_source, filename=script_path, mode="exec")
    except SyntaxError:
        return []
    lines = timed_source.splitlines()
    compiled = []
    for index, stmt in enumerate(tree.body):
        line_start = timed_line_offset + getattr(stmt, "lineno", 1) - 1
        line_end = timed_line_offset + getattr(stmt, "end_lineno", getattr(stmt, "lineno", 1)) - 1
        source = ast.get_source_segment(timed_source, stmt)
        if source is None:
            local_start = max(getattr(stmt, "lineno", 1) - 1, 0)
            local_end = max(getattr(stmt, "end_lineno", getattr(stmt, "lineno", 1)), local_start + 1)
            source = "\\n".join(lines[local_start:local_end])
        module = ast.Module(body=[stmt], type_ignores=[])
        ast.fix_missing_locations(module)
        compiled.append({
            "index": index,
            "line_start": line_start,
            "line_end": line_end,
            "source": source.strip(),
            "tags": _statement_tags(source),
            "code": compile(module, script_path, "exec"),
        })
    return compiled


def _summarize_statement_profile(stages):
    totals_by_tag = {}
    totals_by_backend = {}
    materializations_by_boundary = {}
    materializations_by_surface = {}
    materialization_count = 0
    materialization_d2h_count = 0
    for stage in stages:
        elapsed = float(stage.get("elapsed_seconds", 0.0))
        backend = stage.get("actual_backend", "unknown")
        totals_by_backend[backend] = totals_by_backend.get(backend, 0.0) + elapsed
        for tag in stage.get("tags", []):
            totals_by_tag[tag] = totals_by_tag.get(tag, 0.0) + elapsed
        materialization_count += int(stage.get("materialization_event_count", 0) or 0)
        materialization_d2h_count += int(stage.get("materialization_d2h_event_count", 0) or 0)
        for event in stage.get("materialization_events", []):
            boundary = event.get("boundary", "")
            surface = event.get("surface", "")
            materializations_by_boundary[boundary] = materializations_by_boundary.get(boundary, 0) + 1
            materializations_by_surface[surface] = materializations_by_surface.get(surface, 0) + 1
    return {
        "stage_total_seconds": sum(float(stage.get("elapsed_seconds", 0.0)) for stage in stages),
        "stage_totals_by_tag": dict(sorted(totals_by_tag.items(), key=lambda item: (-item[1], item[0]))),
        "stage_totals_by_backend": dict(sorted(totals_by_backend.items(), key=lambda item: (-item[1], item[0]))),
        "stage_materialization_count": materialization_count,
        "stage_materialization_d2h_event_count": materialization_d2h_count,
        "stage_materialization_counts_by_boundary": dict(
            sorted(materializations_by_boundary.items(), key=lambda item: (-item[1], item[0]))
        ),
        "stage_materialization_counts_by_surface": dict(
            sorted(materializations_by_surface.items(), key=lambda item: (-item[1], item[0]))
        ),
    }


def _profile_execution_deltas(trace, before_steps, before_transfers):
    steps = list(getattr(trace, "steps", []))[before_steps:]
    transfers = list(getattr(trace, "transfers", []))[before_transfers:]
    gpu_steps = sum(1 for step in steps if str(getattr(getattr(step, "selected", ""), "value", "")) == "gpu")
    cpu_steps = sum(1 for step in steps if str(getattr(getattr(step, "selected", ""), "value", "")) == "cpu")
    d2h_transfers = [
        transfer for transfer in transfers
        if getattr(transfer, "direction", "") == "d2h"
    ]
    h2d_transfers = [
        transfer for transfer in transfers
        if getattr(transfer, "direction", "") == "h2d"
    ]
    runtime_d2h = [
        transfer for transfer in d2h_transfers
        if getattr(transfer, "source", "semantic") == "cuda_runtime"
    ]
    runtime_h2d = [
        transfer for transfer in h2d_transfers
        if getattr(transfer, "source", "semantic") == "cuda_runtime"
    ]
    if gpu_steps and cpu_steps:
        backend = "mixed"
    elif gpu_steps:
        backend = "gpu"
    elif cpu_steps:
        backend = "cpu"
    else:
        backend = "python"
    return {
        "actual_backend": backend,
        "trace_steps": len(steps),
        "gpu_steps": gpu_steps,
        "cpu_steps": cpu_steps,
        "d2h_transfers": len(d2h_transfers),
        "h2d_transfers": len(h2d_transfers),
        "d2h_transfer_bytes": sum(
            int(getattr(transfer, "bytes_transferred", 0) or 0)
            for transfer in d2h_transfers
        ),
        "h2d_transfer_bytes": sum(
            int(getattr(transfer, "bytes_transferred", 0) or 0)
            for transfer in h2d_transfers
        ),
        "d2h_transfer_seconds": sum(
            float(getattr(transfer, "elapsed_seconds", 0.0) or 0.0)
            for transfer in d2h_transfers
        ),
        "h2d_transfer_seconds": sum(
            float(getattr(transfer, "elapsed_seconds", 0.0) or 0.0)
            for transfer in h2d_transfers
        ),
        "runtime_d2h_transfers": len(runtime_d2h),
        "runtime_h2d_transfers": len(runtime_h2d),
        "runtime_d2h_transfer_bytes": sum(
            int(getattr(transfer, "bytes_transferred", 0) or 0)
            for transfer in runtime_d2h
        ),
        "runtime_h2d_transfer_bytes": sum(
            int(getattr(transfer, "bytes_transferred", 0) or 0)
            for transfer in runtime_h2d
        ),
        "runtime_d2h_transfer_seconds": sum(
            float(getattr(transfer, "elapsed_seconds", 0.0) or 0.0)
            for transfer in runtime_d2h
        ),
        "runtime_h2d_transfer_seconds": sum(
            float(getattr(transfer, "elapsed_seconds", 0.0) or 0.0)
            for transfer in runtime_h2d
        ),
    }


def _read_d2h_transfer_stats(get_d2h_transfer_stats):
    if get_d2h_transfer_stats is None:
        return None, None, None
    stats = get_d2h_transfer_stats()
    if len(stats) >= 3:
        count, bytes_transferred, seconds = stats[:3]
    else:
        count, bytes_transferred = stats
        seconds = 0.0
    return int(count), int(bytes_transferred), float(seconds)


def _execute_profiled_timed_sections(
    script_path,
    sections,
    globals_dict,
    trace,
    *,
    get_fallback_events,
    get_materialization_events,
    materialization_context,
    get_d2h_transfer_stats,
):
    _preamble_code, timed_code, _postamble_code, timed_source, timed_line_offset = sections
    statements = _compile_timed_statements(script_path, timed_source, timed_line_offset)
    if not statements:
        start = time.perf_counter()
        exec(timed_code, globals_dict)
        return time.perf_counter() - start, []

    stages = []
    total_start = time.perf_counter()
    for stmt in statements:
        before_steps = len(getattr(trace, "steps", []))
        before_transfers = len(getattr(trace, "transfers", []))
        before_fallbacks = len(get_fallback_events(clear=False))
        before_materializations = len(get_materialization_events(clear=False))
        before_d2h, before_d2h_bytes, before_d2h_seconds = _read_d2h_transfer_stats(
            get_d2h_transfer_stats
        )
        start = time.perf_counter()
        with materialization_context(
            pipeline="shootout",
            dataset=os.path.basename(script_path),
            stage=f"statement_{stmt['index']}",
            stage_category=",".join(stmt["tags"]),
        ):
            exec(stmt["code"], globals_dict)
        elapsed = time.perf_counter() - start
        after_fallbacks = len(get_fallback_events(clear=False))
        materialization_events = get_materialization_events(clear=False)
        stage_materialization_events = materialization_events[before_materializations:]
        after_d2h, after_d2h_bytes, after_d2h_seconds = _read_d2h_transfer_stats(
            get_d2h_transfer_stats
        )
        stage = {
            key: value
            for key, value in stmt.items()
            if key != "code"
        }
        stage.update(_profile_execution_deltas(trace, before_steps, before_transfers))
        stage["elapsed_seconds"] = elapsed
        stage["fallback_event_count"] = max(after_fallbacks - before_fallbacks, 0)
        stage["materialization_event_count"] = len(stage_materialization_events)
        stage["materialization_d2h_event_count"] = sum(
            1 for event in stage_materialization_events if getattr(event, "d2h_transfer", False)
        )
        if stage_materialization_events:
            stage["materialization_events"] = [
                _as_dict_event(event) for event in stage_materialization_events[:10]
            ]
        if before_d2h is not None and after_d2h is not None:
            stage["d2h_transfer_count_delta"] = max(after_d2h - before_d2h, 0)
            stage["runtime_d2h_transfer_count_delta"] = stage["d2h_transfer_count_delta"]
        if before_d2h_bytes is not None and after_d2h_bytes is not None:
            stage["d2h_transfer_bytes_delta"] = max(
                after_d2h_bytes - before_d2h_bytes, 0
            )
            stage["runtime_d2h_transfer_bytes_delta"] = stage["d2h_transfer_bytes_delta"]
        if before_d2h_seconds is not None and after_d2h_seconds is not None:
            stage["d2h_transfer_seconds_delta"] = max(
                after_d2h_seconds - before_d2h_seconds, 0.0
            )
            stage["runtime_d2h_transfer_seconds_delta"] = stage["d2h_transfer_seconds_delta"]
        stages.append(stage)
    return time.perf_counter() - total_start, stages


def _profile_timed_sections(script_path, sections):
    profile = {
        "available": False,
        "backend": "vibespatial",
        "mode": "post_timing_profile",
    }
    old_stdout = sys.stdout
    old_hotpath = os.environ.get("VIBESPATIAL_HOTPATH_TRACE")
    try:
        from vibespatial.bench.pipeline import _OwnedAudit
        from vibespatial.runtime.execution_trace import execution_trace
        from vibespatial.runtime.fallbacks import clear_fallback_events, get_fallback_events
        from vibespatial.runtime.hotpath_trace import reset_hotpath_trace, summarize_hotpath_trace
        from vibespatial.runtime.materialization import (
            clear_materialization_events,
            get_materialization_events,
            materialization_context,
        )
        try:
            from vibespatial.cuda._runtime import get_d2h_transfer_profile as get_d2h_transfer_stats, reset_d2h_transfer_count
        except Exception:
            try:
                from vibespatial.cuda._runtime import get_d2h_transfer_stats, reset_d2h_transfer_count
            except Exception:
                get_d2h_transfer_stats = None
                reset_d2h_transfer_count = None

        os.environ["VIBESPATIAL_HOTPATH_TRACE"] = "1"
        clear_fallback_events()
        clear_materialization_events()
        reset_hotpath_trace()
        if reset_d2h_transfer_count is not None:
            reset_d2h_transfer_count()

        audit = _OwnedAudit()
        globals_dict = {
            "__name__": "__profile__",
            "__file__": script_path,
            "__package__": None,
            "__cached__": None,
        }
        capture = io.StringIO()
        sys.stdout = capture
        error = None
        trace = None
        timed_stages = []
        try:
            if sections is None:
                start = time.perf_counter()
                with execution_trace("shootout") as trace:
                    globals_dict = runpy.run_path(script_path, run_name="__profile__")
                elapsed = time.perf_counter() - start
            else:
                preamble_code, _timed_code, _postamble_code, _timed_source, _timed_line_offset = sections
                exec(preamble_code, globals_dict)
                clear_fallback_events()
                clear_materialization_events()
                reset_hotpath_trace()
                if reset_d2h_transfer_count is not None:
                    reset_d2h_transfer_count()
                with execution_trace("shootout") as trace:
                    elapsed, timed_stages = _execute_profiled_timed_sections(
                        script_path,
                        sections,
                        globals_dict,
                        trace,
                        get_fallback_events=get_fallback_events,
                        get_materialization_events=get_materialization_events,
                        materialization_context=materialization_context,
                        get_d2h_transfer_stats=get_d2h_transfer_stats,
                    )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            elapsed = 0.0
        audit.observe(*globals_dict.values())
        fallback_events = [_as_dict_event(event) for event in get_fallback_events(clear=True)]
        hotpath_summary = summarize_hotpath_trace()[:40]
        hotpath_total_seconds = sum(float(stage["elapsed_seconds"]) for stage in hotpath_summary)
        trace_summary = trace.summary() if trace is not None else {}
        trace_steps = [_trace_step_to_dict(step) for step in getattr(trace, "steps", [])[:80]]
        trace_transfers = [
            _trace_transfer_to_dict(transfer)
            for transfer in getattr(trace, "transfers", [])[:80]
        ]
        d2h_transfer_count, d2h_transfer_bytes, d2h_transfer_seconds = _read_d2h_transfer_stats(
            get_d2h_transfer_stats
        )
        profile.update({
            "available": True,
            "elapsed_seconds": elapsed,
            "stdout": capture.getvalue(),
            "fallback_event_count": len(fallback_events),
            "fallback_events": fallback_events[:40],
            "trace_summary": trace_summary,
            "trace_steps": trace_steps,
            "trace_transfers": trace_transfers,
            "hotpath": hotpath_summary,
            "top_hotpath": hotpath_summary[:10],
            "transfer_count": (
                d2h_transfer_count
                if d2h_transfer_count is not None
                else audit.transfer_count
            ),
            "owned_transfer_count": audit.transfer_count,
            "materialization_count": audit.materialization_count,
            "transfer_seconds": audit.transfer_seconds,
            "owned_transfer_seconds": audit.transfer_seconds,
            "runtime_d2h_transfer_seconds": (
                d2h_transfer_seconds
                if d2h_transfer_seconds is not None
                else 0.0
            ),
            "transfer_bytes": (
                d2h_transfer_bytes
                if d2h_transfer_bytes is not None
                else audit.transfer_bytes
            ),
            "owned_transfer_bytes": audit.transfer_bytes,
            "hotpath_total_seconds": hotpath_total_seconds,
            "composition_overhead_seconds": max(elapsed - hotpath_total_seconds, 0.0),
            "composition_overhead_ratio": (
                max(elapsed - hotpath_total_seconds, 0.0) / elapsed
                if elapsed > 0
                else None
            ),
        })
        if timed_stages:
            profile["timed_stages"] = timed_stages[:80]
            profile.update(_summarize_statement_profile(timed_stages))
        if d2h_transfer_count is not None:
            profile["d2h_transfer_count"] = d2h_transfer_count
            profile["runtime_d2h_transfer_count"] = d2h_transfer_count
        if d2h_transfer_bytes is not None:
            profile["runtime_d2h_transfer_bytes"] = d2h_transfer_bytes
        if d2h_transfer_seconds is not None:
            profile["transfer_seconds"] = d2h_transfer_seconds
            profile["runtime_d2h_transfer_seconds"] = d2h_transfer_seconds
        if error is not None:
            profile["error"] = error
    except Exception as exc:
        profile["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        sys.stdout = old_stdout
        if old_hotpath is None:
            os.environ.pop("VIBESPATIAL_HOTPATH_TRACE", None)
        else:
            os.environ["VIBESPATIAL_HOTPATH_TRACE"] = old_hotpath
    return profile

os.chdir(os.path.dirname(os.path.abspath(script)))
sections = _load_script_sections(script)

if do_pipeline_warm:
    try:
        import geopandas as _shootout_gpd  # noqa: F401
        from vibespatial.cuda.cccl_precompile import precompile_all
        precompile_all(timeout=float(dump_after))
    except Exception:
        pass

if do_warmup:
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        if sections is None:
            runpy.run_path(script, run_name="__warmup__")
        else:
            _run_timed_sections(script, sections, run_name="__warmup__")
    except Exception:
        pass
    finally:
        sys.stdout = old

samples = []
captured_stdout = ""
for i in range(repeat):
    old = sys.stdout
    sys.stdout = capture = io.StringIO()
    error = None
    try:
        if sections is None:
            start = time.perf_counter()
            runpy.run_path(script, run_name="__main__")
            elapsed = time.perf_counter() - start
        else:
            elapsed = _run_timed_sections(script, sections, run_name="__main__")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        elapsed = 0.0
    sys.stdout = old
    if i == 0:
        captured_stdout = capture.getvalue()
    samples.append({"elapsed": elapsed, "error": error})

profile = None
if do_pipeline_warm and do_profile:
    profile = _profile_timed_sections(script, sections)

with open(result_path, "w") as f:
    json.dump({"samples": samples, "stdout": captured_stdout, "profile": profile}, f)
"""


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShootoutRun:
    """Timing result from one backend of a shootout."""

    label: str
    timing: TimingSummary
    error: str | None = None
    stdout: str = ""
    profile: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "label": self.label,
            "timing": self.timing.to_dict(),
        }
        if self.error:
            d["error"] = self.error
        if self.stdout:
            d["stdout"] = self.stdout
        if self.profile is not None:
            d["profile"] = self.profile
        return d


@dataclass(frozen=True)
class ShootoutResult:
    """Comparison result from a shootout."""

    script: str
    geopandas: ShootoutRun
    vibespatial: ShootoutRun
    speedup: float | None
    status: str  # "pass" or "error"
    status_reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "type": "shootout",
            "script": self.script,
            "status": self.status,
            "status_reason": self.status_reason,
            "geopandas": self.geopandas.to_dict(),
            "vibespatial": self.vibespatial.to_dict(),
            "speedup": self.speedup,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        import orjson

        return orjson.dumps(self.to_dict(), option=orjson.OPT_INDENT_2).decode()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_uv() -> str | None:
    return shutil.which("uv")


def _fmt_time(seconds: float) -> str:
    if seconds <= 0:
        return "0s"
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f}us"
    if seconds < 1.0:
        return f"{seconds * 1_000:.1f}ms"
    return f"{seconds:.2f}s"


def _build_run_from_result(label: str, raw: dict[str, Any]) -> ShootoutRun:
    """Build a ShootoutRun from harness JSON output."""
    samples_raw = raw.get("samples", [])
    times = [s["elapsed"] for s in samples_raw]
    errors = [s["error"] for s in samples_raw if s.get("error")]

    return ShootoutRun(
        label=label,
        timing=timing_from_samples(times),
        error=errors[0] if errors else None,
        stdout=raw.get("stdout", ""),
        profile=raw.get("profile"),
    )


def _error_run(label: str, error: str) -> ShootoutRun:
    return ShootoutRun(
        label=label,
        timing=timing_from_samples([]),
        error=error,
    )


_FINGERPRINT_PREFIX = "SHOOTOUT_FINGERPRINT: "
_NESTED_GPU_LAUNCH_ERRORS = (
    "GPU execution was requested, but no GPU runtime is available",
    "GPU WKB encode unavailable",
    "strict native mode disallows geopandas fallback: geopandas.read_parquet :: explicit CPU fallback for GeoParquet scan backend selection",
)


def _extract_fingerprint(stdout: str) -> str | None:
    """Extract the SHOOTOUT_FINGERPRINT line from captured stdout."""
    for line in stdout.splitlines():
        if line.startswith(_FINGERPRINT_PREFIX):
            return line[len(_FINGERPRINT_PREFIX):].strip()
    return None


def _load_script_sections(script: Path) -> tuple[object, object, object, str, int] | None:
    """Split a shootout script into preamble, timed body, and postamble."""
    text = script.read_text(encoding="utf-8")
    start = text.find(_TIMED_START_MARKER)
    end = text.find(_TIMED_END_MARKER)
    if start == -1 or end == -1 or end < start:
        return None
    timed_start = start + len(_TIMED_START_MARKER)
    timed_source = text[timed_start:end]
    timed_line_offset = text[:timed_start].count("\n") + 1
    return (
        compile(text[:start], str(script), "exec"),
        compile(timed_source, str(script), "exec"),
        compile(text[end + len(_TIMED_END_MARKER):], str(script), "exec"),
        timed_source,
        timed_line_offset,
    )


def _run_timed_script_sections(
    script: Path,
    sections: tuple[object, object, object, str, int],
    *,
    run_name: str,
) -> tuple[float, str]:
    """Run a marker-delimited script and time only its body."""
    preamble_code, timed_code, postamble_code, _timed_source, _timed_line_offset = sections
    original_stdout = sys.stdout
    capture = io.StringIO()
    globals_dict = {
        "__name__": run_name,
        "__file__": str(script),
        "__package__": None,
        "__cached__": None,
    }
    try:
        sys.stdout = capture
        exec(preamble_code, globals_dict)
        start = time.perf_counter()
        exec(timed_code, globals_dict)
        elapsed = time.perf_counter() - start
        strict_native = os.environ.pop("VIBESPATIAL_STRICT_NATIVE", None)
        try:
            exec(postamble_code, globals_dict)
        finally:
            if strict_native is not None:
                os.environ["VIBESPATIAL_STRICT_NATIVE"] = strict_native
    finally:
        sys.stdout = original_stdout
    return elapsed, capture.getvalue()


_FP_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _fingerprints_match(fp_a: str, fp_b: str, *, rtol: float = 1e-3) -> bool:
    """Compare two fingerprint strings with numeric tolerance.

    Extracts all numbers from each string and compares them pairwise
    with relative tolerance *rtol*.  Non-numeric tokens must match
    exactly.
    """
    if fp_a == fp_b:
        return True
    nums_a = _FP_NUMBER_RE.findall(fp_a)
    nums_b = _FP_NUMBER_RE.findall(fp_b)
    if len(nums_a) != len(nums_b):
        return False
    # Check non-numeric skeleton matches
    skel_a = _FP_NUMBER_RE.sub("@", fp_a)
    skel_b = _FP_NUMBER_RE.sub("@", fp_b)
    if skel_a != skel_b:
        return False
    for sa, sb in zip(nums_a, nums_b):
        va, vb = float(sa), float(sb)
        if va == vb:
            continue
        denom = max(abs(va), abs(vb), 1e-15)
        if abs(va - vb) / denom > rtol:
            return False
    return True


def _infer_physical_shapes(script: Path) -> list[str]:
    """Best-effort physical-plan tags for public shootout scripts."""
    try:
        text = script.read_text(encoding="utf-8")
    except OSError:
        return []

    tags: list[str] = []

    def add(tag: str) -> None:
        if tag not in tags:
            tags.append(tag)

    if "gpd.sjoin(" in text and ".index.unique()" in text:
        add("semijoin")
    if "gpd.sjoin(" in text and ".index.isin(" in text and "~" in text:
        add("anti_semijoin")
    if "gpd.overlay(" in text:
        add("many_few_overlay")
    if "gpd.clip(" in text and (".buffer(" in text or ".dissolve(" in text):
        add("mask_clip")
    if ".dissolve(" in text:
        add("grouped_geometry_reduce")
    if ".area" in text and "gpd.overlay(" in text:
        add("area_filter_after_overlay")
    if ".buffer(" in text and ".dissolve(" in text:
        add("mask_construction")
    return tags


# ---------------------------------------------------------------------------
# Harness runner
# ---------------------------------------------------------------------------

def _run_harness(
    *,
    label: str,
    python_cmd: list[str],
    script: Path,
    repeat: int,
    warmup: bool,
    pipeline_warm: bool = False,
    env: dict[str, str] | None = None,
    timeout: int = 300,
    quiet: bool = False,
    profile: bool = False,
) -> ShootoutRun:
    """Run the timing harness in a subprocess and return results."""
    harness_fd, harness_path = tempfile.mkstemp(suffix=".py", prefix="vsbench_harness_")
    result_fd, result_path = tempfile.mkstemp(suffix=".json", prefix="vsbench_result_")
    os.close(result_fd)
    try:
        with os.fdopen(harness_fd, "w") as f:
            f.write(_HARNESS_CODE)

        cmd = [
            *python_cmd,
            harness_path,
            str(script.resolve()),
            result_path,
            str(repeat),
            "1" if warmup else "0",
            "1" if pipeline_warm else "0",
            str(max(timeout - 5, 1)),
            "1" if profile else "0",
        ]

        if not quiet:
            print(
                f"  Running with {label}...",
                end="",
                flush=True,
                file=sys.stderr,
            )

        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        result_file = Path(result_path)
        if not result_file.exists() or result_file.stat().st_size == 0:
            msg = proc.stderr.strip() or proc.stdout.strip() or "harness produced no output"
            if not quiet:
                print(" ERROR", file=sys.stderr)
            return _error_run(label, msg)

        raw = json.loads(result_file.read_text())
        run = _build_run_from_result(label, raw)

        if not quiet:
            if run.error:
                print(f" ERROR ({run.error})", file=sys.stderr)
            else:
                print(f" {_fmt_time(run.timing.median_seconds)}", file=sys.stderr)

        return run

    except subprocess.TimeoutExpired as exc:
        if not quiet:
            print(" TIMEOUT", file=sys.stderr)
        detail = (exc.stderr or exc.stdout or "").strip()
        if detail:
            return _error_run(
                label,
                f"timeout ({timeout}s limit exceeded)\n{detail}",
            )
        return _error_run(label, f"timeout ({timeout}s limit exceeded)")

    except Exception as exc:
        if not quiet:
            print(f" ERROR ({exc})", file=sys.stderr)
        return _error_run(label, str(exc))

    finally:
        for p in (harness_path, result_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def _run_harness_in_process(
    *,
    label: str,
    script: Path,
    repeat: int,
    warmup: bool,
    pipeline_warm: bool = False,
    env: dict[str, str] | None = None,
    profile: bool = False,
) -> ShootoutRun:
    """Run the timing harness in-process.

    Used only as a recovery path when a nested subprocess launch loses GPU
    runtime visibility despite the current parent process having a working
    GPU runtime.
    """
    original_stdout = sys.stdout
    original_cwd = Path.cwd()
    original_path = list(sys.path)
    original_env = os.environ.copy()

    if env is not None:
        os.environ.clear()
        os.environ.update(env)

    os.chdir(os.path.dirname(os.path.abspath(script)))
    sections = _load_script_sections(script)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            if pipeline_warm:
                try:
                    import geopandas as _shootout_gpd  # noqa: F401
                    from vibespatial.cuda.cccl_precompile import precompile_all

                    precompile_all(timeout=120.0)
                except Exception:
                    pass

            if warmup:
                try:
                    if sections is None:
                        sys.stdout = io.StringIO()
                        runpy.run_path(script, run_name="__warmup__")
                    else:
                        _run_timed_script_sections(
                            script,
                            sections,
                            run_name="__warmup__",
                        )
                except Exception:
                    pass
                finally:
                    sys.stdout = original_stdout

            samples: list[float] = []
            errors: list[str] = []
            captured_stdout = ""
            for i in range(repeat):
                error = None
                try:
                    if sections is None:
                        sys.stdout = capture = io.StringIO()
                        start = time.perf_counter()
                        runpy.run_path(script, run_name="__main__")
                        elapsed = time.perf_counter() - start
                        stdout = capture.getvalue()
                    else:
                        elapsed, stdout = _run_timed_script_sections(
                            script,
                            sections,
                            run_name="__main__",
                        )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    elapsed = 0.0
                    stdout = ""
                sys.stdout = original_stdout
                if i == 0:
                    captured_stdout = stdout
                samples.append(elapsed)
                if error is not None:
                    errors.append(error)

            return ShootoutRun(
                label=label,
                timing=timing_from_samples(samples),
                error=errors[0] if errors else None,
                stdout=captured_stdout,
            )
    finally:
        sys.stdout = original_stdout
        sys.path[:] = original_path
        os.chdir(original_cwd)
        os.environ.clear()
        os.environ.update(original_env)


def _should_retry_vibespatial_in_process(run: ShootoutRun) -> bool:
    error = run.error or ""
    return any(marker in error for marker in _NESTED_GPU_LAUNCH_ERRORS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_shootout(
    script_path: Path,
    *,
    repeat: int = 3,
    warmup: bool = True,
    baseline_python: str | None = None,
    extra_deps: list[str] | None = None,
    timeout: int = 300,
    quiet: bool = False,
    scale: str | None = None,
    profile: bool = False,
) -> ShootoutResult:
    """Run a user script with geopandas and vibespatial, return comparison."""
    from vibespatial.runtime import has_gpu_runtime

    script = Path(script_path)
    if not script.is_file():
        raise FileNotFoundError(f"Script not found: {script}")

    if not quiet:
        print(f"\nvsbench shootout: {script.name}", file=sys.stderr)
        print(
            f"  repeat={repeat}, warmup={'yes' if warmup else 'no'}",
            file=sys.stderr,
        )

    # --- geopandas baseline command ---
    if baseline_python:
        gpd_cmd: list[str] = [baseline_python]
    else:
        uv = _find_uv()
        if uv is None:
            raise RuntimeError(
                "uv not found. Install uv or use --baseline-python to "
                "specify a Python interpreter with geopandas installed."
            )
        deps = ["geopandas", "pyarrow"]
        if extra_deps:
            deps.extend(dep for dep in extra_deps if dep not in deps)
        with_args: list[str] = []
        for dep in deps:
            with_args.extend(["--with", dep])
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        gpd_cmd = [
            uv, "run", "--isolated", "--no-project",
            "--python", py_ver,
            *with_args,
            "--", "python",
        ]

    gpd_env = os.environ.copy()
    gpd_env.pop("_VIBESPATIAL_GEOPANDAS_COMPAT", None)
    if scale is not None:
        gpd_env["VSBENCH_SCALE"] = scale

    gpd_run = _run_harness(
        label="geopandas",
        python_cmd=gpd_cmd,
        script=script,
        repeat=repeat,
        warmup=warmup,
        env=gpd_env,
        timeout=timeout,
        quiet=quiet,
        profile=False,
    )

    # --- vibespatial ---
    # The repo-local geopandas shim is already imported from src/ without the
    # compat flag, so keep the vibespatial subprocess environment as close to
    # the baseline as possible. The compat flag only suppresses a deprecation
    # warning, and on this machine it also perturbs CUDA visibility in the
    # shootout subprocess.
    vs_env = os.environ.copy()
    if scale is not None:
        vs_env["VSBENCH_SCALE"] = scale

    vs_run = _run_harness(
        label="vibespatial",
        python_cmd=[sys.executable],
        script=script,
        repeat=repeat,
        warmup=warmup,
        pipeline_warm=True,
        env=vs_env,
        timeout=timeout,
        quiet=quiet,
        profile=profile,
    )
    launch_mode = "subprocess"
    if (
        vs_run.error is not None
        and has_gpu_runtime()
        and _should_retry_vibespatial_in_process(vs_run)
    ):
        vs_run = _run_harness_in_process(
            label="vibespatial",
            script=script,
            repeat=repeat,
            warmup=warmup,
            pipeline_warm=True,
            env=vs_env,
            profile=profile,
        )
        launch_mode = "in_process_retry"

    # --- comparison ---
    speedup = None
    if gpd_run.timing.median_seconds > 0 and vs_run.timing.median_seconds > 0:
        speedup = gpd_run.timing.median_seconds / vs_run.timing.median_seconds

    has_error = bool(gpd_run.error or vs_run.error)
    errors: list[str] = []
    if gpd_run.error:
        errors.append(f"geopandas: {gpd_run.error}")
    if vs_run.error:
        errors.append(f"vibespatial: {vs_run.error}")

    # --- fingerprint comparison ---
    gpd_fp = _extract_fingerprint(gpd_run.stdout)
    vs_fp = _extract_fingerprint(vs_run.stdout)
    fingerprint_match: str | None = None
    if gpd_fp and vs_fp:
        if _fingerprints_match(gpd_fp, vs_fp):
            fingerprint_match = "match"
        else:
            fingerprint_match = "mismatch"
            if not has_error:
                errors.append(f"fingerprint mismatch: geopandas={gpd_fp} vibespatial={vs_fp}")
                has_error = True

    meta: dict[str, Any] = {
        "repeat": repeat,
        "warmup": warmup,
        "physical_shapes": _infer_physical_shapes(script),
    }
    if not warmup and repeat < 3:
        meta["measurement_mode"] = "cold_start_probe"
        meta["measurement_note"] = (
            "repeat<3 with warmup disabled is cold-start sensitive; "
            "use warmup or repeat>=3 for steady-state parity comparisons"
        )
    if scale is not None:
        meta["scale"] = scale
    meta["vibespatial_launch"] = launch_mode
    if fingerprint_match:
        meta["fingerprint"] = fingerprint_match
        meta["fingerprint_geopandas"] = gpd_fp
        meta["fingerprint_vibespatial"] = vs_fp

    return ShootoutResult(
        script=str(script),
        geopandas=gpd_run,
        vibespatial=vs_run,
        speedup=speedup,
        status="error" if has_error else "pass",
        status_reason="; ".join(errors) if errors else "ok",
        metadata=meta,
    )
