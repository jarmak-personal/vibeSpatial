"""vsbench shootout — run a user script with geopandas then vibespatial."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import TimingSummary, timing_from_samples

# ---------------------------------------------------------------------------
# Harness script — executed inside the subprocess
# ---------------------------------------------------------------------------

_HARNESS_CODE = """\
import io
import json
import os
import runpy
import sys
import time
import warnings

warnings.filterwarnings("ignore")

script = sys.argv[1]
result_path = sys.argv[2]
repeat = int(sys.argv[3])
do_warmup = sys.argv[4] == "1"

os.chdir(os.path.dirname(os.path.abspath(script)))

if do_warmup:
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        runpy.run_path(script, run_name="__warmup__")
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
    start = time.perf_counter()
    try:
        runpy.run_path(script, run_name="__main__")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - start
    sys.stdout = old
    if i == 0:
        captured_stdout = capture.getvalue()
    samples.append({"elapsed": elapsed, "error": error})

with open(result_path, "w") as f:
    json.dump({"samples": samples, "stdout": captured_stdout}, f)
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

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "label": self.label,
            "timing": self.timing.to_dict(),
        }
        if self.error:
            d["error"] = self.error
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
    )


def _error_run(label: str, error: str) -> ShootoutRun:
    return ShootoutRun(
        label=label,
        timing=timing_from_samples([]),
        error=error,
    )


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
    env: dict[str, str] | None = None,
    timeout: int = 300,
    quiet: bool = False,
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

    except subprocess.TimeoutExpired:
        if not quiet:
            print(" TIMEOUT", file=sys.stderr)
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
) -> ShootoutResult:
    """Run a user script with geopandas and vibespatial, return comparison."""
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
        deps = ["geopandas"]
        if extra_deps:
            deps.extend(extra_deps)
        with_args: list[str] = []
        for dep in deps:
            with_args.extend(["--with", dep])
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        gpd_cmd = [
            uv, "run", "--no-project",
            "--python", py_ver,
            *with_args,
            "--", "python",
        ]

    gpd_env = os.environ.copy()
    gpd_env.pop("_VIBESPATIAL_GEOPANDAS_COMPAT", None)

    gpd_run = _run_harness(
        label="geopandas",
        python_cmd=gpd_cmd,
        script=script,
        repeat=repeat,
        warmup=warmup,
        env=gpd_env,
        timeout=timeout,
        quiet=quiet,
    )

    # --- vibespatial ---
    vs_env = os.environ.copy()
    vs_env["_VIBESPATIAL_GEOPANDAS_COMPAT"] = "1"

    vs_run = _run_harness(
        label="vibespatial",
        python_cmd=[sys.executable],
        script=script,
        repeat=repeat,
        warmup=warmup,
        env=vs_env,
        timeout=timeout,
        quiet=quiet,
    )

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

    return ShootoutResult(
        script=str(script),
        geopandas=gpd_run,
        vibespatial=vs_run,
        speedup=speedup,
        status="error" if has_error else "pass",
        status_reason="; ".join(errors) if errors else "ok",
        metadata={"repeat": repeat, "warmup": warmup},
    )
