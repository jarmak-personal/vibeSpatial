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
    return (
        compile(text[:start], script_path, "exec"),
        compile(text[start + len(TIMED_START_MARKER):end], script_path, "exec"),
        compile(text[end + len(TIMED_END_MARKER):], script_path, "exec"),
    )


def _run_timed_sections(script_path, sections, *, run_name):
    preamble_code, timed_code, postamble_code = sections
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
        if self.stdout:
            d["stdout"] = self.stdout
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


def _load_script_sections(script: Path) -> tuple[object, object, object] | None:
    """Split a shootout script into preamble, timed body, and postamble."""
    text = script.read_text(encoding="utf-8")
    start = text.find(_TIMED_START_MARKER)
    end = text.find(_TIMED_END_MARKER)
    if start == -1 or end == -1 or end < start:
        return None
    return (
        compile(text[:start], str(script), "exec"),
        compile(text[start + len(_TIMED_START_MARKER):end], str(script), "exec"),
        compile(text[end + len(_TIMED_END_MARKER):], str(script), "exec"),
    )


def _run_timed_script_sections(
    script: Path,
    sections: tuple[object, object, object],
    *,
    run_name: str,
) -> tuple[float, str]:
    """Run a marker-delimited script and time only its body."""
    preamble_code, timed_code, postamble_code = sections
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

    meta: dict[str, Any] = {"repeat": repeat, "warmup": warmup}
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
