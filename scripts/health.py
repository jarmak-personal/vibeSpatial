from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import tempfile
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    from .check_docs import check_docs
except ImportError:
    from check_docs import check_docs

try:
    from .gpu_acceleration_coverage import run_gpu_acceleration_coverage
except ImportError:
    from gpu_acceleration_coverage import run_gpu_acceleration_coverage

try:
    from . import property_dashboard
except ImportError:
    import property_dashboard

try:
    from vibespatial.runtime import has_gpu_runtime
except ImportError:
    from vibespatial import has_gpu_runtime


HEALTH_TEST_SUITE = [
    "tests/test_runtime_harness.py",
    "tests/test_geopandas_shim.py",
    "tests/test_version_consistency.py",
    "tests/test_reference_oracle.py",
    "tests/test_kernel_scaffold.py",
    "tests/test_decision_log.py",
]

SURFACE_MATRIX_PATH = Path("scripts/health_surfaces.toml")
BASELINE_PATH = Path(".health-baseline.json")


@dataclass(frozen=True)
class HealthSurface:
    name: str
    owners: tuple[str, ...]
    command: str
    required: bool
    runtime: str
    allowed_states: tuple[str, ...]
    baseline_key: str | None = None


@dataclass
class CommandStatus:
    ok: bool
    command: str
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    details: dict[str, Any] = field(default_factory=dict)


def run_command(command: list[str], *, cwd: Path, timeout: int = 30) -> CommandStatus:
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return CommandStatus(
        ok=completed.returncode == 0,
        command=" ".join(command),
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )


def parse_pytest_summary(output: str) -> dict[str, Any]:
    counts = {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0, "xpassed": 0}
    for key in counts:
        matches = re.findall(rf"(\d+) {key}", output)
        if matches:
            counts[key] = max(int(value) for value in matches)
    broken = re.findall(r"^FAILED\s+(.+)$", output, flags=re.MULTILINE)
    return {"counts": counts, "broken": broken}


def collect_coverage_and_tests(root: Path) -> tuple[dict[str, Any], dict[str, Any], CommandStatus]:
    with tempfile.TemporaryDirectory() as temp_dir:
        coverage_path = Path(temp_dir) / "coverage.json"
        command = [
            "uv",
            "run",
            "pytest",
            "-q",
            "--cov=src/vibespatial",
            "--cov=src/geopandas",
            f"--cov-report=json:{coverage_path}",
            *HEALTH_TEST_SUITE,
        ]
        status = run_command(command, cwd=root, timeout=60)
        tests = parse_pytest_summary(status.stdout + "\n" + status.stderr)
        coverage = {"percent": 0.0, "covered_lines": 0, "num_statements": 0}
        if coverage_path.exists():
            payload = json.loads(coverage_path.read_text(encoding="utf-8"))
            totals = payload.get("totals", {})
            coverage = {
                "percent": float(totals.get("percent_covered", 0.0)),
                "covered_lines": int(totals.get("covered_lines", 0)),
                "num_statements": int(totals.get("num_statements", 0)),
            }
        return coverage, tests, status


def collect_lint(root: Path) -> dict[str, Any]:
    ruff = run_command(
        [
            "uv",
            "run",
            "ruff",
            "check",
            "src/vibespatial",
            "src/geopandas",
            "scripts",
            "tests",
            "--exclude",
            "src/vibespatial/_vendor,tests/upstream",
        ],
        cwd=root,
        timeout=60,
    )
    arch = run_command(["uv", "run", "python", "scripts/check_architecture_lints.py", "--all"], cwd=root, timeout=60)
    return {
        "ok": ruff.ok and arch.ok,
        "ruff": asdict(ruff),
        "architecture": asdict(arch),
        "violations": (0 if ruff.ok else 1) + (0 if arch.ok else 1),
    }


def collect_docs(root: Path) -> dict[str, Any]:
    errors = check_docs(root)
    return {"ok": not errors, "errors": errors}


def collect_gpu_acceleration(root: Path, *, include: bool, timeout: int) -> dict[str, Any]:
    if not include:
        return {
            "ok": True,
            "available": False,
            "reason": "run `uv run python scripts/health.py --gpu-coverage` to collect upstream GPU acceleration coverage",
        }
    report = run_gpu_acceleration_coverage(("tests/upstream/geopandas",), cwd=root, timeout=timeout)
    return {
        "ok": report.observed_dispatch.returncode == 0 and report.api_compat.returncode in {0, 1},
        "available": True,
        "gpu_available": report.gpu_available,
        "api_compat_pct": report.api_compat_pct,
        "api_suite_pct": report.api_suite_pct,
        "gpu_accel_pct": report.gpu_accel_pct,
        "raw_dispatch_pct": report.raw_dispatch_pct,
        "value_dispatch_pct": report.value_dispatch_pct,
        "total_dispatches": report.total_dispatches,
        "gpu_dispatches": report.gpu_dispatches,
        "cpu_dispatches": report.cpu_dispatches,
        "fallback_dispatches": report.fallback_dispatches,
        "deferred_dispatches": report.deferred_dispatches,
        "value_dispatches": report.value_dispatches,
        "value_gpu_dispatches": report.value_gpu_dispatches,
        "weighted_dispatch_units": report.weighted_dispatch_units,
        "weighted_gpu_units": report.weighted_gpu_units,
        "family_breakdown": report.family_breakdown,
        "fallback_reasons": report.fallback_reasons,
        "fallback_surfaces": report.fallback_surfaces,
    }


def collect_property_summary() -> dict[str, Any]:
    props = property_dashboard.snapshot()
    clean = sum(1 for prop in props if prop.status == "ok")
    debt = sum(1 for prop in props if prop.status == "debt")
    regressed = sum(1 for prop in props if prop.status == "regressed")
    status = "regressed" if regressed else "debt" if debt else "ok"
    return {
        "ok": regressed == 0,
        "status": status,
        "clean": clean,
        "debt": debt,
        "regressed": regressed,
        "total": len(props),
        "total_distance": property_dashboard.total_distance(props),
    }


def load_health_surfaces(config_path: Path) -> list[HealthSurface]:
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    surfaces = []
    for entry in payload.get("surface", []):
        surfaces.append(
            HealthSurface(
                name=str(entry["name"]),
                owners=tuple(str(owner) for owner in entry.get("owners", [])),
                command=str(entry["command"]),
                required=bool(entry.get("required", True)),
                runtime=str(entry.get("runtime", "any")),
                allowed_states=tuple(str(state) for state in entry.get("allowed_states", ("pass", "fail"))),
                baseline_key=str(entry["baseline_key"]) if "baseline_key" in entry else None,
            )
        )
    return surfaces


def _pytest_total(counts: dict[str, int]) -> int:
    return counts["passed"] + counts["failed"] + counts["skipped"] + counts["xfailed"] + counts["xpassed"]


def _surface_state_ok(state: str) -> bool:
    return state in {"pass", "unsupported", "skipped-no-gpu"}


def run_health_surface(
    surface: HealthSurface,
    *,
    root: Path,
    timeout: int,
    gpu_available: bool,
) -> dict[str, Any]:
    if surface.runtime in {"gpu-preferred", "gpu-required"} and not gpu_available:
        return {
            "name": surface.name,
            "owners": list(surface.owners),
            "command": surface.command,
            "required": surface.required,
            "runtime": surface.runtime,
            "allowed_states": list(surface.allowed_states),
            "baseline_key": surface.baseline_key,
            "state": "skipped-no-gpu",
            "ok": "skipped-no-gpu" in surface.allowed_states,
            "counts": {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0, "xpassed": 0},
            "passed": 0,
            "total": 0,
            "returncode": 0,
            "failing_tests": [],
            "detail": "visible NVIDIA runtime not available",
        }

    status = run_command(shlex.split(surface.command), cwd=root, timeout=timeout)
    tests = parse_pytest_summary(status.stdout + "\n" + status.stderr)
    counts = tests["counts"]
    total = _pytest_total(counts)

    if status.returncode == 0:
        state = "pass"
    elif status.returncode == 5 and "unsupported" in surface.allowed_states:
        state = "unsupported"
    elif status.returncode == 1:
        state = "fail"
    else:
        state = "broken"

    state_allowed = state in surface.allowed_states or state == "broken"
    detail = f"{counts['passed']}/{total} passed"
    if state == "unsupported":
        detail = "surface not supported in this environment"
    elif state == "broken":
        detail = f"command exited {status.returncode}"

    return {
        "name": surface.name,
        "owners": list(surface.owners),
        "command": surface.command,
        "required": surface.required,
        "runtime": surface.runtime,
        "allowed_states": list(surface.allowed_states),
        "baseline_key": surface.baseline_key,
        "state": state,
        "ok": state_allowed and _surface_state_ok(state),
        "counts": counts,
        "passed": counts["passed"],
        "total": total,
        "returncode": status.returncode,
        "failing_tests": list(tests["broken"]),
        "detail": detail,
    }


def collect_contract_health_report(
    root: Path,
    *,
    config_path: Path | None = None,
    surface_timeout: int = 600,
) -> dict[str, Any]:
    matrix_path = config_path or root / SURFACE_MATRIX_PATH
    surfaces = load_health_surfaces(matrix_path)
    gpu_available = has_gpu_runtime()
    surface_reports = [
        run_health_surface(
            surface,
            root=root,
            timeout=surface_timeout,
            gpu_available=gpu_available,
        )
        for surface in surfaces
    ]
    required_reports = [surface for surface in surface_reports if surface["required"]]
    required_passing = sum(1 for surface in required_reports if surface["ok"])
    ok = required_passing == len(required_reports)
    return {
        "tier": "contract",
        "status": "pass" if ok else "fail",
        "ok": ok,
        "gpu_available": gpu_available,
        "required_total": len(required_reports),
        "required_passing": required_passing,
        "surface_count": len(surface_reports),
        "surfaces": surface_reports,
        "matrix_path": str(matrix_path.relative_to(root)),
        "exit_code": 0 if ok else 1,
    }


def load_health_baseline(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_health_baseline(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def snapshot_report_baseline(report: dict[str, Any]) -> dict[str, Any]:
    tier = report["tier"]
    if tier == "contract":
        return {
            "surfaces": {
                surface["name"]: {
                    "state": surface["state"],
                    "passed": surface["passed"],
                    "total": surface["total"],
                }
                for surface in report["surfaces"]
            }
        }
    if tier == "gpu":
        if report["status"] == "skipped-no-gpu":
            return {"status": "skipped-no-gpu"}
        gpu_acceleration = report["gpu_acceleration"]
        total_dispatches = gpu_acceleration["total_dispatches"]
        fallback_rate = 0.0
        if total_dispatches:
            fallback_rate = gpu_acceleration["fallback_dispatches"] / total_dispatches
        return {
            "status": report["status"],
            "raw_dispatch_pct": gpu_acceleration.get("raw_dispatch_pct", gpu_acceleration["gpu_accel_pct"]),
            "value_dispatch_pct": gpu_acceleration.get("value_dispatch_pct", gpu_acceleration["gpu_accel_pct"]),
            "fallback_rate": round(fallback_rate, 6),
        }
    return {}


def update_baseline_payload(payload: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    baseline = dict(payload)
    baseline["schema_version"] = max(int(baseline.get("schema_version", 1)), 2)
    baseline[report["tier"]] = snapshot_report_baseline(report)
    return baseline


def compare_contract_report_against_baseline(
    report: dict[str, Any],
    baseline: dict[str, Any],
) -> list[str]:
    regressions: list[str] = []
    surface_baselines = baseline.get("surfaces", {})
    for surface in report["surfaces"]:
        base = surface_baselines.get(surface["name"])
        if base is None:
            if surface["required"] and not surface["ok"]:
                regressions.append(f"{surface['name']}: no baseline and current state is {surface['state']}")
            continue

        baseline_state = str(base.get("state", "fail"))
        baseline_passed = int(base.get("passed", 0))
        baseline_total = int(base.get("total", 0))
        baseline_failures = max(baseline_total - baseline_passed, 0)
        current_failures = max(surface["total"] - surface["passed"], 0)

        if surface["state"] == "broken":
            regressions.append(f"{surface['name']}: command broke (exit {surface['returncode']})")
            continue

        if surface["state"] == "skipped-no-gpu" and surface["runtime"] in {"gpu-preferred", "gpu-required"}:
            continue

        if baseline_state == "pass":
            if surface["state"] != "pass":
                regressions.append(f"{surface['name']}: regressed from pass to {surface['state']}")
            continue

        if baseline_state == "fail":
            if surface["state"] == "fail" and current_failures > baseline_failures:
                regressions.append(
                    f"{surface['name']}: failures increased from {baseline_failures} to {current_failures}"
                )
            elif surface["state"] not in {"fail", "pass", "skipped-no-gpu", "unsupported"}:
                regressions.append(f"{surface['name']}: unexpected state {surface['state']}")
            continue

        if baseline_state in {"unsupported", "skipped-no-gpu"} and surface["state"] == "broken":
            regressions.append(f"{surface['name']}: command broke from baseline {baseline_state}")
    return regressions


def compare_gpu_report_against_baseline(report: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    if report["status"] == "skipped-no-gpu":
        return []
    if not baseline:
        return []
    if baseline.get("status") == "skipped-no-gpu":
        return []
    regressions: list[str] = []
    baseline_status = str(baseline.get("status", "fail"))
    if baseline_status == "pass" and report["status"] != "pass":
        regressions.append(f"gpu tier regressed from pass to {report['status']}")
    gpu_acceleration = report["gpu_acceleration"]
    total_dispatches = gpu_acceleration["total_dispatches"]
    fallback_rate = 0.0
    if total_dispatches:
        fallback_rate = gpu_acceleration["fallback_dispatches"] / total_dispatches
    baseline_value_dispatch = float(baseline.get("value_dispatch_pct", baseline.get("gpu_accel_pct", 0.0)))
    current_value_dispatch = float(gpu_acceleration.get("value_dispatch_pct", gpu_acceleration.get("gpu_accel_pct", 0.0)))
    baseline_fallback_rate = float(baseline.get("fallback_rate", 0.0))
    current_fallback_rate = fallback_rate
    # GPU native-coverage is a single long upstream sweep and still exhibits
    # small run-to-run jitter in the weighted public-op percentage. Ignore
    # changes below 0.1 percentage points so the ratchet tracks real movement
    # instead of basis-point noise from one full-suite sample.
    if current_value_dispatch + 0.1 < baseline_value_dispatch:
        regressions.append(
            "value-weighted gpu acceleration regressed "
            f"from {baseline_value_dispatch:.2f}% to {current_value_dispatch:.2f}%"
        )
    if current_fallback_rate > baseline_fallback_rate + 0.0001:
        regressions.append(
            f"fallback rate regressed from {baseline_fallback_rate:.4f} to {current_fallback_rate:.4f}"
        )
    return regressions


def evaluate_check_report(report: dict[str, Any], baseline_payload: dict[str, Any]) -> tuple[int, list[str]]:
    tier = report["tier"]
    if tier == "contract":
        contract_baseline = baseline_payload.get("contract", {})
        if not contract_baseline:
            return report["exit_code"], []
        regressions = compare_contract_report_against_baseline(report, contract_baseline)
        return (1 if regressions else 0), regressions
    if tier == "gpu":
        gpu_baseline = baseline_payload.get("gpu", {})
        if not gpu_baseline:
            return report["exit_code"], []
        regressions = compare_gpu_report_against_baseline(report, gpu_baseline)
        return (1 if regressions else 0), regressions
    return report["exit_code"], []


def collect_health_report(
    root: Path,
    *,
    include_gpu_coverage: bool = False,
    gpu_coverage_timeout: int = 600,
) -> dict[str, Any]:
    coverage, tests, pytest_status = collect_coverage_and_tests(root)
    lint = collect_lint(root)
    docs = collect_docs(root)
    gpu_acceleration = collect_gpu_acceleration(root, include=include_gpu_coverage, timeout=gpu_coverage_timeout)
    properties = collect_property_summary()
    profile_script = root / "scripts" / "profile_kernels.py"
    pipeline_script = root / "scripts" / "benchmark_pipelines.py"
    pipeline_compare_module = root / "src" / "vibespatial" / "bench" / "compare.py"
    pipeline_workflow = root / ".github" / "workflows" / "pipeline-benchmarks.yml"
    profiling_rails = [
        path
        for path in (
            "scripts/profile_kernels.py" if profile_script.exists() else "",
            "scripts/benchmark_pipelines.py" if pipeline_script.exists() else "",
            "src/vibespatial/bench/compare.py" if pipeline_compare_module.exists() else "",
            ".github/workflows/pipeline-benchmarks.yml" if pipeline_workflow.exists() else "",
        )
        if path
    ]
    benchmarks = {
        "status": "configured" if profiling_rails else "unmeasured",
        "available": bool(profiling_rails),
        "regressions": [],
        "profiling_rails": profiling_rails,
        "detail": (
            "profiling rails defined: " + ", ".join(profiling_rails)
            if profiling_rails
            else "profiling rails not detected in this checkout"
        ),
    }
    transfer_audit = {
        "status": "configured",
        "available": True,
        "violations": [],
        "policy_defined": True,
        "detail": "transfer audit policy defined for owned geometry arrays",
    }
    dispatch_policy = {
        "status": "configured",
        "available": True,
        "policy_defined": True,
        "detail": "dispatch policy defined for bootstrap health reporting",
    }
    healthy = pytest_status.ok and lint["ok"] and docs["ok"] and properties["ok"]
    broken = pytest_status.returncode not in {0, 1}
    exit_code = 2 if broken else 0 if healthy else 1
    return {
        "tier": "bootstrap",
        "properties": properties,
        "coverage": coverage,
        "tests": {**tests, "ok": pytest_status.ok, "command": pytest_status.command},
        "lint": lint,
        "docs": docs,
        "benchmarks": benchmarks,
        "dispatch_policy": dispatch_policy,
        "transfer_audit": transfer_audit,
        "gpu_acceleration": gpu_acceleration,
        "exit_code": exit_code,
    }


def _bootstrap_label(exit_code: int) -> str:
    return {0: "PASS", 1: "FAIL", 2: "BROKEN"}.get(exit_code, "UNKNOWN")


def _smoke_suite_total(counts: dict[str, int]) -> int:
    return _pytest_total(counts)


def _format_optional_signal(label: str, signal: dict[str, Any]) -> str:
    detail = signal.get("detail")
    detail_suffix = f" ({detail})" if detail else ""
    return f"  {label}: {signal['status'].upper()}{detail_suffix}"


def print_summary(report: dict[str, Any]) -> None:
    counts = report["tests"]["counts"]
    total = _smoke_suite_total(counts)
    bootstrap_status = _bootstrap_label(report["exit_code"])
    properties = report["properties"]
    print(f"Repo Health — bootstrap: {bootstrap_status}")
    print(
        "Property summary: "
        f"{properties['clean']}/{properties['total']} clean, "
        f"{properties['debt']} debt, "
        f"{properties['regressed']} regressed, "
        f"total distance {properties['total_distance']:.2f}"
    )
    gpu_acceleration = report["gpu_acceleration"]
    if gpu_acceleration.get("available"):
        print(
            "Value-weighted GPU acceleration: "
            f"{gpu_acceleration.get('value_dispatch_pct', gpu_acceleration['gpu_accel_pct']):.2f}% "
            f"(gpu available: {'yes' if gpu_acceleration['gpu_available'] else 'no'}; "
            f"{gpu_acceleration.get('value_gpu_dispatches', gpu_acceleration['gpu_dispatches'])} GPU / "
            f"{gpu_acceleration.get('value_dispatches', gpu_acceleration['total_dispatches'])} tracked dispatches)"
        )
        print(
            "Raw dispatch breadth: "
            f"{gpu_acceleration.get('raw_dispatch_pct', gpu_acceleration['gpu_accel_pct']):.2f}% "
            f"({gpu_acceleration['gpu_dispatches']} GPU / {gpu_acceleration['total_dispatches']} dispatches)"
        )
        if gpu_acceleration.get("deferred_dispatches"):
            print(f"Deferred dispatch records: {gpu_acceleration['deferred_dispatches']}")
    else:
        print(f"GPU acceleration: {gpu_acceleration.get('reason', 'unavailable')}")
    print(
        "Smoke suite: "
        f"{'PASS' if report['tests']['ok'] else 'FAIL'} "
        f"({counts['passed']}/{total} passed)"
    )
    print(f"Lint: {'PASS' if report['lint']['ok'] else 'FAIL'}")
    print(f"Docs: {'PASS' if report['docs']['ok'] else 'FAIL'}")
    print("Additional signals:")
    print(_format_optional_signal("benchmarks", report["benchmarks"]))
    print(_format_optional_signal("transfer_audit", report["transfer_audit"]))
    print(_format_optional_signal("dispatch_policy", report["dispatch_policy"]))
    print(json.dumps(report, indent=2))


def print_contract_summary(report: dict[str, Any]) -> None:
    print(f"Repo Health — contract: {report['status'].upper()}")
    print(f"Surface matrix: {report['matrix_path']}")
    print(f"GPU available: {'yes' if report['gpu_available'] else 'no'}")
    print("Maintained surfaces:")
    for surface in report["surfaces"]:
        required = "required" if surface["required"] else "optional"
        print(
            f"  {surface['name']:<18} {surface['state'].upper():<15} "
            f"{surface['passed']}/{surface['total']} {required}"
            + (f" ({surface['detail']})" if surface.get("detail") else "")
        )
    print(json.dumps(report, indent=2))


def collect_gpu_health_report(
    root: Path,
    *,
    gpu_coverage_timeout: int = 600,
) -> dict[str, Any]:
    gpu_available = has_gpu_runtime()
    if not gpu_available:
        return {
            "tier": "gpu",
            "status": "skipped-no-gpu",
            "ok": True,
            "gpu_available": False,
            "detail": "visible NVIDIA runtime required for gpu tier",
            "exit_code": 0,
        }

    properties = collect_property_summary()
    gpu_acceleration = collect_gpu_acceleration(root, include=True, timeout=gpu_coverage_timeout)
    ok = gpu_acceleration["ok"] and properties["ok"]
    return {
        "tier": "gpu",
        "status": "pass" if ok else "fail",
        "ok": ok,
        "gpu_available": True,
        "properties": properties,
        "gpu_acceleration": gpu_acceleration,
        "exit_code": 0 if ok else 1,
    }


def print_gpu_summary(report: dict[str, Any]) -> None:
    print(f"Repo Health — gpu: {report['status'].upper()}")
    print(f"GPU available: {'yes' if report['gpu_available'] else 'no'}")
    if report["status"] == "skipped-no-gpu":
        print(f"Detail: {report['detail']}")
    else:
        properties = report["properties"]
        gpu_acceleration = report["gpu_acceleration"]
        print(
            "Property summary: "
            f"{properties['clean']}/{properties['total']} clean, "
            f"{properties['regressed']} regressed, "
            f"distance {properties['total_distance']:.2f}"
        )
        print(
            "Value-weighted GPU acceleration: "
            f"{gpu_acceleration.get('value_dispatch_pct', gpu_acceleration['gpu_accel_pct']):.2f}% "
            f"({gpu_acceleration.get('value_gpu_dispatches', gpu_acceleration['gpu_dispatches'])} GPU / "
            f"{gpu_acceleration.get('value_dispatches', gpu_acceleration['total_dispatches'])} tracked dispatches)"
        )
        print(
            "Raw dispatch breadth: "
            f"{gpu_acceleration.get('raw_dispatch_pct', gpu_acceleration['gpu_accel_pct']):.2f}% "
            f"({gpu_acceleration['gpu_dispatches']} GPU / {gpu_acceleration['total_dispatches']} dispatches)"
        )
        if gpu_acceleration.get("deferred_dispatches"):
            print(f"Deferred dispatch records: {gpu_acceleration['deferred_dispatches']}")
        total_dispatches = gpu_acceleration["total_dispatches"]
        fallback_rate = 0.0 if total_dispatches == 0 else gpu_acceleration["fallback_dispatches"] / total_dispatches
        print(f"Fallback rate: {fallback_rate:.4f}")
        family_breakdown = gpu_acceleration.get("family_breakdown", {})
        tracked_families = [
            (family, details)
            for family, details in family_breakdown.items()
            if details.get("included_in_value_metric")
        ]
        if tracked_families:
            print("Tracked families:")
            for family, details in tracked_families:
                family_pct = details.get("gpu_work_pct", details["gpu_accel_pct"])
                gpu_units = details.get("gpu_work_units", details["gpu_dispatches"])
                total_units = details.get("total_work_units", details["total_dispatches"])
                print(
                    f"  {family:<14} {family_pct:.2f}% "
                    f"({gpu_units}/{total_units} work-units, "
                    f"{details['gpu_dispatches']}/{details['total_dispatches']} dispatches, "
                    f"weight {details['weight']})"
                )
    print(json.dumps(report, indent=2))


def _status_detail(status: CommandStatus) -> str:
    output = "\n".join(part for part in (status.stderr, status.stdout) if part).strip()
    if not output:
        return f"command exited {status.returncode}"
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    generic_suffixes = {
        "Please also report this if it was a user error, so that a better error message can be provided next time.",
        "Thanks!",
    }
    for line in reversed(lines):
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*Error:", line):
            return line
        if line.startswith("ERROR:"):
            return line
    for line in reversed(lines):
        if line in generic_suffixes:
            continue
        if line.startswith(("To report this error", "The full traceback has been saved")):
            continue
        return line
    return f"command exited {status.returncode}"


def collect_release_packaging_build(root: Path) -> dict[str, str]:
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "dist"
        output_dir.mkdir()
        status = run_command(["uv", "build", "--out-dir", str(output_dir)], cwd=root, timeout=180)
        wheel_count = len(list(output_dir.glob("*.whl")))
        sdist_count = len(list(output_dir.glob("*.tar.gz")))

    if not status.ok:
        return {"name": "packaging_build", "status": "fail", "detail": _status_detail(status)}
    if wheel_count != 1 or sdist_count != 1:
        return {
            "name": "packaging_build",
            "status": "fail",
            "detail": f"expected 1 wheel + 1 sdist, got {wheel_count} wheel and {sdist_count} sdist",
        }
    return {"name": "packaging_build", "status": "pass", "detail": "1 wheel, 1 sdist"}


def collect_release_doc_build(root: Path) -> dict[str, str]:
    with tempfile.TemporaryDirectory() as temp_dir:
        build_dir = Path(temp_dir) / "html"
        doctree_dir = Path(temp_dir) / "doctrees"
        status = run_command(
            [
                "uv",
                "run",
                "sphinx-build",
                "-b",
                "html",
                "-d",
                str(doctree_dir),
                "docs",
                str(build_dir),
            ],
            cwd=root,
            timeout=180,
        )
        index_exists = (build_dir / "index.html").exists()

    if not status.ok:
        return {"name": "doc_build", "status": "fail", "detail": _status_detail(status)}
    if not index_exists:
        return {"name": "doc_build", "status": "fail", "detail": "docs build completed without index.html"}
    return {"name": "doc_build", "status": "pass", "detail": "html docs built"}


def collect_release_notes_check(root: Path) -> dict[str, str]:
    workflow_path = root / ".github" / "workflows" / "release.yml"
    contributing_path = root / "docs" / "dev" / "contributing.md"
    workflow = workflow_path.read_text(encoding="utf-8")
    contributing = contributing_path.read_text(encoding="utf-8")

    workflow_ok = (
        "Build patch notes from commit subjects" in workflow
        and "gh release create" in workflow
        and "--notes" in workflow
    )
    policy_ok = "GitHub Releases are the source of truth for release notes" in contributing

    if not workflow_ok:
        return {
            "name": "release_notes",
            "status": "fail",
            "detail": "release workflow is missing release-notes generation wiring",
        }
    if not policy_ok:
        return {
            "name": "release_notes",
            "status": "fail",
            "detail": "contributing guide is missing the release-notes policy",
        }
    return {
        "name": "release_notes",
        "status": "pass",
        "detail": "GitHub Release notes auto-generated from commit subjects",
    }


def collect_release_generated_artifacts(root: Path) -> dict[str, str]:
    status = run_command(
        ["git", "status", "--short", "--", "docs/ops/intake-index.json"],
        cwd=root,
        timeout=30,
    )
    if not status.ok:
        return {"name": "generated_artifacts", "status": "fail", "detail": _status_detail(status)}
    if status.stdout.strip():
        return {
            "name": "generated_artifacts",
            "status": "fail",
            "detail": "tracked generated artifact is dirty: docs/ops/intake-index.json",
        }
    return {"name": "generated_artifacts", "status": "pass", "detail": "tracked generated artifacts clean"}


def collect_release_health_report(root: Path) -> dict[str, Any]:
    version_status = run_command(
        ["uv", "run", "pytest", "-q", "tests/test_version_consistency.py"],
        cwd=root,
        timeout=60,
    )
    version_tests = parse_pytest_summary(version_status.stdout + "\n" + version_status.stderr)
    docs = collect_docs(root)
    checks = [
        {
            "name": "versioning",
            "status": "pass" if version_status.ok else "fail",
            "detail": f"{version_tests['counts']['passed']}/{_pytest_total(version_tests['counts'])} passed",
        },
        {
            "name": "doc_hygiene",
            "status": "pass" if docs["ok"] else "fail",
            "detail": "generated docs fresh" if docs["ok"] else "; ".join(docs["errors"]),
        },
        collect_release_packaging_build(root),
        collect_release_doc_build(root),
        collect_release_notes_check(root),
        collect_release_generated_artifacts(root),
    ]
    measured_failures = any(check["status"] == "fail" for check in checks)
    unmeasured = any(check["status"] == "unmeasured" for check in checks)
    status = "fail" if measured_failures else "unmeasured" if unmeasured else "pass"
    return {
        "tier": "release",
        "status": status,
        "ok": not measured_failures,
        "checks": checks,
        "exit_code": 1 if measured_failures else 0,
    }


def print_release_summary(report: dict[str, Any]) -> None:
    print(f"Repo Health — release: {report['status'].upper()}")
    print("Release checks:")
    for check in report["checks"]:
        print(f"  {check['name']}: {check['status'].upper()} ({check['detail']})")
    print(json.dumps(report, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize repo health for session bootstrap.")
    parser.add_argument(
        "--tier",
        choices=("bootstrap", "contract", "gpu", "release"),
        default="bootstrap",
        help="Select which health tier to collect.",
    )
    parser.add_argument("--json", action="store_true", help="Print only JSON.")
    parser.add_argument("--check", action="store_true", help="Exit non-zero when repo health is degraded.")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Write the current tier report into the committed health baseline file.",
    )
    parser.add_argument(
        "--gpu-coverage",
        action="store_true",
        help="Run upstream GPU acceleration coverage alongside the standard health checks.",
    )
    parser.add_argument("--gpu-coverage-timeout", type=int, default=600)
    parser.add_argument("--surface-timeout", type=int, default=600)
    parser.add_argument(
        "--baseline-path",
        default=str(BASELINE_PATH),
        help="Path to the committed health baseline JSON file.",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    if args.tier == "bootstrap":
        report = collect_health_report(
            root,
            include_gpu_coverage=args.gpu_coverage,
            gpu_coverage_timeout=args.gpu_coverage_timeout,
        )
    elif args.tier == "contract":
        report = collect_contract_health_report(root, surface_timeout=args.surface_timeout)
    elif args.tier == "gpu":
        report = collect_gpu_health_report(root, gpu_coverage_timeout=args.gpu_coverage_timeout)
    else:
        report = collect_release_health_report(root)

    baseline_path = (root / args.baseline_path).resolve()
    baseline_payload = load_health_baseline(baseline_path)

    if args.update_baseline:
        baseline_payload = update_baseline_payload(baseline_payload, report)
        write_health_baseline(baseline_path, baseline_payload)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        if args.tier == "bootstrap":
            print_summary(report)
        elif args.tier == "contract":
            print_contract_summary(report)
        elif args.tier == "gpu":
            print_gpu_summary(report)
        else:
            print_release_summary(report)
        if args.update_baseline:
            print(f"Baseline updated: {baseline_path.relative_to(root)}")

    if not args.check:
        return 0

    check_exit_code, regressions = evaluate_check_report(report, baseline_payload)
    if regressions and not args.json:
        print("Regressions:")
        for regression in regressions:
            print(f"- {regression}")
    return check_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
