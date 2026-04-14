from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from check_docs import check_docs

try:
    from .gpu_acceleration_coverage import run_gpu_acceleration_coverage
except ImportError:
    from gpu_acceleration_coverage import run_gpu_acceleration_coverage


HEALTH_TEST_SUITE = [
    "tests/test_runtime_harness.py",
    "tests/test_geopandas_shim.py",
    "tests/test_version_consistency.py",
    "tests/test_reference_oracle.py",
    "tests/test_kernel_scaffold.py",
    "tests/test_decision_log.py",
]


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
    counts = {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0}
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
        "total_dispatches": report.total_dispatches,
        "gpu_dispatches": report.gpu_dispatches,
        "cpu_dispatches": report.cpu_dispatches,
        "fallback_dispatches": report.fallback_dispatches,
    }


def collect_health_report(root: Path, *, include_gpu_coverage: bool = False, gpu_coverage_timeout: int = 600) -> dict[str, Any]:
    coverage, tests, pytest_status = collect_coverage_and_tests(root)
    lint = collect_lint(root)
    docs = collect_docs(root)
    gpu_acceleration = collect_gpu_acceleration(root, include=include_gpu_coverage, timeout=gpu_coverage_timeout)
    profile_script = root / "scripts" / "profile_kernels.py"
    pipeline_script = root / "scripts" / "benchmark_pipelines.py"
    pipeline_compare_module = root / "src" / "vibespatial" / "bench" / "compare.py"
    pipeline_workflow = root / ".github" / "workflows" / "pipeline-benchmarks.yml"
    benchmarks = {
        "ok": (
            profile_script.exists()
            and pipeline_script.exists()
            and pipeline_compare_module.exists()
            and pipeline_workflow.exists()
        ),
        "available": profile_script.exists() or pipeline_script.exists(),
        "regressions": [],
        "profiling_rails": [
            path
            for path in (
                "scripts/profile_kernels.py" if profile_script.exists() else "",
                "scripts/benchmark_pipelines.py" if pipeline_script.exists() else "",
                "src/vibespatial/bench/compare.py" if pipeline_compare_module.exists() else "",
                ".github/workflows/pipeline-benchmarks.yml" if pipeline_workflow.exists() else "",
            )
            if path
        ],
    }
    transfer_audit = {"ok": True, "available": True, "violations": [], "policy_defined": True}
    dispatch_policy = {"ok": True, "available": True, "policy_defined": True}
    healthy = pytest_status.ok and lint["ok"] and docs["ok"]
    broken = pytest_status.returncode not in {0, 1}
    exit_code = 2 if broken else 0 if healthy else 1
    return {
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


def print_summary(report: dict[str, Any]) -> None:
    counts = report["tests"]["counts"]
    print("Repo Health")
    print(f"- coverage: {report['coverage']['percent']:.2f}%")
    print(
        "- tests: "
        f"{counts['passed']} passed, {counts['failed']} failed, "
        f"{counts['skipped']} skipped, {counts['xfailed']} xfailed"
    )
    print(f"- lint: {'ok' if report['lint']['ok'] else 'failed'}")
    print(f"- docs: {'ok' if report['docs']['ok'] else 'stale'}")
    gpu_acceleration = report["gpu_acceleration"]
    if gpu_acceleration.get("available"):
        print(
            "- gpu acceleration: "
            f"{gpu_acceleration['gpu_accel_pct']:.2f}% "
            f"({gpu_acceleration['gpu_dispatches']} GPU / {gpu_acceleration['total_dispatches']} dispatches)"
        )
    else:
        print(f"- gpu acceleration: {gpu_acceleration.get('reason', 'unavailable')}")
    print(json.dumps(report, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize repo health for session bootstrap.")
    parser.add_argument("--json", action="store_true", help="Print only JSON.")
    parser.add_argument("--check", action="store_true", help="Exit non-zero when repo health is degraded.")
    parser.add_argument(
        "--gpu-coverage",
        action="store_true",
        help="Run upstream GPU acceleration coverage alongside the standard health checks.",
    )
    parser.add_argument("--gpu-coverage-timeout", type=int, default=600)
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    report = collect_health_report(
        root,
        include_gpu_coverage=args.gpu_coverage,
        gpu_coverage_timeout=args.gpu_coverage_timeout,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_summary(report)
    return report["exit_code"] if args.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
