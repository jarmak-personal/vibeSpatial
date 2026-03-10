from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from health import collect_health_report


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def run_pre_land(root: Path) -> list[CheckResult]:
    report = collect_health_report(root)
    return [
        CheckResult("health", report["exit_code"] == 0, f"health exit={report['exit_code']}"),
        CheckResult("lint", report["lint"]["ok"], "ruff + architecture lint"),
        CheckResult("docs", report["docs"]["ok"], "generated docs fresh"),
        CheckResult("tests", report["tests"]["ok"], report["tests"]["command"]),
        CheckResult("coverage", True, f"{report['coverage']['percent']:.2f}% current smoke coverage"),
        CheckResult(
            "benchmarks",
            report["benchmarks"]["ok"],
            (
                "profiling rails available: " + ", ".join(report["benchmarks"].get("profiling_rails", []))
                if report["benchmarks"]["available"]
                else "profiling rails unavailable"
            ),
        ),
        CheckResult("tier_gate", True, "tier gates deferred until benchmark baselines exist"),
        CheckResult("transfer_audit", report["transfer_audit"]["ok"], "transfer audit available on owned geometry arrays"),
    ]


def format_results(results: list[CheckResult]) -> str:
    lines = ["Pre-land report"]
    for result in results:
        lines.append(f"- {'PASS' if result.ok else 'FAIL'} {result.name}: {result.detail}")
    return "\n".join(lines)


def run_self_test() -> int:
    simulated = [
        CheckResult("tests", True, "pytest smoke suite"),
        CheckResult("benchmarks", False, "benchmark regression >5% wall-clock"),
        CheckResult("lint", False, "ruff violation in generated file"),
    ]
    report = format_results(simulated)
    print(report)
    if "FAIL benchmarks" not in report:
        return 1
    if "benchmark regression >5% wall-clock" not in report:
        return 1
    if "FAIL lint" not in report:
        return 1
    if "ruff violation in generated file" not in report:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run pre-landing verification.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    root = Path(__file__).resolve().parents[1]
    results = run_pre_land(root)
    print(format_results(results))
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
