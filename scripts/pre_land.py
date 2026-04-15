from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

try:
    from .health import collect_health_report
except ImportError:
    from health import collect_health_report


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str

    @property
    def ok(self) -> bool:
        return self.status != "FAIL"


def _pass_fail(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _signal_status(signal: dict[str, object]) -> str:
    return str(signal.get("status", "unmeasured")).upper()


def run_pre_land(root: Path) -> list[CheckResult]:
    report = collect_health_report(root)
    return [
        CheckResult("health", _pass_fail(report["exit_code"] == 0), f"health exit={report['exit_code']}"),
        CheckResult(
            "properties",
            _pass_fail(report["properties"]["ok"]),
            (
                f"{report['properties']['clean']}/{report['properties']['total']} clean, "
                f"{report['properties']['regressed']} regressed, "
                f"distance={report['properties']['total_distance']:.2f}"
            ),
        ),
        CheckResult("lint", _pass_fail(report["lint"]["ok"]), "ruff + architecture lint"),
        CheckResult("docs", _pass_fail(report["docs"]["ok"]), "generated docs fresh"),
        CheckResult("tests", _pass_fail(report["tests"]["ok"]), report["tests"]["command"]),
        CheckResult(
            "benchmarks",
            _signal_status(report["benchmarks"]),
            str(report["benchmarks"].get("detail", "profiling rails status unknown")),
        ),
        CheckResult("tier_gate", "UNMEASURED", "tier gates deferred until benchmark baselines exist"),
        CheckResult(
            "transfer_audit",
            _signal_status(report["transfer_audit"]),
            str(report["transfer_audit"].get("detail", "transfer audit status unknown")),
        ),
        CheckResult(
            "dispatch_policy",
            _signal_status(report["dispatch_policy"]),
            str(report["dispatch_policy"].get("detail", "dispatch policy status unknown")),
        ),
    ]


def format_results(results: list[CheckResult]) -> str:
    lines = ["Pre-land report"]
    for result in results:
        lines.append(f"- {result.status} {result.name}: {result.detail}")
    return "\n".join(lines)


def run_self_test() -> int:
    simulated = [
        CheckResult("tests", "PASS", "pytest smoke suite"),
        CheckResult("benchmarks", "FAIL", "benchmark regression >5% wall-clock"),
        CheckResult("transfer_audit", "CONFIGURED", "transfer audit policy defined"),
        CheckResult("lint", "FAIL", "ruff violation in generated file"),
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
