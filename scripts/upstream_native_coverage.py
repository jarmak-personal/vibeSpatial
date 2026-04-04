from __future__ import annotations

import argparse
import json
import os
import re
import select
import subprocess
import sys
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

from vibespatial import STRICT_NATIVE_ENV_VAR

DEFAULT_TARGETS = ("tests/upstream/geopandas",)
DEFAULT_GROUP_NAMES = ("tests", "io", "tools")


@dataclass(frozen=True)
class NativeCoverageReport:
    command: str
    targets: tuple[str, ...]
    strict_native: bool
    counts: dict[str, int]
    native_pass_rate_percent: float
    suite_pass_rate_percent: float
    failing_tests: tuple[str, ...]
    returncode: int


@dataclass(frozen=True)
class GroupedNativeCoverageReport:
    group_by: str
    reports: tuple[NativeCoverageReport, ...]

    @property
    def counts(self) -> dict[str, int]:
        totals = {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0, "xpassed": 0}
        for report in self.reports:
            for key in totals:
                totals[key] += report.counts.get(key, 0)
        return totals

    @property
    def failing_tests(self) -> tuple[str, ...]:
        return tuple(test for report in self.reports for test in report.failing_tests)

    @property
    def native_pass_rate_percent(self) -> float:
        native_pass_rate, _suite_pass_rate = compute_native_pass_rates(self.counts)
        return native_pass_rate

    @property
    def suite_pass_rate_percent(self) -> float:
        _native_pass_rate, suite_pass_rate = compute_native_pass_rates(self.counts)
        return suite_pass_rate

    @property
    def returncode(self) -> int:
        return 0 if all(report.returncode == 0 for report in self.reports) else 1


def parse_pytest_summary(output: str) -> tuple[dict[str, int], tuple[str, ...]]:
    counts = {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0, "xpassed": 0}
    for key in counts:
        matches = re.findall(rf"(\d+) {key}", output)
        if matches:
            counts[key] = max(int(value) for value in matches)
    failing_tests = tuple(re.findall(r"^FAILED\s+(.+)$", output, flags=re.MULTILINE))
    return counts, failing_tests


def compute_native_pass_rates(counts: dict[str, int]) -> tuple[float, float]:
    executed_total = counts["passed"] + counts["failed"] + counts["xfailed"] + counts["xpassed"]
    collected_total = executed_total + counts["skipped"]
    native_pass_rate = 0.0 if executed_total == 0 else 100.0 * counts["passed"] / executed_total
    suite_pass_rate = 0.0 if collected_total == 0 else 100.0 * counts["passed"] / collected_total
    return native_pass_rate, suite_pass_rate


def discover_group_targets(
    targets: tuple[str, ...],
    *,
    cwd: Path,
    group_names: tuple[str, ...] = DEFAULT_GROUP_NAMES,
    group_by: str = "topdir",
) -> OrderedDict[str, tuple[str, ...]]:
    grouped: OrderedDict[str, tuple[str, ...]] = OrderedDict()
    if group_by not in {"topdir", "file"}:
        raise ValueError(f"unsupported group_by mode: {group_by}")
    for target in targets:
        target_path = (cwd / target).resolve()
        if group_by == "file":
            if target_path.is_dir():
                for path in sorted(target_path.rglob("test_*.py")):
                    relpath = str(path.relative_to(cwd))
                    grouped[relpath] = (relpath,)
            else:
                grouped[target] = (target,)
            continue
        if target_path.is_dir():
            group_paths = []
            for group_name in group_names:
                candidate = target_path / group_name
                if candidate.is_dir():
                    group_paths.append(str(candidate.relative_to(cwd)))
            if group_paths:
                grouped[target] = tuple(group_paths)
                continue
        grouped[target] = (target,)
    return grouped


def _emit_progress(message: str) -> None:
    print(f"[native-coverage] {message}", file=sys.stderr, flush=True)


def _run_command_capture(
    command: list[str],
    *,
    cwd: Path,
    timeout: int,
    env: dict[str, str],
    progress: bool,
    label: str,
    heartbeat_seconds: int,
) -> subprocess.CompletedProcess[str]:
    if not progress:
        return subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )

    _emit_progress(f"START {label}: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    assert process.stdout is not None
    fd = process.stdout.fileno()
    chunks: list[bytes] = []
    start = time.monotonic()
    last_heartbeat = start

    while True:
        now = time.monotonic()
        if now - start > timeout:
            process.kill()
            remaining, _ = process.communicate()
            if remaining:
                chunks.append(remaining)
                sys.stderr.buffer.write(remaining)
                sys.stderr.flush()
            raise subprocess.TimeoutExpired(
                command,
                timeout,
                output=b"".join(chunks),
            )

        ready, _, _ = select.select([fd], [], [], 1.0)
        if ready:
            data = os.read(fd, 4096)
            if data:
                chunks.append(data)
                sys.stderr.buffer.write(data)
                sys.stderr.flush()
            elif process.poll() is not None:
                break
        elif now - last_heartbeat >= heartbeat_seconds:
            _emit_progress(f"RUNNING {label}: {int(now - start)}s elapsed")
            last_heartbeat = now
            if process.poll() is not None:
                break

    returncode = process.wait()
    output = b"".join(chunks).decode("utf-8", errors="replace")
    _emit_progress(f"END {label}: returncode={returncode}")
    return subprocess.CompletedProcess(
        command,
        returncode,
        stdout=output,
        stderr="",
    )


def run_native_coverage(
    targets: tuple[str, ...],
    *,
    cwd: Path,
    timeout: int,
    progress: bool = True,
    heartbeat_seconds: int = 30,
) -> NativeCoverageReport:
    command = ["uv", "run", "pytest", "-q", *targets]
    env = dict(os.environ)
    env[STRICT_NATIVE_ENV_VAR] = "1"
    completed = _run_command_capture(
        command,
        cwd=cwd,
        timeout=timeout,
        env=env,
        progress=progress,
        label=", ".join(targets),
        heartbeat_seconds=heartbeat_seconds,
    )
    combined_output = completed.stdout + "\n" + completed.stderr
    counts, failing_tests = parse_pytest_summary(combined_output)
    native_pass_rate, suite_pass_rate = compute_native_pass_rates(counts)
    return NativeCoverageReport(
        command=" ".join(command),
        targets=targets,
        strict_native=True,
        counts=counts,
        native_pass_rate_percent=native_pass_rate,
        suite_pass_rate_percent=suite_pass_rate,
        failing_tests=failing_tests,
        returncode=completed.returncode,
    )


def run_grouped_native_coverage(
    targets: tuple[str, ...],
    *,
    cwd: Path,
    timeout: int,
    group_by: str = "topdir",
    progress: bool = True,
    heartbeat_seconds: int = 30,
) -> GroupedNativeCoverageReport:
    grouped_targets = discover_group_targets(targets, cwd=cwd, group_by=group_by)
    reports = tuple(
        run_native_coverage(
            group_targets,
            cwd=cwd,
            timeout=timeout,
            progress=progress,
            heartbeat_seconds=heartbeat_seconds,
        )
        for group_targets in grouped_targets.values()
    )
    return GroupedNativeCoverageReport(group_by=group_by, reports=reports)


def print_human_summary(report: NativeCoverageReport) -> None:
    counts = report.counts
    print("Upstream Native Coverage")
    print(f"- command: {report.command}")
    print(f"- strict native env: {STRICT_NATIVE_ENV_VAR}=1")
    print(
        f"- outcomes: {counts['passed']} passed, {counts['failed']} failed, "
        f"{counts['skipped']} skipped, {counts['xfailed']} xfailed, {counts['xpassed']} xpassed"
    )
    print(f"- native pass rate: {report.native_pass_rate_percent:.2f}%")
    print(f"- suite pass rate: {report.suite_pass_rate_percent:.2f}%")
    if report.failing_tests:
        print("- failing tests:")
        for test in report.failing_tests[:20]:
            print(f"  - {test}")


def print_grouped_human_summary(report: GroupedNativeCoverageReport) -> None:
    print("Upstream Native Coverage")
    print(f"- grouped by: {report.group_by}")
    counts = report.counts
    print(
        f"- outcomes: {counts['passed']} passed, {counts['failed']} failed, "
        f"{counts['skipped']} skipped, {counts['xfailed']} xfailed, {counts['xpassed']} xpassed"
    )
    print(f"- native pass rate: {report.native_pass_rate_percent:.2f}%")
    print(f"- suite pass rate: {report.suite_pass_rate_percent:.2f}%")
    print("- group reports:")
    for child in report.reports:
        child_counts = child.counts
        print(
            f"  - {', '.join(child.targets)}: "
            f"{child_counts['passed']} passed, {child_counts['failed']} failed, "
            f"{child_counts['skipped']} skipped, {child_counts['xfailed']} xfailed, "
            f"{child_counts['xpassed']} xpassed, "
            f"native={child.native_pass_rate_percent:.2f}%"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure vendored GeoPandas upstream test coverage under strict native mode."
    )
    parser.add_argument("targets", nargs="*", default=list(DEFAULT_TARGETS))
    parser.add_argument("--json", action="store_true", help="Print JSON instead of the human summary.")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream chunk progress and child pytest output to stderr while preserving final stdout output.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=30,
        help="Emit a progress heartbeat to stderr after this many seconds without child output.",
    )
    parser.add_argument("--check", action="store_true", help="Exit non-zero when native coverage is not 100%.")
    parser.add_argument(
        "--grouped",
        action="store_true",
        help="Run separate strict-native coverage sweeps for each upstream top-level area (tests/io/tools).",
    )
    parser.add_argument(
        "--group-by",
        choices=("topdir", "file"),
        default="topdir",
        help="How grouped mode should chunk directory targets.",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    if args.grouped:
        report = run_grouped_native_coverage(
            tuple(args.targets),
            cwd=root,
            timeout=args.timeout,
            group_by=args.group_by,
            progress=args.progress,
            heartbeat_seconds=args.heartbeat_seconds,
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "group_by": report.group_by,
                        "counts": report.counts,
                        "native_pass_rate_percent": report.native_pass_rate_percent,
                        "suite_pass_rate_percent": report.suite_pass_rate_percent,
                        "failing_tests": list(report.failing_tests),
                        "returncode": report.returncode,
                        "reports": [asdict(child) for child in report.reports],
                    },
                    indent=2,
                )
            )
        else:
            print_grouped_human_summary(report)
        if args.check:
            return 0 if report.native_pass_rate_percent == 100.0 and report.returncode == 0 else 1
    else:
        report = run_native_coverage(
            tuple(args.targets),
            cwd=root,
            timeout=args.timeout,
            progress=args.progress,
            heartbeat_seconds=args.heartbeat_seconds,
        )
        if args.json:
            print(json.dumps(asdict(report), indent=2))
        else:
            print_human_summary(report)
        if args.check:
            return 0 if report.native_pass_rate_percent == 100.0 and report.counts["failed"] == 0 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
