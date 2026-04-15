from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrendRow:
    tier: str
    status: str
    summary: str
    path: str
    modified_at: str


def _iter_report_paths(items: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(candidate for candidate in path.glob("*.json") if candidate.is_file()))
        elif path.is_file():
            paths.append(path)
    return sorted(paths, key=lambda candidate: (candidate.stat().st_mtime, candidate.name))


def _format_timestamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def _fallback_rate(report: dict[str, Any]) -> float:
    total_dispatches = report["gpu_acceleration"]["total_dispatches"]
    if not total_dispatches:
        return 0.0
    return report["gpu_acceleration"]["fallback_dispatches"] / total_dispatches


def summarize_report(report: dict[str, Any]) -> str:
    tier = report["tier"]
    if tier == "bootstrap":
        counts = report["tests"]["counts"]
        total = counts["passed"] + counts["failed"] + counts["skipped"] + counts["xfailed"] + counts["xpassed"]
        return (
            f"smoke {counts['passed']}/{total} passed, "
            f"distance={report['properties']['total_distance']:.2f}"
        )
    if tier == "contract":
        failing = [surface["name"] for surface in report["surfaces"] if not surface["ok"]]
        detail = f"required {report['required_passing']}/{report['required_total']}"
        if failing:
            detail += f", failing={','.join(failing)}"
        return detail
    if tier == "gpu":
        if report["status"] == "skipped-no-gpu":
            return "visible NVIDIA runtime unavailable"
        return (
            f"accel={report['gpu_acceleration']['gpu_accel_pct']:.2f}%, "
            f"fallback_rate={_fallback_rate(report):.4f}"
        )
    if tier == "release":
        measured_failures = [check["name"] for check in report["checks"] if check["status"] == "fail"]
        return "measured failures=" + (",".join(measured_failures) if measured_failures else "none")
    return "unknown tier"


def build_trend_rows(paths: list[Path], *, last: int) -> list[TrendRow]:
    grouped: dict[str, list[TrendRow]] = defaultdict(list)
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        row = TrendRow(
            tier=str(report.get("tier", "unknown")),
            status=str(report.get("status", "unknown")).upper(),
            summary=summarize_report(report),
            path=str(path),
            modified_at=_format_timestamp(path),
        )
        grouped[row.tier].append(row)

    rows: list[TrendRow] = []
    for tier in sorted(grouped):
        rows.extend(grouped[tier][-last:])
    return rows


def format_rows(rows: list[TrendRow]) -> str:
    lines = ["Health Trends", ""]
    for row in rows:
        lines.append(
            f"{row.tier:<10} {row.status:<15} {row.summary} [{row.modified_at}] {row.path}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize saved health JSON reports.")
    parser.add_argument("paths", nargs="+", help="Health report files or directories containing them.")
    parser.add_argument("--last", type=int, default=5, help="Keep the last N reports per tier.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of the summary table.")
    args = parser.parse_args(argv)

    rows = build_trend_rows(_iter_report_paths(args.paths), last=args.last)
    if args.json:
        print(json.dumps([asdict(row) for row in rows], indent=2))
    else:
        print(format_rows(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
