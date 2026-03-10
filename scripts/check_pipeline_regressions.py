from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


WALL_CLOCK_THRESHOLD = 0.05
DEVICE_MEMORY_THRESHOLD = 0.10


@dataclass(frozen=True)
class RegressionFinding:
    pipeline: str
    scale: int
    metric: str
    baseline: float | int
    current: float | int
    detail: str


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _index_results(payload: dict) -> dict[tuple[str, int], dict]:
    return {
        (result["pipeline"], int(result["scale"])): result
        for result in payload.get("results", [])
        if result.get("status") != "deferred"
    }


def compare_results(current: dict, baseline: dict) -> list[RegressionFinding]:
    findings: list[RegressionFinding] = []
    indexed_current = _index_results(current)
    indexed_baseline = _index_results(baseline)
    for key, base in indexed_baseline.items():
        if key not in indexed_current:
            continue
        cur = indexed_current[key]
        pipeline, scale = key

        base_elapsed = float(base["elapsed_seconds"])
        cur_elapsed = float(cur["elapsed_seconds"])
        if base_elapsed > 0.0 and cur_elapsed > base_elapsed * (1.0 + WALL_CLOCK_THRESHOLD):
            findings.append(
                RegressionFinding(
                    pipeline=pipeline,
                    scale=scale,
                    metric="wall_clock",
                    baseline=base_elapsed,
                    current=cur_elapsed,
                    detail=f"wall-clock regression exceeds {WALL_CLOCK_THRESHOLD:.0%}",
                )
            )

        base_transfers = int(base.get("transfer_count", 0))
        cur_transfers = int(cur.get("transfer_count", 0))
        if cur_transfers > base_transfers:
            findings.append(
                RegressionFinding(
                    pipeline=pipeline,
                    scale=scale,
                    metric="transfer_count",
                    baseline=base_transfers,
                    current=cur_transfers,
                    detail="host/device transfer count increased",
                )
            )

        base_materializations = int(base.get("materialization_count", 0))
        cur_materializations = int(cur.get("materialization_count", 0))
        if cur_materializations > base_materializations:
            findings.append(
                RegressionFinding(
                    pipeline=pipeline,
                    scale=scale,
                    metric="materialization_count",
                    baseline=base_materializations,
                    current=cur_materializations,
                    detail="host materialization count increased",
                )
            )

        base_memory = base.get("peak_device_memory_bytes")
        cur_memory = cur.get("peak_device_memory_bytes")
        if (
            isinstance(base_memory, int)
            and isinstance(cur_memory, int)
            and base_memory > 0
            and cur_memory > int(base_memory * (1.0 + DEVICE_MEMORY_THRESHOLD))
        ):
            findings.append(
                RegressionFinding(
                    pipeline=pipeline,
                    scale=scale,
                    metric="peak_device_memory_bytes",
                    baseline=base_memory,
                    current=cur_memory,
                    detail=f"device memory regression exceeds {DEVICE_MEMORY_THRESHOLD:.0%}",
                )
            )
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare pipeline benchmark suites and flag regressions.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    args = parser.parse_args(argv)

    baseline = _load(args.baseline)
    current = _load(args.current)
    findings = compare_results(current, baseline)
    payload = {
        "baseline": str(args.baseline),
        "current": str(args.current),
        "findings": [
            {
                "pipeline": finding.pipeline,
                "scale": finding.scale,
                "metric": finding.metric,
                "baseline": finding.baseline,
                "current": finding.current,
                "detail": finding.detail,
            }
            for finding in findings
        ],
    }
    print(json.dumps(payload, indent=2))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
