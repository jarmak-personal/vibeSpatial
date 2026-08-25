"""Aggregate work-amplification evidence from saved benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, NoReturn

from vibespatial.bench.amplification import Record
from vibespatial.bench.amplification_analysis import analyze_work_amplification

REPORT_SCHEMA_VERSION = 1


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON numeric constant {value!r}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _load_artifact(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read artifact {path}: {exc}") from exc
    try:
        payload = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON in artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"artifact {path} root must be a JSON object")
    return payload


def _validated_analysis(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    analysis = analyze_work_amplification(artifact)
    if not isinstance(analysis, dict):
        raise TypeError(f"analysis for {path} must be a dictionary")
    expected = {"artifact", "records", "findings", "observer_effects"}
    if set(analysis) != expected:
        missing = expected - analysis.keys()
        unknown = analysis.keys() - expected
        details = []
        if missing:
            details.append(f"missing {', '.join(sorted(missing))}")
        if unknown:
            details.append(f"unknown {', '.join(sorted(unknown))}")
        raise ValueError(f"invalid analysis schema for {path}: {'; '.join(details)}")
    if not isinstance(analysis["artifact"], dict):
        raise TypeError(f"analysis artifact for {path} must be a dictionary")
    for field_name in ("records", "findings", "observer_effects"):
        if not isinstance(analysis[field_name], list):
            raise TypeError(f"analysis {field_name} for {path} must be a list")
    if not analysis["records"] and not analysis["observer_effects"]:
        raise ValueError(f"artifact {path} has no recognized work-amplification evidence")
    for index, record in enumerate(analysis["records"]):
        try:
            Record.from_dict(record)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid work-amplification record {index} in {path}: {exc}") from exc
    for field_name in ("findings", "observer_effects"):
        if any(not isinstance(item, dict) for item in analysis[field_name]):
            raise TypeError(f"analysis {field_name} entries for {path} must be dictionaries")
    return analysis


def build_report(paths: Sequence[str | Path]) -> dict[str, Any]:
    """Read and aggregate artifacts without modifying their in-memory or on-disk data."""
    if isinstance(paths, (str, bytes, Path)):
        raise TypeError("paths must be a sequence of artifact paths")
    normalized_paths = tuple(Path(path) for path in paths)
    if not normalized_paths:
        raise ValueError("at least one artifact path is required")

    artifacts: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    observer_effects: list[dict[str, Any]] = []
    for artifact_index, path in enumerate(normalized_paths):
        original = _load_artifact(path)
        analysis = _validated_analysis(path, original)
        path_text = str(path)
        artifacts.append(
            {
                "artifact_index": artifact_index,
                "path": path_text,
                "artifact": analysis["artifact"],
            }
        )
        artifact_record_base = len(records)
        records.extend(
            {
                "artifact_index": artifact_index,
                "path": path_text,
                "record": record,
            }
            for record in analysis["records"]
        )
        for finding in analysis["findings"]:
            aggregate_finding = deepcopy(finding)
            aggregate_finding["artifact_rank"] = aggregate_finding.pop("rank", None)
            artifact_record_index = aggregate_finding.pop("record_index")
            aggregate_finding["artifact_record_index"] = artifact_record_index
            aggregate_finding["global_record_index"] = (
                artifact_record_base + artifact_record_index
            )
            aggregate_finding["artifact_index"] = artifact_index
            aggregate_finding["path"] = path_text
            findings.append(aggregate_finding)
        observer_effects.extend(
            {
                "artifact_index": artifact_index,
                "path": path_text,
                "observer_effect": deepcopy(observer_effect),
            }
            for observer_effect in analysis["observer_effects"]
        )

    findings.sort(
        key=lambda item: (
            -float(item.get("priority_score", 0.0)),
            -float(item.get("stage_elapsed_seconds", 0.0)),
            str(item.get("operation", "")),
            str(item.get("stage", "")),
            item["artifact_index"],
        )
    )
    for rank, finding in enumerate(findings, 1):
        finding["rank"] = rank

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "summary": {
            "artifact_count": len(artifacts),
            "record_count": len(records),
            "finding_count": len(findings),
            "observer_effect_count": len(observer_effects),
        },
        "artifacts": artifacts,
        "records": records,
        "findings": findings,
        "observer_effects": observer_effects,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate offline work-amplification evidence from benchmark JSON artifacts."
    )
    parser.add_argument("artifacts", nargs="+", type=Path, help="Benchmark artifact JSON paths")
    parser.add_argument("-o", "--output", type=Path, help="Write the report to this JSON path")
    return parser


def _reject_input_output_alias(artifacts: Sequence[Path], output: Path | None) -> None:
    if output is None:
        return
    for artifact in artifacts:
        try:
            same_file = output.samefile(artifact)
        except OSError:
            same_file = output.resolve(strict=False) == artifact.resolve(strict=False)
        if same_file:
            raise ValueError(f"output path must not overwrite input artifact {artifact}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _reject_input_output_alias(args.artifacts, args.output)
        report = build_report(args.artifacts)
        output_text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is None:
            print(output_text, end="")
        else:
            args.output.write_text(output_text, encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
