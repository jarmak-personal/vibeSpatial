from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_work_amplification import REPORT_SCHEMA_VERSION, build_report, main
from vibespatial.bench.amplification import Record


def _pipeline_artifact(
    *,
    operation: str = "spatial-query",
    elapsed_seconds: float = 2.0,
    candidate_pairs: int = 12,
    refined_pairs: int = 3,
) -> dict[str, object]:
    return {
        "suite": "work-amplification-test",
        "results": [
            {
                "pipeline": operation,
                "elapsed_seconds": elapsed_seconds,
                "stages": [
                    {
                        "trace": {
                            "operation": operation,
                            "total_elapsed_seconds": elapsed_seconds,
                            "stages": [
                                {
                                    "name": "exact-refinement",
                                    "elapsed_seconds": elapsed_seconds,
                                    "device": "cpu",
                                    "metadata": {
                                        "profile_boundary": "compute",
                                        "candidate_pairs": candidate_pairs,
                                        "refined_pairs": refined_pairs,
                                    },
                                }
                            ],
                        }
                    }
                ],
            }
        ],
    }


def _write_json(path: Path, payload: object) -> str:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def test_build_report_aggregates_multiple_artifacts_and_globally_ranks_findings(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_text = _write_json(
        first_path,
        _pipeline_artifact(operation="first-op", elapsed_seconds=2.0, candidate_pairs=12),
    )
    second_text = _write_json(
        second_path,
        _pipeline_artifact(operation="second-op", elapsed_seconds=3.0, candidate_pairs=30),
    )

    report = build_report((first_path, second_path))

    assert report["schema_version"] == REPORT_SCHEMA_VERSION == 1
    assert report["summary"] == {
        "artifact_count": 2,
        "record_count": 2,
        "finding_count": 2,
        "observer_effect_count": 0,
    }
    assert [item["path"] for item in report["artifacts"]] == [
        str(first_path),
        str(second_path),
    ]
    assert report["findings"][0]["operation"] == "second-op"
    assert [item["rank"] for item in report["findings"]] == [1, 2]
    assert all("artifact_rank" in item for item in report["findings"])
    assert all("artifact_record_index" in item for item in report["findings"])
    assert all("global_record_index" in item for item in report["findings"])
    assert all("record_index" not in item for item in report["findings"])
    assert all(Record.from_dict(item["record"]) for item in report["records"])
    json.dumps(report, allow_nan=False)
    assert first_path.read_text(encoding="utf-8") == first_text
    assert second_path.read_text(encoding="utf-8") == second_text


def test_main_prints_report_to_stdout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    artifact_path = tmp_path / "artifact.json"
    _write_json(artifact_path, _pipeline_artifact())

    exit_code = main([str(artifact_path)])
    captured = capsys.readouterr()
    report = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert report["summary"]["artifact_count"] == 1
    assert report["records"][0]["record"]["schema_version"] == 1


def test_main_writes_optional_output_without_writing_stdout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact_path = tmp_path / "artifact.json"
    output_path = tmp_path / "report.json"
    _write_json(artifact_path, _pipeline_artifact())

    exit_code = main([str(artifact_path), "--output", str(output_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out == ""
    assert captured.err == ""
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["summary"]["record_count"] == 1


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("{broken", "invalid JSON"),
        ('{"value": NaN}', "non-standard JSON numeric constant"),
        ('{"results": [], "results": []}', "duplicate JSON object key"),
        ("[]", "root must be a JSON object"),
        ("{}", "no recognized work-amplification evidence"),
    ],
)
def test_main_fails_closed_on_invalid_json_or_schema(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    contents: str,
    message: str,
) -> None:
    artifact_path = tmp_path / "invalid.json"
    artifact_path.write_text(contents, encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        main([str(artifact_path)])
    captured = capsys.readouterr()

    assert exc_info.value.code == 2
    assert captured.out == ""
    assert message in captured.err


def test_output_is_not_modified_when_any_input_is_invalid(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    valid_path = tmp_path / "valid.json"
    invalid_path = tmp_path / "invalid.json"
    output_path = tmp_path / "existing-report.json"
    _write_json(valid_path, _pipeline_artifact())
    invalid_path.write_text("{}", encoding="utf-8")
    output_path.write_text("existing report\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                str(valid_path),
                str(invalid_path),
                "--output",
                str(output_path),
            ]
        )
    capsys.readouterr()

    assert exc_info.value.code == 2
    assert output_path.read_text(encoding="utf-8") == "existing report\n"


def test_output_must_not_alias_an_input_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact_path = tmp_path / "artifact.json"
    original = _write_json(artifact_path, _pipeline_artifact())

    with pytest.raises(SystemExit) as exc_info:
        main([str(artifact_path), "--output", str(artifact_path)])
    captured = capsys.readouterr()

    assert exc_info.value.code == 2
    assert "must not overwrite input artifact" in captured.err
    assert artifact_path.read_text(encoding="utf-8") == original


def test_build_report_requires_a_nonempty_path_sequence() -> None:
    with pytest.raises(ValueError, match="at least one artifact path"):
        build_report(())
    with pytest.raises(TypeError, match="sequence of artifact paths"):
        build_report("artifact.json")
