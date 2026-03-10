from __future__ import annotations

from scripts.contract_promotion import PROMOTION_GROUPS, _resolve_group, format_group


def test_phase_seven_registry_includes_extension_array_group() -> None:
    group = _resolve_group("vibeSpatial-o17.7.1")

    assert group.name == "extension_array_array"
    assert "test_extension_array.py" in group.test_paths[0]
    assert "test_array.py" in group.test_paths[1]
    assert len(group.smoke_commands) == 2
    assert group.smoke_commands[0].cmd.endswith("test_extension_array.py -q")


def test_format_group_lists_smoke_commands_and_criteria() -> None:
    group = PROMOTION_GROUPS[0]

    formatted = format_group(group)

    assert group.group_id in formatted
    assert "smoke_commands:" in formatted
    assert "tracked_pass_criteria:" in formatted


def test_query_promotion_group_includes_benchmark_watch_commands() -> None:
    group = _resolve_group("vibeSpatial-o17.7.2")

    assert len(group.benchmark_commands) == 2
    assert "benchmark_spatial_query.py" in group.benchmark_commands[1]


def test_io_promotion_group_marks_optional_dependency_slices() -> None:
    group = _resolve_group("vibeSpatial-o17.7.4")

    assert group.smoke_commands[1].allow_skip_only is True
    assert group.smoke_commands[2].allow_skip_only is True
    assert group.smoke_commands[3].allow_skip_only is True
