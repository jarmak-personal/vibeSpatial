from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class PromotionCommand:
    cmd: str
    allow_skip_only: bool = False


@dataclass(frozen=True)
class PromotionGroup:
    group_id: str
    name: str
    rationale: str
    test_paths: tuple[str, ...]
    smoke_commands: tuple[PromotionCommand, ...]
    tracked_pass_criteria: tuple[str, ...]
    benchmark_commands: tuple[str, ...] = ()


PROMOTION_GROUPS: tuple[PromotionGroup, ...] = (
    PromotionGroup(
        group_id="vibeSpatial-o17.7.1",
        name="extension_array_array",
        rationale=(
            "Promote the extension-array core first because it is the hottest public "
            "contract surface between GeoPandas objects and repo-owned execution paths."
        ),
        test_paths=(
            "tests/upstream/geopandas/tests/test_extension_array.py",
            "tests/upstream/geopandas/tests/test_array.py",
        ),
        smoke_commands=(
            PromotionCommand("uv run pytest tests/upstream/geopandas/tests/test_extension_array.py -q"),
            PromotionCommand("uv run pytest tests/upstream/geopandas/tests/test_array.py -q"),
        ),
        tracked_pass_criteria=(
            "test_extension_array.py passes aside from intentional skips and documented xfails",
            "test_array.py passes as the geometry-array semantic companion slice",
        ),
    ),
    PromotionGroup(
        group_id="vibeSpatial-o17.7.2",
        name="sindex_geom_methods_sjoin",
        rationale="Promote the spatial query and join contract slice after the index stack stabilizes.",
        test_paths=(
            "tests/upstream/geopandas/tests/test_sindex.py",
            "tests/upstream/geopandas/tests/test_geom_methods.py",
            "tests/upstream/geopandas/tools/tests/test_sjoin.py",
        ),
        smoke_commands=(
            PromotionCommand("uv run pytest tests/upstream/geopandas/tests/test_sindex.py -q"),
            PromotionCommand("uv run pytest tests/upstream/geopandas/tests/test_geom_methods.py -q"),
            PromotionCommand("uv run pytest tests/upstream/geopandas/tools/tests/test_sjoin.py -q"),
        ),
        tracked_pass_criteria=(
            "test_sindex.py stays green aside from the existing unordered-result xfails",
            "test_geom_methods.py stays green with dependency-aware skips only",
            "test_sjoin.py stays green aside from the existing documented xfail and skip",
        ),
        benchmark_commands=(
            "uv run vsbench run bounds-pairs --rows 20000 --arg dataset=both --arg tile_size=256",
            "uv run vsbench run spatial-query --rows 20000 --arg overlap_ratio=0.2",
        ),
    ),
    PromotionGroup(
        group_id="vibeSpatial-o17.7.3",
        name="overlay_clip_geodataframe",
        rationale="Promote constructive and GeoDataFrame-heavy groups after predicate and overlay paths stabilize.",
        test_paths=(
            "tests/upstream/geopandas/tests/test_overlay.py",
            "tests/upstream/geopandas/tools/tests/test_clip.py",
            "tests/upstream/geopandas/tests/test_geodataframe.py",
        ),
        smoke_commands=(
            PromotionCommand("uv run pytest tests/upstream/geopandas/tests/test_overlay.py -q"),
            PromotionCommand("uv run pytest tests/upstream/geopandas/tools/tests/test_clip.py -q"),
            PromotionCommand("uv run pytest tests/upstream/geopandas/tests/test_geodataframe.py -q"),
        ),
        tracked_pass_criteria=(
            "test_overlay.py stays green with the current constructive skips only",
            "test_clip.py stays green against the owned clip and fallback surface",
            "test_geodataframe.py stays green with dependency-aware skips only",
        ),
        benchmark_commands=(
            "uv run vsbench run clip-rect --arg kind=polygon",
            "uv run vsbench run make-valid --scale 10k",
        ),
    ),
    PromotionGroup(
        group_id="vibeSpatial-o17.7.4",
        name="io_groups",
        rationale="Promote IO groups with dependency-aware gates so the default bootstrap path stays lightweight.",
        test_paths=(
            "tests/upstream/geopandas/io/tests/test_file.py",
            "tests/upstream/geopandas/io/tests/test_arrow.py",
            "tests/upstream/geopandas/io/tests/test_geoarrow.py",
            "tests/upstream/geopandas/io/tests/test_sql.py",
        ),
        smoke_commands=(
            PromotionCommand("uv run pytest tests/upstream/geopandas/io/tests/test_file.py -q"),
            PromotionCommand(
                "uv run pytest tests/upstream/geopandas/io/tests/test_arrow.py -q",
                allow_skip_only=True,
            ),
            PromotionCommand(
                "uv run pytest tests/upstream/geopandas/io/tests/test_geoarrow.py -q",
                allow_skip_only=True,
            ),
            PromotionCommand("uv run pytest tests/upstream/geopandas/io/tests/test_sql.py -q", allow_skip_only=True),
        ),
        tracked_pass_criteria=(
            "file IO stays green with explicit optional-driver skips",
            "Arrow and GeoArrow slices may skip cleanly when pyarrow is absent",
            "SQL slices may skip cleanly when PostGIS drivers or live databases are absent",
        ),
    ),
)


def _resolve_group(identifier: str) -> PromotionGroup:
    for group in PROMOTION_GROUPS:
        if identifier in {group.group_id, group.name}:
            return group
    raise KeyError(f"unknown promotion group: {identifier}")


def format_group(group: PromotionGroup) -> str:
    lines = [
        f"{group.group_id} :: {group.name}",
        f"rationale: {group.rationale}",
        "tests:",
        *[f"- {path}" for path in group.test_paths],
        "smoke_commands:",
        *[
            f"- {command.cmd}{' [skip-only ok]' if command.allow_skip_only else ''}"
            for command in group.smoke_commands
        ],
        "tracked_pass_criteria:",
        *[f"- {criterion}" for criterion in group.tracked_pass_criteria],
    ]
    if group.benchmark_commands:
        lines.extend(
            [
                "benchmark_commands:",
                *[f"- {cmd}" for cmd in group.benchmark_commands],
            ]
        )
    return "\n".join(lines)


def run_group(group: PromotionGroup) -> int:
    for command in group.smoke_commands:
        completed = subprocess.run(command.cmd, shell=True, check=False)
        if completed.returncode == 0:
            continue
        if command.allow_skip_only and completed.returncode == 5:
            continue
        if completed.returncode != 0:
            return completed.returncode
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show or run upstream contract promotion groups.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List known promotion groups.")

    show_parser = subparsers.add_parser("show", help="Show one promotion group.")
    show_parser.add_argument("identifier")

    check_parser = subparsers.add_parser("check", help="Run the smoke commands for one promotion group.")
    check_parser.add_argument("identifier")

    args = parser.parse_args(argv)

    if args.command == "list":
        for group in PROMOTION_GROUPS:
            print(f"{group.group_id} {group.name}")
        return 0

    group = _resolve_group(args.identifier)
    if args.command == "show":
        print(format_group(group))
        return 0
    if args.command == "check":
        return run_group(group)
    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
