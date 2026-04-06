"""vsbench — unified benchmarking CLI for vibeSpatial.

Entry point registered as ``vsbench`` in pyproject.toml.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vsbench",
        description="vibeSpatial benchmark CLI — GPU-first geospatial benchmarks",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- vsbench run <operation> -------------------------------------------
    p_run = sub.add_parser("run", help="Run a single operation benchmark")
    p_run.add_argument("operation", help="Operation name (see 'vsbench list operations')")
    p_run.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Exact input row count override (mutually exclusive with --scale)",
    )
    p_run.add_argument(
        "--arg",
        action="append",
        dest="operation_args",
        metavar="key=value",
        help="Typed operation-specific argument; repeat as needed",
    )
    _add_common_flags(p_run)

    # --- vsbench pipeline <name> -------------------------------------------
    p_pipe = sub.add_parser("pipeline", help="Run a named pipeline benchmark")
    p_pipe.add_argument("name", help="Pipeline name (see 'vsbench list pipelines')")
    _add_common_flags(p_pipe)
    p_pipe.add_argument(
        "--suite-level",
        choices=("smoke", "ci", "full"),
        default="ci",
        help="Suite level controls scale mapping (default: ci)",
    )

    # --- vsbench suite {smoke,ci,full} -------------------------------------
    p_suite = sub.add_parser("suite", help="Run a predefined benchmark suite")
    p_suite.add_argument("level", choices=("smoke", "ci", "full"))
    _add_common_flags(p_suite)
    p_suite.add_argument(
        "--pipeline",
        action="append",
        dest="pipelines",
        help="Limit to specific pipelines (can repeat)",
    )

    # --- vsbench kernel <name> ---------------------------------------------
    p_kernel = sub.add_parser("kernel", help="NVBench kernel microbenchmark (Tier 2)")
    p_kernel.add_argument("name", help="Kernel name (see 'vsbench list kernels')")
    p_kernel.add_argument("--bandwidth", action="store_true", help="Report GB/s and %%-of-peak")
    _add_common_flags(p_kernel)

    # --- vsbench compare <baseline> <current> ------------------------------
    p_cmp = sub.add_parser("compare", help="Regression detection between two result files")
    p_cmp.add_argument("baseline", type=Path, help="Baseline JSON result file")
    p_cmp.add_argument("current", type=Path, help="Current JSON result file")
    p_cmp.add_argument("--json", action="store_true", dest="json_output")
    p_cmp.add_argument("--quiet", action="store_true")

    # --- vsbench report <results-dir> --------------------------------------
    p_report = sub.add_parser("report", help="Generate unified HTML report from results")
    p_report.add_argument("input", type=Path, help="JSON result file or directory")
    p_report.add_argument("-o", "--output", type=Path, help="Output HTML path")

    # --- vsbench list {operations,pipelines,fixtures,kernels} --------------
    p_list = sub.add_parser("list", help="List available benchmarks and resources")
    p_list.add_argument(
        "kind",
        choices=("operations", "pipelines", "fixtures", "kernels"),
    )
    p_list.add_argument("--json", action="store_true", dest="json_output")
    p_list.add_argument("--category", help="Filter operations by category")

    # --- vsbench fixtures generate --------------------------------------------
    p_fix = sub.add_parser("fixtures", help="Manage benchmark fixture files")
    fix_sub = p_fix.add_subparsers(dest="fixtures_command", required=True)
    p_gen = fix_sub.add_parser("generate", help="Generate fixture files in various formats")
    p_gen.add_argument(
        "--scale",
        choices=("1k", "10k", "100k", "1m", "all"),
        default="all",
        help="Scale to generate (default: all)",
    )
    p_gen.add_argument(
        "--format",
        choices=("parquet", "geojson", "shapefile", "gpkg", "all"),
        default="all",
        dest="gen_format",
        help="Output format (default: all)",
    )
    p_gen.add_argument("--force", action="store_true", help="Regenerate even if cached")

    # --- vsbench shootout <script.py> ----------------------------------------
    p_shootout = sub.add_parser(
        "shootout",
        help="Compare a script running with real geopandas vs vibespatial (prefer `uv run vsbench shootout`)",
    )
    p_shootout.add_argument(
        "script",
        type=Path,
        help="Python script or directory of scripts",
    )
    p_shootout.add_argument("--repeat", type=int, default=3, help="Number of timed runs (default: 3)")
    p_shootout.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip the untimed script warmup run; vibespatial still prewarms registered GPU pipelines before timing",
    )
    p_shootout.add_argument(
        "--baseline-python", type=str, default=None,
        help="Python interpreter with real geopandas (skips uv isolation)",
    )
    p_shootout.add_argument(
        "--with", action="append", dest="extra_deps",
        help="Extra pip dep for the geopandas env (can repeat)",
    )
    p_shootout.add_argument("--scale", type=str, default=None, help="Data scale for scripts (e.g. 1000, 10K, 1M)")
    p_shootout.add_argument("--timeout", type=int, default=300, help="Per-run timeout in seconds (default: 300)")
    p_shootout.add_argument("--json", action="store_true", dest="json_output")
    p_shootout.add_argument("--quiet", action="store_true")
    p_shootout.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write results to file. For GPU/strict-native runs, invoke the CLI via `uv run vsbench ...`.",
    )

    args = parser.parse_args(argv)
    return _dispatch(args)


# ---------------------------------------------------------------------------
# Common flags
# ---------------------------------------------------------------------------

def _add_common_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scale", choices=("1k", "10k", "100k", "1m"), default=None)
    parser.add_argument("--compare", choices=("shapely", "geopandas"), default=None)
    parser.add_argument("--precision", choices=("fp32", "fp64", "auto"), default="auto")
    parser.add_argument(
        "--input-format",
        choices=("parquet", "geojson", "shapefile", "gpkg"),
        default="parquet",
        help="Input file format for fixtures (default: parquet)",
    )
    parser.add_argument("--gpu-sparkline", action="store_true")
    parser.add_argument("--nvtx", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="Write results to file")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _output_mode(args: argparse.Namespace) -> str:
    if getattr(args, "json_output", False):
        return "json"
    if getattr(args, "quiet", False):
        return "quiet"
    return "human"


def _write_output(text: str, output_path: Path | None) -> None:
    print(text)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(f"Written to {output_path}", file=sys.stderr)


def _dispatch(args: argparse.Namespace) -> int:
    match args.command:
        case "run":
            return _cmd_run(args)
        case "pipeline":
            return _cmd_pipeline(args)
        case "suite":
            return _cmd_suite(args)
        case "kernel":
            return _cmd_kernel(args)
        case "compare":
            return _cmd_compare(args)
        case "report":
            return _cmd_report(args)
        case "list":
            return _cmd_list(args)
        case "fixtures":
            return _cmd_fixtures(args)
        case "shootout":
            return _cmd_shootout(args)
    return 1


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------

def _cmd_run(args: argparse.Namespace) -> int:
    from .catalog import ensure_operations_loaded, get_operation, resolve_operation_args
    from .output import render_result
    from .runner import resolve_scale, run_operation

    ensure_operations_loaded()
    spec = get_operation(args.operation)
    if args.rows is not None and args.scale is not None:
        print("vsbench run: use either --rows or --scale, not both", file=sys.stderr)
        return 2
    if args.rows is not None:
        if args.rows <= 0:
            print("vsbench run: --rows must be a positive integer", file=sys.stderr)
            return 2
        scale = args.rows
    else:
        scale = resolve_scale(args.scale, default=spec.default_scale)
    try:
        operation_args = resolve_operation_args(spec, args.operation_args)
    except ValueError as exc:
        print(f"vsbench run: {exc}", file=sys.stderr)
        return 2

    result = run_operation(
        args.operation,
        scale=scale,
        repeat=args.repeat,
        compare=args.compare,
        precision=args.precision,
        input_format=args.input_format,
        nvtx=args.nvtx,
        gpu_sparkline=args.gpu_sparkline,
        trace=args.trace,
        operation_args=operation_args,
    )

    text = render_result(result, mode=_output_mode(args))
    _write_output(text, args.output)
    return 0 if result.status in ("pass", "skip") else 1


def _cmd_pipeline(args: argparse.Namespace) -> int:
    from .output import render_result, render_suite
    from .runner import run_pipeline
    from .schema import SuiteResult

    results = run_pipeline(
        args.name,
        suite=args.suite_level,
        repeat=args.repeat,
        nvtx=args.nvtx,
        gpu_sparkline=args.gpu_sparkline,
        trace=args.trace,
    )

    mode = _output_mode(args)
    if len(results) == 1:
        text = render_result(results[0], mode=mode)
    else:
        suite = SuiteResult(suite_name=f"pipeline:{args.name}", results=results)
        text = render_suite(suite, mode=mode)

    _write_output(text, args.output)
    return 0 if all(r.status in ("pass", "skip") for r in results) else 1


def _cmd_suite(args: argparse.Namespace) -> int:
    from .output import render_suite
    from .runner import run_suite

    suite_result = run_suite(
        args.level,
        repeat=args.repeat,
        compare=args.compare,
        precision=args.precision,
        input_format=args.input_format,
        nvtx=args.nvtx,
        gpu_sparkline=args.gpu_sparkline,
        trace=args.trace,
        pipelines_filter=args.pipelines,
    )

    text = render_suite(suite_result, mode=_output_mode(args))
    _write_output(text, args.output)

    failed = sum(1 for r in suite_result.results if r.status not in ("pass", "skip"))
    return 1 if failed else 0


def _cmd_kernel(args: argparse.Namespace) -> int:
    from .output import render_result
    from .runner import resolve_scale

    try:
        from .nvbench_runner import run_kernel_bench
    except ImportError:
        print(
            "cuda-bench is not installed. Install GPU deps with:\n"
            "  uv sync --extra cu12",
            file=sys.stderr,
        )
        return 1

    scale = resolve_scale(args.scale, default=100_000)

    result = run_kernel_bench(
        args.name,
        scale=scale,
        precision=args.precision,
        bandwidth=args.bandwidth,
    )

    text = render_result(result, mode=_output_mode(args))
    _write_output(text, args.output)
    return 0 if result.status in ("pass", "skip") else 1


def _cmd_compare(args: argparse.Namespace) -> int:
    from .compare import compare
    from .output import render_list

    result = compare(args.baseline, args.current)
    mode = "json" if args.json_output else ("quiet" if args.quiet else "human")

    if mode == "json":
        import orjson

        text = orjson.dumps(result.to_dict(), option=orjson.OPT_INDENT_2).decode()
    else:
        findings = result.to_dict()["findings"]
        if not findings:
            text = "No regressions detected."
        else:
            text = render_list(
                findings,
                columns=[
                    ("pipeline", "left"),
                    ("scale", "right"),
                    ("metric", "left"),
                    ("baseline", "right"),
                    ("current", "right"),
                    ("detail", "left"),
                ],
                title=f"Regressions: {args.baseline.name} -> {args.current.name}",
                mode=mode,
            )

    print(text)
    return 1 if result.has_regressions else 0


def _cmd_report(args: argparse.Namespace) -> int:
    import json

    # Reuse existing HTML report generator
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    from benchmark_report import render_html

    data = json.loads(args.input.read_text(encoding="utf-8"))
    html_content = render_html(data)

    out_path = args.output or args.input.with_suffix(".html")
    out_path.write_text(html_content, encoding="utf-8")
    print(f"Report written to {out_path}")
    return 0


def _cmd_fixtures(args: argparse.Namespace) -> int:
    from .fixtures import (
        ALL_INPUT_FORMATS,
        InputFormat,
        ensure_fixture_format,
        list_fixture_specs,
    )
    from .runner import SCALE_MAP

    specs = list_fixture_specs()
    if args.scale != "all":
        target = SCALE_MAP[args.scale]
        specs = tuple(s for s in specs if s.rows == target)

    if args.gen_format == "all":
        formats = ALL_INPUT_FORMATS
    else:
        formats = (InputFormat(args.gen_format),)

    total = len(specs) * len(formats)
    count = 0
    skipped = 0
    for spec in specs:
        for fmt in formats:
            count += 1
            print(
                f"  [{count}/{total}] {spec.name} [{fmt}]...",
                end="",
                flush=True,
                file=sys.stderr,
            )
            try:
                path = ensure_fixture_format(spec, fmt, force=args.force)
                print(f" {path}", file=sys.stderr)
            except ValueError as exc:
                skipped += 1
                print(f" SKIP ({exc})", file=sys.stderr)

    generated = count - skipped
    msg = f"\nGenerated {generated} fixture files."
    if skipped:
        msg += f" Skipped {skipped} (unsupported format combos)."
    print(msg, file=sys.stderr)
    return 0


def _cmd_list(args: argparse.Namespace) -> int:

    mode = "json" if args.json_output else "human"

    match args.kind:
        case "operations":
            return _list_operations(mode, category=getattr(args, "category", None))
        case "pipelines":
            return _list_pipelines(mode)
        case "fixtures":
            return _list_fixtures(mode)
        case "kernels":
            return _list_kernels(mode)
    return 1


def _list_operations(mode: str, *, category: str | None = None) -> int:
    from .catalog import ensure_operations_loaded, list_operations
    from .output import render_list

    ensure_operations_loaded()
    ops = list_operations(category=category)

    items = [op.to_dict() for op in ops]
    text = render_list(
        items,
        columns=[
            ("name", "left"),
            ("category", "left"),
            ("description", "left"),
            ("default_scale", "right"),
            ("tier", "right"),
        ],
        title="Registered Operations",
        mode=mode,
    )
    print(text)
    return 0


def _list_pipelines(mode: str) -> int:
    from .output import render_list
    from .pipeline import PIPELINE_DEFINITIONS

    items = [{"name": name} for name in PIPELINE_DEFINITIONS]
    text = render_list(
        items,
        columns=[("name", "left")],
        title="Pipeline Definitions",
        mode=mode,
    )
    print(text)
    return 0


def _list_fixtures(mode: str) -> int:
    from .fixtures import list_fixture_specs
    from .output import render_list

    specs = list_fixture_specs()
    items = [
        {
            "name": s.name,
            "geometry_type": s.geometry_type,
            "distribution": s.distribution,
            "rows": s.rows,
        }
        for s in specs
    ]
    text = render_list(
        items,
        columns=[
            ("name", "left"),
            ("geometry_type", "left"),
            ("distribution", "left"),
            ("rows", "right"),
        ],
        title="Benchmark Fixtures",
        mode=mode,
    )
    print(text)
    return 0


def _list_kernels(mode: str) -> int:
    from .output import render_list

    try:
        from .nvbench_runner import list_kernel_benches

        specs = list_kernel_benches()
        items = [
            {
                "name": s.kernel_name,
                "description": s.description,
                "geometry_types": ", ".join(s.geometry_types),
                "default_scale": s.default_scale,
            }
            for s in specs
        ]
    except ImportError:
        items = []

    if not items:
        text = "(No kernel benchmarks registered. Install cuda-bench for Tier 2 support.)"
        print(text)
        return 0

    text = render_list(
        items,
        columns=[
            ("name", "left"),
            ("description", "left"),
            ("geometry_types", "left"),
            ("default_scale", "right"),
        ],
        title="NVBench Kernel Benchmarks (Tier 2)",
        mode=mode,
    )
    print(text)
    return 0


def _cmd_shootout(args: argparse.Namespace) -> int:
    from .output import render_shootout
    from .shootout import run_shootout

    target = args.script.resolve()

    # Collect script(s) to run
    if target.is_dir():
        scripts = sorted(p for p in target.glob("*.py") if not p.name.startswith("_"))
        if not scripts:
            print(f"Error: no .py files found in {target}", file=sys.stderr)
            return 1
    elif target.is_file():
        scripts = [target]
    else:
        print(f"Error: script not found: {target}", file=sys.stderr)
        return 1

    mode = _output_mode(args)
    all_failed = 0
    texts: list[str] = []

    for script in scripts:
        result = run_shootout(
            script,
            repeat=args.repeat,
            warmup=not args.no_warmup,
            extra_deps=args.extra_deps,
            baseline_python=args.baseline_python,
            timeout=args.timeout,
            quiet=args.quiet,
            scale=args.scale,
        )
        texts.append(render_shootout(result, mode=mode))
        if result.status != "pass":
            all_failed += 1

    separator = "\n" if mode == "json" else "\n\n"
    text = separator.join(texts)

    if len(scripts) > 1 and mode == "human":
        text += f"\n\n{'─' * 60}\n{len(scripts)} scripts, {len(scripts) - all_failed} passed, {all_failed} failed"

    _write_output(text, args.output)
    return 1 if all_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
