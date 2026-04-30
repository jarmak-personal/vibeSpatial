from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmark_report import render_html

from vibespatial.bench.pipeline import (
    PIPELINE_DEFINITIONS,
    benchmark_pipeline_suite,
    render_gpu_sparkline_report,
    suite_to_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run end-to-end pipeline benchmark suites.")
    parser.add_argument("--suite", choices=("smoke", "ci", "full"), default="ci")
    parser.add_argument("--repeat", type=int, default=1, help="Number of timed runs per pipeline/scale; median is reported.")
    parser.add_argument("--nvtx", action="store_true", help="Emit NVTX stage ranges when the optional nvtx package is installed.")
    parser.add_argument("--gpu-trace", action="store_true", help="Retain raw per-stage NVML GPU samples in the JSON artifact.")
    parser.add_argument("--gpu-sparkline", action="store_true", help="Embed GPU sparkline summaries in stage metadata and print a human-readable summary to stderr.")
    parser.add_argument(
        "--profile-mode",
        choices=("lean", "audit"),
        default="lean",
        help=(
            "Instrumentation mode. lean records wall-clock plus transfer "
            "counters only; audit also enables NVML sampling and CUDA event "
            "stage timing. --gpu-trace and --gpu-sparkline imply audit."
        ),
    )
    parser.add_argument(
        "--pipeline",
        action="append",
        choices=PIPELINE_DEFINITIONS,
        help="Limit execution to one or more named pipelines.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    pipelines = tuple(args.pipeline) if args.pipeline else (
        "join-heavy",
        "relation-semijoin",
        "small-grouped-constructive-reduce",
        "constructive-output-native",
        "constructive",
        "predicate-heavy",
        "zero-transfer",
    )
    results = benchmark_pipeline_suite(
        suite=args.suite,
        pipelines=pipelines,
        repeat=args.repeat,
        enable_nvtx=args.nvtx,
        retain_gpu_trace=args.gpu_trace,
        include_gpu_sparklines=args.gpu_sparkline,
        profile_mode=args.profile_mode,
    )
    if args.gpu_sparkline:
        report = render_gpu_sparkline_report(results)
        if report:
            print(report, file=sys.stderr)
    payload = suite_to_json(results, suite=args.suite, repeat=args.repeat)
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
        html_path = args.output.with_suffix(".html")
        html_path.write_text(render_html(json.loads(payload)), encoding="utf-8")
        print(f"HTML report: {html_path}", file=sys.stderr)
    print(payload)
    if any(result.pipeline == "zero-transfer" and result.status == "failed" for result in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
