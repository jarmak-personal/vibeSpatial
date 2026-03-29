"""Output rendering for vsbench CLI: Rich (human), JSON (agent), quiet (CI)."""
from __future__ import annotations

from typing import Any

from .schema import BenchmarkResult, SuiteResult

# ---------------------------------------------------------------------------
# Rich availability
# ---------------------------------------------------------------------------

_RICH_AVAILABLE: bool | None = None


def _has_rich() -> bool:
    global _RICH_AVAILABLE
    if _RICH_AVAILABLE is None:
        try:
            import rich  # noqa: F401

            _RICH_AVAILABLE = True
        except ImportError:
            _RICH_AVAILABLE = False
    return _RICH_AVAILABLE


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    if seconds <= 0:
        return "0s"
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f}\u00b5s"
    if seconds < 1.0:
        return f"{seconds * 1_000:.1f}ms"
    return f"{seconds:.2f}s"


def _fmt_speedup(speedup: float | None) -> str:
    if speedup is None:
        return "-"
    return f"{speedup:.1f}x"


def _fmt_scale(scale: int) -> str:
    if scale >= 1_000_000:
        return f"{scale // 1_000_000}M"
    if scale >= 1_000:
        return f"{scale // 1_000}K"
    return str(scale)


def _fmt_bytes(b: int) -> str:
    if b < 1024:
        return f"{b}B"
    if b < 1024 * 1024:
        return f"{b / 1024:.1f}KB"
    if b < 1024 * 1024 * 1024:
        return f"{b / (1024 * 1024):.1f}MB"
    return f"{b / (1024 * 1024 * 1024):.2f}GB"


def _status_symbol(status: str) -> str:
    return {"pass": "PASS", "fail": "FAIL", "error": "ERR", "skip": "SKIP"}.get(status, "?")


# ---------------------------------------------------------------------------
# Quiet mode (CI)
# ---------------------------------------------------------------------------

def _render_quiet(result: BenchmarkResult) -> str:
    status = _status_symbol(result.status)
    parts = [f"[{status}]", result.operation, f"scale={_fmt_scale(result.scale)}"]
    parts.append(f"time={_fmt_time(result.timing.median_seconds)}")
    if result.speedup is not None:
        parts.append(f"speedup={_fmt_speedup(result.speedup)}")
    if result.transfers and (result.transfers.d2h_count + result.transfers.h2d_count) > 0:
        parts.append(f"transfers={result.transfers.d2h_count + result.transfers.h2d_count}")
    return " ".join(parts)


def _render_quiet_suite(suite: SuiteResult) -> str:
    lines = [_render_quiet(r) for r in suite.results]
    passed = sum(1 for r in suite.results if r.status == "pass")
    total = len(suite.results)
    lines.append(f"\n{passed}/{total} passed")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plain text fallback (no Rich)
# ---------------------------------------------------------------------------

def _render_plain(result: BenchmarkResult) -> str:
    lines = [
        f"  Operation: {result.operation}",
        f"  Scale:     {_fmt_scale(result.scale)}",
        f"  Time:      {_fmt_time(result.timing.median_seconds)}",
        f"  Status:    {_status_symbol(result.status)}",
    ]
    if result.speedup is not None:
        lines.append(f"  Speedup:   {_fmt_speedup(result.speedup)} vs {result.baseline_name}")
    if result.transfers:
        xfer_total = result.transfers.d2h_count + result.transfers.h2d_count
        lines.append(f"  Transfers: {xfer_total} (D2H={result.transfers.d2h_count})")
    if result.gpu_util:
        lines.append(
            f"  GPU Util:  avg={result.gpu_util.sm_utilization_pct_avg:.0f}% "
            f"max={result.gpu_util.sm_utilization_pct_max:.0f}%"
        )
        if result.gpu_util.sparkline:
            lines.append(f"  Sparkline: {result.gpu_util.sparkline}")
    if result.kernel_timing:
        kt = result.kernel_timing
        lines.append(f"  GPU Time:  {_fmt_time(kt.gpu_time_seconds)}")
        if kt.bandwidth_gb_per_second is not None:
            lines.append(f"  Bandwidth: {kt.bandwidth_gb_per_second:.1f} GB/s")
    return "\n".join(lines)


def _render_plain_suite(suite: SuiteResult) -> str:
    sections: list[str] = [f"Suite: {suite.suite_name}", "=" * 40]
    for r in suite.results:
        sections.append(_render_plain(r))
        sections.append("-" * 40)
    passed = sum(1 for r in suite.results if r.status == "pass")
    total = len(suite.results)
    sections.append(f"\n{passed}/{total} passed")
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Rich mode
# ---------------------------------------------------------------------------

def _render_rich(result: BenchmarkResult) -> str:
    from rich.console import Console
    from rich.table import Table

    table = Table(title=f"vsbench: {result.operation}", show_edge=False, pad_edge=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Scale", _fmt_scale(result.scale))
    table.add_row("Precision", result.precision)
    table.add_row("Time (median)", _fmt_time(result.timing.median_seconds))
    if result.timing.sample_count > 1:
        table.add_row(
            "Time (range)",
            f"{_fmt_time(result.timing.min_seconds)} .. {_fmt_time(result.timing.max_seconds)}",
        )

    if result.speedup is not None:
        style = "bold green" if result.speedup >= 1.0 else "bold red"
        table.add_row("Speedup", f"[{style}]{_fmt_speedup(result.speedup)}[/] vs {result.baseline_name}")

    if result.transfers:
        xfer_total = result.transfers.d2h_count + result.transfers.h2d_count
        style = "green" if xfer_total == 0 else "yellow"
        table.add_row("Transfers", f"[{style}]{xfer_total}[/] (D2H={result.transfers.d2h_count})")

    if result.gpu_util:
        gu = result.gpu_util
        table.add_row("GPU", gu.device_name)
        table.add_row(
            "SM Util",
            f"avg={gu.sm_utilization_pct_avg:.0f}% max={gu.sm_utilization_pct_max:.0f}%",
        )
        if gu.sparkline:
            table.add_row("Sparkline", gu.sparkline)
        table.add_row("VRAM Peak", _fmt_bytes(gu.vram_used_bytes_max))

    if result.kernel_timing:
        kt = result.kernel_timing
        table.add_row("GPU Time", _fmt_time(kt.gpu_time_seconds))
        if kt.bandwidth_gb_per_second is not None:
            bw_style = "green" if (kt.bandwidth_pct_of_peak or 0) > 50 else "yellow"
            pct = f" ({kt.bandwidth_pct_of_peak:.0f}%)" if kt.bandwidth_pct_of_peak else ""
            table.add_row("Bandwidth", f"[{bw_style}]{kt.bandwidth_gb_per_second:.1f} GB/s{pct}[/]")
        flags: list[str] = []
        if kt.l2_cache_flushed:
            flags.append("L2-flushed")
        if kt.throttle_detected:
            flags.append("[red]throttle[/]")
        if kt.convergence_met:
            flags.append("[green]converged[/]")
        if flags:
            table.add_row("Flags", " ".join(flags))

    if result.tier_gate_passed is not None:
        gate_style = "bold green" if result.tier_gate_passed else "bold red"
        gate_str = "PASS" if result.tier_gate_passed else "FAIL"
        table.add_row(
            f"Tier {result.metadata.get('tier_number', '?')} Gate",
            f"[{gate_style}]{gate_str}[/] (threshold: {_fmt_speedup(result.tier_gate_threshold)})",
        )

    status_style = {"pass": "bold green", "fail": "bold red", "error": "bold red", "skip": "dim"}.get(
        result.status, ""
    )
    table.add_row("Status", f"[{status_style}]{_status_symbol(result.status)}[/] {result.status_reason}")

    console = Console(file=None, force_terminal=False)
    with console.capture() as capture:
        console.print(table)
    return capture.get()


def _render_rich_suite(suite: SuiteResult) -> str:
    from rich.console import Console
    from rich.table import Table

    table = Table(
        title=f"vsbench suite: {suite.suite_name}",
        show_edge=False,
        pad_edge=False,
    )
    table.add_column("Operation", style="bold")
    table.add_column("Scale", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("Speedup", justify="right")
    table.add_column("Transfers", justify="right")
    table.add_column("GPU Util", justify="right")
    table.add_column("Status", justify="center")

    for r in suite.results:
        speedup_str = ""
        if r.speedup is not None:
            style = "green" if r.speedup >= 1.0 else "red"
            speedup_str = f"[{style}]{_fmt_speedup(r.speedup)}[/]"

        xfer_str = ""
        if r.transfers:
            total = r.transfers.d2h_count + r.transfers.h2d_count
            style = "green" if total == 0 else "yellow"
            xfer_str = f"[{style}]{total}[/]"

        gpu_str = ""
        if r.gpu_util:
            gpu_str = f"{r.gpu_util.sm_utilization_pct_avg:.0f}%"

        status_style = {
            "pass": "bold green", "fail": "bold red", "error": "bold red", "skip": "dim",
        }.get(r.status, "")
        status_str = f"[{status_style}]{_status_symbol(r.status)}[/]"

        table.add_row(
            r.operation,
            _fmt_scale(r.scale),
            _fmt_time(r.timing.median_seconds),
            speedup_str,
            xfer_str,
            gpu_str,
            status_str,
        )

    passed = sum(1 for r in suite.results if r.status == "pass")
    total = len(suite.results)
    table.add_section()
    table.add_row("", "", "", "", "", "", f"{passed}/{total}")

    console = Console(file=None, force_terminal=False)
    with console.capture() as capture:
        console.print(table)
    return capture.get()


# ---------------------------------------------------------------------------
# List rendering
# ---------------------------------------------------------------------------

def _render_list_rich(
    items: list[dict[str, Any]],
    *,
    columns: list[tuple[str, str]],
    title: str,
) -> str:
    from rich.console import Console
    from rich.table import Table

    table = Table(title=title, show_edge=False, pad_edge=False)
    for col_name, justify in columns:
        table.add_column(col_name, justify=justify)

    for item in items:
        row = [str(item.get(col_name, "")) for col_name, _ in columns]
        table.add_row(*row)

    console = Console(file=None, force_terminal=False)
    with console.capture() as capture:
        console.print(table)
    return capture.get()


def _render_list_plain(
    items: list[dict[str, Any]],
    *,
    columns: list[tuple[str, str]],
    title: str,
) -> str:
    lines = [title, "-" * len(title)]
    col_names = [c[0] for c in columns]
    if not items:
        lines.append("(none)")
        return "\n".join(lines)

    widths = [max(len(cn), max((len(str(item.get(cn, ""))) for item in items), default=0)) for cn in col_names]
    header = "  ".join(cn.ljust(w) for cn, w in zip(col_names, widths))
    lines.append(header)
    lines.append("  ".join("-" * w for w in widths))
    for item in items:
        row = "  ".join(str(item.get(cn, "")).ljust(w) for cn, w in zip(col_names, widths))
        lines.append(row)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_result(result: BenchmarkResult, *, mode: str = "human") -> str:
    """Render a single benchmark result.

    *mode*: ``'human'`` (Rich or plain), ``'json'``, ``'quiet'``.
    """
    if mode == "json":
        return result.to_json()
    if mode == "quiet":
        return _render_quiet(result)
    if _has_rich():
        return _render_rich(result)
    return _render_plain(result)


def render_suite(suite: SuiteResult, *, mode: str = "human") -> str:
    """Render a full suite result."""
    if mode == "json":
        return suite.to_json()
    if mode == "quiet":
        return _render_quiet_suite(suite)
    if _has_rich():
        return _render_rich_suite(suite)
    return _render_plain_suite(suite)


def render_list(
    items: list[dict[str, Any]],
    *,
    columns: list[tuple[str, str]],
    title: str,
    mode: str = "human",
) -> str:
    """Render a list (operations, pipelines, fixtures, kernels).

    *columns* is a list of ``(column_name, justify)`` tuples.
    """
    if mode == "json":
        import orjson

        return orjson.dumps(items, option=orjson.OPT_INDENT_2).decode()
    if _has_rich():
        return _render_list_rich(items, columns=columns, title=title)
    return _render_list_plain(items, columns=columns, title=title)


# ---------------------------------------------------------------------------
# Shootout rendering
# ---------------------------------------------------------------------------

def _shootout_status(run: Any) -> str:
    return "ERR" if run.error else "OK"


def _fingerprint_label(result: Any) -> str | None:
    fp = result.metadata.get("fingerprint")
    if not fp:
        return None
    return "MATCH" if fp == "match" else "MISMATCH"


def _render_shootout_quiet(result: Any) -> str:
    status = "PASS" if result.status == "pass" else "ERR"
    parts = [
        f"[{status}]",
        "shootout",
        result.script,
        f"geopandas={_fmt_time(result.geopandas.timing.median_seconds)}",
        f"vibespatial={_fmt_time(result.vibespatial.timing.median_seconds)}",
    ]
    if result.speedup is not None:
        parts.append(f"speedup={result.speedup:.1f}x")
    fp = _fingerprint_label(result)
    if fp:
        parts.append(f"fingerprint={fp}")
    return " ".join(parts)


def _render_shootout_plain(result: Any) -> str:
    from pathlib import Path

    name = Path(result.script).name
    lines = [f"Shootout: {name}", "=" * 50]

    for run in (result.geopandas, result.vibespatial):
        t = run.timing
        status = _shootout_status(run)
        lines.append(
            f"  {run.label:<14} median={_fmt_time(t.median_seconds):>8}  "
            f"min={_fmt_time(t.min_seconds):>8}  max={_fmt_time(t.max_seconds):>8}  "
            f"[{status}]"
        )
        if run.error:
            lines.append(f"    error: {run.error}")

    lines.append("-" * 50)
    if result.speedup is not None:
        label = "faster" if result.speedup >= 1.0 else "slower"
        lines.append(f"  Speedup: {result.speedup:.2f}x (vibespatial is {label})")
    else:
        lines.append("  Speedup: N/A")
    fp = _fingerprint_label(result)
    if fp:
        lines.append(f"  Fingerprint: {fp}")
    return "\n".join(lines)


def _render_shootout_rich(result: Any) -> str:
    from pathlib import Path

    from rich.console import Console
    from rich.table import Table

    name = Path(result.script).name
    table = Table(
        title=f"vsbench shootout: {name}",
        show_edge=False,
        pad_edge=False,
    )
    table.add_column("Runner", style="bold")
    table.add_column("Median", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Samples", justify="right")
    table.add_column("Status", justify="center")

    for run in (result.geopandas, result.vibespatial):
        t = run.timing
        status_style = "green" if not run.error else "bold red"
        status_str = f"[{status_style}]{_shootout_status(run)}[/]"
        table.add_row(
            run.label,
            _fmt_time(t.median_seconds),
            _fmt_time(t.min_seconds),
            _fmt_time(t.max_seconds),
            str(t.sample_count),
            status_str,
        )

    if result.speedup is not None:
        style = "bold green" if result.speedup >= 1.0 else "bold red"
        label = "faster" if result.speedup >= 1.0 else "slower"
        table.add_section()
        table.add_row(
            "Speedup",
            f"[{style}]{result.speedup:.2f}x[/] ({label})",
            "", "", "", "",
        )

    fp = _fingerprint_label(result)
    if fp:
        fp_style = "bold green" if fp == "MATCH" else "bold red"
        table.add_row(
            "Fingerprint",
            f"[{fp_style}]{fp}[/]",
            "", "", "", "",
        )

    console = Console(file=None, force_terminal=False)
    with console.capture() as capture:
        console.print(table)
    return capture.get()


def render_shootout(result: Any, *, mode: str = "human") -> str:
    """Render a shootout comparison result."""
    if mode == "json":
        return result.to_json()
    if mode == "quiet":
        return _render_shootout_quiet(result)
    if _has_rich():
        return _render_shootout_rich(result)
    return _render_shootout_plain(result)
