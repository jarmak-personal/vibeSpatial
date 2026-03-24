"""Convert benchmark JSON into a self-contained HTML report.

Usage:
    python scripts/benchmark_report.py benchmarks/results/foo.json
    python scripts/benchmark_report.py benchmarks/results/foo.json -o report.html
"""
from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path


def _fmt_time(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f}\u00b5s"
    if seconds < 1.0:
        return f"{seconds * 1_000:.1f}ms"
    return f"{seconds:.2f}s"


def _pct_of(part: float, whole: float) -> float:
    return (part / whole * 100) if whole > 0 else 0


def _bar_segment_html(
    segments: list[tuple[str, float, str, str]],
    total: float,
    bar_width_px: int = 600,
) -> str:
    """Render a stacked horizontal bar.

    Each segment is (label, seconds, css_color, tooltip_extra).
    """
    parts: list[str] = []
    for label, seconds, color, tip in segments:
        pct = _pct_of(seconds, total)
        if pct < 0.3:
            continue
        width = max(1, int(pct / 100 * bar_width_px))
        tooltip = html.escape(f"{label}: {_fmt_time(seconds)} ({pct:.1f}%)" + (f" — {tip}" if tip else ""))
        parts.append(
            f'<div class="bar-seg" style="width:{width}px;background:{color}" title="{tooltip}">'
            f'<span class="bar-label">{html.escape(label)}</span></div>'
        )
    return f'<div class="bar" style="width:{bar_width_px}px">{"".join(parts)}</div>'


_STAGE_COLORS = {
    "setup": "#6ba3d6",
    "compute": "#f4a261",
    "refine": "#e76f51",
    "filter": "#2a9d8f",
    "emit": "#9b59b6",
    "materialize": "#e9c46a",
    "transfer": "#e74c3c",
}


def _color_for(category: str) -> str:
    return _STAGE_COLORS.get(category, "#95a5a6")


def _runtime_badge(runtime: str) -> str:
    cls = "badge-gpu" if "gpu" in runtime.lower() else "badge-cpu"
    return f'<span class="badge {cls}">{html.escape(runtime)}</span>'


def _fmt_bytes(b: int) -> str:
    if b < 1024:
        return f"{b}B"
    if b < 1024 * 1024:
        return f"{b / 1024:.1f}KB"
    if b < 1024 * 1024 * 1024:
        return f"{b / (1024 * 1024):.1f}MB"
    return f"{b / (1024 * 1024 * 1024):.2f}GB"


def _stage_detail_rows(stages: list[dict], total: float) -> str:
    rows: list[str] = []
    for s in stages:
        pct = _pct_of(s["elapsed_seconds"], total)
        device = s.get("device", "")
        badge = _runtime_badge(device) if device else ""
        meta_bits: list[str] = []
        if s.get("rows_in") is not None:
            meta_bits.append(f'{s["rows_in"]:,} → {s.get("rows_out", "?"):,} rows')
        md = s.get("metadata", {})
        # Transfer cost breakdown
        xfer_secs = md.get("transfer_seconds_delta", 0.0)
        xfer_bytes = md.get("transfer_bytes_delta", 0)
        if xfer_secs > 0 or xfer_bytes > 0:
            xfer_pct = _pct_of(xfer_secs, s["elapsed_seconds"]) if s["elapsed_seconds"] > 0 else 0
            meta_bits.append(
                f'<span class="xfer-tag">transfer: {_fmt_time(xfer_secs)} '
                f'({_fmt_bytes(xfer_bytes)}, {xfer_pct:.0f}% of stage)</span>'
            )
        if md.get("gpu_substage_timings"):
            for k, v in md["gpu_substage_timings"].items():
                if isinstance(v, float):
                    meta_bits.append(f"{k}={_fmt_time(v)}")
        meta_html = " · ".join(meta_bits)
        detail = html.escape(s.get("detail", ""))
        rows.append(
            f"<tr>"
            f'<td class="stage-name">{html.escape(s["name"])}</td>'
            f"<td>{badge}</td>"
            f'<td class="num">{_fmt_time(s["elapsed_seconds"])}</td>'
            f'<td class="num">{pct:.1f}%</td>'
            f"<td>{detail}</td>"
            f'<td class="meta">{meta_html}</td>'
            f"</tr>"
        )
    return "\n".join(rows)


def _discover_pairs(results: dict[str, dict]) -> list[tuple[str, str]]:
    """Auto-discover vibeSpatial vs baseline pipeline/operation pairs.

    Matches ``<base>-geopandas`` or ``<base>-cpu`` with ``<base>``.
    """
    suffixes = ("-geopandas", "-cpu", "-shapely")
    seen_bases: dict[str, str] = {}  # base → paired key
    for key in sorted(results):
        for sfx in suffixes:
            if key.endswith(sfx):
                base = key[: -len(sfx)]
                if base in results and base not in seen_bases:
                    seen_bases[base] = key
                break
    return [(base, paired) for base, paired in seen_bases.items()]


def _render_comparison_section(
    scale: int,
    results: dict[str, dict],
    bar_width: int,
) -> str:
    """Render one scale's comparison card."""
    pairs = _discover_pairs(results)

    # Compute a shared max_time across all paired pipelines so bars share
    # the same x-axis, making visual comparison straightforward.
    shared_max_time = 0.0
    for vs_key, gp_key in pairs:
        vs = results.get(vs_key)
        gp = results.get(gp_key)
        if vs and vs.get("status") == "ok":
            shared_max_time = max(shared_max_time, vs["elapsed_seconds"])
        if gp and gp.get("status") == "ok":
            shared_max_time = max(shared_max_time, gp["elapsed_seconds"])

    shown: set[str] = set()
    sections: list[str] = []

    for vs_key, gp_key in pairs:
        vs = results.get(vs_key)
        gp = results.get(gp_key)
        if not vs and not gp:
            continue
        shown.update({vs_key, gp_key})
        sections.append(_render_pair(vs_key, vs, gp_key, gp, bar_width, shared_max_time=shared_max_time))

    for key, r in sorted(results.items()):
        if key not in shown and r.get("status") == "ok":
            sections.append(_render_single(key, r, bar_width))

    if not sections:
        return ""

    axis_html = ""
    if shared_max_time > 0 and len(pairs) > 1:
        axis_html = (
            f'<div class="shared-axis">'
            f'Shared x-axis: 0 → {_fmt_time(shared_max_time)} '
            f'({len(pairs)} pipeline pairs)'
            f'</div>'
        )

    return (
        f'<div class="scale-card">'
        f'<h2 class="scale-header" onclick="this.parentElement.classList.toggle(\'collapsed\')">'
        f'&#9660; {"Pipelines (variable scale)" if scale == 0 else f"Scale: {scale:,}"}</h2>'
        f'<div class="scale-body">{axis_html}{"".join(sections)}</div>'
        f"</div>"
    )


def _render_pair(
    vs_name: str,
    vs: dict | None,
    gp_name: str,
    gp: dict | None,
    bar_width: int,
    *,
    shared_max_time: float = 0.0,
) -> str:
    parts: list[str] = []
    parts.append('<div class="pair">')

    pair_max = max(
        (vs or {}).get("elapsed_seconds", 0),
        (gp or {}).get("elapsed_seconds", 0),
    )
    # Use the shared x-axis when available so all pairs in a scale card
    # are visually comparable; fall back to pair-local max otherwise.
    max_time = shared_max_time if shared_max_time > 0 else pair_max

    if vs and vs.get("status") == "ok" and gp and gp.get("status") == "ok":
        speedup = gp["elapsed_seconds"] / vs["elapsed_seconds"] if vs["elapsed_seconds"] > 0 else float("inf")
        parts.append(f'<div class="speedup">{speedup:.1f}x faster</div>')
    elif vs and vs.get("status") != "ok":
        parts.append(f'<div class="speedup status-bad">{vs.get("status", "?")}</div>')

    for label, r, css in [(vs_name, vs, "bar-vs"), (gp_name, gp, "bar-gp")]:
        if not r or r.get("status") != "ok":
            parts.append(f'<div class="pipeline-row"><span class="pipe-label">{html.escape(label)}</span>'
                         f'<span class="na">{"deferred" if r and r.get("status") == "deferred" else "n/a"}</span></div>')
            continue
        stages = _get_stages(r)
        segments = [(s["name"], s["elapsed_seconds"], _color_for(s.get("category", "")), "") for s in stages]
        runtime = r.get("selected_runtime", "")
        total_xfer = _transfer_summary(stages)
        xfer_html = f' <span class="xfer-tag">{total_xfer}</span>' if total_xfer else ""
        parts.append(
            f'<div class="pipeline-row">'
            f'<span class="pipe-label">{html.escape(label)} {_runtime_badge(runtime)}'
            f' <span class="total-time">{_fmt_time(r["elapsed_seconds"])}</span>{xfer_html}</span>'
            f'{_bar_segment_html(segments, max_time, bar_width)}'
            f"</div>"
        )
        # Expandable stage detail
        parts.append(
            f'<details class="stage-details"><summary>Stage breakdown</summary>'
            f'<table class="stage-table"><thead><tr>'
            f"<th>Stage</th><th>Device</th><th>Time</th><th>%</th><th>Detail</th><th>Meta</th>"
            f"</tr></thead><tbody>"
            f'{_stage_detail_rows(stages, r["elapsed_seconds"])}'
            f"</tbody></table></details>"
        )

    parts.append("</div>")
    return "\n".join(parts)


def _render_single(name: str, r: dict, bar_width: int) -> str:
    stages = _get_stages(r)
    segments = [(s["name"], s["elapsed_seconds"], _color_for(s.get("category", "")), "") for s in stages]
    runtime = r.get("selected_runtime", "")
    return (
        f'<div class="pair"><div class="pipeline-row">'
        f'<span class="pipe-label">{html.escape(name)} {_runtime_badge(runtime)}'
        f' <span class="total-time">{_fmt_time(r["elapsed_seconds"])}</span></span>'
        f'{_bar_segment_html(segments, r["elapsed_seconds"], bar_width)}'
        f"</div>"
        f'<details class="stage-details"><summary>Stage breakdown</summary>'
        f'<table class="stage-table"><thead><tr>'
        f"<th>Stage</th><th>Device</th><th>Time</th><th>%</th><th>Detail</th><th>Meta</th>"
        f"</tr></thead><tbody>"
        f'{_stage_detail_rows(stages, r["elapsed_seconds"])}'
        f"</tbody></table></details></div>"
    )


def _transfer_summary(stages: list[dict]) -> str:
    """Compute aggregate transfer cost across all stages."""
    total_secs = 0.0
    total_bytes = 0
    total_count = 0
    for s in stages:
        md = s.get("metadata", {})
        total_secs += md.get("transfer_seconds_delta", 0.0)
        total_bytes += md.get("transfer_bytes_delta", 0)
        total_count += md.get("transfer_count_delta", 0)
    if total_count == 0:
        return ""
    return f"\u2194 {total_count} transfers: {_fmt_time(total_secs)} / {_fmt_bytes(total_bytes)}"


def _get_stages(r: dict) -> list[dict]:
    """Extract the inner stage list from a result."""
    top = r.get("stages", [])
    if top and "stages" in top[0]:
        return top[0]["stages"]
    return top


_CSS = """\
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       background: #0d1117; color: #c9d1d9; padding: 24px; max-width: 1100px; margin: 0 auto; }
h1 { color: #58a6ff; margin-bottom: 8px; }
.subtitle { color: #8b949e; margin-bottom: 24px; font-size: 14px; }
.scale-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
              margin-bottom: 16px; overflow: hidden; }
.scale-card.collapsed .scale-body { display: none; }
.scale-header { padding: 14px 18px; cursor: pointer; font-size: 18px; color: #f0f6fc;
                user-select: none; border-bottom: 1px solid #21262d; }
.scale-header:hover { background: #1c2128; }
.scale-body { padding: 16px 18px; }
.pair { margin-bottom: 18px; border-bottom: 1px solid #21262d; padding-bottom: 14px; }
.pair:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
.speedup { font-size: 22px; font-weight: 700; color: #3fb950; margin-bottom: 8px; }
.status-bad { color: #f85149; }
.pipeline-row { margin-bottom: 6px; }
.pipe-label { display: inline-block; min-width: 260px; font-weight: 600; font-size: 14px; vertical-align: middle; }
.total-time { color: #8b949e; font-weight: 400; }
.na { color: #484f58; font-style: italic; }
.bar { display: inline-flex; height: 28px; border-radius: 4px; overflow: hidden; vertical-align: middle; }
.bar-seg { height: 100%; display: flex; align-items: center; justify-content: center;
           overflow: hidden; font-size: 11px; color: #fff; cursor: default;
           border-right: 1px solid rgba(0,0,0,0.15); transition: opacity .15s; }
.bar-seg:hover { opacity: 0.8; }
.bar-label { white-space: nowrap; padding: 0 4px; text-shadow: 0 1px 2px rgba(0,0,0,0.4); }
.badge { font-size: 11px; padding: 2px 6px; border-radius: 4px; font-weight: 600; vertical-align: middle; }
.badge-gpu { background: #1f6feb33; color: #58a6ff; border: 1px solid #1f6feb55; }
.badge-cpu { background: #3fb95033; color: #3fb950; border: 1px solid #3fb95055; }
.stage-details { margin: 6px 0 4px 0; }
.stage-details summary { cursor: pointer; color: #58a6ff; font-size: 13px; padding: 2px 0; }
.stage-details summary:hover { text-decoration: underline; }
.stage-table { width: 100%; font-size: 13px; border-collapse: collapse; margin-top: 6px; }
.stage-table th { text-align: left; color: #8b949e; border-bottom: 1px solid #30363d; padding: 4px 8px; }
.stage-table td { padding: 4px 8px; border-bottom: 1px solid #21262d; }
.stage-name { font-weight: 600; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.meta { color: #8b949e; font-size: 12px; }
.xfer-tag { color: #e74c3c; font-weight: 600; }
.shared-axis { color: #8b949e; font-size: 12px; margin-bottom: 12px; padding: 6px 10px;
               background: #0d1117; border-radius: 4px; border: 1px solid #21262d; }
.legend { display: flex; gap: 14px; flex-wrap: wrap; margin: 16px 0; }
.legend-item { display: flex; align-items: center; gap: 5px; font-size: 13px; color: #8b949e; }
.legend-swatch { width: 14px; height: 14px; border-radius: 3px; }
"""


def _normalize_result(r: dict) -> dict:
    """Normalize operation-level results to the pipeline-result shape.

    Pipeline results have ``elapsed_seconds``, ``status: "ok"``, and
    ``stages``.  Operation results use ``timing.mean_seconds``,
    ``status: "pass"``, and carry a ``baseline_timing`` dict instead of
    a paired ``-geopandas`` entry.  This function bridges the gap so the
    rendering code can handle both.
    """
    if "elapsed_seconds" in r:
        return r  # already pipeline-shaped

    out = dict(r)
    timing = r.get("timing", {})
    out["elapsed_seconds"] = timing.get("mean_seconds", 0.0)

    # Map "pass" → "ok" so the renderer status checks work.
    if out.get("status") == "pass":
        out["status"] = "ok"

    # Synthesise a paired -geopandas entry from baseline_timing when present
    # so the comparison renderer can discover it.
    if r.get("baseline_timing"):
        baseline = dict(r)
        baseline["elapsed_seconds"] = r["baseline_timing"].get("mean_seconds", 0.0)
        baseline["status"] = "ok"
        baseline.pop("baseline_timing", None)
        baseline.pop("speedup", None)
        name = r.get("pipeline") or r.get("operation", "unknown")
        out["_baseline_entry"] = (f"{name}-{r.get('baseline_name', 'cpu')}", baseline)

    return out


def render_html(data: dict) -> str:
    meta = data.get("metadata", {})
    suite = meta.get("suite", "?")
    repeat = meta.get("repeat", 1)

    # Group results by scale
    by_scale: dict[int, dict[str, dict]] = defaultdict(dict)
    for r in data.get("results", []):
        nr = _normalize_result(r)
        name = nr.get("pipeline") or nr.get("operation", "unknown")
        by_scale[nr["scale"]][name] = nr
        # Inject the synthesised baseline entry so _discover_pairs finds it.
        if "_baseline_entry" in nr:
            bl_name, bl_entry = nr.pop("_baseline_entry")
            by_scale[nr["scale"]][bl_name] = bl_entry

    bar_width = 620

    # Legend
    legend_items = "".join(
        f'<div class="legend-item"><div class="legend-swatch" style="background:{c}"></div>{cat}</div>'
        for cat, c in _STAGE_COLORS.items()
    )

    # Scale sections — put scale=0 (variable-scale pipelines) last.
    scale_sections = "\n".join(
        _render_comparison_section(scale, results, bar_width)
        for scale, results in sorted(by_scale.items(), key=lambda kv: (kv[0] == 0, kv[0]))
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>vibeSpatial Benchmark Report</title>
<style>{_CSS}</style>
</head>
<body>
<h1>vibeSpatial Benchmark Report</h1>
<div class="subtitle">Suite: {html.escape(str(suite))} · Repeats: {repeat} · Scales: {", ".join(f"{s:,}" for s in sorted(by_scale))}</div>
<div class="legend">{legend_items}</div>
{scale_sections}
</body>
</html>"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert benchmark JSON to HTML report.")
    parser.add_argument("input", type=Path, help="Benchmark JSON file")
    parser.add_argument("-o", "--output", type=Path, help="Output HTML path (default: same name with .html)")
    args = parser.parse_args(argv)

    data = json.loads(args.input.read_text(encoding="utf-8"))
    html_content = render_html(data)

    out_path = args.output or args.input.with_suffix(".html")
    out_path.write_text(html_content, encoding="utf-8")
    print(f"Report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
