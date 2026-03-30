"""PostToolUse hook: GPU-first hygiene check on code an agent just wrote.

Reads Claude Code PostToolUse JSON from stdin and checks ONLY the text
the agent just wrote (new_string for Edit, content for Write) for
GPU-first violations.  This avoids false positives from pre-existing
code -- the full-file ratchet scripts handle that at commit time.

Scope: src/vibespatial/ excluding api/, testing/, bench/, _vendor/,
       operations/.

Exit codes:
    0 -- clean or out of scope
    2 -- violations found; stderr feedback injected into agent context

Suppress a false positive on a specific line with: # hygiene:ok

Run standalone for testing:
    echo '{"tool_name":"Edit","tool_input":{"file_path":"...","new_string":"import shapely"}}' | \
        python scripts/check_edit_hygiene.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Directories exempt from all checks (boundary/API layers).
_EXEMPT_PREFIXES = ("api/", "testing/", "bench/", "_vendor/", "operations/")

# GPU-pure directories where numpy is forbidden outright.
_GPU_PURE_PREFIXES = ("cuda/", "kernels/", "runtime/")

# Functions where D2H transfers are expected and acceptable.
_MATERIALIZATION_FUNCS = frozenset({
    "to_pandas", "to_numpy", "to_shapely", "values",
    "__repr__", "__str__", "_repr_html_",
    "to_wkb", "to_wkt", "to_geopandas", "_materialize_host",
})


# -- path classification -----------------------------------------------------

def _classify_path(file_path: str) -> tuple[str | None, bool]:
    """Return (rel_path, is_gpu_pure) or (None, _) if out of scope."""
    try:
        rel = str(Path(file_path).relative_to(REPO_ROOT / "src" / "vibespatial"))
    except ValueError:
        return None, False

    if any(rel.startswith(p) for p in _EXEMPT_PREFIXES):
        return None, False

    # __init__.py files are usually re-exports, not GPU logic.
    if Path(file_path).name == "__init__.py":
        return None, False

    is_gpu_pure = any(rel.startswith(p) for p in _GPU_PURE_PREFIXES)
    return rel, is_gpu_pure


# -- line-level checks -------------------------------------------------------

def _check_text(text: str, rel_path: str, is_gpu_pure: bool) -> list[str]:
    """Return a list of violation descriptions found in *text*."""
    violations: list[str] = []
    current_func: str | None = None

    for line in text.splitlines():
        stripped = line.strip()

        if not stripped:
            continue
        if "# hygiene:ok" in line:
            continue
        # Respect zcopy:ok(reason) suppressions from the zero-copy lint.
        if "zcopy:ok(" in line and "zcopy:ok()" not in line:
            continue

        # -- 7. TODO/FIXME deferring GPU work (checked on comments) -----------
        if re.search(
            r"#\s*(?:TODO|FIXME|HACK|XXX)\b.*\b(?:gpu|kernel|device|cuda|cupy)\b",
            line,
            re.IGNORECASE,
        ):
            violations.append(
                f"DEFERRED GPU WORK ({rel_path}):\n"
                f"  {stripped}\n"
                f"  -> Implement the GPU path now. vibeSpatial does not defer\n"
                f"     GPU work to later."
            )

        # Skip pure comments for remaining checks.
        if stripped.startswith("#"):
            continue

        # Track which function we are inside (best-effort for snippets).
        func_m = re.match(r"\s*(?:async\s+)?def\s+(\w+)", line)
        if func_m:
            current_func = func_m.group(1)

        # -- 1. Shapely imports in GPU paths ----------------------------------
        if re.match(r"\s*(import\s+shapely|from\s+shapely[\s.])", line):
            violations.append(
                f"SHAPELY IMPORT in GPU path ({rel_path}):\n"
                f"  {stripped}\n"
                f"  -> Use GPU kernels. Shapely operates on host Python objects\n"
                f"     and forces a device-to-host transfer."
            )

        # -- 2. NumPy imports in GPU-pure directories -------------------------
        if is_gpu_pure and re.match(r"\s*(import\s+numpy|from\s+numpy[\s.])", line):
            violations.append(
                f"NUMPY IMPORT in GPU-pure module ({rel_path}):\n"
                f"  {stripped}\n"
                f"  -> Use CuPy for device-resident data. numpy forces\n"
                f"     host-side execution."
            )

        # -- 3. D2H transfers outside materialization -------------------------
        in_materialization = current_func in _MATERIALIZATION_FUNCS
        if not in_materialization:
            # .get() with no positional args (CuPy signature).
            # dict.get(key) always has a positional arg and will NOT match.
            if re.search(r"\.\s*get\s*\(\s*(?:\)|(?:stream|order)\s*=)", line):
                violations.append(
                    f"D2H TRANSFER via .get() ({rel_path}):\n"
                    f"  {stripped}\n"
                    f"  -> Keep data on device. Use CuPy reductions/indexing\n"
                    f"     instead of pulling to host."
                )
            if re.search(r"\.\s*asnumpy\s*\(", line):
                violations.append(
                    f"D2H TRANSFER via .asnumpy() ({rel_path}):\n"
                    f"  {stripped}\n"
                    f"  -> Stay on device. Use cp.* equivalents."
                )

        # -- 4. Python for-loops over geometry attributes ---------------------
        if re.search(
            r"for\s+\w+\s+in\s+\w+\."
            r"(geoms|geometry|geometries|exterior|interiors)\b",
            line,
        ):
            violations.append(
                f"PYTHON LOOP over geometries ({rel_path}):\n"
                f"  {stripped}\n"
                f"  -> Use vectorized GPU kernel. Serial host-side iteration\n"
                f"     over geometry objects defeats GPU parallelism."
            )

        # -- 5. np.fromiter (serial Python iteration) ------------------------
        if re.search(r"(?:np|numpy)\.fromiter\s*\(", line):
            violations.append(
                f"np.fromiter() call ({rel_path}):\n"
                f"  {stripped}\n"
                f"  -> Pre-allocate array or construct on device with CuPy."
            )

        # -- 6. .astype(object) -- destroys vectorization ---------------------
        if re.search(
            r"\.astype\s*\(\s*(?:object|[\"'](?:object|O)[\"'])\s*\)", line
        ):
            violations.append(
                f"OBJECT DTYPE CAST ({rel_path}):\n"
                f"  {stripped}\n"
                f"  -> Use typed columnar arrays. Object dtype boxes every\n"
                f"     element into a Python object."
            )

        # -- 8. numpy array creation in GPU-pure modules ----------------------
        if is_gpu_pure and re.search(r"\bnp\.(array|asarray|zeros|ones|empty|full)\s*\(", line):
            violations.append(
                f"NUMPY ARRAY in GPU-pure module ({rel_path}):\n"
                f"  {stripped}\n"
                f"  -> Use cp.array / cp.zeros / etc. to create on device."
            )

    return violations


# -- main entry point ---------------------------------------------------------

def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0

    file_path = data.get("tool_input", {}).get("file_path", "")
    if not file_path:
        return 0

    rel_path, is_gpu_pure = _classify_path(file_path)
    if rel_path is None:
        return 0

    # Extract only the text the agent just wrote -- not the full file.
    tool_name = data.get("tool_name", "")
    tool_input = data.get("tool_input", {})

    if tool_name == "Write":
        text = tool_input.get("content", "")
    elif tool_name == "Edit":
        text = tool_input.get("new_string", "")
    else:
        return 0

    if not text.strip():
        return 0

    violations = _check_text(text, rel_path, is_gpu_pure)
    if not violations:
        return 0

    # Format feedback for the agent via stderr (exit 2 injects into context).
    sep = "=" * 60
    header = (
        f"GPU-FIRST HYGIENE: {len(violations)} violation(s) in code "
        f"you just wrote to {rel_path}\n{sep}"
    )
    body = "\n\n".join(violations)
    footer = (
        f"{sep}\n"
        f"Fix these NOW before writing more code on this foundation.\n"
        f"Suppress a false positive with:  # hygiene:ok"
    )

    print(f"{header}\n\n{body}\n\n{footer}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
