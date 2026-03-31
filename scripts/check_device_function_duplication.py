"""Detect inline CUDA __device__ function duplication in kernel source files.

Policy
------
Shared CUDA ``__device__`` functions live in ``cuda/device_functions/``.
Kernel source files should import and prepend the shared strings rather
than re-implementing the same logic inline.  Inline copies diverge over
time, miss bug fixes, and defeat centralized testing.

Detection rules
~~~~~~~~~~~~~~~
Within CUDA C++ string literals (raw strings containing ``__global__`` or
``__device__``), the following patterns are flagged:

1. **orient2d** -- ``__device__`` function containing ``two_product``,
   ``two_sum``, or ``orient2d`` outside ``cuda/device_functions/orient2d.py``
2. **point_on_segment** -- ``__device__`` function containing
   ``point_on_segment`` (but NOT ``vs_point_on_segment``, which is a call
   to the shared version) outside ``cuda/device_functions/point_on_segment.py``
3. **signed_area** -- ``__device__`` function named ``*signed_area*`` or
   ``*ring_area*`` outside ``cuda/device_functions/signed_area.py``
4. **point_in_ring** -- ``__device__`` function containing
   ``ring_contains`` or ``even_odd`` or ``point_in_polygon_simple`` outside
   ``cuda/device_functions/point_in_ring.py``
5. **CX/CY macros** -- ``#define CX`` or ``#define CY`` outside
   ``cuda/preamble.py``
6. **KAHAN_ADD** -- ``#define KAHAN_ADD`` outside ``cuda/preamble.py``
7. **typedef compute_t** -- ``typedef.*compute_t`` outside ``cuda/preamble.py``

Exemptions
~~~~~~~~~~
- Files under ``cuda/device_functions/`` are exempt from all checks
  (they ARE the shared definitions).
- Call sites (e.g. ``vs_point_on_segment(...)``) are not definitions and
  are not flagged.
- Known intentional exceptions are tracked in ``_BASELINE_VIOLATIONS``.

Usage::

    python scripts/check_device_function_duplication.py
    python scripts/check_device_function_duplication.py --verbose

Exit 0 if clean, 1 if violations found.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "vibespatial"

# ── Exempt paths ────────────────────────────────────────────────────────
# Relative to src/vibespatial/, forward-slash normalized.
DEVICE_FUNCTIONS_DIR = "cuda/device_functions/"
PREAMBLE_FILE = "cuda/preamble.py"

# ── Baseline violations ─────────────────────────────────────────────────
# Known intentional inline copies that are tracked as tech debt.
# Format: (relative path from src/vibespatial, rule_name, description).
# These are deducted from the violation count (the baseline MUST only
# shrink over time).
_BASELINE_VIOLATIONS: set[tuple[str, str]] = {
    # PIP kernel has a compute_t-aware wrapper (ring_contains_even_odd)
    # around the shared vs_ring_contains_point_with_boundary.  The wrapper
    # does coordinate un-centering before delegating.  The __device__
    # function name contains "ring_contains" and "even_odd" which trips
    # the point_in_ring rule, but it is a thin adapter, not a re-impl.
    ("kernels/predicates/point_in_polygon_source.py", "point_in_ring"),
    # validity_kernels.py has ring_contains_point_validity -- a thin
    # adapter around vs_ring_contains_point_with_boundary that resolves
    # ring offsets before delegating.  Not a re-implementation.
    ("constructive/validity_kernels.py", "point_in_ring"),
    # PIP kernel defines typedef compute_t + CX/CY inline because the
    # device function strings use raw braces and cannot pass through
    # .format() -- they are concatenated with POINT_ON_SEGMENT_DEVICE
    # and POINT_IN_RING_BOUNDARY_DEVICE which use single-brace format.
    ("kernels/predicates/point_in_polygon_source.py", "cx_cy_macro"),
    ("kernels/predicates/point_in_polygon_source.py", "typedef_compute_t"),
    # equals_exact_source.py uses typedef compute_t directly (no CX/CY
    # centering needed -- coordinate comparison only, not arithmetic).
    ("kernels/predicates/equals_exact_source.py", "typedef_compute_t"),
    # geometry_analysis_source.py uses typedef compute_t for bounds
    # kernels (bounds are always fp64 -- the typedef is for cache-key
    # consistency only).
    ("kernels/core/geometry_analysis_source.py", "typedef_compute_t"),
    # make_valid_pipeline_kernels.py uses typedef compute_t for validity
    # ring checks (simple kernel, no centering).
    ("constructive/make_valid_pipeline_kernels.py", "typedef_compute_t"),
    # normalize_kernels.py uses typedef compute_t for ring normalization
    # kernels (no centering, no Kahan).
    ("constructive/normalize_kernels.py", "typedef_compute_t"),
    # segment_primitives_kernels.py uses typedef compute_t for segment
    # intersection kernels with its own centering approach.
    ("spatial/segment_primitives_kernels.py", "typedef_compute_t"),
}


@dataclass(frozen=True)
class Violation:
    """A single duplication violation."""

    file: str
    rule: str
    detail: str

    def render(self) -> str:
        return f"VIOLATION: {self.file} -- {self.detail}"


# ── Raw string extraction ───────────────────────────────────────────────

# Pattern to find raw string literals (r""" ... """) and plain triple-
# quoted strings (""" ... """) that contain CUDA markers.
_TRIPLE_QUOTE_RE = re.compile(
    r'(?:r|R)?"""(.*?)"""'
    r"|"
    r"(?:r|R)?'''(.*?)'''",
    re.DOTALL,
)

# CUDA marker: string must contain __device__ or __global__ to be a
# kernel source string (not a docstring or comment).
_CUDA_MARKER_RE = re.compile(r"__(?:device|global)__")


def _extract_cuda_strings(source: str) -> list[str]:
    """Extract triple-quoted string contents that contain CUDA markers."""
    results: list[str] = []
    for m in _TRIPLE_QUOTE_RE.finditer(source):
        content = m.group(1) if m.group(1) is not None else m.group(2)
        if _CUDA_MARKER_RE.search(content):
            results.append(content)
    return results


# ── Rule checks ─────────────────────────────────────────────────────────


@dataclass
class _RuleContext:
    """Accumulator for violations found during a file scan."""

    rel_path: str
    violations: list[Violation] = field(default_factory=list)

    def add(self, rule: str, detail: str) -> None:
        self.violations.append(
            Violation(file=self.rel_path, rule=rule, detail=detail),
        )


def _check_orient2d(ctx: _RuleContext, cuda_strings: list[str]) -> None:
    """Rule 1: inline orient2d / two_product / two_sum definitions."""
    # We look for __device__ function definitions whose name or body
    # contains orient2d, two_product, or two_sum.  We check function
    # names and also scan for definitions of these specific functions.
    orient2d_def_re = re.compile(
        r"__device__\s+(?:__forceinline__\s+|inline\s+)?"
        r"(?:\w[\w\s*&<>,]*?)\s+"
        r"(\w*(?:two_product|two_sum|orient2d)\w*)\s*\(",
    )
    for cuda_src in cuda_strings:
        for m in orient2d_def_re.finditer(cuda_src):
            fname = m.group(1)
            # Calls to vs_orient2d, vs_two_product, vs_two_sum are OK
            # (they reference the shared version).  But a DEFINITION
            # with that name is a duplication.
            ctx.add(
                "orient2d",
                f"inline orient2d ({fname}) -- "
                f"use cuda/device_functions/orient2d.py",
            )


def _check_point_on_segment(
    ctx: _RuleContext, cuda_strings: list[str],
) -> None:
    """Rule 2: inline point_on_segment definitions."""
    # Flag __device__ functions whose name contains point_on_segment
    # but NOT vs_point_on_segment (which is the shared version name).
    pos_def_re = re.compile(
        r"__device__\s+(?:__forceinline__\s+|inline\s+)?"
        r"(?:\w[\w\s*&<>,]*?)\s+"
        r"(\w*point_on_segment\w*)\s*\(",
    )
    for cuda_src in cuda_strings:
        for m in pos_def_re.finditer(cuda_src):
            fname = m.group(1)
            # vs_point_on_segment and vs_point_on_segment_kind are the
            # shared definitions -- only flag non-vs_ prefixed copies.
            if fname.startswith("vs_"):
                continue
            ctx.add(
                "point_on_segment",
                f"inline point_on_segment ({fname}) -- "
                f"use cuda/device_functions/point_on_segment.py",
            )


def _check_signed_area(ctx: _RuleContext, cuda_strings: list[str]) -> None:
    """Rule 3: inline signed_area / ring_area function definitions."""
    sa_def_re = re.compile(
        r"__device__\s+(?:__forceinline__\s+|inline\s+)?"
        r"(?:\w[\w\s*&<>,]*?)\s+"
        r"(\w*(?:signed_area|ring_area)\w*)\s*\(",
    )
    for cuda_src in cuda_strings:
        for m in sa_def_re.finditer(cuda_src):
            fname = m.group(1)
            # vs_ring_signed_area_2x etc. are the shared versions.
            if fname.startswith("vs_"):
                continue
            ctx.add(
                "signed_area",
                f"inline signed_area ({fname}) -- "
                f"use cuda/device_functions/signed_area.py",
            )


def _check_point_in_ring(
    ctx: _RuleContext, cuda_strings: list[str],
) -> None:
    """Rule 4: inline point_in_ring definitions."""
    pir_def_re = re.compile(
        r"__device__\s+(?:__forceinline__\s+|inline\s+)?"
        r"(?:\w[\w\s*&<>,]*?)\s+"
        r"(\w*(?:ring_contains|even_odd|point_in_polygon_simple)\w*)\s*\(",
    )
    for cuda_src in cuda_strings:
        for m in pir_def_re.finditer(cuda_src):
            fname = m.group(1)
            # vs_ring_contains_point etc. are the shared versions.
            if fname.startswith("vs_"):
                continue
            ctx.add(
                "point_in_ring",
                f"inline point_in_ring ({fname}) -- "
                f"use cuda/device_functions/point_in_ring.py",
            )


def _check_cx_cy_macros(ctx: _RuleContext, source: str) -> None:
    """Rule 5: #define CX / #define CY outside preamble.py."""
    cx_cy_re = re.compile(r"#define\s+C[XY]\b")
    if cx_cy_re.search(source):
        ctx.add(
            "cx_cy_macro",
            "inline #define CX/CY -- use cuda/preamble.py PRECISION_PREAMBLE",
        )


def _check_kahan_add(ctx: _RuleContext, source: str) -> None:
    """Rule 6: #define KAHAN_ADD outside preamble.py."""
    if re.search(r"#define\s+KAHAN_ADD\b", source):
        ctx.add(
            "kahan_add",
            "inline #define KAHAN_ADD -- "
            "use cuda/preamble.py PRECISION_PREAMBLE",
        )


def _check_typedef_compute_t(ctx: _RuleContext, source: str) -> None:
    """Rule 7: typedef.*compute_t outside preamble.py."""
    if re.search(r"typedef\s+.*\bcompute_t\b", source):
        ctx.add(
            "typedef_compute_t",
            "inline typedef compute_t -- "
            "use cuda/preamble.py PRECISION_PREAMBLE",
        )


# ── Baseline filtering ──────────────────────────────────────────────────

def _is_baselined(rel_path: str, rule: str) -> bool:
    """Return True if this (file, rule) pair is in the known baseline."""
    return (rel_path, rule) in _BASELINE_VIOLATIONS


# ── Main scan ───────────────────────────────────────────────────────────

def scan_violations(verbose: bool = False) -> list[Violation]:
    """Scan all Python files in src/vibespatial for device function duplication."""
    violations: list[Violation] = []

    if not SRC_ROOT.exists():
        print(f"ERROR: source root not found: {SRC_ROOT}", file=sys.stderr)
        return violations

    python_files = sorted(
        p for p in SRC_ROOT.rglob("*.py")
        if "__pycache__" not in p.parts
    )

    for path in python_files:
        rel_path = str(path.relative_to(SRC_ROOT)).replace("\\", "/")

        # Exempt: files in cuda/device_functions/ ARE the shared defs.
        if rel_path.startswith(DEVICE_FUNCTIONS_DIR):
            if verbose:
                print(f"  EXEMPT (device_functions): {rel_path}")
            continue

        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            if verbose:
                print(f"  SKIP (read error): {rel_path}: {exc}")
            continue

        ctx = _RuleContext(rel_path=rel_path)

        # Rules 5-7 check against the raw file content (macro/typedef
        # can appear in any string literal, not just raw strings).
        is_preamble = rel_path == PREAMBLE_FILE
        if not is_preamble:
            _check_cx_cy_macros(ctx, source)
            _check_kahan_add(ctx, source)
            _check_typedef_compute_t(ctx, source)

        # Rules 1-4 check within CUDA string literals only.
        cuda_strings = _extract_cuda_strings(source)
        if cuda_strings:
            _check_orient2d(ctx, cuda_strings)
            _check_point_on_segment(ctx, cuda_strings)
            _check_signed_area(ctx, cuda_strings)
            _check_point_in_ring(ctx, cuda_strings)

        for v in ctx.violations:
            if _is_baselined(v.file, v.rule):
                if verbose:
                    print(f"  BASELINED: {v.render()}")
                continue
            violations.append(v)
            if verbose:
                print(f"  FLAGGED: {v.render()}")

    return violations


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Detect inline CUDA __device__ function duplication that "
            "should use shared versions from cuda/device_functions/."
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print exemptions, baselines, and all scan decisions.",
    )
    args = parser.parse_args()

    violations = scan_violations(verbose=args.verbose)

    if violations:
        print(f"\nFound {len(violations)} device function duplication(s):\n")
        for v in violations:
            print(f"  {v.render()}")
        print(
            "\nPolicy: shared __device__ functions belong in "
            "cuda/device_functions/. Kernel files should import and "
            "prepend the shared string constants."
        )
        print(
            "If this is an intentional fast-path copy, add it to "
            "_BASELINE_VIOLATIONS in this script."
        )
        return 1

    print("No device function duplication found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
