"""Zero-copy enforcement lints (ZCOPY001-003).

Detects device-to-host transfer anti-patterns that violate the GPU-first
execution model.  Extends ARCH004 (which bans D/H transfers outside
materialization methods) with deeper pattern analysis for ping-pongs,
per-element transfers in loops, and boundary type leaks.

Uses a ratchet baseline: fails only when violations INCREASE beyond the
known debt count.  Decrease the baseline as debt is paid down.

Run:
    uv run python scripts/check_zero_copy.py --all
    uv run python scripts/check_zero_copy.py --all --detail
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DOC = "docs/architecture/runtime.md"

# Known pre-existing violations as of 2026-03-27.
# Decrease this number as debt is paid.  The check fails only if
# the current count EXCEEDS the baseline (new violations introduced).
_VIOLATION_BASELINE = 18  # FIX-09: keep group boundaries on device in spatial_overlay_owned 2026-03-29
# +3 intentional batch D->H in segment_primitives (OOM prevention)
# +4 materialization D->H in dbf_gpu (DataFrame construction)
# +1 intentional bulk D->H in csv_gpu _extract_wkb_and_parse (hex decode on CPU)

# Method names that pull data from device to host.
D2H_APIS = {"get", "copy_to_host", "to_host", "asnumpy", "tolist", "to_pylist"}

# Method / function names that push data from host to device.
H2D_APIS = {"asarray", "array", "to_device", "as_cupy", "to_gpu"}

# Modules whose attribute calls count as H2D when wrapping host data.
H2D_MODULES = {"cp", "cupy"}

# Directories excluded from scanning (dispatch boundary, instrumentation).
_EXCLUDED_DIRS = {"api", "testing", "_vendor", "operations", "kernels"}

# Filename stems excluded from scanning (benchmarks, profiles, IO boundary,
# bench CLI utilities that operate on JSON dicts not device arrays).
_EXCLUDED_STEMS = {
    "pipeline_benchmarks", "profile_rails", "fixture_profiles",
    "catalog", "cli", "compare", "nvbench_runner", "output", "runner", "schema", "suites",
}

# Filename prefixes excluded from scanning.
_EXCLUDED_PREFIXES = ("benchmark_", "io_", "profile_", "bench_")


@dataclass(frozen=True)
class LintError:
    code: str
    path: Path
    line: int
    message: str
    doc_path: str
    suppressed: bool = False

    def render(self, repo_root: Path) -> str:
        relative = self.path.relative_to(repo_root)
        prefix = "[SUPPRESSED] " if self.suppressed else ""
        return f"{prefix}{relative}:{self.line}: {self.code} {self.message} See {self.doc_path}."


def _is_excluded(path: Path) -> bool:
    if any(d in path.parts for d in _EXCLUDED_DIRS):
        return True
    if path.stem in _EXCLUDED_STEMS:
        return True
    if any(path.name.startswith(p) for p in _EXCLUDED_PREFIXES):
        return True
    return False


def iter_python_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*.py")
        if "__pycache__" not in p.parts and not _is_excluded(p)
    )



def _assign_parents(tree: ast.AST) -> None:
    """Annotate every node in *tree* with a ``_parent`` attribute."""
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._parent = node  # type: ignore[attr-defined]


def _enclosing_func_numpy_names(
    node: ast.AST,
    func_numpy: dict[int, frozenset[str]],
) -> frozenset[str]:
    """Walk up the ``_parent`` chain to find the enclosing function's numpy names.

    Returns ``frozenset()`` if the node is not inside any tracked function
    (e.g., module-level code), which means the caller will conservatively
    flag unknown receivers.
    """
    current: ast.AST = node
    while hasattr(current, "_parent"):
        current = current._parent  # type: ignore[attr-defined]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return func_numpy.get(id(current), frozenset())
    return frozenset()


def _enclosing_func_pyarrow_names(
    node: ast.AST,
    func_pyarrow: dict[int, frozenset[str]],
) -> frozenset[str]:
    """Walk up the ``_parent`` chain to find the enclosing function's pyarrow names.

    Returns ``frozenset()`` if the node is not inside any tracked function
    (module-level code), which is conservative (won't skip to_pylist).
    """
    current: ast.AST = node
    while hasattr(current, "_parent"):
        current = current._parent  # type: ignore[attr-defined]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return func_pyarrow.get(id(current), frozenset())
    return frozenset()


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


# Modules whose function/constructor calls produce host-resident numpy
# arrays (never device arrays).
_NUMPY_MODULES = {"np", "numpy", "shapely"}


def _is_numpy_annotation(ann: ast.expr) -> bool:
    """Return True if *ann* is or contains ``np.ndarray`` / ``numpy.ndarray``.

    Handles:
    - Simple: ``np.ndarray``
    - PEP 604 union: ``np.ndarray | None``
    """
    # Direct ``np.ndarray`` / ``numpy.ndarray``
    if (
        isinstance(ann, ast.Attribute)
        and ann.attr == "ndarray"
        and isinstance(ann.value, ast.Name)
        and ann.value.id in _NUMPY_MODULES
    ):
        return True
    # PEP 604 union: ``X | Y`` — check both sides recursively.
    if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
        return _is_numpy_annotation(ann.left) or _is_numpy_annotation(ann.right)
    return False


def _collect_numpy_names(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> frozenset[str]:
    """Collect variable names in *func_node* that are known numpy/host arrays.

    A variable is considered "known numpy" when:
    1. It is a parameter annotated as ``np.ndarray`` (including union types
       like ``np.ndarray | None``).
    2. It is assigned from a ``np.*()`` / ``numpy.*()`` / ``shapely.*()``
       call (e.g., ``x = np.flatnonzero(mask)``).
    3. It is assigned from a subscript of a known-numpy variable
       (e.g., ``y = x[start:end]``).

    This is intentionally conservative — only clear-cut cases are included.
    If a variable's provenance is ambiguous, it is NOT added (so
    ``_is_d2h_call`` will still flag it).
    """
    names: set[str] = set()

    # 1. Parameters with np.ndarray type annotations.
    for arg in func_node.args.args + func_node.args.kwonlyargs:
        ann = arg.annotation
        if ann is None:
            continue
        if _is_numpy_annotation(ann):
            names.add(arg.arg)

    # 2. Assignments from np.*/numpy.*/shapely.* calls.
    #    We only look at simple ``name = module.func(...)`` patterns —
    #    no chained calls, no tuple unpacking.
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Assign):
            continue
        # Only simple single-target assignments (``x = ...``).
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        target_name = node.targets[0].id
        value = node.value

        # Check for ``module.func(...)`` call where module is numpy/shapely.
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
            receiver = value.func.value
            if isinstance(receiver, ast.Name) and receiver.id in _NUMPY_MODULES:
                names.add(target_name)

    # 3. Propagate: variables assigned from operations on known-numpy names.
    #    This covers:
    #    - Subscripts: ``y = x[start:end]`` where *x* is in *names*.
    #    - Arithmetic: ``z = x - scalar + scalar`` where *x* is in *names*.
    #    A single propagation pass is sufficient for the patterns we see in
    #    practice (direct subscript or one level of arithmetic on a subscript).
    #    We run two passes to handle ``y = x[s:e]; z = y - k``.
    for _pass in range(2):
        for node in ast.walk(func_node):
            if not isinstance(node, ast.Assign):
                continue
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            target_name = node.targets[0].id
            if target_name in names:
                continue  # already tracked
            value = node.value
            if _value_derives_from_numpy(value, names):
                names.add(target_name)

    return frozenset(names)


def _value_derives_from_numpy(value: ast.expr, names: set[str]) -> bool:
    """Return True if *value* is derived from a known-numpy variable.

    Handles:
    - Direct name reference: ``x`` where x is in *names*
    - Subscript: ``x[i]``, ``x[start:end]``
    - Arithmetic (BinOp): ``x - scalar``, ``x + y`` where at least one
      operand derives from numpy
    - Unary op: ``-x``
    """
    if isinstance(value, ast.Name) and value.id in names:
        return True
    if isinstance(value, ast.Subscript):
        if isinstance(value.value, ast.Name) and value.value.id in names:
            return True
    if isinstance(value, ast.BinOp):
        return (
            _value_derives_from_numpy(value.left, names)
            or _value_derives_from_numpy(value.right, names)
        )
    if isinstance(value, ast.UnaryOp):
        return _value_derives_from_numpy(value.operand, names)
    return False


def _scope_has_pyarrow(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True if *func_node* body contains ``import pyarrow`` (any alias).

    Also returns True when the function references known pyarrow aliases
    (``pa``) even without an explicit import in the function body — this
    covers the common pattern of module-level ``import pyarrow as pa``
    with usage inside functions.

    This is intentionally conservative on the "skip" side: we only return
    True when pyarrow usage is unambiguous.
    """
    for node in ast.walk(func_node):
        # ``import pyarrow`` / ``import pyarrow as pa``
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pyarrow" or alias.name.startswith("pyarrow."):
                    return True
        # ``from pyarrow import ...``
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module == "pyarrow" or node.module.startswith("pyarrow."):
                return True
        # Reference to ``pa`` name — common alias for pyarrow at module level.
        # Arrow-specific method calls: .field(), .column(), .type, .num_fields
        if isinstance(node, ast.Attribute) and node.attr in (
            "field", "column", "num_fields",
        ):
            return True
    return False


# PyArrow method names whose return values are host-resident Arrow objects.
_PYARROW_PRODUCER_METHODS = frozenset({
    "field", "column", "to_pandas", "to_pydict", "chunk", "chunks",
    "combine_chunks", "slice", "filter", "take", "cast", "flatten",
})


def _collect_pyarrow_names(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> frozenset[str]:
    """Collect variable names that are known PyArrow host-resident objects.

    A variable is considered "known pyarrow" when it is assigned from a
    call to a PyArrow producer method (e.g., ``child = struct.field(0)``).

    This ONLY runs when the enclosing scope has pyarrow indicators
    (checked by the caller).  Intentionally conservative: only clear-cut
    assignment patterns are tracked.
    """
    names: set[str] = set()
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        target_name = node.targets[0].id
        value = node.value
        # Pattern: ``name = something.field(...)`` / ``.column(...)`` etc.
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
            if value.func.attr in _PYARROW_PRODUCER_METHODS:
                names.add(target_name)
    # One propagation pass: variables assigned from operations on known names.
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        target_name = node.targets[0].id
        if target_name in names:
            continue
        value = node.value
        # ``y = x.field(...)`` where x is already tracked.
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
            receiver = value.func.value
            if isinstance(receiver, ast.Name) and receiver.id in names:
                if value.func.attr in _PYARROW_PRODUCER_METHODS:
                    names.add(target_name)
    return frozenset(names)


def _receiver_is_pyarrow(
    node: ast.Call,
    pyarrow_names: frozenset[str],
) -> bool:
    """Return True if the method-call receiver is a known PyArrow host variable.

    Checks:
    - ``name.to_pylist()`` where *name* is in *pyarrow_names*
    - ``name[slice].to_pylist()``  (Subscript of a pyarrow variable)
    """
    if not isinstance(node.func, ast.Attribute):
        return False
    receiver = node.func.value
    if isinstance(receiver, ast.Name) and receiver.id in pyarrow_names:
        return True
    if isinstance(receiver, ast.Subscript):
        if isinstance(receiver.value, ast.Name) and receiver.value.id in pyarrow_names:
            return True
    return False


def _receiver_is_numpy(
    node: ast.Call,
    numpy_names: frozenset[str],
) -> bool:
    """Return True if the method-call receiver is a known numpy/host variable.

    Checks three patterns:
    - ``name.tolist()``  where *name* is in *numpy_names*
    - ``name[slice].tolist()``  (Subscript of a numpy variable)
    - ``name.attr.tolist()``  (Attribute of a numpy variable — rare)
    - ``np.func(...).tolist()``  (inline numpy call result)
    """
    if not isinstance(node.func, ast.Attribute):
        return False  # bare function call, not a method

    receiver = node.func.value

    # Direct variable: ``arr.tolist()``
    if isinstance(receiver, ast.Name) and receiver.id in numpy_names:
        return True

    # Subscript of a numpy variable: ``arr[:-1].tolist()``
    if isinstance(receiver, ast.Subscript):
        if isinstance(receiver.value, ast.Name) and receiver.value.id in numpy_names:
            return True

    # Inline numpy call result: ``np.flatnonzero(mask).tolist()``
    if isinstance(receiver, ast.Call) and isinstance(receiver.func, ast.Attribute):
        inner_receiver = receiver.func.value
        if isinstance(inner_receiver, ast.Name) and inner_receiver.id in _NUMPY_MODULES:
            return True

    return False


def _is_d2h_call(
    node: ast.Call,
    numpy_names: frozenset[str] = frozenset(),
    *,
    pyarrow_names: frozenset[str] = frozenset(),
) -> bool:
    name = _call_name(node)
    if name not in D2H_APIS:
        return False
    # --- Disambiguate dict.get() from cupy_array.get() ---
    # cupy.ndarray.get() signature: get(stream=None, order='C', out=None)
    #   → 0 positional args in normal usage, or keyword args only.
    # dict.get() signature: get(key[, default])
    #   → always has >=1 positional arg.
    #
    # Rule: if .get() has any positional args, it's dict.get() (not D2H).
    # cupy .get() is only called with 0 positional args (keyword-only).
    if name == "get" and isinstance(node.func, ast.Attribute) and node.args:
        return False  # dict.get(key), dict.get(key, default), etc.
    # --- Disambiguate pyarrow.to_pylist() from cupy.to_pylist() ---
    # to_pylist() on pyarrow arrays is host-to-host, not D2H.
    # Two-level heuristic:
    #   1. Direct receiver chain: receiver is a .field()/.column() call result.
    #   2. Receiver-name tracking: receiver is a variable assigned from a
    #      PyArrow producer method (.field(), .column(), etc.).
    if name == "to_pylist" and isinstance(node.func, ast.Attribute):
        # Level 1: immediate receiver is a .field()/.column() call result.
        if isinstance(node.func.value, ast.Call):
            inner = node.func.value
            if isinstance(inner.func, ast.Attribute) and inner.func.attr in (
                "field", "column", "to_pandas", "to_pydict",
            ):
                return False
        # Level 2: receiver is a known PyArrow variable (assigned from
        # .field() / .column() etc.).  Covers the pattern:
        #   ``var = struct.field(0); var.to_pylist()``
        if _receiver_is_pyarrow(node, pyarrow_names):
            return False
    # --- Disambiguate numpy host-to-host from cupy D->H ---
    # .tolist(), .asnumpy(), .get() on a numpy array is a host-to-host
    # conversion, not a device-to-host transfer. Skip if the receiver is
    # a known numpy/host variable or an inline numpy call result.
    if _receiver_is_numpy(node, numpy_names):
        return False
    return True


def _is_h2d_call(node: ast.Call) -> bool:
    name = _call_name(node)
    if name not in H2D_APIS:
        return False
    # Only count as H2D if the receiver is a device module (cp/cupy).
    # np.asarray() and np.array() are host-to-host — NOT H2D transfers.
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        receiver = node.func.value.id
        if receiver in H2D_MODULES:
            return True  # cp.asarray(), cupy.array() — real H2D
        if receiver in ("np", "numpy"):
            return False  # np.asarray() — host-to-host, not H2D
        # Other receivers (e.g. torch.asarray) — flag conservatively
        return True
    # Bare function call (e.g. asarray() without module prefix) — flag it
    if isinstance(node.func, ast.Name) and name in H2D_APIS:
        return True
    return False


# ---- Inline suppression: # zcopy:ok(reason) ----

_SUPPRESSION_RE = re.compile(r"#\s*zcopy:ok\(([^)]*)\)")


def _parse_suppression_comments(source: str) -> dict[int, str]:
    """Scan *source* for ``# zcopy:ok(reason)`` comments.

    Returns ``{lineno: reason}`` with 1-indexed line numbers.
    Lines where the reason is empty are **not** included — they will be
    reported as ``ZCOPY_EMPTY_SUPPRESSION`` errors by the caller.
    """
    suppressions: dict[int, str] = {}
    for lineno, line in enumerate(source.splitlines(), start=1):
        m = _SUPPRESSION_RE.search(line)
        if m is not None:
            reason = m.group(1).strip()
            if reason:  # empty-reason lines are NOT valid suppressions
                suppressions[lineno] = reason
    return suppressions


def _find_empty_suppression_lines(source: str) -> list[int]:
    """Return 1-indexed line numbers with ``# zcopy:ok()`` (empty reason)."""
    empty_lines: list[int] = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        m = _SUPPRESSION_RE.search(line)
        if m is not None and not m.group(1).strip():
            empty_lines.append(lineno)
    return empty_lines


# ---- Branch mutual-exclusion helpers ----

def _node_is_descendant_of(node: ast.AST, ancestor: ast.AST) -> bool:
    """Return True if *node* is a descendant of *ancestor* (using parent links)."""
    current: ast.AST = node
    while hasattr(current, "_parent"):
        current = current._parent  # type: ignore[attr-defined]
        if current is ancestor:
            return True
    return False


def _stmt_list_always_returns(stmts: list[ast.stmt]) -> bool:
    """Return True if a list of statements always reaches a ``return``.

    Only checks for top-level ``return`` in the statement list and simple
    ``if/else`` branches where both sides return.  This is intentionally
    conservative — complex control flow (try/except, while/break) is NOT
    analysed, so we err on the side of *not* excluding a pair (more false
    positives, fewer false negatives).
    """
    for stmt in stmts:
        if isinstance(stmt, ast.Return):
            return True
        if isinstance(stmt, ast.If):
            if stmt.orelse and _stmt_list_always_returns(stmt.body) and _stmt_list_always_returns(stmt.orelse):
                return True
    return False


def _in_mutually_exclusive_branches(
    d2h_node: ast.Call,
    h2d_node: ast.Call,
) -> bool:
    """Return True if *d2h_node* and *h2d_node* are in mutually exclusive branches.

    Detects the pattern where D2H is in an ``if`` body that always returns
    and H2D is outside that ``if`` body (or in its ``else`` branch).
    Requires ``_assign_parents()`` to have been called on the tree.

    This is intentionally conservative: it only recognises simple patterns
    where the D2H branch unconditionally returns.  Unknown or complex
    control flow is *not* excluded (false-positive safe).
    """
    # Walk up from d2h_node looking for an enclosing If node.
    current: ast.AST = d2h_node
    while hasattr(current, "_parent"):
        parent = current._parent  # type: ignore[attr-defined]
        if isinstance(parent, ast.If):
            # Determine which branch the D2H is in (body vs orelse).
            in_body = any(current is stmt or _node_is_descendant_of(current, stmt) for stmt in parent.body)
            in_orelse = any(current is stmt or _node_is_descendant_of(current, stmt) for stmt in parent.orelse)

            if in_body and _stmt_list_always_returns(parent.body):
                # D2H is in the if-body which always returns.
                # If H2D is NOT in the same body, they are mutually exclusive.
                if not _node_is_descendant_of(h2d_node, parent) or (
                    parent.orelse and any(
                        h2d_node is stmt or _node_is_descendant_of(h2d_node, stmt)
                        for stmt in parent.orelse
                    )
                ):
                    return True

            if in_orelse and _stmt_list_always_returns(parent.orelse):
                # D2H is in the else-branch which always returns.
                # If H2D is NOT in the orelse, they are mutually exclusive.
                if not _node_is_descendant_of(h2d_node, parent) or any(
                    h2d_node is stmt or _node_is_descendant_of(h2d_node, stmt)
                    for stmt in parent.body
                ):
                    return True

        # Stop at function boundary — don't leak into outer scopes.
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            break
        current = parent
    return False


# ---- ZCOPY001: Ping-pong transfers in the same function ----

def check_pingpong_transfers(
    repo_root: Path,
    *,
    include_suppressed: bool = False,
) -> tuple[list[LintError], int]:
    """Find functions where data goes D->H then back H->D (or vice versa).

    Returns ``(errors, suppressed_count)``.  When *include_suppressed* is
    True, suppressed violations appear in *errors* with ``suppressed=True``.
    """
    errors: list[LintError] = []
    suppressed = 0
    root = repo_root / "src" / "vibespatial"
    for path in iter_python_files(root):
        source = path.read_text(encoding="utf-8")
        suppressions = _parse_suppression_comments(source)
        tree = ast.parse(source, filename=str(path))
        _assign_parents(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            numpy_names = _collect_numpy_names(node)
            pa_names = _collect_pyarrow_names(node) if _scope_has_pyarrow(node) else frozenset()
            d2h_calls: list[ast.Call] = []
            h2d_calls: list[ast.Call] = []
            for descendant in ast.walk(node):
                if not isinstance(descendant, ast.Call):
                    continue
                if _is_d2h_call(descendant, numpy_names, pyarrow_names=pa_names):
                    d2h_calls.append(descendant)
                if _is_h2d_call(descendant):
                    h2d_calls.append(descendant)
            if d2h_calls and h2d_calls:
                # Sort by line number for deterministic pairing.
                d2h_sorted = sorted(d2h_calls, key=lambda c: c.lineno)
                h2d_sorted = sorted(h2d_calls, key=lambda c: c.lineno)
                # Find the first D2H → later H2D pair that is NOT in
                # mutually exclusive branches.
                paired_d2h: ast.Call | None = None
                paired_h2d: ast.Call | None = None
                for d2h_call in d2h_sorted:
                    for h2d_call in h2d_sorted:
                        if h2d_call.lineno <= d2h_call.lineno:
                            continue
                        if _in_mutually_exclusive_branches(d2h_call, h2d_call):
                            continue  # skip: branches can't co-execute
                        paired_d2h = d2h_call
                        paired_h2d = h2d_call
                        break
                    if paired_d2h is not None:
                        break
                if paired_d2h is not None and paired_h2d is not None:
                    first_d2h = paired_d2h.lineno
                    first_h2d = paired_h2d.lineno
                    is_suppressed = first_d2h in suppressions
                    if is_suppressed:
                        suppressed += 1
                        if not include_suppressed:
                            continue
                    errors.append(
                        LintError(
                            code="ZCOPY001",
                            path=path,
                            line=first_d2h,
                            message=(
                                f"Ping-pong transfer: D->H at line {first_d2h} followed by "
                                f"H->D at line {first_h2d} in {node.name}(). "
                                "Keep data on device; see execution_trace.py for runtime detection."
                            ),
                            doc_path=RUNTIME_DOC,
                            suppressed=is_suppressed,
                        )
                    )
    return errors, suppressed


# ---- ZCOPY002: Per-element device transfers in loops ----

def check_loop_transfers(
    repo_root: Path,
    *,
    include_suppressed: bool = False,
) -> tuple[list[LintError], int]:
    """Find D->H transfer calls inside for/while loop bodies.

    Returns ``(errors, suppressed_count)``.  When *include_suppressed* is
    True, suppressed violations appear in *errors* with ``suppressed=True``.
    """
    errors: list[LintError] = []
    suppressed = 0
    root = repo_root / "src" / "vibespatial"
    for path in iter_python_files(root):
        source = path.read_text(encoding="utf-8")
        suppressions = _parse_suppression_comments(source)
        tree = ast.parse(source, filename=str(path))

        # Pre-compute numpy name sets and pyarrow names per enclosing function.
        # For loops at module level, use empty sets (conservative).
        func_numpy: dict[int, frozenset[str]] = {}
        func_pyarrow: dict[int, frozenset[str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_numpy[id(node)] = _collect_numpy_names(node)
                func_pyarrow[id(node)] = (
                    _collect_pyarrow_names(node) if _scope_has_pyarrow(node) else frozenset()
                )

        # Assign parent links so we can look up enclosing function.
        _assign_parents(tree)

        for node in ast.walk(tree):
            if not isinstance(node, (ast.For, ast.While)):
                continue
            # Find enclosing function scope for numpy name tracking.
            numpy_names = _enclosing_func_numpy_names(node, func_numpy)
            pa_names = _enclosing_func_pyarrow_names(node, func_pyarrow)
            for descendant in ast.walk(node):
                if not isinstance(descendant, ast.Call):
                    continue
                if _is_d2h_call(descendant, numpy_names, pyarrow_names=pa_names):
                    is_suppressed = descendant.lineno in suppressions
                    if is_suppressed:
                        suppressed += 1
                        if not include_suppressed:
                            continue
                    errors.append(
                        LintError(
                            code="ZCOPY002",
                            path=path,
                            line=descendant.lineno,
                            message=(
                                f"D->H transfer ({_call_name(descendant)}()) inside a loop body. "
                                "Transfer in bulk outside the loop instead of per-element."
                            ),
                            doc_path=RUNTIME_DOC,
                            suppressed=is_suppressed,
                        )
                    )
    return errors, suppressed


# ---- ZCOPY003: Functions that accept device data but return host data ----

def _returns_host_conversion(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> int | None:
    """Return the line number of a D->H call in a return statement, or None."""
    numpy_names = _collect_numpy_names(func_node)
    pa_names = (
        _collect_pyarrow_names(func_node) if _scope_has_pyarrow(func_node) else frozenset()
    )
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        for descendant in ast.walk(node.value):
            if isinstance(descendant, ast.Call) and _is_d2h_call(
                descendant, numpy_names, pyarrow_names=pa_names
            ):
                return descendant.lineno
    return None


def _uses_device_apis(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Heuristic: does this function reference cupy / device APIs?"""
    for node in ast.walk(func_node):
        if isinstance(node, ast.Name) and node.id in {"cp", "cupy"}:
            return True
        if isinstance(node, ast.Attribute) and node.attr in {
            "launch", "compile_kernels", "device_array", "DeviceGeometryArray",
        }:
            return True
    return False


_ALLOWED_HOST_RETURN = {
    "to_pandas", "to_numpy", "values", "__repr__", "__str__",
    "to_pylist", "tolist", "to_host", "to_wkt", "to_wkb",
    "_to_host_array", "_ensure_host_state",
}


def check_boundary_type_leak(
    repo_root: Path,
    *,
    include_suppressed: bool = False,
) -> tuple[list[LintError], int]:
    """Find functions using device APIs that return D->H converted results.

    Returns ``(errors, suppressed_count)``.  When *include_suppressed* is
    True, suppressed violations appear in *errors* with ``suppressed=True``.
    """
    errors: list[LintError] = []
    suppressed = 0
    root = repo_root / "src" / "vibespatial"
    for path in iter_python_files(root):
        source = path.read_text(encoding="utf-8")
        suppressions = _parse_suppression_comments(source)
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name in _ALLOWED_HOST_RETURN or node.name.startswith("_"):
                continue
            if not _uses_device_apis(node):
                continue
            d2h_line = _returns_host_conversion(node)
            if d2h_line is not None:
                is_suppressed = d2h_line in suppressions
                if is_suppressed:
                    suppressed += 1
                    if not include_suppressed:
                        continue
                errors.append(
                    LintError(
                        code="ZCOPY003",
                        path=path,
                        line=d2h_line,
                        message=(
                            f"{node.name}() uses device APIs but returns host-converted data. "
                            "Return device arrays and let the caller materialize."
                        ),
                        doc_path=RUNTIME_DOC,
                        suppressed=is_suppressed,
                    )
                )
    return errors, suppressed


def _check_empty_suppressions(repo_root: Path) -> list[LintError]:
    """Emit errors for ``# zcopy:ok()`` with empty reason strings."""
    errors: list[LintError] = []
    root = repo_root / "src" / "vibespatial"
    for path in iter_python_files(root):
        source = path.read_text(encoding="utf-8")
        for lineno in _find_empty_suppression_lines(source):
            errors.append(
                LintError(
                    code="ZCOPY_EMPTY_SUPPRESSION",
                    path=path,
                    line=lineno,
                    message=(
                        "Suppression comment # zcopy:ok() has empty reason. "
                        "Provide a justification: # zcopy:ok(reason here)."
                    ),
                    doc_path=RUNTIME_DOC,
                )
            )
    return errors


def run_checks(repo_root: Path) -> tuple[list[LintError], int]:
    """Run all ZCOPY checks.

    Returns ``(errors, suppressed_count)``.
    """
    errors: list[LintError] = []
    total_suppressed = 0

    pingpong_errors, pingpong_suppressed = check_pingpong_transfers(repo_root)
    errors.extend(pingpong_errors)
    total_suppressed += pingpong_suppressed

    loop_errors, loop_suppressed = check_loop_transfers(repo_root)
    errors.extend(loop_errors)
    total_suppressed += loop_suppressed

    leak_errors, leak_suppressed = check_boundary_type_leak(repo_root)
    errors.extend(leak_errors)
    total_suppressed += leak_suppressed

    errors.extend(_check_empty_suppressions(repo_root))

    return (
        sorted(errors, key=lambda e: (str(e.path), e.line, e.code)),
        total_suppressed,
    )


def _collect_all_violations(repo_root: Path) -> list[LintError]:
    """Collect all violations including suppressed ones for ``--detail``.

    Reuses the same check functions as ``run_checks()`` with
    ``include_suppressed=True`` so analysis logic is never duplicated.
    """
    all_violations: list[LintError] = []

    pp_errs, _ = check_pingpong_transfers(repo_root, include_suppressed=True)
    all_violations.extend(pp_errs)

    loop_errs, _ = check_loop_transfers(repo_root, include_suppressed=True)
    all_violations.extend(loop_errs)

    leak_errs, _ = check_boundary_type_leak(repo_root, include_suppressed=True)
    all_violations.extend(leak_errs)

    all_violations.extend(_check_empty_suppressions(repo_root))

    return sorted(all_violations, key=lambda e: (str(e.path), e.line, e.code))


def _render_detail(
    all_violations: list[LintError],
    repo_root: Path,
) -> None:
    """Print all violations grouped by file, including suppressed ones."""
    if not all_violations:
        print("No violations found.")
        return

    current_file: Path | None = None
    for error in all_violations:
        if error.path != current_file:
            current_file = error.path
            relative = error.path.relative_to(repo_root)
            print(f"\n{relative}")
        print(f"  {error.render(repo_root)}")

    active = sum(1 for e in all_violations if not e.suppressed)
    suppressed = sum(1 for e in all_violations if e.suppressed)
    print(
        f"\nTotal: {active} active, {suppressed} suppressed, "
        f"{len(all_violations)} total"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check zero-copy device transfer constraints."
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Scan src/vibespatial/ for zero-copy violations.",
    )
    parser.add_argument(
        "--detail", action="store_true",
        help=(
            "Print every violation with file:line, code, and message "
            "regardless of whether count exceeds the baseline."
        ),
    )
    args = parser.parse_args(argv)

    if not args.all:
        parser.error("pass --all")

    errors, suppressed_count = run_checks(REPO_ROOT)
    count = len(errors)

    suppressed_msg = ""
    if suppressed_count:
        suppressed_msg = f" ({suppressed_count} suppressed via # zcopy:ok)"

    if args.detail:
        all_violations = _collect_all_violations(REPO_ROOT)
        _render_detail(all_violations, REPO_ROOT)
        print()

    if count > _VIOLATION_BASELINE:
        if not args.detail:
            for error in errors:
                print(error.render(REPO_ROOT))
        print(
            f"\nZero-copy checks FAILED: {count} violations found, "
            f"baseline is {_VIOLATION_BASELINE}. "
            f"New code introduced {count - _VIOLATION_BASELINE} violation(s)."
            f"{suppressed_msg}",
            file=sys.stderr,
        )
        return 1

    if count < _VIOLATION_BASELINE:
        print(
            f"Zero-copy checks passed ({count} known violations, baseline {_VIOLATION_BASELINE}). "
            f"Debt reduced! Update _VIOLATION_BASELINE to {count}.{suppressed_msg}"
        )
    else:
        print(
            f"Zero-copy checks passed ({count} known violations, baseline holds)."
            f"{suppressed_msg}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
