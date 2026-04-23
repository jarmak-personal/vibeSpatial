# Pre-commit System

<!-- DOC_HEADER:START
Scope: Pre-commit hook architecture, enforcement layers, and AI review workflow.
Read If: You are setting up the repo, debugging a pre-commit failure, or extending the enforcement system.
STOP IF: You only need to run a single lint check (see Verification in AGENTS.md).
Source Of Truth: Pre-commit enforcement policy and AI review workflow.
Body Budget: 200/220 lines
Document: docs/dev/precommit.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-6 | Overview |
| 7-12 | Quick Start |
| 13-41 | Layer 1 Deterministic Checks |
| 42-78 | Layer 2 AI Review Skill |
| 79-93 | Layer 3 PreToolUse Hook |
| 94-105 | How The Layers Interact |
| 106-137 | Ratchet Baseline System |
| 138-163 | Adding New Checks |
| 164-200 | Troubleshooting |
DOC_HEADER:END -->

The pre-commit system enforces three core principles through three defense
layers, each operating at a different level of the stack:

1. **Performance is king** -- no regressions in GPU-first execution
2. **Zero-copy by default** -- data stays on device until explicit materialization
3. **Agent-first discoverability** -- all code is routable through the intake system

## Quick Start

```bash
uv run python scripts/install_githooks.py
```

That's it. The git hooks classify the staged or pushed diff first. Pre-commit
runs the fast deterministic gate. Pre-push runs the expensive contract and GPU
health gate, with a cache keyed to the pushed ref update so a retry of the same
push does not rerun the full GPU sweep. The hook layer is configured via
`.claude/settings.json`; repo-local Codex skills live in `.agents/skills/`.

## Layer 1: Deterministic Checks (git pre-commit hook)

Runs on every commit for all contributors. No network, no AI. The hook starts
with `scripts/precommit_plan.py`, which inspects staged paths and chooses one
of two scopes:

- `docs-only`: staged paths are Markdown, `docs/` assets, or generated docs
  index artifacts. The hook refreshes and validates generated docs, then skips
  ruff, architecture, zero-copy, performance, and import-guard checks.
  Maintainability still runs so ADR/index drift stays visible.
- `full`: any staged code, tests, scripts, hooks, project configuration, or
  unknown file type. The hook runs the deterministic gate.

Pre-commit does not run contract or GPU health by default. Set
`VIBESPATIAL_PRECOMMIT_FORCE_GPU=1` to run the cached heavy gate before a local
commit attempt.

| Order | Check | Script | Rules |
|-------|-------|--------|-------|
| 0 | Staged path plan | `precommit_plan.py --scope` | Select docs-only or full gate |
| 1 | Ruff lint | `ruff check` | Full gate only; E, F, W, I, UP, B, PERF, RUF |
| 2 | Doc refresh | `check_docs.py --refresh` | Auto-refresh generated headers |
| 3 | Doc validate | `check_docs.py --check` | Header budgets, routing sections |
| 4 | Architecture | `check_architecture_lints.py --all` | Full gate only; ARCH001-006 |
| 5 | Zero-copy | `check_zero_copy.py --all` | Full gate only; ZCOPY001-003 (add `--detail` for per-violation breakdown during debt paydown) |
| 6 | Performance | `check_perf_patterns.py --all` | Full gate only; VPAT001-004 |
| 7 | Maintainability | `check_maintainability.py --all` | Docs-only and full gates; MAINT001-003 |
| 8 | Import guards | `check_import_guard.py --all` | Full gate only; IGRD001-002 |
| 9 | Forced heavy gate | `gpu_health_gate.py --staged --force` | Only when `VIBESPATIAL_PRECOMMIT_FORCE_GPU=1` |

### Pre-Push Heavy Gate

The pre-push hook (`.githooks/pre-push`) runs the expensive ratchets before
code leaves the local workstation:

| Order | Check | Script | Rules |
|-------|-------|--------|-------|
| 1 | Contract health | `health.py --tier contract --check` | Cached per pushed ref update |
| 2 | GPU health | `health.py --tier gpu --check` | Cached per pushed ref update |

Docs-only pushes skip the heavy gate unless `VIBESPATIAL_PUSH_FORCE_GPU=1` is
set. Successful heavy-gate runs are cached under `.git/` for 24 hours. Set
`VIBESPATIAL_GPU_GATE_IGNORE_CACHE=1` to force a rerun for the same diff, or
`VIBESPATIAL_GPU_GATE_CACHE_TTL_SECONDS=0` to disable cache hits.
The pre-push hook defaults `VIBESPATIAL_GPU_COVERAGE_WORKERS=auto` so the
upstream GPU coverage sweeps use `pytest-xdist`; set it to `1` if parallel GPU
workers cause local contention.

After checks pass, a non-blocking reminder prints the AI review commands.
This works in terminals, VS Code git output, and CI alike.

### Zero-Copy Rules (ZCOPY)

| Code | Rule | Example |
|------|------|---------|
| ZCOPY001 | No ping-pong D/H transfers in same function | `.get()` followed by `cp.asarray()` |
| ZCOPY002 | No per-element D/H transfers in loops | `.get()` inside a `for` body |
| ZCOPY003 | Functions using device APIs must not return host data | Public function calls `.get()` in return |
| ZCOPY_EMPTY_SUPPRESSION | Suppression comment must include a reason | `# zcopy:ok()` with empty parens |

#### Inline Suppression

Intentional D/H transfers (e.g., materialization boundaries, OOM-prevention
batching) can be suppressed with an inline comment:

```python
host_data = device_array.get()  # zcopy:ok(materialization boundary: DataFrame construction)
```

- The reason inside the parentheses is **required** — `# zcopy:ok()` with an
  empty reason emits `ZCOPY_EMPTY_SUPPRESSION` instead of suppressing.
- Suppressed lines are excluded from the violation count but reported separately
  in the summary output.
- The suppression count is tracked to ensure visibility of documented debt.

### Performance Rules (VPAT)

| Code | Rule | Example |
|------|------|---------|
| VPAT001 | No Python for-loops over geometry objects | `for g in series.geoms` |
| VPAT002 | No Shapely imports in kernel modules | `import shapely` in `kernels/` |
| VPAT003 | No `np.fromiter` in runtime code | `np.fromiter(gen, dtype=float)` |
| VPAT004 | No `.astype(object)` on arrays | `coords.astype(object)` |

### Maintainability Rules (MAINT)

| Code | Rule | Example |
|------|------|---------|
| MAINT001 | New modules in `src/vibespatial/` must be in intake | Missing from `intake-index.json` |
| MAINT002 | New ADRs must be indexed | Missing from `docs/decisions/index.md` |

## Layer 2: Pre-Land Review Skill (Codex, proactive)

The `pre-land-review` skill ([`.agents/skills/pre-land-review/SKILL.md`](/home/picard/repos/vibeSpatial/.agents/skills/pre-land-review/SKILL.md)) fires
**proactively** when Codex detects intent to commit or land work. Unlike
AGENTS.md instructions which can be compressed away in long conversations,
skills are matched against current context on every turn by their description
field. The skill fires on keywords like "commit", "land", "done", "ship it",
"wrap up", and "let's finish".

When triggered, the skill loads the full pre-land checklist fresh into
context, regardless of how long the conversation has been running.

### Workflow

1. Make your changes in a Codex session
2. When you're ready to land, say "commit" or "let's land this"
3. The skill fires automatically and runs the full checklist
4. If verdict is LAND, the skill writes `.agents/.review-completed` marker
5. `git commit` -- deterministic checks run, `commit-msg` hook verifies marker
6. Commit completes

### Available skills

You can also invoke the review skills explicitly:

| Skill | Purpose |
|---------|---------|
| `$pre-land-review` | Skill: full checklist, orchestrates deterministic checks plus AI review |
| `$gpu-code-review` | Skill: targeted GPU kernel review for kernel and dispatch changes |

### Suppressing the pre-commit reminder

The reminder is non-blocking. To suppress it:

```bash
VIBESPATIAL_SKIP_AI_REMINDER=1 git commit -m "message"
```

## Layer 3: PreToolUse Hooks (legacy Claude Code, mechanical)

Two `PreToolUse` hooks run **outside the LLM context window** -- they cannot
be compressed away, forgotten, or skipped by the model.

### Bash guard (`.claude/hooks/pre-land-gate.sh`)

Fires on every `Bash` tool call. Returns a **hard block** (`{"decision":"block"}`)
for commands that would bypass or tamper with enforcement infrastructure:

| Pattern | What it blocks |
|---------|---------------|
| `git commit --no-verify` / `-n` | Skipping all git hooks |
| `git config core.hooksPath` | Redirecting hooks away from `.githooks/` |
| `git -c core.hooksPath=...` | Inline hook path override |
| `rm`/`mv`/`chmod`/`sed`/... on `.githooks/` | Tampering with git hooks |
| `rm`/`mv`/`chmod`/`sed`/... on `.claude/hooks/` | Tampering with Claude hooks |
| `rm`/`mv`/`>` on `.claude/settings*.json` | Removing hook registrations |

For `git commit` commands that pass the above checks, the hook injects a
system message reminding the agent to complete `$pre-land-review` first.

### File guard (`.claude/hooks/file-guard.sh`)

Fires on every `Edit` and `Write` tool call. Returns a **hard block** for
writes to enforcement-critical paths:

| Protected path | Reason |
|----------------|--------|
| `.githooks/*` | Git hook scripts |
| `.claude/hooks/*` | Claude hook scripts |
| `.claude/settings.json` | Hook registrations |
| `.claude/settings.local.json` | Local hook overrides |
| `.agents/skills/pre-land-review/SKILL.md` | Review skill definition |

To modify any of these files, edit them manually outside the agent session.

### Configuration

Both hooks are registered in `.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      {"matcher": "Bash",  "hooks": [{"type": "command", "command": ".claude/hooks/pre-land-gate.sh"}]},
      {"matcher": "Edit",  "hooks": [{"type": "command", "command": ".claude/hooks/file-guard.sh"}]},
      {"matcher": "Write", "hooks": [{"type": "command", "command": ".claude/hooks/file-guard.sh"}]}
    ]
  }
}

## Layer 4: commit-msg Gate (Content-Addressable Marker)

The `commit-msg` hook (`.githooks/commit-msg`) is the **hard gate** for
AI-assisted commits. It checks the commit message for a
`Co-Authored-By: ... Codex` or `Co-Authored-By: ... Claude` line:

- **Present**: the review marker `.agents/.review-completed` must exist,
  be less than 1 hour old, contain a `staged_hash` that matches the current
  `git diff --cached | sha256sum`, and have a `verdict` of `LAND`.
  If any check fails, the commit is blocked.
- **Absent**: human-only commit, passes unconditionally.

The marker is a JSON file written by the `$pre-land-review` skill after a
LAND verdict. It binds the review to the exact staged diff via SHA-256 hash,
so a stub file (`touch .agents/.review-completed`) or a marker from a
different staging state will not pass. If you stage additional changes after
the review, the hash breaks and you must re-run `$pre-land-review`.

## How The Layers Interact

The four layers form a defense-in-depth chain:

```
Layer 2: Skill fires on "commit" / "land" / "done" intent
  --> Loads full checklist, Codex runs AI analysis
  --> Writes .agents/.review-completed marker on LAND verdict
    --> Layer 3: Bash guard hard-blocks --no-verify, hook tampering
    --> Layer 3: File guard hard-blocks Edit/Write to enforcement files
    --> Layer 3: Bash guard injects commit reminder if checks above pass
      --> Layer 1: Pre-commit hook runs deterministic checks
        --> Layer 4: commit-msg hook checks Co-Author + marker
          --> Blocks if Codex or Claude co-authored without review
```

Each layer catches what the previous one might miss:

| Failure mode | Caught by |
|--------------|-----------|
| Agent forgets to review before committing | Layer 2 (skill) |
| Long context compresses skill trigger away | Layer 3 (bash guard reminder) |
| Agent uses `--no-verify` to skip git hooks | Layer 3 (bash guard hard block) |
| Agent redirects `core.hooksPath` | Layer 3 (bash guard hard block) |
| Agent `rm`/`sed`/`chmod` on hook files via Bash | Layer 3 (bash guard hard block) |
| Agent edits hook files via Edit/Write tools | Layer 3 (file guard hard block) |
| Agent ignores skill and hook, commits anyway | Layer 4 (commit-msg gate) |
| Agent creates stub marker file (`touch ...`) | Layer 4 (missing JSON / hash) |
| Agent stages new changes after review | Layer 4 (hash mismatch) |
| Codex not available (human contributor) | Layer 1 (pre-commit, no gate) |
| Non-interactive environment (CI, VS Code) | Layer 1 (pre-commit, no gate) |

## Ratchet Baseline System

The ZCOPY, VPAT, and MAINT checks use a **ratchet** to handle pre-existing
violations without blocking new work:

- Each script has a `_VIOLATION_BASELINE` constant (the known debt count).
- The check **passes** if `current_count <= baseline`.
- The check **fails** if `current_count > baseline` (new violations introduced).
- When debt is paid down, the script prints a reminder to tighten the baseline.

This means:

- **New code** must comply from day one.
- **Existing debt** doesn't block commits but can only shrink, never grow.
- **Paying down debt** is encouraged with a visible nudge.

### Updating baselines

When a script says "Debt reduced! Update `_VIOLATION_BASELINE` to N":

```bash
# Find and update the constant in the script
grep -n _VIOLATION_BASELINE scripts/check_zero_copy.py
# Edit the number, commit the change
```

Do not increase baselines without documenting why (new code should comply,
not get an exemption).

## Adding New Checks

To add a new lint rule to an existing enforcer:

1. Add a `check_*` function in the appropriate `scripts/check_*.py`
2. Add it to the `run_checks()` function
3. Run the script -- if new violations appear, decide:
   - Fix them (preferred)
   - Increase the baseline with a comment explaining why
4. Update the rule table in this doc
5. Run `uv run python scripts/check_docs.py --refresh` to update headers

To add an entirely new enforcer:

1. Create `scripts/check_<name>.py` following the existing pattern
2. Add it to `.githooks/pre-commit` in the Layer 1 block
3. Add it to AGENTS.md project shape and verification sections
4. Add a corresponding Codex skill in `.agents/skills/` for AI-powered analysis
5. Document it in this file

## Troubleshooting

### Pre-commit hook not running

```bash
# Check that hooks are installed
git config --local core.hooksPath
# Should print: .githooks

# Reinstall if needed
uv run python scripts/install_githooks.py
```

### Deterministic check fails on pre-existing code

The ratchet baseline should prevent this. If it happens:

1. Run the failing script standalone to see all violations
2. Check if `_VIOLATION_BASELINE` was accidentally lowered
3. If a dependency update introduced new violations, raise the baseline
   with a comment explaining the cause

### Skill not firing in Codex

The skill triggers on keywords in the conversation. If it doesn't fire:

1. Say `$pre-land-review` explicitly to invoke it
2. Check that `.agents/skills/pre-land-review/SKILL.md` exists
3. Verify the skill appears in Codex's skill list

### Hook not injecting reminder

1. Check that `.claude/settings.json` has the `PreToolUse` hook configured
2. Verify `.claude/hooks/pre-land-gate.sh` is executable (`chmod +x`)
3. Test manually: `echo '{"command":"git commit -m test"}' | .claude/hooks/pre-land-gate.sh`

### Skipping hooks entirely

The `--no-verify` flag is **hard-blocked** for AI agents by the Bash guard
hook. It cannot be used from within the hooked agent sessions.

For human contributors working outside Codex, `--no-verify` still works
in a plain terminal. Use sparingly -- it bypasses ALL checks including
deterministic ones.
