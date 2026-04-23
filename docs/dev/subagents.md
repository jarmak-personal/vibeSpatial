# Codex Subagents

<!-- DOC_HEADER:START
Scope: Repo-managed Codex custom-agent source files and validation workflow.
Read If: You are wiring, editing, or validating repo-specific Codex subagents.
STOP IF: You only need normal skills or built-in `worker` / `explorer` roles.
Source Of Truth: Project-scoped custom-agent files in `.codex/agents/`.
Body Budget: 54/220 lines
Document: docs/dev/subagents.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-5 | Preamble |
| 6-10 | Intent |
| 11-19 | Request Signals |
| 20-26 | Open First |
| 27-31 | Verify |
| 32-39 | Risks |
| 40-54 | Workflow |
DOC_HEADER:END -->

Use project-scoped files under `.codex/agents/` for repo-specific Codex custom
agents.

## Intent

Keep custom-agent definitions concise, repo-reviewed, and versioned alongside
the codebase in the location Codex reads directly.

## Request Signals

- codex subagent
- custom agent
- cuda-engineer
- python-engineer
- fresh set of eyes
- repo-managed agent

## Open First

- docs/dev/subagents.md
- .codex/agents/cuda-engineer.toml
- .codex/agents/python-engineer.toml
- AGENTS.md

## Verify

- `uv run pytest tests/test_codex_subagents.py -q`
- `uv run python scripts/check_docs.py --check`

## Risks

- Letting agent instructions sprawl turns fresh eyes into prompt debt.
- Project-local `.codex` trees are often globally gitignored; keep
  `.codex/agents/*.toml` explicitly tracked.
- If a workstation overlays repo `.codex` with a separate mount, Codex may need
  an unsandboxed write to update the files.

## Workflow

Edit the repo-managed source files under `.codex/agents/`:

```bash
uv run python scripts/intake.py "update cuda engineer subagent instructions"
```

Validate them locally:

```bash
uv run pytest tests/test_codex_subagents.py -q
```

That path is the source of truth Codex reads for project-scoped custom agents.
