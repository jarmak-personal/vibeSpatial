---
name: intake-router
description: "PROACTIVELY USE THIS SKILL to find relevant files and documentation before starting any work. Use it instead of exploring the codebase manually. Trigger on: any new task, request, question, bug, feature, investigation, or when you need to find code, docs, tests, scripts, kernels, configs, ADRs, architecture docs, or understand how something works. Also trigger on: \"where is\", \"find\", \"how does\", \"what file\", \"which module\", \"show me\", \"look up\", \"search for\", \"navigate to\", \"open\", \"locate\", \"explore\", \"investigate\", \"understand\", \"learn about\", \"read about\", \"check\", \"review\", \"audit\", \"debug\", \"fix\", \"modify\", \"update\", \"add\", \"implement\", \"create\", \"write\", \"build\", \"refactor\", \"test\", \"benchmark\", \"profile\", \"optimize\". This is the fastest way to find anything in the repo — always prefer it over manual grep/glob exploration."
---

# Intake Router

You have access to a smart document and file discovery system. **Use it first**
before exploring the codebase manually. It's faster and more accurate.

## How to Use

Run the intake router with a natural language description of what you need:

```bash
uv run python scripts/intake.py "<your request>"
```

### Examples

```bash
# Find code for a specific feature
uv run python scripts/intake.py "GPU kernel for point-in-polygon"
uv run python scripts/intake.py "how does the overlay pipeline work"
uv run python scripts/intake.py "binary predicate dispatch"

# Find docs for a concept
uv run python scripts/intake.py "precision dispatch fp32 fp64"
uv run python scripts/intake.py "device memory management"
uv run python scripts/intake.py "testing strategy for kernels"

# Find where to make changes
uv run python scripts/intake.py "add a new GPU kernel"
uv run python scripts/intake.py "fix a bug in spatial join"
uv run python scripts/intake.py "add a new ADR"

# Understand architecture
uv run python scripts/intake.py "GPU-first execution model"
uv run python scripts/intake.py "fallback behavior"
uv run python scripts/intake.py "how does the dispatch layer work"
```

## What It Returns

The router returns a structured plan with:

1. **Docs** — top 3 most relevant documentation files with:
   - Why they matched (which signals/tokens)
   - What to read them for (scope, readIf)
   - When to stop reading (stopIf)
2. **Files** — top 8 most relevant source/test/script files with:
   - Path and file type (source, test, script, doc)
   - Why they matched
3. **Verify** — commands to run for the matched area
4. **Risks** — known pitfalls for the matched area
5. **Confidence** — high/medium/low based on match quality

## When to Use This vs Other Tools

| Situation | Use |
|-----------|-----|
| Starting any new task | **Intake router** (first!) |
| Need to find a specific file by name | `Glob` tool |
| Need to find a specific string in code | `Grep` tool |
| Need broad codebase understanding | **Intake router** then read returned files |
| Need to understand an architecture decision | **Intake router** with "ADR" + topic |
| Debugging a specific error | **Intake router** with the error context |
| Don't know where to start | **Intake router** with the task description |

## How It Works

The router uses `docs/ops/intake-index.json` (auto-generated from doc headers)
to score your request against:
- **Request signals** (keywords in docs, weight 8x) — highest priority
- **Metadata** (scope, readIf, sourceOfTruth, weight 4x)
- **Titles** (weight 3x)
- **File paths and symbols** (weight 2x)

It returns the best matches without you needing to know the repo structure.

## After Routing

Once you have the plan:
1. Read the top-ranked doc first (it sets context)
2. Open the `open_first` files listed in that doc
3. Read only what's needed — the `stopIf` field tells you when to move on
4. Run the `verify` commands before expanding scope
5. Only then explore further if the routed files don't cover your need
