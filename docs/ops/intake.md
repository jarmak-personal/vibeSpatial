# Intake Router

<!-- DOC_HEADER:START -->
> [!IMPORTANT]
> This block is auto-generated. Edit metadata in `docs/doc_headers.json`.
> Refresh with `uv run python scripts/check_docs.py --refresh` and validate with `uv run python scripts/check_docs.py --check`.

**Scope:** Operational intake workflow and request-to-doc routing policy.
**Read If:** You received a task request and need to decide what to open first.
**STOP IF:** You already have a confirmed work plan and relevant docs open.
**Source Of Truth:** Intake workflow and routing rules for agents in this repo.
**Body Budget:** 59/220 lines
**Document:** `docs/ops/intake.md`

**Section Map (Body Lines)**
| Body Lines | Section |
|---|---|
| 1-4 | Preamble |
| 5-9 | Intent |
| 10-19 | Request Signals |
| 20-26 | Open First |
| 27-31 | Verify |
| 32-37 | Risks |
| 38-47 | Route Families |
| 48-54 | How It Works |
| 55-59 | Notes |
<!-- DOC_HEADER:END -->

Use this file to classify requests before opening more of the repository.

## Intent

Route a request to the narrowest useful docs, files, and verification commands
without maintaining a large static route table.

## Request Signals

- intake
- routing
- docs-only
- workflow
- repo map
- verification plan
- bootstrap

## Open First

- docs/ops/intake.md
- AGENTS.md
- README.md
- pyproject.toml

## Verify

- `uv run python scripts/check_docs.py --check`
- `uv run python scripts/intake.py "<request>"`

## Risks

- Over-reading before classification slows work and hides the local contract.
- Generated headers or intake index can drift if doc changes are not refreshed.
- Cross-cutting requests still need judgment; intake should narrow the start, not replace engineering review.

## Route Families

- `bootstrap`: packaging, `uv`, repo setup, dependency groups, CI scaffolding
- `runtime`: execution mode selection, fallback visibility, CPU/GPU dispatch
- `upstream-tests`: vendored GeoPandas tests, import rewrites, fixture syncing
- `api-surface`: GeoSeries, GeoDataFrame, array behavior, pandas alignment
- `io`: file, arrow, parquet, SQL, and external format coverage
- `kernel`: CUDA kernels, CCCL integration, batching, memory layout
- `docs-only`: docs and agent workflow changes with no runtime delta

## How It Works

1. Generated doc headers carry scope, read/stop conditions, and section maps.
2. The intake script builds a routing index from those docs and the current repo file tree.
3. `Request Signals`, `Open First`, `Verify`, and `Risks` sections become machine-readable input.
4. New files enter the file index automatically; only docs need updates when workflow changes.

## Notes

- Route lightly. This repo should stay easier to traverse than `diaBot`.
- If the task smells cross-cutting, open architecture plus the touched module.
- Use vendored upstream tests as a contract, not as a dumping ground.
