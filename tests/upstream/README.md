# Upstream Tests

<!-- DOC_HEADER:START
Scope: Upstream GeoPandas test vendoring and contract-coverage workflow.
Read If: You are refreshing, debugging, or expanding vendored upstream test coverage.
STOP IF: Your task does not touch vendored tests or compatibility contracts.
Source Of Truth: Vendored upstream test workflow for GeoPandas compatibility.
Body Budget: 75/200 lines
Document: tests/upstream/README.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-4 | Preamble |
| 5-9 | Intent |
| 10-19 | Request Signals |
| 20-27 | Open First |
| 28-34 | Verify |
| 35-40 | Risks |
| 41-46 | Purpose |
| 47-62 | Refresh |
| 63-75 | Current Bootstrap State |
DOC_HEADER:END -->

This directory contains vendored copies of the upstream GeoPandas test suite.

## Intent

Define how vendored GeoPandas tests are refreshed, used as a contract, and
verified locally.

## Request Signals

- upstream tests
- geopandas tests
- vendor
- fixture
- refresh tests
- collect-only
- compatibility contract

## Open First

- tests/upstream/README.md
- docs/testing/upstream-inventory.md
- scripts/vendor_geopandas_tests.py
- tests/upstream/geopandas/conftest.py
- pyproject.toml

## Verify

- `uv run python scripts/vendor_geopandas_tests.py`
- `uv run pytest --collect-only tests/upstream/geopandas`
- `uv run pytest tests/upstream/geopandas/tests/test_config.py`
- `uv run pytest --run-gpu tests/test_runtime_harness.py -m gpu` on GPU-capable machines

## Risks

- Hand-editing vendored files makes upstream refreshes harder to replay.
- Import rewrites can silently drift if the vendoring script changes without a smoke run.
- Collection-only success does not guarantee runtime parity for changed fixtures or helpers.

## Purpose

- Use the upstream tests as the compatibility contract for `vibeSpatial`.
- Keep the copied suite close to upstream so refreshes stay mechanical.
- Allow repo-local smoke and collection runs without mutating the source repo.

## Refresh

```bash
uv run python scripts/vendor_geopandas_tests.py
```

The vendoring script copies:

- `geopandas/conftest.py`
- `geopandas/tests`
- `geopandas/io/tests`
- `geopandas/tools/tests`

It also rewrites imports that assume `geopandas.tests.util` lives inside the
installed package so the vendored tree remains self-contained.

## Current Bootstrap State

The vendored tests now import the local `src/geopandas` compatibility package.
That package delegates to a vendored copy of upstream GeoPandas source in
`src/vibespatial/_vendor/geopandas` while `vibeSpatial` grows its own runtime
and kernel surface underneath.

The repo-level pytest harness also exposes:

- `cuda_available`: session fixture for CUDA runtime detection
- `dispatch_mode`: parametrized runtime mode fixture (`cpu` by default, `gpu`
  added with `--run-gpu` when available)
- `dispatch_selection`: resolved runtime selection for the current dispatch mode
