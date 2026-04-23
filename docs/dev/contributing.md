# Contributing

## Setup

```bash
git clone https://github.com/vibeSpatial/vibeSpatial.git
cd vibeSpatial
uv sync
```

## Code style

- Python 3.12+, enforced by `ruff` (E, F, W, I, UP, B, PERF, RUF rules)
- Line length: 100 characters
- Import sorting: `isort` via ruff, first-party = `vibespatial`, `geopandas`

## Pre-commit hooks

Install the git hooks to enforce code quality on every commit:

```bash
uv run python scripts/install_githooks.py
```

See [Pre-commit System](precommit.md) for the full enforcement architecture,
rule reference, and troubleshooting.

## Verification

Before landing changes, run the appropriate verification gate:

```bash
# Runtime/package changes
uv run pytest

# Pipeline/profiler changes
uv run pytest tests/test_pipeline_benchmarks.py -q

# Doc changes
uv run python scripts/check_docs.py --check

# Full deterministic pre-commit suite
uv run ruff check
uv run python scripts/check_architecture_lints.py --all
uv run python scripts/check_zero_copy.py --all
uv run python scripts/check_perf_patterns.py --all
uv run python scripts/check_maintainability.py --all

# Push-time heavy gate
uv run python scripts/health.py --tier contract --check
uv run python scripts/health.py --tier gpu --check
```

## Release process

1. Run the GitHub Actions `Release` workflow with the desired bump type
2. The workflow updates `src/vibespatial/_version.py`, tags `v{version}`, and creates the GitHub Release
3. GitHub Releases are the source of truth for release notes; this repo does not maintain a `CHANGELOG.md`
4. The publish workflow builds the wheel and publishes to PyPI
