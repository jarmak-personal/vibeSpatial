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

# Full pre-commit suite (what the hook runs)
uv run ruff check
uv run python scripts/check_architecture_lints.py --all
uv run python scripts/check_zero_copy.py --all
uv run python scripts/check_perf_patterns.py --all
uv run python scripts/check_maintainability.py --all
```

## Release process

1. Bump version in `pyproject.toml` and `src/vibespatial/api/_version.py`
2. Update `CHANGELOG.md`
3. Create a GitHub Release with tag `v{version}` (e.g., `v0.1.0`)
4. The publish workflow builds the wheel and publishes to PyPI
