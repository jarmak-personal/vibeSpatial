# Contributing

## Setup

```bash
git clone https://github.com/vibeSpatial/vibeSpatial.git
cd vibeSpatial
uv sync
```

## Code style

- Python 3.12+, enforced by `ruff`
- Line length: 100 characters
- Run `uv run ruff check src/` before committing

## Pre-commit hooks

Install the git hooks for auto-refreshed doc headers:

```bash
uv run python scripts/install_githooks.py
```

## Verification

Before landing changes, run the appropriate verification gate:

```bash
# Runtime/package changes
uv run pytest

# Pipeline/profiler changes
uv run pytest tests/test_pipeline_benchmarks.py -q

# Doc changes
uv run python scripts/check_docs.py --check
```

## Release process

1. Bump version in `pyproject.toml` and `src/vibespatial/api/_version.py`
2. Update `CHANGELOG.md`
3. Create a GitHub Release with tag `v{version}` (e.g., `v0.1.0`)
4. The publish workflow builds the wheel and publishes to PyPI
