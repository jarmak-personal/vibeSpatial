# Testing

## Running tests

```bash
# All owned tests
uv run pytest tests/ --ignore=tests/upstream

# Upstream GeoPandas contract tests
uv run pytest tests/upstream/

# GPU-only tests
uv run pytest tests/ --run-gpu
```

## Test categories

| Category | Location | Purpose |
|----------|----------|---------|
| Owned tests | `tests/test_*.py` | vibeSpatial-specific behavior |
| Upstream contract | `tests/upstream/geopandas/` | GeoPandas API compatibility |
| GPU tests | `@pytest.mark.gpu` | Require CUDA runtime |
| CPU fallback | `@pytest.mark.cpu_fallback` | Verify fallback behavior |

## Shapely oracle pattern

New kernel tests should compare GPU output against a Shapely reference:

```python
def test_my_kernel(dispatch_mode):
    input_geoms = [Point(0, 0), Point(1, 1)]

    # GPU/CPU result
    result = my_kernel(input_geoms, dispatch_mode=dispatch_mode)

    # Shapely reference
    expected = [shapely.my_op(g) for g in input_geoms]

    assert_matches_shapely(result, expected)
```

## Strict native mode

Run tests with `VIBESPATIAL_STRICT_NATIVE=1` to verify no CPU fallbacks
occur. Any fallback raises immediately:

```bash
VIBESPATIAL_STRICT_NATIVE=1 uv run pytest tests/upstream/
```
