# Native Coverage

<!-- DOC_HEADER:START
Scope: Strict native mode upstream test pass rate as the real GeoPandas compatibility metric.
Read If: You are measuring, reporting, or verifying strict native GeoPandas compatibility.
STOP IF: Your task already has the coverage script open and only needs the command.
Source Of Truth: Native coverage metric definition for upstream GeoPandas compatibility.
Body Budget: 65/220 lines
Document: docs/testing/native-coverage.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-7 | Intent |
| 8-14 | Request Signals |
| 15-19 | Open First |
| 20-24 | Verify |
| 25-29 | Risks |
| 30-51 | Definition |
| 52-59 | Command |
| 60-65 | Notes |
DOC_HEADER:END -->

## Intent

Define the real GeoPandas compatibility percentage as upstream test pass rate
under strict native mode, where any GeoPandas fallback is treated as failure.

## Request Signals

- native coverage
- strict native
- compatibility percentage
- upstream pass rate

## Open First

- docs/testing/native-coverage.md
- scripts/upstream_native_coverage.py

## Verify

- `uv run python scripts/upstream_native_coverage.py --json`
- `uv run python scripts/check_docs.py --check`

## Risks

- Missing optional dependencies inflate skipped count, making the primary metric misleading.
- Host fallback counting as a pass in non-strict mode hides real coverage gaps.

## Definition

Run vendored upstream tests with `VIBESPATIAL_STRICT_NATIVE=1`.

In this mode:

- any explicit fallback event raises immediately
- skipped tests remain skips
- xfailed tests count as not passing

Primary metric:

- native pass rate = `passed / (passed + failed + xfailed + xpassed)`

Secondary metric:

- suite pass rate = `passed / (passed + failed + skipped + xfailed + xpassed)`

The primary metric is the one that should appear in commit messages because it
answers the question: "what fraction of the executed upstream GeoPandas tests
passed on repo-owned behavior with no fallback?"

## Command

```bash
uv run python scripts/upstream_native_coverage.py
```

Use `--json` for machine-readable output.

## Notes

- Missing optional dependencies such as `pyarrow`, `fiona`, or PostGIS drivers
  will usually increase `skipped`, not `failed`.
- This metric is intentionally stricter than normal upstream green status,
  because host fallback does not count as native coverage.
