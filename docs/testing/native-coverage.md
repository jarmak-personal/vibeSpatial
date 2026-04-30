# Native Coverage

<!-- DOC_HEADER:START
Scope: Strict native mode upstream test pass rate as the real GeoPandas compatibility metric.
Read If: You are measuring, reporting, or verifying strict native GeoPandas compatibility.
STOP IF: Your task already has the coverage script open and only needs the command.
Source Of Truth: Native coverage metric definition for upstream GeoPandas compatibility.
Body Budget: 86/220 lines
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
| 52-69 | Command |
| 70-86 | Notes |
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

- `VIBESPATIAL_STRICT_NATIVE=1 uv run python scripts/upstream_native_coverage.py --json`
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
VIBESPATIAL_STRICT_NATIVE=1 uv run python scripts/upstream_native_coverage.py
```

Use `--json` for machine-readable output.

For long sweeps, prefer chunked progress so the command does not sit silent on a
single giant pytest subprocess:

```bash
VIBESPATIAL_STRICT_NATIVE=1 uv run python scripts/upstream_native_coverage.py --grouped --group-by file --json
```

Progress streams pytest output to stderr; the final report still prints JSON to
stdout.

## Notes

- Missing optional dependencies such as `pyarrow`, `fiona`, or PostGIS drivers
  will usually increase `skipped`, not `failed`.
- Grouped sweeps treat a pytest return code 5 as success only when the parsed
  chunk has no failures or unexpected passes. This keeps optional-dependency
  all-skipped files from failing the coverage run.
- Timed-out chunks are reported as structured failures instead of Python
  tracebacks, so the JSON output remains usable for PRD gap triage.
- The runner invokes pytest with the current Python interpreter instead of
  nesting `uv run pytest` inside `uv run python`; this avoids false timeout
  chunks from environment/cache contention.
- `VIBESPATIAL_STRICT_NATIVE=1` must be present in the launch environment. GPU
  coverage chunks must not initialize the runtime before strict mode exists, so
  the runner fails fast instead of setting strict mode after Python startup.
- This metric is intentionally stricter than normal upstream green status,
  because host fallback does not count as native coverage.
