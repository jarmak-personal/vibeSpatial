# Repo Health Metrics: Proposal (Draft)

Status: draft for discussion, not yet adopted.

## 1. Problem

`scripts/health.py` today produces a report whose headline is misleading for a
repo of this shape. It runs a 6-file smoke suite but reports coverage across
all of `src/vibespatial` + `src/geopandas`, yielding numbers like `coverage:
9.25%`. That single number drives the "repo health" perception even though:

- the 6-file suite was never intended to cover the full tree,
- GPU correctness, fallback rates, and zero-copy posture are the metrics that
  actually matter for this project,
- the repo already has a property dashboard for the things we say we care
  about, but that signal is disconnected from the printed health headline,
- CPU-only upstream slices fail in ways that are expected (not yet supported),
  but currently read as red,
- there is no trend history, so a regression and a steady-state look identical.

Net: the current health surface is not truthful, not scoped, not
decision-driving, and not trendable. It penalizes us for things we're not
trying to be good at, and hides the things we are.

## 2. Design principles

Good health metrics for this repo should be:

1. **Truthful.** No false green (smoke pass ≠ repo healthy) and no false red
   (CPU-only upstream failure ≠ repo broken).
2. **Scoped.** Each metric must state what it covers and what it explicitly
   does not. Unmeasured is a first-class state, distinct from failing.
3. **Decision-driving.** A failing metric should point to the next fix, not
   just assert "something is bad."
4. **Trendable.** A reader should be able to tell whether things are getting
   better or worse without running archaeology on git log.
5. **Compositional.** Reuse the existing property dashboard and ratchet checks
   where possible; do not build a third parallel definition of repo health.

## 3. Tiered health model

Split the single `health.py` output into four tiers. Each tier answers a
different question and runs in a different place.

### 3.1 Bootstrap health (fast, local, always-on)

Question: *can I trust this checkout?*

Runs in under 60s on a laptop with no GPU. Stays close to what `health.py`
already does, minus the misleading coverage headline.

Reports:

- smoke suite pass / total (the current `HEALTH_TEST_SUITE`)
- lint + architecture lints
- doc hygiene
- version/packaging consistency
- compact property summary from `scripts/property_dashboard.py`

Does **not** report:

- repo-wide coverage %
- upstream slice results
- live GPU runtime metrics

Bootstrap should also be explicit about `unmeasured` and `configured` states.
If a benchmark rail, transfer audit, or dispatch policy is only configured but
not actually exercised by the current run, report that honestly instead of
showing `PASS`.

### 3.2 Contract health (maintained surfaces)

Question: *are the surfaces we claim to support still working?*

Runs a curated set of upstream slices, grouped by product surface. Each
surface has an owner, an executable command, and explicit allowed states.

Reports per surface:

- pass / total
- delta vs last recorded baseline
- required vs optional
- explicit `unsupported` state for surfaces we're not yet targeting (CPU-only
  paths, for instance)
- explicit `skipped-no-gpu` where a surface requires a visible NVIDIA runtime

### 3.3 GPU health (first-class)

Question: *is the GPU path doing what we think it's doing?*

Only runs on machines with a visible NVIDIA runtime. Skipped — not failed —
elsewhere.

Reports:

- GPU available: yes/no
- GPU acceleration rate on maintained GPU suite
- fallback rate on maintained GPU suite (`fallbacks / total dispatches`) plus
  categorized reasons
- zero-copy violations (from `check_zero_copy.py`)
- selected runtime vs requested runtime on benchmark rails
- end-to-end pipeline benchmark deltas vs baseline

These should be the hero numbers on the top of the printed report. Prefer
rates to raw counts for runtime telemetry so runs of different sizes remain
comparable. Raw counts are still useful as supporting detail.

### 3.4 Release health

Question: *if we cut a release today, would it be clean?*

Runs in CI on tag candidates. Reports:

- single-source-of-truth version consistency
- packaging build success
- doc build success
- changelog present for this version
- no unstaged edits to generated artifacts

## 4. Surface matrix

Contract health is driven by a checked-in matrix. Proposed starting file:
`scripts/health_surfaces.toml` (TOML to match the rest of the repo's
configuration style).

Shape:

```toml
[[surface]]
name = "versioning"
owners = ["src/vibespatial/_version.py", "pyproject.toml"]
command = "uv run pytest -q tests/test_version_consistency.py"
required = true
runtime = "any"
allowed_states = ["pass", "fail"]
baseline_key = "contract.versioning"

[[surface]]
name = "shim"
owners = ["src/geopandas"]
command = "uv run pytest -q tests/test_geopandas_shim.py"
required = true
runtime = "any"
allowed_states = ["pass", "fail", "unsupported"]
baseline_key = "contract.shim"

[[surface]]
name = "overlay"
owners = ["src/vibespatial/api/tools/overlay.py"]
command = "uv run pytest -q tests/test_overlay_api.py tests/upstream/geopandas/test_overlay.py"
required = true
runtime = "gpu-preferred"
allowed_states = ["pass", "fail", "unsupported", "skipped-no-gpu"]
baseline_key = "contract.overlay"
```

Surfaces I would seed the matrix with, in priority order:

1. versioning / packaging
2. GeoPandas shim
3. overlay
4. clip
5. Arrow / Parquet
6. IO file layer
7. GPU dispatch correctness
8. fallback observability
9. performance rails

Ship (1)–(3) first. The rest land as the matrix proves out.

The important design choice here is that the matrix should describe how to run
and classify a surface, not just which files feel related to it. Commands,
runtime expectations, and allowed states scale better than file lists once the
suite gets larger.

## 5. What the printed report should look like

Replacing the current `print_summary` with something shaped like:

```
Repo Health — bootstrap: PASS

Property summary: 6/6 clean, total distance 0.00

GPU health: PASS (gpu available: true)
  acceleration rate:     87.4%   (baseline 86.1%, +1.3)
  fallback rate:         1.2%    (12 / 1000 dispatches, baseline 1.4%, -0.2)
  zero-copy violations:  0

Maintained surfaces (contract):
  versioning           PASS    1/1
  shim                 PASS    42/42
  overlay              FAIL    134/218   (baseline 130/218, +4)
  arrow/parquet        FAIL    52/64     (baseline 52/64,   no change)

Configured but unmeasured:
  profile rails:       configured
  transfer audit:      configured
  dispatch policy:     configured

Release integrity:    PASS
```

The contrast with today's `coverage: 9.25% / tests: 23 passed` is the point.

## 6. Things to stop measuring (or rescope)

- **Repo-wide coverage %.** Either drop it entirely from the headline, or
  replace with *changed-surface coverage for touched modules* computed off the
  diff.
- **CPU-only upstream failures as red.** Move to `unsupported` until we
  commit to CPU-only parity.
- **"23 tests passed" as a standalone signal.** Always report pass/total with
  surface context.
- **Placeholder greens.** "Files exist" or "policy defined" should report as
  `configured`, not `PASS`.
- **Raw runtime counts without denominators.** Prefer rates for acceleration
  and fallback telemetry, with counts as supporting detail.

## 7. Ratchets, not perfection

Rather than demanding every surface go green at once, require:

- no regression in maintained surfaces (deltas must be `<= 0` for failures)
- versioning / release integrity always green
- GPU health trend non-decreasing (acceleration rate, fallback rate)
- specific failing surfaces shrink over time

Implementation: a committed `.health-baseline.json` plus a
`--update-baseline` flag on `health.py`. Baseline updates land in their own
commits so the ratchet movement is legible in git log.

## 8. Trend persistence

Bootstrap and GPU tiers should emit a JSON artifact per run. CI uploads the
artifact and a small reporter renders the last N runs as a chart in the
README or a dashboard page. Without this, "trendable" stays aspirational.

Minimum viable persistence:

- CI workflow writes `health-report-<sha>.json` as an artifact
- a follow-up script (`scripts/health_trend.py`) consumes the last N artifacts
  and emits a summary table

## 9. Sequencing (strong recommendation: do this as small tasks)

**Task 1 — Honest bootstrap.**
- Remove repo-wide coverage % from the headline in `health.py`.
- Add a one-line property summary from `scripts/property_dashboard.py`.
- Promote the existing GPU acceleration block near the top of `print_summary`.
- Replace placeholder greens with `configured` / `unmeasured`.

**Task 2 — Surface matrix MVP.**
- Add `scripts/health_surfaces.toml` with 3 surfaces: versioning, shim,
  overlay.
- Extend `health.py` with `--tier={bootstrap,contract,gpu,release}` so one
  entrypoint covers all tiers.
- Implement `--tier=contract` to consume the matrix and emit per-surface
  pass/total.

**Task 3 — Baselines and ratchets.**
- Commit `.health-baseline.json`.
- Add `--check` logic that fails CI on regressions, not on absolute failures.
- Add `--update-baseline` flow.

**Task 4 — Trend persistence.**
- Emit JSON artifacts from CI.
- Add `scripts/health_trend.py` for last-N summaries.

**Task 5+ — Expand surfaces.**
- Fill the rest of the matrix (clip, arrow/parquet, IO, GPU correctness,
  fallback observability, perf rails).

## 10. Explicit pushbacks on the original proposal

- **One entrypoint, not two.** Prefer a `--tier` flag on `health.py` over a
  new `scripts/health_contract.py`. Less surface area, less drift.
- **Use the existing property dashboard.** Bootstrap health should summarize
  `scripts/property_dashboard.py`, not fork its logic into a new parallel
  implementation.
- **TOML, not JSON, for the matrix.** Matches existing repo config style.
- **Ratchets need a committed baseline file.** Without that, "ratchet" becomes
  folklore and the first red CI run gets waved through.
- **Don't block Task 1 on matrix design.** Fixing the misleading coverage
  headline and placeholder greens is most of the immediate win.

## 11. Open questions

- Is the long-term goal CPU parity with upstream GeoPandas, or GPU-only with
  a CPU fallback for a narrow subset? The answer changes what counts as
  "unsupported" vs "compatibility debt."
- Do we want surface ownership in the matrix to map to GitHub CODEOWNERS so
  regressions route automatically?
- Where does the performance rail live in the tier model — under GPU health,
  or as its own tier 5 ("perf health")? I've kept it under GPU health for
  now; splitting it out is defensible.
- How often does the baseline get updated — per task, per release, or on
  demand? I'd default to "on demand, in its own commit."

## 12. Bluntest one-liner

Excellent health metrics come from measuring the repo you are actually
building, not the repo you vaguely aspire to have.
