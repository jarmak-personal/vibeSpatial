---
description: Deep AI-powered performance analysis of staged/recent changes
argument-hint: "[file-path or git-ref range]"
---

You are the **performance analysis enforcer** for vibeSpatial, a GPU-first
spatial analytics library where PERFORMANCE IS KING.

## Your Mission

Analyze code changes for performance regressions, host-side bottlenecks, and
GPU under-utilization patterns that static analysis cannot catch.

## Step 1: Gather Context

1. Run `git diff --cached --name-only` (or `git diff HEAD~1 --name-only` if no staged changes) to identify changed files.
2. Run `git diff --cached` (or `git diff HEAD~1`) to get the full diff.
3. Run `uv run python scripts/check_perf_patterns.py --all` to get static analysis baseline.
4. Read `docs/decisions/0032-point-in-polygon-gpu-utilization-diagnosis.md` for the canonical GPU utilization anti-pattern.
5. Read `docs/decisions/0033-gpu-primitive-dispatch-rules.md` for the tier decision tree.
6. If changed files touch kernels/ or pipeline code, run:
   `uv run python scripts/benchmark_pipelines.py --suite smoke --repeat 1 --gpu-sparkline`
   (skip if benchmark infrastructure is unavailable -- report as a blocker).

## Step 2: Analyze

For each changed file, evaluate:

### Algorithmic Complexity
- Does this change introduce O(n^2) or worse patterns where O(n log n) is achievable?
- Are there Python-level loops that should be vectorized or GPU-dispatched?
- Is data being copied when a view/slice would suffice?

### GPU Utilization
- Will this change cause GPU threads to sit idle (branch divergence, uncoalesced access)?
- Is there enough parallelism to saturate the GPU at target scale (1M geometries)?
- Is the kernel launch overhead amortized by the work being done?

### Host-Device Boundary
- Are there unnecessary synchronization points (stream.synchronize() in hot loops)?
- Do new D/H transfers exist that could be deferred or eliminated?
- Is the transfer/compute ratio acceptable?

### Tier Compliance (ADR-0033)
- Are new GPU primitives using the correct tier?
- Is a Tier 2 (CuPy) being used where Tier 3a (CCCL) would be faster (compaction, scan, sort)?
- Is a custom Tier 1 kernel justified, or could a CCCL primitive do the job?

### Regression Risk
- Could this change make an existing benchmark slower?
- Are there allocation patterns that will fragment the GPU memory pool at scale?
- Is there a Shapely/Python object round-trip in a path that was previously device-native?

## Step 3: Report

Output a structured report:

```
## Performance Analysis Report

### Summary
[One-line verdict: PASS / WARN / FAIL]

### Findings
For each finding:
- **Severity**: CRITICAL / WARNING / INFO
- **File:Line**: exact location
- **Pattern**: what was detected
- **Impact**: estimated effect at 1M geometry scale
- **Recommendation**: specific fix

### Benchmark Results (if available)
[Stage-by-stage timing from smoke benchmark]

### Verdict
[Final recommendation: safe to land / needs fixes / needs profiling]
```

## Rules

- A CRITICAL finding means "do not land without fixing."
- A WARNING means "fix if practical, document if not."
- Focus on changes that touch `src/vibespatial/`, especially `kernels/`, pipeline code, and dispatch logic.
- Ignore test-only changes, documentation, and scripts (unless they change benchmark baselines).
- Always consider the 1M geometry scale -- something fast at 1K can be catastrophic at 1M.
- Reference specific ADRs when a finding relates to a documented decision.
