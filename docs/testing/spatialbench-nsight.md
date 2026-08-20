# SpatialBench Nsight Cross-Device Profiling

<!-- DOC_HEADER:START
Scope: Reproducible query-scoped Nsight Systems capture and RTX 4090 versus H200 comparison workflow for SpatialBench SF100.
Read If: You are profiling SF100 CPU/GPU balance, collecting datacenter traces, or comparing query execution across NVIDIA devices.
STOP IF: You only need clean benchmark wall times or an operation-local kernel microbenchmark.
Source Of Truth: Capture contract and operator workflow for scripts/profile_spatialbench_nsight.py.
Body Budget: 173/180 lines
Document: docs/testing/spatialbench-nsight.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-8 | Intent |
| 9-18 | Request Signals |
| 19-25 | Open First |
| 26-33 | Verify |
| 34-45 | Risks |
| 46-80 | Capture Contract |
| 81-115 | Permission Preflight |
| 116-143 | H200 And RTX 4090 Captures |
| 144-166 | Comparison |
| 167-173 | Nsight Compute Follow-Up |
DOC_HEADER:END -->

## Intent

Capture one measured public vibeSpatial SF100 query per Nsight Systems report,
after an untraced warmup, and produce machine-readable summaries that can be
compared across the RTX 4090 and H200 without relying on product-name guesses.

## Request Signals

- Nsight Systems
- SF100 profiling
- H200 profile
- RTX 4090 comparison
- CPU bottleneck
- GPU idle gap
- CUDA synchronization

## Open First

- `scripts/profile_spatialbench_nsight.py`
- `docs/dev/cross-device-performance-report.md`
- `docs/testing/profiling-rails.md`
- `benchmarks/spatialbench/run_benchmark.py`

## Verify

- `uv run ruff check scripts/profile_spatialbench_nsight.py tests/test_spatialbench_nsight.py`
- `uv run pytest tests/test_spatialbench_nsight.py -q`
- `.venv/bin/python scripts/profile_spatialbench_nsight.py preflight`
- run a Q9 smoke capture before starting a full SF100 capture
- `uv run python scripts/check_docs.py --check`

## Risks

- Wrapping the normal multiprocess benchmark runner mixes parent, fork, warmup,
  and measured work; use the direct query rail instead.
- Nsight instrumentation changes wall time. Compare like-for-like traces and
  retain clean benchmark timings separately.
- Missing CPU sampling or context switches weakens a CPU-bottleneck claim.
- Summed kernel duration can exceed wall time with concurrent streams. Use the
  reported union-based GPU busy fraction for utilization comparisons.
- Nsight Compute replay across an entire Q10 or Q11 is structurally too
  expensive; select kernels only after the Systems trace identifies them.

## Capture Contract

`profile_spatialbench_nsight.py capture` runs each query in its own process:

1. load only the vibeSpatial public query module
2. execute one untimed warmup in a disposable process, then exit that process
   before Nsight reserves device memory
3. start a fresh measured process and reset the vibeSpatial hotpath trace
4. let Nsight start capture on the `spatialbench.<query>.measured` NVTX range
5. execute the public query and synchronize inside that measured range
6. let Nsight stop capture when the NVTX range closes
7. normalize the already-computed result outside capture and hash it
8. tear down the benchmark and use the same hard process exit as the normal
   isolated SpatialBench worker, avoiding interpreter-global CUDA destructor
   ordering after all owned artifacts are closed

Every query directory contains the raw `.nsys-rep`, exported SQLite database,
capture stdout/stderr, standard CUDA/NVTX/OS-runtime CSV summaries, normalized
result, result hash, command, and compact JSON summary. The suite root contains
environment and source fingerprints, an incrementally updated suite summary,
and `SHA256SUMS` after capture completes.

The compact summary reports measured wall, union-based GPU busy/idle time,
largest GPU idle gap, summed kernel time, launch count, memcpy time/bytes,
CUDA synchronization API time, and top kernel/API/OS-runtime entries. Raw
traces remain authoritative for thread scheduling and call-stack inspection.
CUDA allocation-lifetime tracing is disabled by default because its bookkeeping
adds noise to allocator studies; CUDA memcpy events and sizes remain part of the
normal trace. Nsight/CUPTI itself also reserves enough device memory to change
admission for a near-capacity SF100 query on a 24 GiB card. The disposable
prewarm preserves compiled kernels and filesystem cache while ensuring the
measured process has the production allocator shape. If allocation tracing is
enabled for a separate study, that capture is not comparable to the default
suite.

## Permission Preflight

Run before renting or restarting a pod:

```bash
.venv/bin/python scripts/profile_spatialbench_nsight.py preflight \
  --output /tmp/spatialbench-nsight-preflight.json
```

A full CPU-bottleneck comparison requires
`cpu_sampling_available: true`. The local workstation currently has
`kernel.perf_event_paranoid=2` during the canonical local capture, enabling
process-tree CPU sampling and context-switch tracing. If a machine reports a
higher value, temporarily lower it only for the capture, then restore the exact
original value:

```bash
profile_paranoid_before=$(cat /proc/sys/kernel/perf_event_paranoid)
sudo sysctl -w kernel.perf_event_paranoid=2
.venv/bin/python scripts/profile_spatialbench_nsight.py preflight
# Run the capture here.
sudo sysctl -w kernel.perf_event_paranoid="$profile_paranoid_before"
```

Do not proceed with the full comparison until the second preflight reports CPU
sampling available. `--allow-partial` exists only for validating CUDA/NVTX
capture mechanics.

The RTX 4090 also reports `ERR_NVGPUCTRPERM` for sampled GPU hardware counters.
Those counters are optional because CUDA activity and union-based GPU busy are
still captured. Use `--gpu-metrics off` on both machines for the directly
comparable suite. A separate H200 capture may use `--gpu-metrics on` for GH100
SM/tensor/memory-engine telemetry without pretending that those counters exist
in the 4090 report.

## H200 And RTX 4090 Captures

First validate the full contract cheaply with Q9. Then capture Q1-Q12. Use the
same source capsule and GeoParquet hashes on both machines.

```bash
.venv/bin/python scripts/profile_spatialbench_nsight.py capture \
  --data-dir /path/to/sf100-geoparquet \
  --output-dir /path/to/results/nsight-smoke \
  --queries q9 --gpu-metrics off --timeout 1800

.venv/bin/python scripts/profile_spatialbench_nsight.py capture \
  --data-dir /path/to/sf100-geoparquet \
  --output-dir /path/to/results/nsight-sf100 \
  --queries all --gpu-metrics off --timeout 1800
```

Recommended output roots are:

```text
/workspace/results/h200-YYYY-MM-DD/nsight-sf100
benchmark_results/nsight/sf100/YYYY-MM-DD-rtx4090
```

On an ephemeral H200 host, upload the complete directory, including raw reports
and `SHA256SUMS`, before stopping the machine. Resume only into a new empty
output directory; this avoids mixing stale query traces into a suite.

## Comparison

After both `suite-summary.json` files are local:

```bash
.venv/bin/python scripts/profile_spatialbench_nsight.py compare \
  --baseline benchmark_results/nsight/sf100/YYYY-MM-DD-rtx4090/suite-summary.json \
  --candidate benchmark_results/nsight/sf100/YYYY-MM-DD-h200/suite-summary.json \
  --baseline-label "RTX 4090" --candidate-label "H200" \
  --output benchmark_results/nsight/sf100/YYYY-MM-DD-comparison
```

Comparison fails closed when source manifests, warmup/strict-native/GPU-metric
capture fields, or normalized result hashes differ. Nsight CLI version drift is
reported but does not block the normalized SQLite summary; retain both raw
reports and use matching CLI versions when inspecting them in the GUI.

For a query where the 4090 wins, low H200 GPU-busy fraction, larger idle gaps,
similar kernel work, and CPU sampling concentrated in Python/pandas/IO or host
orchestration supports the CPU-bottleneck hypothesis. High H200 GPU busy or
materially slower kernels falsifies that explanation and redirects the review
to GPU execution shape.

## Nsight Compute Follow-Up

Do not run Nsight Compute over the full suite. Use the Systems top-kernel list
to choose one or two dominant kernels from a representative host-bound query
and a compute-bound query, then capture a bounded launch count with the same
NVTX measured range. Record the exact kernel regex, launch index/count, section
set, and source fingerprint beside each `.ncu-rep` before comparing devices.
