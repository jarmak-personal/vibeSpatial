# Cross-Device Performance Report

<!-- DOC_HEADER:START
Scope: Consolidated RTX 4090 and H200 performance, correctness, and capacity evidence for the SF100, public shootout, point-region, and vibeProj validation campaign.
Read If: You need the complete measurement ledger for the 2026-08-18 local through 2026-08-20 H200 validation runs.
STOP IF: You need implementation details for point-region refinement or the historical pylibcudf execution plan.
Source Of Truth: Human-readable consolidated report; raw artifacts remain authoritative for exact machine-readable values.
Body Budget: 180/180 lines
Document: docs/dev/cross-device-performance-report.md

Section Map (Body Lines)
| Body Lines | Section |
|---|---|
| 1-2 | Preamble |
| 3-8 | Intent |
| 9-17 | Request Signals |
| 18-25 | Open First |
| 26-30 | Verify |
| 31-39 | Risks |
| 40-57 | Scope And Method |
| 58-146 | Consolidated Measurements |
| 147-180 | Interpretation And Evidence |
DOC_HEADER:END -->

## Intent

Provide one compact, durable table for every material measurement in the
consumer-versus-datacenter validation campaign, including unsuccessful or
unadmitted capacity attempts.

## Request Signals

- cross-device performance
- H200 validation
- RTX 4090 comparison
- SF100 results
- shootout measurements
- vibeProj throughput

## Open First

- `docs/dev/point-region-execution-evidence.md`
- `docs/dev/evidence-first-point-region-execution-plan.md`
- `docs/dev/pylibcudf-sf100-execution-plan.md`
- `benchmark_results/point_region/`
- `benchmark_results/spatialbench/sf100/`

## Verify

- `uv run python scripts/check_docs.py --check`
- `uv run python scripts/intake.py "cross-device performance report"`

## Risks

- Comparing different repetition contracts as though they were release-grade
  paired trials can overstate small cross-machine differences.
- A suite total can hide query-specific host or GPU bottlenecks.
- Treating the 10M fixture-setup timeout as a vibeSpatial capacity result would
  misattribute benchmark-harness work to the library.
- Raw artifacts can become stale after later runtime or kernel changes.

## Scope And Method

This report consolidates the measurements available after the 2026-08-20 H200
validation and records absent or invalid measurements instead of estimates.

- Local consumer system: NVIDIA RTX 4090, 24 GiB, compute capability 8.9, 128
  SMs, and Intel Core i9-13900K. Dataset and results share an ext4 WD_BLACK
  SN770 1 TB NVMe (`/dev/nvme0n1p2`).
- Datacenter system: NVIDIA H200, 143,771 MiB, compute capability 9.0, 132 SMs,
  and 700 W; clean runs had 12 vCPUs and profiling a shared 20.4-core quota.
- SF100 uses the same 38.096 GB GeoParquet dataset and normalized SQL-derived
  result contract on both systems.
- No GeoPandas or other non-vibeSpatial baseline was run on the billed H200.
- `--` means not measured. Times are seconds unless another unit is shown.
- Local optimized GeoPandas and initial vibeSpatial values are warmup plus
  median-of-three. Current local/H200 suites are one warmup plus one measured
  run, so cross-machine ratios remain directional rather than release trials.

## Consolidated Measurements

| Suite | Workload or metric | Reference or before | RTX 4090 current | H200 current | Outcome |
|---|---|---:|---:|---:|---|
| SF100 | Q1 | GPD-opt 106.03; VS-old 12.39 | 12.43 | 23.75 | Current VS speedup vs local GPD: 8.53x local, 4.46x H200 pod |
| SF100 | Q2 | GPD-opt 113.25; VS-old 7.84 | 7.82 | 12.15 | 14.48x local; 9.32x H200-pod vs local GPD |
| SF100 | Q3 | GPD-opt 405.03; VS-old 12.86 | 12.73 | 23.00 | 31.82x local; 17.61x H200-pod vs local GPD |
| SF100 | Q4 | GPD-opt 231.98; VS-old 8.55 | 8.49 | 9.88 | 27.32x local; 23.48x H200-pod vs local GPD |
| SF100 | Q5 | GPD-opt 834.69; VS-old 126.18 | 17.46 | 31.95 | 47.81x local; 26.12x H200-pod vs local GPD; major reusable native-path gain |
| SF100 | Q6 | GPD-opt 371.42; VS-old 21.13 | 16.25 | 21.53 | 22.86x local; 17.25x H200-pod vs local GPD |
| SF100 | Q7 | GPD-opt 337.44; VS-old 4.23 | 4.20 | 6.64 | 80.34x local; 50.82x H200-pod vs local GPD |
| SF100 | Q8 | GPD-opt 285.06; VS-old 16.94 | 16.95 | 11.07 | 16.82x local; H200 is 1.53x faster than 4090 current |
| SF100 | Q9 | GPD-opt 0.19; VS-old 0.14 | 0.15 | 0.19 | 1.27x local; 1.00x H200-pod vs local GPD; exempt from the per-query 10x goal |
| SF100 | Q10 | GPD-opt 1,738.00; VS-old 143.33 | 125.84 | 68.58 | 13.81x local; H200 is 1.83x faster than 4090 current |
| SF100 | Q11 | GPD-opt 3,127.50; VS-old 268.90 | 237.59 | 107.16 | 13.16x local; H200 is 2.22x faster than 4090 current |
| SF100 | Q12 | GPD-opt 626.64; VS-old 21.28 | 21.15 | 22.14 | 29.63x local; H200 and 4090 are within 5% |
| SF100 | Q1-Q12 total | GPD-opt 8,177.23; VS-old 643.77 | 481.06 | 338.04 | Current local VS is 25.3% faster than VS-old and 17.00x GPD; H200 is 1.42x faster than 4090 current |
| SF100 | Accuracy | SQL-derived oracle: 12/12 | 12/12 | 12/12 | H200: eight byte-identical; Q5/Q6/Q9/Q12 preserve rows, keys, order, and counts with max relative numeric delta below 3e-7 |
| SF100 Nsight | Q1 wall / GPU busy / max idle | -- | 13.114 / 13.7% / 470 ms | 67.830 / 2.2% / 3,298 ms | Kernel time unchanged at 0.98x; H200 wall is host-contended |
| SF100 Nsight | Q2 wall / GPU busy / max idle | -- | 8.654 / 55.4% / 411 ms | 62.122 / 5.4% / 1,117 ms | H200 kernels 1.37x faster; wall is host-contended |
| SF100 Nsight | Q3 wall / GPU busy / max idle | -- | 13.514 / 15.3% / 482 ms | 26.029 / 5.0% / 1,827 ms | H200 kernels 1.41x faster; wall is host-contended |
| SF100 Nsight | Q4 wall / GPU busy / max idle | -- | 9.271 / 67.6% / 457 ms | 16.625 / 27.5% / 2,643 ms | H200 kernels 1.31x faster; wall is host-contended |
| SF100 Nsight | Q5 wall / GPU busy / max idle | -- | 23.769 / 26.9% / 4,942 ms | 154.731 / 28.0% / 14,013 ms | H200 kernels 1.11x faster; severe host/FUSE perturbation |
| SF100 Nsight | Q6 wall / GPU busy / max idle | -- | 17.216 / 54.0% / 411 ms | 45.977 / 18.0% / 486 ms | H200 kernels 1.07x faster; wall is host-contended |
| SF100 Nsight | Q7 wall / GPU busy / max idle | -- | 4.900 / 49.9% / 484 ms | 68.727 / 2.5% / 3,776 ms | H200 kernels 1.23x faster; wall is host-contended |
| SF100 Nsight | Q8 wall / GPU busy / max idle | -- | 17.666 / 83.3% / 427 ms | 52.526 / 10.7% / 1,826 ms | H200 kernels 2.65x faster; host stalls erase the trace-wall gain |
| SF100 Nsight | Q9 wall / GPU busy / max idle | -- | 0.733 / 11.9% / 412 ms | 1.328 / 6.8% / 571 ms | Process envelope dominates; use clean 0.15/0.19 s timings |
| SF100 Nsight | Q10 wall / GPU busy / max idle | -- | 127.523 / 88.4% / 411 ms | 73.456 / 55.8% / 597 ms | H200: 1.74x trace-wall and 2.76x kernel speedup |
| SF100 Nsight | Q11 wall / GPU busy / max idle | -- | 238.269 / 94.1% / 416 ms | 108.374 / 72.1% / 548 ms | H200: 2.20x trace-wall and 2.87x kernel speedup |
| SF100 Nsight | Q12 wall / GPU busy / max idle | -- | 24.759 / 62.5% / 2,088 ms | 56.736 / 11.6% / 10,459 ms | H200 kernels 2.39x faster; severe host/FUSE perturbation |
| Q11 public | Clean-HEAD wall | -- | 311.46 | 116.43 | Same clean commit and public API on both devices |
| Q11 public | Classification-once wall | Clean 311.46 / 116.43 | 238.32 | 107.16 | Gain: 23.5% (1.31x) on 4090; 8.0% (1.09x) on H200 |
| Q11 profile | Exact candidates | 11.266 B | 7.981 B | -- | -29.2% |
| Q11 profile | Exact-kernel time | 290.165 | 210.904 | -- | -27.3% |
| Q11 profile | Instrumented wall | 330.21 | 252.41 | -- | -23.6% |
| Q11 profile | Prepared-index builds | 5 | 5 | -- | Unchanged; each group built once |
| Q11 profile | Peak pool live | 11.690 GB | 11.656 GB | -- | -0.3% |
| Q11 profile | Pool reserved | 17.044 GB | 17.044 GB | -- | Unchanged |
| Protected shapes | Simple short polygon, median | Local change -0.68% | 8.570 ms | 10.231 ms -> 10.258 ms | Fresh local production-auto capture; H200 +0.27% |
| Protected shapes | Long selected bin, median | Local change +0.14% | 99.123 ms | 23.051 ms -> 23.123 ms | Fresh local production-auto capture; H200 +0.31% |
| Protected shapes | Multipart envelope skew, median | Local change +0.26% | 16.307 ms | 11.718 ms -> 11.761 ms | Fresh local production-auto capture; H200 +0.37% |
| Protected shapes | Uniform many small polygons, median | Local change +0.38% | 18.092 ms | 17.502 ms -> 17.608 ms | Fresh local production-auto capture; H200 +0.61% |
| Protected shapes | Four-shape subtotal | -- | -- | 62.501 ms -> 62.751 ms | +0.40%; no Hopper protected-shape regression |
| Public 10K | accessibility_redevelopment | GPD 0.210545 | 0.351002 | -- | Matching fingerprint |
| Public 10K | corridor_flood_priority | GPD 0.162367 | 0.262395 | -- | Matching fingerprint |
| Public 10K | emergency_response_catchments | GPD 0.090655 | 0.201251 | -- | Matching fingerprint |
| Public 10K | flood_exposure | GPD 0.038724 | 0.026012 | -- | Matching fingerprint |
| Public 10K | habitat_corridor_compliance | GPD 0.131367 | 0.252352 | -- | Matching fingerprint |
| Public 10K | insurance_flood_screening | GPD 0.041132 | 0.205596 | -- | Matching fingerprint |
| Public 10K | nearby_buildings | GPD 0.093341 | 0.053557 | -- | Matching fingerprint |
| Public 10K | network_service_area | GPD 0.096280 | 0.093199 | -- | Matching fingerprint |
| Public 10K | parcel_zoning | GPD 0.069289 | 0.118157 | -- | Matching fingerprint |
| Public 10K | redevelopment_screening | GPD 0.725152 | 0.326358 | -- | Matching fingerprint |
| Public 10K | retail_trade_area_screening | GPD 0.646799 | 0.234371 | -- | Matching fingerprint |
| Public 10K | site_suitability | GPD 0.691100 | 0.173461 | -- | Matching fingerprint |
| Public 10K | transit_service_gap | GPD 0.239643 | 0.178505 | -- | Matching fingerprint |
| Public 10K | vegetation_corridor | GPD 0.310681 | 0.186180 | -- | Matching fingerprint |
| Public 10K | 14-workflow subtotal | Prior GPD 3.547737; prior VS 2.672819 | GPD 3.547077; VS 2.662395 | -- | 14/14 fingerprints; VS 0.39% faster than prior capture |
| Public 1M | accessibility_redevelopment | Prior 4.420968 | 4.399125 | -- | VS-only timing recorded |
| Public 1M | corridor_flood_priority | Prior 1.305615 | 1.297462 | -- | Timed leg stable; audit leg records one observable mixed-family buffer CPU fallback |
| Public 1M | emergency_response_catchments | Prior 2.366978 | 2.300236 | -- | 2.8% faster |
| Public 1M | flood_exposure | Prior 0.165967 | 0.166406 | -- | Stable |
| Public 1M | habitat_corridor_compliance | Prior 38.068339 | 38.239832 | -- | +0.45% single-run variation |
| Public 1M | insurance_flood_screening | Prior 1.683776 | 1.673654 | -- | Stable |
| Public 1M | nearby_buildings | Prior 0.660575 | 0.660513 | -- | Stable |
| Public 1M | network_service_area | Prior 0.294260 | 0.300783 | -- | +6.5 ms; +2.2% single-run variation |
| Public 1M | parcel_zoning | Prior 0.451732 | 0.452252 | -- | Stable |
| Public 1M | redevelopment_screening | Prior 404.981677 | 405.238549 | -- | +0.06%; unchanged fingerprint |
| Public 1M | retail_trade_area_screening | Prior 7.011348 | 7.008006 | -- | Stable; unchanged fingerprint |
| Public 1M | site_suitability | Prior 4.141638 | 4.144770 | 6.025949 | Local stable; H200 slower, consistent with host/orchestration dominance |
| Public 1M | transit_service_gap | Prior 60.832563 | 60.436220 | -- | -0.65%; timed result valid; separate telemetry replay exceeded the 24 GiB pool |
| Public 1M | vegetation_corridor | Prior 50.950733 | 51.585557 | -- | +1.25% single-run variation |
| Public 1M | 14-workflow VS subtotal | Prior 577.336170 | 577.903366 | -- | +0.10%; GPD intentionally suppressed, so harness status remains error despite valid VS timing legs |
| Public 10M | site_suitability | Old harness never reached VS | Not admitted: 20.17/21.16 GiB pool plus 2.58 GiB request | Old H200 run never reached VS | Targeted fixture setup preserves 10K/1M fingerprints and now exposes a genuine 24 GiB capacity limit; rerun required on H200 |
| Full 1M profile | Pipeline outcomes | Prior contract | 11 ok; 1 raster deferment | -- | Zero compute D2H, zero compute materialization, zero fallback; max pipeline 0.201 s and max stage 0.075 s |
| Binary predicate | Public maintained benchmark | Prior 1.01 ms / 17.36 ms | 1.007 ms at 10K; 17.424 ms at 1M | -- | Five-run medians; no regression signal |
| GPU tests | Point-region and spatial-query suite | Local: 132 pass, 1 optional skip | -- | 133 pass, 1 optional skip | Documented strict-native invocation; no correctness failure |
| vibeProj | 1M 2D end-to-end | -- | 2.956 ms; 338.3 Mpts/s | 2.928 ms; 341.5 Mpts/s | Equivalent at one chunk |
| vibeProj | 1M 3D end-to-end | -- | 3.051 ms; 327.8 Mpts/s | 2.923 ms; 342.2 Mpts/s | H200 1.04x faster |
| vibeProj | 10M 2D end-to-end | -- | 28.2 ms; 355.1 Mpts/s | 54.1 ms; 184.7 Mpts/s | 4090 host pipeline 1.92x faster |
| vibeProj | 10M 3D end-to-end | -- | 27.7 ms; 360.8 Mpts/s | 41.9 ms; 238.4 Mpts/s | 4090 host pipeline 1.51x faster |
| vibeProj | 100M 2D end-to-end | -- | 269.2 ms; 371.4 Mpts/s | 512.5 ms; 195.1 Mpts/s | 4090 host pipeline 1.90x faster; target remains 128 ms |
| vibeProj | 100M 3D end-to-end | -- | 267.4 ms; 373.9 Mpts/s | 532.2 ms; 187.9 Mpts/s | 4090 host pipeline 1.99x faster |
| vibeProj | 1M kernel-only reference | -- | 0.520 ms; 1.92 Gpts/s | 0.056 ms; 17.97 Gpts/s | H200 kernel 9.3x faster; end-to-end is transfer dominated |
| vibeProj | 10M kernel-only reference | -- | 4.628 ms; 2.16 Gpts/s | 0.263 ms; 38.04 Gpts/s | H200 kernel 17.6x faster |
| vibeProj | 100M kernel-only reference | -- | 46.058 ms; 2.17 Gpts/s | 2.347 ms; 42.61 Gpts/s | H200 kernel 19.6x faster |
| vibeProj | Host-to-device bandwidth | -- | 23.6 GB/s | about 52.8 GB/s | Transfer-pipeline evidence |
| vibeProj | Device-to-host bandwidth | -- | 25.8 GB/s | about 51.0 GB/s | Synchronous `.get(out=)` path identified; async memcpy path verified |

## Interpretation And Evidence

The H200 wins compute-heavy SF100 Q8-Q11, while the 13900K/4090 wins many
orchestration-heavy Q1-Q7 workloads and 1M site. This supports conservative,
shape-aware planning and rejects product-name or 4090-only thresholds.

Nsight confirms it: H200 kernel time is 2.76x/2.87x faster for Q10/Q11 and
produces 1.74x/2.20x trace-wall speedups despite the shared host. Q8/Q12 kernels
are 2.65x/2.39x faster, but host/FUSE gaps erase that gain. Q11's prepared PIP
kernel alone falls from 197.06 to 56.35 seconds (3.50x) with the same 2,310
launches. Local has CPU data at paranoid level 2; the H200 host fixed that
setting at level 4, so its accepted partial capture omits CPU stacks. GPU metrics and allocation
tracing remain disabled on both devices.

Classification-once is safe across the protected corpus, but its Q11 gain
falls from 23.5% locally to 8.0% on H200: safe, but hardware-dependent.

Fresh local results are under `benchmark_results/cross_device/2026-08-19-rtx4090-current/`;
traces are under `benchmark_results/nsight/sf100/2026-08-19-rtx4090-comparable/`.
All trace checksums pass; 11 clean/trace results are byte-identical and Q6
differs by only 1.36e-20 absolute / 4.59e-16 relative. The 32-object H200
bundle, including `SHA256SUMS`, is at:

```text
s3://ec7ngh7mbj/vibespatial-validation/2026-08-19/results/h200-2026-08-19/
```

The 463-object H200 Nsight bundle is local at
`benchmark_results/nsight/sf100/2026-08-20-h200-comparable/` and mirrored at the documented S3 prefix.
Its 12/12 results match the 4090 oracle at `rtol=1e-6`; eight are byte-identical.

The old 10M run generated the entire fixture catalog and never reached VS. The
site-only path is fingerprint-equivalent at 10K/1M and exposes a real 4090
limit: 20.17/21.16 GiB pool plus a failed 2.58 GiB request. Rerun H200 with it.
