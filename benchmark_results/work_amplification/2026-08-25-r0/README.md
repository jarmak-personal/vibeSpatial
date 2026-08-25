# Work-amplification R0 evidence

Uninstrumented evidence captured from clean revision `e8e7f22` on
`picard-4090` (Intel i9-13900K, RTX 4090 24 GiB, driver 580.173.02, local
NVMe) on 2026-08-25.

- `shootout_10k_repeat3.json`: public 10K shootout, three repeats, warmup,
  refreshed static GeoPandas comparator, 14/14 exact.
- `shootout_1m_repeat1_vs_only.json`: strict-native public 1M diagnostic,
  one repeat, no warmup, VS execution only. The deliberately invalid
  `/bin/true` comparator makes the command exit non-zero after recording the
  VS results; do not interpret its comparator status as a workload failure.
- `pipelines_full_repeat1.json`: mandatory full pipeline profile, one repeat,
  strict native, GPU sparkline.
- `sf100_vibespatial_cold1.json`: strict-native SF100 cold-one run, all twelve
  queries. `results/` contains the corresponding public result CSVs.
- `SHA256SUMS`: checksums captured before this README was added.

The 1M vegetation post-timing profile is observer-invalid: 450.82 seconds
versus 53.81 seconds for the lean timed execution. Preserve both values; do
not use the profiled stage wall times as production attribution.
