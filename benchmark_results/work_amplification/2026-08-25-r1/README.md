# Work-amplification R1 evidence

Profiling-only Level-0 counters and the bounded Q11 Level-1 profile captured on
`picard-4090` from the dirty current source rooted at revision `e8e7f22`.

- `shootout_10k_repeat3_counters.json`: 14/14 exact at the captured R1 source,
  validated static GeoPandas timing reused by measurement identity.
- `shootout_1m_repeat1_vs_only_counters.json`: strict-native diagnostic at the
  captured R1 source. The invalid comparator intentionally makes suite
  status non-passing; 13/14 vibeSpatial workflows complete and the corridor
  workflow retains its explicit off-ramp.
- `pipelines_full_repeat1.json`: final mandatory full profile after the
  nonphysicalizing audit fix; 22 successful, 2 deferred, zero compute D2H,
  materialization, or fallback.
- `sf100_vibespatial_lean_cold1.json`: 12/12 exact lean run, 465.39s.
- `sf100_vibespatial_counter_telemetry_cold1.json`: 12/12 counter-only run,
  464.55s.
- `sf100_q11_counter_point_region_cold1.json`: isolated schema-2 point-region
  attribution, 7.981B candidates and five prepared region groups.
- `transit_1m_counter_after_audit_fix.json`: targeted proof that counter replay
  observes the 30,553,577-row native composition without physicalizing it.
- `results/`: normalized public SF100 outputs for the lean and counter runs.

Large offline analyzer expansions are reproducible and intentionally ignored;
the retained raw artifacts are the source of truth. Run
`scripts/analyze_work_amplification.py` to regenerate them.
