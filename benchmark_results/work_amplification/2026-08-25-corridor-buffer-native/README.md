# Corridor Polygonal Buffer Native Follow-up

This capsule records the current-source correction for the historical 1M
`corridor_flood_priority` strict-native off-ramp. The general public
`GeoSeries.buffer` path now accepts device-resident Polygon/MultiPolygon
carriers, including null and empty rows, by expanding Polygon parts, running
the existing fp64 buffer implementation, and reducing parts back to their
original rows with grouped union. Native admission is conservative: finite
nonnegative radii, valid input, and hole-free Polygon parts. Other topology
domains take the existing observable fallback before constructive submission.

Source identity: `3d9dead7a7f220a76ee498ac5e0648d7613da70eb545c868b9e232dea240da0d`
on `picard-4090` (Intel i9-13900K, RTX 4090).

| Scale | Protocol | GeoPandas | vibeSpatial | Result |
|---|---|---:|---:|---|
| 10K | repeat 3, warmup, reused validated comparator | 0.162s | 0.270s median | exact fingerprint; homogeneous fast path retained |
| 1M | repeat 1, warmup, reused validated comparator, counter profile | 10.318s | 1.255s | 8.22x; exact fingerprint; zero fallbacks |

The isolated 1M counter profile completed in 1.381s. All 65 dispatch steps
selected GPU. The combined polygonal buffer, point join, copy, and filter
branch took 98.0ms; the preceding one-million-row line buffer took 2.7ms. The
only materializations in the terminal stage are explicit user exports.

Commands:

```bash
PYTHONPATH=/home/picard/repos/vibeSpatial/benchmarks/shootout \
  uv run vsbench shootout benchmarks/shootout/corridor_flood_priority.py \
  --scale 1M --repeat 1 --timeout 600 --profile-mode counters --json \
  --output corridor_1m_repeat1_counters.json

PYTHONPATH=/home/picard/repos/vibeSpatial/benchmarks/shootout \
  uv run vsbench shootout benchmarks/shootout/corridor_flood_priority.py \
  --scale 10K --repeat 3 \
  --reuse-geopandas ../2026-08-25-r2/shootout_10k_final.json \
  --timeout 300 --profile-mode off --json \
  --output corridor_10k_repeat3.json
```

The older R0/R1/R2 artifacts remain immutable historical evidence of the
off-ramp that existed at those source revisions.
