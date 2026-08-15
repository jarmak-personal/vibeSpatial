# ✅ SpatialBench Correctness (SF100)

Each engine's result was compared against the independent SQL-derived optimized-GeoPandas result on the same converted SF100 GeoParquet dataset at `rtol=1e-6`, `atol=1e-9`. A mismatch (❌), a missing result (🚫), or an unreadable result (⚠️) fails this check.

| Query | 🐼 GeoPandas optimized | ⚡ vibeSpatial |
|:------|:---:|:---:|
| **Q1** | ✅ | ✅ |
| **Q2** | ✅ | ✅ |
| **Q3** | ✅ | ✅ |
| **Q4** | ✅ | ✅ |
| **Q5** | ✅ | ✅ |
| **Q6** | ✅ | ✅ |
| **Q7** | ✅ | ✅ |
| **Q8** | ✅ | ✅ |
| **Q9** | ✅ | ✅ |
| **Q10** | ✅ | ✅ |
| **Q11** | ✅ | ✅ |
| **Q12** | ✅ | ✅ |

| Engine | ✅ Correct | ❌ Failed | Not verified |
|--------|:---------:|:---------:|:------------:|
| 🐼 GeoPandas optimized | 12 | 0 | 0 |
| ⚡ vibeSpatial | 12 | 0 | 0 |

| Legend | Meaning |
|--------|---------|
| ✅ | Matches the same-data optimized-GeoPandas oracle |
| ❌ | Does not match, or the result exists but could not be read/compared — fails CI |
| 🚫 | Reported success but produced no result (dump failed) — fails CI |
| ⏱️ | Engine could not compute the query in time (not a failure) |
| 💀 | Runner killed before this query ran, likely OOM (not a failure) |
| ⚠️ | Engine errored running the query (not a failure) |
| ❔ | Framework produced no result — its benchmark job failed; see the benchmark summary (not a failure) |
| — | No same-data oracle for this query |

*Generated on 2026-08-15 05:01:49 UTC*
