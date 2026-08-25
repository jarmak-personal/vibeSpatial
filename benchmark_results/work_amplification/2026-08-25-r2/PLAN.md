# R2 Work-Amplification Counterfactuals

Status: complete experiment capsule

This capsule tests three independent physical-shape hypotheses against the
same public semantics and current vibeSpatial candidate:

1. Q11 paired classification-once: retain the validated historical forced A/B,
   then compare it with the current exact Q11 candidate and test whether a new
   component-first lowering can remove more refinement work.
2. Grouped constructive reduction: run the pre-rewrite paged public workflow
   and the current reduce-before-construct workflow against identical current
   1M fixtures.
3. Q12 regular distance filtering: compare current indexed nearest with the
   dense bbox-certificate plus exact-distance counterfactual retained in the
   earlier SF100 physical-shape capsule.

No experiment changes production dispatch. Every timed geometry operation is
expressed through public GeoPandas-compatible vibeSpatial APIs. Fixture
rewrites, host oracles, and result validation occur outside the timed region.

## Q11 gates

The immutable classification-once experiment is the forced baseline/alternative
control: two independent endpoint classifications versus exact reuse of the
first conservative superset. The current candidate must retain its exact result
and its 7.981B-candidate physical profile.

Component-first is a follow-up falsifier, not a production proposal.

The parent and component decision arms were rerun through
`q11_component_first_identified.py`; profile arms remain separate. The wrapper
preserves the measured implementation and timing boundary while recording the
exact source tree, dataset inventory, machine/environment, and protocol.

The parent control and component arm consume the same 4M-row trip batch and
the same five SF100 zone partitions. MultiPolygon decomposition preserves
Polygon holes and records stable parent-row lineage. For `contains`, exact
left/right point membership counts must be byte-identical. Component
`shared_count` is not parent-equivalent when aligned endpoints occupy different
components of one parent, so it is measured but not claimed as the terminal
answer.

The hypothesis advances only if component query wall and exact work fall by
enough to pay for a bounded parent reducer. Otherwise it is archived before a
production carrier or selector is designed.

## Grouped constructive gate

For source feature `A_i` and grouped zone rows `B_gj`, compare:

```text
union_i,j(A_i intersection B_gj)
union_i(A_i) intersection union_j(B_gj)
```

The reduced arm may use coverage union only with the existing structural
interior-disjointness certificate. Both arms must emit the same canonical
geometry fingerprint and remain strict-native and fallback-free. The historical
311.996-second result includes a separate provenance defect; the fair current
control is the same paged workflow after provenance continuity.

The retained paged control imports `_data` from `benchmarks/shootout`; its
isolated candidate process must therefore be launched with
`PYTHONPATH=/home/picard/repos/vibeSpatial/benchmarks/shootout`. The canonical
command is retained in `README.md`. The final current-source comparison is
64.095s paged versus 11.116s reduced, with identical fingerprints.

## Q12 regular-distance gate

Rerun the current public indexed-nearest candidate, then the prior dense bbox
lower-bound certificate with public exact distance refinement. Ordered keys
must match and distances must pass the SF100 `rtol=1e-6, atol=1e-9` contract.
The complete query wall, not dense pair throughput, decides the result.

Both arms carry contemporaneous identity for the exact git head and dirty
source tree, the SF100 GeoParquet manifest and file inventory, Python/lockfile
and GPU environment, and the cold single-run timing boundary. The indexed arm
was run first and the dense arm second through `q12_dense_experiment.py`.
