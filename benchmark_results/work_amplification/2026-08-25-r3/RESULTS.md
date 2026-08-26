# R3 Work-Amplification Results

Status: complete for the parent-aware point-region decision; constructive
counterfactuals archived; broad opportunity map recorded separately.

## Decision table

| Hypothesis | Frozen control | Current alternative | Decision |
|---|---:|---:|---|
| Q11 parent MultiPolygon refinement | 226.18s | 190.56s | graduate paired parent-aware component reduction |
| Q10 parent MultiPolygon refinement | 114.80s | 114.85s parent path | do not select component attribute reduction |
| vegetation union before equal-radius buffer | 0.247s control at 10K | 66.33s | archive; global line noding has the wrong shape |
| distribute vegetation intersection before union | current public control | did not complete in one minute | archive; fragment construction amplifies work |

## Parent-aware component reduction

The production path remains behind the existing public
`SpatialIndex.query_pair_aggregate` API. It admits only aligned point indexes,
`contains`/`contains_properly`, homogeneous non-indirected MultiPolygon query
rows, and a measured heavy-tail threshold. It then:

1. reinterprets immutable MultiPolygon parts as Polygon rows once;
2. caches that derived carrier and its prepared point-location directory on
   the original immutable query carrier;
3. classifies component candidates in fp64 without exporting pairs;
4. sorts stable `(point, parent, component)` keys;
5. preserves the first non-exterior component location, including invalid
   overlapping multipart order semantics; and
6. intersects the two parent-key streams before emitting aligned counts.

The complete final SF100 decision run is exact against the frozen R2 result:

| Query | R2 | R3 final | Change |
|---|---:|---:|---:|
| Q10 | 114.80s | 114.85s | +0.04% |
| Q11 | 226.18s | 190.56s | -15.75% |
| combined | 340.98s | 305.41s | -10.43% |

The final one-batch Q11 public path is 1.979s versus the frozen 2.244s parent
control, an 11.8% reduction. The earlier fixed-capacity component prototype was
1.924s and remains an upper-bound diagnostic, not a production claim.

The first uncached full-Q10 prototype took 228.71s. Caching the derived
component carrier reduced it to 109.22s, proving that repeated preparation was
a general amplification mechanism. The final tri-state attribute reducer,
however, measured 120.38s and lost against the 114.80s parent control. It was
removed rather than selected. Q10's final 114.85s confirms the protected path.

Randomized public-path tests cover valid, holed, overlapping-invalid, and
reversed-component MultiPolygons for both predicates. Results match the
Shapely oracle, including the order-sensitive invalid case that disproves
naive component-interior OR reduction. Capacity failure after native selection
propagates without retry.

## Constructive counterfactuals

Vegetation and habitat expose a reusable equal-radius line-buffer coverage
shape, but the obvious algebraic rewrites are wrong physically:

- unioning lines before buffering turns local stroke construction into one
  global line-noding problem and was about 268x slower at 10K;
- distributing intersection before terminal union constructs many fragments
  and did not finish the 10K falsifier in one minute; and
- output differences from buffer-order changes are small but topologically
  real because round-cap approximation and noding order do not commute.

No benchmark-specific rewrite graduates. The remaining high-value research
target is a dedicated exact equal-radius stroke-coverage union that constructs
the coverage boundary directly or reuses tiled stroke topology. That requires
its own GIS contract and is not a continuation of generic `union_all` tuning.

## Regression floor

The final-source 10K run reuses the validated static GeoPandas comparator and
passes 14/14 fingerprints. vibeSpatial totals 2.609s versus 3.524s for the
comparator. The mandatory full pipeline profile passes 22 workflows with two
explicit raster deferrals, zero compute D2H/materialization/fallback, and no
1M stage above one second. The final-source 1M candidate-only diagnostic
produces 14/14 fingerprints in 160.111s. Thirteen match the retained R2
fingerprints; corridor is the new native success and matches its dedicated
oracle artifact. It is not a cross-library speedup claim because no valid 1M
comparator was rerun.
